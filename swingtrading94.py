"""QuantAlgo Pro v6 — All confirmation filters + new SL/Target types"""
import streamlit as st, yfinance as yf, pandas as pd, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, date
from typing import Dict, Tuple, Optional
import warnings; warnings.filterwarnings("ignore")

st.set_page_config(page_title="QuantAlgo Pro", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.stApp{background:#0d1117}
.main .block-container{padding-top:.6rem;padding-bottom:.6rem}
[data-testid="metric-container"]{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:7px 12px}
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
.mc{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:8px;text-align:center}
.ml{font-size:.63rem;color:#8b949e;font-weight:700;text-transform:uppercase;letter-spacing:1px}
.mv{font-size:1rem;font-weight:900;color:#e6edf3}
.ms{font-size:.7rem;color:#8b949e}
.sbuy{background:rgba(63,185,80,.15);border:2px solid #3fb950;border-radius:8px;padding:9px;text-align:center;margin:3px 0}
.ssell{background:rgba(248,81,73,.15);border:2px solid #f85149;border-radius:8px;padding:9px;text-align:center;margin:3px 0}
.swait{background:rgba(139,148,158,.1);border:1px solid #30363d;border-radius:8px;padding:9px;text-align:center;margin:3px 0}
.pos-long{background:rgba(63,185,80,.1);border:1px solid #3fb950;border-radius:8px;padding:12px;margin:3px 0}
.pos-short{background:rgba(248,81,73,.1);border:1px solid #f85149;border-radius:8px;padding:12px;margin:3px 0}
.pos-none{background:rgba(30,35,44,.5);border:1px solid #30363d;border-radius:8px;padding:9px;text-align:center;color:#8b949e}
.auto-on{background:rgba(63,185,80,.12);border:2px solid #3fb950;border-radius:8px;padding:9px;text-align:center}
.warn-box{background:rgba(240,136,62,.1);border:1px solid #f0883e;border-radius:8px;padding:9px;font-size:.79rem}
.cf-active{background:rgba(88,166,255,.08);border-left:3px solid #58a6ff;border-radius:6px;padding:6px 10px;font-size:.78rem;margin:2px 0}
hr{border-color:#21262d !important}
</style>""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────
TICKERS: Dict[str,str] = {
    "🇮🇳 Nifty 50":"^NSEI","🏦 Bank Nifty":"^NSEBANK","📊 Sensex":"^BSESN",
    "₿ Bitcoin":"BTC-USD","Ξ Ethereum":"ETH-USD","💵 USD/INR":"USDINR=X",
    "🥇 Gold":"GC=F","🥈 Silver":"SI=F","✏️ Custom":"CUSTOM",
}
NO_VOLUME_SYMBOLS={"^NSEI","^NSEBANK","^BSESN","^DJI","^IXIC","^GSPC","^FTSE","^N225"}
TIMEFRAME_PERIODS: Dict[str,list]={
    "1m":["1d","5d","7d"],"5m":["1d","5d","7d","1mo"],
    "15m":["1d","5d","7d","1mo"],"1h":["1d","7d","1mo","3mo","6mo","1y"],
    "1d":["7d","1mo","6mo","1y","2y","3y","5y","10y"],
    "1wk":["1mo","3mo","6mo","1y","2y","5y","10y","20y","30y"],
}
STRATEGIES=["EMA Crossover","Simple Buy (Immediate)","Simple Sell (Immediate)",
    "Threshold Price Cross","Price Action (S/R)","Liquidity Zone / VWAP",
    "RSI Strategy","Bollinger Bands","Volume Breakout","Elliott Wave (Simplified)",
    "Volume Profile FR [VPFR]","★ Kalman Mean Reversion [PRO]",
    "★ Order Flow Imbalance [PRO]","★ Volatility Regime Momentum [PRO]"]
IMMEDIATE_ENTRY={"Simple Buy (Immediate)","Simple Sell (Immediate)"}
MANUAL_ONLY={"Simple Buy (Immediate)","Simple Sell (Immediate)"}
VOLUME_REQUIRED={"Volume Breakout","Volume Profile FR [VPFR]",
                  "★ Order Flow Imbalance [PRO]","Liquidity Zone / VWAP"}
SL_TYPES=["Custom Points","Trail SL","Trail – Current Candle Low/High",
    "Trail – Previous Candle Low/High","Trail – Current Swing High/Low",
    "Trail – Previous Swing High/Low","Strategy Signal Exit","EMA Reverse Crossover",
    "ATR Based SL","Risk Reward (min 1:2)","🤖 Autopilot SL",
    "Step Trail SL (N pts lock K pts)",            # NEW
    "Drawdown Recovery Exit (loss+recovery%)"]     # NEW
TARGET_TYPES=["Custom Points","Trail Target (display only – never exits)",
    "Trail – Current Candle Low/High","Trail – Previous Candle Low/High",
    "Trail – Current Swing High/Low","Trail – Previous Swing High/Low",
    "Strategy Signal Exit","EMA Reverse Crossover","ATR Based Target",
    "Risk Reward (min 1:2)","🤖 Autopilot Target",
    "Profit Erosion Exit (peak-erosion%)"]         # NEW

# ── SESSION STATE ─────────────────────────────────────────────
for _k,_v in {
    "backtest_results":None,"live_trades":[],"live_position":None,
    "live_running":False,"current_data":None,"signals":None,"indicators":{},
    "last_data_key":"","last_strat_key":"","last_signal_candle":None,
    "dhan_connected":False,"leaderboard_results":None,"lb_ran_for":"",
    "daily_stats":{"pnl":0.0,"trades":0,"date":None},"trade_paused_reason":"",
}.items():
    if _k not in st.session_state: st.session_state[_k]=_v

# ── INDICATORS ────────────────────────────────────────────────
def _ema(s,n): return s.ewm(span=n,adjust=False).mean()
def _sma(s,n): return s.rolling(n).mean()
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
    tp=(df["High"]+df["Low"]+df["Close"])/3
    return ((tp*df["Volume"]).cumsum()/df["Volume"].cumsum().replace(0,np.nan)
            if df["Volume"].sum()>0 else tp.ewm(span=20,adjust=False).mean())
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
def _adx(df,n=14):
    h,l,c=df["High"],df["Low"],df["Close"]
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    up=h-h.shift(); dn=l.shift()-l
    pdm=np.where((up>dn)&(up>0),up,0.); ndm=np.where((dn>up)&(dn>0),dn,0.)
    atr_s=tr.ewm(span=n,adjust=False).mean()
    pdi=100*pd.Series(pdm,index=df.index).ewm(span=n,adjust=False).mean()/atr_s.replace(0,np.nan)
    ndi=100*pd.Series(ndm,index=df.index).ewm(span=n,adjust=False).mean()/atr_s.replace(0,np.nan)
    dx=100*(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)
    return dx.ewm(span=n,adjust=False).mean()
def _supertrend(df,period=10,mult=3.0):
    a=_atr(df,period); hl2=(df["High"]+df["Low"])/2
    ub=(hl2+mult*a).copy(); lb=(hl2-mult*a).copy()
    trend=pd.Series(1,index=df.index); st_line=lb.copy()
    for i in range(1,len(df)):
        ub.iloc[i]=ub.iloc[i] if (ub.iloc[i]<ub.iloc[i-1] or df["Close"].iloc[i-1]>ub.iloc[i-1]) else ub.iloc[i-1]
        lb.iloc[i]=lb.iloc[i] if (lb.iloc[i]>lb.iloc[i-1] or df["Close"].iloc[i-1]<lb.iloc[i-1]) else lb.iloc[i-1]
        if st_line.iloc[i-1]==ub.iloc[i-1]:
            if df["Close"].iloc[i]>ub.iloc[i]: trend.iloc[i]=1; st_line.iloc[i]=lb.iloc[i]
            else: trend.iloc[i]=-1; st_line.iloc[i]=ub.iloc[i]
        else:
            if df["Close"].iloc[i]<lb.iloc[i]: trend.iloc[i]=-1; st_line.iloc[i]=ub.iloc[i]
            else: trend.iloc[i]=1; st_line.iloc[i]=lb.iloc[i]
    return st_line,trend
def _macd(s,fast=12,slow=26,sig=9):
    m=_ema(s,fast)-_ema(s,slow); signal=_ema(m,sig); return m,signal,m-signal
def _ema_angle(ef,lookback=3):
    """Angle of EMA slope — normalised arctan in degrees."""
    price_range=(ef.rolling(lookback*4).max()-ef.rolling(lookback*4).min()).replace(0,np.nan)
    dy=(ef-ef.shift(lookback))/lookback
    return np.degrees(np.arctan(dy/price_range*100)).fillna(0)
def _fvg(df,lookback=5):
    bull=df["Low"]>df["High"].shift(2); bear=df["High"]<df["Low"].shift(2)
    return bull.rolling(lookback).max().fillna(0).astype(bool), bear.rolling(lookback).max().fillna(0).astype(bool)
def _smc_bos(df,w=5):
    ph,pl=_pivots(df,w)
    sh=df["High"].where(ph).ffill(); sl_=df["Low"].where(pl).ffill()
    return df["Close"]>sh.shift(1), df["Close"]<sl_.shift(1)
def has_volume(df): return df["Volume"].sum()>0
def _vpfr_compute(df,lb=50,n_bins=30):
    n=len(df); poc_a=np.full(n,np.nan); vah_a=np.full(n,np.nan); val_a=np.full(n,np.nan)
    sigs=np.zeros(n,dtype=int)
    if not has_volume(df): return pd.Series(sigs,index=df.index),{"no_volume":True}
    for i in range(lb,n):
        w=df.iloc[i-lb:i]; lo_m=float(w["Low"].min()); hi_m=float(w["High"].max())
        if hi_m<=lo_m: continue
        edges=np.linspace(lo_m,hi_m,n_bins+1); mids=(edges[:-1]+edges[1:])/2
        overlap=(w["Low"].values[:,None]<edges[None,1:])&(w["High"].values[:,None]>edges[None,:-1])
        n_per=np.maximum(overlap.sum(axis=1,keepdims=True),1)
        vol_hist=(w["Volume"].values[:,None]*overlap/n_per).sum(axis=0)
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
        if not h.empty: return float(h["Close"].iloc[-1]),float(h["High"].max()),float(h["Low"].min())
    except: pass
    return 0.,0.,0.
def clear_all_cache(): fetch_data.clear(); fetch_ltp.clear()

# ── CONFIRMATION FILTERS ──────────────────────────────────────
def apply_confirmations(df, signals, no_vol, conf):
    """Apply all active confirmation filters vectorised. Masks out signals
    that don't meet active criteria. Long and short checked independently."""
    if signals.sum()==0: return signals
    if not any(conf.get(k) for k in conf if k.endswith("_enabled")): return signals
    f=signals.copy(); close=df["Close"]

    if conf.get("adx_enabled"):
        adx=_adx(df); ok=(adx>=conf["adx_min"])&(adx<=conf["adx_max"])
        f[(f!=0)&~ok]=0

    if conf.get("rsi_cf_enabled"):
        r=_rsi(close,14)
        long_ok =(r>=conf["rsi_long_min"])&(r<=conf["rsi_long_max"])
        short_ok=(r< conf["rsi_short_min"])|(r> conf["rsi_short_max"])
        f[(f== 1)&~long_ok]=0; f[(f==-1)&~short_ok]=0

    if conf.get("ema20_enabled"):
        e20=_ema(close,20)
        f[(f== 1)&(close<=e20)]=0; f[(f==-1)&(close>=e20)]=0

    if conf.get("sma20_enabled"):
        s20=_sma(close,20)
        f[(f== 1)&(close<=s20)]=0; f[(f==-1)&(close>=s20)]=0

    if conf.get("bb_cf_enabled"):
        up,mid,lo=_boll(close,20,2.)
        # Long: close below mid (oversold in band), Short: close above mid
        f[(f== 1)&(close>=mid)]=0; f[(f==-1)&(close<=mid)]=0

    if conf.get("st_enabled"):
        _,trend=_supertrend(df,conf.get("st_period",10),conf.get("st_mult",3.))
        f[(f== 1)&(trend!= 1)]=0; f[(f==-1)&(trend!=-1)]=0

    if conf.get("vol_cf_enabled") and not no_vol:
        avgv=df["Volume"].rolling(20).mean()
        ok=df["Volume"]>avgv*conf.get("vol_cf_mult",1.5)
        f[(f!=0)&~ok]=0

    if conf.get("macd_enabled"):
        m,sig_,_=_macd(close)
        f[(f== 1)&(m<=sig_)]=0; f[(f==-1)&(m>=sig_)]=0

    if conf.get("sr_cf_enabled"):
        sup,res=_sr(df,10,3)
        # Long must be near support, Short near resistance
        near_any_sup=pd.Series(False,index=df.index)
        for s in sup:
            near_any_sup|=close.between(s*0.995,s*1.005)
        near_any_res=pd.Series(False,index=df.index)
        for r in res:
            near_any_res|=close.between(r*0.995,r*1.005)
        f[(f== 1)&~near_any_sup]=0; f[(f==-1)&~near_any_res]=0

    if conf.get("fvg_enabled"):
        bull_fvg,bear_fvg=_fvg(df)
        f[(f== 1)&~bull_fvg]=0; f[(f==-1)&~bear_fvg]=0

    if conf.get("smc_enabled"):
        bos_bull,bos_bear=_smc_bos(df)
        f[(f== 1)&~bos_bull]=0; f[(f==-1)&~bos_bear]=0

    if conf.get("angle_enabled"):
        ef=_ema(close,conf.get("angle_fast",9)); es=_ema(close,conf.get("angle_slow",15))
        angle=_ema_angle(ef)  # angle of fast EMA
        min_deg=conf.get("angle_min",20)
        f[(f!=0)&(angle.abs()<min_deg)]=0

    if conf.get("candle_size_enabled"):
        candle_size=(close-df["Open"]).abs()
        if conf.get("candle_size_type")=="ATR Based":
            min_size=_atr(df)*conf.get("candle_size_atr_mult",.5)
        else:
            min_size=conf.get("candle_size_pts",10.)
        f[(f!=0)&(candle_size<min_size)]=0

    return f

# ── STRATEGY ENGINE ───────────────────────────────────────────
def run_strategy(df, strat, p, conf=None) -> Tuple[pd.Series, dict]:
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
            v=_vwap(df); a=_atr(df); ind={"vwap":v,"atr_line":a}
            dist=(df["Close"]-v)/v*100
            if no_vol:
                sig[(dist<-0.3)&(dist.shift()>=-0.3)]=1; sig[(dist>0.3)&(dist.shift()<=0.3)]=-1
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
                sig[df["High"]>df["High"].rolling(lb).max().shift()]=1
                sig[df["Low"]<df["Low"].rolling(lb).min().shift()]=-1
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
                body=df["Close"]-df["Open"]; mom=body.rolling(lb).sum()/(df["High"]-df["Low"]).rolling(lb).sum().replace(0,np.nan)
                v=_vwap(df); ind={"vwap":v}
                sig[(mom>thr)&(df["Close"]>v)&(mom.shift()<=thr)]=1
                sig[(mom<-thr)&(df["Close"]<v)&(mom.shift()>=-thr)]=-1
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
            vok=df["Volume"]>df["Volume"].rolling(20).mean() if not no_vol else pd.Series(True,index=df.index)
            sig[(regime>1.1)&(mf>0)&(ms>0)&(ef>es)&vok&(regime.shift()<=1.1)]=1
            sig[(regime>1.1)&(mf<0)&(ms<0)&(ef<es)&vok&(regime.shift()<=1.1)]=-1
    except Exception as e: pass
    # Apply confirmation filters
    if conf:
        sig=apply_confirmations(df,sig,no_vol,conf)
    return sig,ind

# ── PROXIMITY ─────────────────────────────────────────────────
def strategy_proximity(df,strat,sp,sigs):
    if df is None or len(df)<2: return "far","No data","—"
    last_sig=int(sigs.iloc[-1]) if sigs is not None and not sigs.empty else 0
    if last_sig== 1: return "signal","🟢 BUY SIGNAL","Auto-enters when START active"
    if last_sig==-1: return "signal","🔴 SELL SIGNAL","Auto-enters when START active"
    cur=float(df["Close"].iloc[-1])
    try:
        if strat=="EMA Crossover":
            ef=_ema(df["Close"],sp.get("ema_fast",9)); es=_ema(df["Close"],sp.get("ema_slow",15))
            ev,sv_=float(ef.iloc[-1]),float(es.iloc[-1]); gap=abs(ev-sv_); pct=gap/sv_*100
            conv="↗" if gap<abs(float(ef.iloc[-2])-float(es.iloc[-2])) else "↘"
            if pct<0.08: return "near","⚡ IMMINENT CROSSOVER",f"Gap {gap:.2f} ({pct:.3f}%) {conv}"
            if pct<0.25: return "approaching","⚠️ APPROACHING",f"Gap {gap:.2f} ({pct:.3f}%) {conv}"
            return "far",f"⏳ FAR ({'Bullish' if ev>sv_ else 'Bearish'})",f"Gap {gap:.2f} ({pct:.2f}%) {conv}"
        elif strat=="RSI Strategy":
            r=_rsi(df["Close"],sp.get("rsi_period",14)); rv=float(r.iloc[-1])
            ob,os_=sp.get("rsi_ob",70),sp.get("rsi_os",30); d_ob,d_os=abs(rv-ob),abs(rv-os_)
            z="→OB" if d_ob<d_os else "→OS"
            if min(d_ob,d_os)<2: return "near","⚡ RSI AT TRIGGER",f"RSI {rv:.1f} {z}"
            if min(d_ob,d_os)<6: return "approaching","⚠️ RSI BUILDING",f"RSI {rv:.1f} {z}"
            return "far","⏳ RSI NEUTRAL",f"RSI {rv:.1f} (OB:{ob} OS:{os_})"
        elif "Kalman" in strat:
            kf=_kalman(df["Close"]); dev=df["Close"]-kf
            z=float((dev/dev.rolling(30).std().replace(0,np.nan)).iloc[-1]); thr=sp.get("kf_thr",1.5)
            if abs(z)>=thr*.85: return "near","⚡ Z AT EDGE",f"Z={z:.3f} (±{thr})"
            if abs(z)>=thr*.65: return "approaching","⚠️ Z BUILDING",f"Z={z:.3f} (±{thr})"
            return "far","⏳ NEAR FAIR VALUE",f"Z={z:.3f}"
        elif "Volatility Regime" in strat:
            a=_atr(df,14); regime=float(a.iloc[-1]/a.rolling(20).mean().iloc[-1])
            if regime>1.1: return "approaching","⚠️ TRENDING REGIME",f"ATR ratio {regime:.2f}"
            return "far","⏳ RANGE REGIME",f"ATR ratio {regime:.2f} (need >1.1)"
    except: pass
    return "far","⏳ MONITORING",f"Watching for {strat}…"

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
    elif sl_type=="Step Trail SL (N pts lock K pts)":
        # Initial SL is entry - N + K (first step gives entry at breakeven+K)
        # The step tracking happens in backtest/fragment
        N=p.get("step_n",10.); K=p.get("step_k",5.)
        return (entry-N+K) if direction==1 else (entry+N-K)
    elif sl_type in("Drawdown Recovery Exit (loss+recovery%)","Strategy Signal Exit","EMA Reverse Crossover"):
        pts=p.get("sl_points",10.); return entry-pts if direction==1 else entry+pts
    return entry*(.99 if direction==1 else 1.01)

def calc_target(entry,sl,direction,tgt_type,p,atr_val=0.):
    risk=abs(entry-sl) if sl else (atr_val or entry*.01)
    if risk==0: risk=atr_val or entry*.01
    if tgt_type=="Custom Points":
        pts=p.get("target_points",20.); return entry+pts if direction==1 else entry-pts
    elif "display only" in tgt_type: return entry+risk*3 if direction==1 else entry-risk*3
    elif "Trail" in tgt_type: return entry+risk*2 if direction==1 else entry-risk*2
    elif tgt_type=="ATR Based Target":
        m=p.get("target_atr_mult",2.); return entry+m*atr_val if direction==1 else entry-m*atr_val
    elif tgt_type=="Risk Reward (min 1:2)":
        rr=max(p.get("rr_ratio",2.),2.); return entry+rr*risk if direction==1 else entry-rr*risk
    elif tgt_type=="🤖 Autopilot Target": return entry+risk*2.618 if direction==1 else entry-risk*2.618
    elif tgt_type in("EMA Reverse Crossover","Strategy Signal Exit","Profit Erosion Exit (peak-erosion%)"): return None
    return entry+risk*2 if direction==1 else entry-risk*2

def update_trail_sl(cur_sl,candle,direction,sl_type):
    if "Current Candle" in sl_type or sl_type=="Trail SL":
        return max(cur_sl,float(candle["Low"])) if direction==1 else min(cur_sl,float(candle["High"]))
    return cur_sl

def enter_position(direction,cfg,df=None):
    ltp,_,_=fetch_ltp(cfg["tsym"]); df_=df if df is not None else st.session_state.current_data
    entry=ltp if ltp>0 else float(df_["Close"].iloc[-1])
    idx=len(df_)-1; a_val=float(_atr(df_).iloc[-1])
    sl=calc_sl(df_,entry,direction,cfg["sl_type"],cfg["sl_p"],idx)
    tgt=calc_target(entry,sl,direction,cfg["tgt_type"],cfg["tgt_p"],a_val)
    order_id=None
    if cfg.get("dhan_on") and st.session_state.dhan_connected:
        b=DhanBroker(st.session_state.get("d_cid",""),st.session_state.get("d_tok",""))
        b.connect(); order=b.place_order(cfg["tsym"],cfg["qty"],"BUY" if direction==1 else "SELL",entry,sl,tgt or 0)
        order_id=order.get("order_id")
    # For Step Trail: track step state
    N=cfg.get("sl_p",{}).get("step_n",10.); K=cfg.get("sl_p",{}).get("step_k",5.)
    next_step=entry+(N if direction==1 else -N)
    return dict(entry_price=entry,direction=direction,sl=sl,target=tgt,trail_sl=sl,
                qty=cfg["qty"],entry_dt=datetime.now(),order_id=order_id,
                step_n=N,step_k=K,next_step=next_step,
                mae=0.,mfe=0.)  # max adverse/favorable excursion

def exit_position(exit_price,exit_reason,cfg):
    pos=st.session_state.live_position
    if pos is None: return  # already closed (race condition guard)
    if not isinstance(pos,dict): return
    d=pos["direction"]; pnl_pts=(exit_price-pos["entry_price"])*d
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
    # Update daily stats
    ds=st.session_state.daily_stats; ds["pnl"]+=pnl_pts; ds["trades"]+=1
    st.session_state.live_position=None; st.session_state.last_signal_candle=None
    clear_all_cache()

# ── BACKTEST ENGINE ───────────────────────────────────────────
def run_backtest(df,signals,sl_type,sl_p,tgt_type,tgt_p,qty,strat_name="",conf=None):
    """Full backtest with new SL/Target types and daily limit simulation."""
    immediate=strat_name in IMMEDIATE_ENTRY
    trades=[]; in_trade=False; trade={}; a_s=_atr(df)
    daily_pnl=0.; daily_count=0; last_date=None
    # daily limits from conf
    conf=conf or {}
    dl_loss=conf.get("daily_loss_pts",9999) if conf.get("daily_loss_enabled") else 9999
    dl_profit=conf.get("daily_profit_pts",9999) if conf.get("daily_profit_enabled") else 9999
    dl_trades=conf.get("daily_trades_max",9999) if conf.get("daily_trades_enabled") else 9999
    max_dur=conf.get("max_duration_min",9999) if conf.get("max_duration_enabled") else 9999

    for i in range(1,len(df)):
        # Reset daily counters on new day
        cur_date=df.index[i].date() if hasattr(df.index[i],"date") else None
        if cur_date and cur_date!=last_date: daily_pnl=0.; daily_count=0; last_date=cur_date

        if not in_trade:
            # Skip if daily limits hit
            if daily_pnl<=-dl_loss or daily_pnl>=dl_profit or daily_count>=dl_trades: continue
            sv=int(signals.iloc[i-1]) if not pd.isna(signals.iloc[i-1]) else 0
            if sv!=0:
                entry=float(df["Close"].iloc[i-1]) if immediate else float(df["Open"].iloc[i])
                d=sv; a_val=float(a_s.iloc[min(i,len(a_s)-1)])
                sl=calc_sl(df,entry,d,sl_type,sl_p,i)
                tgt=calc_target(entry,sl,d,tgt_type,tgt_p,a_val)
                N=sl_p.get("step_n",10.); K=sl_p.get("step_k",5.)
                trade=dict(entry_idx=i,entry_dt=df.index[i-1 if immediate else i],
                           entry_price=entry,direction=d,initial_sl=sl,trail_sl=sl,
                           target=tgt,qty=qty,step_n=N,step_k=K,
                           next_step=entry+(N*d),mae=0.,mfe=0.)
                in_trade=True
        else:
            c=df.iloc[i]; d=trade["direction"]; sl_now=trade["trail_sl"]; tgt_now=trade["target"]
            ep=None; er=""
            cur_pnl=(float(c["Close"])-trade["entry_price"])*d
            trade["mae"]=min(trade["mae"],cur_pnl); trade["mfe"]=max(trade["mfe"],cur_pnl)

            # ── Max duration (losing) ──────────────────────────
            if max_dur<9999 and cur_pnl<0:
                if hasattr(trade["entry_dt"],"timestamp") and hasattr(df.index[i],"timestamp"):
                    dur_min=(df.index[i]-trade["entry_dt"]).total_seconds()/60
                    if dur_min>=max_dur: ep=float(c["Close"]); er=f"Max Duration ({max_dur}m)"

            # ── Step Trail SL update ───────────────────────────
            if ep is None and sl_type=="Step Trail SL (N pts lock K pts)":
                high_=float(c["High"]); low_=float(c["Low"])
                if d==1 and high_>=trade["next_step"]:
                    new_sl=trade["next_step"]-trade["step_k"]
                    trade["trail_sl"]=max(trade["trail_sl"],new_sl)
                    trade["next_step"]+=trade["step_n"]
                elif d==-1 and low_<=trade["next_step"]:
                    new_sl=trade["next_step"]+trade["step_k"]
                    trade["trail_sl"]=min(trade["trail_sl"],new_sl)
                    trade["next_step"]-=trade["step_n"]

            # ── Drawdown Recovery Exit ─────────────────────────
            if ep is None and sl_type=="Drawdown Recovery Exit (loss+recovery%)":
                exit_loss=sl_p.get("sl_points",10.); rec_pct=sl_p.get("recovery_pct",40.)
                if trade["mae"]<=-exit_loss:
                    req_recovery=exit_loss*rec_pct/100
                    if cur_pnl<trade["mae"]+req_recovery: ep=float(c["Close"]); er="Drawdown No-Recovery"

            # ── Profit Erosion Exit ────────────────────────────
            if ep is None and tgt_type=="Profit Erosion Exit (peak-erosion%)":
                peak_pts=tgt_p.get("peak_profit_pts",30.); ero_pct=tgt_p.get("erosion_pct",40.)
                if trade["mfe"]>=peak_pts:
                    allowed_erosion=trade["mfe"]*ero_pct/100
                    if cur_pnl<trade["mfe"]-allowed_erosion: ep=float(c["Close"]); er="Profit Erosion Exit"

            # ── Standard SL / Target ──────────────────────────
            if ep is None:
                sl_now=trade["trail_sl"]
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
                daily_pnl+=pp; daily_count+=1
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
_CMAP={"ema_fast":"#f0883e","ema_slow":"#58a6ff","bb_upper":"#555","bb_mid":"#777","bb_lower":"#555","kalman_price":"#ce93d8","vwap":"#ffe082","atr_line":"#80cbc4","vpfr_poc":"#ff9800","vpfr_vah":"#ef5350","vpfr_val":"#26a69a"}
def build_chart(df,sig,ind,trades_df=None,title=""):
    has_rsi="rsi" in ind; rows=3 if has_rsi else 2; rh=[.60,.20,.20] if has_rsi else [.72,.28]
    fig=make_subplots(rows=rows,cols=1,shared_xaxes=True,vertical_spacing=.025,row_heights=rh)
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],name="Price",increasing=dict(fillcolor="#26a69a",line=dict(color="#26a69a",width=1)),decreasing=dict(fillcolor="#ef5350",line=dict(color="#ef5350",width=1))),row=1,col=1)
    for k,v in ind.items():
        if not isinstance(v,pd.Series): continue
        if any(x in k for x in("ema","kalman","bb","vwap","atr_line","vpfr")):
            fig.add_trace(go.Scatter(x=df.index,y=v,name=k.replace("_"," ").title(),line=dict(color=_CMAP.get(k,"#aaa"),width=1 if "vpfr" in k else 1.5,dash="dash" if k in("bb_upper","bb_lower","vpfr_vah","vpfr_val") else "solid"),opacity=.85),row=1,col=1)
        elif k=="rsi":
            fig.add_trace(go.Scatter(x=df.index,y=v,name="RSI",line=dict(color="#ab47bc",width=1.5)),row=2,col=1)
            for lv,lc in[(70,"#ef5350"),(30,"#26a69a"),(50,"#444")]: fig.add_hline(y=lv,line=dict(color=lc,dash="dash",width=1),row=2,col=1)
    for s in ind.get("support_levels",[]): fig.add_hline(y=s,line=dict(color="#26a69a",dash="dot",width=1),opacity=.3,row=1,col=1)
    for r in ind.get("resistance_levels",[]): fig.add_hline(y=r,line=dict(color="#ef5350",dash="dot",width=1),opacity=.3,row=1,col=1)
    buys=sig[sig==1].index; sells=sig[sig==-1].index
    if len(buys): fig.add_trace(go.Scatter(x=buys,y=df["Low"].reindex(buys)*.999,mode="markers",name="Buy",marker=dict(symbol="triangle-up",size=10,color="#3fb950",line=dict(color="white",width=1))),row=1,col=1)
    if len(sells): fig.add_trace(go.Scatter(x=sells,y=df["High"].reindex(sells)*1.001,mode="markers",name="Sell",marker=dict(symbol="triangle-down",size=10,color="#f85149",line=dict(color="white",width=1))),row=1,col=1)
    if trades_df is not None and not trades_df.empty:
        for _,t in trades_df.iterrows():
            c_="#3fb950" if t["Result"]=="WIN" else "#f85149"
            fig.add_trace(go.Scatter(x=[t["Entry Date"]],y=[t["Entry Price"]],mode="markers+text",text=["E"],textposition="top center",textfont=dict(size=7,color=c_),marker=dict(size=7,color=c_),showlegend=False),row=1,col=1)
            try: fig.add_shape(type="line",x0=t["Entry Date"],x1=t["Exit Date"],y0=float(t["Initial SL"]),y1=float(t["Initial SL"]),line=dict(color="#f85149",dash="dot",width=1),row=1,col=1)
            except: pass
    vrow=3 if has_rsi else 2
    vcol=["#26a69a" if c>=o else "#ef5350" for c,o in zip(df["Close"],df["Open"])]
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],marker_color=vcol,opacity=.6,name="Volume"),row=vrow,col=1)
    avgv=ind.get("avg_volume")
    if avgv is not None and isinstance(avgv,pd.Series): fig.add_trace(go.Scatter(x=df.index,y=avgv,name="Avg Vol",line=dict(color="#f0883e",width=1,dash="dash"),opacity=.6),row=vrow,col=1)
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",font=dict(family="monospace",size=11,color="#8b949e"),showlegend=True,legend=dict(bgcolor="rgba(22,27,34,.85)",bordercolor="#30363d",borderwidth=1,font=dict(size=10)),xaxis_rangeslider_visible=False,margin=dict(l=0,r=0,t=34,b=0),height=580,title=dict(text=f"<b>{title}</b>",font=dict(size=12,color="#e6edf3"),x=.5))
    fig.update_xaxes(gridcolor="#21262d",zeroline=False,showspikes=True,spikecolor="#58a6ff",spikethickness=1)
    fig.update_yaxes(gridcolor="#21262d",zeroline=False,showspikes=True)
    return fig
def eq_fig(bt):
    cum=bt["P&L (₹)"].cumsum(); pos=cum.iloc[-1]>=0
    fig=go.Figure(go.Scatter(x=list(range(len(cum))),y=cum.values,fill="tozeroy",fillcolor="rgba(63,185,80,.15)" if pos else "rgba(248,81,73,.15)",line=dict(color="#3fb950" if pos else "#f85149",width=2)))
    fig.add_hline(y=0,line=dict(color="#555",dash="dash",width=1))
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",title="Equity Curve",height=230,margin=dict(l=0,r=0,t=26,b=0),font=dict(color="#8b949e"))
    return fig

# ── AUTO-TRADE FRAGMENT  (1-second loop, runs ONLY when START active) ─────────
@st.fragment(run_every=1)
def auto_trade_fragment(cfg,conf):
    ticker=cfg["tsym"]; strat=cfg["strat"]; sp=cfg["sp"]
    # Reset daily stats on new day
    today=date.today(); ds=st.session_state.daily_stats
    if ds.get("date")!=today: st.session_state.daily_stats={"pnl":0.,"trades":0,"date":today}; ds=st.session_state.daily_stats

    # Check daily limits — pause if hit
    paused=""
    if conf.get("daily_loss_enabled") and ds["pnl"]<=-conf.get("daily_loss_pts",100): paused=f"Daily loss limit hit ({ds['pnl']:.0f} pts)"
    if conf.get("daily_profit_enabled") and ds["pnl"]>=conf.get("daily_profit_pts",500): paused=f"Daily profit target hit ({ds['pnl']:.0f} pts)"
    if conf.get("daily_trades_enabled") and ds["trades"]>=conf.get("daily_trades_max",100): paused=f"Daily trade limit hit ({ds['trades']} trades)"
    if paused:
        st.session_state.live_running=False
        st.warning(f"🛑 Auto-trading PAUSED — {paused}"); return

    df=fetch_data(ticker,cfg["interval"],cfg["period"]); ltp,day_hi,day_lo=fetch_ltp(ticker)
    if df is None or df.empty: st.caption("⚠️ Data unavailable"); return
    sigs,inds=run_strategy(df,strat,sp,conf)
    last_sig=int(sigs.iloc[-1]) if not sigs.empty else 0; cur_candle=df.index[-1]
    no_vol=not has_volume(df)

    if no_vol and strat in VOLUME_REQUIRED:
        st.markdown(f"<div class='warn-box'>⚠️ <b>{strat}</b>: no volume — running price-only fallback.</div>",unsafe_allow_html=True)

    _sig_clr = "#3fb950" if last_sig==1 else "#f85149" if last_sig==-1 else "#8b949e"
    _sig_txt = "▲BUY" if last_sig==1 else "▼SELL" if last_sig==-1 else "NONE"
    _pnl_clr2 = "#3fb950" if ds["pnl"]>=0 else "#f85149"
    st.markdown(f"""<div style='display:flex;gap:12px;align-items:center;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:7px 12px;margin-bottom:6px;font-size:.81rem'>
      <span style='color:#3fb950;font-weight:700'>● LIVE AUTO-TRADE</span>
      <span style='color:#8b949e'>|</span>
      <span style='color:#e6edf3'>LTP: <b style='color:#58a6ff'>{ltp:,.2f}</b></span>
      <span style='color:#8b949e'>Signal: <b style='color:{_sig_clr}'>{_sig_txt}</b></span>
      <span style='color:#8b949e'>Daily PnL: <b style='color:{_pnl_clr2}'>{ds["pnl"]:+.1f}pts</b> | Trades: {ds["trades"]}</span>
      <span style='color:#8b949e;margin-left:auto;font-size:.68rem'>{datetime.now().strftime("%H:%M:%S")}</span>
    </div>""", unsafe_allow_html=True)

    pos=st.session_state.live_position
    if pos is None and strat not in MANUAL_ONLY:
        if last_sig!=0 and cur_candle!=st.session_state.last_signal_candle:
            pos_new=enter_position(last_sig,cfg,df)
            st.session_state.live_position=pos_new; st.session_state.last_signal_candle=cur_candle
            st.success(f"🤖 AUTO-ENTERED {'🟢LONG' if last_sig==1 else '🔴SHORT'} @ {pos_new['entry_price']:.2f} | SL {pos_new['sl']:.2f}")
            pos=st.session_state.live_position

    # Auto-exit checks
    if pos is not None and ltp>0:
        d=pos["direction"]; cur_pnl=(ltp-pos["entry_price"])*d; exited=False
        pos["mae"]=min(pos.get("mae",0.),cur_pnl); pos["mfe"]=max(pos.get("mfe",0.),cur_pnl)

        # Max duration losing trade
        if not exited and conf.get("max_duration_enabled") and cur_pnl<0:
            dur_min=(datetime.now()-pos["entry_dt"]).total_seconds()/60
            if dur_min>=conf.get("max_duration_min",5):
                exit_position(ltp,f"Max Duration ({conf['max_duration_min']}m)",cfg); st.warning(f"⏱️ Exited losing trade after {dur_min:.1f}m"); exited=True; st.rerun()

        # Step Trail SL update
        if not exited and cfg["sl_type"]=="Step Trail SL (N pts lock K pts)":
            if d==1 and ltp>=pos.get("next_step",float("inf")):
                new_sl=pos["next_step"]-pos["step_k"]; pos["trail_sl"]=max(pos.get("trail_sl",0),new_sl); pos["next_step"]+=pos["step_n"]
                st.info(f"📈 Step Trail: SL moved to {pos['trail_sl']:.2f}, next step @ {pos['next_step']:.2f}")
            elif d==-1 and ltp<=pos.get("next_step",float("-inf")):
                new_sl=pos["next_step"]+pos["step_k"]; pos["trail_sl"]=min(pos.get("trail_sl",float("inf")),new_sl); pos["next_step"]-=pos["step_n"]
                st.info(f"📉 Step Trail: SL moved to {pos['trail_sl']:.2f}")

        # Drawdown Recovery
        if not exited and cfg["sl_type"]=="Drawdown Recovery Exit (loss+recovery%)":
            exit_loss=cfg["sl_p"].get("sl_points",10.); rec_pct=cfg["sl_p"].get("recovery_pct",40.)
            if pos["mae"]<=-exit_loss and cur_pnl<pos["mae"]+exit_loss*rec_pct/100:
                exit_position(ltp,"Drawdown No-Recovery",cfg); st.error(f"🛑 Drawdown exit @ {ltp:.2f}"); exited=True; st.rerun()

        # Profit Erosion Target
        if not exited and cfg["tgt_type"]=="Profit Erosion Exit (peak-erosion%)":
            peak_pts=cfg["tgt_p"].get("peak_profit_pts",30.); ero_pct=cfg["tgt_p"].get("erosion_pct",40.)
            if pos["mfe"]>=peak_pts and cur_pnl<pos["mfe"]-pos["mfe"]*ero_pct/100:
                exit_position(ltp,"Profit Erosion Exit",cfg); st.success(f"🎯 Profit erosion exit @ {ltp:.2f}"); exited=True; st.rerun()

        # Standard SL / Target
        if not exited:
            sl_eff=pos.get("trail_sl",pos["sl"])
            if (d==1 and ltp<=sl_eff) or (d==-1 and ltp>=sl_eff):
                exit_position(sl_eff,"Stop Loss (Auto)",cfg); st.error(f"🛑 SL HIT @ {sl_eff:.2f}"); exited=True; st.rerun()
            if not exited and pos.get("target") and "display only" not in cfg.get("tgt_type",""):
                if (d==1 and ltp>=pos["target"]) or (d==-1 and ltp<=pos["target"]):
                    exit_position(pos["target"],"Target Hit (Auto)",cfg); st.success(f"🎯 TARGET HIT @ {pos['target']:.2f}"); exited=True; st.rerun()
        if not exited and pos is not None: st.session_state.live_position=pos

    pos=st.session_state.live_position
    if pos:
        d=pos["direction"]; pnl_pts=(ltp-pos["entry_price"])*d if ltp>0 else 0; pnl_val=pnl_pts*pos["qty"]
        pnl_clr="#3fb950" if pnl_pts>=0 else "#f85149"; duration=str(datetime.now()-pos["entry_dt"]).split(".")[0]
        sl_eff=pos.get("trail_sl",pos["sl"]); next_step_txt=f"Next step @ {pos.get('next_step',0):.2f}" if cfg["sl_type"]=="Step Trail SL (N pts lock K pts)" else ""
        st.markdown(f"""<div class='{"pos-long" if d==1 else "pos-short"}'>
          <div style='display:flex;flex-wrap:wrap;gap:14px;align-items:center'>
            <div><div class='ml'>DIR</div><div style='color:{"#3fb950" if d==1 else "#f85149"};font-weight:900;font-size:1.1rem'>{"▲LONG" if d==1 else "▼SHORT"}</div></div>
            <div><div class='ml'>ENTRY</div><div class='mv'>{pos['entry_price']:,.2f}</div></div>
            <div><div class='ml'>LTP</div><div class='mv' style='color:{pnl_clr}'>{ltp:,.2f}</div></div>
            <div><div class='ml'>SL (eff.)</div><div class='mv' style='color:#f85149'>{sl_eff:,.2f}</div></div>
            <div><div class='ml'>TARGET</div><div class='mv' style='color:#3fb950'>{"%.2f"%pos["target"] if pos.get("target") else "—"}</div></div>
            <div><div class='ml'>OPEN P&L</div><div style='color:{pnl_clr};font-weight:900;font-size:1.15rem'>{pnl_pts:+.2f}pts<br><span style='font-size:.88rem'>₹{pnl_val:+,.2f}</span></div></div>
            <div><div class='ml'>MAE/MFE</div><div class='ms' style='color:#8b949e'>{pos["mae"]:+.1f}/{pos["mfe"]:+.1f}</div></div>
            <div><div class='ml'>DUR</div><div class='ms'>{duration}</div></div>
            {f'<div><div class="ml">STEP TRAIL</div><div class="ms" style="color:#58a6ff">{next_step_txt}</div></div>' if next_step_txt else ""}
          </div></div>""",unsafe_allow_html=True)
    else:
        st.markdown("<div class='pos-none'>No open position — waiting for signal</div>",unsafe_allow_html=True)

    # Indicator values row
    val_items=[]
    if "ema_fast" in inds: val_items.append((f"EMA{sp.get('ema_fast',9)}",float(inds["ema_fast"].iloc[-1]),"#f0883e"))
    if "ema_slow" in inds: val_items.append((f"EMA{sp.get('ema_slow',15)}",float(inds["ema_slow"].iloc[-1]),"#58a6ff"))
    if "rsi" in inds: val_items.append(("RSI",float(inds["rsi"].iloc[-1]),"#ab47bc"))
    if "kalman_price" in inds: val_items.append(("Kalman",float(inds["kalman_price"].iloc[-1]),"#ce93d8"))
    if "vwap" in inds: val_items.append(("VWAP",float(inds["vwap"].iloc[-1]),"#ffe082"))
    if "bb_upper" in inds: val_items+=[("BB U",float(inds["bb_upper"].iloc[-1]),"#666"),("BB L",float(inds["bb_lower"].iloc[-1]),"#666")]
    val_items.append(("ATR",float(_atr(df).iloc[-1]),"#80cbc4")); val_items.append(("Close",float(df["Close"].iloc[-1]),"#e6edf3"))
    if val_items:
        ncols=min(len(val_items),8)
        for ri in range((len(val_items)+ncols-1)//ncols):
            chunk=val_items[ri*ncols:(ri+1)*ncols]
            for col,(nm,val,clr) in zip(st.columns(len(chunk)),chunk):
                with col: st.markdown(f"<div class='mc'><div class='ml'>{nm}</div><div class='mv' style='color:{clr};font-size:.85rem'>{val:,.2f}</div></div>",unsafe_allow_html=True)
        st.markdown("<div style='margin:4px'></div>",unsafe_allow_html=True)

    if "ema_fast" in inds and "ema_slow" in inds and len(inds["ema_fast"])>=2:
        ef_v,es_v=float(inds["ema_fast"].iloc[-1]),float(inds["ema_slow"].iloc[-1])
        ef_p,es_p=float(inds["ema_fast"].iloc[-2]),float(inds["ema_slow"].iloc[-2])
        if ef_v>es_v and ef_p<=es_p: st.markdown("<div class='sbuy' style='padding:6px'><b style='color:#3fb950'>▲ BULLISH EMA CROSSOVER on last candle!</b></div>",unsafe_allow_html=True)
        elif ef_v<es_v and ef_p>=es_p: st.markdown("<div class='ssell' style='padding:6px'><b style='color:#f85149'>▼ BEARISH EMA CROSSOVER on last candle!</b></div>",unsafe_allow_html=True)
        else:
            bias="Bullish" if ef_v>es_v else "Bearish"; gap=abs(ef_v-es_v); bclr="#3fb950" if bias=="Bullish" else "#f85149"
            st.markdown(f"<div class='swait' style='padding:5px;font-size:.8rem'><span style='color:{bclr}'>No crossover · {bias} bias</span> · Gap: <b>{gap:.2f}</b></div>",unsafe_allow_html=True)

    fig=build_chart(df,sigs,inds,title=f"LIVE {ticker} | {strat} | {cfg['interval']}/{cfg['period']}")
    for k,nm_,clr in[("ema_fast",f"EMA{sp.get('ema_fast',9)}","#f0883e"),("ema_slow",f"EMA{sp.get('ema_slow',15)}","#58a6ff"),("kalman_price","Kalman","#ce93d8"),("vwap","VWAP","#ffe082")]:
        if k in inds and isinstance(inds[k],pd.Series):
            v=float(inds[k].iloc[-1])
            fig.add_annotation(x=df.index[-1],y=v,text=f"  {nm_}: {v:,.2f}",showarrow=False,xanchor="left",font=dict(size=9,color=clr),bgcolor="rgba(13,17,23,.75)")
    st.plotly_chart(fig,use_container_width=True,key=f"atf_{ticker}_{cfg['interval']}")

# ── SIDEBAR ───────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("""<div style='text-align:center;padding:7px 0 12px'>
          <div style='font-size:1.6rem'>📊</div>
          <div style='font-size:1rem;font-weight:900;color:#e6edf3;letter-spacing:2px'>QuantAlgo Pro</div>
          <div style='font-size:.57rem;color:#8b949e;letter-spacing:2px'>v6.0 · AUTO-LOADS · ALL FILTERS</div>
        </div>""",unsafe_allow_html=True); st.divider()
        st.markdown("**🎯 INSTRUMENT**")
        tname=st.selectbox("Asset",list(TICKERS.keys()),label_visibility="collapsed",key="tname")
        tsym=(st.text_input("Symbol","RELIANCE.NS",key="csym").upper().strip() if tname=="✏️ Custom" else TICKERS[tname])
        st.markdown("**⏱️ TIMEFRAME**")
        c1,c2=st.columns(2); ivs=list(TIMEFRAME_PERIODS.keys())
        with c1: interval=st.selectbox("IV",ivs,index=ivs.index("1m"),key="interval",label_visibility="collapsed")
        plist=TIMEFRAME_PERIODS[interval]; def_p="5d" if "5d" in plist else plist[0]
        with c2: period=st.selectbox("P",plist,index=plist.index(def_p),key="period",label_visibility="collapsed")
        st.markdown("**📦 QTY**")
        qty=st.number_input("Q",min_value=1,value=1,step=1,key="qty",label_visibility="collapsed"); st.divider()
        st.markdown("**🧠 STRATEGY**")
        strat=st.selectbox("S",STRATEGIES,key="strat",label_visibility="collapsed"); sp:dict={}
        with st.expander("⚙️ Strategy Params"):
            if strat=="EMA Crossover": sp["ema_fast"]=st.number_input("Fast EMA",1,200,9,key="ef"); sp["ema_slow"]=st.number_input("Slow EMA",1,500,15,key="es")
            elif strat in IMMEDIATE_ENTRY: st.caption("Enters at LTP instantly.")
            elif strat=="Threshold Price Cross": sp["buy_t"]=st.number_input("Buy ≥",0.,key="bt_"); sp["sell_t"]=st.number_input("Sell ≤",0.,key="st__")
            elif strat=="Price Action (S/R)": sp["sr_window"]=st.number_input("Pivot W",3,50,10,key="srw")
            elif strat=="RSI Strategy": sp["rsi_period"]=st.number_input("Period",2,50,14,key="rp"); sp["rsi_ob"]=st.number_input("OB",50,99,70,key="rob"); sp["rsi_os"]=st.number_input("OS",1,50,30,key="ros")
            elif strat=="Bollinger Bands": sp["bb_period"]=st.number_input("Period",5,100,20,key="bbp"); sp["bb_std"]=st.number_input("StdDev",.5,5.,2.,step=.1,key="bbs")
            elif strat=="Volume Breakout": sp["vol_lb"]=st.number_input("Lookback",5,100,20,key="vlb"); sp["vol_mult"]=st.number_input("Vol×",1.,5.,1.5,step=.1,key="vm")
            elif strat=="Elliott Wave (Simplified)": sp["ew_window"]=st.number_input("Pivot W",3,30,10,key="eww")
            elif strat=="Volume Profile FR [VPFR]": sp["vpfr_lb"]=st.number_input("Lookback",10,200,50,key="vplb"); sp["vpfr_bins"]=st.number_input("Bins",10,60,30,key="vpb")
            elif "Kalman" in strat: sp["kf_thr"]=st.number_input("Z Thr",.5,4.,1.5,step=.1,key="kft")
            elif "Order Flow" in strat: sp["ofi_lb"]=st.number_input("Lookback",3,50,10,key="ofil"); sp["ofi_thr"]=st.number_input("Imbalance",.5,.95,.60,step=.01,key="ofit")
            elif "Volatility Regime" in strat: sp["vrm_fast"]=st.number_input("Fast",3,50,10,key="vrmf"); sp["vrm_slow"]=st.number_input("Slow",10,100,30,key="vrms")
        st.divider()
        st.markdown("**🛡️ STOP LOSS**")
        sl_type=st.selectbox("SL",SL_TYPES,key="sl_type",label_visibility="collapsed"); sl_p:dict={}
        with st.expander("⚙️ SL Params"):
            if sl_type=="Custom Points": sl_p["sl_points"]=st.number_input("Points",.1,1e5,10.,step=.5,key="slp")
            elif sl_type=="ATR Based SL": sl_p["sl_atr_mult"]=st.number_input("ATR×",.5,5.,1.5,step=.1,key="slam")
            elif sl_type=="Risk Reward (min 1:2)": sl_p["sl_points"]=st.number_input("Risk pts",.1,1e5,10.,step=.5,key="rrsl")
            elif sl_type=="🤖 Autopilot SL": sl_p["vol_scale"]=st.slider("Vol Scale",.5,2.,1.,key="vs")
            elif sl_type=="Step Trail SL (N pts lock K pts)":
                st.caption("When price moves N pts in favor → lock SL at N−K. Repeats every N pts.")
                sl_p["step_n"]=st.number_input("N (step size pts)",.5,500.,10.,step=.5,key="sn")
                sl_p["step_k"]=st.number_input("K (SL offset pts)",.5,sl_p.get("step_n",10.),5.,step=.5,key="sk")
                st.caption(f"Step 1: +{sl_p['step_n']:.0f}pts → SL={sl_p['step_n']-sl_p['step_k']:.0f}pts above entry. Step 2: +{2*sl_p['step_n']:.0f}pts → SL={2*sl_p['step_n']-sl_p['step_k']:.0f}pts.")
            elif sl_type=="Drawdown Recovery Exit (loss+recovery%)":
                sl_p["sl_points"]=st.number_input("Trigger loss pts",.5,500.,10.,step=.5,key="drl")
                sl_p["recovery_pct"]=st.number_input("Required recovery %",1.,99.,40.,step=1.,key="drp")
                st.caption(f"Exit if loss hits {sl_p.get('sl_points',10):.0f}pts and price doesn't recover {sl_p.get('recovery_pct',40):.0f}% of that loss.")
        st.markdown("**🎯 TARGET**")
        tgt_type=st.selectbox("Tgt",TARGET_TYPES,key="tgt_type",label_visibility="collapsed"); tgt_p:dict={}
        with st.expander("⚙️ Target Params"):
            if tgt_type=="Custom Points": tgt_p["target_points"]=st.number_input("Points",.1,1e5,20.,step=.5,key="tgtp")
            elif tgt_type=="ATR Based Target": tgt_p["target_atr_mult"]=st.number_input("ATR×",.5,10.,2.,step=.1,key="tam")
            elif tgt_type=="Risk Reward (min 1:2)": tgt_p["rr_ratio"]=st.number_input("R:R ≥2",2.,10.,2.,step=.5,key="rr")
            elif tgt_type=="🤖 Autopilot Target": st.caption("Fibonacci 2.618× risk")
            elif "display only" in tgt_type: st.caption("Shown only — never exits")
            elif tgt_type=="Profit Erosion Exit (peak-erosion%)":
                tgt_p["peak_profit_pts"]=st.number_input("Peak profit threshold (pts)",1.,500.,30.,step=1.,key="pep")
                tgt_p["erosion_pct"]=st.number_input("Erosion % from peak",1.,99.,40.,step=1.,key="erop")
                st.caption(f"Exit when profit was ≥{tgt_p.get('peak_profit_pts',30):.0f}pts and falls {tgt_p.get('erosion_pct',40):.0f}% from peak.")
        st.divider()

        # ── CONFIRMATION FILTERS ───────────────────────────
        st.markdown("**🔍 CONFIRMATION FILTERS**")
        st.caption("All disabled by default. Enable any to add signal confirmation.")
        conf:dict={}
        with st.expander("📐 Technical Confirmations", expanded=False):
            conf["adx_enabled"]=st.checkbox("ADX Filter",value=False,key="adx_en")
            if conf["adx_enabled"]:
                c1,c2=st.columns(2); conf["adx_min"]=c1.number_input("Min ADX",0.,100.,20.,key="adxmn"); conf["adx_max"]=c2.number_input("Max ADX",0.,100.,100.,key="adxmx")
                st.caption("Signal valid only when ADX is in [min, max] — filters choppy/extreme markets.")
            st.divider()
            conf["rsi_cf_enabled"]=st.checkbox("RSI Confirmation",value=False,key="rsi_cf")
            if conf["rsi_cf_enabled"]:
                st.caption("Long: RSI in [min, max]. Short: RSI < min OR RSI > max.")
                c1,c2=st.columns(2); conf["rsi_long_min"]=c1.number_input("Long RSI min",0.,100.,30.,key="rlmin"); conf["rsi_long_max"]=c2.number_input("Long RSI max",0.,100.,80.,key="rlmax")
                c3,c4=st.columns(2); conf["rsi_short_min"]=c3.number_input("Short RSI min",0.,100.,30.,key="rsmin"); conf["rsi_short_max"]=c4.number_input("Short RSI max",0.,100.,70.,key="rsmax")
            st.divider()
            conf["ema20_enabled"]=st.checkbox("EMA 20 Position Filter",value=False,key="ema20_en")
            if conf["ema20_enabled"]: st.caption("Long only when Close > EMA20. Short only when Close < EMA20.")
            conf["sma20_enabled"]=st.checkbox("SMA 20 Position Filter",value=False,key="sma20_en")
            if conf["sma20_enabled"]: st.caption("Long only when Close > SMA20. Short only when Close < SMA20.")
            st.divider()
            conf["bb_cf_enabled"]=st.checkbox("Bollinger Band Filter",value=False,key="bb_cf")
            if conf["bb_cf_enabled"]: st.caption("Long only below BB midline (oversold zone). Short only above midline.")
            st.divider()
            conf["st_enabled"]=st.checkbox("Supertrend Confirmation",value=False,key="st_en")
            if conf["st_enabled"]:
                c1,c2=st.columns(2); conf["st_period"]=c1.number_input("Period",3,50,10,key="stp"); conf["st_mult"]=c2.number_input("Multiplier",.5,6.,3.,step=.1,key="stm")
                st.caption("Long only when Supertrend is bullish (price > ST line). Short only bearish.")
            st.divider()
            conf["vol_cf_enabled"]=st.checkbox("Volume Breakout Confirmation",value=False,key="vol_cf")
            if conf["vol_cf_enabled"]:
                conf["vol_cf_mult"]=st.number_input("Volume ×",1.,5.,1.5,step=.1,key="vcm")
                st.caption("Signal valid only when Volume > avg × multiplier. Skipped on zero-volume symbols.")
            st.divider()
            conf["macd_enabled"]=st.checkbox("MACD Confirmation",value=False,key="macd_en")
            if conf["macd_enabled"]: st.caption("Long only when MACD line > Signal. Short only when MACD < Signal.")
            st.divider()
            conf["sr_cf_enabled"]=st.checkbox("S/R Proximity Filter",value=False,key="sr_cf")
            if conf["sr_cf_enabled"]: st.caption("Long only near support levels. Short only near resistance.")

        with st.expander("🏦 SMC / FVG Confirmations", expanded=False):
            conf["fvg_enabled"]=st.checkbox("Fair Value Gap (FVG)",value=False,key="fvg_en")
            if conf["fvg_enabled"]: st.caption("Long only when bullish FVG exists in last 5 candles. Short only with bearish FVG.")
            st.divider()
            conf["smc_enabled"]=st.checkbox("SMC — Break of Structure (BOS)",value=False,key="smc_en")
            if conf["smc_enabled"]: st.caption("Long only when price breaks above last swing high (bullish BOS). Short vice versa.")

        with st.expander("📏 Crossover Quality Filters", expanded=False):
            conf["angle_enabled"]=st.checkbox("Min Crossover Angle",value=False,key="ang_en")
            if conf["angle_enabled"]:
                conf["angle_min"]=st.number_input("Min angle (°, absolute)",0.,90.,20.,step=1.,key="angmin")
                c1,c2=st.columns(2); conf["angle_fast"]=c1.number_input("Fast EMA",1,100,9,key="angf"); conf["angle_slow"]=c2.number_input("Slow EMA",1,500,15,key="angs")
                st.caption("Filters weak crossovers. |angle| < threshold → signal rejected.")
            st.divider()
            conf["candle_size_enabled"]=st.checkbox("Min Candle Size",value=False,key="cs_en")
            if conf["candle_size_enabled"]:
                conf["candle_size_type"]=st.radio("Size type",["Custom Points","ATR Based"],horizontal=True,key="cst")
                if conf["candle_size_type"]=="Custom Points": conf["candle_size_pts"]=st.number_input("Min candle pts",.1,500.,10.,step=.5,key="csp")
                else: conf["candle_size_atr_mult"]=st.number_input("ATR multiplier",.1,3.,.5,step=.1,key="csam")
                st.caption("Rejects signals on small candles — ensures momentum behind the move.")

        with st.expander("⏱️ Trade Duration & Daily Limits", expanded=False):
            conf["max_duration_enabled"]=st.checkbox("Max Duration for Losing Trade",value=False,key="md_en")
            if conf["max_duration_enabled"]:
                conf["max_duration_min"]=st.slider("Max minutes in loss",5,30,15,key="mdm")
                st.caption(f"Auto-exits losing trade if held > {conf['max_duration_min']} minutes.")
            st.divider()
            conf["daily_loss_enabled"]=st.checkbox("Daily Loss Limit",value=False,key="dl_en")
            if conf["daily_loss_enabled"]:
                conf["daily_loss_pts"]=st.number_input("Max daily loss (pts)",100.,500.,200.,step=10.,key="dlp")
                st.caption(f"Stops trading for the day if cumulative loss ≥ {conf.get('daily_loss_pts',200):.0f} pts.")
            conf["daily_profit_enabled"]=st.checkbox("Daily Profit Target",value=False,key="dp_en")
            if conf["daily_profit_enabled"]:
                conf["daily_profit_pts"]=st.number_input("Daily profit target (pts)",50.,500.,150.,step=10.,key="dpp")
                st.caption(f"Stops trading for the day after earning ≥ {conf.get('daily_profit_pts',150):.0f} pts.")
            conf["daily_trades_enabled"]=st.checkbox("Daily Trade Limit",value=False,key="dt_en")
            if conf["daily_trades_enabled"]:
                conf["daily_trades_max"]=st.number_input("Max trades/day",5,100,10,step=1,key="dtm")
                st.caption(f"Max {conf.get('daily_trades_max',10)} trades per day.")

        st.divider()
        st.markdown("**🏦 DHAN**")
        dhan_on=st.checkbox("Enable Live Orders",value=False,key="dhan_on")
        with st.expander("🔑 Credentials"):
            d_cid=st.text_input("Client ID",type="password",key="d_cid"); d_tok=st.text_input("Token",type="password",key="d_tok")
            if st.button("🔗 Test",key="test_dhan"):
                if DhanBroker(d_cid,d_tok).connect(): st.success("✅ OK"); st.session_state.dhan_connected=True
                else: st.error("❌ Failed")
        if st.session_state.get("last_data_key"): st.caption(f"📊 {st.session_state.last_data_key}")
    return dict(tsym=tsym,tname=tname,interval=interval,period=period,qty=qty,strat=strat,sp=sp,sl_type=sl_type,sl_p=sl_p,tgt_type=tgt_type,tgt_p=tgt_p,dhan_on=dhan_on,conf=conf)

# ── TABS ──────────────────────────────────────────────────────
def _hl(row):
    return [f"background-color:{'rgba(63,185,80,.08)' if row['Result']=='WIN' else 'rgba(248,81,73,.08)'}"]*len(row)

def tab_backtest(cfg):
    df=st.session_state.current_data; sigs=st.session_state.signals; inds=st.session_state.indicators
    if df is None or df.empty: st.info("👈 Select instrument in sidebar — auto-loads."); return
    no_vol=not has_volume(df)
    if no_vol and cfg["strat"] in VOLUME_REQUIRED:
        st.markdown(f"<div class='warn-box'>⚠️ {cfg['strat']}: no volume on this instrument — price-only fallback active.</div>",unsafe_allow_html=True)
    st.markdown(f"### 📈 {cfg['tname']} · {cfg['interval']} · {cfg['period']}")
    cr,ci=st.columns([1,4])
    with cr: run_bt=st.button("▶ Run Backtest",type="primary",use_container_width=True,key="run_bt")
    with ci:
        bc=int((sigs==1).sum()); sc=int((sigs==-1).sum())
        entry_note="Immediate (signal close)" if cfg["strat"] in IMMEDIATE_ENTRY else "Standard (next candle open)"
        active_cf=sum(1 for k,v in cfg["conf"].items() if k.endswith("_enabled") and v)
        st.markdown(f"""<div style='background:#161b22;border:1px solid #30363d;border-radius:6px;padding:8px 12px;font-size:.81rem;display:flex;gap:12px;align-items:center'>
          <span style='color:#8b949e'>{len(df):,} candles</span><span style='color:#3fb950'>▲{bc} BUY</span><span style='color:#f85149'>▼{sc} SELL</span>
          <span style='color:#8b949e'>| {cfg["strat"][:24]}</span>
          {('<span style="color:#58a6ff">· '+str(active_cf)+' filters ON</span>') if active_cf else ''}
          <span style='color:#58a6ff;font-size:.73rem'>· {entry_note}</span>
        </div>""",unsafe_allow_html=True)
    if run_bt:
        with st.spinner("Backtesting…"):
            bt=run_backtest(df,sigs,cfg["sl_type"],cfg["sl_p"],cfg["tgt_type"],cfg["tgt_p"],cfg["qty"],cfg["strat"],cfg["conf"])
        st.session_state.backtest_results=bt
    bt_res=st.session_state.backtest_results
    fig=build_chart(df,sigs,inds,bt_res,title=f"{cfg['tname']} | {cfg['strat']} | {cfg['interval']}/{cfg['period']}")
    st.plotly_chart(fig,use_container_width=True,key="bt_chart")
    if bt_res is None: st.caption("Click ▶ Run Backtest to see results."); return
    if bt_res.empty: st.warning("No trades — widen period or adjust parameters."); return
    s=compute_stats(bt_res); pf_txt=f"{s['pf']:.2f}" if s["pf"]!=float("inf") else "∞"
    st.markdown("### 📊 Results")
    cols=st.columns(7)
    for col,lbl,val,clr in zip(cols,["Trades","Accuracy","Net P&L","Points","Profit Factor","Drawdown","Expectancy"],
        [str(s["total"]),f"{s['acc']:.1f}%",f"₹{s['tp']:,.0f}",f"{s['tpts']:+.1f}",pf_txt,f"₹{s['dd']:,.0f}",f"₹{s['exp']:,.1f}"],
        ["#58a6ff","#3fb950" if s["acc"]>=50 else "#f85149","#3fb950" if s["tp"]>=0 else "#f85149","#3fb950" if s["tpts"]>=0 else "#f85149","#3fb950" if s["pf"]>=1.5 else "#f0883e","#f85149","#58a6ff"]):
        sub_acc = f"<div class='ms'>W:{s['wins']} L:{s['losses']}</div>" if lbl=='Accuracy' else ''
        with col: st.markdown(f"<div class='mc' style='border-top:3px solid {clr}'><div class='ml'>{lbl}</div><div class='mv' style='color:{clr};font-size:.93rem'>{val}</div>{sub_acc}</div>",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    ca,cb=st.columns([3,1])
    with ca: st.plotly_chart(eq_fig(bt_res),use_container_width=True,key="bt_eq")
    with cb:
        pie=go.Figure(go.Pie(labels=["Wins","Losses"],values=[s["wins"],s["losses"]],marker_colors=["#3fb950","#f85149"],hole=.55,textinfo="label+percent"))
        pie.update_layout(template="plotly_dark",paper_bgcolor="#0d1117",height=185,margin=dict(l=0,r=0,t=0,b=0),showlegend=False)
        st.plotly_chart(pie,use_container_width=True,key="bt_pie")
        st.markdown(f"<div style='font-size:.77rem;color:#8b949e;line-height:1.8'>Max Win Str: <span style='color:#3fb950'>{s['mw']}</span><br>Max Loss Str: <span style='color:#f85149'>{s['ml']}</span><br>Avg Win: <span style='color:#3fb950'>₹{s['aw']:,.0f}</span><br>Avg Loss: <span style='color:#f85149'>₹{s['al']:,.0f}</span></div>",unsafe_allow_html=True)
    st.markdown("**📋 Trade Log**")
    st.dataframe(bt_res.style.apply(_hl,axis=1),use_container_width=True,height=280,key="bt_df")
    st.download_button("📥 Export CSV",bt_res.to_csv(index=False),file_name=f"bt_{datetime.now():%Y%m%d_%H%M}.csv",mime="text/csv",key="bt_dl")

def tab_live(cfg):
    df=st.session_state.current_data; sigs=st.session_state.signals; pos=st.session_state.live_position
    st.markdown("### ⚙️ Config")
    c1,c2,c3,c4=st.columns(4)
    for col,(lbl,val,sub) in zip([c1,c2,c3,c4],[("Asset",cfg["tname"],cfg["tsym"]),("Timeframe",cfg["interval"],f"Period:{cfg['period']}"),("Strategy",cfg["strat"][:26],f"Qty:{cfg['qty']}"),("Risk",cfg["sl_type"][:22],cfg["tgt_type"][:22])]):
        with col: st.markdown(f"<div class='icard'><div class='ml'>{lbl}</div><div style='color:#e6edf3;font-size:.85rem;font-weight:700;margin-top:2px'>{val}</div><div class='ms'>{sub}</div></div>",unsafe_allow_html=True)
    active_cf=sum(1 for k,v in cfg["conf"].items() if k.endswith("_enabled") and v)
    if active_cf: st.markdown(f"<div class='cf-active'>🔍 {active_cf} confirmation filter(s) active — signals will be filtered.</div>",unsafe_allow_html=True)
    st.divider()
    if df is None or df.empty: st.info("👈 Select instrument — auto-loads."); return
    running=st.session_state.live_running
    st.markdown("### 🎛️ Controls")
    b1,b2,b3,b4=st.columns(4)
    with b1:
        if not running:
            if st.button("▶ START",type="primary",use_container_width=True,key="start_btn"): st.session_state.live_running=True; st.rerun()
        else: st.markdown("<div class='auto-on'><div style='color:#3fb950;font-weight:900;font-size:.87rem'>● LIVE AUTO-TRADE ON</div><div style='color:#8b949e;font-size:.7rem'>1s signal loop</div></div>",unsafe_allow_html=True)
    with b2:
        if running:
            if st.button("⏹ STOP",use_container_width=True,key="stop_btn"): st.session_state.live_running=False; st.rerun()
        else: st.button("⏹ STOP",use_container_width=True,disabled=True,key="stop_dis")
    with b3:
        if pos:
            if st.button("🔴 SQUARE OFF",use_container_width=True,key="sq_btn"):
                _pos=st.session_state.live_position  # re-read at click time
                if _pos:
                    ltp_,_,_=fetch_ltp(cfg["tsym"]); ep=ltp_ if ltp_>0 else _pos["entry_price"]
                    exit_position(ep,"Manual Square Off",cfg); st.success(f"✅ Squared off @ {ep:.2f}"); st.rerun()
                else:
                    st.warning("Position already closed."); st.rerun()
        else: st.button("🔴 SQUARE OFF",use_container_width=True,disabled=True,key="sq_dis")
    with b4:
        sc="#3fb950" if running else "#8b949e"; pt=f"{'LONG' if pos and pos['direction']==1 else 'SHORT'} @ {pos['entry_price']:.2f}" if pos else "No Position"
        st.markdown(f"<div class='icard' style='text-align:center'><div style='color:{sc};font-weight:700;font-size:.81rem'>{'●LIVE' if running else '○IDLE'}</div><div style='color:#e6edf3;font-size:.76rem;margin-top:2px'>{pt}</div></div>",unsafe_allow_html=True)
    st.divider()
    st.markdown("### ⚡ Manual Instant Entry")
    ltp_c,_,_=fetch_ltp(cfg["tsym"]); cur=ltp_c if ltp_c>0 else float(df["Close"].iloc[-1]); a_v=float(_atr(df).iloc[-1])
    psl_b=calc_sl(df,cur,1,cfg["sl_type"],cfg["sl_p"],len(df)-1); ptgt_b=calc_target(cur,psl_b,1,cfg["tgt_type"],cfg["tgt_p"],a_v)
    psl_s=calc_sl(df,cur,-1,cfg["sl_type"],cfg["sl_p"],len(df)-1); ptgt_s=calc_target(cur,psl_s,-1,cfg["tgt_type"],cfg["tgt_p"],a_v)
    mb,ms_,mi=st.columns([1,1,2])
    with mb:
        rr_b=abs((ptgt_b or cur)-cur)/abs(cur-psl_b) if abs(cur-psl_b) else 0
        st.markdown(f"<div style='background:#1a3d26;border:1px solid #3fb950;border-radius:7px;padding:5px;margin-bottom:5px;text-align:center;font-size:.73rem'><b style='color:#3fb950'>▲ BUY LONG</b><br><span style='color:#8b949e'>Ent {cur:,.2f} SL {psl_b:,.2f} T {'%.2f'%ptgt_b if ptgt_b else '—'} RR {rr_b:.1f}:1</span></div>",unsafe_allow_html=True)
        if st.button("▲ BUY NOW",use_container_width=True,key="buy_now",disabled=pos is not None): st.session_state.live_position=enter_position(1,cfg,df); clear_all_cache(); st.rerun()
    with ms_:
        rr_s=abs((ptgt_s or cur)-cur)/abs(cur-psl_s) if abs(cur-psl_s) else 0
        st.markdown(f"<div style='background:#3d1a1a;border:1px solid #f85149;border-radius:7px;padding:5px;margin-bottom:5px;text-align:center;font-size:.73rem'><b style='color:#f85149'>▼ SELL SHORT</b><br><span style='color:#8b949e'>Ent {cur:,.2f} SL {psl_s:,.2f} T {'%.2f'%ptgt_s if ptgt_s else '—'} RR {rr_s:.1f}:1</span></div>",unsafe_allow_html=True)
        if st.button("▼ SELL NOW",use_container_width=True,key="sell_now",disabled=pos is not None): st.session_state.live_position=enter_position(-1,cfg,df); clear_all_cache(); st.rerun()
    with mi:
        dis="⚠️ Close position first" if pos else "✅ Ready"; dc="#f0883e" if pos else "#3fb950"
        st.markdown(f"<div class='icard'><div style='color:{dc};font-size:.78rem;font-weight:700;margin-bottom:4px'>{dis}</div><div style='color:#8b949e;font-size:.75rem;line-height:1.8'>LTP: <b style='color:#e6edf3'>{cur:,.2f}</b><br>ATR: <b style='color:#80cbc4'>{a_v:.2f}</b></div></div>",unsafe_allow_html=True)
    st.divider()
    last_sig=int(sigs.iloc[-1]) if sigs is not None else 0
    prx_lvl,prx_h,prx_d=strategy_proximity(df,cfg["strat"],cfg["sp"],sigs)
    st.markdown("### 🔔 Signal")
    sc1,sc2=st.columns([1,2])
    with sc1:
        if last_sig==1: st.markdown("<div class='sbuy'><div style='font-size:1.3rem'>▲</div><div style='font-weight:900;color:#3fb950'>BUY SIGNAL</div><div style='color:#8b949e;font-size:.75rem'>Auto-enters when START active</div></div>",unsafe_allow_html=True)
        elif last_sig==-1: st.markdown("<div class='ssell'><div style='font-size:1.3rem'>▼</div><div style='font-weight:900;color:#f85149'>SELL SIGNAL</div><div style='color:#8b949e;font-size:.75rem'>Auto-enters when START active</div></div>",unsafe_allow_html=True)
        else: st.markdown("<div class='swait'><div style='font-size:1.3rem'>⏳</div><div style='font-weight:900;color:#8b949e'>NO SIGNAL</div></div>",unsafe_allow_html=True)
    with sc2:
        cm={"signal":"#3fb950","near":"#f0883e","approaching":"#ffe082","far":"#8b949e"}; bm={"signal":"rgba(63,185,80,.1)","near":"rgba(240,136,62,.1)","approaching":"rgba(255,235,130,.06)","far":"rgba(139,148,158,.05)"}
        bdr=cm.get(prx_lvl,"#8b949e")
        bg_clr=bm.get(prx_lvl,"transparent")
        st.markdown(f"<div style='background:{bg_clr};border:1px solid {bdr};border-left:4px solid {bdr};border-radius:8px;padding:11px'><div style='color:{bdr};font-weight:700;font-size:.86rem'>{prx_h}</div><div style='color:#c9d1d9;font-size:.79rem;margin-top:4px'>{prx_d}</div></div>",unsafe_allow_html=True)
    st.divider()
    st.markdown("### 📊 Live Monitor")
    if running: auto_trade_fragment(cfg,cfg["conf"])
    else:
        pos2=st.session_state.live_position
        if pos2:
            d=pos2["direction"]; sl_eff=pos2.get("trail_sl",pos2["sl"])
            st.markdown(f"<div class='{'pos-long' if d==1 else 'pos-short'}'><div style='display:flex;flex-wrap:wrap;gap:14px'><div><div class='ml'>DIR</div><div style='color:{'#3fb950' if d==1 else '#f85149'};font-weight:900'>{'▲LONG' if d==1 else '▼SHORT'}</div></div><div><div class='ml'>ENTRY</div><div class='mv'>{pos2['entry_price']:,.2f}</div></div><div><div class='ml'>SL (eff.)</div><div class='mv' style='color:#f85149'>{sl_eff:,.2f}</div></div><div><div class='ml'>TARGET</div><div class='mv' style='color:#3fb950'>{'%.2f'%pos2['target'] if pos2.get('target') else '—'}</div></div></div><div style='color:#f0883e;font-size:.77rem;margin-top:6px'>⚠️ Monitor STOPPED. Click ▶ START to resume.</div></div>",unsafe_allow_html=True)
        else: st.markdown("<div class='pos-none'>No position · Click ▶ START to begin auto-trading</div>",unsafe_allow_html=True)
        if df is not None and sigs is not None:
            fig=build_chart(df,sigs,st.session_state.indicators,title=f"{cfg['tname']} | {cfg['strat']} (snapshot)")
            st.plotly_chart(fig,use_container_width=True,key="live_snap"); st.caption("▶ START to enable live auto-refresh.")

def tab_history():
    live=st.session_state.live_trades
    if not live: st.info("No live trades yet. Enter trades via Live Trading tab."); return
    df_h=pd.DataFrame(live); s=compute_stats(df_h); pf_txt=f"{s['pf']:.2f}" if s["pf"]!=float("inf") else "∞"
    st.markdown("### 📊 Live Portfolio")
    c1,c2,c3,c4=st.columns(4)
    for col,(lbl,val,sub,clr) in zip([c1,c2,c3,c4],[("Live Trades",str(s["total"]),f"W:{s['wins']} L:{s['losses']}","#58a6ff"),("Win Rate",f"{s['acc']:.1f}%","Accuracy","#3fb950" if s["acc"]>=50 else "#f85149"),("Net P&L",f"₹{s['tp']:,.0f}",f"{s['tpts']:+.1f}pts","#3fb950" if s["tp"]>=0 else "#f85149"),("Profit Factor",pf_txt,f"Exp ₹{s['exp']:,.1f}","#3fb950" if s["pf"]>=1.5 else "#f0883e")]):
        with col: st.markdown(f"<div class='mc' style='border-top:3px solid {clr}'><div class='ml'>{lbl}</div><div class='mv' style='color:{clr}'>{val}</div><div class='ms'>{sub}</div></div>",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    ca,cb=st.columns([2,1])
    with ca:
        bar=go.Figure(go.Bar(x=list(range(len(df_h))),y=df_h["P&L (₹)"],marker_color=["#3fb950" if x>0 else "#f85149" for x in df_h["P&L (₹)"]]))
        bar.add_hline(y=0,line=dict(color="#555",dash="dash",width=1)); bar.update_layout(template="plotly_dark",paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",height=220,title="P&L Per Trade",margin=dict(l=0,r=0,t=26,b=0),font=dict(color="#8b949e"))
        st.plotly_chart(bar,use_container_width=True,key="hist_bar")
    with cb: st.plotly_chart(eq_fig(df_h),use_container_width=True,key="hist_eq")
    f1,f2,f3=st.columns(3)
    with f1: fr=st.selectbox("Result",["All","WIN","LOSS"],key="fres")
    with f2: fd=st.selectbox("Direction",["All","LONG","SHORT"],key="fdir")
    with f3: fe=st.selectbox("Exit",["All"]+list(df_h["Exit Reason"].unique()),key="fex")
    filt=df_h.copy()
    if fr!="All": filt=filt[filt["Result"]==fr]
    if fd!="All": filt=filt[filt["Direction"]==fd]
    if fe!="All": filt=filt[filt["Exit Reason"]==fe]
    st.dataframe(filt.style.apply(_hl,axis=1),use_container_width=True,height=340,key="hist_df")
    dc,cc=st.columns(2)
    with dc: st.download_button("📥 Export CSV",filt.to_csv(index=False),file_name=f"live_{datetime.now():%Y%m%d_%H%M}.csv",mime="text/csv",key="hist_dl")
    with cc:
        if st.button("🗑️ Clear History",key="hist_clr"): st.session_state.live_trades=[]; st.session_state.live_position=None; clear_all_cache(); st.rerun()

def tab_leaderboard(cfg):
    st.markdown("### 🏆 Strategy Leaderboard")
    st.markdown("<div class='icard' style='margin-bottom:10px'><div style='color:#8b949e;font-size:.81rem'>Tests all strategies on current data with active SL/Target config and confirmation filters. Use this to pick the best strategy for current market conditions.</div></div>",unsafe_allow_html=True)
    df=st.session_state.current_data
    if df is None or df.empty: st.info("👈 Select instrument — auto-loads."); return
    no_vol=not has_volume(df)
    if no_vol: st.markdown("<div class='warn-box'>⚠️ Zero-volume instrument — volume-based strategies use price-only fallback.</div>",unsafe_allow_html=True)
    lb_key=f"{cfg['tsym']}|{cfg['interval']}|{cfg['period']}|{cfg['sl_type']}|{cfg['tgt_type']}"
    if st.button("🚀 Run All Strategies",type="primary",key="run_lb"):
        results=[]; skip={"Volume Profile FR [VPFR]"}; strats_to_run=[s for s in STRATEGIES if s not in skip]
        prog=st.progress(0,"Running…")
        for idx,s_ in enumerate(strats_to_run):
            prog.progress((idx+1)/len(strats_to_run),f"Testing: {s_[:36]}")
            try:
                sg,_=run_strategy(df,s_,{},cfg["conf"])
                if (sg!=0).sum()==0: continue
                bt=run_backtest(df,sg,cfg["sl_type"],cfg["sl_p"],cfg["tgt_type"],cfg["tgt_p"],cfg["qty"],s_,cfg["conf"])
                if bt.empty: continue
                s=compute_stats(bt)
                results.append({"Strategy":s_,"Trades":s["total"],"Win Rate":f"{s['acc']:.1f}%","Net P&L (₹)":round(s["tp"],0),"Profit Factor":round(min(s["pf"],999),2),"Drawdown":round(s["dd"],0),"Expectancy":round(s["exp"],2),"_pf":min(s["pf"],999),"_tp":s["tp"]})
            except: pass
        prog.empty()
        if results:
            df_lb=pd.DataFrame(results).sort_values("_pf",ascending=False).reset_index(drop=True); df_lb.index+=1
            st.session_state.leaderboard_results=df_lb; st.session_state.lb_ran_for=lb_key
        else: st.warning("No trades generated — try a wider period.")
    df_lb=st.session_state.leaderboard_results
    if df_lb is not None and not df_lb.empty:
        if st.session_state.lb_ran_for!=lb_key: st.caption("⚠️ Config changed — re-run for updated results.")
        def style_lb(row):
            pf=row["Profit Factor"]; bg="rgba(63,185,80,.10)" if pf>=1.5 else "rgba(255,235,100,.06)" if pf>=1. else "rgba(248,81,73,.08)"
            return [f"background-color:{bg}"]*len(row)
        display_cols=["Strategy","Trades","Win Rate","Net P&L (₹)","Profit Factor","Drawdown","Expectancy"]
        st.dataframe(df_lb[display_cols].style.apply(style_lb,axis=1),use_container_width=True,height=400,key="lb_df")
        top3=df_lb.head(3); medals=["🥇","🥈","🥉"]; cols_=st.columns(3)
        for i,((_,row),medal,col) in enumerate(zip(top3.iterrows(),medals,cols_)):
            with col:
                pf=row["Profit Factor"]; clr="#f0883e" if i==0 else "#8b949e"
                st.markdown(f"<div class='icard' style='border-top:3px solid {clr};text-align:center'><div style='font-size:1.4rem'>{medal}</div><div style='color:#e6edf3;font-size:.82rem;font-weight:700;margin:4px 0'>{row['Strategy'][:28]}</div><div style='color:#8b949e;font-size:.77rem;line-height:1.8'>PF: <b style='color:{'#3fb950' if pf>=1.5 else '#f85149'}'>{pf}</b><br>Win: <b style='color:#e6edf3'>{row['Win Rate']}</b><br>P&L: <b style='color:{'#3fb950' if row['Net P&L (₹)']>=0 else '#f85149'}'>₹{row['Net P&L (₹)']:,.0f}</b></div></div>",unsafe_allow_html=True)



# ── ANALYSIS TAB ──────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_analysis_data(ticker, years):
    time.sleep(0.3)
    try:
        raw = yf.download(ticker, period=f"{years}y", interval="1d",
                          progress=False, auto_adjust=True)
        if raw.empty: return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
        raw.columns = [str(c).strip() for c in raw.columns]
        return raw[["Open","High","Low","Close","Volume"]].dropna()
    except: return pd.DataFrame()

def _monthly_pivot(df_daily):
    """Build Month x Year pivot of monthly returns (%)."""
    monthly = df_daily["Close"].resample("ME").last().pct_change() * 100
    df_m = pd.DataFrame({"year": monthly.index.year,
                          "month": monthly.index.month,
                          "ret": monthly.values}).dropna()
    pivot = df_m.pivot(index="month", columns="year", values="ret")
    mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.index = [mn[m-1] for m in pivot.index]
    return pivot

def _tf_heatmap_pivot(df, interval):
    """Build heatmap pivot appropriate for the selected timeframe."""
    df2 = df.copy(); df2["ret"] = df2["Close"].pct_change() * 100
    df2 = df2.dropna(subset=["ret"])
    try:
        if interval in ("1m","5m","15m"):
            df2["row"] = df2.index.hour
            df2["col"] = df2.index.dayofweek
            col_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            title = "Avg Return % — Hour of Day × Day of Week"
        elif interval == "1h":
            df2["row"] = df2.index.dayofweek
            df2["col"] = df2.index.hour
            col_names = [f"{h:02d}:00" for h in range(24)]
            title = "Avg Return % — Day of Week × Hour"
        elif interval == "1d":
            df2["row"] = df2.index.month
            df2["col"] = df2.index.dayofweek
            col_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            title = "Avg Return % — Month × Day of Week"
        else:  # 1wk or wider
            df2["row"] = df2.index.year
            df2["col"] = df2.index.quarter
            col_names = ["Q1","Q2","Q3","Q4"]
            title = "Avg Return % — Year × Quarter"
        pivot = df2.groupby(["row","col"])["ret"].mean().unstack()
        if interval == "1d":
            mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            pivot.index = [mn[m-1] for m in pivot.index]
        elif interval in ("1m","5m","15m"):
            pivot.index = [f"{h:02d}:00" for h in pivot.index]
        pivot.columns = [col_names[c] if c < len(col_names) else str(c) for c in pivot.columns]
        return pivot, title
    except Exception as e:
        return pd.DataFrame(), f"Could not build heatmap: {e}"

def _heat_fig(pivot, title, fmt=".1f"):
    """Build a plotly diverging heatmap (red=negative, green=positive)."""
    if pivot.empty: return go.Figure()
    z = pivot.values.tolist()
    text = [[f"{v:{fmt}}%" if not (isinstance(v, float) and np.isnan(v)) else ""
             for v in row] for row in pivot.values]
    zmax = max(abs(np.nanmin(pivot.values)), abs(np.nanmax(pivot.values))) or 1
    fig = go.Figure(go.Heatmap(
        z=z, x=list(pivot.columns), y=list(pivot.index),
        colorscale=[[0,"#c62828"],[0.35,"#ef5350"],[0.5,"#21262d"],
                    [0.65,"#26a69a"],[1,"#00695c"]],
        zmid=0, zmin=-zmax, zmax=zmax,
        text=text, texttemplate="%{text}", textfont={"size": 9, "color": "#e6edf3"},
        colorbar=dict(title=dict(text="Return %", font=dict(color="#8b949e")), tickfont=dict(color="#8b949e")),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        title=dict(text=f"<b>{title}</b>", font=dict(size=13, color="#e6edf3"), x=0.5),
        font=dict(family="monospace", size=11, color="#8b949e"),
        xaxis=dict(gridcolor="#21262d", tickfont=dict(color="#8b949e")),
        yaxis=dict(gridcolor="#21262d", tickfont=dict(color="#8b949e")),
        margin=dict(l=0, r=0, t=40, b=0), height=420,
    )
    return fig

def _style_returns(v):
    if not isinstance(v, (int, float)): return ""
    if v > 0:  return "color:#3fb950;font-weight:600"
    if v < 0:  return "color:#f85149;font-weight:600"
    return "color:#8b949e"

def tab_analysis(cfg):
    st.markdown("### 📊 Market Analysis")
    df_live = st.session_state.current_data

    # ── Settings bar ─────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1: years = st.number_input("Years of history for heatmaps", 1, 30, 10, key="ana_years")
    with c2: rows_show = st.number_input("OHLC rows to display", 10, 500, 100, key="ana_rows")
    with c3: st.markdown(f"""<div class='icard' style='margin-top:4px'>
        <div class='ml'>Instrument</div>
        <div style='color:#e6edf3;font-size:.9rem;font-weight:700'>{cfg["tname"]}</div>
        <div class='ms'>{cfg["tsym"]}</div></div>""", unsafe_allow_html=True)

    # ── OHLC Table ────────────────────────────────────────────
    st.markdown("#### 📋 OHLC Data with Returns")
    df_src = df_live if df_live is not None and not df_live.empty else None
    if df_src is None:
        st.info("👈 Select instrument in sidebar — data loads automatically.")
    else:
        df_tbl = df_src.copy().tail(rows_show)
        df_tbl["Change (pts)"]    = df_tbl["Close"].diff().round(2)
        df_tbl["Return %"]        = (df_tbl["Close"].pct_change() * 100).round(3)
        df_tbl["Cumul Chg (pts)"] = (df_tbl["Close"] - df_src["Close"].iloc[0]).round(2)
        df_tbl["Cumul. Return %"] = ((df_tbl["Close"] / df_src["Close"].iloc[0] - 1) * 100).round(3)
        df_tbl.index = df_tbl.index.strftime("%Y-%m-%d %H:%M") if hasattr(df_tbl.index[0], "hour") else df_tbl.index.strftime("%Y-%m-%d")
        df_tbl = df_tbl.round({"Open":2,"High":2,"Low":2,"Close":2,"Volume":0,"Change (pts)":2,"Return %":3,"Cumul Chg (pts)":2,"Cumul. Return %":3})

        def style_df(df):
            style = pd.DataFrame("", index=df.index, columns=df.columns)
            for col in ["Change (pts)", "Return %", "Cumul Chg (pts)", "Cumul. Return %"]:
                if col in df.columns:
                    style[col] = df[col].map(_style_returns)
            # Color Close relative to Open
            if "Close" in df.columns and "Open" in df.columns:
                style["Close"] = (df["Close"] >= df["Open"]).map(
                    {True: "color:#26a69a;font-weight:600", False: "color:#ef5350;font-weight:600"})
            return style

        try:
            styled = df_tbl.style.apply(style_df, axis=None)
            st.dataframe(styled, use_container_width=True, height=380, key="ana_tbl")
        except Exception:
            st.dataframe(df_tbl, use_container_width=True, height=380, key="ana_tbl2")

        # Quick stats
        ret = df_tbl["Return %"].dropna()
        sc1,sc2,sc3,sc4,sc5 = st.columns(5)
        for col,lbl,val,clr in zip([sc1,sc2,sc3,sc4,sc5],
            ["Avg Daily Ret","Best Day","Worst Day","Positive Days","Volatility (σ)"],
            [f"{ret.mean():.3f}%", f"{ret.max():.2f}%", f"{ret.min():.2f}%",
             f"{(ret>0).sum()}/{len(ret)}", f"{ret.std():.3f}%"],
            ["#58a6ff","#3fb950","#f85149","#3fb950" if (ret>0).mean()>=0.5 else "#f85149","#f0883e"]):
            with col:
                st.markdown(f"""<div class='mc' style='border-top:3px solid {clr}'>
                  <div class='ml'>{lbl}</div>
                  <div class='mv' style='color:{clr};font-size:.88rem'>{val}</div>
                </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Monthly Returns Heatmap (Month × Year) ───────────────
    st.markdown(f"#### 🗓️ Monthly Returns Heatmap — {cfg['tname']} (last {years} years)")
    with st.spinner("Loading historical data…"):
        df_hist = fetch_analysis_data(cfg["tsym"], int(years))

    if df_hist is None or df_hist.empty:
        st.warning("Could not load historical data. Check ticker symbol.")
    else:
        pivot_my = _monthly_pivot(df_hist)
        if not pivot_my.empty:
            fig_my = _heat_fig(pivot_my,
                f"Monthly Returns % — {cfg['tname']} (Month × Year, last {years}y)")
            st.plotly_chart(fig_my, use_container_width=True, key="ana_heat_my")

            # Annual summary bar
            ann = df_hist["Close"].resample("YE").last().pct_change() * 100
            ann = ann.dropna()
            fig_ann = go.Figure(go.Bar(
                x=ann.index.year, y=ann.values,
                marker_color=["#3fb950" if v >= 0 else "#f85149" for v in ann.values],
                text=[f"{v:.1f}%" for v in ann.values],
                textposition="outside", textfont=dict(size=9, color="#e6edf3"),
            ))
            fig_ann.add_hline(y=0, line=dict(color="#555", dash="dash", width=1))
            fig_ann.update_layout(
                template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                title=dict(text=f"<b>Annual Returns % — {cfg['tname']}</b>",
                           font=dict(size=12, color="#e6edf3"), x=0.5),
                font=dict(color="#8b949e"), height=280,
                margin=dict(l=0, r=0, t=34, b=0),
                xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
            )
            st.plotly_chart(fig_ann, use_container_width=True, key="ana_annual")
        else:
            st.warning("Not enough data for monthly heatmap.")

    st.divider()

    # ── Timeframe Returns Heatmap ─────────────────────────────
    st.markdown(f"#### ⏱️ Timeframe Heatmap — {cfg['tname']} ({cfg['interval']}/{cfg['period']})")
    st.caption("Shows average return % per cell based on time patterns in the loaded data.")
    df_tf = st.session_state.current_data
    if df_tf is None or df_tf.empty:
        st.info("Load data first via sidebar selection.")
    else:
        pivot_tf, tf_title = _tf_heatmap_pivot(df_tf, cfg["interval"])
        if not pivot_tf.empty:
            fig_tf = _heat_fig(pivot_tf, tf_title, fmt=".2f")
            fig_tf.update_layout(height=360)
            st.plotly_chart(fig_tf, use_container_width=True, key="ana_heat_tf")

            # Return distribution histogram
            rets = df_tf["Close"].pct_change().dropna() * 100
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=rets, nbinsx=50, name="Return %",
                marker_color=["#3fb950" if v >= 0 else "#f85149" for v in rets],
                marker_line=dict(color="#0d1117", width=0.5),
            ))
            fig_dist.add_vline(x=0, line=dict(color="#555", dash="dash", width=1))
            fig_dist.add_vline(x=float(rets.mean()), line=dict(color="#58a6ff", dash="dot", width=1),
                               annotation_text=f"Mean {rets.mean():.3f}%",
                               annotation_font_color="#58a6ff")
            fig_dist.update_layout(
                template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                title=dict(text=f"<b>Return Distribution — {cfg['interval']}/{cfg['period']}</b>",
                           font=dict(size=12, color="#e6edf3"), x=0.5),
                font=dict(color="#8b949e"), height=260,
                margin=dict(l=0, r=0, t=34, b=0),
                xaxis=dict(title="Return %", gridcolor="#21262d"),
                yaxis=dict(title="Frequency", gridcolor="#21262d"),
            )
            st.plotly_chart(fig_dist, use_container_width=True, key="ana_dist")

            # Stats table for the pivot
            st.markdown("**Pivot Statistics**")
            stats_rows = []
            for col in pivot_tf.columns:
                col_data = pivot_tf[col].dropna()
                if not col_data.empty:
                    stats_rows.append({
                        "Period": str(col),
                        "Avg Ret %": round(float(col_data.mean()), 3),
                        "Best %":   round(float(col_data.max()), 3),
                        "Worst %":  round(float(col_data.min()), 3),
                        "Win Rate": f"{(col_data>0).mean()*100:.0f}%",
                    })
            if stats_rows:
                df_stats = pd.DataFrame(stats_rows).set_index("Period")
                def style_stats(df):
                    sty = pd.DataFrame("", index=df.index, columns=df.columns)
                    for c in ["Avg Ret %","Best %","Worst %"]:
                        if c in df.columns: sty[c] = df[c].map(_style_returns)
                    return sty
                try:
                    st.dataframe(df_stats.style.apply(style_stats, axis=None),
                                 use_container_width=True, height=220, key="ana_stats")
                except Exception:
                    st.dataframe(df_stats, use_container_width=True, height=220, key="ana_stats2")
        else:
            st.warning(f"Could not build timeframe heatmap: {tf_title}")

def recommendations():
    with st.expander("📖 How to use Confirmation Filters + Honest Assessment",expanded=False):
        st.markdown("""<div class='icard'>
<h4 style='color:#58a6ff'>🔍 Confirmation Filters — What They Do & When to Use</h4>
<div style='color:#8b949e;font-size:.81rem;line-height:2'>
<b style='color:#e6edf3'>ADX (20-100)</b>: Only trade when trend strength is between min and max. ADX &lt;20 = chop, &gt;40 = strong trend. Best combo: 25-60.<br>
<b style='color:#e6edf3'>RSI for Long/Short</b>: Long when RSI 30-80 (not overbought), Short when RSI &lt;30 or &gt;70.<br>
<b style='color:#e6edf3'>EMA20/SMA20</b>: Classic trend filter. Eliminates counter-trend entries. Always recommended ON.<br>
<b style='color:#e6edf3'>Supertrend</b>: Strong filter. Only takes trades aligned with supertrend direction. Reduces trades by ~40% but improves quality.<br>
<b style='color:#e6edf3'>FVG</b>: SMC-based. Only enters when a fair value gap exists. Best on 15m+ charts.<br>
<b style='color:#e6edf3'>SMC BOS</b>: Requires break of structure. Highest quality signals, fewest trades.<br>
<b style='color:#e6edf3'>Step Trail SL</b>: N=10, K=5 means: +10pts → SL at +5; +20pts → SL at +15; +30pts → SL at +25. Locks in profit progressively.<br>
<b style='color:#e6edf3'>Drawdown Recovery</b>: Exit if loss hits X pts and doesn't recover Y% from worst point. Cuts stubborn losers.<br>
<b style='color:#e6edf3'>Profit Erosion</b>: Exit if profit peaks at X then falls Y%. Protects accumulated profit.<br>
</div>
<h4 style='color:#f0883e;margin-top:12px'>Recommended Filter Combos</h4>
<div style='color:#8b949e;font-size:.81rem;line-height:2'>
<b style='color:#3fb950'>Conservative (high quality, fewer trades)</b>: ADX 25-60 + EMA20 + Supertrend + Volume<br>
<b style='color:#3fb950'>Aggressive (more trades, still filtered)</b>: RSI Confirmation + EMA20 + MACD<br>
<b style='color:#3fb950'>SMC style</b>: FVG + SMC BOS + S/R Proximity<br>
<b style='color:#f0883e'>Daily risk management</b>: Daily Loss 200pts + Daily Profit 150pts + Max Duration 15min always recommended ON
</div></div>""",unsafe_allow_html=True)

def main():
    st.markdown("""<div style='display:flex;align-items:center;gap:10px;padding:2px 0 9px;border-bottom:1px solid #21262d;margin-bottom:9px'>
      <span style='font-size:1.7rem'>📊</span>
      <div><div style='font-size:1.4rem;font-weight:900;color:#e6edf3;letter-spacing:1px'>QuantAlgo Pro</div>
        <div style='font-size:.61rem;color:#8b949e;letter-spacing:2px'>v6.0 · 14 STRATEGIES · 17 CONFIRMATION FILTERS · 3 SMART SL/TGT TYPES</div></div>
      <div style='margin-left:auto;display:flex;gap:7px'>
        <div style='background:#1a4731;color:#3fb950;padding:2px 9px;border-radius:20px;font-size:.66rem;font-weight:700'>● PAPER MODE</div>
        <div style='background:#1c2128;color:#58a6ff;padding:2px 9px;border-radius:20px;font-size:.66rem'>Auto-load on change</div>
      </div></div>""",unsafe_allow_html=True)

    cfg=sidebar(); recommendations()

    # Auto-load data when instrument/TF changes (no button needed)
    data_key=f"{cfg['tsym']}|{cfg['interval']}|{cfg['period']}"
    if data_key!=st.session_state.last_data_key:
        df=fetch_data(cfg["tsym"],cfg["interval"],cfg["period"])
        if df is not None and not df.empty:
            st.session_state.current_data=df; st.session_state.last_data_key=data_key
            st.session_state.last_strat_key=""; st.session_state.backtest_results=None; st.session_state.leaderboard_results=None

    # Auto rerun strategy when strategy/params/conf change
    sp_key=str(sorted(cfg["sp"].items())); cf_key=str(sorted(cfg["conf"].items()))
    strat_key=f"{data_key}|{cfg['strat']}|{sp_key}|{cf_key}"
    df=st.session_state.current_data
    if df is not None and not df.empty and strat_key!=st.session_state.last_strat_key:
        sigs,inds=run_strategy(df,cfg["strat"],cfg["sp"],cfg["conf"])
        st.session_state.signals=sigs; st.session_state.indicators=inds; st.session_state.last_strat_key=strat_key

    tab1,tab2,tab3,tab4,tab5=st.tabs(["📈 Backtest","📡 Live Trading","📋 Live History","🏆 Leaderboard","📊 Analysis"])
    with tab1: tab_backtest(cfg)
    with tab2: tab_live(cfg)
    with tab3: tab_history()
    with tab4: tab_leaderboard(cfg)
    with tab5: tab_analysis(cfg)

if __name__=="__main__":
    main()
