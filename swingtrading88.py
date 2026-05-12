# =============================================================================
# SMART INVESTING PLATFORM  v3.0
# Fixes: chart row/col error, auto-refresh, Simple Buy/Sell,
#        signal-status panel, stop button, gap-up/gap-down handling
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time, threading, datetime, traceback, warnings
from typing import Optional, Dict, List, Tuple, Any
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Smart Investing Platform", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
  .block-container{padding-top:0.8rem}
  .mbox{background:#0d1628;border:1px solid #1e3a5f;border-radius:9px;
         padding:12px 10px;text-align:center;margin-bottom:5px}
  .mlbl{color:#6b7280;font-size:11px;margin-bottom:3px}
  .mval{font-size:20px;font-weight:700}
  .g{color:#00e676} .r{color:#ff5252} .y{color:#ffca28} .w{color:#e0e0e0}
  .sig-panel{background:#0d1628;border:1px solid #1e3a5f;border-radius:9px;padding:14px;margin:6px 0}
  div[data-testid="stTabs"] button{font-size:14px;font-weight:600}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# THREAD-SAFE PERSISTENT STORE
# ─────────────────────────────────────────────────────────────────────────────
class _Store:
    _inst = None
    _cls_lock = threading.Lock()
    def __new__(cls):
        if cls._inst is None:
            with cls._cls_lock:
                if cls._inst is None:
                    o = super().__new__(cls)
                    o._lk = threading.Lock()
                    o._d  = dict(
                        live_running=False, live_thread=None,
                        live_position=None, live_ltp=None,
                        live_pnl=0.0, live_signal="HOLD",
                        live_log=[], trade_history=[],
                        backtest_trades=[], backtest_result=None,
                        live_df=None, error=None,
                        trailing_sl=None, trailing_target=None,
                        last_fetch=0.0, last_signal_time=None,
                        last_signal_type=None,
                    )
                    cls._inst = o
        return cls._inst
    def get(self, k, d=None):
        with self._lk: return self._d.get(k, d)
    def set(self, k, v):
        with self._lk: self._d[k] = v
    def upd(self, d):
        with self._lk: self._d.update(d)
    def clear_pos(self):
        with self._lk:
            self._d.update(dict(live_position=None, live_pnl=0.0,
                                trailing_sl=None, trailing_target=None))

STORE = _Store()

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = {
    "NIFTY 50":"^NSEI","BANK NIFTY":"^NSEBANK","SENSEX":"^BSESN","NIFTY IT":"^CNXIT",
    "BTC/USD":"BTC-USD","ETH/USD":"ETH-USD","USD/INR":"USDINR=X",
    "GOLD":"GC=F","SILVER":"SI=F","CRUDE OIL":"CL=F",
    "EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X",
    "RELIANCE":"RELIANCE.NS","TCS":"TCS.NS","INFY":"INFY.NS","HDFC BANK":"HDFCBANK.NS",
    "Custom ✏️":"__custom__",
}
INTERVALS = ["1m","5m","15m","30m","1h","4h","1d","1wk"]
IV_MAX    = {"1m":"5d","5m":"60d","15m":"60d","30m":"60d",
             "1h":"730d","4h":"730d","1d":"max","1wk":"max"}
PERIODS   = ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","max"]
STRATEGIES = [
    "🏆 Smart Algo (Recommended)",
    "⚡ Simple Buy",
    "⚡ Simple Sell",
    "SuperTrend + EMA Cloud",
    "EMA Crossover + ADX",
    "MACD Momentum",
    "Bollinger Squeeze",
    "Heikin Ashi Reversal",
    "Smart Money Concepts (SMC)",
    "Opening Range Breakout",
    "Donchian Breakout",
]
SL_TYPES = [
    "ATR Based","Custom Points","Trailing – Fixed Points",
    "Trailing – Candle Low/High","Trailing – Pivot Low/High",
    "Risk Reward Based","Volatility Based (BB)","Signal Reversal Exit",
]
TGT_TYPES = [
    "ATR Based","Custom Points","Risk Reward Ratio",
    "Volatility Based (BB)","Fibonacci Extension",
    "Signal Reversal Exit","Trailing Target (Display Only)",
]
_TRAIL_SL = {"ATR Based","Trailing – Fixed Points","Trailing – Candle Low/High",
             "Trailing – Pivot Low/High","Volatility Based (BB)"}
_NO_TGT   = {"Signal Reversal Exit","Trailing Target (Display Only)"}

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH  (1.5 s rate-limit guard)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_data(ticker, interval, period):
    elapsed = time.time() - STORE.get("last_fetch", 0.0)
    if elapsed < 1.5: time.sleep(1.5 - elapsed)
    STORE.set("last_fetch", time.time())

    fetch_iv = "1h" if interval == "4h" else interval
    # Clamp period
    pd_days  = {"1d":1,"5d":5,"60d":60,"1mo":30,"3mo":90,"6mo":180,
                "1y":365,"2y":730,"730d":730,"5y":1825,"10y":3650,"max":999999}
    max_p    = IV_MAX.get(fetch_iv, "max")
    if max_p != "max" and pd_days.get(period,9999) > pd_days.get(max_p,9999):
        period = max_p

    try:
        raw = yf.download(ticker, interval=fetch_iv, period=period,
                          progress=False, auto_adjust=True, timeout=20)
    except Exception as e:
        STORE.set("error", f"yfinance: {e}"); return None
    if raw is None or raw.empty: return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    df = raw.copy()
    if interval == "4h":
        df = df.resample("4h",label="right",closed="right").agg(
            {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
        ).dropna(subset=["Close"])
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["Close"])
    return df

def has_vol(df): return "Volume" in df.columns and df["Volume"].fillna(0).sum()>0

# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def ema(s,n): return s.ewm(span=n,adjust=False,min_periods=1).mean()
def sma(s,n): return s.rolling(n,min_periods=1).mean()

def calc_rsi(s,n=14):
    d=s.diff(); g=d.clip(lower=0).rolling(n,min_periods=1).mean()
    l=(-d.clip(upper=0)).rolling(n,min_periods=1).mean()
    return 100-100/(1+g/l.replace(0,np.nan))

def calc_macd(s,f=12,sl=26,sg=9):
    m=ema(s,f)-ema(s,sl); sig=ema(m,sg); return m,sig,m-sig

def calc_atr(df,n=14):
    h,l,pc=df.High,df.Low,df.Close.shift(1)
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(span=n,adjust=False,min_periods=1).mean()

def calc_adx(df,n=14):
    h,l=df.High,df.Low
    pdm=(h-h.shift(1)).clip(lower=0); ndm=(l.shift(1)-l).clip(lower=0)
    pdm=pdm.where(pdm>ndm,0); ndm=ndm.where(ndm>pdm,0)
    atr_=calc_atr(df,n)
    pdi=100*ema(pdm,n)/atr_.replace(0,np.nan)
    ndi=100*ema(ndm,n)/atr_.replace(0,np.nan)
    dx=100*(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)
    return ema(dx,n),pdi,ndi

def calc_bb(s,n=20,k=2.):
    m=sma(s,n); std=s.rolling(n,min_periods=1).std(); return m+k*std,m,m-k*std

def calc_st(df,n=10,mult=3.):
    atr_=calc_atr(df,n).values; hl2=((df.High+df.Low)/2).values; cl=df.Close.values
    N=len(df); ub=hl2+mult*atr_; lb=hl2-mult*atr_
    st=np.full(N,np.nan); dr=np.zeros(N,int)
    st[0]=ub[0]; dr[0]=-1
    for i in range(1,N):
        lb[i]=lb[i] if lb[i]>lb[i-1] or cl[i-1]<lb[i-1] else lb[i-1]
        ub[i]=ub[i] if ub[i]<ub[i-1] or cl[i-1]>ub[i-1] else ub[i-1]
        prev=st[i-1]
        if np.isnan(prev) or prev==ub[i-1]:
            dr[i]=1;  st[i]=lb[i] if cl[i]>ub[i] else (dr.__setitem__(i,-1) or ub[i])
        else:
            dr[i]=-1; st[i]=ub[i] if cl[i]<lb[i] else (dr.__setitem__(i,1) or lb[i])
    ix=df.index
    return pd.Series(st,ix),pd.Series(dr,ix),pd.Series(ub,ix),pd.Series(lb,ix)

def calc_ha(df):
    ha=pd.DataFrame(index=df.index)
    ha["Close"]=(df.Open+df.High+df.Low+df.Close)/4
    ha["Open"]=(df.Open.shift(1)+df.Close.shift(1))/2
    ha.loc[ha.index[0],"Open"]=df.Open.iloc[0]
    ha["High"]=pd.concat([df.High,ha.Open,ha.Close],axis=1).max(axis=1)
    ha["Low"] =pd.concat([df.Low, ha.Open,ha.Close],axis=1).min(axis=1)
    return ha

def detect_ob(df,lb=5):
    ob=pd.Series(0,index=df.index)
    for i in range(lb,len(df)):
        if df.Open.iloc[i-lb]>df.Close.iloc[i-lb] and df.Close.iloc[i]>df.High.iloc[i-lb]: ob.iloc[i]=1
        elif df.Close.iloc[i-lb]>df.Open.iloc[i-lb] and df.Close.iloc[i]<df.Low.iloc[i-lb]: ob.iloc[i]=-1
    return ob

def detect_lh(df,lb=10):
    lh=pd.Series(0,index=df.index)
    for i in range(lb+1,len(df)):
        rh=df.High.iloc[i-lb:i].max(); rl=df.Low.iloc[i-lb:i].min()
        c,o,lo,hi=df.Close.iloc[i],df.Open.iloc[i],df.Low.iloc[i],df.High.iloc[i]
        if lo<rl and c>o and c>rl: lh.iloc[i]=1
        if hi>rh and c<o and c<rh: lh.iloc[i]=-1
    return lh

def fib_lvl(df,lb=50):
    r=df.tail(lb); hi,lo=r.High.max(),r.Low.min(); d=hi-lo
    return {"0":hi,"0.236":hi-.236*d,"0.382":hi-.382*d,"0.5":hi-.5*d,
            "0.618":hi-.618*d,"1":lo,"1.618":lo-.618*d}

def add_ind(df):
    c=df.Close
    for p in [9,21,50,200]: df[f"ema{p}"]=ema(c,p)
    df["sma20"]=sma(c,20); df["sma50"]=sma(c,50)
    df["rsi"]=calc_rsi(c)
    df["macd"],df["msig"],df["mhist"]=calc_macd(c)
    df["atr"]=calc_atr(df)
    df["adx"],df["pdi"],df["ndi"]=calc_adx(df)
    df["bb_u"],df["bb_m"],df["bb_l"]=calc_bb(c)
    df["bb_w"]=(df.bb_u-df.bb_l)/df.bb_m
    df["bb_sq"]=df.bb_w<df.bb_w.rolling(50,min_periods=10).quantile(.2)
    df["st"],df["st_d"],df["st_ub"],df["st_lb"]=calc_st(df)
    ha=calc_ha(df); df["ha_c"]=ha.Close; df["ha_o"]=ha.Open
    df["ha_bull"]=df.ha_c>df.ha_o
    if has_vol(df):
        df["vol_ma"]=sma(df.Volume,20)
        df["vol_r"]=df.Volume/df.vol_ma.replace(0,1)
    else:
        df["vol_ma"]=0; df["vol_r"]=1.
    df["ob"]=detect_ob(df); df["lh"]=detect_lh(df)
    df["eb"]=(df.ema9>df.ema21)&(df.ema21>df.ema50)
    df["ebs"]=(df.ema9<df.ema21)&(df.ema21<df.ema50)
    df["mc_u"]=(df.macd>df.msig)&(df.macd.shift(1)<=df.msig.shift(1))
    df["mc_d"]=(df.macd<df.msig)&(df.macd.shift(1)>=df.msig.shift(1))
    df["regime"]=np.where(df.adx>25,"TREND",np.where(df.adx<15,"CHOPPY","NORMAL"))
    return df

# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR FILTER
# ─────────────────────────────────────────────────────────────────────────────
def _ok(df,i,side,ena):
    if not ena: return True
    r=df.iloc[i]
    if side=="BUY":
        if ena.get("rsi")    and r.rsi>75:          return False
        if ena.get("macd")   and r.mhist<0:         return False
        if ena.get("adx")    and r.adx<18:          return False
        if ena.get("ema")    and not r.eb:           return False
        if ena.get("volume") and has_vol(df) and r.vol_r<1.: return False
        if ena.get("bb")     and r.Close>r.bb_u:    return False
    else:
        if ena.get("rsi")    and r.rsi<25:          return False
        if ena.get("macd")   and r.mhist>0:         return False
        if ena.get("adx")    and r.adx<18:          return False
        if ena.get("ema")    and not r.ebs:         return False
        if ena.get("volume") and has_vol(df) and r.vol_r<1.: return False
        if ena.get("bb")     and r.Close<r.bb_l:    return False
    return True

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────
def sig_smart(df, ui=False, ena=None):
    S=pd.Series(0,index=df.index)
    for i in range(55,len(df)):
        r,p=df.iloc[i],df.iloc[i-1]
        vok=(not has_vol(df)) or r.vol_r>=1.1
        if r.adx>=20:
            fu=r.st_d==1  and p.st_d==-1; fd=r.st_d==-1 and p.st_d==1
            b=(fu or (r.st_d==1 and r.mc_u)) and r.eb  and 45<r.rsi<76
            s=(fd or (r.st_d==-1 and r.mc_d)) and r.ebs and 24<r.rsi<56
        else:
            b=(r.Close<=r.bb_l*1.003) and r.rsi<38 and r.ha_bull
            s=(r.Close>=r.bb_u*0.997) and r.rsi>62 and not r.ha_bull
            if r.lh==1:  b=True
            if r.lh==-1: s=True
        if b and vok and _ok(df,i,"BUY",ena if ui else None):   S.iloc[i]=1
        if s and vok and _ok(df,i,"SELL",ena if ui else None):  S.iloc[i]=-1
    return S

def sig_simple_buy(df, ui=False, ena=None):
    """
    Simple Buy: generates BUY on the FIRST candle in backtest.
    In live mode, treated as 'enter immediately' on start.
    Re-entries happen when price bounces off EMA21 after a pullback.
    """
    S=pd.Series(0,index=df.index)
    if len(df)<2: return S
    # First valid candle → BUY
    S.iloc[50] = 1
    # Re-entry: close crosses back above ema21 after being below it
    for i in range(51,len(df)):
        r,p=df.iloc[i],df.iloc[i-1]
        if r.Close>r.ema21 and p.Close<=p.ema21 and r.eb:
            S.iloc[i]=1
    return S

def sig_simple_sell(df, ui=False, ena=None):
    """
    Simple Sell: generates SELL on the FIRST candle.
    In live mode, treated as 'enter short/PE immediately' on start.
    Re-entries when price breaks below EMA21.
    """
    S=pd.Series(0,index=df.index)
    if len(df)<2: return S
    S.iloc[50] = -1
    for i in range(51,len(df)):
        r,p=df.iloc[i],df.iloc[i-1]
        if r.Close<r.ema21 and p.Close>=p.ema21 and r.ebs:
            S.iloc[i]=-1
    return S

def sig_st_ema(df, ui=False, ena=None):
    S=pd.Series(0,index=df.index)
    for i in range(55,len(df)):
        r,p=df.iloc[i],df.iloc[i-1]
        if r.st_d==1 and p.st_d==-1 and r.Close>r.ema50 and _ok(df,i,"BUY",ena if ui else None):   S.iloc[i]=1
        if r.st_d==-1 and p.st_d==1 and r.Close<r.ema50 and _ok(df,i,"SELL",ena if ui else None):  S.iloc[i]=-1
    return S

def sig_ema_x(df, ui=False, ena=None):
    S=pd.Series(0,index=df.index)
    for i in range(55,len(df)):
        r,p=df.iloc[i],df.iloc[i-1]
        xu=r.ema9>r.ema21 and p.ema9<=p.ema21; xd=r.ema9<r.ema21 and p.ema9>=p.ema21
        if xu and r.adx>20 and r.pdi>r.ndi  and _ok(df,i,"BUY",ena if ui else None):  S.iloc[i]=1
        if xd and r.adx>20 and r.ndi>r.pdi  and _ok(df,i,"SELL",ena if ui else None): S.iloc[i]=-1
    return S

def sig_macd(df, ui=False, ena=None):
    S=pd.Series(0,index=df.index)
    for i in range(35,len(df)):
        r=df.iloc[i]
        if r.mc_u and r.ema9>r.ema21 and _ok(df,i,"BUY",ena if ui else None):  S.iloc[i]=1
        if r.mc_d and r.ema9<r.ema21 and _ok(df,i,"SELL",ena if ui else None): S.iloc[i]=-1
    return S

def sig_bb(df, ui=False, ena=None):
    S=pd.Series(0,index=df.index)
    for i in range(55,len(df)):
        r,p=df.iloc[i],df.iloc[i-1]
        sr=p.bb_sq and not r.bb_sq
        if sr:
            if r.Close>r.bb_m and r.mhist>0 and _ok(df,i,"BUY",ena if ui else None):  S.iloc[i]=1
            if r.Close<r.bb_m and r.mhist<0 and _ok(df,i,"SELL",ena if ui else None): S.iloc[i]=-1
        if r.Close>r.bb_u and p.Close<=p.bb_u and r.adx>20 and _ok(df,i,"BUY",ena if ui else None):  S.iloc[i]=1
        if r.Close<r.bb_l and p.Close>=p.bb_l and r.adx>20 and _ok(df,i,"SELL",ena if ui else None): S.iloc[i]=-1
    return S

def sig_ha(df, ui=False, ena=None):
    S=pd.Series(0,index=df.index)
    for i in range(30,len(df)):
        r,p=df.iloc[i],df.iloc[i-1]
        if r.ha_bull and not p.ha_bull and r.Close>r.ema21 and _ok(df,i,"BUY",ena if ui else None):  S.iloc[i]=1
        if not r.ha_bull and p.ha_bull and r.Close<r.ema21 and _ok(df,i,"SELL",ena if ui else None): S.iloc[i]=-1
    return S

def sig_smc(df, ui=False, ena=None):
    S=pd.Series(0,index=df.index)
    for i in range(20,len(df)):
        r=df.iloc[i]
        if (r.lh==1 or r.ob==1) and r.Close>r.ema21  and _ok(df,i,"BUY",ena if ui else None):  S.iloc[i]=1
        if (r.lh==-1 or r.ob==-1) and r.Close<r.ema21 and _ok(df,i,"SELL",ena if ui else None): S.iloc[i]=-1
    return S

def sig_orb(df, ui=False, ena=None):
    S=pd.Series(0,index=df.index)
    if not hasattr(df.index,"date"): return S
    tmp=df.copy(); tmp["_d"]=tmp.index.date; OR=5
    for d,g in tmp.groupby("_d"):
        if len(g)<OR+2: continue
        oh=g.High.iloc[:OR].max(); ol=g.Low.iloc[:OR].min()
        for j in range(OR,len(g)):
            idx=g.index[j]; pr=g.Close.iloc[j]; pp=g.Close.iloc[j-1]
            if pr>oh and pp<=oh: S[idx]=1
            elif pr<ol and pp>=ol: S[idx]=-1
    return S

def sig_don(df, ui=False, ena=None):
    S=pd.Series(0,index=df.index); n=20
    dh=df.High.rolling(n,min_periods=n).max(); dl=df.Low.rolling(n,min_periods=n).min()
    for i in range(n+1,len(df)):
        r,p=df.iloc[i],df.iloc[i-1]
        if r.Close>dh.iloc[i-1] and p.Close<=dh.iloc[i-1] and r.adx>20: S.iloc[i]=1
        if r.Close<dl.iloc[i-1] and p.Close>=dl.iloc[i-1] and r.adx>20: S.iloc[i]=-1
    return S

STFN={
    "🏆 Smart Algo (Recommended)":sig_smart,
    "⚡ Simple Buy":sig_simple_buy,
    "⚡ Simple Sell":sig_simple_sell,
    "SuperTrend + EMA Cloud":sig_st_ema,
    "EMA Crossover + ADX":sig_ema_x,
    "MACD Momentum":sig_macd,
    "Bollinger Squeeze":sig_bb,
    "Heikin Ashi Reversal":sig_ha,
    "Smart Money Concepts (SMC)":sig_smc,
    "Opening Range Breakout":sig_orb,
    "Donchian Breakout":sig_don,
}
def get_sigs(df,strat,ui,ena): return STFN.get(strat,sig_smart)(df,ui,ena)

# ─────────────────────────────────────────────────────────────────────────────
# SL / TARGET  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def isl(df,i,side,t,v,ep):
    r=df.iloc[i]; a=r.atr
    if side=="LONG":
        if t=="ATR Based":            return ep-v*a
        if t=="Custom Points":        return ep-v
        if "Trailing" in t:           return ep-v*a
        if t=="Risk Reward Based":    return ep-v
        if t=="Volatility Based (BB)":return r.bb_l
        if t=="Signal Reversal Exit": return ep-2.5*a
        return ep-v*a
    else:
        if t=="ATR Based":            return ep+v*a
        if t=="Custom Points":        return ep+v
        if "Trailing" in t:           return ep+v*a
        if t=="Risk Reward Based":    return ep+v
        if t=="Volatility Based (BB)":return r.bb_u
        if t=="Signal Reversal Exit": return ep+2.5*a
        return ep+v*a

def itgt(df,i,side,t,v,ep,sl):
    r=df.iloc[i]; a=r.atr; risk=abs(ep-sl)
    fi=fib_lvl(df,50)
    if side=="LONG":
        if t=="ATR Based":            return ep+v*a
        if t=="Custom Points":        return ep+v
        if t=="Risk Reward Ratio":    return ep+v*risk
        if t=="Volatility Based (BB)":return r.bb_u
        if t=="Fibonacci Extension":  return fi.get("1.618",ep+3*a)
        return ep+3*a   # Signal/Trailing – display-only
    else:
        if t=="ATR Based":            return ep-v*a
        if t=="Custom Points":        return ep-v
        if t=="Risk Reward Ratio":    return ep-v*risk
        if t=="Volatility Based (BB)":return r.bb_l
        if t=="Fibonacci Extension":  return fi.get("1.618",ep-3*a)
        return ep-3*a

def trail(cur,price,df,i,side,t,v):
    r=df.iloc[i]; a=r.atr
    if side=="LONG":
        if t=="Trailing – Fixed Points":    return max(cur,price-v)
        if t=="Trailing – Candle Low/High": return max(cur,float(r.Low))
        if t=="Trailing – Pivot Low/High":
            if i>=3: return max(cur,df.Low.iloc[i-3:i].min())
        if t=="ATR Based":                  return max(cur,price-v*a)
        if t=="Volatility Based (BB)":      return max(cur,r.bb_l)
    else:
        if t=="Trailing – Fixed Points":    return min(cur,price+v)
        if t=="Trailing – Candle Low/High": return min(cur,float(r.High))
        if t=="Trailing – Pivot Low/High":
            if i>=3: return min(cur,df.High.iloc[i-3:i].max())
        if t=="ATR Based":                  return min(cur,price+v*a)
        if t=="Volatility Based (BB)":      return min(cur,r.bb_u)
    return cur

# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTING  (N+1 entry, SL-first, gap-aware)
# ─────────────────────────────────────────────────────────────────────────────
def backtest(df,sigs,sl_t,sl_v,tg_t,tg_v,cap=100000.):
    trades=[]; pos=None; eq=cap; eqc=[cap]

    for i in range(1,len(df)):
        row=df.iloc[i]; psig=sigs.iloc[i-1]

        if pos is None:
            if psig not in(1,-1): continue
            side="LONG" if psig==1 else "SHORT"
            ep=float(row.Open)
            sl=isl(df,i-1,side,sl_t,sl_v,ep)
            tg=itgt(df,i-1,side,tg_t,tg_v,ep,sl)
            # ── Gap-up / gap-down entry guard ──
            # If open already breaches SL, skip entry (gap moved against us)
            if side=="LONG"  and (sl>=ep or tg<=ep): continue
            if side=="SHORT" and (sl<=ep or tg>=ep): continue
            risk=abs(ep-sl)
            if risk<=0: continue
            qty=max(1,int(eq*0.02/risk))
            pos=dict(side=side,ep=ep,sl=sl,tg=tg,isl=sl,itg=tg,
                     qty=qty,et=row.name,sig=psig)
            continue

        # ── Trailing SL update ──
        if sl_t in _TRAIL_SL:
            ref=float(row.High) if pos["side"]=="LONG" else float(row.Low)
            pos["sl"]=trail(pos["sl"],ref,df,i,pos["side"],sl_t,sl_v)

        # ── Trailing Target (display only) ──
        if tg_t=="Trailing Target (Display Only)":
            a=float(row.atr)
            pos["tg"]=(float(row.Close)+2*a) if pos["side"]=="LONG" else (float(row.Close)-2*a)

        xp=xr=None
        op=float(row.Open); lo=float(row.Low); hi=float(row.High)

        if pos["side"]=="LONG":
            # ── Gap-down through SL: exit at Open ──
            if op<=pos["sl"]:
                xp,xr=op,"SL_GAP"
            elif lo<=pos["sl"]:
                xp,xr=pos["sl"],"SL"
            elif tg_t not in _NO_TGT:
                # ── Gap-up through target: exit at Open ──
                if op>=pos["tg"]: xp,xr=op,"TGT_GAP"
                elif hi>=pos["tg"]: xp,xr=pos["tg"],"TARGET"
            if xr is None and tg_t=="Signal Reversal Exit" and sigs.iloc[i]==-1:
                xp,xr=float(row.Close),"SIGNAL"
        else:
            if op>=pos["sl"]:
                xp,xr=op,"SL_GAP"
            elif hi>=pos["sl"]:
                xp,xr=pos["sl"],"SL"
            elif tg_t not in _NO_TGT:
                if op<=pos["tg"]: xp,xr=op,"TGT_GAP"
                elif lo<=pos["tg"]: xp,xr=pos["tg"],"TARGET"
            if xr is None and tg_t=="Signal Reversal Exit" and sigs.iloc[i]==1:
                xp,xr=float(row.Close),"SIGNAL"

        if xr:
            pnl=(xp-pos["ep"])*pos["qty"] if pos["side"]=="LONG" else (pos["ep"]-xp)*pos["qty"]
            eq+=pnl
            trades.append(dict(EntryTime=pos["et"],ExitTime=row.name,
                Side=pos["side"],EntryPrice=round(pos["ep"],2),ExitPrice=round(xp,2),
                SL=round(pos["isl"],2),Target=round(pos["itg"],2),
                Qty=pos["qty"],PnL=round(pnl,2),
                PnLpct=round(pnl/(pos["ep"]*pos["qty"])*100,2),
                ExitReason=xr,Equity=round(eq,2)))
            eqc.append(eq); pos=None

    if pos:
        xp=float(df.Close.iloc[-1])
        pnl=(xp-pos["ep"])*pos["qty"] if pos["side"]=="LONG" else (pos["ep"]-xp)*pos["qty"]
        eq+=pnl
        trades.append(dict(EntryTime=pos["et"],ExitTime=df.index[-1],
            Side=pos["side"],EntryPrice=round(pos["ep"],2),ExitPrice=round(xp,2),
            SL=round(pos["isl"],2),Target=round(pos["itg"],2),
            Qty=pos["qty"],PnL=round(pnl,2),
            PnLpct=round(pnl/(pos["ep"]*pos["qty"])*100,2),
            ExitReason="OPEN",Equity=round(eq,2)))

    tdf=pd.DataFrame(trades); m=calc_metrics(tdf,cap,eq,eqc)
    return dict(trades=tdf,eq=eqc,final=eq,metrics=m,df=df,sigs=sigs)

def calc_metrics(tdf,cap,fin,eqc):
    if tdf.empty:
        return dict(n=0,wr=0,pnl=0,pf=0,dd=0,ret=0,aw=0,al=0,rr=0,exp=0,wins=0,loss=0,sh=0)
    w=tdf[tdf.PnL>0]; l=tdf[tdf.PnL<=0]
    wr=len(w)/len(tdf)*100
    gp=w.PnL.sum() if not w.empty else 0
    gl=abs(l.PnL.sum()) if not l.empty else 1e-9
    aw=w.PnL.mean() if not w.empty else 0
    al=abs(l.PnL.mean()) if not l.empty else 0
    eq_s=pd.Series(eqc)
    dd=abs(((eq_s-eq_s.cummax())/eq_s.cummax()*100).min())
    ret=(fin-cap)/cap*100
    arr=np.diff(np.array(eqc))/np.array(eqc[:-1])
    sh=arr.mean()/arr.std()*np.sqrt(252) if len(arr)>1 and arr.std()>0 else 0
    return dict(n=len(tdf),wr=round(wr,1),pnl=round(tdf.PnL.sum(),2),
                pf=round(gp/gl,2),dd=round(dd,1),ret=round(ret,1),
                aw=round(aw,2),al=round(al,2),rr=round(aw/al if al>0 else 0,2),
                exp=round((wr/100*aw)-((1-wr/100)*al),2),
                wins=len(w),loss=len(l),sh=round(sh,2))

# ─────────────────────────────────────────────────────────────────────────────
# DHAN INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────
def dhan_eq(cfg,sid,txn,qty,prod,exch,otype,price=0):
    try:
        from pydhan import pydhan
        d=pydhan(client_id=cfg["cid"],access_token=cfg["tok"])
        return {"ok":True,"resp":d.place_order(
            security_id=str(sid),
            exchange_segment=d.NSE if exch=="NSE" else d.BSE,
            transaction_type=d.BUY if txn=="BUY" else d.SELL,
            quantity=int(qty),
            order_type=d.MARKET if otype=="MARKET" else d.LIMIT,
            product_type=d.INTRADAY if prod=="INTRADAY" else d.DELIVERY,
            price=float(price) if otype=="LIMIT" else 0)}
    except ImportError: return {"ok":False,"err":"pip install pydhan"}
    except Exception as e: return {"ok":False,"err":str(e)}

def dhan_opt(cfg,sid,txn,qty,exch,otype,price=0):
    try:
        from dhanhq import dhanhq
        d=dhanhq(cfg["cid"],cfg["tok"])
        return {"ok":True,"resp":d.place_order(
            transactionType=txn,exchangeSegment=exch,
            productType="INTRADAY",orderType=otype,validity="DAY",
            securityId=str(sid),quantity=int(qty),
            price=float(price) if otype=="LIMIT" else 0,triggerPrice=0)}
    except ImportError: return {"ok":False,"err":"pip install dhanhq"}
    except Exception as e: return {"ok":False,"err":str(e)}

def dhan_entry(sig,ltp,cfg):
    if not cfg.get("on"): return {"ok":False,"err":"disabled"}
    p=ltp if cfg.get("e_ot")=="LIMIT" else 0
    if cfg.get("opts"):
        sid=cfg["ce_id"] if sig=="BUY" else cfg["pe_id"]
        if not sid: return {"ok":False,"err":"Security ID missing"}
        return dhan_opt(cfg,sid,"BUY",cfg.get("oqty",65),cfg.get("o_exch","NSE_FNO"),cfg.get("e_ot","MARKET"),p)
    txn="BUY" if sig=="BUY" else "SELL"
    return dhan_eq(cfg,cfg.get("sid","1594"),txn,cfg.get("qty",1),
                   cfg.get("prod","INTRADAY"),cfg.get("exch","NSE"),cfg.get("e_ot","MARKET"),p)

def dhan_exit(esig,ltp,cfg):
    if not cfg.get("on"): return {"ok":False,"err":"disabled"}
    p=ltp
    if cfg.get("opts"):
        sid=cfg["ce_id"] if esig=="BUY" else cfg["pe_id"]
        if not sid: return {"ok":False,"err":"Security ID missing"}
        return dhan_opt(cfg,sid,"SELL",cfg.get("oqty",65),cfg.get("o_exch","NSE_FNO"),cfg.get("x_ot","MARKET"),p)
    txn="SELL" if esig=="BUY" else "BUY"
    return dhan_eq(cfg,cfg.get("sid","1594"),txn,cfg.get("qty",1),
                   cfg.get("prod","INTRADAY"),cfg.get("exch","NSE"),cfg.get("x_ot","MARKET"),p)

# ─────────────────────────────────────────────────────────────────────────────
# LIVE TRADING THREAD
# ─────────────────────────────────────────────────────────────────────────────
def live_loop(ticker,interval,period,strategy,sl_t,sl_v,tg_t,tg_v,ui,ena,dcfg):
    def log(msg,lv="INFO"):
        ts=datetime.datetime.now().strftime("%H:%M:%S")
        lg=STORE.get("live_log",[]); lg.append(f"[{ts}][{lv}] {msg}")
        STORE.set("live_log",lg[-150:])

    log(f"Started | {ticker} | {interval} | {strategy}")
    is_simple_buy  = "Simple Buy"  in strategy
    is_simple_sell = "Simple Sell" in strategy

    while STORE.get("live_running",False):
        try:
            df=fetch_data(ticker,interval,period)
            if df is None or len(df)<60:
                log("Insufficient data","WARN"); time.sleep(3); continue

            df=add_ind(df); sig=get_sigs(df,strategy,ui,ena)
            ltp=float(df.Close.iloc[-1]); atr_now=float(df.atr.iloc[-1])
            cur=int(sig.iloc[-1]); prv=int(sig.iloc[-2]) if len(sig)>=2 else 0

            sig_txt="BUY" if cur==1 else "SELL" if cur==-1 else "HOLD"
            STORE.upd(dict(live_ltp=ltp,live_df=df,error=None,live_signal=sig_txt))

            # Track last signal time
            if cur!=0:
                STORE.upd(dict(last_signal_time=datetime.datetime.now(),
                               last_signal_type=sig_txt))

            pos=STORE.get("live_position")

            # ── ENTRY ──
            if pos is None:
                # Simple strategies: enter immediately when live starts
                if is_simple_buy:   entry_sig=1
                elif is_simple_sell: entry_sig=-1
                else:               entry_sig=prv

                if entry_sig not in(1,-1):
                    continue

                side="LONG" if entry_sig==1 else "SHORT"
                sl_=isl(df,-2,side,sl_t,sl_v,ltp)
                tg_=itgt(df,-2,side,tg_t,tg_v,ltp,sl_)

                # ── Gap guard: validate SL/Target make sense vs current LTP ──
                if side=="LONG"  and (sl_>=ltp or tg_<=ltp): continue
                if side=="SHORT" and (sl_<=ltp or tg_>=ltp): continue

                new_pos=dict(side=side,sig="BUY" if entry_sig==1 else "SELL",
                             ep=ltp,sl=sl_,tg=tg_,isl=sl_,itg=tg_,
                             qty=1,et=datetime.datetime.now())
                STORE.set("live_position",new_pos)
                STORE.set("trailing_sl",sl_)
                STORE.set("trailing_target",tg_)
                log(f"ENTRY {side} @ {ltp:.2f}  SL={sl_:.2f}  TGT={tg_:.2f}")
                if dcfg.get("on"):
                    r=dhan_entry("BUY" if entry_sig==1 else "SELL",ltp,dcfg)
                    log(f"Dhan entry: {r}")

                # For simple strategies, reset so we don't re-enter on next tick
                if is_simple_buy or is_simple_sell:
                    is_simple_buy=False; is_simple_sell=False

            else:
                # ── Trailing SL ──
                if sl_t in _TRAIL_SL:
                    tsl=STORE.get("trailing_sl",pos["sl"])
                    ref=ltp
                    tsl=trail(tsl,ref,df,-1,pos["side"],sl_t,sl_v)
                    STORE.set("trailing_sl",tsl); pos["sl"]=tsl
                    STORE.set("live_position",pos)

                # ── Trailing Target (display only) ──
                if tg_t=="Trailing Target (Display Only)":
                    STORE.set("trailing_target",
                               ltp+2*atr_now if pos["side"]=="LONG" else ltp-2*atr_now)

                # ── Gap-up / gap-down detection via overnight Open ──
                latest_open=float(df.Open.iloc[-1])
                gap_down_sl  = pos["side"]=="LONG"  and latest_open<=pos["sl"]
                gap_up_sl    = pos["side"]=="SHORT" and latest_open>=pos["sl"]

                pnl=(ltp-pos["ep"])*pos["qty"] if pos["side"]=="LONG" else (pos["ep"]-ltp)*pos["qty"]
                STORE.set("live_pnl",pnl)

                xp=xr=None
                if gap_down_sl:   xp,xr=latest_open,"SL_GAP"
                elif gap_up_sl:   xp,xr=latest_open,"SL_GAP"
                elif pos["side"]=="LONG":
                    if ltp<=pos["sl"]: xp,xr=ltp,"SL"
                    elif tg_t not in _NO_TGT and ltp>=pos["tg"]: xp,xr=ltp,"TARGET"
                    elif tg_t=="Signal Reversal Exit" and cur==-1: xp,xr=ltp,"SIGNAL"
                else:
                    if ltp>=pos["sl"]: xp,xr=ltp,"SL"
                    elif tg_t not in _NO_TGT and ltp<=pos["tg"]: xp,xr=ltp,"TARGET"
                    elif tg_t=="Signal Reversal Exit" and cur==1:  xp,xr=ltp,"SIGNAL"

                if xr:
                    fpnl=(xp-pos["ep"])*pos["qty"] if pos["side"]=="LONG" else (pos["ep"]-xp)*pos["qty"]
                    rec=dict(EntryTime=pos["et"],ExitTime=datetime.datetime.now(),
                             Side=pos["side"],EntryPrice=round(pos["ep"],2),
                             ExitPrice=round(xp,2),SL=round(pos["isl"],2),
                             Target=round(pos["itg"],2),Qty=pos["qty"],
                             PnL=round(fpnl,2),
                             PnLpct=round(fpnl/(pos["ep"]*pos["qty"])*100,2),
                             ExitReason=xr,Source="LIVE")
                    h=STORE.get("trade_history",[]); h.append(rec); STORE.set("trade_history",h)
                    log(f"EXIT {pos['side']} @ {xp:.2f} | {xr} | PnL={fpnl:+.2f}")
                    if dcfg.get("on"):
                        r2=dhan_exit(pos["sig"],xp,dcfg); log(f"Dhan exit: {r2}")
                    STORE.clear_pos()

        except Exception:
            STORE.set("error",traceback.format_exc()[:400]); time.sleep(5)

    log("Live trading stopped.")

# ─────────────────────────────────────────────────────────────────────────────
# CHART  (always 4 rows; secondary_y only on row 4; add_shape for hlines)
# ─────────────────────────────────────────────────────────────────────────────
def make_chart(df, sigs=None, trades_df=None, hlines=None):
    vol=has_vol(df)
    specs=[
        [{"secondary_y":False}],
        [{"secondary_y":False}],
        [{"secondary_y":False}],
        [{"secondary_y":True}],   # row 4: ADX (primary) + Volume (secondary)
    ]
    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,
                       vertical_spacing=0.025,
                       row_heights=[0.54,0.15,0.15,0.16],
                       specs=specs,
                       subplot_titles=["Price + Indicators","RSI","MACD","ADX + Volume"])

    # ── Row 1: Candles ──
    fig.add_trace(go.Candlestick(x=df.index,open=df.Open,high=df.High,
                                  low=df.Low,close=df.Close,name="Price",
                                  increasing_line_color="#26a69a",
                                  decreasing_line_color="#ef5350"),
                  row=1,col=1,secondary_y=False)

    # EMAs
    for p,col,dsh in [(9,"#ff9800","solid"),(21,"#2196f3","solid"),(50,"#ba68c8","dash")]:
        cn=f"ema{p}"
        if cn in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df[cn],name=f"EMA{p}",
                                      line=dict(color=col,width=1,dash=dsh),opacity=.85),
                          row=1,col=1,secondary_y=False)

    # SuperTrend
    if "st" in df.columns and "st_d" in df.columns:
        bull=df["st"].where(df["st_d"]==1); bear=df["st"].where(df["st_d"]==-1)
        fig.add_trace(go.Scatter(x=df.index,y=bull,name="ST↑",
                                  line=dict(color="#00e676",width=1.8),connectgaps=False),
                      row=1,col=1,secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index,y=bear,name="ST↓",
                                  line=dict(color="#ff5252",width=1.8),connectgaps=False),
                      row=1,col=1,secondary_y=False)

    # BB
    if "bb_u" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df.bb_u,name="BB↑",
                                  line=dict(color="rgba(100,181,246,.4)",width=1,dash="dot")),
                      row=1,col=1,secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index,y=df.bb_l,name="BB↓",
                                  line=dict(color="rgba(100,181,246,.4)",width=1,dash="dot"),
                                  fill="tonexty",fillcolor="rgba(100,181,246,.05)"),
                      row=1,col=1,secondary_y=False)

    # Trade markers
    if trades_df is not None and not trades_df.empty:
        L=trades_df[trades_df.Side=="LONG"]; S=trades_df[trades_df.Side=="SHORT"]
        if not L.empty:
            fig.add_trace(go.Scatter(x=L.EntryTime,y=L.EntryPrice,mode="markers",
                                      name="Buy Entry",marker=dict(symbol="triangle-up",
                                      size=11,color="#00e676",line=dict(color="white",width=1))),
                          row=1,col=1,secondary_y=False)
            fig.add_trace(go.Scatter(x=L.ExitTime,y=L.ExitPrice,mode="markers",
                                      name="Buy Exit",marker=dict(symbol="triangle-down",
                                      size=11,color="#ffca28",line=dict(color="white",width=1))),
                          row=1,col=1,secondary_y=False)
        if not S.empty:
            fig.add_trace(go.Scatter(x=S.EntryTime,y=S.EntryPrice,mode="markers",
                                      name="Sell Entry",marker=dict(symbol="triangle-down",
                                      size=11,color="#ff5252",line=dict(color="white",width=1))),
                          row=1,col=1,secondary_y=False)

    # Horizontal lines (SL/Target/Entry) via add_shape – no row/col issues
    if hlines and len(df)>1:
        x0,x1=df.index[0],df.index[-1]
        for hl in hlines:
            fig.add_shape(type="line",x0=x0,x1=x1,y0=hl["y"],y1=hl["y"],
                           line=dict(color=hl["c"],dash=hl.get("d","solid"),width=1.5),
                           row=1,col=1)
            fig.add_annotation(x=df.index[max(0,len(df)-3)],y=hl["y"],
                                text=hl.get("lbl",""),showarrow=False,
                                font=dict(color=hl["c"],size=10),
                                xanchor="right",row=1,col=1)

    # ── Row 2: RSI ──
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df.rsi,name="RSI",
                                  line=dict(color="#ff9800",width=1.5)),
                      row=2,col=1,secondary_y=False)
        x0,x1=df.index[0],df.index[-1]
        for y,c in [(70,"rgba(255,82,82,.5)"),(30,"rgba(0,230,118,.5)"),(50,"rgba(150,150,150,.3)")]:
            fig.add_shape(type="line",x0=x0,x1=x1,y0=y,y1=y,
                           line=dict(color=c,dash="dash",width=1),row=2,col=1)

    # ── Row 3: MACD ──
    if "macd" in df.columns:
        hc=["#26a69a" if v>=0 else "#ef5350" for v in df.mhist]
        fig.add_trace(go.Bar(x=df.index,y=df.mhist,name="Hist",marker_color=hc,opacity=.7),
                      row=3,col=1,secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index,y=df.macd,name="MACD",
                                  line=dict(color="#2196f3",width=1.4)),
                      row=3,col=1,secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index,y=df.msig,name="Sig",
                                  line=dict(color="#ff9800",width=1.4)),
                      row=3,col=1,secondary_y=False)
        x0,x1=df.index[0],df.index[-1]
        fig.add_shape(type="line",x0=x0,x1=x1,y0=0,y1=0,
                       line=dict(color="rgba(150,150,150,.3)",dash="dot",width=1),row=3,col=1)

    # ── Row 4: ADX (primary) ──
    if "adx" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df.adx,name="ADX",
                                  line=dict(color="#ba68c8",width=1.5)),
                      row=4,col=1,secondary_y=False)
        x0,x1=df.index[0],df.index[-1]
        fig.add_shape(type="line",x0=x0,x1=x1,y0=25,y1=25,
                       line=dict(color="rgba(255,202,40,.4)",dash="dash",width=1),row=4,col=1)

    # ── Row 4: Volume (secondary) ──
    if vol:
        vc=["#26a69a" if c>=o else "#ef5350" for c,o in zip(df.Close,df.Open)]
        fig.add_trace(go.Bar(x=df.index,y=df.Volume,name="Vol",
                              marker_color=vc,opacity=.4),
                      row=4,col=1,secondary_y=True)

    fig.update_layout(template="plotly_dark",height=920,
                       showlegend=True,
                       legend=dict(orientation="h",y=1.01,x=0,font_size=10),
                       xaxis_rangeslider_visible=False,
                       plot_bgcolor="#0d1117",paper_bgcolor="#0d1117",
                       margin=dict(l=0,r=0,t=28,b=0))
    return fig

def eq_chart(eqc,cap):
    fig=go.Figure()
    fig.add_trace(go.Scatter(y=eqc,mode="lines",name="Equity",fill="tozeroy",
                              fillcolor="rgba(0,230,118,.07)",
                              line=dict(color="#00e676",width=2)))
    fig.add_shape(type="line",x0=0,x1=len(eqc)-1,y0=cap,y1=cap,
                   line=dict(color="rgba(200,200,200,.35)",dash="dash",width=1))
    fig.update_layout(template="plotly_dark",height=250,title="Equity Curve",
                       showlegend=False,plot_bgcolor="#0d1117",paper_bgcolor="#0d1117",
                       margin=dict(l=0,r=0,t=30,b=0))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL STATUS PANEL  (live tab)
# ─────────────────────────────────────────────────────────────────────────────
def signal_status(df,strategy,ltp):
    if df is None or len(df)<30: return
    r=df.iloc[-1]
    st.markdown('<div class="sig-panel">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Signal Status & Market Conditions")

    # ── Row 1: regime, rsi, ema, supertrend ──
    c1,c2,c3,c4=st.columns(4)
    adx_v=float(r.get("adx",0)) if not pd.isna(r.get("adx",np.nan)) else 0
    reg=str(r.get("regime","N/A"))
    reg_icon="🟢" if reg=="TREND" else "🔴" if reg=="CHOPPY" else "🟡"
    c1.metric("Regime", f"{reg_icon} {reg}", f"ADX {adx_v:.1f}")

    rsi_v=float(r.get("rsi",50)) if not pd.isna(r.get("rsi",np.nan)) else 50
    rsi_note="Overbought ⚠️" if rsi_v>70 else "Oversold ⚠️" if rsi_v<30 else "Neutral ✓"
    c2.metric("RSI", f"{rsi_v:.1f}", rsi_note)

    eb=bool(r.get("eb",False)); ebs=bool(r.get("ebs",False))
    ema_st="🟢 Bullish" if eb else "🔴 Bearish" if ebs else "🟡 Mixed"
    c3.metric("EMA Alignment", ema_st)

    st_dir=int(r.get("st_d",0)) if not pd.isna(r.get("st_d",np.nan)) else 0
    st_st="🟢 Bullish ↑" if st_dir==1 else "🔴 Bearish ↓"
    c4.metric("SuperTrend", st_st)

    # ── Row 2: MACD, BB position, Volume, Last Signal ──
    c5,c6,c7,c8=st.columns(4)
    mh=float(r.get("mhist",0)) if not pd.isna(r.get("mhist",np.nan)) else 0
    c5.metric("MACD Hist", f"{mh:.4f}", "🟢 Positive" if mh>0 else "🔴 Negative")

    if ltp and not pd.isna(r.get("bb_u",np.nan)):
        bu,bm,bl=float(r.bb_u),float(r.bb_m),float(r.bb_l)
        bpct=(ltp-bl)/(bu-bl)*100 if bu>bl else 50
        bb_note=f"Top zone ⚠️" if bpct>80 else f"Bottom zone ⚠️" if bpct<20 else f"Mid zone ✓"
        c6.metric("BB Position", f"{bpct:.0f}%", bb_note)

    vr=float(r.get("vol_r",1)) if not pd.isna(r.get("vol_r",np.nan)) else 1
    c7.metric("Volume Ratio", f"{vr:.2f}×", "🟢 High" if vr>1.5 else "🟡 Avg" if vr>0.8 else "🔴 Low")

    lst=STORE.get("last_signal_time"); lst_type=STORE.get("last_signal_type","—")
    if lst:
        mins=int((datetime.datetime.now()-lst).total_seconds()//60)
        c8.metric("Last Signal", lst_type, f"{mins} min ago")
    else:
        c8.metric("Last Signal","None","Waiting…")

    # ── What's needed for next signal ──
    st.markdown("---")
    needs=[]
    sig_type=STORE.get("live_signal","HOLD")

    if "Simple" in strategy:
        needs.append("✅ Simple strategy – enters immediately when started, exits via SL/Target")
    elif reg=="TREND" or adx_v>=20:
        if st_dir!=1: needs.append(f"⏳ SuperTrend flip to **Bullish** needed  (currently {'Bearish'})")
        if not eb:    needs.append("⏳ EMA alignment needed: **EMA9 > EMA21 > EMA50**")
        if rsi_v>75:  needs.append(f"⏳ RSI overbought at {rsi_v:.1f} — wait for pullback below 75")
        if rsi_v<45:  needs.append(f"⏳ RSI weak at {rsi_v:.1f} — wait for RSI above 45")
        if mh<0:      needs.append("⏳ MACD Histogram negative — wait for MACD crossover")
    else:
        if ltp and not pd.isna(r.get("bb_l",np.nan)):
            if ltp>float(r.bb_l)*1.003:
                needs.append(f"⏳ Price needs to touch **Lower BB** at {float(r.bb_l):.2f} (now {ltp:.2f})")
        if rsi_v>38:
            needs.append(f"⏳ RSI {rsi_v:.1f} needs to drop below **38** for oversold bounce entry")

    if sig_type in("BUY","SELL"):
        st.success(f"✅ **{sig_type} signal active!** Conditions met — watching for entry")
    elif needs:
        for n in needs: st.warning(n)
    else:
        st.info("🔍 Monitoring market — no signal yet. All conditions close to trigger.")

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("### 📊 Market")
        tn=st.selectbox("Asset",list(TICKERS.keys()))
        ticker=st.text_input("Ticker symbol",TICKERS[tn]) if tn=="Custom ✏️" else TICKERS[tn]
        if tn!="Custom ✏️": st.caption(f"`{ticker}`")
        c1,c2=st.columns(2)
        with c1: iv=st.selectbox("Timeframe",INTERVALS,index=4)
        with c2: pd_=st.selectbox("Period",PERIODS,index=4)

        st.markdown("### 🧠 Strategy")
        strat=st.selectbox("Strategy",STRATEGIES)
        DSCS={
            "🏆 Smart Algo (Recommended)":"Regime-aware: Trending→SuperTrend+EMA+MACD / Ranging→BB+RSI+LiqHunt",
            "⚡ Simple Buy":"Enters BUY immediately on start. Re-entry on EMA21 bounce. Pure buyer — use with SL/Target.",
            "⚡ Simple Sell":"Enters SELL/PE immediately on start. Re-entry on EMA21 breakdown.",
            "SuperTrend + EMA Cloud":"SuperTrend flip + EMA50 position filter.",
            "EMA Crossover + ADX":"9/21 EMA cross when ADX>20 (trending).",
            "MACD Momentum":"MACD crossovers with EMA trend filter.",
            "Bollinger Squeeze":"Squeeze release + BB boundary breakout.",
            "Heikin Ashi Reversal":"HA candle flip with EMA21 confirmation.",
            "Smart Money Concepts (SMC)":"Order block + liquidity hunt entries.",
            "Opening Range Breakout":"First-5-candle ORB (best for 1m–5m intraday).",
            "Donchian Breakout":"20-period channel breakout with ADX>20 filter.",
        }
        with st.expander("ℹ️ Strategy Info"):
            st.info(DSCS.get(strat,""))

        cap=st.number_input("Backtest Capital (₹)",10000,10000000,100000,10000)

        st.markdown("### 🎯 Risk Management")
        c1,c2=st.columns(2)
        with c1: sl_t=st.selectbox("SL Type",SL_TYPES)
        with c2: tg_t=st.selectbox("Target Type",TGT_TYPES)
        c3,c4=st.columns(2)
        with c3: sl_v=st.number_input("SL Value",0.1,500.,1.5,.1)
        with c4: tg_v=st.number_input("Target Value",0.1,500.,3.,.1)

        su="×ATR" if "ATR" in sl_t else "pts" if "Custom" in sl_t else "val"
        tu="×ATR" if "ATR" in tg_t else "×Risk" if "Ratio" in tg_t else "pts" if "Custom" in tg_t else ""
        st.caption(f"SL unit: **{su}** | Target unit: **{tu}**")

        st.markdown("### 📈 Indicator Filters")
        ui=st.checkbox("Enable Indicator Filters",value=False)
        ena={}
        if ui:
            with st.expander("Select Filters"):
                c1,c2=st.columns(2)
                with c1:
                    ena["rsi"]=st.checkbox("RSI",True); ena["macd"]=st.checkbox("MACD",True)
                    ena["adx"]=st.checkbox("ADX",True); ena["ema"]=st.checkbox("EMA Align",True)
                with c2:
                    ena["volume"]=st.checkbox("Volume",False); ena["bb"]=st.checkbox("BB Extreme",False)

        st.markdown("### 🏦 Dhan Broker")
        d_on=st.checkbox("Enable Dhan Broker",value=False)
        dcfg={"on":d_on}
        if d_on:
            with st.expander("🔑 Credentials",expanded=True):
                dcfg["cid"]=st.text_input("Client ID",type="password")
                dcfg["tok"]=st.text_input("Access Token",type="password")
            opts=st.checkbox("Options Trading",value=False); dcfg["opts"]=opts
            if opts:
                dcfg["o_exch"]=st.selectbox("F&O Exchange",["NSE_FNO","BSE_FNO"])
                dcfg["ce_id"]=st.text_input("CE Security ID","")
                dcfg["pe_id"]=st.text_input("PE Security ID","")
                dcfg["oqty"]=st.number_input("Lot Qty",1,10000,65)
                dcfg["e_ot"]=st.selectbox("Entry Order",["MARKET","LIMIT"])
                dcfg["x_ot"]=st.selectbox("Exit Order", ["MARKET","LIMIT"])
                st.success("BUY signal → CE Buy  |  SELL signal → PE Buy")
            else:
                dcfg["prod"]=st.selectbox("Product",["INTRADAY","DELIVERY"])
                dcfg["exch"]=st.selectbox("Exchange",["NSE","BSE"])
                dcfg["sid"]=st.text_input("Security ID","1594")
                dcfg["qty"]=st.number_input("Quantity",1,100000,1)
                dcfg["e_ot"]=st.selectbox("Entry Order",["LIMIT","MARKET"])
                dcfg["x_ot"]=st.selectbox("Exit Order", ["MARKET","LIMIT"])
                st.caption("LIMIT orders use current LTP as price.")

        return dict(ticker=ticker,tn=tn,iv=iv,pd=pd_,strat=strat,cap=cap,
                    sl_t=sl_t,sl_v=sl_v,tg_t=tg_t,tg_v=tg_v,ui=ui,ena=ena,dcfg=dcfg)

# ─────────────────────────────────────────────────────────────────────────────
# METRIC CARD HELPER
# ─────────────────────────────────────────────────────────────────────────────
def mbox(lbl,val,good=None):
    c="#00e676" if good is True else "#ff5252" if good is False else "#ffca28" if good is None else "#e0e0e0"
    return (f'<div class="mbox"><div class="mlbl">{lbl}</div>'
            f'<div class="mval" style="color:{c}">{val}</div></div>')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
def tab_bt(cfg):
    st.markdown("## 📊 Strategy Backtester")
    c1,c2,c3=st.columns([3,1,1])
    with c1: st.markdown(f"**{cfg['tn']}** `{cfg['ticker']}` | `{cfg['iv']}` | `{cfg['pd']}` | **{cfg['strat']}**")
    with c2: run=st.button("▶️ Run Backtest",type="primary",use_container_width=True)
    with c3: clr=st.button("🗑️ Clear",use_container_width=True)

    if clr:
        STORE.set("backtest_result",None); STORE.set("backtest_trades",[]); st.rerun()

    if run:
        with st.spinner("Fetching data & running backtest…"):
            df=fetch_data(cfg["ticker"],cfg["iv"],cfg["pd"])
            if df is None or df.empty: st.error("❌ No data — check ticker/period."); return
            if len(df)<65: st.warning(f"⚠️ Only {len(df)} candles — use a longer period."); return
            df=add_ind(df); sig=get_sigs(df,cfg["strat"],cfg["ui"],cfg["ena"])
            res=backtest(df,sig,cfg["sl_t"],cfg["sl_v"],cfg["tg_t"],cfg["tg_v"],cfg["cap"])
            STORE.set("backtest_result",res)
            STORE.set("backtest_trades",res["trades"].to_dict("records") if not res["trades"].empty else [])

    res=STORE.get("backtest_result")
    if res is None: st.info("👆 Press **Run Backtest** to start."); return

    m=res["metrics"]; tdf=res["trades"]
    st.markdown("### 📈 Performance")

    cols=st.columns(6)
    cards=[("Trades",str(m["n"]),None),("Win Rate",f"{m['wr']}%",m["wr"]>=50),
           ("Total PnL",f"₹{m['pnl']:,.0f}",m["pnl"]>=0),
           ("Profit Factor",str(m["pf"]),m["pf"]>=1.5),
           ("Max DD",f"{m['dd']}%",m["dd"]<=15),("Return",f"{m['ret']}%",m["ret"]>=0)]
    for col,(lbl,val,g) in zip(cols,cards):
        col.markdown(mbox(lbl,val,g),unsafe_allow_html=True)
    st.markdown("")
    cols2=st.columns(5)
    for col,(lbl,val) in zip(cols2,[("Avg Win",f"₹{m['aw']:,.0f}"),("Avg Loss",f"₹{m['al']:,.0f}"),
                                     ("Avg R:R",str(m["rr"])),("Expectancy",f"₹{m['exp']:,.0f}"),
                                     ("Sharpe",str(m["sh"]))]):
        col.markdown(mbox(lbl,val,"n"),unsafe_allow_html=True)
    st.markdown("")

    t1,t2,t3=st.tabs(["📉 Chart","💰 Equity Curve","📋 Trades"])
    with t1: st.plotly_chart(make_chart(res["df"],res["sigs"],tdf),use_container_width=True)
    with t2:
        st.plotly_chart(eq_chart(res["eq"],cfg["cap"]),use_container_width=True)
        ca,cb,cc=st.columns(3)
        ca.metric("Initial",f"₹{cfg['cap']:,.0f}")
        cb.metric("Final",f"₹{res['final']:,.0f}",delta=f"₹{res['final']-cfg['cap']:,.0f}")
        cc.metric("W/L",f"{m['wins']}/{m['loss']}")
    with t3:
        if not tdf.empty:
            def clr_row(v):
                try: return "background-color:#00e67620" if float(v)>0 else "background-color:#ff525220"
                except: return ""
            st.dataframe(tdf.style.map(clr_row,subset=["PnL"]),use_container_width=True,height=420)
            st.download_button("📥 CSV",tdf.to_csv(index=False),"bt_trades.csv","text/csv")
        else: st.info("No trades – adjust strategy or parameters.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: LIVE TRADING
# ─────────────────────────────────────────────────────────────────────────────
def tab_live(cfg):
    st.markdown("## ⚡ Live Trading")
    running=STORE.get("live_running",False)

    # ── Controls row ──
    c1,c2,c3=st.columns([4,1,1])
    with c1:
        dot="🟢" if running else "🔴"
        st.markdown(f"{dot} **{'LIVE — auto-refreshing every 3s' if running else 'STOPPED'}** &nbsp;|&nbsp; "
                    f"**{cfg['tn']}** `{cfg['ticker']}` | `{cfg['iv']}` | {cfg['strat']}")
    with c2:
        if not running:
            if st.button("▶️ Start Live",type="primary",use_container_width=True):
                STORE.upd(dict(live_running=True,live_log=[],error=None))
                t=threading.Thread(target=live_loop,daemon=True,
                                   args=(cfg["ticker"],cfg["iv"],cfg["pd"],cfg["strat"],
                                         cfg["sl_t"],cfg["sl_v"],cfg["tg_t"],cfg["tg_v"],
                                         cfg["ui"],cfg["ena"],cfg["dcfg"]))
                t.start(); STORE.set("live_thread",t)
                st.rerun()
    with c3:
        # Stop button always rendered (disabled when not running)
        stop_clicked=st.button("⏹️ Stop",type="secondary",use_container_width=True,
                                disabled=(not running))
        if stop_clicked:
            STORE.upd(dict(live_running=False)); STORE.clear_pos()
            st.warning("⏹️ Live trading stopped."); st.rerun()

    err=STORE.get("error")
    if err: st.error(f"⚠️ {err[:350]}")

    st.divider()

    # ── Signal status panel ──
    ldf=STORE.get("live_df"); ltp=STORE.get("live_ltp"); pnl=STORE.get("live_pnl",0.)
    sig=STORE.get("live_signal","HOLD"); pos=STORE.get("live_position")
    t_sl=STORE.get("trailing_sl"); t_tg=STORE.get("trailing_target")

    signal_status(ldf, cfg["strat"], ltp)
    st.markdown("")

    # ── Live metric tiles ──
    c1,c2,c3,c4=st.columns(4)
    c1.markdown(mbox("LTP", f"₹{ltp:,.2f}" if ltp else "—"),unsafe_allow_html=True)
    sc={"BUY":"#00e676","SELL":"#ff5252","HOLD":"#ffca28"}.get(sig,"#fff")
    si={"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}.get(sig,"")
    c2.markdown(f'<div class="mbox"><div class="mlbl">Signal</div>'
                f'<div class="mval" style="color:{sc}">{si} {sig}</div></div>',unsafe_allow_html=True)
    pc="#00e676" if pnl>=0 else "#ff5252"
    c3.markdown(mbox("Open PnL",f"₹{pnl:+,.2f}" if pnl!=0 else "—",pnl>=0 if pnl!=0 else None),unsafe_allow_html=True)
    pt=f"{pos['side']} @ ₹{pos['ep']:,.2f}" if pos else "Flat"
    pgc="#00e676" if pos and pos["side"]=="LONG" else "#ff5252" if pos else "#6b7280"
    c4.markdown(f'<div class="mbox"><div class="mlbl">Position</div>'
                f'<div class="mval" style="color:{pgc}">{pt}</div></div>',unsafe_allow_html=True)

    # ── Position detail ──
    if pos and ltp:
        sl_now=t_sl or pos["sl"]; tg_now=t_tg or pos["tg"]
        st.markdown("")
        ca,cb,cc,cd=st.columns(4)
        ca.metric("Entry",f"₹{pos['ep']:,.2f}")
        cb.metric("SL",f"₹{sl_now:,.2f}",delta=f"{ltp-sl_now:+.2f} from LTP")
        cc.metric("Target",f"₹{tg_now:,.2f}",delta=f"{tg_now-ltp:+.2f} to go")
        risk=abs(pos["ep"]-sl_now); rew=abs(tg_now-pos["ep"])
        cd.metric("R:R",f"{rew/risk:.2f}" if risk>0 else "—")

        prog_range=tg_now-sl_now if pos["side"]=="LONG" else sl_now-tg_now
        prog=(ltp-sl_now)/prog_range if pos["side"]=="LONG" and prog_range>0 \
              else (sl_now-ltp)/prog_range if prog_range>0 else 0
        st.progress(max(0.,min(1.,prog)),text=f"Trade progress: {max(0.,min(1.,prog))*100:.0f}%")

        if st.button("🚨 Manual Exit",type="secondary"):
            fpnl=(ltp-pos["ep"])*pos["qty"] if pos["side"]=="LONG" else (pos["ep"]-ltp)*pos["qty"]
            rec=dict(EntryTime=pos["et"],ExitTime=datetime.datetime.now(),
                     Side=pos["side"],EntryPrice=round(pos["ep"],2),ExitPrice=round(ltp,2),
                     SL=round(pos["isl"],2),Target=round(pos["itg"],2),Qty=pos["qty"],
                     PnL=round(fpnl,2),PnLpct=round(fpnl/(pos["ep"]*pos["qty"])*100,2),
                     ExitReason="MANUAL",Source="LIVE")
            h=STORE.get("trade_history",[]); h.append(rec); STORE.set("trade_history",h)
            if cfg["dcfg"].get("on"):
                st.info(f"Dhan: {dhan_exit(pos['sig'],ltp,cfg['dcfg'])}")
            STORE.clear_pos()
            st.success(f"Exited @ ₹{ltp:,.2f}  PnL: ₹{fpnl:+,.2f}"); st.rerun()

    st.divider()

    # ── Live chart ──
    if ldf is not None:
        with st.expander("📈 Live Chart",expanded=True):
            hlines=[]
            if pos:
                sl_now=t_sl or pos["sl"]; tg_now=t_tg or pos["tg"]
                hlines=[
                    {"y":pos["ep"],"c":"#ffca28","d":"solid","lbl":f"Entry {pos['ep']:.2f}"},
                    {"y":sl_now,   "c":"#ff5252","d":"dash", "lbl":f"SL {sl_now:.2f}"},
                    {"y":tg_now,   "c":"#00e676","d":"dash", "lbl":f"Tgt {tg_now:.2f}"},
                ]
            st.plotly_chart(make_chart(ldf,hlines=hlines),use_container_width=True)

    # ── Activity Log ──
    with st.expander("📋 Activity Log"):
        lg=STORE.get("live_log",[])
        if lg: st.code("\n".join(reversed(lg[-50:])),language=None)
        else:  st.caption("No activity yet.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: HISTORY
# ─────────────────────────────────────────────────────────────────────────────
def tab_hist():
    st.markdown("## 📚 Trade History")
    lh=STORE.get("trade_history",[]); bh=STORE.get("backtest_trades",[])
    t1,t2,t3=st.tabs(["🔴 Live Trades","📊 Backtest Trades","📈 Combined Stats"])

    with t1:
        if lh:
            st.dataframe(pd.DataFrame(lh),use_container_width=True,height=400)
            w=sum(1 for t in lh if t.get("PnL",0)>0); tp=sum(t.get("PnL",0) for t in lh)
            ca,cb,cc=st.columns(3)
            ca.metric("Trades",len(lh)); cb.metric("Win Rate",f"{w/len(lh)*100:.1f}%")
            cc.metric("Total PnL",f"₹{tp:+,.2f}")
            if st.button("🗑️ Clear Live History"): STORE.set("trade_history",[]); st.rerun()
        else: st.info("No live trades yet.")

    with t2:
        if bh:
            bdf=pd.DataFrame(bh); st.dataframe(bdf,use_container_width=True,height=400)
            st.download_button("📥 CSV",bdf.to_csv(index=False),"bt_trades.csv","text/csv")
        else: st.info("Run a backtest first.")

    with t3:
        all_t=[{**t,"Source":"LIVE"} for t in lh]+[{**t,"Source":"BACKTEST"} for t in bh]
        if not all_t: st.info("No trades yet."); return
        adf=pd.DataFrame(all_t)
        if "PnL" in adf.columns:
            tp=adf.PnL.sum(); w=len(adf[adf.PnL>0]); tot=len(adf)
            ca,cb,cc,cd=st.columns(4)
            ca.metric("Total",tot); cb.metric("Win Rate",f"{w/tot*100:.1f}%")
            cc.metric("Total PnL",f"₹{tp:+,.2f}"); cd.metric("W/L",f"{w}/{tot-w}")
            fig=go.Figure()
            cols=["#00e676" if p>0 else "#ff5252" for p in adf.PnL]
            fig.add_trace(go.Bar(y=adf.PnL,marker_color=cols,name="PnL/Trade"))
            fig.update_layout(template="plotly_dark",height=280,title="Per-Trade PnL",
                               plot_bgcolor="#0d1117",paper_bgcolor="#0d1117",
                               margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig,use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown(
        '<div style="background:linear-gradient(90deg,#0a1628,#0f2952);'
        'padding:12px 20px;border-radius:10px;margin-bottom:14px;">'
        '<span style="color:#00e676;font-size:21px;font-weight:700;">🧠 Smart Investing Platform</span>'
        '<span style="color:#6b7280;font-size:12px;margin-left:14px;">'
        'NSE · BSE · Crypto · Forex · Commodities | v3.0</span></div>',
        unsafe_allow_html=True)

    cfg=sidebar()
    tb1,tb2,tb3=st.tabs(["📊 Backtesting","⚡ Live Trading","📚 Trade History"])
    with tb1: tab_bt(cfg)
    with tb2: tab_live(cfg)
    with tb3: tab_hist()

    # ── Auto-refresh when live trading is active ──
    # Placed AFTER tabs so it doesn't interfere with Stop button clicks
    if STORE.get("live_running",False):
        time.sleep(3)
        st.rerun()

if __name__=="__main__":
    main()
