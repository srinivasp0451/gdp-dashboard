"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          ALGO TRADING PLATFORM  v2  —  Streamlit + yfinance                ║
║  Run:     streamlit run algo_trading.py                                     ║
║  Install: pip install "streamlit>=1.33" yfinance pandas numpy plotly       ║
║  Live tab uses @st.fragment (Streamlit>=1.33) for flicker-free updates.    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
from itertools import product as itertools_product
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AlgoTrader Pro", layout="wide", initial_sidebar_state="expanded")
_HAS_FRAGMENT = hasattr(st, "fragment")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
TICKER_MAP = {
    "Nifty 50":"^NSEI","Bank Nifty":"^NSEBANK","Nifty IT":"^CNXIT","Sensex":"^BSESN",
    "BTC/USD":"BTC-USD","ETH/USD":"ETH-USD","USD/INR":"USDINR=X","Gold":"GC=F",
    "Silver":"SI=F","EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X","USD/JPY":"JPYUSD=X",
    "Crude Oil":"CL=F","Custom":"CUSTOM",
}
TIMEFRAMES = ["1m","5m","15m","30m","1h","4h","1d","1wk"]
PERIODS    = ["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","max"]
STRATEGIES = [
    "EMA Crossover","RSI Overbought/Oversold","Simple Buy","Simple Sell",
    "Price Threshold Cross","Bollinger Bands","RSI Divergence",
    "MACD Crossover","Supertrend","ADX + DI Crossover","Stochastic Oscillator",
    "VWAP Deviation","Ichimoku Cloud","BB + RSI Mean Reversion",
    "Donchian Breakout","Triple EMA Trend","Heikin Ashi EMA",
    "Volume Price Trend (VPT)","Keltner Channel Breakout","Williams %R Reversal",
    "Swing Trend + Pullback","Custom Strategy",
]
SL_TYPES = [
    "Custom Points","Trailing SL (Points)","Trailing Prev Candle Low/High",
    "Trailing Curr Candle Low/High","Trailing Prev Swing Low/High",
    "Trailing Curr Swing Low/High","Cost to Cost (Breakeven)",
    "EMA Reverse Crossover","ATR Based",
]
TARGET_TYPES = [
    "Custom Points","Trailing Target (Display Only)","Trailing Prev Candle High/Low",
    "Trailing Curr Candle High/Low","Trailing Prev Swing High/Low",
    "Trailing Curr Swing High/Low","ATR Based","Risk/Reward Based",
]
YF_IV = {"1m":"1m","5m":"5m","15m":"15m","30m":"30m","1h":"1h","4h":"1h","1d":"1d","1wk":"1wk"}

# ── DHAN PLACEHOLDER ──────────────────────────────────────────────────────────
# from dhanhq import dhanhq
# dhan = dhanhq("CLIENT_ID","ACCESS_TOKEN")
# IS_OPTIONS=False; LOT_SIZE=50; PRODUCT_TYPE="INTRADAY"
# def dhan_place_order(sym,direction,qty):
#   # OPTIONS (buyer only): BUY->CE Buy, SELL->PE Buy
#   # STOCKS (buyer+seller): BUY->Buy, SELL->Short
#   if IS_OPTIONS:
#     opt="CE" if direction=="BUY" else "PE"
#     sid=get_atm_option_security_id(sym,opt); txn=dhan.BUY
#   else:
#     sid=sym; txn=dhan.BUY if direction=="BUY" else dhan.SELL
#   return dhan.place_order(security_id=sid,exchange_segment=dhan.NSE,
#     transaction_type=txn,quantity=qty,order_type=dhan.MARKET,product_type=PRODUCT_TYPE,price=0)
# def dhan_exit_order(sym,direction,qty):
#   txn=dhan.SELL if direction=="BUY" else dhan.BUY
#   return dhan.place_order(security_id=sym,exchange_segment=dhan.NSE,
#     transaction_type=txn,quantity=qty,order_type=dhan.MARKET,product_type=PRODUCT_TYPE,price=0)
# def get_atm_option_security_id(underlying,option_type):
#   raise NotImplementedError("Lookup ATM strike from Dhan instrument CSV")

# ── DATA ──────────────────────────────────────────────────────────────────────
def _flatten(df):
    if isinstance(df.columns,pd.MultiIndex):
        df.columns=[str(c[0]).strip().title() if isinstance(c,tuple) else str(c).strip().title() for c in df.columns]
    else:
        df.columns=[str(c).strip().title() for c in df.columns]
    keep=[c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df=df[keep].copy(); df.dropna(subset=["Open","High","Low","Close"],inplace=True)
    df.index=pd.to_datetime(df.index); return df

def _r4h(df):
    agg={"Open":"first","High":"max","Low":"min","Close":"last"}
    if "Volume" in df.columns: agg["Volume"]="sum"
    return df.resample("4h").agg(agg).dropna()

@st.cache_data(ttl=60)
def fetch_data(ticker,period,interval):
    try:
        time.sleep(1.5)
        raw=yf.download(ticker,period=period,interval=YF_IV.get(interval,interval),progress=False,auto_adjust=True)
        if raw.empty: return pd.DataFrame()
        df=_flatten(raw)
        return _r4h(df) if interval=="4h" and not df.empty else df
    except Exception as e: st.error(f"Fetch error: {e}"); return pd.DataFrame()

def fetch_live(ticker,interval):
    lp="1d" if interval in("1m","5m","15m","30m") else "5d"
    try:
        time.sleep(1.5)
        raw=yf.download(ticker,period=lp,interval=YF_IV.get(interval,interval),progress=False,auto_adjust=True)
        if raw.empty: return pd.DataFrame()
        df=_flatten(raw)
        return _r4h(df) if interval=="4h" and not df.empty else df
    except Exception as e: st.warning(f"Live fetch: {e}"); return pd.DataFrame()

# ── INDICATORS ────────────────────────────────────────────────────────────────
def ema(s,p):  return s.ewm(span=p,adjust=False).mean()
def sma(s,p):  return s.rolling(p).mean()
def rsi(s,p=14):
    d=s.diff(); g=d.clip(lower=0).ewm(alpha=1/p,adjust=False).mean()
    l=(-d).clip(lower=0).ewm(alpha=1/p,adjust=False).mean()
    return 100-100/(1+g/l.replace(0,np.nan))
def atr(df,p=14):
    hl=df["High"]-df["Low"]; hc=(df["High"]-df["Close"].shift()).abs(); lc=(df["Low"]-df["Close"].shift()).abs()
    return pd.concat([hl,hc,lc],axis=1).max(axis=1).ewm(alpha=1/p,adjust=False).mean()
def bollinger(s,p=20,k=2.0): m=sma(s,p); d=s.rolling(p).std(); return m-k*d,m,m+k*d
def macd(s,f=12,sl=26,sig=9): ml=ema(s,f)-ema(s,sl); return ml,ema(ml,sig),ml-ema(ml,sig)
def stoch(df,k=14,d=3):
    lo=df["Low"].rolling(k).min(); hi=df["High"].rolling(k).max()
    pk=100*(df["Close"]-lo)/(hi-lo).replace(0,np.nan); return pk,sma(pk,d)
def vwap_calc(df):
    tp=(df["High"]+df["Low"]+df["Close"])/3; vol=df.get("Volume",pd.Series(1,index=df.index)).replace(0,np.nan)
    return (tp*vol).cumsum()/vol.cumsum()
def supertrend(df,p=7,m=3.0):
    _a=atr(df,p); hl2=(df["High"]+df["Low"])/2; bu=hl2+m*_a; bl=hl2-m*_a
    fu=bu.copy(); fl=bl.copy(); di=pd.Series(1,index=df.index)
    for i in range(1,len(df)):
        fu.iloc[i]=bu.iloc[i] if bu.iloc[i]<fu.iloc[i-1] or df["Close"].iloc[i-1]>fu.iloc[i-1] else fu.iloc[i-1]
        fl.iloc[i]=bl.iloc[i] if bl.iloc[i]>fl.iloc[i-1] or df["Close"].iloc[i-1]<fl.iloc[i-1] else fl.iloc[i-1]
        if df["Close"].iloc[i]>fu.iloc[i-1]: di.iloc[i]=1
        elif df["Close"].iloc[i]<fl.iloc[i-1]: di.iloc[i]=-1
        else: di.iloc[i]=di.iloc[i-1]
    return pd.Series(np.where(di==1,fl.values,fu.values),index=df.index),di
def adx_di(df,p=14):
    _a=atr(df,p); up=df["High"].diff(); dn=-df["Low"].diff()
    pdm=pd.Series(np.where((up>dn)&(up>0),up,0.),index=df.index)
    ndm=pd.Series(np.where((dn>up)&(dn>0),dn,0.),index=df.index)
    pdi=100*pdm.ewm(alpha=1/p,adjust=False).mean()/_a.replace(0,np.nan)
    ndi=100*ndm.ewm(alpha=1/p,adjust=False).mean()/_a.replace(0,np.nan)
    dx=100*(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)
    return dx.ewm(alpha=1/p,adjust=False).mean(),pdi,ndi
def ichimoku(df,te=9,ki=26,sb=52):
    def mid(h,l,p): return (h.rolling(p).max()+l.rolling(p).min())/2
    t=mid(df["High"],df["Low"],te); k=mid(df["High"],df["Low"],ki)
    return t,k,((t+k)/2).shift(ki),mid(df["High"],df["Low"],sb).shift(ki)
def donchian(df,p=20):
    u=df["High"].rolling(p).max(); l=df["Low"].rolling(p).min(); return u,(u+l)/2,l
def heikin_ashi(df):
    ha=pd.DataFrame(index=df.index)
    ha["Close"]=(df["Open"]+df["High"]+df["Low"]+df["Close"])/4
    ha["Open"]=(df["Open"].shift(1)+df["Close"].shift(1))/2
    ha["Open"].iloc[0]=(df["Open"].iloc[0]+df["Close"].iloc[0])/2
    ha["High"]=pd.concat([df["High"],ha["Open"],ha["Close"]],axis=1).max(axis=1)
    ha["Low"]=pd.concat([df["Low"],ha["Open"],ha["Close"]],axis=1).min(axis=1)
    return ha
def vpt_calc(df):
    vol=df.get("Volume",pd.Series(1,index=df.index)); return (df["Close"].pct_change()*vol).cumsum()
def keltner(df,ep=20,ap=10,m=2.0): mid=ema(df["Close"],ep); _a=atr(df,ap); return mid-m*_a,mid,mid+m*_a
def williams_r(df,p=14):
    hi=df["High"].rolling(p).max(); lo=df["Low"].rolling(p).min()
    return -100*(hi-df["Close"])/(hi-lo).replace(0,np.nan)
def _cup(a,b): return (a>b)&(a.shift(1)<=b.shift(1))
def _cdn(a,b): return (a<b)&(a.shift(1)>=b.shift(1))
def _cs(v,idx): return pd.Series(float(v),index=idx)
# ── STRATEGY SIGNAL GENERATORS ────────────────────────────────────────────────
def sig_ema_cross(df,fast=9,slow=15,**_):
    fe=ema(df["Close"],fast); se=ema(df["Close"],slow); s=pd.Series(0,index=df.index)
    s[_cup(fe,se)]=1; s[_cdn(fe,se)]=-1; return s,{"EMA_fast":fe,"EMA_slow":se}

def sig_rsi_osob(df,period=14,ob=70,os_=30,**_):
    """RSI OR condition: BUY when RSI crosses ABOVE oversold | SELL when RSI crosses BELOW overbought"""
    r=rsi(df["Close"],period); s=pd.Series(0,index=df.index)
    s[_cup(r,_cs(os_,df.index))]=1; s[_cdn(r,_cs(ob,df.index))]=-1
    return s,{"RSI":r,"RSI_OB":_cs(ob,df.index),"RSI_OS":_cs(os_,df.index)}

def sig_simple_buy(df,**_): s=pd.Series(0,index=df.index); s.iloc[:-1]=1; return s,{}
def sig_simple_sell(df,**_): s=pd.Series(0,index=df.index); s.iloc[:-1]=-1; return s,{}

def sig_price_thresh(df,threshold=0.,**_):
    th=_cs(threshold,df.index); s=pd.Series(0,index=df.index)
    s[_cup(df["Close"],th)]=1; s[_cdn(df["Close"],th)]=-1; return s,{"Threshold":th}

def sig_bb(df,period=20,std=2.0,**_):
    lo,mid,hi=bollinger(df["Close"],period,std); s=pd.Series(0,index=df.index)
    s[_cup(df["Close"],lo)]=1; s[_cdn(df["Close"],hi)]=-1
    return s,{"BB_upper":hi,"BB_mid":mid,"BB_lower":lo}

def sig_rsi_div(df,period=14,lookback=5,**_):
    r=rsi(df["Close"],period); s=pd.Series(0,index=df.index)
    for i in range(lookback,len(df)):
        pc=df["Close"].iloc[i-lookback:i]; pr=r.iloc[i-lookback:i]
        if df["Close"].iloc[i]<pc.min() and r.iloc[i]>pr.min(): s.iloc[i]=1
        elif df["Close"].iloc[i]>pc.max() and r.iloc[i]<pr.max(): s.iloc[i]=-1
    return s,{"RSI":r}

def sig_macd(df,fast=12,slow=26,signal=9,**_):
    ml,sl,_=macd(df["Close"],fast,slow,signal); s=pd.Series(0,index=df.index)
    s[_cup(ml,sl)]=1; s[_cdn(ml,sl)]=-1; return s,{"MACD":ml,"MACD_Signal":sl}

def sig_supertrend(df,period=7,multiplier=3.0,**_):
    line,di=supertrend(df,period,multiplier); s=pd.Series(0,index=df.index)
    s[(di==1)&(di.shift(1)==-1)]=1; s[(di==-1)&(di.shift(1)==1)]=-1; return s,{"Supertrend":line}

def sig_adx(df,period=14,adx_thresh=25,**_):
    _adx,pdi,ndi=adx_di(df,period); at=_cs(adx_thresh,df.index); s=pd.Series(0,index=df.index)
    s[_cup(pdi,ndi)&(_adx>at)]=1; s[_cdn(pdi,ndi)&(_adx>at)]=-1
    return s,{"ADX":_adx,"+DI":pdi,"-DI":ndi}

def sig_stoch(df,k=14,d=3,ob=80,os_=20,**_):
    pk,pd_=stoch(df,k,d); s=pd.Series(0,index=df.index)
    s[_cup(pk,pd_)&(pk<ob)]=1; s[_cdn(pk,pd_)&(pk>os_)]=-1; return s,{"Stoch_K":pk,"Stoch_D":pd_}

def sig_vwap_dev(df,dev_pct=1.0,**_):
    vw=vwap_calc(df); d=dev_pct/100; hi_b=vw*(1+d); lo_b=vw*(1-d); s=pd.Series(0,index=df.index)
    s[_cdn(df["Close"],lo_b)]=1; s[_cup(df["Close"],hi_b)]=-1
    return s,{"VWAP":vw,"VWAP_hi":hi_b,"VWAP_lo":lo_b}

def sig_ichimoku(df,tenkan=9,kijun=26,**_):
    t,k,sa,sb=ichimoku(df,tenkan,kijun)
    ct=pd.concat([sa,sb],axis=1).max(axis=1); cb=pd.concat([sa,sb],axis=1).min(axis=1)
    s=pd.Series(0,index=df.index); s[_cup(df["Close"],ct)]=1; s[_cdn(df["Close"],cb)]=-1
    return s,{"Tenkan":t,"Kijun":k,"Senkou_A":sa,"Senkou_B":sb}

def sig_bb_rsi(df,bb_period=20,bb_std=2.0,rsi_period=14,rsi_os=35,rsi_ob=65,**_):
    lo,_,hi=bollinger(df["Close"],bb_period,bb_std); r=rsi(df["Close"],rsi_period)
    s=pd.Series(0,index=df.index); s[(df["Close"]<lo)&(r<rsi_os)]=1; s[(df["Close"]>hi)&(r>rsi_ob)]=-1
    return s,{"BB_upper":hi,"BB_lower":lo,"RSI":r}

def sig_donchian(df,period=20,**_):
    hi,_,lo=donchian(df,period); s=pd.Series(0,index=df.index)
    s[df["Close"]>hi.shift(1)]=1; s[df["Close"]<lo.shift(1)]=-1; return s,{"Don_upper":hi,"Don_lower":lo}

def sig_triple_ema(df,f=9,m=21,s_=50,**_):
    e1=ema(df["Close"],f); e2=ema(df["Close"],m); e3=ema(df["Close"],s_)
    bull=(e1>e2)&(e2>e3); bear=(e1<e2)&(e2<e3); s=pd.Series(0,index=df.index)
    s[bull&~bull.shift(1).fillna(False)]=1; s[bear&~bear.shift(1).fillna(False)]=-1
    return s,{"EMA_fast":e1,"EMA_mid":e2,"EMA_slow":e3}

def sig_ha_ema(df,ema_period=20,**_):
    ha=heikin_ashi(df); e=ema(df["Close"],ema_period); bull=ha["Close"]>ha["Open"]
    s=pd.Series(0,index=df.index)
    s[bull&~bull.shift(1).fillna(False)&(df["Close"]>e)]=1
    s[~bull&bull.shift(1).fillna(True)&(df["Close"]<e)]=-1; return s,{"EMA":e}

def sig_vpt(df,vpt_ema_period=14,**_):
    v=vpt_calc(df); vs=ema(v,vpt_ema_period); s=pd.Series(0,index=df.index)
    s[_cup(v,vs)]=1; s[_cdn(v,vs)]=-1; return s,{}

def sig_keltner(df,ema_p=20,atr_p=10,mult=2.0,**_):
    lo,mid,hi=keltner(df,ema_p,atr_p,mult); s=pd.Series(0,index=df.index)
    s[_cup(df["Close"],hi)]=1; s[_cdn(df["Close"],lo)]=-1; return s,{"KC_upper":hi,"KC_mid":mid,"KC_lower":lo}

def sig_williams(df,period=14,ob=-20,os_=-80,**_):
    wr=williams_r(df,period); s=pd.Series(0,index=df.index)
    s[_cup(wr,_cs(os_,df.index))]=1; s[_cdn(wr,_cs(ob,df.index))]=-1; return s,{"Williams_%R":wr}

def sig_swing_pullback(df,trend_ema=50,entry_ema=9,rsi_period=14,
                        rsi_bull_min=40,rsi_bull_max=65,rsi_bear_min=35,rsi_bear_max=60,vol_mult=1.2,**_):
    """Swing Trend+Pullback (swingtrading74): price above Trend EMA, pulls back to Entry EMA,
    RSI in zone, candle closes in trend dir, volume surge."""
    te=ema(df["Close"],trend_ema); ee=ema(df["Close"],entry_ema); r=rsi(df["Close"],rsi_period)
    vol=df.get("Volume",pd.Series(1,index=df.index)); avg_vol=vol.rolling(20).mean().replace(0,np.nan)
    bull_touch=(df["Low"]<=ee)&(df["Close"]>ee)&(df["Close"]>df["Open"])
    bear_touch=(df["High"]>=ee)&(df["Close"]<ee)&(df["Close"]<df["Open"])
    vol_ok=vol>avg_vol*vol_mult; s=pd.Series(0,index=df.index)
    s[(df["Close"]>te)&bull_touch&(r>=rsi_bull_min)&(r<=rsi_bull_max)&vol_ok]=1
    s[(df["Close"]<te)&bear_touch&(r>=rsi_bear_min)&(r<=rsi_bear_max)&vol_ok]=-1
    return s,{"EMA_trend":te,"EMA_entry":ee,"RSI":r}

def sig_custom(df,**_):
    """CUSTOM — replace with your logic. Return (signals Series[1/-1/0], indicators dict).
    Example: fe=ema(df['Close'],9); se=ema(df['Close'],21); r=rsi(df['Close'],14)
             s=pd.Series(0,index=df.index); s[_cup(fe,se)&(r<60)]=1; s[_cdn(fe,se)&(r>40)]=-1
             return s,{'EMA9':fe,'EMA21':se}"""
    return pd.Series(0,index=df.index),{}

STRATEGY_FN={
    "EMA Crossover":sig_ema_cross,"RSI Overbought/Oversold":sig_rsi_osob,
    "Simple Buy":sig_simple_buy,"Simple Sell":sig_simple_sell,
    "Price Threshold Cross":sig_price_thresh,"Bollinger Bands":sig_bb,
    "RSI Divergence":sig_rsi_div,"MACD Crossover":sig_macd,"Supertrend":sig_supertrend,
    "ADX + DI Crossover":sig_adx,"Stochastic Oscillator":sig_stoch,
    "VWAP Deviation":sig_vwap_dev,"Ichimoku Cloud":sig_ichimoku,
    "BB + RSI Mean Reversion":sig_bb_rsi,"Donchian Breakout":sig_donchian,
    "Triple EMA Trend":sig_triple_ema,"Heikin Ashi EMA":sig_ha_ema,
    "Volume Price Trend (VPT)":sig_vpt,"Keltner Channel Breakout":sig_keltner,
    "Williams %R Reversal":sig_williams,"Swing Trend + Pullback":sig_swing_pullback,
    "Custom Strategy":sig_custom,
}
# ── SL / TARGET ENGINE ────────────────────────────────────────────────────────
def _sw_lo(df,idx,lb=5): return float(df["Low"].iloc[max(0,idx-lb):idx].min()) if idx>0 else float(df["Low"].iloc[0])
def _sw_hi(df,idx,lb=5): return float(df["High"].iloc[max(0,idx-lb):idx].max()) if idx>0 else float(df["High"].iloc[0])
def _atr_at(df,idx,p=14):
    v=atr(df,p).iloc[idx]; return float(v) if not np.isnan(v) else 10.0

def init_sl(df,idx,entry,direction,sl_type,sl_pts,params):
    lb=params.get("swing_lookback",5); am=params.get("atr_mult_sl",1.5); av=_atr_at(df,idx); d=direction
    if sl_type=="Custom Points": return entry-d*sl_pts
    if sl_type=="Trailing SL (Points)": return entry-d*sl_pts
    if sl_type=="Trailing Prev Candle Low/High":
        return float(df["Low"].iloc[max(0,idx-1)]) if d==1 else float(df["High"].iloc[max(0,idx-1)])
    if sl_type=="Trailing Curr Candle Low/High":
        return float(df["Low"].iloc[idx]) if d==1 else float(df["High"].iloc[idx])
    if sl_type in("Trailing Prev Swing Low/High","Trailing Curr Swing Low/High"):
        return _sw_lo(df,idx,lb) if d==1 else _sw_hi(df,idx,lb)
    if sl_type=="Cost to Cost (Breakeven)": return entry-d*sl_pts
    if sl_type=="EMA Reverse Crossover": return entry-d*sl_pts
    if sl_type=="ATR Based": return entry-d*am*av
    return entry-d*sl_pts

def init_tgt(df,idx,entry,direction,tgt_type,tgt_pts,sl,params):
    lb=params.get("swing_lookback",5); am=params.get("atr_mult_tgt",2.0); rr=params.get("rr_ratio",2.0)
    av=_atr_at(df,idx); d=direction
    if tgt_type=="Custom Points": return entry+d*tgt_pts
    if tgt_type=="Trailing Target (Display Only)": return entry+d*tgt_pts
    if tgt_type in("Trailing Prev Candle High/Low","Trailing Curr Candle High/Low"):
        return float(df["High"].iloc[idx]) if d==1 else float(df["Low"].iloc[idx])
    if tgt_type in("Trailing Prev Swing High/Low","Trailing Curr Swing High/Low"):
        return _sw_hi(df,idx,lb) if d==1 else _sw_lo(df,idx,lb)
    if tgt_type=="ATR Based": return entry+d*am*av
    if tgt_type=="Risk/Reward Based": return entry+d*rr*abs(entry-sl)
    return entry+d*tgt_pts

def update_sl(df,j,entry,direction,sl_type,sl_pts,cur_sl,params):
    lb=params.get("swing_lookback",5); d=direction
    bh=float(df["High"].iloc[j]); bl=float(df["Low"].iloc[j])
    if sl_type=="Custom Points": return cur_sl
    if sl_type=="Trailing SL (Points)":
        return max(cur_sl,bh-sl_pts) if d==1 else min(cur_sl,bl+sl_pts)
    if sl_type=="Trailing Prev Candle Low/High":
        if j<1: return cur_sl
        return max(cur_sl,float(df["Low"].iloc[j-1])) if d==1 else min(cur_sl,float(df["High"].iloc[j-1]))
    if sl_type=="Trailing Curr Candle Low/High":
        return max(cur_sl,bl) if d==1 else min(cur_sl,bh)
    if sl_type=="Trailing Prev Swing Low/High":
        return max(cur_sl,_sw_lo(df,j,lb)) if d==1 else min(cur_sl,_sw_hi(df,j,lb))
    if sl_type=="Trailing Curr Swing Low/High":
        return max(cur_sl,_sw_lo(df,j+1,lb)) if d==1 else min(cur_sl,_sw_hi(df,j+1,lb))
    if sl_type=="Cost to Cost (Breakeven)":
        sl_dist=abs(entry-cur_sl)
        if d==1 and bh>=entry+sl_dist: return max(cur_sl,entry)
        if d==-1 and bl<=entry-sl_dist: return min(cur_sl,entry)
        return cur_sl
    return cur_sl

def update_tgt(df,j,direction,tgt_type,tgt_pts,cur_tgt,params):
    lb=params.get("swing_lookback",5); d=direction
    bh=float(df["High"].iloc[j]); bl=float(df["Low"].iloc[j])
    if tgt_type=="Custom Points": return cur_tgt,True
    if tgt_type=="Trailing Target (Display Only)":
        return (max(cur_tgt,bh) if d==1 else min(cur_tgt,bl)),False
    if tgt_type in("Trailing Prev Candle High/Low","Trailing Curr Candle High/Low"):
        return (max(cur_tgt,bh) if d==1 else min(cur_tgt,bl)),True
    if tgt_type in("Trailing Prev Swing High/Low","Trailing Curr Swing High/Low"):
        return (max(cur_tgt,_sw_hi(df,j,lb)) if d==1 else min(cur_tgt,_sw_lo(df,j,lb))),True
    if tgt_type in("ATR Based","Risk/Reward Based"): return cur_tgt,True
    return cur_tgt,True

# ── BACKTEST ENGINE ────────────────────────────────────────────────────────────
def run_backtest(df,strategy,params,sl_type,sl_pts,tgt_type,tgt_pts):
    """
    Bar i   → signal on CLOSE (bar fully closed)
    Bar i+1 → entry at OPEN; SL & Target set
    Bar i+1+ → SL checked FIRST (conservative), then Target
    Both breach same candle → SL wins
    Returns (trades, indicators, audit_trail)
    """
    fn=STRATEGY_FN.get(strategy,sig_custom)
    try: sigs,indics=fn(df,**params)
    except: return [],[],[]
    n=len(df); trades=[]; audit_trail=[]; i=0
    while i<n-1:
        sig=int(sigs.iloc[i])
        if sig==0: i+=1; continue
        direction=sig; entry_idx=i+1
        if entry_idx>=n: break
        entry=float(df["Open"].iloc[entry_idx])
        sl=init_sl(df,entry_idx,entry,direction,sl_type,sl_pts,params)
        tgt=init_tgt(df,entry_idx,entry,direction,tgt_type,tgt_pts,sl,params)
        disp_tgt=tgt; highest=entry; lowest=entry; trade_n=len(trades)
        exited=False; exit_bar=None; exit_px=None; exit_why=None
        for j in range(entry_idx,n):
            bh=float(df["High"].iloc[j]); bl=float(df["Low"].iloc[j])
            highest=max(highest,bh); lowest=min(lowest,bl)
            sl=update_sl(df,j,entry,direction,sl_type,sl_pts,sl,params)
            disp_tgt,tf=update_tgt(df,j,direction,tgt_type,tgt_pts,disp_tgt,params)
            if tf: tgt=disp_tgt
            if sl_type=="EMA Reverse Crossover":
                rev=int(sigs.iloc[j])
                if rev!=0 and rev!=direction:
                    exit_bar,exit_px,exit_why=j,float(df["Open"].iloc[j]),"EMA Reverse Crossover"; exited=True
            if not exited:
                sl_hit=(direction==1 and bl<=sl) or (direction==-1 and bh>=sl)
                tgt_hit=tf and ((direction==1 and bh>=tgt) or (direction==-1 and bl<=tgt))
                if sl_hit: exit_bar,exit_px,exit_why=j,sl,"SL Hit"; exited=True
                elif tgt_hit: exit_bar,exit_px,exit_why=j,tgt,"Target Hit"; exited=True
            audit_trail.append({
                "trade_idx":trade_n,"bar_dt":df.index[j],"entry_price":round(entry,4),
                "direction":"LONG" if direction==1 else "SHORT","sl_level":round(sl,4),
                "target_level":round(disp_tgt,4),"bar_open":float(df["Open"].iloc[j]),
                "bar_high":round(bh,4),"bar_low":round(bl,4),"bar_close":float(df["Close"].iloc[j]),
                "sl_breached":bool((direction==1 and bl<=sl) or (direction==-1 and bh>=sl)),
                "tgt_breached":bool(tf and ((direction==1 and bh>=tgt) or (direction==-1 and bl<=tgt))),
                "trade_exited":exited,"exit_reason":exit_why if exited else None,
            })
            if exited: break
        if exit_bar is None:
            exit_bar=n-1; exit_px=float(df["Close"].iloc[exit_bar]); exit_why="End of Data"
        pnl=round((exit_px-entry)*direction,4)
        trades.append({
            "Signal Bar":df.index[i],"Entry DateTime":df.index[entry_idx],
            "Entry Price":round(entry,4),"Exit Price":round(exit_px,4),
            "SL Level":round(sl,4),"Target Level":round(disp_tgt,4),
            "Highest Price":round(highest,4),"Lowest Price":round(lowest,4),
            "Direction":"LONG" if direction==1 else "SHORT","SL Type":sl_type,
            "Target Type":tgt_type,"Exit DateTime":df.index[exit_bar],
            "Exit Reason":exit_why,"Points Gained":round(max(pnl,0),4),
            "Points Lost":round(abs(min(pnl,0)),4),"PnL":pnl,
        })
        i=exit_bar+1
    return trades,indics,audit_trail

def calc_perf(trades):
    if not trades: return {}
    t=len(trades); wins=[x for x in trades if x["PnL"]>0]; loss=[x for x in trades if x["PnL"]<0]
    pnls=[x["PnL"] for x in trades]
    return {
        "Total Trades":t,"Wins":len(wins),"Losses":len(loss),
        "Accuracy (%)":round(len(wins)/t*100,2),"Total PnL":round(sum(pnls),2),
        "Total Pts Won":round(sum(x["Points Gained"] for x in trades),2),
        "Total Pts Lost":round(sum(x["Points Lost"] for x in trades),2),
        "Avg Win":round(np.mean([x["PnL"] for x in wins]) if wins else 0,2),
        "Avg Loss":round(np.mean([x["PnL"] for x in loss]) if loss else 0,2),
        "Max Win":round(max(pnls),2),"Max Loss":round(min(pnls),2),
        "Profit Factor":round(sum(x["PnL"] for x in wins)/abs(sum(x["PnL"] for x in loss)) if loss else float("inf"),2),
    }

# ── OPTIMIZATION ──────────────────────────────────────────────────────────────
PARAM_GRIDS={
    "EMA Crossover":{"fast":[5,9,12,20],"slow":[15,21,26,50]},
    "RSI Overbought/Oversold":{"period":[9,14,21],"ob":[65,70,75],"os_":[25,30,35]},
    "Bollinger Bands":{"period":[15,20,25],"std":[1.5,2.0,2.5]},
    "MACD Crossover":{"fast":[8,12,16],"slow":[21,26,30],"signal":[7,9,11]},
    "Supertrend":{"period":[5,7,10,14],"multiplier":[2.0,2.5,3.0,3.5]},
    "ADX + DI Crossover":{"period":[10,14,20],"adx_thresh":[20,25,30]},
    "Stochastic Oscillator":{"k":[9,14,21],"d":[3,5],"ob":[75,80],"os_":[20,25]},
    "Donchian Breakout":{"period":[10,15,20,30]},
    "Triple EMA Trend":{"f":[5,9,12],"m":[15,21,26],"s_":[40,50,60]},
    "BB + RSI Mean Reversion":{"bb_period":[15,20],"bb_std":[1.5,2.0,2.5],"rsi_period":[10,14],"rsi_os":[25,30],"rsi_ob":[65,70]},
    "Keltner Channel Breakout":{"ema_p":[14,20,26],"atr_p":[10,14],"mult":[1.5,2.0,2.5]},
    "Williams %R Reversal":{"period":[9,14,21],"ob":[-20,-25],"os_":[-75,-80]},
    "Swing Trend + Pullback":{"trend_ema":[20,50,100],"entry_ema":[5,9,15],"rsi_period":[10,14],"vol_mult":[1.0,1.2,1.5]},
}
_BP={"atr_mult_sl":1.5,"atr_mult_tgt":2.0,"rr_ratio":2.0,"swing_lookback":5}

def optimize(df,strategy,sl_type,sl_pts,tgt_type,tgt_pts,desired_acc,min_pts,min_trades,progress_cb=None):
    grid=PARAM_GRIDS.get(strategy)
    if not grid:
        t,_,_=run_backtest(df,strategy,_BP,sl_type,sl_pts,tgt_type,tgt_pts); p=calc_perf(t)
        if p: return [{"params":_BP,**p,"Meets_Accuracy":p.get("Accuracy (%)",0)>=desired_acc,"Meets_Pts":p.get("Total Pts Won",0)>=min_pts}]
        return []
    keys=list(grid.keys()); combos=list(itertools_product(*[grid[k] for k in keys])); total=len(combos); results=[]
    for idx,combo in enumerate(combos):
        p={**dict(zip(keys,combo)),**_BP}
        try:
            t,_,_=run_backtest(df,strategy,p,sl_type,sl_pts,tgt_type,tgt_pts); perf=calc_perf(t)
            if perf.get("Total Trades",0)>=min_trades:
                results.append({"params":p,**perf,
                    "Meets_Accuracy":perf.get("Accuracy (%)",0)>=desired_acc,
                    "Meets_Pts":perf.get("Total Pts Won",0)>=min_pts})
        except: pass
        if progress_cb: progress_cb(min((idx+1)/total,1.0))
    results.sort(key=lambda r:(-r.get("Accuracy (%)",0),-r.get("Total PnL",0)))
    return results
# ── PLOTTING ──────────────────────────────────────────────────────────────────
_SKIP={"RSI","RSI_OB","RSI_OS","MACD","MACD_Signal","ADX","+DI","-DI","Stoch_K","Stoch_D","Williams_%R"}
_CLR=["#2196F3","#FF9800","#9C27B0","#00BCD4","#4CAF50","#F44336","#FFEB3B","#E91E63"]

def plot_ohlc(df,trades=None,indics=None,title="OHLC"):
    hv="Volume" in df.columns and df["Volume"].sum()>0
    fig=make_subplots(rows=2 if hv else 1,cols=1,shared_xaxes=True,
        row_heights=[0.72,0.28] if hv else [1.0],vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],
        low=df["Low"],close=df["Close"],name="Price",
        increasing_line_color="#26a69a",decreasing_line_color="#ef5350"),row=1,col=1)
    if hv:
        bc=["#26a69a" if c>=o else "#ef5350" for c,o in zip(df["Close"],df["Open"])]
        fig.add_trace(go.Bar(x=df.index,y=df["Volume"],name="Volume",marker_color=bc,opacity=0.45),row=2,col=1)
    if indics:
        ci=0
        for name,ser in indics.items():
            if not isinstance(ser,pd.Series) or name in _SKIP: continue
            dash="dash" if any(x in name.lower() for x in ["lower","lo_","_lo"]) else "solid"
            fig.add_trace(go.Scatter(x=ser.index,y=ser,name=name,
                line=dict(color=_CLR[ci%len(_CLR)],width=1.5,dash=dash),opacity=0.85),row=1,col=1)
            ci+=1
    if trades:
        ex=[t["Entry DateTime"] for t in trades]; ey=[t["Entry Price"] for t in trades]
        xx=[t["Exit DateTime"] for t in trades]; xy=[t["Exit Price"] for t in trades]
        ec=["#00E676" if t["Direction"]=="LONG" else "#FF5252" for t in trades]
        xc=["#26a69a" if t["Exit Reason"]=="Target Hit" else "#ef5350" for t in trades]
        es=["triangle-up" if t["Direction"]=="LONG" else "triangle-down" for t in trades]
        fig.add_trace(go.Scatter(x=ex,y=ey,mode="markers",
            marker=dict(symbol=es,size=13,color=ec,line=dict(color="white",width=1)),name="Entry"),row=1,col=1)
        fig.add_trace(go.Scatter(x=xx,y=xy,mode="markers",
            marker=dict(symbol="x",size=11,color=xc,line=dict(color="white",width=1)),name="Exit"),row=1,col=1)
    fig.update_layout(title=title,template="plotly_dark",height=600,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h",y=1.02,x=1,xanchor="right"),margin=dict(t=60,b=20))
    return fig

def plot_equity(trades,title="Equity Curve"):
    if not trades: return None
    cum=np.cumsum([t["PnL"] for t in trades])
    # handle both backtest ("Exit DateTime") and live trade ("Exit Time") dicts
    times=[t.get("Exit DateTime", t.get("Exit Time", "")) for t in trades]
    color="#00E676" if cum[-1]>=0 else "#FF5252"; fill="rgba(0,230,118,0.1)" if cum[-1]>=0 else "rgba(255,82,82,0.1)"
    fig=go.Figure(go.Scatter(x=times,y=cum,mode="lines+markers",fill="tozeroy",fillcolor=fill,
        line=dict(color=color,width=2),name="PnL"))
    fig.update_layout(title=title,template="plotly_dark",height=280,
        yaxis_title="Cumulative PnL (Points)",margin=dict(t=40,b=20))
    return fig

# ── STRATEGY PARAMS UI ────────────────────────────────────────────────────────
def strategy_params_ui(strategy,prefix,applied=None):
    def _n(label,lo,hi,default,key,step=1,fmt="%.4g"):
        try: v=float(applied.get(key.split("_",1)[-1],default)) if applied else float(default)
        except: v=float(default)
        return st.number_input(label,float(lo),float(hi),v,step=float(step),format=fmt,key=key)
    p={}
    if strategy=="EMA Crossover":
        c1,c2=st.columns(2)
        with c1: p["fast"]=int(_n("EMA Fast",2,200,9,f"{prefix}_fast"))
        with c2: p["slow"]=int(_n("EMA Slow",2,500,15,f"{prefix}_slow"))
    elif strategy=="RSI Overbought/Oversold":
        st.caption("**OR condition:** 🟢 BUY — RSI crosses **above** Oversold  |  🔴 SELL — RSI crosses **below** Overbought")
        c1,c2,c3=st.columns(3)
        with c1: p["period"]=int(_n("RSI Period",2,100,14,f"{prefix}_rp"))
        with c2: p["ob"]=int(_n("Overbought (sell trigger)",50,95,70,f"{prefix}_ob"))
        with c3: p["os_"]=int(_n("Oversold (buy trigger)",5,50,30,f"{prefix}_os"))
    elif strategy=="Price Threshold Cross":
        p["threshold"]=float(_n("Threshold Price",0,1e9,0,f"{prefix}_thresh",step=0.01,fmt="%.2f"))
    elif strategy=="Bollinger Bands":
        c1,c2=st.columns(2)
        with c1: p["period"]=int(_n("BB Period",5,200,20,f"{prefix}_bbp"))
        with c2: p["std"]=float(_n("Std Dev",0.5,5,2,f"{prefix}_bbs",step=0.1))
    elif strategy=="RSI Divergence":
        c1,c2=st.columns(2)
        with c1: p["period"]=int(_n("RSI Period",2,100,14,f"{prefix}_rp2"))
        with c2: p["lookback"]=int(_n("Lookback",2,50,5,f"{prefix}_lb"))
    elif strategy=="MACD Crossover":
        c1,c2,c3=st.columns(3)
        with c1: p["fast"]=int(_n("Fast",2,100,12,f"{prefix}_mf"))
        with c2: p["slow"]=int(_n("Slow",5,200,26,f"{prefix}_ms"))
        with c3: p["signal"]=int(_n("Signal",2,100,9,f"{prefix}_msig"))
    elif strategy=="Supertrend":
        c1,c2=st.columns(2)
        with c1: p["period"]=int(_n("Period",2,50,7,f"{prefix}_stp"))
        with c2: p["multiplier"]=float(_n("Multiplier",0.5,10,3,f"{prefix}_stm",step=0.5))
    elif strategy=="ADX + DI Crossover":
        c1,c2=st.columns(2)
        with c1: p["period"]=int(_n("ADX Period",2,50,14,f"{prefix}_ap"))
        with c2: p["adx_thresh"]=int(_n("ADX Threshold",10,50,25,f"{prefix}_at"))
    elif strategy=="Stochastic Oscillator":
        c1,c2,c3,c4=st.columns(4)
        with c1: p["k"]=int(_n("%K",2,50,14,f"{prefix}_k"))
        with c2: p["d"]=int(_n("%D",2,20,3,f"{prefix}_d"))
        with c3: p["ob"]=int(_n("OB",50,95,80,f"{prefix}_so"))
        with c4: p["os_"]=int(_n("OS",5,50,20,f"{prefix}_su"))
    elif strategy=="VWAP Deviation":
        p["dev_pct"]=float(_n("Deviation %",0.1,10,1,f"{prefix}_vd",step=0.1))
    elif strategy=="Ichimoku Cloud":
        c1,c2=st.columns(2)
        with c1: p["tenkan"]=int(_n("Tenkan",2,50,9,f"{prefix}_it"))
        with c2: p["kijun"]=int(_n("Kijun",5,100,26,f"{prefix}_ik"))
    elif strategy=="BB + RSI Mean Reversion":
        c1,c2=st.columns(2)
        with c1: p["bb_period"]=int(_n("BB Period",5,100,20,f"{prefix}_brbbp"))
        with c2: p["bb_std"]=float(_n("BB Std",0.5,5,2,f"{prefix}_brbbs",step=0.1))
        c3,c4,c5=st.columns(3)
        with c3: p["rsi_period"]=int(_n("RSI Period",2,50,14,f"{prefix}_brrp"))
        with c4: p["rsi_os"]=int(_n("RSI OS",5,50,35,f"{prefix}_bro"))
        with c5: p["rsi_ob"]=int(_n("RSI OB",50,95,65,f"{prefix}_brob"))
    elif strategy=="Donchian Breakout":
        p["period"]=int(_n("Channel Period",5,200,20,f"{prefix}_dp"))
    elif strategy=="Triple EMA Trend":
        c1,c2,c3=st.columns(3)
        with c1: p["f"]=int(_n("EMA1",2,50,9,f"{prefix}_tf"))
        with c2: p["m"]=int(_n("EMA2",5,100,21,f"{prefix}_tm"))
        with c3: p["s_"]=int(_n("EMA3",10,300,50,f"{prefix}_ts"))
    elif strategy=="Heikin Ashi EMA":
        p["ema_period"]=int(_n("EMA Period",5,200,20,f"{prefix}_hap"))
    elif strategy=="Volume Price Trend (VPT)":
        p["vpt_ema_period"]=int(_n("Signal EMA",2,100,14,f"{prefix}_vp"))
    elif strategy=="Keltner Channel Breakout":
        c1,c2,c3=st.columns(3)
        with c1: p["ema_p"]=int(_n("EMA Period",5,100,20,f"{prefix}_kep"))
        with c2: p["atr_p"]=int(_n("ATR Period",2,50,10,f"{prefix}_kap"))
        with c3: p["mult"]=float(_n("Multiplier",0.5,5,2,f"{prefix}_km",step=0.25))
    elif strategy=="Williams %R Reversal":
        c1,c2,c3=st.columns(3)
        with c1: p["period"]=int(_n("Period",2,50,14,f"{prefix}_wrp"))
        with c2: p["ob"]=int(_n("OB(e.g.-20)",-5,-1,-20,f"{prefix}_wrob"))
        with c3: p["os_"]=int(_n("OS(e.g.-80)",-99,-50,-80,f"{prefix}_wros"))
    elif strategy=="Swing Trend + Pullback":
        st.caption("Trend EMA filter + pullback to Entry EMA + RSI zone + volume surge")
        c1,c2,c3=st.columns(3)
        with c1:
            p["trend_ema"]=int(_n("Trend EMA",10,200,50,f"{prefix}_ste"))
            p["entry_ema"]=int(_n("Entry EMA",2,50,9,f"{prefix}_see"))
            p["rsi_period"]=int(_n("RSI Period",2,50,14,f"{prefix}_srp"))
        with c2:
            p["rsi_bull_min"]=int(_n("RSI Bull Min",20,60,40,f"{prefix}_sbmin"))
            p["rsi_bull_max"]=int(_n("RSI Bull Max",50,90,65,f"{prefix}_sbmax"))
            p["vol_mult"]=float(_n("Vol Mult",0.5,5,1.2,f"{prefix}_svm",step=0.1))
        with c3:
            p["rsi_bear_min"]=int(_n("RSI Bear Min",20,60,35,f"{prefix}_snmin"))
            p["rsi_bear_max"]=int(_n("RSI Bear Max",50,90,60,f"{prefix}_snmax"))
    p.setdefault("atr_mult_sl",1.5); p.setdefault("atr_mult_tgt",2.0)
    p.setdefault("rr_ratio",2.0); p.setdefault("swing_lookback",5)
    return p

def config_banner(strategy,interval,period,sym,sl_type,sl_pts,tgt_type,tgt_pts,extra=None):
    items=[("Strategy",strategy[:16]),("Interval",interval),("Period",period),("Ticker",sym),
           ("SL Type",sl_type[:14]),("SL Pts",sl_pts),("Tgt Type",tgt_type[:14]),("Tgt Pts",tgt_pts)]
    if extra:
        for k,v in list(extra.items())[:4]: items.append((k,v))
    for col,(label,val) in zip(st.columns(len(items)),items): col.metric(label,val)
# ── SESSION STATE ─────────────────────────────────────────────────────────────
for _k,_v in {"live_active":False,"live_trades":[],"live_position":None,"live_tick":0,
               "opt_applied":None,"opt_results":None,"opt_res_meta":None,
               "opt_df":None}.items():
    if _k not in st.session_state: st.session_state[_k]=_v

def _idx(lst,val,default=0): return lst.index(val) if val in lst else default

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Global Config")
    _oa=st.session_state.get("opt_applied")
    t_choice=st.selectbox("Instrument",list(TICKER_MAP.keys()),
        index=_idx(list(TICKER_MAP.keys()),_oa.get("instrument","Nifty 50") if _oa else "Nifty 50"),key="g_ticker")
    sym=st.text_input("Yahoo Ticker","RELIANCE.NS",key="g_custom").strip() if t_choice=="Custom" else TICKER_MAP[t_choice]
    interval=st.selectbox("Timeframe",TIMEFRAMES,index=_idx(TIMEFRAMES,_oa.get("interval","1h") if _oa else "1h",4),key="g_interval")
    period=st.selectbox("Period",PERIODS,index=_idx(PERIODS,_oa.get("period","3mo") if _oa else "3mo",4),key="g_period")
    st.markdown("---")
    st.subheader("📈 Strategy")
    strategy=st.selectbox("Strategy",STRATEGIES,index=_idx(STRATEGIES,_oa.get("strategy","EMA Crossover") if _oa else "EMA Crossover"),key="g_strategy")
    st.subheader("🛡 Stop Loss")
    sl_type=st.selectbox("SL Type",SL_TYPES,index=_idx(SL_TYPES,_oa.get("sl_type","Custom Points") if _oa else "Custom Points"),key="g_sl_type")
    sl_pts=st.number_input("SL Value (pts)",0.01,1e6,float(_oa.get("sl_pts",10)) if _oa else 10.,step=0.5,key="g_sl_pts")
    st.subheader("🎯 Target")
    tgt_type=st.selectbox("Target Type",TARGET_TYPES,index=_idx(TARGET_TYPES,_oa.get("tgt_type","Custom Points") if _oa else "Custom Points"),key="g_tgt_type")
    tgt_pts=st.number_input("Target Value (pts)",0.01,1e6,float(_oa.get("tgt_pts",20)) if _oa else 20.,step=0.5,key="g_tgt_pts")
    if _oa:
        st.success(f"✅ Applied: {_oa.get('strategy')}\nAcc: {_oa.get('accuracy','?')}%")
        if st.button("Clear Applied"): st.session_state.opt_applied=None; st.rerun()
    st.markdown("---")
    st.caption("1.5s rate-limit delay between all yfinance requests.")
    if not _HAS_FRAGMENT: st.caption("Upgrade Streamlit ≥1.33 for flicker-free live tab.")

# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab_bt,tab_live,tab_opt=st.tabs(["📊 Backtesting","⚡ Live Trading","🔬 Optimization"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.subheader(f"Backtesting · {t_choice} [{interval}/{period}]")
    _app=st.session_state.get("opt_applied")
    with st.expander("⚙️ Strategy Parameters",expanded=True):
        bt_params=strategy_params_ui(strategy,prefix="bt",applied=_app.get("params") if _app else None)
    config_banner(strategy,interval,period,sym,sl_type,sl_pts,tgt_type,tgt_pts,
        extra={k:v for k,v in bt_params.items() if k not in("atr_mult_sl","atr_mult_tgt","rr_ratio","swing_lookback")})

    if st.button("▶ Run Backtest",type="primary",key="btn_bt"):
        with st.spinner("Fetching data…"): df_bt=fetch_data(sym,period,interval)
        if df_bt is None or df_bt.empty:
            st.error("No data returned. Try different ticker/interval/period.")
        else:
            with st.spinner("Running backtest…"):
                trades_bt,indics_bt,audit_bt=run_backtest(df_bt,strategy,bt_params,sl_type,sl_pts,tgt_type,tgt_pts)
            perf_bt=calc_perf(trades_bt)
            st.markdown("### 📋 Performance Summary")
            if perf_bt:
                mc=st.columns(len(perf_bt))
                for col,(k,v) in zip(mc,perf_bt.items()):
                    col.metric(k,v,delta=f"{v:.1f}%" if k=="Accuracy (%)" else None)
            else: st.info("No trades generated.")
            st.markdown("### 📈 Price Chart")
            st.plotly_chart(plot_ohlc(df_bt,trades_bt,indics_bt,title=f"{t_choice}—{strategy}[{interval}]"),use_container_width=True)
            if trades_bt:
                eq=plot_equity(trades_bt)
                if eq: st.markdown("### 💹 Equity Curve"); st.plotly_chart(eq,use_container_width=True)
                st.markdown("### 📜 Trade Log  *(price columns grouped for comparison)*")
                tdf=pd.DataFrame(trades_bt)
                COL=["Entry DateTime","Exit DateTime","Direction",
                     "Entry Price","Exit Price","SL Level","Target Level",
                     "Highest Price","Lowest Price",
                     "Exit Reason","SL Type","Target Type",
                     "Points Gained","Points Lost","PnL","Signal Bar"]
                tdf=tdf[[c for c in COL if c in tdf.columns]]
                # Only color the PnL / points columns — no full-row background
                def _cell_color(v):
                    if isinstance(v,(int,float)): return "color:#00E676;font-weight:bold" if v>0 else ("color:#FF5252;font-weight:bold" if v<0 else "")
                    return ""
                style_cols=[c for c in ["PnL","Points Gained","Points Lost"] if c in tdf.columns]
                st.dataframe(tdf.style.map(_cell_color,subset=style_cols),use_container_width=True,height=430)
                with st.expander("📂 Raw OHLC Data"): st.dataframe(df_bt,use_container_width=True)
                # ── VERIFICATION ──────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 🔍 Backtest Logic Verification")
                st.caption(
                    "This section checks whether the backtest engine correctly exited trades at SL and Target levels. "
                    "**✅ Green = results are correct** (no missed exits). "
                    "**❌ Red = a bug was found** (trade continued even after candle Low/High crossed SL or Target). "
                    "SL is always checked first (conservative — if both breach same candle, SL wins)."
                )
                if audit_bt:
                    adf=pd.DataFrame(audit_bt)
                    # SL check
                    sl_breach=adf[((adf["direction"]=="LONG")&(adf["bar_low"]<=adf["sl_level"]))|
                                  ((adf["direction"]=="SHORT")&(adf["bar_high"]>=adf["sl_level"]))]
                    sl_anom=[]
                    for tid,grp in sl_breach.groupby("trade_idx"):
                        fb=grp.index[0]
                        eb=adf[(adf["trade_idx"]==tid)&(adf["trade_exited"]==True)]
                        if len(eb) and eb.index[0]>fb:
                            sl_anom.append({"trade":tid,"1st SL breach":grp.iloc[0]["bar_dt"],
                                "SL level":grp.iloc[0]["sl_level"],
                                "bar Low/High":(grp.iloc[0]["bar_low"] if grp.iloc[0]["direction"]=="LONG" else grp.iloc[0]["bar_high"]),
                                "actual exit":eb.iloc[0]["bar_dt"],"issue":"continued past SL"})
                    if sl_anom: st.error(f"❌ {len(sl_anom)} SL anomalies!"); st.dataframe(pd.DataFrame(sl_anom),use_container_width=True)
                    else: st.success("✅ SL exits verified — all SL-breached bars triggered exit correctly.")
                    # Target check
                    tdo=(tgt_type=="Trailing Target (Display Only)")
                    tgt_breach=adf[((adf["direction"]=="LONG")&(adf["bar_high"]>=adf["target_level"]))|
                                   ((adf["direction"]=="SHORT")&(adf["bar_low"]<=adf["target_level"]))]
                    tgt_anom=[]
                    for tid,grp in tgt_breach.groupby("trade_idx"):
                        if tdo: continue
                        fb=grp.index[0]; ta=adf[adf["trade_idx"]==tid]
                        sl_b=ta[((ta["direction"]=="LONG")&(ta["bar_low"]<=ta["sl_level"]))|
                                ((ta["direction"]=="SHORT")&(ta["bar_high"]>=ta["sl_level"]))]
                        sl_before=len(sl_b)>0 and sl_b.index[0]<=fb
                        eb=ta[ta["trade_exited"]==True]
                        if len(eb) and eb.index[0]>fb and not sl_before:
                            tgt_anom.append({"trade":tid,"1st tgt breach":grp.iloc[0]["bar_dt"],
                                "target level":grp.iloc[0]["target_level"],
                                "bar High/Low":(grp.iloc[0]["bar_high"] if grp.iloc[0]["direction"]=="LONG" else grp.iloc[0]["bar_low"]),
                                "actual exit":eb.iloc[0]["bar_dt"],"issue":"continued past target"})
                    if tdo: st.info("ℹ️ Trailing Target (Display Only) — target intentionally never triggers. ✓")
                    elif tgt_anom: st.error(f"❌ {len(tgt_anom)} Target anomalies!"); st.dataframe(pd.DataFrame(tgt_anom),use_container_width=True)
                    else: st.success("✅ Target exits verified — all target-breached bars triggered correctly.")
                    sc=st.columns(3)
                    sc[0].metric("SL-breach bars",len(sl_breach))
                    sc[1].metric("Tgt-breach bars",len(tgt_breach))
                    sc[2].metric("Audit bars",len(adf))
                    with st.expander("📋 Full Bar-Level Audit Trail"):
                        disp=adf[["trade_idx","direction","bar_dt","entry_price","bar_high","bar_low","bar_close","sl_level","target_level","sl_breached","tgt_breached","trade_exited","exit_reason"]]
                        def _ast(row):
                            if row["trade_exited"]: return ["background-color:#1a2a1a"]*len(row)
                            if row["sl_breached"]: return ["background-color:#2a1a1a"]*len(row)
                            return [""]*len(row)
                        st.dataframe(disp.style.apply(_ast,axis=1),use_container_width=True,height=400)
                    with st.expander("ℹ️ Bars where SL intact (trade correctly open)"):
                        safe=adf[((adf["direction"]=="LONG")&(adf["sl_level"]<adf["bar_low"]))|
                                 ((adf["direction"]=="SHORT")&(adf["sl_level"]>adf["bar_high"]))]
                        st.caption(f"{len(safe)} bars — SL below Low/above High — trade correctly remained open.")
                        st.dataframe(safe[["trade_idx","bar_dt","direction","sl_level","bar_low","bar_high","target_level","trade_exited"]],use_container_width=True,height=300)
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE TRADING  (flicker-free via @st.fragment on Streamlit>=1.33)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.subheader(f"⚡ Live Trading · {t_choice} [{interval}]")
    with st.expander("⚙️ Strategy Parameters (Live)",expanded=False):
        live_params=strategy_params_ui(strategy,prefix="lv")
    config_banner(strategy,interval,"—",sym,sl_type,sl_pts,tgt_type,tgt_pts,
        extra={k:v for k,v in live_params.items() if k not in("atr_mult_sl","atr_mult_tgt","rr_ratio","swing_lookback")})
    st.markdown("---")
    _bc=st.columns([1,1,1,5])
    if _bc[0].button("▶ Start",type="primary",disabled=st.session_state.live_active,key="btn_lv_start"):
        st.session_state.update({"live_active":True,"live_trades":[],"live_position":None,"live_tick":0}); st.rerun()
    if _bc[1].button("⏹ Stop",disabled=not st.session_state.live_active,key="btn_lv_stop"):
        st.session_state.live_active=False; st.rerun()
    if _bc[2].button("🗑 Clear",key="btn_lv_clear"):
        st.session_state.update({"live_trades":[],"live_position":None}); st.rerun()
    _bc[3].info("Smooth updates: " + ("✅ Active (Streamlit≥1.33)" if _HAS_FRAGMENT else "⚠️ Upgrade Streamlit≥1.33"))
    sub_mon,sub_hist=st.tabs(["📡 Live Monitor","📜 Trade History"])

    def _live_render():
        if not st.session_state.live_active:
            st.info("Press ▶ Start to begin live monitoring."); return
        st.session_state.live_tick+=1; tick=st.session_state.live_tick
        st.caption(f"Tick **#{tick}** — {datetime.now().strftime('%H:%M:%S')}  |  1.5s rate-limit enforced")
        lv_df=fetch_live(sym,interval)
        if lv_df is None or lv_df.empty: st.warning("No data. Retrying…"); return
        lv_n=len(lv_df); cl=float(lv_df["Close"].iloc[-1])
        bh_cur=float(lv_df["High"].iloc[-1]); bl_cur=float(lv_df["Low"].iloc[-1])
        last_bar=lv_df.index[-1]
        fn=STRATEGY_FN.get(strategy,sig_custom)
        try: lv_sigs,lv_indics=fn(lv_df,**live_params)
        except Exception as e: lv_sigs=pd.Series(0,index=lv_df.index); lv_indics={}; st.warning(f"Strategy error:{e}")
        last_sig=int(lv_sigs.iloc[-2]) if len(lv_sigs)>1 else 0
        # Price row
        m=st.columns(6)
        m[0].metric("LTP",f"{cl:.2f}")
        m[1].metric("High",f"{bh_cur:.2f}")
        m[2].metric("Low",f"{bl_cur:.2f}")
        m[3].metric("Spread",f"{bh_cur-bl_cur:.2f}")
        sig_txt="🟢 BUY" if last_sig==1 else ("🔴 SELL" if last_sig==-1 else "⚪ FLAT")
        m[4].metric("Signal",sig_txt)
        m[5].metric("Last Bar",str(last_bar)[:19])
        pos=st.session_state.live_position
        if pos is None and last_sig!=0:
            d=last_sig; ep=cl
            lv_sl=init_sl(lv_df,lv_n-1,ep,d,sl_type,sl_pts,live_params)
            lv_tgt=init_tgt(lv_df,lv_n-1,ep,d,tgt_type,tgt_pts,lv_sl,live_params)
            st.session_state.live_position={"entry":ep,"direction":d,"sl":lv_sl,"target":lv_tgt,
                "disp_tgt":lv_tgt,"entry_time":last_bar,"highest":ep,"lowest":ep}
            # dhan_place_order(sym,"BUY" if d==1 else "SELL",LOT_SIZE)
            st.success(f"🚀 NEW {'LONG' if d==1 else 'SHORT'}  Entry:{ep:.2f}  SL:{lv_sl:.2f}  Target:{lv_tgt:.2f}")
        elif pos is not None:
            d=pos["direction"]; ep=pos["entry"]
            pos["highest"]=max(pos["highest"],bh_cur); pos["lowest"]=min(pos["lowest"],bl_cur)
            pos["sl"]=update_sl(lv_df,lv_n-1,ep,d,sl_type,sl_pts,pos["sl"],live_params)
            new_t,tf=update_tgt(lv_df,lv_n-1,d,tgt_type,tgt_pts,pos["disp_tgt"],live_params)
            pos["disp_tgt"]=new_t
            if tf: pos["target"]=new_t
            exited=False; exit_px=None; exit_why=None
            if d==1:
                if bl_cur<=pos["sl"]: exited,exit_px,exit_why=True,pos["sl"],"SL Hit"
                elif tf and bh_cur>=pos["target"]: exited,exit_px,exit_why=True,pos["target"],"Target Hit"
            else:
                if bh_cur>=pos["sl"]: exited,exit_px,exit_why=True,pos["sl"],"SL Hit"
                elif tf and bl_cur<=pos["target"]: exited,exit_px,exit_why=True,pos["target"],"Target Hit"
            if exited:
                pnl=round((exit_px-ep)*d,4)
                st.session_state.live_trades.append({"Entry Time":last_bar,"Entry Price":ep,
                    "Direction":"LONG" if d==1 else "SHORT","Exit Time":last_bar,"Exit Price":exit_px,
                    "Exit Reason":exit_why,"SL":pos["sl"],"Target":pos["disp_tgt"],
                    "Highest":pos["highest"],"Lowest":pos["lowest"],"PnL":pnl})
                st.session_state.live_position=None
                # dhan_exit_order(sym,"BUY" if d==1 else "SELL",LOT_SIZE)
                (st.success if pnl>0 else st.error)(f"CLOSED {exit_why} | PnL: {'+'if pnl>0 else ''}{pnl:.2f}")
            else:
                unreal=round((cl-ep)*d,4)
                st.markdown("#### 📌 Open Position")
                p_=st.columns(8)
                p_[0].metric("Direction","🟢 LONG" if d==1 else "🔴 SHORT")
                p_[1].metric("Entry",f"{ep:.2f}")
                p_[2].metric("LTP",f"{cl:.2f}")
                p_[3].metric("SL",f"{pos['sl']:.2f}",delta=f"dist {abs(cl-pos['sl']):.2f}",delta_color="inverse")
                p_[4].metric("Target",f"{pos['disp_tgt']:.2f}",delta=f"dist {abs(pos['disp_tgt']-cl):.2f}")
                p_[5].metric("Highest",f"{pos['highest']:.2f}")
                p_[6].metric("Lowest",f"{pos['lowest']:.2f}")
                p_[7].metric("Unrealised",f"{unreal:.2f}",delta=f"{unreal:.2f}",delta_color="normal" if unreal>=0 else "inverse")
        st.plotly_chart(plot_ohlc(lv_df,indics=lv_indics,title=f"LIVE:{t_choice}({interval}) Tick#{tick}"),use_container_width=True)

    def _hist_render():
        t_=st.session_state.live_trades
        if t_:
            tot=len(t_); wins=sum(1 for x in t_ if x["PnL"]>0); pnl_=sum(x["PnL"] for x in t_)
            pw=sum(x["PnL"] for x in t_ if x["PnL"]>0); pl=abs(sum(x["PnL"] for x in t_ if x["PnL"]<0))
            hc=st.columns(5)
            hc[0].metric("Trades",tot); hc[1].metric("Accuracy",f"{wins/tot*100:.1f}%")
            hc[2].metric("Total PnL",f"{pnl_:.2f}"); hc[3].metric("Pts Won",f"{pw:.2f}"); hc[4].metric("Pts Lost",f"{pl:.2f}")
            hdf=pd.DataFrame(t_)
            def _sc(v):
                if isinstance(v,(int,float)): return "color:#00E676" if v>0 else ("color:#FF5252" if v<0 else "")
                return ""
            st.dataframe(hdf.style.map(_sc,subset=["PnL"]),use_container_width=True)
            eq_=plot_equity(t_,"Live Equity")
            if eq_: st.plotly_chart(eq_,use_container_width=True)
        else: st.info("No completed live trades yet.")

    with sub_mon:
        if _HAS_FRAGMENT and st.session_state.live_active:
            @st.fragment(run_every=2)
            def _lv_frag(): _live_render()
            _lv_frag()
        else:
            _live_render()
            if st.session_state.live_active: time.sleep(1.5); st.rerun()
    with sub_hist:
        if _HAS_FRAGMENT and st.session_state.live_active:
            @st.fragment(run_every=3)
            def _hist_frag(): _hist_render()
            _hist_frag()
        else: _hist_render()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_opt:
    st.subheader("🔬 Strategy Parameter Optimization")
    st.markdown("Grid-searches all combinations. **All** results shown sorted by accuracy. "
                "Highlighted rows meet your targets. Check rows → **Apply to Config** → use directly in Backtest/Live.")
    with st.expander("📥 Optimization Inputs",expanded=True):
        oc1,oc2,oc3=st.columns(3)
        opt_t=oc1.selectbox("Instrument",list(TICKER_MAP.keys()),key="opt_t")
        opt_sym=oc1.text_input("Custom Ticker","RELIANCE.NS",key="opt_csym").strip() if opt_t=="Custom" else TICKER_MAP[opt_t]
        opt_iv=oc1.selectbox("Timeframe",TIMEFRAMES,index=4,key="opt_iv")
        opt_pd=oc2.selectbox("Period",PERIODS,index=5,key="opt_pd")
        opt_st=oc2.selectbox("Strategy",[s for s in STRATEGIES if s in PARAM_GRIDS],key="opt_strat")
        opt_acc=oc3.slider("Desired Accuracy (%)",40,99,60,key="opt_acc")
        opt_pts=oc3.number_input("Min Total Pts Won",0.,1e6,0.,step=10.,key="opt_pts")
        opt_mt=int(oc3.number_input("Min Trades",1,50,3,step=1,key="opt_mt"))
        oc4,oc5=st.columns(2)
        opt_sl=oc4.selectbox("SL Type",SL_TYPES,key="opt_sl")
        opt_slp=oc4.number_input("SL Pts",0.01,1e6,10.,step=0.5,key="opt_slp")
        opt_tgt=oc5.selectbox("Target Type",TARGET_TYPES,key="opt_tgt")
        opt_tgtp=oc5.number_input("Target Pts",0.01,1e6,20.,step=0.5,key="opt_tgtp")

    if st.button("🔬 Run Optimization",type="primary",key="btn_opt"):
        with st.spinner("Fetching data…"): df_opt=fetch_data(opt_sym,opt_pd,opt_iv)
        if df_opt is None or df_opt.empty:
            st.error("No data returned.")
        else:
            g=PARAM_GRIDS.get(opt_st,{}); n_combos=1
            for v in g.values(): n_combos*=len(v)
            st.info(f"Data: **{len(df_opt)} bars** · Grid: **{n_combos}** combos · Min trades: {opt_mt}")
            prog=st.progress(0)
            with st.spinner(f"Optimising {opt_st}…"):
                opt_res=optimize(df_opt,opt_st,opt_sl,opt_slp,opt_tgt,opt_tgtp,
                    opt_acc,float(opt_pts),opt_mt,progress_cb=prog.progress)
            prog.empty()
            # ── Store results + metadata in session state so they survive reruns ──
            st.session_state.opt_results = opt_res
            st.session_state.opt_df      = df_opt
            st.session_state.opt_res_meta= {
                "strategy":opt_st,"instrument":opt_t,"interval":opt_iv,"period":opt_pd,
                "sl":opt_sl,"slp":opt_slp,"tgt":opt_tgt,"tgtp":opt_tgtp,
                "acc":opt_acc,"pts":opt_pts,
            }

    # ── Display results from session state (persists across reruns / checkbox ticks) ──
    opt_res   = st.session_state.get("opt_results")
    _meta     = st.session_state.get("opt_res_meta") or {}
    df_opt_ss = st.session_state.get("opt_df")

    if opt_res is not None:
        _opt_acc  = _meta.get("acc", opt_acc)
        _opt_pts  = _meta.get("pts", opt_pts)
        _opt_st   = _meta.get("strategy", opt_st)
        _opt_sl   = _meta.get("sl", opt_sl)
        _opt_slp  = _meta.get("slp", opt_slp)
        _opt_tgt  = _meta.get("tgt", opt_tgt)
        _opt_tgtp = _meta.get("tgtp", opt_tgtp)
        _opt_t    = _meta.get("instrument", opt_t)
        _opt_iv   = _meta.get("interval", opt_iv)
        _opt_pd   = _meta.get("period", opt_pd)

        if not opt_res:
            st.warning("No combinations produced enough trades. Try longer period, lower min-trades, or different SL/Target.")
        else:
            meets=[r for r in opt_res if r.get("Meets_Accuracy") and r.get("Meets_Pts",True)]
            st.success(f"✅ **{len(opt_res)}** results found  |  **{len(meets)}** meet Accuracy≥{_opt_acc}%"
                + (f" & Pts≥{int(_opt_pts)}" if _opt_pts>0 else ""))
            rows=[]
            for r in opt_res:
                row={}
                for k,v in r["params"].items():
                    if k not in("atr_mult_sl","atr_mult_tgt","rr_ratio","swing_lookback"): row[k]=v
                for k,v in r.items():
                    if k not in("params","Meets_Accuracy","Meets_Pts"): row[k]=v
                row["✓ Meets"]=("✅" if r.get("Meets_Accuracy") and r.get("Meets_Pts",True) else "—")
                rows.append(row)
            res_df=pd.DataFrame(rows)

            # Highlight rows that meet criteria (background on ✓ Meets col only)
            def _hl_meets(v): return "background-color:#0d3b0d;color:white" if v=="✅" else ""

            st.markdown("#### 📊 All Results (sorted by accuracy) — ☑ check a row then click Apply")
            # Use a unique key that does NOT include the button click so it persists
            edited=st.data_editor(
                res_df,
                column_config={c: st.column_config.Column(disabled=True)
                               for c in res_df.columns},
                hide_index=True, use_container_width=True, height=450,
                key="opt_editor",
                # We manage selection via a separate checkboxes approach below
            )

            # ── Row selection via separate checkbox list (stable across reruns) ──
            st.markdown("**Select a row to apply:**")
            sel_idx = st.selectbox(
                "Choose result row # (0 = best)",
                options=list(range(len(opt_res))),
                format_func=lambda i: (
                    f"Row {i} | Acc={opt_res[i].get('Accuracy (%)','?')}% | "
                    f"PnL={opt_res[i].get('Total PnL','?')} | "
                    f"Trades={opt_res[i].get('Total Trades','?')} | "
                    f"Params={dict(list(opt_res[i]['params'].items())[:3])} {'✅' if opt_res[i].get('Meets_Accuracy') else ''}"
                ),
                key="opt_sel_idx",
            )

            sel_result = opt_res[sel_idx]
            # Show selected row details
            with st.expander(f"📌 Selected Row {sel_idx} — details", expanded=True):
                sc=st.columns(4)
                sc[0].metric("Accuracy", f"{sel_result.get('Accuracy (%)','?')}%")
                sc[1].metric("Total PnL", sel_result.get("Total PnL","?"))
                sc[2].metric("Total Trades", sel_result.get("Total Trades","?"))
                sc[3].metric("Meets Targets", "✅ Yes" if sel_result.get("Meets_Accuracy") else "— No")
                param_disp={k:v for k,v in sel_result["params"].items()
                            if k not in("atr_mult_sl","atr_mult_tgt","rr_ratio","swing_lookback")}
                st.write("**Parameters:**", param_disp)

            if st.button("⚡ Apply Selected Row to Config (Sidebar + Backtest)",
                         type="primary", key="btn_apply_opt"):
                param_keys=list(PARAM_GRIDS.get(_opt_st,{}).keys())
                applied_params={pk:sel_result["params"][pk] for pk in param_keys if pk in sel_result["params"]}
                st.session_state.opt_applied={
                    "strategy":_opt_st,"instrument":_opt_t,"interval":_opt_iv,"period":_opt_pd,
                    "sl_type":_opt_sl,"sl_pts":_opt_slp,"tgt_type":_opt_tgt,"tgt_pts":_opt_tgtp,
                    "params":applied_params,
                    "accuracy":sel_result.get("Accuracy (%)","?"),
                    "pnl":sel_result.get("Total PnL","?"),
                }
                st.success(f"✅ Row {sel_idx} applied: {_opt_st}  Accuracy={sel_result.get('Accuracy (%)','?')}%  "
                           f"→ Switch to 📊 Backtesting tab to run it!")
                st.rerun()

            # Best result preview
            if df_opt_ss is not None:
                st.markdown("---"); st.markdown("### 🥇 Best Result Preview (Row 0)")
                best=opt_res[0]; best_p={**best["params"],**_BP}
                bt2,ind2,_=run_backtest(df_opt_ss,_opt_st,best_p,_opt_sl,_opt_slp,_opt_tgt,_opt_tgtp)
                bp2=calc_perf(bt2)
                if bp2:
                    bmc=st.columns(len(bp2))
                    for col,(k,v) in zip(bmc,bp2.items()): col.metric(k,v)
                acc_lbl=f"{best.get('Accuracy (%)','?'):.1f}%" if isinstance(best.get('Accuracy (%)'),float) else str(best.get('Accuracy (%)','?'))
                st.plotly_chart(plot_ohlc(df_opt_ss,bt2,ind2,
                    title=f"Best Params: {dict(list(best['params'].items())[:4])} | Acc={acc_lbl}"),
                    use_container_width=True)
                eq_b=plot_equity(bt2)
                if eq_b: st.plotly_chart(eq_b,use_container_width=True)
                if bt2:
                    with st.expander("📜 Trade Log (Best Params)"):
                        btdf=pd.DataFrame(bt2)
                        CO=["Entry DateTime","Exit DateTime","Direction","Entry Price","Exit Price",
                            "SL Level","Target Level","Highest Price","Lowest Price",
                            "Exit Reason","Points Gained","Points Lost","PnL"]
                        CO=[c for c in CO if c in btdf.columns]
                        def _sp(v):
                            if isinstance(v,(int,float)): return "color:#00E676;font-weight:bold" if v>0 else ("color:#FF5252;font-weight:bold" if v<0 else "")
                            return ""
                        style_co=[c for c in ["PnL","Points Gained","Points Lost"] if c in btdf.columns]
                        st.dataframe(btdf[CO].style.map(_sp,subset=style_co),use_container_width=True)
