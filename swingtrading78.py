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
try:
    import pytz
    _IST = pytz.timezone("Asia/Kolkata")
    def to_ist(dt):
        try:
            if pd.isna(dt): return ""
            dt = pd.Timestamp(dt)
            if dt.tzinfo is None: dt = dt.tz_localize("UTC")
            return dt.tz_convert(_IST).strftime("%d-%b-%Y %H:%M IST")
        except: return str(dt)[:19]
except ImportError:
    def to_ist(dt):
        try: return str(pd.Timestamp(dt))[:19]
        except: return str(dt)[:19]
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# ██  EDIT THESE TWO LINES TO UPDATE YOUR DHAN CREDENTIALS  ██████████████████
# ══════════════════════════════════════════════════════════════════════════════
DHAN_CLIENT_ID = "1104779876"
DHAN_ACCESS_TOKEN = (
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9"
    ".eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzczODAyMDg2"
    ",\"aWF0IjoxNzczNzE1Njg2LCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3"
    "ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNDc3OTg3NiJ9"
    ".L5ULyf8AfeaoZS_kn95rtQZ6qNRJF3EUimCJw_8q12k2FZHEGEPNySKrYOBP9"
    "vRfBHKvEqWoB0ZC7GRUd7zyMg"
)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="AlgoTrader Pro", layout="wide", initial_sidebar_state="expanded")
_HAS_FRAGMENT = hasattr(st, "fragment")

# ── CSS: sidebar strict containment + scrolling ──────────────────────────────
st.markdown("""
<style>
/* Main layout: ensure main area is never pushed by sidebar overflow */
.main .block-container { overflow-x: hidden !important; }

/* Sidebar strict containment */
section[data-testid="stSidebar"] {
    overflow: hidden !important;
    position: relative !important;
}
section[data-testid="stSidebar"] > div {
    overflow-y: auto !important;
    overflow-x: hidden !important;
    width: 300px !important;
    max-width: 300px !important;
    min-width: 240px !important;
}
/* Every element inside sidebar is clipped to sidebar width */
section[data-testid="stSidebar"] * {
    box-sizing: border-box !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}
/* Text wraps, never overflows */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    white-space: normal !important;
    word-break: break-word !important;
    overflow-wrap: anywhere !important;
    font-size: 13px !important;
}
/* Inputs and selects fill sidebar width exactly */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] textarea {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    font-size: 13px !important;
}
/* Streamlit's internal stSelectbox/stNumberInput wrappers */
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stNumberInput,
section[data-testid="stSidebar"] .stTextInput,
section[data-testid="stSidebar"] .stCheckbox {
    width: 100% !important;
    max-width: 100% !important;
}
/* Metric values */
[data-testid="stMetricValue"] { font-size: 13px !important; }
[data-testid="stMetricLabel"] { font-size: 11px !important; white-space: normal !important; }
/* Expanders inside sidebar */
section[data-testid="stSidebar"] details { width: 100% !important; max-width: 100% !important; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
TICKER_MAP = {
    "Nifty 50":"^NSEI","Bank Nifty":"^NSEBANK","Nifty IT":"^CNXIT","Sensex":"^BSESN",
    "BTC/USD":"BTC-USD","ETH/USD":"ETH-USD","USD/INR":"USDINR=X","Gold":"GC=F",
    "Silver":"SI=F","EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X","USD/JPY":"JPYUSD=X",
    "Crude Oil":"CL=F","Custom":"CUSTOM",
}

# ── EMAIL ALERT HELPER ────────────────────────────────────────────────────────
def send_alert(subject: str, body: str, sender: str, app_password: str, to: str):
    """
    Send a plain-text email via Gmail SMTP.
    Uses TLS (port 587). Requires a Gmail App Password
    (Account → Security → 2-Step Verification → App Passwords).
    Silently does nothing if any credential is empty.
    """
    if not sender or not app_password or not to:
        return
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"]    = sender
        msg["To"]      = to
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(sender, app_password)
            server.sendmail(sender, [to], msg.as_string())
    except Exception as _email_err:
        # Don't crash the app if email fails; log silently
        print(f"[Email alert error] {_email_err}")


# ── Nifty 50 Constituent Stocks ───────────────────────────────────────────────
NIFTY50_STOCKS = {
    "Reliance":"RELIANCE.NS","TCS":"TCS.NS","HDFC Bank":"HDFCBANK.NS",
    "Infosys":"INFY.NS","ICICI Bank":"ICICIBANK.NS","HUL":"HINDUNILVR.NS",
    "ITC":"ITC.NS","SBI":"SBIN.NS","Bajaj Finance":"BAJFINANCE.NS",
    "Bharti Airtel":"BHARTIARTL.NS","Kotak Mahindra":"KOTAKBANK.NS",
    "L&T":"LT.NS","Asian Paints":"ASIANPAINT.NS","HCL Tech":"HCLTECH.NS",
    "Axis Bank":"AXISBANK.NS","Wipro":"WIPRO.NS","Maruti Suzuki":"MARUTI.NS",
    "Sun Pharma":"SUNPHARMA.NS","Titan":"TITAN.NS","UltraTech Cement":"ULTRACEMCO.NS",
    "Power Grid":"POWERGRID.NS","NTPC":"NTPC.NS","Nestle India":"NESTLEIND.NS",
    "Tech Mahindra":"TECHM.NS","M&M":"M&M.NS","Bajaj Auto":"BAJAJ-AUTO.NS",
    "JSW Steel":"JSWSTEEL.NS","Tata Steel":"TATASTEEL.NS","Adani Ports":"ADANIPORTS.NS",
    "ONGC":"ONGC.NS","Coal India":"COALINDIA.NS","Cipla":"CIPLA.NS",
    "Hindalco":"HINDALCO.NS","Dr Reddy":"DRREDDY.NS","IndusInd Bank":"INDUSINDBK.NS",
    "Bajaj Finserv":"BAJAJFINSV.NS","HDFC Life":"HDFCLIFE.NS","SBI Life":"SBILIFE.NS",
    "Divis Labs":"DIVISLAB.NS","Eicher Motors":"EICHERMOT.NS","Apollo Hospitals":"APOLLOHOSP.NS",
    "Grasim":"GRASIM.NS","BEL":"BEL.NS","BPCL":"BPCL.NS","Shriram Finance":"SHRIRAMFIN.NS",
    "Tata Consumer":"TATACONSUM.NS","Tata Motors":"TATAMOTORS.NS",
    "Hero MotoCorp":"HEROMOTOCO.NS","Britannia":"BRITANNIA.NS","HDFC AMC":"HDFCAMC.NS",
}
NIFTY50_SYMBOLS = list(NIFTY50_STOCKS.values())
TIMEFRAMES = ["1m","5m","15m","30m","1h","4h","1d","1wk"]
PERIODS    = ["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","max"]
STRATEGIES = [
    "EMA Crossover","RSI Overbought/Oversold","Simple Buy","Simple Sell",
    "Price Threshold Cross","Bollinger Bands","RSI Divergence",
    "MACD Crossover","Supertrend","ADX + DI Crossover","Stochastic Oscillator",
    "VWAP Deviation","Ichimoku Cloud","BB + RSI Mean Reversion",
    "Donchian Breakout","Triple EMA Trend","Heikin Ashi EMA",
    "Volume Price Trend (VPT)","Keltner Channel Breakout","Williams %R Reversal",
    "Swing Trend + Pullback",
    # ── Advanced / High-Probability ────────────────────────────────────────
    "Elliott Wave (Simplified)","Elliott Wave v2 (Swing+Fib)","Elliott Wave v3 (Extrema)",
    "HV Percentile (IV Proxy)","SMC Order Blocks",
    "Price Action Patterns","Breakout Strategy","Support & Resistance + EMA",
    "VWAP + EMA Confluence",        # price above VWAP and EMA, pullback entry
    "Opening Range Breakout (ORB)", # first N-min high/low break with volume
    "Mean Reversion Bollinger",     # BB squeeze then expansion trade
    "Trend Momentum (ADX+EMA)",     # ADX strong trend + EMA pullback
    "Gap & Go",                     # gap-up/gap-down continuation trade
    "Inside Bar Breakout",          # inside bar pattern breakout
    # ── Custom builder ──────────────────────────
    "Custom Strategy Builder","Custom Strategy",
]
SL_TYPES = [
    "Custom Points","Trailing SL (Points)","Trailing Prev Candle Low/High",
    "Trailing Curr Candle Low/High","Trailing Prev Swing Low/High",
    "Trailing Curr Swing Low/High","Cost to Cost (Breakeven)",
    "Cost-to-Cost K-Shift Trailing","EMA Reverse Crossover",
    "ATR Based","Risk/Reward Based SL",
    "Strategy Signal Exit",   # exit when strategy fires opposite signal
]
TARGET_TYPES = [
    "Custom Points","Trailing Target (Display Only)","Trailing Prev Candle High/Low",
    "Trailing Curr Candle High/Low","Trailing Prev Swing High/Low",
    "Trailing Curr Swing High/Low","ATR Based","Risk/Reward Based",
    "Reverse EMA Crossover","Strategy Signal Exit",  # exit on strategy reversal
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

def fetch_live(ticker, interval):
    """
    Fetch live data for the given ticker and interval.

    Root cause of stale data: yf.download(period=...) uses yfinance's internal
    SQLite cache (~/.cache/py-yfinance). The same (ticker, period, interval)
    combination returns cached data for several minutes — causing the app to
    show 14:15 candle while standalone script gets 15:29.

    Fix: use explicit start/end datetimes instead of period=. This forces a
    fresh HTTP request every time, bypassing the period-based cache.
    Also explicitly call yf.download with auto_adjust=True and enough lookback
    (same bar count as before) so indicators have sufficient warmup bars.
    """
    import pytz as _pytz2
    from datetime import timedelta as _td

    _ist2 = _pytz2.timezone("Asia/Kolkata")
    _now2 = datetime.now(_ist2)

    # Lookback durations matching the old period= strings (conservative)
    _lookback_days = {
        "1m": 7, "5m": 30, "15m": 30, "30m": 60,
        "1h": 60, "4h": 90, "1d": 365,
    }
    _days = _lookback_days.get(interval, 30)
    _start = (_now2 - _td(days=_days)).strftime("%Y-%m-%d")
    # end = tomorrow to ensure today's partial day is fully included
    _end   = (_now2 + _td(days=1)).strftime("%Y-%m-%d")

    try:
        time.sleep(1.5)
        # Clear yfinance's internal SQLite cache before fetching so we never
        # get a stale period-cached response. This is the primary fix for
        # "app shows 14:15 candle while market closed at 15:30" issue.
        try:
            yf.set_tz_cache_location(None)   # disable tz cache
        except: pass
        try:
            import yfinance.cache as _yfc
            _yfc.clear_cache()
        except: pass
        raw = yf.download(
            ticker,
            start    = _start,
            end      = _end,
            interval = YF_IV.get(interval, interval),
            progress = False,
            auto_adjust = True,
            prepost  = False,   # regular market hours only
        )
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = _flatten(raw)
        return _r4h(df) if interval == "4h" and not df.empty else df
    except Exception as e:
        st.warning(f"Live fetch: {e}")
        return pd.DataFrame()

# ── INDICATORS  (TradingView-compatible) ──────────────────────────────────────
# EMA: span=p, adjust=False, min_periods=p  → matches TV EMA exactly
# RSI: Wilder RMA = ewm(alpha=1/p, adjust=False, min_periods=p)
# ATR: Wilder RMA same
def ema(s,p):
    """TV-compatible EMA: starts from bar p, seeds with SMA of first p bars."""
    result = s.ewm(span=p, adjust=False, min_periods=p).mean()
    return result

def sma(s,p):  return s.rolling(p, min_periods=p).mean()

def rma(s,p):
    """Wilder's Moving Average (RMA) — used by TV for RSI, ATR."""
    return s.ewm(alpha=1/p, adjust=False, min_periods=p).mean()

def rsi(s,p=14):
    """TV-compatible RSI using Wilder's RMA (same as Pine Script ta.rsi)."""
    d=s.diff()
    g=d.clip(lower=0); l=(-d).clip(lower=0)
    ag=rma(g,p); al=rma(l,p)
    rs=ag/al.replace(0,np.nan)
    return 100-100/(1+rs)

def atr(df,p=14):
    """TV-compatible ATR using Wilder's RMA."""
    hl=df["High"]-df["Low"]
    hc=(df["High"]-df["Close"].shift()).abs()
    lc=(df["Low"]-df["Close"].shift()).abs()
    tr=pd.concat([hl,hc,lc],axis=1).max(axis=1)
    return rma(tr,p)

def bollinger(s,p=20,k=2.0):
    m=sma(s,p); d=s.rolling(p,min_periods=p).std(); return m-k*d,m,m+k*d

def macd(s,f=12,sl=26,sig=9):
    ml=ema(s,f)-ema(s,sl); return ml,ema(ml,sig),ml-ema(ml,sig)

def stoch(df,k=14,d=3):
    lo=df["Low"].rolling(k,min_periods=k).min()
    hi=df["High"].rolling(k,min_periods=k).max()
    pk=100*(df["Close"]-lo)/(hi-lo).replace(0,np.nan); return pk,sma(pk,d)

def vwap_calc(df):
    tp=(df["High"]+df["Low"]+df["Close"])/3
    vol=df.get("Volume",pd.Series(1,index=df.index)).replace(0,np.nan)
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
    pdi=100*rma(pdm,p)/_a.replace(0,np.nan)
    ndi=100*rma(ndm,p)/_a.replace(0,np.nan)
    dx=100*(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)
    return rma(dx,p),pdi,ndi

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

def sig_price_thresh(df, threshold=0., thresh_dir="Above", thresh_action="Buy", **_):
    """
    Threshold strategy:
      thresh_dir:    "Above" = fires when price crosses ABOVE threshold
                     "Below" = fires when price crosses BELOW threshold
      thresh_action: "Buy"  = generate BUY  (long) signal
                     "Sell" = generate SELL (short) signal
                     "Both" = Above→BUY, Below→SELL (original behavior)
    """
    th = _cs(threshold, df.index)
    s  = pd.Series(0, index=df.index)
    if thresh_action == "Both":
        s[_cup(df["Close"], th)] = 1; s[_cdn(df["Close"], th)] = -1
    elif thresh_dir == "Above":
        sig_val = 1 if thresh_action == "Buy" else -1
        s[_cup(df["Close"], th)] = sig_val
    else:  # Below
        sig_val = 1 if thresh_action == "Buy" else -1
        s[_cdn(df["Close"], th)] = sig_val
    return s, {"Threshold": th}

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

# ── ADVANCED STRATEGY: Elliott Wave (Simplified zigzag wave count) ──────────
def _ew_build_pivots(df, min_wave_pct=1.0):
    """
    Build zigzag pivot list from Close prices.
    Returns list of (bar_idx, price, direction)  direction: 1=swing-high, -1=swing-low
    Also returns (last_dir, last_px, last_idx) — the in-progress (unconfirmed) swing.
    """
    cl     = df["Close"]
    n      = len(cl)
    min_mv = min_wave_pct / 100.0
    pivots = []
    last_dir = 0
    last_px  = float(cl.iloc[0])
    last_idx = 0

    for i in range(1, n):
        p = float(cl.iloc[i])
        if last_dir == 0:
            if   p > last_px * (1 + min_mv): last_dir = 1
            elif p < last_px * (1 - min_mv): last_dir = -1
        elif last_dir == 1:
            if p < float(cl.iloc[last_idx]) * (1 - min_mv):
                pivots.append((last_idx, float(cl.iloc[last_idx]), 1))
                last_dir, last_px, last_idx = -1, p, i
            elif p > last_px: last_px, last_idx = p, i
        else:
            if p > float(cl.iloc[last_idx]) * (1 + min_mv):
                pivots.append((last_idx, float(cl.iloc[last_idx]), -1))
                last_dir, last_px, last_idx = 1, p, i
            elif p < last_px: last_px, last_idx = p, i

    return pivots, (last_dir, last_px, last_idx)


def _ew_diagnostics(df, min_wave_pct=1.0):
    """
    Returns a dict with rich Elliott Wave diagnostic state for live display:
      - pivots: full list
      - confirmed_count: number of confirmed zigzag pivots
      - last_3: last 3 confirmed pivots (wave labels)
      - current_wave_dir: direction of in-progress (unconfirmed) swing
      - current_swing_pct: % move of in-progress swing so far
      - needed_for_long / needed_for_short: conditions + current values
      - last_signal_type: LONG / SHORT / None
      - retrace_pct: how far current pullback has retraced prior leg
    """
    cl = df["Close"]
    n  = len(cl)
    cur_price = float(cl.iloc[-1])
    min_mv = min_wave_pct / 100.0

    pivots, (last_dir, last_px, last_idx) = _ew_build_pivots(df, min_wave_pct)

    # Current in-progress swing stats
    if last_idx < n and last_idx >= 0:
        swing_start_px = float(cl.iloc[last_idx])
    else:
        swing_start_px = cur_price

    current_swing_pct = (cur_price - swing_start_px) / max(abs(swing_start_px), 0.001) * 100

    # In-progress: how far has price moved from last confirmed pivot?
    if pivots:
        last_confirmed_px  = pivots[-1][1]
        last_confirmed_dir = pivots[-1][2]
        move_from_last     = (cur_price - last_confirmed_px) / max(abs(last_confirmed_px), 0.001) * 100
        needed_pct         = min_wave_pct
        pct_done           = abs(move_from_last)
        pct_remaining      = max(0, needed_pct - pct_done)
        pivot_flip_pct     = pct_done / max(needed_pct, 0.001) * 100
    else:
        last_confirmed_px  = float(cl.iloc[0])
        last_confirmed_dir = 0
        move_from_last     = 0.0
        pct_done           = 0.0
        pct_remaining      = min_wave_pct
        pivot_flip_pct     = 0.0

    # Last signal type
    last_signal = None
    for k in range(2, len(pivots)):
        p0, p1, p2 = pivots[k-2], pivots[k-1], pivots[k]
        if p0[2]==-1 and p1[2]==1 and p2[2]==-1 and p2[1]>p0[1]:
            last_signal = "LONG"
        elif p0[2]==1 and p1[2]==-1 and p2[2]==1 and p2[1]<p0[1]:
            last_signal = "SHORT"

    # Last 3 pivots for wave labelling
    last_3 = pivots[-3:] if len(pivots)>=3 else pivots

    # Retracement of most recent completed leg
    retrace_pct = None
    if len(pivots) >= 2:
        leg_start = pivots[-2][1]
        leg_end   = pivots[-1][1]
        leg_size  = abs(leg_end - leg_start)
        retrace   = abs(cur_price - leg_end)
        retrace_pct = retrace / max(leg_size, 0.001) * 100

    # Describe next signal conditions — SEQUENTIAL steps, not parallel checks.
    # Later conditions must NOT show ✅ until all prior conditions are complete.
    long_conditions  = []
    short_conditions = []

    if len(pivots) >= 2:
        p_last  = pivots[-1]
        p_prev  = pivots[-2]
        leg_dir = p_last[2]   # direction of most recent confirmed pivot

        if leg_dir == 1:
            # Last confirmed pivot was a SWING HIGH.
            # LONG setup: price needs to pull back → form a HIGHER LOW → then signal fires.
            # Step 1: price drops enough to confirm Pivot C (corrective low)
            req_low_px  = p_last[1] * (1 - min_mv)
            step1_done  = cur_price < req_low_px
            # Step 2: only relevant after step 1 — price bounces back up (confirms Pivot C as low)
            # Step 3: Pivot C must be above prior Pivot A (p_prev, the prior low)
            prior_low   = p_prev[1]

            long_conditions = [
                {
                    "Step":      "CURRENT → Step 1 of 3",
                    "Condition": "Price must drop ≥ min_wave_pct% from the last HIGH to form Pivot C (corrective low)",
                    "Required":  f"< {req_low_px:.2f}  (last high {p_last[1]:.2f} × {100-min_wave_pct:.2f}%)",
                    "Current":   f"{cur_price:.2f}",
                    "Met":       "✅ Pivot C low confirmed — move to Step 2" if step1_done
                                 else f"❌ Need {req_low_px - cur_price:.2f} pts more drop",
                    "Progress":  f"{min(100,(p_last[1]-cur_price)/max(p_last[1]-req_low_px,0.001)*100):.0f}%",
                },
                {
                    "Step":      "Step 2 of 3 (active only after Step 1)",
                    "Condition": "After Pivot C low forms, price must BOUNCE back up ≥ min_wave_pct% — this confirms the signal bar",
                    "Required":  f"+{min_wave_pct:.2f}% bounce from Pivot C low",
                    "Current":   "Waiting for Step 1 to complete first" if not step1_done else f"Current low: {cur_price:.2f} — watching for bounce",
                    "Met":       "⏳ Locked — complete Step 1 first" if not step1_done else "⏳ Watching for bounce confirmation",
                    "Progress":  "—",
                },
                {
                    "Step":      "Step 3 of 3 (checked simultaneously with Step 2)",
                    "Condition": "Pivot C low must be HIGHER than prior swing low (= Pivot A). This makes it a higher-low → bullish structure",
                    "Required":  f"> {prior_low:.2f}  (Pivot A, the prior swing low)",
                    "Current":   f"Will be checked after Step 1 completes" if not step1_done else f"{cur_price:.2f}",
                    "Met":       "⏳ Locked — complete Step 1 first" if not step1_done
                                 else ("✅ YES — higher low structure intact" if cur_price > prior_low
                                       else f"❌ Below Pivot A ({prior_low:.2f}) — bullish structure broken"),
                    "Progress":  "—" if not step1_done else f"{(cur_price-prior_low)/max(abs(prior_low),0.001)*100:+.2f}%",
                },
            ]
            short_conditions = [
                {
                    "Step":      "CURRENT → Step 1 of 2",
                    "Condition": "Price must RALLY further above the last confirmed HIGH to set a new high (Pivot B extension)",
                    "Required":  f"> {p_last[1]:.2f}  (extend beyond last high)",
                    "Current":   f"{cur_price:.2f}",
                    "Met":       "✅ Extended above last high" if cur_price > p_last[1] else "❌ Not yet above last high",
                    "Progress":  f"{(cur_price-p_last[1])/max(abs(p_last[1]),0.001)*100:+.2f}%",
                },
                {
                    "Step":      "Step 2 of 2 (active after Step 1)",
                    "Condition": "Then price must DROP ≥ min_wave_pct% from the new high → forming a LOWER HIGH → SHORT signal",
                    "Required":  f"−{min_wave_pct:.2f}% drop from new high",
                    "Current":   "Waiting for Step 1 first" if cur_price <= p_last[1] else f"New high set — watching for drop",
                    "Met":       "⏳ Watching",
                    "Progress":  "—",
                },
            ]

        else:  # leg_dir == -1 → last confirmed pivot was a SWING LOW
            # LONG setup: price is currently rallying from the low.
            # KEY FIX: Use the HIGHEST candle High seen since the last confirmed pivot,
            # NOT cur_price. This prevents Step 1 from flickering on/off as price wiggles
            # around the threshold tick-by-tick.
            req_high_px = p_last[1] * (1 + min_mv)
            # Rolling high from last pivot to current bar (uses High column, not Close)
            _pivot_bar_idx = last_idx if last_idx >= 0 and last_idx < n else max(0, n-50)
            _high_since_pivot = float(df["High"].iloc[_pivot_bar_idx:].max())
            step1_done  = _high_since_pivot > req_high_px   # STABLE — only goes True, never reverts
            prior_low   = p_last[1]

            # Pullback: how much has price dropped from the highest point since pivot
            _recent_high     = _high_since_pivot if step1_done else cur_price
            _pullback_so_far = max(0, (_recent_high - cur_price) / max(abs(_recent_high), 0.001) * 100) if step1_done else 0
            _pullback_needed = min_wave_pct
            step2_progress   = min(100, _pullback_so_far / max(_pullback_needed, 0.001) * 100) if step1_done else 0
            step2_done_approx = step1_done and _pullback_so_far >= _pullback_needed

            _step1_pct = min(100,(_high_since_pivot-p_last[1])/max(req_high_px-p_last[1],0.001)*100)

            long_conditions = [
                {
                    "Step":      "CURRENT → Step 1 of 3",
                    "Condition": "Highest candle High since pivot A must exceed pivot A × (1 + min_wave_pct%)",
                    "Required":  f"> {req_high_px:.2f}  (pivot A {p_last[1]:.2f} + {min_wave_pct:.2f}%)",
                    "Current":   f"Highest High since pivot: {_high_since_pivot:.2f}  |  LTP: {cur_price:.2f}",
                    "Met":       "✅ Rally confirmed (Pivot B set) — now watching for pullback" if step1_done
                                 else f"❌ Need {req_high_px-_high_since_pivot:.2f} pts more  ({_step1_pct:.0f}% there)",
                    "Progress":  f"{_step1_pct:.0f}%",
                },
                {
                    "Step":      "Step 2 of 3" + (" ← ACTIVE NOW" if step1_done else " (locked until Step 1 done)"),
                    "Condition": "LTP must drop ≥ min_wave_pct% from Pivot B high — forms Pivot C (corrective low) = SIGNAL BAR",
                    "Required":  f"Price ≤ {_recent_high*(1-min_mv/100):.2f}  (= {_recent_high:.2f} − {min_wave_pct:.2f}%)" if step1_done
                                 else f"−{min_wave_pct:.2f}% from Pivot B",
                    "Current":   f"Pullback so far: {_pullback_so_far:.2f}% / {_pullback_needed:.2f}% needed" if step1_done
                                 else "⏳ Complete Step 1 first",
                    "Met":       "✅ Pullback complete — Pivot C formed!" if step2_done_approx
                                 else (f"⏳ Watching: {step2_progress:.0f}% of {_pullback_needed:.2f}% done" if step1_done
                                       else "⏳ Locked"),
                    "Progress":  f"{step2_progress:.0f}%" if step1_done else "—",
                },
                {
                    "Step":      "Step 3 of 3 (auto-checked after Step 2)",
                    "Condition": "Pivot C low must stay ABOVE pivot A ({:.2f}) → higher-low = bullish → LONG signal fires".format(prior_low),
                    "Required":  f"> {prior_low:.2f}  (pivot A)",
                    "Current":   f"~{cur_price:.2f}  (approx — Pivot C = actual reversal low)" if step1_done else "—",
                    "Met":       "⏳ Locked (Steps 1+2 first)" if not step2_done_approx
                                 else ("✅ Higher-low confirmed → LONG fires!" if cur_price > prior_low
                                       else f"❌ Below pivot A ({prior_low:.2f}) — structure broken"),
                    "Progress":  f"{(cur_price-prior_low)/max(abs(prior_low),0.001)*100:+.2f}% above pivot A" if step2_done_approx else "—",
                },
            ]
            # SHORT conditions when last pivot was a LOW
            req_low_short = p_last[1] * (1 - min_mv)
            step1s_done   = cur_price < req_low_short
            short_conditions = [
                {
                    "Step":      "CURRENT → Step 1 of 2",
                    "Condition": "Price must DROP ≥ min_wave_pct% below the prior swing LOW to confirm a new lower low → SHORT structure",
                    "Required":  f"< {req_low_short:.2f}",
                    "Current":   f"{cur_price:.2f}",
                    "Met":       "✅ New lower low confirmed" if step1s_done
                                 else f"❌ Need {cur_price - req_low_short:.2f} pts more drop",
                    "Progress":  f"{min(100,(p_last[1]-cur_price)/max(p_last[1]-req_low_short,0.001)*100):.0f}%",
                },
                {
                    "Step":      "Step 2 of 2 (active after Step 1)",
                    "Condition": "Bounce ≥ min_wave_pct% from new low, then another lower low → SHORT signal fires",
                    "Required":  f"+{min_wave_pct:.2f}% bounce then lower low",
                    "Current":   "Waiting for Step 1" if not step1s_done else "Watching for bounce then lower low",
                    "Met":       "⏳ Watching",
                    "Progress":  "—",
                },
            ]

    return {
        "confirmed_pivots":   len(pivots),
        "last_3_pivots":      last_3,
        "current_wave_dir":   last_dir,
        "current_swing_pct":  current_swing_pct,
        "move_from_last_pct": move_from_last,
        "pct_done":           pct_done,
        "pct_remaining":      pct_remaining,
        "pivot_flip_pct":     pivot_flip_pct,
        "last_confirmed_px":  last_confirmed_px,
        "last_confirmed_dir": last_confirmed_dir,
        "retrace_pct":        retrace_pct,
        "last_signal":        last_signal,
        "long_conditions":    long_conditions,
        "short_conditions":   short_conditions,
        "cur_price":          cur_price,
        "min_wave_pct":       min_wave_pct,
    }


def sig_elliott_wave(df, swing_lookback=10, min_wave_pct=1.0, **_):
    """
    Simplified Elliott Wave: uses zigzag to count alternating waves.
    BUY  at completion of corrective wave 2/4 in uptrend (retracement of prior up-move).
    SELL at completion of corrective wave 2/4 in downtrend.
    min_wave_pct: minimum % move to qualify as a wave leg.
    """
    cl  = df["Close"]
    n   = len(cl)
    s   = pd.Series(0, index=df.index)

    pivots, _ = _ew_build_pivots(df, min_wave_pct)

    for k in range(2, len(pivots)):
        p0, p1, p2 = pivots[k-2], pivots[k-1], pivots[k]
        idx2 = p2[0]
        if idx2 >= n: continue
        if p0[2]==-1 and p1[2]==1 and p2[2]==-1 and p2[1]>p0[1]:
            s.iloc[idx2] = 1
        elif p0[2]==1 and p1[2]==-1 and p2[2]==1 and p2[1]<p0[1]:
            s.iloc[idx2] = -1

    e20 = ema(df["Close"], 20)
    return s, {"EMA_20": e20}

# ── ADVANCED STRATEGY: Elliott Wave v2 (Bar-Based Swing + Fibonacci) ─────────
# v1 Problem on 5-min: min_wave_pct=1.0% → needs 1% per leg → 660pts for 3 legs → 3-4 trades/month
# v2 Solution: BAR-COUNT swing detection (N-bar pivot confirmation), not %-move threshold.
# v2 adds: Fibonacci retracement filter (38.2-88.6%), 5-wave impulse detection,
#          EMA trend filter, volume confirmation.

def _ew2_build_swings(df, swing_bars=5):
    """
    Build swing highs/lows using bar-count method (NOT percentage-move).
    A swing HIGH at bar i: df["High"][i] is the max in [i-swing_bars .. i+swing_bars].
    A swing LOW  at bar i: df["Low"][i]  is the min in [i-swing_bars .. i+swing_bars].

    Returns list of (bar_idx, price, direction)
      direction: 1 = swing high, -1 = swing low
    Alternates strictly (no two consecutive highs or lows).
    """
    hi = df["High"]; lo = df["Low"]
    n  = len(df)
    sb = max(1, swing_bars)

    # Find all candidate swing highs and lows
    candidates = []
    for i in range(sb, n - sb):
        hi_window = hi.iloc[max(0,i-sb):i+sb+1]
        lo_window = lo.iloc[max(0,i-sb):i+sb+1]
        is_sh = float(hi.iloc[i]) >= float(hi_window.max())
        is_sl = float(lo.iloc[i]) <= float(lo_window.min())
        if is_sh: candidates.append((i, float(hi.iloc[i]),  1))
        if is_sl: candidates.append((i, float(lo.iloc[i]), -1))

    # Sort by bar index, then enforce alternation (no two same-direction in a row)
    candidates.sort(key=lambda x: x[0])
    pivots = []
    last_dir = 0
    for (idx, px, d) in candidates:
        if d == last_dir:
            # Keep the more extreme value
            if pivots:
                prev = pivots[-1]
                if (d == 1 and px > prev[1]) or (d == -1 and px < prev[1]):
                    pivots[-1] = (idx, px, d)
        else:
            pivots.append((idx, px, d))
            last_dir = d

    return pivots


def _ew2_diagnostics(df, swing_bars=5, fib_min=0.382, fib_max=0.886, ema_period=50):
    """
    Rich diagnostics for EW v2 used by the Signal Progress expander.
    Returns same structure as _ew_diagnostics for compatibility.
    """
    cl       = df["Close"]
    n        = len(df)
    cur_price= float(cl.iloc[-1])

    pivots   = _ew2_build_swings(df, swing_bars)
    np_      = len(pivots)

    # Current in-progress swing (bars since last confirmed pivot)
    last_confirmed_px  = pivots[-1][1] if pivots else float(cl.iloc[0])
    last_confirmed_dir = pivots[-1][2] if pivots else 0
    last_confirmed_idx = pivots[-1][0] if pivots else 0

    # Bars since last pivot
    bars_since_pivot   = n - 1 - last_confirmed_idx

    # Current move from last pivot (%)
    move_from_last     = (cur_price - last_confirmed_px)/max(abs(last_confirmed_px),0.001)*100

    # Fibonacci retracement of most recent completed leg
    retrace_pct = None
    if np_ >= 2:
        leg_start = pivots[-2][1]; leg_end = pivots[-1][1]
        leg_size  = abs(leg_end - leg_start)
        retrace   = abs(cur_price - leg_end)
        retrace_pct = retrace / max(leg_size, 0.001) * 100

    # Check if in Fib zone (38.2–88.6% retrace)
    in_fib_zone = (retrace_pct is not None and fib_min*100 <= retrace_pct <= fib_max*100)

    # Last signal type
    last_signal = None
    for k in range(2, np_):
        p0,p1,p2 = pivots[k-2],pivots[k-1],pivots[k]
        leg_ab = abs(p1[1]-p0[1])
        if leg_ab == 0: continue
        retrace_c = abs(p2[1]-p1[1])/leg_ab
        if p0[2]==-1 and p1[2]==1 and p2[2]==-1 and p2[1]>p0[1]:
            if fib_min <= retrace_c <= fib_max: last_signal="LONG"
        elif p0[2]==1 and p1[2]==-1 and p2[2]==1 and p2[1]<p0[1]:
            if fib_min <= retrace_c <= fib_max: last_signal="SHORT"

    # Long / short conditions
    long_conds=[]; short_conds=[]
    if np_>=2:
        pl=pivots[-1]; pp=pivots[-2]
        if pl[2]==1:  # last pivot was HIGH → looking for corrective LOW to form LONG entry
            req_low = pl[1]*(1-fib_min)
            fib_lo  = pl[1]*(1-fib_max)
            long_conds=[
                {"Condition":"A) Price must pull back to Fib 38.2%-88.6% of prior up-leg",
                 "Required": f"{fib_lo:.2f} – {req_low:.2f}  (Fib 88.6%–38.2% of {pp[1]:.2f}→{pl[1]:.2f})",
                 "Current":  f"{cur_price:.2f}",
                 "Met":      "✅ In Fib zone!" if fib_lo<=cur_price<=req_low else f"❌ Need price between {fib_lo:.2f}–{req_low:.2f}",
                 "Progress": f"{min(100,retrace_pct):.0f}% retrace" if retrace_pct else "—"},
                {"Condition":"B) Must form a swing LOW (price stops falling & reverses up)",
                 "Required": f"New swing low in next {swing_bars} bars",
                 "Current":  f"Bars since last pivot: {bars_since_pivot}",
                 "Met":      "⏳ Watching for reversal" if bars_since_pivot<swing_bars*3 else "⏳ Forming",
                 "Progress": "—"},
                {"Condition":"C) New low must be HIGHER than prior swing low (higher-low = bullish)",
                 "Required": f"> {pp[1]:.2f}  (prior swing low)",
                 "Current":  f"{cur_price:.2f}",
                 "Met":      "✅ YES" if cur_price>pp[1] else "❌ Below prior low — structure broken",
                 "Progress": f"{(cur_price-pp[1])/max(abs(pp[1]),0.001)*100:+.2f}%"},
            ]
        else:  # last pivot was LOW → looking for corrective HIGH to form SHORT entry
            leg_ab=abs(pl[1]-pp[1]) if np_>=2 else 1
            req_hi=pl[1]+leg_ab*fib_min; req_hi_max=pl[1]+leg_ab*fib_max
            long_conds=[
                {"Condition":"A) Price must rally to Fib 38.2%-88.6% of prior down-leg",
                 "Required": f"{req_hi:.2f} – {req_hi_max:.2f}",
                 "Current":  f"{cur_price:.2f}",
                 "Met":      "✅ In Fib zone!" if req_hi<=cur_price<=req_hi_max else f"❌ Need rally to {req_hi:.2f}–{req_hi_max:.2f}",
                 "Progress": f"{min(100,move_from_last/((req_hi-pl[1])/max(abs(pl[1]),0.001)*100)*100):.0f}%" if move_from_last>0 else "0%"},
                {"Condition":"B) Must form swing HIGH and then pull back (confirms corrective move)",
                 "Required": "New swing high then pullback",
                 "Current":  f"Move so far: {move_from_last:+.2f}%",
                 "Met":      "⏳ Pending",
                 "Progress": "—"},
            ]

    return {
        "confirmed_pivots":   np_,
        "last_3_pivots":      pivots[-3:] if np_>=3 else pivots,
        "current_wave_dir":   last_confirmed_dir,
        "current_swing_pct":  move_from_last,
        "move_from_last_pct": move_from_last,
        "pct_done":           abs(move_from_last),
        "pct_remaining":      max(0, fib_min*100 - abs(move_from_last)),
        "pivot_flip_pct":     min(100, abs(move_from_last)/max(fib_min*100,0.001)*100),
        "last_confirmed_px":  last_confirmed_px,
        "last_confirmed_dir": last_confirmed_dir,
        "retrace_pct":        retrace_pct,
        "in_fib_zone":        in_fib_zone,
        "bars_since_pivot":   bars_since_pivot,
        "last_signal":        last_signal,
        "long_conditions":    long_conds,
        "short_conditions":   short_conds,
        "cur_price":          cur_price,
        "min_wave_pct":       0.0,   # not used in v2
        "swing_bars":         swing_bars,
    }


def sig_elliott_wave_v2(df,
                         swing_bars=5,
                         fib_min=0.382,
                         fib_max=0.886,
                         ema_period=50,
                         use_volume=True,
                         use_5wave=True,
                         **_):
    """
    Elliott Wave v2 — Bar-based swing detection + Fibonacci validation.

    Why more signals than v1:
      - v1: needs 1% move per leg → ~220pts on Nifty 5min → 3-4 trades/month
      - v2: needs N-bar swing confirmation → detects every meaningful swing regardless of %
            → 10-30x more signals on 5min, still filtered by Fibonacci for quality

    Signal types:
      1. 3-Wave Correction (A-B-C):
         LONG:  swing-low(A) → swing-high(B) → higher-swing-low(C) in Fib 38.2-88.6% zone + above EMA
         SHORT: swing-high(A) → swing-low(B) → lower-swing-high(C) in Fib zone + below EMA

      2. 5-Wave Impulse (if use_5wave=True):
         LONG:  five alternating pivots (low-high-low-high-low) all making higher values
         SHORT: five alternating pivots (high-low-high-low-high) all making lower values

    Parameters:
      swing_bars: bars each side to confirm a swing pivot (default 5 → 11-bar window)
      fib_min:    minimum retracement ratio for wave C (default 0.382 = 38.2%)
      fib_max:    maximum retracement ratio for wave C (default 0.886 = 88.6%)
      ema_period: trend filter EMA period (default 50)
      use_volume: require above-average volume on signal bar
      use_5wave:  also detect 5-wave impulse completions
    """
    n   = len(df)
    s   = pd.Series(0, index=df.index)
    cl  = df["Close"]

    pivots = _ew2_build_swings(df, swing_bars)
    np_    = len(pivots)

    # Trend filter
    e  = ema(cl, ema_period)

    # Volume filter
    vol     = df.get("Volume", pd.Series(1, index=df.index))
    avg_vol = vol.rolling(20, min_periods=5).mean().replace(0, np.nan)

    def _vol_ok(idx):
        if not use_volume: return True
        try:
            return float(vol.iloc[idx]) >= float(avg_vol.iloc[idx]) * 0.8
        except: return True

    # ── 3-Wave A-B-C correction signals ──────────────────────────────────────
    for k in range(2, np_):
        p0, p1, p2 = pivots[k-2], pivots[k-1], pivots[k]
        idx0, px0, d0 = p0
        idx1, px1, d1 = p1
        idx2, px2, d2 = p2
        if idx2 >= n: continue

        leg_AB = abs(px1 - px0)
        if leg_AB == 0: continue
        leg_BC = abs(px2 - px1)
        retrace_ratio = leg_BC / leg_AB

        # Fibonacci filter: wave C retraces 38.2%–88.6% of wave AB
        fib_ok = fib_min <= retrace_ratio <= fib_max

        # ── LONG: low → high → higher-low (bullish correction in uptrend)
        if d0==-1 and d1==1 and d2==-1 and px2 > px0 and fib_ok:
            trend_ok = float(e.iloc[idx2]) > float(e.iloc[idx2]) * 0.995  # price near/above EMA
            try: price_above_ema = float(cl.iloc[idx2]) >= float(e.iloc[idx2]) * 0.99
            except: price_above_ema = True
            if price_above_ema and _vol_ok(idx2):
                s.iloc[idx2] = 1

        # ── SHORT: high → low → lower-high (bearish correction in downtrend)
        elif d0==1 and d1==-1 and d2==1 and px2 < px0 and fib_ok:
            try: price_below_ema = float(cl.iloc[idx2]) <= float(e.iloc[idx2]) * 1.01
            except: price_below_ema = True
            if price_below_ema and _vol_ok(idx2):
                s.iloc[idx2] = -1

    # ── 5-Wave Impulse signals ────────────────────────────────────────────────
    if use_5wave and np_ >= 5:
        for k in range(4, np_):
            p0=pivots[k-4]; p1=pivots[k-3]; p2=pivots[k-2]; p3=pivots[k-1]; p4=pivots[k]
            idx4 = p4[0]
            if idx4 >= n: continue

            # Bullish impulse: L-H-L-H-L (each L higher, each H higher)
            if (p0[2]==-1 and p1[2]==1 and p2[2]==-1 and p3[2]==1 and p4[2]==-1 and
                p2[1]>p0[1] and p4[1]>p2[1] and  # higher lows
                p3[1]>p1[1] and                    # higher highs
                p4[1]>p2[1] and _vol_ok(idx4)):    # still rising structure
                # Wave 5 end = entry after impulse completes (price to reverse)
                # Actually this signals end of impulse → take the LONG on the pullback
                # Here we signal at the wave-5 low for continuation after correction
                s.iloc[idx4] = 1

            # Bearish impulse: H-L-H-L-H (each H lower, each L lower)
            elif (p0[2]==1 and p1[2]==-1 and p2[2]==1 and p3[2]==-1 and p4[2]==1 and
                  p2[1]<p0[1] and p4[1]<p2[1] and  # lower highs
                  p3[1]<p1[1] and                    # lower lows
                  _vol_ok(idx4)):
                s.iloc[idx4] = -1

    # Indicator overlays for chart
    fib_lo = pd.Series(np.nan, index=df.index)
    fib_hi = pd.Series(np.nan, index=df.index)
    if np_ >= 2:
        pl = pivots[-1]; pp = pivots[-2]
        leg = abs(pl[1] - pp[1])
        if pl[2] == 1:   # last pivot is high → Fib retracement levels below
            fib_lo.iloc[-1] = pl[1] - leg * fib_max
            fib_hi.iloc[-1] = pl[1] - leg * fib_min
        else:            # last pivot is low → Fib extension levels above
            fib_lo.iloc[-1] = pl[1] + leg * fib_min
            fib_hi.iloc[-1] = pl[1] + leg * fib_max

    return s, {"EMA_trend": e, "Fib_Low": fib_lo, "Fib_High": fib_hi}
def sig_hv_percentile(df, hv_period=20, lookback=252, ob_pct=80, os_pct=20, **_):
    """
    Historical Volatility Percentile — proxy for IV Rank / IV Percentile.
    Computes 20-bar HV as annualised std of log returns.
    Then finds percentile of current HV vs past `lookback` bars.
    LOW HV percentile (<20%) → volatility likely to expand → consider BUY on next up-move.
    HIGH HV percentile (>80%) → volatility likely to contract → consider SELL / mean-revert.
    """
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    hv = log_ret.rolling(hv_period).std() * np.sqrt(252) * 100   # annualised %

    hv_pct = hv.rolling(lookback, min_periods=hv_period).apply(
        lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]) * 100
        if len(x) > 1 else 50, raw=False)

    s = pd.Series(0, index=df.index)
    # Low HV% + price above EMA → buy (volatility expansion upward expected)
    e50 = ema(df["Close"], 50)
    s[(hv_pct < os_pct) & (df["Close"] > e50) & (_cup(df["Close"], e50))] = 1
    # High HV% + price below EMA → sell (volatility contraction, mean revert)
    s[(hv_pct > ob_pct) & (df["Close"] < e50) & (_cdn(df["Close"], e50))] = -1

    return s, {"HV%": hv_pct, "HV": hv, "EMA_50": e50}

# ── ADVANCED STRATEGY: SMC Order Blocks ─────────────────────────────────────
def sig_smc_order_blocks(df, ob_lookback=10, fvg_min_pct=0.05, **_):
    """
    Smart Money Concepts:
    - Bullish Order Block: last bearish candle before a strong bullish impulse
    - Bearish Order Block: last bullish candle before a strong bearish impulse
    - Fair Value Gap (FVG): gap between bar[i-2].High and bar[i].Low (bullish FVG)
    - BOS (Break of Structure): price breaks above recent swing high → bullish
    Signal: price returns to OB / FVG zone → entry
    """
    cl=df["Close"]; op=df["Open"]; hi=df["High"]; lo=df["Low"]
    s=pd.Series(0,index=df.index)
    n=len(df)

    for i in range(ob_lookback+2, n-1):
        # ── Bullish OB: bearish candle before bullish impulse
        if cl.iloc[i]>cl.iloc[i-1] and cl.iloc[i-1]<op.iloc[i-1]:  # prev was bearish
            ob_lo=lo.iloc[i-1]; ob_hi=hi.iloc[i-1]
            if cl.iloc[i] >= ob_hi * (1 + fvg_min_pct/100):  # strong impulse
                # Check if price is currently near OB level
                if ob_lo <= cl.iloc[i] <= ob_hi * 1.01:
                    s.iloc[i] = 1

        # ── Bearish OB: bullish candle before bearish impulse
        if cl.iloc[i]<cl.iloc[i-1] and cl.iloc[i-1]>op.iloc[i-1]:  # prev was bullish
            ob_lo=lo.iloc[i-1]; ob_hi=hi.iloc[i-1]
            if cl.iloc[i] <= ob_lo * (1 - fvg_min_pct/100):
                if ob_lo*0.99 <= cl.iloc[i] <= ob_hi:
                    s.iloc[i] = -1

        # ── Fair Value Gap: gap between hi[i-2] and lo[i]
        if i >= 2:
            fvg_bull = lo.iloc[i] > hi.iloc[i-2]  # bullish FVG
            fvg_bear = hi.iloc[i] < lo.iloc[i-2]  # bearish FVG
            if fvg_bull and cl.iloc[i]>cl.iloc[i-1]: s.iloc[i] = 1
            if fvg_bear and cl.iloc[i]<cl.iloc[i-1]: s.iloc[i] = -1

    e20=ema(df["Close"],20); e50=ema(df["Close"],50)
    return s, {"EMA_20":e20,"EMA_50":e50}

# ── ADVANCED STRATEGY: Price Action Patterns ────────────────────────────────
def sig_price_action(df, pin_bar_ratio=2.0, engulf_pct=0.5, **_):
    """
    Price Action signals:
    - Bullish Pin Bar: long lower shadow, small body near top, shadow >= pin_bar_ratio * body
    - Bearish Pin Bar: long upper shadow, small body near bottom
    - Bullish Engulfing: green candle body fully engulfs prior red candle body
    - Bearish Engulfing: red candle body fully engulfs prior green candle body
    - Doji: body is < engulf_pct% of total candle range
    """
    op=df["Open"]; hi=df["High"]; lo=df["Low"]; cl=df["Close"]
    body=(cl-op).abs(); rng=hi-lo; rng=rng.replace(0,np.nan)
    upper_shadow=hi-pd.concat([cl,op],axis=1).max(axis=1)
    lower_shadow=pd.concat([cl,op],axis=1).min(axis=1)-lo
    s=pd.Series(0,index=df.index)

    # Pin bars
    bull_pin=(lower_shadow>=pin_bar_ratio*body)&(upper_shadow<body)&(cl>op)
    bear_pin=(upper_shadow>=pin_bar_ratio*body)&(lower_shadow<body)&(cl<op)
    # Engulfing
    bull_eng=(cl>op)&(op<=op.shift(1))&(cl>=cl.shift(1))&(cl.shift(1)<op.shift(1))
    bear_eng=(cl<op)&(op>=op.shift(1))&(cl<=cl.shift(1))&(cl.shift(1)>op.shift(1))
    # Doji
    doji=body<(engulf_pct/100)*rng

    s[bull_pin|bull_eng]=1; s[bear_pin|bear_eng]=-1
    e20=ema(df["Close"],20)
    return s,{"EMA_20":e20}

# ── ADVANCED STRATEGY: Breakdown Strategy ───────────────────────────────────
def sig_breakout(df, lookback=20, break_pct=0.2, vol_mult=1.5, **_):
    """
    Breakout Strategy:
    BUY  when price closes ABOVE the `lookback`-bar high with volume surge.
    SELL when price closes BELOW the `lookback`-bar low with volume surge.
    """
    hi_l=df["High"].rolling(lookback).max().shift(1)
    lo_l=df["Low"].rolling(lookback).min().shift(1)
    vol=df.get("Volume",pd.Series(1,index=df.index))
    avg_v=vol.rolling(20).mean().replace(0,np.nan)
    vol_ok=vol>avg_v*vol_mult
    bp=break_pct/100
    s=pd.Series(0,index=df.index)
    s[_cup(df["Close"],hi_l*(1+bp))&vol_ok]=1
    s[_cdn(df["Close"],lo_l*(1-bp))&vol_ok]=-1
    return s,{"Breakout_Hi":hi_l,"Breakout_Lo":lo_l}

# ── ADVANCED STRATEGY: Support & Resistance + EMA ───────────────────────────
def sig_sr_ema(df, sr_lookback=20, ema_period=50, touch_pct=0.3, **_):
    """
    S&R Levels + EMA:
    Support = rolling low over sr_lookback bars.
    Resistance = rolling high over sr_lookback bars.
    BUY  when price touches support AND is above EMA (uptrend) AND bounces up.
    SELL when price touches resistance AND is below EMA (downtrend) AND bounces down.
    touch_pct: how close to S/R level counts as "touch" (%).
    """
    sup=df["Low"].rolling(sr_lookback).min().shift(1)
    res=df["High"].rolling(sr_lookback).max().shift(1)
    e=ema(df["Close"],ema_period); tp=touch_pct/100
    near_sup=(df["Low"]<=sup*(1+tp))&(df["Close"]>sup)
    near_res=(df["High"]>=res*(1-tp))&(df["Close"]<res)
    s=pd.Series(0,index=df.index)
    s[near_sup&(df["Close"]>e)&(df["Close"]>df["Open"])]=1
    s[near_res&(df["Close"]<e)&(df["Close"]<df["Open"])]=-1
    return s,{"Support":sup,"Resistance":res,"EMA":e}

# ── CUSTOM STRATEGY BUILDER ──────────────────────────────────────────────────
_BUILDER_INDS=["Close","EMA_fast","EMA_slow","RSI","MACD","BB_upper","BB_lower",
               "VWAP","Supertrend","ATR","HV","Fixed Value"]

def _get_builder_series(df, ind_name, period=14, fixed_val=0.0):
    """Compute indicator series by name for custom builder."""
    if ind_name=="Close":      return df["Close"]
    if ind_name=="EMA_fast":   return ema(df["Close"],period)
    if ind_name=="EMA_slow":   return ema(df["Close"],period*2)
    if ind_name=="RSI":        return rsi(df["Close"],period)
    if ind_name=="MACD":       m,_,_=macd(df["Close"]); return m
    if ind_name=="BB_upper":   _,_,u=bollinger(df["Close"],period); return u
    if ind_name=="BB_lower":   l,_,_=bollinger(df["Close"],period); return l
    if ind_name=="VWAP":       return vwap_calc(df)
    if ind_name=="Supertrend": line,_=supertrend(df); return line
    if ind_name=="ATR":        return atr(df,period)
    if ind_name=="HV":
        lr=np.log(df["Close"]/df["Close"].shift(1))
        return lr.rolling(period).std()*np.sqrt(252)*100
    if ind_name=="Fixed Value":return pd.Series(float(fixed_val),index=df.index)
    return df["Close"]

def sig_custom_builder(df,
                        ind1="Close", ind1_period=14, ind1_fixed=0.0,
                        condition="crosses above",
                        ind2="EMA_fast", ind2_period=14, ind2_fixed=0.0,
                        signal_dir="Long",
                        use_cond2=False,
                        ind1b="RSI", ind1b_period=14, ind1b_fixed=50.0,
                        cond2_op="is above",
                        ind2b="Fixed Value", ind2b_period=14, ind2b_fixed=50.0,
                        logic="AND",
                        **_):
    """
    Custom Strategy Builder:
    Generates signal when:
      Condition 1: ind1 [crosses above / crosses below / is above / is below] ind2
    Optionally AND/OR:
      Condition 2: ind1b [condition] ind2b
    signal_dir: Long (→1) or Short (→-1)
    """
    s1 = _get_builder_series(df, ind1, ind1_period, ind1_fixed)
    s2 = _get_builder_series(df, ind2, ind2_period, ind2_fixed)

    def _eval_cond(a, b, cond):
        if cond=="crosses above": return _cup(a, b)
        if cond=="crosses below": return _cdn(a, b)
        if cond=="is above":      return a > b
        if cond=="is below":      return a < b
        return a > b

    mask = _eval_cond(s1, s2, condition)

    if use_cond2:
        s1b = _get_builder_series(df, ind1b, ind1b_period, ind1b_fixed)
        s2b = _get_builder_series(df, ind2b, ind2b_period, ind2b_fixed)
        mask2 = _eval_cond(s1b, s2b, cond2_op)
        mask = (mask & mask2) if logic=="AND" else (mask | mask2)

    sig_val = 1 if signal_dir=="Long" else -1
    s = pd.Series(0, index=df.index)
    s[mask] = sig_val
    indics = {}
    if ind1 not in ("Close","Fixed Value"):   indics[f"{ind1}(cond1_A)"] = s1
    if ind2 not in ("Close","Fixed Value"):   indics[f"{ind2}(cond1_B)"] = s2
    return s, indics

def sig_custom(df, **_):
    """
    Manual Custom Strategy — replace body with your own logic.
    Return: (signals: pd.Series[1/-1/0], indicators: dict[name->Series])
    Example:
        fe=ema(df['Close'],9); se=ema(df['Close'],21); r=rsi(df['Close'],14)
        s=pd.Series(0,index=df.index); s[_cup(fe,se)&(r<60)]=1; s[_cdn(fe,se)&(r>40)]=-1
        return s,{'EMA9':fe,'EMA21':se}
    """
    return pd.Series(0,index=df.index),{}

# ── NEW HIGH-PROBABILITY STRATEGIES ──────────────────────────────────────────

def sig_vwap_ema_confluence(df, ema_period=20, dev_pct=0.5, **_):
    """VWAP + EMA Confluence: price above both VWAP and EMA on pullback."""
    vw = vwap_calc(df); e = ema(df["Close"], ema_period)
    d  = dev_pct/100
    s  = pd.Series(0, index=df.index)
    # Long: price dips to EMA from above, VWAP also below price, bullish close
    bull=(df["Low"]<=e)&(df["Close"]>e)&(df["Close"]>vw)&(df["Close"]>df["Open"])
    bear=(df["High"]>=e)&(df["Close"]<e)&(df["Close"]<vw)&(df["Close"]<df["Open"])
    s[bull]=1; s[bear]=-1
    return s,{"VWAP":vw,"EMA":e}

def sig_orb(df, orb_minutes=15, vol_mult=1.3, **_):
    """
    Opening Range Breakout (ORB):
    First orb_minutes of session form the range (approx first N bars).
    BUY  on break above opening range high with volume.
    SELL on break below opening range low with volume.
    Uses first 3 bars as proxy opening range for bar-based data.
    """
    n_orb = max(1, orb_minutes // 5)   # ~3 bars for 5m
    s = pd.Series(0, index=df.index)
    vol = df.get("Volume", pd.Series(1, index=df.index))
    avg_v = vol.rolling(20).mean().replace(0, np.nan)
    # Rolling opening range: high/low of past n_orb bars
    or_hi = df["High"].rolling(n_orb).max().shift(1)
    or_lo = df["Low"].rolling(n_orb).min().shift(1)
    vol_ok = vol > avg_v * vol_mult
    s[_cup(df["Close"], or_hi) & vol_ok] =  1
    s[_cdn(df["Close"], or_lo) & vol_ok] = -1
    return s, {"OR_Hi": or_hi, "OR_Lo": or_lo}

def sig_mean_reversion_bb(df, period=20, std=2.0, rsi_period=14, **_):
    """
    Mean Reversion BB: BB squeeze (width < threshold) then expansion.
    BUY  when price closes above lower band after touching it + RSI oversold.
    SELL when price closes below upper band after touching it + RSI overbought.
    """
    lo, mid, hi = bollinger(df["Close"], period, std)
    r = rsi(df["Close"], rsi_period)
    bw = (hi - lo) / mid.replace(0, np.nan)   # BB width
    squeeze = bw < bw.rolling(50).mean() * 0.8
    s = pd.Series(0, index=df.index)
    s[_cup(df["Close"], lo) & (r < 40) & ~squeeze] =  1
    s[_cdn(df["Close"], hi) & (r > 60) & ~squeeze] = -1
    return s, {"BB_upper": hi, "BB_mid": mid, "BB_lower": lo, "RSI": r}

def sig_trend_momentum(df, ema_period=50, adx_period=14, adx_thresh=25, **_):
    """
    Trend Momentum: ADX confirms strong trend, EMA pullback for entry.
    Long  when ADX>thresh, price>EMA, last candle pulls back to EMA and closes bullish.
    Short when ADX>thresh, price<EMA, last candle rallies to EMA and closes bearish.
    """
    e = ema(df["Close"], ema_period)
    _adx, pdi, ndi = adx_di(df, adx_period)
    at = _cs(adx_thresh, df.index)
    touch_bull = (df["Low"] <= e) & (df["Close"] > e) & (df["Close"] > df["Open"])
    touch_bear = (df["High"] >= e) & (df["Close"] < e) & (df["Close"] < df["Open"])
    s = pd.Series(0, index=df.index)
    s[(_adx > at) & (pdi > ndi) & touch_bull] =  1
    s[(_adx > at) & (ndi > pdi) & touch_bear] = -1
    return s, {"EMA": e, "ADX": _adx}

def sig_gap_and_go(df, gap_pct=0.3, vol_mult=1.5, **_):
    """
    Gap & Go: trades gap-up/gap-down continuation.
    Gap-up (today Open > yesterday Close by gap_pct%) + bullish candle + volume → BUY.
    Gap-down → SELL.
    """
    prev_close = df["Close"].shift(1)
    gap_up   = (df["Open"] - prev_close) / prev_close.replace(0, np.nan) * 100 >= gap_pct
    gap_down = (prev_close - df["Open"]) / prev_close.replace(0, np.nan) * 100 >= gap_pct
    vol = df.get("Volume", pd.Series(1, index=df.index))
    avg_v = vol.rolling(20).mean().replace(0, np.nan)
    vol_ok = vol > avg_v * vol_mult
    bull_candle = df["Close"] > df["Open"]
    bear_candle = df["Close"] < df["Open"]
    s = pd.Series(0, index=df.index)
    s[gap_up   & bull_candle & vol_ok] =  1
    s[gap_down & bear_candle & vol_ok] = -1
    e20 = ema(df["Close"], 20)
    return s, {"EMA_20": e20}

def sig_inside_bar(df, ema_period=50, **_):
    """
    Inside Bar Breakout:
    Inside bar = High < prev High AND Low > prev Low (contained within prior candle).
    Breakout: next candle breaks above inside bar High → BUY.
               next candle breaks below inside bar Low → SELL.
    Trend filter: price above EMA for longs, below EMA for shorts.
    """
    prev_hi = df["High"].shift(1); prev_lo = df["Low"].shift(1)
    is_inside = (df["High"] < prev_hi) & (df["Low"] > prev_lo)
    # Signal fires when current bar breaks out of prior inside bar
    ib_hi = df["High"].shift(1).where(is_inside.shift(1))
    ib_lo = df["Low"].shift(1).where(is_inside.shift(1))
    e = ema(df["Close"], ema_period)
    s = pd.Series(0, index=df.index)
    s[_cup(df["High"], ib_hi.ffill()) & (df["Close"] > e)] =  1
    s[_cdn(df["Low"],  ib_lo.ffill()) & (df["Close"] < e)] = -1
    return s, {"EMA": e}


# ── ELLIOTT WAVE v3 (argrelextrema + user's pattern logic) ───────────────────
def sig_ew_v3(df, wave_lookback=50, order=3, use_ema_filter=True, ema_period=50, **_):
    """
    Elliott Wave v3 — based on user's argrelextrema approach.

    Logic:
      1. Use scipy.signal.argrelextrema to find swing highs/lows across full data
         (order=N means each extreme must be the highest/lowest within N bars either side)
      2. Enforce strict alternation of highs and lows (zigzag)
      3. Look at last 5 extrema: pattern p0 < p1 > p2 < p3 > p4
         (low - HIGH - low - HIGH - low sequence)
         - BUY  if final low (p4) < middle low (p2): descending correction completing
         - SELL if final low (p4) > middle low (p2): ascending structure

    Generates significantly MORE signals than v1/v2 on intraday (5min, 15min)
    because order=3 detects every 7-bar swing, not requiring large % moves.

    Why more trades:
      v1: needs 1% move per leg → ~220pts on Nifty → 3-4 trades/month on 5min
      v3: order=3 → detects swings every few bars → 20-50+ signals/month on 5min

    Falls back to pure bar-based swing if scipy unavailable.
    """
    n  = len(df)
    s  = pd.Series(0, index=df.index)
    e  = ema(df["Close"], ema_period)

    if n < wave_lookback + order * 2:
        return s, {"EMA_v3": e}

    try:
        from scipy.signal import argrelextrema as _are
        _scipy_ok = True
    except ImportError:
        _scipy_ok = False

    if _scipy_ok:
        hi_arr = df["High"].values
        lo_arr = df["Low"].values
        hi_idxs = _are(hi_arr, np.greater_equal, order=order)[0]
        lo_idxs = _are(lo_arr, np.less_equal,    order=order)[0]
        raw = sorted(
            [(i, hi_arr[i],  1) for i in hi_idxs] +
            [(i, lo_arr[i], -1) for i in lo_idxs],
            key=lambda x: x[0]
        )
    else:
        # Fallback: simple N-bar swing detection (same as v2 but smaller window)
        raw = []
        hi = df["High"]; lo = df["Low"]
        for i in range(order, n - order):
            w_hi = hi.iloc[i-order:i+order+1]
            w_lo = lo.iloc[i-order:i+order+1]
            if float(hi.iloc[i]) >= float(w_hi.max()):
                raw.append((i, float(hi.iloc[i]),  1))
            if float(lo.iloc[i]) <= float(w_lo.min()):
                raw.append((i, float(lo.iloc[i]), -1))
        raw.sort(key=lambda x: x[0])

    # Enforce strict alternation (keep more extreme when same direction consecutive)
    alt = []
    last_d = 0
    for (idx, px, d) in raw:
        if d != last_d:
            alt.append((idx, px, d)); last_d = d
        elif alt:
            prev = alt[-1]
            if (d==1 and px > prev[1]) or (d==-1 and px < prev[1]):
                alt[-1] = (idx, px, d)

    # Scan groups of 5 extrema for pattern
    for k in range(4, len(alt)):
        p0,p1,p2,p3,p4 = alt[k-4], alt[k-3], alt[k-2], alt[k-1], alt[k]
        idx4 = p4[0]
        if idx4 >= n: continue

        pp0,pp1,pp2,pp3,pp4 = p0[1],p1[1],p2[1],p3[1],p4[1]

        # User's pattern: low-HIGH-low-HIGH-low (p1 and p3 are highs)
        if pp0 < pp1 and pp1 > pp2 and pp2 < pp3 and pp3 > pp4:
            if use_ema_filter:
                _ema_val = float(e.iloc[idx4]) if idx4 < len(e) else float(e.iloc[-1])
                _cl_val  = float(df["Close"].iloc[idx4])
            else:
                _ema_val = 0; _cl_val = 1   # bypass filter
            # BUY: final low lower than mid low → wave 5 of corrective completing → reverse up
            if pp4 < pp2 and _cl_val >= _ema_val * 0.99:
                s.iloc[idx4] = 1
            # SELL: final "low" actually higher than mid low → bearish exhaustion
            elif pp4 > pp2 and _cl_val <= _ema_val * 1.01:
                s.iloc[idx4] = -1

    return s, {"EMA_v3": e}


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
    "Elliott Wave (Simplified)":sig_elliott_wave,
    "Elliott Wave v2 (Swing+Fib)":sig_elliott_wave_v2,
    "Elliott Wave v3 (Extrema)":sig_ew_v3,
    "HV Percentile (IV Proxy)":sig_hv_percentile,
    "SMC Order Blocks":sig_smc_order_blocks,
    "Price Action Patterns":sig_price_action,
    "Breakout Strategy":sig_breakout,
    "Support & Resistance + EMA":sig_sr_ema,
    # ── New high-probability strategies ──────────────────────────────────────
    "VWAP + EMA Confluence":    sig_vwap_ema_confluence,
    "Opening Range Breakout (ORB)": sig_orb,
    "Mean Reversion Bollinger": sig_mean_reversion_bb,
    "Trend Momentum (ADX+EMA)": sig_trend_momentum,
    "Gap & Go":                 sig_gap_and_go,
    "Inside Bar Breakout":      sig_inside_bar,
    # ── Custom ───────────────────────────────────────────────────────────────
    "Custom Strategy Builder":sig_custom_builder,
    "Custom Strategy":sig_custom,
}
# ── SL / TARGET ENGINE ────────────────────────────────────────────────────────
def _sw_lo(df,idx,lb=5): return float(df["Low"].iloc[max(0,idx-lb):idx].min()) if idx>0 else float(df["Low"].iloc[0])
def _sw_hi(df,idx,lb=5): return float(df["High"].iloc[max(0,idx-lb):idx].max()) if idx>0 else float(df["High"].iloc[0])
def _atr_at(df,idx,p=14):
    v=atr(df,p).iloc[idx]; return float(v) if not np.isnan(v) else 10.0

def init_sl(df,idx,entry,direction,sl_type,sl_pts,params):
    lb=params.get("swing_lookback",5); am=params.get("atr_mult_sl",1.5); av=_atr_at(df,idx); d=direction
    rr=params.get("rr_ratio",2.0)
    if sl_type=="Custom Points":             return entry-d*sl_pts
    if sl_type=="Trailing SL (Points)":      return entry-d*sl_pts
    if sl_type=="Trailing Prev Candle Low/High":
        return float(df["Low"].iloc[max(0,idx-1)]) if d==1 else float(df["High"].iloc[max(0,idx-1)])
    if sl_type=="Trailing Curr Candle Low/High":
        return float(df["Low"].iloc[idx]) if d==1 else float(df["High"].iloc[idx])
    if sl_type in("Trailing Prev Swing Low/High","Trailing Curr Swing Low/High"):
        return _sw_lo(df,idx,lb) if d==1 else _sw_hi(df,idx,lb)
    if sl_type=="Cost to Cost (Breakeven)":  return entry-d*sl_pts
    if sl_type=="Cost-to-Cost K-Shift Trailing": return entry-d*sl_pts   # initial same as custom
    if sl_type=="EMA Reverse Crossover":     return entry-d*sl_pts
    if sl_type=="ATR Based":                 return entry-d*am*av
    if sl_type=="Risk/Reward Based SL":
        # SL derived from target / RR ratio; tgt_pts passed via params
        tgt_pts=params.get("tgt_pts_for_sl", sl_pts*rr)
        return entry-d*(tgt_pts/rr)
    if sl_type=="Strategy Signal Exit":      return entry-d*sl_pts  # placeholder; exit handled via signal
    return entry-d*sl_pts

def update_sl(df,j,entry,direction,sl_type,sl_pts,cur_sl,params):
    lb=params.get("swing_lookback",5); d=direction
    bh=float(df["High"].iloc[j]); bl=float(df["Low"].iloc[j])
    kn=params.get("kshift_n",sl_pts)     # trigger pts (default = sl_pts)
    kk=params.get("kshift_k",sl_pts/2)   # lock-in pts from entry
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
    if sl_type=="Cost-to-Cost K-Shift Trailing":
        """
        Phase 1: price hasn't moved kn points in favor → hold initial SL
        Phase 2: price moved kn points → lock SL at entry+kk (for LONG)
        Phase 3: trail normally at price-sl_pts (but floor = lock level)
        """
        lock_level = entry + d*kk   # entry+kk for LONG, entry-kk for SHORT
        trigger    = entry + d*kn
        if d==1:
            if bh>=trigger:  # lock triggered or already locked
                normal_trail = bh - sl_pts
                return max(cur_sl, lock_level, normal_trail) if bh>trigger else max(cur_sl,lock_level)
            return cur_sl
        else:
            if bl<=trigger:
                normal_trail = bl + sl_pts
                return min(cur_sl, lock_level, normal_trail) if bl<trigger else min(cur_sl,lock_level)
            return cur_sl
    if sl_type=="ATR Based":     return cur_sl   # ATR SL is fixed
    if sl_type=="Risk/Reward Based SL": return cur_sl
    return cur_sl

def init_tgt(df,idx,entry,direction,tgt_type,tgt_pts,sl,params):
    lb=params.get("swing_lookback",5); am=params.get("atr_mult_tgt",2.0); rr=params.get("rr_ratio",2.0)
    av=_atr_at(df,idx); d=direction
    if tgt_type=="Custom Points":                        return entry+d*tgt_pts
    if tgt_type=="Trailing Target (Display Only)":       return entry+d*tgt_pts
    if tgt_type in("Trailing Prev Candle High/Low","Trailing Curr Candle High/Low"):
        return float(df["High"].iloc[idx]) if d==1 else float(df["Low"].iloc[idx])
    if tgt_type in("Trailing Prev Swing High/Low","Trailing Curr Swing High/Low"):
        return _sw_hi(df,idx,lb) if d==1 else _sw_lo(df,idx,lb)
    if tgt_type in("ATR Based","Risk/Reward Based"): return entry+d*am*av if tgt_type=="ATR Based" else entry+d*rr*abs(entry-sl)
    if tgt_type=="Reverse EMA Crossover":  return entry+d*tgt_pts   # placeholder; exit driven by signal
    if tgt_type=="Strategy Signal Exit":   return entry+d*tgt_pts   # placeholder; exit driven by signal
    return entry+d*tgt_pts

def update_tgt(df,j,direction,tgt_type,tgt_pts,cur_tgt,params,sigs=None):
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
    if tgt_type in("Reverse EMA Crossover","Strategy Signal Exit"):
        # Exit when strategy signal reverses — handled externally in backtest loop
        return cur_tgt, False   # never triggers via price; exits via signal reversal
    return cur_tgt,True

# ── BACKTEST ENGINE ────────────────────────────────────────────────────────────
def run_backtest(df, strategy, params, sl_type, sl_pts, tgt_type, tgt_pts,
                  tw_from_str=None, tw_to_str=None):
    """
    Bar i   → signal on CLOSE (bar fully closed).
               If time_window active: signal only valid if bar time is within window.
    Bar i+1 → entry at OPEN; SL & Target set
    Bar i+1+ → SL checked FIRST (conservative), then Target
    Both breach same candle → SL wins.
    tw_from_str / tw_to_str: "HH:MM" strings for IST time window (None = disabled).
    """
    fn=STRATEGY_FN.get(strategy,sig_custom)
    try: sigs,indics=fn(df,**params)
    except: return [],[],[]

    # Parse time window if provided
    _tw_active = (tw_from_str is not None and tw_to_str is not None)
    _tw_from = _tw_to = None
    if _tw_active:
        try:
            _tw_from = datetime.strptime(tw_from_str,"%H:%M").time()
            _tw_to   = datetime.strptime(tw_to_str,"%H:%M").time()
        except: _tw_active = False

    def _bar_in_window(bar_dt):
        if not _tw_active: return True
        try:
            bt = pd.Timestamp(bar_dt)
            if bt.tzinfo is not None:
                import pytz; bt=bt.tz_convert(pytz.timezone("Asia/Kolkata"))
            t=bt.time(); return _tw_from<=t<=_tw_to
        except: return True

    n=len(df); trades=[]; audit_trail=[]; i=0
    while i<n-1:
        sig=int(sigs.iloc[i])
        # Time window: skip signals outside trading hours
        if sig!=0 and _tw_active and not _bar_in_window(df.index[i]):
            i+=1; continue
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
            disp_tgt,tf=update_tgt(df,j,direction,tgt_type,tgt_pts,disp_tgt,params,sigs)
            if tf: tgt=disp_tgt
            # Signal-based exit: fires when strategy signal reverses direction
            _sig_exit = (sl_type in ("EMA Reverse Crossover","Strategy Signal Exit") or
                         tgt_type in ("Reverse EMA Crossover","Strategy Signal Exit"))
            if _sig_exit:
                rev=int(sigs.iloc[j])
                if rev!=0 and rev!=direction:
                    exit_bar,exit_px,exit_why=j,float(df["Open"].iloc[j]),"Strategy Signal Exit"; exited=True
            # Time window: force exit if outside hours
            if not exited and _tw_active and not _bar_in_window(df.index[j]):
                exit_bar,exit_px,exit_why=j,float(df["Close"].iloc[j]),"Time Window Exit"; exited=True
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
    # ── Advanced strategies ─────────────────────────────────────────────────
    "Elliott Wave (Simplified)":{"swing_lookback_ew":[5,10,15],"min_wave_pct":[0.5,1.0,1.5]},
    "Elliott Wave v2 (Swing+Fib)":{"swing_bars":[3,5,8,10],"fib_min":[0.382,0.5],"fib_max":[0.786,0.886],"ema_period":[20,50]},
    "Elliott Wave v3 (Extrema)":{"wave_lookback":[30,50,80],"order":[2,3,5],"ema_period":[20,50,100]},
    "HV Percentile (IV Proxy)":{"hv_period":[10,20,30],"ob_pct":[70,80],"os_pct":[20,30]},
    "SMC Order Blocks":{"ob_lookback":[5,10,20],"fvg_min_pct":[0.03,0.05,0.1]},
    "Price Action Patterns":{"pin_bar_ratio":[1.5,2.0,3.0],"engulf_pct":[0.3,0.5,0.7]},
    "Breakout Strategy":{"lookback":[10,15,20,30],"break_pct":[0.1,0.2,0.3],"vol_mult":[1.2,1.5,2.0]},
    "Support & Resistance + EMA":{"sr_lookback":[10,20,30],"ema_period":[20,50,100],"touch_pct":[0.2,0.3,0.5]},
    "VWAP Deviation":{"dev_pct":[0.5,1.0,1.5,2.0]},
    "Ichimoku Cloud":{"tenkan":[7,9,12],"kijun":[20,26,30]},
    "RSI Divergence":{"period":[10,14,21],"lookback":[3,5,8]},
    "Heikin Ashi EMA":{"ema_period":[10,20,30,50]},
    "Volume Price Trend (VPT)":{"vpt_ema_period":[9,14,21]},
    "Price Threshold Cross":{"threshold":[0.0]},   # threshold set by user; grid over direction
}
_BP={"atr_mult_sl":1.5,"atr_mult_tgt":2.0,"rr_ratio":2.0,"swing_lookback":5}

# Full default params for every strategy — optimization uses these as base so
# results exactly match what backtesting produces with sidebar defaults.
_STRATEGY_DEFAULTS = {
    "EMA Crossover":            {"fast":9,"slow":15},
    "RSI Overbought/Oversold":  {"period":14,"ob":70,"os_":30},
    "Simple Buy":               {},
    "Simple Sell":              {},
    "Price Threshold Cross":    {"threshold":0.,"thresh_dir":"Above","thresh_action":"Buy"},
    "Bollinger Bands":          {"period":20,"std":2.0},
    "RSI Divergence":           {"period":14,"lookback":5},
    "MACD Crossover":           {"fast":12,"slow":26,"signal":9},
    "Supertrend":               {"period":7,"multiplier":3.0},
    "ADX + DI Crossover":       {"period":14,"adx_thresh":25},
    "Stochastic Oscillator":    {"k":14,"d":3,"ob":80,"os_":20},
    "VWAP Deviation":           {"dev_pct":1.0},
    "Ichimoku Cloud":           {"tenkan":9,"kijun":26},
    "BB + RSI Mean Reversion":  {"bb_period":20,"bb_std":2.0,"rsi_period":14,"rsi_os":35,"rsi_ob":65},
    "Donchian Breakout":        {"period":20},
    "Triple EMA Trend":         {"f":9,"m":21,"s_":50},
    "Heikin Ashi EMA":          {"ema_period":20},
    "Volume Price Trend (VPT)": {"vpt_ema_period":14},
    "Keltner Channel Breakout": {"ema_p":20,"atr_p":10,"mult":2.0},
    "Williams %R Reversal":     {"period":14,"ob":-20,"os_":-80},
    "Swing Trend + Pullback":   {"trend_ema":50,"entry_ema":9,"rsi_period":14,
                                  "rsi_bull_min":40,"rsi_bull_max":65,
                                  "rsi_bear_min":35,"rsi_bear_max":60,"vol_mult":1.2},
    "Elliott Wave (Simplified)":{"swing_lookback_ew":10,"min_wave_pct":1.0},
    "Elliott Wave v2 (Swing+Fib)":{"swing_bars":5,"fib_min":0.382,"fib_max":0.886,"ema_period":50,"use_volume":True,"use_5wave":True},
    "Elliott Wave v3 (Extrema)":{"wave_lookback":50,"order":3,"use_ema_filter":True,"ema_period":50},
    "HV Percentile (IV Proxy)": {"hv_period":20,"lookback":252,"ob_pct":80,"os_pct":20},
    "SMC Order Blocks":         {"ob_lookback":10,"fvg_min_pct":0.05},
    "Price Action Patterns":    {"pin_bar_ratio":2.0,"engulf_pct":0.5},
    "Breakout Strategy":        {"lookback":20,"break_pct":0.2,"vol_mult":1.5},
    "Support & Resistance + EMA":{"sr_lookback":20,"ema_period":50,"touch_pct":0.3},
}

def optimize(df,strategy,sl_type,sl_pts,tgt_type,tgt_pts,desired_acc,min_pts,min_trades,progress_cb=None,
             tw_from_str=None,tw_to_str=None):
    """
    Uses same run_backtest logic as the Backtesting tab.
    Base params = _STRATEGY_DEFAULTS[strategy] merged with _BP,
    then individual grid params override — ensures exact match with backtest.
    """
    # Build full base params: strategy defaults + common defaults
    _strat_defaults = {**_STRATEGY_DEFAULTS.get(strategy, {}), **_BP}

    grid = PARAM_GRIDS.get(strategy)
    if not grid:
        # No grid: run single backtest with full defaults
        t,_,_ = run_backtest(df, strategy, _strat_defaults, sl_type, sl_pts, tgt_type, tgt_pts,
                              tw_from_str, tw_to_str)
        p = calc_perf(t)
        if p: return [{"params":_strat_defaults,**p,
                       "Meets_Accuracy":p.get("Accuracy (%)",0)>=desired_acc,
                       "Meets_Pts":p.get("Total Pts Won",0)>=min_pts}]
        return []

    keys   = list(grid.keys())
    combos = list(itertools_product(*[grid[k] for k in keys]))
    total  = len(combos)
    results = []
    for idx, combo in enumerate(combos):
        # Start from full strategy defaults, then override with grid combo values
        p = {**_strat_defaults, **dict(zip(keys, combo))}
        try:
            t,_,_ = run_backtest(df, strategy, p, sl_type, sl_pts, tgt_type, tgt_pts,
                                  tw_from_str, tw_to_str)
            perf = calc_perf(t)
            if perf.get("Total Trades",0) >= min_trades:
                results.append({"params":p, **perf,
                    "Meets_Accuracy": perf.get("Accuracy (%)",0)   >= desired_acc,
                    "Meets_Pts":      perf.get("Total Pts Won",0)  >= min_pts})
        except: pass
        if progress_cb: progress_cb(min((idx+1)/total, 1.0))
    results.sort(key=lambda r:(-r.get("Accuracy (%)",0), -r.get("Total PnL",0)))
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

# ── STRATEGY PARAMS UI  (single-column, sidebar-safe) ────────────────────────
def strategy_params_ui(strategy, prefix, applied=None):
    """
    Renders strategy parameter inputs.
    Uses FULL-WIDTH inputs (no st.columns) so it works in the sidebar
    without truncation on desktop or mobile.
    """
    def _n(label, lo, hi, default, key, step=1, fmt="%.4g", help=None):
        try: v = float(applied.get(key.split("_",1)[-1], default)) if applied else float(default)
        except: v = float(default)
        return st.number_input(label, float(lo), float(hi), v, step=float(step), format=fmt, key=key, help=help)

    def _sel(label, opts, default, key):
        idx = opts.index(default) if default in opts else 0
        cur = st.session_state.get(key, default)
        idx = opts.index(cur) if cur in opts else idx
        return st.selectbox(label, opts, index=idx, key=key)

    p = {}
    if strategy == "EMA Crossover":
        p["fast"] = int(_n("EMA Fast period", 2, 200, 9,  f"{prefix}_fast"))
        p["slow"] = int(_n("EMA Slow period", 2, 500, 15, f"{prefix}_slow"))

    elif strategy == "RSI Overbought/Oversold":
        st.caption("🟢 BUY: RSI crosses **above** Oversold  |  🔴 SELL: RSI crosses **below** Overbought")
        p["period"] = int(_n("RSI Period",          2, 100, 14, f"{prefix}_rp"))
        p["ob"]     = int(_n("Overbought (sell ↓)", 50, 95, 70, f"{prefix}_ob"))
        p["os_"]    = int(_n("Oversold  (buy ↑)",   5,  50, 30, f"{prefix}_os"))

    elif strategy == "Price Threshold Cross":
        p["threshold"]     = float(_n("Threshold Price", 0, 1e9, 0, f"{prefix}_thresh", step=0.01, fmt="%.2f"))
        p["thresh_dir"]    = _sel("Trigger Direction", ["Above","Below","Both"], "Above", f"{prefix}_tdir")
        p["thresh_action"] = _sel("Signal Action",     ["Buy","Sell","Both"],    "Buy",   f"{prefix}_tact")
        st.caption(
            f"When price crosses **{p['thresh_dir']}** {p['threshold']:.2f} → **{p['thresh_action']}** signal"
        )

    elif strategy == "Bollinger Bands":
        p["period"] = int  (_n("BB Period", 5, 200, 20, f"{prefix}_bbp"))
        p["std"]    = float(_n("Std Dev",  0.5, 5,  2,  f"{prefix}_bbs", step=0.1))

    elif strategy == "RSI Divergence":
        p["period"]   = int(_n("RSI Period",    2, 100, 14, f"{prefix}_rp2"))
        p["lookback"] = int(_n("Lookback bars", 2,  50,  5, f"{prefix}_lb"))

    elif strategy == "MACD Crossover":
        p["fast"]   = int(_n("MACD Fast",   2, 100, 12, f"{prefix}_mf"))
        p["slow"]   = int(_n("MACD Slow",   5, 200, 26, f"{prefix}_ms"))
        p["signal"] = int(_n("MACD Signal", 2, 100,  9, f"{prefix}_msig"))

    elif strategy == "Supertrend":
        p["period"]     = int  (_n("Period",     2,  50,  7, f"{prefix}_stp"))
        p["multiplier"] = float(_n("Multiplier", 0.5,10,  3, f"{prefix}_stm", step=0.5))

    elif strategy == "ADX + DI Crossover":
        p["period"]     = int(_n("ADX Period",    2,  50, 14, f"{prefix}_ap"))
        p["adx_thresh"] = int(_n("ADX Threshold",10,  50, 25, f"{prefix}_at"))

    elif strategy == "Stochastic Oscillator":
        p["k"]   = int(_n("Stoch %K",   2,  50, 14, f"{prefix}_k"))
        p["d"]   = int(_n("Stoch %D",   2,  20,  3, f"{prefix}_d"))
        p["ob"]  = int(_n("Overbought", 50, 95, 80, f"{prefix}_so"))
        p["os_"] = int(_n("Oversold",    5, 50, 20, f"{prefix}_su"))

    elif strategy == "VWAP Deviation":
        p["dev_pct"] = float(_n("Deviation %", 0.1, 10, 1, f"{prefix}_vd", step=0.1))

    elif strategy == "Ichimoku Cloud":
        p["tenkan"] = int(_n("Tenkan period",  2,  50,  9, f"{prefix}_it"))
        p["kijun"]  = int(_n("Kijun period",   5, 100, 26, f"{prefix}_ik"))

    elif strategy == "BB + RSI Mean Reversion":
        p["bb_period"]  = int  (_n("BB Period",  5, 100, 20, f"{prefix}_brbbp"))
        p["bb_std"]     = float(_n("BB Std Dev", 0.5, 5,  2, f"{prefix}_brbbs", step=0.1))
        p["rsi_period"] = int  (_n("RSI Period", 2,  50, 14, f"{prefix}_brrp"))
        p["rsi_os"]     = int  (_n("RSI Oversold",  5, 50, 35, f"{prefix}_bro"))
        p["rsi_ob"]     = int  (_n("RSI Overbought",50, 95, 65, f"{prefix}_brob"))

    elif strategy == "Donchian Breakout":
        p["period"] = int(_n("Channel Period", 5, 200, 20, f"{prefix}_dp"))

    elif strategy == "Triple EMA Trend":
        p["f"]  = int(_n("EMA1 Fast",  2,  50,  9, f"{prefix}_tf"))
        p["m"]  = int(_n("EMA2 Mid",   5, 100, 21, f"{prefix}_tm"))
        p["s_"] = int(_n("EMA3 Slow", 10, 300, 50, f"{prefix}_ts"))

    elif strategy == "Heikin Ashi EMA":
        p["ema_period"] = int(_n("EMA Period", 5, 200, 20, f"{prefix}_hap"))

    elif strategy == "Volume Price Trend (VPT)":
        p["vpt_ema_period"] = int(_n("Signal EMA", 2, 100, 14, f"{prefix}_vp"))

    elif strategy == "Keltner Channel Breakout":
        p["ema_p"] = int  (_n("EMA Period",  5, 100, 20, f"{prefix}_kep"))
        p["atr_p"] = int  (_n("ATR Period",  2,  50, 10, f"{prefix}_kap"))
        p["mult"]  = float(_n("Multiplier", 0.5,  5,  2, f"{prefix}_km", step=0.25))

    elif strategy == "Williams %R Reversal":
        p["period"] = int(_n("Period",   2,  50,  14, f"{prefix}_wrp"))
        p["ob"]     = int(_n("OB level", -5,  -1, -20, f"{prefix}_wrob"))
        p["os_"]    = int(_n("OS level",-99, -50, -80, f"{prefix}_wros"))

    elif strategy == "Swing Trend + Pullback":
        st.caption("Trend EMA filter + pullback to Entry EMA + RSI zone + volume surge")
        p["trend_ema"]    = int  (_n("Trend EMA period", 10, 200, 50, f"{prefix}_ste"))
        p["entry_ema"]    = int  (_n("Entry EMA period",  2,  50,  9, f"{prefix}_see"))
        p["rsi_period"]   = int  (_n("RSI Period",        2,  50, 14, f"{prefix}_srp"))
        p["rsi_bull_min"] = int  (_n("RSI Bull Min",     20,  60, 40, f"{prefix}_sbmin"))
        p["rsi_bull_max"] = int  (_n("RSI Bull Max",     50,  90, 65, f"{prefix}_sbmax"))
        p["rsi_bear_min"] = int  (_n("RSI Bear Min",     20,  60, 35, f"{prefix}_snmin"))
        p["rsi_bear_max"] = int  (_n("RSI Bear Max",     50,  90, 60, f"{prefix}_snmax"))
        p["vol_mult"]     = float(_n("Volume Multiplier",0.5,   5,1.2, f"{prefix}_svm", step=0.1))

    elif strategy == "Elliott Wave (Simplified)":
        st.success(
            "✅ **Works for BOTH LONG and SHORT signals.**\n\n"
            "LONG: LOW → HIGH → HIGHER LOW (bullish correction completing).\n\n"
            "SHORT: HIGH → LOW → LOWER HIGH (bearish correction completing)."
        )
        st.caption(
            "**For 5-min charts:** Use min_wave_pct = 0.3–0.5% (not 1.0%). "
            "At 1.0%, Nifty needs ~220pts per leg → very few signals. "
            "At 0.3%, you get 5-10x more signals. "
            "Live trading lookback = 20 bars, so signals up to ~20 closed bars ago are caught."
        )
        p["swing_lookback_ew"] = int  (_n("Swing lookback",3,30,10,f"{prefix}_ewlb"))
        p["min_wave_pct"]      = float(_n("Min wave move % (use 0.3-0.5 for 5min, 1.0 for daily)",
                                           0.05,10,0.5,f"{prefix}_ewpct",step=0.05))

    elif strategy == "Elliott Wave v2 (Swing+Fib)":
        st.caption(
            "V2 uses **bar-count swing detection** (not % move) — produces 10-30x more signals on 5min charts. "
            "Fibonacci filter keeps only high-probability setups."
        )
        p["swing_bars"]  = int  (_n("Swing bars (pivot confirmation)",  2, 20,  5, f"{prefix}_ewv2sb"))
        p["fib_min"]     = float(_n("Fib min retracement (e.g. 0.382)", 0.1, 0.7, 0.382, f"{prefix}_ewv2fmin", step=0.001, fmt="%.3f"))
        p["fib_max"]     = float(_n("Fib max retracement (e.g. 0.886)", 0.5, 1.0, 0.886, f"{prefix}_ewv2fmax", step=0.001, fmt="%.3f"))
        p["ema_period"]  = int  (_n("Trend EMA period",                  5, 200, 50, f"{prefix}_ewv2ep"))
        p["use_volume"]  = st.checkbox("Volume confirmation filter", value=True,  key=f"{prefix}_ewv2vol")
        p["use_5wave"]   = st.checkbox("Also detect 5-wave impulse",   value=True,  key=f"{prefix}_ewv25w")
        st.caption(
            f"Fib zone: **{p['fib_min']*100:.1f}% – {p['fib_max']*100:.1f}%** retracement of prior leg. "
            f"Swing pivot confirmed when price is highest/lowest in **{p['swing_bars']}** bars each side."
        )

    elif strategy == "Elliott Wave v3 (Extrema)":
        st.caption(
            "**V3** — argrelextrema swing detection (your logic). order=3 finds a swing every ~7 bars "
            "→ 20-50+ trades/month on 5min. Install scipy: `pip install scipy` "
            "(falls back to bar-based swings if scipy not available)."
        )
        p["wave_lookback"]  = int (_n("Lookback bars",       20, 300, 50, f"{prefix}_ewv3lb"))
        p["order"]          = int (_n("Extrema order N (bars each side, lower=more signals)", 2, 15, 3, f"{prefix}_ewv3ord"))
        st.caption(f"order={p.get('order',3)} → pivot confirmed in a {p.get('order',3)*2+1}-bar window. "
                   "Lower = more signals. order=2 gives ~5-bar swings; order=3 gives ~7-bar swings.")
        p["ema_period"]     = int (_n("Trend EMA period",     5, 200, 50, f"{prefix}_ewv3ep"))
        p["use_ema_filter"] = st.checkbox("EMA trend filter (long only above EMA, short only below)",
                                           value=True, key=f"{prefix}_ewv3emaf")
        st.caption(
            f"Pattern: low–HIGH–low–HIGH–**low** sequence. "
            f"BUY when final low < mid low (corrective structure completing). "
            f"order={p['order']} → pivot confirmed in {p['order']*2+1}-bar window."
        )

    elif strategy == "HV Percentile (IV Proxy)":
        p["hv_period"]  = int  (_n("HV Period",   5,100,20,f"{prefix}_hvp"))
        p["lookback"]   = int  (_n("Lookback",    50,500,252,f"{prefix}_hvlb"))
        p["ob_pct"]     = float(_n("High HV%",   60,99,80,f"{prefix}_hvob"))
        p["os_pct"]     = float(_n("Low HV%",    1,40,20,f"{prefix}_hvos"))
        st.caption("Low HV% (<20%) + price above EMA → BUY (expect expansion).  High HV% (>80%) → SELL (contraction).")

    elif strategy == "SMC Order Blocks":
        p["ob_lookback"] = int  (_n("OB Lookback",3,50,10,f"{prefix}_smcob"))
        p["fvg_min_pct"] = float(_n("Min impulse %",0.01,5,0.05,f"{prefix}_smcfvg",step=0.01))

    elif strategy == "Price Action Patterns":
        p["pin_bar_ratio"] = float(_n("Pin bar shadow:body ratio",1.5,10,2.0,f"{prefix}_pbr",step=0.1))
        p["engulf_pct"]    = float(_n("Doji body threshold %",0.1,10,0.5,f"{prefix}_epct",step=0.1))

    elif strategy == "Breakdown Strategy":
        # kept for backward compat — redirects to Breakout Strategy
        p["lookback"]  = int  (_n("Channel Period",5,100,20,f"{prefix}_bdlb"))
        p["break_pct"] = float(_n("Break Threshold %",0.01,5,0.2,f"{prefix}_bdpct",step=0.05))
        p["vol_mult"]  = float(_n("Volume Multiplier",1.0,5,1.5,f"{prefix}_bdvol",step=0.1))

    elif strategy == "Breakout Strategy":
        p["lookback"]  = int  (_n("Channel Period (bars)",5,100,20,f"{prefix}_bklb"))
        p["break_pct"] = float(_n("Break Threshold %",0.01,5,0.2,f"{prefix}_bkpct",step=0.05))
        p["vol_mult"]  = float(_n("Volume Multiplier",1.0,5,1.5,f"{prefix}_bkvol",step=0.1))
        st.caption("BUY: close above N-bar high with volume surge. SELL: close below N-bar low.")

    elif strategy == "Support & Resistance + EMA":
        p["sr_lookback"] = int  (_n("S&R Period",5,200,20,f"{prefix}_srlb"))
        p["ema_period"]  = int  (_n("EMA Period", 5,200,50,f"{prefix}_srema"))
        p["touch_pct"]   = float(_n("Touch threshold %",0.01,5,0.3,f"{prefix}_srtp",step=0.05))

    elif strategy == "VWAP + EMA Confluence":
        p["ema_period"] = int  (_n("EMA Period",5,200,20,f"{prefix}_vece"))
        p["dev_pct"]    = float(_n("VWAP Deviation %",0.1,5,0.5,f"{prefix}_vedev",step=0.1))

    elif strategy == "Opening Range Breakout (ORB)":
        p["orb_minutes"] = int  (_n("Opening Range (minutes)",5,60,15,f"{prefix}_orbm",step=5))
        p["vol_mult"]    = float(_n("Volume Multiplier",1.0,5,1.3,f"{prefix}_orbv",step=0.1))
        st.caption("ORB: BUY above first N-min high, SELL below first N-min low with volume surge.")

    elif strategy == "Mean Reversion Bollinger":
        p["period"]     = int  (_n("BB Period",5,200,20,f"{prefix}_mrbp"))
        p["std"]        = float(_n("Std Dev",0.5,5,2.0,f"{prefix}_mrbs",step=0.1))
        p["rsi_period"] = int  (_n("RSI Period",2,50,14,f"{prefix}_mrrp"))

    elif strategy == "Trend Momentum (ADX+EMA)":
        p["ema_period"]  = int(_n("EMA Period",5,200,50,f"{prefix}_tmep"))
        p["adx_period"]  = int(_n("ADX Period",2,50,14,f"{prefix}_tmap"))
        p["adx_thresh"]  = int(_n("ADX Threshold",10,50,25,f"{prefix}_tmat"))

    elif strategy == "Gap & Go":
        p["gap_pct"]  = float(_n("Gap % threshold",0.1,10,0.3,f"{prefix}_ggpct",step=0.1))
        p["vol_mult"] = float(_n("Volume Multiplier",1.0,5,1.5,f"{prefix}_ggvol",step=0.1))
        st.caption("Gap-up/gap-down continuation. Filters with volume surge.")

    elif strategy == "Inside Bar Breakout":
        p["ema_period"] = int(_n("Trend EMA Period",5,200,50,f"{prefix}_ibep"))
        st.caption("Inside bar forms => breakout above/below triggers with trend filter.")

    elif strategy == "Custom Strategy":
        st.markdown("""
**Manual Custom Strategy** - edit sig_custom() in the source file.

Available helpers: ema(s,p), sma(s,p), rsi(s,p), atr(df,p),
bollinger(s,p,k), macd(s,f,sl,sig), stoch(df,k,d), vwap_calc(df),
_cup(a,b) (cross up), _cdn(a,b) (cross down)

Example in sig_custom:
    fe=ema(df['Close'],9); se=ema(df['Close'],21); r=rsi(df['Close'],14)
    s=pd.Series(0,index=df.index)
    s[_cup(fe,se)&(r<60)]=1
    s[_cdn(fe,se)&(r>40)]=-1
    return s, {'EMA9':fe,'EMA21':se}
        """)
        st.info("No parameters here - configure logic in code, then click Run Backtest.")

    elif strategy == "Custom Strategy Builder":
        st.caption("Build signal rules visually — up to 5 conditions with AND/OR logic.")
        _INDS  = _BUILDER_INDS
        _CONDS = ["crosses above","crosses below","is above","is below"]

        _nckey = f"{prefix}_cb_nconds"
        if _nckey not in st.session_state:
            st.session_state[_nckey] = 1

        _ca, _cb = st.columns(2)
        if _ca.button("➕ Add condition", key=f"{prefix}_cb_add"):
            if st.session_state[_nckey] < 5:
                st.session_state[_nckey] = int(st.session_state[_nckey]) + 1
        if _cb.button("➖ Remove last",   key=f"{prefix}_cb_rem"):
            if st.session_state[_nckey] > 1:
                st.session_state[_nckey] = int(st.session_state[_nckey]) - 1

        n_conds  = int(st.session_state[_nckey])
        cond_list = []
        for ci in range(n_conds):
            with st.expander(f"Condition {ci+1}", expanded=(ci == 0)):
                logic = st.selectbox(f"Logic (join cond {ci} → {ci+1})",
                                      ["AND","OR"], key=f"{prefix}_lg{ci}") if ci>0 else "AND"
                a_ind    = st.selectbox(f"Indicator A",    _INDS,  key=f"{prefix}_ai{ci}")
                a_period = int  (_n(f"Period A",  2,500,14, f"{prefix}_ap{ci}"))
                a_fixed  = float(_n(f"Fixed A",   0,1e9, 0, f"{prefix}_af{ci}",step=0.01,fmt="%.2f"))
                cond_op  = st.selectbox(f"Condition", _CONDS, key=f"{prefix}_op{ci}")
                b_ind    = st.selectbox(f"Indicator B",    _INDS,  key=f"{prefix}_bi{ci}")
                b_period = int  (_n(f"Period B",  2,500,14, f"{prefix}_bp{ci}"))
                b_fixed  = float(_n(f"Fixed B",   0,1e9, 0, f"{prefix}_bf{ci}",step=0.01,fmt="%.2f"))
                st.caption(f"**{a_ind}** {cond_op} **{b_ind}**")
                cond_list.append({"logic":logic,"a_ind":a_ind,"a_period":a_period,"a_fixed":a_fixed,
                                   "cond_op":cond_op,"b_ind":b_ind,"b_period":b_period,"b_fixed":b_fixed})

        p["signal_dir"] = st.selectbox("Overall Signal Direction",["Long","Short"],key=f"{prefix}_cb_dir")
        p["_cond_list"] = cond_list
        # Flatten first two conditions for sig_custom_builder compatibility
        if cond_list:
            c0=cond_list[0]
            p["ind1"]=c0["a_ind"]; p["ind1_period"]=c0["a_period"]; p["ind1_fixed"]=c0["a_fixed"]
            p["condition"]=c0["cond_op"]
            p["ind2"]=c0["b_ind"]; p["ind2_period"]=c0["b_period"]; p["ind2_fixed"]=c0["b_fixed"]
        if len(cond_list)>1:
            c1=cond_list[1]
            p["use_cond2"]=True; p["logic"]=c1["logic"]
            p["ind1b"]=c1["a_ind"]; p["ind1b_period"]=c1["a_period"]; p["ind1b_fixed"]=c1["a_fixed"]
            p["cond2_op"]=c1["cond_op"]
            p["ind2b"]=c1["b_ind"]; p["ind2b_period"]=c1["b_period"]; p["ind2b_fixed"]=c1["b_fixed"]
        else:
            p["use_cond2"]=False

    p.setdefault("atr_mult_sl",    1.5)
    p.setdefault("atr_mult_tgt",   2.0)
    p.setdefault("rr_ratio",       2.0)
    p.setdefault("swing_lookback", 5)
    return p

def config_banner(strategy,interval,period,sym,sl_type,sl_pts,tgt_type,tgt_pts,extra=None):
    """Render a compact config summary as HTML table — never truncates on desktop."""
    items = [
        ("Strategy",  str(strategy)),
        ("Interval",  str(interval)),
        ("Period",    str(period)),
        ("Ticker",    str(sym)),
        ("SL Type",   str(sl_type)),
        ("SL Pts",    str(sl_pts)),
        ("Tgt Type",  str(tgt_type)),
        ("Tgt Pts",   str(tgt_pts)),
    ]
    if extra:
        for k,v in list(extra.items())[:6]:
            items.append((str(k), str(v)))
    cells = "".join(
        f'<td style="padding:4px 10px;border:1px solid #444;white-space:nowrap;">'
        f'<span style="font-size:10px;color:#aaa;">{k}</span><br>'
        f'<b style="font-size:13px;">{v}</b></td>'
        for k,v in items
    )
    html = (
        '<div style="overflow-x:auto;margin-bottom:8px;">'
        f'<table style="border-collapse:collapse;width:100%;"><tr>{cells}</tr></table>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
# ── SESSION STATE ─────────────────────────────────────────────────────────────
for _k,_v in {"live_active":False,"live_trades":[],"live_position":None,"live_tick":0,
               "opt_applied":None,"opt_results":None,"opt_res_meta":None,"opt_df":None,
               "_oa_hash_prev":"","no_overlap":True,"time_filter":False,
               "dhan_enabled":False,"cooldown_enabled":True,"cooldown_secs":5,
               "last_trade_close_ts":0.0,"_last_trade_fp":"",
               "g_interval":"1m","g_period":"7d",
               "g_strategy":"Elliott Wave (Simplified)"}.items():
    if _k not in st.session_state: st.session_state[_k]=_v

def _idx(lst,val,default=0): return lst.index(val) if val in lst else default

# ── PRE-POPULATE SIDEBAR WIDGET STATES FROM APPLIED OPTIMIZATION ─────────────
# Runs every render; only writes widget states when opt_applied changes.
# This ensures sidebar, backtest AND live trading all reflect the same params.
_oa_cur      = st.session_state.get("opt_applied")
_oa_hash_new = str(id(_oa_cur)) + str(_oa_cur) if _oa_cur else ""
if _oa_cur and _oa_hash_new != st.session_state.get("_oa_hash_prev",""):
    st.session_state["_oa_hash_prev"] = _oa_hash_new

    # ── Instrument / ticker ────────────────────────────────────────────────
    _inst = _oa_cur.get("instrument","")
    if _inst in TICKER_MAP:
        st.session_state["g_ticker"] = _inst
    elif _inst not in TICKER_MAP and _inst:
        # It's a raw Yahoo symbol (e.g. "KAYNES.NS") — set Custom mode
        st.session_state["g_ticker"] = "Custom"
        st.session_state["g_custom"] = _inst

    # Also handle explicit custom_sym stored during apply
    _csym = _oa_cur.get("custom_sym","")
    if _csym:
        st.session_state["g_ticker"] = "Custom"
        st.session_state["g_custom"] = _csym

    # ── Timeframe / Period / Strategy / SL / Target ────────────────────────
    if _oa_cur.get("interval") in TIMEFRAMES:  st.session_state["g_interval"] = _oa_cur["interval"]
    if _oa_cur.get("period")   in PERIODS:     st.session_state["g_period"]   = _oa_cur["period"]
    if _oa_cur.get("strategy") in STRATEGIES:  st.session_state["g_strategy"] = _oa_cur["strategy"]
    if _oa_cur.get("sl_type")  in SL_TYPES:    st.session_state["g_sl_type"]  = _oa_cur["sl_type"]
    if _oa_cur.get("tgt_type") in TARGET_TYPES:st.session_state["g_tgt_type"] = _oa_cur["tgt_type"]
    st.session_state["g_sl_pts"]  = float(_oa_cur.get("sl_pts",  10))
    st.session_state["g_tgt_pts"] = float(_oa_cur.get("tgt_pts", 20))

    # ── Strategy param widgets (prefix="sb") ──────────────────────────────
    _ap = _oa_cur.get("params", {})
    _PKMAP = {
        "EMA Crossover":            {"fast":"sb_fast","slow":"sb_slow"},
        "RSI Overbought/Oversold":  {"period":"sb_rp","ob":"sb_ob","os_":"sb_os"},
        "Bollinger Bands":          {"period":"sb_bbp","std":"sb_bbs"},
        "MACD Crossover":           {"fast":"sb_mf","slow":"sb_ms","signal":"sb_msig"},
        "Supertrend":               {"period":"sb_stp","multiplier":"sb_stm"},
        "ADX + DI Crossover":       {"period":"sb_ap","adx_thresh":"sb_at"},
        "Stochastic Oscillator":    {"k":"sb_k","d":"sb_d","ob":"sb_so","os_":"sb_su"},
        "Donchian Breakout":        {"period":"sb_dp"},
        "Triple EMA Trend":         {"f":"sb_tf","m":"sb_tm","s_":"sb_ts"},
        "BB + RSI Mean Reversion":  {"bb_period":"sb_brbbp","bb_std":"sb_brbbs",
                                     "rsi_period":"sb_brrp","rsi_os":"sb_bro","rsi_ob":"sb_brob"},
        "Keltner Channel Breakout": {"ema_p":"sb_kep","atr_p":"sb_kap","mult":"sb_km"},
        "Williams %R Reversal":     {"period":"sb_wrp","ob":"sb_wrob","os_":"sb_wros"},
        "Swing Trend + Pullback":   {"trend_ema":"sb_ste","entry_ema":"sb_see","rsi_period":"sb_srp",
                                     "rsi_bull_min":"sb_sbmin","rsi_bull_max":"sb_sbmax",
                                     "vol_mult":"sb_svm","rsi_bear_min":"sb_snmin","rsi_bear_max":"sb_snmax"},
        "VWAP Deviation":           {"dev_pct":"sb_vd"},
        "Ichimoku Cloud":           {"tenkan":"sb_it","kijun":"sb_ik"},
        "Heikin Ashi EMA":          {"ema_period":"sb_hap"},
        "Volume Price Trend (VPT)": {"vpt_ema_period":"sb_vp"},
        "RSI Divergence":           {"period":"sb_rp2","lookback":"sb_lb"},
        "Price Threshold Cross":    {"threshold":"sb_thresh"},
        "Elliott Wave (Simplified)":{"swing_lookback_ew":"sb_ewlb","min_wave_pct":"sb_ewpct"},
        "Elliott Wave v2 (Swing+Fib)":{"swing_bars":"sb_ewv2sb","fib_min":"sb_ewv2fmin","fib_max":"sb_ewv2fmax","ema_period":"sb_ewv2ep"},
        "HV Percentile (IV Proxy)": {"hv_period":"sb_hvp","ob_pct":"sb_hvob","os_pct":"sb_hvos"},
        "SMC Order Blocks":         {"ob_lookback":"sb_smcob","fvg_min_pct":"sb_smcfvg"},
        "Price Action Patterns":    {"pin_bar_ratio":"sb_pbr","engulf_pct":"sb_epct"},
        "Breakout Strategy":        {"lookback":"sb_bklb","break_pct":"sb_bkpct","vol_mult":"sb_bkvol"},
        "Support & Resistance + EMA":{"sr_lookback":"sb_srlb","ema_period":"sb_srema","touch_pct":"sb_srtp"},
    }
    _st_key = _oa_cur.get("strategy","")
    for pname, wkey in _PKMAP.get(_st_key, {}).items():
        if pname in _ap:
            try: st.session_state[wkey] = float(_ap[pname])
            except: pass

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Global Config")
    _oa=st.session_state.get("opt_applied")
    t_choice=st.selectbox("Instrument",list(TICKER_MAP.keys()),key="g_ticker")
    sym=st.text_input("Yahoo Ticker","RELIANCE.NS",key="g_custom").strip() if t_choice=="Custom" else TICKER_MAP[t_choice]
    interval=st.selectbox("Timeframe",TIMEFRAMES,key="g_interval")
    period=st.selectbox("Period",PERIODS,key="g_period")
    st.markdown("---")
    st.subheader("📈 Strategy")
    strategy=st.selectbox("Strategy",STRATEGIES,key="g_strategy")
    # ── Strategy parameters live in sidebar so both Backtest + Live share them ─
    with st.expander("⚙️ Strategy Parameters", expanded=True):
        sb_params = strategy_params_ui(strategy, prefix="sb",
                                        applied=_oa.get("params") if _oa else None)
    st.subheader("🛡 Stop Loss")
    sl_type=st.selectbox("SL Type",SL_TYPES,key="g_sl_type")
    sl_pts=st.number_input("SL Value (pts)",0.01,1e6,10.,step=0.5,key="g_sl_pts")
    # Extra params for special SL types
    if sl_type=="Cost-to-Cost K-Shift Trailing":
        st.caption("Trail SL: when price moves N pts in favor → lock SL at entry+K pts, then trail")
        _kn=st.number_input("Trigger N (pts in favor)",0.01,1e6,10.,step=0.5,key="g_kshift_n")
        _kk=st.number_input("Lock-in K (pts from entry)",0.01,1e6,3.,step=0.5,key="g_kshift_k")
        sb_params["kshift_n"]=float(_kn); sb_params["kshift_k"]=float(_kk)
    elif sl_type=="Risk/Reward Based SL":
        st.caption("SL = Target / RR ratio (uses Target pts and RR from params)")
        sb_params["tgt_pts_for_sl"]=tgt_pts if "tgt_pts" in dir() else 20.0
    st.subheader("🎯 Target")
    tgt_type=st.selectbox("Target Type",TARGET_TYPES,key="g_tgt_type")
    tgt_pts=st.number_input("Target Value (pts)",0.01,1e6,20.,step=0.5,key="g_tgt_pts")
    if _oa:
        st.success(f"✅ Applied: {_oa.get('strategy')}  Acc:{_oa.get('accuracy','?')}%")
        if st.button("Clear Applied"): st.session_state.opt_applied=None; st.rerun()
    st.markdown("---")

    # ── DHAN BROKER CONFIG ────────────────────────────────────────────────────
    st.subheader("🔌 Dhan Broker")
    dhan_enabled=st.checkbox("Enable Dhan Broker",value=False,key="dhan_enabled")
    if dhan_enabled:
        dhan_client=st.text_input("Client ID", DHAN_CLIENT_ID, key="dhan_client")
        dhan_token =st.text_input("Access Token", DHAN_ACCESS_TOKEN, key="dhan_token", type="password")
        st.caption(f"💡 Edit `DHAN_CLIENT_ID` and `DHAN_ACCESS_TOKEN` at the top of the .py file to change credentials permanently.")
        st.caption("**Order type — always BUYER (never seller in options)**")
        is_stocks=st.checkbox("Stocks / Intraday mode  (uncheck = Options CE/PE buyer)",value=False,key="dhan_is_stocks")
        if is_stocks:
            st.caption("LONG signal → Stock BUY  |  SHORT signal → Stock SELL/Short")
            dhan_s_sid=st.text_input("Stock Security ID","12092",key="dhan_s_sid",
                                      help="Dhan Security ID for the stock (e.g. 12092 = Kaynes Technology). "
                                           "Find in Dhan instrument master CSV.")
            st.caption("Default: 12092 = Kaynes Technology (NSE)")
            dhan_prod =st.selectbox("Trading Type",["INTRADAY","DELIVERY"],key="dhan_prod")
            dhan_exch =st.selectbox("Exchange",["NSE","BSE"],key="dhan_exch")
            dhan_s_qty=st.number_input("Quantity",1,10000,1,step=1,key="dhan_s_qty")
            dhan_ce_sid=""; dhan_pe_sid=""
            dhan_o_exch="NSE.FNO"; dhan_o_qty=65
        else:
            st.caption("LONG signal → CE BUY  |  SHORT signal → PE BUY  (always buying options)")
            dhan_s_sid=""
            dhan_ce_sid=st.text_input("CE Security ID (ATM call)","57749",key="dhan_ce_sid",
                                       help="Default 57749 — update to current ATM CE security_id from Dhan instrument master")
            dhan_pe_sid=st.text_input("PE Security ID (ATM put)","57716",key="dhan_pe_sid",
                                       help="Default 57716 — update to current ATM PE security_id from Dhan instrument master")
            dhan_o_exch=st.selectbox("F&O Exchange",["NSE_OPT","BSE_OPT"],key="dhan_o_exch")
            dhan_o_qty =st.number_input("Options Quantity",1,10000,65,step=1,key="dhan_o_qty")
            st.caption("CE sid=57749 | PE sid=57716 | qty=65 (Nifty lot size). Update sids daily to current ATM strike.")
            dhan_prod="INTRADAY"; dhan_exch="NSE"; dhan_s_qty=1
        dhan_sq_all=st.checkbox("Square off ALL open positions before new order",value=False,key="dhan_sq_all")
        # Default: Entry = LIMIT (price control), Exit = MARKET (fast fill)
        dhan_entry_ot=st.selectbox("Entry Order Type",["LIMIT","MARKET"],index=0,key="dhan_entry_ot")
        dhan_exit_ot =st.selectbox("Exit Order Type", ["MARKET","LIMIT"],index=0,key="dhan_exit_ot")
        if dhan_entry_ot=="LIMIT":
            st.caption("LIMIT entry: LTP at signal time will be used as limit price.")
        if dhan_exit_ot=="LIMIT":
            st.caption("LIMIT exit: LTP at exit time will be used as limit price.")
        # ── Multi-account support ─────────────────────────────────────────
        multi_acc = st.checkbox("Enable Multi-Account (place orders in multiple Dhan accounts)",
                                value=False, key="dhan_multi_acc")
        if multi_acc:
            st.caption("Add extra Dhan accounts below. Orders placed simultaneously in ALL accounts.")
            _n_extra = st.session_state.get("dhan_n_extra", 0)
            _ca1, _ca2 = st.columns(2)
            if _ca1.button("➕ Add Account", key="dhan_add_acc"):
                st.session_state["dhan_n_extra"] = min(_n_extra + 1, 5)
                st.rerun()
            if _ca2.button("➖ Remove Last", key="dhan_rem_acc"):
                st.session_state["dhan_n_extra"] = max(_n_extra - 1, 0)
                st.rerun()
            _extra_accounts = []
            for _ai in range(int(st.session_state.get("dhan_n_extra", 0))):
                with st.expander(f"Account {_ai+2}", expanded=True):
                    _ecid = st.text_input(f"Client ID (acc{_ai+2})", "", key=f"dhan_ecid_{_ai}")
                    _etok = st.text_input(f"Token (acc{_ai+2})", "", key=f"dhan_etok_{_ai}", type="password")
                    _extra_accounts.append({"client":_ecid, "token":_etok})
            st.session_state["dhan_extra_accounts"] = _extra_accounts
        else:
            st.session_state["dhan_extra_accounts"] = []
    else:
        dhan_client="1104779876"; dhan_token=""; is_stocks=False
        dhan_prod="INTRADAY"; dhan_exch="NSE"; dhan_s_qty=1
        dhan_s_sid="12092"
        dhan_ce_sid="57749"; dhan_pe_sid="57716"; dhan_o_exch="NSE_OPT"; dhan_o_qty=65
        dhan_sq_all=False; dhan_entry_ot="LIMIT"; dhan_exit_ot="MARKET"

    # ── TRADE MANAGEMENT ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚙️ Trade Management")
    main_qty = st.number_input("Position Quantity (main algo)", 1, 100000, 1, step=1,
                                key="g_main_qty",
                                help="Quantity used by main algo signals. Default=1.")
    no_overlap =st.checkbox("Prevent Overlapping Trades",value=True,key="no_overlap",
        help="If a position is already open, ignore new signals until it closes.")
    cooldown_enabled=st.checkbox("Cooldown Between Trades",value=True,key="cooldown_enabled",
        help="Minimum wait time after a trade closes before a new one can open.")
    if cooldown_enabled:
        cooldown_secs=st.number_input("Cooldown (seconds)",1,300,5,step=1,key="cooldown_secs")
    else:
        cooldown_secs=0
    # Adjusted target: when entering late, reduce target by already-captured move
    adj_target_enabled = st.checkbox(
        "Adjusted Target (Late Entry Mode)",
        value=False, key="adj_target_enabled",
        help="When you enter after the signal already fired and price has moved in signal direction, "
             "the remaining target is reduced to (original_target - already_captured_move). "
             "Example: Signal fired, target=20pts, price already moved 8pts → your target = 12pts. "
             "This keeps your exit realistic for the remaining move available."
    )
    if adj_target_enabled:
        st.caption(
            "✅ Active — target will be set to the REMAINING portion of the original signal target. "
            "If entering fresh (0 bars elapsed), target is unchanged."
        )
    time_filter=st.checkbox("Time Window Filter (IST)",value=False,key="time_filter",
        help="Only place/exit orders within the specified IST time window.")
    if time_filter:
        _twc1,_twc2=st.columns(2)
        tw_from=_twc1.time_input("From",value=datetime.strptime("09:15","%H:%M").time(),key="tw_from")
        tw_to  =_twc2.time_input("To",  value=datetime.strptime("15:00","%H:%M").time(),key="tw_to")
    else:
        tw_from=datetime.strptime("09:15","%H:%M").time()
        tw_to  =datetime.strptime("15:00","%H:%M").time()

    # ── DATA FRESHNESS HANDLING ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📡 Data Freshness (yfinance Delay)")
    freshness_enabled = st.checkbox(
        "Enable Data Freshness Check & EW Next-Pivot Predictor",
        value=False, key="freshness_enabled",
        help="yfinance typically delivers 4-7 min delayed candles on 1m charts. "
             "When enabled:\n"
             "1. Compares last candle time vs current IST time — warns if data is stale.\n"
             "2. Retries fetch up to 3x if delay > threshold to get latest candle.\n"
             "3. For Elliott Wave: predicts the next expected pivot (High or Low), "
             "    and shows projected entry/SL/target levels so you can enter manually "
             "    at the right price in your broker app."
    )
    if freshness_enabled:
        _delay_thresh = st.slider(
            "Max acceptable candle delay (minutes)",
            1, 15, 5, key="delay_thresh",
            help="If last candle is older than this many minutes, trigger a re-fetch."
        )
        st.caption(
            f"⚡ Active — if last 1m candle is > {_delay_thresh} min old, "
            "the app will retry the data fetch up to 3 times. "
            "EW Next-Pivot panel will also appear in Live Trading."
        )
    else:
        _delay_thresh = 7

    # ── EMAIL NOTIFICATIONS ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📧 Email Notifications")
    email_enabled = st.checkbox("Enable Email Alerts (Gmail)",value=False,key="email_enabled",
        help="Sends email when a BUY/SELL signal fires or a trade exits.")
    if email_enabled:
        email_sender  = st.text_input("Gmail sender address","srinivas.trml@gmail.com",key="email_sender",
                                       help="Your Gmail address (e.g. you@gmail.com)")
        email_apppass = st.text_input("Gmail App Password","",key="email_apppass",type="password",
                                       help="Use a Gmail App Password (not your login password). "
                                            "Enable 2FA → Google Account → Security → App Passwords.")
        email_to      = st.text_input("Send alerts to","srinivas.trml@gmail.com",key="email_to",
                                       help="Recipient email (can be same as sender)")
        st.caption("Alerts sent for: BUY signal, SELL signal, Trade exit (SL/Target/Strategy). "
                   "Uses Gmail SMTP (port 587).")
    else:
        email_sender=""; email_apppass=""; email_to=""

    st.markdown("---")
    st.caption("1.5s rate-limit delay between all yfinance requests.")
    if not _HAS_FRAGMENT: st.caption("Upgrade Streamlit ≥1.33 for flicker-free live tab.")

# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab_bt,tab_live,tab_opt,tab_nte=st.tabs(
    ["📊 Backtesting","⚡ Live Trading","🔬 Optimization","🎯 Near to Entry"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.subheader(f"Backtesting · {t_choice} [{interval}/{period}]")
    bt_params = sb_params
    st.caption("⚙️ Strategy parameters are configured in the **sidebar** and shared with Live Trading.")

    # ── Mismatch fix: use stored opt_df when opt_applied to get exact match ──
    _oa_bt = st.session_state.get("opt_applied")
    _use_opt_data = False
    if _oa_bt:
        _stored_df = st.session_state.get("opt_df")
        if _stored_df is not None and not _stored_df.empty:
            _use_opt_data = st.checkbox(
                "Use optimization data (ensures exact match with opt results)",
                value=True, key="bt_use_opt_data",
                help="When ON, backtest uses the same data the optimizer used. "
                     "OFF = fresh yfinance fetch (may differ by a few bars)."
            )

    config_banner(strategy,interval,period,sym,sl_type,sl_pts,tgt_type,tgt_pts,
        extra={k:v for k,v in bt_params.items() if k not in("atr_mult_sl","atr_mult_tgt","rr_ratio","swing_lookback",
                                                               "thresh_dir","thresh_action","kshift_n","kshift_k",
                                                               "tgt_pts_for_sl","use_cond2")})

    # Time window for backtest (reads from sidebar)
    _bt_tw_from_str = tw_from.strftime("%H:%M") if st.session_state.get("time_filter",False) else None
    _bt_tw_to_str   = tw_to.strftime("%H:%M")   if st.session_state.get("time_filter",False) else None

    if st.button("▶ Run Backtest",type="primary",key="btn_bt"):
        if _use_opt_data and _stored_df is not None:
            df_bt = _stored_df
            st.info(f"Using stored optimization data ({len(df_bt)} bars). "
                    "Results will exactly match optimization output.")
        else:
            with st.spinner("Fetching data…"): df_bt=fetch_data(sym,period,interval)
        if df_bt is None or df_bt.empty:
            st.error("No data returned. Try different ticker/interval/period.")
        else:
            with st.spinner("Running backtest…"):
                trades_bt,indics_bt,audit_bt=run_backtest(
                    df_bt,strategy,bt_params,sl_type,sl_pts,tgt_type,tgt_pts,
                    tw_from_str=_bt_tw_from_str, tw_to_str=_bt_tw_to_str)
            perf_bt=calc_perf(trades_bt)
            st.markdown("### 📋 Performance Summary")
            if perf_bt:
                mc=st.columns(len(perf_bt))
                for col,(k,v) in zip(mc,perf_bt.items()):
                    col.metric(k,v,delta=f"{v:.1f}%" if k=="Accuracy (%)" else None)
            else: st.info("No trades generated.")
            st.markdown("### 📈 Price Chart")
            st.plotly_chart(plot_ohlc(df_bt,trades_bt,indics_bt,title=f"{t_choice}—{strategy}[{interval}]"),use_container_width=True,key="bt_ohlc_chart")
            if trades_bt:
                eq=plot_equity(trades_bt)
                if eq: st.markdown("### 💹 Equity Curve"); st.plotly_chart(eq,use_container_width=True,key="bt_equity_chart")

                # ── Build audit index for cross-referencing ────────────────
                adf=pd.DataFrame(audit_bt) if audit_bt else pd.DataFrame()

                # Identify anomalous trade indices from audit
                _sl_anom_ids=set(); _tgt_anom_ids=set()
                _sl_anom_bars={}; _tgt_anom_bars={}   # trade_id -> first breach bar info
                if not adf.empty:
                    _tdo=(tgt_type=="Trailing Target (Display Only)")
                    for tid,grp in adf.groupby("trade_idx"):
                        # SL anomaly: SL was breached but trade continued
                        sl_b=grp[((grp["direction"]=="LONG")&(grp["bar_low"]<=grp["sl_level"]))|
                                 ((grp["direction"]=="SHORT")&(grp["bar_high"]>=grp["sl_level"]))]
                        if len(sl_b):
                            eb=grp[grp["trade_exited"]==True]
                            if len(eb) and eb.index[0]>sl_b.index[0]:
                                _sl_anom_ids.add(tid)
                                _sl_anom_bars[tid]={"breach_bar":sl_b.iloc[0]["bar_dt"],
                                    "sl_level":sl_b.iloc[0]["sl_level"],
                                    "candle_low_high":(sl_b.iloc[0]["bar_low"] if sl_b.iloc[0]["direction"]=="LONG" else sl_b.iloc[0]["bar_high"])}
                        # Target anomaly: target breached but trade continued (and no SL hit before)
                        if not _tdo:
                            tgt_b=grp[((grp["direction"]=="LONG")&(grp["bar_high"]>=grp["target_level"]))|
                                      ((grp["direction"]=="SHORT")&(grp["bar_low"]<=grp["target_level"]))]
                            if len(tgt_b):
                                sl_before=len(sl_b)>0 and sl_b.index[0]<=tgt_b.index[0]
                                eb=grp[grp["trade_exited"]==True]
                                if len(eb) and eb.index[0]>tgt_b.index[0] and not sl_before:
                                    _tgt_anom_ids.add(tid)
                                    _tgt_anom_bars[tid]={"breach_bar":tgt_b.iloc[0]["bar_dt"],
                                        "target_level":tgt_b.iloc[0]["target_level"],
                                        "candle_high_low":(tgt_b.iloc[0]["bar_high"] if tgt_b.iloc[0]["direction"]=="LONG" else tgt_b.iloc[0]["bar_low"])}

                _anom_ids = _sl_anom_ids | _tgt_anom_ids

                COL=["Entry DateTime","Exit DateTime","Direction",
                     "Entry Price","Exit Price","SL Level","Target Level",
                     "Highest Price","Lowest Price",
                     "Exit Reason","SL Type","Target Type",
                     "Points Gained","Points Lost","PnL","Signal Bar"]

                tdf_all=pd.DataFrame(trades_bt)
                tdf_all=tdf_all[[c for c in COL if c in tdf_all.columns]].reset_index(drop=True)

                # ── TABLE 1: Correct trades ────────────────────────────────
                st.markdown("### 📜 Table 1 — Correct Trades  *(SL / Target correctly obeyed)*")
                st.caption(
                    "These trades exited **exactly** when the candle Low (for LONG) or High (for SHORT) "
                    "crossed the SL level, or when High/Low crossed the Target level. "
                    "'End of Data' exits mean neither SL nor Target was hit before the data ended — "
                    "**this is correct behavior, not a missed exit.**  "
                    "All values are accurate and match what live trading would show.  "
                    "Gap-up/gap-down: entry uses bar Open price (already reflects the gap). "
                    "Prev-candle SL uses prior bar Low/High regardless of gap — correct for Indian markets."
                )
                correct_idx=[i for i in range(len(trades_bt)) if i not in _anom_ids]
                tdf_ok=tdf_all.iloc[correct_idx].copy()
                # Convert datetime columns to IST for display
                for _dc in ["Entry DateTime","Exit DateTime","Signal Bar"]:
                    if _dc in tdf_ok.columns:
                        tdf_ok[_dc]=tdf_ok[_dc].apply(to_ist)
                def _pnl_color(v):
                    if isinstance(v,(int,float)):
                        if v>0: return "color:#2e7d32;font-weight:bold"
                        if v<0: return "color:#c62828;font-weight:bold"
                    return ""
                sc=[c for c in ["PnL","Points Gained","Points Lost"] if c in tdf_ok.columns]
                st.dataframe(tdf_ok.style.map(_pnl_color,subset=sc) if sc else tdf_ok,
                             use_container_width=True,height=420)
                st.caption(f"✅ {len(tdf_ok)} correct trades  |  "
                           f"{len(tdf_ok[tdf_ok['Exit Reason']=='SL Hit'])} SL exits  |  "
                           f"{len(tdf_ok[tdf_ok['Exit Reason']=='Target Hit'])} Target exits  |  "
                           f"{len(tdf_ok[tdf_ok['Exit Reason']=='End of Data'])} End-of-Data exits" if not tdf_ok.empty else "")

                # ── TABLE 2: Anomalies ─────────────────────────────────────
                st.markdown("### 🔍 Table 2 — Anomaly Trades  *(SL or Target level was inside candle range but trade did NOT exit)*")
                if not _anom_ids:
                    st.success("✅ **No anomalies found.** The backtest engine correctly obeyed all SL and Target levels.  "
                               "You can trust these results for live trading alignment.")
                else:
                    st.warning(
                        f"⚠️ {len(_anom_ids)} trades had a potential anomaly.  "
                        "This typically happens when **trailing SL/Target moves past its initial value** "
                        "on the same bar it was set, making the initial 'breach' stale. "
                        "Cells highlighted in light red show which level was inside the candle range."
                    )
                    tdf_anom=tdf_all.iloc[sorted(_anom_ids)].copy()
                    # Add info columns
                    tdf_anom["SL Breach Bar"]  = tdf_anom.index.map(lambda i: _sl_anom_bars.get(i,{}).get("breach_bar",""))
                    tdf_anom["Tgt Breach Bar"] = tdf_anom.index.map(lambda i: _tgt_anom_bars.get(i,{}).get("breach_bar",""))

                    # Style: light red only on the anomalous cells
                    def _anom_cell(val,col_name,row_idx):
                        if col_name=="SL Level" and row_idx in _sl_anom_ids:
                            return "background-color:#ffcdd2;color:#212121;font-weight:bold"
                        if col_name=="Target Level" and row_idx in _tgt_anom_ids:
                            return "background-color:#ffcdd2;color:#212121;font-weight:bold"
                        return ""

                    styled=tdf_anom.style
                    for col_n in tdf_anom.columns:
                        styled=styled.apply(
                            lambda col: [_anom_cell(v,col.name,i)
                                         for i,v in zip(tdf_anom.index,col)],
                            subset=[col_n]
                        )
                    st.dataframe(styled,use_container_width=True,height=300)

                # ── WHY explanation ────────────────────────────────────────
                with st.expander("❓ Why might a trade miss SL/Target? (click to read)"):
                    st.markdown("""
**Short answer: The backtest is correct.** Here's the full explanation:

**OHLC bar data limitation:**
Each candle only tells you Open, High, Low, Close — it does **not** tell you the order
in which High and Low were reached within the candle.

**When both SL and Target are inside the same candle range:**
- Candle Low ≤ SL (would stop you out)  AND  Candle High ≥ Target (would hit target) — on the **same bar**
- The engine conservatively takes **SL first** (price went against you before hitting target)
- This is the most prudent assumption for bar data

**"End of Data" exits:**
These are NOT missed SL/Targets. It means neither SL nor Target was hit during the
entire life of that trade — the data simply ran out. In live trading the position
stays open until SL/Target fires or you manually close.

**If Table 2 shows anomalies:**
These are usually caused by trailing SL/Target types where the level is updated
intra-bar (e.g. trailing candle low/high updates on every bar). An initial "breach"
reading may become stale after the trailing update. This is a known limitation
of bar-based backtesting vs. tick data — live trading will be more precise.
                    """)

                with st.expander("📂 Raw OHLC Data"):
                    st.dataframe(df_bt,use_container_width=True)
                with st.expander("📋 Full Bar-Level Audit Trail"):
                    if not adf.empty:
                        disp=adf[["trade_idx","direction","bar_dt","entry_price","bar_high","bar_low","bar_close","sl_level","target_level","sl_breached","tgt_breached","trade_exited","exit_reason"]]
                        def _ast(row):
                            if row["trade_exited"]: return ["background-color:#e8f5e9"]*len(row)
                            if row["sl_breached"]:  return ["background-color:#ffebee"]*len(row)
                            return [""]*len(row)
                        st.dataframe(disp.style.apply(_ast,axis=1),use_container_width=True,height=400)
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE TRADING  (flicker-free via @st.fragment on Streamlit>=1.33)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.subheader(f"⚡ Live Trading · {t_choice} [{interval}]")
    # Use sidebar params — same as backtesting
    live_params = sb_params
    st.caption("⚙️ Strategy parameters are configured in the **sidebar** — same as Backtesting tab.")
    config_banner(strategy,interval,"—",sym,sl_type,sl_pts,tgt_type,tgt_pts,
        extra={k:v for k,v in live_params.items() if k not in("atr_mult_sl","atr_mult_tgt","rr_ratio","swing_lookback",
                                                                "thresh_dir","thresh_action")})
    st.markdown("---")
    _bc=st.columns([1,1,1,5])
    if _bc[0].button("▶ Start",type="primary",disabled=st.session_state.live_active,key="btn_lv_start"):
        st.session_state.update({"live_active":True,"live_trades":[],"live_position":None,
                                  "live_tick":0,"_last_trade_fp":"","last_trade_close_ts":0.0}); st.rerun()
    if _bc[1].button("⏹ Stop",disabled=not st.session_state.live_active,key="btn_lv_stop"):
        st.session_state.live_active=False; st.rerun()
    if _bc[2].button("🗑 Clear",key="btn_lv_clear"):
        st.session_state.update({"live_trades":[],"live_position":None,
                                  "_last_trade_fp":"","last_trade_close_ts":0.0}); st.rerun()
    _bc[3].info("Smooth updates: " + ("✅ Active (Streamlit≥1.33)" if _HAS_FRAGMENT else "⚠️ Upgrade Streamlit≥1.33"))
    sub_mon,sub_hist=st.tabs(["📡 Live Monitor","📜 Trade History"])

    def _live_render():
        if not st.session_state.live_active:
            st.info("Press ▶ Start to begin live monitoring."); return
        st.session_state.live_tick+=1; tick=st.session_state.live_tick
        _now=datetime.now()
        st.caption(f"Tick **#{tick}** — {_now.strftime('%H:%M:%S IST')}  |  1.5s rate-limit enforced")

        # ── Time window check ─────────────────────────────────────────────────
        _in_window=True
        if st.session_state.get("time_filter",False):
            _ct=_now.time()
            _in_window = tw_from <= _ct <= tw_to
            if not _in_window:
                st.warning(f"⏰ Outside trading window ({tw_from.strftime('%H:%M')} – {tw_to.strftime('%H:%M')} IST). "
                           "Monitoring only — no new orders.")

        # ── Cooldown check ────────────────────────────────────────────────────
        _cooldown_ok = True
        _cd_secs = int(st.session_state.get("cooldown_secs",5)) if st.session_state.get("cooldown_enabled",True) else 0
        if _cd_secs > 0:
            _elapsed = time.time() - st.session_state.get("last_trade_close_ts",0.0)
            if _elapsed < _cd_secs:
                _cooldown_ok = False
                st.info(f"⏳ Cooldown: {_cd_secs-int(_elapsed)}s remaining before next entry.")

        lv_df=fetch_live(sym,interval)
        if lv_df is None or lv_df.empty: st.warning("No data. Retrying…"); return

        # ── Data freshness check & retry (when enabled) ───────────────────────
        _mins_map_fr = {"1m":1,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}
        _bar_mins    = _mins_map_fr.get(interval, 5)
        if st.session_state.get("freshness_enabled", False):
            _now_ist  = datetime.now(_IST)
            try:
                _last_c = lv_df.index[-1]
                if hasattr(_last_c, 'tzinfo') and _last_c.tzinfo is None:
                    _last_c = _IST.localize(_last_c.to_pydatetime())
                elif hasattr(_last_c, 'tzinfo') and _last_c.tzinfo is not None:
                    _last_c = _last_c.to_pydatetime().astimezone(_IST)
                else:
                    _last_c = _IST.localize(_last_c)
                _delay_mins = (_now_ist - _last_c).total_seconds() / 60
            except:
                _delay_mins = 0

            _thresh = float(st.session_state.get("delay_thresh", 7))
            if _delay_mins > _thresh + _bar_mins:
                # Data is stale — warn but do NOT retry here (retries cause freeze).
                # The next auto-refresh tick will fetch fresh data naturally.
                st.warning(
                    f"⚠️ Last candle is **{_delay_mins:.0f} min** old "
                    f"(threshold: {_thresh:.0f} min). "
                    f"Data may be stale — will refresh on next tick."
                )
            else:
                _delay_mins_disp = max(0, _delay_mins)
                if _delay_mins_disp <= _thresh:
                    st.caption(f"📡 Data freshness: last candle **{_delay_mins_disp:.0f} min** old — ✅ within threshold.")
                else:
                    st.caption(f"📡 Data freshness: last candle **{_delay_mins_disp:.0f} min** old — ⚠️ slightly stale.")
        lv_n=len(lv_df)

        # ── Always show last candle time vs current time (data freshness) ─────
        _now_lv   = datetime.now(_IST)
        try:
            _lc_ts = lv_df.index[-1]
            if hasattr(_lc_ts, 'tzinfo') and _lc_ts.tzinfo is None:
                _lc_ts = _IST.localize(_lc_ts.to_pydatetime())
            elif hasattr(_lc_ts, 'tzinfo') and _lc_ts.tzinfo is not None:
                _lc_ts = _lc_ts.to_pydatetime().astimezone(_IST)
            _data_delay_mins = (_now_lv - _lc_ts).total_seconds() / 60
            _delay_icon = "✅" if _data_delay_mins < 10 else ("⚠️" if _data_delay_mins < 60 else "🔴")
            st.caption(
                f"📡 Last candle: **{to_ist(lv_df.index[-1])}**  |  "
                f"Current time: **{_now_lv.strftime('%d-%b-%Y %H:%M:%S IST')}**  |  "
                f"Data delay: {_delay_icon} **{_data_delay_mins:.0f} min** "
                f"({'within 1 candle' if _data_delay_mins < 10 else 'market may be closed or data delayed'})"
            )
        except: pass

        # ── Price references ───────────────────────────────────────────────────
        # During live market: iloc[-1] = forming/current bar (incomplete)
        #                     iloc[-2] = last FULLY closed bar  ← use for SL/Target
        # When market is CLOSED: iloc[-1] IS a fully closed bar (no forming bar).
        #   Using iloc[-2] in this case shows the PREVIOUS bar (e.g. 14:15 instead of 15:15).
        # Fix: detect market-closed by comparing last candle time to current time.
        #   If delay > 1.5× bar duration → market closed → use iloc[-1] as last closed bar.
        _mins_per_bar_pr = {"1m":1,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}.get(interval,5)
        try:
            _last_bar_ts = lv_df.index[-1]
            _now_pr = datetime.now(_IST)
            if hasattr(_last_bar_ts,'tzinfo') and _last_bar_ts.tzinfo is None:
                _last_bar_ts_tz = _IST.localize(_last_bar_ts.to_pydatetime())
            elif hasattr(_last_bar_ts,'tzinfo') and _last_bar_ts.tzinfo is not None:
                _last_bar_ts_tz = _last_bar_ts.to_pydatetime().astimezone(_IST)
            else:
                _last_bar_ts_tz = _IST.localize(_last_bar_ts)
            _bar_age_mins = (_now_pr - _last_bar_ts_tz).total_seconds() / 60
            # If last bar is older than 1.5 bars → no forming bar → market closed
            _market_closed = _bar_age_mins > _mins_per_bar_pr * 1.5
        except:
            _market_closed = False

        cl      = float(lv_df["Close"].iloc[-1])    # LTP / most recent close
        if _market_closed:
            # Market closed: all bars are complete — use the very last bar as the closed bar
            _closed_bar_idx = lv_n - 1
        else:
            # Market open: iloc[-1] is forming — use iloc[-2] as last complete bar
            _closed_bar_idx = max(0, lv_n - 2)

        bh_cur  = float(lv_df["High"].iloc[_closed_bar_idx])
        bl_cur  = float(lv_df["Low"].iloc[_closed_bar_idx])
        last_bar= lv_df.index[_closed_bar_idx]

        fn=STRATEGY_FN.get(strategy,sig_custom)
        try: lv_sigs,lv_indics=fn(lv_df,**live_params)
        except Exception as e: lv_sigs=pd.Series(0,index=lv_df.index); lv_indics={}; st.warning(f"Strategy error:{e}")

        # ── Signal detection: lookback window matched to strategy type ────────
        # BACKTEST: scans every bar sequentially → sees every historical signal
        # LIVE:     re-runs strategy each tick → must check recent bars for fresh signal
        #
        # Non-pattern strategies: signal fires ON the bar (EMA cross, RSI cross etc.)
        #   → 1 bar lookback is enough (just closed bar)
        # Pattern strategies: confirmation requires price to move away from the signal bar
        #   → need wider lookback (Elliott Wave pivot confirmed after N bars of opposite move)
        #
        # Rule: use the most recent non-zero signal within the lookback window.
        # If that signal is older than the window, it is stale — do NOT enter.
        # EW v1 (Simplified): pivot confirmation requires price to move min_wave_pct% AWAY
        # from the extreme point. On Nifty 5min (0.1% typical bar), 1% threshold = ~10 bars lag.
        # Lookback must be ≥ (min_wave_pct / typical_bar_move). Use 20 to be safe.
        _WIDE_LOOKBACK  = {"Elliott Wave (Simplified)":20,   # ~10-15 bar confirmation lag
                           "Elliott Wave v2 (Swing+Fib)":10,
                           "Elliott Wave v3 (Extrema)":5,
                           "SMC Order Blocks":3,
                           "Price Action Patterns":2,
                           "Inside Bar Breakout":2,
                           "Breakout Strategy":2,
                           "Support & Resistance + EMA":2}
        _sig_lookback = _WIDE_LOOKBACK.get(strategy, 1)
        last_sig        = 0
        _sig_bars_ago   = 0
        _sig_bar_dt     = None
        _sig_bar_price  = None
        for _si in range(1, _sig_lookback + 2):
            if len(lv_sigs) > _si:
                _sv = int(lv_sigs.iloc[-_si])
                if _sv != 0:
                    last_sig      = _sv
                    _sig_bars_ago = _si - 1
                    try:
                        _sig_bar_dt    = lv_df.index[-_si]
                        _sig_bar_price = float(lv_df["Close"].iloc[-_si])
                    except: pass
                    break

        # ── pos must be assigned first — referenced throughout this function ──
        pos = st.session_state.live_position

        # ── Persist signal state so the panel survives page redraws ──────────
        # Signal state is stored in session_state keyed by ticker.
        # It is preserved across ALL ticks including when pos is open,
        # so the timing panel stays visible throughout the position life.
        # It is only cleared when a new trade is LOGGED (fingerprint changes).
        _sig_key = f"_live_sig_{sym}"
        if last_sig != 0 and _sig_bar_dt is not None:
            # Store/update the latest signal
            st.session_state[_sig_key] = {
                "sig":        last_sig,
                "bars_ago":   _sig_bars_ago,
                "bar_dt":     _sig_bar_dt,
                "bar_price":  _sig_bar_price,
                "bar_low":    float(lv_df["Low"].iloc[-(_sig_bars_ago+1)]) if lv_n > _sig_bars_ago+1 else _sig_bar_price - 10,
                "bar_high":   float(lv_df["High"].iloc[-(_sig_bars_ago+1)]) if lv_n > _sig_bars_ago+1 else _sig_bar_price + 10,
                "stored_tick": tick,
            }
        # Use stored signal when current scan returns 0 (transient data gap)
        # — works whether flat OR in a position so panel stays visible
        _stored_sig = st.session_state.get(_sig_key, {})
        if last_sig == 0 and _stored_sig:
            _stored_bars_ago   = _stored_sig.get("bars_ago", 0)
            _ticks_since_store = tick - _stored_sig.get("stored_tick", tick)
            last_sig        = _stored_sig["sig"]
            _sig_bars_ago   = _stored_bars_ago + _ticks_since_store
            _sig_bar_dt     = _stored_sig["bar_dt"]
            _sig_bar_price  = _stored_sig["bar_price"]
        # Clear only when a NEW TRADE fingerprint is logged (i.e. a fresh signal on
        # a different bar replaces the old one — handled naturally by the store above)
        # Do NOT clear just because pos is not None — that caused the panel to disappear.

        # ── Price row ─────────────────────────────────────────────────────────
        m=st.columns(6)
        m[0].metric("LTP",f"{cl:.2f}")
        m[1].metric("High",f"{bh_cur:.2f}")
        m[2].metric("Low",f"{bl_cur:.2f}")
        m[3].metric("Spread",f"{bh_cur-bl_cur:.2f}")
        sig_txt="🟢 BUY" if last_sig==1 else ("🔴 SELL" if last_sig==-1 else "⚪ FLAT")
        m[4].metric("Signal",sig_txt)
        m[5].metric("Last Candle", to_ist(lv_df.index[-1]),
                    help="Timestamp of the most recent candle fetched from yfinance. "
                         "If this is old, data may be delayed.")

        # ── Last candle full OHLCV row ────────────────────────────────────────
        try:
            _lc = lv_df.iloc[-1]
            _lc_o = float(_lc["Open"]); _lc_h = float(_lc["High"])
            _lc_l = float(_lc["Low"]);  _lc_c = float(_lc["Close"])
            _lc_v = float(_lc.get("Volume", 0))
            _lc_dir = "🟢" if _lc_c >= _lc_o else "🔴"
            _lc_chg = _lc_c - _lc_o
            _lc_cols = st.columns(7)
            _lc_cols[0].metric("Candle",  f"{_lc_dir} {to_ist(lv_df.index[-1])[10:16]}",
                               help="Last fetched candle — may be delayed by 4-7 min on 1m.")
            _lc_cols[1].metric("Open",    f"{_lc_o:.2f}")
            _lc_cols[2].metric("High",    f"{_lc_h:.2f}")
            _lc_cols[3].metric("Low",     f"{_lc_l:.2f}")
            _lc_cols[4].metric("Close",   f"{_lc_c:.2f}",
                               delta=f"{_lc_chg:+.2f}",
                               delta_color="normal" if _lc_chg >= 0 else "inverse")
            _lc_cols[5].metric("Range",   f"{_lc_h-_lc_l:.2f}")
            _lc_cols[6].metric("Volume",  f"{int(_lc_v):,}" if _lc_v else "—")
        except: pass

        # ── Signal Timing Panel — detailed, anti-FOMO entry guidance ────────
        # Shows whenever there's a known signal — whether flat OR in a position.
        if last_sig != 0 and _sig_bar_dt is not None:
            _sig_dir_label = "🟢 BUY (LONG)" if last_sig==1 else "🔴 SELL (SHORT)"
            _price_move    = cl - _sig_bar_price
            _move_vs_sl    = abs(_price_move) / max(sl_pts, 0.01)
            _favorable     = (_price_move > 0 and last_sig==1) or (_price_move < 0 and last_sig==-1)
            _mins_map      = {"1m":1,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}
            _mins_per_bar  = _mins_map.get(interval, 5)
            _mins_elapsed  = _sig_bars_ago * _mins_per_bar

            # SL/Target levels if entering right now at cl
            _sl_now  = cl - last_sig * sl_pts
            _tgt_now = cl + last_sig * tgt_pts
            _rr_now  = tgt_pts / max(sl_pts, 0.01)

            # Original SL/Target at signal bar price
            _sl_orig  = _sig_bar_price - last_sig * sl_pts
            _tgt_orig = _sig_bar_price + last_sig * tgt_pts

            # How much of target already captured by price moving in signal direction
            _tgt_consumed_pct = abs(_price_move) / max(tgt_pts, 0.01) * 100 if _favorable else 0

            # Freshness score 0-100 (100 = just fired, 0 = very stale)
            # Degrades with bars elapsed and SL consumed
            _freshness = max(0, 100 - _sig_bars_ago * 8 - _move_vs_sl * 60)

            # ── Verdict logic ─────────────────────────────────────────────────
            if _sig_bars_ago == 0:
                _verdict = "FRESH"
                _action  = "ENTER NOW"
                _color   = "success"
            elif _favorable and _move_vs_sl < 0.25:
                _verdict = "VALID"
                _action  = "ENTER — still good"
                _color   = "success"
            elif _favorable and _move_vs_sl < 0.55:
                _verdict = "BORDERLINE"
                _action  = "ENTER CAREFULLY — reduced R:R"
                _color   = "warning"
            elif _favorable and _move_vs_sl >= 0.55:
                _verdict = "TOO LATE"
                _action  = "SKIP — wait for next signal"
                _color   = "error"
            elif not _favorable and _move_vs_sl < 0.30:
                _verdict = "PULLBACK"
                _action  = "MONITOR — price pulling back, may still be OK"
                _color   = "warning"
            else:
                _verdict = "INVALIDATED"
                _action  = "DO NOT ENTER — signal broken"
                _color   = "error"

            # Freshness bar
            _fresh_filled = int(_freshness / 5)
            _fresh_bar    = "█" * _fresh_filled + "░" * (20 - _fresh_filled)
            _fresh_label  = ("🟢 Very Fresh" if _freshness > 75 else
                             "🟡 Moderately Fresh" if _freshness > 45 else
                             "🟠 Getting Stale" if _freshness > 20 else
                             "🔴 Stale")

            st.markdown("---")
            st.markdown("## ⏱️ Signal Entry Intelligence")
            st.caption("Updated every tick — tells you exactly whether to enter, wait, or skip.")

            # ── Big verdict banner ─────────────────────────────────────────────
            _banner_text = (
                f"**{_verdict}** — {_action}  |  "
                f"Signal: {_sig_dir_label}  |  "
                f"Fired: {to_ist(_sig_bar_dt)}  |  "
                f"{_sig_bars_ago} bar(s) ago (~{_mins_elapsed} min elapsed)"
            )
            if _color == "success": st.success(_banner_text)
            elif _color == "warning": st.warning(_banner_text)
            else: st.error(_banner_text)

            # ── Freshness gauge ───────────────────────────────────────────────
            st.markdown(
                f"**Signal Freshness:** `[{_fresh_bar}]` {_freshness:.0f}/100  —  {_fresh_label}"
            )

            # ── Row 1: Signal origin ───────────────────────────────────────────
            st.markdown("#### 📍 When & Where the Signal Fired")
            _r1c1,_r1c2,_r1c3,_r1c4,_r1c5 = st.columns(5)
            _r1c1.metric("Signal",        _sig_dir_label)
            _r1c2.metric("Signal Date",   to_ist(_sig_bar_dt).split(" ")[0])
            _r1c3.metric("Signal Time",   " ".join(to_ist(_sig_bar_dt).split(" ")[1:]))
            _r1c4.metric("Signal Candle", f"Bar {_sig_bars_ago} back" if _sig_bars_ago > 0 else "Last closed bar")
            _r1c5.metric("Timeframe",     interval)

            # ── Row 2: Price context ───────────────────────────────────────────
            st.markdown("#### 💰 Price Context")
            _r2c1,_r2c2,_r2c3,_r2c4,_r2c5 = st.columns(5)
            _r2c1.metric("Price at Signal",   f"{_sig_bar_price:.2f}",
                         help="Close price of the candle where signal fired")
            _r2c2.metric("Current LTP",        f"{cl:.2f}")
            _r2c3.metric("Price Moved",         f"{abs(_price_move):.2f} pts",
                         delta=f"{'in your favor ✅' if _favorable else 'against signal ⚠️'}",
                         delta_color="normal" if _favorable else "inverse")
            _r2c4.metric("Direction of Move",
                         f"{'↓ Down' if cl < _sig_bar_price else '↑ Up'}",
                         help="Has price moved in the signal direction since it fired?")
            _r2c5.metric("SL Consumed",       f"{_move_vs_sl*100:.0f}%",
                         delta=f"{'Safe' if _move_vs_sl < 0.4 else 'Caution' if _move_vs_sl < 0.65 else 'DANGER'}",
                         delta_color="normal" if _move_vs_sl < 0.4 else "inverse")

            # ── Row 3: Entry-now levels ────────────────────────────────────────
            st.markdown("#### 🎯 If You Enter RIGHT NOW at LTP")
            _r3c1,_r3c2,_r3c3,_r3c4,_r3c5 = st.columns(5)
            _r3c1.metric("Entry Price (LTP)",   f"{cl:.2f}")
            _r3c2.metric("SL if Enter Now",     f"{_sl_now:.2f}",
                         help=f"{sl_pts:.0f} pts {'below' if last_sig==1 else 'above'} LTP")
            _r3c3.metric("Target if Enter Now", f"{_tgt_now:.2f}",
                         help=f"{tgt_pts:.0f} pts {'above' if last_sig==1 else 'below'} LTP")
            _r3c4.metric("R:R Ratio",           f"1 : {_rr_now:.1f}",
                         help=f"Risk {sl_pts:.0f} pts to make {tgt_pts:.0f} pts")
            _r3c5.metric("Target % Remaining",  f"{max(0,100-_tgt_consumed_pct):.0f}%",
                         help="How much of the original target move is still left to capture")

            # ── Row 4: Original signal levels (for reference) ──────────────────
            with st.expander("📊 Original signal-bar levels (for reference)", expanded=False):
                _r4c1,_r4c2,_r4c3,_r4c4 = st.columns(4)
                _r4c1.metric("Signal-bar Entry",  f"{_sig_bar_price:.2f}")
                _r4c2.metric("Original SL",        f"{_sl_orig:.2f}",
                             help=f"SL if you had entered AT the signal bar")
                _r4c3.metric("Original Target",    f"{_tgt_orig:.2f}",
                             help=f"Target if you had entered AT the signal bar")
                _r4c4.metric("Missed move",        f"{abs(_price_move):.2f} pts",
                             help="Price already captured since signal — you missed this portion")

            # ── Detailed guidance text ─────────────────────────────────────────
            st.markdown("#### 📋 Entry Guidance")
            if _verdict == "FRESH":
                st.success(
                    f"✅ **ENTER NOW — Signal is brand new (just fired on the last closed candle).**\n\n"
                    f"The {_sig_dir_label} signal fired at **{to_ist(_sig_bar_dt)}** on the candle that just closed. "
                    f"Price at signal: **{_sig_bar_price:.2f}** | Current LTP: **{cl:.2f}**.\n\n"
                    f"You are entering at the ideal time — 0 bars of delay. "
                    f"Set SL at **{_sl_now:.2f}** ({sl_pts:.0f} pts) and Target at **{_tgt_now:.2f}** ({tgt_pts:.0f} pts). "
                    f"R:R = 1:{_rr_now:.1f}."
                )
            elif _verdict == "VALID":
                st.success(
                    f"✅ **ENTER — Signal is still fresh and valid.**\n\n"
                    f"Signal fired **{_sig_bars_ago} candle(s) ago** at **{to_ist(_sig_bar_dt)}** (~{_mins_elapsed} min elapsed). "
                    f"Price has moved **{abs(_price_move):.2f} pts** {'in your favor' if _favorable else 'against signal'} since then.\n\n"
                    f"Only **{_move_vs_sl*100:.0f}%** of your SL distance has been consumed — there is still meaningful room. "
                    f"Enter at **{cl:.2f}**, SL **{_sl_now:.2f}**, Target **{_tgt_now:.2f}**. R:R = 1:{_rr_now:.1f}.\n\n"
                    f"*Note: You missed {abs(_price_move):.2f} pts of the move. Your target is still fully reachable.*"
                )
            elif _verdict == "BORDERLINE":
                st.warning(
                    f"⚠️ **ENTER WITH CAUTION — Signal is {_sig_bars_ago} candle(s) old (~{_mins_elapsed} min).**\n\n"
                    f"Price has moved **{abs(_price_move):.2f} pts** in signal direction, consuming **{_move_vs_sl*100:.0f}%** "
                    f"of your SL distance. The R:R is reduced from 1:{_rr_now:.1f} to approximately "
                    f"1:{(_tgt_now - cl)*last_sig / max(sl_pts,0.01):.1f} adjusted for missed move.\n\n"
                    f"**If you enter:** SL = **{_sl_now:.2f}**, Target = **{_tgt_now:.2f}**.\n\n"
                    f"**Recommendation:** Enter with reduced position size (50% of normal). "
                    f"If the next candle closes in signal direction, that's additional confirmation. "
                    f"Do NOT chase if price spikes further — wait for a small pullback to the **{_sig_bar_price:.2f}** level."
                )
            elif _verdict == "TOO LATE":
                st.error(
                    f"🔴 **DO NOT ENTER — Signal is too old and the move has largely happened.**\n\n"
                    f"Signal fired **{_sig_bars_ago} candle(s) ago** at **{to_ist(_sig_bar_dt)}** (~{_mins_elapsed} min ago). "
                    f"Price moved **{abs(_price_move):.2f} pts** — that is **{_move_vs_sl*100:.0f}%** of your SL already used up.\n\n"
                    f"If you enter at **{cl:.2f}** with SL at **{_sl_now:.2f}**, your actual risk (SL distance) is the same "
                    f"{sl_pts:.0f} pts, but the remaining target move is only ~{max(0, tgt_pts - abs(_price_move)):.0f} pts "
                    f"— R:R has collapsed.\n\n"
                    f"**What to do:** Do NOT enter. The FOMO urge is strong here — resist it. "
                    f"Monitor for the next corrective pullback. If price pulls back close to **{_sig_bar_price:.2f}** "
                    f"and holds, that might be a secondary entry. Or wait for the strategy to fire a fresh signal."
                )
            elif _verdict == "PULLBACK":
                st.warning(
                    f"🟡 **MONITOR — Price pulled back against signal direction after firing.**\n\n"
                    f"Signal fired at **{to_ist(_sig_bar_dt)}** (candle {_sig_bars_ago} back). "
                    f"Price has moved **{abs(_price_move):.2f} pts AGAINST** the signal direction — "
                    f"this is a pullback, NOT necessarily a signal failure.\n\n"
                    f"**Pullback context:** In Elliott Wave, a pullback after the signal bar is expected and normal. "
                    f"The signal remains valid as long as price stays {'above' if last_sig==1 else 'below'} "
                    f"the original signal-bar {'low' if last_sig==1 else 'high'}.\n\n"
                    f"**Action:** Watch the current candle. If it closes {'above' if last_sig==1 else 'below'} "
                    f"**{_sig_bar_price:.2f}** (signal close), the pattern is intact — enter on the next bar open. "
                    f"If it closes {'below' if last_sig==1 else 'above'} that level, skip this signal."
                )
            else:  # INVALIDATED
                st.error(
                    f"🔴 **DO NOT ENTER — Signal has been invalidated by price action.**\n\n"
                    f"Signal fired at **{to_ist(_sig_bar_dt)}** ({_sig_bars_ago} candles ago). "
                    f"Price has since moved **{abs(_price_move):.2f} pts against** the signal direction, "
                    f"exceeding **{_move_vs_sl*100:.0f}%** of your SL.\n\n"
                    f"The wave structure that generated this signal is no longer reliable. "
                    f"Entering now would be a FOMO trade with very poor risk-reward.\n\n"
                    f"**What to do:** Close this panel mentally, wait for the next fresh signal. "
                    f"The strategy will fire again — patience is more profitable than chasing."
                )

            # ── Late-start handler: if user just started monitoring ───────────
            # If the signal fired many bars ago and user just opened the app,
            # show a dedicated guidance section.
            _just_started = tick <= 3   # first 3 ticks = user probably just started
            if _sig_bars_ago >= 3 and _just_started:
                st.markdown("#### 🆕 You Started Late — Here's What To Do")
                _late_start_adj = st.session_state.get("adj_target_enabled", False)
                if _verdict in ("TOO LATE", "INVALIDATED"):
                    st.error(
                        f"**You opened this app after the signal already fired and the move has largely happened.**\n\n"
                        f"Signal fired at **{to_ist(_sig_bar_dt)}** — that was **{_sig_bars_ago} candles** "
                        f"({_mins_elapsed} min) ago. Price has already moved **{abs(_price_move):.2f} pts** "
                        f"which is **{_move_vs_sl*100:.0f}%** of your SL distance.\n\n"
                        f"**Do NOT enter this signal.** Chasing it now is a FOMO trade.\n\n"
                        f"✅ **What to do:** Keep live trading running. The strategy will fire the next "
                        f"signal on a fresh candle. When it does, the freshness gauge will show 🟢 FRESH "
                        f"and you'll have a clean entry. Patience here saves money."
                    )
                elif _verdict in ("BORDERLINE", "VALID"):
                    st.warning(
                        f"**You started monitoring after the signal fired, but it may still be tradeable.**\n\n"
                        f"Signal at **{to_ist(_sig_bar_dt)}** ({_sig_bars_ago} candles ago, ~{_mins_elapsed} min). "
                        f"Price moved **{abs(_price_move):.2f} pts** in signal direction — "
                        f"**{_move_vs_sl*100:.0f}%** of SL consumed.\n\n"
                        f"{'✅ Adjusted Target is ON — your target will be reduced to the remaining move.' if _late_start_adj else '⚠️ Consider enabling **Adjusted Target (Late Entry Mode)** in Trade Management sidebar — it will automatically set your target to the remaining move portion, not the original full target.'}\n\n"
                        f"**If you enter:** SL = **{_sl_now:.2f}**, Target = **{_tgt_now:.2f}** "
                        f"({'adjusted' if _late_start_adj else 'original — consider enabling Adjusted Target'}). "
                        f"Use reduced position size (50% normal)."
                    )
                else:
                    st.success(
                        f"**You started monitoring recently and the signal is still fresh enough.**\n\n"
                        f"Signal at **{to_ist(_sig_bar_dt)}** ({_sig_bars_ago} candle(s) ago). "
                        f"Price barely moved — you haven't missed much. Enter at full size."
                    )
                try:
                    _sig_bar_low  = float(lv_df["Low"].iloc[-(_sig_bars_ago+1)])
                    _sig_bar_high = float(lv_df["High"].iloc[-(_sig_bars_ago+1)])
                    _invalidation = _sig_bar_low if last_sig==1 else _sig_bar_high
                    _still_valid  = (cl > _invalidation) if last_sig==1 else (cl < _invalidation)
                    st.info(
                        f"**Elliott Wave structure validity check:**\n\n"
                        f"Signal candle: Open {lv_df['Open'].iloc[-(_sig_bars_ago+1)]:.2f} | "
                        f"High {_sig_bar_high:.2f} | Low {_sig_bar_low:.2f} | Close {_sig_bar_price:.2f}\n\n"
                        f"**Invalidation level: {_invalidation:.2f}** "
                        f"({'signal-bar Low — if price breaks below this the BUY is invalid' if last_sig==1 else 'signal-bar High — if price breaks above this the SELL is invalid'})\n\n"
                        f"Current price {cl:.2f} is {'✅ ABOVE' if last_sig==1 and _still_valid else '✅ BELOW' if last_sig==-1 and _still_valid else '❌ BREACHED'} "
                        f"the invalidation level → Wave pattern is "
                        f"{'✅ STILL VALID' if _still_valid else '❌ INVALIDATED — do not enter'}."
                    )
                except: pass

            st.markdown("---")

        # ── EW Next-Pivot Predictor (when freshness enabled + EW strategy) ────
        if (st.session_state.get("freshness_enabled", False) and
                "Elliott Wave" in strategy):
            _ew_pred_mwp = float(live_params.get("min_wave_pct", 0.5))
            try:
                _ew_all_pivots, (last_dir_raw, last_px_raw, last_idx_raw) = \
                    _ew_build_pivots(lv_df, _ew_pred_mwp)

                st.markdown("---")
                st.markdown("### 🔮 Elliott Wave — Next Pivot Predictor")
                st.caption(
                    "Because yfinance delivers 4-7 min delayed candles, you often cannot "
                    "act on the CLOSED bar signal in time. This panel predicts **where the "
                    "next pivot will form** and gives you exact entry/SL/target levels to "
                    "enter MANUALLY in your broker app — before the signal bar even closes."
                )

                if len(_ew_all_pivots) >= 2:
                    _lp    = _ew_all_pivots[-1]   # last confirmed pivot
                    _lp2   = _ew_all_pivots[-2]   # pivot before that
                    _lp_dir  = _lp[2]             # 1=High, -1=Low
                    _lp_px   = _lp[1]
                    _lp2_px  = _lp2[1]
                    _leg_sz  = abs(_lp_px - _lp2_px)

                    # ── Current in-progress swing info ─────────────────────────
                    _cur_move_pct = (_ew_pred_mwp / 100) * _lp_px
                    _next_pivot_dir = "LOW ↓" if _lp_dir == 1 else "HIGH ↑"
                    _next_pivot_side = -1 if _lp_dir == 1 else 1

                    # Price needs to move _cur_move_pct in _next_pivot_side direction
                    # from current extreme to confirm next pivot
                    if last_dir_raw == _lp_dir:
                        # In-progress swing is same direction as last pivot — tracking current extreme
                        _swing_extreme = last_px_raw
                    else:
                        _swing_extreme = cl

                    # Next pivot confirmation threshold
                    _pivot_confirm_px = _swing_extreme * (
                        1 - _ew_pred_mwp/100 if _lp_dir == 1 else 1 + _ew_pred_mwp/100
                    )
                    _dist_to_pivot    = abs(cl - _pivot_confirm_px)
                    _pct_to_pivot     = abs(cl - _pivot_confirm_px) / max(abs(_pivot_confirm_px),0.01) * 100

                    # What signal will fire when next pivot confirms?
                    # Pattern needs 3 pivots: check last 2 + what the next will be
                    _next_sig_type = None
                    if len(_ew_all_pivots) >= 2:
                        # After next pivot forms, we have triplet: _lp2, _lp, next_pivot
                        # LONG fires if: _lp2=LOW, _lp=HIGH, next=HIGHER LOW
                        # SHORT fires if: _lp2=HIGH, _lp=LOW, next=LOWER HIGH
                        if _lp2[2]==-1 and _lp[2]==1:
                            # Next will be LOW — LONG if higher than _lp2
                            _proj_low_px = cl - _dist_to_pivot
                            if _proj_low_px > _lp2_px:
                                _next_sig_type = "LONG 🟢"
                                _next_sig_entry = _proj_low_px
                            else:
                                _next_sig_type = "No signal (lower low = bearish)"
                                _next_sig_entry = None
                        elif _lp2[2]==1 and _lp[2]==-1:
                            # Next will be HIGH — SHORT if lower than _lp2
                            _proj_high_px = cl + _dist_to_pivot
                            if _proj_high_px < _lp2_px:
                                _next_sig_type = "SHORT 🔴"
                                _next_sig_entry = _proj_high_px
                            else:
                                _next_sig_type = "No signal (higher high = bullish)"
                                _next_sig_entry = None
                        else:
                            _next_sig_type = "Pattern building…"
                            _next_sig_entry = None
                    else:
                        _next_sig_type = "Need more pivots"
                        _next_sig_entry = None

                    # ── Display ───────────────────────────────────────────────
                    _pred_c1,_pred_c2,_pred_c3,_pred_c4 = st.columns(4)
                    _pred_c1.metric("Last Confirmed Pivot",
                                    f"{'HIGH' if _lp_dir==1 else 'LOW'} @ {_lp_px:.2f}")
                    _pred_c2.metric("Next Pivot Expected",  _next_pivot_dir)
                    _pred_c3.metric("Pivot Confirm at LTP", f"{_pivot_confirm_px:.2f}",
                                    help=f"Price needs to reach {_pivot_confirm_px:.2f} "
                                         f"(= {_swing_extreme:.2f} ± {_ew_pred_mwp:.2f}%) "
                                         f"to confirm next pivot")
                    _pred_c4.metric("Distance to Confirm",
                                    f"{_dist_to_pivot:.2f} pts  ({_pct_to_pivot:.2f}%)")

                    _pred_c5,_pred_c6,_pred_c7,_pred_c8 = st.columns(4)
                    _pred_c5.metric("Expected Signal",      _next_sig_type or "—")
                    _pred_c6.metric("Projected Entry",
                                    f"{_next_sig_entry:.2f}" if _next_sig_entry else "—")
                    _pred_c7.metric("Projected SL",
                                    f"{_next_sig_entry - sl_pts if _next_sig_entry and last_sig!=0 else '—'}" if _next_sig_entry
                                    else "—",
                                    help=f"Entry ± {sl_pts:.0f} pts (your SL setting)")
                    _pred_c8.metric("Projected Target",
                                    f"{_next_sig_entry + tgt_pts if _next_sig_entry and last_sig!=0 else '—'}" if _next_sig_entry
                                    else "—",
                                    help=f"Entry ± {tgt_pts:.0f} pts (your Target setting)")

                    # Progress bar to next pivot
                    _prog_to_pivot = min(100, max(0,
                        (abs(cl - _lp_px) / max(_dist_to_pivot + abs(cl - _lp_px), 0.01)) * 100
                    ))
                    _bar_f = int(_prog_to_pivot / 5)
                    _bar_s = "█" * _bar_f + "░" * (20 - _bar_f)
                    st.markdown(
                        f"**Progress to next pivot confirmation:** `[{_bar_s}]` {_prog_to_pivot:.0f}%  "
                        f"— need price to reach **{_pivot_confirm_px:.2f}** "
                        f"({_dist_to_pivot:.2f} pts away)"
                    )

                    # ── Manual entry guidance ─────────────────────────────────
                    if _next_sig_entry:
                        _proj_sl  = _next_sig_entry + (_lp_dir * sl_pts)   # opposite of signal direction
                        _proj_tgt = _next_sig_entry - (_lp_dir * tgt_pts)
                        _sig_d    = "LONG 🟢" in (_next_sig_type or "")

                        st.markdown("#### 📋 Manual Entry Plan (for broker app)")
                        if _sig_d:
                            st.success(
                                f"**Projected LONG entry near {_next_sig_entry:.2f}**\n\n"
                                f"yfinance data is delayed — by the time the signal bar closes and the app detects it, "
                                f"price may have already moved away. Use this plan to enter MANUALLY:\n\n"
                                f"• **Watch price:** When LTP approaches **{_pivot_confirm_px:.2f}** and starts reversing up\n"
                                f"• **Enter (BUY):** Around **{_next_sig_entry:.2f} – {_next_sig_entry + 5:.2f}** "
                                f"(allow ±5pt buffer from projected pivot)\n"
                                f"• **SL:** **{_proj_tgt:.2f}** ({sl_pts:.0f} pts below entry)\n"
                                f"• **Target:** **{_next_sig_entry + tgt_pts:.2f}** ({tgt_pts:.0f} pts above entry)\n"
                                f"• **Trigger:** Enter when a 1m candle CLOSES above the pivot confirmation price "
                                f"({_pivot_confirm_px:.2f}) after touching the low\n\n"
                                f"⏱️ Remaining distance: **{_dist_to_pivot:.2f} pts** — "
                                f"{'imminent, watch closely!' if _dist_to_pivot < sl_pts * 0.5 else 'not yet, monitor.'}"
                            )
                        else:
                            st.error(
                                f"**Projected SHORT entry near {_next_sig_entry:.2f}**\n\n"
                                f"• **Watch price:** When LTP approaches **{_pivot_confirm_px:.2f}** and starts reversing down\n"
                                f"• **Enter (SELL):** Around **{_next_sig_entry:.2f} – {_next_sig_entry - 5:.2f}**\n"
                                f"• **SL:** **{_proj_tgt:.2f}** ({sl_pts:.0f} pts above entry)\n"
                                f"• **Target:** **{_next_sig_entry - tgt_pts:.2f}** ({tgt_pts:.0f} pts below entry)\n"
                                f"• **Trigger:** Enter when a 1m candle CLOSES below {_pivot_confirm_px:.2f} after touching the high\n\n"
                                f"⏱️ Remaining distance: **{_dist_to_pivot:.2f} pts** — "
                                f"{'imminent!' if _dist_to_pivot < sl_pts * 0.5 else 'monitor.'}"
                            )
                    else:
                        st.info(
                            f"**{_next_sig_type}** — The upcoming pivot will not immediately generate a "
                            f"tradeable signal. Watch for the pattern to complete over the next few pivots."
                        )

                    # Delay context note
                    st.caption(
                        f"⚠️ yfinance 1m delay is typically 4-7 min. This prediction is based on confirmed pivots "
                        f"and current LTP ({cl:.2f}). The actual pivot confirmation price may shift slightly as new "
                        f"candles form. Treat as a guide — not a guarantee. Use your broker's live chart to time "
                        f"the exact entry candle."
                    )
                else:
                    st.info("Need at least 2 confirmed EW pivots to predict the next one. More data/bars needed.")
            except Exception as _ew_pred_err:
                st.caption(f"EW predictor error: {_ew_pred_err}")

        # ── Indicator values including EMA labels ─────────────────────────────
        _ov_indics={k:v for k,v in lv_indics.items()
                    if isinstance(v,pd.Series) and k not in _SKIP and len(v)>0}
        if _ov_indics:
            st.markdown("**📐 Indicator Values (current bar)**")
            _ic_cols=st.columns(min(len(_ov_indics),6))
            for ci,(name,ser) in enumerate(_ov_indics.items()):
                try:
                    val=float(ser.iloc[-1]); prev=float(ser.iloc[-2]) if len(ser)>1 else val
                    if not np.isnan(val):
                        # Build descriptive label: "EMA 9", "EMA 15", "BB Upper", etc.
                        _lbl = name.replace("_"," ")
                        # Append period if derivable from params
                        if name=="EMA_fast" and "fast" in live_params:    _lbl=f"EMA {live_params['fast']}"
                        elif name=="EMA_slow" and "slow" in live_params:  _lbl=f"EMA {live_params['slow']}"
                        elif name=="EMA_trend" and "trend_ema" in live_params: _lbl=f"EMA {live_params['trend_ema']} (Trend)"
                        elif name=="EMA_entry" and "entry_ema" in live_params: _lbl=f"EMA {live_params['entry_ema']} (Entry)"
                        elif name=="EMA" and "ema_period" in live_params:  _lbl=f"EMA {live_params['ema_period']}"
                        elif name=="EMA_mid" and "m" in live_params:       _lbl=f"EMA {live_params['m']} (Mid)"
                        elif name=="RSI" and "period" in live_params:      _lbl=f"RSI({live_params.get('period',14)})"
                        elif name=="VWAP":                                  _lbl="VWAP"
                        elif name=="Supertrend":                            _lbl="Supertrend"
                        _ic_cols[ci%len(_ic_cols)].metric(
                            _lbl, f"{val:.2f}",
                            delta=f"{val-prev:+.2f}" if not np.isnan(prev) else None)
                except: pass

        # ── Dhan order helper (inner scope has access to sidebar vars) ────────
        def _dhan_place(direction):
            if not st.session_state.get("dhan_enabled",False): return
            if not dhan_client or not dhan_token: st.warning("Dhan: credentials not set."); return
            # Build list of all accounts: primary + extra
            _accounts = [{"client":dhan_client,"token":dhan_token}]
            for _ea in st.session_state.get("dhan_extra_accounts",[]):
                if _ea.get("client") and _ea.get("token"):
                    _accounts.append(_ea)
            for _acc_idx, _acc in enumerate(_accounts):
                try:
                    from dhanhq import dhanhq as _Dhan
                    _d=_Dhan(_acc["client"],_acc["token"])
                    _acc_label = f"Acc{_acc_idx+1}"
                    if st.session_state.get("dhan_sq_all",False):
                        try: _d.cancel_all_orders()
                        except: pass
                    _ot=_d.MARKET if dhan_entry_ot=="MARKET" else _d.LIMIT
                    _lp=cl if dhan_entry_ot=="LIMIT" else 0
                    if st.session_state.get("dhan_is_stocks",False):
                        _txn=_d.BUY if direction==1 else _d.SELL
                        _exch={"NSE":_d.NSE,"BSE":_d.BSE}.get(dhan_exch,_d.NSE)
                        _stock_sid = dhan_s_sid if dhan_s_sid else sym
                        _d.place_order(
                            security_id   = _stock_sid,
                            exchange_segment = _exch,
                            transaction_type = _txn,
                            quantity      = int(dhan_s_qty),
                            order_type    = _ot,
                            product_type  = _d.INTRA if dhan_prod=="INTRADAY" else _d.DELIVERY,
                            price         = _lp,
                            validity      = "DAY",
                        )
                        st.info(f"Dhan {_acc_label} ({dhan_entry_ot}): "
                                f"{'BUY' if direction==1 else 'SELL'} {dhan_s_qty}x "
                                f"sid={_stock_sid} on {dhan_exch}")
                    else:
                        # ── OPTIONS: BUY CE on LONG signal, BUY PE on SHORT signal ──────
                        _sid  = dhan_ce_sid if direction==1 else dhan_pe_sid
                        _opt  = "CE"        if direction==1 else "PE"
                        if not _sid:
                            st.warning(f"Dhan {_acc_label}: {_opt} Security ID not set. "
                                       "Enter CE/PE security_id in sidebar."); continue
                        # Exchange segment: NSE_OPT for NSE options
                        # Try attribute first (dhanhq ≥2.x), fall back to string constant
                        try:    _exch = _d.NSE_OPT
                        except AttributeError: _exch = "NSE_OPT"
                        # Order type and price per user setting
                        _ot   = _d.LIMIT  if dhan_entry_ot == "LIMIT"  else _d.MARKET
                        _price= float(cl) if dhan_entry_ot == "LIMIT"  else 0
                        # Product type: INTRA for intraday options
                        try:    _prod = _d.INTRA
                        except AttributeError: _prod = "INTRA"
                        _resp = _d.place_order(
                            security_id      = int(_sid),       # must be int
                            exchange_segment = _exch,           # NSE_OPT
                            transaction_type = _d.BUY,          # always buying options
                            quantity         = int(dhan_o_qty),
                            order_type       = _ot,             # LIMIT or MARKET
                            product_type     = _prod,           # INTRA
                            price            = _price,          # LTP for LIMIT, 0 for MARKET
                        )
                        st.info(f"Dhan {_acc_label}: BUY {dhan_o_qty}x {_opt} "
                                f"sid={_sid} | {dhan_entry_ot} @ {_price:.2f if _price else 'MKT'} | resp={_resp}")
                except ImportError: st.error("pip install dhanhq"); break
                except Exception as ex: st.error(f"Dhan {_acc_label} order error: {ex}")

        def _dhan_exit(direction):
            if not st.session_state.get("dhan_enabled",False): return
            if not dhan_client or not dhan_token: return
            try:
                from dhanhq import dhanhq as _Dhan
                _d=_Dhan(dhan_client,dhan_token)
                _ot=_d.MARKET if dhan_exit_ot=="MARKET" else _d.LIMIT
                _lp=cl if dhan_exit_ot=="LIMIT" else 0
                if st.session_state.get("dhan_is_stocks",False):
                    _txn=_d.SELL if direction==1 else _d.BUY
                    _exch={"NSE":_d.NSE,"BSE":_d.BSE}.get(dhan_exch,_d.NSE)
                    _stock_sid = dhan_s_sid if dhan_s_sid else sym
                    _d.place_order(
                        security_id      = _stock_sid,
                        exchange_segment = _exch,
                        transaction_type = _txn,
                        quantity         = int(dhan_s_qty),
                        order_type       = _ot,
                        product_type     = _d.INTRA if dhan_prod=="INTRADAY" else _d.DELIVERY,
                        price            = _lp,
                        validity         = "DAY",
                    )
                    st.info(f"Dhan exit ({dhan_exit_ot}): "
                            f"{'SELL' if direction==1 else 'BUY'} {dhan_s_qty}x sid={_stock_sid}")
                else:
                    # ── OPTIONS EXIT: SELL the option we bought ──────────────────
                    _sid  = dhan_ce_sid if direction==1 else dhan_pe_sid
                    _opt  = "CE"        if direction==1 else "PE"
                    if not _sid: return
                    try:    _exch = _d.NSE_OPT
                    except AttributeError: _exch = "NSE_OPT"
                    _ot   = _d.MARKET if dhan_exit_ot == "MARKET" else _d.LIMIT
                    _price= float(cl)  if dhan_exit_ot == "LIMIT"  else 0
                    try:    _prod = _d.INTRA
                    except AttributeError: _prod = "INTRA"
                    _resp = _d.place_order(
                        security_id      = int(_sid),   # must be int
                        exchange_segment = _exch,       # NSE_OPT
                        transaction_type = _d.SELL,     # square off by selling
                        quantity         = int(dhan_o_qty),
                        order_type       = _ot,
                        product_type     = _prod,
                        price            = _price,
                    )
                    st.info(f"Dhan exit: SELL {dhan_o_qty}x {_opt} sid={_sid} | "
                            f"{dhan_exit_ot} @ {_price:.2f if _price else 'MKT'} | resp={_resp}")
            except Exception as ex: st.error(f"Dhan exit error: {ex}")

        # ── Position management ──────────────────────────────────────────────
        # pos was already read above (before signal timing panel)
        _allow_new = (_in_window and _cooldown_ok and
                      (not st.session_state.get("no_overlap",True) or pos is None))

        # ── Entry Gate Diagnostic — shows exactly why algo did/didn't enter ──
        with st.expander("🔑 Entry Gate Status — why algo entered or didn't", expanded=(last_sig!=0)):
            _g1,_g2,_g3,_g4 = st.columns(4)
            _g1.metric("Strategy Signal", "🟢 BUY" if last_sig==1 else ("🔴 SELL" if last_sig==-1 else "⚪ FLAT (0)"),
                       help="0 = no signal on last closed bar(s). Entry requires non-zero.")
            _g2.metric("Time Window", "✅ In window" if _in_window else "❌ Outside window",
                       help="Time window filter in Trade Management. If disabled always ✅.")
            _g3.metric("Cooldown", "✅ Ready" if _cooldown_ok else "⏳ Waiting",
                       help="Cooldown between trades. Counts down after each exit.")
            _g4.metric("Position Slot", "✅ Free" if pos is None else "🔒 Position open",
                       help="Overlap prevention. Only one position at a time when enabled.")
            if last_sig == 0:
                st.warning(
                    "**No entry because: strategy signal = 0 on last closed bar(s).** "
                    "The Signal Progress expander shows *approximate diagnostic conditions* — "
                    "they are display helpers, not the exact signal computation. "
                    "The actual entry fires when `sig_elliott_wave()` (or whichever strategy) "
                    "places a non-zero value at a recent bar. "
                    "If the Signal Progress shows all conditions met but signal is still 0, "
                    "it means the pivot hasn't been *confirmed and placed* at a bar within the "
                    f"lookback window ({_WIDE_LOOKBACK.get(strategy, 1)} bars for {strategy}). "
                    "Keep watching — once the pattern completes on a closed bar the algo WILL enter."
                )
            elif not _allow_new:
                st.warning("Signal detected but entry blocked — check gate status above.")
            else:
                if pos is None:
                    st.success(f"✅ All gates open — entry triggered! Signal={last_sig}")
                else:
                    st.info("Position already open — monitoring for exit.")

        if pos is None and last_sig!=0 and _allow_new:
            d=last_sig; ep=cl
            lv_sl =init_sl(lv_df,lv_n-1,ep,d,sl_type,sl_pts,live_params)
            lv_tgt=init_tgt(lv_df,lv_n-1,ep,d,tgt_type,tgt_pts,lv_sl,live_params)

            # ── Adjusted Target (Late Entry Mode) ─────────────────────────────
            # When price has already moved in signal direction, reduce target by
            # the amount already captured so exit is realistic for remaining move.
            _adj_tgt_note = ""
            if st.session_state.get("adj_target_enabled", False) and _sig_bar_price is not None:
                _already_moved = (ep - _sig_bar_price) * d   # positive if price moved in signal dir
                if _already_moved > 0:
                    _remaining_tgt_pts = max(tgt_pts - _already_moved, sl_pts * 0.5)  # floor at 0.5×SL
                    _adj_lv_tgt = ep + d * _remaining_tgt_pts
                    _adj_tgt_note = (
                        f" [Adjusted: signal moved {_already_moved:.1f}pts already, "
                        f"remaining target = {_remaining_tgt_pts:.1f}pts → {_adj_lv_tgt:.2f}]"
                    )
                    lv_tgt = _adj_lv_tgt

            # With LTP-based SL/Target checking there's no false immediate trigger.
            st.session_state.live_position={"entry":ep,"direction":d,"sl":lv_sl,"target":lv_tgt,
                "disp_tgt":lv_tgt,"entry_time":last_bar,"highest":ep,"lowest":ep}
            _dhan_place(d)
            _sig_label = "BUY (LONG)" if d==1 else "SELL (SHORT)"
            st.success(f"🚀 NEW {'LONG' if d==1 else 'SHORT'}  Entry:{ep:.2f}  "
                       f"SL:{lv_sl:.2f}  Target:{lv_tgt:.2f}  "
                       f"[{to_ist(last_bar)}]{_adj_tgt_note}")
            # ── Email alert: new signal ───────────────────────────────────
            if st.session_state.get("email_enabled",False):
                send_alert(
                    subject=f"AlgoTrader: {_sig_label} Signal — {sym}",
                    body=(f"Strategy: {strategy}\n"
                          f"Ticker:   {sym}\n"
                          f"Signal:   {_sig_label}\n"
                          f"Entry:    {ep:.2f}\n"
                          f"SL:       {lv_sl:.2f}\n"
                          f"Target:   {lv_tgt:.2f}\n"
                          f"Time:     {to_ist(last_bar)}\n"
                          f"Interval: {interval}\n"),
                    sender=email_sender, app_password=email_apppass, to=email_to,
                )

        elif pos is not None:
            d=pos["direction"]; ep=pos["entry"]
            pos["highest"]=max(pos["highest"],bh_cur); pos["lowest"]=min(pos["lowest"],bl_cur)
            pos["sl"]=update_sl(lv_df,lv_n-1,ep,d,sl_type,sl_pts,pos["sl"],live_params)
            new_t,tf=update_tgt(lv_df,lv_n-1,d,tgt_type,tgt_pts,pos["disp_tgt"],live_params)
            pos["disp_tgt"]=new_t
            if tf: pos["target"]=new_t
            exited=False; exit_px=None; exit_why=None

            # ── Strategy Signal Exit: exit when strategy fires opposite signal ──
            # This mirrors the same logic in run_backtest (was missing in live trading)
            _live_sig_exit = (sl_type in ("EMA Reverse Crossover","Strategy Signal Exit") or
                              tgt_type in ("Reverse EMA Crossover","Strategy Signal Exit"))
            if _live_sig_exit:
                _rev_sig = int(lv_sigs.iloc[-2]) if len(lv_sigs)>1 else 0
                if _rev_sig != 0 and _rev_sig != d:
                    exited,exit_px,exit_why = True, cl, "Strategy Signal Exit"

            if not exited:
                # ── LIVE TRADING: use LTP (last traded price = cl) for SL/Target ──
                # BACKTEST uses candle High/Low because it processes closed bars where the
                # full range is known. LIVE TRADING processes tick-by-tick where the current
                # candle is still forming — its High/Low includes price action from the entire
                # bar so far, which can falsely trigger SL on a spike even while trend continues.
                #
                # Example (your data): SHORT at 71194, SL=71204. Current candle High=71252
                # (accumulated from bar start). Even though price is now at 71170 and falling,
                # bh_cur=71252 > SL=71204 → SL fires wrongly. Using cl=71170 → no SL hit ✓
                #
                # Rule: compare SL/Target against the LAST TRADED PRICE (cl) only.
                # The position tracks `highest` and `lowest` for display — still updated.
                if d==1:
                    if cl <= pos["sl"]:             exited,exit_px,exit_why=True,pos["sl"],"SL Hit"
                    elif tf and cl >= pos["target"]:exited,exit_px,exit_why=True,pos["target"],"Target Hit"
                else:
                    if cl >= pos["sl"]:             exited,exit_px,exit_why=True,pos["sl"],"SL Hit"
                    elif tf and cl <= pos["target"]:exited,exit_px,exit_why=True,pos["target"],"Target Hit"
            if not exited and st.session_state.get("time_filter",False) and not _in_window:
                exited,exit_px,exit_why=True,cl,"Time Window Close"
            if exited:
                pnl=round((exit_px-ep)*d,4)
                # ── DEDUP: create a fingerprint of this trade to prevent double-recording ──
                # The fragment can re-render mid-execution causing the same exit to be
                # appended multiple times. Guard with a unique key in session state.
                _trade_fp = f"{sym}_{ep:.4f}_{to_ist(pos['entry_time'])}_{d}"
                _already_recorded = st.session_state.get("_last_trade_fp","") == _trade_fp
                if not _already_recorded:
                    st.session_state["_last_trade_fp"] = _trade_fp
                    st.session_state.live_trades.append({
                        "Ticker":     sym,
                        "Strategy":   strategy,
                        "Entry Time": to_ist(pos["entry_time"]),
                        "Entry Price":round(ep,2),
                        "Direction":  "LONG" if d==1 else "SHORT",
                        "Exit Time":  to_ist(last_bar),
                        "Exit Price": round(exit_px,2),
                        "Exit Reason":exit_why,
                        "SL":         round(pos["sl"],2),
                        "Target":     round(pos["disp_tgt"],2),
                        "Highest":    round(pos["highest"],2),
                        "Lowest":     round(pos["lowest"],2),
                        "PnL":        pnl,
                    })
                # ── Graceful full reset — MUST happen before any st.* call ────
                # Setting live_position=None here prevents the fragment from
                # re-reading the same pos dict if it re-renders before this tick ends.
                st.session_state.live_position = None
                st.session_state.last_trade_close_ts = time.time()
                _dhan_exit(d)
                if not _already_recorded:
                    (st.success if pnl>0 else st.error)(
                        f"CLOSED {exit_why} | PnL: {'+'if pnl>0 else ''}{pnl:.2f} | {to_ist(last_bar)}")
                # ── Email alert: trade exit ───────────────────────────────
                if st.session_state.get("email_enabled",False):
                    send_alert(
                        subject=f"AlgoTrader: Trade EXIT ({exit_why}) — {sym}  PnL:{'+' if pnl>0 else ''}{pnl:.2f}",
                        body=(f"Strategy:    {strategy}\n"
                              f"Ticker:      {sym}\n"
                              f"Direction:   {'LONG' if d==1 else 'SHORT'}\n"
                              f"Entry Price: {ep:.2f}\n"
                              f"Exit Price:  {exit_px:.2f}\n"
                              f"Exit Reason: {exit_why}\n"
                              f"PnL:         {'+' if pnl>0 else ''}{pnl:.2f} pts\n"
                              f"SL Level:    {pos['sl']:.2f}\n"
                              f"Target:      {pos['disp_tgt']:.2f}\n"
                              f"Highest:     {pos['highest']:.2f}\n"
                              f"Lowest:      {pos['lowest']:.2f}\n"
                              f"Exit Time:   {to_ist(last_bar)}\n"
                              f"Interval:    {interval}\n"),
                        sender=email_sender, app_password=email_apppass, to=email_to,
                    )
            else:
                unreal=round((cl-ep)*d,4)
                st.markdown("#### 📌 Open Position")
                p_=st.columns(8)
                p_[0].metric("Direction","🟢 LONG" if d==1 else "🔴 SHORT")
                p_[1].metric("Entry",f"{ep:.2f}")
                p_[2].metric("LTP",f"{cl:.2f}")
                p_[3].metric("SL",f"{pos['sl']:.2f}",delta=f"↓{abs(cl-pos['sl']):.2f}",delta_color="inverse")
                p_[4].metric("Target",f"{pos['disp_tgt']:.2f}",delta=f"↑{abs(pos['disp_tgt']-cl):.2f}")
                p_[5].metric("Highest",f"{pos['highest']:.2f}")
                p_[6].metric("Lowest",f"{pos['lowest']:.2f}")
                p_[7].metric("Unrealised",f"{unreal:.2f}",delta=f"{unreal:+.2f}",
                              delta_color="normal" if unreal>=0 else "inverse")

        st.plotly_chart(plot_ohlc(lv_df,indics=lv_indics,title=f"LIVE:{t_choice}({interval}) Tick#{tick}"),use_container_width=True,key=f"lv_ohlc_{tick}")

        # ── Compact closed-trade summary — always visible without switching tabs ──
        # Filter by current ticker so other-ticker trades never pollute display
        _all_closed  = st.session_state.get("live_trades", [])
        _closed      = [t for t in _all_closed if t.get("Ticker","") == sym]
        _other_count = len(_all_closed) - len(_closed)
        if _closed:
            _ct=len(_closed); _cw=sum(1 for x in _closed if x["PnL"]>0)
            _cp=sum(x["PnL"] for x in _closed)
            st.markdown("---")
            _sm1,_sm2,_sm3,_sm4 = st.columns(4)
            _sm1.metric("Closed Trades", _ct, help=f"For {sym} only. {_other_count} trades for other tickers hidden.")
            _sm2.metric("Win Rate",  f"{_cw/_ct*100:.1f}%")
            _sm3.metric("Total PnL", f"{_cp:+.2f}", delta=f"{_cp:+.2f}",
                        delta_color="normal" if _cp>=0 else "inverse")
            _sm4.metric("Last Exit", _closed[-1].get("Exit Reason","—"))
            with st.expander(f"📜 Last 5 trades for {sym} (full list in Trade History tab)", expanded=False):
                _last5 = pd.DataFrame(_closed[-5:])
                # Show full datetime — not truncated
                _disp_cols = [c for c in ["Entry Time","Exit Time","Direction","Entry Price",
                                          "Exit Price","SL","Target","PnL","Exit Reason"]
                              if c in _last5.columns]
                def _sc2(v):
                    if isinstance(v,(int,float)): return "color:#00E676" if v>0 else ("color:#FF5252" if v<0 else "")
                    return ""
                st.dataframe(
                    _last5[_disp_cols].style.map(_sc2, subset=["PnL"]),
                    use_container_width=True,
                    key=f"lv_last5_{_ct}_{sym}",
                    column_config={
                        "Entry Time": st.column_config.TextColumn("Entry Time", width="large"),
                        "Exit Time":  st.column_config.TextColumn("Exit Time",  width="large"),
                    }
                )
        elif _other_count:
            st.info(f"No trades for **{sym}** yet. ({_other_count} trades exist for other tickers — switch ticker to view them.)")

        # ── Signal Progress Expander ─────────────────────────────────────────────
        with st.expander("📊 Signal Progress — Current indicator state & distance to signal", expanded=False):
            st.caption(
                "Shows **every indicator** for the selected strategy with current values, "
                "direction of movement, and distance to the next signal trigger. "
                "Updated every tick — no need to watch the screen all day."
            )

            # ── Helper: safely get last 2 values from a Series ───────────────
            def _last2(ser):
                """Return (current, previous) floats from a Series. None if unavailable."""
                try:
                    clean = ser.dropna()
                    if len(clean) == 0: return None, None
                    cur  = float(clean.iloc[-1])
                    prev = float(clean.iloc[-2]) if len(clean) > 1 else cur
                    return cur, prev
                except:
                    return None, None

            def _trend_arrow(cur, prev):
                if cur is None or prev is None: return "—"
                if cur > prev:  return "rising ↗"
                if cur < prev:  return "falling ↘"
                return "flat →"

            def _est_bars(gap, delta, label="crossover"):
                """Rough estimate of bars until gap closes."""
                if delta is None or delta == 0 or gap is None: return "—"
                if gap <= 0: return "Signal may fire now"
                bars = abs(gap) / max(abs(delta), 1e-6)
                if bars < 1:   return "< 1 bar"
                if bars < 3:   return f"~{int(bars)+1} bars"
                if bars < 10:  return f"~{int(bars)} bars"
                return f"~{int(bars)} bars (distant)"

            _sp_items = []

            # ════════════════════════════════════════════════════════════════
            # STRATEGY-SPECIFIC SMART ANALYSIS
            # Each block computes signal distance for the indicators it knows
            # ════════════════════════════════════════════════════════════════

            _handled_keys = set()  # track which indic keys got smart treatment

            # ── EMA pairs (crossover strategies) ─────────────────────────────
            _ema_pairs = [
                ("EMA_fast","EMA_slow","fast","slow"),
                ("EMA_trend","EMA_entry","trend_ema","entry_ema"),
                ("EMA_fast","EMA_mid","f","m"),
                ("EMA_mid","EMA_slow","m","s_"),
            ]
            for _ek1, _ek2, _pk1, _pk2 in _ema_pairs:
                if _ek1 in lv_indics and _ek2 in lv_indics:
                    _v1,_p1 = _last2(lv_indics[_ek1])
                    _v2,_p2 = _last2(lv_indics[_ek2])
                    if _v1 and _v2:
                        _gap      = _v1 - _v2
                        _prev_gap = (_p1 - _p2) if _p1 and _p2 else _gap
                        _delta    = abs(_gap) - abs(_prev_gap)   # negative = narrowing
                        _narrowing= abs(_gap) < abs(_prev_gap)
                        _n1 = live_params.get(_pk1, _ek1)
                        _n2 = live_params.get(_pk2, _ek2)
                        _sp_items.append({
                            "Indicator":  f"{_ek1}({_n1})",
                            "Value":      f"{_v1:.4f}",
                            "vs":         f"{_ek2}({_n2}) = {_v2:.4f}",
                            "Gap":        f"{_gap:+.4f} ({abs(_gap)/max(abs(_v2),0.001)*100:.2f}%)",
                            "Direction":  "Fast > Slow (LONG side)" if _gap>0 else "Fast < Slow (SHORT side)",
                            "Momentum":   ("narrowing ↘ — crossover approaching!" if _narrowing else "widening ↗"),
                            "Est. Bars":  _est_bars(abs(_gap), abs(_delta)/max(1,abs(_prev_gap))*abs(_gap), "crossover") if _narrowing else "—",
                            "Signal":     "🟢 LONG imminent" if (_narrowing and _gap<0.5) else ("🔴 SHORT imminent" if (_narrowing and _gap>-0.5) else "Monitoring"),
                        })
                        _handled_keys.update({_ek1, _ek2})

            # Single EMA (trend reference vs price) ───────────────────────────
            for _ek in ["EMA","EMA_20","EMA_50"]:
                if _ek in lv_indics and _ek not in _handled_keys:
                    _v,_pv = _last2(lv_indics[_ek])
                    if _v:
                        _dist = cl - _v
                        _sp_items.append({
                            "Indicator":  f"{_ek}({live_params.get('ema_period',live_params.get('trend_ema',20))})",
                            "Value":      f"{_v:.4f}",
                            "vs":         f"LTP = {cl:.4f}",
                            "Gap":        f"{_dist:+.4f} ({abs(_dist)/cl*100:.2f}%)",
                            "Direction":  "Price above EMA (bullish)" if _dist>0 else "Price below EMA (bearish)",
                            "Momentum":   _trend_arrow(_v,_pv),
                            "Est. Bars":  "—",
                            "Signal":     "🟢 Bullish" if _dist>0 else "🔴 Bearish",
                        })
                        _handled_keys.add(_ek)

            # ── RSI ───────────────────────────────────────────────────────────
            if "RSI" in lv_indics:
                _rv, _rpv = _last2(lv_indics["RSI"])
                if _rv is not None:
                    _ob  = live_params.get("ob",   live_params.get("rsi_ob",   70))
                    _os  = live_params.get("os_",  live_params.get("rsi_os",   30))
                    _bull_min = live_params.get("rsi_bull_min", _os)
                    _bull_max = live_params.get("rsi_bull_max", _ob)
                    _to_ob = _ob - _rv;  _to_os = _rv - _os
                    _delta_rv = (_rv - _rpv) if _rpv else 0
                    if _rv <= _os:         _sig = "🟢 OVERSOLD — BUY signal triggered"
                    elif _rv >= _ob:       _sig = "🔴 OVERBOUGHT — SELL signal triggered"
                    elif _to_ob < 8:       _sig = f"⚠️ Approaching Overbought ({_rv:.1f}/{_ob}) — SELL soon"
                    elif _to_os < 8:       _sig = f"⚠️ Approaching Oversold ({_rv:.1f}/{_os}) — BUY soon"
                    elif _bull_min <= _rv <= _bull_max:
                        _sig = f"In Bull zone ({_bull_min}-{_bull_max}) — valid for LONG entry"
                    else:                  _sig = f"Neutral ({_rv:.1f})"
                    _sp_items.append({
                        "Indicator":  f"RSI({live_params.get('period',live_params.get('rsi_period',14))})",
                        "Value":      f"{_rv:.2f}",
                        "vs":         f"OB={_ob} | OS={_os}",
                        "Gap":        f"{_to_ob:+.1f} to OB  |  {_to_os:+.1f} to OS",
                        "Direction":  _trend_arrow(_rv,_rpv),
                        "Momentum":   f"Change: {_delta_rv:+.2f}/bar",
                        "Est. Bars":  _est_bars(min(abs(_to_ob),abs(_to_os)), abs(_delta_rv), "threshold"),
                        "Signal":     _sig,
                    })
                    _handled_keys.add("RSI")

            # RSI_OB / RSI_OS threshold lines (display only)
            for _rk in ["RSI_OB","RSI_OS"]:
                if _rk in lv_indics: _handled_keys.add(_rk)

            # ── MACD ──────────────────────────────────────────────────────────
            if "MACD" in lv_indics and "MACD_Signal" in lv_indics:
                _mv,_mpv   = _last2(lv_indics["MACD"])
                _msv,_mspv = _last2(lv_indics["MACD_Signal"])
                if _mv is not None and _msv is not None:
                    _hist      = _mv - _msv
                    _prev_hist = (_mpv-_mspv) if _mpv and _mspv else _hist
                    _conv      = abs(_hist) < abs(_prev_hist)
                    _delta_h   = abs(_hist) - abs(_prev_hist)
                    _sp_items.append({
                        "Indicator": "MACD",
                        "Value":     f"MACD={_mv:.4f}",
                        "vs":        f"Signal={_msv:.4f}",
                        "Gap":       f"Histogram={_hist:+.5f}",
                        "Direction": "Bullish" if _hist>0 else "Bearish",
                        "Momentum":  "converging — crossover near! ⚡" if _conv else "diverging",
                        "Est. Bars": _est_bars(abs(_hist), abs(_delta_h), "crossover") if _conv else "—",
                        "Signal":    "🟢 MACD above Signal (LONG bias)" if _hist>0 else "🔴 MACD below Signal (SHORT bias)",
                    })
                    _handled_keys.update({"MACD","MACD_Signal"})

            # ── Supertrend ────────────────────────────────────────────────────
            if "Supertrend" in lv_indics:
                _stv,_stpv = _last2(lv_indics["Supertrend"])
                if _stv:
                    _dist  = cl - _stv
                    _prev_dist = (cl - _stpv) if _stpv else _dist
                    _approaching = abs(_dist) < abs(_prev_dist)
                    _sp_items.append({
                        "Indicator": f"Supertrend(p={live_params.get('period',7)},m={live_params.get('multiplier',3.0)})",
                        "Value":     f"{_stv:.2f}",
                        "vs":        f"LTP={cl:.2f}",
                        "Gap":       f"{_dist:+.2f} ({abs(_dist)/cl*100:.2f}%)",
                        "Direction": "Bullish — price above line" if _dist>0 else "Bearish — price below line",
                        "Momentum":  "price approaching line — flip possible!" if _approaching else "moving away",
                        "Est. Bars": "< 2 bars (flip imminent!)" if abs(_dist)/cl*100 < 0.25 else "—",
                        "Signal":    "🟢 Bullish trend active" if _dist>0 else "🔴 Bearish trend active",
                    })
                    _handled_keys.add("Supertrend")

            # ── Bollinger Bands ───────────────────────────────────────────────
            if "BB_upper" in lv_indics and "BB_lower" in lv_indics:
                _bu,_bup = _last2(lv_indics["BB_upper"])
                _bl2,_blp= _last2(lv_indics["BB_lower"])
                if _bu and _bl2:
                    _bw = _bu - _bl2
                    _to_u = _bu - cl;  _to_l = cl - _bl2
                    _mid_val = (_bu+_bl2)/2
                    _sp_items.append({
                        "Indicator": f"BB(p={live_params.get('period',20)},σ={live_params.get('std',2)})",
                        "Value":     f"Upper={_bu:.2f} | Mid={_mid_val:.2f} | Lower={_bl2:.2f}",
                        "vs":        f"LTP={cl:.2f}, Width={_bw:.2f}",
                        "Gap":       f"{_to_u:+.2f} to upper | {_to_l:+.2f} to lower",
                        "Direction": "—",
                        "Momentum":  f"Band width {'contracting (squeeze)' if _bw<(_bup-_blp if _bup and _blp else _bw)*0.9 else 'expanding'}",
                        "Est. Bars": "—",
                        "Signal":    ("🔴 Near upper band — SELL zone" if _to_u/cl*100 < 0.4
                                      else "🟢 Near lower band — BUY zone" if _to_l/cl*100 < 0.4 else "Mid range — wait"),
                    })
                    _handled_keys.update({"BB_upper","BB_lower","BB_mid"})

            # ── ADX / DI ──────────────────────────────────────────────────────
            if "ADX" in lv_indics:
                _av,_apv = _last2(lv_indics["ADX"])
                _pdi_v,_ = _last2(lv_indics.get("+DI", pd.Series(dtype=float)))
                _ndi_v,_ = _last2(lv_indics.get("-DI", pd.Series(dtype=float)))
                _thresh   = live_params.get("adx_thresh",25)
                if _av is not None:
                    _sp_items.append({
                        "Indicator": f"ADX({live_params.get('period',14)})",
                        "Value":     f"ADX={_av:.2f}" + (f" | +DI={_pdi_v:.2f} | -DI={_ndi_v:.2f}" if _pdi_v else ""),
                        "vs":        f"Threshold={_thresh}",
                        "Gap":       f"{_av-_thresh:+.2f} from threshold",
                        "Direction": _trend_arrow(_av,_apv),
                        "Momentum":  f"Trend {'STRONG' if _av>_thresh else 'weak — waiting for ' + str(_thresh)}",
                        "Est. Bars": _est_bars(_thresh-_av, abs((_av-_apv) if _apv else 0), f"ADX={_thresh}") if _av<_thresh else "—",
                        "Signal":    ("🟢 Strong bullish (+DI>-DI)" if (_pdi_v and _ndi_v and _pdi_v>_ndi_v and _av>_thresh)
                                      else "🔴 Strong bearish (-DI>+DI)" if (_pdi_v and _ndi_v and _ndi_v>_pdi_v and _av>_thresh)
                                      else f"Weak trend (ADX={_av:.1f} < {_thresh})"),
                    })
                    _handled_keys.update({"ADX","+DI","-DI"})

            # ── Stochastic ────────────────────────────────────────────────────
            if "Stoch_K" in lv_indics and "Stoch_D" in lv_indics:
                _kv,_kpv = _last2(lv_indics["Stoch_K"])
                _dv,_dpv = _last2(lv_indics["Stoch_D"])
                _sob = live_params.get("ob",80); _sos = live_params.get("os_",20)
                if _kv is not None and _dv is not None:
                    _gap_kd   = _kv - _dv
                    _prev_gap = (_kpv-_dpv) if _kpv and _dpv else _gap_kd
                    _conv     = abs(_gap_kd) < abs(_prev_gap)
                    _sp_items.append({
                        "Indicator": f"Stoch(%K={live_params.get('k',14)},%D={live_params.get('d',3)})",
                        "Value":     f"%K={_kv:.2f} | %D={_dv:.2f}",
                        "vs":        f"OB={_sob} | OS={_sos}",
                        "Gap":       f"%K−%D={_gap_kd:+.2f}",
                        "Direction": _trend_arrow(_kv,_kpv),
                        "Momentum":  "converging — crossover near!" if _conv else "diverging",
                        "Est. Bars": "< 2 bars" if _conv and abs(_gap_kd)<2 else "—",
                        "Signal":    ("🟢 Oversold & %K crossing up — BUY soon" if _kv<_sos and _conv and _kv>_dv
                                      else "🔴 Overbought & %K crossing down — SELL soon" if _kv>_sob and _conv and _kv<_dv
                                      else f"Neutral (%K={_kv:.1f})"),
                    })
                    _handled_keys.update({"Stoch_K","Stoch_D"})

            # ── VWAP ──────────────────────────────────────────────────────────
            for _vk in ["VWAP","VWAP_hi","VWAP_lo"]:
                if _vk == "VWAP" and _vk in lv_indics and _vk not in _handled_keys:
                    _vv,_vpv = _last2(lv_indics["VWAP"])
                    if _vv:
                        _vdist = cl - _vv
                        _vhi = float(lv_indics["VWAP_hi"].dropna().iloc[-1]) if "VWAP_hi" in lv_indics and len(lv_indics["VWAP_hi"].dropna())>0 else None
                        _vlo = float(lv_indics["VWAP_lo"].dropna().iloc[-1]) if "VWAP_lo" in lv_indics and len(lv_indics["VWAP_lo"].dropna())>0 else None
                        _sp_items.append({
                            "Indicator": "VWAP",
                            "Value":     f"{_vv:.2f}",
                            "vs":        f"LTP={cl:.2f}" + (f" | Hi={_vhi:.2f} | Lo={_vlo:.2f}" if _vhi else ""),
                            "Gap":       f"{_vdist:+.2f} ({abs(_vdist)/cl*100:.2f}%)",
                            "Direction": "Price above VWAP (bullish)" if _vdist>0 else "Price below VWAP (bearish)",
                            "Momentum":  _trend_arrow(_vv,_vpv),
                            "Est. Bars": "—",
                            "Signal":    ("🟢 Above VWAP — buy bias" if _vdist>0 else "🔴 Below VWAP — sell bias"),
                        })
                        _handled_keys.update({"VWAP","VWAP_hi","VWAP_lo"})

            # ── Ichimoku ──────────────────────────────────────────────────────
            if "Tenkan" in lv_indics and "Kijun" in lv_indics:
                _tv,_tpv = _last2(lv_indics["Tenkan"])
                _kv,_kpv = _last2(lv_indics["Kijun"])
                if _tv and _kv:
                    _gap = _tv - _kv
                    _sp_items.append({
                        "Indicator": f"Ichimoku Tenkan({live_params.get('tenkan',9)})/Kijun({live_params.get('kijun',26)})",
                        "Value":     f"Tenkan={_tv:.2f} | Kijun={_kv:.2f}",
                        "vs":        f"LTP={cl:.2f}",
                        "Gap":       f"T−K={_gap:+.2f}",
                        "Direction": "Bullish (T>K)" if _gap>0 else "Bearish (K>T)",
                        "Momentum":  _trend_arrow(_tv,_tpv),
                        "Est. Bars": "—",
                        "Signal":    "🟢 Price above cloud" if cl>max(_tv,_kv) else "🔴 Price below cloud" if cl<min(_tv,_kv) else "⚠️ Price inside cloud",
                    })
                    _handled_keys.update({"Tenkan","Kijun","Senkou_A","Senkou_B"})

            # ── Williams %R ───────────────────────────────────────────────────
            if "Williams_%R" in lv_indics:
                _wrv,_wrpv = _last2(lv_indics["Williams_%R"])
                if _wrv is not None:
                    _wob = live_params.get("ob",-20); _wos = live_params.get("os_",-80)
                    _sp_items.append({
                        "Indicator": f"Williams %R({live_params.get('period',14)})",
                        "Value":     f"{_wrv:.2f}",
                        "vs":        f"OB={_wob} | OS={_wos}",
                        "Gap":       f"{_wrv-_wob:+.1f} to OB | {_wos-_wrv:+.1f} to OS",
                        "Direction": _trend_arrow(_wrv,_wrpv),
                        "Momentum":  "—",
                        "Est. Bars": "—",
                        "Signal":    ("🟢 Oversold — BUY zone" if _wrv<_wos else "🔴 Overbought — SELL zone" if _wrv>_wob else "Neutral"),
                    })
                    _handled_keys.add("Williams_%R")

            # ── HV% (IV Proxy) ────────────────────────────────────────────────
            if "HV%" in lv_indics:
                _hvv,_hvpv = _last2(lv_indics["HV%"])
                _hvraw,_   = _last2(lv_indics.get("HV", pd.Series(dtype=float)))
                if _hvv is not None:
                    _ob_pct = live_params.get("ob_pct",80); _os_pct = live_params.get("os_pct",20)
                    _sp_items.append({
                        "Indicator": f"HV Percentile({live_params.get('hv_period',20)}-bar)",
                        "Value":     f"{_hvv:.1f}%" + (f" | HV={_hvraw:.2f}%" if _hvraw else ""),
                        "vs":        f"High={_ob_pct}% | Low={_os_pct}%",
                        "Gap":       f"{_hvv-_ob_pct:+.1f}% from high | {_os_pct-_hvv:+.1f}% from low",
                        "Direction": _trend_arrow(_hvv,_hvpv),
                        "Momentum":  "—",
                        "Est. Bars": "—",
                        "Signal":    (f"🟢 IV Rank LOW ({_hvv:.0f}%) — good to BUY options" if _hvv<_os_pct
                                      else f"🔴 IV Rank HIGH ({_hvv:.0f}%) — option buying expensive" if _hvv>_ob_pct
                                      else f"Neutral IV rank ({_hvv:.0f}%)"),
                    })
                    _handled_keys.update({"HV%","HV"})

            # ── S&R Levels ────────────────────────────────────────────────────
            for _sk, _label in [("Support","Support"), ("Resistance","Resistance")]:
                if _sk in lv_indics and _sk not in _handled_keys:
                    _sv,_spv = _last2(lv_indics[_sk])
                    if _sv:
                        _dist = cl - _sv if _label=="Support" else _sv - cl
                        _sp_items.append({
                            "Indicator": f"{_label}(p={live_params.get('sr_lookback',20)})",
                            "Value":     f"{_sv:.2f}",
                            "vs":        f"LTP={cl:.2f}",
                            "Gap":       f"{_dist:+.2f} ({abs(_dist)/cl*100:.2f}%)",
                            "Direction": _trend_arrow(_sv,_spv),
                            "Momentum":  "—",
                            "Est. Bars": "—",
                            "Signal":    (f"⚠️ Price near {_label}!" if abs(_dist)/cl*100 < 0.5 else f"Away from {_label}"),
                        })
                        _handled_keys.add(_sk)

            # ── Donchian / Breakout levels ─────────────────────────────────────
            for _dk, _label2 in [("Don_upper","Donchian Upper"),("Don_lower","Donchian Lower"),
                                   ("Breakout_Hi","Breakout High"),("Breakout_Lo","Breakout Low"),
                                   ("OR_Hi","ORB High"),("OR_Lo","ORB Low")]:
                if _dk in lv_indics and _dk not in _handled_keys:
                    _dv,_ = _last2(lv_indics[_dk])
                    if _dv:
                        _dist_d = _dv - cl if "upper" in _dk.lower() or "hi" in _dk.lower() or "High" in _label2 else cl - _dv
                        _sp_items.append({
                            "Indicator": f"{_label2}(p={live_params.get('period',live_params.get('lookback',20))})",
                            "Value":     f"{_dv:.2f}",
                            "vs":        f"LTP={cl:.2f}",
                            "Gap":       f"{_dist_d:+.2f} ({abs(_dist_d)/cl*100:.2f}%)",
                            "Direction": "—",
                            "Momentum":  "—",
                            "Est. Bars": "—",
                            "Signal":    "⚡ Price at breakout level!" if abs(_dist_d)/cl*100 < 0.3 else f"{abs(_dist_d)/cl*100:.2f}% away",
                        })
                        _handled_keys.add(_dk)

            # ════════════════════════════════════════════════════════════════
            # ELLIOTT WAVE: dedicated rich diagnostic panel
            # ════════════════════════════════════════════════════════════════
            if strategy in ("Elliott Wave (Simplified)", "Elliott Wave v2 (Swing+Fib)"):
                if strategy == "Elliott Wave v2 (Swing+Fib)":
                    _ew_d = _ew2_diagnostics(
                        lv_df,
                        swing_bars  = int(live_params.get("swing_bars", 5)),
                        fib_min     = float(live_params.get("fib_min", 0.382)),
                        fib_max     = float(live_params.get("fib_max", 0.886)),
                        ema_period  = int(live_params.get("ema_period", 50)),
                    )
                    _mwp = _ew_d.get("min_wave_pct", 0)
                else:
                    _ew_d = _ew_diagnostics(lv_df,
                                             min_wave_pct=float(live_params.get("min_wave_pct",1.0)))
                    _mwp  = _ew_d["min_wave_pct"]
                _cpx  = _ew_d["cur_price"]

                st.markdown("---")
                st.markdown("### Elliott Wave Live Analysis")

                # ── Key explanation for why live trading hasn't entered ────────
                st.info(
                    "**When will live trading enter a position?**\n\n"
                    "The live engine enters ONLY when a new Elliott Wave signal fires on "
                    "the **most recently closed bar** (the bar just before the current forming bar). "
                    "A message like *'SELL signal already fired at pivot C'* means that pattern "
                    "completed several bars ago — the live system correctly did NOT enter late on "
                    "a stale signal.\n\n"
                    "**What to watch for:** When the in-progress swing below reaches "
                    f"**{_mwp:.2f}% move** and flips to a new pivot, completing a fresh "
                    "LOW→HIGH→LOW (for LONG) or HIGH→LOW→HIGH (for SHORT) pattern on the most "
                    "recent bar — that is when live trading will automatically open a position."
                )

                st.caption(
                    f"Zigzag pivot detection — a move of **{_mwp:.2f}%** qualifies as a wave leg. "
                    "Signal fires when a fresh 3-pivot A-B-C corrective sequence completes on the last closed bar."
                )

                # ── Summary metrics row ──────────────────────────────────────
                _em1,_em2,_em3,_em4,_em5 = st.columns(5)
                _em1.metric("Confirmed Pivots",   _ew_d["confirmed_pivots"])
                _em2.metric("LTP",                f"{_cpx:.2f}")
                _em3.metric("Last Pivot Price",   f"{_ew_d['last_confirmed_px']:.2f}")
                _em4.metric("Last Pivot Type",    "Swing HIGH ↑" if _ew_d["last_confirmed_dir"]==1 else
                                                   ("Swing LOW ↓" if _ew_d["last_confirmed_dir"]==-1 else "None yet"))
                _em5.metric("Last Fired Signal",  _ew_d["last_signal"] or "None yet")

                # ── Current in-progress swing ────────────────────────────────
                st.markdown("#### Current In-Progress Swing (unconfirmed)")
                _ew_c1,_ew_c2,_ew_c3,_ew_c4 = st.columns(4)
                _ew_c1.metric("Direction", "Upward ↑" if _ew_d["current_wave_dir"]==1 else
                                            ("Downward ↓" if _ew_d["current_wave_dir"]==-1 else "Not started"))
                _ew_c2.metric("Move so far", f"{_ew_d['move_from_last_pct']:+.2f}%")
                _ew_c3.metric(f"Needed for pivot ({_mwp:.2f}%)",
                               f"{_ew_d['pct_done']:.2f}% done",
                               delta=f"{_ew_d['pct_remaining']:.2f}% remaining",
                               delta_color="inverse")
                _ew_c4.metric("Pivot confirmation", f"{min(100,_ew_d['pivot_flip_pct']):.0f}% there")

                # Visual progress bar for pivot confirmation
                _prog_pct = min(int(_ew_d["pivot_flip_pct"]), 100)
                _bar_filled = int(_prog_pct / 5)
                _prog_bar = "█" * _bar_filled + "░" * (20 - _bar_filled)
                st.markdown(
                    f"**Pivot progress:** `[{_prog_bar}]` {_prog_pct}%  "
                    f"— need price to move **{_ew_d['pct_remaining']:.2f}% more** "
                    f"({_mwp:.2f}% total) to confirm next pivot"
                )

                if _ew_d["retrace_pct"] is not None:
                    st.markdown(
                        f"**Current retracement of last completed leg:** "
                        f"**{_ew_d['retrace_pct']:.1f}%**  "
                        f"(typical Elliott corrective waves retrace 38–62% of prior leg)"
                    )

                # ── Last 3 confirmed pivots (wave structure) ─────────────────
                if _ew_d["last_3_pivots"]:
                    st.markdown("#### Last Confirmed Pivot Structure")
                    _piv_rows = []
                    _wave_labels_long  = ["Wave A (Low)", "Wave B (High)", "Wave C (Low)"]
                    _wave_labels_short = ["Wave A (High)", "Wave B (Low)", "Wave C (High)"]
                    _lp3 = _ew_d["last_3_pivots"]
                    for _pi, (_pidx, _ppx, _pdir) in enumerate(_lp3):
                        _plabel = f"Pivot {_pi+1} of {len(_lp3)}"
                        try:
                            _pdt = to_ist(lv_df.index[_pidx])
                        except: _pdt = str(_pidx)
                        _piv_rows.append({
                            "Pivot":     _plabel,
                            "Type":      "Swing HIGH ↑" if _pdir==1 else "Swing LOW ↓",
                            "Price":     f"{_ppx:.2f}",
                            "Bar Index": _pidx,
                            "Time (IST)":_pdt,
                        })
                    _piv_df = pd.DataFrame(_piv_rows)
                    st.dataframe(_piv_df, use_container_width=True, hide_index=True)

                    # Describe the pattern formed
                    if len(_lp3) == 3:
                        _p0d = _lp3[0][2]; _p1d = _lp3[1][2]; _p2d = _lp3[2][2]
                        _p0p = _lp3[0][1]; _p1p = _lp3[1][1]; _p2p = _lp3[2][1]
                        if _p0d==-1 and _p1d==1 and _p2d==-1:
                            _leg1 = abs(_p1p-_p0p); _leg2 = abs(_p2p-_p1p)
                            _ret  = _leg2/_leg1*100 if _leg1>0 else 0
                            if _p2p > _p0p:
                                st.success(f"Pattern: LOW → HIGH → LOW (higher low = bullish). "
                                           f"Leg1={_leg1:.2f} pts, Retrace={_ret:.1f}%. "
                                           f"**BUY signal already fired at pivot C ({_p2p:.2f}). "
                                           f"Watch for next corrective structure to re-enter.**")
                            else:
                                st.warning(f"Pattern: LOW → HIGH → LOWER LOW (lower low = bearish continuation). "
                                           f"Leg1={_leg1:.2f}, Retrace={_ret:.1f}%. No LONG signal — structure bearish.")
                        elif _p0d==1 and _p1d==-1 and _p2d==1:
                            _leg1 = abs(_p1p-_p0p); _leg2 = abs(_p2p-_p1p)
                            _ret  = _leg2/_leg1*100 if _leg1>0 else 0
                            if _p2p < _p0p:
                                st.error(f"Pattern: HIGH → LOW → HIGH (lower high = bearish). "
                                         f"Leg1={_leg1:.2f} pts, Retrace={_ret:.1f}%. "
                                         f"**SELL signal already fired at pivot C ({_p2p:.2f}). "
                                         f"Watch for next corrective structure to re-enter.**")
                            else:
                                st.warning(f"Pattern: HIGH → LOW → HIGHER HIGH (higher high = bullish continuation). "
                                           f"Leg1={_leg1:.2f}, Retrace={_ret:.1f}%. No SHORT signal — structure bullish.")

                # ── What is needed to fire next signal ───────────────────────
                st.markdown("---")
                _long_tab, _short_tab = st.tabs(["🟢 Steps for LONG signal", "🔴 Steps for SHORT signal"])

                with _long_tab:
                    st.caption(
                        "**How to read this table:** These are strictly sequential steps. "
                        "Step 2 only activates after Step 1 finishes. Step 3 only activates after Step 2 finishes. "
                        "A step showing ⏳ Locked means it is NOT checkable yet — you must complete the prior step first. "
                        "The signal fires the moment ALL steps complete on a single closed bar. "
                        "If Step 1 shows ✅ (rally confirmed) but Step 2 shows ⏳ (watching for pullback), "
                        "you simply need to wait for price to pull back the required % — Step 3 will be verified automatically then."
                    )
                    if _ew_d["long_conditions"]:
                        _lc_df = pd.DataFrame(_ew_d["long_conditions"])
                        def _met_color(v):
                            if "✅" in str(v): return "color:#2e7d32;font-weight:bold"
                            if "❌" in str(v): return "color:#c62828;font-weight:bold"
                            if "⏳" in str(v): return "color:#e65100"
                            return ""
                        st.dataframe(_lc_df.style.map(_met_color, subset=["Met"]),
                                     use_container_width=True, hide_index=True)
                        _long_done  = sum(1 for c in _ew_d["long_conditions"] if "✅" in c["Met"])
                        _long_total = len(_ew_d["long_conditions"])
                        _cur_step   = next((i+1 for i,c in enumerate(_ew_d["long_conditions"])
                                            if "✅" not in c["Met"] and "Locked" not in c["Met"]), _long_total)
                        if _long_done == _long_total:
                            st.success(f"🚀 ALL {_long_total} steps complete — LONG signal should have fired!")
                        else:
                            st.info(f"**Currently on Step {_cur_step} of {_long_total}.** "
                                    f"Complete Step {_cur_step} to advance. Signal fires when all {_long_total} steps are done.")
                    else:
                        st.info("Need at least 2 confirmed pivots to show LONG steps.")

                with _short_tab:
                    st.caption(
                        "Sequential steps — each must complete before the next is checked."
                    )
                    if _ew_d["short_conditions"]:
                        _sc_df = pd.DataFrame(_ew_d["short_conditions"])
                        st.dataframe(_sc_df.style.map(_met_color, subset=["Met"]),
                                     use_container_width=True, hide_index=True)
                        _short_done  = sum(1 for c in _ew_d["short_conditions"] if "✅" in c["Met"])
                        _short_total = len(_ew_d["short_conditions"])
                        if _short_done == _short_total:
                            st.error(f"🔴 ALL {_short_total} steps complete — SHORT signal should have fired!")
                        else:
                            _cur_step_s = next((i+1 for i,c in enumerate(_ew_d["short_conditions"])
                                                if "✅" not in c["Met"] and "Locked" not in c["Met"]), _short_total)
                            st.info(f"**Currently on Step {_cur_step_s} of {_short_total}.** "
                                    f"Signal fires when all {_short_total} steps complete.")
                    else:
                        st.info("Need at least 2 confirmed pivots to show SHORT steps.")

                # ── EW Wave Structure Plot ────────────────────────────────────
                st.markdown("---")
                st.markdown("#### 📈 Current Wave Structure (zigzag chart)")
                try:
                    _ew_pivots_plot = _ew_d.get("last_3_pivots", [])
                    # Get all pivots from full diagnostics
                    _all_pivots, _ = _ew_build_pivots(lv_df, min_wave_pct=float(live_params.get("min_wave_pct",0.5)))
                    _last_n_pivots = _all_pivots[-8:] if len(_all_pivots)>=8 else _all_pivots

                    if _last_n_pivots:
                        # Build a simple zigzag line plot over last N bars
                        _plot_bars = min(100, lv_n)
                        _plot_df   = lv_df.iloc[-_plot_bars:].copy()
                        _ew_fig    = go.Figure()

                        # Candlestick base
                        _ew_fig.add_trace(go.Candlestick(
                            x=_plot_df.index, open=_plot_df["Open"], high=_plot_df["High"],
                            low=_plot_df["Low"],  close=_plot_df["Close"],
                            name="Price", increasing_line_color="#26a69a",
                            decreasing_line_color="#ef5350", showlegend=False
                        ))

                        # Zigzag pivot line
                        _visible_pivots = [(idx, px, d) for idx, px, d in _last_n_pivots
                                           if 0 <= idx < lv_n and idx >= lv_n - _plot_bars]
                        if _visible_pivots:
                            _px_list  = [lv_df.index[idx] for idx, px, d in _visible_pivots]
                            _py_list  = [px for idx, px, d in _visible_pivots]
                            _ew_fig.add_trace(go.Scatter(
                                x=_px_list, y=_py_list, mode="lines+markers+text",
                                line=dict(color="#FF9800", width=2, dash="dash"),
                                marker=dict(size=10, color=["#2196F3" if d==-1 else "#F44336"
                                                             for idx, px, d in _visible_pivots]),
                                text=[f"{'L' if d==-1 else 'H'}{i+1}" for i,(idx,px,d) in enumerate(_visible_pivots)],
                                textposition="top center",
                                name="EW Zigzag"
                            ))

                        # Mark signal bars
                        _sig_bars = [(i, lv_df.index[i]) for i in range(max(0,lv_n-_plot_bars), lv_n)
                                     if int(lv_sigs.iloc[i]) != 0]
                        for _si, _sdt in _sig_bars:
                            _sv = int(lv_sigs.iloc[_si])
                            _ew_fig.add_trace(go.Scatter(
                                x=[_sdt], y=[float(lv_df["Low"].iloc[_si]) if _sv==1
                                             else float(lv_df["High"].iloc[_si])],
                                mode="markers+text",
                                marker=dict(size=14, color="#00E676" if _sv==1 else "#FF5252",
                                            symbol="triangle-up" if _sv==1 else "triangle-down"),
                                text=["BUY" if _sv==1 else "SELL"],
                                textposition="bottom center" if _sv==1 else "top center",
                                showlegend=False
                            ))

                        _ew_fig.update_layout(
                            height=350, title=f"EW Wave Structure — last {_plot_bars} bars",
                            xaxis_rangeslider_visible=False,
                            margin=dict(l=10,r=10,t=40,b=10),
                            legend=dict(orientation="h",y=-0.15),
                        )
                        st.plotly_chart(_ew_fig, use_container_width=True,
                                        key=f"ew_wave_plot_{tick}")
                        st.caption(
                            "🔵 Blue markers = swing LOWS (support pivots)  |  "
                            "🔴 Red markers = swing HIGHS (resistance pivots)  |  "
                            "Orange dashed line = zigzag wave structure  |  "
                            "▲ BUY / ▼ SELL = signal bars"
                        )
                    else:
                        st.info("Not enough pivot data for wave plot yet. More data needed.")
                except Exception as _ewplot_err:
                    st.caption(f"Wave plot: {_ewplot_err}")

                # Mark EMA_20 as handled (it's the only indicator EW returns)
                _handled_keys.update({"EMA_20"})

            # ════════════════════════════════════════════════════════════════
            # GENERIC FALLBACK: show any remaining indicator not yet handled
            # ════════════════════════════════════════════════════════════════
            for _ik, _iser in lv_indics.items():
                if _ik in _handled_keys:
                    continue
                if not isinstance(_iser, pd.Series):
                    continue
                _gv, _gpv = _last2(_iser)
                if _gv is None:
                    continue
                _delta_g = _gv - _gpv if _gpv is not None else 0
                _sp_items.append({
                    "Indicator": _ik,
                    "Value":     f"{_gv:.4f}",
                    "vs":        f"vs LTP={cl:.4f}",
                    "Gap":       f"delta={_delta_g:+.4f}/bar",
                    "Direction": _trend_arrow(_gv, _gpv),
                    "Momentum":  "—",
                    "Est. Bars": "—",
                    "Signal":    "Above price" if _gv > cl else ("Below price" if _gv < cl else "At price"),
                })

            # ── Strategy parameter summary (always shown) ─────────────────────
            st.markdown("**Strategy Parameters (active)**")
            _param_show = {k: v for k, v in live_params.items()
                           if k not in ("atr_mult_sl","atr_mult_tgt","rr_ratio","swing_lookback",
                                        "thresh_dir","thresh_action","kshift_n","kshift_k",
                                        "tgt_pts_for_sl","use_cond2","_cond_list")}
            if _param_show:
                _pc = st.columns(min(len(_param_show), 6))
                for _ci, (k, v) in enumerate(list(_param_show.items())[:12]):
                    _pc[_ci % len(_pc)].metric(k, v)

            # ── Last signal status ────────────────────────────────────────────
            st.markdown("**Signal State**")
            _sc1, _sc2, _sc3 = st.columns(3)
            _sc1.metric("Last Completed Bar", to_ist(lv_df.index[-1]))
            _sc2.metric("Current Signal", "🟢 BUY" if last_sig==1 else ("🔴 SELL" if last_sig==-1 else "⚪ FLAT"))
            # Count bars since last signal
            _bars_since_sig = 0
            for _bi in range(len(lv_sigs)-2, max(len(lv_sigs)-20,0)-1, -1):
                if lv_sigs.iloc[_bi] != 0: break
                _bars_since_sig += 1
            _sc3.metric("Bars since last signal", _bars_since_sig if _bars_since_sig < 19 else "20+")

            st.markdown("---")

            # ── Indicator progress table ──────────────────────────────────────
            if _sp_items:
                st.markdown("**Indicator Progress**")
                _sp_df = pd.DataFrame(_sp_items)
                # Color the Signal column
                def _sig_cell(v):
                    if "🟢" in str(v): return "color:#2e7d32;font-weight:bold"
                    if "🔴" in str(v): return "color:#c62828;font-weight:bold"
                    if "⚠️" in str(v): return "color:#e65100;font-weight:bold"
                    if "⚡" in str(v): return "color:#0277bd;font-weight:bold"
                    return ""
                st.dataframe(
                    _sp_df.style.map(_sig_cell, subset=["Signal"]),
                    use_container_width=True, hide_index=True,
                )
            else:
                # Even if no indicators, show raw price data and strategy name
                st.info(
                    f"Strategy **{strategy}** does not produce chartable indicator series "
                    f"(e.g. pattern-only strategies like Elliott Wave compute patterns internally). "
                    f"The signal state above shows whether a signal was generated. "
                    f"Current LTP: **{cl:.2f}** | Last bar: **{to_ist(lv_df.index[-1])}**"
                )

    def _hist_render():
        _all_t = st.session_state.get("live_trades", [])
        # Show trades for current ticker; allow toggle to see all
        _show_all_tickers = st.checkbox("Show trades for ALL tickers", value=False, key="hist_show_all")
        t_ = _all_t if _show_all_tickers else [t for t in _all_t if t.get("Ticker","") == sym]
        _other = len(_all_t) - len(t_) if not _show_all_tickers else 0

        if _other:
            st.caption(f"Showing {len(t_)} trades for **{sym}**. "
                       f"{_other} trades for other tickers hidden — enable 'Show all tickers' above.")

        if t_:
            tot  = len(t_)
            wins = sum(1 for x in t_ if x["PnL"] > 0)
            pnl_ = sum(x["PnL"] for x in t_)
            pw   = sum(x["PnL"] for x in t_ if x["PnL"] > 0)
            pl   = abs(sum(x["PnL"] for x in t_ if x["PnL"] < 0))
            hc   = st.columns(5)
            hc[0].metric("Trades",     tot)
            hc[1].metric("Accuracy",   f"{wins/tot*100:.1f}%")
            hc[2].metric("Total PnL",  f"{pnl_:.2f}", delta=f"{pnl_:+.2f}",
                         delta_color="normal" if pnl_ >= 0 else "inverse")
            hc[3].metric("Pts Won",    f"{pw:.2f}")
            hc[4].metric("Pts Lost",   f"{pl:.2f}")
            hdf = pd.DataFrame(t_)
            def _sc(v):
                if isinstance(v,(int,float)): return "color:#00E676" if v>0 else ("color:#FF5252" if v<0 else "")
                return ""
            # Full datetime columns — never truncated
            st.dataframe(
                hdf.style.map(_sc, subset=["PnL"]),
                use_container_width=True,
                column_config={
                    "Entry Time": st.column_config.TextColumn("Entry Time (IST)", width="large"),
                    "Exit Time":  st.column_config.TextColumn("Exit Time (IST)",  width="large"),
                    "Ticker":     st.column_config.TextColumn("Ticker", width="small"),
                    "Strategy":   st.column_config.TextColumn("Strategy", width="medium"),
                }
            )
            # Use a dynamic key based on trade count so it never conflicts across reruns
            eq_ = plot_equity(t_, "Live Equity")
            if eq_:
                st.plotly_chart(eq_, use_container_width=True,
                                key=f"lv_hist_equity_{tot}_{int(pnl_*100)}")
        else:
            st.info("No completed live trades yet. Trades appear here as soon as they close.")

    with sub_mon:
        if _HAS_FRAGMENT and st.session_state.live_active:
            @st.fragment(run_every=2)
            def _lv_frag(): _live_render()
            _lv_frag()
        else:
            _live_render()
            if st.session_state.live_active: time.sleep(1.5); st.rerun()

    with sub_hist:
        # Always render trade history regardless of fragment state
        # so trades show immediately when closed without needing to stop live trading.
        # The fragment auto-refreshes every 3s; if fragment unavailable, render directly.
        if _HAS_FRAGMENT and st.session_state.live_active:
            @st.fragment(run_every=3)
            def _hist_frag(): _hist_render()
            _hist_frag()
        else:
            _hist_render()

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
        opt_st=oc2.selectbox("Strategy", STRATEGIES, key="opt_strat")
        opt_acc=oc3.slider("Desired Accuracy (%)",40,99,60,key="opt_acc")
        opt_pts=oc3.number_input("Min Total Pts Won",0.,1e6,0.,step=10.,key="opt_pts")
        opt_mt=int(oc3.number_input("Min Trades",1,50,3,step=1,key="opt_mt"))
        oc4,oc5=st.columns(2)
        opt_sl=oc4.selectbox("SL Type",SL_TYPES,key="opt_sl")
        opt_slp=oc4.number_input("SL Pts",0.01,1e6,10.,step=0.5,key="opt_slp")
        opt_tgt=oc5.selectbox("Target Type",TARGET_TYPES,key="opt_tgt")
        opt_tgtp=oc5.number_input("Target Pts",0.01,1e6,20.,step=0.5,key="opt_tgtp")
        # Time window filter for optimization
        opt_tw=st.checkbox("Time Window Filter (IST) — only count signals in trading hours",
                            value=False,key="opt_tw")
        if opt_tw:
            _otc1,_otc2=st.columns(2)
            opt_tw_from=_otc1.time_input("From (IST)",value=datetime.strptime("09:15","%H:%M").time(),key="opt_tw_from")
            opt_tw_to  =_otc2.time_input("To (IST)",  value=datetime.strptime("15:00","%H:%M").time(),key="opt_tw_to")
        else:
            opt_tw_from=datetime.strptime("09:15","%H:%M").time()
            opt_tw_to  =datetime.strptime("15:00","%H:%M").time()

    if st.button("🔬 Run Optimization",type="primary",key="btn_opt"):
        with st.spinner("Fetching data…"): df_opt=fetch_data(opt_sym,opt_pd,opt_iv)
        if df_opt is None or df_opt.empty:
            st.error("No data returned.")
        else:
            g=PARAM_GRIDS.get(opt_st,{}); n_combos=1
            for v in g.values(): n_combos*=len(v)
            st.info(f"Data: **{len(df_opt)} bars** · Grid: **{n_combos}** combos · Min trades: {opt_mt}"
                    + (f" · Time window: {opt_tw_from.strftime('%H:%M')}–{opt_tw_to.strftime('%H:%M')} IST" if opt_tw else ""))
            prog=st.progress(0)
            _otw_from_str = opt_tw_from.strftime("%H:%M") if opt_tw else None
            _otw_to_str   = opt_tw_to.strftime("%H:%M")   if opt_tw else None
            with st.spinner(f"Optimising {opt_st}…"):
                opt_res=optimize(df_opt,opt_st,opt_sl,opt_slp,opt_tgt,opt_tgtp,
                    opt_acc,float(opt_pts),opt_mt,progress_cb=prog.progress,
                    tw_from_str=_otw_from_str, tw_to_str=_otw_to_str)
            prog.empty()
            st.session_state.opt_results = opt_res
            st.session_state.opt_df      = df_opt
            st.session_state.opt_res_meta= {
                "strategy":opt_st,"instrument":opt_t,
                "custom_sym": opt_sym if opt_t=="Custom" else "",
                "interval":opt_iv,"period":opt_pd,
                "sl":opt_sl,"slp":opt_slp,"tgt":opt_tgt,"tgtp":opt_tgtp,
                "acc":opt_acc,"pts":opt_pts,
                "tw_from":_otw_from_str,"tw_to":_otw_to_str,
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
        _opt_csym = _meta.get("custom_sym", "")
        # Resolve actual Yahoo symbol
        opt_sym_resolved = _opt_csym if (_opt_t=="Custom" and _opt_csym) else (TICKER_MAP.get(_opt_t, _opt_t))

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

            st.markdown("#### 📊 All Results (sorted by accuracy)")
            edited=st.data_editor(
                res_df,
                column_config={c: st.column_config.Column(disabled=True) for c in res_df.columns},
                hide_index=True, use_container_width=True, height=450,
                key="opt_editor",
            )

            # ── Row selection ─────────────────────────────────────────────────
            st.markdown("**Select a row to apply:**")
            sel_idx = st.selectbox(
                "Choose result row # (0 = best)",
                options=list(range(len(opt_res))),
                format_func=lambda i: (
                    f"Row {i} | Acc={opt_res[i].get('Accuracy (%)','?')}% | "
                    f"PnL={opt_res[i].get('Total PnL','?')} | "
                    f"Trades={opt_res[i].get('Total Trades','?')} | "
                    f"Params={dict(list(opt_res[i]['params'].items())[:3])} "
                    f"{'✅' if opt_res[i].get('Meets_Accuracy') else ''}"
                ),
                key="opt_sel_idx",
            )

            sel_result = opt_res[sel_idx]
            # ── Selected row detail (original format restored) ────────────────
            with st.expander(f"📌 Selected Row {sel_idx} — details", expanded=True):
                _dc = st.columns(5)
                _dc[0].metric("Accuracy",      f"{sel_result.get('Accuracy (%)','?')}%")
                _dc[1].metric("Total PnL",     sel_result.get("Total PnL","?"))
                _dc[2].metric("Total Trades",  sel_result.get("Total Trades","?"))
                _dc[3].metric("Wins",          sel_result.get("Wins","?"))
                _dc[4].metric("Meets Targets", "✅ Yes" if sel_result.get("Meets_Accuracy") else "— No")
                # Parameters table
                _p_disp = {k:v for k,v in sel_result["params"].items()
                           if k not in("atr_mult_sl","atr_mult_tgt","rr_ratio","swing_lookback")}
                if _p_disp:
                    _pc_cols = st.columns(min(len(_p_disp),6))
                    for _pci,(pk,pv) in enumerate(list(_p_disp.items())[:12]):
                        _pc_cols[_pci%len(_pc_cols)].metric(pk, pv)
                # Full stats row
                _stats_keys = ["Total Pts Won","Total Pts Lost","Avg Win","Avg Loss",
                               "Max Win","Max Loss","Profit Factor"]
                _sc2 = st.columns(len([k for k in _stats_keys if k in sel_result]))
                for _sci,_sk in enumerate([k for k in _stats_keys if k in sel_result]):
                    _sc2[_sci].metric(_sk, sel_result[_sk])

            # ── Apply button (just adds button, original display unchanged) ───
            if st.button("⚡ Apply Selected Row to Config (Sidebar + Backtest + Live)",
                         type="primary", key="btn_apply_opt"):
                param_keys = list(PARAM_GRIDS.get(_opt_st, {}).keys())
                # Store FULL params (defaults + grid override) so backtest matches exactly
                _full_p = {**_STRATEGY_DEFAULTS.get(_opt_st, {}), **_BP}
                applied_params = {**_full_p,
                                  **{pk: sel_result["params"][pk]
                                     for pk in param_keys if pk in sel_result["params"]}}
                # Store custom symbol if needed
                _csym_val = ""
                if _opt_t == "Custom":
                    _csym_val = opt_sym  # the actual Yahoo symbol string
                elif _opt_t not in TICKER_MAP:
                    _csym_val = _opt_t

                st.session_state.opt_applied = {
                    "strategy":   _opt_st,
                    "instrument": _opt_t,
                    "custom_sym": _csym_val,   # e.g. "KAYNES.NS"
                    "interval":   _opt_iv,
                    "period":     _opt_pd,
                    "sl_type":    _opt_sl,
                    "sl_pts":     _opt_slp,
                    "tgt_type":   _opt_tgt,
                    "tgt_pts":    _opt_tgtp,
                    "params":     applied_params,
                    "accuracy":   sel_result.get("Accuracy (%)", "?"),
                    "pnl":        sel_result.get("Total PnL", "?"),
                }
                st.session_state["opt_apply_msg"] = (
                    f"✅ Applied Row {sel_idx} → **{_opt_st}** | "
                    f"Ticker={_opt_t}{(' ('+_csym_val+')') if _csym_val else ''} | "
                    f"Interval={_opt_iv} | Period={_opt_pd} | "
                    f"Accuracy={sel_result.get('Accuracy (%)','?')}% | "
                    f"PnL={sel_result.get('Total PnL','?')}"
                )
                # Reset hash so pre-populate fires on next rerun
                st.session_state["_oa_hash_prev"] = ""
                st.rerun()

            # Show green confirmation if recently applied
            if st.session_state.get("opt_apply_msg"):
                st.success(st.session_state["opt_apply_msg"])
                st.info("👈 Sidebar and Backtesting tab have been updated. Switch to 📊 Backtesting and click **Run Backtest**.")

            # Best result preview
            if df_opt_ss is not None:
                st.markdown("---"); st.markdown("### 🥇 Best Result Preview (Row 0)")
                best   = opt_res[0]
                # Use full strategy defaults merged with best params for exact match
                best_p = {**_STRATEGY_DEFAULTS.get(_opt_st,{}), **_BP, **best["params"]}
                _meta_tw_from = _meta.get("tw_from")
                _meta_tw_to   = _meta.get("tw_to")
                bt2,ind2,_ = run_backtest(df_opt_ss, _opt_st, best_p,
                                           _opt_sl, _opt_slp, _opt_tgt, _opt_tgtp,
                                           _meta_tw_from, _meta_tw_to)
                bp2=calc_perf(bt2)
                if bp2:
                    bmc=st.columns(len(bp2))
                    for col,(k,v) in zip(bmc,bp2.items()): col.metric(k,v)
                acc_lbl=f"{best.get('Accuracy (%)','?'):.1f}%" if isinstance(best.get('Accuracy (%)'),float) else str(best.get('Accuracy (%)','?'))
                st.plotly_chart(plot_ohlc(df_opt_ss,bt2,ind2,
                    title=f"Best Params: {dict(list(best['params'].items())[:4])} | Acc={acc_lbl}"),
                    use_container_width=True,key="opt_best_ohlc")
                eq_b=plot_equity(bt2)
                if eq_b: st.plotly_chart(eq_b,use_container_width=True,key="opt_best_equity")
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

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — NEAR TO ENTRY
# Scans Nifty 50 stocks (or custom list) for upcoming signals on chosen strategy.
# For each ticker: fetches OHLCV, computes strategy signals, checks last N bars
# to see if a signal just fired (already triggered) or is "approaching" based on
# indicator proximity to crossover / threshold.
# ═══════════════════════════════════════════════════════════════════════════════
with tab_nte:
    st.subheader("🎯 Near to Entry — Signal Scanner")

    # ── Helper: apply a ticker+strategy+timeframe to sidebar via pending state ──
    # We CANNOT write to g_ticker/g_interval/g_period/g_strategy directly after
    # their widgets are instantiated — Streamlit raises StreamlitAPIException.
    # Solution: store in a "pending" dict in session_state; the pre-populate block
    # (which runs BEFORE widgets are rendered on next rerun) applies it.
    def _nte_apply_to_live(ticker_sym, ticker_label, sig_direction, iv, pd_val, strat=None):
        _apply_strat = strat if strat and strat in STRATEGIES else (
            "Simple Buy" if sig_direction=="BUY" else "Simple Sell"
        )
        # Resolve instrument label
        if ticker_sym in TICKER_MAP.values():
            _tk_label = [k for k,v in TICKER_MAP.items() if v==ticker_sym][0]
            _custom   = ""
        else:
            _tk_label = "Custom"
            _custom   = ticker_sym

        # Store as opt_applied so the existing pre-populate block handles it
        st.session_state["opt_applied"] = {
            "strategy":   _apply_strat,
            "instrument": _tk_label,
            "custom_sym": _custom,
            "interval":   iv if iv in TIMEFRAMES else "1h",
            "period":     pd_val if pd_val in PERIODS else "1mo",
            "sl_type":    SL_TYPES[0],
            "sl_pts":     10.0,
            "tgt_type":   TARGET_TYPES[0],
            "tgt_pts":    20.0,
            "params":     {**_STRATEGY_DEFAULTS.get(_apply_strat,{}), **_BP},
            "accuracy":   "—",
            "pnl":        "—",
        }
        st.session_state["_oa_hash_prev"] = ""   # force pre-populate to fire
        st.session_state["nte_applied_msg"] = (
            f"✅ Applied: **{ticker_sym}** | Signal: **{sig_direction}** | "
            f"Strategy: **{_apply_strat}** | TF: **{iv}** | Period: **{pd_val}** → "
            "Switch to ⚡ Live Trading or 📊 Backtesting"
        )

    # Show confirmation if recently applied
    if st.session_state.get("nte_applied_msg"):
        st.success(st.session_state["nte_applied_msg"])

    # ── Sub-tabs: Scanner | Quick Search ──────────────────────────────────────
    nte_scan_tab, nte_quick_tab, nte_ew_tab, nte_pos_tab, nte_fresh_tab, nte_dive_tab = st.tabs([
        "🔍 Strategy Scanner", "⚡ Quick Signal Search",
        "🌊 Elliott Wave Monitor", "📍 Position Tracker",
        "🎯 Fresh Signal Finder", "🔎 Signal Deep Dive"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 1: Strategy Scanner (existing scanner with Apply buttons)
    # ══════════════════════════════════════════════════════════════════════════
    with nte_scan_tab:
        st.markdown(
            "Scans selected tickers using the chosen strategy and flags stocks that have "
            "**just fired a signal** or are **approaching a signal**. "
            "1.5 s rate-limit delay between each ticker fetch."
        )

        with st.expander("🔧 Scanner Inputs", expanded=True):
            _nc1, _nc2, _nc3 = st.columns(3)

            _nte_universe = _nc1.selectbox(
                "Universe",
                ["All Nifty 50 Stocks", "Custom Tickers"] + list(TICKER_MAP.keys()),
                key="nte_universe",
            )
            if _nte_universe == "Custom Tickers":
                _nte_custom_raw = _nc1.text_area(
                    "Enter Yahoo symbols (one per line)",
                    "RELIANCE.NS\nINFY.NS\nHDFCBANK.NS",
                    key="nte_custom_tickers", height=100,
                )
                _nte_symbols = [s.strip() for s in _nte_custom_raw.splitlines() if s.strip()]
            elif _nte_universe == "All Nifty 50 Stocks":
                _nte_symbols = NIFTY50_SYMBOLS
                _nc1.caption(f"Will scan {len(_nte_symbols)} Nifty 50 stocks.")
            else:
                _nte_symbols = [TICKER_MAP[_nte_universe]]
                _nc1.caption(f"Single ticker: {_nte_symbols[0]}")

            _nte_iv  = _nc2.selectbox("Timeframe", TIMEFRAMES, index=4, key="nte_iv")
            _nte_pd  = _nc2.selectbox("Period",    PERIODS,    index=4, key="nte_pd")
            _nte_st  = _nc2.selectbox("Strategy",  STRATEGIES, key="nte_strategy")
            _nte_sl_type  = _nc3.selectbox("SL Type",     SL_TYPES,     key="nte_sl")
            _nte_sl_pts   = _nc3.number_input("SL Pts",  0.01,1e6,10.,step=0.5,key="nte_slp")
            _nte_tgt_type = _nc3.selectbox("Target Type",TARGET_TYPES,  key="nte_tgt")
            _nte_tgt_pts  = _nc3.number_input("Tgt Pts", 0.01,1e6,20.,step=0.5,key="nte_tgtp")

            _nte_lookback = st.slider(
                "Near-signal lookback (bars) — how recently a signal fired counts as 'fresh'",
                1, 10, 3, key="nte_lookback",
            )
            _nte_prox_pct = st.slider(
                "Proximity threshold % — indicators within this % of crossover count as 'approaching'",
                0.1, 5.0, 1.0, step=0.1, key="nte_prox",
            )
            _nte_run_opt = st.checkbox(
                "Also run optimization on each ticker (finds best params, slower)",
                value=False, key="nte_run_opt",
            )
            if _nte_run_opt:
                _nte_min_acc = st.slider("Min accuracy for best params (%)", 40, 99, 55, key="nte_minacc")
            else:
                _nte_min_acc = 55

        if st.button("🔍 Scan Now", type="primary", key="btn_nte_scan"):
            if not _nte_symbols:
                st.error("No tickers selected.")
            else:
                _nte_params = {**sb_params}
                _nte_results  = []
                _nte_approach = []
                _nte_errors   = []
                _prog = st.progress(0)
                _status = st.empty()
                _total  = len(_nte_symbols)

                for _ti, _sym in enumerate(_nte_symbols):
                    _status.caption(f"Scanning {_sym} ({_ti+1}/{_total})…")
                    _prog.progress((_ti+1)/_total)
                    try:
                        time.sleep(1.5)
                        _raw = yf.download(_sym, period=_nte_pd,
                                           interval=YF_IV.get(_nte_iv,_nte_iv),
                                           progress=False, auto_adjust=True)
                        if _raw is None or _raw.empty:
                            _nte_errors.append({"Ticker":_sym,"Issue":"No data"}); continue
                        _df = _flatten(_raw)
                        if _nte_iv=="4h": _df=_r4h(_df)
                        if len(_df)<20:
                            _nte_errors.append({"Ticker":_sym,"Issue":"Too few bars"}); continue

                        _fn = STRATEGY_FN.get(_nte_st, sig_custom)
                        try: _sigs, _indics = _fn(_df, **_nte_params)
                        except Exception as _e:
                            _nte_errors.append({"Ticker":_sym,"Issue":f"Strategy error: {_e}"}); continue

                        _last_close = float(_df["Close"].iloc[-1])
                        _last_bar   = to_ist(_df.index[-1])
                        _n          = len(_df)
                        _recent_sigs= _sigs.iloc[max(0,_n-_nte_lookback-1):_n-1]
                        _fired_long = int((_recent_sigs==1).sum())
                        _fired_short= int((_recent_sigs==-1).sum())
                        _last_sig_val=int(_sigs.iloc[-2]) if _n>1 else 0
                        _bars_since = None
                        for _bi in range(_n-2, max(_n-20,0)-1, -1):
                            if _sigs.iloc[_bi]!=0: _bars_since=_n-2-_bi; break

                        _best_acc=None; _best_p=None
                        if _nte_run_opt and _nte_st in PARAM_GRIDS:
                            try:
                                _or = optimize(_df,_nte_st,_nte_sl_type,_nte_sl_pts,
                                               _nte_tgt_type,_nte_tgt_pts,_nte_min_acc,0,3)
                                if _or:
                                    _best_acc=_or[0].get("Accuracy (%)","?")
                                    _best_p={k:v for k,v in _or[0]["params"].items()
                                             if k not in("atr_mult_sl","atr_mult_tgt","rr_ratio","swing_lookback")}
                            except: pass

                        _proximity_note=""; _is_approaching=False
                        if "EMA_fast" in _indics and "EMA_slow" in _indics:
                            _fe_now=float(_indics["EMA_fast"].iloc[-1]); _se_now=float(_indics["EMA_slow"].iloc[-1])
                            _fe_prev=float(_indics["EMA_fast"].iloc[-2]) if _n>1 else _fe_now
                            _se_prev=float(_indics["EMA_slow"].iloc[-2]) if _n>1 else _se_now
                            _gap_pct=abs(_fe_now-_se_now)/max(abs(_se_now),0.01)*100
                            _narrowing=(abs(_fe_now-_se_now)<abs(_fe_prev-_se_prev))
                            if _gap_pct<=_nte_prox_pct:
                                _is_approaching=True
                                _dir_hint="LONG" if _fe_now>_se_now else "SHORT"
                                _proximity_note=f"EMA gap={_gap_pct:.2f}% ({'narrowing' if _narrowing else 'widening'}) → possible {_dir_hint}"
                        elif "RSI" in _indics:
                            _r_now=float(_indics["RSI"].iloc[-1]) if not _indics["RSI"].empty else 50
                            _ob=_nte_params.get("ob",70); _os=_nte_params.get("os_",30)
                            if abs(_r_now-_os)/_os*100<=_nte_prox_pct*5:
                                _is_approaching=True; _proximity_note=f"RSI={_r_now:.1f} near OS({_os}) → BUY soon"
                            elif abs(_r_now-_ob)/_ob*100<=_nte_prox_pct*5:
                                _is_approaching=True; _proximity_note=f"RSI={_r_now:.1f} near OB({_ob}) → SELL soon"
                        elif "MACD" in _indics and "MACD_Signal" in _indics:
                            _m_now=float(_indics["MACD"].iloc[-1]); _ms_now=float(_indics["MACD_Signal"].iloc[-1])
                            _m_prev=float(_indics["MACD"].iloc[-2]) if _n>1 else _m_now
                            _ms_prev=float(_indics["MACD_Signal"].iloc[-2]) if _n>1 else _ms_now
                            _gap_now=_m_now-_ms_now; _gap_prev=_m_prev-_ms_prev
                            if abs(_gap_now)<abs(_gap_prev)*0.3:
                                _is_approaching=True; _proximity_note=f"MACD hist={_gap_now:.3f} converging"

                        _row = {
                            "Ticker":_sym,"Last Close":round(_last_close,2),"Last Bar":_last_bar,
                            "Last Signal":"🟢 BUY" if _last_sig_val==1 else ("🔴 SELL" if _last_sig_val==-1 else "⚪"),
                            "Fired Long":_fired_long,"Fired Short":_fired_short,
                            "Bars Since":_bars_since if _bars_since is not None else "—",
                            "Approaching":"✅" if _is_approaching else "—",
                            "Proximity Note":_proximity_note,
                            "Best Acc":f"{_best_acc:.1f}%" if isinstance(_best_acc,(int,float)) else "—",
                            "_sig_dir":"BUY" if _last_sig_val==1 else ("SELL" if _last_sig_val==-1 else
                                       ("BUY" if _fired_long>0 else "SELL")),
                        }
                        if _fired_long>0 or _fired_short>0 or _last_sig_val!=0:
                            _nte_results.append(_row)
                        elif _is_approaching:
                            _nte_approach.append(_row)
                    except Exception as _ex:
                        _nte_errors.append({"Ticker":_sym,"Issue":str(_ex)[:80]})

                _prog.empty(); _status.empty()
                st.session_state["nte_scan_results"]  = _nte_results
                st.session_state["nte_scan_approach"] = _nte_approach
                st.session_state["nte_scan_errors"]   = _nte_errors
                st.session_state["nte_scan_meta"]     = {
                    "iv":_nte_iv,"pd":_nte_pd,"st":_nte_st,"total":len(_nte_symbols)
                }

        # ── Display scan results (from session state so Apply buttons persist) ──
        _sr  = st.session_state.get("nte_scan_results")
        _sa  = st.session_state.get("nte_scan_approach")
        _se  = st.session_state.get("nte_scan_errors")
        _sm  = st.session_state.get("nte_scan_meta",{})

        if _sr is not None:
            st.markdown(f"**Scan complete.** {_sm.get('total','?')} tickers scanned.")
            _r1,_r2,_r3=st.columns(3)
            _r1.metric("Fired Signals",len(_sr)); _r2.metric("Approaching",len(_sa or []))
            _r3.metric("Errors",len(_se or []))

            # ── Fired signals — each row has an Apply button ──────────────────
            st.markdown("### 🔴🟢 Fired Signals")
            if _sr:
                for _ri, _row in enumerate(_sr):
                    _sig_dir = _row.get("_sig_dir","BUY")
                    _rc1,_rc2,_rc3,_rc4,_rc5,_rc6 = st.columns([2,1,1.2,1,1.5,2])
                    _rc1.markdown(f"**{_row['Ticker']}**")
                    _rc2.markdown(f"`{_row['Last Signal']}`")
                    _rc3.markdown(f"Close: **{_row['Last Close']}**")
                    _rc4.markdown(f"Bar: {_row['Fired Long']}L / {_row['Fired Short']}S")
                    _rc5.markdown(f"Since: {_row['Bars Since']} bars")
                    if _rc6.button(
                        f"{'🟢 Apply BUY' if _sig_dir=='BUY' else '🔴 Apply SELL'} → Live",
                        key=f"nte_apply_{_ri}_{_row['Ticker']}",
                        type="primary" if _sig_dir=="BUY" else "secondary",
                    ):
                        _nte_apply_to_live(
                            ticker_sym=_row["Ticker"],
                            ticker_label=_row["Ticker"],
                            sig_direction=_sig_dir,
                            iv=_sm.get("iv","1h"),
                            pd_val=_sm.get("pd","3mo"),
                            strat=_sm.get("st","Simple Buy") if _sig_dir=="BUY" else "Simple Sell",
                        )
                        st.rerun()

                # Mini-charts for top 5
                with st.expander("📈 Preview Charts (top 5)"):
                    for _pi, _row in enumerate(_sr[:5]):
                        _psym=_row["Ticker"]
                        try:
                            time.sleep(1.5)
                            _praw=yf.download(_psym,period=_sm.get("pd","3mo"),
                                              interval=YF_IV.get(_sm.get("iv","1h"),_sm.get("iv","1h")),
                                              progress=False,auto_adjust=True)
                            if not _praw.empty:
                                _pdf=_flatten(_praw)
                                if _sm.get("iv")=="4h": _pdf=_r4h(_pdf)
                                _fn2=STRATEGY_FN.get(_sm.get("st","EMA Crossover"),sig_custom)
                                _s2,_i2=_fn2(_pdf,**sb_params)
                                _t2,_,_=run_backtest(_pdf,_sm.get("st","EMA Crossover"),sb_params,
                                                      "Custom Points",10.,"Custom Points",20.)
                                st.plotly_chart(plot_ohlc(_pdf,_t2,_i2,
                                    title=f"{_psym} | {_sm.get('st','')} | {_sm.get('iv','')}"),
                                    use_container_width=True,key=f"nte_chart_{_psym}_{_pi}")
                        except Exception as _ce:
                            st.caption(f"Chart error {_psym}: {_ce}")
            else:
                st.info("No fired signals. Try larger lookback, different strategy, or longer period.")

            # ── Approaching signals ────────────────────────────────────────────
            st.markdown("### 🔔 Approaching Signals")
            if _sa:
                for _ai, _row in enumerate(_sa):
                    _ac1,_ac2,_ac3,_ac4,_ac5 = st.columns([2,1,2,2,2])
                    _ac1.markdown(f"**{_row['Ticker']}**")
                    _ac2.markdown(f"`{_row['Last Close']}`")
                    _ac3.markdown(f"{_row.get('Proximity Note','')}")
                    _ac4.markdown(f"Bars since sig: {_row.get('Bars Since','—')}")
                    _asig = "BUY" if _row.get("Fired Long",0)>0 else "SELL"
                    if _ac5.button(f"📋 Watchlist → Apply {_asig}",key=f"nte_appr_apply_{_ai}"):
                        _nte_apply_to_live(_row["Ticker"],_row["Ticker"],_asig,
                                           _sm.get("iv","1h"),_sm.get("pd","3mo"))
                        st.rerun()
            else:
                st.info("No stocks approaching a signal.")

            if _se:
                with st.expander(f"⚠️ {len(_se)} errors"):
                    st.dataframe(pd.DataFrame(_se),use_container_width=True)

            _long_c=sum(1 for r in _sr if "BUY"  in r.get("Last Signal",""))
            _short_c=sum(1 for r in _sr if "SELL" in r.get("Last Signal",""))
            st.info(f"**Market Bias:** {_long_c} BUY · {_short_c} SELL · "
                    f"{len(_sa or [])} approaching | Strategy: **{_sm.get('st','')}** | "
                    f"TF: **{_sm.get('iv','')}** | Universe: **{_nte_universe}**")
        else:
            st.info("Configure inputs above and click 🔍 Scan Now.")

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 2: Quick Signal Search — auto multi-TF scan for a specific ticker
    # ══════════════════════════════════════════════════════════════════════════
    with nte_quick_tab:
        st.markdown(
            "**Auto-scan a specific ticker across multiple timeframes and periods simultaneously.** "
            "Finds which combinations have a **Simple Buy / Simple Sell signal fired** on the "
            "most recent bar. Click any row's Apply button to trade it directly."
        )

        # Scan matrix: {interval: [periods]}
        _QS_MATRIX = {
            "1m":  ["1d","5d","7d"],
            "5m":  ["1d","5d","7d","1mo"],
            "15m": ["1d","5d","7d","1mo"],
            "1h":  ["7d","1mo","3mo","6mo"],
            "1d":  ["1mo","3mo","6mo","1y"],
        }

        _qc1,_qc2,_qc3 = st.columns(3)
        _qs_universe = _qc1.selectbox(
            "Ticker(s) to search",
            ["Custom Tickers"] + list(TICKER_MAP.keys()) + ["All Nifty 50 Stocks"],
            key="qs_universe",
        )
        if _qs_universe == "Custom Tickers":
            _qs_raw = _qc1.text_area("Yahoo symbols (one per line)",
                                      "KAYNES.NS\nRELIANCE.NS", key="qs_tickers", height=90)
            _qs_symbols = [s.strip() for s in _qs_raw.splitlines() if s.strip()]
        elif _qs_universe == "All Nifty 50 Stocks":
            _qs_symbols = NIFTY50_SYMBOLS
            _qc1.caption(f"{len(_qs_symbols)} Nifty 50 stocks")
        else:
            _qs_symbols = [TICKER_MAP[_qs_universe]]
            _qc1.caption(f"Ticker: {_qs_symbols[0]}")

        _qs_strategy = _qc2.selectbox("Strategy for scan", STRATEGIES,
                                       index=STRATEGIES.index("EMA Crossover"), key="qs_strategy")
        _qs_lookback = _qc2.slider("Signal lookback (bars)", 1, 5, 2, key="qs_lookback")
        _qc3.markdown("**Scan matrix:**")
        for _iv_k, _pd_list in _QS_MATRIX.items():
            _qc3.caption(f"`{_iv_k}` → {', '.join(_pd_list)}")

        if st.button("⚡ Quick Scan All Timeframes", type="primary", key="btn_qs_scan"):
            if not _qs_symbols:
                st.error("No tickers selected.")
            else:
                _qs_results = []   # {ticker, interval, period, signal, close, bar_time}
                _qs_params  = {**sb_params}
                _fn_qs      = STRATEGY_FN.get(_qs_strategy, sig_custom)

                # Total combos for progress bar
                _qs_combos = [(sym, iv, pd_v)
                              for sym in _qs_symbols
                              for iv, pd_list in _QS_MATRIX.items()
                              for pd_v in pd_list]
                _qs_total = len(_qs_combos)
                _qprog = st.progress(0)
                _qstat = st.empty()

                for _qi, (_qsym, _qiv, _qpd) in enumerate(_qs_combos):
                    _qstat.caption(f"Scanning {_qsym} {_qiv}/{_qpd} ({_qi+1}/{_qs_total})…")
                    _qprog.progress((_qi+1)/_qs_total)
                    try:
                        time.sleep(1.5)
                        _qraw = yf.download(_qsym, period=_qpd,
                                            interval=YF_IV.get(_qiv,_qiv),
                                            progress=False, auto_adjust=True)
                        if _qraw is None or _qraw.empty: continue
                        _qdf = _flatten(_qraw)
                        if _qiv=="4h": _qdf=_r4h(_qdf)
                        if len(_qdf)<10: continue

                        try: _qsigs, _ = _fn_qs(_qdf, **_qs_params)
                        except: continue

                        _qn = len(_qdf)
                        # Check last _qs_lookback bars
                        _q_recent = _qsigs.iloc[max(0,_qn-_qs_lookback-1):_qn-1]
                        _q_long   = int((_q_recent==1).sum())
                        _q_short  = int((_q_recent==-1).sum())
                        _q_last   = int(_qsigs.iloc[-2]) if _qn>1 else 0
                        _q_close  = float(_qdf["Close"].iloc[-1])
                        _q_bar    = to_ist(_qdf.index[-1])

                        if _q_long>0 or _q_short>0 or _q_last!=0:
                            _sig_d = "🟢 BUY" if (_q_last==1 or _q_long>0) else "🔴 SELL"
                            _qs_results.append({
                                "Ticker":   _qsym,
                                "Interval": _qiv,
                                "Period":   _qpd,
                                "Signal":   _sig_d,
                                "Long Fired":  _q_long,
                                "Short Fired": _q_short,
                                "Last Bar Signal": "BUY" if _q_last==1 else ("SELL" if _q_last==-1 else "—"),
                                "Close":    round(_q_close,2),
                                "Bar Time": _q_bar,
                                "_dir":     "BUY" if (_q_last==1 or _q_long>0) else "SELL",
                            })
                    except: pass

                _qprog.empty(); _qstat.empty()
                st.session_state["qs_results"]      = _qs_results
                st.session_state["qs_scan_strategy"]= _qs_strategy

        # ── Display quick search results ──────────────────────────────────────
        _qsr = st.session_state.get("qs_results")
        _qss = st.session_state.get("qs_scan_strategy","")

        if _qsr is not None:
            st.markdown(f"**Quick scan complete.** Found **{len(_qsr)}** signal(s) across all timeframes.")

            if _qsr:
                st.markdown("### 📊 Signals Found — click Apply to trade directly")
                st.caption("Each row shows a fired signal on a specific ticker + timeframe + period. "
                           "Clicking Apply sets the sidebar to that exact combination and uses "
                           "Simple Buy / Simple Sell strategy for immediate live/backtest execution.")

                # Group by ticker for cleaner display
                _qs_tickers = list(dict.fromkeys(r["Ticker"] for r in _qsr))
                for _qt in _qs_tickers:
                    _qt_rows = [r for r in _qsr if r["Ticker"]==_qt]
                    st.markdown(f"#### {_qt}  ({len(_qt_rows)} signal timeframe(s))")
                    for _qri, _qrow in enumerate(_qt_rows):
                        _qd = _qrow["_dir"]
                        _col1,_col2,_col3,_col4,_col5,_col6,_col7 = st.columns([1,1,1,1.2,1.2,1,2])
                        _col1.markdown(f"**{_qrow['Interval']}**")
                        _col2.markdown(f"_{_qrow['Period']}_")
                        _col3.markdown(f"`{_qrow['Signal']}`")
                        _col4.markdown(f"Long:{_qrow['Long Fired']} Short:{_qrow['Short Fired']}")
                        _col5.markdown(f"Close: {_qrow['Close']}")
                        _col6.markdown(f"{_qrow['Bar Time'][:13]}")
                        _apply_strat_qs = ("Simple Buy" if _qd=="BUY" else "Simple Sell")
                        if _col7.button(
                            f"{'🟢' if _qd=='BUY' else '🔴'} Apply {_qd} → Live + BT",
                            key=f"qs_apply_{_qt}_{_qrow['Interval']}_{_qrow['Period']}_{_qri}",
                            type="primary" if _qd=="BUY" else "secondary",
                        ):
                            _nte_apply_to_live(
                                ticker_sym=_qt,
                                ticker_label=_qt,
                                sig_direction=_qd,
                                iv=_qrow["Interval"],
                                pd_val=_qrow["Period"],
                                strat=_apply_strat_qs,
                            )
                            st.rerun()
                    st.markdown("---")

                # Summary table (no buttons — just overview)
                with st.expander("📋 Full results table"):
                    _qs_disp = pd.DataFrame([
                        {k:v for k,v in r.items() if k!="_dir"} for r in _qsr
                    ])
                    def _qs_sig_col(v):
                        if "BUY"  in str(v): return "color:#2e7d32;font-weight:bold"
                        if "SELL" in str(v): return "color:#c62828;font-weight:bold"
                        return ""
                    st.dataframe(_qs_disp.style.map(_qs_sig_col,subset=["Signal"]),
                                 use_container_width=True)
            else:
                st.info("No signals found across any timeframe/period combination. "
                        "Try a different strategy or increase the lookback slider.")
        else:
            st.info("Select tickers above and click ⚡ Quick Scan All Timeframes.")

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 3: Elliott Wave Monitor — continuous EW signal scanner
    # ══════════════════════════════════════════════════════════════════════════
    with nte_ew_tab:
        st.markdown(
            "**Continuously monitor Elliott Wave signals across selected tickers.** "
            "Shows exact wave stage, progress toward next signal, and fires an Apply button "
            "the moment a pattern completes. No need to watch manually — run this tab and "
            "it tells you exactly which stock is closest to an EW entry."
        )

        _ew_c1, _ew_c2, _ew_c3 = st.columns(3)
        _ew_universe = _ew_c1.selectbox(
            "Universe",
            ["All Nifty 50 Stocks","Custom Tickers"] + list(TICKER_MAP.keys()),
            key="ewm_universe"
        )
        if _ew_universe == "Custom Tickers":
            _ew_raw = _ew_c1.text_area("Tickers (one per line)",
                                        "RELIANCE.NS\nINFY.NS\nHDFCBANK.NS",
                                        key="ewm_tickers", height=80)
            _ew_symbols = [s.strip() for s in _ew_raw.splitlines() if s.strip()]
        elif _ew_universe == "All Nifty 50 Stocks":
            _ew_symbols = NIFTY50_SYMBOLS
            _ew_c1.caption(f"{len(_ew_symbols)} stocks")
        else:
            _ew_symbols = [TICKER_MAP[_ew_universe]]

        _ew_iv  = _ew_c2.selectbox("Timeframe", TIMEFRAMES, index=4, key="ewm_iv")
        _ew_pd  = _ew_c2.selectbox("Period",    PERIODS,    index=4, key="ewm_pd")
        _ew_mwp = _ew_c2.slider("Min Wave % (pivot threshold)", 0.2, 5.0, 1.0, step=0.1, key="ewm_mwp")
        _ew_show_all = _ew_c3.checkbox("Show all tickers (not just signals)", value=False, key="ewm_show_all")

        if st.button("🌊 Scan Elliott Waves", type="primary", key="btn_ew_scan"):
            _ew_results = []
            _ew_prog = st.progress(0); _ew_stat = st.empty()
            _ew_total = len(_ew_symbols)

            for _ewi, _ewsym in enumerate(_ew_symbols):
                _ew_stat.caption(f"Scanning EW: {_ewsym} ({_ewi+1}/{_ew_total})…")
                _ew_prog.progress((_ewi+1)/_ew_total)
                try:
                    time.sleep(1.5)
                    _ewraw = yf.download(_ewsym, period=_ew_pd,
                                         interval=YF_IV.get(_ew_iv,_ew_iv),
                                         progress=False, auto_adjust=True)
                    if _ewraw is None or _ewraw.empty: continue
                    _ewdf = _flatten(_ewraw)
                    if _ew_iv=="4h": _ewdf=_r4h(_ewdf)
                    if len(_ewdf)<20: continue

                    # Run EW diagnostics (use v2 if available)
                    try:
                        _ewd = _ew2_diagnostics(_ewdf,
                                                 swing_bars=int(sb_params.get("swing_bars",5)),
                                                 fib_min=float(sb_params.get("fib_min",0.382)),
                                                 fib_max=float(sb_params.get("fib_max",0.886)))
                        _ew_sigs, _ = sig_elliott_wave_v2(_ewdf,
                                                            swing_bars=int(sb_params.get("swing_bars",5)),
                                                            fib_min=float(sb_params.get("fib_min",0.382)),
                                                            fib_max=float(sb_params.get("fib_max",0.886)),
                                                            ema_period=int(sb_params.get("ema_period",50)))
                    except:
                        _ewd = _ew_diagnostics(_ewdf, min_wave_pct=float(_ew_mwp))
                        _ew_sigs, _ = sig_elliott_wave(_ewdf, min_wave_pct=float(_ew_mwp))
                    _ew_last_sig = int(_ew_sigs.iloc[-2]) if len(_ew_sigs)>1 else 0
                    _ew_close    = float(_ewdf["Close"].iloc[-1])

                    # Progress toward next pivot flip
                    _prog_pct = min(100, int(_ewd["pivot_flip_pct"]))
                    _pivot_remaining = _ewd["pct_remaining"]

                    # Categorize: FIRED / VERY CLOSE / CLOSE / BUILDING
                    if _ew_last_sig != 0:
                        _ew_status = "🚨 SIGNAL FIRED"
                        _ew_priority = 0
                    elif _prog_pct >= 80:
                        _ew_status = "🔥 VERY CLOSE"
                        _ew_priority = 1
                    elif _prog_pct >= 50:
                        _ew_status = "⚡ APPROACHING"
                        _ew_priority = 2
                    else:
                        _ew_status = "🔨 BUILDING"
                        _ew_priority = 3

                    _ew_results.append({
                        "_priority": _ew_priority,
                        "Ticker":      _ewsym,
                        "Status":      _ew_status,
                        "Signal":      "🟢 LONG" if _ew_last_sig==1 else ("🔴 SHORT" if _ew_last_sig==-1 else "—"),
                        "Pivot Progress": f"{_prog_pct}%",
                        "Pts to Next Pivot": f"{_pivot_remaining:.2f}%",
                        "Confirmed Pivots": _ewd["confirmed_pivots"],
                        "Last Pivot Type": "HIGH ↑" if _ewd["last_confirmed_dir"]==1 else ("LOW ↓" if _ewd["last_confirmed_dir"]==-1 else "—"),
                        "Last Pivot Px": f"{_ewd['last_confirmed_px']:.2f}",
                        "LTP":         round(_ew_close, 2),
                        "Move from Last": f"{_ewd['move_from_last_pct']:+.2f}%",
                        "Retrace %":   f"{_ewd['retrace_pct']:.1f}%" if _ewd["retrace_pct"] else "—",
                        "Last Fired":  _ewd["last_signal"] or "None",
                        "Bar Time":    to_ist(_ewdf.index[-1]),
                        "_sig_dir":    "BUY" if _ew_last_sig==1 else ("SELL" if _ew_last_sig==-1 else
                                       "BUY" if _ewd["last_confirmed_dir"]==-1 else "SELL"),
                    })
                except Exception as _ewex:
                    pass

            _ew_prog.empty(); _ew_stat.empty()

            # Sort by priority then progress
            _ew_results.sort(key=lambda r: (r["_priority"], -int(r["Pivot Progress"].rstrip("%"))))
            st.session_state["ew_scan_results"] = _ew_results
            st.session_state["ew_scan_meta"]    = {"iv":_ew_iv,"pd":_ew_pd,"mwp":_ew_mwp}

        # ── Display EW scan results ───────────────────────────────────────────
        _ewr = st.session_state.get("ew_scan_results")
        _ewm = st.session_state.get("ew_scan_meta",{})

        if _ewr is not None:
            # Filter if not showing all
            _ewr_disp = _ewr if _ew_show_all else [r for r in _ewr if r["_priority"]<=2]
            st.markdown(f"**Scan complete.** {len(_ewr)} tickers. Showing {len(_ewr_disp)} (priority: fired+close+approaching).")

            # Summary counts
            _ew_s1,_ew_s2,_ew_s3,_ew_s4 = st.columns(4)
            _ew_s1.metric("🚨 Signal Fired", sum(1 for r in _ewr if r["_priority"]==0))
            _ew_s2.metric("🔥 Very Close",   sum(1 for r in _ewr if r["_priority"]==1))
            _ew_s3.metric("⚡ Approaching",  sum(1 for r in _ewr if r["_priority"]==2))
            _ew_s4.metric("🔨 Building",     sum(1 for r in _ewr if r["_priority"]==3))

            st.markdown("---")
            for _ewri, _erow in enumerate(_ewr_disp):
                _prio = _erow["_priority"]
                _bg   = {"0":"#1b5e20","1":"#b71c1c","2":"#e65100","3":""}
                _ec1,_ec2,_ec3,_ec4,_ec5,_ec6,_ec7,_ec8 = st.columns([1.5,1,1,1,1,1,1,1.8])
                _ec1.markdown(f"**{_erow['Ticker']}**")
                _ec2.markdown(f"{_erow['Status']}")
                _ec3.markdown(f"{_erow.get('Signal','—')}")
                _ec4.markdown(f"Progress: **{_erow['Pivot Progress']}**")
                _ec5.markdown(f"Pivots: {_erow['Confirmed Pivots']}")
                _ec6.markdown(f"LTP: {_erow['LTP']}")
                _ec7.markdown(f"Move: {_erow['Move from Last']}")

                _ewsig_dir = _erow["_sig_dir"]
                if _ec8.button(
                    f"{'🟢 Apply BUY' if _ewsig_dir=='BUY' else '🔴 Apply SELL'} → Live",
                    key=f"ew_apply_{_ewri}_{_erow['Ticker']}",
                    type="primary" if _prio<=1 else "secondary",
                ):
                    _nte_apply_to_live(
                        ticker_sym  = _erow["Ticker"],
                        ticker_label= _erow["Ticker"],
                        sig_direction= _ewsig_dir,
                        iv          = _ewm.get("iv","1h"),
                        pd_val      = _ewm.get("pd","3mo"),
                        strat       = "Elliott Wave (Simplified)",
                    )
                    st.rerun()

                # Expandable detail row
                if _prio <= 1:
                    with st.expander(f"📊 {_erow['Ticker']} — full wave detail", expanded=(_ewri==0 and _prio==0)):
                        _ed1,_ed2,_ed3,_ed4 = st.columns(4)
                        _ed1.metric("Last Pivot",   f"{_erow['Last Pivot Type']} @ {_erow['Last Pivot Px']}")
                        _ed2.metric("Retrace",      _erow["Retrace %"])
                        _ed3.metric("Last Signal",  _erow["Last Fired"])
                        _ed4.metric("Bar Time",     _erow["Bar Time"])
                        _prog_int = int(_erow["Pivot Progress"].rstrip("%"))
                        _bar = "█"*int(_prog_int/5) + "░"*(20-int(_prog_int/5))
                        st.markdown(f"**Pivot confirmation:** `[{_bar}]` {_prog_int}% — "
                                    f"need **{_erow['Pts to Next Pivot']}** more move")
            if not _ewr_disp:
                st.info("No tickers in fired/approaching state. All waves are still in early building phase. "
                        "Enable 'Show all tickers' to see the full list.")
        else:
            st.info("Click 🌊 Scan Elliott Waves to start monitoring. "
                    "Re-run periodically (every few minutes) to catch new signals. "
                    "Tickers with 🚨 SIGNAL FIRED or 🔥 VERY CLOSE (>80%) are the ones to watch immediately.")

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 4: Position Tracker — what's entered, what's about to enter
    # Shows live position status, recent EW signals across tickers, and
    # an EW "proximity alert" so you don't have to watch all day.
    # ══════════════════════════════════════════════════════════════════════════
    with nte_pos_tab:
        st.markdown(
            "**Central dashboard:** Shows your current open position (if any), "
            "recently fired EW signals across watched tickers, and which tickers "
            "are within N bars of generating an EW signal — so you only need to "
            "monitor at the right time."
        )

        # ── Section 1: Current open position from live trading ────────────────
        st.markdown("### 📌 Current Live Position")
        _lv_pos = st.session_state.get("live_position")
        _lv_active = st.session_state.get("live_active", False)
        if not _lv_active:
            st.info("Live trading is not running. Start it in the ⚡ Live Trading tab.")
        elif _lv_pos is None:
            st.success("✅ Live trading is running — **no open position** right now. Waiting for signal.")
        else:
            _d = _lv_pos["direction"]
            _ep = _lv_pos["entry"]
            # Get current LTP from last known lv_df (not available here — use sidebar sym)
            _pos_cols = st.columns(6)
            _pos_cols[0].metric("Direction",  "🟢 LONG" if _d==1 else "🔴 SHORT")
            _pos_cols[1].metric("Entry Price", f"{_ep:.2f}")
            _pos_cols[2].metric("SL",          f"{_lv_pos['sl']:.2f}")
            _pos_cols[3].metric("Target",      f"{_lv_pos['disp_tgt']:.2f}")
            _pos_cols[4].metric("Highest",     f"{_lv_pos['highest']:.2f}")
            _pos_cols[5].metric("Lowest",      f"{_lv_pos['lowest']:.2f}")
            st.caption(f"Entry time: {to_ist(_lv_pos['entry_time'])}")

        # ── Section 2: Trade History Summary ─────────────────────────────────
        st.markdown("### 📜 Completed Trades (this session)")
        _hist = st.session_state.get("live_trades", [])
        # Filter to current sidebar ticker for clean display
        _hist_sym = [t for t in _hist if t.get("Ticker","") == sym]
        _hist_other = len(_hist) - len(_hist_sym)
        if _hist_sym:
            _tot  = len(_hist_sym); _wins = sum(1 for x in _hist_sym if x["PnL"]>0)
            _pnl  = sum(x["PnL"] for x in _hist_sym)
            _hc   = st.columns(4)
            _hc[0].metric("Total Trades", _tot)
            _hc[1].metric("Wins",         f"{_wins} ({_wins/_tot*100:.0f}%)")
            _hc[2].metric("PnL",          f"{_pnl:+.2f}", delta_color="normal" if _pnl>=0 else "inverse")
            _hc[3].metric("Last Exit",    _hist_sym[-1].get("Exit Reason","—"))
            if _hist_other:
                st.caption(f"Showing {_tot} trades for **{sym}**. {_hist_other} trades for other tickers in session — see Trade History tab.")
            _hdf = pd.DataFrame(_hist_sym)
            def _pnl_c(v):
                if isinstance(v,(int,float)): return "color:#00E676" if v>0 else ("color:#FF5252" if v<0 else "")
                return ""
            st.dataframe(_hdf.style.map(_pnl_c, subset=["PnL"]),
                         use_container_width=True,
                         column_config={
                             "Entry Time": st.column_config.TextColumn("Entry Time (IST)", width="large"),
                             "Exit Time":  st.column_config.TextColumn("Exit Time (IST)",  width="large"),
                         })
        else:
            st.info(f"No completed trades for **{sym}** this session." +
                    (f" ({_hist_other} trades for other tickers exist.)" if _hist_other else ""))

        st.markdown("---")

        # ── Section 3: EW Proximity Alert — smart monitoring ─────────────────
        st.markdown("### 🎯 Elliott Wave Proximity Alert")
        st.caption(
            "Scan multiple tickers to find which are CLOSE to generating an EW signal. "
            "Instead of watching all day, check this every 15-30 minutes. "
            "When a ticker shows 🔥 VERY CLOSE, start monitoring that ticker actively."
        )

        _pt_c1, _pt_c2, _pt_c3 = st.columns(3)
        _pt_universe = _pt_c1.selectbox(
            "Tickers to watch",
            ["All Nifty 50 Stocks","Custom Tickers"] + list(TICKER_MAP.keys()),
            key="pt_universe"
        )
        if _pt_universe == "Custom Tickers":
            _pt_raw = _pt_c1.text_area("Symbols (one per line)",
                                        "RELIANCE.NS\nINFY.NS\nHDFCBANK.NS",
                                        key="pt_tickers", height=80)
            _pt_symbols = [s.strip() for s in _pt_raw.splitlines() if s.strip()]
        elif _pt_universe == "All Nifty 50 Stocks":
            _pt_symbols = NIFTY50_SYMBOLS
        else:
            _pt_symbols = [TICKER_MAP[_pt_universe]]

        _pt_iv      = _pt_c2.selectbox("Timeframe", TIMEFRAMES, index=2, key="pt_iv")   # 15m default
        _pt_pd      = _pt_c2.selectbox("Period",    PERIODS,    index=2, key="pt_pd")   # 7d default
        _pt_mwp     = _pt_c2.slider("Min Wave %", 0.1, 3.0, 0.5, step=0.1, key="pt_mwp")
        _pt_alert   = _pt_c3.slider("Alert when progress ≥ (%)", 50, 95, 70, key="pt_alert")
        _pt_lookback= _pt_c3.slider("Signal lookback (bars)", 3, 30, 15, key="pt_lb")

        st.info(
            f"**How to use:** Click Scan every 15-30 mins. Tickers at ≥{_pt_alert}% progress "
            "are worth monitoring actively. 🚨 FIRED = signal already in recent bars — "
            "switch to Live Trading immediately and check Entry Gate."
        )

        if st.button("🔍 Scan for EW Proximity", type="primary", key="btn_pt_scan"):
            _pt_results = []
            _pt_prog = st.progress(0); _pt_stat = st.empty()
            for _pti, _ptsym in enumerate(_pt_symbols):
                _pt_stat.caption(f"Scanning {_ptsym} ({_pti+1}/{len(_pt_symbols)})…")
                _pt_prog.progress((_pti+1)/max(len(_pt_symbols),1))
                try:
                    time.sleep(1.5)
                    _ptraw = yf.download(_ptsym, period=_pt_pd,
                                          interval=YF_IV.get(_pt_iv,_pt_iv),
                                          progress=False, auto_adjust=True)
                    if _ptraw is None or _ptraw.empty: continue
                    _ptdf = _flatten(_ptraw)
                    if _pt_iv=="4h": _ptdf=_r4h(_ptdf)
                    if len(_ptdf)<20: continue

                    # Run EW signal and diagnostics
                    _pt_sigs, _ = sig_elliott_wave(_ptdf,
                                                    swing_lookback=10,
                                                    min_wave_pct=float(_pt_mwp))
                    _pt_ewd = _ew_diagnostics(_ptdf, min_wave_pct=float(_pt_mwp))
                    _pt_n   = len(_ptdf)
                    _pt_close = float(_ptdf["Close"].iloc[-1])
                    _pt_bar   = to_ist(_ptdf.index[-1])

                    # Check recent signal (within lookback)
                    _pt_last_sig = 0
                    for _psi in range(1, _pt_lookback+2):
                        if _pt_n > _psi and int(_pt_sigs.iloc[-_psi]) != 0:
                            _pt_last_sig = int(_pt_sigs.iloc[-_psi]); break

                    _prog_pct  = min(100, int(_pt_ewd["pivot_flip_pct"]))
                    _bars_wait = int(_pt_ewd.get("bars_since_pivot", 0))

                    # Status
                    if _pt_last_sig != 0:
                        _pt_status = "🚨 SIGNAL FIRED"; _pt_pri = 0
                    elif _prog_pct >= _pt_alert:
                        _pt_status = "🔥 VERY CLOSE";   _pt_pri = 1
                    elif _prog_pct >= 50:
                        _pt_status = "⚡ APPROACHING";  _pt_pri = 2
                    else:
                        _pt_status = "🔨 BUILDING";     _pt_pri = 3

                    # Estimate bars to signal
                    _move_per_bar = abs(_pt_ewd.get("current_swing_pct",0)) / max(_bars_wait,1)
                    _needed       = max(0, _pt_mwp - abs(_pt_ewd.get("move_from_last_pct",0)))
                    _est_bars     = int(_needed / max(_move_per_bar, 0.01)) if _move_per_bar > 0 else 999
                    _est_time_str = (f"~{_est_bars} bars" if _est_bars < 999 else "unclear") + (
                        f" (~{_est_bars*_pt_iv_mins(_pt_iv)} min)" if _pt_iv in ("1m","5m","15m","30m") else ""
                    )

                    _pt_results.append({
                        "_pri":          _pt_pri,
                        "Ticker":        _ptsym,
                        "Status":        _pt_status,
                        "Signal":        "🟢 LONG" if _pt_last_sig==1 else ("🔴 SHORT" if _pt_last_sig==-1 else "—"),
                        "Progress":      f"{_prog_pct}%",
                        "Est. Time":     _est_time_str,
                        "LTP":           round(_pt_close, 2),
                        "Pivots":        _pt_ewd["confirmed_pivots"],
                        "Last Pivot":    "HIGH" if _pt_ewd["last_confirmed_dir"]==1 else ("LOW" if _pt_ewd["last_confirmed_dir"]==-1 else "—"),
                        "Bars Since Pv": _bars_wait,
                        "Last Signal":   _pt_ewd.get("last_signal","—") or "—",
                        "Bar Time":      _pt_bar,
                        "_sig_dir":      "BUY" if _pt_last_sig==1 else ("SELL" if _pt_last_sig==-1 else
                                         "BUY" if _pt_ewd["last_confirmed_dir"]==-1 else "SELL"),
                    })
                except: pass

            _pt_prog.empty(); _pt_stat.empty()
            _pt_results.sort(key=lambda r: (r["_pri"], -int(r["Progress"].rstrip("%"))))
            st.session_state["pt_scan_results"] = _pt_results
            st.session_state["pt_scan_meta"]    = {"iv":_pt_iv,"pd":_pt_pd,"mwp":_pt_mwp}

        # helper
        def _pt_iv_mins(iv):
            return {"1m":1,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240}.get(iv,0)

        # Display
        _ptr = st.session_state.get("pt_scan_results")
        _ptm = st.session_state.get("pt_scan_meta",{})
        if _ptr is not None:
            _fired   = [r for r in _ptr if r["_pri"]==0]
            _close   = [r for r in _ptr if r["_pri"]==1]
            _appr    = [r for r in _ptr if r["_pri"]==2]
            _build   = [r for r in _ptr if r["_pri"]==3]

            _ps1,_ps2,_ps3,_ps4 = st.columns(4)
            _ps1.metric("🚨 Fired",      len(_fired))
            _ps2.metric("🔥 Very Close", len(_close))
            _ps3.metric("⚡ Approaching",len(_appr))
            _ps4.metric("🔨 Building",   len(_build))

            # Show fired + very close + approaching
            _show_r = _fired + _close + _appr
            if _show_r:
                st.markdown(f"#### Tickers requiring attention ({len(_show_r)})")
                for _pri, _prow in enumerate(_show_r):
                    _pc1,_pc2,_pc3,_pc4,_pc5,_pc6,_pc7 = st.columns([1.5,1.2,1,1,1,1,2])
                    _pc1.markdown(f"**{_prow['Ticker']}**")
                    _pc2.markdown(f"{_prow['Status']}")
                    _pc3.markdown(f"`{_prow['Signal']}`")
                    _pc4.markdown(f"**{_prow['Progress']}**")
                    _pc5.markdown(f"Est: {_prow['Est. Time']}")
                    _pc6.markdown(f"LTP: {_prow['LTP']}")
                    _psig = _prow["_sig_dir"]
                    if _pc7.button(
                        f"{'🟢' if _psig=='BUY' else '🔴'} Apply → Live",
                        key=f"pt_apply_{_pri}_{_prow['Ticker']}",
                        type="primary" if _prow["_pri"]<=1 else "secondary"
                    ):
                        _nte_apply_to_live(_prow["Ticker"],_prow["Ticker"],_psig,
                                           _ptm.get("iv","15m"),_ptm.get("pd","7d"),
                                           "Elliott Wave (Simplified)")
                        st.rerun()

                # Full table
                with st.expander("📊 Full results table"):
                    _ptdf_disp = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                                                for r in _show_r])
                    st.dataframe(_ptdf_disp, use_container_width=True,
                                 column_config={
                                     "Bar Time": st.column_config.TextColumn("Bar Time (IST)", width="large"),
                                     "Est. Time": st.column_config.TextColumn("Est. Time to Signal", width="medium"),
                                 })
            else:
                st.info(f"No tickers reached {_pt_alert}% progress. All waves are still building. "
                        "Check back in 15-30 mins.")

            if _build:
                with st.expander(f"🔨 {len(_build)} tickers in building phase"):
                    _ptdf_b = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                                             for r in _build])
                    st.dataframe(_ptdf_b, use_container_width=True)
        else:
            st.info("Click 🔍 Scan for EW Proximity to see which tickers are close to a signal.")

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 5: Fresh Signal Finder
    # Finds the BEST combination of (ticker × timeframe × period × strategy)
    # where the signal fired very recently, price hasn't run away, and you're
    # not late — so you can apply it directly and start live trading NOW.
    # ══════════════════════════════════════════════════════════════════════════
    with nte_fresh_tab:
        st.markdown(
            "**Find fresh, actionable signals right now.** "
            "Scans combinations of tickers, timeframes and periods to find signals that:\n"
            "- Fired within your max bars threshold (you're not late)\n"
            "- Price hasn't moved more than your max-pts limit toward target (move still available)\n"
            "- Minimal time delay between signal bar and now\n\n"
            "Click **Apply** on any result to instantly configure sidebar, backtesting and live trading."
        )

        # ── Inputs ────────────────────────────────────────────────────────────
        with st.expander("⚙️ Search Parameters", expanded=True):
            _fc1, _fc2, _fc3 = st.columns(3)

            # Tickers
            _fs_univ = _fc1.selectbox("Universe",
                ["All Nifty 50 Stocks","Custom Tickers"] + list(TICKER_MAP.keys()),
                key="fs_universe")
            if _fs_univ == "Custom Tickers":
                _fs_raw = _fc1.text_area("Symbols (one per line)",
                                          "RELIANCE.NS\nINFY.NS\nHDFCBANK.NS",
                                          key="fs_tickers", height=80)
                _fs_symbols = [s.strip() for s in _fs_raw.splitlines() if s.strip()]
            elif _fs_univ == "All Nifty 50 Stocks":
                _fs_symbols = NIFTY50_SYMBOLS
                _fc1.caption(f"{len(_fs_symbols)} Nifty 50 stocks")
            else:
                _fs_symbols = [TICKER_MAP[_fs_univ]]
                _fc1.caption(f"Ticker: {_fs_symbols[0]}")

            # Timeframes to scan
            _fs_ivs_opts = ["1m","5m","15m","30m","1h"]
            _fs_ivs = _fc1.multiselect("Timeframes to scan",
                _fs_ivs_opts, default=["1m","5m","15m"], key="fs_ivs")

            # Periods to scan
            _fs_pds_opts = ["1d","5d","7d","1mo"]
            _fs_pds = _fc1.multiselect("Periods to scan",
                _fs_pds_opts, default=["1d","5d","7d"], key="fs_pds")

            # Strategy
            _fs_strategy = _fc2.selectbox("Strategy", STRATEGIES,
                index=STRATEGIES.index("Elliott Wave (Simplified)") if "Elliott Wave (Simplified)" in STRATEGIES else 0,
                key="fs_strategy")

            # Key thresholds
            _fs_max_bars = _fc2.number_input(
                "Max bars since signal fired",
                min_value=1, max_value=50, value=2, step=1,
                key="fs_max_bars",
                help="Only show signals that fired within this many bars. "
                     "2 bars on 1m = signal fired within 2 minutes. "
                     "2 bars on 5m = within 10 minutes."
            )
            _fs_max_pts = _fc2.number_input(
                "Max price move toward target (pts)",
                min_value=0.0, max_value=500.0, value=10.0, step=0.5,
                key="fs_max_pts",
                help="Skip signals where price already moved more than this many points "
                     "in the signal direction. Keeps entries where the move is still available."
            )
            _fs_max_delay_mins = _fc2.number_input(
                "Max acceptable signal age (minutes)",
                min_value=1, max_value=480, value=30, step=1,
                key="fs_max_delay",
                help="Skip signals where signal bar time vs current time gap exceeds this. "
                     "Prevents acting on very old signals."
            )

            # SL/Target for scoring
            _fs_sl  = _fc3.number_input("SL (pts)",  0.1, 1e6, 10., step=0.5, key="fs_sl")
            _fs_tgt = _fc3.number_input("Tgt (pts)", 0.1, 1e6, 20., step=0.5, key="fs_tgt")
            _fs_params = {**sb_params}

            # Min bars of data needed
            _fc3.markdown("**Scoring weights:**")
            _fs_w_bars  = _fc3.slider("Weight: Freshness (fewer bars = better)", 0, 10, 5, key="fs_w_bars")
            _fs_w_pts   = _fc3.slider("Weight: Price move (less moved = better)", 0, 10, 5, key="fs_w_pts")
            _fs_w_delay = _fc3.slider("Weight: Time delay (less delay = better)", 0, 10, 3, key="fs_w_delay")

        st.caption(
            f"Will scan: **{len(_fs_symbols)} ticker(s)** × "
            f"**{len(_fs_ivs)} timeframe(s)** × "
            f"**{len(_fs_pds)} period(s)** = "
            f"**{len(_fs_symbols)*len(_fs_ivs)*len(_fs_pds)} combinations**. "
            f"Rate-limited to 1.5s per fetch. Estimate: "
            f"~{len(_fs_symbols)*len(_fs_ivs)*len(_fs_pds)*2:.0f}s total."
        )

        if st.button("🎯 Find Fresh Signals", type="primary", key="btn_fs_scan"):
            if not _fs_ivs or not _fs_pds:
                st.error("Select at least one timeframe and one period.")
            elif not _fs_symbols:
                st.error("No tickers selected.")
            else:
                _fs_results = []
                _fs_fn = STRATEGY_FN.get(_fs_strategy, sig_custom)
                _total_combos = len(_fs_symbols) * len(_fs_ivs) * len(_fs_pds)
                _fs_prog = st.progress(0); _fs_stat = st.empty(); _fs_idx = 0

                import pytz as _pytz_fs
                _ist_fs = _pytz_fs.timezone("Asia/Kolkata")
                _now_fs = datetime.now(_ist_fs)
                _mins_map_fs = {"1m":1,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}

                for _fssym in _fs_symbols:
                    for _fsiv in _fs_ivs:
                        for _fspd in _fs_pds:
                            _fs_idx += 1
                            _fs_stat.caption(f"Scanning {_fssym} | {_fsiv} | {_fspd} "
                                             f"({_fs_idx}/{_total_combos})…")
                            _fs_prog.progress(_fs_idx / _total_combos)
                            try:
                                time.sleep(1.5)
                                _fsraw = yf.download(_fssym, period=_fspd,
                                    interval=YF_IV.get(_fsiv,_fsiv),
                                    progress=False, auto_adjust=True)
                                if _fsraw is None or _fsraw.empty: continue
                                _fsdf = _flatten(_fsraw)
                                if _fsiv == "4h": _fsdf = _r4h(_fsdf)
                                if len(_fsdf) < 20: continue

                                # Run strategy
                                try:
                                    _fssigs, _ = _fs_fn(_fsdf, **_fs_params)
                                except: continue

                                _fsn      = len(_fsdf)
                                _fsltp    = float(_fsdf["Close"].iloc[-1])
                                _fsmins   = _mins_map_fs.get(_fsiv, 5)
                                _fs_wide  = {"Elliott Wave (Simplified)":20,
                                             "Elliott Wave v2 (Swing+Fib)":10,
                                             "Elliott Wave v3 (Extrema)":5}.get(_fs_strategy, 3)

                                # Find most recent signal within lookback
                                _fslast_sig = 0; _fsbars_ago = 0
                                _fssig_px = None; _fssig_dt = None
                                for _fsi in range(1, _fs_wide + 2):
                                    if _fsn > _fsi and int(_fssigs.iloc[-_fsi]) != 0:
                                        _fslast_sig = int(_fssigs.iloc[-_fsi])
                                        _fsbars_ago = _fsi - 1
                                        try:
                                            _fssig_dt = _fsdf.index[-_fsi]
                                            _fssig_px = float(_fsdf["Close"].iloc[-_fsi])
                                        except: pass
                                        break

                                if _fslast_sig == 0 or _fssig_dt is None: continue

                                # Filter 1: bars since signal
                                if _fsbars_ago > _fs_max_bars: continue

                                # Filter 2: time since signal
                                try:
                                    _sig_ts = _fssig_dt
                                    if hasattr(_sig_ts,'tzinfo') and _sig_ts.tzinfo is None:
                                        _sig_ts = _ist_fs.localize(_sig_ts.to_pydatetime())
                                    elif hasattr(_sig_ts,'tzinfo') and _sig_ts.tzinfo is not None:
                                        _sig_ts = _sig_ts.to_pydatetime().astimezone(_ist_fs)
                                    _signal_age_mins = (_now_fs - _sig_ts).total_seconds() / 60
                                except: _signal_age_mins = _fsbars_ago * _fsmins

                                if _signal_age_mins > _fs_max_delay_mins: continue

                                # Filter 3: price move toward target
                                _price_moved = (_fsltp - _fssig_px) * _fslast_sig
                                if _price_moved > _fs_max_pts: continue

                                # Compute score (lower = better entry opportunity)
                                # Score: weighted sum of: bars_ago, price_moved_pct_of_tgt, signal_age
                                _score_bars  = (_fsbars_ago / max(_fs_max_bars, 1)) * _fs_w_bars
                                _score_pts   = (max(0, _price_moved) / max(_fs_max_pts, 0.01)) * _fs_w_pts
                                _score_delay = (_signal_age_mins / max(_fs_max_delay_mins, 1)) * _fs_w_delay
                                _total_score = _score_bars + _score_pts + _score_delay
                                # Freshness rank: 100 = perfect entry, 0 = at threshold
                                _freshness_rank = max(0, 100 - int(_total_score / max(_fs_w_bars+_fs_w_pts+_fs_w_delay,1) * 100))

                                # Projected SL/Target if entering now
                                _entry_now = _fsltp
                                _sl_now    = _entry_now - _fslast_sig * _fs_sl
                                _tgt_now   = _entry_now + _fslast_sig * _fs_tgt
                                # Adjusted target if price already moved
                                _remaining_tgt = _fs_tgt - max(0, _price_moved)

                                _fs_results.append({
                                    "_score":      _total_score,
                                    "_freshness":  _freshness_rank,
                                    "Ticker":      _fssym,
                                    "Timeframe":   _fsiv,
                                    "Period":      _fspd,
                                    "Signal":      "🟢 BUY" if _fslast_sig==1 else "🔴 SELL",
                                    "Freshness":   f"{'🟢' if _freshness_rank>=75 else '🟡' if _freshness_rank>=45 else '🟠'} {_freshness_rank}/100",
                                    "Bars Since":  _fsbars_ago,
                                    "Age (min)":   round(_signal_age_mins,1),
                                    "Signal Time": to_ist(_fssig_dt),
                                    "Signal Price":round(_fssig_px,2) if _fssig_px else "—",
                                    "LTP":         round(_fsltp,2),
                                    "Moved (pts)": round(_price_moved,2),
                                    "Remaining Tgt":round(_remaining_tgt,2),
                                    "SL if Enter": round(_sl_now,2),
                                    "Tgt if Enter":round(_tgt_now,2),
                                    "Accuracy":    "—",
                                    "Total Trades":"—",
                                    "Total PnL":   "—",
                                    "_sig_dir":    "BUY" if _fslast_sig==1 else "SELL",
                                })
                            except: pass

                _fs_prog.empty(); _fs_stat.empty()

                # ── Run mini backtest on each result to get Accuracy/PnL ────────
                if _fs_results:
                    _bt_prog = st.progress(0); _bt_stat = st.empty()
                    _bt_total = len(_fs_results)
                    for _bi, _br in enumerate(_fs_results):
                        _bt_stat.caption(f"Backtesting {_br['Ticker']} {_br['Timeframe']}/{_br['Period']} "
                                         f"({_bi+1}/{_bt_total})…")
                        _bt_prog.progress((_bi+1)/_bt_total)
                        try:
                            time.sleep(1.5)
                            _bt_raw = yf.download(_br["Ticker"], period=_br["Period"],
                                interval=YF_IV.get(_br["Timeframe"],_br["Timeframe"]),
                                progress=False, auto_adjust=True)
                            if _bt_raw is not None and not _bt_raw.empty:
                                _bt_df = _flatten(_bt_raw)
                                if _br["Timeframe"]=="4h": _bt_df=_r4h(_bt_df)
                                if len(_bt_df) >= 20:
                                    _bt_res, _, _ = run_backtest(
                                        _bt_df, _fs_strategy, _fs_params,
                                        "Custom Points", _fs_sl,
                                        "Custom Points", _fs_tgt
                                    )
                                    if not _bt_res.empty:
                                        _bt_wins  = int((_bt_res["PnL"]>0).sum())
                                        _bt_total_t = len(_bt_res)
                                        _bt_acc   = round(_bt_wins/_bt_total_t*100,1) if _bt_total_t>0 else 0
                                        _bt_pnl   = round(_bt_res["PnL"].sum(),2)
                                        _br["Accuracy"]     = f"{_bt_acc}%"
                                        _br["Total Trades"] = _bt_total_t
                                        _br["Total PnL"]    = _bt_pnl
                                        # Boost score for high accuracy (subtract up to 3 from score)
                                        if _bt_acc >= 60:
                                            _br["_score"] = max(0, _br["_score"] - (_bt_acc-50)/50*3)
                        except: pass
                    _bt_prog.empty(); _bt_stat.empty()

                # Sort: primary = score ascending (lower=better freshness),
                # secondary = accuracy descending (higher=better backtest)
                def _fs_sort_key(r):
                    _acc = float(r["Accuracy"].rstrip("%")) if r["Accuracy"] not in ("—","") else 0
                    return (r["_score"], -_acc)
                _fs_results.sort(key=_fs_sort_key)
                st.session_state["fs_results"]  = _fs_results
                st.session_state["fs_scan_meta"] = {
                    "strategy": _fs_strategy, "sl": _fs_sl, "tgt": _fs_tgt,
                    "max_bars": _fs_max_bars, "max_pts": _fs_max_pts,
                }

        # ── Display results ────────────────────────────────────────────────────
        _fsr  = st.session_state.get("fs_results")
        _fsmeta = st.session_state.get("fs_scan_meta", {})

        if _fsr is not None:
            if not _fsr:
                st.warning(
                    "No fresh signals found matching your criteria. Try:\n"
                    "- Increasing **Max bars since signal fired** (e.g. 5 or 10)\n"
                    "- Increasing **Max price move toward target** (e.g. 20 pts)\n"
                    "- Increasing **Max acceptable signal age** (e.g. 60 min)\n"
                    "- Adding more timeframes or periods to scan\n"
                    "- Switching strategy (e.g. EMA Crossover generates more frequent signals)"
                )
            else:
                st.success(f"✅ Found **{len(_fsr)}** fresh signal(s). Sorted by Freshness (best first).")

                # Summary metrics
                _fsr_buy  = [r for r in _fsr if r["_sig_dir"]=="BUY"]
                _fsr_sell = [r for r in _fsr if r["_sig_dir"]=="SELL"]
                _s1,_s2,_s3,_s4 = st.columns(4)
                _s1.metric("Total Results",    len(_fsr))
                _s2.metric("🟢 BUY Signals",   len(_fsr_buy))
                _s3.metric("🔴 SELL Signals",  len(_fsr_sell))
                _s4.metric("Best Freshness",   _fsr[0]["Freshness"] if _fsr else "—")

                st.markdown("---")
                st.markdown("### 🎯 Results — Best entry opportunities (sorted best first)")
                st.caption(
                    "Each row is a unique (Ticker × Timeframe × Period) combination. "
                    "Click **Apply → Live** to configure sidebar, backtesting and live trading instantly."
                )

                for _fri, _frow in enumerate(_fsr):
                    _fr = _frow["_freshness"]
                    _badge = "🟢" if _fr >= 75 else ("🟡" if _fr >= 45 else "🟠")
                    _acc_str = _frow.get("Accuracy","—")
                    _acc_badge = ("✅" if _acc_str not in ("—","") and float(_acc_str.rstrip("%"))>=60
                                  else ("⚠️" if _acc_str not in ("—","") else ""))

                    _rc1,_rc2,_rc3,_rc4,_rc5,_rc6,_rc7,_rc8,_rc9,_rc10 = st.columns(
                        [1.2,0.7,0.7,0.9,0.9,0.8,0.8,0.9,0.9,2])
                    _rc1.markdown(f"**{_frow['Ticker']}**")
                    _rc2.markdown(f"`{_frow['Timeframe']}`")
                    _rc3.markdown(f"_{_frow['Period']}_")
                    _rc4.markdown(f"{_frow['Signal']}")
                    _rc5.markdown(f"{_badge} **{_fr}/100**")
                    _rc6.markdown(f"{_frow['Bars Since']}bar/{_frow['Age (min)']}m")
                    _rc7.markdown(f"Mv:{_frow['Moved (pts)']}pt")
                    _rc8.markdown(f"{_acc_badge} {_acc_str}")
                    _rc9.markdown(f"PnL:{_frow.get('Total PnL','—')}")

                    _fsd = _frow["_sig_dir"]
                    if _rc10.button(
                        f"{'🟢' if _fsd=='BUY' else '🔴'} Apply → Live+BT",
                        key=f"fs_apply_{_fri}_{_frow['Ticker']}_{_frow['Timeframe']}_{_frow['Period']}",
                        type="primary" if _fr >= 75 else "secondary",
                    ):
                        _nte_apply_to_live(
                            ticker_sym   = _frow["Ticker"],
                            ticker_label = _frow["Ticker"],
                            sig_direction= _fsd,
                            iv           = _frow["Timeframe"],
                            pd_val       = _frow["Period"],
                            strat        = _fsmeta.get("strategy", _fs_strategy),
                        )
                        st.rerun()

                    # Detail expander for top results
                    if _fri < 5:
                        with st.expander(
                            f"📊 {_frow['Ticker']} {_frow['Timeframe']}/{_frow['Period']} — entry details",
                            expanded=(_fri == 0)
                        ):
                            _dc1,_dc2,_dc3,_dc4,_dc5,_dc6,_dc7,_dc8 = st.columns(8)
                            _dc1.metric("Signal Time",     _frow["Signal Time"][:16])
                            _dc2.metric("Signal Price",    _frow["Signal Price"])
                            _dc3.metric("LTP Now",         _frow["LTP"])
                            _dc4.metric("SL if Enter",     _frow["SL if Enter"])
                            _dc5.metric("Tgt if Enter",    _frow["Tgt if Enter"])
                            _dc6.metric("Remaining Tgt",   f"{_frow['Remaining Tgt']:.1f} pts")
                            _dc7.metric("Backtest Acc",    _frow.get("Accuracy","—"),
                                        help="Historical accuracy of this strategy on this ticker/TF/period")
                            _dc8.metric("Backtest PnL",    str(_frow.get("Total PnL","—")),
                                        help="Total points profit/loss in backtest")

                            _sig_d = _frow["_sig_dir"]
                            if _sig_d == "BUY":
                                st.success(
                                    f"**BUY opportunity:** Signal fired {_frow['Bars Since']} bar(s) ago "
                                    f"at {_frow['Signal Time'][:16]}. "
                                    f"Price moved only {_frow['Moved (pts)']} pts in signal direction — "
                                    f"{_frow['Remaining Tgt']:.1f} pts of target still available.\n\n"
                                    f"**Enter:** {_frow['LTP']} | **SL:** {_frow['SL if Enter']} | "
                                    f"**Target:** {_frow['Tgt if Enter']} | "
                                    f"**R:R:** 1:{_fsmeta.get('tgt',20)/_fsmeta.get('sl',10):.1f}"
                                )
                            else:
                                st.error(
                                    f"**SELL opportunity:** Signal fired {_frow['Bars Since']} bar(s) ago "
                                    f"at {_frow['Signal Time'][:16]}. "
                                    f"Price moved only {_frow['Moved (pts)']} pts — "
                                    f"{_frow['Remaining Tgt']:.1f} pts of downside still available.\n\n"
                                    f"**Enter:** {_frow['LTP']} | **SL:** {_frow['SL if Enter']} | "
                                    f"**Target:** {_frow['Tgt if Enter']} | "
                                    f"**R:R:** 1:{_fsmeta.get('tgt',20)/_fsmeta.get('sl',10):.1f}"
                                )

                # Full table
                with st.expander("📋 Full results table"):
                    _fs_disp = pd.DataFrame([
                        {k:v for k,v in r.items() if not k.startswith("_")}
                        for r in _fsr
                    ])
                    def _fs_sig_col(v):
                        if "BUY"  in str(v): return "color:#2e7d32;font-weight:bold"
                        if "SELL" in str(v): return "color:#c62828;font-weight:bold"
                        return ""
                    st.dataframe(
                        _fs_disp.style.map(_fs_sig_col, subset=["Signal"]),
                        use_container_width=True,
                        column_config={
                            "Signal Time": st.column_config.TextColumn("Signal Time (IST)", width="large"),
                        }
                    )
        else:
            st.info(
                "Configure your criteria above and click **🎯 Find Fresh Signals**.\n\n"
                "**Tip:** Start with:\n"
                "- Timeframes: 1m, 5m, 15m\n"
                "- Periods: 1d, 5d\n"
                "- Max bars since signal: 2\n"
                "- Max price move: 10 pts\n"
                "- Max signal age: 30 min\n\n"
                "The scanner will return only the combinations where you're genuinely not late."
            )

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 6: Signal Deep Dive
    # Pick a ticker + strategy, scan ALL meaningful TF/period combos.
    # For each combo: find the most recent signal, show complete trade details
    # so you can decide whether to enter the remaining move or skip.
    # ══════════════════════════════════════════════════════════════════════════
    with nte_dive_tab:
        st.markdown(
            "**Deep dive into a single ticker across ALL timeframe/period combinations.** "
            "Finds every fired or approaching signal and tells you exactly:\n"
            "- When the signal fired, entry price, direction, SL, target\n"
            "- How much has already moved and how much is still left\n"
            "- Whether to still enter or skip\n"
            "- One-click Apply to configure sidebar + live trading immediately"
        )

        # ── Inputs ────────────────────────────────────────────────────────────
        _dd_c1, _dd_c2, _dd_c3 = st.columns(3)

        # Build dropdown: Custom + TICKER_MAP instruments + all Nifty50 stocks
        _dd_all_choices = (
            ["Custom"]
            + list(TICKER_MAP.keys())
            + [f"NSE: {name}" for name, sym in sorted(NIFTY50_STOCKS.items())]
        )
        _dd_ticker_choice = _dd_c1.selectbox(
            "Ticker", _dd_all_choices,
            index=0, key="dd_ticker_choice"
        )
        if _dd_ticker_choice == "Custom":
            _dd_sym = _dd_c1.text_input("Yahoo Symbol", "INFY.NS", key="dd_custom_sym").strip()
        elif _dd_ticker_choice.startswith("NSE: "):
            # Extract symbol from NIFTY50_STOCKS dict
            _dd_nse_name = _dd_ticker_choice[5:]
            _dd_sym = NIFTY50_STOCKS.get(_dd_nse_name, "INFY.NS")
        else:
            _dd_sym = TICKER_MAP[_dd_ticker_choice]
        _dd_c1.caption(f"Scanning: **{_dd_sym}**")

        _dd_strategy = _dd_c2.selectbox(
            "Strategy",
            STRATEGIES,
            index=STRATEGIES.index("Elliott Wave (Simplified)") if "Elliott Wave (Simplified)" in STRATEGIES else 0,
            key="dd_strategy"
        )
        _dd_sl  = _dd_c2.number_input("SL (pts)",  0.1, 1e6, 10.0, step=0.5, key="dd_sl")
        _dd_tgt = _dd_c2.number_input("Tgt (pts)", 0.1, 1e6, 20.0, step=0.5, key="dd_tgt")
        _dd_mwp = _dd_c2.number_input(
            "Min Wave % (EW only)",
            min_value=0.05, max_value=10.0, value=0.5, step=0.05,
            key="dd_mwp",
            help="Min wave percentage for Elliott Wave zigzag detection. "
                 "Default 0.5% matches Elliott Wave (Simplified) default. "
                 "Lower = more signals. Use 0.2-0.5% for 1m/5m, 0.5-1.0% for 15m/1h."
        )
        _dd_params = {**sb_params}
        # Override min_wave_pct in params with the DD-specific value when using EW
        if "Elliott Wave" in _dd_strategy:
            _dd_params["min_wave_pct"] = _dd_mwp

        _dd_lookback = _dd_c3.slider("Signal lookback (bars)", 1, 50, 20, key="dd_lookback",
            help="How many recent bars to search for the most recent signal.")
        _dd_show_approaching = _dd_c3.checkbox("Also show approaching signals", value=True,
            key="dd_show_appr",
            help="Show combos where signal hasn't fired yet but is within 30% of triggering.")
        _dd_c3.markdown("**All TF × Period combos scanned:**")
        _dd_c3.caption(
            "1m: 1d,5d,7d | 5m: 1d,5d,7d,1mo | 15m: 1d,5d,7d,1mo | "
            "1h: 1d,5d,7d,1mo,3mo,6mo,1y,2y | 4h: 1d,5d,7d,1mo,3mo,6mo,1y,2y | "
            "1d: 5d,7d,1mo,3mo,6mo,1y,2y,5y | 1wk: 1d,5d,7d,1mo,3mo,6mo,1y,2y,5y,10y"
        )

        # Full scan matrix
        _DD_MATRIX = {
            "1m":  ["1d","5d","7d"],
            "5m":  ["1d","5d","7d","1mo"],
            "15m": ["1d","5d","7d","1mo"],
            "1h":  ["1d","5d","7d","1mo","3mo","6mo","1y","2y"],
            "4h":  ["1d","5d","7d","1mo","3mo","6mo","1y","2y"],
            "1d":  ["5d","7d","1mo","3mo","6mo","1y","2y","5y"],
            "1wk": ["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y","10y"],
        }
        _dd_total_combos = sum(len(v) for v in _DD_MATRIX.values())
        st.caption(f"Total combinations to scan: **{_dd_total_combos}** — rate-limited 1.5s each. "
                   f"Est. time: ~{_dd_total_combos*3}s")

        if st.button("🔎 Deep Scan", type="primary", key="btn_dd_scan"):
            _dd_fn = STRATEGY_FN.get(_dd_strategy, sig_custom)
            _dd_results = []   # fired signals
            _dd_approaching = []  # near signals

            _dd_prog = st.progress(0); _dd_stat = st.empty(); _dd_idx = 0
            _mins_map_dd = {"1m":1,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440,"1wk":10080}
            _wide_lb_dd  = {"Elliott Wave (Simplified)":20,"Elliott Wave v2 (Swing+Fib)":10,
                             "Elliott Wave v3 (Extrema)":5}
            _sig_lb_dd   = _wide_lb_dd.get(_dd_strategy, 3)

            _now_dd = datetime.now(_IST)

            for _ddiv, _ddpds in _DD_MATRIX.items():
                for _ddpd in _ddpds:
                    _dd_idx += 1
                    _dd_stat.caption(f"Scanning {_dd_sym} | {_ddiv} | {_ddpd} "
                                     f"({_dd_idx}/{_dd_total_combos})…")
                    _dd_prog.progress(_dd_idx / _dd_total_combos)
                    try:
                        time.sleep(1.5)
                        _ddraw = yf.download(_dd_sym, period=_ddpd,
                            interval=YF_IV.get(_ddiv, _ddiv),
                            progress=False, auto_adjust=True)
                        if _ddraw is None or _ddraw.empty: continue
                        _dddf = _flatten(_ddraw)
                        if _ddiv == "4h": _dddf = _r4h(_dddf)
                        if len(_dddf) < 10: continue

                        try:
                            _ddsigs, _ddindics = _dd_fn(_dddf, **_dd_params)
                        except: continue

                        _ddn       = len(_dddf)
                        _ddltp     = float(_dddf["Close"].iloc[-1])
                        _dd_bar_mins = _mins_map_dd.get(_ddiv, 5)

                        # ── Market-closed detection ────────────────────────────
                        try:
                            _dd_last_ts = _dddf.index[-1]
                            if hasattr(_dd_last_ts,'tzinfo') and _dd_last_ts.tzinfo is None:
                                _dd_last_ts = _IST.localize(_dd_last_ts.to_pydatetime())
                            elif hasattr(_dd_last_ts,'tzinfo') and _dd_last_ts.tzinfo is not None:
                                _dd_last_ts = _dd_last_ts.to_pydatetime().astimezone(_IST)
                            _dd_bar_age = (_now_dd - _dd_last_ts).total_seconds() / 60
                            _dd_mkt_closed = _dd_bar_age > _dd_bar_mins * 1.5
                        except: _dd_mkt_closed = True

                        # ── Find most recent signal ────────────────────────────
                        _dd_last_sig = 0; _dd_bars_ago = 0
                        _dd_sig_dt = None; _dd_sig_px = None
                        _dd_sig_lo = None; _dd_sig_hi = None
                        for _ddi in range(1, max(_sig_lb_dd, _dd_lookback) + 2):
                            if _ddn > _ddi and int(_ddsigs.iloc[-_ddi]) != 0:
                                _dd_last_sig = int(_ddsigs.iloc[-_ddi])
                                _dd_bars_ago = _ddi - 1
                                try:
                                    _dd_sig_dt  = _dddf.index[-_ddi]
                                    _dd_sig_px  = float(_dddf["Close"].iloc[-_ddi])
                                    _dd_sig_lo  = float(_dddf["Low"].iloc[-_ddi])
                                    _dd_sig_hi  = float(_dddf["High"].iloc[-_ddi])
                                except: pass
                                break

                        if _dd_last_sig != 0 and _dd_sig_dt is not None:
                            # Signal age
                            try:
                                _ddts = _dd_sig_dt
                                if hasattr(_ddts,'tzinfo') and _ddts.tzinfo is None:
                                    _ddts = _IST.localize(_ddts.to_pydatetime())
                                elif hasattr(_ddts,'tzinfo') and _ddts.tzinfo is not None:
                                    _ddts = _ddts.to_pydatetime().astimezone(_IST)
                                _dd_age_mins  = (_now_dd - _ddts).total_seconds() / 60
                                _dd_age_str   = (f"{int(_dd_age_mins//60)}h {int(_dd_age_mins%60)}m"
                                                 if _dd_age_mins >= 60 else f"{int(_dd_age_mins)}m")
                            except: _dd_age_mins=0; _dd_age_str="?"

                            # Signal levels (at signal bar)
                            _dd_sl_at_sig  = _dd_sig_px - _dd_last_sig * _dd_sl
                            _dd_tgt_at_sig = _dd_sig_px + _dd_last_sig * _dd_tgt

                            # Price movement since signal in signal direction
                            _dd_moved      = (_ddltp - _dd_sig_px) * _dd_last_sig  # +ve = favorable
                            _dd_moved_pts  = round(_dd_moved, 2)

                            # How far from current price to original SL and Target
                            _dd_dist_sl  = (_ddltp - _dd_sl_at_sig) * _dd_last_sig   # +ve = SL not yet hit
                            _dd_dist_tgt = (_dd_tgt_at_sig - _ddltp) * _dd_last_sig  # +ve = target not hit
                            _dd_dist_sl  = round(_dd_dist_sl, 2)
                            _dd_dist_tgt = round(_dd_dist_tgt, 2)

                            # If you enter NOW at LTP
                            _dd_sl_now  = _ddltp - _dd_last_sig * _dd_sl
                            _dd_tgt_now = _ddltp + _dd_last_sig * _dd_tgt
                            _dd_remaining_tgt = max(0, _dd_tgt - max(0, _dd_moved_pts))
                            _dd_tgt_pct_left  = round(_dd_remaining_tgt / max(_dd_tgt,0.01) * 100, 1)

                            # Status: has SL been hit? Target hit?
                            _dd_sl_hit  = _dd_dist_sl < 0
                            _dd_tgt_hit = _dd_dist_tgt < 0
                            if _dd_tgt_hit:
                                _dd_status = "🎯 TARGET HIT"
                                _dd_color  = "success"
                            elif _dd_sl_hit:
                                _dd_status = "🛑 SL HIT"
                                _dd_color  = "error"
                            elif _dd_moved_pts > _dd_tgt * 0.6:
                                _dd_status = "🔥 ALMOST TARGET"
                                _dd_color  = "success"
                            elif _dd_moved_pts > 0:
                                _dd_status = "✅ IN PROFIT"
                                _dd_color  = "success"
                            elif _dd_moved_pts > -_dd_sl * 0.5:
                                _dd_status = "⚠️ SMALL PULLBACK"
                                _dd_color  = "warning"
                            else:
                                _dd_status = "❌ NEAR SL"
                                _dd_color  = "error"

                            # Entry advice
                            if _dd_tgt_hit or _dd_sl_hit:
                                _dd_advice = "SKIP — trade already closed"
                            elif _dd_tgt_pct_left < 20:
                                _dd_advice = "SKIP — only {:.0f}% of target left".format(_dd_tgt_pct_left)
                            elif _dd_tgt_pct_left >= 70:
                                _dd_advice = "ENTER — most of target still available"
                            else:
                                _dd_advice = "CONSIDER — {:.0f}% target remaining".format(_dd_tgt_pct_left)

                            _dd_results.append({
                                "_iv":          _ddiv,
                                "_pd":          _ddpd,
                                "_sig":         _dd_last_sig,
                                "_moved":       _dd_moved_pts,
                                "_tgt_pct":     _dd_tgt_pct_left,
                                "_color":       _dd_color,
                                "_sl_hit":      _dd_sl_hit,
                                "_tgt_hit":     _dd_tgt_hit,
                                "Timeframe":    _ddiv,
                                "Period":       _ddpd,
                                "Direction":    "🟢 LONG" if _dd_last_sig==1 else "🔴 SHORT",
                                "Status":       _dd_status,
                                "Signal Time":  to_ist(_dd_sig_dt),
                                "Age":          _dd_age_str,
                                "Age (min)":    round(_dd_age_mins,1),
                                "Signal Price": round(_dd_sig_px, 2),
                                "SL at Signal": round(_dd_sl_at_sig, 2),
                                "Tgt at Signal":round(_dd_tgt_at_sig, 2),
                                "Current LTP":  round(_ddltp, 2),
                                "Moved (pts)":  _dd_moved_pts,
                                "Dist to SL":   _dd_dist_sl,
                                "Dist to Tgt":  _dd_dist_tgt,
                                "Bars Ago":     _dd_bars_ago,
                                "SL if Enter":  round(_dd_sl_now, 2),
                                "Tgt if Enter": round(_dd_tgt_now, 2),
                                "Tgt % Left":   _dd_tgt_pct_left,
                                "Remaining Tgt":round(_dd_remaining_tgt, 2),
                                "Advice":       _dd_advice,
                            })

                        elif _dd_show_approaching and _dd_last_sig == 0:
                            # Check EW approach progress
                            if "Elliott Wave" in _dd_strategy:
                                try:
                                    _dd_ewd = _ew_diagnostics(_dddf,
                                        min_wave_pct=float(_dd_mwp))
                                    _pf = min(100, int(_dd_ewd["pivot_flip_pct"]))
                                    if _pf >= 40:
                                        _dd_approaching.append({
                                            "_iv": _ddiv, "_pd": _ddpd,
                                            "Timeframe": _ddiv, "Period": _ddpd,
                                            "Progress": f"{_pf}%",
                                            "Next Pivot": "LOW ↓" if _dd_ewd["last_confirmed_dir"]==1 else "HIGH ↑",
                                            "Last Confirmed Pivot": f"{_dd_ewd['last_confirmed_px']:.2f}",
                                            "LTP": round(_ddltp,2),
                                            "Move Needed": f"{_dd_ewd['pct_remaining']:.2f}%",
                                        })
                                except: pass

                    except: pass

            _dd_prog.empty(); _dd_stat.empty()

            # Sort: active trades first (not sl/tgt hit), then by target % left desc
            _dd_results.sort(key=lambda r: (
                1 if (r["_sl_hit"] or r["_tgt_hit"]) else 0,
                -r["_tgt_pct"]
            ))
            st.session_state["dd_results"]   = _dd_results
            st.session_state["dd_approach"]  = _dd_approaching
            st.session_state["dd_meta"]      = {
                "sym":_dd_sym,"strategy":_dd_strategy,"sl":_dd_sl,"tgt":_dd_tgt,"mwp":_dd_mwp
            }

        # ── Display ────────────────────────────────────────────────────────────
        _ddr  = st.session_state.get("dd_results")
        _dda  = st.session_state.get("dd_approach",[])
        _ddm  = st.session_state.get("dd_meta",{})

        if _ddr is not None:
            _ddm_sym  = _ddm.get("sym","")
            _ddm_strat= _ddm.get("strategy","")
            _ddm_sl   = _ddm.get("sl",10)
            _ddm_tgt  = _ddm.get("tgt",20)

            # Summary
            _active  = [r for r in _ddr if not r["_sl_hit"] and not r["_tgt_hit"]]
            _sl_hit  = [r for r in _ddr if r["_sl_hit"]]
            _tgt_hit = [r for r in _ddr if r["_tgt_hit"]]

            st.markdown(f"### 📊 Scan results for **{_ddm_sym}** using **{_ddm_strat}**")
            _s1,_s2,_s3,_s4,_s5 = st.columns(5)
            _s1.metric("Total Fired",   len(_ddr))
            _s2.metric("✅ Active",      len(_active))
            _s3.metric("🎯 Target Hit", len(_tgt_hit))
            _s4.metric("🛑 SL Hit",     len(_sl_hit))
            _s5.metric("⚡ Approaching", len(_dda))

            if not _ddr:
                st.info("No signals found. Try increasing the lookback slider or choosing a different strategy.")
            else:
                # ── Active signals (most valuable) ──────────────────────────────
                st.markdown("---")
                st.markdown("### 🟢🔴 Active Signals — Trade Still Running")
                if _active:
                    st.caption("Signals where neither SL nor Target has been hit yet. "
                               "Green advice = still worth entering. Red = skip.")
                    for _ri, _row in enumerate(_active):
                        _clr = _row["_color"]
                        # Header row
                        _hd = st.columns([0.8,0.8,1,1,1,1,1,1,1,2])
                        _hd[0].markdown(f"**{_row['Timeframe']}**")
                        _hd[1].markdown(f"_{_row['Period']}_")
                        _hd[2].markdown(f"{_row['Direction']}")
                        _hd[3].markdown(f"{_row['Status']}")
                        _hd[4].markdown(f"Age: **{_row['Age']}**")
                        _hd[5].markdown(f"Moved: **{_row['Moved (pts)']:+.1f}pts**")
                        _hd[6].markdown(f"Tgt left: **{_row['Tgt % Left']:.0f}%**")
                        _hd[7].markdown(f"LTP: {_row['Current LTP']}")
                        _hd[8].markdown(f"Dist SL: {_row['Dist to SL']:.1f}pt")
                        _ddsd = "BUY" if _row["_sig"]==1 else "SELL"
                        if _hd[9].button(
                            f"{'🟢' if _ddsd=='BUY' else '🔴'} Apply → Live",
                            key=f"dd_apply_{_ri}_{_row['Timeframe']}_{_row['Period']}",
                            type="primary" if _row['_tgt_pct']>=50 else "secondary"
                        ):
                            _nte_apply_to_live(
                                ticker_sym  =_ddm_sym, ticker_label=_ddm_sym,
                                sig_direction=_ddsd,
                                iv=_row["Timeframe"], pd_val=_row["Period"],
                                strat=_ddm_strat,
                            )
                            st.rerun()

                        # Detail expander
                        with st.expander(
                            f"📋 {_row['Timeframe']}/{_row['Period']} — Full Details",
                            expanded=(_ri==0 and _row['_tgt_pct']>=50)
                        ):
                            _dc = st.columns(4)
                            _dc[0].markdown(f"**Signal Fired**")
                            _dc[0].markdown(f"Time: `{_row['Signal Time']}`")
                            _dc[0].markdown(f"Price: `{_row['Signal Price']}`")
                            _dc[0].markdown(f"Bars ago: `{_row['Bars Ago']}`")

                            _dc[1].markdown(f"**At Signal**")
                            _dc[1].markdown(f"SL: `{_row['SL at Signal']}`")
                            _dc[1].markdown(f"Target: `{_row['Tgt at Signal']}`")
                            _dc[1].markdown(f"R:R: `1:{_ddm_tgt/_ddm_sl:.1f}`")

                            _dc[2].markdown(f"**Right Now**")
                            _dc[2].markdown(f"LTP: `{_row['Current LTP']}`")
                            _dc[2].markdown(f"Dist to SL: `{_row['Dist to SL']:.2f}pts`")
                            _dc[2].markdown(f"Dist to Tgt: `{_row['Dist to Tgt']:.2f}pts`")
                            _dc[2].markdown(f"Tgt % Left: `{_row['Tgt % Left']:.1f}%`")

                            _dc[3].markdown(f"**If You Enter Now**")
                            _dc[3].markdown(f"Entry: `{_row['Current LTP']}`")
                            _dc[3].markdown(f"SL: `{_row['SL if Enter']}`")
                            _dc[3].markdown(f"Tgt: `{_row['Tgt if Enter']}`")
                            _dc[3].markdown(f"Remaining Tgt: `{_row['Remaining Tgt']:.1f}pts`")

                            # Advice box
                            _adv = _row["Advice"]
                            if "ENTER" in _adv:
                                st.success(
                                    f"✅ **{_adv}**\n\n"
                                    f"Signal fired **{_row['Age']} ago** at `{_row['Signal Price']}`. "
                                    f"Price moved **{_row['Moved (pts)']:+.1f} pts** in signal direction — "
                                    f"**{_row['Tgt % Left']:.0f}% of target ({_row['Remaining Tgt']:.1f}pts) still available**.\n\n"
                                    f"Enter at `{_row['Current LTP']}` | "
                                    f"SL: `{_row['SL if Enter']}` | "
                                    f"Target: `{_row['Tgt if Enter']}` | "
                                    f"Remaining R:R ≈ 1:{_row['Remaining Tgt']/_ddm_sl:.1f}"
                                )
                            elif "CONSIDER" in _adv:
                                st.warning(
                                    f"⚠️ **{_adv}**\n\n"
                                    f"Signal fired **{_row['Age']} ago**. Only **{_row['Tgt % Left']:.0f}%** "
                                    f"of target left ({_row['Remaining Tgt']:.1f}pts). "
                                    f"Enter with **50% position size** only. "
                                    f"SL: `{_row['SL if Enter']}` | Target: `{_row['Tgt if Enter']}`"
                                )
                            else:
                                st.error(
                                    f"🔴 **{_adv}**\n\n"
                                    f"Less than 20% of target remaining or trade effectively over. "
                                    f"Wait for the next fresh signal on this combo."
                                )
                else:
                    st.info("No active (open) signals found — all signals on this ticker have either hit SL, hit Target, or are too stale.")

                # ── Completed trades (SL/Target hit) ────────────────────────────
                if _tgt_hit or _sl_hit:
                    with st.expander(f"📜 Completed Signals (🎯 {len(_tgt_hit)} target hit | 🛑 {len(_sl_hit)} SL hit)"):
                        _completed = sorted(_tgt_hit + _sl_hit,
                                            key=lambda r: r["Age (min)"])
                        _cpdf = pd.DataFrame([
                            {k:v for k,v in r.items() if not k.startswith("_")}
                            for r in _completed
                        ])
                        def _cp_color(v):
                            if "TARGET" in str(v) or "PROFIT" in str(v): return "color:#2e7d32;font-weight:bold"
                            if "SL" in str(v) or "NEAR" in str(v): return "color:#c62828;font-weight:bold"
                            return ""
                        st.dataframe(_cpdf.style.map(_cp_color, subset=["Status"]),
                                     use_container_width=True,
                                     column_config={
                                         "Signal Time": st.column_config.TextColumn("Signal Time (IST)", width="large"),
                                     })

            # ── Approaching signals ─────────────────────────────────────────────
            if _dda:
                st.markdown("---")
                st.markdown("### ⚡ Approaching Signals (Elliott Wave progress ≥ 40%)")
                st.caption("These combos haven't fired yet but are building toward a signal. "
                           "Apply one to Live Trading and monitor — signal may fire soon.")
                for _dai, _darow in enumerate(_dda):
                    _da_cols = st.columns([1,1,1,1.5,1.5,1.5,2])
                    _da_cols[0].markdown(f"**{_darow['Timeframe']}**")
                    _da_cols[1].markdown(f"_{_darow['Period']}_")
                    _da_cols[2].markdown(f"Progress: **{_darow['Progress']}**")
                    _da_cols[3].markdown(f"Next: {_darow['Next Pivot']}")
                    _da_cols[4].markdown(f"Last pivot: {_darow['Last Confirmed Pivot']}")
                    _da_cols[5].markdown(f"Need: {_darow['Move Needed']} more")
                    if _da_cols[6].button(
                        f"📡 Apply → Live+BT",
                        key=f"dd_appr_{_dai}_{_darow['Timeframe']}_{_darow['Period']}",
                        type="secondary"
                    ):
                        _appr_dir = "BUY" if _darow["Next Pivot"] == "LOW ↓" else "SELL"
                        # Build opt_applied dict so pre-populate block handles it correctly
                        _appr_strat = _ddm_strat
                        _appr_sym   = _ddm_sym
                        _appr_iv    = _darow["Timeframe"]
                        _appr_pd    = _darow["Period"]
                        if _appr_sym in TICKER_MAP.values():
                            _appr_tk = [k for k,v in TICKER_MAP.items() if v==_appr_sym][0]
                            _appr_custom = ""
                        else:
                            _appr_tk = "Custom"; _appr_custom = _appr_sym
                        # Build full params including mwp override
                        _appr_params = {**_STRATEGY_DEFAULTS.get(_appr_strat,{}), **_BP}
                        if "Elliott Wave" in _appr_strat:
                            _appr_params["min_wave_pct"] = _ddm.get("mwp", 0.5)
                        st.session_state["opt_applied"] = {
                            "strategy":   _appr_strat,
                            "instrument": _appr_tk,
                            "custom_sym": _appr_custom,
                            "interval":   _appr_iv,
                            "period":     _appr_pd,
                            "sl_type":    SL_TYPES[0],
                            "sl_pts":     _ddm.get("sl", 10.0),
                            "tgt_type":   TARGET_TYPES[0],
                            "tgt_pts":    _ddm.get("tgt", 20.0),
                            "params":     _appr_params,
                            "accuracy":   "—",
                            "pnl":        "—",
                        }
                        st.session_state["_oa_hash_prev"] = ""
                        st.session_state["nte_applied_msg"] = (
                            f"✅ Applied {_appr_iv}/{_appr_pd} {_appr_strat} → "
                            f"Switch to ⚡ Live Trading and start monitoring. "
                            f"Signal is {_darow['Progress']} of the way to firing."
                        )
                        st.rerun()

        else:
            st.info(
                "Select a ticker and strategy above, then click **🔎 Deep Scan**.\n\n"
                "The scanner will check every timeframe/period combination and show you:\n"
                "- Every signal that fired recently and whether it's still worth entering\n"
                "- How much the price has already moved and how much target is left\n"
                "- Exact entry/SL/target levels if you enter right now\n"
                "- One-click Apply to configure sidebar and live trading"
            )
