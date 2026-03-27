"""
╔══════════════════════════════════════════════════════════════╗
║       🌊  ELLIOTT WAVE ALGO TRADER  v5.3                     ║
║  Full Auto Mode · Multi-Factor Accuracy · Dhan Integration  ║
╚══════════════════════════════════════════════════════════════╝
pip install streamlit>=1.37 yfinance plotly pandas numpy dhanhq requests
streamlit run elliott_wave_algo_trader.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ▼▼▼  CHANGE YOUR CREDENTIALS HERE  ▼▼▼
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_CLIENT_ID    = "1104779876"
DEFAULT_ACCESS_TOKEN = ("eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9"
                        ".eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzc0NjY3MTA4"
                        "LCJpYXQiOjE3NzQ1ODA3MDgsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIs"
                        "IndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTA0Nzc5ODc2In0"
                        ".tkaJSjQTuku8cS_lDrI6Y__7grZv6lsA_Sc4BGuRA_T4yMlj_hCNtXQRYB4"
                        "g3uMVva6z66nYDgpy6z6nibBo8Q")
# ▲▲▲  END CREDENTIALS  ▲▲▲

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import threading
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta
from typing import Optional
import itertools
import warnings

warnings.filterwarnings("ignore")

try:
    from dhanhq import dhanhq as DhanHQLib
    DHAN_LIB_OK = True
except ImportError:
    DHAN_LIB_OK = False

IST = timezone(timedelta(hours=5, minutes=30))
def now_ist(): return datetime.now(IST)
def fmt_ist(dt=None):
    if dt is None: return now_ist().strftime("%H:%M:%S IST")
    try:
        if hasattr(dt,"to_pydatetime"): dt=dt.to_pydatetime()
        if dt.tzinfo is None: dt=dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(IST).strftime("%d-%b %H:%M:%S IST")
    except: return str(dt)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="🌊 EW Algo Trader",page_icon="🌊",
                   layout="wide",initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Exo 2',sans-serif;}
section[data-testid="stSidebar"]>div{min-width:320px;width:320px!important;}
section[data-testid="stSidebar"] *{white-space:normal!important;word-break:break-word!important;}
div[data-testid="metric-container"] *{white-space:normal!important;overflow:visible!important;text-overflow:clip!important;}
div[data-testid="stMetricValue"]{font-size:.95rem!important;}
div[data-testid="stMetricDelta"]{font-size:.76rem!important;}
.main-hdr{background:linear-gradient(135deg,#0a0e1a,#0d1b2a,#0a1628);
  border:1px solid #1e3a5f;border-radius:14px;padding:18px 24px;margin-bottom:10px;
  box-shadow:0 4px 24px rgba(0,229,255,.08);}
.main-hdr h1{font-family:'Exo 2',sans-serif;font-weight:700;
  background:linear-gradient(90deg,#00e5ff,#4dd0e1,#00bcd4);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;font-size:1.7rem;}
.main-hdr p{color:#546e7a;margin:4px 0 0;font-size:.82rem;}
.pos-none{background:#07111e;border:2px dashed #263238;border-radius:14px;padding:16px 20px;}
.pos-buy{background:rgba(76,175,80,.11);border:3px solid #4caf50;border-radius:14px;padding:16px 20px;}
.pos-sell{background:rgba(244,67,54,.11);border:3px solid #f44336;border-radius:14px;padding:16px 20px;}
.pos-signal{background:rgba(0,229,255,.08);border:2px solid #00e5ff;border-radius:14px;padding:16px 20px;}
.pos-reversing{background:rgba(171,71,188,.14);border:3px solid #ab47bc;border-radius:14px;padding:16px 20px;text-align:center;}
.auto-banner{background:linear-gradient(135deg,rgba(0,229,255,.10),rgba(76,175,80,.08));
  border:2px solid #00e5ff;border-radius:14px;padding:16px 20px;margin-bottom:12px;}
.cfg-card{background:#060d14;border:1px solid #0288d1;border-radius:10px;padding:12px 16px;margin:6px 0;}
.wave-card{background:#060d14;border:1px solid #1e3a5f;border-radius:8px;
  padding:10px 14px;font-family:'Share Tech Mono',monospace;font-size:.80rem;
  line-height:1.85;white-space:pre-wrap;word-break:break-word;}
.info-box{background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;
  padding:12px 14px;font-size:.84rem;line-height:1.85;word-break:break-word;}
.best-cfg{background:rgba(0,229,255,.07);border:1px solid #00bcd4;border-radius:10px;
  padding:12px 16px;margin:6px 0;word-break:break-word;}
.acc-bar{height:8px;border-radius:4px;background:linear-gradient(90deg,#f44336,#ffb300,#4caf50);}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:transparent;}
.stTabs [data-baseweb="tab"]{background:#0d1b2a;border-radius:8px;color:#546e7a;
  border:1px solid #1e3a5f;padding:5px 11px;font-size:.79rem;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#0d3349,#0a2540)!important;
  color:#00e5ff!important;border-color:#00bcd4!important;}
div[data-testid="metric-container"]{background:#0a1628;border:1px solid #1e3a5f;
  border-radius:8px;padding:8px 10px;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
TICKER_GROUPS={
    "🇮🇳 Indian Indices":{"Nifty 50":"^NSEI","Bank Nifty":"^NSEBANK","Sensex":"^BSESN",
        "Nifty IT":"^CNXIT","Nifty Midcap":"^NSEMDCP50","Nifty Auto":"^CNXAUTO"},
    "🇮🇳 NSE Stocks":{"Reliance":"RELIANCE.NS","TCS":"TCS.NS","HDFC Bank":"HDFCBANK.NS",
        "Infosys":"INFY.NS","ICICI Bank":"ICICIBANK.NS","SBI":"SBIN.NS",
        "Wipro":"WIPRO.NS","Axis Bank":"AXISBANK.NS","Maruti":"MARUTI.NS"},
    "₿ Crypto":{"Bitcoin":"BTC-USD","Ethereum":"ETH-USD","BNB":"BNB-USD","Solana":"SOL-USD"},
    "💱 Forex":{"USD/INR":"USDINR=X","EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X"},
    "🥇 Commodities":{"Gold":"GC=F","Silver":"SI=F","Crude Oil":"CL=F"},
    "🌐 US Stocks":{"Apple":"AAPL","Tesla":"TSLA","NVIDIA":"NVDA","Microsoft":"MSFT"},
    "✏️ Custom Ticker":{"Custom":"__CUSTOM__"},
}
TIMEFRAMES=["1m","5m","15m","30m","1h","4h","1d","1wk"]
PERIODS=["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"]
VALID_PERIODS={
    "1m":["1d","5d","7d"],"5m":["1d","5d","7d","1mo","3mo"],
    "15m":["1d","5d","7d","1mo","3mo"],"30m":["1d","5d","7d","1mo","3mo"],
    "1h":["1d","5d","7d","1mo","3mo","6mo","1y","2y"],
    "4h":["1mo","3mo","6mo","1y","2y","5y"],
    "1d":["1mo","3mo","6mo","1y","2y","5y","10y","20y"],
    "1wk":["3mo","6mo","1y","2y","5y","10y","20y"],
}
POLL_SECS={"1m":60,"5m":300,"15m":900,"30m":1800,"1h":3600,"4h":14400,"1d":86400,"1wk":604800}
MTF_COMBOS=[("1d","1y","Daily"),("4h","3mo","4-Hour"),("1h","1mo","1-Hour"),("15m","5d","15-Min")]

# Full-Auto scan grid (TF, period pairs)
AUTO_SCAN_GRID=[
    ("1d","2y"),("1d","1y"),("4h","6mo"),("4h","3mo"),
    ("1h","3mo"),("1h","1mo"),("15m","1mo"),("15m","5d"),
]

SL_WAVE="wave_auto"; SL_PTS="__sl_pts__"; SL_SIGREV="__sl_sigrev__"; SL_TRAIL="__sl_trail__"
TGT_PTS="__tgt_pts__"; TGT_SIGREV="__tgt_sigrev__"; TGT_TRAIL="__tgt_trail__"
SL_MAP={
    "Wave Auto (Pivot)":SL_WAVE,
    "0.5%":0.005,"1%":0.01,"1.5%":0.015,"2%":0.02,"2.5%":0.025,"3%":0.03,"5%":0.05,
    "Custom Points ▼":SL_PTS,"Trailing SL ▼":SL_TRAIL,"Exit on Signal Reverse":SL_SIGREV,
}
TGT_MAP={
    "Wave Auto (Fib 1.618×W1)":"wave_auto",
    "R:R 1:1":1.0,"R:R 1:1.5":1.5,"R:R 1:2":2.0,"R:R 1:2.5":2.5,"R:R 1:3":3.0,
    "Fib 1.618×Wave 1":"fib_1618","Fib 2.618×Wave 1":"fib_2618",
    "Custom Points ▼":TGT_PTS,"Trailing Target ▼ (display only)":TGT_TRAIL,
    "Exit on Signal Reverse":TGT_SIGREV,
}
SL_KEYS=list(SL_MAP.keys()); TGT_KEYS=list(TGT_MAP.keys())

# Minimum confidence threshold for firing a signal (quality gate)
MIN_CONFIDENCE_THRESHOLD = 0.68

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
_DEF={
    "live_running":False,"live_phase":"idle",
    "live_ltp":None,"live_ltp_prev":None,"live_ltp_ts":None,
    "live_last_candle_ist":"—","live_delay_s":0,"live_delay_ctx":"",
    "live_next_check_ist":"—","last_bar_ts":None,"last_fetch_wall":0.0,
    "live_position":None,"live_pnl":0.0,
    "live_signals":[],"live_trades":[],"live_log":[],
    "live_wave_state":None,"live_last_sig":None,
    "live_no_pos_reason":"Click ▶ Start Live. Everything runs automatically.",
    "_scan_sig":None,"_scan_df":None,
    "bt_results":None,"opt_results":None,
    "_analysis_results":None,"_analysis_overall":"HOLD","_analysis_symbol":"",
    "applied_depth":5,"applied_sl_lbl":"Wave Auto (Pivot)",
    "applied_tgt_lbl":"Wave Auto (Fib 1.618×W1)","best_cfg_applied":False,
    "custom_sl_pts":50.0,"custom_tgt_pts":100.0,
    "trailing_sl_pts":50.0,"trailing_tgt_pts":150.0,
    # Full Auto Mode
    "auto_mode":False,
    "auto_running":False,
    "auto_status":"",
    "auto_best_cfg":None,    # {depth, sl_type, tgt_type, tf, period, score, win_rate, ...}
    "auto_all_results":None, # list of all scan+opt results
    "auto_log":[],
}
for _k,_v in _DEF.items():
    if _k not in st.session_state: st.session_state[_k]=_v

# ═══════════════════════════════════════════════════════════════════════════
# DHAN CLIENT
# ═══════════════════════════════════════════════════════════════════════════
class DhanClient:
    def __init__(self,cid:str,tok:str):
        self.cid=cid; self.tok=tok; self._lib=None
        if DHAN_LIB_OK:
            try: self._lib=DhanHQLib(cid,tok)
            except: pass

    def place_order(self,sec_id:str,segment:str,txn:str,qty:int,
                    order_type:str="MARKET",price:float=0.0,
                    product_type:str="INTRADAY",validity:str="DAY")->dict:
        if self._lib:
            try:
                tc=self._lib.BUY if txn=="BUY" else self._lib.SELL
                ot=self._lib.MARKET if order_type=="MARKET" else self._lib.LIMIT
                pt={"INTRADAY":self._lib.INTRA,"DELIVERY":self._lib.CNC,
                    "MARGIN":self._lib.MARGIN}.get(product_type,self._lib.INTRA)
                seg_map={"NSE":"NSE_EQ","BSE":"BSE_EQ","NSE_FNO":"NSE_FNO","BSE_FNO":"BSE_FNO"}
                seg=getattr(self._lib,seg_map.get(segment,segment),segment)
                return self._lib.place_order(security_id=sec_id,exchange_segment=seg,
                    transaction_type=tc,quantity=qty,order_type=ot,
                    product_type=pt,price=float(price),validity=validity)
            except Exception as e: return{"error":str(e),"via":"dhanhq"}
        else:
            import requests
            try:
                r=requests.post("https://api.dhan.co/orders",timeout=10,
                    headers={"Content-Type":"application/json","access-token":self.tok},
                    json={"dhanClientId":self.cid,"transactionType":txn,
                          "exchangeSegment":segment,"productType":product_type,
                          "orderType":order_type,"validity":validity,
                          "securityId":sec_id,"quantity":qty,"price":price})
                return r.json()
            except Exception as e: return{"error":str(e),"via":"requests"}

    def fund_limit(self)->dict:
        if self._lib:
            try: return self._lib.get_fund_limits()
            except Exception as e: return{"error":str(e)}
        import requests
        try:
            r=requests.get("https://api.dhan.co/fundlimit",timeout=10,
                headers={"Content-Type":"application/json","access-token":self.tok})
            return r.json()
        except Exception as e: return{"error":str(e)}

# ═══════════════════════════════════════════════════════════════════════════
# FETCH
# ═══════════════════════════════════════════════════════════════════════════
_flock=threading.Lock(); _fts=[0.0]

def fetch_ohlcv(sym:str,iv:str,per:str,md:float=1.5)->Optional[pd.DataFrame]:
    with _flock:
        gap=time.time()-_fts[0]
        if gap<md: time.sleep(md-gap)
        try:
            df=yf.download(sym,interval=iv,period=per,progress=False,auto_adjust=True)
            _fts[0]=time.time()
        except: _fts[0]=time.time(); return None
    if df is None or df.empty: return None
    if isinstance(df.columns,pd.MultiIndex): df.columns=[c[0] for c in df.columns]
    df=df.dropna(subset=["Open","High","Low","Close"]); df.index=pd.to_datetime(df.index)
    return df

def fetch_ltp(sym:str)->Optional[float]:
    try:
        fi=yf.Ticker(sym).fast_info
        p=fi.get("lastPrice") or fi.get("regularMarketPrice")
        if p and float(p)>0: return float(p)
    except: pass
    return None

def delay_ctx(delay_s:int,interval:str)->tuple:
    cs=POLL_SECS.get(interval,3600); ratio=delay_s/cs if cs>0 else 0
    if delay_s<120: return "🟢 Fresh","#4caf50",f"Data is {delay_s}s old — fresh."
    elif ratio<1.0: return f"🟡 {delay_s}s","#ffb300",f"Current candle still forming ({int(ratio*100)}% elapsed). Normal."
    elif ratio<2.0: return f"🟡 {delay_s//60}min","#ff9800",f"~{delay_s//60}min old. Market may be between sessions. Using last closed bar."
    else:
        h=delay_s//3600
        return f"🔴 {h}h+","#f44336",f"{h}h old. Market closed/holiday. Signal valid for next session open."

# ═══════════════════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════════════════
def add_indicators(df:pd.DataFrame)->pd.DataFrame:
    df=df.copy(); c=df["Close"]
    df["EMA_9"]=c.ewm(span=9,adjust=False).mean()
    df["EMA_21"]=c.ewm(span=21,adjust=False).mean()
    df["EMA_50"]=c.ewm(span=50,adjust=False).mean()
    df["EMA_200"]=c.ewm(span=200,adjust=False).mean()
    d=c.diff(); g=d.clip(lower=0).rolling(14).mean(); l=(-d.clip(upper=0)).rolling(14).mean()
    df["RSI"]=100-(100/(1+g/l.replace(0,np.nan)))
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    df["MACD"]=e12-e26; df["MACD_Signal"]=df["MACD"].ewm(span=9,adjust=False).mean()
    df["MACD_Hist"]=df["MACD"]-df["MACD_Signal"]
    # ATR
    hl=df["High"]-df["Low"]
    hc=(df["High"]-c.shift()).abs(); lc=(df["Low"]-c.shift()).abs()
    df["ATR"]=pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(14).mean()
    # Stoch RSI
    rsi=df["RSI"]; rsi_min=rsi.rolling(14).min(); rsi_max=rsi.rolling(14).max()
    df["StochRSI"]=(rsi-rsi_min)/(rsi_max-rsi_min+1e-10)
    # Volume MA
    if "Volume" in df.columns:
        df["Vol_MA"]=df["Volume"].rolling(20).mean()
        df["Vol_Ratio"]=df["Volume"]/df["Vol_MA"].replace(0,1)
    else:
        df["Vol_MA"]=0; df["Vol_Ratio"]=1
    # Bollinger Bands
    ma20=c.rolling(20).mean(); std20=c.rolling(20).std()
    df["BB_Upper"]=ma20+2*std20; df["BB_Lower"]=ma20-2*std20
    df["BB_Width"]=(df["BB_Upper"]-df["BB_Lower"])/(ma20+1e-10)
    return df

# ═══════════════════════════════════════════════════════════════════════════
# PIVOTS
# ═══════════════════════════════════════════════════════════════════════════
def find_pivots(df:pd.DataFrame,depth:int=5)->list:
    H,L,n=df["High"].values,df["Low"].values,len(df)
    raw=[]
    for i in range(depth,n-depth):
        wh=H[max(0,i-depth):i+depth+1]; wl=L[max(0,i-depth):i+depth+1]
        if H[i]==wh.max(): raw.append((i,float(H[i]),"H"))
        elif L[i]==wl.min(): raw.append((i,float(L[i]),"L"))
    clean=[]
    for p in raw:
        if not clean or clean[-1][2]!=p[2]: clean.append(list(p))
        else:
            if p[2]=="H" and p[1]>clean[-1][1]: clean[-1]=list(p)
            elif p[2]=="L" and p[1]<clean[-1][1]: clean[-1]=list(p)
    return [tuple(x) for x in clean]

# ═══════════════════════════════════════════════════════════════════════════
# MULTI-FACTOR SIGNAL QUALITY SCORER
# ═══════════════════════════════════════════════════════════════════════════
def score_signal_factors(df_ind:pd.DataFrame, signal_dir:str, retr:float, w1:float) -> dict:
    """
    Returns {score: float 0-1, factors: dict, details: str}
    Combines 8 factors for high-accuracy signal filtering.
    """
    factors={}; score=0.0; n=len(df_ind)
    if n<30: return{"score":0.0,"factors":{},"details":"Insufficient data"}

    cur  = float(df_ind["Close"].iloc[-1])
    rsi  = float(df_ind["RSI"].iloc[-1]) if not df_ind["RSI"].isna().iloc[-1] else 50
    srsi = float(df_ind["StochRSI"].iloc[-1]) if not df_ind["StochRSI"].isna().iloc[-1] else 0.5
    macd_h = float(df_ind["MACD_Hist"].iloc[-1]) if not df_ind["MACD_Hist"].isna().iloc[-1] else 0
    ema9  = float(df_ind["EMA_9"].iloc[-1]) if not df_ind["EMA_9"].isna().iloc[-1] else cur
    ema21 = float(df_ind["EMA_21"].iloc[-1]) if not df_ind["EMA_21"].isna().iloc[-1] else cur
    ema50 = float(df_ind["EMA_50"].iloc[-1]) if not df_ind["EMA_50"].isna().iloc[-1] else cur
    ema200= float(df_ind["EMA_200"].iloc[-1]) if not df_ind["EMA_200"].isna().iloc[-1] else cur
    atr   = float(df_ind["ATR"].iloc[-1]) if not df_ind["ATR"].isna().iloc[-1] else cur*0.01
    vol_r = float(df_ind["Vol_Ratio"].iloc[-1]) if not df_ind["Vol_Ratio"].isna().iloc[-1] else 1.0
    bb_low= float(df_ind["BB_Lower"].iloc[-1]) if not df_ind["BB_Lower"].isna().iloc[-1] else cur*0.98
    bb_hi = float(df_ind["BB_Upper"].iloc[-1]) if not df_ind["BB_Upper"].isna().iloc[-1] else cur*1.02

    # MACD previous bars trend
    macd_prev = [float(df_ind["MACD_Hist"].iloc[max(0,n-1-j)]) for j in range(1,4)]
    macd_turning = any(h for h in macd_prev)

    if signal_dir=="BUY":
        # F1: RSI oversold zone (ideal 30-50 for W2 bottom, penalty if overbought)
        if rsi<=35:         f1=1.0
        elif rsi<=45:       f1=0.85
        elif rsi<=55:       f1=0.65
        elif rsi<=65:       f1=0.40
        else:               f1=0.10  # overbought on BUY = bad
        factors["RSI"]=f"{f'{rsi:.0f}'} ({'oversold' if rsi<45 else 'neutral' if rsi<60 else 'overbought'})"; score+=f1*0.18

        # F2: StochRSI oversold
        if srsi<=0.20:      f2=1.0
        elif srsi<=0.40:    f2=0.75
        elif srsi<=0.60:    f2=0.45
        else:               f2=0.20
        factors["StochRSI"]=f"{srsi:.2f} ({'oversold' if srsi<0.3 else 'neutral'})"; score+=f2*0.12

        # F3: MACD histogram turning up or positive
        if macd_h>0 and any(h<0 for h in macd_prev[:2]):  f3=1.0  # just crossed positive
        elif macd_h>0:                                      f3=0.80
        elif macd_h>macd_prev[0] if macd_prev else False:  f3=0.65  # rising
        else:                                               f3=0.25
        factors["MACD"]=f"hist={macd_h:.2f} ({'bullish cross' if f3==1.0 else 'bullish' if macd_h>0 else 'bearish'})"; score+=f3*0.15

        # F4: EMA trend alignment (price above key EMAs = uptrend)
        if cur>ema9>ema21>ema50:   f4=1.0
        elif cur>ema21>ema50:      f4=0.80
        elif cur>ema50:            f4=0.60
        elif cur>ema200:           f4=0.40
        else:                      f4=0.15
        factors["EMA Trend"]=f"{'Strong up' if f4>0.9 else 'Up' if f4>0.6 else 'Weak/down'}"; score+=f4*0.15

        # F5: Fibonacci retracement quality (61.8% golden = best for W2)
        if abs(retr-0.618)<0.04:   f5=1.0
        elif abs(retr-0.500)<0.04: f5=0.88
        elif abs(retr-0.382)<0.04: f5=0.80
        elif 0.50<=retr<=0.786:    f5=0.72
        elif 0.382<=retr<=0.886:   f5=0.55
        else:                      f5=0.30
        factors["Fib Retrace"]=f"{retr:.1%} ({'Golden' if f5>=1.0 else '50%' if f5>=0.88 else '38.2%' if f5>=0.80 else 'Valid' if f5>=0.55 else 'Marginal'})"; score+=f5*0.18

        # F6: BB position (near lower band = oversold area)
        if cur<=bb_low*1.005:   f6=1.0
        elif cur<=bb_low*1.02:  f6=0.75
        elif cur<=(bb_low+bb_hi)/2: f6=0.50
        else:                   f6=0.20
        factors["BB Position"]=f"{'Near lower' if f6>0.7 else 'Mid' if f6>0.4 else 'Upper zone'}"; score+=f6*0.10

        # F7: Volume surge (high volume confirms pivot)
        if vol_r>=1.5:    f7=1.0
        elif vol_r>=1.2:  f7=0.80
        elif vol_r>=0.8:  f7=0.60
        else:             f7=0.40
        factors["Volume"]=f"{vol_r:.1f}× avg ({'surge' if vol_r>=1.5 else 'normal'})"; score+=f7*0.07

        # F8: Wave 3 projection viability (W1 length vs ATR)
        if atr>0:
            w1_atr_ratio=w1/atr
            if w1_atr_ratio>=3.0:   f8=1.0
            elif w1_atr_ratio>=2.0: f8=0.80
            elif w1_atr_ratio>=1.0: f8=0.60
            else:                   f8=0.35
        else: f8=0.50
        factors["W1/ATR"]=f"{w1/atr:.1f}× ({'strong' if f8>0.8 else 'ok'})"; score+=f8*0.05

    else:  # SELL
        if rsi>=65:         f1=1.0
        elif rsi>=55:       f1=0.85
        elif rsi>=45:       f1=0.65
        elif rsi>=35:       f1=0.40
        else:               f1=0.10
        factors["RSI"]=f"{rsi:.0f} ({'overbought' if rsi>65 else 'neutral' if rsi>45 else 'oversold'})"; score+=f1*0.18

        if srsi>=0.80:      f2=1.0
        elif srsi>=0.60:    f2=0.75
        elif srsi>=0.40:    f2=0.45
        else:               f2=0.20
        factors["StochRSI"]=f"{srsi:.2f} ({'overbought' if srsi>0.7 else 'neutral'})"; score+=f2*0.12

        if macd_h<0 and any(h>0 for h in macd_prev[:2]):  f3=1.0
        elif macd_h<0:                                      f3=0.80
        elif macd_h<macd_prev[0] if macd_prev else False:  f3=0.65
        else:                                               f3=0.25
        factors["MACD"]=f"hist={macd_h:.2f} ({'bearish cross' if f3==1.0 else 'bearish' if macd_h<0 else 'bullish'})"; score+=f3*0.15

        if cur<ema9<ema21<ema50:   f4=1.0
        elif cur<ema21<ema50:      f4=0.80
        elif cur<ema50:            f4=0.60
        elif cur<ema200:           f4=0.40
        else:                      f4=0.15
        factors["EMA Trend"]=f"{'Strong down' if f4>0.9 else 'Down' if f4>0.6 else 'Weak/up'}"; score+=f4*0.15

        if abs(retr-0.618)<0.04:   f5=1.0
        elif abs(retr-0.500)<0.04: f5=0.88
        elif abs(retr-0.382)<0.04: f5=0.80
        elif 0.50<=retr<=0.786:    f5=0.72
        elif 0.382<=retr<=0.886:   f5=0.55
        else:                      f5=0.30
        factors["Fib Retrace"]=f"{retr:.1%}"; score+=f5*0.18

        if cur>=bb_hi*0.995:    f6=1.0
        elif cur>=bb_hi*0.98:   f6=0.75
        elif cur>=(bb_low+bb_hi)/2: f6=0.50
        else:                   f6=0.20
        factors["BB Position"]=f"{'Near upper' if f6>0.7 else 'Mid' if f6>0.4 else 'Lower zone'}"; score+=f6*0.10

        if vol_r>=1.5:    f7=1.0
        elif vol_r>=1.2:  f7=0.80
        elif vol_r>=0.8:  f7=0.60
        else:             f7=0.40
        factors["Volume"]=f"{vol_r:.1f}×"; score+=f7*0.07

        f8=0.8 if w1/atr>=2.0 else 0.5 if w1/atr>=1.0 else 0.3 if atr>0 else 0.5
        factors["W1/ATR"]=f"{w1/atr:.1f}×" if atr>0 else "—"; score+=f8*0.05

    details="; ".join(f"{k}: {v}" for k,v in factors.items())
    return{"score":min(score,1.0),"factors":factors,"details":details}

# ═══════════════════════════════════════════════════════════════════════════
# CORE SIGNAL  (multi-factor, high accuracy)
# ═══════════════════════════════════════════════════════════════════════════
def _blank(reason:str="")->dict:
    return{"signal":"HOLD","entry_price":None,"sl":None,"target":None,
           "confidence":0.0,"mf_score":0.0,"reason":reason or "No pattern",
           "pattern":"—","wave_pivots":None,"wave1_len":0.0,"retracement":0.0,
           "factors":{}}

def _calc_target(tt,entry:float,direction:str,w1:float,risk:float,
                 ctp:float=100.0,atr:float=0.0)->float:
    s=1 if direction=="BUY" else -1
    if tt in("wave_auto","fib_1618"): return entry+s*w1*1.618
    if tt=="fib_2618": return entry+s*w1*2.618
    if tt==TGT_PTS:    return entry+s*ctp
    if tt==TGT_TRAIL:  return entry+s*ctp
    if tt==TGT_SIGREV: return entry+s*w1*1.618
    if isinstance(tt,(int,float)): return entry+s*risk*float(tt)
    return entry+s*risk*2.0

def ew_signal(df:pd.DataFrame,depth:int=5,sl_type="wave_auto",tgt_type="wave_auto",
              csl:float=50.0,ctgt:float=100.0,
              min_conf:float=MIN_CONFIDENCE_THRESHOLD)->dict:
    """
    Multi-factor Elliott Wave signal with 8-indicator confluence scoring.
    Only fires when combined score >= min_conf threshold.
    """
    n=len(df)
    if n<max(40,depth*5): return _blank("Need more bars")
    pivots=find_pivots(df,depth)
    if len(pivots)<4: return _blank("Not enough pivots — try smaller depth")

    df_ind=add_indicators(df)
    cur=float(df["Close"].iloc[-1])
    atr=float(df_ind["ATR"].iloc[-1]) if not df_ind["ATR"].isna().iloc[-1] else cur*0.01
    best,best_score=_blank(),0.0

    for i in range(len(pivots)-2):
        p0,p1,p2=pivots[i],pivots[i+1],pivots[i+2]; bs=n-1-p2[0]

        # ── BUY: Low→High→Low ────────────────────────────────────────────
        if p0[2]=="L" and p1[2]=="H" and p2[2]=="L":
            w1=p1[1]-p0[1]
            if w1<=0: continue
            r=(p1[1]-p2[1])/w1
            # Elliott rules: W2 must not exceed W0, retrace 23.6–88.6%
            if not(0.236<=r<=0.886 and p2[1]>p0[1] and bs<=depth*5): continue
            # Wave 3 must be longer than Wave 1 (core Elliott rule)
            if i+3<len(pivots) and pivots[i+3][2]=="H":
                w3=pivots[i+3][1]-p2[1]
                if w3<w1*0.90: continue  # W3 shorter than W1 → invalid pattern

            # Base EW confidence
            ew_conf=0.55
            if abs(r-0.618)<0.04: ew_conf=0.85
            elif abs(r-0.500)<0.04: ew_conf=0.75
            elif 0.50<=r<=0.786: ew_conf=0.70
            elif 0.382<=r<=0.618: ew_conf=0.65
            if i+3<len(pivots) and pivots[i+3][2]=="H":
                w3=pivots[i+3][1]-p2[1]
                if w3>w1: ew_conf=min(ew_conf+0.08,0.95)
                if abs(w3/w1-1.618)<0.15: ew_conf=min(ew_conf+0.06,0.98)

            # Multi-factor scorer
            mf=score_signal_factors(df_ind,"BUY",r,w1)
            mf_score=mf["score"]

            # Combined confidence: 55% EW + 45% multi-factor
            combined=ew_conf*0.55+mf_score*0.45

            # Minimum R:R enforcement
            e=cur
            if sl_type in(SL_WAVE,SL_TRAIL): sl_=p2[1]*0.998
            elif sl_type==SL_PTS: sl_=e-csl
            elif sl_type==SL_SIGREV: sl_=e*(1-0.05)
            else: sl_=e*(1-float(sl_type))
            risk=e-sl_
            if risk<=0: continue
            tgt_=_calc_target(tgt_type,e,"BUY",w1,risk,ctgt,atr)
            if tgt_<=e: continue
            rr=(tgt_-e)/risk
            if rr<1.2: continue  # enforce minimum 1.2:1 R:R

            if combined<=best_score: continue
            best_score=combined
            best={"signal":"BUY","entry_price":e,"sl":sl_,"target":tgt_,
                  "confidence":round(combined,3),"mf_score":round(mf_score,3),
                  "ew_conf":round(ew_conf,3),"retracement":r,
                  "reason":f"W2 bottom {r:.1%} retrace | Score {combined:.0%} | R:R 1:{rr:.1f}",
                  "pattern":f"W2 Bottom ({r:.1%})","wave_pivots":[p0,p1,p2],
                  "wave1_len":w1,"factors":mf["factors"],"rr":round(rr,2)}

        # ── SELL: High→Low→High ───────────────────────────────────────────
        elif p0[2]=="H" and p1[2]=="L" and p2[2]=="H":
            w1=p0[1]-p1[1]
            if w1<=0: continue
            r=(p2[1]-p1[1])/w1
            if not(0.236<=r<=0.886 and p2[1]<p0[1] and bs<=depth*5): continue
            if i+3<len(pivots) and pivots[i+3][2]=="L":
                w3=p2[1]-pivots[i+3][1]
                if w3<w1*0.90: continue

            ew_conf=0.55
            if abs(r-0.618)<0.04: ew_conf=0.85
            elif abs(r-0.500)<0.04: ew_conf=0.75
            elif 0.50<=r<=0.786: ew_conf=0.70
            elif 0.382<=r<=0.618: ew_conf=0.65
            if i+3<len(pivots) and pivots[i+3][2]=="L":
                w3=p2[1]-pivots[i+3][1]
                if w3>w1: ew_conf=min(ew_conf+0.08,0.95)
                if abs(w3/w1-1.618)<0.15: ew_conf=min(ew_conf+0.06,0.98)

            mf=score_signal_factors(df_ind,"SELL",r,w1)
            mf_score=mf["score"]; combined=ew_conf*0.55+mf_score*0.45

            e=cur
            if sl_type in(SL_WAVE,SL_TRAIL): sl_=p2[1]*1.002
            elif sl_type==SL_PTS: sl_=e+csl
            elif sl_type==SL_SIGREV: sl_=e*(1+0.05)
            else: sl_=e*(1+float(sl_type))
            risk=sl_-e
            if risk<=0: continue
            tgt_=_calc_target(tgt_type,e,"SELL",w1,risk,ctgt,atr)
            if tgt_>=e: continue
            rr=(e-tgt_)/risk
            if rr<1.2: continue

            if combined<=best_score: continue
            best_score=combined
            best={"signal":"SELL","entry_price":e,"sl":sl_,"target":tgt_,
                  "confidence":round(combined,3),"mf_score":round(mf_score,3),
                  "ew_conf":round(ew_conf,3),"retracement":r,
                  "reason":f"W2 top {r:.1%} retrace | Score {combined:.0%} | R:R 1:{rr:.1f}",
                  "pattern":f"W2 Top ({r:.1%})","wave_pivots":[p0,p1,p2],
                  "wave1_len":w1,"factors":mf["factors"],"rr":round(rr,2)}

    # Apply confidence threshold gate
    if best["signal"]!="HOLD" and best["confidence"]<min_conf:
        return _blank(f"Signal found ({best['pattern']}) but confidence {best['confidence']:.0%} < threshold {min_conf:.0%}. "
                     f"Waiting for stronger setup. Factors: {best.get('reason','')}")
    return best

def update_trailing_sl(pos:dict,ltp:float,sl_type:str,trail_pts:float)->dict:
    if sl_type!=SL_TRAIL or not pos: return pos
    pos=pos.copy()
    if pos["type"]=="BUY":
        ns=ltp-trail_pts
        if ns>pos["sl"]: pos["sl"]=ns
    else:
        ns=ltp+trail_pts
        if ns<pos["sl"]: pos["sl"]=ns
    return pos

def update_trailing_tgt(pos:dict,ltp:float,tgt_type:str,trail_pts:float)->dict:
    if tgt_type!=TGT_TRAIL or not pos: return pos
    pos=pos.copy()
    if pos["type"]=="BUY":
        nt=ltp+trail_pts
        if nt>pos.get("target_display",pos["target"]): pos["target_display"]=nt
    else:
        nt=ltp-trail_pts
        if nt<pos.get("target_display",pos["target"]): pos["target_display"]=nt
    return pos

# ═══════════════════════════════════════════════════════════════════════════
# WAVE STATE
# ═══════════════════════════════════════════════════════════════════════════
def analyze_wave_state(df:pd.DataFrame,pivots:list,sig:dict)->dict:
    cur=float(df["Close"].iloc[-1]) if len(df) else 0
    if not pivots: return{"current_wave":"Collecting data…","next_wave":"—",
        "direction":"NEUTRAL","fib_levels":{},"action":"Need more bars.","auto_action":"Wait"}
    if sig["signal"]=="BUY":
        wp=sig.get("wave_pivots",[None,None,None])
        p0,p1,p2=wp[0],wp[1],wp[2]
        w1=(p1[1]-p0[1]) if p0 and p1 else cur*0.02
        w2l=p2[1] if p2 else cur*0.99
        fibs={"W3 (1.618×W1)":round(w2l+w1*1.618,2),"W3 (1.000×W1)":round(w2l+w1,2),"W3 (2.618×W1)":round(w2l+w1*2.618,2)}
        return{"current_wave":"✅ Wave-2 Bottom — ENTRY ZONE","next_wave":"Wave-3 UP ↑ (strongest)",
               "direction":"BULLISH","fib_levels":fibs,
               "action":f"BUY at market. SL: {w2l:.2f}\nW3 targets: {fibs['W3 (1.618×W1)']:.2f}·{fibs['W3 (2.618×W1)']:.2f}\nAuto-managed — no action needed.","auto_action":"BUY"}
    if sig["signal"]=="SELL":
        wp=sig.get("wave_pivots",[None,None,None])
        p0,p1,p2=wp[0],wp[1],wp[2]
        w1=(p0[1]-p1[1]) if p0 and p1 else cur*0.02
        w2h=p2[1] if p2 else cur*1.01
        fibs={"W3 (1.618×W1)":round(w2h-w1*1.618,2),"W3 (1.000×W1)":round(w2h-w1,2),"W3 (2.618×W1)":round(w2h-w1*2.618,2)}
        return{"current_wave":"🔴 Wave-2 Top — ENTRY ZONE","next_wave":"Wave-3 DOWN ↓ (strongest)",
               "direction":"BEARISH","fib_levels":fibs,
               "action":f"SELL at market. SL: {w2h:.2f}\nW3 targets: {fibs['W3 (1.618×W1)']:.2f}·{fibs['W3 (2.618×W1)']:.2f}\nAuto-managed — no action needed.","auto_action":"SELL"}
    lp=pivots[-1]; rp=0.0
    if len(pivots)>=3:
        pa,pb,pc=pivots[-3],pivots[-2],pivots[-1]
        if abs(pb[1]-pa[1])>0: rp=abs(pc[1]-pb[1])/abs(pb[1]-pa[1])*100
    return{"current_wave":f"{'High' if lp[2]=='H' else 'Low'} @ {lp[1]:.2f}",
           "next_wave":f"Waiting for W2 ({rp:.1f}% retrace, need 38–79%)" if rp else "Waiting for Wave-2 pivot",
           "direction":"NEUTRAL","fib_levels":{},
           "action":f"No signal. {sig.get('reason','')}\nSystem scans every candle. Nothing to do.","auto_action":"Wait"}

# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(df:pd.DataFrame,depth:int=5,sl_type="wave_auto",tgt_type="wave_auto",
                 capital:float=100_000.0,csl:float=50.0,ctgt:float=100.0,
                 trail_sl_pts:float=50.0,min_conf:float=MIN_CONFIDENCE_THRESHOLD)->dict:
    MB=max(40,depth*5)
    if len(df)<MB+10: return{"error":f"Need ≥{MB+10} bars.","equity_curve":[capital]}
    trades,equity_curve=[],[capital]; equity,pos=capital,None
    for i in range(MB,len(df)-1):
        bdf=df.iloc[:i+1]; nb=df.iloc[i+1]
        hi_i=float(df.iloc[i]["High"]); lo_i=float(df.iloc[i]["Low"]); cl_i=float(df.iloc[i]["Close"])
        tsig=None
        if pos:
            if sl_type==SL_TRAIL: pos=update_trailing_sl(pos,cl_i,sl_type,trail_sl_pts)
            if sl_type==SL_SIGREV or tgt_type==TGT_SIGREV:
                tsig=ew_signal(bdf,depth,sl_type,tgt_type,csl,ctgt,min_conf)
            ep_,er_=None,None
            if pos["type"]=="BUY":
                if lo_i<=pos["sl"]:       ep_,er_=pos["sl"],   "SL (Low≤SL)"
                elif hi_i>=pos["target"]: ep_,er_=pos["target"],"Target (High≥Target)"
                elif tsig and tsig["signal"]=="SELL": ep_,er_=cl_i,"Signal Reverse"
            else:
                if hi_i>=pos["sl"]:       ep_,er_=pos["sl"],   "SL (High≥SL)"
                elif lo_i<=pos["target"]: ep_,er_=pos["target"],"Target (Low≤Target)"
                elif tsig and tsig["signal"]=="BUY": ep_,er_=cl_i,"Signal Reverse"
            if ep_ is not None:
                qty=pos["qty"]; pnl=(ep_-pos["entry"])*qty if pos["type"]=="BUY" else (pos["entry"]-ep_)*qty
                equity+=pnl; equity_curve.append(equity)
                trades.append({"Entry Time":pos["entry_time"],"Exit Time":df.index[i],
                    "Type":pos["type"],"Entry":round(pos["entry"],2),"Exit":round(ep_,2),
                    "SL":round(pos["sl"],2),"Target":round(pos["target"],2),
                    "Exit Bar Low":round(lo_i,2),"Exit Bar High":round(hi_i,2),
                    "Exit Reason":er_,"PnL Rs":round(pnl,2),
                    "PnL %":round(pnl/(pos["entry"]*qty)*100,2),
                    "Equity Rs":round(equity,2),"Bars Held":i-pos["entry_bar"],
                    "Confidence":round(pos["conf"],2),"MF Score":round(pos.get("mf",0),2)})
                pos=None
        if pos is None:
            sig=ew_signal(bdf,depth,sl_type,tgt_type,csl,ctgt,min_conf)
            if sig["signal"] in("BUY","SELL"):
                ep=float(nb["Open"]); w1=sig.get("wave1_len",ep*0.02) or ep*0.02
                atr_=float(add_indicators(bdf)["ATR"].iloc[-1]) if len(bdf)>20 else ep*0.01
                if sl_type in(SL_WAVE,SL_TRAIL): sl_=sig["sl"]
                elif sl_type==SL_PTS: sl_=ep-csl if sig["signal"]=="BUY" else ep+csl
                elif sl_type==SL_SIGREV: sl_=ep*(1-0.05) if sig["signal"]=="BUY" else ep*(1+0.05)
                else: sl_=ep*(1-float(sl_type)) if sig["signal"]=="BUY" else ep*(1+float(sl_type))
                rk=abs(ep-sl_)
                if rk<=0: continue
                tgt_=_calc_target(tgt_type,ep,sig["signal"],w1,rk,ctgt,atr_)
                if sig["signal"]=="BUY" and tgt_<=ep: continue
                if sig["signal"]=="SELL" and tgt_>=ep: continue
                rr_bt=(tgt_-ep)/rk if sig["signal"]=="BUY" else (ep-tgt_)/rk
                if rr_bt<1.2: continue
                qty=max(1,int(equity*0.95/ep))
                pos={"type":sig["signal"],"entry":ep,"sl":sl_,"target":tgt_,
                     "entry_bar":i+1,"entry_time":df.index[i+1],"qty":qty,
                     "conf":sig["confidence"],"mf":sig.get("mf_score",0)}
    if pos:
        ep2=float(df["Close"].iloc[-1]); qty=pos["qty"]
        pnl=(ep2-pos["entry"])*qty if pos["type"]=="BUY" else (pos["entry"]-ep2)*qty; equity+=pnl
        trades.append({"Entry Time":pos["entry_time"],"Exit Time":df.index[-1],
            "Type":pos["type"],"Entry":round(pos["entry"],2),"Exit":round(ep2,2),
            "SL":round(pos["sl"],2),"Target":round(pos["target"],2),
            "Exit Bar Low":round(float(df["Low"].iloc[-1]),2),
            "Exit Bar High":round(float(df["High"].iloc[-1]),2),
            "Exit Reason":"Open@End","PnL Rs":round(pnl,2),
            "PnL %":round(pnl/(pos["entry"]*qty)*100,2),
            "Equity Rs":round(equity,2),"Bars Held":len(df)-1-pos["entry_bar"],
            "Confidence":round(pos["conf"],2),"MF Score":round(pos.get("mf",0),2)})
    if not trades: return{"error":"No trades. Try smaller depth, longer period, or lower confidence threshold.","equity_curve":equity_curve}
    tdf=pd.DataFrame(trades)
    wins=tdf[tdf["PnL Rs"]>0]; loss=tdf[tdf["PnL Rs"]<=0]; ntot=len(tdf)
    wr=len(wins)/ntot*100 if ntot else 0
    pf=abs(wins["PnL Rs"].sum()/loss["PnL Rs"].sum()) if len(loss) and loss["PnL Rs"].sum()!=0 else 9999.0
    ea=np.array(equity_curve); pk=np.maximum.accumulate(ea)
    mdd=float(((ea-pk)/pk*100).min())
    rets=tdf["PnL %"].values
    sharpe=float(rets.mean()/rets.std()*np.sqrt(252)) if len(rets)>1 and rets.std()!=0 else 0.0
    tdf["SL Verified"]=tdf.apply(lambda r:"✅" if(
        (r["Type"]=="BUY" and "SL" in r["Exit Reason"] and r["Exit Bar Low"]<=r["SL"]) or
        (r["Type"]=="SELL" and "SL" in r["Exit Reason"] and r["Exit Bar High"]>=r["SL"]) or
        "SL" not in r["Exit Reason"]) else "⚠️",axis=1)
    tdf["TGT Verified"]=tdf.apply(lambda r:"✅" if(
        (r["Type"]=="BUY" and "Target" in r["Exit Reason"] and r["Exit Bar High"]>=r["Target"]) or
        (r["Type"]=="SELL" and "Target" in r["Exit Reason"] and r["Exit Bar Low"]<=r["Target"]) or
        "Target" not in r["Exit Reason"]) else "⚠️",axis=1)
    return{"trades":tdf,"equity_curve":equity_curve,
           "exit_breakdown":{"SL hits":len(tdf[tdf["Exit Reason"].str.contains("SL",na=False)]),
               "Target hits":len(tdf[tdf["Exit Reason"].str.contains("Target",na=False)]),
               "Signal Reverse":len(tdf[tdf["Exit Reason"].str.contains("Reverse",na=False)]),
               "Still open":len(tdf[tdf["Exit Reason"].str.contains("Open",na=False)])},
           "metrics":{"Total Trades":ntot,"Win Rate %":round(wr,1),
               "Profit Factor":round(pf,2),"Total Return %":round((equity-capital)/capital*100,2),
               "Final Equity Rs":round(equity,2),"Max Drawdown %":round(mdd,2),
               "Sharpe Ratio":round(sharpe,2),
               "Avg Win Rs":round(float(wins["PnL Rs"].mean()),2) if len(wins) else 0.0,
               "Avg Loss Rs":round(float(loss["PnL Rs"].mean()),2) if len(loss) else 0.0,
               "Wins":len(wins),"Losses":len(loss)}}

# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════
def run_optimization(df,capital=100_000.0,csl=50.0,ctgt=100.0,trail_sl=50.0,prog_obj=None,prog_offset=0,prog_total=1):
    DEPTHS=[3,5,7,10]; SL_OPTS=[0.01,0.02,0.03,"wave_auto"]; TGT_OPTS=[1.5,2.0,3.0,"wave_auto","fib_1618"]
    combos=list(itertools.product(DEPTHS,SL_OPTS,TGT_OPTS)); rows=[]
    for idx,(dep,sl,tgt) in enumerate(combos):
        try:
            r=run_backtest(df,depth=dep,sl_type=sl,tgt_type=tgt,capital=capital,
                           csl=csl,ctgt=ctgt,trail_sl_pts=trail_sl)
            if "metrics" in r:
                m=r["metrics"]
                rows.append({"Depth":dep,"SL":str(sl),"Target":str(tgt),
                    "Trades":m["Total Trades"],"Win %":m["Win Rate %"],
                    "Return %":m["Total Return %"],"PF":m["Profit Factor"],
                    "Max DD %":m["Max Drawdown %"],"Sharpe":m["Sharpe Ratio"]})
        except Exception: pass
        if prog_obj:
            frac=prog_offset/prog_total+(idx+1)/len(combos)/prog_total
            prog_obj.progress(min(frac,0.999),text=f"Optimizing {idx+1}/{len(combos)} combos…")
    if not rows: return pd.DataFrame()
    out=pd.DataFrame(rows)
    out["Score"]=(out["Return %"].clip(lower=0)*(out["Win %"]/100)
                  *out["PF"].clip(upper=10)/(out["Max DD %"].abs()+1))
    return out.sort_values("Score",ascending=False).reset_index(drop=True)

def sl_lbl(v):
    for k,vv in SL_MAP.items():
        if str(vv)==str(v): return k
    return SL_KEYS[0]
def tgt_lbl(v):
    for k,vv in TGT_MAP.items():
        if str(vv)==str(v): return k
    try:
        fv=float(v)
        for k,vv in TGT_MAP.items():
            if isinstance(vv,float) and abs(vv-fv)<0.01: return k
    except: pass
    return TGT_KEYS[0]

# ═══════════════════════════════════════════════════════════════════════════
# FULL AUTO MODE ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def run_full_auto(symbol:str, capital:float, csl:float, ctgt:float):
    """
    Scans AUTO_SCAN_GRID of TF/period combos, runs optimization on each,
    scores all results, picks best config, applies to session state.
    """
    ss=st.session_state
    def alog(msg):
        ss.auto_log.append(f"[{now_ist().strftime('%H:%M:%S IST')}] {msg}")
        ss.auto_log=ss.auto_log[-200:]

    alog("🚀 Full Auto Mode started")
    ss.auto_status="Scanning TF/period combinations…"

    all_results=[]; valid_grid=[(tf,per,nm) for tf,per,nm in
        [(tf,per,f"{tf}·{per}") for tf,per in AUTO_SCAN_GRID]
        if per in VALID_PERIODS.get(tf,[])]

    prog=st.progress(0,text="Full Auto: starting…")
    best_overall=None; best_score_overall=-999

    for grid_idx,(tf,per,nm) in enumerate(valid_grid):
        prog_frac_start=grid_idx/len(valid_grid)
        alog(f"📡 Fetching {nm}…")
        ss.auto_status=f"Fetching {nm} ({grid_idx+1}/{len(valid_grid)})…"
        df=fetch_ohlcv(symbol,tf,per,md=1.5)
        if df is None or len(df)<50:
            alog(f"⚠️ {nm}: insufficient data — skipping"); continue

        # Quick signal check
        try:
            sig=ew_signal(df.iloc[:-1],5,"wave_auto","wave_auto",csl,ctgt)
            sig_label=sig["signal"]
            alog(f"✅ {nm}: {len(df)} bars | Signal: {sig_label} ({sig.get('confidence',0):.0%})")
        except Exception as e:
            alog(f"⚠️ {nm}: signal error {e}"); sig_label="HOLD"

        # Run optimization
        alog(f"🔬 Running optimization on {nm}…")
        ss.auto_status=f"Optimizing {nm}…"
        try:
            odf=run_optimization(df,capital,csl,ctgt,trail_sl=50.0,
                                  prog_obj=prog,
                                  prog_offset=grid_idx,
                                  prog_total=len(valid_grid))
            if odf.empty:
                alog(f"⚠️ {nm}: no optimization results"); continue
            best_row=odf.iloc[0]
            entry={
                "tf":tf,"period":per,"tf_name":nm,
                "depth":int(best_row["Depth"]),"sl":best_row["SL"],"target":best_row["Target"],
                "trades":best_row["Trades"],"win_pct":best_row["Win %"],
                "return_pct":best_row["Return %"],"pf":best_row["PF"],
                "max_dd":best_row["Max DD %"],"sharpe":best_row["Sharpe"],
                "score":best_row["Score"],"current_signal":sig_label,
                "current_conf":sig.get("confidence",0),
                "opt_df":odf,
            }
            all_results.append(entry)
            alog(f"📊 {nm}: best score={best_row['Score']:.2f} win={best_row['Win %']:.0f}% return={best_row['Return %']:.1f}%")

            # Weighted score: opt score + win rate bonus + signal alignment bonus
            bonus=0.0
            if sig_label in("BUY","SELL") and sig.get("confidence",0)>=MIN_CONFIDENCE_THRESHOLD:
                bonus=sig.get("confidence",0)*20  # up to 20 bonus points
            total_weighted=best_row["Score"]+bonus
            if total_weighted>best_score_overall:
                best_score_overall=total_weighted
                best_overall=entry.copy()
                best_overall["weighted_score"]=total_weighted
                alog(f"⭐ New best config: {nm} | score {total_weighted:.2f}")
        except Exception as e:
            alog(f"❌ {nm}: optimization failed — {e}")

    prog.empty()

    if best_overall is None:
        ss.auto_status="❌ Full Auto: no valid results found. Try different symbol or period."
        alog("❌ Full Auto completed — no results"); ss.auto_running=False; return

    # Apply best config to session state (sidebar + backtest + live)
    ss.applied_depth=best_overall["depth"]
    ss.applied_sl_lbl=sl_lbl(best_overall["sl"])
    ss.applied_tgt_lbl=tgt_lbl(best_overall["target"])
    ss.best_cfg_applied=True
    ss.auto_best_cfg=best_overall
    ss.auto_all_results=all_results
    ss.auto_status=(f"✅ Full Auto complete! Best: {best_overall['tf_name']} | "
                    f"Win {best_overall['win_pct']:.0f}% | Return {best_overall['return_pct']:.1f}%")
    alog(f"✅ DONE — Applied: TF={best_overall['tf']}/{best_overall['period']} "
         f"Depth={best_overall['depth']} SL={best_overall['sl']} Target={best_overall['target']}")
    alog(f"📌 Win Rate={best_overall['win_pct']:.0f}% | Return={best_overall['return_pct']:.1f}% | "
         f"Sharpe={best_overall['sharpe']:.2f} | Signal={best_overall['current_signal']}")
    ss.auto_running=False

# ═══════════════════════════════════════════════════════════════════════════
# LIVE ENGINE TICK
# ═══════════════════════════════════════════════════════════════════════════
def live_engine_tick(symbol,interval,period,depth,sl_type,tgt_type,
                     csl,ctgt,trail_sl_pts,trail_tgt_pts,
                     dhan_on,dhan_client,trade_mode,entry_ot,exit_ot,
                     product_type,segment,ce_sec_id,pe_sec_id,live_qty)->bool:
    ss=st.session_state; now_ts=time.time()

    ltp_new=fetch_ltp(symbol)
    if ltp_new and ltp_new>0:
        ss.live_ltp_prev=ss.live_ltp; ss.live_ltp=ltp_new
        ss.live_ltp_ts=now_ist().strftime("%H:%M:%S IST")

    ltp=ss.live_ltp
    pos=ss.live_position
    if pos and ltp:
        pos["unreal_pnl"]=(ltp-pos["entry"])*pos["qty"] if pos["type"]=="BUY" else (pos["entry"]-ltp)*pos["qty"]
        pos["dist_sl"]=abs(ltp-pos["sl"])/ltp*100
        pos["dist_tgt"]=abs(pos.get("target_display",pos["target"])-ltp)/ltp*100
        if sl_type==SL_TRAIL: pos=update_trailing_sl(pos,ltp,sl_type,trail_sl_pts)
        if tgt_type==TGT_TRAIL: pos=update_trailing_tgt(pos,ltp,tgt_type,trail_tgt_pts)
        ss.live_position=pos

    cs=POLL_SECS.get(interval,3600); ts2=now_ts-ss.last_fetch_wall
    if ts2<cs and ss.last_fetch_wall>0:
        rem=int(cs-ts2)
        ss.live_next_check_ist=(now_ist()+timedelta(seconds=rem)).strftime("%H:%M:%S IST")
        return False

    ss.live_phase="scanning"; ss.last_fetch_wall=now_ts

    def _log(msg,lvl="INFO"):
        ss.live_log.append(f"[{now_ist().strftime('%H:%M:%S IST')}][{lvl}] {msg}")
        ss.live_log=ss.live_log[-150:]

    df=fetch_ohlcv(symbol,interval,period,md=1.5)
    if df is None or len(df)<45:
        _log("⚠️  Insufficient data","WARN")
        ss.live_no_pos_reason="⚠️ No/insufficient data. Check symbol/internet."; ss.live_phase="no_signal"; return False

    ohlcv_ltp=float(df["Close"].iloc[-1])
    if not ss.live_ltp or ss.live_ltp<=0: ss.live_ltp=ohlcv_ltp

    lbdt_idx=df.index[-2] if len(df)>=2 else df.index[-1]
    ss.live_last_candle_ist=fmt_ist(lbdt_idx)
    try:
        lbdt=lbdt_idx.to_pydatetime()
        if lbdt.tzinfo is None: lbdt=lbdt.replace(tzinfo=timezone.utc)
        ss.live_delay_s=int((now_ist()-lbdt.astimezone(IST)).total_seconds())
    except: ss.live_delay_s=0
    _,_,ss.live_delay_ctx=delay_ctx(ss.live_delay_s,interval)

    df_closed=df.iloc[:-1]; latest_ts=str(df_closed.index[-1])
    pivots=find_pivots(df_closed,depth)
    sig=ew_signal(df_closed,depth,sl_type,tgt_type,csl,ctgt)
    ss.live_last_sig=sig
    ss.live_wave_state=analyze_wave_state(df_closed,pivots,sig)
    ss._scan_df=df; ss._scan_sig=sig

    nxt=(now_ist()+timedelta(seconds=cs)).strftime("%H:%M:%S IST")
    ss.live_next_check_ist=nxt

    if ss.last_bar_ts==latest_ts:
        _log(f"⏭  Same bar {latest_ts[-10:]} next check {nxt}")
        if ss.live_position:
            _manage_pos(ss.live_ltp or ohlcv_ltp,sig,sl_type,tgt_type,
                        dhan_on,dhan_client,trade_mode,exit_ot,segment,ce_sec_id,pe_sec_id,live_qty,_log)
        ss.live_phase=f"pos_{ss.live_position['type'].lower()}" if ss.live_position else "no_signal"
        return False

    ss.last_bar_ts=latest_ts; ltp_now=ss.live_ltp or ohlcv_ltp
    _log(f"🕯 Bar {latest_ts[-10:]} | LTP {ltp_now:.2f} | Signal {sig['signal']} ({sig.get('confidence',0):.0%})")
    _manage_pos(ltp_now,sig,sl_type,tgt_type,dhan_on,dhan_client,
                trade_mode,exit_ot,segment,ce_sec_id,pe_sec_id,live_qty,_log)

    pos=ss.live_position
    if pos is None and sig["signal"] in("BUY","SELL"):
        ep=ltp_now; w1=sig.get("wave1_len",ep*0.02) or ep*0.02
        df_ind_live=add_indicators(df_closed)
        atr_l=float(df_ind_live["ATR"].iloc[-1]) if not df_ind_live["ATR"].isna().iloc[-1] else ep*0.01
        if sl_type in(SL_WAVE,SL_TRAIL): sl_=sig["sl"]
        elif sl_type==SL_PTS: sl_=ep-csl if sig["signal"]=="BUY" else ep+csl
        elif sl_type==SL_SIGREV: sl_=ep*(1-0.05) if sig["signal"]=="BUY" else ep*(1+0.05)
        else: sl_=ep*(1-float(sl_type)) if sig["signal"]=="BUY" else ep*(1+float(sl_type))
        rk=abs(ep-sl_)
        if rk<=0:
            ss.live_no_pos_reason=f"⚠️ Risk=0 (SL {sl_:.2f} ≈ entry {ep:.2f}). Increase SL distance."
            ss.live_phase="no_signal"; return True
        tgt_=_calc_target(tgt_type,ep,sig["signal"],w1,rk,ctgt,atr_l)
        if (sig["signal"]=="BUY" and tgt_<=ep) or (sig["signal"]=="SELL" and tgt_>=ep):
            ss.live_no_pos_reason=f"⚠️ Invalid target ({tgt_:.2f}). Adjust Target setting."
            ss.live_phase="no_signal"; return True
        rr_live=(tgt_-ep)/rk if sig["signal"]=="BUY" else (ep-tgt_)/rk
        if rr_live<1.2:
            ss.live_no_pos_reason=f"⚠️ R:R {rr_live:.2f}:1 below minimum 1.2:1. Signal skipped."
            ss.live_phase="no_signal"; return True

        ei=now_ist().strftime("%d-%b %H:%M:%S IST")
        # Limit price from signal entry_price, not from user
        limit_price=sig["entry_price"] if entry_ot=="LIMIT" else 0.0

        ss.live_position={"type":sig["signal"],"entry":ep,"sl":sl_,"target":tgt_,"target_display":tgt_,
            "qty":live_qty,"entry_ist":ei,"symbol":symbol,"pattern":sig["pattern"],
            "confidence":sig["confidence"],"mf_score":sig.get("mf_score",0),
            "rr":rr_live,"unreal_pnl":0.0,"dist_sl":rk/ep*100,"dist_tgt":abs(tgt_-ep)/ep*100}
        ss.live_phase=f"pos_{sig['signal'].lower()}"
        ss.live_no_pos_reason=""
        ss.live_signals.append({"Time (IST)":ei,"Bar":fmt_ist(df_closed.index[-1]),
            "TF":interval,"Period":period,"Signal":sig["signal"],
            "Entry":round(ep,2),"SL":round(sl_,2),"Target":round(tgt_,2),
            "Conf":f"{sig['confidence']:.0%}","MF Score":f"{sig.get('mf_score',0):.0%}",
            "R:R":f"1:{rr_live:.1f}","Pattern":sig["pattern"]})
        if dhan_on and dhan_client:
            sec=ce_sec_id if(trade_mode=="options" and sig["signal"]=="BUY") else pe_sec_id if trade_mode=="options" else ""
            r=dhan_client.place_order(sec or "",segment,sig["signal"],live_qty,
                order_type=entry_ot,price=limit_price,product_type=product_type)
            _log(f"📤 Dhan entry: {r}")
        em="🟢" if sig["signal"]=="BUY" else "🔴"
        _log(f"{em} ENTERED {sig['signal']} @ {ep:.2f} | SL {sl_:.2f} | T {tgt_:.2f} | Conf {sig['confidence']:.0%} | MF {sig.get('mf_score',0):.0%}")
    elif pos is None:
        ss.live_phase="no_signal"
        ss.live_no_pos_reason=f"📊 HOLD. {sig.get('reason','')}\nNext check: {nxt}"
        _log(f"⏸  HOLD @ {ltp_now:.2f} | {sig.get('reason','')}")
    return True

def _manage_pos(ltp,sig,sl_type,tgt_type,dhan_on,dhan_client,
                trade_mode,exit_ot,segment,ce_sec_id,pe_sec_id,qty_,log_fn):
    ss=st.session_state; pos=ss.live_position
    if not pos or not ltp: return
    hit_p,hit_r=None,None; pt=pos["type"]; sl_=pos["sl"]; tgt_=pos["target"]
    if pt=="BUY":
        if ltp<=sl_:       hit_p,hit_r=sl_,  "SL Hit (≤SL)"
        elif ltp>=tgt_ and tgt_type!=TGT_TRAIL: hit_p,hit_r=tgt_,"Target Hit (≥Target)"
        elif sig["signal"]=="SELL": hit_p,hit_r=ltp,"Signal Reversed→SELL"
    else:
        if ltp>=sl_:       hit_p,hit_r=sl_,  "SL Hit (≥SL)"
        elif ltp<=tgt_ and tgt_type!=TGT_TRAIL: hit_p,hit_r=tgt_,"Target Hit (≤Target)"
        elif sig["signal"]=="BUY": hit_p,hit_r=ltp,"Signal Reversed→BUY"
    if hit_p is None: return
    qty=pos["qty"]; pnl=(hit_p-pos["entry"])*qty if pt=="BUY" else (pos["entry"]-hit_p)*qty
    ss.live_pnl+=pnl  # ONLY place live_pnl is modified
    ss.live_trades.append({"Time (IST)":now_ist().strftime("%Y-%m-%d %H:%M:%S IST"),
        "Symbol":pos["symbol"],"TF":ss.get("_live_interval","—"),
        "Type":pt,"Entry":round(pos["entry"],2),"Exit":round(hit_p,2),
        "SL":round(pos["sl"],2),"Target":round(pos["target"],2),
        "Qty":qty,"PnL Rs":round(pnl,2),"Reason":hit_r})
    if dhan_on and dhan_client:
        xt="SELL" if pt=="BUY" else "BUY"
        sec=ce_sec_id if(trade_mode=="options" and xt=="BUY") else pe_sec_id if trade_mode=="options" else ""
        # Limit exit price from current LTP (from signal area)
        xp=hit_p if exit_ot=="LIMIT" else 0.0
        log_fn(f"📤 Dhan exit: {dhan_client.place_order(sec or '',segment,xt,qty,order_type=exit_ot,price=xp)}")
    em="✅" if "Target" in hit_r else ("🔄" if "Reversed" in hit_r else "❌")
    log_fn(f"{em} {pt} CLOSED @ {hit_p:.2f} | {hit_r} | Rs{pnl:+.2f}")
    ss.live_position=None
    ss.live_phase="reversing" if "Reversed" in hit_r else "no_signal"
    if "Reversed" not in hit_r: ss.live_no_pos_reason=f"Last: {hit_r} | Rs{pnl:+.2f}"

# ═══════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════
def chart_waves(df,pivots,sig=None,trades=None,symbol="",tf_label=""):
    sig=sig or _blank(); df_ind=add_indicators(df)
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,
                      row_heights=[0.60,0.20,0.20],vertical_spacing=0.02,
                      subplot_titles=("","Volume","RSI + StochRSI"))
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],name="Price",
        increasing=dict(line=dict(color="#26a69a"),fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350"),fillcolor="#ef5350")),row=1,col=1)
    for col,clr,nm in[("EMA_9","#ffeb3b","EMA9"),("EMA_21","#ffb300","EMA21"),("EMA_50","#ab47bc","EMA50"),("EMA_200","#ef5350","EMA200")]:
        if col in df_ind.columns:
            fig.add_trace(go.Scatter(x=df_ind.index,y=df_ind[col],mode="lines",
                line=dict(color=clr,width=1.0),name=nm,opacity=0.65),row=1,col=1)
    if "Volume" in df.columns:
        vc=["#26a69a" if c>=o else "#ef5350" for c,o in zip(df["Close"],df["Open"])]
        fig.add_trace(go.Bar(x=df.index,y=df["Volume"],marker_color=vc,opacity=0.4,name="Vol",showlegend=False),row=2,col=1)
    if "RSI" in df_ind.columns:
        fig.add_trace(go.Scatter(x=df_ind.index,y=df_ind["RSI"],mode="lines",
            line=dict(color="#00e5ff",width=1.5),name="RSI",showlegend=False),row=3,col=1)
        fig.add_hline(y=70,line=dict(dash="dot",color="#ef5350",width=1),row=3,col=1)
        fig.add_hline(y=30,line=dict(dash="dot",color="#4caf50",width=1),row=3,col=1)
    if "StochRSI" in df_ind.columns:
        fig.add_trace(go.Scatter(x=df_ind.index,y=df_ind["StochRSI"]*100,mode="lines",
            line=dict(color="#ff9800",width=1.0,dash="dot"),name="StochRSI×100",showlegend=False),row=3,col=1)
    vp=[p for p in pivots if p[0]<len(df)]
    if vp:
        fig.add_trace(go.Scatter(x=[df.index[p[0]] for p in vp],y=[p[1] for p in vp],mode="lines+markers",
            line=dict(color="rgba(255,180,0,.5)",width=1.5,dash="dot"),
            marker=dict(size=7,color=["#4caf50" if p[2]=="L" else "#f44336" for p in vp],
                        symbol=["triangle-up" if p[2]=="L" else "triangle-down" for p in vp]),name="ZigZag"),row=1,col=1)
    wp=sig.get("wave_pivots")
    if wp:
        vwp=[p for p in wp if p[0]<len(df)]
        if vwp:
            clr="#00e5ff" if sig["signal"]=="BUY" else "#ff4081"
            fig.add_trace(go.Scatter(x=[df.index[p[0]] for p in vwp],y=[p[1] for p in vwp],
                mode="lines+markers+text",line=dict(color=clr,width=2.5),marker=dict(size=12,color=clr),
                text=["W0","W1","W2"][:len(vwp)],textposition="top center",
                textfont=dict(color=clr,size=12,family="Share Tech Mono"),name="EW Pattern"),row=1,col=1)
    if sig["signal"] in("BUY","SELL"):
        sc="#4caf50" if sig["signal"]=="BUY" else "#f44336"
        fig.add_trace(go.Scatter(x=[df.index[-1]],y=[df["Close"].iloc[-1]],mode="markers",
            marker=dict(size=20,color=sc,symbol="triangle-up" if sig["signal"]=="BUY" else "triangle-down",
                        line=dict(color="white",width=1.5)),name=f"▶ {sig['signal']}"),row=1,col=1)
        if sig.get("sl"):
            fig.add_hline(y=sig["sl"],line=dict(dash="dash",color="#ff7043",width=1.5),
                annotation_text="  SL",annotation_position="right",row=1,col=1)
        if sig.get("target"):
            fig.add_hline(y=sig["target"],line=dict(dash="dash",color="#66bb6a",width=1.5),
                annotation_text="  Target",annotation_position="right",row=1,col=1)
    if trades is not None and not trades.empty:
        for tt,sy,cl in[("BUY","triangle-up","#4caf50"),("SELL","triangle-down","#f44336")]:
            sub=trades[trades["Type"]==tt]
            if not sub.empty:
                fig.add_trace(go.Scatter(x=sub["Entry Time"],y=sub["Entry"],mode="markers",
                    marker=dict(size=9,color=cl,symbol=sy,line=dict(color="white",width=0.8)),
                    name=f"{tt} Entry"),row=1,col=1)
        for rsn,sy,cl in[("Target","circle","#66bb6a"),("SL","x","#ef5350"),("Reverse","diamond","#ab47bc")]:
            sub=trades[trades["Exit Reason"].str.contains(rsn,na=False)]
            if not sub.empty:
                fig.add_trace(go.Scatter(x=sub["Exit Time"],y=sub["Exit"],mode="markers",
                    marker=dict(size=7,color=cl,symbol=sy),name=f"Exit({rsn})",visible="legendonly"),row=1,col=1)
    fig.update_layout(title=dict(text=f"🌊 {symbol}"+(f" · {tf_label}" if tf_label else ""),font=dict(size=13,color="#00e5ff")),
        template="plotly_dark",height=560,xaxis_rangeslider_visible=False,
        plot_bgcolor="#06101a",paper_bgcolor="#06101a",font=dict(color="#b0bec5",family="Exo 2"),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,x=0,font=dict(size=10)),
        margin=dict(l=10,r=70,t=45,b=10))
    return fig

def chart_equity(ec):
    eq=np.array(ec,dtype=float); pk=np.maximum.accumulate(eq); dd=(eq-pk)/pk*100
    fig=make_subplots(rows=2,cols=1,row_heights=[0.65,0.35],vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=eq,mode="lines",name="Equity",line=dict(color="#00bcd4",width=2),
        fill="tozeroy",fillcolor="rgba(0,188,212,.07)"),row=1,col=1)
    fig.add_trace(go.Scatter(y=dd,mode="lines",name="Drawdown %",line=dict(color="#f44336",width=1.5),
        fill="tozeroy",fillcolor="rgba(244,67,54,.12)"),row=2,col=1)
    fig.add_hline(y=0,line=dict(dash="dot",color="#546e7a",width=1),row=2,col=1)
    fig.update_layout(template="plotly_dark",height=330,plot_bgcolor="#06101a",
        paper_bgcolor="#06101a",font=dict(color="#b0bec5",family="Exo 2"),
        margin=dict(l=10,r=10,t=15,b=10))
    return fig

def chart_opt_scatter(odf):
    fig=go.Figure(go.Scatter(x=odf["Max DD %"].abs(),y=odf["Return %"],mode="markers",
        marker=dict(size=(odf["Win %"]/5).clip(lower=4),color=odf["Score"],colorscale="Plasma",showscale=True,
                    colorbar=dict(title=dict(text="Score",font=dict(color="#b0bec5",size=11)),
                    tickfont=dict(color="#b0bec5")),line=dict(color="rgba(255,255,255,.2)",width=0.5)),
        text=[f"D={r.Depth} SL={r.SL} T={r.Target}" for _,r in odf.iterrows()],
        hovertemplate="<b>%{text}</b><br>Ret %{y:.1f}% DD %{x:.1f}%<extra></extra>"))
    fig.update_layout(title=dict(text="Return vs Max Drawdown (bubble=WinRate)",font=dict(size=12,color="#00e5ff")),
        xaxis_title="Max Drawdown %",yaxis_title="Total Return %",template="plotly_dark",height=370,
        plot_bgcolor="#06101a",paper_bgcolor="#06101a",font=dict(color="#b0bec5",family="Exo 2"),
        margin=dict(l=10,r=10,t=40,b=10))
    return fig

def generate_mtf_summary(symbol,results,overall_sig):
    lines=[f"## 🌊 {symbol} — Elliott Wave Analysis",f"*{now_ist().strftime('%Y-%m-%d %H:%M:%S IST')}*\n"]
    bc=sum(1 for r in results if r["signal"]["signal"]=="BUY")
    sc=sum(1 for r in results if r["signal"]["signal"]=="SELL")
    hc=sum(1 for r in results if r["signal"]["signal"]=="HOLD")
    vi={"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}.get(overall_sig,"⚪")
    lines.append(f"### {vi} Overall: **{overall_sig}** ({bc}B·{sc}S·{hc}H)\n")
    for r in results:
        sig=r["signal"]; s=sig["signal"]; em={"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}.get(s,"⚪")
        lines.append(f"#### {em} {r['tf_name']}")
        if s in("BUY","SELL"):
            ep=sig["entry_price"]; sl_=sig["sl"]; tgt_=sig["target"]
            rr=abs(tgt_-ep)/abs(ep-sl_) if abs(ep-sl_)>0 else 0
            lines.append(f"- {s} | Entry:{ep:.2f} SL:{sl_:.2f} Target:{tgt_:.2f} R:R 1:{rr:.1f}")
            lines.append(f"- Confidence:{sig['confidence']:.0%} | MF Score:{sig.get('mf_score',0):.0%} | {sig['pattern']}")
            factors=sig.get("factors",{})
            if factors: lines.append(f"- Factors: {'; '.join(f'{k}:{v}' for k,v in factors.items())}")
        else:
            lines.append(f"- HOLD: {sig.get('reason','—')}")
        lines.append("")
    lines+=["---","| Wave | Meaning | App Signal |","|------|---------|------------|",
            "| W1 | First impulse | Detected |","| **W2** | **38–79% retrace** | **🟢 BUY/SELL here** |",
            "| **W3** | **Strongest** | **Hold** |","| W4 | Pullback | Partial exit |","| W5 | Final | Full exit |",
            "\n> ⚠️ *Not financial advice.*"]
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════
# ███  STREAMLIT UI  ███
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="main-hdr">
  <h1>🌊 Elliott Wave Algo Trader v5.3</h1>
  <p>Full Auto Mode · 8-Factor Signal Scoring · Dhan Stocks+Options · IST · Trailing SL/TGT</p>
</div>""",unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Instrument")
    group_sel=st.selectbox("Category",list(TICKER_GROUPS.keys()),index=0)
    gmap=TICKER_GROUPS[group_sel]
    if group_sel=="✏️ Custom Ticker":
        symbol=st.text_input("Yahoo Finance ticker","^NSEI")
    else:
        tn=st.selectbox("Instrument",list(gmap.keys()))
        symbol=gmap[tn]; st.caption(f"Yahoo: `{symbol}`")
    st.session_state["_live_interval"]=st.session_state.get("_live_interval","1d")

    st.markdown("---")
    # Full Auto Mode checkbox
    auto_mode=st.checkbox("🤖 Full Auto Mode",value=st.session_state.auto_mode,
        help="Automatically analyzes all TF/periods, optimizes, finds best config, applies to Live Trading. One click → auto trade!")
    st.session_state.auto_mode=auto_mode
    if auto_mode:
        st.markdown("""<div style="background:rgba(0,229,255,.1);border:1px solid #00e5ff;
        border-radius:8px;padding:8px 11px;font-size:.8rem;color:#00e5ff">
        🤖 <b>Full Auto Mode ON</b><br>
        <span style="color:#78909c;font-size:.77rem">Scans 8 TF/period combos → optimizes → picks best config → applies to Live Trading automatically</span>
        </div>""",unsafe_allow_html=True)
        if st.button("🚀 Run Full Auto Analysis",type="primary",use_container_width=True):
            if not st.session_state.auto_running:
                st.session_state.auto_running=True
                st.session_state.auto_log=[]
                st.session_state.auto_status="Starting…"
                run_full_auto(symbol,100_000.0,50.0,100.0)
                st.rerun()

    if st.session_state.best_cfg_applied:
        st.markdown("""<div style="background:rgba(0,229,255,.08);border:1px solid #00bcd4;
        border-radius:8px;padding:7px 10px;font-size:.79rem;margin-top:6px">
        ✨ <b style="color:#00e5ff">Optimized Config Active</b></div>""",unsafe_allow_html=True)

    st.markdown("---")
    c1,c2=st.columns(2)
    interval=c1.selectbox("⏱ TF",TIMEFRAMES,index=6)
    vpl=VALID_PERIODS.get(interval,PERIODS)
    period=c2.selectbox("📅 Period",vpl,index=min(4,len(vpl)-1))
    st.session_state["_live_interval"]=interval

    st.markdown("---")
    st.markdown("### 🌊 Elliott Wave")
    depth=st.slider("Pivot Depth",2,15,st.session_state.applied_depth)
    conf_thresh=st.slider("Min Confidence %",40,90,int(MIN_CONFIDENCE_THRESHOLD*100),5,
        help="Higher = fewer but higher quality signals. Default 68%. Raise to 75%+ for fewer false signals.")/100

    st.markdown("---")
    st.markdown("### 🛡️ Risk")
    si=SL_KEYS.index(st.session_state.applied_sl_lbl) if st.session_state.applied_sl_lbl in SL_KEYS else 0
    ti=TGT_KEYS.index(st.session_state.applied_tgt_lbl) if st.session_state.applied_tgt_lbl in TGT_KEYS else 0
    sl_lbl_sel=st.selectbox("Stop Loss",SL_KEYS,index=si)
    tgt_lbl_sel=st.selectbox("Target",TGT_KEYS,index=ti)
    sl_val=SL_MAP[sl_lbl_sel]; tgt_val=TGT_MAP[tgt_lbl_sel]
    if sl_val==SL_PTS: st.session_state.custom_sl_pts=st.number_input("SL Points",1.0,1e6,st.session_state.custom_sl_pts,5.0)
    if tgt_val==TGT_PTS: st.session_state.custom_tgt_pts=st.number_input("Target Points",1.0,1e6,st.session_state.custom_tgt_pts,10.0)
    if sl_val==SL_TRAIL: st.session_state.trailing_sl_pts=st.number_input("Trailing SL Pts",1.0,1e6,st.session_state.trailing_sl_pts,5.0)
    if tgt_val==TGT_TRAIL: st.session_state.trailing_tgt_pts=st.number_input("Trailing TGT Pts (display)",1.0,1e6,st.session_state.trailing_tgt_pts,10.0)
    csl=st.session_state.custom_sl_pts; ctgt=st.session_state.custom_tgt_pts
    trail_sl_pts=st.session_state.trailing_sl_pts; trail_tgt_pts=st.session_state.trailing_tgt_pts
    capital=st.number_input("💰 Capital (Rs)",10_000,50_000_000,100_000,10_000)

    st.markdown("---")
    st.markdown("### 🏦 Dhan Brokerage")
    if not DHAN_LIB_OK: st.warning("Install: `pip install dhanhq`")
    dhan_on=st.checkbox("Enable Dhan Integration",value=False)
    dhan_client=None; trade_mode="stocks"; product_type="INTRADAY"
    segment="NSE"; entry_ot="MARKET"; exit_ot="MARKET"
    ce_sec_id=""; pe_sec_id=""; live_qty=1

    if dhan_on:
        d_cid=st.text_input("Client ID",value=DEFAULT_CLIENT_ID)
        d_tok=st.text_area("Access Token",value=DEFAULT_ACCESS_TOKEN,height=80)
        if d_cid and d_tok:
            dhan_client=DhanClient(d_cid,d_tok)
            if st.button("🔌 Test"): st.json(dhan_client.fund_limit())
        trade_mode=st.radio("Mode",["Stocks","Options"],horizontal=True).lower()
        if trade_mode=="stocks":
            c1d,c2d=st.columns(2)
            product_type=c1d.selectbox("Type",["INTRADAY","DELIVERY"],index=0)
            exchange=c2d.selectbox("Exchange",["NSE","BSE"],index=0); segment=exchange
            sec_id_s=st.text_input("Security ID","1333"); live_qty=st.number_input("Qty",1,100_000,1)
            ce_sec_id=pe_sec_id=sec_id_s
        else:
            c1d,c2d=st.columns(2)
            segment=c1d.selectbox("Segment",["NSE_FNO","BSE_FNO"],index=0)
            product_type=c2d.selectbox("Product",["INTRADAY","MARGIN"],index=0)
            ce_sec_id=st.text_input("CE Security ID","52175")
            pe_sec_id=st.text_input("PE Security ID","52176")
            live_qty=st.number_input("Lot Size",1,10_000,50)
            st.caption("BUY→CE | SELL→PE. Limit price from signal.")
        c1o,c2o=st.columns(2)
        entry_ot=c1o.selectbox("Entry",["LIMIT","MARKET"],index=0)
        exit_ot=c2o.selectbox("Exit",["MARKET","LIMIT"],index=0)
        if entry_ot=="LIMIT": st.caption("📌 Limit price auto-taken from signal entry level.")
        if exit_ot=="LIMIT": st.caption("📌 Limit exit price = SL/Target level from signal.")

    st.markdown("---")
    st.caption(f"⚡ Rate-limit 1.5s | LTP every 2s | Conf≥{conf_thresh:.0%}")
    st.caption(f"`{symbol}` · `{interval}` · `{period}`")

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
t_analysis,t_live,t_bt,t_opt,t_help=st.tabs([
    "🔭  Wave Analysis","🔴  Live Trading","📊  Backtest","🔬  Optimization","❓  Help"])

# ── WAVE ANALYSIS ─────────────────────────────────────────────────────────
with t_analysis:
    st.markdown("### 🔭 Multi-Timeframe Elliott Wave Analysis")
    ac1,ac2,ac3=st.columns([1.2,1,2.4])
    with ac1: run_analysis=st.button("🔭 Run Full Analysis",type="primary",use_container_width=True)
    with ac2: custom_tf_only=st.checkbox("Sidebar TF only",value=False)
    with ac3: st.caption(f"{'Sidebar TF' if custom_tf_only else 'Daily·4H·1H·15M'} | Min confidence {conf_thresh:.0%}")
    if run_analysis:
        sc2=[(interval,period,f"{interval}·{period}")] if custom_tf_only \
            else [(tf,per,nm) for tf,per,nm in MTF_COMBOS if per in VALID_PERIODS.get(tf,[])]
        results=[]; prog=st.progress(0,text="Scanning…")
        for idx,(tf,per,nm) in enumerate(sc2):
            prog.progress((idx+1)/len(sc2),text=f"Fetching {nm}…")
            try:
                dfa=fetch_ohlcv(symbol,tf,per,md=1.5)
                if dfa is not None and len(dfa)>=45:
                    sa=ew_signal(dfa.iloc[:-1],depth,sl_val,tgt_val,csl,ctgt,conf_thresh)
                    results.append({"tf_name":nm,"interval":tf,"period":per,"signal":sa,"df":dfa,"pivots":find_pivots(dfa.iloc[:-1],depth)})
                else:
                    results.append({"tf_name":nm,"interval":tf,"period":per,"signal":_blank("No data"),"df":None,"pivots":[]})
            except Exception as e:
                results.append({"tf_name":nm,"interval":tf,"period":per,"signal":_blank(str(e)),"df":None,"pivots":[]})
        prog.empty()
        bs2=sum(r["signal"]["confidence"] for r in results if r["signal"]["signal"]=="BUY")
        ss2=sum(r["signal"]["confidence"] for r in results if r["signal"]["signal"]=="SELL")
        ov="BUY" if bs2>ss2 and bs2>0.5 else "SELL" if ss2>bs2 and ss2>0.5 else "HOLD"
        st.session_state.update({"_analysis_results":results,"_analysis_overall":ov,"_analysis_symbol":symbol})
    ar=st.session_state.get("_analysis_results")
    if ar:
        ov=st.session_state.get("_analysis_overall","HOLD"); asym=st.session_state.get("_analysis_symbol",symbol)
        vc={"BUY":"#4caf50","SELL":"#f44336","HOLD":"#ffb300"}; vb={"BUY":"rgba(76,175,80,.10)","SELL":"rgba(244,67,54,.10)","HOLD":"rgba(255,179,0,.10)"}; vi={"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}
        st.markdown(f"""<div style="background:{vb[ov]};border:2px solid {vc[ov]};border-radius:12px;padding:14px 20px;margin-bottom:10px;text-align:center">
        <span style="font-size:1.5rem;color:{vc[ov]};font-weight:700">{vi[ov]} Overall: {ov}</span><br>
        <span style="color:#78909c;font-size:.84rem">{asym} · Multi-TF Consensus · Min confidence {conf_thresh:.0%}</span></div>""",unsafe_allow_html=True)
        tfc=st.columns(min(len(ar),4))
        for i,r in enumerate(ar):
            with tfc[i%4]:
                s=r["signal"]["signal"]; c_=r["signal"]["confidence"]; mf_=r["signal"].get("mf_score",0)
                sc3=vc.get(s,"#546e7a"); em=vi.get(s,"⚪")
                st.markdown(f"""<div style="background:rgba(255,255,255,.03);border:1px solid #1e3a5f;border-radius:8px;padding:9px 11px;text-align:center;margin-bottom:4px">
                <div style="font-size:.75rem;color:#546e7a">{r['tf_name']}</div>
                <div style="font-size:1.05rem;color:{sc3};font-weight:700">{em} {s}</div>
                <div style="font-size:.73rem;color:#78909c">{r['signal']['pattern']}</div>
                <div style="font-size:.77rem;color:#00bcd4">Conf {c_:.0%} | MF {mf_:.0%}</div></div>""",unsafe_allow_html=True)
        st.markdown("---")
        for r in ar:
            if r["df"] is not None:
                s_=r["signal"]["signal"]
                with st.expander(f"📈 {r['tf_name']} — {s_} (Conf {r['signal']['confidence']:.0%} | MF {r['signal'].get('mf_score',0):.0%})",expanded=(s_!="HOLD")):
                    st.plotly_chart(chart_waves(r["df"],r["pivots"],r["signal"],symbol=asym,tf_label=r["tf_name"]),use_container_width=True)
                    if s_ in("BUY","SELL"):
                        sg=r["signal"]; ep,sl_,tgt_=sg["entry_price"],sg["sl"],sg["target"]
                        rr=abs(tgt_-ep)/abs(ep-sl_) if abs(ep-sl_)>0 else 0
                        mc1,mc2,mc3,mc4,mc5=st.columns(5)
                        mc1.metric("Entry",f"{ep:.2f}"); mc2.metric("SL",f"{sl_:.2f}",delta=f"-{abs(ep-sl_)/ep*100:.1f}%",delta_color="inverse")
                        mc3.metric("Target",f"{tgt_:.2f}",delta=f"+{abs(tgt_-ep)/ep*100:.1f}%"); mc4.metric("R:R",f"1:{rr:.1f}")
                        mc5.metric("MF Score",f"{sg.get('mf_score',0):.0%}")
                        if sg.get("factors"):
                            st.markdown("**8-Factor Analysis:**")
                            fcols=st.columns(4)
                            for j,(fk,fv) in enumerate(sg["factors"].items()):
                                fcols[j%4].markdown(f"<small style='color:#78909c'><b style='color:#00bcd4'>{fk}</b>: {fv}</small>",unsafe_allow_html=True)
        st.markdown("---"); st.markdown("### 📋 Analysis & Recommendations")
        st.markdown(generate_mtf_summary(asym,ar,ov))
    else:
        st.info("Click 🔭 Run Full Analysis to scan all timeframes with 8-factor confirmation.")

# ── LIVE TRADING ───────────────────────────────────────────────────────────
with t_live:
    ctl1,ctl2,ctl3=st.columns([1,1,4])
    with ctl1:
        if not st.session_state.live_running:
            if st.button("▶ Start Live",type="primary",use_container_width=True):
                st.session_state.update({"live_running":True,"live_phase":"scanning",
                    "live_log":[],"last_bar_ts":None,"last_fetch_wall":0.0,
                    "live_no_pos_reason":"System starting…"}); st.rerun()
        else:
            if st.button("⏹ Stop",type="secondary",use_container_width=True):
                st.session_state.update({"live_running":False,"live_phase":"idle"}); st.rerun()
    with ctl2:
        if st.button("🔄 Reset All",use_container_width=True):
            for k,v in _DEF.items(): st.session_state[k]=v
            st.success("Reset ✓"); time.sleep(0.3); st.rerun()
    with ctl3:
        if st.session_state.live_running:
            st.success(f"🟢 RUNNING — `{symbol}` · `{interval}` · `{period}` | LTP 2s | Candle {POLL_SECS.get(interval,3600)//60}min | MinConf {conf_thresh:.0%}")
        else:
            st.warning("⚫ STOPPED — Click ▶ Start Live. Everything runs automatically.")

    # ── Full Auto Config banner on Live tab ──────────────────────────────
    auto_best=st.session_state.auto_best_cfg
    if auto_best:
        st.markdown(f"""
        <div class="auto-banner">
        <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px">
        <div>
          <div style="font-size:.75rem;color:#78909c;letter-spacing:.5px">🤖 FULL AUTO — BEST CONFIGURATION</div>
          <div style="font-size:1.25rem;font-weight:700;color:#00e5ff">
            {auto_best['tf_name']} · Depth {auto_best['depth']} · SL: {auto_best['sl']} · Target: {auto_best['target']}
          </div>
          <div style="font-size:.82rem;color:#b0bec5;margin-top:3px">
            Win Rate: <b style="color:#4caf50">{auto_best['win_pct']:.0f}%</b> ·
            Return: <b style="color:#4caf50">{auto_best['return_pct']:.1f}%</b> ·
            PF: <b>{auto_best['pf']:.2f}</b> ·
            Sharpe: <b>{auto_best['sharpe']:.2f}</b> ·
            Score: <b>{auto_best['score']:.2f}</b>
          </div>
        </div>
        <div style="text-align:right">
          <div style="font-size:.74rem;color:#546e7a">Current Signal</div>
          <div style="font-size:1.1rem;font-weight:700;color:{'#4caf50' if auto_best['current_signal']=='BUY' else '#f44336' if auto_best['current_signal']=='SELL' else '#78909c'}">
            {'🟢' if auto_best['current_signal']=='BUY' else '🔴' if auto_best['current_signal']=='SELL' else '⏸'} {auto_best['current_signal']}
          </div>
          <div style="font-size:.79rem;color:#00bcd4">{auto_best['current_conf']:.0%} confidence</div>
        </div>
        </div>
        <div style="margin-top:8px;font-size:.77rem;color:#546e7a">
        ✅ This config is applied to sidebar, backtest, and live trading. System will trade automatically based on this config.
        </div>
        </div>""",unsafe_allow_html=True)

    pos_ph=st.empty()

    def render_pos(container):
        ss=st.session_state; pos=ss.live_position; ltp=ss.live_ltp; phase=ss.live_phase
        with container:
            if pos:
                pt=pos["type"]; clr="#4caf50" if pt=="BUY" else "#f44336"; css="pos-buy" if pt=="BUY" else "pos-sell"
                up=pos.get("unreal_pnl",0); up_c="#4caf50" if up>=0 else "#f44336"; up_sym="▲" if up>=0 else "▼"
                sl_=pos["sl"]; tgt_=pos.get("target_display",pos["target"])
                ds=pos.get("dist_sl",0); dt=pos.get("dist_tgt",0); ltp_s=f"{ltp:,.2f}" if ltp else "—"
                mf_badge=f"MF {pos.get('mf_score',0):.0%}" if pos.get("mf_score") else ""
                st.markdown(f"""
                <div class="{css}">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px">
                <div>
                  <div style="font-size:.74rem;color:#78909c">POSITION STATUS</div>
                  <div style="font-size:1.9rem;font-weight:700;color:{clr}">{pt} OPEN ✅</div>
                  <div style="font-size:.80rem;color:#b0bec5">{pos.get('pattern','—')} | Conf {pos.get('confidence',0):.0%} {mf_badge}</div>
                  <div style="font-size:.74rem;color:#546e7a">R:R 1:{pos.get('rr',0):.1f} | Entered {pos.get('entry_ist','—')}</div>
                </div>
                <div style="text-align:right">
                  <div style="font-size:.73rem;color:#546e7a">UNREALIZED P&amp;L</div>
                  <div style="font-size:1.6rem;font-weight:700;color:{up_c};font-family:'Share Tech Mono'">{up_sym} Rs{abs(up):,.2f}</div>
                  <div style="font-size:.77rem;color:#78909c">LTP: {ltp_s}</div>
                </div>
                </div>
                <hr style="border-color:rgba(255,255,255,.08);margin:8px 0">
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;font-size:.81rem">
                  <div><div style="color:#546e7a;font-size:.71rem">ENTRY</div><div style="color:#b0bec5;font-weight:600">{pos['entry']:,.2f}</div></div>
                  <div><div style="color:#546e7a;font-size:.71rem">SL</div><div style="color:#ff7043;font-weight:600">{sl_:,.2f}<br><span style="color:#546e7a;font-size:.70rem">{ds:.1f}% away</span></div></div>
                  <div><div style="color:#546e7a;font-size:.71rem">TARGET</div><div style="color:#66bb6a;font-weight:600">{tgt_:,.2f}<br><span style="color:#546e7a;font-size:.70rem">{dt:.1f}% away</span></div></div>
                  <div><div style="color:#546e7a;font-size:.71rem">QTY</div><div style="color:#b0bec5;font-weight:600">{pos['qty']}</div></div>
                </div>
                <div style="margin-top:8px;padding:6px 10px;background:rgba(0,0,0,.3);border-radius:6px;font-size:.76rem;color:#78909c">
                🤖 Auto-managed. Closes at SL / Target / Signal Reverse. <b>No action needed.</b>
                </div></div>""",unsafe_allow_html=True)
            elif phase=="reversing":
                st.markdown("""<div class="pos-reversing">
                <div style="font-size:1.3rem;font-weight:700;color:#ab47bc">🔄 AUTO-REVERSING…</div>
                <div style="font-size:.82rem;color:#b0bec5;margin-top:4px">Closing old position and opening reverse. No action needed.</div>
                </div>""",unsafe_allow_html=True)
            elif phase in("scanning","signal_ready"):
                sig_=ss.live_last_sig
                if sig_ and sig_.get("signal") in("BUY","SELL"):
                    sc_="#00e5ff"
                    st.markdown(f"""<div class="pos-signal">
                    <div style="font-size:.74rem;color:#78909c">SIGNAL DETECTED — OPENING POSITION</div>
                    <div style="font-size:1.5rem;font-weight:700;color:{sc_}">💡 {sig_['signal']} FIRED</div>
                    <div style="font-size:.80rem;color:#b0bec5">{sig_.get('pattern','—')} | Conf {sig_.get('confidence',0):.0%} | MF {sig_.get('mf_score',0):.0%}</div>
                    <div style="font-size:.78rem;color:#78909c;margin-top:4px">Entry: ~{sig_.get('entry_price',0):,.2f} | SL: {sig_.get('sl',0):,.2f} | Target: {sig_.get('target',0):,.2f}</div>
                    <div style="font-size:.75rem;color:#546e7a;margin-top:5px">🤖 Opening automatically on next bar.</div>
                    </div>""",unsafe_allow_html=True)
                else:
                    nr=ss.live_no_pos_reason
                    st.markdown(f"""<div class="pos-none">
                    <div style="font-size:.74rem;color:#546e7a">POSITION STATUS</div>
                    <div style="font-size:1.35rem;font-weight:600;color:#37474f;margin:4px 0">⏸ NO OPEN POSITION</div>
                    <div style="font-size:.80rem;color:#455a64;white-space:pre-line;word-break:break-word">{nr}</div>
                    </div>""",unsafe_allow_html=True)
            else:
                nr=ss.live_no_pos_reason
                st.markdown(f"""<div class="pos-none">
                <div style="font-size:.74rem;color:#546e7a">POSITION STATUS</div>
                <div style="font-size:1.35rem;font-weight:600;color:#37474f;margin:4px 0">⏸ NO OPEN POSITION</div>
                <div style="font-size:.80rem;color:#455a64;white-space:pre-line;word-break:break-word">{nr}</div>
                </div>""",unsafe_allow_html=True)

    render_pos(pos_ph)
    st.markdown("---")

    try:
        @st.fragment(run_every=2)
        def live_frag():
            ss=st.session_state
            if not ss.live_running: return
            new_cycle=live_engine_tick(symbol,interval,period,depth,sl_val,tgt_val,
                csl,ctgt,trail_sl_pts,trail_tgt_pts,dhan_on,dhan_client,
                trade_mode,entry_ot,exit_ot,product_type,segment,ce_sec_id,pe_sec_id,live_qty)
            if new_cycle: render_pos(pos_ph)

            ltp_=ss.live_ltp; ltp_ts=ss.live_ltp_ts or "—"
            real_pnl=ss.live_pnl  # realized only
            pos_=ss.live_position
            unreal=pos_.get("unreal_pnl",0) if pos_ else 0
            _,dclr,_=delay_ctx(ss.live_delay_s,interval)
            tc=st.columns(6)
            tc[0].metric("📊 LTP",f"{ltp_:,.2f}" if ltp_ else "—",delta=ltp_ts,delta_color="off")
            tc[1].metric("⏩ Data Age",f"{ss.live_delay_s}s",delta=ss.get("live_delay_ctx","")[:30],delta_color="off")
            tc[2].metric("🕒 IST",now_ist().strftime("%H:%M:%S"),delta=ss.live_last_candle_ist,delta_color="off")
            tc[3].metric("🔜 Next Check",ss.live_next_check_ist)
            tc[4].metric("💰 Realized P&L",f"Rs{real_pnl:,.2f}",delta=f"{len(ss.live_trades)} trades",delta_color="normal" if real_pnl>=0 else "inverse")
            tc[5].metric("📈 Unrealized",f"Rs{unreal:,.2f}" if pos_ else "No pos",delta="Open" if pos_ else "—",delta_color="normal" if unreal>=0 else "inverse")

            if ss.live_delay_s>=POLL_SECS.get(interval,3600)*2:
                st.info(f"ℹ️ {ss.get('live_delay_ctx','')}")

            ws_col,sig_col=st.columns([1.4,1])
            with ws_col:
                ws=ss.live_wave_state
                if ws:
                    dc={"BULLISH":"#4caf50","BEARISH":"#f44336","NEUTRAL":"#78909c"}.get(ws.get("direction","NEUTRAL"),"#78909c")
                    di={"BULLISH":"📈","BEARISH":"📉","NEUTRAL":"➡️"}.get(ws.get("direction","NEUTRAL"),"➡️")
                    st.markdown(f"""<div class="wave-card"><b style="color:#00e5ff">🌊 Current Wave</b>
<span style="color:{dc}">{di} {ws.get('current_wave','—')}</span>
<b style="color:#00e5ff">Next Expected</b>
<span style="color:#b0bec5">{ws.get('next_wave','—')}</span></div>""",unsafe_allow_html=True)
                    fibs=ws.get("fib_levels",{})
                    if fibs:
                        st.markdown("**📐 Wave-3 Fib Targets**")
                        for lbl_f,val_f in fibs.items():
                            bc2="#4caf50" if "1.618" in lbl_f else("#ab47bc" if "2.618" in lbl_f else "#ffb300")
                            st.markdown(f"""<div style="display:flex;justify-content:space-between;background:#060d14;border-left:3px solid {bc2};border-radius:0 4px 4px 0;padding:4px 9px;margin-bottom:3px;font-size:.80rem">
                            <span style="color:#78909c">{lbl_f}</span><span style="color:{bc2};font-family:'Share Tech Mono'">{val_f:,.2f}</span></div>""",unsafe_allow_html=True)
                    st.markdown(f"""<div class="wave-card" style="margin-top:8px">{ws.get('action','')}</div>""",unsafe_allow_html=True)
            with sig_col:
                sig_=ss.live_last_sig
                if sig_ and sig_.get("signal")!="HOLD":
                    s_=sig_["signal"]; sc_="#4caf50" if s_=="BUY" else "#f44336"; em_="🟢" if s_=="BUY" else "🔴"
                    rr_=sig_.get("rr",0)
                    mf_=sig_.get("mf_score",0); mf_c="#4caf50" if mf_>=0.7 else "#ffb300" if mf_>=0.55 else "#f44336"
                    st.markdown(f"""<div style="background:rgba(255,255,255,.03);border:2px solid {sc_};border-radius:10px;padding:14px 16px">
                    <div style="font-size:.73rem;color:#546e7a">LAST SIGNAL</div>
                    <div style="font-size:1.4rem;color:{sc_};font-weight:700">{em_} {s_}</div>
                    <div style="font-size:.79rem;color:#b0bec5">{sig_.get('pattern','—')}</div>
                    <div style="font-size:.77rem;color:#00bcd4">Conf {sig_.get('confidence',0):.0%} | MF <span style="color:{mf_c}">{mf_:.0%}</span></div>
                    <hr style="border-color:rgba(255,255,255,.08);margin:8px 0">
                    <div style="font-size:.78rem;color:#78909c">
                    Entry: <b style="color:#b0bec5">{sig_.get('entry_price',0):,.2f}</b> (limit price)<br>
                    SL: <b style="color:#ff7043">{sig_.get('sl',0):,.2f}</b><br>
                    Target: <b style="color:#66bb6a">{sig_.get('target',0):,.2f}</b><br>
                    R:R: <b style="color:#b0bec5">1:{rr_:.1f}</b>
                    </div>
                    <div style="font-size:.73rem;color:#455a64;margin-top:5px">{sig_.get('reason','—')[:120]}</div>
                    </div>""",unsafe_allow_html=True)
                    if sig_.get("factors"):
                        with st.expander("📊 8-Factor Breakdown",expanded=False):
                            for fk,fv in sig_["factors"].items():
                                st.caption(f"**{fk}**: {fv}")
                else:
                    nr=ss.live_no_pos_reason
                    st.markdown(f"""<div style="background:#060d14;border:1px solid #263238;border-radius:10px;padding:14px 16px">
                    <div style="font-size:.73rem;color:#546e7a">LAST SIGNAL</div>
                    <div style="font-size:1.1rem;color:#37474f;font-weight:600">⏸ HOLD</div>
                    <div style="font-size:.78rem;color:#37474f;margin-top:5px;line-height:1.55;word-break:break-word">{nr}</div>
                    </div>""",unsafe_allow_html=True)

            if ss._scan_df is not None:
                piv_=find_pivots(ss._scan_df.iloc[:-1],depth)
                st.plotly_chart(chart_waves(ss._scan_df,piv_,ss._scan_sig,symbol=symbol),use_container_width=True)

            h1c,h2c=st.columns(2)
            with h1c:
                if ss.live_signals:
                    st.markdown("##### 📋 Signals")
                    st.dataframe(pd.DataFrame(ss.live_signals).tail(8),use_container_width=True,height=140)
            with h2c:
                if ss.live_trades:
                    st.markdown("##### 🏁 Trades")
                    td=pd.DataFrame(ss.live_trades); st.dataframe(td.tail(8),use_container_width=True,height=140)
                    wns=(td["PnL Rs"]>0).sum(); tot=len(td); pnl_=td["PnL Rs"].sum()
                    st.caption(f"Win {wns}/{tot} ({wns/tot*100:.0f}%) | Realized Rs{pnl_:,.2f}")
            if ss.live_log:
                with st.expander("📜 Log",expanded=False):
                    st.code("\n".join(reversed(ss.live_log[-60:])),language=None)
            # Full Auto log
            if ss.auto_log:
                with st.expander("🤖 Auto Mode Log",expanded=False):
                    st.code("\n".join(ss.auto_log[-40:]),language=None)

        live_frag()
    except Exception as fe:
        st.warning(f"⚠️ Fragment unavailable: {fe}. Upgrade: `pip install streamlit --upgrade`")
        if st.button("🔍 Manual Scan",use_container_width=True):
            with st.spinner("Running…"):
                live_engine_tick(symbol,interval,period,depth,sl_val,tgt_val,
                    csl,ctgt,trail_sl_pts,trail_tgt_pts,dhan_on,dhan_client,
                    trade_mode,entry_ot,exit_ot,product_type,segment,ce_sec_id,pe_sec_id,live_qty)
            render_pos(pos_ph); st.rerun()

# ── BACKTEST ───────────────────────────────────────────────────────────────
with t_bt:
    bl,br=st.columns([1,2.6],gap="medium")
    with bl:
        st.markdown("### ⚙️ Config")
        st.markdown(f"""<div class="info-box">
        📈 <code>{symbol}</code> · <code>{interval}</code> · <code>{period}</code><br>
        🌊 Depth <code>{depth}</code> · Conf≥<code>{conf_thresh:.0%}</code><br>
        🛡 SL <code>{sl_lbl_sel}</code><br>
        🎯 Target <code>{tgt_lbl_sel}</code> · Rs<code>{capital:,}</code><br>
        <small style="color:#546e7a">Signal bar N → entry open bar N+1<br>
        Min R:R 1.2 enforced<br>8-factor confluence scoring<br>
        SL: Low(BUY)/High(SELL) first</small></div>""",unsafe_allow_html=True)
        if st.session_state.best_cfg_applied: st.success("✨ Optimized config active")
        if st.button("🚀 Run Backtest",type="primary",use_container_width=True):
            with st.spinner("Fetching…"):
                dbt=fetch_ohlcv(symbol,interval,period,md=1.5)
            if dbt is None or len(dbt)<50: st.error("Not enough data. Use longer period.")
            else:
                with st.spinner(f"Running on {len(dbt)} bars…"):
                    res=run_backtest(dbt,depth,sl_val,tgt_val,capital,csl,ctgt,trail_sl_pts,conf_thresh)
                    res.update({"df":dbt,"pivots":find_pivots(dbt,depth),"symbol":symbol,"interval":interval,"period":period})
                st.session_state.bt_results=res
                if "error" in res: st.error(res["error"])
                else: st.success(f"✅ {res['metrics']['Total Trades']} trades! Win Rate: {res['metrics']['Win Rate %']}%")
    with br:
        r=st.session_state.bt_results
        if r and "metrics" in r:
            m=r["metrics"]
            wr_c="#4caf50" if m["Win Rate %"]>=60 else "#ffb300" if m["Win Rate %"]>=50 else "#f44336"
            st.markdown(f"### Results — `{r.get('symbol','')}` · `{r.get('interval','')}` · `{r.get('period','')}`")
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Win Rate",f"{m['Win Rate %']}%",delta=f"{m['Wins']}W/{m['Losses']}L")
            c2.metric("Total Return",f"{m['Total Return %']}%",delta=f"Rs{m['Final Equity Rs']:,}")
            c3.metric("Profit Factor",str(m["Profit Factor"]))
            c4.metric("Max Drawdown",f"{m['Max Drawdown %']}%")
            c5,c6,c7,c8=st.columns(4)
            c5.metric("Sharpe",str(m["Sharpe Ratio"])); c6.metric("Trades",str(m["Total Trades"]))
            c7.metric("Avg Win",f"Rs{m['Avg Win Rs']:,}"); c8.metric("Avg Loss",f"Rs{m['Avg Loss Rs']:,}")
            eb=r.get("exit_breakdown",{})
            if eb:
                ea,eb2,ec2,ed=st.columns(4)
                ea.metric("SL Hits",str(eb.get("SL hits",0))); eb2.metric("Target Hits",str(eb.get("Target hits",0)))
                ec2.metric("Sig Reverse",str(eb.get("Signal Reverse",0))); ed.metric("Still Open",str(eb.get("Still open",0)))
            tc1,tc2,tc3=st.tabs(["🕯 Wave Chart","📈 Equity","📋 Trades"])
            with tc1: st.plotly_chart(chart_waves(r["df"],r["pivots"],_blank(),r["trades"],r["symbol"]),use_container_width=True)
            with tc2: st.plotly_chart(chart_equity(r["equity_curve"]),use_container_width=True)
            with tc3:
                st.dataframe(r["trades"],use_container_width=True,height=400)
                st.info("SL Verified ✅=Low≤SL(BUY)/High≥SL(SELL). TGT Verified ✅=High≥Target(BUY)/Low≤Target(SELL).")
                st.download_button("📥 CSV",data=r["trades"].to_csv(index=False),
                                   file_name=f"ew_bt_{symbol}_{interval}_{period}.csv",mime="text/csv")
        elif r and "error" in r: st.error(r["error"])
        else: st.markdown("<div style='text-align:center;padding:60px;color:#37474f'><h3>Run backtest to see results</h3></div>",unsafe_allow_html=True)

# ── OPTIMIZATION ───────────────────────────────────────────────────────────
with t_opt:
    ol,or_=st.columns([1,3],gap="medium")
    with ol:
        st.markdown("### ⚙️ Config")
        st.markdown(f"""<div class="info-box"><code>{symbol}</code> · <code>{interval}</code> · <code>{period}</code><br>
        Rs<code>{capital:,}</code> · Conf≥<code>{conf_thresh:.0%}</code><br><br>
        <b>80 combos</b>: 4×4×5 grid<br>
        <small style="color:#546e7a">8-factor scoring applied.<br>Failed combos skipped gracefully.</small></div>""",unsafe_allow_html=True)
        st.warning("⏳ ~80 backtests (~2–4 min with 8-factor scoring)")
        if st.button("🔬 Run Optimization",type="primary",use_container_width=True):
            with st.spinner("Fetching…"):
                dopt=fetch_ohlcv(symbol,interval,period,md=1.5)
            if dopt is None or len(dopt)<55: st.error("Not enough data.")
            else:
                prog_opt=st.progress(0,text="Optimizing…")
                with st.spinner("Running 80 combinations with 8-factor scoring…"):
                    odf=run_optimization(dopt,capital,csl,ctgt,trail_sl_pts,prog_obj=prog_opt,prog_offset=0,prog_total=1)
                prog_opt.empty()
                st.session_state.opt_results={"df":odf,"symbol":symbol,"interval":interval,"period":period}
                if odf.empty: st.warning("No results. Try longer period or reduce confidence threshold.")
                else: st.success(f"✅ Best Score: {odf['Score'].iloc[0]:.2f} | Win: {odf['Win %'].iloc[0]:.0f}%")
    with or_:
        optr=st.session_state.opt_results
        if optr and optr.get("df") is not None and not optr["df"].empty:
            odf=optr["df"]
            st.markdown(f"### Results — `{optr['symbol']}` · `{optr['interval']}` · `{optr['period']}`")
            br_=odf.iloc[0]
            b1,b2,b3,b4,b5=st.columns(5)
            b1.metric("Best Depth",str(int(br_["Depth"]))); b2.metric("Best SL",str(br_["SL"]))
            b3.metric("Best Target",str(br_["Target"])); b4.metric("Win Rate",f"{br_['Win %']:.0f}%")
            b5.metric("Best Score",f"{br_['Score']:.2f}")
            st.markdown("---")
            ac1,ac2=st.columns([1.6,2.4])
            with ac1:
                nr=st.number_input("Apply top N",min_value=1,max_value=min(5,len(odf)),value=1,step=1)
                ar_=odf.iloc[int(nr)-1]
            with ac2:
                st.markdown(f"""<div class="best-cfg">
                <b style="color:#00e5ff">Config #{int(nr)}</b><br>
                Depth <b>{int(ar_['Depth'])}</b> · SL <b>{ar_['SL']}</b> · Target <b>{ar_['Target']}</b><br>
                Win <b>{ar_['Win %']:.0f}%</b> · Return <b>{ar_['Return %']:.1f}%</b> · Sharpe <b>{ar_['Sharpe']:.2f}</b> · Score <b>{ar_['Score']:.2f}</b>
                </div>""",unsafe_allow_html=True)
            if st.button(f"✨ Apply Config #{int(nr)} → Sidebar + Backtest + Live",type="primary",use_container_width=True):
                st.session_state.update({"applied_depth":int(ar_["Depth"]),"applied_sl_lbl":sl_lbl(ar_["SL"]),
                                          "applied_tgt_lbl":tgt_lbl(ar_["Target"]),"best_cfg_applied":True})
                st.success(f"✅ Applied! Depth={int(ar_['Depth'])} · SL={sl_lbl(ar_['SL'])} · Target={tgt_lbl(ar_['Target'])}")
                time.sleep(0.4); st.rerun()
            st.markdown("---")
            oc1,oc2=st.tabs(["📊 Scatter","📋 Table"])
            with oc1:
                st.plotly_chart(chart_opt_scatter(odf),use_container_width=True)
                st.caption("Top-left bubble = best. Bubble = WinRate. Color = Score.")
            with oc2:
                def _hl(row):
                    if row.name==0: return["background-color:rgba(0,229,255,.18)"]*len(row)
                    elif row.name<3: return["background-color:rgba(0,229,255,.08)"]*len(row)
                    return[""]*len(row)
                st.dataframe(odf.style.apply(_hl,axis=1),use_container_width=True,height=500)
                st.download_button("📥 CSV",data=odf.to_csv(index=False),
                                   file_name=f"ew_opt_{symbol}_{interval}_{period}.csv",mime="text/csv")
        else:
            st.markdown("<div style='text-align:center;padding:60px;color:#37474f'><h3>Run optimization to see results</h3></div>",unsafe_allow_html=True)

# ── HELP ───────────────────────────────────────────────────────────────────
with t_help:
    st.markdown("## 📖 Help — Elliott Wave Algo Trader v5.3")
    h1,h2=st.columns(2,gap="large")
    with h1:
        st.markdown(f"""
### 🤖 Full Auto Mode

1. Enable **🤖 Full Auto Mode** checkbox in sidebar
2. Click **🚀 Run Full Auto Analysis**
3. System automatically:
   - Scans {len(AUTO_SCAN_GRID)} TF/period combinations
   - Runs 80-combo optimization on each
   - Scores by win rate + return + Sharpe + active signal
   - Applies best config to sidebar + backtest + live
4. Go to **Live Trading tab** → click **▶ Start Live**
5. System trades automatically. No other action needed.

Best config is shown in a banner on the Live Trading tab.

---

### 📈 Accuracy Improvements in v5.3

8-factor confluence scoring:
| Factor | Weight | Why |
|--------|--------|-----|
| RSI | 18% | Oversold/overbought zone |
| Stochastic RSI | 12% | Momentum timing |
| MACD Histogram | 15% | Trend direction + crossover |
| EMA Trend | 15% | Price above/below EMA 9/21/50 |
| Fib Retracement | 18% | Golden ratio 61.8% = strongest |
| Bollinger Band | 10% | Oversold/overbought zone |
| Volume Surge | 7% | Confirms pivot momentum |
| W1/ATR Ratio | 5% | Minimum wave size validation |

Additional filters:
- **Minimum confidence {int(MIN_CONFIDENCE_THRESHOLD*100)}%** (adjustable in sidebar)
- **Wave-3 > Wave-1** (core Elliott rule enforced)
- **Minimum R:R 1.2:1** (rejects bad setups)

Combined: EW pattern score (55%) + Multi-factor (45%)

---

### 🔑 Default Credentials
```
Client ID: {DEFAULT_CLIENT_ID}
Token: {DEFAULT_ACCESS_TOKEN[:40]}…
```
Change `DEFAULT_CLIENT_ID` and `DEFAULT_ACCESS_TOKEN` at top of file.
        """)
    with h2:
        st.markdown("""
### 🏦 Dhan: Stocks vs Options

**Stocks mode:**
- INTRADAY (default) or DELIVERY
- Exchange: NSE / BSE
- Enter Security ID (e.g. 1333 = HDFC Bank)

**Options mode:**
- NSE_FNO / BSE_FNO
- CE ID → used when BUY signal fires
- PE ID → used when SELL signal fires
- Lot size (Nifty=50, BankNifty=15)

**Order types:**
- Entry LIMIT: price auto-taken from signal entry level
- Exit MARKET: immediate fill at SL/Target price
- Exit LIMIT: price auto-taken from SL/Target level

---

### 🔄 Trailing SL/Target
- **Trailing SL**: Ratchets in profit direction, never reverses
- **Trailing Target**: Display only — never triggers exit

---

### 📊 Confidence Threshold
Higher threshold = fewer but better signals.
- 60%: More signals, lower accuracy
- 68%: Balanced (default)
- 75%: Fewer signals, higher accuracy
- 80%+: Very selective, may miss setups

---

### ⚠️ Common Issues
| Problem | Solution |
|---------|---------|
| No trades in backtest | Lower conf threshold or use longer period |
| Optimization 0 results | Try 1d/1y with lower threshold |
| Full Auto takes too long | Normal — scanning 8 TFs + 80 combos each |
| Position not opening | Check "No Open Position" reason text |
| dhanhq missing | `pip install dhanhq` |
        """)
    st.markdown("---")
    st.markdown("""<div style="text-align:center;color:#37474f;font-size:.81rem;padding:8px">
    🌊 Elliott Wave Algo Trader v5.3 · 8-Factor Scoring · Full Auto Mode ·
    <b style="color:#f44336">Not financial advice. Paper trade first.</b>
    </div>""",unsafe_allow_html=True)
