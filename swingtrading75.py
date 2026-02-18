"""
AlphaEdge v3 - Professional Trading Platform
Install: pip install streamlit yfinance pandas numpy plotly scipy ta requests
Run:     streamlit run trading_platform.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime
import warnings, math, time, requests
from scipy.stats import norm

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AlphaEdge v3", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

for k, v in {
    "theme":"White","trading_active":False,"trade_history":[],"open_positions":{},
    "live_realized_pnl":0.0,"live_unrealized_pnl":0.0,"paper_capital":100000.0,
    "last_fetch_time":{},"bt_run":False,"last_auto_refresh":0.0,"auto_refresh_interval":30,
}.items():
    if k not in st.session_state: st.session_state[k]=v

THEME=st.session_state["theme"]
if THEME=="White":
    T={"BG":"#FFFFFF","CARD_BG":"#F3F4F6","CARD_BG2":"#E9ECF0","BORDER":"#D1D5DB",
       "TEXT":"#111827","TEXT_MUTED":"#374151","TEXT_FAINT":"#6B7280",
       "INPUT_BG":"#FFFFFF","INPUT_TEXT":"#111827","INPUT_BORDER":"#9CA3AF",
       "SIDEBAR_BG":"#F9FAFB","SIDEBAR_BORDER":"#E5E7EB","PLOT_BG":"#FFFFFF",
       "PLOT_PAPER":"#F3F4F6","GRID":"#E5E7EB","ACCENT":"#F59E0B",
       "GREEN":"#059669","RED":"#DC2626","BLUE":"#2563EB","PURPLE":"#7C3AED",
       "ORANGE":"#EA580C","METRIC_VAL":"#1D4ED8","NAV":"#374151"}
else:
    T={"BG":"#0A0E1A","CARD_BG":"#111827","CARD_BG2":"#1F2937","BORDER":"#374151",
       "TEXT":"#F9FAFB","TEXT_MUTED":"#D1D5DB","TEXT_FAINT":"#9CA3AF",
       "INPUT_BG":"#1F2937","INPUT_TEXT":"#F9FAFB","INPUT_BORDER":"#4B5563",
       "SIDEBAR_BG":"#0D1117","SIDEBAR_BORDER":"#1F2937","PLOT_BG":"#111827",
       "PLOT_PAPER":"#0A0E1A","GRID":"#1F2937","ACCENT":"#F59E0B",
       "GREEN":"#10B981","RED":"#EF4444","BLUE":"#60A5FA","PURPLE":"#A78BFA",
       "ORANGE":"#FB923C","METRIC_VAL":"#F59E0B","NAV":"#F9FAFB"}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*, html, body, [class*="css"] {{ font-family: 'Space Grotesk', sans-serif !important; }}
.stApp, .main, .block-container {{ background-color:{T['BG']} !important; color:{T['TEXT']} !important; }}
[data-testid="collapsedControl"] {{ color:transparent !important; font-size:0 !important; width:32px; }}
[data-testid="collapsedControl"]::after {{ content:"☰"; color:{T['NAV']} !important; font-size:22px !important; }}
[data-testid="stSidebar"] {{ background:{T['SIDEBAR_BG']} !important; border-right:1px solid {T['SIDEBAR_BORDER']}; }}
[data-testid="stSidebar"] * {{ color:{T['TEXT']} !important; }}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] small {{ color:{T['TEXT_MUTED']} !important; }}
p,span,div,label,h1,h2,h3,h4,h5,h6,.stMarkdown {{ color:{T['TEXT']} !important; }}
small {{ color:{T['TEXT_MUTED']} !important; }}
input,textarea {{ background-color:{T['INPUT_BG']} !important; color:{T['INPUT_TEXT']} !important; border:1px solid {T['INPUT_BORDER']} !important; border-radius:6px !important; }}
input:focus {{ border-color:{T['ACCENT']} !important; }}
.stNumberInput input {{ background:{T['INPUT_BG']} !important; color:{T['INPUT_TEXT']} !important; }}
.stSelectbox > div > div {{ background:{T['INPUT_BG']} !important; color:{T['INPUT_TEXT']} !important; border:1px solid {T['INPUT_BORDER']} !important; }}
.stSelectbox svg {{ fill:{T['TEXT_MUTED']} !important; }}
[data-baseweb="popover"] *,[data-baseweb="menu"] * {{ background:{T['CARD_BG']} !important; color:{T['TEXT']} !important; }}
[data-baseweb="option"]:hover {{ background:{T['CARD_BG2']} !important; }}
.stSlider > div > div > div {{ background:{T['ACCENT']} !important; }}
.stSlider [role="slider"] {{ background:{T['ACCENT']} !important; border-color:{T['ACCENT']} !important; }}
.stRadio label,.stCheckbox label {{ color:{T['TEXT']} !important; }}
.stTabs [data-baseweb="tab-list"] {{ background:{T['CARD_BG']}; border-radius:12px; padding:4px; border:1px solid {T['BORDER']}; gap:4px; flex-wrap:wrap; }}
.stTabs [data-baseweb="tab"] {{ border-radius:8px; color:{T['TEXT_MUTED']} !important; font-weight:600; font-size:13px; padding:7px 14px; }}
.stTabs [aria-selected="true"] {{ background:{T['ACCENT']} !important; color:#000 !important; font-weight:700; }}
.stTabs [data-testid="stTabPanel"] {{ background:transparent !important; }}
[data-testid="metric-container"] {{ background:{T['CARD_BG']} !important; border:1px solid {T['BORDER']}; border-radius:10px; padding:10px 14px; }}
[data-testid="stMetricLabel"] {{ color:{T['TEXT_MUTED']} !important; font-size:11px; }}
[data-testid="stMetricValue"] {{ color:{T['METRIC_VAL']} !important; font-family:'JetBrains Mono',monospace !important; font-size:18px !important; font-weight:700 !important; }}
[data-testid="stMetricDelta"] {{ font-size:11px !important; }}
.stDataFrame,.stDataFrame * {{ background:{T['CARD_BG']} !important; color:{T['TEXT']} !important; }}
.stDataFrame thead th {{ background:{T['CARD_BG2']} !important; color:{T['ACCENT']} !important; font-weight:700 !important; }}
.stButton > button {{ background:linear-gradient(135deg,{T['ACCENT']},#D97706) !important; color:#000 !important; font-weight:700 !important; border:none !important; border-radius:8px !important; padding:9px 18px !important; font-size:13px !important; transition:all 0.2s !important; }}
.stButton > button:hover {{ transform:translateY(-1px); box-shadow:0 4px 16px rgba(245,158,11,0.5) !important; }}
.alpha-card {{ background:{T['CARD_BG']}; border:1px solid {T['BORDER']}; border-radius:12px; padding:14px 18px; margin:6px 0; color:{T['TEXT']}; }}
.section-hdr {{ font-size:17px; font-weight:700; color:{T['ACCENT']}; border-bottom:2px solid {T['BORDER']}; padding-bottom:7px; margin:16px 0 12px; }}
.tf-badge {{ display:inline-block; background:{T['CARD_BG2']}; color:{T['ACCENT']}; border:1px solid rgba(245,158,11,0.4); border-radius:6px; padding:2px 8px; font-size:11px; font-weight:700; font-family:'JetBrains Mono',monospace; margin:2px; }}
.stAlert,[data-testid="stAlert"] {{ background:{T['CARD_BG']} !important; color:{T['TEXT']} !important; border-color:{T['BORDER']} !important; }}
.stProgress > div > div {{ background:{T['ACCENT']} !important; }}
</style>
""", unsafe_allow_html=True)

def badge(t): return f'<span class="tf-badge">{t}</span>'
def section(t): st.markdown(f'<div class="section-hdr">{t}</div>', unsafe_allow_html=True)
def acard(h): st.markdown(f'<div class="alpha-card">{h}</div>', unsafe_allow_html=True)
NIFTY50_STOCKS = {
    "Adani Enterprises":"ADANIENT.NS","Adani Ports":"ADANIPORTS.NS",
    "Apollo Hospitals":"APOLLOHOSP.NS","Asian Paints":"ASIANPAINT.NS",
    "Axis Bank":"AXISBANK.NS","Bajaj Auto":"BAJAJ-AUTO.NS",
    "Bajaj Finance":"BAJFINANCE.NS","Bajaj Finserv":"BAJAJFINSV.NS",
    "BPCL":"BPCL.NS","Bharti Airtel":"BHARTIARTL.NS",
    "Britannia":"BRITANNIA.NS","Cipla":"CIPLA.NS",
    "Coal India":"COALINDIA.NS","Divi Labs":"DIVISLAB.NS",
    "Dr Reddy":"DRREDDY.NS","Eicher Motors":"EICHERMOT.NS",
    "Grasim":"GRASIM.NS","HCL Tech":"HCLTECH.NS",
    "HDFC Bank":"HDFCBANK.NS","HDFC Life":"HDFCLIFE.NS",
    "Hero MotoCorp":"HEROMOTOCO.NS","Hindalco":"HINDALCO.NS",
    "HUL":"HINDUNILVR.NS","ICICI Bank":"ICICIBANK.NS",
    "ITC":"ITC.NS","IndusInd Bank":"INDUSINDBK.NS",
    "Infosys":"INFY.NS","JSW Steel":"JSWSTEEL.NS",
    "Kotak Bank":"KOTAKBANK.NS","L&T":"LT.NS",
    "LTIMindtree":"LTIM.NS","M&M":"M&M.NS",
    "Maruti":"MARUTI.NS","Nestle India":"NESTLEIND.NS",
    "NTPC":"NTPC.NS","ONGC":"ONGC.NS",
    "Power Grid":"POWERGRID.NS","Reliance":"RELIANCE.NS",
    "SBI Life":"SBILIFE.NS","Shriram Finance":"SHRIRAMFIN.NS",
    "SBI":"SBIN.NS","Sun Pharma":"SUNPHARMA.NS",
    "TCS":"TCS.NS","Tata Consumer":"TATACONSUM.NS",
    "Tata Motors":"TATAMOTORS.NS","Tata Steel":"TATASTEEL.NS",
    "Tech Mahindra":"TECHM.NS","Titan":"TITAN.NS",
    "UltraTech Cement":"ULTRACEMCO.NS","Wipro":"WIPRO.NS",
}

ASSET_MAP = {
    "Nifty 50":"^NSEI","BankNifty":"^NSEBANK","Sensex":"^BSESN",
    "Nifty IT":"^CNXIT","Nifty Pharma":"^CNXPHARMA","Nifty Midcap 100":"^CNXMIDCAP",
    **NIFTY50_STOCKS,
    "Bitcoin (BTC)":"BTC-USD","Ethereum (ETH)":"ETH-USD","Solana (SOL)":"SOL-USD",
    "USD/INR":"USDINR=X","EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X","USD/JPY":"USDJPY=X",
    "Gold":"GC=F","Silver":"SI=F","Crude Oil (WTI)":"CL=F","Natural Gas":"NG=F",
    "Custom Ticker":"CUSTOM","All Nifty 50 Stocks":"NIFTY50_ALL",
}

ASSET_GROUPS = {
    "Indian Indices":["Nifty 50","BankNifty","Sensex","Nifty IT","Nifty Pharma","Nifty Midcap 100"],
    "Nifty 50 Stocks":list(NIFTY50_STOCKS.keys()),
    "All Nifty 50 Batch":["All Nifty 50 Stocks"],
    "Crypto":["Bitcoin (BTC)","Ethereum (ETH)","Solana (SOL)"],
    "Forex":["USD/INR","EUR/USD","GBP/USD","USD/JPY"],
    "Commodities":["Gold","Silver","Crude Oil (WTI)","Natural Gas"],
    "Custom":["Custom Ticker"],
}

STRATEGIES = {
    "Trend+Structure+Momentum (Pro)":"TSM",
    "ORB - Opening Range Breakout":"ORB",
    "VWAP + RSI Reversal/Trend":"VWAP_RSI",
    "Swing: EMA + MACD + RSI":"SWING",
    "Scalping: EMA9 + RSI + Volume":"SCALP",
    "Combined (All Signals)":"COMBINED",
}

NSE_OC_MAP = {"^NSEI":"NIFTY","^NSEBANK":"BANKNIFTY","^BSESN":"SENSEX"}
FETCH_DELAY = 1.5

def _wait(key):
    now=time.time(); last=st.session_state["last_fetch_time"].get(key,0)
    gap=now-last
    if gap<FETCH_DELAY: time.sleep(FETCH_DELAY-gap)
    st.session_state["last_fetch_time"][key]=time.time()

@st.cache_data(ttl=300,show_spinner=False)
def fetch_data(ticker,period="1y",interval="1d"):
    try:
        _wait(f"h_{ticker}")
        df=yf.download(ticker,period=period,interval=interval,auto_adjust=True,progress=False,threads=False)
        if df.empty: return pd.DataFrame()
        df.columns=[c[0] if isinstance(c,tuple) else c for c in df.columns]
        df.index=pd.to_datetime(df.index)
        return df.dropna()
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=45,show_spinner=False)
def fetch_live(ticker,interval="15m",period="5d"):
    try:
        _wait(f"l_{ticker}")
        df=yf.download(ticker,period=period,interval=interval,auto_adjust=True,progress=False,threads=False)
        if df.empty: return pd.DataFrame()
        df.columns=[c[0] if isinstance(c,tuple) else c for c in df.columns]
        df.index=pd.to_datetime(df.index)
        return df.dropna()
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=90,show_spinner=False)
def fetch_nse_oc(sym):
    hdr={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
         "Accept":"application/json","Referer":"https://www.nseindia.com/"}
    try:
        s=requests.Session(); s.get("https://www.nseindia.com",headers=hdr,timeout=8); time.sleep(1.5)
        r=s.get(f"https://www.nseindia.com/api/option-chain-indices?symbol={sym}",headers=hdr,timeout=10)
        if r.status_code==200: return r.json()
        r2=s.get(f"https://www.nseindia.com/api/option-chain-equities?symbol={sym}",headers=hdr,timeout=10)
        if r2.status_code==200: return r2.json()
    except Exception: pass
    return {}

def parse_nse_oc(data,spot):
    rows=[]
    try:
        for rec in data.get("records",{}).get("data",[]):
            ce=rec.get("CE",{}); pe=rec.get("PE",{})
            rows.append({
                "Strike":rec.get("strikePrice",0),
                "CE_LTP":ce.get("lastPrice",0),"CE_IV":ce.get("impliedVolatility",0),
                "CE_OI":ce.get("openInterest",0),"CE_ChgOI":ce.get("changeinOpenInterest",0),
                "CE_Delta":ce.get("delta",""),"CE_Gamma":ce.get("gamma",""),
                "CE_Theta":ce.get("theta",""),"CE_Vega":ce.get("vega",""),
                "PE_LTP":pe.get("lastPrice",0),"PE_IV":pe.get("impliedVolatility",0),
                "PE_OI":pe.get("openInterest",0),"PE_ChgOI":pe.get("changeinOpenInterest",0),
                "PE_Delta":pe.get("delta",""),"PE_Gamma":pe.get("gamma",""),
                "PE_Theta":pe.get("theta",""),"PE_Vega":pe.get("vega",""),
            })
    except Exception: pass
    if not rows: return pd.DataFrame()
    df=pd.DataFrame(rows).sort_values("Strike")
    df["d"]=(df["Strike"]-spot).abs()
    return df.nsmallest(21,"d").sort_values("Strike").drop("d",axis=1)

def bs_price(S,K,T,r,sigma,typ="CE"):
    if T<=0 or sigma<=0:
        return (max(S-K,0) if typ=="CE" else max(K-S,0)),0,0,0,0
    d1=(math.log(S/K)+(r+.5*sigma**2)*T)/(sigma*math.sqrt(T)); d2=d1-sigma*math.sqrt(T)
    if typ=="CE": p=S*norm.cdf(d1)-K*math.exp(-r*T)*norm.cdf(d2); dlt=norm.cdf(d1)
    else: p=K*math.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1); dlt=-norm.cdf(-d1)
    gam=norm.pdf(d1)/(S*sigma*math.sqrt(T))
    tht=(-(S*norm.pdf(d1)*sigma)/(2*math.sqrt(T))-r*K*math.exp(-r*T)*(norm.cdf(d2) if typ=="CE" else norm.cdf(-d2)))/365
    vga=S*norm.pdf(d1)*math.sqrt(T)/100
    return round(p,2),round(dlt,4),round(gam,6),round(tht,2),round(vga,4)

def get_live_iv(ticker,spot):
    sym=NSE_OC_MAP.get(ticker)
    if sym:
        try:
            d=fetch_nse_oc(sym); atm=round(spot/100)*100
            for rec in d.get("records",{}).get("data",[]):
                if rec.get("strikePrice")==atm:
                    iv=rec.get("CE",{}).get("impliedVolatility",0)
                    if iv and iv>0: return iv/100
        except Exception: pass
    try:
        df=fetch_data(ticker,"3mo","1d")
        if not df.empty and len(df)>20:
            return float(np.log(df["Close"]/df["Close"].shift(1)).dropna().std()*np.sqrt(252))
    except Exception: pass
    return 0.18
def compute_indicators(df):
    d=df.copy()
    for p in [9,20,50,200]: d[f"EMA{p}"]=d["Close"].ewm(span=p,adjust=False).mean()
    d["SMA20"]=d["Close"].rolling(20).mean(); d["SMA50"]=d["Close"].rolling(50).mean()
    delta=d["Close"].diff(); gain=delta.clip(lower=0).rolling(14).mean()
    loss=(-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"]=100-100/(1+gain/loss.replace(0,np.nan))
    e12=d["Close"].ewm(12,adjust=False).mean(); e26=d["Close"].ewm(26,adjust=False).mean()
    d["MACD"]=e12-e26; d["MACD_Signal"]=d["MACD"].ewm(9,adjust=False).mean()
    d["MACD_Hist"]=d["MACD"]-d["MACD_Signal"]
    d["BB_Mid"]=d["Close"].rolling(20).mean(); std=d["Close"].rolling(20).std()
    d["BB_Upper"]=d["BB_Mid"]+2*std; d["BB_Lower"]=d["BB_Mid"]-2*std
    d["BB_Width"]=(d["BB_Upper"]-d["BB_Lower"])/d["BB_Mid"].replace(0,np.nan)
    tr=pd.concat([d["High"]-d["Low"],(d["High"]-d["Close"].shift()).abs(),(d["Low"]-d["Close"].shift()).abs()],axis=1).max(axis=1)
    d["ATR"]=tr.rolling(14).mean()
    vol=d["Volume"].replace(0,np.nan); d["TP"]=(d["High"]+d["Low"]+d["Close"])/3
    d["VWAP"]=(d["TP"]*vol).rolling(14).sum()/vol.rolling(14).sum()
    d["Vol_MA20"]=vol.rolling(20).mean(); d["Vol_Ratio"]=vol/d["Vol_MA20"]
    lo14=d["Low"].rolling(14).min(); hi14=d["High"].rolling(14).max()
    d["Stoch_K"]=100*(d["Close"]-lo14)/(hi14-lo14).replace(0,np.nan)
    d["Stoch_D"]=d["Stoch_K"].rolling(3).mean()
    d["Pivot"]=(d["High"]+d["Low"]+d["Close"])/3
    d["R1"]=2*d["Pivot"]-d["Low"]; d["S1"]=2*d["Pivot"]-d["High"]
    d["R2"]=d["Pivot"]+(d["High"]-d["Low"]); d["S2"]=d["Pivot"]-(d["High"]-d["Low"])
    lr=np.log(d["Close"]/d["Close"].shift(1)); d["HV20"]=lr.rolling(20).std()*np.sqrt(252)*100
    d["Ret_Pts"]=d["Close"].diff(); d["Ret_Pct"]=d["Close"].pct_change()*100
    return d

def strategy_tsm(df,am=1.5,rr=2.0,tsl=True,tp=1.5):
    d=df.copy(); bull=(d["Close"]>d["EMA20"])&(d["EMA20"]>d["EMA50"])
    bear=(d["Close"]<d["EMA20"])&(d["EMA20"]<d["EMA50"]); vol=d["Vol_Ratio"]>=1.5
    d["Signal"]=0
    d.loc[bull&(d["High"]>d["High"].shift(1))&vol&d["RSI"].between(50,75)&(d["MACD"]>d["MACD_Signal"]),"Signal"]=1
    d.loc[bear&(d["Low"]<d["Low"].shift(1))&vol&d["RSI"].between(25,50)&(d["MACD"]<d["MACD_Signal"]),"Signal"]=-1
    d["Strategy"]="TSM"; return d

def strategy_orb(df,am=1.0,rr=1.5,tsl=True,tp=1.2):
    d=df.copy(); vol=d["Vol_Ratio"]>1.8; d["Signal"]=0
    d.loc[(d["High"]>d["High"].shift(1))&vol&(d["Close"]>d["EMA20"])&(d["RSI"]>50),"Signal"]=1
    d.loc[(d["Low"]<d["Low"].shift(1))&vol&(d["Close"]<d["EMA20"])&(d["RSI"]<50),"Signal"]=-1
    d["Strategy"]="ORB"; return d

def strategy_vwap_rsi(df,am=1.2,rr=1.8,tsl=True,tp=1.0):
    d=df.copy(); d["Signal"]=0
    nbv=(d["Low"]<=d["VWAP"]*1.003)&(d["Close"]>d["VWAP"]*0.997)
    nbs=(d["High"]>=d["VWAP"]*0.997)&(d["Close"]<d["VWAP"]*1.003)
    d.loc[(d["Close"]>d["VWAP"])&nbv&d["RSI"].between(40,60)&(d["EMA20"]>d["EMA50"]),"Signal"]=1
    d.loc[(d["Close"]<d["VWAP"])&nbs&d["RSI"].between(40,60)&(d["EMA20"]<d["EMA50"]),"Signal"]=-1
    d["Strategy"]="VWAP_RSI"; return d

def strategy_swing(df,am=2.0,rr=2.5,tsl=True,tp=2.0):
    d=df.copy()
    gc=(d["EMA20"]>d["EMA50"])&(d["EMA20"].shift(1)<=d["EMA50"].shift(1))
    dc=(d["EMA20"]<d["EMA50"])&(d["EMA20"].shift(1)>=d["EMA50"].shift(1))
    mu=(d["MACD"]>d["MACD_Signal"])&(d["MACD"].shift(1)<=d["MACD_Signal"].shift(1))
    md=(d["MACD"]<d["MACD_Signal"])&(d["MACD"].shift(1)>=d["MACD_Signal"].shift(1))
    vol=d["Vol_Ratio"]>1.3; d["Signal"]=0
    d.loc[(gc|mu)&d["RSI"].between(45,70)&vol,"Signal"]=1
    d.loc[(dc|md)&d["RSI"].between(30,55)&vol,"Signal"]=-1
    d["Strategy"]="SWING"; return d

def strategy_scalp(df,am=0.5,rr=1.5,tsl=True,tp=0.5):
    d=df.copy()
    cu=(d["EMA9"]>d["EMA20"])&(d["EMA9"].shift(1)<=d["EMA20"].shift(1))
    cd=(d["EMA9"]<d["EMA20"])&(d["EMA9"].shift(1)>=d["EMA20"].shift(1))
    vb=d["Vol_Ratio"]>2.0; d["Signal"]=0
    d.loc[cu&(d["RSI"]>52)&vb,"Signal"]=1
    d.loc[cd&(d["RSI"]<48)&vb,"Signal"]=-1
    d["Strategy"]="SCALP"; return d

def strategy_combined(df,am=1.5,rr=2.0,tsl=True,tp=1.5):
    votes=(strategy_tsm(df)["Signal"]+strategy_orb(df)["Signal"]+
           strategy_vwap_rsi(df)["Signal"]+strategy_swing(df)["Signal"]+strategy_scalp(df)["Signal"])
    d=df.copy(); d["Signal"]=0
    d.loc[votes>=2,"Signal"]=1; d.loc[votes<=-2,"Signal"]=-1
    d["Strategy"]="COMBINED"; return d

def get_strat(df,key,av,rr,tsl,tp):
    return {"TSM":strategy_tsm,"ORB":strategy_orb,"VWAP_RSI":strategy_vwap_rsi,
            "SWING":strategy_swing,"SCALP":strategy_scalp,"COMBINED":strategy_combined}[key](df,av,rr,tsl,tp)

def calc_sl_tgt(price,atr,direction,sl_type,sl_val,rr):
    if sl_type=="Points-based": dist=sl_val
    elif sl_type=="Percentage-based": dist=price*sl_val/100
    else: dist=sl_val*atr
    return round(price-direction*dist,2),round(price+direction*dist*rr,2)

def run_backtest(df_sig,capital,risk_pct,atr_sl,rr,tsl,tp,ttgt,ttgt_pct,sl_type,sl_val):
    trades=[]; equity=[capital]; eq_dates=[df_sig.index[0]]
    cash=capital; pos=0; entry_px=sl=tgt=tsl_px=ttgt_px=0.0; entry_date=None
    df=df_sig.dropna(subset=["ATR","Signal"]).copy()
    for i in range(1,len(df)):
        row=df.iloc[i]; price=float(row["Close"]); atr=max(float(row["ATR"]),price*0.001)
        if pos!=0:
            if tsl:
                if pos==1: tsl_px=max(tsl_px,price*(1-tp/100)); eff_sl=max(sl,tsl_px)
                else: tsl_px=min(tsl_px,price*(1+tp/100)); eff_sl=min(sl,tsl_px)
            else: eff_sl=sl
            if ttgt:
                if pos==1: ttgt_px=max(ttgt_px,price*(1+ttgt_pct/100))
                else: ttgt_px=min(ttgt_px,price*(1-ttgt_pct/100))
            ex_px=ex_rsn=None; eff_tgt=ttgt_px if ttgt and ttgt_px else tgt
            if pos==1:
                if row["Low"]<=eff_sl: ex_px=eff_sl; ex_rsn="Trail SL" if tsl else "SL Hit"
                elif row["High"]>=eff_tgt: ex_px=eff_tgt; ex_rsn="Trail Tgt" if ttgt else "Target"
            else:
                if row["High"]>=eff_sl: ex_px=eff_sl; ex_rsn="Trail SL" if tsl else "SL Hit"
                elif row["Low"]<=eff_tgt: ex_px=eff_tgt; ex_rsn="Trail Tgt" if ttgt else "Target"
            if ex_px is None and row["Signal"]==-pos: ex_px=price; ex_rsn="Signal Rev"
            if ex_px is not None:
                pnl_pct=(ex_px-entry_px)/entry_px*pos*100
                qty=(cash*risk_pct/100)/(abs(entry_px-sl)+1e-9)
                pnl_inr=(ex_px-entry_px)*pos*qty; cash+=pnl_inr
                trades.append({"Entry Date":entry_date,"Exit Date":row.name,
                    "Direction":"LONG" if pos==1 else "SHORT",
                    "Entry":round(entry_px,2),"Exit":round(ex_px,2),
                    "SL":round(sl,2),"Target":round(tgt,2),
                    "P&L %":round(pnl_pct,2),"P&L Rs":round(pnl_inr,2),
                    "Capital":round(cash,2),"Exit Reason":ex_rsn,"Strategy":row.get("Strategy","")})
                pos=0; tsl_px=ttgt_px=0.0
        if pos==0 and row["Signal"]!=0:
            pos=int(row["Signal"]); entry_px=price; entry_date=row.name
            sl,tgt=calc_sl_tgt(price,atr,pos,sl_type,sl_val,rr)
            tsl_px=price*(1-tp/100) if pos==1 else price*(1+tp/100); ttgt_px=tgt
        equity.append(cash); eq_dates.append(row.name)
    if not trades:
        return {"trades":pd.DataFrame(),"equity":pd.Series(equity,index=eq_dates),
                "metrics":{},"drawdown":pd.Series([0]*len(equity),index=eq_dates)}
    tdf=pd.DataFrame(trades); wins=tdf[tdf["P&L %"]>0]; losses=tdf[tdf["P&L %"]<=0]
    wr=len(wins)/len(tdf)*100
    pf=wins["P&L Rs"].sum()/abs(losses["P&L Rs"].sum()) if losses["P&L Rs"].sum()<0 else float("inf")
    eq_s=pd.Series(equity,index=eq_dates); dd=(eq_s-eq_s.cummax())/eq_s.cummax()*100
    nd=max((df.index[-1]-df.index[0]).days,1); cagr=((cash/capital)**(365/nd)-1)*100
    sr=0.0
    if len(tdf)>1:
        r=tdf["P&L %"]/100; sr=r.mean()/(r.std()+1e-9)*np.sqrt(252)
    return {"trades":tdf,"equity":eq_s,"drawdown":dd,"metrics":{
        "Total Trades":len(tdf),"Win Rate %":round(wr,1),
        "Avg Win %":round(wins["P&L %"].mean() if len(wins) else 0,2),
        "Avg Loss %":round(losses["P&L %"].mean() if len(losses) else 0,2),
        "Profit Factor":round(pf,2),"Max Drawdown %":round(dd.min(),2),
        "Total Return %":round((cash-capital)/capital*100,2),
        "CAGR %":round(cagr,2),"Sharpe Ratio":round(sr,2),
        "Net P&L":round(cash-capital,2),"Final Capital":round(cash,2),
    }}
def generate_analysis(df,ticker,asset_name,sl_type="ATR-based",sl_val=1.5,rr=2.0,capital=100000,risk_pct=2.0):
    if df is None or len(df)<50: return {}
    cur=df.iloc[-1]
    price=float(cur["Close"]); atr=float(cur["ATR"]); rsi=float(cur["RSI"])
    macd=float(cur["MACD"]); macd_s=float(cur["MACD_Signal"])
    ema20=float(cur["EMA20"]); ema50=float(cur["EMA50"]); ema200=float(cur["EMA200"])
    vwap=float(cur["VWAP"]); vol_r=float(cur["Vol_Ratio"])
    bb_u=float(cur["BB_Upper"]); bb_l=float(cur["BB_Lower"]); bb_w=float(cur["BB_Width"])
    pivot=float(cur["Pivot"]); r1=float(cur["R1"]); s1=float(cur["S1"])
    r2=float(cur["R2"]); s2=float(cur["S2"]); stoch=float(cur["Stoch_K"]); hv20=float(cur.get("HV20",18.0))
    ts=0; tn=[]
    for cond,pts,yes,no in [(price>ema20,2,"Yes Above EMA20","No Below EMA20"),
        (price>ema50,2,"Yes Above EMA50","No Below EMA50"),
        (price>ema200,3,"Yes Above EMA200 Bull","No Below EMA200 Bear"),
        (ema20>ema50,2,"Yes EMA20>EMA50 Bullish","No EMA20<EMA50 Bearish"),
        (price>vwap,1,"Yes Above VWAP","No Below VWAP")]:
        if cond: ts+=pts; tn.append(f"✅ {yes}")
        else: tn.append(f"❌ {no}")
    ms=0; mn=[]
    if 50<rsi<70: ms+=3; mn.append(f"✅ RSI {rsi:.1f} Healthy bull")
    elif rsi>=70: ms+=1; mn.append(f"⚠️ RSI {rsi:.1f} Overbought")
    elif 30<rsi<=50: mn.append(f"⚠️ RSI {rsi:.1f} Weak/bear")
    else: ms-=1; mn.append(f"❌ RSI {rsi:.1f} Oversold")
    if macd>macd_s: ms+=3; mn.append("✅ MACD bullish")
    else: mn.append("❌ MACD bearish")
    if vol_r>1.5: ms+=2; mn.append(f"✅ Vol {vol_r:.1f}x Strong")
    elif vol_r>1.0: ms+=1; mn.append(f"⚠️ Vol {vol_r:.1f}x Average")
    else: mn.append(f"❌ Vol {vol_r:.1f}x Weak")
    if 20<stoch<80: ms+=2; mn.append(f"✅ Stoch {stoch:.1f} Neutral")
    elif stoch>80: mn.append(f"⚠️ Stoch {stoch:.1f} Overbought")
    else: mn.append(f"⚠️ Stoch {stoch:.1f} Oversold")
    combined=ts+ms
    bias=next(b for thr,b in [(14,"STRONG BUY"),(10,"BUY"),(7,"WEAK BUY"),
        (4,"NEUTRAL"),(1,"WEAK SELL"),(-2,"SELL"),(-99,"STRONG SELL")] if combined>=thr)
    is_long="BUY" in bias; strength="🟢" if "BUY" in bias else ("🔴" if "SELL" in bias else "⚪")
    dir_=1 if is_long else -1
    sl_f,t1=calc_sl_tgt(price,atr,dir_,sl_type,sl_val,rr)
    _,t2=calc_sl_tgt(price,atr,dir_,sl_type,sl_val,rr*1.5)
    _,t3=calc_sl_tgt(price,atr,dir_,sl_type,sl_val,rr*2.0)
    sl_a=round(price-dir_*1.5*atr,2); sl_dist=abs(price-sl_f)
    risk_amt=capital*risk_pct/100; qty_est=int(risk_amt/sl_dist) if sl_dist>0 else 0
    iv_live=get_live_iv(ticker,price); iv_pct=round(iv_live*100,1)
    iv_rank=("✅ LOW — Buy options" if iv_pct<15 else "⚠️ MEDIUM — ATM only" if iv_pct<25 else "❌ HIGH — Avoid buying")
    ts_step=100 if price>5000 else (50 if price>1000 else 10)
    atm_k=round(price/ts_step)*ts_step
    return {"price":price,"bias":bias,"strength":strength,"trend_score":ts,"mom_score":ms,
        "combined":combined,"trend_notes":tn,"mom_notes":mn,"is_long":is_long,
        "entry":round(price,2),"sl_fixed":sl_f,"sl_atr":sl_a,"tgt1":t1,"tgt2":t2,"tgt3":t3,
        "trail_sl_start":round(price-atr if is_long else price+atr,2),
        "sl_dist":round(sl_dist,2),"qty_est":qty_est,"risk_amt":risk_amt,
        "atr":round(atr,2),"rsi":round(rsi,2),"macd":round(macd,4),
        "vol_ratio":round(vol_r,2),"hv20":round(hv20,1),
        "iv_live":iv_live,"iv_pct":iv_pct,"iv_rank_str":iv_rank,
        "bb_width":round(bb_w,4),"stoch_k":round(stoch,1),
        "atm_strike":atm_k,"itm_strike":atm_k-ts_step if is_long else atm_k+ts_step,
        "opt_rec":"CE" if is_long else "PE",
        "levels":{"EMA200":round(ema200,2),"EMA50":round(ema50,2),"EMA20":round(ema20,2),
            "VWAP":round(vwap,2),"Pivot":round(pivot,2),"R1":round(r1,2),"R2":round(r2,2),
            "S1":round(s1,2),"S2":round(s2,2),"BB Upper":round(bb_u,2),"BB Lower":round(bb_l,2)}}

def compute_hist_straddle(df,days_exp=7,rf=0.065):
    rows=[]; close_arr=df["Close"].values
    hv_arr=df["HV20"].values if "HV20" in df.columns else [18.0]*len(df)
    for i in range(len(df)):
        S=float(close_arr[i]); iv=float(hv_arr[i])/100 if float(hv_arr[i])>0 else 0.18
        T=days_exp/365; K=round(S/100)*100
        cp,_,_,_,_=bs_price(S,K,T,rf,iv,"CE"); pp,_,_,_,_=bs_price(S,K,T,rf,iv,"PE")
        rows.append({"Date":df.index[i],"Spot":round(S,2),"ATM_K":K,"IV_pct":round(iv*100,2),
            "CE_Prem":cp,"PE_Prem":pp,"Straddle":round(cp+pp,2),"Straddle_pct":round((cp+pp)/S*100,2)})
    return pd.DataFrame(rows).set_index("Date")

def straddle_reversal_stats(sdf,threshold=10.0):
    sp=sdf["Straddle"]; roc=sp.pct_change(periods=5)*100; results=[]
    for i in range(5,len(sp)-5):
        ch=float(roc.iloc[i])
        if abs(ch)>=threshold:
            future5=float(sp.iloc[i+5])-float(sp.iloc[i])
            results.append({"Date":sp.index[i],"Straddle":round(float(sp.iloc[i]),2),
                "5Bar_Chg_pct":round(ch,1),"Direction":"EXPANSION" if ch>0 else "CONTRACTION",
                "Next5_Delta":round(future5,2),
                "Reversal":"Yes" if (ch>0 and future5<0) or (ch<0 and future5>0) else "No"})
    return pd.DataFrame(results)

def zero_hero_analysis(oc_df,spot,iv_pct,days_to_exp,rf):
    if oc_df.empty: return {},[]
    atm_k=round(spot/100)*100; signals=[]
    total_ce=float(oc_df["CE_OI"].sum()) if "CE_OI" in oc_df else 0
    total_pe=float(oc_df["PE_OI"].sum()) if "PE_OI" in oc_df else 0
    pcr=round(total_pe/total_ce,2) if total_ce>0 else 1.0
    max_pain_val=atm_k; min_pain=float("inf")
    for _,row in oc_df.iterrows():
        k=float(row["Strike"])
        cp_=float(oc_df[oc_df["Strike"]>=k]["CE_OI"].sum())*max(spot-k,0)
        pp_=float(oc_df[oc_df["Strike"]<=k]["PE_OI"].sum())*max(k-spot,0)
        tot=cp_+pp_
        if tot<min_pain: min_pain=tot; max_pain_val=k
    T=max(days_to_exp,0.01)/365; iv=iv_pct/100
    cp,cd,cg,cth,cv=bs_price(spot,atm_k,T,rf,iv,"CE"); pp,pd_,pg,pth,pv=bs_price(spot,atm_k,T,rf,iv,"PE")
    straddle_prem=round(cp+pp,2); theta_daily=round(cth+pth,2)
    upper_bep=round(atm_k+straddle_prem,2); lower_bep=round(atm_k-straddle_prem,2)
    bep_pct=round(straddle_prem/spot*100,2)
    if pcr>1.3: signals.append({"Signal":"🟢 PCR Bullish","Detail":f"PCR={pcr} — PE heavy (support)","Score":2})
    elif pcr<0.7: signals.append({"Signal":"🔴 PCR Bearish","Detail":f"PCR={pcr} — CE heavy (resistance)","Score":-2})
    else: signals.append({"Signal":"⚪ PCR Neutral","Detail":f"PCR={pcr} — Balanced market","Score":0})
    if iv_pct>25: signals.append({"Signal":"🔴 High IV — Sell","Detail":f"IV={iv_pct}% — Premium overpriced, sell straddle","Score":-1})
    elif iv_pct<12: signals.append({"Signal":"🟢 Low IV — Buy","Detail":f"IV={iv_pct}% — Premium cheap, buy gamma","Score":1})
    else: signals.append({"Signal":"⚪ Medium IV","Detail":f"IV={iv_pct}% — Fair premium","Score":0})
    if abs(theta_daily)>straddle_prem*0.05:
        signals.append({"Signal":"⚠️ High Theta Burn","Detail":f"Rs{theta_daily}/day = {round(abs(theta_daily)/straddle_prem*100,1)}% of prem","Score":-1})
    if days_to_exp<=1:
        signals.append({"Signal":"⚡ Gamma Blast Zone","Detail":f"0DTE gamma={cg:.5f} — Explosive moves near ATM","Score":0})
    mp_dist=abs(spot-max_pain_val)
    if mp_dist<100: signals.append({"Signal":"📌 Near Max Pain","Detail":f"MaxPain={max_pain_val} — Expect pinning today","Score":0})
    elif spot>max_pain_val: signals.append({"Signal":"🔴 Above Max Pain","Detail":f"MaxPain={max_pain_val} — Gravity pull down","Score":-1})
    else: signals.append({"Signal":"🟢 Below Max Pain","Detail":f"MaxPain={max_pain_val} — Gravity pull up","Score":1})
    atm_row=oc_df[oc_df["Strike"]==atm_k]
    if not atm_row.empty:
        ce_chg=float(atm_row["CE_ChgOI"].iloc[0]) if "CE_ChgOI" in atm_row.columns else 0
        pe_chg=float(atm_row["PE_ChgOI"].iloc[0]) if "PE_ChgOI" in atm_row.columns else 0
        if pe_chg>ce_chg*1.5: signals.append({"Signal":"🟢 PE OI Building","Detail":"Fresh PE writing at ATM — bullish support","Score":1})
        elif ce_chg>pe_chg*1.5: signals.append({"Signal":"🔴 CE OI Building","Detail":"Fresh CE writing at ATM — resistance above","Score":-1})
    net_score=sum(s["Score"] for s in signals)
    zh_bias=("STRONG BUY 🟢" if net_score>=3 else "BUY 🟢" if net_score>=1
             else "SELL 🔴" if net_score<=-1 else "NEUTRAL ⚪")
    return {"PCR":pcr,"Max Pain":max_pain_val,"ATM Strike":atm_k,
        "Straddle Premium":straddle_prem,"CE Premium":cp,"PE Premium":pp,
        "Upper BEP":upper_bep,"Lower BEP":lower_bep,"BEP Range %":bep_pct,
        "Gamma ATM":cg,"Theta/day":theta_daily,"Vega":cv,"Delta CE":cd,"Delta PE":pd_,
        "Net Score":net_score,"Bias":zh_bias},signals

def process_live_signal(signal,price,atr,strategy,asset,sl_type,sl_val,rr,trail_pct,capital,risk_pct):
    pos=st.session_state["open_positions"]; hist=st.session_state["trade_history"]
    key=f"{asset}_{strategy}"
    if key in pos:
        p=pos[key]
        if signal==-p["direction"]:
            ppts=(price-p["entry"])*p["direction"]; pnl_pct=ppts/p["entry"]*100
            qty=(capital*risk_pct/100)/(abs(p["entry"]-p["sl"])+1e-9)
            pnl_inr=ppts*qty*p["direction"]; st.session_state["live_realized_pnl"]+=pnl_inr
            hist.append({"Date":datetime.now().strftime("%Y-%m-%d"),"Time":datetime.now().strftime("%H:%M:%S"),
                "Asset":asset,"Strategy":strategy,"Direction":"LONG" if p["direction"]==1 else "SHORT",
                "Entry":round(p["entry"],2),"Exit":round(price,2),"SL":round(p["sl"],2),"Target":round(p["target"],2),
                "P&L %":round(pnl_pct,2),"P&L Rs":round(pnl_inr,2),"Reason":"Signal Rev",
                "Status":"WIN" if pnl_inr>0 else "LOSS"})
            del pos[key]
    if key not in pos and signal!=0:
        sl,tgt=calc_sl_tgt(price,atr,signal,sl_type,sl_val,rr)
        pos[key]={"direction":signal,"entry":price,"sl":sl,"target":tgt,"asset":asset,
            "strategy":strategy,"entry_time":datetime.now().strftime("%H:%M:%S"),
            "trail_high":price,"trail_pct":trail_pct}

def check_sl_tgt_hits(cur_price,asset,capital,risk_pct):
    pos=st.session_state["open_positions"]; hist=st.session_state["trade_history"]; n=0
    for key in list(pos.keys()):
        p=pos[key]
        if p["asset"]!=asset: continue
        hit=None
        if p["direction"]==1:
            if cur_price<=p["sl"]: hit="SL Hit"
            elif cur_price>=p["target"]: hit="Target Hit"
        else:
            if cur_price>=p["sl"]: hit="SL Hit"
            elif cur_price<=p["target"]: hit="Target Hit"
        if hit:
            ex=p["sl"] if "SL" in hit else p["target"]
            ppts=(ex-p["entry"])*p["direction"]
            qty=(capital*risk_pct/100)/(abs(p["entry"]-p["sl"])+1e-9)
            pnl=ppts*qty*p["direction"]; st.session_state["live_realized_pnl"]+=pnl
            hist.append({"Date":datetime.now().strftime("%Y-%m-%d"),"Time":datetime.now().strftime("%H:%M:%S"),
                "Asset":p["asset"],"Strategy":p["strategy"],"Direction":"LONG" if p["direction"]==1 else "SHORT",
                "Entry":round(p["entry"],2),"Exit":round(ex,2),"SL":round(p["sl"],2),"Target":round(p["target"],2),
                "P&L %":round(ppts/p["entry"]*100,2),"P&L Rs":round(pnl,2),"Reason":hit,
                "Status":"WIN" if pnl>0 else "LOSS"})
            del pos[key]; n+=1
    return n

def update_unreal(cur_price,asset,capital,risk_pct):
    total=0.0
    for key,p in list(st.session_state["open_positions"].items()):
        if p["asset"]!=asset: continue
        ppts=(cur_price-p["entry"])*p["direction"]
        qty=(capital*risk_pct/100)/(abs(p["entry"]-p["sl"])+1e-9)
        total+=ppts*qty*p["direction"]
        if p["direction"]==1:
            new_tsl=cur_price*(1-p["trail_pct"]/100)
            st.session_state["open_positions"][key]["sl"]=max(p["sl"],new_tsl)
        else:
            new_tsl=cur_price*(1+p["trail_pct"]/100)
            st.session_state["open_positions"][key]["sl"]=min(p["sl"],new_tsl)
    st.session_state["live_unrealized_pnl"]=total
def price_chart(df,sig_df=None,an=None,title="",tflabel=""):
    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,row_heights=[0.55,0.15,0.15,0.15],
        vertical_spacing=0.02,subplot_titles=(f"{title} {tflabel}","Volume","RSI(14)","MACD"))
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],
        increasing_line_color=T["GREEN"],decreasing_line_color=T["RED"],name="OHLC"),row=1,col=1)
    for col,clr in [("EMA9",T["ORANGE"]),("EMA20",T["BLUE"]),("EMA50",T["ACCENT"]),("EMA200",T["PURPLE"])]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df[col],name=col,line=dict(color=clr,width=1.5),opacity=0.9),row=1,col=1)
    if "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_Upper"],line=dict(color="rgba(139,92,246,0.5)",width=1,dash="dot"),name="BB U",showlegend=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["BB_Lower"],line=dict(color="rgba(139,92,246,0.5)",width=1,dash="dot"),fill="tonexty",fillcolor="rgba(139,92,246,0.06)",name="BB",showlegend=False),row=1,col=1)
    if "VWAP" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["VWAP"],line=dict(color=T["ORANGE"],width=1.5,dash="dash"),name="VWAP"),row=1,col=1)
    if sig_df is not None and "Signal" in sig_df.columns:
        for sv,sym,clr,lbl in [(1,"triangle-up",T["GREEN"],"BUY"),(-1,"triangle-down",T["RED"],"SELL")]:
            s=sig_df[sig_df["Signal"]==sv]
            if len(s):
                yvals=s["Low"]*0.994 if sv==1 else s["High"]*1.006
                fig.add_trace(go.Scatter(x=s.index,y=yvals,mode="markers",
                    marker=dict(symbol=sym,size=11,color=clr,line=dict(color="white",width=1)),name=lbl),row=1,col=1)
    if an:
        for lbl,val,clr,dash in [("Entry",an["entry"],T["ACCENT"],"dash"),("SL",an["sl_fixed"],T["RED"],"dot"),
            ("T1",an["tgt1"],T["GREEN"],"dash"),("T2",an["tgt2"],T["GREEN"],"dashdot"),("T3",an["tgt3"],T["GREEN"],"longdash")]:
            fig.add_hline(y=val,line_color=clr,line_dash=dash,line_width=1.5,
                annotation_text=f"  {lbl}:{val}",annotation_font_color=clr,row=1,col=1)
    vc=[T["GREEN"] if c>=o else T["RED"] for c,o in zip(df["Close"],df["Open"])]
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],marker_color=vc,name="Vol",opacity=0.7),row=2,col=1)
    if "Vol_MA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["Vol_MA20"],line=dict(color=T["ACCENT"],width=1.2),name="VolMA"),row=2,col=1)
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["RSI"],line=dict(color=T["PURPLE"],width=1.5),name="RSI"),row=3,col=1)
        for lv,clr in [(70,T["RED"]),(50,T["ACCENT"]),(30,T["GREEN"])]:
            fig.add_hline(y=lv,line_color=clr,line_dash="dot",line_width=1,row=3,col=1)
    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["MACD"],line=dict(color=T["BLUE"],width=1.5),name="MACD"),row=4,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["MACD_Signal"],line=dict(color=T["RED"],width=1.2),name="Sig"),row=4,col=1)
        hc=[T["GREEN"] if v>=0 else T["RED"] for v in df["MACD_Hist"]]
        fig.add_trace(go.Bar(x=df.index,y=df["MACD_Hist"],marker_color=hc,name="Hist",opacity=0.6),row=4,col=1)
    fig.update_layout(paper_bgcolor=T["PLOT_PAPER"],plot_bgcolor=T["PLOT_BG"],font=dict(color=T["TEXT"],size=11),
        xaxis_rangeslider_visible=False,height=720,margin=dict(l=10,r=10,t=40,b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10,color=T["TEXT"])))
    for i in range(1,5):
        fig.update_xaxes(gridcolor=T["GRID"],tickfont=dict(color=T["TEXT"]),row=i,col=1)
        fig.update_yaxes(gridcolor=T["GRID"],tickfont=dict(color=T["TEXT"]),row=i,col=1)
    return fig

def equity_chart(equity,drawdown,tdf):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.65,0.35],vertical_spacing=0.04,
        subplot_titles=("Equity Curve","Drawdown %"))
    fig.add_trace(go.Scatter(x=equity.index,y=equity,fill="tozeroy",fillcolor="rgba(96,165,250,0.12)",
        line=dict(color=T["BLUE"],width=2),name="Portfolio"),row=1,col=1)
    if tdf is not None and len(tdf):
        w=tdf[tdf["P&L %"]>0]; l=tdf[tdf["P&L %"]<=0]
        if len(w): fig.add_trace(go.Scatter(x=w["Exit Date"],y=w["Capital"],mode="markers",
            marker=dict(symbol="circle",size=7,color=T["GREEN"]),name="Win"),row=1,col=1)
        if len(l): fig.add_trace(go.Scatter(x=l["Exit Date"],y=l["Capital"],mode="markers",
            marker=dict(symbol="x",size=7,color=T["RED"]),name="Loss"),row=1,col=1)
    fig.add_trace(go.Scatter(x=drawdown.index,y=drawdown,fill="tozeroy",fillcolor="rgba(239,68,68,0.12)",
        line=dict(color=T["RED"],width=1.5),name="Drawdown"),row=2,col=1)
    fig.update_layout(paper_bgcolor=T["PLOT_PAPER"],plot_bgcolor=T["PLOT_BG"],font=dict(color=T["TEXT"]),
        height=420,margin=dict(l=10,r=10,t=40,b=10),legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=T["TEXT"])))
    for i in range(1,3):
        fig.update_xaxes(gridcolor=T["GRID"],tickfont=dict(color=T["TEXT"]),row=i,col=1)
        fig.update_yaxes(gridcolor=T["GRID"],tickfont=dict(color=T["TEXT"]),row=i,col=1)
    return fig

def gauge_chart(val,max_val,title,color):
    fig=go.Figure(go.Indicator(mode="gauge+number",value=val,
        title={"text":title,"font":{"color":T["TEXT"],"size":13}},
        gauge={"axis":{"range":[0,max_val],"tickcolor":T["TEXT_MUTED"]},
            "bar":{"color":color},"bgcolor":T["CARD_BG"],"bordercolor":T["BORDER"],
            "steps":[{"range":[0,max_val*.33],"color":"rgba(220,38,38,0.18)"},
                {"range":[max_val*.33,max_val*.66],"color":"rgba(245,158,11,0.18)"},
                {"range":[max_val*.66,max_val],"color":"rgba(16,185,129,0.18)"}]},
        number={"font":{"color":color,"family":"JetBrains Mono"}}))
    fig.update_layout(paper_bgcolor=T["CARD_BG"],height=200,margin=dict(l=20,r=20,t=40,b=10),font_color=T["TEXT_MUTED"])
    return fig

def returns_chart(df,label):
    d=df.copy(); d["Dpts"]=d["Close"].diff().round(2); d["Dpct"]=(d["Close"].pct_change()*100).round(2)
    d=d.dropna().tail(252)
    cp=[T["GREEN"] if v>=0 else T["RED"] for v in d["Dpts"]]
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.5,0.5],vertical_spacing=0.04,
        subplot_titles=("Daily Return (Points)","Daily Return (%)"))
    fig.add_trace(go.Bar(x=d.index,y=d["Dpts"],marker_color=cp,name="Pts",opacity=0.85),row=1,col=1)
    fig.add_trace(go.Bar(x=d.index,y=d["Dpct"],marker_color=cp,name="Pct",opacity=0.85),row=2,col=1)
    for i in range(1,3): fig.add_hline(y=0,line_color=T["BORDER"],line_width=1,row=i,col=1)
    fig.update_layout(paper_bgcolor=T["PLOT_PAPER"],plot_bgcolor=T["PLOT_BG"],font=dict(color=T["TEXT"],size=11),
        height=440,title=dict(text=f"📈 {label} — 1Y Daily Returns",font=dict(size=14,color=T["TEXT"])),
        margin=dict(l=10,r=10,t=60,b=10),legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=T["TEXT"])))
    for i in range(1,3):
        fig.update_xaxes(gridcolor=T["GRID"],tickfont=dict(color=T["TEXT"]),row=i,col=1)
        fig.update_yaxes(gridcolor=T["GRID"],tickfont=dict(color=T["TEXT"]),row=i,col=1)
    return fig

def ohlc_table(df):
    d=df.copy().tail(252); d["Dpts"]=d["Close"].diff().round(2); d["Dpct"]=(d["Close"].pct_change()*100).round(2)
    d=d[["Open","High","Low","Close","Volume","Dpts","Dpct"]].dropna()
    d.index=d.index.strftime("%d-%b-%Y"); return d[::-1]

def straddle_chart(sdf,label):
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[0.5,0.25,0.25],vertical_spacing=0.04,
        subplot_titles=("ATM Straddle Premium","Straddle % of Spot","IV% Used"))
    ma10=sdf["Straddle"].rolling(10).mean()
    fig.add_trace(go.Scatter(x=sdf.index,y=sdf["Straddle"],fill="tozeroy",fillcolor="rgba(139,92,246,0.12)",
        line=dict(color=T["PURPLE"],width=2),name="Straddle"),row=1,col=1)
    fig.add_trace(go.Scatter(x=sdf.index,y=ma10,line=dict(color=T["ACCENT"],width=1.5,dash="dash"),name="MA10"),row=1,col=1)
    top5=sdf.nlargest(5,"Straddle"); bot5=sdf.nsmallest(5,"Straddle")
    fig.add_trace(go.Scatter(x=top5.index,y=top5["Straddle"],mode="markers",
        marker=dict(symbol="star",size=12,color=T["RED"]),name="High (Sell)"),row=1,col=1)
    fig.add_trace(go.Scatter(x=bot5.index,y=bot5["Straddle"],mode="markers",
        marker=dict(symbol="star",size=12,color=T["GREEN"]),name="Low (Buy)"),row=1,col=1)
    fig.add_trace(go.Bar(x=sdf.index,y=sdf["Straddle_pct"],marker_color=T["BLUE"],opacity=0.6,name="Straddle%"),row=2,col=1)
    fig.add_trace(go.Scatter(x=sdf.index,y=sdf["IV_pct"],line=dict(color=T["ORANGE"],width=1.5),name="IV%"),row=3,col=1)
    fig.update_layout(paper_bgcolor=T["PLOT_PAPER"],plot_bgcolor=T["PLOT_BG"],font=dict(color=T["TEXT"],size=11),
        height=620,title=dict(text=f"📊 {label} — Historical ATM Straddle Premium",font=dict(size=15,color=T["TEXT"])),
        margin=dict(l=10,r=10,t=60,b=10),legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10,color=T["TEXT"])))
    for i in range(1,4):
        fig.update_xaxes(gridcolor=T["GRID"],tickfont=dict(color=T["TEXT"]),row=i,col=1)
        fig.update_yaxes(gridcolor=T["GRID"],tickfont=dict(color=T["TEXT"]),row=i,col=1)
    return fig

def run_nifty50_scan(sl_type,sl_val,rr,capital,risk_pct):
    results=[]; prog=st.progress(0); total=len(NIFTY50_STOCKS)
    for idx,(name,tick) in enumerate(NIFTY50_STOCKS.items()):
        prog.progress((idx+1)/total,text=f"Scanning {name} ({idx+1}/{total})...")
        df=fetch_data(tick,"3mo","1d"); time.sleep(FETCH_DELAY)
        if df.empty: continue
        dfi=compute_indicators(df)
        an=generate_analysis(dfi,tick,name,sl_type,sl_val,rr,capital,risk_pct)
        if an:
            results.append({"Stock":name,"Ticker":tick,"Price":an["price"],"Signal":an["bias"],
                "Score":an["combined"],"RSI":an["rsi"],"Vol":an["vol_ratio"],
                "IV%":an["iv_pct"],"ATR":an["atr"],"Entry":an["entry"],
                "SL":an["sl_fixed"],"Target1":an["tgt1"]})
    prog.empty()
    if not results: return pd.DataFrame()
    return pd.DataFrame(results).sort_values("Score",ascending=False)
# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:14px 0 8px">
        <span style="font-size:22px;font-weight:800;
            background:linear-gradient(135deg,{T['ACCENT']},#FCD34D);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            ⚡ AlphaEdge v3
        </span><br>
        <span style="font-size:10px;color:{T['TEXT_MUTED']};letter-spacing:3px;text-transform:uppercase">
        Pro Trading Platform</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    theme_choice=st.radio("🎨 Theme",["White","Dark"],index=0 if THEME=="White" else 1,horizontal=True)
    if theme_choice!=st.session_state["theme"]:
        st.session_state["theme"]=theme_choice; st.rerun()
    st.markdown("---")
    st.markdown("**📊 Asset**")
    group_sel=st.selectbox("Group",list(ASSET_GROUPS.keys()))
    assets_in=ASSET_GROUPS[group_sel]
    asset_name_sel=st.selectbox("Asset",assets_in)
    ticker_raw=ASSET_MAP.get(asset_name_sel,"^NSEI")
    if ticker_raw=="CUSTOM": ticker=st.text_input("Custom Ticker","AAPL")
    elif ticker_raw=="NIFTY50_ALL": ticker="NIFTY50_ALL"
    else: ticker=ticker_raw
    st.markdown("---")
    st.markdown("**🧠 Strategy & Timeframe**")
    strategy_name=st.selectbox("Strategy",list(STRATEGIES.keys()),index=5)
    strategy_key=STRATEGIES[strategy_name]
    trade_type=st.selectbox("Trade Type",["Scalping (1m)","Scalping (3m)","Intraday (15m)","Swing (Daily)","Positional (Weekly)"],index=2)
    TF_MAP={"Scalping (1m)":("5d","1d","1m","1d"),"Scalping (3m)":("15d","1d","3m","2d"),
        "Intraday (15m)":("6mo","1d","15m","5d"),"Swing (Daily)":("2y","1d","1d","60d"),
        "Positional (Weekly)":("5y","1wk","1d","90d")}
    bt_period,bt_interval,live_interval,live_period=TF_MAP[trade_type]
    st.markdown(f"<small style='color:{T['TEXT_MUTED']}'>BT: <b style='color:{T['ACCENT']}'>{bt_period}</b>@<b style='color:{T['ACCENT']}'>{bt_interval}</b> | Live: <b style='color:{T['ACCENT']}'>{live_period}</b>@<b style='color:{T['ACCENT']}'>{live_interval}</b></small>",unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**⚙️ Risk**")
    capital=st.number_input("Capital (Rs)",10000,50000000,100000,step=5000)
    risk_pct=st.slider("Risk/Trade %",0.5,5.0,2.0,0.5)
    rr_ratio=st.slider("Risk:Reward",1.0,5.0,2.0,0.25)
    st.markdown("---")
    st.markdown("**🎯 SL Type**")
    sl_type=st.selectbox("Method",["ATR-based","Points-based","Percentage-based"])
    if sl_type=="ATR-based": sl_val=st.slider("ATR Multiplier",0.5,4.0,1.5,0.25); sl_label=f"{sl_val}x ATR"
    elif sl_type=="Points-based": sl_val=float(st.number_input("SL Points",5,5000,50,5)); sl_label=f"{int(sl_val)} pts"
    else: sl_val=st.number_input("SL %",0.1,20.0,1.0,0.1); sl_label=f"{sl_val}%"
    st.markdown("---")
    st.markdown("**🔄 Trailing**")
    c1_,c2_=st.columns(2)
    with c1_: trail_sl=st.checkbox("Trail SL",True); trail_tgt=st.checkbox("Trail Tgt",True)
    with c2_: tsl_pct=st.number_input("TSL%",0.3,10.0,1.5,0.25); ttgt_pct=st.number_input("TTgt%",0.3,10.0,2.0,0.25)
    st.markdown("---")
    st.markdown("**📊 Options**")
    days_exp=st.slider("Days to Expiry",0,90,7)
    risk_free=st.slider("Risk-Free Rate %",4.0,9.0,6.5,0.5)
    opt_type=st.radio("Option Type",["CE (Call)","PE (Put)"])
    st.markdown("---")
    refresh_int=st.slider("Auto-Refresh (sec)",15,120,30,5)
    st.session_state["auto_refresh_interval"]=refresh_int
    st.markdown("---")
    run_btn=st.button("🚀 Run Analysis",use_container_width=True)
    if run_btn: st.session_state["bt_run"]=True; st.cache_data.clear()

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4=st.tabs(["📊 Backtesting","⚡ Live Trading","🔭 Analyse","🎯 Straddle & Zero Hero"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — BACKTESTING
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    section("📊 Strategy Backtesting Engine")
    if ticker=="NIFTY50_ALL":
        section("All Nifty 50 Stocks — Batch Analysis")
        st.info("Scans all 50 stocks. Allow ~2 min with rate limiting (1.5s delay).")
        if st.button("▶ Run Nifty 50 Scan"):
            df_all=run_nifty50_scan(sl_type,sl_val,rr_ratio,capital,risk_pct)
            if not df_all.empty:
                st.success(f"Scanned {len(df_all)} stocks")
                buys=df_all[df_all["Signal"].str.contains("BUY")].head(10)
                sells=df_all[df_all["Signal"].str.contains("SELL")].head(5)
                cb,cs=st.columns(2)
                with cb:
                    st.markdown(f"<b style='color:{T['GREEN']}'>🟢 Top BUY Signals</b>",unsafe_allow_html=True)
                    st.dataframe(buys,use_container_width=True)
                with cs:
                    st.markdown(f"<b style='color:{T['RED']}'>🔴 Top SELL Signals</b>",unsafe_allow_html=True)
                    st.dataframe(sells,use_container_width=True)
                st.dataframe(df_all,use_container_width=True,height=400)
        st.stop()
    st.markdown(f"Asset: {badge(asset_name_sel)} Strategy: {badge(strategy_name)} Period: {badge(bt_period)} Interval: {badge(bt_interval)} SL: {badge(sl_label)} RR: {badge('1:'+str(rr_ratio))}",unsafe_allow_html=True)
    st.markdown("")
    if st.session_state.get("bt_run"):
        prog=st.progress(0,"Fetching data...")
        raw_df=fetch_data(ticker,bt_period,bt_interval)
        prog.progress(40,"Computing indicators...")
        if raw_df.empty: st.error("No data. Check ticker/internet."); prog.empty()
        else:
            df_ind=compute_indicators(raw_df)
            prog.progress(70,"Generating signals...")
            df_sig=get_strat(df_ind,strategy_key,sl_val,rr_ratio,trail_sl,tsl_pct)
            prog.progress(90,"Running backtest...")
            result=run_backtest(df_sig,capital,risk_pct,sl_val,rr_ratio,trail_sl,tsl_pct,trail_tgt,ttgt_pct,sl_type,sl_val)
            prog.progress(100); prog.empty()
            m=result.get("metrics",{})
            if not m: st.warning("No trades generated. Try wider period or different strategy.")
            else:
                dr_s=raw_df.index[0].strftime("%d %b %Y"); dr_e=raw_df.index[-1].strftime("%d %b %Y")
                st.info(f"📅 {dr_s} → {dr_e} | {len(raw_df)} candles | {bt_interval} | {sl_type} ({sl_label}) | R:R 1:{rr_ratio}")
                mc=st.columns(6)
                for col,lbl,val in zip(mc,["Trades","Win Rate","Prof.Factor","Max DD","CAGR","Sharpe"],
                    [m["Total Trades"],f"{m['Win Rate %']}%",m["Profit Factor"],
                     f"{m['Max Drawdown %']}%",f"{m['CAGR %']}%",m["Sharpe Ratio"]]):
                    with col: st.metric(lbl,val)
                mc2=st.columns(4)
                for col,lbl,val in zip(mc2,["Net P&L","Total Return","Avg Win","Avg Loss"],
                    [f"Rs{m['Net P&L']:,.0f}",f"{m['Total Return %']}%",f"{m['Avg Win %']}%",f"{m['Avg Loss %']}%"]):
                    with col: st.metric(lbl,val)
                st.plotly_chart(price_chart(df_ind.tail(250),df_sig.tail(250),None,f"{asset_name_sel} — {strategy_name}",f"[{bt_interval}|{bt_period}]"),use_container_width=True)
                st.plotly_chart(equity_chart(result["equity"],result["drawdown"],result["trades"]),use_container_width=True)
                section("📋 Trade Log")
                tdf2=result["trades"].copy()
                st.dataframe(tdf2[["Entry Date","Exit Date","Direction","Entry","Exit","SL","Target","P&L %","P&L Rs","Capital","Exit Reason"]].tail(60),use_container_width=True,height=300)
                cc1,cc2=st.columns(2)
                with cc1:
                    ec=result["trades"]["Exit Reason"].value_counts().reset_index()
                    pie=px.pie(ec,names="Exit Reason",values="count",color_discrete_sequence=[T["GREEN"],T["RED"],T["ACCENT"],T["BLUE"],T["PURPLE"]],title="Exit Reasons")
                    pie.update_layout(paper_bgcolor=T["PLOT_PAPER"],font_color=T["TEXT"],height=300)
                    st.plotly_chart(pie,use_container_width=True)
                with cc2:
                    hist_fig=px.histogram(result["trades"],x="P&L %",nbins=30,color_discrete_sequence=[T["BLUE"]],title="P&L Distribution")
                    hist_fig.update_layout(paper_bgcolor=T["PLOT_PAPER"],plot_bgcolor=T["PLOT_BG"],font_color=T["TEXT"],height=300)
                    st.plotly_chart(hist_fig,use_container_width=True)
                section("📈 OHLC Daily Returns — 1Y")
                ohlc1y=fetch_data(ticker,"1y","1d")
                if not ohlc1y.empty:
                    st.plotly_chart(returns_chart(ohlc1y,asset_name_sel),use_container_width=True)
                    tbl=ohlc_table(ohlc1y)
                    st.dataframe(tbl,use_container_width=True,height=320)
    else:
        st.info("Configure in sidebar and click **🚀 Run Analysis**")
# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE TRADING (auto-refresh)
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    section("⚡ Live Trading Dashboard")
    ctrl=st.columns([1,1,1,4])
    with ctrl[0]:
        if st.button("▶ Start",disabled=st.session_state["trading_active"],use_container_width=True):
            st.session_state["trading_active"]=True
            st.session_state["paper_capital"]=float(capital)
            st.session_state["live_realized_pnl"]=0.0
            st.session_state["live_unrealized_pnl"]=0.0
            st.session_state["last_auto_refresh"]=time.time()
            st.rerun()
    with ctrl[1]:
        if st.button("⏹ Stop",disabled=not st.session_state["trading_active"],use_container_width=True):
            st.session_state["trading_active"]=False; st.rerun()
    with ctrl[2]:
        if st.button("🔄 Refresh",use_container_width=True):
            st.cache_data.clear(); st.rerun()
    with ctrl[3]:
        tc=T["GREEN"] if st.session_state["trading_active"] else T["RED"]
        tt="🟢 LIVE — Auto-refreshing" if st.session_state["trading_active"] else "🔴 Stopped"
        st.markdown(f"""<div style="background:{T['CARD_BG']};border:1px solid {tc}40;border-radius:8px;padding:8px 14px">
            <b style="color:{tc}">{tt}</b> | <span style="color:{T['TEXT_MUTED']};font-size:12px">
            Capital: <b>Rs{st.session_state['paper_capital']:,.0f}</b> | {live_interval} | {live_period} | Refresh: {refresh_int}s</span></div>""",unsafe_allow_html=True)
    if ticker=="NIFTY50_ALL": st.warning("Select a specific ticker for Live Trading."); st.stop()
    st.info(f"Yahoo Finance (15-20 min delay India) | Rate limit: {FETCH_DELAY}s | Interval: {live_interval} | Period: {live_period}")
    with st.spinner("Fetching live data..."):
        live_raw=fetch_live(ticker,live_interval,live_period)
    if live_raw.empty: st.error("No live data."); st.stop()
    live_df=compute_indicators(live_raw)
    live_sig=get_strat(live_df,strategy_key,sl_val,rr_ratio,trail_sl,tsl_pct)
    an=generate_analysis(live_df,ticker,asset_name_sel,sl_type,sl_val,rr_ratio,capital,risk_pct)
    if not an: st.warning("Insufficient data."); st.stop()
    cur_price=an["price"]
    if st.session_state["trading_active"]:
        latest_sig=int(live_sig["Signal"].iloc[-1])
        latest_atr=float(live_sig["ATR"].iloc[-1])
        if latest_sig!=0:
            process_live_signal(latest_sig,cur_price,latest_atr,strategy_key,asset_name_sel,
                sl_type,sl_val,rr_ratio,tsl_pct,capital,risk_pct)
        hits=check_sl_tgt_hits(cur_price,asset_name_sel,capital,risk_pct)
        if hits>0: st.success(f"🎯 {hits} position(s) auto-closed (SL/Target hit)")
    update_unreal(cur_price,asset_name_sel,capital,risk_pct)
    realized=st.session_state["live_realized_pnl"]
    unrealized=st.session_state["live_unrealized_pnl"]
    total_pnl=realized+unrealized
    prev_px=float(live_df["Close"].iloc[-2]) if len(live_df)>1 else cur_price
    price_chg=cur_price-prev_px; price_pct=price_chg/prev_px*100 if prev_px else 0
    pm=st.columns(6)
    with pm[0]: st.metric("📍 LTP",f"Rs{cur_price:,.2f}",f"{price_chg:+.2f} ({price_pct:+.2f}%)")
    with pm[1]: st.metric("💰 Realized",f"Rs{realized:+,.0f}")
    with pm[2]: st.metric("📊 Unrealized",f"Rs{unrealized:+,.0f}")
    with pm[3]: st.metric("🏦 Total P&L",f"Rs{total_pnl:+,.0f}")
    with pm[4]: st.metric("📂 Open Pos",len(st.session_state["open_positions"]))
    with pm[5]: st.metric("RSI / ATR",f"{an['rsi']:.1f} / {an['atr']:.1f}")
    st.markdown("---")
    sig_c=T["GREEN"] if "BUY" in an["bias"] else (T["RED"] if "SELL" in an["bias"] else T["ACCENT"])
    st.markdown(f"""<div style="background:{sig_c}10;border:1px solid {sig_c}40;border-left:5px solid {sig_c};
        border-radius:10px;padding:14px 20px;margin:8px 0">
        <span style="font-size:22px;font-weight:800;color:{sig_c}">{an['strength']} {an['bias']}</span>
        <span style="color:{T['TEXT_MUTED']};font-size:12px"> | {strategy_name} | Trend:{an['trend_score']}/10 | Mom:{an['mom_score']}/10 | Score:{an['combined']}/20 | HV:{an['hv20']}% | IV:{an['iv_pct']}% | {datetime.now().strftime('%H:%M:%S')}</span>
    </div>""",unsafe_allow_html=True)
    section("🎯 Entry | SL | Targets")
    em=st.columns(7)
    for col,(lbl,val,dlt) in zip(em,[("Entry",f"Rs{an['entry']:,.2f}",""),
        ("Fixed SL",f"Rs{an['sl_fixed']:,.2f}",f"-{an['sl_dist']:.1f}pts"),
        ("ATR SL",f"Rs{an['sl_atr']:,.2f}",f"ATR={an['atr']:.1f}"),
        ("Target 1",f"Rs{an['tgt1']:,.2f}",f"+{abs(an['tgt1']-an['entry']):.1f}pts"),
        ("Target 2",f"Rs{an['tgt2']:,.2f}",f"+{abs(an['tgt2']-an['entry']):.1f}pts"),
        ("Target 3",f"Rs{an['tgt3']:,.2f}",f"+{abs(an['tgt3']-an['entry']):.1f}pts"),
        ("R:R",f"1:{rr_ratio}","")]):
        with col: st.metric(lbl,val,dlt if dlt else None)
    if st.session_state["open_positions"]:
        section("📂 Open Positions (Paper)")
        rows=[]
        for key,pos in st.session_state["open_positions"].items():
            cp=cur_price if pos["asset"]==asset_name_sel else pos["entry"]
            ppts=(cp-pos["entry"])*pos["direction"]
            qty=(capital*risk_pct/100)/(abs(pos["entry"]-pos["sl"])+1e-9)
            upnl=ppts*qty*pos["direction"]
            rows.append({"Asset":pos["asset"],"Dir":"LONG" if pos["direction"]==1 else "SHORT",
                "Entry":pos["entry"],"Current":round(cp,2),"Trail SL":round(pos["sl"],2),
                "Target":round(pos["target"],2),"Time":pos["entry_time"],
                "Unreal P&L":f"Rs{upnl:+,.0f}","P&L%":f"{ppts/pos['entry']*100*pos['direction']:+.2f}%"})
        st.dataframe(pd.DataFrame(rows),use_container_width=True)
    section(f"📈 Live Chart [{live_interval} | {live_period}]")
    st.plotly_chart(price_chart(live_df.tail(120),live_sig.tail(120),an,f"Live {asset_name_sel}",f"[{live_interval}|{live_period}]"),use_container_width=True)
    recent=live_sig[live_sig["Signal"]!=0].tail(10)
    if not recent.empty:
        section("📡 Recent Signals")
        showcols=[c for c in ["Close","Signal","RSI","MACD","ATR","Vol_Ratio","EMA20","VWAP"] if c in recent.columns]
        st.dataframe(recent[showcols].round(2),use_container_width=True)
    section("📜 Trade History (Paper Trades)")
    hist_list=st.session_state["trade_history"]
    if hist_list:
        hdf=pd.DataFrame(hist_list)
        st.dataframe(hdf[["Date","Time","Asset","Strategy","Direction","Entry","Exit","SL","Target","P&L %","P&L Rs","Reason","Status"]],use_container_width=True,height=260)
        wins_h=hdf[hdf["P&L Rs"]>0]; total_r=hdf["P&L Rs"].sum()
        sm=st.columns(4)
        with sm[0]: st.metric("Trades",len(hdf))
        with sm[1]: st.metric("Win Rate",f"{len(wins_h)/len(hdf)*100:.1f}%")
        with sm[2]: st.metric("Total P&L",f"Rs{total_r:+,.0f}")
        with sm[3]: st.metric("Avg/Trade",f"Rs{total_r/len(hdf):+,.0f}")
        if st.button("🗑️ Clear History"):
            st.session_state["trade_history"]=[]; st.session_state["live_realized_pnl"]=0.0; st.rerun()
    else:
        st.info("No paper trades yet. Click Start to begin.")
    section("📊 Option Pricing (Live)")
    nse_sym=NSE_OC_MAP.get(ticker)
    opt_t="CE" if "CE" in opt_type else "PE"
    T_exp=max(days_exp,1)/365; sigma=an["iv_live"]; atm_k_live=an["atm_strike"]
    with st.spinner("Loading option data..."):
        if nse_sym:
            nse_data=fetch_nse_oc(nse_sym); oc_df_live=parse_nse_oc(nse_data,cur_price)
        else: nse_data={}; oc_df_live=pd.DataFrame()
        time.sleep(FETCH_DELAY)
    if not oc_df_live.empty:
        st.success(f"NSE Live Chain — {nse_sym} | Spot: Rs{cur_price:,.2f} | IV: {an['iv_pct']}%")
        st.dataframe(oc_df_live,use_container_width=True,height=280)
        total_ce=oc_df_live["CE_OI"].sum(); total_pe=oc_df_live["PE_OI"].sum()
        pcr_live=round(total_pe/total_ce,2) if total_ce>0 else 0
        pc1,pc2,pc3=st.columns(3)
        with pc1: st.metric("PCR",pcr_live,"Bullish" if pcr_live>1.2 else "Bearish" if pcr_live<0.8 else "Neutral")
        with pc2: st.metric("CE OI",f"{total_ce:,.0f}")
        with pc3: st.metric("PE OI",f"{total_pe:,.0f}")
    else:
        st.info(f"BS pricing | IV: {an['iv_pct']}% | Exp: {days_exp}d")
        bs_rows=[]
        for k in [atm_k_live+i*50 for i in range(-5,6)]:
            p,d,g,th,v=bs_price(cur_price,k,T_exp,risk_free/100,sigma,opt_t)
            mon="ATM" if k==atm_k_live else ("ITM" if ((opt_t=="CE" and cur_price>k) or (opt_t=="PE" and cur_price<k)) else "OTM")
            bs_rows.append({"Strike":k,"Type":opt_t,"Moneyness":mon,"Premium":p,"Delta":d,"Gamma":g,"Theta":th,"Vega":v,"BEP":round(k+p if opt_t=="CE" else k-p,2)})
        st.dataframe(pd.DataFrame(bs_rows),use_container_width=True)
    section("📐 Live Greeks (ATM)")
    cp_,cd_,cg_,cth_,cv_=bs_price(cur_price,atm_k_live,T_exp,risk_free/100,sigma,"CE")
    pp_,pd__,pg_,pth_,pv_=bs_price(cur_price,atm_k_live,T_exp,risk_free/100,sigma,"PE")
    gk=st.columns(6)
    for col,lbl,val in zip(gk,["IV Live","CE Prem","PE Prem","Straddle","Theta/day","Vega"],
        [f"{an['iv_pct']}%",f"Rs{cp_}",f"Rs{pp_}",f"Rs{cp_+pp_}",f"Rs{cth_+pth_}",cv_]):
        with col: st.metric(lbl,val)
    section(f"📊 {asset_name_sel} OHLC Returns — 1Y Daily")
    ohlc_1y=fetch_data(ticker,"1y","1d")
    if not ohlc_1y.empty:
        st.plotly_chart(returns_chart(ohlc_1y,asset_name_sel),use_container_width=True)
        tbl2=ohlc_table(ohlc_1y)
        st.dataframe(tbl2,use_container_width=True,height=300)
    # AUTO-REFRESH
    if st.session_state["trading_active"]:
        now=time.time(); elapsed=now-st.session_state.get("last_auto_refresh",0)
        remaining=int(st.session_state["auto_refresh_interval"]-elapsed)
        if remaining>0:
            st.markdown(f"""<div style="text-align:center;padding:10px;color:{T['TEXT_MUTED']};font-size:12px">
            Auto-refresh in <b style="color:{T['ACCENT']}">{remaining}s</b> (every {refresh_int}s) | 1.5s rate limit between requests</div>""",unsafe_allow_html=True)
            time.sleep(max(remaining,1))
        st.session_state["last_auto_refresh"]=time.time()
        st.cache_data.clear(); st.rerun()
# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYSE
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    section("🔭 Deep Analysis & Recommendations")
    if ticker=="NIFTY50_ALL": st.warning("Select a specific ticker."); st.stop()
    with st.spinner("Fetching 1Y daily data..."):
        an_raw=fetch_data(ticker,"1y","1d")
    if an_raw.empty: st.error("No data."); st.stop()
    an_df=compute_indicators(an_raw)
    an=generate_analysis(an_df,ticker,asset_name_sel,sl_type,sl_val,rr_ratio,capital,risk_pct)
    if not an: st.warning("Insufficient data."); st.stop()
    dr_s=an_raw.index[0].strftime("%d %b %Y"); dr_e=an_raw.index[-1].strftime("%d %b %Y")
    st.markdown(f"Data: {badge('1Y Daily')} Range: {badge(dr_s+' to '+dr_e)} SL: {badge(sl_type+' '+sl_label)} RR: {badge('1:'+str(rr_ratio))}",unsafe_allow_html=True)
    dir_c=T["GREEN"] if "BUY" in an["bias"] else T["RED"]
    st.markdown(f"""<div style="background:{dir_c}10;border:1px solid {dir_c}40;border-left:6px solid {dir_c};
        border-radius:14px;padding:22px 26px;margin-bottom:18px">
        <div style="font-size:28px;font-weight:800;color:{dir_c}">{an['strength']} {an['bias']}</div>
        <div style="color:{T['TEXT_MUTED']};font-size:13px;margin-top:6px">
        {asset_name_sel} ({ticker}) · LTP: <b style="color:{T['TEXT']}">Rs{an['price']:,.2f}</b>
        · ATR: <b>{an['atr']}</b> · RSI: <b>{an['rsi']}</b> · HV20: <b>{an['hv20']}%</b> · IV: <b>{an['iv_pct']}%</b>
        </div></div>""",unsafe_allow_html=True)
    section("📋 Complete Trade Plan")
    tp1,tp2=st.columns(2)
    with tp1:
        dc2=T["GREEN"] if an["is_long"] else T["RED"]
        plan_rows=[("Direction",f'<b style="color:{dc2}">{"LONG / BUY CE" if an["is_long"] else "SHORT / BUY PE"}</b>'),
            ("Entry",f'<b style="color:{T["ACCENT"]}">Rs{an["entry"]:,.2f}</b>'),
            ("SL Method",sl_type+" — "+sl_label),
            ("Stop Loss",f'<b style="color:{T["RED"]}">Rs{an["sl_fixed"]:,.2f} ({an["sl_dist"]:.1f}pts)</b>'),
            ("ATR SL (1.5x)",f'<span style="color:{T["RED"]}">Rs{an["sl_atr"]:,.2f}</span>'),
            ("Target 1 (50%)",f'<b style="color:{T["GREEN"]}">Rs{an["tgt1"]:,.2f} (+{abs(an["tgt1"]-an["entry"]):.1f}pts)</b>'),
            ("Target 2 (30%)",f'<span style="color:{T["GREEN"]}">Rs{an["tgt2"]:,.2f} (+{abs(an["tgt2"]-an["entry"]):.1f}pts)</span>'),
            ("Target 3 (20%)",f'<span style="color:{T["GREEN"]}">Rs{an["tgt3"]:,.2f} (+{abs(an["tgt3"]-an["entry"]):.1f}pts)</span>'),
            ("Trail SL",f"Trail {tsl_pct}% below peak"),
            ("Trail Target",f"Trail {ttgt_pct}% above price"),
            ("Risk Amount",f"Rs{an['risk_amt']:,.0f} ({risk_pct}% of Rs{capital:,})"),
            ("Est. Qty",f"{an['qty_est']} units"),
            ("R:R",f'<b>1:{abs(an["tgt1"]-an["entry"])/(an["sl_dist"]+1e-9):.1f}</b>')]
        hr="".join([f'<tr><td style="color:{T["TEXT_MUTED"]};padding:5px 8px 5px 0;font-size:13px;border-bottom:1px solid {T["BORDER"]}20">{r}</td>'
            f'<td style="font-size:13px;padding:5px 0;border-bottom:1px solid {T["BORDER"]}20">{v}</td></tr>' for r,v in plan_rows])
        st.markdown(f'<div class="alpha-card"><b style="color:{T["ACCENT"]};font-size:15px">🎯 Trade Setup</b><br><br><table style="width:100%;border-collapse:collapse">{hr}</table></div>',unsafe_allow_html=True)
    with tp2:
        lv_html="".join([f'<tr><td style="color:{T["TEXT_MUTED"]};font-size:13px;padding:5px 8px 5px 0;border-bottom:1px solid {T["BORDER"]}20">{n}</td>'
            f'<td style="color:{c};font-weight:600;font-size:13px;font-family:JetBrains Mono;padding:5px 0;border-bottom:1px solid {T["BORDER"]}20">Rs{v:,.2f}'
            f'{"  Near" if abs(v-an["price"])<=an["atr"] else ""}</td></tr>'
            for n,v,c in [("EMA200",an["levels"]["EMA200"],T["PURPLE"]),("EMA50",an["levels"]["EMA50"],T["ACCENT"]),
                ("EMA20",an["levels"]["EMA20"],T["BLUE"]),("VWAP",an["levels"]["VWAP"],T["ORANGE"]),
                ("Pivot",an["levels"]["Pivot"],T["TEXT"]),("R1",an["levels"]["R1"],T["GREEN"]),
                ("R2",an["levels"]["R2"],T["GREEN"]),("S1",an["levels"]["S1"],T["RED"]),
                ("S2",an["levels"]["S2"],T["RED"]),("BB Upper",an["levels"]["BB Upper"],T["PURPLE"]),
                ("BB Lower",an["levels"]["BB Lower"],T["PURPLE"])]])
        st.markdown(f'<div class="alpha-card"><b style="color:{T["ACCENT"]};font-size:15px">📐 Key Levels</b><br><br><table style="width:100%;border-collapse:collapse">{lv_html}</table></div>',unsafe_allow_html=True)
    st.markdown("---")
    ta1,ta2=st.columns(2)
    with ta1:
        section("📈 Trend Analysis")
        for note in an["trend_notes"]:
            bg=f"{T['GREEN']}10" if "✅" in note else f"{T['ACCENT']}10" if "⚠️" in note else f"{T['RED']}10"
            st.markdown(f'<div style="background:{bg};border-radius:6px;padding:8px 12px;margin:4px 0;font-size:13px;color:{T["TEXT"]}">{note}</div>',unsafe_allow_html=True)
    with ta2:
        section("⚡ Momentum Analysis")
        for note in an["mom_notes"]:
            bg=f"{T['GREEN']}10" if "✅" in note else f"{T['ACCENT']}10" if "⚠️" in note else f"{T['RED']}10"
            st.markdown(f'<div style="background:{bg};border-radius:6px;padding:8px 12px;margin:4px 0;font-size:13px;color:{T["TEXT"]}">{note}</div>',unsafe_allow_html=True)
    st.markdown("---")
    an_sig=get_strat(an_df,strategy_key,sl_val,rr_ratio,trail_sl,tsl_pct)
    st.plotly_chart(price_chart(an_df.tail(250),an_sig.tail(250),an,f"🔭 {asset_name_sel}","[1D|1Y]"),use_container_width=True)
    section("📡 Signal Gauges")
    gc=st.columns(4)
    for col,v,mx,ttl,clr in zip(gc,
        [an["trend_score"],an["mom_score"],an["rsi"],min(an["vol_ratio"]*4,20)],
        [10,10,100,20],["Trend","Momentum","RSI","Volume"],
        [T["BLUE"],T["PURPLE"],T["ACCENT"],T["GREEN"]]):
        with col: st.plotly_chart(gauge_chart(v,mx,ttl,clr),use_container_width=True)
    section("📈 OHLC Daily Returns (1Y | 1D)")
    ohlc_an=fetch_data(ticker,"1y","1d")
    if not ohlc_an.empty:
        st.plotly_chart(returns_chart(ohlc_an,asset_name_sel),use_container_width=True)
        tbl_an=ohlc_table(ohlc_an)
        st.dataframe(tbl_an,use_container_width=True,height=320)
    st.markdown("---")
    section("🔍 Multi-Asset Quick Scan")
    SCAN_ASSETS={"Nifty":"^NSEI","BankNifty":"^NSEBANK","Reliance":"RELIANCE.NS",
        "HDFC Bank":"HDFCBANK.NS","BTC":"BTC-USD","ETH":"ETH-USD",
        "Gold":"GC=F","Silver":"SI=F","USD/INR":"USDINR=X"}
    if st.button("🔍 Run Scan"):
        scan_res=[]; prog_s=st.progress(0)
        for i,(sn,st_) in enumerate(SCAN_ASSETS.items()):
            prog_s.progress((i+1)/len(SCAN_ASSETS),text=f"Scanning {sn}...")
            sdf=fetch_data(st_,"3mo","1d"); time.sleep(FETCH_DELAY)
            if sdf.empty: continue
            sdfi=compute_indicators(sdf)
            san=generate_analysis(sdfi,st_,sn,sl_type,sl_val,rr_ratio,capital,risk_pct)
            if san:
                scan_res.append({"Asset":sn,"Price":san["price"],"Signal":san["bias"],
                    "Score":san["combined"],"RSI":san["rsi"],"IV%":san["iv_pct"],
                    "Entry":san["entry"],"SL":san["sl_fixed"],"Target":san["tgt1"]})
        prog_s.empty()
        if scan_res:
            sdf_out=pd.DataFrame(scan_res).sort_values("Score",ascending=False)
            st.dataframe(sdf_out,use_container_width=True)
    st.markdown(f"""<div style="background:{T['CARD_BG']};border:1px solid {T['ACCENT']}30;border-radius:10px;padding:12px 16px;font-size:12px;color:{T['TEXT_MUTED']}">
    ⚠️ <b style="color:{T['ACCENT']}">Disclaimer:</b> Educational use only. Not financial advice. Options trading involves substantial risk.
    </div>""",unsafe_allow_html=True)
# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — STRADDLE & ZERO HERO
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    section("🎯 Straddle Premium Chart & Zero Hero Analysis")
    if ticker=="NIFTY50_ALL": st.warning("Select a specific ticker."); st.stop()
    st.markdown(f"Asset: {badge(asset_name_sel)} Data: {badge('1Y Daily')} Exp: {badge(str(days_exp)+'d')} IV: {badge('NSE/HV Proxy')}",unsafe_allow_html=True)
    sub1,sub2=st.tabs(["📊 Straddle Premium History","⚡ Zero Hero (0DTE)"])

    with sub1:
        section("📊 Historical ATM Straddle Premium")
        st.markdown(f"""<div class="alpha-card">
        <b style="color:{T['ACCENT']}">What is Straddle Premium?</b><br>
        <span style="font-size:13px;color:{T['TEXT_MUTED']}">
        ATM Straddle = Buy ATM CE + ATM PE. Combined premium = expected move in points.<br>
        <b>HIGH premium</b> = market expects big move → consider SELLING.<br>
        <b>LOW premium</b> = cheap optionality → consider BUYING gamma.<br><br>
        Red stars = historically high (sell zone) | Green stars = historically low (buy zone)
        </span></div>""",unsafe_allow_html=True)
        sd1,sd2=st.columns(2)
        with sd1: threshold_pct=st.slider("Reversal Threshold %",5.0,30.0,10.0,1.0)
        with sd2: straddle_days=st.slider("Days to Expiry (Straddle)",1,30,7)
        with st.spinner("Computing historical straddle..."):
            hist_df=fetch_data(ticker,"1y","1d"); time.sleep(FETCH_DELAY)
        if not hist_df.empty:
            hist_dfi=compute_indicators(hist_df)
            sdf_hist=compute_hist_straddle(hist_dfi,straddle_days,risk_free/100)
            sp_mean=sdf_hist["Straddle"].mean(); sp_std=sdf_hist["Straddle"].std()
            sp_curr=sdf_hist["Straddle"].iloc[-1]; sp_pct=sdf_hist["Straddle_pct"].iloc[-1]
            sp_z=(sp_curr-sp_mean)/sp_std if sp_std>0 else 0
            sm1,sm2,sm3,sm4,sm5=st.columns(5)
            with sm1: st.metric("Current Straddle",f"Rs{sp_curr:.2f}")
            with sm2: st.metric("Avg (1Y)",f"Rs{sp_mean:.2f}")
            with sm3: st.metric("Straddle % Spot",f"{sp_pct:.2f}%")
            with sm4: st.metric("Z-Score",f"{sp_z:.2f}","High" if sp_z>1.5 else "Low" if sp_z<-1.5 else "Normal")
            with sm5: st.metric("Expected Move",f"+-Rs{sp_curr:.0f}")
            if sp_z>1.5:
                st.markdown(f'<div style="background:{T["RED"]}10;border-left:4px solid {T["RED"]};border-radius:8px;padding:12px 16px;font-size:13px">'
                    f'🔴 <b>Straddle Overpriced (Z={sp_z:.2f})</b> — {((sp_curr-sp_mean)/sp_mean*100):.1f}% above avg. '
                    f'Consider SELLING straddle/iron condor. Avoid buying options.</div>',unsafe_allow_html=True)
            elif sp_z<-1.5:
                st.markdown(f'<div style="background:{T["GREEN"]}10;border-left:4px solid {T["GREEN"]};border-radius:8px;padding:12px 16px;font-size:13px">'
                    f'🟢 <b>Straddle Cheap (Z={sp_z:.2f})</b> — {((sp_mean-sp_curr)/sp_mean*100):.1f}% below avg. '
                    f'Consider BUYING straddle/strangle. Volatility expansion expected.</div>',unsafe_allow_html=True)
            else: st.info(f"Straddle at normal level (Z={sp_z:.2f}). No strong edge for sellers or buyers.")
            st.plotly_chart(straddle_chart(sdf_hist,asset_name_sel),use_container_width=True)
            section(f"🔄 Reversal Analysis — After ±{threshold_pct:.0f}% move")
            rev_df=straddle_reversal_stats(sdf_hist,threshold_pct)
            if not rev_df.empty:
                tot_rev=len(rev_df); yes_rev=len(rev_df[rev_df["Reversal"]=="Yes"])
                rev_wr=round(yes_rev/tot_rev*100,1) if tot_rev>0 else 0
                rc1,rc2,rc3=st.columns(3)
                with rc1: st.metric("Instances",tot_rev)
                with rc2: st.metric("Reversals",yes_rev)
                with rc3: st.metric("Reversal Rate",f"{rev_wr}%","High Edge" if rev_wr>65 else "Moderate" if rev_wr>50 else "Low")
                st.dataframe(rev_df,use_container_width=True,height=230)
                if rev_wr>60:
                    st.markdown(f"""<div class="alpha-card">
                    <b style="color:{T['ACCENT']}">📐 Straddle Reversal Strategy (Edge Found)</b><br><br>
                    After straddle moves >{threshold_pct:.0f}%, reversal happens {rev_wr}% of time in 5 bars.<br><br>
                    <b style="color:{T['RED']}">After EXPANSION (+{threshold_pct:.0f}%):</b> Sell straddle | SL: 20% above entry | Target: 30% below<br>
                    <b style="color:{T['BLUE']}">After CONTRACTION (-{threshold_pct:.0f}%):</b> Buy straddle | SL: 25% below | Target: ATR expansion<br><br>
                    <span style="color:{T['TEXT_MUTED']};font-size:12px">Confluence: PCR + RSI divergence + Volume spike. Best on Budget/RBI/Expiry days.</span>
                    </div>""",unsafe_allow_html=True)
            dist_fig=px.histogram(sdf_hist,x="Straddle",nbins=40,title="Straddle Premium Distribution (1Y)",color_discrete_sequence=[T["PURPLE"]])
            dist_fig.add_vline(x=sp_curr,line_color=T["ACCENT"],line_dash="dash",annotation_text=f"Current:{sp_curr:.0f}")
            dist_fig.add_vline(x=sp_mean,line_color=T["BLUE"],line_dash="dot",annotation_text=f"Mean:{sp_mean:.0f}")
            dist_fig.update_layout(paper_bgcolor=T["PLOT_PAPER"],plot_bgcolor=T["PLOT_BG"],font_color=T["TEXT"],height=300)
            st.plotly_chart(dist_fig,use_container_width=True)

    with sub2:
        section("⚡ Zero Hero — 0DTE Options Analysis")
        st.markdown(f"""<div class="alpha-card">
        <b style="color:{T['ACCENT']}">Zero Hero Strategy (0DTE = Same-Day Expiry)</b><br>
        <span style="font-size:13px;color:{T['TEXT_MUTED']}">
        0DTE options have maximum Theta decay and explosive Gamma near ATM.<br>
        Analyze OI, PCR, IV, Max Pain, Gamma, Theta for high-probability setups.<br>
        <b>Best for:</b> Thursday (Nifty) | Wednesday (BankNifty) | Entry window: 9:20-9:45 AM
        </span></div>""",unsafe_allow_html=True)
        with st.spinner("Fetching live data for Zero Hero..."):
            zh_nse_sym=NSE_OC_MAP.get(ticker)
            if zh_nse_sym:
                zh_oc_data=fetch_nse_oc(zh_nse_sym)
            else: zh_oc_data={}
            time.sleep(FETCH_DELAY)
            zh_df=fetch_data(ticker,"3mo","1d"); time.sleep(FETCH_DELAY)
        if not zh_df.empty:
            zh_dfi=compute_indicators(zh_df)
            zh_an=generate_analysis(zh_dfi,ticker,asset_name_sel)
        else: zh_an={}
        zh_spot=zh_an.get("price",20000); zh_iv=zh_an.get("iv_pct",15.0)
        zh_oc_df=parse_nse_oc(zh_oc_data,zh_spot) if zh_oc_data else pd.DataFrame()
        zh_days=min(days_exp,1) if days_exp==0 else days_exp
        zh_summary,zh_signals=zero_hero_analysis(zh_oc_df,zh_spot,zh_iv,zh_days,risk_free/100)
        if zh_summary:
            zm=st.columns(6)
            for col,lbl,val in zip(zm,["PCR","Max Pain","Straddle","Upper BEP","Lower BEP","BEP%"],
                [zh_summary["PCR"],f"Rs{zh_summary['Max Pain']:,.0f}",f"Rs{zh_summary['Straddle Premium']}",
                 f"Rs{zh_summary['Upper BEP']:,.0f}",f"Rs{zh_summary['Lower BEP']:,.0f}",f"+-{zh_summary['BEP Range %']}%"]):
                with col: st.metric(lbl,val)
            zg=st.columns(5)
            for col,lbl,val in zip(zg,["Gamma","Theta/day","Vega","Delta CE","Delta PE"],
                [zh_summary["Gamma ATM"],f"Rs{zh_summary['Theta/day']}",zh_summary["Vega"],zh_summary["Delta CE"],zh_summary["Delta PE"]]):
                with col: st.metric(lbl,val)
            bias_c=T["GREEN"] if "BUY" in zh_summary["Bias"] else (T["RED"] if "SELL" in zh_summary["Bias"] else T["ACCENT"])
            st.markdown(f"""<div style="background:{bias_c}10;border-left:5px solid {bias_c};border-radius:10px;padding:14px 20px;margin:12px 0">
            <span style="font-size:20px;font-weight:800;color:{bias_c}">Zero Hero Bias: {zh_summary['Bias']}</span>
            <span style="color:{T['TEXT_MUTED']};font-size:12px"> | Score:{zh_summary['Net Score']} | Spot:Rs{zh_spot:,.2f} | ATM:Rs{zh_summary['ATM Strike']} | IV:{zh_iv}%</span>
            </div>""",unsafe_allow_html=True)
        section("📡 Zero Hero Signals")
        if zh_signals:
            for sig in zh_signals:
                sc=T["GREEN"] if sig["Score"]>0 else T["RED"] if sig["Score"]<0 else T["ACCENT"]
                st.markdown(f'<div style="background:{sc}10;border-left:4px solid {sc};border-radius:8px;padding:10px 14px;margin:4px 0;font-size:13px">'
                    f'<b style="color:{sc}">{sig["Signal"]}</b> — <span style="color:{T["TEXT_MUTED"]}">{sig["Detail"]}</span></div>',unsafe_allow_html=True)
        else: st.info("Load NSE option chain (Nifty/BankNifty) for Zero Hero signals.")
        section("📐 Gamma & Theta Curves (ATM Options)")
        spot_zh=zh_spot; iv_zh=zh_iv/100
        strikes_zh=[round(spot_zh/50)*50+i*50 for i in range(-10,11)]
        curves=[]
        for k in strikes_zh:
            _,_,g,th,_=bs_price(spot_zh,k,max(zh_days,0.01)/365,risk_free/100,iv_zh,"CE")
            curves.append({"Strike":k,"Gamma":g,"Theta":abs(th)})
        curves_df=pd.DataFrame(curves)
        fig_gth=make_subplots(rows=1,cols=2,subplot_titles=("Gamma by Strike","Theta Decay by Strike"))
        gcolors=[T["GREEN"] if abs(k-spot_zh)<100 else T["BLUE"] for k in curves_df["Strike"]]
        fig_gth.add_trace(go.Bar(x=curves_df["Strike"],y=curves_df["Gamma"],marker_color=gcolors,name="Gamma"),row=1,col=1)
        fig_gth.add_trace(go.Bar(x=curves_df["Strike"],y=curves_df["Theta"],
            marker_color=[T["RED"] if abs(k-spot_zh)<100 else T["ORANGE"] for k in curves_df["Strike"]],name="Theta"),row=1,col=2)
        fig_gth.add_vline(x=spot_zh,line_color=T["ACCENT"],line_dash="dash",annotation_text="Spot",row=1,col=1)
        fig_gth.add_vline(x=spot_zh,line_color=T["ACCENT"],line_dash="dash",row=1,col=2)
        fig_gth.update_layout(paper_bgcolor=T["PLOT_PAPER"],plot_bgcolor=T["PLOT_BG"],font_color=T["TEXT"],
            height=350,showlegend=False,margin=dict(l=10,r=10,t=40,b=10))
        for i in range(1,3):
            fig_gth.update_xaxes(gridcolor=T["GRID"],tickfont=dict(color=T["TEXT"]),row=1,col=i)
            fig_gth.update_yaxes(gridcolor=T["GRID"],tickfont=dict(color=T["TEXT"]),row=1,col=i)
        st.plotly_chart(fig_gth,use_container_width=True)
        section("⏱️ Theta Decay Over Time (ATM)")
        atm_k_zh=round(spot_zh/100)*100; days_arr=list(range(max(days_exp,1),0,-1)); decay_rows=[]
        for d in days_arr:
            p,_,_,th,_=bs_price(spot_zh,atm_k_zh,d/365,risk_free/100,iv_zh,"CE")
            decay_rows.append({"Days":d,"CE_Prem":p,"Theta":abs(th),"Theta_pct":round(abs(th)/p*100,2) if p>0 else 0})
        decay_df=pd.DataFrame(decay_rows)
        fig_dec=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.6,0.4],subplot_titles=("ATM CE vs Days Left","Theta % of Premium"))
        fig_dec.add_trace(go.Scatter(x=decay_df["Days"],y=decay_df["CE_Prem"],fill="tozeroy",fillcolor="rgba(96,165,250,0.12)",
            line=dict(color=T["BLUE"],width=2),name="CE Prem"),row=1,col=1)
        fig_dec.add_trace(go.Bar(x=decay_df["Days"],y=decay_df["Theta_pct"],marker_color=T["RED"],opacity=0.7,name="Theta%"),row=2,col=1)
        fig_dec.update_layout(paper_bgcolor=T["PLOT_PAPER"],plot_bgcolor=T["PLOT_BG"],font_color=T["TEXT"],height=400,margin=dict(l=10,r=10,t=40,b=10))
        fig_dec.update_xaxes(autorange="reversed")
        for i in range(1,3):
            fig_dec.update_xaxes(gridcolor=T["GRID"],tickfont=dict(color=T["TEXT"]),row=i,col=1)
            fig_dec.update_yaxes(gridcolor=T["GRID"],tickfont=dict(color=T["TEXT"]),row=i,col=1)
        st.plotly_chart(fig_dec,use_container_width=True)
        if not zh_oc_df.empty:
            section("📊 Open Interest — CE vs PE")
            fig_oi=go.Figure()
            fig_oi.add_trace(go.Bar(x=zh_oc_df["Strike"],y=zh_oc_df["CE_OI"],name="CE OI",marker_color=T["RED"],opacity=0.8))
            fig_oi.add_trace(go.Bar(x=zh_oc_df["Strike"],y=zh_oc_df["PE_OI"],name="PE OI",marker_color=T["GREEN"],opacity=0.8))
            fig_oi.add_vline(x=zh_spot,line_color=T["ACCENT"],line_dash="dash",annotation_text="Spot")
            if zh_summary: fig_oi.add_vline(x=zh_summary["Max Pain"],line_color=T["PURPLE"],line_dash="dot",annotation_text=f"MaxPain:{zh_summary['Max Pain']:.0f}")
            fig_oi.update_layout(barmode="group",paper_bgcolor=T["PLOT_PAPER"],plot_bgcolor=T["PLOT_BG"],
                font_color=T["TEXT"],height=380,margin=dict(l=10,r=10,t=40,b=10),title="OI Distribution",
                legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=T["TEXT"])))
            fig_oi.update_xaxes(gridcolor=T["GRID"],tickfont=dict(color=T["TEXT"]))
            fig_oi.update_yaxes(gridcolor=T["GRID"],tickfont=dict(color=T["TEXT"]))
            st.plotly_chart(fig_oi,use_container_width=True)
            section("📈 Change in OI")
            fig_coi=go.Figure()
            ce_chg_clr=[T["GREEN"] if v>0 else T["RED"] for v in zh_oc_df["CE_ChgOI"]]
            pe_chg_clr=[T["GREEN"] if v>0 else T["RED"] for v in zh_oc_df["PE_ChgOI"]]
            fig_coi.add_trace(go.Bar(x=zh_oc_df["Strike"],y=zh_oc_df["CE_ChgOI"],name="CE dOI",marker_color=ce_chg_clr,opacity=0.85))
            fig_coi.add_trace(go.Bar(x=zh_oc_df["Strike"],y=zh_oc_df["PE_ChgOI"],name="PE dOI",marker_color=pe_chg_clr,opacity=0.85,visible="legendonly"))
            fig_coi.update_layout(barmode="overlay",paper_bgcolor=T["PLOT_PAPER"],plot_bgcolor=T["PLOT_BG"],
                font_color=T["TEXT"],height=320,margin=dict(l=10,r=10,t=40,b=10),title="Change in OI — Unwinding vs Writing",
                legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=T["TEXT"])))
            fig_coi.update_xaxes(gridcolor=T["GRID"],tickfont=dict(color=T["TEXT"]))
            fig_coi.update_yaxes(gridcolor=T["GRID"],tickfont=dict(color=T["TEXT"]))
            st.plotly_chart(fig_coi,use_container_width=True)
        if zh_summary:
            section("🎯 Zero Hero Trade Recommendation")
            zh_dir="BUY" in zh_summary["Bias"]; zh_opt="CE" if zh_dir else "PE"
            zh_strike=zh_an.get("atm_strike",atm_k_zh)
            zh_sl_pts=round(zh_summary["Straddle Premium"]*0.30,2)
            zh_tgt=round(zh_summary["Straddle Premium"]*0.50,2)
            zh_c=T["GREEN"] if zh_dir else T["RED"]
            st.markdown(f"""<div class="alpha-card">
            <b style="color:{T['ACCENT']};font-size:15px">⚡ Zero Hero Trade Plan</b><br><br>
            <table style="width:100%;border-collapse:collapse;font-size:13px">
            <tr><td style="color:{T['TEXT_MUTED']};padding:5px 8px 5px 0">Direction</td><td><b style="color:{zh_c}">BUY {zh_opt} {"▲" if zh_dir else "▼"}</b></td></tr>
            <tr><td style="color:{T['TEXT_MUTED']}">ATM Strike</td><td><b>Rs{zh_strike} {zh_opt}</b> (highest Delta)</td></tr>
            <tr><td style="color:{T['TEXT_MUTED']}">Entry Window</td><td>9:20–9:45 AM (after open volatility settles)</td></tr>
            <tr><td style="color:{T['TEXT_MUTED']}">Premium (BS)</td><td>Rs{zh_summary['CE Premium'] if zh_opt=='CE' else zh_summary['PE Premium']}</td></tr>
            <tr><td style="color:{T['TEXT_MUTED']}">SL on Premium</td><td style="color:{T['RED']}">Rs{zh_sl_pts} (30% of straddle)</td></tr>
            <tr><td style="color:{T['TEXT_MUTED']}">Target on Prem</td><td style="color:{T['GREEN']}">Rs{zh_tgt} (50% of straddle)</td></tr>
            <tr><td style="color:{T['TEXT_MUTED']}">Hard Exit</td><td>11:30 AM (Theta accelerates after this)</td></tr>
            <tr><td style="color:{T['TEXT_MUTED']}">Avoid if</td><td>IV>25%, PCR neutral, no clear trend signal</td></tr>
            <tr><td style="color:{T['TEXT_MUTED']}">Max Risk</td><td>Full premium paid — never exceed 1% of capital</td></tr>
            </table></div>""",unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"""<div style="text-align:center;padding:10px;color:{T['TEXT_FAINT']};font-size:11px">
⚡ <b style="color:{T['ACCENT']}">AlphaEdge v3</b> | Streamlit + yFinance + Plotly + NSE API | Theme: {THEME} | Rate-limit: {FETCH_DELAY}s<br>
Nifty50 · BankNifty · Sensex · All Nifty50 Stocks · Crypto · Forex · Commodities | TSM · ORB · VWAP · Swing · Scalping · Combined
</div>""",unsafe_allow_html=True)
