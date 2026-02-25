"""
Pro Options Dashboard v5.0 — India + Global
4-source Indian chain waterfall: nsepython → NSE-A → NSE-B → Opstra → CSV upload
All existing BS/signals/backtest/plots logic unchanged.
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests, time, random, warnings, io
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

_T = {"last": 0.0}
def _wait(gap=1.5):
    now = time.time()
    diff = now - _T["last"]
    if diff < gap:
        time.sleep(gap - diff + random.uniform(0.1, 0.3))
    _T["last"] = time.time()

st.set_page_config(page_title="⚡ ProOptions India", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
:root{--bg:#060a10;--s1:#0d1520;--s2:#111d2e;--acc:#00e5ff;--grn:#00ff88;
      --red:#ff3b5c;--ora:#ff9500;--txt:#c8d6e5;--mut:#5a7a9a;--brd:#1e3050}
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:var(--bg)!important;color:var(--txt)!important}
.stApp{background:var(--bg)}
.stMetric{background:var(--s1)!important;border:1px solid var(--brd)!important;border-radius:10px!important;padding:12px!important}
.stMetric label{color:var(--mut)!important;font-size:10px!important;letter-spacing:1px;text-transform:uppercase}
.stMetric [data-testid="stMetricValue"]{color:var(--acc)!important;font-family:'Space Mono',monospace;font-size:18px!important}
div[data-testid="stTabs"] button{color:var(--mut)!important;font-family:'Space Mono',monospace;font-size:11px}
div[data-testid="stTabs"] button[aria-selected="true"]{color:var(--acc)!important;border-bottom:2px solid var(--acc)!important}
h1,h2,h3{font-family:'Space Mono',monospace!important;color:var(--acc)!important}
.stButton>button{background:transparent!important;border:1px solid var(--acc)!important;
  color:var(--acc)!important;font-family:'Space Mono',monospace!important;border-radius:8px!important;font-size:12px!important}
.stButton>button:hover{background:var(--acc)!important;color:#000!important}
.explain{background:var(--s2);border-left:3px solid var(--acc);border-radius:0 8px 8px 0;padding:14px 18px;margin:10px 0;font-size:14px;line-height:1.7}
.explain b{color:var(--acc)}
.sig-box{background:var(--s1);border:1px solid var(--brd);border-radius:12px;padding:18px 22px;margin:10px 0;position:relative;overflow:hidden}
.sig-box::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--acc),var(--grn))}
.tag-b{background:#002a15;color:#00ff88;border:1px solid #00aa44;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;display:inline-block;margin:3px 2px}
.tag-w{background:#2a1a00;color:#ff9500;border:1px solid #aa5500;padding:4px 12px;border-radius:20px;font-size:12px;display:inline-block;margin:3px 2px}
.alert{background:#1a0a00;border-left:3px solid var(--ora);border-radius:0 6px 6px 0;padding:10px 14px;margin:4px 0;font-size:13px}
.gamma-blast{background:#1a000a;border:2px solid #ff3b5c;border-radius:10px;padding:14px;animation:pulse 2s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(255,59,92,.5)}70%{box-shadow:0 0 0 10px rgba(255,59,92,0)}100%{box-shadow:0 0 0 0 rgba(255,59,92,0)}}
.tc{background:var(--s2);border:1px solid var(--brd);border-radius:10px;padding:14px;margin:6px 0}
.tc-o{border-left:3px solid var(--acc)}.tc-w{border-left:3px solid var(--grn)}.tc-l{border-left:3px solid var(--red)}
.card{background:var(--s1);border:1px solid var(--brd);border-radius:12px;padding:18px;margin:8px 0}
.src-badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-family:'Space Mono',monospace;font-weight:700;letter-spacing:1px}
.src-nsepy{background:#003a1a;color:#00ff88;border:1px solid #00aa44}
.src-nse{background:#001a3a;color:#00e5ff;border:1px solid #0066aa}
.src-opstra{background:#2a1500;color:#ff9500;border:1px solid #884400}
.src-csv{background:#1a001a;color:#cc88ff;border:1px solid #664488}
hr{border-color:var(--brd)!important}
</style>""", unsafe_allow_html=True)

RFR = 0.065
DARK = "plotly_dark"

NSE_INDICES = ["NIFTY","BANKNIFTY","SENSEX","FINNIFTY","MIDCPNIFTY"]
NSE_TO_YF   = {"NIFTY":"^NSEI","BANKNIFTY":"^NSEBANK","SENSEX":"^BSESN",
               "FINNIFTY":"NIFTY_FIN_SERVICE.NS","MIDCPNIFTY":"^NSMIDCP"}
NSE_FNO_STOCKS = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN","BHARTIARTL",
    "KOTAKBANK","ITC","LT","AXISBANK","ASIANPAINT","MARUTI","TITAN","ULTRACEMCO",
    "SUNPHARMA","BAJFINANCE","WIPRO","NESTLEIND","TECHM","HCLTECH","POWERGRID",
    "NTPC","ONGC","TATAMOTORS","TATASTEEL","JSWSTEEL","HINDALCO","COALINDIA",
    "ADANIENT","ADANIPORTS","ADANIGREEN","SIEMENS","ABB","HAVELLS","GRASIM",
    "INDUSINDBK","BAJAJFINSV","BAJAJ-AUTO","HEROMOTOCO","EICHERMOT","M&M",
    "TATACONSUM","DIVISLAB","CIPLA","DRREDDY","APOLLOHOSP","LUPIN","BIOCON",
    "TORNTPHARM","PIDILITIND","BERGEPAINT","BHEL","SAIL","NMDC","VEDL","HINDZINC",
    "IOC","BPCL","HPCL","GAIL","PETRONET","CONCOR","IRCTC","DMART","TRENT",
    "NYKAA","ZOMATO","INDIGO","DLF","GODREJPROP","OBEROIRLTY","PRESTIGE",
    "BALKRISIND","APOLLOTYRE","MRF","BANKBARODA","PNB","CANBK","UNIONBANK",
    "FEDERALBNK","IDFCFIRSTB","MUTHOOTFIN","CHOLAFIN","LICHSGFIN","RECLTD","PFC",
    "HDFCLIFE","SBILIFE","ICICIPRULI","LICI","PIIND","UPL","COROMANDEL","GNFC",
    "LTIM","LTTS","PERSISTENT","MPHASIS","COFORGE","KPITTECH","ZEEL","SUNTV",
    "SUZLON","TATAPOWER","NHPC","SJVN","TORNTPOWER","MANAPPURAM","ANGELONE",
    "DIXON","VOLTAS","BLUESTARCO","Custom NSE Symbol"
]
YF_OPTIONS_SYMBOLS = {
    "SPY (S&P 500)":"SPY","QQQ (Nasdaq)":"QQQ","AAPL":"AAPL",
    "TSLA":"TSLA","NVDA":"NVDA","AMZN":"AMZN","META":"META","Custom":"__custom__"
}
YF_SPOT_ONLY = {"BTC/USD":"BTC-USD","GOLD":"GC=F","SILVER":"SI=F",
                "USD/INR":"USDINR=X","CRUDE OIL":"CL=F","Custom":"__custom__"}

for k,v in {"trades":[],"active":[],"bt_result":None,"data_source":""}.items():
    if k not in st.session_state: st.session_state[k] = v

# ═══════════════════════════════════════════
# 4-SOURCE INDIAN OPTIONS WATERFALL
# ═══════════════════════════════════════════

def _parse_nse_json(data):
    rec  = data.get("records", {})
    spot = float(rec.get("underlyingValue") or 0)
    exps = rec.get("expiryDates", [])
    rows = []
    for r in rec.get("data", []):
        ce = r.get("CE", {}); pe = r.get("PE", {})
        rows.append({
            "strikePrice": float(r.get("strikePrice",0)),
            "expiryDate":  r.get("expiryDate",""),
            "CE_LTP":      float(ce.get("lastPrice",0)),
            "CE_OI":       float(ce.get("openInterest",0)),
            "CE_changeOI": float(ce.get("changeinOpenInterest",0)),
            "CE_volume":   float(ce.get("totalTradedVolume",0)),
            "CE_IV":       float(ce.get("impliedVolatility",0)),
            "CE_bid":      float(ce.get("bidprice",0)),
            "CE_ask":      float(ce.get("askPrice",0)),
            "PE_LTP":      float(pe.get("lastPrice",0)),
            "PE_OI":       float(pe.get("openInterest",0)),
            "PE_changeOI": float(pe.get("changeinOpenInterest",0)),
            "PE_volume":   float(pe.get("totalTradedVolume",0)),
            "PE_IV":       float(pe.get("impliedVolatility",0)),
            "PE_bid":      float(pe.get("bidprice",0)),
            "PE_ask":      float(pe.get("askPrice",0)),
        })
    df = pd.DataFrame(rows)
    if df.empty or spot==0: return None,0.0,[]
    return df, spot, exps

def _try_nsepython(symbol):
    try:
        from nsepython import nse_optionchain_scrapper
        raw = nse_optionchain_scrapper(symbol)
        if not raw: return None,0.0,[],"nsepython returned empty"
        df,spot,exps = _parse_nse_json(raw)
        if df is not None: return df,spot,exps,None
        return None,0.0,[],"nsepython parse failed"
    except ImportError:
        return None,0.0,[],"nsepython not installed — run: pip install nsepython"
    except Exception as e:
        return None,0.0,[],f"nsepython: {e}"

def _nse_session_a():
    """Chrome on Windows headers — Strategy A"""
    nav = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
           "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
           "Accept-Language":"en-US,en;q=0.9,hi;q=0.8","Accept-Encoding":"gzip, deflate, br",
           "Connection":"keep-alive","Sec-Fetch-Dest":"document","Sec-Fetch-Mode":"navigate","Upgrade-Insecure-Requests":"1"}
    api = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
           "Accept":"*/*","Accept-Language":"en-US,en;q=0.9,hi;q=0.8","Accept-Encoding":"gzip, deflate, br",
           "Referer":"https://www.nseindia.com/option-chain","X-Requested-With":"XMLHttpRequest",
           "Sec-Fetch-Dest":"empty","Sec-Fetch-Mode":"cors","Sec-Fetch-Site":"same-origin","Connection":"keep-alive"}
    s = requests.Session()
    s.get("https://www.nseindia.com/", timeout=10, headers=nav)
    time.sleep(1.2)
    s.get("https://www.nseindia.com/option-chain", timeout=10, headers=nav)
    time.sleep(1.0)
    return s, api

def _nse_session_b():
    """Chrome on Linux headers — Strategy B via market-data page"""
    ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    nav = {"User-Agent":ua,"Accept":"text/html,*/*","Accept-Language":"en-IN,en;q=0.9,hi;q=0.8","Sec-Fetch-Mode":"navigate"}
    api = {"User-Agent":ua,"Accept":"*/*","Accept-Language":"en-IN,en;q=0.9","X-Requested-With":"XMLHttpRequest",
           "Referer":"https://www.nseindia.com/market-data/equity-derivatives-watch"}
    s = requests.Session()
    s.get("https://www.nseindia.com/", timeout=8, headers=nav)
    time.sleep(1.0)
    s.get("https://www.nseindia.com/market-data/equity-derivatives-watch", timeout=8,
          headers={**nav,"Referer":"https://www.nseindia.com/"})
    time.sleep(0.9)
    s.get("https://www.nseindia.com/get-quotes/derivatives", timeout=8,
          headers={**nav,"Referer":"https://www.nseindia.com/market-data/equity-derivatives-watch"})
    time.sleep(0.8)
    return s, api

def _try_nse_direct(symbol, strategy="A"):
    try:
        s, api_hdrs = _nse_session_a() if strategy=="A" else _nse_session_b()
        url = (f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
               if symbol in NSE_INDICES else
               f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}")
        r = s.get(url, timeout=15, headers=api_hdrs)
        if r.status_code != 200:
            return None,0.0,[],f"NSE-{strategy} HTTP {r.status_code}"
        df,spot,exps = _parse_nse_json(r.json())
        if df is not None: return df,spot,exps,None
        return None,0.0,[],f"NSE-{strategy}: empty/parse error"
    except Exception as e:
        return None,0.0,[],f"NSE-{strategy}: {e}"

def _try_opstra(symbol):
    """Opstra by Definedge — free, no auth, indices only"""
    opstra_map = {"NIFTY":"NIFTY","BANKNIFTY":"BANKNIFTY","FINNIFTY":"FINNIFTY","MIDCPNIFTY":"MIDCPNIFTY"}
    if symbol not in opstra_map:
        return None,0.0,[],f"Opstra supports only indices. {symbol} is a stock — use NSE direct."
    try:
        hdrs = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0",
                "Accept":"application/json,*/*","Referer":"https://opstra.definedge.com/",
                "Origin":"https://opstra.definedge.com"}
        s = requests.Session(); s.headers.update(hdrs)
        er = s.get(f"https://opstra.definedge.com/api/openinterest/expiry/{opstra_map[symbol]}", timeout=10)
        if er.status_code != 200: return None,0.0,[],f"Opstra expiry HTTP {er.status_code}"
        expiries = er.json().get("data",[])
        if not expiries: return None,0.0,[],"Opstra: no expiries"
        expiry = expiries[0]
        or_ = s.get(f"https://opstra.definedge.com/api/openinterest/{opstra_map[symbol]}/{expiry}", timeout=10)
        if or_.status_code != 200: return None,0.0,[],f"Opstra chain HTTP {or_.status_code}"
        jdata = or_.json()
        spot = float(jdata.get("underlyingValue",0) or jdata.get("spotPrice",0) or 0)
        rows = []
        for item in jdata.get("data",[]):
            ce=item.get("CE",{}); pe=item.get("PE",{})
            rows.append({"strikePrice":float(item.get("strikePrice",0)),"expiryDate":expiry,
                "CE_LTP":float(ce.get("lastPrice",0) or ce.get("ltp",0)),
                "CE_OI":float(ce.get("openInterest",0) or ce.get("oi",0)),
                "CE_changeOI":float(ce.get("changeinOpenInterest",0) or ce.get("changeInOI",0)),
                "CE_volume":float(ce.get("totalTradedVolume",0) or ce.get("volume",0)),
                "CE_IV":float(ce.get("impliedVolatility",0) or ce.get("iv",0)),
                "CE_bid":float(ce.get("bidprice",0)),"CE_ask":float(ce.get("askPrice",0)),
                "PE_LTP":float(pe.get("lastPrice",0) or pe.get("ltp",0)),
                "PE_OI":float(pe.get("openInterest",0) or pe.get("oi",0)),
                "PE_changeOI":float(pe.get("changeinOpenInterest",0) or pe.get("changeInOI",0)),
                "PE_volume":float(pe.get("totalTradedVolume",0) or pe.get("volume",0)),
                "PE_IV":float(pe.get("impliedVolatility",0) or pe.get("iv",0)),
                "PE_bid":float(pe.get("bidprice",0)),"PE_ask":float(pe.get("askPrice",0))})
        df = pd.DataFrame(rows)
        if df.empty: return None,0.0,[],"Opstra: empty chain"
        if spot==0:
            yf_sym = NSE_TO_YF.get(symbol,"^NSEI")
            try:
                h = yf.Ticker(yf_sym).history(period="1d")
                if not h.empty: spot = float(h["Close"].squeeze().iloc[-1])
            except: pass
        return df, spot, expiries, None
    except Exception as e:
        return None,0.0,[],f"Opstra: {e}"

@st.cache_data(ttl=90, show_spinner=False)
def fetch_india_chain(symbol):
    errors = []
    for src_fn, src_name in [
        (lambda s: _try_nsepython(s),      "nsepython"),
        (lambda s: _try_nse_direct(s,"A"), "NSE-A"),
        (lambda s: _try_nse_direct(s,"B"), "NSE-B"),
        (lambda s: _try_opstra(s),         "Opstra"),
    ]:
        df,spot,exps,err = src_fn(symbol)
        if df is not None: return df,spot,exps,src_name,errors
        errors.append(f"{src_name} → {err}")
    return None,0.0,[],None,errors

def parse_csv_upload(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip() for c in df.columns]
        # NSE Bhavcopy
        if "STRIKE_PR" in df.columns:
            df = df.rename(columns={"STRIKE_PR":"strikePrice"})
            calls = df[df["OPTION_TYP"]=="CE"]; puts = df[df["OPTION_TYP"]=="PE"]
            merged = calls.rename(columns={"CLOSE":"CE_LTP","OPEN_INT":"CE_OI",
                "CHG_IN_OI":"CE_changeOI","CONTRACTS":"CE_volume"}).merge(
                puts.rename(columns={"CLOSE":"PE_LTP","OPEN_INT":"PE_OI",
                    "CHG_IN_OI":"PE_changeOI","CONTRACTS":"PE_volume"}
                )[["strikePrice","PE_LTP","PE_OI","PE_changeOI","PE_volume"]],
                on="strikePrice", how="outer")
            for c in ["CE_IV","PE_IV","CE_bid","CE_ask","PE_bid","PE_ask","CE_changeOI","PE_changeOI"]:
                if c not in merged.columns: merged[c]=0
            if "EXPIRY_DT" in calls.columns:
                merged["expiryDate"] = calls["EXPIRY_DT"].iloc[0] if not calls.empty else "Uploaded"
            return merged.fillna(0)
        # Already correct format
        if all(c in df.columns for c in ["strikePrice","CE_LTP","PE_LTP"]):
            for c in ["CE_OI","PE_OI","CE_IV","PE_IV","CE_changeOI","PE_changeOI",
                      "CE_volume","PE_volume","CE_bid","CE_ask","PE_bid","PE_ask"]:
                if c not in df.columns: df[c]=0
            if "expiryDate" not in df.columns: df["expiryDate"]="Uploaded"
            return df.fillna(0)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"CSV parse error: {e}"); return pd.DataFrame()

# ═══════════════════════════════════════════
# YFINANCE
# ═══════════════════════════════════════════
@st.cache_data(ttl=60, show_spinner=False)
def yf_spot(sym):
    _wait(1.5)
    try:
        fi = yf.Ticker(sym).fast_info
        for attr in ("last_price","lastPrice","regular_market_price","regularMarketPrice","previousClose"):
            v = getattr(fi,attr,None)
            if v and float(v)>0: return float(v)
    except: pass
    _wait(1.5)
    try:
        h = yf.Ticker(sym).history(period="2d",interval="1d")
        if not h.empty: return float(h["Close"].squeeze().iloc[-1])
    except: pass
    return 0.0

@st.cache_data(ttl=120, show_spinner=False)
def yf_expiries(sym):
    _wait(1.5)
    try: return list(yf.Ticker(sym).options or [])
    except: return []

@st.cache_data(ttl=120, show_spinner=False)
def yf_chain(sym, expiry):
    _wait(1.5)
    try:
        c = yf.Ticker(sym).option_chain(expiry)
        return c.calls.copy(), c.puts.copy()
    except Exception as e:
        st.error(f"Chain error: {e}"); return None,None

@st.cache_data(ttl=300, show_spinner=False)
def yf_history(sym, period="60d"):
    _wait(1.5)
    try:
        df = yf.Ticker(sym).history(period=period,interval="1d",auto_adjust=True)
        return df if df is not None and not df.empty else pd.DataFrame()
    except: return pd.DataFrame()

def build_chain_df(calls, puts, expiry):
    if calls is None or calls.empty: return pd.DataFrame()
    df = pd.DataFrame()
    df["strikePrice"] = calls["strike"].values if "strike" in calls.columns else []
    df["expiryDate"]  = expiry
    for col,src in [("CE_LTP","lastPrice"),("CE_OI","openInterest"),("CE_volume","volume"),("CE_bid","bid"),("CE_ask","ask")]:
        df[col] = calls[src].values if src in calls.columns else 0
    df["CE_IV"]       = (calls["impliedVolatility"].values*100) if "impliedVolatility" in calls.columns else 0
    df["CE_changeOI"] = 0
    if puts is not None and not puts.empty and "strike" in puts.columns:
        pi = puts.set_index("strike")
        for col,src in [("PE_LTP","lastPrice"),("PE_OI","openInterest"),("PE_volume","volume"),("PE_bid","bid"),("PE_ask","ask")]:
            df[col] = df["strikePrice"].map(pi[src] if src in pi.columns else pd.Series(dtype=float)).fillna(0).values
        df["PE_IV"] = df["strikePrice"].map(pi["impliedVolatility"]*100 if "impliedVolatility" in pi.columns else pd.Series(dtype=float)).fillna(0).values
    else:
        for c in ["PE_LTP","PE_OI","PE_volume","PE_bid","PE_ask","PE_IV"]: df[c]=0
    df["PE_changeOI"]=0
    return df.fillna(0).reset_index(drop=True)

# ═══════════════════════════════════════════
# BLACK-SCHOLES (unchanged)
# ═══════════════════════════════════════════
def bs(S,K,T,r,sig,kind="call"):
    if T<1e-6 or sig<1e-6 or S<=0 or K<=0: return dict(delta=0,gamma=0,theta=0,vega=0,price=0)
    d1=(np.log(S/K)+(r+.5*sig**2)*T)/(sig*np.sqrt(T)); d2=d1-sig*np.sqrt(T)
    if kind=="call":
        price=S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2); delta=norm.cdf(d1)
        theta=(-(S*norm.pdf(d1)*sig)/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2))/365
    else:
        price=K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1); delta=norm.cdf(d1)-1
        theta=(-(S*norm.pdf(d1)*sig)/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2))/365
    gamma=norm.pdf(d1)/(S*sig*np.sqrt(T)); vega=S*norm.pdf(d1)*np.sqrt(T)/100
    return dict(delta=round(delta,4),gamma=round(gamma,8),theta=round(theta,4),vega=round(vega,4),price=round(max(price,0),4))

def calc_iv(mkt,S,K,T,r,kind="call"):
    if T<=0 or mkt<=0 or S<=0: return 0.20
    try: return max(brentq(lambda s:bs(S,K,T,r,s,kind)["price"]-mkt,1e-4,20.0,xtol=1e-5),0.001)
    except: return 0.20

def add_greeks(df,spot,expiry_str):
    if df is None or df.empty: return df
    try:
        fmt="%d-%b-%Y" if "-" in str(expiry_str) and not str(expiry_str)[4:5].isdigit() else "%Y-%m-%d"
        T=max((datetime.strptime(str(expiry_str),fmt)-datetime.now()).days/365.0,1/365)
    except: T=7/365
    df=df.copy()
    for i,row in df.iterrows():
        K=float(row["strikePrice"])
        for kind,px in [("call","CE"),("put","PE")]:
            ltp=float(row.get(f"{px}_LTP",0)); iv_pct=float(row.get(f"{px}_IV",0))
            sig=iv_pct/100 if iv_pct>0.5 else calc_iv(ltp,spot,K,T,RFR,kind)
            sig=max(sig,0.01); g=bs(spot,K,T,RFR,sig,kind)
            df.at[i,f"{px}_delta"]=g["delta"]; df.at[i,f"{px}_gamma"]=g["gamma"]
            df.at[i,f"{px}_theta"]=g["theta"]; df.at[i,f"{px}_vega"]=g["vega"]
            if iv_pct==0 and ltp>0: df.at[i,f"{px}_IV"]=round(sig*100,2)
    return df

# ═══════════════════════════════════════════
# SIGNALS (unchanged)
# ═══════════════════════════════════════════
def compute_signals(df,spot):
    out=dict(pcr=0,max_pain=spot,atm=spot,straddle=0,resistance=spot,support=spot,
             atm_iv=20.0,skew=0.0,gamma_blast=False,abnormal=[],
             rec="⚪ NO TRADE",direction="NONE",conf=0,
             scalp="WAIT",intraday="WAIT",swing="WAIT",pos="WAIT",reasons=[])
    if df is None or df.empty or spot==0: return out
    df=df.copy(); df["strikePrice"]=df["strikePrice"].astype(float)
    ai=(df["strikePrice"]-spot).abs().idxmin(); atm=df.loc[ai]
    out["atm"]=float(atm["strikePrice"])
    out["straddle"]=round(float(atm.get("CE_LTP",0))+float(atm.get("PE_LTP",0)),2)
    ce_oi=float(df["CE_OI"].sum()); pe_oi=float(df["PE_OI"].sum())
    out["pcr"]=round(pe_oi/ce_oi,4) if ce_oi>0 else 0
    strikes=df["strikePrice"].values; c_oi=df["CE_OI"].values; p_oi=df["PE_OI"].values
    pain=[sum(max(0,k-s)*o for k,o in zip(strikes,c_oi))+sum(max(0,s-k)*o for k,o in zip(strikes,p_oi)) for s in strikes]
    out["max_pain"]=float(strikes[int(np.argmin(pain))]) if pain else spot
    out["resistance"]=float(df.loc[df["CE_OI"].idxmax(),"strikePrice"])
    out["support"]=float(df.loc[df["PE_OI"].idxmax(),"strikePrice"])
    ce_iv=float(atm.get("CE_IV",0)); pe_iv=float(atm.get("PE_IV",0))
    out["atm_iv"]=(ce_iv+pe_iv)/2 if (ce_iv+pe_iv)>0 else 20.0; out["skew"]=round(pe_iv-ce_iv,2)
    near=(df["CE_OI"].where(df["strikePrice"].between(spot*.99,spot*1.01),0).sum()+
          df["PE_OI"].where(df["strikePrice"].between(spot*.99,spot*1.01),0).sum())
    out["gamma_blast"]=(ce_oi+pe_oi)>0 and near/(ce_oi+pe_oi+1)>0.35
    iv=out["atm_iv"]
    if iv>55: out["abnormal"].append(f"🔴 IV {iv:.0f}%: Options VERY expensive — don't buy, premium crashes after event")
    elif iv>35: out["abnormal"].append(f"⚠️ IV {iv:.0f}%: Costly — theta will eat profits faster than usual")
    elif 0<iv<15: out["abnormal"].append(f"✅ IV {iv:.0f}%: Options CHEAP — best time to buy, getting premium at discount!")
    sk=out["skew"]
    if sk>8: out["abnormal"].append(f"📊 Puts costlier than calls (skew +{sk:.0f}%): Big money hedging a fall — lean BEARISH")
    elif sk<-8: out["abnormal"].append(f"📊 Calls costlier than puts (skew {sk:.0f}%): Aggressive call buying — lean BULLISH")
    mp_pct=(out["max_pain"]-spot)/spot*100 if spot>0 else 0
    if abs(mp_pct)>2: out["abnormal"].append(f"🎯 Max pain {mp_pct:+.1f}% away at {out['max_pain']:,.0f} — market drifts here near expiry")
    if out["gamma_blast"]: out["abnormal"].append("⚡ GAMMA BLAST: Huge OI at current price — options will explode on breakout. Wait for direction then enter fast.")
    score=50; direction="NONE"; reasons=[]
    pcr=out["pcr"]
    if pcr>1.5: score+=12; direction="CALL"; reasons.append(f"PCR {pcr:.2f}: Too many puts — contrarian bullish, market likely UP")
    elif pcr>1.1: score+=6; direction="CALL"; reasons.append(f"PCR {pcr:.2f}: More puts than calls — mild bullish lean")
    elif pcr<0.5: score+=12; direction="PUT"; reasons.append(f"PCR {pcr:.2f}: Too many calls — contrarian bearish, market likely DOWN")
    elif pcr<0.9: score+=6; direction="PUT"; reasons.append(f"PCR {pcr:.2f}: More calls than puts — mild bearish lean")
    else: reasons.append(f"PCR {pcr:.2f}: Balanced — no clear direction")
    if iv>50: score-=25; reasons.append("IV >50%: AVOID buying — overpriced")
    elif iv>35: score-=12; reasons.append(f"IV {iv:.0f}%: Expensive, decay will hurt")
    elif 0<iv<18: score+=15; reasons.append(f"IV {iv:.0f}%: Cheap — perfect buying conditions!")
    elif iv<=30: score+=8; reasons.append(f"IV {iv:.0f}%: Reasonable")
    mp=out["max_pain"]
    if spot<mp and direction=="CALL": score+=10; reasons.append(f"Below max pain {mp:,.0f} — gravity pulls UP")
    elif spot>mp and direction=="PUT": score+=10; reasons.append(f"Above max pain {mp:,.0f} — gravity pulls DOWN")
    elif spot<mp and direction=="PUT": score-=8
    elif spot>mp and direction=="CALL": score-=8
    if direction=="CALL" and out["support"]<spot<out["resistance"]: score+=8; reasons.append(f"In buy zone between S:{out['support']:,.0f} & R:{out['resistance']:,.0f}")
    if direction=="PUT"  and out["support"]<spot<out["resistance"]: score+=8; reasons.append(f"Near resistance ceiling {out['resistance']:,.0f}")
    ce_chg=float(df["CE_changeOI"].sum()); pe_chg=float(df["PE_changeOI"].sum())
    if pe_chg>ce_chg*1.3 and direction=="CALL": score+=7; reasons.append("Fresh put writing: support floor building")
    if ce_chg>pe_chg*1.3 and direction=="PUT": score+=7; reasons.append("Fresh call writing: ceiling building")
    score=min(max(int(score),0),100); out["conf"]=score; out["direction"]=direction; out["reasons"]=reasons
    A=int(out["atm"]); otm=int(A*(1.005 if direction=="CALL" else 0.995))
    if score>=72 and direction=="CALL":
        out["rec"]="🟢 BUY CALL (CE)"; out["scalp"]=f"Buy {A} CE — exit +20 to 40%, SL -30%"
        out["intraday"]=f"Buy {A} CE — target +60%, SL -30%"
        out["swing"]=f"Buy {otm} CE next expiry — target +80 to 100%"
        out["pos"]=f"Buy {otm} CE far expiry — hold for big move"
    elif score>=72 and direction=="PUT":
        out["rec"]="🔴 BUY PUT (PE)"; out["scalp"]=f"Buy {A} PE — exit +20 to 40%, SL -30%"
        out["intraday"]=f"Buy {A} PE — target +60%, SL -30%"
        out["swing"]=f"Buy {otm} PE next expiry — target +80 to 100%"
        out["pos"]=f"Buy {otm} PE far expiry — hold for big move"
    elif score>=58:
        out["rec"]="🟡 WATCH — Signal forming"
        out["scalp"]=out["intraday"]=out["swing"]=out["pos"]="Wait for confirmation"
    else:
        out["rec"]="⚪ NO TRADE — Stay out"
        out["scalp"]=out["intraday"]=out["swing"]=out["pos"]="No edge. Cash is a position."
    return out

# ═══════════════════════════════════════════
# BACKTEST (unchanged)
# ═══════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def run_backtest(hist_sym,lookback,capital,pos_pct):
    raw=yf_history(hist_sym,f"{lookback}d")
    if raw.empty or len(raw)<15: return None,"Not enough data."
    close=raw["Close"].squeeze().astype(float)
    vol7=close.pct_change().rolling(7).std()*np.sqrt(252)*100
    sma10=close.rolling(10).mean(); sma20=close.rolling(20).mean()
    vol3=close.pct_change().rolling(3).std()*np.sqrt(252)*100
    vol10=close.pct_change().rolling(10).std()*np.sqrt(252)*100
    trades=[]; equity=float(capital); curve=[equity]
    for i in range(20,len(close)-1):
        iv=float(vol7.iloc[i]) if not np.isnan(vol7.iloc[i]) else 20.0
        spot=float(close.iloc[i])
        v3=float(vol3.iloc[i]) if not np.isnan(vol3.iloc[i]) else 20.0
        v10=float(vol10.iloc[i]) if not np.isnan(vol10.iloc[i]) else 20.0
        pcr=1.0+(v10-v3)/(v10+1); mp=(float(sma10.iloc[i])+float(sma20.iloc[i]))/2
        score=50; direction="NONE"
        if pcr>1.5: score+=12; direction="CALL"
        elif pcr>1.1: score+=6; direction="CALL"
        elif pcr<0.5: score+=12; direction="PUT"
        elif pcr<0.9: score+=6; direction="PUT"
        if iv>50: score-=25
        elif iv>35: score-=12
        elif iv<18: score+=15
        elif iv<=30: score+=8
        if spot<mp and direction=="CALL": score+=10
        elif spot>mp and direction=="PUT": score+=10
        elif spot<mp and direction=="PUT": score-=8
        elif spot>mp and direction=="CALL": score-=8
        if float(sma10.iloc[i])>float(sma20.iloc[i]) and direction=="CALL": score+=7
        elif float(sma10.iloc[i])<float(sma20.iloc[i]) and direction=="PUT": score+=7
        score=min(max(int(score),0),100); curve.append(equity)
        if score<72 or direction=="NONE": continue
        opt_px=spot*max(iv/100,0.01)*np.sqrt(7/365)*0.4
        if opt_px<=0: continue
        ret=(float(close.iloc[i+1])-spot)/spot
        raw_gain=(ret*0.5*spot) if direction=="CALL" else (-ret*0.5*spot)
        pnl_pct=max(min((raw_gain-opt_px/7)/opt_px,2.5),-0.5)
        pnl=equity*(pos_pct/100)*pnl_pct; equity=max(equity+pnl,0)
        trades.append({"Date":raw.index[i].strftime("%d-%b-%Y"),"Direction":direction,
            "Spot":round(spot,2),"Est.OptPx":round(opt_px,2),"IV%":round(iv,1),
            "Score":score,"NextDay%":round(ret*100,2),"P&L(₹)":round(pnl,2),
            "P&L(%)":round(pnl_pct*100,2),"Equity":round(equity,2),
            "Result":"✅ WIN" if pnl>0 else "❌ LOSS"})
        curve[-1]=equity
    if not trades: return None,"No trades — try longer lookback."
    tdf=pd.DataFrame(trades)
    wins=(tdf["P&L(₹)"]>0).sum(); total=len(tdf)
    aw=tdf.loc[tdf["P&L(₹)"]>0,"P&L(₹)"].mean() if wins>0 else 0
    al=tdf.loc[tdf["P&L(₹)"]<=0,"P&L(₹)"].mean() if (total-wins)>0 else 0
    peak=capital; mdd=0
    for eq in curve:
        if eq>peak: peak=eq
        mdd=max(mdd,(peak-eq)/peak*100)
    stats={"total":total,"wins":int(wins),"wr":round(wins/total*100,1),
           "total_pnl":round(tdf["P&L(₹)"].sum(),2),"aw":round(aw,2),"al":round(al,2),
           "rr":round(abs(aw/al),2) if al!=0 else 0,"mdd":round(mdd,2),
           "final":round(equity,2),"ret%":round((equity-capital)/capital*100,2),"curve":curve}
    return tdf,stats

# ═══════════════════════════════════════════
# PLOTS (unchanged, all colors valid)
# ═══════════════════════════════════════════
def vl(fig,x,lbl="",color="yellow",dash="dash",row=1,col=1):
    fig.add_vline(x=x,line_dash=dash,line_color=color,line_width=1.5,
                  annotation_text=lbl,annotation_font_color=color,row=row,col=col)

def plot_chain(df,spot):
    rng=spot*0.05; sub=df[df["strikePrice"].between(spot-rng,spot+rng)]
    if sub.empty: sub=df
    fig=make_subplots(2,2,subplot_titles=["CE vs PE Premium","Open Interest","IV Smile","Volume"])
    x=sub["strikePrice"]
    fig.add_trace(go.Bar(name="CE",x=x,y=sub["CE_LTP"],marker_color="#00ff88",opacity=.85),1,1)
    fig.add_trace(go.Bar(name="PE",x=x,y=sub["PE_LTP"],marker_color="#ff3b5c",opacity=.85),1,1)
    fig.add_trace(go.Bar(name="CE OI",x=x,y=sub["CE_OI"],marker_color="#00e5ff",opacity=.8),1,2)
    fig.add_trace(go.Bar(name="PE OI",x=x,y=sub["PE_OI"],marker_color="#ff9500",opacity=.8),1,2)
    fig.add_trace(go.Scatter(name="CE IV%",x=x,y=sub["CE_IV"],mode="lines+markers",line=dict(color="#00ff88",width=2)),2,1)
    fig.add_trace(go.Scatter(name="PE IV%",x=x,y=sub["PE_IV"],mode="lines+markers",line=dict(color="#ff3b5c",width=2)),2,1)
    fig.add_trace(go.Bar(name="CE Vol",x=x,y=sub["CE_volume"],marker_color="rgba(0,255,136,0.4)"),2,2)
    fig.add_trace(go.Bar(name="PE Vol",x=x,y=sub["PE_volume"],marker_color="rgba(255,59,92,0.4)"),2,2)
    for r in [1,2]:
        for c in [1,2]: vl(fig,spot,"Spot",row=r,col=c)
    fig.update_layout(template=DARK,height=520,barmode="group",title="Option Chain",margin=dict(t=50,b=10))
    return fig

def plot_oi(df,spot,sig):
    rng=spot*0.055; sub=df[df["strikePrice"].between(spot-rng,spot+rng)]
    if sub.empty: sub=df
    sub=sub.copy()
    sub["CE_pct"]=(sub["CE_changeOI"]/(sub["CE_OI"]-sub["CE_changeOI"]).clip(lower=1)*100).fillna(0)
    sub["PE_pct"]=(sub["PE_changeOI"]/(sub["PE_OI"]-sub["PE_changeOI"]).clip(lower=1)*100).fillna(0)
    fig=make_subplots(3,1,subplot_titles=["Total OI","Change in OI today","% Change in OI"])
    x=sub["strikePrice"]
    fig.add_trace(go.Bar(name="CE OI",x=x,y=sub["CE_OI"],marker_color="#00e5ff",opacity=.8),1,1)
    fig.add_trace(go.Bar(name="PE OI",x=x,y=sub["PE_OI"],marker_color="#ff9500",opacity=.8),1,1)
    fig.add_trace(go.Bar(name="ΔCE",x=x,y=sub["CE_changeOI"],marker_color=["#00ff88" if v>=0 else "#ff3b5c" for v in sub["CE_changeOI"]]),2,1)
    fig.add_trace(go.Bar(name="ΔPE",x=x,y=sub["PE_changeOI"],marker_color=["#ff9500" if v>=0 else "#8888ff" for v in sub["PE_changeOI"]]),2,1)
    fig.add_trace(go.Bar(name="CE%",x=x,y=sub["CE_pct"],marker_color=["rgba(0,255,136,0.47)" if v>=0 else "rgba(255,59,92,0.47)" for v in sub["CE_pct"]]),3,1)
    fig.add_trace(go.Bar(name="PE%",x=x,y=sub["PE_pct"],marker_color=["rgba(255,149,0,0.47)" if v>=0 else "rgba(136,136,255,0.47)" for v in sub["PE_pct"]]),3,1)
    for row in [1,2,3]:
        vl(fig,spot,"Spot",row=row)
        if sig["resistance"]: vl(fig,sig["resistance"],"R",color="#ff3b5c",dash="dot",row=row,col=1)
        if sig["support"]:    vl(fig,sig["support"],"S",color="#00ff88",dash="dot",row=row,col=1)
    fig.update_layout(template=DARK,height=620,barmode="group",title="OI Analysis",margin=dict(t=50,b=10))
    return fig

def plot_greeks_chart(df,spot):
    rng=spot*0.04; sub=df[df["strikePrice"].between(spot-rng,spot+rng)]
    if sub.empty: sub=df
    fig=make_subplots(2,2,subplot_titles=["Delta","Gamma","Theta (daily decay)","Vega"])
    x=sub["strikePrice"]
    for (r,c,cc,pc) in [(1,1,"CE_delta","PE_delta"),(1,2,"CE_gamma","PE_gamma"),(2,1,"CE_theta","PE_theta"),(2,2,"CE_vega","PE_vega")]:
        for col,color in [(cc,"#00ff88"),(pc,"#ff3b5c")]:
            if col in sub.columns:
                fig.add_trace(go.Scatter(name=col,x=x,y=sub[col],mode="lines",line=dict(color=color,width=2)),r,c)
        vl(fig,spot,row=r,col=c)
    fig.update_layout(template=DARK,height=520,title="Greeks",margin=dict(t=50,b=10))
    return fig

def plot_straddle(hist_sym,straddle_now):
    hist=yf_history(hist_sym,"45d")
    if hist.empty: return None,None
    close=hist["Close"].squeeze().astype(float)
    rv=close.pct_change().rolling(7).std()*np.sqrt(252)
    est=(close*rv*np.sqrt(7/252)*0.8).dropna()
    if est.empty: return None,None
    p25,p50,p75=est.quantile([.25,.5,.75])
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=est.index,y=est,name="Historical estimate",line=dict(color="#00e5ff",width=2)))
    if straddle_now>0:
        fig.add_hline(y=straddle_now,line_color="yellow",line_dash="dash",annotation_text=f"Today:{straddle_now:.1f}",annotation_font_color="yellow")
    fig.add_hline(y=p25,line_color="#00ff88",line_dash="dot",annotation_text=f"Cheap:{p25:.1f}")
    fig.add_hline(y=p75,line_color="#ff3b5c",line_dash="dot",annotation_text=f"Expensive:{p75:.1f}")
    fig.add_hline(y=p50,line_color="#ff9500",line_dash="dot",annotation_text=f"Average:{p50:.1f}")
    fig.update_layout(template=DARK,height=320,title="Straddle vs 45-Day History")
    if straddle_now>0:
        if straddle_now<p25:   v=("✅ CHEAP — Options below average. Great time to buy!","#00ff88")
        elif straddle_now>p75: v=("⚠️ EXPENSIVE — Options overpriced. Avoid buying now.","#ff3b5c")
        else:                   v=("🟡 FAIR VALUE — Normal pricing. Be selective.","#ff9500")
    else: v=None
    return fig,v

# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    st.markdown("<h1 style='font-size:24px;letter-spacing:2px'>⚡ PRO OPTIONS INTELLIGENCE — INDIA</h1>",unsafe_allow_html=True)
    st.markdown("<p style='color:#5a7a9a;margin-top:-10px'>4-Source Indian Chain · 100+ F&O Stocks · Greeks · OI · Signals · Backtest · Plain English</p>",unsafe_allow_html=True)
    st.divider()

    with st.sidebar:
        st.markdown("### ⚙️ Instrument")
        src_type=st.selectbox("Market",["🇮🇳 NSE India","🌐 Global (US Stocks)","📊 Spot Only (BTC/Gold/FX)"])
        fetch_sym=""; hist_sym=""; is_nse=False; has_opts=True

        if "NSE India" in src_type:
            is_nse=True
            nse_cat=st.radio("Category",["Indices","F&O Stocks"],horizontal=True)
            if nse_cat=="Indices":
                nse_sym=st.selectbox("Index",NSE_INDICES)
            else:
                nse_sym=st.selectbox("Stock",NSE_FNO_STOCKS)
                if nse_sym=="Custom NSE Symbol":
                    nse_sym=st.text_input("NSE symbol","RELIANCE").upper().strip()
            fetch_sym=nse_sym; hist_sym=NSE_TO_YF.get(nse_sym,nse_sym+".NS")
            st.markdown("""<div style='background:#0d2a0d;border:1px solid #1a5c1a;border-radius:8px;padding:10px;font-size:12px;margin-top:4px'>
<b style='color:#00ff88'>🔄 Auto-tries 4 sources:</b><br>
<span style='color:#8899aa'>1. nsepython (most reliable)<br>2. NSE direct API route A<br>3. NSE direct API route B<br>4. Opstra (indices only)</span>
</div>""",unsafe_allow_html=True)
            st.markdown("**For best results, run once:**")
            st.code("pip install nsepython",language="bash")

        elif "Global" in src_type:
            choice=st.selectbox("Instrument",list(YF_OPTIONS_SYMBOLS.keys()))
            fetch_sym=st.text_input("yFinance ticker","AAPL").upper().strip() if choice=="Custom" else YF_OPTIONS_SYMBOLS[choice]
            hist_sym=fetch_sym
        else:
            choice=st.selectbox("Instrument",list(YF_SPOT_ONLY.keys()))
            fetch_sym=st.text_input("yFinance ticker","BTC-USD").upper().strip() if choice=="Custom" else YF_SPOT_ONLY[choice]
            hist_sym=fetch_sym; has_opts=False
            st.warning("No options for BTC/Gold/FX. Spot + volatility only.")

        st.divider()
        if is_nse:
            st.markdown("### 📂 CSV Fallback")
            st.markdown('<span style="font-size:12px;color:#5a7a9a">If all sources fail: download from<br><b>nseindia.com → Option Chain → Download</b></span>',unsafe_allow_html=True)
            csv_file=st.file_uploader("Upload NSE CSV",type=["csv"],key="chain_csv")
        else:
            csv_file=None

        st.divider()
        st.markdown("### 💰 Risk Settings")
        capital=st.number_input("Capital (₹)",50_000,10_000_000,100_000,10_000)
        pos_pct=st.slider("% per trade",2,15,5)
        sl_pct=st.slider("Stop loss %",20,50,30)
        tgt_pct=st.slider("Target %",30,200,60)
        st.divider()
        st.button("🔄 FETCH LIVE DATA",type="primary",use_container_width=True)
        if st.button("🗑️ Clear Cache & Retry",use_container_width=True):
            st.cache_data.clear(); _T["last"]=0.0; st.rerun()
        auto_ref=st.checkbox("Auto-refresh every 90s")
        st.caption("⚠️ Educational only. Not financial advice.")

    # ── FETCH ──────────────────────────────
    df_exp=pd.DataFrame(); spot=0.0; expiries=[]; sel_expiry=""; data_src=""; fetch_errors=[]
    status_ph=st.empty()

    with st.spinner(f"Fetching {fetch_sym}…"):
        if is_nse:
            # CSV takes priority if uploaded
            if csv_file is not None:
                status_ph.info("📂 Parsing uploaded CSV…")
                df_csv=parse_csv_upload(csv_file)
                if not df_csv.empty and "strikePrice" in df_csv.columns:
                    df_exp=df_csv; data_src="csv"; spot=yf_spot(hist_sym)
                    expiries=df_csv["expiryDate"].unique().tolist() if "expiryDate" in df_csv.columns else ["Uploaded"]
                    sel_expiry=expiries[0]
                    status_ph.success(f"✅ CSV loaded — {len(df_exp)} strikes | Spot ₹{spot:,.2f}")

            if df_exp.empty:
                status_ph.info("📡 Trying 4 sources for Indian option chain…")
                df_raw,spot_raw,exps_raw,src,errs=fetch_india_chain(fetch_sym)
                fetch_errors=errs
                if df_raw is not None and spot_raw>0:
                    df_exp=df_raw; spot=spot_raw; expiries=exps_raw; data_src=src
                    status_ph.success(f"✅ {fetch_sym} loaded via **{src}** — {len(df_exp)} strikes | Spot ₹{spot:,.2f}")
                else:
                    spot=yf_spot(hist_sym); has_opts=False
                    if spot>0:
                        status_ph.error(f"❌ All sources failed for {fetch_sym} chain. Spot ₹{spot:,.2f} from yFinance.")
                        with st.expander("🔍 What failed (click to debug)"):
                            for e in fetch_errors: st.caption(e)
                        st.warning("**Fix options:** `pip install nsepython` then restart app.\n\nOr download CSV from nseindia.com → Option Chain → Download and upload in sidebar.\n\nChain only available 9:15 AM–3:30 PM IST on trading days.")
                    else:
                        status_ph.error("❌ No data at all. Check internet.")

            if not df_exp.empty and expiries:
                with st.sidebar:
                    sel_expiry=st.selectbox("Expiry",expiries[:10])
                if "expiryDate" in df_exp.columns:
                    mask=df_exp["expiryDate"]==sel_expiry
                    df_exp=df_exp[mask].copy() if mask.any() else df_exp.copy()

        elif has_opts:
            status_ph.info("📡 Fetching from yFinance…")
            spot=yf_spot(fetch_sym)
            if spot>0:
                expiries=yf_expiries(fetch_sym)
                if expiries:
                    with st.sidebar:
                        sel_expiry=st.selectbox("Expiry",expiries[:8])
                    calls,puts=yf_chain(fetch_sym,sel_expiry)
                    if calls is not None and not calls.empty:
                        df_exp=build_chain_df(calls,puts,sel_expiry); data_src="yfinance"
                        status_ph.success(f"✅ {fetch_sym} — {len(df_exp)} strikes | Spot {spot:,.2f}")
                    else:
                        status_ph.error("Chain fetch failed. Try again in 15s."); has_opts=False
                else:
                    status_ph.warning(f"No options for {fetch_sym} (market closed?)."); has_opts=False
            else:
                status_ph.error(f"Cannot get price for {fetch_sym}.")
        else:
            spot=yf_spot(fetch_sym)
            if spot>0: status_ph.success(f"✅ {fetch_sym}: {spot:,.4f}")
            else: status_ph.error(f"Cannot get price for {fetch_sym}.")

    if spot==0:
        st.error("❌ No data. Try Clear Cache, check internet, or upload CSV.")
        return

    spot=float(spot)
    has_chain=has_opts and not df_exp.empty and "strikePrice" in df_exp.columns

    if has_chain:
        with st.spinner("Computing Greeks…"):
            df_exp=add_greeks(df_exp,spot,sel_expiry)
        all_strikes=sorted(df_exp["strikePrice"].unique().tolist())
        atm_pos=min(range(len(all_strikes)),key=lambda i:abs(all_strikes[i]-spot))
    else:
        all_strikes=[]; atm_pos=0

    sig=compute_signals(df_exp,spot)

    # Source badge
    src_cls={"nsepython":"src-nsepy","NSE-A":"src-nse","NSE-B":"src-nse","Opstra":"src-opstra","csv":"src-csv","yfinance":"src-nse","":""}
    src_badge=f'<span class="src-badge {src_cls.get(data_src,"src-nse")}">DATA: {data_src.upper() or "—"}</span>' if data_src else ""

    # ── METRICS ──
    cols=st.columns(7)
    m=[("📍 Spot",f"₹{spot:,.2f}"),
       ("🎯 ATM",f"{sig['atm']:,.0f}" if has_chain else "N/A"),
       ("📊 PCR",f"{sig['pcr']:.3f}" if has_chain else "N/A"),
       ("💀 Max Pain",f"{sig['max_pain']:,.0f}" if has_chain else "N/A"),
       ("🌡 IV%",f"{sig['atm_iv']:.1f}%" if has_chain else "N/A"),
       ("↕ Skew",f"{sig['skew']:+.1f}%" if has_chain else "N/A"),
       ("♟ Straddle",f"{sig['straddle']:.2f}" if has_chain else "N/A")]
    for col,(lbl,val) in zip(cols,m): col.metric(lbl,val)

    # ── SIGNAL BANNER ──
    conf=sig["conf"]; bc="#00ff88" if conf>=72 else "#ff9500" if conf>=58 else "#5a7a9a"
    def tag(s): return f'<span class="tag-b">{s}</span>' if "BUY" in s else f'<span class="tag-w">{s}</span>'

    if has_chain:
        st.markdown(f"""<div class="sig-box">
  <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;align-items:center">
    <div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
        <span style="color:#5a7a9a;font-size:10px;letter-spacing:2px">SIGNAL</span>{src_badge}
      </div>
      <div style="font-size:24px;font-weight:700;color:{bc};font-family:Space Mono">{sig['rec']}</div>
      <div style="color:#7a9ab5;font-size:12px;margin-top:6px">{' · '.join(sig['reasons'][:3])}</div>
    </div>
    <div style="text-align:center">
      <div style="color:#5a7a9a;font-size:10px;letter-spacing:2px">CONFIDENCE</div>
      <div style="font-size:42px;font-weight:700;color:{bc};font-family:Space Mono;line-height:1.1">{conf}%</div>
      <div style="background:{bc}22;border-radius:20px;height:6px;width:120px;margin:6px auto 0">
        <div style="background:{bc};border-radius:20px;height:6px;width:{conf}%"></div>
      </div>
    </div>
  </div>
  <div style="margin-top:14px;display:flex;flex-wrap:wrap;gap:6px">
    {tag("⚡ Scalp: "+sig['scalp'])} {tag("📅 Intraday: "+sig['intraday'])}
    {tag("🗓 Swing: "+sig['swing'])} {tag("📆 Positional: "+sig['pos'])}
  </div>
</div>""",unsafe_allow_html=True)
        for ab in sig["abnormal"]: st.markdown(f'<div class="alert">{ab}</div>',unsafe_allow_html=True)
        if sig["gamma_blast"]:
            st.markdown('<div class="gamma-blast">⚡ <b style="color:#ff3b5c">GAMMA BLAST</b> — Huge OI at current price. Options EXPLODE on breakout. Enter immediately on confirmation.</div>',unsafe_allow_html=True)
    else:
        st.info(f"📊 Spot only — **{fetch_sym}: ₹{spot:,.2f}**. Chain unavailable. Backtest & volatility analysis still work below.")

    st.markdown("<br>",unsafe_allow_html=True)

    # ── TABS ──────────────────────────────
    tabs=st.tabs(["📊 Chain","🔢 Greeks","📈 OI Analysis","⚡ Live Trade","🔬 Backtest","📋 History","🧠 Analysis"])

    with tabs[0]:
        st.markdown("""<div class="explain"><b>Option Chain:</b> Every available strike price around current market price.
CE = Call (you buy when market goes UP). PE = Put (you buy when market goes DOWN).
ATM = the strike closest to current spot price — most action happens here.
High OI at a strike = big money parked there = strong support/resistance.</div>""",unsafe_allow_html=True)
        if not has_chain:
            st.warning("Chain unavailable. Run `pip install nsepython`, click Clear Cache, retry. Or upload CSV from NSE website in sidebar.")
        else:
            def_st=all_strikes[max(0,atm_pos-8):atm_pos+9]
            sel_st=st.multiselect("Show strikes",all_strikes,default=def_st)
            if not sel_st: sel_st=def_st
            show_c=[c for c in ["strikePrice","CE_LTP","CE_OI","CE_changeOI","CE_IV","CE_volume","PE_LTP","PE_OI","PE_changeOI","PE_IV","PE_volume"] if c in df_exp.columns]
            st.dataframe(df_exp[df_exp["strikePrice"].isin(sel_st)][show_c].round(2),use_container_width=True,height=260)
            st.plotly_chart(plot_chain(df_exp,spot),use_container_width=True)
            st.markdown("#### Is today's straddle cheap or expensive vs last 45 days?")
            st.markdown("""<div class="explain"><b>Straddle</b> = CE + PE combined cost at same strike.
If cheap vs history → options underpriced → great time to buy. If expensive → wait.</div>""",unsafe_allow_html=True)
            fs,fv=plot_straddle(hist_sym,sig["straddle"])
            if fs: st.plotly_chart(fs,use_container_width=True)
            if fv: st.markdown(f'<div style="background:{fv[1]}11;border:1px solid {fv[1]}44;border-radius:8px;padding:12px;color:{fv[1]};font-weight:600;font-size:15px">{fv[0]}</div>',unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("""<div class="explain"><b>Greeks in plain English:</b><br><br>
🟢 <b>Delta</b> = If Nifty moves ₹100, your option moves this much. Delta 0.5 → ₹50 gain per ₹100 move.<br>
🔵 <b>Gamma</b> = How fast Delta changes. High near expiry = option can explode in value on a big move.<br>
🔴 <b>Theta</b> = Money lost per day just by holding. The option buyer's enemy — always ticking.<br>
🟡 <b>Vega</b> = Gain if fear (IV) rises. Buy before big events, sell after volatility collapses.<br><br>
<b>Rule:</b> Only buy when Vega > |Theta| and IV below 25%.</div>""",unsafe_allow_html=True)
        if not has_chain:
            st.warning("Greeks require options chain.")
        else:
            st.plotly_chart(plot_greeks_chart(df_exp,spot),use_container_width=True)
            ai2=(df_exp["strikePrice"]-spot).abs().idxmin(); atm2=df_exp.loc[ai2]
            st.markdown(f"#### ATM Strike ₹{atm2['strikePrice']:,.0f} — Your Greeks Right Now")
            g1,g2=st.columns(2)
            for col,px,label,color in [(g1,"CE","📗 CALL — buy if market goes UP","#00ff88"),(g2,"PE","📕 PUT — buy if market goes DOWN","#ff3b5c")]:
                with col:
                    st.markdown(f"<h5 style='color:{color}'>{label}</h5>",unsafe_allow_html=True)
                    cs=st.columns(3)
                    for i,(name,key,tip) in enumerate([("Delta",f"{px}_delta","Move/₹100"),("Gamma",f"{px}_gamma","Delta speed"),("Theta",f"{px}_theta","Daily decay"),("Vega",f"{px}_vega","Fear gain"),("IV%",f"{px}_IV","Volatility")]):
                        val=float(atm2.get(key,0))
                        cs[i%3].metric(name,f"{val:.2f}%" if "IV" in name else f"{val:.4f}",tip)
            iv=sig["atm_iv"]; ce_th=float(atm2.get("CE_theta",0)); ce_ve=float(atm2.get("CE_vega",0))
            tips=[]
            if iv>40: tips.append(("🔴 IV>40%: Too expensive. Theta destroys value faster than market can move. WAIT.","#ff3b5c"))
            elif iv<15: tips.append(("✅ IV<15%: Cheap! Small correct move = good profit.","#00ff88"))
            if abs(ce_th)>abs(ce_ve): tips.append(("⚠️ Theta>Vega: You lose more per day than you gain from volatility. Prefer ITM strikes.","#ff9500"))
            else: tips.append(("✅ Vega>Theta: Good balance — market moves outpace time decay.","#00ff88"))
            for t,clr in tips:
                st.markdown(f'<div style="background:{clr}11;border-left:3px solid {clr};padding:12px;border-radius:0 6px 6px 0;margin:6px 0">{t}</div>',unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("""<div class="explain">
🔵 <b>High CE OI</b> = resistance — market struggles above this level<br>
🟡 <b>High PE OI</b> = support — market struggles below this level<br>
Rising OI + Rising Price = long buildup = BULLISH | Rising OI + Falling Price = short buildup = BEARISH<br>
<b>Max Pain</b> = strike where most option buyers lose money. Near expiry, market drifts here.
</div>""",unsafe_allow_html=True)
        if not has_chain:
            st.warning("OI analysis requires chain.")
        else:
            st.plotly_chart(plot_oi(df_exp,spot,sig),use_container_width=True)
            m1,m2,m3,m4=st.columns(4)
            m1.metric("🔴 Resistance",f"₹{sig['resistance']:,.0f}","CE writers defend here")
            m2.metric("🟢 Support",f"₹{sig['support']:,.0f}","PE writers defend here")
            m3.metric("🎯 Max Pain",f"₹{sig['max_pain']:,.0f}","Expiry magnet")
            m4.metric("📊 PCR",f"{sig['pcr']:.4f}",">1.2 bullish lean")
            b1,b2=st.columns(2)
            with b1:
                st.markdown("**🔴 Top CE OI — Resistance walls**")
                ct=df_exp.nlargest(5,"CE_OI")[["strikePrice","CE_OI","CE_changeOI","CE_LTP"]].copy()
                ct["Signal"]=ct["CE_changeOI"].apply(lambda x:"🔴 Fresh resistance" if x>=0 else "🟡 Weakening")
                st.dataframe(ct,use_container_width=True)
            with b2:
                st.markdown("**🟢 Top PE OI — Support walls**")
                pt=df_exp.nlargest(5,"PE_OI")[["strikePrice","PE_OI","PE_changeOI","PE_LTP"]].copy()
                pt["Signal"]=pt["PE_changeOI"].apply(lambda x:"🟢 Fresh support" if x>=0 else "🟡 Weakening")
                st.dataframe(pt,use_container_width=True)

    with tabs[3]:
        st.markdown("""<div class="explain">Paper trading — no real money, just practice.
CE = Call (UP trade) | PE = Put (DOWN trade) | SL = Stop Loss (exit if option falls this %) | R:R = Reward÷Risk (never trade below 1:1.5)
</div>""",unsafe_allow_html=True)
        if not has_chain:
            st.warning("Live trading requires chain.")
        else:
            la,lb=st.columns([3,2])
            with la:
                st.markdown("#### 🚨 New Paper Trade")
                t1,t2=st.columns(2)
                side=t1.selectbox("Direction",["CE — Buy Call (UP)","PE — Buy Put (DOWN)"])
                px_="CE" if "CE" in side else "PE"
                t_strike=t1.selectbox("Strike",all_strikes,index=atm_pos)
                lots=t2.number_input("Lots",1,100,1)
                l_sl=t2.slider("SL %",20,50,sl_pct,key="l_sl")
                l_tgt=t2.slider("Target %",30,200,tgt_pct,key="l_tgt")
                row_s=df_exp[df_exp["strikePrice"]==t_strike]
                opt_px=float(row_s[f"{px_}_LTP"].values[0]) if not row_s.empty and f"{px_}_LTP" in row_s.columns else 0
                if opt_px>0:
                    sl_a=round(opt_px*(1-l_sl/100),2); tgt_a=round(opt_px*(1+l_tgt/100),2)
                    risk=(opt_px-sl_a)*lots; rew=(tgt_a-opt_px)*lots; rr=round(rew/risk,2) if risk>0 else 0
                    pm1,pm2,pm3,pm4=st.columns(4)
                    pm1.metric("Entry",f"₹{opt_px:.2f}","Buy here"); pm2.metric("SL",f"₹{sl_a:.2f}",f"-{l_sl}%")
                    pm3.metric("Target",f"₹{tgt_a:.2f}",f"+{l_tgt}%"); pm4.metric("R:R",f"1:{rr}","")
                    if rr<1.5: st.warning("⚠️ R:R below 1:1.5 — poor trade.")
                else:
                    sl_a=tgt_a=0; st.warning("Price is 0 — market closed or too far OTM.")
                st.markdown(f'<div style="background:#0a1929;border:1px solid #1e3050;border-radius:8px;padding:12px"><b style="color:#00e5ff">{sig["rec"]}</b> | {sig["conf"]}% confidence</div>',unsafe_allow_html=True)
                if st.button("📈 ENTER PAPER TRADE",type="primary",use_container_width=True):
                    if opt_px>0:
                        t={"id":len(st.session_state.trades)+1,"sym":fetch_sym,"expiry":sel_expiry,
                           "strike":t_strike,"side":px_,"entry":opt_px,"sl":sl_a,"target":tgt_a,
                           "lots":lots,"conf":sig["conf"],"rec":sig["rec"],
                           "time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "status":"OPEN","exit":None,"pnl":None,"exit_time":None}
                        st.session_state.active.append(t); st.session_state.trades.append(t)
                        st.success(f"✅ Entered: {fetch_sym} {t_strike} {px_} @ ₹{opt_px:.2f}")
                    else: st.error("Cannot enter — price is 0.")
            with lb:
                st.markdown("#### 📊 Open Positions")
                if not [t for t in st.session_state.active if t["status"]=="OPEN"]: st.info("No open trades.")
                for i,t in enumerate(st.session_state.active):
                    if t["status"]!="OPEN": continue
                    r=df_exp[df_exp["strikePrice"]==t["strike"]]; ltp_col=f"{t['side']}_LTP"
                    curr=float(r[ltp_col].values[0]) if not r.empty and ltp_col in r.columns else t["entry"]
                    pnl=round((curr-t["entry"])*t["lots"],2); pp=round((curr-t["entry"])/t["entry"]*100,2) if t["entry"]>0 else 0
                    clr="#00ff88" if pnl>=0 else "#ff3b5c"; cls="tc-w" if pnl>0 else "tc-l" if pnl<0 else "tc-o"
                    warn=("⚠️ <b style='color:#ff3b5c'>STOP LOSS HIT — EXIT NOW</b><br>" if curr<=t["sl"] and t["sl"]>0 else
                          "🎯 <b style='color:#00ff88'>TARGET HIT — BOOK PROFIT</b><br>" if curr>=t["target"] and t["target"]>0 else "")
                    st.markdown(f"""<div class="tc {cls}">{warn}<b>{t['sym']} {t['strike']} {t['side']}</b><br>
Entry ₹{t['entry']:.2f} → Now ₹{curr:.2f} | SL ₹{t['sl']:.2f} | Tgt ₹{t['target']:.2f}<br>
<b style='color:{clr}'>P&L: ₹{pnl:+.2f} ({pp:+.1f}%)</b></div>""",unsafe_allow_html=True)
                    if st.button(f"Exit #{t['id']}",key=f"ex_{i}_{t['id']}"):
                        for j,x in enumerate(st.session_state.active):
                            if x["id"]==t["id"]:
                                st.session_state.active[j].update({"status":"CLOSED","exit":curr,"pnl":pnl,"exit_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                                for h in st.session_state.trades:
                                    if h["id"]==t["id"]: h.update(st.session_state.active[j])
                                break
                        st.rerun()

    with tabs[4]:
        st.markdown("""<div class="explain"><b>Backtest — plain English:</b><br>
Step 1: Look at each past day's close price · Step 2: Estimate IV from last 7 days' moves ·
Step 3: Score using same PCR/IV/trend rules as live signal · Step 4: If score ≥72%, "buy" an ATM option ·
Step 5: Check NEXT day's price — did we win or lose? · Step 6: Subtract 1 day of theta per trade · Step 7: Compound into equity curve<br><br>
<b>Good = Win Rate 60%+ | R:R above 1.5 | Max Drawdown below 20%</b></div>""",unsafe_allow_html=True)
        ba,bb,bc_=st.columns(3)
        bt_look=ba.slider("Lookback days",20,120,60)
        bt_cap=bb.number_input("Starting capital ₹",50_000,5_000_000,int(capital),10_000,key="bt_cap")
        bt_pos=bc_.slider("% per trade",2,20,pos_pct,key="bt_pos")
        st.markdown(f'<div class="explain">With ₹{int(bt_cap):,} and {bt_pos}%: each trade risks <b>₹{int(bt_cap*bt_pos/100):,}</b>. Remaining ₹{int(bt_cap*(1-bt_pos/100)):,} stays safe even if first trade is a total loss.</div>',unsafe_allow_html=True)
        if st.button("🔬 RUN BACKTEST",type="primary",use_container_width=True):
            with st.spinner("Running…"):
                tdf,stats=run_backtest(hist_sym,bt_look,float(bt_cap),float(bt_pos))
                st.session_state.bt_result=(tdf,stats)
        if st.session_state.bt_result:
            tdf,stats=st.session_state.bt_result
            if tdf is None: st.error(f"Backtest failed: {stats}")
            else:
                k1,k2,k3,k4,k5,k6,k7=st.columns(7)
                k1.metric("Trades",stats["total"]); k2.metric("Win Rate",f"{stats['wr']}%")
                k3.metric("Total P&L",f"₹{stats['total_pnl']:+,.0f}"); k4.metric("Avg Win",f"₹{stats['aw']:,.0f}")
                k5.metric("Avg Loss",f"₹{stats['al']:,.0f}"); k6.metric("R:R",stats['rr']); k7.metric("Max DD",f"{stats['mdd']}%")
                rc="#00ff88" if stats["ret%"]>=0 else "#ff3b5c"
                st.markdown(f'<div style="text-align:center;font-size:26px;color:{rc};font-family:Space Mono;margin:12px 0;padding:14px;background:#0a1929;border-radius:10px">Final: ₹{stats["final"]:,.0f} | Return: {stats["ret%"]:+.2f}%</div>',unsafe_allow_html=True)
                wr=stats["wr"]; rr=stats["rr"]; mdd=stats["mdd"]
                if wr>=60 and rr>=1.5 and mdd<20: v=("✅ Strong strategy — win rate good, wins bigger than losses, drawdown controlled.","#00ff88")
                elif wr>=50 and rr>=1.2: v=("🟡 Decent — profitable but needs tight discipline.","#ff9500")
                else: v=("🔴 Weak — don't trade real money yet. Optimize signals first.","#ff3b5c")
                st.markdown(f'<div style="background:{v[1]}11;border-left:3px solid {v[1]};padding:14px;border-radius:0 8px 8px 0;margin:8px 0">{v[0]}</div>',unsafe_allow_html=True)
                fig_eq=go.Figure(go.Scatter(y=stats["curve"],mode="lines",line=dict(color="#00e5ff",width=2.5),fill="tozeroy",fillcolor="rgba(0,229,255,0.05)"))
                fig_eq.add_hline(y=bt_cap,line_dash="dash",line_color="#ff9500",annotation_text=f"Start ₹{int(bt_cap):,}")
                fig_eq.update_layout(template=DARK,height=320,title="Equity Curve",yaxis_title="Capital ₹")
                st.plotly_chart(fig_eq,use_container_width=True)
                w=tdf[tdf["P&L(₹)"]>0]["P&L(₹)"]; l=tdf[tdf["P&L(₹)"]<=0]["P&L(₹)"]
                fig_d=go.Figure()
                fig_d.add_trace(go.Histogram(x=w,name="Wins",marker_color="rgba(0,255,136,0.47)",nbinsx=20))
                fig_d.add_trace(go.Histogram(x=l,name="Losses",marker_color="rgba(255,59,92,0.47)",nbinsx=20))
                fig_d.update_layout(template=DARK,height=240,title="Win vs Loss Distribution",barmode="overlay")
                st.plotly_chart(fig_d,use_container_width=True)
                st.dataframe(tdf,use_container_width=True,height=280)

    with tabs[5]:
        if not st.session_state.trades: st.info("No paper trades yet.")
        else:
            all_t=pd.DataFrame(st.session_state.trades)
            closed=all_t[all_t["status"]=="CLOSED"].copy()
            if not closed.empty:
                closed["pnl"]=pd.to_numeric(closed["pnl"],errors="coerce").fillna(0)
                tot=closed["pnl"].sum(); wr=(closed["pnl"]>0).mean()*100
                h1,h2,h3,h4=st.columns(4)
                h1.metric("Total",len(all_t)); h2.metric("Closed",len(closed))
                h3.metric("Win Rate",f"{wr:.1f}%"); h4.metric("Net P&L",f"₹{tot:+,.2f}")
                fig_p=go.Figure(go.Bar(y=closed["pnl"].values,marker_color=["#00ff88" if p>0 else "#ff3b5c" for p in closed["pnl"]]))
                fig_p.update_layout(template=DARK,height=240,title="Per-Trade P&L")
                st.plotly_chart(fig_p,use_container_width=True)
            dc=[c for c in ["id","time","sym","strike","side","entry","exit","lots","pnl","status","rec","conf"] if c in all_t.columns]
            st.dataframe(all_t[dc],use_container_width=True)
            st.download_button("📥 Download CSV",all_t.to_csv(index=False),"trades.csv","text/csv",use_container_width=True)
            if st.button("🗑️ Clear All Trades",use_container_width=True):
                st.session_state.trades=[]; st.session_state.active=[]; st.rerun()

    with tabs[6]:
        iv=sig["atm_iv"]; pcr=sig["pcr"]; mp=sig["max_pain"]
        res=sig["resistance"]; sup=sig["support"]; mp_pct=(mp-spot)/spot*100 if spot>0 else 0
        st.markdown(f"""<div class="card"><h4 style='color:#00e5ff;margin-top:0'>📖 Full Market Reading — {fetch_sym} @ ₹{spot:,.2f} {src_badge}</h4>
<p style='color:#5a7a9a'>Every indicator explained in plain English:</p></div>""",unsafe_allow_html=True)
        if has_chain:
            pcr_text=("PCR HIGH — more puts than calls. Sounds scary but it's CONTRARIAN BULLISH. Big money is buying insurance (puts) to protect longs — they're not betting on a crash. Market often goes UP from here." if pcr>1.2
                     else "PCR LOW — more calls than puts. Sounds bullish but CONTRARIAN BEARISH. When everyone buys calls optimistically, market is near a top. It may go DOWN." if pcr<0.8
                     else "PCR BALANCED — no strong signal. Wait for clearer setup.")
            st.markdown(f'<div class="explain"><b>📊 PCR = {pcr:.4f}</b><br>{pcr_text}</div>',unsafe_allow_html=True)
            iv_text=("Options CHEAP — like petrol at a discount, fill up. Low IV = pay less upfront + gain extra if IV rises later." if iv<18
                    else "Options VERY EXPENSIVE — market is scared, everyone rushing to buy protection. Worst time to buy. Even right direction = loss as IV crashes after the move." if iv>40
                    else f"IV at {iv:.0f}% is moderate. Not cheap, not expensive. Only buy on strong signals.")
            st.markdown(f'<div class="explain"><b>🌡 IV = {iv:.1f}%</b><br>{iv_text}</div>',unsafe_allow_html=True)
            mp_text=(f"Max pain {mp_pct:+.1f}% ABOVE at ₹{mp:,.0f}. Market gets pulled up near expiry. Favors CALL buyers." if mp_pct>2
                    else f"Max pain {abs(mp_pct):.1f}% BELOW at ₹{mp:,.0f}. Market pulled down near expiry. Favors PUT buyers." if mp_pct<-2
                    else f"Max pain ₹{mp:,.0f} close to current price. No gravity pull. Matters most in last 2–3 days before expiry.")
            st.markdown(f'<div class="explain"><b>🎯 Max Pain = ₹{mp:,.0f}</b><br>{mp_text}</div>',unsafe_allow_html=True)
            st.markdown(f"""<div class="explain"><b>🔴 Resistance ₹{res:,.0f} | 🟢 Support ₹{sup:,.0f}</b><br>
<b>₹{res:,.0f}</b> = biggest CE OI = CEILING. Call writers defend this aggressively. If price breaks above and stays, those writers panic-buy back → sharp rally.<br><br>
<b>₹{sup:,.0f}</b> = biggest PE OI = FLOOR. Put writers defend this. If price breaks below, they panic-buy puts → sharp fall.<br><br>
<b>Trade the breakout, not the range.</b> Buy options only when price clearly breaks one of these walls with volume.
</div>""",unsafe_allow_html=True)
            st.markdown("#### 🗺 Three Scenarios")
            s1,s2,s3=st.columns(3)
            for col,title,trigger,action,why,color in [
                (s1,"🟢 BULLISH",f"Break & hold above ₹{res:,.0f}",f"Buy {int(sig['atm'])} CE\nTarget: +60%\nSL: -30%","Resistance breaks → call writers panic → premium explodes.","#00ff88"),
                (s2,"⚪ SIDEWAYS",f"Range ₹{sup:,.0f}–₹{res:,.0f}","DO NOT BUY OPTIONS.\nTheta kills you.\nCash is best.","Every day you hold options in a sideways market = money lost to theta.","#5a7a9a"),
                (s3,"🔴 BEARISH",f"Break & hold below ₹{sup:,.0f}",f"Buy {int(sig['atm'])} PE\nTarget: +60%\nSL: -30%","Support breaks → put writers panic → put premium explodes.","#ff3b5c"),
            ]:
                with col:
                    st.markdown(f"""<div style='background:#0a1929;border-top:4px solid {color};border:1px solid #1e3050;border-radius:10px;padding:16px;height:100%'>
<h5 style='color:{color};margin-top:0'>{title}</h5><b style='color:#5a7a9a'>Trigger:</b><br>{trigger}<br><br>
<b style='color:#5a7a9a'>Action:</b><br><pre style='color:#c8d6e5;font-size:12px;background:transparent;margin:4px 0'>{action}</pre>
<b style='color:#5a7a9a'>Why:</b><br><span style='font-size:12px;color:#8899aa'>{why}</span></div>""",unsafe_allow_html=True)
        st.markdown("""<div class="card" style='margin-top:16px'><h4 style='color:#ff9500;margin-top:0'>📜 Golden Rules</h4>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:14px;line-height:1.8'>
<div>1. <b>Never risk more than 5% per trade.</b><br>2. <b>Only buy when IV below 30%.</b><br>3. <b>Always set SL before entering.</b><br>4. <b>Exit 5+ days before expiry.</b></div>
<div>5. <b>Book 50% at first target, trail rest.</b><br>6. <b>Never average a losing trade.</b><br>7. <b>Avoid buying on event days</b> (RBI, earnings).<br>8. <b>No trade is also a trade.</b></div>
</div></div>""",unsafe_allow_html=True)
        st.caption(f"Updated: {datetime.now().strftime('%d-%b-%Y %H:%M:%S')} | Educational only")

    if auto_ref:
        time.sleep(90); st.cache_data.clear(); _T["last"]=0.0; st.rerun()

if __name__=="__main__":
    main()
