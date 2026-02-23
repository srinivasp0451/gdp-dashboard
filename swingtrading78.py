"""
Professional Options Trading Dashboard
Live CE/PE chain, Greeks, OI analysis, signals, backtest, trade log.
yfinance: NO custom session — let yfinance handle its own curl_cffi internals.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time, random, warnings
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ProOptions ⚡", page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
:root{--bg:#060a10;--s1:#0d1520;--s2:#111d2e;--acc:#00e5ff;--grn:#00ff88;--red:#ff3b5c;--ora:#ff9500;--txt:#c8d6e5;--mut:#5a7a9a;--brd:#1e3050}
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:var(--bg)!important;color:var(--txt)!important}
.stApp{background:var(--bg)}
.stMetric{background:var(--s1)!important;border:1px solid var(--brd)!important;border-radius:10px!important;padding:12px!important}
.stMetric label{color:var(--mut)!important;font-size:10px!important;letter-spacing:1px;text-transform:uppercase}
.stMetric [data-testid="stMetricValue"]{color:var(--acc)!important;font-family:'Space Mono',monospace;font-size:18px!important}
div[data-testid="stTabs"] button{color:var(--mut)!important;font-family:'Space Mono',monospace;font-size:11px}
div[data-testid="stTabs"] button[aria-selected="true"]{color:var(--acc)!important;border-bottom:2px solid var(--acc)!important}
h1,h2,h3{font-family:'Space Mono',monospace!important;color:var(--acc)!important}
.stButton>button{background:transparent!important;border:1px solid var(--acc)!important;color:var(--acc)!important;font-family:'Space Mono',monospace!important;border-radius:8px!important;font-size:12px!important}
.stButton>button:hover{background:var(--acc)!important;color:#000!important}
.signal-box{background:var(--s1);border:1px solid var(--brd);border-radius:12px;padding:18px 22px;margin:10px 0;position:relative;overflow:hidden}
.signal-box::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--acc),var(--grn))}
.tag-buy{background:#002a15;color:#00ff88;border:1px solid #00aa44;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600}
.tag-sell{background:#2a0010;color:#ff3b5c;border:1px solid #aa0022;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600}
.tag-wait{background:#2a1a00;color:#ff9500;border:1px solid #aa5500;padding:3px 10px;border-radius:20px;font-size:12px}
.sum{background:var(--s2);border:1px solid var(--brd);border-radius:8px;padding:12px 16px;font-size:13px;color:var(--mut);margin-bottom:14px}
.alert{background:#1a0a00;border-left:3px solid var(--ora);border-radius:6px;padding:10px 14px;margin:4px 0;font-size:13px}
.gamma{background:#1a000a;border:1px solid #ff3b5c88;border-radius:10px;padding:14px;animation:pulse 2s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(255,59,92,.4)}70%{box-shadow:0 0 0 8px rgba(255,59,92,0)}100%{box-shadow:0 0 0 0 rgba(255,59,92,0)}}
.trade-card{background:var(--s2);border:1px solid var(--brd);border-radius:10px;padding:14px;margin:6px 0}
.tc-open{border-left:3px solid var(--acc)}
.tc-win{border-left:3px solid var(--grn)}
.tc-loss{border-left:3px solid var(--red)}
hr{border-color:var(--brd)!important}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS & SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
RFR = 0.065
DARK = "plotly_dark"
THROTTLE_KEY = "_yf_last_call"

NSE_IDX = {"NIFTY":"^NSEI","BANKNIFTY":"^NSEBANK","SENSEX":"^BSESN",
            "FINNIFTY":"NIFTY_FIN_SERVICE.NS","MIDCPNIFTY":"^NSMIDCP"}
GLOBAL_MAP = {"BTC/USD":"BTC-USD","GOLD":"GC=F","SILVER":"SI=F",
              "USD/INR":"USDINR=X","CRUDE OIL":"CL=F","NATURAL GAS":"NG=F"}

for k,v in {"trades":[],"active":[],"bt_result":None,"last_yf":0.0}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────────────────────────────────────
# RATE LIMITER  — simple, reliable, no sessions
# ──────────────────────────────────────────────────────────────────────────────
def _wait(min_gap: float = 1.5):
    """Block until min_gap seconds have passed since the last yfinance call."""
    now = time.time()
    elapsed = now - st.session_state.last_yf
    if elapsed < min_gap:
        time.sleep(min_gap - elapsed + random.uniform(0.1, 0.3))
    st.session_state.last_yf = time.time()

# ──────────────────────────────────────────────────────────────────────────────
# YFINANCE FETCHERS  — plain yf.Ticker(), no session arg
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=120, show_spinner=False)
def get_expiries(sym: str):
    _wait(1.5)
    try:
        return list(yf.Ticker(sym).options or [])
    except Exception as e:
        return []

@st.cache_data(ttl=120, show_spinner=False)
def get_chain(sym: str, expiry: str):
    _wait(1.5)
    try:
        chain = yf.Ticker(sym).option_chain(expiry)
        return chain.calls.copy(), chain.puts.copy()
    except Exception as e:
        st.error(f"Chain fetch failed: {e}")
        return None, None

@st.cache_data(ttl=60, show_spinner=False)
def get_spot(sym: str) -> float:
    _wait(1.5)
    try:
        tk = yf.Ticker(sym)
        fi = tk.fast_info
        price = (getattr(fi, "last_price", None) or
                 getattr(fi, "regular_market_price", None) or
                 getattr(fi, "previousClose", None))
        if price and float(price) > 0:
            return float(price)
        # fallback: 1-day history
        _wait(1.5)
        h = tk.history(period="2d", interval="1d")
        if not h.empty:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0

@st.cache_data(ttl=300, show_spinner=False)
def get_history(sym: str, period: str = "60d") -> pd.DataFrame:
    _wait(1.5)
    try:
        df = yf.Ticker(sym).history(period=period, interval="1d", auto_adjust=True)
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ──────────────────────────────────────────────────────────────────────────────
# NSE OPTION CHAIN  — direct API with cookie handshake
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=90, show_spinner=False)
def get_nse_chain(symbol: str):
    sess = requests.Session()
    sess.headers.update({
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36",
        "Accept":"*/*","Referer":"https://www.nseindia.com/",
        "Accept-Language":"en-US,en;q=0.9","Connection":"keep-alive",
    })
    try:
        sess.get("https://www.nseindia.com/", timeout=8)
        time.sleep(0.8)
        sess.get("https://www.nseindia.com/market-data/equity-derivatives-watch", timeout=8)
        time.sleep(0.5)
        if symbol in list(NSE_IDX.keys()):
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
        r = sess.get(url, timeout=12)
        if r.status_code != 200:
            return None, 0, []
        data = r.json().get("records", {})
        spot = float(data.get("underlyingValue", 0))
        expiries = data.get("expiryDates", [])
        rows = []
        for rec in data.get("data", []):
            ce, pe = rec.get("CE", {}), rec.get("PE", {})
            rows.append({"strikePrice": rec.get("strikePrice",0),"expiryDate": rec.get("expiryDate",""),
                "CE_LTP":ce.get("lastPrice",0),"CE_OI":ce.get("openInterest",0),
                "CE_changeOI":ce.get("changeinOpenInterest",0),"CE_volume":ce.get("totalTradedVolume",0),
                "CE_IV":ce.get("impliedVolatility",0),"CE_bid":ce.get("bidprice",0),"CE_ask":ce.get("askPrice",0),
                "PE_LTP":pe.get("lastPrice",0),"PE_OI":pe.get("openInterest",0),
                "PE_changeOI":pe.get("changeinOpenInterest",0),"PE_volume":pe.get("totalTradedVolume",0),
                "PE_IV":pe.get("impliedVolatility",0),"PE_bid":pe.get("bidprice",0),"PE_ask":pe.get("askPrice",0),
            })
        return pd.DataFrame(rows), spot, expiries
    except Exception as e:
        return None, 0, []

# ──────────────────────────────────────────────────────────────────────────────
# BUILD CHAIN DATAFRAME from yfinance calls + puts
# ──────────────────────────────────────────────────────────────────────────────
def build_chain_df(calls: pd.DataFrame, puts: pd.DataFrame, expiry: str, spot: float) -> pd.DataFrame:
    if calls is None or calls.empty:
        return pd.DataFrame()
    df = pd.DataFrame()
    df["strikePrice"] = calls["strike"].values if "strike" in calls.columns else []
    df["expiryDate"] = expiry

    def gc(col, src, factor=1):
        return calls[src].values * factor if src in calls.columns else np.zeros(len(calls))

    df["CE_LTP"]      = gc("CE_LTP", "lastPrice")
    df["CE_OI"]       = gc("CE_OI", "openInterest")
    df["CE_volume"]   = gc("CE_volume", "volume")
    df["CE_IV"]       = gc("CE_IV", "impliedVolatility", 100)
    df["CE_bid"]      = gc("CE_bid", "bid")
    df["CE_ask"]      = gc("CE_ask", "ask")
    df["CE_changeOI"] = 0

    if puts is not None and not puts.empty and "strike" in puts.columns:
        pi = puts.set_index("strike")
        for col, src, factor in [("PE_LTP","lastPrice",1),("PE_OI","openInterest",1),
                                   ("PE_volume","volume",1),("PE_bid","bid",1),("PE_ask","ask",1)]:
            df[col] = df["strikePrice"].map(pi[src] if src in pi.columns else pd.Series(dtype=float)).fillna(0).values
        df["PE_IV"] = df["strikePrice"].map(
            pi["impliedVolatility"]*100 if "impliedVolatility" in pi.columns else pd.Series(dtype=float)
        ).fillna(0).values
    else:
        for c in ["PE_LTP","PE_OI","PE_volume","PE_bid","PE_ask","PE_IV"]:
            df[c] = 0
    df["PE_changeOI"] = 0
    df = df.fillna(0)
    df = df[(df["CE_LTP"]+df["PE_LTP"]) > 0].reset_index(drop=True)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES + GREEKS
# ──────────────────────────────────────────────────────────────────────────────
def bs(S,K,T,r,sig,kind="call"):
    if T<1e-6 or sig<1e-6 or S<=0 or K<=0:
        return dict(delta=0,gamma=0,theta=0,vega=0,rho=0,price=0)
    d1=(np.log(S/K)+(r+.5*sig**2)*T)/(sig*np.sqrt(T)); d2=d1-sig*np.sqrt(T)
    if kind=="call":
        price=S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
        delta=norm.cdf(d1); rho=K*T*np.exp(-r*T)*norm.cdf(d2)/100
        theta=(-(S*norm.pdf(d1)*sig)/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2))/365
    else:
        price=K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
        delta=norm.cdf(d1)-1; rho=-K*T*np.exp(-r*T)*norm.cdf(-d2)/100
        theta=(-(S*norm.pdf(d1)*sig)/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2))/365
    gamma=norm.pdf(d1)/(S*sig*np.sqrt(T)); vega=S*norm.pdf(d1)*np.sqrt(T)/100
    return dict(delta=round(delta,4),gamma=round(gamma,8),theta=round(theta,4),
                vega=round(vega,4),rho=round(rho,4),price=round(max(price,0),4))

def calc_iv(mkt,S,K,T,r,kind="call"):
    if T<=0 or mkt<=0 or S<=0: return 0.20
    try:
        return max(brentq(lambda s: bs(S,K,T,r,s,kind)["price"]-mkt, 1e-4, 20.0, xtol=1e-5, maxiter=200), 0.001)
    except: return 0.20

def add_greeks(df: pd.DataFrame, spot: float, expiry_str: str) -> pd.DataFrame:
    if df is None or df.empty: return df
    try:
        fmt = "%d-%b-%Y" if "-" in str(expiry_str) and not str(expiry_str)[4:5].isdigit() else "%Y-%m-%d"
        T = max((datetime.strptime(str(expiry_str), fmt)-datetime.now()).days/365.0, 1/365)
    except: T = 7/365
    df = df.copy()
    for i, row in df.iterrows():
        K = float(row["strikePrice"])
        for kind, px in [("call","CE"), ("put","PE")]:
            ltp = float(row.get(f"{px}_LTP",0))
            iv_pct = float(row.get(f"{px}_IV",0))
            sig = iv_pct/100 if iv_pct > 0.5 else calc_iv(ltp, spot, K, T, RFR, kind)
            sig = max(sig, 0.01)
            g = bs(spot, K, T, RFR, sig, kind)
            df.at[i,f"{px}_delta"]=g["delta"]; df.at[i,f"{px}_gamma"]=g["gamma"]
            df.at[i,f"{px}_theta"]=g["theta"]; df.at[i,f"{px}_vega"]=g["vega"]
            if iv_pct==0 and ltp>0: df.at[i,f"{px}_IV"]=round(sig*100,2)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL ENGINE  (identical code used in live trading AND backtesting)
# ──────────────────────────────────────────────────────────────────────────────
def signals(df: pd.DataFrame, spot: float) -> dict:
    out = dict(pcr=0.0,max_pain=spot,atm=spot,straddle=0.0,
               resistance=spot,support=spot,atm_iv=20.0,ce_iv=20.0,pe_iv=20.0,
               skew=0.0,gamma_blast=False,abnormal=[],
               rec="⚪ NO TRADE",direction="NONE",conf=0,
               scalp="WAIT",intraday="WAIT",swing="WAIT",pos="WAIT",
               reasons=[])
    if df is None or df.empty or spot==0: return out
    df = df.copy(); df["strikePrice"]=df["strikePrice"].astype(float)
    ai = (df["strikePrice"]-spot).abs().idxmin(); atm=df.loc[ai]
    out["atm"]=float(atm["strikePrice"])
    out["straddle"]=round(float(atm.get("CE_LTP",0))+float(atm.get("PE_LTP",0)),2)
    ce_oi=float(df["CE_OI"].sum()); pe_oi=float(df["PE_OI"].sum())
    out["pcr"]=round(pe_oi/ce_oi,4) if ce_oi>0 else 0
    # max pain
    strikes=df["strikePrice"].values; c_oi=df["CE_OI"].values; p_oi=df["PE_OI"].values
    pain=[sum(max(0,k-s)*o for k,o in zip(strikes,c_oi))+sum(max(0,s-k)*o for k,o in zip(strikes,p_oi)) for s in strikes]
    out["max_pain"]=float(strikes[int(np.argmin(pain))]) if pain else spot
    out["resistance"]=float(df.loc[df["CE_OI"].idxmax(),"strikePrice"])
    out["support"]=float(df.loc[df["PE_OI"].idxmax(),"strikePrice"])
    ce_iv=float(atm.get("CE_IV",0)); pe_iv=float(atm.get("PE_IV",0))
    out["ce_iv"]=ce_iv; out["pe_iv"]=pe_iv
    out["atm_iv"]=(ce_iv+pe_iv)/2 if (ce_iv+pe_iv)>0 else 20.0
    out["skew"]=round(pe_iv-ce_iv,2)
    near=(df["CE_OI"].where(df["strikePrice"].between(spot*.99,spot*1.01),0).sum()+
          df["PE_OI"].where(df["strikePrice"].between(spot*.99,spot*1.01),0).sum())
    out["gamma_blast"]=((ce_oi+pe_oi)>0) and near/(ce_oi+pe_oi+1)>0.35

    iv=out["atm_iv"]
    if iv>55:   out["abnormal"].append(f"🔴 IV EXTREME {iv:.1f}%: premium crushing — avoid buying now")
    elif iv>35: out["abnormal"].append(f"⚠️ High IV {iv:.1f}%: expensive for buyers, be very selective")
    elif 0<iv<14: out["abnormal"].append(f"✅ Low IV {iv:.1f}%: options cheap — prime buying environment!")
    sk=out["skew"]
    if sk>8:  out["abnormal"].append(f"📊 Put skew +{sk:.1f}%: heavy hedging/fear, bearish bias")
    elif sk<-8: out["abnormal"].append(f"📊 Call skew {sk:.1f}%: aggressive call buying, bullish momentum")
    mp_pct=(out["max_pain"]-spot)/spot*100 if spot>0 else 0
    if abs(mp_pct)>2: out["abnormal"].append(f"🎯 Max pain {mp_pct:+.1f}% from spot — gravitational pull near expiry")
    if out["gamma_blast"]: out["abnormal"].append("⚡ GAMMA BLAST: OI concentrated at ATM — explosive move likely on breakout")

    # === SCORING (identical in backtest) ===
    score=50; direction="NONE"; reasons=[]
    pcr=out["pcr"]
    if pcr>1.5:   score+=12; direction="CALL"; reasons.append(f"PCR {pcr:.2f} extreme put→contrarian BULLISH")
    elif pcr>1.1: score+=6;  direction="CALL"; reasons.append(f"PCR {pcr:.2f} put-heavy→mild bullish")
    elif pcr<0.5: score+=12; direction="PUT";  reasons.append(f"PCR {pcr:.2f} extreme call→contrarian BEARISH")
    elif pcr<0.9: score+=6;  direction="PUT";  reasons.append(f"PCR {pcr:.2f} call-heavy→mild bearish")
    else: reasons.append(f"PCR {pcr:.2f} balanced")

    if iv>50:   score-=25; reasons.append("IV >50% avoid buying")
    elif iv>35: score-=12; reasons.append(f"IV {iv:.1f}% high theta risk")
    elif 0<iv<18: score+=15; reasons.append(f"IV {iv:.1f}% cheap for buyers!")
    elif iv<=30:  score+=8;  reasons.append(f"IV {iv:.1f}% acceptable")

    mp=out["max_pain"]
    if spot<mp and direction=="CALL": score+=10; reasons.append(f"spot below max pain {mp:,.0f}→upward pull")
    elif spot>mp and direction=="PUT": score+=10; reasons.append(f"spot above max pain {mp:,.0f}→downward pull")
    elif spot<mp and direction=="PUT": score-=8; reasons.append("spot below max pain→against put trade")
    elif spot>mp and direction=="CALL": score-=8; reasons.append("spot above max pain→against call trade")

    res=out["resistance"]; sup=out["support"]
    if direction=="CALL" and sup<spot<res: score+=8; reasons.append(f"in call zone {sup:,.0f}→{res:,.0f}")
    if direction=="PUT"  and sup<spot<res: score+=8; reasons.append(f"near resistance wall {res:,.0f}")

    ce_chg=float(df["CE_changeOI"].sum()); pe_chg=float(df["PE_changeOI"].sum())
    if pe_chg>ce_chg*1.3 and direction=="CALL": score+=7; reasons.append("aggressive put writing→support building")
    if ce_chg>pe_chg*1.3 and direction=="PUT":  score+=7; reasons.append("aggressive call writing→resistance building")

    score=min(max(int(score),0),100)
    out["conf"]=score; out["direction"]=direction; out["reasons"]=reasons

    atm_str=int(out["atm"]); otm1=int(atm_str*1.005 if direction=="CALL" else atm_str*0.995)
    if score>=72 and direction=="CALL":
        out["rec"]="🟢 BUY CALL (CE)"
        out["scalp"]=f"Buy {atm_str} CE (scalp 20-40% target, SL 30%)"
        out["intraday"]=f"Buy {atm_str} CE (target 60%, SL 30%)"
        out["swing"]=f"Buy {otm1} CE next expiry (target 80-100%)"
        out["pos"]=f"Buy {otm1} CE far expiry (positional)"
    elif score>=72 and direction=="PUT":
        out["rec"]="🔴 BUY PUT (PE)"
        out["scalp"]=f"Buy {atm_str} PE (scalp 20-40% target, SL 30%)"
        out["intraday"]=f"Buy {atm_str} PE (target 60%, SL 30%)"
        out["swing"]=f"Buy {otm1} PE next expiry (target 80-100%)"
        out["pos"]=f"Buy {otm1} PE far expiry (positional)"
    elif score>=58:
        out["rec"]="🟡 WATCH — Weak Signal"
        out["scalp"]=out["intraday"]=out["swing"]=out["pos"]="WATCH — wait for confirmation"
    else:
        out["rec"]="⚪ NO TRADE — Low Edge"
        out["scalp"]=out["intraday"]=out["swing"]=out["pos"]="NO TRADE"
    return out

# ──────────────────────────────────────────────────────────────────────────────
# BACKTEST (uses IDENTICAL scoring as signals())
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def backtest(sym: str, lookback: int, capital: float, pos_pct: float):
    raw = get_history(sym, f"{lookback}d")
    if raw.empty or len(raw)<15: return None, "Not enough price history"
    close=raw["Close"].squeeze().astype(float)
    vol7=close.pct_change().rolling(7).std()*np.sqrt(252)*100
    sma10=close.rolling(10).mean(); sma20=close.rolling(20).mean()
    vol_ma=close.pct_change().rolling(3).std()*np.sqrt(252)*100
    vol_ma_long=close.pct_change().rolling(10).std()*np.sqrt(252)*100

    trades=[]; equity=float(capital); equity_curve=[equity]
    for i in range(20,len(close)-1):
        iv=float(vol7.iloc[i]) if not np.isnan(vol7.iloc[i]) else 20.0
        spot=float(close.iloc[i])
        mp=(float(sma10.iloc[i])+float(sma20.iloc[i]))/2
        # proxy PCR from vol ratio
        vm=float(vol_ma.iloc[i]) if not np.isnan(vol_ma.iloc[i]) else 20.0
        vml=float(vol_ma_long.iloc[i]) if not np.isnan(vol_ma_long.iloc[i]) else 20.0
        pcr=1.0+(vml-vm)/(vml+1)

        # === IDENTICAL scoring logic ===
        score=50; direction="NONE"
        if pcr>1.5:   score+=12; direction="CALL"
        elif pcr>1.1: score+=6;  direction="CALL"
        elif pcr<0.5: score+=12; direction="PUT"
        elif pcr<0.9: score+=6;  direction="PUT"

        if iv>50:   score-=25
        elif iv>35: score-=12
        elif iv<18: score+=15
        elif iv<=30: score+=8

        if spot<mp and direction=="CALL": score+=10
        elif spot>mp and direction=="PUT": score+=10
        elif spot<mp and direction=="PUT": score-=8
        elif spot>mp and direction=="CALL": score-=8

        if float(sma10.iloc[i])>float(sma20.iloc[i]) and direction=="CALL": score+=7
        elif float(sma10.iloc[i])<float(sma20.iloc[i]) and direction=="PUT": score+=7

        score=min(max(int(score),0),100)
        equity_curve.append(equity)
        if score<72 or direction=="NONE": continue

        T=7/365; sig=max(iv/100,0.01)
        opt_px=spot*sig*np.sqrt(T)*0.4
        if opt_px<=0: continue
        next_spot=float(close.iloc[i+1])
        ret=(next_spot-spot)/spot
        raw_gain=(ret*0.5*spot) if direction=="CALL" else (-ret*0.5*spot)
        theta_drag=-(opt_px/7)
        gross=raw_gain+theta_drag
        pnl_pct=max(min(gross/opt_px, 2.5), -0.5)
        position=equity*(pos_pct/100)
        pnl=position*pnl_pct
        equity=max(equity+pnl,0)
        trades.append({"Date":raw.index[i].strftime("%Y-%m-%d"),"Dir":direction,
                       "Spot":round(spot,2),"OptPx":round(opt_px,2),"IV%":round(iv,1),
                       "Score":score,"P&L(₹)":round(pnl,2),"P&L(%)":round(pnl_pct*100,2),
                       "Equity":round(equity,2),"Result":"✅WIN" if pnl>0 else "❌LOSS"})
        equity_curve[-1]=equity

    if not trades: return None, "No trades generated — try longer lookback or different ticker"
    tdf=pd.DataFrame(trades)
    wins=(tdf["P&L(₹)"]>0).sum(); total=len(tdf)
    avg_w=tdf.loc[tdf["P&L(₹)"]>0,"P&L(₹)"].mean() if wins>0 else 0
    avg_l=tdf.loc[tdf["P&L(₹)"]<=0,"P&L(₹)"].mean() if (total-wins)>0 else 0
    peak=capital; mdd=0
    for eq in equity_curve:
        if eq>peak: peak=eq
        mdd=max(mdd,(peak-eq)/peak*100)
    stats={"total":total,"wins":int(wins),"wr":round(wins/total*100,1),
           "total_pnl":round(tdf["P&L(₹)"].sum(),2),
           "avg_w":round(avg_w,2),"avg_l":round(avg_l,2),
           "rr":round(abs(avg_w/avg_l),2) if avg_l!=0 else 0,
           "mdd":round(mdd,2),"final":round(equity,2),
           "ret_pct":round((equity-capital)/capital*100,2),
           "curve":equity_curve}
    return tdf, stats

# ──────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def vline(fig, x, label="", color="yellow", dash="dash", row=1, col=1):
    fig.add_vline(x=x, line_dash=dash, line_color=color, line_width=1.5,
                  annotation_text=label, annotation_font_color=color, row=row, col=col)

def plot_chain(df, spot):
    rng=spot*0.05
    sub=df[df["strikePrice"].between(spot-rng,spot+rng)]
    if sub.empty: sub=df
    fig=make_subplots(2,2,subplot_titles=["CE vs PE Premium","Open Interest","IV Smile","Volume"])
    x=sub["strikePrice"]
    fig.add_trace(go.Bar(name="CE",x=x,y=sub["CE_LTP"],marker_color="#00ff88",opacity=.85),1,1)
    fig.add_trace(go.Bar(name="PE",x=x,y=sub["PE_LTP"],marker_color="#ff3b5c",opacity=.85),1,1)
    fig.add_trace(go.Bar(name="CE OI",x=x,y=sub["CE_OI"],marker_color="#00e5ff",opacity=.8),1,2)
    fig.add_trace(go.Bar(name="PE OI",x=x,y=sub["PE_OI"],marker_color="#ff9500",opacity=.8),1,2)
    fig.add_trace(go.Scatter(name="CE IV",x=x,y=sub["CE_IV"],mode="lines+markers",line=dict(color="#00ff88",width=2)),2,1)
    fig.add_trace(go.Scatter(name="PE IV",x=x,y=sub["PE_IV"],mode="lines+markers",line=dict(color="#ff3b5c",width=2)),2,1)
    fig.add_trace(go.Bar(name="CE Vol",x=x,y=sub["CE_volume"],marker_color="#00ff8866"),2,2)
    fig.add_trace(go.Bar(name="PE Vol",x=x,y=sub["PE_volume"],marker_color="#ff3b5c66"),2,2)
    for r in [1,2]:
        for c in [1,2]: vline(fig,spot,"Spot","yellow",row=r,col=c)
    fig.update_layout(template=DARK,height=520,barmode="group",title="Live Option Chain",margin=dict(t=50,b=10))
    return fig

def plot_oi(df, spot, sig):
    rng=spot*0.055
    sub=df[df["strikePrice"].between(spot-rng,spot+rng)]
    if sub.empty: sub=df
    sub=sub.copy()
    sub["CE_prev"]=(sub["CE_OI"]-sub["CE_changeOI"]).clip(lower=1)
    sub["PE_prev"]=(sub["PE_OI"]-sub["PE_changeOI"]).clip(lower=1)
    sub["CE_pct"]=(sub["CE_changeOI"]/sub["CE_prev"]*100).fillna(0)
    sub["PE_pct"]=(sub["PE_changeOI"]/sub["PE_prev"]*100).fillna(0)
    fig=make_subplots(3,1,subplot_titles=["Open Interest","Change in OI","% Change in OI"])
    x=sub["strikePrice"]
    fig.add_trace(go.Bar(name="CE OI",x=x,y=sub["CE_OI"],marker_color="#00e5ff",opacity=.8),1,1)
    fig.add_trace(go.Bar(name="PE OI",x=x,y=sub["PE_OI"],marker_color="#ff9500",opacity=.8),1,1)
    fig.add_trace(go.Bar(name="ΔCE",x=x,y=sub["CE_changeOI"],
                          marker_color=["#00ff88" if v>=0 else "#ff3b5c" for v in sub["CE_changeOI"]]),2,1)
    fig.add_trace(go.Bar(name="ΔPE",x=x,y=sub["PE_changeOI"],
                          marker_color=["#ff9500" if v>=0 else "#8888ff" for v in sub["PE_changeOI"]]),2,1)
    fig.add_trace(go.Bar(name="CE%",x=x,y=sub["CE_pct"],
                          marker_color=["#00ff8888" if v>=0 else "#ff3b5c88" for v in sub["CE_pct"]]),3,1)
    fig.add_trace(go.Bar(name="PE%",x=x,y=sub["PE_pct"],
                          marker_color=["#ff950088" if v>=0 else "#8888ff88" for v in sub["PE_pct"]]),3,1)
    for r in [1,2,3]:
        vline(fig,spot,"Spot","yellow",row=r)
        if sig["resistance"]: vline(fig,sig["resistance"],"R","#ff3b5c","dot",r,1)
        if sig["support"]:    vline(fig,sig["support"],"S","#00ff88","dot",r,1)
    fig.update_layout(template=DARK,height=620,barmode="group",title="OI Analysis",margin=dict(t=50,b=10))
    return fig

def plot_greeks(df, spot):
    rng=spot*0.04
    sub=df[df["strikePrice"].between(spot-rng,spot+rng)]
    if sub.empty: sub=df
    fig=make_subplots(2,2,subplot_titles=["Delta","Gamma","Theta (daily ₹)","Vega"])
    x=sub["strikePrice"]
    for (r,c,name,cc,cp) in [(1,1,"Delta","CE_delta","PE_delta"),(1,2,"Gamma","CE_gamma","PE_gamma"),
                              (2,1,"Theta","CE_theta","PE_theta"),(2,2,"Vega","CE_vega","PE_vega")]:
        for col,color in [(cc,"#00ff88"),(cp,"#ff3b5c")]:
            if col in sub.columns:
                fig.add_trace(go.Scatter(name=col,x=x,y=sub[col],mode="lines",line=dict(color=color,width=2)),r,c)
        vline(fig,spot,"",row=r,col=c)
    fig.update_layout(template=DARK,height=520,title="Live Greeks",margin=dict(t=50,b=10))
    return fig

def plot_straddle_hist(sym, straddle_now):
    hist=get_history(sym,"45d")
    if hist.empty: return None, None
    close=hist["Close"].squeeze().astype(float)
    rv7=close.pct_change().rolling(7).std()*np.sqrt(252)
    est=(close*rv7*np.sqrt(7/252)*0.8).dropna()
    if est.empty: return None, None
    p25,p50,p75=est.quantile([.25,.5,.75])
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=est.index,y=est,name="Hist est",line=dict(color="#00e5ff",width=2)))
    if straddle_now>0:
        fig.add_hline(y=straddle_now,line_color="yellow",line_dash="dash",
                      annotation_text=f"Now: {straddle_now:.1f}",annotation_font_color="yellow")
    fig.add_hline(y=p25,line_color="#00ff88",line_dash="dot",annotation_text=f"P25:{p25:.1f}")
    fig.add_hline(y=p75,line_color="#ff3b5c",line_dash="dot",annotation_text=f"P75:{p75:.1f}")
    fig.add_hline(y=p50,line_color="#ff9500",line_dash="dot",annotation_text=f"P50:{p50:.1f}")
    fig.update_layout(template=DARK,height=340,title="Straddle vs 45-Day History")
    if straddle_now>0:
        if straddle_now<p25:   verdict=("✅ CHEAP — below P25, great time to buy!", "#00ff88")
        elif straddle_now>p75: verdict=("⚠️ EXPENSIVE — above P75, avoid buying!", "#ff3b5c")
        else:                   verdict=("🟡 FAIR VALUE — be selective", "#ff9500")
    else: verdict=("ℹ️ No straddle price","#5a7a9a")
    return fig, verdict

# ──────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown("<h1 style='font-size:24px;letter-spacing:2px'>⚡ PRO OPTIONS INTELLIGENCE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#5a7a9a;margin-top:-10px'>Live Chain · Greeks · OI · Signals · Backtest · Trade History</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("### ⚙️ Setup")
        src_type = st.selectbox("Data source", ["🇮🇳 NSE Index","🇮🇳 NSE Stock","🌐 Global (yFinance)"])
        yf_hist_sym = "^NSEI"

        if src_type=="🇮🇳 NSE Index":
            nse_sym = st.selectbox("Index", list(NSE_IDX.keys()))
            yf_hist_sym = NSE_IDX[nse_sym]
            data_src = "nse"; fetch_sym = nse_sym
        elif src_type=="🇮🇳 NSE Stock":
            nse_sym = st.text_input("NSE symbol","RELIANCE").upper().strip()
            yf_hist_sym = nse_sym+".NS"
            data_src = "nse"; fetch_sym = nse_sym
        else:
            g_choice = st.selectbox("Instrument", list(GLOBAL_MAP.keys())+["Custom"])
            if g_choice=="Custom":
                fetch_sym = st.text_input("yFinance ticker","AAPL").upper().strip()
            else:
                fetch_sym = GLOBAL_MAP[g_choice]
            yf_hist_sym = fetch_sym
            data_src = "yf"

        st.divider()
        st.markdown("### 💰 Risk")
        capital  = st.number_input("Capital (₹)", 50_000, 10_000_000, 100_000, 10_000)
        pos_pct  = st.slider("Position size %", 2, 15, 5)
        sl_pct   = st.slider("Stop loss % (premium)", 20, 50, 30)
        tgt_pct  = st.slider("Target % (premium)", 30, 200, 60)
        st.divider()
        fetch_btn   = st.button("🔄 FETCH LIVE DATA", type="primary", use_container_width=True)
        clear_cache = st.button("🗑️ Clear Cache", use_container_width=True)
        auto_ref    = st.checkbox("Auto-refresh (90s)")
        if clear_cache:
            st.cache_data.clear()
            st.rerun()
        st.caption("⚠️ Educational only. Not financial advice.")

    # ── FETCH ──
    df_raw=None; spot=0; expiries=[]

    with st.spinner("⏳ Fetching — respecting 1.5s rate limit between calls…"):
        if data_src=="nse":
            df_raw, spot, expiries = get_nse_chain(fetch_sym)
            if df_raw is None or spot==0:
                st.info("NSE API unavailable (needs Indian IP). Trying yfinance fallback…")
                exps = get_expiries(yf_hist_sym)
                if exps:
                    sel_exp = exps[0]
                    calls, puts = get_chain(yf_hist_sym, sel_exp)
                    spot = get_spot(yf_hist_sym)
                    if calls is not None and spot>0:
                        df_raw = build_chain_df(calls, puts, sel_exp, spot)
                        expiries = exps
        else:
            exps = get_expiries(fetch_sym)
            if exps:
                sel_exp = exps[0]
                calls, puts = get_chain(fetch_sym, sel_exp)
                spot = get_spot(fetch_sym)
                if calls is not None and spot>0:
                    df_raw = build_chain_df(calls, puts, sel_exp, spot)
                    expiries = exps
            else:
                st.warning("No options found — trying spot price only…")
                spot = get_spot(fetch_sym)

    if df_raw is None or df_raw.empty or spot==0:
        st.error("❌ Could not fetch live data.")
        st.markdown("""
<div style='background:#0a1929;border:1px solid #1e3050;border-radius:12px;padding:18px;margin:10px 0'>
<b style='color:#ff9500'>Troubleshooting:</b><br><br>
1. <b>Rate limited?</b> Wait 20s → click Fetch again. The 1.5s throttle resets each button press.<br>
2. <b>Market closed?</b> Options data unavailable outside market hours. Try BTC-USD (24/7) or GC=F.<br>
3. <b>NSE outside India?</b> Use Global → BTC/USD, Gold, or any US ticker like AAPL, TSLA.<br>
4. <b>Wrong symbol?</b> Check yfinance ticker: AAPL, BTC-USD, GC=F, ^NSEI, ^NSEBANK<br>
</div>""", unsafe_allow_html=True)
        if st.button("🔄 Force Clear Cache & Retry"):
            st.cache_data.clear()
            st.rerun()
        return

    spot = float(spot)

    # Expiry selector
    if expiries:
        with st.sidebar:
            sel_expiry = st.selectbox("Expiry", expiries[:8])
    else:
        sel_expiry = df_raw["expiryDate"].iloc[0] if "expiryDate" in df_raw.columns and not df_raw.empty else ""

    # Filter + refetch chain for selected expiry if using yfinance
    if data_src=="yf" and sel_expiry and expiries and sel_expiry!=expiries[0]:
        calls2, puts2 = get_chain(fetch_sym, sel_expiry)
        if calls2 is not None:
            df_raw = build_chain_df(calls2, puts2, sel_expiry, spot)
    elif "expiryDate" in df_raw.columns and sel_expiry:
        filt = df_raw[df_raw["expiryDate"]==sel_expiry]
        if not filt.empty: df_raw = filt.copy()

    df_exp = df_raw.reset_index(drop=True)

    with st.spinner("🔢 Computing Greeks…"):
        df_exp = add_greeks(df_exp, spot, sel_expiry)

    sig = signals(df_exp, spot)

    all_strikes = sorted(df_exp["strikePrice"].unique().tolist())
    atm_pos = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i]-spot)) if all_strikes else 0

    # ── TOP METRICS ──
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("📍 Spot",     f"{spot:,.2f}")
    c2.metric("🎯 ATM",      f"{sig['atm']:,.0f}")
    c3.metric("📊 PCR",      f"{sig['pcr']:.3f}")
    c4.metric("💀 Max Pain", f"{sig['max_pain']:,.0f}")
    c5.metric("🌡 ATM IV",   f"{sig['atm_iv']:.1f}%")
    c6.metric("↕ Skew",      f"{sig['skew']:+.2f}%")
    c7.metric("♟ Straddle",  f"{sig['straddle']:.2f}")

    # ── SIGNAL BANNER ──
    conf=sig["conf"]
    bc="#00ff88" if conf>=72 else "#ff9500" if conf>=58 else "#ff3b5c"
    def tag(s):
        if "BUY" in s: return f'<span class="tag-buy">{s}</span>'
        if "WAIT" in s or "WATCH" in s or "TRADE" in s: return f'<span class="tag-wait">{s}</span>'
        return f'<span class="tag-wait">{s}</span>'

    st.markdown(f"""
<div class="signal-box">
  <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px">
    <div>
      <div style="color:#5a7a9a;font-size:10px;letter-spacing:2px">PRIMARY SIGNAL</div>
      <div style="font-size:22px;font-weight:700;color:{bc};font-family:Space Mono">{sig['rec']}</div>
      <div style="color:#7a9ab5;font-size:12px;margin-top:4px">{' · '.join(sig['reasons'][:3])}</div>
    </div>
    <div style="text-align:right">
      <div style="color:#5a7a9a;font-size:10px;letter-spacing:2px">CONFIDENCE</div>
      <div style="font-size:34px;font-weight:700;color:{bc};font-family:Space Mono">{conf}%</div>
      <div style="background:{bc}22;border-radius:20px;height:5px;width:100px;margin-top:4px;margin-left:auto">
        <div style="background:{bc};border-radius:20px;height:5px;width:{conf}%"></div>
      </div>
    </div>
  </div>
  <div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap">
    {tag("⚡ " + sig['scalp'])} {tag("📅 " + sig['intraday'])} {tag("🗓 " + sig['swing'])} {tag("📆 " + sig['pos'])}
  </div>
</div>""", unsafe_allow_html=True)

    for ab in sig["abnormal"]:
        st.markdown(f'<div class="alert">{ab}</div>', unsafe_allow_html=True)
    if sig["gamma_blast"]:
        st.markdown('<div class="gamma">⚡ <b style="color:#ff3b5c">GAMMA BLAST ALERT</b> — Massive OI at ATM. Breakout will cause explosive premium move. Straddle buyers: wait for direction then enter immediately.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TABS
    # ─────────────────────────────────────────────────────────────────────────
    tabs = st.tabs(["📊 Chain","🔢 Greeks","📈 OI","⚡ Live Trade","🔬 Backtest","📋 History","🧠 Analysis"])

    # ═══════════════════ TAB 1 — CHAIN ═══════════════════════════════════════
    with tabs[0]:
        st.markdown('<div class="sum">Live CE/PE premiums, OI, IV across strikes. Green = calls, Red = puts. Straddle vs 45-day history tells if options are cheap or expensive for buyers right now.</div>', unsafe_allow_html=True)
        def_strikes = all_strikes[max(0,atm_pos-8):atm_pos+9]
        sel_st = st.multiselect("Strikes", all_strikes, default=def_strikes)
        if not sel_st: sel_st=def_strikes
        show_cols=[c for c in ["CE_LTP","CE_OI","CE_changeOI","CE_IV","CE_volume","strikePrice","PE_LTP","PE_OI","PE_changeOI","PE_IV","PE_volume"] if c in df_exp.columns]
        st.dataframe(df_exp[df_exp["strikePrice"].isin(sel_st)][show_cols].round(2), use_container_width=True, height=260)
        st.plotly_chart(plot_chain(df_exp, spot), use_container_width=True)
        st.markdown("#### 📉 Straddle vs History")
        fig_s, verdict = plot_straddle_hist(yf_hist_sym, sig["straddle"])
        if fig_s: st.plotly_chart(fig_s, use_container_width=True)
        if verdict: st.markdown(f'<div style="background:{verdict[1]}11;border:1px solid {verdict[1]}44;border-radius:8px;padding:10px 14px;color:{verdict[1]};font-weight:600">{verdict[0]}</div>', unsafe_allow_html=True)

    # ═══════════════════ TAB 2 — GREEKS ══════════════════════════════════════
    with tabs[1]:
        st.markdown('<div class="sum">Delta=direction, Gamma=acceleration (peaks at ATM), Theta=daily decay (enemy of buyers), Vega=IV sensitivity (gains on spikes). Buy when: Delta 0.3-0.6, high Vega, IV &lt; 25%.</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_greeks(df_exp, spot), use_container_width=True)
        ai2=(df_exp["strikePrice"]-spot).abs().idxmin()
        atm2=df_exp.loc[ai2]
        st.markdown(f"#### 🎯 ATM Strike {atm2['strikePrice']:,.0f} Greeks")
        g1,g2=st.columns(2)
        for col, px, label, color in [(g1,"CE","📗 CALL","#00ff88"),(g2,"PE","📕 PUT","#ff3b5c")]:
            with col:
                st.markdown(f"<h5 style='color:{color}'>{label}</h5>", unsafe_allow_html=True)
                cs=st.columns(3)
                for i,(name,key) in enumerate([("Delta",f"{px}_delta"),("Gamma",f"{px}_gamma"),("Theta",f"{px}_theta"),("Vega",f"{px}_vega"),("IV%",f"{px}_IV"),("Rho",f"{px}_rho")]):
                    fmt=f"{float(atm2.get(key,0)):.2f}%" if "IV" in name else f"{float(atm2.get(key,0)):.4f}"
                    cs[i%3].metric(name, fmt)
        iv=sig["atm_iv"]; ce_th=float(atm2.get("CE_theta",0)); ce_ve=float(atm2.get("CE_vega",0))
        tips=[]
        if iv>40: tips.append(("🔴 IV >40%: theta far exceeds vega benefit for buyers — WAIT","#ff3b5c"))
        elif iv<15: tips.append(("✅ IV <15%: premium cheap, any IV spike gives massive vega gains","#00ff88"))
        if abs(ce_th)>abs(ce_ve): tips.append(("⚠️ |Theta| > Vega: decay eating gains — prefer ITM or nearer strikes","#ff9500"))
        else: tips.append(("✅ Vega > |Theta|: good time/IV balance for buyers","#00ff88"))
        for t,clr in tips:
            st.markdown(f'<div style="background:{clr}11;border-left:3px solid {clr};padding:10px;border-radius:6px;margin:4px 0;font-size:13px">{t}</div>', unsafe_allow_html=True)
        gcols=[c for c in ["strikePrice","CE_IV","CE_delta","CE_gamma","CE_theta","CE_vega","PE_IV","PE_delta","PE_gamma","PE_theta","PE_vega"] if c in df_exp.columns]
        st.dataframe(df_exp[df_exp["strikePrice"].isin(sel_st)][gcols].round(5), use_container_width=True)

    # ═══════════════════ TAB 3 — OI ══════════════════════════════════════════
    with tabs[2]:
        st.markdown('<div class="sum">Max CE OI = resistance wall. Max PE OI = support floor. Rising OI + rising price = long buildup (bullish). Rising OI + falling price = short buildup (bearish).</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_oi(df_exp, spot, sig), use_container_width=True)
        m1,m2,m3,m4=st.columns(4)
        m1.metric("🔴 Resistance",f"{sig['resistance']:,.0f}")
        m2.metric("🟢 Support",   f"{sig['support']:,.0f}")
        m3.metric("🎯 Max Pain",  f"{sig['max_pain']:,.0f}")
        m4.metric("📊 PCR",       f"{sig['pcr']:.4f}")
        b1,b2=st.columns(2)
        with b1:
            st.markdown("**Top CE OI strikes (resistance)**")
            ce_top=df_exp.nlargest(5,"CE_OI")[["strikePrice","CE_OI","CE_changeOI","CE_LTP"]].copy()
            ce_top["Signal"]=ce_top["CE_changeOI"].apply(lambda x:"🔴 Call Writing" if x>=0 else "🟢 Call Unwinding")
            st.dataframe(ce_top, use_container_width=True)
        with b2:
            st.markdown("**Top PE OI strikes (support)**")
            pe_top=df_exp.nlargest(5,"PE_OI")[["strikePrice","PE_OI","PE_changeOI","PE_LTP"]].copy()
            pe_top["Signal"]=pe_top["PE_changeOI"].apply(lambda x:"🟢 Put Writing" if x>=0 else "🔴 Put Unwinding")
            st.dataframe(pe_top, use_container_width=True)

    # ═══════════════════ TAB 4 — LIVE TRADE ══════════════════════════════════
    with tabs[3]:
        st.markdown('<div class="sum">Paper trade execution using live option prices. 1.5s delay enforced between every API call. Pre-calculated entry/SL/target. Buyer-only. Tracks P&L vs live price in real-time.</div>', unsafe_allow_html=True)
        la,lb=st.columns([3,2])
        with la:
            st.markdown("#### 🚨 New Trade")
            t1,t2=st.columns(2)
            side=t1.selectbox("Side",["CE (Call)","PE (Put)"])
            px="CE" if "CE" in side else "PE"
            t_strike=t1.selectbox("Strike", all_strikes, index=atm_pos)
            lots=t2.number_input("Lots",1,100,1)
            l_sl=t2.slider("SL %",20,50,sl_pct,key="l_sl")
            l_tgt=t2.slider("Target %",30,200,tgt_pct,key="l_tgt")
            row_s=df_exp[df_exp["strikePrice"]==t_strike]
            opt_px=float(row_s[f"{px}_LTP"].values[0]) if not row_s.empty and f"{px}_LTP" in row_s.columns else 0
            if opt_px>0:
                sl_a=round(opt_px*(1-l_sl/100),2); tgt_a=round(opt_px*(1+l_tgt/100),2)
                risk=(opt_px-sl_a)*lots; rew=(tgt_a-opt_px)*lots
                rr=round(rew/risk,2) if risk>0 else 0
                pm1,pm2,pm3,pm4=st.columns(4)
                pm1.metric("Entry",f"₹{opt_px:.2f}")
                pm2.metric("SL",f"₹{sl_a:.2f}",f"-{l_sl}%")
                pm3.metric("Target",f"₹{tgt_a:.2f}",f"+{l_tgt}%")
                pm4.metric("R:R",f"1:{rr}")
            else:
                sl_a=tgt_a=0
            st.markdown(f'<div style="background:#0a1929;border:1px solid #1e3050;border-radius:8px;padding:12px;margin:8px 0"><b style="color:#00e5ff">{sig["rec"]}</b> | Conf: <b>{sig["conf"]}%</b><br><span style="color:#5a7a9a;font-size:12px">{" · ".join(sig["reasons"][:2])}</span></div>', unsafe_allow_html=True)
            if st.button("📈 ENTER PAPER TRADE", type="primary", use_container_width=True):
                if opt_px>0:
                    t={"id":len(st.session_state.trades)+1,"sym":fetch_sym,"expiry":sel_expiry,
                       "strike":t_strike,"side":px,"entry":opt_px,"sl":sl_a,"target":tgt_a,
                       "lots":lots,"conf":sig["conf"],"rec":sig["rec"],
                       "time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"status":"OPEN",
                       "exit":None,"pnl":None,"exit_time":None}
                    st.session_state.active.append(t)
                    st.session_state.trades.append(t)
                    st.success(f"✅ Entered: {fetch_sym} {t_strike} {px} @ ₹{opt_px:.2f}")
                else:
                    st.error("Option price is 0 — market may be closed.")
        with lb:
            st.markdown("#### 📊 Open Positions")
            if not st.session_state.active:
                st.info("No open trades. Enter one from the left.")
            for i,t in enumerate(st.session_state.active):
                if t["status"]!="OPEN": continue
                r=df_exp[df_exp["strikePrice"]==t["strike"]]
                curr=float(r[f"{t['side']}_LTP"].values[0]) if not r.empty and f"{t['side']}_LTP" in r.columns else t["entry"]
                pnl=round((curr-t["entry"])*t["lots"],2)
                pp=round((curr-t["entry"])/t["entry"]*100,2) if t["entry"]>0 else 0
                clr="#00ff88" if pnl>=0 else "#ff3b5c"
                cls="tc-open" if pnl==0 else ("tc-win" if pnl>0 else "tc-loss")
                st.markdown(f"""
<div class="trade-card {cls}">
<b>{t['sym']} {t['strike']} {t['side']}</b><br>
Entry ₹{t['entry']:.2f} → Live ₹{curr:.2f}<br>
SL ₹{t['sl']:.2f} | Tgt ₹{t['target']:.2f}<br>
<b style="color:{clr}">P&L: ₹{pnl:+.2f} ({pp:+.1f}%)</b>
</div>""", unsafe_allow_html=True)
                if st.button(f"Exit #{t['id']}", key=f"ex_{i}_{t['id']}"):
                    st.session_state.active[i]["status"]="CLOSED"
                    st.session_state.active[i]["exit"]=curr
                    st.session_state.active[i]["pnl"]=pnl
                    st.session_state.active[i]["exit_time"]=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    for h in st.session_state.trades:
                        if h["id"]==t["id"]: h.update(st.session_state.active[i])
                    st.rerun()

    # ═══════════════════ TAB 5 — BACKTEST ════════════════════════════════════
    with tabs[4]:
        st.markdown('<div class="sum">Uses IDENTICAL PCR scoring, IV filter, and position sizing as live trading. Ensures live results match backtest. Equity curve, win rate, drawdown, full trade log all shown.</div>', unsafe_allow_html=True)
        ba,bb,bc_=st.columns(3)
        bt_look=ba.slider("Lookback days",20,120,60)
        bt_cap=bb.number_input("Capital",50_000,5_000_000,int(capital),10_000,key="bt_cap")
        bt_pos=bc_.slider("Position size %",2,20,pos_pct,key="bt_pos")
        if st.button("🔬 RUN BACKTEST", type="primary", use_container_width=True):
            with st.spinner("Running backtest with same signal engine as live trading…"):
                tdf,stats=backtest(yf_hist_sym,bt_look,bt_cap,bt_pos)
                st.session_state.bt_result=(tdf,stats)
        if st.session_state.bt_result:
            tdf,stats=st.session_state.bt_result
            if tdf is None:
                st.error(f"Backtest failed: {stats}")
            else:
                k1,k2,k3,k4,k5,k6,k7=st.columns(7)
                k1.metric("Trades",stats["total"]); k2.metric("Win Rate",f"{stats['wr']}%")
                k3.metric("Total P&L",f"₹{stats['total_pnl']:+,.0f}"); k4.metric("Avg Win",f"₹{stats['avg_w']:,.0f}")
                k5.metric("Avg Loss",f"₹{stats['avg_l']:,.0f}"); k6.metric("R:R",stats['rr'])
                k7.metric("Max DD",f"{stats['mdd']}%")
                rc="#00ff88" if stats["ret_pct"]>=0 else "#ff3b5c"
                st.markdown(f'<div style="text-align:center;font-size:26px;color:{rc};font-family:Space Mono;margin:10px 0">Return: {stats["ret_pct"]:+.2f}%  ·  Final: ₹{stats["final"]:,.0f}</div>', unsafe_allow_html=True)
                eq=stats["curve"]
                fig_eq=go.Figure(go.Scatter(y=eq,mode="lines",line=dict(color="#00e5ff",width=2),
                                             fill="tozeroy",fillcolor="rgba(0,229,255,0.06)"))
                fig_eq.add_hline(y=bt_cap,line_dash="dash",line_color="#ff9500",annotation_text="Start")
                fig_eq.update_layout(template=DARK,height=320,title="Equity Curve",xaxis_title="Step",yaxis_title="₹")
                st.plotly_chart(fig_eq, use_container_width=True)
                w=tdf[tdf["P&L(₹)"]>0]["P&L(₹)"]; l=tdf[tdf["P&L(₹)"]<=0]["P&L(₹)"]
                fig_d=go.Figure()
                fig_d.add_trace(go.Histogram(x=w,name="Wins",marker_color="#00ff8877",nbinsx=20))
                fig_d.add_trace(go.Histogram(x=l,name="Losses",marker_color="#ff3b5c77",nbinsx=20))
                fig_d.update_layout(template=DARK,height=260,title="P&L Distribution",barmode="overlay")
                st.plotly_chart(fig_d, use_container_width=True)
                st.dataframe(tdf, use_container_width=True, height=280)
                st.success("✅ Backtest-Live parity: identical scoring thresholds, IV filters, and position sizing used in both.")

    # ═══════════════════ TAB 6 — HISTORY ══════════════════════════════════════
    with tabs[5]:
        st.markdown('<div class="sum">Complete paper trade log with per-trade P&L and outcome. Track win patterns, review mistakes, and export for analysis. Only closed trades show final P&L.</div>', unsafe_allow_html=True)
        if not st.session_state.trades:
            st.info("No trades yet — use the Live Trade tab to paper trade.")
        else:
            all_t=pd.DataFrame(st.session_state.trades)
            closed=all_t[all_t["status"]=="CLOSED"].copy()
            if not closed.empty:
                closed["pnl"]=pd.to_numeric(closed["pnl"],errors="coerce").fillna(0)
                total_pnl=closed["pnl"].sum(); wr=(closed["pnl"]>0).mean()*100
                h1,h2,h3,h4=st.columns(4)
                h1.metric("Total Trades",len(all_t)); h2.metric("Closed",len(closed))
                h3.metric("Win Rate",f"{wr:.1f}%"); h4.metric("Net P&L",f"₹{total_pnl:+,.2f}")
                fig_p=go.Figure(go.Bar(y=closed["pnl"].values,
                    marker_color=["#00ff88" if p>0 else "#ff3b5c" for p in closed["pnl"]]))
                fig_p.update_layout(template=DARK,height=260,title="Per-Trade P&L")
                st.plotly_chart(fig_p, use_container_width=True)
            disp_cols=[c for c in ["id","time","sym","strike","side","entry","exit","lots","pnl","status","rec","conf"] if c in all_t.columns]
            st.dataframe(all_t[disp_cols], use_container_width=True)
            st.download_button("📥 Export CSV", all_t.to_csv(index=False), "trades.csv","text/csv",use_container_width=True)
            if st.button("🗑️ Clear All Trades", use_container_width=True):
                st.session_state.trades=[]; st.session_state.active=[]; st.rerun()

    # ═══════════════════ TAB 7 — ANALYSIS ═════════════════════════════════════
    with tabs[6]:
        st.markdown('<div class="sum">Full narrative: PCR, max pain pull, OI walls, IV environment, 3-scenario breakdown (bull/flat/bear). Buyer-specific rules on when to enter, exit, and stay out.</div>', unsafe_allow_html=True)
        pcr=sig["pcr"]; iv=sig["atm_iv"]; mp=sig["max_pain"]; res=sig["resistance"]; sup=sig["support"]
        mp_pct=(mp-spot)/spot*100 if spot>0 else 0
        pcr_t=("extreme put OI → contrarian BULLISH" if pcr>1.5 else "put-heavy → mild bullish lean" if pcr>1.1
               else "extreme call OI → contrarian BEARISH" if pcr<0.5 else "call-heavy → mild bearish lean" if pcr<0.9 else "balanced / neutral")
        iv_t=("AVOID BUYING — crushingly expensive" if iv>50 else "HIGH — selective only" if iv>35
              else "MODERATE — acceptable" if iv>20 else "LOW — prime buyer environment" if iv>0 else "N/A")

        st.markdown(f"""
<div style='background:#0a1929;border:1px solid #1e3050;border-radius:12px;padding:20px;line-height:2'>
<h4 style='color:#00e5ff;margin-top:0'>🏗 Market Structure — {fetch_sym} @ {spot:,.2f}</h4>

**PCR {pcr:.4f}** → {pcr_t}<br>
**ATM IV {iv:.1f}%** → {iv_t}<br>
**Max Pain {mp:,.0f}** → {mp_pct:+.1f}% from spot {"(upward pull)" if mp_pct>0 else "(downward pull)"}<br>
**Resistance** {res:,.0f} (peak CE OI wall) &nbsp;|&nbsp; **Support** {sup:,.0f} (peak PE OI wall)<br>
**Straddle** {sig['straddle']:.2f} = implied move {sig['straddle']/spot*100:.2f}% this expiry
</div>""", unsafe_allow_html=True)

        s1,s2,s3=st.columns(3)
        for col,title,trigger,action,color in [
            (s1,"🟢 BULLISH",f"Break + hold above {res:,.0f}",f"Buy {int(sig['atm'])} CE, target {int(res*1.015):,}, SL 30% of premium","#00ff88"),
            (s2,"⚪ NEUTRAL",f"Range {sup:,.0f} – {res:,.0f}","DO NOT buy options. Theta kills all premium. Wait for breakout.","#5a7a9a"),
            (s3,"🔴 BEARISH", f"Break + hold below {sup:,.0f}",f"Buy {int(sig['atm'])} PE, target {int(sup*0.985):,}, SL 30% of premium","#ff3b5c"),
        ]:
            with col:
                st.markdown(f"""
<div style='background:#0a1929;border-top:3px solid {color};border:1px solid #1e3050;border-radius:10px;padding:16px'>
<h5 style='color:{color};margin-top:0'>{title}</h5>
<b>Trigger:</b> {trigger}<br><br>
<b>Action:</b> {action}
</div>""", unsafe_allow_html=True)

        rc2="#00ff88" if "CALL" in sig["rec"] else "#ff3b5c" if "PUT" in sig["rec"] else "#ff9500"
        st.markdown(f"""
<div style='background:#060e1a;border:1px solid {rc2}44;border-radius:12px;padding:20px;margin-top:16px'>
<h4 style='color:{rc2};margin-top:0'>🎯 {sig['rec']} — Confidence {sig['conf']}%</h4>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px'>
<div><b style='color:#00e5ff'>Trade Horizons:</b>
  <ul style='color:#8899aa'>
    <li>⚡ Scalping: {sig['scalp']}</li>
    <li>📅 Intraday: {sig['intraday']}</li>
    <li>🗓 Swing: {sig['swing']}</li>
    <li>📆 Positional: {sig['pos']}</li>
  </ul>
</div>
<div><b style='color:#00e5ff'>Option Buyer Rules:</b>
  <ul style='color:#8899aa'>
    <li>Never risk >5% capital per trade</li>
    <li>Only buy when ATM IV &lt; 30%</li>
    <li>Exit positions 5+ days before expiry</li>
    <li>Pre-set SL before entry — never widen it</li>
    <li>Book 50% at first target, trail rest at entry</li>
    <li>Never buy on high-IV event days (earnings, RBI etc.)</li>
  </ul>
</div>
</div>
<b>Reasoning:</b> {" · ".join(sig['reasons'])}
</div>""", unsafe_allow_html=True)
        st.caption(f"Last computed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Educational only")

    if auto_ref:
        time.sleep(90)
        st.cache_data.clear()
        st.rerun()

if __name__=="__main__":
    main()
