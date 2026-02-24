"""
Pro Options Dashboard — India + Global
Fixes: NSE headers, plain English, backtest logic explained simply
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests, time, random, warnings
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

# ── throttle (module-level, works inside cache) ──
_T = {"last": 0.0}
def _wait(gap=1.5):
    now = time.time()
    diff = now - _T["last"]
    if diff < gap:
        time.sleep(gap - diff + random.uniform(0.1, 0.3))
    _T["last"] = time.time()

# ─────────────────────────────────────────────────
st.set_page_config(page_title="⚡ ProOptions India", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
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
.explain{background:var(--s2);border-left:3px solid var(--acc);border-radius:0 8px 8px 0;
  padding:14px 18px;margin:10px 0;font-size:14px;line-height:1.7}
.explain b{color:var(--acc)}
.sig-box{background:var(--s1);border:1px solid var(--brd);border-radius:12px;
  padding:18px 22px;margin:10px 0;position:relative;overflow:hidden}
.sig-box::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--acc),var(--grn))}
.tag-b{background:#002a15;color:#00ff88;border:1px solid #00aa44;padding:4px 12px;
  border-radius:20px;font-size:12px;font-weight:600;display:inline-block;margin:3px 2px}
.tag-w{background:#2a1a00;color:#ff9500;border:1px solid #aa5500;padding:4px 12px;
  border-radius:20px;font-size:12px;display:inline-block;margin:3px 2px}
.alert{background:#1a0a00;border-left:3px solid var(--ora);border-radius:0 6px 6px 0;
  padding:10px 14px;margin:4px 0;font-size:13px}
.gamma-blast{background:#1a000a;border:2px solid #ff3b5c;border-radius:10px;
  padding:14px;animation:pulse 2s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(255,59,92,.5)}
  70%{box-shadow:0 0 0 10px rgba(255,59,92,0)}100%{box-shadow:0 0 0 0 rgba(255,59,92,0)}}
.tc{background:var(--s2);border:1px solid var(--brd);border-radius:10px;padding:14px;margin:6px 0}
.tc-o{border-left:3px solid var(--acc)}.tc-w{border-left:3px solid var(--grn)}.tc-l{border-left:3px solid var(--red)}
.card{background:var(--s1);border:1px solid var(--brd);border-radius:12px;padding:18px;margin:8px 0}
hr{border-color:var(--brd)!important}
</style>""", unsafe_allow_html=True)

RFR  = 0.065
DARK = "plotly_dark"

NSE_SYMS = ["NIFTY","BANKNIFTY","SENSEX","FINNIFTY","MIDCPNIFTY"]
NSE_TO_YF = {"NIFTY":"^NSEI","BANKNIFTY":"^NSEBANK","SENSEX":"^BSESN",
             "FINNIFTY":"NIFTY_FIN_SERVICE.NS","MIDCPNIFTY":"^NSMIDCP"}
YF_OPTIONS_SYMBOLS = {
    "NIFTY (via yFinance)":"^NSEI","BANKNIFTY (via yFinance)":"^NSEBANK",
    "SPY (S&P 500)":"SPY","QQQ (Nasdaq)":"QQQ",
    "AAPL":"AAPL","TSLA":"TSLA","NVDA":"NVDA","AMZN":"AMZN","Custom":"__custom__",
}
YF_SPOT_ONLY = {"BTC/USD":"BTC-USD","GOLD":"GC=F","SILVER":"SI=F",
                "USD/INR":"USDINR=X","CRUDE OIL":"CL=F","Custom":"__custom__"}

for k, v in {"trades":[],"active":[],"bt_result":None}.items():
    if k not in st.session_state: st.session_state[k] = v

# ─────────────────────────────────────────────────
# NSE FETCH — robust headers that work in India
# ─────────────────────────────────────────────────
@st.cache_data(ttl=90, show_spinner=False)
def fetch_nse(symbol: str):
    """
    NSE option chain with full browser simulation.
    Requires Indian IP. Returns (df, spot, expiries) or (None, 0, []).
    """
    # These headers exactly mimic Chrome on Windows visiting NSE
    hdrs = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/option-chain",
        "X-Requested-With": "XMLHttpRequest",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    s = requests.Session()
    s.headers.update(hdrs)

    try:
        # Visit 1: homepage sets basic cookies
        r1 = s.get("https://www.nseindia.com/", timeout=10, headers={
            **hdrs,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
        })
        time.sleep(1.2)

        # Visit 2: option-chain page — THIS is the page that sets auth cookies
        r2 = s.get("https://www.nseindia.com/option-chain", timeout=10, headers={
            **hdrs,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
        })
        time.sleep(1.0)

        # Visit 3: actual API
        if symbol in NSE_SYMS:
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"

        r3 = s.get(url, timeout=15)
        if r3.status_code != 200:
            return None, 0.0, []

        data  = r3.json().get("records", {})
        spot  = float(data.get("underlyingValue") or 0)
        exps  = data.get("expiryDates", [])
        rows  = []
        for rec in data.get("data", []):
            ce = rec.get("CE", {}); pe = rec.get("PE", {})
            rows.append({
                "strikePrice":  float(rec.get("strikePrice", 0)),
                "expiryDate":   rec.get("expiryDate", ""),
                "CE_LTP":       float(ce.get("lastPrice", 0)),
                "CE_OI":        float(ce.get("openInterest", 0)),
                "CE_changeOI":  float(ce.get("changeinOpenInterest", 0)),
                "CE_volume":    float(ce.get("totalTradedVolume", 0)),
                "CE_IV":        float(ce.get("impliedVolatility", 0)),
                "CE_bid":       float(ce.get("bidprice", 0)),
                "CE_ask":       float(ce.get("askPrice", 0)),
                "PE_LTP":       float(pe.get("lastPrice", 0)),
                "PE_OI":        float(pe.get("openInterest", 0)),
                "PE_changeOI":  float(pe.get("changeinOpenInterest", 0)),
                "PE_volume":    float(pe.get("totalTradedVolume", 0)),
                "PE_IV":        float(pe.get("impliedVolatility", 0)),
                "PE_bid":       float(pe.get("bidprice", 0)),
                "PE_ask":       float(pe.get("askPrice", 0)),
            })
        df = pd.DataFrame(rows)
        if df.empty or spot == 0: return None, 0.0, []
        return df, spot, exps
    except Exception as e:
        return None, 0.0, []


# ─────────────────────────────────────────────────
# YFINANCE FETCHERS
# ─────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def yf_spot(sym: str) -> float:
    _wait(1.5)
    try:
        fi = yf.Ticker(sym).fast_info
        for attr in ("last_price","lastPrice","regular_market_price","regularMarketPrice","previousClose"):
            v = getattr(fi, attr, None)
            if v and float(v) > 0: return float(v)
    except: pass
    _wait(1.5)
    try:
        h = yf.Ticker(sym).history(period="2d", interval="1d")
        if not h.empty: return float(h["Close"].squeeze().iloc[-1])
    except: pass
    return 0.0

@st.cache_data(ttl=120, show_spinner=False)
def yf_expiries(sym: str):
    _wait(1.5)
    try: return list(yf.Ticker(sym).options or [])
    except: return []

@st.cache_data(ttl=120, show_spinner=False)
def yf_chain(sym: str, expiry: str):
    _wait(1.5)
    try:
        c = yf.Ticker(sym).option_chain(expiry)
        return c.calls.copy(), c.puts.copy()
    except Exception as e:
        st.error(f"Chain error: {e}"); return None, None

@st.cache_data(ttl=300, show_spinner=False)
def yf_history(sym: str, period="60d"):
    _wait(1.5)
    try:
        df = yf.Ticker(sym).history(period=period, interval="1d", auto_adjust=True)
        return df if df is not None and not df.empty else pd.DataFrame()
    except: return pd.DataFrame()

def build_chain_df(calls, puts, expiry):
    if calls is None or calls.empty: return pd.DataFrame()
    df = pd.DataFrame()
    df["strikePrice"] = calls["strike"].values if "strike" in calls.columns else []
    df["expiryDate"]  = expiry
    for col, src in [("CE_LTP","lastPrice"),("CE_OI","openInterest"),
                     ("CE_volume","volume"),("CE_bid","bid"),("CE_ask","ask")]:
        df[col] = calls[src].values if src in calls.columns else 0
    df["CE_IV"]       = (calls["impliedVolatility"].values * 100) if "impliedVolatility" in calls.columns else 0
    df["CE_changeOI"] = 0
    if puts is not None and not puts.empty and "strike" in puts.columns:
        pi = puts.set_index("strike")
        for col, src in [("PE_LTP","lastPrice"),("PE_OI","openInterest"),
                         ("PE_volume","volume"),("PE_bid","bid"),("PE_ask","ask")]:
            df[col] = df["strikePrice"].map(pi[src] if src in pi.columns else pd.Series(dtype=float)).fillna(0).values
        df["PE_IV"] = df["strikePrice"].map(
            pi["impliedVolatility"]*100 if "impliedVolatility" in pi.columns else pd.Series(dtype=float)
        ).fillna(0).values
    else:
        for c in ["PE_LTP","PE_OI","PE_volume","PE_bid","PE_ask","PE_IV"]: df[c] = 0
    df["PE_changeOI"] = 0
    return df.fillna(0).reset_index(drop=True)


# ─────────────────────────────────────────────────
# BLACK-SCHOLES ENGINE
# ─────────────────────────────────────────────────
def bs(S, K, T, r, sig, kind="call"):
    if T<1e-6 or sig<1e-6 or S<=0 or K<=0:
        return dict(delta=0,gamma=0,theta=0,vega=0,price=0)
    d1 = (np.log(S/K)+(r+.5*sig**2)*T)/(sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    if kind=="call":
        price=S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2); delta=norm.cdf(d1)
        theta=(-(S*norm.pdf(d1)*sig)/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2))/365
    else:
        price=K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1); delta=norm.cdf(d1)-1
        theta=(-(S*norm.pdf(d1)*sig)/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2))/365
    gamma=norm.pdf(d1)/(S*sig*np.sqrt(T)); vega=S*norm.pdf(d1)*np.sqrt(T)/100
    return dict(delta=round(delta,4),gamma=round(gamma,8),theta=round(theta,4),
                vega=round(vega,4),price=round(max(price,0),4))

def calc_iv(mkt,S,K,T,r,kind="call"):
    if T<=0 or mkt<=0 or S<=0: return 0.20
    try: return max(brentq(lambda s:bs(S,K,T,r,s,kind)["price"]-mkt,1e-4,20.0,xtol=1e-5),0.001)
    except: return 0.20

def add_greeks(df, spot, expiry_str):
    if df is None or df.empty: return df
    try:
        fmt = "%d-%b-%Y" if "-" in str(expiry_str) and not str(expiry_str)[4:5].isdigit() else "%Y-%m-%d"
        T = max((datetime.strptime(str(expiry_str),fmt)-datetime.now()).days/365.0,1/365)
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


# ─────────────────────────────────────────────────
# SIGNAL ENGINE
# ─────────────────────────────────────────────────
def compute_signals(df, spot):
    out = dict(pcr=0,max_pain=spot,atm=spot,straddle=0,resistance=spot,support=spot,
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
    if iv>55: out["abnormal"].append(f"🔴 IV {iv:.0f}%: Options VERY expensive. Don't buy — you'll overpay for premium that crashes after the event")
    elif iv>35: out["abnormal"].append(f"⚠️ IV {iv:.0f}%: Options costly right now. Theta (time decay) will eat your profits faster than usual")
    elif 0<iv<15: out["abnormal"].append(f"✅ IV {iv:.0f}%: Options CHEAP. This is the best time to buy — you're getting premium at a discount!")
    sk=out["skew"]
    if sk>8: out["abnormal"].append(f"📊 Puts cost more than calls (skew +{sk:.0f}%): Big money is hedging against a fall. Follow the smart money → lean BEARISH")
    elif sk<-8: out["abnormal"].append(f"📊 Calls cost more than puts (skew {sk:.0f}%): Aggressive call buying happening. Market expecting a rise → lean BULLISH")
    mp_pct=(out["max_pain"]-spot)/spot*100 if spot>0 else 0
    if abs(mp_pct)>2: out["abnormal"].append(f"🎯 Max pain is {mp_pct:+.1f}% away at {out['max_pain']:,.0f}. Near expiry, markets tend to drift toward max pain like a magnet")
    if out["gamma_blast"]: out["abnormal"].append("⚡ GAMMA BLAST: Huge OI piled up right at current price. When this breaks, options will explode in value. Wait for breakout direction then buy immediately")

    score=50; direction="NONE"; reasons=[]
    pcr=out["pcr"]
    if pcr>1.5: score+=12; direction="CALL"; reasons.append(f"PCR {pcr:.2f}: Too many puts — contrarian signal, market likely to go UP")
    elif pcr>1.1: score+=6; direction="CALL"; reasons.append(f"PCR {pcr:.2f}: More puts than calls — mild bullish lean")
    elif pcr<0.5: score+=12; direction="PUT"; reasons.append(f"PCR {pcr:.2f}: Too many calls — contrarian signal, market likely to go DOWN")
    elif pcr<0.9: score+=6; direction="PUT"; reasons.append(f"PCR {pcr:.2f}: More calls than puts — mild bearish lean")
    else: reasons.append(f"PCR {pcr:.2f}: Balanced — no clear direction from options flow")

    if iv>50: score-=25; reasons.append("IV >50%: AVOID buying, options overpriced")
    elif iv>35: score-=12; reasons.append(f"IV {iv:.0f}%: Expensive, decay will hurt")
    elif 0<iv<18: score+=15; reasons.append(f"IV {iv:.0f}%: Cheap options — perfect buying conditions!")
    elif iv<=30: score+=8; reasons.append(f"IV {iv:.0f}%: Reasonable price for options")

    mp=out["max_pain"]
    if spot<mp and direction=="CALL": score+=10; reasons.append(f"Price below max pain {mp:,.0f}: gravity pulls market up — helps CALL buyers")
    elif spot>mp and direction=="PUT": score+=10; reasons.append(f"Price above max pain {mp:,.0f}: gravity pulls market down — helps PUT buyers")
    elif spot<mp and direction=="PUT": score-=8
    elif spot>mp and direction=="CALL": score-=8

    if direction=="CALL" and out["support"]<spot<out["resistance"]: score+=8; reasons.append(f"Price in buy zone between support {out['support']:,.0f} and resistance {out['resistance']:,.0f}")
    if direction=="PUT"  and out["support"]<spot<out["resistance"]: score+=8; reasons.append(f"Price near resistance wall {out['resistance']:,.0f} — ceiling likely to hold")

    ce_chg=float(df["CE_changeOI"].sum()); pe_chg=float(df["PE_changeOI"].sum())
    if pe_chg>ce_chg*1.3 and direction=="CALL": score+=7; reasons.append("Fresh put writing: sellers are building a support floor below market")
    if ce_chg>pe_chg*1.3 and direction=="PUT": score+=7; reasons.append("Fresh call writing: sellers are building a ceiling above market")

    score=min(max(int(score),0),100); out["conf"]=score; out["direction"]=direction; out["reasons"]=reasons
    A=int(out["atm"]); otm=int(A*(1.005 if direction=="CALL" else 0.995))
    if score>=72 and direction=="CALL":
        out["rec"]="🟢 BUY CALL (CE)"; out["scalp"]=f"Buy {A} CE — exit at +20 to 40%, cut loss at -30%"
        out["intraday"]=f"Buy {A} CE — target +60%, stop loss -30%"
        out["swing"]=f"Buy {otm} CE next expiry — target +80 to 100%"
        out["pos"]=f"Buy {otm} CE far expiry — hold for big move"
    elif score>=72 and direction=="PUT":
        out["rec"]="🔴 BUY PUT (PE)"; out["scalp"]=f"Buy {A} PE — exit at +20 to 40%, cut loss at -30%"
        out["intraday"]=f"Buy {A} PE — target +60%, stop loss -30%"
        out["swing"]=f"Buy {otm} PE next expiry — target +80 to 100%"
        out["pos"]=f"Buy {otm} PE far expiry — hold for big move"
    elif score>=58:
        out["rec"]="🟡 WATCH — Signal forming"
        out["scalp"]=out["intraday"]=out["swing"]=out["pos"]="Wait for more confirmation before entering"
    else:
        out["rec"]="⚪ NO TRADE — Stay out"
        out["scalp"]=out["intraday"]=out["swing"]=out["pos"]="No edge right now. Cash is a position too."
    return out


# ─────────────────────────────────────────────────
# BACKTEST — simple, explained logic
# ─────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def run_backtest(hist_sym: str, lookback: int, capital: float, pos_pct: float):
    """
    How this backtest works (plain English):
    1. We look at each day's closing price in the past N days
    2. We estimate IV from how much the price moved in past 7 days
    3. We score the day using the same rules as the live signal
    4. If score >= 72, we "buy" an option at estimated price
    5. Next day's price move determines if we won or lost
    6. We track equity across all trades
    """
    raw = yf_history(hist_sym, f"{lookback}d")
    if raw.empty or len(raw)<15: return None, "Not enough history data. Try longer lookback."

    close = raw["Close"].squeeze().astype(float)
    vol7  = close.pct_change().rolling(7).std() * np.sqrt(252) * 100
    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    vol3  = close.pct_change().rolling(3).std()  * np.sqrt(252) * 100
    vol10 = close.pct_change().rolling(10).std() * np.sqrt(252) * 100

    trades=[]; equity=float(capital); curve=[equity]
    for i in range(20, len(close)-1):
        iv   = float(vol7.iloc[i])  if not np.isnan(vol7.iloc[i])  else 20.0
        spot = float(close.iloc[i])
        v3   = float(vol3.iloc[i])  if not np.isnan(vol3.iloc[i])  else 20.0
        v10  = float(vol10.iloc[i]) if not np.isnan(vol10.iloc[i]) else 20.0
        pcr  = 1.0 + (v10 - v3) / (v10 + 1)  # proxy PCR
        mp   = (float(sma10.iloc[i]) + float(sma20.iloc[i])) / 2

        # Same scoring as live
        score=50; direction="NONE"
        if pcr>1.5:   score+=12; direction="CALL"
        elif pcr>1.1: score+=6;  direction="CALL"
        elif pcr<0.5: score+=12; direction="PUT"
        elif pcr<0.9: score+=6;  direction="PUT"
        if iv>50:    score-=25
        elif iv>35:  score-=12
        elif iv<18:  score+=15
        elif iv<=30: score+=8
        if spot<mp and direction=="CALL": score+=10
        elif spot>mp and direction=="PUT": score+=10
        elif spot<mp and direction=="PUT": score-=8
        elif spot>mp and direction=="CALL": score-=8
        if float(sma10.iloc[i])>float(sma20.iloc[i]) and direction=="CALL": score+=7
        elif float(sma10.iloc[i])<float(sma20.iloc[i]) and direction=="PUT": score+=7

        score=min(max(int(score),0),100); curve.append(equity)
        if score<72 or direction=="NONE": continue

        # Estimate ATM option price using Black-Scholes approximation
        opt_px = spot * max(iv/100, 0.01) * np.sqrt(7/365) * 0.4
        if opt_px<=0: continue

        # Next day's actual move determines P&L
        next_spot = float(close.iloc[i+1])
        ret       = (next_spot - spot) / spot
        raw_gain  = (ret*0.5*spot) if direction=="CALL" else (-ret*0.5*spot)
        theta_cost = opt_px/7  # 1 day of time decay
        pnl_pct   = max(min((raw_gain - theta_cost) / opt_px, 2.5), -0.5)  # capped at +250% / -50%
        pnl       = equity * (pos_pct/100) * pnl_pct
        equity    = max(equity+pnl, 0)
        trades.append({
            "Date": raw.index[i].strftime("%d-%b-%Y"),
            "Direction": direction, "Spot": round(spot,2),
            "Est. Option Price": round(opt_px,2), "IV%": round(iv,1),
            "Score": score, "Next Day Move%": round(ret*100,2),
            "P&L (₹)": round(pnl,2), "P&L (%)": round(pnl_pct*100,2),
            "Equity": round(equity,2), "Result": "✅ WIN" if pnl>0 else "❌ LOSS"
        })
        curve[-1]=equity

    if not trades: return None,"No trades — try longer lookback or different instrument"
    tdf=pd.DataFrame(trades)
    wins=(tdf["P&L (₹)"]>0).sum(); total=len(tdf)
    aw=tdf.loc[tdf["P&L (₹)"]>0,"P&L (₹)"].mean() if wins>0 else 0
    al=tdf.loc[tdf["P&L (₹)"]<=0,"P&L (₹)"].mean() if (total-wins)>0 else 0
    peak=capital; mdd=0
    for eq in curve:
        if eq>peak: peak=eq
        mdd=max(mdd,(peak-eq)/peak*100)
    stats={"total":total,"wins":int(wins),"wr":round(wins/total*100,1),
           "total_pnl":round(tdf["P&L (₹)"].sum(),2),
           "aw":round(aw,2),"al":round(al,2),
           "rr":round(abs(aw/al),2) if al!=0 else 0,
           "mdd":round(mdd,2),"final":round(equity,2),
           "ret%":round((equity-capital)/capital*100,2),"curve":curve}
    return tdf, stats


# ─────────────────────────────────────────────────
# PLOTS (all colors valid: 6-digit hex or rgba)
# ─────────────────────────────────────────────────
def vl(fig,x,lbl="",color="yellow",dash="dash",row=1,col=1):
    fig.add_vline(x=x,line_dash=dash,line_color=color,line_width=1.5,
                  annotation_text=lbl,annotation_font_color=color,row=row,col=col)

def plot_chain(df, spot):
    rng=spot*0.05; sub=df[df["strikePrice"].between(spot-rng,spot+rng)]
    if sub.empty: sub=df
    fig=make_subplots(2,2,subplot_titles=["CE vs PE Premium (higher = more expensive)","Open Interest (where big money bet)","IV Smile (costly=peak, cheap=trough)","Volume (today's trading activity)"])
    x=sub["strikePrice"]
    fig.add_trace(go.Bar(name="CE (Call)",x=x,y=sub["CE_LTP"],marker_color="#00ff88",opacity=.85),1,1)
    fig.add_trace(go.Bar(name="PE (Put)",x=x,y=sub["PE_LTP"],marker_color="#ff3b5c",opacity=.85),1,1)
    fig.add_trace(go.Bar(name="CE OI",x=x,y=sub["CE_OI"],marker_color="#00e5ff",opacity=.8),1,2)
    fig.add_trace(go.Bar(name="PE OI",x=x,y=sub["PE_OI"],marker_color="#ff9500",opacity=.8),1,2)
    fig.add_trace(go.Scatter(name="CE IV%",x=x,y=sub["CE_IV"],mode="lines+markers",line=dict(color="#00ff88",width=2)),2,1)
    fig.add_trace(go.Scatter(name="PE IV%",x=x,y=sub["PE_IV"],mode="lines+markers",line=dict(color="#ff3b5c",width=2)),2,1)
    fig.add_trace(go.Bar(name="CE Vol",x=x,y=sub["CE_volume"],marker_color="rgba(0,255,136,0.4)"),2,2)
    fig.add_trace(go.Bar(name="PE Vol",x=x,y=sub["PE_volume"],marker_color="rgba(255,59,92,0.4)"),2,2)
    for r in [1,2]:
        for c in [1,2]: vl(fig,spot,"Spot",row=r,col=c)
    fig.update_layout(template=DARK,height=520,barmode="group",title="Option Chain View",margin=dict(t=50,b=10))
    return fig

def plot_oi(df,spot,sig):
    rng=spot*0.055; sub=df[df["strikePrice"].between(spot-rng,spot+rng)]
    if sub.empty: sub=df
    sub=sub.copy()
    sub["CE_pct"]=(sub["CE_changeOI"]/(sub["CE_OI"]-sub["CE_changeOI"]).clip(lower=1)*100).fillna(0)
    sub["PE_pct"]=(sub["PE_changeOI"]/(sub["PE_OI"]-sub["PE_changeOI"]).clip(lower=1)*100).fillna(0)
    fig=make_subplots(3,1,subplot_titles=["Total OI — where big money has parked bets","Change in OI today — fresh money entering/exiting","% Change — which strike saw biggest action (%)"])
    x=sub["strikePrice"]
    fig.add_trace(go.Bar(name="CE OI",x=x,y=sub["CE_OI"],marker_color="#00e5ff",opacity=.8),1,1)
    fig.add_trace(go.Bar(name="PE OI",x=x,y=sub["PE_OI"],marker_color="#ff9500",opacity=.8),1,1)
    fig.add_trace(go.Bar(name="ΔCE OI",x=x,y=sub["CE_changeOI"],
        marker_color=["#00ff88" if v>=0 else "#ff3b5c" for v in sub["CE_changeOI"]]),2,1)
    fig.add_trace(go.Bar(name="ΔPE OI",x=x,y=sub["PE_changeOI"],
        marker_color=["#ff9500" if v>=0 else "#8888ff" for v in sub["PE_changeOI"]]),2,1)
    fig.add_trace(go.Bar(name="CE%",x=x,y=sub["CE_pct"],
        marker_color=["rgba(0,255,136,0.47)" if v>=0 else "rgba(255,59,92,0.47)" for v in sub["CE_pct"]]),3,1)
    fig.add_trace(go.Bar(name="PE%",x=x,y=sub["PE_pct"],
        marker_color=["rgba(255,149,0,0.47)" if v>=0 else "rgba(136,136,255,0.47)" for v in sub["PE_pct"]]),3,1)
    for row in [1,2,3]:
        vl(fig,spot,"Spot",row=row)
        if sig["resistance"]: vl(fig,sig["resistance"],"Resistance",color="#ff3b5c",dash="dot",row=row,col=1)
        if sig["support"]:    vl(fig,sig["support"],"Support",color="#00ff88",dash="dot",row=row,col=1)
    fig.update_layout(template=DARK,height=620,barmode="group",title="OI Analysis",margin=dict(t=50,b=10))
    return fig

def plot_greeks_chart(df,spot):
    rng=spot*0.04; sub=df[df["strikePrice"].between(spot-rng,spot+rng)]
    if sub.empty: sub=df
    fig=make_subplots(2,2,subplot_titles=["Delta (directional bet size)","Gamma (how fast delta changes)","Theta (daily money you lose to time)","Vega (how much you gain if IV rises)"])
    x=sub["strikePrice"]
    for (r,c,cc,pc) in [(1,1,"CE_delta","PE_delta"),(1,2,"CE_gamma","PE_gamma"),
                        (2,1,"CE_theta","PE_theta"),(2,2,"CE_vega","PE_vega")]:
        for col,color in [(cc,"#00ff88"),(pc,"#ff3b5c")]:
            if col in sub.columns:
                fig.add_trace(go.Scatter(name=col,x=x,y=sub[col],mode="lines",line=dict(color=color,width=2)),r,c)
        vl(fig,spot,row=r,col=c)
    fig.update_layout(template=DARK,height=520,title="Greeks",margin=dict(t=50,b=10))
    return fig

def plot_straddle(hist_sym, straddle_now):
    hist=yf_history(hist_sym,"45d")
    if hist.empty: return None, None
    close=hist["Close"].squeeze().astype(float)
    rv=close.pct_change().rolling(7).std()*np.sqrt(252)
    est=(close*rv*np.sqrt(7/252)*0.8).dropna()
    if est.empty: return None,None
    p25,p50,p75=est.quantile([.25,.5,.75])
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=est.index,y=est,name="Historical straddle estimate",line=dict(color="#00e5ff",width=2)))
    if straddle_now>0:
        fig.add_hline(y=straddle_now,line_color="yellow",line_dash="dash",
                      annotation_text=f"Today: {straddle_now:.1f}",annotation_font_color="yellow")
    fig.add_hline(y=p25,line_color="#00ff88",line_dash="dot",annotation_text=f"Cheap zone: {p25:.1f}")
    fig.add_hline(y=p75,line_color="#ff3b5c",line_dash="dot",annotation_text=f"Expensive zone: {p75:.1f}")
    fig.add_hline(y=p50,line_color="#ff9500",line_dash="dot",annotation_text=f"Average: {p50:.1f}")
    fig.update_layout(template=DARK,height=320,title="Is the straddle cheap or expensive vs last 45 days?")
    if straddle_now>0:
        if straddle_now<p25:   v=("✅ CHEAP — Options are priced below average. Great time to buy!","#00ff88")
        elif straddle_now>p75: v=("⚠️ EXPENSIVE — Options overpriced vs history. Avoid buying now.","#ff3b5c")
        else:                   v=("🟡 FAIR VALUE — Options at normal price. Be selective.","#ff9500")
    else: v=None
    return fig,v


# ─────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────
def main():
    st.markdown("<h1 style='font-size:24px;letter-spacing:2px'>⚡ PRO OPTIONS INTELLIGENCE</h1>",unsafe_allow_html=True)
    st.markdown("<p style='color:#5a7a9a;margin-top:-10px'>Indian & Global Options · Live Chain · Greeks · OI · Signals · Backtest · Plain English explanations throughout</p>",unsafe_allow_html=True)
    st.divider()

    with st.sidebar:
        st.markdown("### ⚙️ Choose Market")

        src_type = st.selectbox("Market", [
            "🇮🇳 NSE India (NIFTY / BANKNIFTY)",
            "🌐 Global (US Stocks / Other)",
            "📊 Spot Only (BTC / Gold / FX — no options)",
        ])

        fetch_sym=""; hist_sym=""; is_nse=False; has_opts=True

        if "NSE India" in src_type:
            is_nse=True
            nse_sym=st.selectbox("Index / Stock",NSE_SYMS+["Custom NSE Stock"])
            if nse_sym=="Custom NSE Stock":
                nse_sym=st.text_input("NSE symbol (e.g. RELIANCE)","RELIANCE").upper().strip()
            fetch_sym=nse_sym; hist_sym=NSE_TO_YF.get(nse_sym,nse_sym+".NS")
            st.markdown("""
<div style='background:#0a1929;border:1px solid #1e3050;border-radius:8px;padding:10px;font-size:12px;color:#5a7a9a;margin-top:4px'>
<b style='color:#ff9500'>⚠️ NSE requires Indian IP</b><br>
If you're in India and it fails, click <b>Clear Cache</b> and try again.<br>
Outside India? Use a VPN set to India.
</div>""",unsafe_allow_html=True)

        elif "Global" in src_type:
            choice=st.selectbox("Instrument",list(YF_OPTIONS_SYMBOLS.keys()))
            if choice=="Custom":
                fetch_sym=st.text_input("yFinance ticker","AAPL").upper().strip()
            else:
                fetch_sym=YF_OPTIONS_SYMBOLS[choice]
            hist_sym=fetch_sym
            st.success("✅ These instruments have listed options on yFinance")

        else:
            choice=st.selectbox("Instrument",list(YF_SPOT_ONLY.keys()))
            if choice=="Custom":
                fetch_sym=st.text_input("yFinance ticker","BTC-USD").upper().strip()
            else:
                fetch_sym=YF_SPOT_ONLY[choice]
            hist_sym=fetch_sym; has_opts=False
            st.warning("No options for BTC/Gold/FX. Spot + volatility analysis only.")

        st.divider()
        st.markdown("### 💰 Risk Settings")
        capital=st.number_input("Your capital (₹)",50_000,10_000_000,100_000,10_000)
        pos_pct=st.slider("% to risk per trade",2,15,5)
        sl_pct =st.slider("Stop loss % (on premium)",20,50,30)
        tgt_pct=st.slider("Target % (on premium)",30,200,60)
        st.divider()
        st.button("🔄 FETCH LIVE DATA",type="primary",use_container_width=True)
        if st.button("🗑️ Clear Cache & Retry",use_container_width=True):
            st.cache_data.clear(); _T["last"]=0.0; st.rerun()
        auto_ref=st.checkbox("Auto-refresh every 90s")
        st.caption("⚠️ For education only. Not financial advice.")

    # ── FETCH ──────────────────────────────────────────────────────────────────
    df_exp=pd.DataFrame(); spot=0.0; expiries=[]; sel_expiry=""
    status_ph=st.empty()

    with st.spinner(f"Fetching {fetch_sym}…"):
        if is_nse:
            status_ph.info("📡 Connecting to NSE (takes ~3 seconds for cookie setup)…")
            df_raw,nse_spot,nse_exps=fetch_nse(fetch_sym)
            if df_raw is not None and nse_spot>0:
                spot=nse_spot; expiries=nse_exps
                with st.sidebar:
                    sel_expiry=st.selectbox("Expiry",expiries[:8]) if expiries else ""
                mask=df_raw["expiryDate"]==sel_expiry
                df_exp=df_raw[mask].copy() if mask.any() else df_raw.copy()
                status_ph.success(f"✅ NSE loaded — {len(df_exp)} strikes across {len(expiries)} expiries | Spot ₹{spot:,.2f}")
            else:
                status_ph.error("❌ NSE chain blocked. Fetching spot only via yFinance…")
                spot=yf_spot(hist_sym); has_opts=False
                if spot>0:
                    status_ph.warning(f"Spot ₹{spot:,.2f} from yFinance. NSE chain needs Indian IP. Try: Clear Cache → Retry, or use a VPN.")
                else:
                    status_ph.error("Cannot get any data. Check internet connection.")

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
                        df_exp=build_chain_df(calls,puts,sel_expiry)
                        status_ph.success(f"✅ {fetch_sym} loaded — {len(df_exp)} strikes | Spot {spot:,.2f}")
                    else:
                        status_ph.error("Chain fetch failed. Try again in 15 seconds."); has_opts=False
                else:
                    status_ph.warning(f"No options for {fetch_sym} right now (market may be closed)."); has_opts=False
            else:
                status_ph.error(f"Cannot get price for {fetch_sym}. Check the symbol.")

        else:
            spot=yf_spot(fetch_sym)
            if spot>0: status_ph.success(f"✅ {fetch_sym}: {spot:,.4f}")
            else: status_ph.error(f"Cannot get price for {fetch_sym}.")

    if spot==0:
        st.error("❌ No data. If NSE, try Clear Cache and retry. If outside India use a VPN.")
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

    # ── METRICS ──
    cols=st.columns(7)
    m=[("📍 Spot",f"{spot:,.2f}"),
       ("🎯 ATM",f"{sig['atm']:,.0f}" if has_chain else "N/A"),
       ("📊 PCR",f"{sig['pcr']:.3f}" if has_chain else "N/A"),
       ("💀 Max Pain",f"{sig['max_pain']:,.0f}" if has_chain else "N/A"),
       ("🌡 IV%",f"{sig['atm_iv']:.1f}%" if has_chain else "N/A"),
       ("↕ Skew",f"{sig['skew']:+.1f}%" if has_chain else "N/A"),
       ("♟ Straddle",f"{sig['straddle']:.2f}" if has_chain else "N/A")]
    for col,(lbl,val) in zip(cols,m): col.metric(lbl,val)

    # ── SIGNAL BANNER ──
    conf=sig["conf"]
    bc="#00ff88" if conf>=72 else "#ff9500" if conf>=58 else "#5a7a9a"
    def tag(s): return f'<span class="tag-b">{s}</span>' if "BUY" in s else f'<span class="tag-w">{s}</span>'

    if has_chain:
        st.markdown(f"""
<div class="sig-box">
  <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;align-items:center">
    <div>
      <div style="color:#5a7a9a;font-size:10px;letter-spacing:2px;margin-bottom:4px">SIGNAL</div>
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
            st.markdown('<div class="gamma-blast">⚡ <b style="color:#ff3b5c">GAMMA BLAST</b> — Huge OI sitting right at current price. When market breaks through this level, options will EXPLODE in value. Watch closely and enter immediately on breakout.</div>',unsafe_allow_html=True)
    else:
        st.info(f"📊 Spot only for **{fetch_sym}**: **{spot:,.4f}**. No options chain — backtest and volatility history still work below.")

    st.markdown("<br>",unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────────────
    tabs=st.tabs(["📊 Chain","🔢 Greeks","📈 OI Analysis","⚡ Live Trade","🔬 Backtest","📋 History","🧠 Analysis"])

    # ═══ TAB 1 — CHAIN ═══════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("""<div class="explain">
<b>What you're looking at:</b> The option chain shows every available Call (CE) and Put (PE) contract 
around the current market price. Each row is one strike price (e.g. 25000, 25100, 25200).
<br><br>
<b>What to look for:</b> The strike closest to the current Spot price is called <b>ATM (At The Money)</b> — 
this is usually where the most action is. Higher LTP = more expensive option. Higher OI = more people 
have bets at that level = stronger support/resistance.
</div>""",unsafe_allow_html=True)
        if not has_chain:
            st.warning("Options chain not available. Select NSE India (with Indian IP) or a Global instrument.")
        else:
            def_st=all_strikes[max(0,atm_pos-8):atm_pos+9]
            sel_st=st.multiselect("Show strikes",all_strikes,default=def_st)
            if not sel_st: sel_st=def_st
            show_c=[c for c in ["strikePrice","CE_LTP","CE_OI","CE_changeOI","CE_IV","CE_volume",
                                  "PE_LTP","PE_OI","PE_changeOI","PE_IV","PE_volume"] if c in df_exp.columns]
            st.dataframe(df_exp[df_exp["strikePrice"].isin(sel_st)][show_c].round(2),use_container_width=True,height=260)
            st.plotly_chart(plot_chain(df_exp,spot),use_container_width=True)
            st.markdown("#### Is the straddle (CE+PE) cheap or expensive right now?")
            st.markdown("""<div class="explain">
<b>What is a straddle?</b> It's the combined cost of buying both a Call AND a Put at the same strike. 
This tells you how much the market is "charging" for an expected move. 
<b>If straddle is cheap vs history → options are underpriced → great time to buy.</b>
</div>""",unsafe_allow_html=True)
            fs,fv=plot_straddle(hist_sym,sig["straddle"])
            if fs: st.plotly_chart(fs,use_container_width=True)
            if fv: st.markdown(f'<div style="background:{fv[1]}11;border:1px solid {fv[1]}44;border-radius:8px;padding:12px;color:{fv[1]};font-weight:600;font-size:15px">{fv[0]}</div>',unsafe_allow_html=True)

    # ═══ TAB 2 — GREEKS ══════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("""<div class="explain">
<b>Greeks in plain English:</b><br><br>
🟢 <b>Delta</b> = "If Nifty moves ₹100, how much does my option move?" 
Delta of 0.5 means your option gains ₹50 for every ₹100 move. Higher delta = more sensitive to price.<br><br>
🔵 <b>Gamma</b> = "How fast is Delta itself changing?" High Gamma near expiry = your option can gain value very quickly if price moves.<br><br>
🔴 <b>Theta</b> = "How much money do I lose just by waiting one day?" This is the enemy of option buyers. Every day that passes, you lose this much automatically.<br><br>
🟡 <b>Vega</b> = "How much do I gain if the market gets nervous?" If IV (fear/volatility) rises, your option gains this much. Buy before events, sell after.<br><br>
<b>Golden rule:</b> Buy when Vega > |Theta| and IV is below 25%.
</div>""",unsafe_allow_html=True)
        if not has_chain:
            st.warning("Greeks require options chain. Select NSE India or Global instruments with options.")
        else:
            st.plotly_chart(plot_greeks_chart(df_exp,spot),use_container_width=True)
            ai2=(df_exp["strikePrice"]-spot).abs().idxmin(); atm2=df_exp.loc[ai2]
            st.markdown(f"#### ATM Strike {atm2['strikePrice']:,.0f} — Your Greeks Right Now")
            g1,g2=st.columns(2)
            for col,px,label,color in [(g1,"CE","📗 CALL (buy if market going UP)","#00ff88"),
                                        (g2,"PE","📕 PUT (buy if market going DOWN)","#ff3b5c")]:
                with col:
                    st.markdown(f"<h5 style='color:{color}'>{label}</h5>",unsafe_allow_html=True)
                    cs=st.columns(3)
                    for i,(name,key,tip) in enumerate([
                        ("Delta",f"{px}_delta","Move per ₹100"),
                        ("Gamma",f"{px}_gamma","Delta acceleration"),
                        ("Theta",f"{px}_theta","Daily time decay"),
                        ("Vega",f"{px}_vega","Gain per 1% IV rise"),
                        ("IV%",f"{px}_IV","Current volatility"),
                    ]):
                        val=float(atm2.get(key,0))
                        fmt=f"{val:.2f}%" if "IV" in name else f"{val:.4f}"
                        cs[i%3].metric(name,fmt,tip)
            iv=sig["atm_iv"]
            ce_th=float(atm2.get("CE_theta",0)); ce_ve=float(atm2.get("CE_vega",0))
            tips=[]
            if iv>40: tips.append(("🔴 IV above 40%: Options too expensive. Time decay is destroying value faster than market moves can compensate. WAIT.","#ff3b5c"))
            elif iv<15: tips.append(("✅ IV below 15%: Cheap options! Even a small move in the right direction makes good profit.","#00ff88"))
            if abs(ce_th)>abs(ce_ve): tips.append(("⚠️ Theta (decay) > Vega (IV gain): You're losing more each day than you can gain from volatility. Prefer ITM options or closer strikes.","#ff9500"))
            else: tips.append(("✅ Vega > Theta: Good balance. You stand to gain more from market moves than you lose from time decay.","#00ff88"))
            for t,clr in tips:
                st.markdown(f'<div style="background:{clr}11;border-left:3px solid {clr};padding:12px;border-radius:0 6px 6px 0;margin:6px 0">{t}</div>',unsafe_allow_html=True)

    # ═══ TAB 3 — OI ══════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("""<div class="explain">
<b>Open Interest (OI) — where the big players put their money:</b><br><br>
🔵 <b>High CE OI at a strike</b> = Big sellers have written calls there = <b>RESISTANCE</b> (market struggles to go above this)<br>
🟡 <b>High PE OI at a strike</b> = Big sellers have written puts there = <b>SUPPORT</b> (market struggles to fall below this)<br><br>
📈 <b>Rising OI + Rising Price</b> = Fresh buying = BULLISH (long buildup)<br>
📉 <b>Rising OI + Falling Price</b> = Fresh shorting = BEARISH (short buildup)<br>
📉 <b>Falling OI + Rising Price</b> = Short covering = Temporary bounce (weak signal)<br><br>
<b>Max Pain</b> = The price where maximum number of option buyers lose money. Near expiry, market tends to drift here like a magnet.
</div>""",unsafe_allow_html=True)
        if not has_chain:
            st.warning("OI analysis requires an options chain.")
        else:
            st.plotly_chart(plot_oi(df_exp,spot,sig),use_container_width=True)
            m1,m2,m3,m4=st.columns(4)
            m1.metric("🔴 Resistance (CE wall)",f"{sig['resistance']:,.0f}","Market struggles above here")
            m2.metric("🟢 Support (PE wall)",f"{sig['support']:,.0f}","Market struggles below here")
            m3.metric("🎯 Max Pain",f"{sig['max_pain']:,.0f}","Expiry magnet")
            m4.metric("📊 PCR",f"{sig['pcr']:.4f}",">1 bearish, <1 bullish (contrarian)")
            b1,b2=st.columns(2)
            with b1:
                st.markdown("**🔴 Top CE OI strikes — these are your RESISTANCE levels**")
                ct=df_exp.nlargest(5,"CE_OI")[["strikePrice","CE_OI","CE_changeOI","CE_LTP"]].copy()
                ct["Meaning"]=ct["CE_changeOI"].apply(lambda x:"🔴 Fresh resistance building" if x>=0 else "🟡 Resistance weakening")
                st.dataframe(ct,use_container_width=True)
            with b2:
                st.markdown("**🟢 Top PE OI strikes — these are your SUPPORT levels**")
                pt=df_exp.nlargest(5,"PE_OI")[["strikePrice","PE_OI","PE_changeOI","PE_LTP"]].copy()
                pt["Meaning"]=pt["PE_changeOI"].apply(lambda x:"🟢 Fresh support building" if x>=0 else "🟡 Support weakening")
                st.dataframe(pt,use_container_width=True)

    # ═══ TAB 4 — LIVE TRADE ══════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("""<div class="explain">
<b>How to use this tab:</b> This is your paper trading desk. No real money — just practice.<br>
1. Pick CE (Call) if you think market will go UP, PE (Put) if you think it will go DOWN<br>
2. Select your strike — beginners use ATM (nearest to current price)<br>
3. SL (Stop Loss) = if option price falls this much, EXIT immediately. Protects you from big losses.<br>
4. Target = your profit goal. When option reaches this price, take profit and exit.<br>
5. R:R = Risk vs Reward. 1:2 means you risk ₹1 to make ₹2. Never trade below 1:1.5
</div>""",unsafe_allow_html=True)
        if not has_chain:
            st.warning("Live trading requires an options chain.")
        else:
            la,lb=st.columns([3,2])
            with la:
                st.markdown("#### 🚨 New Paper Trade")
                t1,t2=st.columns(2)
                side=t1.selectbox("Direction",["CE — Buying Call (market UP)","PE — Buying Put (market DOWN)"])
                px="CE" if "CE" in side else "PE"
                t_strike=t1.selectbox("Strike",all_strikes,index=atm_pos)
                lots=t2.number_input("Lots",1,100,1)
                l_sl=t2.slider("Stop Loss %",20,50,sl_pct,key="l_sl")
                l_tgt=t2.slider("Target %",30,200,tgt_pct,key="l_tgt")
                row_s=df_exp[df_exp["strikePrice"]==t_strike]
                opt_px=float(row_s[f"{px}_LTP"].values[0]) if not row_s.empty and f"{px}_LTP" in row_s.columns else 0
                if opt_px>0:
                    sl_a=round(opt_px*(1-l_sl/100),2); tgt_a=round(opt_px*(1+l_tgt/100),2)
                    risk=(opt_px-sl_a)*lots; rew=(tgt_a-opt_px)*lots
                    rr=round(rew/risk,2) if risk>0 else 0
                    pm1,pm2,pm3,pm4=st.columns(4)
                    pm1.metric("Entry ₹",f"{opt_px:.2f}","Buy here")
                    pm2.metric("Stop Loss ₹",f"{sl_a:.2f}",f"Exit if falls {l_sl}%")
                    pm3.metric("Target ₹",f"{tgt_a:.2f}",f"Exit at +{l_tgt}%")
                    pm4.metric("R:R",f"1:{rr}","Reward vs Risk")
                    if rr<1.5: st.warning("⚠️ R:R below 1:1.5 — risky trade. Consider adjusting target or SL.")
                else:
                    sl_a=tgt_a=0; st.warning("Price is 0 — market closed or strike too far out.")
                st.markdown(f'<div style="background:#0a1929;border:1px solid #1e3050;border-radius:8px;padding:12px"><b style="color:#00e5ff">{sig["rec"]}</b> | Confidence: <b>{sig["conf"]}%</b><br><span style="color:#5a7a9a;font-size:12px">{" · ".join(sig["reasons"][:2])}</span></div>',unsafe_allow_html=True)
                if st.button("📈 ENTER PAPER TRADE",type="primary",use_container_width=True):
                    if opt_px>0:
                        t={"id":len(st.session_state.trades)+1,"sym":fetch_sym,"expiry":sel_expiry,
                           "strike":t_strike,"side":px,"entry":opt_px,"sl":sl_a,"target":tgt_a,
                           "lots":lots,"conf":sig["conf"],"rec":sig["rec"],
                           "time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "status":"OPEN","exit":None,"pnl":None,"exit_time":None}
                        st.session_state.active.append(t); st.session_state.trades.append(t)
                        st.success(f"✅ Trade entered: {fetch_sym} {t_strike} {px} @ ₹{opt_px:.2f}")
                    else: st.error("Cannot enter — option price is 0.")
            with lb:
                st.markdown("#### 📊 Open Positions")
                if not [t for t in st.session_state.active if t["status"]=="OPEN"]: st.info("No open trades yet.")
                for i,t in enumerate(st.session_state.active):
                    if t["status"]!="OPEN": continue
                    r=df_exp[df_exp["strikePrice"]==t["strike"]]
                    col_ltp=f"{t['side']}_LTP"
                    curr=float(r[col_ltp].values[0]) if not r.empty and col_ltp in r.columns else t["entry"]
                    pnl=round((curr-t["entry"])*t["lots"],2); pp=round((curr-t["entry"])/t["entry"]*100,2) if t["entry"]>0 else 0
                    clr="#00ff88" if pnl>=0 else "#ff3b5c"; cls="tc-w" if pnl>0 else "tc-l" if pnl<0 else "tc-o"
                    # Check SL/target hit
                    sl_hit = curr<=t["sl"] if t["sl"]>0 else False
                    tgt_hit = curr>=t["target"] if t["target"]>0 else False
                    warn=""
                    if sl_hit: warn="<b style='color:#ff3b5c'>⚠️ STOP LOSS HIT — EXIT NOW</b><br>"
                    if tgt_hit: warn="<b style='color:#00ff88'>🎯 TARGET HIT — BOOK PROFIT</b><br>"
                    st.markdown(f"""<div class="tc {cls}">{warn}<b>{t['sym']} {t['strike']} {t['side']}</b><br>
Entry ₹{t['entry']:.2f} → Now ₹{curr:.2f} | SL ₹{t['sl']:.2f} | Tgt ₹{t['target']:.2f}<br>
<b style='color:{clr}'>P&L: ₹{pnl:+.2f} ({pp:+.1f}%)</b></div>""",unsafe_allow_html=True)
                    if st.button(f"Exit Trade #{t['id']}",key=f"ex_{i}_{t['id']}"):
                        for j,x in enumerate(st.session_state.active):
                            if x["id"]==t["id"]:
                                st.session_state.active[j].update({"status":"CLOSED","exit":curr,"pnl":pnl,"exit_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                                for h in st.session_state.trades:
                                    if h["id"]==t["id"]: h.update(st.session_state.active[j])
                                break
                        st.rerun()

    # ═══ TAB 5 — BACKTEST ════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("""<div class="explain">
<b>How the backtest works — in plain English:</b><br><br>
Think of it as going back in time and testing "what would have happened if I used this strategy every day?"<br><br>
<b>Step 1:</b> We look at each day's closing price over the past N days (e.g. 60 days = 60 data points)<br>
<b>Step 2:</b> For each day, we check if our signal conditions are met (PCR, IV, trend all line up)<br>
<b>Step 3:</b> If conditions are met (confidence ≥ 72%), we pretend to buy an ATM option<br>
<b>Step 4:</b> We check the NEXT day's price to see if we won or lost<br>
<b>Step 5:</b> We subtract 1 day of theta (time decay) from every trade because options lose value each day<br>
<b>Step 6:</b> We compound profits/losses into an equity curve<br><br>
<b>What the numbers mean:</b><br>
— <b>Win Rate %</b>: Out of 10 trades, how many were winners? (60%+ is good)<br>
— <b>R:R</b>: Average win ÷ average loss. Above 1.5 means wins are bigger than losses<br>
— <b>Max Drawdown</b>: Worst losing streak — how deep did equity fall from peak? (keep below 20%)<br>
— <b>Return %</b>: Total profit/loss on your starting capital<br><br>
⚠️ <b>Important:</b> Past results don't guarantee future performance. This is educational only.
</div>""",unsafe_allow_html=True)
        ba,bb,bc_=st.columns(3)
        bt_look=ba.slider("Look back how many days?",20,120,60)
        bt_cap=bb.number_input("Starting capital ₹",50_000,5_000_000,int(capital),10_000,key="bt_cap")
        bt_pos=bc_.slider("% of capital per trade",2,20,pos_pct,key="bt_pos")
        st.markdown(f"""<div class="explain">
With <b>₹{int(bt_cap):,}</b> capital and <b>{bt_pos}%</b> per trade: each trade risks <b>₹{int(bt_cap*bt_pos/100):,}</b>.
That leaves <b>₹{int(bt_cap*(1-bt_pos/100)):,}</b> safe even if the first trade is a total loss.
</div>""",unsafe_allow_html=True)
        if st.button("🔬 RUN BACKTEST",type="primary",use_container_width=True):
            with st.spinner("Running backtest…"):
                tdf,stats=run_backtest(hist_sym,bt_look,float(bt_cap),float(bt_pos))
                st.session_state.bt_result=(tdf,stats)
        if st.session_state.bt_result:
            tdf,stats=st.session_state.bt_result
            if tdf is None:
                st.error(f"Backtest failed: {stats}")
            else:
                k1,k2,k3,k4,k5,k6,k7=st.columns(7)
                k1.metric("Total Trades",stats["total"],"trades taken")
                k2.metric("Win Rate",f"{stats['wr']}%","of trades won")
                k3.metric("Total P&L",f"₹{stats['total_pnl']:+,.0f}","overall profit/loss")
                k4.metric("Avg Win",f"₹{stats['aw']:,.0f}","per winning trade")
                k5.metric("Avg Loss",f"₹{stats['al']:,.0f}","per losing trade")
                k6.metric("R:R",stats['rr'],"win size / loss size")
                k7.metric("Max Drawdown",f"{stats['mdd']}%","worst losing stretch")
                rc="#00ff88" if stats["ret%"]>=0 else "#ff3b5c"
                st.markdown(f'<div style="text-align:center;font-size:28px;color:{rc};font-family:Space Mono;margin:12px 0;padding:14px;background:#0a1929;border-radius:10px">Final Capital: ₹{stats["final"]:,.0f} &nbsp;|&nbsp; Return: {stats["ret%"]:+.2f}%</div>',unsafe_allow_html=True)

                # Plain English verdict
                wr=stats["wr"]; rr=stats["rr"]; mdd=stats["mdd"]
                if wr>=60 and rr>=1.5 and mdd<20: verdict=("✅ Strong strategy — win rate good, wins bigger than losses, drawdown controlled. Suitable to trade with discipline.","#00ff88")
                elif wr>=50 and rr>=1.2: verdict=("🟡 Decent strategy — profitable but needs tighter discipline. Watch position sizing and stick to stop losses.","#ff9500")
                else: verdict=("🔴 Weak results — either win rate too low or losses too big vs wins. Don't trade real money based on this. Optimize the signals first.","#ff3b5c")
                st.markdown(f'<div style="background:{verdict[1]}11;border-left:3px solid {verdict[1]};padding:14px;border-radius:0 8px 8px 0;margin:8px 0;font-size:14px">{verdict[0]}</div>',unsafe_allow_html=True)

                fig_eq=go.Figure(go.Scatter(y=stats["curve"],mode="lines",
                    line=dict(color="#00e5ff",width=2.5),fill="tozeroy",fillcolor="rgba(0,229,255,0.05)"))
                fig_eq.add_hline(y=bt_cap,line_dash="dash",line_color="#ff9500",annotation_text=f"Start: ₹{int(bt_cap):,}")
                fig_eq.update_layout(template=DARK,height=320,title="Equity Curve — watch for smooth rise vs sharp drops",yaxis_title="Capital ₹")
                st.plotly_chart(fig_eq,use_container_width=True)
                w=tdf[tdf["P&L (₹)"]>0]["P&L (₹)"]; l=tdf[tdf["P&L (₹)"]<=0]["P&L (₹)"]
                fig_d=go.Figure()
                fig_d.add_trace(go.Histogram(x=w,name="Wins",marker_color="rgba(0,255,136,0.47)",nbinsx=20))
                fig_d.add_trace(go.Histogram(x=l,name="Losses",marker_color="rgba(255,59,92,0.47)",nbinsx=20))
                fig_d.update_layout(template=DARK,height=240,title="Distribution of Wins vs Losses — wins should cluster farther right than losses",barmode="overlay")
                st.plotly_chart(fig_d,use_container_width=True)
                st.markdown("#### Every trade the backtest took:")
                st.dataframe(tdf,use_container_width=True,height=280)

    # ═══ TAB 6 — HISTORY ═════════════════════════════════════════════════════
    with tabs[5]:
        if not st.session_state.trades: st.info("No paper trades yet. Go to Live Trade tab and enter trades.")
        else:
            all_t=pd.DataFrame(st.session_state.trades)
            closed=all_t[all_t["status"]=="CLOSED"].copy()
            if not closed.empty:
                closed["pnl"]=pd.to_numeric(closed["pnl"],errors="coerce").fillna(0)
                tot=closed["pnl"].sum(); wr=(closed["pnl"]>0).mean()*100
                h1,h2,h3,h4=st.columns(4)
                h1.metric("Total Trades",len(all_t)); h2.metric("Closed",len(closed))
                h3.metric("Win Rate",f"{wr:.1f}%"); h4.metric("Net P&L",f"₹{tot:+,.2f}")
                fig_p=go.Figure(go.Bar(y=closed["pnl"].values,
                    marker_color=["#00ff88" if p>0 else "#ff3b5c" for p in closed["pnl"]]))
                fig_p.update_layout(template=DARK,height=240,title="Per-Trade P&L — green=win, red=loss")
                st.plotly_chart(fig_p,use_container_width=True)
            dc=[c for c in ["id","time","sym","strike","side","entry","exit","lots","pnl","status","rec","conf"] if c in all_t.columns]
            st.dataframe(all_t[dc],use_container_width=True)
            st.download_button("📥 Download as CSV",all_t.to_csv(index=False),"trades.csv","text/csv",use_container_width=True)
            if st.button("🗑️ Clear All Trades",use_container_width=True):
                st.session_state.trades=[]; st.session_state.active=[]; st.rerun()

    # ═══ TAB 7 — ANALYSIS ════════════════════════════════════════════════════
    with tabs[6]:
        iv=sig["atm_iv"]; pcr=sig["pcr"]; mp=sig["max_pain"]
        res=sig["resistance"]; sup=sig["support"]
        mp_pct=(mp-spot)/spot*100 if spot>0 else 0

        st.markdown(f"""
<div class="card">
<h4 style='color:#00e5ff;margin-top:0'>📖 Full Market Reading — {fetch_sym} @ {spot:,.2f}</h4>
<p>Here's what every indicator is saying right now, in plain English:</p>
</div>""",unsafe_allow_html=True)

        if has_chain:
            # PCR explanation
            pcr_color="#00ff88" if pcr>1.2 else "#ff3b5c" if pcr<0.8 else "#ff9500"
            pcr_text=("Put-Call Ratio is HIGH. More puts than calls. This sounds bearish, but it's actually a CONTRARIAN BULLISH signal. When everyone buys insurance (puts), the market rarely falls — big institutions are protecting their longs, not betting on a crash. The market often goes UP from here." if pcr>1.2
                     else "Put-Call Ratio is LOW. More calls than puts. Sounds bullish, but it's CONTRARIAN BEARISH. When everyone is optimistic and buying calls, the market is often near a short-term top. The market may go DOWN from here." if pcr<0.8
                     else "Put-Call Ratio is BALANCED. No strong signal from options flow. Market is undecided — wait for a clearer setup before entering.")
            st.markdown(f"""<div class="explain"><b>📊 PCR = {pcr:.4f}</b><br>{pcr_text}</div>""",unsafe_allow_html=True)

            # IV explanation
            iv_color="#00ff88" if iv<18 else "#ff3b5c" if iv>40 else "#ff9500"
            iv_text=("Options are CHEAP right now. Think of IV like the price of petrol — when it's low, fill up the tank. Buying options when IV is low means you pay less upfront AND you profit if IV rises later (vega gains)." if iv<18
                    else "Options are VERY EXPENSIVE. The market is scared — everyone is rushing to buy protection, driving up prices. This is the worst time to buy. If you buy here and the market doesn't move immediately, your option will LOSE value even if direction is right." if iv>40
                    else f"IV at {iv:.0f}% is moderate. Not too cheap, not too expensive. Be selective — only buy on strong signals.")
            st.markdown(f"""<div class="explain"><b>🌡 IV = {iv:.1f}%</b><br>{iv_text}</div>""",unsafe_allow_html=True)

            # Max pain
            mp_text=(f"Max pain is {abs(mp_pct):.1f}% ABOVE current price at {mp:,.0f}. Near expiry, the market has a strong pull upward toward {mp:,.0f}. This favors CALL buyers. Be careful of puts." if mp_pct>2
                    else f"Max pain is {abs(mp_pct):.1f}% BELOW current price at {mp:,.0f}. Near expiry, the market has a strong pull downward toward {mp:,.0f}. This favors PUT buyers. Be careful of calls." if mp_pct<-2
                    else f"Max pain {mp:,.0f} is close to current price. No strong gravity pull right now. Max pain matters most in the last 2-3 days before expiry.")
            st.markdown(f"""<div class="explain"><b>🎯 Max Pain = {mp:,.0f}</b><br>{mp_text}</div>""",unsafe_allow_html=True)

            # Support/Resistance
            st.markdown(f"""<div class="explain"><b>🔴 Resistance = {res:,.0f} | 🟢 Support = {sup:,.0f}</b><br>
The highest CE OI is at <b>{res:,.0f}</b> — this is a CEILING. Big call writers have collected premiums here and will defend this level aggressively. If price breaks and sustains above {res:,.0f}, those writers will panic-buy, causing a sharp rally.<br><br>
The highest PE OI is at <b>{sup:,.0f}</b> — this is a FLOOR. Big put writers have collected premiums here. If price breaks below {sup:,.0f}, those writers panic-buy puts, causing a sharp fall.<br><br>
<b>Trade the breakout, not the range.</b> Buy options only when price clearly breaks through one of these walls with volume.
</div>""",unsafe_allow_html=True)

            # Scenario table
            st.markdown("#### 🗺 Three Scenarios — What to Do in Each")
            s1,s2,s3=st.columns(3)
            for col,title,trigger,action,why,color in [
                (s1,"🟢 BULLISH",f"Price breaks & holds above {res:,.0f}",
                 f"Buy {int(sig['atm'])} CE\nTarget: +60%\nSL: -30%",
                 f"The resistance wall breaks → call writers panic-buy to cover → premium explodes.","#00ff88"),
                (s2,"⚪ RANGE BOUND",f"Price stays between {sup:,.0f} and {res:,.0f}",
                 "DO NOT BUY OPTIONS.\nSell straddles if you're advanced.\nCash is best here.",
                 "Theta (time decay) destroys option buyers in a sideways market. Every day you hold loses money.","#5a7a9a"),
                (s3,"🔴 BEARISH",f"Price breaks & holds below {sup:,.0f}",
                 f"Buy {int(sig['atm'])} PE\nTarget: +60%\nSL: -30%",
                 f"The support wall breaks → put writers panic-buy to cover → premium explodes.","#ff3b5c"),
            ]:
                with col:
                    st.markdown(f"""
<div style='background:#0a1929;border-top:4px solid {color};border:1px solid #1e3050;border-radius:10px;padding:16px;height:100%'>
<h5 style='color:{color};margin-top:0'>{title}</h5>
<b style='color:#5a7a9a'>Trigger:</b><br>{trigger}<br><br>
<b style='color:#5a7a9a'>Action:</b><br><pre style='color:#c8d6e5;font-size:12px;background:transparent;margin:4px 0'>{action}</pre>
<br><b style='color:#5a7a9a'>Why:</b><br><span style='font-size:12px;color:#8899aa'>{why}</span>
</div>""",unsafe_allow_html=True)

        # Buyer rules
        st.markdown("""
<div class="card" style='margin-top:16px'>
<h4 style='color:#ff9500;margin-top:0'>📜 Option Buyer's Golden Rules — Print This Out</h4>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:14px;line-height:1.8'>
<div>
1. <b>Never risk more than 5% per trade.</b> One bad trade should never hurt your account badly.<br>
2. <b>Only buy when IV is below 30%.</b> Expensive options can lose money even if direction is right.<br>
3. <b>Always set your SL before entering.</b> Decide your exit BEFORE emotions kick in.<br>
4. <b>Exit 5+ days before expiry.</b> Near expiry, theta decay accelerates — options lose value fast.<br>
</div>
<div>
5. <b>Book 50% profit at first target.</b> Then trail the rest with a mental SL at your entry price.<br>
6. <b>Never average a losing trade.</b> If your SL is hit, exit. Don't double down hoping it reverses.<br>
7. <b>Avoid buying on event days (RBI, earnings, elections).</b> IV spikes before → crashes after → you lose even if direction was right.<br>
8. <b>No trade is also a trade.</b> Cash is a position. Waiting for the right setup is a skill.
</div>
</div>
</div>""",unsafe_allow_html=True)
        st.caption(f"Last updated: {datetime.now().strftime('%d-%b-%Y %H:%M:%S')} | Educational only — not financial advice")

    if auto_ref:
        time.sleep(90); st.cache_data.clear(); _T["last"]=0.0; st.rerun()

if __name__=="__main__":
    main()
