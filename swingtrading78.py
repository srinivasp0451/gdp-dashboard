"""
Pro Options Dashboard v7 — India (NSE + BSE/SENSEX) + Global
=============================================================
Data sources:
  NSE  : nsepython → NSE API (3 routes) → Opstra → CSV upload
  BSE  : BSE API → paste-from-website fallback
  YF   : global options (SPY/QQQ/AAPL …)
  Spot : BTC/Gold/FX

Key improvements in v7:
  • BSE/SENSEX option chain (live + paste)
  • Specific strike recommendations ("Buy 23100 CE" not just "Buy Call")
  • Why each indicator is saying what it says — per-indicator verdict
  • OI change narrative with plain-English summary
  • Dynamic straddle chart (from actual chain, not hardcoded)
  • Text explanation under every plot
  • Confidence breakdown table
"""

import streamlit as st
import pandas as pd
import numpy as np
import io, re, csv as _csv, time, random, warnings, datetime as dt
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import brentq

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

warnings.filterwarnings("ignore")

# ─── rate-limit guard ─────────────────────────────────────────────────
_T = {"last": 0.0}
def _wait(g=1.2):
    d = time.time() - _T["last"]
    if d < g: time.sleep(g - d + random.uniform(0.05, 0.2))
    _T["last"] = time.time()

# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG + CSS
# ═══════════════════════════════════════════════════════════════════════
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
h1,h2,h3,h4{font-family:'Space Mono',monospace!important}
h1,h2,h3{color:var(--acc)!important}
h4{color:#c8d6e5!important}
.stButton>button{background:transparent!important;border:1px solid var(--acc)!important;color:var(--acc)!important;font-family:'Space Mono',monospace!important;border-radius:8px!important;font-size:12px!important}
.stButton>button:hover{background:var(--acc)!important;color:#000!important}
.explain{background:var(--s2);border-left:3px solid var(--acc);border-radius:0 8px 8px 0;padding:14px 18px;margin:10px 0;font-size:14px;line-height:1.7}
.explain b{color:var(--acc)}
.plot-explain{background:#0a1520;border:1px solid #1a3050;border-radius:8px;padding:14px 18px;margin:6px 0 14px 0;font-size:13px;line-height:1.75;color:#9ab5cc}
.sig-box{background:var(--s1);border:1px solid var(--brd);border-radius:12px;padding:18px 22px;margin:10px 0;position:relative;overflow:hidden}
.sig-box::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--acc),var(--grn))}
.ind-row{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid #1a2a3a;font-size:13px}
.ind-row:last-child{border-bottom:none}
.ind-val{font-family:'Space Mono',monospace;font-size:13px}
.tag-b{background:#002a15;color:#00ff88;border:1px solid #00aa44;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;display:inline-block;margin:3px 2px}
.tag-w{background:#2a1a00;color:#ff9500;border:1px solid #aa5500;padding:4px 12px;border-radius:20px;font-size:12px;display:inline-block;margin:3px 2px}
.alert{background:#1a0a00;border-left:3px solid var(--ora);border-radius:0 6px 6px 0;padding:10px 14px;margin:4px 0;font-size:13px}
.gamma-blast{background:#1a000a;border:2px solid #ff3b5c;border-radius:10px;padding:14px;animation:pulse 2s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(255,59,92,.5)}70%{box-shadow:0 0 0 10px rgba(255,59,92,0)}100%{box-shadow:0 0 0 0 rgba(255,59,92,0)}}
.tc{background:var(--s2);border:1px solid var(--brd);border-radius:10px;padding:14px;margin:6px 0}
.tc-o{border-left:3px solid var(--acc)}.tc-w{border-left:3px solid var(--grn)}.tc-l{border-left:3px solid var(--red)}
.card{background:var(--s1);border:1px solid var(--brd);border-radius:12px;padding:18px;margin:8px 0}
.oi-sum{background:#0a1929;border:1px solid var(--brd);border-radius:10px;padding:14px 18px;margin:8px 0;font-size:13px;line-height:1.85}
.oi-sum b{color:var(--acc)}
.src{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-family:'Space Mono',monospace;font-weight:700;letter-spacing:1px}
.src-a{background:#003a1a;color:#00ff88;border:1px solid #00aa44}
.src-b{background:#001a3a;color:#00e5ff;border:1px solid #0066aa}
.src-c{background:#2a1500;color:#ff9500;border:1px solid #884400}
.src-d{background:#1a001a;color:#cc88ff;border:1px solid #664488}
hr{border-color:var(--brd)!important}
</style>""", unsafe_allow_html=True)

RFR  = 0.065
DARK = "plotly_dark"

NSE_INDICES = ["NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY"]
BSE_INDICES = ["SENSEX","BANKEX"]
NSE_TO_YF   = {"NIFTY":"^NSEI","BANKNIFTY":"^NSEBANK","FINNIFTY":"NIFTY_FIN_SERVICE.NS",
               "MIDCPNIFTY":"^NSMIDCP","SENSEX":"^BSESN","BANKEX":"^BSESN"}
BSE_SCRIP_MAP = {"SENSEX": ("1","SENSEX"), "BANKEX": ("2","BANKEX")}

NSE_FNO = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","SBIN","HINDUNILVR","BHARTIARTL",
    "KOTAKBANK","ITC","LT","AXISBANK","BAJFINANCE","MARUTI","TITAN","WIPRO",
    "HCLTECH","TECHM","SUNPHARMA","ULTRACEMCO","ASIANPAINT","NESTLEIND","POWERGRID",
    "NTPC","ONGC","TATAMOTORS","TATASTEEL","JSWSTEEL","HINDALCO","COALINDIA",
    "ADANIENT","ADANIPORTS","SIEMENS","ABB","HAVELLS","GRASIM","INDUSINDBK",
    "BAJAJFINSV","BAJAJ-AUTO","HEROMOTOCO","EICHERMOT","M&M","TATACONSUM",
    "DIVISLAB","CIPLA","DRREDDY","APOLLOHOSP","LUPIN","BIOCON","TORNTPHARM",
    "PIDILITIND","BERGEPAINT","BHEL","SAIL","NMDC","VEDL","HINDZINC",
    "IOC","BPCL","HPCL","GAIL","PETRONET","CONCOR","IRCTC","DMART","TRENT",
    "NYKAA","ZOMATO","PAYTM","INDIGO","DLF","GODREJPROP","BALKRISIND",
    "APOLLOTYRE","MRF","BANKBARODA","PNB","CANBK","FEDERALBNK","IDFCFIRSTB",
    "MUTHOOTFIN","CHOLAFIN","LICHSGFIN","RECLTD","PFC","HDFCLIFE","SBILIFE",
    "ICICIPRULI","LTIM","LTTS","PERSISTENT","MPHASIS","COFORGE","KPITTECH",
    "SUZLON","TATAPOWER","NHPC","TORNTPOWER","MANAPPURAM","ANGELONE","DIXON","VOLTAS",
    "Custom symbol…"
]
YF_OPTS = {"SPY":"SPY","QQQ":"QQQ","AAPL":"AAPL","TSLA":"TSLA","NVDA":"NVDA","AMZN":"AMZN","Custom":"__custom__"}
YF_SPOT = {"BTC/USD":"BTC-USD","GOLD":"GC=F","SILVER":"SI=F","USD/INR":"USDINR=X","CRUDE OIL":"CL=F","Custom":"__custom__"}

for k,v in {"trades":[],"active":[],"bt_result":None}.items():
    if k not in st.session_state: st.session_state[k]=v


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════
def to_f(s):
    """Strip Indian-format commas and convert to float, return 0 for dashes/empty."""
    try:
        s = str(s).replace(",","").strip()
        if s in ("-","","-","–","—","nan","None","none","--","-"): return 0.0
        return float(s)
    except: return 0.0


# ═══════════════════════════════════════════════════════════════════════
# NSE CSV PARSER (actual NSE download: 2-row header, STRIKE col anchor)
# ═══════════════════════════════════════════════════════════════════════
def parse_nse_csv(raw_bytes, filename=""):
    debug = []
    try:
        content = raw_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        return pd.DataFrame(), [f"Cannot read file: {e}"]

    debug.append(f"File size: {len(raw_bytes)} bytes")
    expiry = "Uploaded"
    m = re.search(r'(\d{2}-[A-Za-z]{3}-\d{4})', filename)
    if m: expiry = m.group(1); debug.append(f"Expiry from filename: {expiry}")

    all_rows = list(_csv.reader(io.StringIO(content)))
    debug.append(f"CSV rows: {len(all_rows)}")
    for i,r in enumerate(all_rows[:3]): debug.append(f"  Row {i}: {r[:8]}")

    if len(all_rows) < 3:
        return pd.DataFrame(), debug + ["Too few rows"]

    # Find STRIKE column
    strike_col = None
    header_row_idx = None
    for ri in range(min(4, len(all_rows))):
        for ci, val in enumerate(all_rows[ri]):
            if str(val).strip().upper() == "STRIKE":
                strike_col = ci; header_row_idx = ri
                debug.append(f"STRIKE at row={ri} col={ci}")
                break
        if strike_col is not None: break

    if strike_col is None:
        # Fallback: already-correct format
        try:
            df = pd.read_csv(io.StringIO(content))
            df.columns = [c.strip() for c in df.columns]
            if "strikePrice" in df.columns:
                for c in ["CE_OI","PE_OI","CE_IV","PE_IV","CE_changeOI","PE_changeOI","CE_volume","PE_volume","CE_bid","CE_ask","PE_bid","PE_ask"]:
                    if c not in df.columns: df[c]=0.0
                if "expiryDate" not in df.columns: df["expiryDate"]=expiry
                df = df[df["strikePrice"]>0].reset_index(drop=True)
                if not df.empty:
                    debug.append(f"✅ Fallback strikePrice format — {len(df)} strikes")
                    return df.fillna(0), debug
        except: pass
        return pd.DataFrame(), debug + ["Cannot find STRIKE column"]

    sc = strike_col
    offsets = {"strikePrice":0,"CE_OI":-10,"CE_changeOI":-9,"CE_volume":-8,
               "CE_IV":-7,"CE_LTP":-6,"CE_bid":-3,"CE_ask":-2,
               "PE_bid":+2,"PE_ask":+3,"PE_LTP":+6,"PE_IV":+7,
               "PE_volume":+8,"PE_changeOI":+9,"PE_OI":+10}

    records = []
    for row in all_rows[header_row_idx+1:]:
        if len(row) <= sc: continue
        sp = to_f(row[sc])
        if sp <= 0: continue
        rec = {"expiryDate": expiry}
        for col_name, offset in offsets.items():
            idx = sc + offset
            rec[col_name] = to_f(row[idx]) if 0 <= idx < len(row) else 0.0
        records.append(rec)

    if not records:
        return pd.DataFrame(), debug + ["No valid strike rows"]

    df = pd.DataFrame(records).fillna(0)
    df = df[df["strikePrice"]>0].reset_index(drop=True)
    debug.append(f"✅ Parsed {len(df)} strikes | expiry={expiry}")
    return df, debug


# ═══════════════════════════════════════════════════════════════════════
# BSE PASTE PARSER
# ═══════════════════════════════════════════════════════════════════════
def parse_bse_paste(text, expiry="Uploaded"):
    """
    Parse data copy-pasted from BSE website option chain table.
    Format (tab-separated, 3 lines per strike):
      Line A: CE fields (Chg_OI OI VOL IV LTP CHNG BID_QTY BID ASK ASK_QTY [tab])
      Line B: STRIKE_PRICE (alone, e.g. "71,200.00")
      Line C: PE fields (BID_QTY BID ASK ASK_QTY CHNG LTP IV VOL OI Chg_OI)
    """
    debug = []
    lines = [l for l in text.strip().split("\n")]
    debug.append(f"Total lines: {len(lines)}")

    # Skip header lines (CALLS / PUTS / Chg in OI ...)
    data_start = 0
    for i, line in enumerate(lines):
        l = line.strip().upper()
        if l.startswith("CALLS") or l.startswith("CHG IN OI") or l.startswith("OI"):
            data_start = i + 1
        elif to_f(l) > 1000:  # first real strike price
            data_start = i - 1 if i > 0 else 0
            break
    if data_start < 0: data_start = 0
    debug.append(f"Data starts at line {data_start}")

    data_lines = lines[data_start:]
    records = []
    i = 0
    while i < len(data_lines):
        # Try to find a triplet: CE line, strike line, PE line
        ce_raw = data_lines[i].strip()
        if not ce_raw or ce_raw.upper() in ("CALLS","PUTS"): i += 1; continue

        # Check if next line is a standalone strike price
        if i+1 >= len(data_lines): break
        strike_candidate = to_f(data_lines[i+1].strip())
        if strike_candidate <= 100:
            i += 1; continue
        if i+2 >= len(data_lines): break

        ce_fields = ce_raw.split("\t")
        pe_fields = data_lines[i+2].strip().split("\t")

        def g(arr, idx): return to_f(arr[idx]) if idx < len(arr) else 0.0

        # CE columns (Chg_OI,OI,VOL,IV,LTP,CHNG,BID_QTY,BID,ASK,ASK_QTY)
        # PE columns (BID_QTY,BID,ASK,ASK_QTY,CHNG,LTP,IV,VOL,OI,Chg_OI)
        records.append({
            "strikePrice": strike_candidate,
            "expiryDate":  expiry,
            "CE_changeOI": g(ce_fields,0),
            "CE_OI":       g(ce_fields,1),
            "CE_volume":   g(ce_fields,2),
            "CE_IV":       g(ce_fields,3),
            "CE_LTP":      g(ce_fields,4),
            "CE_bid":      g(ce_fields,7),
            "CE_ask":      g(ce_fields,8),
            "PE_bid":      g(pe_fields,1),
            "PE_ask":      g(pe_fields,2),
            "PE_LTP":      g(pe_fields,5),
            "PE_IV":       g(pe_fields,6),
            "PE_volume":   g(pe_fields,7),
            "PE_OI":       g(pe_fields,8),
            "PE_changeOI": g(pe_fields,9),
        })
        i += 3

    if not records:
        return pd.DataFrame(), debug + ["Could not parse any strikes from pasted text"]

    df = pd.DataFrame(records).fillna(0)
    df = df[df["strikePrice"]>0].reset_index(drop=True)
    debug.append(f"✅ Parsed {len(df)} BSE strikes")
    return df, debug


# ═══════════════════════════════════════════════════════════════════════
# BSE LIVE CHAIN
# ═══════════════════════════════════════════════════════════════════════
_UA_CHROME = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

@st.cache_data(ttl=90, show_spinner=False)
def fetch_bse_chain(symbol):
    """Fetch BSE option chain via BSE API. Returns (df, spot, expiries, source, errors)"""
    errors = []
    pid, scrip = BSE_SCRIP_MAP.get(symbol, ("1","SENSEX"))

    hdrs = {
        "User-Agent": _UA_CHROME,
        "Accept": "application/json, text/plain, */*",
        "Referer": f"https://www.bseindia.com/stock-share-price/future-options/derivatives/{pid}/",
        "Origin": "https://www.bseindia.com",
    }

    try:
        s = requests.Session(); s.headers.update(hdrs)
        # Step 1: get expiry dates
        er = s.get(
            f"https://api.bseindia.com/BseIndiaAPI/api/ddlExpiry_Options/w"
            f"?productid={pid}&scripcode=&ExpiryDate=&optionType=CE&StrikePrice=&SeleScrip={scrip}",
            timeout=10)
        if er.status_code != 200:
            errors.append(f"BSE expiry API: HTTP {er.status_code}")
            return None, 0.0, [], None, errors
        ej = er.json()
        expiries = [item["ExpiryDate"] for item in (ej.get("Table") or ej.get("table") or [])]
        if not expiries:
            errors.append("BSE expiry API: no expiries returned")
            return None, 0.0, [], None, errors

        # Step 2: get chain data for nearest expiry
        exp = expiries[0]
        cr = s.get(
            f"https://api.bseindia.com/BseIndiaAPI/api/FnOChainData/w"
            f"?productid={pid}&scripcode=&ExpiryDate={exp}&optionType=&StrikePrice=&SeleScrip={scrip}",
            timeout=12)
        if cr.status_code != 200:
            errors.append(f"BSE chain API: HTTP {cr.status_code}")
            return None, 0.0, [], None, errors

        cj = cr.json()
        rows_raw = cj.get("Table") or cj.get("table") or []
        spot = float(cj.get("CurrentValue") or cj.get("UnderlyingValue") or 0)

        records = []
        for row in rows_raw:
            sp = to_f(row.get("StrikePrice","0"))
            if sp <= 0: continue
            records.append({
                "strikePrice": sp, "expiryDate": exp,
                "CE_LTP":      to_f(row.get("CE_LTP",0)),
                "CE_OI":       to_f(row.get("CE_OI",0)),
                "CE_changeOI": to_f(row.get("CE_ChgOI",0)),
                "CE_volume":   to_f(row.get("CE_Volume",0)),
                "CE_IV":       to_f(row.get("CE_IV",0)),
                "CE_bid":      to_f(row.get("CE_BidPrice",0)),
                "CE_ask":      to_f(row.get("CE_AskPrice",0)),
                "PE_LTP":      to_f(row.get("PE_LTP",0)),
                "PE_OI":       to_f(row.get("PE_OI",0)),
                "PE_changeOI": to_f(row.get("PE_ChgOI",0)),
                "PE_volume":   to_f(row.get("PE_Volume",0)),
                "PE_IV":       to_f(row.get("PE_IV",0)),
                "PE_bid":      to_f(row.get("PE_BidPrice",0)),
                "PE_ask":      to_f(row.get("PE_AskPrice",0)),
            })

        df = pd.DataFrame(records)
        if df.empty:
            errors.append("BSE chain API: empty chain")
            return None, 0.0, [], None, errors

        if spot == 0 and _HAS_YF:
            try:
                h = yf.Ticker(NSE_TO_YF.get(symbol,"^BSESN")).history(period="1d")
                if not h.empty: spot = float(h["Close"].squeeze().iloc[-1])
            except: pass

        return df, spot, expiries, "BSE-API", errors

    except Exception as e:
        errors.append(f"BSE API: {e}")
        return None, 0.0, [], None, errors


# ═══════════════════════════════════════════════════════════════════════
# NSE LIVE CHAIN — 5 strategies
# ═══════════════════════════════════════════════════════════════════════
_UA_FF     = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0"
_UA_MOBILE = "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36"

def _nse_url(sym):
    return (f"https://www.nseindia.com/api/option-chain-indices?symbol={sym}"
            if sym in NSE_INDICES else
            f"https://www.nseindia.com/api/option-chain-equities?symbol={sym}")

def _parse_nse_resp(r):
    if r.status_code != 200: return None,0.0,[],f"HTTP {r.status_code}"
    try:
        j=r.json(); rec=j.get("records",{})
        spot=float(rec.get("underlyingValue") or 0)
        exps=rec.get("expiryDates",[])
        rows=[]
        for item in rec.get("data",[]):
            ce=item.get("CE",{}); pe=item.get("PE",{})
            rows.append({"strikePrice":float(item.get("strikePrice",0)),"expiryDate":item.get("expiryDate",""),
                "CE_LTP":float(ce.get("lastPrice",0)),"CE_OI":float(ce.get("openInterest",0)),
                "CE_changeOI":float(ce.get("changeinOpenInterest",0)),"CE_volume":float(ce.get("totalTradedVolume",0)),
                "CE_IV":float(ce.get("impliedVolatility",0)),"CE_bid":float(ce.get("bidprice",0)),"CE_ask":float(ce.get("askPrice",0)),
                "PE_LTP":float(pe.get("lastPrice",0)),"PE_OI":float(pe.get("openInterest",0)),
                "PE_changeOI":float(pe.get("changeinOpenInterest",0)),"PE_volume":float(pe.get("totalTradedVolume",0)),
                "PE_IV":float(pe.get("impliedVolatility",0)),"PE_bid":float(pe.get("bidprice",0)),"PE_ask":float(pe.get("askPrice",0))})
        df=pd.DataFrame(rows)
        if df.empty or spot==0: return None,0.0,[],"Empty"
        return df,spot,exps,None
    except Exception as e: return None,0.0,[],f"Parse: {e}"

def _nse_A(sym):
    try:
        s=requests.Session(); s.headers.update({"User-Agent":_UA_CHROME,"Accept-Language":"en-US,en;q=0.9,hi;q=0.8","Accept-Encoding":"gzip, deflate, br"})
        nav={"Accept":"text/html,application/xhtml+xml,*/*;q=0.8","Sec-Fetch-Dest":"document","Sec-Fetch-Mode":"navigate"}
        api={"Accept":"*/*","Referer":"https://www.nseindia.com/option-chain","X-Requested-With":"XMLHttpRequest","Sec-Fetch-Mode":"cors","Sec-Fetch-Site":"same-origin"}
        s.get("https://www.nseindia.com/",timeout=10,headers=nav); time.sleep(1.5)
        s.get("https://www.nseindia.com/option-chain",timeout=10,headers=nav); time.sleep(1.2)
        return _parse_nse_resp(s.get(_nse_url(sym),timeout=15,headers=api))
    except Exception as e: return None,0.0,[],f"NSE-A: {e}"

def _nse_B(sym):
    try:
        s=requests.Session(); s.headers.update({"User-Agent":_UA_FF,"Accept-Language":"en-US,en;q=0.5","Accept-Encoding":"gzip, deflate, br","DNT":"1"})
        nav={"Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
        api={"Accept":"application/json,*/*","Referer":"https://www.nseindia.com/market-data/equity-derivatives-watch","X-Requested-With":"XMLHttpRequest"}
        s.get("https://www.nseindia.com/",timeout=10,headers=nav); time.sleep(1.3)
        s.get("https://www.nseindia.com/market-data/equity-derivatives-watch",timeout=10,headers={**nav,"Referer":"https://www.nseindia.com/"}); time.sleep(1.0)
        return _parse_nse_resp(s.get(_nse_url(sym),timeout=15,headers=api))
    except Exception as e: return None,0.0,[],f"NSE-B: {e}"

def _nse_C(sym):
    try:
        s=requests.Session(); s.headers.update({"User-Agent":_UA_MOBILE,"Accept-Language":"en-IN,en;q=0.9,hi;q=0.8","Accept-Encoding":"gzip, deflate, br"})
        nav={"Accept":"text/html,*/*;q=0.8","Sec-Fetch-Mode":"navigate","Upgrade-Insecure-Requests":"1"}
        s.get("https://www.nseindia.com/",timeout=10,headers=nav); time.sleep(1.4)
        s.get("https://www.nseindia.com/option-chain",timeout=10,headers={**nav,"Referer":"https://www.nseindia.com/"}); time.sleep(1.0)
        api={"Accept":"*/*","Referer":"https://www.nseindia.com/option-chain","X-Requested-With":"XMLHttpRequest","Sec-Fetch-Mode":"cors"}
        return _parse_nse_resp(s.get(_nse_url(sym),timeout=15,headers=api))
    except Exception as e: return None,0.0,[],f"NSE-C: {e}"

def _nse_nsepython(sym):
    try:
        from nsepython import nse_optionchain_scrapper
        raw=nse_optionchain_scrapper(sym)
        if not raw: return None,0.0,[],"nsepython: empty"
        rec=raw.get("records",{})
        spot=float(rec.get("underlyingValue") or 0); exps=rec.get("expiryDates",[])
        rows=[]
        for item in rec.get("data",[]):
            ce=item.get("CE",{}); pe=item.get("PE",{})
            rows.append({"strikePrice":float(item.get("strikePrice",0)),"expiryDate":item.get("expiryDate",""),
                "CE_LTP":float(ce.get("lastPrice",0)),"CE_OI":float(ce.get("openInterest",0)),
                "CE_changeOI":float(ce.get("changeinOpenInterest",0)),"CE_volume":float(ce.get("totalTradedVolume",0)),
                "CE_IV":float(ce.get("impliedVolatility",0)),"CE_bid":float(ce.get("bidprice",0)),"CE_ask":float(ce.get("askPrice",0)),
                "PE_LTP":float(pe.get("lastPrice",0)),"PE_OI":float(pe.get("openInterest",0)),
                "PE_changeOI":float(pe.get("changeinOpenInterest",0)),"PE_volume":float(pe.get("totalTradedVolume",0)),
                "PE_IV":float(pe.get("impliedVolatility",0)),"PE_bid":float(pe.get("bidprice",0)),"PE_ask":float(pe.get("askPrice",0))})
        df=pd.DataFrame(rows)
        if df.empty or spot==0: return None,0.0,[],"nsepython: empty"
        return df,spot,exps,None
    except ImportError: return None,0.0,[],"nsepython not installed"
    except Exception as e: return None,0.0,[],f"nsepython: {e}"

def _nse_opstra(sym):
    om={"NIFTY":"NIFTY","BANKNIFTY":"BANKNIFTY","FINNIFTY":"FINNIFTY","MIDCPNIFTY":"MIDCPNIFTY"}
    if sym not in om: return None,0.0,[],f"Opstra: indices only, not {sym}"
    try:
        hdrs={"User-Agent":_UA_CHROME,"Accept":"application/json,*/*","Referer":"https://opstra.definedge.com/","Origin":"https://opstra.definedge.com"}
        s=requests.Session(); s.headers.update(hdrs)
        er=s.get(f"https://opstra.definedge.com/api/openinterest/expiry/{om[sym]}",timeout=10)
        if er.status_code!=200: return None,0.0,[],f"Opstra expiry HTTP {er.status_code}"
        exps=er.json().get("data",[])
        if not exps: return None,0.0,[],"Opstra: no expiries"
        oc=s.get(f"https://opstra.definedge.com/api/openinterest/{om[sym]}/{exps[0]}",timeout=10)
        if oc.status_code!=200: return None,0.0,[],f"Opstra chain HTTP {oc.status_code}"
        j=oc.json(); spot=float(j.get("underlyingValue",0) or j.get("spotPrice",0) or 0)
        rows=[]
        for item in j.get("data",[]):
            ce=item.get("CE",{}); pe=item.get("PE",{})
            rows.append({"strikePrice":float(item.get("strikePrice",0)),"expiryDate":exps[0],
                "CE_LTP":float(ce.get("lastPrice",0) or ce.get("ltp",0)),"CE_OI":float(ce.get("openInterest",0) or ce.get("oi",0)),
                "CE_changeOI":float(ce.get("changeinOpenInterest",0)),"CE_volume":float(ce.get("totalTradedVolume",0)),
                "CE_IV":float(ce.get("impliedVolatility",0)),"CE_bid":0.0,"CE_ask":0.0,
                "PE_LTP":float(pe.get("lastPrice",0) or pe.get("ltp",0)),"PE_OI":float(pe.get("openInterest",0) or pe.get("oi",0)),
                "PE_changeOI":float(pe.get("changeinOpenInterest",0)),"PE_volume":float(pe.get("totalTradedVolume",0)),
                "PE_IV":float(pe.get("impliedVolatility",0)),"PE_bid":0.0,"PE_ask":0.0})
        df=pd.DataFrame(rows)
        if df.empty: return None,0.0,[],"Opstra: empty"
        if spot==0 and _HAS_YF:
            try:
                h=yf.Ticker(NSE_TO_YF.get(sym,"^NSEI")).history(period="1d")
                if not h.empty: spot=float(h["Close"].squeeze().iloc[-1])
            except: pass
        return df,spot,exps,None
    except Exception as e: return None,0.0,[],f"Opstra: {e}"

@st.cache_data(ttl=90, show_spinner=False)
def fetch_nse_chain(symbol):
    errors=[]
    for name,fn in [("nsepython",_nse_nsepython),("NSE-A",_nse_A),("NSE-B",_nse_B),("NSE-C",_nse_C),("Opstra",_nse_opstra)]:
        try:
            df,spot,exps,err=fn(symbol)
            if df is not None and spot>0: return df,spot,exps,name,errors
            errors.append(f"❌ {name}: {err}")
        except Exception as e: errors.append(f"❌ {name}: {e}")
    return None,0.0,[],None,errors


# ═══════════════════════════════════════════════════════════════════════
# SPOT + HISTORY
# ═══════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60, show_spinner=False)
def get_spot(yf_sym):
    if not _HAS_YF: return 0.0
    _wait(1.2)
    try:
        fi=yf.Ticker(yf_sym).fast_info
        for a in ("last_price","lastPrice","regular_market_price","regularMarketPrice","previousClose"):
            v=getattr(fi,a,None)
            if v and float(v)>0: return float(v)
    except: pass
    try:
        h=yf.Ticker(yf_sym).history(period="2d",interval="1d")
        if not h.empty: return float(h["Close"].squeeze().iloc[-1])
    except: pass
    return 0.0

@st.cache_data(ttl=300, show_spinner=False)
def get_history(yf_sym, period="60d"):
    if not _HAS_YF: return pd.DataFrame()
    _wait(1.2)
    try:
        df=yf.Ticker(yf_sym).history(period=period,interval="1d",auto_adjust=True)
        return df if not df.empty else pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=120, show_spinner=False)
def get_yf_chain(sym, expiry):
    if not _HAS_YF: return None,None
    _wait(1.2)
    try:
        c=yf.Ticker(sym).option_chain(expiry); return c.calls.copy(),c.puts.copy()
    except: return None,None

def build_yf_df(calls, puts, expiry):
    if calls is None or calls.empty: return pd.DataFrame()
    df=pd.DataFrame()
    df["strikePrice"]=calls["strike"].values if "strike" in calls.columns else []
    df["expiryDate"]=expiry
    for col,src in [("CE_LTP","lastPrice"),("CE_OI","openInterest"),("CE_volume","volume"),("CE_bid","bid"),("CE_ask","ask")]:
        df[col]=calls[src].values if src in calls.columns else 0.0
    df["CE_IV"]=(calls["impliedVolatility"].values*100) if "impliedVolatility" in calls.columns else 0.0
    df["CE_changeOI"]=0.0
    if puts is not None and not puts.empty and "strike" in puts.columns:
        pi=puts.set_index("strike")
        for col,src in [("PE_LTP","lastPrice"),("PE_OI","openInterest"),("PE_volume","volume"),("PE_bid","bid"),("PE_ask","ask")]:
            df[col]=df["strikePrice"].map(pi[src] if src in pi.columns else pd.Series(dtype=float)).fillna(0.0).values
        df["PE_IV"]=df["strikePrice"].map(pi["impliedVolatility"]*100 if "impliedVolatility" in pi.columns else pd.Series(dtype=float)).fillna(0.0).values
    else:
        for c in ["PE_LTP","PE_OI","PE_volume","PE_bid","PE_ask","PE_IV"]: df[c]=0.0
    df["PE_changeOI"]=0.0
    return df.fillna(0.0).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES + IV
# ═══════════════════════════════════════════════════════════════════════
def bs(S,K,T,r,sig,kind="call"):
    if T<1e-6 or sig<1e-6 or S<=0 or K<=0: return dict(delta=0,gamma=0,theta=0,vega=0,price=0)
    d1=(np.log(S/K)+(r+.5*sig**2)*T)/(sig*np.sqrt(T)); d2=d1-sig*np.sqrt(T)
    if kind=="call":
        p=S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2); delta=norm.cdf(d1)
        theta=(-(S*norm.pdf(d1)*sig)/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2))/365
    else:
        p=K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1); delta=norm.cdf(d1)-1
        theta=(-(S*norm.pdf(d1)*sig)/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2))/365
    g=norm.pdf(d1)/(S*sig*np.sqrt(T)); v=S*norm.pdf(d1)*np.sqrt(T)/100
    return dict(delta=round(delta,4),gamma=round(g,8),theta=round(theta,4),vega=round(v,4),price=round(max(p,0),4))

def calc_iv(mkt,S,K,T,r,kind="call"):
    if T<=0 or mkt<=0 or S<=0: return 0.20
    try: return max(brentq(lambda s:bs(S,K,T,r,s,kind)["price"]-mkt,1e-4,20.0,xtol=1e-5),0.001)
    except: return 0.20

def add_greeks(df,spot,expiry_str):
    if df is None or df.empty: return df
    try:
        fmt="%d-%b-%Y" if ("-" in str(expiry_str) and not str(expiry_str)[4:5].isdigit()) else "%Y-%m-%d"
        T=max((datetime.datetime.strptime(str(expiry_str),fmt)-datetime.datetime.now()).days/365.0,1/365)
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


# ═══════════════════════════════════════════════════════════════════════
# SIGNALS — specific strike, per-indicator verdict, full narrative
# ═══════════════════════════════════════════════════════════════════════
def compute_signals(df, spot):
    out = dict(
        pcr=0, max_pain=spot, atm=spot, straddle=0,
        resistance=spot, support=spot, atm_iv=20.0, skew=0.0,
        gamma_blast=False, abnormal=[],
        rec="⚪ NO TRADE", direction="NONE", conf=0,
        scalp="—", intraday="—", swing="—", pos="—",
        buy_strike_atm=0, buy_strike_otm=0, buy_side="CE",
        indicators=[],  # list of (name, value_str, verdict, score_impact, explanation)
    )
    if df is None or df.empty or spot == 0: return out

    df = df.copy(); df["strikePrice"] = df["strikePrice"].astype(float)
    ai  = (df["strikePrice"] - spot).abs().idxmin()
    atm = df.loc[ai]
    out["atm"] = float(atm["strikePrice"])

    ce_ltp = float(atm.get("CE_LTP",0)); pe_ltp = float(atm.get("PE_LTP",0))
    out["straddle"] = round(ce_ltp + pe_ltp, 2)
    ce_oi = float(df["CE_OI"].sum()); pe_oi = float(df["PE_OI"].sum())
    out["pcr"] = round(pe_oi/ce_oi, 4) if ce_oi > 0 else 0

    strikes = df["strikePrice"].values; c_oi = df["CE_OI"].values; p_oi = df["PE_OI"].values
    pain = [sum(max(0,k-s)*o for k,o in zip(strikes,c_oi)) +
            sum(max(0,s-k)*o for k,o in zip(strikes,p_oi)) for s in strikes]
    out["max_pain"] = float(strikes[int(np.argmin(pain))]) if pain else spot
    out["resistance"] = float(df.loc[df["CE_OI"].idxmax(), "strikePrice"])
    out["support"]    = float(df.loc[df["PE_OI"].idxmax(), "strikePrice"])

    ce_iv = float(atm.get("CE_IV",0)); pe_iv = float(atm.get("PE_IV",0))
    out["atm_iv"] = (ce_iv+pe_iv)/2 if (ce_iv+pe_iv)>0 else 20.0
    out["skew"]   = round(pe_iv - ce_iv, 2)

    near = (df["CE_OI"].where(df["strikePrice"].between(spot*.99,spot*1.01),0).sum() +
            df["PE_OI"].where(df["strikePrice"].between(spot*.99,spot*1.01),0).sum())
    out["gamma_blast"] = (ce_oi+pe_oi)>0 and near/(ce_oi+pe_oi+1)>0.35

    iv = out["atm_iv"]; pcr = out["pcr"]; mp = out["max_pain"]
    skew = out["skew"]
    score = 50; direction = "NONE"
    inds = []  # (name, val_str, verdict, pts, explanation)

    # ── Indicator 1: PCR ──────────────────────────────────────────
    if pcr > 1.5:
        pts=12; direction="CALL"; verdict="BULLISH"
        exp = (f"PCR {pcr:.2f} is HIGH — significantly more put OI than call OI. "
               f"This is CONTRARIAN BULLISH. The put writers (who profit when market goes UP) "
               f"have sold ₹{pe_oi/1e7:.1f} Cr worth of puts, creating a support floor. "
               f"Market statistically tends to move UP from high PCR levels.")
    elif pcr > 1.1:
        pts=6; direction="CALL"; verdict="MILDLY BULLISH"
        exp = (f"PCR {pcr:.2f} — more puts than calls. Put writers are defending support. "
               f"Mild bullish bias.")
    elif pcr < 0.5:
        pts=12; direction="PUT"; verdict="BEARISH"
        exp = (f"PCR {pcr:.2f} is VERY LOW — far more call OI than put OI. "
               f"This is CONTRARIAN BEARISH. The market is optimistically positioned; "
               f"when everyone is long calls, the market often disappoints and falls.")
    elif pcr < 0.9:
        pts=6; direction="PUT"; verdict="MILDLY BEARISH"
        exp = (f"PCR {pcr:.2f} — more calls than puts. Mildly bearish contrarian lean.")
    else:
        pts=0; verdict="NEUTRAL"
        exp = (f"PCR {pcr:.2f} — balanced between calls and puts. "
               f"No directional bias from OI positioning.")
    score += pts
    inds.append(("Put-Call Ratio (PCR)", f"{pcr:.3f}", verdict, f"{pts:+d}", exp))

    # ── Indicator 2: IV ───────────────────────────────────────────
    if iv > 50:
        pts=-25; verdict="SELL IV (too expensive)"
        exp = (f"IV {iv:.0f}% is VERY HIGH. Options are overpriced — "
               f"premium is inflated by fear (event risk, macro uncertainty). "
               f"If you buy now, even being right on direction may not profit "
               f"because IV will collapse after the event, crushing the premium. "
               f"Strategy: sell options (straddle/strangle) or wait.")
    elif iv > 35:
        pts=-12; verdict="EXPENSIVE — avoid buying"
        exp = (f"IV {iv:.0f}% is high. Each day of holding costs you ₹{float(atm.get('CE_theta',0)):.0f} "
               f"in theta. Buy only with very high confidence signal.")
    elif iv < 15:
        pts=15; verdict="✅ CHEAP — ideal buying conditions"
        exp = (f"IV {iv:.0f}% is LOW — options are cheap relative to historical volatility. "
               f"This is the BEST time to buy options: low premium cost means even a moderate "
               f"directional move gives large % returns. Also: if IV rises, your option gains extra.")
    elif iv < 25:
        pts=10; verdict="REASONABLE — good to buy"
        exp = (f"IV {iv:.0f}% is moderate. Fair pricing — decent conditions to buy options "
               f"if direction signal is strong.")
    else:
        pts=4; verdict="MODERATE"
        exp = (f"IV {iv:.0f}% is above average. Be selective — only enter on strong setups.")
    score += pts
    inds.append(("Implied Volatility (IV)", f"{iv:.1f}%", verdict, f"{pts:+d}", exp))

    # ── Indicator 3: Max Pain ────────────────────────────────────
    mp_pct = (mp - spot)/spot*100 if spot > 0 else 0
    if mp_pct > 2 and direction == "CALL":
        pts=10; verdict="✅ BULLISH — market pulled toward max pain"
        exp = (f"Max pain ₹{mp:,.0f} is {mp_pct:.1f}% ABOVE current price ₹{spot:,.0f}. "
               f"Near expiry, the market gravitates toward max pain (where most options expire worthless). "
               f"This creates an upward magnetic pull, supporting the call buying thesis.")
    elif mp_pct < -2 and direction == "PUT":
        pts=10; verdict="✅ BEARISH — max pain below spot"
        exp = (f"Max pain ₹{mp:,.0f} is {abs(mp_pct):.1f}% BELOW spot. "
               f"Downward gravitational pull near expiry, supporting puts.")
    elif abs(mp_pct) < 0.5:
        pts=2; verdict="NEUTRAL — at max pain"
        exp = (f"Spot ₹{spot:,.0f} is almost exactly at max pain ₹{mp:,.0f}. "
               f"This means option sellers are already winning. Market may stay sideways near expiry.")
    else:
        pts=-5; verdict="AGAINST DIRECTION"
        exp = (f"Max pain {mp_pct:+.1f}% is in the OPPOSITE direction to the signal. "
               f"This is a negative factor — gravity pulls against the trade.")
    score += pts
    inds.append(("Max Pain", f"₹{mp:,.0f} ({mp_pct:+.1f}%)", verdict, f"{pts:+d}", exp))

    # ── Indicator 4: Skew ─────────────────────────────────────────
    if abs(skew) > 0:
        if skew > 8:
            pts=5 if direction=="PUT" else -4; verdict="PUT SKEW — bearish lean"
            exp=(f"PE IV {pe_iv:.0f}% vs CE IV {ce_iv:.0f}% — puts cost more than calls. "
                 f"Institutions are paying up for downside protection, signaling bearish expectation.")
        elif skew < -8:
            pts=5 if direction=="CALL" else -4; verdict="CALL SKEW — bullish lean"
            exp=(f"CE IV {ce_iv:.0f}% vs PE IV {pe_iv:.0f}% — calls cost more than puts. "
                 f"Aggressive call buying, bullish expectation from big players.")
        else:
            pts=0; verdict="BALANCED"
            exp=(f"CE IV {ce_iv:.0f}% ≈ PE IV {pe_iv:.0f}%. "
                 f"No strong skew — market not pricing in a clear directional move.")
        inds.append(("IV Skew (PE−CE)", f"{skew:+.1f}%", verdict, f"{pts:+d}", exp))
        score += pts

    # ── Indicator 5: OI Change ───────────────────────────────────
    ce_chg = float(df["CE_changeOI"].sum()); pe_chg = float(df["PE_changeOI"].sum())
    if (ce_chg + pe_chg) != 0:
        if pe_chg > ce_chg*1.3 and direction == "CALL":
            pts=7; verdict="✅ FRESH PUT WRITING — bullish"
            exp=(f"PE OI added {pe_chg/1000:.0f}K contracts today vs CE added {ce_chg/1000:.0f}K. "
                 f"Put writers are selling new puts — they profit if market stays above support. "
                 f"This builds a fresh support floor, confirming the bullish call.")
        elif ce_chg > pe_chg*1.3 and direction == "PUT":
            pts=7; verdict="✅ FRESH CALL WRITING — bearish"
            exp=(f"CE OI added {ce_chg/1000:.0f}K contracts today vs PE added {pe_chg/1000:.0f}K. "
                 f"Call writers are selling new calls — they profit if market stays below resistance. "
                 f"This builds a fresh resistance ceiling, confirming the bearish thesis.")
        elif ce_chg > pe_chg*1.3 and direction == "CALL":
            pts=-5; verdict="AGAINST — call writing on a bullish signal"
            exp=(f"Call writers are adding aggressively ({ce_chg/1000:.0f}K new CE). "
                 f"Smart money is betting the market WON'T go higher — this conflicts with the bullish signal.")
        else:
            pts=0; verdict="MIXED"
            exp=(f"OI change: +{ce_chg/1000:.0f}K CE, +{pe_chg/1000:.0f}K PE. No clear positioning edge.")
        inds.append(("OI Change Today", f"CE {ce_chg/1000:+.0f}K / PE {pe_chg/1000:+.0f}K", verdict, f"{pts:+d}", exp))
        score += pts

    # ── Indicator 6: Support/Resistance location ──────────────────
    res = out["resistance"]; sup = out["support"]
    if sup < spot < res:
        pts=6; verdict="IN BUY ZONE"
        exp=(f"Spot ₹{spot:,.0f} is between support ₹{sup:,.0f} and resistance ₹{res:,.0f}. "
             f"Market has room to move in either direction. "
             f"Buy calls near support, buy puts near resistance.")
    elif spot > res:
        pts=8 if direction=="CALL" else -5; verdict="ABOVE RESISTANCE — breakout" if direction=="CALL" else "EXTENDED"
        exp=(f"Spot ₹{spot:,.0f} has broken ABOVE resistance ₹{res:,.0f}. "
             f"{'This is a strong bullish breakout signal — previous resistance becomes support.' if direction=='CALL' else 'Market is extended above resistance — risky to buy puts here.'}")
    elif spot < sup:
        pts=8 if direction=="PUT" else -5; verdict="BELOW SUPPORT — breakdown" if direction=="PUT" else "EXTENDED DOWN"
        exp=(f"Spot ₹{spot:,.0f} has broken BELOW support ₹{sup:,.0f}. "
             f"{'Bearish breakdown — previous support becomes resistance.' if direction=='PUT' else 'Market is extended below support — risky to buy calls here.'}")
    else:
        pts=0; verdict="AT KEY LEVEL"
        exp=(f"Spot is at or near a key S/R level. Wait for clear directional break before entering.")
    inds.append(("S/R Location", f"S:{sup:,.0f} | Now:{spot:,.0f} | R:{res:,.0f}", verdict, f"{pts:+d}", exp))
    score += pts

    # ── Finalise ─────────────────────────────────────────────────
    score = min(max(int(score),0),100)
    out["conf"] = score; out["direction"] = direction; out["indicators"] = inds

    # ── Specific strike selection ─────────────────────────────────
    atm_s = int(out["atm"])
    step  = round((df["strikePrice"].diff().dropna().abs().mode().iloc[0] if len(df)>1 else 50))
    if direction == "CALL":
        otm_s  = atm_s + step           # 1 strike OTM
        otm2_s = atm_s + 2*step         # 2 strikes OTM
        side   = "CE"
    else:
        otm_s  = atm_s - step
        otm2_s = atm_s - 2*step
        side   = "PE"

    out["buy_strike_atm"] = atm_s
    out["buy_strike_otm"] = otm_s
    out["buy_side"]       = side

    def _ltp(strike, col):
        r = df[df["strikePrice"]==strike]
        return float(r[col].values[0]) if not r.empty and col in r.columns else 0.0

    atm_px  = _ltp(atm_s,  f"{side}_LTP")
    otm_px  = _ltp(otm_s,  f"{side}_LTP")
    otm2_px = _ltp(otm2_s, f"{side}_LTP")
    pxstr   = lambda k,p: f"₹{p:.2f}" if p>0 else "—"
    sym_sfx = "CE" if side=="CE" else "PE"

    if score >= 72 and direction in ("CALL","PUT"):
        dir_word = "UP" if direction=="CALL" else "DOWN"
        out["rec"] = f"🟢 BUY {side} — {dir_word}" if direction=="CALL" else f"🔴 BUY {side} — {dir_word}"
        out["scalp"]   = f"Buy {atm_s} {sym_sfx} @ {pxstr(atm_s,atm_px)} | Exit +20–35% | SL -30%"
        out["intraday"]= f"Buy {atm_s} {sym_sfx} @ {pxstr(atm_s,atm_px)} | Exit +50–70% | SL -30%"
        out["swing"]   = f"Buy {otm_s} {sym_sfx} @ {pxstr(otm_s,otm_px)} (next expiry) | Exit +80–100% | SL -30%"
        out["pos"]     = f"Buy {otm2_s} {sym_sfx} @ {pxstr(otm2_s,otm2_px)} (far expiry) | Hold for big move"
    elif score >= 58:
        out["rec"] = "🟡 WATCH — signal forming"
        out["scalp"]   = f"Wait | Potential: {atm_s} {sym_sfx}"
        out["intraday"]=out["swing"]=out["pos"]="Wait for confirmation"
    else:
        out["rec"] = "⚪ NO TRADE — stay out"
        out["scalp"]=out["intraday"]=out["swing"]=out["pos"]="No edge. Cash is a position."

    # ── Alerts ────────────────────────────────────────────────────
    if iv > 55: out["abnormal"].append(f"🔴 IV {iv:.0f}%: Premium very expensive — don't buy options now")
    elif iv > 35: out["abnormal"].append(f"⚠️ IV {iv:.0f}%: Options pricey — theta decays fast")
    elif 0 < iv < 15: out["abnormal"].append(f"✅ IV {iv:.0f}%: Options cheap — best time to buy")
    if abs(skew) > 8: out["abnormal"].append(f"📊 IV Skew {skew:+.0f}%: {'Big money buying puts = bearish hedge' if skew>0 else 'Aggressive call buying = bullish positioning'}")
    if abs(mp_pct) > 2: out["abnormal"].append(f"🎯 Max pain ₹{mp:,.0f} ({mp_pct:+.1f}%): Market gravitates here near expiry")
    if out["gamma_blast"]: out["abnormal"].append("⚡ GAMMA BLAST: Huge OI at ATM — options explode on any breakout. Enter fast on direction.")

    return out


# ═══════════════════════════════════════════════════════════════════════
# OI NARRATIVE (plain English summary of OI table)
# ═══════════════════════════════════════════════════════════════════════
def oi_narrative(df, spot, sig):
    if df is None or df.empty: return ""
    res = sig["resistance"]; sup = sig["support"]; mp = sig["max_pain"]

    # Top CE OI
    top_ce = df.nlargest(3,"CE_OI")
    top_pe = df.nlargest(3,"PE_OI")
    ce_walls = ", ".join([f"₹{int(r['strikePrice']):,}" for _,r in top_ce.iterrows()])
    pe_walls  = ", ".join([f"₹{int(r['strikePrice']):,}" for _,r in top_pe.iterrows()])

    # OI change direction
    ce_chg = df["CE_changeOI"].sum(); pe_chg = df["PE_changeOI"].sum()
    if ce_chg > 0 and pe_chg > 0:
        chg_msg = f"Both CE (+{ce_chg/1000:.0f}K) and PE (+{pe_chg/1000:.0f}K) OI are being added today, meaning both bulls and bears are building fresh positions — a volatile setup."
    elif ce_chg > 0 and pe_chg < 0:
        chg_msg = f"Call OI rising (+{ce_chg/1000:.0f}K) while Put OI falling ({pe_chg/1000:.0f}K): call writers building ceilings, put writers exiting floors. BEARISH positioning."
    elif ce_chg < 0 and pe_chg > 0:
        chg_msg = f"Put OI rising (+{pe_chg/1000:.0f}K) while Call OI falling ({ce_chg/1000:.0f}K): put writers building floors, call writers covering. BULLISH positioning."
    else:
        chg_msg = f"Both CE ({ce_chg/1000:.0f}K) and PE ({pe_chg/1000:.0f}K) OI declining — positions being squared off. Market may be indecisive or near a big move."

    # OI % change for top strikes
    df2 = df.copy()
    prev_ce = (df2["CE_OI"] - df2["CE_changeOI"]).clip(lower=1)
    prev_pe = (df2["PE_OI"] - df2["PE_changeOI"]).clip(lower=1)
    df2["CE_chg_pct"] = (df2["CE_changeOI"]/prev_ce*100).fillna(0)
    df2["PE_chg_pct"] = (df2["PE_changeOI"]/prev_pe*100).fillna(0)
    max_ce_pct = df2.loc[df2["CE_chg_pct"].idxmax()]
    max_pe_pct = df2.loc[df2["PE_chg_pct"].idxmax()]
    pct_msg = (f"Biggest % surge: <b>{int(max_ce_pct['strikePrice']):,} CE +{max_ce_pct['CE_chg_pct']:.0f}%</b> "
               f"(new call writers at this resistance) and "
               f"<b>{int(max_pe_pct['strikePrice']):,} PE +{max_pe_pct['PE_chg_pct']:.0f}%</b> "
               f"(put writers building support here).")

    # Location of spot
    if spot > res: loc_msg = f"⚠️ Spot ₹{spot:,.0f} has <b>broken above resistance ₹{res:,.0f}</b>. Previous resistance now support. Watch if it sustains."
    elif spot < sup: loc_msg = f"⚠️ Spot ₹{spot:,.0f} has <b>broken below support ₹{sup:,.0f}</b>. Previous support now resistance. Bears in control."
    else: loc_msg = f"Spot ₹{spot:,.0f} is inside the range: support ₹{sup:,.0f} → resistance ₹{res:,.0f}."

    return (f"<b>🔴 Call (CE) walls at:</b> {ce_walls} — these are resistance zones where sellers are positioned.<br>"
            f"<b>🟢 Put (PE) walls at:</b> {pe_walls} — these are support zones where put sellers defend.<br>"
            f"<b>📊 OI Change:</b> {chg_msg}<br>"
            f"<b>🔥 Biggest moves today:</b> {pct_msg}<br>"
            f"<b>📍 Location:</b> {loc_msg}<br>"
            f"<b>🎯 Max Pain:</b> ₹{mp:,.0f} — near expiry the market tends to drift here, causing maximum pain for option buyers.")


# ═══════════════════════════════════════════════════════════════════════
# DYNAMIC STRADDLE from actual chain
# ═══════════════════════════════════════════════════════════════════════
def plot_straddle_dynamic(df, spot, yf_sym, straddle_now):
    """
    Straddle chart = two parts:
    1. Straddle across strikes (how much does straddle cost at each strike?)
    2. Historical context: estimate from price history
    """
    figs = []
    texts = []

    # Part 1: Straddle across strikes (from chain data)
    if df is not None and not df.empty and "CE_LTP" in df.columns and "PE_LTP" in df.columns:
        rng = spot * 0.04
        sub = df[df["strikePrice"].between(spot-rng, spot+rng)].copy()
        if sub.empty: sub = df.copy()
        sub["straddle"] = sub["CE_LTP"] + sub["PE_LTP"]
        sub["strangle_10"] = (sub.get("CE_LTP",0) + sub.get("PE_LTP",0))

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(name="CE Premium",x=sub["strikePrice"],y=sub["CE_LTP"],
                              marker_color="#00ff88",opacity=0.8))
        fig1.add_trace(go.Bar(name="PE Premium",x=sub["strikePrice"],y=sub["PE_LTP"],
                              marker_color="#ff3b5c",opacity=0.8))
        fig1.add_trace(go.Scatter(name="Straddle (CE+PE)",x=sub["strikePrice"],y=sub["straddle"],
                                  mode="lines+markers",line=dict(color="yellow",width=2.5)))
        fig1.add_vline(x=spot,line_color="#00e5ff",line_dash="dash",
                       annotation_text=f"Spot ₹{spot:,.0f}",annotation_font_color="#00e5ff")
        atm_straddle = sub.loc[(sub["strikePrice"]-spot).abs().idxmin(),"straddle"] if not sub.empty else 0
        if atm_straddle > 0:
            fig1.add_annotation(x=spot,y=atm_straddle,
                                text=f"ATM straddle = ₹{atm_straddle:.2f}",
                                showarrow=True,arrowhead=2,
                                font=dict(color="yellow",size=12),
                                bgcolor="#1a1a00",bordercolor="yellow")
        fig1.update_layout(template=DARK,height=320,barmode="stack",
                           title="Straddle Cost Across Strikes (Live from Chain)",
                           xaxis_title="Strike Price",yaxis_title="Premium ₹")
        figs.append(fig1)
        texts.append(
            "The yellow line shows the <b>total straddle cost</b> (CE + PE) at each strike. "
            "The ATM strike (closest to current price) usually has the highest combined premium. "
            f"<b>ATM straddle = ₹{atm_straddle:.2f}</b> — this is what you pay to be positioned for a move in either direction. "
            "If the market moves more than this amount by expiry, the straddle buyer profits. "
            "If the market moves less, the straddle seller profits. "
            "Cheap straddle (below historical average) = options underpriced = buy. Expensive = overpriced = sell or wait."
        )

    # Part 2: Historical estimated straddle
    hist = get_history(yf_sym, "60d")
    if not hist.empty:
        close = hist["Close"].squeeze().astype(float)
        rv7   = close.pct_change().rolling(7).std() * np.sqrt(252)
        est   = (close * rv7 * np.sqrt(7/252) * 0.8).dropna()
        if not est.empty:
            p10,p25,p50,p75,p90 = est.quantile([.10,.25,.50,.75,.90])
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=est.index,y=est,name="Estimated ATM straddle",
                                      line=dict(color="#00e5ff",width=2),fill="tozeroy",
                                      fillcolor="rgba(0,229,255,0.04)"))
            for val,lbl,clr in [(p10,"Cheap P10","#00ff88"),(p25,"Low P25","#66cc66"),
                                 (p50,"Average","#ff9500"),(p75,"High P75","#ff6633"),(p90,"Expensive P90","#ff3b5c")]:
                fig2.add_hline(y=val,line_color=clr,line_dash="dot",
                               annotation_text=f"{lbl}: ₹{val:.0f}",annotation_font_color=clr)
            if straddle_now > 0:
                fig2.add_hline(y=straddle_now,line_color="yellow",line_dash="solid",line_width=2,
                               annotation_text=f"Today: ₹{straddle_now:.0f}",annotation_font_color="yellow")
            fig2.update_layout(template=DARK,height=300,
                               title="ATM Straddle vs 60-Day Historical Range",
                               xaxis_title="Date",yaxis_title="Estimated straddle ₹")
            figs.append(fig2)

            # Dynamic valuation
            if straddle_now > 0:
                pctile = (est < straddle_now).mean()*100
                if pctile < 25:
                    val_msg = (f"✅ Today's straddle ₹{straddle_now:.0f} is at the <b>{pctile:.0f}th percentile</b> — "
                               f"CHEAP vs 60-day history. You're paying less than you would on {100-pctile:.0f}% of days. "
                               f"Best conditions to buy options.")
                    val_clr = "#00ff88"
                elif pctile > 75:
                    val_msg = (f"⚠️ Today's straddle ₹{straddle_now:.0f} is at the <b>{pctile:.0f}th percentile</b> — "
                               f"EXPENSIVE vs 60-day history. You're overpaying. "
                               f"IV will likely fall after the next move, hurting your premium even if direction is right.")
                    val_clr = "#ff3b5c"
                else:
                    val_msg = (f"🟡 Today's straddle ₹{straddle_now:.0f} is at the <b>{pctile:.0f}th percentile</b> — "
                               f"FAIR VALUE vs history. Only buy on strong directional signals.")
                    val_clr = "#ff9500"
            else:
                val_msg = f"Historical straddle range: P10=₹{p10:.0f}, Avg=₹{p50:.0f}, P90=₹{p90:.0f}."
                val_clr = "#5a7a9a"

            texts.append(
                f"This chart shows how today's straddle cost compares to the last 60 days. "
                f"The coloured lines are percentile bands — <b>P10 = cheap zone</b> (bottom 10% of days), "
                f"<b>P90 = expensive zone</b> (top 10% of days). <br>{val_msg}"
            )
            return figs, texts, val_msg, val_clr

    return figs, texts, "", "#5a7a9a"


# ═══════════════════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def run_backtest(yf_sym,lookback,capital,pos_pct):
    raw=get_history(yf_sym,f"{lookback}d")
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
        pcr=1.0+(v10-v3)/(v10+1)
        mp=(float(sma10.iloc[i])+float(sma20.iloc[i]))/2
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
        trades.append({"Date":raw.index[i].strftime("%d-%b-%Y"),"Dir":direction,
            "Spot":round(spot,2),"OptPx":round(opt_px,2),"IV%":round(iv,1),
            "Score":score,"Move%":round(ret*100,2),"P&L₹":round(pnl,2),
            "P&L%":round(pnl_pct*100,2),"Equity":round(equity,2),
            "Result":"✅ WIN" if pnl>0 else "❌ LOSS"})
        curve[-1]=equity
    if not trades: return None,"No trades taken — try longer lookback."
    tdf=pd.DataFrame(trades); wins=(tdf["P&L₹"]>0).sum(); total=len(tdf)
    aw=tdf.loc[tdf["P&L₹"]>0,"P&L₹"].mean() if wins>0 else 0
    al=tdf.loc[tdf["P&L₹"]<=0,"P&L₹"].mean() if (total-wins)>0 else 0
    peak=capital; mdd=0
    for eq in curve:
        if eq>peak: peak=eq
        mdd=max(mdd,(peak-eq)/peak*100)
    stats={"total":total,"wins":int(wins),"wr":round(wins/total*100,1),
           "pnl":round(tdf["P&L₹"].sum(),2),"aw":round(aw,2),"al":round(al,2),
           "rr":round(abs(aw/al),2) if al else 0,"mdd":round(mdd,2),
           "final":round(equity,2),"ret%":round((equity-capital)/capital*100,2),"curve":curve}
    return tdf,stats


# ═══════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════
def _vl(fig,x,lbl="",color="yellow",dash="dash",row=1,col=1):
    fig.add_vline(x=x,line_dash=dash,line_color=color,line_width=1.5,
                  annotation_text=lbl,annotation_font_color=color,row=row,col=col)

def plot_chain(df, spot):
    rng=spot*0.05; sub=df[df["strikePrice"].between(spot-rng,spot+rng)]
    if sub.empty: sub=df
    fig=make_subplots(2,2,subplot_titles=["CE vs PE Premium","Open Interest","IV Smile","Volume"])
    x=sub["strikePrice"]
    fig.add_trace(go.Bar(name="CE LTP",x=x,y=sub["CE_LTP"],marker_color="#00ff88",opacity=.85),1,1)
    fig.add_trace(go.Bar(name="PE LTP",x=x,y=sub["PE_LTP"],marker_color="#ff3b5c",opacity=.85),1,1)
    fig.add_trace(go.Bar(name="CE OI",x=x,y=sub["CE_OI"],marker_color="#00e5ff",opacity=.8),1,2)
    fig.add_trace(go.Bar(name="PE OI",x=x,y=sub["PE_OI"],marker_color="#ff9500",opacity=.8),1,2)
    fig.add_trace(go.Scatter(name="CE IV%",x=x,y=sub["CE_IV"],mode="lines+markers",line=dict(color="#00ff88",width=2)),2,1)
    fig.add_trace(go.Scatter(name="PE IV%",x=x,y=sub["PE_IV"],mode="lines+markers",line=dict(color="#ff3b5c",width=2)),2,1)
    fig.add_trace(go.Bar(name="CE Vol",x=x,y=sub["CE_volume"],marker_color="rgba(0,255,136,0.4)"),2,2)
    fig.add_trace(go.Bar(name="PE Vol",x=x,y=sub["PE_volume"],marker_color="rgba(255,59,92,0.4)"),2,2)
    for r in [1,2]:
        for c in [1,2]: _vl(fig,spot,"Spot",row=r,col=c)
    fig.update_layout(template=DARK,height=520,barmode="group",
                      title="Option Chain Overview",margin=dict(t=50,b=10))
    return fig

CHAIN_EXPLAIN = """
<b>Top-left — Premium:</b> How much each strike costs. Calls (CE, green) are expensive when the strike is below spot (in-the-money). 
Puts (PE, red) are expensive when above spot. ATM options (closest to spot) have the most time value and highest activity.<br>
<b>Top-right — OI (Open Interest):</b> Total contracts outstanding. Peaks in CE OI = resistance walls (sellers defend this). 
Peaks in PE OI = support walls. These are the key levels the market will fight at near expiry.<br>
<b>Bottom-left — IV Smile:</b> Implied volatility at each strike. Normally OTM puts have higher IV (fear premium). 
If the smile is steep on the put side, big money is paying for downside protection (bearish). 
If calls are expensive, aggressive bullish positioning.<br>
<b>Bottom-right — Volume:</b> Today's trading activity. High volume at a strike = strong interest = the market is actively betting on that level.
"""

def plot_oi_chart(df, spot, sig):
    rng=spot*0.055; sub=df[df["strikePrice"].between(spot-rng,spot+rng)]
    if sub.empty: sub=df
    sub=sub.copy()
    prev_ce=(sub["CE_OI"]-sub["CE_changeOI"]).clip(lower=1)
    prev_pe=(sub["PE_OI"]-sub["PE_changeOI"]).clip(lower=1)
    sub["CE_pct"]=(sub["CE_changeOI"]/prev_ce*100).fillna(0)
    sub["PE_pct"]=(sub["PE_changeOI"]/prev_pe*100).fillna(0)
    fig=make_subplots(3,1,subplot_titles=["Total OI — resistance & support walls","OI Added/Removed Today","% Change in OI Today (intensity of positioning)"],
                      vertical_spacing=0.1)
    x=sub["strikePrice"]
    fig.add_trace(go.Bar(name="CE OI",x=x,y=sub["CE_OI"],marker_color="#00e5ff",opacity=.8),1,1)
    fig.add_trace(go.Bar(name="PE OI",x=x,y=sub["PE_OI"],marker_color="#ff9500",opacity=.8),1,1)
    fig.add_trace(go.Bar(name="ΔCE",x=x,y=sub["CE_changeOI"],marker_color=["#00ff88" if v>=0 else "#ff3b5c" for v in sub["CE_changeOI"]]),2,1)
    fig.add_trace(go.Bar(name="ΔPE",x=x,y=sub["PE_changeOI"],marker_color=["#ff9500" if v>=0 else "#8888ff" for v in sub["PE_changeOI"]]),2,1)
    fig.add_trace(go.Bar(name="CE%",x=x,y=sub["CE_pct"],marker_color=["rgba(0,229,255,0.6)" if v>=0 else "rgba(255,59,92,0.5)" for v in sub["CE_pct"]]),3,1)
    fig.add_trace(go.Bar(name="PE%",x=x,y=sub["PE_pct"],marker_color=["rgba(255,149,0,0.6)" if v>=0 else "rgba(136,136,255,0.5)" for v in sub["PE_pct"]]),3,1)
    for row in [1,2,3]:
        _vl(fig,spot,"Spot",row=row)
        _vl(fig,sig["resistance"],"Resistance",color="#ff3b5c",dash="dot",row=row,col=1)
        _vl(fig,sig["support"],"Support",color="#00ff88",dash="dot",row=row,col=1)
    fig.update_layout(template=DARK,height=680,barmode="group",title="OI Analysis",margin=dict(t=50,b=10))
    return fig

OI_EXPLAIN = """
<b>Panel 1 — Total OI:</b> Blue (CE) = call sellers' total position. Orange (PE) = put sellers. 
The tallest blue bar = strongest resistance (sellers will aggressively defend this). The tallest orange bar = strongest support. 
Watch if spot approaches these walls — sellers defend them hard, creating reversals.<br>
<b>Panel 2 — OI Change today:</b> Green = new OI added (fresh positions). Red = OI removed (positions closed). 
Fresh CE addition at a strike = new call sellers building a resistance wall. Fresh PE addition = new put sellers building support.
If OI increases WITH price rising = long buildup (bullish). If OI increases WITH price falling = short buildup (bearish).<br>
<b>Panel 3 — % Change in OI:</b> Shows WHERE the most aggressive new positioning is happening today. 
A 200% spike means that strike's OI tripled today — someone is making a massive directional bet here. 
Track this to see where smart money is building today.
"""

def plot_greeks_chart(df, spot):
    rng=spot*0.04; sub=df[df["strikePrice"].between(spot-rng,spot+rng)]
    if sub.empty: sub=df
    fig=make_subplots(2,2,subplot_titles=["Delta","Gamma","Theta (daily ₹ lost)","Vega"])
    x=sub["strikePrice"]
    for (r,c,cc,pc) in [(1,1,"CE_delta","PE_delta"),(1,2,"CE_gamma","PE_gamma"),(2,1,"CE_theta","PE_theta"),(2,2,"CE_vega","PE_vega")]:
        for col,clr in [(cc,"#00ff88"),(pc,"#ff3b5c")]:
            if col in sub.columns:
                fig.add_trace(go.Scatter(name=col,x=x,y=sub[col],mode="lines",line=dict(color=clr,width=2)),r,c)
        _vl(fig,spot,row=r,col=c)
    fig.update_layout(template=DARK,height=520,title="Greeks",margin=dict(t=50,b=10))
    return fig

GREEKS_EXPLAIN = """
<b>Delta:</b> For every ₹100 move in the underlying, your option gains/loses (Delta × ₹100). 
CE delta = 0.5 means ₹50 gain for every ₹100 market rise. ATM = ~0.5 delta. Deep ITM = ~1.0. Deep OTM = ~0.1.<br>
<b>Gamma:</b> How fast Delta itself is changing. High gamma near ATM means your delta can jump quickly on a move — 
this is good for buyers (positions can accelerate) but dangerous if the market moves against you.<br>
<b>Theta:</b> The daily rent you pay for owning an option. If CE theta = -₹50, you lose ₹50 per day even if the market doesn't move. 
This accelerates sharply near expiry. Always compare theta to expected move — only hold if expected gain > theta cost.<br>
<b>Vega:</b> How much your option gains if implied volatility rises by 1%. Buy before events (IV rises = you gain). 
Sell after events (IV collapses = you gain). High vega options benefit most from fear spikes.
"""


# ═══════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════
def main():
    st.markdown("<h1 style='font-size:24px;letter-spacing:2px'>⚡ PRO OPTIONS — NSE + BSE INDIA</h1>",unsafe_allow_html=True)
    st.markdown("<p style='color:#5a7a9a;margin-top:-10px'>NIFTY · BANKNIFTY · SENSEX (BSE) · 80 F&O stocks · Live + CSV + BSE paste · Full analysis</p>",unsafe_allow_html=True)
    st.divider()

    # ── SIDEBAR ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Market")
        market=st.selectbox("",["🇮🇳 NSE India","🇮🇳 BSE/SENSEX","🌐 Global (yFinance)","📊 Spot Only (BTC/Gold)"],label_visibility="collapsed")
        fetch_sym=""; yf_sym=""; is_nse=False; is_bse=False; has_opts=True

        if "NSE" in market and "BSE" not in market:
            is_nse=True
            cat=st.radio("Category",["Indices","F&O Stocks"],horizontal=True)
            if cat=="Indices":
                nse_sym=st.selectbox("Index",NSE_INDICES)
            else:
                nse_sym=st.selectbox("Stock",NSE_FNO)
                if nse_sym=="Custom symbol…":
                    nse_sym=st.text_input("NSE symbol","RELIANCE").upper().strip()
            fetch_sym=nse_sym; yf_sym=NSE_TO_YF.get(nse_sym,nse_sym+".NS")
            with st.expander("ℹ️ Live sources"):
                st.markdown("Tries: nsepython → NSE-A → NSE-B → NSE-C → Opstra")
            st.code("pip install nsepython",language="bash")

        elif "BSE" in market:
            is_bse=True
            bse_sym=st.selectbox("BSE Index",["SENSEX","BANKEX"])
            fetch_sym=bse_sym; yf_sym=NSE_TO_YF.get(bse_sym,"^BSESN")

        elif "Global" in market:
            ch=st.selectbox("Instrument",list(YF_OPTS.keys()))
            fetch_sym=st.text_input("Ticker","AAPL").upper() if ch=="Custom" else YF_OPTS[ch]
            yf_sym=fetch_sym
        else:
            ch=st.selectbox("Instrument",list(YF_SPOT.keys()))
            fetch_sym=st.text_input("Ticker","BTC-USD").upper() if ch=="Custom" else YF_SPOT[ch]
            yf_sym=fetch_sym; has_opts=False

        st.divider()

        # NSE CSV upload
        if is_nse:
            st.markdown("### 📂 NSE CSV Fallback")
            st.markdown('<span style="font-size:12px;color:#5a7a9a">Download from nseindia.com → Option Chain → ↓ CSV button</span>',unsafe_allow_html=True)
            csv_file=st.file_uploader("Upload NSE CSV",type=["csv"],key="csv_nse")
        else:
            csv_file=None

        # BSE paste
        if is_bse:
            st.markdown("### 📋 BSE Data Paste")
            st.markdown('<span style="font-size:12px;color:#5a7a9a">Go to bseindia.com → Derivatives → Select expiry → Select all table → Copy → Paste below</span>',unsafe_allow_html=True)
            bse_expiry=st.text_input("Expiry date (dd-Mon-yyyy)","06-Mar-2026",key="bse_exp")
            bse_paste=st.text_area("Paste BSE option chain data here",height=160,key="bse_paste",
                                   placeholder="CALLS\tSTRIKE PRICE\tPUTS\nChg in OI\tOI\t...")
        else:
            bse_paste=None; bse_expiry=""

        st.divider()
        st.markdown("### 💰 Risk")
        capital=st.number_input("Capital ₹",50000,10000000,100000,10000)
        pos_pct=st.slider("% per trade",2,15,5)
        sl_pct =st.slider("Stop loss %",20,50,30)
        tgt_pct=st.slider("Target %",30,200,60)
        st.divider()
        st.button("🔄 Refresh",type="primary",use_container_width=True)
        if st.button("🗑️ Clear Cache",use_container_width=True):
            st.cache_data.clear(); _T["last"]=0.0; st.rerun()
        auto_ref=st.checkbox("Auto-refresh 90s")
        st.caption("Educational only · Not financial advice")

    # ── DATA FETCH ────────────────────────────────────────────────────
    df_exp=pd.DataFrame(); spot=0.0; expiries=[]; sel_expiry=""
    data_src=""; fetch_errors=[]
    ph=st.empty()

    if is_nse:
        # CSV path
        if csv_file is not None:
            ph.info("📂 Parsing NSE CSV…")
            raw=csv_file.read(); fname=getattr(csv_file,"name","")
            df_csv,csv_dbg=parse_nse_csv(raw,filename=fname)
            if not df_csv.empty:
                df_exp=df_csv; data_src="CSV"
                spot=get_spot(yf_sym)
                expiries=df_csv["expiryDate"].unique().tolist(); sel_expiry=expiries[0]
                ph.success(f"✅ CSV — {len(df_exp)} strikes | Expiry: {sel_expiry} | Spot ₹{spot:,.2f}")
                with st.expander("CSV parse details"): [st.text(l) for l in csv_dbg]
            else:
                ph.error("❌ CSV parse failed")
                with st.expander("CSV debug"): [st.text(l) for l in csv_dbg]

        if df_exp.empty:
            ph.info("📡 Fetching NSE live (5 sources)…")
            with st.spinner("Contacting NSE…"):
                df_raw,spot_raw,exps_raw,src,errs=fetch_nse_chain(fetch_sym)
            fetch_errors=errs
            if df_raw is not None and spot_raw>0:
                df_exp=df_raw; spot=spot_raw; expiries=exps_raw; data_src=src
                ph.success(f"✅ Live via **{src}** — {len(df_exp)} strikes | Spot ₹{spot:,.2f}")
            else:
                spot=get_spot(yf_sym); has_opts=False
                ph.warning(f"⚠️ All live sources failed. Spot ₹{spot:,.2f} from yFinance only.")
                for e in fetch_errors: st.markdown(f"- {e}")
                st.info("**Fix:** `pip install nsepython` → restart. Or upload NSE CSV from sidebar.")

        if not df_exp.empty and expiries:
            with st.sidebar:
                sel_expiry=st.selectbox("Expiry",expiries[:10])
            if data_src!="CSV" and "expiryDate" in df_exp.columns:
                mask=df_exp["expiryDate"]==sel_expiry
                df_exp=df_exp[mask].copy() if mask.any() else df_exp.copy()

    elif is_bse:
        # Paste path first
        if bse_paste and bse_paste.strip():
            ph.info("📋 Parsing BSE pasted data…")
            df_paste,p_dbg=parse_bse_paste(bse_paste,expiry=bse_expiry)
            if not df_paste.empty:
                df_exp=df_paste; data_src="BSE-paste"
                spot=get_spot(yf_sym)
                expiries=[bse_expiry]; sel_expiry=bse_expiry
                ph.success(f"✅ BSE paste — {len(df_exp)} strikes | Expiry: {sel_expiry} | Spot ₹{spot:,.2f}")
                with st.expander("Paste parse details"): [st.text(l) for l in p_dbg]
            else:
                ph.error("❌ BSE paste parse failed")
                with st.expander("Debug"): [st.text(l) for l in p_dbg]

        if df_exp.empty:
            ph.info("📡 Trying BSE live API…")
            with st.spinner("Contacting BSE…"):
                df_raw,spot_raw,exps_raw,src,errs=fetch_bse_chain(fetch_sym)
            if df_raw is not None and spot_raw>0:
                df_exp=df_raw; spot=spot_raw; expiries=exps_raw; data_src=src
                ph.success(f"✅ BSE live — {len(df_exp)} strikes | Spot ₹{spot:,.2f}")
                if expiries:
                    with st.sidebar:
                        sel_expiry=st.selectbox("Expiry",expiries[:8])
                    mask=df_exp["expiryDate"]==sel_expiry
                    df_exp=df_exp[mask].copy() if mask.any() else df_exp.copy()
            else:
                spot=get_spot(yf_sym); has_opts=False
                ph.warning(f"⚠️ BSE live failed. Spot ₹{spot:,.2f} from yFinance.")
                for e in errs: st.markdown(f"- {e}")
                st.info("**Use BSE paste:** Go to bseindia.com → Derivatives, select all table data, paste in sidebar.")

    elif has_opts:
        ph.info("📡 Fetching from yFinance…")
        spot=get_spot(fetch_sym)
        if spot>0 and _HAS_YF:
            _wait(1.2)
            try: exps=list(yf.Ticker(fetch_sym).options or [])
            except: exps=[]
            if exps:
                with st.sidebar:
                    sel_expiry=st.selectbox("Expiry",exps[:8])
                calls,puts=get_yf_chain(fetch_sym,sel_expiry)
                if calls is not None and not calls.empty:
                    df_exp=build_yf_df(calls,puts,sel_expiry); data_src="yFinance"
                    ph.success(f"✅ {fetch_sym} — {len(df_exp)} strikes | Spot {spot:,.2f}")
                else: ph.error("Chain fetch failed."); has_opts=False
            else: ph.warning("No options available."); has_opts=False
        elif spot<=0: ph.error(f"Cannot get price for {fetch_sym}.")
    else:
        spot=get_spot(fetch_sym)
        if spot>0: ph.success(f"✅ {fetch_sym}: {spot:,.4f}")
        else: ph.error(f"Cannot get {fetch_sym}.")

    if spot==0: return
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

    src_cls={"nsepython":"src-a","NSE-A":"src-b","NSE-B":"src-b","NSE-C":"src-b",
             "Opstra":"src-c","CSV":"src-d","BSE-API":"src-b","BSE-paste":"src-c",
             "yFinance":"src-b","":""}
    badge=f'<span class="src {src_cls.get(data_src,"src-b")}">{data_src}</span>' if data_src else ""

    # ── KEY METRICS ──────────────────────────────────────────────────
    c1,c2,c3,c4,c5,c6,c7=st.columns(7)
    c1.metric("📍 Spot",f"₹{spot:,.2f}")
    c2.metric("🎯 ATM",f"₹{sig['atm']:,.0f}" if has_chain else "—")
    c3.metric("📊 PCR",f"{sig['pcr']:.3f}" if has_chain else "—")
    c4.metric("💀 Max Pain",f"₹{sig['max_pain']:,.0f}" if has_chain else "—")
    c5.metric("🌡 ATM IV",f"{sig['atm_iv']:.1f}%" if has_chain else "—")
    c6.metric("↕ Skew",f"{sig['skew']:+.1f}%" if has_chain else "—")
    c7.metric("♟ Straddle",f"₹{sig['straddle']:.2f}" if has_chain else "—")

    # ── SIGNAL BANNER ────────────────────────────────────────────────
    conf=sig["conf"]; bc="#00ff88" if conf>=72 else "#ff9500" if conf>=58 else "#5a7a9a"
    def tag(s): return f'<span class="tag-b">{s}</span>' if "BUY" in s else f'<span class="tag-w">{s}</span>'

    if has_chain:
        st.markdown(f"""<div class="sig-box">
<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;align-items:flex-start">
  <div style="flex:2">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
      <span style="color:#5a7a9a;font-size:10px;letter-spacing:2px">SIGNAL</span>{badge}
    </div>
    <div style="font-size:26px;font-weight:700;color:{bc};font-family:Space Mono">{sig['rec']}</div>
    <div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:6px">
      {tag("⚡ Scalp: "+sig['scalp'])} {tag("📅 Intraday: "+sig['intraday'])}
      {tag("🗓 Swing: "+sig['swing'])} {tag("📆 Positional: "+sig['pos'])}
    </div>
  </div>
  <div style="text-align:center;min-width:130px">
    <div style="color:#5a7a9a;font-size:10px;letter-spacing:2px">CONFIDENCE</div>
    <div style="font-size:48px;font-weight:700;color:{bc};font-family:Space Mono;line-height:1.1">{conf}%</div>
    <div style="background:{bc}22;border-radius:20px;height:6px;width:120px;margin:6px auto 0">
      <div style="background:{bc};border-radius:20px;height:6px;width:{conf}%"></div>
    </div>
    <div style="font-size:11px;color:#5a7a9a;margin-top:4px">{'STRONG SIGNAL' if conf>=72 else 'WATCH' if conf>=58 else 'NO EDGE'}</div>
  </div>
</div>
</div>""",unsafe_allow_html=True)

        # ── PER-INDICATOR TABLE ──
        st.markdown("#### 📊 Why this signal? — Indicator breakdown")
        for (name,val,verdict,pts,exp) in sig["indicators"]:
            clr=("#00ff88" if "BULLISH" in verdict or "CHEAP" in verdict or "FRESH" in verdict or "✅" in verdict
                 else "#ff3b5c" if "BEAR" in verdict or "EXPENSIVE" in verdict or "AGAINST" in verdict
                 else "#ff9500" if "WATCH" in verdict or "MODER" in verdict
                 else "#5a7a9a")
            pts_clr="#00ff88" if "+" in str(pts) and "-" not in str(pts) else "#ff3b5c" if "-" in str(pts) else "#5a7a9a"
            st.markdown(f"""<div style="background:#0a1929;border:1px solid #1a3050;border-radius:8px;padding:12px 16px;margin:5px 0">
<div class="ind-row">
  <span style="color:#9ab5cc;font-weight:600;min-width:200px">{name}</span>
  <span class="ind-val" style="color:#c8d6e5;min-width:140px">{val}</span>
  <span style="color:{clr};font-weight:600;min-width:220px">{verdict}</span>
  <span style="color:{pts_clr};font-family:Space Mono;font-size:13px">{pts} pts</span>
</div>
<div style="color:#7a9ab5;font-size:12px;margin-top:6px;line-height:1.6">{exp}</div>
</div>""",unsafe_allow_html=True)

        for ab in sig["abnormal"]:
            st.markdown(f'<div class="alert">{ab}</div>',unsafe_allow_html=True)
        if sig["gamma_blast"]:
            st.markdown('<div class="gamma-blast">⚡ <b style="color:#ff3b5c">GAMMA BLAST</b> — Huge OI at current price. Options explode on breakout. Enter fast on direction confirmation.</div>',unsafe_allow_html=True)
    else:
        st.info(f"📊 Spot only — ₹{spot:,.2f}. Use CSV/paste or install nsepython for full analysis.")

    st.markdown("<br>",unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────────────
    tabs=st.tabs(["📊 Chain","🔢 Greeks","📈 OI Analysis","📐 Straddle","⚡ Trade","🔬 Backtest","📋 History","🧠 Analysis"])

    with tabs[0]:
        if not has_chain:
            st.warning("Chain not loaded. Upload CSV / paste BSE data / install nsepython.")
        else:
            def_st=all_strikes[max(0,atm_pos-8):atm_pos+9]
            sel_st=st.multiselect("Strikes to display",all_strikes,default=def_st)
            if not sel_st: sel_st=def_st
            show_c=[c for c in ["strikePrice","CE_LTP","CE_OI","CE_changeOI","CE_IV","CE_volume","PE_LTP","PE_OI","PE_changeOI","PE_IV","PE_volume"] if c in df_exp.columns]
            st.dataframe(df_exp[df_exp["strikePrice"].isin(sel_st)][show_c].round(2),use_container_width=True,height=260)
            st.plotly_chart(plot_chain(df_exp,spot),use_container_width=True)
            st.markdown(f'<div class="plot-explain">{CHAIN_EXPLAIN}</div>',unsafe_allow_html=True)

    with tabs[1]:
        if not has_chain: st.warning("Needs chain data.")
        else:
            st.plotly_chart(plot_greeks_chart(df_exp,spot),use_container_width=True)
            st.markdown(f'<div class="plot-explain">{GREEKS_EXPLAIN}</div>',unsafe_allow_html=True)
            ai=(df_exp["strikePrice"]-spot).abs().idxmin(); atm2=df_exp.loc[ai]
            g1,g2=st.columns(2)
            for col,px,lbl,clr in [(g1,"CE","📗 CALL — UP","#00ff88"),(g2,"PE","📕 PUT — DOWN","#ff3b5c")]:
                with col:
                    st.markdown(f"<h5 style='color:{clr}'>{lbl} | ATM Strike ₹{int(atm2['strikePrice']):,}</h5>",unsafe_allow_html=True)
                    cs=st.columns(3)
                    items=[("Delta",f"{px}_delta","₹ move/₹100"),("Gamma",f"{px}_gamma","Delta speed"),
                           ("Theta",f"{px}_theta","₹ lost/day"),("Vega",f"{px}_vega","₹ on IV+1%"),("IV%",f"{px}_IV","current IV")]
                    for i,(n,k,t) in enumerate(items):
                        val=float(atm2.get(k,0))
                        cs[i%3].metric(n,f"{val:.2f}{'%' if 'IV' in n else ''}",t)
            # Theta vs Vega verdict
            ce_th=abs(float(atm2.get("CE_theta",0))); ce_ve=float(atm2.get("CE_vega",0))
            if ce_ve > ce_th:
                st.markdown(f'<div style="background:#002a15;border-left:3px solid #00ff88;padding:10px 14px;border-radius:0 6px 6px 0;margin:6px 0">✅ Vega ₹{ce_ve:.2f} > Theta ₹{ce_th:.2f}/day: Volatility gains outpace time decay. Good buying conditions.</div>',unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="background:#1a0a00;border-left:3px solid #ff9500;padding:10px 14px;border-radius:0 6px 6px 0;margin:6px 0">⚠️ Theta ₹{ce_th:.2f}/day > Vega ₹{ce_ve:.2f}: Time decay is eroding value faster than volatility helps. Only hold on very strong signals.</div>',unsafe_allow_html=True)

    with tabs[2]:
        if not has_chain: st.warning("Needs chain data.")
        else:
            st.plotly_chart(plot_oi_chart(df_exp,spot,sig),use_container_width=True)
            st.markdown(f'<div class="plot-explain">{OI_EXPLAIN}</div>',unsafe_allow_html=True)
            narrative=oi_narrative(df_exp,spot,sig)
            if narrative:
                st.markdown(f'<div class="oi-sum">{narrative}</div>',unsafe_allow_html=True)
            st.markdown("#### Top walls")
            b1,b2=st.columns(2)
            with b1:
                st.markdown("**🔴 CE OI — resistance walls**")
                ct=df_exp.nlargest(5,"CE_OI")[["strikePrice","CE_OI","CE_changeOI","CE_LTP"]].copy()
                prev=ct["CE_OI"]-ct["CE_changeOI"]
                ct["Chg%"]=((ct["CE_changeOI"]/prev.clip(lower=1))*100).round(1).astype(str)+"%"
                ct["Signal"]=ct["CE_changeOI"].apply(lambda x:"🔴 Fresh resistance" if x>0 else "🟡 Weakening" if x<0 else "—")
                st.dataframe(ct,use_container_width=True)
            with b2:
                st.markdown("**🟢 PE OI — support walls**")
                pt=df_exp.nlargest(5,"PE_OI")[["strikePrice","PE_OI","PE_changeOI","PE_LTP"]].copy()
                prev=pt["PE_OI"]-pt["PE_changeOI"]
                pt["Chg%"]=((pt["PE_changeOI"]/prev.clip(lower=1))*100).round(1).astype(str)+"%"
                pt["Signal"]=pt["PE_changeOI"].apply(lambda x:"🟢 Fresh support" if x>0 else "🟡 Weakening" if x<0 else "—")
                st.dataframe(pt,use_container_width=True)

    with tabs[3]:
        if not has_chain:
            st.warning("Straddle analysis needs chain data.")
        else:
            st.markdown("### 📐 Straddle Analysis")
            st.markdown("""<div class="explain"><b>Straddle</b> = buy CE + PE at the same strike.
You don't need to predict direction — just bet that the market will MOVE a lot.
If market stays flat, you lose the premium paid. If it moves more than the straddle cost, you profit.<br>
Use this to gauge: are options cheap or expensive right now?</div>""",unsafe_allow_html=True)
            figs,texts,val_msg,val_clr=plot_straddle_dynamic(df_exp,spot,yf_sym,sig["straddle"])
            for fig,txt in zip(figs,texts):
                st.plotly_chart(fig,use_container_width=True)
                st.markdown(f'<div class="plot-explain">{txt}</div>',unsafe_allow_html=True)
            if val_msg:
                st.markdown(f'<div style="background:{val_clr}11;border:1px solid {val_clr}44;border-radius:8px;padding:14px;color:{val_clr};font-weight:600;font-size:15px;margin:10px 0">{val_msg}</div>',unsafe_allow_html=True)

    with tabs[4]:
        st.markdown("""<div class="explain">Paper trade — no real money, just practice.
CE = UP trade · PE = DOWN trade · Always set SL before entering · Never average losers · R:R should be above 1.5
</div>""",unsafe_allow_html=True)
        if not has_chain: st.warning("Needs chain.")
        else:
            la,lb=st.columns([3,2])
            with la:
                st.markdown("#### New Trade")
                t1,t2=st.columns(2)
                side=t1.selectbox("Direction",["CE — Buy Call (UP)","PE — Buy Put (DOWN)"])
                px_="CE" if "CE" in side else "PE"
                t_str=t1.selectbox("Strike",all_strikes,index=atm_pos)
                lots=t2.number_input("Lots",1,100,1)
                l_sl=t2.slider("SL %",20,50,sl_pct,key="l_sl")
                l_tgt=t2.slider("Target %",30,200,tgt_pct,key="l_tgt")
                r_row=df_exp[df_exp["strikePrice"]==t_str]; lc=f"{px_}_LTP"
                opt_px=float(r_row[lc].values[0]) if not r_row.empty and lc in r_row.columns else 0
                if opt_px>0:
                    sl_a=round(opt_px*(1-l_sl/100),2); tgt_a=round(opt_px*(1+l_tgt/100),2)
                    risk=(opt_px-sl_a)*lots; rew=(tgt_a-opt_px)*lots; rr=round(rew/risk,2) if risk>0 else 0
                    pm1,pm2,pm3,pm4=st.columns(4)
                    pm1.metric("Entry",f"₹{opt_px:.2f}"); pm2.metric("SL",f"₹{sl_a:.2f}",f"-{l_sl}%")
                    pm3.metric("Target",f"₹{tgt_a:.2f}",f"+{l_tgt}%"); pm4.metric("R:R",f"1:{rr}")
                    if rr<1.5: st.warning("⚠️ R:R < 1.5 — poor setup. Increase target or reduce SL.")
                else: sl_a=tgt_a=0; st.warning("Price=0 — market closed or too far OTM")
                if st.button("📈 Enter Paper Trade",type="primary",use_container_width=True):
                    if opt_px>0:
                        t={"id":len(st.session_state.trades)+1,"sym":fetch_sym,"expiry":sel_expiry,
                           "strike":t_str,"side":px_,"entry":opt_px,"sl":sl_a,"target":tgt_a,
                           "lots":lots,"conf":sig["conf"],"rec":sig["rec"],
                           "time":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "status":"OPEN","exit":None,"pnl":None}
                        st.session_state.active.append(t); st.session_state.trades.append(t)
                        st.success(f"✅ {fetch_sym} {t_str} {px_} @ ₹{opt_px:.2f}")
                    else: st.error("Cannot enter — price=0")
            with lb:
                st.markdown("#### Open Positions")
                if not [t for t in st.session_state.active if t["status"]=="OPEN"]: st.info("No open trades.")
                for i,t in enumerate(st.session_state.active):
                    if t["status"]!="OPEN": continue
                    rr=df_exp[df_exp["strikePrice"]==t["strike"]]; lc_=f"{t['side']}_LTP"
                    curr=float(rr[lc_].values[0]) if not rr.empty and lc_ in rr.columns else t["entry"]
                    pnl=round((curr-t["entry"])*t["lots"],2); pp=round((curr-t["entry"])/t["entry"]*100,2) if t["entry"]>0 else 0
                    clr_="#00ff88" if pnl>=0 else "#ff3b5c"; cls="tc-w" if pnl>0 else "tc-l" if pnl<0 else "tc-o"
                    warn=("⚠️ <b style='color:#ff3b5c'>SL HIT — EXIT NOW</b><br>" if curr<=t["sl"] and t["sl"]>0 else
                          "🎯 <b style='color:#00ff88'>TARGET HIT — BOOK PROFIT</b><br>" if curr>=t["target"] and t["target"]>0 else "")
                    st.markdown(f"""<div class="tc {cls}">{warn}<b>{t['sym']} {t['strike']} {t['side']}</b><br>
Entry ₹{t['entry']:.2f} → Now ₹{curr:.2f} | SL ₹{t['sl']:.2f} | Tgt ₹{t['target']:.2f}<br>
<b style='color:{clr_}'>P&L: ₹{pnl:+.2f} ({pp:+.1f}%)</b></div>""",unsafe_allow_html=True)
                    if st.button(f"Exit #{t['id']}",key=f"ex{i}{t['id']}"):
                        for j,x in enumerate(st.session_state.active):
                            if x["id"]==t["id"]:
                                st.session_state.active[j].update({"status":"CLOSED","exit":curr,"pnl":pnl})
                                for h in st.session_state.trades:
                                    if h["id"]==t["id"]: h.update(st.session_state.active[j])
                                break
                        st.rerun()

    with tabs[5]:
        st.markdown("""<div class="explain"><b>Backtest:</b> Simulates the signal on historical data.
Each day: score signal using same rules as live → if ≥72%, simulate buying ATM option → check next day's result → compound.
Good: Win Rate >60% · R:R >1.5 · Max Drawdown <20%</div>""",unsafe_allow_html=True)
        ba,bb,bc_=st.columns(3)
        bt_look=ba.slider("Lookback days",20,120,60)
        bt_cap=bb.number_input("Starting ₹",50000,5000000,int(capital),10000,key="bt_cap")
        bt_pos=bc_.slider("% per trade",2,20,pos_pct,key="bt_pos")
        if st.button("🔬 Run Backtest",type="primary",use_container_width=True):
            with st.spinner("Running…"):
                tdf,stats=run_backtest(yf_sym,bt_look,float(bt_cap),float(bt_pos))
                st.session_state.bt_result=(tdf,stats)
        if st.session_state.bt_result:
            tdf,stats=st.session_state.bt_result
            if tdf is None: st.error(f"Backtest: {stats}")
            else:
                k1,k2,k3,k4,k5,k6,k7=st.columns(7)
                k1.metric("Trades",stats["total"]); k2.metric("Win Rate",f"{stats['wr']}%")
                k3.metric("Total P&L",f"₹{stats['pnl']:+,.0f}"); k4.metric("Avg Win",f"₹{stats['aw']:,.0f}")
                k5.metric("Avg Loss",f"₹{stats['al']:,.0f}"); k6.metric("R:R",stats["rr"]); k7.metric("Max DD",f"{stats['mdd']}%")
                rc="#00ff88" if stats["ret%"]>=0 else "#ff3b5c"
                st.markdown(f'<div style="text-align:center;font-size:24px;color:{rc};font-family:Space Mono;padding:12px;background:#0a1929;border-radius:10px;margin:8px 0">Final ₹{stats["final"]:,.0f} · Return {stats["ret%"]:+.2f}%</div>',unsafe_allow_html=True)
                wr=stats["wr"]; rr=stats["rr"]; mdd=stats["mdd"]
                if wr>=60 and rr>=1.5 and mdd<20: vrd=("✅ Strong strategy — confident to use","#00ff88")
                elif wr>=50 and rr>=1.2: vrd=("🟡 Decent — works but needs discipline","#ff9500")
                else: vrd=("🔴 Weak — improve signal before real money","#ff3b5c")
                st.markdown(f'<div style="background:{vrd[1]}11;border-left:3px solid {vrd[1]};padding:12px;border-radius:0 8px 8px 0;margin:6px 0">{vrd[0]}</div>',unsafe_allow_html=True)
                fig_eq=go.Figure(go.Scatter(y=stats["curve"],mode="lines",line=dict(color="#00e5ff",width=2.5),fill="tozeroy",fillcolor="rgba(0,229,255,0.05)"))
                fig_eq.add_hline(y=bt_cap,line_dash="dash",line_color="#ff9500",annotation_text=f"Start ₹{int(bt_cap):,}")
                fig_eq.update_layout(template=DARK,height=300,title="Equity Curve",yaxis_title="₹")
                st.plotly_chart(fig_eq,use_container_width=True)
                st.markdown("""<div class="plot-explain">The equity curve shows how your capital would have grown/shrunk if you had followed this signal for every trade in the backtest period. 
A rising curve = strategy profits over time. Steep drops = the max drawdown periods. Flat periods = signal said "no trade". 
The orange line = starting capital. Above it = profitable; below = losing.</div>""",unsafe_allow_html=True)
                w=tdf[tdf["P&L₹"]>0]["P&L₹"]; l=tdf[tdf["P&L₹"]<=0]["P&L₹"]
                fig_d=go.Figure()
                fig_d.add_trace(go.Histogram(x=w,name="Wins",marker_color="rgba(0,255,136,0.47)",nbinsx=20))
                fig_d.add_trace(go.Histogram(x=l,name="Losses",marker_color="rgba(255,59,92,0.47)",nbinsx=20))
                fig_d.update_layout(template=DARK,height=220,title="Win vs Loss Distribution",barmode="overlay")
                st.plotly_chart(fig_d,use_container_width=True)
                st.dataframe(tdf,use_container_width=True,height=280)

    with tabs[6]:
        if not st.session_state.trades: st.info("No trades yet.")
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
                fig_p.update_layout(template=DARK,height=220,title="Per-Trade P&L")
                st.plotly_chart(fig_p,use_container_width=True)
            dc=[c for c in ["id","time","sym","strike","side","entry","exit","lots","pnl","status"] if c in all_t.columns]
            st.dataframe(all_t[dc],use_container_width=True)
            st.download_button("📥 Download CSV",all_t.to_csv(index=False),"trades.csv","text/csv",use_container_width=True)
            if st.button("🗑️ Clear",use_container_width=True):
                st.session_state.trades=[]; st.session_state.active=[]; st.rerun()

    with tabs[7]:
        iv=sig["atm_iv"]; pcr=sig["pcr"]; mp=sig["max_pain"]
        res=sig["resistance"]; sup=sig["support"]; mp_pct=(mp-spot)/spot*100 if spot>0 else 0
        st.markdown(f"""<div class="card"><h4 style='margin-top:0'>📖 Complete Market Reading — {fetch_sym} @ ₹{spot:,.2f} {badge}</h4></div>""",unsafe_allow_html=True)
        if has_chain:
            for (name,val,verdict,pts,exp) in sig["indicators"]:
                clr=("#00ff88" if "BULLISH" in verdict or "CHEAP" in verdict or "FRESH" in verdict or "✅" in verdict
                     else "#ff3b5c" if "BEAR" in verdict or "EXPENSIVE" in verdict or "AGAINST" in verdict
                     else "#ff9500")
                st.markdown(f'<div class="explain"><b style="color:{clr}">{name}: {val} → {verdict}</b><br>{exp}</div>',unsafe_allow_html=True)

            st.markdown("#### 🗺 Three Scenarios")
            s1,s2,s3=st.columns(3)
            side=sig["buy_side"]; atm_s=sig["buy_strike_atm"]; otm_s=sig["buy_strike_otm"]
            for col,title,trigger,action,why,clr in [
                (s1,"🟢 BULLISH",f"Break + hold above ₹{res:,.0f}",
                 f"Buy {atm_s} CE\nTarget +60%\nSL -30%","Resistance break → call writers panic → CE premium explodes","#00ff88"),
                (s2,"⚪ SIDEWAYS",f"Stay between ₹{sup:,.0f} – ₹{res:,.0f}",
                 "Do NOT buy options\nTheta kills you daily\nCash is position","Every day in a range = money lost to time decay. Wait.","#5a7a9a"),
                (s3,"🔴 BEARISH",f"Break + hold below ₹{sup:,.0f}",
                 f"Buy {atm_s} PE\nTarget +60%\nSL -30%","Support break → put writers panic → PE premium explodes","#ff3b5c"),
            ]:
                with col:
                    st.markdown(f"""<div style='background:#0a1929;border-top:4px solid {clr};border:1px solid #1e3050;border-radius:10px;padding:14px'>
<h5 style='color:{clr};margin-top:0'>{title}</h5>
<b style='color:#5a7a9a'>Trigger:</b> {trigger}<br><br>
<pre style='color:#c8d6e5;font-size:12px;background:transparent;margin:4px 0'>{action}</pre>
<span style='font-size:11px;color:#8899aa'>{why}</span></div>""",unsafe_allow_html=True)

        st.markdown("""<div class="card" style='margin-top:14px'>
<h4 style='color:#ff9500;margin-top:0'>📜 Golden Rules</h4>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:13px;line-height:1.9'>
<div>1. Never risk more than 5% of capital per trade.<br>2. Only buy when IV below 30%.<br>3. Set SL before entering — always.<br>4. Exit 5+ days before expiry.</div>
<div>5. Book 50% profit at first target, trail rest.<br>6. Never average a losing trade.<br>7. Avoid buying on event days (RBI, budget, earnings).<br>8. No trade is also a trade — wait for edge.</div>
</div></div>""",unsafe_allow_html=True)
        st.caption(f"Updated {datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S')} · Educational only · Not financial advice")

    if auto_ref:
        time.sleep(90); st.cache_data.clear(); _T["last"]=0.0; st.rerun()

if __name__=="__main__":
    main()
