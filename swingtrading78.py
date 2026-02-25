"""
Pro Options Dashboard v6 — India First
=======================================
NSE chain via 5 methods:
  1. nsepython library (pip install nsepython)
  2. NSE direct API  — session A (option-chain cookie route)
  3. NSE direct API  — session B (market-data cookie route)
  4. NSE direct API  — session C (mobile app headers)
  5. Opstra API      — no auth, works for indices
  6. CSV upload      — bulletproof parser for actual NSE download format
"""

import streamlit as st
import pandas as pd
import numpy as np
import io, time, random, warnings
import requests
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import brentq
import re

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

warnings.filterwarnings("ignore")

_T = {"last": 0.0}

def _wait(gap=1.2):
    now = time.time()
    d = now - _T["last"]
    if d < gap:
        time.sleep(gap - d + random.uniform(0.05, 0.2))
    _T["last"] = time.time()

# ─────────────────────────── page config ──────────────────────────────
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
.src{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-family:'Space Mono',monospace;font-weight:700;letter-spacing:1px}
.src-a{background:#003a1a;color:#00ff88;border:1px solid #00aa44}
.src-b{background:#001a3a;color:#00e5ff;border:1px solid #0066aa}
.src-c{background:#2a1500;color:#ff9500;border:1px solid #884400}
.src-d{background:#1a001a;color:#cc88ff;border:1px solid #664488}
hr{border-color:var(--brd)!important}
</style>""", unsafe_allow_html=True)

RFR  = 0.065
DARK = "plotly_dark"

NSE_INDICES = ["NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY","SENSEX"]
NSE_TO_YF   = {"NIFTY":"^NSEI","BANKNIFTY":"^NSEBANK","SENSEX":"^BSESN",
               "FINNIFTY":"NIFTY_FIN_SERVICE.NS","MIDCPNIFTY":"^NSMIDCP"}
NSE_FNO = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","SBIN","HINDUNILVR","BHARTIARTL",
    "KOTAKBANK","ITC","LT","AXISBANK","BAJFINANCE","MARUTI","TITAN","WIPRO",
    "HCLTECH","TECHM","SUNPHARMA","ULTRACEMCO","ASIANPAINT","NESTLEIND","POWERGRID",
    "NTPC","ONGC","TATAMOTORS","TATASTEEL","JSWSTEEL","HINDALCO","COALINDIA",
    "ADANIENT","ADANIPORTS","ADANIGREEN","SIEMENS","ABB","HAVELLS","GRASIM",
    "INDUSINDBK","BAJAJFINSV","BAJAJ-AUTO","HEROMOTOCO","EICHERMOT","M&M",
    "TATACONSUM","DIVISLAB","CIPLA","DRREDDY","APOLLOHOSP","LUPIN","BIOCON",
    "TORNTPHARM","PIDILITIND","BERGEPAINT","BHEL","SAIL","NMDC","VEDL","HINDZINC",
    "IOC","BPCL","HPCL","GAIL","PETRONET","CONCOR","IRCTC","DMART","TRENT",
    "NYKAA","ZOMATO","PAYTM","POLICYBZR","INDIGO","DLF","GODREJPROP","OBEROIRLTY",
    "BALKRISIND","APOLLOTYRE","MRF","BANKBARODA","PNB","CANBK","UNIONBANK",
    "FEDERALBNK","IDFCFIRSTB","MUTHOOTFIN","CHOLAFIN","LICHSGFIN","RECLTD","PFC",
    "HDFCLIFE","SBILIFE","ICICIPRULI","LICI","LTIM","LTTS","PERSISTENT","MPHASIS",
    "COFORGE","KPITTECH","SUZLON","TATAPOWER","NHPC","SJVN","TORNTPOWER",
    "MANAPPURAM","ANGELONE","DIXON","VOLTAS","ZEEL","SUNTV",
    "Custom symbol…"
]
YF_OPTS = {"SPY":"SPY","QQQ":"QQQ","AAPL":"AAPL","TSLA":"TSLA","NVDA":"NVDA","AMZN":"AMZN","Custom":"__custom__"}
YF_SPOT = {"BTC/USD":"BTC-USD","GOLD":"GC=F","SILVER":"SI=F","USD/INR":"USDINR=X","CRUDE OIL":"CL=F","Custom":"__custom__"}

for k,v in {"trades":[],"active":[],"bt_result":None}.items():
    if k not in st.session_state: st.session_state[k]=v


# ═══════════════════════════════════════════════════════════════════════
# CSV PARSER — handles actual NSE download format
# ═══════════════════════════════════════════════════════════════════════
def parse_nse_csv(raw_bytes, filename=""):
    """
    Parse the actual NSE option chain CSV downloaded from nseindia.com.

    Exact NSE format (2-row header):
      Row 0: ,CALLS,,,,,,,,,,PUTS,,,,,,,,,,,
      Row 1: ,OI,CHNG IN OI,VOLUME,IV,LTP,CHNG,BID QTY,BID,ASK,ASK QTY,STRIKE,BID QTY,BID,ASK,ASK QTY,CHNG,LTP,IV,VOLUME,CHNG IN OI,OI,
      Rows 2+: data  (numbers use Indian comma format like "2,54,729")

    Strategy: use csv.reader (handles quoted commas), find STRIKE column in row 1,
    then map all other columns by their fixed offset from STRIKE.

    Expiry is extracted from the filename (e.g. option-chain-ED-NIFTY-02-Mar-2026).

    Returns (DataFrame, list_of_debug_messages)
    """
    import csv as _csv

    debug = []

    # ── decode bytes ──────────────────────────────────────────────
    try:
        content = raw_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        return pd.DataFrame(), [f"Cannot read file: {e}"]

    debug.append(f"File size: {len(raw_bytes)} bytes")

    # ── extract expiry from filename ───────────────────────────────
    expiry = "Uploaded"
    m = re.search(r'(\d{2}-[A-Za-z]{3}-\d{4})', filename)
    if m:
        expiry = m.group(1)
        debug.append(f"Expiry from filename: {expiry}")
    else:
        debug.append("No expiry date found in filename — will use 'Uploaded'")

    # ── helper: strip commas and convert to float ──────────────────
    def to_f(s):
        try:
            s = str(s).replace(",", "").strip()
            if s in ("-", "", "nan", "None", "none", "--"): return 0.0
            return float(s)
        except:
            return 0.0

    # ── read all rows with csv.reader (handles "2,54,729" quoting) ─
    all_rows = []
    reader = _csv.reader(io.StringIO(content))
    for row in reader:
        all_rows.append(row)

    debug.append(f"Total CSV rows: {len(all_rows)}")
    for i, r in enumerate(all_rows[:3]):
        debug.append(f"  Row {i}: {r[:8]}")

    if len(all_rows) < 3:
        return pd.DataFrame(), debug + ["File has fewer than 3 rows — not enough data"]

    # ── find STRIKE column — scan rows 0,1,2 for the word "STRIKE" ─
    strike_col = None
    header_row_idx = None
    for row_i in range(min(4, len(all_rows))):
        for col_i, val in enumerate(all_rows[row_i]):
            if str(val).strip().upper() == "STRIKE":
                strike_col = col_i
                header_row_idx = row_i
                debug.append(f"Found STRIKE at row={row_i}, col={col_i}")
                break
        if strike_col is not None:
            break

    if strike_col is None:
        debug.append("❌ Could not find STRIKE column — trying fallback")
        # Fallback: maybe it's already a simple format
        try:
            df = pd.read_csv(io.StringIO(content))
            df.columns = [c.strip() for c in df.columns]
            if "strikePrice" in df.columns:
                for c in ["CE_OI","PE_OI","CE_IV","PE_IV","CE_changeOI","PE_changeOI",
                          "CE_volume","PE_volume","CE_bid","CE_ask","PE_bid","PE_ask"]:
                    if c not in df.columns: df[c] = 0.0
                if "expiryDate" not in df.columns: df["expiryDate"] = expiry
                df = df[df["strikePrice"] > 0].reset_index(drop=True)
                if not df.empty:
                    debug.append(f"✅ Fallback: strikePrice format — {len(df)} strikes")
                    return df.fillna(0), debug
        except: pass
        return pd.DataFrame(), debug + ["Cannot parse: no STRIKE column found"]

    # ── data starts on the row after the header row ────────────────
    data_start = header_row_idx + 1
    sc = strike_col  # short alias

    # Column offsets from STRIKE (verified against actual NSE file):
    # Left side  = CALLS: sc-10=OI, sc-9=CHNG IN OI, sc-8=VOLUME, sc-7=IV, sc-6=LTP
    #              sc-5=CHNG, sc-4=BID QTY, sc-3=BID, sc-2=ASK, sc-1=ASK QTY
    # Right side = PUTS:  sc+1=BID QTY, sc+2=BID, sc+3=ASK, sc+4=ASK QTY, sc+5=CHNG
    #              sc+6=LTP, sc+7=IV, sc+8=VOLUME, sc+9=CHNG IN OI, sc+10=OI
    offsets = {
        "strikePrice": 0,
        "CE_OI":       -10,
        "CE_changeOI": -9,
        "CE_volume":   -8,
        "CE_IV":       -7,
        "CE_LTP":      -6,
        "CE_bid":      -3,
        "CE_ask":      -2,
        "PE_bid":      +2,
        "PE_ask":      +3,
        "PE_LTP":      +6,
        "PE_IV":       +7,
        "PE_volume":   +8,
        "PE_changeOI": +9,
        "PE_OI":       +10,
    }

    records = []
    skipped = 0
    for row in all_rows[data_start:]:
        if len(row) <= sc:
            skipped += 1
            continue
        sp = to_f(row[sc])
        if sp <= 0:
            skipped += 1
            continue
        rec = {"expiryDate": expiry}
        for col_name, offset in offsets.items():
            idx = sc + offset
            rec[col_name] = to_f(row[idx]) if 0 <= idx < len(row) else 0.0
        records.append(rec)

    if skipped:
        debug.append(f"Skipped {skipped} non-data rows")

    if not records:
        return pd.DataFrame(), debug + ["No valid strike rows found after parsing"]

    df = pd.DataFrame(records).fillna(0)
    df = df[df["strikePrice"] > 0].reset_index(drop=True)
    debug.append(f"✅ Parsed {len(df)} strikes | expiry={expiry}")
    debug.append(f"   Sample: strike={df['strikePrice'].iloc[0]:,.0f} "
                 f"CE_LTP={df['CE_LTP'].iloc[0]} CE_IV={df['CE_IV'].iloc[0]}% "
                 f"PE_LTP={df['PE_LTP'].iloc[0]} PE_OI={df['PE_OI'].iloc[0]:,.0f}")
    return df, debug


# ═══════════════════════════════════════════════════════════════════════
# NSE LIVE DATA — 5 independent strategies
# ═══════════════════════════════════════════════════════════════════════

_UA_CHROME  = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
_UA_FIREFOX = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0"
_UA_MOBILE  = "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36"

def _nse_api_url(symbol):
    if symbol in NSE_INDICES:
        return f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    return f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"

def _parse_nse_resp(r):
    if r.status_code != 200:
        return None, 0.0, [], f"HTTP {r.status_code}"
    try:
        j    = r.json()
        rec  = j.get("records", {})
        spot = float(rec.get("underlyingValue") or 0)
        exps = rec.get("expiryDates", [])
        rows = []
        for item in rec.get("data", []):
            ce = item.get("CE",{}); pe = item.get("PE",{})
            rows.append({
                "strikePrice" : float(item.get("strikePrice",0)),
                "expiryDate"  : item.get("expiryDate",""),
                "CE_LTP"      : float(ce.get("lastPrice",0)),
                "CE_OI"       : float(ce.get("openInterest",0)),
                "CE_changeOI" : float(ce.get("changeinOpenInterest",0)),
                "CE_volume"   : float(ce.get("totalTradedVolume",0)),
                "CE_IV"       : float(ce.get("impliedVolatility",0)),
                "CE_bid"      : float(ce.get("bidprice",0)),
                "CE_ask"      : float(ce.get("askPrice",0)),
                "PE_LTP"      : float(pe.get("lastPrice",0)),
                "PE_OI"       : float(pe.get("openInterest",0)),
                "PE_changeOI" : float(pe.get("changeinOpenInterest",0)),
                "PE_volume"   : float(pe.get("totalTradedVolume",0)),
                "PE_IV"       : float(pe.get("impliedVolatility",0)),
                "PE_bid"      : float(pe.get("bidprice",0)),
                "PE_ask"      : float(pe.get("askPrice",0)),
            })
        df = pd.DataFrame(rows)
        if df.empty or spot == 0:
            return None, 0.0, [], "Empty data or spot=0"
        return df, spot, exps, None
    except Exception as e:
        return None, 0.0, [], f"Parse error: {e}"


def _nse_strategy_A(symbol):
    """Chrome + option-chain page cookie"""
    try:
        s = requests.Session()
        s.headers.update({"User-Agent":_UA_CHROME,"Accept-Language":"en-US,en;q=0.9,hi;q=0.8",
                          "Accept-Encoding":"gzip, deflate, br"})
        nav = {"Accept":"text/html,application/xhtml+xml,*/*;q=0.8",
               "Sec-Fetch-Dest":"document","Sec-Fetch-Mode":"navigate"}
        api = {"Accept":"*/*","Referer":"https://www.nseindia.com/option-chain",
               "X-Requested-With":"XMLHttpRequest","Sec-Fetch-Dest":"empty",
               "Sec-Fetch-Mode":"cors","Sec-Fetch-Site":"same-origin"}
        s.get("https://www.nseindia.com/", timeout=10, headers=nav); time.sleep(1.5)
        s.get("https://www.nseindia.com/option-chain", timeout=10, headers=nav); time.sleep(1.2)
        r = s.get(_nse_api_url(symbol), timeout=15, headers=api)
        return _parse_nse_resp(r)
    except Exception as e:
        return None, 0.0, [], f"Strategy-A: {e}"

def _nse_strategy_B(symbol):
    """Firefox + market-data derivatives page"""
    try:
        s = requests.Session()
        s.headers.update({"User-Agent":_UA_FIREFOX,"Accept-Language":"en-US,en;q=0.5",
                          "Accept-Encoding":"gzip, deflate, br","DNT":"1"})
        nav = {"Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
        api = {"Accept":"application/json, text/plain, */*",
               "Referer":"https://www.nseindia.com/market-data/equity-derivatives-watch",
               "X-Requested-With":"XMLHttpRequest"}
        s.get("https://www.nseindia.com/", timeout=10, headers=nav); time.sleep(1.3)
        s.get("https://www.nseindia.com/market-data/equity-derivatives-watch",
              timeout=10, headers={**nav,"Referer":"https://www.nseindia.com/"}); time.sleep(1.0)
        r = s.get(_nse_api_url(symbol), timeout=15, headers=api)
        return _parse_nse_resp(r)
    except Exception as e:
        return None, 0.0, [], f"Strategy-B: {e}"

def _nse_strategy_C(symbol):
    """Mobile Chrome — different cookie path"""
    try:
        s = requests.Session()
        s.headers.update({"User-Agent":_UA_MOBILE,"Accept-Language":"en-IN,en;q=0.9,hi;q=0.8",
                          "Accept-Encoding":"gzip, deflate, br"})
        nav = {"Accept":"text/html,application/xhtml+xml,*/*;q=0.8",
               "Sec-Fetch-Dest":"document","Sec-Fetch-Mode":"navigate","Upgrade-Insecure-Requests":"1"}
        s.get("https://www.nseindia.com/", timeout=10, headers=nav); time.sleep(1.4)
        s.get("https://m.nseindia.com/", timeout=8,
              headers={**nav,"Referer":"https://www.nseindia.com/"}); time.sleep(1.0)
        s.get("https://www.nseindia.com/option-chain", timeout=10,
              headers={**nav,"Referer":"https://www.nseindia.com/"}); time.sleep(1.0)
        api = {"Accept":"*/*","Referer":"https://www.nseindia.com/option-chain",
               "X-Requested-With":"XMLHttpRequest","Sec-Fetch-Mode":"cors"}
        r = s.get(_nse_api_url(symbol), timeout=15, headers=api)
        return _parse_nse_resp(r)
    except Exception as e:
        return None, 0.0, [], f"Strategy-C: {e}"

def _nse_strategy_D(symbol):
    """nsepython library — handles cookies automatically"""
    try:
        from nsepython import nse_optionchain_scrapper
        raw = nse_optionchain_scrapper(symbol)
        if not raw: return None, 0.0, [], "nsepython: empty response"
        df, spot, exps, err = None, 0.0, [], None
        rec = raw.get("records",{})
        spot = float(rec.get("underlyingValue") or 0)
        exps = rec.get("expiryDates",[])
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
        df = pd.DataFrame(rows)
        if df.empty or spot==0: return None,0.0,[],"nsepython: empty/spot=0"
        return df, spot, exps, None
    except ImportError:
        return None,0.0,[],"nsepython not installed (pip install nsepython)"
    except Exception as e:
        return None,0.0,[],f"nsepython: {e}"

def _nse_strategy_E(symbol):
    """Opstra — no auth, works for NIFTY/BANKNIFTY/FINNIFTY only"""
    opstra = {"NIFTY":"NIFTY","BANKNIFTY":"BANKNIFTY","FINNIFTY":"FINNIFTY","MIDCPNIFTY":"MIDCPNIFTY"}
    if symbol not in opstra:
        return None,0.0,[],f"Opstra: only indices supported, not {symbol}"
    try:
        hdrs = {"User-Agent":_UA_CHROME,"Accept":"application/json,*/*",
                "Referer":"https://opstra.definedge.com/","Origin":"https://opstra.definedge.com"}
        s = requests.Session(); s.headers.update(hdrs)
        er = s.get(f"https://opstra.definedge.com/api/openinterest/expiry/{opstra[symbol]}",timeout=10)
        if er.status_code!=200: return None,0.0,[],f"Opstra expiry HTTP {er.status_code}"
        exps = er.json().get("data",[])
        if not exps: return None,0.0,[],"Opstra: no expiries"
        oc = s.get(f"https://opstra.definedge.com/api/openinterest/{opstra[symbol]}/{exps[0]}",timeout=10)
        if oc.status_code!=200: return None,0.0,[],f"Opstra chain HTTP {oc.status_code}"
        j = oc.json()
        spot = float(j.get("underlyingValue",0) or j.get("spotPrice",0) or 0)
        rows=[]
        for item in j.get("data",[]):
            ce=item.get("CE",{}); pe=item.get("PE",{})
            rows.append({"strikePrice":float(item.get("strikePrice",0)),"expiryDate":exps[0],
                "CE_LTP":float(ce.get("lastPrice",0) or ce.get("ltp",0)),
                "CE_OI":float(ce.get("openInterest",0) or ce.get("oi",0)),
                "CE_changeOI":float(ce.get("changeinOpenInterest",0)),"CE_volume":float(ce.get("totalTradedVolume",0)),
                "CE_IV":float(ce.get("impliedVolatility",0)),"CE_bid":0.0,"CE_ask":0.0,
                "PE_LTP":float(pe.get("lastPrice",0) or pe.get("ltp",0)),
                "PE_OI":float(pe.get("openInterest",0) or pe.get("oi",0)),
                "PE_changeOI":float(pe.get("changeinOpenInterest",0)),"PE_volume":float(pe.get("totalTradedVolume",0)),
                "PE_IV":float(pe.get("impliedVolatility",0)),"PE_bid":0.0,"PE_ask":0.0})
        df = pd.DataFrame(rows)
        if df.empty: return None,0.0,[],"Opstra: empty chain"
        if spot==0 and _HAS_YF:
            try:
                h = yf.Ticker(NSE_TO_YF.get(symbol,"^NSEI")).history(period="1d")
                if not h.empty: spot=float(h["Close"].squeeze().iloc[-1])
            except: pass
        return df,spot,exps,None
    except Exception as e:
        return None,0.0,[],f"Opstra: {e}"


@st.cache_data(ttl=90, show_spinner=False)
def fetch_nse_chain(symbol):
    """Try all 5 strategies. Return (df, spot, expiries, source, errors)"""
    errors = []
    strategies = [
        ("nsepython",  _nse_strategy_D),
        ("NSE route-A",_nse_strategy_A),
        ("NSE route-B",_nse_strategy_B),
        ("NSE route-C",_nse_strategy_C),
        ("Opstra",     _nse_strategy_E),
    ]
    for name, fn in strategies:
        try:
            df, spot, exps, err = fn(symbol)
            if df is not None and spot > 0:
                return df, spot, exps, name, errors
            errors.append(f"❌ {name}: {err}")
        except Exception as e:
            errors.append(f"❌ {name}: unexpected {e}")
    return None, 0.0, [], None, errors


# ═══════════════════════════════════════════════════════════════════════
# SPOT PRICE via yfinance
# ═══════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60, show_spinner=False)
def get_spot(yf_sym):
    if not _HAS_YF: return 0.0
    _wait(1.2)
    try:
        fi = yf.Ticker(yf_sym).fast_info
        for a in ("last_price","lastPrice","regular_market_price","regularMarketPrice","previousClose"):
            v = getattr(fi,a,None)
            if v and float(v)>0: return float(v)
    except: pass
    try:
        h = yf.Ticker(yf_sym).history(period="2d",interval="1d")
        if not h.empty: return float(h["Close"].squeeze().iloc[-1])
    except: pass
    return 0.0

@st.cache_data(ttl=300, show_spinner=False)
def get_history(yf_sym, period="60d"):
    if not _HAS_YF: return pd.DataFrame()
    _wait(1.2)
    try:
        df = yf.Ticker(yf_sym).history(period=period,interval="1d",auto_adjust=True)
        return df if not df.empty else pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=120, show_spinner=False)
def get_yf_chain(sym, expiry):
    if not _HAS_YF: return None, None
    _wait(1.2)
    try:
        c = yf.Ticker(sym).option_chain(expiry)
        return c.calls.copy(), c.puts.copy()
    except: return None, None

def build_yf_df(calls, puts, expiry):
    if calls is None or calls.empty: return pd.DataFrame()
    df = pd.DataFrame()
    df["strikePrice"] = calls["strike"].values if "strike" in calls.columns else []
    df["expiryDate"]  = expiry
    for col,src in [("CE_LTP","lastPrice"),("CE_OI","openInterest"),("CE_volume","volume"),("CE_bid","bid"),("CE_ask","ask")]:
        df[col] = calls[src].values if src in calls.columns else 0.0
    df["CE_IV"] = (calls["impliedVolatility"].values*100) if "impliedVolatility" in calls.columns else 0.0
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
# BLACK-SCHOLES
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


# ═══════════════════════════════════════════════════════════════════════
# SIGNALS
# ═══════════════════════════════════════════════════════════════════════
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
    out["atm_iv"]=(ce_iv+pe_iv)/2 if (ce_iv+pe_iv)>0 else 20.0
    out["skew"]=round(pe_iv-ce_iv,2)
    near=(df["CE_OI"].where(df["strikePrice"].between(spot*.99,spot*1.01),0).sum()+
          df["PE_OI"].where(df["strikePrice"].between(spot*.99,spot*1.01),0).sum())
    out["gamma_blast"]=(ce_oi+pe_oi)>0 and near/(ce_oi+pe_oi+1)>0.35
    iv=out["atm_iv"]
    if iv>55: out["abnormal"].append(f"🔴 IV {iv:.0f}%: Very expensive — premium crashes after event, don't buy")
    elif iv>35: out["abnormal"].append(f"⚠️ IV {iv:.0f}%: Costly — theta eats profits fast")
    elif 0<iv<15: out["abnormal"].append(f"✅ IV {iv:.0f}%: Options cheap — great time to buy")
    sk=out["skew"]
    if sk>8: out["abnormal"].append(f"📊 Puts costlier (skew +{sk:.0f}%): Smart money hedging a fall → lean BEARISH")
    elif sk<-8: out["abnormal"].append(f"📊 Calls costlier (skew {sk:.0f}%): Aggressive call buying → lean BULLISH")
    mp_pct=(out["max_pain"]-spot)/spot*100 if spot>0 else 0
    if abs(mp_pct)>2: out["abnormal"].append(f"🎯 Max pain {mp_pct:+.1f}% away at ₹{out['max_pain']:,.0f} — market drifts here near expiry")
    if out["gamma_blast"]: out["abnormal"].append("⚡ GAMMA BLAST: Huge OI at current price — options explode on breakout, enter immediately on direction")
    score=50; direction="NONE"; reasons=[]
    pcr=out["pcr"]
    if pcr>1.5: score+=12; direction="CALL"; reasons.append(f"PCR {pcr:.2f} — contrarian bullish")
    elif pcr>1.1: score+=6; direction="CALL"; reasons.append(f"PCR {pcr:.2f} — mild bullish lean")
    elif pcr<0.5: score+=12; direction="PUT"; reasons.append(f"PCR {pcr:.2f} — contrarian bearish")
    elif pcr<0.9: score+=6; direction="PUT"; reasons.append(f"PCR {pcr:.2f} — mild bearish lean")
    else: reasons.append(f"PCR {pcr:.2f} — balanced")
    if iv>50: score-=25; reasons.append("IV>50%: avoid buying")
    elif iv>35: score-=12; reasons.append(f"IV {iv:.0f}%: expensive")
    elif 0<iv<18: score+=15; reasons.append(f"IV {iv:.0f}%: cheap — buy conditions")
    elif iv<=30: score+=8; reasons.append(f"IV {iv:.0f}%: reasonable")
    mp=out["max_pain"]
    if spot<mp and direction=="CALL": score+=10; reasons.append(f"Below max pain ₹{mp:,.0f} → bullish gravity")
    elif spot>mp and direction=="PUT": score+=10; reasons.append(f"Above max pain ₹{mp:,.0f} → bearish gravity")
    elif spot<mp and direction=="PUT": score-=8
    elif spot>mp and direction=="CALL": score-=8
    ce_chg=float(df["CE_changeOI"].sum()); pe_chg=float(df["PE_changeOI"].sum())
    if pe_chg>ce_chg*1.3 and direction=="CALL": score+=7; reasons.append("Fresh put writing — support building")
    if ce_chg>pe_chg*1.3 and direction=="PUT": score+=7; reasons.append("Fresh call writing — ceiling building")
    score=min(max(int(score),0),100); out["conf"]=score; out["direction"]=direction; out["reasons"]=reasons
    A=int(out["atm"]); otm=int(A*(1.005 if direction=="CALL" else 0.995))
    if score>=72 and direction=="CALL":
        out["rec"]="🟢 BUY CALL (CE)"
        out["scalp"]=f"Buy {A} CE — exit +20–40%, SL -30%"
        out["intraday"]=f"Buy {A} CE — target +60%, SL -30%"
        out["swing"]=f"Buy {otm} CE next expiry — target +80–100%"
        out["pos"]=f"Buy {otm} CE far expiry — hold for big move"
    elif score>=72 and direction=="PUT":
        out["rec"]="🔴 BUY PUT (PE)"
        out["scalp"]=f"Buy {A} PE — exit +20–40%, SL -30%"
        out["intraday"]=f"Buy {A} PE — target +60%, SL -30%"
        out["swing"]=f"Buy {otm} PE next expiry — target +80–100%"
        out["pos"]=f"Buy {otm} PE far expiry — hold for big move"
    elif score>=58:
        out["rec"]="🟡 WATCH — signal forming"
        out["scalp"]=out["intraday"]=out["swing"]=out["pos"]="Wait for confirmation"
    else:
        out["rec"]="⚪ NO TRADE — stay out"
        out["scalp"]=out["intraday"]=out["swing"]=out["pos"]="No edge. Cash is a position."
    return out


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
    tdf=pd.DataFrame(trades)
    wins=(tdf["P&L₹"]>0).sum(); total=len(tdf)
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
    fig.add_trace(go.Scatter(name="CE IV",x=x,y=sub["CE_IV"],mode="lines+markers",line=dict(color="#00ff88",width=2)),2,1)
    fig.add_trace(go.Scatter(name="PE IV",x=x,y=sub["PE_IV"],mode="lines+markers",line=dict(color="#ff3b5c",width=2)),2,1)
    fig.add_trace(go.Bar(name="CE Vol",x=x,y=sub["CE_volume"],marker_color="rgba(0,255,136,0.4)"),2,2)
    fig.add_trace(go.Bar(name="PE Vol",x=x,y=sub["PE_volume"],marker_color="rgba(255,59,92,0.4)"),2,2)
    for r in [1,2]:
        for c in [1,2]: vl(fig,spot,"Spot",row=r,col=c)
    fig.update_layout(template=DARK,height=520,barmode="group",title="Option Chain Overview",margin=dict(t=50,b=10))
    return fig

def plot_oi(df,spot,sig):
    rng=spot*0.055; sub=df[df["strikePrice"].between(spot-rng,spot+rng)]
    if sub.empty: sub=df
    sub=sub.copy()
    sub["CE_pct"]=(sub["CE_changeOI"]/(sub["CE_OI"]-sub["CE_changeOI"]).clip(lower=1)*100).fillna(0)
    sub["PE_pct"]=(sub["PE_changeOI"]/(sub["PE_OI"]-sub["PE_changeOI"]).clip(lower=1)*100).fillna(0)
    fig=make_subplots(3,1,subplot_titles=["Total OI","Change in OI Today","% Change in OI"])
    x=sub["strikePrice"]
    fig.add_trace(go.Bar(name="CE OI",x=x,y=sub["CE_OI"],marker_color="#00e5ff",opacity=.8),1,1)
    fig.add_trace(go.Bar(name="PE OI",x=x,y=sub["PE_OI"],marker_color="#ff9500",opacity=.8),1,1)
    fig.add_trace(go.Bar(name="ΔCE",x=x,y=sub["CE_changeOI"],marker_color=["#00ff88" if v>=0 else "#ff3b5c" for v in sub["CE_changeOI"]]),2,1)
    fig.add_trace(go.Bar(name="ΔPE",x=x,y=sub["PE_changeOI"],marker_color=["#ff9500" if v>=0 else "#8888ff" for v in sub["PE_changeOI"]]),2,1)
    fig.add_trace(go.Bar(name="CE%",x=x,y=sub["CE_pct"],marker_color=["rgba(0,255,136,0.47)" if v>=0 else "rgba(255,59,92,0.47)" for v in sub["CE_pct"]]),3,1)
    fig.add_trace(go.Bar(name="PE%",x=x,y=sub["PE_pct"],marker_color=["rgba(255,149,0,0.47)" if v>=0 else "rgba(136,136,255,0.47)" for v in sub["PE_pct"]]),3,1)
    for row in [1,2,3]:
        vl(fig,spot,"Spot",row=row)
        vl(fig,sig["resistance"],"R",color="#ff3b5c",dash="dot",row=row,col=1)
        vl(fig,sig["support"],"S",color="#00ff88",dash="dot",row=row,col=1)
    fig.update_layout(template=DARK,height=620,barmode="group",title="OI Analysis",margin=dict(t=50,b=10))
    return fig

def plot_greeks_chart(df,spot):
    rng=spot*0.04; sub=df[df["strikePrice"].between(spot-rng,spot+rng)]
    if sub.empty: sub=df
    fig=make_subplots(2,2,subplot_titles=["Delta","Gamma","Theta (daily)","Vega"])
    x=sub["strikePrice"]
    for (r,c,cc,pc) in [(1,1,"CE_delta","PE_delta"),(1,2,"CE_gamma","PE_gamma"),(2,1,"CE_theta","PE_theta"),(2,2,"CE_vega","PE_vega")]:
        for col,clr in [(cc,"#00ff88"),(pc,"#ff3b5c")]:
            if col in sub.columns:
                fig.add_trace(go.Scatter(name=col,x=x,y=sub[col],mode="lines",line=dict(color=clr,width=2)),r,c)
        vl(fig,spot,row=r,col=c)
    fig.update_layout(template=DARK,height=520,title="Greeks",margin=dict(t=50,b=10))
    return fig

def plot_straddle(yf_sym,straddle_now):
    hist=get_history(yf_sym,"45d")
    if hist.empty: return None,None
    close=hist["Close"].squeeze().astype(float)
    rv=close.pct_change().rolling(7).std()*np.sqrt(252)
    est=(close*rv*np.sqrt(7/252)*0.8).dropna()
    if est.empty: return None,None
    p25,p50,p75=est.quantile([.25,.5,.75])
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=est.index,y=est,name="Historical estimate",line=dict(color="#00e5ff",width=2)))
    if straddle_now>0:
        fig.add_hline(y=straddle_now,line_color="yellow",line_dash="dash",
                      annotation_text=f"Today:{straddle_now:.1f}",annotation_font_color="yellow")
    fig.add_hline(y=p25,line_color="#00ff88",line_dash="dot",annotation_text=f"Cheap:{p25:.1f}")
    fig.add_hline(y=p75,line_color="#ff3b5c",line_dash="dot",annotation_text=f"Exp:{p75:.1f}")
    fig.add_hline(y=p50,line_color="#ff9500",line_dash="dot",annotation_text=f"Avg:{p50:.1f}")
    fig.update_layout(template=DARK,height=300,title="Straddle vs 45-Day History")
    if straddle_now>0:
        if straddle_now<p25:   v=("✅ CHEAP — below average. Great time to buy!","#00ff88")
        elif straddle_now>p75: v=("⚠️ EXPENSIVE — overpriced. Avoid buying now.","#ff3b5c")
        else:                   v=("🟡 FAIR VALUE — normal pricing.","#ff9500")
    else: v=None
    return fig,v


# ═══════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════
def main():
    st.markdown("<h1 style='font-size:24px;letter-spacing:2px'>⚡ PRO OPTIONS — INDIA</h1>",unsafe_allow_html=True)
    st.markdown("<p style='color:#5a7a9a;margin-top:-10px'>5 live sources + CSV upload · NSE Indices + 80 F&O stocks · Greeks · OI · Signals · Backtest</p>",unsafe_allow_html=True)
    st.divider()

    # ── SIDEBAR ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Market")
        market = st.selectbox("",["🇮🇳 NSE India","🌐 Global Options (yFinance)","📊 Spot Only (BTC/Gold)"],label_visibility="collapsed")
        fetch_sym=""; yf_sym=""; is_nse=False; has_opts=True

        if "NSE" in market:
            is_nse=True
            cat=st.radio("Category",["Indices","F&O Stocks"],horizontal=True)
            if cat=="Indices":
                nse_sym=st.selectbox("Index",NSE_INDICES)
            else:
                nse_sym=st.selectbox("Stock (80+ F&O)",NSE_FNO)
                if nse_sym=="Custom symbol…":
                    nse_sym=st.text_input("NSE symbol","RELIANCE").upper().strip()
            fetch_sym=nse_sym; yf_sym=NSE_TO_YF.get(nse_sym,nse_sym+".NS")

            with st.expander("ℹ️ How data is fetched"):
                st.markdown("""**Auto-tries 5 sources in order:**
1. `nsepython` library
2. NSE API — Chrome/option-chain route  
3. NSE API — Firefox/market-data route
4. NSE API — Mobile Chrome route
5. Opstra (indices only)

**If all fail → upload CSV below**""")
            st.markdown("**Best: run once then restart**")
            st.code("pip install nsepython",language="bash")

        elif "Global" in market:
            ch=st.selectbox("Instrument",list(YF_OPTS.keys()))
            fetch_sym=st.text_input("Ticker","AAPL").upper() if ch=="Custom" else YF_OPTS[ch]
            yf_sym=fetch_sym
        else:
            ch=st.selectbox("Instrument",list(YF_SPOT.keys()))
            fetch_sym=st.text_input("Ticker","BTC-USD").upper() if ch=="Custom" else YF_SPOT[ch]
            yf_sym=fetch_sym; has_opts=False
            st.info("No options for this instrument.")

        st.divider()

        # CSV upload — always visible for NSE
        if is_nse:
            st.markdown("### 📂 Upload CSV (guaranteed fallback)")
            st.markdown("""<div style='font-size:12px;color:#5a7a9a;line-height:1.6'>
<b>Step 1:</b> Go to <code>nseindia.com</code><br>
<b>Step 2:</b> Click Option Chain in top menu<br>
<b>Step 3:</b> Select your index/stock + expiry<br>
<b>Step 4:</b> Click <b>Download (↓ CSV)</b> button<br>
<b>Step 5:</b> Upload that file here ↓<br><br>
⚠️ Don't open in Excel and re-save
</div>""",unsafe_allow_html=True)
            csv_file=st.file_uploader("Upload NSE option chain CSV",type=["csv"],key="csv_up")
        else:
            csv_file=None

        st.divider()
        st.markdown("### 💰 Risk")
        capital=st.number_input("Capital ₹",50000,10000000,100000,10000)
        pos_pct=st.slider("% per trade",2,15,5)
        sl_pct=st.slider("Stop loss %",20,50,30)
        tgt_pct=st.slider("Target %",30,200,60)
        st.divider()
        st.button("🔄 Refresh Data",type="primary",use_container_width=True)
        if st.button("🗑️ Clear Cache",use_container_width=True):
            st.cache_data.clear(); _T["last"]=0.0; st.rerun()
        auto_ref=st.checkbox("Auto-refresh (90s)")
        st.caption("Educational only · Not financial advice")

    # ── DATA FETCH ────────────────────────────────────────────────
    df_exp=pd.DataFrame(); spot=0.0; expiries=[]; sel_expiry=""
    data_src=""; fetch_errors=[]
    ph=st.empty()

    if is_nse:
        # ── CSV path (guaranteed to work) ──
        if csv_file is not None:
            ph.info("📂 Reading uploaded CSV…")
            raw = csv_file.read()
            fname = getattr(csv_file, "name", "")
            df_csv, csv_dbg = parse_nse_csv(raw, filename=fname)
            if not df_csv.empty:
                df_exp = df_csv
                data_src = "CSV"
                spot = get_spot(yf_sym)
                expiries = df_csv["expiryDate"].unique().tolist()
                sel_expiry = expiries[0]
                ph.success(f"✅ CSV parsed — {len(df_exp)} strikes | Expiry: {sel_expiry} | Spot ₹{spot:,.2f}")
                with st.expander("🔍 CSV parse details"):
                    for line in csv_dbg: st.text(line)
            else:
                ph.error("❌ CSV failed to parse. See debug below.")
                with st.expander("📋 CSV debug — share this to get help"):
                    for line in csv_dbg: st.text(line)
                st.markdown("""**CSV parse failed. Common fixes:**
- Download directly from nseindia.com (do not open in Excel and re-save)
- The file should be the raw .csv, around 10-50 KB""")


        # ── Live fetch path ──
        if df_exp.empty:
            ph.info("📡 Trying 5 live sources… (takes ~15 seconds, please wait)")
            with st.spinner("Contacting NSE servers…"):
                df_raw,spot_raw,exps_raw,src,errs = fetch_nse_chain(fetch_sym)
            fetch_errors=errs
            if df_raw is not None and spot_raw>0:
                df_exp=df_raw; spot=spot_raw; expiries=exps_raw; data_src=src
                ph.success(f"✅ Live chain via **{src}** — {len(df_exp)} strikes | Spot ₹{spot:,.2f}")
            else:
                spot=get_spot(yf_sym)
                has_opts=False
                if spot>0:
                    ph.warning(f"⚠️ All live sources failed. Spot ₹{spot:,.2f} from yFinance only.")
                else:
                    ph.error("❌ Cannot get any data.")
                # Show error details
                st.markdown("---")
                st.markdown("### ❌ All 5 sources failed — here's why:")
                for e in fetch_errors: st.markdown(f"- {e}")
                st.markdown("""---
### ✅ What to do RIGHT NOW:

**Option 1 — Install nsepython (easiest, run once):**
```bash
pip install nsepython
```
Then click **Clear Cache** and refresh.

**Option 2 — Upload CSV (works always):**
1. Open [nseindia.com](https://www.nseindia.com) in your browser
2. Click **Option Chain** in the top navigation bar  
3. Select your symbol and expiry
4. Click the **Download (↓ CSV)** button — top right of the table
5. Upload that file in the sidebar ← 

**Option 3 — Market closed?**  
NSE chain is only live 9:15 AM – 3:30 PM IST on Mon–Fri. Outside hours, use CSV from earlier today.
""")

        # Expiry selector
        if not df_exp.empty and expiries:
            with st.sidebar:
                sel_expiry=st.selectbox("Expiry",expiries[:10] if len(expiries)>10 else expiries)
            if "expiryDate" in df_exp.columns and data_src!="CSV":
                mask=df_exp["expiryDate"]==sel_expiry
                df_exp=df_exp[mask].copy() if mask.any() else df_exp.copy()

    elif has_opts:
        ph.info("📡 Fetching from yFinance…")
        spot=get_spot(fetch_sym)
        if spot>0:
            if _HAS_YF:
                _wait(1.2)
                try:
                    exps=list(yf.Ticker(fetch_sym).options or [])
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
            else: ph.warning("Install yfinance: pip install yfinance"); has_opts=False
        else: ph.error(f"Cannot get price for {fetch_sym}.")
    else:
        spot=get_spot(fetch_sym)
        if spot>0: ph.success(f"✅ {fetch_sym}: {spot:,.4f}")
        else: ph.error(f"Cannot get price for {fetch_sym}.")

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

    # source badge
    src_cls={"nsepython":"src-a","NSE route-A":"src-b","NSE route-B":"src-b",
             "NSE route-C":"src-b","Opstra":"src-c","CSV":"src-d","yFinance":"src-b","":""}
    badge=(f'<span class="src {src_cls.get(data_src,"src-b")}">{data_src}</span>'
           if data_src else "")

    # ── KEY METRICS ──
    c1,c2,c3,c4,c5,c6,c7=st.columns(7)
    c1.metric("📍 Spot",f"₹{spot:,.2f}")
    c2.metric("🎯 ATM",f"₹{sig['atm']:,.0f}" if has_chain else "—")
    c3.metric("📊 PCR",f"{sig['pcr']:.3f}" if has_chain else "—")
    c4.metric("💀 Max Pain",f"₹{sig['max_pain']:,.0f}" if has_chain else "—")
    c5.metric("🌡 IV",f"{sig['atm_iv']:.1f}%" if has_chain else "—")
    c6.metric("↕ Skew",f"{sig['skew']:+.1f}%" if has_chain else "—")
    c7.metric("♟ Straddle",f"₹{sig['straddle']:.2f}" if has_chain else "—")

    # ── SIGNAL BANNER ──
    conf=sig["conf"]; bc="#00ff88" if conf>=72 else "#ff9500" if conf>=58 else "#5a7a9a"
    def tag(s): return f'<span class="tag-b">{s}</span>' if "BUY" in s else f'<span class="tag-w">{s}</span>'

    if has_chain:
        st.markdown(f"""<div class="sig-box">
<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;align-items:center">
  <div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
      <span style="color:#5a7a9a;font-size:10px;letter-spacing:2px">SIGNAL</span>{badge}
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
        for ab in sig["abnormal"]:
            st.markdown(f'<div class="alert">{ab}</div>',unsafe_allow_html=True)
        if sig["gamma_blast"]:
            st.markdown('<div class="gamma-blast">⚡ <b style="color:#ff3b5c">GAMMA BLAST</b> — Huge OI at current price. Options explode on breakout. Enter immediately on confirmation.</div>',unsafe_allow_html=True)
    else:
        st.info(f"📊 Spot only — ₹{spot:,.2f}. Upload CSV or install nsepython for full chain analysis.")

    st.markdown("<br>",unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────
    tabs=st.tabs(["📊 Chain","🔢 Greeks","📈 OI","⚡ Trade","🔬 Backtest","📋 History","🧠 Analysis"])

    with tabs[0]:
        st.markdown("""<div class="explain"><b>Option Chain</b> — each row is one strike price.
CE (Call) = buy when market goes UP. PE (Put) = buy when market goes DOWN.
ATM = closest strike to current price — highest activity. High OI = strong support/resistance wall.</div>""",unsafe_allow_html=True)
        if not has_chain:
            st.warning("Chain not loaded. Upload CSV from sidebar or install nsepython.")
        else:
            def_st=all_strikes[max(0,atm_pos-8):atm_pos+9]
            sel_st=st.multiselect("Strikes",all_strikes,default=def_st)
            if not sel_st: sel_st=def_st
            show_c=[c for c in ["strikePrice","CE_LTP","CE_OI","CE_changeOI","CE_IV","CE_volume","PE_LTP","PE_OI","PE_changeOI","PE_IV","PE_volume"] if c in df_exp.columns]
            st.dataframe(df_exp[df_exp["strikePrice"].isin(sel_st)][show_c].round(2),use_container_width=True,height=260)
            st.plotly_chart(plot_chain(df_exp,spot),use_container_width=True)
            fs,fv=plot_straddle(yf_sym,sig["straddle"])
            if fs:
                st.markdown("#### Straddle vs History — cheap or expensive today?")
                st.plotly_chart(fs,use_container_width=True)
                if fv: st.markdown(f'<div style="background:{fv[1]}11;border:1px solid {fv[1]}44;border-radius:8px;padding:12px;color:{fv[1]};font-weight:600">{fv[0]}</div>',unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("""<div class="explain">
🟢 <b>Delta</b> — Nifty moves ₹100, your option moves this much. Delta 0.5 = ₹50 gain per ₹100 move.<br>
🔵 <b>Gamma</b> — How fast Delta changes. High near expiry = big swings possible.<br>
🔴 <b>Theta</b> — Money lost per day just by holding. Option buyer's enemy.<br>
🟡 <b>Vega</b> — Gain when market fear (IV) rises. Buy before events, sell after.<br>
<b>Rule:</b> Only buy when Vega > |Theta| and IV below 25%.
</div>""",unsafe_allow_html=True)
        if not has_chain: st.warning("Needs chain data.")
        else:
            st.plotly_chart(plot_greeks_chart(df_exp,spot),use_container_width=True)
            ai=(df_exp["strikePrice"]-spot).abs().idxmin(); atm2=df_exp.loc[ai]
            g1,g2=st.columns(2)
            for col,px,lbl,clr in [(g1,"CE","📗 CALL — market goes UP","#00ff88"),(g2,"PE","📕 PUT — market goes DOWN","#ff3b5c")]:
                with col:
                    st.markdown(f"<h5 style='color:{clr}'>{lbl}</h5>",unsafe_allow_html=True)
                    cs=st.columns(3)
                    for i,(n,k,t) in enumerate([("Delta",f"{px}_delta","₹/₹100"),("Gamma",f"{px}_gamma","speed"),("Theta",f"{px}_theta","daily"),("Vega",f"{px}_vega","IV gain"),("IV%",f"{px}_IV","now")]):
                        cs[i%3].metric(n,f"{float(atm2.get(k,0)):.2f}{'%' if 'IV' in n else ''}",t)

    with tabs[2]:
        st.markdown("""<div class="explain">
🔵 <b>High CE OI</b> = resistance — sellers defend this ceiling<br>
🟡 <b>High PE OI</b> = support — sellers defend this floor<br>
Rising OI + Rising Price = long buildup = BULLISH signal<br>
Rising OI + Falling Price = short buildup = BEARISH signal
</div>""",unsafe_allow_html=True)
        if not has_chain: st.warning("Needs chain data.")
        else:
            st.plotly_chart(plot_oi(df_exp,spot,sig),use_container_width=True)
            m1,m2,m3,m4=st.columns(4)
            m1.metric("🔴 Resistance",f"₹{sig['resistance']:,.0f}"); m2.metric("🟢 Support",f"₹{sig['support']:,.0f}")
            m3.metric("🎯 Max Pain",f"₹{sig['max_pain']:,.0f}"); m4.metric("PCR",f"{sig['pcr']:.4f}")
            b1,b2=st.columns(2)
            with b1:
                st.markdown("**Top CE OI — resistance walls**")
                ct=df_exp.nlargest(5,"CE_OI")[["strikePrice","CE_OI","CE_changeOI","CE_LTP"]].copy()
                ct["Signal"]=ct["CE_changeOI"].apply(lambda x:"🔴 Fresh" if x>=0 else "🟡 Fading")
                st.dataframe(ct,use_container_width=True)
            with b2:
                st.markdown("**Top PE OI — support walls**")
                pt=df_exp.nlargest(5,"PE_OI")[["strikePrice","PE_OI","PE_changeOI","PE_LTP"]].copy()
                pt["Signal"]=pt["PE_changeOI"].apply(lambda x:"🟢 Fresh" if x>=0 else "🟡 Fading")
                st.dataframe(pt,use_container_width=True)

    with tabs[3]:
        st.markdown("""<div class="explain">Paper trade — no real money, just practice.
CE = UP trade · PE = DOWN trade · SL = stop loss (exit if option falls this %) · Target = exit here for profit
Never enter without a stop loss. R:R should be above 1.5 (target 1.5x bigger than your risk).
</div>""",unsafe_allow_html=True)
        if not has_chain: st.warning("Needs chain data.")
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
                r_row=df_exp[df_exp["strikePrice"]==t_str]
                opt_px=float(r_row[f"{px_}_LTP"].values[0]) if not r_row.empty and f"{px_}_LTP" in r_row.columns else 0
                if opt_px>0:
                    sl_a=round(opt_px*(1-l_sl/100),2); tgt_a=round(opt_px*(1+l_tgt/100),2)
                    risk=(opt_px-sl_a)*lots; rew=(tgt_a-opt_px)*lots; rr=round(rew/risk,2) if risk>0 else 0
                    pm1,pm2,pm3,pm4=st.columns(4)
                    pm1.metric("Entry",f"₹{opt_px:.2f}"); pm2.metric("SL",f"₹{sl_a:.2f}",f"-{l_sl}%")
                    pm3.metric("Target",f"₹{tgt_a:.2f}",f"+{l_tgt}%"); pm4.metric("R:R",f"1:{rr}")
                    if rr<1.5: st.warning("⚠️ R:R < 1.5 — poor trade setup. Increase target or tighten SL.")
                else:
                    sl_a=tgt_a=0; st.warning("Price=0. Market closed or strike too far OTM.")
                if st.button("📈 Enter Paper Trade",type="primary",use_container_width=True):
                    if opt_px>0:
                        t={"id":len(st.session_state.trades)+1,"sym":fetch_sym,"expiry":sel_expiry,
                           "strike":t_str,"side":px_,"entry":opt_px,"sl":sl_a,"target":tgt_a,
                           "lots":lots,"conf":sig["conf"],"rec":sig["rec"],
                           "time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "status":"OPEN","exit":None,"pnl":None}
                        st.session_state.active.append(t); st.session_state.trades.append(t)
                        st.success(f"✅ {fetch_sym} {t_str} {px_} @ ₹{opt_px:.2f}")
                    else: st.error("Cannot enter — price=0")
            with lb:
                st.markdown("#### Open Positions")
                if not [t for t in st.session_state.active if t["status"]=="OPEN"]: st.info("No open trades.")
                for i,t in enumerate(st.session_state.active):
                    if t["status"]!="OPEN": continue
                    rr=df_exp[df_exp["strikePrice"]==t["strike"]]; lc=f"{t['side']}_LTP"
                    curr=float(rr[lc].values[0]) if not rr.empty and lc in rr.columns else t["entry"]
                    pnl=round((curr-t["entry"])*t["lots"],2); pp=round((curr-t["entry"])/t["entry"]*100,2) if t["entry"]>0 else 0
                    clr="#00ff88" if pnl>=0 else "#ff3b5c"; cls="tc-w" if pnl>0 else "tc-l" if pnl<0 else "tc-o"
                    warn=("⚠️ <b style='color:#ff3b5c'>SL HIT — EXIT NOW</b><br>" if curr<=t["sl"] and t["sl"]>0 else
                          "🎯 <b style='color:#00ff88'>TARGET HIT — BOOK PROFIT</b><br>" if curr>=t["target"] and t["target"]>0 else "")
                    st.markdown(f"""<div class="tc {cls}">{warn}<b>{t['sym']} {t['strike']} {t['side']}</b><br>
Entry ₹{t['entry']:.2f} → Now ₹{curr:.2f} | SL ₹{t['sl']:.2f} | Tgt ₹{t['target']:.2f}<br>
<b style='color:{clr}'>P&L: ₹{pnl:+.2f} ({pp:+.1f}%)</b></div>""",unsafe_allow_html=True)
                    if st.button(f"Exit #{t['id']}",key=f"ex{i}{t['id']}"):
                        for j,x in enumerate(st.session_state.active):
                            if x["id"]==t["id"]:
                                st.session_state.active[j].update({"status":"CLOSED","exit":curr,"pnl":pnl})
                                for h in st.session_state.trades:
                                    if h["id"]==t["id"]: h.update(st.session_state.active[j])
                                break
                        st.rerun()

    with tabs[4]:
        st.markdown("""<div class="explain"><b>Backtest:</b>
Looks at every day in the past N days · Scores it using the same rules as the live signal ·
If score ≥72% pretends to buy an ATM option · Checks next day's move to determine win/loss ·
Subtracts 1 day of theta per trade · Tracks equity across all trades.<br><br>
<b>Win Rate 60%+ is good · R:R above 1.5 is good · Max Drawdown below 20% is good</b>
</div>""",unsafe_allow_html=True)
        ba,bb,bc_=st.columns(3)
        bt_look=ba.slider("Lookback days",20,120,60)
        bt_cap=bb.number_input("Starting capital ₹",50000,5000000,int(capital),10000,key="bt_cap")
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
                if wr>=60 and rr>=1.5 and mdd<20: vrd=("✅ Strong — consistent and controlled","#00ff88")
                elif wr>=50 and rr>=1.2: vrd=("🟡 Decent — works but needs strict discipline","#ff9500")
                else: vrd=("🔴 Weak — don't trade real money on this yet","#ff3b5c")
                st.markdown(f'<div style="background:{vrd[1]}11;border-left:3px solid {vrd[1]};padding:12px;border-radius:0 8px 8px 0;margin:6px 0">{vrd[0]}</div>',unsafe_allow_html=True)
                fig_eq=go.Figure(go.Scatter(y=stats["curve"],mode="lines",line=dict(color="#00e5ff",width=2.5),fill="tozeroy",fillcolor="rgba(0,229,255,0.05)"))
                fig_eq.add_hline(y=bt_cap,line_dash="dash",line_color="#ff9500",annotation_text=f"Start ₹{int(bt_cap):,}")
                fig_eq.update_layout(template=DARK,height=300,title="Equity Curve",yaxis_title="₹")
                st.plotly_chart(fig_eq,use_container_width=True)
                w=tdf[tdf["P&L₹"]>0]["P&L₹"]; l=tdf[tdf["P&L₹"]<=0]["P&L₹"]
                fig_d=go.Figure()
                fig_d.add_trace(go.Histogram(x=w,name="Wins",marker_color="rgba(0,255,136,0.47)",nbinsx=20))
                fig_d.add_trace(go.Histogram(x=l,name="Losses",marker_color="rgba(255,59,92,0.47)",nbinsx=20))
                fig_d.update_layout(template=DARK,height=220,title="Win vs Loss Distribution",barmode="overlay")
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
                fig_p.update_layout(template=DARK,height=220,title="Per-Trade P&L")
                st.plotly_chart(fig_p,use_container_width=True)
            dc=[c for c in ["id","time","sym","strike","side","entry","exit","lots","pnl","status"] if c in all_t.columns]
            st.dataframe(all_t[dc],use_container_width=True)
            st.download_button("📥 Download CSV",all_t.to_csv(index=False),"trades.csv","text/csv",use_container_width=True)
            if st.button("🗑️ Clear Trades",use_container_width=True):
                st.session_state.trades=[]; st.session_state.active=[]; st.rerun()

    with tabs[6]:
        iv=sig["atm_iv"]; pcr=sig["pcr"]; mp=sig["max_pain"]
        res=sig["resistance"]; sup=sig["support"]; mp_pct=(mp-spot)/spot*100 if spot>0 else 0
        st.markdown(f"""<div class="card"><h4 style='color:#00e5ff;margin-top:0'>📖 Market Reading — {fetch_sym} @ ₹{spot:,.2f} {badge}</h4></div>""",unsafe_allow_html=True)
        if has_chain:
            pcr_text=("HIGH PCR — more puts than calls. Sounds scary but it's CONTRARIAN BULLISH. Institutions are protecting longs (not betting on crash). Market often goes UP." if pcr>1.2
                     else "LOW PCR — more calls than puts. Sounds bullish but CONTRARIAN BEARISH. When everyone is optimistic, market may go DOWN." if pcr<0.8
                     else "BALANCED PCR — no strong directional signal. Wait for clearer setup.")
            st.markdown(f'<div class="explain"><b>📊 PCR = {pcr:.4f}</b><br>{pcr_text}</div>',unsafe_allow_html=True)
            iv_text=("CHEAP options — like petrol at a discount. Low IV = pay less + gain if IV rises later." if iv<18
                    else "VERY EXPENSIVE options — market is scared. Worst time to buy; even right direction loses as IV crashes after event." if iv>40
                    else f"MODERATE IV — {iv:.0f}%. Not cheap, not expensive. Only buy on strong signals.")
            st.markdown(f'<div class="explain"><b>🌡 IV = {iv:.1f}%</b><br>{iv_text}</div>',unsafe_allow_html=True)
            mp_text=(f"Max pain {mp_pct:+.1f}% ABOVE at ₹{mp:,.0f}. Near expiry market gets pulled upward. Favors CALL buyers." if mp_pct>2
                    else f"Max pain {abs(mp_pct):.1f}% BELOW at ₹{mp:,.0f}. Near expiry market gets pulled downward. Favors PUT buyers." if mp_pct<-2
                    else f"Max pain ₹{mp:,.0f} close to price. No strong gravity pull. Matters most in last 2–3 expiry days.")
            st.markdown(f'<div class="explain"><b>🎯 Max Pain ₹{mp:,.0f}</b><br>{mp_text}</div>',unsafe_allow_html=True)
            st.markdown(f"""<div class="explain"><b>🔴 Resistance ₹{res:,.0f}  |  🟢 Support ₹{sup:,.0f}</b><br>
Resistance = biggest CE OI = ceiling. Call writers defend this aggressively. Break above → writers panic-buy → sharp rally.<br>
Support = biggest PE OI = floor. Put writers defend this. Break below → writers panic-buy puts → sharp fall.<br>
<b>Trade the breakout, not the range.</b> Options bought inside range lose money to theta every day.
</div>""",unsafe_allow_html=True)
            s1,s2,s3=st.columns(3)
            for col,title,trigger,action,why,clr in [
                (s1,"🟢 BULLISH",f"Break & hold above ₹{res:,.0f}",f"Buy {int(sig['atm'])} CE\nTarget +60%\nSL -30%","Resistance breaks → call writers panic → CE premium explodes","#00ff88"),
                (s2,"⚪ SIDEWAYS",f"Price stays between ₹{sup:,.0f}–₹{res:,.0f}","DO NOT BUY OPTIONS\nTheta kills you daily\nCash is the position","Every day you hold in a range = money lost to time decay","#5a7a9a"),
                (s3,"🔴 BEARISH", f"Break & hold below ₹{sup:,.0f}",f"Buy {int(sig['atm'])} PE\nTarget +60%\nSL -30%","Support breaks → put writers panic → PE premium explodes","#ff3b5c"),
            ]:
                with col:
                    st.markdown(f"""<div style='background:#0a1929;border-top:4px solid {clr};border:1px solid #1e3050;border-radius:10px;padding:14px'>
<h5 style='color:{clr};margin-top:0'>{title}</h5>
<b style='color:#5a7a9a'>Trigger:</b> {trigger}<br><br>
<pre style='color:#c8d6e5;font-size:12px;background:transparent;margin:4px 0'>{action}</pre>
<span style='font-size:11px;color:#8899aa'>{why}</span></div>""",unsafe_allow_html=True)
        st.markdown("""<div class="card" style='margin-top:14px'>
<h4 style='color:#ff9500;margin-top:0'>📜 Golden Rules</h4>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:13px;line-height:1.8'>
<div>1. Never risk more than 5% per trade.<br>2. Only buy when IV below 30%.<br>3. Set SL before entering — always.<br>4. Exit 5+ days before expiry.</div>
<div>5. Book 50% at first target, trail rest.<br>6. Never average a losing trade.<br>7. Avoid buying on event days (RBI, earnings).<br>8. No trade is also a trade — wait for edge.</div>
</div></div>""",unsafe_allow_html=True)
        st.caption(f"Updated {datetime.now().strftime('%d-%b-%Y %H:%M:%S')} · Educational only · Not financial advice")

    if auto_ref:
        time.sleep(90); st.cache_data.clear(); _T["last"]=0.0; st.rerun()

if __name__=="__main__":
    main()
