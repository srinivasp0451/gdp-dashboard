"""
╔══════════════════════════════════════════════════════════════════╗
║   PRO OPTIONS DASHBOARD — NSE + BSE India + Global              ║
║   Single file · Selenium scraper + Full analysis dashboard       ║
╠══════════════════════════════════════════════════════════════════╣
║   Run:  streamlit run options_final.py                           ║
║   Deps: pip install streamlit selenium webdriver-manager         ║
║          yfinance plotly scipy pandas requests                   ║
╠══════════════════════════════════════════════════════════════════╣
║   NSE  : NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY + 80 stocks     ║
║   BSE  : SENSEX, BANKEX                                          ║
║   Global: SPY QQQ AAPL TSLA NVDA + any ticker                   ║
╠══════════════════════════════════════════════════════════════════╣
║   How it works:                                                  ║
║   1. Click START in sidebar — opens Chrome headless ONCE         ║
║   2. Browser navigates NSE/BSE, solves Akamai JS challenge       ║
║   3. Cookies extracted, used for fast API calls every 60s        ║
║   4. Cookies auto-refreshed every 24 min via browser             ║
║   5. Dashboard reads live data instantly, no network calls       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import io, re, csv as _csv, time, random, warnings
import datetime as dt
import json, threading
import requests
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════
RFR           = 0.065
DARK          = "plotly_dark"
DATA_FILE     = Path("nse_live_data.json")
COOKIE_TTL    = 24 * 60   # seconds before browser re-auth
FETCH_INTERVAL= 60        # seconds between API polls

NSE_INDICES = ["NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY"]
BSE_INDICES = ["SENSEX","BANKEX"]
BSE_MAP     = {"SENSEX":("1","SENSEX"),"BANKEX":("2","BANKEX")}
NSE_TO_YF   = {
    "NIFTY":"^NSEI","BANKNIFTY":"^NSEBANK","SENSEX":"^BSESN","BANKEX":"^BSESN",
    "FINNIFTY":"NIFTY_FIN_SERVICE.NS","MIDCPNIFTY":"^NSMIDCP",
}
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
]
YF_OPTS = {"SPY":"SPY","QQQ":"QQQ","AAPL":"AAPL","TSLA":"TSLA","NVDA":"NVDA","AMZN":"AMZN","Custom":"__c__"}
YF_SPOT = {"BTC/USD":"BTC-USD","GOLD":"GC=F","SILVER":"SI=F","USD/INR":"USDINR=X","CRUDE":"CL=F","Custom":"__c__"}
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# ═══════════════════════════════════════════════════
# SHARED MEMORY (scraper thread writes, UI reads)
# ═══════════════════════════════════════════════════
_lock   = threading.Lock()
_store  = {}          # symbol -> {df_rows, spot, expiries, timestamp, source}
_logs   = []          # ring buffer shown in sidebar
_status = {"running":False,"error":"","driver_ok":False}

def _slog(msg):
    ts = dt.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with _lock:
        _logs.append(line)
        if len(_logs) > 100:
            _logs.pop(0)

def _save(sym, df, spot, expiries, source):
    rec = {
        "rows":      df.to_dict("records") if df is not None and not df.empty else [],
        "spot":      float(spot),
        "expiries":  list(expiries),
        "timestamp": dt.datetime.now().isoformat(),
        "time_str":  dt.datetime.now().strftime("%H:%M:%S"),
        "source":    source,
    }
    with _lock:
        _store[sym] = rec
    try:
        with _lock:
            snap = dict(_store)
        tmp = DATA_FILE.with_suffix(".tmp")
        with open(tmp,"w") as f:
            json.dump(snap, f)
        tmp.rename(DATA_FILE)
    except Exception:
        pass

def _load_disk():
    """On startup load persisted data (< 10 min old)."""
    try:
        if not DATA_FILE.exists(): return
        with open(DATA_FILE) as f:
            snap = json.load(f)
        with _lock:
            for sym, rec in snap.items():
                age = (dt.datetime.now()-dt.datetime.fromisoformat(rec["timestamp"])).total_seconds()
                if age < 600:
                    _store[sym] = rec
    except Exception:
        pass

def read_sym(symbol):
    """Return (df, spot, expiries, time_str, age_s, source) or None."""
    with _lock:
        rec = _store.get(symbol)
    if not rec:
        return None
    try:
        age = int((dt.datetime.now()-dt.datetime.fromisoformat(rec["timestamp"])).total_seconds())
        df  = pd.DataFrame(rec["rows"]) if rec["rows"] else pd.DataFrame()
        return df, float(rec["spot"]), list(rec["expiries"]), rec["time_str"], age, rec["source"]
    except Exception:
        return None

# ═══════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════
def _f(s):
    try:
        s = str(s).replace(",","").strip()
        if s in ("-","","nan","None","--","—","–"): return 0.0
        return float(s)
    except: return 0.0

def _nse_parse(j):
    if not j: return None,0.0,[]
    rec  = j.get("records",{})
    spot = float(rec.get("underlyingValue") or 0)
    exps = rec.get("expiryDates",[])
    rows = []
    for item in rec.get("data",[]):
        ce = item.get("CE",{}); pe = item.get("PE",{})
        rows.append({
            "strikePrice": float(item.get("strikePrice",0)),
            "expiryDate":  item.get("expiryDate",""),
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
    return (pd.DataFrame(rows) if rows else None), spot, exps

def _bse_parse(j):
    if not j: return None,0.0,[]
    spot = float(j.get("CurrentValue") or j.get("UnderlyingValue") or 0)
    rows = []
    for row in (j.get("Table") or []):
        sp = _f(row.get("StrikePrice",0))
        if sp<=0: continue
        rows.append({
            "strikePrice": sp,
            "expiryDate":  row.get("ExpiryDate",""),
            "CE_LTP":  _f(row.get("CE_LTP",0)),  "CE_OI":  _f(row.get("CE_OI",0)),
            "CE_changeOI": _f(row.get("CE_ChgOI",0)), "CE_volume": _f(row.get("CE_Volume",0)),
            "CE_IV":   _f(row.get("CE_IV",0)),   "CE_bid": _f(row.get("CE_BidPrice",0)),
            "CE_ask":  _f(row.get("CE_AskPrice",0)),
            "PE_LTP":  _f(row.get("PE_LTP",0)),  "PE_OI":  _f(row.get("PE_OI",0)),
            "PE_changeOI": _f(row.get("PE_ChgOI",0)), "PE_volume": _f(row.get("PE_Volume",0)),
            "PE_IV":   _f(row.get("PE_IV",0)),   "PE_bid": _f(row.get("PE_BidPrice",0)),
            "PE_ask":  _f(row.get("PE_AskPrice",0)),
        })
    exps = sorted({r["expiryDate"] for r in rows if r["expiryDate"]})
    return (pd.DataFrame(rows) if rows else None), spot, exps

# ═══════════════════════════════════════════════════
# SELENIUM SCRAPER THREAD
# ═══════════════════════════════════════════════════
class Scraper(threading.Thread):
    def __init__(self, nse_syms, bse_syms, visible=False, interval=60):
        super().__init__(daemon=True)
        self.nse     = nse_syms
        self.bse     = bse_syms
        self.visible = visible
        self.interval= max(interval, 30)
        self.driver  = None
        self.ns      = requests.Session()   # NSE session
        self.bs_     = requests.Session()   # BSE session
        self._ck_ts  = 0
        self._stop   = threading.Event()

    def _browser_start(self):
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            svc = Service(ChromeDriverManager().install())
        except ImportError:
            svc = Service()
        opts = Options()
        if not self.visible:
            opts.add_argument("--headless=new")
        for arg in ["--no-sandbox","--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                    "--window-size=1366,768",f"--user-agent={_UA}","--lang=en-IN"]:
            opts.add_argument(arg)
        opts.add_experimental_option("excludeSwitches",["enable-automation"])
        opts.add_experimental_option("useAutomationExtension",False)
        self.driver = webdriver.Chrome(service=svc, options=opts)
        self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument",{"source":"""
            Object.defineProperty(navigator,'webdriver',{get:()=>undefined});
            Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3,4,5]});
            Object.defineProperty(navigator,'languages',{get:()=>['en-IN','en','hi']});
            window.chrome={runtime:{}};
        """})
        _status["driver_ok"] = True
        _slog("✅ Chrome started")

    def _human(self):
        from selenium.webdriver.common.action_chains import ActionChains
        try:
            for _ in range(random.randint(2,4)):
                self.driver.execute_script(f"window.scrollBy(0,{random.randint(80,300)});")
                time.sleep(random.uniform(0.3,0.7))
            ActionChains(self.driver).move_by_offset(
                random.randint(80,400),random.randint(40,250)).perform()
            time.sleep(random.uniform(0.2,0.4))
        except Exception:
            pass

    def _refresh_nse(self):
        _slog("🔄 Refreshing NSE cookies via browser…")
        self.driver.get("https://www.nseindia.com/")
        time.sleep(random.uniform(3.5,5.0))
        self._human()
        self.driver.get("https://www.nseindia.com/option-chain")
        time.sleep(random.uniform(2.5,3.5))
        self._human()
        self.ns.cookies.clear()
        names = []
        for c in self.driver.get_cookies():
            self.ns.cookies.set(c["name"],c["value"],domain=c.get("domain",""))
            names.append(c["name"])
        self.ns.headers.update({
            "User-Agent":_UA,"Accept":"*/*",
            "Accept-Language":"en-IN,en;q=0.9","Accept-Encoding":"gzip,deflate,br",
            "Referer":"https://www.nseindia.com/option-chain",
            "X-Requested-With":"XMLHttpRequest","Sec-Fetch-Mode":"cors","Sec-Fetch-Site":"same-origin",
        })
        self._ck_ts = time.time()
        key = [n for n in names if n in ("nsit","nseappid","ak_bmsc","bm_sz","AKA_A2")]
        _slog(f"✅ NSE cookies: {', '.join(key) or 'basic set'}")

    def _refresh_bse(self):
        try:
            self.driver.get("https://www.bseindia.com/")
            time.sleep(random.uniform(2.0,3.0))
            self.bs_.cookies.clear()
            for c in self.driver.get_cookies():
                self.bs_.cookies.set(c["name"],c["value"],domain=c.get("domain",""))
            self.bs_.headers.update({
                "User-Agent":_UA,"Accept":"application/json,*/*",
                "Referer":"https://www.bseindia.com/","Origin":"https://www.bseindia.com",
            })
            _slog("✅ BSE cookies refreshed")
        except Exception as e:
            _slog(f"⚠️ BSE cookies: {e}")

    def _need_ck(self):
        return (time.time()-self._ck_ts) > COOKIE_TTL

    def _nse_get(self, url):
        try:
            r = self.ns.get(url, timeout=12)
            if r.status_code in (401,403):
                _slog(f"⚠️ NSE {r.status_code} — refreshing")
                self._refresh_nse()
                r = self.ns.get(url, timeout=12)
            return r.json() if r.status_code==200 else None
        except Exception as e:
            _slog(f"⚠️ NSE req: {e}"); return None

    def _nse_url(self, sym):
        if sym in NSE_INDICES:
            return f"https://www.nseindia.com/api/option-chain-indices?symbol={sym}"
        return f"https://www.nseindia.com/api/option-chain-equities?symbol={sym}"

    def _fetch_bse(self, sym):
        pid,scrip = BSE_MAP.get(sym,("1","SENSEX"))
        try:
            er = self.bs_.get(
                f"https://api.bseindia.com/BseIndiaAPI/api/ddlExpiry_Options/w"
                f"?productid={pid}&scripcode=&ExpiryDate=&optionType=CE&StrikePrice=&SeleScrip={scrip}",
                timeout=10)
            if er.status_code!=200: _slog(f"⚠️ BSE expiry {er.status_code}"); return
            exps = [x["ExpiryDate"] for x in (er.json().get("Table") or [])]
            if not exps: _slog(f"⚠️ BSE {sym}: no expiries"); return
            cr = self.bs_.get(
                f"https://api.bseindia.com/BseIndiaAPI/api/FnOChainData/w"
                f"?productid={pid}&scripcode=&ExpiryDate={exps[0]}&optionType=&StrikePrice=&SeleScrip={scrip}",
                timeout=12)
            if cr.status_code!=200: _slog(f"⚠️ BSE chain {cr.status_code}"); return
            df,spot,_ = _bse_parse(cr.json())
            if df is not None and spot>0:
                _save(sym,df,spot,exps,"BSE-Selenium")
                _slog(f"✅ {sym} ₹{spot:,.2f} | {len(df)} strikes")
            else:
                _slog(f"⚠️ BSE {sym}: empty response")
        except Exception as e:
            _slog(f"❌ BSE {sym}: {e}")

    def run(self):
        _status["running"]=True
        _slog("Scraper thread started")
        try:
            self._browser_start()
        except Exception as e:
            _status["error"]=str(e)
            _slog(f"❌ Chrome failed: {e}")
            _slog("Fix: pip install selenium webdriver-manager + install Google Chrome")
            _status["running"]=False; return
        try:
            self._refresh_nse()
            if self.bse: self._refresh_bse()
        except Exception as e:
            _slog(f"❌ Initial auth failed: {e}")
            _status["running"]=False; return

        errors=0
        while not self._stop.is_set():
            try:
                if self._need_ck():
                    self._refresh_nse()
                    if self.bse: self._refresh_bse()

                for sym in self.nse:
                    if self._stop.is_set(): break
                    j = self._nse_get(self._nse_url(sym))
                    if j:
                        df,spot,exps = _nse_parse(j)
                        if df is not None and spot>0:
                            _save(sym,df,spot,exps,"NSE-Selenium")
                            _slog(f"✅ {sym} ₹{spot:,.2f} | {len(df)} strikes")
                            errors=0
                        else:
                            _slog(f"⚠️ {sym}: empty"); errors+=1
                    else:
                        errors+=1
                    if len(self.nse)>1:
                        time.sleep(random.uniform(2,4))

                for sym in self.bse:
                    if self._stop.is_set(): break
                    self._fetch_bse(sym)
                    if len(self.bse)>1:
                        time.sleep(random.uniform(2,3))

                if errors>=4:
                    _slog(f"⚠️ {errors} errors — forcing re-auth")
                    self._refresh_nse(); errors=0

            except Exception as e:
                _slog(f"❌ Loop: {e}"); errors+=1; time.sleep(10)

            _slog(f"⏳ Next fetch in {self.interval}s")
            self._stop.wait(self.interval)

        try: self.driver.quit()
        except Exception: pass
        _status["running"]=False
        _slog("Scraper stopped")

    def stop(self):
        self._stop.set()

_scraper: Scraper = None

def ensure_scraper(nse,bse,visible=False,interval=60):
    global _scraper
    if _scraper and _scraper.is_alive(): return
    _scraper = Scraper(nse,bse,visible,interval)
    _scraper.start()

# ═══════════════════════════════════════════════════
# CSV PARSER  (NSE download: 2-row header, STRIKE anchor)
# ═══════════════════════════════════════════════════
def parse_nse_csv(raw, fname=""):
    debug=[]
    try: content=raw.decode("utf-8",errors="ignore")
    except Exception as e: return pd.DataFrame(),[f"Decode error: {e}"]
    expiry="Uploaded"
    m=re.search(r'(\d{2}-[A-Za-z]{3}-\d{4})',fname)
    if m: expiry=m.group(1)
    rows=list(_csv.reader(io.StringIO(content)))
    debug.append(f"Rows: {len(rows)}")
    sc=None; hr=None
    for ri in range(min(5,len(rows))):
        for ci,v in enumerate(rows[ri]):
            if str(v).strip().upper()=="STRIKE":
                sc=ci; hr=ri; debug.append(f"STRIKE col={ci} row={ri}"); break
        if sc is not None: break
    if sc is None: return pd.DataFrame(),debug+["No STRIKE column"]
    offs={"strikePrice":0,"CE_OI":-10,"CE_changeOI":-9,"CE_volume":-8,
          "CE_IV":-7,"CE_LTP":-6,"CE_bid":-3,"CE_ask":-2,
          "PE_bid":+2,"PE_ask":+3,"PE_LTP":+6,"PE_IV":+7,
          "PE_volume":+8,"PE_changeOI":+9,"PE_OI":+10}
    recs=[]
    for row in rows[hr+1:]:
        if len(row)<=sc: continue
        sp=_f(row[sc])
        if sp<=0: continue
        rec={"expiryDate":expiry}
        for col,off in offs.items():
            idx=sc+off
            rec[col]=_f(row[idx]) if 0<=idx<len(row) else 0.0
        recs.append(rec)
    if not recs: return pd.DataFrame(),debug+["No data rows"]
    df=pd.DataFrame(recs).fillna(0)
    debug.append(f"✅ {len(df)} strikes"); return df[df["strikePrice"]>0].reset_index(drop=True),debug

# ═══════════════════════════════════════════════════
# BSE PASTE PARSER  (copy-paste from bseindia.com)
# ═══════════════════════════════════════════════════
def parse_bse_paste(text, expiry="Uploaded"):
    debug=[]; lines=[l for l in text.strip().split("\n")]
    debug.append(f"Lines: {len(lines)}")
    ds=0
    for i,line in enumerate(lines):
        l=line.strip().upper()
        if l.startswith("CALLS") or l.startswith("CHG IN OI"): ds=i+1
        elif _f(l)>1000: ds=max(0,i-1); break
    recs=[]; i=ds
    while i<len(lines):
        ce=lines[i].strip()
        if not ce: i+=1; continue
        if i+1>=len(lines): break
        sp=_f(lines[i+1].strip())
        if sp<=100: i+=1; continue
        if i+2>=len(lines): break
        cf=ce.split("\t"); pf=lines[i+2].strip().split("\t")
        def g(a,n): return _f(a[n]) if n<len(a) else 0.0
        recs.append({
            "strikePrice":sp,"expiryDate":expiry,
            "CE_changeOI":g(cf,0),"CE_OI":g(cf,1),"CE_volume":g(cf,2),
            "CE_IV":g(cf,3),"CE_LTP":g(cf,4),"CE_bid":g(cf,7),"CE_ask":g(cf,8),
            "PE_bid":g(pf,1),"PE_ask":g(pf,2),"PE_LTP":g(pf,5),
            "PE_IV":g(pf,6),"PE_volume":g(pf,7),"PE_OI":g(pf,8),"PE_changeOI":g(pf,9),
        })
        i+=3
    if not recs: return pd.DataFrame(),debug+["No strikes found"]
    df=pd.DataFrame(recs).fillna(0); debug.append(f"✅ {len(df)} strikes")
    return df[df["strikePrice"]>0].reset_index(drop=True),debug

# ═══════════════════════════════════════════════════
# YFINANCE
# ═══════════════════════════════════════════════════
_yft={"last":0.0}
def _yw():
    d=time.time()-_yft["last"]
    if d<1.2: time.sleep(1.2-d+random.uniform(0.05,0.2))
    _yft["last"]=time.time()

@st.cache_data(ttl=60,show_spinner=False)
def get_spot(ysym):
    if not _HAS_YF: return 0.0
    _yw()
    try:
        fi=yf.Ticker(ysym).fast_info
        for a in ("last_price","lastPrice","regular_market_price","regularMarketPrice","previousClose"):
            v=getattr(fi,a,None)
            if v and float(v)>0: return float(v)
    except Exception: pass
    try:
        h=yf.Ticker(ysym).history(period="2d",interval="1d")
        if not h.empty: return float(h["Close"].squeeze().iloc[-1])
    except Exception: pass
    return 0.0

@st.cache_data(ttl=300,show_spinner=False)
def get_hist(ysym,period="60d"):
    if not _HAS_YF: return pd.DataFrame()
    _yw()
    try:
        df=yf.Ticker(ysym).history(period=period,interval="1d",auto_adjust=True)
        return df if not df.empty else pd.DataFrame()
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=120,show_spinner=False)
def get_yf_chain(sym,expiry):
    if not _HAS_YF: return None,None
    _yw()
    try:
        c=yf.Ticker(sym).option_chain(expiry); return c.calls.copy(),c.puts.copy()
    except Exception: return None,None

def build_yf_df(calls,puts,expiry):
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
            df[col]=df["strikePrice"].map(pi[src] if src in pi.columns else pd.Series(dtype=float)).fillna(0).values
        df["PE_IV"]=df["strikePrice"].map(pi["impliedVolatility"]*100 if "impliedVolatility" in pi.columns else pd.Series(dtype=float)).fillna(0).values
    else:
        for c in ["PE_LTP","PE_OI","PE_volume","PE_bid","PE_ask","PE_IV"]: df[c]=0.0
    df["PE_changeOI"]=0.0
    return df.fillna(0).reset_index(drop=True)

# ═══════════════════════════════════════════════════
# BLACK-SCHOLES + GREEKS
# ═══════════════════════════════════════════════════
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
    except Exception: return 0.20

def add_greeks(df,spot,expiry_str):
    if df is None or df.empty: return df
    try:
        fmt="%d-%b-%Y" if ("-" in str(expiry_str) and not str(expiry_str)[4:5].isdigit()) else "%Y-%m-%d"
        T=max((dt.datetime.strptime(str(expiry_str),fmt)-dt.datetime.now()).days/365.0,1/365)
    except Exception: T=7/365
    df=df.copy()
    for i,row in df.iterrows():
        K=float(row["strikePrice"])
        for kind,px in [("call","CE"),("put","PE")]:
            ltp=float(row.get(f"{px}_LTP",0)); ivp=float(row.get(f"{px}_IV",0))
            sig=ivp/100 if ivp>0.5 else calc_iv(ltp,spot,K,T,RFR,kind)
            sig=max(sig,0.01); g=bs(spot,K,T,RFR,sig,kind)
            df.at[i,f"{px}_delta"]=g["delta"]; df.at[i,f"{px}_gamma"]=g["gamma"]
            df.at[i,f"{px}_theta"]=g["theta"]; df.at[i,f"{px}_vega"]=g["vega"]
            if ivp==0 and ltp>0: df.at[i,f"{px}_IV"]=round(sig*100,2)
    return df

# ═══════════════════════════════════════════════════
# SIGNAL ENGINE
# ═══════════════════════════════════════════════════
def signals(df,spot):
    out=dict(pcr=0,max_pain=spot,atm=spot,straddle=0,resistance=spot,support=spot,
             atm_iv=20.0,skew=0.0,gamma_blast=False,abnormal=[],
             rec="⚪ NO TRADE",direction="NONE",conf=0,
             scalp="—",intraday="—",swing="—",pos="—",
             buy_strike_atm=0,buy_strike_otm=0,buy_side="CE",indicators=[])
    if df is None or df.empty or spot==0: return out
    df=df.copy(); df["strikePrice"]=df["strikePrice"].astype(float)
    ai=(df["strikePrice"]-spot).abs().idxmin(); atm=df.loc[ai]
    out["atm"]=float(atm["strikePrice"])
    out["straddle"]=round(float(atm.get("CE_LTP",0))+float(atm.get("PE_LTP",0)),2)
    ce_oi=float(df["CE_OI"].sum()); pe_oi=float(df["PE_OI"].sum())
    out["pcr"]=round(pe_oi/ce_oi,4) if ce_oi>0 else 0
    st_=df["strikePrice"].values; c_=df["CE_OI"].values; p_=df["PE_OI"].values
    pain=[sum(max(0,k-s)*o for k,o in zip(st_,c_))+sum(max(0,s-k)*o for k,o in zip(st_,p_)) for s in st_]
    out["max_pain"]=float(st_[int(np.argmin(pain))]) if pain else spot
    out["resistance"]=float(df.loc[df["CE_OI"].idxmax(),"strikePrice"])
    out["support"]=float(df.loc[df["PE_OI"].idxmax(),"strikePrice"])
    ce_iv=float(atm.get("CE_IV",0)); pe_iv=float(atm.get("PE_IV",0))
    out["atm_iv"]=(ce_iv+pe_iv)/2 if (ce_iv+pe_iv)>0 else 20.0
    out["skew"]=round(pe_iv-ce_iv,2)
    near=(df["CE_OI"].where(df["strikePrice"].between(spot*.99,spot*1.01),0).sum()+
          df["PE_OI"].where(df["strikePrice"].between(spot*.99,spot*1.01),0).sum())
    out["gamma_blast"]=(ce_oi+pe_oi)>0 and near/(ce_oi+pe_oi+1)>0.35
    iv=out["atm_iv"]; pcr=out["pcr"]; mp=out["max_pain"]; skew=out["skew"]
    score=50; direction="NONE"; inds=[]

    # PCR
    if pcr>1.5:   pts=12;direction="CALL";vrd="BULLISH";exp=f"PCR {pcr:.2f} — high put OI, contrarian bullish. Put writers building support, expect market to hold or rise."
    elif pcr>1.1: pts=6; direction="CALL";vrd="MILDLY BULLISH";exp=f"PCR {pcr:.2f} — mild put excess. Slight bullish bias."
    elif pcr<0.5: pts=12;direction="PUT"; vrd="BEARISH";exp=f"PCR {pcr:.2f} — very low. Call OI dominates. Contrarian says market may fall from here."
    elif pcr<0.9: pts=6; direction="PUT"; vrd="MILDLY BEARISH";exp=f"PCR {pcr:.2f} — mild call excess. Slight bearish lean."
    else:         pts=0; vrd="NEUTRAL";exp=f"PCR {pcr:.2f} — balanced. No directional bias."
    score+=pts; inds.append(("Put-Call Ratio",f"{pcr:.3f}",vrd,f"{pts:+d}",exp))

    # IV
    if iv>50:   pts=-25;vrd="VERY EXPENSIVE — avoid";exp=f"IV {iv:.0f}%: Options massively overpriced. Premium will crash after event. Sell strategies only."
    elif iv>35: pts=-12;vrd="EXPENSIVE — risky to buy";exp=f"IV {iv:.0f}%: High IV. Theta erodes fast. Only enter on extremely strong setup."
    elif iv<15: pts=15; vrd="✅ CHEAP — best time to buy";exp=f"IV {iv:.0f}%: Options cheap vs history. Low entry cost + IV expansion potential = ideal buying conditions."
    elif iv<25: pts=10; vrd="REASONABLE — good conditions";exp=f"IV {iv:.0f}%: Moderate IV. Fair conditions on strong directional signal."
    else:       pts=4;  vrd="MODERATE";exp=f"IV {iv:.0f}%: Slightly elevated. Be selective."
    score+=pts; inds.append(("Implied Volatility",f"{iv:.1f}%",vrd,f"{pts:+d}",exp))

    # Max Pain
    mp_pct=(mp-spot)/spot*100 if spot>0 else 0
    if mp_pct>2 and direction=="CALL":   pts=10;vrd="✅ BULLISH gravity";exp=f"Max pain ₹{mp:,.0f} is {mp_pct:.1f}% above spot. Market pulled upward near expiry."
    elif mp_pct<-2 and direction=="PUT": pts=10;vrd="✅ BEARISH gravity";exp=f"Max pain ₹{mp:,.0f} is {abs(mp_pct):.1f}% below spot. Downward pull near expiry."
    elif abs(mp_pct)<0.5:               pts=2; vrd="NEUTRAL — at max pain";exp=f"Spot at max pain ₹{mp:,.0f}. Market may stay flat."
    else:                               pts=-5;vrd="AGAINST direction";exp=f"Max pain {mp_pct:+.1f}% works against signal direction."
    score+=pts; inds.append(("Max Pain",f"₹{mp:,.0f} ({mp_pct:+.1f}%)",vrd,f"{pts:+d}",exp))

    # Skew
    if abs(skew)>0:
        if skew>8:    pts=5 if direction=="PUT" else -4;vrd="PUT SKEW — bearish hedge";exp=f"PE IV {pe_iv:.0f}% > CE IV {ce_iv:.0f}%. Institutions buying downside protection."
        elif skew<-8: pts=5 if direction=="CALL" else -4;vrd="CALL SKEW — bullish chase";exp=f"CE IV {ce_iv:.0f}% > PE IV {pe_iv:.0f}%. Aggressive call buying by big players."
        else:         pts=0;vrd="BALANCED";exp=f"CE IV ≈ PE IV. No directional skew."
        score+=pts; inds.append(("IV Skew (PE-CE)",f"{skew:+.1f}%",vrd,f"{pts:+d}",exp))

    # OI Change
    ce_chg=float(df["CE_changeOI"].sum()); pe_chg=float(df["PE_changeOI"].sum())
    if (abs(ce_chg)+abs(pe_chg))>0:
        if pe_chg>ce_chg*1.3 and direction=="CALL":   pts=7; vrd="✅ Fresh put writing — bullish";exp=f"+{pe_chg/1000:.0f}K PE OI added. Put writers building floor, expect market holds."
        elif ce_chg>pe_chg*1.3 and direction=="PUT":  pts=7; vrd="✅ Fresh call writing — bearish";exp=f"+{ce_chg/1000:.0f}K CE OI added. Call writers building ceiling, expect market falls."
        elif ce_chg>pe_chg*1.3 and direction=="CALL": pts=-5;vrd="⚠️ Call writing vs bullish signal";exp=f"Call writers ({ce_chg/1000:.0f}K) while signal says bullish. Smart money disagrees."
        else:                                          pts=0; vrd="MIXED";exp=f"CE {ce_chg/1000:+.0f}K / PE {pe_chg/1000:+.0f}K. No clear OI edge."
        score+=pts; inds.append(("OI Change Today",f"CE {ce_chg/1000:+.0f}K / PE {pe_chg/1000:+.0f}K",vrd,f"{pts:+d}",exp))

    # S/R
    res=out["resistance"]; sup=out["support"]
    if sup<spot<res:  pts=6; vrd="IN BUY ZONE";exp=f"Spot ₹{spot:,.0f} between S ₹{sup:,.0f} and R ₹{res:,.0f}. Room to move both ways."
    elif spot>res:    pts=8 if direction=="CALL" else -5;vrd="BREAKOUT above R" if direction=="CALL" else "EXTENDED above R";exp=f"Spot broke above resistance ₹{res:,.0f}. {'Bullish continuation.' if direction=='CALL' else 'Risky to short here.'}"
    elif spot<sup:    pts=8 if direction=="PUT" else -5; vrd="BREAKDOWN below S" if direction=="PUT" else "EXTENDED below S";exp=f"Spot broke below support ₹{sup:,.0f}. {'Bearish continuation.' if direction=='PUT' else 'Risky to buy calls here.'}"
    else:             pts=0; vrd="AT KEY LEVEL";exp="At a key S/R level. Wait for clear break."
    score+=pts; inds.append(("S/R Location",f"S:{sup:,.0f} | Now:{spot:,.0f} | R:{res:,.0f}",vrd,f"{pts:+d}",exp))

    score=min(max(int(score),0),100); out["conf"]=score; out["direction"]=direction; out["indicators"]=inds

    # Specific strike
    atm_s=int(out["atm"])
    step=round(df["strikePrice"].diff().dropna().abs().mode().iloc[0]) if len(df)>1 else 50
    side="CE" if direction=="CALL" else "PE"
    otm_s=atm_s+step if direction=="CALL" else atm_s-step
    otm2_s=atm_s+2*step if direction=="CALL" else atm_s-2*step
    out["buy_strike_atm"]=atm_s; out["buy_strike_otm"]=otm_s; out["buy_side"]=side

    def ltp(k):
        r=df[df["strikePrice"]==k]; v=float(r[f"{side}_LTP"].values[0]) if not r.empty and f"{side}_LTP" in r.columns else 0
        return f"₹{v:.2f}" if v>0 else "—"

    if score>=72 and direction in ("CALL","PUT"):
        dw="UP" if direction=="CALL" else "DOWN"
        out["rec"]=f"{'🟢' if direction=='CALL' else '🔴'} BUY {side} — {dw}"
        out["scalp"]   =f"Buy {atm_s} {side} @ {ltp(atm_s)} | Target +25% | SL -30%"
        out["intraday"]=f"Buy {atm_s} {side} @ {ltp(atm_s)} | Target +60% | SL -30%"
        out["swing"]   =f"Buy {otm_s} {side} @ {ltp(otm_s)} (next expiry) | Target +90% | SL -30%"
        out["pos"]     =f"Buy {otm2_s} {side} @ {ltp(otm2_s)} (far expiry) | Hold for big move"
    elif score>=58:
        out["rec"]="🟡 WATCH — forming"
        out["scalp"]=f"Wait | Potential: {atm_s} {side}"
        out["intraday"]=out["swing"]=out["pos"]="Wait for confirmation"
    else:
        out["rec"]="⚪ NO TRADE — stay out"
        out["scalp"]=out["intraday"]=out["swing"]=out["pos"]="No edge. Cash is a position."

    if iv>55:       out["abnormal"].append(f"🔴 IV {iv:.0f}%: Very expensive — don't buy options now")
    elif 0<iv<15:   out["abnormal"].append(f"✅ IV {iv:.0f}%: Cheap — best time to buy")
    if abs(skew)>8: out["abnormal"].append(f"📊 Skew {skew:+.0f}%: {'Bearish hedge by big money' if skew>0 else 'Bullish call chase'}")
    if abs(mp_pct)>2: out["abnormal"].append(f"🎯 Max pain ₹{mp:,.0f} ({mp_pct:+.1f}%) — gravity near expiry")
    if out["gamma_blast"]: out["abnormal"].append("⚡ GAMMA BLAST — huge OI at ATM, options explode on any breakout")
    return out

# ═══════════════════════════════════════════════════
# OI NARRATIVE
# ═══════════════════════════════════════════════════
def oi_narrative(df,spot,sig):
    if df is None or df.empty: return ""
    res=sig["resistance"]; sup=sig["support"]; mp=sig["max_pain"]
    ce_w=", ".join([f"₹{int(r['strikePrice']):,}" for _,r in df.nlargest(3,"CE_OI").iterrows()])
    pe_w=", ".join([f"₹{int(r['strikePrice']):,}" for _,r in df.nlargest(3,"PE_OI").iterrows()])
    cc=df["CE_changeOI"].sum(); pc=df["PE_changeOI"].sum()
    if cc>0 and pc>0:  chg=f"Both CE (+{cc/1000:.0f}K) and PE (+{pc/1000:.0f}K) OI rising — volatile setup, big positions being built."
    elif pc>0 and cc<=0: chg=f"Put writers active (+{pc/1000:.0f}K PE) while call writers exit. Floor being built. BULLISH positioning."
    elif cc>0 and pc<=0: chg=f"Call writers active (+{cc/1000:.0f}K CE) while put writers exit. Ceiling being built. BEARISH positioning."
    else:              chg=f"Both sides unwinding (CE {cc/1000:.0f}K, PE {pc/1000:.0f}K). Positions closing — indecisive or near a big move."
    d2=df.copy()
    pc_=((d2["CE_OI"]-d2["CE_changeOI"]).clip(lower=1)); pp_=((d2["PE_OI"]-d2["PE_changeOI"]).clip(lower=1))
    d2["cpct"]=(d2["CE_changeOI"]/pc_*100).fillna(0); d2["ppct"]=(d2["PE_changeOI"]/pp_*100).fillna(0)
    mc=d2.loc[d2["cpct"].idxmax()]; mp_=d2.loc[d2["ppct"].idxmax()]
    pct=f"Biggest surge: <b>{int(mc['strikePrice']):,} CE +{mc['cpct']:.0f}%</b> and <b>{int(mp_['strikePrice']):,} PE +{mp_['ppct']:.0f}%</b> — aggressive positioning at these strikes today."
    if spot>res:   loc=f"⚠️ Spot ₹{spot:,.0f} <b>above resistance ₹{res:,.0f}</b> — breakout territory."
    elif spot<sup: loc=f"⚠️ Spot ₹{spot:,.0f} <b>below support ₹{sup:,.0f}</b> — breakdown territory."
    else:          loc=f"Spot ₹{spot:,.0f} inside range: support ₹{sup:,.0f} → resistance ₹{res:,.0f}."
    return (f"<b>🔴 CE walls (resistance):</b> {ce_w}<br>"
            f"<b>🟢 PE walls (support):</b> {pe_w}<br>"
            f"<b>📊 OI flow today:</b> {chg}<br>"
            f"<b>🔥 Biggest moves today:</b> {pct}<br>"
            f"<b>📍 Location:</b> {loc}<br>"
            f"<b>🎯 Max pain:</b> ₹{mp:,.0f} — market drifts here near expiry, causing maximum pain for option buyers.")

# ═══════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════
def _vl(fig,x,lbl="",color="yellow",dash="dash",row=1,col=1):
    fig.add_vline(x=x,line_dash=dash,line_color=color,line_width=1.5,
                  annotation_text=lbl,annotation_font_color=color,row=row,col=col)

def chart_chain(df,spot):
    rng=spot*0.05; sub=df[df["strikePrice"].between(spot-rng,spot+rng)]
    if sub.empty: sub=df
    fig=make_subplots(2,2,subplot_titles=["CE vs PE Premium","Open Interest","IV Smile","Volume"])
    x=sub["strikePrice"]
    fig.add_trace(go.Bar(name="CE LTP",x=x,y=sub["CE_LTP"],marker_color="#00ff88",opacity=.85),1,1)
    fig.add_trace(go.Bar(name="PE LTP",x=x,y=sub["PE_LTP"],marker_color="#ff3b5c",opacity=.85),1,1)
    fig.add_trace(go.Bar(name="CE OI",x=x,y=sub["CE_OI"],marker_color="#00e5ff",opacity=.8),1,2)
    fig.add_trace(go.Bar(name="PE OI",x=x,y=sub["PE_OI"],marker_color="#ff9500",opacity=.8),1,2)
    fig.add_trace(go.Scatter(name="CE IV",x=x,y=sub["CE_IV"],mode="lines+markers",line=dict(color="#00ff88",width=2)),2,1)
    fig.add_trace(go.Scatter(name="PE IV",x=x,y=sub["PE_IV"],mode="lines+markers",line=dict(color="#ff3b5c",width=2)),2,1)
    fig.add_trace(go.Bar(name="CE Vol",x=x,y=sub["CE_volume"],marker_color="rgba(0,255,136,0.4)"),2,2)
    fig.add_trace(go.Bar(name="PE Vol",x=x,y=sub["PE_volume"],marker_color="rgba(255,59,92,0.4)"),2,2)
    for r in [1,2]:
        for c in [1,2]: _vl(fig,spot,"Spot",row=r,col=c)
    fig.update_layout(template=DARK,height=520,barmode="group",title="Option Chain Overview",margin=dict(t=50,b=10))
    return fig

def chart_oi(df,spot,sig):
    rng=spot*0.055; sub=df[df["strikePrice"].between(spot-rng,spot+rng)]
    if sub.empty: sub=df
    sub=sub.copy()
    pc_=(sub["CE_OI"]-sub["CE_changeOI"]).clip(lower=1); pp_=(sub["PE_OI"]-sub["PE_changeOI"]).clip(lower=1)
    sub["cpct"]=(sub["CE_changeOI"]/pc_*100).fillna(0); sub["ppct"]=(sub["PE_changeOI"]/pp_*100).fillna(0)
    fig=make_subplots(3,1,subplot_titles=["Total OI","OI Change Today","% OI Change Today"],vertical_spacing=0.1)
    x=sub["strikePrice"]
    fig.add_trace(go.Bar(name="CE OI",x=x,y=sub["CE_OI"],marker_color="#00e5ff",opacity=.8),1,1)
    fig.add_trace(go.Bar(name="PE OI",x=x,y=sub["PE_OI"],marker_color="#ff9500",opacity=.8),1,1)
    fig.add_trace(go.Bar(name="CE chg",x=x,y=sub["CE_changeOI"],marker_color=["#00ff88" if v>=0 else "#ff3b5c" for v in sub["CE_changeOI"]]),2,1)
    fig.add_trace(go.Bar(name="PE chg",x=x,y=sub["PE_changeOI"],marker_color=["#ff9500" if v>=0 else "#8888ff" for v in sub["PE_changeOI"]]),2,1)
    fig.add_trace(go.Bar(name="CE%",x=x,y=sub["cpct"],marker_color=["rgba(0,229,255,0.6)" if v>=0 else "rgba(255,59,92,0.5)" for v in sub["cpct"]]),3,1)
    fig.add_trace(go.Bar(name="PE%",x=x,y=sub["ppct"],marker_color=["rgba(255,149,0,0.6)" if v>=0 else "rgba(136,136,255,0.5)" for v in sub["ppct"]]),3,1)
    for row in [1,2,3]:
        _vl(fig,spot,"Spot",row=row)
        _vl(fig,sig["resistance"],"R",color="#ff3b5c",dash="dot",row=row,col=1)
        _vl(fig,sig["support"],"S",color="#00ff88",dash="dot",row=row,col=1)
    fig.update_layout(template=DARK,height=680,barmode="group",title="OI Analysis",margin=dict(t=50,b=10))
    return fig

def chart_greeks(df,spot):
    rng=spot*0.04; sub=df[df["strikePrice"].between(spot-rng,spot+rng)]
    if sub.empty: sub=df
    fig=make_subplots(2,2,subplot_titles=["Delta","Gamma","Theta (₹/day)","Vega"])
    x=sub["strikePrice"]
    for r,c,ck,pk in [(1,1,"CE_delta","PE_delta"),(1,2,"CE_gamma","PE_gamma"),(2,1,"CE_theta","PE_theta"),(2,2,"CE_vega","PE_vega")]:
        for col,clr in [(ck,"#00ff88"),(pk,"#ff3b5c")]:
            if col in sub.columns:
                fig.add_trace(go.Scatter(name=col,x=x,y=sub[col],mode="lines",line=dict(color=clr,width=2)),r,c)
        _vl(fig,spot,row=r,col=c)
    fig.update_layout(template=DARK,height=520,title="Greeks",margin=dict(t=50,b=10))
    return fig

def chart_straddle(df,spot,yfsym,straddle_now):
    figs=[]; texts=[]; val_msg=""; val_clr="#5a7a9a"
    if df is not None and not df.empty:
        rng=spot*0.04; sub=df[df["strikePrice"].between(spot-rng,spot+rng)]
        if sub.empty: sub=df
        sub=sub.copy(); sub["straddle"]=sub["CE_LTP"]+sub["PE_LTP"]
        fig1=go.Figure()
        fig1.add_trace(go.Bar(name="CE",x=sub["strikePrice"],y=sub["CE_LTP"],marker_color="#00ff88",opacity=0.8))
        fig1.add_trace(go.Bar(name="PE",x=sub["strikePrice"],y=sub["PE_LTP"],marker_color="#ff3b5c",opacity=0.8))
        fig1.add_trace(go.Scatter(name="Straddle",x=sub["strikePrice"],y=sub["straddle"],mode="lines+markers",line=dict(color="yellow",width=2.5)))
        fig1.add_vline(x=spot,line_color="#00e5ff",line_dash="dash",annotation_text=f"Spot ₹{spot:,.0f}",annotation_font_color="#00e5ff")
        ai=(sub["strikePrice"]-spot).abs().idxmin(); atm_v=float(sub.loc[ai,"straddle"])
        if atm_v>0:
            fig1.add_annotation(x=float(sub.loc[ai,"strikePrice"]),y=atm_v,text=f"ATM straddle = ₹{atm_v:.2f}",
                                showarrow=True,arrowhead=2,font=dict(color="yellow",size=12),bgcolor="#1a1a00",bordercolor="yellow")
        fig1.update_layout(template=DARK,height=320,barmode="stack",title="Straddle Cost (from live chain)",xaxis_title="Strike",yaxis_title="₹")
        figs.append(fig1)
        texts.append(f"Yellow = CE+PE straddle cost at each strike. <b>ATM straddle = ₹{atm_v:.2f}</b>. Market must move more than this by expiry for straddle buyer to profit. Cheaper straddle = better to buy; expensive = sell or wait.")
    hist=get_hist(yfsym,"60d")
    if not hist.empty:
        close=hist["Close"].squeeze().astype(float)
        rv7=close.pct_change().rolling(7).std()*np.sqrt(252)
        est=(close*rv7*np.sqrt(7/252)*0.8).dropna()
        if not est.empty:
            p10,p25,p50,p75,p90=est.quantile([.10,.25,.50,.75,.90])
            fig2=go.Figure()
            fig2.add_trace(go.Scatter(x=est.index,y=est,name="Estimated straddle",line=dict(color="#00e5ff",width=2),fill="tozeroy",fillcolor="rgba(0,229,255,0.04)"))
            for v,l,c in [(p10,"Cheap P10","#00ff88"),(p25,"Low P25","#66cc66"),(p50,"Average","#ff9500"),(p75,"High P75","#ff6633"),(p90,"Expensive P90","#ff3b5c")]:
                fig2.add_hline(y=v,line_color=c,line_dash="dot",annotation_text=f"{l}: ₹{v:.0f}",annotation_font_color=c)
            if straddle_now>0:
                fig2.add_hline(y=straddle_now,line_color="yellow",line_dash="solid",line_width=2,annotation_text=f"Today: ₹{straddle_now:.0f}",annotation_font_color="yellow")
            fig2.update_layout(template=DARK,height=300,title="ATM Straddle vs 60-Day History",xaxis_title="Date",yaxis_title="₹")
            figs.append(fig2)
            if straddle_now>0:
                pct=(est<straddle_now).mean()*100
                if pct<25:   val_msg=f"✅ Straddle ₹{straddle_now:.0f} at {pct:.0f}th percentile — CHEAP vs 60 days. Great buying conditions."; val_clr="#00ff88"
                elif pct>75: val_msg=f"⚠️ Straddle ₹{straddle_now:.0f} at {pct:.0f}th percentile — EXPENSIVE. Overpaying. IV will fall after next move."; val_clr="#ff3b5c"
                else:        val_msg=f"🟡 Straddle ₹{straddle_now:.0f} at {pct:.0f}th percentile — FAIR VALUE. Only buy on strong signals."; val_clr="#ff9500"
            texts.append(f"Percentile bands from 60 days of price history. P10=cheapest, P90=most expensive. {val_msg}")
    return figs,texts,val_msg,val_clr

@st.cache_data(ttl=300,show_spinner=False)
def backtest(yfsym,lookback,capital,pos_pct):
    raw=get_hist(yfsym,f"{lookback}d")
    if raw.empty or len(raw)<15: return None,"Not enough data."
    close=raw["Close"].squeeze().astype(float)
    v7=close.pct_change().rolling(7).std()*np.sqrt(252)*100
    s10=close.rolling(10).mean(); s20=close.rolling(20).mean()
    v3=close.pct_change().rolling(3).std()*np.sqrt(252)*100; v10=close.pct_change().rolling(10).std()*np.sqrt(252)*100
    trades=[]; equity=float(capital); curve=[equity]
    for i in range(20,len(close)-1):
        iv=float(v7.iloc[i]) if not np.isnan(v7.iloc[i]) else 20.0; spot_=float(close.iloc[i])
        vv3=float(v3.iloc[i]) if not np.isnan(v3.iloc[i]) else 20.0; vv10=float(v10.iloc[i]) if not np.isnan(v10.iloc[i]) else 20.0
        pcr=1.0+(vv10-vv3)/(vv10+1); mp_=(float(s10.iloc[i])+float(s20.iloc[i]))/2
        sc=50; dr="NONE"
        if pcr>1.5: sc+=12;dr="CALL"
        elif pcr>1.1: sc+=6;dr="CALL"
        elif pcr<0.5: sc+=12;dr="PUT"
        elif pcr<0.9: sc+=6;dr="PUT"
        if iv>50: sc-=25
        elif iv>35: sc-=12
        elif iv<18: sc+=15
        elif iv<=30: sc+=8
        if spot_<mp_ and dr=="CALL": sc+=10
        elif spot_>mp_ and dr=="PUT": sc+=10
        elif spot_<mp_ and dr=="PUT": sc-=8
        elif spot_>mp_ and dr=="CALL": sc-=8
        if float(s10.iloc[i])>float(s20.iloc[i]) and dr=="CALL": sc+=7
        elif float(s10.iloc[i])<float(s20.iloc[i]) and dr=="PUT": sc+=7
        sc=min(max(int(sc),0),100); curve.append(equity)
        if sc<72 or dr=="NONE": continue
        opt=spot_*max(iv/100,0.01)*np.sqrt(7/365)*0.4
        if opt<=0: continue
        ret=(float(close.iloc[i+1])-spot_)/spot_
        rg=(ret*0.5*spot_) if dr=="CALL" else (-ret*0.5*spot_)
        pp=max(min((rg-opt/7)/opt,2.5),-0.5); pnl=equity*(pos_pct/100)*pp; equity=max(equity+pnl,0)
        trades.append({"Date":raw.index[i].strftime("%d-%b-%Y"),"Dir":dr,"Spot":round(spot_,2),
            "OptPx":round(opt,2),"IV%":round(iv,1),"Score":sc,"Move%":round(ret*100,2),
            "P&L₹":round(pnl,2),"P&L%":round(pp*100,2),"Equity":round(equity,2),
            "Result":"✅ WIN" if pnl>0 else "❌ LOSS"})
        curve[-1]=equity
    if not trades: return None,"No trades generated."
    tdf=pd.DataFrame(trades); wins=(tdf["P&L₹"]>0).sum(); tot=len(tdf)
    aw=tdf.loc[tdf["P&L₹"]>0,"P&L₹"].mean() if wins>0 else 0
    al=tdf.loc[tdf["P&L₹"]<=0,"P&L₹"].mean() if (tot-wins)>0 else 0
    peak=capital; mdd=0
    for eq in curve:
        if eq>peak: peak=eq
        mdd=max(mdd,(peak-eq)/peak*100)
    return tdf,{"total":tot,"wins":int(wins),"wr":round(wins/tot*100,1),"pnl":round(tdf["P&L₹"].sum(),2),
        "aw":round(aw,2),"al":round(al,2),"rr":round(abs(aw/al),2) if al else 0,"mdd":round(mdd,2),
        "final":round(equity,2),"ret%":round((equity-capital)/capital*100,2),"curve":curve}

# ═══════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════
st.set_page_config(page_title="⚡ ProOptions India",page_icon="⚡",layout="wide",initial_sidebar_state="expanded")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
:root{--bg:#060a10;--s1:#0d1520;--s2:#111d2e;--acc:#00e5ff;--grn:#00ff88;--red:#ff3b5c;--ora:#ff9500;--txt:#c8d6e5;--mut:#5a7a9a;--brd:#1e3050}
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:var(--bg)!important;color:var(--txt)!important}
.stApp{background:var(--bg)}.stMetric{background:var(--s1)!important;border:1px solid var(--brd)!important;border-radius:10px!important;padding:12px!important}
.stMetric label{color:var(--mut)!important;font-size:10px!important;letter-spacing:1px;text-transform:uppercase}
.stMetric [data-testid="stMetricValue"]{color:var(--acc)!important;font-family:'Space Mono',monospace;font-size:18px!important}
div[data-testid="stTabs"] button{color:var(--mut)!important;font-size:11px}
div[data-testid="stTabs"] button[aria-selected="true"]{color:var(--acc)!important;border-bottom:2px solid var(--acc)!important}
h1,h2,h3,h4{font-family:'Space Mono',monospace!important}h1,h2,h3{color:var(--acc)!important}
.stButton>button{background:transparent!important;border:1px solid var(--acc)!important;color:var(--acc)!important;font-family:'Space Mono',monospace!important;border-radius:8px!important;font-size:12px!important}
.stButton>button:hover{background:var(--acc)!important;color:#000!important}
.exp{background:var(--s2);border-left:3px solid var(--acc);border-radius:0 8px 8px 0;padding:14px 18px;margin:10px 0;font-size:14px;line-height:1.7}
.exp b{color:var(--acc)}.pe{background:#0a1520;border:1px solid #1a3050;border-radius:8px;padding:12px 16px;margin:6px 0 14px;font-size:13px;line-height:1.75;color:#9ab5cc}
.sb{background:var(--s1);border:1px solid var(--brd);border-radius:12px;padding:18px 22px;margin:10px 0;position:relative;overflow:hidden}
.sb::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--acc),var(--grn))}
.ir{display:flex;justify-content:space-between;align-items:flex-start;padding:10px 14px;background:#0a1929;border:1px solid #1a3050;border-radius:8px;margin:5px 0;gap:10px;flex-wrap:wrap}
.tag-b{background:#002a15;color:#00ff88;border:1px solid #00aa44;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;display:inline-block;margin:3px 2px}
.tag-w{background:#2a1a00;color:#ff9500;border:1px solid #aa5500;padding:4px 12px;border-radius:20px;font-size:12px;display:inline-block;margin:3px 2px}
.al{background:#1a0a00;border-left:3px solid var(--ora);border-radius:0 6px 6px 0;padding:10px 14px;margin:4px 0;font-size:13px}
.gb{background:#1a000a;border:2px solid #ff3b5c;border-radius:10px;padding:14px;animation:pulse 2s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(255,59,92,.5)}70%{box-shadow:0 0 0 10px rgba(255,59,92,0)}100%{box-shadow:0 0 0 0 rgba(255,59,92,0)}}
.tc{background:var(--s2);border:1px solid var(--brd);border-radius:10px;padding:14px;margin:6px 0}
.tc-w{border-left:3px solid var(--grn)}.tc-l{border-left:3px solid var(--red)}.tc-o{border-left:3px solid var(--acc)}
.oin{background:#0a1929;border:1px solid var(--brd);border-radius:10px;padding:14px 18px;margin:8px 0;font-size:13px;line-height:1.85}
.oin b{color:var(--acc)}.sok{color:#00ff88;font-size:12px;font-family:'Space Mono',monospace}
.serr{color:#ff3b5c;font-size:12px;font-family:'Space Mono',monospace}
.src{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-family:'Space Mono',monospace;font-weight:700}
.src-a{background:#003a1a;color:#00ff88;border:1px solid #00aa44}
.src-b{background:#001a3a;color:#00e5ff;border:1px solid #0066aa}
.src-c{background:#2a1500;color:#ff9500;border:1px solid #884400}
hr{border-color:var(--brd)!important}
</style>""",unsafe_allow_html=True)

for k,v in {"trades":[],"active":[],"bt":None}.items():
    if k not in st.session_state: st.session_state[k]=v

_load_disk()

# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════
def main():
    st.markdown("<h1 style='font-size:22px;letter-spacing:2px'>⚡ PRO OPTIONS — NSE + BSE INDIA</h1>",unsafe_allow_html=True)
    st.markdown("<p style='color:#5a7a9a;margin-top:-10px'>Selenium live scraper · NIFTY · BANKNIFTY · SENSEX · 80 F&O stocks · Greeks · OI · Signals · Backtest</p>",unsafe_allow_html=True)

    # ─── SIDEBAR ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Scraper Control")
        st.markdown('<span style="font-size:11px;color:#5a7a9a">Select ALL symbols to monitor. Selenium fetches them every N seconds via real Chrome browser.</span>',unsafe_allow_html=True)
        nse_idx = st.multiselect("NSE Indices",NSE_INDICES,default=["NIFTY","BANKNIFTY"])
        bse_idx = st.multiselect("BSE Indices",BSE_INDICES,default=["SENSEX"])
        stocks  = st.multiselect("F&O Stocks",NSE_FNO,default=[])
        interval= st.slider("Fetch interval (s)",30,300,60,10)
        show_br = st.checkbox("Show browser window",False)
        all_nse = list(dict.fromkeys(nse_idx+stocks))
        all_bse = list(dict.fromkeys(bse_idx))
        c1,c2=st.columns(2)
        start=c1.button("▶ Start",type="primary",use_container_width=True)
        stop =c2.button("⏹ Stop",use_container_width=True)
        if start:
            global _scraper
            if _scraper and _scraper.is_alive(): _scraper.stop(); time.sleep(0.5)
            ensure_scraper(all_nse,all_bse,visible=show_br,interval=interval)
        if stop and _scraper: _scraper.stop()
        running=_scraper is not None and _scraper.is_alive()
        st.markdown(f'<div class="{"sok" if running else "serr"}">{"🟢 SCRAPER RUNNING" if running else "🔴 SCRAPER STOPPED"}</div>',unsafe_allow_html=True)
        if _status.get("error"): st.error(_status["error"])
        with st.expander("📋 Live log"):
            with _lock: logs=list(reversed(_logs[-20:]))
            st.markdown("<br>".join(f'<span style="font-size:11px;color:#7a9ab5">{l}</span>' for l in logs),unsafe_allow_html=True)
        st.divider()
        st.markdown("### 📊 View Symbol")
        mkt=st.selectbox("Data type",["🇮🇳 NSE/BSE India","🌐 Global (yFinance)","📊 Spot Only"])
        if "India" in mkt:
            avail=sorted(_store.keys())
            sym=st.selectbox("Symbol",avail if avail else ["NIFTY"],key="india_sym")
            st.markdown("**Fallbacks (if scraper not running):**")
            csv_file=st.file_uploader("NSE CSV",type=["csv"])
            bse_exp=st.text_input("BSE expiry (dd-Mon-yyyy)","06-Mar-2026")
            bse_paste_=st.text_area("BSE paste",height=100,placeholder="Paste bseindia.com table…")
        elif "Global" in mkt:
            ch=st.selectbox("Instrument",list(YF_OPTS.keys()))
            sym=st.text_input("Ticker","AAPL").upper() if ch=="Custom" else YF_OPTS[ch]
            csv_file=None;bse_paste_=None;bse_exp=""
        else:
            ch=st.selectbox("Instrument",list(YF_SPOT.keys()))
            sym=st.text_input("Ticker","BTC-USD").upper() if ch=="Custom" else YF_SPOT[ch]
            csv_file=None;bse_paste_=None;bse_exp=""
        st.divider()
        st.markdown("### 💰 Risk")
        capital=st.number_input("Capital ₹",50000,10000000,100000,10000)
        pos_pct=st.slider("% per trade",2,15,5)
        sl_pct =st.slider("Stop loss %",20,50,30)
        tgt_pct=st.slider("Target %",30,200,60)
        st.divider()
        if st.button("🗑️ Clear Cache",use_container_width=True): st.cache_data.clear();st.rerun()
        auto=st.checkbox("Auto-refresh UI (60s)")
        st.caption("Educational only · Not financial advice")

    # ─── DATA ──────────────────────────────────────────────────────────────
    fetch_sym=sym; is_india="India" in mkt; is_spot="Spot" in mkt
    yfsym=NSE_TO_YF.get(fetch_sym,fetch_sym+".NS") if is_india else fetch_sym
    df_exp=pd.DataFrame();spot=0.0;expiries=[];sel_expiry="";data_src=""
    ph=st.empty(); has_opts=not is_spot

    if is_india:
        # 1. Live scraper data
        jd=read_sym(fetch_sym)
        if jd:
            df_j,spot_j,exps_j,ts_j,age_j,src_j=jd
            if not df_j.empty and spot_j>0:
                df_exp=df_j;spot=spot_j;expiries=exps_j;data_src=src_j
                ph.success(f"✅ **{src_j}** | ₹{spot:,.2f} | Updated {ts_j} ({age_j}s ago)")
        # 2. CSV fallback
        if df_exp.empty and csv_file is not None:
            raw=csv_file.read();fname=getattr(csv_file,"name","")
            df_csv,dbg=parse_nse_csv(raw,fname)
            if not df_csv.empty:
                df_exp=df_csv;data_src="CSV";spot=get_spot(yfsym)
                expiries=df_csv["expiryDate"].unique().tolist()
                ph.success(f"✅ CSV — {len(df_exp)} strikes | Spot ₹{spot:,.2f}")
            else:
                ph.error("❌ CSV parse failed")
                with st.expander("CSV debug"): [st.text(l) for l in dbg]
        # 3. BSE paste fallback
        if df_exp.empty and bse_paste_ and bse_paste_.strip():
            df_p,dbg=parse_bse_paste(bse_paste_,expiry=bse_exp)
            if not df_p.empty:
                df_exp=df_p;data_src="BSE-paste";spot=get_spot(yfsym);expiries=[bse_exp]
                ph.success(f"✅ BSE paste — {len(df_exp)} strikes | Spot ₹{spot:,.2f}")
            else:
                ph.error("❌ BSE paste parse failed")
                with st.expander("Paste debug"): [st.text(l) for l in dbg]
        # 4. Spot-only
        if df_exp.empty:
            spot=get_spot(yfsym); has_opts=False
            if spot>0:
                ph.warning(f"⚠️ No chain for **{fetch_sym}** yet. Spot ₹{spot:,.2f} from yFinance.")
                if not running:
                    st.info("Press **▶ Start** to begin live scraping via Selenium.")
                else:
                    st.info(f"Scraper running. Waiting for first **{fetch_sym}** fetch…")
            else:
                ph.error(f"Cannot get data for {fetch_sym}.")
        # Expiry selector
        if not df_exp.empty and expiries:
            with st.sidebar:
                sel_expiry=st.selectbox("Expiry",expiries[:12])
            if data_src not in ("CSV","BSE-paste") and "expiryDate" in df_exp.columns:
                mask=df_exp["expiryDate"]==sel_expiry
                df_exp=df_exp[mask].copy() if mask.any() else df_exp.copy()

    elif not is_spot:
        spot=get_spot(fetch_sym)
        if spot>0 and _HAS_YF:
            try: exps=list(yf.Ticker(fetch_sym).options or [])
            except Exception: exps=[]
            if exps:
                with st.sidebar: sel_expiry=st.selectbox("Expiry",exps[:8])
                calls,puts=get_yf_chain(fetch_sym,sel_expiry)
                if calls is not None and not calls.empty:
                    df_exp=build_yf_df(calls,puts,sel_expiry);data_src="yFinance"
                    ph.success(f"✅ {fetch_sym} | {len(df_exp)} strikes | {spot:,.2f}")
                else: ph.error("Chain fetch failed.");has_opts=False
            else: ph.warning("No options.");has_opts=False
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
        all_strikes=[];atm_pos=0

    sig=signals(df_exp,spot)
    sb_cls={"NSE-Selenium":"src-a","BSE-Selenium":"src-a","BSE-API":"src-b","CSV":"src-c","BSE-paste":"src-c","yFinance":"src-b"}
    badge=f'<span class="src {sb_cls.get(data_src,"src-b")}">{data_src}</span>' if data_src else ""

    # ─── METRICS ───────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5,c6,c7=st.columns(7)
    c1.metric("📍 Spot",f"₹{spot:,.2f}")
    c2.metric("🎯 ATM",f"₹{sig['atm']:,.0f}" if has_chain else "—")
    c3.metric("📊 PCR",f"{sig['pcr']:.3f}" if has_chain else "—")
    c4.metric("💀 Max Pain",f"₹{sig['max_pain']:,.0f}" if has_chain else "—")
    c5.metric("🌡 IV",f"{sig['atm_iv']:.1f}%" if has_chain else "—")
    c6.metric("↕ Skew",f"{sig['skew']:+.1f}%" if has_chain else "—")
    c7.metric("♟ Straddle",f"₹{sig['straddle']:.2f}" if has_chain else "—")

    # ─── SIGNAL BANNER ─────────────────────────────────────────────────────
    conf=sig["conf"]; bc="#00ff88" if conf>=72 else "#ff9500" if conf>=58 else "#5a7a9a"
    def tag(s): return f'<span class="tag-b">{s}</span>' if "BUY" in s else f'<span class="tag-w">{s}</span>'

    if has_chain:
        st.markdown(f"""<div class="sb">
<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;align-items:flex-start">
<div style="flex:2">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
    <span style="color:#5a7a9a;font-size:10px;letter-spacing:2px">SIGNAL</span>{badge}
  </div>
  <div style="font-size:26px;font-weight:700;color:{bc};font-family:Space Mono">{sig['rec']}</div>
  <div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:6px">
    {tag("⚡ Scalp: "+sig['scalp'])}
    {tag("📅 Intraday: "+sig['intraday'])}
    {tag("🗓 Swing: "+sig['swing'])}
    {tag("📆 Positional: "+sig['pos'])}
  </div>
</div>
<div style="text-align:center;min-width:130px">
  <div style="color:#5a7a9a;font-size:10px;letter-spacing:2px">CONFIDENCE</div>
  <div style="font-size:48px;font-weight:700;color:{bc};font-family:Space Mono;line-height:1.1">{conf}%</div>
  <div style="background:{bc}22;border-radius:20px;height:6px;width:120px;margin:6px auto 0">
    <div style="background:{bc};border-radius:20px;height:6px;width:{conf}%"></div>
  </div>
  <div style="font-size:11px;color:#5a7a9a;margin-top:4px">{'STRONG' if conf>=72 else 'WATCH' if conf>=58 else 'NO EDGE'}</div>
</div>
</div></div>""",unsafe_allow_html=True)

        st.markdown("#### Why this signal? — Indicator breakdown")
        for name,val,verd,pts,exp in sig["indicators"]:
            clr=("#00ff88" if any(x in verd for x in ("BULLISH","CHEAP","✅","Fresh","BREAK"))
                 else "#ff3b5c" if any(x in verd for x in ("BEAR","EXPENSIVE","AGAINST","VERY","conflict"))
                 else "#ff9500")
            pc="#00ff88" if "+" in str(pts) and not str(pts).startswith("-") else "#ff3b5c" if "-" in str(pts) else "#5a7a9a"
            st.markdown(f"""<div class="ir">
<span style="color:#9ab5cc;font-weight:600;min-width:180px">{name}</span>
<span style="color:#c8d6e5;font-family:Space Mono;font-size:12px;min-width:130px">{val}</span>
<span style="color:{clr};font-weight:600;min-width:200px">{verd}</span>
<span style="color:{pc};font-family:Space Mono;font-size:12px;min-width:50px">{pts}</span>
<span style="color:#7a9ab5;font-size:12px;flex:1">{exp}</span>
</div>""",unsafe_allow_html=True)

        for ab in sig["abnormal"]:
            st.markdown(f'<div class="al">{ab}</div>',unsafe_allow_html=True)
        if sig["gamma_blast"]:
            st.markdown('<div class="gb">⚡ <b style="color:#ff3b5c">GAMMA BLAST</b> — Huge OI at ATM. Options explode on breakout. Enter fast on direction confirmation.</div>',unsafe_allow_html=True)
    else:
        st.info(f"Spot ₹{spot:,.2f}. Press ▶ Start to begin live scraping, or use CSV/paste fallback for full analysis.")

    st.markdown("<br>",unsafe_allow_html=True)
    tabs=st.tabs(["📊 Chain","🔢 Greeks","📈 OI","📐 Straddle","⚡ Trade","🔬 Backtest","📋 History","🧠 Analysis"])

    # ─── Chain ─────────────────────────────────────────────────────────────
    with tabs[0]:
        if not has_chain: st.warning("No chain data. Start scraper or use fallbacks.")
        else:
            def_st=all_strikes[max(0,atm_pos-8):atm_pos+9]
            sel_st=st.multiselect("Strikes",all_strikes,default=def_st)
            if not sel_st: sel_st=def_st
            sc=[c for c in ["strikePrice","CE_LTP","CE_OI","CE_changeOI","CE_IV","CE_volume","PE_LTP","PE_OI","PE_changeOI","PE_IV","PE_volume"] if c in df_exp.columns]
            st.dataframe(df_exp[df_exp["strikePrice"].isin(sel_st)][sc].round(2),use_container_width=True,height=260)
            st.plotly_chart(chart_chain(df_exp,spot),use_container_width=True)
            st.markdown("""<div class="pe">
<b>Top-left Premium:</b> Green=CE, Red=PE. ATM has highest time value. CE gets expensive below spot (ITM), PE above spot.<br>
<b>Top-right OI:</b> Tallest blue bar = strongest resistance (call sellers defend). Tallest orange = strongest support (put sellers defend).<br>
<b>Bottom-left IV Smile:</b> PE IV &gt; CE IV = institutional fear buying (bearish hedge). CE IV &gt; PE IV = aggressive bullish call buying.<br>
<b>Bottom-right Volume:</b> Today's activity. High volume at a strike = market actively betting on that level right now.
</div>""",unsafe_allow_html=True)

    # ─── Greeks ────────────────────────────────────────────────────────────
    with tabs[1]:
        if not has_chain: st.warning("Needs chain data.")
        else:
            st.plotly_chart(chart_greeks(df_exp,spot),use_container_width=True)
            st.markdown("""<div class="pe">
<b>Delta:</b> ₹ gain per ₹100 market move. CE delta 0.5 = ₹50 gain per ₹100 Nifty rise. ATM ≈ 0.5, deep ITM ≈ 1.0, deep OTM ≈ 0.1.<br>
<b>Gamma:</b> How fast Delta changes. High near ATM — positions can accelerate quickly on any move. Gamma risk is highest near expiry.<br>
<b>Theta:</b> Daily cost of holding. If theta = -₹50, you lose ₹50/day even if market doesn't move. Only hold when expected move > theta.<br>
<b>Vega:</b> Gain per 1% IV rise. Buy before events (IV spikes = you gain). Sell after events (IV crashes = you gain if you sold).
</div>""",unsafe_allow_html=True)
            ai=(df_exp["strikePrice"]-spot).abs().idxmin(); atm2=df_exp.loc[ai]
            g1,g2=st.columns(2)
            for col_,px_,lbl,clr in [(g1,"CE","📗 CALL (UP)","#00ff88"),(g2,"PE","📕 PUT (DOWN)","#ff3b5c")]:
                with col_:
                    st.markdown(f"<h5 style='color:{clr}'>{lbl} | ATM ₹{int(atm2['strikePrice']):,}</h5>",unsafe_allow_html=True)
                    cs=st.columns(3)
                    for i,(n,k,t) in enumerate([("Delta",f"{px_}_delta","₹/₹100"),("Gamma",f"{px_}_gamma","speed"),
                                                ("Theta",f"{px_}_theta","₹/day"),("Vega",f"{px_}_vega","₹/1%IV"),("IV%",f"{px_}_IV","now")]):
                        cs[i%3].metric(n,f"{float(atm2.get(k,0)):.2f}{'%' if 'IV' in n else ''}",t)
            ct=abs(float(atm2.get("CE_theta",0))); cv=float(atm2.get("CE_vega",0))
            if cv>ct: st.markdown(f'<div style="background:#002a15;border-left:3px solid #00ff88;padding:10px 14px;border-radius:0 6px 6px 0;margin:6px 0">✅ Vega ₹{cv:.2f} > Theta ₹{ct:.2f}/day — volatility gains outpace time decay. Good to buy.</div>',unsafe_allow_html=True)
            else:     st.markdown(f'<div style="background:#1a0a00;border-left:3px solid #ff9500;padding:10px 14px;border-radius:0 6px 6px 0;margin:6px 0">⚠️ Theta ₹{ct:.2f}/day > Vega ₹{cv:.2f} — time decay faster than volatility benefit. Only enter on very strong signal.</div>',unsafe_allow_html=True)

    # ─── OI ────────────────────────────────────────────────────────────────
    with tabs[2]:
        if not has_chain: st.warning("Needs chain data.")
        else:
            st.plotly_chart(chart_oi(df_exp,spot,sig),use_container_width=True)
            st.markdown("""<div class="pe">
<b>Panel 1 Total OI:</b> Blue=call sellers (resistance), Orange=put sellers (support). Tallest bar = strongest wall. Sellers defend these hard near expiry.<br>
<b>Panel 2 OI Change:</b> Green=new positions added (fresh bets). Red=positions closed. Fresh CE = new resistance wall being built. Fresh PE = new support floor.<br>
<b>Panel 3 % Change:</b> Where the most aggressive new positioning is happening TODAY. A 200% spike means OI tripled — someone making a massive directional bet here.
</div>""",unsafe_allow_html=True)
            narr=oi_narrative(df_exp,spot,sig)
            if narr: st.markdown(f'<div class="oin">{narr}</div>',unsafe_allow_html=True)
            b1,b2=st.columns(2)
            with b1:
                st.markdown("**🔴 Top CE OI — resistance walls**")
                ct_=df_exp.nlargest(5,"CE_OI")[["strikePrice","CE_OI","CE_changeOI","CE_LTP"]].copy()
                prev_=(ct_["CE_OI"]-ct_["CE_changeOI"]).clip(lower=1)
                ct_["Chg%"]=(ct_["CE_changeOI"]/prev_*100).round(1).astype(str)+"%"
                ct_["Signal"]=ct_["CE_changeOI"].apply(lambda x:"🔴 Fresh" if x>0 else "🟡 Fading" if x<0 else "—")
                st.dataframe(ct_,use_container_width=True)
            with b2:
                st.markdown("**🟢 Top PE OI — support walls**")
                pt_=df_exp.nlargest(5,"PE_OI")[["strikePrice","PE_OI","PE_changeOI","PE_LTP"]].copy()
                prev_=(pt_["PE_OI"]-pt_["PE_changeOI"]).clip(lower=1)
                pt_["Chg%"]=(pt_["PE_changeOI"]/prev_*100).round(1).astype(str)+"%"
                pt_["Signal"]=pt_["PE_changeOI"].apply(lambda x:"🟢 Fresh" if x>0 else "🟡 Fading" if x<0 else "—")
                st.dataframe(pt_,use_container_width=True)

    # ─── Straddle ───────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("""<div class="exp"><b>Straddle</b> = buy CE + PE at same strike. Profit if market moves more than the straddle cost by expiry.
Use this to judge: are options cheap or expensive right now vs history?</div>""",unsafe_allow_html=True)
        if not has_chain: st.warning("Needs chain data.")
        else:
            figs,texts,val_msg,val_clr=chart_straddle(df_exp,spot,yfsym,sig["straddle"])
            for fig,txt in zip(figs,texts):
                st.plotly_chart(fig,use_container_width=True)
                st.markdown(f'<div class="pe">{txt}</div>',unsafe_allow_html=True)
            if val_msg: st.markdown(f'<div style="background:{val_clr}11;border:1px solid {val_clr}44;border-radius:8px;padding:14px;color:{val_clr};font-weight:600;font-size:15px;margin:10px 0">{val_msg}</div>',unsafe_allow_html=True)

    # ─── Trade ──────────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("""<div class="exp">Paper trading — practice without real money. Rule: never enter without setting SL. Never average losers. Target R:R > 1.5.</div>""",unsafe_allow_html=True)
        if not has_chain: st.warning("Needs chain.")
        else:
            la,lb=st.columns([3,2])
            with la:
                st.markdown("#### New Trade")
                t1,t2=st.columns(2)
                sd=t1.selectbox("Direction",["CE — Call (UP)","PE — Put (DOWN)"])
                px_="CE" if "CE" in sd else "PE"
                ts_=t1.selectbox("Strike",all_strikes,index=atm_pos)
                lots=t2.number_input("Lots",1,100,1)
                lsl=t2.slider("SL %",20,50,sl_pct,key="lsl")
                ltgt=t2.slider("Target %",30,200,tgt_pct,key="ltgt")
                rr_=df_exp[df_exp["strikePrice"]==ts_]; lc_=f"{px_}_LTP"
                op=float(rr_[lc_].values[0]) if not rr_.empty and lc_ in rr_.columns else 0
                if op>0:
                    sla=round(op*(1-lsl/100),2); tgta=round(op*(1+ltgt/100),2)
                    risk=(op-sla)*lots; rew=(tgta-op)*lots; rr=round(rew/risk,2) if risk>0 else 0
                    m1,m2,m3,m4=st.columns(4)
                    m1.metric("Entry",f"₹{op:.2f}"); m2.metric("SL",f"₹{sla:.2f}",f"-{lsl}%")
                    m3.metric("Target",f"₹{tgta:.2f}",f"+{ltgt}%"); m4.metric("R:R",f"1:{rr}")
                    if rr<1.5: st.warning("⚠️ R:R < 1.5 — poor setup")
                else: sla=tgta=0; st.warning("Price = 0 — market closed or too far OTM")
                if st.button("📈 Enter Paper Trade",type="primary",use_container_width=True):
                    if op>0:
                        t={"id":len(st.session_state.trades)+1,"sym":fetch_sym,"expiry":sel_expiry,
                           "strike":ts_,"side":px_,"entry":op,"sl":sla,"target":tgta,"lots":lots,
                           "conf":sig["conf"],"rec":sig["rec"],"time":dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "status":"OPEN","exit":None,"pnl":None}
                        st.session_state.active.append(t); st.session_state.trades.append(t)
                        st.success(f"✅ {fetch_sym} {ts_} {px_} @ ₹{op:.2f}")
                    else: st.error("Price=0")
            with lb:
                st.markdown("#### Open Positions")
                if not [t for t in st.session_state.active if t["status"]=="OPEN"]: st.info("No open trades.")
                for i,t in enumerate(st.session_state.active):
                    if t["status"]!="OPEN": continue
                    rr_=df_exp[df_exp["strikePrice"]==t["strike"]]; lc_=f"{t['side']}_LTP"
                    curr=float(rr_[lc_].values[0]) if not rr_.empty and lc_ in rr_.columns else t["entry"]
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
                        st.rerun()

    # ─── Backtest ───────────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("""<div class="exp"><b>Backtest</b>: Simulates the signal on historical price data. Each day: apply same scoring rules → if score ≥ 72 → simulate buying ATM option → check next day result → compound.
Good results: Win Rate &gt; 60% · R:R &gt; 1.5 · Max Drawdown &lt; 20%</div>""",unsafe_allow_html=True)
        ba,bb,bc_=st.columns(3)
        bl=ba.slider("Lookback days",20,120,60)
        bk=bb.number_input("Start ₹",50000,5000000,int(capital),10000,key="bk")
        bp=bc_.slider("% per trade",2,20,pos_pct,key="bp")
        if st.button("🔬 Run Backtest",type="primary",use_container_width=True):
            with st.spinner("Running…"):
                tdf,stats=backtest(yfsym,bl,float(bk),float(bp))
                st.session_state.bt=(tdf,stats)
        if st.session_state.bt:
            tdf,stats=st.session_state.bt
            if tdf is None: st.error(f"Backtest: {stats}")
            else:
                k1,k2,k3,k4,k5,k6,k7=st.columns(7)
                k1.metric("Trades",stats["total"]); k2.metric("Win Rate",f"{stats['wr']}%")
                k3.metric("Total P&L",f"₹{stats['pnl']:+,.0f}"); k4.metric("Avg Win",f"₹{stats['aw']:,.0f}")
                k5.metric("Avg Loss",f"₹{stats['al']:,.0f}"); k6.metric("R:R",stats["rr"]); k7.metric("Max DD",f"{stats['mdd']}%")
                rc="#00ff88" if stats["ret%"]>=0 else "#ff3b5c"
                st.markdown(f'<div style="text-align:center;font-size:22px;color:{rc};font-family:Space Mono;padding:12px;background:#0a1929;border-radius:10px;margin:8px 0">Final ₹{stats["final"]:,.0f} · Return {stats["ret%"]:+.2f}%</div>',unsafe_allow_html=True)
                wr=stats["wr"]; rr=stats["rr"]; mdd=stats["mdd"]
                if wr>=60 and rr>=1.5 and mdd<20: vrd="✅ Strong strategy — use with confidence"; vc="#00ff88"
                elif wr>=50 and rr>=1.2:           vrd="🟡 Decent — works but needs discipline"; vc="#ff9500"
                else:                              vrd="🔴 Weak — improve signal before real money"; vc="#ff3b5c"
                st.markdown(f'<div style="background:{vc}11;border-left:3px solid {vc};padding:12px;border-radius:0 8px 8px 0;margin:6px 0">{vrd}</div>',unsafe_allow_html=True)
                fig_eq=go.Figure(go.Scatter(y=stats["curve"],mode="lines",line=dict(color="#00e5ff",width=2.5),fill="tozeroy",fillcolor="rgba(0,229,255,0.05)"))
                fig_eq.add_hline(y=bk,line_dash="dash",line_color="#ff9500",annotation_text=f"Start ₹{int(bk):,}")
                fig_eq.update_layout(template=DARK,height=300,title="Equity Curve",yaxis_title="₹")
                st.plotly_chart(fig_eq,use_container_width=True)
                st.markdown('<div class="pe">Equity curve shows capital growth following this signal over the backtest period. Above orange line = profitable. Steep drops = max drawdown periods. Flat sections = no trade signal (cash preserved).</div>',unsafe_allow_html=True)
                w=tdf[tdf["P&L₹"]>0]["P&L₹"]; l=tdf[tdf["P&L₹"]<=0]["P&L₹"]
                fig_d=go.Figure()
                fig_d.add_trace(go.Histogram(x=w,name="Wins",marker_color="rgba(0,255,136,0.47)",nbinsx=20))
                fig_d.add_trace(go.Histogram(x=l,name="Losses",marker_color="rgba(255,59,92,0.47)",nbinsx=20))
                fig_d.update_layout(template=DARK,height=220,title="Win vs Loss Distribution",barmode="overlay")
                st.plotly_chart(fig_d,use_container_width=True)
                st.dataframe(tdf,use_container_width=True,height=280)

    # ─── History ────────────────────────────────────────────────────────────
    with tabs[6]:
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
            if st.button("🗑️ Clear History",use_container_width=True):
                st.session_state.trades=[]; st.session_state.active=[]; st.rerun()

    # ─── Analysis ───────────────────────────────────────────────────────────
    with tabs[7]:
        st.markdown(f"""<div style="background:var(--s1);border:1px solid var(--brd);border-radius:12px;padding:18px;margin:8px 0">
<h4 style='margin-top:0;color:#00e5ff'>📖 Full Market Reading — {fetch_sym} @ ₹{spot:,.2f} {badge}</h4></div>""",unsafe_allow_html=True)
        if has_chain:
            for name,val,verd,pts,exp in sig["indicators"]:
                clr=("#00ff88" if any(x in verd for x in ("BULLISH","CHEAP","✅","Fresh"))
                     else "#ff3b5c" if any(x in verd for x in ("BEAR","EXPENSIVE","AGAINST"))
                     else "#ff9500")
                st.markdown(f'<div class="exp"><b style="color:{clr}">{name}: {val} → {verd} ({pts})</b><br>{exp}</div>',unsafe_allow_html=True)
            st.markdown("#### 🗺 Three Scenarios")
            s1,s2,s3=st.columns(3)
            res=sig["resistance"]; sup=sig["support"]; atm_s=sig["buy_strike_atm"]
            for col,title,trigger,action,why,clr in [
                (s1,"🟢 BULLISH",f"Break + hold above ₹{res:,.0f}",f"Buy {atm_s} CE\nTarget +60% | SL -30%","CE writers panic → CE premium explodes","#00ff88"),
                (s2,"⚪ SIDEWAYS",f"Stay ₹{sup:,.0f}–₹{res:,.0f}","Do NOT buy options\nTheta kills daily\nCash is position","Every day in range = money lost to time decay. Wait.","#5a7a9a"),
                (s3,"🔴 BEARISH",f"Break + hold below ₹{sup:,.0f}",f"Buy {atm_s} PE\nTarget +60% | SL -30%","PE writers panic → PE premium explodes","#ff3b5c"),
            ]:
                with col:
                    st.markdown(f"""<div style='background:#0a1929;border-top:4px solid {clr};border:1px solid #1e3050;border-radius:10px;padding:14px'>
<h5 style='color:{clr};margin-top:0'>{title}</h5>
<b style='color:#5a7a9a;font-size:12px'>TRIGGER:</b><br>{trigger}<br><br>
<pre style='color:#c8d6e5;font-size:12px;background:transparent;margin:4px 0'>{action}</pre>
<span style='font-size:11px;color:#8899aa'>{why}</span></div>""",unsafe_allow_html=True)
        st.markdown("""<div style="background:var(--s1);border:1px solid var(--brd);border-radius:12px;padding:18px;margin-top:14px">
<h4 style='color:#ff9500;margin-top:0'>📜 8 Golden Rules</h4>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:13px;line-height:1.9'>
<div>1. Never risk more than 5% of capital per trade.<br>2. Only buy options when IV &lt; 30%.<br>3. Set stop-loss before entering — always.<br>4. Exit at least 5 days before expiry.</div>
<div>5. Book 50% profit at first target, trail the rest.<br>6. Never average a losing trade — ever.<br>7. Avoid buying on event days (RBI, earnings, budget).<br>8. No trade is also a trade — wait for edge.</div>
</div></div>""",unsafe_allow_html=True)
        st.caption(f"Updated {dt.datetime.now().strftime('%d-%b-%Y %H:%M:%S')} · Educational only · Not financial advice")

    if auto:
        time.sleep(60); st.cache_data.clear(); st.rerun()

if __name__=="__main__":
    main()

