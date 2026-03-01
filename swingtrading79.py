"""
Option Chain Live Analyzer \u2014 Nifty 50 & Sensex
Streamlit App | Live Scraping | Greeks | AI Recommendation | Start/Stop Control
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import time
from datetime import datetime, date
import warnings
import random

warnings.filterwarnings("ignore")

# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# PAGE CONFIG
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
st.set_page_config(
    page_title="Option Chain Analyzer",
    page_icon="\ud83d\udcca",
    layout="wide",
    initial_sidebar_state="expanded",
)

# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# GLOBAL CSS
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg:      #070a0f;
    --bg2:     #0d1117;
    --bg3:     #131920;
    --bg4:     #1a2233;
    --accent:  #00d4ff;
    --green:   #00e676;
    --red:     #ff3d5a;
    --yellow:  #ffd600;
    --text:    #cdd9e5;
    --muted:   #4a5568;
    --border:  #1e2d3d;
}

html, body, [class*="css"], .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; letter-spacing:-0.5px; }

.main .block-container { padding: 0.8rem 1.5rem 2rem; max-width: 100%; }

/* \u2500\u2500 Control Banner \u2500\u2500 */
.ctrl-banner {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.status-dot-live  { width:10px;height:10px;border-radius:50%;background:#00e676;
                    box-shadow:0 0 8px #00e676;animation:pulse 1.5s infinite; }
.status-dot-stop  { width:10px;height:10px;border-radius:50%;background:#ff3d5a; }
.status-dot-idle  { width:10px;height:10px;border-radius:50%;background:#4a5568; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(1.3)} }

/* \u2500\u2500 Metric Cards \u2500\u2500 */
.kpi-grid { display:grid; grid-template-columns:repeat(6,1fr); gap:0.6rem; margin-bottom:1rem; }
.kpi {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    position: relative;
    overflow: hidden;
}
.kpi::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent);
    border-radius: 3px 0 0 3px;
}
.kpi.bull::before { background: var(--green); }
.kpi.bear::before { background: var(--red); }
.kpi.warn::before { background: var(--yellow); }
.kpi-label { font-size:0.62rem; color:var(--muted); text-transform:uppercase; letter-spacing:2px; }
.kpi-value { font-size:1.25rem; font-weight:700; color:var(--accent); margin:0.15rem 0; }
.kpi-sub   { font-size:0.68rem; color:var(--muted); }

/* \u2500\u2500 Recommendation Box \u2500\u2500 */
.rec-wrap {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    margin: 0.5rem 0 1rem;
}
.rec-header {
    padding: 0.7rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.rec-header.ce-bg { background: rgba(0,230,118,0.08); border-bottom: 1px solid rgba(0,230,118,0.15); }
.rec-header.pe-bg { background: rgba(255,61,90,0.08);  border-bottom: 1px solid rgba(255,61,90,0.15); }
.rec-header.hl-bg { background: rgba(255,214,0,0.06);  border-bottom: 1px solid rgba(255,214,0,0.15); }
.rec-action {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: 1px;
}
.rec-body { padding: 1rem 1.2rem; background: var(--bg2); }
.trade-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.5rem;
    margin: 0.8rem 0 0.5rem;
}
.trade-box {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.5rem 0.7rem;
    text-align: center;
}
.trade-box-label { font-size:0.6rem; color:var(--muted); text-transform:uppercase; letter-spacing:2px; }
.trade-box-val   { font-size:1rem; font-weight:700; color:var(--accent); }

/* \u2500\u2500 Signal Rows \u2500\u2500 */
.sig-row {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    padding: 0.45rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.78rem;
    line-height: 1.5;
}
.sig-row:last-child { border-bottom: none; }
.sig-icon { font-size:1rem; flex-shrink:0; margin-top:1px; }
.sig-label { color:var(--muted); min-width:90px; flex-shrink:0; }
.sig-text  { color:var(--text); flex:1; }
.pill { display:inline-block; border-radius:3px; padding:1px 7px;
        font-size:0.62rem; font-weight:700; letter-spacing:1px; margin-left:6px; }
.pill-bull { background:rgba(0,230,118,0.15); color:#00e676; border:1px solid rgba(0,230,118,0.3); }
.pill-bear { background:rgba(255,61,90,0.15);  color:#ff3d5a; border:1px solid rgba(255,61,90,0.3); }
.pill-neut { background:rgba(0,212,255,0.1);   color:#00d4ff; border:1px solid rgba(0,212,255,0.2); }

/* \u2500\u2500 Greeks Panel \u2500\u2500 */
.greek-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; }
.greek-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem;
}
.greek-title { font-size:0.65rem; text-transform:uppercase; letter-spacing:2px;
               color:var(--muted); margin-bottom:0.6rem; }
.greek-row { display:flex; justify-content:space-between; padding:3px 0;
             border-bottom:1px solid var(--border); font-size:0.78rem; }
.greek-row:last-child { border-bottom:none; }
.greek-name { color:var(--muted); }
.greek-val-ce { color:#00e676; font-weight:600; }
.greek-val-pe { color:#ff3d5a; font-weight:600; }

/* \u2500\u2500 Log Panel \u2500\u2500 */
.log-box {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.72rem;
    line-height: 1.8;
    max-height: 200px;
    overflow-y: auto;
    color: var(--muted);
}

/* \u2500\u2500 Sidebar \u2500\u2500 */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container { padding:1rem; }

/* \u2500\u2500 Streamlit overrides \u2500\u2500 */
.stButton > button {
    width: 100%;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border: none !important;
    transition: all 0.2s !important;
}
.stDataFrame { background: var(--bg3) !important; }
div[data-testid="stDataFrameResizable"] { background: var(--bg3) !important; }
.stSelectbox > div > div, .stTextInput > div > div {
    background: var(--bg3) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* \u2500\u2500 Top header \u2500\u2500 */
.top-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
}
.app-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d4ff 0%, #00e676 50%, #ffd600 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
}
.app-sub {
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ATM row highlight in dataframe */
.atm-highlight { background: rgba(0,212,255,0.06) !important; }

/* fetch error box */
.err-box {
    background: rgba(255,61,90,0.07);
    border: 1px solid rgba(255,61,90,0.25);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.82rem;
    line-height: 1.8;
    margin: 0.5rem 0;
}

/* info box */
.info-box {
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.82rem;
    line-height: 1.8;
    margin: 0.5rem 0;
}

hr { border-color: var(--border) !important; margin: 0.8rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# SESSION STATE INIT
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
def init_state():
    defaults = {
        "trading_active": False,
        "last_fetch_time": 0.0,
        "fetch_count": 0,
        "df": None,
        "spot": 0.0,
        "expiries": [],
        "chosen_expiry": "",
        "rec": None,
        "log": [],
        "error": "",
        "nse_session": None,
        "nse_session_ready": False,
        "fetch_status": "idle",   # idle | fetching | ok | error
        "symbol": "NIFTY",
        "exchange": "NSE",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def log(msg: str, level: str = "info"):
    ts  = datetime.now().strftime("%H:%M:%S")
    icon = {"info": "\u2139", "ok": "\u2705", "warn": "\u26a0\ufe0f", "err": "\u274c"}.get(level, "\u2022")
    entry = f"[{ts}] {icon} {msg}"
    st.session_state["log"].insert(0, entry)
    st.session_state["log"] = st.session_state["log"][:80]   # keep last 80 lines


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# BLACK-SCHOLES + GREEKS
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
def bs_price(S, K, T, r, sigma, opt='CE'):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if opt == 'CE' else max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == 'CE':
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def calc_iv(market_px, S, K, T, r, opt='CE'):
    if T <= 0 or market_px <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        intrinsic = max(0.0, S - K if opt == 'CE' else K - S)
        if market_px <= intrinsic + 0.001:
            return 0.0
        lo, hi = 0.001, 15.0
        if bs_price(S, K, T, r, lo, opt) > market_px:
            return lo
        if bs_price(S, K, T, r, hi, opt) < market_px:
            return hi
        return float(brentq(lambda sig: bs_price(S, K, T, r, sig, opt) - market_px,
                             lo, hi, maxiter=120, xtol=1e-5))
    except Exception:
        return 0.0


def calc_greeks(S, K, T, r, sigma, opt='CE'):
    empty = dict(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return empty
    try:
        sqT = np.sqrt(T)
        d1  = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqT)
        d2  = d1 - sigma * sqT
        gma = float(norm.pdf(d1) / (S * sigma * sqT))
        vga = float(S * sqT * norm.pdf(d1) / 100)
        if opt == 'CE':
            dlt = float(norm.cdf(d1))
            tht = float((-(S * sigma * norm.pdf(d1)) / (2 * sqT)
                         - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365)
            rho = float(K * T * np.exp(-r * T) * norm.cdf(d2) / 100)
        else:
            dlt = float(norm.cdf(d1) - 1)
            tht = float((-(S * sigma * norm.pdf(d1)) / (2 * sqT)
                         + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365)
            rho = float(-K * T * np.exp(-r * T) * norm.cdf(-d2) / 100)
        return dict(delta=round(dlt,4), gamma=round(gma,6),
                    theta=round(tht,4), vega=round(vga,4), rho=round(rho,4))
    except Exception:
        return empty


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# NSE FETCH  (cookie-warm approach)
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
_UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]


def _nse_headers(ua=None):
    ua = ua or random.choice(_UA_LIST)
    return {
        "User-Agent": ua,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/option-chain",
        "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "Connection": "keep-alive",
        "DNT": "1",
    }


def warm_nse_session(force=False):
    """
    Build a session with valid NSE cookies.
    Returns (session, success_bool).
    Cached in session_state to avoid re-warming on every rerun.
    """
    if (not force
            and st.session_state["nse_session"] is not None
            and st.session_state["nse_session_ready"]):
        return st.session_state["nse_session"], True

    ua = random.choice(_UA_LIST)
    sess = requests.Session()
    sess.headers.update(_nse_headers(ua))

    steps = [
        ("https://www.nseindia.com", "NSE homepage"),
        ("https://www.nseindia.com/option-chain", "NSE option-chain page"),
    ]
    for url, label in steps:
        try:
            r = sess.get(url, timeout=12)
            log(f"Warmed {label} \u2192 HTTP {r.status_code}", "info")
            time.sleep(1.5)
        except Exception as e:
            log(f"Warm-up failed at {label}: {e}", "warn")
            time.sleep(2)

    st.session_state["nse_session"] = sess
    st.session_state["nse_session_ready"] = True
    return sess, True


def fetch_nse_oc(symbol: str, max_retries: int = 4):
    """
    Fetch NSE option chain JSON.
    Returns (raw_dict | None, error_str).
    """
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    sess, _ = warm_nse_session()

    for attempt in range(max_retries):
        try:
            resp = sess.get(url, timeout=15)
            code = resp.status_code

            if code == 200:
                data = resp.json()
                if data.get("records"):
                    log(f"NSE {symbol} fetched OK (attempt {attempt+1})", "ok")
                    return data, ""
                else:
                    return None, "Empty records in NSE response"

            elif code == 429:
                wait = 8 * (attempt + 1)
                log(f"NSE rate-limited (429). Waiting {wait}s\u2026", "warn")
                time.sleep(wait)

            elif code in (401, 403):
                log(f"NSE session expired ({code}). Re-warming\u2026", "warn")
                # Force new session
                st.session_state["nse_session_ready"] = False
                sess, _ = warm_nse_session(force=True)
                time.sleep(3)

            elif code == 503:
                log("NSE service unavailable (503). Market may be closed.", "warn")
                time.sleep(5)

            else:
                log(f"NSE HTTP {code} on attempt {attempt+1}", "warn")
                time.sleep(3)

        except requests.exceptions.SSLError as e:
            log(f"NSE SSL error: {e}", "err")
            time.sleep(4)
        except requests.exceptions.ConnectionError as e:
            log(f"NSE connection error: {e}", "err")
            time.sleep(5)
        except requests.exceptions.Timeout:
            log("NSE timeout \u2014 retrying\u2026", "warn")
            time.sleep(4)
        except ValueError:
            log("NSE returned non-JSON response", "err")
            time.sleep(3)
        except Exception as e:
            log(f"NSE unexpected: {e}", "err")
            time.sleep(3)

    return None, f"NSE fetch failed after {max_retries} attempts."


def parse_nse(raw: dict, expiry: str):
    """
    Parse NSE raw JSON for a specific expiry.
    Returns (DataFrame, spot_price).
    """
    try:
        records = raw.get("records", {}).get("data", [])
        spot    = float(raw.get("records", {}).get("underlyingValue", 0) or 0)
        rows    = []
        for rec in records:
            if expiry and rec.get("expiryDate", "") != expiry:
                continue
            k  = float(rec.get("strikePrice", 0) or 0)
            ce = rec.get("CE", {}) or {}
            pe = rec.get("PE", {}) or {}
            rows.append({
                "Strike":   k,
                "C_OI":     float(ce.get("openInterest", 0) or 0),
                "C_ChgOI":  float(ce.get("changeinOpenInterest", 0) or 0),
                "C_Vol":    float(ce.get("totalTradedVolume", 0) or 0),
                "C_IV":     float(ce.get("impliedVolatility", 0) or 0),
                "C_LTP":    float(ce.get("lastPrice", 0) or 0),
                "C_Bid":    float(ce.get("bidprice", 0) or 0),
                "C_Ask":    float(ce.get("askPrice", 0) or 0),
                "C_Chng":   float(ce.get("change", 0) or 0),
                "P_OI":     float(pe.get("openInterest", 0) or 0),
                "P_ChgOI":  float(pe.get("changeinOpenInterest", 0) or 0),
                "P_Vol":    float(pe.get("totalTradedVolume", 0) or 0),
                "P_IV":     float(pe.get("impliedVolatility", 0) or 0),
                "P_LTP":    float(pe.get("lastPrice", 0) or 0),
                "P_Bid":    float(pe.get("bidprice", 0) or 0),
                "P_Ask":    float(pe.get("askPrice", 0) or 0),
                "P_Chng":   float(pe.get("change", 0) or 0),
            })
        df = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)
        return df, spot
    except Exception as e:
        log(f"NSE parse error: {e}", "err")
        return pd.DataFrame(), 0.0


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# BSE FETCH
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
def fetch_bse_oc(expiry: str = "", max_retries: int = 3):
    """
    Try multiple known BSE Sensex option chain endpoints.
    Returns (raw_dict | None, error_str).
    """
    ua = random.choice(_UA_LIST)
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": ua,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.bseindia.com/markets/Derivatives/DeriReports/DeriOptionchain.html",
        "Origin": "https://www.bseindia.com",
        "Connection": "keep-alive",
    })

    # Warm BSE session
    try:
        sess.get("https://www.bseindia.com", timeout=10)
        time.sleep(1)
        sess.get(
            "https://www.bseindia.com/markets/Derivatives/DeriReports/DeriOptionchain.html",
            timeout=10
        )
        time.sleep(1.5)
    except Exception as e:
        log(f"BSE warm-up warning: {e}", "warn")

    # Candidate endpoints (BSE changes these periodically)
    endpoints = [
        f"https://api.bseindia.com/BseIndiaAPI/api/OptionChain/w?scripcd=&Expiry={expiry}&StrikePrice=&optiontype=&seriesid=12",
        f"https://api.bseindia.com/BseIndiaAPI/api/OptionChain/w?scripcd=&Expiry={expiry}&optiontype=&seriesid=12",
        "https://api.bseindia.com/BseIndiaAPI/api/OptionChainData/w?expirydate=",
        "https://api.bseindia.com/BseIndiaAPI/api/OptionChainSensex/w",
    ]

    for url in endpoints:
        for attempt in range(max_retries):
            try:
                r = sess.get(url, timeout=15)
                if r.status_code == 200:
                    try:
                        data = r.json()
                        if data:
                            log(f"BSE fetched OK via {url.split('?')[0]}", "ok")
                            return data, ""
                    except ValueError:
                        pass
                elif r.status_code == 429:
                    time.sleep(8 * (attempt + 1))
                elif r.status_code in (403, 401):
                    time.sleep(5)
                    break   # try next endpoint
                time.sleep(2)
            except Exception as e:
                log(f"BSE attempt {attempt+1} error: {e}", "warn")
                time.sleep(3)

    return None, "BSE fetch failed on all endpoints. BSE API may require browser auth."


def parse_bse(raw):
    """Parse BSE JSON into wide-format DataFrame."""
    try:
        if isinstance(raw, dict):
            records = raw.get("Table", raw.get("Table1", raw.get("data", [])))
        elif isinstance(raw, list):
            records = raw
        else:
            return pd.DataFrame(), 0.0

        if not records:
            return pd.DataFrame(), 0.0

        rows = []
        for rec in records:
            strike = float(rec.get("StrikePrice", rec.get("STRIKE_PRICE", 0)) or 0)
            opt_type = str(rec.get("CPType", rec.get("OPTION_TYPE", rec.get("CP_TYPE", ""))) or "").upper().strip()
            ltp = float(rec.get("LTP", rec.get("LAST_PRICE", 0)) or 0)
            oi  = float(rec.get("OI",  rec.get("OPEN_INT", 0)) or 0)
            chg_oi = float(rec.get("ChgOI", rec.get("CHG_IN_OI", 0)) or 0)
            vol = float(rec.get("TotalVolume", rec.get("VOLUME", rec.get("TOTAL_VOLUME", 0))) or 0)
            iv  = float(rec.get("IV", rec.get("IMP_VOL", 0)) or 0)
            bid = float(rec.get("BidPrice", rec.get("BID_PRICE", 0)) or 0)
            ask = float(rec.get("AskPrice", rec.get("ASK_PRICE", 0)) or 0)
            chng = float(rec.get("Chng", rec.get("NET_CHANGE", 0)) or 0)
            rows.append({"Strike": strike, "Type": opt_type, "OI": oi,
                          "ChgOI": chg_oi, "Vol": vol, "IV": iv,
                          "LTP": ltp, "Bid": bid, "Ask": ask, "Chng": chng})

        df_raw = pd.DataFrame(rows)
        if df_raw.empty:
            return pd.DataFrame(), 0.0

        ce = df_raw[df_raw["Type"].isin(["CE", "C", "CALL"])].copy()
        pe = df_raw[df_raw["Type"].isin(["PE", "P", "PUT"])].copy()
        ce = ce.set_index("Strike").add_prefix("C_").drop(columns="C_Type", errors="ignore")
        pe = pe.set_index("Strike").add_prefix("P_").drop(columns="P_Type", errors="ignore")
        df = ce.join(pe, how="outer").fillna(0).reset_index().sort_values("Strike")
        return df, 0.0
    except Exception as e:
        log(f"BSE parse error: {e}", "err")
        return pd.DataFrame(), 0.0


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# GREEKS ENRICHMENT
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
def enrich_greeks(df: pd.DataFrame, spot: float, T: float, r: float):
    if df.empty or spot <= 0 or T <= 0:
        return df
    for opt_type, pfx in [("CE", "C_"), ("PE", "P_")]:
        ltp_col = f"{pfx}LTP"
        iv_col  = f"{pfx}IV"
        if ltp_col not in df.columns:
            continue
        cols = {k: [] for k in ["Delta","Gamma","Theta","Vega","Rho","IV_Calc"]}
        for _, row in df.iterrows():
            K   = float(row.get("Strike", 0))
            ltp = float(row.get(ltp_col, 0))
            iv_given = float(row.get(iv_col, 0))
            iv = iv_given / 100 if iv_given > 0 else calc_iv(ltp, spot, K, T, r, opt_type)
            g  = calc_greeks(spot, K, T, r, iv, opt_type)
            for gn in ["delta","gamma","theta","vega","rho"]:
                cols[gn.capitalize()].append(g[gn])
            cols["IV_Calc"].append(round(iv * 100, 2))
        for gn in ["Delta","Gamma","Theta","Vega","Rho","IV_Calc"]:
            df[f"{pfx}{gn}"] = cols[gn]
    return df


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# OPTION CHAIN METRICS
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
def compute_pcr(df):
    c = df["C_OI"].sum() if "C_OI" in df.columns else 0
    p = df["P_OI"].sum() if "P_OI" in df.columns else 0
    return round(p / c, 3) if c > 0 else 0.0


def compute_max_pain(df):
    strikes = df["Strike"].tolist()
    if not strikes:
        return 0.0
    pains = []
    for s in strikes:
        pain = 0.0
        for _, row in df.iterrows():
            k = row["Strike"]
            pain += max(0, s - k) * float(row.get("C_OI", 0))
            pain += max(0, k - s) * float(row.get("P_OI", 0))
        pains.append(pain)
    return strikes[int(np.argmin(pains))]


def find_atm(df, spot):
    if df.empty or spot <= 0:
        return 0.0
    return float(df.loc[(df["Strike"] - spot).abs().idxmin(), "Strike"])


def oi_walls(df):
    if df.empty:
        return 0.0, 0.0, 0, 0
    ci = df["C_OI"].idxmax() if "C_OI" in df.columns else 0
    pi = df["P_OI"].idxmax() if "P_OI" in df.columns else 0
    return (float(df.loc[ci, "Strike"]),
            float(df.loc[pi, "Strike"]),
            float(df.loc[ci, "C_OI"]),
            float(df.loc[pi, "P_OI"]))


def vw_iv(df):
    c_iv = p_iv = 0.0
    cv = df["C_Vol"].sum() if "C_Vol" in df.columns else 0
    pv = df["P_Vol"].sum() if "P_Vol" in df.columns else 0
    if cv > 0 and "C_IV" in df.columns:
        c_iv = round(float((df["C_IV"] * df["C_Vol"]).sum() / cv), 2)
    if pv > 0 and "P_IV" in df.columns:
        p_iv = round(float((df["P_IV"] * df["P_Vol"]).sum() / pv), 2)
    return c_iv, p_iv


def iv_skew(df, atm):
    idx = df[df["Strike"] == atm].index
    if len(idx) == 0:
        return 0.0, 0.0
    i = idx[0]
    n = 4
    otm_puts  = df.iloc[max(0, i - n): i]
    otm_calls = df.iloc[i + 1: min(len(df), i + 1 + n)]
    put_iv_m  = otm_puts["P_IV"].replace(0, np.nan).mean()  if "P_IV" in df.columns else 0
    call_iv_m = otm_calls["C_IV"].replace(0, np.nan).mean() if "C_IV" in df.columns else 0
    return round(float(put_iv_m or 0), 2), round(float(call_iv_m or 0), 2)


def atm_ltp(df, atm):
    row = df[df["Strike"] == atm]
    if row.empty:
        return 0.0, 0.0
    r = row.iloc[0]
    return float(r.get("C_LTP", 0)), float(r.get("P_LTP", 0))


def atm_row_greeks(df, atm):
    row = df[df["Strike"] == atm]
    if row.empty:
        return {}, {}
    r = row.iloc[0]
    ce = {g: round(float(r.get(f"C_{g}", 0)), 4) for g in ["Delta","Gamma","Theta","Vega","Rho"]}
    pe = {g: round(float(r.get(f"P_{g}", 0)), 4) for g in ["Delta","Gamma","Theta","Vega","Rho"]}
    return ce, pe


def days_to_exp(expiry_str: str):
    if not expiry_str:
        return 7
    try:
        exp = datetime.strptime(expiry_str, "%d-%b-%Y").date()
        return max((exp - date.today()).days, 0)
    except Exception:
        return 7


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# RECOMMENDATION ENGINE
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
def make_recommendation(df, spot, expiry_str, symbol):
    days_left = days_to_exp(expiry_str)
    T = days_left / 365.0

    pcr_val              = compute_pcr(df)
    pain                 = compute_max_pain(df)
    atm                  = find_atm(df, spot)
    call_wall, put_wall, cw_oi, pw_oi = oi_walls(df)
    vw_civ, vw_piv       = vw_iv(df)
    put_iv_skew, call_iv_skew = iv_skew(df, atm)
    ce_ltp, pe_ltp       = atm_ltp(df, atm)
    ce_g, pe_g           = atm_row_greeks(df, atm)
    c_chgoi = float(df["C_ChgOI"].sum()) if "C_ChgOI" in df.columns else 0
    p_chgoi = float(df["P_ChgOI"].sum()) if "P_ChgOI" in df.columns else 0
    overall_iv = max(vw_civ, vw_piv)

    # \u2500\u2500 Scoring \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    score = 0
    signals = {}

    # 1. PCR
    if pcr_val >= 1.4:
        score += 2
        signals["PCR"] = (f"{pcr_val:.3f} \u2014 Heavily bearish put writing \u2192 bullish", "bull")
    elif pcr_val >= 1.1:
        score += 1
        signals["PCR"] = (f"{pcr_val:.3f} \u2014 Slightly elevated \u2192 mildly bullish", "bull")
    elif pcr_val <= 0.6:
        score -= 2
        signals["PCR"] = (f"{pcr_val:.3f} \u2014 Heavy call writing / put unwinding \u2192 bearish", "bear")
    elif pcr_val <= 0.9:
        score -= 1
        signals["PCR"] = (f"{pcr_val:.3f} \u2014 Slightly low \u2192 mildly bearish", "bear")
    else:
        signals["PCR"] = (f"{pcr_val:.3f} \u2014 Balanced \u2192 neutral", "neut")

    # 2. Max Pain
    mp_diff = pain - spot
    mp_pct  = mp_diff / spot * 100 if spot > 0 else 0
    if mp_pct > 0.5:
        score += 1
        signals["Max Pain"] = (f"\u20b9{pain:.0f} is {mp_pct:.1f}% above spot \u2192 price may pull up", "bull")
    elif mp_pct < -0.5:
        score -= 1
        signals["Max Pain"] = (f"\u20b9{pain:.0f} is {abs(mp_pct):.1f}% below spot \u2192 price may pull down", "bear")
    else:
        signals["Max Pain"] = (f"\u20b9{pain:.0f} \u2248 spot \u2014 near equilibrium", "neut")

    # 3. OI walls
    if call_wall > 0 and put_wall > 0:
        call_dist_pct = (call_wall - spot) / spot * 100 if spot > 0 else 999
        put_dist_pct  = (spot - put_wall) / spot * 100  if spot > 0 else 999
        if call_dist_pct < 0.4:
            score -= 1
            signals["OI Walls"] = (
                f"Call wall \u20b9{call_wall:.0f} (OI: {cw_oi/1e5:.1f}L) very close above \u2014 strong resistance", "bear")
        elif put_dist_pct < 0.4:
            score += 1
            signals["OI Walls"] = (
                f"Put wall \u20b9{put_wall:.0f} (OI: {pw_oi/1e5:.1f}L) very close below \u2014 strong support", "bull")
        else:
            signals["OI Walls"] = (
                f"Call wall \u20b9{call_wall:.0f} (R) | Put wall \u20b9{put_wall:.0f} (S) | "
                f"Range {call_wall-put_wall:.0f} pts", "neut")

    # 4. OI change
    if c_chgoi > 0 and p_chgoi < 0:
        score -= 1
        signals["OI Change"] = ("Fresh CE OI + PE OI unwinding \u2192 bearish writing", "bear")
    elif p_chgoi > 0 and c_chgoi < 0:
        score += 1
        signals["OI Change"] = ("Fresh PE OI + CE OI unwinding \u2192 bullish writing", "bull")
    elif c_chgoi > 0 and p_chgoi > 0:
        signals["OI Change"] = ("Both CE & PE OI rising \u2192 vol expansion expected", "neut")
    else:
        signals["OI Change"] = ("OI change ambiguous", "neut")

    # 5. IV Skew
    iv_diff = put_iv_skew - call_iv_skew
    if iv_diff > 4:
        score -= 1
        signals["IV Skew"] = (
            f"OTM Put IV {put_iv_skew:.1f}% >> Call IV {call_iv_skew:.1f}% \u2192 bearish skew (fear)", "bear")
    elif iv_diff < -4:
        score += 1
        signals["IV Skew"] = (
            f"OTM Call IV {call_iv_skew:.1f}% >> Put IV {put_iv_skew:.1f}% \u2192 bullish skew (unusual)", "bull")
    else:
        signals["IV Skew"] = (
            f"Skew balanced \u2014 Put IV {put_iv_skew:.1f}% vs Call IV {call_iv_skew:.1f}%", "neut")

    # 6. IV regime
    if overall_iv > 28:
        signals["IV Regime"] = (
            f"High IV ({overall_iv:.1f}%) \u2192 buy options only for breakout; selling is safer", "warn")
    elif overall_iv < 11:
        signals["IV Regime"] = (
            f"Low IV ({overall_iv:.1f}%) \u2192 options cheap; ideal for buying directional", "bull")
    else:
        signals["IV Regime"] = (
            f"Normal IV ({overall_iv:.1f}%) \u2192 directional buying viable", "neut")

    # 7. Time decay
    if days_left <= 2:
        signals["Time Decay"] = (
            f"Only {days_left}d left! Theta is devastating \u2014 avoid buying unless immediate breakout", "bear")
        score = max(score - 1, score)
    elif days_left <= 7:
        signals["Time Decay"] = (
            f"{days_left}d to expiry \u2014 Theta decaying fast; prefer scalps", "warn")
    else:
        signals["Time Decay"] = (
            f"{days_left}d to expiry \u2014 Adequate time for positional trades", "bull")

    # \u2500\u2500 Build Trade Params \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    sl_pct  = 0.40   # SL at 40% of entry
    tgt1_x  = 1.50   # Target 1 at 1.5\u00d7
    tgt2_x  = 2.30   # Target 2 at 2.3\u00d7

    conf = min(round(abs(score) / 7 * 100, 0), 90.0)

    if score >= 2:
        action = "BUY CE"
        color  = "var(--green)"
        css_hdr = "ce-bg"
        strike_rec = atm
        entry = round(ce_ltp, 2) if ce_ltp > 0 else round(spot * 0.006, 2)
    elif score <= -2:
        action = "BUY PE"
        color  = "var(--red)"
        css_hdr = "pe-bg"
        strike_rec = atm
        entry = round(pe_ltp, 2) if pe_ltp > 0 else round(spot * 0.006, 2)
    elif score == 1:
        action = "MILD BUY CE"
        color  = "var(--green)"
        css_hdr = "ce-bg"
        strike_rec = atm
        entry = round(ce_ltp, 2) if ce_ltp > 0 else round(spot * 0.006, 2)
        sl_pct = 0.50; tgt1_x = 1.35; tgt2_x = 1.7
    elif score == -1:
        action = "MILD BUY PE"
        color  = "var(--red)"
        css_hdr = "pe-bg"
        strike_rec = atm
        entry = round(pe_ltp, 2) if pe_ltp > 0 else round(spot * 0.006, 2)
        sl_pct = 0.50; tgt1_x = 1.35; tgt2_x = 1.7
    else:
        action = "HOLD / WAIT"
        color  = "var(--yellow)"
        css_hdr = "hl-bg"
        strike_rec = atm
        entry = round(max(ce_ltp, pe_ltp), 2)
        sl_pct = 0; tgt1_x = 0; tgt2_x = 0

    sl   = round(entry * (1 - sl_pct), 2) if sl_pct and entry > 0 else 0.0
    tgt1 = round(entry * tgt1_x, 2) if tgt1_x and entry > 0 else 0.0
    tgt2 = round(entry * tgt2_x, 2) if tgt2_x and entry > 0 else 0.0

    # \u2500\u2500 Narrative \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if score >= 2:
        narrative = (
            f"\ud83d\udcc8 Bullish setup on {symbol} @ \u20b9{spot:,.0f}. PCR {pcr_val:.2f} shows heavy put writing, "
            f"Max Pain at \u20b9{pain:.0f} creates upward magnetic pull, and put wall at \u20b9{put_wall:.0f} "
            f"acts as strong support. Consider {atm:.0f} CE."
        )
    elif score <= -2:
        narrative = (
            f"\ud83d\udcc9 Bearish setup on {symbol} @ \u20b9{spot:,.0f}. PCR {pcr_val:.2f} signals excessive call "
            f"accumulation, Max Pain at \u20b9{pain:.0f} pulls price down, and call writers defend "
            f"\u20b9{call_wall:.0f} aggressively. Consider {atm:.0f} PE."
        )
    elif score > 0:
        narrative = (
            f"\u2197\ufe0f Mildly bullish on {symbol} @ \u20b9{spot:,.0f}. Slight positive edge but limited conviction. "
            f"Wait for spot to close above {atm+50:.0f} on 5-min candle before entering."
        )
    elif score < 0:
        narrative = (
            f"\u2198\ufe0f Mildly bearish on {symbol} @ \u20b9{spot:,.0f}. Slight negative tilt. "
            f"Wait for spot to break {atm-50:.0f} with volume before entering."
        )
    else:
        narrative = (
            f"\u23f3 Neutral on {symbol} @ \u20b9{spot:,.0f}. No clear directional edge. "
            f"Watch for breakout above \u20b9{call_wall:.0f} (bullish) or below \u20b9{put_wall:.0f} (bearish) "
            f"before committing."
        )

    return {
        "action": action, "score": score, "conf": conf,
        "color": color, "css_hdr": css_hdr,
        "narrative": narrative, "signals": signals,
        "entry": entry, "sl": sl, "tgt1": tgt1, "tgt2": tgt2,
        "strike": strike_rec, "atm": atm,
        "pcr": pcr_val, "max_pain": pain,
        "call_wall": call_wall, "put_wall": put_wall,
        "vw_iv": overall_iv, "put_iv": put_iv_skew, "call_iv": call_iv_skew,
        "days_left": days_left, "ce_ltp": ce_ltp, "pe_ltp": pe_ltp,
        "ce_g": ce_g, "pe_g": pe_g,
    }


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# FETCH ORCHESTRATOR
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
def do_fetch(exchange, symbol, expiry, r_rate):
    """
    Central fetch + parse + enrich.
    Updates session_state df, spot, rec.
    """
    now
