# -*- coding: utf-8 -*-
"""
Option Chain Live Analyzer - Nifty 50 & Sensex
Streamlit App | Live Scraping | Greeks | AI Recommendation | Start/Stop Control
"""
import sys

# Prevent surrogate/encoding errors on Windows and narrow-build Python
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

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

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Option Chain Analyzer",
    page_icon="[CHT]",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════
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
.pill-warn { background:rgba(255,214,0,0.1);   color:#ffd600; border:1px solid rgba(255,214,0,0.2); }

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

section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border);
}

.stButton > button {
    width: 100%;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border: none !important;
    transition: all 0.2s !important;
}

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

.err-box {
    background: rgba(255,61,90,0.07);
    border: 1px solid rgba(255,61,90,0.25);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.82rem;
    line-height: 1.8;
    margin: 0.5rem 0;
}

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


# ══════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "trading_active":    False,
        "last_fetch_time":   0.0,
        "fetch_count":       0,
        "df":                None,
        "spot":              0.0,
        "expiries":          [],
        "chosen_expiry":     "",
        "rec":               None,
        "log":               [],
        "error":             "",
        "nse_session":       None,
        "nse_session_ready": False,
        "fetch_status":      "idle",
        "symbol":            "NIFTY",
        "exchange":          "NSE",
        "manual_spot":       0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def log_msg(msg: str, level: str = "info"):
    ts   = datetime.now().strftime("%H:%M:%S")
    icon = {"info": "ℹ", "ok": "✅", "warn": "⚠️", "err": "❌"}.get(level, "•")
    st.session_state["log"].insert(0, f"[{ts}] {icon} {msg}")
    st.session_state["log"] = st.session_state["log"][:80]


# ══════════════════════════════════════════════════════════
# BLACK-SCHOLES + GREEKS
# ══════════════════════════════════════════════════════════
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
                             lo, hi, maxiter=100, xtol=1e-5))
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


# ══════════════════════════════════════════════════════════
# NSE FETCH  (proper cookie-warm approach)
# ══════════════════════════════════════════════════════════
_UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]


def nse_headers(ua=None):
    ua = ua or random.choice(_UA_LIST)
    return {
        "User-Agent":        ua,
        "Accept":            "application/json, text/plain, */*",
        "Accept-Language":   "en-US,en;q=0.9",
        "Accept-Encoding":   "gzip, deflate, br",
        "Referer":           "https://www.nseindia.com/option-chain",
        "sec-ch-ua":         '"Chromium";v="124", "Google Chrome";v="124"',
        "sec-ch-ua-mobile":  "?0",
        "sec-ch-ua-platform": '"Windows"',
        "Connection":        "keep-alive",
        "DNT":               "1",
    }


def warm_nse_session(force=False):
    """Build a requests.Session with valid NSE cookies."""
    if (not force
            and st.session_state["nse_session"] is not None
            and st.session_state["nse_session_ready"]):
        return st.session_state["nse_session"], True

    ua   = random.choice(_UA_LIST)
    sess = requests.Session()
    sess.headers.update(nse_headers(ua))

    for url, label in [
        ("https://www.nseindia.com",              "NSE homepage"),
        ("https://www.nseindia.com/option-chain",  "NSE option-chain page"),
    ]:
        try:
            r = sess.get(url, timeout=15)
            log_msg(f"Warmed {label} → HTTP {r.status_code}", "info")
            time.sleep(2)
        except Exception as e:
            log_msg(f"Warm-up warning at {label}: {e}", "warn")
            time.sleep(2)

    st.session_state["nse_session"]       = sess
    st.session_state["nse_session_ready"] = True
    return sess, True


def fetch_nse_oc(symbol: str, max_retries: int = 4):
    """Fetch NSE option chain JSON. Returns (dict|None, error_str)."""
    url  = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    sess, _ = warm_nse_session()

    for attempt in range(max_retries):
        try:
            resp = sess.get(url, timeout=15)
            code = resp.status_code

            if code == 200:
                data = resp.json()
                if data.get("records"):
                    log_msg(f"NSE {symbol} OK (attempt {attempt+1})", "ok")
                    return data, ""
                return None, "NSE returned empty records. Market may be closed."

            elif code == 429:
                wait = 10 * (attempt + 1)
                log_msg(f"NSE rate-limited (429). Waiting {wait}s…", "warn")
                time.sleep(wait)

            elif code in (401, 403):
                log_msg(f"NSE cookie expired ({code}). Re-warming…", "warn")
                st.session_state["nse_session_ready"] = False
                sess, _ = warm_nse_session(force=True)
                time.sleep(3)

            elif code == 503:
                log_msg("NSE service unavailable (503).", "warn")
                time.sleep(6)

            else:
                log_msg(f"NSE HTTP {code} on attempt {attempt+1}", "warn")
                time.sleep(4)

        except requests.exceptions.SSLError as e:
            log_msg(f"NSE SSL error: {e}", "err"); time.sleep(5)
        except requests.exceptions.ConnectionError as e:
            log_msg(f"NSE connection error: {e}", "err"); time.sleep(6)
        except requests.exceptions.Timeout:
            log_msg("NSE timeout - retrying…", "warn"); time.sleep(5)
        except ValueError:
            log_msg("NSE returned non-JSON", "err"); time.sleep(4)
        except Exception as e:
            log_msg(f"NSE unexpected: {e}", "err"); time.sleep(4)

    return None, (f"NSE fetch failed after {max_retries} attempts. "
                  "Try Re-warm Session or check market hours (9:15–15:30 IST).")


def parse_nse(raw: dict, expiry: str):
    """Parse NSE JSON for a specific expiry. Returns (DataFrame, spot)."""
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
                "Strike":  k,
                "C_OI":    float(ce.get("openInterest", 0) or 0),
                "C_ChgOI": float(ce.get("changeinOpenInterest", 0) or 0),
                "C_Vol":   float(ce.get("totalTradedVolume", 0) or 0),
                "C_IV":    float(ce.get("impliedVolatility", 0) or 0),
                "C_LTP":   float(ce.get("lastPrice", 0) or 0),
                "C_Bid":   float(ce.get("bidprice", 0) or 0),
                "C_Ask":   float(ce.get("askPrice", 0) or 0),
                "C_Chng":  float(ce.get("change", 0) or 0),
                "P_OI":    float(pe.get("openInterest", 0) or 0),
                "P_ChgOI": float(pe.get("changeinOpenInterest", 0) or 0),
                "P_Vol":   float(pe.get("totalTradedVolume", 0) or 0),
                "P_IV":    float(pe.get("impliedVolatility", 0) or 0),
                "P_LTP":   float(pe.get("lastPrice", 0) or 0),
                "P_Bid":   float(pe.get("bidprice", 0) or 0),
                "P_Ask":   float(pe.get("askPrice", 0) or 0),
                "P_Chng":  float(pe.get("change", 0) or 0),
            })
        df = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)
        return df, spot
    except Exception as e:
        log_msg(f"NSE parse error: {e}", "err")
        return pd.DataFrame(), 0.0


# ══════════════════════════════════════════════════════════
# BSE FETCH
# ══════════════════════════════════════════════════════════
def fetch_bse_oc(expiry: str = "", max_retries: int = 3):
    """Try multiple BSE endpoints. Returns (dict|None, error_str)."""
    ua   = random.choice(_UA_LIST)
    sess = requests.Session()
    sess.headers.update({
        "User-Agent":      ua,
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         "https://www.bseindia.com/markets/Derivatives/DeriReports/DeriOptionchain.html",
        "Origin":          "https://www.bseindia.com",
        "Connection":      "keep-alive",
    })

    # Warm cookies
    try:
        sess.get("https://www.bseindia.com", timeout=10)
        time.sleep(1.5)
        sess.get("https://www.bseindia.com/markets/Derivatives/DeriReports/DeriOptionchain.html",
                 timeout=10)
        time.sleep(2)
    except Exception as e:
        log_msg(f"BSE warm-up: {e}", "warn")

    endpoints = [
        f"https://api.bseindia.com/BseIndiaAPI/api/OptionChain/w?scripcd=&Expiry={expiry}&StrikePrice=&optiontype=&seriesid=12",
        f"https://api.bseindia.com/BseIndiaAPI/api/OptionChain/w?scripcd=&Expiry={expiry}&optiontype=&seriesid=12",
        "https://api.bseindia.com/BseIndiaAPI/api/OptionChainData/w",
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
                            log_msg(f"BSE OK via {url.split('?')[0].split('/')[-1]}", "ok")
                            return data, ""
                    except ValueError:
                        pass
                elif r.status_code == 429:
                    time.sleep(10 * (attempt + 1))
                elif r.status_code in (403, 401):
                    time.sleep(5); break
                time.sleep(2)
            except Exception as e:
                log_msg(f"BSE {attempt+1}: {e}", "warn"); time.sleep(3)

    return None, ("BSE API blocked. BSE requires browser-based session. "
                  "Switch to NSE for reliable data, or use Manual Spot Override.")


def parse_bse(raw):
    """Parse BSE JSON → wide-format DataFrame."""
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
            opt_type = str(rec.get("CPType", rec.get("OPTION_TYPE",
                              rec.get("CP_TYPE", ""))) or "").upper().strip()
            ltp    = float(rec.get("LTP",  rec.get("LAST_PRICE", 0)) or 0)
            oi     = float(rec.get("OI",   rec.get("OPEN_INT", 0)) or 0)
            chgoi  = float(rec.get("ChgOI",rec.get("CHG_IN_OI", 0)) or 0)
            vol    = float(rec.get("TotalVolume", rec.get("VOLUME", 0)) or 0)
            iv     = float(rec.get("IV",   rec.get("IMP_VOL", 0)) or 0)
            bid    = float(rec.get("BidPrice", rec.get("BID_PRICE", 0)) or 0)
            ask    = float(rec.get("AskPrice", rec.get("ASK_PRICE", 0)) or 0)
            chng   = float(rec.get("Chng", rec.get("NET_CHANGE", 0)) or 0)
            rows.append({"Strike": strike, "Type": opt_type, "OI": oi,
                          "ChgOI": chgoi, "Vol": vol, "IV": iv,
                          "LTP": ltp, "Bid": bid, "Ask": ask, "Chng": chng})

        df_raw = pd.DataFrame(rows)
        if df_raw.empty:
            return pd.DataFrame(), 0.0

        ce = df_raw[df_raw["Type"].isin(["CE","C","CALL"])].set_index("Strike").add_prefix("C_").drop(columns="C_Type", errors="ignore")
        pe = df_raw[df_raw["Type"].isin(["PE","P","PUT"])].set_index("Strike").add_prefix("P_").drop(columns="P_Type", errors="ignore")
        df = ce.join(pe, how="outer").fillna(0).reset_index().sort_values("Strike")
        return df, 0.0
    except Exception as e:
        log_msg(f"BSE parse error: {e}", "err")
        return pd.DataFrame(), 0.0


# ══════════════════════════════════════════════════════════
# GREEKS ENRICHMENT
# ══════════════════════════════════════════════════════════
def enrich_greeks(df, spot, T, r):
    if df.empty or spot <= 0 or T <= 0:
        return df
    for opt_type, pfx in [("CE","C_"), ("PE","P_")]:
        ltp_col = f"{pfx}LTP"
        iv_col  = f"{pfx}IV"
        if ltp_col not in df.columns:
            continue
        cols = {k: [] for k in ["Delta","Gamma","Theta","Vega","Rho","IV_Calc"]}
        for _, row in df.iterrows():
            K        = float(row.get("Strike", 0))
            ltp      = float(row.get(ltp_col, 0))
            iv_given = float(row.get(iv_col, 0))
            iv       = iv_given / 100 if iv_given > 0 else calc_iv(ltp, spot, K, T, r, opt_type)
            g        = calc_greeks(spot, K, T, r, iv, opt_type)
            for gn in ["delta","gamma","theta","vega","rho"]:
                cols[gn.capitalize()].append(g[gn])
            cols["IV_Calc"].append(round(iv * 100, 2))
        for gn in ["Delta","Gamma","Theta","Vega","Rho","IV_Calc"]:
            df[f"{pfx}{gn}"] = cols[gn]
    return df


# ══════════════════════════════════════════════════════════
# OPTION CHAIN METRICS
# ══════════════════════════════════════════════════════════
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
        pain = sum(
            max(0, s - row["Strike"]) * float(row.get("C_OI", 0)) +
            max(0, row["Strike"] - s) * float(row.get("P_OI", 0))
            for _, row in df.iterrows()
        )
        pains.append(pain)
    return strikes[int(np.argmin(pains))]


def find_atm(df, spot):
    if df.empty or spot <= 0:
        return 0.0
    return float(df.loc[(df["Strike"] - spot).abs().idxmin(), "Strike"])


def oi_walls(df):
    if df.empty:
        return 0.0, 0.0, 0.0, 0.0
    ci = df["C_OI"].idxmax() if "C_OI" in df.columns else 0
    pi = df["P_OI"].idxmax() if "P_OI" in df.columns else 0
    return (float(df.loc[ci, "Strike"]), float(df.loc[pi, "Strike"]),
            float(df.loc[ci, "C_OI"]),   float(df.loc[pi, "P_OI"]))


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
    i = idx[0]; n = 4
    put_iv  = df.iloc[max(0,i-n):i]["P_IV"].replace(0, np.nan).mean()  if "P_IV"  in df.columns else 0
    call_iv = df.iloc[i+1:min(len(df),i+1+n)]["C_IV"].replace(0, np.nan).mean() if "C_IV" in df.columns else 0
    return round(float(put_iv or 0), 2), round(float(call_iv or 0), 2)


def atm_ltp_greeks(df, atm):
    row = df[df["Strike"] == atm]
    if row.empty:
        return 0.0, 0.0, {}, {}
    r    = row.iloc[0]
    ce_l = float(r.get("C_LTP", 0))
    pe_l = float(r.get("P_LTP", 0))
    ce_g = {g: round(float(r.get(f"C_{g}", 0)), 4) for g in ["Delta","Gamma","Theta","Vega","Rho"]}
    pe_g = {g: round(float(r.get(f"P_{g}", 0)), 4) for g in ["Delta","Gamma","Theta","Vega","Rho"]}
    return ce_l, pe_l, ce_g, pe_g


def days_to_exp(expiry_str: str):
    if not expiry_str:
        return 7
    try:
        exp = datetime.strptime(expiry_str, "%d-%b-%Y").date()
        return max((exp - date.today()).days, 0)
    except Exception:
        return 7


# ══════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE  (7-factor scoring)
# ══════════════════════════════════════════════════════════
def make_recommendation(df, spot, expiry_str, symbol):
    days_left            = days_to_exp(expiry_str)
    pcr_val              = compute_pcr(df)
    pain                 = compute_max_pain(df)
    atm                  = find_atm(df, spot)
    call_wall, put_wall, cw_oi, pw_oi = oi_walls(df)
    vw_civ, vw_piv       = vw_iv(df)
    put_iv_sk, call_iv_sk = iv_skew(df, atm)
    ce_ltp, pe_ltp, ce_g, pe_g = atm_ltp_greeks(df, atm)
    c_chgoi = float(df["C_ChgOI"].sum()) if "C_ChgOI" in df.columns else 0
    p_chgoi = float(df["P_ChgOI"].sum()) if "P_ChgOI" in df.columns else 0
    overall_iv = max(vw_civ, vw_piv)

    score   = 0
    signals = {}

    # 1. PCR
    if   pcr_val >= 1.4: score += 2; signals["PCR"] = (f"{pcr_val:.3f} - Heavy put writing → bullish", "bull")
    elif pcr_val >= 1.1: score += 1; signals["PCR"] = (f"{pcr_val:.3f} - Elevated puts → mildly bullish", "bull")
    elif pcr_val <= 0.6: score -= 2; signals["PCR"] = (f"{pcr_val:.3f} - Heavy call writing → bearish", "bear")
    elif pcr_val <= 0.9: score -= 1; signals["PCR"] = (f"{pcr_val:.3f} - Low PCR → mildly bearish", "bear")
    else:                             signals["PCR"] = (f"{pcr_val:.3f} - Balanced → neutral", "neut")

    # 2. Max Pain
    mp_pct = (pain - spot) / spot * 100 if spot > 0 else 0
    if   mp_pct >  0.5: score += 1; signals["Max Pain"] = (f"₹{pain:.0f} is {mp_pct:.1f}% above spot → upward pull", "bull")
    elif mp_pct < -0.5: score -= 1; signals["Max Pain"] = (f"₹{pain:.0f} is {abs(mp_pct):.1f}% below spot → downward pull", "bear")
    else:                             signals["Max Pain"] = (f"₹{pain:.0f} ≈ spot - equilibrium", "neut")

    # 3. OI Walls
    if call_wall > 0 and put_wall > 0:
        cd = (call_wall - spot) / spot * 100 if spot > 0 else 999
        pd_ = (spot - put_wall)  / spot * 100 if spot > 0 else 999
        if   cd  < 0.4: score -= 1; signals["OI Walls"] = (f"Call wall ₹{call_wall:.0f} very close - strong resistance", "bear")
        elif pd_ < 0.4: score += 1; signals["OI Walls"] = (f"Put wall ₹{put_wall:.0f} very close - strong support", "bull")
        else:                        signals["OI Walls"] = (f"CE wall ₹{call_wall:.0f} (R) | PE wall ₹{put_wall:.0f} (S)", "neut")

    # 4. OI Change
    if   c_chgoi > 0 and p_chgoi < 0: score -= 1; signals["OI Change"] = ("Fresh CE OI + PE OI unwinding → bearish writing", "bear")
    elif p_chgoi > 0 and c_chgoi < 0: score += 1; signals["OI Change"] = ("Fresh PE OI + CE OI unwinding → bullish writing", "bull")
    elif c_chgoi > 0 and p_chgoi > 0:               signals["OI Change"] = ("Both CE & PE OI rising → breakout expected", "neut")
    else:                                            signals["OI Change"] = ("OI change ambiguous", "neut")

    # 5. IV Skew
    iv_diff = put_iv_sk - call_iv_sk
    if   iv_diff >  4: score -= 1; signals["IV Skew"] = (f"Put IV {put_iv_sk:.1f}% >> Call IV {call_iv_sk:.1f}% → fear/bearish skew", "bear")
    elif iv_diff < -4: score += 1; signals["IV Skew"] = (f"Call IV {call_iv_sk:.1f}% >> Put IV {put_iv_sk:.1f}% → unusual bullish skew", "bull")
    else:                           signals["IV Skew"] = (f"Balanced - Put IV {put_iv_sk:.1f}% vs Call IV {call_iv_sk:.1f}%", "neut")

    # 6. IV Regime
    if   overall_iv > 28: signals["IV Regime"] = (f"High IV ({overall_iv:.1f}%) → option premium elevated; prefer selling", "warn")
    elif overall_iv < 11: signals["IV Regime"] = (f"Low IV ({overall_iv:.1f}%) → options cheap; ideal for buying", "bull")
    else:                  signals["IV Regime"] = (f"Normal IV ({overall_iv:.1f}%) → directional buying viable", "neut")

    # 7. Time Decay
    if   days_left <= 2: score -= 1; signals["Time Decay"] = (f"Only {days_left}d left - Theta devastating; avoid buying", "bear")
    elif days_left <= 7:              signals["Time Decay"] = (f"{days_left}d to expiry - Theta accelerating; prefer scalps", "warn")
    else:                             signals["Time Decay"] = (f"{days_left}d to expiry - Adequate time for positional trades", "bull")

    # Trade parameters
    conf    = min(round(abs(score) / 7 * 100), 90)
    sl_pct  = 0.40; tgt1_x = 1.50; tgt2_x = 2.30

    if score >= 2:
        action = "BUY CE"; css = "ce-bg"; color = "var(--green)"
        entry  = round(ce_ltp, 2) if ce_ltp > 0 else round(spot * 0.006, 2)
        narrative = (f"[UP] Bullish setup on {symbol} @ ₹{spot:,.0f}. PCR {pcr_val:.2f} shows heavy "
                     f"put writing. Max Pain ₹{pain:.0f} creates upward pull. Put wall @ ₹{put_wall:.0f} is support. "
                     f"Consider {atm:.0f} CE.")
    elif score <= -2:
        action = "BUY PE"; css = "pe-bg"; color = "var(--red)"
        entry  = round(pe_ltp, 2) if pe_ltp > 0 else round(spot * 0.006, 2)
        narrative = (f"[DN] Bearish setup on {symbol} @ ₹{spot:,.0f}. PCR {pcr_val:.2f} signals excessive "
                     f"call accumulation. Max Pain ₹{pain:.0f} pulls down. Call writers defend ₹{call_wall:.0f}. "
                     f"Consider {atm:.0f} PE.")
    elif score == 1:
        action = "MILD BUY CE"; css = "ce-bg"; color = "var(--green)"
        entry  = round(ce_ltp, 2) if ce_ltp > 0 else round(spot * 0.006, 2)
        sl_pct = 0.50; tgt1_x = 1.35; tgt2_x = 1.70
        narrative = (f"↗️ Mildly bullish on {symbol} @ ₹{spot:,.0f}. Slight positive edge. "
                     f"Wait for spot to sustain above {atm+50:.0f} before entering. Small size only.")
    elif score == -1:
        action = "MILD BUY PE"; css = "pe-bg"; color = "var(--red)"
        entry  = round(pe_ltp, 2) if pe_ltp > 0 else round(spot * 0.006, 2)
        sl_pct = 0.50; tgt1_x = 1.35; tgt2_x = 1.70
        narrative = (f"↘️ Mildly bearish on {symbol} @ ₹{spot:,.0f}. Slight negative tilt. "
                     f"Wait for spot to break {atm-50:.0f} with volume. Tight stops.")
    else:
        action = "HOLD / WAIT"; css = "hl-bg"; color = "var(--yellow)"
        entry  = round(max(ce_ltp, pe_ltp), 2)
        sl_pct = 0; tgt1_x = 0; tgt2_x = 0
        narrative = (f"⏳ Neutral on {symbol} @ ₹{spot:,.0f}. No clear edge. "
                     f"Breakout above ₹{call_wall:.0f} = bullish. Break below ₹{put_wall:.0f} = bearish. "
                     f"Wait for confirmation.")

    sl   = round(entry * (1 - sl_pct), 2) if sl_pct and entry > 0 else 0.0
    tgt1 = round(entry * tgt1_x, 2) if tgt1_x and entry > 0 else 0.0
    tgt2 = round(entry * tgt2_x, 2) if tgt2_x and entry > 0 else 0.0

    return {
        "action": action, "score": score, "conf": conf,
        "color": color, "css_hdr": css, "narrative": narrative,
        "signals": signals, "entry": entry, "sl": sl, "tgt1": tgt1, "tgt2": tgt2,
        "strike": atm, "atm": atm,
        "pcr": pcr_val, "max_pain": pain,
        "call_wall": call_wall, "put_wall": put_wall,
        "vw_iv": overall_iv, "put_iv": put_iv_sk, "call_iv": call_iv_sk,
        "days_left": days_left, "ce_ltp": ce_ltp, "pe_ltp": pe_ltp,
        "ce_g": ce_g, "pe_g": pe_g,
    }


# ══════════════════════════════════════════════════════════
# FETCH ORCHESTRATOR
# ══════════════════════════════════════════════════════════
def do_fetch(exchange, symbol, expiry, r_rate):
    now     = time.time()
    elapsed = now - st.session_state["last_fetch_time"]
    if elapsed < 2.0:
        time.sleep(2.0 - elapsed)

    st.session_state["fetch_status"] = "fetching"
    st.session_state["error"]        = ""

    if exchange == "NSE":
        raw, err = fetch_nse_oc(symbol)
        if raw is None:
            st.session_state["fetch_status"] = "error"
            st.session_state["error"] = err
            return

        exp_list = raw.get("records", {}).get("expiryDates", [])
        if exp_list:
            st.session_state["expiries"] = exp_list
            if not expiry or expiry not in exp_list:
                expiry = exp_list[0]
                st.session_state["chosen_expiry"] = expiry

        df, spot = parse_nse(raw, expiry)
    else:
        raw, err = fetch_bse_oc(expiry)
        if raw is None:
            st.session_state["fetch_status"] = "error"
            st.session_state["error"] = err
            return
        df, spot = parse_bse(raw)

    if df.empty:
        st.session_state["fetch_status"] = "error"
        st.session_state["error"] = ("Data fetched but option chain is empty. "
                                      "Market may be closed or selected expiry has no data.")
        return

    # Manual spot override
    manual = st.session_state.get("manual_spot", 0.0)
    if manual > 0:
        spot = manual

    if spot <= 0:
        st.session_state["fetch_status"] = "error"
        st.session_state["error"] = ("Spot price = 0. Set Manual Spot Override in sidebar.")
        return

    days = days_to_exp(expiry)
    T    = max(days, 1) / 365.0
    df   = enrich_greeks(df, spot, T, r_rate)

    st.session_state["df"]              = df
    st.session_state["spot"]            = spot
    st.session_state["last_fetch_time"] = time.time()
    st.session_state["fetch_count"]    += 1
    st.session_state["fetch_status"]    = "ok"
    st.session_state["rec"]             = make_recommendation(df, spot, expiry, symbol)

    log_msg(f"Updated - spot ₹{spot:.0f}, {len(df)} strikes, expiry {expiry}", "ok")


# ══════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════
def render_kpi_html(label, value, sub="", klass=""):
    return (f'<div class="kpi {klass}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')


def render_rec(rec):
    action = rec["action"]; css = rec["css_hdr"]; color = rec["color"]
    entry  = rec["entry"];  sl  = rec["sl"];      tgt1  = rec["tgt1"]; tgt2 = rec["tgt2"]
    rr     = round((tgt1 - entry) / (entry - sl), 2) if sl and entry > sl and tgt1 > entry else "-"
    st.markdown(f"""
    <div class="rec-wrap">
      <div class="rec-header {css}">
        <div class="rec-action" style="color:{color}">{action}</div>
        <div style="font-size:0.78rem;color:var(--muted);">
          Strike <b style="color:var(--text)">₹{int(rec['strike'])}</b> &nbsp;|&nbsp;
          Confidence <b style="color:{color}">{rec['conf']}%</b> &nbsp;|&nbsp;
          Score <b style="color:var(--text)">{rec['score']:+d}/7</b>
        </div>
      </div>
      <div class="rec-body">
        <div style="font-size:0.82rem;line-height:1.7;margin-bottom:0.6rem;">{rec['narrative']}</div>
        <div class="trade-grid">
          <div class="trade-box"><div class="trade-box-label">Entry</div>
            <div class="trade-box-val" style="color:{color}">₹{entry}</div></div>
          <div class="trade-box"><div class="trade-box-label">Stop Loss</div>
            <div class="trade-box-val" style="color:var(--red)">₹{sl}</div></div>
          <div class="trade-box"><div class="trade-box-label">Target 1</div>
            <div class="trade-box-val" style="color:var(--green)">₹{tgt1}</div></div>
          <div class="trade-box"><div class="trade-box-label">Target 2</div>
            <div class="trade-box-val" style="color:var(--green)">₹{tgt2}</div></div>
          <div class="trade-box"><div class="trade-box-label">Risk:Reward</div>
            <div class="trade-box-val">{rr}</div></div>
        </div>
        <div style="font-size:0.65rem;color:var(--muted);margin-top:0.5rem;">
          ⚠ Algorithmic analysis only. Not SEBI-registered advice.
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


def render_signals(signals):
    icons = {"PCR":"[CHT]","Max Pain":"[*]","OI Walls":"[OI]",
             "OI Change":"Refresh","IV Skew":"[GRK]","IV Regime":"[IV]","Time Decay":"[T]"}
    rows  = ""
    for k, (txt, sent) in signals.items():
        pcls  = {"bull":"pill-bull","bear":"pill-bear","warn":"pill-warn"}.get(sent,"pill-neut")
        plbl  = {"bull":"BULL","bear":"BEAR","warn":"WARN"}.get(sent,"NEUT")
        rows += (f'<div class="sig-row"><div class="sig-icon">{icons.get(k,"•")}</div>'
                 f'<div class="sig-label">{k}</div>'
                 f'<div class="sig-text">{txt}<span class="pill {pcls}">{plbl}</span></div></div>')
    st.markdown(f'<div style="background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:0.8rem 1rem;">{rows}</div>',
                unsafe_allow_html=True)


def render_greeks(ce_g, pe_g):
    gnames = ["Delta","Gamma","Theta","Vega","Rho"]
    ce_r = "".join(f'<div class="greek-row"><span class="greek-name">{g}</span><span class="greek-val-ce">{ce_g.get(g,"-")}</span></div>' for g in gnames)
    pe_r = "".join(f'<div class="greek-row"><span class="greek-name">{g}</span><span class="greek-val-pe">{pe_g.get(g,"-")}</span></div>' for g in gnames)
    st.markdown(f"""
    <div class="greek-grid">
      <div class="greek-card"><div class="greek-title" style="color:var(--green)">ATM Call Greeks</div>{ce_r}</div>
      <div class="greek-card"><div class="greek-title" style="color:var(--red)">ATM Put Greeks</div>{pe_r}</div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════
def main():
    # ── Title ─────────────────────────────────────────────
    st.markdown("""
    <div class="top-header">
      <div>
        <div class="app-title">OPTION CHAIN ANALYZER</div>
        <div class="app-sub">Nifty 50 · Sensex · Live Greeks · AI Recommendation</div>
      </div>
    </div><hr>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("### [CFG] Settings")

        exchange = st.selectbox("Exchange", ["NSE (Nifty)", "BSE (Sensex)"])
        is_nse   = "NSE" in exchange
        exch_key = "NSE" if is_nse else "BSE"

        symbol   = st.selectbox("Symbol",
                                 ["NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY"] if is_nse else ["SENSEX"])

        exp_list = st.session_state.get("expiries", [])
        if exp_list:
            chosen_expiry = st.selectbox("Expiry", exp_list)
            st.session_state["chosen_expiry"] = chosen_expiry
        else:
            chosen_expiry = st.text_input("Expiry (DD-Mon-YYYY)",
                                          value=st.session_state["chosen_expiry"])
            st.session_state["chosen_expiry"] = chosen_expiry

        r_rate = st.slider("Risk-Free Rate (%)", 4.0, 9.0, 6.5, 0.25) / 100

        manual_spot = st.number_input(
            "Manual Spot Override (0 = auto)",
            min_value=0.0, value=st.session_state.get("manual_spot", 0.0), step=100.0
        )
        st.session_state["manual_spot"] = manual_spot

        refresh_sec  = st.selectbox("Auto-Refresh (seconds)", [10, 30, 60, 120, 300], index=1,
                                     format_func=lambda x: f"{x}s")
        n_strikes    = st.slider("±Strikes around ATM", 5, 30, 15)
        show_chain   = st.checkbox("Show Option Chain Table", True)

        st.markdown("---")
        st.markdown("""
<div style="font-size:0.68rem;color:#4a5568;line-height:1.8;">
<b>Tips if fetch fails:</b><br>
• Click Re-warm Session<br>
• Wait 10-15s and retry<br>
• Set Manual Spot Override<br>
• NSE works 9:15–15:30 IST<br>
• BSE API often needs browser auth<br>
• Use NSE as primary source
</div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # CONTROL PANEL  ← START / STOP / FETCH ONCE / RE-WARM
    # ══════════════════════════════════════════════════════
    is_active    = st.session_state["trading_active"]
    fetch_status = st.session_state["fetch_status"]
    fc           = st.session_state["fetch_count"]
    last_ts      = (datetime.fromtimestamp(st.session_state["last_fetch_time"]).strftime("%H:%M:%S")
                    if st.session_state["last_fetch_time"] > 0 else "-")

    # Status badge
    if is_active:
        dot_html  = '<div class="status-dot-live"></div>'
        stat_lbl  = "LIVE - Auto-refreshing"
        stat_col  = "var(--green)"
    elif fetch_status == "ok":
        dot_html  = '<div class="status-dot-stop"></div>'
        stat_lbl  = "PAUSED"
        stat_col  = "var(--yellow)"
    else:
        dot_html  = '<div class="status-dot-idle"></div>'
        stat_lbl  = "IDLE"
        stat_col  = "var(--muted)"

    st.markdown(f"""
    <div class="ctrl-banner">
      {dot_html}
      <span style="color:{stat_col};font-family:'Syne',sans-serif;font-weight:700;
                   font-size:0.85rem;letter-spacing:1px;">{stat_lbl}</span>
      <span style="color:var(--muted);font-size:0.72rem;">
        Fetches: <b style="color:var(--text)">{fc}</b> &nbsp;|&nbsp;
        Last: <b style="color:var(--text)">{last_ts}</b> &nbsp;|&nbsp;
        <b style="color:var(--accent)">{symbol}</b> / <b style="color:var(--accent)">{exch_key}</b>
      </span>
    </div>""", unsafe_allow_html=True)

    # ── 4 control buttons ──────────────────────────────────
    b1, b2, b3, b4 = st.columns([1, 1, 1.2, 1.5])

    with b1:
        start_clicked = st.button(
            "[>]  START",
            disabled=is_active,
            help="Start live auto-refresh",
            use_container_width=True,
            type="primary"
        )

    with b2:
        stop_clicked = st.button(
            "[.]  STOP",
            disabled=not is_active,
            help="Stop auto-refresh",
            use_container_width=True,
            type="secondary"
        )

    with b3:
        fetch_once_clicked = st.button(
            "Refresh  Fetch Once",
            help="Single manual fetch without starting auto-refresh",
            use_container_width=True
        )

    with b4:
        rewarm_clicked = st.button(
            "Key  Re-warm NSE Session",
            help="Force-refresh NSE cookies (use if getting 403 / blocked errors)",
            use_container_width=True
        )

    # ── Handle button actions ──────────────────────────────
    if start_clicked:
        st.session_state["trading_active"] = True
        st.session_state["symbol"]         = symbol
        st.session_state["exchange"]       = exch_key
        log_msg(f"[>] Trading STARTED - {exch_key} {symbol} every {refresh_sec}s", "ok")
        with st.spinner("Warming NSE session and fetching first data…"):
            do_fetch(exch_key, symbol, st.session_state["chosen_expiry"], r_rate)
        st.rerun()

    if stop_clicked:
        st.session_state["trading_active"] = False
        log_msg("[.] Trading STOPPED by user", "warn")
        st.rerun()

    if fetch_once_clicked:
        st.session_state["symbol"]   = symbol
        st.session_state["exchange"] = exch_key
        with st.spinner("Fetching option chain…"):
            do_fetch(exch_key, symbol, st.session_state["chosen_expiry"], r_rate)
        st.rerun()

    if rewarm_clicked:
        st.session_state["nse_session_ready"] = False
        st.session_state["nse_session"]       = None
        with st.spinner("Re-warming NSE session (takes ~5s)…"):
            warm_nse_session(force=True)
        log_msg("Key NSE session re-warmed", "ok")
        st.rerun()

    # ── Error display ──────────────────────────────────────
    err = st.session_state.get("error", "")
    if err:
        st.markdown(f"""
        <div class="err-box">
          ❌ <b>Fetch Error:</b> {err}<br><br>
          <b>Solutions:</b><br>
          &nbsp;&nbsp;1. Click <b>Key Re-warm NSE Session</b> then <b>Refresh Fetch Once</b><br>
          &nbsp;&nbsp;2. Set <b>Manual Spot Override</b> in sidebar if spot = 0<br>
          &nbsp;&nbsp;3. Confirm market hours: NSE works 9:15 AM–3:30 PM IST on weekdays<br>
          &nbsp;&nbsp;4. BSE often requires browser-based auth - switch to NSE<br>
          &nbsp;&nbsp;5. Wait 30s between attempts to avoid rate limits
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # DATA DISPLAY
    # ══════════════════════════════════════════════════════
    df   = st.session_state.get("df")
    spot = st.session_state.get("spot", 0.0)
    rec  = st.session_state.get("rec")

    if df is None or df.empty or spot == 0:
        st.markdown("""
        <div class="info-box">
          [SIG] <b>No data loaded yet.</b><br><br>
          <b>Quick start:</b><br>
          &nbsp;&nbsp;1. Select Exchange &amp; Symbol in sidebar<br>
          &nbsp;&nbsp;2. Click <b>[>] START</b> for live auto-refresh, or <b>Refresh Fetch Once</b> for a single pull<br>
          &nbsp;&nbsp;3. If blocked, click <b>Key Re-warm NSE Session</b> then retry<br>
          &nbsp;&nbsp;4. If spot = 0, set Manual Spot Override in sidebar<br><br>
          NSE cookie warm-up takes ~5s on first run - this is normal.
        </div>""", unsafe_allow_html=True)
    else:
        exp_used = st.session_state.get("chosen_expiry", "")

        # ── KPI Strip ──────────────────────────────────────
        pcr_cls  = "bull" if rec["pcr"] > 1.0 else "bear"
        iv_cls   = "bear" if rec["vw_iv"] > 28 else "bull" if rec["vw_iv"] < 12 else ""
        pain_cls = "bull" if rec["max_pain"] > spot else "bear"

        kpis  = '<div class="kpi-grid">'
        kpis += render_kpi_html("SPOT PRICE",    f"₹{spot:,.0f}", symbol)
        kpis += render_kpi_html("PUT-CALL RATIO", f"{rec['pcr']:.3f}",
                                "Bullish >1.2" if rec["pcr"] > 1.2 else "Bearish <0.8" if rec["pcr"] < 0.8 else "Neutral",
                                pcr_cls)
        kpis += render_kpi_html("MAX PAIN",       f"₹{rec['max_pain']:,.0f}",
                                f"{'▲' if rec['max_pain'] > spot else '▼'} {abs(rec['max_pain']-spot):.0f}pts",
                                pain_cls)
        kpis += render_kpi_html("CALL WALL",      f"₹{rec['call_wall']:,.0f}", "Resistance", "bear")
        kpis += render_kpi_html("PUT WALL",       f"₹{rec['put_wall']:,.0f}",  "Support",    "bull")
        kpis += render_kpi_html("VW IV",          f"{rec['vw_iv']:.1f}%",
                                "High" if rec["vw_iv"] > 28 else "Low" if rec["vw_iv"] < 11 else "Normal",
                                iv_cls)
        kpis += '</div>'
        st.markdown(kpis, unsafe_allow_html=True)

        # ── Two-column: Rec + Greeks ───────────────────────
        left, right = st.columns([3, 2], gap="medium")

        with left:
            st.markdown("#### [AI] Recommendation")
            render_rec(rec)
            st.markdown("#### [>>] Signal Breakdown")
            render_signals(rec["signals"])

        with right:
            st.markdown("#### [GRK] ATM Greeks")
            render_greeks(rec["ce_g"], rec["pe_g"])
            st.markdown("")
            st.markdown("#### [CHT] Trade Summary")
            summary = {
                "Metric": ["ATM Strike","CE LTP","PE LTP","ATM Call IV","ATM Put IV",
                            "Days to Expiry","Score","Confidence"],
                "Value":  [f"₹{rec['atm']:.0f}",
                           f"₹{rec['ce_ltp']:.2f}", f"₹{rec['pe_ltp']:.2f}",
                           f"{rec['call_iv']:.2f}%", f"{rec['put_iv']:.2f}%",
                           str(rec["days_left"]),
                           f"{rec['score']:+d}/7",   f"{rec['conf']}%"]
            }
            st.dataframe(pd.DataFrame(summary), use_container_width=True,
                         hide_index=True, height=320)

        # ── Option Chain Table ──────────────────────────────
        if show_chain:
            st.markdown("---")
            st.markdown(f"#### [=] Option Chain")
            st.caption(f"±{n_strikes} strikes | ATM ₹{rec['atm']:.0f} | Expiry: {exp_used} | "
                       f"Spot ₹{spot:,.0f}")

            atm     = rec["atm"]
            atm_idx = df[df["Strike"] == atm].index
            if len(atm_idx):
                i  = atm_idx[0]
                df_show = df.iloc[max(0, i - n_strikes): min(len(df), i + n_strikes + 1)].copy()
            else:
                df_show = df.copy()

            keep = ["Strike",
                    "C_OI","C_ChgOI","C_Vol","C_IV","C_LTP","C_Delta","C_Gamma","C_Theta","C_Vega",
                    "P_OI","P_ChgOI","P_Vol","P_IV","P_LTP","P_Delta","P_Gamma","P_Theta","P_Vega"]
            keep      = [c for c in keep if c in df_show.columns]
            df_show   = df_show[keep].reset_index(drop=True)
            int_cols  = [c for c in ["Strike","C_OI","C_ChgOI","C_Vol","P_OI","P_ChgOI","P_Vol"] if c in keep]
            flt_cols  = [c for c in keep if c not in int_cols]
            fmt       = {c: "{:,.0f}" for c in int_cols}
            fmt.update({c: "{:.2f}"  for c in flt_cols})

            styled = (df_show.style.format(fmt, na_rep="-")
                      .apply(lambda row: [
                          "background-color:rgba(0,212,255,0.07);font-weight:bold"
                          if row["Strike"] == atm else "" for _ in row
                      ], axis=1))
            st.dataframe(styled, use_container_width=True, height=460)

            st.download_button(
                "⬇  Download CSV", data=df.to_csv(index=False).encode(),
                file_name=f"{symbol}_OC_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

    # ── Activity Log ──────────────────────────────────────
    st.markdown("---")
    with st.expander("[LOG] Activity Log", expanded=False):
        lines = st.session_state.get("log", [])
        if lines:
            st.markdown(f'<div class="log-box">{"<br>".join(lines[:40])}</div>',
                        unsafe_allow_html=True)
        else:
            st.caption("No activity yet.")

    # ══════════════════════════════════════════════════════
    # AUTO-REFRESH LOOP
    # ══════════════════════════════════════════════════════
    if st.session_state["trading_active"]:
        elapsed = time.time() - st.session_state["last_fetch_time"]
        wait    = max(0.0, refresh_sec - elapsed)

        if wait <= 0:
            do_fetch(
                st.session_state["exchange"],
                st.session_state["symbol"],
                st.session_state["chosen_expiry"],
                r_rate
            )
            st.rerun()
        else:
            st.markdown(
                f'<div style="text-align:right;font-size:0.68rem;color:var(--muted);">'
                f'[T] Next refresh in <b style="color:var(--accent)">{wait:.0f}s</b></div>',
                unsafe_allow_html=True
            )
            time.sleep(min(wait, 1.5))
            st.rerun()


if __name__ == "__main__":
    main()
