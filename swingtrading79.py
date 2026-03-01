"""
Option Chain Live Analyzer — Nifty 50 & Sensex
Streamlit App with Greeks, AI Recommendation, Entry/Target/SL
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import time
import json
from datetime import datetime, date
import warnings
import math

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Option Chain Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS STYLING  (dark industrial terminal theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg: #0a0c10;
    --bg2: #10141c;
    --bg3: #161b26;
    --accent: #00e5ff;
    --accent2: #ff4c6a;
    --accent3: #39ff6a;
    --warn: #ffc107;
    --text: #c9d6e3;
    --muted: #5a6478;
    --border: #1e2535;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.main .block-container { padding: 1rem 2rem; max-width: 100%; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border);
}

/* Cards */
.card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.3rem;
}
.card-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
}
.card-sub { font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem; }

/* Recommendation box */
.rec-box {
    background: linear-gradient(135deg, #0d1117 0%, #161b26 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent3);
    border-radius: 8px;
    padding: 1.4rem;
    margin: 1rem 0;
}
.rec-buy-ce { border-left-color: #39ff6a !important; }
.rec-buy-pe { border-left-color: #ff4c6a !important; }
.rec-hold   { border-left-color: #ffc107 !important; }
.rec-neutral{ border-left-color: #00e5ff !important; }

.rec-badge {
    display: inline-block;
    padding: 0.2rem 0.9rem;
    border-radius: 4px;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.1rem;
    letter-spacing: 2px;
    margin-bottom: 0.7rem;
}
.badge-ce  { background: rgba(57,255,106,0.15); color: #39ff6a; border: 1px solid #39ff6a; }
.badge-pe  { background: rgba(255,76,106,0.15); color: #ff4c6a; border: 1px solid #ff4c6a; }
.badge-hold{ background: rgba(255,193,7,0.15);  color: #ffc107; border: 1px solid #ffc107; }

.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 0.5rem 0; }
.metric-chip {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.3rem 0.7rem;
    font-size: 0.8rem;
}
.metric-chip span { color: var(--accent); font-weight: 600; }

/* Table styling */
.dataframe { background: var(--bg3) !important; font-size: 0.78rem !important; }
thead th { background: var(--bg2) !important; color: var(--accent) !important; }

/* Tags */
.tag { display: inline-block; border-radius: 3px; padding: 1px 6px; font-size: 0.7rem; font-weight: 600; }
.tag-bull { background: rgba(57,255,106,0.2); color: #39ff6a; }
.tag-bear { background: rgba(255,76,106,0.2); color: #ff4c6a; }
.tag-neut { background: rgba(0,229,255,0.15); color: #00e5ff; }

/* Status */
.status-live { color: #39ff6a; font-size: 0.7rem; }
.status-err  { color: #ff4c6a; font-size: 0.7rem; }

/* Divider */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# BLACK-SCHOLES FUNCTIONS
# ─────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, opt='CE'):
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if opt == 'CE' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == 'CE':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(market_px, S, K, T, r, opt='CE'):
    if T <= 0 or market_px <= 0:
        return 0.0
    try:
        intrinsic = max(0, S - K) if opt == 'CE' else max(0, K - S)
        if market_px <= intrinsic:
            return 0.001
        f = lambda sig: bs_price(S, K, T, r, sig, opt) - market_px
        return brentq(f, 0.001, 10.0, maxiter=100)
    except Exception:
        return 0.0


def greeks(S, K, T, r, sigma, opt='CE'):
    empty = dict(delta=0, gamma=0, theta=0, vega=0, rho=0)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return empty
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega  = S * np.sqrt(T) * norm.pdf(d1) / 100
        if opt == 'CE':
            delta = norm.cdf(d1)
            theta = (-(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) -
                     r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            delta = norm.cdf(d1) - 1
            theta = (-(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) +
                     r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        return dict(
            delta=round(delta, 4), gamma=round(gamma, 6),
            theta=round(theta, 4), vega=round(vega,  4), rho=round(rho, 4)
        )
    except Exception:
        return empty


# ─────────────────────────────────────────────
# NSE SCRAPER
# ─────────────────────────────────────────────
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/option-chain",
    "X-Requested-With": "XMLHttpRequest",
    "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Connection": "keep-alive",
}

BSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.bseindia.com/markets/Derivatives/DeriReports/DeriOptionchain.html",
    "Origin": "https://www.bseindia.com",
    "Connection": "keep-alive",
}


@st.cache_resource
def get_nse_session():
    """Create & warm up an NSE session (cached across reruns)."""
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)
        session.get("https://www.nseindia.com/option-chain", timeout=10)
    except Exception:
        pass
    return session


def fetch_nse_option_chain(symbol: str = "NIFTY", retries: int = 3):
    """
    Fetch NSE option chain JSON with retry logic and back-off.
    Returns raw JSON dict or None.
    """
    session = get_nse_session()
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 5 * (attempt + 1)
                st.warning(f"⚠️ NSE rate-limited. Waiting {wait}s… (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            elif resp.status_code in (403, 401):
                # Refresh session cookies
                st.warning("🔄 NSE session expired — refreshing cookies…")
                session.get("https://www.nseindia.com", timeout=10)
                time.sleep(2)
                session.get("https://www.nseindia.com/option-chain", timeout=10)
                time.sleep(2)
            else:
                st.error(f"NSE HTTP {resp.status_code}")
                time.sleep(3)
        except requests.exceptions.ConnectionError:
            st.error("🔌 NSE connection error. Retrying…")
            time.sleep(5 * (attempt + 1))
        except requests.exceptions.Timeout:
            st.error("⏱️ NSE request timed out. Retrying…")
            time.sleep(5)
        except Exception as e:
            st.error(f"NSE unexpected error: {e}")
            time.sleep(3)
    return None


def parse_nse(raw: dict, expiry: str):
    """Parse NSE JSON into a clean DataFrame."""
    try:
        records = raw.get("records", {}).get("data", [])
        ul_price = raw.get("records", {}).get("underlyingValue", 0)
        rows = []
        for rec in records:
            if rec.get("expiryDate", "") != expiry:
                continue
            strike = rec.get("strikePrice", 0)
            ce = rec.get("CE", {})
            pe = rec.get("PE", {})
            rows.append({
                "Strike":      strike,
                # CALL fields
                "C_OI":        ce.get("openInterest", 0),
                "C_ChgOI":     ce.get("changeinOpenInterest", 0),
                "C_Vol":       ce.get("totalTradedVolume", 0),
                "C_IV":        ce.get("impliedVolatility", 0),
                "C_LTP":       ce.get("lastPrice", 0),
                "C_Bid":       ce.get("bidprice", 0),
                "C_Ask":       ce.get("askPrice", 0),
                "C_Chng":      ce.get("change", 0),
                # PUT fields
                "P_OI":        pe.get("openInterest", 0),
                "P_ChgOI":     pe.get("changeinOpenInterest", 0),
                "P_Vol":       pe.get("totalTradedVolume", 0),
                "P_IV":        pe.get("impliedVolatility", 0),
                "P_LTP":       pe.get("lastPrice", 0),
                "P_Bid":       pe.get("bidprice", 0),
                "P_Ask":       pe.get("askPrice", 0),
                "P_Chng":      pe.get("change", 0),
            })
        df = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)
        return df, ul_price
    except Exception as e:
        st.error(f"NSE parse error: {e}")
        return pd.DataFrame(), 0


def fetch_bse_option_chain(expiry: str = "", retries: int = 3):
    """
    Fetch BSE Sensex option chain.
    BSE API endpoint for option chain data.
    """
    # BSE uses a different API; try known endpoint
    url = (
        "https://api.bseindia.com/BseIndiaAPI/api/OptionChain/w?"
        f"scripcd=&Expiry={expiry}&StrikePrice=&optiontype=&seriesid=12"
    )
    session = requests.Session()
    session.headers.update(BSE_HEADERS)
    for attempt in range(retries):
        try:
            # Warm up with main page cookies
            if attempt == 0:
                session.get(
                    "https://www.bseindia.com/markets/Derivatives/DeriReports/DeriOptionchain.html",
                    timeout=12
                )
                time.sleep(1.5)
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                try:
                    return resp.json()
                except Exception:
                    st.error("BSE: Invalid JSON response")
                    return None
            elif resp.status_code == 429:
                wait = 5 * (attempt + 1)
                st.warning(f"⚠️ BSE rate-limited. Waiting {wait}s…")
                time.sleep(wait)
            elif resp.status_code in (403, 401):
                st.warning("🔄 BSE access denied — refreshing session…")
                session.get(
                    "https://www.bseindia.com",
                    timeout=10
                )
                time.sleep(3)
            else:
                time.sleep(3 * (attempt + 1))
        except requests.exceptions.ConnectionError:
            st.error("🔌 BSE connection error.")
            time.sleep(5 * (attempt + 1))
        except requests.exceptions.Timeout:
            st.error("⏱️ BSE request timed out.")
            time.sleep(5)
        except Exception as e:
            st.error(f"BSE error: {e}")
            time.sleep(3)
    return None


def parse_bse(raw: dict):
    """Parse BSE JSON into clean DataFrame."""
    try:
        records = raw.get("Table", raw.get("data", []))
        if not records:
            return pd.DataFrame(), 0
        rows = []
        ul_price = 0
        for rec in records:
            strike = float(rec.get("StrikePrice", rec.get("STRIKE_PRICE", 0)) or 0)
            opt_type = str(rec.get("CPType", rec.get("OPTION_TYPE", ""))).upper()
            ltp   = float(rec.get("LTP", 0) or 0)
            oi    = float(rec.get("OI", 0) or 0)
            chgoi = float(rec.get("ChgOI", 0) or 0)
            vol   = float(rec.get("TotalVolume", rec.get("VOLUME", 0)) or 0)
            iv    = float(rec.get("IV", 0) or 0)
            bid   = float(rec.get("BidPrice", 0) or 0)
            ask   = float(rec.get("AskPrice", 0) or 0)
            rows.append({
                "Strike": strike, "Type": opt_type,
                "OI": oi, "ChgOI": chgoi, "Vol": vol,
                "IV": iv, "LTP": ltp, "Bid": bid, "Ask": ask,
            })
        df_raw = pd.DataFrame(rows)
        if df_raw.empty:
            return pd.DataFrame(), 0
        # Pivot to wide format matching NSE structure
        ce = df_raw[df_raw["Type"] == "CE"].set_index("Strike").add_prefix("C_").drop(columns="C_Type", errors="ignore")
        pe = df_raw[df_raw["Type"] == "PE"].set_index("Strike").add_prefix("P_").drop(columns="P_Type", errors="ignore")
        df = ce.join(pe, how="outer").reset_index()
        df.rename(columns={
            "C_OI": "C_OI", "C_ChgOI": "C_ChgOI", "C_Vol": "C_Vol",
            "C_IV": "C_IV", "C_LTP": "C_LTP", "C_Bid": "C_Bid", "C_Ask": "C_Ask",
            "P_OI": "P_OI", "P_ChgOI": "P_ChgOI", "P_Vol": "P_Vol",
            "P_IV": "P_IV", "P_LTP": "P_LTP", "P_Bid": "P_Bid", "P_Ask": "P_Ask",
        }, inplace=True)
        df = df.fillna(0).sort_values("Strike").reset_index(drop=True)
        return df, ul_price
    except Exception as e:
        st.error(f"BSE parse error: {e}")
        return pd.DataFrame(), 0


# ─────────────────────────────────────────────
# ANALYSIS ENGINE
# ─────────────────────────────────────────────
def days_to_expiry(expiry_str: str):
    """Return fractional years to expiry."""
    try:
        exp = datetime.strptime(expiry_str, "%d-%b-%Y").date()
        delta = (exp - date.today()).days
        return max(delta, 0) / 365.0
    except Exception:
        return 1 / 365.0


def enrich_with_greeks(df: pd.DataFrame, spot: float, T: float, r: float = 0.065):
    """Append Greeks columns to option chain DataFrame."""
    if df.empty or spot <= 0:
        return df

    for opt, prefix in [('CE', 'C_'), ('PE', 'P_')]:
        ltp_col = f"{prefix}LTP"
        iv_col  = f"{prefix}IV"
        if ltp_col not in df.columns:
            continue
        deltas, gammas, thetas, vegas, rhos, ivs_calc = [], [], [], [], [], []
        for _, row in df.iterrows():
            K   = float(row.get("Strike", 0))
            ltp = float(row.get(ltp_col, 0))
            iv_given = float(row.get(iv_col, 0)) / 100 if row.get(iv_col, 0) else 0
            # Prefer market-provided IV, fall back to calculated
            iv = iv_given if iv_given > 0 else implied_vol(ltp, spot, K, T, r, opt)
            g = greeks(spot, K, T, r, iv, opt)
            deltas.append(g["delta"]); gammas.append(g["gamma"])
            thetas.append(g["theta"]); vegas.append(g["vega"])
            rhos.append(g["rho"]);     ivs_calc.append(round(iv * 100, 2))

        df[f"{prefix}Delta"] = deltas
        df[f"{prefix}Gamma"] = gammas
        df[f"{prefix}Theta"] = thetas
        df[f"{prefix}Vega"]  = vegas
        df[f"{prefix}Rho"]   = rhos
        df[f"{prefix}IV_calc"] = ivs_calc
    return df


def max_pain(df: pd.DataFrame):
    """Calculate max-pain strike."""
    if df.empty:
        return 0
    pains = []
    strikes = df["Strike"].tolist()
    for s in strikes:
        pain = 0
        for _, row in df.iterrows():
            k = row["Strike"]
            c_oi = float(row.get("C_OI", 0))
            p_oi = float(row.get("P_OI", 0))
            pain += max(0, s - k) * c_oi   # CE writer pain
            pain += max(0, k - s) * p_oi   # PE writer pain
        pains.append(pain)
    if pains:
        return strikes[int(np.argmin(pains))]
    return 0


def pcr(df: pd.DataFrame):
    """Put-Call Ratio by OI."""
    if df.empty:
        return 0
    total_c = df["C_OI"].sum() if "C_OI" in df.columns else 0
    total_p = df["P_OI"].sum() if "P_OI" in df.columns else 0
    return round(total_p / total_c, 3) if total_c > 0 else 0


def find_atm(df: pd.DataFrame, spot: float):
    """Closest strike to spot."""
    if df.empty or spot <= 0:
        return 0
    diffs = (df["Strike"] - spot).abs()
    return df.loc[diffs.idxmin(), "Strike"]


def iv_skew(df: pd.DataFrame, atm_strike: float, n: int = 3):
    """
    Simple IV skew: compare avg IV of OTM puts vs OTM calls (n strikes away from ATM).
    Positive skew → puts more expensive → bearish sentiment.
    """
    if df.empty:
        return 0, 0
    idx = df[df["Strike"] == atm_strike].index
    if len(idx) == 0:
        return 0, 0
    i = idx[0]
    otm_puts  = df.iloc[max(0, i - n): i]
    otm_calls = df.iloc[i + 1: i + 1 + n]
    avg_put_iv  = otm_puts["P_IV"].replace(0, np.nan).mean() if "P_IV" in df.columns else 0
    avg_call_iv = otm_calls["C_IV"].replace(0, np.nan).mean() if "C_IV" in df.columns else 0
    return round(float(avg_put_iv or 0), 2), round(float(avg_call_iv or 0), 2)


def oi_analysis(df: pd.DataFrame, spot: float):
    """
    Detect support/resistance via max OI in calls (resistance) and puts (support).
    """
    if df.empty:
        return 0, 0, 0, 0
    c_max_idx   = df["C_OI"].idxmax() if "C_OI" in df.columns else 0
    p_max_idx   = df["P_OI"].idxmax() if "P_OI" in df.columns else 0
    call_wall   = df.loc[c_max_idx, "Strike"] if "C_OI" in df.columns else 0
    put_wall    = df.loc[p_max_idx, "Strike"] if "P_OI" in df.columns else 0
    call_wall_oi = df.loc[c_max_idx, "C_OI"]  if "C_OI" in df.columns else 0
    put_wall_oi  = df.loc[p_max_idx, "P_OI"]  if "P_OI" in df.columns else 0
    return call_wall, put_wall, call_wall_oi, put_wall_oi


def vwiv(df: pd.DataFrame):
    """Volume-weighted implied volatility."""
    if df.empty:
        return 0, 0
    c_iv, p_iv = 0.0, 0.0
    c_vol = df["C_Vol"].sum() if "C_Vol" in df.columns else 0
    p_vol = df["P_Vol"].sum() if "P_Vol" in df.columns else 0
    if c_vol > 0 and "C_IV" in df.columns:
        c_iv = round((df["C_IV"] * df["C_Vol"]).sum() / c_vol, 2)
    if p_vol > 0 and "P_IV" in df.columns:
        p_iv = round((df["P_IV"] * df["P_Vol"]).sum() / p_vol, 2)
    return c_iv, p_iv


def atm_greeks(df: pd.DataFrame, atm_strike: float):
    """Return ATM row greeks for CE and PE."""
    row = df[df["Strike"] == atm_strike]
    if row.empty:
        return {}, {}
    r = row.iloc[0]
    ce_g = {k: r.get(f"C_{k}", 0) for k in ["Delta", "Gamma", "Theta", "Vega", "Rho"]}
    pe_g = {k: r.get(f"P_{k}", 0) for k in ["Delta", "Gamma", "Theta", "Vega", "Rho"]}
    return ce_g, pe_g


# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
def generate_recommendation(df, spot, expiry_str, symbol="NIFTY"):
    """
    Multi-factor recommendation engine.
    Returns dict with action, reasoning, entry, target, sl, confidence.
    """
    T = days_to_expiry(expiry_str)
    days_left = round(T * 365)

    # Metrics
    pcr_val      = pcr(df)
    atm          = find_atm(df, spot)
    pain         = max_pain(df)
    put_iv, call_iv = iv_skew(df, atm)
    call_wall, put_wall, cw_oi, pw_oi = oi_analysis(df, spot)
    vw_c_iv, vw_p_iv = vwiv(df)
    ce_g, pe_g   = atm_greeks(df, atm)

    # OI change sums (above/below ATM)
    atm_idx = df[df["Strike"] == atm].index
    if len(atm_idx):
        i = atm_idx[0]
        otm_calls = df.iloc[i:]
        otm_puts  = df.iloc[:i+1]
    else:
        otm_calls = otm_puts = df

    c_chgoi = df["C_ChgOI"].sum() if "C_ChgOI" in df.columns else 0
    p_chgoi = df["P_ChgOI"].sum() if "P_ChgOI" in df.columns else 0

    # ── Scoring System ──────────────────────────────
    score = 0  # positive = bullish, negative = bearish

    # 1. PCR signal
    pcr_signal = ""
    if pcr_val >= 1.4:
        score += 2; pcr_signal = f"PCR {pcr_val} → Extremely bullish (oversold puts)"
    elif pcr_val >= 1.1:
        score += 1; pcr_signal = f"PCR {pcr_val} → Mildly bullish"
    elif pcr_val <= 0.6:
        score -= 2; pcr_signal = f"PCR {pcr_val} → Extremely bearish (oversold calls)"
    elif pcr_val <= 0.9:
        score -= 1; pcr_signal = f"PCR {pcr_val} → Mildly bearish"
    else:
        pcr_signal = f"PCR {pcr_val} → Neutral"

    # 2. Max Pain vs Spot
    pain_diff = pain - spot
    pain_signal = ""
    if pain_diff > spot * 0.005:
        score += 1; pain_signal = f"Max Pain {pain:.0f} is ABOVE spot {spot:.0f} → upward pull likely"
    elif pain_diff < -spot * 0.005:
        score -= 1; pain_signal = f"Max Pain {pain:.0f} is BELOW spot {spot:.0f} → downward pull likely"
    else:
        pain_signal = f"Max Pain {pain:.0f} ≈ Spot {spot:.0f} → near equilibrium"

    # 3. IV Skew
    iv_signal = ""
    iv_diff = put_iv - call_iv
    if iv_diff > 3:
        score -= 1; iv_signal = f"Put IV ({put_iv}%) >> Call IV ({call_iv}%) → bearish skew"
    elif iv_diff < -3:
        score += 1; iv_signal = f"Call IV ({call_iv}%) >> Put IV ({put_iv}%) → bullish skew (rare)"
    else:
        iv_signal = f"IV skew neutral → Put IV {put_iv}%, Call IV {call_iv}%"

    # 4. OI walls (support/resistance)
    wall_signal = ""
    if call_wall > spot and put_wall < spot:
        range_pct = (call_wall - put_wall) / spot * 100
        wall_signal = (f"Call Wall @ {call_wall:.0f} (resistance), Put Wall @ {put_wall:.0f} (support)."
                       f" Trading range: {range_pct:.1f}% wide")
        # How close is spot to walls?
        call_dist = (call_wall - spot) / spot * 100
        put_dist  = (spot - put_wall) / spot * 100
        if put_dist < 0.5:
            score -= 1; wall_signal += " — Spot near PUT wall → support test"
        elif call_dist < 0.5:
            score += 1; wall_signal += " — Spot near CALL wall → resistance test"

    # 5. OI change direction
    oi_chg_signal = ""
    if c_chgoi > 0 and p_chgoi < 0:
        score -= 1; oi_chg_signal = "Fresh CALL OI buildup + PUT OI unwinding → bearish writing activity"
    elif p_chgoi > 0 and c_chgoi < 0:
        score += 1; oi_chg_signal = "Fresh PUT OI buildup + CALL OI unwinding → bullish writing activity"
    elif c_chgoi > 0 and p_chgoi > 0:
        oi_chg_signal = "Both CE & PE OI increasing → market expects volatility (possible breakout)"
    else:
        oi_chg_signal = "OI change direction mixed or flat"

    # 6. ATM Greeks check
    greek_signal = ""
    atm_delta = float(ce_g.get("Delta", 0.5))
    atm_theta = float(ce_g.get("Theta", 0))
    atm_vega  = float(ce_g.get("Vega",  0))
    iv_env = vw_c_iv or vw_p_iv
    if iv_env > 25:
        greek_signal = f"High IV environment ({iv_env:.1f}%) → option premium elevated; favour selling strategies"
    elif iv_env < 12:
        greek_signal = f"Low IV environment ({iv_env:.1f}%) → cheap options; favour buying strategies"
    else:
        greek_signal = f"Moderate IV ({iv_env:.1f}%) → directional buying viable"

    # 7. Time decay
    td_signal = ""
    if days_left <= 3:
        td_signal = f"⚠️ Only {days_left} days to expiry — Theta decay is aggressive. Avoid buying unless breakout confirmed."
    elif days_left <= 7:
        td_signal = f"{days_left} days left — Time decay accelerating. Prefer scalps over positional."
    else:
        td_signal = f"{days_left} days to expiry — Adequate time for positional trades."

    # ── Decision ──────────────────────────────────
    atm_ltp_ce = float(df[df["Strike"] == atm]["C_LTP"].values[0]) if len(df[df["Strike"] == atm]) else 0
    atm_ltp_pe = float(df[df["Strike"] == atm]["P_LTP"].values[0]) if len(df[df["Strike"] == atm]) else 0

    confidence = min(abs(score) / 5 * 100, 95)
    
    # Typical option premium buffer
    ce_sl_pct = 0.40   # 40% of premium as SL
    pe_sl_pct = 0.40
    ce_tgt_1  = 1.5    # 1.5× premium
    ce_tgt_2  = 2.2    # 2.2× premium

    if score >= 2:
        action = "BUY CE"
        entry  = round(atm_ltp_ce or (spot * 0.005), 2)
        sl     = round(entry * (1 - ce_sl_pct), 2)
        tgt1   = round(entry * ce_tgt_1, 2)
        tgt2   = round(entry * ce_tgt_2, 2)
        strike_rec = atm
        badge  = "badge-ce"
        rec_cls = "rec-buy-ce"
        summary = (
            f"📈 **BULLISH SETUP DETECTED** on {symbol} @ {spot:.0f}\n\n"
            f"Multiple indicators align for an upward move. "
            f"The Put-Call Ratio at **{pcr_val}** indicates elevated put writing (bullish), "
            f"Max Pain at **{pain:.0f}** is pulling price upward, "
            f"and OI analysis shows strong put support at **{put_wall:.0f}**. "
            f"Consider buying **{atm:.0f} CE** for the current expiry."
        )
    elif score <= -2:
        action = "BUY PE"
        entry  = round(atm_ltp_pe or (spot * 0.005), 2)
        sl     = round(entry * (1 - pe_sl_pct), 2)
        tgt1   = round(entry * ce_tgt_1, 2)
        tgt2   = round(entry * ce_tgt_2, 2)
        strike_rec = atm
        badge  = "badge-pe"
        rec_cls = "rec-buy-pe"
        summary = (
            f"📉 **BEARISH SETUP DETECTED** on {symbol} @ {spot:.0f}\n\n"
            f"Multiple indicators point to downside risk. "
            f"PCR at **{pcr_val}** shows excessive call accumulation, "
            f"Max Pain at **{pain:.0f}** exerts downward pressure, "
            f"and call writers are defending **{call_wall:.0f}** aggressively. "
            f"Consider buying **{atm:.0f} PE** for the current expiry."
        )
    elif score == 1:
        action = "MILD BUY CE"
        entry  = round(atm_ltp_ce or (spot * 0.005), 2)
        sl     = round(entry * 0.50, 2)
        tgt1   = round(entry * 1.4, 2)
        tgt2   = round(entry * 1.8, 2)
        strike_rec = atm
        badge  = "badge-ce"
        rec_cls = "rec-buy-ce"
        summary = (
            f"↗️ **MILDLY BULLISH** on {symbol} @ {spot:.0f}\n\n"
            f"Slight positive bias with limited conviction. "
            f"PCR at {pcr_val} leans bullish but confirmation is needed. "
            f"Keep position size small and wait for spot to sustain above {atm:.0f}."
        )
    elif score == -1:
        action = "MILD BUY PE"
        entry  = round(atm_ltp_pe or (spot * 0.005), 2)
        sl     = round(entry * 0.50, 2)
        tgt1   = round(entry * 1.4, 2)
        tgt2   = round(entry * 1.8, 2)
        strike_rec = atm
        badge  = "badge-pe"
        rec_cls = "rec-buy-pe"
        summary = (
            f"↘️ **MILDLY BEARISH** on {symbol} @ {spot:.0f}\n\n"
            f"Slight negative tilt but not a high-conviction setup. "
            f"PCR at {pcr_val} and call-side OI buildup suggest caution. "
            f"Trail with tight stops."
        )
    else:
        action = "HOLD / WAIT"
        entry  = atm_ltp_ce or atm_ltp_pe
        sl     = 0
        tgt1   = 0
        tgt2   = 0
        strike_rec = atm
        badge  = "badge-hold"
        rec_cls = "rec-hold"
        summary = (
            f"⏳ **NEUTRAL — WAIT FOR CLARITY** on {symbol} @ {spot:.0f}\n\n"
            f"Indicators are mixed or offsetting each other. "
            f"PCR at {pcr_val} and OI data do not show a clear directional bias. "
            f"Best to wait for a decisive breakout above {call_wall:.0f} "
            f"or breakdown below {put_wall:.0f} before entering."
        )

    return {
        "action":       action,
        "score":        score,
        "confidence":   round(confidence, 1),
        "summary":      summary,
        "entry":        entry,
        "sl":           sl,
        "target1":      tgt1,
        "target2":      tgt2,
        "strike":       strike_rec,
        "badge":        badge,
        "rec_cls":      rec_cls,
        # Signal breakdown
        "signals": {
            "PCR":          pcr_signal,
            "Max Pain":     pain_signal,
            "IV Skew":      iv_signal,
            "OI Walls":     wall_signal,
            "OI Change":    oi_chg_signal,
            "IV Regime":    greek_signal,
            "Time Decay":   td_signal,
        },
        # Metrics
        "pcr":          pcr_val,
        "max_pain":     pain,
        "atm":          atm,
        "call_wall":    call_wall,
        "put_wall":     put_wall,
        "vw_iv":        max(vw_c_iv, vw_p_iv),
        "put_iv":       put_iv,
        "call_iv":      call_iv,
        "days_left":    days_left,
        "atm_ce_ltp":   atm_ltp_ce,
        "atm_pe_ltp":   atm_ltp_pe,
        "ce_greeks":    ce_g,
        "pe_greeks":    pe_g,
    }


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
def render_metric_card(title, value, sub="", color="var(--accent)"):
    st.markdown(f"""
    <div class="card">
        <div class="card-title">{title}</div>
        <div class="card-value" style="color:{color}">{value}</div>
        <div class="card-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def render_recommendation(rec: dict, symbol: str):
    action  = rec["action"]
    badge   = rec["badge"]
    rec_cls = rec["rec_cls"]
    conf    = rec["confidence"]
    entry   = rec["entry"]
    sl      = rec["sl"]
    t1      = rec["target1"]
    t2      = rec["target2"]
    strike  = rec["strike"]

    st.markdown(f"""
    <div class="rec-box {rec_cls}">
        <span class="rec-badge {badge}">{action}</span>
        <div style="font-size:0.85rem; line-height:1.7; color:#c9d6e3; margin-bottom:0.8rem;">
            {rec['summary'].replace(chr(10), '<br>')}
        </div>
        <div class="metric-row">
            <div class="metric-chip">Strike <span>{int(strike)}</span></div>
            <div class="metric-chip">Entry <span>₹{entry}</span></div>
            <div class="metric-chip">SL <span>₹{sl}</span></div>
            <div class="metric-chip">Target 1 <span>₹{t1}</span></div>
            <div class="metric-chip">Target 2 <span>₹{t2}</span></div>
            <div class="metric-chip">Confidence <span>{conf}%</span></div>
        </div>
        <div style="font-size:0.72rem;color:var(--muted);margin-top:0.5rem;">
            ⚠️ This is algorithmic analysis only — not SEBI registered advice. Always use your own judgement and consult a financial advisor.
        </div>
    </div>""", unsafe_allow_html=True)


def render_signals(signals: dict):
    st.markdown("**🔍 Signal Breakdown**")
    icons = {
        "PCR": "📊", "Max Pain": "🎯", "IV Skew": "📐",
        "OI Walls": "🧱", "OI Change": "🔄", "IV Regime": "🌡️", "Time Decay": "⏱️"
    }
    for k, v in signals.items():
        if v:
            bull = any(w in v.lower() for w in ["bullish", "upward", "support", "above"])
            bear = any(w in v.lower() for w in ["bearish", "downward", "resistance", "below", "warning"])
            tag  = '<span class="tag tag-bull">BULL</span>' if bull else \
                   '<span class="tag tag-bear">BEAR</span>' if bear else \
                   '<span class="tag tag-neut">NEUT</span>'
            st.markdown(
                f"<div style='padding:4px 0;font-size:0.8rem;'>{icons.get(k,'●')} <b>{k}</b>: {v} {tag}</div>",
                unsafe_allow_html=True
            )


def render_atm_greeks(ce_g, pe_g):
    cols = st.columns(2)
    for col, label, g, color in [
        (cols[0], "ATM CALL Greeks", ce_g, "#39ff6a"),
        (cols[1], "ATM PUT Greeks",  pe_g, "#ff4c6a"),
    ]:
        with col:
            st.markdown(f"<div style='color:{color};font-size:0.75rem;font-weight:700;letter-spacing:2px;margin-bottom:6px;'>{label}</div>", unsafe_allow_html=True)
            for gname in ["Delta", "Gamma", "Theta", "Vega", "Rho"]:
                val = g.get(gname, "—")
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;font-size:0.8rem;padding:3px 0;border-bottom:1px solid #1e2535;'>"
                    f"<span style='color:#5a6478;'>{gname}</span><span style='color:{color};'>{val}</span></div>",
                    unsafe_allow_html=True
                )


def style_option_chain(df: pd.DataFrame, spot: float):
    """Highlight ATM row and color the DataFrame."""
    display_cols = ["Strike", "C_OI", "C_ChgOI", "C_Vol", "C_IV", "C_LTP",
                    "C_Delta", "C_Theta", "C_Vega",
                    "P_OI", "P_ChgOI", "P_Vol", "P_IV", "P_LTP",
                    "P_Delta", "P_Theta", "P_Vega"]
    existing = [c for c in display_cols if c in df.columns]
    dff = df[existing].copy()

    def highlight_atm(row):
        atm_dist = abs(row["Strike"] - spot)
        min_dist = (df["Strike"] - spot).abs().min()
        if atm_dist == min_dist:
            return ["background-color: rgba(0,229,255,0.08); font-weight:bold;"] * len(row)
        return [""] * len(row)

    styled = dff.style.apply(highlight_atm, axis=1)
    styled = styled.format({
        c: "{:,.0f}" for c in ["Strike", "C_OI", "C_ChgOI", "C_Vol", "P_OI", "P_ChgOI", "P_Vol"]
        if c in dff.columns
    }, na_rep="—")
    styled = styled.format({
        c: "{:.2f}" for c in ["C_IV", "C_LTP", "C_Delta", "C_Theta", "C_Vega",
                               "P_IV", "P_LTP", "P_Delta", "P_Theta", "P_Vega"]
        if c in dff.columns
    }, na_rep="—")
    return styled


# ─────────────────────────────────────────────
# MAIN STREAMLIT APP
# ─────────────────────────────────────────────
def main():
    # ── Header ──────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.5rem;">
        <div>
            <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
                        background:linear-gradient(90deg,#00e5ff,#39ff6a);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                OPTION CHAIN ANALYZER
            </div>
            <div style="font-size:0.7rem;color:#5a6478;letter-spacing:3px;text-transform:uppercase;">
                Nifty 50 & Sensex · Live Greeks · AI Recommendation
            </div>
        </div>
    </div>
    <hr style="margin:0.5rem 0 1rem 0;">
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        exchange = st.selectbox("Exchange", ["NSE (Nifty)", "BSE (Sensex)"])
        is_nse   = "NSE" in exchange

        if is_nse:
            symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"])
        else:
            symbol = "SENSEX"

        # Fetch expiries dynamically below; default placeholder
        expiry_placeholder = st.empty()

        r_rate = st.slider("Risk-Free Rate (%)", 4.0, 8.0, 6.5, 0.25) / 100

        refresh_interval = st.selectbox(
            "Auto-Refresh (seconds)", [0, 10, 30, 60, 120], index=0,
            format_func=lambda x: "Manual" if x == 0 else f"Every {x}s"
        )

        manual_spot = st.number_input(
            "Override Spot Price (0 = auto)", min_value=0.0, value=0.0, step=50.0
        )

        show_full_chain = st.checkbox("Show Full Option Chain Table", value=True)
        n_strikes       = st.slider("Strikes to Display (±ATM)", 5, 30, 12)

        st.markdown("---")
        fetch_btn = st.button("🔄 Fetch / Refresh Now", use_container_width=True)
        st.markdown("""
        <div style="font-size:0.65rem;color:#5a6478;margin-top:1rem;">
        Data: NSE India API / BSE India API<br>
        Delay: ≥2s between requests<br>
        Greeks: Black-Scholes model<br><br>
        <b>⚠️ Not SEBI registered. For educational use only.</b>
        </div>""", unsafe_allow_html=True)

    # ── State management ─────────────────────────
    if "last_fetch" not in st.session_state:
        st.session_state["last_fetch"] = 0
    if "df_cache" not in st.session_state:
        st.session_state["df_cache"] = None
    if "spot_cache" not in st.session_state:
        st.session_state["spot_cache"] = 0
    if "expiries" not in st.session_state:
        st.session_state["expiries"] = []
    if "raw_cache" not in st.session_state:
        st.session_state["raw_cache"] = None

    now = time.time()
    should_fetch = (
        fetch_btn or
        st.session_state["df_cache"] is None or
        (refresh_interval > 0 and (now - st.session_state["last_fetch"]) >= refresh_interval)
    )

    # ── Fetch ──────────────────────────────────────
    status_ph = st.empty()
    if should_fetch:
        # Enforce 2-second minimum between requests
        elapsed = now - st.session_state["last_fetch"]
        if elapsed < 2:
            time.sleep(2 - elapsed)

        with st.spinner("Fetching option chain data…"):
            if is_nse:
                raw = fetch_nse_option_chain(symbol)
                if raw:
                    # Build expiry list
                    expiries_raw = raw.get("records", {}).get("expiryDates", [])
                    st.session_state["expiries"] = expiries_raw
                    st.session_state["raw_cache"] = raw
                    st.session_state["last_fetch"] = time.time()
            else:
                # For BSE we need expiry first; try with empty expiry to get list
                raw = fetch_bse_option_chain("")
                if raw:
                    st.session_state["raw_cache"] = raw
                    st.session_state["last_fetch"] = time.time()

    # ── Expiry Selector ─────────────────────────────
    expiries = st.session_state.get("expiries", [])
    if expiries:
        chosen_expiry = expiry_placeholder.selectbox("Expiry", expiries, key="expiry_sel")
    else:
        chosen_expiry = expiry_placeholder.text_input("Expiry (DD-Mon-YYYY)", value="", key="expiry_inp")

    # ── Parse & Enrich ─────────────────────────────
    raw = st.session_state.get("raw_cache")
    df, spot = pd.DataFrame(), 0.0

    if raw:
        if is_nse and chosen_expiry:
            df, spot = parse_nse(raw, chosen_expiry)
        elif not is_nse:
            df, spot = parse_bse(raw)

    if manual_spot > 0:
        spot = manual_spot

    if not df.empty and chosen_expiry and spot > 0:
        T = days_to_expiry(chosen_expiry) if is_nse else 7/365
        df = enrich_with_greeks(df, spot, T, r_rate)

        # Filter to n_strikes around ATM
        atm = find_atm(df, spot)
        atm_idx = df[df["Strike"] == atm].index
        if len(atm_idx):
            i = atm_idx[0]
            lo = max(0, i - n_strikes)
            hi = min(len(df), i + n_strikes + 1)
            df_display = df.iloc[lo:hi].copy()
        else:
            df_display = df.copy()

        rec = generate_recommendation(df, spot, chosen_expiry if is_nse else "", symbol)

        # ── Top Metrics Row ──────────────────────────
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1: render_metric_card("SPOT", f"₹{spot:,.0f}", symbol)
        with m2:
            pcr_col = "#39ff6a" if rec["pcr"] >= 1.0 else "#ff4c6a"
            render_metric_card("PCR", f"{rec['pcr']:.3f}", "Put-Call Ratio", pcr_col)
        with m3: render_metric_card("MAX PAIN", f"₹{rec['max_pain']:,.0f}", "Max Pain Strike")
        with m4: render_metric_card("CALL WALL", f"₹{rec['call_wall']:,.0f}", "Resistance (Max Call OI)")
        with m5: render_metric_card("PUT WALL", f"₹{rec['put_wall']:,.0f}", "Support (Max Put OI)")
        with m6:
            iv_col = "#ff4c6a" if rec["vw_iv"] > 25 else "#39ff6a" if rec["vw_iv"] < 12 else "#ffc107"
            render_metric_card("VW IV", f"{rec['vw_iv']:.1f}%", "Vol-Weighted IV", iv_col)

        st.markdown("")

        # ── Main Layout: Recommendation + Greeks ──────
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("### 🤖 Recommendation")
            render_recommendation(rec, symbol)

            st.markdown("### 🔍 Signal Analysis")
            render_signals(rec["signals"])

        with col_right:
            st.markdown("### 📐 ATM Greeks")
            render_atm_greeks(rec["ce_greeks"], rec["pe_greeks"])

            st.markdown("")
            st.markdown("### 📊 IV Summary")
            iv_data = {
                "Metric": ["ATM Call IV", "ATM Put IV", "VW Call IV", "VW Put IV",
                            "CE ATM Vega", "PE ATM Vega"],
                "Value":  [
                    f"{rec['call_iv']:.2f}%", f"{rec['put_iv']:.2f}%",
                    f"{vwiv(df)[0]:.2f}%", f"{vwiv(df)[1]:.2f}%",
                    str(rec['ce_greeks'].get('Vega', '—')),
                    str(rec['pe_greeks'].get('Vega', '—')),
                ]
            }
            st.dataframe(pd.DataFrame(iv_data), use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Option Chain Table ───────────────────────
        if show_full_chain:
            st.markdown("### 📋 Option Chain (with Greeks)")
            st.caption(f"Showing ±{n_strikes} strikes around ATM {atm:.0f} | Spot: {spot:.0f} | Expiry: {chosen_expiry}")
            try:
                styled = style_option_chain(df_display, spot)
                st.dataframe(styled, use_container_width=True, height=420)
            except Exception:
                st.dataframe(df_display, use_container_width=True, height=420)

            # Download button
            csv = df.to_csv(index=False).encode()
            st.download_button(
                "⬇️  Download Full Chain CSV",
                data=csv,
                file_name=f"{symbol}_option_chain_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

        # Status bar
        ts = datetime.fromtimestamp(st.session_state["last_fetch"]).strftime("%H:%M:%S")
        st.markdown(
            f'<div class="status-live">✅ Last fetched: {ts} | Next in: '
            f'{"Manual" if refresh_interval == 0 else str(refresh_interval)+"s"}</div>',
            unsafe_allow_html=True
        )

    elif raw is None and not should_fetch:
        st.info("👆 Click **Fetch / Refresh Now** in the sidebar to load option chain data.")
    elif raw is not None and df.empty:
        st.warning(
            "⚠️ Option chain data fetched but could not be parsed.\n\n"
            "Possible reasons:\n"
            "- The selected expiry has no data\n"
            "- NSE/BSE changed their API structure\n"
            "- Market is closed\n\n"
            "Try a different expiry or check your network connection."
        )
    elif spot == 0 and not df.empty:
        st.warning("⚠️ Spot price could not be detected. Please set it manually in the sidebar.")

    # ── Auto-refresh ─────────────────────────────
    if refresh_interval > 0:
        time.sleep(0.5)
        st.rerun()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
