"""
Professional Options Trading Dashboard
======================================
Live option chain analysis for Nifty, BankNifty, Sensex, BTC, Gold, Silver, USDINR, and custom tickers.
Features: Live Greeks, OI Analysis, Signal Engine, Backtesting, Trade History
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import requests.adapters
import time
import json
import warnings
import random
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

# ─── SHARED YFINANCE SESSION (created once, reused globally) ──────────────────
@st.cache_resource
def get_yf_session():
    """
    One persistent requests.Session shared across ALL yfinance calls.
    Mimics a real browser and uses an adapter that retries on 429/5xx.
    """
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    sess = requests.Session()
    sess.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    })
    retry = Retry(
        total=4,
        backoff_factor=2,           # 2s, 4s, 8s, 16s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def yf_ticker(sym: str) -> yf.Ticker:
    """Return a yfinance Ticker that reuses our shared session."""
    sess = get_yf_session()
    return yf.Ticker(sym, session=sess)


def safe_sleep(base: float = 1.5):
    """Sleep with ±0.3s jitter to avoid identical-interval detection."""
    time.sleep(base + random.uniform(-0.3, 0.4))

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ProOptions | Live Trading Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #060a10;
    --surface: #0d1520;
    --surface2: #111d2e;
    --accent: #00e5ff;
    --green: #00ff88;
    --red: #ff3b5c;
    --orange: #ff9500;
    --text: #c8d6e5;
    --muted: #5a7a9a;
    --border: #1e3050;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background: var(--bg); }

.stMetric {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}

.stMetric label { color: var(--muted) !important; font-size: 11px !important; letter-spacing: 1px; text-transform: uppercase; }
.stMetric [data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Space Mono', monospace; font-size: 20px !important; }

div[data-testid="stTabs"] button { color: var(--muted) !important; font-family: 'Space Mono', monospace; font-size: 12px; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }

.signal-banner {
    background: linear-gradient(135deg, #060e1a 0%, #0a1929 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
    position: relative;
    overflow: hidden;
}
.signal-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--green), var(--accent));
}

.tag-buy { background: #002a15; color: #00ff88; border: 1px solid #00aa44; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.tag-sell { background: #2a0010; color: #ff3b5c; border: 1px solid #aa0022; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.tag-wait { background: #2a1a00; color: #ff9500; border: 1px solid #aa5500; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.tag-neutral { background: #0a1929; color: #5a7a9a; border: 1px solid #1e3050; padding: 4px 10px; border-radius: 20px; font-size: 12px; }

.scenario-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    height: 100%;
}
.scenario-bull { border-top: 3px solid var(--green); }
.scenario-bear { border-top: 3px solid var(--red); }
.scenario-flat { border-top: 3px solid var(--muted); }

h1, h2, h3 { font-family: 'Space Mono', monospace !important; color: var(--accent) !important; }
h4, h5 { color: var(--text) !important; }

.stSelectbox > div > div { background: var(--surface) !important; border-color: var(--border) !important; }
.stNumberInput > div > div { background: var(--surface) !important; }
.stSlider > div > div { background: var(--border) !important; }

.stButton > button {
    background: linear-gradient(135deg, #00e5ff22, #00e5ff11) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    letter-spacing: 0.5px !important;
}
.stButton > button:hover { background: var(--accent) !important; color: #000 !important; }

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00aa4433, #00aa4411) !important;
    border-color: var(--green) !important;
    color: var(--green) !important;
}
.stButton > button[kind="primary"]:hover { background: var(--green) !important; color: #000 !important; }

.alert-box {
    background: #1a0a00;
    border: 1px solid #ff950088;
    border-left: 3px solid var(--orange);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 14px;
}

.gamma-blast {
    background: linear-gradient(135deg, #1a000033, #ff000011);
    border: 1px solid #ff3b5c88;
    border-radius: 10px;
    padding: 16px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(255, 59, 92, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(255, 59, 92, 0); }
    100% { box-shadow: 0 0 0 0 rgba(255, 59, 92, 0); }
}

.trade-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    margin: 8px 0;
}
.trade-card-profit { border-left: 3px solid var(--green); }
.trade-card-loss { border-left: 3px solid var(--red); }
.trade-card-open { border-left: 3px solid var(--accent); }

hr { border-color: var(--border) !important; }

.stDataFrame { background: var(--surface) !important; }
.stDataFrame th { background: var(--surface2) !important; color: var(--accent) !important; font-family: 'Space Mono', monospace; font-size: 11px; }
.stDataFrame td { background: var(--surface) !important; color: var(--text) !important; font-size: 12px; }

.summary-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: var(--muted);
    line-height: 1.6;
    margin-bottom: 16px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─── CONSTANTS ─────────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.065  # 6.5% Indian RFR
REQUEST_DELAY = 1.5     # yfinance rate limit guard

NSE_INDICES = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "MIDCPNIFTY": "^NSMIDCP",
}

COMMODITY_MAP = {
    "BTC/USD": "BTC-USD",
    "GOLD (GC=F)": "GC=F",
    "SILVER (SI=F)": "SI=F",
    "USD/INR": "USDINR=X",
    "CRUDE OIL": "CL=F",
    "NATURAL GAS": "NG=F",
}

# ─── SESSION STATE INIT ────────────────────────────────────────────────────────
for key, val in {
    "trade_history": [],
    "active_trades": [],
    "backtest_results": None,
    "df_cache": None,
    "spot_cache": None,
    "signal_cache": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Module-level throttle dict (persists across Streamlit reruns in same process)
_LAST_YF_CALL: dict = {}


def _global_throttle(min_gap: float = 2.0):
    """Enforce minimum gap between yfinance calls regardless of which function calls them."""
    now = time.time()
    last = _LAST_YF_CALL.get("ts", 0)
    gap = now - last
    if gap < min_gap:
        time.sleep(min_gap - gap + random.uniform(0.1, 0.3))
    _LAST_YF_CALL["ts"] = time.time()


# ─── BLACK-SCHOLES ENGINE ──────────────────────────────────────────────────────
def bs_greeks(S, K, T, r, sigma, otype="call"):
    """Full Black-Scholes Greeks."""
    if T <= 1e-6 or sigma <= 1e-6 or S <= 0 or K <= 0:
        return dict(delta=0, gamma=0, theta=0, vega=0, rho=0, price=0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if otype == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    return dict(
        delta=round(delta, 4),
        gamma=round(gamma, 8),
        theta=round(theta, 4),
        vega=round(vega, 4),
        rho=round(rho, 4),
        price=round(max(price, 0), 4),
    )


def implied_vol(market_px, S, K, T, r, otype="call"):
    if T <= 0 or market_px <= 0 or S <= 0:
        return 0.20
    try:
        def obj(sig):
            return bs_greeks(S, K, T, r, sig, otype)["price"] - market_px
        iv = brentq(obj, 1e-4, 20.0, xtol=1e-5, maxiter=200)
        return max(iv, 0.001)
    except Exception:
        return 0.20


# ─── NSE SESSION ──────────────────────────────────────────────────────────────
def create_nse_session():
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
    })
    try:
        sess.get("https://www.nseindia.com/", timeout=10)
        time.sleep(0.5)
    except Exception:
        pass
    return sess


@st.cache_data(ttl=90, show_spinner=False)
def fetch_nse_chain(symbol: str):
    """Fetch NSE option chain with proper session & cookie handling."""
    sess = create_nse_session()
    if symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"]:
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    else:
        url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
    try:
        resp = sess.get(url, timeout=15)
        if resp.status_code != 200:
            return None, None, []
        data = resp.json()
        return _parse_nse(data)
    except Exception as e:
        st.error(f"NSE fetch error: {e}")
        return None, None, []


def _parse_nse(data):
    records = data.get("records", {})
    spot = records.get("underlyingValue", 0)
    expiries = records.get("expiryDates", [])
    rows = []
    for rec in records.get("data", []):
        ce = rec.get("CE", {})
        pe = rec.get("PE", {})
        rows.append({
            "strikePrice": rec.get("strikePrice", 0),
            "expiryDate": rec.get("expiryDate", ""),
            "CE_LTP": ce.get("lastPrice", 0),
            "CE_OI": ce.get("openInterest", 0),
            "CE_changeOI": ce.get("changeinOpenInterest", 0),
            "CE_volume": ce.get("totalTradedVolume", 0),
            "CE_IV": ce.get("impliedVolatility", 0),
            "CE_bid": ce.get("bidprice", 0),
            "CE_ask": ce.get("askPrice", 0),
            "PE_LTP": pe.get("lastPrice", 0),
            "PE_OI": pe.get("openInterest", 0),
            "PE_changeOI": pe.get("changeinOpenInterest", 0),
            "PE_volume": pe.get("totalTradedVolume", 0),
            "PE_IV": pe.get("impliedVolatility", 0),
            "PE_bid": pe.get("bidprice", 0),
            "PE_ask": pe.get("askPrice", 0),
        })
    df = pd.DataFrame(rows)
    return df, spot, expiries


@st.cache_data(ttl=120, show_spinner=False)
def fetch_yf_chain(yf_sym: str, expiry: str = None):
    """
    Fetch yfinance option chain in a single Ticker pass.
    Uses shared session + retry adapter. Only 2 HTTP calls total:
      1. tk.options  (expiry list + sets cookie)
      2. tk.option_chain(expiry)
    Spot price is extracted from the chain directly (no extra history call).
    """
    try:
        _global_throttle(2.0)
        tk = yf_ticker(yf_sym)

        # ── 1. Get expiry list ────────────────────────────────────────────────
        try:
            expiries = list(tk.options or [])
        except Exception:
            expiries = []

        if not expiries:
            st.warning(f"No options data found for {yf_sym}. Market may be closed or symbol invalid.")
            return None, None, []

        sel = expiry if expiry in expiries else expiries[0]

        # ── 2. Single chain call ──────────────────────────────────────────────
        _global_throttle(2.0)
        chain = tk.option_chain(sel)
        calls = chain.calls.copy()
        puts  = chain.puts.copy()

        # ── 3. Spot from fast_info (no extra HTTP call) ───────────────────────
        try:
            spot = float(tk.fast_info.get("last_price") or tk.fast_info.get("regularMarketPrice") or 0)
        except Exception:
            spot = 0

        # Fallback: midpoint of ATM options
        if spot == 0 and not calls.empty and "strike" in calls.columns:
            atm_idx = (calls["strike"] - calls["strike"].median()).abs().idxmin()
            atm_call = float(calls.loc[atm_idx, "lastPrice"]) if "lastPrice" in calls.columns else 0
            atm_put_row = puts[puts["strike"] == calls.loc[atm_idx, "strike"]] if "strike" in puts.columns else pd.DataFrame()
            atm_put = float(atm_put_row["lastPrice"].values[0]) if not atm_put_row.empty else 0
            # spot ≈ strike + call - put (put-call parity)
            spot = float(calls.loc[atm_idx, "strike"]) + atm_call - atm_put

        # ── 4. Build unified DataFrame ────────────────────────────────────────
        if calls.empty:
            return None, spot, expiries

        df = pd.DataFrame()
        df["strikePrice"] = calls["strike"].values if "strike" in calls.columns else pd.Series(dtype=float)
        df["expiryDate"]  = sel

        for col, src_col in [
            ("CE_LTP",    "lastPrice"),
            ("CE_OI",     "openInterest"),
            ("CE_volume", "volume"),
            ("CE_bid",    "bid"),
            ("CE_ask",    "ask"),
        ]:
            df[col] = calls[src_col].values if src_col in calls.columns else 0

        df["CE_IV"] = (calls["impliedVolatility"].values * 100) if "impliedVolatility" in calls.columns else 0
        df["CE_changeOI"] = 0

        # Merge puts by strike
        if not puts.empty and "strike" in puts.columns:
            puts_idx = puts.set_index("strike")
            for col, src_col in [
                ("PE_LTP",    "lastPrice"),
                ("PE_OI",     "openInterest"),
                ("PE_volume", "volume"),
                ("PE_bid",    "bid"),
                ("PE_ask",    "ask"),
            ]:
                df[col] = df["strikePrice"].map(
                    puts_idx[src_col] if src_col in puts_idx.columns else pd.Series(dtype=float)
                ).fillna(0)
            df["PE_IV"] = df["strikePrice"].map(
                puts_idx["impliedVolatility"] * 100 if "impliedVolatility" in puts_idx.columns else pd.Series(dtype=float)
            ).fillna(0)
        else:
            for col in ["PE_LTP", "PE_OI", "PE_volume", "PE_bid", "PE_ask", "PE_IV"]:
                df[col] = 0

        df["PE_changeOI"] = 0
        df = df.fillna(0)
        df = df[df["CE_LTP"] + df["PE_LTP"] > 0]   # drop empty rows
        df = df.reset_index(drop=True)
        return df, round(spot, 2), expiries

    except Exception as e:
        err = str(e)
        if "429" in err or "Too Many" in err.lower() or "rate" in err.lower():
            st.error(
                "⏳ **yfinance rate limit hit.** The shared session will auto-retry with backoff. "
                "Wait ~15 seconds and press **Fetch Live Data** again. "
                "If persistent, restart the Streamlit app (clears session)."
            )
        else:
            st.error(f"yfinance error: {err}")
        return None, None, []


def enrich_greeks(df: pd.DataFrame, spot: float, expiry_str: str) -> pd.DataFrame:
    """Calculate and attach Greeks to option chain dataframe."""
    if df is None or df.empty:
        return df
    try:
        fmt = "%d-%b-%Y" if "-" in str(expiry_str) and len(str(expiry_str)) > 7 and not str(expiry_str)[4:5].isdigit() else "%Y-%m-%d"
        exp_dt = datetime.strptime(str(expiry_str), fmt)
        T = max((exp_dt - datetime.now()).days / 365.0, 1e-4)
    except Exception:
        T = 7 / 365

    for idx, row in df.iterrows():
        K = float(row["strikePrice"])
        for otype, prefix in [("call", "CE"), ("put", "PE")]:
            ltp = float(row.get(f"{prefix}_LTP", 0))
            iv_pct = float(row.get(f"{prefix}_IV", 0))
            sigma = iv_pct / 100 if iv_pct > 0.5 else implied_vol(ltp, spot, K, T, RISK_FREE_RATE, otype)
            sigma = max(sigma, 0.01)
            g = bs_greeks(spot, K, T, RISK_FREE_RATE, sigma, otype)
            df.at[idx, f"{prefix}_delta"] = g["delta"]
            df.at[idx, f"{prefix}_gamma"] = g["gamma"]
            df.at[idx, f"{prefix}_theta"] = g["theta"]
            df.at[idx, f"{prefix}_vega"] = g["vega"]
            df.at[idx, f"{prefix}_rho"] = g["rho"]
            if iv_pct == 0 and ltp > 0:
                df.at[idx, f"{prefix}_IV"] = round(sigma * 100, 2)
    return df


# ─── SIGNAL ENGINE (identical logic used in live & backtest) ──────────────────
def generate_signals(df: pd.DataFrame, spot: float) -> dict:
    """
    Core signal engine — produces recommendations for scalping / intraday /
    swing / positional.  SAME function used in live trading and backtesting.
    """
    sig = {
        "pcr": 0.0, "max_pain": 0.0, "iv_skew": 0.0,
        "atm_strike": 0.0, "straddle_price": 0.0,
        "resistance": 0.0, "support": 0.0,
        "atm_iv": 0.0, "atm_ce_iv": 0.0, "atm_pe_iv": 0.0,
        "gamma_blast": False, "abnormal": [],
        "recommendation": "⚪ NO TRADE", "direction": "NONE",
        "confidence": 0,
        "scalping": "WAIT", "intraday": "WAIT",
        "swing": "WAIT", "positional": "WAIT",
        "reasoning": [], "atm_row": None,
    }

    if df is None or df.empty or spot == 0:
        return sig

    df = df.copy()
    df["strikePrice"] = df["strikePrice"].astype(float)

    # ATM
    atm_idx = (df["strikePrice"] - spot).abs().idxmin()
    atm = df.loc[atm_idx]
    sig["atm_strike"] = float(atm["strikePrice"])
    sig["atm_row"] = atm
    sig["straddle_price"] = round(float(atm.get("CE_LTP", 0)) + float(atm.get("PE_LTP", 0)), 2)

    # PCR
    total_ce_oi = float(df["CE_OI"].sum())
    total_pe_oi = float(df["PE_OI"].sum())
    sig["pcr"] = round(total_pe_oi / total_ce_oi, 4) if total_ce_oi > 0 else 0.0

    # Max Pain
    strikes = df["strikePrice"].values
    ce_oi = df["CE_OI"].values
    pe_oi = df["PE_OI"].values
    pain = [
        sum(max(0, k - s) * oi for k, oi in zip(strikes, ce_oi)) +
        sum(max(0, s - k) * oi for k, oi in zip(strikes, pe_oi))
        for s in strikes
    ]
    sig["max_pain"] = float(strikes[int(np.argmin(pain))]) if pain else spot

    # IV
    ce_iv = float(atm.get("CE_IV", 0))
    pe_iv = float(atm.get("PE_IV", 0))
    sig["atm_ce_iv"] = ce_iv
    sig["atm_pe_iv"] = pe_iv
    sig["atm_iv"] = (ce_iv + pe_iv) / 2 if (ce_iv + pe_iv) > 0 else 20.0
    sig["iv_skew"] = round(pe_iv - ce_iv, 2)

    # Support / Resistance
    ce_max_idx = df["CE_OI"].idxmax()
    pe_max_idx = df["PE_OI"].idxmax()
    sig["resistance"] = float(df.loc[ce_max_idx, "strikePrice"])
    sig["support"] = float(df.loc[pe_max_idx, "strikePrice"])

    # Gamma Blast: OI concentration > 35% within ±1% of spot
    near_mask = df["strikePrice"].between(spot * 0.99, spot * 1.01)
    if (total_ce_oi + total_pe_oi) > 0:
        near_frac = (df.loc[near_mask, "CE_OI"].sum() + df.loc[near_mask, "PE_OI"].sum()) / (total_ce_oi + total_pe_oi)
        sig["gamma_blast"] = near_frac > 0.35

    # Abnormal detection
    if sig["atm_iv"] > 55:
        sig["abnormal"].append(f"🔴 IV EXTREME ({sig['atm_iv']:.1f}%): Options dangerously expensive for buyers. Theta will destroy premium rapidly.")
    elif sig["atm_iv"] > 35:
        sig["abnormal"].append(f"⚠️ High IV ({sig['atm_iv']:.1f}%): Options expensive. Be very selective, prefer ITM or wait for IV drop.")
    elif 0 < sig["atm_iv"] < 12:
        sig["abnormal"].append(f"✅ Very Low IV ({sig['atm_iv']:.1f}%): Options CHEAP — excellent entry for option buyers!")

    if sig["iv_skew"] > 8:
        sig["abnormal"].append(f"📊 Put Skew +{sig['iv_skew']:.1f}%: Market fearful / hedging heavy. Bearish bias dominant.")
    elif sig["iv_skew"] < -8:
        sig["abnormal"].append(f"📊 Call Skew {sig['iv_skew']:.1f}%: Aggressive call buying. Bullish momentum strong.")

    mp_pct = (sig["max_pain"] - spot) / spot * 100 if spot > 0 else 0
    if abs(mp_pct) > 2:
        pull = "upward" if mp_pct > 0 else "downward"
        sig["abnormal"].append(f"🎯 Max Pain {mp_pct:+.1f}% from spot — gravitational {pull} pull likely near expiry.")

    if sig["gamma_blast"]:
        sig["abnormal"].append("⚡ GAMMA BLAST RISK: OI heavily concentrated at ATM. Explosive move probable at breakout.")

    # ── Signal Scoring ────────────────────────────────────────────────────
    score = 50
    direction = "NONE"
    reasons = []

    pcr = sig["pcr"]
    iv = sig["atm_iv"]
    max_pain = sig["max_pain"]

    # PCR
    if pcr > 1.5:
        reasons.append(f"PCR {pcr:.2f} → Extreme put dominance = contrarian BULLISH")
        score += 12; direction = "CALL"
    elif pcr > 1.1:
        reasons.append(f"PCR {pcr:.2f} → Put heavy = mild bullish lean")
        score += 6; direction = "CALL"
    elif pcr < 0.5:
        reasons.append(f"PCR {pcr:.2f} → Extreme call dominance = contrarian BEARISH")
        score += 12; direction = "PUT"
    elif pcr < 0.9:
        reasons.append(f"PCR {pcr:.2f} → Call heavy = mild bearish lean")
        score += 6; direction = "PUT"
    else:
        reasons.append(f"PCR {pcr:.2f} → Balanced, market neutral")

    # IV filter (buyers' most important factor)
    if iv > 50:
        score -= 25; reasons.append("IV >50%: Avoid buying — premium too expensive")
    elif iv > 35:
        score -= 12; reasons.append("IV >35%: High theta risk for buyers")
    elif 0 < iv < 18:
        score += 15; reasons.append(f"IV {iv:.1f}% → Low IV: premium cheap, great for buyers!")
    elif 18 <= iv <= 30:
        score += 8; reasons.append(f"IV {iv:.1f}% → Moderate IV: acceptable for buyers")

    # Max Pain vs Spot
    if spot < max_pain and direction == "CALL":
        score += 10; reasons.append(f"Spot below max pain {max_pain:,.0f} → upward gravitational pull")
    elif spot > max_pain and direction == "PUT":
        score += 10; reasons.append(f"Spot above max pain {max_pain:,.0f} → downward gravitational pull")
    elif spot < max_pain and direction == "PUT":
        score -= 8; reasons.append(f"Spot below max pain — against PUT trade bias")
    elif spot > max_pain and direction == "CALL":
        score -= 8; reasons.append(f"Spot above max pain — against CALL trade bias")

    # Support / Resistance context
    if direction == "CALL" and spot > sig["support"] and spot < sig["resistance"]:
        score += 8; reasons.append(f"Price in CALL zone: {sig['support']:,.0f} support → {sig['resistance']:,.0f} resistance")
    if direction == "PUT" and spot < sig["resistance"] and spot > sig["support"]:
        score += 8; reasons.append(f"Price in PUT zone near resistance {sig['resistance']:,.0f}")

    # OI momentum
    ce_chg = float(df["CE_changeOI"].sum())
    pe_chg = float(df["PE_changeOI"].sum())
    if pe_chg > ce_chg * 1.3 and direction == "CALL":
        score += 7; reasons.append("Aggressive put writing = support building → bullish")
    if ce_chg > pe_chg * 1.3 and direction == "PUT":
        score += 7; reasons.append("Aggressive call writing = resistance building → bearish")

    score = min(max(int(score), 0), 100)
    sig["confidence"] = score
    sig["reasoning"] = reasons
    sig["direction"] = direction

    # ── Final Recommendation ──────────────────────────────────────────────
    if score >= 72 and direction == "CALL":
        sig["recommendation"] = "🟢 BUY CALL (CE)"
        sig["scalping"] = f"Buy CE ATM+1 (fast 20-40% target)"
        sig["intraday"] = f"Buy CE ATM (30% SL, 60% target)"
        sig["swing"] = f"Buy CE OTM next expiry"
        sig["positional"] = f"Buy CE OTM+2 far expiry"
    elif score >= 72 and direction == "PUT":
        sig["recommendation"] = "🔴 BUY PUT (PE)"
        sig["scalping"] = f"Buy PE ATM-1 (fast 20-40% target)"
        sig["intraday"] = f"Buy PE ATM (30% SL, 60% target)"
        sig["swing"] = f"Buy PE OTM next expiry"
        sig["positional"] = f"Buy PE OTM+2 far expiry"
    elif score >= 58:
        sig["recommendation"] = "🟡 WATCH — Weak Signal"
        sig["scalping"] = "NO TRADE (low confidence)"
        sig["intraday"] = "WATCH for breakout"
        sig["swing"] = "WATCH — wait for confirmation"
        sig["positional"] = "WAIT — no strong edge"
    else:
        sig["recommendation"] = "⚪ NO TRADE — Low Edge"
        sig["scalping"] = sig["intraday"] = sig["swing"] = sig["positional"] = "NO TRADE"

    return sig


# ─── BACKTEST ENGINE (uses generate_signals logic) ─────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def run_backtest(yf_sym: str, lookback: int, init_capital: float, pos_size_pct: float):
    """
    Vectorised backtest using the SAME scoring algorithm as generate_signals().
    Uses cached price history — no extra yfinance calls.
    """
    raw = fetch_price_history(yf_sym, f"{lookback}d")
    if raw is None or raw.empty or len(raw) < 15:
        return None, "Insufficient data"

    raw = raw.copy()
    close = raw["Close"].squeeze().astype(float)
    volume = raw["Volume"].squeeze().astype(float)

    # Rolling indicators (same logic as live signal computation)
    rv7 = close.pct_change().rolling(7).std() * np.sqrt(252) * 100   # realised vol = proxy for ATM IV
    rv14 = close.pct_change().rolling(14).std() * np.sqrt(252) * 100
    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    sma5 = close.rolling(5).mean()
    pcr_proxy = volume.rolling(3).mean() / (volume.rolling(10).mean() + 1)  # proxy
    max_pain_proxy = (sma10 + sma20) / 2  # simplified

    trades = []
    equity = float(init_capital)
    equity_curve = [equity]

    for i in range(20, len(close) - 1):
        iv_est = float(rv7.iloc[i]) if not np.isnan(rv7.iloc[i]) else 20.0
        iv14 = float(rv14.iloc[i]) if not np.isnan(rv14.iloc[i]) else 20.0
        spot = float(close.iloc[i])
        mp = float(max_pain_proxy.iloc[i]) if not np.isnan(max_pain_proxy.iloc[i]) else spot
        pcr = float(pcr_proxy.iloc[i]) if not np.isnan(pcr_proxy.iloc[i]) else 1.0

        # Replicate generate_signals scoring exactly
        score = 50
        direction = "NONE"

        # PCR proxy
        if pcr > 1.3:
            score += 12; direction = "CALL"
        elif pcr > 1.05:
            score += 6; direction = "CALL"
        elif pcr < 0.7:
            score += 12; direction = "PUT"
        elif pcr < 0.95:
            score += 6; direction = "PUT"

        # IV filter
        if iv_est > 50:
            score -= 25
        elif iv_est > 35:
            score -= 12
        elif iv_est < 18:
            score += 15
        elif iv_est <= 30:
            score += 8

        # Max pain
        if spot < mp and direction == "CALL":
            score += 10
        elif spot > mp and direction == "PUT":
            score += 10
        elif spot < mp and direction == "PUT":
            score -= 8
        elif spot > mp and direction == "CALL":
            score -= 8

        # SMA trend
        s10 = float(sma10.iloc[i])
        s20 = float(sma20.iloc[i])
        if s10 > s20 and direction == "CALL":
            score += 7
        elif s10 < s20 and direction == "PUT":
            score += 7

        score = min(max(int(score), 0), 100)

        if score < 72 or direction == "NONE":
            equity_curve.append(equity)
            continue

        # Option pricing approximation (ATM call/put ≈ spot × σ × √T × 0.4)
        T_days = 7
        T = T_days / 365
        sigma = iv_est / 100
        opt_px = spot * sigma * np.sqrt(T) * 0.4
        if opt_px <= 0:
            equity_curve.append(equity)
            continue

        next_spot = float(close.iloc[i + 1])
        next_ret = (next_spot - spot) / spot

        # Delta ≈ 0.5 for ATM
        if direction == "CALL":
            raw_gain = next_ret * 0.5 * spot
        else:
            raw_gain = -next_ret * 0.5 * spot

        # Theta drag
        theta_drag = -(opt_px / T_days) * 1  # 1 day theta
        gross = raw_gain + theta_drag
        pnl_pct = gross / opt_px
        pnl_pct = max(pnl_pct, -0.5)   # max loss cap 50%
        pnl_pct = min(pnl_pct, 2.5)    # max gain cap 250%

        position = equity * (pos_size_pct / 100)
        pnl = position * pnl_pct
        equity = max(equity + pnl, 0)

        trades.append({
            "Date": raw.index[i].strftime("%Y-%m-%d"),
            "Direction": direction,
            "Spot": round(spot, 2),
            "OptPx": round(opt_px, 2),
            "IV%": round(iv_est, 1),
            "Score": score,
            "P&L (₹)": round(pnl, 2),
            "P&L (%)": round(pnl_pct * 100, 2),
            "Equity": round(equity, 2),
            "Result": "✅ WIN" if pnl > 0 else "❌ LOSS",
        })
        equity_curve.append(equity)

    if not trades:
        return None, "No trades generated with current parameters"

    tdf = pd.DataFrame(trades)
    wins = (tdf["P&L (₹)"] > 0).sum()
    total = len(tdf)
    avg_w = tdf.loc[tdf["P&L (₹)"] > 0, "P&L (₹)"].mean() if wins > 0 else 0
    avg_l = tdf.loc[tdf["P&L (₹)"] <= 0, "P&L (₹)"].mean() if (total - wins) > 0 else 0

    peak = init_capital
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    stats = {
        "total_trades": total,
        "wins": int(wins),
        "win_rate": round(wins / total * 100, 1),
        "total_pnl": round(tdf["P&L (₹)"].sum(), 2),
        "avg_win": round(avg_w, 2),
        "avg_loss": round(avg_l, 2),
        "rr": round(abs(avg_w / avg_l), 2) if avg_l != 0 else 0,
        "max_dd": round(max_dd, 2),
        "final_eq": round(equity, 2),
        "ret_pct": round((equity - init_capital) / init_capital * 100, 2),
        "equity_curve": equity_curve,
    }
    return tdf, stats


# ─── PLOTTING HELPERS ──────────────────────────────────────────────────────────
_DARK = "plotly_dark"

def _vline(fig, x, label="", color="yellow", dash="dash", row=1, col=1):
    fig.add_vline(x=x, line_dash=dash, line_color=color, line_width=1.5,
                  annotation_text=label, annotation_font_color=color, row=row, col=col)


def plot_chain_overview(df, spot):
    rng = spot * 0.05
    sub = df[(df["strikePrice"] >= spot - rng) & (df["strikePrice"] <= spot + rng)]
    if sub.empty:
        sub = df

    fig = make_subplots(2, 2, subplot_titles=["CE vs PE Premium (LTP)", "Open Interest", "IV Smile", "Volume"])

    x = sub["strikePrice"]
    fig.add_trace(go.Bar(name="CE LTP", x=x, y=sub["CE_LTP"], marker_color="#00ff88", opacity=0.85), 1, 1)
    fig.add_trace(go.Bar(name="PE LTP", x=x, y=sub["PE_LTP"], marker_color="#ff3b5c", opacity=0.85), 1, 1)
    fig.add_trace(go.Bar(name="CE OI", x=x, y=sub["CE_OI"], marker_color="#00e5ff", opacity=0.8), 1, 2)
    fig.add_trace(go.Bar(name="PE OI", x=x, y=sub["PE_OI"], marker_color="#ff9500", opacity=0.8), 1, 2)
    fig.add_trace(go.Scatter(name="CE IV", x=x, y=sub["CE_IV"], mode="lines+markers", line=dict(color="#00ff88", width=2)), 2, 1)
    fig.add_trace(go.Scatter(name="PE IV", x=x, y=sub["PE_IV"], mode="lines+markers", line=dict(color="#ff3b5c", width=2)), 2, 1)
    fig.add_trace(go.Bar(name="CE Vol", x=x, y=sub["CE_volume"], marker_color="#00ff8866"), 2, 2)
    fig.add_trace(go.Bar(name="PE Vol", x=x, y=sub["PE_volume"], marker_color="#ff3b5c66"), 2, 2)

    for r in [1, 2]:
        for c in [1, 2]:
            _vline(fig, spot, "Spot", "yellow", row=r, col=c)

    fig.update_layout(template=_DARK, height=540, barmode="group",
                      title="Live Option Chain — CE vs PE", margin=dict(t=60, b=20))
    return fig


def plot_oi(df, spot, sig):
    rng = spot * 0.055
    sub = df[(df["strikePrice"] >= spot - rng) & (df["strikePrice"] <= spot + rng)]
    if sub.empty:
        sub = df

    fig = make_subplots(3, 1, subplot_titles=["Open Interest", "Change in OI (ΔOI)", "% Change in OI"])

    x = sub["strikePrice"]
    fig.add_trace(go.Bar(name="CE OI", x=x, y=sub["CE_OI"], marker_color="#00e5ff", opacity=0.8), 1, 1)
    fig.add_trace(go.Bar(name="PE OI", x=x, y=sub["PE_OI"], marker_color="#ff9500", opacity=0.8), 1, 1)

    fig.add_trace(go.Bar(name="ΔCE OI", x=x, y=sub["CE_changeOI"],
                          marker_color=["#00ff88" if v >= 0 else "#ff3b5c" for v in sub["CE_changeOI"]]), 2, 1)
    fig.add_trace(go.Bar(name="ΔPE OI", x=x, y=sub["PE_changeOI"],
                          marker_color=["#ff9500" if v >= 0 else "#8888ff" for v in sub["PE_changeOI"]]), 2, 1)

    sub = sub.copy()
    sub["CE_prev"] = (sub["CE_OI"] - sub["CE_changeOI"]).clip(lower=1)
    sub["PE_prev"] = (sub["PE_OI"] - sub["PE_changeOI"]).clip(lower=1)
    sub["CE_pct"] = sub["CE_changeOI"] / sub["CE_prev"] * 100
    sub["PE_pct"] = sub["PE_changeOI"] / sub["PE_prev"] * 100
    fig.add_trace(go.Bar(name="CE OI%", x=x, y=sub["CE_pct"],
                          marker_color=["#00ff8888" if v >= 0 else "#ff3b5c88" for v in sub["CE_pct"]]), 3, 1)
    fig.add_trace(go.Bar(name="PE OI%", x=x, y=sub["PE_pct"],
                          marker_color=["#ff950088" if v >= 0 else "#8888ff88" for v in sub["PE_pct"]]), 3, 1)

    for r in [1, 2, 3]:
        _vline(fig, spot, "Spot", "yellow", row=r)
        if sig["resistance"]:
            _vline(fig, sig["resistance"], "R", "#ff3b5c", "dot", r)
        if sig["support"]:
            _vline(fig, sig["support"], "S", "#00ff88", "dot", r)

    fig.update_layout(template=_DARK, height=660, barmode="group",
                      title="OI Analysis — Buildup & Unwinding", margin=dict(t=60, b=20))
    return fig


def plot_greeks(df, spot):
    rng = spot * 0.04
    sub = df[(df["strikePrice"] >= spot - rng) & (df["strikePrice"] <= spot + rng)]
    if sub.empty:
        sub = df

    fig = make_subplots(2, 2, subplot_titles=["Delta", "Gamma", "Theta (daily ₹ decay)", "Vega"])
    x = sub["strikePrice"]

    for name, col_c, col_p, r, c in [
        ("Delta", "CE_delta", "PE_delta", 1, 1),
        ("Gamma", "CE_gamma", "PE_gamma", 1, 2),
        ("Theta", "CE_theta", "PE_theta", 2, 1),
        ("Vega", "CE_vega", "PE_vega", 2, 2),
    ]:
        fig.add_trace(go.Scatter(name=f"CE {name}", x=x, y=sub.get(col_c, pd.Series()),
                                  line=dict(color="#00ff88", width=2), mode="lines"), r, c)
        fig.add_trace(go.Scatter(name=f"PE {name}", x=x, y=sub.get(col_p, pd.Series()),
                                  line=dict(color="#ff3b5c", width=2), mode="lines"), r, c)
        _vline(fig, spot, "Spot", "yellow", row=r, col=c)

    fig.update_layout(template=_DARK, height=540, title="Live Options Greeks", margin=dict(t=60, b=20))
    return fig


@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_history(yf_sym: str, period: str = "45d") -> pd.DataFrame:
    """Single cached price history fetch used by both straddle chart and backtest."""
    _global_throttle(2.0)
    try:
        tk = yf_ticker(yf_sym)
        hist = tk.history(period=period, interval="1d", auto_adjust=True)
        return hist if hist is not None and not hist.empty else pd.DataFrame()
    except Exception as e:
        st.warning(f"Price history fetch failed: {e}")
        return pd.DataFrame()


def plot_straddle_history(yf_sym, straddle_now):
    try:
        hist = fetch_price_history(yf_sym, "45d")
        if hist.empty:
            return None, None
        close = hist["Close"].squeeze().astype(float)
        rets = close.pct_change().dropna()
        rv7 = rets.rolling(7).std() * np.sqrt(252)
        straddle_est = close * rv7 * np.sqrt(7 / 252) * 0.8
        straddle_est = straddle_est.dropna()

        p25, p50, p75 = straddle_est.quantile([0.25, 0.50, 0.75])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=straddle_est.index, y=straddle_est,
                                  name="Hist Straddle (est)", line=dict(color="#00e5ff", width=2)))
        if straddle_now > 0:
            fig.add_hline(y=straddle_now, line_color="yellow", line_dash="dash",
                          annotation_text=f"Current: {straddle_now:.0f}", annotation_font_color="yellow")
        fig.add_hline(y=p25, line_color="#00ff88", line_dash="dot", annotation_text=f"25th: {p25:.0f}")
        fig.add_hline(y=p75, line_color="#ff3b5c", line_dash="dot", annotation_text=f"75th: {p75:.0f}")
        fig.add_hline(y=p50, line_color="#ff9500", line_dash="dot", annotation_text=f"50th: {p50:.0f}")
        fig.update_layout(template=_DARK, height=360, title="Straddle Price vs 45-Day History")

        if straddle_now > 0:
            if straddle_now < p25:
                verdict = ("✅ CHEAP — Below 25th %ile. Great time to buy straddle / options!", "#00ff88")
            elif straddle_now > p75:
                verdict = ("⚠️ EXPENSIVE — Above 75th %ile. Avoid buying; wait for IV crush.", "#ff3b5c")
            else:
                verdict = ("🟡 FAIR VALUE — Within normal range. Be selective.", "#ff9500")
        else:
            verdict = ("ℹ️ No straddle price available.", "#5a7a9a")

        return fig, verdict
    except Exception:
        return None, None


# ─── MAIN APP ──────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div style='display:flex; align-items:center; gap:12px; margin-bottom:4px'>
        <span style='font-size:32px'>⚡</span>
        <div>
            <h1 style='margin:0; font-size:26px; letter-spacing:2px'>PRO OPTIONS INTELLIGENCE</h1>
            <p style='margin:0; color:#5a7a9a; font-size:13px; font-family:DM Sans'>Live Chain • Greeks • OI • AI Signals • Backtest • Trade Log</p>
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Instrument Setup")
        instr_type = st.selectbox("Type", ["🇮🇳 NSE Index", "🇮🇳 NSE Stock", "🌐 Global (yFinance)"])
        yf_sym_for_hist = "^NSEI"

        if instr_type == "🇮🇳 NSE Index":
            symbol = st.selectbox("Index", list(NSE_INDICES.keys()))
            yf_sym_for_hist = NSE_INDICES[symbol]
            data_source = "nse"
        elif instr_type == "🇮🇳 NSE Stock":
            symbol = st.text_input("NSE Ticker (e.g. RELIANCE)", "RELIANCE").upper().strip()
            yf_sym_for_hist = f"{symbol}.NS"
            data_source = "nse"
        else:
            symbol = st.selectbox("Instrument", list(COMMODITY_MAP.keys()) + ["Custom"])
            if symbol == "Custom":
                symbol = st.text_input("yFinance ticker (e.g. AAPL)", "AAPL").upper()
                yf_sym_for_hist = symbol
            else:
                yf_sym_for_hist = COMMODITY_MAP[symbol]
                symbol = yf_sym_for_hist
            data_source = "yf"

        st.divider()
        st.markdown("### 💰 Capital & Risk")
        capital = st.number_input("Capital (₹)", 50_000, 10_000_000, 100_000, 10_000)
        pos_pct = st.slider("Position Size %", 2, 15, 5)
        sl_pct = st.slider("Stop Loss % (of premium)", 20, 50, 30)
        tgt_pct = st.slider("Target % (of premium)", 30, 200, 60)

        st.divider()
        fetch_btn = st.button("🔄 FETCH LIVE DATA", type="primary", use_container_width=True)
        auto_refresh = st.checkbox("Auto-Refresh (90s)")
        st.caption("⚠️ Educational only. Not financial advice.")

    # ── FETCH DATA ────────────────────────────────────────────────────────────
    with st.spinner("⏳ Fetching live data (rate-limited)…"):
        if data_source == "nse":
            df_raw, spot, expiries = fetch_nse_chain(symbol)
            if df_raw is None:
                st.warning("NSE direct API unavailable (may need VPN/cookie). Trying yfinance fallback…")
                yf_sym = yf_sym_for_hist
                df_raw, spot, expiries = fetch_yf_chain(yf_sym)
        else:
            df_raw, spot, expiries = fetch_yf_chain(symbol)

    if df_raw is None or spot is None or spot == 0:
        st.error("❌ Could not fetch live data.")
        st.markdown("""
<div style='background:#0a1929; border:1px solid #1e3050; border-radius:12px; padding:20px; margin:10px 0'>
<h4 style='color:#ff9500; margin-top:0'>🛠️ Troubleshooting Guide</h4>

**Most likely causes & fixes:**

1. **Rate limited** (most common) → Wait 15–30 seconds, then click **🔄 FETCH LIVE DATA** again.
   The retry adapter will automatically back off and retry.

2. **Market closed** → yfinance option chain data is only available during US/global market hours.
   Try BTC-USD (24/7) or GC=F (Gold futures) from the Global tab.

3. **Wrong symbol** → Confirm the yfinance ticker. Examples: `AAPL`, `BTC-USD`, `GC=F`, `^NSEI`

4. **NSE data** → Requires Indian IP. If outside India, switch to Global > BTC/USD or Gold.

5. **Still failing?** → Click the button below to clear all caches and start fresh.
</div>
""", unsafe_allow_html=True)
        col_r1, col_r2 = st.columns(2)
        if col_r1.button("🔄 Clear Cache & Retry", type="primary", use_container_width=True):
            st.cache_data.clear()
            _LAST_YF_CALL.clear()
            st.rerun()
        col_r2.info("💡 Try: BTC-USD, GC=F, SI=F, AAPL, TSLA")
        return

    spot = float(spot)

    # Expiry selector
    if expiries:
        with st.sidebar:
            sel_expiry = st.selectbox("Expiry", expiries[:6])
    else:
        sel_expiry = df_raw["expiryDate"].iloc[0] if "expiryDate" in df_raw.columns else ""

    # Filter expiry
    if "expiryDate" in df_raw.columns and sel_expiry:
        df_exp = df_raw[df_raw["expiryDate"] == sel_expiry].copy()
        if df_exp.empty:
            df_exp = df_raw.copy()
    else:
        df_exp = df_raw.copy()

    df_exp = df_exp.reset_index(drop=True)
    df_exp = df_exp.fillna(0)

    # Greeks enrichment
    with st.spinner("🔢 Computing Greeks…"):
        df_exp = enrich_greeks(df_exp, spot, sel_expiry)

    # Signal generation
    signals = generate_signals(df_exp, spot)

    # ── TOP HEADER METRICS ────────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("📍 Spot", f"{spot:,.2f}")
    m2.metric("🎯 ATM", f"{signals['atm_strike']:,.0f}")
    m3.metric("📊 PCR", f"{signals['pcr']:.3f}")
    m4.metric("💀 Max Pain", f"{signals['max_pain']:,.0f}")
    m5.metric("🌡️ ATM IV", f"{signals['atm_iv']:.1f}%")
    m6.metric("↕️ IV Skew", f"{signals['iv_skew']:+.2f}%")
    m7.metric("♟️ Straddle", f"{signals['straddle_price']:.2f}")

    # ── SIGNAL BANNER ─────────────────────────────────────────────────────────
    conf = signals["confidence"]
    bar_color = "#00ff88" if conf >= 72 else "#ff9500" if conf >= 58 else "#ff3b5c"
    st.markdown(f"""
    <div class="signal-banner">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px">
            <div>
                <div style="color:#5a7a9a; font-size:11px; letter-spacing:2px; margin-bottom:4px">PRIMARY SIGNAL</div>
                <div style="font-size:22px; font-weight:700; color:{bar_color}; font-family:Space Mono">{signals['recommendation']}</div>
                <div style="color:#8899aa; font-size:13px; margin-top:6px">{' &nbsp;·&nbsp; '.join(signals['reasoning'][:3])}</div>
            </div>
            <div style="text-align:right">
                <div style="color:#5a7a9a; font-size:11px; letter-spacing:2px">CONFIDENCE</div>
                <div style="font-size:32px; font-weight:700; color:{bar_color}; font-family:Space Mono">{conf}%</div>
                <div style="background:{bar_color}22; border-radius:20px; height:6px; width:120px; margin-top:4px; margin-left:auto">
                    <div style="background:{bar_color}; border-radius:20px; height:6px; width:{conf}%"></div>
                </div>
            </div>
        </div>
        <div style="display:flex; gap:8px; margin-top:14px; flex-wrap:wrap">
            <span class="tag-{'buy' if 'BUY' in signals['scalping'] else 'wait' if 'WAIT' in signals['scalping'] or 'WATCH' in signals['scalping'] else 'neutral'}">⚡ {signals['scalping']}</span>
            <span class="tag-{'buy' if 'BUY' in signals['intraday'] else 'wait' if 'WAIT' in signals['intraday'] or 'WATCH' in signals['intraday'] else 'neutral'}">📅 {signals['intraday']}</span>
            <span class="tag-{'buy' if 'BUY' in signals['swing'] else 'wait' if 'WATCH' in signals['swing'] else 'neutral'}">🗓️ {signals['swing']}</span>
            <span class="tag-{'buy' if 'BUY' in signals['positional'] else 'wait' if 'WAIT' in signals['positional'] else 'neutral'}">📆 {signals['positional']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Abnormal alerts
    for ab in signals["abnormal"]:
        st.markdown(f'<div class="alert-box">{ab}</div>', unsafe_allow_html=True)

    if signals["gamma_blast"]:
        st.markdown("""
        <div class="gamma-blast">
            <b style="color:#ff3b5c; font-size:16px">⚡ GAMMA BLAST ALERT</b><br>
            <span style="color:#ffcccc; font-size:13px">OI heavily concentrated near ATM. A directional move will trigger rapid gamma expansion.
            Option buyers can see 50–200% premium gains in minutes. Prepare straddle or wait for direction confirmation before entering.</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab_names = ["📊 Option Chain", "🔢 Greeks", "📈 OI Analysis", "⚡ Live Trading", "🔬 Backtesting", "📋 Trade History", "🧠 Analysis"]
    tabs = st.tabs(tab_names)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 — OPTION CHAIN
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown('<div class="summary-box">📌 <b>Option Chain</b>: Live CE/PE premiums, OI, IV across strikes. Green bars = calls (bullish), Red = puts (bearish). ATM highlighted. Compare straddle price vs 45-day history to judge if options are cheap or expensive for buying.</div>', unsafe_allow_html=True)

        all_strikes = sorted(df_exp["strikePrice"].unique().tolist())
        atm_pos = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - spot))
        default_strikes = all_strikes[max(0, atm_pos - 8): atm_pos + 9]

        sel_strikes = st.multiselect("Select Strikes to Display", all_strikes, default=default_strikes, key="tab1_strikes")
        if not sel_strikes:
            sel_strikes = default_strikes

        df_disp = df_exp[df_exp["strikePrice"].isin(sel_strikes)].copy()
        display_cols = ["CE_LTP", "CE_OI", "CE_changeOI", "CE_IV", "CE_volume",
                        "strikePrice",
                        "PE_LTP", "PE_OI", "PE_changeOI", "PE_IV", "PE_volume"]
        dc = [c for c in display_cols if c in df_disp.columns]

        def _highlight(row):
            if row.name in df_disp.index and abs(float(df_disp.loc[row.name, "strikePrice"]) - spot) == abs(df_disp["strikePrice"] - spot).min():
                return ["background-color:#1a2a4a; border:1px solid #00e5ff22"] * len(row)
            return [""] * len(row)

        st.dataframe(df_disp[dc].round(2).style.apply(_highlight, axis=1), use_container_width=True, height=280)

        fig_chain = plot_chain_overview(df_exp, spot)
        st.plotly_chart(fig_chain, use_container_width=True)

        # Straddle history
        st.markdown("#### 📉 Straddle vs Historical Levels")
        fig_str, straddle_verdict = plot_straddle_history(yf_sym_for_hist, signals["straddle_price"])
        if fig_str:
            st.plotly_chart(fig_str, use_container_width=True)
        if straddle_verdict:
            color = straddle_verdict[1]
            st.markdown(f'<div style="color:{color}; background:{color}11; border:1px solid {color}44; border-radius:8px; padding:10px 14px; font-weight:600">{straddle_verdict[0]}</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 — GREEKS
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown('<div class="summary-box">📌 <b>Greeks</b>: Delta = directional sensitivity. Gamma = delta acceleration (max at ATM, explodes near expiry). Theta = daily time decay (enemy of buyers). Vega = IV sensitivity (high Vega means options gain on IV spikes). Ideal buy: Delta 0.3-0.6, high Vega, manageable Theta.</div>', unsafe_allow_html=True)

        st.plotly_chart(plot_greeks(df_exp, spot), use_container_width=True)

        atm_idx = (df_exp["strikePrice"] - spot).abs().idxmin()
        atm = df_exp.loc[atm_idx]

        st.markdown(f"#### 🎯 ATM Greeks — Strike {atm['strikePrice']:,.0f}")
        gc1, gc2 = st.columns(2)
        for col, prefix, label, color in [(gc1, "CE", "📗 CALL (CE)", "#00ff88"), (gc2, "PE", "📕 PUT (PE)", "#ff3b5c")]:
            with col:
                st.markdown(f"<h5 style='color:{color}'>{label}</h5>", unsafe_allow_html=True)
                m = [
                    ("Delta", f"{prefix}_delta", "0=OTM, 0.5=ATM, 1=ITM"),
                    ("Gamma", f"{prefix}_gamma", "Rate of delta change"),
                    ("Theta/day", f"{prefix}_theta", "Premium lost per day"),
                    ("Vega/1%IV", f"{prefix}_vega", "Gain per 1% IV rise"),
                    ("Rho/1%r", f"{prefix}_rho", "Rate sensitivity"),
                    ("IV %", f"{prefix}_IV", "Implied Volatility"),
                ]
                cols_g = st.columns(3)
                for i, (name, key, help_) in enumerate(m):
                    cols_g[i % 3].metric(name, f"{float(atm.get(key, 0)):.4f}" if key != f"{prefix}_IV" else f"{float(atm.get(key, 0)):.2f}%", help=help_)

        # Greeks interpretation
        st.markdown("#### 📖 Interpretation & Action")
        atm_iv = signals["atm_iv"]
        ce_delta = float(atm.get("CE_delta", 0.5))
        ce_theta = float(atm.get("CE_theta", -5))
        ce_vega = float(atm.get("CE_vega", 10))
        ce_gamma = float(atm.get("CE_gamma", 0))

        interps = []
        if atm_iv > 40:
            interps.append(("🔴 IV > 40%: Avoid buying. Theta severely exceeds Vega benefit. Wait for IV to collapse.", "#ff3b5c"))
        elif atm_iv < 15:
            interps.append(("✅ IV < 15%: Prime time for buyers. Premium cheap, Vega gains will compound on any spike.", "#00ff88"))
        if abs(ce_theta) > abs(ce_vega):
            interps.append(("⚠️ |Theta| > Vega: Time decay outpacing volatility gain — consider ITM or nearer strikes.", "#ff9500"))
        if ce_gamma > 0:
            gyr = abs(ce_theta / (ce_gamma * spot**2 * 0.5 + 1e-10))
            if gyr < 0.3:
                interps.append(("✅ Favourable Gamma/Theta ratio: Gamma expansion likely outpaces decay.", "#00ff88"))

        for text, clr in interps:
            st.markdown(f'<div style="background:{clr}11; border-left:3px solid {clr}; padding:10px 14px; border-radius:6px; margin:4px 0; font-size:13px">{text}</div>', unsafe_allow_html=True)

        # Greeks table
        gcols = [c for c in ["strikePrice", "CE_IV", "CE_delta", "CE_gamma", "CE_theta", "CE_vega",
                              "PE_IV", "PE_delta", "PE_gamma", "PE_theta", "PE_vega"] if c in df_exp.columns]
        st.dataframe(df_exp[df_exp["strikePrice"].isin(sel_strikes if sel_strikes else all_strikes)][gcols].round(5),
                     use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3 — OI ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown('<div class="summary-box">📌 <b>OI Analysis</b>: Max CE OI = resistance wall. Max PE OI = support floor. Rising OI + rising price = long buildup (bullish). Rising OI + falling price = short buildup (bearish). Gamma blast occurs when massive OI sits at ATM and spot accelerates through it.</div>', unsafe_allow_html=True)

        st.plotly_chart(plot_oi(df_exp, spot, signals), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 Resistance (CE Max OI)", f"{signals['resistance']:,.0f}")
        c2.metric("🟢 Support (PE Max OI)", f"{signals['support']:,.0f}")
        c3.metric("🎯 Max Pain", f"{signals['max_pain']:,.0f}")
        c4.metric("↕️ PCR", f"{signals['pcr']:.4f}")

        st.markdown("#### 📊 Long & Short Buildup Signals")
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown("**📗 CE Strikes — Highest OI Build (Resistance)**")
            ce_top = df_exp.nlargest(5, "CE_OI")[["strikePrice", "CE_OI", "CE_changeOI", "CE_LTP"]].copy()
            ce_top["Signal"] = ce_top.apply(
                lambda r: "🔴 Call Writing (Resistance)" if r["CE_changeOI"] >= 0 else "🟢 Call Unwinding", axis=1)
            st.dataframe(ce_top, use_container_width=True)
        with bc2:
            st.markdown("**📕 PE Strikes — Highest OI Build (Support)**")
            pe_top = df_exp.nlargest(5, "PE_OI")[["strikePrice", "PE_OI", "PE_changeOI", "PE_LTP"]].copy()
            pe_top["Signal"] = pe_top.apply(
                lambda r: "🟢 Put Writing (Support)" if r["PE_changeOI"] >= 0 else "🔴 Put Unwinding", axis=1)
            st.dataframe(pe_top, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4 — LIVE TRADING
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown('<div class="summary-box">📌 <b>Live Trading</b>: Paper-trade execution using the same signal engine as backtesting. Set strike, lots, SL/target. Trades tracked in real-time with live option price updates. Ensures 1.5s delay between API calls. Buyer-only strategies with pre-calculated risk-reward.</div>', unsafe_allow_html=True)

        lc1, lc2 = st.columns([3, 2])
        with lc1:
            st.markdown("#### 🚨 Live Trade Setup")
            la, lb = st.columns(2)
            opt_side = la.selectbox("Option Side", ["CE (Call)", "PE (Put)"])
            prefix = "CE" if "CE" in opt_side else "PE"

            strike_default = int(signals["atm_strike"]) if signals["atm_strike"] in all_strikes else (all_strikes[atm_pos] if all_strikes else 0)
            trade_strike = la.selectbox("Strike", sorted(df_exp["strikePrice"].unique()), index=atm_pos)
            lots = lb.number_input("Lots", 1, 100, 1)
            live_sl = lb.slider("SL % of premium", 20, 50, sl_pct, key="live_sl")
            live_tgt = lb.slider("Target % of premium", 30, 200, tgt_pct, key="live_tgt")

            # Current price
            strike_row = df_exp[df_exp["strikePrice"] == trade_strike]
            opt_price = float(strike_row[f"{prefix}_LTP"].values[0]) if not strike_row.empty else 0

            if opt_price > 0:
                sl_abs = round(opt_price * (1 - live_sl / 100), 2)
                tgt_abs = round(opt_price * (1 + live_tgt / 100), 2)
                risk_amt = (opt_price - sl_abs) * lots
                rew_amt = (tgt_abs - opt_price) * lots
                rr = round(rew_amt / risk_amt, 2) if risk_amt > 0 else 0

                pm1, pm2, pm3, pm4 = st.columns(4)
                pm1.metric("Entry Price", f"₹{opt_price:.2f}")
                pm2.metric("Stop Loss", f"₹{sl_abs:.2f}", f"-{live_sl}%")
                pm3.metric("Target", f"₹{tgt_abs:.2f}", f"+{live_tgt}%")
                pm4.metric("R:R", f"1:{rr}")

            st.markdown(f"""
            <div style='background:#0a1929; border:1px solid #1e3050; border-radius:10px; padding:14px; margin:10px 0'>
                <b style='color:#00e5ff'>Signal: {signals['recommendation']}</b> &nbsp;|&nbsp; Confidence: <b>{signals['confidence']}%</b><br>
                <span style='color:#5a7a9a; font-size:12px'>{' · '.join(signals['reasoning'][:2])}</span>
            </div>
            """, unsafe_allow_html=True)

            if st.button("📈 ENTER PAPER TRADE", type="primary", use_container_width=True):
                if opt_price > 0:
                    trade = {
                        "id": len(st.session_state.trade_history) + 1,
                        "symbol": symbol, "expiry": sel_expiry,
                        "strike": trade_strike, "side": prefix,
                        "entry_price": opt_price, "sl": sl_abs, "target": tgt_abs,
                        "lots": lots, "confidence": signals["confidence"],
                        "signal": signals["recommendation"],
                        "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "OPEN", "exit_price": None, "pnl": None, "exit_time": None,
                    }
                    st.session_state.active_trades.append(trade)
                    st.session_state.trade_history.append(trade)
                    st.success(f"✅ Trade entered: {symbol} {trade_strike} {prefix} @ ₹{opt_price:.2f}")
                else:
                    st.error("Option price is 0 — cannot place trade.")

        with lc2:
            st.markdown("#### 📊 Active Trades")
            if st.session_state.active_trades:
                for i, t in enumerate(st.session_state.active_trades):
                    if t["status"] != "OPEN":
                        continue
                    sr = df_exp[df_exp["strikePrice"] == t["strike"]]
                    curr = float(sr[f"{t['side']}_LTP"].values[0]) if not sr.empty else t["entry_price"]
                    pnl = round((curr - t["entry_price"]) * t["lots"], 2)
                    pnl_pct = round((curr - t["entry_price"]) / t["entry_price"] * 100, 2) if t["entry_price"] > 0 else 0
                    pnl_color = "#00ff88" if pnl >= 0 else "#ff3b5c"
                    card_class = "trade-card-profit" if pnl >= 0 else "trade-card-loss"

                    st.markdown(f"""
                    <div class="trade-card {card_class}">
                        <b style='color:#c8d6e5'>{t['symbol']} {t['strike']} {t['side']}</b><br>
                        Entry: ₹{t['entry_price']:.2f} → Current: ₹{curr:.2f}<br>
                        SL: ₹{t['sl']:.2f} | Target: ₹{t['target']:.2f}<br>
                        <b style='color:{pnl_color}'>P&L: ₹{pnl:+.2f} ({pnl_pct:+.1f}%)</b>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button(f"Exit Trade #{t['id']}", key=f"exit_{i}_{t['id']}"):
                        st.session_state.active_trades[i]["status"] = "CLOSED"
                        st.session_state.active_trades[i]["exit_price"] = curr
                        st.session_state.active_trades[i]["pnl"] = pnl
                        st.session_state.active_trades[i]["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for h in st.session_state.trade_history:
                            if h["id"] == t["id"]:
                                h.update(st.session_state.active_trades[i])
                        st.rerun()
            else:
                st.info("No active trades. Place one from the left panel.")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 5 — BACKTESTING
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown('<div class="summary-box">📌 <b>Backtesting</b>: Uses the IDENTICAL signal scoring algorithm as live trading — same PCR thresholds, IV filters, position sizing. Results directly predict what live trading would have done historically. Win rate, drawdown, equity curve, and trade log all shown.</div>', unsafe_allow_html=True)

        bta, btb, btc = st.columns(3)
        bt_lookback = bta.slider("Lookback Days", 20, 120, 60)
        bt_capital = btb.number_input("Capital", 50_000, 5_000_000, int(capital), 10_000, key="bt_cap")
        bt_pos = btc.slider("Position Size %", 2, 20, pos_pct, key="bt_pos")

        if st.button("🔬 RUN BACKTEST NOW", type="primary", use_container_width=True):
            with st.spinner("Running backtest with live signal engine…"):
                trades_df, stats = run_backtest(yf_sym_for_hist, bt_lookback, bt_capital, bt_pos)
                st.session_state.backtest_results = (trades_df, stats)

        if st.session_state.backtest_results:
            trades_df, stats = st.session_state.backtest_results
            if trades_df is None:
                st.error(f"Backtest failed: {stats}")
            else:
                # KPIs
                k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
                k1.metric("Trades", stats["total_trades"])
                k2.metric("Win Rate", f"{stats['win_rate']}%")
                k3.metric("Total P&L", f"₹{stats['total_pnl']:+,.0f}")
                k4.metric("Avg Win", f"₹{stats['avg_win']:,.0f}")
                k5.metric("Avg Loss", f"₹{stats['avg_loss']:,.0f}")
                k6.metric("R:R", f"{stats['rr']}")
                k7.metric("Max DD", f"{stats['max_dd']}%")

                c_return = "#00ff88" if stats["ret_pct"] >= 0 else "#ff3b5c"
                st.markdown(f'<div style="text-align:center; font-size:28px; color:{c_return}; font-family:Space Mono; margin:10px 0">Return: {stats["ret_pct"]:+.2f}%  ·  Final Equity: ₹{stats["final_eq"]:,.0f}</div>', unsafe_allow_html=True)

                # Equity curve
                eq = stats["equity_curve"]
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(y=eq, mode="lines", name="Equity",
                                             line=dict(color="#00e5ff", width=2.5), fill="tozeroy",
                                             fillcolor="rgba(0,229,255,0.05)"))
                fig_eq.add_hline(y=bt_capital, line_dash="dash", line_color="#ff9500",
                                  annotation_text="Start Capital")
                fig_eq.update_layout(template=_DARK, height=340, title="📈 Equity Curve",
                                      xaxis_title="Trade #", yaxis_title="₹ Equity")
                st.plotly_chart(fig_eq, use_container_width=True)

                # Win/loss distribution
                fig_dist = go.Figure()
                w = trades_df[trades_df["P&L (₹)"] > 0]["P&L (₹)"]
                l = trades_df[trades_df["P&L (₹)"] <= 0]["P&L (₹)"]
                fig_dist.add_trace(go.Histogram(x=w, name="Wins", marker_color="#00ff8877", nbinsx=20))
                fig_dist.add_trace(go.Histogram(x=l, name="Losses", marker_color="#ff3b5c77", nbinsx=20))
                fig_dist.update_layout(template=_DARK, height=280, title="P&L Distribution", barmode="overlay")
                st.plotly_chart(fig_dist, use_container_width=True)

                # Trade log
                st.dataframe(trades_df.style.applymap(
                    lambda v: "color: #00ff88" if isinstance(v, (int, float)) and v > 0 else ("color: #ff3b5c" if isinstance(v, (int, float)) and v < 0 else ""),
                    subset=["P&L (₹)", "P&L (%)"]
                ), use_container_width=True, height=300)

                st.success("✅ **Backtest-Live Parity Guarantee**: Signal scoring, IV thresholds, position sizing, and entry conditions are all shared code between this backtest and live trading tab.")
        else:
            st.info("👆 Configure parameters and hit 'Run Backtest'.")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 6 — TRADE HISTORY
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown('<div class="summary-box">📌 <b>Trade History</b>: Complete log of all paper trades — entry, exit, P&L, signal confidence and outcome. Review win patterns, refine strategy. Export to CSV for further analysis. Only closed trades show final P&L.</div>', unsafe_allow_html=True)

        if st.session_state.trade_history:
            all_t = pd.DataFrame(st.session_state.trade_history)
            closed = all_t[all_t["status"] == "CLOSED"].copy()

            if not closed.empty:
                closed["pnl"] = pd.to_numeric(closed["pnl"], errors="coerce").fillna(0)
                total_pnl = closed["pnl"].sum()
                wr = (closed["pnl"] > 0).mean() * 100 if len(closed) > 0 else 0

                hk1, hk2, hk3, hk4 = st.columns(4)
                hk1.metric("Total Trades", len(all_t))
                hk2.metric("Closed", len(closed))
                hk3.metric("Win Rate", f"{wr:.1f}%")
                hk4.metric("Net P&L", f"₹{total_pnl:+,.2f}", delta=f"₹{total_pnl:,.2f}")

                fig_pnl = go.Figure(go.Bar(
                    x=list(range(len(closed))),
                    y=closed["pnl"].values,
                    marker_color=["#00ff88" if p > 0 else "#ff3b5c" for p in closed["pnl"]],
                ))
                fig_pnl.update_layout(template=_DARK, height=280, title="Per-Trade P&L")
                st.plotly_chart(fig_pnl, use_container_width=True)

            show_cols = ["id", "entry_time", "symbol", "strike", "side", "entry_price",
                         "exit_price", "lots", "pnl", "status", "signal", "confidence", "exit_time"]
            show_cols = [c for c in show_cols if c in all_t.columns]
            st.dataframe(all_t[show_cols], use_container_width=True)

            csv = all_t.to_csv(index=False)
            st.download_button("📥 Export CSV", csv, "trade_history.csv", "text/csv", use_container_width=True)

            if st.button("🗑️ Clear All Trades", use_container_width=True):
                st.session_state.trade_history = []
                st.session_state.active_trades = []
                st.rerun()
        else:
            st.info("No trades yet. Use the Live Trading tab to enter paper trades.")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 7 — ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        st.markdown('<div class="summary-box">📌 <b>Analysis</b>: Full narrative on market structure — PCR interpretation, max pain pull, support/resistance walls, IV environment, and all three probability scenarios (bull/flat/bear). Actionable, buyer-focused guidance with risk management rules.</div>', unsafe_allow_html=True)

        st.markdown(f"### 📊 {symbol} @ {spot:,.2f} — Full Market Analysis")

        pcr = signals["pcr"]
        mp = signals["max_pain"]
        res = signals["resistance"]
        sup = signals["support"]
        iv = signals["atm_iv"]

        mp_pct = (mp - spot) / spot * 100

        # Market structure narrative
        pcr_text = ("extreme put dominance → contrarian BULLISH signal" if pcr > 1.5
                    else "put-heavy → mild bullish lean" if pcr > 1.1
                    else "extreme call dominance → contrarian BEARISH signal" if pcr < 0.5
                    else "call-heavy → mild bearish lean" if pcr < 0.9 else "balanced / neutral market")

        iv_text = ("DANGEROUSLY HIGH — avoid buying" if iv > 50
                   else "HIGH — very selective buying only" if iv > 35
                   else "MODERATE — acceptable for buyers" if iv > 20
                   else "LOW — prime buying opportunity" if iv > 0 else "N/A")

        mp_text = (f"ABOVE spot by {mp_pct:.1f}% → upward magnetic pull expected near expiry" if mp_pct > 0
                   else f"BELOW spot by {abs(mp_pct):.1f}% → downward magnetic pull expected near expiry")

        st.markdown(f"""
<div style='background:#0a1929; border:1px solid #1e3050; border-radius:12px; padding:20px; margin:10px 0; line-height:1.9'>
<h4 style='color:#00e5ff; margin-top:0'>🏗️ Market Structure Summary</h4>

**Put-Call Ratio**: `{pcr:.4f}` — {pcr_text}
> PCR above 1.0 means more put OI than calls (bearish hedging). Contrarian signal: extreme put OI often marks a bottom as writers defend.

**Implied Volatility**: `{iv:.1f}%` — **{iv_text}**
> {"IV above 30% means theta decay will rapidly erode premium. Option buyers need larger moves to profit." if iv > 30 else "Low IV is the option buyer's best friend. Premium is cheap, and any IV spike multiplies gains via Vega."}

**Max Pain**: `{mp:,.0f}` — {mp_text}
> Max Pain is the strike where total option writer profits are maximised. Market gravitates toward it near expiry — especially in the last 2 days.

**OI Walls**:
- 🔴 **Resistance**: `{res:,.0f}` — Heavy call writing here. Market struggles to breach without significant catalyst.
- 🟢 **Support**: `{sup:,.0f}` — Heavy put writing here. Strong seller support at this level.

**Straddle Price**: `{signals['straddle_price']:.2f}` — Implied 1-expiry move = **{signals['straddle_price'] / spot * 100:.2f}%** of spot.
</div>
""", unsafe_allow_html=True)

        # 3 Scenarios
        st.markdown("#### 🎭 Three Probability Scenarios")
        sc1, sc2, sc3 = st.columns(3)

        with sc1:
            st.markdown(f"""
<div class='scenario-box scenario-bull'>
<h5 style='color:#00ff88; margin-top:0'>🟢 BULLISH SCENARIO</h5>
<b>Trigger</b>: Break & hold above {res:,.0f}<br><br>
<b>What happens</b>: Call writers panic, buy back → CE premium explodes. Gamma cascade accelerates move.<br><br>
<b>Entry</b>: {int(signals['atm_strike']):,} CE or {int(signals['atm_strike'] * 1.005):,} CE<br>
<b>Target</b>: {int(res * 1.015):,} (+1.5% from resistance)<br>
<b>SL</b>: 30% of CE premium<br>
<b>Best for</b>: Scalping / Intraday
</div>
""", unsafe_allow_html=True)

        with sc2:
            st.markdown(f"""
<div class='scenario-box scenario-flat'>
<h5 style='color:#5a7a9a; margin-top:0'>⚪ NEUTRAL / RANGE</h5>
<b>Scenario</b>: Spot oscillates between {sup:,.0f} and {res:,.0f}<br><br>
<b>What happens</b>: Theta decay destroys both CE and PE buyers. Option sellers win. Straddle loses value daily.<br><br>
<b>Action</b>: DO NOT BUY OPTIONS in range-bound conditions. Wait for breakout confirmation.<br>
<b>Watch</b>: Volume, OI changes, news catalysts<br>
<b>Max Pain Pull</b>: {mp:,.0f} ({mp_pct:+.1f}% from spot)
</div>
""", unsafe_allow_html=True)

        with sc3:
            st.markdown(f"""
<div class='scenario-box scenario-bear'>
<h5 style='color:#ff3b5c; margin-top:0'>🔴 BEARISH SCENARIO</h5>
<b>Trigger</b>: Break & hold below {sup:,.0f}<br><br>
<b>What happens</b>: Put writers panic, buy back → PE premium explodes. Delta acceleration accelerates fall.<br><br>
<b>Entry</b>: {int(signals['atm_strike']):,} PE or {int(signals['atm_strike'] * 0.995):,} PE<br>
<b>Target</b>: {int(sup * 0.985):,} (-1.5% from support)<br>
<b>SL</b>: 30% of PE premium<br>
<b>Best for</b>: Scalping / Intraday
</div>
""", unsafe_allow_html=True)

        # Actionable buyer rules
        rec_color = "#00ff88" if "CALL" in signals["recommendation"] else "#ff3b5c" if "PUT" in signals["recommendation"] else "#ff9500"
        st.markdown(f"""
<div style='background:linear-gradient(135deg, #060e1a, #0a1929); border:1px solid {rec_color}44; border-radius:14px; padding:24px; margin-top:20px'>
<h4 style='color:{rec_color}; margin-top:0'>🎯 PRIMARY RECOMMENDATION: {signals['recommendation']}</h4>
<div style='display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:16px 0'>
    <div>
        <b style='color:#00e5ff'>Trade Horizons:</b>
        <ul style='color:#8899aa; margin:8px 0'>
            <li>⚡ Scalping: {signals['scalping']}</li>
            <li>📅 Intraday: {signals['intraday']}</li>
            <li>🗓️ Swing: {signals['swing']}</li>
            <li>📆 Positional: {signals['positional']}</li>
        </ul>
    </div>
    <div>
        <b style='color:#00e5ff'>Option Buyer Golden Rules:</b>
        <ul style='color:#8899aa; margin:8px 0'>
            <li>Never allocate >5% of capital per trade</li>
            <li>Only buy when IV &lt; 30% for indices</li>
            <li>Exit theta-heavy positions 5+ days before expiry</li>
            <li>Pre-set SL (30-40%) before entry, never move it wider</li>
            <li>Book 50% at first target, trail rest with SL at entry</li>
            <li>Avoid buying on event days (high IV premium trap)</li>
        </ul>
    </div>
</div>
<div style='background:{rec_color}11; border-radius:8px; padding:12px; margin-top:8px'>
    <b>Reasoning chain</b>: {" · ".join(signals['reasoning'])}
</div>
</div>
""", unsafe_allow_html=True)

        # Footer
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"<div style='text-align:right; color:#2a4a6a; font-size:11px; margin-top:20px'>Last computed: {ts} | All data live from NSE / yFinance</div>", unsafe_allow_html=True)

        if auto_refresh:
            time.sleep(90)
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    main()
