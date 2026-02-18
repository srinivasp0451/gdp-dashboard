"""
╔══════════════════════════════════════════════════════════════════════╗
║        ALPHA EDGE — Professional Trading Strategy Platform v2       ║
║   Nifty | BankNifty | Sensex | Stocks | BTC | ETH | Forex | Gold   ║
╚══════════════════════════════════════════════════════════════════════╝

Install Requirements:
    pip install streamlit yfinance pandas numpy plotly scipy ta requests

Run:
    streamlit run trading_platform.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import math
from scipy.stats import norm
import time
import requests
import json
import threading

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlphaEdge Trading Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
defaults = {
    "theme":              "White",
    "trading_active":     False,
    "trade_history":      [],
    "open_positions":     {},
    "live_realized_pnl":  0.0,
    "live_unrealized_pnl":0.0,
    "paper_capital":      100000.0,
    "last_signal":        None,
    "last_fetch_time":    {},
    "bt_run":             False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# THEME — read BEFORE injecting CSS
# ─────────────────────────────────────────────────────────────────────────────
THEME = st.session_state["theme"]

if THEME == "White":
    T = {
        "BG":           "#FFFFFF",
        "CARD_BG":      "#F3F4F6",
        "CARD_BG2":     "#E9ECF0",
        "BORDER":       "#D1D5DB",
        "TEXT":         "#111827",
        "TEXT_MUTED":   "#374151",
        "TEXT_FAINT":   "#6B7280",
        "INPUT_BG":     "#FFFFFF",
        "INPUT_TEXT":   "#111827",
        "INPUT_BORDER": "#9CA3AF",
        "SIDEBAR_BG":   "#F9FAFB",
        "SIDEBAR_BORDER":"#E5E7EB",
        "PLOT_BG":      "#FFFFFF",
        "PLOT_PAPER":   "#F3F4F6",
        "GRID":         "#E5E7EB",
        "ACCENT":       "#F59E0B",
        "GREEN":        "#059669",
        "RED":          "#DC2626",
        "BLUE":         "#2563EB",
        "PURPLE":       "#7C3AED",
        "ORANGE":       "#EA580C",
        "METRIC_VAL":   "#1D4ED8",
    }
else:
    T = {
        "BG":           "#0A0E1A",
        "CARD_BG":      "#111827",
        "CARD_BG2":     "#1F2937",
        "BORDER":       "#374151",
        "TEXT":         "#F9FAFB",
        "TEXT_MUTED":   "#D1D5DB",   # much brighter than before
        "TEXT_FAINT":   "#9CA3AF",   # still visible on dark
        "INPUT_BG":     "#1F2937",
        "INPUT_TEXT":   "#F9FAFB",
        "INPUT_BORDER": "#4B5563",
        "SIDEBAR_BG":   "#0D1117",
        "SIDEBAR_BORDER":"#1F2937",
        "PLOT_BG":      "#111827",
        "PLOT_PAPER":   "#0A0E1A",
        "GRID":         "#1F2937",
        "ACCENT":       "#F59E0B",
        "GREEN":        "#10B981",
        "RED":          "#EF4444",
        "BLUE":         "#60A5FA",
        "PURPLE":       "#A78BFA",
        "ORANGE":       "#FB923C",
        "METRIC_VAL":   "#F59E0B",
    }

# ─────────────────────────────────────────────────────────────────────────────
# CSS INJECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, html, body, [class*="css"] {{
    font-family: 'Space Grotesk', sans-serif !important;
}}
.stApp, .main, .block-container {{
    background-color: {T['BG']} !important;
    color: {T['TEXT']} !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {T['SIDEBAR_BG']} !important;
    border-right: 1px solid {T['SIDEBAR_BORDER']};
}}
[data-testid="stSidebar"] * {{
    color: {T['TEXT']} !important;
}}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] small {{
    color: {T['TEXT_MUTED']} !important;
}}

/* ── All text ── */
p, span, div, label, h1, h2, h3, h4, h5, h6,
.stMarkdown, .stText {{
    color: {T['TEXT']} !important;
}}
small {{ color: {T['TEXT_MUTED']} !important; }}

/* ── INPUT FIELDS — fix white-on-white / invisible text ── */
input, textarea {{
    background-color: {T['INPUT_BG']} !important;
    color: {T['INPUT_TEXT']} !important;
    border: 1px solid {T['INPUT_BORDER']} !important;
    border-radius: 6px !important;
}}
input:focus, textarea:focus {{
    border-color: {T['ACCENT']} !important;
    box-shadow: 0 0 0 2px {T['ACCENT']}30 !important;
}}
/* Number inputs */
.stNumberInput input {{
    background: {T['INPUT_BG']} !important;
    color: {T['INPUT_TEXT']} !important;
}}
/* Select boxes */
.stSelectbox > div > div {{
    background: {T['INPUT_BG']} !important;
    color: {T['INPUT_TEXT']} !important;
    border: 1px solid {T['INPUT_BORDER']} !important;
}}
.stSelectbox svg {{ fill: {T['TEXT_MUTED']} !important; }}
/* Dropdown options */
[data-baseweb="popover"] *, [data-baseweb="menu"] * {{
    background: {T['CARD_BG']} !important;
    color: {T['TEXT']} !important;
}}
[data-baseweb="option"]:hover {{
    background: {T['CARD_BG2']} !important;
}}
/* Slider */
.stSlider > div > div > div {{
    background: {T['ACCENT']} !important;
}}
.stSlider [data-baseweb="slider"] [role="slider"] {{
    background: {T['ACCENT']} !important;
    border-color: {T['ACCENT']} !important;
}}
/* Radio */
.stRadio > div, .stCheckbox > div {{
    color: {T['TEXT']} !important;
}}
.stRadio label, .stCheckbox label {{
    color: {T['TEXT']} !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {T['CARD_BG']};
    border-radius: 12px;
    padding: 4px;
    border: 1px solid {T['BORDER']};
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    color: {T['TEXT_MUTED']} !important;
    font-weight: 600;
    font-size: 14px;
    padding: 8px 20px;
}}
.stTabs [aria-selected="true"] {{
    background: {T['ACCENT']} !important;
    color: #000 !important;
    font-weight: 700;
}}
.stTabs [data-testid="stTabPanel"] {{
    background: transparent !important;
}}

/* ── Metric cards ── */
[data-testid="metric-container"] {{
    background: {T['CARD_BG']} !important;
    border: 1px solid {T['BORDER']};
    border-radius: 10px;
    padding: 12px 16px;
}}
[data-testid="stMetricLabel"] {{ color: {T['TEXT_MUTED']} !important; font-size:12px; }}
[data-testid="stMetricValue"] {{
    color: {T['METRIC_VAL']} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 22px !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricDelta"] {{ font-size: 12px !important; }}

/* ── Dataframe ── */
.stDataFrame, .stDataFrame * {{
    background: {T['CARD_BG']} !important;
    color: {T['TEXT']} !important;
}}
.stDataFrame thead th {{
    background: {T['CARD_BG2']} !important;
    color: {T['ACCENT']} !important;
    font-weight: 700 !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background: linear-gradient(135deg, {T['ACCENT']}, #D97706) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 20px {T['ACCENT']}60 !important;
}}

/* ── Custom cards ── */
.alpha-card {{
    background: {T['CARD_BG']};
    border: 1px solid {T['BORDER']};
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
    color: {T['TEXT']};
}}
.section-hdr {{
    font-size: 18px;
    font-weight: 700;
    color: {T['ACCENT']};
    border-bottom: 2px solid {T['BORDER']};
    padding-bottom: 8px;
    margin: 18px 0 14px;
}}
.tf-badge {{
    display:inline-block;
    background:{T['CARD_BG2']};
    color:{T['ACCENT']};
    border:1px solid {T['ACCENT']}60;
    border-radius:6px;
    padding:2px 10px;
    font-size:11px;
    font-weight:700;
    letter-spacing:0.5px;
    font-family:'JetBrains Mono',monospace;
}}
.pnl-pos {{ color: {T['GREEN']} !important; font-weight:700; }}
.pnl-neg {{ color: {T['RED']} !important; font-weight:700; }}

/* ── Info / warning ── */
.stAlert, [data-testid="stAlert"] {{
    background: {T['CARD_BG']} !important;
    color: {T['TEXT']} !important;
    border-color: {T['BORDER']} !important;
}}

/* ── Expander ── */
.streamlit-expanderHeader {{
    background: {T['CARD_BG']} !important;
    color: {T['TEXT']} !important;
}}

/* ── Progress bar ── */
.stProgress > div > div {{ background: {T['ACCENT']} !important; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def card(html_body: str) -> str:
    return f'<div class="alpha-card">{html_body}</div>'

def badge(text: str) -> str:
    return f'<span class="tf-badge">{text}</span>'

def section(title: str) -> None:
    st.markdown(f'<div class="section-hdr">{title}</div>', unsafe_allow_html=True)

def color_val(v, pos_label="", neg_label=""):
    cls = "pnl-pos" if v >= 0 else "pnl-neg"
    sign = "+" if v >= 0 else ""
    return f'<span class="{cls}">{sign}{v}{pos_label if v>=0 else neg_label}</span>'

# ─────────────────────────────────────────────────────────────────────────────
# ASSET UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────
ASSET_MAP = {
    "Nifty 50":            "^NSEI",
    "BankNifty":           "^NSEBANK",
    "Sensex":              "^BSESN",
    "Nifty IT":            "^CNXIT",
    "Nifty Pharma":        "^CNXPHARMA",
    "Nifty Midcap 100":    "^CNXMIDCAP",
    "Nifty Smallcap 100":  "NIFTY_SMLCAP100.NS",
    "Reliance Industries": "RELIANCE.NS",
    "TCS":                 "TCS.NS",
    "HDFC Bank":           "HDFCBANK.NS",
    "Infosys":             "INFY.NS",
    "ICICI Bank":          "ICICIBANK.NS",
    "Bajaj Finance":       "BAJFINANCE.NS",
    "Tata Motors":         "TATAMOTORS.NS",
    "Adani Ports":         "ADANIPORTS.NS",
    "SBI":                 "SBIN.NS",
    "Wipro":               "WIPRO.NS",
    "Maruti Suzuki":       "MARUTI.NS",
    "L&T":                 "LT.NS",
    "HCL Technologies":    "HCLTECH.NS",
    "Bitcoin (BTC)":       "BTC-USD",
    "Ethereum (ETH)":      "ETH-USD",
    "Solana (SOL)":        "SOL-USD",
    "USD/INR":             "USDINR=X",
    "EUR/USD":             "EURUSD=X",
    "GBP/USD":             "GBPUSD=X",
    "USD/JPY":             "USDJPY=X",
    "Gold":                "GC=F",
    "Silver":              "SI=F",
    "Crude Oil (WTI)":     "CL=F",
    "Natural Gas":         "NG=F",
    "Custom Ticker":       "CUSTOM",
}

ASSET_GROUPS = {
    "🇮🇳 Indian Indices":    ["Nifty 50","BankNifty","Sensex","Nifty IT","Nifty Pharma","Nifty Midcap 100","Nifty Smallcap 100"],
    "📈 Indian Stocks":      ["Reliance Industries","TCS","HDFC Bank","Infosys","ICICI Bank","Bajaj Finance","Tata Motors","Adani Ports","SBI","Wipro","Maruti Suzuki","L&T","HCL Technologies"],
    "₿ Crypto":              ["Bitcoin (BTC)","Ethereum (ETH)","Solana (SOL)"],
    "💱 Forex":               ["USD/INR","EUR/USD","GBP/USD","USD/JPY"],
    "🥇 Commodities":         ["Gold","Silver","Crude Oil (WTI)","Natural Gas"],
    "⚙️ Custom":              ["Custom Ticker"],
}

STRATEGIES = {
    "Trend + Structure + Momentum (Pro)": "TSM",
    "ORB — Opening Range Breakout":       "ORB",
    "VWAP + RSI Reversal/Trend":          "VWAP_RSI",
    "Swing: EMA + MACD + RSI":            "SWING",
    "Combined (All Signals)":             "COMBINED",
}

# NSE symbol mapping for option chains
NSE_OC_MAP = {
    "^NSEI":    "NIFTY",
    "^NSEBANK": "BANKNIFTY",
    "^BSESN":   "SENSEX",
}

# ─────────────────────────────────────────────────────────────────────────────
# RATE-LIMITED DATA FETCHERS
# ─────────────────────────────────────────────────────────────────────────────
FETCH_DELAY = 1.5   # seconds between yfinance calls

def _wait_rate_limit(key: str):
    """Enforce min delay between repeated calls for same key."""
    now = time.time()
    last = st.session_state["last_fetch_time"].get(key, 0)
    elapsed = now - last
    if elapsed < FETCH_DELAY:
        time.sleep(FETCH_DELAY - elapsed)
    st.session_state["last_fetch_time"][key] = time.time()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    try:
        _wait_rate_limit(f"hist_{ticker}")
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False, threads=False)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_live(ticker: str, interval: str = "15m", period: str = "5d") -> pd.DataFrame:
    try:
        _wait_rate_limit(f"live_{ticker}")
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False, threads=False)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=30, show_spinner=False)
def fetch_live_price(ticker: str) -> float | None:
    try:
        _wait_rate_limit(f"price_{ticker}")
        t = yf.Ticker(ticker)
        info = t.fast_info
        return float(info.last_price)
    except Exception:
        try:
            df = fetch_live(ticker, interval="1m", period="1d")
            return float(df["Close"].iloc[-1]) if not df.empty else None
        except Exception:
            return None

@st.cache_data(ttl=120, show_spinner=False)
def fetch_ticker_info(ticker: str) -> dict:
    try:
        _wait_rate_limit(f"info_{ticker}")
        t = yf.Ticker(ticker)
        return t.info or {}
    except Exception:
        return {}

# ─────────────────────────────────────────────────────────────────────────────
# OPTION CHAIN FETCHER — NSE (Indian) + yfinance (Global)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=120, show_spinner=False)
def fetch_nse_option_chain(nse_symbol: str) -> dict:
    """Fetch live option chain from NSE India public API."""
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
    }
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=8)
        time.sleep(1.5)
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={nse_symbol}"
        resp = session.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data
        # Try equities endpoint for stocks
        url2 = f"https://www.nseindia.com/api/option-chain-equities?symbol={nse_symbol}"
        resp2 = session.get(url2, headers=headers, timeout=10)
        if resp2.status_code == 200:
            return resp2.json()
    except Exception:
        pass
    return {}

@st.cache_data(ttl=120, show_spinner=False)
def fetch_yf_option_chain(ticker: str) -> dict:
    """Fetch option chain from yfinance (works for US stocks, some global)."""
    try:
        _wait_rate_limit(f"oc_{ticker}")
        t = yf.Ticker(ticker)
        expiries = t.options
        if not expiries:
            return {}
        results = {}
        for exp in expiries[:3]:    # max 3 expiries
            time.sleep(FETCH_DELAY)
            try:
                chain = t.option_chain(exp)
                results[exp] = {
                    "calls": chain.calls.to_dict("records"),
                    "puts":  chain.puts.to_dict("records"),
                }
            except Exception:
                continue
        return results
    except Exception:
        return {}

def parse_nse_oc(data: dict, spot: float) -> pd.DataFrame:
    """Parse NSE option chain JSON into a clean DataFrame around ATM."""
    rows = []
    try:
        records = data.get("records", {}).get("data", [])
        for rec in records:
            strike = rec.get("strikePrice", 0)
            ce = rec.get("CE", {})
            pe = rec.get("PE", {})
            rows.append({
                "Strike":    strike,
                "CE_LTP":    ce.get("lastPrice", 0),
                "CE_IV":     ce.get("impliedVolatility", 0),
                "CE_OI":     ce.get("openInterest", 0),
                "CE_ChgOI":  ce.get("changeinOpenInterest", 0),
                "CE_Delta":  ce.get("delta", ""),
                "CE_Theta":  ce.get("theta", ""),
                "CE_Gamma":  ce.get("gamma", ""),
                "CE_Vega":   ce.get("vega", ""),
                "PE_LTP":    pe.get("lastPrice", 0),
                "PE_IV":     pe.get("impliedVolatility", 0),
                "PE_OI":     pe.get("openInterest", 0),
                "PE_ChgOI":  pe.get("changeinOpenInterest", 0),
                "PE_Delta":  pe.get("delta", ""),
                "PE_Theta":  pe.get("theta", ""),
                "PE_Gamma":  pe.get("gamma", ""),
                "PE_Vega":   pe.get("vega", ""),
            })
    except Exception:
        pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("Strike")
    # Filter to ±10 strikes around ATM
    df["ATM_Dist"] = (df["Strike"] - spot).abs()
    df = df.nsmallest(21, "ATM_Dist").sort_values("Strike")
    return df.drop("ATM_Dist", axis=1)

def compute_implied_vol(market_price: float, S: float, K: float,
                        T: float, r: float, opt_type: str = "CE") -> float:
    """Newton-Raphson implied volatility solver."""
    if T <= 0 or market_price <= 0:
        return 0.20
    sigma = 0.20
    for _ in range(200):
        price, delta, gamma, theta, vega = bs_price(S, K, T, r, sigma, opt_type)
        diff = price - market_price
        if abs(diff) < 0.001:
            break
        # vega is per 1% change, convert to per unit change
        vega_unit = vega * 100
        if abs(vega_unit) < 1e-8:
            break
        sigma = sigma - diff / vega_unit
        sigma = max(0.001, min(sigma, 5.0))
    return round(sigma, 4)

def get_live_iv(ticker: str, spot: float, days_to_exp: int = 7) -> float:
    """Get live IV: try NSE first, then compute from HV, fallback 0.18."""
    nse_sym = NSE_OC_MAP.get(ticker)
    if nse_sym:
        try:
            oc_data = fetch_nse_option_chain(nse_sym)
            atm_k   = round(spot / 100) * 100
            records = oc_data.get("records", {}).get("data", [])
            for rec in records:
                if rec.get("strikePrice") == atm_k:
                    iv = rec.get("CE", {}).get("impliedVolatility", 0)
                    if iv and iv > 0:
                        return iv / 100   # NSE gives IV as percentage
        except Exception:
            pass
    # Fallback: compute HV from recent price data
    try:
        df = fetch_data(ticker, period="3mo", interval="1d")
        if not df.empty and len(df) > 20:
            log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
            hv = log_ret.std() * np.sqrt(252)
            return round(float(hv), 4)
    except Exception:
        pass
    return 0.18  # market average fallback

# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS  (unchanged core logic)
# ─────────────────────────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for p in [9, 20, 50, 200]:
        d[f"EMA{p}"] = d["Close"].ewm(span=p, adjust=False).mean()
    d["SMA20"] = d["Close"].rolling(20).mean()
    d["SMA50"] = d["Close"].rolling(50).mean()

    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"]        = ema12 - ema26
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_Hist"]   = d["MACD"] - d["MACD_Signal"]

    d["BB_Mid"]   = d["Close"].rolling(20).mean()
    std20         = d["Close"].rolling(20).std()
    d["BB_Upper"] = d["BB_Mid"] + 2 * std20
    d["BB_Lower"] = d["BB_Mid"] - 2 * std20
    d["BB_Width"] = (d["BB_Upper"] - d["BB_Lower"]) / d["BB_Mid"].replace(0, np.nan)

    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - d["Close"].shift()).abs(),
        (d["Low"]  - d["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()

    d["TP"]   = (d["High"] + d["Low"] + d["Close"]) / 3
    vol_safe  = d["Volume"].replace(0, np.nan)
    d["VWAP"] = (d["TP"] * vol_safe).rolling(14).sum() / vol_safe.rolling(14).sum()

    d["Vol_MA20"] = d["Volume"].rolling(20).mean()
    d["Vol_Ratio"] = d["Volume"] / d["Vol_MA20"].replace(0, np.nan)

    low14  = d["Low"].rolling(14).min()
    high14 = d["High"].rolling(14).max()
    rng    = (high14 - low14).replace(0, np.nan)
    d["Stoch_K"] = 100 * (d["Close"] - low14) / rng
    d["Stoch_D"] = d["Stoch_K"].rolling(3).mean()

    d["Pivot"] = (d["High"] + d["Low"] + d["Close"]) / 3
    d["R1"]    = 2 * d["Pivot"] - d["Low"]
    d["S1"]    = 2 * d["Pivot"] - d["High"]
    d["R2"]    = d["Pivot"] + (d["High"] - d["Low"])
    d["S2"]    = d["Pivot"] - (d["High"] - d["Low"])

    d["Body"]     = (d["Close"] - d["Open"]).abs()
    d["Range"]    = d["High"] - d["Low"]
    d["UpCandle"] = d["Close"] > d["Open"]

    # Historical Volatility (20-day)
    log_ret    = np.log(d["Close"] / d["Close"].shift(1))
    d["HV20"]  = log_ret.rolling(20).std() * np.sqrt(252) * 100

    return d

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGIES  (all 5 — unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────
def strategy_tsm(df, atr_mult=1.5, rr=2.0, trail_sl=True, trail_pct=1.5):
    d = df.copy()
    bull = (d["Close"] > d["EMA20"]) & (d["EMA20"] > d["EMA50"])
    bear = (d["Close"] < d["EMA20"]) & (d["EMA20"] < d["EMA50"])
    vol  = d["Vol_Ratio"] >= 1.5
    buy  = bull & (d["High"] > d["High"].shift(1)) & vol & d["RSI"].between(50,75) & (d["MACD"] > d["MACD_Signal"])
    sel  = bear & (d["Low"]  < d["Low"].shift(1))  & vol & d["RSI"].between(25,50) & (d["MACD"] < d["MACD_Signal"])
    d["Signal"] = 0
    d.loc[buy, "Signal"] =  1
    d.loc[sel, "Signal"] = -1
    d["Strategy"] = "TSM"
    return d

def strategy_orb(df, atr_mult=1.0, rr=1.5, trail_sl=True, trail_pct=1.2):
    d = df.copy()
    d["Signal"] = 0
    vol = d["Vol_Ratio"] > 1.8
    buy = (d["High"] > d["High"].shift(1)) & vol & (d["Close"] > d["EMA20"]) & (d["RSI"] > 50)
    sel = (d["Low"]  < d["Low"].shift(1))  & vol & (d["Close"] < d["EMA20"]) & (d["RSI"] < 50)
    d.loc[buy, "Signal"] =  1
    d.loc[sel, "Signal"] = -1
    d["Strategy"] = "ORB"
    return d

def strategy_vwap_rsi(df, atr_mult=1.2, rr=1.8, trail_sl=True, trail_pct=1.0):
    d = df.copy()
    d["Signal"] = 0
    near_vwap_buy = (d["Low"] <= d["VWAP"] * 1.003) & (d["Close"] > d["VWAP"] * 0.997)
    near_vwap_sel = (d["High"] >= d["VWAP"] * 0.997) & (d["Close"] < d["VWAP"] * 1.003)
    buy = (d["Close"] > d["VWAP"]) & near_vwap_buy & d["RSI"].between(40,60) & (d["EMA20"] > d["EMA50"])
    sel = (d["Close"] < d["VWAP"]) & near_vwap_sel & d["RSI"].between(40,60) & (d["EMA20"] < d["EMA50"])
    d.loc[buy, "Signal"] =  1
    d.loc[sel, "Signal"] = -1
    d["Strategy"] = "VWAP_RSI"
    return d

def strategy_swing(df, atr_mult=2.0, rr=2.5, trail_sl=True, trail_pct=2.0):
    d = df.copy()
    d["Signal"] = 0
    gc = (d["EMA20"] > d["EMA50"]) & (d["EMA20"].shift(1) <= d["EMA50"].shift(1))
    dc = (d["EMA20"] < d["EMA50"]) & (d["EMA20"].shift(1) >= d["EMA50"].shift(1))
    mu = (d["MACD"] > d["MACD_Signal"]) & (d["MACD"].shift(1) <= d["MACD_Signal"].shift(1))
    md = (d["MACD"] < d["MACD_Signal"]) & (d["MACD"].shift(1) >= d["MACD_Signal"].shift(1))
    vol = d["Vol_Ratio"] > 1.3
    buy = (gc | mu) & d["RSI"].between(45,70) & vol
    sel = (dc | md) & d["RSI"].between(30,55) & vol
    d.loc[buy, "Signal"] =  1
    d.loc[sel, "Signal"] = -1
    d["Strategy"] = "SWING"
    return d

def strategy_combined(df, atr_mult=1.5, rr=2.0, trail_sl=True, trail_pct=1.5):
    votes = (strategy_tsm(df, atr_mult, rr, trail_sl, trail_pct)["Signal"] +
             strategy_orb(df, atr_mult, rr, trail_sl, trail_pct)["Signal"] +
             strategy_vwap_rsi(df, atr_mult, rr, trail_sl, trail_pct)["Signal"] +
             strategy_swing(df, atr_mult, rr, trail_sl, trail_pct)["Signal"])
    d = df.copy()
    d["Signal"] = 0
    d.loc[votes >= 2,  "Signal"] =  1
    d.loc[votes <= -2, "Signal"] = -1
    d["Strategy"] = "COMBINED"
    return d

def get_strategy_df(df, key, atr_sl, rr, trail_sl, trail_pct):
    fn = {"TSM": strategy_tsm, "ORB": strategy_orb, "VWAP_RSI": strategy_vwap_rsi,
          "SWING": strategy_swing, "COMBINED": strategy_combined}
    return fn[key](df, atr_sl, rr, trail_sl, trail_pct)

# ─────────────────────────────────────────────────────────────────────────────
# SL / TARGET CALCULATOR  (supports ATR / Points / Percentage)
# ─────────────────────────────────────────────────────────────────────────────
def calc_sl_target(price: float, atr: float, direction: int,
                   sl_type: str, sl_val: float, rr: float) -> tuple:
    """
    Returns (sl, target1) based on selected SL type.
    direction: +1 = long, -1 = short
    sl_type: 'ATR-based' | 'Points-based' | 'Percentage-based'
    """
    if sl_type == "Points-based":
        dist = sl_val
    elif sl_type == "Percentage-based":
        dist = price * sl_val / 100
    else:  # ATR-based
        dist = sl_val * atr

    sl  = round(price - direction * dist, 2)
    tgt = round(price + direction * dist * rr, 2)
    return sl, tgt

# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTER  (unchanged core + SL type support)
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df_sig, capital, risk_pct, atr_sl, rr,
                 trail_sl, trail_pct, trail_tgt, trail_tgt_pct,
                 sl_type="ATR-based", sl_val=1.5) -> dict:
    trades, equity, eq_dates = [], [capital], [df_sig.index[0]]
    cash = capital
    pos = 0; entry_px = sl = tgt = trail_sl_px = trail_tgt_px = 0.0
    entry_date = None

    df = df_sig.dropna(subset=["ATR", "Signal"]).copy()

    for i in range(1, len(df)):
        row  = df.iloc[i]
        price = float(row["Close"])
        atr   = max(float(row["ATR"]), price * 0.001)

        if pos != 0:
            # update trailing SL
            if trail_sl:
                if pos == 1:
                    trail_sl_px = max(trail_sl_px, price * (1 - trail_pct / 100))
                    eff_sl = max(sl, trail_sl_px)
                else:
                    trail_sl_px = min(trail_sl_px, price * (1 + trail_pct / 100))
                    eff_sl = min(sl, trail_sl_px)
            else:
                eff_sl = sl

            # update trailing target
            if trail_tgt:
                if pos == 1:
                    trail_tgt_px = max(trail_tgt_px, price * (1 + trail_tgt_pct / 100))
                else:
                    trail_tgt_px = min(trail_tgt_px, price * (1 - trail_tgt_pct / 100))

            exit_px = exit_rsn = None
            eff_tgt = (trail_tgt_px if (trail_tgt and trail_tgt_px) else tgt)

            if pos == 1:
                if row["Low"]  <= eff_sl:  exit_px = eff_sl;  exit_rsn = "Trail SL" if trail_sl else "SL"
                elif row["High"] >= eff_tgt: exit_px = eff_tgt; exit_rsn = "Trail Tgt" if trail_tgt else "Target"
            else:
                if row["High"] >= eff_sl:  exit_px = eff_sl;  exit_rsn = "Trail SL" if trail_sl else "SL"
                elif row["Low"]  <= eff_tgt: exit_px = eff_tgt; exit_rsn = "Trail Tgt" if trail_tgt else "Target"

            if exit_px is None and row["Signal"] == -pos:
                exit_px = price; exit_rsn = "Signal Rev"

            if exit_px is not None:
                pnl_pct = (exit_px - entry_px) / entry_px * pos * 100
                risk_amt = cash * risk_pct / 100
                qty      = risk_amt / (abs(entry_px - sl) + 1e-9)
                pnl_abs  = (exit_px - entry_px) * pos * qty
                cash    += pnl_abs
                trades.append({
                    "Entry Date": entry_date, "Exit Date": row.name,
                    "Direction":  "LONG" if pos == 1 else "SHORT",
                    "Entry": round(entry_px, 2), "Exit": round(exit_px, 2),
                    "SL": round(sl, 2), "Target": round(tgt, 2),
                    "P&L %": round(pnl_pct, 2), "P&L ₹": round(pnl_abs, 2),
                    "Capital": round(cash, 2), "Exit Reason": exit_rsn,
                    "Strategy": row.get("Strategy", ""),
                })
                pos = 0; trail_sl_px = trail_tgt_px = 0.0

        if pos == 0 and row["Signal"] != 0:
            pos = int(row["Signal"]); entry_px = price; entry_date = row.name
            sl, tgt = calc_sl_target(price, atr, pos, sl_type, sl_val, rr)
            trail_sl_px  = price * (1 - trail_pct/100) if pos == 1 else price * (1 + trail_pct/100)
            trail_tgt_px = tgt

        equity.append(cash); eq_dates.append(row.name)

    if not trades:
        return {"trades": pd.DataFrame(), "equity": pd.Series(equity, index=eq_dates),
                "metrics": {}, "drawdown": pd.Series([0] * len(equity), index=eq_dates)}

    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["P&L %"] > 0]; losses = tdf[tdf["P&L %"] <= 0]
    win_rate = len(wins) / len(tdf) * 100
    pf = wins["P&L ₹"].sum() / abs(losses["P&L ₹"].sum()) if losses["P&L ₹"].sum() < 0 else float("inf")
    eq_s = pd.Series(equity, index=eq_dates)
    dd   = (eq_s - eq_s.cummax()) / eq_s.cummax() * 100
    n_days = max((df.index[-1] - df.index[0]).days, 1)
    cagr = ((cash / capital) ** (365 / n_days) - 1) * 100
    sharpe = 0.0
    if len(tdf) > 1:
        r = tdf["P&L %"] / 100
        sharpe = r.mean() / (r.std() + 1e-9) * np.sqrt(252)

    return {
        "trades":  tdf,
        "equity":  eq_s,
        "drawdown": dd,
        "metrics": {
            "Total Trades":   len(tdf),
            "Win Rate %":     round(win_rate, 1),
            "Avg Win %":      round(wins["P&L %"].mean() if len(wins) else 0, 2),
            "Avg Loss %":     round(losses["P&L %"].mean() if len(losses) else 0, 2),
            "Profit Factor":  round(pf, 2),
            "Max Drawdown %": round(dd.min(), 2),
            "Total Return %": round((cash - capital) / capital * 100, 2),
            "CAGR %":         round(cagr, 2),
            "Sharpe Ratio":   round(sharpe, 2),
            "Net P&L":        round(cash - capital, 2),
            "Final Capital":  round(cash, 2),
        }
    }

# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES + GREEKS
# ─────────────────────────────────────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, opt_type="CE"):
    if T <= 0 or sigma <= 0:
        intrinsic = max(S-K, 0) if opt_type == "CE" else max(K-S, 0)
        return intrinsic, 0, 0, 0, 0
    d1  = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2  = d1 - sigma*math.sqrt(T)
    if opt_type == "CE":
        price = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        delta = -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S*sigma*math.sqrt(T))
    theta = (-(S*norm.pdf(d1)*sigma)/(2*math.sqrt(T)) -
             r*K*math.exp(-r*T)*(norm.cdf(d2) if opt_type=="CE" else norm.cdf(-d2))) / 365
    vega  = S*norm.pdf(d1)*math.sqrt(T)/100
    return round(price,2), round(delta,4), round(gamma,6), round(theta,2), round(vega,4)

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS ENGINE  (enhanced with live data)
# ─────────────────────────────────────────────────────────────────────────────
def generate_analysis(df, ticker, asset_name, sl_type="ATR-based",
                      sl_val=1.5, rr=2.0, capital=100000, risk_pct=2.0) -> dict:
    if df is None or len(df) < 50:
        return {}
    cur  = df.iloc[-1]; prev = df.iloc[-2]
    price  = float(cur["Close"])
    atr    = float(cur["ATR"])
    rsi    = float(cur["RSI"])
    macd   = float(cur["MACD"]); macd_s = float(cur["MACD_Signal"])
    ema20  = float(cur["EMA20"]); ema50 = float(cur["EMA50"])
    ema200 = float(cur["EMA200"])
    vwap   = float(cur["VWAP"])
    vol_r  = float(cur["Vol_Ratio"])
    bb_u   = float(cur["BB_Upper"]); bb_l = float(cur["BB_Lower"])
    bb_w   = float(cur["BB_Width"])
    pivot  = float(cur["Pivot"])
    r1 = float(cur["R1"]); s1 = float(cur["S1"])
    r2 = float(cur["R2"]); s2 = float(cur["S2"])
    stoch  = float(cur["Stoch_K"])
    hv20   = float(cur.get("HV20", 18.0))

    # Trend score
    ts, tn = 0, []
    checks = [
        (price > ema20,   2, "✅ Price above EMA20",              "❌ Price below EMA20"),
        (price > ema50,   2, "✅ Price above EMA50",              "❌ Price below EMA50"),
        (price > ema200,  3, "✅ Price above EMA200 (Bull trend)","❌ Price below EMA200 (Bear trend)"),
        (ema20 > ema50,   2, "✅ EMA20 > EMA50 (Bullish cross)",  "❌ EMA20 < EMA50 (Bearish)"),
        (price > vwap,    1, "✅ Price above VWAP",               "❌ Price below VWAP"),
    ]
    for cond, pts, yes, no in checks:
        if cond: ts += pts; tn.append(yes)
        else:    tn.append(no)

    # Momentum score
    ms, mn = 0, []
    if   50 < rsi < 70:  ms += 3; mn.append(f"✅ RSI {rsi:.1f} — Healthy bullish")
    elif rsi >= 70:      ms += 1; mn.append(f"⚠️ RSI {rsi:.1f} — Overbought")
    elif 30 < rsi <= 50: mn.append(f"⚠️ RSI {rsi:.1f} — Weak/bearish")
    else:                ms -= 1; mn.append(f"❌ RSI {rsi:.1f} — Oversold")

    if macd > macd_s:    ms += 3; mn.append("✅ MACD bullish")
    else:                mn.append("❌ MACD bearish")

    if   vol_r > 1.5: ms += 2; mn.append(f"✅ Volume {vol_r:.1f}x avg — Strong")
    elif vol_r > 1.0: ms += 1; mn.append(f"⚠️ Volume {vol_r:.1f}x avg — Average")
    else:             mn.append(f"❌ Volume {vol_r:.1f}x avg — Weak")

    if   20 < stoch < 80: ms += 2; mn.append(f"✅ Stoch {stoch:.1f} — Not extreme")
    elif stoch > 80:      mn.append(f"⚠️ Stoch {stoch:.1f} — Overbought zone")
    else:                 mn.append(f"⚠️ Stoch {stoch:.1f} — Oversold zone")

    combined = ts + ms
    bias_map = [
        (14, "STRONG BUY"), (10, "BUY"), (7, "WEAK BUY"), (4, "NEUTRAL"),
        (1, "WEAK SELL"), (-2, "SELL"), (-99, "STRONG SELL"),
    ]
    bias = next(b for thr, b in bias_map if combined >= thr)
    is_long = "BUY" in bias
    strength = "🟢" if "BUY" in bias else ("🔴" if "SELL" in bias else "⚪")

    # SL / Target
    sl_fixed, tgt1 = calc_sl_target(price, atr, 1 if is_long else -1,
                                     sl_type, sl_val, rr)
    _, tgt2 = calc_sl_target(price, atr, 1 if is_long else -1,
                              sl_type, sl_val, rr * 1.5)
    _, tgt3 = calc_sl_target(price, atr, 1 if is_long else -1,
                              sl_type, sl_val, rr * 2.0)
    sl_atr  = round(price - (1 if is_long else -1) * 1.5 * atr, 2)

    trail_sl_start = round(price - atr if is_long else price + atr, 2)

    # Risk calculations
    sl_dist  = abs(price - sl_fixed)
    risk_amt = capital * risk_pct / 100
    qty_est  = int(risk_amt / sl_dist) if sl_dist > 0 else 0

    # Live IV (tries NSE API, HV fallback)
    iv_live = get_live_iv(ticker, price)
    iv_pct  = round(iv_live * 100, 1)
    iv_rank_str = ("✅ LOW — Good to buy options" if iv_pct < 15
                   else "⚠️ MEDIUM — Buy ATM only" if iv_pct < 25
                   else "❌ HIGH — Options overpriced, avoid naked buys")

    # Option strike suggestion
    tick_step   = 100 if price > 5000 else (50 if price > 1000 else 10)
    atm_strike  = round(price / tick_step) * tick_step
    opt_rec     = ("CE" if is_long else "PE")
    itm_strike  = (atm_strike - tick_step if is_long else atm_strike + tick_step)

    levels = {
        "EMA200": round(ema200, 2), "EMA50": round(ema50, 2),
        "EMA20":  round(ema20, 2),  "VWAP":  round(vwap, 2),
        "Pivot":  round(pivot, 2),
        "R1": round(r1, 2), "R2": round(r2, 2),
        "S1": round(s1, 2), "S2": round(s2, 2),
        "BB Upper": round(bb_u, 2), "BB Lower": round(bb_l, 2),
    }

    return {
        "price": price, "bias": bias, "strength": strength,
        "trend_score": ts, "mom_score": ms, "combined": combined,
        "trend_notes": tn, "mom_notes": mn,
        "entry": round(price, 2), "sl_fixed": sl_fixed, "sl_atr": sl_atr,
        "tgt1": tgt1, "tgt2": tgt2, "tgt3": tgt3,
        "trail_sl_start": trail_sl_start,
        "sl_dist": round(sl_dist, 2), "qty_est": qty_est,
        "atr": round(atr, 2), "rsi": round(rsi, 2), "macd": round(macd, 4),
        "vol_ratio": round(vol_r, 2), "hv20": round(hv20, 1),
        "iv_live": iv_live, "iv_pct": iv_pct, "iv_rank_str": iv_rank_str,
        "bb_width": round(bb_w, 4), "stoch_k": round(stoch, 1),
        "levels": levels, "is_long": is_long,
        "opt_rec": opt_rec, "atm_strike": atm_strike, "itm_strike": itm_strike,
        "risk_amt": risk_amt, "qty_est": qty_est,
    }

# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def build_price_chart(df, signals_df=None, analysis=None, title="",
                      timeframe_label="") -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        vertical_spacing=0.02,
        subplot_titles=(
            f"{title} {timeframe_label}",
            "Volume", "RSI (14)", "MACD (12,26,9)"
        )
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color=T["GREEN"], decreasing_line_color=T["RED"],
        name="OHLC"
    ), row=1, col=1)

    # EMAs
    for ema_col, clr in [("EMA20", T["BLUE"]), ("EMA50", T["ACCENT"]), ("EMA200", T["PURPLE"])]:
        if ema_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[ema_col], name=ema_col,
                line=dict(color=clr, width=1.5), opacity=0.9
            ), row=1, col=1)

    # BB
    if "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"],
            line=dict(color="rgba(139,92,246,0.5)", width=1, dash="dot"),
            name="BB Upper", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"],
            line=dict(color="rgba(139,92,246,0.5)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(139,92,246,0.06)",
            name="BB", showlegend=False), row=1, col=1)

    # VWAP
    if "VWAP" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"],
            line=dict(color=T["ORANGE"], width=1.5, dash="dash"), name="VWAP"
        ), row=1, col=1)

    # Signals
    if signals_df is not None and "Signal" in signals_df.columns:
        for sig_val, sym, clr, lbl in [(1, "triangle-up", T["GREEN"], "BUY"),
                                        (-1, "triangle-down", T["RED"], "SELL")]:
            s = signals_df[signals_df["Signal"] == sig_val]
            if len(s):
                y_vals = s["Low"] * 0.994 if sig_val == 1 else s["High"] * 1.006
                fig.add_trace(go.Scatter(
                    x=s.index, y=y_vals, mode="markers",
                    marker=dict(symbol=sym, size=11, color=clr,
                                line=dict(color="white", width=1)),
                    name=f"{lbl} Signal"
                ), row=1, col=1)

    # Analysis levels
    if analysis:
        levels_plot = [
            ("Entry",  analysis["entry"],    T["ACCENT"],  "dash"),
            ("SL",     analysis["sl_fixed"], T["RED"],     "dot"),
            ("Tgt 1",  analysis["tgt1"],     T["GREEN"],   "dash"),
            ("Tgt 2",  analysis["tgt2"],     T["GREEN"],   "dashdot"),
            ("Tgt 3",  analysis["tgt3"],     T["GREEN"],   "longdash"),
        ]
        for lbl, val, clr, dash in levels_plot:
            fig.add_hline(y=val, line_color=clr, line_dash=dash, line_width=1.5,
                          annotation_text=f"  {lbl}: {val}",
                          annotation_font_color=clr, row=1, col=1)

    # Volume
    vol_colors = [T["GREEN"] if c >= o else T["RED"]
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
        marker_color=vol_colors, name="Volume", opacity=0.7), row=2, col=1)
    if "Vol_MA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["Vol_MA20"],
            line=dict(color=T["ACCENT"], width=1.2), name="Vol MA20"), row=2, col=1)

    # RSI
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"],
            line=dict(color=T["PURPLE"], width=1.5), name="RSI"), row=3, col=1)
        for lv, clr in [(70, T["RED"]), (50, T["ACCENT"]), (30, T["GREEN"])]:
            fig.add_hline(y=lv, line_color=clr, line_dash="dot",
                          line_width=1, row=3, col=1)

    # MACD
    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],
            line=dict(color=T["BLUE"], width=1.5), name="MACD"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"],
            line=dict(color=T["RED"], width=1.2), name="Signal"), row=4, col=1)
        hist_colors = [T["GREEN"] if v >= 0 else T["RED"] for v in df["MACD_Hist"]]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"],
            marker_color=hist_colors, name="Hist", opacity=0.6), row=4, col=1)

    fig.update_layout(
        paper_bgcolor=T["PLOT_PAPER"], plot_bgcolor=T["PLOT_BG"],
        font=dict(color=T["TEXT"], size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=T["TEXT"])),
        xaxis_rangeslider_visible=False, height=720,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor=T["GRID"], showgrid=True,
                         tickfont=dict(color=T["TEXT"]), row=i, col=1)
        fig.update_yaxes(gridcolor=T["GRID"], showgrid=True,
                         tickfont=dict(color=T["TEXT"]), row=i, col=1)

    return fig

def build_equity_chart(equity, drawdown, trade_df) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.04,
                        subplot_titles=("Equity Curve", "Drawdown %"))
    fig.add_trace(go.Scatter(x=equity.index, y=equity,
        fill="tozeroy", fillcolor="rgba(96,165,250,0.12)",
        line=dict(color=T["BLUE"], width=2), name="Portfolio Value"), row=1, col=1)
    if trade_df is not None and len(trade_df):
        wins = trade_df[trade_df["P&L %"] > 0]
        losses = trade_df[trade_df["P&L %"] <= 0]
        if len(wins):
            fig.add_trace(go.Scatter(x=wins["Exit Date"], y=wins["Capital"],
                mode="markers", marker=dict(symbol="circle", size=7, color=T["GREEN"]),
                name="Win"), row=1, col=1)
        if len(losses):
            fig.add_trace(go.Scatter(x=losses["Exit Date"], y=losses["Capital"],
                mode="markers", marker=dict(symbol="x", size=7, color=T["RED"]),
                name="Loss"), row=1, col=1)
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown,
        fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
        line=dict(color=T["RED"], width=1.5), name="Drawdown %"), row=2, col=1)
    fig.update_layout(paper_bgcolor=T["PLOT_PAPER"], plot_bgcolor=T["PLOT_BG"],
        font=dict(color=T["TEXT"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=T["TEXT"])),
        height=450, margin=dict(l=10, r=10, t=40, b=10))
    for i in range(1, 3):
        fig.update_xaxes(gridcolor=T["GRID"], tickfont=dict(color=T["TEXT"]), row=i, col=1)
        fig.update_yaxes(gridcolor=T["GRID"], tickfont=dict(color=T["TEXT"]), row=i, col=1)
    return fig

def build_gauge(val, max_val, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        title={"text": title, "font": {"color": T["TEXT"], "size": 13}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": T["TEXT_MUTED"]},
            "bar": {"color": color}, "bgcolor": T["CARD_BG"],
            "bordercolor": T["BORDER"],
            "steps": [
                {"range": [0, max_val*.33], "color": "rgba(220,38,38,0.18)"},
                {"range": [max_val*.33, max_val*.66], "color": "rgba(245,158,11,0.18)"},
                {"range": [max_val*.66, max_val],  "color": "rgba(16,185,129,0.18)"},
            ],
        },
        number={"font": {"color": color, "family": "JetBrains Mono"}}
    ))
    fig.update_layout(paper_bgcolor=T["CARD_BG"], height=200,
                      margin=dict(l=20, r=20, t=40, b=10),
                      font_color=T["TEXT_MUTED"])
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# LIVE TRADING ENGINE  (paper-trade simulation)
# ─────────────────────────────────────────────────────────────────────────────
def process_live_signal(signal: int, price: float, atr: float,
                        strategy: str, asset: str,
                        sl_type: str, sl_val: float, rr: float,
                        trail_pct: float, capital: float, risk_pct: float):
    """Process a new signal: open/close paper positions, update P&L."""
    positions = st.session_state["open_positions"]
    history   = st.session_state["trade_history"]
    key = f"{asset}_{strategy}"

    # Check if we have an open position for this asset/strategy
    if key in positions:
        pos = positions[key]
        # Close if opposite signal or SL/TGT hit will be checked at next refresh
        if signal == -pos["direction"]:
            # Close position
            pnl_pts  = (price - pos["entry"]) * pos["direction"]
            pnl_pct  = pnl_pts / pos["entry"] * 100
            risk_amt = capital * risk_pct / 100
            qty      = risk_amt / (abs(pos["entry"] - pos["sl"]) + 1e-9)
            pnl_inr  = pnl_pts * qty * pos["direction"]
            st.session_state["live_realized_pnl"] += pnl_inr
            history.append({
                "Time":       datetime.now().strftime("%H:%M:%S"),
                "Date":       datetime.now().strftime("%Y-%m-%d"),
                "Asset":      asset,
                "Strategy":   strategy,
                "Direction":  "LONG" if pos["direction"] == 1 else "SHORT",
                "Entry":      round(pos["entry"], 2),
                "Exit":       round(price, 2),
                "SL":         round(pos["sl"], 2),
                "Target":     round(pos["target"], 2),
                "P&L %":      round(pnl_pct, 2),
                "P&L ₹":      round(pnl_inr, 2),
                "Reason":     "Signal Rev",
                "Status":     "✅ WIN" if pnl_inr > 0 else "❌ LOSS",
            })
            del positions[key]

    if key not in positions and signal != 0:
        sl, tgt = calc_sl_target(price, atr, signal, sl_type, sl_val, rr)
        positions[key] = {
            "direction": signal, "entry": price, "sl": sl, "target": tgt,
            "asset": asset, "strategy": strategy,
            "entry_time": datetime.now().strftime("%H:%M:%S"),
            "trail_high": price, "trail_pct": trail_pct,
        }

def update_unrealized_pnl(current_prices: dict, capital: float, risk_pct: float):
    """Recalculate unrealized P&L for all open positions."""
    total_unreal = 0.0
    positions = st.session_state["open_positions"]
    for key, pos in positions.items():
        cur_px = current_prices.get(pos["asset"], pos["entry"])
        pnl_pts = (cur_px - pos["entry"]) * pos["direction"]
        risk_amt = capital * risk_pct / 100
        qty      = risk_amt / (abs(pos["entry"] - pos["sl"]) + 1e-9)
        total_unreal += pnl_pts * qty * pos["direction"]

        # Update trailing SL
        if pos["direction"] == 1 and cur_px > pos.get("trail_high", cur_px):
            positions[key]["trail_high"] = cur_px
            new_tsl = cur_px * (1 - pos["trail_pct"] / 100)
            positions[key]["sl"] = max(pos["sl"], new_tsl)
        elif pos["direction"] == -1 and cur_px < pos.get("trail_high", cur_px):
            positions[key]["trail_high"] = cur_px
            new_tsl = cur_px * (1 + pos["trail_pct"] / 100)
            positions[key]["sl"] = min(pos["sl"], new_tsl)

    st.session_state["live_unrealized_pnl"] = total_unreal

# ─────────────────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # THEME SELECTOR at the very top
    st.markdown(f"""
    <div style="text-align:center;padding:16px 0 8px">
        <span style="font-size:22px;font-weight:800;
            background:linear-gradient(135deg,{T['ACCENT']},#FCD34D);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            ⚡ AlphaEdge
        </span><br>
        <span style="font-size:10px;color:{T['TEXT_MUTED']};
            letter-spacing:3px;text-transform:uppercase">Pro Trading v2</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Theme radio
    theme_choice = st.radio("🎨 Theme", ["White", "Dark"],
                             index=0 if THEME == "White" else 1,
                             horizontal=True)
    if theme_choice != st.session_state["theme"]:
        st.session_state["theme"] = theme_choice
        st.rerun()

    st.markdown("---")
    st.markdown(f"**📊 Asset Selection**")

    group_sel = st.selectbox("Asset Group", list(ASSET_GROUPS.keys()))
    assets_in_group = ASSET_GROUPS[group_sel]
    asset_name_sel  = st.selectbox("Asset", assets_in_group)
    ticker_raw      = ASSET_MAP.get(asset_name_sel, "^NSEI")

    if ticker_raw == "CUSTOM":
        ticker = st.text_input("Custom Ticker (Yahoo Finance)", "AAPL",
                               help="e.g. AAPL, TSLA, NIFTY.NS, BTC-USD")
    else:
        ticker = ticker_raw

    st.markdown("---")
    st.markdown("**🧠 Strategy & Timeframe**")

    strategy_name = st.selectbox("Strategy", list(STRATEGIES.keys()), index=4)
    strategy_key  = STRATEGIES[strategy_name]

    trade_type = st.selectbox("Trade Type", [
        "Intraday (15m)", "Swing (Daily)", "Positional (Weekly)"
    ], index=0)
    TF_MAP = {
        "Intraday (15m)":       ("6mo",  "1d",  "15m",  "5d"),
        "Swing (Daily)":        ("2y",   "1d",  "1d",   "60d"),
        "Positional (Weekly)":  ("5y",   "1wk", "1d",   "90d"),
    }
    bt_period, bt_interval, live_interval, live_period = TF_MAP[trade_type]

    st.markdown(f"""
    <small style="color:{T['TEXT_MUTED']}">
    Backtest: <b style="color:{T['ACCENT']}">{bt_period}</b> @
    <b style="color:{T['ACCENT']}">{bt_interval}</b> &nbsp;|&nbsp;
    Live: <b style="color:{T['ACCENT']}">{live_period}</b> @
    <b style="color:{T['ACCENT']}">{live_interval}</b>
    </small>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚙️ Risk Parameters**")

    capital    = st.number_input("Capital (₹)", 10000, 50000000, 100000, step=5000)
    risk_pct   = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5)
    rr_ratio   = st.slider("Risk : Reward", 1.0, 5.0, 2.0, 0.25,
                            help="1:X — if SL is 50pts, target is 50*X pts")

    st.markdown("---")
    st.markdown("**🎯 Stop Loss Type**")

    sl_type = st.selectbox("SL Calculation Method", [
        "ATR-based", "Points-based", "Percentage-based"
    ], help="Choose how SL distance is calculated")

    if sl_type == "ATR-based":
        atr_sl   = st.slider("ATR Multiplier (SL)", 0.5, 4.0, 1.5, 0.25)
        sl_val   = atr_sl
        sl_label = f"{atr_sl}× ATR"
    elif sl_type == "Points-based":
        sl_pts   = st.number_input("SL in Points", 5, 5000, 50, step=5)
        sl_val   = float(sl_pts)
        sl_label = f"{sl_pts} pts"
    else:  # Percentage
        sl_pct   = st.number_input("SL %", 0.1, 20.0, 1.0, step=0.1)
        sl_val   = sl_pct
        sl_label = f"{sl_pct}%"

    st.markdown("---")
    st.markdown("**🔄 Trailing Settings**")
    c1_, c2_ = st.columns(2)
    with c1_:
        trail_sl  = st.checkbox("Trail SL",  True)
        trail_tgt = st.checkbox("Trail Tgt", True)
    with c2_:
        trail_pct_val = st.number_input("Trail SL %",  0.3, 10.0, 1.5, 0.25)
        trail_tgt_val = st.number_input("Trail Tgt %", 0.3, 10.0, 2.0, 0.25)

    st.markdown("---")
    st.markdown("**📊 Options Settings**")
    days_exp   = st.slider("Days to Expiry", 1, 90, 7, 1)
    risk_free  = st.slider("Risk-Free Rate %", 4.0, 9.0, 6.5, 0.5)
    opt_type   = st.radio("Option Type", ["CE (Call)", "PE (Put)"])

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True)
    if run_btn:
        st.session_state["bt_run"] = True
        st.cache_data.clear()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊  Backtesting",
    "⚡  Live Trading",
    "🔭  Analyse & Recommend",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — BACKTESTING
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    section("📊 Strategy Backtesting Engine")

    # Timeframe badge row
    st.markdown(
        f"Asset: {badge(asset_name_sel)} &nbsp;"
        f"Strategy: {badge(strategy_name)} &nbsp;"
        f"Period: {badge(bt_period)} &nbsp;"
        f"Interval: {badge(bt_interval)} &nbsp;"
        f"SL Type: {badge(sl_type + ' — ' + sl_label)} &nbsp;"
        f"R:R: {badge('1:'+str(rr_ratio))} &nbsp;"
        f"Capital: {badge('₹'+str(f'{capital:,}'))}",
        unsafe_allow_html=True
    )
    st.markdown("")

    if st.session_state.get("bt_run"):
        prog = st.progress(0, text="Fetching historical data...")
        with st.spinner(""):
            raw_df = fetch_data(ticker, period=bt_period, interval=bt_interval)
        prog.progress(40, text="Computing indicators...")

        if raw_df.empty:
            st.error("❌ Could not fetch data. Verify ticker symbol and internet connection.")
            prog.empty()
        else:
            df_ind = compute_indicators(raw_df)
            prog.progress(70, text="Generating signals...")
            df_sig = get_strategy_df(df_ind, strategy_key, sl_val, rr_ratio,
                                      trail_sl, trail_pct_val)
            prog.progress(90, text="Running backtest...")
            result = run_backtest(
                df_sig, capital, risk_pct, sl_val, rr_ratio,
                trail_sl, trail_pct_val, trail_tgt, trail_tgt_val,
                sl_type, sl_val
            )
            prog.progress(100, text="Done!")
            prog.empty()

            m = result.get("metrics", {})
            if not m:
                st.warning("⚠️ No trades generated. Try a longer period or different strategy.")
            else:
                # Data range info
                dr_start = raw_df.index[0].strftime("%d %b %Y")
                dr_end   = raw_df.index[-1].strftime("%d %b %Y")
                st.info(f"📅 Data range: **{dr_start}** to **{dr_end}** | "
                        f"**{len(raw_df)}** candles | Interval: **{bt_interval}** | "
                        f"SL method: **{sl_type}** ({sl_label}) | R:R **1:{rr_ratio}**")

                # Metric grid
                c1,c2,c3,c4,c5,c6 = st.columns(6)
                with c1: st.metric("Total Trades",   m["Total Trades"])
                with c2: st.metric("Win Rate",        f"{m['Win Rate %']}%")
                with c3: st.metric("Profit Factor",   m["Profit Factor"])
                with c4: st.metric("Max Drawdown",    f"{m['Max Drawdown %']}%")
                with c5: st.metric("CAGR",            f"{m['CAGR %']}%")
                with c6: st.metric("Sharpe Ratio",    m["Sharpe Ratio"])

                c7,c8,c9,c10 = st.columns(4)
                with c7: st.metric("Net P&L",     f"₹{m['Net P&L']:,.0f}")
                with c8: st.metric("Total Return", f"{m['Total Return %']}%")
                with c9: st.metric("Avg Win",      f"{m['Avg Win %']}%")
                with c10:st.metric("Avg Loss",     f"{m['Avg Loss %']}%")

                st.markdown("---")

                # Price chart
                price_fig = build_price_chart(
                    df_ind.tail(250), df_sig.tail(250), None,
                    f"{asset_name_sel} — {strategy_name}",
                    f"[{bt_interval} | {bt_period}]"
                )
                st.plotly_chart(price_fig, use_container_width=True)

                # Equity + drawdown
                eq_fig = build_equity_chart(result["equity"], result["drawdown"],
                                             result["trades"])
                st.plotly_chart(eq_fig, use_container_width=True)

                # Trade log
                section("📋 Trade Log")
                tdf = result["trades"].copy()
                tdf["P&L %"] = tdf["P&L %"].apply(
                    lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%")
                st.dataframe(tdf[[
                    "Entry Date","Exit Date","Direction","Entry","Exit",
                    "SL","Target","P&L %","P&L ₹","Capital","Exit Reason","Strategy"
                ]].tail(60), use_container_width=True, height=320)

                # Distribution charts
                tdf_raw = result["trades"]
                cc1, cc2 = st.columns(2)
                with cc1:
                    exits_cnt = tdf_raw["Exit Reason"].value_counts().reset_index()
                    pie = px.pie(exits_cnt, names="Exit Reason", values="count",
                                  title="Exit Reason Distribution",
                                  color_discrete_sequence=[T["GREEN"],T["RED"],T["ACCENT"],
                                                            T["BLUE"],T["PURPLE"]])
                    pie.update_layout(paper_bgcolor=T["PLOT_PAPER"],
                                      font_color=T["TEXT"], height=320)
                    st.plotly_chart(pie, use_container_width=True)
                with cc2:
                    hist = px.histogram(tdf_raw, x="P&L %", nbins=30,
                                         title="P&L Distribution",
                                         color_discrete_sequence=[T["BLUE"]])
                    hist.update_layout(paper_bgcolor=T["PLOT_PAPER"],
                                       plot_bgcolor=T["PLOT_BG"],
                                       font_color=T["TEXT"], height=320)
                    st.plotly_chart(hist, use_container_width=True)

                # Strategy breakdown
                if "Strategy" in tdf_raw.columns and tdf_raw["Strategy"].nunique() > 1:
                    section("📊 Strategy Breakdown")
                    sb = tdf_raw.groupby("Strategy").agg(
                        Trades=("P&L %","count"),
                        Win_Rate=("P&L %", lambda x: f"{(x>0).mean()*100:.1f}%"),
                        Avg_PnL=("P&L %","mean"),
                        Total_PnL=("P&L ₹","sum"),
                    ).reset_index()
                    st.dataframe(sb, use_container_width=True)
    else:
        st.info("👈 Configure parameters in the sidebar and click **🚀 Run Analysis** to start.")
        # Show quick instructions
        st.markdown(f"""
        <div class="alpha-card">
        <b style="color:{T['ACCENT']}">Quick Guide — Backtesting Tab</b><br><br>
        1. Select <b>Asset Group + Asset</b> in sidebar<br>
        2. Pick a <b>Strategy</b> (Combined recommended for best accuracy)<br>
        3. Choose <b>Trade Type</b> (Intraday/Swing/Positional)<br>
        4. Set <b>SL Type</b>: ATR-based (dynamic), Points-based (fixed pts), or Percentage<br>
        5. Configure <b>Trailing SL %</b> and <b>Trailing Target %</b><br>
        6. Click <b>🚀 Run Analysis</b><br><br>
        <small style="color:{T['TEXT_MUTED']}">Backtest uses historical OHLCV data from Yahoo Finance.
        Results include full trade log, equity curve, win rate, Sharpe, CAGR, drawdown.</small>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE TRADING
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    section("⚡ Live Trading Dashboard")

    # ── Control panel ─────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1,1,1,3])
    with ctrl1:
        if st.button("▶ Start Trading",
                     disabled=st.session_state["trading_active"],
                     use_container_width=True):
            st.session_state["trading_active"] = True
            st.session_state["paper_capital"]  = float(capital)
            st.session_state["live_realized_pnl"] = 0.0
            st.session_state["live_unrealized_pnl"] = 0.0
            st.success("✅ Trading Started")

    with ctrl2:
        if st.button("⏹ Stop Trading",
                     disabled=not st.session_state["trading_active"],
                     use_container_width=True):
            st.session_state["trading_active"]  = False
            st.info("⏹ Trading Stopped")

    with ctrl3:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with ctrl4:
        status_clr = T["GREEN"] if st.session_state["trading_active"] else T["RED"]
        status_txt = "🟢 LIVE — Paper Trading Active" if st.session_state["trading_active"] else "🔴 Stopped"
        st.markdown(f"""
        <div style="background:{T['CARD_BG']};border:1px solid {status_clr}40;
                    border-radius:8px;padding:8px 16px;margin-top:4px">
            <span style="color:{status_clr};font-weight:700">{status_txt}</span>
            &nbsp;|&nbsp;
            <span style="color:{T['TEXT_MUTED']};font-size:12px">
            Paper Capital: <b style="color:{T['TEXT']}">₹{st.session_state['paper_capital']:,.0f}</b>
            &nbsp;|&nbsp;
            Interval: <b style="color:{T['ACCENT']}">{live_interval}</b>
            &nbsp;|&nbsp;
            Period: <b style="color:{T['ACCENT']}">{live_period}</b>
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.info(f"📡 Data: Yahoo Finance (delayed ~15-20 min for Indian markets) | "
            f"Rate limiting: {FETCH_DELAY}s between calls | "
            f"Timeframe: **{live_interval}** | Period fetched: **{live_period}**")

    # ── Fetch live data ────────────────────────────────────────────────────
    with st.spinner(f"Fetching live data for {asset_name_sel} [{live_interval}]..."):
        live_df_raw = fetch_live(ticker, interval=live_interval, period=live_period)

    if live_df_raw.empty:
        st.error("❌ Could not fetch live data. Check ticker and internet connection.")
    else:
        live_df  = compute_indicators(live_df_raw)
        live_sig = get_strategy_df(live_df, strategy_key, sl_val, rr_ratio,
                                    trail_sl, trail_pct_val)
        an = generate_analysis(live_df, ticker, asset_name_sel,
                                sl_type, sl_val, rr_ratio, capital, risk_pct)

        if not an:
            st.warning("Insufficient data for live analysis.")
        else:
            cur_price = an["price"]

            # Auto-process signal if trading is active
            if st.session_state["trading_active"]:
                latest_sig = int(live_sig["Signal"].iloc[-1])
                latest_atr = float(live_sig["ATR"].iloc[-1])
                if latest_sig != 0:
                    process_live_signal(
                        latest_sig, cur_price, latest_atr,
                        strategy_key, asset_name_sel,
                        sl_type, sl_val, rr_ratio,
                        trail_pct_val, capital, risk_pct
                    )

            # Update unrealized P&L
            update_unrealized_pnl({asset_name_sel: cur_price}, capital, risk_pct)

            # ── Live P&L header ───────────────────────────────────────────
            realized   = st.session_state["live_realized_pnl"]
            unrealized = st.session_state["live_unrealized_pnl"]
            total_pnl  = realized + unrealized
            open_pos_count = len(st.session_state["open_positions"])

            pc1,pc2,pc3,pc4,pc5,pc6 = st.columns(6)
            price_prev = float(live_df["Close"].iloc[-2]) if len(live_df) > 1 else cur_price
            price_chg  = cur_price - price_prev
            price_pct  = price_chg / price_prev * 100 if price_prev else 0

            with pc1: st.metric("📍 LTP", f"₹{cur_price:,.2f}",
                                 f"{price_chg:+.2f} ({price_pct:+.2f}%)")
            with pc2: st.metric("💰 Realized P&L",
                                 f"₹{realized:+,.0f}",
                                 delta_color="normal" if realized >= 0 else "inverse")
            with pc3: st.metric("📊 Unrealized P&L",
                                 f"₹{unrealized:+,.0f}",
                                 delta_color="normal" if unrealized >= 0 else "inverse")
            with pc4: st.metric("🏦 Total P&L",     f"₹{total_pnl:+,.0f}")
            with pc5: st.metric("📂 Open Positions", open_pos_count)
            with pc6: st.metric("🧮 RSI / ATR",
                                 f"{an['rsi']:.1f} / {an['atr']:.1f}")

            st.markdown("---")

            # ── Signal box ────────────────────────────────────────────────
            sig_cls = ("signal-bull" if "BUY" in an["bias"]
                       else "signal-bear" if "SELL" in an["bias"]
                       else "signal-neutral")
            sig_border = T["GREEN"] if "BUY" in an["bias"] else (
                         T["RED"] if "SELL" in an["bias"] else T["ACCENT"])
            sig_bg = (f"{T['GREEN']}12" if "BUY" in an["bias"] else
                      f"{T['RED']}12" if "SELL" in an["bias"] else
                      f"{T['ACCENT']}12")
            st.markdown(f"""
            <div style="background:{sig_bg};border:1px solid {sig_border}50;
                        border-left:5px solid {sig_border};border-radius:10px;
                        padding:16px 22px;margin:8px 0">
                <span style="font-size:24px;font-weight:800;color:{sig_border}">
                    {an['strength']} {an['bias']}
                </span>
                &nbsp;&nbsp;
                <span style="color:{T['TEXT_MUTED']};font-size:13px">
                    Strategy: <b style="color:{T['ACCENT']}">{strategy_name}</b>
                    &nbsp;|&nbsp; Trend: <b>{an['trend_score']}/10</b>
                    &nbsp;|&nbsp; Momentum: <b>{an['mom_score']}/10</b>
                    &nbsp;|&nbsp; Combined: <b>{an['combined']}/20</b>
                    &nbsp;|&nbsp; HV20: <b>{an['hv20']}%</b>
                    &nbsp;|&nbsp; IV: <b>{an['iv_pct']}%</b>
                    &nbsp;|&nbsp; Timeframe: <b>{live_interval}</b>
                </span>
            </div>
            """, unsafe_allow_html=True)

            # ── Entry / SL / Target row ────────────────────────────────────
            section("🎯 Entry | SL | Targets")
            ec1,ec2,ec3,ec4,ec5,ec6,ec7 = st.columns(7)
            with ec1: st.metric("🟡 Entry",     f"₹{an['entry']:,.2f}")
            with ec2: st.metric("🔴 Fixed SL",  f"₹{an['sl_fixed']:,.2f}",
                                 f"−{an['sl_dist']:.1f} pts")
            with ec3: st.metric("🟠 ATR SL",    f"₹{an['sl_atr']:,.2f}",
                                 f"ATR={an['atr']:.1f}")
            with ec4: st.metric("🟢 Target 1",  f"₹{an['tgt1']:,.2f}",
                                 f"+{abs(an['tgt1']-an['entry']):.1f} pts")
            with ec5: st.metric("🟢 Target 2",  f"₹{an['tgt2']:,.2f}",
                                 f"+{abs(an['tgt2']-an['entry']):.1f} pts")
            with ec6: st.metric("🟢 Target 3",  f"₹{an['tgt3']:,.2f}",
                                 f"+{abs(an['tgt3']-an['entry']):.1f} pts")
            with ec7: st.metric("📐 R:R",       f"1:{rr_ratio}")

            # Trailing info
            trl1, trl2 = st.columns(2)
            with trl1:
                st.markdown(f"""
                <div class="alpha-card">
                <b style="color:{T['ACCENT']}">🔄 Trailing Stop Loss</b>
                &nbsp; {badge(sl_type + ' — ' + sl_label)}<br><br>
                Start at: <b>₹{an['trail_sl_start']:,.2f}</b> &nbsp;|&nbsp;
                Trail %: <b>{trail_pct_val}%</b> below peak<br><br>
                <span style="color:{T['TEXT_MUTED']};font-size:12px">
                Once price moves {trail_pct_val}% in your favor, SL locks in gains.
                If price reverses {trail_pct_val}% from peak → auto-exit.
                Recalculates every candle.
                </span>
                </div>
                """, unsafe_allow_html=True)
            with trl2:
                st.markdown(f"""
                <div class="alpha-card">
                <b style="color:{T['ACCENT']}">🎯 Trailing Target</b>
                &nbsp; {badge('Trail ' + str(trail_tgt_val) + '% above tgt')}<br><br>
                Initial: <b>₹{an['tgt1']:,.2f}</b> &nbsp;|&nbsp;
                Trail: <b>{trail_tgt_val}%</b> above price<br><br>
                <span style="color:{T['TEXT_MUTED']};font-size:12px">
                Book 50% at Target 1. Trail remaining {trail_tgt_val}% behind price.
                Exit rest if momentum fails (RSI &lt; 45 + MACD cross).
                </span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Open Positions ─────────────────────────────────────────────
            if st.session_state["open_positions"]:
                section("📂 Open Positions")
                pos_rows = []
                for key_, pos_ in st.session_state["open_positions"].items():
                    cp = cur_price if pos_["asset"] == asset_name_sel else pos_["entry"]
                    pnl_pts = (cp - pos_["entry"]) * pos_["direction"]
                    risk_a  = capital * risk_pct / 100
                    qty_    = risk_a / (abs(pos_["entry"] - pos_["sl"]) + 1e-9)
                    upnl    = pnl_pts * qty_ * pos_["direction"]
                    pos_rows.append({
                        "Asset":     pos_["asset"],
                        "Strategy":  pos_["strategy"],
                        "Direction": "LONG ▲" if pos_["direction"]==1 else "SHORT ▼",
                        "Entry":     pos_["entry"],
                        "Current":   round(cp, 2),
                        "SL (Trail)":round(pos_["sl"], 2),
                        "Target":    round(pos_["target"], 2),
                        "Entry Time":pos_["entry_time"],
                        "Unreal P&L":f"₹{upnl:+,.0f}",
                        "P&L %":     f"{pnl_pts/pos_['entry']*100*pos_['direction']:+.2f}%",
                    })
                st.dataframe(pd.DataFrame(pos_rows), use_container_width=True)

            # ── Live chart ─────────────────────────────────────────────────
            section(f"📈 Live Chart — {asset_name_sel}  [{live_interval} | {live_period}]")
            live_fig = build_price_chart(
                live_df.tail(120), live_sig.tail(120), an,
                f"⚡ {asset_name_sel}", f"[{live_interval} | {live_period}]"
            )
            st.plotly_chart(live_fig, use_container_width=True)

            # ── Recent signals ─────────────────────────────────────────────
            section("📡 Recent Signals (Last 15)")
            recent = live_sig[live_sig["Signal"] != 0].tail(15).copy()
            if not recent.empty:
                show_cols = [c for c in ["Close","Signal","RSI","MACD","ATR",
                                          "Vol_Ratio","EMA20","EMA50","VWAP"]
                             if c in recent.columns]
                st.dataframe(recent[show_cols].round(2), use_container_width=True)
            else:
                st.info("No signals generated in the current window.")

            # ── Trade History ──────────────────────────────────────────────
            section("📜 Trade History (Paper Trades)")
            if st.session_state["trade_history"]:
                hist_df = pd.DataFrame(st.session_state["trade_history"])
                # Color code
                st.dataframe(hist_df[[
                    "Date","Time","Asset","Strategy","Direction",
                    "Entry","Exit","SL","Target","P&L %","P&L ₹","Reason","Status"
                ]], use_container_width=True, height=280)

                # Summary stats
                if len(hist_df) > 0:
                    wins_h  = hist_df[hist_df["P&L ₹"] > 0]
                    total_r = hist_df["P&L ₹"].sum()
                    wr_h    = len(wins_h) / len(hist_df) * 100
                    sh1,sh2,sh3,sh4 = st.columns(4)
                    with sh1: st.metric("Total Trades",    len(hist_df))
                    with sh2: st.metric("Win Rate",        f"{wr_h:.1f}%")
                    with sh3: st.metric("Total P&L",       f"₹{total_r:+,.0f}")
                    with sh4: st.metric("Avg P&L/Trade",   f"₹{total_r/len(hist_df):+,.0f}")

                if st.button("🗑️ Clear Trade History"):
                    st.session_state["trade_history"] = []
                    st.session_state["live_realized_pnl"] = 0.0
                    st.rerun()
            else:
                st.info("No trade history yet. Start trading to see paper trades appear here.")

            # ── Live Options Pricing ───────────────────────────────────────
            section("⚙️ Options Chain & Live Pricing")

            nse_sym = NSE_OC_MAP.get(ticker)
            opt_t = "CE" if "CE" in opt_type else "PE"
            T_exp = days_exp / 365
            sigma = an["iv_live"]

            # Try NSE option chain first
            with st.spinner("Fetching option chain..."):
                if nse_sym:
                    nse_data = fetch_nse_option_chain(nse_sym)
                    oc_df = parse_nse_oc(nse_data, cur_price)
                else:
                    nse_data = {}
                    oc_df = pd.DataFrame()
                time.sleep(FETCH_DELAY)

            if not oc_df.empty:
                st.success(f"✅ Live NSE Option Chain — {nse_sym} | "
                           f"Spot: ₹{cur_price:,.2f} | IV (ATM): {an['iv_pct']}%")
                st.markdown(f"*Timeframe: Live | Data: NSE India API | "
                            f"Fetched at: {datetime.now().strftime('%H:%M:%S')}*")
                st.dataframe(oc_df, use_container_width=True, height=320)

                # PCR calculation
                total_ce_oi = oc_df["CE_OI"].sum()
                total_pe_oi = oc_df["PE_OI"].sum()
                pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0
                pcr1,pcr2,pcr3 = st.columns(3)
                with pcr1: st.metric("Put-Call Ratio (OI)",  pcr,
                                      "Bullish" if pcr > 1.2 else "Bearish" if pcr < 0.8 else "Neutral")
                with pcr2: st.metric("Total CE OI",  f"{total_ce_oi:,.0f}")
                with pcr3: st.metric("Total PE OI",  f"{total_pe_oi:,.0f}")
            else:
                # Fallback: compute BS for multiple strikes
                st.info(f"ℹ️ Using Black-Scholes pricing (NSE chain not available for {ticker}). "
                        f"IV used: {an['iv_pct']}% (HV-derived) | Days to Expiry: {days_exp}")
                bs_rows = []
                atm_k   = round(cur_price / 100) * 100
                strikes = [atm_k + i*50 for i in range(-6, 7)]
                for k in strikes:
                    p, d, g, th, v = bs_price(cur_price, k, T_exp,
                                               risk_free/100, sigma, opt_t)
                    mon = "ATM" if k == atm_k else (
                          "ITM" if ((opt_t=="CE" and cur_price>k) or
                                    (opt_t=="PE" and cur_price<k)) else "OTM")
                    bs_rows.append({
                        "Strike": k, "Type": opt_t, "Moneyness": mon,
                        "Premium ₹": p, "Delta": d, "Gamma": g,
                        "Theta/day ₹": th, "Vega": v,
                        "Break-even": round(k + p if opt_t=="CE" else k - p, 2),
                    })
                bs_df = pd.DataFrame(bs_rows)
                st.dataframe(bs_df, use_container_width=True)

                # Best recommendation
                best_cands = bs_df[bs_df["Moneyness"].isin(["ATM","ITM"])]
                if not best_cands.empty:
                    best = best_cands.iloc[0]
                    rec_bg  = f"{T['GREEN']}12" if "BUY" in an["bias"] else f"{T['RED']}12"
                    rec_brd = T["GREEN"] if "BUY" in an["bias"] else T["RED"]
                    st.markdown(f"""
                    <div style="background:{rec_bg};border-left:4px solid {rec_brd};
                                border-radius:8px;padding:14px 18px">
                    <b>💡 Recommended Option:</b>
                    {opt_t} Strike <b>{best['Strike']}</b>
                    | Premium: <b>₹{best['Premium ₹']}</b>
                    | Delta: <b>{best['Delta']}</b>
                    | Theta: <b>{best['Theta/day ₹']}/day</b>
                    | Break-even: <b>₹{best['Break-even']}</b><br>
                    <small style="color:{T['TEXT_MUTED']}">
                    IV: {an['iv_pct']}% | Days to Expiry: {days_exp} | {an['iv_rank_str']}
                    </small>
                    </div>
                    """, unsafe_allow_html=True)

            # ── IV + Greeks display ────────────────────────────────────────
            section("📐 Live IV & Greeks Summary")
            gk1,gk2,gk3,gk4,gk5 = st.columns(5)
            p_atm, d_atm, g_atm, th_atm, v_atm = bs_price(
                cur_price, atm_k if nse_sym else round(cur_price/100)*100,
                T_exp, risk_free/100, sigma, opt_t
            )
            with gk1: st.metric("Live IV",        f"{an['iv_pct']}%")
            with gk2: st.metric("ATM Premium",    f"₹{p_atm}")
            with gk3: st.metric("Delta (ATM)",    d_atm)
            with gk4: st.metric("Theta/day",      f"₹{th_atm}")
            with gk5: st.metric("Vega",           v_atm)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYSE & RECOMMEND
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    section("🔭 Deep Analysis & Recommendations")

    with st.spinner(f"Fetching 1-year daily data for {asset_name_sel}..."):
        an_df_raw = fetch_data(ticker, period="1y", interval="1d")

    if an_df_raw.empty:
        st.error("❌ Could not fetch data.")
    else:
        an_df = compute_indicators(an_df_raw)
        an    = generate_analysis(an_df, ticker, asset_name_sel,
                                   sl_type, sl_val, rr_ratio, capital, risk_pct)

        if not an:
            st.warning("Insufficient data.")
        else:
            dr_s = an_df_raw.index[0].strftime("%d %b %Y")
            dr_e = an_df_raw.index[-1].strftime("%d %b %Y")
            st.markdown(
                f"Data: {badge('1Y Daily')} &nbsp;"
                f"Range: {badge(dr_s + ' → ' + dr_e)} &nbsp;"
                f"Candles: {badge(str(len(an_df_raw)))} &nbsp;"
                f"SL: {badge(sl_type + ' — ' + sl_label)} &nbsp;"
                f"R:R: {badge('1:' + str(rr_ratio))}",
                unsafe_allow_html=True
            )
            st.markdown("")

            # ── Top banner ────────────────────────────────────────────────
            dir_c = T["GREEN"] if "BUY" in an["bias"] else T["RED"]
            st.markdown(f"""
            <div style="background:{dir_c}10;border:1px solid {dir_c}40;
                        border-left:6px solid {dir_c};border-radius:14px;
                        padding:24px 28px;margin-bottom:20px">
                <div style="font-size:32px;font-weight:800;color:{dir_c}">
                    {an['strength']} {an['bias']}
                </div>
                <div style="color:{T['TEXT_MUTED']};font-size:14px;margin-top:6px">
                    {asset_name_sel} ({ticker}) &nbsp;·&nbsp;
                    LTP: <b style="color:{T['TEXT']}">₹{an['price']:,.2f}</b> &nbsp;·&nbsp;
                    ATR: <b style="color:{T['TEXT']}">{an['atr']}</b> &nbsp;·&nbsp;
                    RSI: <b style="color:{T['TEXT']}">{an['rsi']}</b> &nbsp;·&nbsp;
                    HV(20): <b style="color:{T['TEXT']}">{an['hv20']}%</b> &nbsp;·&nbsp;
                    IV: <b style="color:{T['TEXT']}">{an['iv_pct']}%</b>
                </div>
                <div style="margin-top:16px;display:flex;gap:40px;flex-wrap:wrap">
                    {''.join([
                        f'<div><div style="color:{T["TEXT_MUTED"]};font-size:10px;'
                        f'text-transform:uppercase;letter-spacing:1px">{lbl}</div>'
                        f'<div style="font-size:22px;font-weight:700;font-family:JetBrains Mono;'
                        f'color:{T["ACCENT"]}">{val}</div></div>'
                        for lbl, val in [
                            ("Trend Score", f"{an['trend_score']}/10"),
                            ("Momentum Score", f"{an['mom_score']}/10"),
                            ("Combined Score", f"{an['combined']}/20"),
                            ("IV Rank", an['iv_rank_str'].split("—")[0].strip()),
                            ("Options Rec", an['opt_rec']),
                        ]
                    ])}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Trade Plan ────────────────────────────────────────────────
            section("📋 Complete Trade Plan")
            tp1, tp2 = st.columns(2)

            with tp1:
                direction_lbl = "▲ LONG / BUY CE" if an["is_long"] else "▼ SHORT / BUY PE"
                dir_c2 = T["GREEN"] if an["is_long"] else T["RED"]
                rows_plan = [
                    ("Direction",           f'<b style="color:{dir_c2}">{direction_lbl}</b>'),
                    ("Entry",               f'<b style="color:{T["ACCENT"]}">₹{an["entry"]:,.2f}</b>'),
                    ("SL Method",           sl_type + " — " + sl_label),
                    ("Stop Loss (Fixed)",   f'<b style="color:{T["RED"]}">₹{an["sl_fixed"]:,.2f} ({an["sl_dist"]:.1f} pts)</b>'),
                    ("Stop Loss (ATR 1.5x)",f'<span style="color:{T["RED"]}">₹{an["sl_atr"]:,.2f}</span>'),
                    ("Target 1 — Book 50%", f'<b style="color:{T["GREEN"]}">₹{an["tgt1"]:,.2f} (+{abs(an["tgt1"]-an["entry"]):.1f} pts)</b>'),
                    ("Target 2 — Book 30%", f'<span style="color:{T["GREEN"]}">₹{an["tgt2"]:,.2f} (+{abs(an["tgt2"]-an["entry"]):.1f} pts)</span>'),
                    ("Target 3 — Hold 20%", f'<span style="color:{T["GREEN"]}">₹{an["tgt3"]:,.2f} (+{abs(an["tgt3"]-an["entry"]):.1f} pts)</span>'),
                    ("Trailing SL",         f'Trail {trail_pct_val}% below peak'),
                    ("Trailing Target",     f'Trail {trail_tgt_val}% above price'),
                    ("Risk Amount",         f'₹{an["risk_amt"]:,.0f} ({risk_pct}% of ₹{capital:,})'),
                    ("Est. Qty (Equity)",   f'{an["qty_est"]} shares'),
                    ("R:R Ratio",           f'<b>1:{abs(an["tgt1"]-an["entry"])/(an["sl_dist"]+1e-9):.1f}</b>'),
                ]
                html_rows = "".join(
                    f'<tr><td style="color:{T["TEXT_MUTED"]};padding:5px 8px 5px 0;'
                    f'font-size:13px;border-bottom:1px solid {T["BORDER"]}20">{r}</td>'
                    f'<td style="font-size:13px;padding:5px 0;border-bottom:1px solid {T["BORDER"]}20">{v}</td></tr>'
                    for r, v in rows_plan
                )
                st.markdown(f"""
                <div class="alpha-card">
                <b style="color:{T['ACCENT']};font-size:15px">🎯 Trade Setup</b><br><br>
                <table style="width:100%;border-collapse:collapse">{html_rows}</table>
                </div>
                """, unsafe_allow_html=True)

            with tp2:
                level_html = ""
                for lv_name, lv_val, lv_clr in [
                    ("EMA 200 (LT trend)", an["levels"]["EMA200"], T["PURPLE"]),
                    ("EMA 50",             an["levels"]["EMA50"],  T["ACCENT"]),
                    ("EMA 20",             an["levels"]["EMA20"],  T["BLUE"]),
                    ("VWAP (14-bar)",       an["levels"]["VWAP"],   T["ORANGE"]),
                    ("Pivot",              an["levels"]["Pivot"],  T["TEXT"]),
                    ("R1",                 an["levels"]["R1"],     T["GREEN"]),
                    ("R2",                 an["levels"]["R2"],     T["GREEN"]),
                    ("S1",                 an["levels"]["S1"],     T["RED"]),
                    ("S2",                 an["levels"]["S2"],     T["RED"]),
                    ("BB Upper",           an["levels"]["BB Upper"], T["PURPLE"]),
                    ("BB Lower",           an["levels"]["BB Lower"], T["PURPLE"]),
                ]:
                    near = " ⬅ Near price" if abs(lv_val - an["price"]) <= an["atr"] else ""
                    level_html += (
                        f'<tr><td style="color:{T["TEXT_MUTED"]};font-size:13px;'
                        f'padding:5px 8px 5px 0;border-bottom:1px solid {T["BORDER"]}20">{lv_name}</td>'
                        f'<td style="color:{lv_clr};font-weight:600;font-size:13px;font-family:JetBrains Mono;'
                        f'padding:5px 0;border-bottom:1px solid {T["BORDER"]}20">'
                        f'₹{lv_val:,.2f}<small style="color:{T["ACCENT"]}">{near}</small></td></tr>'
                    )
                st.markdown(f"""
                <div class="alpha-card">
                <b style="color:{T['ACCENT']};font-size:15px">📐 Key Levels (Live)</b>
                &nbsp; <small style="color:{T['TEXT_MUTED']}">Fetched: 1Y daily data</small><br><br>
                <table style="width:100%;border-collapse:collapse">{level_html}</table>
                </div>
                """, unsafe_allow_html=True)

            # ── Trend + Momentum analysis ─────────────────────────────────
            st.markdown("---")
            ta1, ta2 = st.columns(2)
            with ta1:
                section("📈 Trend Analysis")
                for note in an["trend_notes"]:
                    bg = (f"{T['GREEN']}10" if "✅" in note else
                          f"{T['ACCENT']}10" if "⚠️" in note else f"{T['RED']}10")
                    st.markdown(
                        f'<div style="background:{bg};border-radius:6px;'
                        f'padding:8px 12px;margin:4px 0;font-size:13px;'
                        f'color:{T["TEXT"]}">{note}</div>',
                        unsafe_allow_html=True)

            with ta2:
                section("⚡ Momentum Analysis")
                for note in an["mom_notes"]:
                    bg = (f"{T['GREEN']}10" if "✅" in note else
                          f"{T['ACCENT']}10" if "⚠️" in note else f"{T['RED']}10")
                    st.markdown(
                        f'<div style="background:{bg};border-radius:6px;'
                        f'padding:8px 12px;margin:4px 0;font-size:13px;'
                        f'color:{T["TEXT"]}">{note}</div>',
                        unsafe_allow_html=True)

            # ── Options Recommendation ────────────────────────────────────
            st.markdown("---")
            section("📊 Options Recommendation")
            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                st.markdown(f"""
                <div class="alpha-card">
                <b style="color:{T['ACCENT']}">Which Option to Buy?</b><br><br>
                <b style="color:{dir_c};font-size:16px">{an['opt_rec']}</b><br><br>
                ATM Strike: <b>₹{an['atm_strike']}</b><br>
                ITM Strike: <b>₹{an['itm_strike']}</b>
                (higher Delta, more reliable)<br><br>
                <small style="color:{T['TEXT_MUTED']}">
                Use nearest weekly expiry for intraday,
                next month's expiry for swing (reduces theta decay).
                </small>
                </div>
                """, unsafe_allow_html=True)
            with oc2:
                iv_bg = (f"{T['GREEN']}10" if an['iv_pct'] < 15 else
                         f"{T['ACCENT']}10" if an['iv_pct'] < 25 else f"{T['RED']}10")
                st.markdown(f"""
                <div class="alpha-card">
                <b style="color:{T['ACCENT']}">Live IV Conditions</b><br><br>
                <div style="font-size:24px;font-weight:800;font-family:JetBrains Mono;
                            color:{T['METRIC_VAL']}">{an['iv_pct']}%</div>
                <div style="background:{iv_bg};border-radius:6px;padding:6px 10px;margin:8px 0;font-size:12px">
                    {an['iv_rank_str']}
                </div>
                HV (20-day): <b>{an['hv20']}%</b><br>
                <small style="color:{T['TEXT_MUTED']}">
                IV Source: NSE live API (indices) / HV proxy (stocks/crypto).
                Low IV = cheap options = buy. High IV = sell spreads.
                </small>
                </div>
                """, unsafe_allow_html=True)
            with oc3:
                st.markdown(f"""
                <div class="alpha-card">
                <b style="color:{T['ACCENT']}">Capital Allocation</b><br><br>
                Total Capital: <b>₹{capital:,}</b><br>
                Max/Trade: <b>₹{int(an['risk_amt']):,}</b> ({risk_pct}%)<br>
                Est. Qty: <b>{an['qty_est']}</b> units<br><br>
                <b>Position Sizing Rule:</b><br>
                <small style="color:{T['TEXT_MUTED']}">
                Risk ₹{int(an['risk_amt']):,} on SL of {an['sl_dist']:.1f} pts.
                Book 50% at Tgt1, trail 50% with {trail_pct_val}% TSL.
                Never average a losing option. Max 2 trades/day.
                </small>
                </div>
                """, unsafe_allow_html=True)

            # ── Trailing SL logic explained ───────────────────────────────
            st.markdown("---")
            section("🔄 Trailing SL & Target — Step-by-Step Logic")

            dw = "rises" if an["is_long"] else "falls"
            trail_dir  = "below" if an["is_long"] else "above"
            trail_dir2 = "above" if an["is_long"] else "below"

            tsl1 = round(an['tgt1'] * (1 - trail_pct_val/100 if an['is_long'] else 1 + trail_pct_val/100), 2)
            tsl2 = round(an['tgt2'] * (1 - trail_pct_val/100 if an['is_long'] else 1 + trail_pct_val/100), 2)

            st.markdown(f"""
            <div class="alpha-card">
            <b style="color:{T['ACCENT']};font-size:15px">
            Trailing SL Logic — {sl_type} | {sl_label} | TSL: {trail_pct_val}%
            </b><br><br>
            <ol style="font-size:13px;color:{T['TEXT_MUTED']};line-height:2.2;margin:0;padding-left:18px">
                <li>Enter at <b style="color:{T['ACCENT']}">₹{an['entry']:,.2f}</b>
                    → Initial SL = <b style="color:{T['RED']}">₹{an['sl_fixed']:,.2f}</b>
                    (distance: {an['sl_dist']:.1f} pts via {sl_type})</li>
                <li>Price {dw}s to <b>₹{an['tgt1']:,.2f}</b> (Target 1)
                    → <b style="color:{T['GREEN']}">Book 50% of position</b></li>
                <li>Trail SL moves to <b>₹{tsl1:,.2f}</b>
                    ({trail_pct_val}% {trail_dir} ₹{an['tgt1']:,.2f})</li>
                <li>Price continues to <b>₹{an['tgt2']:,.2f}</b> (Target 2)
                    → <b style="color:{T['GREEN']}">Book 30% more</b></li>
                <li>Trail SL locks to <b>₹{tsl2:,.2f}</b>
                    ({trail_pct_val}% {trail_dir} ₹{an['tgt2']:,.2f})</li>
                <li>Hold remaining 20% → Target 3 = <b>₹{an['tgt3']:,.2f}</b>
                    or exit when TSL triggers</li>
            </ol>
            <br>
            <b style="color:{T['ACCENT']}">Trailing Target Logic:</b><br>
            <span style="font-size:13px;color:{T['TEXT_MUTED']}">
            Once price exceeds Target 1, target "trails" {trail_tgt_val}%
            {trail_dir2} price. This prevents premature exit when momentum is strong.
            Exit only when price reverses {trail_tgt_val}% from the highest point reached,
            OR when RSI crosses below 50 + MACD turns negative simultaneously.
            </span><br><br>
            <b style="color:{T['RED']}">⚡ Hard Stop Rule:</b>
            <span style="font-size:13px;color:{T['TEXT_MUTED']}">
            If price closes {trail_dir} ₹{an['sl_fixed']:,.2f} on the 15-min candle →
            <b>Exit immediately.</b> Capital preservation overrides everything.
            Set GTT orders on your broker so exits happen automatically.
            </span>
            </div>
            """, unsafe_allow_html=True)

            # ── Historical chart ──────────────────────────────────────────
            st.markdown("---")
            an_sig = get_strategy_df(an_df, strategy_key, sl_val, rr_ratio,
                                      trail_sl, trail_pct_val)
            an_fig = build_price_chart(
                an_df.tail(250), an_sig.tail(250), an,
                f"🔭 {asset_name_sel}", "[1D | 1Y — Full Analysis]"
            )
            st.plotly_chart(an_fig, use_container_width=True)

            # ── Gauges ────────────────────────────────────────────────────
            st.markdown("---")
            section("📡 Signal Gauges")
            gc1,gc2,gc3,gc4 = st.columns(4)
            with gc1: st.plotly_chart(build_gauge(an["trend_score"],10,"Trend Score",T["BLUE"]),   use_container_width=True)
            with gc2: st.plotly_chart(build_gauge(an["mom_score"],  10,"Momentum",   T["PURPLE"]), use_container_width=True)
            with gc3: st.plotly_chart(build_gauge(an["rsi"],       100,"RSI",        T["ACCENT"]), use_container_width=True)
            with gc4: st.plotly_chart(build_gauge(min(an["vol_ratio"]*4,20),20,"Volume×Avg",T["GREEN"]), use_container_width=True)

            # ── Multi-asset scan ──────────────────────────────────────────
            st.markdown("---")
            section("🔍 Multi-Asset Scan")

            SCAN_ASSETS = {
                "Nifty 50": "^NSEI", "BankNifty": "^NSEBANK", "Sensex": "^BSESN",
                "Reliance": "RELIANCE.NS", "HDFC Bank": "HDFCBANK.NS",
                "BTC": "BTC-USD", "ETH": "ETH-USD",
                "Gold": "GC=F", "Silver": "SI=F",
                "USD/INR": "USDINR=X",
            }

            if st.button("🔍 Run Multi-Asset Scan (applies 1.5s rate limit per asset)"):
                scan_results = []
                prog_scan = st.progress(0)
                total_s   = len(SCAN_ASSETS)
                for idx_s, (sname, stick) in enumerate(SCAN_ASSETS.items()):
                    prog_scan.progress((idx_s+1)/total_s,
                                        text=f"Scanning {sname} ({idx_s+1}/{total_s})...")
                    sdf = fetch_data(stick, period="3mo", interval="1d")
                    time.sleep(FETCH_DELAY)
                    if sdf.empty:
                        continue
                    sdf_i = compute_indicators(sdf)
                    san   = generate_analysis(sdf_i, stick, sname,
                                               sl_type, sl_val, rr_ratio,
                                               capital, risk_pct)
                    if san:
                        scan_results.append({
                            "Asset":      sname, "Ticker": stick,
                            "Price":      san["price"],
                            "Signal":     san["bias"],
                            "Score":      san["combined"],
                            "RSI":        san["rsi"],
                            "Vol Ratio":  san["vol_ratio"],
                            "ATR":        san["atr"],
                            "IV %":       san["iv_pct"],
                            "HV %":       san["hv20"],
                            "Entry":      san["entry"],
                            "SL":         san["sl_fixed"],
                            "Target 1":   san["tgt1"],
                        })

                prog_scan.empty()
                if scan_results:
                    scan_df = pd.DataFrame(scan_results).sort_values("Score", ascending=False)
                    st.dataframe(scan_df, use_container_width=True)
                    # Top 3 picks
                    st.markdown(f"<b style='color:{T['ACCENT']}'>🏆 Top 3 Opportunities:</b>",
                                unsafe_allow_html=True)
                    for i, row in scan_df.head(3).iterrows():
                        c = T["GREEN"] if "BUY" in row["Signal"] else T["RED"]
                        st.markdown(
                            f'<div style="background:{c}10;border-left:4px solid {c};'
                            f'border-radius:8px;padding:10px 14px;margin:4px 0;font-size:13px">'
                            f'<b style="color:{c}">{row["Asset"]}</b> ({row["Ticker"]}) — '
                            f'<b>{row["Signal"]}</b> | Score: {row["Score"]}/20 | '
                            f'Entry: ₹{row["Entry"]:,.2f} | SL: ₹{row["SL"]:,.2f} | '
                            f'Tgt: ₹{row["Target 1"]:,.2f}</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.warning("No scan results. Check internet connection.")

            # ── Disclaimer ────────────────────────────────────────────────
            st.markdown("---")
            st.markdown(f"""
            <div style="background:{T['CARD_BG']};border:1px solid {T['ACCENT']}30;
                        border-radius:10px;padding:14px 18px;font-size:12px;
                        color:{T['TEXT_MUTED']}">
            ⚠️ <b style="color:{T['ACCENT']}">Disclaimer:</b>
            For educational and research purposes only. Not financial/investment advice.
            Options trading involves substantial risk. Past performance ≠ future results.
            All signals are algorithmic. Always do independent research.
            SEBI registration required for investment advisory in India.
            Data sourced from Yahoo Finance & NSE India public APIs.
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;padding:12px;color:{T['TEXT_FAINT']};font-size:12px">
    ⚡ <b style="color:{T['ACCENT']}">AlphaEdge v2</b> &nbsp;|&nbsp;
    Streamlit + yFinance + Plotly + NSE API &nbsp;|&nbsp;
    Strategies: TSM · ORB · VWAP+RSI · Swing · Combined &nbsp;|&nbsp;
    Theme: <b style="color:{T['ACCENT']}">{THEME}</b> &nbsp;|&nbsp;
    Rate limit: <b>{FETCH_DELAY}s</b> between calls<br>
    <span>
    Nifty · BankNifty · Sensex · Stocks · BTC · ETH · Forex · Gold · Silver · Crude
    </span>
</div>
""", unsafe_allow_html=True)
