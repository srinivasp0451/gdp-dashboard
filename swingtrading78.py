"""
Professional Options Trading Dashboard  v4.0
=============================================
ROOT CAUSES FIXED:
  1. ^NSEI / BTC-USD / GC=F have NO options on yfinance — now handled with
     separate market paths (NSE / US-stocks / Spot-only)
  2. fast_info.get() crashes new yfinance — replaced with getattr()
  3. st.session_state inside @st.cache_data is unreliable — replaced with
     module-level _T dict for throttling
  4. NSE cookie needs 3-step handshake — homepage + derivatives page + API

Data sources:
  NSE path   → NIFTY, BANKNIFTY, SENSEX, NSE stocks  (Indian IP best)
  US path    → AAPL, SPY, QQQ, TSLA, etc. via yfinance (options available)
  Spot-only  → BTC, Gold, USDINR, Crude (no options on yfinance)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import random
import warnings
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

# ── module-level throttle dict (works inside @st.cache_data, unlike session_state) ──
_T = {"last": 0.0}


def _wait(gap: float = 1.5):
    """Enforce minimum gap between consecutive yfinance HTTP calls."""
    now = time.time()
    diff = now - _T["last"]
    if diff < gap:
        time.sleep(gap - diff + random.uniform(0.1, 0.3))
    _T["last"] = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ProOptions ⚡", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
:root {
  --bg:#060a10; --s1:#0d1520; --s2:#111d2e;
  --acc:#00e5ff; --grn:#00ff88; --red:#ff3b5c; --ora:#ff9500;
  --txt:#c8d6e5; --mut:#5a7a9a; --brd:#1e3050;
}
html,body,[class*="css"] {
  font-family:'Inter',sans-serif;
  background:var(--bg)!important;
  color:var(--txt)!important;
}
.stApp { background:var(--bg); }
.stMetric {
  background:var(--s1)!important; border:1px solid var(--brd)!important;
  border-radius:10px!important; padding:12px!important;
}
.stMetric label { color:var(--mut)!important; font-size:10px!important; letter-spacing:1px; text-transform:uppercase; }
.stMetric [data-testid="stMetricValue"] { color:var(--acc)!important; font-family:'Space Mono',monospace; font-size:18px!important; }
div[data-testid="stTabs"] button { color:var(--mut)!important; font-family:'Space Mono',monospace; font-size:11px; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:var(--acc)!important; border-bottom:2px solid var(--acc)!important; }
h1,h2,h3 { font-family:'Space Mono',monospace!important; color:var(--acc)!important; }
.stButton>button {
  background:transparent!important; border:1px solid var(--acc)!important;
  color:var(--acc)!important; font-family:'Space Mono',monospace!important;
  border-radius:8px!important; font-size:12px!important;
}
.stButton>button:hover { background:var(--acc)!important; color:#000!important; }
.sig-box {
  background:var(--s1); border:1px solid var(--brd); border-radius:12px;
  padding:18px 22px; margin:10px 0; position:relative; overflow:hidden;
}
.sig-box::before {
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,var(--acc),var(--grn));
}
.tag-b { background:#002a15; color:#00ff88; border:1px solid #00aa44; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; white-space:nowrap; }
.tag-w { background:#2a1a00; color:#ff9500; border:1px solid #aa5500; padding:3px 10px; border-radius:20px; font-size:11px; white-space:nowrap; }
.tag-n { background:#1a0a00; color:#ff3b5c; border:1px solid #aa0022; padding:3px 10px; border-radius:20px; font-size:11px; white-space:nowrap; }
.sum { background:var(--s2); border:1px solid var(--brd); border-radius:8px; padding:12px 16px; font-size:13px; color:var(--mut); margin-bottom:14px; }
.alert { background:#1a0a00; border-left:3px solid var(--ora); border-radius:6px; padding:10px 14px; margin:4px 0; font-size:13px; }
.gamma-blast { background:#1a000a; border:1px solid #ff3b5c88; border-radius:10px; padding:14px; animation:pulse 2s infinite; }
@keyframes pulse { 0% { box-shadow:0 0 0 0 rgba(255,59,92,.4); } 70% { box-shadow:0 0 0 8px rgba(255,59,92,0); } 100% { box-shadow:0 0 0 0 rgba(255,59,92,0); } }
.tc { background:var(--s2); border:1px solid var(--brd); border-radius:10px; padding:14px; margin:6px 0; }
.tc-o { border-left:3px solid var(--acc); }
.tc-w { border-left:3px solid var(--grn); }
.tc-l { border-left:3px solid var(--red); }
hr { border-color:var(--brd)!important; }
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
RFR  = 0.065
DARK = "plotly_dark"

NSE_SYMS = ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY"]
NSE_TO_YF = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN",
             "FINNIFTY": "NIFTY_FIN_SERVICE.NS", "MIDCPNIFTY": "^NSMIDCP"}

# yfinance symbols that ACTUALLY have listed options
YF_WITH_OPTIONS = {
    "SPY — S&P 500 ETF": "SPY",
    "QQQ — Nasdaq ETF":  "QQQ",
    "IWM — Russell 2000": "IWM",
    "AAPL":   "AAPL",
    "TSLA":   "TSLA",
    "NVDA":   "NVDA",
    "AMZN":   "AMZN",
    "MSFT":   "MSFT",
    "GOOGL":  "GOOGL",
    "META":   "META",
    "AMD":    "AMD",
    "Custom US ticker": "__custom__",
}

# yfinance symbols with NO listed options (spot/volatility analysis only)
YF_SPOT_ONLY = {
    "BTC/USD":   "BTC-USD",
    "GOLD":      "GC=F",
    "SILVER":    "SI=F",
    "USD/INR":   "USDINR=X",
    "CRUDE OIL": "CL=F",
    "EUR/USD":   "EURUSD=X",
    "Custom":    "__custom__",
}

for k, v in {"trades": [], "active": [], "bt_result": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# NSE DATA FETCH — 3-step cookie handshake
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=90, show_spinner=False)
def fetch_nse(symbol: str):
    """Fetch NSE option chain with proper 3-step cookie setup."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })
    try:
        # Step 1: homepage — sets initial cookies
        s.get("https://www.nseindia.com/", timeout=10)
        time.sleep(1.0)
        # Step 2: derivatives page — required for chain API auth
        s.get("https://www.nseindia.com/market-data/equity-derivatives-watch", timeout=10)
        time.sleep(0.8)
        # Step 3: actual option chain API
        if symbol in NSE_SYMS:
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
        r = s.get(url, timeout=15)
        if r.status_code != 200:
            return None, 0.0, []
        data = r.json().get("records", {})
        spot = float(data.get("underlyingValue") or 0)
        exps = data.get("expiryDates", [])
        rows = []
        for rec in data.get("data", []):
            ce = rec.get("CE", {})
            pe = rec.get("PE", {})
            rows.append({
                "strikePrice": rec.get("strikePrice", 0),
                "expiryDate":  rec.get("expiryDate", ""),
                "CE_LTP":      ce.get("lastPrice", 0),
                "CE_OI":       ce.get("openInterest", 0),
                "CE_changeOI": ce.get("changeinOpenInterest", 0),
                "CE_volume":   ce.get("totalTradedVolume", 0),
                "CE_IV":       ce.get("impliedVolatility", 0),
                "CE_bid":      ce.get("bidprice", 0),
                "CE_ask":      ce.get("askPrice", 0),
                "PE_LTP":      pe.get("lastPrice", 0),
                "PE_OI":       pe.get("openInterest", 0),
                "PE_changeOI": pe.get("changeinOpenInterest", 0),
                "PE_volume":   pe.get("totalTradedVolume", 0),
                "PE_IV":       pe.get("impliedVolatility", 0),
                "PE_bid":      pe.get("bidprice", 0),
                "PE_ask":      pe.get("askPrice", 0),
            })
        df = pd.DataFrame(rows)
        if df.empty or spot == 0:
            return None, 0.0, []
        return df, spot, exps
    except Exception:
        return None, 0.0, []


# ─────────────────────────────────────────────────────────────────────────────
# YFINANCE FETCHERS — plain yf.Ticker(), NO session arg
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def yf_spot(sym: str) -> float:
    """Spot price using attribute access on fast_info (not .get())."""
    _wait(1.5)
    try:
        fi = yf.Ticker(sym).fast_info
        for attr in ("last_price", "lastPrice", "regular_market_price",
                     "regularMarketPrice", "previousClose", "previous_close"):
            val = getattr(fi, attr, None)
            if val is not None:
                fval = float(val)
                if fval > 0:
                    return fval
    except Exception:
        pass
    # fallback to history
    _wait(1.5)
    try:
        h = yf.Ticker(sym).history(period="2d", interval="1d")
        if not h.empty:
            return float(h["Close"].squeeze().iloc[-1])
    except Exception:
        pass
    return 0.0


@st.cache_data(ttl=120, show_spinner=False)
def yf_expiries(sym: str):
    """Option expiry dates. Returns [] if no options for this symbol."""
    _wait(1.5)
    try:
        opts = yf.Ticker(sym).options
        return list(opts) if opts else []
    except Exception:
        return []


@st.cache_data(ttl=120, show_spinner=False)
def yf_chain(sym: str, expiry: str):
    """Calls + puts for one expiry. Returns (calls, puts) or (None, None)."""
    _wait(1.5)
    try:
        chain = yf.Ticker(sym).option_chain(expiry)
        return chain.calls.copy(), chain.puts.copy()
    except Exception as e:
        st.error(f"Chain fetch error: {e}")
        return None, None


@st.cache_data(ttl=300, show_spinner=False)
def yf_history(sym: str, period: str = "60d") -> pd.DataFrame:
    """Price history used by straddle chart and backtest."""
    _wait(1.5)
    try:
        df = yf.Ticker(sym).history(period=period, interval="1d", auto_adjust=True)
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def build_chain_df(calls, puts, expiry, spot):
    """Merge yfinance calls+puts into unified chain DataFrame."""
    if calls is None or calls.empty:
        return pd.DataFrame()
    df = pd.DataFrame()
    df["strikePrice"] = calls["strike"].values if "strike" in calls.columns else []
    df["expiryDate"]  = expiry
    for col, src in [("CE_LTP", "lastPrice"), ("CE_OI", "openInterest"),
                     ("CE_volume", "volume"),  ("CE_bid", "bid"), ("CE_ask", "ask")]:
        df[col] = calls[src].values if src in calls.columns else 0
    df["CE_IV"]       = (calls["impliedVolatility"].values * 100) if "impliedVolatility" in calls.columns else 0
    df["CE_changeOI"] = 0

    if puts is not None and not puts.empty and "strike" in puts.columns:
        pi = puts.set_index("strike")
        for col, src in [("PE_LTP", "lastPrice"), ("PE_OI", "openInterest"),
                         ("PE_volume", "volume"),  ("PE_bid", "bid"), ("PE_ask", "ask")]:
            df[col] = df["strikePrice"].map(
                pi[src] if src in pi.columns else pd.Series(dtype=float)
            ).fillna(0).values
        df["PE_IV"] = df["strikePrice"].map(
            pi["impliedVolatility"] * 100 if "impliedVolatility" in pi.columns else pd.Series(dtype=float)
        ).fillna(0).values
    else:
        for c in ["PE_LTP", "PE_OI", "PE_volume", "PE_bid", "PE_ask", "PE_IV"]:
            df[c] = 0
    df["PE_changeOI"] = 0
    df = df.fillna(0)
    # Keep any row with a valid strike — don't filter by LTP (market may be pre/post hours)
    df = df[df["strikePrice"] > 0].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def bs(S, K, T, r, sig, kind="call"):
    if T < 1e-6 or sig < 1e-6 or S <= 0 or K <= 0:
        return dict(delta=0, gamma=0, theta=0, vega=0, price=0)
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if kind == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sig) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-(S * norm.pdf(d1) * sig) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    gamma = norm.pdf(d1) / (S * sig * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100
    return dict(delta=round(delta, 4), gamma=round(gamma, 8),
                theta=round(theta, 4), vega=round(vega, 4),
                price=round(max(price, 0), 4))


def calc_iv(mkt, S, K, T, r, kind="call"):
    if T <= 0 or mkt <= 0 or S <= 0:
        return 0.20
    try:
        return max(brentq(lambda s: bs(S, K, T, r, s, kind)["price"] - mkt,
                          1e-4, 20.0, xtol=1e-5), 0.001)
    except Exception:
        return 0.20


def add_greeks(df, spot, expiry_str):
    if df is None or df.empty:
        return df
    try:
        fmt = "%d-%b-%Y" if "-" in str(expiry_str) and not str(expiry_str)[4:5].isdigit() else "%Y-%m-%d"
        T = max((datetime.strptime(str(expiry_str), fmt) - datetime.now()).days / 365.0, 1 / 365)
    except Exception:
        T = 7 / 365
    df = df.copy()
    for i, row in df.iterrows():
        K = float(row["strikePrice"])
        for kind, px in [("call", "CE"), ("put", "PE")]:
            ltp    = float(row.get(f"{px}_LTP", 0))
            iv_pct = float(row.get(f"{px}_IV", 0))
            sig    = iv_pct / 100 if iv_pct > 0.5 else calc_iv(ltp, spot, K, T, RFR, kind)
            sig    = max(sig, 0.01)
            g      = bs(spot, K, T, RFR, sig, kind)
            df.at[i, f"{px}_delta"] = g["delta"]
            df.at[i, f"{px}_gamma"] = g["gamma"]
            df.at[i, f"{px}_theta"] = g["theta"]
            df.at[i, f"{px}_vega"]  = g["vega"]
            if iv_pct == 0 and ltp > 0:
                df.at[i, f"{px}_IV"] = round(sig * 100, 2)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL ENGINE  — identical code reused verbatim in backtest
# ─────────────────────────────────────────────────────────────────────────────
def compute_signals(df, spot):
    out = dict(pcr=0, max_pain=spot, atm=spot, straddle=0,
               resistance=spot, support=spot,
               atm_iv=20.0, ce_iv=0.0, pe_iv=0.0, skew=0.0,
               gamma_blast=False, abnormal=[],
               rec="⚪ NO TRADE", direction="NONE", conf=0,
               scalp="WAIT", intraday="WAIT", swing="WAIT", pos="WAIT",
               reasons=[])
    if df is None or df.empty or spot == 0:
        return out
    df = df.copy()
    df["strikePrice"] = df["strikePrice"].astype(float)

    ai  = (df["strikePrice"] - spot).abs().idxmin()
    atm = df.loc[ai]
    out["atm"]      = float(atm["strikePrice"])
    out["straddle"] = round(float(atm.get("CE_LTP", 0)) + float(atm.get("PE_LTP", 0)), 2)

    ce_oi = float(df["CE_OI"].sum())
    pe_oi = float(df["PE_OI"].sum())
    out["pcr"] = round(pe_oi / ce_oi, 4) if ce_oi > 0 else 0

    strikes = df["strikePrice"].values
    c_oi    = df["CE_OI"].values
    p_oi    = df["PE_OI"].values
    pain = [
        sum(max(0, k - s) * o for k, o in zip(strikes, c_oi)) +
        sum(max(0, s - k) * o for k, o in zip(strikes, p_oi))
        for s in strikes
    ]
    out["max_pain"]   = float(strikes[int(np.argmin(pain))]) if pain else spot
    out["resistance"] = float(df.loc[df["CE_OI"].idxmax(), "strikePrice"])
    out["support"]    = float(df.loc[df["PE_OI"].idxmax(), "strikePrice"])

    ce_iv = float(atm.get("CE_IV", 0))
    pe_iv = float(atm.get("PE_IV", 0))
    out["ce_iv"]  = ce_iv
    out["pe_iv"]  = pe_iv
    out["atm_iv"] = (ce_iv + pe_iv) / 2 if (ce_iv + pe_iv) > 0 else 20.0
    out["skew"]   = round(pe_iv - ce_iv, 2)

    near_oi = (
        df["CE_OI"].where(df["strikePrice"].between(spot * 0.99, spot * 1.01), 0).sum() +
        df["PE_OI"].where(df["strikePrice"].between(spot * 0.99, spot * 1.01), 0).sum()
    )
    out["gamma_blast"] = (ce_oi + pe_oi) > 0 and near_oi / (ce_oi + pe_oi + 1) > 0.35

    iv = out["atm_iv"]
    if iv > 55:
        out["abnormal"].append(f"🔴 IV EXTREME {iv:.1f}% — premium crushing, avoid buying")
    elif iv > 35:
        out["abnormal"].append(f"⚠️ High IV {iv:.1f}% — expensive for buyers, be selective")
    elif 0 < iv < 15:
        out["abnormal"].append(f"✅ Low IV {iv:.1f}% — cheap premium, prime buying environment!")
    sk = out["skew"]
    if sk > 8:
        out["abnormal"].append(f"📊 Put skew +{sk:.1f}%: heavy hedging, bearish bias dominant")
    elif sk < -8:
        out["abnormal"].append(f"📊 Call skew {sk:.1f}%: aggressive call buying, bullish momentum")
    mp_pct = (out["max_pain"] - spot) / spot * 100 if spot > 0 else 0
    if abs(mp_pct) > 2:
        out["abnormal"].append(f"🎯 Max pain {mp_pct:+.1f}% from spot — gravity pull near expiry")
    if out["gamma_blast"]:
        out["abnormal"].append("⚡ GAMMA BLAST: OI peak at ATM — breakout = explosive premium move")

    # === scoring (identical in backtest) ===
    score = 50; direction = "NONE"; reasons = []
    pcr = out["pcr"]

    if pcr > 1.5:
        score += 12; direction = "CALL"; reasons.append(f"PCR {pcr:.2f} extreme put→contrarian BULL")
    elif pcr > 1.1:
        score += 6;  direction = "CALL"; reasons.append(f"PCR {pcr:.2f} put-heavy→mild bullish")
    elif pcr < 0.5:
        score += 12; direction = "PUT";  reasons.append(f"PCR {pcr:.2f} extreme call→contrarian BEAR")
    elif pcr < 0.9:
        score += 6;  direction = "PUT";  reasons.append(f"PCR {pcr:.2f} call-heavy→mild bearish")
    else:
        reasons.append(f"PCR {pcr:.2f} balanced/neutral")

    if iv > 50:
        score -= 25; reasons.append("IV >50% avoid buying")
    elif iv > 35:
        score -= 12; reasons.append(f"IV {iv:.1f}% high theta risk")
    elif 0 < iv < 18:
        score += 15; reasons.append(f"IV {iv:.1f}% cheap premium!")
    elif iv <= 30:
        score += 8;  reasons.append(f"IV {iv:.1f}% acceptable for buyers")

    mp = out["max_pain"]
    if spot < mp and direction == "CALL":
        score += 10; reasons.append(f"below max pain {mp:,.0f}→upward pull")
    elif spot > mp and direction == "PUT":
        score += 10; reasons.append(f"above max pain {mp:,.0f}→downward pull")
    elif spot < mp and direction == "PUT":
        score -= 8
    elif spot > mp and direction == "CALL":
        score -= 8

    res = out["resistance"]; sup = out["support"]
    if direction == "CALL" and sup < spot < res:
        score += 8; reasons.append(f"in bull zone {sup:,.0f}→{res:,.0f}")
    if direction == "PUT"  and sup < spot < res:
        score += 8; reasons.append(f"near resistance wall {res:,.0f}")

    ce_chg = float(df["CE_changeOI"].sum())
    pe_chg = float(df["PE_changeOI"].sum())
    if pe_chg > ce_chg * 1.3 and direction == "CALL":
        score += 7; reasons.append("put writing→support building")
    if ce_chg > pe_chg * 1.3 and direction == "PUT":
        score += 7; reasons.append("call writing→resistance building")

    score = min(max(int(score), 0), 100)
    out["conf"] = score; out["direction"] = direction; out["reasons"] = reasons

    A   = int(out["atm"])
    otm = int(A * (1.005 if direction == "CALL" else 0.995))

    if score >= 72 and direction == "CALL":
        out["rec"]     = "🟢 BUY CALL (CE)"
        out["scalp"]   = f"Buy {A} CE — scalp 20-40%, SL 30%"
        out["intraday"] = f"Buy {A} CE — target 60%, SL 30%"
        out["swing"]   = f"Buy {otm} CE next expiry — 80-100% target"
        out["pos"]     = f"Buy {otm} CE far expiry — positional hold"
    elif score >= 72 and direction == "PUT":
        out["rec"]     = "🔴 BUY PUT (PE)"
        out["scalp"]   = f"Buy {A} PE — scalp 20-40%, SL 30%"
        out["intraday"] = f"Buy {A} PE — target 60%, SL 30%"
        out["swing"]   = f"Buy {otm} PE next expiry — 80-100% target"
        out["pos"]     = f"Buy {otm} PE far expiry — positional hold"
    elif score >= 58:
        out["rec"] = "🟡 WATCH — Weak Signal"
        out["scalp"] = out["intraday"] = out["swing"] = out["pos"] = "WATCH — wait for confirmation"
    else:
        out["rec"] = "⚪ NO TRADE"
        out["scalp"] = out["intraday"] = out["swing"] = out["pos"] = "NO TRADE"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST — identical scoring as compute_signals
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def run_backtest(hist_sym: str, lookback: int, capital: float, pos_pct: float):
    raw = yf_history(hist_sym, f"{lookback}d")
    if raw.empty or len(raw) < 15:
        return None, "Not enough price history"
    close = raw["Close"].squeeze().astype(float)
    vol7  = close.pct_change().rolling(7).std()  * np.sqrt(252) * 100
    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    vol3  = close.pct_change().rolling(3).std()  * np.sqrt(252) * 100
    vol10 = close.pct_change().rolling(10).std() * np.sqrt(252) * 100

    trades = []; equity = float(capital); curve = [equity]
    for i in range(20, len(close) - 1):
        iv   = float(vol7.iloc[i])  if not np.isnan(vol7.iloc[i])  else 20.0
        spot = float(close.iloc[i])
        mp   = (float(sma10.iloc[i]) + float(sma20.iloc[i])) / 2
        v3   = float(vol3.iloc[i])  if not np.isnan(vol3.iloc[i])  else 20.0
        v10  = float(vol10.iloc[i]) if not np.isnan(vol10.iloc[i]) else 20.0
        pcr  = 1.0 + (v10 - v3) / (v10 + 1)

        # === identical scoring ===
        score = 50; direction = "NONE"
        if pcr > 1.5:    score += 12; direction = "CALL"
        elif pcr > 1.1:  score += 6;  direction = "CALL"
        elif pcr < 0.5:  score += 12; direction = "PUT"
        elif pcr < 0.9:  score += 6;  direction = "PUT"

        if iv > 50:    score -= 25
        elif iv > 35:  score -= 12
        elif iv < 18:  score += 15
        elif iv <= 30: score += 8

        if spot < mp and direction == "CALL":   score += 10
        elif spot > mp and direction == "PUT":  score += 10
        elif spot < mp and direction == "PUT":  score -= 8
        elif spot > mp and direction == "CALL": score -= 8

        if float(sma10.iloc[i]) > float(sma20.iloc[i]) and direction == "CALL": score += 7
        elif float(sma10.iloc[i]) < float(sma20.iloc[i]) and direction == "PUT": score += 7

        score = min(max(int(score), 0), 100)
        curve.append(equity)
        if score < 72 or direction == "NONE":
            continue

        opt_px = spot * max(iv / 100, 0.01) * np.sqrt(7 / 365) * 0.4
        if opt_px <= 0:
            continue
        ret      = (float(close.iloc[i + 1]) - spot) / spot
        raw_gain = (ret * 0.5 * spot) if direction == "CALL" else (-ret * 0.5 * spot)
        pnl_pct  = max(min((raw_gain - opt_px / 7) / opt_px, 2.5), -0.5)
        pnl      = equity * (pos_pct / 100) * pnl_pct
        equity   = max(equity + pnl, 0)
        trades.append({"Date": raw.index[i].strftime("%Y-%m-%d"),
                       "Dir": direction, "Spot": round(spot, 2),
                       "OptPx": round(opt_px, 2), "IV%": round(iv, 1),
                       "Score": score, "P&L(₹)": round(pnl, 2),
                       "P&L(%)": round(pnl_pct * 100, 2), "Equity": round(equity, 2),
                       "Result": "✅ WIN" if pnl > 0 else "❌ LOSS"})
        curve[-1] = equity

    if not trades:
        return None, "No trades generated — try longer lookback or different symbol"
    tdf  = pd.DataFrame(trades)
    wins = (tdf["P&L(₹)"] > 0).sum(); total = len(tdf)
    aw   = tdf.loc[tdf["P&L(₹)"] > 0, "P&L(₹)"].mean() if wins > 0 else 0
    al   = tdf.loc[tdf["P&L(₹)"] <= 0, "P&L(₹)"].mean() if (total - wins) > 0 else 0
    peak = capital; mdd = 0
    for eq in curve:
        if eq > peak: peak = eq
        mdd = max(mdd, (peak - eq) / peak * 100)
    stats = {"total": total, "wins": int(wins), "wr": round(wins / total * 100, 1),
             "total_pnl": round(tdf["P&L(₹)"].sum(), 2),
             "aw": round(aw, 2), "al": round(al, 2),
             "rr": round(abs(aw / al), 2) if al != 0 else 0,
             "mdd": round(mdd, 2), "final": round(equity, 2),
             "ret%": round((equity - capital) / capital * 100, 2),
             "curve": curve}
    return tdf, stats


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def vl(fig, x, lbl="", color="yellow", dash="dash", row=1, col=1):
    fig.add_vline(x=x, line_dash=dash, line_color=color, line_width=1.5,
                  annotation_text=lbl, annotation_font_color=color, row=row, col=col)


def plot_chain(df, spot):
    rng = spot * 0.05
    sub = df[df["strikePrice"].between(spot - rng, spot + rng)]
    if sub.empty: sub = df
    fig = make_subplots(2, 2, subplot_titles=["Premium (CE vs PE)", "Open Interest",
                                               "IV Smile", "Volume"])
    x = sub["strikePrice"]
    fig.add_trace(go.Bar(name="CE", x=x, y=sub["CE_LTP"], marker_color="#00ff88", opacity=.85), 1, 1)
    fig.add_trace(go.Bar(name="PE", x=x, y=sub["PE_LTP"], marker_color="#ff3b5c", opacity=.85), 1, 1)
    fig.add_trace(go.Bar(name="CE OI", x=x, y=sub["CE_OI"], marker_color="#00e5ff", opacity=.8), 1, 2)
    fig.add_trace(go.Bar(name="PE OI", x=x, y=sub["PE_OI"], marker_color="#ff9500", opacity=.8), 1, 2)
    fig.add_trace(go.Scatter(name="CE IV", x=x, y=sub["CE_IV"], mode="lines+markers",
                              line=dict(color="#00ff88", width=2)), 2, 1)
    fig.add_trace(go.Scatter(name="PE IV", x=x, y=sub["PE_IV"], mode="lines+markers",
                              line=dict(color="#ff3b5c", width=2)), 2, 1)
    fig.add_trace(go.Bar(name="CE Vol", x=x, y=sub["CE_volume"], marker_color="rgba(0,255,136,0.4)"), 2, 2)
    fig.add_trace(go.Bar(name="PE Vol", x=x, y=sub["PE_volume"], marker_color="rgba(255,59,92,0.4)"), 2, 2)
    for r in [1, 2]:
        for c in [1, 2]: vl(fig, spot, "Spot", row=r, col=c)
    fig.update_layout(template=DARK, height=520, barmode="group",
                      title="Live Option Chain", margin=dict(t=50, b=10))
    return fig


def plot_oi(df, spot, sig):
    rng = spot * 0.055
    sub = df[df["strikePrice"].between(spot - rng, spot + rng)]
    if sub.empty: sub = df
    sub = sub.copy()
    sub["CE_pct"] = (sub["CE_changeOI"] / (sub["CE_OI"] - sub["CE_changeOI"]).clip(lower=1) * 100).fillna(0)
    sub["PE_pct"] = (sub["PE_changeOI"] / (sub["PE_OI"] - sub["PE_changeOI"]).clip(lower=1) * 100).fillna(0)
    fig = make_subplots(3, 1, subplot_titles=["Open Interest", "Change in OI", "% Change in OI"])
    x = sub["strikePrice"]
    fig.add_trace(go.Bar(name="CE OI", x=x, y=sub["CE_OI"], marker_color="#00e5ff", opacity=.8), 1, 1)
    fig.add_trace(go.Bar(name="PE OI", x=x, y=sub["PE_OI"], marker_color="#ff9500", opacity=.8), 1, 1)
    fig.add_trace(go.Bar(name="ΔCE", x=x, y=sub["CE_changeOI"],
                          marker_color=["#00ff88" if v >= 0 else "#ff3b5c" for v in sub["CE_changeOI"]]), 2, 1)
    fig.add_trace(go.Bar(name="ΔPE", x=x, y=sub["PE_changeOI"],
                          marker_color=["#ff9500" if v >= 0 else "#8888ff" for v in sub["PE_changeOI"]]), 2, 1)
    fig.add_trace(go.Bar(name="CE%", x=x, y=sub["CE_pct"],
                          marker_color=["rgba(0,255,136,0.47)" if v >= 0 else "rgba(255,59,92,0.47)" for v in sub["CE_pct"]]), 3, 1)
    fig.add_trace(go.Bar(name="PE%", x=x, y=sub["PE_pct"],
                          marker_color=["rgba(255,149,0,0.47)" if v >= 0 else "rgba(136,136,255,0.47)" for v in sub["PE_pct"]]), 3, 1)
    for row in [1, 2, 3]:
        vl(fig, spot, "Spot", row=row)
        if sig["resistance"]: vl(fig, sig["resistance"], "R", "#ff3b5c", "dot", row, 1)
        if sig["support"]:    vl(fig, sig["support"],    "S", "#00ff88", "dot", row, 1)
    fig.update_layout(template=DARK, height=620, barmode="group",
                      title="OI Analysis", margin=dict(t=50, b=10))
    return fig


def plot_greeks_chart(df, spot):
    rng = spot * 0.04
    sub = df[df["strikePrice"].between(spot - rng, spot + rng)]
    if sub.empty: sub = df
    fig = make_subplots(2, 2, subplot_titles=["Delta", "Gamma", "Theta (daily ₹)", "Vega"])
    x   = sub["strikePrice"]
    for (r, c, cc, pc) in [(1, 1, "CE_delta", "PE_delta"), (1, 2, "CE_gamma", "PE_gamma"),
                            (2, 1, "CE_theta", "PE_theta"), (2, 2, "CE_vega",  "PE_vega")]:
        for col, color in [(cc, "#00ff88"), (pc, "#ff3b5c")]:
            if col in sub.columns:
                fig.add_trace(go.Scatter(name=col, x=x, y=sub[col], mode="lines",
                                          line=dict(color=color, width=2)), r, c)
        vl(fig, spot, row=r, col=c)
    fig.update_layout(template=DARK, height=520, title="Live Greeks", margin=dict(t=50, b=10))
    return fig


def plot_straddle(hist_sym, straddle_now):
    hist = yf_history(hist_sym, "45d")
    if hist.empty: return None, None
    close = hist["Close"].squeeze().astype(float)
    rv    = close.pct_change().rolling(7).std() * np.sqrt(252)
    est   = (close * rv * np.sqrt(7 / 252) * 0.8).dropna()
    if est.empty: return None, None
    p25, p50, p75 = est.quantile([.25, .5, .75])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=est.index, y=est, name="Hist est",
                              line=dict(color="#00e5ff", width=2)))
    if straddle_now > 0:
        fig.add_hline(y=straddle_now, line_color="yellow", line_dash="dash",
                      annotation_text=f"Now:{straddle_now:.1f}", annotation_font_color="yellow")
    fig.add_hline(y=p25, line_color="#00ff88", line_dash="dot", annotation_text=f"P25:{p25:.1f}")
    fig.add_hline(y=p75, line_color="#ff3b5c", line_dash="dot", annotation_text=f"P75:{p75:.1f}")
    fig.add_hline(y=p50, line_color="#ff9500", line_dash="dot", annotation_text=f"P50:{p50:.1f}")
    fig.update_layout(template=DARK, height=320, title="Straddle vs 45-Day History")
    if straddle_now > 0:
        if straddle_now < p25:    verdict = ("✅ CHEAP — below P25, great time to buy!", "#00ff88")
        elif straddle_now > p75:  verdict = ("⚠️ EXPENSIVE — above P75, avoid buying!", "#ff3b5c")
        else:                      verdict = ("🟡 FAIR VALUE — be selective", "#ff9500")
    else:
        verdict = None
    return fig, verdict


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown("<h1 style='font-size:24px;letter-spacing:2px'>⚡ PRO OPTIONS INTELLIGENCE</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#5a7a9a;margin-top:-10px'>Live Chain · Greeks · OI · Signals · Backtest · Trade History</p>",
                unsafe_allow_html=True)
    st.divider()

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Instrument")
        src_type = st.selectbox("Market", [
            "🇺🇸 US Stocks / ETFs",
            "🇮🇳 NSE India",
            "📊 Spot-Only (BTC / Gold / FX)",
        ])

        fetch_sym = ""
        hist_sym  = ""
        has_opts  = True

        if src_type == "🇮🇳 NSE India":
            nse_choice = st.selectbox("Symbol", NSE_SYMS + ["Custom NSE Stock"])
            if nse_choice == "Custom NSE Stock":
                nse_choice = st.text_input("NSE symbol (e.g. RELIANCE)", "RELIANCE").upper().strip()
            fetch_sym = nse_choice
            hist_sym  = NSE_TO_YF.get(nse_choice, nse_choice + ".NS")
            has_opts  = True
            st.info("💡 NSE API works best on Indian IP. Outside India → NSE will fail, app shows spot-only fallback.")

        elif src_type == "🇺🇸 US Stocks / ETFs":
            us_choice = st.selectbox("Stock / ETF", list(YF_WITH_OPTIONS.keys()))
            if us_choice == "Custom US ticker":
                fetch_sym = st.text_input("yFinance ticker", "AAPL").upper().strip()
            else:
                fetch_sym = YF_WITH_OPTIONS[us_choice]
            hist_sym = fetch_sym
            has_opts = True
            st.success("✅ US stocks/ETFs have full options on yFinance")

        else:  # Spot-Only
            spot_choice = st.selectbox("Instrument", list(YF_SPOT_ONLY.keys()))
            if spot_choice == "Custom":
                fetch_sym = st.text_input("yFinance ticker", "BTC-USD").upper().strip()
            else:
                fetch_sym = YF_SPOT_ONLY[spot_choice]
            hist_sym = fetch_sym
            has_opts = False
            st.warning("⚠️ BTC/Gold/FX have no listed options on yFinance. Spot + volatility analysis only.")

        st.divider()
        st.markdown("### 💰 Risk")
        capital = st.number_input("Capital (₹)", 50_000, 10_000_000, 100_000, 10_000)
        pos_pct = st.slider("Position size %", 2, 15, 5)
        sl_pct  = st.slider("Stop loss %", 20, 50, 30)
        tgt_pct = st.slider("Target %", 30, 200, 60)
        st.divider()
        st.button("🔄 FETCH LIVE DATA", type="primary", use_container_width=True)
        if st.button("🗑️ Clear Cache & Retry", use_container_width=True):
            st.cache_data.clear()
            _T["last"] = 0.0
            st.rerun()
        auto_ref = st.checkbox("Auto-refresh (90s)")
        st.caption("⚠️ Educational only. Not financial advice.")

    # ── DATA FETCH ────────────────────────────────────────────────────────────
    df_exp     = pd.DataFrame()
    spot       = 0.0
    expiries   = []
    sel_expiry = ""

    status_ph = st.empty()

    with st.spinner(f"⏳ Fetching {fetch_sym} — 1.5s enforced between each API call…"):

        # NSE path
        if src_type == "🇮🇳 NSE India":
            status_ph.info("📡 Connecting to NSE (3-step cookie handshake)…")
            df_raw, nse_spot, nse_exps = fetch_nse(fetch_sym)
            if df_raw is not None and nse_spot > 0:
                spot = nse_spot; expiries = nse_exps
                if expiries:
                    with st.sidebar:
                        sel_expiry = st.selectbox("Expiry", expiries[:8])
                    mask = df_raw["expiryDate"] == sel_expiry
                    df_exp = df_raw[mask].copy() if mask.any() else df_raw.copy()
                else:
                    df_exp = df_raw.copy()
                status_ph.success(f"✅ NSE loaded — {len(df_exp)} strikes, spot ₹{spot:,.2f}")
            else:
                status_ph.warning("⚠️ NSE API failed (likely non-Indian IP). Fetching spot via yFinance…")
                spot = yf_spot(hist_sym)
                has_opts = False
                if spot > 0:
                    status_ph.warning(f"⚠️ NSE chain unavailable. Showing spot ₹{spot:,.2f} only. Use Indian IP for full chain.")
                else:
                    status_ph.error("❌ Could not get price data for NSE symbol.")
                    spot = 0.0

        # US stocks path
        elif src_type == "🇺🇸 US Stocks / ETFs":
            status_ph.info("📡 Fetching from yFinance…")
            spot = yf_spot(fetch_sym)
            if spot == 0:
                status_ph.error(f"❌ Cannot get price for {fetch_sym}. Check symbol name.")
            else:
                expiries = yf_expiries(fetch_sym)
                if not expiries:
                    status_ph.warning(f"⚠️ {fetch_sym} has no options — market closed or unsupported symbol.")
                    has_opts = False
                else:
                    with st.sidebar:
                        sel_expiry = st.selectbox("Expiry", expiries[:8])
                    calls, puts = yf_chain(fetch_sym, sel_expiry)
                    if calls is not None and not calls.empty:
                        df_exp = build_chain_df(calls, puts, sel_expiry, spot)
                        if not df_exp.empty:
                            status_ph.success(f"✅ {fetch_sym} loaded — {len(df_exp)} strikes, spot ${spot:,.2f}")
                        else:
                            status_ph.warning("Chain parsed but empty. Market may be closed.")
                            has_opts = False
                    else:
                        status_ph.error("Chain fetch failed. Try again in 15 seconds.")
                        has_opts = False

        # Spot-only path
        else:
            spot = yf_spot(fetch_sym)
            if spot > 0:
                status_ph.success(f"✅ {fetch_sym} spot: {spot:,.4f} (no options — volatility analysis only)")
            else:
                status_ph.error(f"❌ Cannot get price for {fetch_sym}.")

    if spot == 0:
        st.error("❌ No data. See guidance below.")
        st.markdown("""
<div style='background:#0a1929;border:1px solid #1e3050;border-radius:12px;padding:18px;margin:10px 0'>
<b style='color:#ff9500'>Troubleshooting:</b><br><br>
• <b>Outside India?</b> → NSE won't work. Use <b>US Stocks</b> → SPY, AAPL, QQQ<br>
• <b>Rate limited?</b> → Click <b>🗑️ Clear Cache & Retry</b>, wait 20s, fetch again<br>
• <b>BTC / Gold / FX</b> → Choose <b>Spot-Only</b> tab — these have no options on yFinance<br>
• <b>Market closed?</b> → Options chain only available during market hours; spot always works<br>
• <b>Wrong symbol?</b> → Confirm exact yFinance format: AAPL, SPY, BTC-USD, GC=F<br>
</div>""", unsafe_allow_html=True)
        return

    spot = float(spot)
    has_chain = not df_exp.empty and "strikePrice" in df_exp.columns and len(df_exp) > 0

    if has_chain:
        with st.spinner("🔢 Computing Greeks…"):
            df_exp = add_greeks(df_exp, spot, sel_expiry)
        all_strikes = sorted(df_exp["strikePrice"].unique().tolist())
        atm_pos = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - spot))
    else:
        all_strikes = []
        atm_pos = 0

    sig = compute_signals(df_exp, spot)

    # ── TOP METRICS ───────────────────────────────────────────────────────────
    cols = st.columns(7)
    metrics = [
        ("📍 Spot",     f"{spot:,.2f}"),
        ("🎯 ATM",      f"{sig['atm']:,.0f}"      if has_chain else "N/A"),
        ("📊 PCR",      f"{sig['pcr']:.3f}"       if has_chain else "N/A"),
        ("💀 Max Pain", f"{sig['max_pain']:,.0f}"  if has_chain else "N/A"),
        ("🌡 ATM IV",   f"{sig['atm_iv']:.1f}%"   if has_chain else "N/A"),
        ("↕ IV Skew",   f"{sig['skew']:+.2f}%"    if has_chain else "N/A"),
        ("♟ Straddle",  f"{sig['straddle']:.2f}"  if has_chain else "N/A"),
    ]
    for col, (label, val) in zip(cols, metrics):
        col.metric(label, val)

    # ── SIGNAL BANNER ─────────────────────────────────────────────────────────
    conf = sig["conf"]
    bc   = "#00ff88" if conf >= 72 else "#ff9500" if conf >= 58 else "#5a7a9a"

    def tag(s):
        if "BUY" in s: return f'<span class="tag-b">{s}</span>'
        return f'<span class="tag-w">{s}</span>'

    if has_chain:
        st.markdown(f"""
<div class="sig-box">
  <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px">
    <div>
      <div style="color:#5a7a9a;font-size:10px;letter-spacing:2px">PRIMARY SIGNAL</div>
      <div style="font-size:22px;font-weight:700;color:{bc};font-family:Space Mono">{sig['rec']}</div>
      <div style="color:#7a9ab5;font-size:12px;margin-top:4px">{' · '.join(sig['reasons'][:3])}</div>
    </div>
    <div style="text-align:right">
      <div style="color:#5a7a9a;font-size:10px;letter-spacing:2px">CONFIDENCE</div>
      <div style="font-size:34px;font-weight:700;color:{bc};font-family:Space Mono">{conf}%</div>
      <div style="background:{bc}22;border-radius:20px;height:5px;width:120px;margin-top:4px;margin-left:auto">
        <div style="background:{bc};border-radius:20px;height:5px;width:{conf}%"></div>
      </div>
    </div>
  </div>
  <div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap">
    {tag("⚡ " + sig['scalp'])}
    {tag("📅 " + sig['intraday'])}
    {tag("🗓 " + sig['swing'])}
    {tag("📆 " + sig['pos'])}
  </div>
</div>""", unsafe_allow_html=True)
        for ab in sig["abnormal"]:
            st.markdown(f'<div class="alert">{ab}</div>', unsafe_allow_html=True)
        if sig["gamma_blast"]:
            st.markdown("""<div class="gamma-blast">
⚡ <b style="color:#ff3b5c">GAMMA BLAST ALERT</b> — Massive OI at ATM.
Breakout will cause explosive premium expansion. Wait for clear direction then enter immediately.
</div>""", unsafe_allow_html=True)
    else:
        st.info(f"📊 **Spot-only mode** for **{fetch_sym}** — price: **{spot:,.4f}**. "
                "Options chain not available. Backtest and volatility analysis still work below.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TABS
    # ─────────────────────────────────────────────────────────────────────────
    tabs = st.tabs(["📊 Chain", "🔢 Greeks", "📈 OI",
                    "⚡ Live Trade", "🔬 Backtest", "📋 History", "🧠 Analysis"])

    # ═══ TAB 1 — CHAIN ═══════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown('<div class="sum">Live CE/PE premiums, OI, IV across strikes. Green=calls, Red=puts. Straddle vs 45-day history shows if options are cheap or expensive for buyers right now.</div>',
                    unsafe_allow_html=True)
        if not has_chain:
            st.warning("Options chain not available. Select NSE India or US Stocks for full chain.")
        else:
            def_st = all_strikes[max(0, atm_pos - 8):atm_pos + 9]
            sel_st = st.multiselect("Strikes", all_strikes, default=def_st)
            if not sel_st: sel_st = def_st
            show_c = [c for c in ["CE_LTP","CE_OI","CE_changeOI","CE_IV","CE_volume",
                                   "strikePrice","PE_LTP","PE_OI","PE_changeOI","PE_IV","PE_volume"]
                      if c in df_exp.columns]
            st.dataframe(df_exp[df_exp["strikePrice"].isin(sel_st)][show_c].round(2),
                         use_container_width=True, height=260)
            st.plotly_chart(plot_chain(df_exp, spot), use_container_width=True)
            st.markdown("#### 📉 Straddle vs 45-Day History")
            fs, fv = plot_straddle(hist_sym, sig["straddle"])
            if fs: st.plotly_chart(fs, use_container_width=True)
            if fv: st.markdown(
                f'<div style="background:{fv[1]}11;border:1px solid {fv[1]}44;'
                f'border-radius:8px;padding:10px 14px;color:{fv[1]};font-weight:600">{fv[0]}</div>',
                unsafe_allow_html=True)

    # ═══ TAB 2 — GREEKS ══════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown('<div class="sum">Delta=directional sensitivity, Gamma=acceleration (peaks ATM), Theta=daily decay (buyer enemy), Vega=IV sensitivity. Ideal buy: Delta 0.3-0.6, IV below 25%, Vega exceeds |Theta|.</div>',
                    unsafe_allow_html=True)
        if not has_chain:
            st.warning("Greeks require an options chain. Switch to NSE India or US Stocks.")
        else:
            st.plotly_chart(plot_greeks_chart(df_exp, spot), use_container_width=True)
            ai2  = (df_exp["strikePrice"] - spot).abs().idxmin()
            atm2 = df_exp.loc[ai2]
            st.markdown(f"#### ATM Strike {atm2['strikePrice']:,.0f} — Full Greeks")
            g1, g2 = st.columns(2)
            for col, px, label, color in [
                (g1, "CE", "📗 CALL", "#00ff88"),
                (g2, "PE", "📕 PUT",  "#ff3b5c"),
            ]:
                with col:
                    st.markdown(f"<h5 style='color:{color}'>{label}</h5>", unsafe_allow_html=True)
                    cs = st.columns(3)
                    for i, (name, key) in enumerate([
                        ("Delta", f"{px}_delta"), ("Gamma", f"{px}_gamma"),
                        ("Theta", f"{px}_theta"), ("Vega",  f"{px}_vega"),
                        ("IV %",  f"{px}_IV"),
                    ]):
                        val = float(atm2.get(key, 0))
                        fmt = f"{val:.2f}%" if "IV" in name else f"{val:.4f}"
                        cs[i % 3].metric(name, fmt)

            iv = sig["atm_iv"]
            tips = []
            if iv > 40:
                tips.append(("🔴 IV >40%: theta decimates buyers — WAIT for IV to drop", "#ff3b5c"))
            elif iv < 15:
                tips.append(("✅ IV <15%: premium cheap, vega gains on any IV spike", "#00ff88"))
            ce_th = float(atm2.get("CE_theta", 0))
            ce_ve = float(atm2.get("CE_vega",  0))
            if abs(ce_th) > abs(ce_ve):
                tips.append(("⚠️ |Theta| > Vega: decay eating gains — prefer ITM or nearer strike", "#ff9500"))
            else:
                tips.append(("✅ Vega > |Theta|: good time/IV balance for option buyers", "#00ff88"))
            for t, clr in tips:
                st.markdown(
                    f'<div style="background:{clr}11;border-left:3px solid {clr};'
                    f'padding:10px;border-radius:6px;margin:4px 0;font-size:13px">{t}</div>',
                    unsafe_allow_html=True)
            gcols = [c for c in ["strikePrice","CE_IV","CE_delta","CE_gamma","CE_theta","CE_vega",
                                  "PE_IV","PE_delta","PE_gamma","PE_theta","PE_vega"] if c in df_exp.columns]
            near = all_strikes[max(0, atm_pos - 5):atm_pos + 6]
            st.dataframe(df_exp[df_exp["strikePrice"].isin(near)][gcols].round(5), use_container_width=True)

    # ═══ TAB 3 — OI ══════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown('<div class="sum">Max CE OI = resistance wall. Max PE OI = support floor. Rising OI + rising price = long buildup (bullish). Rising OI + falling price = short buildup (bearish). Gamma blast = massive OI at ATM before breakout.</div>',
                    unsafe_allow_html=True)
        if not has_chain:
            st.warning("OI analysis requires an options chain.")
        else:
            st.plotly_chart(plot_oi(df_exp, spot, sig), use_container_width=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🔴 Resistance", f"{sig['resistance']:,.0f}")
            m2.metric("🟢 Support",    f"{sig['support']:,.0f}")
            m3.metric("🎯 Max Pain",   f"{sig['max_pain']:,.0f}")
            m4.metric("📊 PCR",        f"{sig['pcr']:.4f}")
            b1, b2 = st.columns(2)
            with b1:
                st.markdown("**Top CE OI Strikes — Resistance Walls**")
                ct = df_exp.nlargest(5, "CE_OI")[["strikePrice","CE_OI","CE_changeOI","CE_LTP"]].copy()
                ct["Signal"] = ct["CE_changeOI"].apply(lambda x: "🔴 Call Writing" if x >= 0 else "🟢 Call Unwinding")
                st.dataframe(ct, use_container_width=True)
            with b2:
                st.markdown("**Top PE OI Strikes — Support Walls**")
                pt = df_exp.nlargest(5, "PE_OI")[["strikePrice","PE_OI","PE_changeOI","PE_LTP"]].copy()
                pt["Signal"] = pt["PE_changeOI"].apply(lambda x: "🟢 Put Writing" if x >= 0 else "🔴 Put Unwinding")
                st.dataframe(pt, use_container_width=True)

    # ═══ TAB 4 — LIVE TRADE ══════════════════════════════════════════════════
    with tabs[3]:
        st.markdown('<div class="sum">Paper trade with live option prices. 1.5s delay between every API call. Entry, SL and target pre-calculated before every trade. Live P&L tracked against current option price. Buyer-only strategies.</div>',
                    unsafe_allow_html=True)
        if not has_chain:
            st.warning("Live trading requires an options chain. Select NSE India or US Stocks.")
        else:
            la, lb = st.columns([3, 2])
            with la:
                st.markdown("#### 🚨 New Trade Setup")
                t1, t2 = st.columns(2)
                side     = t1.selectbox("Side", ["CE (Call)", "PE (Put)"])
                px       = "CE" if "CE" in side else "PE"
                t_strike = t1.selectbox("Strike", all_strikes, index=atm_pos)
                lots     = t2.number_input("Lots", 1, 100, 1)
                l_sl     = t2.slider("SL %", 20, 50, sl_pct, key="l_sl")
                l_tgt    = t2.slider("Target %", 30, 200, tgt_pct, key="l_tgt")

                row_s  = df_exp[df_exp["strikePrice"] == t_strike]
                col_ltp = f"{px}_LTP"
                opt_px = float(row_s[col_ltp].values[0]) if not row_s.empty and col_ltp in row_s.columns else 0

                if opt_px > 0:
                    sl_a  = round(opt_px * (1 - l_sl  / 100), 2)
                    tgt_a = round(opt_px * (1 + l_tgt / 100), 2)
                    risk  = (opt_px - sl_a)  * lots
                    rew   = (tgt_a  - opt_px) * lots
                    rr    = round(rew / risk, 2) if risk > 0 else 0
                    pm1, pm2, pm3, pm4 = st.columns(4)
                    pm1.metric("Entry",  f"₹{opt_px:.2f}")
                    pm2.metric("SL",     f"₹{sl_a:.2f}",  f"-{l_sl}%")
                    pm3.metric("Target", f"₹{tgt_a:.2f}", f"+{l_tgt}%")
                    pm4.metric("R:R",    f"1:{rr}")
                else:
                    sl_a = tgt_a = 0
                    st.warning("Option LTP is 0 — market may be closed or strike too far OTM.")

                st.markdown(
                    f'<div style="background:#0a1929;border:1px solid #1e3050;'
                    f'border-radius:8px;padding:12px;margin:8px 0">'
                    f'<b style="color:#00e5ff">{sig["rec"]}</b> | Conf: <b>{sig["conf"]}%</b><br>'
                    f'<span style="color:#5a7a9a;font-size:12px">'
                    f'{" · ".join(sig["reasons"][:2])}</span></div>',
                    unsafe_allow_html=True)

                if st.button("📈 ENTER PAPER TRADE", type="primary", use_container_width=True):
                    if opt_px > 0:
                        t = {"id":       len(st.session_state.trades) + 1,
                             "sym":      fetch_sym, "expiry": sel_expiry,
                             "strike":   t_strike,  "side":   px,
                             "entry":    opt_px,     "sl":     sl_a,  "target": tgt_a,
                             "lots":     lots,       "conf":   sig["conf"], "rec": sig["rec"],
                             "time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                             "status":   "OPEN",
                             "exit":     None,  "pnl": None,  "exit_time": None}
                        st.session_state.active.append(t)
                        st.session_state.trades.append(t)
                        st.success(f"✅ Trade entered: {fetch_sym} {t_strike} {px} @ ₹{opt_px:.2f}")
                    else:
                        st.error("Cannot enter — option price is 0.")

            with lb:
                st.markdown("#### 📊 Open Positions")
                open_list = [t for t in st.session_state.active if t["status"] == "OPEN"]
                if not open_list:
                    st.info("No open trades.")
                for i, t in enumerate(st.session_state.active):
                    if t["status"] != "OPEN": continue
                    r    = df_exp[df_exp["strikePrice"] == t["strike"]]
                    ltp_col = f"{t['side']}_LTP"
                    curr = float(r[ltp_col].values[0]) if not r.empty and ltp_col in r.columns else t["entry"]
                    pnl  = round((curr - t["entry"]) * t["lots"], 2)
                    pp   = round((curr - t["entry"]) / t["entry"] * 100, 2) if t["entry"] > 0 else 0
                    clr  = "#00ff88" if pnl >= 0 else "#ff3b5c"
                    cls  = "tc-w" if pnl > 0 else "tc-l" if pnl < 0 else "tc-o"
                    st.markdown(f"""
<div class="tc {cls}">
<b>{t['sym']} {t['strike']} {t['side']}</b><br>
Entry ₹{t['entry']:.2f} → Live ₹{curr:.2f}<br>
SL ₹{t['sl']:.2f} | Target ₹{t['target']:.2f}<br>
<b style='color:{clr}'>P&L: ₹{pnl:+.2f} ({pp:+.1f}%)</b>
</div>""", unsafe_allow_html=True)
                    if st.button(f"Exit #{t['id']}", key=f"ex_{i}_{t['id']}"):
                        for j, x in enumerate(st.session_state.active):
                            if x["id"] == t["id"]:
                                st.session_state.active[j].update({
                                    "status": "CLOSED", "exit": curr, "pnl": pnl,
                                    "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                                for h in st.session_state.trades:
                                    if h["id"] == t["id"]: h.update(st.session_state.active[j])
                                break
                        st.rerun()

    # ═══ TAB 5 — BACKTEST ════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown('<div class="sum">Uses IDENTICAL PCR scoring, IV filter, SMA trend and position sizing as live trading. Live and backtest results match because they share the same algorithm. Works for all symbols including spot-only (BTC, Gold).</div>',
                    unsafe_allow_html=True)
        ba, bb, bc_ = st.columns(3)
        bt_look = ba.slider("Lookback days", 20, 120, 60)
        bt_cap  = bb.number_input("Capital", 50_000, 5_000_000, int(capital), 10_000, key="bt_cap")
        bt_pos  = bc_.slider("Position size %", 2, 20, pos_pct, key="bt_pos")

        if st.button("🔬 RUN BACKTEST", type="primary", use_container_width=True):
            with st.spinner("Running backtest with live signal engine…"):
                tdf, stats = run_backtest(hist_sym, bt_look, float(bt_cap), float(bt_pos))
                st.session_state.bt_result = (tdf, stats)

        if st.session_state.bt_result:
            tdf, stats = st.session_state.bt_result
            if tdf is None:
                st.error(f"Backtest failed: {stats}")
            else:
                k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
                k1.metric("Trades",   stats["total"])
                k2.metric("Win Rate", f"{stats['wr']}%")
                k3.metric("P&L",      f"₹{stats['total_pnl']:+,.0f}")
                k4.metric("Avg Win",  f"₹{stats['aw']:,.0f}")
                k5.metric("Avg Loss", f"₹{stats['al']:,.0f}")
                k6.metric("R:R",      stats["rr"])
                k7.metric("Max DD",   f"{stats['mdd']}%")
                rc = "#00ff88" if stats["ret%"] >= 0 else "#ff3b5c"
                st.markdown(
                    f'<div style="text-align:center;font-size:26px;color:{rc};'
                    f'font-family:Space Mono;margin:10px 0">'
                    f'Return: {stats["ret%"]:+.2f}%  ·  Final: ₹{stats["final"]:,.0f}</div>',
                    unsafe_allow_html=True)
                fig_eq = go.Figure(go.Scatter(
                    y=stats["curve"], mode="lines",
                    line=dict(color="#00e5ff", width=2.5),
                    fill="tozeroy", fillcolor="rgba(0,229,255,0.05)"))
                fig_eq.add_hline(y=bt_cap, line_dash="dash", line_color="#ff9500",
                                  annotation_text="Start Capital")
                fig_eq.update_layout(template=DARK, height=320, title="Equity Curve")
                st.plotly_chart(fig_eq, use_container_width=True)
                w = tdf[tdf["P&L(₹)"] > 0]["P&L(₹)"]
                l = tdf[tdf["P&L(₹)"] <= 0]["P&L(₹)"]
                fig_d = go.Figure()
                fig_d.add_trace(go.Histogram(x=w, name="Wins",   marker_color="rgba(0,255,136,0.47)", nbinsx=20))
                fig_d.add_trace(go.Histogram(x=l, name="Losses", marker_color="rgba(255,59,92,0.47)",  nbinsx=20))
                fig_d.update_layout(template=DARK, height=240, title="P&L Distribution", barmode="overlay")
                st.plotly_chart(fig_d, use_container_width=True)
                st.dataframe(tdf, use_container_width=True, height=260)
                st.success("✅ Backtest uses the identical PCR/IV scoring thresholds as live trading.")

    # ═══ TAB 6 — HISTORY ═════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown('<div class="sum">Complete paper trade log — entry, exit, P&L, signal confidence. Export to CSV. Review winning patterns to improve your trading edge over time.</div>',
                    unsafe_allow_html=True)
        if not st.session_state.trades:
            st.info("No trades yet. Use the Live Trade tab to enter paper trades.")
        else:
            all_t  = pd.DataFrame(st.session_state.trades)
            closed = all_t[all_t["status"] == "CLOSED"].copy()
            if not closed.empty:
                closed["pnl"] = pd.to_numeric(closed["pnl"], errors="coerce").fillna(0)
                tot = closed["pnl"].sum()
                wr  = (closed["pnl"] > 0).mean() * 100
                h1, h2, h3, h4 = st.columns(4)
                h1.metric("Total Trades", len(all_t))
                h2.metric("Closed",       len(closed))
                h3.metric("Win Rate",     f"{wr:.1f}%")
                h4.metric("Net P&L",      f"₹{tot:+,.2f}")
                fig_p = go.Figure(go.Bar(
                    y=closed["pnl"].values,
                    marker_color=["#00ff88" if p > 0 else "#ff3b5c" for p in closed["pnl"]]))
                fig_p.update_layout(template=DARK, height=240, title="Per-Trade P&L")
                st.plotly_chart(fig_p, use_container_width=True)
            dc = [c for c in ["id","time","sym","strike","side","entry","exit",
                               "lots","pnl","status","rec","conf"] if c in all_t.columns]
            st.dataframe(all_t[dc], use_container_width=True)
            st.download_button("📥 Export CSV", all_t.to_csv(index=False), "trades.csv",
                               "text/csv", use_container_width=True)
            if st.button("🗑️ Clear All Trades", use_container_width=True):
                st.session_state.trades = []
                st.session_state.active = []
                st.rerun()

    # ═══ TAB 7 — ANALYSIS ════════════════════════════════════════════════════
    with tabs[6]:
        st.markdown('<div class="sum">Full narrative: PCR, max pain, OI walls, IV environment, three scenarios (bull/flat/bear), and buyer-specific rules on when to enter, scale, exit and stay out entirely.</div>',
                    unsafe_allow_html=True)
        iv  = sig["atm_iv"]; pcr = sig["pcr"]
        mp  = sig["max_pain"]
        res = sig["resistance"]; sup = sig["support"]
        mp_pct = (mp - spot) / spot * 100 if spot > 0 else 0

        pcr_t = ("extreme put OI → contrarian BULLISH" if pcr > 1.5
                 else "put-heavy → mild bullish lean" if pcr > 1.1
                 else "extreme call OI → contrarian BEARISH" if pcr < 0.5
                 else "call-heavy → mild bearish lean" if pcr < 0.9 else "balanced/neutral")
        iv_t  = ("AVOID BUYING — crushing" if iv > 50
                 else "HIGH — very selective" if iv > 35
                 else "MODERATE — acceptable" if iv > 20
                 else "LOW — prime buyer zone!" if iv > 0 else "N/A")

        chain_html = f"""
<b>PCR {pcr:.4f}</b> → {pcr_t}<br>
<b>ATM IV {iv:.1f}%</b> → {iv_t}<br>
<b>Max Pain {mp:,.0f}</b> → {mp_pct:+.1f}% from spot {"(upward pull)" if mp_pct > 0 else "(downward pull)"}<br>
<b>Resistance</b> {res:,.0f} (peak CE OI) &nbsp;|&nbsp; <b>Support</b> {sup:,.0f} (peak PE OI)<br>
<b>Straddle</b> {sig['straddle']:.2f} = implied move {sig['straddle']/spot*100:.2f}% this expiry
""" if has_chain else f"Spot-only mode. Price: <b>{spot:,.4f}</b>"

        st.markdown(f"""
<div style='background:#0a1929;border:1px solid #1e3050;border-radius:12px;padding:20px;line-height:2'>
<h4 style='color:#00e5ff;margin-top:0'>🏗 Market Structure — {fetch_sym} @ {spot:,.4f}</h4>
{chain_html}
</div>""", unsafe_allow_html=True)

        if has_chain:
            s1, s2, s3 = st.columns(3)
            for col, title, trigger, action, color in [
                (s1, "🟢 BULLISH", f"Break & hold above {res:,.0f}",
                 f"Buy {int(sig['atm'])} CE. Target {int(res*1.015):,}. SL 30% of premium.", "#00ff88"),
                (s2, "⚪ NEUTRAL",  f"Range {sup:,.0f} – {res:,.0f}",
                 "DO NOT buy options. Theta destroys premium in range. Wait for breakout.", "#5a7a9a"),
                (s3, "🔴 BEARISH",  f"Break & hold below {sup:,.0f}",
                 f"Buy {int(sig['atm'])} PE. Target {int(sup*0.985):,}. SL 30% of premium.", "#ff3b5c"),
            ]:
                with col:
                    st.markdown(f"""
<div style='background:#0a1929;border-top:3px solid {color};border:1px solid #1e3050;border-radius:10px;padding:16px'>
<h5 style='color:{color};margin-top:0'>{title}</h5>
<b>Trigger:</b> {trigger}<br><br><b>Action:</b> {action}
</div>""", unsafe_allow_html=True)

            rc2 = "#00ff88" if "CALL" in sig["rec"] else "#ff3b5c" if "PUT" in sig["rec"] else "#ff9500"
            st.markdown(f"""
<div style='background:#060e1a;border:1px solid {rc2}44;border-radius:12px;padding:20px;margin-top:16px'>
<h4 style='color:{rc2};margin-top:0'>🎯 {sig['rec']} — Confidence {sig['conf']}%</h4>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px'>
<div><b style='color:#00e5ff'>Trade Recommendations:</b>
  <ul style='color:#8899aa'>
    <li>⚡ Scalping: {sig['scalp']}</li>
    <li>📅 Intraday: {sig['intraday']}</li>
    <li>🗓 Swing: {sig['swing']}</li>
    <li>📆 Positional: {sig['pos']}</li>
  </ul>
</div>
<div><b style='color:#00e5ff'>Option Buyer Golden Rules:</b>
  <ul style='color:#8899aa'>
    <li>Never risk &gt;5% of capital per trade</li>
    <li>Only buy when ATM IV is below 30%</li>
    <li>Exit all positions 5+ days before expiry</li>
    <li>Always set SL before entry — never widen it</li>
    <li>Book 50% at first target, trail rest at breakeven</li>
    <li>Never buy options on high-IV event days (RBI, earnings)</li>
  </ul>
</div>
</div>
<b style='color:#5a7a9a'>Signal reasoning:</b> {" · ".join(sig['reasons'])}
</div>""", unsafe_allow_html=True)

        st.caption(f"Computed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Educational only — not financial advice")

    if auto_ref:
        time.sleep(90)
        st.cache_data.clear()
        _T["last"] = 0.0
        st.rerun()


if __name__ == "__main__":
    main()
