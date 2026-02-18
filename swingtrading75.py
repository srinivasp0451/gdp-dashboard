"""
╔══════════════════════════════════════════════════════════════════════╗
║        ALPHA EDGE — Professional Trading Strategy Platform          ║
║   Nifty | BankNifty | Sensex | Stocks | BTC | ETH | Forex | Gold   ║
╚══════════════════════════════════════════════════════════════════════╝

Install Requirements:
    pip install streamlit yfinance pandas numpy plotly scipy ta

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
from datetime import datetime, timedelta, date
import warnings
import math
from scipy.stats import norm
import time

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlphaEdge Trading Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

DARK_BG       = "#0A0E1A"
CARD_BG       = "#111827"
BORDER        = "#1F2937"
ACCENT_GOLD   = "#F59E0B"
ACCENT_GREEN  = "#10B981"
ACCENT_RED    = "#EF4444"
ACCENT_BLUE   = "#3B82F6"
ACCENT_PURPLE = "#8B5CF6"
TEXT_PRIMARY  = "#F9FAFB"
TEXT_MUTED    = "#6B7280"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Space Grotesk', sans-serif;
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
}}

.stApp {{ background-color: {DARK_BG}; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0D1117 0%, #111827 100%);
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color: {TEXT_PRIMARY} !important; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: {CARD_BG};
    border-radius: 12px;
    padding: 4px;
    border: 1px solid {BORDER};
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    color: {TEXT_MUTED} !important;
    font-weight: 500;
    font-size: 14px;
    padding: 8px 20px;
    transition: all 0.2s;
}}
.stTabs [aria-selected="true"] {{
    background: {ACCENT_GOLD} !important;
    color: #000 !important;
    font-weight: 700;
}}

/* Cards */
.metric-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}}
.metric-value {{
    font-size: 28px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin: 4px 0;
}}
.metric-label {{
    font-size: 12px;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 1px;
}}

/* Signal boxes */
.signal-box {{
    border-radius: 10px;
    padding: 18px 22px;
    margin: 8px 0;
    border-left: 4px solid;
}}
.signal-bull {{
    background: rgba(16,185,129,0.1);
    border-color: {ACCENT_GREEN};
}}
.signal-bear {{
    background: rgba(239,68,68,0.1);
    border-color: {ACCENT_RED};
}}
.signal-neutral {{
    background: rgba(245,158,11,0.1);
    border-color: {ACCENT_GOLD};
}}

/* Section headers */
.section-header {{
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: {ACCENT_GOLD};
    border-bottom: 1px solid {BORDER};
    padding-bottom: 10px;
    margin: 20px 0 16px;
}}

/* Strategy card */
.strategy-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
}}

/* Inputs */
.stSelectbox > div, .stNumberInput > div {{
    background: {CARD_BG} !important;
    border-color: {BORDER} !important;
    color: {TEXT_PRIMARY} !important;
}}

/* Dataframe */
.stDataFrame {{ background: {CARD_BG}; border-radius: 8px; }}

/* Button */
.stButton > button {{
    background: linear-gradient(135deg, {ACCENT_GOLD}, #D97706) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-size: 14px !important;
    letter-spacing: 0.5px;
    transition: all 0.2s !important;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(245,158,11,0.4) !important;
}}

/* Logo */
.logo-header {{
    text-align: center;
    padding: 20px 0 10px;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 20px;
}}
.logo-title {{
    font-size: 22px;
    font-weight: 700;
    background: linear-gradient(135deg, {ACCENT_GOLD}, #FCD34D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}}
.logo-sub {{
    font-size: 10px;
    color: {TEXT_MUTED};
    letter-spacing: 3px;
    text-transform: uppercase;
}}

div[data-testid="stMetricValue"] {{ color: {ACCENT_GOLD} !important; font-family: 'JetBrains Mono'; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ASSET UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────
ASSET_MAP = {
    "── INDIAN INDICES ──": None,
    "Nifty 50":            "^NSEI",
    "BankNifty":           "^NSEBANK",
    "Sensex":              "^BSESN",
    "Nifty IT":            "^CNXIT",
    "Nifty Pharma":        "^CNXPHARMA",
    "── INDIAN STOCKS ──": None,
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
    "── CRYPTO ──": None,
    "Bitcoin (BTC)":       "BTC-USD",
    "Ethereum (ETH)":      "ETH-USD",
    "── FOREX / COMMODITIES ──": None,
    "USD/INR":             "USDINR=X",
    "EUR/USD":             "EURUSD=X",
    "GBP/USD":             "GBPUSD=X",
    "Gold":                "GC=F",
    "Silver":              "SI=F",
    "Crude Oil":           "CL=F",
    "── CUSTOM ──": None,
    "Custom Ticker":       "CUSTOM",
}

STRATEGIES = {
    "Trend + Structure + Momentum (Pro)":  "TSM",
    "ORB — Opening Range Breakout":        "ORB",
    "VWAP + RSI Reversal/Trend":           "VWAP_RSI",
    "Swing: EMA + MACD + RSI":             "SWING",
    "Combined (All Signals)":              "COMBINED",
}

TRADE_TYPES = ["Intraday (15m)", "Swing (Daily)", "Positional (Weekly)"]


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()


def fetch_live(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="5d", interval="15m",
                         auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # EMAs
    for p in [9, 20, 50, 200]:
        d[f"EMA{p}"] = d["Close"].ewm(span=p, adjust=False).mean()
    # SMAs
    d["SMA20"] = d["Close"].rolling(20).mean()
    d["SMA50"] = d["Close"].rolling(50).mean()

    # RSI
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_Hist"] = d["MACD"] - d["MACD_Signal"]

    # Bollinger Bands
    d["BB_Mid"]   = d["Close"].rolling(20).mean()
    std20         = d["Close"].rolling(20).std()
    d["BB_Upper"] = d["BB_Mid"] + 2 * std20
    d["BB_Lower"] = d["BB_Mid"] - 2 * std20
    d["BB_Width"] = (d["BB_Upper"] - d["BB_Lower"]) / d["BB_Mid"]

    # ATR
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - d["Close"].shift()).abs(),
        (d["Low"]  - d["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()

    # VWAP (rolling daily proxy)
    d["TP"] = (d["High"] + d["Low"] + d["Close"]) / 3
    d["VWAP"] = (d["TP"] * d["Volume"]).rolling(14).sum() / d["Volume"].rolling(14).sum()

    # Volume MA
    d["Vol_MA20"] = d["Volume"].rolling(20).mean()
    d["Vol_Ratio"] = d["Volume"] / d["Vol_MA20"]

    # Stochastic
    low14  = d["Low"].rolling(14).min()
    high14 = d["High"].rolling(14).max()
    d["Stoch_K"] = 100 * (d["Close"] - low14) / (high14 - low14).replace(0, np.nan)
    d["Stoch_D"] = d["Stoch_K"].rolling(3).mean()

    # Support / Resistance (pivot-based)
    d["Pivot"]    = (d["High"] + d["Low"] + d["Close"]) / 3
    d["R1"]       = 2 * d["Pivot"] - d["Low"]
    d["S1"]       = 2 * d["Pivot"] - d["High"]
    d["R2"]       = d["Pivot"] + (d["High"] - d["Low"])
    d["S2"]       = d["Pivot"] - (d["High"] - d["Low"])

    # Candle features
    d["Body"]     = (d["Close"] - d["Open"]).abs()
    d["Range"]    = d["High"] - d["Low"]
    d["UpCandle"] = d["Close"] > d["Open"]

    return d


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY: TSM — Trend + Structure + Momentum
# ─────────────────────────────────────────────────────────────────────────────
def strategy_tsm(df: pd.DataFrame, atr_mult_sl: float = 1.5,
                 rr: float = 2.0, trail_sl: bool = True,
                 trail_pct: float = 1.5) -> pd.DataFrame:
    d = df.copy()

    bullish_trend = (d["Close"] > d["EMA20"]) & (d["EMA20"] > d["EMA50"])
    bearish_trend = (d["Close"] < d["EMA20"]) & (d["EMA20"] < d["EMA50"])

    # Breakout of previous high/low with volume
    prev_high = d["High"].shift(1)
    prev_low  = d["Low"].shift(1)
    vol_ok    = d["Vol_Ratio"] >= 1.5

    # Momentum filters
    rsi_bull = (d["RSI"] > 50) & (d["RSI"] < 75)
    rsi_bear = (d["RSI"] < 50) & (d["RSI"] > 25)
    macd_bull = d["MACD"] > d["MACD_Signal"]
    macd_bear = d["MACD"] < d["MACD_Signal"]

    d["Signal"] = 0
    buy_cond  = bullish_trend & (d["High"] > prev_high) & vol_ok & rsi_bull & macd_bull
    sell_cond = bearish_trend & (d["Low"]  < prev_low)  & vol_ok & rsi_bear & macd_bear

    d.loc[buy_cond,  "Signal"] =  1
    d.loc[sell_cond, "Signal"] = -1
    d["Strategy"] = "TSM"
    return d


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY: ORB — Opening Range Breakout
# ─────────────────────────────────────────────────────────────────────────────
def strategy_orb(df: pd.DataFrame, atr_mult_sl: float = 1.0,
                 rr: float = 1.5, trail_sl: bool = True,
                 trail_pct: float = 1.2) -> pd.DataFrame:
    d = df.copy()
    d["Signal"] = 0

    # For daily data: use previous day's range as proxy for ORB
    d["ORB_High"] = d["High"].shift(1)
    d["ORB_Low"]  = d["Low"].shift(1)

    # Breakout of yesterday's range with volume and trend
    above_ema20 = d["Close"] > d["EMA20"]
    below_ema20 = d["Close"] < d["EMA20"]
    vol_spike   = d["Vol_Ratio"] > 1.8

    buy_cond  = (d["High"] > d["ORB_High"]) & vol_spike & above_ema20 & (d["RSI"] > 50)
    sell_cond = (d["Low"]  < d["ORB_Low"])  & vol_spike & below_ema20 & (d["RSI"] < 50)

    d.loc[buy_cond,  "Signal"] =  1
    d.loc[sell_cond, "Signal"] = -1
    d["Strategy"] = "ORB"
    return d


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY: VWAP + RSI
# ─────────────────────────────────────────────────────────────────────────────
def strategy_vwap_rsi(df: pd.DataFrame, atr_mult_sl: float = 1.2,
                      rr: float = 1.8, trail_sl: bool = True,
                      trail_pct: float = 1.0) -> pd.DataFrame:
    d = df.copy()
    d["Signal"] = 0

    above_vwap  = d["Close"] > d["VWAP"]
    below_vwap  = d["Close"] < d["VWAP"]
    rsi_pullback_bull = (d["RSI"] > 40) & (d["RSI"] < 60)
    rsi_pullback_bear = (d["RSI"] > 40) & (d["RSI"] < 60)

    # Dip to VWAP in uptrend → Buy
    price_near_vwap = (d["Low"] <= d["VWAP"] * 1.003) & (d["Close"] > d["VWAP"] * 0.997)
    buy_cond  = above_vwap & price_near_vwap & rsi_pullback_bull & (d["EMA20"] > d["EMA50"])

    # Rally to VWAP in downtrend → Sell
    price_near_vwap_r = (d["High"] >= d["VWAP"] * 0.997) & (d["Close"] < d["VWAP"] * 1.003)
    sell_cond = below_vwap & price_near_vwap_r & rsi_pullback_bear & (d["EMA20"] < d["EMA50"])

    d.loc[buy_cond,  "Signal"] =  1
    d.loc[sell_cond, "Signal"] = -1
    d["Strategy"] = "VWAP_RSI"
    return d


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY: SWING — EMA + MACD + RSI
# ─────────────────────────────────────────────────────────────────────────────
def strategy_swing(df: pd.DataFrame, atr_mult_sl: float = 2.0,
                   rr: float = 2.5, trail_sl: bool = True,
                   trail_pct: float = 2.0) -> pd.DataFrame:
    d = df.copy()
    d["Signal"] = 0

    # Golden/Death Cross
    golden_cross = (d["EMA20"] > d["EMA50"]) & (d["EMA20"].shift(1) <= d["EMA50"].shift(1))
    death_cross  = (d["EMA20"] < d["EMA50"]) & (d["EMA20"].shift(1) >= d["EMA50"].shift(1))

    # MACD cross
    macd_cross_up   = (d["MACD"] > d["MACD_Signal"]) & (d["MACD"].shift(1) <= d["MACD_Signal"].shift(1))
    macd_cross_down = (d["MACD"] < d["MACD_Signal"]) & (d["MACD"].shift(1) >= d["MACD_Signal"].shift(1))

    # RSI zones
    rsi_healthy_bull = (d["RSI"] > 45) & (d["RSI"] < 70)
    rsi_healthy_bear = (d["RSI"] > 30) & (d["RSI"] < 55)

    # Volume confirmation
    vol_confirm = d["Vol_Ratio"] > 1.3

    buy_cond  = (golden_cross | macd_cross_up)  & rsi_healthy_bull & vol_confirm
    sell_cond = (death_cross  | macd_cross_down) & rsi_healthy_bear & vol_confirm

    d.loc[buy_cond,  "Signal"] =  1
    d.loc[sell_cond, "Signal"] = -1
    d["Strategy"] = "SWING"
    return d


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED STRATEGY
# ─────────────────────────────────────────────────────────────────────────────
def strategy_combined(df: pd.DataFrame, atr_mult_sl: float = 1.5,
                      rr: float = 2.0, trail_sl: bool = True,
                      trail_pct: float = 1.5) -> pd.DataFrame:
    d1 = strategy_tsm(df, atr_mult_sl, rr, trail_sl, trail_pct)
    d2 = strategy_orb(df, atr_mult_sl, rr, trail_sl, trail_pct)
    d3 = strategy_vwap_rsi(df, atr_mult_sl, rr, trail_sl, trail_pct)
    d4 = strategy_swing(df, atr_mult_sl, rr, trail_sl, trail_pct)

    combo = df.copy()
    votes = d1["Signal"] + d2["Signal"] + d3["Signal"] + d4["Signal"]
    combo["Signal"] = 0
    combo.loc[votes >= 2,  "Signal"] =  1   # ≥2 strategies agree → Buy
    combo.loc[votes <= -2, "Signal"] = -1   # ≥2 strategies agree → Sell
    combo["Strategy"] = "COMBINED"
    return combo


def get_strategy_df(df: pd.DataFrame, strat_key: str,
                    atr_sl: float, rr: float,
                    trail_sl: bool, trail_pct: float) -> pd.DataFrame:
    fns = {
        "TSM":      strategy_tsm,
        "ORB":      strategy_orb,
        "VWAP_RSI": strategy_vwap_rsi,
        "SWING":    strategy_swing,
        "COMBINED": strategy_combined,
    }
    return fns[strat_key](df, atr_sl, rr, trail_sl, trail_pct)


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTER
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df_sig: pd.DataFrame, capital: float, risk_pct: float,
                 atr_sl_mult: float, rr: float,
                 trail_sl: bool, trail_pct: float,
                 trail_tgt: bool, trail_tgt_pct: float) -> dict:
    """
    Full backtest engine with:
    - Fixed SL / Target
    - Trailing SL (ratchets up with price)
    - Trailing Target (moves target up)
    Returns trade log + equity curve + metrics
    """
    trades   = []
    equity   = [capital]
    equity_dates = [df_sig.index[0]]
    cash     = capital
    position = 0   # 1 = long, -1 = short, 0 = flat
    entry_px = 0.0
    sl       = 0.0
    tgt      = 0.0
    trail_sl_px  = 0.0
    trail_tgt_px = 0.0
    entry_date   = None
    atr_at_entry = 0.0

    df = df_sig.dropna(subset=["ATR", "Signal"]).copy()

    for i in range(1, len(df)):
        row   = df.iloc[i]
        prev  = df.iloc[i - 1]
        price = float(row["Close"])
        atr   = float(row["ATR"]) if row["ATR"] > 0 else price * 0.01

        # ── MANAGE OPEN POSITION ──────────────────────────────────────────
        if position != 0:
            # Update trailing SL
            if trail_sl and position == 1:
                new_trail = price * (1 - trail_pct / 100)
                trail_sl_px = max(trail_sl_px, new_trail)
                effective_sl = max(sl, trail_sl_px)
            elif trail_sl and position == -1:
                new_trail = price * (1 + trail_pct / 100)
                trail_sl_px = min(trail_sl_px, new_trail)
                effective_sl = min(sl, trail_sl_px)
            else:
                effective_sl = sl

            # Update trailing target
            if trail_tgt and position == 1:
                new_trail_tgt = price * (1 + trail_tgt_pct / 100)
                trail_tgt_px  = max(trail_tgt_px, new_trail_tgt)
            elif trail_tgt and position == -1:
                new_trail_tgt = price * (1 - trail_tgt_pct / 100)
                trail_tgt_px  = min(trail_tgt_px, new_trail_tgt)

            exit_price  = None
            exit_reason = None

            # Check SL
            if position == 1 and row["Low"] <= effective_sl:
                exit_price  = effective_sl
                exit_reason = "SL Hit" if not trail_sl else "Trail SL Hit"
            elif position == -1 and row["High"] >= effective_sl:
                exit_price  = effective_sl
                exit_reason = "SL Hit" if not trail_sl else "Trail SL Hit"

            # Check Target (static or trailing)
            effective_tgt = trail_tgt_px if (trail_tgt and trail_tgt_px != 0) else tgt
            if exit_price is None:
                if position == 1 and row["High"] >= effective_tgt:
                    exit_price  = effective_tgt
                    exit_reason = "Target Hit" if not trail_tgt else "Trail Target Hit"
                elif position == -1 and row["Low"] <= effective_tgt:
                    exit_price  = effective_tgt
                    exit_reason = "Target Hit" if not trail_tgt else "Trail Target Hit"

            # Forced exit on opposite signal
            if exit_price is None and row["Signal"] == -position:
                exit_price  = price
                exit_reason = "Signal Reversal"

            if exit_price is not None:
                pnl_pct = (exit_price - entry_px) / entry_px * position * 100
                risk_amt = cash * risk_pct / 100
                qty      = risk_amt / (abs(entry_px - sl) + 1e-9)
                pnl_abs  = (exit_price - entry_px) * position * qty
                cash    += pnl_abs
                trades.append({
                    "Entry Date":  entry_date,
                    "Exit Date":   row.name,
                    "Direction":   "LONG" if position == 1 else "SHORT",
                    "Entry":       round(entry_px, 2),
                    "Exit":        round(exit_price, 2),
                    "SL":          round(sl, 2),
                    "Target":      round(tgt, 2),
                    "P&L %":       round(pnl_pct, 2),
                    "P&L ₹":       round(pnl_abs, 2),
                    "Capital":     round(cash, 2),
                    "Exit Reason": exit_reason,
                    "ATR":         round(atr_at_entry, 2),
                })
                position    = 0
                trail_sl_px = 0.0
                trail_tgt_px= 0.0

        # ── ENTER NEW POSITION ────────────────────────────────────────────
        if position == 0 and row["Signal"] != 0:
            position    = int(row["Signal"])
            entry_px    = price
            entry_date  = row.name
            atr_at_entry= atr

            if position == 1:
                sl  = entry_px - atr_sl_mult * atr
                tgt = entry_px + rr * atr_sl_mult * atr
                trail_sl_px  = entry_px * (1 - trail_pct / 100)
                trail_tgt_px = tgt
            else:
                sl  = entry_px + atr_sl_mult * atr
                tgt = entry_px - rr * atr_sl_mult * atr
                trail_sl_px  = entry_px * (1 + trail_pct / 100)
                trail_tgt_px = tgt

        equity.append(cash)
        equity_dates.append(row.name)

    if not trades:
        return {"trades": pd.DataFrame(), "equity": pd.Series(equity, index=equity_dates),
                "metrics": {}}

    trade_df = pd.DataFrame(trades)
    wins     = trade_df[trade_df["P&L %"] > 0]
    losses   = trade_df[trade_df["P&L %"] <= 0]

    win_rate    = len(wins) / len(trade_df) * 100 if len(trade_df) else 0
    avg_win     = wins["P&L %"].mean()   if len(wins)   else 0
    avg_loss    = losses["P&L %"].mean() if len(losses) else 0
    profit_factor = (wins["P&L ₹"].sum() / abs(losses["P&L ₹"].sum())
                     if losses["P&L ₹"].sum() < 0 else float("inf"))

    equity_s    = pd.Series(equity, index=equity_dates)
    running_max = equity_s.cummax()
    drawdown    = (equity_s - running_max) / running_max * 100
    max_dd      = drawdown.min()

    total_pnl   = cash - capital
    total_ret   = total_pnl / capital * 100
    n_days      = max((df.index[-1] - df.index[0]).days, 1)
    cagr        = ((cash / capital) ** (365 / n_days) - 1) * 100

    sharpe = 0.0
    if len(trade_df) > 1:
        ret_series = trade_df["P&L %"] / 100
        sharpe = (ret_series.mean() / (ret_series.std() + 1e-9)) * np.sqrt(252)

    metrics = {
        "Total Trades":    len(trade_df),
        "Win Rate %":      round(win_rate, 1),
        "Avg Win %":       round(avg_win, 2),
        "Avg Loss %":      round(avg_loss, 2),
        "Profit Factor":   round(profit_factor, 2),
        "Max Drawdown %":  round(max_dd, 2),
        "Total Return %":  round(total_ret, 2),
        "CAGR %":          round(cagr, 2),
        "Sharpe Ratio":    round(sharpe, 2),
        "Net P&L":         round(total_pnl, 2),
        "Final Capital":   round(cash, 2),
    }

    return {
        "trades":  trade_df,
        "equity":  equity_s,
        "metrics": metrics,
        "drawdown": drawdown,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES OPTIONS PRICING
# ─────────────────────────────────────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, option_type="CE"):
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == "CE" else max(K - S, 0)
        return intrinsic, 0, 0, 0, 0

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "CE":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)

    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) -
             r * K * math.exp(-r * T) * (norm.cdf(d2) if option_type == "CE" else norm.cdf(-d2))) / 365
    vega  = S * norm.pdf(d1) * math.sqrt(T) / 100

    return round(price, 2), round(delta, 4), round(gamma, 6), round(theta, 2), round(vega, 4)


# ─────────────────────────────────────────────────────────────────────────────
# LIVE ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def generate_analysis(df: pd.DataFrame, ticker: str, asset_name: str) -> dict:
    """Generate full analysis report for current market state."""
    if df is None or len(df) < 50:
        return {}

    d = df.tail(2)
    cur  = d.iloc[-1]
    prev = d.iloc[-2]

    price   = float(cur["Close"])
    atr     = float(cur["ATR"])
    rsi     = float(cur["RSI"])
    macd    = float(cur["MACD"])
    macd_s  = float(cur["MACD_Signal"])
    ema20   = float(cur["EMA20"])
    ema50   = float(cur["EMA50"])
    ema200  = float(cur["EMA200"])
    vwap    = float(cur["VWAP"])
    vol_r   = float(cur["Vol_Ratio"])
    bb_u    = float(cur["BB_Upper"])
    bb_l    = float(cur["BB_Lower"])
    bb_w    = float(cur["BB_Width"])
    pivot   = float(cur["Pivot"])
    r1      = float(cur["R1"])
    s1      = float(cur["S1"])
    r2      = float(cur["R2"])
    s2      = float(cur["S2"])
    stoch_k = float(cur["Stoch_K"])

    # ── TREND SCORE (0–10) ──────────────────────────────────────────────────
    trend_score = 0
    trend_notes = []

    if price > ema20:
        trend_score += 2; trend_notes.append("✅ Price above EMA20")
    else:
        trend_notes.append("❌ Price below EMA20")

    if price > ema50:
        trend_score += 2; trend_notes.append("✅ Price above EMA50")
    else:
        trend_notes.append("❌ Price below EMA50")

    if price > ema200:
        trend_score += 3; trend_notes.append("✅ Price above EMA200 (Long-term bull)")
    else:
        trend_notes.append("❌ Price below EMA200 (Long-term bear)")

    if ema20 > ema50:
        trend_score += 2; trend_notes.append("✅ EMA20 > EMA50 (Bullish cross)")
    else:
        trend_notes.append("❌ EMA20 < EMA50 (Bearish cross)")

    if price > vwap:
        trend_score += 1; trend_notes.append("✅ Price above VWAP")
    else:
        trend_notes.append("❌ Price below VWAP")

    # ── MOMENTUM SCORE ──────────────────────────────────────────────────────
    mom_score = 0
    mom_notes = []

    if 50 < rsi < 70:
        mom_score += 3; mom_notes.append(f"✅ RSI {rsi:.1f} — Healthy bull momentum")
    elif rsi >= 70:
        mom_score += 1; mom_notes.append(f"⚠️ RSI {rsi:.1f} — Overbought, caution")
    elif 30 < rsi <= 50:
        mom_notes.append(f"⚠️ RSI {rsi:.1f} — Weak / bearish momentum")
    else:
        mom_score -= 1; mom_notes.append(f"❌ RSI {rsi:.1f} — Oversold")

    if macd > macd_s:
        mom_score += 3; mom_notes.append("✅ MACD bullish cross / positive")
    else:
        mom_notes.append("❌ MACD bearish — below signal")

    if vol_r > 1.5:
        mom_score += 2; mom_notes.append(f"✅ Volume {vol_r:.1f}x avg — Strong participation")
    elif vol_r > 1.0:
        mom_score += 1; mom_notes.append(f"⚠️ Volume {vol_r:.1f}x avg — Average")
    else:
        mom_notes.append(f"❌ Volume {vol_r:.1f}x avg — Weak")

    if 20 < stoch_k < 80:
        mom_score += 2; mom_notes.append(f"✅ Stochastic {stoch_k:.1f} — Not extreme")
    elif stoch_k > 80:
        mom_notes.append(f"⚠️ Stochastic {stoch_k:.1f} — Overbought zone")
    else:
        mom_notes.append(f"⚠️ Stochastic {stoch_k:.1f} — Oversold zone")

    # ── OVERALL SIGNAL ──────────────────────────────────────────────────────
    combined = trend_score + mom_score
    if   combined >= 14: bias, strength = "STRONG BUY",  "🟢"
    elif combined >= 10: bias, strength = "BUY",          "🟢"
    elif combined >= 7:  bias, strength = "WEAK BUY",     "🟡"
    elif combined >= 4:  bias, strength = "NEUTRAL",      "⚪"
    elif combined >= 1:  bias, strength = "WEAK SELL",    "🟠"
    elif combined >= -2: bias, strength = "SELL",         "🔴"
    else:                bias, strength = "STRONG SELL",  "🔴"

    # ── ENTRY / SL / TARGETS ────────────────────────────────────────────────
    is_long = "BUY" in bias
    if is_long:
        entry    = price
        sl_fixed = round(max(s1, price - 1.5 * atr), 2)
        sl_atr   = round(price - 1.5 * atr, 2)
        tgt1     = round(price + 1.5 * atr, 2)
        tgt2     = round(r1, 2)
        tgt3     = round(r2, 2)
        trail_sl_start = round(price - atr, 2)
        trail_pct_val  = 1.5
    else:
        entry    = price
        sl_fixed = round(min(r1, price + 1.5 * atr), 2)
        sl_atr   = round(price + 1.5 * atr, 2)
        tgt1     = round(price - 1.5 * atr, 2)
        tgt2     = round(s1, 2)
        tgt3     = round(s2, 2)
        trail_sl_start = round(price + atr, 2)
        trail_pct_val  = 1.5

    # ── KEY LEVELS ──────────────────────────────────────────────────────────
    levels = {
        "Pivot": round(pivot, 2),
        "R1": round(r1, 2), "R2": round(r2, 2),
        "S1": round(s1, 2), "S2": round(s2, 2),
        "BB Upper": round(bb_u, 2), "BB Lower": round(bb_l, 2),
        "VWAP": round(vwap, 2),
        "EMA20": round(ema20, 2), "EMA50": round(ema50, 2), "EMA200": round(ema200, 2),
    }

    # IV estimate (approximated from BB width as a proxy)
    iv_proxy = round(bb_w * 100, 1)
    iv_rank  = "LOW ✅" if iv_proxy < 5 else ("MEDIUM ⚠️" if iv_proxy < 10 else "HIGH ❌")

    # Options recommendation
    if "BUY" in bias:
        opt_rec = f"Buy CE (ATM or 1 strike ITM)\nBest Strike: ≈ {int(price // 100) * 100}"
        opt_exp = "Use nearest weekly expiry for intraday, next month for swing"
    else:
        opt_rec = f"Buy PE (ATM or 1 strike ITM)\nBest Strike: ≈ {int(price // 100) * 100}"
        opt_exp = "Use nearest weekly expiry for intraday, next month for swing"

    return {
        "price":         price,
        "bias":          bias,
        "strength":      strength,
        "trend_score":   trend_score,
        "mom_score":     mom_score,
        "combined":      combined,
        "trend_notes":   trend_notes,
        "mom_notes":     mom_notes,
        "entry":         round(entry, 2),
        "sl_fixed":      sl_fixed,
        "sl_atr":        sl_atr,
        "tgt1":          tgt1,
        "tgt2":          tgt2,
        "tgt3":          tgt3,
        "trail_sl_start": trail_sl_start,
        "trail_pct":     trail_pct_val,
        "atr":           round(atr, 2),
        "rsi":           round(rsi, 2),
        "macd":          round(macd, 4),
        "vol_ratio":     round(vol_r, 2),
        "iv_proxy":      iv_proxy,
        "iv_rank":       iv_rank,
        "levels":        levels,
        "opt_rec":       opt_rec,
        "opt_exp":       opt_exp,
        "is_long":       is_long,
        "bb_width":      round(bb_w, 4),
        "stoch_k":       round(stoch_k, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def build_price_chart(df: pd.DataFrame, signals_df: pd.DataFrame = None,
                      analysis: dict = None, title: str = "") -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        vertical_spacing=0.02,
        subplot_titles=("", "Volume", "RSI", "MACD")
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color=ACCENT_GREEN,
        decreasing_line_color=ACCENT_RED,
        name="Price"
    ), row=1, col=1)

    # EMAs
    for ema, color in [("EMA20", "#3B82F6"), ("EMA50", "#F59E0B"), ("EMA200", "#8B5CF6")]:
        if ema in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[ema], name=ema,
                line=dict(color=color, width=1.2), opacity=0.85
            ), row=1, col=1)

    # Bollinger Bands
    if "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"],
            line=dict(color="rgba(99,102,241,0.4)", width=1, dash="dot"),
            name="BB Upper", showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"],
            line=dict(color="rgba(99,102,241,0.4)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(99,102,241,0.05)",
            name="BB", showlegend=False
        ), row=1, col=1)

    # VWAP
    if "VWAP" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VWAP"],
            line=dict(color="#F97316", width=1.5, dash="dash"),
            name="VWAP"
        ), row=1, col=1)

    # Signals
    if signals_df is not None and "Signal" in signals_df.columns:
        buy_sig  = signals_df[signals_df["Signal"] ==  1]
        sell_sig = signals_df[signals_df["Signal"] == -1]
        if len(buy_sig):
            fig.add_trace(go.Scatter(
                x=buy_sig.index, y=buy_sig["Low"] * 0.995,
                mode="markers",
                marker=dict(symbol="triangle-up", size=10,
                            color=ACCENT_GREEN, line=dict(color="white", width=1)),
                name="BUY Signal"
            ), row=1, col=1)
        if len(sell_sig):
            fig.add_trace(go.Scatter(
                x=sell_sig.index, y=sell_sig["High"] * 1.005,
                mode="markers",
                marker=dict(symbol="triangle-down", size=10,
                            color=ACCENT_RED, line=dict(color="white", width=1)),
                name="SELL Signal"
            ), row=1, col=1)

    # Analysis levels
    if analysis:
        for label, val, clr in [
            ("Entry",  analysis["entry"],    ACCENT_GOLD),
            ("SL",     analysis["sl_fixed"], ACCENT_RED),
            ("Tgt 1",  analysis["tgt1"],     ACCENT_GREEN),
            ("Tgt 2",  analysis["tgt2"],     "#10B981"),
            ("Tgt 3",  analysis["tgt3"],     "#059669"),
        ]:
            fig.add_hline(y=val, line_color=clr, line_dash="dash",
                          line_width=1.5,
                          annotation_text=f"  {label}: {val}",
                          annotation_font_color=clr,
                          row=1, col=1)

    # Volume
    colors = [ACCENT_GREEN if c >= o else ACCENT_RED
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=colors, name="Volume", opacity=0.7
    ), row=2, col=1)
    if "Vol_MA20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Vol_MA20"],
            line=dict(color=ACCENT_GOLD, width=1.2),
            name="Vol MA20"
        ), row=2, col=1)

    # RSI
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"],
            line=dict(color=ACCENT_PURPLE, width=1.5),
            name="RSI"
        ), row=3, col=1)
        for level, color in [(70, ACCENT_RED), (50, ACCENT_GOLD), (30, ACCENT_GREEN)]:
            fig.add_hline(y=level, line_color=color, line_dash="dot",
                          line_width=1, row=3, col=1)

    # MACD
    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"],
            line=dict(color=ACCENT_BLUE, width=1.5), name="MACD"
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_Signal"],
            line=dict(color=ACCENT_RED, width=1.2), name="Signal"
        ), row=4, col=1)
        macd_colors = [ACCENT_GREEN if v >= 0 else ACCENT_RED
                       for v in df["MACD_Hist"]]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_Hist"],
            marker_color=macd_colors, name="Histogram", opacity=0.6
        ), row=4, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=TEXT_PRIMARY)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_MUTED, size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis_rangeslider_visible=False,
        height=700,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor="#1F2937", showgrid=True, row=i, col=1)
        fig.update_yaxes(gridcolor="#1F2937", showgrid=True, row=i, col=1)

    return fig


def build_equity_chart(equity: pd.Series, drawdown: pd.Series,
                       trade_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.04,
        subplot_titles=("Equity Curve", "Drawdown %")
    )

    # Equity
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity,
        fill="tozeroy", fillcolor="rgba(59,130,246,0.1)",
        line=dict(color=ACCENT_BLUE, width=2),
        name="Portfolio Value"
    ), row=1, col=1)

    # Trade markers
    if trade_df is not None and len(trade_df):
        wins  = trade_df[trade_df["P&L %"] > 0]
        losses= trade_df[trade_df["P&L %"] <= 0]
        if len(wins):
            fig.add_trace(go.Scatter(
                x=wins["Exit Date"], y=wins["Capital"],
                mode="markers",
                marker=dict(symbol="circle", size=7, color=ACCENT_GREEN),
                name="Win"
            ), row=1, col=1)
        if len(losses):
            fig.add_trace(go.Scatter(
                x=losses["Exit Date"], y=losses["Capital"],
                mode="markers",
                marker=dict(symbol="x", size=7, color=ACCENT_RED),
                name="Loss"
            ), row=1, col=1)

    # Drawdown
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown,
        fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        line=dict(color=ACCENT_RED, width=1.5),
        name="Drawdown %"
    ), row=2, col=1)

    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_MUTED, size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    for i in range(1, 3):
        fig.update_xaxes(gridcolor="#1F2937", row=i, col=1)
        fig.update_yaxes(gridcolor="#1F2937", row=i, col=1)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-header">
        <div class="logo-title">⚡ AlphaEdge</div>
        <div class="logo-sub">Pro Trading Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Asset Selection")
    asset_names    = [k for k, v in ASSET_MAP.items() if v is not None and not k.startswith("──")]
    asset_name_sel = st.selectbox("Asset", asset_names, index=0)
    ticker_raw     = ASSET_MAP.get(asset_name_sel, "^NSEI")

    if ticker_raw == "CUSTOM":
        ticker = st.text_input("Custom Ticker (e.g. AAPL, MSFT, NIFTY)", "AAPL")
    else:
        ticker = ticker_raw

    st.markdown("---")
    st.markdown("### 🧠 Strategy")
    strategy_name = st.selectbox("Strategy", list(STRATEGIES.keys()), index=4)
    strategy_key  = STRATEGIES[strategy_name]

    trade_type = st.selectbox("Trade Type", TRADE_TYPES, index=0)
    interval_map = {
        "Intraday (15m)": ("6mo",  "1d"),
        "Swing (Daily)":  ("2y",   "1d"),
        "Positional (Weekly)": ("5y","1wk"),
    }
    period, interval = interval_map[trade_type]

    st.markdown("---")
    st.markdown("### ⚙️ Parameters")
    capital    = st.number_input("Capital (₹)", 50000, 10000000, 100000, step=10000)
    risk_pct   = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5)
    atr_sl     = st.slider("ATR SL Multiplier",  0.5, 4.0, 1.5, 0.25)
    rr_ratio   = st.slider("Risk:Reward Ratio",  1.0, 5.0, 2.0, 0.25)

    st.markdown("---")
    st.markdown("### 🎯 Stop Loss & Target")
    col_a, col_b = st.columns(2)
    with col_a:
        trail_sl  = st.checkbox("Trailing SL",  True)
        trail_tgt = st.checkbox("Trailing Tgt", True)
    with col_b:
        trail_pct_val  = st.number_input("Trail SL %",  0.5, 10.0, 1.5, 0.25)
        trail_tgt_val  = st.number_input("Trail Tgt %", 0.5, 10.0, 2.0, 0.25)

    st.markdown("---")
    st.markdown("### 📈 Options")
    spot_price = st.number_input("Spot Price (for BS pricing)", 100.0, 100000.0, 19000.0, 100.0)
    iv_pct     = st.slider("IV % (Implied Volatility)", 5.0, 100.0, 18.0, 1.0)
    days_exp   = st.slider("Days to Expiry", 1, 90, 7, 1)
    risk_free  = st.slider("Risk-Free Rate %", 4.0, 9.0, 6.5, 0.5)
    opt_type   = st.radio("Option Type", ["CE (Call)", "PE (Put)"])

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True)

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
    st.markdown('<div class="section-header">📊 Strategy Backtesting Engine</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"""
        <div class="strategy-card">
            <b style="color:{ACCENT_GOLD}">Asset</b><br>{asset_name_sel}<br>
            <small style="color:{TEXT_MUTED}">{ticker}</small><br><br>
            <b style="color:{ACCENT_GOLD}">Strategy</b><br>
            <small>{strategy_name}</small><br><br>
            <b style="color:{ACCENT_GOLD}">Trade Type</b><br>
            <small>{trade_type}</small><br><br>
            <b style="color:{ACCENT_GOLD}">Capital</b><br>₹{capital:,}<br><br>
            <b style="color:{ACCENT_GOLD}">Risk/Trade</b><br>{risk_pct}%<br><br>
            <b style="color:{ACCENT_GOLD}">ATR SL Mult</b><br>{atr_sl}x<br><br>
            <b style="color:{ACCENT_GOLD}">R:R</b><br>1:{rr_ratio}<br><br>
            <b style="color:{ACCENT_GOLD}">Trailing SL</b><br>{"ON ✅" if trail_sl else "OFF"}<br>
            <b style="color:{ACCENT_GOLD}">Trailing Tgt</b><br>{"ON ✅" if trail_tgt else "OFF"}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if run_btn or st.session_state.get("bt_run"):
            st.session_state["bt_run"] = True
            with st.spinner("⏳ Fetching data and running backtest..."):
                raw_df = fetch_data(ticker, period=period, interval=interval)

            if raw_df.empty:
                st.error("❌ Could not fetch data. Check the ticker symbol.")
            else:
                df_ind = compute_indicators(raw_df)
                df_sig = get_strategy_df(df_ind, strategy_key, atr_sl, rr_ratio,
                                         trail_sl, trail_pct_val)

                result = run_backtest(
                    df_sig, capital, risk_pct, atr_sl, rr_ratio,
                    trail_sl, trail_pct_val, trail_tgt, trail_tgt_val
                )
                m = result.get("metrics", {})

                if not m:
                    st.warning("⚠️ No trades generated with current settings. "
                               "Try adjusting parameters or selecting a longer period.")
                else:
                    # ── METRIC CARDS ──────────────────────────────────────
                    c1, c2, c3, c4, c5, c6 = st.columns(6)
                    metric_data = [
                        (c1, "Total Trades",   m["Total Trades"],    ""),
                        (c2, "Win Rate",        f"{m['Win Rate %']}%", ""),
                        (c3, "Profit Factor",   m["Profit Factor"],   ""),
                        (c4, "Max Drawdown",    f"{m['Max Drawdown %']}%", ""),
                        (c5, "CAGR",            f"{m['CAGR %']}%",    ""),
                        (c6, "Sharpe",          m["Sharpe Ratio"],    ""),
                    ]
                    for col_, label_, val_, _ in metric_data:
                        with col_:
                            col_.metric(label_, val_)

                    c7, c8, c9 = st.columns(3)
                    with c7: st.metric("Net P&L", f"₹{m['Net P&L']:,.0f}")
                    with c8: st.metric("Avg Win %", f"{m['Avg Win %']}%")
                    with c9: st.metric("Avg Loss %", f"{m['Avg Loss %']}%")

                    st.markdown("---")

                    # ── CHARTS ────────────────────────────────────────────
                    price_fig = build_price_chart(
                        df_ind.tail(200), df_sig.tail(200), None,
                        f"{asset_name_sel} — {strategy_name} Signals"
                    )
                    st.plotly_chart(price_fig, use_container_width=True)

                    equity_fig = build_equity_chart(
                        result["equity"], result["drawdown"], result["trades"]
                    )
                    st.plotly_chart(equity_fig, use_container_width=True)

                    # ── TRADE LOG ─────────────────────────────────────────
                    st.markdown('<div class="section-header">📋 Trade Log</div>',
                                unsafe_allow_html=True)

                    trades_show = result["trades"].copy()
                    trades_show["P&L %"] = trades_show["P&L %"].apply(
                        lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%")

                    st.dataframe(
                        trades_show[[
                            "Entry Date", "Exit Date", "Direction",
                            "Entry", "Exit", "SL", "Target",
                            "P&L %", "P&L ₹", "Exit Reason"
                        ]].tail(50),
                        use_container_width=True, height=300
                    )

                    # ── WIN/LOSS PIE ──────────────────────────────────────
                    trade_df_r = result["trades"]
                    exit_counts = trade_df_r["Exit Reason"].value_counts().reset_index()
                    pie_fig = px.pie(
                        exit_counts, names="Exit Reason", values="count",
                        color_discrete_sequence=[ACCENT_GREEN, ACCENT_RED,
                                                  ACCENT_GOLD, ACCENT_BLUE, ACCENT_PURPLE],
                        title="Exit Reason Distribution"
                    )
                    pie_fig.update_layout(
                        paper_bgcolor=DARK_BG, font_color=TEXT_PRIMARY, height=350
                    )

                    pnl_hist = px.histogram(
                        trade_df_r, x="P&L %",
                        nbins=30, title="P&L Distribution",
                        color_discrete_sequence=[ACCENT_BLUE]
                    )
                    pnl_hist.update_layout(
                        paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
                        font_color=TEXT_PRIMARY, height=350
                    )

                    cc1, cc2 = st.columns(2)
                    with cc1: st.plotly_chart(pie_fig,  use_container_width=True)
                    with cc2: st.plotly_chart(pnl_hist, use_container_width=True)

        else:
            st.info("👈 Configure settings in the sidebar and click **Run Analysis** to start backtesting.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE TRADING
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">⚡ Live Market Scanner & Signals</div>',
                unsafe_allow_html=True)

    note_col, refresh_col = st.columns([4, 1])
    with note_col:
        st.warning("📡 Data is fetched from Yahoo Finance. Indian market data may be delayed by 15–20 min during market hours.")
    with refresh_col:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()

    with st.spinner("Fetching live data..."):
        live_df_raw = fetch_live(ticker)

    if live_df_raw.empty:
        st.error("Could not fetch live data. Check ticker or internet connection.")
    else:
        live_df   = compute_indicators(live_df_raw)
        live_sig  = get_strategy_df(live_df, strategy_key, atr_sl, rr_ratio,
                                     trail_sl, trail_pct_val)
        analysis  = generate_analysis(live_df, ticker, asset_name_sel)

        if not analysis:
            st.warning("Insufficient data for live analysis.")
        else:
            # ── LIVE HEADER METRICS ───────────────────────────────────────
            price_chg = live_df["Close"].iloc[-1] - live_df["Close"].iloc[-2]
            price_pct = price_chg / live_df["Close"].iloc[-2] * 100

            lc1, lc2, lc3, lc4, lc5 = st.columns(5)
            with lc1: st.metric("📍 LTP", f"₹{analysis['price']:,.2f}",
                                 f"{price_chg:+.2f} ({price_pct:+.2f}%)")
            with lc2: st.metric("💪 RSI", analysis["rsi"])
            with lc3: st.metric("📊 Vol Ratio", f"{analysis['vol_ratio']}x")
            with lc4: st.metric("📐 ATR", analysis["atr"])
            with lc5: st.metric("📉 IV Proxy", f"{analysis['iv_proxy']}%")

            st.markdown("---")

            # ── SIGNAL BOX ────────────────────────────────────────────────
            sig_class = ("signal-bull" if "BUY" in analysis["bias"]
                         else ("signal-bear" if "SELL" in analysis["bias"]
                               else "signal-neutral"))
            st.markdown(f"""
            <div class="signal-box {sig_class}">
                <h2 style="margin:0">{analysis["strength"]} {analysis["bias"]}</h2>
                <div style="margin-top:8px;color:{TEXT_MUTED};font-size:13px">
                    Strategy: <b style="color:{ACCENT_GOLD}">{strategy_name}</b> &nbsp;|&nbsp;
                    Trend Score: <b>{analysis["trend_score"]}/10</b> &nbsp;|&nbsp;
                    Momentum Score: <b>{analysis["mom_score"]}/10</b> &nbsp;|&nbsp;
                    Combined: <b>{analysis["combined"]}/20</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # ── ENTRY / SL / TARGETS ──────────────────────────────────────
            st.markdown('<div class="section-header">🎯 Entry | SL | Targets</div>',
                        unsafe_allow_html=True)

            ec1, ec2, ec3, ec4, ec5, ec6 = st.columns(6)
            with ec1: st.metric("🟡 Entry",        f"₹{analysis['entry']:,.2f}")
            with ec2: st.metric("🔴 Fixed SL",     f"₹{analysis['sl_fixed']:,.2f}")
            with ec3: st.metric("🟠 ATR SL",        f"₹{analysis['sl_atr']:,.2f}")
            with ec4: st.metric("🟢 Target 1",      f"₹{analysis['tgt1']:,.2f}")
            with ec5: st.metric("🟢 Target 2",      f"₹{analysis['tgt2']:,.2f}")
            with ec6: st.metric("🟢 Target 3",      f"₹{analysis['tgt3']:,.2f}")

            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown(f"""
                <div class="strategy-card">
                <b style="color:{ACCENT_GOLD}">🔄 Trailing Stop Loss</b><br><br>
                Start Trail: <b>₹{analysis['trail_sl_start']:,.2f}</b><br>
                Trail %: <b>{analysis['trail_pct']}%</b><br>
                <small style="color:{TEXT_MUTED}">Trail SL ratchets up/down with every ₹{analysis['atr']:,.2f} (1 ATR) move in your favor.
                Once price moves {analysis['trail_pct']}% in your direction, SL locks in profit.
                If price reverses {analysis['trail_pct']}% from high → Exit.</small>
                </div>
                """, unsafe_allow_html=True)
            with tc2:
                st.markdown(f"""
                <div class="strategy-card">
                <b style="color:{ACCENT_GOLD}">🎯 Trailing Target</b><br><br>
                Initial Target: <b>₹{analysis['tgt1']:,.2f}</b><br>
                Trail Tgt %: <b>{trail_tgt_val}%</b><br>
                <small style="color:{TEXT_MUTED}">Target moves up with price. At Target 1 → book 40%.
                Trail the remaining using ATR. Exit rest at Target 2 or when momentum fades (RSI drops below 50 / MACD cross).</small>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # ── LIVE CHART ────────────────────────────────────────────────
            live_fig = build_price_chart(
                live_df.tail(100), live_sig.tail(100), analysis,
                f"⚡ {asset_name_sel} — Live Chart with Signals"
            )
            st.plotly_chart(live_fig, use_container_width=True)

            # ── LIVE SIGNAL TABLE ─────────────────────────────────────────
            recent_signals = live_sig[live_sig["Signal"] != 0].tail(10).copy()
            if not recent_signals.empty:
                st.markdown('<div class="section-header">📡 Recent Signals</div>',
                            unsafe_allow_html=True)
                display_cols = ["Close", "Signal", "RSI", "MACD",
                                "ATR", "Vol_Ratio", "EMA20", "EMA50"]
                display_cols = [c for c in display_cols if c in recent_signals.columns]
                st.dataframe(recent_signals[display_cols].round(2),
                             use_container_width=True)

            # ── OPTIONS PRICING ───────────────────────────────────────────
            st.markdown("---")
            st.markdown('<div class="section-header">⚙️ Options Pricing (Black-Scholes)</div>',
                        unsafe_allow_html=True)

            opt_t  = "CE" if "CE" in opt_type else "PE"
            T_days = days_exp / 365
            sigma  = iv_pct / 100

            atm_k   = round(spot_price / 100) * 100
            strikes = [atm_k - 200, atm_k - 100, atm_k, atm_k + 100, atm_k + 200]

            bs_rows = []
            for k in strikes:
                p, d, g, th, v = bs_price(spot_price, k, T_days, risk_free/100, sigma, opt_t)
                moneyness = "ATM" if k == atm_k else ("ITM" if (
                    (opt_t == "CE" and spot_price > k) or
                    (opt_t == "PE" and spot_price < k)) else "OTM")
                bs_rows.append({
                    "Strike": k,
                    "Type": opt_t,
                    "Moneyness": moneyness,
                    "Premium": p,
                    "Delta": d,
                    "Gamma": g,
                    "Theta/day": th,
                    "Vega": v,
                })

            bs_df = pd.DataFrame(bs_rows)
            st.dataframe(bs_df.style.highlight_max(subset=["Delta"], color="#1a3a2a")
                         .highlight_max(subset=["Premium"], color="#1a3a2a"),
                         use_container_width=True)

            best_opt = bs_df[bs_df["Moneyness"].isin(["ATM", "ITM"])]
            if not best_opt.empty:
                best = best_opt.iloc[0]
                st.markdown(f"""
                <div class="signal-box {'signal-bull' if 'BUY' in analysis['bias'] else 'signal-bear'}">
                <b>Recommended Option:</b>
                {opt_t} Strike <b>{best['Strike']}</b> | Premium: <b>₹{best['Premium']}</b>
                | Delta: <b>{best['Delta']}</b> | Theta: <b>{best['Theta/day']}/day</b><br>
                <small style="color:{TEXT_MUTED}">IV: {iv_pct}% | Days to Expiry: {days_exp}
                | IV Rank: {analysis['iv_rank']}</small>
                </div>
                """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYSE & RECOMMEND
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🔭 Deep Analysis & Recommendations</div>',
                unsafe_allow_html=True)

    with st.spinner("Generating full analysis..."):
        an_df_raw = fetch_data(ticker, period="1y", interval="1d")

    if an_df_raw.empty:
        st.error("Could not fetch data.")
    else:
        an_df  = compute_indicators(an_df_raw)
        an     = generate_analysis(an_df, ticker, asset_name_sel)

        if not an:
            st.warning("Insufficient data.")
        else:
            # ── TOP SIGNAL BANNER ─────────────────────────────────────────
            dir_color = ACCENT_GREEN if "BUY" in an["bias"] else ACCENT_RED
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{dir_color}18,{dir_color}05);
                        border:1px solid {dir_color}50;border-radius:16px;
                        padding:28px 32px;margin:0 0 24px;">
                <div style="font-size:36px;font-weight:700;color:{dir_color}">
                    {an['strength']} {an['bias']}
                </div>
                <div style="color:{TEXT_MUTED};font-size:14px;margin-top:8px">
                    {asset_name_sel} ({ticker}) &nbsp;·&nbsp;
                    Last Price: <b style="color:{TEXT_PRIMARY}">₹{an['price']:,.2f}</b> &nbsp;·&nbsp;
                    ATR: <b style="color:{TEXT_PRIMARY}">{an['atr']}</b> &nbsp;·&nbsp;
                    RSI: <b style="color:{TEXT_PRIMARY}">{an['rsi']}</b>
                </div>
                <div style="margin-top:16px;display:flex;gap:40px;flex-wrap:wrap">
                    <div><div style="color:{TEXT_MUTED};font-size:11px;text-transform:uppercase;letter-spacing:1px">Trend Score</div>
                         <div style="font-size:24px;font-weight:700;font-family:JetBrains Mono;color:{ACCENT_GOLD}">{an['trend_score']}<span style="font-size:14px;color:{TEXT_MUTED}">/10</span></div></div>
                    <div><div style="color:{TEXT_MUTED};font-size:11px;text-transform:uppercase;letter-spacing:1px">Momentum Score</div>
                         <div style="font-size:24px;font-weight:700;font-family:JetBrains Mono;color:{ACCENT_GOLD}">{an['mom_score']}<span style="font-size:14px;color:{TEXT_MUTED}">/10</span></div></div>
                    <div><div style="color:{TEXT_MUTED};font-size:11px;text-transform:uppercase;letter-spacing:1px">Combined Score</div>
                         <div style="font-size:24px;font-weight:700;font-family:JetBrains Mono;color:{ACCENT_GOLD}">{an['combined']}<span style="font-size:14px;color:{TEXT_MUTED}">/20</span></div></div>
                    <div><div style="color:{TEXT_MUTED};font-size:11px;text-transform:uppercase;letter-spacing:1px">IV Rank</div>
                         <div style="font-size:20px;font-weight:700;color:{ACCENT_GOLD}">{an['iv_rank']}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── TRADE PLAN ────────────────────────────────────────────────
            st.markdown('<div class="section-header">📋 Complete Trade Plan</div>',
                        unsafe_allow_html=True)

            r1c, r2c = st.columns(2)
            with r1c:
                risk_amt    = capital * risk_pct / 100
                sl_dist     = abs(an["entry"] - an["sl_fixed"])
                qty_est     = int(risk_amt / sl_dist) if sl_dist > 0 else 0
                reward1     = abs(an["tgt1"] - an["entry"])
                reward2     = abs(an["tgt2"] - an["entry"])

                st.markdown(f"""
                <div class="strategy-card">
                    <b style="color:{ACCENT_GOLD};font-size:16px">🎯 Trade Setup</b><br><br>
                    <table style="width:100%;border-collapse:collapse;font-size:13px">
                        <tr><td style="color:{TEXT_MUTED};padding:5px 0">Direction</td>
                            <td style="color:{'#10B981' if an['is_long'] else '#EF4444'};font-weight:700">
                            {"▲ LONG / BUY CE" if an['is_long'] else "▼ SHORT / BUY PE"}</td></tr>
                        <tr><td style="color:{TEXT_MUTED}">Entry</td>
                            <td style="color:{ACCENT_GOLD};font-weight:700">₹{an['entry']:,.2f}</td></tr>
                        <tr><td style="color:{TEXT_MUTED}">Stop Loss (Fixed)</td>
                            <td style="color:{ACCENT_RED};font-weight:700">₹{an['sl_fixed']:,.2f}
                            <small>({sl_dist:.1f} pts)</small></td></tr>
                        <tr><td style="color:{TEXT_MUTED}">Stop Loss (ATR)</td>
                            <td style="color:{ACCENT_RED}">₹{an['sl_atr']:,.2f}</td></tr>
                        <tr><td style="color:{TEXT_MUTED}">Target 1 (50%)</td>
                            <td style="color:{ACCENT_GREEN};font-weight:700">₹{an['tgt1']:,.2f}
                            <small>(+{reward1:.1f} pts)</small></td></tr>
                        <tr><td style="color:{TEXT_MUTED}">Target 2 (30%)</td>
                            <td style="color:{ACCENT_GREEN}">₹{an['tgt2']:,.2f}</td></tr>
                        <tr><td style="color:{TEXT_MUTED}">Target 3 (20%)</td>
                            <td style="color:{ACCENT_GREEN}">₹{an['tgt3']:,.2f}</td></tr>
                        <tr><td style="color:{TEXT_MUTED}">Trailing SL %</td>
                            <td style="color:{ACCENT_BLUE}">Trail {trail_pct_val}% from peak</td></tr>
                        <tr><td style="color:{TEXT_MUTED}">Trailing Tgt %</td>
                            <td style="color:{ACCENT_BLUE}">Trail {trail_tgt_val}% from tgt</td></tr>
                        <tr><td style="color:{TEXT_MUTED}">Risk Amount</td>
                            <td>₹{risk_amt:,.0f} ({risk_pct}%)</td></tr>
                        <tr><td style="color:{TEXT_MUTED}">Est. Qty (Shares)</td>
                            <td>{qty_est}</td></tr>
                        <tr><td style="color:{TEXT_MUTED}">R:R Ratio</td>
                            <td style="color:{ACCENT_GOLD}">1:{reward1/sl_dist:.1f}</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

            with r2c:
                st.markdown(f"""
                <div class="strategy-card">
                    <b style="color:{ACCENT_GOLD};font-size:16px">📐 Key Levels</b><br><br>
                    <table style="width:100%;border-collapse:collapse;font-size:13px">
                """, unsafe_allow_html=True)

                levels_display = [
                    ("EMA200",   an["levels"]["EMA200"],   "#8B5CF6"),
                    ("EMA50",    an["levels"]["EMA50"],    "#F59E0B"),
                    ("EMA20",    an["levels"]["EMA20"],    "#3B82F6"),
                    ("VWAP",     an["levels"]["VWAP"],     "#F97316"),
                    ("Pivot",    an["levels"]["Pivot"],    "#E5E7EB"),
                    ("R1",       an["levels"]["R1"],       "#10B981"),
                    ("R2",       an["levels"]["R2"],       "#059669"),
                    ("S1",       an["levels"]["S1"],       "#EF4444"),
                    ("S2",       an["levels"]["S2"],       "#DC2626"),
                    ("BB Upper", an["levels"]["BB Upper"], "#6366F1"),
                    ("BB Lower", an["levels"]["BB Lower"], "#6366F1"),
                ]
                level_html = '<div class="strategy-card">'
                level_html += f'<b style="color:{ACCENT_GOLD};font-size:16px">📐 Key Levels</b><br><br>'
                level_html += '<table style="width:100%;border-collapse:collapse;font-size:13px">'
                for name_, val_, clr_ in levels_display:
                    marker = " ← Price" if abs(val_ - an["price"]) < an["atr"] else ""
                    level_html += (
                        f'<tr><td style="color:{TEXT_MUTED};padding:4px 0">{name_}</td>'
                        f'<td style="color:{clr_};font-weight:600">₹{val_:,.2f}'
                        f'<small style="color:{ACCENT_GOLD}">{marker}</small></td></tr>'
                    )
                level_html += "</table></div>"
                st.markdown(level_html, unsafe_allow_html=True)

            # ── TREND ANALYSIS ────────────────────────────────────────────
            st.markdown("---")
            ta1, ta2 = st.columns(2)
            with ta1:
                st.markdown('<div class="section-header">📈 Trend Analysis</div>',
                            unsafe_allow_html=True)
                for note in an["trend_notes"]:
                    bg = "rgba(16,185,129,0.08)" if "✅" in note else "rgba(239,68,68,0.08)"
                    st.markdown(
                        f'<div style="background:{bg};border-radius:6px;'
                        f'padding:8px 12px;margin:4px 0;font-size:13px">{note}</div>',
                        unsafe_allow_html=True)

            with ta2:
                st.markdown('<div class="section-header">⚡ Momentum Analysis</div>',
                            unsafe_allow_html=True)
                for note in an["mom_notes"]:
                    bg = "rgba(16,185,129,0.08)" if "✅" in note else (
                         "rgba(245,158,11,0.08)" if "⚠️" in note else "rgba(239,68,68,0.08)")
                    st.markdown(
                        f'<div style="background:{bg};border-radius:6px;'
                        f'padding:8px 12px;margin:4px 0;font-size:13px">{note}</div>',
                        unsafe_allow_html=True)

            # ── OPTIONS RECOMMENDATION ────────────────────────────────────
            st.markdown("---")
            st.markdown('<div class="section-header">📊 Options Recommendation</div>',
                        unsafe_allow_html=True)

            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                st.markdown(f"""
                <div class="strategy-card">
                    <b style="color:{ACCENT_GOLD}">Which Option to Buy?</b><br><br>
                    <div style="font-size:14px">{an['opt_rec']}</div><br>
                    <div style="font-size:12px;color:{TEXT_MUTED}">{an['opt_exp']}</div>
                </div>
                """, unsafe_allow_html=True)
            with oc2:
                st.markdown(f"""
                <div class="strategy-card">
                    <b style="color:{ACCENT_GOLD}">IV Conditions</b><br><br>
                    IV Proxy: <b>{an['iv_proxy']}%</b><br>
                    IV Rank: <b>{an['iv_rank']}</b><br><br>
                    <div style="font-size:12px;color:{TEXT_MUTED}">
                    {"✅ Low IV — Good time to BUY options. Premium is cheap." 
                     if an['iv_proxy'] < 5 else 
                     "⚠️ Medium IV — Fair premium. Buy ATM, avoid OTM." 
                     if an['iv_proxy'] < 10 else
                     "❌ High IV — Options expensive. Consider spreads or avoid buying."}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with oc3:
                st.markdown(f"""
                <div class="strategy-card">
                    <b style="color:{ACCENT_GOLD}">Capital Allocation</b><br><br>
                    Total Capital: <b>₹{capital:,}</b><br>
                    Max per Trade: <b>₹{int(capital * risk_pct / 100):,}</b> ({risk_pct}%)<br>
                    Max 2 trades/day<br>
                    Partial Exit at Tgt1: <b>50%</b><br>
                    Trail rest to Tgt2/3<br>
                    <div style="font-size:12px;color:{TEXT_MUTED};margin-top:8px">
                    Never average a losing option position.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── TRAILING SL LOGIC EXPLAINED ───────────────────────────────
            st.markdown("---")
            st.markdown('<div class="section-header">🔄 Trailing SL & Target — Complete Logic</div>',
                        unsafe_allow_html=True)

            direction_word = "rises" if an["is_long"] else "falls"
            trail_dir      = "below" if an["is_long"] else "above"
            trail_dir2     = "above" if an["is_long"] else "below"

            st.markdown(f"""
            <div class="strategy-card">
            <b style="color:{ACCENT_GOLD};font-size:15px">How Trailing Stop Loss Works for This Trade</b><br><br>

            <b>Setup:</b> Entry at ₹{an['entry']:,.2f} | Initial SL at ₹{an['sl_fixed']:,.2f}
            | Trail: {trail_pct_val}% from peak<br><br>

            <b>Step-by-step:</b><br>
            <ol style="font-size:13px;color:{TEXT_MUTED};line-height:2">
                <li>Enter at ₹{an['entry']:,.2f} → Initial SL = ₹{an['sl_fixed']:,.2f}</li>
                <li>Price {direction_word} to ₹{an['tgt1']:,.2f} (Target 1) → <b>Book 50% position</b></li>
                <li>Trail SL moves to ₹{round(an['tgt1'] * (1 - trail_pct_val/100 if an['is_long'] else 1 + trail_pct_val/100), 2):,.2f} 
                   ({trail_pct_val}% {trail_dir} the new high)</li>
                <li>Price continues to ₹{an['tgt2']:,.2f} (Target 2) → <b>Book 30% more</b></li>
                <li>Trail SL again locks in gain at {trail_pct_val}% {trail_dir} ₹{an['tgt2']:,.2f}</li>
                <li>Hold remaining 20% for Target 3 (₹{an['tgt3']:,.2f}) or until SL hits</li>
            </ol>

            <b>Trailing Target Logic:</b><br>
            <div style="font-size:13px;color:{TEXT_MUTED}">
            Once price exceeds Target 1, your target "trails" at {trail_tgt_val}% 
            {trail_dir2} price. This ensures you don't exit too early if momentum is strong.
            If price {direction_words2 if False else direction_word}s past ₹{an['tgt2']:,.2f}, 
            the trailing target moves automatically. You only exit when price 
            reverses {trail_tgt_val}% from the highest point reached.
            </div><br>

            <b style="color:{ACCENT_RED}">Hard Stop:</b> 
            If price closes {trail_dir} ₹{an['sl_fixed']:,.2f} on the 15-min candle → 
            Exit <b>immediately</b>, no hesitation. Capital preservation > everything.
            </div>
            """, unsafe_allow_html=True)

            # ── HISTORICAL CHART ──────────────────────────────────────────
            st.markdown("---")
            an_sig  = get_strategy_df(an_df, strategy_key, atr_sl, rr_ratio,
                                       trail_sl, trail_pct_val)
            an_fig  = build_price_chart(
                an_df.tail(250), an_sig.tail(250), an,
                f"🔭 {asset_name_sel} — Full Analysis Chart (1 Year)"
            )
            st.plotly_chart(an_fig, use_container_width=True)

            # ── GAUGE CHARTS ──────────────────────────────────────────────
            st.markdown("---")
            st.markdown('<div class="section-header">📡 Signal Gauges</div>',
                        unsafe_allow_html=True)

            def make_gauge(val, max_val, title, color):
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=val,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": title, "font": {"color": TEXT_PRIMARY, "size": 13}},
                    gauge={
                        "axis": {"range": [0, max_val], "tickcolor": TEXT_MUTED},
                        "bar": {"color": color},
                        "bgcolor": CARD_BG,
                        "bordercolor": BORDER,
                        "steps": [
                            {"range": [0, max_val * 0.33], "color": "rgba(239,68,68,0.2)"},
                            {"range": [max_val * 0.33, max_val * 0.66], "color": "rgba(245,158,11,0.2)"},
                            {"range": [max_val * 0.66, max_val], "color": "rgba(16,185,129,0.2)"},
                        ],
                    },
                    number={"font": {"color": color, "family": "JetBrains Mono"}}
                ))
                fig.update_layout(
                    paper_bgcolor=CARD_BG, height=200,
                    margin=dict(l=20, r=20, t=40, b=10),
                    font_color=TEXT_MUTED
                )
                return fig

            gc1, gc2, gc3, gc4 = st.columns(4)
            with gc1: st.plotly_chart(make_gauge(an["trend_score"], 10, "Trend Score", ACCENT_BLUE), use_container_width=True)
            with gc2: st.plotly_chart(make_gauge(an["mom_score"],   10, "Momentum Score", ACCENT_PURPLE), use_container_width=True)
            with gc3: st.plotly_chart(make_gauge(an["rsi"],         100, "RSI", ACCENT_GOLD), use_container_width=True)
            with gc4: st.plotly_chart(make_gauge(an["vol_ratio"] * 5, 20, "Volume (×avg)", ACCENT_GREEN), use_container_width=True)

            # ── SCANNER: TOP SIGNALS ACROSS ASSETS ───────────────────────
            st.markdown("---")
            st.markdown('<div class="section-header">🔍 Multi-Asset Quick Scan</div>',
                        unsafe_allow_html=True)

            scan_assets = {
                "Nifty 50": "^NSEI",
                "BankNifty": "^NSEBANK",
                "Sensex": "^BSESN",
                "BTC": "BTC-USD",
                "ETH": "ETH-USD",
                "Gold": "GC=F",
                "USD/INR": "USDINR=X",
            }

            if st.button("🔍 Run Multi-Asset Scan"):
                scan_results = []
                scan_prog = st.progress(0)
                for idx_, (name_, tick_) in enumerate(scan_assets.items()):
                    scan_prog.progress((idx_ + 1) / len(scan_assets),
                                       text=f"Scanning {name_}...")
                    sdf = fetch_data(tick_, period="3mo", interval="1d")
                    if sdf.empty:
                        continue
                    sdf_i = compute_indicators(sdf)
                    san   = generate_analysis(sdf_i, tick_, name_)
                    if san:
                        scan_results.append({
                            "Asset":    name_,
                            "Ticker":   tick_,
                            "Price":    san["price"],
                            "Signal":   san["bias"],
                            "Score":    san["combined"],
                            "RSI":      san["rsi"],
                            "Vol Ratio":san["vol_ratio"],
                            "ATR":      san["atr"],
                        })

                scan_prog.empty()
                if scan_results:
                    sdf_out = pd.DataFrame(scan_results).sort_values("Score", ascending=False)
                    st.dataframe(sdf_out, use_container_width=True)
                else:
                    st.warning("No scan results available.")

            # ── RISK DISCLAIMER ───────────────────────────────────────────
            st.markdown("---")
            st.markdown(f"""
            <div style="background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.3);
                        border-radius:10px;padding:16px 20px;font-size:12px;color:{TEXT_MUTED}">
            ⚠️ <b style="color:{ACCENT_GOLD}">Disclaimer:</b>
            This platform is for educational and research purposes only. All signals are
            algorithmic and do not constitute financial advice. Options trading involves
            substantial risk of loss. Past performance does not guarantee future results.
            Always do your own research. Never risk more than you can afford to lose.
            SEBI Registration is required for providing investment advice in India.
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;padding:16px;color:{TEXT_MUTED};font-size:12px">
    ⚡ <b style="color:{ACCENT_GOLD}">AlphaEdge</b> Professional Trading Platform &nbsp;|&nbsp;
    Built with Streamlit + yFinance + Plotly &nbsp;|&nbsp;
    Strategies: TSM · ORB · VWAP+RSI · Swing · Combined<br>
    <span style="color:{TEXT_MUTED}">
    Indian Indices · Stocks · BTC · ETH · Forex · Gold · Silver · Custom Tickers
    </span>
</div>
""", unsafe_allow_html=True)
