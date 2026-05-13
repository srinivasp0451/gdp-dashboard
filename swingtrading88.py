#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║        ⚡  PROFESSIONAL ALGORITHMIC TRADING PLATFORM  ⚡             ║
║   Backtesting · Live Trading · Trade History                        ║
║   Strategies : ORB (5-min) · Simple Buy · Simple Sell              ║
║   Broker     : Dhan API  |  Data : yfinance                        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════
# § 1 · IMPORTS
# ═══════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time, json, uuid, math, warnings, os
import requests
from datetime import datetime, timedelta, date, time as dtime
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

try:
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
except Exception:
    IST = None

try:
    from ta.momentum import RSIIndicator
    from ta.trend import MACD as TAMacd, EMAIndicator, SMAIndicator, ADXIndicator
    from ta.volatility import AverageTrueRange, BollingerBands
    from ta.volume import OnBalanceVolumeIndicator
    TA_OK = True
except Exception:
    TA_OK = False

# ═══════════════════════════════════════════════════════════════
# § 2 · PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="⚡ Pro Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# § 3 · CSS  — dark terminal / Bloomberg aesthetic
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
  --bg:        #0a0e17;
  --bg2:       #111827;
  --bg3:       #1a2235;
  --border:    #1e2d45;
  --accent:    #00d4aa;
  --accent2:   #f59e0b;
  --accent3:   #ef4444;
  --accent4:   #3b82f6;
  --text:      #e2e8f0;
  --text2:     #94a3b8;
  --green:     #10b981;
  --red:       #ef4444;
  --gold:      #f59e0b;
}

html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border);
}

.stTabs [data-baseweb="tab-list"] {
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  gap: 4px;
}
.stTabs [data-baseweb="tab"] {
  color: var(--text2) !important;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.05em;
  padding: 10px 24px;
  border-radius: 4px 4px 0 0;
  border: 1px solid transparent;
}
.stTabs [aria-selected="true"] {
  background: var(--bg3) !important;
  color: var(--accent) !important;
  border-color: var(--border) !important;
  border-bottom-color: var(--bg3) !important;
}

div[data-testid="metric-container"] {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: var(--text2) !important; font-size:11px; font-family:'JetBrains Mono',monospace; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family:'JetBrains Mono',monospace; }

.stButton > button {
  background: var(--bg3);
  color: var(--accent);
  border: 1px solid var(--accent);
  border-radius: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.08em;
  transition: all 0.15s;
}
.stButton > button:hover {
  background: var(--accent);
  color: var(--bg);
}

.stSelectbox > div, .stTextInput > div > div, .stNumberInput > div > div {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 6px !important;
}

.metric-card {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 20px;
  margin-bottom: 8px;
}
.metric-card .label { font-size:10px; color:var(--text2); font-family:'JetBrains Mono',monospace; letter-spacing:.1em; text-transform:uppercase; }
.metric-card .val   { font-size:24px; font-family:'JetBrains Mono',monospace; font-weight:700; margin-top:4px; }
.metric-card .val.green  { color: var(--green); }
.metric-card .val.red    { color: var(--red); }
.metric-card .val.gold   { color: var(--gold); }
.metric-card .val.blue   { color: var(--accent4); }
.metric-card .val.accent { color: var(--accent); }

.signal-badge {
  display:inline-block; padding:4px 12px; border-radius:20px;
  font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:700; letter-spacing:.1em;
}
.signal-buy   { background:#10b98122; color:#10b981; border:1px solid #10b981; }
.signal-sell  { background:#ef444422; color:#ef4444; border:1px solid #ef4444; }
.signal-hold  { background:#f59e0b22; color:#f59e0b; border:1px solid #f59e0b; }
.signal-none  { background:#1e2d4588; color:#94a3b8; border:1px solid #1e2d45; }

.trade-row {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
}
.section-header {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: .15em;
  text-transform: uppercase;
  color: var(--text2);
  border-bottom: 1px solid var(--border);
  padding-bottom: 6px;
  margin-bottom: 14px;
}
.live-dot {
  display:inline-block; width:8px; height:8px; border-radius:50%;
  background:var(--green); margin-right:6px;
  animation: pulse 1.4s infinite;
}
@keyframes pulse {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:.4; transform:scale(1.3); }
}
.info-box {
  background: #0d2137;
  border-left: 3px solid var(--accent4);
  border-radius: 0 6px 6px 0;
  padding: 10px 14px;
  font-size: 12px;
  color: var(--text2);
  margin-bottom: 10px;
}
.warn-box {
  background: #1f1107;
  border-left: 3px solid var(--gold);
  border-radius: 0 6px 6px 0;
  padding: 10px 14px;
  font-size: 12px;
  color: #d97706;
  margin-bottom: 10px;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility:hidden; }
.stDeployButton { display:none; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# § 4 · CONSTANTS & MAPPINGS
# ═══════════════════════════════════════════════════════════════
TICKER_MAP = {
    "NIFTY 50"      : "^NSEI",
    "BANK NIFTY"    : "^NSEBANK",
    "SENSEX"        : "^BSESN",
    "NIFTY IT"      : "^CNXIT",
    "NIFTY PHARMA"  : "^CNXPHARMA",
    "NIFTY AUTO"    : "^CNXAUTO",
    "NIFTY FMCG"    : "^CNXFMCG",
    "BTC/USD"       : "BTC-USD",
    "ETH/USD"       : "ETH-USD",
    "BNB/USD"       : "BNB-USD",
    "SOL/USD"       : "SOL-USD",
    "USD/INR"       : "USDINR=X",
    "EUR/INR"       : "EURINR=X",
    "EUR/USD"       : "EURUSD=X",
    "GBP/USD"       : "GBPUSD=X",
    "USD/JPY"       : "USDJPY=X",
    "GOLD (MCX)"    : "GC=F",
    "SILVER (MCX)"  : "SI=F",
    "CRUDE OIL"     : "CL=F",
    "NATURAL GAS"   : "NG=F",
    "COPPER"        : "HG=F",
    "RELIANCE"      : "RELIANCE.NS",
    "TCS"           : "TCS.NS",
    "HDFC BANK"     : "HDFCBANK.NS",
    "INFOSYS"       : "INFY.NS",
    "ICICI BANK"    : "ICICIBANK.NS",
    "SBI"           : "SBIN.NS",
    "WIPRO"         : "WIPRO.NS",
    "BAJAJ FINANCE" : "BAJFINANCE.NS",
    "CUSTOM"        : "CUSTOM",
}

TIMEFRAME_MAP = {
    "1m"  : "1m",
    "5m"  : "5m",
    "15m" : "15m",
    "30m" : "30m",
    "1h"  : "1h",
    "4h"  : "4h",
    "1d"  : "1d",
    "1wk" : "1wk",
}

PERIOD_MAP = {
    "1 Day"    : "1d",
    "5 Days"   : "5d",
    "7 Days"   : "7d",
    "1 Month"  : "1mo",
    "3 Months" : "3mo",
    "6 Months" : "6mo",
    "1 Year"   : "1y",
    "2 Years"  : "2y",
    "5 Years"  : "5y",
    "10 Years" : "10y",
    "20 Years" : "max",
}

# Valid yfinance period/interval combinations
VALID_COMBOS = {
    "1m" : ["1d","2d","5d","7d"],
    "5m" : ["1d","5d","7d","1mo"],
    "15m": ["1d","5d","7d","1mo","3mo"],
    "30m": ["1d","5d","7d","1mo","3mo","6mo"],
    "1h" : ["1d","5d","7d","1mo","3mo","6mo","1y","2y"],
    "4h" : ["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y"],
    "1d" : ["1mo","3mo","6mo","1y","2y","5y","10y","max"],
    "1wk": ["3mo","6mo","1y","2y","5y","10y","max"],
}

NSE_SECTORS = {
    "🏦 Banking":    ["HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS","SBIN.NS","BANDHANBNK.NS","FEDERALBNK.NS","IDFCFIRSTB.NS"],
    "💻 IT/Tech":    ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS","MPHASIS.NS","PERSISTENT.NS","COFORGE.NS"],
    "💊 Pharma":     ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","TORNTPHARM.NS","AUROPHARMA.NS","ALKEM.NS"],
    "🚗 Auto":       ["MARUTI.NS","TATAMOTORS.NS","M&M.NS","BAJAJ-AUTO.NS","EICHERMOT.NS","HEROMOTOCO.NS","APOLLOTYRE.NS"],
    "⚡ Energy":     ["RELIANCE.NS","ONGC.NS","NTPC.NS","POWERGRID.NS","ADANIPOWER.NS","TATAPOWER.NS","ADANIGREEN.NS"],
    "🏗️ Infra":      ["LT.NS","ADANIPORTS.NS","SIEMENS.NS","ABB.NS","BHEL.NS","IRFC.NS"],
    "🛒 FMCG":       ["HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS","MARICO.NS","GODREJCP.NS"],
    "💰 Finance":    ["BAJFINANCE.NS","BAJAJFINSV.NS","HDFC.NS","CHOLAFIN.NS","MUTHOOTFIN.NS","RECLTD.NS","PFC.NS"],
}

STRATEGIES    = ["ORB (Opening Range Breakout)", "Simple Buy", "Simple Sell"]
SL_TYPES      = ["Fixed Points","ATR Based","Risk-Reward Based","Trailing (Points)","Trailing (Candle Low/High)","Trailing (Swing Low/High)","Signal Reversal","Volatility (BB) Based"]
TARGET_TYPES  = ["Fixed Points","ATR Based","Risk-Reward Based","Trailing Target (Display Only)","Volatility (BB) Based"]
INDICATORS    = ["RSI","MACD","Volume","ADX","ATR Filter","Order Block","Liquidity Hunt","Volatility Filter","EMA","SMA","Volume Profile","Fibonacci","EMA/SMA Crossover"]

# ═══════════════════════════════════════════════════════════════
# § 5 · SESSION STATE BOOTSTRAP
# ═══════════════════════════════════════════════════════════════
def init_state():
    defs = dict(
        trade_history    = [],
        live_position    = None,
        live_active      = False,
        live_ticker      = "",
        live_ltp         = 0.0,
        live_signals     = [],
        last_fetch_time  = 0.0,
        bt_results       = None,
        bt_trades        = [],
        bt_chart_data    = None,
        dhan_client_id   = "",
        dhan_token       = "",
        dhan_security_id = "",
        dhan_exchange    = "NSE_EQ",
        dhan_product     = "INTRADAY",
        live_log         = deque(maxlen=50),
        partial_closed   = False,
        half_closed      = False,
        trailing_high    = 0.0,
    )
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ═══════════════════════════════════════════════════════════════
# § 6 · DATA FETCHER  (rate-limited yfinance wrapper)
# ═══════════════════════════════════════════════════════════════
class DataFetcher:
    MIN_DELAY = 1.5   # seconds between requests

    @staticmethod
    def _wait():
        elapsed = time.time() - st.session_state.last_fetch_time
        if elapsed < DataFetcher.MIN_DELAY:
            time.sleep(DataFetcher.MIN_DELAY - elapsed)
        st.session_state.last_fetch_time = time.time()

    @staticmethod
    def resolve_ticker(name: str, custom: str = "") -> str:
        if name == "CUSTOM":
            return custom.upper().strip()
        return TICKER_MAP.get(name, name)

    @staticmethod
    def fetch(ticker: str, interval: str, period: str,
              retries: int = 3) -> Optional[pd.DataFrame]:
        for attempt in range(retries):
            try:
                DataFetcher._wait()
                df = yf.download(ticker, interval=interval, period=period,
                                 auto_adjust=True, progress=False, timeout=15)
                if df is None or df.empty:
                    return None
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[["Open","High","Low","Close","Volume"]].copy()
                df.dropna(subset=["Open","High","Low","Close"], inplace=True)
                # Localise index
                if df.index.tz is None and IST:
                    try:
                        df.index = df.index.tz_localize("UTC").tz_convert(IST)
                    except Exception:
                        pass
                elif df.index.tz is not None and IST:
                    try:
                        df.index = df.index.tz_convert(IST)
                    except Exception:
                        pass
                return df
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2.0)
                else:
                    st.error(f"⚠️ Data fetch failed: {e}")
        return None

    @staticmethod
    def get_ltp(ticker: str) -> float:
        """Single-ticker LTP via fast 1-min bar."""
        try:
            DataFetcher._wait()
            t = yf.Ticker(ticker)
            info = t.fast_info
            price = getattr(info, "last_price", None)
            if price and price > 0:
                return float(price)
            # fallback
            df = yf.download(ticker, interval="1m", period="1d",
                             auto_adjust=True, progress=False, timeout=10)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return float(df["Close"].iloc[-1])
        except Exception:
            pass
        return 0.0

    @staticmethod
    def momentum_pct(ticker: str) -> float:
        """% change from previous close to current price."""
        try:
            DataFetcher._wait()
            df = yf.download(ticker, interval="1m", period="2d",
                             auto_adjust=True, progress=False, timeout=10)
            if df is None or df.empty:
                return 0.0
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            closes = df["Close"].dropna()
            if len(closes) < 2:
                return 0.0
            prev_close = closes.iloc[-2] if len(closes) >= 2 else closes.iloc[0]
            ltp        = closes.iloc[-1]
            return float((ltp - prev_close) / prev_close * 100)
        except Exception:
            return 0.0

# ═══════════════════════════════════════════════════════════════
# § 7 · INDICATOR ENGINE
# ═══════════════════════════════════════════════════════════════
class Indicators:

    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        vol   = df.get("Volume", pd.Series(0, index=df.index))

        n = len(df)

        # ── ATR (always computed for SL management) ──
        df["ATR"] = Indicators._atr(high, low, close)

        if n < 3:
            return df

        # ── RSI ──
        if n >= 15 and TA_OK:
            df["RSI"] = RSIIndicator(close, window=14).rsi()
        else:
            df["RSI"] = np.nan

        # ── MACD ──
        if n >= 27 and TA_OK:
            macd_obj = TAMacd(close)
            df["MACD"]        = macd_obj.macd()
            df["MACD_Signal"] = macd_obj.macd_signal()
            df["MACD_Hist"]   = macd_obj.macd_diff()
        else:
            df["MACD"] = df["MACD_Signal"] = df["MACD_Hist"] = np.nan

        # ── ADX ──
        if n >= 14 and TA_OK:
            adx = ADXIndicator(high, low, close, window=14)
            df["ADX"] = adx.adx()
        else:
            df["ADX"] = np.nan

        # ── EMA ──
        for w in [9, 20, 50, 200]:
            if n >= w and TA_OK:
                df[f"EMA{w}"] = EMAIndicator(close, window=w).ema_indicator()
            else:
                df[f"EMA{w}"] = np.nan

        # ── SMA ──
        for w in [20, 50, 200]:
            if n >= w and TA_OK:
                df[f"SMA{w}"] = SMAIndicator(close, window=w).sma_indicator()
            else:
                df[f"SMA{w}"] = np.nan

        # ── Bollinger Bands ──
        if n >= 21 and TA_OK:
            bb = BollingerBands(close, window=20, window_dev=2)
            df["BB_Upper"] = bb.bollinger_hband()
            df["BB_Lower"] = bb.bollinger_lband()
            df["BB_Mid"]   = bb.bollinger_mavg()
            df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
        else:
            df["BB_Upper"] = df["BB_Lower"] = df["BB_Mid"] = df["BB_Width"] = np.nan

        # ── Volume indicators ──
        has_vol = vol is not None and (vol > 0).any()
        if has_vol:
            df["VOL_MA"] = vol.rolling(20).mean()
            df["VOL_RATIO"] = vol / df["VOL_MA"].replace(0, np.nan)
            if TA_OK and n >= 2:
                df["OBV"] = OnBalanceVolumeIndicator(close, vol).on_balance_volume()
        else:
            df["VOL_MA"] = df["VOL_RATIO"] = df["OBV"] = np.nan

        # ── VWAP (Volume Profile proxy) ──
        if has_vol and vol.sum() > 0:
            typical_price = (high + low + close) / 3
            cum_tp_vol    = (typical_price * vol).cumsum()
            cum_vol       = vol.cumsum()
            df["VWAP"]    = cum_tp_vol / cum_vol.replace(0, np.nan)
        else:
            df["VWAP"] = (high + low + close) / 3  # unweighted

        # ── Fibonacci levels (on rolling 100-bar window) ──
        roll_high = high.rolling(min(100, n)).max()
        roll_low  = low.rolling(min(100, n)).min()
        rng       = roll_high - roll_low
        df["FIB_236"] = roll_high - 0.236 * rng
        df["FIB_382"] = roll_high - 0.382 * rng
        df["FIB_500"] = roll_high - 0.500 * rng
        df["FIB_618"] = roll_high - 0.618 * rng
        df["FIB_786"] = roll_high - 0.786 * rng

        # ── Order Blocks ──
        df["OB_Bull"] = Indicators._bull_order_blocks(df)
        df["OB_Bear"] = Indicators._bear_order_blocks(df)

        # ── Liquidity Hunt levels ──
        df["LIQ_High"] = high.rolling(min(20, n)).max()
        df["LIQ_Low"]  = low.rolling(min(20, n)).min()

        return df

    @staticmethod
    def _atr(high, low, close, window=14):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low  - close.shift(1)).abs()
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=window, adjust=False).mean()
        # Fallback if too short
        atr = atr.fillna(tr.rolling(window, min_periods=1).mean())
        return atr

    @staticmethod
    def _bull_order_blocks(df: pd.DataFrame):
        """Last bearish candle before a bullish impulse (demand zone)."""
        close = df["Close"]
        open_ = df["Open"]
        ob = pd.Series(np.nan, index=df.index)
        for i in range(2, len(df) - 1):
            # Bearish candle i followed by bullish move i+1
            if (open_.iloc[i] > close.iloc[i] and          # bearish body
                close.iloc[i+1] > open_.iloc[i]):           # next closes above ob open
                ob.iloc[i] = df["Low"].iloc[i]
        return ob

    @staticmethod
    def _bear_order_blocks(df: pd.DataFrame):
        """Last bullish candle before a bearish impulse (supply zone)."""
        close = df["Close"]
        open_ = df["Open"]
        ob = pd.Series(np.nan, index=df.index)
        for i in range(2, len(df) - 1):
            if (close.iloc[i] > open_.iloc[i] and
                close.iloc[i+1] < open_.iloc[i]):
                ob.iloc[i] = df["High"].iloc[i]
        return ob

# ═══════════════════════════════════════════════════════════════
# § 8 · SUPPORT / RESISTANCE
# ═══════════════════════════════════════════════════════════════
class SRDetector:

    @staticmethod
    def find_levels(df: pd.DataFrame, window: int = 10,
                    max_levels: int = 8) -> List[float]:
        highs  = df["High"].values
        lows   = df["Low"].values
        levels = []
        for i in range(window, len(df) - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                levels.append(float(highs[i]))
            if lows[i]  == min(lows[i-window:i+window+1]):
                levels.append(float(lows[i]))
        # Cluster nearby levels
        if not levels:
            return []
        levels.sort()
        clustered = [levels[0]]
        atr = df["ATR"].iloc[-1] if "ATR" in df else (df["High"] - df["Low"]).mean()
        for lvl in levels[1:]:
            if lvl - clustered[-1] > atr * 0.5:
                clustered.append(lvl)
        return clustered[-max_levels:]

    @staticmethod
    def prev_day_hl(df: pd.DataFrame) -> Tuple[float, float]:
        """Previous day high/low."""
        if "date" not in df.columns:
            df = df.copy()
            try:
                df["date"] = df.index.date
            except Exception:
                return 0.0, 0.0
        dates = sorted(df["date"].unique())
        if len(dates) < 2:
            return 0.0, 0.0
        prev = df[df["date"] == dates[-2]]
        return float(prev["High"].max()), float(prev["Low"].min())

    @staticmethod
    def nearest_resistance_above(price: float, levels: List[float],
                                  margin: float = 0.0) -> Optional[float]:
        above = [l for l in levels if l > price + margin]
        return min(above) if above else None

    @staticmethod
    def nearest_support_below(price: float, levels: List[float],
                               margin: float = 0.0) -> Optional[float]:
        below = [l for l in levels if l < price - margin]
        return max(below) if below else None

# ═══════════════════════════════════════════════════════════════
# § 9 · ORB STRATEGY  (Opening Range Breakout — 5-min rules)
# ═══════════════════════════════════════════════════════════════
class ORBStrategy:
    """
    Rules (from strategy image):
    ─────────────────────────────
    • Opening Range = first OR candle(s) from 9:15
    • Stock must be in 1–1.5 % momentum (avoid >1.5 %)
    • Find setup before 9:45, no entry after 10:00
    • Exit all by 14:30
    • RR target 1:2  |  partial close at 1:1 (half qty)
    • After 1:1 trail SL to CTC (breakeven)
    • Buy-only (long only)

    Anti-fake-breakout filters:
    ───────────────────────────
    1. Close confirmation (candle CLOSES above OR high)
    2. Volume spike (volume > 1.3× 20-bar avg)
    3. ATR-based distance (breakout > 0.25 × ATR)
    4. Candle body ratio (body > 40 % of range)
    5. No major resistance within 1.5 × ATR above entry
    6. Not a repeated failed breakout (same OR high already tested)
    """

    OR_CANDLES      = 1     # # of 5-min bars for opening range (1 = 09:15-09:20)
    NO_ENTRY_HOUR   = 10    # no new entries at or after this hour
    NO_ENTRY_MIN    = 0
    SETUP_DEADLINE  = (9, 45)
    EXIT_HOUR       = 14
    EXIT_MIN        = 30
    MOM_MIN         = 1.0   # % minimum momentum
    MOM_MAX         = 1.5   # % avoid if already moved more than this

    @staticmethod
    def generate_signals(df: pd.DataFrame,
                          ind_config: Dict[str, bool] = None,
                          for_backtest: bool = True) -> pd.Series:
        """Return Series of 'BUY' / '' at each candle index."""
        if ind_config is None:
            ind_config = {}

        df    = df.copy()
        sig   = pd.Series("", index=df.index)

        # ── Ensure date/time columns ──
        try:
            df["_date"] = df.index.date
            df["_hour"] = df.index.hour
            df["_min"]  = df.index.minute
        except Exception:
            return sig

        for day, ddf in df.groupby("_date"):
            ddf = ddf.sort_index()
            if len(ddf) < ORBStrategy.OR_CANDLES + 2:
                continue

            # Opening Range
            or_bars = ddf.iloc[:ORBStrategy.OR_CANDLES]
            or_high = float(or_bars["High"].max())
            or_low  = float(or_bars["Low"].min())
            or_rng  = or_high - or_low

            atr_val = ddf["ATR"].median() if "ATR" in ddf else or_rng
            if atr_val == 0 or np.isnan(atr_val):
                atr_val = or_rng or 1.0

            # S/R levels for the day
            sr_levels = SRDetector.find_levels(ddf, window=min(5, len(ddf)//3))

            failed_breakout_count = 0

            for i in range(ORBStrategy.OR_CANDLES, len(ddf)):
                c     = ddf.iloc[i]
                ts    = ddf.index[i]
                hour  = ts.hour
                minute = ts.minute

                # Time gate
                if (hour > ORBStrategy.NO_ENTRY_HOUR or
                   (hour == ORBStrategy.NO_ENTRY_HOUR and minute >= ORBStrategy.NO_ENTRY_MIN)):
                    break
                if (hour > ORBStrategy.SETUP_DEADLINE[0] or
                   (hour == ORBStrategy.SETUP_DEADLINE[0] and minute > ORBStrategy.SETUP_DEADLINE[1])):
                    break

                # Already have a signal today
                if (sig[ddf.index[:i]] == "BUY").any():
                    break

                close = float(c["Close"])
                open_ = float(c["Open"])
                high  = float(c["High"])
                low_  = float(c["Low"])
                vol   = float(c.get("Volume", 0) or 0)
                body  = abs(close - open_)
                rng   = high - low_
                if rng == 0:
                    continue

                # ─── BULLISH BREAKOUT CHECK ───
                if close <= or_high:
                    # Check for failed breakout (wick above but closed below)
                    if high > or_high and close <= or_high:
                        failed_breakout_count += 1
                    continue

                breakout_dist = close - or_high

                # Filter 1 – Close confirmation (already ensured above)
                # Filter 2 – ATR minimum distance
                if breakout_dist < 0.25 * atr_val:
                    continue

                # Filter 3 – Body ratio (avoid doji/spinning-top breakouts)
                if rng > 0 and body / rng < 0.40:
                    continue

                # Filter 4 – Volume spike
                vol_ma = float(ddf["VOL_MA"].iloc[i]) if "VOL_MA" in ddf else 0
                if vol_ma > 0 and vol < vol_ma * 1.1:   # relaxed to 1.1×
                    continue

                # Filter 5 – No immediate heavy resistance above
                near_res = SRDetector.nearest_resistance_above(
                    close, sr_levels, margin=atr_val * 0.1)
                if near_res and (near_res - close) < atr_val * 1.5:
                    continue

                # Filter 6 – Repeated failed breakout at same level → skip
                if failed_breakout_count >= 2:
                    continue

                # ─── INDICATOR CONFIRMATION (optional) ───
                if not ORBStrategy._confirm_indicators(ddf, i, ind_config):
                    continue

                sig.at[ts] = "BUY"
                break  # one signal per day

        return sig

    @staticmethod
    def _confirm_indicators(ddf: pd.DataFrame, i: int,
                             cfg: Dict[str, bool]) -> bool:
        c = ddf.iloc[i]

        def val(col):
            return float(c[col]) if col in c and not pd.isna(c[col]) else None

        if cfg.get("RSI"):
            v = val("RSI")
            if v is not None and v < 50:
                return False

        if cfg.get("MACD"):
            m, s = val("MACD"), val("MACD_Signal")
            if m is not None and s is not None and m < s:
                return False

        if cfg.get("ADX"):
            v = val("ADX")
            if v is not None and v < 20:
                return False

        if cfg.get("EMA"):
            e9, e20 = val("EMA9"), val("EMA20")
            close   = val("Close")
            if e20 is not None and close is not None and close < e20:
                return False

        if cfg.get("SMA"):
            s20  = val("SMA20")
            close = val("Close")
            if s20 is not None and close is not None and close < s20:
                return False

        if cfg.get("EMA/SMA Crossover"):
            e9, s20 = val("EMA9"), val("SMA20")
            if e9 is not None and s20 is not None and e9 < s20:
                return False

        if cfg.get("Volume"):
            vr = val("VOL_RATIO")
            if vr is not None and vr < 1.0:
                return False

        if cfg.get("Order Block"):
            ob_bull = val("OB_Bull")
            close   = val("Close")
            if ob_bull is not None and close is not None and close < ob_bull:
                return False

        if cfg.get("Volatility Filter"):
            bb_w = val("BB_Width")
            atr  = val("ATR")
            if bb_w is not None and atr is not None and atr > 0:
                if bb_w < atr * 0.5:   # too quiet market
                    return False

        if cfg.get("Volume Profile"):
            vwap  = val("VWAP")
            close = val("Close")
            if vwap is not None and close is not None and close < vwap:
                return False

        if cfg.get("Fibonacci"):
            fib618 = val("FIB_618")
            close  = val("Close")
            if fib618 is not None and close is not None and close < fib618:
                return False

        if cfg.get("Liquidity Hunt"):
            liq_h  = val("LIQ_High")
            close  = val("Close")
            if liq_h is not None and close is not None and close < liq_h * 0.995:
                return False

        return True

# ═══════════════════════════════════════════════════════════════
# § 10 · SIMPLE STRATEGIES
# ═══════════════════════════════════════════════════════════════
class SimpleStrategy:

    @staticmethod
    def simple_buy(df: pd.DataFrame, cfg: Dict[str, bool]) -> pd.Series:
        """
        Simple Buy: Enter when close > EMA20 and RSI > 50 (or no indicators).
        One trade per bar allowed.
        """
        df  = df.copy()
        sig = pd.Series("", index=df.index)
        if len(df) < 3:
            return sig
        for i in range(2, len(df)):
            if not ORBStrategy._confirm_indicators(df, i, cfg):
                continue
            c  = df.iloc[i]
            p1 = df.iloc[i-1]
            # Base: close breaks above previous high
            if float(c["Close"]) > float(p1["High"]):
                sig.iloc[i] = "BUY"
        return sig

    @staticmethod
    def simple_sell(df: pd.DataFrame, cfg: Dict[str, bool]) -> pd.Series:
        """
        Simple Sell exit: Close below EMA20 or RSI cross below 50.
        (Exit signal only — no shorting as user is buy-only.)
        """
        df  = df.copy()
        sig = pd.Series("", index=df.index)
        if len(df) < 3:
            return sig
        for i in range(2, len(df)):
            c  = df.iloc[i]
            p1 = df.iloc[i-1]
            # Exit trigger: close below previous low
            if float(c["Close"]) < float(p1["Low"]):
                sig.iloc[i] = "EXIT"
        return sig

# ═══════════════════════════════════════════════════════════════
# § 11 · SL / TARGET MANAGER
# ═══════════════════════════════════════════════════════════════
class SLManager:
    """Compute, trail and check SL and Target for a position."""

    @staticmethod
    def compute_sl(entry: float, sl_type: str, params: dict,
                   df: pd.DataFrame = None, idx: int = -1) -> float:
        atr = float(df["ATR"].iloc[idx]) if df is not None and "ATR" in df else 0
        atr = atr if atr > 0 else entry * 0.01

        if sl_type == "Fixed Points":
            return entry - params.get("sl_pts", 5)

        if sl_type == "ATR Based":
            return entry - params.get("sl_atr_mult", 1.5) * atr

        if sl_type == "Risk-Reward Based":
            rr  = params.get("rr", 2.0)
            tgt = entry + params.get("target_pts", 10)
            return entry - (tgt - entry) / rr

        if sl_type == "Trailing (Points)":
            return entry - params.get("sl_pts", 5)

        if sl_type == "Trailing (Candle Low/High)":
            if df is not None and idx >= 1:
                return float(df["Low"].iloc[idx - 1])
            return entry - params.get("sl_pts", 5)

        if sl_type == "Trailing (Swing Low/High)":
            if df is not None and idx >= 5:
                return float(df["Low"].iloc[max(0, idx-5):idx].min())
            return entry - params.get("sl_pts", 5)

        if sl_type == "Signal Reversal":
            return entry - params.get("sl_pts", 5)

        if sl_type == "Volatility (BB) Based":
            if df is not None and "BB_Lower" in df and idx >= 0:
                bb_low = float(df["BB_Lower"].iloc[idx])
                if not np.isnan(bb_low):
                    return bb_low
            return entry - params.get("sl_atr_mult", 2.0) * atr

        return entry - params.get("sl_pts", 5)

    @staticmethod
    def compute_target(entry: float, sl: float, target_type: str,
                        params: dict, df: pd.DataFrame = None,
                        idx: int = -1) -> float:
        atr = float(df["ATR"].iloc[idx]) if df is not None and "ATR" in df else 0
        atr = atr if atr > 0 else entry * 0.01
        risk = entry - sl

        if target_type == "Fixed Points":
            return entry + params.get("target_pts", 10)

        if target_type == "ATR Based":
            return entry + params.get("target_atr_mult", 2.0) * atr

        if target_type == "Risk-Reward Based":
            return entry + params.get("rr", 2.0) * risk

        if target_type in ("Trailing Target (Display Only)",):
            return entry + params.get("rr", 2.0) * risk   # display calc

        if target_type == "Volatility (BB) Based":
            if df is not None and "BB_Upper" in df and idx >= 0:
                bb_up = float(df["BB_Upper"].iloc[idx])
                if not np.isnan(bb_up):
                    return bb_up
            return entry + params.get("rr", 2.0) * risk

        return entry + params.get("target_pts", 10)

    @staticmethod
    def trail_sl(current_sl: float, sl_type: str, ltp: float,
                 entry: float, params: dict, df: pd.DataFrame = None,
                 idx: int = -1, highest_price: float = 0.0) -> float:
        """Return new (possibly higher) SL based on trail type."""
        if sl_type == "Trailing (Points)":
            new_sl = ltp - params.get("sl_pts", 5)
            return max(current_sl, new_sl)

        if sl_type == "Trailing (Candle Low/High)":
            if df is not None and idx >= 1:
                return max(current_sl, float(df["Low"].iloc[idx-1]))
            return current_sl

        if sl_type == "Trailing (Swing Low/High)":
            if df is not None and idx >= 5:
                swing_low = float(df["Low"].iloc[max(0,idx-5):idx].min())
                return max(current_sl, swing_low)
            return current_sl

        if sl_type == "ATR Based":
            atr = float(df["ATR"].iloc[idx]) if df is not None and "ATR" in df else 0
            if atr <= 0:
                return current_sl
            new_sl = ltp - params.get("sl_atr_mult", 1.5) * atr
            return max(current_sl, new_sl)

        return current_sl

# ═══════════════════════════════════════════════════════════════
# § 12 · BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════════════
class BacktestEngine:
    """
    Strict simulation rules:
    • Signal on candle N  →  entry at candle N+1 OPEN
    • SL check first against candle LOW  (conservative for long)
    • Target check against candle HIGH
    • ORB time windows enforced
    • Partial close at 1:1  +  trail to CTC
    """

    @staticmethod
    def run(df: pd.DataFrame, strategy: str,
            sl_type: str, target_type: str,
            params: dict, ind_config: Dict[str, bool]) -> Dict:

        df = Indicators.add_all(df)

        # Generate signals
        if strategy == "ORB (Opening Range Breakout)":
            signals = ORBStrategy.generate_signals(df, ind_config, for_backtest=True)
        elif strategy == "Simple Buy":
            signals = SimpleStrategy.simple_buy(df, ind_config)
        else:
            signals = SimpleStrategy.simple_buy(df, ind_config)

        trades    : List[dict] = []
        equity    = [params.get("initial_capital", 100000.0)]
        capital   = equity[0]
        qty       = params.get("quantity", 1)
        in_trade  = False
        entry_price = sl = target = target1 = 0.0
        entry_time = None
        highest_seen = 0.0
        partial_done = False

        try:
            df["_date"] = df.index.date
            df["_hour"] = df.index.hour
            df["_min"]  = df.index.minute
        except Exception:
            pass

        for i in range(len(df)):
            c     = df.iloc[i]
            ts    = df.index[i]
            candle_open  = float(c["Open"])
            candle_high  = float(c["High"])
            candle_low   = float(c["Low"])
            candle_close = float(c["Close"])

            # ── Force exit at 14:30 ──
            if in_trade:
                try:
                    if c["_hour"] == 14 and c["_min"] >= 30:
                        pnl = (candle_close - entry_price) * qty
                        pnl -= pnl * 0.01   # rough charges
                        capital += pnl
                        trades.append(BacktestEngine._trade(
                            entry_time, ts, entry_price, candle_close,
                            sl, target, pnl, "Force Exit 14:30", capital))
                        equity.append(capital)
                        in_trade = partial_done = False
                        continue
                except Exception:
                    pass

            # ── SL / Target check (while in trade) ──
            if in_trade:
                highest_seen = max(highest_seen, candle_high)

                # Update trailing SL
                sl = SLManager.trail_sl(sl, sl_type, candle_close, entry_price,
                                        params, df, i, highest_seen)

                # Half-position target (CTC trail at 1:1)
                target1_hit = candle_high >= target1

                if not partial_done and target1_hit:
                    partial_done = True
                    # Close half; trail SL to entry (CTC = breakeven)
                    half_pnl = (target1 - entry_price) * (qty // 2)
                    capital += half_pnl
                    sl = entry_price   # trail to breakeven

                # SL hit — check LOW first (conservative)
                if candle_low <= sl:
                    exit_p = sl
                    pnl    = (exit_p - entry_price) * qty
                    pnl   -= abs(pnl) * 0.01
                    capital += pnl
                    trades.append(BacktestEngine._trade(
                        entry_time, ts, entry_price, exit_p,
                        sl, target, pnl, "SL Hit", capital))
                    equity.append(capital)
                    in_trade = partial_done = False
                    continue

                # Target hit — check HIGH
                if candle_high >= target:
                    exit_p = target
                    rem_qty = qty // 2 if partial_done else qty
                    pnl    = (exit_p - entry_price) * rem_qty
                    pnl   -= abs(pnl) * 0.01
                    capital += pnl
                    trades.append(BacktestEngine._trade(
                        entry_time, ts, entry_price, exit_p,
                        sl, target, pnl, "Target Hit", capital))
                    equity.append(capital)
                    in_trade = partial_done = False
                    continue

                # Signal reversal exit
                if sl_type == "Signal Reversal" and strategy != "ORB (Opening Range Breakout)":
                    exit_sig = SimpleStrategy.simple_sell(
                        df.iloc[max(0,i-5):i+1], {})
                    if exit_sig.iloc[-1] == "EXIT":
                        pnl = (candle_close - entry_price) * qty
                        pnl -= abs(pnl) * 0.01
                        capital += pnl
                        trades.append(BacktestEngine._trade(
                            entry_time, ts, entry_price, candle_close,
                            sl, target, pnl, "Signal Exit", capital))
                        equity.append(capital)
                        in_trade = partial_done = False
                        continue

            # ── Entry on N+1 open ──
            if not in_trade and i > 0:
                prev_sig = signals.iloc[i - 1]
                if prev_sig == "BUY":
                    entry_price  = candle_open   # N+1 OPEN
                    entry_time   = ts
                    sl           = SLManager.compute_sl(
                        entry_price, sl_type, params, df, i)
                    target       = SLManager.compute_target(
                        entry_price, sl, target_type, params, df, i)
                    # 1:1 partial target
                    target1      = entry_price + (entry_price - sl)
                    highest_seen = entry_price
                    partial_done = False
                    in_trade     = True

        # Close open trade at end
        if in_trade and len(df) > 0:
            exit_p = float(df["Close"].iloc[-1])
            pnl    = (exit_p - entry_price) * qty
            pnl   -= abs(pnl) * 0.01
            capital += pnl
            trades.append(BacktestEngine._trade(
                entry_time, df.index[-1], entry_price, exit_p,
                sl, target, pnl, "End of Data", capital))
            equity.append(capital)

        return BacktestEngine._summarise(
            trades, equity, params.get("initial_capital", 100000.0))

    @staticmethod
    def _trade(entry_t, exit_t, ep, xp, sl, tgt, pnl, reason, cap):
        return dict(
            entry_time   = str(entry_t)[:19],
            exit_time    = str(exit_t)[:19],
            entry_price  = round(ep, 4),
            exit_price   = round(xp, 4),
            sl           = round(sl, 4),
            target       = round(tgt, 4),
            pnl          = round(pnl, 2),
            reason       = reason,
            capital      = round(cap, 2),
            status       = "WIN" if pnl >= 0 else "LOSS",
        )

    @staticmethod
    def _summarise(trades: List[dict], equity: List[float],
                   initial: float) -> Dict:
        if not trades:
            return dict(trades=[], equity=equity, summary={})
        df = pd.DataFrame(trades)
        wins   = (df["pnl"] >= 0).sum()
        losses = (df["pnl"] <  0).sum()
        total  = len(df)
        final_cap = equity[-1] if equity else initial
        gross_pnl = df["pnl"].sum()
        max_dd    = BacktestEngine._max_drawdown(equity)
        avg_win   = df.loc[df["pnl"]>=0,"pnl"].mean() if wins > 0 else 0
        avg_loss  = df.loc[df["pnl"]< 0,"pnl"].mean() if losses > 0 else 0
        profit_factor = (
            df.loc[df["pnl"]>=0,"pnl"].sum() /
            abs(df.loc[df["pnl"]<0,"pnl"].sum())
            if losses > 0 and df.loc[df["pnl"]<0,"pnl"].sum() != 0 else float("inf"))

        return dict(
            trades  = trades,
            equity  = equity,
            summary = dict(
                total_trades   = total,
                wins           = int(wins),
                losses         = int(losses),
                win_rate       = round(wins/total*100, 1) if total else 0,
                total_pnl      = round(gross_pnl, 2),
                initial_capital= round(initial, 2),
                final_capital  = round(final_cap, 2),
                return_pct     = round((final_cap-initial)/initial*100, 2),
                max_drawdown   = round(max_dd, 2),
                profit_factor  = round(profit_factor, 2),
                avg_win        = round(avg_win, 2),
                avg_loss       = round(avg_loss, 2),
                sharpe         = BacktestEngine._sharpe(
                    [t["pnl"] for t in trades]),
            )
        )

    @staticmethod
    def _max_drawdown(equity: List[float]) -> float:
        if not equity:
            return 0.0
        peak = equity[0]
        md   = 0.0
        for v in equity:
            peak = max(peak, v)
            md   = max(md, peak - v)
        return md

    @staticmethod
    def _sharpe(pnls: List[float]) -> float:
        if len(pnls) < 2:
            return 0.0
        arr = np.array(pnls)
        std = arr.std()
        if std == 0:
            return 0.0
        return round(float(arr.mean() / std * np.sqrt(252)), 2)

# ═══════════════════════════════════════════════════════════════
# § 13 · DHAN API CLIENT
# ═══════════════════════════════════════════════════════════════
class DhanAPI:
    BASE = "https://api.dhan.co"

    def __init__(self, client_id: str, access_token: str):
        self.client_id    = client_id
        self.access_token = access_token
        self.session      = requests.Session()
        self.session.headers.update({
            "client-id"    : client_id,
            "access-token" : access_token,
            "Content-Type" : "application/json",
        })

    def _get(self, path: str, params=None):
        try:
            r = self.session.get(f"{self.BASE}{path}", params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def _post(self, path: str, body: dict):
        try:
            r = self.session.post(f"{self.BASE}{path}", json=body, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def _delete(self, path: str):
        try:
            r = self.session.delete(f"{self.BASE}{path}", timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def get_ltp(self, security_id: str, exchange: str = "NSE_EQ") -> float:
        body = {
            "NSE_EQ": [int(security_id)] if exchange == "NSE_EQ" else [],
            "NSE_FNO": [int(security_id)] if exchange == "NSE_FNO" else [],
            "BSE_EQ": [int(security_id)] if exchange == "BSE_EQ" else [],
        }
        r = self._post("/v2/marketfeed/ltp", body)
        try:
            data = r.get("data", {})
            for k, v in data.items():
                for item in v:
                    if "last_price" in item:
                        return float(item["last_price"])
        except Exception:
            pass
        return 0.0

    def place_order(self, trading_symbol: str, security_id: str,
                    qty: int, order_type: str = "MARKET",
                    price: float = 0.0, exchange: str = "NSE_EQ",
                    product: str = "INTRADAY") -> dict:
        body = {
            "dhanClientId"    : self.client_id,
            "correlationId"   : str(uuid.uuid4())[:20],
            "transactionType" : "BUY",
            "exchangeSegment" : exchange,
            "productType"     : product,
            "orderType"       : order_type,
            "validity"        : "DAY",
            "tradingSymbol"   : trading_symbol.upper(),
            "securityId"      : str(security_id),
            "quantity"        : int(qty),
            "disclosedQuantity": 0,
            "price"           : round(float(price), 2),
            "triggerPrice"    : 0,
            "afterMarketOrder": False,
        }
        return self._post("/v2/orders", body)

    def cancel_order(self, order_id: str) -> dict:
        return self._delete(f"/v2/orders/{order_id}")

    def get_positions(self) -> list:
        r = self._get("/v2/positions")
        return r if isinstance(r, list) else []

    def get_orders(self) -> list:
        r = self._get("/v2/orders")
        return r if isinstance(r, list) else []

    def is_connected(self) -> bool:
        r = self._get("/v2/fundlimit")
        return "error" not in r

# ═══════════════════════════════════════════════════════════════
# § 14 · CHART BUILDER
# ═══════════════════════════════════════════════════════════════
class ChartBuilder:

    @staticmethod
    def candle_chart(df: pd.DataFrame,
                     trades: List[dict] = None,
                     signals: pd.Series = None,
                     enabled_indicators: Dict[str, bool] = None,
                     title: str = "Price Chart") -> go.Figure:

        if enabled_indicators is None:
            enabled_indicators = {}

        rows  = 1
        specs = [[{"type":"candlestick"}]]
        row_heights = [0.55]

        show_vol   = enabled_indicators.get("Volume", False) and "Volume" in df
        show_rsi   = enabled_indicators.get("RSI", False) and "RSI" in df
        show_macd  = enabled_indicators.get("MACD", False) and "MACD" in df

        if show_vol:
            rows += 1; specs.append([{"type":"bar"}]); row_heights.append(0.15)
        if show_rsi:
            rows += 1; specs.append([{"type":"scatter"}]); row_heights.append(0.15)
        if show_macd:
            rows += 1; specs.append([{"type":"scatter"}]); row_heights.append(0.15)

        total = sum(row_heights)
        row_heights = [h/total for h in row_heights]

        fig = make_subplots(
            rows=rows, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            specs=specs,
        )

        # ── Candlestick ──
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"],  close=df["Close"],
            increasing_line_color="#10b981", decreasing_line_color="#ef4444",
            increasing_fillcolor="#10b98144",decreasing_fillcolor="#ef444444",
            name="Price", line_width=1,
        ), row=1, col=1)

        # ── Overlays ──
        ema_colors = {"EMA9":"#f59e0b","EMA20":"#3b82f6","EMA50":"#8b5cf6","EMA200":"#ec4899"}
        for k, color in ema_colors.items():
            if enabled_indicators.get("EMA") and k in df:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[k], name=k, line=dict(color=color,width=1),
                    opacity=0.85,
                ), row=1, col=1)

        if enabled_indicators.get("SMA"):
            for k, color in [("SMA20","#06b6d4"),("SMA50","#a78bfa")]:
                if k in df:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df[k], name=k,
                        line=dict(color=color, width=1, dash="dash"), opacity=0.8,
                    ), row=1, col=1)

        if enabled_indicators.get("Bollinger Bands") or enabled_indicators.get("Volatility Filter"):
            if "BB_Upper" in df:
                for k, clr in [("BB_Upper","#94a3b8"),("BB_Mid","#64748b"),("BB_Lower","#94a3b8")]:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df[k], name=k,
                        line=dict(color=clr,width=1,dash="dot"), opacity=0.5,
                    ), row=1, col=1)

        if enabled_indicators.get("Volume Profile") and "VWAP" in df:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["VWAP"], name="VWAP",
                line=dict(color="#f97316",width=1.5), opacity=0.9,
            ), row=1, col=1)

        # ── Fibonacci ──
        if enabled_indicators.get("Fibonacci"):
            for fib, clr in [("FIB_382","#fbbf24"),("FIB_500","#fb923c"),("FIB_618","#f87171")]:
                if fib in df:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df[fib], name=fib.replace("FIB_","Fib "),
                        line=dict(color=clr, width=1, dash="dot"), opacity=0.7,
                    ), row=1, col=1)

        # ── Order Blocks ──
        if enabled_indicators.get("Order Block") and "OB_Bull" in df:
            ob_idx = df["OB_Bull"].dropna().index
            if len(ob_idx):
                fig.add_trace(go.Scatter(
                    x=ob_idx, y=df.loc[ob_idx,"OB_Bull"],
                    mode="markers", marker=dict(symbol="square",size=8,
                        color="#10b981",opacity=0.7),
                    name="Bull OB",
                ), row=1, col=1)

        # ── Signals ──
        if signals is not None:
            buy_idx = signals[signals=="BUY"].index
            if len(buy_idx):
                fig.add_trace(go.Scatter(
                    x=buy_idx, y=df.loc[buy_idx,"Low"]*0.999,
                    mode="markers+text",
                    marker=dict(symbol="triangle-up",size=12,color="#10b981"),
                    text=["▲"]*len(buy_idx), textposition="bottom center",
                    name="BUY Signal",
                ), row=1, col=1)

        # ── Trade annotations ──
        if trades:
            for t in trades[-50:]:   # last 50 for performance
                try:
                    ep = t["entry_price"]; xp = t["exit_price"]
                    colour = "#10b981" if t.get("status")=="WIN" else "#ef4444"
                    fig.add_hline(y=ep, line=dict(color=colour,width=0.8,dash="dot"),
                                  row=1, col=1, opacity=0.4)
                except Exception:
                    pass

        cur_row = 2

        # ── Volume sub-plot ──
        if show_vol:
            colors = ["#10b981" if float(df["Close"].iloc[i]) >= float(df["Open"].iloc[i])
                      else "#ef4444" for i in range(len(df))]
            fig.add_trace(go.Bar(
                x=df.index, y=df["Volume"], name="Volume",
                marker_color=colors, opacity=0.6,
            ), row=cur_row, col=1)
            if "VOL_MA" in df:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["VOL_MA"], name="Vol MA",
                    line=dict(color="#f59e0b",width=1), opacity=0.8,
                ), row=cur_row, col=1)
            fig.update_yaxes(title_text="Vol", row=cur_row, col=1,
                             title_font=dict(size=10))
            cur_row += 1

        # ── RSI sub-plot ──
        if show_rsi:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["RSI"], name="RSI",
                line=dict(color="#a78bfa",width=1.5),
            ), row=cur_row, col=1)
            for lvl, clr in [(70,"#ef4444"),(50,"#94a3b8"),(30,"#10b981")]:
                fig.add_hline(y=lvl, line=dict(color=clr,width=0.8,dash="dot"),
                              row=cur_row, col=1, opacity=0.5)
            fig.update_yaxes(title_text="RSI", range=[0,100],
                             row=cur_row, col=1, title_font=dict(size=10))
            cur_row += 1

        # ── MACD sub-plot ──
        if show_macd:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["MACD"], name="MACD",
                line=dict(color="#3b82f6",width=1.5),
            ), row=cur_row, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df["MACD_Signal"], name="Signal",
                line=dict(color="#f59e0b",width=1,dash="dash"),
            ), row=cur_row, col=1)
            hist_col = ["#10b981" if (df["MACD_Hist"].iloc[i] or 0) >= 0
                        else "#ef4444" for i in range(len(df))]
            fig.add_trace(go.Bar(
                x=df.index, y=df["MACD_Hist"], name="Hist",
                marker_color=hist_col, opacity=0.5,
            ), row=cur_row, col=1)
            fig.update_yaxes(title_text="MACD", row=cur_row, col=1,
                             title_font=dict(size=10))

        fig.update_layout(
            title=dict(text=title, font=dict(family="JetBrains Mono",size=14,color="#e2e8f0")),
            paper_bgcolor="#0a0e17", plot_bgcolor="#0a0e17",
            font=dict(color="#94a3b8", family="JetBrains Mono", size=11),
            xaxis_rangeslider_visible=False,
            legend=dict(bgcolor="#111827", bordercolor="#1e2d45",
                        borderwidth=1, font=dict(size=10)),
            margin=dict(l=50,r=20,t=40,b=20),
            height=600 if rows == 1 else 600 + (rows-1)*120,
        )
        fig.update_xaxes(
            showgrid=True, gridcolor="#1e2d45", gridwidth=0.5,
            showline=True, linecolor="#1e2d45",
        )
        fig.update_yaxes(
            showgrid=True, gridcolor="#1e2d45", gridwidth=0.5,
            showline=True, linecolor="#1e2d45",
        )
        return fig

    @staticmethod
    def equity_curve(equity: List[float], trades: List[dict]) -> go.Figure:
        if not equity:
            return go.Figure()
        x = list(range(len(equity)))
        base = equity[0]
        colors = ["#10b981" if v >= base else "#ef4444" for v in equity]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=equity, mode="lines",
            line=dict(color="#00d4aa", width=2),
            fill="tozeroy", fillcolor="#00d4aa11",
            name="Equity",
        ))
        fig.add_hline(y=base, line=dict(color="#94a3b8",width=1,dash="dot"))
        fig.update_layout(
            title=dict(text="Equity Curve",
                       font=dict(family="JetBrains Mono",size=13,color="#e2e8f0")),
            paper_bgcolor="#0a0e17", plot_bgcolor="#0a0e17",
            font=dict(color="#94a3b8",family="JetBrains Mono",size=11),
            margin=dict(l=50,r=20,t=40,b=20), height=280,
        )
        fig.update_xaxes(showgrid=True,gridcolor="#1e2d45",showline=True,linecolor="#1e2d45")
        fig.update_yaxes(showgrid=True,gridcolor="#1e2d45",showline=True,linecolor="#1e2d45")
        return fig

# ═══════════════════════════════════════════════════════════════
# § 15 · SECTOR SCANNER  (live trading only)
# ═══════════════════════════════════════════════════════════════
class SectorScanner:
    """Scan NSE sectors for momentum stocks (1–1.5 %)."""

    @staticmethod
    def scan(sectors: List[str], top_n: int = 4) -> List[dict]:
        results = []
        for sec in sectors:
            tickers = NSE_SECTORS.get(sec, [])[:top_n]
            for t in tickers:
                try:
                    mom = DataFetcher.momentum_pct(t)
                    if 1.0 <= mom <= 1.5:
                        results.append({"ticker": t, "sector": sec,
                                        "momentum": round(mom, 2)})
                except Exception:
                    pass
        return results

# ═══════════════════════════════════════════════════════════════
# § 16 · SIDEBAR  (shared config panel)
# ═══════════════════════════════════════════════════════════════
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:8px 0 16px'>
          <span style='font-family:JetBrains Mono;font-size:18px;
                       font-weight:700;color:#00d4aa;letter-spacing:.1em'>
            ⚡ TRADING LAB
          </span><br>
          <span style='font-size:10px;color:#475569;font-family:JetBrains Mono;
                       letter-spacing:.15em'>PROFESSIONAL PLATFORM</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">📌 INSTRUMENT</div>',
                    unsafe_allow_html=True)

        ticker_name = st.selectbox("Ticker", list(TICKER_MAP.keys()),
                                   index=0, key="sb_ticker")
        custom_tk   = ""
        if ticker_name == "CUSTOM":
            custom_tk = st.text_input("Custom Ticker", "AAPL", key="sb_custom")

        ticker_sym  = DataFetcher.resolve_ticker(ticker_name, custom_tk)
        st.caption(f"Symbol: `{ticker_sym}`")

        tf_list   = list(TIMEFRAME_MAP.keys())
        period_list = list(PERIOD_MAP.keys())

        timeframe = st.selectbox("Timeframe", tf_list,
                                 index=tf_list.index("5m"), key="sb_tf")
        period    = st.selectbox("Period", period_list,
                                 index=period_list.index("1 Month"), key="sb_period")

        # Auto-correct period for interval
        allowed_periods = VALID_COMBOS.get(timeframe, list(PERIOD_MAP.values()))
        period_code     = PERIOD_MAP[period]
        if period_code not in allowed_periods:
            period_code = allowed_periods[-1]
            found_key   = [k for k,v in PERIOD_MAP.items() if v==period_code]
            if found_key:
                st.warning(f"Period auto-adjusted to **{found_key[0]}** for {timeframe} tf")

        st.markdown('<div class="section-header" style="margin-top:14px">🎯 STRATEGY</div>',
                    unsafe_allow_html=True)

        strategy    = st.selectbox("Strategy", STRATEGIES, key="sb_strategy")
        qty         = st.number_input("Quantity (Lots/Shares)", 1, 10000, 1, key="sb_qty")
        init_capital= st.number_input("Initial Capital (₹)", 1000, 10_000_000,
                                       100_000, step=10000, key="sb_capital")

        st.markdown('<div class="section-header" style="margin-top:14px">🛡️ SL & TARGET</div>',
                    unsafe_allow_html=True)

        sl_type     = st.selectbox("Stop Loss Type", SL_TYPES, key="sb_sl_type")
        target_type = st.selectbox("Target Type",   TARGET_TYPES, key="sb_tgt_type")

        c1, c2 = st.columns(2)
        sl_pts      = c1.number_input("SL Points",  0.1, 10000.0, 5.0,  step=0.5, key="sb_sl_pts")
        target_pts  = c2.number_input("Tgt Points", 0.1, 10000.0, 10.0, step=0.5, key="sb_tgt_pts")

        c3, c4 = st.columns(2)
        sl_atr_mult  = c3.number_input("SL ATR ×", 0.1, 10.0, 1.5, step=0.1, key="sb_sl_atr")
        tgt_atr_mult = c4.number_input("Tgt ATR ×", 0.1, 10.0, 2.0, step=0.1, key="sb_tgt_atr")
        rr           = st.slider("Risk:Reward (1:N)", 1.0, 5.0, 2.0, 0.5, key="sb_rr")

        if target_type == "Trailing Target (Display Only)":
            st.markdown(
                '<div class="info-box">📊 Trailing Target is <b>display-only</b>'
                ' — it tracks but never triggers an exit.</div>',
                unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:14px">📊 INDICATORS</div>',
                    unsafe_allow_html=True)

        if not TA_OK:
            st.warning("`ta` library missing — run `pip install ta`")

        ind_config = {}
        cols = st.columns(2)
        for j, ind in enumerate(INDICATORS):
            ind_config[ind] = cols[j%2].checkbox(ind, value=False,
                                                  key=f"ind_{ind.replace('/','_').replace(' ','_')}")

        st.markdown('<div class="section-header" style="margin-top:14px">🔑 DHAN API</div>',
                    unsafe_allow_html=True)

        dhan_client  = st.text_input("Client ID",    st.session_state.dhan_client_id,
                                      type="default", key="sb_dhan_cid")
        dhan_token   = st.text_input("Access Token", st.session_state.dhan_token,
                                      type="password", key="sb_dhan_tok")
        dhan_secid   = st.text_input("Security ID",  st.session_state.dhan_security_id,
                                      key="sb_dhan_secid")
        dhan_exch    = st.selectbox("Exchange", ["NSE_EQ","NSE_FNO","BSE_EQ","MCX_COMM","NSE_CURRENCY"],
                                     key="sb_dhan_exch")
        dhan_prod    = st.selectbox("Product",  ["INTRADAY","CNC","MARGIN","MTF"],
                                     key="sb_dhan_prod")

        if dhan_client:  st.session_state.dhan_client_id   = dhan_client
        if dhan_token:   st.session_state.dhan_token        = dhan_token
        if dhan_secid:   st.session_state.dhan_security_id  = dhan_secid
        st.session_state.dhan_exchange = dhan_exch
        st.session_state.dhan_product  = dhan_prod

        if st.button("🔗 Test Dhan Connection", key="sb_test_dhan"):
            if dhan_client and dhan_token:
                with st.spinner("Connecting…"):
                    api = DhanAPI(dhan_client, dhan_token)
                    if api.is_connected():
                        st.success("✅ Dhan Connected!")
                    else:
                        st.error("❌ Connection failed. Check credentials.")
            else:
                st.warning("Enter Client ID and Token first.")

    return dict(
        ticker_sym   = ticker_sym,
        ticker_name  = ticker_name,
        timeframe    = TIMEFRAME_MAP[timeframe],
        period       = period_code,
        strategy     = strategy,
        qty          = qty,
        init_capital = init_capital,
        sl_type      = sl_type,
        target_type  = target_type,
        sl_pts       = sl_pts,
        target_pts   = target_pts,
        sl_atr_mult  = sl_atr_mult,
        tgt_atr_mult = tgt_atr_mult,
        rr           = rr,
        ind_config   = ind_config,
    )

# ═══════════════════════════════════════════════════════════════
# § 17 · BACKTESTING TAB
# ═══════════════════════════════════════════════════════════════
def tab_backtest(cfg: dict):
    st.markdown("""
    <div style='padding:4px 0 16px'>
      <span style='font-family:JetBrains Mono;font-size:16px;font-weight:700;color:#00d4aa'>
        ◈ BACKTESTING ENGINE
      </span>
      <span style='font-size:11px;color:#475569;font-family:JetBrains Mono;margin-left:12px'>
        Entry: Candle N+1 Open  ·  SL vs Low first  ·  Target vs High
      </span>
    </div>
    """, unsafe_allow_html=True)

    col_run, col_clr, _ = st.columns([1,1,4])
    run_bt  = col_run.button("▶  RUN BACKTEST", key="bt_run", type="primary")
    clr_bt  = col_clr.button("🗑  CLEAR",        key="bt_clr")

    if clr_bt:
        st.session_state.bt_results   = None
        st.session_state.bt_trades    = []
        st.session_state.bt_chart_data= None
        st.rerun()

    if run_bt:
        with st.spinner("📡 Fetching data…"):
            df = DataFetcher.fetch(cfg["ticker_sym"],
                                   cfg["timeframe"], cfg["period"])
        if df is None or df.empty:
            st.error("❌ Failed to fetch data. Check ticker/timeframe/period.")
            return

        params = dict(
            initial_capital = cfg["init_capital"],
            quantity        = cfg["qty"],
            sl_pts          = cfg["sl_pts"],
            target_pts      = cfg["target_pts"],
            sl_atr_mult     = cfg["sl_atr_mult"],
            target_atr_mult = cfg["tgt_atr_mult"],
            rr              = cfg["rr"],
        )

        with st.spinner("⚙️ Running backtest…"):
            result = BacktestEngine.run(
                df, cfg["strategy"], cfg["sl_type"],
                cfg["target_type"], params, cfg["ind_config"])

        st.session_state.bt_results    = result["summary"]
        st.session_state.bt_trades     = result["trades"]
        st.session_state.bt_equity     = result["equity"]
        st.session_state.bt_chart_data = df
        st.session_state.bt_signals    = ORBStrategy.generate_signals(
            Indicators.add_all(df), cfg["ind_config"]) if cfg["strategy"].startswith("ORB") else None

    if st.session_state.bt_results is None:
        st.markdown("""
        <div class="info-box">
          ⬆️ Configure settings in the sidebar and click <b>RUN BACKTEST</b>.
          <br>ORB rules: Entry window 09:15–09:45 · No entry after 10:00 · Exit by 14:30
          · Partial close at 1:1 · Trail SL to CTC
        </div>
        """, unsafe_allow_html=True)
        return

    s = st.session_state.bt_results

    # ── Summary Metrics ──
    st.markdown('<div class="section-header">📈 PERFORMANCE SUMMARY</div>',
                unsafe_allow_html=True)
    m1,m2,m3,m4,m5,m6,m7,m8 = st.columns(8)
    def mc(col, label, val, clr="accent"):
        col.markdown(f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="val {clr}">{val}</div>
        </div>""", unsafe_allow_html=True)

    pnl_clr = "green" if s["total_pnl"] >= 0 else "red"
    ret_clr = "green" if s["return_pct"] >= 0 else "red"
    mc(m1,"Trades",      s["total_trades"])
    mc(m2,"Win Rate",    f"{s['win_rate']}%",
       "green" if s["win_rate"]>=50 else "red")
    mc(m3,"Total P&L",   f"₹{s['total_pnl']:,.0f}", pnl_clr)
    mc(m4,"Return",      f"{s['return_pct']}%", ret_clr)
    mc(m5,"Max DD",      f"₹{s['max_drawdown']:,.0f}", "red")
    mc(m6,"Profit Fact.",f"{s['profit_factor']:.2f}",
       "green" if s["profit_factor"]>=1 else "red")
    mc(m7,"Sharpe",      f"{s['sharpe']:.2f}",
       "green" if s["sharpe"]>=1 else "gold")
    mc(m8,"Final Cap",   f"₹{s['final_capital']:,.0f}")

    c_w, c_l = st.columns(2)
    c_w.metric("Avg Win",  f"₹{s['avg_win']:,.2f}")
    c_l.metric("Avg Loss", f"₹{s['avg_loss']:,.2f}")

    # ── Equity Curve ──
    if st.session_state.get("bt_equity"):
        st.plotly_chart(
            ChartBuilder.equity_curve(st.session_state.bt_equity,
                                      st.session_state.bt_trades),
            use_container_width=True)

    # ── Candle Chart with signals ──
    df_chart = st.session_state.bt_chart_data
    if df_chart is not None:
        df_ind = Indicators.add_all(df_chart)
        sigs   = st.session_state.get("bt_signals")
        fig    = ChartBuilder.candle_chart(
            df_ind, st.session_state.bt_trades, sigs,
            cfg["ind_config"],
            title=f"{cfg['ticker_sym']} · {cfg['timeframe']} · {cfg['strategy']}")
        st.plotly_chart(fig, use_container_width=True)

    # ── Trade Table ──
    if st.session_state.bt_trades:
        st.markdown('<div class="section-header" style="margin-top:20px">📋 TRADE LOG</div>',
                    unsafe_allow_html=True)
        df_trades = pd.DataFrame(st.session_state.bt_trades)
        df_trades["pnl_fmt"] = df_trades["pnl"].apply(
            lambda x: f"{'+'if x>=0 else ''}₹{x:,.2f}")

        st.dataframe(
            df_trades[["entry_time","exit_time","entry_price","exit_price",
                       "sl","target","pnl","reason","status","capital"]],
            use_container_width=True,
            column_config={
                "pnl"   : st.column_config.NumberColumn("P&L (₹)", format="%.2f"),
                "status": st.column_config.TextColumn("Result"),
            },
            hide_index=True,
        )

        # Download
        csv = df_trades.to_csv(index=False)
        st.download_button("⬇ Download CSV", csv,
                           f"bt_{cfg['ticker_sym']}_{datetime.now():%Y%m%d}.csv",
                           "text/csv", key="bt_dl")

# ═══════════════════════════════════════════════════════════════
# § 18 · LIVE TRADING TAB
# ═══════════════════════════════════════════════════════════════
def tab_live(cfg: dict):
    st.markdown("""
    <div style='padding:4px 0 16px'>
      <span style='font-family:JetBrains Mono;font-size:16px;font-weight:700;color:#00d4aa'>
        ◈ LIVE TRADING
      </span>
      <span style='font-size:11px;color:#475569;font-family:JetBrains Mono;margin-left:12px'>
        Dhan API · Tick-by-Tick · Auto-Refresh every 1.5s
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Config display ──
    with st.expander("⚙️ Active Configuration", expanded=False):
        cc1,cc2,cc3,cc4 = st.columns(4)
        cc1.info(f"**Ticker:** {cfg['ticker_sym']}")
        cc2.info(f"**TF:** {cfg['timeframe']}  **Period:** {cfg['period']}")
        cc3.info(f"**Strategy:** {cfg['strategy']}")
        cc4.info(f"**Qty:** {cfg['qty']}  **SL:** {cfg['sl_type']}")

    # ── Sector Scanner ──
    with st.expander("🔭 Sector Momentum Scanner (NSE ORB Setup)", expanded=False):
        sel_sectors = st.multiselect("Select Sectors to Scan",
                                     list(NSE_SECTORS.keys()),
                                     default=["🏦 Banking","💻 IT/Tech"],
                                     key="lt_sectors")
        if st.button("🔍 Scan Now", key="lt_scan"):
            with st.spinner("Scanning sectors (1.5s per ticker)…"):
                hits = SectorScanner.scan(sel_sectors)
            if hits:
                st.dataframe(pd.DataFrame(hits), use_container_width=True, hide_index=True)
                st.markdown('<div class="info-box">✅ Stocks above show 1–1.5% momentum. '
                            'Select one and start live trading.</div>',
                            unsafe_allow_html=True)
            else:
                st.warning("No stocks found in 1–1.5% momentum range at this time.")

    st.divider()

    # ── Start / Stop ──
    col_st, col_sp, col_sq, _ = st.columns([1,1,1,3])
    if col_st.button("▶  START", key="lt_start", type="primary",
                     disabled=st.session_state.live_active):
        st.session_state.live_active    = True
        st.session_state.partial_closed = False
        st.session_state.half_closed    = False
        st.session_state.trailing_high  = 0.0
        st.session_state.live_ticker    = cfg["ticker_sym"]
        st.session_state.live_log.clear()
        st.rerun()

    if col_sp.button("⏹  STOP",  key="lt_stop",
                     disabled=not st.session_state.live_active):
        st.session_state.live_active = False
        st.rerun()

    if col_sq.button("🔪 SQUARE OFF", key="lt_sq",
                     disabled=st.session_state.live_position is None):
        pos = st.session_state.live_position
        if pos:
            ltp = DataFetcher.get_ltp(st.session_state.live_ticker)
            pnl = (ltp - pos["entry"]) * pos["qty"]
            _record_closed_trade(pos, ltp, "Manual Square Off", pnl)
            if st.session_state.dhan_client_id:
                api = DhanAPI(st.session_state.dhan_client_id,
                              st.session_state.dhan_token)
                api.place_order(
                    trading_symbol  = pos.get("symbol",""),
                    security_id     = st.session_state.dhan_security_id,
                    qty             = pos["qty"],
                    order_type      = "MARKET",
                    exchange        = st.session_state.dhan_exchange,
                    product         = st.session_state.dhan_product,
                )
        st.session_state.live_position  = None
        st.session_state.partial_closed = False
        st.session_state.half_closed    = False
        st.rerun()

    # ── STATUS INDICATOR ──
    if st.session_state.live_active:
        st.markdown(
            '<span class="live-dot"></span>'
            '<span style="font-family:JetBrains Mono;font-size:12px;'
            'color:#10b981">LIVE  ·  Auto-refresh 1.5 s</span>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<span style="font-size:11px;color:#475569;'
            'font-family:JetBrains Mono">⏸  STOPPED</span>',
            unsafe_allow_html=True)

    # ── Dynamic section (use fragment if available, else placeholder) ──
    _render_live_dynamic(cfg)

    # ── Activity Log ──
    if st.session_state.live_log:
        with st.expander("📟 Activity Log", expanded=False):
            for msg in reversed(list(st.session_state.live_log)):
                st.caption(msg)


def _render_live_dynamic(cfg: dict):
    """Inner function — called inside fragment or directly."""

    # Refresh data if live is active
    if st.session_state.live_active:
        ticker  = st.session_state.live_ticker or cfg["ticker_sym"]
        df_live = DataFetcher.fetch(ticker, cfg["timeframe"], cfg["period"])

        if df_live is not None and not df_live.empty:
            df_live  = Indicators.add_all(df_live)
            ltp      = float(df_live["Close"].iloc[-1])
            st.session_state.live_ltp = ltp

            _process_live_signals(df_live, ltp, cfg)
        else:
            ltp = st.session_state.live_ltp
    else:
        ltp = st.session_state.live_ltp
        df_live = None

    pos = st.session_state.live_position

    # ── Metrics row ──
    st.markdown('<div class="section-header" style="margin-top:10px">📡 LIVE METRICS</div>',
                unsafe_allow_html=True)

    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("LTP",    f"₹{ltp:,.2f}" if ltp else "—")

    if pos:
        entry   = pos["entry"]
        sl_live = pos["sl"]
        tgt     = pos["target"]
        pnl_live= (ltp - entry) * pos["qty"]
        m2.metric("Entry",   f"₹{entry:,.2f}")
        m3.metric("SL",      f"₹{sl_live:,.2f}",
                  delta=f"{(ltp-sl_live):+.2f}", delta_color="normal")
        m4.metric("Target",  f"₹{tgt:,.2f}",
                  delta=f"{(tgt-ltp):+.2f}", delta_color="normal")
        m5.metric("P & L",   f"₹{pnl_live:,.2f}",
                  delta=f"{pnl_live:+.2f}",
                  delta_color="normal" if pnl_live >= 0 else "inverse")
        m6.metric("Qty",     pos["qty"])

        # Trailing target display
        if cfg["target_type"] == "Trailing Target (Display Only)":
            params = dict(sl_pts=cfg["sl_pts"], target_pts=cfg["target_pts"],
                          rr=cfg["rr"], sl_atr_mult=cfg["sl_atr_mult"],
                          target_atr_mult=cfg["tgt_atr_mult"])
            if df_live is not None:
                trail_tgt = SLManager.compute_target(
                    ltp, sl_live, "Trailing Target (Display Only)",
                    params, df_live, -1)
            else:
                trail_tgt = pos.get("trail_target", tgt)
            pos["trail_target"] = trail_tgt
            st.markdown(
                f'<div class="warn-box">📊 <b>Trailing Target (Display)</b>: '
                f'₹{trail_tgt:,.2f}  — This level tracks price but does NOT trigger exit.</div>',
                unsafe_allow_html=True)
    else:
        for col, label in zip([m2,m3,m4,m5,m6],
                              ["Entry","SL","Target","P&L","Qty"]):
            col.metric(label, "—")

    # ── Position detail ──
    if pos:
        st.markdown('<div class="section-header" style="margin-top:14px">🟢 OPEN POSITION</div>',
                    unsafe_allow_html=True)
        pc1,pc2,pc3,pc4 = st.columns(4)
        pc1.markdown(f"""<div class="metric-card">
          <div class="label">ENTRY PRICE</div>
          <div class="val accent">₹{pos['entry']:,.2f}</div></div>""",
          unsafe_allow_html=True)
        pc2.markdown(f"""<div class="metric-card">
          <div class="label">STOP LOSS</div>
          <div class="val red">₹{pos['sl']:,.2f}</div></div>""",
          unsafe_allow_html=True)
        pc3.markdown(f"""<div class="metric-card">
          <div class="label">TARGET</div>
          <div class="val green">₹{pos['target']:,.2f}</div></div>""",
          unsafe_allow_html=True)
        pc4.markdown(f"""<div class="metric-card">
          <div class="label">UNREALISED P&L</div>
          <div class="val {'green' if (ltp-pos['entry'])*pos['qty']>=0 else 'red'}">
          ₹{(ltp-pos['entry'])*pos['qty']:,.2f}</div></div>""",
          unsafe_allow_html=True)

        if st.session_state.get("partial_closed"):
            st.success("✅ Partial close (½ qty) executed at 1:1. SL trailed to CTC (breakeven).")

    # ── Live chart ──
    if df_live is not None:
        sigs = st.session_state.get("live_signals", pd.Series(dtype=str))
        if not isinstance(sigs, pd.Series):
            sigs = None
        fig = ChartBuilder.candle_chart(
            df_live, [], sigs, cfg["ind_config"],
            title=f"LIVE · {st.session_state.live_ticker} · {cfg['timeframe']}")
        st.plotly_chart(fig, use_container_width=True)

    # ── Trigger next refresh ──
    if st.session_state.live_active:
        time.sleep(1.5)
        st.rerun()


def _process_live_signals(df: pd.DataFrame, ltp: float, cfg: dict):
    """Core live signal + SL/target logic executed each refresh."""
    pos    = st.session_state.live_position
    params = dict(
        sl_pts       = cfg["sl_pts"],
        target_pts   = cfg["target_pts"],
        rr           = cfg["rr"],
        sl_atr_mult  = cfg["sl_atr_mult"],
        target_atr_mult = cfg["tgt_atr_mult"],
    )
    now = datetime.now()
    ticker = st.session_state.live_ticker

    # ── Time gates (NSE session) ──
    market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    no_entry     = now.replace(hour=10, minute=0,  second=0, microsecond=0)
    force_exit   = now.replace(hour=14, minute=30, second=0, microsecond=0)

    # Force exit at 14:30
    if pos and now >= force_exit:
        pnl = (ltp - pos["entry"]) * pos["qty"]
        _record_closed_trade(pos, ltp, "Force Exit 14:30", pnl)
        st.session_state.live_position  = None
        st.session_state.partial_closed = False
        st.session_state.live_log.append(f"[{now:%H:%M:%S}] Force exit 14:30 · LTP={ltp:.2f}")
        return

    # ── Manage open position ──
    if pos:
        sl  = pos["sl"]
        tgt = pos["target"]
        t1  = pos.get("target1", pos["entry"] + (pos["entry"] - sl))

        # Trail SL
        new_sl = SLManager.trail_sl(sl, cfg["sl_type"], ltp,
                                    pos["entry"], params, df, -1,
                                    st.session_state.trailing_high)
        if new_sl > sl:
            pos["sl"] = new_sl
            st.session_state.live_position = pos
            st.session_state.live_log.append(
                f"[{now:%H:%M:%S}] SL trailed → ₹{new_sl:.2f}")

        st.session_state.trailing_high = max(
            st.session_state.trailing_high, ltp)

        # Partial close at 1:1
        if not st.session_state.partial_closed and ltp >= t1:
            st.session_state.partial_closed = True
            pos["sl"] = pos["entry"]   # CTC = breakeven
            pos["qty"] = max(1, pos["qty"] // 2)
            st.session_state.live_position = pos
            st.session_state.live_log.append(
                f"[{now:%H:%M:%S}] ½ qty closed at 1:1 (₹{t1:.2f}), SL → CTC")
            # Dhan partial exit
            if st.session_state.dhan_client_id:
                api = DhanAPI(st.session_state.dhan_client_id,
                              st.session_state.dhan_token)
                api.place_order(
                    trading_symbol  = pos.get("symbol",""),
                    security_id     = st.session_state.dhan_security_id,
                    qty             = max(1, cfg["qty"] // 2),
                    order_type      = "MARKET",
                    exchange        = st.session_state.dhan_exchange,
                    product         = st.session_state.dhan_product,
                )

        # SL hit (check LTP first in live)
        if ltp <= pos["sl"]:
            pnl = (pos["sl"] - pos["entry"]) * pos["qty"]
            _record_closed_trade(pos, pos["sl"], "SL Hit", pnl)
            st.session_state.live_position  = None
            st.session_state.partial_closed = False
            st.session_state.live_log.append(
                f"[{now:%H:%M:%S}] ❌ SL HIT @ ₹{pos['sl']:.2f}  PnL={pnl:+.2f}")
            if st.session_state.dhan_client_id:
                api = DhanAPI(st.session_state.dhan_client_id,
                              st.session_state.dhan_token)
                api.place_order(
                    trading_symbol = pos.get("symbol",""),
                    security_id    = st.session_state.dhan_security_id,
                    qty            = pos["qty"], order_type="MARKET",
                    exchange       = st.session_state.dhan_exchange,
                    product        = st.session_state.dhan_product,
                )
            return

        # Target hit
        if ltp >= pos["target"]:
            pnl = (pos["target"] - pos["entry"]) * pos["qty"]
            _record_closed_trade(pos, pos["target"], "Target Hit", pnl)
            st.session_state.live_position  = None
            st.session_state.partial_closed = False
            st.session_state.live_log.append(
                f"[{now:%H:%M:%S}] ✅ TARGET HIT @ ₹{pos['target']:.2f}  PnL={pnl:+.2f}")
            if st.session_state.dhan_client_id:
                api = DhanAPI(st.session_state.dhan_client_id,
                              st.session_state.dhan_token)
                api.place_order(
                    trading_symbol = pos.get("symbol",""),
                    security_id    = st.session_state.dhan_security_id,
                    qty            = pos["qty"], order_type="MARKET",
                    exchange       = st.session_state.dhan_exchange,
                    product        = st.session_state.dhan_product,
                )
            return

        return   # still in trade, nothing to do

    # ── Entry signal ──
    if pos is None and now < no_entry:
        if cfg["strategy"] == "ORB (Opening Range Breakout)":
            signals = ORBStrategy.generate_signals(df, cfg["ind_config"], for_backtest=False)
        elif cfg["strategy"] == "Simple Buy":
            signals = SimpleStrategy.simple_buy(df, cfg["ind_config"])
        else:
            signals = pd.Series("", index=df.index)

        st.session_state.live_signals = signals
        latest_sig = signals.iloc[-1] if len(signals) else ""

        if latest_sig == "BUY":
            entry_price = ltp   # live: enter at current LTP (not next candle)
            sl   = SLManager.compute_sl(entry_price, cfg["sl_type"], params, df, -1)
            tgt  = SLManager.compute_target(entry_price, sl,
                                            cfg["target_type"], params, df, -1)
            t1   = entry_price + (entry_price - sl)  # 1:1 partial

            st.session_state.live_position = dict(
                symbol  = ticker,
                entry   = entry_price,
                sl      = sl,
                target  = tgt,
                target1 = t1,
                qty     = cfg["qty"],
                time    = str(now)[:19],
            )
            st.session_state.trailing_high = entry_price
            st.session_state.live_log.append(
                f"[{now:%H:%M:%S}] 🟢 BUY @ ₹{entry_price:.2f}  SL=₹{sl:.2f}  TGT=₹{tgt:.2f}")

            # ── Dhan order ──
            if st.session_state.dhan_client_id:
                api = DhanAPI(st.session_state.dhan_client_id,
                              st.session_state.dhan_token)
                resp = api.place_order(
                    trading_symbol = ticker.replace(".NS","").replace(".BO",""),
                    security_id    = st.session_state.dhan_security_id,
                    qty            = cfg["qty"], order_type="MARKET",
                    exchange       = st.session_state.dhan_exchange,
                    product        = st.session_state.dhan_product,
                )
                order_id = resp.get("orderId","")
                st.session_state.live_log.append(
                    f"[{now:%H:%M:%S}] Dhan Order → {order_id or resp}")


def _record_closed_trade(pos: dict, exit_price: float,
                          reason: str, pnl: float):
    """Save closed trade to trade history."""
    trade = dict(
        ticker      = pos.get("symbol",""),
        entry_time  = pos.get("time",""),
        exit_time   = str(datetime.now())[:19],
        entry_price = round(pos["entry"], 4),
        exit_price  = round(exit_price,   4),
        qty         = pos.get("qty", 1),
        sl          = round(pos.get("sl",0), 4),
        target      = round(pos.get("target",0), 4),
        pnl         = round(pnl, 2),
        reason      = reason,
        status      = "WIN" if pnl >= 0 else "LOSS",
    )
    st.session_state.trade_history.append(trade)

# ═══════════════════════════════════════════════════════════════
# § 19 · TRADE HISTORY TAB
# ═══════════════════════════════════════════════════════════════
def tab_history():
    st.markdown("""
    <div style='padding:4px 0 16px'>
      <span style='font-family:JetBrains Mono;font-size:16px;font-weight:700;color:#00d4aa'>
        ◈ TRADE HISTORY
      </span>
      <span style='font-size:11px;color:#475569;font-family:JetBrains Mono;margin-left:12px'>
        Live session trades  ·  In-memory  ·  No database
      </span>
    </div>
    """, unsafe_allow_html=True)

    trades = st.session_state.trade_history

    col_clr, _ = st.columns([1,5])
    if col_clr.button("🗑  Clear History", key="hist_clr"):
        st.session_state.trade_history = []
        st.rerun()

    if not trades:
        st.markdown("""
        <div class="info-box">
          No live trades yet. Execute trades from the <b>Live Trading</b> tab.
        </div>
        """, unsafe_allow_html=True)
        return

    df_h = pd.DataFrame(trades)
    total_pnl = df_h["pnl"].sum()
    wins      = (df_h["pnl"] >= 0).sum()
    losses    = (df_h["pnl"] <  0).sum()
    win_rate  = round(wins / len(df_h) * 100, 1)

    # Summary
    c1,c2,c3,c4,c5 = st.columns(5)
    def hmc(col, label, val, clr=""):
        col.markdown(f"""<div class="metric-card">
          <div class="label">{label}</div>
          <div class="val {clr}">{val}</div></div>""",
          unsafe_allow_html=True)

    hmc(c1,"Total Trades", len(df_h))
    hmc(c2,"Wins",   wins,   "green")
    hmc(c3,"Losses", losses, "red")
    hmc(c4,"Win Rate", f"{win_rate}%",
       "green" if win_rate>=50 else "red")
    hmc(c5,"Total P&L", f"₹{total_pnl:,.2f}",
       "green" if total_pnl>=0 else "red")

    # Mini equity curve
    running_cap = [0.0]
    for t in trades:
        running_cap.append(running_cap[-1] + t["pnl"])
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        y=running_cap, mode="lines+markers",
        line=dict(color="#00d4aa",width=2),
        fill="tozeroy", fillcolor="#00d4aa11",
    ))
    fig_eq.add_hline(y=0,line=dict(color="#94a3b8",width=1,dash="dot"))
    fig_eq.update_layout(
        title=dict(text="Cumulative P&L",
                   font=dict(family="JetBrains Mono",size=12,color="#e2e8f0")),
        paper_bgcolor="#0a0e17",plot_bgcolor="#0a0e17",
        font=dict(color="#94a3b8",family="JetBrains Mono",size=10),
        margin=dict(l=40,r=10,t=35,b=20),height=220,
    )
    fig_eq.update_xaxes(showgrid=True,gridcolor="#1e2d45")
    fig_eq.update_yaxes(showgrid=True,gridcolor="#1e2d45",tickprefix="₹")
    st.plotly_chart(fig_eq, use_container_width=True)

    # Trade rows
    st.markdown('<div class="section-header">📋 ALL TRADES</div>',
                unsafe_allow_html=True)

    for t in reversed(trades):
        clr  = "#10b98133" if t["status"]=="WIN" else "#ef444422"
        brd  = "#10b981"   if t["status"]=="WIN" else "#ef4444"
        badge_cls = "signal-buy" if t["status"]=="WIN" else "signal-sell"
        st.markdown(f"""
        <div class="trade-row" style="border-left:3px solid {brd}">
          <span class="signal-badge {badge_cls}">{t['status']}</span>
          <span style='margin-left:10px;color:#94a3b8'>{t.get('ticker','')}</span>
          <span style='margin-left:10px;color:#e2e8f0;font-weight:600'>
            ₹{t['pnl']:+,.2f}</span>
          <span style='margin-left:14px;color:#64748b'>
            Entry ₹{t['entry_price']:,.2f} → Exit ₹{t['exit_price']:,.2f}
            · Qty {t.get('qty',1)} · {t.get('reason','')}
          </span>
          <span style='float:right;color:#475569;font-size:10px'>
            {t.get('exit_time','')}</span>
        </div>
        """, unsafe_allow_html=True)

    # Download
    csv = df_h.to_csv(index=False)
    st.download_button(
        "⬇ Download Trade History",
        csv,
        f"trades_{datetime.now():%Y%m%d_%H%M}.csv",
        "text/csv", key="hist_dl")

# ═══════════════════════════════════════════════════════════════
# § 20 · MAIN APP ENTRY
# ═══════════════════════════════════════════════════════════════
def main():
    cfg = render_sidebar()

    st.markdown("""
    <div style='padding:10px 0 8px'>
      <span style='font-family:JetBrains Mono;font-size:22px;font-weight:700;
                   background:linear-gradient(90deg,#00d4aa,#3b82f6);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        ⚡ PROFESSIONAL TRADING PLATFORM
      </span>
      <span style='font-size:10px;color:#334155;font-family:JetBrains Mono;
                   vertical-align:middle;margin-left:14px;letter-spacing:.15em'>
        BACKTEST · LIVE · HISTORY
      </span>
    </div>
    """, unsafe_allow_html=True)

    if not TA_OK:
        st.warning("⚠️  `ta` library not found. Run: `pip install ta`  "
                   "— Indicators will be unavailable until installed.")

    tab1, tab2, tab3 = st.tabs([
        "📊  BACKTESTING",
        "⚡  LIVE TRADING",
        "📋  TRADE HISTORY",
    ])

    with tab1:
        tab_backtest(cfg)

    with tab2:
        tab_live(cfg)

    with tab3:
        tab_history()


if __name__ == "__main__":
    main()
