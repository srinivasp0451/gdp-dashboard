# =========================================================
# INSTITUTIONAL GRADE LIVE ALGO ENGINE
# =========================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime
import pytz
from scipy.signal import argrelextrema
from scipy.stats import zscore

IST = pytz.timezone("Asia/Kolkata")
REFRESH_SEC = 1.7

st.set_page_config("Pro Algo Trading Engine", layout="wide")

# ===================== UTIL =====================
def to_ist(df):
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.tz_convert(IST)

def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(n).mean() / loss.rolling(n).mean()
    return 100 - (100 / (1 + rs))

# ===================== STRATEGIES =====================
class StrategyEngine:

    # ---------- EMA CROSSOVER ----------
    def ema_crossover(self, df):
        df["EMA9"] = ema(df.Close, 9)
        df["EMA20"] = ema(df.Close, 20)

        prev, curr = df.iloc[-2], df.iloc[-1]

        if prev.EMA9 < prev.EMA20 and curr.EMA9 > curr.EMA20:
            return self._build(
                "LONG",
                "EMA9 crossed above EMA20 → bullish momentum",
                "Fast EMA crossing slow EMA",
                "SL below recent swing low",
                "Risk:Reward 1:2",
                {"EMA9": curr.EMA9, "EMA20": curr.EMA20},
                {"ema": [(9, df.EMA9), (20, df.EMA20)]}
            )

        if prev.EMA9 > prev.EMA20 and curr.EMA9 < curr.EMA20:
            return self._build(
                "SHORT",
                "EMA9 crossed below EMA20 → bearish momentum",
                "Fast EMA breakdown",
                "SL above recent swing high",
                "Risk:Reward 1:2",
                {"EMA9": curr.EMA9, "EMA20": curr.EMA20},
                {"ema": [(9, df.EMA9), (20, df.EMA20)]}
            )

        return self._none()

    # ---------- ELLIOTT WAVES ----------
    def elliott_wave(self, df):
        closes = df.Close.values
        highs = argrelextrema(closes, np.greater, order=5)[0]
        lows = argrelextrema(closes, np.less, order=5)[0]

        waves = sorted(list(highs) + list(lows))
        if len(waves) >= 5:
            return self._build(
                "LONG",
                "5-wave structure detected → impulse phase",
                "Wave 4 correction likely complete",
                "Below Wave 4 low",
                "Wave 5 projection",
                {"waves": len(waves)},
                {"elliott": waves[-5:]}
            )
        return self._none()

    # ---------- DEMAND SUPPLY + EMA PULLBACK ----------
    def demand_supply(self, df):
        df["EMA50"] = ema(df.Close, 50)
        price = df.Close.iloc[-1]

        if price > df.EMA50.iloc[-1] and df.Low.iloc[-1] <= df.EMA50.iloc[-1]:
            return self._build(
                "LONG",
                "Price pulled back into demand zone + EMA50",
                "Trend continuation setup",
                "Below demand zone",
                "Previous high",
                {"EMA50": df.EMA50.iloc[-1]},
                {"ema": [(50, df.EMA50)]}
            )
        return self._none()

    # ---------- FIBONACCI ----------
    def fibonacci(self, df):
        high = df.High.max()
        low = df.Low.min()
        fib_618 = high - 0.618 * (high - low)
        price = df.Close.iloc[-1]

        if abs(price - fib_618) / price < 0.002:
            return self._build(
                "LONG",
                "Price reacting at 61.8% Fibonacci retracement",
                "Golden ratio support",
                "Below Fib 78.6%",
                "Previous high",
                {"Fib 61.8": fib_618},
                {"fib": [low, fib_618, high]}
            )
        return self._none()

    # ---------- Z-SCORE ----------
    def zscore_reversion(self, df):
        z = zscore(df.Close)[-1]
        if z < -2:
            return self._build(
                "LONG",
                f"Z-Score {z:.2f} → extreme oversold",
                "Mean reversion expectation",
                "Below recent low",
                "VWAP / mean",
                {"Z": z},
                {}
            )
        if z > 2:
            return self._build(
                "SHORT",
                f"Z-Score {z:.2f} → extreme overbought",
                "Mean reversion expectation",
                "Above recent high",
                "VWAP / mean",
                {"Z": z},
                {}
            )
        return self._none()

    # ---------- HELPERS ----------
    def _build(self, direction, reason, entry, sl, target, indicators, draw):
        return {
            "signal": True,
            "direction": direction,
            "reason": reason,
            "entry_logic": entry,
            "sl_logic": sl,
            "target_logic": target,
            "indicators": indicators,
            "draw": draw
        }

    def _none(self):
        return {
            "signal": False,
            "direction": "NONE",
            "reason": "No valid setup",
            "entry_logic": "",
            "sl_logic": "",
            "target_logic": "",
            "indicators": {},
            "draw": {}
        }

# ===================== SESSION =====================
if "active" not in st.session_state:
    st.session_state.update({
        "active": False,
        "position": None,
        "iteration": 0
    })

engine = StrategyEngine()

# ===================== UI =====================
st.sidebar.title("⚙ Strategy Control")
ticker = st.sidebar.text_input("Ticker", "^NSEI")
strategy = st.sidebar.selectbox(
    "Strategy",
    ["EMA Crossover", "Elliott Waves", "Demand Supply", "Fibonacci", "Z-Score"]
)
interval = st.sidebar.selectbox("Interval", ["1m","5m","15m"])
period = st.sidebar.selectbox("Period", ["1d","5d","1mo"])

if st.sidebar.button("▶ START"): st.session_state.active = True
if st.sidebar.button("⛔ STOP"): st.session_state.active = False

# ===================== LIVE =====================
if st.session_state.active:
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    df = to_ist(flatten(df))

    strat_map = {
        "EMA Crossover": engine.ema_crossover,
        "Elliott Waves": engine.elliott_wave,
        "Demand Supply": engine.demand_supply,
        "Fibonacci": engine.fibonacci,
        "Z-Score": engine.zscore_reversion
    }

    result = strat_map[strategy](df)

    st.subheader("📌 Strategy Details")
    st.json(result)

    fig = go.Figure(go.Candlestick(
        x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close
    ))

    # ---- DRAW OVERLAYS ----
    if "ema" in result["draw"]:
        for n, s in result["draw"]["ema"]:
            fig.add_trace(go.Scatter(x=df.index, y=s, name=f"EMA {n}"))

    if "elliott" in result["draw"]:
        idx = result["draw"]["elliott"]
        fig.add_trace(go.Scatter(
            x=df.index[idx],
            y=df.Close.iloc[idx],
            mode="lines+markers",
            name="Elliott Waves",
            line=dict(width=3)
        ))

    if "fib" in result["draw"]:
        for lvl in result["draw"]["fib"]:
            fig.add_hline(y=lvl, line_dash="dot")

    st.plotly_chart(fig, use_container_width=True)

    st.session_state.iteration += 1
    time.sleep(REFRESH_SEC)
    st.rerun()
