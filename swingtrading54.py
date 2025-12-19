# =========================================================
# PRO-LEVEL STREAMLIT ALGO TRADING ENGINE
# =========================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import pytz
from scipy.signal import argrelextrema
from datetime import datetime

# ================= CONFIG =================
IST = pytz.timezone("Asia/Kolkata")
REFRESH = 1.7

st.set_page_config("Institutional Algo Engine", layout="wide")

# ================= UTIL =================
def to_ist(df):
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.tz_convert(IST)

def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(n).mean() / loss.rolling(n).mean()
    return 100 - (100 / (1 + rs))

# ================= SESSION =================
if "trade" not in st.session_state:
    st.session_state.trade = None
if "history" not in st.session_state:
    st.session_state.history = []
if "active" not in st.session_state:
    st.session_state.active = False
if "iteration" not in st.session_state:
    st.session_state.iteration = 0

# ================= SIDEBAR =================
st.sidebar.title("⚙ Trading Control")

ticker = st.sidebar.text_input("Ticker", "^NSEI")
interval = st.sidebar.selectbox("Timeframe", ["1m","5m","15m"])
period = st.sidebar.selectbox("Period", ["1d","5d","1mo"])

ema_fast = st.sidebar.number_input("EMA Fast", 5, 50, 9)
ema_slow = st.sidebar.number_input("EMA Slow", 10, 200, 20)

if st.sidebar.button("▶ START"):
    st.session_state.active = True

if st.sidebar.button("⛔ STOP"):
    st.session_state.active = False

# ================= LIVE =================
if st.session_state.active:

    df = yf.download(ticker, interval=interval, period=period, progress=False)
    df = to_ist(flatten(df))

    # Indicators
    df["EMA_FAST"] = ema(df.Close, ema_fast)
    df["EMA_SLOW"] = ema(df.Close, ema_slow)
    df["RSI"] = rsi(df.Close)

    price = df.Close.iloc[-1]

    # ================= PSYCHOLOGY =================
    slope = df.EMA_FAST.iloc[-1] - df.EMA_FAST.iloc[-10]
    psychology = (
        "Strong institutional trend"
        if abs(slope) > df.Close.std() * 0.1
        else "Choppy / emotional market"
    )

    # ================= RSI DIVERGENCE =================
    rsi_signal = None
    if df.RSI.iloc[-1] < 35 and df.Close.iloc[-1] < df.Close.iloc[-5]:
        rsi_signal = "Bullish divergence forming"

    # ================= ELLIOTT WAVES =================
    swings = argrelextrema(df.Close.values, np.greater, order=5)[0]
    wave_signal = "No clear wave"
    if len(swings) >= 5:
        wave_signal = "Elliott impulse likely in progress"

    # ================= FIB =================
    high, low = df.High.max(), df.Low.min()
    fib618 = high - 0.618 * (high - low)
    fib_signal = (
        "Price reacting at 61.8% retracement"
        if abs(price - fib618)/price < 0.002
        else None
    )

    # ================= ENTRY LOGIC =================
    signal = None
    reason = []

    if df.EMA_FAST.iloc[-2] < df.EMA_SLOW.iloc[-2] and df.EMA_FAST.iloc[-1] > df.EMA_SLOW.iloc[-1]:
        signal = "LONG"
        reason.append("EMA bullish crossover")

    if rsi_signal:
        reason.append(rsi_signal)

    if fib_signal:
        reason.append(fib_signal)

    # ================= ENTER TRADE =================
    if signal and st.session_state.trade is None:
        st.session_state.trade = {
            "side": signal,
            "entry": price,
            "sl": price - 10 if signal=="LONG" else price + 10,
            "target": price + 20 if signal=="LONG" else price - 20,
            "reason": ", ".join(reason),
            "start": datetime.now(IST)
        }

    trade = st.session_state.trade

    # ================= GUIDANCE =================
    guidance = "Waiting for structured opportunity."

    if trade:
        pnl = price - trade["entry"] if trade["side"]=="LONG" else trade["entry"] - price
        if pnl > 0:
            guidance = (
                "Trade behaving well. Momentum intact. "
                "Do NOT exit early. Trail SL only."
            )
        else:
            guidance = (
                "Temporary adverse move. "
                "Watch structure — exit only if SL breaks."
            )

    # ================= UI =================
    st.markdown("## 📊 Live Trade Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Position", trade["side"] if trade else "Waiting")
    c2.metric("Price", f"{price:.2f}")
    c3.metric("SL", f"{trade['sl']:.2f}" if trade else "-")
    c4.metric("Target", f"{trade['target']:.2f}" if trade else "-")

    st.markdown("### 🧠 Market Psychology")
    st.success(psychology)

    st.markdown("### 🎯 Trade Reasoning")
    st.info(trade["reason"] if trade else "No active trade")

    st.markdown("### 🧭 Live Guidance (Friend Mode)")
    st.warning(guidance)

    # ================= CHART =================
    fig = go.Figure(go.Candlestick(
        x=df.index, open=df.Open, high=df.High,
        low=df.Low, close=df.Close
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df.EMA_FAST, name="EMA Fast"))
    fig.add_trace(go.Scatter(x=df.index, y=df.EMA_SLOW, name="EMA Slow"))

    if trade:
        fig.add_hline(y=trade["entry"], line_dash="dot")
        fig.add_hline(y=trade["sl"], line_color="red", line_dash="dot")
        fig.add_hline(y=trade["target"], line_color="green", line_dash="dot")

    st.plotly_chart(fig, use_container_width=True)

    st.session_state.iteration += 1
    time.sleep(REFRESH)
    st.rerun()
