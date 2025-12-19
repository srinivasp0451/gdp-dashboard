# ============================================================
# PRO LIVE ALGO TRADING ENGINE – EXECUTION GRADE
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from scipy.signal import argrelextrema
from datetime import datetime
import pytz

# ------------------ CONFIG ------------------
IST = pytz.timezone("Asia/Kolkata")
REFRESH_SEC = 1.7

st.set_page_config("Institutional Algo Engine", layout="wide")

# ------------------ UTILITIES ------------------
def to_ist(df):
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.tz_convert(IST)

def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def atr(df, n=14):
    tr = np.maximum(
        df.High - df.Low,
        np.maximum(abs(df.High - df.Close.shift()), abs(df.Low - df.Close.shift()))
    )
    return tr.rolling(n).mean()

# ------------------ SESSION ------------------
if "position" not in st.session_state:
    st.session_state.position = None
if "active" not in st.session_state:
    st.session_state.active = False
if "iteration" not in st.session_state:
    st.session_state.iteration = 0

# ------------------ STRATEGY ------------------
def ema_crossover_strategy(df, ema1, ema2, candle_mode, candle_value):
    df["EMA1"] = ema(df.Close, ema1)
    df["EMA2"] = ema(df.Close, ema2)
    df["ATR"] = atr(df)

    prev, curr = df.iloc[-2], df.iloc[-1]
    candle_size = abs(curr.Close - curr.Open)

    valid_candle = True
    if candle_mode == "Custom":
        valid_candle = candle_size >= candle_value
    elif candle_mode == "System":
        valid_candle = candle_size >= 1.2 * curr.ATR

    if prev.EMA1 < prev.EMA2 and curr.EMA1 > curr.EMA2 and valid_candle:
        return {
            "signal": "LONG",
            "reason": "EMA crossover with bullish momentum",
            "entry": curr.Close,
            "sl": curr.Low,
            "target": curr.Close + (curr.Close - curr.Low) * 2,
            "indicators": {
                "EMA1": curr.EMA1,
                "EMA2": curr.EMA2,
                "ATR": curr.ATR,
                "Candle Size": candle_size
            }
        }

    if prev.EMA1 > prev.EMA2 and curr.EMA1 < curr.EMA2 and valid_candle:
        return {
            "signal": "SHORT",
            "reason": "EMA breakdown with bearish momentum",
            "entry": curr.Close,
            "sl": curr.High,
            "target": curr.Close - (curr.High - curr.Close) * 2,
            "indicators": {
                "EMA1": curr.EMA1,
                "EMA2": curr.EMA2,
                "ATR": curr.ATR,
                "Candle Size": candle_size
            }
        }

    return {"signal": "NONE"}

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙ Strategy Control")

ticker = st.sidebar.text_input("Ticker", "^NSEI")
interval = st.sidebar.selectbox("Interval", ["1m","5m","15m"])
period = st.sidebar.selectbox("Period", ["1d","5d","1mo"])

st.sidebar.subheader("EMA Settings")
ema1 = st.sidebar.number_input("EMA 1", 5, 50, 9)
ema2 = st.sidebar.number_input("EMA 2", 10, 100, 20)

candle_mode = st.sidebar.selectbox(
    "Crossover Confirmation",
    ["None", "Custom", "System"]
)
candle_value = st.sidebar.number_input(
    "Custom Candle Size (points)", value=20.0
)

st.sidebar.subheader("Risk Management")

sl_type = st.sidebar.selectbox(
    "Stop Loss Type",
    ["System", "Custom Points", "Trailing"]
)
sl_points = st.sidebar.number_input("SL Points", value=20.0)

target_type = st.sidebar.selectbox(
    "Target Type",
    ["System", "Custom Points", "Trailing"]
)
target_points = st.sidebar.number_input("Target Points", value=40.0)

if st.sidebar.button("▶ START"):
    st.session_state.active = True

if st.sidebar.button("⛔ STOP"):
    st.session_state.active = False
    st.session_state.position = None

# ------------------ LIVE LOOP ------------------
if st.session_state.active:

    df = yf.download(ticker, interval=interval, period=period, progress=False)
    df = to_ist(flatten(df))

    result = ema_crossover_strategy(
        df, ema1, ema2, candle_mode, candle_value
    )

    price = df.Close.iloc[-1]

    st.title("📊 LIVE TRADE STATUS")

    if result["signal"] != "NONE" and st.session_state.position is None:
        st.session_state.position = result

    pos = st.session_state.position

    if pos:
        pnl = (price - pos["entry"]) if pos["signal"] == "LONG" else (pos["entry"] - price)

        st.success(f"🟢 {pos['signal']} POSITION ACTIVE")

        col1, col2, col3 = st.columns(3)
        col1.metric("Entry", round(pos["entry"],2))
        col2.metric("Current Price", round(price,2))
        col3.metric("P&L", round(pnl,2))

        st.subheader("📌 Strategy Reason")
        st.write(pos["reason"])

        st.subheader("📊 Indicator Snapshot")
        for k,v in pos["indicators"].items():
            st.write(f"**{k}:** {round(v,2)}")

        st.subheader("🧠 Live Trade Guidance")
        if pnl > 0:
            st.info(
                "Market is moving in your favor. "
                "Avoid early exit. Let trailing logic do its job. "
                "Do NOT move SL away."
            )
        else:
            st.warning(
                "Trade is under pressure. "
                "Watch reaction near EMA zone. "
                "Exit only if structure breaks."
            )

    # ------------------ CHART ------------------
    fig = go.Figure(go.Candlestick(
        x=df.index,
        open=df.Open,
        high=df.High,
        low=df.Low,
        close=df.Close
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df.EMA1, name="EMA 1"))
    fig.add_trace(go.Scatter(x=df.index, y=df.EMA2, name="EMA 2"))

    st.plotly_chart(fig, use_container_width=True)

    st.session_state.iteration += 1
    time.sleep(REFRESH_SEC)
    st.rerun()
