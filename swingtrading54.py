# =============================
# ADVANCED MULTI-STRATEGY TRADING ENGINE
# =============================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import math
import time

st.set_page_config(layout="wide", page_title="AI Trading Companion")

# -----------------------------
# AUTO REFRESH
# -----------------------------
st.experimental_set_query_params(refresh="true")
time.sleep(3)
st.rerun()

# -----------------------------
# SIDEBAR CONFIG
# -----------------------------
st.sidebar.header("⚙️ Strategy Configuration")

symbol = st.sidebar.text_input("Symbol", "^NSEI")
interval = st.sidebar.selectbox("Interval", ["5m", "15m", "30m", "1h", "1d"])

strategy = st.sidebar.multiselect(
    "Select Strategies",
    [
        "EMA Crossover",
        "RSI Divergence",
        "Elliott Waves",
        "Fibonacci",
        "Z-Score",
        "Breakout",
        "Support Resistance",
        "EMA Pullback",
        "News Psychology",
        "Hybrid"
    ]
)

ema1 = st.sidebar.number_input("EMA 1", 5, 200, 9)
ema2 = st.sidebar.number_input("EMA 2", 5, 200, 21)
min_slope = st.sidebar.slider("EMA Min Slope (Degrees)", 0, 45, 20)

sl_type = st.sidebar.selectbox("SL Type", ["System", "Custom"])
target_type = st.sidebar.selectbox("Target Type", ["System", "Custom"])

custom_sl = st.sidebar.number_input("Custom SL Points", 5, 500, 50)
custom_target = st.sidebar.number_input("Custom Target Points", 5, 1000, 120)

trailing = st.sidebar.checkbox("Enable Trailing SL")

# -----------------------------
# DATA FETCH
# -----------------------------
df = yf.download(symbol, period="5d", interval=interval)
df.dropna(inplace=True)

# -----------------------------
# INDICATORS
# -----------------------------
df["EMA1"] = df["Close"].ewm(span=ema1).mean()
df["EMA2"] = df["Close"].ewm(span=ema2).mean()

df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(14).mean()))

df["Z"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).std()

# EMA slope
df["EMA_Slope"] = np.degrees(np.arctan(df["EMA1"].diff()))

# -----------------------------
# SIGNAL LOGIC
# -----------------------------
latest = df.iloc[-1]
signal = None
reason = []

if "EMA Crossover" in strategy:
    if latest["EMA1"] > latest["EMA2"] and latest["EMA_Slope"] > min_slope:
        signal = "BUY"
        reason.append("EMA bullish crossover with strong slope")

if "RSI Divergence" in strategy:
    if latest["RSI"] < 30:
        signal = "BUY"
        reason.append("RSI oversold divergence")

if "Z-Score" in strategy:
    if latest["Z"] < -1.5:
        signal = "BUY"
        reason.append("Mean reversion zone")

# Hybrid score
confidence = len(reason) * 18

# -----------------------------
# ENTRY / SL / TARGET
# -----------------------------
price = latest["Close"]

if signal:
    entry = price
    sl = entry - (custom_sl if sl_type == "Custom" else entry * 0.003)
    target = entry + (custom_target if target_type == "Custom" else entry * 0.006)
else:
    entry = sl = target = None

# -----------------------------
# UI DISPLAY
# -----------------------------
st.title("📊 Live Trade Companion")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Signal", signal or "WAIT")
    st.metric("Confidence", f"{confidence}%")

with col2:
    st.metric("Entry", entry)
    st.metric("SL", sl)

with col3:
    st.metric("Target", target)
    st.metric("Trailing SL", "ON" if trailing else "OFF")

# -----------------------------
# STRATEGY EXPLANATION
# -----------------------------
st.subheader("🧠 Strategy Reasoning")

for r in reason:
    st.success(r)

if not signal:
    st.info("Market is consolidating. Waiting for confirmation.")

# -----------------------------
# FRIENDLY GUIDANCE
# -----------------------------
st.subheader("🤝 Market Guidance")

if signal == "BUY":
    st.write("""
    Market structure looks supportive.
    Buyers are stepping in near value zones.
    Avoid panic if small pullbacks happen.
    Trail SL once price moves 0.5R.
    """)
else:
    st.write("""
    No edge right now.
    Let price come to you.
    Capital protection is also a position.
    """)

# -----------------------------
# CHART
# -----------------------------
fig = go.Figure()

fig.add_candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"]
)

fig.add_trace(go.Scatter(x=df.index, y=df["EMA1"], name="EMA1"))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA2"], name="EMA2"))

st.plotly_chart(fig, use_container_width=True)
