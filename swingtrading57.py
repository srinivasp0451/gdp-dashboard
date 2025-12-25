import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time, random
from datetime import datetime
import pytz
import plotly.graph_objects as go

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(layout="wide")
IST = pytz.timezone("Asia/Kolkata")

# =========================================================
# SESSION STATE
# =========================================================
def ss(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

ss("data", None)
ss("live_on", False)
ss("live_pos", None)
ss("trade_history", [])
ss("logs", [])

# =========================================================
# LOGGING
# =========================================================
def log(msg):
    st.session_state.logs.append(
        f"{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    )

# =========================================================
# SAFE TIMEZONE HANDLING (FIXED)
# =========================================================
def ensure_ist(df):
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.tz_convert(IST)

# =========================================================
# UTILS
# =========================================================
def flatten_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def rate_limit():
    time.sleep(random.uniform(1.0, 1.5))

# =========================================================
# INDICATORS (TRADINGVIEW STYLE)
# =========================================================
def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h-l, abs(h-c.shift()), abs(l-c.shift())], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# =========================================================
# DATA FETCH
# =========================================================
def fetch_data(ticker, interval, period):
    rate_limit()
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    df = flatten_df(df)
    df.dropna(inplace=True)
    df = ensure_ist(df)
    return df

# =========================================================
# SIGNAL ENGINE (SINGLE SOURCE OF TRUTH)
# =========================================================
def ema_crossover_signal(df, ema_fast, ema_slow):
    ef = ema(df["Close"], ema_fast)
    es = ema(df["Close"], ema_slow)

    long = (ef > es) & (ef.shift() <= es.shift())
    short = (ef < es) & (ef.shift() >= es.shift())

    return np.where(long, 1, np.where(short, -1, 0))

# =========================================================
# TRAILING SL (TRUE TRAILING)
# =========================================================
def update_trailing_sl(pos, price, atr_val, method, points):
    d = pos["dir"]

    if method == "Points":
        candidate = price - d * points
    elif method == "ATR":
        candidate = price - d * atr_val
    elif method == "Candle":
        candidate = pos["last_low"] if d == 1 else pos["last_high"]
    else:
        return pos["sl"]

    # trail only forward
    if d == 1:
        return max(pos["sl"], candidate)
    else:
        return min(pos["sl"], candidate)

# =========================================================
# SIGNAL-BASED EXIT (CORE FIX)
# =========================================================
def signal_based_exit(pos, df, current_index, ema_fast, ema_slow):
    """
    Exit ONLY IF:
    - At least one candle AFTER entry
    - Reverse EMA crossover
    """
    if current_index <= pos["entry_index"]:
        return False

    sig = ema_crossover_signal(df, ema_fast, ema_slow)[-1]

    if pos["dir"] == 1 and sig == -1:
        return True
    if pos["dir"] == -1 and sig == 1:
        return True

    return False

# =========================================================
# BACKTEST (USES SAME LOGIC AS LIVE)
# =========================================================
def backtest(df, strategy):
    trades = []
    pos = None

    for i in range(30, len(df)):
        sub = df.iloc[:i+1]
        price = sub["Close"].iloc[-1]
        atr_val = atr(sub).iloc[-1]

        if strategy == "EMA Crossover":
            sig = ema_crossover_signal(sub, EMA_FAST, EMA_SLOW)[-1]
        elif strategy == "Simple Buy":
            sig = 1
        else:
            sig = -1

        if pos is None and sig != 0:
            pos = {
                "dir": sig,
                "entry": price,
                "sl": price - sig * SL_POINTS,
                "entry_index": i,
                "entry_time": sub.index[-1],
                "last_low": sub["Low"].iloc[-1],
                "last_high": sub["High"].iloc[-1],
            }

        elif pos:
            pos["last_low"] = sub["Low"].iloc[-1]
            pos["last_high"] = sub["High"].iloc[-1]

            pos["sl"] = update_trailing_sl(
                pos, price, atr_val, TRAIL_TYPE, SL_POINTS
            )

            if signal_based_exit(pos, sub, i, EMA_FAST, EMA_SLOW):
                pos["exit"] = price
                pos["pnl"] = (price - pos["entry"]) * pos["dir"]
                trades.append(pos)
                pos = None

            elif (pos["dir"] == 1 and price <= pos["sl"]) or \
                 (pos["dir"] == -1 and price >= pos["sl"]):
                pos["exit"] = price
                pos["pnl"] = (price - pos["entry"]) * pos["dir"]
                trades.append(pos)
                pos = None

    return pd.DataFrame(trades)

# =========================================================
# UI
# =========================================================
st.title("EMA Core Engine – Trailing + Signal-Based SL")

ticker = st.text_input("Ticker", "^NSEI")
interval = st.selectbox("Interval", ["1m","5m","15m","30m","1h"])
period = st.selectbox("Period", ["1d","5d","1mo"])

strategy = st.selectbox("Strategy", ["EMA Crossover", "Simple Buy", "Simple Sell"])

EMA_FAST = st.number_input("EMA Fast", 1, 50, 9)
EMA_SLOW = st.number_input("EMA Slow", 1, 100, 15)

TRAIL_TYPE = st.selectbox("Trailing SL Type", ["Points", "ATR", "Candle"])
SL_POINTS = st.number_input("SL / Trail Points", 1, 500, 10)

if st.button("Fetch Data"):
    st.session_state.data = fetch_data(ticker, interval, period)

# =========================================================
# MAIN
# =========================================================
if st.session_state.data is not None:
    df = st.session_state.data

    if st.button("Run Backtest"):
        res = backtest(df, strategy)
        st.subheader("Backtest Trades")
        st.dataframe(res)

    st.divider()
    st.subheader("Live Trading (Paper)")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start Live"):
            st.session_state.live_on = True
            st.session_state.live_pos = None
            log("Live trading started")

    with col2:
        if st.button("⏹ Stop Live"):
            st.session_state.live_on = False
            st.session_state.live_pos = None
            log("Live trading stopped")

    chart_placeholder = st.empty()
    status_placeholder = st.empty()

    while st.session_state.live_on:
        df = fetch_data(ticker, interval, period)
        price = df["Close"].iloc[-1]
        atr_val = atr(df).iloc[-1]

        if strategy == "EMA Crossover":
            sig = ema_crossover_signal(df, EMA_FAST, EMA_SLOW)[-1]
        elif strategy == "Simple Buy":
            sig = 1
        else:
            sig = -1

        pos = st.session_state.live_pos
        idx = len(df) - 1

        # ENTRY
        if pos is None and sig != 0:
            st.session_state.live_pos = {
                "dir": sig,
                "entry": price,
                "sl": price - sig * SL_POINTS,
                "entry_index": idx,
                "last_low": df["Low"].iloc[-1],
                "last_high": df["High"].iloc[-1],
            }
            log(f"ENTRY {'LONG' if sig==1 else 'SHORT'} @ {price}")

        # MANAGEMENT
        elif pos:
            pos["last_low"] = df["Low"].iloc[-1]
            pos["last_high"] = df["High"].iloc[-1]

            pos["sl"] = update_trailing_sl(
                pos, price, atr_val, TRAIL_TYPE, SL_POINTS
            )

            if signal_based_exit(pos, df, idx, EMA_FAST, EMA_SLOW):
                log(f"SIGNAL EXIT @ {price}")
                st.session_state.trade_history.append(pos)
                st.session_state.live_pos = None

            elif (pos["dir"] == 1 and price <= pos["sl"]) or \
                 (pos["dir"] == -1 and price >= pos["sl"]):
                log(f"SL HIT @ {price}")
                st.session_state.trade_history.append(pos)
                st.session_state.live_pos = None

        # CHART (KEY FIXED)
        fig = go.Figure()
        fig.add_candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"]
        )
        fig.add_scatter(x=df.index, y=ema(df["Close"], EMA_FAST), name="EMA Fast")
        fig.add_scatter(x=df.index, y=ema(df["Close"], EMA_SLOW), name="EMA Slow")

        chart_placeholder.plotly_chart(
            fig, use_container_width=True, key="live_chart"
        )
        status_placeholder.write(st.session_state.live_pos)

        time.sleep(random.uniform(1.0, 1.5))

st.subheader("Logs")
st.text("\n".join(st.session_state.logs[-20:]))
