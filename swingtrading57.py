import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time, random, math
from datetime import datetime
import pytz
import plotly.graph_objects as go

# ================= CONFIG =================
IST = pytz.timezone("Asia/Kolkata")
st.set_page_config(layout="wide")

# ================= SESSION =================
def ss(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

ss("data", None)
ss("live_on", False)
ss("live_pos", None)
ss("trades", [])
ss("logs", [])

# ================= LOG =================
def log(msg):
    st.session_state.logs.append(
        f"{datetime.now(IST).strftime('%H:%M:%S')} | {msg}"
    )

# ================= SAFE TZ =================
def ensure_ist(df):
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.tz_convert(IST)

# ================= UTILS =================
def flatten_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def rate_limit():
    time.sleep(random.uniform(1.0, 1.5))

# ================= INDICATORS =================
def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h-l, abs(h-c.shift()), abs(l-c.shift())], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ================= DATA =================
def fetch_data(ticker, interval, period):
    rate_limit()
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    df = flatten_df(df)
    df.dropna(inplace=True)
    df = ensure_ist(df)
    return df

# ================= SIGNAL =================
def ema_signal(df, e1, e2):
    df["ema1"] = ema(df["Close"], e1)
    df["ema2"] = ema(df["Close"], e2)

    long = (df["ema1"] > df["ema2"]) & (df["ema1"].shift() <= df["ema2"].shift())
    short = (df["ema1"] < df["ema2"]) & (df["ema1"].shift() >= df["ema2"].shift())
    return np.where(long, 1, np.where(short, -1, 0))

# ================= TRAILING SL =================
def update_trailing_sl(pos, price, atr_val, method, pts):
    dir = pos["dir"]

    if method == "Points":
        new_sl = price - dir * pts

    elif method == "ATR":
        new_sl = price - dir * atr_val

    elif method == "Candle":
        new_sl = pos["last_low"] if dir == 1 else pos["last_high"]

    else:
        return pos["sl"]

    # TRAIL ONLY FORWARD
    if dir == 1:
        return max(pos["sl"], new_sl)
    else:
        return min(pos["sl"], new_sl)

# ================= BACKTEST =================
def backtest(df):
    trades = []
    pos = None

    for i in range(20, len(df)):
        sub = df.iloc[:i+1]
        price = sub["Close"].iloc[-1]
        atr_val = atr(sub).iloc[-1]
        sig = ema_signal(sub, ema1, ema2)[-1]

        if pos is None and sig != 0:
            pos = {
                "dir": sig,
                "entry": price,
                "sl": price - sig * sl_points,
                "entry_time": sub.index[-1],
                "last_low": sub["Low"].iloc[-1],
                "last_high": sub["High"].iloc[-1],
            }

        elif pos:
            pos["last_low"] = sub["Low"].iloc[-1]
            pos["last_high"] = sub["High"].iloc[-1]

            pos["sl"] = update_trailing_sl(
                pos, price, atr_val, trail_type, sl_points
            )

            if (pos["dir"] == 1 and price <= pos["sl"]) or \
               (pos["dir"] == -1 and price >= pos["sl"]):
                pos["exit"] = price
                pos["pnl"] = (price - pos["entry"]) * pos["dir"]
                trades.append(pos)
                pos = None

    return pd.DataFrame(trades)

# ================= UI =================
st.title("EMA Strategy with TRUE Trailing SL")

ticker = st.text_input("Ticker", "^NSEI")
interval = st.selectbox("Interval", ["1m","5m","15m","30m","1h"])
period = st.selectbox("Period", ["1d","5d","1mo"])

ema1 = st.number_input("EMA 1", 1, 100, 9)
ema2 = st.number_input("EMA 2", 1, 200, 15)

trail_type = st.selectbox("Trailing SL Type", ["Points","ATR","Candle"])
sl_points = st.number_input("SL / Trail Points", 1, 500, 10)

if st.button("Fetch Data"):
    st.session_state.data = fetch_data(ticker, interval, period)

if st.session_state.data is not None:
    df = st.session_state.data

    if st.button("Run Backtest"):
        res = backtest(df)
        st.subheader("Backtest Trades")
        st.dataframe(res)

    # ===== LIVE =====
    st.subheader("Live Trading (Paper)")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start"):
            st.session_state.live_on = True
            st.session_state.live_pos = None

    with col2:
        if st.button("⏹ Stop"):
            st.session_state.live_on = False
            st.session_state.live_pos = None

    chart = st.empty()
    status = st.empty()

    while st.session_state.live_on:
        df = fetch_data(ticker, interval, period)
        price = df["Close"].iloc[-1]
        atr_val = atr(df).iloc[-1]
        sig = ema_signal(df, ema1, ema2)[-1]

        pos = st.session_state.live_pos

        if pos is None and sig != 0:
            st.session_state.live_pos = {
                "dir": sig,
                "entry": price,
                "sl": price - sig * sl_points,
                "last_low": df["Low"].iloc[-1],
                "last_high": df["High"].iloc[-1],
            }
            log(f"ENTRY {'LONG' if sig==1 else 'SHORT'} @ {price}")

        elif pos:
            pos["last_low"] = df["Low"].iloc[-1]
            pos["last_high"] = df["High"].iloc[-1]

            pos["sl"] = update_trailing_sl(
                pos, price, atr_val, trail_type, sl_points
            )

            if (pos["dir"] == 1 and price <= pos["sl"]) or \
               (pos["dir"] == -1 and price >= pos["sl"]):
                log(f"EXIT @ {price}")
                st.session_state.live_pos = None

        fig = go.Figure()
        fig.add_candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"]
        )
        fig.add_scatter(x=df.index, y=ema(df["Close"], ema1), name="EMA1")
        fig.add_scatter(x=df.index, y=ema(df["Close"], ema2), name="EMA2")

        chart.plotly_chart(fig, use_container_width=True)
        status.write(st.session_state.live_pos)

        time.sleep(random.uniform(1.0, 1.5))

st.subheader("Logs")
st.text("\n".join(st.session_state.logs[-20:]))
