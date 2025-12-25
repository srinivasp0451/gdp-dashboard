import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time, random, math
from datetime import datetime
import pytz
import plotly.graph_objects as go

IST = pytz.timezone("Asia/Kolkata")
st.set_page_config(layout="wide")

# ================= SESSION STATE =================
def ss(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

ss("data", None)
ss("live_trade", None)
ss("trade_history", [])
ss("logs", [])
ss("live_on", False)

# ================= LOGGING =================
def log(msg):
    st.session_state.logs.append(
        f"{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    )

# ================= UTILS =================
def flatten_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def rate_limit():
    time.sleep(random.uniform(1.0, 1.5))

# ================= INDICATORS =================
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def atr(df, period=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h-l, abs(h-c.shift()), abs(l-c.shift())], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def swing_low(df, lookback=5):
    return df["Low"].rolling(lookback).min()

def swing_high(df, lookback=5):
    return df["High"].rolling(lookback).max()

# ================= DATA FETCH =================
def fetch_data(ticker, interval, period):
    rate_limit()
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    df = flatten_df(df)
    df.dropna(inplace=True)
    df.index = df.index.tz_localize("UTC").tz_convert(IST)
    return df

# ================= EMA CROSS SIGNAL =================
def ema_signal(df, ema1, ema2, angle_min):
    df["ema1"] = ema(df["Close"], ema1)
    df["ema2"] = ema(df["Close"], ema2)

    slope = np.degrees(np.arctan(df["ema1"].diff()))
    long = (
        (df["ema1"] > df["ema2"]) &
        (df["ema1"].shift() <= df["ema2"].shift()) &
        (slope > angle_min)
    )
    short = (
        (df["ema1"] < df["ema2"]) &
        (df["ema1"].shift() >= df["ema2"].shift()) &
        (slope < -angle_min)
    )

    signal = np.where(long, 1, np.where(short, -1, 0))
    return signal, slope.iloc[-1]

# ================= SL / TARGET =================
def compute_sl_target(df, entry, direction, sl_type, tgt_type, atr_val, rr):
    if sl_type == "Custom":
        sl = entry - direction * st.session_state.sl_points
    elif sl_type == "ATR":
        sl = entry - direction * atr_val
    elif sl_type == "Prev Candle":
        sl = df["Low"].iloc[-2] if direction == 1 else df["High"].iloc[-2]
    elif sl_type == "Swing":
        sl = swing_low(df).iloc[-1] if direction == 1 else swing_high(df).iloc[-1]
    else:
        sl = entry - direction * 10

    risk = abs(entry - sl)

    if tgt_type == "RR":
        tgt = entry + direction * risk * rr
    elif tgt_type == "ATR":
        tgt = entry + direction * atr_val
    elif tgt_type == "Custom":
        tgt = entry + direction * st.session_state.tgt_points
    else:
        tgt = entry + direction * risk * 2

    return sl, tgt

# ================= BACKTEST =================
def backtest(df, strategy):
    trades = []
    pos = None

    for i in range(50, len(df)):
        sub = df.iloc[:i+1]
        atr_val = atr(sub).iloc[-1]

        if strategy == "EMA Crossover":
            sig, angle = ema_signal(
                sub, st.session_state.ema1, st.session_state.ema2, st.session_state.angle
            )
            sig = sig[-1]
        elif strategy == "Simple Buy":
            sig = 1
        else:
            sig = -1

        price = sub["Close"].iloc[-1]

        if pos is None and sig != 0:
            sl, tgt = compute_sl_target(sub, price, sig,
                                        st.session_state.sl_type,
                                        st.session_state.tgt_type,
                                        atr_val,
                                        st.session_state.rr)

            pos = {
                "dir": sig,
                "entry": price,
                "sl": sl,
                "tgt": tgt,
                "time": sub.index[-1]
            }

        elif pos:
            if (pos["dir"] == 1 and price <= pos["sl"]) or \
               (pos["dir"] == -1 and price >= pos["sl"]):
                pos["exit"] = price
                pos["pnl"] = (price - pos["entry"]) * pos["dir"]
                trades.append(pos)
                pos = None

            elif (pos["dir"] == 1 and price >= pos["tgt"]) or \
                 (pos["dir"] == -1 and price <= pos["tgt"]):
                pos["exit"] = price
                pos["pnl"] = (price - pos["entry"]) * pos["dir"]
                trades.append(pos)
                pos = None

    return pd.DataFrame(trades)

# ================= UI =================
st.title("EMA Crossover – Professional Trading Engine")

ticker = st.text_input("Ticker", "^NSEI")
interval = st.selectbox("Interval", ["1m","5m","15m","30m","1h","1d"])
period = st.selectbox("Period", ["1d","5d","1mo","1y"])

if st.button("Fetch Data"):
    st.session_state.data = fetch_data(ticker, interval, period)

if st.session_state.data is not None:
    df = st.session_state.data

    strategy = st.selectbox("Strategy", ["EMA Crossover","Simple Buy","Simple Sell"])

    st.session_state.ema1 = st.number_input("EMA 1", 1, 100, 9)
    st.session_state.ema2 = st.number_input("EMA 2", 1, 200, 15)
    st.session_state.angle = st.number_input("Min Angle", 0, 90, 20)

    st.session_state.sl_type = st.selectbox(
        "SL Type", ["Custom","ATR","Prev Candle","Swing"]
    )
    st.session_state.tgt_type = st.selectbox(
        "Target Type", ["RR","ATR","Custom"]
    )

    st.session_state.sl_points = st.number_input("SL Points", 1, 500, 10)
    st.session_state.tgt_points = st.number_input("Target Points", 1, 500, 20)
    st.session_state.rr = st.number_input("Risk Reward", 0.5, 5.0, 2.0)

    if st.button("Run Backtest"):
        res = backtest(df, strategy)
        st.subheader("Trade History")
        st.dataframe(res)

    fig = go.Figure()
    fig.add_candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"]
    )
    fig.add_scatter(x=df.index, y=ema(df["Close"], st.session_state.ema1),
                    name="EMA1")
    fig.add_scatter(x=df.index, y=ema(df["Close"], st.session_state.ema2),
                    name="EMA2")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Logs")
st.text("\n".join(st.session_state.logs[-20:]))
