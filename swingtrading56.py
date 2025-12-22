# ============================================================
# PROFESSIONAL ALGO TRADING RESEARCH ENGINE
# ============================================================
# Educational only. No financial advice.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import zscore
from datetime import timedelta
import time

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Algo Trading Research Engine", layout="wide")

TIMEFRAME_PERIOD_MAP = {
    "1m": ["1d", "5d"],
    "5m": ["5d", "1mo"],
    "15m": ["5d", "1mo"],
    "1h": ["1mo", "3mo", "6mo", "1y"],
    "1d": ["1y", "2y", "5y", "10y"]
}

IST_OFFSET = timedelta(hours=5, minutes=30)

# ---------------- UTILS ----------------
def flatten_yf(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep]

def to_ist(df):
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Asia/Kolkata")
    df.index.name = "DateTime"
    return df

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = -delta.clip(upper=0).rolling(n).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def volatility(series, n=20):
    return series.pct_change().rolling(n).std() * np.sqrt(252)

# ---------------- SUPPORT / RESISTANCE ----------------
def support_resistance(df, tolerance=0.002):
    prices = df["Close"].values
    levels = []

    for i in range(2, len(prices)-2):
        if prices[i] == min(prices[i-2:i+3]):
            levels.append(prices[i])
        if prices[i] == max(prices[i-2:i+3]):
            levels.append(prices[i])

    zones = {}
    for lvl in levels:
        matched = False
        for z in zones:
            if abs(z - lvl) / z <= tolerance:
                zones[z].append(lvl)
                matched = True
                break
        if not matched:
            zones[lvl] = [lvl]

    rows = []
    for z, hits in zones.items():
        rows.append({
            "Level": round(np.mean(hits), 2),
            "Touches": len(hits),
            "Strength": "Strong" if len(hits) >= 4 else "Moderate"
        })

    return pd.DataFrame(rows).sort_values("Touches", ascending=False)

# ---------------- BACKTEST ----------------
def backtest(df):
    df = df.copy()
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["RSI"] = rsi(df["Close"])

    df["Signal"] = np.where(
        (df["EMA20"] > df["EMA50"]) & (df["RSI"] > 55), 1,
        np.where((df["EMA20"] < df["EMA50"]) & (df["RSI"] < 45), -1, 0)
    )

    df["Returns"] = df["Close"].pct_change()
    df["Strategy"] = df["Signal"].shift(1) * df["Returns"]
    df["Equity"] = (1 + df["Strategy"]).cumprod()

    accuracy = (df["Strategy"] > 0).mean() * 100
    total_return = df["Equity"].iloc[-1] - 1

    return df, accuracy, total_return

# ---------------- UI ----------------
st.title("📊 Professional Algorithmic Trading Research Engine")

with st.sidebar:
    ticker = st.text_input("Ticker", "^NSEI")
    tf = st.selectbox("Timeframe", list(TIMEFRAME_PERIOD_MAP.keys()))
    period = st.selectbox("Period", TIMEFRAME_PERIOD_MAP[tf])
    run = st.button("Fetch & Analyze")

if run:
    bar = st.progress(0)
    st.write("Fetching data…")

    df = yf.download(ticker, interval=tf, period=period, progress=False)
    time.sleep(1.8)
    df = flatten_yf(df)
    df = to_ist(df)
    bar.progress(25)

    df["RSI"] = rsi(df["Close"])
    df["EMA9"] = ema(df["Close"], 9)
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["Volatility"] = volatility(df["Close"])
    df["ZScore"] = zscore(df["Close"].pct_change().dropna())
    bar.progress(55)

    sr = support_resistance(df)
    bt, acc, ret = backtest(df)
    bar.progress(80)

    col1, col2, col3 = st.columns(3)
    col1.metric("Price", f"{df['Close'].iloc[-1]:.2f}")
    col2.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
    col3.metric("Volatility", f"{df['Volatility'].iloc[-1]:.2f}")

    fig = go.Figure()
    fig.add_candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"]
    )
    fig.add_scatter(x=df.index, y=df["EMA20"], name="EMA20")
    fig.add_scatter(x=df.index, y=df["EMA50"], name="EMA50")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Strong Support & Resistance Zones")
    st.dataframe(sr)

    st.subheader("Backtest Results")
    st.write(f"Accuracy: {acc:.2f}% | Total Return: {ret*100:.2f}%")
    st.dataframe(bt[["Close", "Signal", "Equity"]].tail(50))

    st.success(
        f"""
        Market is currently trading {'above' if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] else 'below'}
        key EMAs. RSI at {df['RSI'].iloc[-1]:.2f} suggests
        {'bullish momentum' if df['RSI'].iloc[-1] > 55 else 'neutral to weak structure'}.
        Strong demand/supply zones identified with repeated reactions.
        """
    )

    bar.progress(100)
