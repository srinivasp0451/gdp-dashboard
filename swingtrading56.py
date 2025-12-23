# ============================
# app.py — Professional Algorithmic Trading Analysis System
# Single-file institutional-grade implementation
# ============================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import zscore
from scipy.signal import argrelextrema
import pytz
import time
from datetime import datetime, timedelta

# ============================
# CONFIGURATION
# ============================

IST = pytz.timezone("Asia/Kolkata")

ASSETS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "USD/INR": "INR=X",
    "EUR/USD": "EURUSD=X",
}

VALID_COMBOS = {
    "1m": ["1d", "5d"],
    "5m": ["1d", "1mo"],
    "15m": ["1d", "1mo"],
    "30m": ["1d", "1mo"],
    "1h": ["1mo", "3mo"],
    "1d": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
    "1wk": ["1y", "2y", "5y"],
}

MAX_TIMEFRAMES = 20

# ============================
# UTILITIES
# ============================

def human_time(dt):
    now = datetime.now(IST)
    diff = now - dt
    if diff < timedelta(hours=1):
        return f"{int(diff.total_seconds()//60)} minutes ago"
    elif diff < timedelta(days=1):
        return f"{int(diff.total_seconds()//3600)} hours ago"
    elif diff < timedelta(days=30):
        return f"{diff.days} days ago"
    else:
        return f"{diff.days//30} months and {diff.days%30} days ago ({dt.strftime('%Y-%m-%d %H:%M:%S IST')})"

def to_ist(df):
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(IST)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "DateTime_IST"}, inplace=True)
    return df

def flatten_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]]

# ============================
# INDICATORS
# ============================

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(close, period):
    return close.ewm(span=period, adjust=False).mean()

def calculate_volatility(close, window=20):
    returns = close.pct_change()
    return returns.rolling(window).std() * np.sqrt(252) * 100

# ============================
# SUPPORT / RESISTANCE
# ============================

def detect_support_resistance(df, order=5, tolerance=0.002):
    prices = df["Close"]
    supports = prices.iloc[argrelextrema(prices.values, np.less_equal, order=order)[0]]
    resistances = prices.iloc[argrelextrema(prices.values, np.greater_equal, order=order)[0]]
    levels = pd.concat([supports, resistances]).sort_values()
    clustered = []
    for price in levels:
        if not any(abs(price - c) / c < tolerance for c in clustered):
            clustered.append(price)
    return clustered

# ============================
# DATA FETCH
# ============================

def fetch_data(ticker, interval, period):
    try:
        time.sleep(1)
        df = yf.download(
            ticker,
            interval=interval,
            period=period,
            progress=False,
            auto_adjust=False
        )
        if df.empty:
            return None
        df = flatten_df(df)
        df = to_ist(df)
        return df.dropna()
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return None

# ============================
# MULTI-TIMEFRAME ANALYSIS
# ============================

def analyze_timeframe(df):
    close = df["Close"]
    result = {}

    result["price"] = close.iloc[-1]
    result["rsi"] = calculate_rsi(close).iloc[-1]
    result["ema20"] = calculate_ema(close, 20).iloc[-1]
    result["ema50"] = calculate_ema(close, 50).iloc[-1]

    returns = close.pct_change()
    z = zscore(returns.dropna())[-1]
    result["zscore"] = z

    vol = calculate_volatility(close).iloc[-1]
    result["volatility"] = vol

    if result["rsi"] < 30:
        result["bias"] = "Bullish"
    elif result["rsi"] > 70:
        result["bias"] = "Bearish"
    else:
        result["bias"] = "Neutral"

    return result

# ============================
# AI SCORING
# ============================

def score_signal(res):
    score = 0
    if res["rsi"] < 30:
        score += 20
    elif res["rsi"] > 70:
        score -= 20

    if res["ema20"] > res["ema50"]:
        score += 15
    else:
        score -= 15

    if res["zscore"] < -2:
        score += 20
    elif res["zscore"] > 2:
        score -= 20

    if res["volatility"] < 15:
        score += 5
    elif res["volatility"] > 30:
        score -= 5

    return score

# ============================
# STREAMLIT UI
# ============================

st.set_page_config(layout="wide", page_title="Professional Algo Trading System")

st.title("📊 Professional Algorithmic Trading Analysis System")

ticker_name = st.selectbox("Select Asset", list(ASSETS.keys()))
custom_ticker = st.text_input("Or Custom Ticker (yfinance)")
ticker = custom_ticker if custom_ticker else ASSETS[ticker_name]

intervals = list(VALID_COMBOS.keys())
selected_intervals = st.multiselect("Select Timeframes", intervals, default=["5m", "15m", "1d"])

period_map = {}
for i in selected_intervals:
    period_map[i] = st.selectbox(f"Period for {i}", VALID_COMBOS[i], key=i)

if st.button("🚀 Run Full Multi-Timeframe Analysis"):
    all_results = []
    progress = st.progress(0)
    for idx, tf in enumerate(selected_intervals):
        df = fetch_data(ticker, tf, period_map[tf])
        if df is None:
            continue

        res = analyze_timeframe(df)
        score = score_signal(res)

        all_results.append({
            "Timeframe": tf,
            "Period": period_map[tf],
            "Price": res["price"],
            "RSI": res["rsi"],
            "Z-Score": res["zscore"],
            "Volatility %": res["volatility"],
            "EMA20": res["ema20"],
            "EMA50": res["ema50"],
            "Bias": res["bias"],
            "Score": score,
        })

        progress.progress((idx + 1) / len(selected_intervals))

    df_final = pd.DataFrame(all_results)

    st.subheader("📋 Multi-Timeframe Overview")
    st.dataframe(df_final, use_container_width=True)

    avg_score = df_final["Score"].mean()
    bullish = (df_final["Score"] > 0).sum()
    bearish = (df_final["Score"] < 0).sum()

    if avg_score > 30:
        signal = "🟢 STRONG BUY"
    elif avg_score > 15:
        signal = "🟢 BUY"
    elif avg_score < -30:
        signal = "🔴 STRONG SELL"
    elif avg_score < -15:
        signal = "🔴 SELL"
    else:
        signal = "🟡 HOLD"

    confidence = min(95, 60 + (bullish / len(df_final)) * 30 + abs(avg_score) * 0.3)

    st.markdown(f"""
    ## {signal}
    **Confidence:** {confidence:.1f}%  
    **Average Score:** {avg_score:.2f}

    **Bullish Timeframes:** {bullish}  
    **Bearish Timeframes:** {bearish}
    """)

    price = df_final["Price"].iloc[0]
    sl = price * 0.985
    target = price * 1.0175

    st.markdown(f"""
    ### 📌 Trading Plan
    **Entry:** ₹{price:,.2f}  
    **Stop Loss:** ₹{sl:,.2f}  
    **Target:** ₹{target:,.2f}  
    """)

st.caption("⚠️ Educational purposes only. Trading involves risk.")
