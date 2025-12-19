# =========================================================
# REAL LIVE AUTO-REFRESHING ALGO TRADING SYSTEM
# =========================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time, math
from datetime import datetime
import pytz
from scipy.signal import argrelextrema

# ================== CONFIG ==================
IST = pytz.timezone("Asia/Kolkata")
REFRESH_SEC = 1.7

st.set_page_config("Live Algo Trading Pro", layout="wide")

# ================== UTIL ==================
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

# ================== STRATEGIES ==================
class StrategyEngine:

    def ema_crossover(self, df):
        df["EMA9"] = ema(df["Close"], 9)
        df["EMA20"] = ema(df["Close"], 20)
        prev, curr = df.iloc[-2], df.iloc[-1]
        return (
            prev.EMA9 < prev.EMA20 and curr.EMA9 > curr.EMA20,
            prev.EMA9 > prev.EMA20 and curr.EMA9 < curr.EMA20,
            {"EMA9": curr.EMA9, "EMA20": curr.EMA20}
        )

    def rsi_divergence(self, df):
        df["RSI"] = rsi(df["Close"])
        if len(df) < 20:
            return False, False, {}
        bullish = df.Close.iloc[-1] < df.Close.iloc[-5] and df.RSI.iloc[-1] > df.RSI.iloc[-5]
        bearish = df.Close.iloc[-1] > df.Close.iloc[-5] and df.RSI.iloc[-1] < df.RSI.iloc[-5]
        return bullish, bearish, {"RSI": df.RSI.iloc[-1]}

    def elliott_wave(self, df):
        closes = df["Close"]
        highs = argrelextrema(closes.values, np.greater, order=5)[0]
        lows = argrelextrema(closes.values, np.less, order=5)[0]
        if len(highs) + len(lows) >= 5:
            return True, False, {"waves": len(highs)+len(lows)}
        return False, False, {}

    def simple_buy(self, df):
        return True, False, {"reason": "Manual Buy"}

    def simple_sell(self, df):
        return False, True, {"reason": "Manual Sell"}

# ================== SESSION ==================
for k, v in {
    "active": False,
    "position": None,
    "log": [],
    "history": [],
    "iteration": 0
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================== SIDEBAR ==================
st.sidebar.title("⚙️ Controls")

ticker = st.sidebar.text_input("Ticker", "^NSEI")
interval = st.sidebar.selectbox("Interval", ["1m","5m","15m","30m","1h","1d"])
period = st.sidebar.selectbox("Period", ["1d","5d","1mo","3mo","6mo","1y"])

strategy_name = st.sidebar.selectbox(
    "Strategy",
    ["EMA Crossover","RSI Divergence","Elliott Waves","Simple Buy","Simple Sell"]
)

qty = st.sidebar.number_input("Quantity", 1, step=1)

sl_pts = st.sidebar.number_input("Stop Loss (points)", 20.0)
tgt_pts = st.sidebar.number_input("Target (points)", 40.0)

start = st.sidebar.button("▶ START")
stop = st.sidebar.button("⛔ STOP")

engine = StrategyEngine()

# ================== TABS ==================
tab1, tab2, tab3 = st.tabs(["📈 Live Trading","📊 Trade History","🧾 Log"])

# ================== LIVE ==================
with tab1:
    if start:
        st.session_state.active = True
        st.session_state.log.append("Trading Started")

    if stop:
        st.session_state.active = False
        st.session_state.log.append("Trading Stopped")

    if st.session_state.active:
        st.success(f"🔴 LIVE | Iteration {st.session_state.iteration}")

        df = yf.download(ticker, interval=interval, period=period, progress=False)
        df = flatten(df)
        df = to_ist(df)

        if df.empty:
            st.error("No data")
        else:
            if strategy_name == "EMA Crossover":
                bull, bear, info = engine.ema_crossover(df)
            elif strategy_name == "RSI Divergence":
                bull, bear, info = engine.rsi_divergence(df)
            elif strategy_name == "Elliott Waves":
                bull, bear, info = engine.elliott_wave(df)
            elif strategy_name == "Simple Buy":
                bull, bear, info = engine.simple_buy(df)
            else:
                bull, bear, info = engine.simple_sell(df)

            price = df.Close.iloc[-1]

            if st.session_state.position is None:
                st.write("### 🕒 Waiting for Entry")
                st.write(info)

                if bull or bear:
                    side = "LONG" if bull else "SHORT"
                    st.session_state.position = {
                        "side": side,
                        "entry": price,
                        "sl": price - sl_pts if side=="LONG" else price + sl_pts,
                        "tgt": price + tgt_pts if side=="LONG" else price - tgt_pts,
                        "time": datetime.now(IST)
                    }
                    st.session_state.log.append(f"ENTRY {side} @ {price}")

            else:
                p = st.session_state.position
                pnl = (price - p["entry"]) * (1 if p["side"]=="LONG" else -1)

                st.write("### 📊 Live Trade")
                st.write(f"""
Side: {p['side']}  
Entry: {p['entry']:.2f}  
Current: {price:.2f}  
PnL: {pnl:.2f}  
SL: {p['sl']:.2f}  
Target: {p['tgt']:.2f}
""")

                if (p["side"]=="LONG" and price <= p["sl"]) or (p["side"]=="SHORT" and price >= p["sl"]):
                    st.session_state.log.append("EXIT SL")
                    st.session_state.history.append(p)
                    st.session_state.position = None

                if (p["side"]=="LONG" and price >= p["tgt"]) or (p["side"]=="SHORT" and price <= p["tgt"]):
                    st.session_state.log.append("EXIT TARGET")
                    st.session_state.history.append(p)
                    st.session_state.position = None

            fig = go.Figure(go.Candlestick(
                x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close
            ))
            if st.session_state.position:
                fig.add_hline(y=st.session_state.position["sl"], line_color="red")
                fig.add_hline(y=st.session_state.position["tgt"], line_color="green")
            st.plotly_chart(fig, use_container_width=True)

        st.session_state.iteration += 1
        time.sleep(REFRESH_SEC)
        st.rerun()

# ================== HISTORY ==================
with tab2:
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history))
    else:
        st.info("No trades")

# ================== LOG ==================
with tab3:
    st.text_area("Log", "\n".join(st.session_state.log[-100:]), height=600)
