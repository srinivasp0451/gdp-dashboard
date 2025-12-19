# ============================================================
# PROFESSIONAL LIVE ALGO TRADING SYSTEM (STREAMLIT ONLY)
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time, math
from datetime import datetime
import pytz

# ================= CONFIG =================
IST = pytz.timezone("Asia/Kolkata")
RATE_LIMIT = 1.7

st.set_page_config("Algo Trading Pro", layout="wide")

# ================= UTILITIES =================
def safe_sleep():
    time.sleep(RATE_LIMIT)

def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def to_ist(df):
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.tz_convert(IST)

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()

def atr(df, n=14):
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift()),
        abs(df['Low'] - df['Close'].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ================= STRATEGY =================
class EMACrossover:
    name = "EMA Crossover"

    def __init__(self, fast=9, slow=20):
        self.fast, self.slow = fast, slow

    def calculate(self, df):
        df['EMA_FAST'] = ema(df['Close'], self.fast)
        df['EMA_SLOW'] = ema(df['Close'], self.slow)
        df['ATR'] = atr(df)
        return df

    def signal(self, df):
        if len(df) < self.slow + 2:
            return False, False, {}

        prev, curr = df.iloc[-2], df.iloc[-1]
        bullish = prev.EMA_FAST < prev.EMA_SLOW and curr.EMA_FAST > curr.EMA_SLOW
        bearish = prev.EMA_FAST > prev.EMA_SLOW and curr.EMA_FAST < curr.EMA_SLOW

        slope = curr.EMA_FAST - prev.EMA_FAST
        angle = abs(math.degrees(math.atan(slope)))

        return bullish, bearish, {
            "ema_fast": round(curr.EMA_FAST,2),
            "ema_slow": round(curr.EMA_SLOW,2),
            "angle": round(angle,2),
            "atr": round(curr.ATR,2)
        }

# ================= RISK MANAGER =================
class RiskManager:

    def sl(self, entry, df, side, typ, val):
        if typ == "Custom Points":
            return entry - val if side=="LONG" else entry + val
        if typ == "ATR Based":
            return entry - df['ATR'].iloc[-1]*val if side=="LONG" else entry + df['ATR'].iloc[-1]*val
        if typ == "Percentage Based":
            p = entry*val/100
            return entry - p if side=="LONG" else entry + p
        return None

    def target(self, entry, sl, side, typ, val):
        if typ == "Custom Points":
            return entry + val if side=="LONG" else entry - val
        if typ == "Risk Reward Ratio":
            risk = abs(entry-sl)
            return entry + risk*val if side=="LONG" else entry - risk*val
        if typ == "Percentage Based":
            p = entry*val/100
            return entry + p if side=="LONG" else entry - p
        return None

# ================= SESSION =================
for k,v in {
    "active":False,
    "position":None,
    "history":[],
    "log":[],
    "iter":0
}.items():
    if k not in st.session_state:
        st.session_state[k]=v

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Settings")

ticker = st.sidebar.text_input("Ticker", "^NSEI")
interval = st.sidebar.selectbox("Interval", ["1m","5m","15m","30m","1h","1d"])
period = st.sidebar.selectbox("Period", ["1d","5d","1mo","3mo","6mo","1y"])

qty = st.sidebar.number_input("Quantity", 1, step=1)

sl_type = st.sidebar.selectbox("Stop Loss Type",
    ["Custom Points","ATR Based","Percentage Based"])
sl_val = st.sidebar.number_input("SL Value", 20.0)

tgt_type = st.sidebar.selectbox("Target Type",
    ["Custom Points","Risk Reward Ratio","Percentage Based"])
tgt_val = st.sidebar.number_input("Target Value", 2.0)

start = st.sidebar.button("▶ Start Trading")
stop = st.sidebar.button("⛔ Stop Trading")

# ================= STRATEGY =================
strategy = EMACrossover()
rm = RiskManager()

# ================= TABS =================
tab1, tab2, tab3 = st.tabs(["📈 Live Trading","📊 Trade History","🧾 Trade Log"])

# ================= LIVE =================
with tab1:
    if start:
        st.session_state.active=True
        st.session_state.log.append("Trading Started")

    if stop:
        st.session_state.active=False
        st.session_state.log.append("Trading Stopped")

    if st.session_state.active:
        st.info("LIVE – auto refresh 1.7s")

        df = yf.download(ticker, interval=interval, period=period, progress=False)
        safe_sleep()

        if not df.empty:
            df = flatten(df)
            df = to_ist(df)
            df = strategy.calculate(df)

            bull, bear, info = strategy.signal(df)
            price = df['Close'].iloc[-1]

            # -------- BEFORE ENTRY STATUS --------
            if st.session_state.position is None:
                st.subheader("🕒 Pre-Entry Status")
                st.write(info)
                st.write("Waiting for signal...")

                if bull or bear:
                    side = "LONG" if bull else "SHORT"
                    sl = rm.sl(price, df, side, sl_type, sl_val)
                    tgt = rm.target(price, sl, side, tgt_type, tgt_val)

                    st.session_state.position = {
                        "side":side,
                        "entry":price,
                        "sl":sl,
                        "target":tgt,
                        "qty":qty,
                        "entry_info":info,
                        "time":datetime.now(IST)
                    }

                    st.session_state.log.append(
                        f"ENTRY {side} @ {price:.2f} SL={sl:.2f} TGT={tgt:.2f}"
                    )

            # -------- AFTER ENTRY STATUS --------
            else:
                pos = st.session_state.position
                pnl = (price-pos["entry"])*pos["qty"]*(1 if pos["side"]=="LONG" else -1)
                pnl_pct = pnl/(pos["entry"]*pos["qty"])*100

                st.subheader("📊 Live Trade Status")
                st.write(f"""
**Side:** {pos['side']}  
**Entry:** {pos['entry']:.2f}  
**Current:** {price:.2f}  
**SL:** {pos['sl']:.2f}  
**Target:** {pos['target']:.2f}  
**PnL:** {pnl:.2f} ({pnl_pct:.2f}%)  
""")

                # Mentor Guidance
                if pnl_pct > 0.5:
                    st.success("Strong momentum. Hold discipline.")
                elif pnl_pct > 0:
                    st.info("Trade moving gradually. Avoid impatience.")
                else:
                    st.warning("In drawdown. Respect SL, no panic.")

                # EXIT LOGIC
                exit_reason=None
                if pos["side"]=="LONG" and price<=pos["sl"]:
                    exit_reason="Stop Loss Hit"
                if pos["side"]=="LONG" and price>=pos["target"]:
                    exit_reason="Target Hit"
                if pos["side"]=="SHORT" and price>=pos["sl"]:
                    exit_reason="Stop Loss Hit"
                if pos["side"]=="SHORT" and price<=pos["target"]:
                    exit_reason="Target Hit"

                if exit_reason:
                    pos["exit"]=price
                    pos["exit_reason"]=exit_reason
                    pos["pnl"]=pnl
                    pos["exit_time"]=datetime.now(IST)
                    st.session_state.history.append(pos)
                    st.session_state.log.append(f"EXIT @ {price:.2f} {exit_reason}")
                    st.session_state.position=None

            # -------- CHART --------
            fig = go.Figure(go.Candlestick(
                x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close
            ))

            if st.session_state.position:
                p=st.session_state.position
                fig.add_hline(y=p["entry"], line_dash="dot")
                fig.add_hline(y=p["sl"], line_dash="dot", line_color="red")
                fig.add_hline(y=p["target"], line_dash="dot", line_color="green")

            st.plotly_chart(fig, use_container_width=True)

# ================= HISTORY =================
with tab2:
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history))
    else:
        st.info("No trades yet")

# ================= LOG =================
with tab3:
    st.text_area(
        "Trade Log",
        "\n".join(st.session_state.log[-100:]),
        height=600
    )
