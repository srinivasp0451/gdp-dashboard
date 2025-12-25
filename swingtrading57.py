import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pytz
import time
import random
from datetime import datetime

# ---------------- CONFIG ---------------- #
IST = pytz.timezone("Asia/Kolkata")

st.set_page_config(layout="wide")
st.title("EMA Crossover Algo (Stable Core Version)")

# ---------------- UTILITIES ---------------- #
def safe_sleep():
    time.sleep(random.uniform(1.0, 1.5))

def convert_to_ist(idx):
    if idx.tz is None:
        return idx.tz_localize("UTC").tz_convert(IST)
    return idx.tz_convert(IST)

def flatten_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def ema_angle(ema_series):
    slope = ema_series.diff()
    angle = np.degrees(np.arctan(slope))
    return angle

# ---------------- DATA FETCH ---------------- #
@st.cache_data(show_spinner=False)
def fetch_data(ticker, interval, period):
    safe_sleep()
    df = yf.download(
        ticker,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    df = flatten_df(df)
    df.index = convert_to_ist(df.index)
    return df.dropna()

# ---------------- STRATEGY ENGINE ---------------- #
def backtest_ema(
    df,
    fast=9,
    slow=15,
    angle_thresh=20,
    trailing_sl_points=10,
    use_signal_sl=True,
):
    df = df.copy()
    df["ema_fast"] = ema(df["Close"], fast)
    df["ema_slow"] = ema(df["Close"], slow)
    df["angle"] = ema_angle(df["ema_fast"])

    trades = []
    in_trade = False
    entry_price = sl = None

    for i in range(2, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]

        # ENTRY
        if not in_trade:
            crossed_up = (
                prev.ema_fast < prev.ema_slow
                and row.ema_fast > row.ema_slow
                and row.angle > angle_thresh
            )
            if crossed_up:
                entry_price = row.Close
                sl = entry_price - trailing_sl_points
                in_trade = True
                trades.append(
                    {
                        "Entry Time": row.name,
                        "Entry": entry_price,
                        "SL Type": "Signal / Trailing",
                    }
                )

        # MANAGE TRADE
        else:
            # Trailing SL
            sl = max(sl, row.Close - trailing_sl_points)

            # Exit by SL
            if row.Low <= sl:
                trades[-1].update(
                    {
                        "Exit Time": row.name,
                        "Exit": sl,
                        "Reason": "Trailing SL",
                        "PnL": sl - entry_price,
                    }
                )
                in_trade = False

            # Signal-based SL (reverse crossover)
            elif (
                use_signal_sl
                and prev.ema_fast > prev.ema_slow
                and row.ema_fast < row.ema_slow
            ):
                trades[-1].update(
                    {
                        "Exit Time": row.name,
                        "Exit": row.Close,
                        "Reason": "Reverse EMA",
                        "PnL": row.Close - entry_price,
                    }
                )
                in_trade = False

    return pd.DataFrame(trades), df

# ---------------- UI ---------------- #
ticker = st.text_input("Ticker", "^NSEI")
interval = st.selectbox("Interval", ["1m", "5m", "15m"])
period = st.selectbox("Period", ["1d", "5d", "1mo"])

fast = st.number_input("Fast EMA", 1, 50, 9)
slow = st.number_input("Slow EMA", 1, 100, 15)
angle = st.number_input("Min EMA Angle (deg)", 0, 90, 20)
trail = st.number_input("Trailing SL Points", 1, 1000, 10)

if "df" not in st.session_state:
    st.session_state.df = None

if st.button("Fetch Data"):
    st.session_state.df = fetch_data(ticker, interval, period)

if st.session_state.df is not None:
    df = st.session_state.df

    trades, df2 = backtest_ema(
        df,
        fast=fast,
        slow=slow,
        angle_thresh=angle,
        trailing_sl_points=trail,
    )

    # --------- CHART --------- #
    fig = go.Figure()
    fig.add_candlestick(
        x=df2.index,
        open=df2.Open,
        high=df2.High,
        low=df2.Low,
        close=df2.Close,
        name="Price",
    )
    fig.add_line(
        x=df2.index,
        y=df2.ema_fast,
        name=f"EMA {fast}",
    )
    fig.add_line(
        x=df2.index,
        y=df2.ema_slow,
        name=f"EMA {slow}",
    )

    st.plotly_chart(fig, use_container_width=True, key="chart_main")

    # --------- RESULTS --------- #
    st.subheader("Trades")
    if not trades.empty:
        st.dataframe(trades)
        st.metric("Total PnL", round(trades.PnL.sum(), 2))
    else:
        st.info("No trades generated")
