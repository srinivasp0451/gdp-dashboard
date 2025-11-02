import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
import pytz
import plotly.graph_objs as go

# ---------------------- SETTINGS ----------------------
DEFAULT_TICKERS = {
    "Nifty50": "^NSEI",
    "Sensex": "^BSESN",
    "BankNifty": "^NSEBANK",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "USDINR": "USDINR=X",
    "EURINR": "EURINR=X",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "RELIANCE": "RELIANCE.NS"
}
INTERVAL_PERIOD_MAP = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
    "1h": "730d",
    "1d": "30y",
    "5d": "30y",
    "1wk": "30y",
    "1mo": "30y",
    "3mo": "30y"
}
INTERVALS = list(INTERVAL_PERIOD_MAP.keys())

# ---------------------- UTILS -------------------------
def safe_fetch(ticker, period, interval, tzstr="Asia/Kolkata"):
    if INTERVAL_PERIOD_MAP.get(interval):
        max_period = INTERVAL_PERIOD_MAP[interval]
        # Adjust period if needed
        def period_order(per):
            order = ["7d", "60d", "730d", "30y"]
            return order.index(per) if per in order else len(order)-1
        got = period_order(period) if period in INTERVAL_PERIOD_MAP.values() else 0
        allowed = period_order(max_period)
        if got > allowed:
            period = max_period
            st.info(f"Interval '{interval}' supports max period '{max_period}', period auto-set.")
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, threads=False)
    # Check and flatten multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]
    df.reset_index(inplace=True)
    # Timezone conversion to IST (and remove tz awareness for Excel export)
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.tz_convert(tzstr).dt.tz_localize(None)
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(tzstr).dt.tz_localize(None)
    elif "index" in df.columns:  # fallback
        df["index"] = pd.to_datetime(df["index"]).dt.tz_localize(tzstr).dt.tz_localize(None)
    if df.empty:
        st.error("No data returned. Try a supported ticker, period, or interval.")
        return None
    return df

def color_points(x):
    return ['background-color: #2ecc40' if v > 0 else 'background-color: #ff4136' for v in x]

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ---------------------- STREAMLIT UI ------------------
st.set_page_config(page_title="MULTI-INSTRUMENT DASH", layout="wide")
st.title("🎯 Multi-Instrument Trading Dashboard")

# Sidebar Controls
selected = st.sidebar.multiselect("Select Symbols", list(DEFAULT_TICKERS.keys()),
                                 default=list(DEFAULT_TICKERS.keys())[:3])
custom_ticker = st.sidebar.text_input("Custom Ticker (Yahoo format)", "")
interval = st.sidebar.selectbox("Interval", INTERVALS, index=INTERVALS.index("5m"))
period = st.sidebar.selectbox("Period", sorted(set(INTERVAL_PERIOD_MAP.values())),
                              index=1)
export_fmt = st.sidebar.selectbox("Export Format", ["CSV", "Excel"])
theme = st.sidebar.radio("Theme", ["Light", "Dark"])
fetch_button = st.button("Fetch & Analyze Data")

if theme == "Dark":
    st.markdown("""<style>
        body {background-color: #222 !important; color: #ddd !important;}
        .reportview-container .main .block-container{padding-top:2rem;}
    </style>""", unsafe_allow_html=True)

# --------------- FETCH + ANALYSIS + DISPLAY -----------
if fetch_button:
    symbols = selected.copy()
    if custom_ticker:
        symbols.append(custom_ticker)
    for sym in symbols:
        st.header(f"{sym} ({DEFAULT_TICKERS.get(sym, sym)}) Analysis")
        df = safe_fetch(DEFAULT_TICKERS.get(sym, sym), period, interval)
        if df is None or "Close" not in df.columns: continue
        # Data & calculations
        df["Change"] = df["Close"].diff()
        df["Change %"] = df["Close"].pct_change()*100
        df["RSI"] = compute_rsi(df["Close"])
        # Display Table
        dshow = df.copy()
        dshow["Up/Down"] = np.where(dshow["Change"] > 0, "Up", "Down")
        st.dataframe(dshow.style.apply(color_points, subset=["Change"]), height=300)
        # Candlestick Chart
        fig = go.Figure(data=[go.Candlestick(
            x=df[df.columns[0]], open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing_line_color='green', decreasing_line_color='red')])
        fig.update_layout(xaxis_rangeslider_visible=False,
                          template="plotly_dark" if theme == "Dark" else "plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        # RSI chart
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df[df.columns[0]], y=df["RSI"], mode="lines", name="RSI", line=dict(color="orange")))
        st.plotly_chart(fig2, use_container_width=True)
        # Insights
        st.info(f"{sym}: Last close {df['Close'].iloc[-1]:.2f}, move: {'UP' if df['Change'].iloc[-1]>0 else 'DOWN'}, RSI: {df['RSI'].iloc[-1]:.2f}")
        # Export
        fname = f"{sym.replace('^','')}_data.{export_fmt.lower()}"
        if export_fmt == "CSV":
            st.download_button(f"Download {sym} CSV", data=dshow.to_csv(index=False), file_name=fname)
        else:
            dshow.to_excel("temp.xlsx", index=False)
            with open("temp.xlsx", "rb") as f:
                st.download_button(f"Download {sym} Excel", data=f, file_name=fname)
