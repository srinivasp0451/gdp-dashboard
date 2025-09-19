import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime

# -----------------------
# Moving Average Strategy
# -----------------------
def moving_average_strategy(df, short_window=20, long_window=50):
    df['SMA20'] = df['Close'].rolling(window=short_window).mean()
    df['SMA50'] = df['Close'].rolling(window=long_window).mean()

    df['Signal'] = 0
    df.loc[df['SMA20'] > df['SMA50'], 'Signal'] = 1
    df.loc[df['SMA20'] < df['SMA50'], 'Signal'] = -1

    df['Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Return']
    return df

def backtest_performance(df):
    cum_returns = (1 + df['Return']).cumprod()
    cum_strategy = (1 + df['Strategy_Return']).cumprod()
    return cum_returns, cum_strategy

def live_signal(df):
    latest = df.iloc[-1]  # pick last row
    if latest['SMA20'] > latest['SMA50']:
        return "BUY 📈 (Bullish)", "green"
    elif latest['SMA20'] < latest['SMA50']:
        return "SELL 📉 (Bearish)", "red"
    else:
        return "HOLD 🤝 (Neutral)", "gray"

# -----------------------
# Streamlit App
# -----------------------
st.title("📊 Swing Trading Strategy (SMA Crossover)")

# Ticker selection
tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "^NSEI", "^BSESN"]
ticker = st.selectbox("Select Stock/Index", tickers + ["Other"])
if ticker == "Other":
    ticker = st.text_input("Enter Ticker Symbol (e.g. SBIN.NS)", "SBIN.NS")

# Timeframe selection
timeframe_map = {
    "1 Minute": ("1m", "5d"), "3 Minutes": ("3m", "5d"), "5 Minutes": ("5m", "30d"),
    "10 Minutes": ("10m", "60d"), "15 Minutes": ("15m", "60d"), "30 Minutes": ("30m", "60d"),
    "1 Hour": ("60m", "730d"), "4 Hours": ("1h", "730d"),
    "1 Day": ("1d", "10y"), "1 Week": ("1wk", "10y"), "1 Month": ("1mo", "10y")
}
timeframe = st.selectbox("Select Timeframe", list(timeframe_map.keys()))
interval, period = timeframe_map[timeframe]

# Fetch data
try:
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)

    if df.empty:
        st.error("No data fetched. Try different ticker or timeframe.")
    else:
        df = moving_average_strategy(df)
        cum_returns, cum_strategy = backtest_performance(df)

        # Show backtest chart
        st.subheader("📈 Backtest Performance")
        st.line_chart(pd.DataFrame({"Buy & Hold": cum_returns, "Strategy": cum_strategy}))

        # Show live recommendation
        signal, color = live_signal(df)
        st.subheader("🚦 Live Recommendation")
        st.markdown(f"<h3 style='color:{color}'>{signal}</h3>", unsafe_allow_html=True)

        st.dataframe(df.tail(10))  # show last 10 rows for reference

except Exception as e:
    st.error(f"Error fetching or processing data: {e}")
