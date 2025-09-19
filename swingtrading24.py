import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Indicator Functions
# -----------------------------
def ema(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(series, period=20, std=2):
    ma = series.rolling(period).mean()
    std_dev = series.rolling(period).std()
    upper = ma + std * std_dev
    lower = ma - std * std_dev
    return ma, upper, lower

def atr(data, period=14):
    data['H-L'] = data['High'] - data['Low']
    data['H-PC'] = (data['High'] - data['Close'].shift(1)).abs()
    data['L-PC'] = (data['Low'] - data['Close'].shift(1)).abs()
    tr = data[['H-L','H-PC','L-PC']].max(axis=1)
    return tr.rolling(period).mean()

# -----------------------------
# Strategy Logic
# -----------------------------
def generate_signals(data):
    data['EMA20'] = ema(data['Close'], 20)
    data['EMA200'] = ema(data['Close'], 200)
    data['RSI'] = rsi(data['Close'])
    data['BB_Mid'], data['BB_Upper'], data['BB_Lower'] = bollinger_bands(data['Close'])
    data['ATR'] = atr(data)

    signals = []
    position = None

    for i in range(len(data)):
        row = data.iloc[i]

        # Long Setup
        if row['Close'] > row['EMA200'] and row['Close'] <= row['EMA20'] and row['RSI'] > 45:
            if position != "LONG":
                signals.append("BUY")
                position = "LONG"
            else:
                signals.append(None)

        # Short Setup
        elif row['Close'] < row['EMA200'] and row['Close'] >= row['EMA20'] and row['RSI'] < 55:
            if position != "SHORT":
                signals.append("SELL")
                position = "SHORT"
            else:
                signals.append(None)

        else:
            signals.append(None)

    data['Signal'] = signals
    return data

# -----------------------------
# Backtest
# -----------------------------
def backtest(data, capital=100000, risk_per_trade=0.01):
    balance = capital
    equity_curve = []
    position = None
    entry_price = 0
    qty = 0

    for i in range(len(data)):
        row = data.iloc[i]

        if row['Signal'] == "BUY":
            sl = row['Close'] - 2 * row['ATR']
            if sl > 0:
                qty = (balance * risk_per_trade) // (row['Close'] - sl)
                entry_price = row['Close']
                position = "LONG"

        elif row['Signal'] == "SELL":
            sl = row['Close'] + 2 * row['ATR']
            if sl > 0:
                qty = (balance * risk_per_trade) // (sl - row['Close'])
                entry_price = row['Close']
                position = "SHORT"

        if position == "LONG":
            pnl = (row['Close'] - entry_price) * qty
        elif position == "SHORT":
            pnl = (entry_price - row['Close']) * qty
        else:
            pnl = 0

        balance = capital + pnl
        equity_curve.append(balance)

    data['Equity'] = equity_curve
    return data

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📈 Swing Trading Strategy - India Market")

tickers = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "FIN NIFTY": "NIFTY_FIN_SERVICE.NS",
    "SENSEX": "^BSESN",
    "RELIANCE": "RELIANCE.NS",
    "INFY": "INFY.NS",
    "TCS": "TCS.NS",
    "HDFC BANK": "HDFCBANK.NS"
}
choice = st.selectbox("Select Ticker", list(tickers.keys()) + ["Other"])
if choice == "Other":
    ticker = st.text_input("Enter Yahoo Ticker (e.g. SBIN.NS)", "SBIN.NS")
else:
    ticker = tickers[choice]

# Valid periods & intervals
period_options = ["1d","5d","1mo","3mo","6mo","1y","2y","3y","5y","10y","ytd","max"]
interval_options = ["1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"]

period = st.selectbox("Select Period", period_options, index=5)
interval = st.selectbox("Select Interval", interval_options, index=8)

# Data fetch with error handling
try:
    data = yf.download(ticker, period=period, interval=interval)
    if data.empty:
        st.error("⚠️ No data available. Try another period/interval combination.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error fetching data: {e}")
    st.stop()

# Strategy + Backtest
data = data.dropna()
data = generate_signals(data)
data = backtest(data)

# Show latest signal
st.subheader("📢 Latest Signal")
latest_signal = data['Signal'].dropna().iloc[-1] if data['Signal'].dropna().any() else "No Signal"
st.write(f"**{ticker}** → {latest_signal}")

# Plot price & signals
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(data.index, data['Close'], label="Close Price", color="blue")
ax.plot(data.index, data['EMA20'], label="EMA20", color="orange")
ax.plot(data.index, data['EMA200'], label="EMA200", color="red")
ax.scatter(data.index[data['Signal']=="BUY"], data['Close'][data['Signal']=="BUY"], marker="^", color="green", label="Buy Signal", s=100)
ax.scatter(data.index[data['Signal']=="SELL"], data['Close'][data['Signal']=="SELL"], marker="v", color="red", label="Sell Signal", s=100)
ax.legend()
st.pyplot(fig)

# Equity curve
st.subheader("💰 Equity Curve")
fig2, ax2 = plt.subplots(figsize=(12,4))
ax2.plot(data.index, data['Equity'], color="purple")
st.pyplot(fig2)
