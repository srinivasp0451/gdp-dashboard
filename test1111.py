import streamlit as st
import yfinance as yf
import pandas as pd

# Dictionary for index options
index_options = {
    "Bank Nifty": "^NSEBANK",
    "Nifty 50": "^NSEI",
    "FinNifty": "NIFTY_FIN_SERVICE.NS",
    "Sensex": "^BSESN",
    "Midcap Nifty": "NIFTY_MID_SELECT.NS",
    "BANKEX": "BSE-BANK.BO"
}

# Function to fetch 5-minute interval data
def fetch_5min_data(symbol, period='1mo'):
    df = yf.download(symbol, period=period, interval='5m')
    return df

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['EMA_12'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

# Function to calculate ATR
def calculate_atr(data, window=14):
    data['High-Low'] = data['High'] - data['Low']
    data['High-Close'] = abs(data['High'] - data['Close'].shift(1))
    data['Low-Close'] = abs(data['Low'] - data['Close'].shift(1))
    data['True_Range'] = data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    data['ATR'] = data['True_Range'].rolling(window).mean()
    return data

# Function to calculate 50-SMA
def calculate_sma(data, window=50):
    data['SMA_50'] = data['Close'].rolling(window=window).mean()
    return data

# Function to generate buy/sell signals
def generate_signals(data):
    signals = []
    for i in range(len(data)):
        if (data['RSI'].iloc[i] < 40 and
            data['MACD'].iloc[i] > data['Signal_Line'].iloc[i] and
            data['Close'].iloc[i] > data['SMA_50'].iloc[i]):
            signals.append('Buy')
        elif (data['RSI'].iloc[i] > 60 and
              data['MACD'].iloc[i] < data['Signal_Line'].iloc[i] and
              data['Close'].iloc[i] < data['SMA_50'].iloc[i]):
            signals.append('Sell')
        else:
            signals.append('Hold')
    data['Signal'] = signals
    return data

# Backtest strategy
def backtest_strategy(data):
    balance = 10000
    position = None
    entry_price = 0
    trade_log = []

    for i in range(len(data)):
        if data['Signal'].iloc[i] == 'Buy' and position is None:
            position = 'Long'
            entry_price = data['Close'].iloc[i]
        elif position == 'Long':
            if data['Signal'].iloc[i] == 'Sell':
                exit_price = data['Close'].iloc[i]
                balance += exit_price - entry_price
                trade_log.append(exit_price - entry_price)
                position = None

    return balance, trade_log

# Live trading recommendations
def live_trading_recommendations(data):
    latest_signal = data['Signal'].iloc[-1]
    latest_price = data['Close'].iloc[-1]
    return latest_price, latest_signal

# Streamlit UI
st.title("Trading Strategy: Backtest and Live Trading")

# Select index
index_name = st.selectbox("Select Index", list(index_options.keys()))
symbol = index_options[index_name]

# Input for period and interval
period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y"], index=0)
interval = st.selectbox("Select Interval", ["1m", "5m", "15m", "1h", "1d"], index=1)

if st.button("Run Strategy"):
    with st.spinner("Fetching data..."):
        nifty_bank_data = fetch_5min_data(symbol, period)

    nifty_bank_data = calculate_rsi(nifty_bank_data)
    nifty_bank_data = calculate_macd(nifty_bank_data)
    nifty_bank_data = calculate_atr(nifty_bank_data)
    nifty_bank_data = calculate_sma(nifty_bank_data)
    nifty_bank_data = generate_signals(nifty_bank_data)

    if st.radio("Choose Mode:", ["Backtest", "Live Trading"]) == "Backtest":
        balance, trade_log = backtest_strategy(nifty_bank_data)
        st.write(f"Final Balance: {balance:.2f}")
        st.write("Trade Log:", trade_log)
    else:
        latest_price, latest_signal = live_trading_recommendations(nifty_bank_data)
        st.write(f"Latest Price: {latest_price:.2f}, Signal: {latest_signal}")

    if st.button("Stop"):
        st.write("Stopping the program...")

