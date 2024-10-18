import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# Function to fetch index data
def fetch_data(ticker='^NSEI', period='2d', interval='5m'):
    data = yf.download(ticker, period=period, interval=interval)
    return data

# Function to calculate indicators
def calculate_indicators(data):
    data['Short_MA'] = data['Close'].rolling(window=5).mean()
    data['Long_MA'] = data['Close'].rolling(window=20).mean()
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    data['Middle_Band'] = data['Close'].rolling(window=20).mean()
    data['Upper_Band'] = data['Middle_Band'] + (data['Close'].rolling(window=20).std() * 2)
    data['Lower_Band'] = data['Middle_Band'] - (data['Close'].rolling(window=20).std() * 2)

    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Function to generate entry/exit recommendations
def generate_recommendation(data):
    latest = data.iloc[-1]
    entry_price = latest['Close']
    target_points = 50  # Customizable target points
    stop_loss_points = 30  # Customizable stop loss points
    
    if latest['Close'] > latest['Short_MA'] and latest['Short_MA'] > latest['Long_MA'] and latest['RSI'] < 70:
        reason = "Bullish crossover, RSI indicates potential upside."
        recommendation = f"Currently at {latest['Close']}, \n\n Buy at index {entry_price:.2f}, target {entry_price + target_points:.2f}, stop loss {entry_price - stop_loss_points:.2f}."
        return recommendation, reason

    elif latest['Close'] < latest['Short_MA'] and latest['Short_MA'] < latest['Long_MA'] and latest['RSI'] > 30:
        reason = "Bearish crossover, RSI indicates potential downside."
        recommendation = f"Currently at {latest['Close']}, \n\n Sell at index {entry_price:.2f}, target {entry_price - target_points:.2f}, stop loss {entry_price + stop_loss_points:.2f}."
        return recommendation, reason

    return "Hold", "No clear signal."

# Streamlit app layout
st.title("Index Trading Strategy")
st.sidebar.header("Select Index")
index_options = {
    "Bank Nifty": "^NSEBANK",
    "Nifty 50": "^NSEI",
    "FinNifty": "NIFTY_FIN_SERVICE.NS",
    
    "Sensex": "^BSESN",
    "Midcap Nifty": "NIFTY_MID_SELECT.NS",
    "BANKEX" : "BSE-BANK.BO"
}


#indices = {
#    "NIFTY 50": "^NSEI",
#    "BANK NIFTY": "^NSEBANK",
#    "FIN NIFTY": "NIFTY_FIN_SERVICE.NS",
#    "MIDCAP NIFTY": "NIFTY_MID_SELECT.NS",
#    "SENSEX": "^BSESN",
#    "BANKEX": "BSE-BANK.BO",
#    "ALL": "ALL"
#}

selected_index = st.sidebar.selectbox("Choose an index:", list(index_options.keys()))
ticker = index_options[selected_index]

if st.sidebar.button("Run Strategy"):
    with st.spinner("Fetching data..."):
        data = fetch_data(ticker)
        if len(data) < 30:  # Ensure there are enough data points
            st.error("Not enough data available.")
        else:
            calculate_indicators(data)
            recommendation, reason = generate_recommendation(data)
            st.subheader("Recommendation")
            st.write(recommendation)
            st.write("Reason:", reason)

            # Plotting
            plt.figure(figsize=(14, 7))
            plt.plot(data['Close'], label='Close Price', color='blue')
            plt.plot(data['Short_MA'], label='Short Moving Average', color='orange')
            plt.plot(data['Long_MA'], label='Long Moving Average', color='red')
            plt.plot(data['Upper_Band'], label='Upper Bollinger Band', color='green')
            plt.plot(data['Lower_Band'], label='Lower Bollinger Band', color='brown')
            plt.title(f'{selected_index} Trading Strategy with Indicators')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            #st.pyplot(plt)

st.sidebar.button("Stop Program", on_click=lambda: st.stop())
