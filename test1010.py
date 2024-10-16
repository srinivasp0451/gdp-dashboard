import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import time

# Function to fetch recent 1-minute data for a given ticker
def fetch_recent_minute_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    return yf.download(ticker, start=start_date, end=end_date, interval='1m')

# Calculate technical indicators
def calculate_indicators(data):
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Middle_Band'] = data['Close'].rolling(window=20).mean()
    data['Upper_Band'] = data['Middle_Band'] + (data['Close'].rolling(window=20).std() * 2)
    data['Lower_Band'] = data['Middle_Band'] - (data['Close'].rolling(window=20).std() * 2)
    return data

# Identify multiple support and resistance levels
def identify_support_resistance(data):
    data['Support1'] = data['Low'].rolling(window=50).min()
    data['Resistance1'] = data['High'].rolling(window=50).max()
    pivots = (data['High'] + data['Low'] + data['Close']) / 3
    data['Pivot'] = pivots
    data['Support2'] = 2 * pivots - data['High']
    data['Resistance2'] = 2 * pivots - data['Low']
    data['Support3'] = data['Support1'].rolling(window=100).min()
    data['Resistance3'] = data['Resistance1'].rolling(window=100).max()
    data['Support4'] = data['Support2'].rolling(window=100).min()
    data['Resistance4'] = data['Resistance2'].rolling(window=100).max()
    return data

# Backtest the strategy
def backtest(data):
    initial_capital = 100000
    position = 0
    cash = initial_capital
    max_position_size = 10
    trades = []
    profit_trades = 0
    loss_trades = 0
    total_profit = 0
    total_loss = 0

    for i in range(1, len(data)):
        entry_condition = (
            data['RSI'].iloc[i] < 50 and
            data['Close'].iloc[i] < data['Lower_Band'].iloc[i] + 10 and
            data['MACD'].iloc[i] > data['Signal_Line'].iloc[i] and
            position < max_position_size
        )
        exit_condition = (
            data['RSI'].iloc[i] > 70 or
            data['Close'].iloc[i] > data['Upper_Band'].iloc[i] or
            data['MACD'].iloc[i] < data['Signal_Line'].iloc[i]
        )

        if entry_condition:
            position += 1
            cash -= data['Close'].iloc[i]
            trades.append({
                'entry_date': data.index[i],
                'entry_price': data['Close'].iloc[i],
                'logic': 'RSI < 50 and Close < Lower Band + 10 and MACD > Signal Line'
            })
        elif exit_condition and position > 0:
            position -= 1
            cash += data['Close'].iloc[i]
            exit_price = data['Close'].iloc[i]
            trades[-1].update({
                'exit_date': data.index[i],
                'exit_price': exit_price
            })
            profit_loss = exit_price - trades[-1]['entry_price']
            if profit_loss > 0:
                profit_trades += 1
                total_profit += profit_loss
            else:
                loss_trades += 1
                total_loss += abs(profit_loss)

    final_value = cash + (position * data['Close'].iloc[-1]) if position > 0 else cash
    return final_value

# Generate live trading signals
def live_trading(symbol):
    st.write("Starting live trading recommendations...")
    while True:
        data = fetch_recent_minute_data(symbol)
        if not data.empty:
            data = calculate_indicators(data)
            last_row = data.iloc[-1]
            entry_condition = (
                last_row['RSI'] < 50 and
                last_row['Close'] < last_row['Lower_Band'] + 10 and
                last_row['MACD'] > last_row['Signal_Line']
            )
            exit_condition = (
                last_row['RSI'] > 70 or
                last_row['Close'] > last_row['Upper_Band'] or
                last_row['MACD'] < last_row['Signal_Line']
            )
            if entry_condition:
                st.write("Recommendation: Buy")
            elif exit_condition:
                st.write("Recommendation: Sell")
            else:
                st.write("Recommendation: Hold")
        time.sleep(60)

# Main function
def main():
    st.title("Trading Strategy Dashboard")
    
    mode = st.radio("Select Mode", ("Backtesting", "Live Trading"))
    symbol = st.selectbox("Select Index", ['^NSEI', '^NSEBANK', '^BSESN', '^NSEMDCP', '^NSEBANKEX', '^NSEFIN'])

    if mode == "Backtesting":
        data = fetch_recent_minute_data(symbol)
        if data.empty:
            st.write("No data found for the specified ticker.")
            return

        data = calculate_indicators(data)
        data = identify_support_resistance(data)
        final_value = backtest(data)
        st.write(f"Final Portfolio Value: {final_value:.2f}")

    elif mode == "Live Trading":
        live_trading(symbol)

if __name__ == "__main__":
    main()
