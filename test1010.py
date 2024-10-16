import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import streamlit as st

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

# Backtest the strategy with detailed logging
def backtest(data, stop_loss_pct=0.02, target_pct=0.05, exit_threshold=10):
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

        if position > 0:
            entry_price = trades[-1]['entry_price']
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            target_price = entry_price * (1 + target_pct)

            exit_condition = (
                data['Close'].iloc[i] <= stop_loss_price or
                data['Close'].iloc[i] >= target_price or
                data['MACD'].iloc[i] < data['Signal_Line'].iloc[i] or
                data['Close'].iloc[i] <= entry_price - exit_threshold
            )

            if exit_condition:
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

        if entry_condition:
            position += 1
            cash -= data['Close'].iloc[i]
            trades.append({
                'entry_date': data.index[i],
                'entry_price': data['Close'].iloc[i],
                'logic': 'RSI < 50 and Close < Lower Band + 10 and MACD > Signal Line'
            })

    final_value = cash + (position * data['Close'].iloc[-1]) if position > 0 else cash

    total_trades = profit_trades + loss_trades
    accuracy = (profit_trades / total_trades * 100) if total_trades > 0 else 0

    return final_value, trades, profit_trades, loss_trades, total_profit, total_loss, accuracy

# Generate live trading recommendations
def live_trading(symbol):
    st.write(f"Starting live trading for {symbol}...")
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
                st.write("Recommendation: **Buy**")
            elif exit_condition:
                st.write("Recommendation: **Sell**")
            else:
                st.write("Recommendation: **Hold**")

        time.sleep(60)  # Refresh every minute

# Main function
def main():
    st.title("Trading Strategy Dashboard")

    option = st.selectbox("Choose an option", ["Backtesting", "Live Trading"])
    symbol = st.selectbox("Select Index", ['^NSEI', '^NSEBANK', '^BSESN', '^NSEMDCP', '^NSEBANKEX', '^NSEFIN'])

    if option == "Backtesting":
        if st.button("Run Backtest"):
            data = fetch_recent_minute_data(symbol)
            if data.empty:
                st.write("No data found for the specified ticker.")
                return

            data = calculate_indicators(data)
            final_value, trades, profit_trades, loss_trades, total_profit, total_loss, accuracy = backtest(data, stop_loss_pct=0.02, target_pct=0.05, exit_threshold=10)

            st.write(f"Initial Portfolio Value: 100000")
            st.write(f"Final Portfolio Value: {final_value:.2f}")
            st.write(f"Total Trades: {profit_trades + loss_trades}")
            st.write(f"Profitable Trades: {profit_trades}")
            st.write(f"Loss Trades: {loss_trades}")
            st.write(f"Total Profit: {total_profit:.2f}")
            st.write(f"Total Loss: {total_loss:.2f}")
            st.write(f"Accuracy: {accuracy:.2f}%")

            for trade in trades:
                exit_date = trade.get('exit_date', 'N/A')
                exit_price = trade.get('exit_price', 'N/A')
    
                if exit_price != 'N/A':
                    exit_price = f"{exit_price:.2f}"

                st.write(f"Trade Entry: {trade['entry_date']} at {trade['entry_price']:.2f}, "
                         f"Exit: {exit_date} at {exit_price}, "
                         f"Logic: {trade['logic']}")

    elif option == "Live Trading":
        live_trading(symbol)

if __name__ == "__main__":
    main()
