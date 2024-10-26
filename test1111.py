import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Function to fetch market data
def fetch_data(symbol, start_date, end_date, interval='5m'):
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    return data

# Calculate moving averages
def calculate_moving_averages(data, short_window=5, long_window=15):
    data['EMA_short'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_long'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    return data

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

# Define the scalping strategy
def scalping_strategy(data, stop_loss_points):
    data = calculate_moving_averages(data)
    data = calculate_rsi(data)

    data['Signal'] = 0
    data.loc[(data['EMA_short'] > data['EMA_long']) & (data['RSI'] < 35), 'Signal'] = 1  # Buy signal
    data.loc[(data['EMA_short'] < data['EMA_long']) & (data['RSI'] > 65), 'Signal'] = -1  # Sell signal

    trades = []
    position = None

    for index in range(len(data)):
        row = data.iloc[index]
        close_price = row['Close'].item()
        signal = int(row['Signal'].item())

        # Entry condition
        if position is None and signal == 1:
            position = {
                'entry_price': close_price,
                'entry_date': row.name,
                'reason': 'EMA short crossed above EMA long and RSI < 35'
            }
            stop_loss = close_price - stop_loss_points

        # Exit condition
        elif position is not None:
            close_price_scalar = float(close_price)

            # Adjust stop-loss to lock in profits
            if close_price_scalar > position['entry_price']:
                stop_loss = max(stop_loss, close_price_scalar - stop_loss_points)

            if close_price_scalar <= stop_loss or signal == -1:
                exit_price = close_price
                points = float(exit_price - position['entry_price'])
                trades.append({
                    'entry_level': float(position['entry_price']),
                    'entry_date': position['entry_date'],
                    'exit_level': float(exit_price),
                    'exit_date': row.name,
                    'reason': 'Hit Stop Loss or Signal to Sell',
                    'points': points
                })
                position = None

    return trades

# Analyze trades
def analyze_trades(trades):
    total_trades = len(trades)
    profit_trades = sum(1 for trade in trades if trade['points'] > 0)
    loss_trades = total_trades - profit_trades
    total_profit_points = sum(trade['points'] for trade in trades if trade['points'] > 0)
    total_loss_points = sum(abs(trade['points']) for trade in trades if trade['points'] < 0)
    accuracy = (profit_trades / total_trades * 100) if total_trades > 0 else 0

    return {
        'total_trades': total_trades,
        'profit_trades': profit_trades,
        'loss_trades': loss_trades,
        'total_profit_points': total_profit_points,
        'total_loss_points': total_loss_points,
        'accuracy': accuracy
    }

# Main function for Streamlit app
def main():
    st.title("Options Scalping Strategy")
    
    # User inputs

    indices = {
    "Nifty 50": "^NSEI",
    "Bank Nifty": "^NSEBANK",
    "Sensex": "^BSESN",
    "Midcap Nifty": "^NSEMDCP",
    "Fin Nifty": "^NSEFIN",
    }

    #symbol = st.selectbox("Select an index", list(indices.keys()), index=1)  # Default to Bank Nifty

    symbol = st.selectbox("Select Index", ["^NSEI", "^NSEBANK","^NSEFIN","^NSEMDCP","^BSESN" ], index=1)  # Default is Bank Nifty
    backtest_days = st.number_input("Backtest Days", min_value=1, max_value=90, value=58)  # Default 58
    #interval = st.selectbox("Select Interval", ["1m", "2m", "5m", "10m", "15m", "30m", "60m"], index=2)  # Default 5m
    interval = "5m"
    stop_loss_points = st.selectbox("Select Stop Loss Points", [5,10, 15, 20,25,30,35,40,45,50], index=2)  # Default 20
    mode = st.selectbox("Select Mode", ["Backtest", "Live Trading"], index=0)  # Default Backtest

    if st.button("Run Strategy"):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=backtest_days)

        data = fetch_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), interval)

        if len(data) < 10:
            st.error("Not enough data to analyze.")
        else:
            trades = scalping_strategy(data, stop_loss_points)
            results = analyze_trades(trades)

            # Display results
            st.success(f"Total Trades: {results['total_trades']}")
            st.success(f"Profit Trades: {results['profit_trades']}")
            st.success(f"Loss Trades: {results['loss_trades']}")
            st.success(f"Total Profit Points: {results['total_profit_points']}")
            st.success(f"Total Loss Points: {results['total_loss_points']}")
            st.success(f"Accuracy: {results['accuracy']:.2f}%")

            # Print trade logs
            for trade in trades:
                st.write(f"Entry Level: {trade['entry_level']} on {trade['entry_date']} (Reason: {trade['reason']})")
                st.write(f"Exit Level: {trade['exit_level']} on {trade['exit_date']} (Points: {trade['points']})")

    if mode == "Live Trading":
        if st.button("Start Live Trading"):
            # Placeholder to store live trades
            live_trades = []
            while True:
                # Fetch new data for the last hour
                live_data = fetch_data(symbol, datetime.now() - timedelta(minutes=60), datetime.now(), interval)
                if len(live_data) < 10:
                    st.error("Not enough data for live trading.")
                    break

                new_trades = scalping_strategy(live_data, stop_loss_points)
                live_trades.extend(new_trades)

                # Display live trade recommendations
                if new_trades:
                    for trade in new_trades:
                        st.write(f"Live Recommendation: Buy at {trade['entry_level']} on {trade['entry_date']}")
                        st.write(f"Target Exit: {trade['exit_level']} on {trade['exit_date']} (Points: {trade['points']})")

                # Sleep for a short duration before fetching new data
                time.sleep(60)  # Delay for 60 seconds

                # Check if user has clicked the stop button
                if st.button("Stop Live Trading"):
                    st.write("Live trading stopped.")
                    break

if __name__ == "__main__":
    main()
