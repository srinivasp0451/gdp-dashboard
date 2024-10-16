import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import time

# Define the tickers for various indices
tickers = {
    "Nifty 50": '^NSEI',
    "Bank Nifty": '^NSEBANK',
    "Fin Nifty": '^NSEFIN',
    "Midcap Nifty": '^NSEMCAP',
    "Sensex": '^BSESN',
    "Bankex": '^NSEBANKEX'
}

# Function to fetch recent minute-level data (only last 7 days)
def fetch_recent_minute_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    return yf.download(ticker, start=start_date, end=end_date, interval='1m')

# Function to fetch daily data for the specified period
def fetch_daily_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

# Function to run backtesting
def backtest_strategy(selected_ticker, start_date, end_date):
    # Fetch daily historical data
    daily_data = fetch_daily_data(selected_ticker, start_date, end_date)

    # Fetch recent minute-level historical data
    recent_minute_data = fetch_recent_minute_data(selected_ticker)

    if recent_minute_data.index.tzinfo is None:
        recent_minute_data.index = recent_minute_data.index.tz_localize('Asia/Kolkata')

    daily_data['Price_Change'] = daily_data['Close'].diff()
    significant_moves = daily_data[abs(daily_data['Price_Change']) >= 50]

    trades = []
    stop_loss_points = 50  # 50 points stop loss
    target_points = 50  # 50 points target

    if significant_moves.empty:
        return "No clear signals found."

    for i in range(len(significant_moves) - 1):
        entry_price = significant_moves['Close'].iloc[i]
        entry_date = significant_moves.index[i]
        if entry_date.tzinfo is None:
            entry_date = entry_date.tz_localize('Asia/Kolkata')

        trade_data = recent_minute_data[recent_minute_data.index >= entry_date]
        target_price = entry_price + target_points
        stop_loss_price = entry_price - stop_loss_points

        exit_price = None
        trade_result = None
        for time, row in trade_data.iterrows():
            if row['Close'] >= target_price:
                exit_price = target_price
                trade_result = 'Win'
                break
            elif row['Close'] <= stop_loss_price:
                exit_price = stop_loss_price
                trade_result = 'Loss'
                break
        if trade_result is None:
            exit_price = trade_data['Close'].iloc[-1]
            trade_result = 'No Hit'

        trades.append({
            'Entry_Date': entry_date,
            'Entry_Price': entry_price,
            'Exit_Date': trade_data.index[-1],
            'Exit_Price': exit_price,
            'Trade_Result': trade_result,
            'Points_Captured': exit_price - entry_price
        })

    trades_df = pd.DataFrame(trades)
    return trades_df

# Function for live trading (simplified version)
def live_trading(selected_ticker):
    stop_loss_points = 50  # 50 points stop loss
    target_points = 50  # 50 points target
    while True:
        recent_minute_data = fetch_recent_minute_data(selected_ticker)
        current_price = recent_minute_data['Close'].iloc[-1]
        entry_price = current_price
        entry_date = recent_minute_data.index[-1]
        target_price = entry_price + target_points
        stop_loss_price = entry_price - stop_loss_points

        st.write(f"Entry Price: {entry_price}, Target: {target_price}, Stop Loss: {stop_loss_price}")

        time.sleep(60)  # Wait for the next minute to check exit conditions
        recent_minute_data = fetch_recent_minute_data(selected_ticker)
        latest_price = recent_minute_data['Close'].iloc[-1]

        if latest_price >= target_price:
            st.write(f"Trade Result: Win, Exit Price: {target_price}")
            break
        elif latest_price <= stop_loss_price:
            st.write(f"Trade Result: Loss, Exit Price: {stop_loss_price}")
            break
        else:
            st.write(f"No Hit, Latest Price: {latest_price}")

# Streamlit UI
st.title("Trading Strategy Application")

selected_index = st.selectbox("Select Index", list(tickers.keys()))
selected_ticker = tickers[selected_index]

mode = st.radio("Select Mode", ('Live Trading', 'Backtesting'))

if mode == 'Backtesting':
    # Date selection for backtesting
    start_date = st.date_input("Select Start Date", datetime(2020, 1, 1))
    end_date = st.date_input("Select End Date", datetime.now())

    if st.button("Run Backtest"):
        trades_df = backtest_strategy(selected_ticker, start_date, end_date)
        if isinstance(trades_df, str):  # Check if message string is returned
            st.write(trades_df)  # Display no clear signals found message
        else:
            st.write("Backtesting Results:")
            st.dataframe(trades_df)

            # Calculate strategy performance
            total_trades = len(trades_df)
            wins = len(trades_df[trades_df['Trade_Result'] == 'Win'])
            losses = len(trades_df[trades_df['Trade_Result'] == 'Loss'])
            no_hits = len(trades_df[trades_df['Trade_Result'] == 'No Hit'])

            st.write(f"Total Trades: {total_trades}")
            st.write(f"Wins: {wins} ({(wins / total_trades * 100) if total_trades > 0 else 0:.2f}%)")
            st.write(f"Losses: {losses} ({(losses / total_trades * 100) if total_trades > 0 else 0:.2f}%)")
            st.write(f"No Hit: {no_hits} ({(no_hits / total_trades * 100) if total_trades > 0 else 0:.2f}%)")

elif mode == 'Live Trading':
    st.write("Live Trading Mode Activated. Check the console for real-time updates.")
    live_trading(selected_ticker)
# Button to stop live trading
if st.button("Stop Live Trading"):
    st.session_state.stop_clicked 
