import yfinance as yf
import pandas as pd
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

# Function to fetch daily data
def fetch_daily_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

# Function to compute indicators for daily data
def compute_daily_indicators(df):
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate Bollinger Bands
    df['Middle_BB'] = df['Close'].rolling(window=20).mean()
    df['Upper_BB'] = df['Middle_BB'] + (df['Close'].rolling(window=20).std() * 2)
    df['Lower_BB'] = df['Middle_BB'] - (df['Close'].rolling(window=20).std() * 2)
    
    return df

# Backtesting function
def backtest_strategy(selected_ticker, start_date, end_date):
    daily_data = fetch_daily_data(selected_ticker, start_date, end_date)
    daily_data = compute_daily_indicators(daily_data)

    trades = []

    for i in range(1, len(daily_data)):
        row = daily_data.iloc[i]
        previous_row = daily_data.iloc[i - 1]

        # Buy signal
        if (row['Close'] < row['Lower_BB'] and 
            row['RSI'] < 35 and 
            previous_row['MACD'] < previous_row['MACD_Signal'] and 
            row['MACD'] > row['MACD_Signal']):
            entry_price = row['Close']
            entry_date = row.name
            trades.append({'Entry_Date': entry_date, 'Entry_Price': entry_price, 'Trade_Result': 'Buy', 'Exit_Price': None, 'Exit_Date': None, 'Points': None})

        # Sell signal
        elif (row['Close'] > row['Upper_BB'] and 
              row['RSI'] > 65 and 
              previous_row['MACD'] > previous_row['MACD_Signal'] and 
              row['MACD'] < row['MACD_Signal']):
            entry_price = row['Close']
            entry_date = row.name
            trades.append({'Entry_Date': entry_date, 'Entry_Price': entry_price, 'Trade_Result': 'Sell', 'Exit_Price': None, 'Exit_Date': None, 'Points': None})

    # Simulate exits for trades
    for trade in trades:
        entry_price = trade['Entry_Price']
        entry_date = trade['Entry_Date']
        exit_price = None
        exit_date = None

        recent_data = fetch_recent_minute_data(selected_ticker)
        target_price = entry_price + 50 if trade['Trade_Result'] == 'Buy' else entry_price - 50
        stop_loss_price = entry_price - 50 if trade['Trade_Result'] == 'Buy' else entry_price + 50
        
        for time_index, row in recent_data.iterrows():
            if (trade['Trade_Result'] == 'Buy' and row['Close'] >= target_price) or (trade['Trade_Result'] == 'Sell' and row['Close'] <= target_price):
                exit_price = target_price
                exit_date = time_index
                break
            elif (trade['Trade_Result'] == 'Buy' and row['Close'] <= stop_loss_price) or (trade['Trade_Result'] == 'Sell' and row['Close'] >= stop_loss_price):
                exit_price = stop_loss_price
                exit_date = time_index
                break
        
        # Calculate points captured
        if exit_price is not None:
            trade['Exit_Price'] = exit_price
            trade['Exit_Date'] = exit_date
            trade['Points'] = exit_price - entry_price
    
    trades_df = pd.DataFrame(trades)
    return trades_df

# Function to fetch recent minute-level data
def fetch_recent_minute_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    return yf.download(ticker, start=start_date, end=end_date, interval='1m')

# Streamlit UI
st.title("Trading Strategy Application")

selected_index = st.selectbox("Select Index", list(tickers.keys()))
selected_ticker = tickers[selected_index]

mode = st.radio("Select Mode", ('Backtesting', 'Live Trading'))

if mode == 'Backtesting':
    start_date = st.date_input("Select Start Date", datetime(2020, 1, 1))
    end_date = st.date_input("Select End Date", datetime.now())

    if st.button("Run Backtest"):
        trades_df = backtest_strategy(selected_ticker, start_date, end_date)
        
        if trades_df.empty:
            st.write("No trades were generated during backtesting.")
        else:
            # Display trades as a scrollable DataFrame
            st.write("Backtesting Results:")
            st.dataframe(trades_df)

            # Summary of results
            total_trades = len(trades_df)
            total_profit = trades_df['Points'].sum() if not trades_df['Points'].isnull().all() else 0
            total_loss = trades_df['Points'].dropna().where(lambda x: x < 0).sum()
            total_profitable_trades = len(trades_df[trades_df['Points'] > 0])
            total_loss_trades = len(trades_df[trades_df['Points'] <= 0])
            accuracy = (total_profitable_trades / total_trades * 100) if total_trades > 0 else 0

            st.write("Summary of Results:")
            st.write(f"Total Trades: {total_trades}")
            st.write(f"Total Profit: {total_profit:.2f}")
            st.write(f"Total Loss: {total_loss:.2f}")
            st.write(f"Total Profitable Trades: {total_profitable_trades}")
            st.write(f"Total Loss Trades: {total_loss_trades}")
            st.write(f"Accuracy: {accuracy:.2f}%")

elif mode == 'Live Trading':
    if 'run_live' not in st.session_state:
        st.session_state.run_live = True

    if st.button("Stop Live Trading"):
        st.session_state.run_live = False
        st.write("Live trading has been stopped.")

    if st.session_state.run_live:
        st.write("Live Trading Mode Activated. Check the console for real-time updates.")
        live_trading(selected_ticker)
