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

        # Logging the values for debugging
        st.write(f"Date: {row.name}, Close: {row['Close']}, EMA_9: {row['EMA_9']}, "
                 f"EMA_21: {row['EMA_21']}, RSI: {row['RSI']}, "
                 f"MACD: {row['MACD']}, MACD_Signal: {row['MACD_Signal']}, "
                 f"Upper_BB: {row['Upper_BB']}, Lower_BB: {row['Lower_BB']}")

        # Buy signal
        if (row['Close'] < row['Lower_BB'] and 
            row['RSI'] < 35 and  # Loosened condition
            previous_row['MACD'] < previous_row['MACD_Signal'] and 
            row['MACD'] > row['MACD_Signal']):
            entry_price = row['Close']
            entry_date = row.name
            trades.append({'Entry_Date': entry_date, 'Entry_Price': entry_price, 'Trade_Result': 'Buy'})

        # Sell signal
        elif (row['Close'] > row['Upper_BB'] and 
              row['RSI'] > 65 and  # Loosened condition
              previous_row['MACD'] > previous_row['MACD_Signal'] and 
              row['MACD'] < row['MACD_Signal']):
            entry_price = row['Close']
            entry_date = row.name
            trades.append({'Entry_Date': entry_date, 'Entry_Price': entry_price, 'Trade_Result': 'Sell'})

    trades_df = pd.DataFrame(trades)
    return trades_df

# Function to fetch recent minute-level data
def fetch_recent_minute_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    return yf.download(ticker, start=start_date, end=end_date, interval='1m')

# Function for live trading
def live_trading(selected_ticker):
    stop_loss_points = 50
    target_points = 50
    run_live = st.session_state.get('run_live', True)

    while run_live:
        recent_data = fetch_recent_minute_data(selected_ticker)
        recent_data = compute_daily_indicators(recent_data)
        current_price = recent_data['Close'].iloc[-1]

        # Buy conditions
        if (current_price < recent_data['Lower_BB'].iloc[-1] and 
            recent_data['RSI'].iloc[-1] < 35 and  # Loosened condition
            recent_data['MACD'].iloc[-1] > recent_data['MACD_Signal'].iloc[-1]):
            entry_price = current_price
            entry_date = recent_data.index[-1]
            target_price = entry_price + target_points
            stop_loss_price = entry_price - stop_loss_points
            
            st.write(f"Buy Signal: Entry Price: {entry_price}, Target: {target_price}, Stop Loss: {stop_loss_price}")

            # Exit logic
            while run_live:
                time.sleep(60)  # Wait for the next minute to check exit conditions
                recent_data = fetch_recent_minute_data(selected_ticker)
                recent_data = compute_daily_indicators(recent_data)
                latest_price = recent_data['Close'].iloc[-1]

                if latest_price >= target_price:
                    st.write(f"Trade Result: Win, Exit Price: {latest_price} on {recent_data.index[-1]}")
                    break
                elif latest_price <= stop_loss_price:
                    st.write(f"Trade Result: Loss, Exit Price: {latest_price} on {recent_data.index[-1]}")
                    break
        
        # Sell conditions
        elif (current_price > recent_data['Upper_BB'].iloc[-1] and 
              recent_data['RSI'].iloc[-1] > 65 and  # Loosened condition
              recent_data['MACD'].iloc[-1] < recent_data['MACD_Signal'].iloc[-1]):
            entry_price = current_price
            entry_date = recent_data.index[-1]
            target_price = entry_price - target_points
            stop_loss_price = entry_price + stop_loss_points
            
            st.write(f"Sell Signal: Entry Price: {entry_price}, Target: {target_price}, Stop Loss: {stop_loss_price}")

            # Exit logic
            while run_live:
                time.sleep(60)  # Wait for the next minute to check exit conditions
                recent_data = fetch_recent_minute_data(selected_ticker)
                recent_data = compute_daily_indicators(recent_data)
                latest_price = recent_data['Close'].iloc[-1]

                if latest_price <= target_price:
                    st.write(f"Trade Result: Win, Exit Price: {latest_price} on {recent_data.index[-1]}")
                    break
                elif latest_price >= stop_loss_price:
                    st.write(f"Trade Result: Loss, Exit Price: {latest_price} on {recent_data.index[-1]}")
                    break

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
        st.write("Backtesting Results:")
        st.dataframe(trades_df)

        # Performance metrics
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['Trade_Result'] == 'Buy'])  # Adjust based on your logic
        losses = len(trades_df[trades_df['Trade_Result'] == 'Sell'])  # Adjust based on your logic
        
        accuracy = (wins / total_trades * 100) if total_trades > 0 else 0

        st.write(f"Total Trades: {total_trades}")
        st.write(f"Wins: {wins} ({accuracy:.2f}%)")
        st.write(f"Losses: {losses} (0.00%)")

elif mode == 'Live Trading':
    if 'run_live' not in st.session_state:
        st.session_state.run_live = True

    if st.button("Stop Live Trading"):
        st.session_state.run_live = False
        st.write("Live trading has been stopped.")

    if st.session_state.run_live:
        st.write("Live Trading Mode Activated. Check the console for real-time updates.")
        live_trading(selected_ticker)
