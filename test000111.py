import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import time
import streamlit as st
import threading

# Constants
STOPLOSS = 10  # Default stop loss in points
#NIFTY50_TICKER = "^NSEBANK"  # Default Nifty50 index symbol
PERIOD = "1mo"  # Default period for backtesting
INTERVAL = "5m"  # Default interval for backtesting

# Fetch Data for Backtesting or Live
def fetch_data(ticker,period=PERIOD, interval=INTERVAL):
    data = yf.download(ticker, period=period, interval=interval)
    return data

# Simple Moving Average Crossover strategy for scalping
def apply_strategy(data,ma1,ma2,matype):
    if len(data) < 200:  # Ensure there's enough data for SMA calculation
        print("Not enough data to apply strategy (need at least 200 rows).")
        return data

    if(matype=="ema"):
        data['SMA_10'] = data['Close'].ewm(span=ma1, adjust=False).mean()
        data['SMA_50'] = data['Close'].ewm(span=ma2, adjust=False).mean()
    else:
        data['SMA_10'] = data['Close'].rolling(window=ma1).mean()  # 10-period SMA
        data['SMA_50'] = data['Close'].rolling(window=ma2).mean()  # 50-period SMA
    data.dropna(inplace=True)  # Remove NaN values after moving average calculations
    return data

# Backtesting Logic
def backtest_strategy(data, stoploss_points,target, index):
    global STOPLOSS
    STOPLOSS = stoploss_points  # Set stop loss based on user input
    
    trades = []
    in_position = False
    entry_price = 0
    entry_time = ""

    total_profit_points = 0
    total_loss_points = 0
    profit_trades = 0
    loss_trades = 0

    for i in range(1, len(data)):
        current_sma_10 = data['SMA_10'].iloc[i]  
        current_sma_50 = data['SMA_50'].iloc[i]
        prev_sma_10 = data['SMA_10'].iloc[i - 1]
        prev_sma_50 = data['SMA_50'].iloc[i - 1]
        current_close = data['Close'].iloc[i]

        if current_sma_10 > current_sma_50 and prev_sma_10 <= prev_sma_50 and not in_position:
            entry_price = current_close.values[0]
            entry_time = data.index[i]
            in_position = True
            #st.write(f"Buy Signal at {entry_time} | Price: {entry_price}")

        if in_position:
            stop_loss_level = entry_price - STOPLOSS
            take_profit_level = entry_price + target

            if current_close.values[0] <= stop_loss_level or current_close.values[0] >= take_profit_level:
                exit_price = current_close.values[0]
                exit_time = data.index[i]
                profit_loss_points = exit_price - entry_price

                if profit_loss_points > 0:
                    profit_trades += 1
                    total_profit_points += profit_loss_points
                    trades.append(('Buy', entry_time, entry_price, 'Sell', exit_time, exit_price, profit_loss_points))
                else:
                    loss_trades += 1
                    total_loss_points += abs(profit_loss_points)
                    trades.append(('Buy', entry_time, entry_price, 'Sell', exit_time, exit_price, profit_loss_points))

                in_position = False

    total_trades = profit_trades + loss_trades
    accuracy = (profit_trades / total_trades) * 100 if total_trades else 0

    st.write(f"\nBacktest Results for {index}:")
    for trade in trades:
        st.write(f"{trade[0]} at {trade[1]} Price: {trade[2]} -> {trade[3]} at {trade[4]} Price: {trade[5]} P/L Points: {trade[6]}")

    st.write(f"\nTotal Trades: {total_trades}")
    st.write(f"Profit Trades: {profit_trades}")
    st.write(f"Loss Trades: {loss_trades}")
    st.write(f"Total Profit Points: {total_profit_points}")
    st.write(f"Total Loss Points: {total_loss_points}")
    st.write(f"Net Profit/Loss Points: {total_profit_points - total_loss_points}")
    st.write(f"Accuracy: {accuracy:.2f}%")

# Live Trading Recommendation
def live_trading_recommendation(index, stoploss_points,ma1,ma2,matype):
    global STOPLOSS
    STOPLOSS = stoploss_points  # Set stop loss based on user input
    
    while True:
        data = fetch_data(index,period="1d", interval="5m")  # 5 days, 5-minute intervals
        if len(data) < 2:
            st.write("Not enough data to generate recommendation. Waiting for more data...")
            time.sleep(120)  # Sleep for 2 minutes before trying again
            continue

        data = apply_strategy(data,ma1,ma2,matype)

        if len(data) < 2:
            st.write("Not enough data after applying strategy. Waiting for more data...")
            time.sleep(120)
            continue

        st.write("before")
        current_data = data.iloc[-1]
        prev_data = data.iloc[-2]
        st.write(current_data)

        current_sma_10 = current_data['SMA_10']
        current_sma_50 = current_data['SMA_50']
        current_close = current_data['Close']

        prev_sma_10 = prev_data['SMA_10']
        prev_sma_50 = prev_data['SMA_50']
        st.write("hello")

        if current_sma_10.values[0] > current_sma_50.values[0] and prev_sma_10.values[0] <= prev_sma_50.values[0]:
            st.write(f"Recommendation: BUY at {current_close.values[0]} | {datetime.now()}")
        elif current_sma_10.values[0] < current_sma_50.values[0] and prev_sma_10.values[0] >= prev_sma_50.values[0]:
            st.write(f"Recommendation: SELL at {current_close.values[0]} | {datetime.now()}")
        else:
            st.write(f"Recommendation: HOLD at {current_close.values[0]} | {datetime.now()}")

        time.sleep(60)  # Sleep for 1 minute before trying again
        global stop_threads
        if stop_threads:
            break

# Main Streamlit UI
def main():
    st.title("Stock Strategy Backtesting / Live Trading")

    # Select the index
    index = st.selectbox("Select Index", ["^NSEBANK", "^NSEI", "^BSESN", "NIFTY_FIN_SERVICE.NS", "^NSEMDCP50", "BSE-BANK.BO","BTC-USD","Other"],index=0)
    # Variable to store the custom index
    custom_index = index

    # If "Other" is selected, show the text input for custom index
    if index == "Other":
        custom_index = st.text_input("Enter Custom Index")
        index = custom_index

    # Select the strategy type
    strategy_type = st.selectbox("Select Strategy Type", ["Backtesting", "Live Trading"], index=0)
    matype = st.selectbox("Select MA Type", ["ema", "sma"], index=1)
    ma1 = st.number_input("Enter MA1",min_value=1,value=100)
    ma2 = st.number_input("Enter MA2",min_value=1,value=200)


    # Backtesting settings
    if strategy_type == "Backtesting":
        period = st.selectbox("Select Period", ["1d", "5d", "1mo", "3mo", "6mo","1y","2y","5y"], index=2)
        interval = st.selectbox("Select Interval", ["1m", "2m", "5m", "15m", "30m", "60m","1d","1wk"], index=3)
        target = st.number_input("Enter Target (Points)", min_value=1, value=10)
        stoploss_points = st.number_input("Enter Stop Loss (Points)", min_value=1, value=10)
        
        # Button to run backtest
        if st.button("Run Backtest"):
            data = fetch_data(index,period=period, interval=interval)
            data = apply_strategy(data,ma1,ma2,matype)
            backtest_strategy(data, stoploss_points, target, index)

    # Live trading settings
    elif strategy_type == "Live Trading":
        stoploss_points = st.number_input("Enter Stop Loss (Points)", min_value=1, value=10)

        # Button to start live trading
        if st.button("Start Live Trading"):
            st.write("Starting live trading...")
            t1 = threading.Thread(target=live_trading_recommendation, args=(index, stoploss_points,ma1,ma2,matype), daemon=True).start()
        if st.button("stop run"):
            #stop_threads=False
            stop_threads = True
            #t1.join()
        

if __name__ == "__main__":
    main()
