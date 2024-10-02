import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, time
import numpy as np

# Function to fetch data
def fetch_data(ticker, interval='5m', period='5d'):
    try:
        data = yf.download(ticker, interval=interval, period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Complex scalping strategy
def complex_scalping_strategy(data):
    try:
        data['UpperBB'], data['MiddleBB'], data['LowerBB'] = ta.bbands(data['Close'], length=20).T
        data['MA_Short'] = ta.sma(data['Close'], length=9)
        data['MA_Long'] = ta.sma(data['Close'], length=21)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
        data['MACD'] = macd['MACD_12_26_9']
        data['MACD_Signal'] = macd['MACDs_12_26_9']
        data['Signal'] = 0
        buy_condition = (data['RSI'] < 30) & (data['MACD'] > data['MACD_Signal']) & (data['Close'] <= data['LowerBB'])
        sell_condition = (data['RSI'] > 70) & (data['MACD'] < data['MACD_Signal']) & (data['Close'] >= data['UpperBB'])
        data['Signal'] = np.where(buy_condition, 1, np.where(sell_condition, -1, 0))
        data['Position'] = data['Signal'].diff()
        return data
    except Exception as e:
        st.error(f"Error in strategy calculation: {e}")
        return None

# Backtest function
def backtest_strategy(data, stop_loss=0.005, take_profit=0.01):
    try:
        initial_balance = 100000
        balance = initial_balance
        positions = []
        win_count = 0
        loss_count = 0

        for index, row in data.iterrows():
            if row['Position'] == 1:
                entry_price = row['Close']
                target_price = entry_price * (1 + take_profit)
                stop_price = entry_price * (1 - stop_loss)
                positions.append(entry_price)
            if row['Position'] == -1 and positions:
                exit_price = row['Close']
                entry_price = positions.pop(0)
                profit = exit_price - entry_price
                if profit > 0:
                    win_count += 1
                else:
                    loss_count += 1
                balance += profit * 10

        net_profit = balance - initial_balance
        total_trades = win_count + loss_count
        accuracy = (win_count / total_trades) * 100 if total_trades > 0 else 0
        return net_profit, accuracy, total_trades
    except Exception as e:
        st.error(f"Error during backtesting: {e}")
        return None, None, None

# Check if it's trading hours
def is_trading_hours():
    now = datetime.now().time()
    market_open = time(9, 15)
    market_close = time(15, 30)
    return market_open <= now <= market_close

# Live trading (dummy)
def live_trading(data):
    try:
        if is_trading_hours():
            # Simulate live trading with the same strategy
            return backtest_strategy(data)  # Reuse backtest logic for demo purposes
        else:
            st.warning("Market is closed. Live trading only works during trading hours (9:15 AM to 3:30 PM).")
            return None, None, None
    except Exception as e:
        st.error(f"Error during live trading: {e}")
        return None, None, None

# Streamlit UI
st.title("Scalping Strategy for Nifty50 and Other Indices")

# Dropdown for index selection
indices = {
    'Nifty 50': '^NSEI',
    'Bank Nifty': '^NSEBANK',
    'Sensex': '^BSESN',
    'Midcap Nifty': 'NIFTY_MID_SELECT.NS',
    'Fin Nifty': 'NIFTY_FIN_SERVICE.NS',
    'Bankex': 'BSE-BANK.BO'
}

selected_index = st.selectbox("Select Index", list(indices.keys()))

# Dropdown for backtesting or live trading
trade_mode = st.selectbox("Mode", ["Backtesting", "Live Trading"])

# Run the selected strategy
if st.button("Run Strategy"):
    data = fetch_data(indices[selected_index])
    
    if data is not None:
        data = complex_scalping_strategy(data)

        if data is not None:
            if trade_mode == "Backtesting":
                net_profit, accuracy, total_trades = backtest_strategy(data)
            else:
                net_profit, accuracy, total_trades = live_trading(data)

            if net_profit is not None:
                st.write(f"**Net Profit:** Rs {net_profit:.2f}")
                st.write(f"**Accuracy:** {accuracy:.2f}%")
                st.write(f"**Total Trades:** {total_trades}")
            
            # Plot the strategy
            st.line_chart(data['Close'])
            plt.figure(figsize=(14,8))
            plt.plot(data['Close'], label='Close Price')
            plt.plot(data['UpperBB'], linestyle='--', color='r')
            plt.plot(data['LowerBB'], linestyle='--', color='g')
            plt.plot(data['MA_Short'], color='b')
            plt.plot(data['MA_Long'], color='purple')
            plt.plot(data[data['Position'] == 1].index, data['Close'][data['Position'] == 1], '^', markersize=12, color='g', lw=0, label='Buy Signal')
            plt.plot(data[data['Position'] == -1].index, data['Close'][data['Position'] == -1], 'v', markersize=12, color='r', lw=0, label='Sell Signal')
            plt.title(f'{selected_index} - Scalping Strategy')
            st.pyplot(plt)
        else:
            st.error("Error calculating strategy signals.")
else:
    st.write("Click on 'Run Strategy' to backtest or run live trading.")
