import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time

# Function to fetch data
def fetch_data(ticker, interval='5m', period='5d'):
    try:
        data = yf.download(ticker, interval=interval, period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Moving Average (SMA)
def sma(data, window):
    return data['Close'].rolling(window=window).mean()

# Relative Strength Index (RSI)
def rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Bollinger Bands
def bollinger_bands(data, window=20, no_of_std=2):
    rolling_mean = sma(data, window)
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * no_of_std)
    lower_band = rolling_mean - (rolling_std * no_of_std)
    return upper_band, rolling_mean, lower_band

# MACD
def macd(data, fast_period=12, slow_period=26, signal_period=9):
    exp1 = data['Close'].ewm(span=fast_period, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow_period, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, signal_line

# Scalping strategy with manual indicator calculation
def scalping_strategy(data, ma_short_len, ma_long_len, rsi_len, rsi_lower, rsi_upper, bb_len, stop_loss, take_profit):
    try:
        # Calculate indicators manually
        data['MA_Short'] = sma(data, ma_short_len)
        data['MA_Long'] = sma(data, ma_long_len)
        data['RSI'] = rsi(data, rsi_len)
        data['UpperBB'], data['MiddleBB'], data['LowerBB'] = bollinger_bands(data, bb_len)
        data['MACD'], data['MACD_Signal'] = macd(data)

        # Buy and sell conditions
        buy_condition = (data['RSI'] < rsi_lower) & (data['MACD'] > data['MACD_Signal']) & (data['Close'] <= data['LowerBB'])
        sell_condition = (data['RSI'] > rsi_upper) & (data['MACD'] < data['MACD_Signal']) & (data['Close'] >= data['UpperBB'])

        data['Signal'] = np.where(buy_condition, 1, np.where(sell_condition, -1, 0))
        data['Position'] = data['Signal'].diff()

        return data
    except Exception as e:
        st.error(f"Error in strategy calculation: {e}")
        return None

# Backtesting function
def backtest_strategy(data, stop_loss=0.005, take_profit=0.01):
    try:
        initial_balance = 100000
        balance = initial_balance
        positions = []
        win_count = 0
        loss_count = 0

        for index, row in data.iterrows():
            if row['Position'] == 1:  # Buy
                entry_price = row['Close']
                target_price = entry_price * (1 + take_profit)
                stop_price = entry_price * (1 - stop_loss)
                positions.append(entry_price)
            if row['Position'] == -1 and positions:  # Sell
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

# Optimized strategy (for future enhancements)
def optimize_strategy(data):
    # Example implementation for strategy optimization
    pass

# Check if it's trading hours
def is_trading_hours():
    now = datetime.now().time()
    market_open = time(9, 15)
    market_close = time(15, 30)
    return market_open <= now <= market_close

# Live trading (dummy for simulation)
def live_trading(data):
    try:
        if is_trading_hours():
            return backtest_strategy(data)
        else:
            st.warning("Market is closed. Live trading only works during trading hours (9:15 AM to 3:30 PM).")
            return None, None, None
    except Exception as e:
        st.error(f"Error during live trading: {e}")
        return None, None, None

# Streamlit UI
st.title("Manual Scalping Strategy for Nifty50 and Other Indices")

# Dropdown for index selection
indices = {
    'All': 'All',
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
        if trade_mode == "Backtesting":
            # Manual scalping strategy
            strategy_data = scalping_strategy(data, ma_short_len=9, ma_long_len=21, rsi_len=14, rsi_lower=30, rsi_upper=70, bb_len=20, stop_loss=0.005, take_profit=0.01)
            if strategy_data is not None:
                net_profit, accuracy, total_trades = backtest_strategy(strategy_data)
                st.write(f"**Net Profit:** Rs {net_profit:.2f}")
                st.write(f"**Accuracy:** {accuracy:.2f}%")
                st.write(f"**Total Trades:** {total_trades}")
        else:
            # Run live trading simulation
            net_profit, accuracy, total_trades = live_trading(data)
            if net_profit is not None:
                st.write(f"**Net Profit:** Rs {net_profit:.2f}")
                st.write(f"**Accuracy:** {accuracy:.2f}%")
                st.write(f"**Total Trades:** {total_trades}")
else:
    st.write("Click on 'Run Strategy' to backtest or trade live.")
