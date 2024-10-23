import yfinance as yf
import pandas as pd
import streamlit as st

# Define index options
index_options = {
    "Bank Nifty": "^NSEBANK",
    "Nifty 50": "^NSEI",
    "FinNifty": "NIFTY_FIN_SERVICE.NS",
    "Sensex": "^BSESN",
    "Midcap Nifty": "NIFTY_MID_SELECT.NS",
    "BANKEX": "BSE-BANK.BO"
}

# Function to fetch data
def fetch_5min_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval)
    return df

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['EMA_12'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

# Function to calculate ATR
def calculate_atr(data, window=14):
    data['High-Low'] = data['High'] - data['Low']
    data['High-Close'] = abs(data['High'] - data['Close'].shift(1))
    data['Low-Close'] = abs(data['Low'] - data['Close'].shift(1))
    data['True_Range'] = data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    data['ATR'] = data['True_Range'].rolling(window).mean()
    return data

# Function to calculate 50-SMA
def calculate_sma(data, window=50):
    data['SMA_50'] = data['Close'].rolling(window=window).mean()
    return data

# Function to generate signals
def generate_signals(data):
    signals = []
    for i in range(len(data)):
        rsi = data['RSI'].iloc[i]
        macd = data['MACD'].iloc[i]
        signal_line = data['Signal_Line'].iloc[i]
        close_price = data['Close'].iloc[i]
        sma_50 = data['SMA_50'].iloc[i]

        buy_signal = (rsi < 40) and (macd > signal_line) and (close_price > sma_50)
        sell_signal = (rsi > 60) and (macd < signal_line) and (close_price < sma_50)

        if buy_signal:
            signals.append('Buy')
        elif sell_signal:
            signals.append('Sell')
        else:
            signals.append('Hold')
    data['Signal'] = signals
    return data

# Function to backtest strategy
def backtest_strategy(data, initial_balance=10000, atr_multiplier=1.5, trailing_take_profit_multiplier=3.0):
    balance = initial_balance
    position = None
    entry_price = 0
    trade_log = []
    stop_loss = 0
    trailing_take_profit = 0

    for i in range(len(data)):
        if data['Signal'].iloc[i] == 'Buy' and position is None:
            position = 'Long'
            entry_price = data['Close'].iloc[i]
            atr = data['ATR'].iloc[i]
            stop_loss = entry_price - (atr_multiplier * atr)
            trailing_take_profit = entry_price + (trailing_take_profit_multiplier * atr)

        elif position == 'Long':
            current_price = data['Close'].iloc[i]

            if current_price > trailing_take_profit:
                trailing_take_profit = current_price

            if current_price <= stop_loss:
                balance += current_price - entry_price
                trade_log.append(current_price - entry_price)
                position = None

            elif current_price >= trailing_take_profit:
                balance += trailing_take_profit - entry_price
                trade_log.append(trailing_take_profit - entry_price)
                position = None

        elif data['Signal'].iloc[i] == 'Sell' and position == 'Long':
            exit_price = data['Close'].iloc[i]
            balance += exit_price - entry_price
            trade_log.append(exit_price - entry_price)
            position = None

    num_trades = len(trade_log)
    num_profit_trades = len([trade for trade in trade_log if trade > 0])
    accuracy = (num_profit_trades / num_trades) * 100 if num_trades > 0 else 0

    return accuracy, balance

# Streamlit app
def main():
    st.title("Trading Strategy App")

    mode_choice = st.selectbox("Choose Mode", ["Backtest", "Live Trading"])
    index_choice = st.selectbox("Select Index", list(index_options.keys()))
    period_choice = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y"], index=0)
    interval_choice = st.selectbox("Select Interval", ["1m", "5m", "15m", "30m", "1h"], index=1)

    if st.button("Run Strategy"):
        symbol = index_options[index_choice]
        data = fetch_5min_data(symbol, period_choice, interval_choice)

        data = calculate_rsi(data)
        data = calculate_macd(data)
        data = calculate_atr(data)
        data = calculate_sma(data)
        data = generate_signals(data)

        if mode_choice == "Backtest":
            accuracy, final_balance = backtest_strategy(data)
            st.write(f"Backtesting Results:")
            st.write(f"Accuracy: {accuracy:.2f}%")
            st.write(f"Final Balance: ${final_balance:.2f}")
        else:
            latest_signal = data['Signal'].iloc[-1]
            latest_price = data['Close'].iloc[-1]
            st.write(f"Latest Price: ${latest_price:.2f}, Signal: {latest_signal}")

    if st.button("Stop Program"):
        st.write("Program stopped.")  # Placeholder for stopping logic

if __name__ == "__main__":
    main()
