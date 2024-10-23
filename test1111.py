import streamlit as st
import yfinance as yf
import pandas as pd

# Function to fetch data
def fetch_data(symbol, period, interval):
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

# Function to generate buy/sell signals
def generate_signals(data):
    signals = []
    for i in range(len(data)):
        if (data['RSI'].iloc[i] < 40 and
            data['MACD'].iloc[i] > data['Signal_Line'].iloc[i] and
            data['Close'].iloc[i] > data['SMA_50'].iloc[i]):
            signals.append('Buy')
        elif (data['RSI'].iloc[i] > 60 and
              data['MACD'].iloc[i] < data['Signal_Line'].iloc[i] and
              data['Close'].iloc[i] < data['SMA_50'].iloc[i]):
            signals.append('Sell')
        else:
            signals.append('Hold')
    data['Signal'] = signals
    return data

# Backtest strategy
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

            # Adjust trailing take-profit
            if current_price > trailing_take_profit:
                trailing_take_profit = current_price

            # Check stop-loss
            if current_price <= stop_loss:
                balance += current_price - entry_price
                trade_log.append(current_price - entry_price)
                position = None

            # Check trailing take-profit
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
    num_loss_trades = num_trades - num_profit_trades
    accuracy = (num_profit_trades / num_trades) * 100 if num_trades > 0 else 0

    return accuracy, num_trades, num_profit_trades, num_loss_trades, balance

# Function for live trading recommendations
def live_trading_recommendations(data):
    latest_signal = data['Signal'].iloc[-1]
    latest_price = data['Close'].iloc[-1]
    return latest_price, latest_signal

# Main Streamlit app
def main():
    st.title("Trading Strategy with Streamlit")

    index_options = {
        "Bank Nifty": "^NSEBANK",
        "Nifty 50": "^NSEI",
        "FinNifty": "NIFTY_FIN_SERVICE.NS",
        "Sensex": "^BSESN",
        "Midcap Nifty": "NIFTY_MID_SELECT.NS",
        "BANKEX": "BSE-BANK.BO"
    }

    # Dropdown for selecting the index
    selected_index = st.selectbox("Select an Index", list(index_options.keys()))

    # Dropdown for selecting period
    period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y"], index=0)

    # Dropdown for selecting interval
    interval = st.selectbox("Select Interval", ["1m", "5m", "15m", "30m", "1h", "1d"], index=1)

    # Dropdown for backtesting or live trading
    trading_choice = st.selectbox("Choose an Option", ["Backtest", "Live Trading"])

    if st.button("Run Strategy"):
        with st.spinner("Fetching data..."):
            symbol = index_options[selected_index]
            data = fetch_data(symbol, period, interval)

            # Calculate indicators
            data = calculate_rsi(data)
            data = calculate_macd(data)
            data = calculate_atr(data)
            data = calculate_sma(data)

            # Generate signals
            data = generate_signals(data)

            if trading_choice == "Backtest":
                accuracy, total_trades, profit_trades, loss_trades, final_balance = backtest_strategy(data)
                st.success(f"Backtesting Results:\nAccuracy: {accuracy:.2f}%\nTotal Trades: {total_trades}\nProfitable Trades: {profit_trades}\nLoss Trades: {loss_trades}\nFinal Balance: {final_balance:.2f}")

            else:  # Live Trading
                latest_price, latest_signal = live_trading_recommendations(data)
                st.success(f"Latest Price: {latest_price:.2f}, Signal: {latest_signal}")

if __name__ == "__main__":
    main()
