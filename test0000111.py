import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import time

# Function to calculate Exponential Moving Average (EMA)
def calculate_ema(data, period):
    """
    This function calculates the Exponential Moving Average (EMA) for a given period.
    The function uses pandas ewm (exponential weighted moving average) method.
    """
    ema = data.ewm(span=period, adjust=False).mean()  # Calculate EMA
    return ema

# Function to log trade details (for backtesting)
def log_trade(trade_type, entry_time, entry_price, exit_time, exit_price, trade_profit_loss):
    """
    Logs the details of each trade, such as:
    - Trade Type (Buy/Sell)
    - Entry Time and Price
    - Exit Time and Price
    - Profit/Loss points from the trade
    """
    log_entry = {
        'Trade Type': trade_type,
        'Entry Time': entry_time,
        'Entry Price': entry_price,
        'Exit Time': exit_time,
        'Exit Price': exit_price,
        'P/L Points': trade_profit_loss
    }
    return log_entry

# Backtest function to simulate the strategy on historical data
def backtest_strategy(df, ema_short_period, ema_long_period, initial_balance=100000):
    """
    Backtest the strategy on historical data by simulating Buy/Sell decisions based on EMA crossovers.
    Parameters:
    - df: DataFrame containing historical price data (Close prices).
    - ema_short_period: Period for short-term EMA (e.g., 9).
    - ema_long_period: Period for long-term EMA (e.g., 21).
    - initial_balance: Starting balance for the backtest (default is 100,000).

    Returns:
    - trades: List of trades executed during backtesting.
    - total_profit_loss: Total profit/loss in points.
    - balance: Final balance after all trades.
    - accuracy: Percentage of profitable trades.
    """
    # Calculate short and long EMAs
    df['EMA_short'] = calculate_ema(df['Close'], ema_short_period)
    df['EMA_long'] = calculate_ema(df['Close'], ema_long_period)

    # Convert the index (timestamps) to IST
    #df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert('Asia/Kolkata')
    df.index = df.index.tz_convert('Asia/Kolkata')

    # Define Buy and Sell signals based on EMA crossovers
    df['Buy_Signal'] = (df['EMA_short'] > df['EMA_long']) & (df['EMA_short'].shift(1) <= df['EMA_long'].shift(1))
    df['Sell_Signal'] = (df['EMA_short'] < df['EMA_long']) & (df['EMA_short'].shift(1) >= df['EMA_long'].shift(1))

    balance = initial_balance  # Initialize balance
    position = 0  # 1 for buy, -1 for sell, 0 for no position
    trades = []  # List to store trade logs
    total_profit_points = 0  # Track total profit points
    total_loss_points = 0  # Track total loss points
    profit_trades = 0  # Count of profitable trades
    loss_trades = 0  # Count of loss trades
    total_trades = 0  # Track total number of trades

    # Backtesting loop: Iterate over the data and simulate trades
    for i in range(1, len(df)):  # Start from index 1 to handle shift() logic
        if df['Buy_Signal'].iloc[i] and position == 0:  # If Buy signal and no open position
            position = 1  # Open buy position
            entry_price = df['Close'].iloc[i]
            entry_time = df.index[i]
            total_trades += 1  # Increment total trades

        elif df['Sell_Signal'].iloc[i] and position == 1:  # If Sell signal and open buy position
            position = 0  # Close buy position
            exit_price = df['Close'].iloc[i]
            exit_time = df.index[i]

            # Calculate the profit or loss from the trade
            trade_profit_loss = float(exit_price - entry_price)  # Ensure it's a scalar value (float)

            # Update balance (assume 1 point = 50 units of currency)
            balance += trade_profit_loss * 50

            # Update total profit/loss and count profit/loss trades
            if trade_profit_loss > 0:
                total_profit_points += trade_profit_loss
                profit_trades += 1
            else:
                total_loss_points += abs(trade_profit_loss)
                loss_trades += 1

            # Log this trade
            trades.append(log_trade("Buy", entry_time, entry_price, exit_time, exit_price, trade_profit_loss))

    # Calculate the accuracy of the strategy (percentage of profitable trades)
    if total_trades > 0:
        accuracy = (profit_trades / total_trades) * 100
    else:
        accuracy = 0.0  # If no trades were executed

    net_profit_loss_points = total_profit_points - total_loss_points  # Net profit/loss

    return trades, total_profit_points, total_loss_points, profit_trades, loss_trades, net_profit_loss_points, accuracy, total_trades

# Function to generate live trading recommendation (Buy, Sell, Hold)
def live_trading_recommendation(df, ema_short_period, ema_long_period):
    """
    This function generates a Buy/Sell/Hold recommendation based on the latest EMA crossover.
    Parameters:
    - df: DataFrame containing historical price data (Close prices).
    - ema_short_period: Period for short-term EMA (e.g., 9).
    - ema_long_period: Period for long-term EMA (e.g., 21).

    Returns:
    - recommendation: "Buy", "Sell", or "Hold"
    - price: Latest price for the recommendation
    """
    df['EMA_short'] = calculate_ema(df['Close'], ema_short_period)
    df['EMA_long'] = calculate_ema(df['Close'], ema_long_period)

    # Generate Buy and Sell signals based on the most recent crossover
    buy_signal = df['EMA_short'].iloc[-1] > df['EMA_long'].iloc[-1] and df['EMA_short'].iloc[-2] <= df['EMA_long'].iloc[-2]
    sell_signal = df['EMA_short'].iloc[-1] < df['EMA_long'].iloc[-1] and df['EMA_short'].iloc[-2] >= df['EMA_long'].iloc[-2]

    if buy_signal:
        return "Buy", df['Close'].iloc[-1]  # Return Buy recommendation and latest price
    elif sell_signal:
        return "Sell", df['Close'].iloc[-1]  # Return Sell recommendation and latest price
    else:
        return "Hold", None  # Hold recommendation if no crossover

# Streamlit App UI and Main Function
def main():
    # Streamlit UI setup for user inputs
    st.title("Nifty50 Option Scalping Strategy")

    # User input for the stock/index symbol
    index_symbol = st.text_input("Enter the Index Symbol (e.g., ^NSEBANK for Nifty50)", "^NSEBANK")

    # User input for data period (1 day, 1 week, 1 month, etc.)
    period = st.selectbox("Select Data Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])

    # User input for timeframe (1 minute, 5 minutes, 1 hour, etc.)
    timeframe = st.selectbox("Select Timeframe", ["1m", "2m", "5m", "10m", "15m", "60m"])

    # User input for EMA short and long periods
    ema_short_period = st.number_input("Short EMA Period", min_value=1, max_value=5000, value=9)
    ema_long_period = st.number_input("Long EMA Period", min_value=1, max_value=5000, value=21)

    # User input to select Backtest or Live Trading mode
    mode = st.radio("Select Mode", ("Backtest", "Live Trading"))

    # Button to run the strategy
    run_button = st.button("Run Strategy")

    if run_button:
        st.write(f"Running Strategy for {index_symbol}...")
        try:
            # Fetch data from Yahoo Finance
            df = yf.download(index_symbol, period=period, interval=timeframe)

            # Run Backtest Mode
            if mode == "Backtest":
                trades, total_profit_points, total_loss_points, profit_trades, loss_trades, net_profit_loss_points, accuracy, total_trades = backtest_strategy(df, ema_short_period, ema_long_period)

                # Display results for backtesting
                st.write(f"Total Trades: {total_trades}")
                st.write(f"Profit Trades: {profit_trades}")
                st.write(f"Loss Trades: {loss_trades}")
                st.write(f"Total Profit Points: {total_profit_points}")
                st.write(f"Total Loss Points: {total_loss_points}")
                st.write(f"Net Profit/Loss Points: {net_profit_loss_points}")
                st.write(f"Accuracy: {accuracy:.2f}%")

                st.write("--- Trade Logs ---")
                for trade in trades:
                    st.write(f"Buy at {trade['Entry Time']} Price: {trade['Entry Price']} → Sell at {trade['Exit Time']} Price: {trade['Exit Price']} P/L Points: {trade['P/L Points']}")

            # Run Live Trading Mode
            elif mode == "Live Trading":
                stop_button = st.button("Stop Live Trading", key="stop_button")

                live_status = True
                while live_status:
                    recommendation, price = live_trading_recommendation(df, ema_short_period, ema_long_period)

                    # Display live trading recommendations
                    st.write(f"Latest Recommendation: {recommendation} at {price}")

                    time.sleep(5)  # Wait for 5 seconds before updating again

                    # Stop live trading if the user clicks stop
                    if stop_button:
                        st.write("Live Trading Stopped.")
                        live_status = False

        except Exception as e:
            st.write(f"Error: {e}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
