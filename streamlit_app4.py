import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetch Historical Data for NIFTY or BANK NIFTY
def get_data(ticker, interval, period):
    data = yf.download(ticker, interval=interval, period=period)
    return data

# Identify Support and Resistance Zones using Pivot Points
def calculate_pivot_points(data):
    data['Pivot'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['Resistance1'] = (2 * data['Pivot']) - data['Low']
    data['Support1'] = (2 * data['Pivot']) - data['High']
    data['Resistance2'] = data['Pivot'] + (data['High'] - data['Low'])
    data['Support2'] = data['Pivot'] - (data['High'] - data['Low'])
    return data

# Simple Buy/Sell Signals based on Support and Resistance
def generate_signals(data):
    data['Buy_Signal'] = np.where((data['Low'] < data['Support1']), 1, 0)
    data['Sell_Signal'] = np.where((data['High'] > data['Resistance1']), -1, 0)
    return data

# Backtest the strategy
def backtest(data, initial_balance=10000, position_size=100):
    balance = initial_balance
    position = 0
    trade_log = []

    for i in range(len(data)):
        # If Buy signal is triggered
        if data['Buy_Signal'].iloc[i] == 1 and position == 0:
            position = position_size / data['Close'].iloc[i]
            balance -= position_size
            trade_log.append({'Date': data.index[i], 'Type': 'Buy', 'Price': data['Close'].iloc[i]})

        # If Sell signal is triggered and we have a position
        elif data['Sell_Signal'].iloc[i] == -1 and position > 0:
            balance += position * data['Close'].iloc[i]
            trade_log.append({'Date': data.index[i], 'Type': 'Sell', 'Price': data['Close'].iloc[i]})
            position = 0

    # Closing open position at the end of the backtest period
    if position > 0:
        balance += position * data['Close'].iloc[-1]
        trade_log.append({'Date': data.index[-1], 'Type': 'Sell', 'Price': data['Close'].iloc[-1]})
        position = 0

    return balance, trade_log

# Plot the Candlestick Chart with Buy/Sell Signals
def plot_chart(data, trade_log):
    plt.figure(figsize=(14, 8))
    plt.plot(data['Close'], label='Close Price', color='blue')

    # Plot Buy signals
    buy_signals = [log['Date'] for log in trade_log if log['Type'] == 'Buy']
    buy_prices = data.loc[buy_signals]['Close']
    plt.scatter(buy_prices.index, buy_prices, marker='^', color='green', label='Buy Signal', alpha=1)

    # Plot Sell signals
    sell_signals = [log['Date'] for log in trade_log if log['Type'] == 'Sell']
    sell_prices = data.loc[sell_signals]['Close']
    plt.scatter(sell_prices.index, sell_prices, marker='v', color='red', label='Sell Signal', alpha=1)

    plt.title('Backtest Results with Buy/Sell Signals')
    plt.legend()
    plt.show()

# Main Function to Fetch Data, Analyze, and Backtest the Strategy
def main():
    # Fetch 5-minute historical data for NIFTY
    ticker = "^NSEI"  # For Nifty, or use "^NSEBANK" for Bank Nifty
    data = get_data(ticker, interval='5m', period='1mo')

    # Calculate Support and Resistance Levels
    data = calculate_pivot_points(data)

    # Generate Buy and Sell Signals
    data = generate_signals(data)

    # Backtest the Strategy
    initial_balance = 10000  # Starting balance
    final_balance, trade_log = backtest(data, initial_balance)

    print(f"Initial Balance: {initial_balance}")
    print(f"Final Balance after Backtest: {final_balance}")
    print(f"Profit/Loss: {final_balance - initial_balance}")
    print(f"Total Trades: {len(trade_log)}")

    # Plot Candlestick Chart with Buy/Sell Signals
    plot_chart(data, trade_log)

if __name__ == "__main__":
    main()
