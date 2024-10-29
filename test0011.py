import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time

def fetch_data(symbol, start_date, end_date, interval):
    """Fetch historical data from Yahoo Finance."""
    return yf.download(symbol, start=start_date, end=end_date, interval=interval)

def backtest(symbol, stop_loss, target, backtest_days, interval):
    # Calculate start and end dates for backtest
    end_date = datetime.now()
    start_date = end_date - timedelta(days=backtest_days)

    # Fetch historical data
    data = fetch_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), interval)

    # Ensure the data is properly formatted
    data = data[['Close', 'High', 'Low']]

    # Calculate EMAs, support, and resistance
    data['EMA9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA15'] = data['Close'].ewm(span=15, adjust=False).mean()
    data['Support'] = data['Low'].rolling(window=5).min()
    data['Resistance'] = data['High'].rolling(window=5).max()

    # Trading signals
    signals = []
    trades = []
    trade_details = []

    for i in range(1, len(data)):
        try:
            close = data['Close'].iloc[i].item()
            ema9 = data['EMA9'].iloc[i].item()
            ema15 = data['EMA15'].iloc[i].item()
            support = data['Support'].iloc[i].item()
            resistance = data['Resistance'].iloc[i].item()

            # Buy signal
            if (ema9 > ema15) and (close > support):
                signals.append("Buy")
                entry_price = close
                stop_loss_price = entry_price - stop_loss
                target_price = entry_price + target
                trades.append({"Type": "Buy", "Entry": entry_price, "Stop Loss": stop_loss_price, "Target": target_price, "Entry Time": data.index[i]})

            # Sell signal
            elif (ema9 < ema15) and (close < resistance):
                signals.append("Sell")
                entry_price = close
                stop_loss_price = entry_price + stop_loss
                target_price = entry_price - target
                trades.append({"Type": "Sell", "Entry": entry_price, "Stop Loss": stop_loss_price, "Target": target_price, "Entry Time": data.index[i]})

            else:
                signals.append("Hold")

        except Exception as e:
            print(f"Error at index {i}: {e}")

    while len(signals) < len(data):
        signals.append("Hold")

    data['Signal'] = signals

    # Evaluate trades
    total_profit_points = 0
    total_loss_points = 0
    total_trades = 0

    for trade in trades:
        entry_price = trade["Entry"]
        entry_time = trade["Entry Time"]

        for j in range(data.index.get_loc(entry_time) + 1, len(data)):
            close_price = data['Close'].iloc[j].item()
            if trade["Type"] == "Buy":
                if close_price >= trade["Target"]:
                    points = target
                    total_profit_points += points
                    total_trades += 1
                    trade_details.append({"Entry Time": entry_time, "Exit Time": data.index[j], "Entry Price": entry_price, "Exit Price": trade["Target"], "Points": points, "Result": "Profit"})
                    break
                elif close_price <= trade["Stop Loss"]:
                    points = -stop_loss
                    total_loss_points += -points
                    total_trades += 1
                    trade_details.append({"Entry Time": entry_time, "Exit Time": data.index[j], "Entry Price": entry_price, "Exit Price": trade["Stop Loss"], "Points": points, "Result": "Loss"})
                    break
            elif trade["Type"] == "Sell":
                if close_price <= trade["Target"]:
                    points = target
                    total_profit_points += points
                    total_trades += 1
                    trade_details.append({"Entry Time": entry_time, "Exit Time": data.index[j], "Entry Price": entry_price, "Exit Price": trade["Target"], "Points": points, "Result": "Profit"})
                    break
                elif close_price >= trade["Stop Loss"]:
                    points = -stop_loss
                    total_loss_points += -points
                    total_trades += 1
                    trade_details.append({"Entry Time": entry_time, "Exit Time": data.index[j], "Entry Price": entry_price, "Exit Price": trade["Stop Loss"], "Points": points, "Result": "Loss"})
                    break

    # Calculate accuracy
    accuracy = (total_trades - total_loss_points / stop_loss) / total_trades * 100 if total_trades > 0 else 0

    # Display results
    st.write("\nBacktest Results:")
    st.write(f"Total Trades: {total_trades}")
    st.write(f"Total Profit Points: {total_profit_points}")
    st.write(f"Total Loss Points: {total_loss_points}")
    st.write(f"Net Profit/Loss Points: {total_profit_points - total_loss_points}")
    st.write(f"Accuracy: {accuracy:.2f}%")
    st.write("Trade Details:")
    st.write(pd.DataFrame(trade_details))

def live_trade(symbol, stop_loss, target):
    st.write("\nLive Trading Recommendations (Simulated):")
    
    # Simulated recommendation loop (you would replace this with real-time data)
    while True:
        # Fetch live data
        data = yf.download(symbol, period='1d', interval='1m')
        data = data[['Close', 'High', 'Low']]

        # Calculate EMAs
        data['EMA9'] = data['Close'].ewm(span=9, adjust=False).mean()
        data['EMA15'] = data['Close'].ewm(span=15, adjust=False).mean()

        # Latest values
        close = data['Close'].iloc[-1].item()
        ema9 = data['EMA9'].iloc[-1].item()
        ema15 = data['EMA15'].iloc[-1].item()

        # Generate recommendation
        if ema9 > ema15:
            st.write(f"Recommendation: Buy at {close}")
        elif ema9 < ema15:
            st.write(f"Recommendation: Sell at {close}")
        else:
            st.write(f"Recommendation: Hold at {close}")

        # Wait for a minute before fetching new data
        time.sleep(60)  # Adjust based on your needs

def main():
    st.title("Trading Strategy Backtest and Live Recommendations")
    
    # Dropdown for selecting index
    symbol = st.selectbox("Select Index", ["^NSEI", "^NSEBANK", "^NSEFIN", "^NSEMDCP", "^BSESN"], index=1)  # Default is Bank Nifty
    backtest_days = st.number_input("Backtest Days", min_value=1, max_value=90, value=58)  # Default 58
    #interval = st.selectbox("Select Interval", ["1m", "2m", "5m", "10m", "15m", "30m", "60m"], index=2)  # Default 5m
    interval ="5min"
    # Dropdown for target and stop loss
    target = st.selectbox("Select Target Points:", [10, 20, 30, 40, 50])
    stop_loss = st.selectbox("Select Stop Loss Points:", [5, 10, 15, 20, 25])

    choice = st.radio("Select an option:", ('Backtest', 'Live Trading Recommendations'))

    if choice == 'Backtest':
        if st.button("Run Strategy"):
            backtest(symbol, stop_loss, target, backtest_days, interval)
    elif choice == 'Live Trading Recommendations':
        live_trade(symbol, stop_loss, target)

if __name__ == "__main__":
    main()
