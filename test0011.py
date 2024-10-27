import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

def backtest(symbol, stop_loss, target):
    # Fetch historical data
    data = yf.download(symbol, period='5d', interval='1m')
    data = data[['Close', 'High', 'Low']]

    # Calculate EMAs, support, and resistance
    data['EMA9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA15'] = data['Close'].ewm(span=15, adjust=False).mean()
    data['Support'] = data['Low'].rolling(window=5).min()
    data['Resistance'] = data['High'].rolling(window=5).max()

    # Trading signals
    trades = []
    trade_details = []

    for i in range(1, len(data)):
        close = data['Close'].iloc[i]
        ema9 = data['EMA9'].iloc[i]
        ema15 = data['EMA15'].iloc[i]
        support = data['Support'].iloc[i]
        resistance = data['Resistance'].iloc[i]

        # Buy signal
        if (ema9 > ema15) and (close > support):
            entry_price = close
            stop_loss_price = entry_price - stop_loss
            target_price = entry_price + target
            trades.append({"Type": "Buy", "Entry": entry_price, "Stop Loss": stop_loss_price, "Target": target_price, "Entry Time": data.index[i]})

        # Sell signal
        elif (ema9 < ema15) and (close < resistance):
            entry_price = close
            stop_loss_price = entry_price + stop_loss
            target_price = entry_price - target
            trades.append({"Type": "Sell", "Entry": entry_price, "Stop Loss": stop_loss_price, "Target": target_price, "Entry Time": data.index[i]})

    # Evaluate trades
    total_profit_points = 0
    total_loss_points = 0
    profitable_trades = 0
    loss_trades = 0

    for trade in trades:
        entry_price = trade["Entry"]
        entry_time = trade["Entry Time"]

        for j in range(data.index.get_loc(entry_time) + 1, len(data)):
            close_price = data['Close'].iloc[j]
            if trade["Type"] == "Buy":
                if close_price >= trade["Target"]:
                    points = target
                    total_profit_points += points
                    profitable_trades += 1
                    trade_details.append({"Entry Time": entry_time, "Exit Time": data.index[j], "Entry Price": entry_price, "Exit Price": trade["Target"], "Points": points, "Result": "Profit"})
                    break
                elif close_price <= trade["Stop Loss"]:
                    points = -stop_loss
                    total_loss_points += -points
                    loss_trades += 1
                    trade_details.append({"Entry Time": entry_time, "Exit Time": data.index[j], "Entry Price": entry_price, "Exit Price": trade["Stop Loss"], "Points": points, "Result": "Loss"})
                    break
            elif trade["Type"] == "Sell":
                if close_price <= trade["Target"]:
                    points = target
                    total_profit_points += points
                    profitable_trades += 1
                    trade_details.append({"Entry Time": entry_time, "Exit Time": data.index[j], "Entry Price": entry_price, "Exit Price": trade["Target"], "Points": points, "Result": "Profit"})
                    break
                elif close_price >= trade["Stop Loss"]:
                    points = -stop_loss
                    total_loss_points += -points
                    loss_trades += 1
                    trade_details.append({"Entry Time": entry_time, "Exit Time": data.index[j], "Entry Price": entry_price, "Exit Price": trade["Stop Loss"], "Points": points, "Result": "Loss"})
                    break

    # Calculate performance metrics
    total_trades = profitable_trades + loss_trades
    accuracy = (profitable_trades / total_trades * 100) if total_trades > 0 else 0

    return {
        "Total Trades": total_trades,
        "Profitable Trades": profitable_trades,
        "Loss Trades": loss_trades,
        "Accuracy": accuracy,
        "Total Profit Points": total_profit_points,
        "Total Loss Points": total_loss_points,
        "Net Profit/Loss Points": total_profit_points - total_loss_points,
        "Trade Details": trade_details
    }

def main():
    st.title("Trading Backtest App")

    # User inputs
    symbol = st.selectbox("Select Index", ["^NSEI", "^NSEIBANK"])
    stop_loss = st.number_input("Stop Loss (Points)", value=10)
    target = st.number_input("Target (Points)", value=20)

    if st.button("Run Backtest"):
        results = backtest(symbol, stop_loss, target)

        # Display results
        st.subheader("Backtest Results")
        st.write(f"Total Trades: {results['Total Trades']}")
        st.write(f"Profitable Trades: {results['Profitable Trades']}")
        st.write(f"Loss Trades: {results['Loss Trades']}")
        st.write(f"Accuracy: {results['Accuracy']:.2f}%")
        st.write(f"Total Profit Points: {results['Total Profit Points']}")
        st.write(f"Total Loss Points: {results['Total Loss Points']}")
        st.write(f"Net Profit/Loss Points: {results['Net Profit/Loss Points']}")

        # Display trade details
        st.subheader("Trade Details")
        for trade in results["Trade Details"]:
            st.write(f"Entry Time: {trade['Entry Time']}, Exit Time: {trade['Exit Time']}, Entry Price: {trade['Entry Price']}, Exit Price: {trade['Exit Price']}, Points: {trade['Points']}, Result: {trade['Result']}")

if __name__ == "__main__":
    main()
