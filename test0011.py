import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time

def backtest():
    # Parameters
    symbol = '^NSEI'  # Nifty index ticker
    stop_loss = 10
    target = 20

    # Fetch historical data
    data = yf.download(symbol, period='5d', interval='1m')
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
        close = data['Close'].iloc[i]
        ema9 = data['EMA9'].iloc[i]
        ema15 = data['EMA15'].iloc[i]
        support = data['Support'].iloc[i]
        resistance = data['Resistance'].iloc[i]

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

    data['Signal'] = signals

    # Evaluate trades
    total_profit_points = 0
    total_loss_points = 0

    for trade in trades:
        entry_price = trade["Entry"]
        entry_time = trade["Entry Time"]

        for j in range(data.index.get_loc(entry_time) + 1, len(data)):
            close_price = data['Close'].iloc[j]
            if trade["Type"] == "Buy":
                if close_price >= trade["Target"]:
                    points = target
                    total_profit_points += points
                    trade_details.append({"Entry Time": entry_time, "Exit Time": data.index[j], "Entry Price": entry_price, "Exit Price": trade["Target"], "Points": points, "Result": "Profit"})
                    break
                elif close_price <= trade["Stop Loss"]:
                    points = -stop_loss
                    total_loss_points += -points
                    trade_details.append({"Entry Time": entry_time, "Exit Time": data.index[j], "Entry Price": entry_price, "Exit Price": trade["Stop Loss"], "Points": points, "Result": "Loss"})
                    break
            elif trade["Type"] == "Sell":
                if close_price <= trade["Target"]:
                    points = target
                    total_profit_points += points
                    trade_details.append({"Entry Time": entry_time, "Exit Time": data.index[j], "Entry Price": entry_price, "Exit Price": trade["Target"], "Points": points, "Result": "Profit"})
                    break
                elif close_price >= trade["Stop Loss"]:
                    points = -stop_loss
                    total_loss_points += -points
                    trade_details.append({"Entry Time": entry_time, "Exit Time": data.index[j], "Entry Price": entry_price, "Exit Price": trade["Stop Loss"], "Points": points, "Result": "Loss"})
                    break

    return len(trade_details), total_profit_points, total_loss_points, total_profit_points - total_loss_points

def live_trade():
    symbol = '^NSEI'
    stop_loss = 10
    target = 20

    # Placeholder for live trading recommendations
    close = None
    recommendation = "Waiting for live data..."

    # Fetch live data
    data = yf.download(symbol, period='1d', interval='1m')
    data = data[['Close', 'High', 'Low']]
    
    # Calculate EMAs
    data['EMA9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA15'] = data['Close'].ewm(span=15, adjust=False).mean()

    close = data['Close'].iloc[-1]
    ema9 = data['EMA9'].iloc[-1]
    ema15 = data['EMA15'].iloc[-1]

    # Generate recommendation
    if ema9 > ema15:
        recommendation = f"Recommendation: Buy at {close:.2f}"
    elif ema9 < ema15:
        recommendation = f"Recommendation: Sell at {close:.2f}"
    else:
        recommendation = f"Recommendation: Hold at {close:.2f}"

    return recommendation

def main():
    st.title("Trading Strategy Application")

    option = st.selectbox("Choose an option:", ["Backtest", "Live Trading Recommendations"])

    if option == "Backtest":
        if st.button("Run Backtest"):
            trades, total_profit, total_loss, net_profit = backtest()
            st.write("### Backtest Results")
            st.write(f"Total Trades: {trades}")
            st.write(f"Total Profit Points: {total_profit}")
            st.write(f"Total Loss Points: {total_loss}")
            st.write(f"Net Profit/Loss Points: {net_profit}")

    elif option == "Live Trading Recommendations":
        if st.button("Get Live Recommendations"):
            recommendation = live_trade()
            st.write("### Live Trading Recommendation")
            st.write(recommendation)

if __name__ == "__main__":
    main()
