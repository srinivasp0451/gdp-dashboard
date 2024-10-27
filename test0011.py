import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time

# Function to perform backtesting
def backtest(symbol, stop_loss, target):
    data = yf.download(symbol, period='5d', interval='5m')
    data = data[['Close', 'High', 'Low']]
    
    # Calculate EMAs, support, and resistance
    data['EMA9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA15'] = data['Close'].ewm(span=15, adjust=False).mean()
    data['Support'] = data['Low'].rolling(window=5).min()
    data['Resistance'] = data['High'].rolling(window=5).max()

    signals = []
    trades = []
    total_profit_points = 0
    total_loss_points = 0

    for i in range(len(data)):
        close = data['Close'].iloc[i]
        ema9 = data['EMA9'].iloc[i]
        ema15 = data['EMA15'].iloc[i]
        
        if pd.isna(close) or pd.isna(ema9) or pd.isna(ema15):
            signals.append("Hold")
            continue
        
        # Buy signal
        if ema9 > ema15 and close > data['Support'].iloc[i]:
            signals.append("Buy")
            entry_price = close
            stop_loss_price = entry_price - stop_loss
            target_price = entry_price + target
            trades.append({"Type": "Buy", "Entry": entry_price, "Stop Loss": stop_loss_price, "Target": target_price})

        # Sell signal
        elif ema9 < ema15 and close < data['Resistance'].iloc[i]:
            signals.append("Sell")
            entry_price = close
            stop_loss_price = entry_price + stop_loss
            target_price = entry_price - target
            trades.append({"Type": "Sell", "Entry": entry_price, "Stop Loss": stop_loss_price, "Target": target_price})

        else:
            signals.append("Hold")

    # Assign signals to DataFrame
    data['Signal'] = pd.Series(signals, index=data.index[:len(signals)])

    # Evaluate trades
    for trade in trades:
        entry_price = trade["Entry"]
        entry_time = data[data['Close'] == entry_price].index[0]
        for j in range(data.index.get_loc(entry_time) + 1, len(data)):
            close_price = data['Close'].iloc[j]
            if trade["Type"] == "Buy":
                if close_price >= trade["Target"]:
                    total_profit_points += target
                    break
                elif close_price <= trade["Stop Loss"]:
                    total_loss_points += stop_loss
                    break
            elif trade["Type"] == "Sell":
                if close_price <= trade["Target"]:
                    total_profit_points += target
                    break
                elif close_price >= trade["Stop Loss"]:
                    total_loss_points += stop_loss
                    break

    # Display results
    st.write("### Backtest Results:")
    st.write(f"Total Trades: {len(trades)}")
    st.write(f"Total Profit Points: {total_profit_points}")
    st.write(f"Total Loss Points: {total_loss_points}")
    st.write(f"Net Profit/Loss Points: {total_profit_points - total_loss_points}")

# Function for live trading
def live_trade(symbol, stop_loss, target):
    st.write("### Live Trading Recommendations:")
    live_status = st.empty()  # Placeholder for live trading status
    stop_live = st.button("Stop Live Trading")
    
    while True:
        data = yf.download(symbol, period='1d', interval='1m')
        data = data[['Close', 'High', 'Low']]
        
        # Calculate EMAs
        data['EMA9'] = data['Close'].ewm(span=9, adjust=False).mean()
        data['EMA15'] = data['Close'].ewm(span=15, adjust=False).mean()

        close = data['Close'].iloc[-1]
        ema9 = data['EMA9'].iloc[-1]
        ema15 = data['EMA15'].iloc[-1]

        if ema9 > ema15:
            live_status.text(f"Recommendation: Buy at {close}")
        elif ema9 < ema15:
            live_status.text(f"Recommendation: Sell at {close}")
        else:
            live_status.text(f"Recommendation: Hold at {close}")

        # Break the loop if Stop Live Trading button is clicked
        if stop_live:
            break

        time.sleep(60)  # Wait for a minute

# Streamlit UI
st.title("Trading Strategy")
symbol = st.selectbox("Select Index", ["^NSEI", "^NSEBANK", "^NSEFIN", "^NSEMDCP", "^BSESN"], index=1)  
stop_loss_points = st.selectbox("Select Stop Loss Points", [5, 10, 15, 20, 25, 30, 35, 40, 45, 50], index=3)  
target_points = 20  # Fixed target points
mode = st.selectbox("Select Mode", ["Backtest", "Live Trading"], index=0)  

if st.button("Run Strategy"):
    if mode == "Backtest":
        backtest(symbol, stop_loss_points, target_points)
    elif mode == "Live Trading":
        live_trade(symbol, stop_loss_points, target_points)
