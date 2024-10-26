import streamlit as st
import yfinance as yf
import pandas as pd

# Function to fetch and calculate breakout signals
def fetch_data(ticker):
    data = yf.download(ticker, period='1mo', interval='5m')
    if data.empty:
        return None
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    return data

# Backtesting function
def backtest(data):
    trades = []
    entry_level = None

    for i in range(1, len(data)):
        # Entry conditions
        if (data['Close'].iloc[i] > data['SMA20'].iloc[i]) and (data['Close'].iloc[i-1] <= data['SMA20'].iloc[i-1]):
            entry_level = data['Close'].iloc[i]

        # Exit conditions
        if entry_level is not None and (
            data['Close'].iloc[i] < data['SMA20'].iloc[i] or data['Close'].iloc[i] < data['SMA50'].iloc[i]):
            exit_level = data['Close'].iloc[i]
            trades.append((entry_level, exit_level))
            entry_level = None

    # Calculate results
    total_trades = len(trades)
    profit_trades = sum(1 for entry, exit in trades if exit > entry)
    loss_trades = total_trades - profit_trades
    accuracy = profit_trades / total_trades * 100 if total_trades > 0 else 0
    total_profit_points = sum(exit - entry for entry, exit in trades)

    return {
        "total_trades": total_trades,
        "profit_trades": profit_trades,
        "loss_trades": loss_trades,
        "accuracy": accuracy,
        "total_profit_points": total_profit_points,
    }

# List of indices
indices = {
    "Nifty 50": "^NSEI",
    "Bank Nifty": "^NSEBANK",
    "Sensex": "^BSESN",
    "Midcap Nifty": "^NSEMDCP",
    "Fin Nifty": "^NSEFIN",
}

# Streamlit app
st.title("Live Indices Breakout Signals")

# Set default values for dropdowns
backtest_option = st.selectbox("Select an option", ["Backtesting", "Live Trading"], index=0)
index_name = st.selectbox("Select an index", list(indices.keys()), index=1)  # Default to Bank Nifty

# Stop button logic
if 'stop_execution' not in st.session_state:
    st.session_state.stop_execution = False

if backtest_option == "Live Trading":
    stop_button = st.button("Stop Execution")
    if stop_button:
        st.session_state.stop_execution = True

# Fetch data for the selected index
data = fetch_data(indices[index_name])

if data is not None:
    latest_close = data['Close'].iloc[-1]
    sma20_latest = data['SMA20'].iloc[-1]
    sma50_latest = data['SMA50'].iloc[-1]

    if backtest_option == "Live Trading":
        if st.session_state.stop_execution:
            st.warning("Execution stopped by user.")
        else:
            breakout_up = (latest_close > sma20_latest) and (latest_close > sma50_latest)
            breakout_down = (latest_close < sma20_latest) and (latest_close < sma50_latest)

            st.write(f"Latest Close Price: {latest_close:.2f}")
            st.write(f"SMA 20: {sma20_latest:.2f}")
            st.write(f"SMA 50: {sma50_latest:.2f}")

            if breakout_up:
                st.success("Potential breakout to the upside detected.")
            elif breakout_down:
                st.warning("Potential breakout to the downside detected.")
            else:
                st.info("No clear breakout signal.")
    elif backtest_option == "Backtesting":
        results = backtest(data)
        st.write(f"Total Trades: {results['total_trades']}")
        st.write(f"Profit Trades: {results['profit_trades']}")
        st.write(f"Loss Trades: {results['loss_trades']}")
        st.write(f"Accuracy: {results['accuracy']:.2f}%")
        st.write(f"Total Profit Points Captured: {results['total_profit_points']:.2f}")
else:
    st.error("Data not available for this index.")

# Run the Streamlit app
if __name__ == "__main__":
    st.write("Refresh the page to update data.")
