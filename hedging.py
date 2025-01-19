import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

# Function to fetch live option chain data for a given index
def fetch_option_chain(symbol, expiry_date):
    try:
        ticker = yf.Ticker(symbol)
        available_expiries = ticker.options
        if expiry_date not in available_expiries:
            st.error(f"Expiration {expiry_date} not found. Available expirations are: {available_expiries}")
            return None
        option_chain = ticker.option_chain(expiry_date)
        return option_chain
    except Exception as e:
        st.error(f"Error fetching option chain: {e}")
        return None

# Function to calculate Iron Condor max profit and max loss
def iron_condor(strike_low, strike_high, premiums):
    max_profit = premiums['put_sell'] + premiums['call_sell'] - (premiums['put_buy'] + premiums['call_buy'])
    max_loss = (strike_high - strike_low) - max_profit
    return max_profit, max_loss

# Function to calculate Iron Butterfly max profit and max loss
def iron_butterfly(strike, premiums):
    max_profit = premiums['put_sell'] + premiums['call_sell'] - (premiums['put_buy'] + premiums['call_buy'])
    max_loss = (strike - min(strike, premiums.get('put_buy_strike', 0))) - max_profit
    return max_profit, max_loss

# Function to calculate Covered Call max profit and max loss
def covered_call(strike, stock_price, premium):
    max_profit = (strike - stock_price) + premium
    max_loss = stock_price - premium
    return max_profit, max_loss

# Function to calculate Quantity of contracts
def calculate_quantity(capital, max_loss_per_contract):
    if max_loss_per_contract > 0:
        return capital // max_loss_per_contract
    return 0

# Streamlit UI
st.title("Options Strategy Analysis with Real-Time Data")

# Select index
index_choice = st.selectbox("Select an Index", ["Nifty50", "BankNifty", "Sensex", "FinNifty", "MidCapNifty"])
index_symbols = {
    "Nifty50": "^NSEI",
    "BankNifty": "^NSEBANK",
    "Sensex": "^BSESN",
    "FinNifty": "^NSEFIN",
    "MidCapNifty": "^NSEMDCP"
}
symbol = index_symbols.get(index_choice, "^NSEI")

# Fetch available expiry dates for the selected symbol
ticker = yf.Ticker(symbol)
available_expiries = ticker.options
expiry_date = st.selectbox("Select Expiry Date", available_expiries)

# Fetch the live option chain data for the selected expiry date
option_chain_data = fetch_option_chain(symbol, expiry_date)
if option_chain_data is None:
    st.stop()

st.subheader("Available Option Chain Data")
st.write(option_chain_data)

# Input for Capital
capital = st.number_input("Enter your available capital (₹)", min_value=1000, value=100000)

# Strategy selection
strategy_choice = st.selectbox("Select Option Strategy", ["Iron Condor", "Iron Butterfly", "Covered Call"])

# Common for all strategies
premiums = {
    'put_sell': st.number_input("Enter premium received for selling put (₹)", min_value=0, value=0),
    'put_buy': st.number_input("Enter premium paid for buying put (₹)", min_value=0, value=0),
    'call_sell': st.number_input("Enter premium received for selling call (₹)", min_value=0, value=0),
    'call_buy': st.number_input("Enter premium paid for buying call (₹)", min_value=0, value=0)
}

# Strategy-specific inputs with automatic suggestions
if strategy_choice == "Iron Condor":
    strikes = st.slider("Select strike price range for Iron Condor", min_value=0, max_value=10000, value=(1000, 2000))
    max_profit, max_loss = iron_condor(strikes[0], strikes[1], premiums)

elif strategy_choice == "Iron Butterfly":
    strike = st.number_input("Enter central strike price for Iron Butterfly", min_value=0, value=option_chain_data.calls.iloc[len(option_chain_data.calls)//2]['strike'])
    max_profit, max_loss = iron_butterfly(strike, premiums)

elif strategy_choice == "Covered Call":
    stock_price = yf.Ticker(symbol).history(period="1d")['Close'][0]
    strike = st.number_input("Enter strike price for Covered Call", min_value=0, value=stock_price)
    max_profit, max_loss = covered_call(strike, stock_price, premiums['call_sell'])

# Display results
st.write(f"Max Profit: ₹{max_profit:.2f}")
st.write(f"Max Loss: ₹{max_loss:.2f}")
quantity = calculate_quantity(capital, max_loss)
st.write(f"Quantity of contracts you can take: {quantity}")
