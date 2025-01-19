import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_option_chain(symbol, expiry_date=None):
    try:
        ticker = yf.Ticker(symbol)
        available_expiries = ticker.options
        if not available_expiries:
            st.error(f"No expiration dates available for {symbol}. Please check if the symbol is correct or try later.")
            return None
        if expiry_date and expiry_date not in available_expiries:
            st.error(f"Expiration {expiry_date} not found. Available expirations are: {available_expiries}")
            return None
        elif expiry_date is None:
            expiry_date = available_expiries[0]  # Use the first available expiry if none specified
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
    "Nifty50": "NIFTY",  # Changed from ^NSEI to NIFTY for testing
    "BankNifty": "BANKNIFTY",  # Changed from ^NSEBANK to BANKNIFTY
    "Sensex": "^BSESN",
    "FinNifty": "FINNIFTY",  # Assuming this exists, adjust if not
    "MidCapNifty": "MIDCAPNIFTY"  # Assuming this exists, adjust if not
}
symbol = index_symbols.get(index_choice, "NIFTY")

# Fetch available expiry dates for the selected symbol
option_chain_data = fetch_option_chain(symbol)
if option_chain_data is None:
    st.stop()

available_expiries = yf.Ticker(symbol).options
if available_expiries:
    expiry_date = st.selectbox("Select Expiry Date", available_expiries)
    option_chain_data = fetch_option_chain(symbol, expiry_date)
else:
    st.error("No options data available for this index. Please choose another index.")
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
    if option_chain_data:
        strikes = [option_chain_data.calls.iloc[0]['strike'], option_chain_data.calls.iloc[-1]['strike']]
        strikes = st.slider("Select strike price range for Iron Condor", min_value=strikes[0], max_value=strikes[1], value=strikes)
        max_profit, max_loss = iron_condor(strikes[0], strikes[1], premiums)
    else:
        st.error("Unable to fetch strikes for Iron Condor. Please select another strategy.")
        st.stop()

elif strategy_choice == "Iron Butterfly":
    if option_chain_data and not option_chain_data.calls.empty:
        strike = st.number_input("Enter central strike price for Iron Butterfly", min_value=0, value=option_chain_data.calls.iloc[len(option_chain_data.calls)//2]['strike'])
        max_profit, max_loss = iron_butterfly(strike, premiums)
    else:
        st.error("Unable to fetch strikes for Iron Butterfly. Please select another strategy.")
        st.stop()

elif strategy_choice == "Covered Call":
    try:
        stock_price = yf.Ticker(symbol).history(period="1d")['Close'][0]
        strike = st.number_input("Enter strike price for Covered Call", min_value=0, value=round(stock_price))
        max_profit, max_loss = covered_call(strike, stock_price, premiums['call_sell'])
    except Exception as e:
        st.error(f"Error fetching current stock price: {e}")
        st.stop()

# Display results
st.write(f"Max Profit: ₹{max_profit:.2f}")
st.write(f"Max Loss: ₹{max_loss:.2f}")
quantity = calculate_quantity(capital, max_loss)
st.write(f"Quantity of contracts you can take: {quantity}")
