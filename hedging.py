import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

# Function to fetch live option chain data for a given index
def fetch_option_chain(symbol, expiry_date):
    try:
        option_chain = yf.Ticker(symbol).options
        if expiry_date not in option_chain:
            st.error(f"Expiry date {expiry_date} not available for {symbol}")
            return None

        options_data = yf.Ticker(symbol).option_chain(expiry_date)
        return options_data
    except Exception as e:
        st.error(f"Error fetching option chain: {e}")
        return None

# Function to calculate Iron Condor max profit and max loss
def iron_condor(strike_low, strike_high, premium_put_sell, premium_put_buy, premium_call_sell, premium_call_buy):
    max_profit = premium_put_sell + premium_call_sell - (premium_put_buy + premium_call_buy)
    max_loss = (strike_high - strike_low) - max_profit
    return max_profit, max_loss

# Function to calculate Iron Butterfly max profit and max loss
def iron_butterfly(strike, premium_put_sell, premium_call_sell, premium_put_buy, premium_call_buy):
    max_profit = premium_put_sell + premium_call_sell - (premium_put_buy + premium_call_buy)
    max_loss = (strike - strike_put_buy) - max_profit
    return max_profit, max_loss

# Function to calculate Covered Call max profit and max loss
def covered_call(strike, stock_price, premium_call_sell):
    max_profit = (strike - stock_price) + premium_call_sell
    max_loss = stock_price - premium_call_sell
    return max_profit, max_loss

# Function to calculate Quantity of contracts
def calculate_quantity(capital, max_loss_per_contract):
    quantity = capital // max_loss_per_contract
    return quantity

# Streamlit UI
st.title("Options Strategy Analysis with Real-Time Data")

# Select index
index_choice = st.selectbox("Select an Index", ["Nifty50", "BankNifty", "Sensex", "FinNifty", "MidCapNifty"])

# Set correct symbol for the selected index
if index_choice == "Nifty50":
    symbol = "^NSEI"
elif index_choice == "BankNifty":
    symbol = "^NSEBANK"
elif index_choice == "Sensex":
    symbol = "^BSESN"
elif index_choice == "FinNifty":
    symbol = "^NSEFIN"
else:
    symbol = "^NSEMDCP"

# Input for Capital
capital = st.number_input("Enter your available capital (₹)", min_value=1000, value=100000)

# Input for manual expiry date selection
expiry_date = st.date_input("Select an expiry date", min_value=datetime.today())

# Fetch the live option chain data for the selected expiry date
option_chain_data = fetch_option_chain(symbol, str(expiry_date))
if option_chain_data is None:
    st.stop()

# Display live data for calls and puts
st.subheader("Available Option Chain Data")
st.write(option_chain_data)

# User inputs for strikes and premiums
strike_low = st.number_input("Enter lower strike price for Iron Condor", min_value=0, value=0)
strike_high = st.number_input("Enter higher strike price for Iron Condor", min_value=0, value=0)
premium_put_sell = st.number_input("Enter premium received for selling put (₹)", min_value=0, value=0)
premium_put_buy = st.number_input("Enter premium paid for buying put (₹)", min_value=0, value=0)
premium_call_sell = st.number_input("Enter premium received for selling call (₹)", min_value=0, value=0)
premium_call_buy = st.number_input("Enter premium paid for buying call (₹)", min_value=0, value=0)

# Strategy selection
strategy_choice = st.selectbox("Select Option Strategy", ["Iron Condor", "Iron Butterfly", "Covered Call"])

# Calculate max profit and max loss for selected strategy
if strategy_choice == "Iron Condor":
    max_profit, max_loss = iron_condor(strike_low, strike_high, premium_put_sell, premium_put_buy, premium_call_sell, premium_call_buy)
elif strategy_choice == "Iron Butterfly":
    strike = st.number_input("Enter central strike price for Iron Butterfly", min_value=0, value=0)
    max_profit, max_loss = iron_butterfly(strike, premium_put_sell, premium_call_sell, premium_put_buy, premium_call_buy)
elif strategy_choice == "Covered Call":
    stock_price = yf.Ticker(symbol).history(period="1d")['Close'][0]  # Get the current stock price
    strike = st.number_input("Enter strike price for Covered Call", min_value=0, value=0)
    premium_call_sell = st.number_input("Enter premium received for selling call (₹)", min_value=0, value=0)
    max_profit, max_loss = covered_call(strike, stock_price, premium_call_sell)

# Show max profit, max loss, and quantity of contracts
st.write(f"Max Profit: ₹{max_profit}")
st.write(f"Max Loss: ₹{max_loss}")
quantity = calculate_quantity(capital, max_loss)
st.write(f"Quantity of contracts you can take: {quantity}")
