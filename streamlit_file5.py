import streamlit as st
import pandas as pd
import requests as req
import matplotlib.pyplot as plt
import numpy as np
import time

# Function to fetch option chain data
def get_option_chain_data(index="NIFTY"):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={index}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br"
    }

    session = req.Session()
    session.get("https://www.nseindia.com", headers=headers)  # Initialize session

    try:
        response = session.get(url, headers=headers)
        data = response.json()

        spot_price = data['records']['underlyingValue']
        option_data = data['records']['data']

        # Extract strike prices
        strike_prices = sorted(set(item['strikePrice'] for item in option_data))

        return spot_price, strike_prices, option_data

    except Exception as e:
        st.error(f"Error fetching option chain data: {e}")
        return None, [], []

# Function to get strike prices above and below spot price
def get_relevant_strike_prices(spot_price, strike_prices):
    # Find the nearest strike price
    nearest_strike = min(strike_prices, key=lambda x: abs(x - spot_price))

    # Get five strike prices above and below the nearest strike price
    idx = strike_prices.index(nearest_strike)
    lower_strikes = strike_prices[max(0, idx - 5):idx]
    upper_strikes = strike_prices[idx + 1:idx + 6]

    return lower_strikes + [nearest_strike] + upper_strikes

# Function to filter CE and PE data for a selected strike price
def filter_option_data(option_data, strike_price, option_type):
    for data in option_data:
        if data['strikePrice'] == strike_price:
            if option_type == 'CE':
                return data['CE']
            elif option_type == 'PE':
                return data['PE']
    return None

# Simulating premium variation for different timeframes
def simulate_premium_variation(initial_price, num_points, variation_range=5):
    # Generate random premium variation
    return np.random.normal(initial_price, variation_range, num_points).tolist()

# Streamlit UI
st.title("NIFTY Option Chain Analysis")

# Fetch option chain data
spot_price, strike_prices, option_data = get_option_chain_data()

if spot_price:
    st.write(f"Spot Price: {spot_price}")

    # Get relevant strike prices (5 above and below spot price)
    relevant_strikes = get_relevant_strike_prices(spot_price, strike_prices)

    # Dropdowns for selecting strike prices for CE and PE
    ce_strike_price = st.selectbox("Select Call (CE) Strike Price", relevant_strikes)
    pe_strike_price = st.selectbox("Select Put (PE) Strike Price", relevant_strikes)

    # Dropdown to select time frame for premium variation
    time_frame = st.selectbox("Select Time Frame for Premium Variation (minutes)", [1, 2, 5, 10, 15, 30, 60])

    # Filter option chain data for selected strike prices
    ce_data = filter_option_data(option_data, ce_strike_price, 'CE')
    pe_data = filter_option_data(option_data, pe_strike_price, 'PE')

    # Refresh Button
    if st.button('Refresh Data'):
        st.experimental_rerun()

    if ce_data and pe_data:
        # Simulate premium variation over selected time frame
        time_series = list(range(1, 11))  # Simulating 10 time points
        ce_premiums = simulate_premium_variation(ce_data['lastPrice'], len(time_series), variation_range=5)
        pe_premiums = simulate_premium_variation(pe_data['lastPrice'], len(time_series), variation_range=5)

        st.write(f"Selected CE Strike Price: {ce_strike_price}, Current Premium: {ce_data['lastPrice']}")
        st.write(f"Selected PE Strike Price: {pe_strike_price}, Current Premium: {pe_data['lastPrice']}")

        # Plot CE and PE premiums over the selected time frame
        fig, ax = plt.subplots()
        ax.plot(time_series, ce_premiums, label=f"CE Premium ({time_frame} min)", color='green')
        ax.plot(time_series, pe_premiums, label=f"PE Premium ({time_frame} min)", color='red')
        ax.set_xlabel(f'Time ({time_frame} min intervals)')
        ax.set_ylabel('Premium')
        ax.set_title(f'CE and PE Premiums over Time ({time_frame} min)')
        ax.legend()

        st.pyplot(fig)
    else:
        st.write("Could not fetch option data for the selected strike prices.")
else:
    st.write("Failed to fetch option chain data.")
