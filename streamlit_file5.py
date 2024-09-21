import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import time

# Scraping NSE Option Chain Data (with caution)
def get_nse_option_chain(symbol='NIFTY'):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    
    try:
        url = f"https://www.nseindia.com/option-chain"
        session = requests.Session()
        
        # First, make a request to the homepage to get cookies
        session.get("https://www.nseindia.com", headers=headers)
        response = session.get(url, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            script_tag = soup.find('script', text=lambda x: x and 'optionChains' in x)
            if script_tag:
                start = script_tag.text.find('optionChains')
                end = script_tag.text.find('];', start)
                raw_data = script_tag.text[start:end].split('=')[-1].strip() + ']'
                data = eval(raw_data)
                
                return data
            else:
                st.error("Failed to extract option chain data.")
        else:
            st.error(f"Failed to fetch data: {response.status_code}")
            
    except Exception as e:
        st.error(f"Error: {e}")

# Function to get nearby 5 strike prices based on spot price
def get_relevant_strikes(spot_price, option_chain):
    strike_prices = [item['strikePrice'] for item in option_chain]
    nearest_strike = min(strike_prices, key=lambda x: abs(x - spot_price))
    
    idx = strike_prices.index(nearest_strike)
    return strike_prices[max(0, idx - 5):idx + 6]

# Plot CE and PE premium variation
def plot_premiums(option_chain, ce_strike_price, pe_strike_price):
    ce_data = [item for item in option_chain if item['strikePrice'] == ce_strike_price and item['optionType'] == 'CE']
    pe_data = [item for item in option_chain if item['strikePrice'] == pe_strike_price and item['optionType'] == 'PE']
    
    time_intervals = [1, 5, 10, 15, 30, 60]
    ce_premiums = [item['lastPrice'] for item in ce_data]
    pe_premiums = [item['lastPrice'] for item in pe_data]
    
    # Plotting the data
    fig, ax = plt.subplots()
    ax.plot(time_intervals, ce_premiums, label=f"CE {ce_strike_price}", color='green')
    ax.plot(time_intervals, pe_premiums, label=f"PE {pe_strike_price}", color='red')
    
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Premium")
    ax.set_title("Premium Variation Over Time")
    ax.legend()
    
    st.pyplot(fig)

# Streamlit UI
st.title("NSE Option Chain Scraper")

symbol = st.text_input("Enter Symbol (NIFTY, BANKNIFTY):", value="NIFTY")
data = get_nse_option_chain(symbol)

if data:
    spot_price = data['underlyingValue']
    st.write(f"Spot Price: {spot_price}")
    
    # Extract relevant strikes near spot price
    relevant_strikes = get_relevant_strikes(spot_price, data['optionChains'])
    
    ce_strike = st.selectbox("Select CE Strike Price", relevant_strikes)
    pe_strike = st.selectbox("Select PE Strike Price", relevant_strikes)
    
    # Select time frame for plotting premium variation
    time_frame = st.selectbox("Select Time Frame (min)", [1, 5, 10, 15, 30, 60])
    
    if st.button("Plot Premiums"):
        plot_premiums(data['optionChains'], ce_strike, pe_strike)

else:
    st.write("Failed to fetch option chain data.")
