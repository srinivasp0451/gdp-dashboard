import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import datetime

# Function to get index data dynamically
def get_index_data(ticker, interval="1m"):
    try:
        data = yf.download(ticker, period="1d", interval=interval)
        return data
    except Exception as e:
        st.error(f"Error fetching index data: {e}")
        return None

# Function to get FII/DII and VIX data
def get_fii_dii_vix_data():
    try:
        # Simulated data, replace with actual API calls
        fii = 1500
        dii = -500
        vix_data = yf.download("^INDIAVIX", period="1d", interval="1m")
        return fii, dii, vix_data
    except Exception as e:
        st.error(f"Error fetching FII/DII or VIX data: {e}")
        return None, None, None

# Function to predict market sentiment
def predict_market_movement(fii, dii, vix, index_price):
    try:
        if fii > 1000 and vix < 15:
            return "Bullish", "green"
        elif dii > 1000 or vix > 20:
            return "Bearish", "red"
        else:
            return "Neutral", "gray"
    except Exception as e:
        st.error(f"Error predicting market movement: {e}")
        return "Error", "gray"

# Function to get option chain data
def get_option_chain(index, expiry_date):
    try:
        # Simulated data, replace with actual API calls
        option_chain = pd.DataFrame({
            'Strike Price': [15000, 15100, 15200],
            'CE Premium': [100, 95, 90],
            'PE Premium': [110, 105, 100],
            'OI': [10000, 9000, 8000],
            'Change in OI': [500, 300, -200]
        })
        return option_chain
    except Exception as e:
        st.error(f"Error fetching option chain data: {e}")
        return None

# Function to calculate probability of profit (POP)
def calculate_pop(premium, strike_price, current_price):
    try:
        return max(0, 100 - abs(current_price - strike_price) / premium * 100)
    except Exception as e:
        st.error(f"Error calculating POP: {e}")
        return None

# Main App
def main():
    st.sidebar.title("Market Analysis Modules")

    # Navigation sidebar
    module = st.sidebar.radio("Go to", ["Market Sentiment", "OI Analysis", "Strike Price Plots", "Probability of Profit"])

    # Get inputs common to all modules
    index = st.sidebar.selectbox("Select Index", ["Nifty 50", "Bank Nifty", "Sensex"])
    time_interval = st.sidebar.selectbox("Select Time Interval", ["1 min", "2 min", "5 min"])
    expiry_date = st.sidebar.date_input("Select Expiry Date")

    # Mapping tickers for indices
    ticker_map = {"Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK", "Sensex": "^BSESN"}
    ticker = ticker_map.get(index, "^NSEI")

    # Fetch index data
    index_data = get_index_data(ticker, time_interval)
    if index_data is not None:
        current_price = index_data['Close'].iloc[-1]

    # Fetch FII, DII, and VIX data
    fii, dii, vix_data = get_fii_dii_vix_data()

    # Market Sentiment Module
    if module == "Market Sentiment":
        st.header("Market Sentiment")
        if fii is not None and vix_data is not None:
            vix = vix_data['Close'].iloc[-1]
            movement, color = predict_market_movement(fii, dii, vix, current_price)
            st.markdown(f"<div style='background-color:{color};padding:10px;'>Market Movement: {movement}</div>", unsafe_allow_html=True)
            st.write(f"Current {index} Price: {current_price:.2f}")
            st.line_chart(index_data['Close'])

    # OI Analysis Module
    elif module == "OI Analysis":
        st.header("OI Analysis")
        option_data = get_option_chain(index, expiry_date)
        if option_data is not None:
            fig = px.bar(option_data, x='Strike Price', y='OI', title='Open Interest')
            fig.add_scatter(x=option_data['Strike Price'], y=option_data['Change in OI'], mode='lines', name='Change in OI')
            st.plotly_chart(fig)

    # Strike Price Plots Module
    elif module == "Strike Price Plots":
        st.header("Strike Price Plots")
        option_data = get_option_chain(index, expiry_date)
        if option_data is not None:
            fig = px.line(option_data, x='Strike Price', y=['CE Premium', 'PE Premium'], title='Option Premiums')
            st.plotly_chart(fig)

    # Probability of Profit Module
    elif module == "Probability of Profit":
        st.header("Probability of Profit")
        option_data = get_option_chain(index, expiry_date)
        if option_data is not None:
            option_data['POP'] = option_data.apply(lambda row: calculate_pop(row['CE Premium'], row['Strike Price'], current_price), axis=1)
            st.table(option_data[['Strike Price', 'POP']])

if __name__ == '__main__':
    main()
