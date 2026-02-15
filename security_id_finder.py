import streamlit as st
import pandas as pd
import requests
from io import StringIO
import datetime

# Function to load the CSV data from the URL
def load_csv_data():
    url = "https://images.dhan.co/api-data/api-scrip-master.csv"
    response = requests.get(url)
    data = StringIO(response.text)
    df = pd.read_csv(data)
    return df

# Function to filter the data based on selected parameters
def filter_data(df, asset_type, selected_index=None, custom_ticker=None, expiry_date=None, strike_price=None, option_type=None):
    # Filter out rows with NaN values in relevant columns based on asset type
    if asset_type == "Index Options":
        df = df.dropna(subset=['SEM_CUSTOM_SYMBOL', 'SEM_STRIKE_PRICE', 'SEM_EXPIRY_DATE', 'SEM_OPTION_TYPE'])
        
        # Filter by custom symbol (e.g., 'SENSEX 18 FEB' or 'NIFTY')
        if selected_index == 'Nifty':
            df = df[df['SEM_CUSTOM_SYMBOL'].str.lower().str.startswith('nifty')]
        elif selected_index == 'Sensex':
            df = df[df['SEM_CUSTOM_SYMBOL'].str.lower().str.startswith('sensex')]
        elif selected_index == 'Bank Nifty':
            df = df[df['SEM_CUSTOM_SYMBOL'].str.lower().str.startswith('banknifty')]
        elif selected_index == 'Fin Nifty':
            df = df[df['SEM_CUSTOM_SYMBOL'].str.lower().str.startswith('finnifty')]
        elif selected_index == 'Midcap Nifty':
            df = df[df['SEM_CUSTOM_SYMBOL'].str.lower().str.startswith('midcpnifty')]
        elif selected_index == 'Bankex':
            df = df[df['SEM_CUSTOM_SYMBOL'].str.lower().str.startswith('bankex')]
        
        # Convert the SEM_EXPIRY_DATE to datetime format
        df['SEM_EXPIRY_DATE'] = pd.to_datetime(df['SEM_EXPIRY_DATE'], errors='coerce')
        
        # Filter by expiry date
        if expiry_date:
            df = df[df['SEM_EXPIRY_DATE'].dt.date == expiry_date]
        
        # Filter by strike price and option type (call/put)
        if strike_price and option_type:
            df = df[((df['SEM_STRIKE_PRICE'] == strike_price) & (df['SEM_OPTION_TYPE'] == option_type))]
    
    elif asset_type == "Stock Options":
        df = df.dropna(subset=['SEM_CUSTOM_SYMBOL', 'SEM_STRIKE_PRICE', 'SEM_EXPIRY_DATE', 'SEM_OPTION_TYPE'])
        
        # Filter by custom ticker (case-insensitive)
        if custom_ticker:
            # Match stocks that start with the ticker symbol
            df = df[df['SEM_CUSTOM_SYMBOL'].str.lower().str.startswith(custom_ticker.lower())]
        
        # Convert the SEM_EXPIRY_DATE to datetime format
        df['SEM_EXPIRY_DATE'] = pd.to_datetime(df['SEM_EXPIRY_DATE'], errors='coerce')
        
        # Filter by expiry date
        if expiry_date:
            df = df[df['SEM_EXPIRY_DATE'].dt.date == expiry_date]
        
        # Filter by strike price and option type (call/put)
        if strike_price and option_type:
            df = df[((df['SEM_STRIKE_PRICE'] == strike_price) & (df['SEM_OPTION_TYPE'] == option_type))]
    
    elif asset_type == "Stocks":
        # For regular stocks, we don't need strike price, expiry date, or option type
        df = df.dropna(subset=['SEM_CUSTOM_SYMBOL'])
        
        # Filter by custom ticker (case-insensitive)
        if custom_ticker:
            # For stocks, we want exact match or close match
            # Filter stocks that match the ticker symbol
            df = df[df['SEM_TRADING_SYMBOL'].str.lower() == custom_ticker.lower()]
            
            # If no exact match, try with SEM_CUSTOM_SYMBOL
            if df.empty:
                df = load_csv_data()  # Reload to try again
                df = df.dropna(subset=['SEM_CUSTOM_SYMBOL'])
                df = df[df['SEM_CUSTOM_SYMBOL'].str.lower() == custom_ticker.lower()]
        
        # Filter out options - only keep stocks (rows without strike price or option type)
        df = df[df['SEM_STRIKE_PRICE'].isna() & df['SEM_OPTION_TYPE'].isna()]

    return df

# Streamlit UI
st.title("Security ID Fetcher for Algo Trading")

# Radio button to select asset type
asset_type = st.radio("Select Asset Type", ["Index Options", "Stock Options", "Stocks"])

# Conditional UI based on asset type
if asset_type == "Index Options":
    # Dropdown to select Index
    selected_index = st.selectbox("Select Index", ["Nifty", "Sensex", "Bank Nifty", "Fin Nifty", "Midcap Nifty", "Bankex"])
    custom_ticker = None
    
    # Calendar widget to select expiry date
    expiry_date = st.date_input("Select Expiry Date", min_value=datetime.date(2025, 1, 1))
    
    # Dropdown for Option Type (CE or PE)
    option_type = st.selectbox("Select Option Type", ["CE", "PE"])
    
    # Input for strike price
    strike_price = st.number_input("Select Strike Price", min_value=0, step=50)

elif asset_type == "Stock Options":
    # Text input for custom ticker
    custom_ticker = st.text_input("Enter Stock Ticker Symbol (e.g., RELIANCE, TCS, INFY)")
    selected_index = None
    
    # Calendar widget to select expiry date
    expiry_date = st.date_input("Select Expiry Date", min_value=datetime.date(2025, 1, 1))
    
    # Dropdown for Option Type (CE or PE)
    option_type = st.selectbox("Select Option Type", ["CE", "PE"])
    
    # Input for strike price
    strike_price = st.number_input("Select Strike Price", min_value=0, step=50)

elif asset_type == "Stocks":
    # Text input for custom ticker
    custom_ticker = st.text_input("Enter Stock Ticker Symbol (e.g., RELIANCE, TCS, INFY)")
    selected_index = None
    expiry_date = None
    option_type = None
    strike_price = None

# Fetch the data from the CSV URL
df = load_csv_data()

if st.button("Get ID"):
    # Filtered data based on selection
    filtered_df = filter_data(df, asset_type, selected_index, custom_ticker, expiry_date, strike_price, option_type)

    # Display results
    if not filtered_df.empty:
        # Show the corresponding security ID
        security_id = filtered_df.iloc[0]['SEM_SMST_SECURITY_ID']
        
        if asset_type == "Index Options":
            st.success(f"✅ Security ID for {selected_index} - Strike: {strike_price}, Type: {option_type}")
        elif asset_type == "Stock Options":
            st.success(f"✅ Security ID for {custom_ticker.upper()} - Strike: {strike_price}, Type: {option_type}")
        elif asset_type == "Stocks":
            st.success(f"✅ Security ID for {custom_ticker.upper()} Stock")
        
        st.write(f"**Security ID:** {security_id}")
        
        # Show detailed data
        st.subheader("Filtered Data:")
        st.dataframe(filtered_df)
    else:
        st.error("❌ No data available for the selected criteria. Please check your inputs.")
