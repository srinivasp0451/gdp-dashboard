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

# Function to filter the data based on selected index and expiry
def filter_data(df, selected_index, expiry_date, strike_price, option_type):
    # Filter out rows with NaN values in relevant columns
    df = df.dropna(subset=['SEM_CUSTOM_SYMBOL', 'SEM_STRIKE_PRICE', 'SEM_EXPIRY_DATE', 'SEM_OPTION_TYPE'])
    
    
    # Filter by exchange (BSE) for consistency with the sample code
    #df = df[df['SEM_EXM_EXCH_ID'] == 'BSE']
    
    
    # Filter by custom symbol (e.g., 'SENSEX 18 FEB' or 'NIFTY')
    if selected_index == 'Nifty':
        df = df[df['SEM_CUSTOM_SYMBOL'].str.startswith('NIFTY', case=False)]
    elif selected_index == 'Sensex':
        df = df[df['SEM_CUSTOM_SYMBOL'].str.startswith('SENSEX', case=False)]
    elif selected_index == 'Bank Nifty':
        df = df[df['SEM_CUSTOM_SYMBOL'].str.startswith('BANKNIFTY', case=False)]
    elif selected_index == 'Fin Nifty':
        df = df[df['SEM_CUSTOM_SYMBOL'].str.startswith('FINNIFTY', case=False)]
    elif selected_index == 'Midcap Nifty':
        df = df[df['SEM_CUSTOM_SYMBOL'].str.startswith('MIDCPNIFTY', case=False)]
    elif selected_index == 'Bankex':
        df = df[df['SEM_CUSTOM_SYMBOL'].str.startswith('BANKEX', case=False)]
    
    # Convert the SEM_EXPIRY_DATE to datetime format
    df['SEM_EXPIRY_DATE'] = pd.to_datetime(df['SEM_EXPIRY_DATE'], errors='coerce')

    
    # Filter by expiry date
    if expiry_date:
        df = df[df['SEM_EXPIRY_DATE'].dt.date == expiry_date]
    
    # Filter by strike price and option type (call/put)
    if strike_price and option_type:
        df = df[((df['SEM_STRIKE_PRICE'] == strike_price) & (df['SEM_OPTION_TYPE'] == option_type))]

    return df

# Streamlit UI
st.title("Security ID Fetcher for Algo Trading")

# Dropdown to select Nifty or Sensex
selected_index = st.selectbox("Select Index", ["Nifty", "Sensex","Bank Nifty","Fin Nifty","Midcap Nifty","Bankex"])

# Calendar widget to select expiry date
expiry_date = st.date_input("Select Expiry Date", min_value=datetime.date(2025, 1, 1))

# Dropdown for Option Type (CE or PE)
option_type = st.selectbox("Select Option Type", ["CE", "PE"])

# Dropdown for selecting strike price (you can manually add options or make it dynamic later)
strike_price = st.number_input("Select Strike Price", min_value=0, step=50)

# Fetch the data from the CSV URL
df = load_csv_data()

if st.button("Get ID"):

    # Filtered data based on selection
    filtered_df = filter_data(df, selected_index, expiry_date, strike_price, option_type)

    #print("filtered data will be ",filtered_df)

    # Option chain display
    if not filtered_df.empty:
        # Show the corresponding security ID
        security_id = filtered_df.iloc[0]['SEM_SMST_SECURITY_ID']
        st.write(f"Security ID for Strike Price {strike_price} and Option Type {option_type}: {security_id}")
    else:
        st.write("No data available for the selected criteria.")
