import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- NSE API URL ---
NSE_URL = "https://www.nseindia.com/api/oi-spurts"

# --- Function to fetch NSE data ---
def fetch_oi_spurts():
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers)  # Get cookies
    response = session.get(NSE_URL, headers=headers)
    data = response.json()
    return pd.DataFrame(data['data'])

# --- Function to process data ---
def process_data(df):
    # Keep important columns
    df = df[['symbol', 'priceChange', 'pchangeinOpenInterest', 'lastPrice', 'totalTradedVolume', 'industry']]
    df.rename(columns={
        'symbol': 'Stock',
        'priceChange': 'Price % Change',
        'pchangeinOpenInterest': 'OI % Change',
        'lastPrice': 'LTP',
        'totalTradedVolume': 'Volume',
        'industry': 'Sector'
    }, inplace=True)

    # Bullish: Price ↑, OI ↑
    bullish = df[(df['Price % Change'] > 0) & (df['OI % Change'] > 0)]
    bullish = bullish.sort_values(by=['OI % Change', 'Volume'], ascending=False).head(5)

    # Bearish: Price ↓, OI ↑
    bearish = df[(df['Price % Change'] < 0) & (df['OI % Change'] > 0)]
    bearish = bearish.sort_values(by=['OI % Change', 'Volume'], ascending=False).head(5)

    return bullish, bearish

# --- Streamlit UI ---
st.set_page_config(page_title="NSE OI Spurt Scanner", layout="wide")

st.title("📊 NSE OI Spurt Scanner with Volume & Sector")
st.markdown("Get **Top 5 Bullish** and **Top 5 Bearish** intraday stocks using live OI spurt data from NSE, with volume and sector info.")

if st.button("🔄 Refresh Data"):
    try:
        df = fetch_oi_spurts()
        bullish, bearish = process_data(df)

        st.subheader("✅ Top 5 Bullish Stocks (Long Build-up)")
        st.dataframe(bullish.style.background_gradient(cmap='Greens'))

        st.subheader("❌ Top 5 Bearish Stocks (Short Build-up)")
        st.dataframe(bearish.style.background_gradient(cmap='Reds'))

        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        st.error(f"Error fetching NSE data: {e}")
