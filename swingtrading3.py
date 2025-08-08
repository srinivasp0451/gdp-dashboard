import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- NSE API URLs ---
NSE_HOME = "https://www.nseindia.com"
NSE_URL = "https://www.nseindia.com/api/oi-spurts"

# --- Function to fetch NSE data ---
def fetch_oi_spurts():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/"
    }
    session = requests.Session()
    session.get(NSE_HOME, headers=headers)  # set cookies
    response = session.get(NSE_URL, headers=headers, timeout=10)
    data = response.json()
    return pd.DataFrame(data['data'])

# --- Function to process data and calculate levels ---
def process_data(df):
    df = df[['symbol', 'priceChange', 'pchangeinOpenInterest', 'lastPrice', 'totalTradedVolume', 'industry']]
    df.rename(columns={
        'symbol': 'Stock',
        'priceChange': 'Price % Change',
        'pchangeinOpenInterest': 'OI % Change',
        'lastPrice': 'LTP',
        'totalTradedVolume': 'Volume',
        'industry': 'Sector'
    }, inplace=True)

    # Calculate entry/target/SL
    df['Entry'] = df['LTP']
    df['Target'] = df['LTP'] * (1 + 0.007)  # 0.7% profit
    df['Stop Loss'] = df['LTP'] * (1 - 0.004)  # 0.4% risk

    # Bullish
    bullish = df[(df['Price % Change'] > 0) & (df['OI % Change'] > 0)]
    bullish = bullish.sort_values(by=['OI % Change', 'Volume'], ascending=False).head(5)

    # Bearish
    bearish = df[(df['Price % Change'] < 0) & (df['OI % Change'] > 0)]
    bearish['Target'] = bearish['LTP'] * (1 - 0.007)
    bearish['Stop Loss'] = bearish['LTP'] * (1 + 0.004)
    bearish = bearish.sort_values(by=['OI % Change', 'Volume'], ascending=False).head(5)

    return bullish, bearish

# --- Streamlit UI ---
st.set_page_config(page_title="NSE OI Spurt Scanner", layout="wide")

st.title("📊 NSE OI Spurt Scanner with Entry/Target/SL")
st.markdown("Live **Top 5 Bullish** and **Top 5 Bearish** intraday stocks with suggested entry, target, and stop loss.")

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
