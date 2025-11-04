import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
import pytz
import time

# ------------------------------
# Streamlit Page Configuration
# ------------------------------
st.set_page_config(page_title="AI Market Analyzer", layout="wide")
st.title("📊 AI Market Analyzer (NIFTY, Stocks, Crypto, Forex, etc.)")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("📌 Input Parameters")

ticker = st.sidebar.text_input("Enter Ticker Symbol", value="^NSEI", help="Example: ^NSEI, ^NSEBANK, BTC-USD, AAPL, USDINR=X")
period = st.sidebar.selectbox("Select Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=1)
interval = st.sidebar.selectbox("Select Time Interval", ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "5d", "1wk", "1mo"], index=3)

# Groq API config
GROQ_API_KEY = "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"
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # ✅ Updated working model

if not GROQ_API_KEY:
    st.sidebar.warning("Please enter your Groq API key to continue.")

# ------------------------------
# Function to fetch data safely
# ------------------------------
def fetch_data(ticker, period, interval, retries=3, delay=2):
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
            if not df.empty:
                df.reset_index(inplace=True)
                return df
        except Exception as e:
            st.warning(f"Attempt {attempt+1} failed: {e}")
            time.sleep(delay)
    st.error("⚠️ Failed to fetch data from yfinance after several attempts.")
    return pd.DataFrame()

# ------------------------------
# AI Summary Function (Groq)
# ------------------------------
def analyze_market(df, ticker, interval, api_key):
    df = df.tail(20).copy()

    # Convert timezone to IST
    if not df.empty:
        if not df["Datetime"].dt.tz:
            df["Datetime"] = df["Datetime"].dt.tz_localize("UTC")
        df["Datetime"] = df["Datetime"].dt.tz_convert("Asia/Kolkata")

    # Flatten and summarize data for LLM
    summary_text = df.to_string(index=False)

    prompt = f"""
You are a financial analyst AI.
Analyze the following {ticker} data ({interval} interval) and summarize in simple layman terms:
1. Trend direction (bullish, bearish, or sideways)
2. Any visible reversal or breakout signals
3. Suggested next move (Buy, Sell, or Hold) with simple reason
4. Mention clear price zone if possible.

Data:
{summary_text}
"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"❌ API Error {response.status_code}: {response.text}"

# ------------------------------
# Main Execution (on Button Click)
# ------------------------------
if st.sidebar.button("🔍 Analyze Market", use_container_width=True):
    if not GROQ_API_KEY:
        st.error("Please provide Groq API key.")
    else:
        with st.spinner("Fetching data and analyzing... ⏳"):
            df = fetch_data(ticker, period, interval)
            if not df.empty:
                st.success(f"✅ Successfully fetched {ticker} data ({period}, {interval})")

                # Display recent data
                st.subheader("📈 Latest 20 Data Points (IST)")
                st.dataframe(df.tail(20), use_container_width=True)

                # Get AI summary
                ai_summary = analyze_market(df, ticker, interval, GROQ_API_KEY)
                st.subheader("🤖 AI Market Summary")
                st.write(ai_summary)
            else:
                st.error("No data fetched. Please try again with a different combination.")
else:
    st.info("👉 Enter inputs and click **Analyze Market** to start analysis.")
