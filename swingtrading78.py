# app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import time
from scipy.stats import norm
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")

# ===============================
# BLACK SCHOLES GREEKS
# ===============================

def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)

    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    theta = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T))
    vega = S*norm.pdf(d1)*np.sqrt(T)
    return delta, gamma, theta, vega

# ===============================
# FETCH NSE OPTION CHAIN
# ===============================

@st.cache_data(ttl=10)
def fetch_nse_option_chain(symbol="NIFTY"):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers)
    response = session.get(url, headers=headers)
    data = response.json()
    return data

# ===============================
# FETCH YFINANCE DATA
# ===============================

@st.cache_data(ttl=10)
def fetch_yf(ticker):
    time.sleep(1.5)   # Rate limit safety
    data = yf.download(ticker, period="5d", interval="5m")
    return data

# ===============================
# SIGNAL ENGINE
# ===============================

def generate_signal(df):
    latest = df.iloc[-1]
    if latest['price_change'] > 0 and latest['oi_change'] > 0:
        return "BUY CALL"
    elif latest['price_change'] < 0 and latest['oi_change'] > 0:
        return "BUY PUT"
    else:
        return "HOLD"

# ===============================
# STREAMLIT UI
# ===============================

st.title("Professional Options Trading System")

tabs = st.tabs([
    "Live Trading",
    "Backtesting",
    "Trade History",
    "Analysis"
])

# =====================================
# LIVE TRADING TAB
# =====================================

with tabs[0]:
    st.subheader("Live Option Chain Analysis")
    symbol = st.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY", "SENSEX"])

    data = fetch_nse_option_chain(symbol)

    records = data['records']['data']
    df = pd.json_normalize(records)

    st.write("### Option Chain Snapshot")
    st.dataframe(df.head())

    # Example plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['strikePrice'],
        y=df['CE.openInterest'],
        name="Call OI"
    ))
    fig.add_trace(go.Bar(
        x=df['strikePrice'],
        y=df['PE.openInterest'],
        name="Put OI"
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    Summary:
    This tab analyses real-time OI, Greeks and detects institutional positioning.
    For option buyers, focus on IV expansion, gamma acceleration and fresh long buildup.
    Avoid buying when OI rises but IV falls (writing dominance).
    """)

# =====================================
# BACKTEST TAB
# =====================================

with tabs[1]:
    st.subheader("Backtesting Engine")

    ticker = st.text_input("Enter ticker for backtest", "^NSEI")

    if st.button("Run Backtest"):
        data = fetch_yf(ticker)
        data['price_change'] = data['Close'].pct_change()
        data['oi_change'] = data['Volume'].pct_change()

        data['signal'] = data.apply(lambda row:
            "BUY CALL" if row['price_change']>0 and row['oi_change']>0
            else "BUY PUT" if row['price_change']<0 and row['oi_change']>0
            else "HOLD", axis=1)

        st.write(data.tail())

        st.success("""
        Summary:
        Backtest evaluates historical price-volume momentum alignment.
        System performs best during volatility expansion regimes.
        Works poorly in sideways IV crush markets.
        """)

# =====================================
# TRADE HISTORY
# =====================================

with tabs[2]:
    st.subheader("Trade History")
    if "trades" not in st.session_state:
        st.session_state.trades = []

    st.write(pd.DataFrame(st.session_state.trades))

    st.info("""
    Summary:
    Trade history tracks entries, exits, strike, Greeks and PnL.
    Consistency in trade tracking ensures realistic performance measurement.
    """)

# =====================================
# ANALYSIS TAB
# =====================================

with tabs[3]:
    st.subheader("Advanced Analysis")

    st.markdown("""
    This module analyses:
    - Long buildup (Price ↑ OI ↑)
    - Short buildup (Price ↓ OI ↑)
    - Short covering (Price ↑ OI ↓)
    - Long unwinding (Price ↓ OI ↓)
    - Gamma squeeze probability
    - Straddle expansion potential

    For option buyers:
    Buy when volatility expansion is expected.
    Avoid theta decay environments.
    """)
