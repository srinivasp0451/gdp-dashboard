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
st.title("Professional Options Trading System (Buyer Bias)")

# =====================================================
# BLACK SCHOLES GREEKS
# =====================================================

def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    if sigma <= 0 or T <= 0:
        return 0,0,0,0

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

# =====================================================
# NSE OPTION CHAIN FETCH
# =====================================================

@st.cache_data(ttl=10)
def fetch_nse_option_chain(symbol="NIFTY"):

    base_url = "https://www.nseindia.com/"

    if symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"]:
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    else:
        url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br"
    }

    session = requests.Session()
    session.get(base_url, headers=headers)
    response = session.get(url, headers=headers)

    if response.status_code != 200:
        return None

    data = response.json()

    if "records" in data:
        records = data["records"]["data"]
    elif "filtered" in data:
        records = data["filtered"]["data"]
    else:
        return None

    df = pd.json_normalize(records)
    return df


# =====================================================
# YFINANCE FETCH WITH RATE LIMIT CONTROL
# =====================================================

@st.cache_data(ttl=10)
def fetch_yf_data(ticker):
    time.sleep(1.5)   # Rate limit protection
    data = yf.download(ticker, period="5d", interval="5m", progress=False)
    return data


# =====================================================
# BUYER BIAS SIGNAL ENGINE
# =====================================================

def generate_signal(price_change, oi_change, iv_change):

    if price_change > 0 and oi_change > 0 and iv_change > 0:
        return "BUY CALL"
    elif price_change < 0 and oi_change > 0 and iv_change > 0:
        return "BUY PUT"
    else:
        return "HOLD"


# =====================================================
# TABS
# =====================================================

tabs = st.tabs(["Live Trading", "Backtesting", "Trade History", "Analysis"])


# =====================================================
# LIVE TRADING TAB
# =====================================================

with tabs[0]:

    st.subheader("Live Option Chain Analysis")

    symbol = st.selectbox("Select NSE Index",
                          ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"])

    df = fetch_nse_option_chain(symbol)

    if df is None:
        st.error("Failed to fetch NSE data.")
        st.stop()

    st.write("### Raw Option Chain Snapshot")
    st.dataframe(df.head())

    # -------------------------
    # OI PLOT
    # -------------------------

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["strikePrice"],
        y=df["CE.openInterest"],
        name="Call OI"
    ))
    fig.add_trace(go.Bar(
        x=df["strikePrice"],
        y=df["PE.openInterest"],
        name="Put OI"
    ))

    fig.update_layout(title="Open Interest by Strike")
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # STRADDLE CALCULATION
    # -------------------------

    spot = df['CE.underlyingValue'].dropna().iloc[0]

    df["distance"] = abs(df["strikePrice"] - spot)
    atm_strike = df.sort_values("distance").iloc[0]["strikePrice"]

    atm_row = df[df["strikePrice"] == atm_strike]

    call_ltp = atm_row["CE.lastPrice"].values[0]
    put_ltp = atm_row["PE.lastPrice"].values[0]

    straddle_price = call_ltp + put_ltp

    st.metric("Spot Price", round(spot,2))
    st.metric("ATM Strike", atm_strike)
    st.metric("ATM Straddle Price", round(straddle_price,2))

    df["straddle"] = df["CE.lastPrice"].fillna(0) + df["PE.lastPrice"].fillna(0)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["strikePrice"],
        y=df["straddle"],
        mode="lines+markers",
        name="Straddle Premium"
    ))
    fig2.add_vline(x=atm_strike, line_dash="dash")
    fig2.update_layout(title="Straddle Premium Across Strikes")

    st.plotly_chart(fig2, use_container_width=True)

    # -------------------------
    # GREEKS CALCULATION ATM
    # -------------------------

    T = 5/365
    r = 0.06
    sigma = 0.2

    call_delta, call_gamma, call_theta, call_vega = black_scholes_greeks(
        spot, atm_strike, T, r, sigma, "call")

    put_delta, put_gamma, put_theta, put_vega = black_scholes_greeks(
        spot, atm_strike, T, r, sigma, "put")

    st.write("### ATM Greeks")

    greeks_df = pd.DataFrame({
        "Type":["Call","Put"],
        "Delta":[call_delta, put_delta],
        "Gamma":[call_gamma, put_gamma],
        "Theta":[call_theta, put_theta],
        "Vega":[call_vega, put_vega]
    })

    st.dataframe(greeks_df)

    st.info("""
    Summary:
    Buy options when price, OI and IV expand together.
    Avoid buying during IV crush or sideways theta decay markets.
    Straddle low → volatility expansion possible.
    """)


# =====================================================
# BACKTEST TAB
# =====================================================

with tabs[1]:

    st.subheader("Backtesting Engine (Price-Volume Proxy)")

    ticker = st.text_input("Enter Ticker (Example: ^NSEI, BTC-USD, RELIANCE.NS)", "^NSEI")

    if st.button("Run Backtest"):

        data = fetch_yf_data(ticker)

        if data.empty:
            st.error("No data fetched.")
            st.stop()

        data["price_change"] = data["Close"].pct_change()
        data["oi_change"] = data["Volume"].pct_change()
        data["iv_change"] = data["price_change"].rolling(5).std()

        data["signal"] = data.apply(
            lambda row: generate_signal(
                row["price_change"],
                row["oi_change"],
                row["iv_change"]
            ), axis=1
        )

        st.dataframe(data.tail())

        st.success("""
        Summary:
        Backtest identifies volatility expansion regimes.
        Works best during trending markets.
        Weak during range-bound or low IV markets.
        """)


# =====================================================
# TRADE HISTORY TAB
# =====================================================

with tabs[2]:

    st.subheader("Trade History")

    if "trades" not in st.session_state:
        st.session_state.trades = []

    st.dataframe(pd.DataFrame(st.session_state.trades))

    st.info("""
    Summary:
    Tracks simulated trades.
    Helps evaluate consistency.
    Focus on win-rate + risk-reward ratio.
    """)


# =====================================================
# ANALYSIS TAB
# =====================================================

with tabs[3]:

    st.subheader("Advanced Option Chain Interpretation")

    st.markdown("""
    Long Buildup  → Price ↑ OI ↑ → Bullish  
    Short Buildup → Price ↓ OI ↑ → Bearish  
    Short Cover   → Price ↑ OI ↓ → Short squeeze  
    Long Unwind   → Price ↓ OI ↓ → Weakness  

    Gamma High near ATM → explosive move possible  
    Straddle cheap → volatility expansion  
    Straddle expensive → avoid buying  

    Option buyers must trade only volatility expansion regimes.
    Avoid expiry theta decay periods.
    """)
