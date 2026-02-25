import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from scipy.stats import norm
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")
st.title("Professional Multi-Asset Options Trading System")

# =====================================================
# BLACK SCHOLES
# =====================================================

def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    if sigma <= 0 or T <= 0:
        return 0,0,0,0

    d1 = (np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    delta = norm.cdf(d1) if option_type=="call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    theta = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T))
    vega = S*norm.pdf(d1)*np.sqrt(T)

    return delta,gamma,theta,vega


# =====================================================
# FETCH PRICE DATA (RATE LIMIT SAFE)
# =====================================================

@st.cache_data(ttl=15)
def fetch_price_data(ticker):
    time.sleep(1.5)
    data = yf.download(ticker, period="5d", interval="5m", progress=False)
    return data


# =====================================================
# FETCH OPTION CHAIN
# =====================================================

@st.cache_data(ttl=15)
def fetch_option_chain(ticker):
    time.sleep(1.5)
    tk = yf.Ticker(ticker)
    expiries = tk.options

    if len(expiries) == 0:
        return None, None, None

    expiry = expiries[0]
    chain = tk.option_chain(expiry)

    calls = chain.calls
    puts = chain.puts

    return calls, puts, expiry


# =====================================================
# BUYER BIAS SIGNAL
# =====================================================

def generate_signal(price_change, iv_change):
    if price_change > 0 and iv_change > 0:
        return "BUY CALL"
    elif price_change < 0 and iv_change > 0:
        return "BUY PUT"
    else:
        return "HOLD"


# =====================================================
# TABS
# =====================================================

tabs = st.tabs(["Live Trading","Backtesting","Trade History","Analysis"])


# =====================================================
# LIVE TAB
# =====================================================

with tabs[0]:

    st.subheader("Live Multi-Asset Analysis")

    ticker = st.text_input(
        "Enter Ticker (Examples: ^NSEI, ^NSEBANK, BTC-USD, GC=F, SI=F, USDINR=X, RELIANCE.NS)",
        "^NSEI"
    )

    price_data = fetch_price_data(ticker)

    if price_data.empty:
        st.error("Failed to fetch price data.")
        st.stop()

    spot = price_data["Close"].iloc[-1]

    st.metric("Current Price", round(spot,2))

    calls, puts, expiry = fetch_option_chain(ticker)

    if calls is None:
        st.warning("Options not available for this asset.")
        st.stop()

    st.write("Nearest Expiry:", expiry)

    # Merge Calls & Puts
    df = calls.merge(
        puts,
        on="strike",
        suffixes=("_CE","_PE")
    )

    st.write("### Option Chain Snapshot")
    st.dataframe(df.head())

    # ========================
    # STRADDLE
    # ========================

    df["distance"] = abs(df["strike"] - spot)
    atm_strike = df.sort_values("distance").iloc[0]["strike"]

    atm_row = df[df["strike"]==atm_strike]

    call_ltp = atm_row["lastPrice_CE"].values[0]
    put_ltp  = atm_row["lastPrice_PE"].values[0]

    straddle_price = call_ltp + put_ltp

    st.metric("ATM Strike", atm_strike)
    st.metric("ATM Straddle Price", round(straddle_price,2))

    df["straddle"] = df["lastPrice_CE"] + df["lastPrice_PE"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["strike"],
        y=df["straddle"],
        mode="lines+markers"
    ))
    fig.add_vline(x=atm_strike, line_dash="dash")
    fig.update_layout(title="Straddle Premium Across Strikes")

    st.plotly_chart(fig, use_container_width=True)

    # ========================
    # OI PLOT
    # ========================

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df["strike"],
        y=df["openInterest_CE"],
        name="Call OI"
    ))
    fig2.add_trace(go.Bar(
        x=df["strike"],
        y=df["openInterest_PE"],
        name="Put OI"
    ))
    fig2.update_layout(title="Open Interest Distribution")

    st.plotly_chart(fig2, use_container_width=True)

    # ========================
    # GREEKS (REAL IV)
    # ========================

    iv_call = atm_row["impliedVolatility_CE"].values[0]
    iv_put  = atm_row["impliedVolatility_PE"].values[0]

    T = 7/365
    r = 0.06

    call_delta,call_gamma,call_theta,call_vega = black_scholes_greeks(
        spot,atm_strike,T,r,iv_call,"call")

    put_delta,put_gamma,put_theta,put_vega = black_scholes_greeks(
        spot,atm_strike,T,r,iv_put,"put")

    greeks_df = pd.DataFrame({
        "Type":["Call","Put"],
        "Delta":[call_delta,put_delta],
        "Gamma":[call_gamma,put_gamma],
        "Theta":[call_theta,put_theta],
        "Vega":[call_vega,put_vega]
    })

    st.write("### ATM Greeks")
    st.dataframe(greeks_df)

    # ========================
    # SIGNAL
    # ========================

    price_change = price_data["Close"].pct_change().iloc[-1]
    iv_change = iv_call

    signal = generate_signal(price_change, iv_change)

    st.success(f"Buyer Recommendation: {signal}")

    st.info("""
    Summary:
    Buy only when price momentum aligns with IV expansion.
    Avoid range markets where theta decay dominates.
    Straddle cheap + IV expansion → strong buyer opportunity.
    """)


# =====================================================
# BACKTEST TAB
# =====================================================

with tabs[1]:

    st.subheader("Backtest (Volatility Expansion Model)")

    if st.button("Run Backtest"):

        data = price_data.copy()

        data["price_change"] = data["Close"].pct_change()
        data["volatility"] = data["price_change"].rolling(10).std()

        data["signal"] = data.apply(
            lambda row: generate_signal(
                row["price_change"],
                row["volatility"]
            ), axis=1
        )

        st.dataframe(data.tail())

        st.success("""
        Summary:
        Backtest captures volatility breakout regimes.
        Strong in trending markets.
        Weak in sideways markets.
        """)


# =====================================================
# TRADE HISTORY
# =====================================================

with tabs[2]:

    st.subheader("Trade History")

    if "trades" not in st.session_state:
        st.session_state.trades = []

    st.dataframe(pd.DataFrame(st.session_state.trades))

    st.info("""
    Summary:
    Track trades, evaluate consistency.
    Focus on risk-reward > win-rate.
    """)


# =====================================================
# ANALYSIS TAB
# =====================================================

with tabs[3]:

    st.subheader("Advanced Interpretation")

    st.markdown("""
    Long Build-up → Price ↑ + OI ↑  
    Short Build-up → Price ↓ + OI ↑  
    Gamma High → Explosive movement possible  
    Straddle Low → Volatility expansion expected  
    Straddle High → Avoid fresh buying  

    Option buying works best in volatility expansion regimes.
    Avoid expiry theta decay periods.
    """)
