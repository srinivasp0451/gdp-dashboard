import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Professional Hybrid Options Trading System")

# =====================================================
# BLACK SCHOLES
# =====================================================

def black_scholes_greeks(S,K,T,r,sigma,option_type="call"):
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
# NSE OPTION CHAIN (INDEX ONLY)
# =====================================================

def fetch_nse_index_options(symbol="NIFTY"):

    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"

    headers = {
        "User-Agent":"Mozilla/5.0",
        "Accept-Language":"en-US,en;q=0.9",
        "Accept-Encoding":"gzip, deflate, br"
    }

    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers)
    response = session.get(url, headers=headers)

    if response.status_code != 200:
        return None,None,None

    data = response.json()

    records = data.get("records",{}).get("data",[])
    spot = data.get("records",{}).get("underlyingValue",None)

    if len(records)==0:
        return None,None,None

    df = pd.json_normalize(records)

    calls = df[["strikePrice","CE.lastPrice","CE.openInterest","CE.impliedVolatility"]].dropna()
    puts  = df[["strikePrice","PE.lastPrice","PE.openInterest","PE.impliedVolatility"]].dropna()

    calls.columns = ["strike","lastPrice_CE","openInterest_CE","iv_CE"]
    puts.columns  = ["strike","lastPrice_PE","openInterest_PE","iv_PE"]

    merged = calls.merge(puts,on="strike")

    return merged,spot,"NSE"

# =====================================================
# YFINANCE OPTION CHAIN
# =====================================================

def fetch_yf_options(ticker):

    time.sleep(1.5)
    tk = yf.Ticker(ticker)

    expiries = tk.options
    if len(expiries)==0:
        return None,None,None

    expiry = expiries[0]
    chain = tk.option_chain(expiry)

    calls = chain.calls
    puts  = chain.puts

    df = calls.merge(puts,on="strike",suffixes=("_CE","_PE"))

    spot = tk.history(period="1d")["Close"].iloc[-1]

    df = df[["strike","lastPrice_CE","openInterest_CE","impliedVolatility_CE",
             "lastPrice_PE","openInterest_PE","impliedVolatility_PE"]]

    df.columns = ["strike","lastPrice_CE","openInterest_CE","iv_CE",
                  "lastPrice_PE","openInterest_PE","iv_PE"]

    return df,spot,"YF"

# =====================================================
# UI
# =====================================================

ticker = st.text_input("Enter Ticker (NIFTY, BANKNIFTY, RELIANCE.NS, BTC-USD)",
                       "NIFTY")

# Decide data source
if ticker.upper() in ["NIFTY","BANKNIFTY","FINNIFTY"]:
    df,spot,source = fetch_nse_index_options(ticker.upper())
else:
    df,spot,source = fetch_yf_options(ticker)

if df is None:
    st.error("Options data not available for this asset.")
    st.stop()

st.success(f"Data Source: {source}")
st.metric("Spot Price",round(spot,2))

st.dataframe(df.head())

# =====================================================
# STRADDLE
# =====================================================

df["distance"] = abs(df["strike"]-spot)
atm_strike = df.sort_values("distance").iloc[0]["strike"]
atm = df[df["strike"]==atm_strike]

call_ltp = atm["lastPrice_CE"].values[0]
put_ltp  = atm["lastPrice_PE"].values[0]

straddle = call_ltp+put_ltp

st.metric("ATM Strike",atm_strike)
st.metric("ATM Straddle",round(straddle,2))

df["straddle"] = df["lastPrice_CE"]+df["lastPrice_PE"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["strike"],y=df["straddle"],mode="lines"))
fig.add_vline(x=atm_strike,line_dash="dash")
fig.update_layout(title="Straddle Premium")
st.plotly_chart(fig,use_container_width=True)

# =====================================================
# OI
# =====================================================

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=df["strike"],y=df["openInterest_CE"],name="Call OI"))
fig2.add_trace(go.Bar(x=df["strike"],y=df["openInterest_PE"],name="Put OI"))
fig2.update_layout(title="Open Interest")
st.plotly_chart(fig2,use_container_width=True)

# =====================================================
# GREEKS
# =====================================================

T = 7/365
r = 0.06

call_delta,call_gamma,call_theta,call_vega = black_scholes_greeks(
    spot,atm_strike,T,r,atm["iv_CE"].values[0],"call")

put_delta,put_gamma,put_theta,put_vega = black_scholes_greeks(
    spot,atm_strike,T,r,atm["iv_PE"].values[0],"put")

greeks = pd.DataFrame({
    "Type":["Call","Put"],
    "Delta":[call_delta,put_delta],
    "Gamma":[call_gamma,put_gamma],
    "Theta":[call_theta,put_theta],
    "Vega":[call_vega,put_vega]
})

st.write("ATM Greeks")
st.dataframe(greeks)
