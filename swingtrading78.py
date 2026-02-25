import streamlit as st
import pandas as pd
import numpy as np
import requests
from nsepython import *
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime

# --- PRO GREEKS ENGINE ---
def calculate_greeks(S, K, T, r, sigma, type="call"):
    if T <= 0 or sigma <= 0 or S <= 0: return 0, 0, 0, 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if type == "call":
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100
    return round(delta, 3), round(gamma, 4), round(theta, 2), round(vega, 2)

# --- DATA FETCHING WITH SESSION BYPASS ---
@st.cache_data(ttl=60)
def get_nifty_data_fixed():
    try:
        # Step 1: Manual Session Initialization (Crucial for fixing 'None' error)
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'en-US,en;q=0.9'
        }
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        
        # Step 2: Fetch Data
        spot = nse_quote_ltp("NIFTY")
        if spot == 0 or spot is None: raise ValueError("Spot price returned None")
        
        oc_data = nse_optionchain_scrapper("NIFTY")
        if not oc_data or 'records' not in oc_data: raise ValueError("Option Chain returned None")
        
        expiry_dates = oc_data['records']['expiryDates']
        target_expiry = expiry_dates[0] # Nearest expiry
        raw_records = oc_data['records']['data']
        
        # Step 3: Processing
        processed_list = []
        exp_date = datetime.strptime(target_expiry, '%d-%b-%Y')
        T = max((exp_date - datetime.now()).days + 1, 1) / 365
        R = 0.07 # Risk-free rate (~7% repo rate)

        for item in raw_records:
            if item['expiryDate'] == target_expiry:
                strike = item['strikePrice']
                row = {'Strike': strike}
                for side in ['CE', 'PE']:
                    if side in item:
                        iv = item[side]['impliedVolatility'] / 100
                        d, g, t, v = calculate_greeks(spot, strike, T, R, iv, side.lower())
                        row.update({
                            f'{side}_LTP': item[side]['lastPrice'], f'{side}_OI': item[side]['openInterest'],
                            f'{side}_Delta': d, f'{side}_Theta': t, f'{side}_Vega': v
                        })
                processed_list.append(row)
        
        return pd.DataFrame(processed_list), spot, target_expiry
    except Exception as e:
        return str(e), None, None

# --- UI LAYER ---
df, spot_price, expiry = get_nifty_data_fixed()

if isinstance(df, str):
    st.error(f"🛑 Connection Blocked: {df}")
    st.info("The NSE server is blocking the request. Try running this on your local machine rather than a cloud IDE.")
else:
    st.title(f"📊 Nifty 50 Pro Analyzer | Spot: {spot_price}")
    
    # PCR & Recommendations
    pcr = df['PE_OI'].sum() / df['CE_OI'].sum()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Put-Call Ratio (PCR)", round(pcr, 2))
        if pcr > 1.25:
            st.warning("⚠️ Situation: OVERBOUGHT. Historically, Nifty struggles to sustain when PCR > 1.3.")
        elif pcr < 0.75:
            st.success("✅ Situation: OVERSOLD. Historically, sharp bounces occur at these levels.")
        else:
            st.info("⚖️ Situation: NEUTRAL. Market is in equilibrium.")

    # Straddle Payoff
    st.subheader("ATM Straddle Visualizer")
    atm_strike = round(spot_price / 50) * 50
    atm_row = df[df['Strike'] == atm_strike].iloc[0]
    total_prem = atm_row['CE_LTP'] + atm_row['PE_LTP']
    
    x_range = np.linspace(atm_strike - 400, atm_strike + 400, 100)
    y_payoff = [abs(s - atm_strike) - total_prem for s in x_range]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_range, y=y_payoff, fill='tozeroy', name='Short Straddle'))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    # Pro Greeks Dataframe
    st.subheader("Professional Greeks Analysis")
    st.dataframe(df[['Strike', 'CE_Delta', 'CE_Theta', 'PE_Delta', 'PE_Theta']].style.background_gradient(cmap='RdYlGn'))
