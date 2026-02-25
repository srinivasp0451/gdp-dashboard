import streamlit as st
import pandas as pd
import numpy as np
from nsepython import *
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime

# --- SETTINGS & CONFIG ---
st.set_page_config(page_title="Nifty50 Pro Option Chain", layout="wide")

# Black-Scholes Formula for Greeks
def calculate_greeks(S, K, T, r, sigma, type="call"):
    if T <= 0 or sigma <= 0: return 0, 0, 0, 0
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

@st.cache_data(ttl=60)
def fetch_nifty_pro_data():
    try:
        # BUG FIX: Use nse_quote_ltp for spot price
        spot = nse_quote_ltp("NIFTY")
        
        # Fetch Option Chain
        oc_data = nse_optionchain_scrapper("NIFTY")
        expiry_dates = oc_data['records']['expiryDates']
        target_expiry = expiry_dates[0] # Nearest expiry
        
        raw_records = oc_data['records']['data']
        processed_list = []
        
        # Time to Expiry (in years)
        exp_date = datetime.strptime(target_expiry, '%d-%b-%Y')
        days_to_expiry = (exp_date - datetime.now()).days + 1
        T = max(days_to_expiry, 1) / 365
        R = 0.10 # Risk-free rate (approx 10% as per NSE standards)

        for item in raw_records:
            if item['expiryDate'] == target_expiry:
                strike = item['strikePrice']
                row = {'Strike': strike}
                
                # Pro Analysis: Greeks Calculation
                for side in ['CE', 'PE']:
                    if side in item:
                        ltp = item[side]['lastPrice']
                        iv = item[side]['impliedVolatility'] / 100
                        oi = item[side]['openInterest']
                        d, g, t, v = calculate_greeks(spot, strike, T, R, iv, side.lower())
                        
                        row.update({
                            f'{side}_LTP': ltp, f'{side}_OI': oi, f'{side}_IV': round(iv*100, 2),
                            f'{side}_Delta': d, f'{side}_Theta': t, f'{side}_Vega': v
                        })
                processed_list.append(row)
        
        return pd.DataFrame(processed_list), spot, target_expiry
    except Exception as e:
        st.error(f"Critical Data Error: {e}")
        return None, None, None

# --- UI EXECUTION ---
df, spot_price, current_expiry = fetch_nifty_pro_data()

if df is not None:
    st.title(f"🚀 Nifty 50 Pro Analyzer (Spot: {spot_price})")
    
    # 1. Market Indicators
    col1, col2, col3 = st.columns(3)
    pcr = df['PE_OI'].sum() / df['CE_OI'].sum()
    col1.metric("Put-Call Ratio (PCR)", round(pcr, 2))
    col2.metric("Nearest Expiry", current_expiry)
    col3.metric("ATM Strike", round(spot_price / 50) * 50)

    # 2. Straddle Graph
    st.subheader("ATM Straddle Payoff Analysis")
    atm_strike = round(spot_price / 50) * 50
    atm_row = df[df['Strike'] == atm_strike].iloc[0]
    total_premium = atm_row['CE_LTP'] + atm_row['PE_LTP']
    
    strikes_range = np.linspace(atm_strike - 500, atm_strike + 500, 100)
    payoff = [abs(s - atm_strike) - total_premium for s in strikes_range]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes_range, y=payoff, fill='tozeroy', name='Short Straddle'))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    # 3. Greeks Table
    st.subheader("Professional Greeks Analysis")
    display_cols = ['Strike', 'CE_OI', 'CE_Delta', 'CE_Theta', 'PE_Delta', 'PE_Theta', 'PE_OI']
    st.dataframe(df[display_cols].style.background_gradient(cmap='RdYlGn', subset=['CE_Delta', 'PE_Delta']), use_container_width=True)

    # 4. Summary & Historical Context
    st.divider()
    
    # Logic for Recommendation
    if pcr > 1.3:
        status, rec = "Overbought / Extremely Bullish", "Avoid fresh longs; Look for reversal patterns or Sell OTM Puts."
    elif pcr < 0.7:
        status, rec = "Oversold / Extremely Bearish", "Expect a bounce; Look for Call buying opportunities at support."
    else:
        status, rec = "Neutral / Rangebound", "Ideal for Non-Directional strategies (Iron Condors / Straddles)."

    st.subheader("📊 Market Summary & Pro Recommendation")
    st.write(f"**Current Situation:** {status}. The PCR of {round(pcr, 2)} suggests that market participants are {'hedging aggressively' if pcr > 1 else 'expecting resistance'}.")
    st.success(f"**Actionable Advice:** {rec}")
    
    with st.expander("📜 Historical Context of this Situation"):
        st.write("""
        - **High PCR (>1.4):** Historically, whenever Nifty PCR crosses 1.4, the market faces a 'Pullback' within 3-5 sessions as call writers are trapped.
        - **Low PCR (<0.65):** In late 2023 and early 2024, these levels marked local bottoms followed by sharp V-shaped recoveries.
        - **High Vega/IV:** When IV spikes (>20), straddle sellers often get 'Gamma Scalped.' In such cases, history suggests moving to defined-risk spreads.
        """)
