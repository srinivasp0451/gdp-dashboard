import streamlit as st
import pandas as pd
import numpy as np
from nsepython import *
import plotly.graph_objects as go
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Nifty50 Pro Option Chain", layout="wide")

st.title("📊 Nifty 50 Option Chain Analysis & AI-Grade Recommendations")
st.markdown("Real-time analysis using `nsepython`")

# 1. Fetch Option Chain Data
@st.cache_data(ttl=60) # Cache data for 60 seconds
def get_nifty_data():
    symbol = "NIFTY"
    try:
        # Get live data
        oc = option_chain(symbol)
        lTP = nse_live(symbol)
        spot_price = lTP['underlyingValue']
        
        # Get expiry dates
        expiries = get_expiry_dates(symbol)
        
        # Get option chain data for nearest expiry
        oi_data = oc['records']['data']
        df = pd.DataFrame(oi_data)
        
        # Structure data
        df['Call_OI'] = df['CE'].apply(lambda x: x['openInterest'] if isinstance(x, dict) else 0)
        df['Call_Vol'] = df['CE'].apply(lambda x: x['totalTradedVolume'] if isinstance(x, dict) else 0)
        df['Call_LTP'] = df['CE'].apply(lambda x: x['lastPrice'] if isinstance(x, dict) else 0)
        df['Call_IV'] = df['CE'].apply(lambda x: x['impliedVolatility'] if isinstance(x, dict) else 0)
        df['Put_OI'] = df['PE'].apply(lambda x: x['openInterest'] if isinstance(x, dict) else 0)
        df['Put_Vol'] = df['PE'].apply(lambda x: x['totalTradedVolume'] if isinstance(x, dict) else 0)
        df['Put_LTP'] = df['PE'].apply(lambda x: x['lastPrice'] if isinstance(x, dict) else 0)
        df['Put_IV'] = df['PE'].apply(lambda x: x['impliedVolatility'] if isinstance(x, dict) else 0)
        
        df = df[['strikePrice', 'Call_OI', 'Call_Vol', 'Call_LTP', 'Call_IV', 'Put_OI', 'Put_Vol', 'Put_LTP', 'Put_IV']]
        
        return df, spot_price, expiries[0]
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None

# Load Data
df, spot_price, expiry = get_nifty_data()

if df is not None:
    st.sidebar.header("Market Overview")
    st.sidebar.metric("Nifty Spot Price", spot_price)
    st.sidebar.write(f"Expiry: {expiry}")

    # --- 2. Option Chain Dataframe ---
    st.subheader("Option Chain Data")
    st.dataframe(df.style.highlight_between(subset=['strikePrice'], left=spot_price-100, right=spot_price+100, color='lightyellow'), use_container_width=True)

    # --- 3. Straddle Graph ---
    st.subheader("At-the-Money (ATM) Straddle Graph")
    atm_strike = round(spot_price / 50) * 50
    atm_data = df[df['strikePrice'] == atm_strike].iloc[0]
    straddle_premium = atm_data['Call_LTP'] + atm_data['Put_LTP']
    
    st.write(f"**ATM Strike:** {atm_strike} | **Straddle Premium:** {straddle_premium:.2f}")

    # Plotting Straddle Payoff
    strikes = np.linspace(atm_strike - 300, atm_strike + 300, 100)
    payoff = [abs(s - atm_strike) - straddle_premium for s in strikes]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes, y=payoff, mode='lines', name='Short Straddle Payoff'))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(title="Short Straddle Payoff Profile", xaxis_title="Nifty Strike", yaxis_title="Profit/Loss")
    st.plotly_chart(fig, use_container_width=True)

    # --- 4. Analyze Option Greeks (Pro-level) ---
    st.subheader("Option Greeks & Market Analysis")
    
    # Calculate PCR
    total_call_oi = df['Call_OI'].sum()
    total_put_oi = df['Put_OI'].sum()
    pcr = total_put_oi / total_call_oi
    
    # Simple "Pro" Metrics
    st.write(f"**Put Call Ratio (PCR):** {pcr:.2f} (PCR > 1: Bullish, < 1: Bearish)")
    
    # Find Support/Resistance
    max_call_oi = df.loc[df['Call_OI'].idxmax()]
    max_put_oi = df.loc[df['Put_OI'].idxmax()]
    
    st.write(f"**Resistance (Max Call OI):** {max_call_oi['strikePrice']}")
    st.write(f"**Support (Max Put OI):** {max_put_oi['strikePrice']}")

    # --- 5. Summary & Recommendation ---
    st.subheader("💡 Market Situation & Recommendation")
    
    analysis = ""
    recommendation = ""
    
    if pcr > 1.2:
        analysis = "The market is heavily oversold on Puts (overbought), indicating strong bullish sentiment, but potentially overextended."
        recommendation = "Hold existing longs. Caution on new aggressive buying. Consider booking partial profits if PCR stays above 1.4."
    elif pcr < 0.8:
        analysis = "The market has heavy Call writing, indicating bearish sentiment and fear of a downfall."
        recommendation = "Consider Selling Covered Calls or initiating Bear Call Spread."
    else:
        analysis = "The market is in a neutral/consolidation phase. Support and Resistance are balanced."
        recommendation = "Best for Iron Condor or Straddle Selling (Rangebound)."
        
    st.info(f"**Current Situation:** {analysis}")
    st.success(f"**Recommendation:** {recommendation}")
    
    st.markdown("""
    **Historical Context:** 
    1.  **PCR < 0.7:** Often marks a short-term bottom (2022-2023 trends).
    2.  **PCR > 1.4:** Usually leads to sharp volatility or a correction (Mid-2024 trends).
    3.  **High IV + High PCR:** Usually results in a sharp breakout in either direction.
    """)

else:
    st.warning("Unable to fetch data.")

