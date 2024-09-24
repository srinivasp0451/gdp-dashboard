import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

# Define available indices and time frames
indices = {
    "Nifty 50": "^NSEI",
    "Bank Nifty": "^NSEBANK",
    "Fin Nifty": "NIFTY_FIN_SERVICE.NS",
    "Midcap Nifty": "NIFTY_MID_SELECT.NS",
    "Sensex": "^BSESN",
    "Bankex": "BSE-BANK.BO"
}

timeframes = {
    "1 Minute": "1m",
    "5 Minutes": "5m",
    "15 Minutes": "15m",
    "30 Minutes": "30m",
    "1 Hour": "60m",
    "1 Day": "1d"
}

# Function to calculate support and resistance
def find_support_resistance(data):
    support = data['Low'].min()
    resistance = data['High'].max()
    return support, resistance

# Streamlit UI
st.title("Candlestick Chart with Support and Resistance Analysis")
selected_index = st.selectbox("Select Index", list(indices.keys()))
selected_timeframe = st.selectbox("Select Time Frame", list(timeframes.keys()))

if st.button("Plot Candlestick"):
    ticker = indices[selected_index]
    timeframe = timeframes[selected_timeframe]

    # Fetching the last 5 days of data
    all_days_data = []
    for i in range(5):
        data = yf.download(ticker, period='1d', interval='5m', 
                           start=pd.Timestamp.now() - pd.Timedelta(days=i + 1), 
                           end=pd.Timestamp.now() - pd.Timedelta(days=i))
        if not data.empty:
            all_days_data.append(data)

    # Plotting and analysis for each day
    for idx, day_data in enumerate(all_days_data):
        day_data['Date'] = day_data.index
        support, resistance = find_support_resistance(day_data)

        # Create the candlestick chart
        fig = go.Figure(data=[go.Candlestick(x=day_data.index,
                                             open=day_data['Open'],
                                             high=day_data['High'],
                                             low=day_data['Low'],
                                             close=day_data['Close'])])

        # Add support and resistance lines
        fig.add_trace(go.Scatter(x=day_data.index, y=[support] * len(day_data),
                                 mode='lines', name='Support', line=dict(color='green', dash='dash')))
        fig.add_trace(go.Scatter(x=day_data.index, y=[resistance] * len(day_data),
                                 mode='lines', name='Resistance', line=dict(color='red', dash='dash')))

        # Customize layout
        fig.update_layout(title=f'{selected_index} Candlestick Chart - Day {idx + 1}',
                          xaxis_title='Time',
                          yaxis_title='Price',
                          xaxis_rangeslider_visible=False)

        # Show plot for each day
        st.plotly_chart(fig)

        # Print support and resistance
        st.write(f"**Day {idx + 1}:** Support = {support:.2f}, Resistance = {resistance:.2f}")

    # Final analysis
    st.write("### Trading Psychology and Recommendations")
    if resistance > support:
        st.write(f"**Recommended Strategy:**")
        st.write(f"Consider buying Call options (CE) near the support level of {support:.2f} for potential upside, and buying Put options (PE) near the resistance level of {resistance:.2f} to hedge against downward moves.")
    else:
        st.write("Market conditions may require further analysis.")
