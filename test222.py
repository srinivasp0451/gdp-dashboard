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

support_resistance_methods = {
    "Min/Max": "minmax",
    "Moving Average": "ma",
    "Pivot Points": "pivot",
    "Fibonacci Retracement": "fibonacci"
}

# Function to calculate support and resistance
def calculate_support_resistance(data, method):
    if method == "minmax":
        support = data['Low'].min()
        resistance = data['High'].max()
        return support, resistance
    elif method == "ma":
        support = data['Close'].rolling(window=5).mean().min()
        resistance = data['Close'].rolling(window=5).mean().max()
        return support, resistance
    elif method == "pivot":
        pivot = (data['High'].max() + data['Low'].min() + data['Close'].iloc[-1]) / 3
        support = pivot - (data['High'].max() - data['Low'].min())
        resistance = pivot + (data['High'].max() - data['Low'].min())
        return support, resistance
    elif method == "fibonacci":
        high = data['High'].max()
        low = data['Low'].min()
        diff = high - low
        levels = {
            "Level 0": high,
            "Level 1": high - 0.236 * diff,
            "Level 2": high - 0.382 * diff,
            "Level 3": high - 0.618 * diff,
            "Level 4": low
        }
        return levels

# Streamlit UI
st.title("Candlestick Chart with Support and Resistance Analysis")
selected_index = st.selectbox("Select Index", list(indices.keys()))
selected_timeframe = st.selectbox("Select Time Frame", list(timeframes.keys()))
selected_method = st.selectbox("Select Support/Resistance Method", list(support_resistance_methods.keys()))

if st.button("Plot Candlestick"):
    ticker = indices[selected_index]
    timeframe = timeframes[selected_timeframe]

    # Fetching the last 5 days of data
    all_days_data = []
    for i in range(5):
        try:
            data = yf.download(ticker, period='1d', interval='5m', 
                               start=pd.Timestamp.now() - pd.Timedelta(days=i + 1), 
                               end=pd.Timestamp.now() - pd.Timedelta(days=i))
            if not data.empty:
                all_days_data.append(data)
            else:
                st.warning(f"No data available for {ticker} on day {i + 1}.")
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            break

    if not all_days_data:
        st.error("No valid data was retrieved. Please try again.")
    else:
        # Analyze only the most recent day's data
        recent_data = all_days_data[0]  # Most recent day's data

        # Calculate support and resistance for the most recent day
        if selected_method == "fibonacci":
            fibonacci_levels = calculate_support_resistance(recent_data, support_resistance_methods[selected_method])
        else:
            recent_support, recent_resistance = calculate_support_resistance(recent_data, support_resistance_methods[selected_method])

        # Create the candlestick chart for the most recent day
        fig = go.Figure(data=[go.Candlestick(x=recent_data.index,
                                             open=recent_data['Open'],
                                             high=recent_data['High'],
                                             low=recent_data['Low'],
                                             close=recent_data['Close'])])

        # Add support and resistance lines
        if selected_method == "fibonacci":
            for level, value in fibonacci_levels.items():
                fig.add_trace(go.Scatter(x=recent_data.index, y=[value] * len(recent_data),
                                         mode='lines', name=level, line=dict(dash='dash')))
        else:
            fig.add_trace(go.Scatter(x=recent_data.index, y=[recent_support] * len(recent_data),
                                     mode='lines', name='Support', line=dict(color='green', dash='dash')))
            fig.add_trace(go.Scatter(x=recent_data.index, y=[recent_resistance] * len(recent_data),
                                     mode='lines', name='Resistance', line=dict(color='red', dash='dash')))

        # Customize layout
        fig.update_layout(title=f'{selected_index} Candlestick Chart - Most Recent Day',
                          xaxis_title='Time',
                          yaxis_title='Price',
                          xaxis_rangeslider_visible=False)

        # Show plot for the most recent day
        st.plotly_chart(fig)

        # Print support and resistance for the most recent day
        if selected_method == "fibonacci":
            st.write(f"**Most Recent Day Fibonacci Levels:**")
            for level, value in fibonacci_levels.items():
                st.write(f"{level} = {value:.2f}")
        else:
            st.write(f"**Most Recent Day Support:** {recent_support:.2f}, **Most Recent Day Resistance:** {recent_resistance:.2f}")

        # Final analysis using the most recent support and resistance
        st.write("### Trading Psychology and Recommendations")
        if selected_method != "fibonacci" and recent_resistance and recent_support:
            st.write(f"**Most Recent Support:** {recent_support:.2f}, **Most Recent Resistance:** {recent_resistance:.2f}")
            st.write(f"**Recommended Strategy:**")
            st.write(f"Consider buying Call options (CE) near the support level of {recent_support:.2f} for potential upside, and buying Put options (PE) near the resistance level of {recent_resistance:.2f} to hedge against downward moves.")
        else:
            st.write("Market conditions may require further analysis.")
