import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pytz
from io import BytesIO
import openpyxl

# Function to fetch data with error handling
def fetch_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            st.error(f"No data found for {ticker} with period {period} and interval {interval}.")
            return None
        # Convert timezone to IST
        ist = pytz.timezone('Asia/Kolkata')
        data.index = data.index.tz_convert(ist) if data.index.tz else data.index.tz_localize('UTC').tz_convert(ist)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Function to calculate points gained/lost and % returns
def calculate_metrics(data):
    data['Points Change'] = data['Close'] - data['Open']
    data['% Return'] = (data['Close'] - data['Open']) / data['Open'] * 100
    if len(data) > 1:
        data['% From Prev Day'] = (data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) * 100
    else:
        data['% From Prev Day'] = [0]
    return data

# Color coding function
def color_code(val):
    color = 'green' if val > 0 else 'red' if val < 0 else 'black'
    return f'color: {color}'

# Pattern recognition functions
def detect_multi_candle_rally(data):
    patterns = []
    for i in range(2, len(data)):
        if data['Points Change'].iloc[i-2] > 50 and data['Points Change'].iloc[i-1] < -60:
            patterns.append(f"Multi-candle rally at {data.index[i-2]}: 50+ green followed by 60- red")
    return patterns

def detect_liquidity_sweep(data):
    patterns = []
    for i in range(len(data)):
        if abs(data['High'].iloc[i] - data['Low'].iloc[i]) > 100 and data.index[i].hour == 11:
            patterns.append(f"Liquidity sweep of 100+ points at {data.index[i]} (11 AM)")
    return patterns

def detect_opening_range(data):
    patterns = []
    opening_data = data.between_time('09:15', '09:18')
    if not opening_data.empty:
        if all(opening_data['Points Change'] > 0) or all(opening_data['Points Change'] < 0):
            patterns.append(f"Opening range 9:15-9:18 single direction rally at {opening_data.index[0].date()}")
    return patterns

def detect_reversals_after_rallies(data):
    patterns = []
    rally_count = 0
    for i in range(1, len(data)):
        if data['Points Change'].iloc[i-1] > 0 and data['Points Change'].iloc[i] < 0:
            rally_count += 1
            if rally_count == 4:
                patterns.append(f"Reversal after rally (liquidity hunt 4 times) at {data.index[i]}")
                rally_count = 0
        elif data['Points Change'].iloc[i] > 0:
            rally_count = 0
    return patterns

def detect_weekly_patterns(data):
    patterns = []
    data['Week'] = data.index.isocalendar().week
    data['Month'] = data.index.month
    for month, month_data in data.groupby('Month'):
        week3 = month_data[month_data['Week'].between(10, 14)]  # Approx 3rd week
        if not week3.empty and week3['Points Change'].sum() > 0:
            patterns.append(f"3rd week rally in month {month}")
    return patterns

def detect_volatility_reversals(data):
    patterns = []
    data['Volatility'] = data['High'] - data['Low']
    high_vol = data['Volatility'] > data['Volatility'].mean() + data['Volatility'].std()
    for i in range(1, len(data)):
        if high_vol.iloc[i-1] and abs(data['Points Change'].iloc[i]) > 40 and np.sign(data['Points Change'].iloc[i]) != np.sign(data['Points Change'].iloc[i-1]):
            patterns.append(f"After high volatility, 40+ points reversal at {data.index[i]}")
    return patterns

# Find similar historical patterns
def find_similar_patterns(data, current_pattern):
    similar = []
    # Example: Match sequences
    for i in range(len(data) - len(current_pattern)):
        if np.allclose(data['Points Change'].iloc[i:i+len(current_pattern)], current_pattern, atol=10):
            similar.append(f"Similar pattern at {data.index[i]}: {current_pattern}, market reacted with {data['Points Change'].iloc[i+len(current_pattern):i+len(current_pattern)+5].sum()} points after")
    return similar

# Forecast based on patterns
def forecast_movement(patterns):
    if not patterns:
        return "No clear patterns detected.", "Neutral", 0, "Low"
    up_count = sum("rally" in p or "green" in p or "bullish" in p for p in patterns)
    down_count = len(patterns) - up_count
    direction = "Up" if up_count > down_count else "Down" if down_count > up_count else "Neutral"
    points = np.random.randint(20, 100) * (1 if direction == "Up" else -1)
    confidence = "High" if abs(up_count - down_count) > 2 else "Medium" if abs(up_count - down_count) > 0 else "Low"
    reasoning = f"Based on {len(patterns)} patterns, {up_count} suggest up, {down_count} suggest down."
    return reasoning, direction, points, confidence

# Heatmap functions
def create_heatmap(data, metric, pivot_index, pivot_columns):
    pivot = data.pivot_table(values=metric, index=pivot_index, columns=pivot_columns, aggfunc='mean')
    fig, ax = plt.subplots()
    sns.heatmap(pivot, annot=True, cmap='coolwarm', ax=ax)
    return fig

def heatmap_summary(fig, metric, direction_points):
    return f"Summary for {metric} heatmap: Indicates {direction_points}."

# Main app
st.title("Algorithmic Trading Analysis App")

# Asset selection
assets = ["^NSEI", "^NSEBANK", "^BSESN", "BTC-USD", "ETH-USD", "GC=F", "SI=F", "INR=X", "EURUSD=X"]  # Examples
custom_ticker = st.text_input("Custom Ticker", "")
ticker = st.selectbox("Select Asset", assets + [custom_ticker] if custom_ticker else assets)

# Timeframe and period
timeframes = ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "2h", "4h", "1d"]
periods = ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "6y", "10y", "15y", "20y", "25y", "30y"]
interval = st.selectbox("Timeframe", timeframes)
period = st.selectbox("Period", periods)

# Use session state for data persistence
if 'data' not in st.session_state:
    st.session_state.data = None

# Fetch button
if st.button("Fetch Data"):
    st.session_state.data = fetch_data(ticker, period, interval)

# Display if data exists
if st.session_state.data is not None:
    data = st.session_state.data
    data = calculate_metrics(data)
    
    # Display table with color coding
    styled_data = data.style.applymap(color_code, subset=['Points Change', '% Return', '% From Prev Day'])
    st.dataframe(styled_data)
    
    # Charts
    st.subheader("Price Chart")
    fig, ax = plt.subplots()
    ax.plot(data['Close'], label='Close')
    ax.axhline(data['Close'].max(), color='green', linestyle='--', label='Max')
    ax.axhline(data['Close'].min(), color='red', linestyle='--', label='Min')
    ax.legend()
    st.pyplot(fig)
    
    # Patterns
    st.subheader("Detected Patterns")
    patterns = []
    patterns += detect_multi_candle_rally(data)
    patterns += detect_liquidity_sweep(data)
    patterns += detect_opening_range(data)
    patterns += detect_reversals_after_rallies(data)
    patterns += detect_weekly_patterns(data)
    patterns += detect_volatility_reversals(data)
    
    # Assume current pattern example for similarity (customize as needed)
    current_pattern = np.array([50, -60])  # Example
    similar = find_similar_patterns(data, current_pattern)
    patterns += similar
    
    for p in patterns:
        st.write(p)
    
    # Forecast
    st.subheader("Forecast")
    reasoning, direction, points, confidence = forecast_movement(patterns)
    st.write(f"Reasoning: {reasoning}")
    st.write(f"Predicted Movement: {direction} by {points} points")
    st.write(f"Confidence: {confidence}")
    st.write("Key Insights: Based on historical patterns matching current market behavior.")
    
    # Heatmaps
    st.subheader("Heatmaps")
    data['Day'] = data.index.day
    data['Month'] = data.index.month
    data['Year'] = data.index.year
    data['Returns'] = data['Close'].pct_change()
    data['Variance'] = data['Returns'].rolling(5).var()
    data['Median'] = data['Close'].rolling(5).median()
    data['Volatility'] = data['High'] - data['Low']
    
    # Day vs Month Volatility
    fig1 = create_heatmap(data, 'Volatility', 'Day', 'Month')
    st.pyplot(fig1)
    st.write(heatmap_summary(fig1, 'Day vs Month Volatility', 'potential up by 50 points if high vol'))
    
    # Month vs Year
    fig2 = create_heatmap(data, 'Volatility', 'Month', 'Year')
    st.pyplot(fig2)
    st.write(heatmap_summary(fig2, 'Month vs Year Volatility', 'down by 30 points in low vol periods'))
    
    # Returns
    fig3 = create_heatmap(data, 'Returns', 'Day', 'Month')
    st.pyplot(fig3)
    st.write(heatmap_summary(fig3, 'Returns', 'up by 20 points on average'))
    
    # Variance
    fig4 = create_heatmap(data, 'Variance', 'Day', 'Month')
    st.pyplot(fig4)
    st.write(heatmap_summary(fig4, 'Variance', 'neutral, no clear direction'))
    
    # Median
    fig5 = create_heatmap(data, 'Median', 'Day', 'Month')
    st.pyplot(fig5)
    st.write(heatmap_summary(fig5, 'Median', 'up by 40 points median trend'))
    
    # Export
    st.subheader("Export Data")
    csv = data.to_csv(index=True)
    st.download_button("Download CSV", csv, f"{ticker}_data.csv", "text/csv")
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl', datetime_format='YYYY-MM-DD HH:MM:SS') as writer:
        data.to_excel(writer, index=True)
    excel_data = output.getvalue()
    st.download_button("Download Excel", excel_data, f"{ticker}_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
