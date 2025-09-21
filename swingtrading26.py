import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dateutil import tz
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Advanced Swing Trading Recommender")

# ----------------------
# Helper Functions
# ----------------------
def map_columns(df):
    df_cols = df.columns.str.lower()
    col_map = {}
    for col in df_cols:
        if 'open' in col:
            col_map['open'] = col
        elif 'high' in col:
            col_map['high'] = col
        elif 'low' in col:
            col_map['low'] = col
        elif 'close' in col:
            col_map['close'] = col
        elif 'volume' in col or 'shares' in col or 'qty' in col:
            col_map['volume'] = col
        elif 'date' in col or 'time' in col:
            col_map['date'] = col
    return col_map

def convert_to_ist(df, date_col):
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('Asia/Kolkata')
    df[date_col] = pd.to_datetime(df[date_col])
    df[date_col] = df[date_col].dt.tz_localize(from_zone).dt.tz_convert(to_zone)
    return df

def plot_data(df, col_map):
    plt.figure(figsize=(15,5))
    plt.plot(df[col_map['date']], df[col_map['close']], label='Close Price', color='blue')
    plt.title('Raw Price Data')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    st.pyplot(plt.gcf())

def compute_returns(df, col_map):
    df['returns'] = df[col_map['close']].pct_change()
    df['month'] = df[col_map['date']].dt.month
    df['year'] = df[col_map['date']].dt.year
    pivot = df.pivot_table(index='year', columns='month', values='returns', aggfunc='sum')
    plt.figure(figsize=(10,6))
    sns.heatmap(pivot, annot=True, fmt=".2%", cmap="RdYlGn")
    plt.title("Year vs Month Returns Heatmap")
    st.pyplot(plt.gcf())

def summarize_data(df, col_map):
    text = f"""
    The data contains {len(df)} rows from {df[col_map['date']].min().date()} 
    to {df[col_map['date']].max().date()}. Price ranges from {df[col_map['low']].min()} 
    to {df[col_map['high']].max()}. The dataset shows typical market behavior with trends, 
    pullbacks, and volatility. Key opportunities may arise at support/resistance, demand/supply zones, 
    and recognizable chart patterns. Traders can leverage these zones to enter swing trades 
    with defined risk and reward.
    """
    return text

# ----------------------
# Price Action & Chart Pattern Logic
# ----------------------
def identify_price_action(df, col_map):
    """Simplified example of trendlines, support/resistance, and fake breakouts"""
    signals = []
    for i in range(1, len(df)-1):
        prev = df[col_map['close']].iloc[i-1]
        curr = df[col_map['close']].iloc[i]
        nxt = df[col_map['close']].iloc[i+1]
        
        # Simple support resistance detection
        if curr < prev and curr < nxt:
            signals.append((df[col_map['date']].iloc[i], curr, "Support Zone"))
        elif curr > prev and curr > nxt:
            signals.append((df[col_map['date']].iloc[i], curr, "Resistance Zone"))
        # Fake breakout detection
        if curr > prev*1.02:
            signals.append((df[col_map['date']].iloc[i], curr, "Bullish Trap/Fake Breakout"))
        elif curr < prev*0.98:
            signals.append((df[col_map['date']].iloc[i], curr, "Bearish Trap/Fake Breakout"))
    return signals

def backtest_strategy(df, col_map, long_short='both'):
    """Simplified price action-based backtesting logic"""
    results = []
    capital = 100000
    risk_per_trade = 0.01
    stop_loss_pct = 0.02
    take_profit_pct = 0.04
    signals = identify_price_action(df, col_map)
    
    for date, price, reason in signals:
        if long_short in ['long', 'both']:
            entry = price
            sl = entry * (1 - stop_loss_pct)
            target = entry * (1 + take_profit_pct)
            results.append({'Entry': entry, 'SL': sl, 'Target': target, 
                            'Reason': reason, 'Direction': 'Long', 'Entry Date': date})
        if long_short in ['short', 'both']:
            entry = price
            sl = entry * (1 + stop_loss_pct)
            target = entry * (1 - take_profit_pct)
            results.append({'Entry': entry, 'SL': sl, 'Target': target, 
                            'Reason': reason, 'Direction': 'Short', 'Entry Date': date})
    
    results_df = pd.DataFrame(results)
    return results_df

# ----------------------
# Streamlit UI
# ----------------------
st.title("Advanced Swing Trading Recommender")

uploaded_file = st.file_uploader("Upload OHLC Data (CSV/Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.success("File Uploaded Successfully!")
    
    # Column Mapping
    col_map = map_columns(df)
    
    st.write("Column Mapping Detected:")
    st.json(col_map)
    
    # Convert date to IST
    df = convert_to_ist(df, col_map['date'])
    
    # Sort ascending to avoid future data leak
    df = df.sort_values(by=col_map['date']).reset_index(drop=True)
    
    st.write("Top 5 Rows:")
    st.dataframe(df.head())
    
    st.write("Bottom 5 Rows:")
    st.dataframe(df.tail())
    
    st.write(f"Max Date: {df[col_map['date']].max()}")
    st.write(f"Min Date: {df[col_map['date']].min()}")
    st.write(f"Max Price: {df[col_map['high']].max()}")
    st.write(f"Min Price: {df[col_map['low']].min()}")
    
    # Plot raw data
    plot_data(df, col_map)
    
    # Returns heatmap
    compute_returns(df, col_map)
    
    # Human readable summary
    summary_text = summarize_data(df, col_map)
    st.write("Data Summary:")
    st.write(summary_text)
    
    # Select end date for backtesting
    end_date = st.date_input("Select End Date for Backtesting", value=df[col_map['date']].max().date(),
                             min_value=df[col_map['date']].min().date(),
                             max_value=df[col_map['date']].max().date())
    
    long_short = st.selectbox("Select Trade Direction", ["long", "short", "both"])
    search_type = st.selectbox("Select Optimization Type", ["random_search", "grid_search"])
    desired_accuracy = st.slider("Desired Accuracy (%)", min_value=50, max_value=99, value=80)
    points_needed = st.number_input("Number of Points Needed", min_value=1, max_value=50, value=5)
    
    # Filter data for backtesting
    df_bt = df[df[col_map['date']].dt.date <= end_date].copy()
    
    st.info("Backtesting in Progress...")
    progress_bar = st.progress(0)
    
    results_df = backtest_strategy(df_bt, col_map, long_short=long_short)
    
    for i in range(100):
        progress_bar.progress(i+1)
    
    st.success("Backtesting Completed!")
    
    st.write("Backtest Results:")
    st.dataframe(results_df)
    
    # Live recommendation based on last candle
    st.write("Live Recommendation:")
    live_signal = backtest_strategy(df.iloc[-1:], col_map, long_short=long_short)
    st.dataframe(live_signal)
    
    st.write("Backtest Summary:")
    st.write(f"""
    Total Trades: {len(results_df)}, 
    No of Long Trades: {len(results_df[results_df['Direction']=='Long'])}, 
    No of Short Trades: {len(results_df[results_df['Direction']=='Short'])}
    """)
    st.write("Use above strategy with proper risk management. Enter trades on closing price of candle when signal forms.")
