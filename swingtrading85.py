import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# --- State Management for Streamlit UI ---
if 'live_running' not in st.session_state:
    st.session_state.live_running = False
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'current_position' not in st.session_state:
    st.session_state.current_position = None

def fetch_padded_data(ticker, period, interval, pad_days=60):
    """Fetches extra historical data to ensure EMA calculations match TradingView."""
    try:
        # yfinance logic to extend the period for calculation padding
        # (Simplified for Streamlit deployment reliability)
        data = yf.download(ticker, period="1y", interval=interval, progress=False)
        if data.empty: return data
        
        # Calculate indicators on the full dataset
        data['EMA_Fast'] = data['Close'].ewm(span=st.session_state.get('ema_fast', 9), adjust=False).mean()
        data['EMA_Slow'] = data['Close'].ewm(span=st.session_state.get('ema_slow', 15), adjust=False).mean()
        data['ATR'] = calculate_atr(data)
        
        # Slicing to the user's requested period to display
        # Note: Period string parsing would go here depending on user input (1d, 5d, etc.)
        return data.tail(500) # Returns last 500 candles for the UI/Backtest
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return pd.DataFrame()

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def anticipatory_ema_crossover(df, fast=9, slow=15):
    """Anticipates a crossover by looking at the momentum of the EMA gap narrowing."""
    df['EMA_Diff'] = df['EMA_Fast'] - df['EMA_Slow']
    df['Diff_Momentum'] = df['EMA_Diff'] - df['EMA_Diff'].shift(1)
    # If the gap is negative (fast below slow) but momentum is strongly positive and accelerating
    buy_anticipation = (df['EMA_Diff'] < 0) & (df['Diff_Momentum'] > df['ATR'] * 0.1)
    return buy_anticipation
