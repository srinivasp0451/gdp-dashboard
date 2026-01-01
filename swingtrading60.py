import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

IST = pytz.timezone('Asia/Kolkata')

ASSET_CATEGORIES = {
    "Indian Indices": {
        "NIFTY 50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "SENSEX": "^BSESN"
    },
    "Crypto": {
        "BTC-USD": "BTC-USD",
        "ETH-USD": "ETH-USD"
    },
    "Forex": {
        "USD-INR": "USDINR=X",
        "EUR-USD": "EURUSD=X",
        "GBP-USD": "GBPUSD=X"
    },
    "Commodities": {
        "Gold": "GC=F",
        "Silver": "SI=F"
    }
}

VALID_INTERVALS = {
    "1m": ["1d", "5d"],
    "5m": ["1d", "1mo"],
    "15m": ["1mo"],
    "30m": ["1mo"],
    "1h": ["1mo"],
    "4h": ["1mo"],
    "1d": ["1mo", "1y", "2y", "5y"],
    "1wk": ["1mo", "1y", "5y", "10y", "15y", "20y"],
    "1mo": ["1y", "2y", "5y", "10y", "15y", "20y", "25y", "30y"]
}

STRATEGY_TYPES = ["EMA Crossover", "Simple Buy", "Simple Sell"]

SL_TYPES = [
    "Custom Points",
    "Trailing SL (Points)",
    "Trailing SL + Current Candle",
    "Trailing SL + Previous Candle",
    "Trailing SL + Current Swing",
    "Trailing SL + Previous Swing",
    "Trailing SL + Signal Based",
    "ATR-based",
    "Current Candle Low/High",
    "Previous Candle Low/High",
    "Current Swing Low/High",
    "Previous Swing Low/High",
    "Signal-based"
]

TARGET_TYPES = [
    "Custom Points",
    "Trailing Target (Points)",
    "Current Candle Low/High",
    "Previous Candle Low/High",
    "Current Swing Low/High",
    "Previous Swing Low/High",
    "Trailing Target + Signal Based",
    "ATR-based",
    "Risk-Reward Based",
    "Signal-based"
]

ENTRY_FILTERS = ["None", "Simple Crossover", "Strong Candle", "ATR Based"]

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    if 'live_running' not in st.session_state:
        st.session_state.live_running = False
    if 'live_data' not in st.session_state:
        st.session_state.live_data = None
    if 'live_trades' not in st.session_state:
        st.session_state.live_trades = []
    if 'live_logs' not in st.session_state:
        st.session_state.live_logs = []
    if 'in_position' not in st.session_state:
        st.session_state.in_position = False
    if 'position_type' not in st.session_state:
        st.session_state.position_type = 0
    if 'entry_price' not in st.session_state:
        st.session_state.entry_price = 0
    if 'stop_loss' not in st.session_state:
        st.session_state.stop_loss = 0
    if 'target' not in st.session_state:
        st.session_state.target = 0
    if 'entry_idx' not in st.session_state:
        st.session_state.entry_idx = 0
    if 'entry_time' not in st.session_state:
        st.session_state.entry_time = None
    if 'trailing_sl_highest' not in st.session_state:
        st.session_state.trailing_sl_highest = 0
    if 'trailing_sl_lowest' not in st.session_state:
        st.session_state.trailing_sl_lowest = 0
    if 'last_fetch_time' not in st.session_state:
        st.session_state.last_fetch_time = 0
    if 'last_signal_candle' not in st.session_state:
        st.session_state.last_signal_candle = None
    if 'sl_ref_value' not in st.session_state:
        st.session_state.sl_ref_value = None
    if 'target_ref_value' not in st.session_state:
        st.session_state.target_ref_value = None

def reset_position_state():
    """Reset all position-related state variables"""
    st.session_state.in_position = False
    st.session_state.position_type = 0
    st.session_state.entry_price = 0
    st.session_state.stop_loss = 0
    st.session_state.target = 0
    st.session_state.entry_idx = 0
    st.session_state.entry_time = None
    st.session_state.trailing_sl_highest = 0
    st.session_state.trailing_sl_lowest = 0
    st.session_state.last_signal_candle = None
    st.session_state.sl_ref_value = None
    st.session_state.target_ref_value = None

# ============================================================================
# DATA FETCHING & PROCESSING
# ============================================================================

def fetch_data_with_delay(ticker, interval, period):
    """Fetch data from yfinance with delay and proper timezone handling"""
    try:
        # Add randomized delay
        time.sleep(random.uniform(1.0, 1.5))
        
        # Fetch data
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if data.empty:
            return None
        
        # Flatten multi-index if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Select OHLCV columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[required_cols].copy()
        
        # Handle timezone conversion
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert(IST)
        else:
            data.index = data.index.tz_convert(IST)
        
        return data
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# ============================================================================
# MANUAL INDICATOR CALCULATIONS
# ============================================================================

def calculate_ema(series, period):
    """Calculate EMA manually using pandas ewm"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_ema_angle(ema_series, lookback=2):
    """Calculate EMA angle in degrees"""
    if len(ema_series) < lookback:
        return 0
    
    y_diff = ema_series.iloc[-1] - ema_series.iloc[-lookback]
    x_diff = lookback
    
    slope = y_diff / x_diff
    angle_rad = np.arctan(slope)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def calculate_atr(df, period=14):
    """Calculate ATR manually"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    
    return atr

def calculate_adx(df, period=14):
    """Calculate ADX manually"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Calculate +DM and -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smooth using Wilder's smoothing
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx

def detect_swing_points(df, lookback=5):
    """Detect swing highs and lows"""
    swing_high = pd.Series(np.nan, index=df.index)
    swing_low = pd.Series(np.nan, index=df.index)
    
    for i in range(lookback, len(df) - lookback):
        # Check swing high
        is_high = True
        for j in range(1, lookback + 1):
            if df['High'].iloc[i] <= df['High'].iloc[i-j] or df['High'].iloc[i] <= df['High'].iloc[i+j]:
                is_high = False
                break
        if is_high:
            swing_high.iloc[i] = df['High'].iloc[i]
        
        # Check swing low
        is_low = True
        for j in range(1, lookback + 1):
            if df['Low'].iloc[i] >= df['Low'].iloc[i-j] or df['Low'].iloc[i] >= df['Low'].iloc[i+j]:
                is_low = False
                break
        if is_low:
            swing_low.iloc[i] = df['Low'].iloc[i]
    
    # Forward fill swing points
    swing_high_filled = swing_high.fillna(method='ffill')
    swing_low_filled = swing_low.fillna(method='ffill')
    
    return swing_high_filled, swing_low_filled

def add_indicators(df, params):
    """Add all required indicators to dataframe"""
    df = df.copy()
    
    strategy = params['strategy']
    
    if strategy == "EMA Crossover":
        # Calculate EMAs
        df['EMA_Fast'] = calculate_ema(df['Close'], params['ema_fast'])
        df['EMA_Slow'] = calculate_ema(df['Close'], params['ema_slow'])
        df['EMA_Fast_Angle'] = df['EMA_Fast'].rolling(window=3).apply(
            lambda x: calculate_ema_angle(x, lookback=2), raw=False
        )
        
        # Calculate ADX if enabled
        if params['use_adx']:
            df['ADX'] = calculate_adx(df, params['adx_period'])
    
    # Always calculate ATR and swing points (may be needed for SL/Target)
    df['ATR'] = calculate_atr(df, 14)
    df['Swing_High'], df['Swing_Low'] = detect_swing_points(df, lookback=5)
    
    return df

# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def check_ema_crossover(df, idx, params):
    """Check for EMA crossover signal"""
    if idx < 1:
        return 0
    
    ema_fast_curr = df['EMA_Fast'].iloc[idx]
    ema_slow_curr = df['EMA_Slow'].iloc[idx]
    ema_fast_prev = df['EMA_Fast'].iloc[idx-1]
    ema_slow_prev = df['EMA_Slow'].iloc[idx-1]
    
    # Check for bullish crossover
    if ema_fast_prev <= ema_slow_prev and ema_fast_curr > ema_slow_curr:
        # Check angle
        angle = df['EMA_Fast_Angle'].iloc[idx]
        if abs(angle) >= params['min_angle']:
            # Check ADX if enabled
            if params['use_adx']:
                adx = df['ADX'].iloc[idx]
                if adx < params['min_adx']:
                    return 0
            
            # Check entry filter
            if params['entry_filter'] == "Strong Candle":
                body = abs(df['Close'].iloc[idx] - df['Open'].iloc[idx])
                candle_range = df['High'].iloc[idx] - df['Low'].iloc[idx]
                if body < 0.6 * candle_range:
                    return 0
            elif params['entry_filter'] == "ATR Based":
                body = abs(df['Close'].iloc[idx] - df['Open'].iloc[idx])
                atr = df['ATR'].iloc[idx]
                if body < 0.5 * atr:
                    return 0
            
            return 1  # LONG signal
    
    # Check for bearish crossover
    elif ema_fast_prev >= ema_slow_prev and ema_fast_curr < ema_slow_curr:
        # Check angle
        angle = df['EMA_Fast_Angle'].iloc[idx]
        if abs(angle) >= params['min_angle']:
            # Check ADX if enabled
            if params['use_adx']:
                adx = df['ADX'].iloc[idx]
                if adx < params['min_adx']:
                    return 0
            
            # Check entry filter
            if params['entry_filter'] == "Strong Candle":
                body = abs(df['Close'].iloc[idx] - df['Open'].iloc[idx])
                candle_range = df['High'].iloc[idx] - df['Low'].iloc[idx]
                if body < 0.6 * candle_range:
                    return 0
            elif params['entry_filter'] == "ATR Based":
                body = abs(df['Close'].iloc[idx] - df['Open'].iloc[idx])
                atr = df['ATR'].iloc[idx]
                if body < 0.5 * atr:
                    return 0
            
            return -1  # SHORT signal
    
    return 0

def check_signal_based_exit(df, idx, position_type, params):
    """Check for signal-based exit (reverse crossover)"""
    if params['strategy'] != "EMA Crossover":
        return False
    
    if idx < 1:
        return False
    
    ema_fast_curr = df['EMA_Fast'].iloc[idx]
    ema_slow_curr = df['EMA_Slow'].iloc[idx]
    ema_fast_prev = df['EMA_Fast'].iloc[idx-1]
    ema_slow_prev = df['EMA_Slow'].iloc[idx-1]
    
    # For LONG: exit when EMA_Fast crosses below EMA_Slow
    if position_type == 1:
        if ema_fast_prev >= ema_slow_prev and ema_fast_curr < ema_slow_curr:
            return True
    
    # For SHORT: exit when EMA_Fast crosses above EMA_Slow
    elif position_type == -1:
        if ema_fast_prev <= ema_slow_prev and ema_fast_curr > ema_slow_curr:
            return True
    
    return False

# ============================================================================
# STOP LOSS & TARGET CALCULATION
# ============================================================================

def calculate_stop_loss(df, idx, position_type, params):
    """Calculate initial stop loss based on selected type"""
    sl_type = params['sl_type']
    sl_points = params['sl_points']
    current_price = df['Close'].iloc[idx]
    ref_value = None
    
    if sl_type == "Custom Points":
        if position_type == 1:  # LONG
            sl = current_price - sl_points
        else:  # SHORT
            sl = current_price + sl_points
    
    elif sl_type in ["Trailing SL (Points)", "Trailing SL + Signal Based"]:
        if position_type == 1:
            sl = current_price - sl_points
        else:
            sl = current_price + sl_points
    
    elif sl_type == "Trailing SL + Current Candle":
        if position_type == 1:
            ref_value = df['Low'].iloc[idx]
            sl = ref_value - sl_points
        else:
            ref_value = df['High'].iloc[idx]
            sl = ref_value + sl_points
    
    elif sl_type == "Trailing SL + Previous Candle":
        if idx > 0:
            if position_type == 1:
                ref_value = df['Low'].iloc[idx-1]
                sl = ref_value - sl_points
            else:
                ref_value = df['High'].iloc[idx-1]
                sl = ref_value + sl_points
        else:
            sl = current_price - sl_points if position_type == 1 else current_price + sl_points
    
    elif sl_type == "Trailing SL + Current Swing":
        if position_type == 1:
            ref_value = df['Swing_Low'].iloc[idx]
            if pd.isna(ref_value):
                ref_value = df['Low'].iloc[idx]
            sl = ref_value - sl_points
        else:
            ref_value = df['Swing_High'].iloc[idx]
            if pd.isna(ref_value):
                ref_value = df['High'].iloc[idx]
            sl = ref_value + sl_points
    
    elif sl_type == "Trailing SL + Previous Swing":
        if idx > 0:
            if position_type == 1:
                ref_value = df['Swing_Low'].iloc[idx-1]
                if pd.isna(ref_value):
                    ref_value = df['Low'].iloc[idx-1]
                sl = ref_value - sl_points
            else:
                ref_value = df['Swing_High'].iloc[idx-1]
                if pd.isna(ref_value):
                    ref_value = df['High'].iloc[idx-1]
                sl = ref_value + sl_points
        else:
            sl = current_price - sl_points if position_type == 1 else current_price + sl_points
    
    elif sl_type == "ATR-based":
        atr = df['ATR'].iloc[idx]
        if position_type == 1:
            sl = current_price - (2 * atr)
        else:
            sl = current_price + (2 * atr)
    
    elif sl_type == "Current Candle Low/High":
        if position_type == 1:
            ref_value = df['Low'].iloc[idx]
            sl = ref_value
        else:
            ref_value = df['High'].iloc[idx]
            sl = ref_value
    
    elif sl_type == "Previous Candle Low/High":
        if idx > 0:
            if position_type == 1:
                ref_value = df['Low'].iloc[idx-1]
                sl = ref_value
            else:
                ref_value = df['High'].iloc[idx-1]
                sl = ref_value
        else:
            sl = df['Low'].iloc[idx] if position_type == 1 else df['High'].iloc[idx]
    
    elif sl_type == "Current Swing Low/High":
        if position_type == 1:
            ref_value = df['Swing_Low'].iloc[idx]
            if pd.isna(ref_value):
                ref_value = df['Low'].iloc[idx]
            sl = ref_value
        else:
            ref_value = df['Swing_High'].iloc[idx]
            if pd.isna(ref_value):
                ref_value = df['High'].iloc[idx]
            sl = ref_value
    
    elif sl_type == "Previous Swing Low/High":
        if idx > 0:
            if position_type == 1:
                ref_value = df['Swing_Low'].iloc[idx-1]
                if pd.isna(ref_value):
                    ref_value = df['Low'].iloc[idx-1]
                sl = ref_value
            else:
                ref_value = df['Swing_High'].iloc[idx-1]
                if pd.isna(ref_value):
                    ref_value = df['High'].iloc[idx-1]
                sl = ref_value
        else:
            sl = df['Low'].iloc[idx] if position_type == 1 else df['High'].iloc[idx]
    
    elif sl_type == "Signal-based":
        # Initial SL based on points, will be updated on signal
        if position_type == 1:
            sl = current_price - sl_points
        else:
            sl = current_price + sl_points
    
    else:
        # Default
        if position_type == 1:
            sl = current_price - sl_points
        else:
            sl = current_price + sl_points
    
    return sl, ref_value

def calculate_target(df, idx, position_type, params, entry_price, stop_loss):
    """Calculate initial target based on selected type"""
    target_type = params['target_type']
    target_points = params['target_points']
    current_price = df['Close'].iloc[idx]
    ref_value = None
    
    if target_type == "Custom Points":
        if position_type == 1:  # LONG
            target = current_price + target_points
        else:  # SHORT
            target = current_price - target_points
    
    elif target_type in ["Trailing Target (Points)", "Trailing Target + Signal Based"]:
        if position_type == 1:
            target = current_price + target_points
        else:
            target = current_price - target_points
    
    elif target_type == "Current Candle Low/High":
        if position_type == 1:
            ref_value = df['High'].iloc[idx]
            target = ref_value
        else:
            ref_value = df['Low'].iloc[idx]
            target = ref_value
    
    elif target_type == "Previous Candle Low/High":
        if idx > 0:
            if position_type == 1:
                ref_value = df['High'].iloc[idx-1]
                target = ref_value
            else:
                ref_value = df['Low'].iloc[idx-1]
                target = ref_value
        else:
            target = df['High'].iloc[idx] if position_type == 1 else df['Low'].iloc[idx]
    
    elif target_type == "Current Swing Low/High":
        if position_type == 1:
            ref_value = df['Swing_High'].iloc[idx]
            if pd.isna(ref_value):
                ref_value = df['High'].iloc[idx]
            target = ref_value
        else:
            ref_value = df['Swing_Low'].iloc[idx]
            if pd.isna(ref_value):
                ref_value = df['Low'].iloc[idx]
            target = ref_value
    
    elif target_type == "Previous Swing Low/High":
        if idx > 0:
            if position_type == 1:
                ref_value = df['Swing_High'].iloc[idx-1]
                if pd.isna(ref_value):
                    ref_value = df['High'].iloc[idx-1]
                target = ref_value
            else:
                ref_value = df['Swing_Low'].iloc[idx-1]
                if pd.isna(ref_value):
                    ref_value = df['Low'].iloc[idx-1]
                target = ref_value
        else:
            target = df['High'].iloc[idx] if position_type == 1 else df['Low'].iloc[idx]
    
    elif target_type == "ATR-based":
        atr = df['ATR'].iloc[idx]
        if position_type == 1:
            target = current_price + (2 * atr)
        else:
            target = current_price - (2 * atr)
    
    elif target_type == "Risk-Reward Based":
        risk = abs(entry_price - stop_loss)
        reward = risk * 2  # 1:2 risk-reward
        if position_type == 1:
            target = entry_price + reward
        else:
            target = entry_price - reward
    
    elif target_type == "Signal-based":
        # Initial target based on points, will be updated on signal
        if position_type == 1:
            target = current_price + target_points
        else:
            target = current_price - target_points
    
    else:
        # Default
        if position_type == 1:
            target = current_price + target_points
        else:
            target = current_price - target_points
    
    return target, ref_value

def update_trailing_sl(df, idx, params):
    """Update trailing stop loss if applicable"""
    sl_type = params['sl_type']
    trailing_threshold = params['trailing_threshold']
    position_type = st.session_state.position_type
    entry_price = st.session_state.entry_price
    current_price = df['Close'].iloc[idx]
    current_sl = st.session_state.stop_loss
    
    # Check if trailing is applicable
    if sl_type not in ["Trailing SL (Points)", "Trailing SL + Current Candle", 
                       "Trailing SL + Previous Candle", "Trailing SL + Current Swing",
                       "Trailing SL + Previous Swing", "Trailing SL + Signal Based"]:
        return current_sl, st.session_state.sl_ref_value
    
    # Initialize tracking variables
    if st.session_state.trailing_sl_highest == 0:
        st.session_state.trailing_sl_highest = current_price
    if st.session_state.trailing_sl_lowest == 0:
        st.session_state.trailing_sl_lowest = current_price
    
    # Update highest/lowest
    if current_price > st.session_state.trailing_sl_highest:
        st.session_state.trailing_sl_highest = current_price
    if current_price < st.session_state.trailing_sl_lowest:
        st.session_state.trailing_sl_lowest = current_price
    
    new_sl = current_sl
    ref_value = st.session_state.sl_ref_value
    
    if position_type == 1:  # LONG
        # Check if price moved beyond threshold
        price_move = current_price - entry_price
        if price_move >= trailing_threshold:
            if sl_type == "Trailing SL (Points)":
                new_sl = current_price - params['sl_points']
            elif sl_type == "Trailing SL + Current Candle":
                ref_value = df['Low'].iloc[idx]
                new_sl = ref_value - params['sl_points']
            elif sl_type == "Trailing SL + Previous Candle":
                if idx > 0:
                    ref_value = df['Low'].iloc[idx-1]
                    new_sl = ref_value - params['sl_points']
            elif sl_type == "Trailing SL + Current Swing":
                ref_value = df['Swing_Low'].iloc[idx]
                if pd.isna(ref_value):
                    ref_value = df['Low'].iloc[idx]
                new_sl = ref_value - params['sl_points']
            elif sl_type == "Trailing SL + Previous Swing":
                if idx > 0:
                    ref_value = df['Swing_Low'].iloc[idx-1]
                    if pd.isna(ref_value):
                        ref_value = df['Low'].iloc[idx-1]
                    new_sl = ref_value - params['sl_points']
            elif sl_type == "Trailing SL + Signal Based":
                new_sl = current_price - params['sl_points']
            
            # Only update if new SL is higher
            if new_sl > current_sl:
                return new_sl, ref_value
    
    else:  # SHORT
        # Check if price moved beyond threshold
        price_move = entry_price - current_price
        if price_move >= trailing_threshold:
            if sl_type == "Trailing SL (Points)":
                new_sl = current_price + params['sl_points']
            elif sl_type == "Trailing SL + Current Candle":
                ref_value = df['High'].iloc[idx]
                new_sl = ref_value + params['sl_points']
            elif sl_type == "Trailing SL + Previous Candle":
                if idx > 0:
                    ref_value = df['High'].iloc[idx-1]
                    new_sl = ref_value + params['sl_points']
            elif sl_type == "Trailing SL + Current Swing":
                ref_value = df['Swing_High'].iloc[idx]
                if pd.isna(ref_value):
                    ref_value = df['High'].iloc[idx]
                new_sl = ref_value + params['sl_points']
            elif sl_type == "Trailing SL + Previous Swing":
                if idx > 0:
                    ref_value = df['Swing_High'].iloc[idx-1]
                    if pd.isna(ref_value):
                        ref_value = df['High'].iloc[idx-1]
                    new_sl = ref_value + params['sl_points']
            elif sl_type == "Trailing SL + Signal Based":
                new_sl = current_price + params['sl_points']
            
            # Only update if new SL is lower
            if new_sl < current_sl:
                return new_sl, ref_value
    
    return current_sl, ref_value

def update_trailing_target(df, idx, params):
    """Update trailing target if applicable"""
    target_type = params['target_type']
    position_type = st.session_state.position_type
    current_price = df['Close'].iloc[idx]
    current_target = st.session_state.target
    
    # Check if trailing is applicable
    if target_type not in ["Trailing Target (Points)", "Trailing Target + Signal Based"]:
        return current_target, st.session_state.target_ref_value
    
    new_target = current_target
    ref_value = st.session_state.target_ref_value
    
    if position_type == 1:  # LONG
        if target_type in ["Trailing Target (Points)", "Trailing Target + Signal Based"]:
            new_target = current_price + params['target_points']
            # Only update if new target is higher
            if new_target > current_target:
                return new_target, ref_value
    
    else:  # SHORT
        if target_type in ["Trailing Target (Points)", "Trailing Target + Signal Based"]:
            new_target = current_price - params['target_points']
            # Only update if new target is lower
            if new_target < current_target:
                return new_target, ref_value
    
    return current_target, ref_value

# ============================================================================
# TRADE EXECUTION & LOGGING
# ============================================================================

def execute_entry(df, idx, signal, params):
    """Execute entry trade"""
    if st.session_state.in_position:
        return
    
    # Prevent duplicate entries on same candle
    candle_time = df.index[idx]
    if st.session_state.last_signal_candle == candle_time:
        return
    
    st.session_state.last_signal_candle = candle_time
    
    entry_price = df['Close'].iloc[idx]
    position_type = signal
    
    # Calculate SL and Target
    sl, sl_ref = calculate_stop_loss(df, idx, position_type, params)
    target, target_ref = calculate_target(df, idx, position_type, params, entry_price, sl)
    
    # Update session state
    st.session_state.in_position = True
    st.session_state.position_type = position_type
    st.session_state.entry_price = entry_price
    st.session_state.stop_loss = sl
    st.session_state.target = target
    st.session_state.entry_idx = idx
    st.session_state.entry_time = df.index[idx]
    st.session_state.trailing_sl_highest = entry_price
    st.session_state.trailing_sl_lowest = entry_price
    st.session_state.sl_ref_value = sl_ref
    st.session_state.target_ref_value = target_ref
    
    # Log entry
    position_str = "LONG" if position_type == 1 else "SHORT"
    log_msg = f"ENTRY: {position_str} @ {entry_price:.2f} | SL: {sl:.2f} | Target: {target:.2f}"
    add_log(log_msg, "entry")
    
    # Placeholder for Dhan order placement
    # place_order_dhan(symbol=ticker, quantity=qty, order_type="BUY" if position_type==1 else "SELL", price=entry_price)

def execute_exit(df, idx, exit_reason, params):
    """Execute exit trade"""
    if not st.session_state.in_position:
        return
    
    exit_price = df['Close'].iloc[idx]
    entry_price = st.session_state.entry_price
    position_type = st.session_state.position_type
    
    # Calculate PnL
    if position_type == 1:  # LONG
        pnl = exit_price - entry_price
    else:  # SHORT
        pnl = entry_price - exit_price
    
    # Record trade
    trade = {
        'Entry Time': st.session_state.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
        'Exit Time': df.index[idx].strftime('%Y-%m-%d %H:%M:%S'),
        'Position': "LONG" if position_type == 1 else "SHORT",
        'Entry Price': entry_price,
        'Exit Price': exit_price,
        'SL': st.session_state.stop_loss,
        'Target': st.session_state.target,
        'PnL': pnl,
        'Exit Reason': exit_reason
    }
    st.session_state.live_trades.append(trade)
    
    # Log exit
    log_color = "win" if pnl > 0 else "loss"
    log_msg = f"EXIT: {exit_reason} @ {exit_price:.2f} | PnL: {pnl:.2f}"
    add_log(log_msg, log_color)
    
    # Placeholder for Dhan order placement
    # place_order_dhan(symbol=ticker, quantity=qty, order_type="SELL" if position_type==1 else "BUY", price=exit_price)
    
    # Reset position
    reset_position_state()

def add_log(message, log_type="info"):
    """Add log entry with color coding"""
    timestamp = datetime.now(IST).strftime('%H:%M:%S')
    log_entry = {
        'time': timestamp,
        'message': message,
        'type': log_type
    }
    st.session_state.live_logs.append(log_entry)
    
    # Keep only last 100 logs
    if len(st.session_state.live_logs) > 100:
        st.session_state.live_logs = st.session_state.live_logs[-100:]

# ============================================================================
# LIVE TRADING LOGIC
# ============================================================================

def process_live_tick(df, idx, params):
    """Process each live tick"""
    if df is None or len(df) == 0:
        return
    
    current_price = df['Close'].iloc[idx]
    
    # Check if in position
    if st.session_state.in_position:
        position_type = st.session_state.position_type
        
        # Update trailing SL
        new_sl, new_sl_ref = update_trailing_sl(df, idx, params)
        if new_sl != st.session_state.stop_loss:
            st.session_state.stop_loss = new_sl
            st.session_state.sl_ref_value = new_sl_ref
        
        # Update trailing target
        new_target, new_target_ref = update_trailing_target(df, idx, params)
        if new_target != st.session_state.target:
            st.session_state.target = new_target
            st.session_state.target_ref_value = new_target_ref
        
        # Check for signal-based exit
        if params['sl_type'] in ["Signal-based", "Trailing SL + Signal Based"]:
            if check_signal_based_exit(df, idx, position_type, params):
                execute_exit(df, idx, "Signal-based Exit", params)
                return
        
        if params['target_type'] in ["Signal-based", "Trailing Target + Signal Based"]:
            if check_signal_based_exit(df, idx, position_type, params):
                execute_exit(df, idx, "Signal-based Exit (Target)", params)
                return
        
        # Check SL hit
        if position_type == 1:  # LONG
            if current_price <= st.session_state.stop_loss:
                execute_exit(df, idx, "Stop Loss Hit", params)
                return
        else:  # SHORT
            if current_price >= st.session_state.stop_loss:
                execute_exit(df, idx, "Stop Loss Hit", params)
                return
        
        # Check Target hit
        if position_type == 1:  # LONG
            if current_price >= st.session_state.target:
                execute_exit(df, idx, "Target Hit", params)
                return
        else:  # SHORT
            if current_price <= st.session_state.target:
                execute_exit(df, idx, "Target Hit", params)
                return
    
    else:
        # Check for entry signal
        strategy = params['strategy']
        
        if strategy == "EMA Crossover":
            signal = check_ema_crossover(df, idx, params)
            if signal != 0:
                execute_entry(df, idx, signal, params)
        
        elif strategy == "Simple Buy":
            # Enter immediately on first tick
            if st.session_state.last_signal_candle is None:
                execute_entry(df, idx, 1, params)
        
        elif strategy == "Simple Sell":
            # Enter immediately on first tick
            if st.session_state.last_signal_candle is None:
                execute_entry(df, idx, -1, params)

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_chart(df, params, entry_price=None, sl=None, target=None):
    """Create candlestick chart with indicators"""
    fig = go.Figure()
    
    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # Add EMAs if EMA Crossover strategy
    if params['strategy'] == "EMA Crossover":
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['EMA_Fast'],
            mode='lines',
            name=f"EMA {params['ema_fast']}",
            line=dict(color='blue', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['EMA_Slow'],
            mode='lines',
            name=f"EMA {params['ema_slow']}",
            line=dict(color='orange', width=1)
        ))
    
    # Add position lines if in position
    if entry_price is not None:
        # Entry line
        fig.add_hline(y=entry_price, line_dash="dash", line_color="white", 
                      annotation_text="Entry", annotation_position="right")
        
        if sl is not None:
            fig.add_hline(y=sl, line_dash="dash", line_color="red",
                          annotation_text="SL", annotation_position="right")
        
        if target is not None:
            fig.add_hline(y=target, line_dash="dash", line_color="green",
                          annotation_text="Target", annotation_position="right")
    
    fig.update_layout(
        title="Live Trading Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        height=500,
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )
    
    return fig

# ============================================================================
# UI COMPONENTS
# ============================================================================

def display_strategy_config(params):
    """Display strategy configuration"""
    st.markdown("### 📊 Strategy Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Strategy", params['strategy'])
        if params['strategy'] == "EMA Crossover":
            st.metric("EMA Fast", params['ema_fast'])
            st.metric("EMA Slow", params['ema_slow'])
    
    with col2:
        if params['strategy'] == "EMA Crossover":
            st.metric("Min Angle", f"{params['min_angle']}°")
            st.metric("ADX Filter", "ON" if params['use_adx'] else "OFF")
            if params['use_adx']:
                st.metric("Min ADX", params['min_adx'])
    
    with col3:
        st.metric("SL Type", params['sl_type'])
        st.metric("SL Points", params['sl_points'])
        st.metric("Target Type", params['target_type'])
        st.metric("Target Points", params['target_points'])

def display_live_market_data(df, params):
    """Display live market data"""
    st.markdown("### 📈 Live Market Data")
    
    if df is None or len(df) == 0:
        st.warning("No market data available")
        return
    
    latest = df.iloc[-1]
    prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest['Close']
    
    current_price = latest['Close']
    price_change = current_price - prev_close
    price_change_pct = (price_change / prev_close) * 100 if prev_close != 0 else 0
    
    # Time since last update
    time_diff = time.time() - st.session_state.last_fetch_time
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"{current_price:.2f}", 
                  f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
    
    if params['strategy'] == "EMA Crossover":
        with col2:
            st.metric("EMA Fast", f"{latest['EMA_Fast']:.2f}")
        with col3:
            st.metric("EMA Slow", f"{latest['EMA_Slow']:.2f}")
        with col4:
            if params['use_adx']:
                st.metric("ADX", f"{latest['ADX']:.2f}")
            else:
                st.metric("Volume", f"{latest['Volume']:.0f}")
    else:
        with col2:
            st.metric("Open", f"{latest['Open']:.2f}")
        with col3:
            st.metric("High", f"{latest['High']:.2f}")
        with col4:
            st.metric("Low", f"{latest['Low']:.2f}")
    
    st.caption(f"Last update: {time_diff:.1f}s ago")

def display_position_status(df, params):
    """Display current position status"""
    st.markdown("### 💼 Position Status")
    
    if st.session_state.in_position:
        position_str = "🟢 LONG" if st.session_state.position_type == 1 else "🔴 SHORT"
        current_price = df['Close'].iloc[-1] if df is not None and len(df) > 0 else 0
        
        # Calculate unrealized PnL
        if st.session_state.position_type == 1:
            unrealized_pnl = current_price - st.session_state.entry_price
        else:
            unrealized_pnl = st.session_state.entry_price - current_price
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Position", position_str)
            st.metric("Entry Price", f"{st.session_state.entry_price:.2f}")
        
        with col2:
            pnl_color = "normal" if unrealized_pnl >= 0 else "inverse"
            st.metric("Unrealized P&L", f"{unrealized_pnl:.2f}", 
                      delta_color=pnl_color)
            st.caption(f"Entry: {st.session_state.entry_time.strftime('%H:%M:%S')}")
        
        with col3:
            sl_distance = abs(current_price - st.session_state.stop_loss)
            st.metric("Stop Loss", f"{st.session_state.stop_loss:.2f}")
            st.caption(f"Distance: {sl_distance:.2f}")
            if st.session_state.sl_ref_value is not None:
                st.caption(f"Ref: {st.session_state.sl_ref_value:.2f}")
        
        with col4:
            target_distance = abs(st.session_state.target - current_price)
            st.metric("Target", f"{st.session_state.target:.2f}")
            st.caption(f"Distance: {target_distance:.2f}")
            if st.session_state.target_ref_value is not None:
                st.caption(f"Ref: {st.session_state.target_ref_value:.2f}")
    
    else:
        # Calculate total PnL
        total_pnl = sum([t['PnL'] for t in st.session_state.live_trades])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Position", "⚪ NONE")
        
        with col2:
            pnl_color = "normal" if total_pnl >= 0 else "inverse"
            st.metric("Total P&L", f"{total_pnl:.2f}", delta_color=pnl_color)
        
        # Show current signal for EMA Crossover
        if params['strategy'] == "EMA Crossover" and df is not None and len(df) > 0:
            signal = check_ema_crossover(df, len(df)-1, params)
            if signal == 1:
                st.info("📊 Current Signal: LONG")
            elif signal == -1:
                st.info("📊 Current Signal: SHORT")
            else:
                st.info("📊 Current Signal: NONE")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(page_title="Trading System", layout="wide", initial_sidebar_state="expanded")
    
    # Initialize session state
    init_session_state()
    
    st.title("🚀 Professional Trading System")
    
    # ========================================================================
    # SIDEBAR CONFIGURATION
    # ========================================================================
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Asset Selection
        st.subheader("Asset Selection")
        
        asset_mode = st.radio("Select Mode", ["Predefined", "Custom Ticker"])
        
        if asset_mode == "Predefined":
            category = st.selectbox("Category", list(ASSET_CATEGORIES.keys()))
            asset_name = st.selectbox("Asset", list(ASSET_CATEGORIES[category].keys()))
            ticker = ASSET_CATEGORIES[category][asset_name]
        else:
            ticker = st.text_input("Enter Ticker Symbol", "^NSEI")
        
        # Timeframe Selection
        st.subheader("Timeframe")
        interval = st.selectbox("Interval", list(VALID_INTERVALS.keys()), index=6)
        valid_periods = VALID_INTERVALS[interval]
        period = st.selectbox("Period", valid_periods)
        
        # Strategy Selection
        st.subheader("Strategy")
        strategy = st.selectbox("Strategy Type", STRATEGY_TYPES)
        
        # Strategy Parameters
        params = {'strategy': strategy}
        
        if strategy == "EMA Crossover":
            st.markdown("**EMA Settings**")
            params['ema_fast'] = st.number_input("EMA Fast", min_value=1, max_value=200, value=9)
            params['ema_slow'] = st.number_input("EMA Slow", min_value=1, max_value=200, value=15)
            params['min_angle'] = st.number_input("Min Angle (degrees)", min_value=0.0, max_value=90.0, value=1.0, step=0.1)
            
            st.markdown("**ADX Filter**")
            params['use_adx'] = st.checkbox("Enable ADX Filter", value=False)
            if params['use_adx']:
                params['min_adx'] = st.number_input("Min ADX Value", min_value=0, max_value=100, value=20)
                params['adx_period'] = st.number_input("ADX Period", min_value=1, max_value=50, value=14)
            else:
                params['min_adx'] = 20
                params['adx_period'] = 14
            
            st.markdown("**Entry Filter**")
            params['entry_filter'] = st.selectbox("Entry Filter", ENTRY_FILTERS)
        else:
            # Default values for non-EMA strategies
            params['ema_fast'] = 9
            params['ema_slow'] = 15
            params['min_angle'] = 1.0
            params['use_adx'] = False
            params['min_adx'] = 20
            params['adx_period'] = 14
            params['entry_filter'] = "None"
        
        # Stop Loss Settings
        st.subheader("Stop Loss")
        params['sl_type'] = st.selectbox("SL Type", SL_TYPES)
        params['sl_points'] = st.number_input("SL Points", min_value=0.0, value=15.0, step=0.5)
        
        if "Trailing" in params['sl_type']:
            params['trailing_threshold'] = st.number_input("Trailing Threshold (Points)", 
                                                           min_value=0.0, value=0.0, step=0.5)
        else:
            params['trailing_threshold'] = 0.0
        
        # Target Settings
        st.subheader("Target")
        params['target_type'] = st.selectbox("Target Type", TARGET_TYPES)
        params['target_points'] = st.number_input("Target Points", min_value=0.0, value=10.0, step=0.5)
    
    # ========================================================================
    # MAIN PAGE - START/STOP BUTTONS
    # ========================================================================
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if st.button("▶️ Start Trading", type="primary", use_container_width=True):
            # Fetch initial data
            df = fetch_data_with_delay(ticker, interval, period)
            if df is not None and not df.empty:
                df = add_indicators(df, params)
                st.session_state.live_data = df
                st.session_state.live_running = True
                st.session_state.last_fetch_time = time.time()
                reset_position_state()
                add_log("Trading started", "info")
                st.rerun()
            else:
                st.error("Failed to fetch data. Please check ticker and try again.")
    
    with col2:
        if st.button("⏹️ Stop Trading", type="secondary", use_container_width=True):
            # Check if in position and execute manual exit
            if st.session_state.in_position and st.session_state.live_data is not None:
                df = st.session_state.live_data
                idx = len(df) - 1
                execute_exit(df, idx, "Manual Exit - Trading Stopped", params)
            
            st.session_state.live_running = False
            add_log("Trading stopped", "info")
            st.rerun()
    
    with col3:
        if st.session_state.live_running:
            st.success("🟢 Trading is ACTIVE")
        else:
            st.info("⚪ Trading is STOPPED")
    
    st.markdown("---")
    
    # ========================================================================
    # TABS - ALWAYS VISIBLE
    # ========================================================================
    
    tab1, tab2, tab3 = st.tabs(["📊 Live Trading", "📜 Trade History", "📝 Trade Logs"])
    
    # ========================================================================
    # TAB 1: LIVE TRADING DASHBOARD
    # ========================================================================
    
    with tab1:
        if st.session_state.live_data is not None:
            df = st.session_state.live_data
            
            # Strategy Configuration (at top)
            display_strategy_config(params)
            
            st.markdown("---")
            
            # Live Market Data
            display_live_market_data(df, params)
            
            st.markdown("---")
            
            # Position Status
            display_position_status(df, params)
            
            st.markdown("---")
            
            # Chart (at bottom)
            st.markdown("### 📉 Chart")
            
            if st.session_state.in_position:
                chart = create_chart(df, params, 
                                   st.session_state.entry_price,
                                   st.session_state.stop_loss,
                                   st.session_state.target)
            else:
                chart = create_chart(df, params)
            
            # Use timestamp for unique key
            chart_key = f"chart_{int(time.time() * 1000)}"
            st.plotly_chart(chart, use_container_width=True, key=chart_key)
        
        else:
            st.info("👆 Click 'Start Trading' to begin")
    
    # ========================================================================
    # TAB 2: TRADE HISTORY
    # ========================================================================
    
    with tab2:
        if len(st.session_state.live_trades) > 0:
            trades_df = pd.DataFrame(st.session_state.live_trades)
            
            # Calculate metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['PnL'] > 0])
            losing_trades = len(trades_df[trades_df['PnL'] <= 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = trades_df['PnL'].sum()
            avg_pnl = trades_df['PnL'].mean()
            
            # Display metrics
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.metric("Total Trades", total_trades)
            with col2:
                st.metric("Winning", f"{winning_trades} ({win_rate:.1f}%)")
            with col3:
                st.metric("Losing", losing_trades)
            with col4:
                st.metric("Total P&L", f"{total_pnl:.2f}")
            with col5:
                st.metric("Avg P&L", f"{avg_pnl:.2f}")
            with col6:
                max_win = trades_df['PnL'].max()
                max_loss = trades_df['PnL'].min()
                st.metric("Max Win", f"{max_win:.2f}")
            
            st.markdown("---")
            
            # Display trade table
            st.dataframe(trades_df, use_container_width=True, height=400)
        
        else:
            st.info("No trades executed yet")
    
    # ========================================================================
    # TAB 3: TRADE LOGS
    # ========================================================================
    
    with tab3:
        if len(st.session_state.live_logs) > 0:
            for log in reversed(st.session_state.live_logs):
                if log['type'] == 'entry':
                    st.success(f"[{log['time']}] {log['message']}")
                elif log['type'] == 'loss':
                    st.error(f"[{log['time']}] {log['message']}")
                elif log['type'] == 'win':
                    st.info(f"[{log['time']}] {log['message']}")
                else:
                    st.write(f"[{log['time']}] {log['message']}")
        else:
            st.info("No logs yet")
    
    # ========================================================================
    # AUTO-REFRESH LOGIC (AT END, OUTSIDE TABS)
    # ========================================================================
    
    if st.session_state.live_running:
        current_time = time.time()
        time_diff = current_time - st.session_state.last_fetch_time
        
        if time_diff >= 1.0:
            df = fetch_data_with_delay(ticker, interval, period)
            if df is not None and not df.empty:
                df = add_indicators(df, params)
                st.session_state.live_data = df
                st.session_state.last_fetch_time = current_time
                process_live_tick(df, len(df)-1, params)
            
            time.sleep(0.1)
            st.rerun()

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
