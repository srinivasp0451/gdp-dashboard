import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import time
import random

# ==================== CONFIGURATION ====================
ASSET_MAPPING = {
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "USDINR": "USDINR=X",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "Gold": "GC=F",
    "Silver": "SI=F"
}

# Minimum initial target distances for different assets
ASSET_MIN_TARGET = {
    "^NSEI": 10,      # NIFTY
    "^NSEBANK": 20,   # BANKNIFTY
    "^BSESN": 10,     # SENSEX
    "BTC-USD": 150,   # BTC
    "ETH-USD": 10,    # ETH
}

TIMEFRAME_PERIODS = {
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

# ==================== UTILITY FUNCTIONS ====================
def get_ist_time():
    """Get current time in IST"""
    return datetime.now(pytz.timezone('Asia/Kolkata'))

def convert_to_ist(df):
    """Convert dataframe index to IST"""
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
    else:
        df.index = df.index.tz_convert('Asia/Kolkata')
    return df

def get_min_target_for_asset(ticker):
    """Get minimum initial target distance based on asset"""
    return ASSET_MIN_TARGET.get(ticker, 15)  # Default 15 if not specified

def fetch_data(ticker, interval, period):
    """Fetch data from yfinance with rate limiting"""
    time.sleep(random.uniform(1.0, 1.5))
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return None
        
        # Flatten multi-index if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in data.columns.values]
            # Clean column names
            data.columns = [col.split('_')[0] if '_' in col else col for col in data.columns]
        
        # Select OHLCV
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[required_cols]
        
        # Convert to IST
        data = convert_to_ist(data)
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# ==================== INDICATOR CALCULATIONS ====================
def calculate_ema(series, period):
    """Calculate EMA"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    """Calculate ATR"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()

def calculate_adx(df, period=14):
    """Calculate ADX"""
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    atr = calculate_atr(df, period)
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return adx

def calculate_swing_points(df, lookback=5):
    """Calculate swing highs and lows"""
    swing_high = df['High'].rolling(window=lookback*2+1, center=True).apply(
        lambda x: x[lookback] if x[lookback] == max(x) else np.nan, raw=True
    )
    swing_low = df['Low'].rolling(window=lookback*2+1, center=True).apply(
        lambda x: x[lookback] if x[lookback] == min(x) else np.nan, raw=True
    )
    return swing_high.ffill(), swing_low.ffill()

def calculate_ema_angle(ema_fast, ema_slow, idx):
    """Calculate EMA crossover angle in degrees"""
    if idx < 1:
        return 0
    
    fast_current = ema_fast.iloc[idx]
    fast_prev = ema_fast.iloc[idx-1]
    slow_current = ema_slow.iloc[idx]
    slow_prev = ema_slow.iloc[idx-1]
    
    fast_slope = fast_current - fast_prev
    slow_slope = slow_current - slow_prev
    
    angle_rad = np.arctan(fast_slope - slow_slope)
    angle_deg = np.degrees(angle_rad)
    
    return abs(angle_deg)

# ==================== SIGNAL GENERATION ====================
def check_entry_filter(df, idx, filter_type, custom_points, atr_multiplier):
    """Check if entry filter conditions are met"""
    candle_size = abs(df['Close'].iloc[idx] - df['Open'].iloc[idx])
    
    if filter_type == "Simple Crossover":
        return True, candle_size, 0
    elif filter_type == "Custom Candle (Points)":
        min_size = custom_points
        return candle_size >= min_size, candle_size, min_size
    elif filter_type == "ATR-based Candle":
        atr_value = df['ATR'].iloc[idx]
        min_size = atr_value * atr_multiplier
        return candle_size >= min_size, candle_size, min_size
    
    return False, candle_size, 0

def generate_ema_signal(df, idx, ema_fast_period, ema_slow_period, min_angle, 
                        entry_filter, custom_points, atr_multiplier, use_adx, adx_threshold):
    """Generate EMA crossover signal"""
    if idx < 1:
        return 0, None, None, None, None, None, None, None
    
    ema_fast = df['EMA_Fast'].iloc[idx]
    ema_slow = df['EMA_Slow'].iloc[idx]
    ema_fast_prev = df['EMA_Fast'].iloc[idx-1]
    ema_slow_prev = df['EMA_Slow'].iloc[idx-1]
    
    # Calculate angle
    angle = calculate_ema_angle(df['EMA_Fast'], df['EMA_Slow'], idx)
    
    # Check crossover
    bullish_cross = (ema_fast > ema_slow) and (ema_fast_prev <= ema_slow_prev)
    bearish_cross = (ema_fast < ema_slow) and (ema_fast_prev >= ema_slow_prev)
    
    # Check angle
    angle_valid = angle >= min_angle
    
    # Check entry filter
    filter_valid, candle_size, min_candle_size = check_entry_filter(
        df, idx, entry_filter, custom_points, atr_multiplier
    )
    
    # Check ADX
    adx_valid = True
    adx_value = None
    if use_adx:
        adx_value = df['ADX'].iloc[idx]
        adx_valid = adx_value >= adx_threshold
    
    signal = 0
    if bullish_cross and angle_valid and filter_valid and adx_valid:
        signal = 1
    elif bearish_cross and angle_valid and filter_valid and adx_valid:
        signal = -1
    
    return signal, angle, filter_valid, candle_size, min_candle_size, adx_valid, adx_value, ema_fast

def check_reverse_signal(df, idx, position_type):
    """Check for reverse EMA crossover (signal-based exit)"""
    if idx < 1:
        return False
    
    ema_fast = df['EMA_Fast'].iloc[idx]
    ema_slow = df['EMA_Slow'].iloc[idx]
    ema_fast_prev = df['EMA_Fast'].iloc[idx-1]
    ema_slow_prev = df['EMA_Slow'].iloc[idx-1]
    
    if position_type == 1:  # LONG
        # Check for bearish crossover
        if (ema_fast < ema_slow) and (ema_fast_prev >= ema_slow_prev):
            return True
    elif position_type == -1:  # SHORT
        # Check for bullish crossover
        if (ema_fast > ema_slow) and (ema_fast_prev <= ema_slow_prev):
            return True
    
    return False

# ==================== SL/TARGET CALCULATIONS ====================
def calculate_stop_loss(df, idx, entry_price, signal, sl_type, sl_points, 
                       trailing_threshold, atr_multiplier, current_sl=None):
    """Calculate stop loss based on selected type"""
    if sl_type == "Custom Points":
        if signal == 1:
            return entry_price - sl_points
        else:
            return entry_price + sl_points
    
    elif sl_type == "Trailing SL (Points)":
        if current_sl is None:
            if signal == 1:
                return entry_price - sl_points
            else:
                return entry_price + sl_points
        
        current_price = df['Close'].iloc[idx]
        
        if signal == 1:
            # Check if price moved enough to update trailing
            if current_price - entry_price >= trailing_threshold:
                high_price = st.session_state.get('trailing_sl_high', current_price)
                if current_price > high_price:
                    st.session_state.trailing_sl_high = current_price
                    high_price = current_price
                new_sl = high_price - sl_points
                return max(current_sl, new_sl)
        else:
            # Check if price moved enough to update trailing
            if entry_price - current_price >= trailing_threshold:
                low_price = st.session_state.get('trailing_sl_low', current_price)
                if current_price < low_price:
                    st.session_state.trailing_sl_low = current_price
                    low_price = current_price
                new_sl = low_price + sl_points
                return min(current_sl, new_sl)
        
        return current_sl
    
    elif sl_type == "Trailing SL + Current Candle":
        current_price = df['Close'].iloc[idx]
        if signal == 1:
            candle_low = df['Low'].iloc[idx]
            base_sl = candle_low - sl_points
            if current_sl is None:
                return max(entry_price - 10, base_sl)
            return max(current_sl, base_sl)
        else:
            candle_high = df['High'].iloc[idx]
            base_sl = candle_high + sl_points
            if current_sl is None:
                return min(entry_price + 10, base_sl)
            return min(current_sl, base_sl)
    
    elif sl_type == "Trailing SL + Previous Candle":
        if idx < 1:
            if signal == 1:
                return entry_price - sl_points
            else:
                return entry_price + sl_points
        
        if signal == 1:
            prev_low = df['Low'].iloc[idx-1]
            base_sl = prev_low - sl_points
            if current_sl is None:
                return max(entry_price - 10, base_sl)
            return max(current_sl, base_sl)
        else:
            prev_high = df['High'].iloc[idx-1]
            base_sl = prev_high + sl_points
            if current_sl is None:
                return min(entry_price + 10, base_sl)
            return min(current_sl, base_sl)
    
    elif sl_type == "Trailing SL + Current Swing":
        if signal == 1:
            swing_low = df['Swing_Low'].iloc[idx]
            base_sl = swing_low - sl_points
            if current_sl is None:
                return max(entry_price - 10, base_sl)
            return max(current_sl, base_sl)
        else:
            swing_high = df['Swing_High'].iloc[idx]
            base_sl = swing_high + sl_points
            if current_sl is None:
                return min(entry_price + 10, base_sl)
            return min(current_sl, base_sl)
    
    elif sl_type == "Trailing SL + Previous Swing":
        if signal == 1:
            swing_low = df['Swing_Low'].iloc[idx]
            base_sl = swing_low - sl_points
            if current_sl is None:
                return max(entry_price - 10, base_sl)
            return max(current_sl, base_sl)
        else:
            swing_high = df['Swing_High'].iloc[idx]
            base_sl = swing_high + sl_points
            if current_sl is None:
                return min(entry_price + 10, base_sl)
            return min(current_sl, base_sl)
    
    elif sl_type == "Trailing SL + Signal Based":
        # Use trailing logic but will exit on reverse signal
        if current_sl is None:
            if signal == 1:
                return entry_price - sl_points
            else:
                return entry_price + sl_points
        
        current_price = df['Close'].iloc[idx]
        
        if signal == 1:
            high_price = st.session_state.get('trailing_sl_high', current_price)
            if current_price > high_price:
                st.session_state.trailing_sl_high = current_price
                high_price = current_price
            new_sl = high_price - sl_points
            return max(current_sl, new_sl)
        else:
            low_price = st.session_state.get('trailing_sl_low', current_price)
            if current_price < low_price:
                st.session_state.trailing_sl_low = current_price
                low_price = current_price
            new_sl = low_price + sl_points
            return min(current_sl, new_sl)
    
    elif sl_type == "ATR-based":
        atr_value = df['ATR'].iloc[idx]
        if signal == 1:
            return entry_price - (atr_value * atr_multiplier)
        else:
            return entry_price + (atr_value * atr_multiplier)
    
    elif sl_type == "Current Candle Low/High":
        if signal == 1:
            return df['Low'].iloc[idx]
        else:
            return df['High'].iloc[idx]
    
    elif sl_type == "Previous Candle Low/High":
        if idx < 1:
            if signal == 1:
                return entry_price - 10
            else:
                return entry_price + 10
        if signal == 1:
            return df['Low'].iloc[idx-1]
        else:
            return df['High'].iloc[idx-1]
    
    elif sl_type == "Current Swing Low/High":
        if signal == 1:
            return df['Swing_Low'].iloc[idx]
        else:
            return df['Swing_High'].iloc[idx]
    
    elif sl_type == "Previous Swing Low/High":
        if signal == 1:
            return df['Swing_Low'].iloc[idx]
        else:
            return df['Swing_High'].iloc[idx]
    
    elif sl_type == "Signal-based":
        # No fixed price SL, will exit on reverse crossover
        return 0
    
    # Default
    if signal == 1:
        return entry_price - 10
    else:
        return entry_price + 10

def calculate_target(df, idx, entry_price, signal, target_type, target_points, 
                     atr_multiplier, risk_reward_ratio, current_target=None, min_target_distance=15):
    """Calculate target based on selected type"""
    if target_type == "Custom Points":
        if signal == 1:
            return entry_price + target_points
        else:
            return entry_price - target_points
    
    elif target_type == "Trailing Target (Points)":
        current_price = df['Close'].iloc[idx]
        
        # Initialize trailing target on first call
        if current_target is None or current_target == 0:
            if signal == 1:
                initial_target = entry_price + min_target_distance
                st.session_state.trailing_target_high = entry_price
                st.session_state.trailing_profit_points = 0
                return initial_target
            else:
                initial_target = entry_price - min_target_distance
                st.session_state.trailing_target_low = entry_price
                st.session_state.trailing_profit_points = 0
                return initial_target
        
        if signal == 1:
            # Calculate profit from entry
            profit_from_entry = current_price - entry_price
            
            # Update highest price reached
            highest = st.session_state.get('trailing_target_high', entry_price)
            if current_price > highest:
                st.session_state.trailing_target_high = current_price
                highest = current_price
            
            # Calculate profit points moved from initial entry
            profit_points = highest - entry_price
            prev_profit_points = st.session_state.get('trailing_profit_points', 0)
            
            # Check if we've moved target_points from last update
            if profit_points >= prev_profit_points + target_points:
                # Update the tracking
                st.session_state.trailing_profit_points = profit_points
                # New target is highest - target_points (trailing behind)
                new_target = highest
                return new_target
            
            # Return current target if threshold not met
            return current_target
        
        else:  # SHORT
            # Calculate profit from entry
            profit_from_entry = entry_price - current_price
            
            # Update lowest price reached
            lowest = st.session_state.get('trailing_target_low', entry_price)
            if current_price < lowest:
                st.session_state.trailing_target_low = current_price
                lowest = current_price
            
            # Calculate profit points moved from initial entry
            profit_points = entry_price - lowest
            prev_profit_points = st.session_state.get('trailing_profit_points', 0)
            
            # Check if we've moved target_points from last update
            if profit_points >= prev_profit_points + target_points:
                # Update the tracking
                st.session_state.trailing_profit_points = profit_points
                # New target is lowest + target_points (trailing behind)
                new_target = lowest
                return new_target
            
            # Return current target if threshold not met
            return current_target
    
    elif target_type == "Trailing Target + Signal Based":
        current_price = df['Close'].iloc[idx]
        
        if current_target is None or current_target == 0:
            if signal == 1:
                initial_target = entry_price + min_target_distance
                st.session_state.trailing_target_high = entry_price
                st.session_state.trailing_profit_points = 0
                return initial_target
            else:
                initial_target = entry_price - min_target_distance
                st.session_state.trailing_target_low = entry_price
                st.session_state.trailing_profit_points = 0
                return initial_target
        
        if signal == 1:
            highest = st.session_state.get('trailing_target_high', entry_price)
            if current_price > highest:
                st.session_state.trailing_target_high = current_price
                highest = current_price
            
            profit_points = highest - entry_price
            prev_profit_points = st.session_state.get('trailing_profit_points', 0)
            
            if profit_points >= prev_profit_points + target_points:
                st.session_state.trailing_profit_points = profit_points
                return highest
            
            return current_target
        else:
            lowest = st.session_state.get('trailing_target_low', entry_price)
            if current_price < lowest:
                st.session_state.trailing_target_low = current_price
                lowest = current_price
            
            profit_points = entry_price - lowest
            prev_profit_points = st.session_state.get('trailing_profit_points', 0)
            
            if profit_points >= prev_profit_points + target_points:
                st.session_state.trailing_profit_points = profit_points
                return lowest
            
            return current_target
    
    elif target_type == "Current Candle Low/High":
        if signal == 1:
            return df['High'].iloc[idx]
        else:
            return df['Low'].iloc[idx]
    
    elif target_type == "Previous Candle Low/High":
        if idx < 1:
            if signal == 1:
                return entry_price + 15
            else:
                return entry_price - 15
        if signal == 1:
            return df['High'].iloc[idx-1]
        else:
            return df['Low'].iloc[idx-1]
    
    elif target_type == "Current Swing Low/High":
        if signal == 1:
            return df['Swing_High'].iloc[idx]
        else:
            return df['Swing_Low'].iloc[idx]
    
    elif target_type == "Previous Swing Low/High":
        if signal == 1:
            return df['Swing_High'].iloc[idx]
        else:
            return df['Swing_Low'].iloc[idx]
    
    elif target_type == "ATR-based":
        atr_value = df['ATR'].iloc[idx]
        if signal == 1:
            return entry_price + (atr_value * atr_multiplier)
        else:
            return entry_price - (atr_value * atr_multiplier)
    
    elif target_type == "Risk-Reward Based":
        # Calculate based on SL distance
        sl_distance = abs(entry_price - df['SL'].iloc[idx]) if 'SL' in df.columns else 10
        if signal == 1:
            return entry_price + (sl_distance * risk_reward_ratio)
        else:
            return entry_price - (sl_distance * risk_reward_ratio)
    
    elif target_type == "Signal-based":
        # No fixed target, will exit on reverse crossover
        return 0
    
    # Default
    if signal == 1:
        return entry_price + 15
    else:
        return entry_price - 15

# ==================== LIVE TRADING FUNCTIONS ====================
def add_trade_log(message):
    """Add timestamped log entry - only important events"""
    if 'trade_logs' not in st.session_state:
        st.session_state.trade_logs = []
    
    # Keep only last 50 logs to prevent memory overflow
    if len(st.session_state.trade_logs) > 50:
        st.session_state.trade_logs = st.session_state.trade_logs[-50:]
    
    timestamp = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.trade_logs.append(f"[{timestamp}] {message}")

def add_trade_to_history(trade_data):
    """Add completed trade to history"""
    if 'trade_history' not in st.session_state:
        st.session_state.trade_history = []
    
    st.session_state.trade_history.append(trade_data)

def create_live_chart(df, position=None):
    """Create live candlestick chart with indicators"""
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # EMAs
    if 'EMA_Fast' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['EMA_Fast'],
            mode='lines',
            name='EMA Fast',
            line=dict(color='blue', width=1)
        ))
    
    if 'EMA_Slow' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['EMA_Slow'],
            mode='lines',
            name='EMA Slow',
            line=dict(color='orange', width=1)
        ))
    
    # Position lines
    if position:
        entry_price = position['entry_price']
        sl_price = position['sl']
        target_price = position['target']
        
        # Entry line
        fig.add_hline(y=entry_price, line_dash="dash", 
                     line_color="yellow", annotation_text="Entry")
        
        # SL line
        if sl_price and sl_price != 0:
            fig.add_hline(y=sl_price, line_dash="dash", 
                         line_color="red", annotation_text="Stop Loss")
        
        # Target line
        if target_price and target_price != 0:
            fig.add_hline(y=target_price, line_dash="dash", 
                         line_color="green", annotation_text="Target")
    
    fig.update_layout(
        title="Live Trading Chart",
        xaxis_title="Time (IST)",
        yaxis_title="Price",
        height=600,
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )
    
    return fig

# ==================== MAIN APPLICATION ====================
def initialize_session_state():
    """Initialize session state variables"""
    if 'trading_active' not in st.session_state:
        st.session_state.trading_active = False
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'position' not in st.session_state:
        st.session_state.position = None
    if 'trade_history' not in st.session_state:
        st.session_state.trade_history = []
    if 'trade_logs' not in st.session_state:
        st.session_state.trade_logs = []
    if 'trailing_sl_high' not in st.session_state:
        st.session_state.trailing_sl_high = None
    if 'trailing_sl_low' not in st.session_state:
        st.session_state.trailing_sl_low = None
    if 'trailing_target_high' not in st.session_state:
        st.session_state.trailing_target_high = None
    if 'trailing_target_low' not in st.session_state:
        st.session_state.trailing_target_low = None
    if 'trailing_profit_points' not in st.session_state:
        st.session_state.trailing_profit_points = 0
    if 'last_fetch_time' not in st.session_state:
        st.session_state.last_fetch_time = None
    if 'threshold_crossed' not in st.session_state:
        st.session_state.threshold_crossed = False

def main():
    st.set_page_config(page_title="Live Trading Dashboard", layout="wide")
    st.title("🚀 Live Trading Dashboard - Production Grade")
    
    initialize_session_state()
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Trading Configuration")
        
        # Asset Selection
        asset_type = st.selectbox("Asset Type", 
                                 ["Predefined", "Custom Ticker"])
        
        if asset_type == "Predefined":
            asset = st.selectbox("Select Asset", list(ASSET_MAPPING.keys()))
            ticker = ASSET_MAPPING[asset]
        else:
            ticker = st.text_input("Enter Ticker Symbol", "AAPL")
            asset = ticker
        
        # Timeframe Selection
        interval = st.selectbox("Timeframe", list(TIMEFRAME_PERIODS.keys()))
        allowed_periods = TIMEFRAME_PERIODS[interval]
        period = st.selectbox("Period", allowed_periods)
        
        st.divider()
        
        # Strategy Selection
        strategy = st.selectbox("Strategy", 
                               ["EMA Crossover", "Simple Buy", "Simple Sell",
                                "Price Crosses Threshold"])
        
        # Strategy Parameters
        if strategy == "EMA Crossover":
            st.subheader("EMA Parameters")
            ema_fast = st.number_input("EMA Fast", min_value=1, value=9)
            ema_slow = st.number_input("EMA Slow", min_value=1, value=15)
            min_angle = st.number_input("Min Crossover Angle (°)", 
                                       min_value=0.0, value=1.0, step=0.1)
            
            st.subheader("Entry Filter")
            entry_filter = st.selectbox("Filter Type", 
                                       ["Simple Crossover", 
                                        "Custom Candle (Points)", 
                                        "ATR-based Candle"])
            
            custom_points = 0
            atr_multiplier = 1.0
            
            if entry_filter == "Custom Candle (Points)":
                custom_points = st.number_input("Minimum Candle Points", 
                                               min_value=1, value=10)
            elif entry_filter == "ATR-based Candle":
                atr_multiplier = st.number_input("ATR Multiplier", 
                                                min_value=0.1, value=1.0, step=0.1)
            
            st.subheader("ADX Filter")
            use_adx = st.checkbox("Enable ADX Filter", value=False)
            adx_period = 14
            adx_threshold = 25
            if use_adx:
                adx_period = st.number_input("ADX Period", min_value=1, value=14)
                adx_threshold = st.number_input("ADX Threshold", 
                                               min_value=1, value=25)
        
        elif strategy == "Price Crosses Threshold":
            threshold_price = st.number_input("Threshold Price", 
                                             min_value=0.0, value=100.0, step=0.01)
            threshold_direction = st.selectbox("Entry Direction", 
                                              ["LONG (Price >= Threshold)", 
                                               "SHORT (Price >= Threshold)",
                                               "LONG (Price <= Threshold)",
                                               "SHORT (Price <= Threshold)"])
        
        st.divider()
        
        # Stop Loss Configuration
        st.subheader("Stop Loss")
        sl_type = st.selectbox("SL Type", [
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
        ])
        
        sl_points = 10
        trailing_threshold = 0
        sl_atr_multiplier = 1.5
        
        if "Points" in sl_type or "Trailing" in sl_type:
            sl_points = st.number_input("SL Points", min_value=1, value=10)
        
        if "Trailing" in sl_type:
            trailing_threshold = st.number_input("Trailing Threshold (Points)", 
                                                min_value=0, value=0)
        
        if sl_type == "ATR-based":
            sl_atr_multiplier = st.number_input("SL ATR Multiplier", 
                                               min_value=0.1, value=1.5, step=0.1)
        
        st.divider()
        
        # Target Configuration
        st.subheader("Target")
        target_type = st.selectbox("Target Type", [
            "Custom Points",
            "Trailing Target (Points)",
            "Trailing Target + Signal Based",
            "Current Candle Low/High",
            "Previous Candle Low/High",
            "Current Swing Low/High",
            "Previous Swing Low/High",
            "ATR-based",
            "Risk-Reward Based",
            "Signal-based"
        ])
        
        target_points = 20
        target_atr_multiplier = 2.0
        risk_reward_ratio = 2.0
        
        if "Points" in target_type:
            target_points = st.number_input("Target Points", min_value=1, value=50)
        
        if target_type == "ATR-based":
            target_atr_multiplier = st.number_input("Target ATR Multiplier", 
                                                   min_value=0.1, value=2.0, step=0.1)
        
        if target_type == "Risk-Reward Based":
            risk_reward_ratio = st.number_input("Risk-Reward Ratio", 
                                               min_value=0.1, value=2.0, step=0.1)
        
        st.divider()
        
        # Trading Controls
        st.subheader("Trading Controls")
        
        st.info("💡 Trading controls have been moved to the top of the Live Dashboard tab for easier access.")
    
    # Main Content Area - Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Live Dashboard", "📜 Trade History", "📝 Trade Logs"])
    
    # ==================== TAB 1: LIVE DASHBOARD ====================
    with tab1:
        # Trading Controls at Top
        st.subheader("🎮 Trading Controls")
        control_col1, control_col2, control_col3 = st.columns([1, 1, 2])
        
        with control_col1:
            if st.button("▶️ Start Trading", use_container_width=True, type="primary"):
                st.session_state.trading_active = True
                st.session_state.position = None
                st.session_state.trailing_sl_high = None
                st.session_state.trailing_sl_low = None
                st.session_state.trailing_target_high = None
                st.session_state.trailing_target_low = None
                st.session_state.trailing_profit_points = 0
                st.session_state.threshold_crossed = False
                add_trade_log("Trading started")
                st.rerun()
        
        with control_col2:
            if st.button("⏹️ Stop Trading", use_container_width=True, type="secondary"):
                st.session_state.trading_active = False
                if st.session_state.position:
                    # Close position
                    pos = st.session_state.position
                    if st.session_state.current_data is not None:
                        exit_price = st.session_state.current_data['Close'].iloc[-1]
                        pnl = (exit_price - pos['entry_price']) * pos['signal']
                        
                        duration = get_ist_time() - pos['entry_datetime']
                        duration_str = str(duration).split('.')[0]
                        
                        trade_data = {
                            'entry_time': pos['entry_time'],
                            'exit_time': get_ist_time().strftime("%Y-%m-%d %H:%M:%S"),
                            'duration': duration_str,
                            'signal': pos['signal'],
                            'entry_price': pos['entry_price'],
                            'exit_price': exit_price,
                            'sl': pos['sl'],
                            'target': pos['target'],
                            'pnl': pnl,
                            'exit_reason': "Manual Stop"
                        }
                        add_trade_to_history(trade_data)
                        add_trade_log(f"Position closed manually. PnL: {pnl:.2f}")
                    st.session_state.position = None
                
                add_trade_log("Trading stopped")
                st.rerun()
        
        with control_col3:
            if st.session_state.trading_active:
                st.success("🟢 Trading is ACTIVE")
            else:
                st.info("⚪ Trading is STOPPED")
        
        st.divider()
        
        if not st.session_state.trading_active:
            st.info("👉 Click 'Start Trading' to begin live trading simulation")
            st.stop()
        
        # Live Trading Loop
        placeholder = st.empty()
        
        while st.session_state.trading_active:
            try:
                # Fetch latest data
                with st.spinner("Fetching latest data..."):
                    df = fetch_data(ticker, interval, period)
                
                if df is None or df.empty:
                    st.error("Failed to fetch data. Retrying...")
                    time.sleep(2)
                    continue
                
                st.session_state.current_data = df
                st.session_state.last_fetch_time = get_ist_time()
                
                # Get minimum target distance for this asset
                min_target_distance = get_min_target_for_asset(ticker)
                
                # Calculate indicators
                if strategy == "EMA Crossover":
                    df['EMA_Fast'] = calculate_ema(df['Close'], ema_fast)
                    df['EMA_Slow'] = calculate_ema(df['Close'], ema_slow)
                
                df['ATR'] = calculate_atr(df, 14)
                
                if strategy == "EMA Crossover" and use_adx:
                    df['ADX'] = calculate_adx(df, adx_period)
                
                # Calculate swing points
                df['Swing_High'], df['Swing_Low'] = calculate_swing_points(df, 5)
                
                current_idx = len(df) - 1
                current_price = df['Close'].iloc[current_idx]
                
                # Display live metrics
                with placeholder.container():
                    # Status row
                    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
                    
                    with status_col1:
                        st.metric("Current Price", f"{current_price:.2f}")
                    
                    with status_col2:
                        if st.session_state.position:
                            pos_type = "LONG 📈" if st.session_state.position['signal'] == 1 else "SHORT 📉"
                            st.metric("Position", pos_type)
                        else:
                            st.metric("Position", "NONE")
                    
                    with status_col3:
                        if st.session_state.position:
                            pnl = (current_price - st.session_state.position['entry_price']) * st.session_state.position['signal']
                            pnl_color = "🟢" if pnl > 0 else "🔴"
                            st.metric("Unrealized P&L", f"{pnl_color} {pnl:.2f}")
                        else:
                            st.metric("Unrealized P&L", "0.00")
                    
                    with status_col4:
                        last_update = st.session_state.last_fetch_time.strftime("%H:%M:%S") if st.session_state.last_fetch_time else "N/A"
                        st.metric("Last Update", last_update)
                    
                    st.divider()
                    
                    # Strategy-specific metrics
                    if strategy == "EMA Crossover":
                        ema_col1, ema_col2, ema_col3 = st.columns(3)
                        
                        with ema_col1:
                            ema_fast_val = df['EMA_Fast'].iloc[current_idx]
                            st.metric("EMA Fast", f"{ema_fast_val:.2f}")
                        
                        with ema_col2:
                            ema_slow_val = df['EMA_Slow'].iloc[current_idx]
                            st.metric("EMA Slow", f"{ema_slow_val:.2f}")
                        
                        with ema_col3:
                            angle = calculate_ema_angle(df['EMA_Fast'], df['EMA_Slow'], current_idx)
                            angle_status = "✅" if angle >= min_angle else "❌"
                            st.metric("Crossover Angle", f"{angle_status} {angle:.2f}°")
                        
                        # Entry Filter Status
                        filter_valid, candle_size, min_candle_size = check_entry_filter(
                            df, current_idx, entry_filter, custom_points, atr_multiplier
                        )
                        
                        filter_status = "✅" if filter_valid else "❌"
                        
                        if entry_filter == "Custom Candle (Points)":
                            st.info(f"{filter_status} Entry Filter: {entry_filter} | Candle Size: {candle_size:.2f} / Min: {min_candle_size:.2f}")
                        elif entry_filter == "ATR-based Candle":
                            st.info(f"{filter_status} Entry Filter: {entry_filter} | Candle Size: {candle_size:.2f} / Min (ATR×M): {min_candle_size:.2f}")
                        else:
                            st.info(f"{filter_status} Entry Filter: {entry_filter}")
                        
                        # ADX Status
                        if use_adx:
                            adx_val = df['ADX'].iloc[current_idx]
                            adx_status = "✅" if adx_val >= adx_threshold else "❌"
                            st.info(f"{adx_status} ADX Filter: {adx_val:.2f} / Threshold: {adx_threshold}")
                        
                        # Current Signal
                        signal, angle, filter_ok, _, _, adx_ok, _, _ = generate_ema_signal(
                            df, current_idx, ema_fast, ema_slow, min_angle,
                            entry_filter, custom_points, atr_multiplier, use_adx, adx_threshold
                        )
                        
                        if signal == 1:
                            st.success("🔥 Current Signal: BUY")
                        elif signal == -1:
                            st.warning("🔥 Current Signal: SELL")
                        else:
                            st.info("📊 Current Signal: NONE")
                    
                    elif strategy == "Price Crosses Threshold":
                        st.info(f"Monitoring: {threshold_direction} | Threshold: {threshold_price:.2f} | Current: {current_price:.2f}")
                    
                    st.divider()
                    
                    # Position Management
                    if not st.session_state.position:
                        # Check for entry
                        enter_trade = False
                        signal = 0
                        
                        if strategy == "Simple Buy":
                            enter_trade = True
                            signal = 1
                            add_trade_log("Simple Buy - Entering LONG")
                        
                        elif strategy == "Simple Sell":
                            enter_trade = True
                            signal = -1
                            add_trade_log("Simple Sell - Entering SHORT")
                        
                        elif strategy == "Price Crosses Threshold":
                            # Check if threshold is crossed
                            if not st.session_state.threshold_crossed:
                                if "Price >= Threshold" in threshold_direction:
                                    if current_price >= threshold_price:
                                        st.session_state.threshold_crossed = True
                                        if "LONG" in threshold_direction:
                                            enter_trade = True
                                            signal = 1
                                            add_trade_log(f"Price crossed above {threshold_price} - Entering LONG")
                                        else:
                                            enter_trade = True
                                            signal = -1
                                            add_trade_log(f"Price crossed above {threshold_price} - Entering SHORT")
                                elif "Price <= Threshold" in threshold_direction:
                                    if current_price <= threshold_price:
                                        st.session_state.threshold_crossed = True
                                        if "LONG" in threshold_direction:
                                            enter_trade = True
                                            signal = 1
                                            add_trade_log(f"Price crossed below {threshold_price} - Entering LONG")
                                        else:
                                            enter_trade = True
                                            signal = -1
                                            add_trade_log(f"Price crossed below {threshold_price} - Entering SHORT")
                        
                        elif strategy == "EMA Crossover":
                            signal, angle, filter_ok, candle_sz, min_candle, adx_ok, adx_val, _ = generate_ema_signal(
                                df, current_idx, ema_fast, ema_slow, min_angle,
                                entry_filter, custom_points, atr_multiplier, use_adx, adx_threshold
                            )
                            
                            if signal != 0:
                                enter_trade = True
                                signal_type = "BUY" if signal == 1 else "SELL"
                                add_trade_log(f"EMA Crossover - {signal_type} signal")
                        
                        if enter_trade and signal != 0:
                            # Calculate SL and Target
                            entry_price = current_price
                            
                            sl = calculate_stop_loss(
                                df, current_idx, entry_price, signal, sl_type,
                                sl_points, trailing_threshold, sl_atr_multiplier
                            )
                            
                            target = calculate_target(
                                df, current_idx, entry_price, signal, target_type,
                                target_points, target_atr_multiplier, risk_reward_ratio, None, min_target_distance
                            )
                            
                            # Apply minimum distances
                            if sl != 0:
                                if signal == 1:
                                    sl = min(sl, entry_price - 10)
                                else:
                                    sl = max(sl, entry_price + 10)
                            
                            if target != 0 and target_type not in ["Trailing Target (Points)", "Trailing Target + Signal Based"]:
                                if signal == 1:
                                    target = max(target, entry_price + min_target_distance)
                                else:
                                    target = min(target, entry_price - min_target_distance)
                            
                            # Initialize trailing variables
                            st.session_state.trailing_sl_high = entry_price
                            st.session_state.trailing_sl_low = entry_price
                            st.session_state.trailing_target_high = entry_price
                            st.session_state.trailing_target_low = entry_price
                            st.session_state.trailing_profit_points = 0
                            
                            # Create position
                            st.session_state.position = {
                                'signal': signal,
                                'entry_price': entry_price,
                                'entry_time': get_ist_time().strftime("%Y-%m-%d %H:%M:%S"),
                                'entry_datetime': get_ist_time(),
                                'sl': sl,
                                'target': target,
                                'entry_idx': current_idx
                            }
                            
                            pos_type = "LONG" if signal == 1 else "SHORT"
                            sl_str = f"{sl:.2f}" if sl != 0 else "Signal Based"
                            target_str = f"{target:.2f}" if target != 0 else "Signal Based"
                            
                            add_trade_log(f"Position opened: {pos_type} @ {entry_price:.2f} | SL: {sl_str} | Target: {target_str}")
                            
                            st.success(f"✅ {pos_type} Position Opened @ {entry_price:.2f}")
                    
                    else:
                        # Position is open - check for exit
                        pos = st.session_state.position
                        signal = pos['signal']
                        entry_price = pos['entry_price']
                        current_sl = pos['sl']
                        current_target = pos['target']
                        entry_datetime = pos['entry_datetime']
                        
                        # Calculate trade duration
                        duration = get_ist_time() - entry_datetime
                        duration_str = str(duration).split('.')[0]  # Remove microseconds
                        
                        # Update SL
                        new_sl = calculate_stop_loss(
                            df, current_idx, entry_price, signal, sl_type,
                            sl_points, trailing_threshold, sl_atr_multiplier, current_sl
                        )
                        
                        # Update Target
                        new_target = calculate_target(
                            df, current_idx, entry_price, signal, target_type,
                            target_points, target_atr_multiplier, risk_reward_ratio, current_target, min_target_distance
                        )
                        
                        st.session_state.position['sl'] = new_sl
                        st.session_state.position['target'] = new_target
                        
                        # Display position info with entry time and duration
                        st.info(f"📍 **Entry Time:** {pos['entry_time']} | **Duration:** {duration_str}")
                        
                        pos_col1, pos_col2, pos_col3 = st.columns(3)
                        
                        with pos_col1:
                            st.metric("Entry Price", f"{entry_price:.2f}")
                        
                        with pos_col2:
                            sl_display = f"{new_sl:.2f}" if new_sl != 0 else "Signal Based"
                            if new_sl != 0:
                                dist_to_sl = abs(current_price - new_sl)
                                st.metric("Stop Loss", sl_display, f"Dist: {dist_to_sl:.2f}")
                            else:
                                st.metric("Stop Loss", sl_display)
                        
                        with pos_col3:
                            target_display = f"{new_target:.2f}" if new_target != 0 else "Signal Based"
                            if new_target != 0:
                                dist_to_target = abs(current_price - new_target)
                                st.metric("Target", target_display, f"Dist: {dist_to_target:.2f}")
                            else:
                                st.metric("Target", target_display)
                        
                        # Show trailing info for trailing targets
                        if target_type in ["Trailing Target (Points)", "Trailing Target + Signal Based"]:
                            profit_moved = st.session_state.get('trailing_profit_points', 0)
                            st.caption(f"Trailing: Profit moved {profit_moved:.2f} points | Next update at {profit_moved + target_points:.2f} points")
                        
                        # Check exit conditions
                        exit_trade = False
                        exit_reason = ""
                        exit_price = current_price
                        
                        # Check signal-based exit
                        if sl_type == "Signal-based" or target_type == "Signal-based" or \
                           sl_type == "Trailing SL + Signal Based" or target_type == "Trailing Target + Signal Based":
                            if strategy == "EMA Crossover":
                                reverse_signal = check_reverse_signal(df, current_idx, signal)
                                if reverse_signal:
                                    exit_trade = True
                                    exit_reason = "Reverse Signal - EMA Crossover"
                        
                        # Check SL hit
                        if not exit_trade and new_sl != 0:
                            if signal == 1 and current_price <= new_sl:
                                exit_trade = True
                                exit_reason = "Stop Loss Hit"
                                exit_price = new_sl
                            elif signal == -1 and current_price >= new_sl:
                                exit_trade = True
                                exit_reason = "Stop Loss Hit"
                                exit_price = new_sl
                        
                        # Check Target hit - for non-trailing targets only
                        if not exit_trade and new_target != 0:
                            if target_type not in ["Trailing Target (Points)", "Trailing Target + Signal Based"]:
                                # For fixed targets, check if hit
                                if signal == 1 and current_price >= new_target:
                                    exit_trade = True
                                    exit_reason = "Target Hit"
                                    exit_price = new_target
                                elif signal == -1 and current_price <= new_target:
                                    exit_trade = True
                                    exit_reason = "Target Hit"
                                    exit_price = new_target
                        
                        if exit_trade:
                            pnl = (exit_price - entry_price) * signal
                            
                            trade_data = {
                                'entry_time': pos['entry_time'],
                                'exit_time': get_ist_time().strftime("%Y-%m-%d %H:%M:%S"),
                                'duration': duration_str,
                                'signal': signal,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'sl': new_sl,
                                'target': new_target,
                                'pnl': pnl,
                                'exit_reason': exit_reason
                            }
                            
                            add_trade_to_history(trade_data)
                            add_trade_log(f"Position closed: {exit_reason} | PnL: {pnl:.2f}")
                            
                            # Reset position and threshold
                            st.session_state.position = None
                            st.session_state.trailing_sl_high = None
                            st.session_state.trailing_sl_low = None
                            st.session_state.trailing_target_high = None
                            st.session_state.trailing_target_low = None
                            st.session_state.trailing_profit_points = 0
                            st.session_state.threshold_crossed = False
                            
                            pnl_color = "🟢" if pnl > 0 else "🔴"
                            st.success(f"{pnl_color} Position Closed: {exit_reason} | PnL: {pnl:.2f}")
                        else:
                            # Show guidance
                            pnl = (current_price - entry_price) * signal
                            if pnl > 0:
                                st.info("💰 In Profit - Hold position or trail stops")
                            else:
                                st.warning("⚠️ In Loss - Monitor stop loss")
                    
                    st.divider()
                    
                    # Chart
                    chart_key = f"live_chart_{get_ist_time().timestamp()}"
                    fig = create_live_chart(df, st.session_state.position)
                    st.plotly_chart(fig, use_container_width=True, key=chart_key)
                    
                    # DHAN Integration Placeholder
                    with st.expander("🔌 Broker Integration (Future)"):
                        st.code("""
# DHAN ORDER EXECUTION (FUTURE INTEGRATION)
# if signal == 1:
#     dhan.place_order(order_type='BUY', quantity=lot_size, price=entry_price)
# else:
#     dhan.place_order(order_type='SELL', quantity=lot_size, price=entry_price)
                        """)
                
                # Auto-refresh delay
                time.sleep(random.uniform(1.0, 1.5))
                
                # Check if trading is still active
                if not st.session_state.trading_active:
                    break
            
            except Exception as e:
                st.error(f"Error in trading loop: {e}")
                add_trade_log(f"Error: {e}")
                time.sleep(2)
    
    # ==================== TAB 2: TRADE HISTORY ====================
    with tab2:
        st.header("📜 Trade History")
        
        # Always check current state
        trade_history = st.session_state.get('trade_history', [])
        
        if not trade_history or len(trade_history) == 0:
            st.info("📋 No completed trades yet. Trade history will appear here once trades are closed.")
            st.write("")
            st.write("**Note:** Trades will appear here after they are closed (either by hitting SL/Target or manual stop).")
        else:
            # Calculate statistics
            total_trades = len(trade_history)
            winning_trades = sum(1 for t in trade_history if t.get('pnl', 0) > 0)
            losing_trades = total_trades - winning_trades
            accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = sum(t.get('pnl', 0) for t in trade_history)
            
            # Display metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Total Trades", total_trades)
            with metric_col2:
                st.metric("Winning Trades", f"{winning_trades} ({accuracy:.1f}%)")
            with metric_col3:
                st.metric("Losing Trades", losing_trades)
            with metric_col4:
                pnl_color = "🟢" if total_pnl > 0 else "🔴"
                st.metric("Total P&L", f"{pnl_color} {total_pnl:.2f}")
            
            st.divider()
            
            # Display trades
            for idx, trade in enumerate(reversed(trade_history), 1):
                pnl_emoji = "🟢" if trade.get('pnl', 0) > 0 else "🔴"
                with st.expander(f"Trade #{total_trades - idx + 1} - {pnl_emoji} {trade.get('exit_reason', 'N/A')} - P&L: {trade.get('pnl', 0):.2f}"):
                    trade_type = "LONG 📈" if trade.get('signal', 0) == 1 else "SHORT 📉"
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Type:** {trade_type}")
                        st.write(f"**Entry Time:** {trade.get('entry_time', 'N/A')}")
                        st.write(f"**Entry Price:** {trade.get('entry_price', 0):.2f}")
                        sl_display = f"{trade.get('sl', 0):.2f}" if trade.get('sl', 0) != 0 else "Signal Based"
                        st.write(f"**Stop Loss:** {sl_display}")
                    
                    with col2:
                        pnl_display_emoji = "🟢" if trade.get('pnl', 0) > 0 else "🔴"
                        st.write(f"**P&L:** {pnl_display_emoji} {trade.get('pnl', 0):.2f}")
                        st.write(f"**Exit Time:** {trade.get('exit_time', 'N/A')}")
                        st.write(f"**Duration:** {trade.get('duration', 'N/A')}")
                        st.write(f"**Exit Price:** {trade.get('exit_price', 0):.2f}")
                        target_display = f"{trade.get('target', 0):.2f}" if trade.get('target', 0) != 0 else "Signal Based"
                        st.write(f"**Target:** {target_display}")
                    
                    st.write(f"**Exit Reason:** {trade.get('exit_reason', 'N/A')}")
    
    # ==================== TAB 3: TRADE LOGS ====================
    with tab3:
        st.header("📝 Trade Logs")
        
        # Always check current state
        trade_logs = st.session_state.get('trade_logs', [])
        
        if not trade_logs or len(trade_logs) == 0:
            st.info("📝 No logs yet. Important trading events will be logged here.")
            st.write("")
            st.write("**Logs include:** Trading start/stop, position entries/exits, and important events.")
        else:
            st.caption(f"Showing last {len(trade_logs)} logs (max 50 kept in memory)")
            st.divider()
            # Display logs in reverse order (newest first)
            for log in reversed(trade_logs):
                st.text(log)

if __name__ == "__main__":
    main()
