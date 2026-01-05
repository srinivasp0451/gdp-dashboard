import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import time
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(page_title="Live Trading System", layout="wide", initial_sidebar_state="expanded")

# IST timezone
IST = pytz.timezone('Asia/Kolkata')

def get_ist_time():
    """Get current time in IST"""
    return datetime.now(IST)

# Asset mapping
ASSET_MAPPING = {
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "USDINR": "USDINR=X",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Custom": ""
}

# Timeframe validation
TIMEFRAME_RULES = {
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

# Initialize session state
def init_session_state():
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
    if 'last_fetch_time' not in st.session_state:
        st.session_state.last_fetch_time = None
    if 'trailing_sl_high' not in st.session_state:
        st.session_state.trailing_sl_high = None
    if 'trailing_sl_low' not in st.session_state:
        st.session_state.trailing_sl_low = None

def reset_trading_state():
    """Reset trading state after exit"""
    st.session_state.position = None
    st.session_state.trailing_sl_high = None
    st.session_state.trailing_sl_low = None

def add_log(message):
    """Add timestamped log entry"""
    timestamp = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.trade_logs.append(f"[{timestamp}] {message}")

def fetch_data_yfinance(ticker, interval, period):
    """Fetch data from yfinance with rate limiting"""
    try:
        time.sleep(random.uniform(1.0, 1.5))
        
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if data.empty:
            return None
        
        # Flatten multi-index
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Select OHLCV
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[required_cols].copy()
        
        # Handle timezone
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert(IST)
        else:
            data.index = data.index.tz_convert(IST)
        
        data = data.dropna()
        return data
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_ema(data, period):
    """Calculate EMA manually"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_atr(data, period=14):
    """Calculate ATR"""
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_adx(data, period=14):
    """Calculate ADX"""
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    atr = calculate_atr(data, period)
    
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return adx

def find_swing_points(data, lookback=5):
    """Find swing highs and lows"""
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(data) - lookback):
        high_slice = data['High'].iloc[i-lookback:i+lookback+1]
        low_slice = data['Low'].iloc[i-lookback:i+lookback+1]
        
        if data['High'].iloc[i] == high_slice.max():
            swing_highs.append(i)
        
        if data['Low'].iloc[i] == low_slice.min():
            swing_lows.append(i)
    
    return swing_highs, swing_lows

def calculate_ema_angle(data, ema_fast, ema_slow):
    """Calculate crossover angle in degrees"""
    if len(data) < 2:
        return 0
    
    fast_diff = ema_fast.iloc[-1] - ema_fast.iloc[-2]
    slow_diff = ema_slow.iloc[-1] - ema_slow.iloc[-2]
    
    angle_diff = fast_diff - slow_diff
    angle_rad = np.arctan(angle_diff)
    angle_deg = np.degrees(angle_rad)
    
    return abs(angle_deg)

def check_ema_crossover(data, fast_period, slow_period, min_angle=1.0, entry_filter="Simple Crossover", config=None):
    """Check for EMA crossover with angle validation and entry filters"""
    if len(data) < max(fast_period, slow_period) + 1:
        return 0, 0
    
    ema_fast = calculate_ema(data['Close'], fast_period)
    ema_slow = calculate_ema(data['Close'], slow_period)
    
    # Current and previous values
    fast_curr = ema_fast.iloc[-1]
    fast_prev = ema_fast.iloc[-2]
    slow_curr = ema_slow.iloc[-1]
    slow_prev = ema_slow.iloc[-2]
    
    # Calculate angle
    angle = calculate_ema_angle(data, ema_fast, ema_slow)
    
    # Check for crossover
    bullish_cross = fast_curr > slow_curr and fast_prev <= slow_prev
    bearish_cross = fast_curr < slow_curr and fast_prev >= slow_prev
    
    # If no crossover, return early
    if not bullish_cross and not bearish_cross:
        return 0, angle
    
    # Check angle requirement
    if angle < min_angle:
        return 0, angle
    
    # Apply entry filter
    if entry_filter == "Simple Crossover":
        # No additional filter - just angle check
        if bullish_cross:
            return 1, angle
        if bearish_cross:
            return -1, angle
    
    elif entry_filter == "Custom Candle (Points)":
        # Check candle size in points
        candle_size = abs(data['Close'].iloc[-1] - data['Open'].iloc[-1])
        min_candle_size = config.get('candle_points', 10)
        
        if candle_size >= min_candle_size:
            if bullish_cross:
                return 1, angle
            if bearish_cross:
                return -1, angle
        return 0, angle
    
    elif entry_filter == "ATR-based Candle":
        # Check candle size relative to ATR
        candle_size = abs(data['Close'].iloc[-1] - data['Open'].iloc[-1])
        atr = calculate_atr(data).iloc[-1]
        atr_multiplier = config.get('atr_multiplier', 1.0)
        min_candle_size = atr * atr_multiplier
        
        if candle_size >= min_candle_size:
            if bullish_cross:
                return 1, angle
            if bearish_cross:
                return -1, angle
        return 0, angle
    
    return 0, angle

def calculate_sl_target(data, signal, sl_type, target_type, config):
    """Calculate stop loss and target based on types"""
    entry_price = data['Close'].iloc[-1]
    atr = calculate_atr(data).iloc[-1]
    
    sl_price = 0
    target_price = 0
    
    # Calculate Stop Loss
    if sl_type == "Custom Points":
        offset = config['sl_points']
        sl_price = entry_price - offset if signal == 1 else entry_price + offset
    
    elif sl_type == "Trailing SL (Points)":
        offset = config['sl_points']
        sl_price = entry_price - offset if signal == 1 else entry_price + offset
    
    elif sl_type == "ATR-based":
        multiplier = config.get('sl_atr_multiplier', 2.0)
        offset = atr * multiplier
        sl_price = entry_price - offset if signal == 1 else entry_price + offset
    
    elif sl_type == "Current Candle Low/High":
        sl_price = data['Low'].iloc[-1] if signal == 1 else data['High'].iloc[-1]
    
    elif sl_type == "Previous Candle Low/High":
        sl_price = data['Low'].iloc[-2] if signal == 1 else data['High'].iloc[-2]
    
    elif sl_type == "Current Swing Low/High":
        swing_highs, swing_lows = find_swing_points(data)
        if signal == 1 and swing_lows:
            sl_price = data['Low'].iloc[swing_lows[-1]]
        elif signal == -1 and swing_highs:
            sl_price = data['High'].iloc[swing_highs[-1]]
        else:
            sl_price = entry_price - 10 if signal == 1 else entry_price + 10
    
    elif sl_type == "Previous Swing Low/High":
        swing_highs, swing_lows = find_swing_points(data)
        if signal == 1 and len(swing_lows) > 1:
            sl_price = data['Low'].iloc[swing_lows[-2]]
        elif signal == -1 and len(swing_highs) > 1:
            sl_price = data['High'].iloc[swing_highs[-2]]
        else:
            sl_price = entry_price - 10 if signal == 1 else entry_price + 10
    
    elif sl_type == "Signal-based":
        sl_price = 0
    
    elif "Trailing SL + Current Candle" in sl_type:
        sl_price = data['Low'].iloc[-1] if signal == 1 else data['High'].iloc[-1]
    
    elif "Trailing SL + Previous Candle" in sl_type:
        sl_price = data['Low'].iloc[-2] if signal == 1 else data['High'].iloc[-2]
    
    elif "Trailing SL + Current Swing" in sl_type:
        swing_highs, swing_lows = find_swing_points(data)
        if signal == 1 and swing_lows:
            sl_price = data['Low'].iloc[swing_lows[-1]]
        elif signal == -1 and swing_highs:
            sl_price = data['High'].iloc[swing_highs[-1]]
        else:
            sl_price = entry_price - 10 if signal == 1 else entry_price + 10
    
    elif "Trailing SL + Previous Swing" in sl_type:
        swing_highs, swing_lows = find_swing_points(data)
        if signal == 1 and len(swing_lows) > 1:
            sl_price = data['Low'].iloc[swing_lows[-2]]
        elif signal == -1 and len(swing_highs) > 1:
            sl_price = data['High'].iloc[swing_highs[-2]]
        else:
            sl_price = entry_price - 10 if signal == 1 else entry_price + 10
    
    elif "Trailing SL + Signal Based" in sl_type:
        offset = config['sl_points']
        sl_price = entry_price - offset if signal == 1 else entry_price + offset
    
    # Ensure minimum SL distance
    if sl_price != 0:
        min_sl_distance = 10
        if signal == 1:
            sl_price = min(sl_price, entry_price - min_sl_distance)
        else:
            sl_price = max(sl_price, entry_price + min_sl_distance)
    
    # Calculate Target
    if target_type == "Custom Points":
        offset = config['target_points']
        target_price = entry_price + offset if signal == 1 else entry_price - offset
    
    elif target_type == "ATR-based":
        multiplier = config.get('target_atr_multiplier', 3.0)
        offset = atr * multiplier
        target_price = entry_price + offset if signal == 1 else entry_price - offset
    
    elif target_type == "Risk-Reward Based":
        rr_ratio = config.get('risk_reward_ratio', 2.0)
        sl_distance = abs(entry_price - sl_price)
        target_price = entry_price + (sl_distance * rr_ratio) if signal == 1 else entry_price - (sl_distance * rr_ratio)
    
    elif target_type == "Current Candle Low/High":
        target_price = data['High'].iloc[-1] if signal == 1 else data['Low'].iloc[-1]
    
    elif target_type == "Previous Candle Low/High":
        target_price = data['High'].iloc[-2] if signal == 1 else data['Low'].iloc[-2]
    
    elif target_type == "Current Swing Low/High":
        swing_highs, swing_lows = find_swing_points(data)
        if signal == 1 and swing_highs:
            target_price = data['High'].iloc[swing_highs[-1]]
        elif signal == -1 and swing_lows:
            target_price = data['Low'].iloc[swing_lows[-1]]
        else:
            target_price = entry_price + 15 if signal == 1 else entry_price - 15
    
    elif target_type == "Previous Swing Low/High":
        swing_highs, swing_lows = find_swing_points(data)
        if signal == 1 and len(swing_highs) > 1:
            target_price = data['High'].iloc[swing_highs[-2]]
        elif signal == -1 and len(swing_lows) > 1:
            target_price = data['Low'].iloc[swing_lows[-2]]
        else:
            target_price = entry_price + 15 if signal == 1 else entry_price - 15
    
    elif target_type == "Signal-based":
        target_price = 0
    
    elif target_type == "Trailing Target (Points)":
        offset = config['target_points']
        # target_price = entry_price + offset if signal == 1 else entry_price - offset
        # target_price = data['Close'].iloc[-1] - offset if signal == 1 else data['Close'].iloc[-1] + offset
        target_price = (data['Close'].cummax().iloc[-1] + offset if signal == 1 else data['Close'].cummin().iloc[-1] - offset)

    
    elif "Trailing Target + Signal Based" in target_type:
        offset = config['target_points']
        target_price = entry_price + offset if signal == 1 else entry_price - offset
    
    # Ensure minimum target distance
    if target_price != 0:
        min_target_distance = 15
        if signal == 1:
            target_price = max(target_price, entry_price + min_target_distance)
        else:
            target_price = min(target_price, entry_price - min_target_distance)
    
    return entry_price, sl_price, target_price

def update_trailing_sl(data, position, config):
    """Update trailing stop loss"""
    current_price = data['Close'].iloc[-1]
    entry_price = position['entry_price']
    signal = position['signal']
    sl_type = position['sl_type']
    trailing_threshold = config.get('trailing_threshold', 0)
    
    # Initialize trailing values if not set
    if st.session_state.trailing_sl_high is None:
        st.session_state.trailing_sl_high = current_price
    if st.session_state.trailing_sl_low is None:
        st.session_state.trailing_sl_low = current_price
    
    # Update trailing high/low
    if current_price > st.session_state.trailing_sl_high:
        st.session_state.trailing_sl_high = current_price
    if current_price < st.session_state.trailing_sl_low:
        st.session_state.trailing_sl_low = current_price
    
    new_sl = position['sl_price']
    
    # Check if price moved enough to update SL
    if signal == 1:
        price_movement = st.session_state.trailing_sl_high - entry_price
    else:
        price_movement = entry_price - st.session_state.trailing_sl_low
    
    if price_movement < trailing_threshold:
        return new_sl
    
    # Update based on SL type
    if "Trailing SL (Points)" in sl_type:
        offset = config['sl_points']
        if signal == 1:
            new_sl = max(new_sl, st.session_state.trailing_sl_high - offset)
        else:
            new_sl = min(new_sl, st.session_state.trailing_sl_low + offset)
    
    elif "Trailing SL + Current Candle" in sl_type:
        if signal == 1:
            new_sl = max(new_sl, data['Low'].iloc[-1])
        else:
            new_sl = min(new_sl, data['High'].iloc[-1])
    
    elif "Trailing SL + Previous Candle" in sl_type:
        if signal == 1:
            new_sl = max(new_sl, data['Low'].iloc[-2])
        else:
            new_sl = min(new_sl, data['High'].iloc[-2])
    
    elif "Trailing SL + Current Swing" in sl_type:
        swing_highs, swing_lows = find_swing_points(data)
        if signal == 1 and swing_lows:
            new_sl = max(new_sl, data['Low'].iloc[swing_lows[-1]])
        elif signal == -1 and swing_highs:
            new_sl = min(new_sl, data['High'].iloc[swing_highs[-1]])
    
    elif "Trailing SL + Previous Swing" in sl_type:
        swing_highs, swing_lows = find_swing_points(data)
        if signal == 1 and len(swing_lows) > 1:
            new_sl = max(new_sl, data['Low'].iloc[swing_lows[-2]])
        elif signal == -1 and len(swing_highs) > 1:
            new_sl = min(new_sl, data['High'].iloc[swing_highs[-2]])
    
    return new_sl

def check_exit_conditions(data, position, config):
    """Check if position should be exited"""
    current_price = data['Close'].iloc[-1]
    entry_price = position['entry_price']
    sl_price = position['sl_price']
    target_price = position['target_price']
    signal = position['signal']
    sl_type = position['sl_type']
    target_type = position['target_type']
    
    # Update trailing SL if applicable
    if "Trailing" in sl_type:
        sl_price = update_trailing_sl(data, position, config)
        position['sl_price'] = sl_price
    
    # Check signal-based exits
    if "Signal-based" in sl_type or "Signal-based" in target_type or "Signal Based" in sl_type or "Signal Based" in target_type:
        if len(data) >= 2:
            ema_fast = calculate_ema(data['Close'], config['ema_fast'])
            ema_slow = calculate_ema(data['Close'], config['ema_slow'])
            
            fast_curr = ema_fast.iloc[-1]
            fast_prev = ema_fast.iloc[-2]
            slow_curr = ema_slow.iloc[-1]
            slow_prev = ema_slow.iloc[-2]
            
            # Check reverse crossover
            if signal == 1:
                if fast_curr < slow_curr and fast_prev >= slow_prev:
                    pnl = current_price - entry_price
                    return True, "Reverse Signal - Bearish Crossover", current_price, pnl
            
            elif signal == -1:
                if fast_curr > slow_curr and fast_prev <= slow_prev:
                    pnl = entry_price - current_price
                    return True, "Reverse Signal - Bullish Crossover", current_price, pnl
    
    # Check SL hit
    if sl_price != 0:
        if signal == 1 and current_price <= sl_price:
            pnl = current_price - entry_price
            return True, "Stop Loss Hit", current_price, pnl
        
        if signal == -1 and current_price >= sl_price:
            pnl = entry_price - current_price
            return True, "Stop Loss Hit", current_price, pnl
    
    # Check target hit
    if target_price != 0:
        if signal == 1 and current_price >= target_price:
            pnl = current_price - entry_price
            return True, "Target Hit", current_price, pnl
        
        if signal == -1 and current_price <= target_price:
            pnl = entry_price - current_price
            return True, "Target Hit", current_price, pnl
    
    return False, None, None, None

def plot_live_chart(data, position, config):
    """Create live candlestick chart with EMAs"""
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    # EMAs
    if config['strategy'] == "EMA Crossover":
        ema_fast = calculate_ema(data['Close'], config['ema_fast'])
        ema_slow = calculate_ema(data['Close'], config['ema_slow'])
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=ema_fast,
            mode='lines',
            name=f"EMA {config['ema_fast']}",
            line=dict(color='blue', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=ema_slow,
            mode='lines',
            name=f"EMA {config['ema_slow']}",
            line=dict(color='red', width=1)
        ))
    
    # Position lines
    if position:
        # Entry line
        fig.add_hline(
            y=position['entry_price'],
            line_dash="dash",
            line_color="yellow",
            annotation_text="Entry",
            annotation_position="right"
        )
        
        # SL line
        if position['sl_price'] != 0:
            fig.add_hline(
                y=position['sl_price'],
                line_dash="dash",
                line_color="red",
                annotation_text="SL",
                annotation_position="right"
            )
        
        # Target line
        if position['target_price'] != 0:
            fig.add_hline(
                y=position['target_price'],
                line_dash="dash",
                line_color="green",
                annotation_text="Target",
                annotation_position="right"
            )
    
    fig.update_layout(
        title="Live Trading Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    return fig

# Main App
def main():
    st.title("🚀 Professional Live Trading System")
    
    init_session_state()
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Asset Selection
    asset_name = st.sidebar.selectbox("Select Asset", list(ASSET_MAPPING.keys()))
    
    if asset_name == "Custom":
        ticker = st.sidebar.text_input("Enter Custom Ticker", "AAPL")
    else:
        ticker = ASSET_MAPPING[asset_name]
    
    # Timeframe
    interval = st.sidebar.selectbox("Interval", list(TIMEFRAME_RULES.keys()))
    allowed_periods = TIMEFRAME_RULES[interval]
    period = st.sidebar.selectbox("Period", allowed_periods)
    
    # Strategy Selection
    strategy = st.sidebar.selectbox(
        "Strategy",
        ["EMA Crossover", "Simple Buy", "Simple Sell"]
    )
    
    # Strategy Configuration
    config = {'strategy': strategy}
    
    if strategy == "EMA Crossover":
        config['ema_fast'] = st.sidebar.number_input("EMA Fast", min_value=2, value=9)
        config['ema_slow'] = st.sidebar.number_input("EMA Slow", min_value=2, value=15)
        config['min_angle'] = st.sidebar.number_input("Min Crossover Angle (°)", min_value=0.0, value=1.0, step=0.1)
        
        # Entry Filter Type
        config['entry_filter'] = st.sidebar.selectbox(
            "Entry Filter",
            ["Simple Crossover", "Custom Candle (Points)", "ATR-based Candle"]
        )
        
        if config['entry_filter'] == "Custom Candle (Points)":
            config['candle_points'] = st.sidebar.number_input("Min Candle Size (Points)", min_value=1, value=10)
        elif config['entry_filter'] == "ATR-based Candle":
            config['atr_multiplier'] = st.sidebar.number_input("ATR Multiplier for Candle", min_value=0.1, value=1.0, step=0.1)
        
        config['use_adx'] = st.sidebar.checkbox("Include ADX Filter", value=False)
        
        if config['use_adx']:
            config['adx_period'] = st.sidebar.number_input("ADX Period", min_value=5, value=14)
            config['adx_threshold'] = st.sidebar.number_input("ADX Threshold", min_value=0.0, value=25.0)
    
    # SL Configuration
    sl_type = st.sidebar.selectbox(
        "Stop Loss Type",
        [
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
    )
    
    config['sl_type'] = sl_type
    
    if "Custom Points" in sl_type or "Trailing" in sl_type:
        config['sl_points'] = st.sidebar.number_input("SL Points", min_value=1, value=10)
    
    if "Trailing" in sl_type:
        config['trailing_threshold'] = st.sidebar.number_input("Trailing Threshold (Points)", min_value=0, value=0)
    
    if "ATR" in sl_type:
        config['sl_atr_multiplier'] = st.sidebar.number_input("SL ATR Multiplier", min_value=0.1, value=2.0, step=0.1)
    
    # Target Configuration
    target_type = st.sidebar.selectbox(
        "Target Type",
        [
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
        ]
    )
    
    config['target_type'] = target_type
    
    if "Custom Points" in target_type or "Trailing" in target_type:
        config['target_points'] = st.sidebar.number_input("Target Points", min_value=1, value=20)
    
    if "ATR" in target_type:
        config['target_atr_multiplier'] = st.sidebar.number_input("Target ATR Multiplier", min_value=0.1, value=3.0, step=0.1)
    
    if "Risk-Reward" in target_type:
        config['risk_reward_ratio'] = st.sidebar.number_input("Risk-Reward Ratio", min_value=0.1, value=2.0, step=0.1)
    
    # Trading Controls
    st.sidebar.markdown("---")
    st.sidebar.header("🎮 Trading Controls")
    
    col1, col2 = st.sidebar.columns(2)
    
    if col1.button("▶️ Start Trading", type="primary", use_container_width=True):
        if not st.session_state.trading_active:
            with st.spinner("Fetching data..."):
                data = fetch_data_yfinance(ticker, interval, period)
                if data is not None and len(data) > 0:
                    st.session_state.current_data = data
                    st.session_state.trading_active = True
                    st.session_state.last_fetch_time = get_ist_time()
                    add_log(f"Trading started for {ticker} ({interval}/{period})")
                    st.success("Trading started!")
                    st.rerun()
                else:
                    st.error("Failed to fetch data. Please check your inputs.")
    
    if col2.button("⏹️ Stop Trading", type="secondary", use_container_width=True):
        if st.session_state.trading_active:
            st.session_state.trading_active = False
            if st.session_state.position:
                # Close open position
                current_price = st.session_state.current_data['Close'].iloc[-1]
                entry_price = st.session_state.position['entry_price']
                signal = st.session_state.position['signal']
                
                pnl_value = (current_price - entry_price) if signal == 1 else (entry_price - current_price)
                
                trade_record = {
                    'entry_time': st.session_state.position['entry_time'],
                    'exit_time': get_ist_time().strftime("%Y-%m-%d %H:%M:%S"),
                    'signal': 'BUY' if signal == 1 else 'SELL',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'sl_price': st.session_state.position['sl_price'],
                    'target_price': st.session_state.position['target_price'],
                    'exit_reason': 'Manual Stop',
                    'pnl': pnl_value,
                    'strategy': config['strategy']
                }
                st.session_state.trade_history.append(trade_record)
                add_log(f"Position closed manually. PnL: {pnl_value:.2f}")
            
            reset_trading_state()
            add_log("Trading stopped")
            st.success("Trading stopped!")
            st.rerun()
    
    # Main Content Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Live Trading Dashboard", "📈 Trade History", "📝 Trade Logs"])
    
    # Tab 1: Live Trading Dashboard
    with tab1:
        if not st.session_state.trading_active:
            st.info("👈 Click 'Start Trading' to begin")
            st.markdown("### Configuration Summary")
            st.write(f"**Asset:** {ticker}")
            st.write(f"**Timeframe:** {interval} / {period}")
            st.write(f"**Strategy:** {strategy}")
            st.write(f"**Stop Loss:** {sl_type}")
            st.write(f"**Target:** {target_type}")
        else:
            # Auto-refresh mechanism
            placeholder = st.empty()
            
            while st.session_state.trading_active:
                # Fetch latest data
                data = fetch_data_yfinance(ticker, interval, period)
                
                if data is None or len(data) == 0:
                    st.error("Failed to fetch data")
                    time.sleep(random.uniform(1.0, 1.5))
                    continue
                
                st.session_state.current_data = data
                current_price = data['Close'].iloc[-1]
                
                with placeholder.container():
                    # Strategy Metrics
                    st.markdown("### 🎯 Live Strategy Metrics")
                    
                    if strategy == "EMA Crossover":
                        ema_fast = calculate_ema(data['Close'], config['ema_fast'])
                        ema_slow = calculate_ema(data['Close'], config['ema_slow'])
                        
                        fast_val = ema_fast.iloc[-1]
                        slow_val = ema_slow.iloc[-1]
                        
                        signal, angle = check_ema_crossover(
                            data, 
                            config['ema_fast'], 
                            config['ema_slow'], 
                            config['min_angle'],
                            config.get('entry_filter', 'Simple Crossover'),
                            config
                        )
                        
                        # Check ADX if enabled
                        adx_pass = True
                        adx_value = 0
                        if config.get('use_adx', False):
                            adx = calculate_adx(data, config['adx_period'])
                            adx_value = adx.iloc[-1]
                            if adx_value < config['adx_threshold']:
                                adx_pass = False
                                signal = 0
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Current Price", f"{current_price:.2f}")
                        col2.metric(f"EMA {config['ema_fast']}", f"{fast_val:.2f}")
                        col3.metric(f"EMA {config['ema_slow']}", f"{slow_val:.2f}")
                        
                        col4, col5, col6 = st.columns(3)
                        col4.metric("Crossover Angle", f"{angle:.2f}°")
                        
                        signal_text = "BUY" if signal == 1 else "SELL" if signal == -1 else "NONE"
                        signal_color = "🟢" if signal == 1 else "🔴" if signal == -1 else "⚪"
                        col5.metric("Current Signal", f"{signal_color} {signal_text}")
                        
                        if st.session_state.position:
                            pos_status = "🟢 OPEN"
                        else:
                            pos_status = "⚪ NO POSITION"
                        col6.metric("Position Status", pos_status)
                        
                        # Display entry filter info
                        entry_filter = config.get('entry_filter', 'Simple Crossover')
                        filter_info = f"📋 Entry Filter: {entry_filter}"
                        
                        if entry_filter == "Custom Candle (Points)":
                            candle_size = abs(data['Close'].iloc[-1] - data['Open'].iloc[-1])
                            min_size = config.get('candle_points', 10)
                            filter_pass = "✅" if candle_size >= min_size else "❌"
                            filter_info += f" | Candle Size: {candle_size:.2f} / Min: {min_size} {filter_pass}"
                        elif entry_filter == "ATR-based Candle":
                            candle_size = abs(data['Close'].iloc[-1] - data['Open'].iloc[-1])
                            atr = calculate_atr(data).iloc[-1]
                            multiplier = config.get('atr_multiplier', 1.0)
                            min_size = atr * multiplier
                            filter_pass = "✅" if candle_size >= min_size else "❌"
                            filter_info += f" | Candle Size: {candle_size:.2f} / Min (ATR×{multiplier}): {min_size:.2f} {filter_pass}"
                        
                        st.info(filter_info)
                        
                        if config.get('use_adx', False):
                            adx_status = '✅ PASS' if adx_pass else '❌ FAIL'
                            st.info(f"ADX Value: {adx_value:.2f} | Threshold: {config['adx_threshold']:.2f} | Filter: {adx_status}")
                    
                    elif strategy in ["Simple Buy", "Simple Sell"]:
                        st.metric("Current Price", f"{current_price:.2f}")
                        signal = 1 if strategy == "Simple Buy" else -1
                    
                    # Position Management
                    if st.session_state.position:
                        st.markdown("---")
                        st.markdown("### 💼 Open Position")
                        
                        position = st.session_state.position
                        entry_price = position['entry_price']
                        sl_price = position['sl_price']
                        target_price = position['target_price']
                        signal_type = position['signal']
                        
                        # Calculate unrealized PnL
                        unrealized_pnl = (current_price - entry_price) if signal_type == 1 else (entry_price - current_price)
                        pnl_color = "green" if unrealized_pnl > 0 else "red"
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Entry Price", f"{entry_price:.2f}")
                        
                        sl_display = f"{sl_price:.2f}" if sl_price != 0 else "Signal Based"
                        col2.metric("Stop Loss", sl_display)
                        
                        target_display = f"{target_price:.2f}" if target_price != 0 else "Signal Based"
                        col3.metric("Target", target_display)
                        
                        col4.markdown(f"**Unrealized PnL**")
                        col4.markdown(f"<h3 style='color: {pnl_color};'>{unrealized_pnl:.2f}</h3>", unsafe_allow_html=True)
                        
                        # Distance to SL/Target
                        if sl_price != 0:
                            sl_distance = abs(current_price - sl_price)
                            st.info(f"📏 Distance to SL: {sl_distance:.2f} points")
                        
                        if target_price != 0:
                            target_distance = abs(target_price - current_price)
                            st.success(f"📏 Distance to Target: {target_distance:.2f} points")
                        
                        # Check exit conditions
                        should_exit, exit_reason, exit_price, pnl = check_exit_conditions(data, position, config)
                        
                        if should_exit:
                            # Record trade
                            trade_record = {
                                'entry_time': position['entry_time'],
                                'exit_time': get_ist_time().strftime("%Y-%m-%d %H:%M:%S"),
                                'signal': 'BUY' if signal_type == 1 else 'SELL',
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'sl_price': sl_price,
                                'target_price': target_price,
                                'exit_reason': exit_reason,
                                'pnl': pnl,
                                'strategy': config['strategy']
                            }
                            st.session_state.trade_history.append(trade_record)
                            
                            add_log(f"Position closed: {exit_reason}. PnL: {pnl:.2f}")
                            
                            # Placeholder for broker integration
                            # DHAN ORDER EXECUTION (FUTURE INTEGRATION)
                            # if signal_type == 1:
                            #     dhan.place_order(order_type='SELL', quantity=position['quantity'], price=exit_price)
                            # else:
                            #     dhan.place_order(order_type='BUY', quantity=position['quantity'], price=exit_price)
                            
                            reset_trading_state()
                            st.success(f"✅ Position Closed: {exit_reason}")
                            time.sleep(1)
                            st.rerun()
                    
                    else:
                        # Check for entry signal
                        if strategy == "EMA Crossover":
                            if signal != 0:
                                # Entry logic
                                entry_price, sl_price, target_price = calculate_sl_target(data, signal, sl_type, target_type, config)
                                
                                st.session_state.position = {
                                    'entry_time': get_ist_time().strftime("%Y-%m-%d %H:%M:%S"),
                                    'entry_price': entry_price,
                                    'sl_price': sl_price,
                                    'target_price': target_price,
                                    'signal': signal,
                                    'sl_type': sl_type,
                                    'target_type': target_type
                                }
                                
                                # Initialize trailing values
                                st.session_state.trailing_sl_high = entry_price
                                st.session_state.trailing_sl_low = entry_price
                                
                                signal_name = "BUY" if signal == 1 else "SELL"
                                add_log(f"{signal_name} signal detected. Entry: {entry_price:.2f}, SL: {sl_price:.2f}, Target: {target_price:.2f}")
                                
                                # Placeholder for broker integration
                                # DHAN ORDER EXECUTION (FUTURE INTEGRATION)
                                # if signal == 1:
                                #     dhan.place_order(order_type='BUY', quantity=lot_size, price=entry_price)
                                # else:
                                #     dhan.place_order(order_type='SELL', quantity=lot_size, price=entry_price)
                                
                                st.success(f"🎯 {signal_name} Signal Detected!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.info("⏳ Waiting for signal...")
                        
                        elif strategy in ["Simple Buy", "Simple Sell"]:
                            # Immediate entry for simple strategies
                            entry_price, sl_price, target_price = calculate_sl_target(data, signal, sl_type, target_type, config)
                            
                            st.session_state.position = {
                                'entry_time': get_ist_time().strftime("%Y-%m-%d %H:%M:%S"),
                                'entry_price': entry_price,
                                'sl_price': sl_price,
                                'target_price': target_price,
                                'signal': signal,
                                'sl_type': sl_type,
                                'target_type': target_type
                            }
                            
                            # Initialize trailing values
                            st.session_state.trailing_sl_high = entry_price
                            st.session_state.trailing_sl_low = entry_price
                            
                            signal_name = "BUY" if signal == 1 else "SELL"
                            add_log(f"{signal_name} entry executed. Entry: {entry_price:.2f}, SL: {sl_price:.2f}, Target: {target_price:.2f}")
                            
                            # Placeholder for broker integration
                            # DHAN ORDER EXECUTION (FUTURE INTEGRATION)
                            # if signal == 1:
                            #     dhan.place_order(order_type='BUY', quantity=lot_size, price=entry_price)
                            # else:
                            #     dhan.place_order(order_type='SELL', quantity=lot_size, price=entry_price)
                            
                            st.success(f"🎯 {signal_name} Position Opened!")
                            time.sleep(1)
                            st.rerun()
                    
                    # Live Chart
                    st.markdown("---")
                    st.markdown("### 📈 Live Chart")
                    chart_key = f"live_chart_{get_ist_time().timestamp()}"
                    fig = plot_live_chart(data, st.session_state.position, config)
                    st.plotly_chart(fig, use_container_width=True, key=chart_key)
                    
                    # Last update time
                    st.caption(f"Last updated: {get_ist_time().strftime('%Y-%m-%d %H:%M:%S IST')}")
                
                # Auto-refresh delay
                time.sleep(random.uniform(1.0, 1.5))
                
                # Check if trading is still active
                if not st.session_state.trading_active:
                    break
    
    # Tab 2: Trade History
    with tab2:
        st.markdown("### 📈 Trade History")
        
        if len(st.session_state.trade_history) == 0:
            st.info("No trades executed yet")
        else:
            # Calculate metrics
            total_trades = len(st.session_state.trade_history)
            winning_trades = sum(1 for t in st.session_state.trade_history if t['pnl'] > 0)
            accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = sum(t['pnl'] for t in st.session_state.trade_history)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Trades", total_trades)
            col2.metric("Winning Trades", winning_trades)
            col3.metric("Accuracy", f"{accuracy:.2f}%")
            
            pnl_color = "green" if total_pnl > 0 else "red"
            col4.markdown("**Total PnL**")
            col4.markdown(f"<h3 style='color: {pnl_color};'>{total_pnl:.2f}</h3>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Display trades
            for idx, trade in enumerate(reversed(st.session_state.trade_history), 1):
                pnl_emoji = "🟢" if trade['pnl'] > 0 else "🔴"
                with st.expander(f"Trade #{total_trades - idx + 1} - {trade['signal']} {pnl_emoji} PnL: {trade['pnl']:.2f}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Entry Time:** {trade['entry_time']}")
                        st.write(f"**Exit Time:** {trade['exit_time']}")
                        st.write(f"**Strategy:** {trade['strategy']}")
                        st.write(f"**Signal:** {trade['signal']}")
                    
                    with col2:
                        st.write(f"**Entry Price:** {trade['entry_price']:.2f}")
                        st.write(f"**Exit Price:** {trade['exit_price']:.2f}")
                        
                        sl_text = f"{trade['sl_price']:.2f}" if trade['sl_price'] != 0 else "Signal Based"
                        st.write(f"**Stop Loss:** {sl_text}")
                        
                        target_text = f"{trade['target_price']:.2f}" if trade['target_price'] != 0 else "Signal Based"
                        st.write(f"**Target:** {target_text}")
                    
                    st.write(f"**Exit Reason:** {trade['exit_reason']}")
                    
                    pnl_status = "🟢 Profit" if trade['pnl'] > 0 else "🔴 Loss"
                    st.write(f"**PnL:** {pnl_status} {trade['pnl']:.2f}")
    
    # Tab 3: Trade Logs
    with tab3:
        st.markdown("### 📝 Trade Logs")
        
        if len(st.session_state.trade_logs) == 0:
            st.info("No logs available")
        else:
            # Display logs in reverse order (newest first)
            for log in reversed(st.session_state.trade_logs):
                st.text(log)

if __name__ == "__main__":
    main()
