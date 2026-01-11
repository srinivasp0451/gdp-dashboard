import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import time
import random

# ==================== UTILITY FUNCTIONS ====================

def get_ist_time():
    """Get current time in IST"""
    return datetime.now(pytz.timezone('Asia/Kolkata'))

def convert_to_ist(df):
    """Convert dataframe index to IST timezone"""
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
    else:
        df.index = df.index.tz_convert('Asia/Kolkata')
    return df

def fetch_data_with_delay(ticker, period, interval):
    """Fetch data from yfinance with rate limit handling"""
    time.sleep(random.uniform(1.0, 1.5))
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return None
        
        # Flatten multi-index columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in data.columns.values]
            # Keep only OHLCV
            cols_to_keep = []
            for base in ['Open', 'High', 'Low', 'Close', 'Volume']:
                matching = [c for c in data.columns if base in c]
                if matching:
                    data[base] = data[matching[0]]
                    cols_to_keep.append(base)
            data = data[cols_to_keep]
        
        # Convert to IST
        data = convert_to_ist(data)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def validate_timeframe_combination(interval, period):
    """Validate if interval and period combination is allowed"""
    valid_combinations = {
        '1m': ['1d', '5d'],
        '5m': ['1d', '1mo'],
        '15m': ['1mo'],
        '30m': ['1mo'],
        '1h': ['1mo'],
        '4h': ['1mo'],
        '1d': ['1mo', '1y', '2y', '5y'],
        '1wk': ['1mo', '1y', '5y', '10y', '15y', '20y'],
        '1mo': ['1y', '2y', '5y', '10y', '15y', '20y', '25y', '30y']
    }
    return period in valid_combinations.get(interval, [])

# ==================== INDICATOR CALCULATIONS ====================

def calculate_ema(data, period):
    """Calculate EMA manually"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_adx(df, period=14):
    """Calculate ADX indicator"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_swing_points(df, lookback=5):
    """Calculate swing highs and lows"""
    swing_high = df['High'].rolling(window=lookback*2+1, center=True).max()
    swing_low = df['Low'].rolling(window=lookback*2+1, center=True).min()
    
    is_swing_high = df['High'] == swing_high
    is_swing_low = df['Low'] == swing_low
    
    return is_swing_high, is_swing_low

def calculate_ema_angle(df, fast_col='EMA_Fast', lookback=1):
    """Calculate EMA crossover angle in degrees"""
    ema_diff = df[fast_col] - df[fast_col].shift(lookback)
    angle = np.degrees(np.arctan(ema_diff / lookback))
    return angle

# ==================== STRATEGY LOGIC ====================

def check_ema_crossover_entry(df, i, ema_fast, ema_slow, min_angle, entry_filter, 
                               custom_points, atr_multiplier, use_adx, adx_threshold):
    """Check EMA crossover entry conditions with filters"""
    if i < 1:
        return 0, None
    
    # Basic crossover detection
    bullish_cross = (df['EMA_Fast'].iloc[i] > df['EMA_Slow'].iloc[i] and 
                     df['EMA_Fast'].iloc[i-1] <= df['EMA_Slow'].iloc[i-1])
    bearish_cross = (df['EMA_Fast'].iloc[i] < df['EMA_Slow'].iloc[i] and 
                     df['EMA_Fast'].iloc[i-1] >= df['EMA_Slow'].iloc[i-1])
    
    if not (bullish_cross or bearish_cross):
        return 0, None
    
    # Calculate angle
    angle = abs(df['EMA_Angle'].iloc[i])
    if angle < min_angle:
        return 0, None
    
    # Check ADX filter
    if use_adx:
        if df['ADX'].iloc[i] < adx_threshold:
            return 0, None
    
    # Entry filter validation
    candle_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
    filter_passed = False
    
    if entry_filter == "Simple Crossover":
        filter_passed = True
    elif entry_filter == "Custom Candle (Points)":
        filter_passed = candle_size >= custom_points
    elif entry_filter == "ATR-based Candle":
        min_candle = df['ATR'].iloc[i] * atr_multiplier
        filter_passed = candle_size >= min_candle
    
    if not filter_passed:
        return 0, None
    
    # Return signal
    if bullish_cross:
        return 1, df['Close'].iloc[i]
    elif bearish_cross:
        return -1, df['Close'].iloc[i]
    
    return 0, None

def check_rsi_adx_ema_strategy(df, i, ema_fast, ema_slow):
    """Strategy: RSI>80 ADX<20 EMA1<EMA2 = SELL, RSI<20 ADX>20 EMA1>EMA2 = BUY"""
    if i < 1:
        return 0, None
    
    rsi = df['RSI'].iloc[i]
    adx = df['ADX'].iloc[i]
    ema1 = df['EMA_Fast'].iloc[i]
    ema2 = df['EMA_Slow'].iloc[i]
    
    # SELL signal
    if rsi > 80 and adx < 20 and ema1 < ema2:
        return -1, df['Close'].iloc[i]
    
    # BUY signal
    if rsi < 20 and adx > 20 and ema1 > ema2:
        return 1, df['Close'].iloc[i]
    
    return 0, None

def check_price_threshold_entry(df, i, threshold, threshold_type):
    """Check price threshold crossing"""
    current_price = df['Close'].iloc[i]
    
    if threshold_type == "LONG (Price >= Threshold)":
        if current_price >= threshold:
            return 1, current_price
    elif threshold_type == "SHORT (Price >= Threshold)":
        if current_price >= threshold:
            return -1, current_price
    elif threshold_type == "LONG (Price <= Threshold)":
        if current_price <= threshold:
            return 1, current_price
    elif threshold_type == "SHORT (Price <= Threshold)":
        if current_price <= threshold:
            return -1, current_price
    
    return 0, None

def calculate_stop_loss(df, i, signal, entry_price, sl_type, custom_sl_points, 
                        atr_multiplier, trailing_sl_high, trailing_sl_low, 
                        trailing_threshold):
    """Calculate stop loss based on selected type"""
    min_sl_distance = 10
    
    if sl_type == "Custom Points":
        if signal == 1:  # LONG
            sl = entry_price - custom_sl_points
        else:  # SHORT
            sl = entry_price + custom_sl_points
    
    elif sl_type == "ATR-based":
        atr_value = df['ATR'].iloc[i]
        if signal == 1:
            sl = entry_price - (atr_value * atr_multiplier)
        else:
            sl = entry_price + (atr_value * atr_multiplier)
    
    elif sl_type == "Current Candle Low/High":
        if signal == 1:
            sl = df['Low'].iloc[i]
        else:
            sl = df['High'].iloc[i]
    
    elif sl_type == "Previous Candle Low/High":
        if i > 0:
            if signal == 1:
                sl = df['Low'].iloc[i-1]
            else:
                sl = df['High'].iloc[i-1]
        else:
            sl = entry_price - custom_sl_points if signal == 1 else entry_price + custom_sl_points
    
    elif sl_type == "Current Swing Low/High":
        swing_highs, swing_lows = calculate_swing_points(df)
        if signal == 1:
            recent_swing_low = df.loc[swing_lows, 'Low'].iloc[-1] if swing_lows.any() else df['Low'].iloc[i]
            sl = recent_swing_low
        else:
            recent_swing_high = df.loc[swing_highs, 'High'].iloc[-1] if swing_highs.any() else df['High'].iloc[i]
            sl = recent_swing_high
    
    elif sl_type == "Previous Swing Low/High":
        swing_highs, swing_lows = calculate_swing_points(df)
        if signal == 1:
            recent_swing_low = df.loc[swing_lows, 'Low'].iloc[-2] if swing_lows.sum() > 1 else df['Low'].iloc[i]
            sl = recent_swing_low
        else:
            recent_swing_high = df.loc[swing_highs, 'High'].iloc[-2] if swing_highs.sum() > 1 else df['High'].iloc[i]
            sl = recent_swing_high
    
    elif "Trailing SL" in sl_type:
        current_price = df['Close'].iloc[i]
        
        if sl_type == "Trailing SL (Points)":
            offset = custom_sl_points
        elif sl_type == "Trailing SL + Current Candle":
            offset = abs(current_price - (df['Low'].iloc[i] if signal == 1 else df['High'].iloc[i]))
        elif sl_type == "Trailing SL + Previous Candle":
            if i > 0:
                offset = abs(current_price - (df['Low'].iloc[i-1] if signal == 1 else df['High'].iloc[i-1]))
            else:
                offset = custom_sl_points
        elif sl_type == "Trailing SL + Current Swing":
            swing_highs, swing_lows = calculate_swing_points(df)
            if signal == 1:
                swing_low = df.loc[swing_lows, 'Low'].iloc[-1] if swing_lows.any() else df['Low'].iloc[i]
                offset = abs(current_price - swing_low)
            else:
                swing_high = df.loc[swing_highs, 'High'].iloc[-1] if swing_highs.any() else df['High'].iloc[i]
                offset = abs(current_price - swing_high)
        elif sl_type == "Trailing SL + Previous Swing":
            swing_highs, swing_lows = calculate_swing_points(df)
            if signal == 1:
                swing_low = df.loc[swing_lows, 'Low'].iloc[-2] if swing_lows.sum() > 1 else df['Low'].iloc[i]
                offset = abs(current_price - swing_low)
            else:
                swing_high = df.loc[swing_highs, 'High'].iloc[-2] if swing_highs.sum() > 1 else df['High'].iloc[i]
                offset = abs(current_price - swing_high)
        else:  # Trailing SL + Signal Based
            offset = custom_sl_points
        
        if signal == 1:  # LONG
            sl = trailing_sl_high - offset if trailing_sl_high else entry_price - offset
        else:  # SHORT
            sl = trailing_sl_low + offset if trailing_sl_low else entry_price + offset
    
    elif sl_type == "Signal-based (reverse EMA crossover)":
        # No fixed SL price for signal-based
        sl = 0
    
    else:
        sl = entry_price - custom_sl_points if signal == 1 else entry_price + custom_sl_points
    
    # Ensure minimum distance
    if sl != 0:
        if signal == 1:
            sl = min(sl, entry_price - min_sl_distance)
        else:
            sl = max(sl, entry_price + min_sl_distance)
    
    return sl

def calculate_target(df, i, signal, entry_price, target_type, custom_target_points, 
                     atr_multiplier, rr_ratio, trailing_target_high, trailing_target_low,
                     asset_name):
    """Calculate target based on selected type"""
    # Asset-specific minimum distances
    min_distances = {
        'NIFTY 50': 10,
        'BANKNIFTY': 20,
        'BTC': 150,
        'ETH': 10
    }
    min_target_distance = min_distances.get(asset_name, 15)
    
    if target_type == "Custom Points":
        if signal == 1:
            target = entry_price + custom_target_points
        else:
            target = entry_price - custom_target_points
    
    elif target_type == "ATR-based":
        atr_value = df['ATR'].iloc[i]
        if signal == 1:
            target = entry_price + (atr_value * atr_multiplier)
        else:
            target = entry_price - (atr_value * atr_multiplier)
    
    elif target_type == "Current Candle Low/High":
        if signal == 1:
            target = df['High'].iloc[i]
        else:
            target = df['Low'].iloc[i]
    
    elif target_type == "Previous Candle Low/High":
        if i > 0:
            if signal == 1:
                target = df['High'].iloc[i-1]
            else:
                target = df['Low'].iloc[i-1]
        else:
            target = entry_price + custom_target_points if signal == 1 else entry_price - custom_target_points
    
    elif target_type == "Current Swing Low/High":
        swing_highs, swing_lows = calculate_swing_points(df)
        if signal == 1:
            recent_swing_high = df.loc[swing_highs, 'High'].iloc[-1] if swing_highs.any() else df['High'].iloc[i]
            target = recent_swing_high
        else:
            recent_swing_low = df.loc[swing_lows, 'Low'].iloc[-1] if swing_lows.any() else df['Low'].iloc[i]
            target = recent_swing_low
    
    elif target_type == "Previous Swing Low/High":
        swing_highs, swing_lows = calculate_swing_points(df)
        if signal == 1:
            recent_swing_high = df.loc[swing_highs, 'High'].iloc[-2] if swing_highs.sum() > 1 else df['High'].iloc[i]
            target = recent_swing_high
        else:
            recent_swing_low = df.loc[swing_lows, 'Low'].iloc[-2] if swing_lows.sum() > 1 else df['Low'].iloc[i]
            target = recent_swing_low
    
    elif target_type == "Risk-Reward Based":
        sl_distance = abs(entry_price - df['Low'].iloc[i]) if signal == 1 else abs(entry_price - df['High'].iloc[i])
        if signal == 1:
            target = entry_price + (sl_distance * rr_ratio)
        else:
            target = entry_price - (sl_distance * rr_ratio)
    
    elif "Trailing Target" in target_type:
        # Trailing target - display only, never exits
        if signal == 1:
            target = trailing_target_high if trailing_target_high else entry_price + min_target_distance
        else:
            target = trailing_target_low if trailing_target_low else entry_price - min_target_distance
    
    elif target_type == "Signal-based (reverse EMA crossover)":
        # No fixed target for signal-based
        target = 0
    
    else:
        target = entry_price + custom_target_points if signal == 1 else entry_price - custom_target_points
    
    # Ensure minimum distance
    if target != 0:
        if signal == 1:
            target = max(target, entry_price + min_target_distance)
        else:
            target = min(target, entry_price - min_target_distance)
    
    return target

def check_exit_conditions(df, i, position, current_price, sl, target, sl_type, target_type,
                          trailing_sl_high, trailing_sl_low, trailing_threshold,
                          trailing_target_high, trailing_target_low, trailing_profit_points,
                          target_points):
    """Check if exit conditions are met"""
    signal = position['signal']
    entry_price = position['entry_price']
    
    # Update trailing stops
    new_trailing_sl_high = trailing_sl_high
    new_trailing_sl_low = trailing_sl_low
    new_sl = sl
    
    if "Trailing SL" in sl_type:
        if signal == 1:  # LONG
            if current_price > trailing_sl_high:
                price_move = current_price - trailing_sl_high
                if price_move >= trailing_threshold:
                    new_trailing_sl_high = current_price
                    # Recalculate SL
                    new_sl = calculate_stop_loss(df, i, signal, entry_price, sl_type, 
                                                position.get('custom_sl_points', 10),
                                                position.get('atr_multiplier', 1.5),
                                                new_trailing_sl_high, new_trailing_sl_low,
                                                trailing_threshold)
        else:  # SHORT
            if current_price < trailing_sl_low:
                price_move = trailing_sl_low - current_price
                if price_move >= trailing_threshold:
                    new_trailing_sl_low = current_price
                    new_sl = calculate_stop_loss(df, i, signal, entry_price, sl_type,
                                                position.get('custom_sl_points', 10),
                                                position.get('atr_multiplier', 1.5),
                                                new_trailing_sl_high, new_trailing_sl_low,
                                                trailing_threshold)
    
    # Update trailing targets (display only, no exit)
    new_trailing_target_high = trailing_target_high
    new_trailing_target_low = trailing_target_low
    new_trailing_profit_points = trailing_profit_points
    new_target = target
    
    if "Trailing Target" in target_type and target_type != "Trailing Target + Signal Based":
        if signal == 1:  # LONG
            if current_price > trailing_target_high:
                profit_points = current_price - entry_price
                if profit_points >= trailing_profit_points + target_points:
                    new_trailing_profit_points = profit_points
                    new_trailing_target_high = current_price
                    new_target = new_trailing_target_high
        else:  # SHORT
            if current_price < trailing_target_low:
                profit_points = entry_price - current_price
                if profit_points >= trailing_profit_points + target_points:
                    new_trailing_profit_points = profit_points
                    new_trailing_target_low = current_price
                    new_target = new_trailing_target_low
    
    # Check signal-based exits
    if sl_type == "Signal-based (reverse EMA crossover)" or target_type == "Signal-based (reverse EMA crossover)" or target_type == "Trailing Target + Signal Based":
        if i > 0:
            if signal == 1:  # LONG - check for bearish crossover
                if (df['EMA_Fast'].iloc[i] < df['EMA_Slow'].iloc[i] and 
                    df['EMA_Fast'].iloc[i-1] >= df['EMA_Slow'].iloc[i-1]):
                    return True, "Reverse Signal - Bearish Crossover", current_price, new_sl, new_target, new_trailing_sl_high, new_trailing_sl_low, new_trailing_target_high, new_trailing_target_low, new_trailing_profit_points
            else:  # SHORT - check for bullish crossover
                if (df['EMA_Fast'].iloc[i] > df['EMA_Slow'].iloc[i] and 
                    df['EMA_Fast'].iloc[i-1] <= df['EMA_Slow'].iloc[i-1]):
                    return True, "Reverse Signal - Bullish Crossover", current_price, new_sl, new_target, new_trailing_sl_high, new_trailing_sl_low, new_trailing_target_high, new_trailing_target_low, new_trailing_profit_points
    
    # Check SL hit (only for non-signal-based SL)
    if sl_type != "Signal-based (reverse EMA crossover)" and new_sl != 0:
        if signal == 1 and current_price <= new_sl:
            return True, "Stop Loss Hit", current_price, new_sl, new_target, new_trailing_sl_high, new_trailing_sl_low, new_trailing_target_high, new_trailing_target_low, new_trailing_profit_points
        elif signal == -1 and current_price >= new_sl:
            return True, "Stop Loss Hit", current_price, new_sl, new_target, new_trailing_sl_high, new_trailing_sl_low, new_trailing_target_high, new_trailing_target_low, new_trailing_profit_points
    
    # Check target hit (only for non-signal-based and non-trailing targets)
    if target_type not in ["Signal-based (reverse EMA crossover)", "Trailing Target (Points)", "Trailing Target + Signal Based"] and new_target != 0:
        if signal == 1 and current_price >= new_target:
            return True, "Target Hit", current_price, new_sl, new_target, new_trailing_sl_high, new_trailing_sl_low, new_trailing_target_high, new_trailing_target_low, new_trailing_profit_points
        elif signal == -1 and current_price <= new_target:
            return True, "Target Hit", current_price, new_sl, new_target, new_trailing_sl_high, new_trailing_sl_low, new_trailing_target_high, new_trailing_target_low, new_trailing_profit_points
    
    return False, None, None, new_sl, new_target, new_trailing_sl_high, new_trailing_sl_low, new_trailing_target_high, new_trailing_target_low, new_trailing_profit_points

# ==================== UI FUNCTIONS ====================

def create_live_chart(df, position=None, sl=None, target=None):
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
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], 
                                mode='lines', name='EMA Fast', 
                                line=dict(color='blue', width=1)))
    if 'EMA_Slow' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], 
                                mode='lines', name='EMA Slow', 
                                line=dict(color='orange', width=1)))
    
    # Position lines
    if position:
        entry_price = position['entry_price']
        fig.add_hline(y=entry_price, line_dash="dash", line_color="white", 
                     annotation_text=f"Entry: {entry_price:.2f}")
        
        if sl and sl != 0:
            fig.add_hline(y=sl, line_dash="dash", line_color="red", 
                         annotation_text=f"SL: {sl:.2f}")
        
        if target and target != 0:
            fig.add_hline(y=target, line_dash="dash", line_color="green", 
                         annotation_text=f"Target: {target:.2f}")
    
    fig.update_layout(
        title="Live Trading Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        height=600,
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )
    
    return fig

def add_log(message):
    """Add timestamped log entry"""
    timestamp = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state['trade_logs'].append(log_entry)
    # Keep only last 50 logs
    if len(st.session_state['trade_logs']) > 50:
        st.session_state['trade_logs'] = st.session_state['trade_logs'][-50:]
    # Force update
    st.session_state['trade_logs'] = st.session_state['trade_logs']

def format_duration(seconds):
    """Format duration in human readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"

# ==================== SESSION STATE INITIALIZATION ====================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'trading_active' not in st.session_state:
        st.session_state['trading_active'] = False
    if 'current_data' not in st.session_state:
        st.session_state['current_data'] = None
    if 'position' not in st.session_state:
        st.session_state['position'] = None
    if 'trade_history' not in st.session_state:
        st.session_state['trade_history'] = []
    if 'trade_logs' not in st.session_state:
        st.session_state['trade_logs'] = []
    if 'trailing_sl_high' not in st.session_state:
        st.session_state['trailing_sl_high'] = None
    if 'trailing_sl_low' not in st.session_state:
        st.session_state['trailing_sl_low'] = None
    if 'trailing_target_high' not in st.session_state:
        st.session_state['trailing_target_high'] = None
    if 'trailing_target_low' not in st.session_state:
        st.session_state['trailing_target_low'] = None
    if 'trailing_profit_points' not in st.session_state:
        st.session_state['trailing_profit_points'] = 0
    if 'threshold_crossed' not in st.session_state:
        st.session_state['threshold_crossed'] = False
    if 'highest_price' not in st.session_state:
        st.session_state['highest_price'] = None
    if 'lowest_price' not in st.session_state:
        st.session_state['lowest_price'] = None

# ==================== MAIN APPLICATION ====================

def main():
    st.set_page_config(page_title="Live Trading Dashboard", layout="wide", initial_sidebar_state="expanded")
    
    initialize_session_state()
    
    st.title("🚀 Live Trading Dashboard - Production Grade")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Trading Configuration")
        
        # Asset Selection
        st.subheader("Asset Selection")
        asset_type = st.selectbox("Asset Type", 
            ["Indian Indices", "Crypto", "Forex", "Commodities", "Custom Ticker"])
        
        if asset_type == "Indian Indices":
            asset = st.selectbox("Select Index", ["^NSEI", "^NSEBANK", "^BSESN"])
            asset_map = {"^NSEI": "NIFTY 50", "^NSEBANK": "BANKNIFTY", "^BSESN": "SENSEX"}
            asset_name = asset_map[asset]
        elif asset_type == "Crypto":
            crypto = st.selectbox("Select Crypto", ["BTC-USD", "ETH-USD"])
            asset = crypto
            asset_name = "BTC" if crypto == "BTC-USD" else "ETH"
        elif asset_type == "Forex":
            asset = st.selectbox("Select Forex", ["USDINR=X", "EURUSD=X", "GBPUSD=X"])
            asset_name = asset.replace("=X", "")
        elif asset_type == "Commodities":
            asset = st.selectbox("Select Commodity", ["GC=F", "SI=F"])
            asset_name = "Gold" if asset == "GC=F" else "Silver"
        else:
            asset = st.text_input("Enter Custom Ticker", "AAPL")
            asset_name = asset
        
        # Timeframe Selection
        st.subheader("Timeframe")
        interval = st.selectbox("Interval", 
            ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"])
        
        # Period options based on interval
        period_options = {
            '1m': ['1d', '5d'],
            '5m': ['1d', '1mo'],
            '15m': ['1mo'],
            '30m': ['1mo'],
            '1h': ['1mo'],
            '4h': ['1mo'],
            '1d': ['1mo', '1y', '2y', '5y'],
            '1wk': ['1mo', '1y', '5y', '10y', '15y', '20y'],
            '1mo': ['1y', '2y', '5y', '10y', '15y', '20y', '25y', '30y']
        }
        
        period = st.selectbox("Period", period_options[interval])
        
        if not validate_timeframe_combination(interval, period):
            st.error("⚠️ Invalid timeframe combination!")
        
        # Strategy Selection
        st.subheader("Strategy")
        strategy = st.selectbox("Select Strategy", 
            ["EMA Crossover", "Simple Buy", "Simple Sell", "Price Threshold", "RSI-ADX-EMA Strategy"])
        
        # Strategy-specific parameters
        if strategy == "EMA Crossover":
            st.markdown("**EMA Parameters**")
            ema_fast = st.number_input("EMA Fast", min_value=2, max_value=200, value=9)
            ema_slow = st.number_input("EMA Slow", min_value=2, max_value=200, value=15)
            min_angle = st.number_input("Minimum Crossover Angle (degrees)", 
                                       min_value=0.0, max_value=90.0, value=1.0, step=0.1)
            
            st.markdown("**Entry Filter**")
            entry_filter = st.selectbox("Entry Filter Type", 
                ["Simple Crossover", "Custom Candle (Points)", "ATR-based Candle"])
            
            custom_points = 0
            atr_multiplier = 1.5
            if entry_filter == "Custom Candle (Points)":
                custom_points = st.number_input("Minimum Candle Size (Points)", 
                                               min_value=1, value=10)
            elif entry_filter == "ATR-based Candle":
                atr_multiplier = st.number_input("ATR Multiplier", 
                                                min_value=0.1, max_value=5.0, value=1.5, step=0.1)
            
            st.markdown("**ADX Filter**")
            use_adx = st.checkbox("Enable ADX Filter", value=False)
            adx_threshold = 25
            adx_period = 14
            if use_adx:
                adx_period = st.number_input("ADX Period", min_value=2, max_value=50, value=14)
                adx_threshold = st.number_input("ADX Threshold", 
                                               min_value=0, max_value=100, value=25)
        
        elif strategy == "Price Threshold":
            threshold = st.number_input("Price Threshold", min_value=0.0, value=100.0, step=0.1)
            threshold_type = st.selectbox("Threshold Type", 
                ["LONG (Price >= Threshold)", "SHORT (Price >= Threshold)", 
                 "LONG (Price <= Threshold)", "SHORT (Price <= Threshold)"])
        
        elif strategy == "RSI-ADX-EMA Strategy":
            ema_fast = st.number_input("EMA Fast", min_value=2, max_value=200, value=9)
            ema_slow = st.number_input("EMA Slow", min_value=2, max_value=200, value=15)
        
        # Stop Loss Configuration
        st.subheader("Stop Loss")
        sl_type = st.selectbox("Stop Loss Type", [
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
            "Signal-based (reverse EMA crossover)"
        ])
        
        custom_sl_points = 10
        sl_atr_multiplier = 1.5
        trailing_threshold = 0
        
        if sl_type == "Custom Points":
            custom_sl_points = st.number_input("SL Points", min_value=1, value=10)
        elif "Trailing SL" in sl_type:
            custom_sl_points = st.number_input("Trailing Offset (Points)", min_value=1, value=10)
            trailing_threshold = st.number_input("Trailing Threshold (Points)", 
                                                 min_value=0, value=0, 
                                                 help="SL updates only after price moves by this amount")
        elif sl_type == "ATR-based":
            sl_atr_multiplier = st.number_input("ATR Multiplier (SL)", 
                                               min_value=0.1, max_value=5.0, value=1.5, step=0.1)
        
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
            "Signal-based (reverse EMA crossover)"
        ])
        
        custom_target_points = 20
        target_atr_multiplier = 2.0
        rr_ratio = 2.0
        
        if target_type == "Custom Points":
            custom_target_points = st.number_input("Target Points", min_value=1, value=20)
        elif "Trailing Target" in target_type:
            custom_target_points = st.number_input("Trailing Target Points", min_value=1, value=20,
                                                   help="Target updates after price moves by this amount")
        elif target_type == "ATR-based":
            target_atr_multiplier = st.number_input("ATR Multiplier (Target)", 
                                                   min_value=0.1, max_value=10.0, value=2.0, step=0.1)
        elif target_type == "Risk-Reward Based":
            rr_ratio = st.number_input("Risk:Reward Ratio", 
                                      min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    
    # Main Trading Interface - 3 Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Live Trading Dashboard", "📈 Trade History", "📝 Trade Logs"])
    
    # ==================== TAB 1: LIVE TRADING DASHBOARD ====================
    with tab1:
        # Trading Controls at TOP
        st.markdown("### 🎮 Trading Controls")
        col1, col2, col3 = st.columns([1, 1, 3])
        
        with col1:
            if st.button("▶️ Start Trading", type="primary", use_container_width=True):
                st.session_state['trading_active'] = True
                add_log("Trading started by user")
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop Trading", type="secondary", use_container_width=True):
                st.session_state['trading_active'] = False
                add_log("Trading stopped by user")
                st.rerun()
        
        with col3:
            if st.session_state['trading_active']:
                st.success("🟢 Trading is ACTIVE")
            else:
                st.info("⚪ Trading is STOPPED")
        
        st.divider()
        
        # Live Trading Loop
        if st.session_state['trading_active']:
            # Container for live updates
            status_container = st.container()
            metrics_container = st.container()
            chart_container = st.container()
            position_container = st.container()
            
            while st.session_state['trading_active']:
                # Fetch fresh data
                with status_container:
                    with st.spinner("Fetching live data..."):
                        df = fetch_data_with_delay(asset, period, interval)
                
                if df is None or df.empty:
                    st.error("Failed to fetch data. Stopping trading.")
                    st.session_state['trading_active'] = False
                    break
                
                # Calculate indicators
                df['EMA_Fast'] = calculate_ema(df['Close'], ema_fast if strategy in ["EMA Crossover", "RSI-ADX-EMA Strategy"] else 9)
                df['EMA_Slow'] = calculate_ema(df['Close'], ema_slow if strategy in ["EMA Crossover", "RSI-ADX-EMA Strategy"] else 15)
                df['EMA_Angle'] = calculate_ema_angle(df)
                df['ATR'] = calculate_atr(df)
                df['ADX'] = calculate_adx(df, adx_period if strategy == "EMA Crossover" else 14)
                df['RSI'] = calculate_rsi(df['Close'])
                
                st.session_state['current_data'] = df
                
                # Get current index
                i = len(df) - 1
                current_price = df['Close'].iloc[i]
                current_time = get_ist_time()
                
                # Live Metrics Display
                with metrics_container:
                    st.markdown("### 📊 Live Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Current Price", f"{current_price:.2f}")
                    
                    with col2:
                        position_status = "IN POSITION" if st.session_state['position'] else "NO POSITION"
                        st.metric("Position Status", position_status)
                    
                    with col3:
                        if st.session_state['position']:
                            entry_price = st.session_state['position']['entry_price']
                            signal = st.session_state['position']['signal']
                            pnl = (current_price - entry_price) * signal
                            pnl_color = "normal" if pnl >= 0 else "inverse"
                            st.metric("Unrealized P&L", f"{pnl:.2f}", delta=f"{pnl:.2f}", delta_color=pnl_color)
                        else:
                            st.metric("Unrealized P&L", "0.00")
                    
                    with col4:
                        st.metric("Last Update", current_time.strftime("%H:%M:%S"))
                    
                    # Additional Metrics
                    col5, col6, col7, col8 = st.columns(4)
                    
                    with col5:
                        st.metric("EMA Fast", f"{df['EMA_Fast'].iloc[i]:.2f}")
                    
                    with col6:
                        st.metric("EMA Slow", f"{df['EMA_Slow'].iloc[i]:.2f}")
                    
                    with col7:
                        angle = df['EMA_Angle'].iloc[i]
                        st.metric("Crossover Angle", f"{angle:.2f}°")
                    
                    with col8:
                        if strategy == "EMA Crossover" and use_adx:
                            adx_val = df['ADX'].iloc[i]
                            adx_status = "✅" if adx_val >= adx_threshold else "❌"
                            st.metric("ADX", f"{adx_val:.2f} {adx_status}")
                        else:
                            st.metric("ADX", f"{df['ADX'].iloc[i]:.2f}")
                    
                    # Entry Filter Status
                    if strategy == "EMA Crossover":
                        st.markdown("**Entry Filter Status**")
                        candle_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
                        
                        if entry_filter == "Simple Crossover":
                            st.info(f"✅ Simple Crossover Mode - Angle: {angle:.2f}° (Min: {min_angle}°)")
                        elif entry_filter == "Custom Candle (Points)":
                            status = "✅" if candle_size >= custom_points else "❌"
                            st.info(f"{status} Candle Size: {candle_size:.2f} / Min: {custom_points}")
                        elif entry_filter == "ATR-based Candle":
                            min_candle = df['ATR'].iloc[i] * atr_multiplier
                            status = "✅" if candle_size >= min_candle else "❌"
                            min_str = f"{min_candle:.2f}"
                            st.info(f"{status} Candle Size: {candle_size:.2f} / Min (ATR×{atr_multiplier}): {min_str}")
                
                # Check for entry if no position
                if st.session_state['position'] is None:
                    signal = 0
                    entry_price = None
                    
                    if strategy == "EMA Crossover":
                        signal, entry_price = check_ema_crossover_entry(
                            df, i, ema_fast, ema_slow, min_angle, entry_filter, 
                            custom_points, atr_multiplier, use_adx, adx_threshold
                        )
                    elif strategy == "Simple Buy":
                        signal = 1
                        entry_price = current_price
                    elif strategy == "Simple Sell":
                        signal = -1
                        entry_price = current_price
                    elif strategy == "Price Threshold":
                        if not st.session_state['threshold_crossed']:
                            signal, entry_price = check_price_threshold_entry(
                                df, i, threshold, threshold_type
                            )
                            if signal != 0:
                                st.session_state['threshold_crossed'] = True
                    elif strategy == "RSI-ADX-EMA Strategy":
                        signal, entry_price = check_rsi_adx_ema_strategy(df, i, ema_fast, ema_slow)
                    
                    # Enter position
                    if signal != 0 and entry_price:
                        # Initialize trailing values
                        st.session_state['trailing_sl_high'] = entry_price
                        st.session_state['trailing_sl_low'] = entry_price
                        st.session_state['trailing_target_high'] = entry_price
                        st.session_state['trailing_target_low'] = entry_price
                        st.session_state['trailing_profit_points'] = 0
                        st.session_state['highest_price'] = entry_price
                        st.session_state['lowest_price'] = entry_price
                        
                        # Calculate SL and Target
                        sl = calculate_stop_loss(
                            df, i, signal, entry_price, sl_type, custom_sl_points,
                            sl_atr_multiplier, st.session_state['trailing_sl_high'],
                            st.session_state['trailing_sl_low'], trailing_threshold
                        )
                        
                        target = calculate_target(
                            df, i, signal, entry_price, target_type, custom_target_points,
                            target_atr_multiplier, rr_ratio, st.session_state['trailing_target_high'],
                            st.session_state['trailing_target_low'], asset_name
                        )
                        
                        # Create position
                        st.session_state['position'] = {
                            'signal': signal,
                            'entry_price': entry_price,
                            'entry_time': current_time,
                            'sl': sl,
                            'target': target,
                            'custom_sl_points': custom_sl_points,
                            'custom_target_points': custom_target_points,
                            'atr_multiplier': sl_atr_multiplier,
                            'sl_type': sl_type,
                            'target_type': target_type
                        }
                        
                        signal_str = "LONG" if signal == 1 else "SHORT"
                        sl_str = f"{sl:.2f}" if sl != 0 else "Signal Based"
                        target_str = f"{target:.2f}" if target != 0 else "Signal Based"
                        add_log(f"Entered {signal_str} position at {entry_price:.2f} | SL: {sl_str} | Target: {target_str}")
                
                # Check exit conditions if in position
                elif st.session_state['position']:
                    position = st.session_state['position']
                    
                    # Update highest/lowest prices
                    if st.session_state['highest_price'] is None or current_price > st.session_state['highest_price']:
                        st.session_state['highest_price'] = current_price
                    if st.session_state['lowest_price'] is None or current_price < st.session_state['lowest_price']:
                        st.session_state['lowest_price'] = current_price
                    
                    exit_triggered, exit_reason, exit_price, new_sl, new_target, new_tsl_h, new_tsl_l, new_tt_h, new_tt_l, new_tp = check_exit_conditions(
                        df, i, position, current_price, position['sl'], position['target'],
                        position['sl_type'], position['target_type'],
                        st.session_state['trailing_sl_high'], st.session_state['trailing_sl_low'],
                        trailing_threshold, st.session_state['trailing_target_high'],
                        st.session_state['trailing_target_low'], st.session_state['trailing_profit_points'],
                        custom_target_points
                    )
                    
                    # Update trailing values
                    st.session_state['trailing_sl_high'] = new_tsl_h
                    st.session_state['trailing_sl_low'] = new_tsl_l
                    st.session_state['trailing_target_high'] = new_tt_h
                    st.session_state['trailing_target_low'] = new_tt_l
                    st.session_state['trailing_profit_points'] = new_tp
                    position['sl'] = new_sl
                    position['target'] = new_target
                    
                    if exit_triggered:
                        # Calculate PnL
                        signal = position['signal']
                        entry_price = position['entry_price']
                        pnl = (exit_price - entry_price) * signal
                        
                        # Calculate duration
                        duration = (current_time - position['entry_time']).total_seconds()
                        
                        # Record trade
                        trade_record = {
                            'entry_time': position['entry_time'].strftime("%Y-%m-%d %H:%M:%S"),
                            'exit_time': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            'duration': format_duration(duration),
                            'signal': "LONG" if signal == 1 else "SHORT",
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'sl': position.get('sl', 0),
                            'target': position.get('target', 0),
                            'exit_reason': exit_reason,
                            'pnl': pnl,
                            'highest_price': st.session_state.get('highest_price', entry_price),
                            'lowest_price': st.session_state.get('lowest_price', entry_price)
                        }
                        
                        st.session_state['trade_history'].append(trade_record)
                        st.session_state['trade_history'] = st.session_state['trade_history']
                        
                        add_log(f"Exited {trade_record['signal']} position | Exit: {exit_price:.2f} | Reason: {exit_reason} | PnL: {pnl:.2f}")
                        
                        # Reset position
                        st.session_state['position'] = None
                        st.session_state['trailing_sl_high'] = None
                        st.session_state['trailing_sl_low'] = None
                        st.session_state['trailing_target_high'] = None
                        st.session_state['trailing_target_low'] = None
                        st.session_state['trailing_profit_points'] = 0
                        st.session_state['highest_price'] = None
                        st.session_state['lowest_price'] = None
                
                # Display Current Signal
                with metrics_container:
                    st.markdown("### 🎯 Current Signal")
                    if st.session_state['position']:
                        signal = st.session_state['position']['signal']
                        signal_text = "🟢 LONG" if signal == 1 else "🔴 SHORT"
                        st.success(f"**IN POSITION: {signal_text}**")
                    else:
                        st.info("**WAITING FOR SIGNAL**")
                
                # Position Information
                if st.session_state['position']:
                    with position_container:
                        st.markdown("### 💼 Position Information")
                        pos = st.session_state['position']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Entry Details**")
                            st.text(f"Entry Time: {pos['entry_time'].strftime('%H:%M:%S')}")
                            duration = (current_time - pos['entry_time']).total_seconds()
                            st.text(f"Duration: {format_duration(duration)}")
                            st.text(f"Entry Price: {pos['entry_price']:.2f}")
                        
                        with col2:
                            st.markdown("**Stop Loss & Target**")
                            sl_display = f"{pos['sl']:.2f}" if pos['sl'] != 0 else "Signal Based"
                            target_display = f"{pos['target']:.2f}" if pos['target'] != 0 else "Signal Based"
                            st.text(f"Stop Loss: {sl_display}")
                            st.text(f"Target: {target_display}")
                            
                            if pos['sl'] != 0:
                                dist_sl = abs(current_price - pos['sl'])
                                st.text(f"Distance to SL: {dist_sl:.2f}")
                            if pos['target'] != 0:
                                dist_target = abs(pos['target'] - current_price)
                                st.text(f"Distance to Target: {dist_target:.2f}")
                        
                        with col3:
                            st.markdown("**Price Range**")
                            highest = st.session_state.get('highest_price', pos['entry_price'])
                            lowest = st.session_state.get('lowest_price', pos['entry_price'])
                            st.text(f"Highest: {highest:.2f}")
                            st.text(f"Lowest: {lowest:.2f}")
                            st.text(f"Range: {abs(highest - lowest):.2f}")
                        
                        # Trailing Target Info
                        if "Trailing Target" in pos['target_type']:
                            st.info(f"📈 Profit moved: {st.session_state['trailing_profit_points']:.2f} points | Next update at: {st.session_state['trailing_profit_points'] + custom_target_points:.2f} points")
                        
                        # Guidance
                        pnl = (current_price - pos['entry_price']) * pos['signal']
                        if pnl > 0:
                            st.success("💰 **In Profit** - Hold for target or trailing exit")
                        elif pnl < 0:
                            st.warning("⚠️ **In Loss** - Monitor stop loss")
                        else:
                            st.info("➡️ **Breakeven** - Wait for price movement")
                
                # Display Chart
                with chart_container:
                    st.markdown("### 📈 Live Chart")
                    fig = create_live_chart(
                        df,
                        st.session_state['position'],
                        st.session_state['position']['sl'] if st.session_state['position'] else None,
                        st.session_state['position']['target'] if st.session_state['position'] else None
                    )
                    chart_key = f"live_chart_{get_ist_time().timestamp()}"
                    st.plotly_chart(fig, use_container_width=True, key=chart_key)
                
                # Sleep before next iteration
                time.sleep(random.uniform(1.0, 1.5))
                
                # Force refresh
                st.rerun()
        
        else:
            st.info("👆 Click **Start Trading** to begin live monitoring")
            
            # Show last known data if available
            if st.session_state['current_data'] is not None:
                st.markdown("### 📊 Last Known Data")
                df = st.session_state['current_data']
                fig = create_live_chart(df)
                st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 2: TRADE HISTORY ====================
    with tab2:
        st.markdown("### 📈 Trade History")
        
        if len(st.session_state['trade_history']) == 0:
            st.info("No trades executed yet. Start trading to see history.")
        else:
            # Calculate statistics
            total_trades = len(st.session_state['trade_history'])
            winning_trades = sum(1 for t in st.session_state['trade_history'] if t.get('pnl', 0) > 0)
            losing_trades = sum(1 for t in st.session_state['trade_history'] if t.get('pnl', 0) < 0)
            accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = sum(t.get('pnl', 0) for t in st.session_state['trade_history'])
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Trades", total_trades)
            with col2:
                st.metric("Winning Trades", winning_trades)
            with col3:
                st.metric("Losing Trades", losing_trades)
            with col4:
                st.metric("Accuracy", f"{accuracy:.1f}%")
            with col5:
                pnl_color = "normal" if total_pnl >= 0 else "inverse"
                st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color=pnl_color)
            
            st.divider()
            
            # Display individual trades
            for idx, trade in enumerate(reversed(st.session_state['trade_history']), 1):
                pnl = trade.get('pnl', 0)
                pnl_emoji = "✅" if pnl > 0 else "❌"
                signal = trade.get('signal', 'UNKNOWN')
                
                with st.expander(f"{pnl_emoji} Trade #{total_trades - idx + 1} - {signal} - P&L: {pnl:.2f}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Entry Information**")
                        st.text(f"Entry Time: {trade.get('entry_time', 'N/A')}")
                        st.text(f"Entry Price: {trade.get('entry_price', 0):.2f}")
                        st.text(f"Signal: {signal}")
                        sl_val = trade.get('sl', 0)
                        sl_display = f"{sl_val:.2f}" if sl_val != 0 else "Signal Based"
                        st.text(f"Stop Loss: {sl_display}")
                        target_val = trade.get('target', 0)
                        target_display = f"{target_val:.2f}" if target_val != 0 else "Signal Based"
                        st.text(f"Target: {target_display}")
                    
                    with col2:
                        st.markdown("**Exit Information**")
                        st.text(f"Exit Time: {trade.get('exit_time', 'N/A')}")
                        st.text(f"Exit Price: {trade.get('exit_price', 0):.2f}")
                        st.text(f"Exit Reason: {trade.get('exit_reason', 'N/A')}")
                        st.text(f"Duration: {trade.get('duration', 'N/A')}")
                        st.text(f"P&L: {pnl:.2f}")
                    
                    st.markdown("**Price Range During Trade**")
                    col3, col4, col5 = st.columns(3)
                    with col3:
                        st.text(f"Highest: {trade.get('highest_price', 0):.2f}")
                    with col4:
                        st.text(f"Lowest: {trade.get('lowest_price', 0):.2f}")
                    with col5:
                        price_range = abs(trade.get('highest_price', 0) - trade.get('lowest_price', 0))
                        st.text(f"Range: {price_range:.2f}")
    
    # ==================== TAB 3: TRADE LOGS ====================
    with tab3:
        st.markdown("### 📝 Trade Logs")
        
        if len(st.session_state['trade_logs']) == 0:
            st.info("No logs available yet. Start trading to generate logs.")
        else:
            st.markdown(f"**Showing last {len(st.session_state['trade_logs'])} log entries**")
            st.divider()
            
            for log in reversed(st.session_state['trade_logs']):
                st.text(log)

if __name__ == "__main__":
    main()
