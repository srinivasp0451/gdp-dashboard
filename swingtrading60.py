import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
import pytz

# Page config
st.set_page_config(page_title="Quantitative Trading System", layout="wide")

# Constants
ASSET_MAPPING = {
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "USD-INR": "USDINR=X",
    "EUR-USD": "EURUSD=X",
    "GBP-USD": "GBPUSD=X",
    "Gold": "GC=F",
    "Silver": "SI=F"
}

INTERVAL_PERIODS = {
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

IST = pytz.timezone('Asia/Kolkata')

# Initialize session state
def init_session_state():
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
        st.session_state.last_fetch_time = None

def reset_position_state():
    """Reset all position-related session state variables"""
    st.session_state.in_position = False
    st.session_state.position_type = 0
    st.session_state.entry_price = 0
    st.session_state.stop_loss = 0
    st.session_state.target = 0
    st.session_state.entry_idx = 0
    st.session_state.entry_time = None
    st.session_state.trailing_sl_highest = 0
    st.session_state.trailing_sl_lowest = 0

# Data fetching with rate limiting
def fetch_data_with_delay(ticker, interval, period):
    time.sleep(random.uniform(1.0, 1.5))
    try:
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if data.empty:
            return None
        
        # Flatten multi-index columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[0] != '' else col[1] for col in data.columns]
        
        # Select OHLCV
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[required_cols].copy()
        
        # Handle timezone
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert(IST)
        else:
            data.index = data.index.tz_convert(IST)
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Calculate EMA
def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

# Calculate EMA angle
def calculate_ema_angle(ema_series, lookback=2):
    if len(ema_series) < lookback + 1:
        return 0
    
    y_diff = ema_series.iloc[-1] - ema_series.iloc[-lookback-1]
    x_diff = lookback
    
    angle_rad = np.arctan2(y_diff, x_diff)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

# Calculate ADX
def calculate_adx(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Calculate +DM and -DM
    up_move = high.diff()
    down_move = -low.diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # Smooth TR, +DM, -DM
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx, plus_di, minus_di

# Calculate ATR
def calculate_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    return atr

# Find swing points
def find_swing_points(df, lookback=5):
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(df) - lookback):
        is_swing_high = True
        is_swing_low = True
        
        for j in range(1, lookback + 1):
            if df['High'].iloc[i] <= df['High'].iloc[i-j] or df['High'].iloc[i] <= df['High'].iloc[i+j]:
                is_swing_high = False
            if df['Low'].iloc[i] >= df['Low'].iloc[i-j] or df['Low'].iloc[i] >= df['Low'].iloc[i+j]:
                is_swing_low = False
        
        swing_highs.append(df['High'].iloc[i] if is_swing_high else np.nan)
        swing_lows.append(df['Low'].iloc[i] if is_swing_low else np.nan)
    
    # Pad for alignment
    swing_highs = [np.nan] * lookback + swing_highs + [np.nan] * lookback
    swing_lows = [np.nan] * lookback + swing_lows + [np.nan] * lookback
    
    return pd.Series(swing_highs, index=df.index), pd.Series(swing_lows, index=df.index)

# Add indicators
def add_indicators(df, params):
    # Set default EMA values if not present
    ema_fast = params.get('ema_fast', 9)
    ema_slow = params.get('ema_slow', 15)
    
    df['EMA_Fast'] = calculate_ema(df['Close'], ema_fast)
    df['EMA_Slow'] = calculate_ema(df['Close'], ema_slow)
    
    if params.get('use_adx', False):
        df['ADX'], df['Plus_DI'], df['Minus_DI'] = calculate_adx(df, params.get('adx_period', 14))
    
    df['ATR'] = calculate_atr(df, 14)
    df['Swing_High'], df['Swing_Low'] = find_swing_points(df)
    
    return df

# Generate signals for EMA Crossover
def generate_ema_crossover_signal(df, idx, params):
    if idx < 1:
        return 0, {}
    
    ema_fast_curr = df['EMA_Fast'].iloc[idx]
    ema_slow_curr = df['EMA_Slow'].iloc[idx]
    ema_fast_prev = df['EMA_Fast'].iloc[idx-1]
    ema_slow_prev = df['EMA_Slow'].iloc[idx-1]
    
    # Check crossover
    bullish_cross = ema_fast_prev <= ema_slow_prev and ema_fast_curr > ema_slow_curr
    bearish_cross = ema_fast_prev >= ema_slow_prev and ema_fast_curr < ema_slow_curr
    
    if not bullish_cross and not bearish_cross:
        return 0, {}
    
    # Check EMA angle
    min_angle = params.get('min_ema_angle', 1)
    ema_angle = calculate_ema_angle(df['EMA_Fast'].iloc[:idx+1])
    
    if bullish_cross and ema_angle < min_angle:
        return 0, {}
    if bearish_cross and abs(ema_angle) < min_angle:
        return 0, {}
    
    # ADX filter
    if params.get('use_adx', False):
        adx_value = df['ADX'].iloc[idx]
        min_adx = params.get('min_adx', 20)
        if pd.isna(adx_value) or adx_value < min_adx:
            return 0, {}
    
    # Entry filters
    entry_filter = params.get('entry_filter', 'simple_crossover')
    candle_size = df['Close'].iloc[idx] - df['Open'].iloc[idx]
    
    if entry_filter == 'strong_candle':
        min_points = params.get('strong_candle_points', 10)
        if bullish_cross and candle_size < min_points:
            return 0, {}
        if bearish_cross and candle_size > -min_points:
            return 0, {}
    
    elif entry_filter == 'atr_based':
        atr_value = df['ATR'].iloc[idx]
        atr_multiplier = params.get('atr_multiplier', 0.5)
        if abs(candle_size) < atr_value * atr_multiplier:
            return 0, {}
    
    signal = 1 if bullish_cross else -1
    signal_data = {
        'ema_angle': ema_angle,
        'adx': df['ADX'].iloc[idx] if params.get('use_adx', False) else None
    }
    
    return signal, signal_data

# Calculate Stop Loss
def calculate_stop_loss(df, idx, signal, params):
    sl_type = params['sl_type']
    current_price = df['Close'].iloc[idx]
    
    if sl_type == 'custom_points':
        sl_points = params.get('sl_points', 5)
        if signal == 1:
            return current_price - sl_points
        else:
            return current_price + sl_points
    
    elif sl_type == 'trailing_points':
        sl_points = params.get('sl_points', 5)
        if signal == 1:
            return current_price - sl_points
        else:
            return current_price + sl_points
    
    elif sl_type == 'trailing_current_candle':
        sl_points = params.get('sl_points', 5)
        if signal == 1:
            return df['Low'].iloc[idx] - sl_points
        else:
            return df['High'].iloc[idx] + sl_points
    
    elif sl_type == 'trailing_previous_candle':
        sl_points = params.get('sl_points', 5)
        if idx > 0:
            if signal == 1:
                return df['Low'].iloc[idx-1] - sl_points
            else:
                return df['High'].iloc[idx-1] + sl_points
        else:
            if signal == 1:
                return current_price - sl_points
            else:
                return current_price + sl_points
    
    elif sl_type == 'trailing_current_swing':
        if signal == 1:
            swing_low = df['Swing_Low'].iloc[:idx+1].dropna()
            if len(swing_low) > 0:
                return swing_low.iloc[-1]
            else:
                return current_price - params.get('sl_points', 5)
        else:
            swing_high = df['Swing_High'].iloc[:idx+1].dropna()
            if len(swing_high) > 0:
                return swing_high.iloc[-1]
            else:
                return current_price + params.get('sl_points', 5)
    
    elif sl_type == 'trailing_previous_swing':
        if signal == 1:
            swing_low = df['Swing_Low'].iloc[:idx].dropna()
            if len(swing_low) > 0:
                return swing_low.iloc[-1]
            else:
                return current_price - params.get('sl_points', 5)
        else:
            swing_high = df['Swing_High'].iloc[:idx].dropna()
            if len(swing_high) > 0:
                return swing_high.iloc[-1]
            else:
                return current_price + params.get('sl_points', 5)
    
    elif sl_type == 'trailing_signal_based':
        sl_points = params.get('sl_points', 5)
        if signal == 1:
            return current_price - sl_points
        else:
            return current_price + sl_points
    
    elif sl_type == 'atr_based':
        atr_value = df['ATR'].iloc[idx]
        atr_multiplier = params.get('atr_multiplier', 2)
        if signal == 1:
            return current_price - (atr_value * atr_multiplier)
        else:
            return current_price + (atr_value * atr_multiplier)
    
    elif sl_type == 'current_candle':
        if signal == 1:
            return df['Low'].iloc[idx]
        else:
            return df['High'].iloc[idx]
    
    elif sl_type == 'previous_candle':
        if idx > 0:
            if signal == 1:
                return df['Low'].iloc[idx-1]
            else:
                return df['High'].iloc[idx-1]
        else:
            if signal == 1:
                return df['Low'].iloc[idx]
            else:
                return df['High'].iloc[idx]
    
    elif sl_type == 'current_swing':
        if signal == 1:
            swing_low = df['Swing_Low'].iloc[:idx+1].dropna()
            if len(swing_low) > 0:
                return swing_low.iloc[-1]
            else:
                return df['Low'].iloc[idx]
        else:
            swing_high = df['Swing_High'].iloc[:idx+1].dropna()
            if len(swing_high) > 0:
                return swing_high.iloc[-1]
            else:
                return df['High'].iloc[idx]
    
    elif sl_type == 'previous_swing':
        if signal == 1:
            swing_low = df['Swing_Low'].iloc[:idx].dropna()
            if len(swing_low) > 0:
                return swing_low.iloc[-1]
            else:
                return df['Low'].iloc[idx]
        else:
            swing_high = df['Swing_High'].iloc[:idx].dropna()
            if len(swing_high) > 0:
                return swing_high.iloc[-1]
            else:
                return df['High'].iloc[idx]
    
    elif sl_type == 'signal_based':
        return None
    
    return None

# Calculate Target
def calculate_target(df, idx, signal, params):
    target_type = params['target_type']
    current_price = df['Close'].iloc[idx]
    
    if target_type == 'custom_points':
        target_points = params.get('target_points', 2)
        if signal == 1:
            return current_price + target_points
        else:
            return current_price - target_points
    
    elif target_type == 'trailing_points':
        target_points = params.get('target_points', 2)
        if signal == 1:
            return current_price + target_points
        else:
            return current_price - target_points
    
    elif target_type == 'current_candle':
        if signal == 1:
            return df['High'].iloc[idx]
        else:
            return df['Low'].iloc[idx]
    
    elif target_type == 'previous_candle':
        if idx > 0:
            if signal == 1:
                return df['High'].iloc[idx-1]
            else:
                return df['Low'].iloc[idx-1]
        else:
            if signal == 1:
                return df['High'].iloc[idx]
            else:
                return df['Low'].iloc[idx]
    
    elif target_type == 'current_swing':
        if signal == 1:
            swing_high = df['Swing_High'].iloc[:idx+1].dropna()
            if len(swing_high) > 0:
                return swing_high.iloc[-1]
            else:
                return current_price + params.get('target_points', 2)
        else:
            swing_low = df['Swing_Low'].iloc[:idx+1].dropna()
            if len(swing_low) > 0:
                return swing_low.iloc[-1]
            else:
                return current_price - params.get('target_points', 2)
    
    elif target_type == 'previous_swing':
        if signal == 1:
            swing_high = df['Swing_High'].iloc[:idx].dropna()
            if len(swing_high) > 0:
                return swing_high.iloc[-1]
            else:
                return current_price + params.get('target_points', 2)
        else:
            swing_low = df['Swing_Low'].iloc[:idx].dropna()
            if len(swing_low) > 0:
                return swing_low.iloc[-1]
            else:
                return current_price - params.get('target_points', 2)
    
    elif target_type == 'trailing_signal_based':
        target_points = params.get('target_points', 2)
        if signal == 1:
            return current_price + target_points
        else:
            return current_price - target_points
    
    elif target_type == 'atr_based':
        atr_value = df['ATR'].iloc[idx]
        atr_multiplier = params.get('target_atr_multiplier', 3)
        if signal == 1:
            return current_price + (atr_value * atr_multiplier)
        else:
            return current_price - (atr_value * atr_multiplier)
    
    elif target_type == 'risk_reward':
        entry = st.session_state.entry_price
        sl = st.session_state.stop_loss
        rr_ratio = params.get('risk_reward_ratio', 2)
        risk = abs(entry - sl)
        if signal == 1:
            return entry + (risk * rr_ratio)
        else:
            return entry - (risk * rr_ratio)
    
    elif target_type == 'signal_based':
        return None
    
    return None

# Update trailing stop loss
def update_trailing_sl(df, idx, params):
    if not st.session_state.in_position:
        return st.session_state.stop_loss
    
    sl_type = params['sl_type']
    if 'trailing' not in sl_type:
        return st.session_state.stop_loss
    
    current_price = df['Close'].iloc[idx]
    position_type = st.session_state.position_type
    trailing_threshold = params.get('trailing_threshold', 0)
    
    # Update highest/lowest
    if position_type == 1:
        if current_price > st.session_state.trailing_sl_highest:
            st.session_state.trailing_sl_highest = current_price
    else:
        if st.session_state.trailing_sl_lowest == 0 or current_price < st.session_state.trailing_sl_lowest:
            st.session_state.trailing_sl_lowest = current_price
    
    # Check if threshold is met
    if position_type == 1:
        price_movement = st.session_state.trailing_sl_highest - st.session_state.entry_price
    else:
        price_movement = st.session_state.entry_price - st.session_state.trailing_sl_lowest
    
    if price_movement < trailing_threshold:
        return st.session_state.stop_loss
    
    # Update SL based on type
    new_sl = st.session_state.stop_loss
    
    if sl_type == 'trailing_points':
        sl_points = params.get('sl_points', 5)
        if position_type == 1:
            new_sl = current_price - sl_points
            new_sl = max(new_sl, st.session_state.stop_loss)
        else:
            new_sl = current_price + sl_points
            new_sl = min(new_sl, st.session_state.stop_loss)
    
    elif sl_type == 'trailing_current_candle':
        sl_points = params.get('sl_points', 5)
        if position_type == 1:
            new_sl = df['Low'].iloc[idx] - sl_points
            new_sl = max(new_sl, st.session_state.stop_loss)
        else:
            new_sl = df['High'].iloc[idx] + sl_points
            new_sl = min(new_sl, st.session_state.stop_loss)
    
    elif sl_type == 'trailing_previous_candle':
        sl_points = params.get('sl_points', 5)
        if idx > 0:
            if position_type == 1:
                new_sl = df['Low'].iloc[idx-1] - sl_points
                new_sl = max(new_sl, st.session_state.stop_loss)
            else:
                new_sl = df['High'].iloc[idx-1] + sl_points
                new_sl = min(new_sl, st.session_state.stop_loss)
    
    elif sl_type == 'trailing_current_swing':
        if position_type == 1:
            swing_low = df['Swing_Low'].iloc[:idx+1].dropna()
            if len(swing_low) > 0:
                new_sl = swing_low.iloc[-1]
                new_sl = max(new_sl, st.session_state.stop_loss)
        else:
            swing_high = df['Swing_High'].iloc[:idx+1].dropna()
            if len(swing_high) > 0:
                new_sl = swing_high.iloc[-1]
                new_sl = min(new_sl, st.session_state.stop_loss)
    
    elif sl_type == 'trailing_previous_swing':
        if position_type == 1:
            swing_low = df['Swing_Low'].iloc[:idx].dropna()
            if len(swing_low) > 0:
                new_sl = swing_low.iloc[-1]
                new_sl = max(new_sl, st.session_state.stop_loss)
        else:
            swing_high = df['Swing_High'].iloc[:idx].dropna()
            if len(swing_high) > 0:
                new_sl = swing_high.iloc[-1]
                new_sl = min(new_sl, st.session_state.stop_loss)
    
    return new_sl

# Update trailing target
def update_trailing_target(df, idx, params):
    if not st.session_state.in_position:
        return st.session_state.target
    
    target_type = params['target_type']
    if 'trailing' not in target_type:
        return st.session_state.target
    
    current_price = df['Close'].iloc[idx]
    position_type = st.session_state.position_type
    
    new_target = st.session_state.target
    
    if target_type == 'trailing_points':
        target_points = params.get('target_points', 2)
        if position_type == 1:
            new_target = current_price + target_points
            new_target = max(new_target, st.session_state.target)
        else:
            new_target = current_price - target_points
            new_target = min(new_target, st.session_state.target)
    
    return new_target

# Check signal-based exit
def check_signal_based_exit(df, idx, params):
    if not st.session_state.in_position:
        return False
    
    sl_type = params['sl_type']
    target_type = params['target_type']
    
    if sl_type != 'signal_based' and sl_type != 'trailing_signal_based' and target_type != 'signal_based' and target_type != 'trailing_signal_based':
        return False
    
    if idx < 1:
        return False
    
    ema_fast_curr = df['EMA_Fast'].iloc[idx]
    ema_slow_curr = df['EMA_Slow'].iloc[idx]
    
    position_type = st.session_state.position_type
    
    if position_type == 1:
        if ema_fast_curr < ema_slow_curr:
            return True
    else:
        if ema_fast_curr > ema_slow_curr:
            return True
    
    return False

# Execute entry
def execute_entry(df, idx, signal, params):
    entry_price = df['Close'].iloc[idx]
    stop_loss = calculate_stop_loss(df, idx, signal, params)
    target = calculate_target(df, idx, signal, params)
    
    st.session_state.in_position = True
    st.session_state.position_type = signal
    st.session_state.entry_price = entry_price
    st.session_state.stop_loss = stop_loss
    st.session_state.target = target
    st.session_state.entry_idx = idx
    st.session_state.entry_time = df.index[idx]
    st.session_state.trailing_sl_highest = entry_price
    st.session_state.trailing_sl_lowest = entry_price
    
    position_name = "LONG" if signal == 1 else "SHORT"
    log_entry = f"ENTRY: {position_name} | Time: {df.index[idx]} | Price: {entry_price:.2f} | SL: {stop_loss if stop_loss else 'Signal Based'} | Target: {target if target else 'Signal Based'}"
    st.session_state.live_logs.append(log_entry)
    if len(st.session_state.live_logs) > 100:
        st.session_state.live_logs = st.session_state.live_logs[-100:]

# Execute exit
def execute_exit(df, idx, exit_price, exit_reason, params):
    pnl = 0
    if st.session_state.position_type == 1:
        pnl = exit_price - st.session_state.entry_price
    else:
        pnl = st.session_state.entry_price - exit_price
    
    trade_record = {
        'Entry Time': st.session_state.entry_time,
        'Exit Time': df.index[idx],
        'Type': 'LONG' if st.session_state.position_type == 1 else 'SHORT',
        'Entry Price': st.session_state.entry_price,
        'Exit Price': exit_price,
        'PnL': pnl,
        'Exit Reason': exit_reason
    }
    st.session_state.live_trades.append(trade_record)
    
    log_entry = f"EXIT: Time: {df.index[idx]} | Price: {exit_price:.2f} | PnL: {pnl:.2f} | Reason: {exit_reason}"
    st.session_state.live_logs.append(log_entry)
    if len(st.session_state.live_logs) > 100:
        st.session_state.live_logs = st.session_state.live_logs[-100:]
    
    reset_position_state()

# Process live tick
def process_live_tick(df, idx, params):
    if st.session_state.in_position:
        st.session_state.stop_loss = update_trailing_sl(df, idx, params)
        st.session_state.target = update_trailing_target(df, idx, params)
        
        current_price = df['Close'].iloc[idx]
        
        if check_signal_based_exit(df, idx, params):
            execute_exit(df, idx, current_price, "Signal Based Exit", params)
            return
        
        if st.session_state.stop_loss is not None:
            if st.session_state.position_type == 1:
                if current_price <= st.session_state.stop_loss:
                    execute_exit(df, idx, st.session_state.stop_loss, "Stop Loss Hit", params)
                    return
            else:
                if current_price >= st.session_state.stop_loss:
                    execute_exit(df, idx, st.session_state.stop_loss, "Stop Loss Hit", params)
                    return
        
        if st.session_state.target is not None:
            if st.session_state.position_type == 1:
                if current_price >= st.session_state.target:
                    execute_exit(df, idx, st.session_state.target, "Target Hit", params)
                    return
            else:
                if current_price <= st.session_state.target:
                    execute_exit(df, idx, st.session_state.target, "Target Hit", params)
                    return
    else:
        strategy_type = params['strategy_type']
        signal = 0
        
        if strategy_type == 'ema_crossover':
            signal, _ = generate_ema_crossover_signal(df, idx, params)
        elif strategy_type == 'simple_buy':
            signal = 1
        elif strategy_type == 'simple_sell':
            signal = -1
        
        if signal != 0:
            execute_entry(df, idx, signal, params)

# Create candlestick chart
def create_candlestick_chart(df, params):
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    if params['strategy_type'] == 'ema_crossover':
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
            line=dict(color='red', width=1)
        ))
    
    if st.session_state.in_position:
        fig.add_hline(
            y=st.session_state.entry_price,
            line_dash="dash",
            line_color="white",
            annotation_text="Entry",
            annotation_position="right"
        )
        
        if st.session_state.stop_loss is not None:
            fig.add_hline(
                y=st.session_state.stop_loss,
                line_dash="dash",
                line_color="red",
                annotation_text="SL",
                annotation_position="right"
            )
        
        if st.session_state.target is not None:
            fig.add_hline(
                y=st.session_state.target,
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

# Main app
def main():
    init_session_state()
    
    st.title("🚀 Professional Quantitative Trading System")
    
    st.sidebar.header("Configuration")
    
    asset_type = st.sidebar.selectbox("Asset Type", ["Predefined", "Custom Ticker"])
    
    if asset_type == "Predefined":
        asset_name = st.sidebar.selectbox("Select Asset", list(ASSET_MAPPING.keys()))
        ticker = ASSET_MAPPING[asset_name]
    else:
        ticker = st.sidebar.text_input("Enter Ticker", value="AAPL")
    
    interval = st.sidebar.selectbox("Interval", list(INTERVAL_PERIODS.keys()))
    allowed_periods = INTERVAL_PERIODS[interval]
    period = st.sidebar.selectbox("Period", allowed_periods)
    
    st.sidebar.header("Strategy Configuration")
    strategy_type = st.sidebar.selectbox(
        "Strategy Type",
        ["ema_crossover", "simple_buy", "simple_sell"]
    )
    
    strategy_params = {
        'strategy_type': strategy_type,
        'ema_fast': 9,
        'ema_slow': 15
    }
    
    if strategy_type == 'ema_crossover':
        strategy_params['ema_fast'] = st.sidebar.number_input("EMA Fast", min_value=1, value=9)
        strategy_params['ema_slow'] = st.sidebar.number_input("EMA Slow", min_value=1, value=15)
        strategy_params['min_ema_angle'] = st.sidebar.number_input("Min EMA Angle (degrees)", min_value=0.0, value=1.0, step=0.1)
        
        strategy_params['use_adx'] = st.sidebar.checkbox("Use ADX Filter", value=False)
        if strategy_params['use_adx']:
            strategy_params['min_adx'] = st.sidebar.number_input("Min ADX Value", min_value=1, value=20)
            strategy_params['adx_period'] = st.sidebar.number_input("ADX Period", min_value=1, value=14)
        
        strategy_params['entry_filter'] = st.sidebar.selectbox(
            "Entry Filter",
            ["simple_crossover", "strong_candle", "atr_based"]
        )
        
        if strategy_params['entry_filter'] == 'strong_candle':
            strategy_params['strong_candle_points'] = st.sidebar.number_input("Strong Candle Points", min_value=1, value=10)
        elif strategy_params['entry_filter'] == 'atr_based':
            strategy_params['atr_multiplier'] = st.sidebar.number_input("ATR Multiplier", min_value=0.1, value=0.5, step=0.1)
    
    st.sidebar.header("Stop Loss Configuration")
    sl_type = st.sidebar.selectbox(
        "SL Type",
        [
            "custom_points",
            "trailing_points",
            "trailing_current_candle",
            "trailing_previous_candle",
            "trailing_current_swing",
            "trailing_previous_swing",
            "trailing_signal_based",
            "atr_based",
            "current_candle",
            "previous_candle",
            "current_swing",
            "previous_swing",
            "signal_based"
        ]
    )
    strategy_params['sl_type'] = sl_type
    
    if 'trailing' in sl_type:
        strategy_params['trailing_threshold'] = st.sidebar.number_input(
            "Trailing Threshold (Points)",
            min_value=0,
            value=0,
            help="SL updates only after price moves by this amount"
        )
    
    if sl_type in ['custom_points', 'trailing_points', 'trailing_current_candle', 'trailing_previous_candle', 'trailing_current_swing', 'trailing_previous_swing', 'trailing_signal_based']:
        strategy_params['sl_points'] = st.sidebar.number_input("SL Points", min_value=1, value=5)
    
    if sl_type == 'atr_based':
        strategy_params['atr_multiplier'] = st.sidebar.number_input("ATR Multiplier (SL)", min_value=0.1, value=2.0, step=0.1)
    
    st.sidebar.header("Target Configuration")
    target_type = st.sidebar.selectbox(
        "Target Type",
        [
            "custom_points",
            "trailing_points",
            "current_candle",
            "previous_candle",
            "current_swing",
            "previous_swing",
            "trailing_signal_based",
            "atr_based",
            "risk_reward",
            "signal_based"
        ]
    )
    strategy_params['target_type'] = target_type
    
    if target_type in ['custom_points', 'trailing_points']:
        strategy_params['target_points'] = st.sidebar.number_input("Target Points", min_value=1, value=2)
    
    if target_type == 'atr_based':
        strategy_params['target_atr_multiplier'] = st.sidebar.number_input("ATR Multiplier (Target)", min_value=0.1, value=3.0, step=0.1)
    
    if target_type == 'risk_reward':
        strategy_params['risk_reward_ratio'] = st.sidebar.number_input("Risk:Reward Ratio", min_value=0.1, value=2.0, step=0.1)
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if st.button("🚀 Start Trading", use_container_width=True):
            if not st.session_state.live_running:
                with st.spinner("Fetching data..."):
                    df = fetch_data_with_delay(ticker, interval, period)
                    if df is not None and not df.empty:
                        df = add_indicators(df, strategy_params)
                        st.session_state.live_data = df
                        st.session_state.live_running = True
                        st.session_state.last_fetch_time = time.time()
                        reset_position_state()
                        st.success("Trading started!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Failed to fetch data")
    
    with col2:
        if st.button("⏹️ Stop Trading", use_container_width=True):
            if st.session_state.live_running:
                if st.session_state.in_position and st.session_state.live_data is not None:
                    df = st.session_state.live_data
                    current_price = df['Close'].iloc[-1]
                    execute_exit(df, len(df)-1, current_price, "Manual Exit - Trading Stopped", strategy_params)
                
                st.session_state.live_running = False
                st.success("Trading stopped!")
                time.sleep(0.5)
                st.rerun()
    
    if st.session_state.live_running:
        current_time = time.time()
        time_since_last_fetch = current_time - st.session_state.last_fetch_time if st.session_state.last_fetch_time else 999
        
        if time_since_last_fetch >= 1.0:
            df = fetch_data_with_delay(ticker, interval, period)
            if df is not None and not df.empty:
                df = add_indicators(df, strategy_params)
                st.session_state.live_data = df
                st.session_state.last_fetch_time = current_time
                
                process_live_tick(df, len(df)-1, strategy_params)
            
            time.sleep(0.1)
            st.rerun()
    
    tab1, tab2, tab3 = st.tabs(["📊 Live Dashboard", "📈 Trade History", "📝 Trade Logs"])
    
    with tab1:
        if st.session_state.live_data is not None:
            df = st.session_state.live_data
            
            # Live indicator values at top
            st.subheader("📊 Live Market Data")
            live_col1, live_col2, live_col3, live_col4, live_col5 = st.columns(5)
            
            current_price = df['Close'].iloc[-1]
            ema_fast_val = df['EMA_Fast'].iloc[-1]
            ema_slow_val = df['EMA_Slow'].iloc[-1]
            atr_val = df['ATR'].iloc[-1]
            
            with live_col1:
                st.metric("Current Price", f"{current_price:.2f}")
            with live_col2:
                st.metric(f"EMA {strategy_params.get('ema_fast', 9)}", f"{ema_fast_val:.2f}")
            with live_col3:
                st.metric(f"EMA {strategy_params.get('ema_slow', 15)}", f"{ema_slow_val:.2f}")
            with live_col4:
                st.metric("ATR", f"{atr_val:.2f}")
            with live_col5:
                if strategy_params.get('use_adx', False):
                    adx_val = df['ADX'].iloc[-1]
                    st.metric("ADX", f"{adx_val:.2f}" if not pd.isna(adx_val) else "N/A")
                else:
                    st.metric("Volume", f"{int(df['Volume'].iloc[-1]):,}")
            
            # Auto-refresh indicator
            if st.session_state.live_running:
                time_since_fetch = time.time() - st.session_state.last_fetch_time if st.session_state.last_fetch_time else 0
                st.caption(f"🔄 Auto-refreshing... Last update: {time_since_fetch:.1f}s ago")
            
            chart_key = f"chart_{int(time.time() * 1000)}"
            fig = create_candlestick_chart(df, strategy_params)
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
            
            st.subheader("⚙️ Active Strategy Configuration")
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                st.write("**Strategy Settings:**")
                st.write(f"• Type: {strategy_params['strategy_type'].replace('_', ' ').title()}")
                if strategy_params['strategy_type'] == 'ema_crossover':
                    st.write(f"• EMA Fast: {strategy_params.get('ema_fast', 9)}")
                    st.write(f"• EMA Slow: {strategy_params.get('ema_slow', 15)}")
                    st.write(f"• Min Angle: {strategy_params.get('min_ema_angle', 1.0)}°")
                    if strategy_params.get('use_adx', False):
                        st.write(f"• ADX Filter: ON (Min: {strategy_params.get('min_adx', 20)})")
                    st.write(f"• Entry Filter: {strategy_params.get('entry_filter', 'simple_crossover').replace('_', ' ').title()}")
            
            with config_col2:
                st.write("**Risk Management:**")
                st.write(f"• SL Type: {strategy_params['sl_type'].replace('_', ' ').title()}")
                if strategy_params['sl_type'] in ['custom_points', 'trailing_points']:
                    st.write(f"• SL Points: {strategy_params.get('sl_points', 5)}")
                if 'trailing' in strategy_params['sl_type']:
                    st.write(f"• Trailing Threshold: {strategy_params.get('trailing_threshold', 0)} pts")
                st.write(f"• Target Type: {strategy_params['target_type'].replace('_', ' ').title()}")
                if strategy_params['target_type'] in ['custom_points', 'trailing_points']:
                    st.write(f"• Target Points: {strategy_params.get('target_points', 2)}")
            
            st.subheader("📍 Position Status")
            info_col1, info_col2, info_col3 = st.columns(3)
            
            with info_col1:
                if st.session_state.in_position:
                    position_name = "LONG" if st.session_state.position_type == 1 else "SHORT"
                    st.metric("Current Position", position_name)
                    
                    current_price = df['Close'].iloc[-1]
                    unrealized_pnl = 0
                    if st.session_state.position_type == 1:
                        unrealized_pnl = current_price - st.session_state.entry_price
                    else:
                        unrealized_pnl = st.session_state.entry_price - current_price
                    
                    st.metric("Unrealized P&L", f"{unrealized_pnl:.2f}", delta=f"{unrealized_pnl:.2f}")
                else:
                    st.metric("Current Position", "NONE")
                    
                    if strategy_params['strategy_type'] == 'ema_crossover':
                        signal, _ = generate_ema_crossover_signal(df, len(df)-1, strategy_params)
                        signal_text = "BUY" if signal == 1 else ("SELL" if signal == -1 else "NONE")
                        signal_color = "🟢" if signal == 1 else ("🔴" if signal == -1 else "⚪")
                        st.metric("Current Signal", f"{signal_color} {signal_text}")
            
            with info_col2:
                if st.session_state.in_position:
                    st.write("**Entry Price:**")
                    st.write(f"{st.session_state.entry_price:.2f}")
                    st.write("**Entry Time:**")
                    st.write(f"{st.session_state.entry_time}")
                else:
                    st.write("**Market Status:**")
                    st.write("Waiting for signal...")
                    st.write("**Last Candle:**")
                    st.write(f"{df.index[-1]}")
            
            with info_col3:
                if st.session_state.in_position:
                    if st.session_state.stop_loss is not None:
                        st.write("**Stop Loss:**")
                        st.write(f"{st.session_state.stop_loss:.2f}")
                        dist_to_sl = abs(current_price - st.session_state.stop_loss)
                        st.caption(f"Distance: {dist_to_sl:.2f} pts")
                    else:
                        st.write("**Stop Loss:**")
                        st.write("Signal Based")
                    
                    if st.session_state.target is not None:
                        st.write("**Target:**")
                        st.write(f"{st.session_state.target:.2f}")
                        dist_to_target = abs(current_price - st.session_state.target)
                        st.caption(f"Distance: {dist_to_target:.2f} pts")
                    else:
                        st.write("**Target:**")
                        st.write("Signal Based")
                else:
                    # Show trade statistics
                    if len(st.session_state.live_trades) > 0:
                        trades_df = pd.DataFrame(st.session_state.live_trades)
                        total_pnl = trades_df['PnL'].sum()
                        st.write("**Total P&L:**")
                        st.write(f"{total_pnl:.2f}")
                        st.write("**Total Trades:**")
                        st.write(f"{len(trades_df)}")
        else:
            st.info("👆 Click 'Start Trading' to begin live trading simulation")
            st.write("")
            st.write("**What you'll see once trading starts:**")
            st.write("• 📊 Live price chart with EMA indicators")
            st.write("• 📈 Real-time EMA values and market data")
            st.write("• ⚙️ Your active strategy configuration")
            st.write("• 📍 Current position status and P&L")
            st.write("• 🔄 Auto-refresh every 1-1.5 seconds")
            st.write("")
            st.write("**Configure your strategy in the sidebar:**")
            st.write("1. Select asset and timeframe")
            st.write("2. Choose strategy type")
            st.write("3. Set stop loss and target parameters")
            st.write("4. Click 'Start Trading' to begin!")
    
    with tab2:
        st.subheader("Trade History")
        
        if len(st.session_state.live_trades) > 0:
            trades_df = pd.DataFrame(st.session_state.live_trades)
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['PnL'] > 0])
            accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = trades_df['PnL'].sum()
            
            with metric_col1:
                st.metric("Total Trades", total_trades)
            with metric_col2:
                st.metric("Winning Trades", winning_trades)
            with metric_col3:
                st.metric("Accuracy", f"{accuracy:.2f}%")
            with metric_col4:
                st.metric("Total PnL", f"{total_pnl:.2f}")
            
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("No trades executed yet")
    
    with tab3:
        st.subheader("Trade Logs")
        
        if len(st.session_state.live_logs) > 0:
            for log in reversed(st.session_state.live_logs):
                st.text(log)
        else:
            st.info("No trade logs yet")

if __name__ == "__main__":
    main()
