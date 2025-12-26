import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import time
import random

# Page configuration
st.set_page_config(page_title="Live Trading System", layout="wide", initial_sidebar_state="expanded")

# IST timezone
IST = pytz.timezone('Asia/Kolkata')

def get_ist_time():
    """Get current time in IST"""
    return datetime.now(IST)

# Initialize session state
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
    if 'trailing_sl_highest' not in st.session_state:
        st.session_state.trailing_sl_highest = 0
    if 'trailing_sl_lowest' not in st.session_state:
        st.session_state.trailing_sl_lowest = 0
    if 'quantity' not in st.session_state:
        st.session_state.quantity = 1
    if 'last_fetch_time' not in st.session_state:
        st.session_state.last_fetch_time = None
    if 'iteration_count' not in st.session_state:
        st.session_state.iteration_count = 0
    if 'potential_pnl_sl' not in st.session_state:
        st.session_state.potential_pnl_sl = 0
    if 'potential_pnl_target' not in st.session_state:
        st.session_state.potential_pnl_target = 0

init_session_state()

# Asset mappings
ASSET_MAPPINGS = {
    'NIFTY 50': '^NSEI',
    'BANKNIFTY': '^NSEBANK',
    'SENSEX': '^BSESN',
    'BTC-USD': 'BTC-USD',
    'ETH-USD': 'ETH-USD',
    'USDINR': 'USDINR=X',
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'Gold': 'GC=F',
    'Silver': 'SI=F'
}

# Timeframe validation
VALID_PERIODS = {
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

def add_log(message):
    """Add timestamped log entry"""
    timestamp = get_ist_time().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    st.session_state.live_logs.append(log_entry)
    if len(st.session_state.live_logs) > 100:
        st.session_state.live_logs = st.session_state.live_logs[-100:]

def fetch_data_with_retry(ticker, interval, period, max_retries=3):
    """Fetch data from yfinance with retry logic and rate limiting"""
    for attempt in range(max_retries):
        try:
            delay = random.uniform(1.0, 1.5)
            time.sleep(delay)
            
            data = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=False)
            
            if data.empty:
                add_log(f"No data returned for {ticker}")
                return None
            
            # Handle multi-index columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # Select OHLCV columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            data = data[required_cols].copy()
            
            # Handle timezone
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC').tz_convert(IST)
            else:
                data.index = data.index.tz_convert(IST)
            
            data = data.reset_index()
            data.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            add_log(f"Data fetched successfully: {len(data)} candles")
            return data
            
        except Exception as e:
            add_log(f"Fetch attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)
            else:
                add_log(f"Failed to fetch data after {max_retries} attempts")
                return None
    
    return None

def calculate_ema(data, period):
    """Calculate EMA manually"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    """Calculate ATR manually"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    
    return atr

def calculate_ema_angle(df, fast_period, window=3):
    """Calculate EMA crossover angle in degrees"""
    if len(df) < window + 1:
        return 0
    
    ema_fast = df['EMA_Fast'].iloc[-window:]
    ema_slow = df['EMA_Slow'].iloc[-window:]
    
    # Calculate slopes
    x = np.arange(window)
    fast_slope = np.polyfit(x, ema_fast, 1)[0]
    slow_slope = np.polyfit(x, ema_slow, 1)[0]
    
    # Relative slope
    relative_slope = fast_slope - slow_slope
    
    # Convert to degrees
    angle = np.arctan(relative_slope) * (180 / np.pi)
    
    return angle

def detect_swing_points(df, window=5):
    """Detect swing highs and lows"""
    swing_highs = []
    swing_lows = []
    
    for i in range(window, len(df) - window):
        # Swing high
        is_high = True
        for j in range(1, window + 1):
            if df['High'].iloc[i] <= df['High'].iloc[i - j] or df['High'].iloc[i] <= df['High'].iloc[i + j]:
                is_high = False
                break
        if is_high:
            swing_highs.append((i, df['High'].iloc[i]))
        
        # Swing low
        is_low = True
        for j in range(1, window + 1):
            if df['Low'].iloc[i] >= df['Low'].iloc[i - j] or df['Low'].iloc[i] >= df['Low'].iloc[i + j]:
                is_low = False
                break
        if is_low:
            swing_lows.append((i, df['Low'].iloc[i]))
    
    return swing_highs, swing_lows

def calculate_stop_loss(df, i, entry_price, position_type, sl_type, sl_value, atr_multiplier=2):
    """Calculate stop loss based on selected type"""
    if sl_type == 'Custom Points':
        if position_type == 1:
            sl = entry_price - sl_value
        else:
            sl = entry_price + sl_value
    
    elif sl_type == 'ATR Based':
        atr = df['ATR'].iloc[i]
        if position_type == 1:
            sl = entry_price - (atr * atr_multiplier)
        else:
            sl = entry_price + (atr * atr_multiplier)
    
    elif sl_type == 'Current Candle Low/High':
        if position_type == 1:
            sl = df['Low'].iloc[i]
        else:
            sl = df['High'].iloc[i]
    
    elif sl_type == 'Previous Candle Low/High':
        if i > 0:
            if position_type == 1:
                sl = df['Low'].iloc[i - 1]
            else:
                sl = df['High'].iloc[i - 1]
        else:
            sl = entry_price * 0.98 if position_type == 1 else entry_price * 1.02
    
    elif sl_type == 'Current Swing Low/High':
        swing_highs, swing_lows = detect_swing_points(df[:i+1])
        if position_type == 1 and swing_lows:
            sl = swing_lows[-1][1]
        elif position_type == -1 and swing_highs:
            sl = swing_highs[-1][1]
        else:
            sl = entry_price * 0.98 if position_type == 1 else entry_price * 1.02
    
    elif sl_type == 'Previous Swing Low/High':
        swing_highs, swing_lows = detect_swing_points(df[:i+1])
        if position_type == 1 and len(swing_lows) >= 2:
            sl = swing_lows[-2][1]
        elif position_type == -1 and len(swing_highs) >= 2:
            sl = swing_highs[-2][1]
        else:
            sl = entry_price * 0.98 if position_type == 1 else entry_price * 1.02
    
    elif sl_type == 'Signal Based':
        sl = 0
    
    elif sl_type == 'Cost to Cost':
        sl = entry_price
    
    else:
        if position_type == 1:
            sl = entry_price * 0.98
        else:
            sl = entry_price * 1.02
    
    # Minimum distance check
    if sl != 0:
        min_distance = entry_price * 0.005
        if position_type == 1:
            sl = min(sl, entry_price - min_distance)
        else:
            sl = max(sl, entry_price + min_distance)
    
    return sl

def calculate_target(df, i, entry_price, position_type, target_type, target_value, rr_ratio=2, atr_multiplier=3):
    """Calculate target based on selected type"""
    if target_type == 'Custom Points':
        if position_type == 1:
            target = entry_price + target_value
        else:
            target = entry_price - target_value
    
    elif target_type == 'ATR Based':
        atr = df['ATR'].iloc[i]
        if position_type == 1:
            target = entry_price + (atr * atr_multiplier)
        else:
            target = entry_price - (atr * atr_multiplier)
    
    elif target_type == 'Risk Reward Based':
        risk = abs(entry_price - st.session_state.stop_loss)
        if position_type == 1:
            target = entry_price + (risk * rr_ratio)
        else:
            target = entry_price - (risk * rr_ratio)
    
    elif target_type == 'Signal Based':
        target = 0
    
    elif target_type == 'Cost to Cost':
        target = entry_price
    
    else:
        if position_type == 1:
            target = entry_price * 1.02
        else:
            target = entry_price * 0.98
    
    # Minimum distance check
    if target != 0:
        min_distance = entry_price * 0.01
        if position_type == 1:
            target = max(target, entry_price + min_distance)
        else:
            target = min(target, entry_price - min_distance)
    
    return target

def check_signal_based_exit(df, i, position_type):
    """Check for signal-based exit"""
    if i < 1:
        return False, None
    
    ema_fast_curr = df['EMA_Fast'].iloc[i]
    ema_slow_curr = df['EMA_Slow'].iloc[i]
    ema_fast_prev = df['EMA_Fast'].iloc[i - 1]
    ema_slow_prev = df['EMA_Slow'].iloc[i - 1]
    
    if position_type == 1:
        if ema_fast_curr < ema_slow_curr and ema_fast_prev >= ema_slow_prev:
            return True, "Reverse Signal - Bearish Crossover"
    
    elif position_type == -1:
        if ema_fast_curr > ema_slow_curr and ema_fast_prev <= ema_slow_prev:
            return True, "Reverse Signal - Bullish Crossover"
    
    return False, None

def update_trailing_sl(current_price, position_type, sl_type, trail_percent=1):
    """Update trailing stop loss"""
    if sl_type != 'Trailing SL':
        return st.session_state.stop_loss
    
    if position_type == 1:
        if current_price > st.session_state.trailing_sl_highest:
            st.session_state.trailing_sl_highest = current_price
            new_sl = current_price * (1 - trail_percent / 100)
            if new_sl > st.session_state.stop_loss:
                st.session_state.stop_loss = new_sl
                add_log(f"Trailing SL updated to {new_sl:.2f}")
    
    else:
        if current_price < st.session_state.trailing_sl_lowest or st.session_state.trailing_sl_lowest == 0:
            st.session_state.trailing_sl_lowest = current_price
            new_sl = current_price * (1 + trail_percent / 100)
            if new_sl < st.session_state.stop_loss or st.session_state.stop_loss == 0:
                st.session_state.stop_loss = new_sl
                add_log(f"Trailing SL updated to {new_sl:.2f}")
    
    return st.session_state.stop_loss

def generate_signals(df, strategy_params):
    """Generate trading signals based on strategy"""
    strategy_type = strategy_params['strategy_type']
    
    if strategy_type == 'EMA Crossover':
        fast_period = strategy_params['ema_fast']
        slow_period = strategy_params['ema_slow']
        min_angle = strategy_params['min_angle']
        entry_filter = strategy_params['entry_filter']
        
        df['EMA_Fast'] = calculate_ema(df['Close'], fast_period)
        df['EMA_Slow'] = calculate_ema(df['Close'], slow_period)
        df['ATR'] = calculate_atr(df, 14)
        
        df['Signal'] = 0
        
        for i in range(1, len(df)):
            ema_fast_curr = df['EMA_Fast'].iloc[i]
            ema_slow_curr = df['EMA_Slow'].iloc[i]
            ema_fast_prev = df['EMA_Fast'].iloc[i - 1]
            ema_slow_prev = df['EMA_Slow'].iloc[i - 1]
            
            angle = calculate_ema_angle(df.iloc[:i+1], fast_period)
            
            if ema_fast_curr > ema_slow_curr and ema_fast_prev <= ema_slow_prev:
                if abs(angle) >= min_angle:
                    signal = 1
                    
                    if entry_filter == 'Strong Candle':
                        candle_size = df['Close'].iloc[i] - df['Open'].iloc[i]
                        avg_candle = df['Close'].rolling(20).mean().iloc[i] - df['Open'].rolling(20).mean().iloc[i]
                        if candle_size < avg_candle * 1.5:
                            signal = 0
                    
                    elif entry_filter == 'ATR Based Candle':
                        candle_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
                        atr = df['ATR'].iloc[i]
                        if candle_size < atr * 0.5:
                            signal = 0
                    
                    df.loc[df.index[i], 'Signal'] = signal
            
            elif ema_fast_curr < ema_slow_curr and ema_fast_prev >= ema_slow_prev:
                if abs(angle) >= min_angle:
                    signal = -1
                    
                    if entry_filter == 'Strong Candle':
                        candle_size = df['Open'].iloc[i] - df['Close'].iloc[i]
                        avg_candle = df['Open'].rolling(20).mean().iloc[i] - df['Close'].rolling(20).mean().iloc[i]
                        if candle_size < avg_candle * 1.5:
                            signal = 0
                    
                    elif entry_filter == 'ATR Based Candle':
                        candle_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
                        atr = df['ATR'].iloc[i]
                        if candle_size < atr * 0.5:
                            signal = 0
                    
                    df.loc[df.index[i], 'Signal'] = signal
    
    elif strategy_type == 'Simple Buy':
        df['Signal'] = 1
        df['ATR'] = calculate_atr(df, 14)
    
    elif strategy_type == 'Simple Sell':
        df['Signal'] = -1
        df['ATR'] = calculate_atr(df, 14)
    
    return df

def process_live_candle(df, strategy_params):
    """Process the latest candle for live trading"""
    if len(df) < 2:
        return
    
    i = len(df) - 1
    current_price = df['Close'].iloc[i]
    signal = df['Signal'].iloc[i]
    
    if st.session_state.in_position:
        position_type = st.session_state.position_type
        entry_price = st.session_state.entry_price
        sl_type = strategy_params['sl_type']
        target_type = strategy_params['target_type']
        
        if sl_type == 'Trailing SL':
            update_trailing_sl(current_price, position_type, sl_type, strategy_params.get('trail_percent', 1))
        
        if sl_type == 'Signal Based' or target_type == 'Signal Based':
            should_exit, exit_reason = check_signal_based_exit(df, i, position_type)
            if should_exit:
                exit_price = current_price
                pnl = (exit_price - entry_price) * position_type * st.session_state.quantity
                
                trade = {
                    'Entry Time': df['Datetime'].iloc[st.session_state.entry_idx].strftime('%Y-%m-%d %H:%M:%S'),
                    'Exit Time': df['Datetime'].iloc[i].strftime('%Y-%m-%d %H:%M:%S'),
                    'Type': 'LONG' if position_type == 1 else 'SHORT',
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Stop Loss': st.session_state.stop_loss,
                    'Target': st.session_state.target,
                    'Exit Reason': exit_reason,
                    'PnL': pnl,
                    'Quantity': st.session_state.quantity
                }
                
                st.session_state.live_trades.append(trade)
                add_log(f"Position closed - {exit_reason} | PnL: {pnl:.2f}")
                
                st.session_state.in_position = False
                st.session_state.position_type = 0
                st.session_state.entry_price = 0
                st.session_state.stop_loss = 0
                st.session_state.target = 0
                st.session_state.trailing_sl_highest = 0
                st.session_state.trailing_sl_lowest = 0
                st.session_state.potential_pnl_sl = 0
                st.session_state.potential_pnl_target = 0
                return
        
        if st.session_state.stop_loss != 0:
            if position_type == 1 and current_price <= st.session_state.stop_loss:
                exit_price = st.session_state.stop_loss
                pnl = (exit_price - entry_price) * st.session_state.quantity
                
                trade = {
                    'Entry Time': df['Datetime'].iloc[st.session_state.entry_idx].strftime('%Y-%m-%d %H:%M:%S'),
                    'Exit Time': df['Datetime'].iloc[i].strftime('%Y-%m-%d %H:%M:%S'),
                    'Type': 'LONG',
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Stop Loss': st.session_state.stop_loss,
                    'Target': st.session_state.target,
                    'Exit Reason': 'Stop Loss Hit',
                    'PnL': pnl,
                    'Quantity': st.session_state.quantity
                }
                
                st.session_state.live_trades.append(trade)
                add_log(f"Position closed - Stop Loss Hit | PnL: {pnl:.2f}")
                
                st.session_state.in_position = False
                st.session_state.position_type = 0
                st.session_state.entry_price = 0
                st.session_state.stop_loss = 0
                st.session_state.target = 0
                st.session_state.trailing_sl_highest = 0
                st.session_state.trailing_sl_lowest = 0
                st.session_state.potential_pnl_sl = 0
                st.session_state.potential_pnl_target = 0
                return
            
            elif position_type == -1 and current_price >= st.session_state.stop_loss:
                exit_price = st.session_state.stop_loss
                pnl = (entry_price - exit_price) * st.session_state.quantity
                
                trade = {
                    'Entry Time': df['Datetime'].iloc[st.session_state.entry_idx].strftime('%Y-%m-%d %H:%M:%S'),
                    'Exit Time': df['Datetime'].iloc[i].strftime('%Y-%m-%d %H:%M:%S'),
                    'Type': 'SHORT',
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Stop Loss': st.session_state.stop_loss,
                    'Target': st.session_state.target,
                    'Exit Reason': 'Stop Loss Hit',
                    'PnL': pnl,
                    'Quantity': st.session_state.quantity
                }
                
                st.session_state.live_trades.append(trade)
                add_log(f"Position closed - Stop Loss Hit | PnL: {pnl:.2f}")
                
                st.session_state.in_position = False
                st.session_state.position_type = 0
                st.session_state.entry_price = 0
                st.session_state.stop_loss = 0
                st.session_state.target = 0
                st.session_state.trailing_sl_highest = 0
                st.session_state.trailing_sl_lowest = 0
                st.session_state.potential_pnl_sl = 0
                st.session_state.potential_pnl_target = 0
                return
        
        if st.session_state.target != 0:
            if position_type == 1 and current_price >= st.session_state.target:
                exit_price = st.session_state.target
                pnl = (exit_price - entry_price) * st.session_state.quantity
                
                trade = {
                    'Entry Time': df['Datetime'].iloc[st.session_state.entry_idx].strftime('%Y-%m-%d %H:%M:%S'),
                    'Exit Time': df['Datetime'].iloc[i].strftime('%Y-%m-%d %H:%M:%S'),
                    'Type': 'LONG',
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Stop Loss': st.session_state.stop_loss,
                    'Target': st.session_state.target,
                    'Exit Reason': 'Target Hit',
                    'PnL': pnl,
                    'Quantity': st.session_state.quantity
                }
                
                st.session_state.live_trades.append(trade)
                add_log(f"Position closed - Target Hit | PnL: {pnl:.2f}")
                
                st.session_state.in_position = False
                st.session_state.position_type = 0
                st.session_state.entry_price = 0
                st.session_state.stop_loss = 0
                st.session_state.target = 0
                st.session_state.trailing_sl_highest = 0
                st.session_state.trailing_sl_lowest = 0
                st.session_state.potential_pnl_sl = 0
                st.session_state.potential_pnl_target = 0
                return
            
            elif position_type == -1 and current_price <= st.session_state.target:
                exit_price = st.session_state.target
                pnl = (entry_price - exit_price) * st.session_state.quantity
                
                trade = {
                    'Entry Time': df['Datetime'].iloc[st.session_state.entry_idx].strftime('%Y-%m-%d %H:%M:%S'),
                    'Exit Time': df['Datetime'].iloc[i].strftime('%Y-%m-%d %H:%M:%S'),
                    'Type': 'SHORT',
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Stop Loss': st.session_state.stop_loss,
                    'Target': st.session_state.target,
                    'Exit Reason': 'Target Hit',
                    'PnL': pnl,
                    'Quantity': st.session_state.quantity
                }
                
                st.session_state.live_trades.append(trade)
                add_log(f"Position closed - Target Hit | PnL: {pnl:.2f}")
                
                st.session_state.in_position = False
                st.session_state.position_type = 0
                st.session_state.entry_price = 0
                st.session_state.stop_loss = 0
                st.session_state.target = 0
                st.session_state.trailing_sl_highest = 0
                st.session_state.trailing_sl_lowest = 0
                st.session_state.potential_pnl_sl = 0
                st.session_state.potential_pnl_target = 0
                return
        
        # Calculate potential P&L
        if st.session_state.stop_loss != 0:
            if position_type == 1:
                st.session_state.potential_pnl_sl = (st.session_state.stop_loss - entry_price) * st.session_state.quantity
            else:
                st.session_state.potential_pnl_sl = (entry_price - st.session_state.stop_loss) * st.session_state.quantity
        
        if st.session_state.target != 0:
            if position_type == 1:
                st.session_state.potential_pnl_target = (st.session_state.target - entry_price) * st.session_state.quantity
            else:
                st.session_state.potential_pnl_target = (entry_price - st.session_state.target) * st.session_state.quantity
    
    else:
        if signal == 1:
            entry_price = current_price
            position_type = 1
            
            sl = calculate_stop_loss(df, i, entry_price, position_type, 
                                    strategy_params['sl_type'], 
                                    strategy_params.get('sl_value', 50),
                                    strategy_params.get('atr_multiplier', 2))
            
            target = calculate_target(df, i, entry_price, position_type,
                                     strategy_params['target_type'],
                                     strategy_params.get('target_value', 100),
                                     strategy_params.get('rr_ratio', 2),
                                     strategy_params.get('atr_multiplier', 3))
            
            st.session_state.in_position = True
            st.session_state.position_type = position_type
            st.session_state.entry_price = entry_price
            st.session_state.stop_loss = sl
            st.session_state.target = target
            st.session_state.entry_idx = i
            st.session_state.trailing_sl_highest = entry_price
            st.session_state.quantity = strategy_params.get('quantity', 1)
            
            # Calculate potential P&L
            if sl != 0:
                st.session_state.potential_pnl_sl = (sl - entry_price) * st.session_state.quantity
            if target != 0:
                st.session_state.potential_pnl_target = (target - entry_price) * st.session_state.quantity
            
            sl_str = f"{sl:.2f}" if sl != 0 else "Signal Based"
            target_str = f"{target:.2f}" if target != 0 else "Signal Based"
            add_log(f"BUY Signal - Entry: {entry_price:.2f} | SL: {sl_str} | Target: {target_str}")
        
        elif signal == -1:
            entry_price = current_price
            position_type = -1
            
            sl = calculate_stop_loss(df, i, entry_price, position_type,
                                    strategy_params['sl_type'],
                                    strategy_params.get('sl_value', 50),
                                    strategy_params.get('atr_multiplier', 2))
            
            target = calculate_target(df, i, entry_price, position_type,
                                     strategy_params['target_type'],
                                     strategy_params.get('target_value', 100),
                                     strategy_params.get('rr_ratio', 2),
                                     strategy_params.get('atr_multiplier', 3))
            
            st.session_state.in_position = True
            st.session_state.position_type = position_type
            st.session_state.entry_price = entry_price
            st.session_state.stop_loss = sl
            st.session_state.target = target
            st.session_state.entry_idx = i
            st.session_state.trailing_sl_lowest = entry_price
            st.session_state.quantity = strategy_params.get('quantity', 1)
            
            # Calculate potential P&L
            if sl != 0:
                st.session_state.potential_pnl_sl = (entry_price - sl) * st.session_state.quantity
            if target != 0:
                st.session_state.potential_pnl_target = (entry_price - target) * st.session_state.quantity
            
            sl_str = f"{sl:.2f}" if sl != 0 else "Signal Based"
            target_str = f"{target:.2f}" if target != 0 else "Signal Based"
            add_log(f"SELL Signal - Entry: {entry_price:.2f} | SL: {sl_str} | Target: {target_str}")

def plot_live_chart(df, strategy_params):
    """Create live candlestick chart with indicators"""
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df['Datetime'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    if strategy_params['strategy_type'] == 'EMA Crossover' and 'EMA_Fast' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Datetime'],
            y=df['EMA_Fast'],
            mode='lines',
            name=f"EMA {strategy_params['ema_fast']}",
            line=dict(color='blue', width=1.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Datetime'],
            y=df['EMA_Slow'],
            mode='lines',
            name=f"EMA {strategy_params['ema_slow']}",
            line=dict(color='orange', width=1.5)
        ))
    
    if st.session_state.in_position:
        fig.add_hline(
            y=st.session_state.entry_price,
            line_dash="dash",
            line_color="white",
            annotation_text=f"Entry: {st.session_state.entry_price:.2f}",
            annotation_position="right"
        )
        
        if st.session_state.stop_loss != 0:
            fig.add_hline(
                y=st.session_state.stop_loss,
                line_dash="dot",
                line_color="red",
                annotation_text=f"SL: {st.session_state.stop_loss:.2f}",
                annotation_position="right"
            )
        
        if st.session_state.target != 0:
            fig.add_hline(
                y=st.session_state.target,
                line_dash="dot",
                line_color="green",
                annotation_text=f"Target: {st.session_state.target:.2f}",
                annotation_position="right"
            )
    
    fig.update_layout(
        title='Live Trading Chart',
        yaxis_title='Price',
        xaxis_title='Time',
        template='plotly_dark',
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    return fig

st.sidebar.title("⚙️ Trading Configuration")

asset_type = st.sidebar.selectbox(
    "Asset Type",
    ['Predefined', 'Custom Ticker']
)

if asset_type == 'Predefined':
    selected_asset_name = st.sidebar.selectbox(
        "Select Asset",
        list(ASSET_MAPPINGS.keys())
    )
    ticker = ASSET_MAPPINGS[selected_asset_name]
else:
    ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")
    selected_asset_name = ticker

interval = st.sidebar.selectbox(
    "Interval",
    ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1wk', '1mo']
)

valid_periods_for_interval = VALID_PERIODS.get(interval, ['1mo'])
period = st.sidebar.selectbox(
    "Period",
    valid_periods_for_interval
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Strategy Configuration")

strategy_type = st.sidebar.selectbox(
    "Strategy Type",
    ['EMA Crossover', 'Simple Buy', 'Simple Sell']
)

strategy_params = {'strategy_type': strategy_type}

if strategy_type == 'EMA Crossover':
    col1, col2 = st.sidebar.columns(2)
    with col1:
        ema_fast = st.number_input("EMA Fast", min_value=1, max_value=200, value=9)
    with col2:
        ema_slow = st.number_input("EMA Slow", min_value=1, max_value=200, value=15)
    
    min_angle = st.sidebar.slider("Minimum Crossover Angle (°)", 0, 90, 2)
    
    entry_filter = st.sidebar.selectbox(
        "Entry Filter",
        ['Simple Crossover', 'Strong Candle', 'ATR Based Candle']
    )
    
    strategy_params.update({
        'ema_fast': ema_fast,
        'ema_slow': ema_slow,
        'min_angle': min_angle,
        'entry_filter': entry_filter
    })

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Risk Management")

sl_type = st.sidebar.selectbox(
    "Stop Loss Type",
    ['Custom Points', 'Trailing SL', 'ATR Based', 'Current Candle Low/High',
     'Previous Candle Low/High', 'Current Swing Low/High', 'Previous Swing Low/High',
     'Signal Based', 'Cost to Cost']
)

sl_value = 0
trail_percent = 1
atr_multiplier = 2

if sl_type == 'Custom Points':
    sl_value = st.sidebar.number_input("SL Points", min_value=1, value=50)
elif sl_type == 'Trailing SL':
    trail_percent = st.sidebar.slider("Trail Percentage", 0.1, 5.0, 1.0, 0.1)
elif sl_type == 'ATR Based':
    atr_multiplier = st.sidebar.slider("ATR Multiplier", 0.5, 5.0, 2.0, 0.1)

strategy_params.update({
    'sl_type': sl_type,
    'sl_value': sl_value,
    'trail_percent': trail_percent,
    'atr_multiplier': atr_multiplier
})

target_type = st.sidebar.selectbox(
    "Target Type",
    ['Custom Points', 'ATR Based', 'Risk Reward Based', 'Signal Based',
     'Trailing Target', 'Cost to Cost']
)

target_value = 0
rr_ratio = 2

if target_type == 'Custom Points':
    target_value = st.sidebar.number_input("Target Points", min_value=1, value=100)
elif target_type == 'ATR Based':
    atr_multiplier_target = st.sidebar.slider("ATR Multiplier (Target)", 0.5, 5.0, 3.0, 0.1)
    strategy_params['atr_multiplier_target'] = atr_multiplier_target
elif target_type == 'Risk Reward Based':
    rr_ratio = st.sidebar.slider("Risk:Reward Ratio", 1.0, 5.0, 2.0, 0.1)

strategy_params.update({
    'target_type': target_type,
    'target_value': target_value,
    'rr_ratio': rr_ratio
})

quantity = st.sidebar.number_input("Quantity", min_value=1, value=1)
strategy_params['quantity'] = quantity

st.sidebar.markdown("---")

st.title("🚀 Professional Live Trading System")
st.markdown(f"**Asset:** {selected_asset_name} | **Interval:** {interval} | **Period:** {period}")

col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

with col1:
    if st.button("▶️ Start Live Trading", use_container_width=True):
        if not st.session_state.live_running:
            with st.spinner("Fetching data..."):
                data = fetch_data_with_retry(ticker, interval, period)
                
                if data is not None and not data.empty:
                    data = generate_signals(data, strategy_params)
                    st.session_state.live_data = data
                    st.session_state.live_running = True
                    st.session_state.last_fetch_time = get_ist_time()
                    st.session_state.iteration_count = 0
                    add_log("Live trading started")
                    st.success("Live trading started!")
                    st.rerun()
                else:
                    st.error("Failed to fetch data. Please check ticker and timeframe.")

with col2:
    if st.button("⏹️ Stop Live Trading", use_container_width=True):
        if st.session_state.live_running:
            st.session_state.live_running = False
            add_log("Live trading stopped")
            st.info("Live trading stopped")
            st.rerun()

with col3:
    st.metric("Iterations", st.session_state.iteration_count)

with col4:
    if st.session_state.live_running:
        time_since_fetch = 0
        if st.session_state.last_fetch_time:
            time_since_fetch = (get_ist_time() - st.session_state.last_fetch_time).total_seconds()
        time_display = f"Last update: {int(time_since_fetch)}s ago"
        st.metric("Status", "🟢 RUNNING", time_display)
    else:
        st.metric("Status", "🔴 STOPPED", "")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📈 Live Dashboard", "📊 Trade History", "📝 Trade Logs"])

with tab1:
    if st.session_state.live_running and st.session_state.live_data is not None:
        df = st.session_state.live_data
        
        current_time = get_ist_time()
        should_refresh = False
        
        if st.session_state.last_fetch_time:
            time_diff = (current_time - st.session_state.last_fetch_time).total_seconds()
            if time_diff >= 2:
                should_refresh = True
        
        if should_refresh:
            new_data = fetch_data_with_retry(ticker, interval, period)
            
            if new_data is not None and not new_data.empty:
                new_data = generate_signals(new_data, strategy_params)
                st.session_state.live_data = new_data
                df = new_data
                st.session_state.last_fetch_time = current_time
                st.session_state.iteration_count += 1
                
                process_live_candle(df, strategy_params)
                
                time.sleep(0.1)
                st.rerun()
        
        if len(df) > 0:
            latest = df.iloc[-1]
            current_price = latest['Close']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"{current_price:.2f}")
            
            with col2:
                st.metric("Total Candles", len(df))
            
            with col3:
                if strategy_params['strategy_type'] == 'EMA Crossover' and 'EMA_Fast' in df.columns:
                    ema_fast_val = latest['EMA_Fast']
                    st.metric(f"EMA {strategy_params['ema_fast']}", f"{ema_fast_val:.2f}")
                else:
                    st.metric("EMA Fast", "N/A")
            
            with col4:
                if strategy_params['strategy_type'] == 'EMA Crossover' and 'EMA_Slow' in df.columns:
                    ema_slow_val = latest['EMA_Slow']
                    st.metric(f"EMA {strategy_params['ema_slow']}", f"{ema_slow_val:.2f}")
                else:
                    st.metric("EMA Slow", "N/A")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if strategy_params['strategy_type'] == 'EMA Crossover' and 'EMA_Fast' in df.columns:
                    current_angle = calculate_ema_angle(df, strategy_params['ema_fast'])
                    angle_display = f"{abs(current_angle):.2f}°"
                    st.metric("Crossover Angle", angle_display)
                else:
                    st.metric("Crossover Angle", "N/A")
            
            with col2:
                if st.session_state.in_position:
                    pos_text = "LONG" if st.session_state.position_type == 1 else "SHORT"
                    st.metric("Position Status", "OPEN", pos_text)
                else:
                    st.metric("Position Status", "CLOSED", "")
            
            with col3:
                signal = latest['Signal']
                if signal == 1:
                    st.metric("Current Signal", "BUY", "🟢")
                elif signal == -1:
                    st.metric("Current Signal", "SELL", "🔴")
                else:
                    st.metric("Current Signal", "NONE", "⚪")
            
            st.markdown("---")
            
            if st.session_state.in_position:
                st.subheader("📍 Active Position")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Entry Price", f"{st.session_state.entry_price:.2f}")
                
                with col2:
                    if st.session_state.stop_loss != 0:
                        sl_display = f"{st.session_state.stop_loss:.2f}"
                        pnl_sl_display = f"(P&L: {st.session_state.potential_pnl_sl:.2f})"
                        st.metric("Stop Loss", sl_display, pnl_sl_display)
                    else:
                        st.metric("Stop Loss", "Signal Based", "")
                
                with col3:
                    if st.session_state.target != 0:
                        target_display = f"{st.session_state.target:.2f}"
                        pnl_target_display = f"(P&L: {st.session_state.potential_pnl_target:.2f})"
                        st.metric("Target", target_display, pnl_target_display)
                    else:
                        st.metric("Target", "Signal Based", "")
                
                with col4:
                    if st.session_state.position_type == 1:
                        unrealized_pnl = (current_price - st.session_state.entry_price) * st.session_state.quantity
                    else:
                        unrealized_pnl = (st.session_state.entry_price - current_price) * st.session_state.quantity
                    
                    st.metric("Unrealized P&L", f"{unrealized_pnl:.2f}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.session_state.stop_loss != 0:
                        if st.session_state.position_type == 1:
                            dist_sl = current_price - st.session_state.stop_loss
                        else:
                            dist_sl = st.session_state.stop_loss - current_price
                        st.metric("Distance to SL", f"{dist_sl:.2f} pts")
                    else:
                        st.metric("Distance to SL", "Signal Based")
                
                with col2:
                    if st.session_state.target != 0:
                        if st.session_state.position_type == 1:
                            dist_target = st.session_state.target - current_price
                        else:
                            dist_target = current_price - st.session_state.target
                        st.metric("Distance to Target", f"{dist_target:.2f} pts")
                    else:
                        st.metric("Distance to Target", "Signal Based")
                
                st.markdown("---")
                
                if st.session_state.stop_loss != 0 and st.session_state.target != 0:
                    if st.session_state.position_type == 1:
                        if current_price >= st.session_state.target:
                            st.success("🎯 **Target reached! Consider exiting.**")
                        elif current_price <= st.session_state.stop_loss:
                            st.error("⚠️ **Stop Loss hit! Exit position.**")
                        elif unrealized_pnl > 0:
                            st.info("📈 **Hold position - In profit**")
                        else:
                            st.warning("📉 **Monitor position - In loss**")
                    else:
                        if current_price <= st.session_state.target:
                            st.success("🎯 **Target reached! Consider exiting.**")
                        elif current_price >= st.session_state.stop_loss:
                            st.error("⚠️ **Stop Loss hit! Exit position.**")
                        elif unrealized_pnl > 0:
                            st.info("📈 **Hold position - In profit**")
                        else:
                            st.warning("📉 **Monitor position - In loss**")
                else:
                    st.info("🔍 **Monitoring for signal-based exit...**")
            
            else:
                st.info("💤 **No active position. Waiting for signal...**")
            
            st.markdown("---")
            
            fig = plot_live_chart(df.tail(100), strategy_params)
            chart_key = f"live_chart_{int(get_ist_time().timestamp() * 1000)}"
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
        
        if st.session_state.live_running:
            time.sleep(0.5)
            st.rerun()
    
    else:
        st.info("👆 Click 'Start Live Trading' to begin")

with tab2:
    if len(st.session_state.live_trades) > 0:
        st.subheader("📊 Trade History")
        
        trades_df = pd.DataFrame(st.session_state.live_trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['PnL'] > 0])
        accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = trades_df['PnL'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", total_trades)
        
        with col2:
            st.metric("Winning Trades", winning_trades)
        
        with col3:
            st.metric("Accuracy", f"{accuracy:.2f}%")
        
        with col4:
            st.metric("Total P&L", f"{total_pnl:.2f}")
        
        st.markdown("---")
        
        st.dataframe(trades_df, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📝 Trade Explanations")
        
        for idx, trade in enumerate(st.session_state.live_trades):
            pnl_indicator = "✅" if trade['PnL'] > 0 else "❌"
            with st.expander(f"{pnl_indicator} Trade #{idx + 1} - {trade['Type']} - P&L: {trade['PnL']:.2f}"):
                st.write(f"**Entry Time:** {trade['Entry Time']}")
                st.write(f"**Exit Time:** {trade['Exit Time']}")
                st.write(f"**Type:** {trade['Type']}")
                st.write(f"**Entry Price:** {trade['Entry Price']:.2f}")
                st.write(f"**Exit Price:** {trade['Exit Price']:.2f}")
                
                if trade['Stop Loss'] != 0:
                    sl_text = f"{trade['Stop Loss']:.2f}"
                else:
                    sl_text = "Signal Based"
                
                if trade['Target'] != 0:
                    target_text = f"{trade['Target']:.2f}"
                else:
                    target_text = "Signal Based"
                
                st.write(f"**Stop Loss:** {sl_text}")
                st.write(f"**Target:** {target_text}")
                st.write(f"**Exit Reason:** {trade['Exit Reason']}")
                st.write(f"**Quantity:** {trade['Quantity']}")
                st.write(f"**P&L:** {trade['PnL']:.2f}")
                
                if trade['Type'] == 'LONG':
                    explanation = f"Entered LONG at {trade['Entry Price']:.2f}. "
                else:
                    explanation = f"Entered SHORT at {trade['Entry Price']:.2f}. "
                
                explanation += f"Exited at {trade['Exit Price']:.2f} due to {trade['Exit Reason']}. "
                
                if trade['PnL'] > 0:
                    explanation += f"Trade was profitable with P&L of {trade['PnL']:.2f}."
                else:
                    explanation += f"Trade resulted in loss of {trade['PnL']:.2f}."
                
                st.info(explanation)
    
    else:
        st.info("No trades executed yet")

with tab3:
    st.subheader("📝 Trade Logs")
    
    if len(st.session_state.live_logs) > 0:
        for log in reversed(st.session_state.live_logs):
            st.text(log)
    else:
        st.info("No logs available")
    
    if st.button("Clear Logs"):
        st.session_state.live_logs = []
        st.rerun()

st.markdown("---")
st.markdown("**⚠️ Disclaimer:** This is a simulated trading system for educational purposes only. Not financial advice.")
