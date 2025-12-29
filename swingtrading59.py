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
st.set_page_config(page_title="Live Trading Engine", layout="wide", initial_sidebar_state="expanded")

# IST timezone
IST = pytz.timezone('Asia/Kolkata')

def get_ist_time():
    """Get current time in IST"""
    return datetime.now(IST)

# Asset mappings
ASSET_MAPPING = {
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "USD/INR": "USDINR=X",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "Gold": "GC=F",
    "Silver": "SI=F"
}

# Timeframe validation
TIMEFRAME_VALIDATION = {
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
def initialize_session_state():
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
    if 'entry_time' not in st.session_state:
        st.session_state.entry_time = None
    if 'quantity' not in st.session_state:
        st.session_state.quantity = 1

def add_log(message):
    """Add timestamped log entry - only for entry/exit events"""
    timestamp = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.live_logs.append(log_entry)
    if len(st.session_state.live_logs) > 100:
        st.session_state.live_logs = st.session_state.live_logs[-100:]

def fetch_data_yfinance(ticker, interval, period):
    """Fetch data from yfinance with rate limiting and proper timezone handling"""
    try:
        time.sleep(random.uniform(1.0, 1.5))
        
        data = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
        
        if data.empty:
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in data.columns.values]
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                matching_cols = [c for c in data.columns if col in c]
                if matching_cols:
                    data[col] = data[matching_cols[0]]
                else:
                    return None
        
        data = data[required_cols].copy()
        
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert(IST)
        else:
            data.index = data.index.tz_convert(IST)
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_ema(data, period):
    """Calculate EMA manually"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    """Calculate ATR manually"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    
    return atr

def calculate_ema_angle(ema_fast, ema_slow, idx):
    """Calculate EMA crossover angle in degrees"""
    if idx < 1:
        return 0
    
    fast_slope = ema_fast.iloc[idx] - ema_fast.iloc[idx-1]
    slow_slope = ema_slow.iloc[idx] - ema_slow.iloc[idx-1]
    
    angle_rad = np.arctan(fast_slope - slow_slope)
    angle_deg = np.degrees(angle_rad)
    
    return abs(angle_deg)

def identify_swing_points(df, lookback=5):
    """Identify swing highs and lows"""
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(df) - lookback):
        if df['High'].iloc[i] == df['High'].iloc[i-lookback:i+lookback+1].max():
            swing_highs.append(i)
        
        if df['Low'].iloc[i] == df['Low'].iloc[i-lookback:i+lookback+1].min():
            swing_lows.append(i)
    
    return swing_highs, swing_lows

def calculate_stop_loss(df, idx, sl_type, sl_value, position_type, entry_price, atr_value=None):
    """Calculate stop loss based on type"""
    min_distance = entry_price * 0.005
    
    if sl_type == "Custom Points":
        if position_type == 1:
            sl = entry_price - sl_value
        else:
            sl = entry_price + sl_value
    
    elif sl_type == "ATR-based":
        if atr_value is None:
            return 0
        if position_type == 1:
            sl = entry_price - (sl_value * atr_value)
        else:
            sl = entry_price + (sl_value * atr_value)
    
    elif sl_type == "Current Candle Low/High":
        if position_type == 1:
            sl = df['Low'].iloc[idx]
        else:
            sl = df['High'].iloc[idx]
    
    elif sl_type == "Previous Candle Low/High":
        if idx < 1:
            return 0
        if position_type == 1:
            sl = df['Low'].iloc[idx-1]
        else:
            sl = df['High'].iloc[idx-1]
    
    elif sl_type == "Current Swing Low/High":
        swing_highs, swing_lows = identify_swing_points(df[:idx+1])
        if position_type == 1 and swing_lows:
            sl = df['Low'].iloc[swing_lows[-1]]
        elif position_type == -1 and swing_highs:
            sl = df['High'].iloc[swing_highs[-1]]
        else:
            return 0
    
    elif sl_type == "Previous Swing Low/High":
        swing_highs, swing_lows = identify_swing_points(df[:idx+1])
        if position_type == 1 and len(swing_lows) > 1:
            sl = df['Low'].iloc[swing_lows[-2]]
        elif position_type == -1 and len(swing_highs) > 1:
            sl = df['High'].iloc[swing_highs[-2]]
        else:
            return 0
    
    elif sl_type == "Signal-based":
        return 0
    
    elif sl_type == "Trailing SL":
        if position_type == 1:
            sl = entry_price - sl_value
        else:
            sl = entry_price + sl_value
    
    else:
        sl = 0
    
    if position_type == 1:
        sl = min(sl, entry_price - min_distance)
    else:
        sl = max(sl, entry_price + min_distance)
    
    return sl

def calculate_target(df, idx, target_type, target_value, position_type, entry_price, atr_value=None, sl_price=0):
    """Calculate target based on type"""
    min_distance = entry_price * 0.01
    
    if target_type == "Custom Points":
        if position_type == 1:
            target = entry_price + target_value
        else:
            target = entry_price - target_value
    
    elif target_type == "ATR-based":
        if atr_value is None:
            return 0
        if position_type == 1:
            target = entry_price + (target_value * atr_value)
        else:
            target = entry_price - (target_value * atr_value)
    
    elif target_type == "Risk-Reward Based":
        if sl_price == 0:
            return 0
        risk = abs(entry_price - sl_price)
        if position_type == 1:
            target = entry_price + (risk * target_value)
        else:
            target = entry_price - (risk * target_value)
    
    elif target_type == "Signal-based":
        return 0
    
    elif target_type == "Trailing Target":
        if position_type == 1:
            target = entry_price + target_value
        else:
            target = entry_price - target_value
    
    else:
        target = 0
    
    if position_type == 1:
        target = max(target, entry_price + min_distance)
    else:
        target = min(target, entry_price - min_distance)
    
    return target

def update_trailing_sl(df, idx, position_type, entry_price, trailing_points):
    """Update trailing stop loss"""
    current_price = df['Close'].iloc[idx]
    
    if position_type == 1:
        if current_price > st.session_state.trailing_sl_highest:
            st.session_state.trailing_sl_highest = current_price
            new_sl = st.session_state.trailing_sl_highest - trailing_points
            if new_sl > st.session_state.stop_loss:
                st.session_state.stop_loss = new_sl
    
    else:
        if st.session_state.trailing_sl_lowest == 0 or current_price < st.session_state.trailing_sl_lowest:
            st.session_state.trailing_sl_lowest = current_price
            new_sl = st.session_state.trailing_sl_lowest + trailing_points
            if st.session_state.stop_loss == 0 or new_sl < st.session_state.stop_loss:
                st.session_state.stop_loss = new_sl

def check_signal_based_exit(df, idx, position_type):
    """Check for signal-based exit"""
    if idx < 1:
        return False, None
    
    ema_fast_curr = df['EMA_Fast'].iloc[idx]
    ema_slow_curr = df['EMA_Slow'].iloc[idx]
    ema_fast_prev = df['EMA_Fast'].iloc[idx-1]
    ema_slow_prev = df['EMA_Slow'].iloc[idx-1]
    
    if position_type == 1:
        if ema_fast_curr < ema_slow_curr and ema_fast_prev >= ema_slow_prev:
            return True, "Reverse Signal - Bearish Crossover"
    
    elif position_type == -1:
        if ema_fast_curr > ema_slow_curr and ema_fast_prev <= ema_slow_prev:
            return True, "Reverse Signal - Bullish Crossover"
    
    return False, None

def generate_ema_crossover_signal(df, idx, strategy_params):
    """Generate EMA crossover signal"""
    if idx < 1:
        return 0, {}
    
    ema_fast = df['EMA_Fast'].iloc[idx]
    ema_slow = df['EMA_Slow'].iloc[idx]
    ema_fast_prev = df['EMA_Fast'].iloc[idx-1]
    ema_slow_prev = df['EMA_Slow'].iloc[idx-1]
    
    bullish_cross = (ema_fast > ema_slow) and (ema_fast_prev <= ema_slow_prev)
    bearish_cross = (ema_fast < ema_slow) and (ema_fast_prev >= ema_slow_prev)
    
    if not bullish_cross and not bearish_cross:
        return 0, {}
    
    angle = calculate_ema_angle(df['EMA_Fast'], df['EMA_Slow'], idx)
    min_angle = strategy_params['min_angle']
    
    if angle < min_angle:
        return 0, {}
    
    entry_filter = strategy_params['entry_filter']
    signal = 0
    
    if entry_filter == "Simple Crossover":
        signal = 1 if bullish_cross else -1
    
    elif entry_filter == "Strong Candle (Points)":
        candle_size = abs(df['Close'].iloc[idx] - df['Open'].iloc[idx])
        min_size = strategy_params['strong_candle_points']
        if candle_size >= min_size:
            signal = 1 if bullish_cross else -1
    
    elif entry_filter == "ATR-based Candle":
        candle_size = abs(df['Close'].iloc[idx] - df['Open'].iloc[idx])
        atr = df['ATR'].iloc[idx]
        multiplier = strategy_params['atr_multiplier']
        if candle_size >= (atr * multiplier):
            signal = 1 if bullish_cross else -1
    
    signal_data = {
        'angle': angle,
        'ema_fast': ema_fast,
        'ema_slow': ema_slow
    }
    
    return signal, signal_data

def execute_live_trade(df, idx, signal, strategy_params):
    """Execute trade entry logic"""
    if signal == 0:
        return
    
    entry_price = df['Close'].iloc[idx]
    position_type = signal
    
    atr_value = df['ATR'].iloc[idx] if 'ATR' in df.columns else None
    
    sl_type = strategy_params['sl_type']
    sl_value = strategy_params['sl_value']
    stop_loss = calculate_stop_loss(df, idx, sl_type, sl_value, position_type, entry_price, atr_value)
    
    target_type = strategy_params['target_type']
    target_value = strategy_params['target_value']
    target = calculate_target(df, idx, target_type, target_value, position_type, entry_price, atr_value, stop_loss)
    
    st.session_state.in_position = True
    st.session_state.position_type = position_type
    st.session_state.entry_price = entry_price
    st.session_state.stop_loss = stop_loss
    st.session_state.target = target
    st.session_state.entry_idx = idx
    st.session_state.entry_time = df.index[idx]
    st.session_state.trailing_sl_highest = entry_price if position_type == 1 else 0
    st.session_state.trailing_sl_lowest = entry_price if position_type == -1 else 0
    
    position_text = "LONG" if position_type == 1 else "SHORT"
    sl_text = f"{stop_loss:.2f}" if stop_loss != 0 else "Signal Based"
    target_text = f"{target:.2f}" if target != 0 else "Signal Based"
    
    entry_time_str = df.index[idx].strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"ENTRY | {position_text} | Time: {entry_time_str} | Price: {entry_price:.2f} | SL: {sl_text} | Target: {target_text}"
    add_log(log_msg)
    
    # PLACEHOLDER: Integrate Dhan order placement here
    # dhan.place_order(
    #     transaction_type="BUY" if position_type == 1 else "SELL",
    #     exchange_segment="NSE_EQ",
    #     product_type="INTRADAY",
    #     order_type="MARKET",
    #     quantity=strategy_params['quantity'],
    #     price=0
    # )

def check_exit_conditions(df, idx, strategy_params):
    """Check if exit conditions are met"""
    if not st.session_state.in_position:
        return False, None, None
    
    current_price = df['Close'].iloc[idx]
    position_type = st.session_state.position_type
    entry_price = st.session_state.entry_price
    stop_loss = st.session_state.stop_loss
    target = st.session_state.target
    
    if strategy_params['sl_type'] == "Trailing SL":
        update_trailing_sl(df, idx, position_type, entry_price, strategy_params['sl_value'])
        stop_loss = st.session_state.stop_loss
    
    if strategy_params['sl_type'] == "Signal-based" or strategy_params['target_type'] == "Signal-based":
        should_exit, reason = check_signal_based_exit(df, idx, position_type)
        if should_exit:
            return True, current_price, reason
    
    if stop_loss != 0:
        if position_type == 1 and current_price <= stop_loss:
            return True, current_price, "Stop Loss Hit"
        elif position_type == -1 and current_price >= stop_loss:
            return True, current_price, "Stop Loss Hit"
    
    if target != 0:
        if position_type == 1 and current_price >= target:
            return True, current_price, "Target Hit"
        elif position_type == -1 and current_price <= target:
            return True, current_price, "Target Hit"
    
    return False, None, None

def execute_exit(exit_price, exit_reason, df, idx):
    """Execute trade exit and record trade"""
    entry_price = st.session_state.entry_price
    position_type = st.session_state.position_type
    entry_time = st.session_state.entry_time
    exit_time = df.index[idx]
    
    if position_type == 1:
        pnl = (exit_price - entry_price) * st.session_state.quantity
    else:
        pnl = (entry_price - exit_price) * st.session_state.quantity
    
    trade = {
        'Entry Time': entry_time,
        'Exit Time': exit_time,
        'Type': "LONG" if position_type == 1 else "SHORT",
        'Entry Price': entry_price,
        'Exit Price': exit_price,
        'PnL': pnl,
        'Reason': exit_reason,
        'Stop Loss': st.session_state.stop_loss,
        'Target': st.session_state.target
    }
    
    st.session_state.live_trades.append(trade)
    
    exit_time_str = exit_time.strftime("%Y-%m-%d %H:%M:%S")
    pnl_text = f"+{pnl:.2f}" if pnl > 0 else f"{pnl:.2f}"
    log_msg = f"EXIT | Time: {exit_time_str} | Price: {exit_price:.2f} | PnL: {pnl_text} | Reason: {exit_reason}"
    add_log(log_msg)
    
    # PLACEHOLDER: Integrate Dhan order placement here for exit
    # dhan.place_order(
    #     transaction_type="SELL" if position_type == 1 else "BUY",
    #     exchange_segment="NSE_EQ",
    #     product_type="INTRADAY",
    #     order_type="MARKET",
    #     quantity=st.session_state.quantity,
    #     price=0
    # )
    
    st.session_state.in_position = False
    st.session_state.position_type = 0
    st.session_state.entry_price = 0
    st.session_state.stop_loss = 0
    st.session_state.target = 0
    st.session_state.entry_idx = 0
    st.session_state.entry_time = None
    st.session_state.trailing_sl_highest = 0
    st.session_state.trailing_sl_lowest = 0

def plot_live_chart(df, strategy_params):
    """Plot live candlestick chart with EMAs and trade levels"""
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    if 'EMA_Fast' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['EMA_Fast'],
            mode='lines',
            name=f"EMA {strategy_params['ema_fast']}",
            line=dict(color='blue', width=1)
        ))
    
    if 'EMA_Slow' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['EMA_Slow'],
            mode='lines',
            name=f"EMA {strategy_params['ema_slow']}",
            line=dict(color='red', width=1)
        ))
    
    if st.session_state.in_position:
        fig.add_hline(
            y=st.session_state.entry_price,
            line_dash="dash",
            line_color="yellow",
            annotation_text="Entry",
            annotation_position="right"
        )
        
        if st.session_state.stop_loss != 0:
            fig.add_hline(
                y=st.session_state.stop_loss,
                line_dash="dash",
                line_color="red",
                annotation_text="SL",
                annotation_position="right"
            )
        
        if st.session_state.target != 0:
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
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )
    
    return fig

def main():
    st.title("🚀 Professional Live Trading Engine")
    
    initialize_session_state()
    
    st.sidebar.header("⚙️ Trading Configuration")
    
    asset_type = st.sidebar.selectbox("Asset Type", ["Predefined", "Custom Ticker"])
    
    if asset_type == "Predefined":
        selected_asset = st.sidebar.selectbox("Select Asset", list(ASSET_MAPPING.keys()))
        ticker = ASSET_MAPPING[selected_asset]
    else:
        ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")
    
    interval = st.sidebar.selectbox(
        "Interval",
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"]
    )
    
    allowed_periods = TIMEFRAME_VALIDATION.get(interval, ["1d"])
    period = st.sidebar.selectbox("Period", allowed_periods)
    
    st.sidebar.subheader("📊 Strategy Settings")
    strategy = st.sidebar.selectbox(
        "Strategy",
        ["EMA Crossover", "Simple Buy", "Simple Sell"]
    )
    
    strategy_params = {}
    
    if strategy == "EMA Crossover":
        strategy_params['ema_fast'] = st.sidebar.number_input("EMA Fast", min_value=1, value=9)
        strategy_params['ema_slow'] = st.sidebar.number_input("EMA Slow", min_value=1, value=15)
        strategy_params['min_angle'] = st.sidebar.number_input("Min Crossover Angle (degrees)", min_value=0, value=1)
        strategy_params['entry_filter'] = st.sidebar.selectbox(
            "Entry Filter",
            ["Simple Crossover", "Strong Candle (Points)", "ATR-based Candle"]
        )
        
        if strategy_params['entry_filter'] == "Strong Candle (Points)":
            strategy_params['strong_candle_points'] = st.sidebar.number_input("Min Candle Size (Points)", min_value=1, value=10)
        elif strategy_params['entry_filter'] == "ATR-based Candle":
            strategy_params['atr_multiplier'] = st.sidebar.number_input("ATR Multiplier", min_value=0.1, value=1.0, step=0.1)
    
    st.sidebar.subheader("🛑 Stop Loss Settings")
    sl_type = st.sidebar.selectbox(
        "SL Type",
        ["Custom Points", "Trailing SL", "ATR-based", "Current Candle Low/High",
         "Previous Candle Low/High", "Current Swing Low/High", "Previous Swing Low/High",
         "Signal-based"]
    )
    strategy_params['sl_type'] = sl_type
    
    if sl_type in ["Custom Points", "Trailing SL"]:
        strategy_params['sl_value'] = st.sidebar.number_input("SL Points", min_value=1, value=5)
    elif sl_type == "ATR-based":
        strategy_params['sl_value'] = st.sidebar.number_input("ATR Multiplier", min_value=0.1, value=2.0, step=0.1)
    else:
        strategy_params['sl_value'] = 0
    
    st.sidebar.subheader("🎯 Target Settings")
    target_type = st.sidebar.selectbox(
        "Target Type",
        ["Custom Points", "ATR-based", "Trailing Target", "Risk-Reward Based", "Signal-based"]
    )
    strategy_params['target_type'] = target_type
    
    if target_type in ["Custom Points", "Trailing Target"]:
        strategy_params['target_value'] = st.sidebar.number_input("Target Points", min_value=1, value=2)
    elif target_type == "ATR-based":
        strategy_params['target_value'] = st.sidebar.number_input("ATR Multiplier (Target)", min_value=0.1, value=3.0, step=0.1)
    elif target_type == "Risk-Reward Based":
        strategy_params['target_value'] = st.sidebar.number_input("Risk-Reward Ratio", min_value=0.1, value=2.0, step=0.1)
    else:
        strategy_params['target_value'] = 0
    
    strategy_params['quantity'] = st.sidebar.number_input("Quantity", min_value=1, value=1)
    st.session_state.quantity = strategy_params['quantity']
    
    # Control buttons on main page
    st.subheader("🎮 Trading Controls")
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if st.button("▶️ Start Trading", disabled=st.session_state.live_running, use_container_width=True):
            with st.spinner("Fetching market data..."):
                data = fetch_data_yfinance(ticker, interval, period)
                
                if data is not None and not data.empty:
                    if strategy == "EMA Crossover":
                        data['EMA_Fast'] = calculate_ema(data['Close'], strategy_params['ema_fast'])
                        data['EMA_Slow'] = calculate_ema(data['Close'], strategy_params['ema_slow'])
                        data['ATR'] = calculate_atr(data, 14)
                    
                    st.session_state.live_data = data
                    st.session_state.live_running = True
                    add_log("✅ Live trading started")
                    st.rerun()
                else:
                    st.error("Failed to fetch data. Please check ticker and try again.")
    
    with col2:
        if st.button("⏹️ Stop Trading", disabled=not st.session_state.live_running, use_container_width=True):
            st.session_state.live_running = False
            add_log("🛑 Live trading stopped")
            st.rerun()
    
    # Tabs always visible
    tab1, tab2, tab3 = st.tabs(["📊 Live Dashboard", "📜 Trade History", "📝 Trade Logs"])
    
    with tab1:
        if st.session_state.live_running:
            time.sleep(random.uniform(1.0, 1.5))
            
            data = fetch_data_yfinance(ticker, interval, period)
            
            if data is not None and not data.empty:
                if strategy == "EMA Crossover":
                    data['EMA_Fast'] = calculate_ema(data['Close'], strategy_params['ema_fast'])
                    data['EMA_Slow'] = calculate_ema(data['Close'], strategy_params['ema_slow'])
                    data['ATR'] = calculate_atr(data, 14)
                
                st.session_state.live_data = data
                
                latest_idx = len(data) - 1
                current_price = data['Close'].iloc[latest_idx]
                
                st.subheader("📈 Live Market Data")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Current Price", f"{current_price:.2f}")
                
                with metric_col2:
                    if strategy == "EMA Crossover":
                        ema_fast_val = data['EMA_Fast'].iloc[latest_idx]
                        st.metric("EMA Fast", f"{ema_fast_val:.2f}")
                
                with metric_col3:
                    if strategy == "EMA Crossover":
                        ema_slow_val = data['EMA_Slow'].iloc[latest_idx]
                        st.metric("EMA Slow", f"{ema_slow_val:.2f}")
                
                metric_col4, metric_col5, metric_col6 = st.columns(3)
                
                with metric_col4:
                    if strategy == "EMA Crossover":
                        angle = calculate_ema_angle(data['EMA_Fast'], data['EMA_Slow'], latest_idx)
                        angle_str = f"{angle:.1f}°"
                        st.metric("Crossover Angle", angle_str)
                
                with metric_col5:
                    position_status = "OPEN" if st.session_state.in_position else "CLOSED"
                    position_color = "🟢" if st.session_state.in_position else "⚪"
                    st.metric("Position Status", f"{position_color} {position_status}")
                
                with metric_col6:
                    current_signal = "NONE"
                    if strategy == "EMA Crossover":
                        signal, signal_data = generate_ema_crossover_signal(data, latest_idx, strategy_params)
                        if signal == 1:
                            current_signal = "🟢 BUY"
                        elif signal == -1:
                            current_signal = "🔴 SELL"
                    st.metric("Current Signal", current_signal)
                
                if st.session_state.in_position:
                    st.subheader("💼 Open Position Details")
                    
                    pos_col1, pos_col2, pos_col3, pos_col4 = st.columns(4)
                    
                    with pos_col1:
                        pos_type = "LONG" if st.session_state.position_type == 1 else "SHORT"
                        st.metric("Type", pos_type)
                    
                    with pos_col2:
                        st.metric("Entry Price", f"{st.session_state.entry_price:.2f}")
                    
                    with pos_col3:
                        if st.session_state.stop_loss != 0:
                            sl_display = f"{st.session_state.stop_loss:.2f}"
                        else:
                            sl_display = "Signal Based"
                        st.metric("Stop Loss", sl_display)
                    
                    with pos_col4:
                        if st.session_state.target != 0:
                            target_display = f"{st.session_state.target:.2f}"
                        else:
                            target_display = "Signal Based"
                        st.metric("Target", target_display)
                    
                    pos_col5, pos_col6, pos_col7 = st.columns(3)
                    
                    with pos_col5:
                        if st.session_state.stop_loss != 0:
                            sl_distance = abs(current_price - st.session_state.stop_loss)
                            st.metric("Distance to SL", f"{sl_distance:.2f}")
                    
                    with pos_col6:
                        if st.session_state.target != 0:
                            target_distance = abs(st.session_state.target - current_price)
                            st.metric("Distance to Target", f"{target_distance:.2f}")
                    
                    with pos_col7:
                        if st.session_state.position_type == 1:
                            unrealized_pnl = (current_price - st.session_state.entry_price) * st.session_state.quantity
                        else:
                            unrealized_pnl = (st.session_state.entry_price - current_price) * st.session_state.quantity
                        
                        st.metric("Unrealized PnL", f"{unrealized_pnl:.2f}", delta=f"{unrealized_pnl:.2f}")
                    
                    st.subheader("💡 Trading Guidance")
                    
                    guidance_text = ""
                    if st.session_state.position_type == 1:
                        if current_price > st.session_state.entry_price:
                            guidance_text = "✅ Position in profit. Monitor for exit signals."
                        else:
                            guidance_text = "⚠️ Position in loss. Watch stop loss level."
                    else:
                        if current_price < st.session_state.entry_price:
                            guidance_text = "✅ Position in profit. Monitor for exit signals."
                        else:
                            guidance_text = "⚠️ Position in loss. Watch stop loss level."
                    
                    st.info(guidance_text)
                
                if not st.session_state.in_position:
                    if strategy == "EMA Crossover":
                        signal, signal_data = generate_ema_crossover_signal(data, latest_idx, strategy_params)
                        if signal != 0:
                            execute_live_trade(data, latest_idx, signal, strategy_params)
                            st.rerun()
                    elif strategy == "Simple Buy":
                        execute_live_trade(data, latest_idx, 1, strategy_params)
                        st.rerun()
                    elif strategy == "Simple Sell":
                        execute_live_trade(data, latest_idx, -1, strategy_params)
                        st.rerun()
                else:
                    should_exit, exit_price, exit_reason = check_exit_conditions(data, latest_idx, strategy_params)
                    if should_exit:
                        execute_exit(exit_price, exit_reason, data, latest_idx)
                        st.rerun()
                
                st.subheader("📊 Live Chart")
                chart_key = f"live_chart_{get_ist_time().timestamp()}"
                fig = plot_live_chart(data, strategy_params)
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
                
                st.subheader("⚙️ Active Strategy Parameters")
                param_col1, param_col2 = st.columns(2)
                
                with param_col1:
                    st.write(f"**Strategy:** {strategy}")
                    if strategy == "EMA Crossover":
                        st.write(f"**EMA Fast:** {strategy_params['ema_fast']}")
                        st.write(f"**EMA Slow:** {strategy_params['ema_slow']}")
                        st.write(f"**Min Angle:** {strategy_params['min_angle']}°")
                        st.write(f"**Entry Filter:** {strategy_params['entry_filter']}")
                
                with param_col2:
                    st.write(f"**SL Type:** {strategy_params['sl_type']}")
                    st.write(f"**Target Type:** {strategy_params['target_type']}")
                    st.write(f"**Quantity:** {strategy_params['quantity']}")
                
                time.sleep(1)
                st.rerun()
        else:
            st.info("Click 'Start Trading' button above to begin live trading.")
    
    with tab2:
        st.subheader("📜 Trade History")
        
        if st.session_state.live_trades:
            trades_df = pd.DataFrame(st.session_state.live_trades)
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['PnL'] > 0])
            losing_trades = len(trades_df[trades_df['PnL'] < 0])
            if total_trades > 0:
                accuracy = (winning_trades / total_trades * 100)
            else:
                accuracy = 0
            total_pnl = trades_df['PnL'].sum()
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Total Trades", total_trades)
            
            with metric_col2:
                st.metric("Winning Trades", winning_trades)
            
            with metric_col3:
                st.metric("Accuracy %", f"{accuracy:.2f}%")
            
            with metric_col4:
                st.metric("Total PnL", f"{total_pnl:.2f}", delta=f"{total_pnl:.2f}")
            
            st.dataframe(trades_df, use_container_width=True)
            
            st.subheader("📖 Trade Explanations")
            for idx, trade in enumerate(st.session_state.live_trades):
                with st.expander(f"Trade #{idx + 1} - {trade['Type']} - PnL: {trade['PnL']:.2f}"):
                    st.write(f"**Entry Time:** {trade['Entry Time']}")
                    st.write(f"**Exit Time:** {trade['Exit Time']}")
                    st.write(f"**Position Type:** {trade['Type']}")
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
                    st.write(f"**Exit Reason:** {trade['Reason']}")
                    st.write(f"**PnL:** {trade['PnL']:.2f}")
                    
                    holding_time = trade['Exit Time'] - trade['Entry Time']
                    st.write(f"**Holding Time:** {holding_time}")
        else:
            st.info("No trades executed yet. Waiting for signals...")
    
    with tab3:
        st.subheader("📝 Trade Logs")
        
        if st.session_state.live_logs:
            for log in reversed(st.session_state.live_logs):
                st.text(log)
        else:
            st.info("No logs yet. Start trading to see activity logs.")

if __name__ == "__main__":
    main()
