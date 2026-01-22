import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import pytz
import random

# Page configuration
st.set_page_config(page_title="Quantitative Trading System", layout="wide")

# Asset configurations
ASSETS = {
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

PERIOD_MAP = {
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

# Initialize session state
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
if 'trailing_high' not in st.session_state:
    st.session_state.trailing_high = 0
if 'trailing_low' not in st.session_state:
    st.session_state.trailing_low = float('inf')
if 'partial_exit_done' not in st.session_state:
    st.session_state.partial_exit_done = False
if 'breakeven_activated' not in st.session_state:
    st.session_state.breakeven_activated = False
if 'first_candle_price' not in st.session_state:
    st.session_state.first_candle_price = None
if 'custom_conditions' not in st.session_state:
    st.session_state.custom_conditions = []

# ==================== INDICATOR FUNCTIONS ====================

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    if len(data) < period:
        return [None] * len(data)
    k = 2 / (period + 1)
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(data[i] * k + ema[i - 1] * (1 - k))
    return ema

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    sma = []
    for i in range(len(data)):
        if i < period - 1:
            sma.append(None)
        else:
            sma.append(sum(data[i - period + 1:i + 1]) / period)
    return sma

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr = [high[0] - low[0]]
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr.append(max(hl, hc, lc))
    return calculate_ema(tr, period)

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    if len(data) < period + 1:
        return [None] * len(data)
    
    gains = []
    losses = []
    for i in range(1, len(data)):
        diff = data[i] - data[i - 1]
        gains.append(diff if diff > 0 else 0)
        losses.append(-diff if diff < 0 else 0)
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    rsi = [None]
    if avg_loss == 0:
        rsi.append(100)
    else:
        rs = avg_gain / avg_loss
        rsi.append(100 - (100 / (1 + rs)))
    
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi.append(100 - (100 / (1 + rs)))
    
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    if len(high) < period + 1:
        return [None] * len(high)
    
    plus_dm = []
    minus_dm = []
    
    for i in range(1, len(high)):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        plus_dm.append(high_diff if high_diff > low_diff and high_diff > 0 else 0)
        minus_dm.append(low_diff if low_diff > high_diff and low_diff > 0 else 0)
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = []
    minus_di = []
    for i in range(len(plus_dm)):
        if atr[i + 1] and atr[i + 1] != 0:
            plus_di.append((plus_dm[i] / atr[i + 1]) * 100)
            minus_di.append((minus_dm[i] / atr[i + 1]) * 100)
        else:
            plus_di.append(0)
            minus_di.append(0)
    
    dx = []
    for i in range(len(plus_di)):
        if plus_di[i] + minus_di[i] != 0:
            dx.append((abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100)
        else:
            dx.append(0)
    
    adx = calculate_ema(dx, period)
    return [None] + adx

def calculate_ema_angle(values, index):
    """Calculate EMA angle in degrees"""
    if index < 1 or not values[index] or not values[index - 1]:
        return 0
    slope = values[index] - values[index - 1]
    angle = np.arctan(slope) * (180 / np.pi)
    return abs(angle)

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd = [ema_fast[i] - ema_slow[i] if ema_fast[i] and ema_slow[i] else None for i in range(len(data))]
    signal_line = calculate_ema([m for m in macd if m is not None], signal)
    return macd, signal_line

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(data, period)
    bb_upper = []
    bb_lower = []
    for i in range(len(data)):
        if i < period - 1:
            bb_upper.append(None)
            bb_lower.append(None)
        else:
            window = data[i - period + 1:i + 1]
            std = np.std(window)
            bb_upper.append(sma[i] + (std * std_dev))
            bb_lower.append(sma[i] - (std * std_dev))
    return bb_upper, bb_lower

def find_swing_highs_lows(high, low, lookback=5):
    """Find swing highs and lows"""
    swing_highs = [None] * len(high)
    swing_lows = [None] * len(low)
    
    for i in range(lookback, len(high) - lookback):
        is_high = all(high[i] >= high[i - j] for j in range(1, lookback + 1)) and \
                  all(high[i] >= high[i + j] for j in range(1, lookback + 1))
        is_low = all(low[i] <= low[i - j] for j in range(1, lookback + 1)) and \
                 all(low[i] <= low[i + j] for j in range(1, lookback + 1))
        
        if is_high:
            swing_highs[i] = high[i]
        if is_low:
            swing_lows[i] = low[i]
    
    return swing_highs, swing_lows

# ==================== DATA FETCHING ====================

def fetch_data(ticker, interval, period):
    """Fetch data from yfinance with proper error handling"""
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if df.empty:
            st.error(f"No data retrieved for {ticker}")
            return None
        
        # Handle multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Select OHLCV columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing column: {col}")
                return None
        
        # Handle timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('Asia/Kolkata')
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']] if 'Volume' in df.columns else df[['Open', 'High', 'Low', 'Close']]
        df.columns = ['open', 'high', 'low', 'close', 'volume'] if 'Volume' in df.columns else ['open', 'high', 'low', 'close']
        
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# ==================== STRATEGY LOGIC ====================

def check_ema_crossover_signal(df, config):
    """Check for EMA crossover signals"""
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    ema_fast = calculate_ema(closes, config['ema_fast'])
    ema_slow = calculate_ema(closes, config['ema_slow'])
    
    idx = len(df) - 1
    prev_idx = idx - 1
    
    if idx < 1 or not ema_fast[idx] or not ema_slow[idx]:
        return None, None
    
    # Check crossover
    bullish_cross = ema_fast[prev_idx] <= ema_slow[prev_idx] and ema_fast[idx] > ema_slow[idx]
    bearish_cross = ema_fast[prev_idx] >= ema_slow[prev_idx] and ema_fast[idx] < ema_slow[idx]
    
    if not bullish_cross and not bearish_cross:
        return None, None
    
    # Check angle
    angle = calculate_ema_angle(ema_fast, idx)
    if angle < config['min_angle']:
        return None, None
    
    # Check entry filter
    candle_size = abs(df.iloc[idx]['close'] - df.iloc[idx]['open'])
    
    if config['entry_filter'] == 'Custom Candle (Points)':
        if candle_size < config['custom_points']:
            return None, None
    elif config['entry_filter'] == 'ATR-based Candle':
        atr = calculate_atr(highs, lows, closes, 14)
        if candle_size < atr[idx] * config['atr_multiplier']:
            return None, None
    
    # Check ADX filter
    if config['use_adx_filter']:
        adx = calculate_adx(highs, lows, closes, config['adx_period'])
        if adx[idx] is None or adx[idx] < config['adx_threshold']:
            return None, None
    
    signal_type = 'LONG' if bullish_cross else 'SHORT'
    return signal_type, {'ema_fast': ema_fast[idx], 'ema_slow': ema_slow[idx], 'angle': angle}

def check_price_threshold_signal(df, config):
    """Check for price threshold signals"""
    current_price = df.iloc[-1]['close']
    
    if config['threshold_direction'] == 'LONG (Price >= Threshold)':
        if current_price >= config['price_threshold']:
            return 'LONG', {}
    elif config['threshold_direction'] == 'SHORT (Price >= Threshold)':
        if current_price >= config['price_threshold']:
            return 'SHORT', {}
    elif config['threshold_direction'] == 'LONG (Price <= Threshold)':
        if current_price <= config['price_threshold']:
            return 'LONG', {}
    elif config['threshold_direction'] == 'SHORT (Price <= Threshold)':
        if current_price <= config['price_threshold']:
            return 'SHORT', {}
    
    return None, None

def check_percentage_change_signal(df, config, first_price):
    """Check for percentage change signals"""
    current_price = df.iloc[-1]['close']
    pct_change = ((current_price - first_price) / first_price) * 100
    
    threshold = config['percentage_threshold']
    direction = config['percentage_direction']
    
    if direction == 'BUY on Fall' and pct_change <= -threshold:
        return 'LONG', {'pct_change': pct_change}
    elif direction == 'SELL on Fall' and pct_change <= -threshold:
        return 'SHORT', {'pct_change': pct_change}
    elif direction == 'BUY on Rise' and pct_change >= threshold:
        return 'LONG', {'pct_change': pct_change}
    elif direction == 'SELL on Rise' and pct_change >= threshold:
        return 'SHORT', {'pct_change': pct_change}
    
    return None, None

def check_rsi_adx_ema_signal(df, config):
    """Check for RSI-ADX-EMA combined signals"""
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    rsi = calculate_rsi(closes, 14)
    adx = calculate_adx(highs, lows, closes, 14)
    ema1 = calculate_ema(closes, config['ema_fast'])
    ema2 = calculate_ema(closes, config['ema_slow'])
    
    idx = len(df) - 1
    
    if not all([rsi[idx], adx[idx], ema1[idx], ema2[idx]]):
        return None, None
    
    # SELL: RSI>80, ADX<20, EMA1<EMA2
    if rsi[idx] > 80 and adx[idx] < 20 and ema1[idx] < ema2[idx]:
        return 'SHORT', {'rsi': rsi[idx], 'adx': adx[idx]}
    
    # BUY: RSI<20, ADX>20, EMA1>EMA2
    if rsi[idx] < 20 and adx[idx] > 20 and ema1[idx] > ema2[idx]:
        return 'LONG', {'rsi': rsi[idx], 'adx': adx[idx]}
    
    return None, None

def check_simple_buy_signal(df):
    """Simple buy at close"""
    return 'LONG', {}

def check_simple_sell_signal(df):
    """Simple sell at close"""
    return 'SHORT', {}

# ==================== POSITION MANAGEMENT ====================

def calculate_stop_loss(entry_price, position_type, sl_type, config, df):
    """Calculate stop loss based on type"""
    if sl_type == 'Custom Points':
        if position_type == 'LONG':
            return entry_price - config['sl_points']
        else:
            return entry_price + config['sl_points']
    
    elif sl_type == 'ATR-based':
        atr = calculate_atr(df['high'].values, df['low'].values, df['close'].values, 14)
        atr_value = atr[-1] if atr[-1] else 50
        if position_type == 'LONG':
            return entry_price - (atr_value * config.get('atr_multiplier', 1.5))
        else:
            return entry_price + (atr_value * config.get('atr_multiplier', 1.5))
    
    elif sl_type == 'Current Candle Low/High':
        if position_type == 'LONG':
            return df.iloc[-1]['low']
        else:
            return df.iloc[-1]['high']
    
    elif sl_type == 'Previous Candle Low/High':
        if position_type == 'LONG':
            return df.iloc[-2]['low']
        else:
            return df.iloc[-2]['high']
    
    elif 'Swing' in sl_type:
        swing_highs, swing_lows = find_swing_highs_lows(df['high'].values, df['low'].values)
        if position_type == 'LONG':
            valid_lows = [l for l in swing_lows if l is not None]
            return valid_lows[-1] if valid_lows else entry_price - config['sl_points']
        else:
            valid_highs = [h for h in swing_highs if h is not None]
            return valid_highs[-1] if valid_highs else entry_price + config['sl_points']
    
    else:
        # Default to custom points
        if position_type == 'LONG':
            return entry_price - config['sl_points']
        else:
            return entry_price + config['sl_points']

def calculate_target(entry_price, position_type, target_type, config, df):
    """Calculate target based on type"""
    if target_type == 'Custom Points':
        if position_type == 'LONG':
            return entry_price + config['target_points']
        else:
            return entry_price - config['target_points']
    
    elif target_type == 'ATR-based':
        atr = calculate_atr(df['high'].values, df['low'].values, df['close'].values, 14)
        atr_value = atr[-1] if atr[-1] else 50
        if position_type == 'LONG':
            return entry_price + (atr_value * config.get('atr_multiplier', 2.0))
        else:
            return entry_price - (atr_value * config.get('atr_multiplier', 2.0))
    
    elif target_type == 'Risk-Reward Based':
        sl_distance = abs(entry_price - config['current_sl'])
        rr_ratio = config.get('risk_reward_ratio', 2.0)
        if position_type == 'LONG':
            return entry_price + (sl_distance * rr_ratio)
        else:
            return entry_price - (sl_distance * rr_ratio)
    
    elif target_type == 'Signal-based (reverse EMA crossover)':
        return None  # No fixed target
    
    else:
        # Default to custom points
        if position_type == 'LONG':
            return entry_price + config['target_points']
        else:
            return entry_price - config['target_points']

def update_trailing_stop(current_price, position, config):
    """Update trailing stop loss"""
    sl_type = config['sl_type']
    
    if 'Trailing' not in sl_type:
        return position['stop_loss']
    
    if position['type'] == 'LONG':
        new_sl = current_price - config['sl_points']
        return max(position['stop_loss'], new_sl)
    else:
        new_sl = current_price + config['sl_points']
        return min(position['stop_loss'], new_sl)

def check_exit_conditions(current_price, current_candle, position, config, df):
    """Check if position should be exited"""
    exit_signal = None
    exit_reason = None
    
    # Check stop loss
    if position['type'] == 'LONG':
        if current_price <= position['stop_loss']:
            exit_signal = current_price
            exit_reason = 'Stop Loss Hit'
    else:
        if current_price >= position['stop_loss']:
            exit_signal = current_price
            exit_reason = 'Stop Loss Hit'
    
    # Check target (if not signal-based)
    if position['target'] is not None:
        if position['type'] == 'LONG':
            if current_price >= position['target']:
                exit_signal = current_price
                exit_reason = 'Target Hit'
        else:
            if current_price <= position['target']:
                exit_signal = current_price
                exit_reason = 'Target Hit'
    
    # Check signal-based exit
    if config['sl_type'] == 'Signal-based (reverse EMA crossover)' or \
       config['target_type'] == 'Signal-based (reverse EMA crossover)':
        closes = df['close'].values
        ema_fast = calculate_ema(closes, config['ema_fast'])
        ema_slow = calculate_ema(closes, config['ema_slow'])
        
        idx = len(df) - 1
        prev_idx = idx - 1
        
        if idx >= 1 and ema_fast[idx] and ema_slow[idx]:
            if position['type'] == 'LONG':
                # Exit on bearish crossover
                if ema_fast[prev_idx] >= ema_slow[prev_idx] and ema_fast[idx] < ema_slow[idx]:
                    exit_signal = current_price
                    exit_reason = 'Signal-based Exit (Reverse Crossover)'
            else:
                # Exit on bullish crossover
                if ema_fast[prev_idx] <= ema_slow[prev_idx] and ema_fast[idx] > ema_slow[idx]:
                    exit_signal = current_price
                    exit_reason = 'Signal-based Exit (Reverse Crossover)'
    
    return exit_signal, exit_reason

def add_trade_log(message):
    """Add entry to trade log"""
    timestamp = datetime.now(pytz.timezone('Asia/Kolkata'))
    st.session_state.trade_logs.insert(0, {
        'timestamp': timestamp,
        'message': message
    })
    if len(st.session_state.trade_logs) > 50:
        st.session_state.trade_logs = st.session_state.trade_logs[:50]

# ==================== MAIN APP ====================

def main():
    st.title("🚀 Professional Quantitative Trading System")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Asset Selection
        st.subheader("Asset Selection")
        asset_name = st.selectbox("Select Asset", list(ASSETS.keys()))
        
        if asset_name == 'Custom':
            custom_ticker = st.text_input("Enter Custom Ticker")
            ticker = custom_ticker if custom_ticker else '^NSEI'
        else:
            ticker = ASSETS[asset_name]
        
        # Timeframe Selection
        st.subheader("Timeframe")
        interval = st.selectbox("Interval", list(PERIOD_MAP.keys()))
        available_periods = PERIOD_MAP[interval]
        period = st.selectbox("Period", available_periods)
        
        # Trading Parameters
        st.subheader("Trading Parameters")
        quantity = st.number_input("Quantity", min_value=1, value=1)
        mode = st.selectbox("Mode", ["Live Trading", "Backtesting"])
        
        # Strategy Selection
        st.subheader("Strategy")
        strategy = st.selectbox("Select Strategy", [
            'EMA Crossover',
            'Simple Buy',
            'Simple Sell',
            'Price Crosses Threshold',
            'RSI-ADX-EMA',
            'Percentage Change',
            'AI Price Action',
            'Custom Strategy Builder'
        ])
        
        # Stop Loss & Target
        st.subheader("Risk Management")
        sl_type = st.selectbox("Stop Loss Type", [
            'Custom Points',
            'Trailing SL (Points)',
            'Trailing SL + Current Candle',
            'Trailing SL + Previous Candle',
            'Trailing SL + Current Swing',
            'Trailing SL + Previous Swing',
            'Trailing SL + Signal Based',
            'Volatility-Adjusted Trailing SL',
            'Break-even After 50% Target',
            'ATR-based',
            'Current Candle Low/High',
            'Previous Candle Low/High',
            'Current Swing Low/High',
            'Previous Swing Low/High',
            'Signal-based (reverse EMA crossover)'
        ])
        
        target_type = st.selectbox("Target Type", [
            'Custom Points',
            'Trailing Target (Points)',
            'Trailing Target + Signal Based',
            '50% Exit at Target',
            'Current Candle Low/High',
            'Previous Candle Low/High',
            'Current Swing Low/High',
            'Previous Swing Low/High',
            'ATR-based',
            'Risk-Reward Based',
            'Signal-based (reverse EMA crossover)'
        ])
        
        sl_points = st.number_input("SL Points", min_value=1, value=10)
        target_points = st.number_input("Target Points", min_value=1, value=20)
        
        # Strategy-specific parameters
        config = {
            'sl_type': sl_type,
            'target_type': target_type,
            'sl_points': sl_points,
            'target_points': target_points,
            'quantity': quantity
        }
        
        if strategy == 'EMA Crossover':
            st.subheader("EMA Crossover Settings")
            ema_fast = st.number_input("EMA Fast", min_value=1, value=9)
            ema_slow = st.number_input("EMA Slow", min_value=1, value=15)
            min_angle = st.number_input("Min Angle (degrees)", min_value=0.0, value=1.0, step=0.1)
            entry_filter = st.selectbox("Entry Filter", [
                'Simple Crossover',
                'Custom Candle (Points)',
                'ATR-based Candle'
            ])
            
            custom_points = 5
            atr_multiplier = 1.5
            if entry_filter == 'Custom Candle (Points)':
                custom_points = st.number_input("Custom Points", min_value=1, value=5)
            elif entry_filter == 'ATR-based Candle':
                atr_multiplier = st.number_input("ATR Multiplier", min_value=0.1, value=1.5, step=0.1)
            
            use_adx_filter = st.checkbox("Use ADX Filter")
            adx_period = 14
            adx_threshold = 25
            if use_adx_filter:
                adx_period = st.number_input("ADX Period", min_value=1, value=14)
                adx_threshold = st.number_input("ADX Threshold", min_value=1, value=25)
            
            config.update({
                'ema_fast': ema_fast,
                'ema_slow': ema_slow,
                'min_angle': min_angle,
                'entry_filter': entry_filter,
                'custom_points': custom_points,
                'atr_multiplier': atr_multiplier,
                'use_adx_filter': use_adx_filter,
                'adx_period': adx_period,
                'adx_threshold': adx_threshold
            })
        
        elif strategy == 'Price Crosses Threshold':
            st.subheader("Threshold Settings")
            price_threshold = st.number_input("Price Threshold", min_value=0.0, value=18000.0)
            threshold_direction = st.selectbox("Direction", [
                'LONG (Price >= Threshold)',
                'SHORT (Price >= Threshold)',
                'LONG (Price <= Threshold)',
                'SHORT (Price <= Threshold)'
            ])
            config.update({
                'price_threshold': price_threshold,
                'threshold_direction': threshold_direction
            })
        
        elif strategy == 'Percentage Change':
            st.subheader("Percentage Change Settings")
            percentage_threshold = st.number_input("Percentage Threshold (%)", min_value=0.001, value=0.01, step=0.001, format="%.3f")
            percentage_direction = st.selectbox("Direction", [
                'BUY on Fall',
                'SELL on Fall',
                'BUY on Rise',
                'SELL on Rise'
            ])
            config.update({
                'percentage_threshold': percentage_threshold,
                'percentage_direction': percentage_direction
            })
        
        elif strategy == 'RSI-ADX-EMA':
            st.subheader("RSI-ADX-EMA Settings")
            ema_fast = st.number_input("EMA Fast", min_value=1, value=9)
            ema_slow = st.number_input("EMA Slow", min_value=1, value=21)
            config.update({
                'ema_fast': ema_fast,
                'ema_slow': ema_slow
            })
        
        # Broker Integration
        st.subheader("🔌 Broker Integration (Optional)")
        use_broker = st.checkbox("Enable Broker API")
        if use_broker:
            client_id = st.text_input("Client ID", type="password")
            api_token = st.text_input("API Token", type="password")
            expiry_date = st.date_input("Expiry Date")
            strike_price = st.number_input("Strike Price", min_value=0.0, value=0.0)
            option_type = st.selectbox("Option Type", ['CE', 'PE', 'FUT', 'EQUITY'])
            order_type = st.selectbox("Order Type", ['MARKET', 'LIMIT'])
            limit_price = 0.0
            if order_type == 'LIMIT':
                limit_price = st.number_input("Limit Price", min_value=0.0, value=0.0)
    
    # Main content area
    if mode == "Live Trading":
        st.header("📊 Live Trading Dashboard")
        
        # Trading controls
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("▶️ Start Trading", use_container_width=True, type="primary"):
                st.session_state.trading_active = True
                st.session_state.position = None
                st.session_state.partial_exit_done = False
                st.session_state.breakeven_activated = False
                add_trade_log("Trading started")
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop Trading", use_container_width=True, type="secondary"):
                if st.session_state.position:
                    # Close position
                    if st.session_state.current_data is not None:
                        current_price = st.session_state.current_data.iloc[-1]['close']
                        pnl = calculate_pnl(
                            st.session_state.position['entry_price'],
                            current_price,
                            st.session_state.position['type'],
                            st.session_state.position['quantity']
                        )
                        
                        trade_record = {
                            'entry_time': st.session_state.position['entry_time'],
                            'exit_time': datetime.now(pytz.timezone('Asia/Kolkata')),
                            'type': st.session_state.position['type'],
                            'entry_price': st.session_state.position['entry_price'],
                            'exit_price': current_price,
                            'quantity': st.session_state.position['quantity'],
                            'pnl': pnl,
                            'exit_reason': 'Manual Close',
                            'stop_loss': st.session_state.position['stop_loss'],
                            'target': st.session_state.position['target']
                        }
                        st.session_state.trade_history.append(trade_record)
                        add_trade_log(f"Position closed manually at {current_price:.2f} | P&L: {pnl:.2f}")
                
                st.session_state.trading_active = False
                st.session_state.position = None
                add_trade_log("Trading stopped")
                st.rerun()
        
        with col3:
            status_color = "🟢" if st.session_state.trading_active else "🔴"
            status_text = "ACTIVE" if st.session_state.trading_active else "INACTIVE"
            st.markdown(f"### {status_color} Status: {status_text}")
        
        st.markdown("---")
        
        # Tabs for different views - MOVED TO TOP
        tab1, tab2, tab3 = st.tabs(["📈 Live Dashboard", "📜 Trade History", "📝 Trade Logs"])
        
        with tab1:
            # Live metrics display
            metrics_placeholder = st.empty()
            
            # Chart placeholder
            chart_placeholder = st.empty()
            
            # Configuration display at bottom
            config_placeholder = st.empty()
            
            # Auto-refresh loop for live trading
            if st.session_state.trading_active:
                # Fetch latest data
                df = fetch_data(ticker, interval, period)
                
                if df is not None and not df.empty:
                    st.session_state.current_data = df
                    
                    # Store first candle price for percentage change strategy
                    if st.session_state.first_candle_price is None:
                        st.session_state.first_candle_price = df.iloc[0]['close']
                    
                    current_price = df.iloc[-1]['close']
                    current_candle = df.iloc[-1]
                    
                    # Check for entry signal if no position
                    if st.session_state.position is None:
                        signal_type = None
                        signal_data = None
                        
                        if strategy == 'EMA Crossover':
                            signal_type, signal_data = check_ema_crossover_signal(df, config)
                        elif strategy == 'Simple Buy':
                            signal_type, signal_data = check_simple_buy_signal(df)
                        elif strategy == 'Simple Sell':
                            signal_type, signal_data = check_simple_sell_signal(df)
                        elif strategy == 'Price Crosses Threshold':
                            signal_type, signal_data = check_price_threshold_signal(df, config)
                        elif strategy == 'Percentage Change':
                            signal_type, signal_data = check_percentage_change_signal(
                                df, config, st.session_state.first_candle_price
                            )
                        elif strategy == 'RSI-ADX-EMA':
                            signal_type, signal_data = check_rsi_adx_ema_signal(df, config)
                        
                        if signal_type:
                            # Enter position
                            entry_price = current_price
                            stop_loss = calculate_stop_loss(entry_price, signal_type, sl_type, config, df)
                            config['current_sl'] = stop_loss
                            target = calculate_target(entry_price, signal_type, target_type, config, df)
                            
                            st.session_state.position = {
                                'type': signal_type,
                                'entry_price': entry_price,
                                'entry_time': datetime.now(pytz.timezone('Asia/Kolkata')),
                                'stop_loss': stop_loss,
                                'target': target,
                                'quantity': quantity,
                                'highest_price': entry_price,
                                'lowest_price': entry_price
                            }
                            
                            add_trade_log(f"Entered {signal_type} position at {entry_price:.2f}")
                    
                    # Update position if exists
                    if st.session_state.position:
                        # Update trailing values
                        st.session_state.position['highest_price'] = max(
                            st.session_state.position['highest_price'], current_price
                        )
                        st.session_state.position['lowest_price'] = min(
                            st.session_state.position['lowest_price'], current_price
                        )
                        
                        # Update trailing stop loss
                        if 'Trailing' in sl_type:
                            new_sl = update_trailing_stop(current_price, st.session_state.position, config)
                            if new_sl != st.session_state.position['stop_loss']:
                                st.session_state.position['stop_loss'] = new_sl
                                add_trade_log(f"Stop loss trailed to {new_sl:.2f}")
                        
                        # Check break-even condition
                        if sl_type == 'Break-even After 50% Target' and not st.session_state.breakeven_activated:
                            if st.session_state.position['target']:
                                target_distance = abs(st.session_state.position['target'] - st.session_state.position['entry_price'])
                                halfway = st.session_state.position['entry_price']
                                
                                if st.session_state.position['type'] == 'LONG':
                                    halfway += target_distance * 0.5
                                    if current_price >= halfway:
                                        st.session_state.position['stop_loss'] = st.session_state.position['entry_price']
                                        st.session_state.breakeven_activated = True
                                        add_trade_log("Stop loss moved to break-even")
                                else:
                                    halfway -= target_distance * 0.5
                                    if current_price <= halfway:
                                        st.session_state.position['stop_loss'] = st.session_state.position['entry_price']
                                        st.session_state.breakeven_activated = True
                                        add_trade_log("Stop loss moved to break-even")
                        
                        # Check exit conditions
                        exit_price, exit_reason = check_exit_conditions(
                            current_price, current_candle, st.session_state.position, config, df
                        )
                        
                        if exit_price:
                            pnl = calculate_pnl(
                                st.session_state.position['entry_price'],
                                exit_price,
                                st.session_state.position['type'],
                                st.session_state.position['quantity']
                            )
                            
                            trade_record = {
                                'entry_time': st.session_state.position['entry_time'],
                                'exit_time': datetime.now(pytz.timezone('Asia/Kolkata')),
                                'type': st.session_state.position['type'],
                                'entry_price': st.session_state.position['entry_price'],
                                'exit_price': exit_price,
                                'quantity': st.session_state.position['quantity'],
                                'pnl': pnl,
                                'exit_reason': exit_reason,
                                'stop_loss': st.session_state.position['stop_loss'],
                                'target': st.session_state.position['target']
                            }
                            st.session_state.trade_history.append(trade_record)
                            add_trade_log(f"Position exited at {exit_price:.2f} | P&L: {pnl:.2f} | Reason: {exit_reason}")
                            
                            st.session_state.position = None
                            st.session_state.partial_exit_done = False
                            st.session_state.breakeven_activated = False
                    
                    # Display live metrics
                    with metrics_placeholder.container():
                        st.subheader("📊 Live Metrics")
                        
                        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                        with met_col1:
                            st.metric("Current Price", f"{current_price:.2f}")
                        
                        if st.session_state.position:
                            with met_col2:
                                pnl = calculate_pnl(
                                    st.session_state.position['entry_price'],
                                    current_price,
                                    st.session_state.position['type'],
                                    st.session_state.position['quantity']
                                )
                                st.metric("Unrealized P&L", f"{pnl:.2f}", 
                                         delta=f"{pnl:.2f}",
                                         delta_color="normal" if pnl >= 0 else "inverse")
                            
                            with met_col3:
                                st.metric("Position", st.session_state.position['type'])
                            
                            with met_col4:
                                st.metric("Entry Price", f"{st.session_state.position['entry_price']:.2f}")
                            
                            # Second row of metrics
                            met_col5, met_col6, met_col7, met_col8 = st.columns(4)
                            
                            with met_col5:
                                st.metric("Stop Loss", f"{st.session_state.position['stop_loss']:.2f}")
                            
                            with met_col6:
                                if st.session_state.position['target']:
                                    st.metric("Target", f"{st.session_state.position['target']:.2f}")
                                else:
                                    st.metric("Target", "Signal-based")
                            
                            with met_col7:
                                st.metric("Quantity", st.session_state.position['quantity'])
                            
                            with met_col8:
                                duration = datetime.now(pytz.timezone('Asia/Kolkata')) - st.session_state.position['entry_time']
                                st.metric("Duration", str(duration).split('.')[0])
                            
                            # Third row - Highest/Lowest
                            met_col9, met_col10 = st.columns(2)
                            with met_col9:
                                st.metric("Highest", f"{st.session_state.position['highest_price']:.2f}")
                            with met_col10:
                                st.metric("Lowest", f"{st.session_state.position['lowest_price']:.2f}")
                        
                        # Strategy-specific metrics
                        if strategy == 'EMA Crossover':
                            closes = df['close'].values
                            ema_fast_vals = calculate_ema(closes, config['ema_fast'])
                            ema_slow_vals = calculate_ema(closes, config['ema_slow'])
                            
                            st.markdown("---")
                            st.markdown("**📈 Indicators:**")
                            ind_col1, ind_col2, ind_col3 = st.columns(3)
                            
                            with ind_col1:
                                if ema_fast_vals[-1]:
                                    st.metric(f"EMA{config['ema_fast']}", f"{ema_fast_vals[-1]:.2f}")
                            with ind_col2:
                                if ema_slow_vals[-1]:
                                    st.metric(f"EMA{config['ema_slow']}", f"{ema_slow_vals[-1]:.2f}")
                            with ind_col3:
                                rsi = calculate_rsi(closes, 14)
                                if rsi[-1]:
                                    st.metric("RSI", f"{rsi[-1]:.2f}")
                        
                        elif strategy == 'Percentage Change':
                            pct_change = ((current_price - st.session_state.first_candle_price) / st.session_state.first_candle_price) * 100
                            st.markdown("---")
                            st.metric("% Change from First Candle", f"{pct_change:.3f}%",
                                     delta=f"{pct_change:.3f}%",
                                     delta_color="normal" if pct_change >= 0 else "inverse")
                    
                    # Display chart
                    with chart_placeholder.container():
                        st.markdown("---")
                        st.subheader("📈 Live Price Chart")
                        fig = create_live_chart(df, st.session_state.position, config, strategy)
                        st.plotly_chart(fig, use_container_width=True, key=f"live_chart_{time.time()}")
                    
                    # Display configuration at bottom
                    with config_placeholder.container():
                        st.markdown("---")
                        st.subheader("📋 Active Configuration")
                        config_col1, config_col2, config_col3 = st.columns(3)
                        
                        with config_col1:
                            st.metric("Asset", asset_name)
                            st.metric("Interval", interval)
                            st.metric("Period", period)
                            st.metric("Quantity", quantity)
                        
                        with config_col2:
                            st.metric("Strategy", strategy)
                            st.metric("SL Type", sl_type)
                            st.metric("Target Type", target_type)
                        
                        with config_col3:
                            st.metric("SL Points", sl_points)
                            st.metric("Target Points", target_points)
                            if strategy == 'EMA Crossover':
                                st.metric("EMA Fast/Slow", f"{config['ema_fast']}/{config['ema_slow']}")
                
                # Add delay and rerun
                time.sleep(random.uniform(1.0, 1.5))
                st.rerun()
            
            else:
                st.info("Click 'Start Trading' to begin live trading")
        
        with tab2:
            display_trade_history()
        
        with tab3:
            display_trade_logs()
    
    else:  # Backtesting mode
        st.header("🔬 Backtesting Mode")
        
        tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "📊 Backtest Results", "📈 Market Analysis"])
        
        with tab1:
            st.subheader("Current Configuration")
            st.write(f"**Asset:** {asset_name} ({ticker})")
            st.write(f"**Interval:** {interval}")
            st.write(f"**Period:** {period}")
            st.write(f"**Strategy:** {strategy}")
            st.write(f"**Stop Loss:** {sl_type} ({sl_points} points)")
            st.write(f"**Target:** {target_type} ({target_points} points)")
            
            if strategy == 'EMA Crossover':
                st.write(f"**EMA Fast:** {config['ema_fast']}")
                st.write(f"**EMA Slow:** {config['ema_slow']}")
                st.write(f"**Min Angle:** {config['min_angle']}°")
        
        with tab2:
            if st.button("🚀 Run Backtest", type="primary"):
                with st.spinner("Running backtest..."):
                    run_backtest(ticker, interval, period, strategy, config)
            
            if st.session_state.trade_history:
                display_backtest_results(st.session_state.current_data, config, strategy)
        
        with tab3:
            if st.button("📊 Analyze Market Data", type="primary"):
                analyze_market_data(ticker, interval, period)

def calculate_pnl(entry_price, exit_price, position_type, quantity):
    """Calculate P&L"""
    if position_type == 'LONG':
        return (exit_price - entry_price) * quantity
    else:
        return (entry_price - exit_price) * quantity

def create_live_chart(df, position, config, strategy):
    """Create live trading chart with indicators"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ))
    
    # Add EMAs if EMA strategy
    if strategy == 'EMA Crossover':
        closes = df['close'].values
        ema_fast = calculate_ema(closes, config['ema_fast'])
        ema_slow = calculate_ema(closes, config['ema_slow'])
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=ema_fast,
            mode='lines',
            name=f"EMA {config['ema_fast']}",
            line=dict(color='cyan', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=ema_slow,
            mode='lines',
            name=f"EMA {config['ema_slow']}",
            line=dict(color='magenta', width=1)
        ))
    
    # Add position lines
    if position:
        # Entry line
        fig.add_hline(
            y=position['entry_price'],
            line_dash="dash",
            line_color="blue",
            annotation_text="Entry",
            annotation_position="right"
        )
        
        # Stop loss line
        fig.add_hline(
            y=position['stop_loss'],
            line_dash="dash",
            line_color="red",
            annotation_text="Stop Loss",
            annotation_position="right"
        )
        
        # Target line (if exists)
        if position['target']:
            fig.add_hline(
                y=position['target'],
                line_dash="dash",
                line_color="green",
                annotation_text="Target",
                annotation_position="right"
            )
    
    fig.update_layout(
        title="Live Price Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        height=500,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def display_trade_history():
    """Display trade history tab"""
    st.subheader("Trade History")
    
    if st.session_state.trade_history:
        # Calculate statistics
        total_trades = len(st.session_state.trade_history)
        winning_trades = len([t for t in st.session_state.trade_history if t['pnl'] > 0])
        losing_trades = len([t for t in st.session_state.trade_history if t['pnl'] < 0])
        total_pnl = sum([t['pnl'] for t in st.session_state.trade_history])
        accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Trades", total_trades)
        col2.metric("Winning", winning_trades)
        col3.metric("Losing", losing_trades)
        col4.metric("Accuracy", f"{accuracy:.1f}%")
        col5.metric("Total P&L", f"{total_pnl:.2f}", 
                   delta=f"{total_pnl:.2f}",
                   delta_color="normal" if total_pnl >= 0 else "inverse")
        
        st.markdown("---")
        
        # Display individual trades
        for i, trade in enumerate(reversed(st.session_state.trade_history)):
            with st.expander(f"Trade #{total_trades - i} - {trade['type']} | P&L: {trade['pnl']:.2f}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Entry Time:** {trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Entry Price:** {trade['entry_price']:.2f}")
                    st.write(f"**Stop Loss:** {trade['stop_loss']:.2f}")
                    st.write(f"**Target:** {trade['target']:.2f}" if trade['target'] else "**Target:** Signal-based")
                
                with col2:
                    st.write(f"**Exit Time:** {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Exit Price:** {trade['exit_price']:.2f}")
                    st.write(f"**Quantity:** {trade['quantity']}")
                    st.write(f"**Exit Reason:** {trade['exit_reason']}")
                
                duration = trade['exit_time'] - trade['entry_time']
                st.write(f"**Duration:** {str(duration).split('.')[0]}")
                
                pnl_color = "green" if trade['pnl'] > 0 else "red"
                st.markdown(f"**P&L:** <span style='color:{pnl_color};font-size:20px;font-weight:bold'>{trade['pnl']:.2f}</span>", unsafe_allow_html=True)
    
    else:
        st.info("No trades yet")

def display_trade_logs():
    """Display trade logs tab"""
    st.subheader("Trade Logs (Last 50)")
    
    if st.session_state.trade_logs:
        for log in st.session_state.trade_logs:
            st.text(f"[{log['timestamp'].strftime('%H:%M:%S')}] {log['message']}")
    else:
        st.info("No logs yet")

def run_backtest(ticker, interval, period, strategy, config):
    """Run backtest on historical data"""
    df = fetch_data(ticker, interval, period)
    
    if df is None or df.empty:
        st.error("Failed to fetch data for backtesting")
        return
    
    st.session_state.current_data = df
    st.session_state.trade_history = []
    st.session_state.first_candle_price = df.iloc[0]['close']
    
    position = None
    
    for i in range(20, len(df)):  # Start after enough data for indicators
        current_df = df.iloc[:i+1]
        current_price = current_df.iloc[-1]['close']
        current_candle = current_df.iloc[-1]
        
        if position is None:
            # Check for entry
            signal_type = None
            
            if strategy == 'EMA Crossover':
                signal_type, _ = check_ema_crossover_signal(current_df, config)
            elif strategy == 'Simple Buy':
                signal_type, _ = check_simple_buy_signal(current_df)
            elif strategy == 'Simple Sell':
                signal_type, _ = check_simple_sell_signal(current_df)
            elif strategy == 'Price Crosses Threshold':
                signal_type, _ = check_price_threshold_signal(current_df, config)
            elif strategy == 'Percentage Change':
                signal_type, _ = check_percentage_change_signal(
                    current_df, config, st.session_state.first_candle_price
                )
            elif strategy == 'RSI-ADX-EMA':
                signal_type, _ = check_rsi_adx_ema_signal(current_df, config)
            
            if signal_type:
                entry_price = current_price
                stop_loss = calculate_stop_loss(entry_price, signal_type, config['sl_type'], config, current_df)
                config['current_sl'] = stop_loss
                target = calculate_target(entry_price, signal_type, config['target_type'], config, current_df)
                
                position = {
                    'type': signal_type,
                    'entry_price': entry_price,
                    'entry_time': current_df.index[-1],
                    'stop_loss': stop_loss,
                    'target': target,
                    'quantity': config['quantity'],
                    'highest_price': entry_price,
                    'lowest_price': entry_price
                }
        
        else:
            # Update position
            position['highest_price'] = max(position['highest_price'], current_price)
            position['lowest_price'] = min(position['lowest_price'], current_price)
            
            # Update trailing stop
            if 'Trailing' in config['sl_type']:
                position['stop_loss'] = update_trailing_stop(current_price, position, config)
            
            # Check exit
            exit_price, exit_reason = check_exit_conditions(
                current_price, current_candle, position, config, current_df
            )
            
            if exit_price:
                pnl = calculate_pnl(
                    position['entry_price'],
                    exit_price,
                    position['type'],
                    position['quantity']
                )
                
                trade_record = {
                    'entry_time': position['entry_time'],
                    'exit_time': current_df.index[-1],
                    'type': position['type'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'quantity': position['quantity'],
                    'pnl': pnl,
                    'exit_reason': exit_reason,
                    'stop_loss': position['stop_loss'],
                    'target': position['target']
                }
                st.session_state.trade_history.append(trade_record)
                position = None
    
    st.success(f"Backtest completed! {len(st.session_state.trade_history)} trades executed.")

def display_backtest_results(df, config, strategy):
    """Display backtest results"""
    st.subheader("Backtest Results")
    
    if not st.session_state.trade_history:
        st.warning("No trades executed in backtest")
        return
    
    # Calculate statistics
    total_trades = len(st.session_state.trade_history)
    winning_trades = len([t for t in st.session_state.trade_history if t['pnl'] > 0])
    losing_trades = len([t for t in st.session_state.trade_history if t['pnl'] < 0])
    total_pnl = sum([t['pnl'] for t in st.session_state.trade_history])
    accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    avg_win = np.mean([t['pnl'] for t in st.session_state.trade_history if t['pnl'] > 0]) if winning_trades > 0 else 0
    avg_loss = np.mean([t['pnl'] for t in st.session_state.trade_history if t['pnl'] < 0]) if losing_trades > 0 else 0
    
    durations = [(t['exit_time'] - t['entry_time']).total_seconds() / 60 for t in st.session_state.trade_history]
    avg_duration = np.mean(durations) if durations else 0
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Trades", total_trades)
    col2.metric("Winning", winning_trades)
    col3.metric("Losing", losing_trades)
    col4.metric("Accuracy", f"{accuracy:.1f}%")
    col5.metric("Total P&L", f"{total_pnl:.2f}",
               delta=f"{total_pnl:.2f}",
               delta_color="normal" if total_pnl >= 0 else "inverse")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Win", f"{avg_win:.2f}")
    col2.metric("Avg Loss", f"{avg_loss:.2f}")
    col3.metric("Avg Duration (min)", f"{avg_duration:.1f}")
    
    st.markdown("---")
    
    # P&L Chart
    st.subheader("P&L by Trade")
    pnl_data = [{'Trade': i+1, 'P&L': t['pnl']} for i, t in enumerate(st.session_state.trade_history)]
    pnl_df = pd.DataFrame(pnl_data)
    
    fig = go.Figure()
    colors = ['green' if pnl > 0 else 'red' for pnl in pnl_df['P&L']]
    fig.add_trace(go.Bar(x=pnl_df['Trade'], y=pnl_df['P&L'], marker_color=colors, name='P&L'))
    fig.update_layout(
        title="P&L Distribution",
        xaxis_title="Trade Number",
        yaxis_title="P&L",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Entry/Exit Visualization Chart
    st.subheader("Entry/Exit Visualization")
    
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ))
    
    # Add EMAs if strategy uses them
    if strategy == 'EMA Crossover':
        closes = df['close'].values
        ema_fast = calculate_ema(closes, config['ema_fast'])
        ema_slow = calculate_ema(closes, config['ema_slow'])
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=ema_fast,
            mode='lines',
            name=f"EMA {config['ema_fast']}",
            line=dict(color='cyan', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=ema_slow,
            mode='lines',
            name=f"EMA {config['ema_slow']}",
            line=dict(color='magenta', width=1)
        ))
    
    # Add entry/exit markers
    for trade in st.session_state.trade_history:
        # Entry marker
        if trade['type'] == 'LONG':
            fig.add_trace(go.Scatter(
                x=[trade['entry_time']],
                y=[trade['entry_price']],
                mode='markers',
                marker=dict(symbol='triangle-up', size=15, color='green'),
                name='LONG Entry',
                showlegend=False
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[trade['entry_time']],
                y=[trade['entry_price']],
                mode='markers',
                marker=dict(symbol='triangle-down', size=15, color='red'),
                name='SHORT Entry',
                showlegend=False
            ))
        
        # Exit marker
        exit_color = 'lime' if trade['pnl'] > 0 else 'red'
        exit_symbol = 'circle' if trade['pnl'] > 0 else 'x'
        fig.add_trace(go.Scatter(
            x=[trade['exit_time']],
            y=[trade['exit_price']],
            mode='markers',
            marker=dict(symbol=exit_symbol, size=12, color=exit_color),
            name='Exit',
            showlegend=False
        ))
        
        # Connect entry to exit
        fig.add_trace(go.Scatter(
            x=[trade['entry_time'], trade['exit_time']],
            y=[trade['entry_price'], trade['exit_price']],
            mode='lines',
            line=dict(color=exit_color, width=1, dash='dot'),
            showlegend=False
        ))
    
    fig.update_layout(
        title="Entry/Exit Points on Price Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Legend explanation
    st.markdown("""
    **Legend:**
    - 🟢 Green Triangle Up: LONG Entry
    - 🔴 Red Triangle Down: SHORT Entry
    - 🟢 Green Circle: Profitable Exit
    - ❌ Red X: Loss Exit
    - Dotted lines connect entry to exit
    """)
    
    st.markdown("---")
    
    # Trade details
    st.subheader("Trade Details")
    for i, trade in enumerate(st.session_state.trade_history):
        with st.expander(f"Trade #{i+1} - {trade['type']} | P&L: {trade['pnl']:.2f}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Entry Time:** {trade['entry_time']}")
                st.write(f"**Entry Price:** {trade['entry_price']:.2f}")
                st.write(f"**Stop Loss:** {trade['stop_loss']:.2f}")
                st.write(f"**Target:** {trade['target']:.2f}" if trade['target'] else "**Target:** Signal-based")
            
            with col2:
                st.write(f"**Exit Time:** {trade['exit_time']}")
                st.write(f"**Exit Price:** {trade['exit_price']:.2f}")
                st.write(f"**Quantity:** {trade['quantity']}")
                st.write(f"**Exit Reason:** {trade['exit_reason']}")
            
            duration = trade['exit_time'] - trade['entry_time']
            st.write(f"**Duration:** {str(duration).split('.')[0]}")
            
            pnl_color = "green" if trade['pnl'] > 0 else "red"
            st.markdown(f"**P&L:** <span style='color:{pnl_color};font-size:20px;font-weight:bold'>{trade['pnl']:.2f}</span>", unsafe_allow_html=True)

def analyze_market_data(ticker, interval, period):
    """Analyze market data"""
    st.subheader("Market Data Analysis")
    
    # Fetch data
    df = fetch_data(ticker, interval, period)
    
    if df is None or df.empty:
        st.error("Failed to fetch data")
        return
    
    st.session_state.current_data = df
    
    # Calculate statistics - FIXED
    df['change_points'] = df['close'] - df['close'].shift(1)  # Fixed: compare with previous close
    df['change_pct'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * 100  # Fixed
    
    # Drop first row with NaN
    df = df.dropna()
    
    # Fix day of week for datetime index
    if isinstance(df.index, pd.DatetimeIndex):
        df['day_of_week'] = df.index.strftime('%A')  # Fixed: proper day name extraction
    else:
        df['day_of_week'] = pd.to_datetime(df.index).strftime('%A')
    
    # Display data table
    st.subheader("📋 Data Table")
    display_df = df[['open', 'high', 'low', 'close', 'change_points', 'change_pct', 'day_of_week']].copy()
    display_df.columns = ['Open', 'High', 'Low', 'Close', 'Change (Points)', 'Change (%)', 'Day of Week']
    
    # Color code changes
    def color_change(val):
        if isinstance(val, (int, float)):
            color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
            return f'color: {color}'
        return ''
    
    styled_df = display_df.style.applymap(color_change, subset=['Change (Points)', 'Change (%)'])
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Change in Points Over Time")
        fig1 = go.Figure()
        colors = ['green' if x > 0 else 'red' for x in df['change_points']]
        fig1.add_trace(go.Bar(
            x=df.index,
            y=df['change_points'],
            marker_color=colors,
            name='Change (Points)'
        ))
        fig1.update_layout(
            xaxis_title="Time",
            yaxis_title="Change (Points)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Change in Percentage Over Time")
        fig2 = go.Figure()
        colors = ['green' if x > 0 else 'red' for x in df['change_pct']]
        fig2.add_trace(go.Bar(
            x=df.index,
            y=df['change_pct'],
            marker_color=colors,
            name='Change (%)'
        ))
        fig2.update_layout(
            xaxis_title="Time",
            yaxis_title="Change (%)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Summary Statistics
    st.subheader("📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Max Price", f"{df['high'].max():.2f}")
        st.metric("Min Price", f"{df['low'].min():.2f}")
    
    with col2:
        st.metric("Average Close", f"{df['close'].mean():.2f}")
        volatility = df['close'].std() / df['close'].mean() * 100
        st.metric("Volatility %", f"{volatility:.2f}%")
    
    with col3:
        total_change_points = df['close'].iloc[-1] - df['close'].iloc[0]
        st.metric("Total Change (Points)", f"{total_change_points:.2f}",
                 delta=f"{total_change_points:.2f}",
                 delta_color="normal" if total_change_points >= 0 else "inverse")
        
        total_change_pct = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
        st.metric("Total Change (%)", f"{total_change_pct:.2f}%")
    
    with col4:
        avg_change = df['change_points'].mean()
        st.metric("Avg Change", f"{avg_change:.2f}")
        
        win_rate = len(df[df['change_points'] > 0]) / len(df) * 100
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_gain = df['change_points'].max()
        st.metric("Max Gain (Points)", f"{max_gain:.2f}")
    
    with col2:
        max_loss = df['change_points'].min()
        st.metric("Max Loss (Points)", f"{max_loss:.2f}")
    
    st.markdown("---")
    
    # Heatmaps for selected period
    st.subheader("📅 Period Heatmaps")
    
    try:
        # Returns heatmap
        if len(df) >= 30:
            # Group by day of week and hour (if intraday data)
            if interval in ['1m', '5m', '15m', '30m', '1h', '4h']:
                df['hour'] = df.index.hour
                pivot_returns = df.pivot_table(
                    values='change_pct',
                    index='day_of_week',
                    columns='hour',
                    aggfunc='mean'
                )
                
                fig_heat1 = go.Figure(data=go.Heatmap(
                    z=pivot_returns.values,
                    x=pivot_returns.columns,
                    y=pivot_returns.index,
                    colorscale='RdYlGn',
                    text=pivot_returns.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 10},
                    colorbar=dict(title="Returns %")
                ))
                
                fig_heat1.update_layout(
                    title="Returns Heatmap by Day and Hour",
                    xaxis_title="Hour",
                    yaxis_title="Day of Week",
                    template="plotly_dark",
                    height=400
                )
                st.plotly_chart(fig_heat1, use_container_width=True)
            else:
                st.info("Intraday data required for hourly heatmap")
    except Exception as e:
        st.warning(f"Could not generate heatmap: {str(e)}")
    
    st.markdown("---")
    
    # 10-Year Heatmaps (Independent)
    st.subheader("📊 10-Year Historical Analysis")
    st.info("Fetching 10 years of daily data for comprehensive analysis...")
    
    try:
        # Fetch 10 years of daily data
        df_10y = fetch_data(ticker, '1d', '10y')
        
        if df_10y is not None and not df_10y.empty:
            df_10y['year'] = df_10y.index.year
            df_10y['month'] = df_10y.index.month
            df_10y['month_name'] = df_10y.index.strftime('%b')
            df_10y['returns'] = df_10y['close'].pct_change() * 100
            
            # Monthly returns heatmap
            pivot_monthly_returns = df_10y.pivot_table(
                values='returns',
                index='year',
                columns='month_name',
                aggfunc='sum'
            )
            
            # Ensure month order
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            pivot_monthly_returns = pivot_monthly_returns.reindex(columns=month_order)
            
            fig_10y_returns = go.Figure(data=go.Heatmap(
                z=pivot_monthly_returns.values,
                x=pivot_monthly_returns.columns,
                y=pivot_monthly_returns.index,
                colorscale='RdYlGn',
                text=pivot_monthly_returns.values,
                texttemplate='%{text:.1f}',
                textfont={"size": 9},
                colorbar=dict(title="Returns %")
            ))
            
            fig_10y_returns.update_layout(
                title="10-Year Monthly Returns Heatmap",
                xaxis_title="Month",
                yaxis_title="Year",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig_10y_returns, use_container_width=True)
            
            # Monthly volatility heatmap
            pivot_monthly_vol = df_10y.pivot_table(
                values='returns',
                index='year',
                columns='month_name',
                aggfunc='std'
            )
            pivot_monthly_vol = pivot_monthly_vol.reindex(columns=month_order)
            
            fig_10y_vol = go.Figure(data=go.Heatmap(
                z=pivot_monthly_vol.values,
                x=pivot_monthly_vol.columns,
                y=pivot_monthly_vol.index,
                colorscale='Reds',
                text=pivot_monthly_vol.values,
                texttemplate='%{text:.1f}',
                textfont={"size": 9},
                colorbar=dict(title="Volatility %")
            ))
            
            fig_10y_vol.update_layout(
                title="10-Year Monthly Volatility Heatmap",
                xaxis_title="Month",
                yaxis_title="Year",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig_10y_vol, use_container_width=True)
            
            # Additional 10-year statistics
            st.subheader("10-Year Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_return_10y = ((df_10y['close'].iloc[-1] - df_10y['close'].iloc[0]) / df_10y['close'].iloc[0]) * 100
                st.metric("Total Return (10Y)", f"{total_return_10y:.2f}%")
            
            with col2:
                annual_return = total_return_10y / 10
                st.metric("Avg Annual Return", f"{annual_return:.2f}%")
            
            with col3:
                sharpe_approx = df_10y['returns'].mean() / df_10y['returns'].std() * np.sqrt(252)
                st.metric("Sharpe Ratio (approx)", f"{sharpe_approx:.2f}")
        
        else:
            st.warning("Could not fetch 10-year data")
    
    except Exception as e:
        st.error(f"Error fetching 10-year data: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()
