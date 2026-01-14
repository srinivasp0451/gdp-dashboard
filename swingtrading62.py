import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import time
import random

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_ist_time():
    """Get current time in IST"""
    return datetime.now(pytz.timezone('Asia/Kolkata'))

def add_log(message):
    """Add timestamped log entry"""
    timestamp = get_ist_time().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    st.session_state['trade_logs'].append(log_entry)
    # Keep only last 50 logs
    if len(st.session_state['trade_logs']) > 50:
        st.session_state['trade_logs'] = st.session_state['trade_logs'][-50:]
    st.session_state['trade_logs'] = st.session_state['trade_logs']

def reset_position_state():
    """Reset position-related state variables"""
    st.session_state['position'] = None
    st.session_state['trailing_sl_high'] = None
    st.session_state['trailing_sl_low'] = None
    st.session_state['trailing_target_high'] = None
    st.session_state['trailing_target_low'] = None
    st.session_state['trailing_profit_points'] = 0
    st.session_state['threshold_crossed'] = False
    st.session_state['highest_price'] = None
    st.session_state['lowest_price'] = None
    st.session_state['partial_exit_done'] = False
    st.session_state['breakeven_activated'] = False

def validate_interval_period(interval, period):
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

def fetch_data(ticker, interval, period, mode='backtest'):
    """Fetch data from yfinance with proper error handling"""
    try:
        if mode == 'live':
            time.sleep(random.uniform(1.0, 1.5))
        
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if data.empty:
            return None
        
        # Flatten multi-index if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Select OHLCV
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[required_cols].copy()
        
        # Handle timezone
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            data.index = data.index.tz_convert('Asia/Kolkata')
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def get_asset_minimum_target(ticker):
    """Get minimum target distance for asset"""
    ticker_upper = ticker.upper()
    if 'NIFTY' in ticker_upper and 'BANK' not in ticker_upper:
        return 10
    elif 'BANKNIFTY' in ticker_upper or 'NIFTYBANK' in ticker_upper:
        return 20
    elif 'BTC' in ticker_upper:
        return 150
    elif 'ETH' in ticker_upper:
        return 10
    else:
        return 15

# ============================================================================
# INDICATOR CALCULATIONS
# ============================================================================

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

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

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_adx(df, period=14):
    """Calculate Average Directional Index"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = calculate_atr(df, 1)
    
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return upper_band, sma, lower_band

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_stochastic_rsi(rsi, period=14, smooth_k=3, smooth_d=3):
    """Calculate Stochastic RSI"""
    rsi_min = rsi.rolling(window=period).min()
    rsi_max = rsi.rolling(window=period).max()
    
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
    stoch_k = stoch_rsi.rolling(window=smooth_k).mean()
    stoch_d = stoch_k.rolling(window=smooth_d).mean()
    
    return stoch_k, stoch_d

def calculate_keltner_channel(df, period=20, atr_multiplier=2):
    """Calculate Keltner Channel"""
    ema = calculate_ema(df['Close'], period)
    atr = calculate_atr(df, period)
    
    upper = ema + (atr * atr_multiplier)
    lower = ema - (atr * atr_multiplier)
    
    return upper, ema, lower

def calculate_pivot_points(df):
    """Calculate Pivot Points"""
    pivot = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    r1 = 2 * pivot - df['Low'].shift(1)
    r2 = pivot + (df['High'].shift(1) - df['Low'].shift(1))
    s1 = 2 * pivot - df['High'].shift(1)
    s2 = pivot - (df['High'].shift(1) - df['Low'].shift(1))
    
    return pivot, r1, r2, s1, s2

def calculate_vwap(df):
    """Calculate Volume Weighted Average Price"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap

def detect_swing_high_low(df, period=5):
    """Detect swing highs and lows"""
    swing_high = df['High'].rolling(window=period*2+1, center=True).max() == df['High']
    swing_low = df['Low'].rolling(window=period*2+1, center=True).min() == df['Low']
    
    return swing_high, swing_low

def calculate_support_resistance(df, periods=20):
    """Calculate support and resistance levels"""
    resistance = df['High'].rolling(window=periods).max()
    support = df['Low'].rolling(window=periods).min()
    
    return support, resistance

def calculate_ema_angle(ema_series, index):
    """Calculate EMA angle in degrees"""
    if index < 1:
        return 0
    
    y_diff = ema_series.iloc[index] - ema_series.iloc[index - 1]
    angle_rad = np.arctan(y_diff)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

# ============================================================================
# STRATEGY FUNCTIONS
# ============================================================================

def ema_crossover_strategy(df, i, config):
    """EMA Crossover Strategy with filters"""
    if i < 2:
        return 0, None
    
    ema_fast = df['EMA_Fast'].iloc[i]
    ema_slow = df['EMA_Slow'].iloc[i]
    prev_ema_fast = df['EMA_Fast'].iloc[i-1]
    prev_ema_slow = df['EMA_Slow'].iloc[i-1]
    
    # Check for crossover
    bullish_cross = (ema_fast > ema_slow and prev_ema_fast <= prev_ema_slow)
    bearish_cross = (ema_fast < ema_slow and prev_ema_fast >= prev_ema_slow)
    
    if not (bullish_cross or bearish_cross):
        return 0, None
    
    # Check EMA angle - must be positive for both bullish and bearish
    angle = abs(calculate_ema_angle(df['EMA_Fast'], i))
    if angle < config['min_ema_angle']:
        return 0, None
    
    # Check ADX filter if enabled
    if config['use_adx_filter']:
        adx = df['ADX'].iloc[i]
        if adx < config['adx_threshold']:
            return 0, None
    
    # Check entry filter
    entry_filter = config['entry_filter']
    candle_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
    
    if entry_filter == 'Custom Candle (Points)':
        if candle_size < config['custom_candle_points']:
            return 0, None
    
    elif entry_filter == 'ATR-based Candle':
        atr = df['ATR'].iloc[i]
        min_size = atr * config['atr_multiplier']
        if candle_size < min_size:
            return 0, None
    
    # Determine signal
    if bullish_cross:
        return 1, df['Close'].iloc[i]
    elif bearish_cross:
        return -1, df['Close'].iloc[i]
    
    return 0, None

def simple_buy_strategy(df, i):
    """Simple Buy Strategy - Enter immediately"""
    return 1, df['Close'].iloc[i]

def simple_sell_strategy(df, i):
    """Simple Sell Strategy - Enter immediately"""
    return -1, df['Close'].iloc[i]

def price_threshold_strategy(df, i, config):
    """Price Crosses Threshold Strategy"""
    current_price = df['Close'].iloc[i]
    threshold = config['threshold_price']
    direction = config['threshold_direction']
    
    # Check if threshold was crossed in this candle
    crossed = False
    signal = 0
    
    if direction == 'LONG (Price >= Threshold)':
        if current_price >= threshold:
            crossed = True
            signal = 1
    elif direction == 'SHORT (Price >= Threshold)':
        if current_price >= threshold:
            crossed = True
            signal = -1
    elif direction == 'LONG (Price <= Threshold)':
        if current_price <= threshold:
            crossed = True
            signal = 1
    elif direction == 'SHORT (Price <= Threshold)':
        if current_price <= threshold:
            crossed = True
            signal = -1
    
    if crossed:
        return signal, df['Close'].iloc[i]
    
    return 0, None

def rsi_adx_ema_strategy(df, i, config):
    """RSI-ADX-EMA Strategy"""
    if i < 1:
        return 0, None
    
    rsi = df['RSI'].iloc[i]
    adx = df['ADX'].iloc[i]
    ema_fast = df['EMA_Fast'].iloc[i]
    ema_slow = df['EMA_Slow'].iloc[i]
    
    # SELL: RSI>80, ADX<20, EMA_Fast<EMA_Slow
    if rsi > 80 and adx < 20 and ema_fast < ema_slow:
        return -1, df['Close'].iloc[i]
    
    # BUY: RSI<20, ADX>20, EMA_Fast>EMA_Slow
    if rsi < 20 and adx > 20 and ema_fast > ema_slow:
        return 1, df['Close'].iloc[i]
    
    return 0, None

def ai_price_action_strategy(df, i, config):
    """AI Price Action Analysis Strategy"""
    if i < 50:
        return 0, None, {}
    
    close = df['Close'].iloc[i]
    ema20 = df['EMA_20'].iloc[i]
    ema50 = df['EMA_50'].iloc[i]
    rsi = df['RSI'].iloc[i]
    macd = df['MACD'].iloc[i]
    macd_signal = df['MACD_Signal'].iloc[i]
    bb_upper = df['BB_Upper'].iloc[i]
    bb_lower = df['BB_Lower'].iloc[i]
    volume = df['Volume'].iloc[i]
    avg_volume = df['Volume'].iloc[i-20:i].mean()
    
    score = 0
    analysis = {}
    
    # Trend Analysis
    if close > ema20 > ema50:
        analysis['Trend'] = 'Strong Uptrend (+2)'
        score += 2
    elif close > ema20:
        analysis['Trend'] = 'Weak Uptrend (+1)'
        score += 1
    elif close < ema20 < ema50:
        analysis['Trend'] = 'Strong Downtrend (-2)'
        score -= 2
    elif close < ema20:
        analysis['Trend'] = 'Weak Downtrend (-1)'
        score -= 1
    else:
        analysis['Trend'] = 'Neutral (0)'
    
    # RSI Analysis
    if rsi < 30:
        analysis['RSI'] = 'Oversold - Bullish (+2)'
        score += 2
    elif rsi < 40:
        analysis['RSI'] = 'Weak - Slightly Bullish (+1)'
        score += 1
    elif rsi > 70:
        analysis['RSI'] = 'Overbought - Bearish (-2)'
        score -= 2
    elif rsi > 60:
        analysis['RSI'] = 'Strong - Slightly Bearish (-1)'
        score -= 1
    else:
        analysis['RSI'] = 'Neutral (0)'
    
    # MACD Analysis
    if macd > macd_signal and df['MACD'].iloc[i-1] <= df['MACD_Signal'].iloc[i-1]:
        analysis['MACD'] = 'Bullish Crossover (+2)'
        score += 2
    elif macd < macd_signal and df['MACD'].iloc[i-1] >= df['MACD_Signal'].iloc[i-1]:
        analysis['MACD'] = 'Bearish Crossover (-2)'
        score -= 2
    elif macd > macd_signal:
        analysis['MACD'] = 'Bullish (+1)'
        score += 1
    elif macd < macd_signal:
        analysis['MACD'] = 'Bearish (-1)'
        score -= 1
    else:
        analysis['MACD'] = 'Neutral (0)'
    
    # Bollinger Bands
    if close > bb_upper:
        analysis['Bollinger'] = 'Above Upper Band - Overbought (-1)'
        score -= 1
    elif close < bb_lower:
        analysis['Bollinger'] = 'Below Lower Band - Oversold (+1)'
        score += 1
    else:
        analysis['Bollinger'] = 'Within Bands (0)'
    
    # Volume Analysis (skip for indices)
    ticker = config.get('ticker', '').upper()
    has_volume = 'NIFTY' not in ticker and 'SENSEX' not in ticker
    
    if has_volume:
        if volume > avg_volume * 1.5:
            analysis['Volume'] = 'High Volume - Strong Move (+1)'
            score += 1
        elif volume < avg_volume * 0.5:
            analysis['Volume'] = 'Low Volume - Weak Move (-1)'
            score -= 1
        else:
            analysis['Volume'] = 'Average Volume (0)'
    else:
        analysis['Volume'] = 'N/A for Index'
    
    # Generate signal
    signal = 0
    entry_price = None
    
    if score >= 3:
        signal = 1
        entry_price = close
        analysis['Signal'] = f'Strong BUY (Score: {score})'
    elif score <= -3:
        signal = -1
        entry_price = close
        analysis['Signal'] = f'Strong SELL (Score: {score})'
    else:
        analysis['Signal'] = f'No Clear Signal (Score: {score})'
    
    return signal, entry_price, analysis

def custom_strategy(df, i, conditions):
    """Custom Strategy Builder"""
    if not conditions:
        return 0, None
    
    buy_conditions = [c for c in conditions if c['action'] == 'BUY' and c.get('use_condition', True)]
    sell_conditions = [c for c in conditions if c['action'] == 'SELL' and c.get('use_condition', True)]
    
    buy_met = 0
    sell_met = 0
    
    for condition in conditions:
        if not condition.get('use_condition', True):
            continue
            
        indicator = condition['indicator']
        operator = condition['operator']
        value = condition['value']
        
        # Get indicator value
        if indicator == 'Price':
            ind_value = df['Close'].iloc[i]
            prev_ind_value = df['Close'].iloc[i-1] if i > 0 else ind_value
        elif indicator in df.columns:
            ind_value = df[indicator].iloc[i]
            prev_ind_value = df[indicator].iloc[i-1] if i > 0 else ind_value
        elif indicator == 'Close':
            ind_value = df['Close'].iloc[i]
            prev_ind_value = df['Close'].iloc[i-1] if i > 0 else ind_value
        elif indicator == 'High':
            ind_value = df['High'].iloc[i]
            prev_ind_value = df['High'].iloc[i-1] if i > 0 else ind_value
        elif indicator == 'Low':
            ind_value = df['Low'].iloc[i]
            prev_ind_value = df['Low'].iloc[i-1] if i > 0 else ind_value
        else:
            continue
        
        # Check condition
        condition_met = False
        
        if operator == '>':
            condition_met = ind_value > value
        elif operator == '<':
            condition_met = ind_value < value
        elif operator == '>=':
            condition_met = ind_value >= value
        elif operator == '<=':
            condition_met = ind_value <= value
        elif operator == '==':
            condition_met = abs(ind_value - value) < 0.01
        elif operator == 'crosses_above':
            condition_met = ind_value > value and prev_ind_value <= value
        elif operator == 'crosses_below':
            condition_met = ind_value < value and prev_ind_value >= value
        
        if condition_met:
            if condition['action'] == 'BUY':
                buy_met += 1
            else:
                sell_met += 1
    
    # Check if all conditions met
    if len(buy_conditions) > 0 and buy_met == len(buy_conditions):
        return 1, df['Close'].iloc[i]
    elif len(sell_conditions) > 0 and sell_met == len(sell_conditions):
        return -1, df['Close'].iloc[i]
    
    return 0, None

# ============================================================================
# STOP LOSS AND TARGET CALCULATIONS
# ============================================================================

def calculate_stop_loss(df, i, entry_price, signal, sl_type, config):
    """Calculate stop loss based on type"""
    sl = 0
    
    if sl_type == 'Custom Points':
        points = config.get('sl_points', 10)
        if signal == 1:  # LONG
            sl = entry_price - points
        else:  # SHORT
            sl = entry_price + points
    
    elif sl_type == 'ATR-based':
        atr = df['ATR'].iloc[i]
        multiplier = config.get('atr_multiplier_sl', 1.5)
        if signal == 1:
            sl = entry_price - (atr * multiplier)
        else:
            sl = entry_price + (atr * multiplier)
    
    elif sl_type == 'Current Candle Low/High':
        if signal == 1:
            sl = df['Low'].iloc[i]
        else:
            sl = df['High'].iloc[i]
    
    elif sl_type == 'Previous Candle Low/High':
        if i > 0:
            if signal == 1:
                sl = df['Low'].iloc[i-1]
            else:
                sl = df['High'].iloc[i-1]
        else:
            sl = entry_price - 10 if signal == 1 else entry_price + 10
    
    elif sl_type == 'Current Swing Low/High':
        if signal == 1:
            swing_lows = df.loc[df['Swing_Low'] == True, 'Low']
            if not swing_lows.empty and i > 0:
                sl = swing_lows.iloc[-1]
            else:
                sl = entry_price - 10
        else:
            swing_highs = df.loc[df['Swing_High'] == True, 'High']
            if not swing_highs.empty and i > 0:
                sl = swing_highs.iloc[-1]
            else:
                sl = entry_price + 10
    
    elif sl_type == 'Previous Swing Low/High':
        if signal == 1:
            swing_lows = df.loc[df['Swing_Low'] == True, 'Low']
            if len(swing_lows) >= 2:
                sl = swing_lows.iloc[-2]
            else:
                sl = entry_price - 10
        else:
            swing_highs = df.loc[df['Swing_High'] == True, 'High']
            if len(swing_highs) >= 2:
                sl = swing_highs.iloc[-2]
            else:
                sl = entry_price + 10
    
    elif 'Trailing' in sl_type:
        # Initial SL for trailing
        points = config.get('sl_points', 10)
        if signal == 1:
            sl = entry_price - points
        else:
            sl = entry_price + points
    
    elif sl_type == 'Signal-based':
        # No fixed SL for signal-based
        sl = 0
    
    # Ensure minimum distance
    min_distance = 10
    if signal == 1:
        sl = min(sl, entry_price - min_distance)
    else:
        sl = max(sl, entry_price + min_distance)
    
    return sl

def calculate_target(df, i, entry_price, signal, target_type, config):
    """Calculate target based on type"""
    target = 0
    
    if target_type == 'Custom Points':
        points = config.get('target_points', 20)
        if signal == 1:
            target = entry_price + points
        else:
            target = entry_price - points
    
    elif target_type == 'ATR-based':
        atr = df['ATR'].iloc[i]
        multiplier = config.get('atr_multiplier_target', 3)
        if signal == 1:
            target = entry_price + (atr * multiplier)
        else:
            target = entry_price - (atr * multiplier)
    
    elif target_type == 'Risk-Reward Based':
        sl = config.get('current_sl', 0)
        rr_ratio = config.get('risk_reward_ratio', 2)
        risk = abs(entry_price - sl)
        if signal == 1:
            target = entry_price + (risk * rr_ratio)
        else:
            target = entry_price - (risk * rr_ratio)
    
    elif target_type == 'Current Candle Low/High':
        if signal == 1:
            target = df['High'].iloc[i]
        else:
            target = df['Low'].iloc[i]
    
    elif target_type == 'Previous Candle Low/High':
        if i > 0:
            if signal == 1:
                target = df['High'].iloc[i-1]
            else:
                target = df['Low'].iloc[i-1]
        else:
            target = entry_price + 20 if signal == 1 else entry_price - 20
    
    elif target_type == 'Current Swing Low/High':
        if signal == 1:
            swing_highs = df.loc[df['Swing_High'] == True, 'High']
            if not swing_highs.empty:
                target = swing_highs.iloc[-1]
            else:
                target = entry_price + 20
        else:
            swing_lows = df.loc[df['Swing_Low'] == True, 'Low']
            if not swing_lows.empty:
                target = swing_lows.iloc[-1]
            else:
                target = entry_price - 20
    
    elif target_type == 'Previous Swing Low/High':
        if signal == 1:
            swing_highs = df.loc[df['Swing_High'] == True, 'High']
            if len(swing_highs) >= 2:
                target = swing_highs.iloc[-2]
            else:
                target = entry_price + 20
        else:
            swing_lows = df.loc[df['Swing_Low'] == True, 'Low']
            if len(swing_lows) >= 2:
                target = swing_lows.iloc[-2]
            else:
                target = entry_price - 20
    
    elif 'Trailing' in target_type or '50% Exit' in target_type:
        # Initial target for trailing
        min_target = get_asset_minimum_target(config.get('ticker', ''))
        if signal == 1:
            target = entry_price + min_target
        else:
            target = entry_price - min_target
    
    elif target_type == 'Signal-based':
        # No fixed target for signal-based
        target = 0
    
    # Ensure minimum distance
    min_distance = get_asset_minimum_target(config.get('ticker', ''))
    if signal == 1:
        target = max(target, entry_price + min_distance)
    else:
        target = min(target, entry_price - min_distance)
    
    return target

def update_trailing_sl(position, current_price, config):
    """Update trailing stop loss"""
    sl_type = config['sl_type']
    entry_price = position['entry_price']
    signal = position['signal']
    current_sl = position['sl']
    
    if 'Trailing SL (Points)' == sl_type:
        # Pure trailing with point offset - distance always maintained
        points = config.get('sl_points', 10)
        threshold = config.get('trailing_threshold', 0)
        
        if signal == 1:  # LONG
            # SL always trails at fixed distance below current price
            new_sl = current_price - points
            
            # Only update if new SL is higher than current SL
            if new_sl > current_sl:
                return new_sl
        
        else:  # SHORT
            # SL always trails at fixed distance above current price
            new_sl = current_price + points
            
            # Only update if new SL is lower than current SL
            if new_sl < current_sl:
                return new_sl
    
    return current_sl

def update_trailing_target(position, current_price, config):
    """Update trailing target (DISPLAY ONLY - DOES NOT EXIT)"""
    target_type = config['target_type']
    entry_price = position['entry_price']
    signal = position['signal']
    target_points = config.get('target_points', get_asset_minimum_target(config.get('ticker', '')))
    
    if 'Trailing Target' in target_type:
        if signal == 1:  # LONG
            if st.session_state['highest_price'] is None:
                st.session_state['highest_price'] = current_price
            
            if current_price > st.session_state['highest_price']:
                st.session_state['highest_price'] = current_price
            
            # Check if profit moved enough
            profit_points = st.session_state['highest_price'] - entry_price
            if profit_points >= st.session_state['trailing_profit_points'] + target_points:
                st.session_state['trailing_profit_points'] = profit_points
                return st.session_state['highest_price']
        
        else:  # SHORT
            if st.session_state['lowest_price'] is None:
                st.session_state['lowest_price'] = current_price
            
            if current_price < st.session_state['lowest_price']:
                st.session_state['lowest_price'] = current_price
            
            profit_points = entry_price - st.session_state['lowest_price']
            if profit_points >= st.session_state['trailing_profit_points'] + target_points:
                st.session_state['trailing_profit_points'] = profit_points
                return st.session_state['lowest_price']
    
    return position['target']

def check_signal_based_exit(df, i, position):
    """Check for signal-based exit (reverse crossover)"""
    if i < 1:
        return False
    
    signal = position['signal']
    ema_fast = df['EMA_Fast'].iloc[i]
    ema_slow = df['EMA_Slow'].iloc[i]
    prev_ema_fast = df['EMA_Fast'].iloc[i-1]
    prev_ema_slow = df['EMA_Slow'].iloc[i-1]
    
    if signal == 1:  # LONG position
        # Exit on bearish crossover
        if ema_fast < ema_slow and prev_ema_fast >= prev_ema_slow:
            return True
    
    elif signal == -1:  # SHORT position
        # Exit on bullish crossover
        if ema_fast > ema_slow and prev_ema_fast <= prev_ema_slow:
            return True
    
    return False

def check_breakeven_sl(position, current_price, config):
    """Check and update breakeven SL after 50% target"""
    if position.get('breakeven_activated', False):
        return position['sl']
    
    entry_price = position['entry_price']
    target = position['target']
    signal = position['signal']
    
    if target == 0:
        return position['sl']
    
    target_distance = abs(target - entry_price)
    current_profit = abs(current_price - entry_price) if signal == 1 else abs(entry_price - current_price)
    
    if current_profit >= target_distance * 0.5:
        st.session_state['breakeven_activated'] = True
        position['breakeven_activated'] = True
        add_log("Break-even activated: SL moved to entry price")
        return entry_price
    
    return position['sl']

# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def run_backtest(df, strategy, config):
    """Run backtest on historical data"""
    backtest_results = {
        'trades': [],
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'total_pnl': 0,
        'accuracy': 0,
        'avg_duration': 0
    }
    
    position = None
    
    for i in range(len(df)):
        if i < 50:  # Need enough data for indicators
            continue
        
        # Check for entry
        if position is None:
            signal = 0
            entry_price = None
            
            if strategy == "EMA Crossover":
                signal, entry_price = ema_crossover_strategy(df, i, config)
            elif strategy == "Simple Buy":
                signal, entry_price = simple_buy_strategy(df, i)
            elif strategy == "Simple Sell":
                signal, entry_price = simple_sell_strategy(df, i)
            elif strategy == "Price Crosses Threshold":
                signal, entry_price = price_threshold_strategy(df, i, config)
            elif strategy == "RSI-ADX-EMA":
                signal, entry_price = rsi_adx_ema_strategy(df, i, config)
            elif strategy == "AI Price Action Analysis":
                signal, entry_price, analysis = ai_price_action_strategy(df, i, config)
                if signal != 0:
                    atr = df['ATR'].iloc[i]
                    if signal == 1:
                        config['auto_sl'] = entry_price - (atr * 1.5)
                        config['auto_target'] = entry_price + (atr * 3)
                    else:
                        config['auto_sl'] = entry_price + (atr * 1.5)
                        config['auto_target'] = entry_price - (atr * 3)
            elif strategy == "Custom Strategy Builder":
                signal, entry_price = custom_strategy(df, i, config.get('custom_conditions', []))
            
            if signal != 0:
                sl = calculate_stop_loss(df, i, entry_price, signal, config['sl_type'], config)
                target = calculate_target(df, i, entry_price, signal, config['target_type'], config)
                
                if strategy == "AI Price Action Analysis":
                    sl = config.get('auto_sl', sl)
                    target = config.get('auto_target', target)
                
                position = {
                    'signal': signal,
                    'entry_price': entry_price,
                    'entry_time': df.index[i],
                    'entry_index': i,
                    'sl': sl,
                    'target': target,
                    'quantity': config['quantity'],
                    'highest_price': entry_price,
                    'lowest_price': entry_price
                }
        
        else:
            # Manage position
            current_price = df['Close'].iloc[i]
            
            # Update highest/lowest
            position['highest_price'] = max(position['highest_price'], current_price)
            position['lowest_price'] = min(position['lowest_price'], current_price)
            
            # Check exits
            exit_trade = False
            exit_reason = ""
            exit_price = current_price
            
            # Check signal-based exit
            if config['sl_type'] == 'Signal-based' or config['target_type'] == 'Signal-based':
                if check_signal_based_exit(df, i, position):
                    exit_trade = True
                    exit_reason = "Reverse Signal"
            
            # Check SL
            if not exit_trade and position['sl'] != 0:
                if position['signal'] == 1 and current_price <= position['sl']:
                    exit_trade = True
                    exit_reason = "Stop Loss"
                    exit_price = position['sl']
                elif position['signal'] == -1 and current_price >= position['sl']:
                    exit_trade = True
                    exit_reason = "Stop Loss"
                    exit_price = position['sl']
            
            # Check target
            if not exit_trade and position['target'] != 0:
                if position['signal'] == 1 and current_price >= position['target']:
                    exit_trade = True
                    exit_reason = "Target"
                    exit_price = position['target']
                elif position['signal'] == -1 and current_price <= position['target']:
                    exit_trade = True
                    exit_reason = "Target"
                    exit_price = position['target']
            
            if exit_trade:
                # Calculate P&L
                if position['signal'] == 1:
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                else:
                    pnl = (position['entry_price'] - exit_price) * position['quantity']
                
                # Record trade
                trade_record = {
                    'entry_time': position['entry_time'],
                    'exit_time': df.index[i],
                    'duration': df.index[i] - position['entry_time'],
                    'signal': 'LONG' if position['signal'] == 1 else 'SHORT',
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'sl': position['sl'],
                    'target': position['target'],
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'quantity': position['quantity'],
                    'highest_price': position['highest_price'],
                    'lowest_price': position['lowest_price'],
                    'range': position['highest_price'] - position['lowest_price']
                }
                
                backtest_results['trades'].append(trade_record)
                backtest_results['total_pnl'] += pnl
                
                if pnl > 0:
                    backtest_results['winning_trades'] += 1
                else:
                    backtest_results['losing_trades'] += 1
                
                position = None
    
    # Calculate stats
    backtest_results['total_trades'] = len(backtest_results['trades'])
    if backtest_results['total_trades'] > 0:
        backtest_results['accuracy'] = (backtest_results['winning_trades'] / backtest_results['total_trades']) * 100
        
        total_duration = sum([trade['duration'].total_seconds() for trade in backtest_results['trades']])
        backtest_results['avg_duration'] = total_duration / backtest_results['total_trades']
    
    return backtest_results

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(page_title="Advanced Trading System", layout="wide", initial_sidebar_state="expanded")
    
    st.title("🚀 Advanced Quantitative Trading System")
    
    # Initialize session state
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
    if 'custom_conditions' not in st.session_state:
        st.session_state['custom_conditions'] = []
    if 'partial_exit_done' not in st.session_state:
        st.session_state['partial_exit_done'] = False
    if 'breakeven_activated' not in st.session_state:
        st.session_state['breakeven_activated'] = False
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode Selection
        mode = st.selectbox("Mode", ["Backtest", "Live Trading"])
        
        # Asset Selection
        st.subheader("Asset Selection")
        asset_type = st.selectbox("Asset Type", [
            "Indian Indices", "Crypto", "Forex", "Commodities", "Custom Ticker"
        ])
        
        if asset_type == "Indian Indices":
            ticker = st.selectbox("Select Index", ["^NSEI", "^NSEBANK", "^BSESN"])
        elif asset_type == "Crypto":
            ticker = st.selectbox("Select Crypto", ["BTC-USD", "ETH-USD"])
        elif asset_type == "Forex":
            ticker = st.selectbox("Select Forex", ["USDINR=X", "EURUSD=X", "GBPUSD=X"])
        elif asset_type == "Commodities":
            ticker = st.selectbox("Select Commodity", ["GC=F", "SI=F"])
        else:
            ticker = st.text_input("Enter Ticker Symbol", "^NSEI")
        
        # Timeframe Selection
        st.subheader("Timeframe")
        interval = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"])
        
        # Period Selection based on interval
        valid_periods = {
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
        
        period = st.selectbox("Period", valid_periods[interval])
        
        # Quantity
        quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
        
        # Strategy Selection
        st.subheader("Strategy")
        strategy = st.selectbox("Select Strategy", [
            "EMA Crossover",
            "Simple Buy",
            "Simple Sell",
            "Price Crosses Threshold",
            "RSI-ADX-EMA",
            "AI Price Action Analysis",
            "Custom Strategy Builder"
        ])
        
        # Strategy Configuration
        config = {'ticker': ticker, 'quantity': quantity}
        
        if strategy == "EMA Crossover":
            st.markdown("**EMA Settings**")
            config['ema_fast'] = st.number_input("EMA Fast", min_value=1, value=9, step=1)
            config['ema_slow'] = st.number_input("EMA Slow", min_value=1, value=15, step=1)
            config['min_ema_angle'] = st.number_input("Min EMA Angle (degrees)", min_value=0.0, value=1.0, step=0.1)
            
            st.markdown("**Entry Filter**")
            config['entry_filter'] = st.selectbox("Entry Filter Type", [
                "Simple Crossover",
                "Custom Candle (Points)",
                "ATR-based Candle"
            ])
            
            if config['entry_filter'] == "Custom Candle (Points)":
                config['custom_candle_points'] = st.number_input("Min Candle Size (Points)", min_value=1.0, value=10.0, step=1.0)
            
            if config['entry_filter'] == "ATR-based Candle":
                config['atr_multiplier'] = st.number_input("ATR Multiplier", min_value=0.1, value=1.0, step=0.1)
            
            st.markdown("**ADX Filter**")
            config['use_adx_filter'] = st.checkbox("Enable ADX Filter", value=False)
            if config['use_adx_filter']:
                config['adx_period'] = st.number_input("ADX Period", min_value=1, value=14, step=1)
                config['adx_threshold'] = st.number_input("ADX Threshold", min_value=0, value=25, step=1)
        
        elif strategy == "Price Crosses Threshold":
            config['threshold_price'] = st.number_input("Threshold Price", min_value=0.0, value=100.0, step=1.0)
            config['threshold_direction'] = st.selectbox("Direction", [
                "LONG (Price >= Threshold)",
                "SHORT (Price >= Threshold)",
                "LONG (Price <= Threshold)",
                "SHORT (Price <= Threshold)"
            ])
        
        elif strategy == "RSI-ADX-EMA":
            config['ema_fast'] = st.number_input("EMA Fast", min_value=1, value=9, step=1)
            config['ema_slow'] = st.number_input("EMA Slow", min_value=1, value=15, step=1)
            config['rsi_period'] = st.number_input("RSI Period", min_value=1, value=14, step=1)
            config['adx_period'] = st.number_input("ADX Period", min_value=1, value=14, step=1)
        
        elif strategy == "Custom Strategy Builder":
            st.markdown("**Build Your Strategy**")
            
            if st.button("➕ Add Condition"):
                st.session_state['custom_conditions'].append({
                    'indicator': 'RSI',
                    'operator': '>',
                    'value': 50,
                    'action': 'BUY'
                })
                st.session_state['custom_conditions'] = st.session_state['custom_conditions']
            
            for idx, condition in enumerate(st.session_state['custom_conditions']):
                with st.expander(f"Condition {idx + 1}", expanded=True):
                    condition['use_condition'] = st.selectbox(
                        "Use this condition?",
                        [True, False],
                        index=0 if condition.get('use_condition', True) else 1,
                        key=f"use_{idx}"
                    )
                    
                    condition['indicator'] = st.selectbox(
                        "Indicator",
                        ["Price", "RSI", "ADX", "EMA_Fast", "EMA_Slow", "MACD", "MACD_Signal", 
                         "BB_Upper", "BB_Lower", "ATR", "Volume", "VWAP", "Close", "High", "Low",
                         "Support", "Resistance", "EMA_20", "EMA_50"],
                        key=f"ind_{idx}"
                    )
                    condition['operator'] = st.selectbox(
                        "Operator",
                        [">", "<", ">=", "<=", "==", "crosses_above", "crosses_below"],
                        key=f"op_{idx}"
                    )
                    condition['value'] = st.number_input(
                        "Value",
                        value=float(condition['value']),
                        key=f"val_{idx}"
                    )
                    condition['action'] = st.selectbox(
                        "Action",
                        ["BUY", "SELL"],
                        key=f"act_{idx}"
                    )
                    
                    if st.button(f"🗑️ Remove", key=f"rem_{idx}"):
                        st.session_state['custom_conditions'].pop(idx)
                        st.session_state['custom_conditions'] = st.session_state['custom_conditions']
                        st.rerun()
        
        # Stop Loss Configuration
        st.subheader("Stop Loss")
        config['sl_type'] = st.selectbox("SL Type", [
            "Custom Points",
            "Trailing SL (Points)",
            "Trailing SL + Current Candle",
            "Trailing SL + Previous Candle",
            "Trailing SL + Current Swing",
            "Trailing SL + Previous Swing",
            "Trailing SL + Signal Based",
            "Volatility-Adjusted Trailing SL",
            "Break-even After 50% Target",
            "ATR-based",
            "Current Candle Low/High",
            "Previous Candle Low/High",
            "Current Swing Low/High",
            "Previous Swing Low/High",
            "Signal-based"
        ])
        
        if 'Custom Points' in config['sl_type'] or 'Trailing' in config['sl_type']:
            config['sl_points'] = st.number_input("SL Points", min_value=1.0, value=10.0, step=1.0, key='sl_points_input')
        
        if 'Trailing' in config['sl_type']:
            config['trailing_threshold'] = st.number_input("Trailing Threshold (Points)", min_value=0.0, value=0.0, step=1.0, key='trail_thresh_input')
        
        if config['sl_type'] == 'ATR-based' or config['sl_type'] == 'Volatility-Adjusted Trailing SL':
            config['atr_multiplier_sl'] = st.number_input("ATR Multiplier (SL)", min_value=0.1, value=1.5, step=0.1)
        
        # Target Configuration
        st.subheader("Target")
        config['target_type'] = st.selectbox("Target Type", [
            "Custom Points",
            "Trailing Target (Points)",
            "Trailing Target + Signal Based",
            "50% Exit at Target (Partial)",
            "Current Candle Low/High",
            "Previous Candle Low/High",
            "Current Swing Low/High",
            "Previous Swing Low/High",
            "ATR-based",
            "Risk-Reward Based",
            "Signal-based"
        ])
        
        if 'Custom Points' in config['target_type'] or 'Trailing' in config['target_type'] or '50%' in config['target_type']:
            default_target = get_asset_minimum_target(ticker)
            config['target_points'] = st.number_input("Target Points", min_value=1.0, value=float(default_target), step=1.0, key='target_points_input')
        
        if config['target_type'] == 'ATR-based':
            config['atr_multiplier_target'] = st.number_input("ATR Multiplier (Target)", min_value=0.1, value=3.0, step=0.1)
        
        if config['target_type'] == 'Risk-Reward Based':
            config['risk_reward_ratio'] = st.number_input("Risk:Reward Ratio", min_value=0.1, value=2.0, step=0.1)
    
    # Main Content Area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Trading Dashboard", "📈 Trade History", "📝 Trade Logs", "📊 Backtest Results"])
    
    with tab1:
        # Trading Controls at top
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("▶️ Start Trading", type="primary", use_container_width=True):
                st.session_state['trading_active'] = True
                add_log("Trading started")
        
        with col2:
            if st.button("⏹️ Stop Trading", type="secondary", use_container_width=True):
                st.session_state['trading_active'] = False
                
                # If position exists, close it manually
                if st.session_state['position'] is not None:
                    pos = st.session_state['position']
                    current_time = get_ist_time()
                    
                    # Get current price from latest data
                    if st.session_state['current_data'] is not None:
                        current_price = st.session_state['current_data']['Close'].iloc[-1]
                    else:
                        current_price = pos['entry_price']
                    
                    # Calculate P&L
                    if pos['signal'] == 1:
                        pnl = (current_price - pos['entry_price']) * pos['quantity']
                    else:
                        pnl = (pos['entry_price'] - current_price) * pos['quantity']
                    
                    # Record trade
                    trade_record = {
                        'entry_time': pos['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                        'exit_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'duration': str(current_time - pos['entry_time']).split('.')[0],
                        'signal': 'LONG' if pos['signal'] == 1 else 'SHORT',
                        'entry_price': pos['entry_price'],
                        'exit_price': current_price,
                        'sl': pos['sl'],
                        'target': pos['target'],
                        'exit_reason': 'Manual Close',
                        'pnl': pnl,
                        'quantity': pos['quantity'],
                        'strategy': pos['strategy'],
                        'highest_price': st.session_state.get('highest_price', 0),
                        'lowest_price': st.session_state.get('lowest_price', 0),
                        'range': st.session_state.get('highest_price', 0) - st.session_state.get('lowest_price', 0)
                    }
                    
                    st.session_state['trade_history'].append(trade_record)
                    st.session_state['trade_history'] = st.session_state['trade_history']
                    
                    add_log(f"EXIT: Manual Close at {current_price:.2f} | P&L: {pnl:.2f}")
                
                reset_position_state()
                add_log("Trading stopped - Position state reset")
        
        with col3:
            if st.session_state['trading_active']:
                st.success("🟢 Trading is ACTIVE")
            else:
                st.info("⚪ Trading is STOPPED")
        
        st.divider()
        
        # Live Trading Logic
        if st.session_state['trading_active'] or mode == "Backtest":
            # Fetch data
            with st.spinner("Fetching market data..."):
                progress_bar = st.progress(0)
                df = fetch_data(ticker, interval, period, mode.lower())
                progress_bar.progress(100)
                progress_bar.empty()
            
            if df is None or df.empty:
                st.error("Failed to fetch data. Please check ticker and try again.")
                st.stop()
            
            # Calculate indicators
            df['EMA_Fast'] = calculate_ema(df['Close'], config.get('ema_fast', 9))
            df['EMA_Slow'] = calculate_ema(df['Close'], config.get('ema_slow', 15))
            df['EMA_20'] = calculate_ema(df['Close'], 20)
            df['EMA_50'] = calculate_ema(df['Close'], 50)
            df['ATR'] = calculate_atr(df, 14)
            df['RSI'] = calculate_rsi(df['Close'], config.get('rsi_period', 14))
            df['ADX'] = calculate_adx(df, config.get('adx_period', 14))
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
            df['VWAP'] = calculate_vwap(df)
            df['Support'], df['Resistance'] = calculate_support_resistance(df)
            df['Swing_High'], df['Swing_Low'] = detect_swing_high_low(df)
            
            st.session_state['current_data'] = df
            
            # Get current index
            current_idx = len(df) - 1
            current_price = df['Close'].iloc[current_idx]
            current_time = df.index[current_idx]
            
            # Live Metrics Display
            st.subheader("📊 Live Market Metrics")
            
            metric_cols = st.columns(5)
            
            with metric_cols[0]:
                st.metric("Current Price", f"{current_price:.2f}")
            
            with metric_cols[1]:
                if st.session_state['position']:
                    entry_price = st.session_state['position']['entry_price']
                    st.metric("Entry Price", f"{entry_price:.2f}")
                else:
                    st.metric("Entry Price", "N/A")
            
            with metric_cols[2]:
                if st.session_state['position']:
                    pos_type = "LONG" if st.session_state['position']['signal'] == 1 else "SHORT"
                    st.metric("Position", pos_type)
                else:
                    st.metric("Position", "NONE")
            
            with metric_cols[3]:
                if st.session_state['position']:
                    entry_price = st.session_state['position']['entry_price']
                    signal = st.session_state['position']['signal']
                    qty = st.session_state['position']['quantity']
                    
                    if signal == 1:
                        pnl = (current_price - entry_price) * qty
                    else:
                        pnl = (entry_price - current_price) * qty
                    
                    delta_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
                    st.metric("Unrealized P&L", f"{pnl:.2f}", delta=delta_str, delta_color="normal" if pnl >= 0 else "inverse")
                else:
                    st.metric("Unrealized P&L", "0.00")
            
            with metric_cols[4]:
                st.metric("Last Update", current_time.strftime('%H:%M:%S'))
            
            st.divider()
            
            # Display Selected Parameters
            st.markdown("### ⚙️ Active Configuration")
            
            param_col1, param_col2, param_col3 = st.columns(3)
            
            with param_col1:
                st.markdown(f"**Asset:** {ticker}")
                st.markdown(f"**Interval:** {interval}")
                st.markdown(f"**Period:** {period}")
                st.markdown(f"**Quantity:** {quantity}")
            
            with param_col2:
                st.markdown(f"**Strategy:** {strategy}")
                st.markdown(f"**SL Type:** {config['sl_type']}")
                if 'sl_points' in config:
                    st.markdown(f"**SL Points:** {config['sl_points']}")
            
            with param_col3:
                st.markdown(f"**Target Type:** {config['target_type']}")
                if 'target_points' in config:
                    st.markdown(f"**Target Points:** {config['target_points']}")
                st.markdown(f"**Mode:** {mode}")
            
            st.divider()
            
            # Indicator Values
            ind_cols = st.columns(4)
            
            with ind_cols[0]:
                ema_fast_val = df['EMA_Fast'].iloc[current_idx]
                st.metric("EMA Fast", f"{ema_fast_val:.2f}")
            
            with ind_cols[1]:
                ema_slow_val = df['EMA_Slow'].iloc[current_idx]
                st.metric("EMA Slow", f"{ema_slow_val:.2f}")
            
            with ind_cols[2]:
                if current_idx >= 1:
                    angle = calculate_ema_angle(df['EMA_Fast'], current_idx)
                    st.metric("Crossover Angle", f"{angle:.2f}°")
                else:
                    st.metric("Crossover Angle", "N/A")
            
            with ind_cols[3]:
                rsi_val = df['RSI'].iloc[current_idx]
                st.metric("RSI", f"{rsi_val:.2f}")
            
            # Entry Filter Status (for EMA Crossover)
            if strategy == "EMA Crossover":
                st.markdown("**Entry Filter Status**")
                
                entry_filter = config.get('entry_filter', 'Simple Crossover')
                candle_size = abs(df['Close'].iloc[current_idx] - df['Open'].iloc[current_idx])
                
                filter_status = "✅"
                filter_info = ""
                
                if entry_filter == "Custom Candle (Points)":
                    min_size = config.get('custom_candle_points', 10)
                    filter_status = "✅" if candle_size >= min_size else "❌"
                    filter_info = f"Candle Size: {candle_size:.2f} / Min: {min_size:.2f} {filter_status}"
                
                elif entry_filter == "ATR-based Candle":
                    atr_val = df['ATR'].iloc[current_idx]
                    multiplier = config.get('atr_multiplier', 1.0)
                    min_size = atr_val * multiplier
                    filter_status = "✅" if candle_size >= min_size else "❌"
                    filter_info = f"Candle Size: {candle_size:.2f} / Min (ATR×{multiplier}): {min_size:.2f} {filter_status}"
                
                else:
                    filter_info = f"Entry Filter: {entry_filter} {filter_status}"
                
                st.info(filter_info)
                
                if config.get('use_adx_filter', False):
                    adx_val = df['ADX'].iloc[current_idx]
                    adx_threshold = config.get('adx_threshold', 25)
                    adx_status = "✅" if adx_val >= adx_threshold else "❌"
                    st.info(f"ADX Filter: {adx_val:.2f} / Threshold: {adx_threshold} {adx_status}")
            
            # AI Analysis Display
            if strategy == "AI Price Action Analysis":
                st.markdown("**🤖 AI Analysis**")
                signal, entry_price, analysis = ai_price_action_strategy(df, current_idx, config)
                
                if analysis:
                    ai_col1, ai_col2 = st.columns(2)
                    
                    with ai_col1:
                        st.markdown(f"**Signal:** {analysis.get('Signal', 'N/A')}")
                        st.markdown(f"**Trend:** {analysis.get('Trend', 'N/A')}")
                        st.markdown(f"**RSI:** {analysis.get('RSI', 'N/A')}")
                    
                    with ai_col2:
                        st.markdown(f"**MACD:** {analysis.get('MACD', 'N/A')}")
                        st.markdown(f"**Bollinger:** {analysis.get('Bollinger', 'N/A')}")
                        st.markdown(f"**Volume:** {analysis.get('Volume', 'N/A')}")
            
            # Position Information
            if st.session_state['position']:
                st.markdown("### 📍 Current Position")
                
                pos = st.session_state['position']
                entry_time = pos['entry_time']
                duration = current_time - entry_time
                
                pos_col1, pos_col2, pos_col3, pos_col4 = st.columns(4)
                
                with pos_col1:
                    st.metric("Entry Time", entry_time.strftime('%Y-%m-%d %H:%M:%S'))
                    st.metric("Duration", str(duration).split('.')[0])
                
                with pos_col2:
                    st.metric("Entry Price", f"{pos['entry_price']:.2f}")
                    
                    sl_str = f"{pos['sl']:.2f}" if pos['sl'] != 0 else "Signal Based"
                    st.metric("Stop Loss", sl_str)
                
                with pos_col3:
                    target_str = f"{pos['target']:.2f}" if pos['target'] != 0 else "Signal Based"
                    st.metric("Target", target_str)
                    
                    if pos['sl'] != 0:
                        dist_sl = abs(current_price - pos['sl'])
                        st.metric("Distance to SL", f"{dist_sl:.2f}")
                
                with pos_col4:
                    if pos['target'] != 0:
                        dist_target = abs(pos['target'] - current_price)
                        st.metric("Distance to Target", f"{dist_target:.2f}")
                    
                    # Show highest/lowest/range
                    if st.session_state['highest_price'] and st.session_state['lowest_price']:
                        price_range = st.session_state['highest_price'] - st.session_state['lowest_price']
                        st.metric("Trade Range", f"{price_range:.2f}")
                
                # Additional position info
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    if st.session_state['highest_price']:
                        st.metric("Highest Price", f"{st.session_state['highest_price']:.2f}")
                
                with info_col2:
                    if st.session_state['lowest_price']:
                        st.metric("Lowest Price", f"{st.session_state['lowest_price']:.2f}")
                
                # Trailing target info
                if 'Trailing Target' in config['target_type']:
                    profit_moved = st.session_state.get('trailing_profit_points', 0)
                    next_update = config.get('target_points', 20)
                    st.info(f"💹 Profit moved {profit_moved:.2f} points | Next update at {profit_moved + next_update:.2f} points")
                
                # Partial exit info
                if pos.get('partial_exit_done', False):
                    st.warning("⚠️ 50% position already exited - Trailing remaining 50%")
                
                # Break-even info
                if pos.get('breakeven_activated', False):
                    st.success("✅ Break-even activated: SL moved to entry price")
                
                # Guidance
                signal_val = pos['signal']
                if signal_val == 1:
                    if current_price > pos['entry_price']:
                        st.success("📈 In Profit - HOLD")
                    else:
                        st.error("📉 In Loss - Monitor SL")
                else:
                    if current_price < pos['entry_price']:
                        st.success("📈 In Profit - HOLD")
                    else:
                        st.error("📉 In Loss - Monitor SL")
            
            # Live Chart
            st.markdown("### 📉 Live Chart")
            
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            )])
            
            fig.add_trace(go.Scatter(
                x=df.index, y=df['EMA_Fast'],
                mode='lines', name='EMA Fast',
                line=dict(color='blue', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index, y=df['EMA_Slow'],
                mode='lines', name='EMA Slow',
                line=dict(color='orange', width=1)
            ))
            
            # Add position lines
            if st.session_state['position']:
                pos = st.session_state['position']
                
                # Entry line
                fig.add_hline(
                    y=pos['entry_price'],
                    line_dash="dash",
                    line_color="yellow",
                    annotation_text="Entry"
                )
                
                # SL line
                if pos['sl'] != 0:
                    fig.add_hline(
                        y=pos['sl'],
                        line_dash="dash",
                        line_color="red",
                        annotation_text="SL"
                    )
                
                # Target line
                if pos['target'] != 0:
                    fig.add_hline(
                        y=pos['target'],
                        line_dash="dash",
                        line_color="green",
                        annotation_text="Target"
                    )
            
            fig.update_layout(
                title=f"{ticker} - {interval}",
                xaxis_title="Time",
                yaxis_title="Price",
                height=600,
                xaxis_rangeslider_visible=False
            )
            
            chart_key = f"live_chart_{get_ist_time().timestamp()}"
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
            
            # Trading Logic
            if mode == "Live Trading" and st.session_state['trading_active']:
                # Check if in position
                if st.session_state['position'] is None:
                    # Look for entry signal
                    signal = 0
                    entry_price = None
                    
                    if strategy == "EMA Crossover":
                        signal, entry_price = ema_crossover_strategy(df, current_idx, config)
                    
                    elif strategy == "Simple Buy":
                        signal, entry_price = simple_buy_strategy(df, current_idx)
                    
                    elif strategy == "Simple Sell":
                        signal, entry_price = simple_sell_strategy(df, current_idx)
                    
                    elif strategy == "Price Crosses Threshold":
                        # Always check threshold condition
                        signal, entry_price = price_threshold_strategy(df, current_idx, config)
                    
                    elif strategy == "RSI-ADX-EMA":
                        signal, entry_price = rsi_adx_ema_strategy(df, current_idx, config)
                    
                    elif strategy == "AI Price Action Analysis":
                        signal, entry_price, analysis = ai_price_action_strategy(df, current_idx, config)
                        
                        if signal != 0:
                            # Auto-set SL and Target using ATR
                            atr = df['ATR'].iloc[current_idx]
                            if signal == 1:
                                config['auto_sl'] = entry_price - (atr * 1.5)
                                config['auto_target'] = entry_price + (atr * 3)
                            else:
                                config['auto_sl'] = entry_price + (atr * 1.5)
                                config['auto_target'] = entry_price - (atr * 3)
                    
                    elif strategy == "Custom Strategy Builder":
                        signal, entry_price = custom_strategy(df, current_idx, st.session_state['custom_conditions'])
                    
                    # Enter position
                    if signal != 0:
                        sl = calculate_stop_loss(df, current_idx, entry_price, signal, config['sl_type'], config)
                        target = calculate_target(df, current_idx, entry_price, signal, config['target_type'], config)
                        
                        # Override with AI auto values if available
                        if strategy == "AI Price Action Analysis":
                            sl = config.get('auto_sl', sl)
                            target = config.get('auto_target', target)
                        
                        st.session_state['position'] = {
                            'signal': signal,
                            'entry_price': entry_price,
                            'entry_time': current_time,
                            'sl': sl,
                            'target': target,
                            'quantity': quantity,
                            'strategy': strategy
                        }
                        
                        # Initialize tracking variables
                        st.session_state['highest_price'] = entry_price
                        st.session_state['lowest_price'] = entry_price
                        st.session_state['trailing_sl_high'] = entry_price
                        st.session_state['trailing_sl_low'] = entry_price
                        st.session_state['trailing_profit_points'] = 0
                        st.session_state['partial_exit_done'] = False
                        st.session_state['breakeven_activated'] = False
                        
                        pos_type = "LONG" if signal == 1 else "SHORT"
                        add_log(f"ENTRY: {pos_type} at {entry_price:.2f} | SL: {sl:.2f} | Target: {target:.2f}")
                        
                        st.success(f"🎯 Entered {pos_type} position at {entry_price:.2f}")
                
                else:
                    # Manage existing position
                    pos = st.session_state['position']
                    
                    # Update highest/lowest prices
                    if st.session_state['highest_price'] is None or current_price > st.session_state['highest_price']:
                        st.session_state['highest_price'] = current_price
                    
                    if st.session_state['lowest_price'] is None or current_price < st.session_state['lowest_price']:
                        st.session_state['lowest_price'] = current_price
                    
                    # Update trailing SL
                    if 'Trailing' in config['sl_type']:
                        new_sl = update_trailing_sl(pos, current_price, config)
                        if new_sl != pos['sl']:
                            old_sl = pos['sl']
                            pos['sl'] = new_sl
                            st.session_state['position']['sl'] = new_sl
                            add_log(f"Trailing SL updated: {old_sl:.2f} → {new_sl:.2f}")
                    
                    # Update breakeven SL
                    if config['sl_type'] == 'Break-even After 50% Target':
                        new_sl = check_breakeven_sl(pos, current_price, config)
                        if new_sl != pos['sl']:
                            pos['sl'] = new_sl
                            st.session_state['position']['sl'] = new_sl
                    
                    # Update trailing target (display only)
                    if 'Trailing Target' in config['target_type']:
                        new_target = update_trailing_target(pos, current_price, config)
                        if new_target != pos['target']:
                            pos['target'] = new_target
                            st.session_state['position']['target'] = new_target
                    
                    # Check for exits
                    exit_trade = False
                    exit_reason = ""
                    exit_price = current_price
                    partial_exit = False
                    
                    # Check 50% partial exit
                    if '50% Exit' in config['target_type'] and not pos.get('partial_exit_done', False):
                        if pos['target'] != 0:
                            if pos['signal'] == 1 and current_price >= pos['target']:
                                partial_exit = True
                                st.session_state['partial_exit_done'] = True
                                pos['partial_exit_done'] = True
                                add_log("50% position exited at target - trailing remaining")
                            elif pos['signal'] == -1 and current_price <= pos['target']:
                                partial_exit = True
                                st.session_state['partial_exit_done'] = True
                                pos['partial_exit_done'] = True
                                add_log("50% position exited at target - trailing remaining")
                    
                    # Check signal-based exit
                    if config['sl_type'] == 'Signal-based' or config['target_type'] == 'Signal-based' or 'Signal Based' in config['sl_type'] or 'Signal Based' in config['target_type']:
                        if check_signal_based_exit(df, current_idx, pos):
                            exit_trade = True
                            exit_reason = "Reverse Signal Crossover"
                    
                    # Check SL hit
                    if not exit_trade and pos['sl'] != 0:
                        if pos['signal'] == 1 and current_price <= pos['sl']:
                            exit_trade = True
                            exit_reason = "Stop Loss Hit"
                            exit_price = pos['sl']
                        elif pos['signal'] == -1 and current_price >= pos['sl']:
                            exit_trade = True
                            exit_reason = "Stop Loss Hit"
                            exit_price = pos['sl']
                    
                    # Check target hit (not for trailing targets)
                    if not exit_trade and pos['target'] != 0 and 'Trailing Target' not in config['target_type']:
                        if pos['signal'] == 1 and current_price >= pos['target']:
                            exit_trade = True
                            exit_reason = "Target Reached"
                            exit_price = pos['target']
                        elif pos['signal'] == -1 and current_price <= pos['target']:
                            exit_trade = True
                            exit_reason = "Target Reached"
                            exit_price = pos['target']
                    
                    # Execute exit
                    if exit_trade:
                        # Calculate P&L
                        if pos['signal'] == 1:
                            pnl = (exit_price - pos['entry_price']) * pos['quantity']
                        else:
                            pnl = (pos['entry_price'] - exit_price) * pos['quantity']
                        
                        # Record trade
                        trade_record = {
                            'entry_time': pos['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                            'exit_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                            'duration': str(current_time - pos['entry_time']).split('.')[0],
                            'signal': 'LONG' if pos['signal'] == 1 else 'SHORT',
                            'entry_price': pos['entry_price'],
                            'exit_price': exit_price,
                            'sl': pos['sl'],
                            'target': pos['target'],
                            'exit_reason': exit_reason,
                            'pnl': pnl,
                            'quantity': pos['quantity'],
                            'strategy': pos['strategy'],
                            'highest_price': st.session_state.get('highest_price', 0),
                            'lowest_price': st.session_state.get('lowest_price', 0),
                            'range': st.session_state.get('highest_price', 0) - st.session_state.get('lowest_price', 0)
                        }
                        
                        st.session_state['trade_history'].append(trade_record)
                        st.session_state['trade_history'] = st.session_state['trade_history']
                        
                        add_log(f"EXIT: {exit_reason} at {exit_price:.2f} | P&L: {pnl:.2f}")
                        
                        # Reset position
                        reset_position_state()
                        
                        if pnl >= 0:
                            st.success(f"✅ Trade closed with profit: {pnl:.2f}")
                        else:
                            st.error(f"❌ Trade closed with loss: {pnl:.2f}")
                
                # Auto-refresh
                time.sleep(random.uniform(1.0, 1.5))
                st.rerun()
            
            # Manual refresh button
            if st.button("🔄 Manual Refresh"):
                st.rerun()
    
    with tab2:
        st.markdown("### 📈 Trade History")
        
        if len(st.session_state['trade_history']) == 0:
            st.info("No trades executed yet")
        else:
            # Calculate metrics
            total_trades = len(st.session_state['trade_history'])
            winning_trades = len([t for t in st.session_state['trade_history'] if t['pnl'] > 0])
            losing_trades = len([t for t in st.session_state['trade_history'] if t['pnl'] < 0])
            total_pnl = sum([t['pnl'] for t in st.session_state['trade_history']])
            accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Display metrics
            metric_cols = st.columns(5)
            
            with metric_cols[0]:
                st.metric("Total Trades", total_trades)
            
            with metric_cols[1]:
                st.metric("Winning", winning_trades)
            
            with metric_cols[2]:
                st.metric("Losing", losing_trades)
            
            with metric_cols[3]:
                st.metric("Accuracy", f"{accuracy:.1f}%")
            
            with metric_cols[4]:
                pnl_delta = f"+{total_pnl:.2f}" if total_pnl >= 0 else f"{total_pnl:.2f}"
                st.metric("Total P&L", f"{total_pnl:.2f}", delta=pnl_delta, delta_color="normal" if total_pnl >= 0 else "inverse")
            
            st.divider()
            
            # Display trades
            for idx, trade in enumerate(reversed(st.session_state['trade_history'])):
                with st.expander(f"Trade #{total_trades - idx} - {trade.get('signal', 'N/A')} - P&L: {trade.get('pnl', 0):.2f}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Entry Time:** {trade.get('entry_time', 'N/A')}")
                        st.markdown(f"**Exit Time:** {trade.get('exit_time', 'N/A')}")
                        st.markdown(f"**Duration:** {trade.get('duration', 'N/A')}")
                        st.markdown(f"**Strategy:** {trade.get('strategy', 'N/A')}")
                        st.markdown(f"**Signal:** {trade.get('signal', 'N/A')}")
                        st.markdown(f"**Quantity:** {trade.get('quantity', 1)}")
                    
                    with col2:
                        st.markdown(f"**Entry Price:** {trade.get('entry_price', 0):.2f}")
                        st.markdown(f"**Exit Price:** {trade.get('exit_price', 0):.2f}")
                        st.markdown(f"**Stop Loss:** {trade.get('sl', 0):.2f}")
                        st.markdown(f"**Target:** {trade.get('target', 0):.2f}")
                        st.markdown(f"**Exit Reason:** {trade.get('exit_reason', 'N/A')}")
                        
                        pnl = trade.get('pnl', 0)
                        pnl_color = "green" if pnl >= 0 else "red"
                        st.markdown(f"**P&L:** :{pnl_color}[{pnl:.2f}]")
                    
                    st.markdown("---")
                    st.markdown(f"**Highest Price:** {trade.get('highest_price', 0):.2f}")
                    st.markdown(f"**Lowest Price:** {trade.get('lowest_price', 0):.2f}")
                    st.markdown(f"**Range:** {trade.get('range', 0):.2f}")
    
    with tab3:
        st.markdown("### 📝 Trade Logs")
        
        if len(st.session_state['trade_logs']) == 0:
            st.info("No logs available")
        else:
            for log in reversed(st.session_state['trade_logs']):
                st.text(log)

    with tab4:
        st.markdown("### 📊 Backtest Results")
        
        if mode == "Backtest":
            if st.button("▶️ Run Backtest", type="primary"):
                with st.spinner("Running backtest..."):
                    # Fetch data
                    df = fetch_data(ticker, interval, period, 'backtest')
                    
                    if df is None or df.empty:
                        st.error("Failed to fetch data")
                    else:
                        # Calculate indicators
                        df['EMA_Fast'] = calculate_ema(df['Close'], config.get('ema_fast', 9))
                        df['EMA_Slow'] = calculate_ema(df['Close'], config.get('ema_slow', 15))
                        df['EMA_20'] = calculate_ema(df['Close'], 20)
                        df['EMA_50'] = calculate_ema(df['Close'], 50)
                        df['ATR'] = calculate_atr(df, 14)
                        df['RSI'] = calculate_rsi(df['Close'], config.get('rsi_period', 14))
                        df['ADX'] = calculate_adx(df, config.get('adx_period', 14))
                        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
                        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
                        df['VWAP'] = calculate_vwap(df)
                        df['Support'], df['Resistance'] = calculate_support_resistance(df)
                        df['Swing_High'], df['Swing_Low'] = detect_swing_high_low(df)
                        
                        # Run backtest
                        config['custom_conditions'] = st.session_state.get('custom_conditions', [])
                        results = run_backtest(df, strategy, config)
                        
                        # Display results
                        st.success("Backtest completed!")
                        
                        # Summary metrics
                        metric_cols = st.columns(5)
                        
                        with metric_cols[0]:
                            st.metric("Total Trades", results['total_trades'])
                        
                        with metric_cols[1]:
                            st.metric("Winning", results['winning_trades'])
                        
                        with metric_cols[2]:
                            st.metric("Losing", results['losing_trades'])
                        
                        with metric_cols[3]:
                            st.metric("Accuracy", f"{results['accuracy']:.1f}%")
                        
                        with metric_cols[4]:
                            pnl = results['total_pnl']
                            pnl_delta = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
                            st.metric("Total P&L", f"{pnl:.2f}", delta=pnl_delta, delta_color="normal" if pnl >= 0 else "inverse")
                        
                        st.divider()
                        
                        # Average duration
                        if results['total_trades'] > 0:
                            avg_duration_hours = results['avg_duration'] / 3600
                            st.markdown(f"**Average Trade Duration:** {avg_duration_hours:.2f} hours")
                        
                        st.divider()
                        
                        # Trade details
                        st.markdown("### 📋 Trade Details")
                        
                        for idx, trade in enumerate(results['trades']):
                            with st.expander(f"Trade #{idx + 1} - {trade['signal']} - P&L: {trade['pnl']:.2f}", expanded=False):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown(f"**Entry Time:** {trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                                    st.markdown(f"**Exit Time:** {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                                    st.markdown(f"**Duration:** {str(trade['duration']).split('.')[0]}")
                                    st.markdown(f"**Signal:** {trade['signal']}")
                                    st.markdown(f"**Entry Price:** {trade['entry_price']:.2f}")
                                
                                with col2:
                                    st.markdown(f"**Exit Price:** {trade['exit_price']:.2f}")
                                    st.markdown(f"**Stop Loss:** {trade['sl']:.2f}")
                                    st.markdown(f"**Target:** {trade['target']:.2f}")
                                    st.markdown(f"**Exit Reason:** {trade['exit_reason']}")
                                    
                                    pnl = trade['pnl']
                                    pnl_color = "green" if pnl >= 0 else "red"
                                    st.markdown(f"**P&L:** :{pnl_color}[{pnl:.2f}]")
                                
                                st.markdown("---")
                                st.markdown(f"**Highest Price:** {trade['highest_price']:.2f}")
                                st.markdown(f"**Lowest Price:** {trade['lowest_price']:.2f}")
                                st.markdown(f"**Range:** {trade['range']:.2f}")
        else:
            st.info("Switch to Backtest mode to run backtests")

if __name__ == "__main__":
    main()
