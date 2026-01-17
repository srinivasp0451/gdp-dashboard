import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import time
import random

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def log_message(message):
    """Add timestamped log to trade logs"""
    timestamp = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    st.session_state['trade_logs'].append(log_entry)
    # Keep only last 50 logs
    if len(st.session_state['trade_logs']) > 50:
        st.session_state['trade_logs'] = st.session_state['trade_logs'][-50:]
    st.session_state['trade_logs'] = st.session_state['trade_logs']

def reset_position_state():
    """Reset position state after trade exit"""
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

# =============================================================================
# INDICATOR CALCULATIONS (MANUALLY IMPLEMENTED)
# =============================================================================

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
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
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    
    atr = calculate_atr(df, period)
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return adx

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_supertrend(df, period=10, multiplier=3):
    """Calculate SuperTrend"""
    atr = calculate_atr(df, period)
    hl_avg = (df['High'] + df['Low']) / 2
    
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)
    
    for i in range(period, len(df)):
        if i == period:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        else:
            if df['Close'].iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
    
    return supertrend

def calculate_vwap(df):
    """Calculate Volume Weighted Average Price"""
    if 'Volume' not in df.columns or df['Volume'].sum() == 0:
        return pd.Series(index=df.index, dtype=float)
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap

def calculate_keltner_channel(df, period=20, atr_mult=2):
    """Calculate Keltner Channel"""
    ema = calculate_ema(df['Close'], period)
    atr = calculate_atr(df, period)
    upper = ema + (atr * atr_mult)
    lower = ema - (atr * atr_mult)
    return upper, ema, lower

def detect_swing_high_low(df, lookback=5):
    """Detect swing highs and lows"""
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
    
    swing_high_series = pd.Series([np.nan] * lookback + swing_highs + [np.nan] * lookback, index=df.index)
    swing_low_series = pd.Series([np.nan] * lookback + swing_lows + [np.nan] * lookback, index=df.index)
    
    return swing_high_series, swing_low_series

def calculate_ema_angle(df, column, period=9):
    """Calculate EMA angle in degrees"""
    ema = calculate_ema(df[column], period)
    
    angles = []
    for i in range(1, len(ema)):
        if pd.isna(ema.iloc[i]) or pd.isna(ema.iloc[i-1]):
            angles.append(0)
        else:
            slope = ema.iloc[i] - ema.iloc[i-1]
            angle = np.degrees(np.arctan(slope))
            angles.append(abs(angle))
    
    return pd.Series([0] + angles, index=df.index)

def calculate_support_resistance(df, lookback=20):
    """Calculate support and resistance levels"""
    highs = df['High'].rolling(window=lookback).max()
    lows = df['Low'].rolling(window=lookback).min()
    return highs, lows

# =============================================================================
# STRATEGY SIGNAL GENERATION
# =============================================================================

def generate_ema_crossover_signal(df, config):
    """Generate EMA Crossover signals with filters"""
    ema_fast = config['ema_fast']
    ema_slow = config['ema_slow']
    min_angle = config['min_angle']
    entry_filter = config['entry_filter']
    custom_points = config.get('custom_points', 10)
    atr_multiplier = config.get('atr_multiplier', 1.5)
    use_adx = config.get('use_adx', False)
    adx_period = config.get('adx_period', 14)
    adx_threshold = config.get('adx_threshold', 25)
    
    df['EMA_Fast'] = calculate_ema(df['Close'], ema_fast)
    df['EMA_Slow'] = calculate_ema(df['Close'], ema_slow)
    df['EMA_Angle'] = calculate_ema_angle(df, 'Close', ema_fast)
    
    if use_adx:
        df['ADX'] = calculate_adx(df, adx_period)
    
    if entry_filter in ['ATR-based Candle']:
        df['ATR'] = calculate_atr(df, 14)
    
    signals = []
    for i in range(1, len(df)):
        signal = 0
        
        # Bullish crossover
        if (df['EMA_Fast'].iloc[i] > df['EMA_Slow'].iloc[i] and 
            df['EMA_Fast'].iloc[i-1] <= df['EMA_Slow'].iloc[i-1]):
            
            if df['EMA_Angle'].iloc[i] >= min_angle:
                candle_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
                
                if entry_filter == 'Simple Crossover':
                    signal = 1
                elif entry_filter == 'Custom Candle (Points)':
                    if candle_size >= custom_points:
                        signal = 1
                elif entry_filter == 'ATR-based Candle':
                    min_candle = df['ATR'].iloc[i] * atr_multiplier
                    if candle_size >= min_candle:
                        signal = 1
                
                if signal == 1 and use_adx:
                    if df['ADX'].iloc[i] < adx_threshold:
                        signal = 0
        
        # Bearish crossover
        elif (df['EMA_Fast'].iloc[i] < df['EMA_Slow'].iloc[i] and 
              df['EMA_Fast'].iloc[i-1] >= df['EMA_Slow'].iloc[i-1]):
            
            if df['EMA_Angle'].iloc[i] >= min_angle:
                candle_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
                
                if entry_filter == 'Simple Crossover':
                    signal = -1
                elif entry_filter == 'Custom Candle (Points)':
                    if candle_size >= custom_points:
                        signal = -1
                elif entry_filter == 'ATR-based Candle':
                    min_candle = df['ATR'].iloc[i] * atr_multiplier
                    if candle_size >= min_candle:
                        signal = -1
                
                if signal == -1 and use_adx:
                    if df['ADX'].iloc[i] < adx_threshold:
                        signal = 0
        
        signals.append(signal)
    
    df['Signal'] = [0] + signals
    return df

def generate_simple_buy_signal(df):
    """Generate simple BUY signals"""
    df['Signal'] = 1
    return df

def generate_simple_sell_signal(df):
    """Generate simple SELL signals"""
    df['Signal'] = -1
    return df

def generate_price_threshold_signal(df, config):
    """Generate price threshold signals"""
    threshold = config['threshold']
    direction = config['direction']
    
    signals = []
    for i in range(len(df)):
        signal = 0
        current_price = df['Close'].iloc[i]
        
        if direction == 'LONG (Price >= Threshold)':
            if current_price >= threshold:
                signal = 1
        elif direction == 'SHORT (Price >= Threshold)':
            if current_price >= threshold:
                signal = -1
        elif direction == 'LONG (Price <= Threshold)':
            if current_price <= threshold:
                signal = 1
        elif direction == 'SHORT (Price <= Threshold)':
            if current_price <= threshold:
                signal = -1
        
        signals.append(signal)
    
    df['Signal'] = signals
    return df

def generate_rsi_adx_ema_signal(df, config):
    """Generate RSI-ADX-EMA signals"""
    rsi_period = config.get('rsi_period', 14)
    adx_period = config.get('adx_period', 14)
    ema1_period = config.get('ema1_period', 9)
    ema2_period = config.get('ema2_period', 21)
    
    df['RSI'] = calculate_rsi(df['Close'], rsi_period)
    df['ADX'] = calculate_adx(df, adx_period)
    df['EMA1'] = calculate_ema(df['Close'], ema1_period)
    df['EMA2'] = calculate_ema(df['Close'], ema2_period)
    
    signals = []
    for i in range(len(df)):
        signal = 0
        
        if (df['RSI'].iloc[i] > 80 and df['ADX'].iloc[i] < 20 and 
            df['EMA1'].iloc[i] < df['EMA2'].iloc[i]):
            signal = -1
        elif (df['RSI'].iloc[i] < 20 and df['ADX'].iloc[i] > 20 and 
              df['EMA1'].iloc[i] > df['EMA2'].iloc[i]):
            signal = 1
        
        signals.append(signal)
    
    df['Signal'] = signals
    return df

def generate_percentage_change_signal(df, config):
    """Generate percentage change signals"""
    threshold = config['percentage_threshold']
    direction = config['direction']
    
    first_price = df['Close'].iloc[0]
    
    signals = []
    for i in range(len(df)):
        signal = 0
        current_price = df['Close'].iloc[i]
        pct_change = ((current_price - first_price) / first_price) * 100
        
        if direction == 'BUY on Fall':
            if pct_change <= -threshold:
                signal = 1
        elif direction == 'SELL on Fall':
            if pct_change <= -threshold:
                signal = -1
        elif direction == 'BUY on Rise':
            if pct_change >= threshold:
                signal = 1
        elif direction == 'SELL on Rise':
            if pct_change >= threshold:
                signal = -1
        
        signals.append(signal)
    
    df['Signal'] = signals
    df['PctChange'] = ((df['Close'] - first_price) / first_price) * 100
    return df

def generate_ai_analysis_signal(df):
    """Generate AI-powered analysis signals"""
    # Calculate indicators
    df['EMA_20'] = calculate_ema(df['Close'], 20)
    df['EMA_50'] = calculate_ema(df['Close'], 50)
    df['RSI'] = calculate_rsi(df['Close'], 14)
    macd, macd_signal, macd_hist = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['Close'])
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    df['ATR'] = calculate_atr(df, 14)
    
    has_volume = 'Volume' in df.columns and df['Volume'].sum() > 0
    
    signals = []
    analysis_list = []
    
    for i in range(50, len(df)):
        score = 0
        analysis = {}
        
        # Trend Analysis
        if df['EMA_20'].iloc[i] > df['EMA_50'].iloc[i]:
            score += 2
            analysis['trend'] = 'Bullish (EMA 20 > EMA 50)'
        else:
            score -= 2
            analysis['trend'] = 'Bearish (EMA 20 < EMA 50)'
        
        # RSI Analysis
        if df['RSI'].iloc[i] < 30:
            score += 2
            analysis['rsi'] = 'Oversold - Buy signal'
        elif df['RSI'].iloc[i] > 70:
            score -= 2
            analysis['rsi'] = 'Overbought - Sell signal'
        else:
            analysis['rsi'] = 'Neutral'
        
        # MACD Analysis
        if df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]:
            score += 1
            analysis['macd'] = 'Bullish crossover'
        else:
            score -= 1
            analysis['macd'] = 'Bearish crossover'
        
        # Bollinger Bands
        if df['Close'].iloc[i] < df['BB_Lower'].iloc[i]:
            score += 1
            analysis['bb'] = 'Price below lower band - oversold'
        elif df['Close'].iloc[i] > df['BB_Upper'].iloc[i]:
            score -= 1
            analysis['bb'] = 'Price above upper band - overbought'
        else:
            analysis['bb'] = 'Price within bands'
        
        # Volume Analysis
        if has_volume:
            avg_volume = df['Volume'].iloc[i-20:i].mean()
            if df['Volume'].iloc[i] > avg_volume * 1.5:
                score += 1
                analysis['volume'] = 'High volume confirmation'
            else:
                analysis['volume'] = 'Normal volume'
        else:
            analysis['volume'] = 'No volume data'
        
        # Generate signal
        if score >= 3:
            signal = 1
        elif score <= -3:
            signal = -1
        else:
            signal = 0
        
        signals.append(signal)
        analysis['score'] = score
        analysis_list.append(analysis)
    
    df['Signal'] = [0] * 50 + signals
    df['AI_Analysis'] = [{}] * 50 + analysis_list
    
    return df

def generate_custom_strategy_signal(df, conditions):
    """Generate signals based on custom conditions"""
    # Calculate all possible indicators
    df['EMA_Fast'] = calculate_ema(df['Close'], 9)
    df['EMA_Slow'] = calculate_ema(df['Close'], 21)
    df['EMA_20'] = calculate_ema(df['Close'], 20)
    df['EMA_50'] = calculate_ema(df['Close'], 50)
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['ADX'] = calculate_adx(df, 14)
    df['ATR'] = calculate_atr(df, 14)
    df['SuperTrend'] = calculate_supertrend(df)
    macd, macd_signal, _ = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    bb_upper, _, bb_lower = calculate_bollinger_bands(df['Close'])
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    df['VWAP'] = calculate_vwap(df)
    kc_upper, kc_middle, kc_lower = calculate_keltner_channel(df)
    df['KC_Upper'] = kc_upper
    df['KC_Lower'] = kc_lower
    resistance, support = calculate_support_resistance(df)
    df['Support'] = support
    df['Resistance'] = resistance
    
    signals = []
    for i in range(1, len(df)):
        buy_conditions_met = []
        sell_conditions_met = []
        
        for cond in conditions:
            if not cond['use']:
                continue
            
            indicator = cond['indicator']
            operator = cond['operator']
            compare_price = cond.get('compare_price', False)
            action = cond['action']
            
            # Get indicator value
            if indicator == 'Price':
                ind_value = df['Close'].iloc[i]
            elif indicator in df.columns:
                ind_value = df[indicator].iloc[i]
            else:
                ind_value = 0
            
            # Get comparison value
            if compare_price:
                compare_indicator = cond.get('compare_indicator', 'EMA_20')
                if compare_indicator in df.columns:
                    comp_value = df[compare_indicator].iloc[i]
                else:
                    comp_value = 0
            else:
                comp_value = cond['value']
            
            # Evaluate condition
            condition_met = False
            if operator == '>':
                condition_met = ind_value > comp_value
            elif operator == '<':
                condition_met = ind_value < comp_value
            elif operator == '>=':
                condition_met = ind_value >= comp_value
            elif operator == '<=':
                condition_met = ind_value <= comp_value
            elif operator == '==':
                condition_met = abs(ind_value - comp_value) < 0.01
            elif operator == 'crosses_above':
                prev_ind = df[indicator].iloc[i-1] if indicator in df.columns else 0
                if compare_price and compare_indicator in df.columns:
                    prev_comp = df[compare_indicator].iloc[i-1]
                    condition_met = ind_value > comp_value and prev_ind <= prev_comp
                else:
                    condition_met = ind_value > comp_value and prev_ind <= comp_value
            elif operator == 'crosses_below':
                prev_ind = df[indicator].iloc[i-1] if indicator in df.columns else 0
                if compare_price and compare_indicator in df.columns:
                    prev_comp = df[compare_indicator].iloc[i-1]
                    condition_met = ind_value < comp_value and prev_ind >= prev_comp
                else:
                    condition_met = ind_value < comp_value and prev_ind >= comp_value
            
            if action == 'BUY':
                buy_conditions_met.append(condition_met)
            else:
                sell_conditions_met.append(condition_met)
        
        signal = 0
        if buy_conditions_met and all(buy_conditions_met):
            signal = 1
        elif sell_conditions_met and all(sell_conditions_met):
            signal = -1
        
        signals.append(signal)
    
    df['Signal'] = [0] + signals
    return df

# =============================================================================
# STOP LOSS CALCULATIONS
# =============================================================================

def calculate_stop_loss(df, i, entry_price, signal, sl_type, config):
    """Calculate stop loss based on type"""
    sl = 0
    
    if sl_type == 'Custom Points':
        points = config['sl_points']
        if signal == 1:
            sl = entry_price - points
        else:
            sl = entry_price + points
    
    elif sl_type == 'Trailing SL (Points)':
        points = config['sl_points']
        current_price = df['Close'].iloc[i]
        if signal == 1:
            sl = current_price - points
        else:
            sl = current_price + points
    
    elif sl_type == 'Trailing SL + Current Candle':
        if signal == 1:
            sl = df['Low'].iloc[i]
        else:
            sl = df['High'].iloc[i]
    
    elif sl_type == 'Trailing SL + Previous Candle':
        if i > 0:
            if signal == 1:
                sl = df['Low'].iloc[i-1]
            else:
                sl = df['High'].iloc[i-1]
    
    elif sl_type == 'Trailing SL + Current Swing':
        swing_high, swing_low = detect_swing_high_low(df)
        if signal == 1:
            sl = swing_low.iloc[i] if not pd.isna(swing_low.iloc[i]) else df['Low'].iloc[i]
        else:
            sl = swing_high.iloc[i] if not pd.isna(swing_high.iloc[i]) else df['High'].iloc[i]
    
    elif sl_type == 'Trailing SL + Previous Swing':
        swing_high, swing_low = detect_swing_high_low(df)
        valid_swings_low = swing_low[:i].dropna()
        valid_swings_high = swing_high[:i].dropna()
        if signal == 1 and len(valid_swings_low) > 0:
            sl = valid_swings_low.iloc[-1]
        elif signal == -1 and len(valid_swings_high) > 0:
            sl = valid_swings_high.iloc[-1]
    
    elif sl_type == 'Trailing SL + Signal Based':
        points = config['sl_points']
        if signal == 1:
            sl = entry_price - points
        else:
            sl = entry_price + points
    
    elif sl_type == 'Volatility-Adjusted Trailing SL':
        atr_mult = config.get('atr_multiplier', 2.0)
        atr = calculate_atr(df, 14).iloc[i]
        current_price = df['Close'].iloc[i]
        if signal == 1:
            sl = current_price - (atr * atr_mult)
        else:
            sl = current_price + (atr * atr_mult)
    
    elif sl_type == 'Break-even After 50% Target':
        points = config['sl_points']
        if signal == 1:
            sl = entry_price - points
        else:
            sl = entry_price + points
    
    elif sl_type == 'ATR-based':
        atr_mult = config.get('atr_multiplier', 2.0)
        atr = calculate_atr(df, 14).iloc[i]
        if signal == 1:
            sl = entry_price - (atr * atr_mult)
        else:
            sl = entry_price + (atr * atr_mult)
    
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
    
    elif sl_type == 'Current Swing Low/High':
        swing_high, swing_low = detect_swing_high_low(df)
        if signal == 1:
            sl = swing_low.iloc[i] if not pd.isna(swing_low.iloc[i]) else df['Low'].iloc[i]
        else:
            sl = swing_high.iloc[i] if not pd.isna(swing_high.iloc[i]) else df['High'].iloc[i]
    
    elif sl_type == 'Previous Swing Low/High':
        swing_high, swing_low = detect_swing_high_low(df)
        valid_swings_low = swing_low[:i].dropna()
        valid_swings_high = swing_high[:i].dropna()
        if signal == 1 and len(valid_swings_low) > 0:
            sl = valid_swings_low.iloc[-1]
        elif signal == -1 and len(valid_swings_high) > 0:
            sl = valid_swings_high.iloc[-1]
    
    elif sl_type == 'Signal-based (reverse EMA crossover)':
        sl = 0
    
    return sl

def update_trailing_sl(current_price, current_sl, signal, sl_type, config, position):
    """Update trailing stop loss"""
    if sl_type not in ['Trailing SL (Points)', 'Trailing SL + Signal Based', 
                       'Volatility-Adjusted Trailing SL']:
        return current_sl
    
    threshold = config.get('trailing_threshold', 0)
    points = config['sl_points']
    
    if sl_type == 'Trailing SL (Points)' or sl_type == 'Trailing SL + Signal Based':
        if signal == 1:
            new_sl = current_price - points
            profit = current_price - position['entry_price']
            if profit >= threshold and new_sl > current_sl:
                return new_sl
        else:
            new_sl = current_price + points
            profit = position['entry_price'] - current_price
            if profit >= threshold and new_sl < current_sl:
                return new_sl
    
    elif sl_type == 'Volatility-Adjusted Trailing SL':
        atr = position.get('atr', points)
        atr_mult = config.get('atr_multiplier', 2.0)
        if signal == 1:
            new_sl = current_price - (atr * atr_mult)
            if new_sl > current_sl:
                return new_sl
        else:
            new_sl = current_price + (atr * atr_mult)
            if new_sl < current_sl:
                return new_sl
    
    return current_sl

# =============================================================================
# TARGET CALCULATIONS
# =============================================================================

def calculate_target(df, i, entry_price, signal, target_type, config, sl_value=0):
    """Calculate target based on type"""
    target = 0
    
    if target_type == 'Custom Points':
        points = config['target_points']
        if signal == 1:
            target = entry_price + points
        else:
            target = entry_price - points
    
    elif target_type == 'Trailing Target (Points)':
        points = config['target_points']
        if signal == 1:
            target = entry_price + points
        else:
            target = entry_price - points
    
    elif target_type == 'Trailing Target + Signal Based':
        points = config['target_points']
        if signal == 1:
            target = entry_price + points
        else:
            target = entry_price - points
    
    elif target_type == '50% Exit at Target (Partial)':
        points = config['target_points']
        if signal == 1:
            target = entry_price + points
        else:
            target = entry_price - points
    
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
    
    elif target_type == 'Current Swing Low/High':
        swing_high, swing_low = detect_swing_high_low(df)
        if signal == 1:
            target = swing_high.iloc[i] if not pd.isna(swing_high.iloc[i]) else df['High'].iloc[i]
        else:
            target = swing_low.iloc[i] if not pd.isna(swing_low.iloc[i]) else df['Low'].iloc[i]
    
    elif target_type == 'Previous Swing Low/High':
        swing_high, swing_low = detect_swing_high_low(df)
        valid_swings_high = swing_high[:i].dropna()
        valid_swings_low = swing_low[:i].dropna()
        if signal == 1 and len(valid_swings_high) > 0:
            target = valid_swings_high.iloc[-1]
        elif signal == -1 and len(valid_swings_low) > 0:
            target = valid_swings_low.iloc[-1]
    
    elif target_type == 'ATR-based':
        atr_mult = config.get('target_atr_multiplier', 3.0)
        atr = calculate_atr(df, 14).iloc[i]
        if signal == 1:
            target = entry_price + (atr * atr_mult)
        else:
            target = entry_price - (atr * atr_mult)
    
    elif target_type == 'Risk-Reward Based':
        rr_ratio = config.get('rr_ratio', 2.0)
        sl_distance = abs(entry_price - sl_value)
        if signal == 1:
            target = entry_price + (sl_distance * rr_ratio)
        else:
            target = entry_price - (sl_distance * rr_ratio)
    
    elif target_type == 'Signal-based (reverse EMA crossover)':
        target = 0
    
    return target

def update_trailing_target(current_price, entry_price, signal, target_type, config, position):
    """Update trailing target - DISPLAY ONLY, NEVER EXITS"""
    if target_type not in ['Trailing Target (Points)', 'Trailing Target + Signal Based']:
        return position.get('target', 0)
    
    target_points = config['target_points']
    trailing_profit = position.get('trailing_profit_points', 0)
    
    if signal == 1:
        highest = position.get('highest_price', current_price)
        profit_points = highest - entry_price
        
        if profit_points >= trailing_profit + target_points:
            position['trailing_profit_points'] = profit_points
            position['highest_price'] = highest
            return highest
    else:
        lowest = position.get('lowest_price', current_price)
        profit_points = entry_price - lowest
        
        if profit_points >= trailing_profit + target_points:
            position['trailing_profit_points'] = profit_points
            position['lowest_price'] = lowest
            return lowest
    
    return position.get('target', 0)

# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_data(symbol, interval, period, mode='Backtest'):
    """Fetch data from yfinance with rate limiting"""
    try:
        if mode == 'Live Trading':
            delay = random.uniform(1.0, 1.5)
            time.sleep(delay)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(interval=interval, period=period)
        
        if df.empty:
            return None
        
        # Flatten multi-index if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Select OHLCV
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            return None
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Convert to IST
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            df.index = df.index.tz_convert('Asia/Kolkata')
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(df, strategy, config, quantity):
    """Run backtest on historical data"""
    results = {
        'trades': [],
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'total_pnl': 0,
        'accuracy': 0,
        'avg_duration': 0
    }
    
    position = None
    
    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        
        # Check for exit if in position
        if position is not None:
            exit_price = None
            exit_reason = None
            
            # Check signal-based exit
            if config['sl_type'] == 'Signal-based (reverse EMA crossover)' or \
               config['target_type'] == 'Signal-based (reverse EMA crossover)':
                if position['signal'] == 1:
                    if (df['EMA_Fast'].iloc[i] < df['EMA_Slow'].iloc[i] and 
                        df['EMA_Fast'].iloc[i-1] >= df['EMA_Slow'].iloc[i-1]):
                        exit_reason = 'Reverse Signal - Bearish Crossover'
                        exit_price = current_price
                elif position['signal'] == -1:
                    if (df['EMA_Fast'].iloc[i] > df['EMA_Slow'].iloc[i] and 
                        df['EMA_Fast'].iloc[i-1] <= df['EMA_Slow'].iloc[i-1]):
                        exit_reason = 'Reverse Signal - Bullish Crossover'
                        exit_price = current_price
            
            # Update trailing values
            if position['signal'] == 1:
                if position['highest_price'] is None or current_price > position['highest_price']:
                    position['highest_price'] = current_price
            else:
                if position['lowest_price'] is None or current_price < position['lowest_price']:
                    position['lowest_price'] = current_price
            
            # Update trailing SL
            if config['sl_type'] in ['Trailing SL (Points)', 'Trailing SL + Signal Based', 
                                     'Volatility-Adjusted Trailing SL']:
                position['sl'] = update_trailing_sl(current_price, position['sl'], 
                                                    position['signal'], config['sl_type'], 
                                                    config, position)
            
            # Check break-even
            if config['sl_type'] == 'Break-even After 50% Target' and not position.get('breakeven_activated', False):
                if position['signal'] == 1:
                    profit = current_price - position['entry_price']
                    target_dist = position['target'] - position['entry_price']
                    if profit >= target_dist * 0.5:
                        position['sl'] = position['entry_price']
                        position['breakeven_activated'] = True
                else:
                    profit = position['entry_price'] - current_price
                    target_dist = position['entry_price'] - position['target']
                    if profit >= target_dist * 0.5:
                        position['sl'] = position['entry_price']
                        position['breakeven_activated'] = True
            
            # Check SL hit
            if config['sl_type'] != 'Signal-based (reverse EMA crossover)' and exit_reason is None:
                if position['signal'] == 1 and current_price <= position['sl']:
                    exit_reason = 'Stop Loss Hit'
                    exit_price = position['sl']
                elif position['signal'] == -1 and current_price >= position['sl']:
                    exit_reason = 'Stop Loss Hit'
                    exit_price = position['sl']
            
            # Check target hit (not for trailing targets)
            if config['target_type'] not in ['Trailing Target (Points)', 'Trailing Target + Signal Based', 
                                             'Signal-based (reverse EMA crossover)'] and exit_reason is None:
                if config['target_type'] == '50% Exit at Target (Partial)':
                    if not position.get('partial_exit_done', False):
                        if position['signal'] == 1 and current_price >= position['target']:
                            position['partial_exit_done'] = True
                        elif position['signal'] == -1 and current_price <= position['target']:
                            position['partial_exit_done'] = True
                else:
                    if position['signal'] == 1 and current_price >= position['target']:
                        exit_reason = 'Target Hit'
                        exit_price = position['target']
                    elif position['signal'] == -1 and current_price <= position['target']:
                        exit_reason = 'Target Hit'
                        exit_price = position['target']
            
            # Exit position
            if exit_reason is not None:
                exit_time = df.index[i]
                duration = (exit_time - position['entry_time']).total_seconds() / 3600
                
                if position['signal'] == 1:
                    pnl = (exit_price - position['entry_price']) * quantity
                else:
                    pnl = (position['entry_price'] - exit_price) * quantity
                
                trade = {
                    'entry_time': position['entry_time'],
                    'exit_time': exit_time,
                    'duration': duration,
                    'signal': 'LONG' if position['signal'] == 1 else 'SHORT',
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'sl': position['sl'],
                    'target': position['target'],
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'highest': position['highest_price'],
                    'lowest': position['lowest_price'],
                    'range': position['highest_price'] - position['lowest_price']
                }
                
                results['trades'].append(trade)
                results['total_trades'] += 1
                results['total_pnl'] += pnl
                
                if pnl > 0:
                    results['winning_trades'] += 1
                else:
                    results['losing_trades'] += 1
                
                position = None
        
        # Check for entry if no position
        if position is None and df['Signal'].iloc[i] != 0:
            signal = df['Signal'].iloc[i]
            entry_price = current_price
            
            sl = calculate_stop_loss(df, i, entry_price, signal, config['sl_type'], config)
            target = calculate_target(df, i, entry_price, signal, config['target_type'], config, sl)
            
            # Apply minimum distances
            min_sl_dist = config.get('min_sl_distance', 10)
            min_target_dist = config.get('min_target_distance', 15)
            
            if signal == 1:
                if abs(entry_price - sl) < min_sl_dist:
                    sl = entry_price - min_sl_dist
                if abs(target - entry_price) < min_target_dist:
                    target = entry_price + min_target_dist
            else:
                if abs(sl - entry_price) < min_sl_dist:
                    sl = entry_price + min_sl_dist
                if abs(entry_price - target) < min_target_dist:
                    target = entry_price - min_target_dist
            
            position = {
                'entry_time': df.index[i],
                'entry_price': entry_price,
                'signal': signal,
                'sl': sl,
                'target': target,
                'highest_price': entry_price,
                'lowest_price': entry_price,
                'partial_exit_done': False,
                'breakeven_activated': False,
                'trailing_profit_points': 0
            }
    
    # Calculate statistics
    if results['total_trades'] > 0:
        results['accuracy'] = (results['winning_trades'] / results['total_trades']) * 100
        avg_duration = sum(t['duration'] for t in results['trades']) / results['total_trades']
        results['avg_duration'] = avg_duration
    
    return results

# =============================================================================
# LIVE TRADING LOGIC
# =============================================================================

def process_live_trading(df, strategy, config, quantity):
    """Process live trading logic"""
    if df is None or len(df) == 0:
        return
    
    current_price = df['Close'].iloc[-1]
    position = st.session_state.get('position')
    
    # Update highest/lowest prices
    if position is not None:
        if position['signal'] == 1:
            if st.session_state['highest_price'] is None or current_price > st.session_state['highest_price']:
                st.session_state['highest_price'] = current_price
        else:
            if st.session_state['lowest_price'] is None or current_price < st.session_state['lowest_price']:
                st.session_state['lowest_price'] = current_price
        
        position['highest_price'] = st.session_state['highest_price']
        position['lowest_price'] = st.session_state['lowest_price']
    
    # Check for exit if in position
    if position is not None:
        exit_price = None
        exit_reason = None
        
        # Check signal-based exit
        if config['sl_type'] == 'Signal-based (reverse EMA crossover)' or \
           config['target_type'] == 'Signal-based (reverse EMA crossover)':
            if position['signal'] == 1:
                if (df['EMA_Fast'].iloc[-1] < df['EMA_Slow'].iloc[-1] and 
                    df['EMA_Fast'].iloc[-2] >= df['EMA_Slow'].iloc[-2]):
                    exit_reason = 'Reverse Signal - Bearish Crossover'
                    exit_price = current_price
            elif position['signal'] == -1:
                if (df['EMA_Fast'].iloc[-1] > df['EMA_Slow'].iloc[-1] and 
                    df['EMA_Fast'].iloc[-2] <= df['EMA_Slow'].iloc[-2]):
                    exit_reason = 'Reverse Signal - Bullish Crossover'
                    exit_price = current_price
        
        # Update trailing SL
        if config['sl_type'] in ['Trailing SL (Points)', 'Trailing SL + Signal Based', 
                                 'Volatility-Adjusted Trailing SL']:
            new_sl = update_trailing_sl(current_price, position['sl'], position['signal'], 
                                        config['sl_type'], config, position)
            if new_sl != position['sl']:
                log_message(f"Trailing SL updated: {position['sl']:.2f} -> {new_sl:.2f}")
                position['sl'] = new_sl
                st.session_state['position'] = position
        
        # Check break-even
        if config['sl_type'] == 'Break-even After 50% Target' and not st.session_state.get('breakeven_activated', False):
            if position['signal'] == 1:
                profit = current_price - position['entry_price']
                target_dist = position['target'] - position['entry_price']
                if profit >= target_dist * 0.5:
                    position['sl'] = position['entry_price']
                    st.info(f"**Highest:** {trade.get('highest', 0):.2f} | **Lowest:** {trade.get('lowest', 0):.2f} | **Range:** {trade.get('range', 0):.2f}")
    
    # Tab 3: Trade Logs
    with tab3:
        st.markdown("### 📝 Trade Logs")
        
        if len(st.session_state['trade_logs']) == 0:
            st.info("No logs yet. Start trading to see logs.")
        else:
            for log in reversed(st.session_state['trade_logs']):
                st.text(log)
    
    # Tab 4: Backtest Results
    with tab4:
        if mode == "Backtest":
            st.markdown("### 🔬 Backtest Results")
            
            if st.button("▶️ Run Backtest", type="primary"):
                with st.spinner("Running backtest..."):
                    # Clear cached data
                    if 'backtest_results' in st.session_state:
                        del st.session_state['backtest_results']
                    
                    # Fetch data
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Fetching historical data...")
                    progress_bar.progress(20)
                    
                    df = fetch_data(symbol, interval, period, mode='Backtest')
                    
                    if df is not None:
                        progress_bar.progress(40)
                        status_text.text("Generating signals...")
                        
                        # Generate signals based on strategy
                        if strategy == "EMA Crossover":
                            df = generate_ema_crossover_signal(df, config)
                        elif strategy == "Simple Buy":
                            df = generate_simple_buy_signal(df)
                        elif strategy == "Simple Sell":
                            df = generate_simple_sell_signal(df)
                        elif strategy == "Price Crosses Threshold":
                            df = generate_price_threshold_signal(df, config)
                        elif strategy == "RSI-ADX-EMA":
                            df = generate_rsi_adx_ema_signal(df, config)
                        elif strategy == "Percentage Change":
                            df = generate_percentage_change_signal(df, config)
                        elif strategy == "AI Price Action Analysis":
                            df = generate_ai_analysis_signal(df)
                        elif strategy == "Custom Strategy Builder":
                            df = generate_custom_strategy_signal(df, st.session_state['custom_conditions'])
                        
                        progress_bar.progress(60)
                        status_text.text("Running backtest...")
                        
                        # Run backtest
                        results = run_backtest(df, strategy, config, quantity)
                        
                        progress_bar.progress(100)
                        status_text.text("Backtest completed!")
                        
                        st.session_state['backtest_results'] = results
                        st.success("✅ Backtest completed successfully!")
                    else:
                        st.error("Failed to fetch data for backtesting")
            
            # Display backtest results
            if 'backtest_results' in st.session_state:
                results = st.session_state['backtest_results']
                
                st.markdown("### 📊 Performance Metrics")
                
                metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
                
                with metric_col1:
                    st.metric("Total Trades", results['total_trades'])
                
                with metric_col2:
                    st.metric("Winning Trades", results['winning_trades'])
                
                with metric_col3:
                    st.metric("Losing Trades", results['losing_trades'])
                
                with metric_col4:
                    st.metric("Accuracy", f"{results['accuracy']:.2f}%")
                
                with metric_col5:
                    total_pnl = results['total_pnl']
                    if total_pnl >= 0:
                        st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"+{total_pnl:.2f}")
                    else:
                        st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color="inverse")
                
                st.metric("Average Trade Duration", f"{results['avg_duration']:.2f} hours")
                
                st.markdown("---")
                st.markdown("### 📋 All Trades")
                
                if len(results['trades']) == 0:
                    st.info("No trades were executed during the backtest period.")
                else:
                    for idx, trade in enumerate(results['trades']):
                        with st.expander(f"Trade #{idx + 1} - {trade['signal']} - P&L: {trade['pnl']:.2f}"):
                            trade_col1, trade_col2 = st.columns(2)
                            
                            with trade_col1:
                                entry_time_str = trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')
                                exit_time_str = trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')
                                
                                st.write(f"**Entry Time:** {entry_time_str}")
                                st.write(f"**Exit Time:** {exit_time_str}")
                                st.write(f"**Duration:** {trade['duration']:.2f} hours")
                                st.write(f"**Signal:** {trade['signal']}")
                                st.write(f"**Entry Price:** {trade['entry_price']:.2f}")
                            
                            with trade_col2:
                                st.write(f"**Exit Price:** {trade['exit_price']:.2f}")
                                
                                sl_str = f"{trade['sl']:.2f}" if trade['sl'] != 0 else "Signal Based"
                                st.write(f"**Stop Loss:** {sl_str}")
                                
                                target_str = f"{trade['target']:.2f}" if trade['target'] != 0 else "Signal Based"
                                st.write(f"**Target:** {target_str}")
                                
                                st.write(f"**Exit Reason:** {trade['exit_reason']}")
                                
                                if trade['pnl'] >= 0:
                                    st.success(f"**P&L:** +{trade['pnl']:.2f}")
                                else:
                                    st.error(f"**P&L:** {trade['pnl']:.2f}")
                            
                            st.info(f"**Highest:** {trade.get('highest', 0):.2f} | **Lowest:** {trade.get('lowest', 0):.2f} | **Range:** {trade.get('range', 0):.2f}")
        else:
            st.info("Backtest Results are only available in Backtest mode. Switch to Backtest in the sidebar.")

if __name__ == "__main__":
    main().session_state['breakeven_activated'] = True
                    st.session_state['position'] = position
                    log_message("Break-even activated - SL moved to entry price")
            else:
                profit = position['entry_price'] - current_price
                target_dist = position['entry_price'] - position['target']
                if profit >= target_dist * 0.5:
                    position['sl'] = position['entry_price']
                    st.session_state['breakeven_activated'] = True
                    st.session_state['position'] = position
                    log_message("Break-even activated - SL moved to entry price")
        
        # Check SL hit
        if config['sl_type'] != 'Signal-based (reverse EMA crossover)' and exit_reason is None:
            if position['signal'] == 1 and current_price <= position['sl']:
                exit_reason = 'Stop Loss Hit'
                exit_price = position['sl']
            elif position['signal'] == -1 and current_price >= position['sl']:
                exit_reason = 'Stop Loss Hit'
                exit_price = position['sl']
        
        # Check target hit
        if config['target_type'] not in ['Trailing Target (Points)', 'Trailing Target + Signal Based', 
                                         'Signal-based (reverse EMA crossover)'] and exit_reason is None:
            if config['target_type'] == '50% Exit at Target (Partial)':
                if not st.session_state.get('partial_exit_done', False):
                    if position['signal'] == 1 and current_price >= position['target']:
                        st.session_state['partial_exit_done'] = True
                        log_message("50% position exited at target - trailing remaining")
                    elif position['signal'] == -1 and current_price <= position['target']:
                        st.session_state['partial_exit_done'] = True
                        log_message("50% position exited at target - trailing remaining")
            else:
                if position['signal'] == 1 and current_price >= position['target']:
                    exit_reason = 'Target Hit'
                    exit_price = position['target']
                elif position['signal'] == -1 and current_price <= position['target']:
                    exit_reason = 'Target Hit'
                    exit_price = position['target']
        
        # Exit position
        if exit_reason is not None:
            exit_time = datetime.now(pytz.timezone('Asia/Kolkata'))
            duration = (exit_time - position['entry_time']).total_seconds() / 3600
            
            if position['signal'] == 1:
                pnl = (exit_price - position['entry_price']) * quantity
            else:
                pnl = (position['entry_price'] - exit_price) * quantity
            
            trade = {
                'entry_time': position['entry_time'],
                'exit_time': exit_time,
                'duration': duration,
                'signal': 'LONG' if position['signal'] == 1 else 'SHORT',
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'sl': position['sl'],
                'target': position['target'],
                'exit_reason': exit_reason,
                'pnl': pnl,
                'highest': st.session_state['highest_price'],
                'lowest': st.session_state['lowest_price'],
                'range': st.session_state['highest_price'] - st.session_state['lowest_price']
            }
            
            st.session_state['trade_history'].append(trade)
            st.session_state['trade_history'] = st.session_state['trade_history']
            
            log_message(f"Position CLOSED: {exit_reason} | PnL: {pnl:.2f}")
            reset_position_state()
    
    # Check for entry if no position
    if st.session_state['position'] is None and df['Signal'].iloc[-1] != 0:
        signal = df['Signal'].iloc[-1]
        entry_price = current_price
        
        sl = calculate_stop_loss(df, len(df)-1, entry_price, signal, config['sl_type'], config)
        target = calculate_target(df, len(df)-1, entry_price, signal, config['target_type'], config, sl)
        
        # Apply minimum distances
        min_sl_dist = config.get('min_sl_distance', 10)
        min_target_dist = config.get('min_target_distance', 15)
        
        if signal == 1:
            if abs(entry_price - sl) < min_sl_dist:
                sl = entry_price - min_sl_dist
            if abs(target - entry_price) < min_target_dist:
                target = entry_price + min_target_dist
        else:
            if abs(sl - entry_price) < min_sl_dist:
                sl = entry_price + min_sl_dist
            if abs(entry_price - target) < min_target_dist:
                target = entry_price - min_target_dist
        
        position = {
            'entry_time': datetime.now(pytz.timezone('Asia/Kolkata')),
            'entry_price': entry_price,
            'signal': signal,
            'sl': sl,
            'target': target,
            'highest_price': entry_price,
            'lowest_price': entry_price
        }
        
        st.session_state['position'] = position
        st.session_state['highest_price'] = entry_price
        st.session_state['lowest_price'] = entry_price
        st.session_state['partial_exit_done'] = False
        st.session_state['breakeven_activated'] = False
        
        signal_type = 'LONG' if signal == 1 else 'SHORT'
        log_message(f"Position OPENED: {signal_type} at {entry_price:.2f} | SL: {sl:.2f} | Target: {target:.2f}")

# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(page_title="Quantitative Trading System", layout="wide")
    st.title("🚀 Professional Quantitative Trading System")
    
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
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Asset selection
        asset_type = st.selectbox("Asset Type", 
            ["Indian Indices", "Crypto", "Forex", "Commodities", "Custom"])
        
        if asset_type == "Indian Indices":
            symbol = st.selectbox("Symbol", ["^NSEI", "^NSEBANK", "^BSESN"])
        elif asset_type == "Crypto":
            symbol = st.selectbox("Symbol", ["BTC-USD", "ETH-USD"])
        elif asset_type == "Forex":
            symbol = st.selectbox("Symbol", ["USDINR=X", "EURUSD=X", "GBPUSD=X"])
        elif asset_type == "Commodities":
            symbol = st.selectbox("Symbol", ["GC=F", "SI=F"])
        else:
            symbol = st.text_input("Custom Ticker", value="AAPL")
        
        # Timeframe and period
        interval = st.selectbox("Interval", 
            ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"])
        
        # Period validation
        period_options = {
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
        
        period = st.selectbox("Period", period_options.get(interval, ["1mo"]))
        
        # Quantity
        quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="qty_input")
        
        # Mode selection
        mode = st.selectbox("Mode", ["Backtest", "Live Trading"])
        
        st.markdown("---")
        
        # Strategy selection
        strategy = st.selectbox("Strategy", [
            "EMA Crossover",
            "Simple Buy",
            "Simple Sell",
            "Price Crosses Threshold",
            "RSI-ADX-EMA",
            "Percentage Change",
            "AI Price Action Analysis",
            "Custom Strategy Builder"
        ])
        
        config = {}
        
        # Strategy-specific parameters
        if strategy == "EMA Crossover":
            st.subheader("EMA Parameters")
            config['ema_fast'] = st.number_input("EMA Fast", min_value=1, value=9, step=1, key="ema_fast_input")
            config['ema_slow'] = st.number_input("EMA Slow", min_value=1, value=15, step=1, key="ema_slow_input")
            config['min_angle'] = st.number_input("Min Angle (degrees)", min_value=0.0, value=1.0, step=0.1, key="min_angle_input")
            
            st.subheader("Entry Filter")
            config['entry_filter'] = st.selectbox("Entry Filter Type", 
                ["Simple Crossover", "Custom Candle (Points)", "ATR-based Candle"])
            
            if config['entry_filter'] == "Custom Candle (Points)":
                config['custom_points'] = st.number_input("Custom Points", min_value=1.0, value=10.0, step=1.0, key="custom_pts_input")
            elif config['entry_filter'] == "ATR-based Candle":
                config['atr_multiplier'] = st.number_input("ATR Multiplier", min_value=0.1, value=1.5, step=0.1, key="atr_mult_input")
            
            st.subheader("ADX Filter")
            config['use_adx'] = st.checkbox("Use ADX Filter", value=False)
            if config['use_adx']:
                config['adx_period'] = st.number_input("ADX Period", min_value=1, value=14, step=1, key="adx_period_input")
                config['adx_threshold'] = st.number_input("ADX Threshold", min_value=1.0, value=25.0, step=1.0, key="adx_threshold_input")
        
        elif strategy == "Price Crosses Threshold":
            config['threshold'] = st.number_input("Price Threshold", min_value=0.0, value=100.0, step=1.0, key="threshold_input")
            config['direction'] = st.selectbox("Direction", [
                "LONG (Price >= Threshold)",
                "SHORT (Price >= Threshold)",
                "LONG (Price <= Threshold)",
                "SHORT (Price <= Threshold)"
            ])
        
        elif strategy == "RSI-ADX-EMA":
            config['rsi_period'] = st.number_input("RSI Period", min_value=1, value=14, step=1, key="rsi_period_input")
            config['adx_period'] = st.number_input("ADX Period", min_value=1, value=14, step=1, key="adx_period_rsi_input")
            config['ema1_period'] = st.number_input("EMA1 Period", min_value=1, value=9, step=1, key="ema1_period_input")
            config['ema2_period'] = st.number_input("EMA2 Period", min_value=1, value=21, step=1, key="ema2_period_input")
        
        elif strategy == "Percentage Change":
            config['percentage_threshold'] = st.number_input("Percentage Threshold (%)", 
                min_value=0.001, value=0.01, step=0.001, format="%.3f", key="pct_threshold_input")
            config['direction'] = st.selectbox("Direction", [
                "BUY on Fall",
                "SELL on Fall",
                "BUY on Rise",
                "SELL on Rise"
            ])
        
        elif strategy == "Custom Strategy Builder":
            st.subheader("Custom Conditions")
            
            if st.button("Add Condition"):
                st.session_state['custom_conditions'].append({
                    'use': True,
                    'compare_price': False,
                    'indicator': 'RSI',
                    'operator': '>',
                    'value': 50,
                    'compare_indicator': 'EMA_20',
                    'action': 'BUY'
                })
            
            conditions_to_remove = []
            for idx, cond in enumerate(st.session_state['custom_conditions']):
                st.markdown(f"**Condition {idx + 1}**")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    cond['use'] = st.checkbox("Use", value=cond['use'], key=f"use_{idx}")
                
                with col2:
                    if st.button("Remove", key=f"remove_{idx}"):
                        conditions_to_remove.append(idx)
                
                if cond['use']:
                    cond['compare_price'] = st.checkbox("Compare Price with Indicator", 
                        value=cond.get('compare_price', False), key=f"compare_price_{idx}")
                    
                    if cond['compare_price']:
                        cond['indicator'] = 'Price'
                        cond['compare_indicator'] = st.selectbox("Compare with", 
                            ['EMA_20', 'EMA_50', 'EMA_Fast', 'EMA_Slow', 'SuperTrend', 'VWAP'],
                            key=f"comp_ind_{idx}")
                    else:
                        cond['indicator'] = st.selectbox("Indicator",
                            ['Price', 'RSI', 'ADX', 'EMA_Fast', 'EMA_Slow', 'SuperTrend', 'EMA_20', 
                             'EMA_50', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR', 
                             'Volume', 'VWAP', 'KC_Upper', 'KC_Lower', 'Close', 'High', 'Low', 
                             'Support', 'Resistance'],
                            key=f"ind_{idx}")
                    
                    cond['operator'] = st.selectbox("Operator",
                        ['>', '<', '>=', '<=', '==', 'crosses_above', 'crosses_below'],
                        key=f"op_{idx}")
                    
                    if not cond['compare_price']:
                        cond['value'] = st.number_input("Value", value=float(cond['value']), 
                            key=f"val_{idx}")
                    
                    cond['action'] = st.selectbox("Action", ['BUY', 'SELL'], key=f"act_{idx}")
                
                st.markdown("---")
            
            for idx in reversed(conditions_to_remove):
                st.session_state['custom_conditions'].pop(idx)
        
        st.markdown("---")
        
        # Stop Loss configuration
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
            "Signal-based (reverse EMA crossover)"
        ])
        
        if config['sl_type'] not in ["Signal-based (reverse EMA crossover)"]:
            config['sl_points'] = st.number_input("SL Points", min_value=1.0, value=10.0, step=1.0, key="sl_points_input")
        
        if 'Trailing' in config['sl_type']:
            config['trailing_threshold'] = st.number_input("Trailing Threshold (points)", 
                min_value=0.0, value=0.0, step=1.0, key="trail_threshold_input")
        
        if 'ATR' in config['sl_type'] or config['sl_type'] == 'Volatility-Adjusted Trailing SL':
            config['atr_multiplier'] = st.number_input("ATR Multiplier (SL)", 
                min_value=0.1, value=2.0, step=0.1, key="atr_mult_sl_input")
        
        config['min_sl_distance'] = st.number_input("Min SL Distance (points)", 
            min_value=0.0, value=10.0, step=1.0, key="min_sl_dist_input")
        
        st.markdown("---")
        
        # Target configuration
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
            "Signal-based (reverse EMA crossover)"
        ])
        
        if config['target_type'] not in ["Signal-based (reverse EMA crossover)"]:
            config['target_points'] = st.number_input("Target Points", min_value=1.0, value=20.0, step=1.0, key="target_points_input")
        
        if config['target_type'] == 'ATR-based':
            config['target_atr_multiplier'] = st.number_input("ATR Multiplier (Target)", 
                min_value=0.1, value=3.0, step=0.1, key="atr_mult_target_input")
        
        if config['target_type'] == 'Risk-Reward Based':
            config['rr_ratio'] = st.number_input("Risk-Reward Ratio", 
                min_value=0.1, value=2.0, step=0.1, key="rr_ratio_input")
        
        config['min_target_distance'] = st.number_input("Min Target Distance (points)", 
            min_value=0.0, value=15.0, step=1.0, key="min_target_dist_input")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Trading Dashboard", 
        "📈 Trade History", 
        "📝 Trade Logs", 
        "🔬 Backtest Results"
    ])
    
    # Tab 1: Live Trading Dashboard
    with tab1:
        if mode == "Live Trading":
            st.markdown("### 🎛️ Trading Controls")
            
            col1, col2, col3 = st.columns([2, 2, 2])
            
            with col1:
                if st.button("▶️ Start Trading", type="primary", use_container_width=True):
                    st.session_state['trading_active'] = True
                    log_message("Trading started")
            
            with col2:
                if st.button("⏸️ Stop Trading", use_container_width=True):
                    if st.session_state['trading_active']:
                        st.session_state['trading_active'] = False
                        
                        # Close open position if exists
                        if st.session_state['position'] is not None:
                            position = st.session_state['position']
                            exit_time = datetime.now(pytz.timezone('Asia/Kolkata'))
                            duration = (exit_time - position['entry_time']).total_seconds() / 3600
                            
                            current_price = st.session_state['current_data']['Close'].iloc[-1]
                            if position['signal'] == 1:
                                pnl = (current_price - position['entry_price']) * quantity
                            else:
                                pnl = (position['entry_price'] - current_price) * quantity
                            
                            trade = {
                                'entry_time': position['entry_time'],
                                'exit_time': exit_time,
                                'duration': duration,
                                'signal': 'LONG' if position['signal'] == 1 else 'SHORT',
                                'entry_price': position['entry_price'],
                                'exit_price': current_price,
                                'sl': position['sl'],
                                'target': position['target'],
                                'exit_reason': 'Manual Close',
                                'pnl': pnl,
                                'highest': st.session_state['highest_price'],
                                'lowest': st.session_state['lowest_price'],
                                'range': st.session_state['highest_price'] - st.session_state['lowest_price']
                            }
                            
                            st.session_state['trade_history'].append(trade)
                            st.session_state['trade_history'] = st.session_state['trade_history']
                            log_message(f"Position manually closed | PnL: {pnl:.2f}")
                        
                        reset_position_state()
                        log_message("Trading stopped")
            
            with col3:
                if st.button("🔄 Manual Refresh", use_container_width=True):
                    st.rerun()
            
            # Status indicator
            if st.session_state['trading_active']:
                st.success("🟢 Trading is ACTIVE")
            else:
                st.info("⚪ Trading is STOPPED")
            
            st.markdown("---")
            
            # Live trading loop
            if st.session_state['trading_active']:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                while st.session_state['trading_active']:
                    status_text.text("Fetching latest data...")
                    progress_bar.progress(30)
                    
                    # Fetch data
                    df = fetch_data(symbol, interval, period, mode)
                    
                    if df is not None:
                        st.session_state['current_data'] = df
                        
                        progress_bar.progress(60)
                        status_text.text("Generating signals...")
                        
                        # Generate signals
                        if strategy == "EMA Crossover":
                            df = generate_ema_crossover_signal(df, config)
                        elif strategy == "Simple Buy":
                            df = generate_simple_buy_signal(df)
                        elif strategy == "Simple Sell":
                            df = generate_simple_sell_signal(df)
                        elif strategy == "Price Crosses Threshold":
                            df = generate_price_threshold_signal(df, config)
                        elif strategy == "RSI-ADX-EMA":
                            df = generate_rsi_adx_ema_signal(df, config)
                        elif strategy == "Percentage Change":
                            df = generate_percentage_change_signal(df, config)
                        elif strategy == "AI Price Action Analysis":
                            df = generate_ai_analysis_signal(df)
                        elif strategy == "Custom Strategy Builder":
                            df = generate_custom_strategy_signal(df, st.session_state['custom_conditions'])
                        
                        progress_bar.progress(80)
                        status_text.text("Processing trading logic...")
                        
                        # Process live trading
                        process_live_trading(df, strategy, config, quantity)
                        
                        progress_bar.progress(100)
                        status_text.text("Data updated successfully")
                        
                        st.session_state['current_data'] = df
                    else:
                        st.error("Failed to fetch data")
                    
                    time.sleep(random.uniform(1.0, 1.5))
                    st.rerun()
            
            # Display live metrics even when not actively trading
            if st.session_state['current_data'] is not None:
                df = st.session_state['current_data']
                current_price = df['Close'].iloc[-1]
                position = st.session_state['position']
                
                st.markdown("### 📊 Active Configuration")
                config_col1, config_col2, config_col3 = st.columns(3)
                
                with config_col1:
                    st.info(f"**Asset:** {symbol}")
                    st.info(f"**Interval:** {interval}")
                    st.info(f"**Period:** {period}")
                
                with config_col2:
                    st.info(f"**Strategy:** {strategy}")
                    st.info(f"**Quantity:** {quantity}")
                    st.info(f"**Mode:** {mode}")
                
                with config_col3:
                    st.info(f"**SL Type:** {config['sl_type']}")
                    sl_points_str = f"{config.get('sl_points', 0):.2f}" if config.get('sl_points') else "Signal Based"
                    st.info(f"**SL Points:** {sl_points_str}")
                    st.info(f"**Target Type:** {config['target_type']}")
                
                st.markdown("---")
                st.markdown("### 📈 Live Metrics")
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Current Price", f"{current_price:.2f}")
                
                with metric_col2:
                    if position:
                        st.metric("Entry Price", f"{position['entry_price']:.2f}")
                    else:
                        st.metric("Entry Price", "N/A")
                
                with metric_col3:
                    if position:
                        pos_type = "LONG" if position['signal'] == 1 else "SHORT"
                        st.metric("Position", pos_type)
                    else:
                        st.metric("Position", "None")
                
                with metric_col4:
                    if position:
                        if position['signal'] == 1:
                            unrealized_pnl = (current_price - position['entry_price']) * quantity
                        else:
                            unrealized_pnl = (position['entry_price'] - current_price) * quantity
                        
                        if unrealized_pnl >= 0:
                            st.metric("Unrealized P&L", f"{unrealized_pnl:.2f}", 
                                     delta=f"+{unrealized_pnl:.2f}", delta_color="normal")
                        else:
                            st.metric("Unrealized P&L", f"{unrealized_pnl:.2f}", 
                                     delta=f"{unrealized_pnl:.2f}", delta_color="inverse")
                    else:
                        st.metric("Unrealized P&L", "0.00")
                
                # Indicator values
                st.markdown("### 🔢 Indicator Values")
                ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
                
                with ind_col1:
                    if 'EMA_Fast' in df.columns:
                        st.metric("EMA Fast", f"{df['EMA_Fast'].iloc[-1]:.2f}")
                
                with ind_col2:
                    if 'EMA_Slow' in df.columns:
                        st.metric("EMA Slow", f"{df['EMA_Slow'].iloc[-1]:.2f}")
                
                with ind_col3:
                    if 'EMA_Angle' in df.columns:
                        st.metric("EMA Angle", f"{df['EMA_Angle'].iloc[-1]:.2f}°")
                
                with ind_col4:
                    if 'RSI' in df.columns:
                        st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
                
                # Entry filter status
                if strategy == "EMA Crossover":
                    st.markdown("### ✅ Entry Filter Status")
                    filter_col1, filter_col2 = st.columns(2)
                    
                    with filter_col1:
                        if config['entry_filter'] == "Custom Candle (Points)":
                            candle_size = abs(df['Close'].iloc[-1] - df['Open'].iloc[-1])
                            min_size = config['custom_points']
                            status = "✅" if candle_size >= min_size else "❌"
                            st.write(f"{status} Candle Size: {candle_size:.2f} / Min: {min_size:.2f}")
                        elif config['entry_filter'] == "ATR-based Candle":
                            if 'ATR' in df.columns:
                                candle_size = abs(df['Close'].iloc[-1] - df['Open'].iloc[-1])
                                min_size = df['ATR'].iloc[-1] * config['atr_multiplier']
                                status = "✅" if candle_size >= min_size else "❌"
                                st.write(f"{status} Candle Size: {candle_size:.2f} / Min: {min_size:.2f}")
                    
                    with filter_col2:
                        if config['use_adx'] and 'ADX' in df.columns:
                            adx_val = df['ADX'].iloc[-1]
                            adx_threshold = config['adx_threshold']
                            status = "✅" if adx_val >= adx_threshold else "❌"
                            st.write(f"{status} ADX: {adx_val:.2f} / Threshold: {adx_threshold:.2f}")
                
                # Current signal
                current_signal = df['Signal'].iloc[-1]
                if current_signal == 1:
                    st.success("🟢 Current Signal: BUY")
                elif current_signal == -1:
                    st.error("🔴 Current Signal: SELL")
                else:
                    st.info("⚪ Current Signal: NONE")
                
                # AI Analysis display
                if strategy == "AI Price Action Analysis" and 'AI_Analysis' in df.columns:
                    analysis = df['AI_Analysis'].iloc[-1]
                    if analysis:
                        st.markdown("### 🤖 AI Analysis")
                        ai_col1, ai_col2 = st.columns(2)
                        
                        with ai_col1:
                            st.write(f"**Trend:** {analysis.get('trend', 'N/A')}")
                            st.write(f"**RSI:** {analysis.get('rsi', 'N/A')}")
                            st.write(f"**MACD:** {analysis.get('macd', 'N/A')}")
                        
                        with ai_col2:
                            st.write(f"**Bollinger Bands:** {analysis.get('bb', 'N/A')}")
                            st.write(f"**Volume:** {analysis.get('volume', 'N/A')}")
                            st.write(f"**Score:** {analysis.get('score', 0)}")
                
                # Percentage change display
                if strategy == "Percentage Change" and 'PctChange' in df.columns:
                    pct_change = df['PctChange'].iloc[-1]
                    st.markdown("### 📊 Percentage Change")
                    st.metric("Change from First Candle", f"{pct_change:.3f}%")
                
                # Position information
                if position:
                    st.markdown("### 💼 Position Information")
                    
                    pos_col1, pos_col2, pos_col3 = st.columns(3)
                    
                    with pos_col1:
                        entry_time_str = position['entry_time'].strftime('%Y-%m-%d %H:%M:%S')
                        st.write(f"**Entry Time:** {entry_time_str}")
                        
                        duration = (datetime.now(pytz.timezone('Asia/Kolkata')) - position['entry_time']).total_seconds() / 60
                        st.write(f"**Duration:** {duration:.1f} minutes")
                        
                        st.write(f"**Entry Price:** {position['entry_price']:.2f}")
                    
                    with pos_col2:
                        sl_str = f"{position['sl']:.2f}" if position['sl'] != 0 else "Signal Based"
                        st.write(f"**Stop Loss:** {sl_str}")
                        
                        target_str = f"{position['target']:.2f}" if position['target'] != 0 else "Signal Based"
                        st.write(f"**Target:** {target_str}")
                        
                        if position['sl'] != 0:
                            dist_to_sl = abs(current_price - position['sl'])
                            st.write(f"**Distance to SL:** {dist_to_sl:.2f}")
                    
                    with pos_col3:
                        if position['target'] != 0:
                            dist_to_target = abs(position['target'] - current_price)
                            st.write(f"**Distance to Target:** {dist_to_target:.2f}")
                        
                        if st.session_state['highest_price']:
                            st.write(f"**Highest Price:** {st.session_state['highest_price']:.2f}")
                        
                        if st.session_state['lowest_price']:
                            st.write(f"**Lowest Price:** {st.session_state['lowest_price']:.2f}")
                    
                    # Trailing target info
                    if config['target_type'] in ['Trailing Target (Points)', 'Trailing Target + Signal Based']:
                        profit_points = st.session_state.get('trailing_profit_points', 0)
                        next_update = profit_points + config['target_points']
                        st.info(f"📊 Trailing Target: Profit moved {profit_points:.2f} points | Next update at {next_update:.2f} points")
                    
                    # Partial exit info
                    if st.session_state.get('partial_exit_done', False):
                        st.success("✅ 50% position already exited - Trailing remaining 50%")
                    
                    # Break-even info
                    if st.session_state.get('breakeven_activated', False):
                        st.success("✅ Break-even activated - SL moved to entry price")
                
                # Live chart
                st.markdown("### 📊 Live Chart")
                
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
                    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], 
                                            mode='lines', name='EMA Fast', 
                                            line=dict(color='blue', width=1)))
                
                if 'EMA_Slow' in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], 
                                            mode='lines', name='EMA Slow', 
                                            line=dict(color='red', width=1)))
                
                if position:
                    fig.add_hline(y=position['entry_price'], line_dash="dash", 
                                 line_color="yellow", annotation_text="Entry")
                    
                    if position['sl'] != 0:
                        fig.add_hline(y=position['sl'], line_dash="dash", 
                                     line_color="red", annotation_text="SL")
                    
                    if position['target'] != 0:
                        fig.add_hline(y=position['target'], line_dash="dash", 
                                     line_color="green", annotation_text="Target")
                
                fig.update_layout(
                    title=f"{symbol} - {interval}",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    height=600,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"live_chart_{int(time.time())}")
                
                # Textual guidance
                st.markdown("### 💡 Guidance")
                if position:
                    if position['signal'] == 1:
                        unrealized_pnl = (current_price - position['entry_price']) * quantity
                    else:
                        unrealized_pnl = (position['entry_price'] - current_price) * quantity
                    
                    if unrealized_pnl > 0:
                        st.success(f"✅ In Profit: {unrealized_pnl:.2f} points")
                    else:
                        st.warning(f"⚠️ In Loss: {unrealized_pnl:.2f} points")
                else:
                    st.info("⏳ Waiting for entry signal...")
        
        else:
            st.info("Live Trading Dashboard is only available in Live Trading mode. Switch to Live Trading in the sidebar.")
    
    # Tab 2: Trade History
    with tab2:
        st.markdown("### 📈 Trade History")
        
        if len(st.session_state['trade_history']) == 0:
            st.info("No trades yet. Start trading to see history.")
        else:
            # Calculate statistics
            total_trades = len(st.session_state['trade_history'])
            winning_trades = sum(1 for t in st.session_state['trade_history'] if t['pnl'] > 0)
            losing_trades = sum(1 for t in st.session_state['trade_history'] if t['pnl'] <= 0)
            total_pnl = sum(t['pnl'] for t in st.session_state['trade_history'])
            accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Display metrics
            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
            
            with metric_col1:
                st.metric("Total Trades", total_trades)
            
            with metric_col2:
                st.metric("Winning Trades", winning_trades)
            
            with metric_col3:
                st.metric("Losing Trades", losing_trades)
            
            with metric_col4:
                st.metric("Accuracy", f"{accuracy:.2f}%")
            
            with metric_col5:
                if total_pnl >= 0:
                    st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"+{total_pnl:.2f}")
                else:
                    st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color="inverse")
            
            st.markdown("---")
            
            # Display trades
            for idx, trade in enumerate(reversed(st.session_state['trade_history'])):
                with st.expander(f"Trade #{total_trades - idx} - {trade['signal']} - P&L: {trade['pnl']:.2f}"):
                    trade_col1, trade_col2 = st.columns(2)
                    
                    with trade_col1:
                        entry_time_str = trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')
                        exit_time_str = trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')
                        
                        st.write(f"**Entry Time:** {entry_time_str}")
                        st.write(f"**Exit Time:** {exit_time_str}")
                        st.write(f"**Duration:** {trade['duration']:.2f} hours")
                        st.write(f"**Signal:** {trade['signal']}")
                        st.write(f"**Entry Price:** {trade['entry_price']:.2f}")
                    
                    with trade_col2:
                        st.write(f"**Exit Price:** {trade['exit_price']:.2f}")
                        
                        sl_str = f"{trade['sl']:.2f}" if trade['sl'] != 0 else "Signal Based"
                        st.write(f"**Stop Loss:** {sl_str}")
                        
                        target_str = f"{trade['target']:.2f}" if trade['target'] != 0 else "Signal Based"
                        st.write(f"**Target:** {target_str}")
                        
                        st.write(f"**Exit Reason:** {trade['exit_reason']}")
                        
                        if trade['pnl'] >= 0:
                            st.success(f"**P&L:** +{trade['pnl']:.2f}")
                        else:
                            st.error(f"**P&L:** {trade['pnl']:.2f}")
                    
                    st
