import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import time
import random

# ==================== UTILITIES ====================

def get_ist_time():
    """Get current time in IST"""
    return datetime.now(pytz.timezone('Asia/Kolkata'))

def add_log(message):
    """Add timestamped log entry"""
    timestamp = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state['trade_logs'].append(log_entry)
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

def validate_timeframe_period(interval, period):
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
            delay = random.uniform(1.0, 1.5)
            time.sleep(delay)
        
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if data.empty:
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                if col == 'Volume':
                    data[col] = 0
                else:
                    return None
        
        data = data[required_cols].copy()
        
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            data.index = data.index.tz_convert('Asia/Kolkata')
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# ==================== INDICATOR CALCULATIONS ====================

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_sma(series, period):
    """Calculate Simple Moving Average"""
    return series.rolling(window=period).mean()

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
    """Calculate Average Directional Index"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = calculate_atr(df, 1)
    atr = tr.rolling(window=period).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx

def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_bollinger_bands(series, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return upper, middle, lower

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_keltner_channel(df, period=20, multiplier=2):
    """Calculate Keltner Channel"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    middle = calculate_ema(typical_price, period)
    atr = calculate_atr(df, period)
    
    upper = middle + (multiplier * atr)
    lower = middle - (multiplier * atr)
    
    return upper, middle, lower

def calculate_vwap(df):
    """Calculate Volume Weighted Average Price"""
    if df['Volume'].sum() == 0:
        return pd.Series(df['Close'], index=df.index)
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    return vwap

def calculate_supertrend(df, period=10, multiplier=3):
    """Calculate SuperTrend indicator"""
    atr = calculate_atr(df, period)
    hl_avg = (df['High'] + df['Low']) / 2
    
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    for i in range(period, len(df)):
        if i == period:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
        else:
            if df['Close'].iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif df['Close'].iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
    
    return supertrend, direction

def detect_swing_highs_lows(df, lookback=5):
    """Detect swing highs and lows"""
    swing_high = pd.Series(index=df.index, dtype=float)
    swing_low = pd.Series(index=df.index, dtype=float)
    
    for i in range(lookback, len(df) - lookback):
        high_window = df['High'].iloc[i-lookback:i+lookback+1]
        low_window = df['Low'].iloc[i-lookback:i+lookback+1]
        
        if df['High'].iloc[i] == high_window.max():
            swing_high.iloc[i] = df['High'].iloc[i]
        
        if df['Low'].iloc[i] == low_window.min():
            swing_low.iloc[i] = df['Low'].iloc[i]
    
    swing_high = swing_high.fillna(method='ffill')
    swing_low = swing_low.fillna(method='ffill')
    
    return swing_high, swing_low

def calculate_support_resistance(df, lookback=20):
    """Calculate support and resistance levels"""
    support = df['Low'].rolling(window=lookback).min()
    resistance = df['High'].rolling(window=lookback).max()
    
    return support, resistance

def calculate_ema_angle(ema_series, lookback=5):
    """Calculate EMA angle in degrees"""
    if len(ema_series) < lookback + 1:
        return pd.Series(0, index=ema_series.index)
    
    angles = pd.Series(index=ema_series.index, dtype=float)
    
    for i in range(lookback, len(ema_series)):
        y_diff = ema_series.iloc[i] - ema_series.iloc[i - lookback]
        x_diff = lookback
        
        angle_rad = np.arctan2(y_diff, x_diff)
        angle_deg = np.degrees(angle_rad)
        
        angles.iloc[i] = abs(angle_deg)
    
    return angles.fillna(0)

def add_all_indicators(df, config):
    """Add all required indicators to dataframe"""
    df['EMA_Fast'] = calculate_ema(df['Close'], config.get('ema_fast', 9))
    df['EMA_Slow'] = calculate_ema(df['Close'], config.get('ema_slow', 15))
    df['EMA_20'] = calculate_ema(df['Close'], 20)
    df['EMA_50'] = calculate_ema(df['Close'], 50)
    df['EMA_Angle'] = calculate_ema_angle(df['EMA_Fast'], lookback=5)
    df['ATR'] = calculate_atr(df, config.get('atr_period', 14))
    df['ADX'] = calculate_adx(df, config.get('adx_period', 14))
    df['RSI'] = calculate_rsi(df['Close'], config.get('rsi_period', 14))
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'], 20, 2)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    df['KC_Upper'], df['KC_Middle'], df['KC_Lower'] = calculate_keltner_channel(df)
    df['VWAP'] = calculate_vwap(df)
    df['SuperTrend'], df['ST_Direction'] = calculate_supertrend(df)
    df['Swing_High'], df['Swing_Low'] = detect_swing_highs_lows(df, lookback=5)
    df['Support'], df['Resistance'] = calculate_support_resistance(df, lookback=20)
    
    return df

# ==================== STRATEGY LOGIC ====================

def check_ema_crossover_entry(df, i, config):
    """Check EMA crossover entry with filters"""
    if i < 1:
        return False, None
    
    ema_fast = df['EMA_Fast'].iloc[i]
    ema_slow = df['EMA_Slow'].iloc[i]
    prev_ema_fast = df['EMA_Fast'].iloc[i-1]
    prev_ema_slow = df['EMA_Slow'].iloc[i-1]
    
    bullish_cross = ema_fast > ema_slow and prev_ema_fast <= prev_ema_slow
    bearish_cross = ema_fast < ema_slow and prev_ema_fast >= prev_ema_slow
    
    if not (bullish_cross or bearish_cross):
        return False, None
    
    angle = df['EMA_Angle'].iloc[i]
    min_angle = config.get('min_angle', 1)
    if angle < min_angle:
        return False, None
    
    entry_filter = config.get('entry_filter', 'Simple Crossover')
    candle_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
    
    if entry_filter == 'Custom Candle (Points)':
        custom_points = config.get('custom_points', 10)
        if candle_size < custom_points:
            return False, None
    
    elif entry_filter == 'ATR-based Candle':
        atr = df['ATR'].iloc[i]
        atr_multiplier = config.get('atr_multiplier', 1.5)
        min_candle = atr * atr_multiplier
        if candle_size < min_candle:
            return False, None
    
    if config.get('use_adx', False):
        adx = df['ADX'].iloc[i]
        adx_threshold = config.get('adx_threshold', 25)
        if adx < adx_threshold:
            return False, None
    
    signal = 1 if bullish_cross else -1
    return True, signal

def check_rsi_adx_ema_entry(df, i):
    """Check RSI-ADX-EMA strategy entry"""
    if i < 1:
        return False, None
    
    rsi = df['RSI'].iloc[i]
    adx = df['ADX'].iloc[i]
    ema_fast = df['EMA_Fast'].iloc[i]
    ema_slow = df['EMA_Slow'].iloc[i]
    
    if rsi > 80 and adx < 20 and ema_fast < ema_slow:
        return True, -1
    
    if rsi < 20 and adx > 20 and ema_fast > ema_slow:
        return True, 1
    
    return False, None

def check_threshold_entry(df, i, config):
    """Check price crosses threshold strategy"""
    if i < 1:
        return False, None
    
    threshold = config.get('threshold_value', 0)
    threshold_type = config.get('threshold_type', 'LONG (Price >= Threshold)')
    current_price = df['Close'].iloc[i]
    
    if threshold_type == 'LONG (Price >= Threshold)':
        if current_price >= threshold:
            return True, 1
    elif threshold_type == 'SHORT (Price >= Threshold)':
        if current_price >= threshold:
            return True, -1
    elif threshold_type == 'LONG (Price <= Threshold)':
        if current_price <= threshold:
            return True, 1
    elif threshold_type == 'SHORT (Price <= Threshold)':
        if current_price <= threshold:
            return True, -1
    
    return False, None

def analyze_ai_signal(df, i):
    """AI Price Action Analysis"""
    if i < 1:
        return None, "", {}
    
    score = 0
    reasons = []
    breakdown = {}
    
    ema_fast = df['EMA_Fast'].iloc[i]
    ema_slow = df['EMA_Slow'].iloc[i]
    if ema_fast > ema_slow:
        score += 2
        reasons.append("Uptrend (EMA)")
        breakdown['Trend'] = "Bullish"
    else:
        score -= 2
        reasons.append("Downtrend (EMA)")
        breakdown['Trend'] = "Bearish"
    
    rsi = df['RSI'].iloc[i]
    if rsi < 30:
        score += 2
        reasons.append("Oversold (RSI)")
        breakdown['RSI'] = f"Oversold ({rsi:.1f})"
    elif rsi > 70:
        score -= 2
        reasons.append("Overbought (RSI)")
        breakdown['RSI'] = f"Overbought ({rsi:.1f})"
    else:
        breakdown['RSI'] = f"Neutral ({rsi:.1f})"
    
    macd = df['MACD'].iloc[i]
    macd_signal = df['MACD_Signal'].iloc[i]
    if macd > macd_signal:
        score += 1
        reasons.append("Bullish MACD")
        breakdown['MACD'] = "Bullish crossover"
    else:
        score -= 1
        reasons.append("Bearish MACD")
        breakdown['MACD'] = "Bearish crossover"
    
    close = df['Close'].iloc[i]
    bb_upper = df['BB_Upper'].iloc[i]
    bb_lower = df['BB_Lower'].iloc[i]
    if close < bb_lower:
        score += 1
        reasons.append("Near lower BB")
        breakdown['Bollinger'] = "Near lower band (oversold)"
    elif close > bb_upper:
        score -= 1
        reasons.append("Near upper BB")
        breakdown['Bollinger'] = "Near upper band (overbought)"
    else:
        breakdown['Bollinger'] = "Within bands"
    
    if df['Volume'].iloc[i] > 0:
        avg_volume = df['Volume'].iloc[max(0, i-20):i].mean()
        if df['Volume'].iloc[i] > avg_volume * 1.5:
            score += 1 if score > 0 else -1
            reasons.append("High volume")
            breakdown['Volume'] = "High (confirming)"
        else:
            breakdown['Volume'] = "Normal"
    else:
        breakdown['Volume'] = "N/A (Index)"
    
    if score >= 3:
        signal = 1
        confidence = min(100, (score / 6) * 100)
    elif score <= -3:
        signal = -1
        confidence = min(100, (abs(score) / 6) * 100)
    else:
        signal = 0
        confidence = 0
    
    reasoning = f"Score: {score} | " + " | ".join(reasons)
    
    return signal, reasoning, breakdown

def check_custom_conditions(df, i, conditions):
    """Check custom strategy conditions"""
    if i < 1:
        return False, None
    
    buy_signal = False
    sell_signal = False
    buy_conditions_met = []
    sell_conditions_met = []
    
    for cond in conditions:
        if not cond.get('active', False):
            continue
        
        use_price = cond.get('use_price', False)
        price_val = df['Close'].iloc[i] if use_price else None
        
        indicator = cond.get('indicator', 'RSI')
        operator = cond.get('operator', '>')
        value = cond.get('value', 0)
        action = cond.get('action', 'BUY')
        
        if indicator == 'Price':
            ind_val = df['Close'].iloc[i]
        elif indicator == 'Close':
            ind_val = df['Close'].iloc[i]
        elif indicator == 'High':
            ind_val = df['High'].iloc[i]
        elif indicator == 'Low':
            ind_val = df['Low'].iloc[i]
        elif indicator == 'Volume':
            ind_val = df['Volume'].iloc[i] if df['Volume'].iloc[i] > 0 else 0
        elif indicator in df.columns:
            ind_val = df[indicator].iloc[i]
        else:
            continue
        
        condition_met = False
        if operator == '>':
            if use_price and price_val is not None:
                condition_met = price_val > ind_val
            else:
                condition_met = ind_val > value
        elif operator == '<':
            if use_price and price_val is not None:
                condition_met = price_val < ind_val
            else:
                condition_met = ind_val < value
        elif operator == '>=':
            if use_price and price_val is not None:
                condition_met = price_val >= ind_val
            else:
                condition_met = ind_val >= value
        elif operator == '<=':
            if use_price and price_val is not None:
                condition_met = price_val <= ind_val
            else:
                condition_met = ind_val <= value
        elif operator == '==':
            if use_price and price_val is not None:
                condition_met = abs(price_val - ind_val) < 0.01
            else:
                condition_met = abs(ind_val - value) < 0.01
        elif operator == 'crosses_above':
            prev_ind = df[indicator].iloc[i-1] if indicator in df.columns else 0
            if use_price and price_val is not None:
                condition_met = ind_val > price_val and prev_ind <= price_val
            else:
                condition_met = ind_val > value and prev_ind <= value
        elif operator == 'crosses_below':
            prev_ind = df[indicator].iloc[i-1] if indicator in df.columns else 0
            if use_price and price_val is not None:
                condition_met = ind_val < price_val and prev_ind >= price_val
            else:
                condition_met = ind_val < value and prev_ind >= value
        
        if condition_met:
            if action == 'BUY':
                buy_conditions_met.append(cond)
            else:
                sell_conditions_met.append(cond)
    
    active_buy = [c for c in conditions if c.get('active', False) and c.get('action') == 'BUY']
    if len(active_buy) > 0 and len(buy_conditions_met) == len(active_buy):
        buy_signal = True
    
    active_sell = [c for c in conditions if c.get('active', False) and c.get('action') == 'SELL']
    if len(active_sell) > 0 and len(sell_conditions_met) == len(active_sell):
        sell_signal = True
    
    if buy_signal:
        return True, 1
    elif sell_signal:
        return True, -1
    
    return False, None

def calculate_sl_target(df, i, signal, config):
    """Calculate stop loss and target based on configuration"""
    entry_price = df['Close'].iloc[i]
    sl_type = config.get('sl_type', 'Custom Points')
    target_type = config.get('target_type', 'Custom Points')
    
    sl = None
    if sl_type == 'Custom Points':
        sl_points = config.get('sl_points', 10)
        sl = entry_price - sl_points if signal == 1 else entry_price + sl_points
    elif 'Trailing SL' in sl_type:
        sl_points = config.get('sl_points', 10)
        sl = entry_price - sl_points if signal == 1 else entry_price + sl_points
    elif sl_type == 'ATR-based':
        atr = df['ATR'].iloc[i]
        if pd.notna(atr) and atr > 0:
            atr_multiplier = config.get('atr_multiplier', 1.5)
            sl = entry_price - (atr * atr_multiplier) if signal == 1 else entry_price + (atr * atr_multiplier)
        else:
            sl_points = config.get('sl_points', 10)
            sl = entry_price - sl_points if signal == 1 else entry_price + sl_points
    elif sl_type == 'Current Candle Low/High':
        sl = df['Low'].iloc[i] if signal == 1 else df['High'].iloc[i]
    elif sl_type == 'Previous Candle Low/High':
        if i > 0:
            sl = df['Low'].iloc[i-1] if signal == 1 else df['High'].iloc[i-1]
        else:
            sl_points = config.get('sl_points', 10)
            sl = entry_price - sl_points if signal == 1 else entry_price + sl_points
    elif sl_type == 'Current Swing Low/High':
        swing_val = df['Swing_Low'].iloc[i] if signal == 1 else df['Swing_High'].iloc[i]
        if pd.notna(swing_val):
            sl = swing_val
        else:
            sl_points = config.get('sl_points', 10)
            sl = entry_price - sl_points if signal == 1 else entry_price + sl_points
    elif sl_type == 'Previous Swing Low/High':
        prev_swing = df['Swing_Low'].iloc[:i].dropna() if signal == 1 else df['Swing_High'].iloc[:i].dropna()
        if len(prev_swing) > 0:
            sl = prev_swing.iloc[-1]
        else:
            sl_points = config.get('sl_points', 10)
            sl = entry_price - sl_points if signal == 1 else entry_price + sl_points
    elif sl_type == 'Signal-based (reverse EMA crossover)':
        sl = 0
    else:
        sl_points = config.get('sl_points', 10)
        sl = entry_price - sl_points if signal == 1 else entry_price + sl_points
    
    if sl is None:
        sl_points = config.get('sl_points', 10)
        sl = entry_price - sl_points if signal == 1 else entry_price + sl_points
    
    if sl is not None and sl != 0:
        min_sl_distance = config.get('min_sl_distance', 10)
        if signal == 1:
            sl = min(sl, entry_price - min_sl_distance)
        else:
            sl = max(sl, entry_price + min_sl_distance)
    
    target = None
    if target_type == 'Custom Points':
        target_points = config.get('target_points', 20)
        target = entry_price + target_points if signal == 1 else entry_price - target_points
    elif 'Trailing Target' in target_type:
        ticker = config.get('ticker', 'BTC-USD')
        target_points = config.get('target_points', 20)
        
        if 'NIFTY' in ticker.upper():
            target_points = max(target_points, 10)
        elif 'BANK' in ticker.upper():
            target_points = max(target_points, 20)
        elif 'BTC' in ticker.upper():
            target_points = max(target_points, 150)
        elif 'ETH' in ticker.upper():
            target_points = max(target_points, 10)
        else:
            target_points = max(target_points, 15)
        
        target = entry_price + target_points if signal == 1 else entry_price - target_points
    elif target_type == 'ATR-based':
        atr = df['ATR'].iloc[i]
        if pd.notna(atr) and atr > 0:
            atr_multiplier = config.get('target_atr_multiplier', 2.0)
            target = entry_price + (atr * atr_multiplier) if signal == 1 else entry_price - (atr * atr_multiplier)
        else:
            target_points = config.get('target_points', 20)
            target = entry_price + target_points if signal == 1 else entry_price - target_points
    elif target_type == 'Risk-Reward Based':
        if sl is not None and sl != 0:
            risk = abs(entry_price - sl)
            rr_ratio = config.get('rr_ratio', 2.0)
            target = entry_price + (risk * rr_ratio) if signal == 1 else entry_price - (risk * rr_ratio)
        else:
            target_points = config.get('target_points', 20)
            target = entry_price + target_points if signal == 1 else entry_price - target_points
    elif target_type == 'Current Candle Low/High':
        target = df['High'].iloc[i] if signal == 1 else df['Low'].iloc[i]
    elif target_type == 'Previous Candle Low/High':
        if i > 0:
            target = df['High'].iloc[i-1] if signal == 1 else df['Low'].iloc[i-1]
        else:
            target_points = config.get('target_points', 20)
            target = entry_price + target_points if signal == 1 else entry_price - target_points
    elif target_type == 'Current Swing Low/High':
        swing_val = df['Swing_High'].iloc[i] if signal == 1 else df['Swing_Low'].iloc[i]
        if pd.notna(swing_val):
            target = swing_val
        else:
            target_points = config.get('target_points', 20)
            target = entry_price + target_points if signal == 1 else entry_price - target_points
    elif target_type == 'Previous Swing Low/High':
        prev_swing = df['Swing_High'].iloc[:i].dropna() if signal == 1 else df['Swing_Low'].iloc[:i].dropna()
        if len(prev_swing) > 0:
            target = prev_swing.iloc[-1]
        else:
            target_points = config.get('target_points', 20)
            target = entry_price + target_points if signal == 1 else entry_price - target_points
    elif target_type == 'Signal-based (reverse EMA crossover)':
        target = 0
    else:
        target_points = config.get('target_points', 20)
        target = entry_price + target_points if signal == 1 else entry_price - target_points
    
    if target is None:
        target_points = config.get('target_points', 20)
        target = entry_price + target_points if signal == 1 else entry_price - target_points
    
    if target is not None and target != 0:
        min_target_distance = config.get('min_target_distance', 15)
        if signal == 1:
            target = max(target, entry_price + min_target_distance)
        else:
            target = min(target, entry_price - min_target_distance)
    
    return sl if sl is not None else 0, target if target is not None else 0

def update_trailing_sl(current_price, entry_price, signal, config, position):
    """Update trailing stop loss"""
    sl_type = config.get('sl_type', 'Custom Points')
    
    if 'Trailing SL' not in sl_type:
        return position.get('sl')
    
    sl_points = config.get('sl_points', 10)
    threshold = config.get('trailing_threshold', 0)
    current_sl = position.get('sl')
    
    if threshold > 0:
        if signal == 1:
            profit = current_price - entry_price
            if profit < threshold:
                return current_sl
        else:
            profit = entry_price - current_price
            if profit < threshold:
                return current_sl
    
    if sl_type == 'Trailing SL (Points)':
        if signal == 1:
            new_sl = current_price - sl_points
            if current_sl is None or new_sl > current_sl:
                return new_sl
        else:
            new_sl = current_price + sl_points
            if current_sl is None or new_sl < current_sl:
                return new_sl
    
    elif sl_type == 'Trailing SL + Current Candle':
        return current_sl
    
    elif sl_type == 'Trailing SL + Signal Based':
        if signal == 1:
            new_sl = current_price - sl_points
            if current_sl is None or new_sl > current_sl:
                return new_sl
        else:
            new_sl = current_price + sl_points
            if current_sl is None or new_sl < current_sl:
                return new_sl
    
    elif sl_type == 'Volatility-Adjusted Trailing SL':
        return current_sl
    
    elif sl_type == 'Break-even After 50% Target':
        target = position.get('target', 0)
        if target > 0:
            if signal == 1:
                halfway = entry_price + (target - entry_price) * 0.5
                if current_price >= halfway and not position.get('breakeven_activated', False):
                    return entry_price
            else:
                halfway = entry_price - (entry_price - target) * 0.5
                if current_price <= halfway and not position.get('breakeven_activated', False):
                    return entry_price
    
    return current_sl

def check_signal_based_exit(df, i, signal):
    """Check for reverse EMA crossover exit"""
    if i < 1:
        return False
    
    ema_fast = df['EMA_Fast'].iloc[i]
    ema_slow = df['EMA_Slow'].iloc[i]
    prev_ema_fast = df['EMA_Fast'].iloc[i-1]
    prev_ema_slow = df['EMA_Slow'].iloc[i-1]
    
    if signal == 1:
        if ema_fast < ema_slow and prev_ema_fast >= prev_ema_slow:
            return True
    else:
        if ema_fast > ema_slow and prev_ema_fast <= prev_ema_slow:
            return True
    
    return False

# ==================== BACKTEST ENGINE ====================

def run_backtest(df, strategy, config):
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
    quantity = config.get('quantity', 1)
    
    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        current_time = df.index[i]
        
        if position is not None:
            if position['signal'] == 1:
                if position.get('highest_price') is None or current_price > position.get('highest_price', current_price):
                    position['highest_price'] = current_price
            else:
                if position.get('lowest_price') is None or current_price < position.get('lowest_price', current_price):
                    position['lowest_price'] = current_price
            
            sl_type = config.get('sl_type', 'Custom Points')
            target_type = config.get('target_type', 'Custom Points')
            
            if 'Signal-based' in sl_type or 'Signal-based' in target_type:
                if check_signal_based_exit(df, i, position['signal']):
                    exit_price = current_price
                    exit_time = current_time
                    pnl = (exit_price - position['entry_price']) * position['signal'] * quantity
                    
                    duration = (exit_time - position['entry_time']).total_seconds() / 3600
                    
                    highest_val = position.get('highest_price', 0)
                    lowest_val = position.get('lowest_price', 0)
                    range_val = abs(highest_val - lowest_val) if highest_val and lowest_val else 0
                    
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': exit_time,
                        'duration': duration,
                        'signal': 'LONG' if position['signal'] == 1 else 'SHORT',
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'sl': position.get('sl', 0),
                        'target': position.get('target', 0),
                        'exit_reason': 'Reverse Signal',
                        'pnl': pnl,
                        'highest_price': highest_val,
                        'lowest_price': lowest_val,
                        'range': range_val
                    }
                    
                    results['trades'].append(trade)
                    results['total_pnl'] += pnl
                    if pnl > 0:
                        results['winning_trades'] += 1
                    else:
                        results['losing_trades'] += 1
                    
                    position = None
                    continue
            
            if position.get('sl', 0) != 0:
                sl_hit = False
                if position['signal'] == 1 and current_price <= position['sl']:
                    sl_hit = True
                elif position['signal'] == -1 and current_price >= position['sl']:
                    sl_hit = True
                
                if sl_hit:
                    exit_price = position['sl']
                    exit_time = current_time
                    pnl = (exit_price - position['entry_price']) * position['signal'] * quantity
                    
                    duration = (exit_time - position['entry_time']).total_seconds() / 3600
                    
                    highest_val = position.get('highest_price', 0)
                    lowest_val = position.get('lowest_price', 0)
                    range_val = abs(highest_val - lowest_val) if highest_val and lowest_val else 0
                    
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': exit_time,
                        'duration': duration,
                        'signal': 'LONG' if position['signal'] == 1 else 'SHORT',
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'sl': position.get('sl', 0),
                        'target': position.get('target', 0),
                        'exit_reason': 'Stop Loss Hit',
                        'pnl': pnl,
                        'highest_price': highest_val,
                        'lowest_price': lowest_val,
                        'range': range_val
                    }
                    
                    results['trades'].append(trade)
                    results['total_pnl'] += pnl
                    results['losing_trades'] += 1
                    
                    position = None
                    continue
            
            if position.get('target', 0) != 0 and 'Trailing Target' not in target_type and 'Signal-based' not in target_type:
                target_hit = False
                if position['signal'] == 1 and current_price >= position['target']:
                    target_hit = True
                elif position['signal'] == -1 and current_price <= position['target']:
                    target_hit = True
                
                if target_hit:
                    if '50% Exit' in target_type and not position.get('partial_exit_done', False):
                        position['partial_exit_done'] = True
                    else:
                        exit_price = position['target']
                        exit_time = current_time
                        pnl = (exit_price - position['entry_price']) * position['signal'] * quantity
                        
                        duration = (exit_time - position['entry_time']).total_seconds() / 3600
                        
                        highest_val = position.get('highest_price', 0)
                        lowest_val = position.get('lowest_price', 0)
                        range_val = abs(highest_val - lowest_val) if highest_val and lowest_val else 0
                        
                        trade = {
                            'entry_time': position['entry_time'],
                            'exit_time': exit_time,
                            'duration': duration,
                            'signal': 'LONG' if position['signal'] == 1 else 'SHORT',
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'sl': position.get('sl', 0),
                            'target': position.get('target', 0),
                            'exit_reason': 'Target Hit',
                            'pnl': pnl,
                            'highest_price': highest_val,
                            'lowest_price': lowest_val,
                            'range': range_val
                        }
                        
                        results['trades'].append(trade)
                        results['total_pnl'] += pnl
                        results['winning_trades'] += 1
                        
                        position = None
                        continue
            
            if 'Trailing SL' in sl_type:
                new_sl = update_trailing_sl(current_price, position['entry_price'], position['signal'], config, position)
                if new_sl is not None:
                    position['sl'] = new_sl
            
            if 'Break-even After 50% Target' in sl_type:
                if position.get('target', 0) != 0 and not position.get('breakeven_activated', False):
                    if position['signal'] == 1:
                        halfway = position['entry_price'] + (position['target'] - position['entry_price']) * 0.5
                        if current_price >= halfway:
                            position['sl'] = position['entry_price']
                            position['breakeven_activated'] = True
                    else:
                        halfway = position['entry_price'] - (position['entry_price'] - position['target']) * 0.5
                        if current_price <= halfway:
                            position['sl'] = position['entry_price']
                            position['breakeven_activated'] = True
            
            continue
        
        entry_signal = False
        signal = None
        
        if strategy == 'EMA Crossover':
            entry_signal, signal = check_ema_crossover_entry(df, i, config)
        elif strategy == 'Simple Buy':
            entry_signal, signal = True, 1
        elif strategy == 'Simple Sell':
            entry_signal, signal = True, -1
        elif strategy == 'Price Crosses Threshold':
            entry_signal, signal = check_threshold_entry(df, i, config)
        elif strategy == 'RSI-ADX-EMA':
            entry_signal, signal = check_rsi_adx_ema_entry(df, i)
        elif strategy == 'AI Price Action Analysis':
            signal, reasoning, breakdown = analyze_ai_signal(df, i)
            entry_signal = signal != 0
            if entry_signal:
                atr = df['ATR'].iloc[i]
                if pd.notna(atr) and atr > 0:
                    config['sl_points'] = atr * 1.5
                    config['target_points'] = atr * 3
        elif strategy == 'Custom Strategy Builder':
            entry_signal, signal = check_custom_conditions(df, i, config.get('custom_conditions', []))
        
        if entry_signal and signal is not None:
            entry_price = current_price
            sl, target = calculate_sl_target(df, i, signal, config)
            
            position = {
                'entry_time': current_time,
                'entry_price': entry_price,
                'signal': signal,
                'sl': sl if sl is not None else 0,
                'target': target if target is not None else 0,
                'highest_price': entry_price if signal == 1 else None,
                'lowest_price': entry_price if signal == -1 else None,
                'partial_exit_done': False,
                'breakeven_activated': False
            }
    
    results['total_trades'] = len(results['trades'])
    if results['total_trades'] > 0:
        results['accuracy'] = (results['winning_trades'] / results['total_trades']) * 100
        total_duration = sum([t['duration'] for t in results['trades']])
        results['avg_duration'] = total_duration / results['total_trades']
    
    return results

# ==================== LIVE TRADING LOOP ====================

def live_trading_loop(config):
    """Main live trading loop with auto-refresh"""
    ticker = config['ticker']
    interval = config['interval']
    period = config['period']
    strategy = config['strategy']
    quantity = config.get('quantity', 1)
    
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    config_placeholder = st.empty()
    chart_placeholder = st.empty()
    position_placeholder = st.empty()
    
    iteration = 0
    
    while st.session_state.get('trading_active', False):
        iteration += 1
        
        with status_placeholder:
            st.info(f"🔄 Fetching data... (Iteration {iteration})")
        
        df = fetch_data(ticker, interval, period, mode='live')
        
        if df is None or len(df) < 50:
            with status_placeholder:
                st.error("Failed to fetch data or insufficient data")
            time.sleep(2)
            continue
        
        df = add_all_indicators(df, config)
        st.session_state['current_data'] = df
        
        current_price = df['Close'].iloc[-1]
        current_time = get_ist_time()
        
        position = st.session_state.get('position')
        
        with config_placeholder:
            st.markdown("### 📋 Active Configuration")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Asset", ticker)
                st.metric("Interval", interval)
            with col2:
                st.metric("Period", period)
                st.metric("Quantity", quantity)
            with col3:
                st.metric("Strategy", strategy)
                sl_type_display = config.get('sl_type', 'Custom Points')
                st.metric("SL Type", sl_type_display[:20])
            with col4:
                target_type_display = config.get('target_type', 'Custom Points')
                st.metric("Target Type", target_type_display[:20])
                st.metric("Mode", "LIVE TRADING")
        
        if position is not None:
            if position['signal'] == 1:
                highest_current = st.session_state.get('highest_price')
                if highest_current is None or current_price > highest_current:
                    st.session_state['highest_price'] = current_price
            else:
                lowest_current = st.session_state.get('lowest_price')
                if lowest_current is None or current_price < lowest_current:
                    st.session_state['lowest_price'] = current_price
            
            sl_type = config.get('sl_type', 'Custom Points')
            target_type = config.get('target_type', 'Custom Points')
            
            should_exit = False
            exit_reason = ""
            
            if 'Signal-based' in sl_type or 'Signal-based' in target_type:
                if check_signal_based_exit(df, -1, position['signal']):
                    should_exit = True
                    exit_reason = "Reverse Signal"
            
            if not should_exit and position.get('sl', 0) != 0:
                if position['signal'] == 1 and current_price <= position['sl']:
                    should_exit = True
                    exit_reason = "Stop Loss Hit"
                elif position['signal'] == -1 and current_price >= position['sl']:
                    should_exit = True
                    exit_reason = "Stop Loss Hit"
            
            if not should_exit and position.get('target', 0) != 0:
                if 'Trailing Target' not in target_type and 'Signal-based' not in target_type:
                    if position['signal'] == 1 and current_price >= position['target']:
                        if '50% Exit' in target_type and not st.session_state.get('partial_exit_done', False):
                            st.session_state['partial_exit_done'] = True
                            add_log("50% position exited - trailing remaining 50%")
                        else:
                            should_exit = True
                            exit_reason = "Target Hit"
                    elif position['signal'] == -1 and current_price <= position['target']:
                        if '50% Exit' in target_type and not st.session_state.get('partial_exit_done', False):
                            st.session_state['partial_exit_done'] = True
                            add_log("50% position exited - trailing remaining 50%")
                        else:
                            should_exit = True
                            exit_reason = "Target Hit"
            
            if should_exit:
                exit_price = current_price
                pnl = (exit_price - position['entry_price']) * position['signal'] * quantity
                duration = (current_time - position['entry_time']).total_seconds() / 3600
                
                highest_val = st.session_state.get('highest_price', 0)
                lowest_val = st.session_state.get('lowest_price', 0)
                range_val = abs(highest_val - lowest_val) if highest_val and lowest_val else 0
                
                trade = {
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'duration': duration,
                    'signal': 'LONG' if position['signal'] == 1 else 'SHORT',
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'sl': position.get('sl', 0),
                    'target': position.get('target', 0),
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'highest_price': highest_val,
                    'lowest_price': lowest_val,
                    'range': range_val
                }
                
                st.session_state['trade_history'].append(trade)
                st.session_state['trade_history'] = st.session_state['trade_history']
                
                add_log(f"EXIT: {exit_reason} | Price: {exit_price:.2f} | P&L: {pnl:.2f}")
                
                # Dhan API placeholder
                # if pnl > 0:
                #     dhan.place_order(security_id=ticker, transaction_type='SELL' if position['signal'] == 1 else 'BUY', 
                #                      quantity=quantity, order_type='MARKET', product_type='INTRADAY')
                
                reset_position_state()
                position = None
            else:
                if 'Trailing SL' in sl_type:
                    new_sl = update_trailing_sl(current_price, position['entry_price'], position['signal'], config, position)
                    if new_sl is not None and new_sl != position.get('sl'):
                        position['sl'] = new_sl
                        st.session_state['position'] = position
                        add_log(f"SL updated to {new_sl:.2f}")
                
                if 'Break-even After 50% Target' in sl_type and not st.session_state.get('breakeven_activated', False):
                    if position.get('target', 0) != 0:
                        if position['signal'] == 1:
                            halfway = position['entry_price'] + (position['target'] - position['entry_price']) * 0.5
                            if current_price >= halfway:
                                position['sl'] = position['entry_price']
                                st.session_state['position'] = position
                                st.session_state['breakeven_activated'] = True
                                add_log("SL moved to break-even")
                        else:
                            halfway = position['entry_price'] - (position['entry_price'] - position['target']) * 0.5
                            if current_price <= halfway:
                                position['sl'] = position['entry_price']
                                st.session_state['position'] = position
                                st.session_state['breakeven_activated'] = True
                                add_log("SL moved to break-even")
        
        if position is None:
            entry_signal = False
            signal = None
            reasoning = ""
            
            if strategy == 'EMA Crossover':
                entry_signal, signal = check_ema_crossover_entry(df, -1, config)
            elif strategy == 'Simple Buy':
                entry_signal, signal = True, 1
            elif strategy == 'Simple Sell':
                entry_signal, signal = True, -1
            elif strategy == 'Price Crosses Threshold':
                entry_signal, signal = check_threshold_entry(df, -1, config)
            elif strategy == 'RSI-ADX-EMA':
                entry_signal, signal = check_rsi_adx_ema_entry(df, -1)
            elif strategy == 'AI Price Action Analysis':
                signal, reasoning, breakdown = analyze_ai_signal(df, -1)
                entry_signal = signal != 0
                if entry_signal:
                    atr = df['ATR'].iloc[-1]
                    config['sl_points'] = atr * 1.5
                    config['target_points'] = atr * 3
            elif strategy == 'Custom Strategy Builder':
                entry_signal, signal = check_custom_conditions(df, -1, config.get('custom_conditions', []))
            
            if entry_signal and signal is not None:
                entry_price = current_price
                sl, target = calculate_sl_target(df, -1, signal, config)
                
                position = {
                    'entry_time': current_time,
                    'entry_price': entry_price,
                    'signal': signal,
                    'sl': sl if sl is not None else 0,
                    'target': target if target is not None else 0
                }
                
                st.session_state['position'] = position
                st.session_state['highest_price'] = entry_price if signal == 1 else None
                st.session_state['lowest_price'] = entry_price if signal == -1 else None
                
                sl_str = f"{sl:.2f}" if sl != 0 else "Signal"
                target_str = f"{target:.2f}" if target != 0 else "Signal"
                add_log(f"ENTRY: {'LONG' if signal == 1 else 'SHORT'} @ {entry_price:.2f} | SL: {sl_str} | Target: {target_str}")
                
                # Dhan API placeholder
                # dhan.place_order(security_id=ticker, transaction_type='BUY' if signal == 1 else 'SELL',
                #                  quantity=quantity, order_type='MARKET', product_type='INTRADAY')
        
        # Display live metrics - ALWAYS SHOW
        with metrics_placeholder:
            st.markdown("### 📊 Live Metrics")
            
            # Row 1 - Critical Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"{current_price:.2f}")
                if position:
                    st.metric("Entry Price", f"{position['entry_price']:.2f}")
                else:
                    st.metric("Entry Price", "-")
            
            with col2:
                pos_status = "🟢 IN POSITION" if position else "⚪ NO POSITION"
                st.markdown(f"**Position Status**")
                st.markdown(pos_status)
                if position:
                    pos_type = "LONG 📈" if position['signal'] == 1 else "SHORT 📉"
                    st.markdown(f"**Position Type:** {pos_type}")
            
            with col3:
                if position:
                    pnl = (current_price - position['entry_price']) * position['signal'] * quantity
                    if pnl >= 0:
                        st.metric("Unrealized P&L", f"₹{pnl:.2f}", delta=f"+{pnl:.2f}", delta_color="normal")
                    else:
                        st.metric("Unrealized P&L", f"₹{pnl:.2f}", delta=f"{pnl:.2f}", delta_color="inverse")
                else:
                    st.metric("Unrealized P&L", "₹0.00", delta="0.00")
            
            with col4:
                st.metric("Last Update", current_time.strftime("%H:%M:%S"))
                st.metric("Iteration", iteration)
            
            # Row 2 - Indicator Values (ALWAYS SHOW)
            st.markdown("---")
            st.markdown("**📉 Technical Indicators**")
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                if len(df) > 0 and 'EMA_Fast' in df.columns and pd.notna(df['EMA_Fast'].iloc[-1]):
                    ema_fast_val = df['EMA_Fast'].iloc[-1]
                    st.metric("EMA Fast", f"{ema_fast_val:.2f}")
                else:
                    st.metric("EMA Fast", "Calculating...")
            
            with col6:
                if len(df) > 0 and 'EMA_Slow' in df.columns and pd.notna(df['EMA_Slow'].iloc[-1]):
                    ema_slow_val = df['EMA_Slow'].iloc[-1]
                    st.metric("EMA Slow", f"{ema_slow_val:.2f}")
                else:
                    st.metric("EMA Slow", "Calculating...")
            
            with col7:
                if len(df) > 0 and 'RSI' in df.columns and pd.notna(df['RSI'].iloc[-1]):
                    rsi_val = df['RSI'].iloc[-1]
                    if rsi_val > 70:
                        st.metric("RSI", f"{rsi_val:.2f}", delta="Overbought")
                    elif rsi_val < 30:
                        st.metric("RSI", f"{rsi_val:.2f}", delta="Oversold")
                    else:
                        st.metric("RSI", f"{rsi_val:.2f}")
                else:
                    st.metric("RSI", "Calculating...")
            
            with col8:
                if len(df) > 0 and 'EMA_Fast' in df.columns and 'EMA_Slow' in df.columns:
                    if pd.notna(df['EMA_Fast'].iloc[-1]) and pd.notna(df['EMA_Slow'].iloc[-1]):
                        current_signal_val = "BUY 🟢" if df['EMA_Fast'].iloc[-1] > df['EMA_Slow'].iloc[-1] else "SELL 🔴"
                        st.markdown("**Current Signal**")
                        st.markdown(current_signal_val)
                    else:
                        st.metric("Current Signal", "Waiting...")
                else:
                    st.metric("Current Signal", "Calculating...")
            
            # Row 3 - Additional Metrics
            col9, col10, col11, col12 = st.columns(4)
            
            with col9:
                if len(df) > 0 and 'EMA_Angle' in df.columns and pd.notna(df['EMA_Angle'].iloc[-1]):
                    st.metric("Crossover Angle", f"{df['EMA_Angle'].iloc[-1]:.2f}°")
                else:
                    st.metric("Crossover Angle", "-")
            
            with col10:
                if len(df) > 0 and 'ADX' in df.columns and pd.notna(df['ADX'].iloc[-1]):
                    st.metric("ADX", f"{df['ADX'].iloc[-1]:.2f}")
                else:
                    st.metric("ADX", "-")
            
            with col11:
                if len(df) > 0 and 'ATR' in df.columns and pd.notna(df['ATR'].iloc[-1]):
                    st.metric("ATR", f"{df['ATR'].iloc[-1]:.2f}")
                else:
                    st.metric("ATR", "-")
            
            with col12:
                if len(df) > 0 and 'Volume' in df.columns:
                    vol = df['Volume'].iloc[-1]
                    if vol > 0:
                        st.metric("Volume", f"{vol:,.0f}")
                    else:
                        st.metric("Volume", "N/A (Index)")
                else:
                    st.metric("Volume", "-")
            
            if strategy == 'EMA Crossover':
                entry_filter = config.get('entry_filter', 'Simple Crossover')
                st.markdown(f"**Entry Filter:** {entry_filter}")
                
                if len(df) > 0 and 'Close' in df.columns and 'Open' in df.columns:
                    candle_size = abs(df['Close'].iloc[-1] - df['Open'].iloc[-1])
                    if entry_filter == 'Custom Candle (Points)':
                        custom_points = config.get('custom_points', 10)
                        status = "✅" if candle_size >= custom_points else "❌"
                        st.markdown(f"{status} Candle Size: {candle_size:.2f} / Min: {custom_points:.2f}")
                    elif entry_filter == 'ATR-based Candle':
                        if 'ATR' in df.columns:
                            atr = df['ATR'].iloc[-1]
                            multiplier = config.get('atr_multiplier', 1.5)
                            min_candle = atr * multiplier
                            status = "✅" if candle_size >= min_candle else "❌"
                            st.markdown(f"{status} Candle Size: {candle_size:.2f} / Min (ATR×{multiplier}): {min_candle:.2f}")
                    
                    if config.get('use_adx', False) and 'ADX' in df.columns:
                        adx = df['ADX'].iloc[-1]
                        threshold = config.get('adx_threshold', 25)
                        status = "✅" if adx >= threshold else "❌"
                        st.markdown(f"{status} ADX: {adx:.2f} / Threshold: {threshold:.2f}")
        
        if position:
            with position_placeholder:
                st.markdown("### 💼 Position Information")
                duration_seconds = (current_time - position['entry_time']).total_seconds()
                duration_str = f"{int(duration_seconds // 3600)}h {int((duration_seconds % 3600) // 60)}m"
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Entry Time", position['entry_time'].strftime("%H:%M:%S"))
                    st.metric("Duration", duration_str)
                with col2:
                    sl_val = position.get('sl', 0)
                    sl_str = f"{sl_val:.2f}" if sl_val != 0 else "Signal Based"
                    st.metric("Stop Loss", sl_str)
                    if sl_val != 0:
                        dist_sl = abs(current_price - sl_val)
                        st.metric("Distance to SL", f"{dist_sl:.2f}")
                with col3:
                    target_val = position.get('target', 0)
                    target_str = f"{target_val:.2f}" if target_val != 0 else "Signal Based"
                    st.metric("Target", target_str)
                    if target_val != 0:
                        dist_target = abs(target_val - current_price)
                        st.metric("Distance to Target", f"{dist_target:.2f}")
                with col4:
                    highest = st.session_state.get('highest_price', 0)
                    lowest = st.session_state.get('lowest_price', 0)
                    st.metric("Highest Price", f"{highest:.2f}" if highest else "N/A")
                    st.metric("Lowest Price", f"{lowest:.2f}" if lowest else "N/A")
                
                if st.session_state.get('partial_exit_done', False):
                    st.success("✅ 50% position already exited - Trailing remaining")
                
                if st.session_state.get('breakeven_activated', False):
                    st.info("ℹ️ SL moved to break-even")
        else:
            # Display message when no position
            with position_placeholder:
                st.info("No active position - Waiting for entry signal...")
        
        with chart_placeholder:
            st.markdown("### 📈 Live Chart")
            
            if len(df) > 0:
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
                    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], name='EMA Fast', line=dict(color='blue', width=1)))
                if 'EMA_Slow' in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], name='EMA Slow', line=dict(color='red', width=1)))
                
                if position:
                    fig.add_hline(y=position['entry_price'], line_dash="dash", line_color="yellow", annotation_text="Entry")
                    if position.get('sl', 0) != 0:
                        fig.add_hline(y=position['sl'], line_dash="dash", line_color="red", annotation_text="SL")
                    if position.get('target', 0) != 0:
                        fig.add_hline(y=position['target'], line_dash="dash", line_color="green", annotation_text="Target")
                
                fig.update_layout(
                    title=f"{ticker} - {interval}",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    height=500,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"live_chart_{current_time.timestamp()}")
            else:
                st.warning("No data available for chart")
        
        with status_placeholder:
            if position:
                pnl = (current_price - position['entry_price']) * position['signal'] * quantity
                if pnl > 0:
                    st.success(f"✅ Trading ACTIVE | In Position | Profit: {pnl:.2f}")
                else:
                    st.warning(f"⚠️ Trading ACTIVE | In Position | Loss: {pnl:.2f}")
            else:
                st.info("🔵 Trading ACTIVE | Waiting for signal...")
        
        time.sleep(random.uniform(1.0, 1.5))
    
    st.info("⚪ Trading STOPPED")

# ==================== MAIN APP ====================

def main():
    st.set_page_config(page_title="Algo Trading System", layout="wide")
    st.title("🚀 Advanced Algo Trading System")
    
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
    st.sidebar.title("⚙️ Configuration")
    
    # Asset selection
    st.sidebar.subheader("📊 Asset Selection")
    
    asset_type = st.sidebar.selectbox(
        "Asset Type",
        ["Indian Indices", "Crypto", "Forex", "Commodities", "Custom Ticker"]
    )
    
    ticker_map = {
        "Indian Indices": {
            "NIFTY 50": "^NSEI",
            "BANKNIFTY": "^NSEBANK",
            "SENSEX": "^BSESN"
        },
        "Crypto": {
            "Bitcoin": "BTC-USD",
            "Ethereum": "ETH-USD"
        },
        "Forex": {
            "USD/INR": "USDINR=X",
            "EUR/USD": "EURUSD=X",
            "GBP/USD": "GBPUSD=X"
        },
        "Commodities": {
            "Gold": "GC=F",
            "Silver": "SI=F"
        }
    }
    
    if asset_type == "Custom Ticker":
        ticker = st.sidebar.text_input("Enter Ticker Symbol", "AAPL")
    else:
        asset_name = st.sidebar.selectbox("Select Asset", list(ticker_map[asset_type].keys()))
        ticker = ticker_map[asset_type][asset_name]
    
    # Timeframe selection
    st.sidebar.subheader("⏰ Timeframe")
    interval = st.sidebar.selectbox(
        "Interval",
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"]
    )
    
    # Period selection based on interval
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
    
    period = st.sidebar.selectbox("Period", period_options[interval])
    
    # Validate combination
    if not validate_timeframe_period(interval, period):
        st.sidebar.error("⚠️ Invalid interval-period combination!")
    
    # Quantity
    quantity = st.sidebar.number_input("Quantity", min_value=1, value=1, step=1, key="quantity_input")
    
    # Mode selection
    mode = st.sidebar.selectbox("Mode", ["Live Trading", "Backtest"])
    
    # Strategy selection
    st.sidebar.subheader("📈 Strategy")
    strategy = st.sidebar.selectbox(
        "Select Strategy",
        [
            "EMA Crossover",
            "Simple Buy",
            "Simple Sell",
            "Price Crosses Threshold",
            "RSI-ADX-EMA",
            "AI Price Action Analysis",
            "Custom Strategy Builder"
        ]
    )
    
    # Strategy-specific configuration
    config = {
        'ticker': ticker,
        'interval': interval,
        'period': period,
        'strategy': strategy,
        'quantity': quantity,
        'mode': mode
    }
    
    # EMA Crossover configuration
    if strategy == "EMA Crossover":
        st.sidebar.subheader("EMA Settings")
        config['ema_fast'] = st.sidebar.number_input("EMA Fast", min_value=1, value=9, step=1, key="ema_fast_input")
        config['ema_slow'] = st.sidebar.number_input("EMA Slow", min_value=1, value=15, step=1, key="ema_slow_input")
        config['min_angle'] = st.sidebar.number_input("Min Crossover Angle (degrees)", min_value=0.0, value=1.0, step=0.1, key="min_angle_input")
        
        st.sidebar.subheader("Entry Filter")
        config['entry_filter'] = st.sidebar.selectbox(
            "Entry Filter Type",
            ["Simple Crossover", "Custom Candle (Points)", "ATR-based Candle"]
        )
        
        if config['entry_filter'] == "Custom Candle (Points)":
            config['custom_points'] = st.sidebar.number_input("Custom Points", min_value=1.0, value=10.0, step=1.0, key="custom_points_input")
        elif config['entry_filter'] == "ATR-based Candle":
            config['atr_multiplier'] = st.sidebar.number_input("ATR Multiplier", min_value=0.1, value=1.5, step=0.1, key="atr_mult_input")
        
        config['use_adx'] = st.sidebar.checkbox("Use ADX Filter", value=False)
        if config['use_adx']:
            config['adx_period'] = st.sidebar.number_input("ADX Period", min_value=1, value=14, step=1, key="adx_period_input")
            config['adx_threshold'] = st.sidebar.number_input("ADX Threshold", min_value=1, value=25, step=1, key="adx_threshold_input")
    
    # Price Crosses Threshold configuration
    elif strategy == "Price Crosses Threshold":
        st.sidebar.subheader("Threshold Settings")
        config['threshold_value'] = st.sidebar.number_input("Threshold Value", min_value=0.0, value=100.0, step=1.0, key="threshold_input")
        config['threshold_type'] = st.sidebar.selectbox(
            "Threshold Type",
            [
                "LONG (Price >= Threshold)",
                "SHORT (Price >= Threshold)",
                "LONG (Price <= Threshold)",
                "SHORT (Price <= Threshold)"
            ]
        )
    
    # Custom Strategy Builder
    elif strategy == "Custom Strategy Builder":
        st.sidebar.subheader("Custom Conditions")
        
        if st.sidebar.button("➕ Add Condition"):
            st.session_state['custom_conditions'].append({
                'active': True,
                'use_price': False,
                'indicator': 'RSI',
                'operator': '>',
                'value': 70,
                'action': 'BUY'
            })
            st.session_state['custom_conditions'] = st.session_state['custom_conditions']
        
        conditions_to_remove = []
        for idx, cond in enumerate(st.session_state['custom_conditions']):
            st.sidebar.markdown(f"**Condition {idx + 1}**")
            
            cond['active'] = st.sidebar.checkbox(f"Use condition {idx + 1}", value=cond.get('active', True), key=f"active_{idx}")
            
            if cond['active']:
                cond['use_price'] = st.sidebar.checkbox(f"Use Price", value=cond.get('use_price', False), key=f"use_price_{idx}")
                
                cond['indicator'] = st.sidebar.selectbox(
                    f"Indicator",
                    ["Price", "RSI", "ADX", "EMA_Fast", "EMA_Slow", "SuperTrend", "MACD", "MACD_Signal", 
                     "BB_Upper", "BB_Lower", "ATR", "Volume", "VWAP", "KC_Upper", "KC_Lower",
                     "Close", "High", "Low", "Support", "Resistance", "EMA_20", "EMA_50"],
                    key=f"indicator_{idx}",
                    index=["Price", "RSI", "ADX", "EMA_Fast", "EMA_Slow", "SuperTrend", "MACD", "MACD_Signal", 
                           "BB_Upper", "BB_Lower", "ATR", "Volume", "VWAP", "KC_Upper", "KC_Lower",
                           "Close", "High", "Low", "Support", "Resistance", "EMA_20", "EMA_50"].index(cond.get('indicator', 'RSI'))
                )
                
                cond['operator'] = st.sidebar.selectbox(
                    f"Operator",
                    [">", "<", ">=", "<=", "==", "crosses_above", "crosses_below"],
                    key=f"operator_{idx}",
                    index=[">", "<", ">=", "<=", "==", "crosses_above", "crosses_below"].index(cond.get('operator', '>'))
                )
                
                if not cond['use_price']:
                    cond['value'] = st.sidebar.number_input(
                        f"Value",
                        value=float(cond.get('value', 70)),
                        step=1.0,
                        key=f"value_{idx}"
                    )
                
                cond['action'] = st.sidebar.selectbox(
                    f"Action",
                    ["BUY", "SELL"],
                    key=f"action_{idx}",
                    index=["BUY", "SELL"].index(cond.get('action', 'BUY'))
                )
                
                if st.sidebar.button(f"🗑️ Remove Condition {idx + 1}", key=f"remove_{idx}"):
                    conditions_to_remove.append(idx)
            
            st.sidebar.markdown("---")
        
        # Remove conditions
        for idx in reversed(conditions_to_remove):
            st.session_state['custom_conditions'].pop(idx)
            st.session_state['custom_conditions'] = st.session_state['custom_conditions']
        
        config['custom_conditions'] = st.session_state['custom_conditions']
    
    # Stop Loss configuration
    st.sidebar.subheader("🛑 Stop Loss")
    config['sl_type'] = st.sidebar.selectbox(
        "SL Type",
        [
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
        ]
    )
    
    if 'Custom Points' in config['sl_type'] or 'Trailing SL' in config['sl_type']:
        config['sl_points'] = st.sidebar.number_input("SL Points", min_value=1.0, value=10.0, step=1.0, key="sl_points_input")
    
    if 'Trailing' in config['sl_type']:
        config['trailing_threshold'] = st.sidebar.number_input("Trailing Threshold (Points)", min_value=0.0, value=0.0, step=1.0, key="trailing_threshold_input")
    
    if config['sl_type'] == "ATR-based" or config['sl_type'] == "Volatility-Adjusted Trailing SL":
        config['atr_multiplier'] = st.sidebar.number_input("ATR Multiplier (SL)", min_value=0.1, value=1.5, step=0.1, key="sl_atr_mult_input")
    
    config['min_sl_distance'] = st.sidebar.number_input("Min SL Distance (Points)", min_value=1.0, value=10.0, step=1.0, key="min_sl_dist_input")
    
    # Target configuration
    st.sidebar.subheader("🎯 Target")
    config['target_type'] = st.sidebar.selectbox(
        "Target Type",
        [
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
        ]
    )
    
    if 'Custom Points' in config['target_type'] or 'Trailing Target' in config['target_type']:
        config['target_points'] = st.sidebar.number_input("Target Points", min_value=1.0, value=20.0, step=1.0, key="target_points_input")
    
    if config['target_type'] == "ATR-based":
        config['target_atr_multiplier'] = st.sidebar.number_input("ATR Multiplier (Target)", min_value=0.1, value=2.0, step=0.1, key="target_atr_mult_input")
    
    if config['target_type'] == "Risk-Reward Based":
        config['rr_ratio'] = st.sidebar.number_input("Risk-Reward Ratio", min_value=0.1, value=2.0, step=0.1, key="rr_ratio_input")
    
    config['min_target_distance'] = st.sidebar.number_input("Min Target Distance (Points)", min_value=1.0, value=15.0, step=1.0, key="min_target_dist_input")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Trading Dashboard", "📈 Trade History", "📝 Trade Logs", "🔬 Backtest Results"])
    
    # Tab 1: Live Trading Dashboard
    with tab1:
        if mode == "Live Trading":
            st.markdown("### 🎮 Trading Controls")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("▶️ Start Trading", type="primary", use_container_width=True):
                    st.session_state['trading_active'] = True
                    add_log("Trading started")
                    st.rerun()
            
            with col2:
                if st.button("⏹️ Stop Trading", use_container_width=True):
                    if st.session_state.get('position') is not None:
                        position = st.session_state['position']
                        current_price = st.session_state.get('current_data')['Close'].iloc[-1] if st.session_state.get('current_data') is not None else position['entry_price']
                        pnl = (current_price - position['entry_price']) * position['signal'] * quantity
                        
                        trade = {
                            'entry_time': position['entry_time'],
                            'exit_time': get_ist_time(),
                            'duration': (get_ist_time() - position['entry_time']).total_seconds() / 3600,
                            'signal': 'LONG' if position['signal'] == 1 else 'SHORT',
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'sl': position['sl'],
                            'target': position['target'],
                            'exit_reason': 'Manual Close',
                            'pnl': pnl,
                            'highest_price': st.session_state.get('highest_price'),
                            'lowest_price': st.session_state.get('lowest_price'),
                            'range': abs(st.session_state.get('highest_price', 0) - st.session_state.get('lowest_price', 0))
                        }
                        
                        st.session_state['trade_history'].append(trade)
                        st.session_state['trade_history'] = st.session_state['trade_history']
                        add_log(f"Position manually closed | P&L: {pnl:.2f}")
                    
                    st.session_state['trading_active'] = False
                    reset_position_state()
                    add_log("Trading stopped")
                    st.rerun()
            
            with col3:
                if st.session_state.get('trading_active', False):
                    st.success("🟢 Trading is ACTIVE")
                else:
                    st.info("⚪ Trading is STOPPED")
            
            if st.button("🔄 Manual Refresh"):
                st.rerun()
            
            st.markdown("---")
            
            if st.session_state.get('trading_active', False):
                live_trading_loop(config)
            else:
                st.info("Click 'Start Trading' to begin live trading monitoring")
        else:
            st.info("Switch to 'Live Trading' mode in the sidebar to access live trading dashboard")
    
    # Tab 2: Trade History
    with tab2:
        st.markdown("### 📈 Trade History")
        
        if len(st.session_state['trade_history']) == 0:
            st.info("No trades yet")
        else:
            total_trades = len(st.session_state['trade_history'])
            winning_trades = len([t for t in st.session_state['trade_history'] if t['pnl'] > 0])
            losing_trades = len([t for t in st.session_state['trade_history'] if t['pnl'] <= 0])
            total_pnl = sum([t['pnl'] for t in st.session_state['trade_history']])
            accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
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
                if total_pnl >= 0:
                    st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"+{total_pnl:.2f}")
                else:
                    st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color="inverse")
            
            st.markdown("---")
            
            for idx, trade in enumerate(reversed(st.session_state['trade_history'])):
                with st.expander(f"Trade #{total_trades - idx} - {trade['signal']} - P&L: {trade['pnl']:.2f}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Entry Time:** {trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"**Exit Time:** {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                        dur_hours = int(trade['duration'])
                        dur_mins = int((trade['duration'] - dur_hours) * 60)
                        st.write(f"**Duration:** {dur_hours}h {dur_mins}m")
                        st.write(f"**Signal:** {trade['signal']}")
                        st.write(f"**Entry Price:** {trade['entry_price']:.2f}")
                        st.write(f"**Exit Price:** {trade['exit_price']:.2f}")
                    
                    with col2:
                        sl_str = f"{trade['sl']:.2f}" if trade.get('sl', 0) != 0 else "Signal Based"
                        target_str = f"{trade['target']:.2f}" if trade.get('target', 0) != 0 else "Signal Based"
                        st.write(f"**Stop Loss:** {sl_str}")
                        st.write(f"**Target:** {target_str}")
                        st.write(f"**Exit Reason:** {trade.get('exit_reason', 'Unknown')}")
                        
                        if trade['pnl'] >= 0:
                            st.success(f"**P&L:** +{trade['pnl']:.2f}")
                        else:
                            st.error(f"**P&L:** {trade['pnl']:.2f}")
                        
                        highest = trade.get('highest_price', 0)
                        lowest = trade.get('lowest_price', 0)
                        range_val = trade.get('range', 0)
                        st.write(f"**Highest Price:** {highest:.2f}" if highest else "**Highest Price:** N/A")
                        st.write(f"**Lowest Price:** {lowest:.2f}" if lowest else "**Lowest Price:** N/A")
                        st.write(f"**Range:** {range_val:.2f}" if range_val else "**Range:** N/A")
    
    # Tab 3: Trade Logs
    with tab3:
        st.markdown("### 📝 Trade Logs")
        
        if len(st.session_state['trade_logs']) == 0:
            st.info("No logs yet")
        else:
            for log in reversed(st.session_state['trade_logs']):
                st.text(log)
    
    # Tab 4: Backtest Results
    with tab4:
        if mode == "Backtest":
            st.markdown("### 🔬 Backtest Results")
            
            if st.button("▶️ Run Backtest", type="primary"):
                with st.spinner("Running backtest..."):
                    df = fetch_data(ticker, interval, period, mode='backtest')
                    
                    if df is None or len(df) < 50:
                        st.error("Failed to fetch data or insufficient data for backtest")
                    else:
                        df = add_all_indicators(df, config)
                        
                        results = run_backtest(df, strategy, config)
                        
                        st.success("Backtest completed!")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Total Trades", results['total_trades'])
                        with col2:
                            st.metric("Winning Trades", results['winning_trades'])
                        with col3:
                            st.metric("Losing Trades", results['losing_trades'])
                        with col4:
                            st.metric("Accuracy", f"{results['accuracy']:.1f}%")
                        with col5:
                            if results['total_pnl'] >= 0:
                                st.metric("Total P&L", f"{results['total_pnl']:.2f}", delta=f"+{results['total_pnl']:.2f}")
                            else:
                                st.metric("Total P&L", f"{results['total_pnl']:.2f}", delta=f"{results['total_pnl']:.2f}", delta_color="inverse")
                        
                        st.metric("Avg Trade Duration", f"{results['avg_duration']:.2f} hours")
                        
                        st.markdown("---")
                        st.markdown("### 📊 All Trades")
                        
                        if len(results['trades']) == 0:
                            st.info("No trades generated in backtest")
                        else:
                            for idx, trade in enumerate(results['trades']):
                                with st.expander(f"Trade #{idx + 1} - {trade['signal']} - P&L: {trade['pnl']:.2f}"):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write(f"**Entry Time:** {trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                                        st.write(f"**Exit Time:** {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                                        dur_hours = int(trade['duration'])
                                        dur_mins = int((trade['duration'] - dur_hours) * 60)
                                        st.write(f"**Duration:** {dur_hours}h {dur_mins}m")
                                        st.write(f"**Signal:** {trade['signal']}")
                                        st.write(f"**Entry Price:** {trade['entry_price']:.2f}")
                                        st.write(f"**Exit Price:** {trade['exit_price']:.2f}")
                                    
                                    with col2:
                                        sl_str = f"{trade['sl']:.2f}" if trade.get('sl', 0) != 0 else "Signal Based"
                                        target_str = f"{trade['target']:.2f}" if trade.get('target', 0) != 0 else "Signal Based"
                                        st.write(f"**Stop Loss:** {sl_str}")
                                        st.write(f"**Target:** {target_str}")
                                        st.write(f"**Exit Reason:** {trade.get('exit_reason', 'Unknown')}")
                                        
                                        if trade['pnl'] >= 0:
                                            st.success(f"**P&L:** +{trade['pnl']:.2f}")
                                        else:
                                            st.error(f"**P&L:** {trade['pnl']:.2f}")
                                        
                                        highest = trade.get('highest_price', 0)
                                        lowest = trade.get('lowest_price', 0)
                                        range_val = trade.get('range', 0)
                                        st.write(f"**Highest Price:** {highest:.2f}" if highest else "**Highest Price:** N/A")
                                        st.write(f"**Lowest Price:** {lowest:.2f}" if lowest else "**Lowest Price:** N/A")
                                        st.write(f"**Range:** {range_val:.2f}" if range_val else "**Range:** N/A")
        else:
            st.info("Switch to 'Backtest' mode in the sidebar to run backtests")

if __name__ == "__main__":
    main()
