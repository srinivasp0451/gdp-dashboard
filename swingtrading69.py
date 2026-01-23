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

def validate_interval_period(interval, period):
    """Validate interval and period combinations"""
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

def convert_to_ist(df):
    """Convert dataframe datetime to IST"""
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
    else:
        df.index = df.index.tz_convert('Asia/Kolkata')
    return df

def fetch_data(ticker, interval, period, mode):
    """Fetch data from yfinance with proper handling"""
    try:
        if mode == "Live Trading":
            time.sleep(random.uniform(1.0, 1.5))
        
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
        
        data = data[required_cols]
        data = convert_to_ist(data)
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def add_log(message):
    """Add timestamped log entry"""
    timestamp = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    st.session_state['trade_logs'].append(log_entry)
    st.session_state['trade_logs'] = st.session_state['trade_logs'][-50:]

def reset_position_state():
    """Reset position-related session state"""
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

# ============================================================================
# INDICATOR CALCULATIONS (MANUAL IMPLEMENTATION)
# ============================================================================

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
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

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
    
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period).mean()
    
    return adx

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data, period=20, std=2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(data, period)
    rolling_std = data.rolling(window=period).std()
    upper = sma + (rolling_std * std)
    lower = sma - (rolling_std * std)
    return upper, sma, lower

def calculate_stochastic_rsi(data, period=14, smooth_k=3, smooth_d=3):
    """Calculate Stochastic RSI"""
    rsi = calculate_rsi(data, period)
    stoch = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min()) * 100
    k = stoch.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d

def calculate_keltner_channel(df, period=20, atr_mult=2):
    """Calculate Keltner Channel"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    middle = calculate_ema(typical_price, period)
    atr = calculate_atr(df, period)
    upper = middle + (atr * atr_mult)
    lower = middle - (atr * atr_mult)
    return upper, middle, lower

def calculate_pivot_points(df):
    """Calculate Pivot Points"""
    pivot = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    r1 = 2 * pivot - df['Low'].shift(1)
    s1 = 2 * pivot - df['High'].shift(1)
    r2 = pivot + (df['High'].shift(1) - df['Low'].shift(1))
    s2 = pivot - (df['High'].shift(1) - df['Low'].shift(1))
    return pivot, r1, r2, s1, s2

def calculate_vwap(df):
    """Calculate VWAP"""
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        return (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return pd.Series(0, index=df.index)

def calculate_supertrend(df, period=10, multiplier=3):
    """Calculate SuperTrend"""
    atr = calculate_atr(df, period)
    hl_avg = (df['High'] + df['Low']) / 2
    
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)
    
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > upper_band.iloc[i-1]:
            direction.iloc[i] = 1
        elif df['Close'].iloc[i] < lower_band.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
        
        if direction.iloc[i] == 1:
            supertrend.iloc[i] = lower_band.iloc[i]
        else:
            supertrend.iloc[i] = upper_band.iloc[i]
    
    return supertrend

def detect_swing_high_low(df, lookback=5):
    """Detect swing highs and lows"""
    swing_high = pd.Series(np.nan, index=df.index)
    swing_low = pd.Series(np.nan, index=df.index)
    
    for i in range(lookback, len(df) - lookback):
        if df['High'].iloc[i] == df['High'].iloc[i-lookback:i+lookback+1].max():
            swing_high.iloc[i] = df['High'].iloc[i]
        if df['Low'].iloc[i] == df['Low'].iloc[i-lookback:i+lookback+1].min():
            swing_low.iloc[i] = df['Low'].iloc[i]
    
    swing_high = swing_high.ffill()
    swing_low = swing_low.ffill()
    
    return swing_high, swing_low

def calculate_support_resistance(df, lookback=20):
    """Calculate support and resistance levels"""
    resistance = df['High'].rolling(window=lookback).max()
    support = df['Low'].rolling(window=lookback).min()
    return support, resistance

def calculate_ema_angle(ema_series):
    """Calculate EMA angle in degrees"""
    slope = ema_series.diff()
    angle = np.degrees(np.arctan(slope))
    return angle

# ============================================================================
# STRATEGY IMPLEMENTATIONS
# ============================================================================

def ema_crossover_strategy(df, config):
    """EMA Crossover Strategy with filters"""
    fast = config.get('ema_fast', 9)
    slow = config.get('ema_slow', 15)
    min_angle = config.get('min_angle', 1)
    entry_filter = config.get('entry_filter', 'Simple Crossover')
    custom_points = config.get('custom_points', 10)
    atr_multiplier = config.get('atr_multiplier', 1.5)
    use_adx = config.get('use_adx', False)
    adx_period = config.get('adx_period', 14)
    adx_threshold = config.get('adx_threshold', 25)
    
    df['EMA_Fast'] = calculate_ema(df['Close'], fast)
    df['EMA_Slow'] = calculate_ema(df['Close'], slow)
    df['EMA_Angle'] = calculate_ema_angle(df['EMA_Fast'])
    df['ATR'] = calculate_atr(df, 14)
    
    if use_adx:
        df['ADX'] = calculate_adx(df, adx_period)
    
    signals = []
    
    for i in range(1, len(df)):
        signal = 0
        
        bullish_cross = (df['EMA_Fast'].iloc[i] > df['EMA_Slow'].iloc[i] and 
                        df['EMA_Fast'].iloc[i-1] <= df['EMA_Slow'].iloc[i-1])
        bearish_cross = (df['EMA_Fast'].iloc[i] < df['EMA_Slow'].iloc[i] and 
                        df['EMA_Fast'].iloc[i-1] >= df['EMA_Slow'].iloc[i-1])
        
        angle_valid = abs(df['EMA_Angle'].iloc[i]) >= min_angle
        
        if bullish_cross and angle_valid:
            if entry_filter == 'Simple Crossover':
                signal = 1
            elif entry_filter == 'Custom Candle (Points)':
                candle_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
                if candle_size >= custom_points:
                    signal = 1
            elif entry_filter == 'ATR-based Candle':
                candle_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
                min_size = df['ATR'].iloc[i] * atr_multiplier
                if candle_size >= min_size:
                    signal = 1
        
        elif bearish_cross and angle_valid:
            if entry_filter == 'Simple Crossover':
                signal = -1
            elif entry_filter == 'Custom Candle (Points)':
                candle_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
                if candle_size >= custom_points:
                    signal = -1
            elif entry_filter == 'ATR-based Candle':
                candle_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
                min_size = df['ATR'].iloc[i] * atr_multiplier
                if candle_size >= min_size:
                    signal = -1
        
        if use_adx and signal != 0:
            if df['ADX'].iloc[i] < adx_threshold:
                signal = 0
        
        signals.append(signal)
    
    df['Signal'] = [0] + signals
    return df

def simple_buy_strategy(df, config):
    """Simple Buy Strategy"""
    df['Signal'] = 1
    return df

def simple_sell_strategy(df, config):
    """Simple Sell Strategy"""
    df['Signal'] = -1
    return df

def price_threshold_strategy(df, config):
    """Price Crosses Threshold Strategy"""
    threshold = config.get('threshold', 0)
    direction = config.get('direction', 'LONG (Price >= Threshold)')
    
    signals = []
    
    for i in range(len(df)):
        signal = 0
        price = df['Close'].iloc[i]
        
        if direction == 'LONG (Price >= Threshold)' and price >= threshold:
            signal = 1
        elif direction == 'SHORT (Price >= Threshold)' and price >= threshold:
            signal = -1
        elif direction == 'LONG (Price <= Threshold)' and price <= threshold:
            signal = 1
        elif direction == 'SHORT (Price <= Threshold)' and price <= threshold:
            signal = -1
        
        signals.append(signal)
    
    df['Signal'] = signals
    return df

def rsi_adx_ema_strategy(df, config):
    """RSI-ADX-EMA Strategy"""
    rsi_period = config.get('rsi_period', 14)
    adx_period = config.get('adx_period', 14)
    ema1_period = config.get('ema1_period', 9)
    ema2_period = config.get('ema2_period', 15)
    
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

def percentage_change_strategy(df, config):
    """Percentage Change Strategy"""
    pct_threshold = config.get('pct_threshold', 0.01)
    direction = config.get('direction', 'BUY on Fall')
    
    first_price = df['Close'].iloc[0]
    signals = []
    
    for i in range(len(df)):
        signal = 0
        current_price = df['Close'].iloc[i]
        pct_change = ((current_price - first_price) / first_price) * 100
        
        if direction == 'BUY on Fall' and pct_change <= -pct_threshold:
            signal = 1
        elif direction == 'SELL on Fall' and pct_change <= -pct_threshold:
            signal = -1
        elif direction == 'BUY on Rise' and pct_change >= pct_threshold:
            signal = 1
        elif direction == 'SELL on Rise' and pct_change >= pct_threshold:
            signal = -1
        
        signals.append(signal)
    
    df['Signal'] = signals
    df['Pct_Change'] = ((df['Close'] - first_price) / first_price) * 100
    return df

def ai_price_action_strategy(df, config):
    """AI Price Action Analysis Strategy"""
    df['EMA_20'] = calculate_ema(df['Close'], 20)
    df['EMA_50'] = calculate_ema(df['Close'], 50)
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    df['ATR'] = calculate_atr(df, 14)
    
    signals = []
    
    for i in range(len(df)):
        score = 0
        
        if df['EMA_20'].iloc[i] > df['EMA_50'].iloc[i]:
            score += 1
        else:
            score -= 1
        
        if df['RSI'].iloc[i] < 30:
            score += 1
        elif df['RSI'].iloc[i] > 70:
            score -= 1
        
        if df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]:
            score += 1
        else:
            score -= 1
        
        if df['Close'].iloc[i] < df['BB_Lower'].iloc[i]:
            score += 1
        elif df['Close'].iloc[i] > df['BB_Upper'].iloc[i]:
            score -= 1
        
        has_volume = 'Volume' in df.columns and df['Volume'].sum() > 0
        if has_volume:
            avg_vol = df['Volume'].rolling(20).mean().iloc[i]
            if df['Volume'].iloc[i] > avg_vol * 1.5:
                score += 0.5
        
        if score >= 2:
            signal = 1
        elif score <= -2:
            signal = -1
        else:
            signal = 0
        
        signals.append(signal)
    
    df['Signal'] = signals
    df['AI_Score'] = [0] * len(df)
    return df

def custom_strategy_builder(df, config):
    """Custom Strategy Builder"""
    conditions = config.get('conditions', [])
    
    df['EMA_Fast'] = calculate_ema(df['Close'], config.get('ema_fast', 9))
    df['EMA_Slow'] = calculate_ema(df['Close'], config.get('ema_slow', 15))
    df['EMA_20'] = calculate_ema(df['Close'], 20)
    df['EMA_50'] = calculate_ema(df['Close'], 50)
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['ADX'] = calculate_adx(df, 14)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    df['ATR'] = calculate_atr(df, 14)
    df['SuperTrend'] = calculate_supertrend(df)
    df['VWAP'] = calculate_vwap(df)
    df['Support'], df['Resistance'] = calculate_support_resistance(df)
    upper_kelt, middle_kelt, lower_kelt = calculate_keltner_channel(df)
    df['Keltner_Upper'] = upper_kelt
    df['Keltner_Lower'] = lower_kelt
    
    signals = []
    
    for i in range(1, len(df)):
        buy_conditions_met = True
        sell_conditions_met = True
        has_buy = False
        has_sell = False
        
        for cond in conditions:
            if not cond.get('use', False):
                continue
            
            indicator = cond.get('indicator', 'RSI')
            operator = cond.get('operator', '>')
            value = cond.get('value', 0)
            action = cond.get('action', 'BUY')
            compare_price = cond.get('compare_price', False)
            compare_indicator = cond.get('compare_indicator', 'EMA_20')
            
            if indicator == 'Price':
                ind_val = df['Close'].iloc[i]
            else:
                ind_val = df.get(indicator, pd.Series(0, index=df.index)).iloc[i]
            
            if compare_price:
                compare_val = df.get(compare_indicator, pd.Series(0, index=df.index)).iloc[i]
            else:
                compare_val = value
            
            condition_met = False
            if operator == '>':
                condition_met = ind_val > compare_val
            elif operator == '<':
                condition_met = ind_val < compare_val
            elif operator == '>=':
                condition_met = ind_val >= compare_val
            elif operator == '<=':
                condition_met = ind_val <= compare_val
            elif operator == '==':
                condition_met = abs(ind_val - compare_val) < 0.0001
            elif operator == 'crosses_above':
                prev_val = df.get(indicator, pd.Series(0, index=df.index)).iloc[i-1]
                if compare_price:
                    prev_compare = df.get(compare_indicator, pd.Series(0, index=df.index)).iloc[i-1]
                else:
                    prev_compare = value
                condition_met = ind_val > compare_val and prev_val <= prev_compare
            elif operator == 'crosses_below':
                prev_val = df.get(indicator, pd.Series(0, index=df.index)).iloc[i-1]
                if compare_price:
                    prev_compare = df.get(compare_indicator, pd.Series(0, index=df.index)).iloc[i-1]
                else:
                    prev_compare = value
                condition_met = ind_val < compare_val and prev_val >= prev_compare
            
            if action == 'BUY':
                has_buy = True
                if not condition_met:
                    buy_conditions_met = False
            else:
                has_sell = True
                if not condition_met:
                    sell_conditions_met = False
        
        signal = 0
        if has_buy and buy_conditions_met:
            signal = 1
        elif has_sell and sell_conditions_met:
            signal = -1
        
        signals.append(signal)
    
    df['Signal'] = [0] + signals
    return df

# ============================================================================
# STOP LOSS AND TARGET CALCULATIONS
# ============================================================================

def calculate_stop_loss(df, entry_price, signal, sl_type, config, position=None):
    """Calculate stop loss based on type"""
    sl_points = config.get('sl_points', 10)
    atr_mult = config.get('atr_multiplier', 1.5)
    min_sl_distance = config.get('min_sl_distance', 10)
    
    current_price = df['Close'].iloc[-1]
    
    # Calculate ATR if needed
    if 'ATR' not in df.columns and 'ATR' in sl_type:
        df['ATR'] = calculate_atr(df, 14)
    
    if sl_type == 'Custom Points':
        if signal == 1:
            sl = entry_price - sl_points
        else:
            sl = entry_price + sl_points
    
    elif sl_type == 'Trailing SL (Points)':
        if signal == 1:
            new_sl = current_price - sl_points
            if position and position.get('sl'):
                sl = max(new_sl, position['sl'])
            else:
                sl = new_sl
        else:
            new_sl = current_price + sl_points
            if position and position.get('sl'):
                sl = min(new_sl, position['sl'])
            else:
                sl = new_sl
    
    elif sl_type == 'ATR-based' or 'Volatility' in sl_type:
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else calculate_atr(df, 14).iloc[-1]
        if signal == 1:
            sl = entry_price - (atr * atr_mult)
        else:
            sl = entry_price + (atr * atr_mult)
    
    elif sl_type == 'Current Candle Low/High':
        if signal == 1:
            sl = df['Low'].iloc[-1]
        else:
            sl = df['High'].iloc[-1]
    
    elif sl_type == 'Previous Candle Low/High':
        if len(df) > 1:
            if signal == 1:
                sl = df['Low'].iloc[-2]
            else:
                sl = df['High'].iloc[-2]
        else:
            sl = entry_price - sl_points if signal == 1 else entry_price + sl_points
    
    elif sl_type == 'Current Swing Low/High':
        swing_high, swing_low = detect_swing_high_low(df)
        if signal == 1:
            sl = swing_low.iloc[-1] if not pd.isna(swing_low.iloc[-1]) else entry_price - sl_points
        else:
            sl = swing_high.iloc[-1] if not pd.isna(swing_high.iloc[-1]) else entry_price + sl_points
    
    elif sl_type == 'Previous Swing Low/High':
        swing_high, swing_low = detect_swing_high_low(df)
        if len(df) > 1:
            if signal == 1:
                sl = swing_low.iloc[-2] if not pd.isna(swing_low.iloc[-2]) else entry_price - sl_points
            else:
                sl = swing_high.iloc[-2] if not pd.isna(swing_high.iloc[-2]) else entry_price + sl_points
        else:
            sl = entry_price - sl_points if signal == 1 else entry_price + sl_points
    
    elif 'Signal-based' in sl_type:
        sl = 0
    
    else:
        if signal == 1:
            sl = entry_price - sl_points
        else:
            sl = entry_price + sl_points
    
    if sl != 0 and sl is not None:
        if signal == 1:
            sl = min(sl, entry_price - min_sl_distance)
        else:
            sl = max(sl, entry_price + min_sl_distance)
    
    return sl

def calculate_target(df, entry_price, signal, target_type, config, position=None):
    """Calculate target based on type"""
    target_points = config.get('target_points', 20)
    atr_mult = config.get('target_atr_mult', 2.0)
    rr_ratio = config.get('rr_ratio', 2.0)
    min_target_distance = config.get('min_target_distance', 15)
    
    current_price = df['Close'].iloc[-1]
    
    # Calculate ATR if needed
    if 'ATR' not in df.columns and 'ATR' in target_type:
        df['ATR'] = calculate_atr(df, 14)
    
    if target_type == 'Custom Points':
        if signal == 1:
            target = entry_price + target_points
        else:
            target = entry_price - target_points
    
    elif 'Trailing Target' in target_type:
        if signal == 1:
            highest = st.session_state.get('highest_price', entry_price)
            target = highest if highest is not None else entry_price
        else:
            lowest = st.session_state.get('lowest_price', entry_price)
            target = lowest if lowest is not None else entry_price
    
    elif target_type == 'ATR-based':
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else calculate_atr(df, 14).iloc[-1]
        if signal == 1:
            target = entry_price + (atr * atr_mult)
        else:
            target = entry_price - (atr * atr_mult)
    
    elif target_type == 'Current Candle Low/High':
        if signal == 1:
            target = df['High'].iloc[-1]
        else:
            target = df['Low'].iloc[-1]
    
    elif target_type == 'Previous Candle Low/High':
        if len(df) > 1:
            if signal == 1:
                target = df['High'].iloc[-2]
            else:
                target = df['Low'].iloc[-2]
        else:
            target = entry_price + target_points if signal == 1 else entry_price - target_points
    
    elif target_type == 'Current Swing Low/High':
        swing_high, swing_low = detect_swing_high_low(df)
        if signal == 1:
            target = swing_high.iloc[-1] if not pd.isna(swing_high.iloc[-1]) else entry_price + target_points
        else:
            target = swing_low.iloc[-1] if not pd.isna(swing_low.iloc[-1]) else entry_price - target_points
    
    elif target_type == 'Previous Swing Low/High':
        swing_high, swing_low = detect_swing_high_low(df)
        if len(df) > 1:
            if signal == 1:
                target = swing_high.iloc[-2] if not pd.isna(swing_high.iloc[-2]) else entry_price + target_points
            else:
                target = swing_low.iloc[-2] if not pd.isna(swing_low.iloc[-2]) else entry_price - target_points
        else:
            target = entry_price + target_points if signal == 1 else entry_price - target_points
    
    elif target_type == 'Risk-Reward Based':
        if position and position.get('sl') and position['sl'] != 0:
            sl_distance = abs(entry_price - position['sl'])
            if signal == 1:
                target = entry_price + (sl_distance * rr_ratio)
            else:
                target = entry_price - (sl_distance * rr_ratio)
        else:
            if signal == 1:
                target = entry_price + target_points
            else:
                target = entry_price - target_points
    
    elif 'Signal-based' in target_type:
        target = 0
    
    else:
        if signal == 1:
            target = entry_price + target_points
        else:
            target = entry_price - target_points
    
    if target != 0 and target is not None:
        if signal == 1:
            target = max(target, entry_price + min_target_distance)
        else:
            target = min(target, entry_price - min_target_distance)
    
    return target

def check_reverse_signal(df, signal, i):
    """Check for reverse EMA crossover signal"""
    if 'EMA_Fast' not in df.columns or 'EMA_Slow' not in df.columns:
        return False
    
    if i < 1:
        return False
    
    if signal == 1:
        if (df['EMA_Fast'].iloc[i] < df['EMA_Slow'].iloc[i] and 
            df['EMA_Fast'].iloc[i-1] >= df['EMA_Slow'].iloc[i-1]):
            return True
    elif signal == -1:
        if (df['EMA_Fast'].iloc[i] > df['EMA_Slow'].iloc[i] and 
            df['EMA_Fast'].iloc[i-1] <= df['EMA_Slow'].iloc[i-1]):
            return True
    
    return False

# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def run_backtest(df, strategy_func, config):
    """Run backtest on historical data"""
    df = strategy_func(df.copy(), config)
    
    trades = []
    position = None
    
    sl_type = config.get('sl_type', 'Custom Points')
    target_type = config.get('target_type', 'Custom Points')
    quantity = config.get('quantity', 1)
    
    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        current_time = df.index[i]
        
        if position is None:
            signal = df['Signal'].iloc[i]
            
            if signal != 0:
                entry_price = current_price
                entry_time = current_time
                
                temp_df = df.iloc[:i+1].copy()
                sl = calculate_stop_loss(temp_df, entry_price, signal, sl_type, config)
                target = calculate_target(temp_df, entry_price, signal, target_type, config)
                
                position = {
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'signal': signal,
                    'sl': sl,
                    'target': target,
                    'highest': entry_price,
                    'lowest': entry_price,
                    'partial_exit_done': False,
                    'breakeven_activated': False
                }
        
        else:
            exit_price = None
            exit_reason = None
            
            position['highest'] = max(position['highest'], df['High'].iloc[i])
            position['lowest'] = min(position['lowest'], df['Low'].iloc[i])
            
            temp_df = df.iloc[:i+1].copy()
            
            if 'Signal-based' in sl_type or 'Signal-based' in target_type:
                if check_reverse_signal(df, position['signal'], i):
                    exit_price = current_price
                    exit_reason = 'Reverse Signal'
            
            if exit_price is None:
                if 'Trailing' in sl_type and 'Signal-based' not in sl_type:
                    new_sl = calculate_stop_loss(temp_df, position['entry_price'], position['signal'], sl_type, config, position)
                    if new_sl is not None:
                        position['sl'] = new_sl
                
                if position['signal'] == 1:
                    if position.get('sl') and position['sl'] > 0 and current_price <= position['sl']:
                        exit_price = position['sl']
                        exit_reason = 'SL Hit'
                else:
                    if position.get('sl') and position['sl'] > 0 and current_price >= position['sl']:
                        exit_price = position['sl']
                        exit_reason = 'SL Hit'
            
            if exit_price is None and target_type == '50% Exit at Target (Partial)':
                if not position.get('partial_exit_done', False):
                    if position['signal'] == 1 and position.get('target') and position['target'] > 0:
                        if current_price >= position['target']:
                            position['partial_exit_done'] = True
                    elif position['signal'] == -1 and position.get('target') and position['target'] > 0:
                        if current_price <= position['target']:
                            position['partial_exit_done'] = True
            
            if exit_price is None and 'Break-even' in sl_type:
                if not position.get('breakeven_activated', False) and position.get('target') and position['target'] > 0:
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
            
            if exit_price is None and position.get('target') and position['target'] > 0 and 'Trailing Target' not in target_type and 'Signal-based' not in target_type:
                if position['signal'] == 1:
                    if current_price >= position['target']:
                        exit_price = position['target']
                        exit_reason = 'Target Hit'
                else:
                    if current_price <= position['target']:
                        exit_price = position['target']
                        exit_reason = 'Target Hit'
            
            if exit_price is not None:
                if position['signal'] == 1:
                    pnl = (exit_price - position['entry_price']) * quantity
                else:
                    pnl = (position['entry_price'] - exit_price) * quantity
                
                duration = (current_time - position['entry_time']).total_seconds() / 3600
                
                trade_record = {
                    'Entry Time': position['entry_time'],
                    'Exit Time': current_time,
                    'Duration (hrs)': duration,
                    'Signal': 'LONG' if position['signal'] == 1 else 'SHORT',
                    'Entry Price': position['entry_price'],
                    'Exit Price': exit_price,
                    'SL': position['sl'],
                    'Target': position['target'],
                    'Exit Reason': exit_reason,
                    'P&L': pnl,
                    'Highest': position['highest'],
                    'Lowest': position['lowest'],
                    'Range': position['highest'] - position['lowest']
                }
                
                trades.append(trade_record)
                position = None
    
    if len(trades) == 0:
        return {
            'trades': [],
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'accuracy': 0,
            'total_pnl': 0,
            'avg_duration': 0
        }
    
    trades_df = pd.DataFrame(trades)
    winning_trades = len(trades_df[trades_df['P&L'] > 0])
    losing_trades = len(trades_df[trades_df['P&L'] <= 0])
    accuracy = (winning_trades / len(trades_df)) * 100 if len(trades_df) > 0 else 0
    total_pnl = trades_df['P&L'].sum()
    avg_duration = trades_df['Duration (hrs)'].mean()
    
    return {
        'trades': trades,
        'total_trades': len(trades_df),
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'accuracy': accuracy,
        'total_pnl': total_pnl,
        'avg_duration': avg_duration
    }

# ============================================================================
# LIVE TRADING FUNCTIONS
# ============================================================================

def execute_live_trading(ticker, interval, period, strategy_func, config, mode):
    """Execute live trading loop"""
    while st.session_state.get('trading_active', False):
        df = fetch_data(ticker, interval, period, mode)
        
        if df is None or len(df) < 50:
            add_log("Failed to fetch data or insufficient data")
            time.sleep(2)
            continue
        
        df = strategy_func(df, config)
        st.session_state['current_data'] = df
        
        current_price = df['Close'].iloc[-1]
        current_signal = df['Signal'].iloc[-1]
        current_time = df.index[-1]
        
        position = st.session_state.get('position')
        quantity = config.get('quantity', 1)
        sl_type = config.get('sl_type', 'Custom Points')
        target_type = config.get('target_type', 'Custom Points')
        
        if position is None:
            if current_signal != 0:
                entry_price = current_price
                sl = calculate_stop_loss(df, entry_price, current_signal, sl_type, config)
                target = calculate_target(df, entry_price, current_signal, target_type, config)
                
                st.session_state['position'] = {
                    'entry_time': current_time,
                    'entry_price': entry_price,
                    'signal': current_signal,
                    'sl': sl,
                    'target': target,
                    'highest': entry_price,
                    'lowest': entry_price
                }
                
                st.session_state['highest_price'] = entry_price
                st.session_state['lowest_price'] = entry_price
                
                signal_text = "LONG" if current_signal == 1 else "SHORT"
                sl_text = f"{sl:.2f}" if sl > 0 else "Signal Based"
                target_text = f"{target:.2f}" if target > 0 else "Signal Based"
                add_log(f"Entered {signal_text} at {entry_price:.2f} | SL: {sl_text} | Target: {target_text}")
        
        else:
            exit_price = None
            exit_reason = None
            
            highest = max(st.session_state.get('highest_price', position['entry_price']), df['High'].iloc[-1])
            lowest = min(st.session_state.get('lowest_price', position['entry_price']), df['Low'].iloc[-1])
            st.session_state['highest_price'] = highest
            st.session_state['lowest_price'] = lowest
            
            position['highest'] = highest
            position['lowest'] = lowest
            
            if 'Signal-based' in sl_type or 'Signal-based' in target_type:
                if check_reverse_signal(df, position['signal'], len(df)-1):
                    exit_price = current_price
                    exit_reason = 'Reverse Signal'
            
            if exit_price is None:
                if 'Trailing' in sl_type and 'Signal-based' not in sl_type:
                    new_sl = calculate_stop_loss(df, position['entry_price'], position['signal'], sl_type, config, position)
                    position['sl'] = new_sl
                    st.session_state['position'] = position
                
                if position['signal'] == 1:
                    if position['sl'] > 0 and current_price <= position['sl']:
                        exit_price = position['sl']
                        exit_reason = 'SL Hit'
                else:
                    if position['sl'] > 0 and current_price >= position['sl']:
                        exit_price = position['sl']
                        exit_reason = 'SL Hit'
            
            if exit_price is None and target_type == '50% Exit at Target (Partial)':
                if not st.session_state.get('partial_exit_done', False):
                    if position['signal'] == 1 and position.get('target') and position['target'] > 0:
                        if current_price >= position['target']:
                            st.session_state['partial_exit_done'] = True
                            add_log("50% position exited at target - trailing remaining")
                    elif position['signal'] == -1 and position.get('target') and position['target'] > 0:
                        if current_price <= position['target']:
                            st.session_state['partial_exit_done'] = True
                            add_log("50% position exited at target - trailing remaining")
            
            if exit_price is None and 'Break-even' in sl_type:
                if not st.session_state.get('breakeven_activated', False) and position.get('target') and position['target'] > 0:
                    if position['signal'] == 1:
                        profit = current_price - position['entry_price']
                        target_dist = position['target'] - position['entry_price']
                        if profit >= target_dist * 0.5:
                            position['sl'] = position['entry_price']
                            st.session_state['breakeven_activated'] = True
                            st.session_state['position'] = position
                            add_log("SL moved to break-even")
                    else:
                        profit = position['entry_price'] - current_price
                        target_dist = position['entry_price'] - position['target']
                        if profit >= target_dist * 0.5:
                            position['sl'] = position['entry_price']
                            st.session_state['breakeven_activated'] = True
                            st.session_state['position'] = position
                            add_log("SL moved to break-even")
            
            if exit_price is None and position.get('target') and position['target'] > 0 and 'Trailing Target' not in target_type and 'Signal-based' not in target_type:
                if position['signal'] == 1:
                    if current_price >= position['target']:
                        exit_price = position['target']
                        exit_reason = 'Target Hit'
                else:
                    if current_price <= position['target']:
                        exit_price = position['target']
                        exit_reason = 'Target Hit'
            
            if exit_price is not None:
                if position['signal'] == 1:
                    pnl = (exit_price - position['entry_price']) * quantity
                else:
                    pnl = (position['entry_price'] - exit_price) * quantity
                
                duration = (current_time - position['entry_time']).total_seconds() / 3600
                
                trade_record = {
                    'Entry Time': position['entry_time'],
                    'Exit Time': current_time,
                    'Duration (hrs)': duration,
                    'Signal': 'LONG' if position['signal'] == 1 else 'SHORT',
                    'Entry Price': position['entry_price'],
                    'Exit Price': exit_price,
                    'SL': position['sl'],
                    'Target': position['target'],
                    'Exit Reason': exit_reason,
                    'P&L': pnl,
                    'Highest': position['highest'],
                    'Lowest': position['lowest'],
                    'Range': position['highest'] - position['lowest']
                }
                
                st.session_state['trade_history'].append(trade_record)
                st.session_state['trade_history'] = st.session_state['trade_history']
                
                add_log(f"Exited at {exit_price:.2f} | Reason: {exit_reason} | P&L: {pnl:.2f}")
                
                reset_position_state()
        
        if mode == "Live Trading":
            time.sleep(random.uniform(1.0, 1.5))
        else:
            time.sleep(0.1)

# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(page_title="Quantitative Trading System", layout="wide")
    
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
    
    st.title("📊 Professional Quantitative Trading System")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        mode = st.selectbox("Mode", ["Backtest", "Live Trading"], key="mode_select")
        
        st.subheader("Asset Selection")
        asset_type = st.selectbox("Asset Type", 
            ["Indian Indices", "Crypto", "Forex", "Commodities", "Custom Ticker"],
            key="asset_type_select")
        
        if asset_type == "Indian Indices":
            ticker = st.selectbox("Index", ["^NSEI", "^NSEBANK", "^BSESN"], 
                                format_func=lambda x: {"^NSEI": "NIFTY 50", "^NSEBANK": "BANKNIFTY", "^BSESN": "SENSEX"}[x],
                                key="index_select")
        elif asset_type == "Crypto":
            ticker = st.selectbox("Crypto", ["BTC-USD", "ETH-USD"], key="crypto_select")
        elif asset_type == "Forex":
            ticker = st.selectbox("Forex", ["USDINR=X", "EURUSD=X", "GBPUSD=X"], key="forex_select")
        elif asset_type == "Commodities":
            ticker = st.selectbox("Commodity", ["GC=F", "SI=F"], 
                                format_func=lambda x: {"GC=F": "Gold", "SI=F": "Silver"}[x],
                                key="commodity_select")
        else:
            ticker = st.text_input("Custom Ticker", "AAPL", key="custom_ticker")
        
        st.subheader("Timeframe")
        interval = st.selectbox("Interval", 
            ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"],
            key="interval_select")
        
        if interval in ['1m']:
            period_options = ['1d', '5d']
        elif interval == '5m':
            period_options = ['1d', '1mo']
        elif interval in ['15m', '30m', '1h', '4h']:
            period_options = ['1mo']
        elif interval == '1d':
            period_options = ['1mo', '1y', '2y', '5y']
        elif interval == '1wk':
            period_options = ['1mo', '1y', '5y', '10y', '15y', '20y']
        else:
            period_options = ['1y', '2y', '5y', '10y', '15y', '20y', '25y', '30y']
        
        period = st.selectbox("Period", period_options, key="period_select")
        
        quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="quantity_input")
        
        st.subheader("Strategy Selection")
        strategy_name = st.selectbox("Strategy", [
            "EMA Crossover",
            "Simple Buy",
            "Simple Sell",
            "Price Threshold",
            "RSI-ADX-EMA",
            "Percentage Change",
            "AI Price Action",
            "Custom Builder"
        ], key="strategy_select")
        
        config = {'quantity': quantity}
        
        if strategy_name == "EMA Crossover":
            config['ema_fast'] = st.number_input("EMA Fast", min_value=2, value=9, key="ema_fast")
            config['ema_slow'] = st.number_input("EMA Slow", min_value=2, value=15, key="ema_slow")
            config['min_angle'] = st.number_input("Min Crossover Angle (°)", min_value=0.0, value=1.0, step=0.1, key="min_angle")
            config['entry_filter'] = st.selectbox("Entry Filter", 
                ["Simple Crossover", "Custom Candle (Points)", "ATR-based Candle"],
                key="entry_filter")
            
            if config['entry_filter'] == "Custom Candle (Points)":
                config['custom_points'] = st.number_input("Custom Points", min_value=1.0, value=10.0, key="custom_points")
            elif config['entry_filter'] == "ATR-based Candle":
                config['atr_multiplier'] = st.number_input("ATR Multiplier", min_value=0.1, value=1.5, step=0.1, key="atr_mult_entry")
            
            config['use_adx'] = st.checkbox("Use ADX Filter", key="use_adx")
            if config['use_adx']:
                config['adx_period'] = st.number_input("ADX Period", min_value=2, value=14, key="adx_period")
                config['adx_threshold'] = st.number_input("ADX Threshold", min_value=1, value=25, key="adx_threshold")
            
            strategy_func = ema_crossover_strategy
        
        elif strategy_name == "Simple Buy":
            strategy_func = simple_buy_strategy
        
        elif strategy_name == "Simple Sell":
            strategy_func = simple_sell_strategy
        
        elif strategy_name == "Price Threshold":
            config['threshold'] = st.number_input("Price Threshold", value=0.0, key="price_threshold")
            config['direction'] = st.selectbox("Direction", [
                "LONG (Price >= Threshold)",
                "SHORT (Price >= Threshold)",
                "LONG (Price <= Threshold)",
                "SHORT (Price <= Threshold)"
            ], key="threshold_direction")
            strategy_func = price_threshold_strategy
        
        elif strategy_name == "RSI-ADX-EMA":
            config['rsi_period'] = st.number_input("RSI Period", min_value=2, value=14, key="rsi_period_rae")
            config['adx_period'] = st.number_input("ADX Period", min_value=2, value=14, key="adx_period_rae")
            config['ema1_period'] = st.number_input("EMA1 Period", min_value=2, value=9, key="ema1_rae")
            config['ema2_period'] = st.number_input("EMA2 Period", min_value=2, value=15, key="ema2_rae")
            strategy_func = rsi_adx_ema_strategy
        
        elif strategy_name == "Percentage Change":
            config['pct_threshold'] = st.number_input("% Threshold", min_value=0.001, value=0.01, step=0.001, format="%.3f", key="pct_threshold")
            config['direction'] = st.selectbox("Direction", [
                "BUY on Fall",
                "SELL on Fall",
                "BUY on Rise",
                "SELL on Rise"
            ], key="pct_direction")
            strategy_func = percentage_change_strategy
        
        elif strategy_name == "AI Price Action":
            st.info("AI will analyze multiple indicators automatically")
            strategy_func = ai_price_action_strategy
        
        elif strategy_name == "Custom Builder":
            st.subheader("Custom Conditions")
            
            num_conditions = st.number_input("Number of Conditions", min_value=1, max_value=10, value=len(st.session_state.get('custom_conditions', [])) or 1, key="num_conditions")
            
            conditions = []
            for idx in range(int(num_conditions)):
                st.markdown(f"**Condition {idx+1}**")
                use_cond = st.checkbox(f"Use Condition {idx+1}", value=True, key=f"use_cond_{idx}")
                
                col1, col2 = st.columns(2)
                with col1:
                    compare_price = st.checkbox(f"Compare with Indicator", key=f"compare_price_{idx}")
                
                indicator = st.selectbox(f"Indicator", [
                    "Price", "RSI", "ADX", "EMA_Fast", "EMA_Slow", "SuperTrend",
                    "EMA_20", "EMA_50", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower",
                    "ATR", "Volume", "VWAP", "Keltner_Upper", "Keltner_Lower",
                    "Close", "High", "Low", "Support", "Resistance"
                ], key=f"indicator_{idx}")
                
                operator = st.selectbox(f"Operator", [
                    ">", "<", ">=", "<=", "==", "crosses_above", "crosses_below"
                ], key=f"operator_{idx}")
                
                if compare_price:
                    compare_indicator = st.selectbox(f"Compare Indicator", [
                        "EMA_20", "EMA_50", "EMA_Fast", "EMA_Slow", "Close", "High", "Low"
                    ], key=f"compare_indicator_{idx}")
                    value = 0
                else:
                    value = st.number_input(f"Value", value=0.0, key=f"value_{idx}")
                    compare_indicator = None
                
                action = st.selectbox(f"Action", ["BUY", "SELL"], key=f"action_{idx}")
                
                conditions.append({
                    'use': use_cond,
                    'indicator': indicator,
                    'operator': operator,
                    'value': value,
                    'action': action,
                    'compare_price': compare_price,
                    'compare_indicator': compare_indicator
                })
            
            config['conditions'] = conditions
            config['ema_fast'] = st.number_input("EMA Fast (for display)", min_value=2, value=9, key="custom_ema_fast")
            config['ema_slow'] = st.number_input("EMA Slow (for display)", min_value=2, value=15, key="custom_ema_slow")
            strategy_func = custom_strategy_builder
        
        st.subheader("Stop Loss Settings")
        sl_type = st.selectbox("SL Type", [
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
        ], key="sl_type_select")
        
        config['sl_type'] = sl_type
        
        if 'Points' in sl_type or sl_type == 'Custom Points':
            config['sl_points'] = st.number_input("SL Points", min_value=1.0, value=10.0, key="sl_points")
        
        if 'ATR' in sl_type:
            config['atr_multiplier'] = st.number_input("ATR Multiplier", min_value=0.1, value=1.5, step=0.1, key="sl_atr_mult")
        
        config['min_sl_distance'] = st.number_input("Min SL Distance (points)", min_value=1.0, value=10.0, key="min_sl_dist")
        
        st.subheader("Target Settings")
        target_type = st.selectbox("Target Type", [
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
        ], key="target_type_select")
        
        config['target_type'] = target_type
        
        if 'Points' in target_type or target_type == 'Custom Points':
            config['target_points'] = st.number_input("Target Points", min_value=1.0, value=20.0, key="target_points")
        
        if 'ATR' in target_type:
            config['target_atr_mult'] = st.number_input("Target ATR Multiplier", min_value=0.1, value=2.0, step=0.1, key="target_atr_mult")
        
        if target_type == 'Risk-Reward Based':
            config['rr_ratio'] = st.number_input("Risk-Reward Ratio", min_value=0.1, value=2.0, step=0.1, key="rr_ratio")
        
        config['min_target_distance'] = st.number_input("Min Target Distance (points)", min_value=1.0, value=15.0, key="min_target_dist")
        
        st.subheader("Dhan Brokerage Integration")
        use_dhan = st.checkbox("Enable Dhan Brokerage", key="use_dhan")
        
        if use_dhan:
            st.warning("⚠️ Dhan integration is a placeholder. Implement actual API calls.")
            dhan_client_id = st.text_input("Client ID", key="dhan_client_id")
            dhan_token = st.text_input("Access Token", type="password", key="dhan_token")
            dhan_strike_price = st.number_input("Strike Price", min_value=0.0, value=0.0, key="dhan_strike")
            dhan_option_type = st.selectbox("Option Type", ["CE", "PE"], key="dhan_option_type")
            dhan_expiry = st.date_input("Expiry Date", key="dhan_expiry")
            dhan_lots = st.number_input("Lots/Quantities", min_value=1, value=1, key="dhan_lots")
            
            config['dhan_config'] = {
                'client_id': dhan_client_id,
                'token': dhan_token,
                'strike_price': dhan_strike_price,
                'option_type': dhan_option_type,
                'expiry': dhan_expiry,
                'lots': dhan_lots
            }
    
    tabs = st.tabs(["📈 Live Trading Dashboard", "📊 Trade History", "📝 Trade Logs", "🔬 Backtest Results"])
    
    with tabs[0]:
        st.header("Live Trading Dashboard")
        
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            if st.button("▶️ Start Trading", type="primary", key="start_trading"):
                if not st.session_state.get('trading_active', False):
                    st.session_state['trading_active'] = True
                    add_log("Trading started")
                    st.rerun()
        
        with col2:
            if st.button("⏹️ Stop Trading", key="stop_trading"):
                if st.session_state.get('trading_active', False):
                    if st.session_state.get('position'):
                        position = st.session_state['position']
                        current_price = st.session_state['current_data']['Close'].iloc[-1] if st.session_state['current_data'] is not None else position['entry_price']
                        current_time = datetime.now(pytz.timezone('Asia/Kolkata'))
                        
                        if position['signal'] == 1:
                            pnl = (current_price - position['entry_price']) * quantity
                        else:
                            pnl = (position['entry_price'] - current_price) * quantity
                        
                        duration = (current_time - position['entry_time']).total_seconds() / 3600
                        
                        trade_record = {
                            'Entry Time': position['entry_time'],
                            'Exit Time': current_time,
                            'Duration (hrs)': duration,
                            'Signal': 'LONG' if position['signal'] == 1 else 'SHORT',
                            'Entry Price': position['entry_price'],
                            'Exit Price': current_price,
                            'SL': position['sl'],
                            'Target': position['target'],
                            'Exit Reason': 'Manual Close',
                            'P&L': pnl,
                            'Highest': position.get('highest', position['entry_price']),
                            'Lowest': position.get('lowest', position['entry_price']),
                            'Range': position.get('highest', position['entry_price']) - position.get('lowest', position['entry_price'])
                        }
                        
                        st.session_state['trade_history'].append(trade_record)
                        st.session_state['trade_history'] = st.session_state['trade_history']
                        add_log(f"Position manually closed at {current_price:.2f} | P&L: {pnl:.2f}")
                    
                    st.session_state['trading_active'] = False
                    reset_position_state()
                    add_log("Trading stopped")
                    st.rerun()
        
        with col3:
            if st.button("🔄 Manual Refresh", key="manual_refresh"):
                st.rerun()
        
        if st.session_state.get('trading_active', False):
            st.success("✅ Trading is ACTIVE")
        else:
            st.info("⚪ Trading is STOPPED")
        
        st.divider()
        
        st.subheader("Active Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Asset", ticker)
            st.metric("Interval", interval)
        with col2:
            st.metric("Period", period)
            st.metric("Quantity", quantity)
        with col3:
            st.metric("Strategy", strategy_name)
            st.metric("SL Type", sl_type)
        with col4:
            st.metric("Target Type", target_type)
            st.metric("Mode", mode)
        
        if strategy_name == "EMA Crossover":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("EMA Fast", config.get('ema_fast', 9))
            with col2:
                st.metric("EMA Slow", config.get('ema_slow', 15))
            with col3:
                st.metric("Min Angle", f"{config.get('min_angle', 1)}°")
            
            st.write(f"**Entry Filter:** {config.get('entry_filter', 'Simple Crossover')}")
            if config.get('use_adx', False):
                st.write(f"**ADX Filter:** Enabled (Threshold: {config.get('adx_threshold', 25)})")
        
        st.divider()
        
        if st.session_state.get('trading_active', False):
            execute_live_trading(ticker, interval, period, strategy_func, config, mode)
        
        # Always show live metrics and chart when data is available
        current_df = st.session_state.get('current_data')
        
        # Fetch initial data if not available
        if current_df is None and not st.session_state.get('trading_active', False):
            with st.spinner("Fetching initial data..."):
                current_df = fetch_data(ticker, interval, period, mode)
                if current_df is not None:
                    current_df = strategy_func(current_df, config)
                    st.session_state['current_data'] = current_df
        
        if st.session_state.get('current_data') is not None:
            df = st.session_state['current_data']
            
            st.subheader("Live Metrics")
            
            current_price = df['Close'].iloc[-1]
            current_time = df.index[-1]
            position = st.session_state.get('position')
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"{current_price:.2f}")
                st.metric("Last Update", current_time.strftime("%H:%M:%S"))
            
            with col2:
                if position:
                    st.metric("Entry Price", f"{position['entry_price']:.2f}")
                    st.metric("Position", "LONG" if position['signal'] == 1 else "SHORT")
                else:
                    st.metric("Entry Price", "N/A")
                    st.metric("Position", "None")
            
            with col3:
                if position:
                    if position['signal'] == 1:
                        pnl = (current_price - position['entry_price']) * quantity
                    else:
                        pnl = (position['entry_price'] - current_price) * quantity
                    
                    if pnl >= 0:
                        st.metric("Unrealized P&L", f"{pnl:.2f}", delta=f"+{pnl:.2f}")
                    else:
                        st.metric("Unrealized P&L", f"{pnl:.2f}", delta=f"{pnl:.2f}", delta_color="inverse")
                    
                    duration = (current_time - position['entry_time']).total_seconds() / 3600
                    st.metric("Duration", f"{duration:.2f} hrs")
                else:
                    st.metric("Unrealized P&L", "N/A")
                    st.metric("Duration", "N/A")
            
            with col4:
                if 'EMA_Fast' in df.columns:
                    st.metric("EMA Fast", f"{df['EMA_Fast'].iloc[-1]:.2f}")
                if 'EMA_Slow' in df.columns:
                    st.metric("EMA Slow", f"{df['EMA_Slow'].iloc[-1]:.2f}")
            
            if 'RSI' in df.columns:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
                if 'ADX' in df.columns:
                    with col2:
                        st.metric("ADX", f"{df['ADX'].iloc[-1]:.2f}")
                if 'EMA_Angle' in df.columns:
                    with col3:
                        st.metric("EMA Angle", f"{df['EMA_Angle'].iloc[-1]:.2f}°")
            
            current_signal = df['Signal'].iloc[-1]
            if current_signal == 1:
                st.success("🟢 Current Signal: BUY")
            elif current_signal == -1:
                st.error("🔴 Current Signal: SELL")
            else:
                st.info("⚪ Current Signal: NONE")
            
            if position:
                st.subheader("Position Details")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    sl_text = f"{position['sl']:.2f}" if position['sl'] > 0 else "Signal Based"
                    st.metric("Stop Loss", sl_text)
                    if position['sl'] > 0:
                        if position['signal'] == 1:
                            dist_sl = current_price - position['sl']
                        else:
                            dist_sl = position['sl'] - current_price
                        st.metric("Distance to SL", f"{dist_sl:.2f} pts")
                
                with col2:
                    target_text = f"{position['target']:.2f}" if position['target'] > 0 else "Signal Based"
                    st.metric("Target", target_text)
                    if position['target'] > 0:
                        if position['signal'] == 1:
                            dist_target = position['target'] - current_price
                        else:
                            dist_target = current_price - position['target']
                        st.metric("Distance to Target", f"{dist_target:.2f} pts")
                
                with col3:
                    st.metric("Highest", f"{position.get('highest', position['entry_price']):.2f}")
                    st.metric("Lowest", f"{position.get('lowest', position['entry_price']):.2f}")
                
                if st.session_state.get('partial_exit_done', False):
                    st.warning("ℹ️ 50% position already exited - Trailing remaining")
                
                if st.session_state.get('breakeven_activated', False):
                    st.info("ℹ️ Stop Loss moved to break-even")
            
            st.divider()
            
            st.subheader("Live Chart")
            
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
                                        line=dict(color='orange', width=1)))
            
            if position:
                fig.add_hline(y=position['entry_price'], line_dash="dash", 
                            line_color="white", annotation_text="Entry")
                
                if position.get('sl') and position['sl'] > 0:
                    fig.add_hline(y=position['sl'], line_dash="dash", 
                                line_color="red", annotation_text="SL")
                
                if position.get('target') and position['target'] > 0:
                    fig.add_hline(y=position['target'], line_dash="dash", 
                                line_color="green", annotation_text="Target")
            
            fig.update_layout(
                title=f"{ticker} - {interval}",
                xaxis_title="Time",
                yaxis_title="Price",
                height=600,
                xaxis_rangeslider_visible=False
            )
            
            chart_key = f"live_chart_{int(time.time())}"
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
            
            if position:
                if position['signal'] == 1:
                    if current_price > position['entry_price']:
                        st.success("✅ In Profit - HOLD")
                    else:
                        st.warning("⚠️ In Loss - Monitor SL")
                else:
                    if current_price < position['entry_price']:
                        st.success("✅ In Profit - HOLD")
                    else:
                        st.warning("⚠️ In Loss - Monitor SL")
            else:
                st.info("ℹ️ No active position - Monitoring for signals")
        
        else:
            st.info("Start trading to see live metrics and charts")
    
    with tabs[1]:
        st.markdown("### 📈 Trade History")
        
        if len(st.session_state['trade_history']) == 0:
            st.info("No trades executed yet")
        else:
            trades_df = pd.DataFrame(st.session_state['trade_history'])
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['P&L'] > 0])
            losing_trades = len(trades_df[trades_df['P&L'] <= 0])
            accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = trades_df['P&L'].sum()
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Trades", total_trades)
            with col2:
                st.metric("Winning", winning_trades)
            with col3:
                st.metric("Losing", losing_trades)
            with col4:
                st.metric("Accuracy", f"{accuracy:.1f}%")
            with col5:
                if total_pnl >= 0:
                    st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"+{total_pnl:.2f}")
                else:
                    st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color="inverse")
            
            st.divider()
            
            for idx, trade in enumerate(reversed(st.session_state['trade_history'])):
                with st.expander(f"Trade {len(st.session_state['trade_history']) - idx}: {trade.get('Signal', 'N/A')} | P&L: {trade.get('P&L', 0):.2f}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Entry Time:** {trade.get('Entry Time', 'N/A')}")
                        st.write(f"**Exit Time:** {trade.get('Exit Time', 'N/A')}")
                        st.write(f"**Duration:** {trade.get('Duration (hrs)', 0):.2f} hours")
                        st.write(f"**Signal:** {trade.get('Signal', 'N/A')}")
                        st.write(f"**Entry Price:** {trade.get('Entry Price', 0):.2f}")
                    
                    with col2:
                        st.write(f"**Exit Price:** {trade.get('Exit Price', 0):.2f}")
                        sl_val = trade.get('SL', 0)
                        sl_text = f"{sl_val:.2f}" if sl_val > 0 else "Signal Based"
                        st.write(f"**Stop Loss:** {sl_text}")
                        
                        target_val = trade.get('Target', 0)
                        target_text = f"{target_val:.2f}" if target_val > 0 else "Signal Based"
                        st.write(f"**Target:** {target_text}")
                        
                        st.write(f"**Exit Reason:** {trade.get('Exit Reason', 'N/A')}")
                        
                        pnl = trade.get('P&L', 0)
                        pnl_color = "green" if pnl > 0 else "red"
                        st.markdown(f"**P&L:** <span style='color:{pnl_color}'>{pnl:.2f}</span>", unsafe_allow_html=True)
                    
                    st.write(f"**Highest Price:** {trade.get('Highest', 0):.2f}")
                    st.write(f"**Lowest Price:** {trade.get('Lowest', 0):.2f}")
                    st.write(f"**Range:** {trade.get('Range', 0):.2f}")
    
    with tabs[2]:
        st.markdown("### 📝 Trade Logs")
        
        if len(st.session_state['trade_logs']) == 0:
            st.info("No logs yet")
        else:
            for log in reversed(st.session_state['trade_logs']):
                st.text(log)
    
    with tabs[3]:
        st.markdown("### 🔬 Backtest Results")
        
        if mode != "Backtest":
            st.warning("Switch to Backtest mode to run backtests")
        else:
            if st.button("▶️ Run Backtest", type="primary", key="run_backtest"):
                with st.spinner("Running backtest..."):
                    st.session_state['current_data'] = None
                    
                    df = fetch_data(ticker, interval, period, mode)
                    
                    if df is None or len(df) < 50:
                        st.error("Failed to fetch data or insufficient data for backtest")
                    else:
                        results = run_backtest(df, strategy_func, config)
                        
                        st.success("Backtest completed!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Trades", results['total_trades'])
                        with col2:
                            st.metric("Winning Trades", results['winning_trades'])
                        with col3:
                            st.metric("Losing Trades", results['losing_trades'])
                        with col4:
                            st.metric("Accuracy", f"{results['accuracy']:.1f}%")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            total_pnl = results['total_pnl']
                            if total_pnl >= 0:
                                st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"+{total_pnl:.2f}")
                            else:
                                st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color="inverse")
                        
                        with col2:
                            st.metric("Avg Duration", f"{results['avg_duration']:.2f} hrs")
                        
                        st.divider()
                        
                        if len(results['trades']) > 0:
                            st.subheader("All Trades")
                            
                            for idx, trade in enumerate(results['trades']):
                                with st.expander(f"Trade {idx+1}: {trade['Signal']} | P&L: {trade['P&L']:.2f}"):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write(f"**Entry Time:** {trade['Entry Time']}")
                                        st.write(f"**Exit Time:** {trade['Exit Time']}")
                                        st.write(f"**Duration:** {trade['Duration (hrs)']:.2f} hours")
                                        st.write(f"**Signal:** {trade['Signal']}")
                                        st.write(f"**Entry Price:** {trade['Entry Price']:.2f}")
                                    
                                    with col2:
                                        st.write(f"**Exit Price:** {trade['Exit Price']:.2f}")
                                        
                                        sl_val = trade['SL']
                                        sl_text = f"{sl_val:.2f}" if sl_val > 0 else "Signal Based"
                                        st.write(f"**Stop Loss:** {sl_text}")
                                        
                                        target_val = trade['Target']
                                        target_text = f"{target_val:.2f}" if target_val > 0 else "Signal Based"
                                        st.write(f"**Target:** {target_text}")
                                        
                                        st.write(f"**Exit Reason:** {trade['Exit Reason']}")
                                        
                                        pnl = trade['P&L']
                                        pnl_color = "green" if pnl > 0 else "red"
                                        st.markdown(f"**P&L:** <span style='color:{pnl_color}'>{pnl:.2f}</span>", unsafe_allow_html=True)
                                    
                                    st.write(f"**Highest Price:** {trade['Highest']:.2f}")
                                    st.write(f"**Lowest Price:** {trade['Lowest']:.2f}")
                                    st.write(f"**Range:** {trade['Range']:.2f}")

if __name__ == "__main__":
    main()
