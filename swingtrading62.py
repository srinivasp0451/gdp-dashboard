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
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in data.columns.values]
            cols_to_keep = []
            for base in ['Open', 'High', 'Low', 'Close', 'Volume']:
                matching = [c for c in data.columns if base in c]
                if matching:
                    data[base] = data[matching[0]]
                    cols_to_keep.append(base)
            data = data[cols_to_keep]
        
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

def calculate_sma(data, period):
    """Calculate SMA"""
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

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_stochastic_rsi(data, period=14, smooth_k=3, smooth_d=3):
    """Calculate Stochastic RSI"""
    rsi = calculate_rsi(data, period)
    stoch_rsi = (rsi - rsi.rolling(window=period).min()) / (rsi.rolling(window=period).max() - rsi.rolling(window=period).min())
    k = stoch_rsi.rolling(window=smooth_k).mean() * 100
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_keltner_channel(df, period=20, atr_mult=2):
    """Calculate Keltner Channel"""
    ema = calculate_ema(df['Close'], period)
    atr = calculate_atr(df, period)
    upper = ema + (atr * atr_mult)
    lower = ema - (atr * atr_mult)
    return upper, ema, lower

def calculate_pivot_points(df):
    """Calculate Pivot Points"""
    if len(df) < 2:
        return None, None, None, None, None
    
    prev_high = df['High'].iloc[-2]
    prev_low = df['Low'].iloc[-2]
    prev_close = df['Close'].iloc[-2]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    return pivot, r1, r2, s1, s2

def calculate_support_resistance(df, lookback=20):
    """Calculate Support and Resistance levels"""
    highs = df['High'].rolling(window=lookback).max()
    lows = df['Low'].rolling(window=lookback).min()
    return lows.iloc[-1], highs.iloc[-1]

def detect_engulfing_pattern(df, i):
    """Detect Bullish/Bearish Engulfing Pattern"""
    if i < 1:
        return 0
    
    curr_open = df['Open'].iloc[i]
    curr_close = df['Close'].iloc[i]
    prev_open = df['Open'].iloc[i-1]
    prev_close = df['Close'].iloc[i-1]
    
    # Bullish Engulfing
    if (prev_close < prev_open and  # Previous bearish
        curr_close > curr_open and  # Current bullish
        curr_open < prev_close and  # Opens below prev close
        curr_close > prev_open):    # Closes above prev open
        return 1
    
    # Bearish Engulfing
    if (prev_close > prev_open and  # Previous bullish
        curr_close < curr_open and  # Current bearish
        curr_open > prev_close and  # Opens above prev close
        curr_close < prev_open):    # Closes below prev open
        return -1
    
    return 0

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

def analyze_price_action(df, i):
    """AI-based price action analysis"""
    if i < 50:
        return 0, "Insufficient data", {}
    
    analysis = {}
    score = 0
    
    # Trend Analysis
    ema20 = df['EMA_20'].iloc[i]
    ema50 = df['EMA_50'].iloc[i]
    current_price = df['Close'].iloc[i]
    
    if current_price > ema20 > ema50:
        score += 2
        analysis['trend'] = "Strong Uptrend"
    elif current_price < ema20 < ema50:
        score -= 2
        analysis['trend'] = "Strong Downtrend"
    else:
        analysis['trend'] = "Sideways"
    
    # RSI Analysis
    rsi = df['RSI'].iloc[i]
    if rsi < 30:
        score += 1
        analysis['rsi'] = "Oversold - Bullish"
    elif rsi > 70:
        score -= 1
        analysis['rsi'] = "Overbought - Bearish"
    else:
        analysis['rsi'] = "Neutral"
    
    # MACD Analysis
    if df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]:
        score += 1
        analysis['macd'] = "Bullish Crossover"
    else:
        score -= 1
        analysis['macd'] = "Bearish"
    
    # Bollinger Bands
    if current_price < df['BB_Lower'].iloc[i]:
        score += 1
        analysis['bb'] = "Below Lower Band - Oversold"
    elif current_price > df['BB_Upper'].iloc[i]:
        score -= 1
        analysis['bb'] = "Above Upper Band - Overbought"
    else:
        analysis['bb'] = "Within Bands"
    
    # Volume Analysis
    avg_volume = df['Volume'].rolling(20).mean().iloc[i]
    current_volume = df['Volume'].iloc[i]
    if current_volume > avg_volume * 1.5:
        analysis['volume'] = "High Volume"
        score += 0.5 if score > 0 else -0.5
    else:
        analysis['volume'] = "Normal Volume"
    
    # Generate signal
    if score >= 3:
        return 1, "Strong BUY Signal", analysis
    elif score <= -3:
        return -1, "Strong SELL Signal", analysis
    else:
        return 0, "No Clear Signal", analysis

# ==================== STRATEGY LOGIC ====================

def check_custom_strategy(df, i, conditions):
    """Check custom strategy conditions"""
    if i < 1:
        return 0, None
    
    buy_conditions_met = 0
    sell_conditions_met = 0
    total_buy = len([c for c in conditions if c['action'] == 'BUY'])
    total_sell = len([c for c in conditions if c['action'] == 'SELL'])
    
    for condition in conditions:
        indicator = condition['indicator']
        operator = condition['operator']
        value = condition['value']
        action = condition['action']
        
        current_val = df[indicator].iloc[i]
        
        condition_met = False
        if operator == '>':
            condition_met = current_val > value
        elif operator == '<':
            condition_met = current_val < value
        elif operator == '>=':
            condition_met = current_val >= value
        elif operator == '<=':
            condition_met = current_val <= value
        elif operator == '==':
            condition_met = abs(current_val - value) < 0.01
        elif operator == 'crosses_above':
            condition_met = current_val > value and df[indicator].iloc[i-1] <= value
        elif operator == 'crosses_below':
            condition_met = current_val < value and df[indicator].iloc[i-1] >= value
        
        if condition_met:
            if action == 'BUY':
                buy_conditions_met += 1
            elif action == 'SELL':
                sell_conditions_met += 1
    
    if total_buy > 0 and buy_conditions_met == total_buy:
        return 1, df['Close'].iloc[i]
    elif total_sell > 0 and sell_conditions_met == total_sell:
        return -1, df['Close'].iloc[i]
    
    return 0, None

def check_ema_crossover_entry(df, i, ema_fast, ema_slow, min_angle, entry_filter, 
                               custom_points, atr_multiplier, use_adx, adx_threshold):
    """Check EMA crossover entry conditions with filters"""
    if i < 1:
        return 0, None
    
    bullish_cross = (df['EMA_Fast'].iloc[i] > df['EMA_Slow'].iloc[i] and 
                     df['EMA_Fast'].iloc[i-1] <= df['EMA_Slow'].iloc[i-1])
    bearish_cross = (df['EMA_Fast'].iloc[i] < df['EMA_Slow'].iloc[i] and 
                     df['EMA_Fast'].iloc[i-1] >= df['EMA_Slow'].iloc[i-1])
    
    if not (bullish_cross or bearish_cross):
        return 0, None
    
    angle = abs(df['EMA_Angle'].iloc[i])
    if angle < min_angle:
        return 0, None
    
    if use_adx:
        if df['ADX'].iloc[i] < adx_threshold:
            return 0, None
    
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
    
    if rsi > 80 and adx < 20 and ema1 < ema2:
        return -1, df['Close'].iloc[i]
    
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
        if signal == 1:
            sl = entry_price - custom_sl_points
        else:
            sl = entry_price + custom_sl_points
    
    elif sl_type == "ATR-based":
        atr_value = df['ATR'].iloc[i]
        if signal == 1:
            sl = entry_price - (atr_value * atr_multiplier)
        else:
            sl = entry_price + (atr_value * atr_multiplier)
    
    elif sl_type == "Break-even After 50% Target":
        # Initial SL, will be moved to breakeven after 50% target
        if signal == 1:
            sl = entry_price - custom_sl_points
        else:
            sl = entry_price + custom_sl_points
    
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
        elif sl_type == "Volatility-Adjusted Trailing SL":
            atr_value = df['ATR'].iloc[i]
            offset = atr_value * atr_multiplier
        else:
            offset = custom_sl_points
        
        if signal == 1:
            sl = trailing_sl_high - offset if trailing_sl_high else entry_price - offset
        else:
            sl = trailing_sl_low + offset if trailing_sl_low else entry_price + offset
    
    elif sl_type == "Signal-based (reverse EMA crossover)":
        sl = 0
    
    else:
        sl = entry_price - custom_sl_points if signal == 1 else entry_price + custom_sl_points
    
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
    
    elif target_type == "50% Exit at Target (Partial)":
        # Initial target for 50% exit
        if signal == 1:
            target = entry_price + custom_target_points
        else:
            target = entry_price - custom_target_points
    
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
        if signal == 1:
            target = trailing_target_high if trailing_target_high else entry_price + min_target_distance
        else:
            target = trailing_target_low if trailing_target_low else entry_price - min_target_distance
    
    elif target_type == "Signal-based (reverse EMA crossover)":
        target = 0
    
    else:
        target = entry_price + custom_target_points if signal == 1 else entry_price - custom_target_points
    
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
    
    new_trailing_sl_high = trailing_sl_high
    new_trailing_sl_low = trailing_sl_low
    new_sl = sl
    
    # Break-even SL logic
    if sl_type == "Break-even After 50% Target":
        profit = (current_price - entry_price) * signal
        target_distance = abs(target - entry_price) if target != 0 else target_points
        
        if profit >= target_distance * 0.5 and not position.get('breakeven_moved', False):
            new_sl = entry_price
            position['breakeven_moved'] = True
    
    # Trailing SL updates
    if "Trailing SL" in sl_type:
        if signal == 1:
            if current_price > trailing_sl_high:
                price_move = current_price - trailing_sl_high
                if price_move >= trailing_threshold:
                    new_trailing_sl_high = current_price
                    new_sl = calculate_stop_loss(df, i, signal, entry_price, sl_type, 
                                                position.get('custom_sl_points', 10),
                                                position.get('atr_multiplier', 1.5),
                                                new_trailing_sl_high, new_trailing_sl_low,
                                                trailing_threshold)
        else:
            if current_price < trailing_sl_low:
                price_move = trailing_sl_low - current_price
                if price_move >= trailing_threshold:
                    new_trailing_sl_low = current_price
                    new_sl = calculate_stop_loss(df, i, signal, entry_price, sl_type,
                                                position.get('custom_sl_points', 10),
                                                position.get('atr_multiplier', 1.5),
                                                new_trailing_sl_high, new_trailing_sl_low,
                                                trailing_threshold)
    
    new_trailing_target_high = trailing_target_high
    new_trailing_target_low = trailing_target_low
    new_trailing_profit_points = trailing_profit_points
    new_target = target
    
    # Trailing target updates
    if "Trailing Target" in target_type and target_type != "Trailing Target + Signal Based":
        if signal == 1:
            if current_price > trailing_target_high:
                profit_points = current_price - entry_price
                if profit_points >= trailing_profit_points + target_points:
                    new_trailing_profit_points = profit_points
                    new_trailing_target_high = current_price
                    new_target = new_trailing_target_high
        else:
            if current_price < trailing_target_low:
                profit_points = entry_price - current_price
                if profit_points >= trailing_profit_points + target_points:
                    new_trailing_profit_points = profit_points
                    new_trailing_target_low = current_price
                    new_target = new_trailing_target_low
    
    # Check 50% partial exit
    partial_exit = False
    if target_type == "50% Exit at Target (Partial)" and not position.get('partial_exit_done', False):
        if signal == 1 and current_price >= target and target != 0:
            partial_exit = True
            position['partial_exit_done'] = True
        elif signal == -1 and current_price <= target and target != 0:
            partial_exit = True
            position['partial_exit_done'] = True
    
    # Signal-based exits
    if sl_type == "Signal-based (reverse EMA crossover)" or target_type == "Signal-based (reverse EMA crossover)" or target_type == "Trailing Target + Signal Based":
        if i > 0:
            if signal == 1:
                if (df['EMA_Fast'].iloc[i] < df['EMA_Slow'].iloc[i] and 
                    df['EMA_Fast'].iloc[i-1] >= df['EMA_Slow'].iloc[i-1]):
                    return True, "Reverse Signal - Bearish Crossover", current_price, new_sl, new_target, new_trailing_sl_high, new_trailing_sl_low, new_trailing_target_high, new_trailing_target_low, new_trailing_profit_points, False
            else:
                if (df['EMA_Fast'].iloc[i] > df['EMA_Slow'].iloc[i] and 
                    df['EMA_Fast'].iloc[i-1] <= df['EMA_Slow'].iloc[i-1]):
                    return True, "Reverse Signal - Bullish Crossover", current_price, new_sl, new_target, new_trailing_sl_high, new_trailing_sl_low, new_trailing_target_high, new_trailing_target_low, new_trailing_profit_points, False
    
    # Check SL hit
    if sl_type != "Signal-based (reverse EMA crossover)" and new_sl != 0:
        if signal == 1 and current_price <= new_sl:
            return True, "Stop Loss Hit", current_price, new_sl, new_target, new_trailing_sl_high, new_trailing_sl_low, new_trailing_target_high, new_trailing_target_low, new_trailing_profit_points, False
        elif signal == -1 and current_price >= new_sl:
            return True, "Stop Loss Hit", current_price, new_sl, new_target, new_trailing_sl_high, new_trailing_sl_low, new_trailing_target_high, new_trailing_target_low, new_trailing_profit_points, False
    
    # Check target hit (full exit only for non-trailing, non-signal, non-partial)
    if target_type not in ["Signal-based (reverse EMA crossover)", "Trailing Target (Points)", "Trailing Target + Signal Based", "50% Exit at Target (Partial)"] and new_target != 0:
        if signal == 1 and current_price >= new_target:
            return True, "Target Hit", current_price, new_sl, new_target, new_trailing_sl_high, new_trailing_sl_low, new_trailing_target_high, new_trailing_target_low, new_trailing_profit_points, False
        elif signal == -1 and current_price <= new_target:
            return True, "Target Hit", current_price, new_sl, new_target, new_trailing_sl_high, new_trailing_sl_low, new_trailing_target_high, new_trailing_target_low, new_trailing_profit_points, False
    
    return False, None, None, new_sl, new_target, new_trailing_sl_high, new_trailing_sl_low, new_trailing_target_high, new_trailing_target_low, new_trailing_profit_points, partial_exit

# ==================== UI FUNCTIONS ====================

def create_live_chart(df, position=None, sl=None, target=None):
    """Create live candlestick chart with indicators"""
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
    if len(st.session_state['trade_logs']) > 50:
        st.session_state['trade_logs'] = st.session_state['trade_logs'][-50:]
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
    defaults = {
        'trading_active': False,
        'current_data': None,
        'position': None,
        'trade_history': [],
        'trade_logs': [],
        'trailing_sl_high': None,
        'trailing_sl_low': None,
        'trailing_target_high': None,
        'trailing_target_low': None,
        'trailing_profit_points': 0,
        'threshold_crossed': False,
        'highest_price': None,
        'lowest_price': None,
        'custom_conditions': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_position_state():
    """Reset position-related state"""
    st.session_state['position'] = None
    st.session_state['trailing_sl_high'] = None
    st.session_state['trailing_sl_low'] = None
    st.session_state['trailing_target_high'] = None
    st.session_state['trailing_target_low'] = None
    st.session_state['trailing_profit_points'] = 0
    st.session_state['highest_price'] = None
    st.session_state['lowest_price'] = None
    st.session_state['threshold_crossed'] = False

# ==================== MAIN APPLICATION ====================

def main():
    st.set_page_config(page_title="Live Trading Dashboard", layout="wide", initial_sidebar_state="expanded")
    
    initialize_session_state()
    
    st.title("🚀 Live Trading Dashboard - Advanced Pro")
    
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
            ["EMA Crossover", "Simple Buy", "Simple Sell", "Price Threshold", 
             "RSI-ADX-EMA Strategy", "AI Price Action Analysis", "Custom Strategy Builder"])
        
        # Strategy-specific parameters
        ema_fast = 9
        ema_slow = 15
        min_angle = 1.0
        entry_filter = "Simple Crossover"
        custom_points = 0
        atr_multiplier = 1.5
        use_adx = False
        adx_threshold = 25
        adx_period = 14
        threshold = 100.0
        threshold_type = "LONG (Price >= Threshold)"
        
        if strategy == "EMA Crossover":
            st.markdown("**EMA Parameters**")
            ema_fast = st.number_input("EMA Fast", min_value=2, max_value=200, value=9)
            ema_slow = st.number_input("EMA Slow", min_value=2, max_value=200, value=15)
            min_angle = st.number_input("Minimum Crossover Angle (degrees)", 
                                       min_value=0.0, max_value=90.0, value=1.0, step=0.1)
            
            st.markdown("**Entry Filter**")
            entry_filter = st.selectbox("Entry Filter Type", 
                ["Simple Crossover", "Custom Candle (Points)", "ATR-based Candle"])
            
            if entry_filter == "Custom Candle (Points)":
                custom_points = st.number_input("Minimum Candle Size (Points)", 
                                               min_value=1, value=10)
            elif entry_filter == "ATR-based Candle":
                atr_multiplier = st.number_input("ATR Multiplier", 
                                                min_value=0.1, max_value=5.0, value=1.5, step=0.1)
            
            st.markdown("**ADX Filter**")
            use_adx = st.checkbox("Enable ADX Filter", value=False)
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
        
        elif strategy == "AI Price Action Analysis":
            st.info("🤖 AI will analyze multiple indicators and provide trading recommendations with automatic SL/Target")
        
        elif strategy == "Custom Strategy Builder":
            st.markdown("**Build Your Custom Strategy**")
            st.info("Add multiple conditions. All BUY conditions must be true for BUY signal, all SELL conditions for SELL signal.")
            
            if st.button("➕ Add Condition"):
                st.session_state['custom_conditions'].append({
                    'indicator': 'RSI',
                    'operator': '>',
                    'value': 50,
                    'action': 'BUY'
                })
            
            for idx, condition in enumerate(st.session_state['custom_conditions']):
                with st.expander(f"Condition {idx + 1}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        condition['indicator'] = st.selectbox(
                            f"Indicator {idx}", 
                            ['RSI', 'ADX', 'EMA_Fast', 'EMA_Slow', 'MACD', 'MACD_Signal',
                             'BB_Upper', 'BB_Lower', 'ATR', 'Volume', 'Close', 'High', 'Low'],
                            key=f"ind_{idx}"
                        )
                        condition['operator'] = st.selectbox(
                            f"Operator {idx}",
                            ['>', '<', '>=', '<=', '==', 'crosses_above', 'crosses_below'],
                            key=f"op_{idx}"
                        )
                    with col2:
                        condition['value'] = st.number_input(
                            f"Value {idx}",
                            value=float(condition['value']),
                            key=f"val_{idx}"
                        )
                        condition['action'] = st.selectbox(
                            f"Action {idx}",
                            ['BUY', 'SELL'],
                            key=f"act_{idx}"
                        )
                    
                    if st.button(f"🗑️ Remove Condition {idx + 1}", key=f"rem_{idx}"):
                        st.session_state['custom_conditions'].pop(idx)
                        st.rerun()
        
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
            "Volatility-Adjusted Trailing SL",
            "Break-even After 50% Target",
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
        elif "Trailing SL" in sl_type or sl_type == "Volatility-Adjusted Trailing SL":
            custom_sl_points = st.number_input("Trailing Offset (Points)", min_value=1, value=10)
            trailing_threshold = st.number_input("Trailing Threshold (Points)", 
                                                 min_value=0, value=0, 
                                                 help="SL updates only after price moves by this amount")
            if sl_type == "Volatility-Adjusted Trailing SL":
                sl_atr_multiplier = st.number_input("ATR Multiplier (Volatility)", 
                                                   min_value=0.1, max_value=5.0, value=1.5, step=0.1)
        elif sl_type == "ATR-based":
            sl_atr_multiplier = st.number_input("ATR Multiplier (SL)", 
                                               min_value=0.1, max_value=5.0, value=1.5, step=0.1)
        elif sl_type == "Break-even After 50% Target":
            st.info("SL moves to entry price after reaching 50% of target")
        
        # Target Configuration
        st.subheader("Target")
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
        ])
        
        custom_target_points = 20
        target_atr_multiplier = 2.0
        rr_ratio = 2.0
        
        if target_type == "Custom Points":
            custom_target_points = st.number_input("Target Points", min_value=1, value=20)
        elif "Trailing Target" in target_type:
            custom_target_points = st.number_input("Trailing Target Points", min_value=1, value=20,
                                                   help="Target updates after price moves by this amount")
        elif target_type == "50% Exit at Target (Partial)":
            custom_target_points = st.number_input("Initial Target Points", min_value=1, value=20)
            st.info("📊 Exits 50% position at target, trails remaining 50%")
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
                reset_position_state()
                add_log("Trading stopped by user - Position reset")
                st.rerun()
        
        with col3:
            if st.session_state['trading_active']:
                st.success("🟢 Trading is ACTIVE")
            else:
                st.info("⚪ Trading is STOPPED")
        
        st.divider()
        
        # Live Trading Loop
        if st.session_state['trading_active']:
            status_container = st.container()
            metrics_container = st.container()
            chart_container = st.container()
            position_container = st.container()
            
            while st.session_state['trading_active']:
                with status_container:
                    with st.spinner("Fetching live data..."):
                        df = fetch_data_with_delay(asset, period, interval)
                
                if df is None or df.empty:
                    st.error("Failed to fetch data. Stopping trading.")
                    st.session_state['trading_active'] = False
                    break
                
                # Calculate all indicators
                df['EMA_Fast'] = calculate_ema(df['Close'], ema_fast)
                df['EMA_Slow'] = calculate_ema(df['Close'], ema_slow)
                df['EMA_20'] = calculate_ema(df['Close'], 20)
                df['EMA_50'] = calculate_ema(df['Close'], 50)
                df['EMA_Angle'] = calculate_ema_angle(df)
                df['ATR'] = calculate_atr(df)
                df['ADX'] = calculate_adx(df, adx_period)
                df['RSI'] = calculate_rsi(df['Close'])
                df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
                df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
                df['Stoch_K'], df['Stoch_D'] = calculate_stochastic_rsi(df['Close'])
                df['Keltner_Upper'], df['Keltner_Middle'], df['Keltner_Lower'] = calculate_keltner_channel(df)
                
                st.session_state['current_data'] = df
                
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
                            
                            if pnl >= 0:
                                st.metric("Unrealized P&L", f"{pnl:.2f}", delta=f"+{pnl:.2f}")
                            else:
                                st.metric("Unrealized P&L", f"{pnl:.2f}", delta=f"{pnl:.2f}", delta_color="inverse")
                        else:
                            st.metric("Unrealized P&L", "0.00")
                    
                    with col4:
                        st.metric("Last Update", current_time.strftime("%H:%M:%S"))
                    
                    col5, col6, col7, col8 = st.columns(4)
                    
                    with col5:
                        st.metric("EMA Fast", f"{df['EMA_Fast'].iloc[i]:.2f}")
                    
                    with col6:
                        st.metric("EMA Slow", f"{df['EMA_Slow'].iloc[i]:.2f}")
                    
                    with col7:
                        angle = df['EMA_Angle'].iloc[i]
                        st.metric("Crossover Angle", f"{angle:.2f}°")
                    
                    with col8:
                        st.metric("RSI", f"{df['RSI'].iloc[i]:.2f}")
                    
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
                            st.info(f"{status} Candle Size: {candle_size:.2f} / Min (ATR×{atr_multiplier}): {min_candle:.2f}")
                    
                    # AI Analysis Display
                    if strategy == "AI Price Action Analysis":
                        ai_signal, ai_reason, ai_analysis = analyze_price_action(df, i)
                        
                        st.markdown("### 🤖 AI Market Analysis")
                        if ai_signal == 1:
                            st.success(f"**{ai_reason}**")
                        elif ai_signal == -1:
                            st.error(f"**{ai_reason}**")
                        else:
                            st.info(f"**{ai_reason}**")
                        
                        col_a1, col_a2, col_a3 = st.columns(3)
                        with col_a1:
                            st.text(f"Trend: {ai_analysis.get('trend', 'N/A')}")
                            st.text(f"RSI: {ai_analysis.get('rsi', 'N/A')}")
                        with col_a2:
                            st.text(f"MACD: {ai_analysis.get('macd', 'N/A')}")
                            st.text(f"BB: {ai_analysis.get('bb', 'N/A')}")
                        with col_a3:
                            st.text(f"Volume: {ai_analysis.get('volume', 'N/A')}")
                
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
                    elif strategy == "AI Price Action Analysis":
                        signal, reason, analysis = analyze_price_action(df, i)
                        if signal != 0:
                            entry_price = current_price
                    elif strategy == "Custom Strategy Builder":
                        if len(st.session_state['custom_conditions']) > 0:
                            signal, entry_price = check_custom_strategy(df, i, st.session_state['custom_conditions'])
                    
                    # Enter position
                    if signal != 0 and entry_price:
                        st.session_state['trailing_sl_high'] = entry_price
                        st.session_state['trailing_sl_low'] = entry_price
                        st.session_state['trailing_target_high'] = entry_price
                        st.session_state['trailing_target_low'] = entry_price
                        st.session_state['trailing_profit_points'] = 0
                        st.session_state['highest_price'] = entry_price
                        st.session_state['lowest_price'] = entry_price
                        
                        # Auto SL/Target for AI strategy
                        if strategy == "AI Price Action Analysis":
                            atr_val = df['ATR'].iloc[i]
                            sl = entry_price - (atr_val * 1.5) if signal == 1 else entry_price + (atr_val * 1.5)
                            target = entry_price + (atr_val * 3) if signal == 1 else entry_price - (atr_val * 3)
                        else:
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
                            'target_type': target_type,
                            'breakeven_moved': False,
                            'partial_exit_done': False
                        }
                        
                        signal_str = "LONG" if signal == 1 else "SHORT"
                        sl_str = f"{sl:.2f}" if sl != 0 else "Signal Based"
                        target_str = f"{target:.2f}" if target != 0 else "Signal Based"
                        add_log(f"Entered {signal_str} position at {entry_price:.2f} | SL: {sl_str} | Target: {target_str}")
                
                # Check exit conditions if in position
                elif st.session_state['position']:
                    position = st.session_state['position']
                    
                    if st.session_state['highest_price'] is None or current_price > st.session_state['highest_price']:
                        st.session_state['highest_price'] = current_price
                    if st.session_state['lowest_price'] is None or current_price < st.session_state['lowest_price']:
                        st.session_state['lowest_price'] = current_price
                    
                    exit_triggered, exit_reason, exit_price, new_sl, new_target, new_tsl_h, new_tsl_l, new_tt_h, new_tt_l, new_tp, partial_exit = check_exit_conditions(
                        df, i, position, current_price, position['sl'], position['target'],
                        position['sl_type'], position['target_type'],
                        st.session_state['trailing_sl_high'], st.session_state['trailing_sl_low'],
                        trailing_threshold, st.session_state['trailing_target_high'],
                        st.session_state['trailing_target_low'], st.session_state['trailing_profit_points'],
                        custom_target_points
                    )
                    
                    st.session_state['trailing_sl_high'] = new_tsl_h
                    st.session_state['trailing_sl_low'] = new_tsl_l
                    st.session_state['trailing_target_high'] = new_tt_h
                    st.session_state['trailing_target_low'] = new_tt_l
                    st.session_state['trailing_profit_points'] = new_tp
                    position['sl'] = new_sl
                    position['target'] = new_target
                    
                    if partial_exit:
                        add_log(f"50% position exited at {current_price:.2f} - Trailing remaining 50%")
                    
                    if exit_triggered:
                        signal = position['signal']
                        entry_price = position['entry_price']
                        pnl = (exit_price - entry_price) * signal
                        
                        duration = (current_time - position['entry_time']).total_seconds()
                        
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
                        
                        add_log(f"Exited {trade_record['signal']} position | Exit: {exit_price:.2f} | Reason: {exit_reason} | PnL: {pnl:.2f}")
                        
                        reset_position_state()
                
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
                        
                        if "Trailing Target" in pos['target_type']:
                            st.info(f"📈 Profit moved: {st.session_state['trailing_profit_points']:.2f} points | Next update at: {st.session_state['trailing_profit_points'] + custom_target_points:.2f} points")
                        
                        if pos.get('partial_exit_done', False):
                            st.success("✅ 50% position already exited - Trailing remaining")
                        
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
                
                time.sleep(random.uniform(1.0, 1.5))
                st.rerun()
        
        else:
            st.info("👆 Click **Start Trading** to begin live monitoring")
            
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
            total_trades = len(st.session_state['trade_history'])
            winning_trades = sum(1 for t in st.session_state['trade_history'] if t.get('pnl', 0) > 0)
            losing_trades = sum(1 for t in st.session_state['trade_history'] if t.get('pnl', 0) < 0)
            accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = sum(t.get('pnl', 0) for t in st.session_state['trade_history'])
            
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
            
            st.divider()
            
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
