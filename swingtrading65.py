import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import pytz
import time
import random

# ==================== CONFIGURATION ====================

ASSET_MAPPING = {
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "USDINR": "USDINR=X",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Custom": ""
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

SL_TYPES = [
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

TARGET_TYPES = [
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

STRATEGIES = [
    "EMA Crossover Strategy",
    "Simple Buy Strategy",
    "Simple Sell Strategy",
    "Price Crosses Threshold Strategy",
    "RSI-ADX-EMA Strategy",
    "Percentage Change Strategy",
    "AI Price Action Analysis",
    "Custom Strategy Builder"
]

# ==================== INDICATOR CALCULATIONS ====================

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data, period):
    return data.rolling(window=period).mean()

def calculate_atr(df, period=14):
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
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_adx(df, period=14):
    high = df['High']
    low = df['Low']
    
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

def calculate_bollinger_bands(data, period=20, std=2):
    sma = calculate_sma(data, period)
    std_dev = data.rolling(window=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper, sma, lower

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    histogram = macd - signal_line
    
    return macd, signal_line, histogram

def calculate_swing_levels(df, lookback=5):
    swing_high = df['High'].rolling(window=lookback*2+1, center=True).max()
    swing_low = df['Low'].rolling(window=lookback*2+1, center=True).min()
    
    return swing_high, swing_low

def calculate_support_resistance(df, lookback=20):
    resistance = df['High'].rolling(window=lookback).max()
    support = df['Low'].rolling(window=lookback).min()
    
    return support, resistance

def calculate_ema_angle(ema_series, lookback=2):
    if len(ema_series) < lookback + 1:
        return pd.Series(0, index=ema_series.index)
    
    slope = ema_series.diff(lookback) / lookback
    angle = np.degrees(np.arctan(slope))
    
    return angle

def calculate_supertrend(df, period=10, multiplier=3):
    atr = calculate_atr(df, period)
    hl_avg = (df['High'] + df['Low']) / 2
    
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    supertrend = pd.Series(0.0, index=df.index)
    
    for i in range(period, len(df)):
        if df['Close'].iloc[i] > upper_band.iloc[i-1]:
            supertrend.iloc[i] = lower_band.iloc[i]
        elif df['Close'].iloc[i] < lower_band.iloc[i-1]:
            supertrend.iloc[i] = upper_band.iloc[i]
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]
    
    return supertrend

def calculate_vwap(df):
    if 'Volume' not in df.columns or df['Volume'].sum() == 0:
        return pd.Series(np.nan, index=df.index)
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    return vwap

def calculate_keltner_channel(df, period=20, atr_mult=2):
    ema = calculate_ema(df['Close'], period)
    atr = calculate_atr(df, period)
    
    upper = ema + (atr * atr_mult)
    lower = ema - (atr * atr_mult)
    
    return upper, ema, lower

# ==================== DATA FETCHING ====================

def fetch_data(ticker, interval, period, mode="Backtesting"):
    try:
        if mode == "Live Trading":
            time.sleep(random.uniform(1.0, 1.5))
        
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if data.empty:
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in required_cols if col in data.columns]
        data = data[available_cols].copy()
        
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            data.index = data.index.tz_convert('Asia/Kolkata')
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# ==================== STRATEGY LOGIC ====================

def ema_crossover_strategy(df, config):
    ema_fast = calculate_ema(df['Close'], config['ema_fast'])
    ema_slow = calculate_ema(df['Close'], config['ema_slow'])
    
    df['EMA_Fast'] = ema_fast
    df['EMA_Slow'] = ema_slow
    
    ema_angle = calculate_ema_angle(ema_fast, lookback=2)
    
    cross_above = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
    cross_below = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
    
    angle_filter = abs(ema_angle) >= config['min_angle']
    
    entry_filter = pd.Series(True, index=df.index)
    
    if config['entry_filter'] == 'Custom Candle (Points)':
        candle_size = abs(df['Close'] - df['Open'])
        entry_filter = candle_size >= config['custom_points']
    
    elif config['entry_filter'] == 'ATR-based Candle':
        atr = calculate_atr(df, 14)
        candle_size = abs(df['Close'] - df['Open'])
        entry_filter = candle_size >= (atr * config['atr_multiplier'])
    
    adx_filter = pd.Series(True, index=df.index)
    if config['use_adx']:
        adx = calculate_adx(df, config['adx_period'])
        df['ADX'] = adx
        adx_filter = adx >= config['adx_threshold']
    
    buy_signal = cross_above & angle_filter & entry_filter & adx_filter
    sell_signal = cross_below & angle_filter & entry_filter & adx_filter
    
    return buy_signal, sell_signal, df

def simple_buy_strategy(df, config):
    buy_signal = pd.Series(False, index=df.index)
    buy_signal.iloc[-1] = True
    sell_signal = pd.Series(False, index=df.index)
    
    return buy_signal, sell_signal, df

def simple_sell_strategy(df, config):
    buy_signal = pd.Series(False, index=df.index)
    sell_signal = pd.Series(False, index=df.index)
    sell_signal.iloc[-1] = True
    
    return buy_signal, sell_signal, df

def price_crosses_threshold_strategy(df, config):
    threshold = config['threshold_price']
    direction = config['threshold_direction']
    
    buy_signal = pd.Series(False, index=df.index)
    sell_signal = pd.Series(False, index=df.index)
    
    if direction == "LONG (Price >= Threshold)":
        buy_signal = df['Close'] >= threshold
    elif direction == "SHORT (Price >= Threshold)":
        sell_signal = df['Close'] >= threshold
    elif direction == "LONG (Price <= Threshold)":
        buy_signal = df['Close'] <= threshold
    elif direction == "SHORT (Price <= Threshold)":
        sell_signal = df['Close'] <= threshold
    
    return buy_signal, sell_signal, df

def rsi_adx_ema_strategy(df, config):
    rsi = calculate_rsi(df['Close'], 14)
    adx = calculate_adx(df, 14)
    ema1 = calculate_ema(df['Close'], config.get('ema_fast', 9))
    ema2 = calculate_ema(df['Close'], config.get('ema_slow', 15))
    
    df['RSI'] = rsi
    df['ADX'] = adx
    df['EMA_Fast'] = ema1
    df['EMA_Slow'] = ema2
    
    buy_signal = (rsi < 20) & (adx > 20) & (ema1 > ema2)
    sell_signal = (rsi > 80) & (adx < 20) & (ema1 < ema2)
    
    return buy_signal, sell_signal, df

def percentage_change_strategy(df, config):
    first_price = df['Close'].iloc[0]
    pct_change = ((df['Close'] - first_price) / first_price) * 100
    
    df['Pct_Change'] = pct_change
    
    threshold = config['pct_threshold']
    direction = config['pct_direction']
    
    buy_signal = pd.Series(False, index=df.index)
    sell_signal = pd.Series(False, index=df.index)
    
    if direction == "BUY on Fall":
        buy_signal = pct_change <= -threshold
    elif direction == "SELL on Fall":
        sell_signal = pct_change <= -threshold
    elif direction == "BUY on Rise":
        buy_signal = pct_change >= threshold
    elif direction == "SELL on Rise":
        sell_signal = pct_change >= threshold
    
    return buy_signal, sell_signal, df

def ai_price_action_strategy(df, config):
    ema_20 = calculate_ema(df['Close'], 20)
    ema_50 = calculate_ema(df['Close'], 50)
    rsi = calculate_rsi(df['Close'], 14)
    macd, macd_signal, _ = calculate_macd(df['Close'])
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(df['Close'])
    atr = calculate_atr(df, 14)
    
    df['EMA_20'] = ema_20
    df['EMA_50'] = ema_50
    df['RSI'] = rsi
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    df['ATR'] = atr
    
    buy_score = 0
    sell_score = 0
    
    if ema_20.iloc[-1] > ema_50.iloc[-1]:
        buy_score += 2
    else:
        sell_score += 2
    
    if rsi.iloc[-1] < 30:
        buy_score += 2
    elif rsi.iloc[-1] > 70:
        sell_score += 2
    
    if macd.iloc[-1] > macd_signal.iloc[-1]:
        buy_score += 1
    else:
        sell_signal += 1
    
    if df['Close'].iloc[-1] < bb_lower.iloc[-1]:
        buy_score += 1
    elif df['Close'].iloc[-1] > bb_upper.iloc[-1]:
        sell_score += 1
    
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        vol_sma = df['Volume'].rolling(20).mean()
        if df['Volume'].iloc[-1] > vol_sma.iloc[-1]:
            if buy_score > sell_score:
                buy_score += 1
            else:
                sell_score += 1
    
    buy_signal = pd.Series(False, index=df.index)
    sell_signal = pd.Series(False, index=df.index)
    
    if buy_score > sell_score and buy_score >= 4:
        buy_signal.iloc[-1] = True
    elif sell_score > buy_score and sell_score >= 4:
        sell_signal.iloc[-1] = True
    
    df['AI_Buy_Score'] = buy_score
    df['AI_Sell_Score'] = sell_score
    
    return buy_signal, sell_signal, df

def custom_strategy_builder(df, config):
    conditions = config.get('custom_conditions', [])
    
    df['Price'] = df['Close']
    df['EMA_20'] = calculate_ema(df['Close'], 20)
    df['EMA_50'] = calculate_ema(df['Close'], 50)
    df['EMA_Fast'] = calculate_ema(df['Close'], config.get('ema_fast', 9))
    df['EMA_Slow'] = calculate_ema(df['Close'], config.get('ema_slow', 15))
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['ADX'] = calculate_adx(df, 14)
    df['ATR'] = calculate_atr(df, 14)
    
    macd, macd_sig, _ = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macd_sig
    
    bb_upper, _, bb_lower = calculate_bollinger_bands(df['Close'])
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    
    df['SuperTrend'] = calculate_supertrend(df)
    df['VWAP'] = calculate_vwap(df)
    
    kelt_upper, _, kelt_lower = calculate_keltner_channel(df)
    df['Keltner_Upper'] = kelt_upper
    df['Keltner_Lower'] = kelt_lower
    
    support, resistance = calculate_support_resistance(df)
    df['Support'] = support
    df['Resistance'] = resistance
    
    buy_conditions = []
    sell_conditions = []
    
    for cond in conditions:
        if not cond.get('use_condition', False):
            continue
        
        indicator = cond['indicator']
        operator = cond['operator']
        action = cond['action']
        
        if cond.get('compare_with_indicator', False):
            compare_indicator = cond.get('compare_indicator', 'EMA_20')
            
            if operator == '>':
                condition_met = df[indicator] > df[compare_indicator]
            elif operator == '<':
                condition_met = df[indicator] < df[compare_indicator]
            elif operator == '>=':
                condition_met = df[indicator] >= df[compare_indicator]
            elif operator == '<=':
                condition_met = df[indicator] <= df[compare_indicator]
            elif operator == '==':
                condition_met = df[indicator] == df[compare_indicator]
            elif operator == 'crosses_above':
                condition_met = (df[indicator] > df[compare_indicator]) & (df[indicator].shift(1) <= df[compare_indicator].shift(1))
            elif operator == 'crosses_below':
                condition_met = (df[indicator] < df[compare_indicator]) & (df[indicator].shift(1) >= df[compare_indicator].shift(1))
            else:
                condition_met = pd.Series(False, index=df.index)
        else:
            value = cond['value']
            
            if operator == '>':
                condition_met = df[indicator] > value
            elif operator == '<':
                condition_met = df[indicator] < value
            elif operator == '>=':
                condition_met = df[indicator] >= value
            elif operator == '<=':
                condition_met = df[indicator] <= value
            elif operator == '==':
                condition_met = df[indicator] == value
            elif operator == 'crosses_above':
                condition_met = (df[indicator] > value) & (df[indicator].shift(1) <= value)
            elif operator == 'crosses_below':
                condition_met = (df[indicator] < value) & (df[indicator].shift(1) >= value)
            else:
                condition_met = pd.Series(False, index=df.index)
        
        if action == 'BUY':
            buy_conditions.append(condition_met)
        else:
            sell_conditions.append(condition_met)
    
    if buy_conditions:
        buy_signal = pd.concat(buy_conditions, axis=1).all(axis=1)
    else:
        buy_signal = pd.Series(False, index=df.index)
    
    if sell_conditions:
        sell_signal = pd.concat(sell_conditions, axis=1).all(axis=1)
    else:
        sell_signal = pd.Series(False, index=df.index)
    
    return buy_signal, sell_signal, df

# ==================== SL & TARGET LOGIC ====================

def calculate_sl_price(df, idx, position_type, entry_price, sl_type, config, position_state):
    if sl_type == "Custom Points":
        sl_points = config.get('sl_points', 10)
        return entry_price - sl_points if position_type == 'LONG' else entry_price + sl_points
    
    elif sl_type == "Trailing SL (Points)":
        trailing_offset = config.get('trailing_sl_points', 10)
        if position_type == 'LONG':
            highest = position_state.get('highest_price', entry_price)
            return highest - trailing_offset
        else:
            lowest = position_state.get('lowest_price', entry_price)
            return lowest + trailing_offset
    
    elif sl_type == "ATR-based":
        atr = calculate_atr(df[:idx+1], 14).iloc[-1]
        atr_mult = config.get('atr_sl_multiplier', 1.5)
        return entry_price - (atr * atr_mult) if position_type == 'LONG' else entry_price + (atr * atr_mult)
    
    elif sl_type == "Signal-based (reverse EMA crossover)":
        if 'EMA_Fast' in df.columns and 'EMA_Slow' in df.columns:
            if position_type == 'LONG':
                if idx > 0 and df['EMA_Fast'].iloc[idx] < df['EMA_Slow'].iloc[idx] and df['EMA_Fast'].iloc[idx-1] >= df['EMA_Slow'].iloc[idx-1]:
                    return df['Close'].iloc[idx]
            else:
                if idx > 0 and df['EMA_Fast'].iloc[idx] > df['EMA_Slow'].iloc[idx] and df['EMA_Fast'].iloc[idx-1] <= df['EMA_Slow'].iloc[idx-1]:
                    return df['Close'].iloc[idx]
        return None
    
    else:
        return entry_price - 10 if position_type == 'LONG' else entry_price + 10

def calculate_target_price(df, idx, position_type, entry_price, target_type, config):
    if target_type == "Custom Points":
        target_points = config.get('target_points', 20)
        return entry_price + target_points if position_type == 'LONG' else entry_price - target_points
    
    elif target_type == "Trailing Target (Points)":
        return None
    
    elif target_type == "50% Exit at Target (Partial)":
        target_points = config.get('target_points', 20)
        return entry_price + target_points if position_type == 'LONG' else entry_price - target_points
    
    elif target_type == "ATR-based":
        atr = calculate_atr(df[:idx+1], 14).iloc[-1]
        atr_mult = config.get('atr_target_multiplier', 3.0)
        return entry_price + (atr * atr_mult) if position_type == 'LONG' else entry_price - (atr * atr_mult)
    
    elif target_type == "Risk-Reward Based":
        sl_price = calculate_sl_price(df, idx, position_type, entry_price, config.get('sl_type', 'Custom Points'), config, {})
        rr_ratio = config.get('risk_reward_ratio', 2.0)
        
        if sl_price:
            risk = abs(entry_price - sl_price)
            reward = risk * rr_ratio
            return entry_price + reward if position_type == 'LONG' else entry_price - reward
        
        return entry_price + 20 if position_type == 'LONG' else entry_price - 20
    
    elif target_type == "Signal-based (reverse EMA crossover)":
        if 'EMA_Fast' in df.columns and 'EMA_Slow' in df.columns:
            if position_type == 'LONG':
                if idx > 0 and df['EMA_Fast'].iloc[idx] < df['EMA_Slow'].iloc[idx] and df['EMA_Fast'].iloc[idx-1] >= df['EMA_Slow'].iloc[idx-1]:
                    return df['Close'].iloc[idx]
            else:
                if idx > 0 and df['EMA_Fast'].iloc[idx] > df['EMA_Slow'].iloc[idx] and df['EMA_Fast'].iloc[idx-1] <= df['EMA_Slow'].iloc[idx-1]:
                    return df['Close'].iloc[idx]
        return None
    
    else:
        return entry_price + 20 if position_type == 'LONG' else entry_price - 20

# ==================== BACKTESTING ENGINE ====================

def run_backtest(df, strategy_func, config):
    buy_signals, sell_signals, df = strategy_func(df, config)
    
    trades = []
    position = None
    position_state = {}
    
    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        current_time = df.index[i]
        
        if position:
            if position['type'] == 'LONG':
                position_state['highest_price'] = max(position_state.get('highest_price', position['entry_price']), current_price)
            else:
                position_state['lowest_price'] = min(position_state.get('lowest_price', position['entry_price']), current_price)
        
        if position:
            sl_price = calculate_sl_price(df, i, position['type'], position['entry_price'], config['sl_type'], config, position_state)
            target_price = calculate_target_price(df, i, position['type'], position['entry_price'], config['target_type'], config)
            
            exit_reason = None
            exit_price = None
            
            if sl_price is not None:
                if position['type'] == 'LONG' and current_price <= sl_price:
                    exit_reason = 'Stop Loss'
                    exit_price = sl_price
                elif position['type'] == 'SHORT' and current_price >= sl_price:
                    exit_reason = 'Stop Loss'
                    exit_price = sl_price
            
            if not exit_reason and target_price is not None:
                if config['target_type'] == "50% Exit at Target (Partial)":
                    if not position_state.get('partial_exit_done', False):
                        if position['type'] == 'LONG' and current_price >= target_price:
                            partial_qty = position['quantity'] / 2
                            pnl = (current_price - position['entry_price']) * partial_qty
                            
                            trades.append({
                                'entry_time': position['entry_time'],
                                'exit_time': current_time,
                                'type': position['type'],
                                'entry_price': position['entry_price'],
                                'exit_price': current_price,
                                'quantity': partial_qty,
                                'pnl': pnl,
                                'exit_reason': '50% Partial Exit',
                                'sl_price': sl_price,
                                'target_price': target_price
                            })
                            
                            position['quantity'] = partial_qty
                            position_state['partial_exit_done'] = True
                        
                        elif position['type'] == 'SHORT' and current_price <= target_price:
                            partial_qty = position['quantity'] / 2
                            pnl = (position['entry_price'] - current_price) * partial_qty
                            
                            trades.append({
                                'entry_time': position['entry_time'],
                                'exit_time': current_time,
                                'type': position['type'],
                                'entry_price': position['entry_price'],
                                'exit_price': current_price,
                                'quantity': partial_qty,
                                'pnl': pnl,
                                'exit_reason': '50% Partial Exit',
                                'sl_price': sl_price,
                                'target_price': target_price
                            })
                            
                            position['quantity'] = partial_qty
                            position_state['partial_exit_done'] = True
                else:
                    if position['type'] == 'LONG' and current_price >= target_price:
                        exit_reason = 'Target Hit'
                        exit_price = target_price
                    elif position['type'] == 'SHORT' and current_price <= target_price:
                        exit_reason = 'Target Hit'
                        exit_price = target_price
            
            if exit_reason and exit_price:
                if position['type'] == 'LONG':
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                else:
                    pnl = (position['entry_price'] - exit_price) * position['quantity']
                
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'type': position['type'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'quantity': position['quantity'],
                    'pnl': pnl,
                    'exit_reason': exit_reason,
                    'sl_price': sl_price,
                    'target_price': target_price
                })
                
                position = None
                position_state = {}
        
        if not position:
            if buy_signals.iloc[i]:
                position = {
                    'type': 'LONG',
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'quantity': config.get('quantity', 1)
                }
                position_state = {
                    'highest_price': current_price,
                    'partial_exit_done': False
                }
            
            elif sell_signals.iloc[i]:
                position = {
                    'type': 'SHORT',
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'quantity': config.get('quantity', 1)
                }
                position_state = {
                    'lowest_price': current_price,
                    'partial_exit_done': False
                }
    
    if position:
        exit_price = df['Close'].iloc[-1]
        if position['type'] == 'LONG':
            pnl = (exit_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - exit_price) * position['quantity']
        
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': df.index[-1],
            'type': position['type'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'pnl': pnl,
            'exit_reason': 'End of Data',
            'sl_price': None,
            'target_price': None
        })
    
    return trades, df

# ==================== LIVE TRADING ENGINE ====================

def live_trading_iteration():
    if not st.session_state.get('trading_active', False):
        return
    
    config = st.session_state['config']
    ticker = config['ticker']
    interval = config['interval']
    period = config['period']
    
    df = fetch_data(ticker, interval, period, mode="Live Trading")
    
    if df is None or df.empty:
        st.session_state['trade_logs'].append({
            'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')),
            'message': 'Error fetching data'
        })
        time.sleep(random.uniform(1.0, 1.5))
        st.rerun()
        return
    
    st.session_state['current_data'] = df
    
    strategy_name = config['strategy']
    strategy_map = {
        "EMA Crossover Strategy": ema_crossover_strategy,
        "Simple Buy Strategy": simple_buy_strategy,
        "Simple Sell Strategy": simple_sell_strategy,
        "Price Crosses Threshold Strategy": price_crosses_threshold_strategy,
        "RSI-ADX-EMA Strategy": rsi_adx_ema_strategy,
        "Percentage Change Strategy": percentage_change_strategy,
        "AI Price Action Analysis": ai_price_action_strategy,
        "Custom Strategy Builder": custom_strategy_builder
    }
    
    strategy_func = strategy_map.get(strategy_name, ema_crossover_strategy)
    buy_signals, sell_signals, df = strategy_func(df, config)
    
    st.session_state['current_data'] = df
    
    current_price = df['Close'].iloc[-1]
    current_time = df.index[-1]
    
    position = st.session_state.get('position', None)
    
    if position:
        if position['type'] == 'LONG':
            st.session_state['highest_price'] = max(st.session_state.get('highest_price', position['entry_price']), current_price)
        else:
            st.session_state['lowest_price'] = min(st.session_state.get('lowest_price', position['entry_price']), current_price)
    
    if position:
        idx = len(df) - 1
        sl_price = calculate_sl_price(df, idx, position['type'], position['entry_price'], config['sl_type'], config, {
            'highest_price': st.session_state.get('highest_price', position['entry_price']),
            'lowest_price': st.session_state.get('lowest_price', position['entry_price'])
        })
        target_price = calculate_target_price(df, idx, position['type'], position['entry_price'], config['target_type'], config)
        
        st.session_state['current_sl'] = sl_price
        st.session_state['current_target'] = target_price
        
        exit_reason = None
        exit_price = None
        
        if sl_price is not None:
            if position['type'] == 'LONG' and current_price <= sl_price:
                exit_reason = 'Stop Loss'
                exit_price = sl_price
            elif position['type'] == 'SHORT' and current_price >= sl_price:
                exit_reason = 'Stop Loss'
                exit_price = sl_price
        
        if not exit_reason and target_price is not None:
            if config['target_type'] == "50% Exit at Target (Partial)":
                if not st.session_state.get('partial_exit_done', False):
                    if position['type'] == 'LONG' and current_price >= target_price:
                        partial_qty = position['quantity'] / 2
                        pnl = (current_price - position['entry_price']) * partial_qty
                        
                        st.session_state['trade_history'].append({
                            'entry_time': position['entry_time'],
                            'exit_time': current_time,
                            'type': position['type'],
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'quantity': partial_qty,
                            'pnl': pnl,
                            'exit_reason': '50% Partial Exit',
                            'sl_price': sl_price,
                            'target_price': target_price
                        })
                        
                        st.session_state['trade_logs'].append({
                            'timestamp': current_time,
                            'message': f"50% Partial Exit - {position['type']} @ {current_price:.2f}, P&L: {pnl:.2f}"
                        })
                        
                        position['quantity'] = partial_qty
                        st.session_state['position'] = position
                        st.session_state['partial_exit_done'] = True
                    
                    elif position['type'] == 'SHORT' and current_price <= target_price:
                        partial_qty = position['quantity'] / 2
                        pnl = (position['entry_price'] - current_price) * partial_qty
                        
                        st.session_state['trade_history'].append({
                            'entry_time': position['entry_time'],
                            'exit_time': current_time,
                            'type': position['type'],
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'quantity': partial_qty,
                            'pnl': pnl,
                            'exit_reason': '50% Partial Exit',
                            'sl_price': sl_price,
                            'target_price': target_price
                        })
                        
                        st.session_state['trade_logs'].append({
                            'timestamp': current_time,
                            'message': f"50% Partial Exit - {position['type']} @ {current_price:.2f}, P&L: {pnl:.2f}"
                        })
                        
                        position['quantity'] = partial_qty
                        st.session_state['position'] = position
                        st.session_state['partial_exit_done'] = True
            else:
                if position['type'] == 'LONG' and current_price >= target_price:
                    exit_reason = 'Target Hit'
                    exit_price = target_price
                elif position['type'] == 'SHORT' and current_price <= target_price:
                    exit_reason = 'Target Hit'
                    exit_price = target_price
        
        if exit_reason and exit_price:
            if position['type'] == 'LONG':
                pnl = (exit_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - exit_price) * position['quantity']
            
            st.session_state['trade_history'].append({
                'entry_time': position['entry_time'],
                'exit_time': current_time,
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'quantity': position['quantity'],
                'pnl': pnl,
                'exit_reason': exit_reason,
                'sl_price': sl_price,
                'target_price': target_price
            })
            
            st.session_state['trade_logs'].append({
                'timestamp': current_time,
                'message': f"Position Closed - {exit_reason} - {position['type']} @ {exit_price:.2f}, P&L: {pnl:.2f}"
            })
            
            st.session_state['position'] = None
            st.session_state['highest_price'] = 0
            st.session_state['lowest_price'] = float('inf')
            st.session_state['partial_exit_done'] = False
    
    if not st.session_state.get('position'):
        if buy_signals.iloc[-1]:
            st.session_state['position'] = {
                'type': 'LONG',
                'entry_price': current_price,
                'entry_time': current_time,
                'quantity': config.get('quantity', 1)
            }
            st.session_state['highest_price'] = current_price
            st.session_state['partial_exit_done'] = False
            
            st.session_state['trade_logs'].append({
                'timestamp': current_time,
                'message': f"LONG Entry @ {current_price:.2f}"
            })
        
        elif sell_signals.iloc[-1]:
            st.session_state['position'] = {
                'type': 'SHORT',
                'entry_price': current_price,
                'entry_time': current_time,
                'quantity': config.get('quantity', 1)
            }
            st.session_state['lowest_price'] = current_price
            st.session_state['partial_exit_done'] = False
            
            st.session_state['trade_logs'].append({
                'timestamp': current_time,
                'message': f"SHORT Entry @ {current_price:.2f}"
            })
    
    time.sleep(random.uniform(1.0, 1.5))
    st.rerun()

# ==================== VISUALIZATION ====================

def create_candlestick_chart(df, live_position=None):
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
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], name='EMA Slow', line=dict(color='orange', width=1)))
    
    if live_position:
        fig.add_hline(y=live_position['entry_price'], line_dash="dash", line_color="blue", annotation_text="Entry")
        
        if st.session_state.get('current_sl'):
            fig.add_hline(y=st.session_state['current_sl'], line_dash="dash", line_color="red", annotation_text="SL")
        
        if st.session_state.get('current_target'):
            fig.add_hline(y=st.session_state['current_target'], line_dash="dash", line_color="green", annotation_text="Target")
    
    fig.update_layout(
        title='Price Chart',
        xaxis_title='Time',
        yaxis_title='Price',
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    return fig

# ==================== STREAMLIT UI ====================

def initialize_session_state():
    if 'trading_active' not in st.session_state:
        st.session_state['trading_active'] = False
    if 'position' not in st.session_state:
        st.session_state['position'] = None
    if 'trade_history' not in st.session_state:
        st.session_state['trade_history'] = []
    if 'trade_logs' not in st.session_state:
        st.session_state['trade_logs'] = []
    if 'current_data' not in st.session_state:
        st.session_state['current_data'] = None
    if 'highest_price' not in st.session_state:
        st.session_state['highest_price'] = 0
    if 'lowest_price' not in st.session_state:
        st.session_state['lowest_price'] = float('inf')
    if 'partial_exit_done' not in st.session_state:
        st.session_state['partial_exit_done'] = False
    if 'current_sl' not in st.session_state:
        st.session_state['current_sl'] = None
    if 'current_target' not in st.session_state:
        st.session_state['current_target'] = None
    if 'custom_conditions' not in st.session_state:
        st.session_state['custom_conditions'] = []

def main():
    st.set_page_config(page_title="Quantitative Trading System", layout="wide")
    initialize_session_state()
    
    st.title("🚀 Professional Quantitative Trading System")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        mode = st.selectbox("Mode", ["Backtesting", "Live Trading"])
        
        asset_name = st.selectbox("Select Asset", list(ASSET_MAPPING.keys()))
        
        if asset_name == "Custom":
            custom_ticker = st.text_input("Enter Custom Ticker", value="AAPL")
            ticker = custom_ticker
        else:
            ticker = ASSET_MAPPING[asset_name]
        
        interval = st.selectbox("Interval", list(INTERVAL_PERIODS.keys()))
        period = st.selectbox("Period", INTERVAL_PERIODS[interval])
        quantity = st.number_input("Quantity", min_value=1, value=1)
        strategy = st.selectbox("Strategy", STRATEGIES)
        
        st.markdown("---")
        
        config = {
            'ticker': ticker,
            'interval': interval,
            'period': period,
            'quantity': quantity,
            'strategy': strategy
        }
        
        if strategy == "EMA Crossover Strategy":
            st.subheader("EMA Crossover Settings")
            config['ema_fast'] = st.number_input("EMA Fast", min_value=1, value=9)
            config['ema_slow'] = st.number_input("EMA Slow", min_value=1, value=15)
            config['min_angle'] = st.number_input("Minimum Angle (degrees)", min_value=0.0, value=1.0)
            
            config['entry_filter'] = st.selectbox("Entry Filter", [
                "Simple Crossover",
                "Custom Candle (Points)",
                "ATR-based Candle"
            ])
            
            if config['entry_filter'] == "Custom Candle (Points)":
                config['custom_points'] = st.number_input("Custom Points", min_value=1.0, value=10.0)
            
            if config['entry_filter'] == "ATR-based Candle":
                config['atr_multiplier'] = st.number_input("ATR Multiplier", min_value=0.1, value=1.0)
            
            config['use_adx'] = st.checkbox("Use ADX Filter", value=False)
            if config['use_adx']:
                config['adx_period'] = st.number_input("ADX Period", min_value=1, value=14)
                config['adx_threshold'] = st.number_input("ADX Threshold", min_value=1, value=25)
        
        elif strategy == "Price Crosses Threshold Strategy":
            st.subheader("Threshold Settings")
            config['threshold_price'] = st.number_input("Threshold Price", min_value=0.0, value=100.0)
            config['threshold_direction'] = st.selectbox("Direction", [
                "LONG (Price >= Threshold)",
                "SHORT (Price >= Threshold)",
                "LONG (Price <= Threshold)",
                "SHORT (Price <= Threshold)"
            ])
        
        elif strategy == "Percentage Change Strategy":
            st.subheader("Percentage Change Settings")
            config['pct_threshold'] = st.number_input("Percentage Threshold", min_value=0.01, value=0.5, step=0.01)
            config['pct_direction'] = st.selectbox("Direction", [
                "BUY on Fall",
                "SELL on Fall",
                "BUY on Rise",
                "SELL on Rise"
            ])
        
        elif strategy == "Custom Strategy Builder":
            st.subheader("Custom Conditions")
            
            if st.button("Add Condition"):
                st.session_state['custom_conditions'].append({
                    'use_condition': True,
                    'indicator': 'RSI',
                    'operator': '>',
                    'value': 50,
                    'action': 'BUY',
                    'compare_with_indicator': False,
                    'compare_indicator': 'EMA_20'
                })
                st.rerun()
            
            for i, cond in enumerate(st.session_state['custom_conditions']):
                with st.expander(f"Condition {i+1}", expanded=True):
                    cond['use_condition'] = st.checkbox("Use", value=cond.get('use_condition', True), key=f"use_{i}")
                    cond['compare_with_indicator'] = st.checkbox("Compare Indicator", value=cond.get('compare_with_indicator', False), key=f"cmp_{i}")
                    
                    indicators = ['Price', 'RSI', 'ADX', 'EMA_Fast', 'EMA_Slow', 'EMA_20', 'EMA_50', 
                                 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR', 'Close', 'High', 'Low']
                    
                    cond['indicator'] = st.selectbox("Indicator", indicators, key=f"ind_{i}")
                    cond['operator'] = st.selectbox("Operator", ['>', '<', '>=', '<=', '==', 'crosses_above', 'crosses_below'], key=f"op_{i}")
                    
                    if cond['compare_with_indicator']:
                        cond['compare_indicator'] = st.selectbox("Compare To", indicators, key=f"cmpind_{i}")
                    else:
                        cond['value'] = st.number_input("Value", value=float(cond.get('value', 50)), key=f"val_{i}")
                    
                    cond['action'] = st.selectbox("Action", ['BUY', 'SELL'], key=f"act_{i}")
                    
                    if st.button("Remove", key=f"rem_{i}"):
                        st.session_state['custom_conditions'].pop(i)
                        st.rerun()
            
            config['custom_conditions'] = st.session_state['custom_conditions']
            config['ema_fast'] = 9
            config['ema_slow'] = 15
        
        else:
            config['ema_fast'] = 9
            config['ema_slow'] = 15
        
        st.markdown("---")
        
        st.subheader("Stop Loss")
        config['sl_type'] = st.selectbox("SL Type", SL_TYPES)
        
        if 'Points' in config['sl_type']:
            config['sl_points'] = st.number_input("SL Points", min_value=1.0, value=10.0)
        
        if 'Trailing SL (Points)' in config['sl_type']:
            config['trailing_sl_points'] = st.number_input("Trailing SL Points", min_value=1.0, value=10.0)
        
        if 'ATR' in config['sl_type']:
            config['atr_sl_multiplier'] = st.number_input("ATR Multiplier (SL)", min_value=0.1, value=1.5, step=0.1)
        
        st.subheader("Target")
        config['target_type'] = st.selectbox("Target Type", TARGET_TYPES)
        
        if 'Points' in config['target_type']:
            config['target_points'] = st.number_input("Target Points", min_value=1.0, value=20.0)
        
        if 'ATR' in config['target_type']:
            config['atr_target_multiplier'] = st.number_input("ATR Multiplier (Target)", min_value=0.1, value=3.0, step=0.1)
        
        if 'Risk-Reward' in config['target_type']:
            config['risk_reward_ratio'] = st.number_input("Risk-Reward Ratio", min_value=0.1, value=2.0, step=0.1)
        
        st.markdown("---")
        
        # Broker Integration Settings (Optional)
        with st.expander("🔌 Broker Settings (Optional)", expanded=False):
            st.info("Configure these parameters for live broker integration")
            config['broker_enabled'] = st.checkbox("Enable Broker Integration", value=False)
            config['expiry_date'] = st.date_input("Expiry Date")
            config['strike_price'] = st.number_input("Strike Price", min_value=0.0, value=0.0, step=50.0)
            config['option_type'] = st.selectbox("Option Type", ["CE", "PE", "FUT", "EQUITY"])
            config['order_type'] = st.selectbox("Order Type", ["MARKET", "LIMIT"])
            
            if config['order_type'] == "LIMIT":
                config['limit_price'] = st.number_input("Limit Price", min_value=0.0, value=0.0)
            
            st.markdown("**Broker API Placeholder Code:**")
            st.code("""
# Example: Dhan/Zerodha Integration
# from dhanhq import dhanhq
# dhan = dhanhq(client_id, access_token)
# 
# order = dhan.place_order(
#     exchange_segment='NSE_FNO',
#     transaction_type='BUY' or 'SELL',
#     quantity=quantity,
#     order_type='MARKET' or 'LIMIT',
#     price=price,
#     strike_price=strike_price,
#     expiry_date=expiry_date,
#     option_type='CE' or 'PE'
# )
            """, language="python")
        
        st.session_state['config'] = config
    
    if mode == "Live Trading":
        render_live_trading(config)
    else:
        render_backtesting(config)

def render_live_trading(config):
    tab1, tab2, tab3 = st.tabs(["📊 Live Dashboard", "📈 Trade History", "📝 Logs"])
    
    with tab1:
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            if st.button("▶️ Start", use_container_width=True):
                st.session_state['trading_active'] = True
                st.session_state['trade_logs'].append({
                    'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')),
                    'message': 'Trading Started'
                })
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop", use_container_width=True):
                if st.session_state.get('position'):
                    position = st.session_state['position']
                    current_price = st.session_state['current_data']['Close'].iloc[-1] if st.session_state.get('current_data') is not None else position['entry_price']
                    
                    pnl = (current_price - position['entry_price']) * position['quantity'] if position['type'] == 'LONG' else (position['entry_price'] - current_price) * position['quantity']
                    
                    st.session_state['trade_history'].append({
                        'entry_time': position['entry_time'],
                        'exit_time': datetime.now(pytz.timezone('Asia/Kolkata')),
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'quantity': position['quantity'],
                        'pnl': pnl,
                        'exit_reason': 'Manual Close',
                        'sl_price': st.session_state.get('current_sl'),
                        'target_price': st.session_state.get('current_target')
                    })
                
                st.session_state['trading_active'] = False
                st.session_state['position'] = None
                st.session_state['highest_price'] = 0
                st.session_state['lowest_price'] = float('inf')
                st.session_state['partial_exit_done'] = False
                st.rerun()
        
        with col3:
            status = "🟢 ACTIVE" if st.session_state.get('trading_active', False) else "🔴 INACTIVE"
            st.markdown(f"### Status: {status}")
        
        st.markdown("---")
        
        if st.session_state.get('current_data') is not None:
            df = st.session_state['current_data']
            current_price = df['Close'].iloc[-1]
            position = st.session_state.get('position')
            
            st.metric("Current Price", f"₹{current_price:.2f}")
            
            if position:
                st.metric("Entry Price", f"₹{position['entry_price']:.2f}")
                st.metric("Position", position['type'])
                
                pnl = (current_price - position['entry_price']) * position['quantity'] if position['type'] == 'LONG' else (position['entry_price'] - current_price) * position['quantity']
                st.metric("Unrealized P&L", f"₹{pnl:.2f}", delta=f"{pnl:.2f}", delta_color="normal" if pnl >= 0 else "inverse")
            
            st.plotly_chart(create_candlestick_chart(df, live_position=position), use_container_width=True)
        else:
            st.info("Start trading to begin...")
    
    with tab2:
        if st.session_state.get('trade_history'):
            trades = st.session_state['trade_history']
            total_pnl = sum([t['pnl'] for t in trades])
            
            st.metric("Total Trades", len(trades))
            st.metric("Total P&L", f"₹{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color="normal" if total_pnl >= 0 else "inverse")
            
            for i, trade in enumerate(reversed(trades)):
                with st.expander(f"Trade {len(trades)-i} - P&L: ₹{trade['pnl']:.2f}"):
                    st.write(f"**Type:** {trade['type']}")
                    st.write(f"**Entry:** ₹{trade['entry_price']:.2f}")
                    st.write(f"**Exit:** ₹{trade['exit_price']:.2f}")
                    st.write(f"**Reason:** {trade['exit_reason']}")
        else:
            st.info("No trades yet")
    
    with tab3:
        if st.session_state.get('trade_logs'):
            for log in reversed(st.session_state['trade_logs'][-50:]):
                st.text(f"[{log['timestamp']}] {log['message']}")
        else:
            st.info("No logs")
    
    if st.session_state.get('trading_active', False):
        live_trading_iteration()

def render_backtesting(config):
    tab1, tab2 = st.tabs(["📊 Results", "📈 Analysis"])
    
    with tab1:
        if st.button("🚀 Run Backtest", use_container_width=True):
            with st.spinner("Running..."):
                df = fetch_data(config['ticker'], config['interval'], config['period'])
                
                if df is not None:
                    strategy_map = {
                        "EMA Crossover Strategy": ema_crossover_strategy,
                        "Simple Buy Strategy": simple_buy_strategy,
                        "Simple Sell Strategy": simple_sell_strategy,
                        "Price Crosses Threshold Strategy": price_crosses_threshold_strategy,
                        "RSI-ADX-EMA Strategy": rsi_adx_ema_strategy,
                        "Percentage Change Strategy": percentage_change_strategy,
                        "AI Price Action Analysis": ai_price_action_strategy,
                        "Custom Strategy Builder": custom_strategy_builder
                    }
                    
                    trades, _ = run_backtest(df, strategy_map[config['strategy']], config)
                    
                    st.session_state['backtest_trades'] = trades
                    st.success("Complete!")
        
        if st.session_state.get('backtest_trades'):
            trades = st.session_state['backtest_trades']
            total_pnl = sum([t['pnl'] for t in trades])
            wins = len([t for t in trades if t['pnl'] > 0])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Trades", len(trades))
            col2.metric("Win Rate", f"{wins/len(trades)*100:.1f}%")
            col3.metric("Total P&L", f"₹{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color="normal" if total_pnl >= 0 else "inverse")
            
            for i, trade in enumerate(trades):
                with st.expander(f"Trade {i+1} - P&L: ₹{trade['pnl']:.2f}"):
                    st.write(f"**Type:** {trade['type']}")
                    st.write(f"**Entry:** ₹{trade['entry_price']:.2f}")
                    st.write(f"**Exit:** ₹{trade['exit_price']:.2f}")
                    st.write(f"**Reason:** {trade['exit_reason']}")
    
    with tab2:
        df = fetch_data(config['ticker'], config['interval'], config['period'])
        if df is not None:
            st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
