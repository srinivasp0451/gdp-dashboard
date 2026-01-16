import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import time
import random

# ==================== CONFIGURATION ====================

ASSET_TICKERS = {
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "USDINR": "INR=X",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "Gold": "GC=F",
    "Silver": "SI=F",
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
    "1mo": ["1y", "2y", "5y", "10y", "15y", "20y", "25y", "30y"],
}

TRAILING_MIN_DISTANCES = {
    "NIFTY 50": 10,
    "BANKNIFTY": 20,
    "BTC": 150,
    "ETH": 10,
}

# ==================== UTILITY FUNCTIONS ====================

def init_session_state():
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
        'custom_conditions': [{'use': True, 'compare_price': False, 'indicator': 'RSI', 'operator': '>', 'value': 30, 'action': 'BUY'}],
        'partial_exit_done': False,
        'breakeven_activated': False,
        'last_refresh': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def add_log(message):
    """Add timestamped log entry"""
    timestamp = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    st.session_state['trade_logs'].append(log_entry)
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

# ==================== DATA FETCHING ====================

def fetch_data(ticker, interval, period, mode="Backtest"):
    """Fetch data from yfinance with proper timezone handling"""
    try:
        if mode == "Live Trading":
            time.sleep(random.uniform(1.0, 1.5))
        
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if data.empty:
            return None
        
        # Flatten multi-index columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in data.columns.values]
            data = data[[col for col in data.columns if ticker.replace('^', '') in col or not any(x in col for x in ['_', ticker])]]
            data.columns = [col.split('_')[0] if '_' in col else col for col in data.columns]
        
        # Select OHLCV
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                if col == 'Volume':
                    data[col] = 0
                else:
                    return None
        
        data = data[required_cols]
        
        # Handle timezone
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        data.index = data.index.tz_convert('Asia/Kolkata')
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# ==================== INDICATOR CALCULATIONS ====================

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
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = calculate_atr(df, 1)
    atr = tr.rolling(period).mean()
    
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    
    return adx

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, sma, lower

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_stochastic_rsi(data, period=14, smooth_k=3, smooth_d=3):
    """Calculate Stochastic RSI"""
    rsi = calculate_rsi(data, period)
    stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min()) * 100
    k = stoch_rsi.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d

def calculate_keltner_channel(df, period=20, atr_mult=2):
    """Calculate Keltner Channel"""
    middle = calculate_ema(df['Close'], period)
    atr = calculate_atr(df, period)
    upper = middle + (atr_mult * atr)
    lower = middle - (atr_mult * atr)
    return upper, middle, lower

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
    if df['Volume'].sum() == 0:
        return df['Close']
    return (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

def calculate_supertrend(df, period=10, multiplier=3):
    """Calculate SuperTrend"""
    atr = calculate_atr(df, period)
    hl_avg = (df['High'] + df['Low']) / 2
    
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > upper_band.iloc[i-1]:
            direction.iloc[i] = 1
        elif df['Close'].iloc[i] < lower_band.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
            if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_band.iloc[i-1]
            if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_band.iloc[i-1]
        
        if direction.iloc[i] == 1:
            supertrend.iloc[i] = lower_band.iloc[i]
        else:
            supertrend.iloc[i] = upper_band.iloc[i]
    
    return supertrend

def calculate_swing_levels(df, period=5):
    """Calculate swing high and low levels"""
    swing_high = df['High'].rolling(window=period, center=True).max()
    swing_low = df['Low'].rolling(window=period, center=True).min()
    return swing_high, swing_low

def calculate_ema_angle(df, ema_col, period=2):
    """Calculate EMA angle in degrees"""
    ema_diff = df[ema_col].diff(period)
    angle_rad = np.arctan(ema_diff / period)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def add_all_indicators(df, ema_fast=9, ema_slow=15, adx_period=14):
    """Add all indicators to dataframe"""
    df['EMA_Fast'] = calculate_ema(df['Close'], ema_fast)
    df['EMA_Slow'] = calculate_ema(df['Close'], ema_slow)
    df['EMA_20'] = calculate_ema(df['Close'], 20)
    df['EMA_50'] = calculate_ema(df['Close'], 50)
    df['EMA_Angle_Fast'] = calculate_ema_angle(df, 'EMA_Fast', 2)
    df['EMA_Angle_Slow'] = calculate_ema_angle(df, 'EMA_Slow', 2)
    df['ATR'] = calculate_atr(df, 14)
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['ADX'] = calculate_adx(df, adx_period)
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    df['SuperTrend'] = calculate_supertrend(df)
    df['VWAP'] = calculate_vwap(df)
    df['Keltner_Upper'], df['Keltner_Middle'], df['Keltner_Lower'] = calculate_keltner_channel(df)
    df['Swing_High'], df['Swing_Low'] = calculate_swing_levels(df)
    df['Pivot'], df['R1'], df['R2'], df['S1'], df['S2'] = calculate_pivot_points(df)
    
    return df

# ==================== STRATEGY IMPLEMENTATIONS ====================

def check_ema_crossover(df, i, config):
    """Check EMA crossover with filters"""
    if i < 1:
        return 0, None, None, None
    
    ema_fast = df['EMA_Fast'].iloc[i]
    ema_slow = df['EMA_Slow'].iloc[i]
    prev_ema_fast = df['EMA_Fast'].iloc[i-1]
    prev_ema_slow = df['EMA_Slow'].iloc[i-1]
    
    # Check bullish crossover
    bullish_cross = (ema_fast > ema_slow) and (prev_ema_fast <= prev_ema_slow)
    # Check bearish crossover
    bearish_cross = (ema_fast < ema_slow) and (prev_ema_fast >= prev_ema_slow)
    
    if not bullish_cross and not bearish_cross:
        return 0, None, None, None
    
    # Check angle
    angle = abs(df['EMA_Angle_Fast'].iloc[i])
    if angle < config['min_angle']:
        return 0, None, None, None
    
    # Check entry filter
    entry_filter = config['entry_filter']
    candle_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
    
    if entry_filter == "Custom Candle (Points)":
        if candle_size < config['custom_points']:
            return 0, None, None, None
    elif entry_filter == "ATR-based Candle":
        atr = df['ATR'].iloc[i]
        if candle_size < (atr * config['atr_multiplier']):
            return 0, None, None, None
    
    # Check ADX filter
    if config['use_adx']:
        adx = df['ADX'].iloc[i]
        if adx < config['adx_threshold']:
            return 0, None, None, None
    
    # Determine signal
    if bullish_cross:
        entry_price = df['Close'].iloc[i]
        sl, target = calculate_sl_target(entry_price, 1, df, i, config)
        return 1, entry_price, sl, target
    elif bearish_cross:
        entry_price = df['Close'].iloc[i]
        sl, target = calculate_sl_target(entry_price, -1, df, i, config)
        return -1, entry_price, sl, target
    
    return 0, None, None, None

def check_simple_buy(df, i, config):
    """Simple buy at candle close"""
    entry_price = df['Close'].iloc[i]
    sl, target = calculate_sl_target(entry_price, 1, df, i, config)
    return 1, entry_price, sl, target

def check_simple_sell(df, i, config):
    """Simple sell at candle close"""
    entry_price = df['Close'].iloc[i]
    sl, target = calculate_sl_target(entry_price, -1, df, i, config)
    return -1, entry_price, sl, target

def check_price_threshold(df, i, config):
    """Check price threshold strategy"""
    current_price = df['Close'].iloc[i]
    threshold = config['threshold_price']
    direction = config['threshold_direction']
    
    signal = 0
    
    if direction == "LONG (Price >= Threshold)":
        if current_price >= threshold:
            signal = 1
    elif direction == "SHORT (Price >= Threshold)":
        if current_price >= threshold:
            signal = -1
    elif direction == "LONG (Price <= Threshold)":
        if current_price <= threshold:
            signal = 1
    elif direction == "SHORT (Price <= Threshold)":
        if current_price <= threshold:
            signal = -1
    
    if signal != 0:
        entry_price = df['Close'].iloc[i]
        sl, target = calculate_sl_target(entry_price, signal, df, i, config)
        return signal, entry_price, sl, target
    
    return 0, None, None, None

def check_rsi_adx_ema(df, i, config):
    """RSI-ADX-EMA Strategy"""
    if i < 1:
        return 0, None, None, None
    
    rsi = df['RSI'].iloc[i]
    adx = df['ADX'].iloc[i]
    ema_fast = df['EMA_Fast'].iloc[i]
    ema_slow = df['EMA_Slow'].iloc[i]
    
    # SELL: RSI>80 AND ADX<20 AND EMA1<EMA2
    if rsi > 80 and adx < 20 and ema_fast < ema_slow:
        entry_price = df['Close'].iloc[i]
        sl, target = calculate_sl_target(entry_price, -1, df, i, config)
        return -1, entry_price, sl, target
    
    # BUY: RSI<20 AND ADX>20 AND EMA1>EMA2
    if rsi < 20 and adx > 20 and ema_fast > ema_slow:
        entry_price = df['Close'].iloc[i]
        sl, target = calculate_sl_target(entry_price, 1, df, i, config)
        return 1, entry_price, sl, target
    
    return 0, None, None, None

def ai_price_action_analysis(df, i, config):
    """AI-based price action analysis"""
    if i < 50:
        return 0, None, None, None, ""
    
    close = df['Close'].iloc[i]
    rsi = df['RSI'].iloc[i]
    macd = df['MACD'].iloc[i]
    macd_signal = df['MACD_Signal'].iloc[i]
    bb_upper = df['BB_Upper'].iloc[i]
    bb_lower = df['BB_Lower'].iloc[i]
    ema_20 = df['EMA_20'].iloc[i]
    ema_50 = df['EMA_50'].iloc[i]
    volume = df['Volume'].iloc[i]
    avg_volume = df['Volume'].iloc[i-20:i].mean()
    
    score = 0
    analysis = []
    
    # Trend Analysis
    if ema_20 > ema_50:
        score += 2
        analysis.append("✅ Trend: Bullish (EMA20 > EMA50)")
    else:
        score -= 2
        analysis.append("❌ Trend: Bearish (EMA20 < EMA50)")
    
    # RSI Analysis
    if rsi < 30:
        score += 2
        analysis.append("✅ RSI: Oversold (Buy signal)")
    elif rsi > 70:
        score -= 2
        analysis.append("❌ RSI: Overbought (Sell signal)")
    else:
        analysis.append("➖ RSI: Neutral")
    
    # MACD Analysis
    if macd > macd_signal:
        score += 1
        analysis.append("✅ MACD: Bullish crossover")
    else:
        score -= 1
        analysis.append("❌ MACD: Bearish crossover")
    
    # Bollinger Bands
    if close < bb_lower:
        score += 1
        analysis.append("✅ BB: Near lower band (Buy signal)")
    elif close > bb_upper:
        score -= 1
        analysis.append("❌ BB: Near upper band (Sell signal)")
    else:
        analysis.append("➖ BB: Mid-range")
    
    # Volume Analysis (skip for indices)
    if volume > 0:
        if volume > avg_volume * 1.5:
            score += 1
            analysis.append("✅ Volume: High (Strong signal)")
        else:
            analysis.append("➖ Volume: Normal")
    else:
        analysis.append("➖ Volume: N/A (Index)")
    
    # Generate signal
    signal = 0
    confidence = abs(score) / 8 * 100
    analysis_text = "\n".join(analysis)
    
    if score >= 3:
        signal = 1
        entry_price = close
        atr = df['ATR'].iloc[i]
        sl = entry_price - (1.5 * atr)
        target = entry_price + (3 * atr)
        analysis_text = f"🔵 BUY Signal (Confidence: {confidence:.1f}%)\n\n{analysis_text}\n\nAuto SL: {sl:.2f} (1.5×ATR)\nAuto Target: {target:.2f} (3×ATR)"
        return signal, entry_price, sl, target, analysis_text
    elif score <= -3:
        signal = -1
        entry_price = close
        atr = df['ATR'].iloc[i]
        sl = entry_price + (1.5 * atr)
        target = entry_price - (3 * atr)
        analysis_text = f"🔴 SELL Signal (Confidence: {confidence:.1f}%)\n\n{analysis_text}\n\nAuto SL: {sl:.2f} (1.5×ATR)\nAuto Target: {target:.2f} (3×ATR)"
        return signal, entry_price, sl, target, analysis_text
    
    analysis_text = f"⚪ NEUTRAL (Score: {score}, Confidence: {confidence:.1f}%)\n\n{analysis_text}"
    return 0, None, None, None, analysis_text

def check_percentage_change(df, i, config):
    """Check percentage change strategy"""
    if i < 1:
        return 0, None, None, None
    
    # Calculate percentage change from first candle to current
    first_price = df['Close'].iloc[0]
    current_price = df['Close'].iloc[i]
    pct_change = ((current_price - first_price) / first_price) * 100
    
    threshold_pct = config['pct_threshold']
    direction = config['pct_direction']
    
    signal = 0
    
    if direction == "BUY on Fall":
        # Buy if price has fallen by threshold percentage or more
        if pct_change <= -threshold_pct:
            signal = 1
    elif direction == "SELL on Fall":
        # Sell if price has fallen by threshold percentage or more
        if pct_change <= -threshold_pct:
            signal = -1
    elif direction == "BUY on Rise":
        # Buy if price has risen by threshold percentage or more
        if pct_change >= threshold_pct:
            signal = 1
    elif direction == "SELL on Rise":
        # Sell if price has risen by threshold percentage or more
        if pct_change >= threshold_pct:
            signal = -1
    
    if signal != 0:
        entry_price = df['Close'].iloc[i]
        sl, target = calculate_sl_target(entry_price, signal, df, i, config)
        return signal, entry_price, sl, target
    
    return 0, None, None, None

def check_custom_strategy(df, i, conditions):
    """Check custom strategy conditions"""
    if i < 1:
        return 0, None, None, None
    
    buy_conditions = [c for c in conditions if c['use'] and c['action'] == 'BUY']
    sell_conditions = [c for c in conditions if c['use'] and c['action'] == 'SELL']
    
    def evaluate_condition(cond):
        if cond['compare_price']:
            left_val = df['Close'].iloc[i]
            right_val = df[cond['compare_indicator']].iloc[i] if cond['compare_indicator'] in df.columns else 0
        else:
            if cond['indicator'] == 'Price':
                left_val = df['Close'].iloc[i]
            else:
                left_val = df[cond['indicator']].iloc[i] if cond['indicator'] in df.columns else 0
            right_val = cond['value']
        
        if cond['indicator'] == 'Price':
            prev_left = df['Close'].iloc[i-1]
        else:
            prev_left = df[cond['indicator']].iloc[i-1] if cond['indicator'] in df.columns else 0
        prev_right = df[cond['compare_indicator']].iloc[i-1] if cond['compare_price'] and cond['compare_indicator'] in df.columns else cond['value']
        
        op = cond['operator']
        if op == '>':
            return left_val > right_val
        elif op == '<':
            return left_val < right_val
        elif op == '>=':
            return left_val >= right_val
        elif op == '<=':
            return left_val <= right_val
        elif op == '==':
            return abs(left_val - right_val) < 0.01
        elif op == 'crosses_above':
            return (left_val > right_val) and (prev_left <= prev_right)
        elif op == 'crosses_below':
            return (left_val < right_val) and (prev_left >= prev_right)
        return False
    
    # Check BUY
    if buy_conditions and all(evaluate_condition(c) for c in buy_conditions):
        return 1, df['Close'].iloc[i], None, None
    
    # Check SELL
    if sell_conditions and all(evaluate_condition(c) for c in sell_conditions):
        return -1, df['Close'].iloc[i], None, None
    
    return 0, None, None, None

# ==================== SL/TARGET CALCULATIONS ====================

def calculate_sl_target(entry_price, signal, df, i, config):
    """Calculate initial SL and Target"""
    sl_type = config['sl_type']
    target_type = config['target_type']
    
    # Calculate SL
    sl = None
    if sl_type == "Custom Points":
        if signal == 1:
            sl = entry_price - config['sl_points']
        else:
            sl = entry_price + config['sl_points']
    
    elif sl_type.startswith("Trailing SL"):
        if signal == 1:
            sl = entry_price - config['sl_points']
        else:
            sl = entry_price + config['sl_points']
    
    elif sl_type == "ATR-based":
        atr = df['ATR'].iloc[i]
        if signal == 1:
            sl = entry_price - (atr * config.get('atr_sl_mult', 1.5))
        else:
            sl = entry_price + (atr * config.get('atr_sl_mult', 1.5))
    
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
    
    elif sl_type == "Current Swing Low/High":
        if signal == 1:
            sl = df['Swing_Low'].iloc[i]
        else:
            sl = df['Swing_High'].iloc[i]
    
    elif sl_type == "Previous Swing Low/High":
        if i > 0:
            if signal == 1:
                sl = df['Swing_Low'].iloc[i-1]
            else:
                sl = df['Swing_High'].iloc[i-1]
    
    elif sl_type == "Signal-based (reverse EMA crossover)":
        sl = 0  # Signal-based, no fixed SL
    
    elif sl_type == "Break-even After 50% Target":
        if signal == 1:
            sl = entry_price - config['sl_points']
        else:
            sl = entry_price + config['sl_points']
    
    elif sl_type == "Volatility-Adjusted Trailing SL":
        atr = df['ATR'].iloc[i]
        if signal == 1:
            sl = entry_price - (atr * config.get('atr_sl_mult', 2))
        else:
            sl = entry_price + (atr * config.get('atr_sl_mult', 2))
    
    # Calculate Target
    target = None
    if target_type == "Custom Points":
        if signal == 1:
            target = entry_price + config['target_points']
        else:
            target = entry_price - config['target_points']
    
    elif target_type.startswith("Trailing Target"):
        if signal == 1:
            target = entry_price + config['target_points']
        else:
            target = entry_price - config['target_points']
    
    elif target_type == "ATR-based":
        atr = df['ATR'].iloc[i]
        if signal == 1:
            target = entry_price + (atr * config.get('atr_target_mult', 3))
        else:
            target = entry_price - (atr * config.get('atr_target_mult', 3))
    
    elif target_type == "Risk-Reward Based":
        rr_ratio = config.get('rr_ratio', 2)
        sl_dist = abs(entry_price - sl) if sl else config['sl_points']
        if signal == 1:
            target = entry_price + (sl_dist * rr_ratio)
        else:
            target = entry_price - (sl_dist * rr_ratio)
    
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
    
    elif target_type == "Current Swing Low/High":
        if signal == 1:
            target = df['Swing_High'].iloc[i]
        else:
            target = df['Swing_Low'].iloc[i]
    
    elif target_type == "Previous Swing Low/High":
        if i > 0:
            if signal == 1:
                target = df['Swing_High'].iloc[i-1]
            else:
                target = df['Swing_Low'].iloc[i-1]
    
    elif target_type == "Signal-based (reverse EMA crossover)":
        target = 0  # Signal-based, no fixed target
    
    elif target_type == "50% Exit at Target (Partial)":
        if signal == 1:
            target = entry_price + config['target_points']
        else:
            target = entry_price - config['target_points']
    
    return sl, target

def update_trailing_sl(position, current_price, config):
    """Update trailing stop loss"""
    sl_type = config['sl_type']
    signal = position['signal']
    current_sl = position['sl']
    
    if "Trailing SL (Points)" in sl_type and "Signal Based" not in sl_type:
        points = config['sl_points']
        
        if signal == 1:  # LONG
            new_sl = current_price - points
            if new_sl > current_sl:
                return new_sl
        else:  # SHORT
            new_sl = current_price + points
            if new_sl < current_sl:
                return new_sl
    
    elif "Volatility-Adjusted Trailing SL" in sl_type:
        points = config.get('sl_points', 10)
        if signal == 1:
            new_sl = current_price - points
            if new_sl > current_sl:
                return new_sl
        else:
            new_sl = current_price + points
            if new_sl < current_sl:
                return new_sl
    
    return current_sl

def check_exit_conditions(position, current_price, df, i, config):
    """Check if position should exit"""
    signal = position['signal']
    entry_price = position['entry_price']
    sl = position['sl']
    target = position['target']
    sl_type = config['sl_type']
    target_type = config['target_type']
    
    # Check SL hit
    if sl and sl != 0:
        if signal == 1 and current_price <= sl:
            return True, "SL Hit", sl
        elif signal == -1 and current_price >= sl:
            return True, "SL Hit", sl
    
    # Check Target hit
    if target and target != 0:
        if "Trailing Target" not in target_type:
            if signal == 1 and current_price >= target:
                return True, "Target Hit", target
            elif signal == -1 and current_price <= target:
                return True, "Target Hit", target
    
    # Check Signal-based exit
    if "Signal-based" in sl_type or "Signal-based" in target_type:
        if i > 0:
            ema_fast = df['EMA_Fast'].iloc[i]
            ema_slow = df['EMA_Slow'].iloc[i]
            prev_ema_fast = df['EMA_Fast'].iloc[i-1]
            prev_ema_slow = df['EMA_Slow'].iloc[i-1]
            
            if signal == 1:  # LONG - check bearish crossover
                if (ema_fast < ema_slow) and (prev_ema_fast >= prev_ema_slow):
                    return True, "Reverse Signal - Bearish Crossover", current_price
            elif signal == -1:  # SHORT - check bullish crossover
                if (ema_fast > ema_slow) and (prev_ema_fast <= prev_ema_slow):
                    return True, "Reverse Signal - Bullish Crossover", current_price
    
    return False, None, None

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
    }
    
    position = None
    
    for i in range(50, len(df)):
        current_price = df['Close'].iloc[i]
        
        # Check if in position
        if position:
            # Update trailing levels
            if position['highest_price'] is None or current_price > position['highest_price']:
                position['highest_price'] = current_price
            if position['lowest_price'] is None or current_price < position['lowest_price']:
                position['lowest_price'] = current_price
            
            # Update trailing SL
            if "Trailing" in config['sl_type']:
                position['sl'] = update_trailing_sl(position, current_price, config)
            
            # Check break-even
            if config['sl_type'] == "Break-even After 50% Target" and not position.get('breakeven_activated', False):
                if position['signal'] == 1:
                    profit_dist = current_price - position['entry_price']
                    target_dist = position['target'] - position['entry_price']
                    if profit_dist >= target_dist * 0.5:
                        position['sl'] = position['entry_price']
                        position['breakeven_activated'] = True
                elif position['signal'] == -1:
                    profit_dist = position['entry_price'] - current_price
                    target_dist = position['entry_price'] - position['target']
                    if profit_dist >= target_dist * 0.5:
                        position['sl'] = position['entry_price']
                        position['breakeven_activated'] = True
            
            # Check partial exit
            if config['target_type'] == "50% Exit at Target (Partial)" and not position.get('partial_exit_done', False):
                if position['signal'] == 1 and current_price >= position['target']:
                    position['partial_exit_done'] = True
                elif position['signal'] == -1 and current_price <= position['target']:
                    position['partial_exit_done'] = True
            
            # Check exit
            should_exit, exit_reason, exit_price = check_exit_conditions(position, current_price, df, i, config)
            
            if should_exit:
                # Calculate PnL
                quantity = config['quantity']
                if position['signal'] == 1:
                    pnl = (exit_price - position['entry_price']) * quantity
                else:
                    pnl = (position['entry_price'] - exit_price) * quantity
                
                # Adjust for partial exit
                if position.get('partial_exit_done', False):
                    pnl = pnl * 0.5
                
                # Record trade
                duration = (df.index[i] - position['entry_time']).total_seconds() / 3600
                trade_record = {
                    'entry_time': position['entry_time'],
                    'exit_time': df.index[i],
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
                    'range': position['highest_price'] - position['lowest_price'],
                }
                
                results['trades'].append(trade_record)
                results['total_trades'] += 1
                results['total_pnl'] += pnl
                
                if pnl > 0:
                    results['winning_trades'] += 1
                else:
                    results['losing_trades'] += 1
                
                position = None
        
        # Check for entry
        if not position:
            signal = 0
            entry_price = None
            sl = None
            target = None
            
            if strategy == "EMA Crossover Strategy":
                signal, entry_price, sl, target = check_ema_crossover(df, i, config)
            elif strategy == "Simple Buy Strategy":
                signal, entry_price, sl, target = check_simple_buy(df, i, config)
            elif strategy == "Simple Sell Strategy":
                signal, entry_price, sl, target = check_simple_sell(df, i, config)
            elif strategy == "Price Crosses Threshold":
                signal, entry_price, sl, target = check_price_threshold(df, i, config)
            elif strategy == "RSI-ADX-EMA Strategy":
                signal, entry_price, sl, target = check_rsi_adx_ema(df, i, config)
            elif strategy == "AI Price Action Analysis":
                signal, entry_price, sl, target, _ = ai_price_action_analysis(df, i, config)
            elif strategy == "Percentage Change Strategy":
                signal, entry_price, sl, target = check_percentage_change(df, i, config)
            elif strategy == "Custom Strategy Builder":
                signal, entry_price, sl, target = check_custom_strategy(df, i, st.session_state['custom_conditions'])
                if signal != 0 and (sl is None or target is None):
                    sl, target = calculate_sl_target(entry_price, signal, df, i, config)
            
            if signal != 0:
                position = {
                    'signal': signal,
                    'entry_price': entry_price,
                    'entry_time': df.index[i],
                    'sl': sl,
                    'target': target,
                    'highest_price': entry_price,
                    'lowest_price': entry_price,
                    'partial_exit_done': False,
                    'breakeven_activated': False,
                }
    
    # Calculate accuracy
    if results['total_trades'] > 0:
        results['accuracy'] = (results['winning_trades'] / results['total_trades']) * 100
    
    # Calculate average duration
    if results['trades']:
        avg_duration = sum(t['duration'] for t in results['trades']) / len(results['trades'])
        results['avg_duration'] = avg_duration
    else:
        results['avg_duration'] = 0
    
    return results

# ==================== LIVE TRADING ENGINE ====================

def live_trading_loop(config):
    """Main live trading loop"""
    while st.session_state['trading_active']:
        # Fetch latest data
        ticker = ASSET_TICKERS.get(config['asset'], config['custom_ticker'])
        df = fetch_data(ticker, config['interval'], config['period'], mode="Live Trading")
        
        if df is None or len(df) < 50:
            add_log("Failed to fetch data")
            time.sleep(2)
            continue
        
        # Add indicators
        df = add_all_indicators(df, config.get('ema_fast', 9), config.get('ema_slow', 15), config.get('adx_period', 14))
        st.session_state['current_data'] = df
        
        i = len(df) - 1
        current_price = df['Close'].iloc[i]
        
        # Update last refresh
        st.session_state['last_refresh'] = datetime.now(pytz.timezone('Asia/Kolkata'))
        
        # Check if in position
        if st.session_state['position']:
            position = st.session_state['position']
            
            # Update highest/lowest
            if st.session_state['highest_price'] is None or current_price > st.session_state['highest_price']:
                st.session_state['highest_price'] = current_price
            if st.session_state['lowest_price'] is None or current_price < st.session_state['lowest_price']:
                st.session_state['lowest_price'] = current_price
            
            # Update trailing SL
            if "Trailing" in config['sl_type']:
                new_sl = update_trailing_sl(position, current_price, config)
                if new_sl != position['sl']:
                    position['sl'] = new_sl
                    st.session_state['position'] = position
                    add_log(f"Trailing SL updated to {new_sl:.2f}")
            
            # Check break-even
            if config['sl_type'] == "Break-even After 50% Target" and not st.session_state['breakeven_activated']:
                if position['signal'] == 1:
                    profit_dist = current_price - position['entry_price']
                    target_dist = position['target'] - position['entry_price']
                    if profit_dist >= target_dist * 0.5:
                        position['sl'] = position['entry_price']
                        st.session_state['position'] = position
                        st.session_state['breakeven_activated'] = True
                        add_log("Break-even activated - SL moved to entry")
                elif position['signal'] == -1:
                    profit_dist = position['entry_price'] - current_price
                    target_dist = position['entry_price'] - position['target']
                    if profit_dist >= target_dist * 0.5:
                        position['sl'] = position['entry_price']
                        st.session_state['position'] = position
                        st.session_state['breakeven_activated'] = True
                        add_log("Break-even activated - SL moved to entry")
            
            # Check partial exit
            if config['target_type'] == "50% Exit at Target (Partial)" and not st.session_state['partial_exit_done']:
                if position['signal'] == 1 and current_price >= position['target']:
                    st.session_state['partial_exit_done'] = True
                    add_log("50% position exited at target - trailing remaining")
                elif position['signal'] == -1 and current_price <= position['target']:
                    st.session_state['partial_exit_done'] = True
                    add_log("50% position exited at target - trailing remaining")
            
            # Check exit
            should_exit, exit_reason, exit_price = check_exit_conditions(position, current_price, df, i, config)
            
            if should_exit:
                # Calculate PnL
                quantity = config['quantity']
                if position['signal'] == 1:
                    pnl = (exit_price - position['entry_price']) * quantity
                else:
                    pnl = (position['entry_price'] - exit_price) * quantity
                
                # Adjust for partial exit
                if st.session_state['partial_exit_done']:
                    pnl = pnl * 0.5
                
                # Record trade
                duration = (datetime.now(pytz.timezone('Asia/Kolkata')) - position['entry_time']).total_seconds() / 3600
                trade_record = {
                    'entry_time': position['entry_time'],
                    'exit_time': datetime.now(pytz.timezone('Asia/Kolkata')),
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
                    'range': st.session_state['highest_price'] - st.session_state['lowest_price'],
                }
                
                st.session_state['trade_history'].append(trade_record)
                st.session_state['trade_history'] = st.session_state['trade_history']
                
                add_log(f"Position closed: {exit_reason} | PnL: {pnl:.2f}")
                
                # Placeholder for Dhan order
                # if position['signal'] == 1:
                #     dhan.place_order(security_id=..., exchange_segment=..., transaction_type=dhan.SELL, ...)
                # else:
                #     dhan.place_order(security_id=..., exchange_segment=..., transaction_type=dhan.BUY, ...)
                
                reset_position_state()
        
        # Check for entry signal
        if not st.session_state['position']:
            signal = 0
            entry_price = None
            sl = None
            target = None
            
            if config['strategy'] == "EMA Crossover Strategy":
                signal, entry_price, sl, target = check_ema_crossover(df, i, config)
            elif config['strategy'] == "Simple Buy Strategy":
                signal, entry_price, sl, target = check_simple_buy(df, i, config)
            elif config['strategy'] == "Simple Sell Strategy":
                signal, entry_price, sl, target = check_simple_sell(df, i, config)
            elif config['strategy'] == "Price Crosses Threshold":
                signal, entry_price, sl, target = check_price_threshold(df, i, config)
            elif config['strategy'] == "RSI-ADX-EMA Strategy":
                signal, entry_price, sl, target = check_rsi_adx_ema(df, i, config)
            elif config['strategy'] == "AI Price Action Analysis":
                signal, entry_price, sl, target, _ = ai_price_action_analysis(df, i, config)
            elif config['strategy'] == "Percentage Change Strategy":
                signal, entry_price, sl, target = check_percentage_change(df, i, config)
            elif config['strategy'] == "Custom Strategy Builder":
                signal, entry_price, sl, target = check_custom_strategy(df, i, st.session_state['custom_conditions'])
                if signal != 0 and (sl is None or target is None):
                    sl, target = calculate_sl_target(entry_price, signal, df, i, config)
            
            if signal != 0:
                position = {
                    'signal': signal,
                    'entry_price': entry_price,
                    'entry_time': datetime.now(pytz.timezone('Asia/Kolkata')),
                    'sl': sl,
                    'target': target,
                }
                
                st.session_state['position'] = position
                st.session_state['highest_price'] = entry_price
                st.session_state['lowest_price'] = entry_price
                
                signal_text = "LONG" if signal == 1 else "SHORT"
                add_log(f"{signal_text} entry at {entry_price:.2f} | SL: {sl:.2f} | Target: {target:.2f}")
                
                # Placeholder for Dhan order
                # if signal == 1:
                #     dhan.place_order(security_id=..., exchange_segment=..., transaction_type=dhan.BUY, ...)
                # else:
                #     dhan.place_order(security_id=..., exchange_segment=..., transaction_type=dhan.SELL, ...)
        
        time.sleep(random.uniform(1.0, 1.5))
        st.rerun()

# ==================== MAIN APP ====================

def main():
    st.set_page_config(page_title="Quantitative Trading System", layout="wide", initial_sidebar_state="expanded")
    
    init_session_state()
    
    st.title("🚀 Professional Quantitative Trading System")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode Selection
        mode = st.selectbox("Mode", ["Backtest", "Live Trading"], key="mode_select")
        
        # Asset Selection
        asset = st.selectbox("Asset", list(ASSET_TICKERS.keys()) + ["Custom"], key="asset_select")
        custom_ticker = ""
        if asset == "Custom":
            custom_ticker = st.text_input("Custom Ticker", value="AAPL", key="custom_ticker_input")
        
        # Interval Selection
        interval = st.selectbox("Interval", list(INTERVAL_PERIODS.keys()), key="interval_select")
        
        # Period Selection
        available_periods = INTERVAL_PERIODS[interval]
        period = st.selectbox("Period", available_periods, key="period_select")
        
        # Quantity
        quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="quantity_input")
        
        # Strategy Selection
        strategy = st.selectbox("Strategy", [
            "EMA Crossover Strategy",
            "Simple Buy Strategy",
            "Simple Sell Strategy",
            "Price Crosses Threshold",
            "RSI-ADX-EMA Strategy",
            "Percentage Change Strategy",
            "AI Price Action Analysis",
            "Custom Strategy Builder"
        ], key="strategy_select")
        
        # Strategy-specific parameters
        config = {
            'asset': asset,
            'custom_ticker': custom_ticker,
            'interval': interval,
            'period': period,
            'quantity': quantity,
            'strategy': strategy,
            'mode': mode,
        }
        
        # EMA Crossover Parameters
        if strategy == "EMA Crossover Strategy":
            st.subheader("EMA Parameters")
            config['ema_fast'] = st.number_input("EMA Fast", min_value=1, value=9, step=1, key="ema_fast_input")
            config['ema_slow'] = st.number_input("EMA Slow", min_value=1, value=15, step=1, key="ema_slow_input")
            config['min_angle'] = st.number_input("Min Angle (degrees)", min_value=0.0, value=1.0, step=0.1, key="min_angle_input")
            
            st.subheader("Entry Filter")
            config['entry_filter'] = st.selectbox("Entry Filter", [
                "Simple Crossover",
                "Custom Candle (Points)",
                "ATR-based Candle"
            ], key="entry_filter_select")
            
            if config['entry_filter'] == "Custom Candle (Points)":
                config['custom_points'] = st.number_input("Custom Points", min_value=1.0, value=10.0, step=1.0, key="custom_points_input")
            elif config['entry_filter'] == "ATR-based Candle":
                config['atr_multiplier'] = st.number_input("ATR Multiplier", min_value=0.1, value=1.0, step=0.1, key="atr_mult_input")
            
            st.subheader("ADX Filter")
            config['use_adx'] = st.checkbox("Use ADX Filter", value=False, key="use_adx_check")
            if config['use_adx']:
                config['adx_period'] = st.number_input("ADX Period", min_value=1, value=14, step=1, key="adx_period_input")
                config['adx_threshold'] = st.number_input("ADX Threshold", min_value=1, value=25, step=1, key="adx_thresh_input")
        else:
            config['ema_fast'] = 9
            config['ema_slow'] = 15
            config['min_angle'] = 1.0
            config['entry_filter'] = "Simple Crossover"
            config['use_adx'] = False
            config['adx_period'] = 14
            config['adx_threshold'] = 25
        
        # Price Threshold Parameters
        if strategy == "Price Crosses Threshold":
            st.subheader("Threshold Parameters")
            config['threshold_price'] = st.number_input("Threshold Price", min_value=0.0, value=100.0, step=1.0, key="threshold_price_input")
            config['threshold_direction'] = st.selectbox("Direction", [
                "LONG (Price >= Threshold)",
                "SHORT (Price >= Threshold)",
                "LONG (Price <= Threshold)",
                "SHORT (Price <= Threshold)"
            ], key="threshold_dir_select")
        
        # Percentage Change Parameters
        if strategy == "Percentage Change Strategy":
            st.subheader("Percentage Change Parameters")
            config['pct_threshold'] = st.number_input("Percentage Threshold (%)", min_value=0.01, value=0.5, step=0.01, key="pct_threshold_input")
            config['pct_direction'] = st.selectbox("Direction", [
                "BUY on Fall",
                "SELL on Fall",
                "BUY on Rise",
                "SELL on Rise"
            ], key="pct_dir_select")
        
        # Stop Loss Configuration
        st.subheader("Stop Loss Configuration")
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
        ], key="sl_type_select")
        
        if "Points" in config['sl_type'] or "Trailing" in config['sl_type'] or config['sl_type'] == "Break-even After 50% Target":
            config['sl_points'] = st.number_input("SL Points", min_value=1.0, value=10.0, step=1.0, key="sl_points_input")
        else:
            config['sl_points'] = 10.0
        
        if config['sl_type'] == "ATR-based":
            config['atr_sl_mult'] = st.number_input("ATR SL Multiplier", min_value=0.1, value=1.5, step=0.1, key="atr_sl_mult_input")
        
        if "Trailing" in config['sl_type']:
            config['sl_threshold'] = st.number_input("Trailing Threshold (Points)", min_value=0.0, value=0.0, step=1.0, key="sl_thresh_input")
        
        # Target Configuration
        st.subheader("Target Configuration")
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
        ], key="target_type_select")
        
        if "Points" in config['target_type'] or "Trailing" in config['target_type'] or config['target_type'] == "50% Exit at Target (Partial)":
            config['target_points'] = st.number_input("Target Points", min_value=1.0, value=20.0, step=1.0, key="target_points_input")
        else:
            config['target_points'] = 20.0
        
        if config['target_type'] == "ATR-based":
            config['atr_target_mult'] = st.number_input("ATR Target Multiplier", min_value=0.1, value=3.0, step=0.1, key="atr_target_mult_input")
        
        if config['target_type'] == "Risk-Reward Based":
            config['rr_ratio'] = st.number_input("Risk:Reward Ratio", min_value=0.1, value=2.0, step=0.1, key="rr_ratio_input")
    
    # Main Content Tabs
    if mode == "Backtest":
        tabs = st.tabs(["📊 Backtest Results", "📈 Trade History", "📝 Trade Logs"])
        
        with tabs[0]:
            st.markdown("### 📊 Backtest Results")
            
            if strategy == "Custom Strategy Builder":
                st.subheader("Custom Conditions")
                
                for idx, cond in enumerate(st.session_state['custom_conditions']):
                    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 2, 2, 2, 1])
                    
                    with col1:
                        cond['use'] = st.checkbox("Use", value=cond['use'], key=f"cond_use_{idx}")
                    
                    with col2:
                        cond['compare_price'] = st.checkbox("Price vs", value=cond.get('compare_price', False), key=f"cond_compare_{idx}")
                    
                    with col3:
                        if cond['compare_price']:
                            cond['compare_indicator'] = st.selectbox("Indicator", [
                                'EMA_Fast', 'EMA_Slow', 'EMA_20', 'EMA_50', 'SuperTrend', 'VWAP', 'BB_Upper', 'BB_Lower'
                            ], key=f"cond_comp_ind_{idx}")
                        else:
                            cond['indicator'] = st.selectbox("Indicator", [
                                'Price', 'RSI', 'ADX', 'EMA_Fast', 'EMA_Slow', 'SuperTrend', 'EMA_20', 'EMA_50',
                                'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR', 'Volume', 'VWAP', 'Close', 'High', 'Low'
                            ], index=['Price', 'RSI', 'ADX'].index(cond.get('indicator', 'RSI')), key=f"cond_ind_{idx}")
                    
                    with col4:
                        cond['operator'] = st.selectbox("Operator", ['>', '<', '>=', '<=', '==', 'crosses_above', 'crosses_below'],
                                                         index=['>', '<', '>=', '<=', '==', 'crosses_above', 'crosses_below'].index(cond.get('operator', '>')), key=f"cond_op_{idx}")
                    
                    with col5:
                        if cond['compare_price']:
                            st.text("(Price comparison)")
                        else:
                            cond['value'] = st.number_input("Value", value=float(cond.get('value', 30)), key=f"cond_val_{idx}")
                    
                    with col6:
                        cond['action'] = st.selectbox("Action", ['BUY', 'SELL'], index=['BUY', 'SELL'].index(cond.get('action', 'BUY')), key=f"cond_act_{idx}")
                
                col_add, col_remove = st.columns(2)
                with col_add:
                    if st.button("➕ Add Condition", key="add_cond_btn"):
                        st.session_state['custom_conditions'].append({
                            'use': True, 'compare_price': False, 'indicator': 'RSI', 'operator': '>', 'value': 30, 'action': 'BUY'
                        })
                        st.rerun()
                
                with col_remove:
                    if len(st.session_state['custom_conditions']) > 1:
                        if st.button("➖ Remove Last", key="remove_cond_btn"):
                            st.session_state['custom_conditions'].pop()
                            st.rerun()
            
            st.markdown("---")
            
            if st.button("🔄 Run Backtest", type="primary", key="run_backtest_btn"):
                with st.spinner("Running backtest..."):
                    # Clear cache
                    st.session_state['current_data'] = None
                    
                    # Fetch data
                    ticker = ASSET_TICKERS.get(config['asset'], config['custom_ticker'])
                    df = fetch_data(ticker, config['interval'], config['period'], mode="Backtest")
                    
                    if df is None or len(df) < 50:
                        st.error("Unable to fetch data. Please check your settings.")
                    else:
                        # Add indicators
                        df = add_all_indicators(df, config.get('ema_fast', 9), config.get('ema_slow', 15), config.get('adx_period', 14))
                        
                        # Run backtest
                        results = run_backtest(df, config['strategy'], config)
                        
                        # Display metrics
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric("Total Trades", results['total_trades'])
                        with col2:
                            st.metric("Winning Trades", results['winning_trades'])
                        with col3:
                            st.metric("Losing Trades", results['losing_trades'])
                        with col4:
                            accuracy_val = results['accuracy']
                            st.metric("Accuracy (%)", f"{accuracy_val:.2f}")
                        with col5:
                            pnl = results['total_pnl']
                            if pnl >= 0:
                                st.metric("Total P&L", f"{pnl:.2f}", delta=f"+{pnl:.2f}")
                            else:
                                st.metric("Total P&L", f"{pnl:.2f}", delta=f"{pnl:.2f}", delta_color="inverse")
                        
                        st.metric("Avg Trade Duration (hours)", f"{results['avg_duration']:.2f}")
                        
                        # Display trades
                        if results['trades']:
                            st.markdown("### Trade Details")
                            
                            for idx, trade in enumerate(results['trades']):
                                with st.expander(f"Trade #{idx + 1} - {trade['signal']} - P&L: {trade['pnl']:.2f}"):
                                    t_col1, t_col2 = st.columns(2)
                                    
                                    with t_col1:
                                        st.write(f"**Entry Time:** {trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                                        st.write(f"**Exit Time:** {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                                        st.write(f"**Duration:** {trade['duration']:.2f} hours")
                                        st.write(f"**Signal:** {trade['signal']}")
                                        st.write(f"**Entry Price:** {trade['entry_price']:.2f}")
                                        st.write(f"**Exit Price:** {trade['exit_price']:.2f}")
                                    
                                    with t_col2:
                                        st.write(f"**Stop Loss:** {trade['sl']:.2f}")
                                        st.write(f"**Target:** {trade['target']:.2f}")
                                        st.write(f"**Exit Reason:** {trade['exit_reason']}")
                                        pnl_color = "green" if trade['pnl'] >= 0 else "red"
                                        st.markdown(f"**P&L:** <span style='color:{pnl_color}'>{trade['pnl']:.2f}</span>", unsafe_allow_html=True)
                                        st.write(f"**Highest:** {trade['highest']:.2f}")
                                        st.write(f"**Lowest:** {trade['lowest']:.2f}")
                                        st.write(f"**Range:** {trade['range']:.2f}")
                        else:
                            st.info("No trades executed during backtest period.")
        
        with tabs[1]:
            st.markdown("### 📈 Trade History")
            
            if len(st.session_state['trade_history']) == 0:
                st.info("No trade history available. Run a backtest or start live trading.")
            else:
                # Summary metrics
                total_trades = len(st.session_state['trade_history'])
                winning_trades = sum(1 for t in st.session_state['trade_history'] if t['pnl'] > 0)
                losing_trades = sum(1 for t in st.session_state['trade_history'] if t['pnl'] <= 0)
                accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                total_pnl = sum(t['pnl'] for t in st.session_state['trade_history'])
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Trades", total_trades)
                with col2:
                    st.metric("Winning", winning_trades)
                with col3:
                    st.metric("Losing", losing_trades)
                with col4:
                    st.metric("Accuracy", f"{accuracy:.2f}%")
                with col5:
                    if total_pnl >= 0:
                        st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"+{total_pnl:.2f}")
                    else:
                        st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color="inverse")
                
                st.markdown("---")
                
                # Display trades
                for idx, trade in enumerate(reversed(st.session_state['trade_history'])):
                    with st.expander(f"Trade #{len(st.session_state['trade_history']) - idx} - {trade.get('signal', 'N/A')} - P&L: {trade.get('pnl', 0):.2f}"):
                        t_col1, t_col2 = st.columns(2)
                        
                        with t_col1:
                            entry_time = trade.get('entry_time', 'N/A')
                            entry_str = entry_time.strftime('%Y-%m-%d %H:%M:%S') if isinstance(entry_time, datetime) else str(entry_time)
                            st.write(f"**Entry Time:** {entry_str}")
                            
                            exit_time = trade.get('exit_time', 'N/A')
                            exit_str = exit_time.strftime('%Y-%m-%d %H:%M:%S') if isinstance(exit_time, datetime) else str(exit_time)
                            st.write(f"**Exit Time:** {exit_str}")
                            
                            st.write(f"**Duration:** {trade.get('duration', 0):.2f} hours")
                            st.write(f"**Signal:** {trade.get('signal', 'N/A')}")
                            st.write(f"**Entry Price:** {trade.get('entry_price', 0):.2f}")
                            st.write(f"**Exit Price:** {trade.get('exit_price', 0):.2f}")
                        
                        with t_col2:
                            st.write(f"**Stop Loss:** {trade.get('sl', 0):.2f}")
                            st.write(f"**Target:** {trade.get('target', 0):.2f}")
                            st.write(f"**Exit Reason:** {trade.get('exit_reason', 'N/A')}")
                            pnl = trade.get('pnl', 0)
                            pnl_color = "green" if pnl >= 0 else "red"
                            st.markdown(f"**P&L:** <span style='color:{pnl_color}'>{pnl:.2f}</span>", unsafe_allow_html=True)
                            st.write(f"**Highest:** {trade.get('highest', 0):.2f}")
                            st.write(f"**Lowest:** {trade.get('lowest', 0):.2f}")
                            st.write(f"**Range:** {trade.get('range', 0):.2f}")
        
        with tabs[2]:
            st.markdown("### 📝 Trade Logs")
            
            if len(st.session_state['trade_logs']) == 0:
                st.info("No logs available yet.")
            else:
                for log in reversed(st.session_state['trade_logs']):
                    st.text(log)
    
    else:  # Live Trading
        tabs = st.tabs(["🔴 Live Trading Dashboard", "📈 Trade History", "📝 Trade Logs"])
        
        with tabs[0]:
            st.markdown("### 🔴 Live Trading Dashboard")
            
            # Trading Controls
            control_col1, control_col2, control_col3, control_col4 = st.columns([2, 2, 2, 2])
            
            with control_col1:
                if st.button("▶️ Start Trading", type="primary", key="start_trading_btn", disabled=st.session_state['trading_active']):
                    st.session_state['trading_active'] = True
                    add_log("Trading started")
                    st.rerun()
            
            with control_col2:
                if st.button("⏹️ Stop Trading", key="stop_trading_btn", disabled=not st.session_state['trading_active']):
                    st.session_state['trading_active'] = False
                    
                    # Close position if open
                    if st.session_state['position']:
                        position = st.session_state['position']
                        current_price = st.session_state['current_data']['Close'].iloc[-1] if st.session_state['current_data'] is not None else position['entry_price']
                        
                        # Calculate PnL
                        quantity = config['quantity']
                        if position['signal'] == 1:
                            pnl = (current_price - position['entry_price']) * quantity
                        else:
                            pnl = (position['entry_price'] - current_price) * quantity
                        
                        # Record manual close
                        duration = (datetime.now(pytz.timezone('Asia/Kolkata')) - position['entry_time']).total_seconds() / 3600
                        trade_record = {
                            'entry_time': position['entry_time'],
                            'exit_time': datetime.now(pytz.timezone('Asia/Kolkata')),
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
                            'range': st.session_state['highest_price'] - st.session_state['lowest_price'],
                        }
                        
                        st.session_state['trade_history'].append(trade_record)
                        st.session_state['trade_history'] = st.session_state['trade_history']
                        
                        add_log(f"Position manually closed | PnL: {pnl:.2f}")
                        reset_position_state()
                    
                    add_log("Trading stopped")
                    st.rerun()
            
            with control_col3:
                if st.session_state['trading_active']:
                    st.success("🟢 Trading is ACTIVE")
                else:
                    st.info("⚪ Trading is STOPPED")
            
            with control_col4:
                if st.button("🔄 Manual Refresh", key="manual_refresh_btn"):
                    st.rerun()
            
            st.markdown("---")
            
            # Active Configuration Display
            st.markdown("### ⚙️ Active Configuration")
            conf_col1, conf_col2, conf_col3 = st.columns(3)
            
            with conf_col1:
                st.write(f"**Asset:** {config['asset']}")
                st.write(f"**Interval:** {config['interval']}")
                st.write(f"**Period:** {config['period']}")
                st.write(f"**Quantity:** {config['quantity']}")
            
            with conf_col2:
                st.write(f"**Strategy:** {config['strategy']}")
                st.write(f"**SL Type:** {config['sl_type']}")
                sl_points_display = f"{config['sl_points']:.0f}" if 'sl_points' in config else "N/A"
                st.write(f"**SL Points:** {sl_points_display}")
            
            with conf_col3:
                st.write(f"**Target Type:** {config['target_type']}")
                target_points_display = f"{config['target_points']:.0f}" if 'target_points' in config else "N/A"
                st.write(f"**Target Points:** {target_points_display}")
                st.write(f"**Mode:** {config['mode']}")
            
            if config['strategy'] == "EMA Crossover Strategy":
                st.write(f"**EMA Fast/Slow:** {config['ema_fast']}/{config['ema_slow']} | **Min Angle:** {config['min_angle']}° | **Entry Filter:** {config['entry_filter']} | **ADX Filter:** {'Enabled' if config['use_adx'] else 'Disabled'}")
            
            st.markdown("---")
            
            # Start trading loop FIRST - fetch data before displaying
            if st.session_state['trading_active']:
                # Fetch data immediately for display
                ticker = ASSET_TICKERS.get(config['asset'], config['custom_ticker'])
                df = fetch_data(ticker, config['interval'], config['period'], mode="Live Trading")
                
                if df is not None and len(df) >= 50:
                    df = add_all_indicators(df, config.get('ema_fast', 9), config.get('ema_slow', 15), config.get('adx_period', 14))
                    st.session_state['current_data'] = df
                    st.session_state['last_refresh'] = datetime.now(pytz.timezone('Asia/Kolkata'))
                    
                    # Process trading logic
                    i = len(df) - 1
                    current_price = df['Close'].iloc[i]
                    
                    # Check if in position
                    if st.session_state['position']:
                        position = st.session_state['position']
                        
                        # Update highest/lowest
                        if st.session_state['highest_price'] is None or current_price > st.session_state['highest_price']:
                            st.session_state['highest_price'] = current_price
                        if st.session_state['lowest_price'] is None or current_price < st.session_state['lowest_price']:
                            st.session_state['lowest_price'] = current_price
                        
                        # Update trailing SL
                        if "Trailing" in config['sl_type']:
                            new_sl = update_trailing_sl(position, current_price, config)
                            if new_sl != position['sl']:
                                position['sl'] = new_sl
                                st.session_state['position'] = position
                                add_log(f"Trailing SL updated to {new_sl:.2f}")
                        
                        # Check break-even
                        if config['sl_type'] == "Break-even After 50% Target" and not st.session_state['breakeven_activated']:
                            if position['signal'] == 1:
                                profit_dist = current_price - position['entry_price']
                                target_dist = position['target'] - position['entry_price']
                                if profit_dist >= target_dist * 0.5:
                                    position['sl'] = position['entry_price']
                                    st.session_state['position'] = position
                                    st.session_state['breakeven_activated'] = True
                                    add_log("Break-even activated - SL moved to entry")
                            elif position['signal'] == -1:
                                profit_dist = position['entry_price'] - current_price
                                target_dist = position['entry_price'] - position['target']
                                if profit_dist >= target_dist * 0.5:
                                    position['sl'] = position['entry_price']
                                    st.session_state['position'] = position
                                    st.session_state['breakeven_activated'] = True
                                    add_log("Break-even activated - SL moved to entry")
                        
                        # Check partial exit
                        if config['target_type'] == "50% Exit at Target (Partial)" and not st.session_state['partial_exit_done']:
                            if position['signal'] == 1 and current_price >= position['target']:
                                st.session_state['partial_exit_done'] = True
                                add_log("50% position exited at target - trailing remaining")
                            elif position['signal'] == -1 and current_price <= position['target']:
                                st.session_state['partial_exit_done'] = True
                                add_log("50% position exited at target - trailing remaining")
                        
                        # Check exit
                        should_exit, exit_reason, exit_price = check_exit_conditions(position, current_price, df, i, config)
                        
                        if should_exit:
                            # Calculate PnL
                            quantity = config['quantity']
                            if position['signal'] == 1:
                                pnl = (exit_price - position['entry_price']) * quantity
                            else:
                                pnl = (position['entry_price'] - exit_price) * quantity
                            
                            # Adjust for partial exit
                            if st.session_state['partial_exit_done']:
                                pnl = pnl * 0.5
                            
                            # Record trade
                            duration = (datetime.now(pytz.timezone('Asia/Kolkata')) - position['entry_time']).total_seconds() / 3600
                            trade_record = {
                                'entry_time': position['entry_time'],
                                'exit_time': datetime.now(pytz.timezone('Asia/Kolkata')),
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
                                'range': st.session_state['highest_price'] - st.session_state['lowest_price'],
                            }
                            
                            st.session_state['trade_history'].append(trade_record)
                            st.session_state['trade_history'] = st.session_state['trade_history']
                            
                            add_log(f"Position closed: {exit_reason} | PnL: {pnl:.2f}")
                            reset_position_state()
                    
                    # Check for entry signal
                    if not st.session_state['position']:
                        signal = 0
                        entry_price = None
                        sl = None
                        target = None
                        
                        if config['strategy'] == "EMA Crossover Strategy":
                            signal, entry_price, sl, target = check_ema_crossover(df, i, config)
                        elif config['strategy'] == "Simple Buy Strategy":
                            signal, entry_price, sl, target = check_simple_buy(df, i, config)
                        elif config['strategy'] == "Simple Sell Strategy":
                            signal, entry_price, sl, target = check_simple_sell(df, i, config)
                        elif config['strategy'] == "Price Crosses Threshold":
                            signal, entry_price, sl, target = check_price_threshold(df, i, config)
                        elif config['strategy'] == "RSI-ADX-EMA Strategy":
                            signal, entry_price, sl, target = check_rsi_adx_ema(df, i, config)
                        elif config['strategy'] == "Percentage Change Strategy":
                            signal, entry_price, sl, target = check_percentage_change(df, i, config)
                        elif config['strategy'] == "AI Price Action Analysis":
                            signal, entry_price, sl, target, _ = ai_price_action_analysis(df, i, config)
                        elif config['strategy'] == "Custom Strategy Builder":
                            signal, entry_price, sl, target = check_custom_strategy(df, i, st.session_state['custom_conditions'])
                            if signal != 0 and (sl is None or target is None):
                                sl, target = calculate_sl_target(entry_price, signal, df, i, config)
                        
                        if signal != 0:
                            position = {
                                'signal': signal,
                                'entry_price': entry_price,
                                'entry_time': datetime.now(pytz.timezone('Asia/Kolkata')),
                                'sl': sl,
                                'target': target,
                            }
                            
                            st.session_state['position'] = position
                            st.session_state['highest_price'] = entry_price
                            st.session_state['lowest_price'] = entry_price
                            
                            signal_text = "LONG" if signal == 1 else "SHORT"
                            add_log(f"{signal_text} entry at {entry_price:.2f} | SL: {sl:.2f} | Target: {target:.2f}")
                    
                    # Schedule next refresh
                    time.sleep(random.uniform(1.0, 1.5))
                    st.rerun()
            
            # Live Metrics Display (ALWAYS show if data exists)
            if st.session_state['current_data'] is not None and len(st.session_state['current_data']) > 0:
                df = st.session_state['current_data']
                current_price = df['Close'].iloc[-1]
                ema_fast = df['EMA_Fast'].iloc[-1]
                ema_slow = df['EMA_Slow'].iloc[-1]
                ema_angle = abs(df['EMA_Angle_Fast'].iloc[-1])
                rsi = df['RSI'].iloc[-1]
                
                st.markdown("### 📊 Live Metrics")
                
                metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
                
                with metric_col1:
                    st.metric("Current Price", f"{current_price:.2f}")
                
                with metric_col2:
                    if st.session_state['position']:
                        entry_price = st.session_state['position']['entry_price']
                        st.metric("Entry Price", f"{entry_price:.2f}")
                    else:
                        st.metric("Entry Price", "N/A")
                
                with metric_col3:
                    if st.session_state['position']:
                        position_type = "LONG" if st.session_state['position']['signal'] == 1 else "SHORT"
                        st.metric("Position", position_type)
                    else:
                        st.metric("Position", "NONE")
                
                with metric_col4:
                    if st.session_state['position']:
                        position = st.session_state['position']
                        quantity = config['quantity']
                        if position['signal'] == 1:
                            unrealized_pnl = (current_price - position['entry_price']) * quantity
                        else:
                            unrealized_pnl = (position['entry_price'] - current_price) * quantity
                        
                        if unrealized_pnl >= 0:
                            st.metric("Unrealized P&L", f"{unrealized_pnl:.2f}", delta=f"+{unrealized_pnl:.2f}")
                        else:
                            st.metric("Unrealized P&L", f"{unrealized_pnl:.2f}", delta=f"{unrealized_pnl:.2f}", delta_color="inverse")
                    else:
                        st.metric("Unrealized P&L", "N/A")
                
                with metric_col5:
                    if st.session_state['last_refresh']:
                        last_update = st.session_state['last_refresh'].strftime('%H:%M:%S')
                        st.metric("Last Update", last_update)
                    else:
                        st.metric("Last Update", "N/A")
                
                # Additional metrics
                add_metric_col1, add_metric_col2, add_metric_col3, add_metric_col4 = st.columns(4)
                
                with add_metric_col1:
                    st.metric("EMA Fast", f"{ema_fast:.2f}")
                
                with add_metric_col2:
                    st.metric("EMA Slow", f"{ema_slow:.2f}")
                
                with add_metric_col3:
                    st.metric("EMA Angle", f"{ema_angle:.2f}°")
                
                with add_metric_col4:
                    st.metric("RSI", f"{rsi:.2f}")
                
                # Entry filter status
                if config['strategy'] == "EMA Crossover Strategy":
                    st.markdown("### 🎯 Entry Filter Status")
                    
                    filter_col1, filter_col2, filter_col3 = st.columns(3)
                    
                    with filter_col1:
                        if config['entry_filter'] == "Custom Candle (Points)":
                            candle_size = abs(df['Close'].iloc[-1] - df['Open'].iloc[-1])
                            min_size = config.get('custom_points', 10)
                            status = "✅" if candle_size >= min_size else "❌"
                            st.write(f"{status} Candle Size: {candle_size:.2f} / Min: {min_size:.2f}")
                        elif config['entry_filter'] == "ATR-based Candle":
                            candle_size = abs(df['Close'].iloc[-1] - df['Open'].iloc[-1])
                            atr = df['ATR'].iloc[-1]
                            min_size = atr * config.get('atr_multiplier', 1.0)
                            status = "✅" if candle_size >= min_size else "❌"
                            st.write(f"{status} Candle Size: {candle_size:.2f} / Min (ATR×M): {min_size:.2f}")
                        else:
                            st.write("✅ Simple Crossover (No filter)")
                    
                    with filter_col2:
                        angle_check = "✅" if ema_angle >= config['min_angle'] else "❌"
                        st.write(f"{angle_check} Angle: {ema_angle:.2f}° / Min: {config['min_angle']:.2f}°")
                    
                    with filter_col3:
                        if config['use_adx']:
                            adx = df['ADX'].iloc[-1]
                            adx_check = "✅" if adx >= config['adx_threshold'] else "❌"
                            st.write(f"{adx_check} ADX: {adx:.2f} / Threshold: {config['adx_threshold']:.2f}")
                        else:
                            st.write("➖ ADX Filter: Disabled")
                
                # Current signal
                st.markdown("### 📡 Current Signal")
                
                if len(df) > 1:
                    i = len(df) - 1
                    
                    if config['strategy'] == "EMA Crossover Strategy":
                        signal, entry_price, sl, target = check_ema_crossover(df, i, config)
                    elif config['strategy'] == "Simple Buy Strategy":
                        signal, entry_price, sl, target = check_simple_buy(df, i, config)
                    elif config['strategy'] == "Simple Sell Strategy":
                        signal, entry_price, sl, target = check_simple_sell(df, i, config)
                    elif config['strategy'] == "Price Crosses Threshold":
                        signal, entry_price, sl, target = check_price_threshold(df, i, config)
                    elif config['strategy'] == "RSI-ADX-EMA Strategy":
                        signal, entry_price, sl, target = check_rsi_adx_ema(df, i, config)
                    elif config['strategy'] == "Percentage Change Strategy":
                        signal, entry_price, sl, target = check_percentage_change(df, i, config)
                        # Show current percentage change
                        first_price = df['Close'].iloc[0]
                        pct_change = ((current_price - first_price) / first_price) * 100
                        pct_color = "green" if pct_change >= 0 else "red"
                        st.markdown(f"**Current % Change from start:** <span style='color:{pct_color}'>{pct_change:+.2f}%</span>", unsafe_allow_html=True)
                    elif config['strategy'] == "AI Price Action Analysis":
                        signal, entry_price, sl, target, analysis = ai_price_action_analysis(df, i, config)
                        if analysis:
                            st.markdown(f"```\n{analysis}\n```")
                    elif config['strategy'] == "Custom Strategy Builder":
                        signal, entry_price, sl, target = check_custom_strategy(df, i, st.session_state['custom_conditions'])
                    else:
                        signal = 0
                    
                    if signal == 1:
                        st.success("🔵 BUY Signal Detected")
                    elif signal == -1:
                        st.error("🔴 SELL Signal Detected")
                    else:
                        st.info("⚪ No Signal")
                
                # Position Information
                if st.session_state['position']:
                    st.markdown("### 💼 Position Information")
                    
                    position = st.session_state['position']
                    entry_time = position['entry_time']
                    duration = (datetime.now(pytz.timezone('Asia/Kolkata')) - entry_time).total_seconds() / 3600
                    
                    pos_col1, pos_col2 = st.columns(2)
                    
                    with pos_col1:
                        st.write(f"**Entry Time:** {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"**Duration:** {duration:.2f} hours")
                        st.write(f"**Entry Price:** {position['entry_price']:.2f}")
                        st.write(f"**Stop Loss:** {position['sl']:.2f}")
                        st.write(f"**Target:** {position['target']:.2f}")
                    
                    with pos_col2:
                        sl_dist = abs(current_price - position['sl'])
                        target_dist = abs(position['target'] - current_price)
                        st.write(f"**Distance to SL:** {sl_dist:.2f} points")
                        st.write(f"**Distance to Target:** {target_dist:.2f} points")
                        st.write(f"**Highest Price:** {st.session_state['highest_price']:.2f}")
                        st.write(f"**Lowest Price:** {st.session_state['lowest_price']:.2f}")
                        st.write(f"**Range:** {st.session_state['highest_price'] - st.session_state['lowest_price']:.2f}")
                    
                    if st.session_state['breakeven_activated']:
                        st.success("✅ Break-even activated - SL at entry")
                    
                    if st.session_state['partial_exit_done']:
                        st.success("✅ 50% position exited - trailing remaining")
                
                # Live Chart
                st.markdown("### 📈 Live Chart")
                
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                )])
                
                # Add EMAs
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], name=f'EMA {config["ema_fast"]}', line=dict(color='blue', width=1)))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], name=f'EMA {config["ema_slow"]}', line=dict(color='red', width=1)))
                
                # Add position lines
                if st.session_state['position']:
                    position = st.session_state['position']
                    
                    # Entry line
                    fig.add_hline(y=position['entry_price'], line_dash="dash", line_color="yellow", annotation_text="Entry")
                    
                    # SL line
                    if position['sl'] and position['sl'] != 0:
                        fig.add_hline(y=position['sl'], line_dash="dash", line_color="red", annotation_text="SL")
                    
                    # Target line
                    if position['target'] and position['target'] != 0:
                        fig.add_hline(y=position['target'], line_dash="dash", line_color="green", annotation_text="Target")
                
                fig.update_layout(
                    title=f"{config['asset']} - {config['interval']}",
                    yaxis_title='Price',
                    xaxis_title='Time',
                    height=600,
                    xaxis_rangeslider_visible=False
                )
                
                chart_key = f"live_chart_{int(time.time())}"
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
                
                # Textual Guidance
                st.markdown("### 💡 Guidance")
                
                if st.session_state['position']:
                    position = st.session_state['position']
                    quantity = config['quantity']
                    if position['signal'] == 1:
                        unrealized_pnl = (current_price - position['entry_price']) * quantity
                    else:
                        unrealized_pnl = (position['entry_price'] - current_price) * quantity
                    
                    if unrealized_pnl > 0:
                        st.success(f"✅ In Profit: {unrealized_pnl:.2f} points - Hold position")
                    else:
                        st.warning(f"⚠️ In Loss: {unrealized_pnl:.2f} points - Monitor SL")
                else:
                    st.info("⏳ Waiting for entry signal...")
            
            else:
                st.info("Start trading to see live data...")
        
        with tabs[1]:
            st.markdown("### 📈 Trade History")
            
            if len(st.session_state['trade_history']) == 0:
                st.info("No trade history available.")
            else:
                # Summary metrics
                total_trades = len(st.session_state['trade_history'])
                winning_trades = sum(1 for t in st.session_state['trade_history'] if t['pnl'] > 0)
                losing_trades = sum(1 for t in st.session_state['trade_history'] if t['pnl'] <= 0)
                accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                total_pnl = sum(t['pnl'] for t in st.session_state['trade_history'])
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Trades", total_trades)
                with col2:
                    st.metric("Winning", winning_trades)
                with col3:
                    st.metric("Losing", losing_trades)
                with col4:
                    st.metric("Accuracy", f"{accuracy:.2f}%")
                with col5:
                    if total_pnl >= 0:
                        st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"+{total_pnl:.2f}")
                    else:
                        st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color="inverse")
                
                st.markdown("---")
                
                # Display trades
                for idx, trade in enumerate(reversed(st.session_state['trade_history'])):
                    with st.expander(f"Trade #{len(st.session_state['trade_history']) - idx} - {trade.get('signal', 'N/A')} - P&L: {trade.get('pnl', 0):.2f}"):
                        t_col1, t_col2 = st.columns(2)
                        
                        with t_col1:
                            entry_time = trade.get('entry_time', 'N/A')
                            entry_str = entry_time.strftime('%Y-%m-%d %H:%M:%S') if isinstance(entry_time, datetime) else str(entry_time)
                            st.write(f"**Entry Time:** {entry_str}")
                            
                            exit_time = trade.get('exit_time', 'N/A')
                            exit_str = exit_time.strftime('%Y-%m-%d %H:%M:%S') if isinstance(exit_time, datetime) else str(exit_time)
                            st.write(f"**Exit Time:** {exit_str}")
                            
                            st.write(f"**Duration:** {trade.get('duration', 0):.2f} hours")
                            st.write(f"**Signal:** {trade.get('signal', 'N/A')}")
                            st.write(f"**Entry Price:** {trade.get('entry_price', 0):.2f}")
                            st.write(f"**Exit Price:** {trade.get('exit_price', 0):.2f}")
                        
                        with t_col2:
                            st.write(f"**Stop Loss:** {trade.get('sl', 0):.2f}")
                            st.write(f"**Target:** {trade.get('target', 0):.2f}")
                            st.write(f"**Exit Reason:** {trade.get('exit_reason', 'N/A')}")
                            pnl = trade.get('pnl', 0)
                            pnl_color = "green" if pnl >= 0 else "red"
                            st.markdown(f"**P&L:** <span style='color:{pnl_color}'>{pnl:.2f}</span>", unsafe_allow_html=True)
                            st.write(f"**Highest:** {trade.get('highest', 0):.2f}")
                            st.write(f"**Lowest:** {trade.get('lowest', 0):.2f}")
                            st.write(f"**Range:** {trade.get('range', 0):.2f}")
        
        with tabs[2]:
            st.markdown("### 📝 Trade Logs")
            
            if len(st.session_state['trade_logs']) == 0:
                st.info("No logs available yet.")
            else:
                for log in reversed(st.session_state['trade_logs']):
                    st.text(log)

if __name__ == "__main__":
    main()..."):
    # Clear cache
    st.session_state['current_data'] = None
                    
    ## Fetch data
