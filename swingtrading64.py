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
    "USDINR": "USDINR=X",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "Gold": "GC=F",
    "Silver": "SI=F"
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

TRAILING_TARGET_MINIMUMS = {
    "NIFTY 50": 10,
    "BANKNIFTY": 20,
    "BTC": 150,
    "ETH": 10,
    "Default": 15
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
        'custom_conditions': [],
        'partial_exit_done': False,
        'breakeven_activated': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def add_log(message):
    """Add timestamped log entry"""
    timestamp = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state['trade_logs'].append(log_entry)
    st.session_state['trade_logs'] = st.session_state['trade_logs'][-50:]  # Keep last 50
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

def fetch_data(ticker, interval, period, mode):
    """Fetch data from yfinance with proper error handling"""
    try:
        # Add delay for live trading mode
        if mode == "Live Trading":
            delay = random.uniform(1.0, 1.5)
            time.sleep(delay)
        
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if data.empty:
            return None
        
        # Flatten multi-index if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in data.columns.values]
            # Select OHLCV columns
            cols_to_keep = [c for c in data.columns if any(x in c for x in ['Open', 'High', 'Low', 'Close', 'Volume'])]
            data = data[cols_to_keep]
            # Rename to standard format
            data.columns = [col.split('_')[0] for col in data.columns]
        
        # Handle timezone
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            data.index = data.index.tz_convert('Asia/Kolkata')
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
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
    minus_dm = low.diff().abs()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm = -low.diff()
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
    pivot = (df['High'] + df['Low'] + df['Close']) / 3
    r1 = 2 * pivot - df['Low']
    r2 = pivot + (df['High'] - df['Low'])
    s1 = 2 * pivot - df['High']
    s2 = pivot - (df['High'] - df['Low'])
    
    return pivot, r1, r2, s1, s2

def calculate_support_resistance(df, window=20):
    """Calculate Support and Resistance levels"""
    support = df['Low'].rolling(window=window).min()
    resistance = df['High'].rolling(window=window).max()
    
    return support, resistance

def detect_swing_points(df, window=5):
    """Detect Swing High and Low points"""
    swing_high = df['High'].rolling(window=window*2+1, center=True).max()
    swing_low = df['Low'].rolling(window=window*2+1, center=True).min()
    
    is_swing_high = df['High'] == swing_high
    is_swing_low = df['Low'] == swing_low
    
    return swing_high.where(is_swing_high), swing_low.where(is_swing_low)

def calculate_vwap(df):
    """Calculate VWAP"""
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        return vwap
    return pd.Series(index=df.index, dtype=float)

def calculate_supertrend(df, period=10, multiplier=3):
    """Calculate SuperTrend"""
    atr = calculate_atr(df, period)
    hl_avg = (df['High'] + df['Low']) / 2
    
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    for i in range(period, len(df)):
        if i == period:
            supertrend.iloc[i] = upper_band.iloc[i] if df['Close'].iloc[i] <= hl_avg.iloc[i] else lower_band.iloc[i]
            direction.iloc[i] = -1 if df['Close'].iloc[i] <= hl_avg.iloc[i] else 1
        else:
            if direction.iloc[i-1] == 1:
                if df['Close'].iloc[i] <= lower_band.iloc[i]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
                    direction.iloc[i] = 1
            else:
                if df['Close'].iloc[i] >= upper_band.iloc[i]:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])
                    direction.iloc[i] = -1
    
    return supertrend, direction

def calculate_ema_angle(ema_series, lookback=1):
    """Calculate EMA angle in degrees"""
    if len(ema_series) < lookback + 1:
        return pd.Series(0, index=ema_series.index)
    
    slope = ema_series.diff(lookback)
    angle = np.degrees(np.arctan(slope))
    
    return angle

# ==================== STRATEGY FUNCTIONS ====================

def ema_crossover_strategy(df, config):
    """EMA Crossover Strategy with filters"""
    fast_period = config['ema_fast']
    slow_period = config['ema_slow']
    min_angle = config['min_angle']
    entry_filter = config['entry_filter']
    custom_points = config.get('custom_points', 10)
    atr_multiplier = config.get('atr_multiplier', 1.5)
    use_adx = config.get('use_adx', False)
    adx_threshold = config.get('adx_threshold', 25)
    adx_period = config.get('adx_period', 14)
    
    # Calculate EMAs
    df['EMA_Fast'] = calculate_ema(df['Close'], fast_period)
    df['EMA_Slow'] = calculate_ema(df['Close'], slow_period)
    
    # Calculate EMA angles
    df['EMA_Fast_Angle'] = calculate_ema_angle(df['EMA_Fast']).abs()
    df['EMA_Slow_Angle'] = calculate_ema_angle(df['EMA_Slow']).abs()
    
    # Calculate ATR if needed
    if entry_filter == 'ATR-based Candle':
        df['ATR'] = calculate_atr(df)
    
    # Calculate ADX if needed
    if use_adx:
        df['ADX'] = calculate_adx(df, adx_period)
    
    # Detect crossovers
    df['Signal'] = 0
    
    for i in range(1, len(df)):
        # Bullish crossover
        bullish_cross = (df['EMA_Fast'].iloc[i] > df['EMA_Slow'].iloc[i] and 
                        df['EMA_Fast'].iloc[i-1] <= df['EMA_Slow'].iloc[i-1])
        
        # Bearish crossover
        bearish_cross = (df['EMA_Fast'].iloc[i] < df['EMA_Slow'].iloc[i] and 
                        df['EMA_Fast'].iloc[i-1] >= df['EMA_Slow'].iloc[i-1])
        
        if bullish_cross or bearish_cross:
            # Check angle
            angle_ok = (df['EMA_Fast_Angle'].iloc[i] >= min_angle and 
                       df['EMA_Slow_Angle'].iloc[i] >= min_angle)
            
            if not angle_ok:
                continue
            
            # Check entry filter
            candle_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
            filter_ok = True
            
            if entry_filter == 'Custom Candle (Points)':
                filter_ok = candle_size >= custom_points
            elif entry_filter == 'ATR-based Candle':
                min_candle = df['ATR'].iloc[i] * atr_multiplier
                filter_ok = candle_size >= min_candle
            
            if not filter_ok:
                continue
            
            # Check ADX if enabled
            if use_adx:
                if df['ADX'].iloc[i] < adx_threshold:
                    continue
            
            # Set signal
            if bullish_cross:
                df.loc[df.index[i], 'Signal'] = 1
            elif bearish_cross:
                df.loc[df.index[i], 'Signal'] = -1
    
    return df

def simple_buy_strategy(df, config):
    """Simple Buy Strategy"""
    df['Signal'] = 1  # Always buy signal at each candle
    return df

def simple_sell_strategy(df, config):
    """Simple Sell Strategy"""
    df['Signal'] = -1  # Always sell signal at each candle
    return df

def price_threshold_strategy(df, config):
    """Price Crosses Threshold Strategy"""
    threshold = config['threshold']
    direction = config['direction']
    
    df['Signal'] = 0
    
    for i in range(len(df)):
        price = df['Close'].iloc[i]
        
        if direction == 'LONG (Price >= Threshold)':
            if price >= threshold:
                df.loc[df.index[i], 'Signal'] = 1
        elif direction == 'SHORT (Price >= Threshold)':
            if price >= threshold:
                df.loc[df.index[i], 'Signal'] = -1
        elif direction == 'LONG (Price <= Threshold)':
            if price <= threshold:
                df.loc[df.index[i], 'Signal'] = 1
        elif direction == 'SHORT (Price <= Threshold)':
            if price <= threshold:
                df.loc[df.index[i], 'Signal'] = -1
    
    return df

def rsi_adx_ema_strategy(df, config):
    """RSI-ADX-EMA Strategy"""
    rsi_period = config.get('rsi_period', 14)
    adx_period = config.get('adx_period', 14)
    ema1_period = config.get('ema1_period', 9)
    ema2_period = config.get('ema2_period', 21)
    
    df['RSI'] = calculate_rsi(df['Close'], rsi_period)
    df['ADX'] = calculate_adx(df, adx_period)
    df['EMA1'] = calculate_ema(df['Close'], ema1_period)
    df['EMA2'] = calculate_ema(df['Close'], ema2_period)
    
    df['Signal'] = 0
    
    for i in range(1, len(df)):
        # SELL: RSI>80 AND ADX<20 AND EMA1<EMA2
        if (df['RSI'].iloc[i] > 80 and 
            df['ADX'].iloc[i] < 20 and 
            df['EMA1'].iloc[i] < df['EMA2'].iloc[i]):
            df.loc[df.index[i], 'Signal'] = -1
        
        # BUY: RSI<20 AND ADX>20 AND EMA1>EMA2
        elif (df['RSI'].iloc[i] < 20 and 
              df['ADX'].iloc[i] > 20 and 
              df['EMA1'].iloc[i] > df['EMA2'].iloc[i]):
            df.loc[df.index[i], 'Signal'] = 1
    
    return df

def percentage_change_strategy(df, config):
    """Percentage Change Strategy"""
    pct_threshold = config['pct_threshold']
    direction = config['pct_direction']
    
    first_price = df['Close'].iloc[0]
    df['Pct_Change'] = ((df['Close'] - first_price) / first_price) * 100
    
    df['Signal'] = 0
    
    for i in range(len(df)):
        pct_change = df['Pct_Change'].iloc[i]
        
        if direction == 'BUY on Fall':
            if pct_change <= -pct_threshold:
                df.loc[df.index[i], 'Signal'] = 1
        elif direction == 'SELL on Fall':
            if pct_change <= -pct_threshold:
                df.loc[df.index[i], 'Signal'] = -1
        elif direction == 'BUY on Rise':
            if pct_change >= pct_threshold:
                df.loc[df.index[i], 'Signal'] = 1
        elif direction == 'SELL on Rise':
            if pct_change >= pct_threshold:
                df.loc[df.index[i], 'Signal'] = -1
    
    return df

def ai_price_action_strategy(df, config):
    """AI Price Action Analysis Strategy"""
    # Calculate all indicators
    df['EMA_20'] = calculate_ema(df['Close'], 20)
    df['EMA_50'] = calculate_ema(df['Close'], 50)
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    df['ATR'] = calculate_atr(df)
    
    # Handle volume for indices
    has_volume = 'Volume' in df.columns and df['Volume'].sum() > 0
    
    df['Signal'] = 0
    df['AI_Confidence'] = 0.0
    df['AI_Analysis'] = ''
    
    for i in range(50, len(df)):
        score = 0
        analysis_parts = []
        
        # Trend Analysis (EMA)
        if df['EMA_20'].iloc[i] > df['EMA_50'].iloc[i]:
            score += 25
            analysis_parts.append("Uptrend (EMA20>EMA50)")
        else:
            score -= 25
            analysis_parts.append("Downtrend (EMA20<EMA50)")
        
        # RSI Analysis
        rsi = df['RSI'].iloc[i]
        if rsi < 30:
            score += 20
            analysis_parts.append("Oversold (RSI<30)")
        elif rsi > 70:
            score -= 20
            analysis_parts.append("Overbought (RSI>70)")
        else:
            analysis_parts.append(f"Neutral RSI ({rsi:.1f})")
        
        # MACD Analysis
        if df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]:
            score += 20
            analysis_parts.append("Bullish MACD")
        else:
            score -= 20
            analysis_parts.append("Bearish MACD")
        
        # Bollinger Bands
        price = df['Close'].iloc[i]
        if price <= df['BB_Lower'].iloc[i]:
            score += 15
            analysis_parts.append("Price at Lower BB")
        elif price >= df['BB_Upper'].iloc[i]:
            score -= 15
            analysis_parts.append("Price at Upper BB")
        else:
            analysis_parts.append("Price in BB range")
        
        # Volume (if available)
        if has_volume:
            avg_volume = df['Volume'].iloc[i-20:i].mean()
            if df['Volume'].iloc[i] > avg_volume * 1.5:
                score += 20
                analysis_parts.append("High volume")
            else:
                analysis_parts.append("Normal volume")
        
        # Confidence
        confidence = min(abs(score), 100)
        
        # Generate signal
        if score >= 50:
            df.loc[df.index[i], 'Signal'] = 1
            df.loc[df.index[i], 'AI_Confidence'] = confidence
        elif score <= -50:
            df.loc[df.index[i], 'Signal'] = -1
            df.loc[df.index[i], 'AI_Confidence'] = confidence
        
        df.loc[df.index[i], 'AI_Analysis'] = ' | '.join(analysis_parts)
    
    return df

def custom_strategy_builder(df, conditions):
    """Custom Strategy Builder"""
    # Calculate all possible indicators
    df['EMA_20'] = calculate_ema(df['Close'], 20)
    df['EMA_50'] = calculate_ema(df['Close'], 50)
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['ADX'] = calculate_adx(df, 14)
    df['ATR'] = calculate_atr(df)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    df['SuperTrend'], _ = calculate_supertrend(df)
    df['Support'], df['Resistance'] = calculate_support_resistance(df)
    
    has_volume = 'Volume' in df.columns and df['Volume'].sum() > 0
    if has_volume:
        df['VWAP'] = calculate_vwap(df)
    
    df['Signal'] = 0
    
    # Separate BUY and SELL conditions
    buy_conditions = [c for c in conditions if c['use_condition'] and c['action'] == 'BUY']
    sell_conditions = [c for c in conditions if c['use_condition'] and c['action'] == 'SELL']
    
    for i in range(50, len(df)):
        # Check BUY conditions
        if buy_conditions:
            buy_met = True
            for cond in buy_conditions:
                if not evaluate_condition(df, i, cond):
                    buy_met = False
                    break
            if buy_met:
                df.loc[df.index[i], 'Signal'] = 1
        
        # Check SELL conditions
        if sell_conditions:
            sell_met = True
            for cond in sell_conditions:
                if not evaluate_condition(df, i, cond):
                    sell_met = False
                    break
            if sell_met:
                df.loc[df.index[i], 'Signal'] = -1
    
    return df

def evaluate_condition(df, i, condition):
    """Evaluate a single condition"""
    indicator = condition['indicator']
    operator = condition['operator']
    compare_price = condition.get('compare_price', False)
    compare_indicator = condition.get('compare_indicator', 'EMA_20')
    value = condition.get('value', 0)
    
    # Get indicator value
    if indicator == 'Price':
        ind_val = df['Close'].iloc[i]
    elif indicator in df.columns:
        ind_val = df[indicator].iloc[i]
    else:
        return False
    
    # Get comparison value
    if compare_price:
        if compare_indicator in df.columns:
            comp_val = df[compare_indicator].iloc[i]
        else:
            return False
    else:
        comp_val = value
    
    # Evaluate operator
    if operator == '>':
        return ind_val > comp_val
    elif operator == '<':
        return ind_val < comp_val
    elif operator == '>=':
        return ind_val >= comp_val
    elif operator == '<=':
        return ind_val <= comp_val
    elif operator == '==':
        return abs(ind_val - comp_val) < 0.01
    elif operator == 'crosses_above':
        if i > 0:
            return ind_val > comp_val and df[indicator].iloc[i-1] <= comp_val
        return False
    elif operator == 'crosses_below':
        if i > 0:
            return ind_val < comp_val and df[indicator].iloc[i-1] >= comp_val
        return False
    
    return False

# ==================== STOP LOSS & TARGET FUNCTIONS ====================

def calculate_stop_loss(df, i, position, sl_type, config):
    """Calculate stop loss based on type"""
    entry_price = position['entry_price']
    position_type = position['type']
    current_price = df['Close'].iloc[i]
    
    sl_points = config.get('sl_points', 10)
    atr_multiplier = config.get('sl_atr_multiplier', 1.5)
    trailing_threshold = config.get('trailing_threshold', 0)
    
    if sl_type == 'Custom Points':
        if position_type == 1:  # LONG
            return entry_price - sl_points
        else:  # SHORT
            return entry_price + sl_points
    
    elif sl_type == 'Trailing SL (Points)':
        if position_type == 1:  # LONG
            new_sl = current_price - sl_points
            current_sl = position.get('sl', entry_price - sl_points)
            if new_sl > current_sl:
                return new_sl
            return current_sl
        else:  # SHORT
            new_sl = current_price + sl_points
            current_sl = position.get('sl', entry_price + sl_points)
            if new_sl < current_sl:
                return new_sl
            return current_sl
    
    elif sl_type == 'Trailing SL + Current Candle':
        if position_type == 1:  # LONG
            new_sl = df['Low'].iloc[i]
            current_sl = position.get('sl', entry_price - sl_points)
            if new_sl > current_sl:
                return new_sl
            return current_sl
        else:  # SHORT
            new_sl = df['High'].iloc[i]
            current_sl = position.get('sl', entry_price + sl_points)
            if new_sl < current_sl:
                return new_sl
            return current_sl
    
    elif sl_type == 'Trailing SL + Previous Candle':
        if i > 0:
            if position_type == 1:  # LONG
                new_sl = df['Low'].iloc[i-1]
                current_sl = position.get('sl', entry_price - sl_points)
                if new_sl > current_sl:
                    return new_sl
                return current_sl
            else:  # SHORT
                new_sl = df['High'].iloc[i-1]
                current_sl = position.get('sl', entry_price + sl_points)
                if new_sl < current_sl:
                    return new_sl
                return current_sl
        return position.get('sl', entry_price - sl_points if position_type == 1 else entry_price + sl_points)
    
    elif sl_type == 'ATR-based':
        atr = df['ATR'].iloc[i] if 'ATR' in df.columns else calculate_atr(df).iloc[i]
        if position_type == 1:  # LONG
            return entry_price - (atr * atr_multiplier)
        else:  # SHORT
            return entry_price + (atr * atr_multiplier)
    
    elif sl_type == 'Current Candle Low/High':
        if position_type == 1:  # LONG
            return df['Low'].iloc[i]
        else:  # SHORT
            return df['High'].iloc[i]
    
    elif sl_type == 'Previous Candle Low/High':
        if i > 0:
            if position_type == 1:  # LONG
                return df['Low'].iloc[i-1]
            else:  # SHORT
                return df['High'].iloc[i-1]
        return entry_price - sl_points if position_type == 1 else entry_price + sl_points
    
    elif sl_type == 'Signal-based (reverse EMA crossover)':
        # No fixed SL - exit on reverse signal
        return None
    
    elif sl_type == 'Break-even After 50% Target':
        target_price = position.get('target', 0)
        if target_price > 0 and not position.get('breakeven_activated', False):
            if position_type == 1:  # LONG
                profit_dist = current_price - entry_price
                target_dist = target_price - entry_price
                if profit_dist >= target_dist * 0.5:
                    return entry_price
            else:  # SHORT
                profit_dist = entry_price - current_price
                target_dist = entry_price - target_price
                if profit_dist >= target_dist * 0.5:
                    return entry_price
        return position.get('sl', entry_price - sl_points if position_type == 1 else entry_price + sl_points)
    
    elif sl_type == 'Volatility-Adjusted Trailing SL':
        atr = df['ATR'].iloc[i] if 'ATR' in df.columns else calculate_atr(df).iloc[i]
        if position_type == 1:  # LONG
            new_sl = current_price - (atr * atr_multiplier)
            current_sl = position.get('sl', entry_price - sl_points)
            if new_sl > current_sl:
                return new_sl
            return current_sl
        else:  # SHORT
            new_sl = current_price + (atr * atr_multiplier)
            current_sl = position.get('sl', entry_price + sl_points)
            if new_sl < current_sl:
                return new_sl
            return current_sl
    
    # Default
    if position_type == 1:
        return entry_price - sl_points
    else:
        return entry_price + sl_points

def calculate_target(df, i, position, target_type, config):
    """Calculate target based on type"""
    entry_price = position['entry_price']
    position_type = position['type']
    current_price = df['Close'].iloc[i]
    
    target_points = config.get('target_points', 20)
    atr_multiplier = config.get('target_atr_multiplier', 2.0)
    rr_ratio = config.get('rr_ratio', 2.0)
    
    if target_type == 'Custom Points':
        if position_type == 1:  # LONG
            return entry_price + target_points
        else:  # SHORT
            return entry_price - target_points
    
    elif target_type == 'Trailing Target (Points)':
        # Display only - never exits
        if position_type == 1:  # LONG
            highest = position.get('highest_price', current_price)
            return highest + target_points
        else:  # SHORT
            lowest = position.get('lowest_price', current_price)
            return lowest - target_points
    
    elif target_type == 'ATR-based':
        atr = df['ATR'].iloc[i] if 'ATR' in df.columns else calculate_atr(df).iloc[i]
        if position_type == 1:  # LONG
            return entry_price + (atr * atr_multiplier)
        else:  # SHORT
            return entry_price - (atr * atr_multiplier)
    
    elif target_type == 'Risk-Reward Based':
        sl_price = position.get('sl', 0)
        if sl_price > 0:
            sl_distance = abs(entry_price - sl_price)
            if position_type == 1:  # LONG
                return entry_price + (sl_distance * rr_ratio)
            else:  # SHORT
                return entry_price - (sl_distance * rr_ratio)
        else:
            if position_type == 1:
                return entry_price + target_points
            else:
                return entry_price - target_points
    
    elif target_type == 'Current Candle Low/High':
        if position_type == 1:  # LONG
            return df['High'].iloc[i]
        else:  # SHORT
            return df['Low'].iloc[i]
    
    elif target_type == 'Previous Candle Low/High':
        if i > 0:
            if position_type == 1:  # LONG
                return df['High'].iloc[i-1]
            else:  # SHORT
                return df['Low'].iloc[i-1]
        return entry_price + target_points if position_type == 1 else entry_price - target_points
    
    elif target_type == 'Signal-based (reverse EMA crossover)':
        # No fixed target
        return None
    
    elif target_type == '50% Exit at Target (Partial)':
        if position_type == 1:  # LONG
            return entry_price + target_points
        else:  # SHORT
            return entry_price - target_points
    
    # Default
    if position_type == 1:
        return entry_price + target_points
    else:
        return entry_price - target_points

def check_reverse_signal(df, i, position_type):
    """Check for reverse EMA crossover signal"""
    if i < 1 or 'EMA_Fast' not in df.columns or 'EMA_Slow' not in df.columns:
        return False
    
    ema_fast_curr = df['EMA_Fast'].iloc[i]
    ema_slow_curr = df['EMA_Slow'].iloc[i]
    ema_fast_prev = df['EMA_Fast'].iloc[i-1]
    ema_slow_prev = df['EMA_Slow'].iloc[i-1]
    
    if position_type == 1:  # LONG - check bearish crossover
        return ema_fast_curr < ema_slow_curr and ema_fast_prev >= ema_slow_prev
    else:  # SHORT - check bullish crossover
        return ema_fast_curr > ema_slow_curr and ema_fast_prev <= ema_slow_prev

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
    sl_type = config.get('sl_type', 'Custom Points')
    target_type = config.get('target_type', 'Custom Points')
    
    for i in range(1, len(df)):
        # Check for entry signal
        if position is None and df['Signal'].iloc[i] != 0:
            entry_price = df['Close'].iloc[i]
            signal = int(df['Signal'].iloc[i])
            
            position = {
                'type': signal,
                'entry_price': entry_price,
                'entry_time': df.index[i],
                'entry_index': i,
                'highest_price': entry_price,
                'lowest_price': entry_price,
                'partial_exit_done': False,
                'breakeven_activated': False
            }
            
            # Calculate SL and Target
            position['sl'] = calculate_stop_loss(df, i, position, sl_type, config)
            position['target'] = calculate_target(df, i, position, target_type, config)
            
        # Manage position
        elif position is not None:
            current_price = df['Close'].iloc[i]
            
            # Update highest/lowest
            position['highest_price'] = max(position['highest_price'], current_price)
            position['lowest_price'] = min(position['lowest_price'], current_price)
            
            # Update SL if trailing
            if sl_type not in ['Signal-based (reverse EMA crossover)', 'Custom Points']:
                new_sl = calculate_stop_loss(df, i, position, sl_type, config)
                if new_sl is not None:
                    if position['type'] == 1 and new_sl > position.get('sl', 0):
                        position['sl'] = new_sl
                        if sl_type == 'Break-even After 50% Target' and new_sl == position['entry_price']:
                            position['breakeven_activated'] = True
                    elif position['type'] == -1 and new_sl < position.get('sl', float('inf')):
                        position['sl'] = new_sl
                        if sl_type == 'Break-even After 50% Target' and new_sl == position['entry_price']:
                            position['breakeven_activated'] = True
            
            # Check exit conditions
            exit_price = None
            exit_reason = None
            partial_exit = False
            
            # Check signal-based exit
            if sl_type == 'Signal-based (reverse EMA crossover)' or target_type == 'Signal-based (reverse EMA crossover)':
                if check_reverse_signal(df, i, position['type']):
                    exit_price = current_price
                    exit_reason = 'Reverse Signal'
            
            # Check SL hit
            if exit_price is None and position.get('sl') is not None:
                if position['type'] == 1 and current_price <= position['sl']:
                    exit_price = position['sl']
                    exit_reason = 'Stop Loss'
                elif position['type'] == -1 and current_price >= position['sl']:
                    exit_price = position['sl']
                    exit_reason = 'Stop Loss'
            
            # Check target hit (except for trailing target)
            if exit_price is None and position.get('target') is not None and target_type != 'Trailing Target (Points)':
                if position['type'] == 1 and current_price >= position['target']:
                    if target_type == '50% Exit at Target (Partial)' and not position.get('partial_exit_done', False):
                        partial_exit = True
                        position['partial_exit_done'] = True
                    else:
                        exit_price = position['target']
                        exit_reason = 'Target Hit'
                elif position['type'] == -1 and current_price <= position['target']:
                    if target_type == '50% Exit at Target (Partial)' and not position.get('partial_exit_done', False):
                        partial_exit = True
                        position['partial_exit_done'] = True
                    else:
                        exit_price = position['target']
                        exit_reason = 'Target Hit'
            
            # Handle partial exit
            if partial_exit:
                # Calculate partial P&L
                if position['type'] == 1:
                    pnl = (position['target'] - position['entry_price']) * (quantity * 0.5)
                else:
                    pnl = (position['entry_price'] - position['target']) * (quantity * 0.5)
                
                results['total_pnl'] += pnl
                # Continue with remaining position
            
            # Exit position
            if exit_price is not None:
                duration = (df.index[i] - position['entry_time']).total_seconds() / 3600
                
                # Calculate P&L
                if position.get('partial_exit_done', False):
                    # 50% already exited
                    if position['type'] == 1:
                        pnl = (exit_price - position['entry_price']) * (quantity * 0.5)
                    else:
                        pnl = (position['entry_price'] - exit_price) * (quantity * 0.5)
                else:
                    if position['type'] == 1:
                        pnl = (exit_price - position['entry_price']) * quantity
                    else:
                        pnl = (position['entry_price'] - exit_price) * quantity
                
                results['total_pnl'] += pnl
                results['total_trades'] += 1
                
                if pnl > 0:
                    results['winning_trades'] += 1
                else:
                    results['losing_trades'] += 1
                
                trade_range = position['highest_price'] - position['lowest_price']
                
                results['trades'].append({
                    'entry_time': position['entry_time'],
                    'exit_time': df.index[i],
                    'duration': duration,
                    'signal': 'LONG' if position['type'] == 1 else 'SHORT',
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'sl': position.get('sl', 0),
                    'target': position.get('target', 0),
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'highest': position['highest_price'],
                    'lowest': position['lowest_price'],
                    'range': trade_range
                })
                
                position = None
    
    # Calculate statistics
    if results['total_trades'] > 0:
        results['accuracy'] = (results['winning_trades'] / results['total_trades']) * 100
        total_duration = sum(t['duration'] for t in results['trades'])
        results['avg_duration'] = total_duration / results['total_trades']
    
    return results

# ==================== LIVE TRADING ENGINE ====================

def live_trading_loop(asset, ticker, interval, period, strategy, config, mode, placeholder):
    """Main live trading loop"""
    st.session_state['trading_active'] = True
    
    while st.session_state.get('trading_active', False):
        # Fetch latest data
        df = fetch_data(ticker, interval, period, mode)
        
        if df is None or df.empty:
            add_log("Error: Unable to fetch data")
            time.sleep(2)
            continue
        
        # Apply strategy
        if strategy == 'EMA Crossover':
            df = ema_crossover_strategy(df, config)
        elif strategy == 'Simple Buy':
            df = simple_buy_strategy(df, config)
        elif strategy == 'Simple Sell':
            df = simple_sell_strategy(df, config)
        elif strategy == 'Price Threshold':
            df = price_threshold_strategy(df, config)
        elif strategy == 'RSI-ADX-EMA':
            df = rsi_adx_ema_strategy(df, config)
        elif strategy == 'Percentage Change':
            df = percentage_change_strategy(df, config)
        elif strategy == 'AI Price Action':
            df = ai_price_action_strategy(df, config)
        elif strategy == 'Custom Builder':
            df = custom_strategy_builder(df, config.get('conditions', []))
        
        st.session_state['current_data'] = df
        
        # Get current values
        current_price = df['Close'].iloc[-1]
        current_signal = df['Signal'].iloc[-1]
        
        position = st.session_state.get('position')
        
        # Entry logic
        if position is None and current_signal != 0:
            entry_price = current_price
            signal = int(current_signal)
            
            position = {
                'type': signal,
                'entry_price': entry_price,
                'entry_time': df.index[-1],
                'highest_price': entry_price,
                'lowest_price': entry_price,
                'partial_exit_done': False,
                'breakeven_activated': False
            }
            
            # Calculate SL and Target
            sl_type = config.get('sl_type', 'Custom Points')
            target_type = config.get('target_type', 'Custom Points')
            
            position['sl'] = calculate_stop_loss(df, len(df)-1, position, sl_type, config)
            position['target'] = calculate_target(df, len(df)-1, position, target_type, config)
            
            st.session_state['position'] = position
            
            signal_text = 'LONG' if signal == 1 else 'SHORT'
            add_log(f"ENTRY: {signal_text} at {entry_price:.2f}")
            
            # Placeholder for Dhan API buy/sell
            # if signal == 1:
            #     dhan.place_order(security_id=ticker, transaction_type=dhan.BUY, quantity=config['quantity'])
            # else:
            #     dhan.place_order(security_id=ticker, transaction_type=dhan.SELL, quantity=config['quantity'])
        
        # Exit logic
        elif position is not None:
            # Update highest/lowest
            position['highest_price'] = max(position.get('highest_price', current_price), current_price)
            position['lowest_price'] = min(position.get('lowest_price', current_price), current_price)
            
            # Update SL if trailing
            sl_type = config.get('sl_type', 'Custom Points')
            target_type = config.get('target_type', 'Custom Points')
            
            if sl_type not in ['Signal-based (reverse EMA crossover)', 'Custom Points']:
                new_sl = calculate_stop_loss(df, len(df)-1, position, sl_type, config)
                if new_sl is not None:
                    if position['type'] == 1 and new_sl > position.get('sl', 0):
                        position['sl'] = new_sl
                        if sl_type == 'Break-even After 50% Target' and new_sl == position['entry_price'] and not position.get('breakeven_activated', False):
                            position['breakeven_activated'] = True
                            add_log("Stop Loss moved to break-even")
                    elif position['type'] == -1 and new_sl < position.get('sl', float('inf')):
                        position['sl'] = new_sl
                        if sl_type == 'Break-even After 50% Target' and new_sl == position['entry_price'] and not position.get('breakeven_activated', False):
                            position['breakeven_activated'] = True
                            add_log("Stop Loss moved to break-even")
            
            st.session_state['position'] = position
            
            # Check exits
            exit_price = None
            exit_reason = None
            
            # Signal-based exit
            if sl_type == 'Signal-based (reverse EMA crossover)' or target_type == 'Signal-based (reverse EMA crossover)':
                if check_reverse_signal(df, len(df)-1, position['type']):
                    exit_price = current_price
                    exit_reason = 'Reverse Signal'
            
            # SL check
            if exit_price is None and position.get('sl') is not None:
                if position['type'] == 1 and current_price <= position['sl']:
                    exit_price = position['sl']
                    exit_reason = 'Stop Loss Hit'
                elif position['type'] == -1 and current_price >= position['sl']:
                    exit_price = position['sl']
                    exit_reason = 'Stop Loss Hit'
            
            # Target check
            if exit_price is None and position.get('target') is not None and target_type != 'Trailing Target (Points)':
                if position['type'] == 1 and current_price >= position['target']:
                    if target_type == '50% Exit at Target (Partial)' and not position.get('partial_exit_done', False):
                        position['partial_exit_done'] = True
                        st.session_state['position'] = position
                        add_log("50% position exited - trailing remaining")
                    else:
                        exit_price = position['target']
                        exit_reason = 'Target Hit'
                elif position['type'] == -1 and current_price <= position['target']:
                    if target_type == '50% Exit at Target (Partial)' and not position.get('partial_exit_done', False):
                        position['partial_exit_done'] = True
                        st.session_state['position'] = position
                        add_log("50% position exited - trailing remaining")
                    else:
                        exit_price = position['target']
                        exit_reason = 'Target Hit'
            
            # Exit
            if exit_price is not None:
                duration = (df.index[-1] - position['entry_time']).total_seconds() / 3600
                
                quantity = config.get('quantity', 1)
                if position.get('partial_exit_done', False):
                    if position['type'] == 1:
                        pnl = (exit_price - position['entry_price']) * (quantity * 0.5)
                    else:
                        pnl = (position['entry_price'] - exit_price) * (quantity * 0.5)
                else:
                    if position['type'] == 1:
                        pnl = (exit_price - position['entry_price']) * quantity
                    else:
                        pnl = (position['entry_price'] - exit_price) * quantity
                
                trade_range = position['highest_price'] - position['lowest_price']
                
                trade_record = {
                    'entry_time': position['entry_time'],
                    'exit_time': df.index[-1],
                    'duration': duration,
                    'signal': 'LONG' if position['type'] == 1 else 'SHORT',
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'sl': position.get('sl', 0),
                    'target': position.get('target', 0),
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'highest': position['highest_price'],
                    'lowest': position['lowest_price'],
                    'range': trade_range
                }
                
                st.session_state['trade_history'].append(trade_record)
                st.session_state['trade_history'] = st.session_state['trade_history']
                
                add_log(f"EXIT: {exit_reason} at {exit_price:.2f} | P&L: {pnl:.2f}")
                
                # Placeholder for Dhan API exit
                # if position['type'] == 1:
                #     dhan.place_order(security_id=ticker, transaction_type=dhan.SELL, quantity=config['quantity'])
                # else:
                #     dhan.place_order(security_id=ticker, transaction_type=dhan.BUY, quantity=config['quantity'])
                
                reset_position_state()
        
        # Update UI placeholder with current data
        with placeholder.container():
            display_live_dashboard(df, position, config, asset, interval, quantity)
        
        # Wait before next iteration
        time.sleep(random.uniform(1.0, 1.5))

def display_live_dashboard(df, position, config, asset, interval, quantity):
    """Display live dashboard content"""
    if df is None or df.empty:
        st.info("⏳ Waiting for data...")
        return
    
    current_price = df['Close'].iloc[-1]
    current_signal = df['Signal'].iloc[-1] if 'Signal' in df.columns else 0
    
    st.subheader("📊 Live Metrics")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Current Price", f"{current_price:.2f}")
        if position:
            st.metric("Entry Price", f"{position['entry_price']:.2f}")
    
    with metric_col2:
        if position:
            status_text = "LONG" if position['type'] == 1 else "SHORT"
            st.metric("Position", status_text)
            
            # Calculate unrealized P&L
            if position['type'] == 1:
                unrealized_pnl = (current_price - position['entry_price']) * quantity
            else:
                unrealized_pnl = (position['entry_price'] - current_price) * quantity
            
            if unrealized_pnl >= 0:
                st.metric("Unrealized P&L", f"{unrealized_pnl:.2f}", delta=f"+{unrealized_pnl:.2f}")
            else:
                st.metric("Unrealized P&L", f"{unrealized_pnl:.2f}", delta=f"{unrealized_pnl:.2f}", delta_color="inverse")
        else:
            st.metric("Position", "None")
    
    with metric_col3:
        if 'EMA_Fast' in df.columns:
            st.metric("EMA Fast", f"{df['EMA_Fast'].iloc[-1]:.2f}")
        if 'EMA_Slow' in df.columns:
            st.metric("EMA Slow", f"{df['EMA_Slow'].iloc[-1]:.2f}")
    
    with metric_col4:
        if 'RSI' in df.columns:
            st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
        
        signal_text = "NONE"
        if current_signal == 1:
            signal_text = "🟢 BUY"
        elif current_signal == -1:
            signal_text = "🔴 SELL"
        st.metric("Current Signal", signal_text)
    
    # Position Details
    if position:
        st.divider()
        st.subheader("📌 Position Information")
        
        pos_col1, pos_col2, pos_col3, pos_col4 = st.columns(4)
        
        with pos_col1:
            entry_time = position['entry_time']
            now_time = df.index[-1]
            duration = (now_time - entry_time).total_seconds() / 3600
            st.metric("Entry Time", entry_time.strftime("%H:%M:%S"))
            st.metric("Duration (hours)", f"{duration:.2f}")
        
        with pos_col2:
            sl_val = position.get('sl', 0)
            if sl_val:
                st.metric("Stop Loss", f"{sl_val:.2f}")
                if position['type'] == 1:
                    dist_to_sl = current_price - sl_val
                else:
                    dist_to_sl = sl_val - current_price
                st.metric("Distance to SL", f"{dist_to_sl:.2f}")
        
        with pos_col3:
            target_val = position.get('target', 0)
            if target_val:
                st.metric("Target", f"{target_val:.2f}")
                if position['type'] == 1:
                    dist_to_target = target_val - current_price
                else:
                    dist_to_target = current_price - target_val
                st.metric("Distance to Target", f"{dist_to_target:.2f}")
        
        with pos_col4:
            st.metric("Highest", f"{position.get('highest_price', current_price):.2f}")
            st.metric("Lowest", f"{position.get('lowest_price', current_price):.2f}")
            range_val = position.get('highest_price', current_price) - position.get('lowest_price', current_price)
            st.metric("Range", f"{range_val:.2f}")
        
        if position.get('breakeven_activated', False):
            st.success("✅ Stop Loss moved to break-even")
        
        if position.get('partial_exit_done', False):
            st.info("ℹ️ 50% position already exited - Trailing remaining")
    
    # Live Chart
    st.divider()
    st.subheader("📈 Live Chart")
    
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    )])
    
    if 'EMA_Fast' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], mode='lines', name='EMA Fast', line=dict(color='blue')))
    
    if 'EMA_Slow' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], mode='lines', name='EMA Slow', line=dict(color='orange')))
    
    if position:
        # Entry line
        fig.add_hline(y=position['entry_price'], line_dash="dash", line_color="yellow", annotation_text="Entry")
        
        # SL line
        if position.get('sl'):
            fig.add_hline(y=position['sl'], line_dash="dash", line_color="red", annotation_text="SL")
        
        # Target line
        if position.get('target'):
            fig.add_hline(y=position['target'], line_dash="dash", line_color="green", annotation_text="Target")
    
    fig.update_layout(
        title=f"{asset} - {interval}",
        xaxis_title="Time",
        yaxis_title="Price",
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    chart_key = f"live_chart_{int(time.time() * 1000)}"
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

# ==================== STREAMLIT UI ====================

def main():
    st.set_page_config(page_title="Quantitative Trading System", layout="wide")
    
    init_session_state()
    
    st.title("📊 Professional Quantitative Trading System")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode Selection
        mode = st.selectbox("Mode", ["Live Trading", "Backtest"], key="mode_select")
        
        # Asset Selection
        asset = st.selectbox("Asset", list(ASSET_TICKERS.keys()) + ["Custom"], key="asset_select")
        
        if asset == "Custom":
            ticker = st.text_input("Custom Ticker", value="AAPL", key="custom_ticker")
        else:
            ticker = ASSET_TICKERS[asset]
        
        # Timeframe Selection
        interval = st.selectbox("Interval", list(INTERVAL_PERIODS.keys()), key="interval_select")
        
        allowed_periods = INTERVAL_PERIODS[interval]
        period = st.selectbox("Period", allowed_periods, key="period_select")
        
        # Quantity
        quantity = st.number_input("Quantity", min_value=1, value=1, key="quantity_input")
        
        # Strategy Selection
        st.subheader("Strategy")
        strategy = st.selectbox("Select Strategy", [
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
        
        # Strategy-specific configuration
        if strategy == 'EMA Crossover':
            config['ema_fast'] = st.number_input("EMA Fast", min_value=1, value=9, key="ema_fast")
            config['ema_slow'] = st.number_input("EMA Slow", min_value=1, value=15, key="ema_slow")
            config['min_angle'] = st.number_input("Minimum Angle (degrees)", min_value=0.0, value=1.0, step=0.1, key="min_angle")
            config['entry_filter'] = st.selectbox("Entry Filter", ["Simple Crossover", "Custom Candle (Points)", "ATR-based Candle"], key="entry_filter")
            
            if config['entry_filter'] == 'Custom Candle (Points)':
                config['custom_points'] = st.number_input("Custom Points", min_value=1.0, value=10.0, step=1.0, key="custom_points")
            elif config['entry_filter'] == 'ATR-based Candle':
                config['atr_multiplier'] = st.number_input("ATR Multiplier", min_value=0.1, value=1.5, step=0.1, key="atr_mult")
            
            config['use_adx'] = st.checkbox("Use ADX Filter", value=False, key="use_adx")
            if config['use_adx']:
                config['adx_period'] = st.number_input("ADX Period", min_value=1, value=14, key="adx_period")
                config['adx_threshold'] = st.number_input("ADX Threshold", min_value=1.0, value=25.0, step=1.0, key="adx_threshold")
        
        elif strategy == 'Price Threshold':
            config['threshold'] = st.number_input("Threshold Price", min_value=0.0, value=100.0, step=1.0, key="threshold_price")
            config['direction'] = st.selectbox("Direction", [
                "LONG (Price >= Threshold)",
                "SHORT (Price >= Threshold)",
                "LONG (Price <= Threshold)",
                "SHORT (Price <= Threshold)"
            ], key="threshold_direction")
        
        elif strategy == 'RSI-ADX-EMA':
            config['rsi_period'] = st.number_input("RSI Period", min_value=1, value=14, key="rsi_period")
            config['adx_period'] = st.number_input("ADX Period", min_value=1, value=14, key="adx_period_rae")
            config['ema1_period'] = st.number_input("EMA1 Period", min_value=1, value=9, key="ema1_period")
            config['ema2_period'] = st.number_input("EMA2 Period", min_value=1, value=21, key="ema2_period")
        
        elif strategy == 'Percentage Change':
            config['pct_threshold'] = st.number_input("Percentage Threshold", min_value=0.01, value=0.5, step=0.01, key="pct_threshold")
            config['pct_direction'] = st.selectbox("Direction", ["BUY on Fall", "SELL on Fall", "BUY on Rise", "SELL on Rise"], key="pct_direction")
        
        elif strategy == 'Custom Builder':
            st.write("### Custom Conditions")
            
            if 'custom_conditions' not in st.session_state or len(st.session_state['custom_conditions']) == 0:
                st.session_state['custom_conditions'] = [{'use_condition': True, 'indicator': 'RSI', 'operator': '>', 'value': 70, 'action': 'SELL', 'compare_price': False, 'compare_indicator': 'EMA_20'}]
            
            for idx, cond in enumerate(st.session_state['custom_conditions']):
                with st.expander(f"Condition {idx+1}", expanded=True):
                    cond['use_condition'] = st.checkbox("Use", value=cond.get('use_condition', True), key=f"use_cond_{idx}")
                    cond['compare_price'] = st.checkbox("Compare Price with Indicator", value=cond.get('compare_price', False), key=f"comp_price_{idx}")
                    
                    if cond['compare_price']:
                        col1, col2 = st.columns(2)
                        with col1:
                            cond['indicator'] = 'Price'
                            st.text("Price")
                        with col2:
                            cond['compare_indicator'] = st.selectbox("vs Indicator", ['EMA_20', 'EMA_50', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower'], key=f"comp_ind_{idx}")
                    else:
                        cond['indicator'] = st.selectbox("Indicator", ['Price', 'RSI', 'ADX', 'EMA_20', 'EMA_50', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR', 'Close', 'High', 'Low'], key=f"ind_{idx}")
                    
                    cond['operator'] = st.selectbox("Operator", ['>', '<', '>=', '<=', '==', 'crosses_above', 'crosses_below'], key=f"op_{idx}")
                    
                    if not cond['compare_price']:
                        cond['value'] = st.number_input("Value", value=cond.get('value', 0), key=f"val_{idx}")
                    
                    cond['action'] = st.selectbox("Action", ['BUY', 'SELL'], key=f"act_{idx}")
                    
                    if st.button(f"Remove Condition {idx+1}", key=f"remove_{idx}"):
                        st.session_state['custom_conditions'].pop(idx)
                        st.rerun()
            
            if st.button("Add Condition"):
                st.session_state['custom_conditions'].append({
                    'use_condition': True,
                    'indicator': 'RSI',
                    'operator': '>',
                    'value': 50,
                    'action': 'BUY',
                    'compare_price': False,
                    'compare_indicator': 'EMA_20'
                })
                st.rerun()
            
            config['conditions'] = st.session_state['custom_conditions']
        
        # Stop Loss Configuration
        st.subheader("Stop Loss")
        sl_type = st.selectbox("SL Type", [
            "Custom Points",
            "Trailing SL (Points)",
            "Trailing SL + Current Candle",
            "Trailing SL + Previous Candle",
            "ATR-based",
            "Current Candle Low/High",
            "Previous Candle Low/High",
            "Signal-based (reverse EMA crossover)",
            "Break-even After 50% Target",
            "Volatility-Adjusted Trailing SL"
        ], key="sl_type_select")
        
        config['sl_type'] = sl_type
        
        if sl_type not in ['Signal-based (reverse EMA crossover)']:
            config['sl_points'] = st.number_input("SL Points", min_value=1.0, value=10.0, step=1.0, key="sl_points")
        
        if 'ATR' in sl_type:
            config['sl_atr_multiplier'] = st.number_input("ATR Multiplier", min_value=0.1, value=1.5, step=0.1, key="sl_atr_mult")
        
        if 'Trailing' in sl_type:
            config['trailing_threshold'] = st.number_input("Trailing Threshold (Points)", min_value=0.0, value=0.0, step=1.0, key="trailing_threshold")
        
        # Target Configuration
        st.subheader("Target")
        target_type = st.selectbox("Target Type", [
            "Custom Points",
            "Trailing Target (Points)",
            "ATR-based",
            "Risk-Reward Based",
            "Current Candle Low/High",
            "Previous Candle Low/High",
            "Signal-based (reverse EMA crossover)",
            "50% Exit at Target (Partial)"
        ], key="target_type_select")
        
        config['target_type'] = target_type
        
        if target_type not in ['Signal-based (reverse EMA crossover)']:
            config['target_points'] = st.number_input("Target Points", min_value=1.0, value=20.0, step=1.0, key="target_points")
        
        if target_type == 'ATR-based':
            config['target_atr_multiplier'] = st.number_input("Target ATR Multiplier", min_value=0.1, value=2.0, step=0.1, key="target_atr_mult")
        
        if target_type == 'Risk-Reward Based':
            config['rr_ratio'] = st.number_input("Risk-Reward Ratio", min_value=0.1, value=2.0, step=0.1, key="rr_ratio")
    
    # Main Content - Tabs
    if mode == "Live Trading":
        tab1, tab2, tab3 = st.tabs(["📊 Live Dashboard", "📈 Trade History", "📝 Trade Logs"])
        
        with tab1:
            # Trading Controls
            col1, col2, col3 = st.columns([2, 2, 3])
            
            with col1:
                start_disabled = st.session_state.get('trading_active', False)
                if st.button("▶️ Start Trading", type="primary", use_container_width=True, disabled=start_disabled):
                    # Create placeholder for live updates
                    st.session_state['live_placeholder'] = st.empty()
                    
                    # Display active configuration
                    st.subheader("📋 Active Configuration")
                    conf_col1, conf_col2, conf_col3 = st.columns(3)
                    
                    with conf_col1:
                        st.metric("Asset", asset)
                        st.metric("Interval", interval)
                        st.metric("Period", period)
                    
                    with conf_col2:
                        st.metric("Quantity", quantity)
                        st.metric("Strategy", strategy)
                        st.metric("SL Type", sl_type)
                    
                    with conf_col3:
                        sl_val_str = "Signal Based" if sl_type == "Signal-based (reverse EMA crossover)" else f"{config.get('sl_points', 0):.2f}"
                        st.metric("SL Points", sl_val_str)
                        st.metric("Target Type", target_type)
                        target_val_str = "Signal Based" if target_type == "Signal-based (reverse EMA crossover)" else f"{config.get('target_points', 0):.2f}"
                        st.metric("Target Points", target_val_str)
                    
                    st.divider()
                    
                    # Start trading with placeholder
                    placeholder = st.empty()
                    add_log("Trading started")
                    live_trading_loop(asset, ticker, interval, period, strategy, config, mode, placeholder)
            
            with col2:
                stop_disabled = not st.session_state.get('trading_active', False)
                if st.button("⏹️ Stop Trading", use_container_width=True, disabled=stop_disabled):
                    if st.session_state.get('trading_active', False):
                        st.session_state['trading_active'] = False
                        
                        # Close open position if exists
                        position = st.session_state.get('position')
                        if position is not None:
                            df = st.session_state.get('current_data')
                            if df is not None and not df.empty:
                                current_price = df['Close'].iloc[-1]
                                duration = (df.index[-1] - position['entry_time']).total_seconds() / 3600
                                
                                quantity = config.get('quantity', 1)
                                if position['type'] == 1:
                                    pnl = (current_price - position['entry_price']) * quantity
                                else:
                                    pnl = (position['entry_price'] - current_price) * quantity
                                
                                trade_range = position.get('highest_price', current_price) - position.get('lowest_price', current_price)
                                
                                trade_record = {
                                    'entry_time': position['entry_time'],
                                    'exit_time': df.index[-1],
                                    'duration': duration,
                                    'signal': 'LONG' if position['type'] == 1 else 'SHORT',
                                    'entry_price': position['entry_price'],
                                    'exit_price': current_price,
                                    'sl': position.get('sl', 0),
                                    'target': position.get('target', 0),
                                    'exit_reason': 'Manual Close',
                                    'pnl': pnl,
                                    'highest': position.get('highest_price', current_price),
                                    'lowest': position.get('lowest_price', current_price),
                                    'range': trade_range
                                }
                                
                                st.session_state['trade_history'].append(trade_record)
                                st.session_state['trade_history'] = st.session_state['trade_history']
                                
                                add_log(f"Manual Close at {current_price:.2f} | P&L: {pnl:.2f}")
                        
                        reset_position_state()
                        add_log("Trading stopped")
                        st.rerun()
            
            with col3:
                if st.session_state.get('trading_active', False):
                    st.success("🟢 Trading is ACTIVE")
                else:
                    st.info("⚪ Trading is STOPPED")
            
            st.divider()
            
            # Show live data if available
            if not st.session_state.get('trading_active', False):
                # Active Configuration Display
                st.subheader("📋 Active Configuration")
                conf_col1, conf_col2, conf_col3 = st.columns(3)
                
                with conf_col1:
                    st.metric("Asset", asset)
                    st.metric("Interval", interval)
                    st.metric("Period", period)
                
                with conf_col2:
                    st.metric("Quantity", quantity)
                    st.metric("Strategy", strategy)
                    st.metric("SL Type", sl_type)
                
                with conf_col3:
                    sl_val_str = "Signal Based" if sl_type == "Signal-based (reverse EMA crossover)" else f"{config.get('sl_points', 0):.2f}"
                    st.metric("SL Points", sl_val_str)
                    st.metric("Target Type", target_type)
                    target_val_str = "Signal Based" if target_type == "Signal-based (reverse EMA crossover)" else f"{config.get('target_points', 0):.2f}"
                    st.metric("Target Points", target_val_str)
                
                st.divider()
                st.info("Click 'Start Trading' to begin live monitoring")
        
        with tab2:
            st.markdown("### 📈 Trade History")
            
            trade_history = st.session_state.get('trade_history', [])
            
            if len(trade_history) == 0:
                st.info("No trades yet")
            else:
                # Calculate statistics
                total_trades = len(trade_history)
                winning_trades = sum(1 for t in trade_history if t['pnl'] > 0)
                losing_trades = sum(1 for t in trade_history if t['pnl'] <= 0)
                total_pnl = sum(t['pnl'] for t in trade_history)
                accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                # Display metrics
                met_col1, met_col2, met_col3, met_col4, met_col5 = st.columns(5)
                
                with met_col1:
                    st.metric("Total Trades", total_trades)
                
                with met_col2:
                    st.metric("Winning Trades", winning_trades)
                
                with met_col3:
                    st.metric("Losing Trades", losing_trades)
                
                with met_col4:
                    st.metric("Accuracy", f"{accuracy:.2f}%")
                
                with met_col5:
                    if total_pnl >= 0:
                        st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"+{total_pnl:.2f}")
                    else:
                        st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color="inverse")
                
                st.divider()
                
                # Display trades
                for idx, trade in enumerate(reversed(trade_history)):
                    with st.expander(f"Trade #{total_trades - idx} - {trade['signal']} - P&L: {trade['pnl']:.2f}"):
                        trade_col1, trade_col2 = st.columns(2)
                        
                        with trade_col1:
                            st.write(f"**Entry Time:** {trade.get('entry_time', 'N/A')}")
                            st.write(f"**Exit Time:** {trade.get('exit_time', 'N/A')}")
                            st.write(f"**Duration:** {trade.get('duration', 0):.2f} hours")
                            st.write(f"**Signal:** {trade.get('signal', 'N/A')}")
                            st.write(f"**Entry Price:** {trade.get('entry_price', 0):.2f}")
                        
                        with trade_col2:
                            exit_price_val = trade.get('exit_price', 0)
                            sl_val = trade.get('sl')
                            target_val = trade.get('target')
                            
                            exit_price_str = f"{exit_price_val:.2f}" if exit_price_val else "N/A"
                            sl_str = f"{sl_val:.2f}" if sl_val is not None else "Signal Based"
                            target_str = f"{target_val:.2f}" if target_val is not None else "Signal Based"
                            
                            st.write(f"**Exit Price:** {exit_price_str}")
                            st.write(f"**Stop Loss:** {sl_str}")
                            st.write(f"**Target:** {target_str}")
                            st.write(f"**Exit Reason:** {trade.get('exit_reason', 'N/A')}")
                            
                            pnl = trade.get('pnl', 0)
                            pnl_color = "green" if pnl > 0 else "red"
                            st.markdown(f"**P&L:** <span style='color:{pnl_color}'>{pnl:.2f}</span>", unsafe_allow_html=True)
                        
                        st.write(f"**Highest:** {trade.get('highest', 0):.2f} | **Lowest:** {trade.get('lowest', 0):.2f} | **Range:** {trade.get('range', 0):.2f}")
        
        with tab3:
            st.markdown("### 📝 Trade Logs")
            
            trade_logs = st.session_state.get('trade_logs', [])
            
            if len(trade_logs) == 0:
                st.info("No logs yet")
            else:
                for log in reversed(trade_logs):
                    st.text(log)
    
    else:  # Backtest mode
        tab1, tab2, tab3 = st.tabs(["📊 Configuration", "📈 Backtest Results", "📊 Market Data Analysis"])
        
        with tab1:
            st.subheader("Backtest Configuration")
            st.write("Configure your strategy in the sidebar and click 'Run Backtest' in the Backtest Results tab.")
            
            # Display current configuration
            st.json({
                'Asset': asset,
                'Ticker': ticker,
                'Interval': interval,
                'Period': period,
                'Quantity': quantity,
                'Strategy': strategy,
                'SL Type': sl_type,
                'Target Type': target_type
            })
        
        with tab2:
            st.markdown("### 📈 Backtest Results")
            
            if st.button("▶️ Run Backtest", type="primary"):
                with st.spinner("Running backtest..."):
                    # Clear any cached data
                    if 'current_data' in st.session_state:
                        st.session_state['current_data'] = None
                    
                    # Fetch data
                    progress_bar = st.progress(0)
                    st.write("Fetching data...")
                    progress_bar.progress(20)
                    
                    df = fetch_data(ticker, interval, period, mode)
                    
                    if df is None or df.empty:
                        st.error("Unable to fetch data. Please check your parameters.")
                    else:
                        progress_bar.progress(40)
                        st.write("Applying strategy...")
                        
                        # Apply strategy
                        if strategy == 'EMA Crossover':
                            df = ema_crossover_strategy(df, config)
                        elif strategy == 'Simple Buy':
                            df = simple_buy_strategy(df, config)
                        elif strategy == 'Simple Sell':
                            df = simple_sell_strategy(df, config)
                        elif strategy == 'Price Threshold':
                            df = price_threshold_strategy(df, config)
                        elif strategy == 'RSI-ADX-EMA':
                            df = rsi_adx_ema_strategy(df, config)
                        elif strategy == 'Percentage Change':
                            df = percentage_change_strategy(df, config)
                        elif strategy == 'AI Price Action':
                            df = ai_price_action_strategy(df, config)
                        elif strategy == 'Custom Builder':
                            df = custom_strategy_builder(df, config.get('conditions', []))
                        
                        progress_bar.progress(60)
                        st.write("Running backtest...")
                        
                        # Run backtest
                        results = run_backtest(df, strategy, config)
                        
                        progress_bar.progress(100)
                        st.success("Backtest completed!")
                        
                        # Display results
                        st.divider()
                        
                        # Metrics
                        res_col1, res_col2, res_col3, res_col4, res_col5 = st.columns(5)
                        
                        with res_col1:
                            st.metric("Total Trades", results['total_trades'])
                        
                        with res_col2:
                            st.metric("Winning Trades", results['winning_trades'])
                        
                        with res_col3:
                            st.metric("Losing Trades", results['losing_trades'])
                        
                        with res_col4:
                            st.metric("Accuracy", f"{results['accuracy']:.2f}%")
                        
                        with res_col5:
                            pnl = results['total_pnl']
                            if pnl >= 0:
                                st.metric("Total P&L", f"{pnl:.2f}", delta=f"+{pnl:.2f}")
                            else:
                                st.metric("Total P&L", f"{pnl:.2f}", delta=f"{pnl:.2f}", delta_color="inverse")
                        
                        st.metric("Avg Duration (hours)", f"{results['avg_duration']:.2f}")
                        
                        st.divider()
                        
                        # Trade list
                        if len(results['trades']) > 0:
                            st.subheader("All Trades")
                            
                            for idx, trade in enumerate(results['trades']):
                                with st.expander(f"Trade #{idx+1} - {trade['signal']} - P&L: {trade['pnl']:.2f}"):
                                    trade_col1, trade_col2 = st.columns(2)
                                    
                                    with trade_col1:
                                        st.write(f"**Entry Time:** {trade['entry_time']}")
                                        st.write(f"**Exit Time:** {trade['exit_time']}")
                                        st.write(f"**Duration:** {trade['duration']:.2f} hours")
                                        st.write(f"**Signal:** {trade['signal']}")
                                        st.write(f"**Entry Price:** {trade['entry_price']:.2f}")
                                    
                                    with trade_col2:
                                        st.write(f"**Exit Price:** {trade['exit_price']:.2f}")
                                        st.write(f"**Stop Loss:** {trade['sl']:.2f}")
                                        st.write(f"**Target:** {trade['target']:.2f}")
                                        st.write(f"**Exit Reason:** {trade['exit_reason']}")
                                        
                                        pnl = trade['pnl']
                                        pnl_color = "green" if pnl > 0 else "red"
                                        st.markdown(f"**P&L:** <span style='color:{pnl_color}'>{pnl:.2f}</span>", unsafe_allow_html=True)
                                    
                                    st.write(f"**Highest:** {trade['highest']:.2f} | **Lowest:** {trade['lowest']:.2f} | **Range:** {trade['range']:.2f}")
        
        with tab3:
            st.markdown("### 📊 Market Data Analysis")
            
            if st.button("📥 Load Market Data", type="primary"):
                with st.spinner("Loading market data..."):
                    # Fetch fresh data
                    raw_df = fetch_data(ticker, interval, period, mode)
                    
                    if raw_df is None or raw_df.empty:
                        st.error("Unable to fetch market data")
                    else:
                        # Prepare data with additional columns
                        analysis_df = raw_df[['Open', 'High', 'Low', 'Close']].copy()
                        
                        # Calculate changes
                        analysis_df['Change_Points'] = analysis_df['Close'].diff()
                        analysis_df['Change_Pct'] = analysis_df['Close'].pct_change() * 100
                        
                        # Add day of week
                        analysis_df['Day_of_Week'] = analysis_df.index.day_name()
                        
                        # Store in session state
                        st.session_state['market_data'] = analysis_df
                        st.success("Market data loaded successfully!")
                        st.rerun()
            
            if 'market_data' in st.session_state and st.session_state['market_data'] is not None:
                analysis_df = st.session_state['market_data']
                
                st.divider()
                
                # Display data table
                st.subheader("📋 Market Data Table")
                
                # Create display dataframe with colored changes
                display_df = analysis_df.copy()
                display_df = display_df.reset_index()
                display_df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Change (Points)', 'Change (%)', 'Day of Week']
                
                # Format numeric columns
                for col in ['Open', 'High', 'Low', 'Close', 'Change (Points)', 'Change (%)']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].round(2)
                
                st.dataframe(display_df, use_container_width=True, height=400)
                
                st.divider()
                
                # Plot 1: Change in Points over Time
                st.subheader("📈 Change in Points Over Time")
                
                fig_points = go.Figure()
                
                colors = ['green' if val >= 0 else 'red' for val in analysis_df['Change_Points']]
                
                fig_points.add_trace(go.Bar(
                    x=analysis_df.index,
                    y=analysis_df['Change_Points'],
                    marker_color=colors,
                    name='Change (Points)'
                ))
                
                fig_points.update_layout(
                    title='Daily Change in Points',
                    xaxis_title='Date',
                    yaxis_title='Change (Points)',
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_points, use_container_width=True)
                
                # Plot 2: Change in Percentage over Time
                st.subheader("📊 Change in Percentage Over Time")
                
                fig_pct = go.Figure()
                
                colors_pct = ['green' if val >= 0 else 'red' for val in analysis_df['Change_Pct']]
                
                fig_pct.add_trace(go.Bar(
                    x=analysis_df.index,
                    y=analysis_df['Change_Pct'],
                    marker_color=colors_pct,
                    name='Change (%)'
                ))
                
                fig_pct.update_layout(
                    title='Daily Change in Percentage',
                    xaxis_title='Date',
                    yaxis_title='Change (%)',
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_pct, use_container_width=True)
                
                st.divider()
                
                # Heatmap 1: Returns Heatmap
                st.subheader("🔥 Returns Heatmap (Percentage)")
                
                # Prepare data for heatmap - group by day of week and hour/date
                heatmap_df = analysis_df.copy()
                heatmap_df['Date'] = heatmap_df.index.date
                heatmap_df['Hour'] = heatmap_df.index.hour
                
                # Create pivot table for heatmap
                if interval in ['1m', '5m', '15m', '30m', '1h', '4h']:
                    # Intraday - use hours
                    pivot_pct = heatmap_df.pivot_table(
                        values='Change_Pct',
                        index='Date',
                        columns='Hour',
                        aggfunc='mean'
                    )
                else:
                    # Daily or higher - use day of week
                    heatmap_df['Week'] = heatmap_df.index.isocalendar().week
                    pivot_pct = heatmap_df.pivot_table(
                        values='Change_Pct',
                        index='Week',
                        columns=heatmap_df.index.day_name(),
                        aggfunc='mean'
                    )
                
                fig_heatmap_pct = go.Figure(data=go.Heatmap(
                    z=pivot_pct.values,
                    x=pivot_pct.columns,
                    y=pivot_pct.index,
                    colorscale='RdYlGn',
                    zmid=0,
                    text=np.round(pivot_pct.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(title="Change (%)")
                ))
                
                fig_heatmap_pct.update_layout(
                    title='Returns Heatmap by Time Period',
                    xaxis_title='Hour' if interval in ['1m', '5m', '15m', '30m', '1h', '4h'] else 'Day of Week',
                    yaxis_title='Date/Week',
                    height=600
                )
                
                st.plotly_chart(fig_heatmap_pct, use_container_width=True)
                
                st.divider()
                
                # Heatmap 2: Points Heatmap
                st.subheader("🔥 Points Change Heatmap")
                
                if interval in ['1m', '5m', '15m', '30m', '1h', '4h']:
                    pivot_points = heatmap_df.pivot_table(
                        values='Change_Points',
                        index='Date',
                        columns='Hour',
                        aggfunc='mean'
                    )
                else:
                    pivot_points = heatmap_df.pivot_table(
                        values='Change_Points',
                        index='Week',
                        columns=heatmap_df.index.day_name(),
                        aggfunc='mean'
                    )
                
                fig_heatmap_points = go.Figure(data=go.Heatmap(
                    z=pivot_points.values,
                    x=pivot_points.columns,
                    y=pivot_points.index,
                    colorscale='RdYlGn',
                    zmid=0,
                    text=np.round(pivot_points.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(title="Change (Points)")
                ))
                
                fig_heatmap_points.update_layout(
                    title='Points Change Heatmap by Time Period',
                    xaxis_title='Hour' if interval in ['1m', '5m', '15m', '30m', '1h', '4h'] else 'Day of Week',
                    yaxis_title='Date/Week',
                    height=600
                )
                
                st.plotly_chart(fig_heatmap_points, use_container_width=True)
                
                # Summary statistics
                st.divider()
                st.subheader("📊 Summary Statistics")
                
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                
                with stats_col1:
                    avg_change_points = analysis_df['Change_Points'].mean()
                    st.metric("Avg Change (Points)", f"{avg_change_points:.2f}")
                
                with stats_col2:
                    avg_change_pct = analysis_df['Change_Pct'].mean()
                    st.metric("Avg Change (%)", f"{avg_change_pct:.2f}%")
                
                with stats_col3:
                    positive_days = (analysis_df['Change_Points'] > 0).sum()
                    total_days = len(analysis_df)
                    win_rate = (positive_days / total_days * 100) if total_days > 0 else 0
                    st.metric("Positive Days", f"{positive_days}/{total_days}")
                    st.metric("Win Rate", f"{win_rate:.2f}%")
                
                with stats_col4:
                    max_gain = analysis_df['Change_Points'].max()
                    max_loss = analysis_df['Change_Points'].min()
                    st.metric("Max Gain (Points)", f"{max_gain:.2f}")
                    st.metric("Max Loss (Points)", f"{max_loss:.2f}")
            
            else:
                st.info("Click 'Load Market Data' to view market analysis")

if __name__ == "__main__":
    main()
