import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import pytz
import math

# ==========================================
# 1. CONFIGURATION & STATE INITIALIZATION
# ==========================================

st.set_page_config(
    page_title="Pro Quantitative Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'init_done' not in st.session_state:
    st.session_state['init_done'] = True
    st.session_state['trading_active'] = False
    st.session_state['current_data'] = None
    st.session_state['position'] = None  # {type: 1/-1, entry_price, quantity, sl, target, entry_time, params...}
    st.session_state['trade_history'] = []
    st.session_state['trade_logs'] = []
    
    # Trailing & State Variables
    st.session_state['trailing_sl_high'] = None
    st.session_state['trailing_sl_low'] = None
    st.session_state['trailing_target_high'] = None
    st.session_state['trailing_target_low'] = None
    st.session_state['trailing_profit_points'] = 0.0
    st.session_state['threshold_crossed'] = False
    st.session_state['highest_price'] = None
    st.session_state['lowest_price'] = None
    st.session_state['partial_exit_done'] = False
    st.session_state['breakeven_activated'] = False
    st.session_state['custom_conditions'] = [] # List of dicts for custom builder

def add_log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state['trade_logs'].append(log_entry)
    # Keep only last 50 logs
    if len(st.session_state['trade_logs']) > 50:
        st.session_state['trade_logs'] = st.session_state['trade_logs'][-50:]

def reset_position_state():
    st.session_state['position'] = None
    st.session_state['trailing_sl_high'] = None
    st.session_state['trailing_sl_low'] = None
    st.session_state['trailing_target_high'] = None
    st.session_state['trailing_target_low'] = None
    st.session_state['trailing_profit_points'] = 0.0
    st.session_state['threshold_crossed'] = False
    st.session_state['highest_price'] = None
    st.session_state['lowest_price'] = None
    st.session_state['partial_exit_done'] = False
    st.session_state['breakeven_activated'] = False

# ==========================================
# 2. UTILITY FUNCTIONS & INDICATORS
# ==========================================

def get_ist_time():
    utc_now = datetime.now(pytz.utc)
    ist = pytz.timezone('Asia/Kolkata')
    return utc_now.astimezone(ist)

def validate_period(interval, period):
    # Strict validation table
    valid_map = {
        '1m': ['1d', '5d'],
        '5m': ['1d', '1mo'],
        '15m': ['1mo'], '30m': ['1mo'], '1h': ['1mo'], '4h': ['1mo'],
        '1d': ['1mo', '1y', '2y', '5y'],
        '1wk': ['1mo', '1y', '5y', '10y', '15y', '20y'],
        '1mo': ['1y', '2y', '5y', '10y', '15y', '20y', '25y', '30y']
    }
    
    # Check if period is allowed for interval
    allowed = valid_map.get(interval, [])
    # For intervals with single allowed list that covers multiple options
    if interval in ['15m', '30m', '1h', '4h'] and period == '1mo': return True
    if period in allowed: return True
    
    # Fallback/Auto-correct
    return False

# --- MANUAL INDICATOR IMPLEMENTATIONS (No Talib/Pandas-TA) ---

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_sma(series, window):
    return series.rolling(window=window).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean() # Simple ATR for stability, can use ewm

def calculate_adx(df, period=14):
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0.0) # minus_dm diff is negative for lower lows
    
    tr = calculate_atr(df, 1) # True Range
    atr = tr.rolling(window=period).mean()
    
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).abs().rolling(window=period).mean() / atr)
    
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period).mean()
    return adx

def calculate_bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calculate_supertrend(df, period=10, multiplier=3):
    # Basic SuperTrend implementation
    hl2 = (df['High'] + df['Low']) / 2
    atr = calculate_atr(df, period)
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    # Initialize result arrays
    supertrend = [True] * len(df) # True = Green/Bullish
    final_upper = [0.0] * len(df)
    final_lower = [0.0] * len(df)
    
    # Need to iterate for SuperTrend logic
    for i in range(1, len(df)):
        curr_close = df['Close'].iloc[i]
        prev_close = df['Close'].iloc[i-1]
        
        # Upper Band Logic
        if upperband.iloc[i] < final_upper[i-1] or prev_close > final_upper[i-1]:
            final_upper[i] = upperband.iloc[i]
        else:
            final_upper[i] = final_upper[i-1]
            
        # Lower Band Logic
        if lowerband.iloc[i] > final_lower[i-1] or prev_close < final_lower[i-1]:
            final_lower[i] = lowerband.iloc[i]
        else:
            final_lower[i] = final_lower[i-1]
            
        # Trend Logic
        if supertrend[i-1] and curr_close < final_lower[i-1]:
            supertrend[i] = False
        elif not supertrend[i-1] and curr_close > final_upper[i-1]:
            supertrend[i] = True
        else:
            supertrend[i] = supertrend[i-1]
            
    return pd.Series(supertrend, index=df.index), pd.Series(final_upper, index=df.index), pd.Series(final_lower, index=df.index)

def calculate_vwap(df):
    if 'Volume' not in df.columns:
        return df['Close'] # Fallback for indices
    v = df['Volume'].values
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return df.assign(vwap=(tp * v).cumsum() / v.cumsum())['vwap']

def calculate_angle(series, period=1):
    # Angle calculation: degrees of slope
    # Use normalized change to approximate visual angle
    diff = series.diff(period)
    # Simple atan of change. Note: this is a mathematical proxy.
    # TradingView angles depend on chart scaling. 
    # We use a sensitivity factor to make 1 degree meaningful.
    angle = np.degrees(np.arctan(diff)) 
    return angle.fillna(0)

# ==========================================
# 3. DATA FETCHING
# ==========================================

def fetch_data(ticker, interval, period, delay=False):
    if delay:
        time.sleep(np.random.uniform(1.0, 1.5))
        
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
        
        # Handling MultiIndex Columns (yfinance update)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty:
            return None
            
        # Timezone Handling
        if df.index.tz is None:
            df.index = df.index.tz_localize(pytz.utc)
        df.index = df.index.tz_convert(pytz.timezone('Asia/Kolkata'))
        
        # Basic cleanup
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None

def process_indicators(df, config):
    # Calculate all necessary indicators here to ensure they exist for strategies
    df['EMA_Fast'] = calculate_ema(df['Close'], config.get('ema_fast', 9))
    df['EMA_Slow'] = calculate_ema(df['Close'], config.get('ema_slow', 15))
    df['EMA_Angle'] = calculate_angle(df['EMA_Fast'])
    
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['ATR'] = calculate_atr(df, 14)
    df['ADX'] = calculate_adx(df, config.get('adx_period', 14))
    
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    df['VWAP'] = calculate_vwap(df)
    
    # SuperTrend for custom builder
    st_trend, st_up, st_low = calculate_supertrend(df)
    df['SuperTrend'] = st_trend # Boolean
    
    # Common moving averages
    df['EMA_20'] = calculate_ema(df['Close'], 20)
    df['EMA_50'] = calculate_ema(df['Close'], 50)
    
    # Candle Size
    df['Candle_Size'] = (df['Close'] - df['Open']).abs()
    
    return df

# ==========================================
# 4. STRATEGY ENGINE
# ==========================================

def check_signal(df, i, strategy_type, config):
    """
    Returns: signal (1=Buy, -1=Sell, 0=None), reason (str)
    i is the index of the current candle (usually -1 for live)
    """
    signal = 0
    reason = ""
    
    # --- A. EMA Crossover Strategy ---
    if strategy_type == "EMA Crossover":
        ema_fast = df['EMA_Fast'].iloc[i]
        ema_slow = df['EMA_Slow'].iloc[i]
        prev_fast = df['EMA_Fast'].iloc[i-1]
        prev_slow = df['EMA_Slow'].iloc[i-1]
        angle = df['EMA_Angle'].iloc[i]
        
        # Basic Crossover Logic
        bullish_cross = (prev_fast <= prev_slow) and (ema_fast > ema_slow)
        bearish_cross = (prev_fast >= prev_slow) and (ema_fast < ema_slow)
        
        # Angle Check
        angle_pass = abs(angle) >= config.get('ema_angle_min', 1.0)
        
        # ADX Filter
        adx_pass = True
        if config.get('use_adx_filter', False):
            adx_pass = df['ADX'].iloc[i] >= config.get('adx_threshold', 25)
            
        # Entry Filters
        entry_filter = config.get('entry_filter', 'Simple Crossover')
        candle_pass = True
        
        candle_size = df['Candle_Size'].iloc[i]
        if entry_filter == "Custom Candle (Points)":
            candle_pass = candle_size >= config.get('custom_candle_points', 10)
        elif entry_filter == "ATR-based Candle":
            min_size = df['ATR'].iloc[i] * config.get('atr_multiplier', 1.0)
            candle_pass = candle_size >= min_size
            
        if bullish_cross and angle_pass and adx_pass and candle_pass:
            signal = 1
            reason = "EMA Bullish Crossover"
        elif bearish_cross and angle_pass and adx_pass and candle_pass:
            signal = -1
            reason = "EMA Bearish Crossover"

    # --- B. Simple Buy ---
    elif strategy_type == "Simple Buy":
        signal = 1
        reason = "Simple Buy Trigger"

    # --- C. Simple Sell ---
    elif strategy_type == "Simple Sell":
        signal = -1
        reason = "Simple Sell Trigger"

    # --- D. Price Threshold ---
    elif strategy_type == "Price Crosses Threshold":
        mode = config.get('threshold_mode', 'LONG (Price >= Threshold)')
        threshold = config.get('price_threshold', 0.0)
        price = df['Close'].iloc[i]
        
        if mode == "LONG (Price >= Threshold)" and price >= threshold:
            signal = 1
            reason = f"Price {price} >= {threshold}"
        elif mode == "SHORT (Price >= Threshold)" and price >= threshold:
            signal = -1
            reason = f"Price {price} >= {threshold}"
        elif mode == "LONG (Price <= Threshold)" and price <= threshold:
            signal = 1
            reason = f"Price {price} <= {threshold}"
        elif mode == "SHORT (Price <= Threshold)" and price <= threshold:
            signal = -1
            reason = f"Price {price} <= {threshold}"

    # --- E. RSI-ADX-EMA ---
    elif strategy_type == "RSI-ADX-EMA":
        rsi = df['RSI'].iloc[i]
        adx = df['ADX'].iloc[i]
        e1 = df['EMA_Fast'].iloc[i]
        e2 = df['EMA_Slow'].iloc[i]
        
        if rsi < 20 and adx > 20 and e1 > e2:
            signal = 1
            reason = "RSI<20, ADX>20, EMA1>EMA2"
        elif rsi > 80 and adx < 20 and e1 < e2:
            signal = -1
            reason = "RSI>80, ADX<20, EMA1<EMA2"

    # --- F. Percentage Change ---
    elif strategy_type == "Percentage Change":
        first_open = df['Open'].iloc[0] # Very first candle in period
        current_close = df['Close'].iloc[i]
        pct_change = ((current_close - first_open) / first_open) * 100
        
        thresh = config.get('pct_threshold', 0.5)
        direction = config.get('pct_direction', 'BUY on Rise')
        
        if direction == 'BUY on Rise' and pct_change >= thresh:
            signal = 1
        elif direction == 'SELL on Rise' and pct_change >= thresh:
            signal = -1
        elif direction == 'BUY on Fall' and pct_change <= -thresh:
            signal = 1
        elif direction == 'SELL on Fall' and pct_change <= -thresh:
            signal = -1
        reason = f"Pct Change {pct_change:.2f}% trigger"

    # --- G. AI Price Action (Simulated) ---
    elif strategy_type == "AI Price Action Analysis":
        # Simplified Logic for Demo: Trend + RSI + MACD confirmation
        trend_up = df['EMA_20'].iloc[i] > df['EMA_50'].iloc[i]
        rsi = df['RSI'].iloc[i]
        macd_hist = df['MACD_Hist'].iloc[i]
        
        score = 0
        if trend_up: score += 1
        else: score -= 1
        
        if 40 < rsi < 60: score += 0 # Neutral
        elif rsi < 30: score += 2 # Oversold/Bounce
        elif rsi > 70: score -= 2 # Overbought/Rejection
        
        if macd_hist > 0: score += 1
        else: score -= 1
        
        if score >= 3:
            signal = 1
            reason = "AI Score High (Bullish)"
        elif score <= -3:
            signal = -1
            reason = "AI Score Low (Bearish)"

    # --- H. Custom Strategy Builder ---
    elif strategy_type == "Custom Strategy Builder":
        conditions = st.session_state['custom_conditions']
        if not conditions:
            return 0, ""
            
        buy_votes = 0
        sell_votes = 0
        total_conditions = 0
        
        for cond in conditions:
            if not cond['use']: continue
            total_conditions += 1
            
            # Get Left Value
            val_a = df[cond['indicator']].iloc[i]
            
            # Get Right Value
            if cond['compare_price']:
                val_b = df[cond['compare_indicator']].iloc[i]
            else:
                val_b = float(cond['value'])
                
            op = cond['operator']
            met = False
            
            if op == '>': met = val_a > val_b
            elif op == '<': met = val_a < val_b
            elif op == '>=': met = val_a >= val_b
            elif op == '<=': met = val_a <= val_b
            elif op == '==': met = val_a == val_b
            elif op == 'crosses_above':
                prev_a = df[cond['indicator']].iloc[i-1]
                prev_b = df[cond['compare_indicator']].iloc[i-1] if cond['compare_price'] else val_b
                met = (prev_a <= prev_b) and (val_a > val_b)
            elif op == 'crosses_below':
                prev_a = df[cond['indicator']].iloc[i-1]
                prev_b = df[cond['compare_indicator']].iloc[i-1] if cond['compare_price'] else val_b
                met = (prev_a >= prev_b) and (val_a < val_b)
                
            if met:
                if cond['action'] == 'BUY': buy_votes += 1
                else: sell_votes += 1
        
        # Logic: All active BUY conditions met = BUY, All active SELL = SELL
        # This is a simplification. Usually separate logic for Long/Short.
        # Here: If any BUY condition exists and ALL are met -> BUY.
        buy_conds = [c for c in conditions if c['use'] and c['action'] == 'BUY']
        sell_conds = [c for c in conditions if c['use'] and c['action'] == 'SELL']
        
        if buy_conds and buy_votes == len(buy_conds):
            signal = 1
            reason = "Custom BUY Conditions Met"
        elif sell_conds and sell_votes == len(sell_conds):
            signal = -1
            reason = "Custom SELL Conditions Met"

    return signal, reason

# ==========================================
# 5. UI COMPONENTS
# ==========================================

def sidebar_ui():
    st.sidebar.title("⚙️ Configuration")
    
    # Assets & Time
    ticker = st.sidebar.text_input("Asset Ticker (yfinance)", value="^NSEI")
    col1, col2 = st.sidebar.columns(2)
    interval = col1.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"], index=1)
    period = col2.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=1)
    
    # Validation
    if not validate_period(interval, period):
        st.sidebar.error(f"Invalid Interval/Period: {interval}/{period}")
    
    quantity = st.sidebar.number_input("Quantity", min_value=1, value=50)
    
    # Strategy Selection
    st.sidebar.markdown("---")
    strategy_mode = st.sidebar.selectbox("Trading Mode", ["Backtest", "Live Trading"])
    strategy_type = st.sidebar.selectbox("Strategy", [
        "EMA Crossover", "Simple Buy", "Simple Sell", 
        "Price Crosses Threshold", "RSI-ADX-EMA", "Percentage Change",
        "AI Price Action Analysis", "Custom Strategy Builder"
    ])
    
    config = {}
    config['ticker'] = ticker
    config['interval'] = interval
    config['period'] = period
    config['quantity'] = quantity
    config['mode'] = strategy_mode
    config['strategy_type'] = strategy_type
    
    # Dynamic Strategy Inputs
    if strategy_type == "EMA Crossover":
        config['ema_fast'] = st.sidebar.number_input("EMA Fast", value=9)
        config['ema_slow'] = st.sidebar.number_input("EMA Slow", value=15)
        config['ema_angle_min'] = st.sidebar.number_input("Min Cross Angle (°)", value=1.0)
        config['use_adx_filter'] = st.sidebar.checkbox("Use ADX Filter")
        if config['use_adx_filter']:
            config['adx_threshold'] = st.sidebar.number_input("ADX Threshold", value=25)
            config['adx_period'] = st.sidebar.number_input("ADX Period", value=14)
        config['entry_filter'] = st.sidebar.selectbox("Entry Filter", ["Simple Crossover", "Custom Candle (Points)", "ATR-based Candle"])
        if config['entry_filter'] == "Custom Candle (Points)":
            config['custom_candle_points'] = st.sidebar.number_input("Min Candle Points", value=10.0)
        elif config['entry_filter'] == "ATR-based Candle":
            config['atr_multiplier'] = st.sidebar.number_input("ATR Multiplier", value=1.0)

    elif strategy_type == "Price Crosses Threshold":
        config['threshold_mode'] = st.sidebar.selectbox("Condition", ["LONG (Price >= Threshold)", "SHORT (Price >= Threshold)", "LONG (Price <= Threshold)", "SHORT (Price <= Threshold)"])
        config['price_threshold'] = st.sidebar.number_input("Price Threshold", value=20000.0)

    elif strategy_type == "Percentage Change":
        config['pct_threshold'] = st.sidebar.number_input("Threshold %", value=0.5, step=0.1)
        config['pct_direction'] = st.sidebar.selectbox("Action", ["BUY on Rise", "SELL on Rise", "BUY on Fall", "SELL on Fall"])

    elif strategy_type == "Custom Strategy Builder":
        st.sidebar.markdown("### Custom Conditions")
        with st.sidebar.expander("Add Condition"):
            c_use = st.checkbox("Enable", key="nc_use")
            c_comp = st.checkbox("Compare w/ Indicator", key="nc_comp")
            c_ind = st.selectbox("Indicator A", ["Close", "RSI", "EMA_Fast", "EMA_Slow", "SuperTrend", "ADX", "Volume"], key="nc_ind")
            c_op = st.selectbox("Operator", [">", "<", ">=", "<=", "==", "crosses_above", "crosses_below"], key="nc_op")
            c_val = st.number_input("Value", disabled=c_comp, key="nc_val")
            c_ind_b = st.selectbox("Indicator B", ["EMA_Fast", "EMA_Slow", "VWAP", "BB_Upper", "BB_Lower", "EMA_20", "EMA_50"], disabled=not c_comp, key="nc_indb")
            c_act = st.selectbox("Action", ["BUY", "SELL"], key="nc_act")
            
            if st.button("Add Condition"):
                cond = {
                    "use": c_use, "compare_price": c_comp, "indicator": c_ind,
                    "operator": c_op, "value": c_val, "compare_indicator": c_ind_b,
                    "action": c_act
                }
                st.session_state['custom_conditions'].append(cond)
        
        # Display existing conditions
        if st.session_state['custom_conditions']:
            st.sidebar.write(f"Active Conditions: {len(st.session_state['custom_conditions'])}")
            if st.sidebar.button("Clear All Conditions"):
                st.session_state['custom_conditions'] = []

    # Risk Management
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛡️ Risk Management")
    sl_type = st.sidebar.selectbox("Stop Loss Type", [
        "Custom Points", "Trailing SL (Points)", "Trailing SL + Current Candle",
        "Trailing SL + Previous Candle", "Trailing SL + Current Swing",
        "Trailing SL + Previous Swing", "Signal-based (Reverse Crossover)",
        "ATR-based", "Current Candle Low/High", "Previous Candle Low/High"
    ])
    sl_points = st.sidebar.number_input("SL Points (Fixed/Trail)", value=10.0)
    sl_trail_threshold = st.sidebar.number_input("Trailing Threshold (Points)", value=0.0)
    
    target_type = st.sidebar.selectbox("Target Type", [
        "Custom Points", "Trailing Target (Display Only)", "50% Exit at Target",
        "ATR-based", "Signal-based (Reverse Crossover)"
    ])
    target_points = st.sidebar.number_input("Target Points", value=20.0)
    
    config['sl_type'] = sl_type
    config['sl_points'] = sl_points
    config['sl_trail_threshold'] = sl_trail_threshold
    config['target_type'] = target_type
    config['target_points'] = target_points

    # Dhan Placeholders
    st.sidebar.markdown("---")
    with st.sidebar.expander("Dhan Brokerage (Placeholder)"):
        st.text_input("Client ID")
        st.text_input("Access Token", type="password")
        st.checkbox("Enable Live Orders (Mock)")

    return config

# ==========================================
# 6. TRADING LOGIC (CALCULATIONS)
# ==========================================

def calculate_initial_sl(entry_price, signal, df, i, config):
    sl_type = config['sl_type']
    sl = 0.0
    
    points = config['sl_points']
    
    if signal == 1: # LONG
        if sl_type == "Custom Points" or sl_type == "Trailing SL (Points)":
            sl = entry_price - points
        elif sl_type == "ATR-based":
            sl = entry_price - (df['ATR'].iloc[i] * 1.5)
        elif "Current Candle" in sl_type:
            sl = df['Low'].iloc[i]
        elif "Previous Candle" in sl_type:
            sl = df['Low'].iloc[i-1]
        elif sl_type == "Signal-based (Reverse Crossover)":
            sl = 0 # Dynamic check
        else:
            sl = entry_price - points # Default fallback
            
    elif signal == -1: # SHORT
        if sl_type == "Custom Points" or sl_type == "Trailing SL (Points)":
            sl = entry_price + points
        elif sl_type == "ATR-based":
            sl = entry_price + (df['ATR'].iloc[i] * 1.5)
        elif "Current Candle" in sl_type:
            sl = df['High'].iloc[i]
        elif "Previous Candle" in sl_type:
            sl = df['High'].iloc[i-1]
        elif sl_type == "Signal-based (Reverse Crossover)":
            sl = 0
        else:
            sl = entry_price + points

    return sl

def calculate_initial_target(entry_price, signal, df, i, config):
    tg_type = config['target_type']
    tg = 0.0
    points = config['target_points']
    
    if signal == 1: # LONG
        if "Custom Points" in tg_type or "Trailing Target" in tg_type:
            tg = entry_price + points
        elif tg_type == "ATR-based":
            tg = entry_price + (df['ATR'].iloc[i] * 3.0)
        elif tg_type == "Signal-based (Reverse Crossover)":
            tg = 0
    elif signal == -1: # SHORT
        if "Custom Points" in tg_type or "Trailing Target" in tg_type:
            tg = entry_price - points
        elif tg_type == "ATR-based":
            tg = entry_price - (df['ATR'].iloc[i] * 3.0)
        elif tg_type == "Signal-based (Reverse Crossover)":
            tg = 0
            
    return tg

def update_trailing_sl(current_price, position, config):
    # Logic for Trailing SL (Points)
    if "Trailing SL (Points)" not in config['sl_type']:
        return position['sl']
        
    old_sl = position['sl']
    points = config['sl_points']
    threshold = config['sl_trail_threshold']
    
    if position['type'] == 1: # LONG
        # New potential SL based on current price
        new_sl = current_price - points
        # Only move up
        if new_sl > old_sl:
            # Check threshold
            if (new_sl - old_sl) >= threshold:
                return new_sl
    else: # SHORT
        new_sl = current_price + points
        # Only move down
        if new_sl < old_sl:
            if (old_sl - new_sl) >= threshold:
                return new_sl
                
    return old_sl

def check_exit(df, i, position, config):
    """
    Returns: exit_signal (bool), exit_price, reason
    """
    current_price = df['Close'].iloc[i]
    pos_type = position['type']
    sl = position['sl']
    target = position['target']
    
    # 1. SL Hit
    if sl != 0:
        if pos_type == 1 and current_price <= sl:
            return True, sl, "Stop Loss Hit"
        if pos_type == -1 and current_price >= sl:
            return True, sl, "Stop Loss Hit"
            
    # 2. Target Hit (Standard)
    if target != 0 and "Trailing Target" not in config['target_type']:
        if pos_type == 1 and current_price >= target:
             return True, target, "Target Hit"
        if pos_type == -1 and current_price <= target:
             return True, target, "Target Hit"
             
    # 3. Signal Based Exit (Reverse Cross)
    if config['sl_type'] == "Signal-based (Reverse Crossover)" or config['target_type'] == "Signal-based (Reverse Crossover)":
        e1 = df['EMA_Fast'].iloc[i]
        e2 = df['EMA_Slow'].iloc[i]
        prev_e1 = df['EMA_Fast'].iloc[i-1]
        prev_e2 = df['EMA_Slow'].iloc[i-1]
        
        if pos_type == 1: # Long Exit on Bearish Cross
             if (prev_e1 >= prev_e2) and (e1 < e2):
                 return True, current_price, "Reverse Signal Exit"
        elif pos_type == -1: # Short Exit on Bullish Cross
             if (prev_e1 <= prev_e2) and (e1 > e2):
                 return True, current_price, "Reverse Signal Exit"
                 
    return False, 0.0, ""

# ==========================================
# 7. BACKTEST ENGINE
# ==========================================

def run_backtest(config):
    st.info(f"Running Backtest for {config['ticker']}...")
    df = fetch_data(config['ticker'], config['interval'], config['period'])
    if df is None:
        st.error("No Data Found.")
        return
        
    df = process_indicators(df, config)
    trades = []
    position = None # {type, entry_price, sl, target}
    
    # Metrics
    win = 0
    loss = 0
    total_pnl = 0.0
    
    # Loop
    for i in range(50, len(df)): # Start after indicators stabilize
        # Check Exit if in position
        if position:
            # Trailing SL Update (Simple approximation for backtest)
            position['sl'] = update_trailing_sl(df['Close'].iloc[i], position, config)
            
            exit_flag, exit_price, exit_reason = check_exit(df, i, position, config)
            
            if exit_flag:
                pnl = (exit_price - position['entry_price']) * position['type'] * config['quantity']
                trades.append({
                    'Entry Time': position['entry_time'],
                    'Exit Time': df.index[i],
                    'Type': 'LONG' if position['type'] == 1 else 'SHORT',
                    'Entry Price': position['entry_price'],
                    'Exit Price': exit_price,
                    'PnL': pnl,
                    'Reason': exit_reason
                })
                total_pnl += pnl
                if pnl > 0: win += 1
                else: loss += 1
                position = None
                continue # Skip entry check on same candle
        
        # Check Entry
        if position is None:
            sig, reason = check_signal(df, i, config['strategy_type'], config)
            if sig != 0:
                price = df['Close'].iloc[i]
                sl = calculate_initial_sl(price, sig, df, i, config)
                tg = calculate_initial_target(price, sig, df, i, config)
                
                position = {
                    'type': sig,
                    'entry_price': price,
                    'sl': sl,
                    'target': tg,
                    'entry_time': df.index[i]
                }
                
    # Results
    st.markdown("### 📊 Backtest Results")
    col1, col2, col3, col4 = st.columns(4)
    total_trades = len(trades)
    accuracy = (win/total_trades*100) if total_trades > 0 else 0
    
    col1.metric("Total Trades", total_trades)
    col2.metric("Accuracy", f"{accuracy:.2f}%")
    
    delta_color = "normal" if total_pnl >= 0 else "inverse"
    pnl_str = f"{total_pnl:.2f}"
    col3.metric("Total P&L", pnl_str, delta=pnl_str, delta_color=delta_color)
    
    if trades:
        trades_df = pd.DataFrame(trades)
        st.dataframe(trades_df.style.map(lambda x: 'color: green' if x > 0 else 'color: red', subset=['PnL']))
    else:
        st.warning("No trades generated.")

# ==========================================
# 8. LIVE TRADING LOOP
# ==========================================

def live_trading_dashboard(config):
    # Layout Tabs
    tab1, tab2, tab3 = st.tabs(["🔴 Live Dashboard", "📜 Trade History", "📝 Logs"])
    
    # --- TAB 1: DASHBOARD ---
    with tab1:
        # Top Controls
        c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
        start = c1.button("▶ Start Trading", type="primary")
        stop = c2.button("⏹ Stop Trading", type="secondary")
        
        if start:
            st.session_state['trading_active'] = True
            add_log("Trading Started")
        if stop:
            st.session_state['trading_active'] = False
            if st.session_state['position']:
                # Close position logic
                pos = st.session_state['position']
                curr_price = st.session_state.get('last_price', pos['entry_price'])
                pnl = (curr_price - pos['entry_price']) * pos['type'] * config['quantity']
                
                # Record Trade
                st.session_state['trade_history'].append({
                    'Entry Time': pos['entry_time'],
                    'Exit Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Signal': 'LONG' if pos['type'] == 1 else 'SHORT',
                    'Entry Price': pos['entry_price'],
                    'Exit Price': curr_price,
                    'PnL': pnl,
                    'Reason': "Manual Stop"
                })
                add_log(f"Position Closed Manually. PnL: {pnl:.2f}")
                reset_position_state()
            add_log("Trading Stopped")
        
        status_color = "green" if st.session_state['trading_active'] else "grey"
        status_text = "ACTIVE" if st.session_state['trading_active'] else "STOPPED"
        c3.markdown(f"**Status:** :{status_color}[{status_text}]")
        
        if c4.button("🔄 Refresh UI"):
            pass
            
        # Placeholders for dynamic updates
        metrics_ph = st.empty()
        chart_ph = st.empty()
        
        # --- THE LOOP ---
        if st.session_state['trading_active']:
            
            # 1. Fetch Data
            df = fetch_data(config['ticker'], config['interval'], config['period'], delay=True)
            
            if df is not None:
                # 2. Calculate Indicators
                df = process_indicators(df, config)
                st.session_state['current_data'] = df
                
                curr_i = -1
                curr_price = df['Close'].iloc[curr_i]
                st.session_state['last_price'] = curr_price
                
                # 3. Position Management
                pos = st.session_state['position']
                
                if pos:
                    # Trailing Logic
                    pos['sl'] = update_trailing_sl(curr_price, pos, config)
                    
                    # Trailing Target (Display Only)
                    if "Trailing Target" in config['target_type']:
                        # Update highest/lowest for tracking
                        if pos['type'] == 1:
                            if st.session_state['highest_price'] is None or curr_price > st.session_state['highest_price']:
                                st.session_state['highest_price'] = curr_price
                            # Update target display based on diff
                            profit_dist = st.session_state['highest_price'] - pos['entry_price']
                            st.session_state['trailing_profit_points'] = profit_dist
                        else:
                            if st.session_state['lowest_price'] is None or curr_price < st.session_state['lowest_price']:
                                st.session_state['lowest_price'] = curr_price
                            profit_dist = pos['entry_price'] - st.session_state['lowest_price']
                            st.session_state['trailing_profit_points'] = profit_dist
                    
                    # Check Exit
                    exit_bool, exit_p, exit_r = check_exit(df, curr_i, pos, config)
                    
                    # Partial Exit Check (50%)
                    if config['target_type'] == "50% Exit at Target" and not st.session_state['partial_exit_done']:
                        tg = pos['target']
                        if (pos['type'] == 1 and curr_price >= tg) or (pos['type'] == -1 and curr_price <= tg):
                            st.session_state['partial_exit_done'] = True
                            pnl_part = (tg - pos['entry_price']) * pos['type'] * (config['quantity'] / 2)
                            add_log(f"Partial Exit triggered at {tg}. PnL: {pnl_part}")
                            # Remaining position trails
                            
                    if exit_bool:
                        # Finalize Trade
                        qty = config['quantity']
                        if st.session_state['partial_exit_done']: qty = qty / 2
                        
                        pnl = (exit_p - pos['entry_price']) * pos['type'] * qty
                        
                        st.session_state['trade_history'].append({
                            'Entry Time': pos['entry_time'],
                            'Exit Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Signal': 'LONG' if pos['type'] == 1 else 'SHORT',
                            'Entry Price': pos['entry_price'],
                            'Exit Price': exit_p,
                            'PnL': pnl,
                            'Reason': exit_r
                        })
                        add_log(f"Trade Exited: {exit_r} | PnL: {pnl:.2f}")
                        reset_position_state()
                        pos = None
                        
                # 4. Signal Generation (if no pos)
                if not pos:
                    sig, reason = check_signal(df, curr_i, config['strategy_type'], config)
                    if sig != 0:
                        sl = calculate_initial_sl(curr_price, sig, df, curr_i, config)
                        tg = calculate_initial_target(curr_price, sig, df, curr_i, config)
                        
                        st.session_state['position'] = {
                            'type': sig,
                            'entry_price': curr_price,
                            'sl': sl,
                            'target': tg,
                            'entry_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'quantity': config['quantity']
                        }
                        # Init Trailing State
                        st.session_state['highest_price'] = curr_price
                        st.session_state['lowest_price'] = curr_price
                        
                        add_log(f"Trade Entered: {'LONG' if sig==1 else 'SHORT'} | {reason}")
                        pos = st.session_state['position']

                # 5. UI Updates (Inside Loop)
                with metrics_ph.container():
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Current Price", f"{curr_price:.2f}")
                    m2.metric("EMA Angle", f"{df['EMA_Angle'].iloc[curr_i]:.2f}°")
                    m3.metric("RSI", f"{df['RSI'].iloc[curr_i]:.1f}")
                    
                    if pos:
                        unrealized = (curr_price - pos['entry_price']) * pos['type'] * config['quantity']
                        delta_c = "normal" if unrealized >= 0 else "inverse"
                        m4.metric("Position", "LONG" if pos['type']==1 else "SHORT")
                        m5.metric("Unrealized P&L", f"{unrealized:.2f}", delta=f"{unrealized:.2f}", delta_color=delta_c)
                        
                        st.markdown(f"**Entry:** {pos['entry_price']:.2f} | **SL:** {pos['sl']:.2f} | **Target:** {pos['target'] if pos['target']!=0 else 'Signal'}")
                        if "Trailing Target" in config['target_type']:
                            st.info(f"Trailing Profit Points: {st.session_state['trailing_profit_points']:.2f}")
                    else:
                        m4.metric("Position", "NONE")
                        m5.metric("Last Signal", reason if 'reason' in locals() else "-")

                # Chart Update
                with chart_ph.container():
                    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price")])
                    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], line=dict(color='blue', width=1), name="EMA Fast"))
                    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], line=dict(color='orange', width=1), name="EMA Slow"))
                    
                    if pos:
                        fig.add_hline(y=pos['entry_price'], line_dash="dash", line_color="yellow", annotation_text="Entry")
                        if pos['sl'] != 0: fig.add_hline(y=pos['sl'], line_dash="dot", line_color="red", annotation_text="SL")
                        if pos['target'] != 0: fig.add_hline(y=pos['target'], line_dash="dot", line_color="green", annotation_text="Target")
                    
                    fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True, key=f"live_chart_{int(time.time())}")
            
            time.sleep(1)
            st.rerun() # Force Streamlit to re-execute loop

    # --- TAB 2: HISTORY ---
    with tab2:
        st.markdown("### 📈 Trade History")
        if st.session_state['trade_history']:
            hist_df = pd.DataFrame(st.session_state['trade_history'])
            total_pnl = hist_df['PnL'].sum()
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Total Trades", len(hist_df))
            k2.metric("Win Rate", f"{(len(hist_df[hist_df['PnL']>0])/len(hist_df)*100):.1f}%")
            k3.metric("Net P&L", f"{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color="normal" if total_pnl>=0 else "inverse")
            
            st.dataframe(hist_df.style.map(lambda x: 'color: green' if x > 0 else 'color: red', subset=['PnL']))
        else:
            st.info("No trades recorded yet.")

    # --- TAB 3: LOGS ---
    with tab3:
        st.markdown("### 📝 System Logs")
        for log in reversed(st.session_state['trade_logs']):
            st.text(log)

# ==========================================
# 9. MAIN APP ENTRY
# ==========================================

def main():
    config = sidebar_ui()
    
    st.title("⚡ Pro Quantitative Trading System")
    
    if config['mode'] == "Backtest":
        st.markdown(f"**Backtest Mode** | Asset: {config['ticker']} | Strategy: {config['strategy_type']}")
        if st.button("Run Backtest"):
            run_backtest(config)
    else:
        live_trading_dashboard(config)

if __name__ == "__main__":
    main()
