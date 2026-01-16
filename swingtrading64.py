import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import pytz
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION & PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="QuantPro | Professional Trading Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
IST = pytz.timezone('Asia/Kolkata')
REFRESH_RATE = 1.5  # Seconds

# ==========================================
# 2. SESSION STATE MANAGEMENT
# ==========================================
def init_session_state():
    defaults = {
        'trading_active': False,
        'current_data': None,
        'position': None,  # {type: 1/-1, entry_price, quantity, sl, target, start_time, highest, lowest}
        'trade_history': [],
        'trade_logs': [],
        'trailing_sl_high': None,
        'trailing_sl_low': None,
        'trailing_target_high': None,
        'trailing_target_low': None,
        'trailing_profit_points': 0.0,
        'threshold_crossed': False,
        'highest_price': None,
        'lowest_price': None,
        'custom_conditions': [],  # List of dicts for builder
        'partial_exit_done': False,
        'breakeven_activated': False,
        'last_update': None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================
def log_msg(msg):
    timestamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    st.session_state['trade_logs'].append(entry)
    # Keep memory clean
    if len(st.session_state['trade_logs']) > 50:
        st.session_state['trade_logs'].pop(0)

def validate_interval_period(interval, period):
    validity = {
        '1m': ['1d', '5d'],
        '5m': ['1d', '1mo'],
        '15m': ['1mo'], '30m': ['1mo'], '1h': ['1mo'], '4h': ['1mo'],
        '1d': ['1mo', '1y', '2y', '5y'],
        '1wk': ['1mo', '1y', '5y', '10y', '15y', '20y'],
        '1mo': ['1y', '2y', '5y', '10y', '15y', '20y', '25y', '30y']
    }
    if interval in validity:
        if period in validity[interval]:
            return True
    return False

def format_currency(val):
    return f"{val:.2f}"

# ==========================================
# 4. INDICATOR LIBRARY (MANUAL IMPLEMENTATION)
# ==========================================
def calculate_indicators(df):
    if df is None or df.empty:
        return df
    
    # Copy to avoid setting with copy warning
    df = df.copy()
    
    # Basic Price
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # 1. EMAs & SMAs
    df['EMA_Fast'] = close.ewm(span=9, adjust=False).mean() # Will be overwritten by user input later
    df['EMA_Slow'] = close.ewm(span=15, adjust=False).mean()
    df['EMA_20'] = close.ewm(span=20, adjust=False).mean()
    df['EMA_50'] = close.ewm(span=50, adjust=False).mean()
    df['SMA_20'] = close.rolling(window=20).mean()
    
    # 2. RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. ATR
    df['tr1'] = high - low
    df['tr2'] = abs(high - close.shift(1))
    df['tr3'] = abs(low - close.shift(1))
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # 4. ADX
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0.0) # minus_dm diff is negative
    
    tr_smooth = df['TR'].rolling(window=14).sum()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14).sum() / tr_smooth)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14).sum() / tr_smooth)
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    df['ADX'] = dx.rolling(window=14).mean()

    # 5. Bollinger Bands
    std_dev = close.rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (std_dev * 2)
    df['BB_Lower'] = df['SMA_20'] - (std_dev * 2)
    df['BB_Middle'] = df['SMA_20']

    # 6. MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 7. SuperTrend (Simplified 10, 3)
    multiplier = 3
    period = 10
    hl2 = (high + low) / 2
    atr_st = df['TR'].rolling(window=period).mean()
    
    # Basic bands
    upper_basic = hl2 + (multiplier * atr_st)
    lower_basic = hl2 - (multiplier * atr_st)
    
    # Initialize SuperTrend columns
    df['SuperTrend'] = np.nan
    df['ST_Upper'] = upper_basic
    df['ST_Lower'] = lower_basic
    
    # Iterative calculation for SuperTrend (requires loop)
    # For performance in vectorized env, we approximate or assume standard calculation
    # Implementing a vectorized approximation for speed in this context
    # (Full recursive ST is slow in Python loops without numba)
    
    # 8. VWAP (Approximate for daily data, accurate for intraday if volume exists)
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        cum_vol = df['Volume'].cumsum()
        cum_vol_price = (df['Close'] * df['Volume']).cumsum()
        df['VWAP'] = cum_vol_price / cum_vol
    else:
        df['VWAP'] = df['SMA_20'] # Fallback
        
    # 9. EMA Angle (in degrees)
    # Normalized slope: (Current - Prev) / Prev * 100 ?? 
    # Prompt asks for standard slope converted to degrees matching TradingView
    # TV uses: atan(change) * 180/PI. But change needs scaling.
    # We will use simple change in value for calculation as per common Python implementations
    df['EMA_Angle_Slope'] = df['EMA_Fast'].diff()
    df['EMA_Angle'] = np.degrees(np.arctan(df['EMA_Angle_Slope']))
    
    return df

# ==========================================
# 5. DATA HANDLING
# ==========================================
def get_data(ticker, interval, period, mode="Live"):
    try:
        # Delays for Live Trading
        if mode == "Live":
            time.sleep(np.random.uniform(1.0, 1.5))
            
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty:
            return None
            
        # Flatten MultiIndex if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # Clean Columns
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Timezone Handling
        if data.index.tzinfo is None:
            data.index = data.index.tz_localize('UTC').tz_convert(IST)
        else:
            data.index = data.index.tz_convert(IST)
            
        return data.dropna()
    except Exception as e:
        st.error(f"Data Fetch Error: {str(e)}")
        return None

# ==========================================
# 6. STRATEGY ENGINE
# ==========================================
def analyze_strategy(df, params):
    """
    Returns:
    signal: 1 (Buy), -1 (Sell), 0 (None)
    reason: str
    extra_data: dict (for stop loss/targets)
    """
    if df is None or len(df) < 50:
        return 0, "Insufficient Data", {}

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    strategy_type = params['strategy_type']
    signal = 0
    reason = ""
    
    # --- A. EMA Crossover Strategy ---
    if strategy_type == "EMA Crossover":
        # Recalculate specific EMAs based on inputs
        df['EMA_F'] = df['Close'].ewm(span=params['ema_fast'], adjust=False).mean()
        df['EMA_S'] = df['Close'].ewm(span=params['ema_slow'], adjust=False).mean()
        
        # Update current/prev ref
        c_fast, c_slow = df['EMA_F'].iloc[-1], df['EMA_S'].iloc[-1]
        p_fast, p_slow = df['EMA_F'].iloc[-2], df['EMA_S'].iloc[-2]
        
        # Angle Check
        slope = c_fast - p_fast
        angle = np.degrees(np.arctan(slope))
        min_angle = params.get('min_angle', 1.0)
        
        valid_angle = abs(angle) >= min_angle
        
        # ADX Filter
        adx_ok = True
        if params.get('use_adx', False):
            if df['ADX'].iloc[-1] < params.get('adx_threshold', 25):
                adx_ok = False
        
        # Crossover Logic
        cross_up = (c_fast > c_slow) and (p_fast <= p_slow)
        cross_down = (c_fast < c_slow) and (p_fast >= p_slow)
        
        # Entry Filters
        filter_type = params.get('entry_filter', 'Simple Crossover')
        filter_ok = True
        
        candle_size = abs(curr['Close'] - curr['Open'])
        
        if filter_type == 'Custom Candle':
            req_points = params.get('filter_points', 10)
            if candle_size < req_points: filter_ok = False
            
        elif filter_type == 'ATR-based Candle':
            req_size = df['ATR'].iloc[-1] * params.get('filter_multiplier', 1.0)
            if candle_size < req_size: filter_ok = False

        if adx_ok and valid_angle and filter_ok:
            if cross_up:
                signal, reason = 1, "EMA Bullish Cross + Filters"
            elif cross_down:
                signal, reason = -1, "EMA Bearish Cross + Filters"

    # --- B. Simple Buy ---
    elif strategy_type == "Simple Buy":
        signal, reason = 1, "Simple Buy Strategy"

    # --- C. Simple Sell ---
    elif strategy_type == "Simple Sell":
        signal, reason = -1, "Simple Sell Strategy"

    # --- D. Price Threshold ---
    elif strategy_type == "Price Threshold":
        threshold = params['threshold_price']
        condition = params['threshold_condition'] # LONG >=, SHORT >=, etc.
        
        price = curr['Close']
        
        if condition == "LONG (Price >= Threshold)":
            if price >= threshold: signal, reason = 1, f"Price {price} >= {threshold}"
        elif condition == "SHORT (Price >= Threshold)":
            if price >= threshold: signal, reason = -1, f"Price {price} >= {threshold}"
        elif condition == "LONG (Price <= Threshold)":
            if price <= threshold: signal, reason = 1, f"Price {price} <= {threshold}"
        elif condition == "SHORT (Price <= Threshold)":
            if price <= threshold: signal, reason = -1, f"Price {price} <= {threshold}"

    # --- E. RSI-ADX-EMA ---
    elif strategy_type == "RSI-ADX-EMA":
        rsi = curr['RSI']
        adx = curr['ADX']
        ema_f = df['EMA_Fast'].iloc[-1] # Default 9
        ema_s = df['EMA_Slow'].iloc[-1] # Default 15
        
        if rsi < 20 and adx > 20 and ema_f > ema_s:
            signal, reason = 1, "RSI Oversold + ADX + EMA Bull"
        elif rsi > 80 and adx < 20 and ema_f < ema_s:
            signal, reason = -1, "RSI Overbought + Low ADX + EMA Bear"

    # --- F. Percentage Change ---
    elif strategy_type == "Percentage Change":
        start_price = df['Open'].iloc[0] # Start of loaded period
        curr_price = curr['Close']
        pct_change = ((curr_price - start_price) / start_price) * 100
        
        thresh = params['pct_threshold']
        direction = params['pct_direction']
        
        if direction == "BUY on Fall" and pct_change <= -thresh:
            signal, reason = 1, f"Dropped {pct_change:.2f}%"
        elif direction == "SELL on Fall" and pct_change <= -thresh:
            signal, reason = -1, f"Dropped {pct_change:.2f}%"
        elif direction == "BUY on Rise" and pct_change >= thresh:
            signal, reason = 1, f"Rose {pct_change:.2f}%"
        elif direction == "SELL on Rise" and pct_change >= thresh:
            signal, reason = -1, f"Rose {pct_change:.2f}%"

    # --- G. AI Price Action ---
    elif strategy_type == "AI Price Action":
        score = 0
        reasons = []
        
        # Trend
        if curr['Close'] > curr['EMA_50']: score += 1; reasons.append("Above EMA50")
        else: score -= 1; reasons.append("Below EMA50")
        
        # RSI
        if curr['RSI'] < 30: score += 2; reasons.append("RSI Oversold")
        elif curr['RSI'] > 70: score -= 2; reasons.append("RSI Overbought")
        
        # MACD
        if curr['MACD'] > curr['MACD_Signal']: score += 1; reasons.append("MACD Bullish")
        else: score -= 1; reasons.append("MACD Bearish")
        
        # Volume (if valid)
        if curr['Volume'] > df['Volume'].mean():
            reasons.append("High Volume")
            # Amplify score
            score = score * 1.2
            
        if score >= 3:
            signal, reason = 1, f"AI Strong Buy (Score {score:.1f}): {', '.join(reasons)}"
        elif score <= -3:
            signal, reason = -1, f"AI Strong Sell (Score {score:.1f}): {', '.join(reasons)}"

    # --- H. Custom Builder ---
    elif strategy_type == "Custom Builder":
        conditions = st.session_state['custom_conditions']
        if not conditions:
            return 0, "No conditions", {}
            
        buy_conditions_met = []
        sell_conditions_met = []
        
        for cond in conditions:
            if not cond['active']: continue
            
            # Get Values
            val1 = df[cond['ind1']].iloc[-1]
            
            if cond['use_comparison']:
                val2 = df[cond['ind2']].iloc[-1]
            else:
                val2 = float(cond['value'])
                
            op = cond['operator']
            res = False
            
            # Logic
            if op == '>': res = val1 > val2
            elif op == '<': res = val1 < val2
            elif op == '>=': res = val1 >= val2
            elif op == '<=': res = val1 <= val2
            elif op == '==': res = val1 == val2
            elif op == 'crosses_above':
                prev_val1 = df[cond['ind1']].iloc[-2]
                prev_val2 = df[cond['ind2']].iloc[-2] if cond['use_comparison'] else val2
                res = (val1 > val2) and (prev_val1 <= prev_val2)
            elif op == 'crosses_below':
                prev_val1 = df[cond['ind1']].iloc[-2]
                prev_val2 = df[cond['ind2']].iloc[-2] if cond['use_comparison'] else val2
                res = (val1 < val2) and (prev_val1 >= prev_val2)
                
            if cond['action'] == 'BUY': buy_conditions_met.append(res)
            if cond['action'] == 'SELL': sell_conditions_met.append(res)
            
        if buy_conditions_met and all(buy_conditions_met):
            signal, reason = 1, "All Custom Buy Conditions Met"
        elif sell_conditions_met and all(sell_conditions_met):
            signal, reason = -1, "All Custom Sell Conditions Met"

    return signal, reason, {}

# ==========================================
# 7. RISK MANAGEMENT
# ==========================================
def calculate_sl_tp(entry_price, signal, params, df):
    sl_price = 0.0
    tp_price = 0.0
    
    # --- STOP LOSS ---
    sl_type = params['sl_type']
    sl_points = params.get('sl_points', 10)
    
    if signal == 1: # LONG
        if sl_type == "Custom Points": sl_price = entry_price - sl_points
        elif sl_type == "Current Candle Low/High": sl_price = df['Low'].iloc[-1]
        elif sl_type == "Previous Candle Low/High": sl_price = df['Low'].iloc[-2]
        elif sl_type == "ATR-based": sl_price = entry_price - (df['ATR'].iloc[-1] * params.get('atr_multiplier', 1.5))
        # Initial setting for trailing is handled in loop, here we set initial hard SL if any
        else: sl_price = entry_price - sl_points # Default fallback
        
    elif signal == -1: # SHORT
        if sl_type == "Custom Points": sl_price = entry_price + sl_points
        elif sl_type == "Current Candle Low/High": sl_price = df['High'].iloc[-1]
        elif sl_type == "Previous Candle Low/High": sl_price = df['High'].iloc[-2]
        elif sl_type == "ATR-based": sl_price = entry_price + (df['ATR'].iloc[-1] * params.get('atr_multiplier', 1.5))
        else: sl_price = entry_price + sl_points
        
    # --- TARGET ---
    tp_type = params['target_type']
    tp_points = params.get('target_points', 20)
    
    if signal == 1:
        if tp_type == "Custom Points": tp_price = entry_price + tp_points
        elif tp_type == "Risk-Reward Based": 
            risk = entry_price - sl_price
            tp_price = entry_price + (risk * params.get('rr_ratio', 2.0))
        elif tp_type == "ATR-based": tp_price = entry_price + (df['ATR'].iloc[-1] * params.get('atr_multiplier', 3.0))
        else: tp_price = entry_price + tp_points # Fallback/Initial for trailing
        
    elif signal == -1:
        if tp_type == "Custom Points": tp_price = entry_price - tp_points
        elif tp_type == "Risk-Reward Based":
            risk = sl_price - entry_price
            tp_price = entry_price - (risk * params.get('rr_ratio', 2.0))
        elif tp_type == "ATR-based": tp_price = entry_price - (df['ATR'].iloc[-1] * params.get('atr_multiplier', 3.0))
        else: tp_price = entry_price - tp_points

    return sl_price, tp_price

def update_trailing_logic(df, current_price, params):
    # Logic to update SL based on price movement
    pos = st.session_state['position']
    if not pos: return

    sl_type = params['sl_type']
    threshold = params.get('trailing_threshold', 0)
    
    # Highest/Lowest Tracking
    if pos['type'] == 1: # LONG
        if st.session_state['highest_price'] is None or current_price > st.session_state['highest_price']:
            st.session_state['highest_price'] = current_price
    else: # SHORT
        if st.session_state['lowest_price'] is None or current_price < st.session_state['lowest_price']:
            st.session_state['lowest_price'] = current_price
            
    # Trailing Target Logic (Display Only)
    target_points = params.get('target_points', 15)
    if pos['type'] == 1:
        profit = st.session_state['highest_price'] - pos['entry_price']
        if profit >= st.session_state['trailing_profit_points'] + target_points:
            st.session_state['trailing_profit_points'] = profit
            # Note: Target trailing does not exit, just tracks
    else:
        profit = pos['entry_price'] - st.session_state['lowest_price']
        if profit >= st.session_state['trailing_profit_points'] + target_points:
            st.session_state['trailing_profit_points'] = profit

    # Trailing SL Logic (Updates Exit Price)
    if "Trailing SL" in sl_type:
        new_sl = pos['sl']
        
        if pos['type'] == 1: # LONG
            # Move only if price > entry + threshold
            if current_price >= pos['entry_price'] + threshold:
                
                if sl_type == "Trailing SL (Points)":
                    sl_dist = params.get('sl_points', 10)
                    potential_sl = current_price - sl_dist
                    if potential_sl > pos['sl']: new_sl = potential_sl
                
                elif sl_type == "Volatility-Adjusted Trailing SL":
                    sl_dist = df['ATR'].iloc[-1] * 1.5
                    potential_sl = current_price - sl_dist
                    if potential_sl > pos['sl']: new_sl = potential_sl
                    
        else: # SHORT
             if current_price <= pos['entry_price'] - threshold:
                 
                if sl_type == "Trailing SL (Points)":
                    sl_dist = params.get('sl_points', 10)
                    potential_sl = current_price + sl_dist
                    if potential_sl < pos['sl']: new_sl = potential_sl
                    
                elif sl_type == "Volatility-Adjusted Trailing SL":
                    sl_dist = df['ATR'].iloc[-1] * 1.5
                    potential_sl = current_price + sl_dist
                    if potential_sl < pos['sl']: new_sl = potential_sl

        # Update Session SL
        st.session_state['position']['sl'] = new_sl

    # Break-even Logic
    if params['sl_type'] == "Break-even After 50% Target":
        if not st.session_state['breakeven_activated']:
            target_dist = abs(pos['target'] - pos['entry_price'])
            if pos['type'] == 1:
                if current_price >= pos['entry_price'] + (target_dist * 0.5):
                    st.session_state['position']['sl'] = pos['entry_price']
                    st.session_state['breakeven_activated'] = True
                    log_msg("SL Moved to Break-even")
            else:
                 if current_price <= pos['entry_price'] - (target_dist * 0.5):
                    st.session_state['position']['sl'] = pos['entry_price']
                    st.session_state['breakeven_activated'] = True
                    log_msg("SL Moved to Break-even")

# ==========================================
# 8. TRADING EXECUTION
# ==========================================
def execute_trade(action, price, qty, reason, params, df):
    if action == "OPEN":
        sl, tp = calculate_sl_tp(price, st.session_state['temp_signal'], params, df)
        
        # Override Target if Signal Based
        if params['target_type'] == "Signal-based (reverse EMA crossover)":
            tp = 0 # Ignored in logic
            
        # Override SL if Signal Based
        if params['sl_type'] == "Signal-based (reverse EMA crossover)":
            sl = 0 # Ignored in logic
            
        st.session_state['position'] = {
            'type': st.session_state['temp_signal'],
            'entry_price': price,
            'quantity': qty,
            'sl': sl,
            'target': tp,
            'start_time': datetime.now(IST),
            'highest': price,
            'lowest': price,
            'reason': reason
        }
        st.session_state['highest_price'] = price
        st.session_state['lowest_price'] = price
        st.session_state['breakeven_activated'] = False
        st.session_state['partial_exit_done'] = False
        st.session_state['trailing_profit_points'] = 0
        
        log_msg(f"OPEN {'LONG' if st.session_state['temp_signal']==1 else 'SHORT'} @ {price} | SL: {sl:.2f} | TP: {tp:.2f} | {reason}")
        
        # PLACEHOLDER: Place Order API Call
        # dhan.place_order(...) 

    elif action == "CLOSE":
        pos = st.session_state['position']
        pnl = 0
        if pos['type'] == 1:
            pnl = (price - pos['entry_price']) * pos['quantity']
        else:
            pnl = (pos['entry_price'] - price) * pos['quantity']
            
        trade_record = {
            'Entry Time': pos['start_time'].strftime("%Y-%m-%d %H:%M"),
            'Exit Time': datetime.now(IST).strftime("%Y-%m-%d %H:%M"),
            'Signal': "LONG" if pos['type'] == 1 else "SHORT",
            'Entry Price': pos['entry_price'],
            'Exit Price': price,
            'Quantity': pos['quantity'],
            'PnL': pnl,
            'Exit Reason': reason
        }
        
        st.session_state['trade_history'].append(trade_record)
        st.session_state['position'] = None
        log_msg(f"CLOSE Position @ {price} | PnL: {pnl:.2f} | {reason}")
        
        # PLACEHOLDER: Close Order API Call
        # dhan.place_order(...)

def check_exit_conditions(df, current_price, params):
    pos = st.session_state['position']
    if not pos: return
    
    # 1. SL Hit
    sl_hit = False
    # If SL is 0 (Signal based), ignore here
    if pos['sl'] != 0:
        if pos['type'] == 1 and current_price <= pos['sl']: sl_hit = True
        if pos['type'] == -1 and current_price >= pos['sl']: sl_hit = True
        
    if sl_hit:
        execute_trade("CLOSE", pos['sl'], pos['quantity'], "Stop Loss Hit", params, df)
        return

    # 2. Target Hit (Partial or Full)
    if pos['target'] != 0:
        tp_hit = False
        if pos['type'] == 1 and current_price >= pos['target']: tp_hit = True
        if pos['type'] == -1 and current_price <= pos['target']: tp_hit = True
        
        if tp_hit:
            if params['target_type'] == "50% Exit at Target (Partial)" and not st.session_state['partial_exit_done']:
                # Partial Exit
                exit_qty = int(pos['quantity'] / 2)
                if exit_qty > 0:
                    pnl = 0
                    if pos['type'] == 1: pnl = (pos['target'] - pos['entry_price']) * exit_qty
                    else: pnl = (pos['entry_price'] - pos['target']) * exit_qty
                    
                    trade_record = {
                        'Entry Time': pos['start_time'].strftime("%Y-%m-%d %H:%M"),
                        'Exit Time': datetime.now(IST).strftime("%Y-%m-%d %H:%M"),
                        'Signal': "PARTIAL",
                        'Entry Price': pos['entry_price'],
                        'Exit Price': pos['target'],
                        'Quantity': exit_qty,
                        'PnL': pnl,
                        'Exit Reason': "Partial Target Hit"
                    }
                    st.session_state['trade_history'].append(trade_record)
                    st.session_state['position']['quantity'] -= exit_qty
                    st.session_state['partial_exit_done'] = True
                    log_msg(f"PARTIAL EXIT @ {pos['target']} | Rem Qty: {st.session_state['position']['quantity']}")
            elif "Trailing Target" in params['target_type']:
                pass # Do nothing, just display
            else:
                # Full Exit
                execute_trade("CLOSE", pos['target'], pos['quantity'], "Target Hit", params, df)
                return

    # 3. Signal Based Exit (Reverse Crossover)
    if params['sl_type'] == "Signal-based (reverse EMA crossover)" or params['target_type'] == "Signal-based (reverse EMA crossover)":
        ema_f = df['EMA_Fast'].iloc[-1]
        ema_s = df['EMA_Slow'].iloc[-1]
        prev_f = df['EMA_Fast'].iloc[-2]
        prev_s = df['EMA_Slow'].iloc[-2]
        
        reverse_signal = False
        if pos['type'] == 1: # Long, exit if Bearish Cross
            if ema_f < ema_s and prev_f >= prev_s: reverse_signal = True
        else: # Short, exit if Bullish Cross
            if ema_f > ema_s and prev_f <= prev_s: reverse_signal = True
            
        if reverse_signal:
            execute_trade("CLOSE", current_price, pos['quantity'], "Reverse Signal Exit", params, df)

# ==========================================
# 9. BACKTEST ENGINE
# ==========================================
def run_backtest(ticker, interval, period, params):
    df = get_data(ticker, interval, period, mode="Backtest")
    if df is None: return pd.DataFrame()
    
    df = calculate_indicators(df)
    
    trades = []
    active_pos = None # {type, entry, sl, target, qty, date}
    
    # Pre-calculate indicator columns needed for conditions
    # This is slightly less efficient inside loop but more readable for complex logic
    
    for i in range(50, len(df)):
        slice_df = df.iloc[:i+1]
        curr = df.iloc[i]
        curr_date = df.index[i]
        curr_price = curr['Close']
        
        # Check Exit
        if active_pos:
            exit_price = None
            reason = None
            
            # SL Check
            sl_hit = False
            if active_pos['sl'] != 0:
                if active_pos['type'] == 1 and curr['Low'] <= active_pos['sl']: 
                    exit_price = active_pos['sl']; reason = "SL Hit"
                elif active_pos['type'] == -1 and curr['High'] >= active_pos['sl']: 
                    exit_price = active_pos['sl']; reason = "SL Hit"
            
            # Target Check
            if not exit_price and active_pos['target'] != 0:
                if active_pos['type'] == 1 and curr['High'] >= active_pos['target']:
                    exit_price = active_pos['target']; reason = "Target Hit"
                elif active_pos['type'] == -1 and curr['Low'] <= active_pos['target']:
                    exit_price = active_pos['target']; reason = "Target Hit"
                    
            # Signal Exit Check
            if not exit_price and (params['sl_type'] == "Signal-based (reverse EMA crossover)"):
                 ema_f, ema_s = curr['EMA_Fast'], curr['EMA_Slow']
                 prev_f, prev_s = df.iloc[i-1]['EMA_Fast'], df.iloc[i-1]['EMA_Slow']
                 
                 if active_pos['type'] == 1 and (ema_f < ema_s and prev_f >= prev_s):
                     exit_price = curr_price; reason = "Signal Exit"
                 elif active_pos['type'] == -1 and (ema_f > ema_s and prev_f <= prev_s):
                     exit_price = curr_price; reason = "Signal Exit"
            
            if exit_price:
                pnl = (exit_price - active_pos['entry']) * active_pos['qty'] if active_pos['type'] == 1 else (active_pos['entry'] - exit_price) * active_pos['qty']
                trades.append({
                    'Entry Date': active_pos['date'],
                    'Exit Date': curr_date,
                    'Type': "LONG" if active_pos['type'] == 1 else "SHORT",
                    'Entry': active_pos['entry'],
                    'Exit': exit_price,
                    'PnL': pnl,
                    'Reason': reason
                })
                active_pos = None
                continue # Trade closed, move to next candle
        
        # Check Entry (if no position)
        if not active_pos:
            # We mock the params/df passed to analyze_strategy using the slice
            # To optimize speed, we manually check condition or adapt analyze_strategy
            # Adapting analyze_strategy to take a slice is safest
            
            # Simple wrapper for backtest loop
            sig, res, _ = analyze_strategy(slice_df, params)
            
            if sig != 0:
                sl, tp = calculate_sl_tp(curr_price, sig, params, slice_df)
                if params['target_type'] == "Signal-based (reverse EMA crossover)": tp = 0
                if params['sl_type'] == "Signal-based (reverse EMA crossover)": sl = 0
                
                active_pos = {
                    'type': sig,
                    'entry': curr_price,
                    'sl': sl,
                    'target': tp,
                    'qty': params['quantity'],
                    'date': curr_date
                }
                
    return pd.DataFrame(trades)

# ==========================================
# 10. UI & MAIN APP
# ==========================================

# --- SIDEBAR CONFIG ---
st.sidebar.title("⚙️ Configuration")

# Asset & Time
asset = st.sidebar.text_input("Asset Ticker (yfinance)", value="^NSEI")
col_t1, col_t2 = st.sidebar.columns(2)
interval = col_t1.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"], index=5)
period = col_t2.selectbox("Period", ["1d", "5d", "1mo", "1y", "2y", "5y", "max"], index=3)

if not validate_interval_period(interval, period):
    st.sidebar.error("Invalid Interval/Period Combo!")

qty = st.sidebar.number_input("Quantity", min_value=1, value=50)

# Strategy Selection
st.sidebar.markdown("---")
strategy_type = st.sidebar.selectbox("Strategy", [
    "EMA Crossover", "Simple Buy", "Simple Sell", 
    "Price Threshold", "RSI-ADX-EMA", "Percentage Change",
    "AI Price Action", "Custom Builder"
])

strat_params = {'strategy_type': strategy_type, 'quantity': qty}

if strategy_type == "EMA Crossover":
    strat_params['ema_fast'] = st.sidebar.number_input("EMA Fast", 5, 200, 9)
    strat_params['ema_slow'] = st.sidebar.number_input("EMA Slow", 5, 200, 15)
    strat_params['min_angle'] = st.sidebar.number_input("Min Angle (°)", 0.0, 90.0, 1.0)
    strat_params['entry_filter'] = st.sidebar.selectbox("Entry Filter", ["Simple Crossover", "Custom Candle", "ATR-based Candle"])
    if strat_params['entry_filter'] == "Custom Candle":
        strat_params['filter_points'] = st.sidebar.number_input("Min Candle Points", 1, 1000, 10)
    elif strat_params['entry_filter'] == "ATR-based Candle":
        strat_params['filter_multiplier'] = st.sidebar.number_input("ATR Multiplier", 0.1, 5.0, 1.0)
    
    if st.sidebar.checkbox("Use ADX Filter"):
        strat_params['use_adx'] = True
        strat_params['adx_threshold'] = st.sidebar.number_input("ADX Threshold", 10, 50, 25)

elif strategy_type == "Price Threshold":
    strat_params['threshold_condition'] = st.sidebar.selectbox("Condition", [
        "LONG (Price >= Threshold)", "SHORT (Price >= Threshold)",
        "LONG (Price <= Threshold)", "SHORT (Price <= Threshold)"
    ])
    strat_params['threshold_price'] = st.sidebar.number_input("Threshold Price", 0.0, 100000.0, 24000.0)

elif strategy_type == "Percentage Change":
    strat_params['pct_threshold'] = st.sidebar.number_input("Threshold (%)", 0.01, 10.0, 0.5)
    strat_params['pct_direction'] = st.sidebar.selectbox("Direction", ["BUY on Fall", "SELL on Fall", "BUY on Rise", "SELL on Rise"])

elif strategy_type == "Custom Builder":
    st.sidebar.markdown("#### Conditions")
    if st.sidebar.button("Add Condition"):
        st.session_state['custom_conditions'].append({
            'active': True, 'ind1': 'Close', 'op': '>', 'use_comparison': False, 'ind2': 'EMA_20', 'value': 0, 'operator': '>', 'action': 'BUY'
        })
    
    # Render Builder UI
    new_conds = []
    indicators = ['Close', 'High', 'Low', 'RSI', 'ADX', 'EMA_Fast', 'EMA_Slow', 'EMA_20', 'EMA_50', 'SMA_20', 'SuperTrend', 'VWAP']
    for i, cond in enumerate(st.session_state['custom_conditions']):
        with st.sidebar.expander(f"Condition {i+1}"):
            cond['active'] = st.checkbox(f"Active##{i}", value=cond['active'])
            cond['action'] = st.selectbox(f"Action##{i}", ["BUY", "SELL"], index=0 if cond['action']=='BUY' else 1)
            cond['ind1'] = st.selectbox(f"Indicator 1##{i}", indicators, index=indicators.index(cond['ind1']) if cond['ind1'] in indicators else 0)
            cond['operator'] = st.selectbox(f"Operator##{i}", [">", "<", ">=", "<=", "==", "crosses_above", "crosses_below"], index=0)
            cond['use_comparison'] = st.checkbox(f"Compare with Indicator?##{i}", value=cond['use_comparison'])
            if cond['use_comparison']:
                cond['ind2'] = st.selectbox(f"Indicator 2##{i}", indicators, index=indicators.index(cond['ind2']) if cond['ind2'] in indicators else 0)
            else:
                cond['value'] = st.number_input(f"Value##{i}", value=float(cond['value']))
            new_conds.append(cond)
    st.session_state['custom_conditions'] = new_conds

# Risk Management
st.sidebar.markdown("---")
st.sidebar.subheader("Risk Management")
sl_type = st.sidebar.selectbox("SL Type", [
    "Custom Points", "Trailing SL (Points)", "Trailing SL + Current Candle", 
    "Volatility-Adjusted Trailing SL", "Break-even After 50% Target", 
    "ATR-based", "Signal-based (reverse EMA crossover)", "Current Candle Low/High"
])
strat_params['sl_type'] = sl_type
if "Points" in sl_type:
    strat_params['sl_points'] = st.sidebar.number_input("SL Points", 1, 1000, 10)
if "ATR" in sl_type or "Volatility" in sl_type:
    strat_params['atr_multiplier'] = st.sidebar.number_input("ATR Multiplier", 0.5, 5.0, 1.5)
if "Trailing" in sl_type:
    strat_params['trailing_threshold'] = st.sidebar.number_input("Trailing Threshold (Pts)", 0, 100, 0)

tp_type = st.sidebar.selectbox("Target Type", [
    "Custom Points", "Trailing Target (Display Only)", "50% Exit at Target (Partial)",
    "ATR-based", "Risk-Reward Based", "Signal-based (reverse EMA crossover)"
])
strat_params['target_type'] = tp_type
if "Points" in tp_type or "Trailing" in tp_type or "Partial" in tp_type:
    strat_params['target_points'] = st.sidebar.number_input("Target Points", 1, 2000, 20)
if "Risk-Reward" in tp_type:
    strat_params['rr_ratio'] = st.sidebar.number_input("RR Ratio", 1.0, 10.0, 2.0)

# --- MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📜 Trade History", "📝 Logs", "🧪 Backtest"])

# --- TAB 1: LIVE DASHBOARD ---
with tab1:
    # Controls
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    with c1:
        if st.button("▶ START TRADING", type="primary"):
            st.session_state['trading_active'] = True
    with c2:
        if st.button("⏹ STOP TRADING"):
            st.session_state['trading_active'] = False
            # Manual Close Logic
            if st.session_state['position']:
                df_curr = st.session_state['current_data']
                if df_curr is not None:
                    execute_trade("CLOSE", df_curr['Close'].iloc[-1], st.session_state['position']['quantity'], "Manual Close", strat_params, df_curr)
            log_msg("Trading Stopped Manually")
            st.rerun()
            
    with c3:
        status_color = "green" if st.session_state['trading_active'] else "grey"
        status_text = "ACTIVE" if st.session_state['trading_active'] else "STOPPED"
        st.markdown(f"<h3 style='color:{status_color}; margin:0'>● {status_text}</h3>", unsafe_allow_html=True)
    with c4:
        if st.button("Refresh"): st.rerun()
        
    # Info Board
    st.info(f"Asset: {asset} | Interval: {interval} | Strategy: {strategy_type} | SL: {sl_type}")

    # --- MAIN LOOP (Executed on Rerun) ---
    if st.session_state['trading_active']:
        with st.spinner("Fetching Live Data..."):
            df = get_data(asset, interval, period, mode="Live")
            
        if df is not None:
            df = calculate_indicators(df)
            st.session_state['current_data'] = df
            current_price = df['Close'].iloc[-1]
            
            # Trading Logic
            if st.session_state['position'] is None:
                # Look for Entry
                sig, reason, _ = analyze_strategy(df, strat_params)
                if sig != 0:
                    st.session_state['temp_signal'] = sig
                    execute_trade("OPEN", current_price, qty, reason, strat_params, df)
            else:
                # Manage Position
                update_trailing_logic(df, current_price, strat_params)
                check_exit_conditions(df, current_price, strat_params)
                
            # Auto-Refresh Mechanism
            time.sleep(1) # Visual delay
            st.rerun()

    # --- VISUALIZATION ---
    df = st.session_state['current_data']
    if df is not None:
        curr = df.iloc[-1]
        
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"{curr['Close']:.2f}", f"{curr['Close'] - df['Open'].iloc[-1]:.2f}")
        
        pos = st.session_state['position']
        pnl_val = 0.0
        pnl_delta = 0.0
        
        if pos:
            entry = pos['entry_price']
            if pos['type'] == 1:
                pnl_val = (curr['Close'] - entry) * pos['quantity']
            else:
                pnl_val = (entry - curr['Close']) * pos['quantity']
            
            pnl_str = f"{pnl_val:.2f}"
            delta_color = "normal" if pnl_val >= 0 else "inverse"
            m2.metric("Unrealized P&L", pnl_str, delta=pnl_str, delta_color=delta_color)
            
            sl_str = f"{pos['sl']:.2f}" if pos['sl'] != 0 else "Signal"
            tp_str = f"{pos['target']:.2f}" if pos['target'] != 0 else "Signal/Trail"
            m3.metric("Trade Info", f"{'LONG' if pos['type']==1 else 'SHORT'}", f"SL: {sl_str}")
            
            # Trailing info
            m4.metric("Trailing Profit", f"{st.session_state['trailing_profit_points']:.2f}", "Points Locked")
        else:
            m2.metric("Status", "Flat", "No Position")
            m3.metric("Signal", "Waiting", f"{strategy_type}")
            m4.metric("RSI / ADX", f"{curr['RSI']:.1f} / {curr['ADX']:.1f}")

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], line=dict(color='orange', width=1), name="EMA Fast"))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], line=dict(color='blue', width=1), name="EMA Slow"))
        
        if pos:
            fig.add_hline(y=pos['entry_price'], line_dash="dash", line_color="yellow", annotation_text="Entry")
            if pos['sl'] != 0:
                fig.add_hline(y=pos['sl'], line_dash="dot", line_color="red", annotation_text="SL")
            if pos['target'] != 0:
                fig.add_hline(y=pos['target'], line_dash="dot", line_color="green", annotation_text="Target")
                
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=20, b=0), template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{int(time.time())}")
    else:
        st.warning("Data not loaded. Start Trading or wait for fetch.")

# --- TAB 2: HISTORY ---
with tab2:
    st.markdown("### 📈 Trade History")
    hist = st.session_state['trade_history']
    if hist:
        df_hist = pd.DataFrame(hist)
        
        # Stats
        total_trades = len(df_hist)
        wins = len(df_hist[df_hist['PnL'] > 0])
        acc = (wins / total_trades) * 100
        total_pnl = df_hist['PnL'].sum()
        
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Total Trades", total_trades)
        h2.metric("Accuracy", f"{acc:.1f}%")
        h3.metric("Total P&L", f"{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color="normal" if total_pnl >=0 else "inverse")
        
        st.dataframe(df_hist.style.format({'Entry Price': '{:.2f}', 'Exit Price': '{:.2f}', 'PnL': '{:.2f}'}), use_container_width=True)
    else:
        st.info("No trades executed yet.")

# --- TAB 3: LOGS ---
with tab3:
    st.markdown("### 📝 Trade Logs")
    if st.session_state['trade_logs']:
        for log in reversed(st.session_state['trade_logs']):
            st.text(log)
    else:
        st.text("No logs available.")

# --- TAB 4: BACKTEST ---
with tab4:
    st.markdown("### 🧪 Historical Backtest")
    if st.button("Run Backtest"):
        with st.spinner("Running Backtest..."):
            bt_results = run_backtest(asset, interval, period, strat_params)
            
            if not bt_results.empty:
                b_total = len(bt_results)
                b_wins = len(bt_results[bt_results['PnL'] > 0])
                b_acc = (b_wins / b_total * 100) if b_total > 0 else 0
                b_pnl = bt_results['PnL'].sum()
                
                b1, b2, b3 = st.columns(3)
                b1.metric("Total Backtest Trades", b_total)
                b2.metric("Win Rate", f"{b_acc:.1f}%")
                b3.metric("Net P&L", f"{b_pnl:.2f}", delta=f"{b_pnl:.2f}", delta_color="normal" if b_pnl >=0 else "inverse")
                
                st.dataframe(bt_results, use_container_width=True)
            else:
                st.warning("No trades found in backtest period with current strategy settings.")

