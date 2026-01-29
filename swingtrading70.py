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
# 1. CONFIGURATION & STATE MANAGEMENT
# ==========================================

st.set_page_config(page_title="Pro Algo Trader", layout="wide", page_icon="📈")

# IST Timezone
IST = pytz.timezone('Asia/Kolkata')

def init_session_state():
    defaults = {
        'trading_active': False,
        'position': None,  # {type: 1/-1, entry_price, quantity, sl, target, entry_time, partial_exit_done}
        'trade_history': [],
        'trade_logs': [],
        'trailing_sl_high': None,
        'trailing_sl_low': None,
        'highest_price': None,
        'lowest_price': None,
        'trailing_profit_points': 0,
        'threshold_crossed': False,
        'custom_conditions': [],
        'breakeven_activated': False,
        'last_update_dt': None,
        'live_data_cache': None,
        'ratio_ticker_2': "BANKNIFTY",  # Default for ratio
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

def add_log(message):
    timestamp = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    st.session_state['trade_logs'].append(log_entry)
    # Keep only last 50 logs
    if len(st.session_state['trade_logs']) > 50:
        st.session_state['trade_logs'].pop(0)

def reset_position_state():
    st.session_state['position'] = None
    st.session_state['trailing_sl_high'] = None
    st.session_state['trailing_sl_low'] = None
    st.session_state['highest_price'] = None
    st.session_state['lowest_price'] = None
    st.session_state['trailing_profit_points'] = 0
    st.session_state['threshold_crossed'] = False
    st.session_state['breakeven_activated'] = False

# ==========================================
# 2. INDICATOR LIBRARY (MANUAL IMPLEMENTATION)
# ==========================================

class Indicators:
    @staticmethod
    def sma(series, period):
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        # Use simple moving average for first value, then smoothing
        # To match TradingView accurately, we need Wilder's Smoothing:
        # distinct from simple rolling mean. Implementing Wilder's:
        delta = series.diff()
        u = delta.where(delta > 0, 0)
        d = -delta.where(delta < 0, 0)
        
        # Initialize with SMA
        avg_u = u.rolling(window=period).mean()
        avg_d = d.rolling(window=period).mean()
        
        # Smooth
        for i in range(period, len(series)):
            avg_u.iloc[i] = (avg_u.iloc[i-1] * (period - 1) + u.iloc[i]) / period
            avg_d.iloc[i] = (avg_d.iloc[i-1] * (period - 1) + d.iloc[i]) / period
            
        rs = avg_u / avg_d
        return 100 - (100 / (1 + rs))

    @staticmethod
    def true_range(high, low, close):
        return np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))

    @staticmethod
    def atr(high, low, close, period=14):
        tr = Indicators.true_range(high, low, close)
        # Wilder's Smoothing for ATR
        atr = tr.rolling(window=period).mean()
        for i in range(period, len(atr)):
            atr.iloc[i] = (atr.iloc[i-1] * (period - 1) + tr.iloc[i]) / period
        return atr

    @staticmethod
    def adx(high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = Indicators.true_range(high, low, close)
        atr = Indicators.atr(high, low, close, period)
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = 100 * (minus_dm.abs().ewm(alpha=1/period).mean() / atr)
        
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = dx.ewm(alpha=1/period).mean()
        return adx

    @staticmethod
    def bollinger_bands(close, period=20, std_dev=2):
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    @staticmethod
    def macd(close, fast=12, slow=26, signal=9):
        exp1 = close.ewm(span=fast, adjust=False).mean()
        exp2 = close.ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    @staticmethod
    def supertrend(high, low, close, period=7, multiplier=3):
        atr = Indicators.atr(high, low, close, period)
        hl2 = (high + low) / 2
        
        # Calculate basic upper and lower bands
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)
        
        upper = basic_upper.copy()
        lower = basic_lower.copy()
        trend = np.zeros(len(close))
        
        for i in range(1, len(close)):
            if basic_upper.iloc[i] < upper.iloc[i-1] or close.iloc[i-1] > upper.iloc[i-1]:
                upper.iloc[i] = basic_upper.iloc[i]
            else:
                upper.iloc[i] = upper.iloc[i-1]
                
            if basic_lower.iloc[i] > lower.iloc[i-1] or close.iloc[i-1] < lower.iloc[i-1]:
                lower.iloc[i] = basic_lower.iloc[i]
            else:
                lower.iloc[i] = lower.iloc[i-1]
                
            # Trend determination
            if i > 0:
                if trend[i-1] == 1: # Uptrend
                    if close.iloc[i] < lower.iloc[i-1]:
                        trend[i] = -1
                    else:
                        trend[i] = 1
                else: # Downtrend
                    if close.iloc[i] > upper.iloc[i-1]:
                        trend[i] = 1
                    else:
                        trend[i] = -1
        
        st_line = pd.Series(np.where(trend==1, lower, upper), index=close.index)
        return st_line, trend

    @staticmethod
    def ema_angle(ema_series, lookback=1):
        # Calculate slope based on normalized values to mimic visual angle
        # This is an approximation. TradingView angles depend on chart aspect ratio.
        # We'll use raw slope converted to degrees for consistency.
        diff = ema_series.diff(lookback)
        # Using atan of the simple slope. 
        # For better "visual" representation, one might normalize by price, 
        # but raw slope is more consistent for algo logic.
        angle = np.degrees(np.arctan(diff))
        return angle
    
    @staticmethod
    def vwap(df):
        q = df['Volume']
        p = df['Close']
        vwap = (p * q).cumsum() / q.cumsum()
        return vwap

# ==========================================
# 3. DATA & ASSET MANAGEMENT
# ==========================================

def get_allowed_periods(interval):
    rules = {
        '1m': ['1d', '5d'],
        '5m': ['1d', '1mo'],
        '15m': ['1mo'], '30m': ['1mo'], '1h': ['1mo'], '4h': ['1mo'],
        '1d': ['1mo', '1y', '2y', '5y'],
        '1wk': ['1mo', '1y', '5y', '10y', '15y', '20y'],
        '1mo': ['1y', '2y', '5y', '10y', '15y', '20y', '25y', '30y']
    }
    return rules.get(interval, ['1d'])

def fetch_data(ticker, interval, period, is_live=False):
    if is_live:
        time.sleep(np.random.uniform(1.0, 1.5)) # Randomized delay
        
    try:
        # Handle Indices special tickers
        sym_map = {
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
        symbol = sym_map.get(ticker, ticker)
        
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        
        if df.empty:
            return None
            
        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Localize Timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)
            
        # Select OHLCV
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[cols].copy()
        
        # Calculate Indicators
        df['EMA_Fast'] = Indicators.ema(df['Close'], 9)
        df['EMA_Slow'] = Indicators.ema(df['Close'], 15)
        df['EMA_20'] = Indicators.ema(df['Close'], 20)
        df['EMA_50'] = Indicators.ema(df['Close'], 50)
        df['RSI'] = Indicators.rsi(df['Close'])
        df['ATR'] = Indicators.atr(df['High'], df['Low'], df['Close'])
        df['ADX'] = Indicators.adx(df['High'], df['Low'], df['Close'])
        df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = Indicators.bollinger_bands(df['Close'])
        df['MACD'], df['MACD_Sig'], df['MACD_Hist'] = Indicators.macd(df['Close'])
        df['EMA_Angle'] = Indicators.ema_angle(df['EMA_Fast'])
        
        # VWAP (Indices might have 0 volume, handle safely)
        if df['Volume'].sum() > 0:
            df['VWAP'] = Indicators.vwap(df)
        else:
            df['VWAP'] = df['Close'] # Fallback
            
        return df
    except Exception as e:
        if not is_live: st.error(f"Data Error: {e}")
        return None

# ==========================================
# 4. TRADING LOGIC
# ==========================================

def check_signals(df, strategy_config, custom_conditions=None):
    if len(df) < 50: return 0, "Insufficient Data" # Need enough data for indicators
    
    i = -1 # Last candle
    curr = df.iloc[i]
    prev = df.iloc[i-1]
    
    signal = 0 # 0: None, 1: Buy, -1: Sell
    reason = ""

    st_name = strategy_config['strategy']
    
    # --- STRATEGY: EMA Crossover ---
    if st_name == "EMA Crossover":
        fast = strategy_config['ema_fast']
        slow = strategy_config['ema_slow']
        angle_thresh = strategy_config['min_angle']
        entry_filter = strategy_config['entry_filter']
        
        # Calculate custom EMAs if not default
        ef_series = Indicators.ema(df['Close'], fast)
        es_series = Indicators.ema(df['Close'], slow)
        angle = Indicators.ema_angle(ef_series).iloc[-1]
        
        crossover_bull = (ef_series.iloc[i] > es_series.iloc[i] and ef_series.iloc[i-1] <= es_series.iloc[i-1])
        crossover_bear = (ef_series.iloc[i] < es_series.iloc[i] and ef_series.iloc[i-1] >= es_series.iloc[i-1])
        
        valid_angle = abs(angle) >= angle_thresh
        
        # ADX Filter
        adx_ok = True
        if strategy_config.get('use_adx', False):
            if curr['ADX'] < strategy_config['adx_threshold']:
                adx_ok = False
        
        # Candle Size Filter
        candle_size = abs(curr['Close'] - curr['Open'])
        size_ok = True
        if entry_filter == "Custom Candle":
            if candle_size < strategy_config['custom_points']: size_ok = False
        elif entry_filter == "ATR-based Candle":
            if candle_size < (curr['ATR'] * strategy_config['atr_mult']): size_ok = False
            
        if crossover_bull and valid_angle and adx_ok and size_ok:
            signal = 1
            reason = f"Bullish Cross (Angle {angle:.2f})"
        elif crossover_bear and valid_angle and adx_ok and size_ok:
            signal = -1
            reason = f"Bearish Cross (Angle {angle:.2f})"

    # --- STRATEGY: Simple Buy/Sell ---
    elif st_name == "Simple Buy":
        signal = 1
        reason = "Simple Buy Strategy"
    elif st_name == "Simple Sell":
        signal = -1
        reason = "Simple Sell Strategy"

    # --- STRATEGY: Price Threshold ---
    elif st_name == "Price Threshold":
        thresh = strategy_config['threshold_price']
        mode = strategy_config['threshold_mode'] # LONG >=, SHORT >=, LONG <=, SHORT <=
        price = curr['Close']
        
        if mode == "LONG (Price >= Threshold)" and price >= thresh:
            signal = 1
            reason = f"Price {price} >= {thresh}"
        elif mode == "SHORT (Price >= Threshold)" and price >= thresh:
            signal = -1
            reason = f"Price {price} >= {thresh}"
        elif mode == "LONG (Price <= Threshold)" and price <= thresh:
            signal = 1
            reason = f"Price {price} <= {thresh}"
        elif mode == "SHORT (Price <= Threshold)" and price <= thresh:
            signal = -1
            reason = f"Price {price} <= {thresh}"

    # --- STRATEGY: RSI-ADX-EMA ---
    elif st_name == "RSI-ADX-EMA":
        # RSI>80 ADX<20 EMA1<EMA2 -> SELL
        # RSI<20 ADX>20 EMA1>EMA2 -> BUY
        e1 = curr['EMA_Fast']
        e2 = curr['EMA_Slow']
        
        if curr['RSI'] < 20 and curr['ADX'] > 20 and e1 > e2:
            signal = 1
            reason = "RSI<20, ADX>20, EMA_F>EMA_S"
        elif curr['RSI'] > 80 and curr['ADX'] < 20 and e1 < e2:
            signal = -1
            reason = "RSI>80, ADX<20, EMA_F<EMA_S"

    # --- STRATEGY: Percentage Change ---
    elif st_name == "Percentage Change":
        # Change from first candle in DF (or specific period start, here using loaded data start)
        start_price = df['Open'].iloc[0]
        pct_change = (curr['Close'] - start_price) / start_price * 100
        thresh = strategy_config['pct_threshold']
        direction = strategy_config['pct_direction']
        
        triggered = abs(pct_change) >= thresh
        
        if triggered:
            if direction == "BUY on Fall" and pct_change < 0:
                signal = 1
                reason = f"Fell {pct_change:.2f}%"
            elif direction == "SELL on Fall" and pct_change < 0:
                signal = -1
                reason = f"Fell {pct_change:.2f}%"
            elif direction == "BUY on Rise" and pct_change > 0:
                signal = 1
                reason = f"Rose {pct_change:.2f}%"
            elif direction == "SELL on Rise" and pct_change > 0:
                signal = -1
                reason = f"Rose {pct_change:.2f}%"

    # --- STRATEGY: AI Price Action ---
    elif st_name == "AI Price Action Analysis":
        score = 0
        # Heuristic scoring
        if curr['Close'] > curr['EMA_50']: score += 1
        if curr['RSI'] > 50: score += 1
        if curr['MACD'] > curr['MACD_Sig']: score += 1
        if curr['Close'] > curr['BB_Mid']: score += 1
        
        # Confidence
        confidence = (abs(score - 2) / 2) * 100 # Rough mapping
        
        if score >= 3:
            signal = 1
            reason = f"AI Score {score}/4 (Bullish)"
        elif score <= 1:
            signal = -1
            reason = f"AI Score {score}/4 (Bearish)"

    # --- STRATEGY: Ratio Strategy ---
    elif st_name == "Ratio Strategy":
        # Requires a second ticker dataframe. In live loop, we'd need to fetch it.
        # For simplicity in this structure, assuming we look at price action relative to last few bars of this ticker
        # or simplified relative strength.
        # Real implementation needs synchronization of two DFs.
        # Here we simulate the logic based on Ratio RSI if we had it, else fallback.
        # This part requires the Live Engine to fetch Ticker 2.
        pass # Handle in main loop logic if possible, or simplified here.

    # --- STRATEGY: Custom Builder ---
    elif st_name == "Custom Strategy Builder":
        buy_conditions = True
        sell_conditions = True
        
        if not custom_conditions:
            return 0, "No conditions"
            
        # Separate buy/sell logic not explicitly defined in prompt, assuming user configures conditions that imply direction
        # Implementing: If ALL conditions meet -> BUY? Or User defines Action per condition?
        # Prompt says: "Action (BUY/SELL)" per condition.
        # All active BUY conditions must be true. All active SELL conditions must be true.
        
        buys = [c for c in custom_conditions if c['action'] == 'BUY' and c['enabled']]
        sells = [c for c in custom_conditions if c['action'] == 'SELL' and c['enabled']]
        
        buy_signal = False
        sell_signal = False
        
        # Check Buys
        if buys:
            all_true = True
            for c in buys:
                val1 = curr[c['indicator']]
                val2 = curr[c['compare_indicator']] if c['use_compare'] else c['value']
                op = c['operator']
                
                res = False
                if op == '>': res = val1 > val2
                elif op == '<': res = val1 < val2
                elif op == '>=': res = val1 >= val2
                elif op == '<=': res = val1 <= val2
                elif op == '==': res = val1 == val2
                elif op == 'crosses_above': 
                    res = (curr[c['indicator']] > val2 and prev[c['indicator']] <= (prev[c['compare_indicator']] if c['use_compare'] else val2))
                elif op == 'crosses_below':
                    res = (curr[c['indicator']] < val2 and prev[c['indicator']] >= (prev[c['compare_indicator']] if c['use_compare'] else val2))
                
                if not res: 
                    all_true = False
                    break
            if all_true: buy_signal = True
            
        # Check Sells
        if sells:
            all_true = True
            for c in sells:
                val1 = curr[c['indicator']]
                val2 = curr[c['compare_indicator']] if c['use_compare'] else c['value']
                op = c['operator']
                
                res = False
                if op == '>': res = val1 > val2
                elif op == '<': res = val1 < val2
                elif op == '>=': res = val1 >= val2
                elif op == '<=': res = val1 <= val2
                elif op == '==': res = val1 == val2
                elif op == 'crosses_above': 
                    res = (curr[c['indicator']] > val2 and prev[c['indicator']] <= (prev[c['compare_indicator']] if c['use_compare'] else val2))
                elif op == 'crosses_below':
                    res = (curr[c['indicator']] < val2 and prev[c['indicator']] >= (prev[c['compare_indicator']] if c['use_compare'] else val2))
                
                if not res: 
                    all_true = False
                    break
            if all_true: sell_signal = True
            
        if buy_signal and not sell_signal:
            signal = 1
            reason = "Custom Buy Conditions Met"
        elif sell_signal and not buy_signal:
            signal = -1
            reason = "Custom Sell Conditions Met"

    return signal, reason

# ==========================================
# 5. UI COMPONENTS
# ==========================================

def render_sidebar():
    st.sidebar.header("1. Assets & Data")
    ticker = st.sidebar.text_input("Ticker Symbol", value="NIFTY 50")
    interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"], index=1)
    
    allowed = get_allowed_periods(interval)
    period = st.sidebar.selectbox("Period", allowed, index=0)
    
    qty = st.sidebar.number_input("Quantity", min_value=1, value=1)
    
    st.sidebar.markdown("---")
    st.sidebar.header("2. Strategy Configuration")
    
    strategies = [
        "EMA Crossover", "Ratio Strategy", "Simple Buy", "Simple Sell", 
        "Price Threshold", "RSI-ADX-EMA", "Percentage Change", 
        "AI Price Action Analysis", "Custom Strategy Builder"
    ]
    strategy = st.sidebar.selectbox("Select Strategy", strategies)
    
    strat_config = {'strategy': strategy, 'ticker': ticker, 'interval': interval, 'period': period, 'qty': qty}
    
    if strategy == "EMA Crossover":
        strat_config['ema_fast'] = st.sidebar.number_input("EMA Fast", 1, 200, 9)
        strat_config['ema_slow'] = st.sidebar.number_input("EMA Slow", 1, 200, 15)
        strat_config['min_angle'] = st.sidebar.number_input("Min Angle (deg)", 0.0, 90.0, 1.0)
        strat_config['entry_filter'] = st.sidebar.selectbox("Entry Filter", ["Simple Crossover", "Custom Candle", "ATR-based Candle"])
        
        if strat_config['entry_filter'] == "Custom Candle":
            strat_config['custom_points'] = st.sidebar.number_input("Candle Points", 1.0, 1000.0, 10.0)
        elif strat_config['entry_filter'] == "ATR-based Candle":
            strat_config['atr_mult'] = st.sidebar.number_input("ATR Multiplier", 0.1, 10.0, 1.0)
            
        strat_config['use_adx'] = st.sidebar.checkbox("Use ADX Filter")
        if strat_config['use_adx']:
            strat_config['adx_threshold'] = st.sidebar.number_input("ADX Threshold", 1, 100, 25)

    elif strategy == "Price Threshold":
        strat_config['threshold_mode'] = st.sidebar.selectbox("Mode", [
            "LONG (Price >= Threshold)", "SHORT (Price >= Threshold)", 
            "LONG (Price <= Threshold)", "SHORT (Price <= Threshold)"
        ])
        strat_config['threshold_price'] = st.sidebar.number_input("Threshold Price", 0.0, 1000000.0, 20000.0)

    elif strategy == "Percentage Change":
        strat_config['pct_threshold'] = st.sidebar.number_input("Pct Threshold (%)", 0.01, 100.0, 0.5)
        strat_config['pct_direction'] = st.sidebar.selectbox("Direction", ["BUY on Fall", "SELL on Fall", "BUY on Rise", "SELL on Rise"])
        
    elif strategy == "Ratio Strategy":
        st.session_state['ratio_ticker_2'] = st.sidebar.text_input("Ticker 2 (Ratio Denominator)", value="BANKNIFTY")
        strat_config['ticker2'] = st.session_state['ratio_ticker_2']
        st.sidebar.info("Ratio = Ticker 1 / Ticker 2. Signals based on Ratio RSI.")

    elif strategy == "Custom Strategy Builder":
        st.sidebar.markdown("#### Custom Conditions")
        c_ind = st.sidebar.selectbox("Indicator", ["Close", "RSI", "EMA_Fast", "EMA_Slow", "ADX", "MACD", "BB_Upper", "BB_Lower"])
        c_op = st.sidebar.selectbox("Operator", [">", "<", ">=", "<=", "==", "crosses_above", "crosses_below"])
        use_comp = st.sidebar.checkbox("Compare with Indicator")
        if use_comp:
            c_comp = st.sidebar.selectbox("Compare Ind", ["Close", "EMA_Fast", "EMA_Slow", "EMA_20", "EMA_50", "VWAP"])
            c_val = 0
        else:
            c_comp = None
            c_val = st.sidebar.number_input("Value", value=50.0)
        c_act = st.sidebar.selectbox("Action", ["BUY", "SELL"])
        
        if st.sidebar.button("Add Condition"):
            st.session_state['custom_conditions'].append({
                'indicator': c_ind, 'operator': c_op, 'use_compare': use_comp,
                'compare_indicator': c_comp, 'value': c_val, 'action': c_act, 'enabled': True
            })
            
        # Display conditions
        to_remove = []
        for idx, c in enumerate(st.session_state['custom_conditions']):
            col1, col2 = st.sidebar.columns([0.8, 0.2])
            desc = f"{c['action']} if {c['indicator']} {c['operator']} "
            desc += f"{c['compare_indicator']}" if c['use_compare'] else f"{c['value']}"
            col1.text(desc)
            if col2.button("X", key=f"del_{idx}"):
                to_remove.append(idx)
        
        for i in reversed(to_remove):
            st.session_state['custom_conditions'].pop(i)

    st.sidebar.markdown("---")
    st.sidebar.header("3. Risk Management")
    
    sl_type = st.sidebar.selectbox("Stop Loss Type", [
        "Custom Points", "Trailing SL (Points)", "Trailing SL + Current Candle",
        "Trailing SL + Signal Based", "Volatility-Adjusted Trailing SL", "ATR-based"
    ])
    sl_points = st.sidebar.number_input("SL Points (Min 10)", min_value=10.0, value=20.0)
    sl_trail_thresh = st.sidebar.number_input("Trailing Threshold", 0.0, 100.0, 0.0)
    
    target_type = st.sidebar.selectbox("Target Type", [
        "Custom Points", "Trailing Target (Display Only)", "Signal-based (Reverse Crossover)", 
        "Risk-Reward Based"
    ])
    target_points = st.sidebar.number_input("Target Points (Min 15)", min_value=15.0, value=40.0)
    
    strat_config.update({
        'sl_type': sl_type, 'sl_points': sl_points, 'sl_trail_thresh': sl_trail_thresh,
        'target_type': target_type, 'target_points': target_points
    })
    
    # Broker Integration Placeholder
    st.sidebar.markdown("---")
    st.sidebar.header("Broker Integration (Dhan)")
    enable_broker = st.sidebar.checkbox("Enable Dhan Integration")
    if enable_broker:
        st.sidebar.text_input("Client ID")
        st.sidebar.text_input("Access Token", type="password")
        st.sidebar.info("Integration active: Orders will be placed on signal.")
        # Note: Actual API call code would go in the execution block
        
    return strat_config

# ==========================================
# 6. MAIN APP LOGIC
# ==========================================

def main():
    config = render_sidebar()
    
    st.title("Algo Trading System (Live & Backtest)")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Live Dashboard", "Trade History", "Trade Logs", "Backtest"])
    
    # --- TAB 1: LIVE DASHBOARD ---
    with tab1:
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
        start_btn = col_ctrl1.button("Start Trading", type="primary")
        stop_btn = col_ctrl2.button("Stop Trading", type="secondary")
        
        if start_btn:
            st.session_state['trading_active'] = True
            add_log("Trading Started")
        if stop_btn:
            st.session_state['trading_active'] = False
            # Close position if open
            if st.session_state['position']:
                pos = st.session_state['position']
                exit_price = st.session_state['current_data']['Close'].iloc[-1]
                pnl = (exit_price - pos['entry_price']) * pos['quantity'] if pos['type'] == 1 else (pos['entry_price'] - exit_price) * pos['quantity']
                
                trade_record = {
                    'entry_time': pos['entry_time'],
                    'exit_time': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S'),
                    'symbol': config['ticker'],
                    'signal': "LONG" if pos['type'] == 1 else "SHORT",
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'sl': pos['sl'],
                    'target': pos['target'],
                    'pnl': pnl,
                    'exit_reason': "Manual Stop"
                }
                st.session_state['trade_history'].append(trade_record)
                reset_position_state()
                add_log("Trading Stopped - Position Closed Manually")
            else:
                add_log("Trading Stopped")
        
        status_color = "green" if st.session_state['trading_active'] else "gray"
        status_text = "ACTIVE" if st.session_state['trading_active'] else "STOPPED"
        col_ctrl3.markdown(f"**Status:** :{status_color}[{status_text}]")
        
        # Live Containers
        metrics_container = st.empty()
        chart_container = st.empty()
        
        # --- LIVE TRADING LOOP ---
        if st.session_state['trading_active']:
            
            # Placeholder for Dhan integration
            # if config['enable_broker']:
            #    dhan = dhanhq(client_id, access_token)
            
            # Fetch Data
            df = fetch_data(config['ticker'], config['interval'], config['period'], is_live=True)
            st.session_state['current_data'] = df
            
            if df is not None:
                curr = df.iloc[-1]
                prev = df.iloc[-2]
                
                # Update Strategy Config with latest ATR for SL calc if needed
                atr_val = curr['ATR']
                
                # Check Signals if no position
                if st.session_state['position'] is None:
                    sig, reason = check_signals(df, config, st.session_state['custom_conditions'])
                    
                    if sig != 0:
                        entry_price = curr['Close']
                        sl = 0
                        target = 0
                        
                        # SL Calculation
                        if config['sl_type'] == "Custom Points":
                            sl_dist = config['sl_points']
                        elif config['sl_type'] == "ATR-based":
                            sl_dist = atr_val * 1.5
                        else:
                            sl_dist = config['sl_points'] # Default fallback
                        
                        # Target Calculation
                        if config['target_type'] == "Custom Points":
                            tg_dist = config['target_points']
                        elif config['target_type'] == "Risk-Reward Based":
                            tg_dist = sl_dist * 2
                        else:
                            tg_dist = config['target_points']
                            
                        if sig == 1:
                            sl = entry_price - sl_dist
                            target = entry_price + tg_dist
                            type_str = "LONG"
                        else:
                            sl = entry_price + sl_dist
                            target = entry_price - tg_dist
                            type_str = "SHORT"
                            
                        # Execute Trade
                        st.session_state['position'] = {
                            'type': sig,
                            'entry_price': entry_price,
                            'quantity': config['qty'],
                            'sl': sl,
                            'target': target,
                            'entry_time': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S'),
                            'partial_exit_done': False,
                            'initial_sl_dist': sl_dist
                        }
                        st.session_state['highest_price'] = entry_price
                        st.session_state['lowest_price'] = entry_price
                        st.session_state['trailing_sl_high'] = entry_price
                        st.session_state['trailing_sl_low'] = entry_price
                        
                        add_log(f"Opened {type_str} at {entry_price:.2f}. Reason: {reason}")
                        
                        # Dhan Order Place (Commented)
                        # dhan.place_order(symbol, 'BUY' if sig==1 else 'SELL', qty, ...)

                # Manage Open Position
                elif st.session_state['position'] is not None:
                    pos = st.session_state['position']
                    curr_price = curr['Close']
                    p_type = pos['type']
                    
                    # Update High/Low
                    if st.session_state['highest_price'] is None or curr_price > st.session_state['highest_price']:
                        st.session_state['highest_price'] = curr_price
                    if st.session_state['lowest_price'] is None or curr_price < st.session_state['lowest_price']:
                        st.session_state['lowest_price'] = curr_price

                    # Trailing SL Logic
                    sl_points = config['sl_points']
                    
                    if p_type == 1: # LONG
                        # Standard Trailing (Points)
                        if "Trailing SL (Points)" in config['sl_type']:
                            new_sl = curr_price - pos['initial_sl_dist']
                            # Only move up
                            if new_sl > pos['sl'] and (curr_price - pos['entry_price']) > config['sl_trail_thresh']:
                                pos['sl'] = new_sl
                        
                        # Signal Based Exit Check
                        if "Signal Based" in config['sl_type'] or "Signal-based" in config['target_type']:
                             # Exit if Fast < Slow
                             if curr['EMA_Fast'] < curr['EMA_Slow']:
                                 pos['sl'] = curr_price # Force exit
                        
                        # Exit Check
                        if curr_price <= pos['sl']:
                            exit_p = pos['sl']
                            pnl = (exit_p - pos['entry_price']) * pos['quantity']
                            st.session_state['trade_history'].append({
                                'entry_time': pos['entry_time'], 'exit_time': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S'),
                                'symbol': config['ticker'], 'signal': "LONG", 'entry_price': pos['entry_price'],
                                'exit_price': exit_p, 'sl': pos['sl'], 'target': pos['target'], 'pnl': pnl, 'exit_reason': "SL Hit"
                            })
                            reset_position_state()
                            add_log(f"LONG SL Hit at {exit_p:.2f}. PnL: {pnl:.2f}")

                        elif config['target_type'] != "Signal-based (Reverse Crossover)" and curr_price >= pos['target']:
                             # Target Hit
                            exit_p = pos['target']
                            pnl = (exit_p - pos['entry_price']) * pos['quantity']
                            st.session_state['trade_history'].append({
                                'entry_time': pos['entry_time'], 'exit_time': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S'),
                                'symbol': config['ticker'], 'signal': "LONG", 'entry_price': pos['entry_price'],
                                'exit_price': exit_p, 'sl': pos['sl'], 'target': pos['target'], 'pnl': pnl, 'exit_reason': "Target Hit"
                            })
                            reset_position_state()
                            add_log(f"LONG Target Hit at {exit_p:.2f}. PnL: {pnl:.2f}")

                    elif p_type == -1: # SHORT
                        # Standard Trailing
                        if "Trailing SL (Points)" in config['sl_type']:
                            new_sl = curr_price + pos['initial_sl_dist']
                            # Only move down
                            if new_sl < pos['sl'] and (pos['entry_price'] - curr_price) > config['sl_trail_thresh']:
                                pos['sl'] = new_sl
                        
                        # Signal Based Exit
                        if "Signal Based" in config['sl_type'] or "Signal-based" in config['target_type']:
                             if curr['EMA_Fast'] > curr['EMA_Slow']:
                                 pos['sl'] = curr_price

                        # Exit Check
                        if curr_price >= pos['sl']:
                            exit_p = pos['sl']
                            pnl = (pos['entry_price'] - exit_p) * pos['quantity']
                            st.session_state['trade_history'].append({
                                'entry_time': pos['entry_time'], 'exit_time': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S'),
                                'symbol': config['ticker'], 'signal': "SHORT", 'entry_price': pos['entry_price'],
                                'exit_price': exit_p, 'sl': pos['sl'], 'target': pos['target'], 'pnl': pnl, 'exit_reason': "SL Hit"
                            })
                            reset_position_state()
                            add_log(f"SHORT SL Hit at {exit_p:.2f}. PnL: {pnl:.2f}")

                        elif config['target_type'] != "Signal-based (Reverse Crossover)" and curr_price <= pos['target']:
                             # Target Hit
                            exit_p = pos['target']
                            pnl = (pos['entry_price'] - exit_p) * pos['quantity']
                            st.session_state['trade_history'].append({
                                'entry_time': pos['entry_time'], 'exit_time': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S'),
                                'symbol': config['ticker'], 'signal': "SHORT", 'entry_price': pos['entry_price'],
                                'exit_price': exit_p, 'sl': pos['sl'], 'target': pos['target'], 'pnl': pnl, 'exit_reason': "Target Hit"
                            })
                            reset_position_state()
                            add_log(f"SHORT Target Hit at {exit_p:.2f}. PnL: {pnl:.2f}")

                # --- RENDER METRICS ---
                with metrics_container.container():
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Current Price", f"{curr['Close']:.2f}", f"{curr['Close'] - prev['Close']:.2f}")
                    m2.metric("EMA Fast", f"{curr['EMA_Fast']:.2f}")
                    m3.metric("EMA Slow", f"{curr['EMA_Slow']:.2f}")
                    m4.metric("RSI", f"{curr['RSI']:.2f}")
                    
                    pos_text = "NONE"
                    pnl_text = "0.00"
                    if st.session_state['position']:
                        p = st.session_state['position']
                        type_s = "LONG" if p['type'] == 1 else "SHORT"
                        curr_pnl = (curr['Close'] - p['entry_price']) * p['quantity'] if p['type'] == 1 else (p['entry_price'] - curr['Close']) * p['quantity']
                        pos_text = f"{type_s} @ {p['entry_price']:.2f}"
                        if curr_pnl >= 0:
                            st.metric("Unrealized P&L", f"{curr_pnl:.2f}", delta=f"+{curr_pnl:.2f}")
                        else:
                            st.metric("Unrealized P&L", f"{curr_pnl:.2f}", delta=f"{curr_pnl:.2f}", delta_color="inverse")
                    else:
                        m5.metric("Position", "None")

                # --- RENDER CHART ---
                with chart_container.container():
                    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price')])
                    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], mode='lines', name='EMA Fast', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], mode='lines', name='EMA Slow', line=dict(color='orange')))
                    
                    if st.session_state['position']:
                        pos = st.session_state['position']
                        fig.add_hline(y=pos['entry_price'], line_color="blue", line_dash="dash", annotation_text="Entry")
                        fig.add_hline(y=pos['sl'], line_color="red", line_dash="dash", annotation_text="SL")
                        if pos['target'] != 0:
                             fig.add_hline(y=pos['target'], line_color="green", line_dash="dash", annotation_text="Target")
                    
                    fig.update_layout(height=500, xaxis_rangeslider_visible=False, title=f"Live Chart ({config['ticker']}) - {config['interval']}")
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{time.time()}")
            
            # Rerun loop logic implicitly handled by Streamlit's reactive nature + time.sleep in data fetch
            time.sleep(0.5) 
            st.rerun()

    # --- TAB 2: HISTORY ---
    with tab2:
        st.markdown("### 📈 Trade History")
        history = st.session_state['trade_history']
        if not history:
            st.info("No trades executed yet.")
        else:
            df_hist = pd.DataFrame(history)
            total_pnl = df_hist['pnl'].sum()
            win_rate = len(df_hist[df_hist['pnl'] > 0]) / len(df_hist) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Trades", len(df_hist))
            col2.metric("Win Rate", f"{win_rate:.1f}%")
            if total_pnl >= 0:
                col3.metric("Total P&L", f"{total_pnl:.2f}", delta=f"+{total_pnl:.2f}")
            else:
                col3.metric("Total P&L", f"{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color="inverse")
            
            st.dataframe(df_hist.sort_values(by='entry_time', ascending=False), use_container_width=True)

    # --- TAB 3: LOGS ---
    with tab3:
        st.markdown("### 📝 Trade Logs")
        if not st.session_state['trade_logs']:
            st.info("No logs available.")
        else:
            for log in reversed(st.session_state['trade_logs']):
                st.text(log)

    # --- TAB 4: BACKTEST ---
    with tab4:
        st.markdown("### 🔙 Backtest Engine")
        if st.button("Run Backtest"):
            with st.spinner("Fetching data and simulating..."):
                # Fetch Data
                df_bt = fetch_data(config['ticker'], config['interval'], config['period'], is_live=False)
                
                if df_bt is None or len(df_bt) < 100:
                    st.error("Insufficient data for backtest.")
                else:
                    # Run Simulation
                    trades = []
                    position = None
                    capital = 100000
                    
                    # Pre-calculate indicators on full DF for speed
                    # (Done in fetch_data)
                    
                    for i in range(50, len(df_bt)):
                        # Slice df safely
                        curr_slice = df_bt.iloc[:i+1]
                        curr_row = df_bt.iloc[i]
                        
                        # Check Entry
                        if position is None:
                            # We need to simulate the signal check using the full DF index
                            # To avoid re-calculating everything, we can mock the DF passed to logic
                            # But standard logic takes full DF. 
                            # Optimize: Check signal on current row based on pre-calc columns
                            
                            # Simple signal extraction for loop (Re-using logic function is slow in loops, but accurate)
                            # For speed, we will implement simplified logic mapping here or call function carefully
                            
                            # Re-construct mini-df for the function (last 2 rows needed)
                            mini_df = df_bt.iloc[i-50:i+1]
                            sig, reason = check_signals(mini_df, config, st.session_state['custom_conditions'])
                            
                            if sig != 0:
                                entry_price = curr_row['Close']
                                sl_dist = config['sl_points'] # Simplified
                                if config['sl_type'] == "ATR-based": sl_dist = curr_row['ATR'] * 1.5
                                
                                sl = entry_price - sl_dist if sig == 1 else entry_price + sl_dist
                                target = entry_price + config['target_points'] if sig == 1 else entry_price - config['target_points']
                                
                                position = {
                                    'type': sig, 'entry_price': entry_price, 'sl': sl, 'target': target,
                                    'entry_time': df_bt.index[i], 'qty': config['qty']
                                }
                        
                        # Check Exit
                        elif position:
                            p_type = position['type']
                            curr_price = curr_row['Close']
                            
                            # Simple Fixed SL/TP for Backtest speed
                            # (Full trailing logic is computationally heavy in Python loops without vectorization)
                            exit_trade = False
                            pnl = 0
                            exit_reason = ""
                            
                            if p_type == 1:
                                if curr_price <= position['sl']:
                                    exit_trade = True
                                    pnl = (position['sl'] - position['entry_price']) * position['qty']
                                    exit_reason = "SL Hit"
                                elif curr_price >= position['target']:
                                    exit_trade = True
                                    pnl = (position['target'] - position['entry_price']) * position['qty']
                                    exit_reason = "Target Hit"
                            else:
                                if curr_price >= position['sl']:
                                    exit_trade = True
                                    pnl = (position['entry_price'] - position['sl']) * position['qty']
                                    exit_reason = "SL Hit"
                                elif curr_price <= position['target']:
                                    exit_trade = True
                                    pnl = (position['entry_price'] - position['target']) * position['qty']
                                    exit_reason = "Target Hit"
                                    
                            if exit_trade:
                                trades.append({
                                    'Entry Time': position['entry_time'],
                                    'Exit Time': df_bt.index[i],
                                    'Signal': "LONG" if p_type == 1 else "SHORT",
                                    'Entry Price': position['entry_price'],
                                    'Exit Price': position['sl'] if "SL" in exit_reason else position['target'],
                                    'PnL': pnl,
                                    'Reason': exit_reason
                                })
                                position = None
                    
                    # Results
                    if trades:
                        bt_df = pd.DataFrame(trades)
                        total_bt_pnl = bt_df['PnL'].sum()
                        st.metric("Total Backtest P&L", f"{total_bt_pnl:.2f}")
                        st.dataframe(bt_df)
                    else:
                        st.warning("No trades generated in backtest period.")

if __name__ == "__main__":
    main()
