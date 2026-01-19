import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
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
    """Calculate Bollinger Bands"""
    sma = calculate_sma(data, period)
    std_dev = data.rolling(window=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper, sma, lower

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    histogram = macd - signal_line
    
    return macd, signal_line, histogram

def calculate_stochastic_rsi(data, period=14, smooth_k=3, smooth_d=3):
    """Calculate Stochastic RSI"""
    rsi = calculate_rsi(data, period)
    
    stoch = (rsi - rsi.rolling(window=period).min()) / (rsi.rolling(window=period).max() - rsi.rolling(window=period).min()) * 100
    k = stoch.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    
    return k, d

def calculate_keltner_channel(df, period=20, atr_mult=2):
    """Calculate Keltner Channel"""
    ema = calculate_ema(df['Close'], period)
    atr = calculate_atr(df, period)
    
    upper = ema + (atr * atr_mult)
    lower = ema - (atr * atr_mult)
    
    return upper, ema, lower

def calculate_vwap(df):
    """Calculate VWAP - handle missing volume"""
    if 'Volume' not in df.columns or df['Volume'].sum() == 0:
        return pd.Series(np.nan, index=df.index)
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    return vwap

def calculate_supertrend(df, period=10, multiplier=3):
    """Calculate SuperTrend"""
    atr = calculate_atr(df, period)
    hl_avg = (df['High'] + df['Low']) / 2
    
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)
    
    for i in range(period, len(df)):
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
    
    return supertrend, direction

def calculate_swing_levels(df, lookback=5):
    """Calculate Swing High and Low"""
    swing_high = df['High'].rolling(window=lookback*2+1, center=True).max()
    swing_low = df['Low'].rolling(window=lookback*2+1, center=True).min()
    
    return swing_high, swing_low

def calculate_support_resistance(df, lookback=20):
    """Calculate Support and Resistance levels"""
    resistance = df['High'].rolling(window=lookback).max()
    support = df['Low'].rolling(window=lookback).min()
    
    return support, resistance

def calculate_ema_angle(ema_series, lookback=2):
    """Calculate EMA angle in degrees"""
    if len(ema_series) < lookback + 1:
        return pd.Series(0, index=ema_series.index)
    
    slope = ema_series.diff(lookback) / lookback
    angle = np.degrees(np.arctan(slope))
    
    return angle

# ==================== DATA FETCHING ====================

def fetch_data(ticker, interval, period, mode="Backtesting"):
    """Fetch data from yfinance with proper error handling"""
    try:
        if mode == "Live Trading":
            time.sleep(random.uniform(1.0, 1.5))
        
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if data.empty:
            return             for i, cond in enumerate(st.session_state['custom_conditions']):
                with st.expander(f"Condition {i+1}", expanded=True):
                    cond['use_condition'] = st.checkbox("Use this condition", value=cond.get('use_condition', True), key=f"use_cond_{i}")
                    
                    cond['compare_with_indicator'] = st.checkbox("Compare with Indicator", value=cond.get('compare_with_indicator', False), key=f"compare_ind_{i}")
                    
                    indicator_options = ['Price', 'RSI', 'ADX', 'EMA_Fast', 'EMA_Slow', 'SuperTrend', 'EMA_20', 'EMA_50', 
                                       'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR', 'Volume', 'VWAP', 
                                       'Keltner_Upper', 'Keltner_Lower', 'Close', 'High', 'Low', 'Support', 'Resistance']
                    
                    cond['indicator'] = st.selectbox("Indicator", indicator_options, 
                                                    index=indicator_options.index(cond.get('indicator', 'RSI')), key=f"ind_{i}")
                    
                    operator_options = ['>', '<', '>=', '<=', '==', 'crosses_above', 'crosses_below']
                    cond['operator'] = st.selectbox("Operator", operator_options, 
                                                   index=operator_options.index(cond.get('operator', '>')), key=f"op_{i}")
                    
                    if cond['compare_with_indicator']:
                        cond['compare_indicator'] = st.selectbox("Compare Indicator", indicator_options, 
                                                                index=indicator_options.index(cond.get('compare_indicator', 'EMA_20')), 
                                                                key=f"comp_ind_{i}")
                    else:
                        cond['value'] = st.number_input("Value", value=float(cond.get('value', 50)), key=f"val_{i}")
                    
                    cond['action'] = st.selectbox("Action", ['BUY', 'SELL'], 
                                                 index=0 if cond.get('action', 'BUY') == 'BUY' else 1, key=f"act_{i}")
                    
                    if st.button("Remove Condition", key=f"remove_cond_{i}"):
                        st.session_state['custom_conditions'].pop(i)
                        st.rerun()
            
            config['custom_conditions'] = st.session_state['custom_conditions']
            config['ema_fast'] = 9
            config['ema_slow'] = 15
        
        # Default EMA values for other strategies
        else:
            config['ema_fast'] = 9
            config['ema_slow'] = 15
        
        st.markdown("---")
        
        # Stop Loss Configuration
        st.subheader("Stop Loss Configuration")
        config['sl_type'] = st.selectbox("Stop Loss Type", SL_TYPES, key="sl_type_select")
        
        if 'Custom Points' in config['sl_type'] or 'Trailing' in config['sl_type']:
            config['sl_points'] = st.number_input("SL Points", min_value=1.0, value=10.0, key="sl_points")
        
        if 'Trailing SL (Points)' in config['sl_type']:
            config['trailing_sl_points'] = st.number_input("Trailing SL Points", min_value=1.0, value=10.0, key="trail_sl_pts")
        
        if 'ATR' in config['sl_type'] or 'Volatility' in config['sl_type']:
            config['atr_sl_multiplier'] = st.number_input("ATR Multiplier (SL)", min_value=0.1, value=1.5, step=0.1, key="atr_sl_mult")
        
        # Target Configuration
        st.subheader("Target Configuration")
        config['target_type'] = st.selectbox("Target Type", TARGET_TYPES, key="target_type_select")
        
        if 'Custom Points' in config['target_type'] or 'Trailing Target' in config['target_type'] or 'Partial' in config['target_type']:
            config['target_points'] = st.number_input("Target Points", min_value=1.0, value=20.0, key="target_points")
        
        if 'Trailing Target (Points)' in config['target_type']:
            config['trailing_target_points'] = st.number_input("Trailing Target Points", min_value=1.0, value=20.0, key="trail_tgt_pts")
        
        if 'ATR-based' in config['target_type']:
            config['atr_target_multiplier'] = st.number_input("ATR Multiplier (Target)", min_value=0.1, value=3.0, step=0.1, key="atr_tgt_mult")
        
        if 'Risk-Reward' in config['target_type']:
            config['risk_reward_ratio'] = st.number_input("Risk-Reward Ratio", min_value=0.1, value=2.0, step=0.1, key="rr_ratio")
        
        st.markdown("---")
        
        # Dhan API Configuration (Optional)
        with st.expander("🔌 Dhan API Settings (Optional)", expanded=False):
            st.info("Configure Dhan API for live order placement")
            dhan_client_id = st.text_input("Client ID", type="password", key="dhan_client_id")
            dhan_access_token = st.text_input("Access Token", type="password", key="dhan_access_token")
            
            st.subheader("Options Trading Settings")
            strike_price = st.number_input("Strike Price", min_value=0.0, value=0.0, key="strike_price")
            expiry_date = st.date_input("Expiry Date", key="expiry_date")
            option_type = st.selectbox("Option Type", ["CE", "PE"], key="option_type")
            profit_threshold = st.number_input("Profit Threshold (%)", min_value=0.0, value=10.0, key="profit_threshold")
            loss_threshold = st.number_input("Loss Threshold (%)", min_value=0.0, value=5.0, key="loss_threshold")
            
            st.code("""
# Dhan API Integration (Placeholder)
# from dhanhq import dhanhq

# def place_order(order_type, quantity, price):
#     dhan = dhanhq(client_id, access_token)
#     
#     order_params = {
#         'exchange_segment': 'NSE_FNO',
#         'transaction_type': 'BUY' if order_type == 'LONG' else 'SELL',
#         'product_type': 'INTRADAY',
#         'order_type': 'LIMIT',
#         'quantity': quantity,
#         'price': price,
#         'strike_price': strike_price,
#         'expiry_date': expiry_date,
#         'option_type': option_type
#     }
#     
#     response = dhan.place_order(order_params)
#     return response
            """, language="python")
        
        st.session_state['config'] = config
    
    # Main Content
    if mode == "Live Trading":
        render_live_trading(config)
    else:
        render_backtesting(config)

def render_live_trading(config):
    """Render Live Trading Interface"""
    tab1, tab2, tab3 = st.tabs(["📊 Live Trading Dashboard", "📈 Trade History", "📝 Trade Logs"])
    
    with tab1:
        # Trading Controls
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            if st.button("▶️ Start Trading", key="start_trading", use_container_width=True):
                st.session_state['trading_active'] = True
                st.session_state['trade_logs'].append({
                    'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')),
                    'message': 'Trading Started'
                })
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop Trading", key="stop_trading", use_container_width=True):
                # Close position if active
                if st.session_state.get('position'):
                    position = st.session_state['position']
                    current_price = st.session_state['current_data']['Close'].iloc[-1] if st.session_state.get('current_data') is not None else position['entry_price']
                    
                    if position['type'] == 'LONG':
                        pnl = (current_price - position['entry_price']) * position['quantity']
                    else:
                        pnl = (position['entry_price'] - current_price) * position['quantity']
                    
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
                    
                    st.session_state['trade_logs'].append({
                        'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')),
                        'message': f"Position Manually Closed - {position['type']} @ {current_price:.2f}, P&L: {pnl:.2f}"
                    })
                
                st.session_state['trading_active'] = False
                st.session_state['position'] = None
                st.session_state['highest_price'] = 0
                st.session_state['lowest_price'] = float('inf')
                st.session_state['partial_exit_done'] = False
                
                st.session_state['trade_logs'].append({
                    'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')),
                    'message': 'Trading Stopped'
                })
                st.rerun()
        
        with col3:
            status = "🟢 ACTIVE" if st.session_state.get('trading_active', False) else "🔴 INACTIVE"
            st.markdown(f"### Status: {status}")
        
        st.markdown("---")
        
        # Active Configuration Display
        st.subheader("🔧 Active Configuration")
        
        config_col = st.columns(1)[0]
        with config_col:
            st.markdown(f"""
            **Asset:** {config['ticker']}  
            **Interval:** {config['interval']}  
            **Period:** {config['period']}  
            **Quantity:** {config['quantity']}  
            **Strategy:** {config['strategy']}  
            **Stop Loss Type:** {config['sl_type']}  
            **Target Type:** {config['target_type']}  
            """)
            
            if config['strategy'] == "EMA Crossover Strategy":
                st.markdown(f"""
                **EMA Fast:** {config.get('ema_fast', 9)}  
                **EMA Slow:** {config.get('ema_slow', 15)}  
                **Min Angle:** {config.get('min_angle', 1)}°  
                **Entry Filter:** {config.get('entry_filter', 'Simple Crossover')}  
                **Use ADX:** {config.get('use_adx', False)}  
                """)
        
        st.markdown("---")
        
        # Live Metrics Display
        if st.session_state.get('current_data') is not None:
            df = st.session_state['current_data']
            current_price = df['Close'].iloc[-1]
            position = st.session_state.get('position')
            
            st.subheader("📊 Live Metrics")
            
            metrics_col = st.columns(1)[0]
            with metrics_col:
                st.metric("Current Price", f"₹{current_price:.2f}")
                
                if position:
                    st.metric("Entry Price", f"₹{position['entry_price']:.2f}")
                    st.metric("Position Status", "ACTIVE", delta=None)
                    st.metric("Position Type", position['type'])
                    
                    # Calculate unrealized P&L
                    if position['type'] == 'LONG':
                        unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
                    else:
                        unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']
                    
                    pnl_color = "normal" if unrealized_pnl >= 0 else "inverse"
                    st.metric("Unrealized P&L", f"₹{unrealized_pnl:.2f}", delta=f"{unrealized_pnl:.2f}", delta_color=pnl_color)
                    
                    # Display indicators if available
                    if 'EMA_Fast' in df.columns:
                        st.metric("EMA Fast", f"₹{df['EMA_Fast'].iloc[-1]:.2f}")
                    if 'EMA_Slow' in df.columns:
                        st.metric("EMA Slow", f"₹{df['EMA_Slow'].iloc[-1]:.2f}")
                    if 'RSI' in df.columns:
                        st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
                    
                    # Position Information
                    st.markdown("---")
                    st.markdown("### Position Information")
                    
                    duration = datetime.now(pytz.timezone('Asia/Kolkata')) - position['entry_time']
                    st.metric("Entry Time", position['entry_time'].strftime("%Y-%m-%d %H:%M:%S"))
                    st.metric("Duration", str(duration).split('.')[0])
                    
                    if st.session_state.get('current_sl'):
                        st.metric("Stop Loss", f"₹{st.session_state['current_sl']:.2f}")
                        sl_distance = abs(current_price - st.session_state['current_sl'])
                        st.metric("Distance to SL", f"₹{sl_distance:.2f}")
                    
                    if st.session_state.get('current_target'):
                        st.metric("Target", f"₹{st.session_state['current_target']:.2f}")
                        target_distance = abs(st.session_state['current_target'] - current_price)
                        st.metric("Distance to Target", f"₹{target_distance:.2f}")
                    
                    st.metric("Highest Price", f"₹{st.session_state.get('highest_price', 0):.2f}")
                    st.metric("Lowest Price", f"₹{st.session_state.get('lowest_price', float('inf')):.2f}")
                    
                    price_range = st.session_state.get('highest_price', 0) - st.session_state.get('lowest_price', float('inf'))
                    if price_range != float('inf'):
                        st.metric("Range", f"₹{abs(price_range):.2f}")
                else:
                    st.metric("Position Status", "NO POSITION")
                    
                    # Display signal status
                    if 'EMA_Fast' in df.columns:
                        st.metric("EMA Fast", f"₹{df['EMA_Fast'].iloc[-1]:.2f}")
                    if 'EMA_Slow' in df.columns:
                        st.metric("EMA Slow", f"₹{df['EMA_Slow'].iloc[-1]:.2f}")
                    if 'RSI' in df.columns:
                        st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
            
            st.markdown("---")
            
            # Live Chart
            st.subheader("📈 Live Chart")
            chart_key = f"live_chart_{datetime.now().timestamp()}"
            fig = create_candlestick_chart(df, live_position=position)
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
        
        else:
            st.info("Waiting for data... Start trading to begin.")
    
    with tab2:
        st.subheader("📈 Trade History")
        
        if st.session_state.get('trade_history'):
            trades = st.session_state['trade_history']
            
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            losing_trades = len([t for t in trades if t['pnl'] < 0])
            accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = sum([t['pnl'] for t in trades])
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Trades", total_trades)
            col2.metric("Winning Trades", winning_trades)
            col3.metric("Losing Trades", losing_trades)
            col4.metric("Accuracy", f"{accuracy:.2f}%")
            
            pnl_color = "normal" if total_pnl >= 0 else "inverse"
            col5.metric("Total P&L", f"₹{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color=pnl_color)
            
            st.markdown("---")
            
            for i, trade in enumerate(reversed(trades)):
                with st.expander(f"Trade {total_trades - i} - {trade['type']} - P&L: ₹{trade['pnl']:.2f}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Entry Time:** {trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(trade['entry_time'], 'strftime') else trade['entry_time']}")
                        st.write(f"**Entry Price:** ₹{trade['entry_price']:.2f}")
                        st.write(f"**Quantity:** {trade['quantity']}")
                        if trade.get('sl_price'):
                            st.write(f"**Stop Loss:** ₹{trade['sl_price']:.2f}")
                        else:
                            st.write(f"**Stop Loss:** N/A")
                    
                    with col2:
                        st.write(f"**Exit Time:** {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(trade['exit_time'], 'strftime') else trade['exit_time']}")
                        st.write(f"**Exit Price:** ₹{trade['exit_price']:.2f}")
                        st.write(f"**Exit Reason:** {trade['exit_reason']}")
                        if trade.get('target_price'):
                            st.write(f"**Target:** ₹{trade['target_price']:.2f}")
                        else:
                            st.write(f"**Target:** N/A")
                    
                    pnl_display = f"₹{trade['pnl']:.2f}"
                    if trade['pnl'] > 0:
                        st.success(f"**P&L:** {pnl_display}")
                    else:
                        st.error(f"**P&L:** {pnl_display}")
        else:
            st.info("No trades executed yet.")
    
    with tab3:
        st.subheader("📝 Trade Logs")
        
        if st.session_state.get('trade_logs'):
            logs = st.session_state['trade_logs'][-50:]  # Last 50 entries
            
            for log in reversed(logs):
                timestamp = log['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(log['timestamp'], 'strftime') else log['timestamp']
                st.text(f"[{timestamp}] {log['message']}")
        else:
            st.info("No logs available.")
    
    # Auto-refresh live trading
    if st.session_state.get('trading_active', False):
        live_trading_iteration()

def render_backtesting(config):
    """Render Backtesting Interface"""
    tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "📊 Backtest Results", "📈 Market Data Analysis"])
    
    with tab1:
        st.subheader("Current Configuration")
        
        st.markdown(f"""
        **Asset:** {config['ticker']}  
        **Interval:** {config['interval']}  
        **Period:** {config['period']}  
        **Quantity:** {config['quantity']}  
        **Strategy:** {config['strategy']}  
        **Stop Loss Type:** {config['sl_type']}  
        **Target Type:** {config['target_type']}  
        """)
        
        if config['strategy'] == "EMA Crossover Strategy":
            st.markdown(f"""
            **EMA Fast:** {config.get('ema_fast', 9)}  
            **EMA Slow:** {config.get('ema_slow', 15)}  
            **Min Angle:** {config.get('min_angle', 1)}°  
            **Entry Filter:** {config.get('entry_filter', 'Simple Crossover')}  
            **Use ADX:** {config.get('use_adx', False)}  
            """)
    
    with tab2:
        st.subheader("Backtest Results")
        
        if st.button("🚀 Run Backtest", key="run_backtest", use_container_width=True):
            with st.spinner("Running backtest..."):
                df = fetch_data(config['ticker'], config['interval'], config['period'], mode="Backtesting")
                
                if df is not None and not df.empty:
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
                    
                    strategy_func = strategy_map.get(config['strategy'], ema_crossover_strategy)
                    
                    trades, df_with_signals = run_backtest(df, strategy_func, config)
                    
                    st.session_state['backtest_trades'] = trades
                    st.session_state['backtest_df'] = df_with_signals
                    
                    st.success("Backtest completed!")
                else:
                    st.error("Failed to fetch data.")
        
        if st.session_state.get('backtest_trades'):
            trades = st.session_state['backtest_trades']
            
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            losing_trades = len([t for t in trades if t['pnl'] < 0])
            accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = sum([t['pnl'] for t in trades])
            
            # Calculate average duration
            durations = [(t['exit_time'] - t['entry_time']).total_seconds() / 3600 for t in trades]
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Trades", total_trades)
            col2.metric("Win Rate", f"{accuracy:.2f}%")
            col3.metric("Avg Duration", f"{avg_duration:.2f}h")
            
            col4, col5, col6 = st.columns(3)
            col4.metric("Winning Trades", winning_trades)
            col5.metric("Losing Trades", losing_trades)
            
            pnl_color = "normal" if total_pnl >= 0 else "inverse"
            col6.metric("Total P&L", f"₹{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color=pnl_color)
            
            st.markdown("---")
            
            st.subheader("Trade List")
            
            for i, trade in enumerate(trades):
                with st.expander(f"Trade {i+1} - {trade['type']} - P&L: ₹{trade['pnl']:.2f}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Entry Time:** {trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"**Entry Price:** ₹{trade['entry_price']:.2f}")
                        st.write(f"**Quantity:** {trade['quantity']}")
                        if trade.get('sl_price'):
                            st.write(f"**Stop Loss:** ₹{trade['sl_price']:.2f}")
                        else:
                            st.write(f"**Stop Loss:** N/A")
                    
                    with col2:
                        st.write(f"**Exit Time:** {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"**Exit Price:** ₹{trade['exit_price']:.2f}")
                        st.write(f"**Exit Reason:** {trade['exit_reason']}")
                        if trade.get('target_price'):
                            st.write(f"**Target:** ₹{trade['target_price']:.2f}")
                        else:
                            st.write(f"**Target:** N/A")
                    
                    pnl_display = f"₹{trade['pnl']:.2f}"
                    if trade['pnl'] > 0:
                        st.success(f"**P&L:** {pnl_display}")
                    else:
                        st.error(f"**P&L:** {pnl_display}")
    
    with tab3:
        st.subheader("Market Data Analysis")
        
        df = fetch_data(config['ticker'], config['interval'], config['period'], mode="Backtesting")
        
        if df is not None and not df.empty:
            # Data Table
            st.markdown("### 📋 Data Table")
            
            display_df = df.copy()
            display_df['Change (Points)'] = display_df['Close'] - display_df['Open']
            display_df['Change (%)'] = ((display_df['Close'] - display_df['Open']) / display_df['Open']) * 100
            display_df['Day of Week'] = display_df.index.day_name()
            
            st.dataframe(
                display_df[['Open', 'High', 'Low', 'Close', 'Change (Points)', 'Change (%)', 'Day of Week']],
                use_container_width=True
            )
            
            st.markdown("---")
            
            # Charts
            st.markdown("### 📊 Change Analysis")
            
            # Change in Points
            fig_points = go.Figure()
            colors = ['green' if x > 0 else 'red' for x in display_df['Change (Points)']]
            fig_points.add_trace(go.Bar(
                x=display_df.index,
                y=display_df['Change (Points)'],
                marker_color=colors,
                name='Change (Points)'
            ))
            fig_points.update_layout(title='Change in Points Over Time', xaxis_title='Time', yaxis_title='Points', height=400)
            st.plotly_chart(fig_points, use_container_width=True)
            
            # Change in Percentage
            fig_pct = go.Figure()
            colors_pct = ['green' if x > 0 else 'red' for x in display_df['Change (%)']]
            fig_pct.add_trace(go.Bar(
                x=display_df.index,
                y=display_df['Change (%)'],
                marker_color=colors_pct,
                name='Change (%)'
            ))
            fig_pct.update_layout(title='Change in Percentage Over Time', xaxis_title='Time', yaxis_title='Percentage', height=400)
            st.plotly_chart(fig_pct, use_container_width=True)
            
            st.markdown("---")
            
            # Summary Statistics
            st.markdown("### 📈 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Max Price", f"₹{display_df['High'].max():.2f}")
            col2.metric("Min Price", f"₹{display_df['Low'].min():.2f}")
            col3.metric("Avg Price", f"₹{display_df['Close'].mean():.2f}")
            col4.metric("Volatility", f"{display_df['Close'].std():.2f}")
            
            col5, col6, col7, col8 = st.columns(4)
            total_change = display_df['Close'].iloc[-1] - display_df['Close'].iloc[0]
            total_change_pct = (total_change / display_df['Close'].iloc[0]) * 100
            col5.metric("Total Change (Points)", f"₹{total_change:.2f}")
            col6.metric("Total Change (%)", f"{total_change_pct:.2f}%")
            
            avg_change = display_df['Change (Points)'].mean()
            col7.metric("Avg Change (Points)", f"₹{avg_change:.2f}")
            
            positive_days = len(display_df[display_df['Change (Points)'] > 0])
            win_rate = (positive_days / len(display_df)) * 100
            col8.metric("Win Rate", f"{win_rate:.2f}%")
            
            col9, col10, col11, col12 = st.columns(4)
            max_gain = display_df['Change (Points)'].max()
            max_loss = display_df['Change (Points)'].min()
            col9.metric("Max Gain (Points)", f"₹{max_gain:.2f}")
            col10.metric("Max Loss (Points)", f"₹{max_loss:.2f}")
            
            max_gain_pct = display_df['Change (%)'].max()
            max_loss_pct = display_df['Change (%)'].min()
            col11.metric("Max Gain (%)", f"{max_gain_pct:.2f}%")
            col12.metric("Max Loss (%)", f"{max_loss_pct:.2f}%")
            
            st.markdown("---")
            
            # 10-Year Heatmaps
            st.markdown("### 🔥 10-Year Historical Analysis")
            
            try:
                # Fetch 10-year daily data
                df_10y = fetch_data(config['ticker'], '1d', '10y', mode="Backtesting")
                
                if df_10y is not None and not df_10y.empty:
                    # Monthly Returns Heatmap
                    df_10y['Year'] = df_10y.index.year
                    df_10y['Month'] = df_10y.index.month
                    df_10y['Returns'] = df_10y['Close'].pct_change() * 100
                    
                    monthly_returns = df_10y.groupby(['Year', 'Month'])['Returns'].sum().unstack(fill_value=0)
                    
                    # Reorder columns to show Jan-Dec
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    monthly_returns.columns = [month_names[int(m)-1] if m <= 12 else m for m in monthly_returns.columns]
                    
                    fig_returns_heatmap = go.Figure(data=go.Heatmap(
                        z=monthly_returns.values,
                        x=monthly_returns.columns,
                        y=monthly_returns.index,
                        colorscale='RdYlGn',
                        text=monthly_returns.values,
                        texttemplate='%{text:.2f}%',
                        textfont={"size": 10},
                        colorbar=dict(title="Returns %")
                    ))
                    fig_returns_heatmap.update_layout(
                        title='Monthly Returns Heatmap (10 Years)',
                        xaxis_title='Month',
                        yaxis_title='Year',
                        height=500
                    )
                    st.plotly_chart(fig_returns_heatmap, use_container_width=True)
                    
                    # Monthly Volatility Heatmap
                    monthly_volatility = df_10y.groupby(['Year', 'Month'])['Returns'].std().unstack(fill_value=0)
                    monthly_volatility.columns = [month_names[int(m)-1] if m <= 12 else m for m in monthly_volatility.columns]
                    
                    fig_vol_heatmap = go.Figure(data=go.Heatmap(
                        z=monthly_volatility.values,
                        x=monthly_volatility.columns,
                        y=monthly_volatility.index,
                        colorscale='Reds',
                        text=monthly_volatility.values,
                        texttemplate='%{text:.2f}%',
                        textfont={"size": 10},
                        colorbar=dict(title="Volatility %")
                    ))
                    fig_vol_heatmap.update_layout(
                        title='Monthly Volatility Heatmap (10 Years)',
                        xaxis_title='Month',
                        yaxis_title='Year',
                        height=500
                    )
                    st.plotly_chart(fig_vol_heatmap, use_container_width=True)
                else:
                    st.warning("Unable to fetch 10-year data for heatmap analysis.")
            
            except Exception as e:
                st.error(f"Error generating heatmaps: {str(e)}")
        
        else:
            st.error("Failed to fetch market data.")

if __name__ == "__main__":
    main()
        
        # Flatten multi-index if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Select OHLCV columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in required_cols if col in data.columns]
        data = data[available_cols].copy()
        
        # Handle timezone
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
    """EMA Crossover Strategy with all filters"""
    ema_fast = calculate_ema(df['Close'], config['ema_fast'])
    ema_slow = calculate_ema(df['Close'], config['ema_slow'])
    
    df['EMA_Fast'] = ema_fast
    df['EMA_Slow'] = ema_slow
    
    # Calculate EMA angle
    ema_angle = calculate_ema_angle(ema_fast, lookback=2)
    
    # Crossover detection
    cross_above = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
    cross_below = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
    
    # Angle filter
    angle_filter = abs(ema_angle) >= config['min_angle']
    
    # Entry filter
    entry_filter = pd.Series(True, index=df.index)
    
    if config['entry_filter'] == 'Custom Candle (Points)':
        candle_size = abs(df['Close'] - df['Open'])
        entry_filter = candle_size >= config['custom_points']
    
    elif config['entry_filter'] == 'ATR-based Candle':
        atr = calculate_atr(df, 14)
        candle_size = abs(df['Close'] - df['Open'])
        entry_filter = candle_size >= (atr * config['atr_multiplier'])
    
    # ADX filter
    adx_filter = pd.Series(True, index=df.index)
    if config['use_adx']:
        adx = calculate_adx(df, config['adx_period'])
        df['ADX'] = adx
        adx_filter = adx >= config['adx_threshold']
    
    # Combine all filters
    buy_signal = cross_above & angle_filter & entry_filter & adx_filter
    sell_signal = cross_below & angle_filter & entry_filter & adx_filter
    
    return buy_signal, sell_signal, df

def simple_buy_strategy(df, config):
    """Simple Buy Strategy"""
    buy_signal = pd.Series(False, index=df.index)
    buy_signal.iloc[-1] = True
    sell_signal = pd.Series(False, index=df.index)
    
    return buy_signal, sell_signal, df

def simple_sell_strategy(df, config):
    """Simple Sell Strategy"""
    buy_signal = pd.Series(False, index=df.index)
    sell_signal = pd.Series(False, index=df.index)
    sell_signal.iloc[-1] = True
    
    return buy_signal, sell_signal, df

def price_crosses_threshold_strategy(df, config):
    """Price Crosses Threshold Strategy"""
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
    """RSI-ADX-EMA Strategy"""
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
    """Percentage Change Strategy"""
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
    """AI Price Action Analysis Strategy"""
    # Calculate indicators
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
    
    # AI Analysis
    buy_score = 0
    sell_score = 0
    
    # Trend analysis
    if ema_20.iloc[-1] > ema_50.iloc[-1]:
        buy_score += 2
    else:
        sell_score += 2
    
    # RSI analysis
    if rsi.iloc[-1] < 30:
        buy_score += 2
    elif rsi.iloc[-1] > 70:
        sell_score += 2
    
    # MACD analysis
    if macd.iloc[-1] > macd_signal.iloc[-1]:
        buy_score += 1
    else:
        sell_score += 1
    
    # Bollinger Bands
    if df['Close'].iloc[-1] < bb_lower.iloc[-1]:
        buy_score += 1
    elif df['Close'].iloc[-1] > bb_upper.iloc[-1]:
        sell_score += 1
    
    # Volume analysis (if available)
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        vol_sma = df['Volume'].rolling(20).mean()
        if df['Volume'].iloc[-1] > vol_sma.iloc[-1]:
            if buy_score > sell_score:
                buy_score += 1
            else:
                sell_score += 1
    
    # Generate signals
    buy_signal = pd.Series(False, index=df.index)
    sell_signal = pd.Series(False, index=df.index)
    
    if buy_score > sell_score and buy_score >= 4:
        buy_signal.iloc[-1] = True
    elif sell_score > buy_score and sell_score >= 4:
        sell_signal.iloc[-1] = True
    
    # Store AI analysis
    df['AI_Buy_Score'] = buy_score
    df['AI_Sell_Score'] = sell_score
    
    return buy_signal, sell_signal, df

def custom_strategy_builder(df, config):
    """Custom Strategy Builder"""
    conditions = config.get('custom_conditions', [])
    
    # Calculate all possible indicators
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
    
    supertrend, _ = calculate_supertrend(df)
    df['SuperTrend'] = supertrend
    
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
        
        # Check if comparing price with indicator
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
    
    # Combine conditions
    if buy_conditions:
        buy_signal = pd.concat(buy_conditions, axis=1).all(axis=1)
    else:
        buy_signal = pd.Series(False, index=df.index)
    
    if sell_conditions:
        sell_signal = pd.concat(sell_conditions, axis=1).all(axis=1)
    else:
        sell_signal = pd.Series(False, index=df.index)
    
    return buy_signal, sell_signal, df

# ==================== STOP LOSS & TARGET LOGIC ====================

def calculate_sl_price(df, idx, position_type, entry_price, sl_type, config, position_state):
    """Calculate stop loss price"""
    if sl_type == "Custom Points":
        sl_points = config.get('sl_points', 10)
        if position_type == 'LONG':
            return entry_price - sl_points
        else:
            return entry_price + sl_points
    
    elif sl_type == "Trailing SL (Points)":
        trailing_offset = config.get('trailing_sl_points', 10)
        if position_type == 'LONG':
            highest = position_state.get('highest_price', entry_price)
            return highest - trailing_offset
        else:
            lowest = position_state.get('lowest_price', entry_price)
            return lowest + trailing_offset
    
    elif sl_type == "Trailing SL + Current Candle":
        if position_type == 'LONG':
            return df['Low'].iloc[idx]
        else:
            return df['High'].iloc[idx]
    
    elif sl_type == "Trailing SL + Previous Candle":
        if idx > 0:
            if position_type == 'LONG':
                return df['Low'].iloc[idx-1]
            else:
                return df['High'].iloc[idx-1]
        return entry_price - 10 if position_type == 'LONG' else entry_price + 10
    
    elif sl_type == "Trailing SL + Current Swing":
        swing_high, swing_low = calculate_swing_levels(df[:idx+1])
        if position_type == 'LONG':
            return swing_low.iloc[-1] if not pd.isna(swing_low.iloc[-1]) else entry_price - 10
        else:
            return swing_high.iloc[-1] if not pd.isna(swing_high.iloc[-1]) else entry_price + 10
    
    elif sl_type == "Trailing SL + Previous Swing":
        swing_high, swing_low = calculate_swing_levels(df[:idx+1])
        if len(swing_low) > 1:
            if position_type == 'LONG':
                return swing_low.iloc[-2] if not pd.isna(swing_low.iloc[-2]) else entry_price - 10
            else:
                return swing_high.iloc[-2] if not pd.isna(swing_high.iloc[-2]) else entry_price + 10
        return entry_price - 10 if position_type == 'LONG' else entry_price + 10
    
    elif sl_type == "Trailing SL + Signal Based":
        # Check for reverse crossover
        if 'EMA_Fast' in df.columns and 'EMA_Slow' in df.columns:
            if position_type == 'LONG':
                if df['EMA_Fast'].iloc[idx] < df['EMA_Slow'].iloc[idx]:
                    return df['Close'].iloc[idx]
            else:
                if df['EMA_Fast'].iloc[idx] > df['EMA_Slow'].iloc[idx]:
                    return df['Close'].iloc[idx]
        
        # Otherwise use trailing
        trailing_offset = config.get('trailing_sl_points', 10)
        if position_type == 'LONG':
            highest = position_state.get('highest_price', entry_price)
            return highest - trailing_offset
        else:
            lowest = position_state.get('lowest_price', entry_price)
            return lowest + trailing_offset
    
    elif sl_type == "Volatility-Adjusted Trailing SL":
        atr = calculate_atr(df[:idx+1], 14).iloc[-1]
        atr_mult = config.get('atr_sl_multiplier', 1.5)
        if position_type == 'LONG':
            highest = position_state.get('highest_price', entry_price)
            return highest - (atr * atr_mult)
        else:
            lowest = position_state.get('lowest_price', entry_price)
            return lowest + (atr * atr_mult)
    
    elif sl_type == "Break-even After 50% Target":
        target_price = calculate_target_price(df, idx, position_type, entry_price, config.get('target_type', 'Custom Points'), config)
        if target_price:
            mid_point = (entry_price + target_price) / 2
            current_price = df['Close'].iloc[idx]
            
            if position_type == 'LONG' and current_price >= mid_point:
                return entry_price
            elif position_type == 'SHORT' and current_price <= mid_point:
                return entry_price
        
        sl_points = config.get('sl_points', 10)
        if position_type == 'LONG':
            return entry_price - sl_points
        else:
            return entry_price + sl_points
    
    elif sl_type == "ATR-based":
        atr = calculate_atr(df[:idx+1], 14).iloc[-1]
        atr_mult = config.get('atr_sl_multiplier', 1.5)
        if position_type == 'LONG':
            return entry_price - (atr * atr_mult)
        else:
            return entry_price + (atr * atr_mult)
    
    elif sl_type == "Current Candle Low/High":
        if position_type == 'LONG':
            return df['Low'].iloc[idx]
        else:
            return df['High'].iloc[idx]
    
    elif sl_type == "Previous Candle Low/High":
        if idx > 0:
            if position_type == 'LONG':
                return df['Low'].iloc[idx-1]
            else:
                return df['High'].iloc[idx-1]
        return entry_price - 10 if position_type == 'LONG' else entry_price + 10
    
    elif sl_type == "Current Swing Low/High":
        swing_high, swing_low = calculate_swing_levels(df[:idx+1])
        if position_type == 'LONG':
            return swing_low.iloc[-1] if not pd.isna(swing_low.iloc[-1]) else entry_price - 10
        else:
            return swing_high.iloc[-1] if not pd.isna(swing_high.iloc[-1]) else entry_price + 10
    
    elif sl_type == "Previous Swing Low/High":
        swing_high, swing_low = calculate_swing_levels(df[:idx+1])
        if len(swing_low) > 1:
            if position_type == 'LONG':
                return swing_low.iloc[-2] if not pd.isna(swing_low.iloc[-2]) else entry_price - 10
            else:
                return swing_high.iloc[-2] if not pd.isna(swing_high.iloc[-2]) else entry_price + 10
        return entry_price - 10 if position_type == 'LONG' else entry_price + 10
    
    elif sl_type == "Signal-based (reverse EMA crossover)":
        # Check for reverse crossover every candle
        if 'EMA_Fast' in df.columns and 'EMA_Slow' in df.columns:
            if position_type == 'LONG':
                # Exit on bearish crossover
                if idx > 0 and df['EMA_Fast'].iloc[idx] < df['EMA_Slow'].iloc[idx] and df['EMA_Fast'].iloc[idx-1] >= df['EMA_Slow'].iloc[idx-1]:
                    return df['Close'].iloc[idx]
            else:
                # Exit on bullish crossover
                if idx > 0 and df['EMA_Fast'].iloc[idx] > df['EMA_Slow'].iloc[idx] and df['EMA_Fast'].iloc[idx-1] <= df['EMA_Slow'].iloc[idx-1]:
                    return df['Close'].iloc[idx]
        return None
    
    return entry_price - 10 if position_type == 'LONG' else entry_price + 10

def calculate_target_price(df, idx, position_type, entry_price, target_type, config):
    """Calculate target price"""
    if target_type == "Custom Points":
        target_points = config.get('target_points', 20)
        if position_type == 'LONG':
            return entry_price + target_points
        else:
            return entry_price - target_points
    
    elif target_type == "Trailing Target (Points)":
        # Display only, never exits
        return None
    
    elif target_type == "Trailing Target + Signal Based":
        # Check for reverse crossover
        if 'EMA_Fast' in df.columns and 'EMA_Slow' in df.columns:
            if position_type == 'LONG':
                if idx > 0 and df['EMA_Fast'].iloc[idx] < df['EMA_Slow'].iloc[idx] and df['EMA_Fast'].iloc[idx-1] >= df['EMA_Slow'].iloc[idx-1]:
                    return df['Close'].iloc[idx]
            else:
                if idx > 0 and df['EMA_Fast'].iloc[idx] > df['EMA_Slow'].iloc[idx] and df['EMA_Fast'].iloc[idx-1] <= df['EMA_Slow'].iloc[idx-1]:
                    return df['Close'].iloc[idx]
        return None
    
    elif target_type == "50% Exit at Target (Partial)":
        target_points = config.get('target_points', 20)
        if position_type == 'LONG':
            return entry_price + target_points
        else:
            return entry_price - target_points
    
    elif target_type == "Current Candle Low/High":
        if position_type == 'LONG':
            return df['High'].iloc[idx]
        else:
            return df['Low'].iloc[idx]
    
    elif target_type == "Previous Candle Low/High":
        if idx > 0:
            if position_type == 'LONG':
                return df['High'].iloc[idx-1]
            else:
                return df['Low'].iloc[idx-1]
        return entry_price + 20 if position_type == 'LONG' else entry_price - 20
    
    elif target_type == "Current Swing Low/High":
        swing_high, swing_low = calculate_swing_levels(df[:idx+1])
        if position_type == 'LONG':
            return swing_high.iloc[-1] if not pd.isna(swing_high.iloc[-1]) else entry_price + 20
        else:
            return swing_low.iloc[-1] if not pd.isna(swing_low.iloc[-1]) else entry_price - 20
    
    elif target_type == "Previous Swing Low/High":
        swing_high, swing_low = calculate_swing_levels(df[:idx+1])
        if len(swing_high) > 1:
            if position_type == 'LONG':
                return swing_high.iloc[-2] if not pd.isna(swing_high.iloc[-2]) else entry_price + 20
            else:
                return swing_low.iloc[-2] if not pd.isna(swing_low.iloc[-2]) else entry_price - 20
        return entry_price + 20 if position_type == 'LONG' else entry_price - 20
    
    elif target_type == "ATR-based":
        atr = calculate_atr(df[:idx+1], 14).iloc[-1]
        atr_mult = config.get('atr_target_multiplier', 3.0)
        if position_type == 'LONG':
            return entry_price + (atr * atr_mult)
        else:
            return entry_price - (atr * atr_mult)
    
    elif target_type == "Risk-Reward Based":
        sl_price = calculate_sl_price(df, idx, position_type, entry_price, config.get('sl_type', 'Custom Points'), config, {})
        rr_ratio = config.get('risk_reward_ratio', 2.0)
        
        if sl_price:
            risk = abs(entry_price - sl_price)
            reward = risk * rr_ratio
            
            if position_type == 'LONG':
                return entry_price + reward
            else:
                return entry_price - reward
        
        return entry_price + 20 if position_type == 'LONG' else entry_price - 20
    
    elif target_type == "Signal-based (reverse EMA crossover)":
        # Check for reverse crossover every candle
        if 'EMA_Fast' in df.columns and 'EMA_Slow' in df.columns:
            if position_type == 'LONG':
                if idx > 0 and df['EMA_Fast'].iloc[idx] < df['EMA_Slow'].iloc[idx] and df['EMA_Fast'].iloc[idx-1] >= df['EMA_Slow'].iloc[idx-1]:
                    return df['Close'].iloc[idx]
            else:
                if idx > 0 and df['EMA_Fast'].iloc[idx] > df['EMA_Slow'].iloc[idx] and df['EMA_Fast'].iloc[idx-1] <= df['EMA_Slow'].iloc[idx-1]:
                    return df['Close'].iloc[idx]
        return None
    
    return entry_price + 20 if position_type == 'LONG' else entry_price - 20

# ==================== BACKTESTING ENGINE ====================

def run_backtest(df, strategy_func, config):
    """Run backtest on historical data"""
    buy_signals, sell_signals, df = strategy_func(df, config)
    
    trades = []
    position = None
    position_state = {}
    
    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        current_time = df.index[i]
        
        # Update position state
        if position:
            if position['type'] == 'LONG':
                position_state['highest_price'] = max(position_state.get('highest_price', position['entry_price']), current_price)
            else:
                position_state['lowest_price'] = min(position_state.get('lowest_price', position['entry_price']), current_price)
        
        # Check for exit if in position
        if position:
            sl_price = calculate_sl_price(df, i, position['type'], position['entry_price'], config['sl_type'], config, position_state)
            target_price = calculate_target_price(df, i, position['type'], position['entry_price'], config['target_type'], config)
            
            exit_reason = None
            exit_price = None
            
            # Check SL hit
            if sl_price is not None:
                if position['type'] == 'LONG' and current_price <= sl_price:
                    exit_reason = 'Stop Loss'
                    exit_price = sl_price
                elif position['type'] == 'SHORT' and current_price >= sl_price:
                    exit_reason = 'Stop Loss'
                    exit_price = sl_price
            
            # Check Target hit
            if not exit_reason and target_price is not None:
                if config['target_type'] == "50% Exit at Target (Partial)":
                    if not position_state.get('partial_exit_done', False):
                        if position['type'] == 'LONG' and current_price >= target_price:
                            # Partial exit
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
            
            # Execute exit
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
        
        # Check for new entry
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
    
    # Close any open position at end
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
    """Single iteration of live trading - called on each rerun"""
    if not st.session_state.get('trading_active', False):
        return
    
    config = st.session_state['config']
    ticker = config['ticker']
    interval = config['interval']
    period = config['period']
    
    # Fetch latest data
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
    
    # Run strategy
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
    
    # Update position state
    if position:
        if position['type'] == 'LONG':
            st.session_state['highest_price'] = max(st.session_state.get('highest_price', position['entry_price']), current_price)
        else:
            st.session_state['lowest_price'] = min(st.session_state.get('lowest_price', position['entry_price']), current_price)
    
    # Check for exit
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
        
        # Check SL
        if sl_price is not None:
            if position['type'] == 'LONG' and current_price <= sl_price:
                exit_reason = 'Stop Loss'
                exit_price = sl_price
            elif position['type'] == 'SHORT' and current_price >= sl_price:
                exit_reason = 'Stop Loss'
                exit_price = sl_price
        
        # Check Target
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
        
        # Execute exit
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
    
    # Check for new entry
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
    
    # Schedule next iteration
    time.sleep(random.uniform(1.0, 1.5))
    st.rerun()

# ==================== VISUALIZATION ====================

def create_candlestick_chart(df, trades=None, live_position=None):
    """Create candlestick chart with indicators"""
    fig = make_subplots(rows=1, cols=1)
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # EMAs if available
    if 'EMA_Fast' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], name='EMA Fast', line=dict(color='blue', width=1)))
    
    if 'EMA_Slow' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], name='EMA Slow', line=dict(color='orange', width=1)))
    
    # Live position lines
    if live_position:
        # Entry line
        fig.add_hline(y=live_position['entry_price'], line_dash="dash", line_color="blue", annotation_text="Entry")
        
        # SL line
        if st.session_state.get('current_sl'):
            fig.add_hline(y=st.session_state['current_sl'], line_dash="dash", line_color="red", annotation_text="SL")
        
        # Target line
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
    """Initialize session state variables"""
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
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode Selection
        mode = st.selectbox("Mode", ["Backtesting", "Live Trading"], key="mode_select")
        
        # Asset Selection
        asset_name = st.selectbox("Select Asset", list(ASSET_MAPPING.keys()), key="asset_select")
        
        if asset_name == "Custom":
            custom_ticker = st.text_input("Enter Custom Ticker", value="AAPL", key="custom_ticker")
            ticker = custom_ticker
        else:
            ticker = ASSET_MAPPING[asset_name]
        
        # Interval Selection
        interval = st.selectbox("Interval", list(INTERVAL_PERIODS.keys()), key="interval_select")
        
        # Period Selection
        period = st.selectbox("Period", INTERVAL_PERIODS[interval], key="period_select")
        
        # Quantity
        quantity = st.number_input("Quantity", min_value=1, value=1, key="quantity_input")
        
        # Strategy Selection
        strategy = st.selectbox("Strategy", STRATEGIES, key="strategy_select")
        
        st.markdown("---")
        
        # Strategy Configuration
        config = {
            'ticker': ticker,
            'interval': interval,
            'period': period,
            'quantity': quantity,
            'strategy': strategy
        }
        
        # EMA Crossover Strategy Config
        if strategy == "EMA Crossover Strategy":
            st.subheader("EMA Crossover Settings")
            config['ema_fast'] = st.number_input("EMA Fast", min_value=1, value=9, key="ema_fast")
            config['ema_slow'] = st.number_input("EMA Slow", min_value=1, value=15, key="ema_slow")
            config['min_angle'] = st.number_input("Minimum Crossover Angle (degrees)", min_value=0.0, value=1.0, key="min_angle")
            
            config['entry_filter'] = st.selectbox("Entry Filter", [
                "Simple Crossover",
                "Custom Candle (Points)",
                "ATR-based Candle"
            ], key="entry_filter")
            
            if config['entry_filter'] == "Custom Candle (Points)":
                config['custom_points'] = st.number_input("Custom Points", min_value=1.0, value=10.0, key="custom_points")
            
            if config['entry_filter'] == "ATR-based Candle":
                config['atr_multiplier'] = st.number_input("ATR Multiplier", min_value=0.1, value=1.0, key="atr_mult")
            
            config['use_adx'] = st.checkbox("Use ADX Filter", value=False, key="use_adx")
            if config['use_adx']:
                config['adx_period'] = st.number_input("ADX Period", min_value=1, value=14, key="adx_period")
                config['adx_threshold'] = st.number_input("ADX Threshold", min_value=1, value=25, key="adx_threshold")
        
        # Price Threshold Strategy Config
        elif strategy == "Price Crosses Threshold Strategy":
            st.subheader("Threshold Settings")
            config['threshold_price'] = st.number_input("Threshold Price", min_value=0.0, value=100.0, key="threshold_price")
            config['threshold_direction'] = st.selectbox("Direction", [
                "LONG (Price >= Threshold)",
                "SHORT (Price >= Threshold)",
                "LONG (Price <= Threshold)",
                "SHORT (Price <= Threshold)"
            ], key="threshold_direction")
        
        # Percentage Change Strategy Config
        elif strategy == "Percentage Change Strategy":
            st.subheader("Percentage Change Settings")
            config['pct_threshold'] = st.number_input("Percentage Threshold", min_value=0.01, value=0.5, step=0.01, key="pct_threshold")
            config['pct_direction'] = st.selectbox("Direction", [
                "BUY on Fall",
                "SELL on Fall",
                "BUY on Rise",
                "SELL on Rise"
            ], key="pct_direction")
        
        # Custom Strategy Builder Config
        elif strategy == "Custom Strategy Builder":
            st.subheader("Custom Conditions")
            
            if st.button("Add Condition", key="add_condition_btn"):
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
                    cond['use_condition'] = st.checkbox("Use this condition", value=cond.get('use_condition', True), key=f"use_
