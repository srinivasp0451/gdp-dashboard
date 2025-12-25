import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import time
import random

# Page config
st.set_page_config(page_title="Quantitative Trading System", layout="wide")

# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def get_ist_time():
    """Get current time in IST"""
    return datetime.now(pytz.timezone('Asia/Kolkata'))

def convert_to_ist(df):
    """Convert dataframe datetime index to IST"""
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
    else:
        df.index = df.index.tz_convert('Asia/Kolkata')
    return df

def safe_yfinance_fetch(ticker, period, interval):
    """Fetch data with rate limiting and error handling"""
    try:
        time.sleep(random.uniform(1.0, 1.5))
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty:
            return None
            
        # Flatten multi-index if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Select OHLCV
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[required_cols]
        
        # Convert to IST
        data = convert_to_ist(data)
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def validate_period_interval(period, interval):
    """Validate period-interval combination"""
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
    
    if interval in valid_combinations:
        return period in valid_combinations[interval]
    return False

# =====================================================
# TECHNICAL INDICATORS
# =====================================================

def calculate_ema(data, period):
    """Calculate EMA manually"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    """Calculate ATR manually"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    
    return atr

def calculate_ema_angle(ema_series, lookback=2):
    """Calculate EMA angle in degrees"""
    if len(ema_series) < lookback + 1:
        return 0
    
    y_diff = ema_series.iloc[-1] - ema_series.iloc[-lookback-1]
    x_diff = lookback
    
    angle_rad = np.arctan2(y_diff, x_diff)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def find_swing_high_low(df, lookback=5):
    """Find swing highs and lows"""
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(df) - lookback):
        # Swing High
        if df['High'].iloc[i] == df['High'].iloc[i-lookback:i+lookback+1].max():
            swing_highs.append((i, df['High'].iloc[i]))
        
        # Swing Low
        if df['Low'].iloc[i] == df['Low'].iloc[i-lookback:i+lookback+1].min():
            swing_lows.append((i, df['Low'].iloc[i]))
    
    return swing_highs, swing_lows

# =====================================================
# STRATEGY LOGIC
# =====================================================

def detect_ema_crossover(df, fast_period, slow_period, min_angle=20):
    """Detect EMA crossover with angle filter"""
    ema_fast = calculate_ema(df['Close'], fast_period)
    ema_slow = calculate_ema(df['Close'], slow_period)
    
    df['EMA_Fast'] = ema_fast
    df['EMA_Slow'] = ema_slow
    
    # Calculate angles
    df['Fast_Angle'] = ema_fast.rolling(3).apply(lambda x: calculate_ema_angle(x, 2))
    df['Slow_Angle'] = ema_slow.rolling(3).apply(lambda x: calculate_ema_angle(x, 2))
    
    # Crossover detection
    df['Signal'] = 0
    
    for i in range(1, len(df)):
        # Bullish crossover
        if (df['EMA_Fast'].iloc[i] > df['EMA_Slow'].iloc[i] and 
            df['EMA_Fast'].iloc[i-1] <= df['EMA_Slow'].iloc[i-1]):
            
            angle = abs(df['Fast_Angle'].iloc[i])
            if angle >= min_angle:
                df.loc[df.index[i], 'Signal'] = 1  # Buy
        
        # Bearish crossover
        elif (df['EMA_Fast'].iloc[i] < df['EMA_Slow'].iloc[i] and 
              df['EMA_Fast'].iloc[i-1] >= df['EMA_Slow'].iloc[i-1]):
            
            angle = abs(df['Fast_Angle'].iloc[i])
            if angle >= min_angle:
                df.loc[df.index[i], 'Signal'] = -1  # Sell
    
    return df

def calculate_stop_loss(df, idx, signal, sl_type, sl_value, atr_multiplier=2):
    """Calculate stop loss based on type"""
    entry_price = df['Close'].iloc[idx]
    
    if sl_type == "Custom Points":
        if signal == 1:  # Buy
            return entry_price - sl_value
        else:  # Sell
            return entry_price + sl_value
    
    elif sl_type == "ATR Based":
        atr = calculate_atr(df).iloc[idx]
        if signal == 1:
            return entry_price - (atr * atr_multiplier)
        else:
            return entry_price + (atr * atr_multiplier)
    
    elif sl_type == "Current Candle Low/High":
        if signal == 1:
            return df['Low'].iloc[idx]
        else:
            return df['High'].iloc[idx]
    
    elif sl_type == "Previous Candle Low/High":
        if idx > 0:
            if signal == 1:
                return df['Low'].iloc[idx-1]
            else:
                return df['High'].iloc[idx-1]
        return entry_price
    
    elif sl_type == "Current Swing Low/High":
        swing_highs, swing_lows = find_swing_high_low(df[:idx+1])
        if signal == 1 and swing_lows:
            return swing_lows[-1][1]
        elif signal == -1 and swing_highs:
            return swing_highs[-1][1]
        return entry_price
    
    elif sl_type == "Previous Swing Low/High":
        swing_highs, swing_lows = find_swing_high_low(df[:idx+1])
        if signal == 1 and len(swing_lows) > 1:
            return swing_lows[-2][1]
        elif signal == -1 and len(swing_highs) > 1:
            return swing_highs[-2][1]
        return entry_price
    
    return entry_price

def calculate_target(df, idx, signal, target_type, target_value, risk_reward=2, atr_multiplier=2):
    """Calculate target based on type"""
    entry_price = df['Close'].iloc[idx]
    
    if target_type == "Custom Points":
        if signal == 1:
            return entry_price + target_value
        else:
            return entry_price - target_value
    
    elif target_type == "ATR Based":
        atr = calculate_atr(df).iloc[idx]
        if signal == 1:
            return entry_price + (atr * atr_multiplier)
        else:
            return entry_price - (atr * atr_multiplier)
    
    elif target_type == "Risk-Reward Based":
        sl = calculate_stop_loss(df, idx, signal, "Custom Points", target_value, atr_multiplier)
        risk = abs(entry_price - sl)
        if signal == 1:
            return entry_price + (risk * risk_reward)
        else:
            return entry_price - (risk * risk_reward)
    
    return entry_price

# =====================================================
# BACKTESTING ENGINE
# =====================================================

def run_backtest(df, strategy_params):
    """Run backtest candle by candle"""
    trades = []
    trade_logs = []
    
    in_position = False
    position_type = 0
    entry_idx = 0
    entry_price = 0
    stop_loss = 0
    target = 0
    trailing_sl_highest = 0
    trailing_sl_lowest = 0
    
    strategy_type = strategy_params['strategy_type']
    sl_type = strategy_params['sl_type']
    target_type = strategy_params['target_type']
    
    for i in range(1, len(df)):
        current_time = df.index[i]
        
        # Check exit conditions if in position
        if in_position:
            exit_reason = None
            exit_price = 0
            
            # Trailing SL update
            if sl_type == "Trailing SL":
                if position_type == 1:  # Long
                    if df['High'].iloc[i] > trailing_sl_highest:
                        trailing_sl_highest = df['High'].iloc[i]
                        stop_loss = trailing_sl_highest - strategy_params['sl_value']
                else:  # Short
                    if df['Low'].iloc[i] < trailing_sl_lowest:
                        trailing_sl_lowest = df['Low'].iloc[i]
                        stop_loss = trailing_sl_lowest + strategy_params['sl_value']
            
            # Check SL hit
            if position_type == 1:  # Long
                if df['Low'].iloc[i] <= stop_loss:
                    exit_reason = "Stop Loss Hit"
                    exit_price = stop_loss
            else:  # Short
                if df['High'].iloc[i] >= stop_loss:
                    exit_reason = "Stop Loss Hit"
                    exit_price = stop_loss
            
            # Check Target hit
            if not exit_reason:
                if target_type != "Signal Based":
                    if position_type == 1:
                        if df['High'].iloc[i] >= target:
                            exit_reason = "Target Hit"
                            exit_price = target
                    else:
                        if df['Low'].iloc[i] <= target:
                            exit_reason = "Target Hit"
                            exit_price = target
            
            # Signal based exit
            if not exit_reason and sl_type == "Signal Based":
                if df['Signal'].iloc[i] == -position_type:
                    exit_reason = "Reverse Signal"
                    exit_price = df['Close'].iloc[i]
            
            # Exit if reason found
            if exit_reason:
                pnl = (exit_price - entry_price) * position_type
                points = pnl
                
                trades.append({
                    'Entry Time': df.index[entry_idx],
                    'Exit Time': current_time,
                    'Type': 'BUY' if position_type == 1 else 'SELL',
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Stop Loss': stop_loss,
                    'Target': target,
                    'Exit Reason': exit_reason,
                    'Points': round(points, 2),
                    'PnL': round(pnl, 2)
                })
                
                trade_logs.append(f"{current_time} - Exited {('LONG' if position_type == 1 else 'SHORT')} at {exit_price:.2f} - Reason: {exit_reason} - PnL: {pnl:.2f}")
                
                in_position = False
                position_type = 0
                continue
        
        # Entry logic if not in position
        if not in_position and df['Signal'].iloc[i] != 0:
            signal = df['Signal'].iloc[i]
            entry_price = df['Close'].iloc[i]
            entry_idx = i
            position_type = signal
            
            # Calculate SL and Target
            stop_loss = calculate_stop_loss(df, i, signal, sl_type, 
                                           strategy_params['sl_value'], 
                                           strategy_params['atr_multiplier'])
            
            if target_type == "Signal Based":
                target = 0  # Will exit on reverse signal
            else:
                target = calculate_target(df, i, signal, target_type, 
                                         strategy_params['target_value'],
                                         strategy_params['risk_reward'],
                                         strategy_params['atr_multiplier'])
            
            # Initialize trailing SL
            if sl_type == "Trailing SL":
                if signal == 1:
                    trailing_sl_highest = df['High'].iloc[i]
                else:
                    trailing_sl_lowest = df['Low'].iloc[i]
            
            in_position = True
            target_str = f"{target:.2f}" if target != 0 else "Signal Based"
            trade_logs.append(f"{current_time} - Entered {('LONG' if signal == 1 else 'SHORT')} at {entry_price:.2f} - SL: {stop_loss:.2f} - Target: {target_str}")
    
    return trades, trade_logs

# =====================================================
# STREAMLIT APP
# =====================================================

def main():
    st.title("🚀 Professional Quantitative Trading System")
    
    # Initialize session state
    if 'live_running' not in st.session_state:
        st.session_state.live_running = False
    if 'live_data' not in st.session_state:
        st.session_state.live_data = None
    if 'live_trades' not in st.session_state:
        st.session_state.live_trades = []
    if 'live_logs' not in st.session_state:
        st.session_state.live_logs = []
    if 'in_position' not in st.session_state:
        st.session_state.in_position = False
    if 'position_type' not in st.session_state:
        st.session_state.position_type = 0
    if 'entry_price' not in st.session_state:
        st.session_state.entry_price = 0
    if 'stop_loss' not in st.session_state:
        st.session_state.stop_loss = 0
    if 'target' not in st.session_state:
        st.session_state.target = 0
    if 'entry_idx' not in st.session_state:
        st.session_state.entry_idx = 0
    if 'trailing_sl_highest' not in st.session_state:
        st.session_state.trailing_sl_highest = 0
    if 'trailing_sl_lowest' not in st.session_state:
        st.session_state.trailing_sl_lowest = 0
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Asset selection
    asset_type = st.sidebar.selectbox("Asset Type", 
        ["Indian Indices", "Crypto", "Forex", "Commodities", "Custom"])
    
    ticker_map = {
        "Indian Indices": {"NIFTY 50": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"},
        "Crypto": {"BTC": "BTC-USD", "ETH": "ETH-USD"},
        "Forex": {"USDINR": "USDINR=X", "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X"},
        "Commodities": {"Gold": "GC=F", "Silver": "SI=F"}
    }
    
    if asset_type == "Custom":
        ticker = st.sidebar.text_input("Enter Ticker", "AAPL")
    else:
        asset_name = st.sidebar.selectbox("Select Asset", list(ticker_map[asset_type].keys()))
        ticker = ticker_map[asset_type][asset_name]
    
    # Timeframe selection
    interval = st.sidebar.selectbox("Interval", 
        ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1wk', '1mo'])
    
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
    
    # Strategy selection
    st.sidebar.header("Strategy Configuration")
    strategy_type = st.sidebar.selectbox("Strategy", 
        ["EMA Crossover", "Simple Buy", "Simple Sell"])
    
    # Strategy parameters
    if strategy_type == "EMA Crossover":
        ema_fast = st.sidebar.number_input("EMA Fast", value=9, min_value=2)
        ema_slow = st.sidebar.number_input("EMA Slow", value=15, min_value=2)
        min_angle = st.sidebar.number_input("Min Crossover Angle (degrees)", value=20.0, min_value=0.0)
    else:
        ema_fast = 9
        ema_slow = 15
        min_angle = 0
    
    # Stop Loss configuration
    st.sidebar.subheader("Stop Loss Configuration")
    sl_type = st.sidebar.selectbox("SL Type", 
        ["Custom Points", "Trailing SL", "ATR Based", "Current Candle Low/High", 
         "Previous Candle Low/High", "Current Swing Low/High", "Previous Swing Low/High",
         "Signal Based"])
    
    if sl_type == "Custom Points" or sl_type == "Trailing SL":
        sl_value = st.sidebar.number_input("SL Points", value=50.0, min_value=0.0)
    else:
        sl_value = 50.0
    
    atr_multiplier = st.sidebar.number_input("ATR Multiplier", value=2.0, min_value=0.1) if sl_type == "ATR Based" else 2.0
    
    # Target configuration
    st.sidebar.subheader("Target Configuration")
    target_type = st.sidebar.selectbox("Target Type", 
        ["Custom Points", "ATR Based", "Risk-Reward Based", "Signal Based"])
    
    if target_type == "Custom Points":
        target_value = st.sidebar.number_input("Target Points", value=100.0, min_value=0.0)
    else:
        target_value = 100.0
    
    risk_reward = st.sidebar.number_input("Risk-Reward Ratio", value=2.0, min_value=0.1) if target_type == "Risk-Reward Based" else 2.0
    
    strategy_params = {
        'strategy_type': strategy_type,
        'ema_fast': ema_fast,
        'ema_slow': ema_slow,
        'min_angle': min_angle,
        'sl_type': sl_type,
        'sl_value': sl_value,
        'atr_multiplier': atr_multiplier,
        'target_type': target_type,
        'target_value': target_value,
        'risk_reward': risk_reward
    }
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Backtesting", "🔴 Live Trading", "📜 Trade History & Logs"])
    
    # =====================================================
    # TAB 1: BACKTESTING
    # =====================================================
    with tab1:
        st.header("Backtesting Engine")
        
        if st.button("Run Backtest", key="backtest_btn"):
            if not validate_period_interval(period, interval):
                st.error(f"Invalid combination: {interval} with {period}")
            else:
                with st.spinner("Fetching data and running backtest..."):
                    progress_bar = st.progress(0)
                    
                    # Fetch data
                    progress_bar.progress(20)
                    df = safe_yfinance_fetch(ticker, period, interval)
                    
                    if df is None or df.empty:
                        st.error("Failed to fetch data")
                    else:
                        progress_bar.progress(40)
                        
                        # Apply strategy
                        if strategy_type == "EMA Crossover":
                            df = detect_ema_crossover(df, ema_fast, ema_slow, min_angle)
                        elif strategy_type == "Simple Buy":
                            df['Signal'] = 0
                            df.loc[df.index[10], 'Signal'] = 1
                        else:  # Simple Sell
                            df['Signal'] = 0
                            df.loc[df.index[10], 'Signal'] = -1
                        
                        progress_bar.progress(60)
                        
                        # Run backtest
                        trades, logs = run_backtest(df, strategy_params)
                        progress_bar.progress(100)
                        
                        # Display results
                        st.success("Backtest completed!")
                        
                        if trades:
                            trades_df = pd.DataFrame(trades)
                            
                            # Calculate metrics
                            total_trades = len(trades_df)
                            winning_trades = len(trades_df[trades_df['PnL'] > 0])
                            accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                            total_pnl = trades_df['PnL'].sum()
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total Trades", total_trades)
                            col2.metric("Winning Trades", winning_trades)
                            col3.metric("Accuracy", f"{accuracy:.2f}%")
                            col4.metric("Total PnL", f"{total_pnl:.2f}")
                            
                            st.subheader("Trade Details")
                            st.dataframe(trades_df, use_container_width=True)
                            
                            # Chart
                            st.subheader("Price Chart with Signals")
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
                                                        mode='lines', name=f'EMA {ema_fast}',
                                                        line=dict(color='blue')))
                                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], 
                                                        mode='lines', name=f'EMA {ema_slow}',
                                                        line=dict(color='red')))
                            
                            # Add entry markers
                            buy_signals = df[df['Signal'] == 1]
                            sell_signals = df[df['Signal'] == -1]
                            
                            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low'],
                                                    mode='markers', name='Buy Signal',
                                                    marker=dict(symbol='triangle-up', size=12, color='green')))
                            
                            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High'],
                                                    mode='markers', name='Sell Signal',
                                                    marker=dict(symbol='triangle-down', size=12, color='red')))
                            
                            fig.update_layout(height=600, xaxis_rangeslider_visible=False)
                            st.plotly_chart(fig, use_container_width=True, key="backtest_chart")
                            
                            # Trade logs
                            st.subheader("Trade Logs")
                            for log in logs:
                                st.text(log)
                        else:
                            st.warning("No trades generated during backtest period")
    
    # =====================================================
    # TAB 2: LIVE TRADING
    # =====================================================
    with tab2:
        st.header("Live Trading Dashboard (Simulated)")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("▶️ Start Live Trading", key="start_live"):
                st.session_state.live_running = True
                st.session_state.live_trades = []
                st.session_state.live_logs = []
                st.session_state.in_position = False
                st.session_state.position_type = 0
                st.session_state.entry_price = 0
                st.session_state.stop_loss = 0
                st.session_state.target = 0
                st.session_state.entry_idx = 0
                st.session_state.trailing_sl_highest = 0
                st.session_state.trailing_sl_lowest = 0
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop Live Trading", key="stop_live"):
                st.session_state.live_running = False
                st.rerun()
        
        if st.session_state.live_running:
            # Fetch latest data
            df = safe_yfinance_fetch(ticker, period, interval)
            
            if df is not None and not df.empty:
                # Apply strategy
                if strategy_type == "EMA Crossover":
                    df = detect_ema_crossover(df, ema_fast, ema_slow, min_angle)
                else:
                    df['Signal'] = 0
                
                st.session_state.live_data = df
                
                # Live trading logic
                current_idx = len(df) - 1
                current_price = df['Close'].iloc[-1]
                current_high = df['High'].iloc[-1]
                current_low = df['Low'].iloc[-1]
                current_time = df.index[-1]
                
                # Check if in position
                if st.session_state.in_position:
                    exit_reason = None
                    exit_price = 0
                    position_type = st.session_state.position_type
                    stop_loss = st.session_state.stop_loss
                    target = st.session_state.target
                    
                    # Update trailing SL
                    if sl_type == "Trailing SL":
                        if position_type == 1:  # Long
                            if current_high > st.session_state.trailing_sl_highest:
                                st.session_state.trailing_sl_highest = current_high
                                st.session_state.stop_loss = st.session_state.trailing_sl_highest - sl_value
                                stop_loss = st.session_state.stop_loss
                        else:  # Short
                            if current_low < st.session_state.trailing_sl_lowest:
                                st.session_state.trailing_sl_lowest = current_low
                                st.session_state.stop_loss = st.session_state.trailing_sl_lowest + sl_value
                                stop_loss = st.session_state.stop_loss
                    
                    # Check SL hit
                    if position_type == 1:  # Long
                        if current_low <= stop_loss:
                            exit_reason = "Stop Loss Hit"
                            exit_price = stop_loss
                    else:  # Short
                        if current_high >= stop_loss:
                            exit_reason = "Stop Loss Hit"
                            exit_price = stop_loss
                    
                    # Check Target hit
                    if not exit_reason and target_type != "Signal Based":
                        if position_type == 1:
                            if current_high >= target:
                                exit_reason = "Target Hit"
                                exit_price = target
                        else:
                            if current_low <= target:
                                exit_reason = "Target Hit"
                                exit_price = target
                    
                    # Signal based exit
                    if not exit_reason and sl_type == "Signal Based":
                        if df['Signal'].iloc[-1] == -position_type:
                            exit_reason = "Reverse Signal"
                            exit_price = current_price
                    
                    # Exit position
                    if exit_reason:
                        pnl = (exit_price - st.session_state.entry_price) * position_type
                        
                        st.session_state.live_trades.append({
                            'Entry Time': df.index[st.session_state.entry_idx],
                            'Exit Time': current_time,
                            'Type': 'BUY' if position_type == 1 else 'SELL',
                            'Entry Price': st.session_state.entry_price,
                            'Exit Price': exit_price,
                            'Stop Loss': stop_loss,
                            'Target': target,
                            'Exit Reason': exit_reason,
                            'Points': round(pnl, 2),
                            'PnL': round(pnl, 2)
                        })
                        
                        st.session_state.live_logs.append(
                            f"{current_time} - Exited {('LONG' if position_type == 1 else 'SHORT')} at {exit_price:.2f} - Reason: {exit_reason} - PnL: {pnl:.2f}"
                        )
                        
                        st.session_state.in_position = False
                        st.session_state.position_type = 0
                
                # Entry logic (only check if not in position and signal present)
                if not st.session_state.in_position and df['Signal'].iloc[-1] != 0:
                    signal = df['Signal'].iloc[-1]
                    entry_price = current_price
                    
                    # Calculate SL
                    stop_loss = calculate_stop_loss(df, current_idx, signal, sl_type, 
                                                   sl_value, atr_multiplier)
                    
                    # Ensure SL is not too close (minimum 0.5% distance)
                    min_sl_distance = entry_price * 0.005
                    if signal == 1:  # Long
                        if entry_price - stop_loss < min_sl_distance:
                            stop_loss = entry_price - min_sl_distance
                    else:  # Short
                        if stop_loss - entry_price < min_sl_distance:
                            stop_loss = entry_price + min_sl_distance
                    
                    # Calculate Target
                    if target_type == "Signal Based":
                        target = 0
                    else:
                        target = calculate_target(df, current_idx, signal, target_type, 
                                                 target_value, risk_reward, atr_multiplier)
                        
                        # Ensure target is not too close
                        min_target_distance = entry_price * 0.01
                        if signal == 1:  # Long
                            if target - entry_price < min_target_distance:
                                target = entry_price + min_target_distance
                        else:  # Short
                            if entry_price - target < min_target_distance:
                                target = entry_price - min_target_distance
                    
                    # Enter position
                    st.session_state.in_position = True
                    st.session_state.position_type = signal
                    st.session_state.entry_price = entry_price
                    st.session_state.stop_loss = stop_loss
                    st.session_state.target = target
                    st.session_state.entry_idx = current_idx
                    
                    # Initialize trailing SL
                    if sl_type == "Trailing SL":
                        if signal == 1:
                            st.session_state.trailing_sl_highest = current_high
                        else:
                            st.session_state.trailing_sl_lowest = current_low
                    
                    target_str = f"{target:.2f}" if target != 0 else "Signal Based"
                    st.session_state.live_logs.append(
                        f"{current_time} - Entered {('LONG' if signal == 1 else 'SHORT')} at {entry_price:.2f} - SL: {stop_loss:.2f} - Target: {target_str}"
                    )
                
                # Display current status
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Price", f"{current_price:.2f}")
                
                with col2:
                    if 'EMA_Fast' in df.columns:
                        current_angle = abs(df['Fast_Angle'].iloc[-1]) if not pd.isna(df['Fast_Angle'].iloc[-1]) else 0
                        st.metric("Current Angle", f"{current_angle:.2f}°")
                
                with col3:
                    position_status = "OPEN" if st.session_state.in_position else "CLOSED"
                    st.metric("Position", position_status)
                
                # Display position details if in position
                if st.session_state.in_position:
                    st.info(f"""
                    📊 **Active Position**
                    - Type: {('LONG' if st.session_state.position_type == 1 else 'SHORT')}
                    - Entry: {st.session_state.entry_price:.2f}
                    - Current: {current_price:.2f}
                    - Stop Loss: {st.session_state.stop_loss:.2f}
                    - Target: {st.session_state.target:.2f if st.session_state.target != 0 else 'Signal Based'}
                    - Unrealized P&L: {((current_price - st.session_state.entry_price) * st.session_state.position_type):.2f}
                    """)
                
                # Strategy guidance
                st.subheader("Strategy Guidance")
                
                if st.session_state.in_position:
                    current_pnl = (current_price - st.session_state.entry_price) * st.session_state.position_type
                    if current_pnl > 0:
                        st.success(f"🟢 Position in PROFIT: +{current_pnl:.2f} points")
                    else:
                        st.warning(f"🟡 Position in LOSS: {current_pnl:.2f} points")
                else:
                    if df['Signal'].iloc[-1] == 1:
                        st.success("🟢 BUY SIGNAL DETECTED - Strong bullish momentum")
                    elif df['Signal'].iloc[-1] == -1:
                        st.error("🔴 SELL SIGNAL DETECTED - Strong bearish momentum")
                    else:
                        st.warning("⏸️ HOLD - No entry signal. Waiting for EMA crossover...")
                
                # Live chart
                st.subheader("Live Price Chart")
                
                chart_df = df.tail(100)  # Show last 100 candles
                
                fig = go.Figure()
                
                fig.add_trace(go.Candlestick(
                    x=chart_df.index,
                    open=chart_df['Open'],
                    high=chart_df['High'],
                    low=chart_df['Low'],
                    close=chart_df['Close'],
                    name='Price'
                ))
                
                if 'EMA_Fast' in df.columns:
                    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_Fast'], 
                                            mode='lines', name=f'EMA {ema_fast}',
                                            line=dict(color='blue', width=2)))
                    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_Slow'], 
                                            mode='lines', name=f'EMA {ema_slow}',
                                            line=dict(color='red', width=2)))
                
                # Add SL and Target lines if in position
                if st.session_state.in_position:
                    fig.add_hline(y=st.session_state.stop_loss, line_dash="dash", 
                                 line_color="red", annotation_text="Stop Loss")
                    if st.session_state.target != 0:
                        fig.add_hline(y=st.session_state.target, line_dash="dash", 
                                     line_color="green", annotation_text="Target")
                    fig.add_hline(y=st.session_state.entry_price, line_dash="dot", 
                                 line_color="yellow", annotation_text="Entry")
                
                fig.update_layout(height=500, xaxis_rangeslider_visible=False, 
                                title=f"{ticker} - {interval}")
                st.plotly_chart(fig, use_container_width=True, key=f"live_chart_{get_ist_time().timestamp()}")
                
                # Display parameters
                with st.expander("Strategy Parameters"):
                    st.write(f"**Strategy:** {strategy_type}")
                    if strategy_type == "EMA Crossover":
                        st.write(f"**EMA Fast:** {ema_fast}{df}")
                        st.write(f"**EMA Slow:** {ema_slow}")
                        st.write(f"**Min Angle:** {min_angle}°")
                    st.write(f"**SL Type:** {sl_type}")
                    st.write(f"**Target Type:** {target_type}")
                
                # Auto refresh
                time.sleep(random.uniform(1.0, 1.5))
                st.rerun()
            
            else:
                st.error("Unable to fetch live data")
        
        else:
            st.info("Click 'Start Live Trading' to begin monitoring")
    
    # =====================================================
    # TAB 3: HISTORY & LOGS
    # =====================================================
    with tab3:
        st.header("Trade History & Logs")
        
        subtab1, subtab2 = st.tabs(["Trade History", "Trade Logs"])
        
        with subtab1:
            if st.session_state.live_trades:
                trades_df = pd.DataFrame(st.session_state.live_trades)
                
                # Calculate live metrics
                total_trades = len(trades_df)
                winning_trades = len(trades_df[trades_df['PnL'] > 0])
                accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                total_pnl = trades_df['PnL'].sum()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Trades", total_trades)
                col2.metric("Winning Trades", winning_trades)
                col3.metric("Accuracy", f"{accuracy:.2f}%")
                col4.metric("Total PnL", f"{total_pnl:.2f}")
                
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.info("No trades executed yet")
        
        with subtab2:
            if st.session_state.live_logs:
                st.subheader("Live Trade Logs")
                for log in st.session_state.live_logs:
                    st.text(log)
            else:
                st.info("No trade logs available")

if __name__ == "__main__":
    main()
