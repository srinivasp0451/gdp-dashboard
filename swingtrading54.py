import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pytz
from abc import ABC, abstractmethod
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
import json
import math

# Set page config
st.set_page_config(page_title="Algo Trading System", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem;}
    .sub-header {font-size: 1.5rem; font-weight: bold; color: #ff7f0e; margin-top: 1rem;}
    .profit {color: #2ca02c; font-weight: bold; font-size: 1.2rem;}
    .loss {color: #d62728; font-weight: bold; font-size: 1.2rem;}
    .neutral {color: #7f7f7f; font-weight: bold;}
    .status-box {padding: 1rem; border-radius: 10px; background: #f0f2f6; margin: 1rem 0;}
    .trade-guidance {padding: 1rem; border-radius: 10px; background: #e1f5ff; border-left: 4px solid #1f77b4; margin: 1rem 0;}
    .log-container {height: 600px; overflow-y: scroll; background: #f8f9fa; padding: 1rem; border-radius: 5px; font-family: monospace; font-size: 0.85rem;}
    .metric-card {background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 0.5rem 0;}
    .stats-box {background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #17a2b8;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trading_active' not in st.session_state:
    st.session_state.trading_active = False
if 'current_position' not in st.session_state:
    st.session_state.current_position =     crossover_type = st.sidebar.radio("Crossover Type:", 
                                      ['simple', 'auto_strong_candle', 
                                       'atr_strong_candle', 'custom_candle'])
    
    min_angle = st.sidebar.number_input("Min Crossover Angle (degrees):", 
                                        min_value=0, max_value=90, value=30, step=5,
                                        help="Minimum angle for strong trend confirmation (0 = no angle requirement)")
    
    if crossover_type == 'atr_strong_candle':
        atr_multiplier = st.sidebar.number_input("ATR Multiplier:", 
                                                  min_value=0.5, max_value=3.0, value=1.0, step=0.1)
        strategy_params['atr_multiplier'] = atr_multiplier
    elif crossover_type == 'custom_candle':
        custom_size = st.sidebar.number_input("Custom Candle Size:", 
                                               min_value=1.0, max_value=100.0, value=10.0, step=0.5)
        strategy_params['custom_candle_size'] = custom_size
    
    strategy_params = {
        'ma_type1': ma_type1,
        'period1': period1,
        'ma_type2': ma_type2,
        'period2': period2,
        'crossover_type': crossover_type,
        'min_crossover_angle': min_angle
    }
    strategy_obj = EMACrossoverStrategy(strategy_params)

elif selected_strategy == 'Elliott Wave':
    st.sidebar.markdown("**Elliott Wave Detection:**")
    st.sidebar.info("Automatically detects 5-wave impulse patterns")
    strategy_obj = ElliottWaveStrategy()

elif selected_strategy == 'Pair Ratio Trading':
    st.sidebar.markdown("**Select Second Ticker:**")
    ticker2_type = st.sidebar.radio("Ticker 2 Type:", ['Preset Assets', 'Custom Ticker', 'Indian Stock'], key='t2')
    
    if ticker2_type == 'Preset Assets':
        selected_asset2 = st.sidebar.selectbox("Choose Asset 2:", list(preset_assets.keys()), key='asset2')
        ticker2 = preset_assets[selected_asset2]
    elif ticker2_type == 'Custom Ticker':
        ticker2 = st.sidebar.text_input("Enter Ticker 2:", "MSFT", key='ticker2')
    else:
        stock_name2 = st.sidebar.text_input("Enter Stock 2:", "TCS", key='stock2')
        ticker2 = f"{stock_name2}.NS"
    
    zscore_threshold = st.sidebar.number_input("Z-Score Threshold:", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    strategy_params = {'zscore_threshold': zscore_threshold}
    strategy_obj = PairRatioStrategy(strategy_params)

elif selected_strategy == 'RSI + Divergence':
    rsi_period = st.sidebar.number_input("RSI Period:", min_value=5, max_value=50, value=14)
    use_divergence = st.sidebar.checkbox("Use Divergence Detection", value=True)
    strategy_params = {'rsi_period': rsi_period, 'use_divergence': use_divergence}
    strategy_obj = RSIDivergenceStrategy(strategy_params)

elif selected_strategy == 'Fibonacci Retracement':
    lookback = st.sidebar.number_input("Lookback Period:", min_value=20, max_value=200, value=50)
    tolerance = st.sidebar.number_input("Level Tolerance (%):", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    strategy_params = {'lookback': lookback, 'tolerance': tolerance / 100}
    strategy_obj = FibonacciRetracementStrategy(strategy_params)

elif selected_strategy == 'Z-Score Mean Reversion':
    window = st.sidebar.number_input("Window Period:", min_value=10, max_value=100, value=20)
    threshold = st.sidebar.number_input("Z-Score Threshold:", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    strategy_params = {'window': window, 'threshold': threshold}
    strategy_obj = ZScoreMeanReversionStrategy(strategy_params)

elif selected_strategy == 'Breakout + Volume':
    period_breakout = st.sidebar.number_input("Channel Period:", min_value=10, max_value=100, value=20)
    volume_multiplier = st.sidebar.number_input("Volume Multiplier:", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
    strategy_params = {'period': period_breakout, 'volume_multiplier': volume_multiplier}
    strategy_obj = BreakoutVolumeStrategy(strategy_params)

elif selected_strategy == 'Simple Buy':
    strategy_obj = SimpleBuyStrategy()

elif selected_strategy == 'Simple Sell':
    strategy_obj = SimpleSellStrategy()

# Risk Management
st.sidebar.subheader("🛡️ Risk Management")

quantity = st.sidebar.number_input("Quantity:", min_value=1, max_value=10000, value=1)

st.sidebar.markdown("**Stop Loss Configuration:**")
sl_types = ['Custom Points', 'Trail SL', 'Signal Based', 'Percentage', 'ATR']
sl_type = st.sidebar.selectbox("SL Type:", sl_types)

sl_config = {'type': sl_type}

if sl_type == 'Custom Points' or sl_type == 'Trail SL':
    sl_points = st.sidebar.number_input("SL Points:", min_value=1.0, max_value=1000.0, value=10.0, step=0.5)
    sl_config['points'] = sl_points
elif sl_type == 'Percentage':
    sl_pct = st.sidebar.number_input("SL Percentage:", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    sl_config['percentage'] = sl_pct
elif sl_type == 'ATR':
    atr_multiplier = st.sidebar.number_input("ATR Multiplier:", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    sl_config['atr_multiplier'] = atr_multiplier

st.sidebar.markdown("**Target Configuration:**")
target_types = ['Custom Points', 'Trail Target', 'Signal Based', 'Percentage', 'Risk Reward']
target_type = st.sidebar.selectbox("Target Type:", target_types)

target_config = {'type': target_type}

if target_type == 'Custom Points' or target_type == 'Trail Target':
    target_points = st.sidebar.number_input("Target Points:", min_value=1.0, max_value=1000.0, value=20.0, step=0.5)
    target_config['points'] = target_points
    if target_type == 'Trail Target':
        tolerance = st.sidebar.number_input("Tolerance (%):", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
        target_config['tolerance'] = tolerance
elif target_type == 'Percentage':
    target_pct = st.sidebar.number_input("Target Percentage:", min_value=0.1, max_value=20.0, value=2.0, step=0.1)
    target_config['percentage'] = target_pct
elif target_type == 'Risk Reward':
    rr_ratio = st.sidebar.number_input("Risk:Reward Ratio:", min_value=1.0, max_value=10.0, value=2.0, step=0.1)
    target_config['rr_ratio'] = rr_ratio

# Control Buttons
st.sidebar.markdown("---")
st.sidebar.subheader("🎮 Controls")

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("▶️ Start Trading", type="primary", use_container_width=True):
        st.session_state.trading_active = True
        st.session_state.iteration_count = 0
        add_log("🚀 Trading started")
        st.rerun()

with col2:
    if st.button("⏹️ Stop Trading", type="secondary", use_container_width=True):
        st.session_state.trading_active = False
        
        # Close any open position
        if st.session_state.current_position:
            trading_system = TradingSystem(ticker, timeframe, period, strategy_obj, quantity, sl_config, target_config, ticker2)
            data, data2 = trading_system.fetch_data()
            if data is not None and len(data) > 0:
                current_price = data['Close'].iloc[-1]
                position = st.session_state.current_position
                
                # Calculate final P&L
                if position['type'] == 'LONG':
                    pnl_points = current_price - position['entry_price']
                else:
                    pnl_points = position['entry_price'] - current_price
                
                pnl_pct = (pnl_points / position['entry_price']) * 100
                
                # Save to history
                trade_record = {
                    'entry_time': position['entry_time'],
                    'exit_time': get_ist_time(),
                    'duration': get_ist_time() - position['entry_time'],
                    'type': position['type'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'quantity': position['quantity'],
                    'pnl_points': pnl_points * position['quantity'],
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'Manual Stop',
                    'strategy': strategy_obj.name
                }
                
                st.session_state.trade_history.append(trade_record)
                add_log(f"🛑 Position closed manually: {position['type']} at {current_price:.2f}, P&L: {pnl_pct:.2f}%")
                st.session_state.current_position = None
        
        add_log("⏹️ Trading stopped")
        st.rerun()

if st.session_state.current_position and not st.session_state.trading_active:
    if st.sidebar.button("❌ Force Close Position", type="secondary", use_container_width=True):
        trading_system = TradingSystem(ticker, timeframe, period, strategy_obj, quantity, sl_config, target_config, ticker2)
        data, data2 = trading_system.fetch_data()
        if data is not None and len(data) > 0:
            current_price = data['Close'].iloc[-1]
            position = st.session_state.current_position
            
            if position['type'] == 'LONG':
                pnl_points = current_price - position['entry_price']
            else:
                pnl_points = position['entry_price'] - current_price
            
            pnl_pct = (pnl_points / position['entry_price']) * 100
            
            trade_record = {
                'entry_time': position['entry_time'],
                'exit_time': get_ist_time(),
                'duration': get_ist_time() - position['entry_time'],
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'quantity': position['quantity'],
                'pnl_points': pnl_points * position['quantity'],
                'pnl_pct': pnl_pct,
                'exit_reason': 'Force Close',
                'strategy': strategy_obj.name
            }
            
            st.session_state.trade_history.append(trade_record)
            add_log(f"❌ Position force closed: {position['type']} at {current_price:.2f}, P&L: {pnl_pct:.2f}%")
            st.session_state.current_position = None
            st.rerun()

# Main Content Area
st.markdown("<div class='main-header'>📈 Algorithmic Trading System</div>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔴 Live Trading", "📊 Trade History", "📝 Trade Log"])

# Tab 1: Live Trading
with tab1:
    if st.session_state.trading_active:
        st.markdown("### 🔴 LIVE - Auto-refreshing every 1.5-2s")
        
        # Initialize trading system
        trading_system = TradingSystem(ticker, timeframe, period, strategy_obj, quantity, sl_config, target_config, ticker2)
        
        # Fetch data
        data, data2 = trading_system.fetch_data()
        
        if data is None or len(data) < 10:
            st.error("❌ Insufficient data. Please check ticker symbol and timeframe/period compatibility.")
        else:
            st.session_state.last_data = (data, data2)
            st.session_state.iteration_count += 1
            
            current_price = data['Close'].iloc[-1]
            
            # Display header info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Candles", len(data))
            with col2:
                st.metric("Timeframe", timeframe)
            with col3:
                st.metric("Iteration", st.session_state.iteration_count)
            with col4:
                st.metric("Strategy", strategy_obj.name)
            
            # Check for entry signal if no position
            if st.session_state.current_position is None:
                if isinstance(strategy_obj, PairRatioStrategy) and data2 is not None:
                    bullish, bearish, signal_data = strategy_obj.generate_signal(data, data2)
                else:
                    bullish, bearish, signal_data = strategy_obj.generate_signal(data)
                
                if bullish or bearish:
                    position_type = 'LONG' if bullish else 'SHORT'
                    entry_price = current_price
                    
                    sl_price, target_price = trading_system.calculate_sl_target(entry_price, position_type, data)
                    
                    st.session_state.current_position = {
                        'type': position_type,
                        'entry_price': entry_price,
                        'entry_time': get_ist_time(),
                        'quantity': quantity,
                        'sl': sl_price,
                        'target': target_price,
                        'signal_data': signal_data
                    }
                    
                    add_log(f"✅ {position_type} position entered at {entry_price:.2f}, SL: {sl_price:.2f}, Target: {target_price:.2f}")
                    st.success(f"✅ {position_type} Position Entered at {entry_price:.2f}")
                else:
                    st.info("⏳ Waiting for entry signal...")
                    
                    # Display strategy parameters while waiting
                    st.markdown("### 📊 Strategy Indicators")
                    param_summary = strategy_obj.get_parameter_summary(data) if not isinstance(strategy_obj, PairRatioStrategy) else strategy_obj.get_parameter_summary(data, data2)
                    
                    if param_summary:
                        cols = st.columns(min(len(param_summary), 6))
                        for i, (key, value) in enumerate(param_summary.items()):
                            with cols[i % len(cols)]:
                                st.metric(key, value)
            
            # Manage existing position
            if st.session_state.current_position:
                position = st.session_state.current_position
                
                # Update trailing SL/Target
                position = trading_system.update_trailing_sl(position, current_price)
                position = trading_system.update_trailing_target(position, current_price)
                st.session_state.current_position = position
                
                # Calculate P&L
                if position['type'] == 'LONG':
                    pnl_points = current_price - position['entry_price']
                else:
                    pnl_points = position['entry_price'] - current_price
                
                pnl_pct = (pnl_points / position['entry_price']) * 100
                position['pnl_points'] = pnl_points
                position['pnl_pct'] = pnl_pct
                
                # Display position status
                st.markdown("### 💼 Current Position")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Position", position['type'])
                with col2:
                    st.metric("Entry Price", f"{position['entry_price']:.2f}")
                with col3:
                    st.metric("Current Price", f"{current_price:.2f}")
                with col4:
                    pnl_color = "profit" if pnl_pct > 0 else "loss"
                    st.markdown(f"<div class='{pnl_color}'>P&L: {pnl_points:.2f} pts<br>({pnl_pct:.2f}%)</div>", unsafe_allow_html=True)
                with col5:
                    st.metric("Quantity", position['quantity'])
                
                # Display strategy indicators
                st.markdown("### 📊 Strategy Indicators")
                param_summary = strategy_obj.get_parameter_summary(data) if not isinstance(strategy_obj, PairRatioStrategy) else strategy_obj.get_parameter_summary(data, data2)
                
                if param_summary:
                    cols = st.columns(min(len(param_summary), 6))
                    for i, (key, value) in enumerate(param_summary.items()):
                        with cols[i % len(cols)]:
                            st.metric(key, value)
                
                # Market status
                market_status = trading_system.get_market_status(position, current_price)
                st.markdown(f"<div class='status-box'>{market_status}</div>", unsafe_allow_html=True)
                
                # Trade guidance
                guidance = trading_system.generate_trade_guidance(position, current_price, data, position['signal_data'])
                st.markdown(f"<div class='trade-guidance'>{guidance}</div>", unsafe_allow_html=True)
                
                # Check exit conditions
                should_exit, exit_reason, exit_price = trading_system.check_exit_conditions(position, current_price, data, data2)
                
                if should_exit:
                    trade_record = {
                        'entry_time': position['entry_time'],
                        'exit_time': get_ist_time(),
                        'duration': get_ist_time() - position['entry_time'],
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'quantity': position['quantity'],
                        'pnl_points': pnl_points * position['quantity'],
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'strategy': strategy_obj.name
                    }
                    
                    st.session_state.trade_history.append(trade_record)
                    add_log(f"🏁 Trade closed: {exit_reason} at {exit_price:.2f}, P&L: {pnl_pct:.2f}%")
                    
                    # Display trade analysis
                    analysis = trading_system.analyze_trade(trade_record)
                    
                    st.markdown("### 🎯 Trade Analysis")
                    st.success(f"Trade closed: {exit_reason}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Performance:** {analysis['performance']}")
                        st.markdown(f"**Exit Quality:** {analysis['exit_quality']}")
                    with col2:
                        st.markdown(f"**Duration:** {analysis['duration_insight']}")
                        st.markdown("**Recommendations:**")
                        for rec in analysis['recommendations']:
                            st.markdown(f"- {rec}")
                    
                    st.session_state.current_position = None
                
                # Display chart with indicators
                st.markdown("### 📈 Live Chart with Indicators")
                
                fig = go.Figure()
                
                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ))
                
                # Add strategy-specific indicators to chart
                strategy_obj.add_to_chart(fig, data)
                
                # Entry marker
                if st.session_state.current_position:
                    entry_time = position['entry_time']
                    entry_price = position['entry_price']
                    marker_color = 'green' if position['type'] == 'LONG' else 'red'
                    marker_symbol = 'triangle-up' if position['type'] == 'LONG' else 'triangle-down'
                    
                    fig.add_trace(go.Scatter(
                        x=[entry_time],
                        y=[entry_price],
                        mode='markers',
                        marker=dict(size=15, color=marker_color, symbol=marker_symbol),
                        name=f"{position['type']} Entry"
                    ))
                    
                    # SL and Target lines
                    if position['sl']:
                        fig.add_hline(y=position['sl'], line_dash="dash", 
                                     annotation_text="Stop Loss", line_color="red")
                    if position['target']:
                        fig.add_hline(y=position['target'], line_dash="dash", 
                                     annotation_text="Target", line_color="green")
                
                fig.update_layout(
                    title=f"{ticker} - {timeframe} Chart with {strategy_obj.name}",
                    xaxis_title="Time (IST)",
                    yaxis_title="Price",
                    height=600,
                    template="plotly_white",
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Display Historical Statistics
            st.markdown("### 📊 Historical Strategy Statistics")
            
            if isinstance(strategy_obj, PairRatioStrategy) and data2 is not None:
                stats = strategy_obj.get_historical_statistics(data, data2)
            else:
                stats = strategy_obj.get_historical_statistics(data)
            
            if stats:
                for category, values in stats.items():
                    st.markdown(f"<div class='stats-box'><strong>{category}</strong></div>", unsafe_allow_html=True)
                    cols = st.columns(min(len(values), 4))
                    for i, (key, value) in enumerate(values.items()):
                        with cols[i % len(cols)]:
                            st.metric(key, value)
            
            # Auto-refresh
            time.sleep(1.5)
            st.rerun()
    
    else:
        st.info("👆 Click 'Start Trading' to begin live monitoring")
        
        if st.session_state.last_data:
            data, data2 = st.session_state.last_data
            
            st.markdown("### 📊 Strategy Preview")
            
            # Show chart with indicators
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ))
            
            # Add strategy indicators
            strategy_obj.add_to_chart(fig, data)
            
            fig.update_layout(
                title=f"{ticker} - {timeframe} Chart with {strategy_obj.name}",
                xaxis_title="Time (IST)",
                yaxis_title="Price",
                height=600,
                template="plotly_white",
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show strategy parameters
            st.markdown("### 📊 Current Strategy Parameters")
            if isinstance(strategy_obj, PairRatioStrategy) and data2 is not None:
                param_summary = strategy_obj.get_parameter_summary(data, data2)
            else:
                param_summary = strategy_obj.get_parameter_summary(data)
            
            if param_summary:
                cols = st.columns(min(len(param_summary), 6))
                for i, (key, value) in enumerate(param_summary.items()):
                    with cols[i % len(cols)]:
                        st.metric(key, value)
            
            # Show historical statistics
            st.markdown("### 📊 Historical Strategy Statistics")
            if isinstance(strategy_obj, PairRatioStrategy) and data2 is not None:
                stats = strategy_obj.get_historical_statistics(data, data2)
            else:
                stats = strategy_obj.get_historical_statistics(data)
            
            if stats:
                for category, values in stats.items():
                    st.markdown(f"<div class='stats-box'><strong>{category}</strong></div>", unsafe_allow_html=True)
                    cols = st.columns(min(len(values), 4))
                    for i, (key, value) in enumerate(values.items()):
                        with cols[i % len(cols)]:
                            st.metric(key, value)
            
            st.markdown("### 📋 Last Fetched Data")
            st.dataframe(data.tail(20))

# Tab 2: Trade History
with tab2:
    st.markdown("### 📊 Trade History & Performance")
    
    if len(st.session_state.trade_history) == 0:
        st.info("No trades yet. Start trading to see your performance history.")
    else:
        trades_df = pd.DataFrame(st.session_state.trade_history)
        
        # Performance Summary
        st.markdown("#### 📈 Performance Summary")
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
        losing_trades = len(trades_df[trades_df['pnl_pct'] <= 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = trades_df['pnl_points'].sum()
        avg_pnl = trades_df['pnl_pct'].mean()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Wins", winning_trades, delta=f"{win_rate:.1f}%")
        with col3:
            st.metric("Losses", losing_trades)
        with col4:
            pnl_color = "normal" if total_pnl >= 0 else "inverse"
            st.metric("Total P&L", f"{total_pnl:.2f} pts", delta_color=pnl_color)
        with col5:
            st.metric("Avg P&L", f"{avg_pnl:.2f}%")
        
        # Performance by Strategy
        st.markdown("#### 🎯 Performance by Strategy")
        strategy_performance = trades_df.groupby('strategy').agg({
            'pnl_points': 'sum',
            'pnl_pct': 'mean',
            'type': 'count'
        }).rename(columns={'type': 'trades'})
        st.dataframe(strategy_performance)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Cumulative P&L
            trades_df['cumulative_pnl'] = trades_df['pnl_points'].cumsum()
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=list(range(len(trades_df))),
                y=trades_df['cumulative_pnl'],
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='blue', width=2)
            ))
            fig_cum.update_layout(
                title="Cumulative P&L",
                xaxis_title="Trade Number",
                yaxis_title="P&L (points)",
                height=400
            )
            st.plotly_chart(fig_cum, use_container_width=True)
        
        with col2:
            # Win/Loss Pie Chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Wins', 'Losses'],
                values=[winning_trades, losing_trades],
                marker=dict(colors=['green', 'red'])
            )])
            fig_pie.update_layout(title="Win/Loss Distribution", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # P&L Distribution
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=trades_df['pnl_pct'],
            nbinsx=20,
            marker=dict(color='lightblue', line=dict(color='darkblue', width=1))
        ))
        fig_hist.update_layout(
            title="P&L Distribution",
            xaxis_title="P&L (%)",
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Detailed Trade Table
        st.markdown("#### 📋 Detailed Trade Log")
        
        # Format the dataframe for display
        display_df = trades_df.copy()
        display_df['entry_time'] = display_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['exit_time'] = display_df['exit_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['duration'] = display_df['duration'].astype(str)
        display_df['pnl_points'] = display_df['pnl_points'].round(2)
        display_df['pnl_pct'] = display_df['pnl_pct'].round(2)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Individual Trade Analysis
        st.markdown("#### 🔍 Individual Trade Analysis")
        
        trade_idx = st.selectbox("Select Trade:", range(len(trades_df)), 
                                 format_func=lambda x: f"Trade {x+1} - {trades_df.iloc[x]['type']} - {trades_df.iloc[x]['exit_reason']}")
        
        if trade_idx is not None:
            selected_trade = st.session_state.trade_history[trade_idx]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Entry:** {selected_trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"**Entry Price:** {selected_trade['entry_price']:.2f}")
                st.markdown(f"**Position:** {selected_trade['type']}")
            with col2:
                st.markdown(f"**Exit:** {selected_trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"**Exit Price:** {selected_trade['exit_price']:.2f}")
                st.markdown(f"**Exit Reason:** {selected_trade['exit_reason']}")
            with col3:
                st.markdown(f"**Duration:** {selected_trade['duration']}")
                pnl_class = "profit" if selected_trade['pnl_pct'] > 0 else "loss"
                st.markdown(f"<div class='{pnl_class}'>P&L: {selected_trade['pnl_points']:.2f} pts ({selected_trade['pnl_pct']:.2f}%)</div>", 
                          unsafe_allow_html=True)
                st.markdown(f"**Strategy:** {selected_trade['strategy']}")
            
            # AI Analysis
            trading_system = TradingSystem(ticker, timeframe, period, strategy_obj, quantity, sl_config, target_config, ticker2)
            analysis = trading_system.analyze_trade(selected_trade)
            
            st.markdown("**AI Analysis:**")
            st.info(f"**Performance:** {analysis['performance']}")
            st.info(f"**Exit Quality:** {analysis['exit_quality']}")
            st.info(f"**Duration Insight:** {analysis['duration_insight']}")
            
            st.markdown("**Recommendations:**")
            for rec in analysis['recommendations']:
                st.markdown(f"- {rec}")

# Tab 3: Trade Log
with tab3:
    st.markdown("### 📝 Trade Log")
    
    if st.button("🗑️ Clear Log"):
        st.session_state.trade_log = []
        st.rerun()
    
    if len(st.session_state.trade_log) == 0:
        st.info("No log entries yet.")
    else:
        log_html = "<div class='log-container'>"
        for entry in st.session_state.trade_log:
            log_html += f"<div>{entry}</div>"
        log_html += "</div>"
        st.markdown(log_html, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><b>Professional Algorithmic Trading System</b></p>
    <p>⚠️ This is for educational purposes only. Trading involves risk. Always do your own research.</p>
    <p>💡 Tip: Use appropriate position sizing and risk management for your capital.</p>
    <p>🎯 Features: Elliott Waves, Enhanced EMA Crossover (Angle & Candle Confirmation), Live Charts, Historical Statistics</p>
</div>
""", unsafe_allow_html=True)
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []
if 'iteration_count' not in st.session_state:
    st.session_state.iteration_count = 0
if 'last_data' not in st.session_state:
    st.session_state.last_data = None

# Timezone
IST = pytz.timezone('Asia/Kolkata')

def get_ist_time():
    """Get current time in IST"""
    return datetime.now(IST)

def add_log(message):
    """Add timestamped log entry"""
    timestamp = get_ist_time().strftime("%Y-%m-%d %H:%M:%S IST")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.trade_log.insert(0, log_entry)
    if len(st.session_state.trade_log) > 100:
        st.session_state.trade_log = st.session_state.trade_log[:100]

def calculate_atr(data, period=14):
    """Calculate Average True Range"""
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_crossover_angle(ma1_current, ma1_prev, ma2_current, ma2_prev):
    """Calculate angle of crossover in degrees"""
    # Calculate slopes
    slope1 = ma1_current - ma1_prev
    slope2 = ma2_current - ma2_prev
    
    # Calculate angle difference
    angle1 = math.degrees(math.atan(slope1))
    angle2 = math.degrees(math.atan(slope2))
    
    angle_diff = abs(angle1 - angle2)
    return angle_diff

# Base Strategy Class
class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, params=None):
        self.params = params or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def calculate_indicators(self, data):
        """Calculate strategy-specific indicators"""
        pass
    
    @abstractmethod
    def generate_signal(self, data):
        """Generate trading signal. Returns (bullish_signal, bearish_signal, signal_data)"""
        pass
    
    def get_parameter_summary(self, data):
        """Get summary of strategy parameters for display"""
        return {}
    
    def get_historical_statistics(self, data):
        """Get historical statistics of strategy indicators"""
        return {}
    
    def add_to_chart(self, fig, data):
        """Add strategy-specific indicators to chart"""
        pass

# Strategy Implementations

class EMACrossoverStrategy(BaseStrategy):
    """EMA/SMA Crossover Strategy with Enhanced Options"""
    
    def calculate_indicators(self, data):
        ma_type1 = self.params.get('ma_type1', 'EMA')
        ma_type2 = self.params.get('ma_type2', 'EMA')
        period1 = self.params.get('period1', 9)
        period2 = self.params.get('period2', 20)
        
        if ma_type1 == 'EMA':
            data['MA1'] = data['Close'].ewm(span=period1, adjust=False).mean()
        else:
            data['MA1'] = data['Close'].rolling(window=period1).mean()
        
        if ma_type2 == 'EMA':
            data['MA2'] = data['Close'].ewm(span=period2, adjust=False).mean()
        else:
            data['MA2'] = data['Close'].rolling(window=period2).mean()
        
        # Calculate ATR for ATR-based candle confirmation
        data['ATR'] = calculate_atr(data)
        
        return data
    
    def generate_signal(self, data):
        if len(data) < 3:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        
        # Check for crossover
        current_ma1 = data['MA1'].iloc[-1]
        current_ma2 = data['MA2'].iloc[-1]
        prev_ma1 = data['MA1'].iloc[-2]
        prev_ma2 = data['MA2'].iloc[-2]
        
        crossover_type = self.params.get('crossover_type', 'simple')
        
        bullish_signal = False
        bearish_signal = False
        
        # Calculate crossover angle
        crossover_angle = calculate_crossover_angle(current_ma1, prev_ma1, current_ma2, prev_ma2)
        min_angle = self.params.get('min_crossover_angle', 0)
        
        # Basic crossover detection
        bullish_crossover = (prev_ma1 <= prev_ma2) and (current_ma1 > current_ma2)
        bearish_crossover = (prev_ma1 >= prev_ma2) and (current_ma1 < current_ma2)
        
        # Check angle requirement
        angle_met = crossover_angle >= min_angle
        
        if crossover_type == 'simple':
            # Simple crossover with angle check
            bullish_signal = bullish_crossover and angle_met
            bearish_signal = bearish_crossover and angle_met
            
        elif crossover_type in ['auto_strong_candle', 'atr_strong_candle', 'custom_candle']:
            current_close = data['Close'].iloc[-1]
            current_open = data['Open'].iloc[-1]
            candle_body = abs(current_close - current_open)
            
            strong_candle = False
            
            if crossover_type == 'auto_strong_candle':
                # Auto: 1.5x average body size
                avg_body = abs(data['Close'] - data['Open']).tail(20).mean()
                strong_candle = candle_body > (avg_body * 1.5)
            
            elif crossover_type == 'atr_strong_candle':
                # ATR-based: candle body > ATR * multiplier
                atr_multiplier = self.params.get('atr_multiplier', 1.0)
                current_atr = data['ATR'].iloc[-1]
                strong_candle = candle_body > (current_atr * atr_multiplier)
            
            elif crossover_type == 'custom_candle':
                # Custom size
                custom_size = self.params.get('custom_candle_size', 10)
                strong_candle = candle_body > custom_size
            
            if bullish_crossover and angle_met:
                bullish_signal = strong_candle and (current_close > current_ma1)
            
            if bearish_crossover and angle_met:
                bearish_signal = strong_candle and (current_close < current_ma1)
        
        signal_data = {
            'MA1': current_ma1,
            'MA2': current_ma2,
            'MA1_prev': prev_ma1,
            'MA2_prev': prev_ma2,
            'Crossover_Angle': crossover_angle,
            'Angle_Met': angle_met
        }
        
        return bullish_signal, bearish_signal, signal_data
    
    def get_parameter_summary(self, data):
        data = self.calculate_indicators(data)
        current_ma1 = data['MA1'].iloc[-1]
        current_ma2 = data['MA2'].iloc[-1]
        prev_ma1 = data['MA1'].iloc[-2]
        prev_ma2 = data['MA2'].iloc[-2]
        
        angle = calculate_crossover_angle(current_ma1, prev_ma1, current_ma2, prev_ma2)
        
        return {
            'Current MA1': f"{current_ma1:.2f}",
            'Current MA2': f"{current_ma2:.2f}",
            'MA1 Avg (10)': f"{data['MA1'].tail(10).mean():.2f}",
            'MA2 Avg (10)': f"{data['MA2'].tail(10).mean():.2f}",
            'Crossover Angle': f"{angle:.1f}°",
            'Current ATR': f"{data['ATR'].iloc[-1]:.2f}"
        }
    
    def get_historical_statistics(self, data):
        data = self.calculate_indicators(data)
        
        return {
            'MA1 Statistics': {
                'Mean': f"{data['MA1'].mean():.2f}",
                'Std Dev': f"{data['MA1'].std():.2f}",
                'Min': f"{data['MA1'].min():.2f}",
                'Max': f"{data['MA1'].max():.2f}",
                '25th %ile': f"{data['MA1'].quantile(0.25):.2f}",
                '75th %ile': f"{data['MA1'].quantile(0.75):.2f}"
            },
            'MA2 Statistics': {
                'Mean': f"{data['MA2'].mean():.2f}",
                'Std Dev': f"{data['MA2'].std():.2f}",
                'Min': f"{data['MA2'].min():.2f}",
                'Max': f"{data['MA2'].max():.2f}",
                '25th %ile': f"{data['MA2'].quantile(0.25):.2f}",
                '75th %ile': f"{data['MA2'].quantile(0.75):.2f}"
            },
            'ATR Statistics': {
                'Current': f"{data['ATR'].iloc[-1]:.2f}",
                'Mean': f"{data['ATR'].mean():.2f}",
                'Min': f"{data['ATR'].min():.2f}",
                'Max': f"{data['ATR'].max():.2f}"
            },
            'Candle Body Statistics': {
                'Avg Body': f"{abs(data['Close'] - data['Open']).mean():.2f}",
                'Max Body': f"{abs(data['Close'] - data['Open']).max():.2f}",
                'Min Body': f"{abs(data['Close'] - data['Open']).min():.2f}"
            }
        }
    
    def add_to_chart(self, fig, data):
        data = self.calculate_indicators(data)
        ma_type1 = self.params.get('ma_type1', 'EMA')
        ma_type2 = self.params.get('ma_type2', 'EMA')
        period1 = self.params.get('period1', 9)
        period2 = self.params.get('period2', 20)
        
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['MA1'], 
            name=f"{ma_type1}{period1}", 
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['MA2'], 
            name=f"{ma_type2}{period2}", 
            line=dict(color='orange', width=2)
        ))

class ElliottWaveStrategy(BaseStrategy):
    """Elliott Wave Strategy - Simplified 5-Wave Pattern Detection"""
    
    def find_waves(self, data):
        """Find potential Elliott Wave patterns"""
        if len(data) < 50:
            return None, []
        
        prices = data['Close'].values
        
        # Find local extrema with a reasonable order
        order = max(5, len(data) // 20)
        peaks = argrelextrema(prices, np.greater, order=order)[0]
        troughs = argrelextrema(prices, np.less, order=order)[0]
        
        # Combine and sort extrema
        extrema = sorted(list(peaks) + list(troughs))
        
        if len(extrema) < 5:
            return None, []
        
        # Look for 5-wave impulse pattern
        # Wave 1: Up, Wave 2: Down, Wave 3: Up (longest), Wave 4: Down, Wave 5: Up
        waves = []
        
        for i in range(len(extrema) - 4):
            try:
                idx = [extrema[i], extrema[i+1], extrema[i+2], extrema[i+3], extrema[i+4]]
                wave_points = [(data.index[j], prices[j]) for j in idx]
                
                # Check if pattern matches impulse wave criteria
                # Wave 1: trough to peak
                # Wave 2: peak to trough
                # Wave 3: trough to peak (should be longest)
                # Wave 4: peak to trough
                # Wave 5: trough to peak
                
                if i % 2 == 0:  # Starting from trough
                    wave1 = prices[idx[1]] - prices[idx[0]]
                    wave2 = prices[idx[1]] - prices[idx[2]]
                    wave3 = prices[idx[3]] - prices[idx[2]]
                    wave4 = prices[idx[3]] - prices[idx[4]]
                    
                    # Impulse wave rules
                    if (wave1 > 0 and wave2 > 0 and wave3 > 0 and wave4 > 0 and
                        wave3 > wave1 and wave3 > abs(wave2)):
                        waves.append({
                            'indices': idx,
                            'points': wave_points,
                            'wave_sizes': [wave1, wave2, wave3, wave4]
                        })
            except:
                continue
        
        if len(waves) > 0:
            # Return most recent wave pattern
            return waves[-1], extrema
        
        return None, extrema
    
    def calculate_indicators(self, data):
        wave_pattern, extrema = self.find_waves(data)
        data['WavePattern'] = None
        data['Extrema'] = False
        
        if extrema:
            for idx in extrema:
                if idx < len(data):
                    data.iloc[idx, data.columns.get_loc('Extrema')] = True
        
        return data
    
    def generate_signal(self, data):
        if len(data) < 50:
            return False, False, {}
        
        wave_pattern, extrema = self.find_waves(data)
        
        bullish_signal = False
        bearish_signal = False
        
        if wave_pattern:
            # Check if we're at the end of wave 5 (potential reversal)
            last_wave_idx = wave_pattern['indices'][-1]
            
            # If we're within 5 candles of the last wave point
            if len(data) - last_wave_idx <= 5:
                # Wave 5 completion suggests potential reversal
                current_price = data['Close'].iloc[-1]
                wave5_start_price = data['Close'].iloc[wave_pattern['indices'][-2]]
                
                if current_price > wave5_start_price:
                    # Completed upward wave 5, potential short
                    bearish_signal = True
                else:
                    # Completed downward wave 5, potential long
                    bullish_signal = True
        
        signal_data = {
            'Wave_Pattern_Found': wave_pattern is not None,
            'Extrema_Count': len(extrema),
            'Wave_Info': wave_pattern if wave_pattern else {}
        }
        
        return bullish_signal, bearish_signal, signal_data
    
    def get_parameter_summary(self, data):
        wave_pattern, extrema = self.find_waves(data)
        
        summary = {
            'Extrema Points': len(extrema),
            'Wave Pattern': 'Found' if wave_pattern else 'Not Found'
        }
        
        if wave_pattern:
            summary['Wave 1 Size'] = f"{wave_pattern['wave_sizes'][0]:.2f}"
            summary['Wave 3 Size'] = f"{wave_pattern['wave_sizes'][2]:.2f}"
        
        return summary
    
    def get_historical_statistics(self, data):
        wave_pattern, extrema = self.find_waves(data)
        
        stats = {
            'Wave Analysis': {
                'Total Extrema': len(extrema),
                'Pattern Detected': 'Yes' if wave_pattern else 'No'
            }
        }
        
        if wave_pattern:
            stats['Current Wave Pattern'] = {
                'Wave 1': f"{wave_pattern['wave_sizes'][0]:.2f}",
                'Wave 2': f"{wave_pattern['wave_sizes'][1]:.2f}",
                'Wave 3': f"{wave_pattern['wave_sizes'][2]:.2f}",
                'Wave 4': f"{wave_pattern['wave_sizes'][3]:.2f}"
            }
        
        return stats
    
    def add_to_chart(self, fig, data):
        wave_pattern, extrema = self.find_waves(data)
        
        # Mark extrema points
        if extrema:
            extrema_prices = [data['Close'].iloc[idx] for idx in extrema if idx < len(data)]
            extrema_times = [data.index[idx] for idx in extrema if idx < len(data)]
            
            fig.add_trace(go.Scatter(
                x=extrema_times,
                y=extrema_prices,
                mode='markers',
                name='Wave Points',
                marker=dict(size=8, color='purple', symbol='diamond')
            ))
        
        # Draw wave pattern if found
        if wave_pattern:
            wave_times = [point[0] for point in wave_pattern['points']]
            wave_prices = [point[1] for point in wave_pattern['points']]
            
            fig.add_trace(go.Scatter(
                x=wave_times,
                y=wave_prices,
                mode='lines+markers+text',
                name='Elliott Waves',
                line=dict(color='red', width=3, dash='dash'),
                marker=dict(size=12, color='red'),
                text=['W1', 'W2', 'W3', 'W4', 'W5'],
                textposition='top center'
            ))

class PairRatioStrategy(BaseStrategy):
    """Pair Ratio Trading with Z-Score"""
    
    def calculate_indicators(self, data, data2):
        # Align the two datasets
        common_index = data.index.intersection(data2.index)
        if len(common_index) < 20:
            return None
        
        price1 = data.loc[common_index, 'Close']
        price2 = data2.loc[common_index, 'Close']
        
        ratio = price1 / price2
        mean_ratio = ratio.rolling(window=20).mean()
        std_ratio = ratio.rolling(window=20).std()
        zscore = (ratio - mean_ratio) / std_ratio
        
        result = data.loc[common_index].copy()
        result['Ratio'] = ratio
        result['ZScore'] = zscore
        result['MeanRatio'] = mean_ratio
        
        return result
    
    def generate_signal(self, data, data2=None):
        if data2 is None:
            return False, False, {}
        
        result = self.calculate_indicators(data, data2)
        if result is None or len(result) < 2:
            return False, False, {}
        
        threshold = self.params.get('zscore_threshold', 2.0)
        current_zscore = result['ZScore'].iloc[-1]
        
        bullish_signal = current_zscore < -threshold
        bearish_signal = current_zscore > threshold
        
        signal_data = {
            'ZScore': current_zscore,
            'Ratio': result['Ratio'].iloc[-1],
            'Threshold': threshold
        }
        
        return bullish_signal, bearish_signal, signal_data
    
    def get_parameter_summary(self, data, data2=None):
        if data2 is None:
            return {}
        result = self.calculate_indicators(data, data2)
        if result is None:
            return {}
        return {
            'Current ZScore': f"{result['ZScore'].iloc[-1]:.2f}",
            'Current Ratio': f"{result['Ratio'].iloc[-1]:.4f}",
            'Mean Ratio': f"{result['Ratio'].tail(20).mean():.4f}",
        }
    
    def get_historical_statistics(self, data, data2=None):
        if data2 is None:
            return {}
        
        result = self.calculate_indicators(data, data2)
        if result is None:
            return {}
        
        return {
            'ZScore Statistics': {
                'Current': f"{result['ZScore'].iloc[-1]:.2f}",
                'Mean': f"{result['ZScore'].mean():.2f}",
                'Std Dev': f"{result['ZScore'].std():.2f}",
                'Min': f"{result['ZScore'].min():.2f}",
                'Max': f"{result['ZScore'].max():.2f}"
            },
            'Ratio Statistics': {
                'Current': f"{result['Ratio'].iloc[-1]:.4f}",
                'Mean': f"{result['Ratio'].mean():.4f}",
                'Min': f"{result['Ratio'].min():.4f}",
                'Max': f"{result['Ratio'].max():.4f}"
            }
        }
    
    def add_to_chart(self, fig, data, data2=None):
        if data2 is None:
            return
        
        result = self.calculate_indicators(data, data2)
        if result is None:
            return
        
        fig.add_trace(go.Scatter(
            x=result.index,
            y=result['MeanRatio'] * data.loc[result.index, 'Close'] / result['Ratio'],
            name='Mean Ratio Line',
            line=dict(color='green', width=2, dash='dash')
        ))

class RSIDivergenceStrategy(BaseStrategy):
    """RSI with Divergence Detection"""
    
    def calculate_rsi(self, data, period=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def detect_divergence(self, data):
        # Find local extrema
        if len(data) < 30:
            return False, False
        
        prices = data['Close'].values
        rsi = data['RSI'].values
        
        # Find peaks and troughs
        price_peaks = argrelextrema(prices, np.greater, order=5)[0]
        price_troughs = argrelextrema(prices, np.less, order=5)[0]
        rsi_peaks = argrelextrema(rsi, np.greater, order=5)[0]
        rsi_troughs = argrelextrema(rsi, np.less, order=5)[0]
        
        bullish_div = False
        bearish_div = False
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
            if price_troughs[-1] > 10 and price_troughs[-2] > 10:
                if prices[price_troughs[-1]] < prices[price_troughs[-2]]:
                    closest_rsi_troughs = [t for t in rsi_troughs if t >= price_troughs[-2]]
                    if len(closest_rsi_troughs) >= 2:
                        if rsi[closest_rsi_troughs[-1]] > rsi[closest_rsi_troughs[-2]]:
                            bullish_div = True
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            if price_peaks[-1] > 10 and price_peaks[-2] > 10:
                if prices[price_peaks[-1]] > prices[price_peaks[-2]]:
                    closest_rsi_peaks = [p for p in rsi_peaks if p >= price_peaks[-2]]
                    if len(closest_rsi_peaks) >= 2:
                        if rsi[closest_rsi_peaks[-1]] < rsi[closest_rsi_peaks[-2]]:
                            bearish_div = True
        
        return bullish_div, bearish_div
    
    def calculate_indicators(self, data):
        period = self.params.get('rsi_period', 14)
        data['RSI'] = self.calculate_rsi(data, period)
        return data
    
    def generate_signal(self, data):
        if len(data) < 30:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        current_rsi = data['RSI'].iloc[-1]
        
        use_divergence = self.params.get('use_divergence', True)
        
        bullish_signal = False
        bearish_signal = False
        
        if use_divergence:
            bullish_div, bearish_div = self.detect_divergence(data)
            bullish_signal = bullish_div or (current_rsi < 30)
            bearish_signal = bearish_div or (current_rsi > 70)
        else:
            bullish_signal = current_rsi < 30
            bearish_signal = current_rsi > 70
        
        signal_data = {
            'RSI': current_rsi,
            'Oversold': current_rsi < 30,
            'Overbought': current_rsi > 70
        }
        
        return bullish_signal, bearish_signal, signal_data
    
    def get_parameter_summary(self, data):
        data = self.calculate_indicators(data)
        return {
            'Current RSI': f"{data['RSI'].iloc[-1]:.2f}",
            'RSI Avg (10)': f"{data['RSI'].tail(10).mean():.2f}",
            'RSI Min': f"{data['RSI'].tail(20).min():.2f}",
            'RSI Max': f"{data['RSI'].tail(20).max():.2f}",
        }
    
    def get_historical_statistics(self, data):
        data = self.calculate_indicators(data)
        
        return {
            'RSI Statistics': {
                'Current': f"{data['RSI'].iloc[-1]:.2f}",
                'Mean': f"{data['RSI'].mean():.2f}",
                'Std Dev': f"{data['RSI'].std():.2f}",
                'Min': f"{data['RSI'].min():.2f}",
                'Max': f"{data['RSI'].max():.2f}",
                'Oversold Count': f"{(data['RSI'] < 30).sum()}",
                'Overbought Count': f"{(data['RSI'] > 70).sum()}"
            }
        }
    
    def add_to_chart(self, fig, data):
        # RSI is typically shown in a separate subplot
        pass

class FibonacciRetracementStrategy(BaseStrategy):
    """Fibonacci Retracement Strategy"""
    
    def calculate_indicators(self, data):
        lookback = self.params.get('lookback', 50)
        recent_data = data.tail(lookback)
        
        swing_high = recent_data['High'].max()
        swing_low = recent_data['Low'].min()
        diff = swing_high - swing_low
        
        fib_levels = {
            '0%': swing_low,
            '23.6%': swing_low + 0.236 * diff,
            '38.2%': swing_low + 0.382 * diff,
            '50%': swing_low + 0.5 * diff,
            '61.8%': swing_low + 0.618 * diff,
            '78.6%': swing_low + 0.786 * diff,
            '100%': swing_high
        }
        
        data['SwingHigh'] = swing_high
        data['SwingLow'] = swing_low
        for level, value in fib_levels.items():
            data[f'Fib_{level}'] = value
        
        return data, fib_levels
    
    def generate_signal(self, data):
        if len(data) < 20:
            return False, False, {}
        
        data, fib_levels = self.calculate_indicators(data)
        current_price = data['Close'].iloc[-1]
        tolerance = self.params.get('tolerance', 0.005)
        
        key_levels = [fib_levels['38.2%'], fib_levels['50%'], fib_levels['61.8%']]
        
        bullish_signal = False
        bearish_signal = False
        near_level = None
        
        for level_val in key_levels:
            if abs(current_price - level_val) / level_val < tolerance:
                near_level = level_val
                # Bullish if approaching from below
                if data['Close'].iloc[-2] < level_val and current_price >= level_val:
                    bullish_signal = True
                # Bearish if approaching from above
                elif data['Close'].iloc[-2] > level_val and current_price <= level_val:
                    bearish_signal = True
                break
        
        signal_data = {
            'Fib_Levels': fib_levels,
            'Near_Level': near_level,
            'Current_Price': current_price
        }
        
        return bullish_signal, bearish_signal, signal_data
    
    def get_parameter_summary(self, data):
        data, fib_levels = self.calculate_indicators(data)
        return {
            'Swing High': f"{fib_levels['100%']:.2f}",
            'Swing Low': f"{fib_levels['0%']:.2f}",
            'Fib 50%': f"{fib_levels['50%']:.2f}",
            'Fib 61.8%': f"{fib_levels['61.8%']:.2f}",
        }
    
    def get_historical_statistics(self, data):
        data, fib_levels = self.calculate_indicators(data)
        
        return {
            'Fibonacci Levels': {
                '0% (Low)': f"{fib_levels['0%']:.2f}",
                '23.6%': f"{fib_levels['23.6%']:.2f}",
                '38.2%': f"{fib_levels['38.2%']:.2f}",
                '50%': f"{fib_levels['50%']:.2f}",
                '61.8%': f"{fib_levels['61.8%']:.2f}",
                '78.6%': f"{fib_levels['78.6%']:.2f}",
                '100% (High)': f"{fib_levels['100%']:.2f}"
            },
            'Range': {
                'Total Range': f"{fib_levels['100%'] - fib_levels['0%']:.2f}",
                'Current from Low': f"{data['Close'].iloc[-1] - fib_levels['0%']:.2f}",
                'Current from High': f"{fib_levels['100%'] - data['Close'].iloc[-1]:.2f}"
            }
        }
    
    def add_to_chart(self, fig, data):
        data, fib_levels = self.calculate_indicators(data)
        
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']
        for i, (level_name, level_value) in enumerate(fib_levels.items()):
            fig.add_hline(
                y=level_value, 
                line_dash="dot", 
                annotation_text=f"Fib {level_name}", 
                line_color=colors[i % len(colors)], 
                opacity=0.6
            )

class ZScoreMeanReversionStrategy(BaseStrategy):
    """Z-Score Mean Reversion Strategy"""
    
    def calculate_indicators(self, data):
        window = self.params.get('window', 20)
        data['Mean'] = data['Close'].rolling(window=window).mean()
        data['Std'] = data['Close'].rolling(window=window).std()
        data['ZScore'] = (data['Close'] - data['Mean']) / data['Std']
        return data
    
    def generate_signal(self, data):
        if len(data) < 21:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        threshold = self.params.get('threshold', 2.0)
        current_zscore = data['ZScore'].iloc[-1]
        
        bullish_signal = current_zscore < -threshold
        bearish_signal = current_zscore > threshold
        
        signal_data = {
            'ZScore': current_zscore,
            'Mean': data['Mean'].iloc[-1],
            'Threshold': threshold
        }
        
        return bullish_signal, bearish_signal, signal_data
    
    def get_parameter_summary(self, data):
        data = self.calculate_indicators(data)
        return {
            'Current ZScore': f"{data['ZScore'].iloc[-1]:.2f}",
            'Mean Price': f"{data['Mean'].iloc[-1]:.2f}",
            'Std Dev': f"{data['Std'].iloc[-1]:.2f}",
        }
    
    def get_historical_statistics(self, data):
        data = self.calculate_indicators(data)
        
        return {
            'ZScore Statistics': {
                'Current': f"{data['ZScore'].iloc[-1]:.2f}",
                'Mean': f"{data['ZScore'].mean():.2f}",
                'Std Dev': f"{data['ZScore'].std():.2f}",
                'Min': f"{data['ZScore'].min():.2f}",
                'Max': f"{data['ZScore'].max():.2f}"
            },
            'Price vs Mean': {
                'Current Price': f"{data['Close'].iloc[-1]:.2f}",
                'Mean Price': f"{data['Mean'].iloc[-1]:.2f}",
                'Deviation': f"{data['Close'].iloc[-1] - data['Mean'].iloc[-1]:.2f}"
            }
        }
    
    def add_to_chart(self, fig, data):
        data = self.calculate_indicators(data)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Mean'],
            name='Mean',
            line=dict(color='purple', width=2, dash='dash')
        ))

class BreakoutVolumeStrategy(BaseStrategy):
    """Breakout Strategy with Volume Confirmation"""
    
    def calculate_indicators(self, data):
        period = self.params.get('period', 20)
        data['Upper'] = data['High'].rolling(window=period).max()
        data['Lower'] = data['Low'].rolling(window=period).min()
        
        if 'Volume' in data.columns:
            data['AvgVolume'] = data['Volume'].rolling(window=period).mean()
        
        return data
    
    def generate_signal(self, data):
        if len(data) < 21:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        current_close = data['Close'].iloc[-1]
        upper_band = data['Upper'].iloc[-2]
        lower_band = data['Lower'].iloc[-2]
        
        volume_confirm = True
        if 'Volume' in data.columns:
            volume_multiplier = self.params.get('volume_multiplier', 1.5)
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['AvgVolume'].iloc[-1]
            volume_confirm = current_volume > (avg_volume * volume_multiplier)
        
        bullish_signal = (current_close > upper_band) and volume_confirm
        bearish_signal = (current_close < lower_band) and volume_confirm
        
        signal_data = {
            'Upper_Band': upper_band,
            'Lower_Band': lower_band,
            'Volume_Confirmed': volume_confirm
        }
        
        return bullish_signal, bearish_signal, signal_data
    
    def get_parameter_summary(self, data):
        data = self.calculate_indicators(data)
        summary = {
            'Upper Band': f"{data['Upper'].iloc[-1]:.2f}",
            'Lower Band': f"{data['Lower'].iloc[-1]:.2f}",
        }
        if 'Volume' in data.columns:
            summary['Avg Volume'] = f"{data['AvgVolume'].iloc[-1]:.0f}"
            summary['Current Volume'] = f"{data['Volume'].iloc[-1]:.0f}"
        return summary
    
    def get_historical_statistics(self, data):
        data = self.calculate_indicators(data)
        
        stats = {
            'Channel Statistics': {
                'Upper Band': f"{data['Upper'].iloc[-1]:.2f}",
                'Lower Band': f"{data['Lower'].iloc[-1]:.2f}",
                'Channel Width': f"{data['Upper'].iloc[-1] - data['Lower'].iloc[-1]:.2f}",
                'Avg Width': f"{(data['Upper'] - data['Lower']).mean():.2f}"
            }
        }
        
        if 'Volume' in data.columns:
            stats['Volume Statistics'] = {
                'Current': f"{data['Volume'].iloc[-1]:.0f}",
                'Average': f"{data['AvgVolume'].iloc[-1]:.0f}",
                'Max': f"{data['Volume'].max():.0f}",
                'Min': f"{data['Volume'].min():.0f}"
            }
        
        return stats
    
    def add_to_chart(self, fig, data):
        data = self.calculate_indicators(data)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Upper'],
            name='Upper Channel',
            line=dict(color='red', width=1, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Lower'],
            name='Lower Channel',
            line=dict(color='green', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.1)'
        ))

class SimpleBuyStrategy(BaseStrategy):
    """Simple Buy Strategy - Always generates buy signal"""
    
    def calculate_indicators(self, data):
        return data
    
    def generate_signal(self, data):
        bullish_signal = True
        bearish_signal = False
        signal_data = {'Type': 'Simple Buy'}
        return bullish_signal, bearish_signal, signal_data
    
    def get_parameter_summary(self, data):
        return {'Strategy': 'Simple Buy'}
    
    def get_historical_statistics(self, data):
        return {'Info': {'Type': 'Immediate Buy'}}
    
    def add_to_chart(self, fig, data):
        pass

class SimpleSellStrategy(BaseStrategy):
    """Simple Sell Strategy - Always generates sell signal"""
    
    def calculate_indicators(self, data):
        return data
    
    def generate_signal(self, data):
        bullish_signal = False
        bearish_signal = True
        signal_data = {'Type': 'Simple Sell'}
        return bullish_signal, bearish_signal, signal_data
    
    def get_parameter_summary(self, data):
        return {'Strategy': 'Simple Sell'}
    
    def get_historical_statistics(self, data):
        return {'Info': {'Type': 'Immediate Sell'}}
    
    def add_to_chart(self, fig, data):
        pass

# Trading System Class
class TradingSystem:
    def __init__(self, ticker, timeframe, period, strategy, quantity, sl_config, target_config, ticker2=None):
        self.ticker = ticker
        self.ticker2 = ticker2
        self.timeframe = timeframe
        self.period = period
        self.strategy = strategy
        self.quantity = quantity
        self.sl_config = sl_config
        self.target_config = target_config
    
    def fetch_data(self):
        """Fetch data with rate limiting and error handling"""
        try:
            time.sleep(1.5)  # Rate limiting
            
            ticker_obj = yf.Ticker(self.ticker)
            data = ticker_obj.history(period=self.period, interval=self.timeframe)
            
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Convert timezone to IST
            if data.index.tz is not None:
                data.index = data.index.tz_convert(IST)
            else:
                data.index = data.index.tz_localize('UTC').tz_convert(IST)
            
            if len(data) == 0:
                return None, None
            
            # Fetch second ticker if needed
            data2 = None
            if self.ticker2:
                time.sleep(1.5)
                ticker_obj2 = yf.Ticker(self.ticker2)
                data2 = ticker_obj2.history(period=self.period, interval=self.timeframe)
                
                if isinstance(data2.columns, pd.MultiIndex):
                    data2.columns = data2.columns.get_level_values(0)
                
                if data2.index.tz is not None:
                    data2.index = data2.index.tz_convert(IST)
                else:
                    data2.index = data2.index.tz_localize('UTC').tz_convert(IST)
            
            return data, data2
        
        except Exception as e:
            add_log(f"Error fetching data: {str(e)}")
            return None, None
    
    def calculate_sl_target(self, entry_price, position_type, data=None):
        """Calculate stop loss and target based on configuration"""
        sl_type = self.sl_config['type']
        target_type = self.target_config['type']
        
        sl_price = None
        target_price = None
        
        # Calculate Stop Loss
        if sl_type == 'Custom Points':
            points = self.sl_config.get('points', 10)
            sl_price = entry_price - points if position_type == 'LONG' else entry_price + points
        
        elif sl_type == 'Percentage':
            pct = self.sl_config.get('percentage', 1.0)
            sl_price = entry_price * (1 - pct/100) if position_type == 'LONG' else entry_price * (1 + pct/100)
        
        elif sl_type == 'ATR' and data is not None:
            atr_multiplier = self.sl_config.get('atr_multiplier', 1.5)
            atr = calculate_atr(data).iloc[-1]
            sl_price = entry_price - (atr * atr_multiplier) if position_type == 'LONG' else entry_price + (atr * atr_multiplier)
        
        # Calculate Target
        if target_type == 'Custom Points':
            points = self.target_config.get('points', 20)
            target_price = entry_price + points if position_type == 'LONG' else entry_price - points
        
        elif target_type == 'Percentage':
            pct = self.target_config.get('percentage', 2.0)
            target_price = entry_price * (1 + pct/100) if position_type == 'LONG' else entry_price * (1 - pct/100)
        
        elif target_type == 'Risk Reward':
            rr_ratio = self.target_config.get('rr_ratio', 2.0)
            if sl_price:
                risk = abs(entry_price - sl_price)
                target_price = entry_price + (risk * rr_ratio) if position_type == 'LONG' else entry_price - (risk * rr_ratio)
        
        return sl_price, target_price
    
    def update_trailing_sl(self, position, current_price):
        """Update trailing stop loss"""
        if self.sl_config['type'] != 'Trail SL':
            return position
        
        trail_points = self.sl_config.get('points', 10)
        position_type = position['type']
        
        if position_type == 'LONG':
            new_sl = current_price - trail_points
            if new_sl > position['sl']:
                old_sl = position['sl']
                position['sl'] = new_sl
                add_log(f"Trailing SL updated: {old_sl:.2f} → {new_sl:.2f}")
        else:
            new_sl = current_price + trail_points
            if new_sl < position['sl']:
                old_sl = position['sl']
                position['sl'] = new_sl
                add_log(f"Trailing SL updated: {old_sl:.2f} → {new_sl:.2f}")
        
        return position
    
    def update_trailing_target(self, position, current_price):
        """Update trailing target with tolerance for high volatility"""
        if self.target_config['type'] != 'Trail Target':
            return position
        
        trail_points = self.target_config.get('points', 20)
        position_type = position['type']
        
        # Add tolerance for high volatility assets
        tolerance = self.target_config.get('tolerance', 0.5)  # Default 0.5% tolerance
        
        if position_type == 'LONG':
            # Check if price reached near target (within tolerance)
            target_reached = current_price >= (position['target'] * (1 - tolerance/100))
            
            if target_reached:
                new_target = current_price + trail_points
                if new_target > position['target']:
                    old_target = position['target']
                    position['target'] = new_target
                    add_log(f"Trailing Target updated: {old_target:.2f} → {new_target:.2f}")
        else:
            target_reached = current_price <= (position['target'] * (1 + tolerance/100))
            
            if target_reached:
                new_target = current_price - trail_points
                if new_target < position['target']:
                    old_target = position['target']
                    position['target'] = new_target
                    add_log(f"Trailing Target updated: {old_target:.2f} → {new_target:.2f}")
        
        return position
    
    def check_exit_conditions(self, position, current_price, data, data2=None):
        """Check if exit conditions are met"""
        position_type = position['type']
        
        # Check SL hit
        if self.sl_config['type'] != 'Signal Based':
            if position_type == 'LONG' and current_price <= position['sl']:
                return True, 'Stop Loss Hit', current_price
            elif position_type == 'SHORT' and current_price >= position['sl']:
                return True, 'Stop Loss Hit', current_price
        
        # Check Target hit
        if self.target_config['type'] != 'Signal Based':
            if position_type == 'LONG' and current_price >= position['target']:
                return True, 'Target Hit', current_price
            elif position_type == 'SHORT' and current_price <= position['target']:
                return True, 'Target Hit', current_price
        
        # Check signal-based exit
        if self.sl_config['type'] == 'Signal Based' or self.target_config['type'] == 'Signal Based':
            if hasattr(self.strategy, 'generate_signal'):
                if isinstance(self.strategy, PairRatioStrategy) and data2 is not None:
                    bullish, bearish, _ = self.strategy.generate_signal(data, data2)
                else:
                    bullish, bearish, _ = self.strategy.generate_signal(data)
                
                if position_type == 'LONG' and bearish:
                    return True, 'Exit Signal', current_price
                elif position_type == 'SHORT' and bullish:
                    return True, 'Exit Signal', current_price
        
        return False, None, None
    
    def get_market_status(self, position, current_price):
        """Generate market status text"""
        pnl_pct = position['pnl_pct']
        position_type = position['type']
        
        status = []
        
        if pnl_pct > 0.5:
            status.append("✅ Strong momentum in favor!")
        elif pnl_pct > 0:
            status.append("↗️ Moving gradually in favor")
        else:
            status.append("⚠️ In loss - monitoring for reversal or SL hit")
        
        if self.sl_config['type'] == 'Trail SL':
            status.append("🔄 Trailing SL active")
        
        # Distance to SL and Target
        if position['sl']:
            sl_distance = abs(current_price - position['sl'])
            status.append(f"📍 Distance to SL: {sl_distance:.2f} points")
        
        if position['target']:
            target_distance = abs(position['target'] - current_price)
            status.append(f"🎯 Distance to Target: {target_distance:.2f} points")
        
        return " | ".join(status)
    
    def generate_trade_guidance(self, position, current_price, data, signal_data):
        """Generate 100-word trade guidance"""
        pnl_pct = position['pnl_pct']
        position_type = position['type']
        
        guidance = f"**Current Trade Analysis:**\n\n"
        
        # What was expected
        guidance += f"**Expected:** {self.strategy.name} signaled a {position_type} position at {position['entry_price']:.2f}. "
        
        # How it's moving
        if pnl_pct > 1:
            guidance += f"The trade is performing excellently with {pnl_pct:.2f}% profit. Price momentum is strong in your favor. "
        elif pnl_pct > 0:
            guidance += f"Price is moving favorably with {pnl_pct:.2f}% profit, though momentum is moderate. "
        else:
            guidance += f"Currently in drawdown at {pnl_pct:.2f}%. Price hasn't moved as expected yet. "
        
        # What to do
        if abs(pnl_pct) < 0.3:
            guidance += "**Action:** Hold position and let strategy play out. Avoid premature exits. "
        elif pnl_pct > 1:
            guidance += "**Action:** Consider partial profit booking or tightening SL to lock gains. "
        elif pnl_pct < -0.5:
            guidance += "**Action:** Stay disciplined. Honor your SL - don't move it away from price. "
        
        # What NOT to do
        guidance += "\n\n**Avoid:** Don't move SL away from price in panic. Don't exit prematurely on small fluctuations. Trust your strategy's logic. "
        
        # Strategy-specific advice
        if isinstance(self.strategy, EMACrossoverStrategy):
            ma1 = signal_data.get('MA1', 0)
            ma2 = signal_data.get('MA2', 0)
            if position_type == 'LONG':
                if current_price > ma1 > ma2:
                    guidance += "EMA alignment supports your position. "
                else:
                    guidance += "Watch for EMA alignment weakening. "
        elif isinstance(self.strategy, ElliottWaveStrategy):
            guidance += "Monitor for wave completion patterns. Elliott Waves can take time to fully develop. "
        
        return guidance
    
    def analyze_trade(self, trade):
        """AI-powered trade analysis"""
        pnl_pct = trade['pnl_pct']
        duration = trade['duration']
        
        analysis = {
            'performance': '',
            'exit_quality': '',
            'duration_insight': '',
            'recommendations': []
        }
        
        # Performance assessment
        if pnl_pct > 2:
            analysis['performance'] = "Excellent trade! Strong profit capture."
        elif pnl_pct > 0:
            analysis['performance'] = "Profitable trade with moderate gains."
        elif pnl_pct > -1:
            analysis['performance'] = "Small loss - acceptable within strategy parameters."
        else:
            analysis['performance'] = "Significant loss - review entry conditions."
        
        # Exit quality
        if trade['exit_reason'] == 'Target Hit':
            analysis['exit_quality'] = "Perfect exit - target achieved as planned."
        elif trade['exit_reason'] == 'Stop Loss Hit':
            if pnl_pct > -2:
                analysis['exit_quality'] = "Good risk management - SL protected capital."
            else:
                analysis['exit_quality'] = "SL was too wide - consider tighter stops."
        elif trade['exit_reason'] == 'Exit Signal':
            analysis['exit_quality'] = "Signal-based exit - strategy-driven decision."
        else:
            analysis['exit_quality'] = "Manual exit - ensure it aligned with strategy."
        
        # Duration insights
        duration_str = str(duration)
        if 'day' in duration_str:
            analysis['duration_insight'] = "Extended hold period - suitable for trend following."
        elif 'hour' in duration_str:
            analysis['duration_insight'] = "Moderate duration - good for intraday strategies."
        else:
            analysis['duration_insight'] = "Very short duration - scalping or quick reversal."
        
        # Recommendations
        if pnl_pct < 0:
            analysis['recommendations'].append("Review entry timing - may have entered too early/late")
            analysis['recommendations'].append("Check if market conditions matched strategy assumptions")
        
        if trade['exit_reason'] == 'Stop Loss Hit' and pnl_pct < -2:
            analysis['recommendations'].append("Consider tighter stop loss placement")
            analysis['recommendations'].append("Verify trend strength before entry")
        
        if pnl_pct > 2 and trade['exit_reason'] != 'Target Hit':
            analysis['recommendations'].append("Consider using trailing stops to lock profits")
        
        if len(analysis['recommendations']) == 0:
            analysis['recommendations'].append("Good trade execution - maintain consistency")
        
        return analysis

# Sidebar Configuration
st.sidebar.markdown("<div class='main-header'>⚙️ Configuration</div>", unsafe_allow_html=True)

# Asset Selection
st.sidebar.subheader("📊 Asset Selection")

preset_assets = {
    'NIFTY 50': '^NSEI',
    'Bank NIFTY': '^NSEBANK',
    'SENSEX': '^BSESN',
    'BTC-USD': 'BTC-USD',
    'ETH-USD': 'ETH-USD',
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'USD/INR': 'INR=X',
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X'
}

ticker_type = st.sidebar.radio("Select Type:", ['Preset Assets', 'Custom Ticker', 'Indian Stock'])

if ticker_type == 'Preset Assets':
    selected_asset = st.sidebar.selectbox("Choose Asset:", list(preset_assets.keys()))
    ticker = preset_assets[selected_asset]
elif ticker_type == 'Custom Ticker':
    ticker = st.sidebar.text_input("Enter Ticker Symbol:", "AAPL")
else:
    stock_name = st.sidebar.text_input("Enter Stock Name:", "RELIANCE")
    ticker = f"{stock_name}.NS"

st.sidebar.write(f"**Selected Ticker:** `{ticker}`")

# Timeframe and Period
st.sidebar.subheader("⏰ Timeframe & Period")

timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1wk', '1mo']
periods = ['1d', '5d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', '20y', '30y']

# Compatibility mapping
timeframe_period_map = {
    '1m': ['1d', '5d'],
    '5m': ['1d', '5d', '1mo'],
    '15m': ['1d', '5d', '1mo'],
    '30m': ['1d', '5d', '1mo'],
    '1h': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
    '2h': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
    '4h': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
    '1d': ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', '20y', '30y'],
    '1wk': ['1y', '2y', '5y', '10y', '20y', '30y'],
    '1mo': ['1y', '2y', '5y', '10y', '20y', '30y']
}

timeframe = st.sidebar.selectbox("Timeframe:", timeframes, index=4)
compatible_periods = timeframe_period_map.get(timeframe, periods)
period = st.sidebar.selectbox("Period:", compatible_periods)

# Strategy Selection
st.sidebar.subheader("🎯 Strategy Selection")

strategy_names = [
    'EMA/SMA Crossover',
    'Elliott Wave',
    'Pair Ratio Trading',
    'RSI + Divergence',
    'Fibonacci Retracement',
    'Z-Score Mean Reversion',
    'Breakout + Volume',
    'Simple Buy',
    'Simple Sell'
]

selected_strategy = st.sidebar.selectbox("Choose Strategy:", strategy_names)

# Strategy Parameters
strategy_params = {}
ticker2 = None

if selected_strategy == 'EMA/SMA Crossover':
    st.sidebar.markdown("**Strategy Parameters:**")
    ma_type1 = st.sidebar.selectbox("MA Type 1:", ['EMA', 'SMA'], key='ma1')
    period1 = st.sidebar.number_input("Period 1:", min_value=2, max_value=200, value=9)
    ma_type2 = st.sidebar.selectbox("MA Type 2:", ['EMA', 'SMA'], key='ma2')
    period2 = st.sidebar.number_input("Period 2:", min_value=2, max_value=200, value=20)
    
    crossover_type = st.sidebar.radio("Crossover Type:", 
                                      ['simple', 'auto_strong_candle'])
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []
if 'iteration_count' not in st.session_state:
    st.session_state.iteration_count = 0
if 'last_data' not in st.session_state:
    st.session_state.last_data = None

# Timezone
IST = pytz.timezone('Asia/Kolkata')

def get_ist_time():
    """Get current time in IST"""
    return datetime.now(IST)

def add_log(message):
    """Add timestamped log entry"""
    timestamp = get_ist_time().strftime("%Y-%m-%d %H:%M:%S IST")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.trade_log.insert(0, log_entry)
    if len(st.session_state.trade_log) > 100:
        st.session_state.trade_log = st.session_state.trade_log[:100]

# Base Strategy Class
class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, params=None):
        self.params = params or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def calculate_indicators(self, data):
        """Calculate strategy-specific indicators"""
        pass
    
    @abstractmethod
    def generate_signal(self, data):
        """Generate trading signal. Returns (bullish_signal, bearish_signal, signal_data)"""
        pass
    
    def get_parameter_summary(self, data):
        """Get summary of strategy parameters for display"""
        return {}

# Strategy Implementations

class EMACrossoverStrategy(BaseStrategy):
    """EMA/SMA Crossover Strategy"""
    
    def calculate_indicators(self, data):
        ma_type1 = self.params.get('ma_type1', 'EMA')
        ma_type2 = self.params.get('ma_type2', 'EMA')
        period1 = self.params.get('period1', 9)
        period2 = self.params.get('period2', 20)
        
        if ma_type1 == 'EMA':
            data['MA1'] = data['Close'].ewm(span=period1, adjust=False).mean()
        else:
            data['MA1'] = data['Close'].rolling(window=period1).mean()
        
        if ma_type2 == 'EMA':
            data['MA2'] = data['Close'].ewm(span=period2, adjust=False).mean()
        else:
            data['MA2'] = data['Close'].rolling(window=period2).mean()
        
        return data
    
    def generate_signal(self, data):
        if len(data) < 3:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        
        # Check for crossover
        current_ma1 = data['MA1'].iloc[-1]
        current_ma2 = data['MA2'].iloc[-1]
        prev_ma1 = data['MA1'].iloc[-2]
        prev_ma2 = data['MA2'].iloc[-2]
        
        crossover_type = self.params.get('crossover_type', 'simple')
        
        bullish_signal = False
        bearish_signal = False
        
        if crossover_type == 'simple':
            # Simple crossover
            bullish_signal = (prev_ma1 <= prev_ma2) and (current_ma1 > current_ma2)
            bearish_signal = (prev_ma1 >= prev_ma2) and (current_ma1 < current_ma2)
        else:
            # Crossover with strong candle confirmation
            current_close = data['Close'].iloc[-1]
            current_open = data['Open'].iloc[-1]
            candle_body = abs(current_close - current_open)
            avg_body = abs(data['Close'] - data['Open']).tail(20).mean()
            
            strong_candle = candle_body > (avg_body * 1.5)
            
            if (prev_ma1 <= prev_ma2) and (current_ma1 > current_ma2):
                bullish_signal = strong_candle and (current_close > current_ma1)
            
            if (prev_ma1 >= prev_ma2) and (current_ma1 < current_ma2):
                bearish_signal = strong_candle and (current_close < current_ma1)
        
        signal_data = {
            'MA1': current_ma1,
            'MA2': current_ma2,
            'MA1_prev': prev_ma1,
            'MA2_prev': prev_ma2
        }
        
        return bullish_signal, bearish_signal, signal_data
    
    def get_parameter_summary(self, data):
        data = self.calculate_indicators(data)
        return {
            'Current MA1': f"{data['MA1'].iloc[-1]:.2f}",
            'Current MA2': f"{data['MA2'].iloc[-1]:.2f}",
            'MA1 Avg (10)': f"{data['MA1'].tail(10).mean():.2f}",
            'MA2 Avg (10)': f"{data['MA2'].tail(10).mean():.2f}",
        }

class PairRatioStrategy(BaseStrategy):
    """Pair Ratio Trading with Z-Score"""
    
    def calculate_indicators(self, data, data2):
        # Align the two datasets
        common_index = data.index.intersection(data2.index)
        if len(common_index) < 20:
            return None
        
        price1 = data.loc[common_index, 'Close']
        price2 = data2.loc[common_index, 'Close']
        
        ratio = price1 / price2
        mean_ratio = ratio.rolling(window=20).mean()
        std_ratio = ratio.rolling(window=20).std()
        zscore = (ratio - mean_ratio) / std_ratio
        
        result = data.loc[common_index].copy()
        result['Ratio'] = ratio
        result['ZScore'] = zscore
        
        return result
    
    def generate_signal(self, data, data2=None):
        if data2 is None:
            return False, False, {}
        
        result = self.calculate_indicators(data, data2)
        if result is None or len(result) < 2:
            return False, False, {}
        
        threshold = self.params.get('zscore_threshold', 2.0)
        current_zscore = result['ZScore'].iloc[-1]
        
        bullish_signal = current_zscore < -threshold
        bearish_signal = current_zscore > threshold
        
        signal_data = {
            'ZScore': current_zscore,
            'Ratio': result['Ratio'].iloc[-1],
            'Threshold': threshold
        }
        
        return bullish_signal, bearish_signal, signal_data
    
    def get_parameter_summary(self, data, data2=None):
        if data2 is None:
            return {}
        result = self.calculate_indicators(data, data2)
        if result is None:
            return {}
        return {
            'Current ZScore': f"{result['ZScore'].iloc[-1]:.2f}",
            'Current Ratio': f"{result['Ratio'].iloc[-1]:.4f}",
            'Mean Ratio': f"{result['Ratio'].tail(20).mean():.4f}",
        }

class RSIDivergenceStrategy(BaseStrategy):
    """RSI with Divergence Detection"""
    
    def calculate_rsi(self, data, period=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def detect_divergence(self, data):
        # Find local extrema
        if len(data) < 30:
            return False, False
        
        prices = data['Close'].values
        rsi = data['RSI'].values
        
        # Find peaks and troughs
        price_peaks = argrelextrema(prices, np.greater, order=5)[0]
        price_troughs = argrelextrema(prices, np.less, order=5)[0]
        rsi_peaks = argrelextrema(rsi, np.greater, order=5)[0]
        rsi_troughs = argrelextrema(rsi, np.less, order=5)[0]
        
        bullish_div = False
        bearish_div = False
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
            if price_troughs[-1] > 10 and price_troughs[-2] > 10:
                if prices[price_troughs[-1]] < prices[price_troughs[-2]]:
                    closest_rsi_troughs = [t for t in rsi_troughs if t >= price_troughs[-2]]
                    if len(closest_rsi_troughs) >= 2:
                        if rsi[closest_rsi_troughs[-1]] > rsi[closest_rsi_troughs[-2]]:
                            bullish_div = True
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            if price_peaks[-1] > 10 and price_peaks[-2] > 10:
                if prices[price_peaks[-1]] > prices[price_peaks[-2]]:
                    closest_rsi_peaks = [p for p in rsi_peaks if p >= price_peaks[-2]]
                    if len(closest_rsi_peaks) >= 2:
                        if rsi[closest_rsi_peaks[-1]] < rsi[closest_rsi_peaks[-2]]:
                            bearish_div = True
        
        return bullish_div, bearish_div
    
    def calculate_indicators(self, data):
        period = self.params.get('rsi_period', 14)
        data['RSI'] = self.calculate_rsi(data, period)
        return data
    
    def generate_signal(self, data):
        if len(data) < 30:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        current_rsi = data['RSI'].iloc[-1]
        
        use_divergence = self.params.get('use_divergence', True)
        
        bullish_signal = False
        bearish_signal = False
        
        if use_divergence:
            bullish_div, bearish_div = self.detect_divergence(data)
            bullish_signal = bullish_div or (current_rsi < 30)
            bearish_signal = bearish_div or (current_rsi > 70)
        else:
            bullish_signal = current_rsi < 30
            bearish_signal = current_rsi > 70
        
        signal_data = {
            'RSI': current_rsi,
            'Oversold': current_rsi < 30,
            'Overbought': current_rsi > 70
        }
        
        return bullish_signal, bearish_signal, signal_data
    
    def get_parameter_summary(self, data):
        data = self.calculate_indicators(data)
        return {
            'Current RSI': f"{data['RSI'].iloc[-1]:.2f}",
            'RSI Avg (10)': f"{data['RSI'].tail(10).mean():.2f}",
            'RSI Min': f"{data['RSI'].tail(20).min():.2f}",
            'RSI Max': f"{data['RSI'].tail(20).max():.2f}",
        }

class FibonacciRetracementStrategy(BaseStrategy):
    """Fibonacci Retracement Strategy"""
    
    def calculate_indicators(self, data):
        lookback = self.params.get('lookback', 50)
        recent_data = data.tail(lookback)
        
        swing_high = recent_data['High'].max()
        swing_low = recent_data['Low'].min()
        diff = swing_high - swing_low
        
        fib_levels = {
            '0%': swing_low,
            '23.6%': swing_low + 0.236 * diff,
            '38.2%': swing_low + 0.382 * diff,
            '50%': swing_low + 0.5 * diff,
            '61.8%': swing_low + 0.618 * diff,
            '78.6%': swing_low + 0.786 * diff,
            '100%': swing_high
        }
        
        data['SwingHigh'] = swing_high
        data['SwingLow'] = swing_low
        for level, value in fib_levels.items():
            data[f'Fib_{level}'] = value
        
        return data, fib_levels
    
    def generate_signal(self, data):
        if len(data) < 20:
            return False, False, {}
        
        data, fib_levels = self.calculate_indicators(data)
        current_price = data['Close'].iloc[-1]
        tolerance = self.params.get('tolerance', 0.005)
        
        key_levels = [fib_levels['38.2%'], fib_levels['50%'], fib_levels['61.8%']]
        
        bullish_signal = False
        bearish_signal = False
        near_level = None
        
        for level_val in key_levels:
            if abs(current_price - level_val) / level_val < tolerance:
                near_level = level_val
                # Bullish if approaching from below
                if data['Close'].iloc[-2] < level_val and current_price >= level_val:
                    bullish_signal = True
                # Bearish if approaching from above
                elif data['Close'].iloc[-2] > level_val and current_price <= level_val:
                    bearish_signal = True
                break
        
        signal_data = {
            'Fib_Levels': fib_levels,
            'Near_Level': near_level,
            'Current_Price': current_price
        }
        
        return bullish_signal, bearish_signal, signal_data
    
    def get_parameter_summary(self, data):
        data, fib_levels = self.calculate_indicators(data)
        return {
            'Swing High': f"{fib_levels['100%']:.2f}",
            'Swing Low': f"{fib_levels['0%']:.2f}",
            'Fib 50%': f"{fib_levels['50%']:.2f}",
            'Fib 61.8%': f"{fib_levels['61.8%']:.2f}",
        }

class ZScoreMeanReversionStrategy(BaseStrategy):
    """Z-Score Mean Reversion Strategy"""
    
    def calculate_indicators(self, data):
        window = self.params.get('window', 20)
        data['Mean'] = data['Close'].rolling(window=window).mean()
        data['Std'] = data['Close'].rolling(window=window).std()
        data['ZScore'] = (data['Close'] - data['Mean']) / data['Std']
        return data
    
    def generate_signal(self, data):
        if len(data) < 21:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        threshold = self.params.get('threshold', 2.0)
        current_zscore = data['ZScore'].iloc[-1]
        
        bullish_signal = current_zscore < -threshold
        bearish_signal = current_zscore > threshold
        
        signal_data = {
            'ZScore': current_zscore,
            'Mean': data['Mean'].iloc[-1],
            'Threshold': threshold
        }
        
        return bullish_signal, bearish_signal, signal_data
    
    def get_parameter_summary(self, data):
        data = self.calculate_indicators(data)
        return {
            'Current ZScore': f"{data['ZScore'].iloc[-1]:.2f}",
            'Mean Price': f"{data['Mean'].iloc[-1]:.2f}",
            'Std Dev': f"{data['Std'].iloc[-1]:.2f}",
        }

class BreakoutVolumeStrategy(BaseStrategy):
    """Breakout Strategy with Volume Confirmation"""
    
    def calculate_indicators(self, data):
        period = self.params.get('period', 20)
        data['Upper'] = data['High'].rolling(window=period).max()
        data['Lower'] = data['Low'].rolling(window=period).min()
        
        if 'Volume' in data.columns:
            data['AvgVolume'] = data['Volume'].rolling(window=period).mean()
        
        return data
    
    def generate_signal(self, data):
        if len(data) < 21:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        current_close = data['Close'].iloc[-1]
        upper_band = data['Upper'].iloc[-2]
        lower_band = data['Lower'].iloc[-2]
        
        volume_confirm = True
        if 'Volume' in data.columns:
            volume_multiplier = self.params.get('volume_multiplier', 1.5)
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['AvgVolume'].iloc[-1]
            volume_confirm = current_volume > (avg_volume * volume_multiplier)
        
        bullish_signal = (current_close > upper_band) and volume_confirm
        bearish_signal = (current_close < lower_band) and volume_confirm
        
        signal_data = {
            'Upper_Band': upper_band,
            'Lower_Band': lower_band,
            'Volume_Confirmed': volume_confirm
        }
        
        return bullish_signal, bearish_signal, signal_data
    
    def get_parameter_summary(self, data):
        data = self.calculate_indicators(data)
        summary = {
            'Upper Band': f"{data['Upper'].iloc[-1]:.2f}",
            'Lower Band': f"{data['Lower'].iloc[-1]:.2f}",
        }
        if 'Volume' in data.columns:
            summary['Avg Volume'] = f"{data['AvgVolume'].iloc[-1]:.0f}"
        return summary

class SimpleBuyStrategy(BaseStrategy):
    """Simple Buy Strategy - Always generates buy signal"""
    
    def calculate_indicators(self, data):
        return data
    
    def generate_signal(self, data):
        # Always signal to buy on first iteration
        bullish_signal = True
        bearish_signal = False
        
        signal_data = {'Type': 'Simple Buy'}
        return bullish_signal, bearish_signal, signal_data
    
    def get_parameter_summary(self, data):
        return {'Strategy': 'Simple Buy'}

class SimpleSellStrategy(BaseStrategy):
    """Simple Sell Strategy - Always generates sell signal"""
    
    def calculate_indicators(self, data):
        return data
    
    def generate_signal(self, data):
        # Always signal to sell on first iteration
        bullish_signal = False
        bearish_signal = True
        
        signal_data = {'Type': 'Simple Sell'}
        return bullish_signal, bearish_signal, signal_data
    
    def get_parameter_summary(self, data):
        return {'Strategy': 'Simple Sell'}

# Trading System Class
class TradingSystem:
    def __init__(self, ticker, timeframe, period, strategy, quantity, sl_config, target_config, ticker2=None):
        self.ticker = ticker
        self.ticker2 = ticker2
        self.timeframe = timeframe
        self.period = period
        self.strategy = strategy
        self.quantity = quantity
        self.sl_config = sl_config
        self.target_config = target_config
    
    def fetch_data(self):
        """Fetch data with rate limiting and error handling"""
        try:
            time.sleep(1.5)  # Rate limiting
            
            ticker_obj = yf.Ticker(self.ticker)
            data = ticker_obj.history(period=self.period, interval=self.timeframe)
            
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Convert timezone to IST
            if data.index.tz is not None:
                data.index = data.index.tz_convert(IST)
            else:
                data.index = data.index.tz_localize('UTC').tz_convert(IST)
            
            if len(data) == 0:
                return None, None
            
            # Fetch second ticker if needed
            data2 = None
            if self.ticker2:
                time.sleep(1.5)
                ticker_obj2 = yf.Ticker(self.ticker2)
                data2 = ticker_obj2.history(period=self.period, interval=self.timeframe)
                
                if isinstance(data2.columns, pd.MultiIndex):
                    data2.columns = data2.columns.get_level_values(0)
                
                if data2.index.tz is not None:
                    data2.index = data2.index.tz_convert(IST)
                else:
                    data2.index = data2.index.tz_localize('UTC').tz_convert(IST)
            
            return data, data2
        
        except Exception as e:
            add_log(f"Error fetching data: {str(e)}")
            return None, None
    
    def calculate_sl_target(self, entry_price, position_type):
        """Calculate stop loss and target based on configuration"""
        sl_type = self.sl_config['type']
        target_type = self.target_config['type']
        
        sl_price = None
        target_price = None
        
        # Calculate Stop Loss
        if sl_type == 'Custom Points':
            points = self.sl_config.get('points', 10)
            sl_price = entry_price - points if position_type == 'LONG' else entry_price + points
        
        elif sl_type == 'Percentage':
            pct = self.sl_config.get('percentage', 1.0)
            sl_price = entry_price * (1 - pct/100) if position_type == 'LONG' else entry_price * (1 + pct/100)
        
        elif sl_type == 'ATR':
            # Will be calculated with data
            sl_price = None
        
        # Calculate Target
        if target_type == 'Custom Points':
            points = self.target_config.get('points', 20)
            target_price = entry_price + points if position_type == 'LONG' else entry_price - points
        
        elif target_type == 'Percentage':
            pct = self.target_config.get('percentage', 2.0)
            target_price = entry_price * (1 + pct/100) if position_type == 'LONG' else entry_price * (1 - pct/100)
        
        elif target_type == 'Risk Reward':
            rr_ratio = self.target_config.get('rr_ratio', 2.0)
            if sl_price:
                risk = abs(entry_price - sl_price)
                target_price = entry_price + (risk * rr_ratio) if position_type == 'LONG' else entry_price - (risk * rr_ratio)
        
        return sl_price, target_price
    
    def update_trailing_sl(self, position, current_price):
        """Update trailing stop loss"""
        if self.sl_config['type'] != 'Trail SL':
            return position
        
        trail_points = self.sl_config.get('points', 10)
        position_type = position['type']
        
        if position_type == 'LONG':
            new_sl = current_price - trail_points
            if new_sl > position['sl']:
                old_sl = position['sl']
                position['sl'] = new_sl
                add_log(f"Trailing SL updated: {old_sl:.2f} → {new_sl:.2f}")
        else:
            new_sl = current_price + trail_points
            if new_sl < position['sl']:
                old_sl = position['sl']
                position['sl'] = new_sl
                add_log(f"Trailing SL updated: {old_sl:.2f} → {new_sl:.2f}")
        
        return position
    
    def update_trailing_target(self, position, current_price):
        """Update trailing target with tolerance for high volatility"""
        if self.target_config['type'] != 'Trail Target':
            return position
        
        trail_points = self.target_config.get('points', 20)
        position_type = position['type']
        
        # Add tolerance for high volatility assets
        tolerance = self.target_config.get('tolerance', 0.5)  # Default 0.5% tolerance
        
        if position_type == 'LONG':
            # Check if price reached near target (within tolerance)
            target_reached = current_price >= (position['target'] * (1 - tolerance/100))
            
            if target_reached:
                new_target = current_price + trail_points
                if new_target > position['target']:
                    old_target = position['target']
                    position['target'] = new_target
                    add_log(f"Trailing Target updated: {old_target:.2f} → {new_target:.2f}")
        else:
            target_reached = current_price <= (position['target'] * (1 + tolerance/100))
            
            if target_reached:
                new_target = current_price - trail_points
                if new_target < position['target']:
                    old_target = position['target']
                    position['target'] = new_target
                    add_log(f"Trailing Target updated: {old_target:.2f} → {new_target:.2f}")
        
        return position
    
    def check_exit_conditions(self, position, current_price, data, data2=None):
        """Check if exit conditions are met"""
        position_type = position['type']
        
        # Check SL hit
        if self.sl_config['type'] != 'Signal Based':
            if position_type == 'LONG' and current_price <= position['sl']:
                return True, 'Stop Loss Hit', current_price
            elif position_type == 'SHORT' and current_price >= position['sl']:
                return True, 'Stop Loss Hit', current_price
        
        # Check Target hit
        if self.target_config['type'] != 'Signal Based':
            if position_type == 'LONG' and current_price >= position['target']:
                return True, 'Target Hit', current_price
            elif position_type == 'SHORT' and current_price <= position['target']:
                return True, 'Target Hit', current_price
        
        # Check signal-based exit
        if self.sl_config['type'] == 'Signal Based' or self.target_config['type'] == 'Signal Based':
            if hasattr(self.strategy, 'generate_signal'):
                if isinstance(self.strategy, PairRatioStrategy) and data2 is not None:
                    bullish, bearish, _ = self.strategy.generate_signal(data, data2)
                else:
                    bullish, bearish, _ = self.strategy.generate_signal(data)
                
                if position_type == 'LONG' and bearish:
                    return True, 'Exit Signal', current_price
                elif position_type == 'SHORT' and bullish:
                    return True, 'Exit Signal', current_price
        
        return False, None, None
    
    def get_market_status(self, position, current_price):
        """Generate market status text"""
        pnl_pct = position['pnl_pct']
        position_type = position['type']
        
        status = []
        
        if pnl_pct > 0.5:
            status.append("✅ Strong momentum in favor!")
        elif pnl_pct > 0:
            status.append("↗️ Moving gradually in favor")
        else:
            status.append("⚠️ In loss - monitoring for reversal or SL hit")
        
        if self.sl_config['type'] == 'Trail SL':
            status.append("🔄 Trailing SL active")
        
        # Distance to SL and Target
        if position['sl']:
            sl_distance = abs(current_price - position['sl'])
            status.append(f"📍 Distance to SL: {sl_distance:.2f} points")
        
        if position['target']:
            target_distance = abs(position['target'] - current_price)
            status.append(f"🎯 Distance to Target: {target_distance:.2f} points")
        
        return " | ".join(status)
    
    def generate_trade_guidance(self, position, current_price, data, signal_data):
        """Generate 100-word trade guidance"""
        pnl_pct = position['pnl_pct']
        position_type = position['type']
        
        guidance = f"**Current Trade Analysis:**\n\n"
        
        # What was expected
        guidance += f"**Expected:** {self.strategy.name} signaled a {position_type} position at {position['entry_price']:.2f}. "
        
        # How it's moving
        if pnl_pct > 1:
            guidance += f"The trade is performing excellently with {pnl_pct:.2f}% profit. Price momentum is strong in your favor. "
        elif pnl_pct > 0:
            guidance += f"Price is moving favorably with {pnl_pct:.2f}% profit, though momentum is moderate. "
        else:
            guidance += f"Currently in drawdown at {pnl_pct:.2f}%. Price hasn't moved as expected yet. "
        
        # What to do
        if abs(pnl_pct) < 0.3:
            guidance += "**Action:** Hold position and let strategy play out. Avoid premature exits. "
        elif pnl_pct > 1:
            guidance += "**Action:** Consider partial profit booking or tightening SL to lock gains. "
        elif pnl_pct < -0.5:
            guidance += "**Action:** Stay disciplined. Honor your SL - don't move it away from price. "
        
        # What NOT to do
        guidance += "\n\n**Avoid:** Don't move SL away from price in panic. Don't exit prematurely on small fluctuations. Trust your strategy's logic. "
        
        # Strategy-specific advice
        if isinstance(self.strategy, EMACrossoverStrategy):
            ma1 = signal_data.get('MA1', 0)
            ma2 = signal_data.get('MA2', 0)
            if position_type == 'LONG':
                if current_price > ma1 > ma2:
                    guidance += "EMA alignment supports your position. "
                else:
                    guidance += "Watch for EMA alignment weakening. "
        
        return guidance
    
    def analyze_trade(self, trade):
        """AI-powered trade analysis"""
        pnl_pct = trade['pnl_pct']
        duration = trade['duration']
        
        analysis = {
            'performance': '',
            'exit_quality': '',
            'duration_insight': '',
            'recommendations': []
        }
        
        # Performance assessment
        if pnl_pct > 2:
            analysis['performance'] = "Excellent trade! Strong profit capture."
        elif pnl_pct > 0:
            analysis['performance'] = "Profitable trade with moderate gains."
        elif pnl_pct > -1:
            analysis['performance'] = "Small loss - acceptable within strategy parameters."
        else:
            analysis['performance'] = "Significant loss - review entry conditions."
        
        # Exit quality
        if trade['exit_reason'] == 'Target Hit':
            analysis['exit_quality'] = "Perfect exit - target achieved as planned."
        elif trade['exit_reason'] == 'Stop Loss Hit':
            if pnl_pct > -2:
                analysis['exit_quality'] = "Good risk management - SL protected capital."
            else:
                analysis['exit_quality'] = "SL was too wide - consider tighter stops."
        elif trade['exit_reason'] == 'Exit Signal':
            analysis['exit_quality'] = "Signal-based exit - strategy-driven decision."
        else:
            analysis['exit_quality'] = "Manual exit - ensure it aligned with strategy."
        
        # Duration insights
        duration_str = str(duration)
        if 'day' in duration_str:
            analysis['duration_insight'] = "Extended hold period - suitable for trend following."
        elif 'hour' in duration_str:
            analysis['duration_insight'] = "Moderate duration - good for intraday strategies."
        else:
            analysis['duration_insight'] = "Very short duration - scalping or quick reversal."
        
        # Recommendations
        if pnl_pct < 0:
            analysis['recommendations'].append("Review entry timing - may have entered too early/late")
            analysis['recommendations'].append("Check if market conditions matched strategy assumptions")
        
        if trade['exit_reason'] == 'Stop Loss Hit' and pnl_pct < -2:
            analysis['recommendations'].append("Consider tighter stop loss placement")
            analysis['recommendations'].append("Verify trend strength before entry")
        
        if pnl_pct > 2 and trade['exit_reason'] != 'Target Hit':
            analysis['recommendations'].append("Consider using trailing stops to lock profits")
        
        if len(analysis['recommendations']) == 0:
            analysis['recommendations'].append("Good trade execution - maintain consistency")
        
        return analysis

# Sidebar Configuration
st.sidebar.markdown("<div class='main-header'>⚙️ Configuration</div>", unsafe_allow_html=True)

# Asset Selection
st.sidebar.subheader("📊 Asset Selection")

preset_assets = {
    'NIFTY 50': '^NSEI',
    'Bank NIFTY': '^NSEBANK',
    'SENSEX': '^BSESN',
    'BTC-USD': 'BTC-USD',
    'ETH-USD': 'ETH-USD',
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'USD/INR': 'INR=X',
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X'
}

ticker_type = st.sidebar.radio("Select Type:", ['Preset Assets', 'Custom Ticker', 'Indian Stock'])

if ticker_type == 'Preset Assets':
    selected_asset = st.sidebar.selectbox("Choose Asset:", list(preset_assets.keys()))
    ticker = preset_assets[selected_asset]
elif ticker_type == 'Custom Ticker':
    ticker = st.sidebar.text_input("Enter Ticker Symbol:", "AAPL")
else:
    stock_name = st.sidebar.text_input("Enter Stock Name:", "RELIANCE")
    ticker = f"{stock_name}.NS"

st.sidebar.write(f"**Selected Ticker:** `{ticker}`")

# Timeframe and Period
st.sidebar.subheader("⏰ Timeframe & Period")

timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1wk', '1mo']
periods = ['1d', '5d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', '20y', '30y']

# Compatibility mapping
timeframe_period_map = {
    '1m': ['1d', '5d'],
    '5m': ['1d', '5d', '1mo'],
    '15m': ['1d', '5d', '1mo'],
    '30m': ['1d', '5d', '1mo'],
    '1h': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
    '2h': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
    '4h': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
    '1d': ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', '20y', '30y'],
    '1wk': ['1y', '2y', '5y', '10y', '20y', '30y'],
    '1mo': ['1y', '2y', '5y', '10y', '20y', '30y']
}

timeframe = st.sidebar.selectbox("Timeframe:", timeframes, index=4)
compatible_periods = timeframe_period_map.get(timeframe, periods)
period = st.sidebar.selectbox("Period:", compatible_periods)

# Strategy Selection
st.sidebar.subheader("🎯 Strategy Selection")

strategy_names = [
    'EMA/SMA Crossover',
    'Pair Ratio Trading',
    'RSI + Divergence',
    'Fibonacci Retracement',
    'Z-Score Mean Reversion',
    'Breakout + Volume',
    'Simple Buy',
    'Simple Sell'
]

selected_strategy = st.sidebar.selectbox("Choose Strategy:", strategy_names)

# Strategy Parameters
strategy_params = {}
ticker2 = None

if selected_strategy == 'EMA/SMA Crossover':
    st.sidebar.markdown("**Strategy Parameters:**")
    ma_type1 = st.sidebar.selectbox("MA Type 1:", ['EMA', 'SMA'], key='ma1')
    period1 = st.sidebar.number_input("Period 1:", min_value=2, max_value=200, value=9)
    ma_type2 = st.sidebar.selectbox("MA Type 2:", ['EMA', 'SMA'], key='ma2')
    period2 = st.sidebar.number_input("Period 2:", min_value=2, max_value=200, value=20)
    crossover_type = st.sidebar.radio("Crossover Type:", ['simple', 'strong_candle'])
    
    strategy_params = {
        'ma_type1': ma_type1,
        'period1': period1,
        'ma_type2': ma_type2,
        'period2': period2,
        'crossover_type': crossover_type
    }
    strategy_obj = EMACrossoverStrategy(strategy_params)

elif selected_strategy == 'Pair Ratio Trading':
    st.sidebar.markdown("**Select Second Ticker:**")
    ticker2_type = st.sidebar.radio("Ticker 2 Type:", ['Preset Assets', 'Custom Ticker', 'Indian Stock'], key='t2')
    
    if ticker2_type == 'Preset Assets':
        selected_asset2 = st.sidebar.selectbox("Choose Asset 2:", list(preset_assets.keys()), key='asset2')
        ticker2 = preset_assets[selected_asset2]
    elif ticker2_type == 'Custom Ticker':
        ticker2 = st.sidebar.text_input("Enter Ticker 2:", "MSFT", key='ticker2')
    else:
        stock_name2 = st.sidebar.text_input("Enter Stock 2:", "TCS", key='stock2')
        ticker2 = f"{stock_name2}.NS"
    
    zscore_threshold = st.sidebar.number_input("Z-Score Threshold:", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    strategy_params = {'zscore_threshold': zscore_threshold}
    strategy_obj = PairRatioStrategy(strategy_params)

elif selected_strategy == 'RSI + Divergence':
    rsi_period = st.sidebar.number_input("RSI Period:", min_value=5, max_value=50, value=14)
    use_divergence = st.sidebar.checkbox("Use Divergence Detection", value=True)
    strategy_params = {'rsi_period': rsi_period, 'use_divergence': use_divergence}
    strategy_obj = RSIDivergenceStrategy(strategy_params)

elif selected_strategy == 'Fibonacci Retracement':
    lookback = st.sidebar.number_input("Lookback Period:", min_value=20, max_value=200, value=50)
    tolerance = st.sidebar.number_input("Level Tolerance (%):", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    strategy_params = {'lookback': lookback, 'tolerance': tolerance / 100}
    strategy_obj = FibonacciRetracementStrategy(strategy_params)

elif selected_strategy == 'Z-Score Mean Reversion':
    window = st.sidebar.number_input("Window Period:", min_value=10, max_value=100, value=20)
    threshold = st.sidebar.number_input("Z-Score Threshold:", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    strategy_params = {'window': window, 'threshold': threshold}
    strategy_obj = ZScoreMeanReversionStrategy(strategy_params)

elif selected_strategy == 'Breakout + Volume':
    period = st.sidebar.number_input("Channel Period:", min_value=10, max_value=100, value=20)
    volume_multiplier = st.sidebar.number_input("Volume Multiplier:", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
    strategy_params = {'period': period, 'volume_multiplier': volume_multiplier}
    strategy_obj = BreakoutVolumeStrategy(strategy_params)

elif selected_strategy == 'Simple Buy':
    strategy_obj = SimpleBuyStrategy()

elif selected_strategy == 'Simple Sell':
    strategy_obj = SimpleSellStrategy()

# Risk Management
st.sidebar.subheader("🛡️ Risk Management")

quantity = st.sidebar.number_input("Quantity:", min_value=1, max_value=10000, value=1)

st.sidebar.markdown("**Stop Loss Configuration:**")
sl_types = ['Custom Points', 'Trail SL', 'Signal Based', 'Percentage', 'ATR']
sl_type = st.sidebar.selectbox("SL Type:", sl_types)

sl_config = {'type': sl_type}

if sl_type == 'Custom Points' or sl_type == 'Trail SL':
    sl_points = st.sidebar.number_input("SL Points:", min_value=1.0, max_value=1000.0, value=10.0, step=0.5)
    sl_config['points'] = sl_points
elif sl_type == 'Percentage':
    sl_pct = st.sidebar.number_input("SL Percentage:", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    sl_config['percentage'] = sl_pct
elif sl_type == 'ATR':
    atr_multiplier = st.sidebar.number_input("ATR Multiplier:", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    sl_config['atr_multiplier'] = atr_multiplier

st.sidebar.markdown("**Target Configuration:**")
target_types = ['Custom Points', 'Trail Target', 'Signal Based', 'Percentage', 'Risk Reward']
target_type = st.sidebar.selectbox("Target Type:", target_types)

target_config = {'type': target_type}

if target_type == 'Custom Points' or target_type == 'Trail Target':
    target_points = st.sidebar.number_input("Target Points:", min_value=1.0, max_value=1000.0, value=20.0, step=0.5)
    target_config['points'] = target_points
    if target_type == 'Trail Target':
        tolerance = st.sidebar.number_input("Tolerance (%):", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
        target_config['tolerance'] = tolerance
elif target_type == 'Percentage':
    target_pct = st.sidebar.number_input("Target Percentage:", min_value=0.1, max_value=20.0, value=2.0, step=0.1)
    target_config['percentage'] = target_pct
elif target_type == 'Risk Reward':
    rr_ratio = st.sidebar.number_input("Risk:Reward Ratio:", min_value=1.0, max_value=10.0, value=2.0, step=0.1)
    target_config['rr_ratio'] = rr_ratio

# Control Buttons
st.sidebar.markdown("---")
st.sidebar.subheader("🎮 Controls")

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("▶️ Start Trading", type="primary", use_container_width=True):
        st.session_state.trading_active = True
        st.session_state.iteration_count = 0
        add_log("🚀 Trading started")
        st.rerun()

with col2:
    if st.button("⏹️ Stop Trading", type="secondary", use_container_width=True):
        st.session_state.trading_active = False
        
        # Close any open position
        if st.session_state.current_position:
            trading_system = TradingSystem(ticker, timeframe, period, strategy_obj, quantity, sl_config, target_config, ticker2)
            data, data2 = trading_system.fetch_data()
            if data is not None and len(data) > 0:
                current_price = data['Close'].iloc[-1]
                position = st.session_state.current_position
                
                # Calculate final P&L
                if position['type'] == 'LONG':
                    pnl_points = current_price - position['entry_price']
                else:
                    pnl_points = position['entry_price'] - current_price
                
                pnl_pct = (pnl_points / position['entry_price']) * 100
                
                # Save to history
                trade_record = {
                    'entry_time': position['entry_time'],
                    'exit_time': get_ist_time(),
                    'duration': get_ist_time() - position['entry_time'],
                    'type': position['type'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'quantity': position['quantity'],
                    'pnl_points': pnl_points * position['quantity'],
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'Manual Stop',
                    'strategy': strategy_obj.name
                }
                
                st.session_state.trade_history.append(trade_record)
                add_log(f"🛑 Position closed manually: {position['type']} at {current_price:.2f}, P&L: {pnl_pct:.2f}%")
                st.session_state.current_position = None
        
        add_log("⏹️ Trading stopped")
        st.rerun()

if st.session_state.current_position and not st.session_state.trading_active:
    if st.sidebar.button("❌ Force Close Position", type="secondary", use_container_width=True):
        trading_system = TradingSystem(ticker, timeframe, period, strategy_obj, quantity, sl_config, target_config, ticker2)
        data, data2 = trading_system.fetch_data()
        if data is not None and len(data) > 0:
            current_price = data['Close'].iloc[-1]
            position = st.session_state.current_position
            
            if position['type'] == 'LONG':
                pnl_points = current_price - position['entry_price']
            else:
                pnl_points = position['entry_price'] - current_price
            
            pnl_pct = (pnl_points / position['entry_price']) * 100
            
            trade_record = {
                'entry_time': position['entry_time'],
                'exit_time': get_ist_time(),
                'duration': get_ist_time() - position['entry_time'],
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'quantity': position['quantity'],
                'pnl_points': pnl_points * position['quantity'],
                'pnl_pct': pnl_pct,
                'exit_reason': 'Force Close',
                'strategy': strategy_obj.name
            }
            
            st.session_state.trade_history.append(trade_record)
            add_log(f"❌ Position force closed: {position['type']} at {current_price:.2f}, P&L: {pnl_pct:.2f}%")
            st.session_state.current_position = None
            st.rerun()

# Main Content Area
st.markdown("<div class='main-header'>📈 Algorithmic Trading System</div>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔴 Live Trading", "📊 Trade History", "📝 Trade Log"])

# Tab 1: Live Trading
with tab1:
    if st.session_state.trading_active:
        st.markdown("### 🔴 LIVE - Auto-refreshing every 1.5-2s")
        
        # Initialize trading system
        trading_system = TradingSystem(ticker, timeframe, period, strategy_obj, quantity, sl_config, target_config, ticker2)
        
        # Fetch data
        data, data2 = trading_system.fetch_data()
        
        if data is None or len(data) < 10:
            st.error("❌ Insufficient data. Please check ticker symbol and timeframe/period compatibility.")
        else:
            st.session_state.last_data = (data, data2)
            st.session_state.iteration_count += 1
            
            current_price = data['Close'].iloc[-1]
            
            # Display header info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Candles", len(data))
            with col2:
                st.metric("Timeframe", timeframe)
            with col3:
                st.metric("Iteration", st.session_state.iteration_count)
            with col4:
                st.metric("Strategy", strategy_obj.name)
            
            # Check for entry signal if no position
            if st.session_state.current_position is None:
                if isinstance(strategy_obj, PairRatioStrategy) and data2 is not None:
                    bullish, bearish, signal_data = strategy_obj.generate_signal(data, data2)
                else:
                    bullish, bearish, signal_data = strategy_obj.generate_signal(data)
                
                if bullish or bearish:
                    position_type = 'LONG' if bullish else 'SHORT'
                    entry_price = current_price
                    
                    sl_price, target_price = trading_system.calculate_sl_target(entry_price, position_type)
                    
                    st.session_state.current_position = {
                        'type': position_type,
                        'entry_price': entry_price,
                        'entry_time': get_ist_time(),
                        'quantity': quantity,
                        'sl': sl_price,
                        'target': target_price,
                        'signal_data': signal_data
                    }
                    
                    add_log(f"✅ {position_type} position entered at {entry_price:.2f}, SL: {sl_price:.2f}, Target: {target_price:.2f}")
                    st.success(f"✅ {position_type} Position Entered at {entry_price:.2f}")
                else:
                    st.info("⏳ Waiting for entry signal...")
                    
                    # Display strategy parameters while waiting
                    st.markdown("### 📊 Strategy Indicators")
                    param_summary = strategy_obj.get_parameter_summary(data) if not isinstance(strategy_obj, PairRatioStrategy) else strategy_obj.get_parameter_summary(data, data2)
                    
                    cols = st.columns(len(param_summary))
                    for i, (key, value) in enumerate(param_summary.items()):
                        with cols[i]:
                            st.metric(key, value)
            
            # Manage existing position
            if st.session_state.current_position:
                position = st.session_state.current_position
                
                # Update trailing SL/Target
                position = trading_system.update_trailing_sl(position, current_price)
                position = trading_system.update_trailing_target(position, current_price)
                st.session_state.current_position = position
                
                # Calculate P&L
                if position['type'] == 'LONG':
                    pnl_points = current_price - position['entry_price']
                else:
                    pnl_points = position['entry_price'] - current_price
                
                pnl_pct = (pnl_points / position['entry_price']) * 100
                position['pnl_points'] = pnl_points
                position['pnl_pct'] = pnl_pct
                
                # Display position status
                st.markdown("### 💼 Current Position")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Position", position['type'])
                with col2:
                    st.metric("Entry Price", f"{position['entry_price']:.2f}")
                with col3:
                    st.metric("Current Price", f"{current_price:.2f}")
                with col4:
                    pnl_color = "profit" if pnl_pct > 0 else "loss"
                    st.markdown(f"<div class='{pnl_color}'>P&L: {pnl_points:.2f} pts<br>({pnl_pct:.2f}%)</div>", unsafe_allow_html=True)
                with col5:
                    st.metric("Quantity", position['quantity'])
                
                # Display strategy indicators
                st.markdown("### 📊 Strategy Indicators")
                param_summary = strategy_obj.get_parameter_summary(data) if not isinstance(strategy_obj, PairRatioStrategy) else strategy_obj.get_parameter_summary(data, data2)
                
                cols = st.columns(max(len(param_summary), 1))
                for i, (key, value) in enumerate(param_summary.items()):
                    with cols[i]:
                        st.metric(key, value)
                
                # Market status
                market_status = trading_system.get_market_status(position, current_price)
                st.markdown(f"<div class='status-box'>{market_status}</div>", unsafe_allow_html=True)
                
                # Trade guidance
                guidance = trading_system.generate_trade_guidance(position, current_price, data, position['signal_data'])
                st.markdown(f"<div class='trade-guidance'>{guidance}</div>", unsafe_allow_html=True)
                
                # Check exit conditions
                should_exit, exit_reason, exit_price = trading_system.check_exit_conditions(position, current_price, data, data2)
                
                if should_exit:
                    trade_record = {
                        'entry_time': position['entry_time'],
                        'exit_time': get_ist_time(),
                        'duration': get_ist_time() - position['entry_time'],
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'quantity': position['quantity'],
                        'pnl_points': pnl_points * position['quantity'],
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'strategy': strategy_obj.name
                    }
                    
                    st.session_state.trade_history.append(trade_record)
                    add_log(f"🏁 Trade closed: {exit_reason} at {exit_price:.2f}, P&L: {pnl_pct:.2f}%")
                    
                    # Display trade analysis
                    analysis = trading_system.analyze_trade(trade_record)
                    
                    st.markdown("### 🎯 Trade Analysis")
                    st.success(f"Trade closed: {exit_reason}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Performance:** {analysis['performance']}")
                        st.markdown(f"**Exit Quality:** {analysis['exit_quality']}")
                    with col2:
                        st.markdown(f"**Duration:** {analysis['duration_insight']}")
                        st.markdown("**Recommendations:**")
                        for rec in analysis['recommendations']:
                            st.markdown(f"- {rec}")
                    
                    st.session_state.current_position = None
                
                # Display chart
                st.markdown("### 📈 Live Chart")
                
                fig = go.Figure()
                
                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ))
                
                # Add strategy indicators
                if isinstance(strategy_obj, EMACrossoverStrategy):
                    data_with_indicators = strategy_obj.calculate_indicators(data)
                    fig.add_trace(go.Scatter(x=data.index, y=data_with_indicators['MA1'], 
                                           name=f"{strategy_params.get('ma_type1', 'EMA')}{strategy_params.get('period1', 9)}", 
                                           line=dict(color='blue', width=2)))
                    fig.add_trace(go.Scatter(x=data.index, y=data_with_indicators['MA2'], 
                                           name=f"{strategy_params.get('ma_type2', 'EMA')}{strategy_params.get('period2', 20)}", 
                                           line=dict(color='orange', width=2)))
                
                elif isinstance(strategy_obj, FibonacciRetracementStrategy):
                    data_with_indicators, fib_levels = strategy_obj.calculate_indicators(data)
                    for level_name, level_value in fib_levels.items():
                        fig.add_hline(y=level_value, line_dash="dot", annotation_text=level_name, 
                                     line_color="purple", opacity=0.5)
                
                # Entry marker
                if st.session_state.current_position:
                    entry_time = position['entry_time']
                    entry_price = position['entry_price']
                    marker_color = 'green' if position['type'] == 'LONG' else 'red'
                    marker_symbol = 'triangle-up' if position['type'] == 'LONG' else 'triangle-down'
                    
                    fig.add_trace(go.Scatter(
                        x=[entry_time],
                        y=[entry_price],
                        mode='markers',
                        marker=dict(size=15, color=marker_color, symbol=marker_symbol),
                        name=f"{position['type']} Entry"
                    ))
                    
                    # SL and Target lines
                    if position['sl']:
                        fig.add_hline(y=position['sl'], line_dash="dash", 
                                     annotation_text="Stop Loss", line_color="red")
                    if position['target']:
                        fig.add_hline(y=position['target'], line_dash="dash", 
                                     annotation_text="Target", line_color="green")
                
                fig.update_layout(
                    title=f"{ticker} - {timeframe} Chart",
                    xaxis_title="Time (IST)",
                    yaxis_title="Price",
                    height=600,
                    template="plotly_white",
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Auto-refresh
            time.sleep(1.5)
            st.rerun()
    
    else:
        st.info("👆 Click 'Start Trading' to begin live monitoring")
        
        if st.session_state.last_data:
            data, data2 = st.session_state.last_data
            st.markdown("### 📊 Last Fetched Data")
            st.dataframe(data.tail(10))

# Tab 2: Trade History
with tab2:
    st.markdown("### 📊 Trade History & Performance")
    
    if len(st.session_state.trade_history) == 0:
        st.info("No trades yet. Start trading to see your performance history.")
    else:
        trades_df = pd.DataFrame(st.session_state.trade_history)
        
        # Performance Summary
        st.markdown("#### 📈 Performance Summary")
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
        losing_trades = len(trades_df[trades_df['pnl_pct'] <= 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = trades_df['pnl_points'].sum()
        avg_pnl = trades_df['pnl_pct'].mean()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Wins", winning_trades, delta=f"{win_rate:.1f}%")
        with col3:
            st.metric("Losses", losing_trades)
        with col4:
            pnl_color = "normal" if total_pnl >= 0 else "inverse"
            st.metric("Total P&L", f"{total_pnl:.2f} pts", delta_color=pnl_color)
        with col5:
            st.metric("Avg P&L", f"{avg_pnl:.2f}%")
        
        # Performance by Strategy
        st.markdown("#### 🎯 Performance by Strategy")
        strategy_performance = trades_df.groupby('strategy').agg({
            'pnl_points': 'sum',
            'pnl_pct': 'mean',
            'type': 'count'
        }).rename(columns={'type': 'trades'})
        st.dataframe(strategy_performance)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Cumulative P&L
            trades_df['cumulative_pnl'] = trades_df['pnl_points'].cumsum()
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=list(range(len(trades_df))),
                y=trades_df['cumulative_pnl'],
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='blue', width=2)
            ))
            fig_cum.update_layout(
                title="Cumulative P&L",
                xaxis_title="Trade Number",
                yaxis_title="P&L (points)",
                height=400
            )
            st.plotly_chart(fig_cum, use_container_width=True)
        
        with col2:
            # Win/Loss Pie Chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Wins', 'Losses'],
                values=[winning_trades, losing_trades],
                marker=dict(colors=['green', 'red'])
            )])
            fig_pie.update_layout(title="Win/Loss Distribution", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # P&L Distribution
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=trades_df['pnl_pct'],
            nbinsx=20,
            marker=dict(color='lightblue', line=dict(color='darkblue', width=1))
        ))
        fig_hist.update_layout(
            title="P&L Distribution",
            xaxis_title="P&L (%)",
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Detailed Trade Table
        st.markdown("#### 📋 Detailed Trade Log")
        
        # Format the dataframe for display
        display_df = trades_df.copy()
        display_df['entry_time'] = display_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['exit_time'] = display_df['exit_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['duration'] = display_df['duration'].astype(str)
        display_df['pnl_points'] = display_df['pnl_points'].round(2)
        display_df['pnl_pct'] = display_df['pnl_pct'].round(2)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Individual Trade Analysis
        st.markdown("#### 🔍 Individual Trade Analysis")
        
        trade_idx = st.selectbox("Select Trade:", range(len(trades_df)), 
                                 format_func=lambda x: f"Trade {x+1} - {trades_df.iloc[x]['type']} - {trades_df.iloc[x]['exit_reason']}")
        
        if trade_idx is not None:
            selected_trade = st.session_state.trade_history[trade_idx]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Entry:** {selected_trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"**Entry Price:** {selected_trade['entry_price']:.2f}")
                st.markdown(f"**Position:** {selected_trade['type']}")
            with col2:
                st.markdown(f"**Exit:** {selected_trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"**Exit Price:** {selected_trade['exit_price']:.2f}")
                st.markdown(f"**Exit Reason:** {selected_trade['exit_reason']}")
            with col3:
                st.markdown(f"**Duration:** {selected_trade['duration']}")
                pnl_class = "profit" if selected_trade['pnl_pct'] > 0 else "loss"
                st.markdown(f"<div class='{pnl_class}'>P&L: {selected_trade['pnl_points']:.2f} pts ({selected_trade['pnl_pct']:.2f}%)</div>", 
                          unsafe_allow_html=True)
                st.markdown(f"**Strategy:** {selected_trade['strategy']}")
            
            # AI Analysis
            trading_system = TradingSystem(ticker, timeframe, period, strategy_obj, quantity, sl_config, target_config, ticker2)
            analysis = trading_system.analyze_trade(selected_trade)
            
            st.markdown("**AI Analysis:**")
            st.info(f"**Performance:** {analysis['performance']}")
            st.info(f"**Exit Quality:** {analysis['exit_quality']}")
            st.info(f"**Duration Insight:** {analysis['duration_insight']}")
            
            st.markdown("**Recommendations:**")
            for rec in analysis['recommendations']:
                st.markdown(f"- {rec}")

# Tab 3: Trade Log
with tab3:
    st.markdown("### 📝 Trade Log")
    
    if st.button("🗑️ Clear Log"):
        st.session_state.trade_log = []
        st.rerun()
    
    if len(st.session_state.trade_log) == 0:
        st.info("No log entries yet.")
    else:
        log_html = "<div class='log-container'>"
        for entry in st.session_state.trade_log:
            log_html += f"<div>{entry}</div>"
        log_html += "</div>"
        st.markdown(log_html, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><b>Professional Algorithmic Trading System</b></p>
    <p>⚠️ This is for educational purposes only. Trading involves risk. Always do your own research.</p>
    <p>💡 Tip: Use appropriate position sizing and risk management for your capital.</p>
</div>
""", unsafe_allow_html=True)
