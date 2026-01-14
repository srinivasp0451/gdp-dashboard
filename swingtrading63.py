import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import time
import random
import threading

# ========================================
# UTILITY FUNCTIONS
# ========================================

def get_ist_time():
    """Get current time in IST"""
    return datetime.now(pytz.timezone('Asia/Kolkata'))

def convert_to_ist(dt_series):
    """Convert datetime series to IST"""
    if dt_series.dt.tz is             config['adx_period'] = st.number_input("ADX Period", min_value=1, value=14, key="adx_period_rsi_input")
            config['ema1_period'] = st.number_input("EMA1 Period", min_value=1, value=9, key="ema1_period_input")
            config['ema2_period'] = st.number_input("EMA2 Period", min_value=1, value=21, key="ema2_period_input")
        
        elif strategy == "Custom Strategy Builder":
            st.subheader("Custom Conditions")
            
            if st.button("Add Condition", key="add_condition_btn"):
                st.session_state['custom_conditions'].append({
                    'use_condition': True,
                    'indicator1': 'Price',
                    'use_indicator': False,
                    'indicator2': 'EMA_20',
                    'operator': '>',
                    'value': 50,
                    'action': 'BUY'
                })
            
            conditions_to_remove = []
            for idx, cond in enumerate(st.session_state['custom_conditions']):
                st.markdown(f"**Condition {idx + 1}**")
                
                cond['use_condition'] = st.checkbox("Use this condition", value=cond['use_condition'], key=f"use_cond_{idx}")
                
                col1, col2 = st.columns(2)
                with col1:
                    cond['indicator1'] = st.selectbox("First Indicator", [
                        'Price', 'RSI', 'ADX', 'EMA_Fast', 'SuperTrend', 'EMA_Slow', 'MACD', 'MACD_Signal',
                        'BB_Upper', 'BB_Lower', 'ATR', 'Volume', 'VWAP', 'KC_Upper', 'KC_Lower',
                        'Close', 'High', 'Low', 'Support', 'Resistance', 'EMA_20', 'EMA_50'
                    ], index=['Price', 'RSI', 'ADX', 'EMA_Fast', 'SuperTrend', 'EMA_Slow', 'MACD', 'MACD_Signal',
                        'BB_Upper', 'BB_Lower', 'ATR', 'Volume', 'VWAP', 'KC_Upper', 'KC_Lower',
                        'Close', 'High', 'Low', 'Support', 'Resistance', 'EMA_20', 'EMA_50'].index(cond['indicator1']), key=f"ind1_{idx}")
                
                with col2:
                    cond['operator'] = st.selectbox("Operator", [
                        '>', '<', '>=', '<=', '==', 'crosses_above', 'crosses_below'
                    ], index=['>', '<', '>=', '<=', '==', 'crosses_above', 'crosses_below'].index(cond['operator']), key=f"op_{idx}")
                
                cond['use_indicator'] = st.checkbox("Compare with Indicator", value=cond.get('use_indicator', False), key=f"use_ind_{idx}")
                
                col3, col4 = st.columns(2)
                with col3:
                    if cond['use_indicator']:
                        cond['indicator2'] = st.selectbox("Second Indicator", [
                            'EMA_20', 'EMA_50', 'RSI', 'ADX', 'EMA_Fast', 'SuperTrend', 'EMA_Slow', 'MACD', 'MACD_Signal',
                            'BB_Upper', 'BB_Lower', 'ATR', 'Volume', 'VWAP', 'KC_Upper', 'KC_Lower',
                            'Close', 'High', 'Low', 'Support', 'Resistance'
                        ], index=['EMA_20', 'EMA_50', 'RSI', 'ADX', 'EMA_Fast', 'SuperTrend', 'EMA_Slow', 'MACD', 'MACD_Signal',
                            'BB_Upper', 'BB_Lower', 'ATR', 'Volume', 'VWAP', 'KC_Upper', 'KC_Lower',
                            'Close', 'High', 'Low', 'Support', 'Resistance'].index(cond.get('indicator2', 'EMA_20')), key=f"ind2_{idx}")
                    else:
                        cond['value'] = st.number_input("Value", value=float(cond['value']), key=f"val_{idx}")
                
                with col4:
                    cond['action'] = st.selectbox("Action", ['BUY', 'SELL'], index=['BUY', 'SELL'].index(cond['action']), key=f"act_{idx}")
                
                if st.button("Remove", key=f"remove_{idx}"):
                    conditions_to_remove.append(idx)
                
                st.markdown("---")
            
            for idx in sorted(conditions_to_remove, reverse=True):
                st.session_state['custom_conditions'].pop(idx)
            
            config['custom_conditions'] = st.session_state['custom_conditions']
        
        # Minimum distances
        st.subheader("Safety Distances")
        config['min_sl_distance'] = st.number_input("Min SL Distance (points)", min_value=0.0, value=10.0, key="min_sl_dist_input")
        config['min_target_distance'] = st.number_input("Min Target Distance (points)", min_value=0.0, value=15.0, key="min_target_dist_input")
        
        # Stop Loss Configuration
        st.subheader("Stop Loss")
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
            "Signal-based"
        ], key="sl_type_select")
        
        if config['sl_type'] not in ["Signal-based"]:
            config['sl_points'] = st.number_input("SL Points", min_value=0.0, value=10.0, key="sl_points_input")
        
        if config['sl_type'].startswith("Trailing"):
            config['trailing_threshold'] = st.number_input("Trailing Threshold (Points)", min_value=0.0, value=0.0, key="trailing_threshold_input")
        
        # Target Configuration
        st.subheader("Target")
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
            "Signal-based"
        ], key="target_type_select")
        
        if config['target_type'] not in ["Signal-based"]:
            config['target_points'] = st.number_input("Target Points", min_value=0.0, value=20.0, key="target_points_input")
        
        if config['target_type'] == "Risk-Reward Based":
            config['rr_ratio'] = st.number_input("Risk:Reward Ratio", min_value=0.1, value=2.0, key="rr_ratio_input")
    
    # Main Content Area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Trading Dashboard", "📈 Trade History", "📝 Trade Logs", "🔬 Backtest Results"])
    
    # Tab 1: Live Trading Dashboard
    with tab1:
        col1, col2, col3, col4 = st.columns([2, 2, 3, 3])
        
        with col1:
            if st.button("▶️ Start Trading", type="primary", key="start_trading_btn"):
                if not st.session_state['trading_active']:
                    st.session_state['trading_active'] = True
                    add_log("Trading started")
                    
                    # Start background thread
                    if st.session_state['trading_thread'] is None or not st.session_state['trading_thread'].is_alive():
                        thread = threading.Thread(
                            target=live_trading_worker,
                            args=(ticker, interval, period, strategy, config, quantity),
                            daemon=True
                        )
                        thread.start()
                        st.session_state['trading_thread'] = thread
                    
                    st.rerun()
        
        with col2:
            if st.button("⏸️ Stop Trading", key="stop_trading_btn"):
                if st.session_state['trading_active']:
                    st.session_state['trading_active'] = False
                    add_log("Trading stopped")
                    
                    position = st.session_state.get('position')
                    if position:
                        df = st.session_state.get('current_data')
                        if df is not None and len(df) > 0:
                            exit_price = df['Close'].iloc[-1]
                            duration = get_ist_time() - position['entry_time']
                            pnl = (exit_price - position['entry_price']) * position['signal'] * position['quantity']
                            
                            trade = {
                                'Entry Time': position['entry_time'],
                                'Exit Time': get_ist_time(),
                                'Duration': str(duration),
                                'Signal': 'LONG' if position['signal'] == 1 else 'SHORT',
                                'Entry Price': position['entry_price'],
                                'Exit Price': exit_price,
                                'SL': position['sl'],
                                'Target': position['target'],
                                'Exit Reason': 'Manual Close',
                                'P&L': pnl,
                                'Highest Price': position.get('highest_price', exit_price),
                                'Lowest Price': position.get('lowest_price', exit_price),
                                'Range': position.get('highest_price', exit_price) - position.get('lowest_price', exit_price)
                            }
                            
                            st.session_state['trade_history'].append(trade)
                            st.session_state['trade_history'] = st.session_state['trade_history']
                            add_log(f"Position closed manually | P&L: {pnl:.2f}")
                    
                    reset_position_state()
                    st.rerun()
        
        with col3:
            if st.session_state['trading_active']:
                st.success("✅ Trading is ACTIVE")
            else:
                st.info("⏸️ Trading is STOPPED")
        
        with col4:
            if st.button("🔄 Manual Refresh", key="manual_refresh_btn"):
                st.rerun()
        
        st.markdown("---")
        
        # Active Configuration Display
        st.subheader("📋 Active Configuration")
        config_col1, config_col2, config_col3 = st.columns(3)
        
        with config_col1:
            st.markdown(f"**Asset:** {ticker}")
            st.markdown(f"**Interval:** {interval}")
            st.markdown(f"**Period:** {period}")
            st.markdown(f"**Quantity:** {quantity}")
        
        with config_col2:
            st.markdown(f"**Strategy:** {strategy}")
            st.markdown(f"**Mode:** {mode}")
            st.markdown(f"**SL Type:** {config.get('sl_type', 'N/A')}")
            if config.get('sl_points'):
                st.markdown(f"**SL Points:** {config['sl_points']}")
        
        with config_col3:
            st.markdown(f"**Target Type:** {config.get('target_type', 'N/A')}")
            if config.get('target_points'):
                st.markdown(f"**Target Points:** {config['target_points']}")
            st.markdown(f"**Min SL Dist:** {config.get('min_sl_distance', 10)}")
            st.markdown(f"**Min Target Dist:** {config.get('min_target_distance', 15)}")
        
        st.markdown("---")
        
        # Live Metrics Display
        df = st.session_state.get('current_data')
        position = st.session_state.get('position')
        
        if df is not None and len(df) > 0:
            current_price = df['Close'].iloc[-1]
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
            
            with metrics_col1:
                st.metric("Current Price", f"{current_price:.2f}")
            
            with metrics_col2:
                if position:
                    st.metric("Entry Price", f"{position['entry_price']:.2f}")
                else:
                    st.metric("Entry Price", "N/A")
            
            with metrics_col3:
                if position:
                    pos_type = "LONG" if position['signal'] == 1 else "SHORT"
                    st.metric("Position", pos_type)
                else:
                    st.metric("Position", "No Position")
            
            with metrics_col4:
                if position:
                    unrealized_pnl = (current_price - position['entry_price']) * position['signal'] * position['quantity']
                    if unrealized_pnl >= 0:
                        st.metric("Unrealized P&L", f"{unrealized_pnl:.2f}", delta=f"+{unrealized_pnl:.2f}")
                    else:
                        st.metric("Unrealized P&L", f"{unrealized_pnl:.2f}", delta=f"{unrealized_pnl:.2f}", delta_color="inverse")
                else:
                    st.metric("Unrealized P&L", "N/A")
            
            with metrics_col5:
                if st.session_state.get('last_update'):
                    last_update = st.session_state['last_update'].strftime("%H:%M:%S")
                    st.metric("Last Update", last_update)
                else:
                    st.metric("Last Update", "N/A")
            
            # Indicator Values
            st.subheader("📊 Indicator Values")
            ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
            
            with ind_col1:
                if 'EMA_Fast' in df.columns:
                    st.markdown(f"**EMA Fast:** {df['EMA_Fast'].iloc[-1]:.2f}")
            
            with ind_col2:
                if 'EMA_Slow' in df.columns:
                    st.markdown(f"**EMA Slow:** {df['EMA_Slow'].iloc[-1]:.2f}")
            
            with ind_col3:
                if 'EMA_Fast' in df.columns and 'EMA_Slow' in df.columns:
                    angle = calculate_ema_angle(df['EMA_Fast'], lookback=2)
                    st.markdown(f"**Crossover Angle:** {angle:.2f}°")
            
            with ind_col4:
                if 'RSI' in df.columns:
                    st.markdown(f"**RSI:** {df['RSI'].iloc[-1]:.2f}")
            
            # Entry Filter Status
            if strategy == "EMA Crossover":
                st.subheader("🎯 Entry Filter Status")
                filter_col1, filter_col2 = st.columns(2)
                
                with filter_col1:
                    entry_filter = config.get('entry_filter', 'Simple Crossover')
                    st.markdown(f"**Filter Type:** {entry_filter}")
                    
                    candle_size = abs(df['Close'].iloc[-1] - df['Open'].iloc[-1])
                    st.markdown(f"**Current Candle Size:** {candle_size:.2f}")
                
                with filter_col2:
                    if entry_filter == "Custom Candle (Points)":
                        min_size = config.get('custom_points', 0)
                        status = "✅" if candle_size >= min_size else "❌"
                        st.markdown(f"**Min Required:** {min_size:.2f} {status}")
                    elif entry_filter == "ATR-based Candle":
                        if 'ATR' in df.columns:
                            atr_mult = config.get('atr_multiplier', 1.0)
                            min_size = df['ATR'].iloc[-1] * atr_mult
                            status = "✅" if candle_size >= min_size else "❌"
                            st.markdown(f"**Min Required (ATR×{atr_mult}):** {min_size:.2f} {status}")
                
                if config.get('use_adx') and 'ADX' in df.columns:
                    adx_val = df['ADX'].iloc[-1]
                    adx_threshold = config.get('adx_threshold', 25)
                    adx_status = "✅" if adx_val >= adx_threshold else "❌"
                    st.markdown(f"**ADX:** {adx_val:.2f} / {adx_threshold:.2f} {adx_status}")
            
            # Current Signal
            current_signal = df['Signal'].iloc[-1]
            if current_signal == 1:
                st.success("🟢 Current Signal: BUY")
            elif current_signal == -1:
                st.error("🔴 Current Signal: SELL")
            else:
                st.info("⚪ Current Signal: NONE")
            
            # AI Analysis Display
            if strategy == "AI Price Action Analysis" and 'AI_Analysis' in df.columns:
                st.subheader("🤖 AI Price Action Analysis")
                ai_col1, ai_col2 = st.columns(2)
                
                with ai_col1:
                    confidence = df['AI_Confidence'].iloc[-1]
                    st.metric("AI Confidence", f"{confidence:.1f}%")
                
                with ai_col2:
                    if current_signal == 1:
                        st.success("AI Signal: BUY")
                    elif current_signal == -1:
                        st.error("AI Signal: SELL")
                    else:
                        st.info("AI Signal: HOLD")
                
                analysis = df['AI_Analysis'].iloc[-1]
                st.markdown("**Analysis Breakdown:**")
                st.text(analysis)
            
            # Position Information
            if position:
                st.subheader("📍 Position Information")
                
                pos_col1, pos_col2, pos_col3 = st.columns(3)
                
                with pos_col1:
                    entry_time = position['entry_time'].strftime("%Y-%m-%d %H:%M:%S")
                    st.markdown(f"**Entry Time:** {entry_time}")
                    
                    duration = get_ist_time() - position['entry_time']
                    hours = duration.total_seconds() / 3600
                    st.markdown(f"**Duration:** {hours:.2f} hours")
                    
                    st.markdown(f"**Entry Price:** {position['entry_price']:.2f}")
                
                with pos_col2:
                    sl_str = f"{position['sl']:.2f}" if position['sl'] != 0 else "Signal Based"
                    st.markdown(f"**Stop Loss:** {sl_str}")
                    
                    target_str = f"{position['target']:.2f}" if position['target'] != 0 else "Signal Based"
                    st.markdown(f"**Target:** {target_str}")
                    
                    if position['sl'] != 0:
                        dist_sl = abs(current_price - position['sl'])
                        st.markdown(f"**Distance to SL:** {dist_sl:.2f}")
                
                with pos_col3:
                    if position['target'] != 0:
                        dist_target = abs(current_price - position['target'])
                        st.markdown(f"**Distance to Target:** {dist_target:.2f}")
                    
                    highest = position.get('highest_price', current_price)
                    lowest = position.get('lowest_price', current_price)
                    range_val = highest - lowest
                    
                    st.markdown(f"**Highest Price:** {highest:.2f}")
                    st.markdown(f"**Lowest Price:** {lowest:.2f}")
                    st.markdown(f"**Range:** {range_val:.2f}")
                
                if config.get('target_type') in ["Trailing Target (Points)", "Trailing Target + Signal Based"]:
                    trailing_profit = st.session_state.get('trailing_profit_points', 0)
                    target_points = config.get('target_points', 20)
                    st.info(f"💰 Profit moved: {trailing_profit:.2f} points | Next update at: {trailing_profit + target_points:.2f} points")
                
                if position.get('partial_exit_done'):
                    st.warning("⚠️ 50% position already exited - Trailing remaining")
                
                if position.get('breakeven_activated'):
                    st.success("✅ Break-even SL activated - Risk-free trade")
            
            # Live Chart
            st.subheader("📈 Live Chart")
            
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=df['Datetime'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ))
            
            if 'EMA_Fast' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['Datetime'],
                    y=df['EMA_Fast'],
                    mode='lines',
                    name='EMA Fast',
                    line=dict(color='blue', width=1)
                ))
            
            if 'EMA_Slow' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['Datetime'],
                    y=df['EMA_Slow'],
                    mode='lines',
                    name='EMA Slow',
                    line=dict(color='orange', width=1)
                ))
            
            if position:
                fig.add_hline(
                    y=position['entry_price'],
                    line_dash="dash",
                    line_color="blue",
                    annotation_text="Entry",
                    annotation_position="right"
                )
                
                if position['sl'] != 0:
                    fig.add_hline(
                        y=position['sl'],
                        line_dash="dash",
                        line_color="red",
                        annotation_text="SL",
                        annotation_position="right"
                    )
                
                if position['target'] != 0:
                    fig.add_hline(
                        y=position['target'],
                        line_dash="dash",
                        line_color="green",
                        annotation_text="Target",
                        annotation_position="right"
                    )
            
            fig.update_layout(
                title=f"{ticker} - {interval}",
                xaxis_title="Time",
                yaxis_title="Price",
                height=600,
                xaxis_rangeslider_visible=False
            )
            
            chart_key = f"live_chart_{get_ist_time().timestamp()}"
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
            
            if position:
                unrealized_pnl = (current_price - position['entry_price']) * position['signal'] * position['quantity']
                if unrealized_pnl > 0:
                    st.success("💰 In Profit - Hold or consider trailing")
                elif unrealized_pnl < 0:
                    st.error("⚠️ In Loss - Monitor SL carefully")
                else:
                    st.info("⚖️ At Entry - Wait for movement")
            else:
                st.info("🔍 Waiting for entry signal...")
        
        else:
            st.info("📊 No data available. Start trading to fetch live data.")
    
    # Tab 2: Trade History
    with tab2:
        st.markdown("### 📈 Trade History")
        
        trade_history = st.session_state.get('trade_history', [])
        
        if len(trade_history) == 0:
            st.info("No trades executed yet.")
        else:
            trades_df = pd.DataFrame(trade_history)
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['P&L'] > 0])
            losing_trades = len(trades_df[trades_df['P&L'] < 0])
            accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = trades_df['P&L'].sum()
            
            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
            
            with metric_col1:
                st.metric("Total Trades", total_trades)
            
            with metric_col2:
                st.metric("Winning Trades", winning_trades)
            
            with metric_col3:
                st.metric("Losing Trades", losing_trades)
            
            with metric_col4:
                st.metric("Accuracy", f"{accuracy:.1f}%")
            
            with metric_col5:
                if total_pnl >= 0:
                    st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"+{total_pnl:.2f}")
                else:
                    st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color="inverse")
            
            st.markdown("---")
            
            for idx, trade in enumerate(reversed(trade_history)):
                with st.expander(f"Trade #{total_trades - idx} - {trade.get('Signal', 'N/A')} - P&L: {trade.get('P&L', 0):.2f}"):
                    trade_col1, trade_col2 = st.columns(2)
                    
                    with trade_col1:
                        st.markdown(f"**Entry Time:** {trade.get('Entry Time', 'N/A')}")
                        st.markdown(f"**Exit Time:** {trade.get('Exit Time', 'N/A')}")
                        st.markdown(f"**Duration:** {trade.get('Duration', 'N/A')}")
                        st.markdown(f"**Signal:** {trade.get('Signal', 'N/A')}")
                        st.markdown(f"**Entry Price:** {trade.get('Entry Price', 0):.2f}")
                    
                    with trade_col2:
                        st.markdown(f"**Exit Price:** {trade.get('Exit Price', 0):.2f}")
                        sl_val = trade.get('SL', 0)
                        sl_str = f"{sl_val:.2f}" if sl_val != 0 else "Signal Based"
                        st.markdown(f"**Stop Loss:** {sl_str}")
                        
                        target_val = trade.get('Target', 0)
                        target_str = f"{target_val:.2f}" if target_val != 0 else "Signal Based"
                        st.markdown(f"**Target:** {target_str}")
                        
                        st.markdown(f"**Exit Reason:** {trade.get('Exit Reason', 'N/A')}")
                        
                        pnl = trade.get('P&L', 0)
                        if pnl >= 0:
                            st.success(f"**P&L:** +{pnl:.2f}")
                        else:
                            st.error(f"**P&L:** {pnl:.2f}")
                    
                    st.markdown(f"**Highest Price:** {trade.get('Highest Price', 0):.2f}")
                    st.markdown(f"**Lowest Price:** {trade.get('Lowest Price', 0):.2f}")
                    st.markdown(f"**Range:** {trade.get('Range', 0):.2f}")
    
    # Tab 3: Trade Logs
    with tab3:
        st.markdown("### 📝 Trade Logs")
        
        trade_logs = st.session_state.get('trade_logs', [])
        
        if len(trade_logs) == 0:
            st.info("No logs yet.")
        else:
            for log in reversed(trade_logs):
                st.text(log)
    
    # Tab 4: Backtest Results
    with tab4:
        st.markdown("### 🔬 Backtest Results")
        
        if mode != "Backtest":
            st.warning("Switch to Backtest mode to run backtests.")
        else:
            if st.button("🚀 Run Backtest", type="primary", key="run_backtest_btn"):
                with st.spinner("Running backtest..."):
                    df = fetch_data(ticker, interval, period, mode)
                    
                    if df is None or len(df) == 0:
                        st.error("Failed to fetch data for backtest")
                    else:
                        results = run_backtest(df, strategy, config, quantity)
                        
                        st.success("Backtest completed!")
                        
                        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5, metric_col6 = st.columns(6)
                        
                        with metric_col1:
                            st.metric("Total Trades", results['Total Trades'])
                        
                        with metric_col2:
                            st.metric("Winning Trades", results['Winning Trades'])
                        
                        with metric_col3:
                            st.metric("Losing Trades", results['Losing Trades'])
                        
                        with metric_col4:
                            st.metric("Accuracy", f"{results['Accuracy']:.1f}%")
                        
                        with metric_col5:
                            total_pnl = results['Total P&L']
                            if total_pnl >= 0:
                                st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"+{total_pnl:.2f}")
                            else:
                                st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color="inverse")
                        
                        with metric_col6:
                            st.metric("Avg Duration", f"{results['Avg Duration (hours)']:.2f}h")
                        
                        st.markdown("---")
                        
                        if not results['Trades'].empty:
                            st.markdown("### 📊 Trade Details")
                            
                            trades_df = results['Trades'].copy()
                            
                            display_df = trades_df[[
                                'Entry Time', 'Exit Time', 'Duration', 'Signal', 
                                'Entry Price', 'Exit Price', 'SL', 'Target', 
                                'Exit Reason', 'P&L', 'Highest Price', 'Lowest Price', 'Range'
                            ]].copy()
                            
                            def color_pnl(val):
                                color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                                return f'color: {color}'
                            
                            styled_df = display_df.style.applymap(color_pnl, subset=['P&L'])
                            
                            st.dataframe(styled_df, use_container_width=True)
                        else:
                            st.info("No trades executed during backtest period.")

if __name__ == "__main__":
    main():
        return dt_series.dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    else:
        return dt_series.dt.tz_convert('Asia/Kolkata')

def add_log(message):
    """Add timestamped log entry"""
    timestamp = get_ist_time().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state['trade_logs'].append(log_entry)
    if len(st.session_state['trade_logs']) > 50:
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

# ========================================
# INDICATOR CALCULATIONS
# ========================================

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
    return true_range.rolling(window=period).mean()

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
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    atr = calculate_atr(df, period)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
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
    middle = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

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
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        else:
            if df['Close'].iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif df['Close'].iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
    
    return supertrend

def calculate_vwap(df):
    """Calculate VWAP"""
    if 'Volume' not in df.columns or df['Volume'].sum() == 0:
        return pd.Series(index=df.index, dtype=float)
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap

def calculate_keltner_channel(df, period=20, atr_period=10, multiplier=2):
    """Calculate Keltner Channel"""
    middle = calculate_ema(df['Close'], period)
    atr = calculate_atr(df, atr_period)
    upper = middle + (multiplier * atr)
    lower = middle - (multiplier * atr)
    return upper, middle, lower

def calculate_ema_angle(ema_series, lookback=2):
    """Calculate EMA angle in degrees"""
    if len(ema_series) < lookback + 1:
        return 0
    
    y_diff = ema_series.iloc[-1] - ema_series.iloc[-lookback-1]
    x_diff = lookback
    
    angle_rad = np.arctan2(y_diff, x_diff)
    angle_deg = np.degrees(angle_rad)
    
    return abs(angle_deg)

def detect_swing_highs_lows(df, lookback=5):
    """Detect swing highs and lows"""
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(df) - lookback):
        if all(df['High'].iloc[i] >= df['High'].iloc[i-j] for j in range(1, lookback+1)) and \
           all(df['High'].iloc[i] >= df['High'].iloc[i+j] for j in range(1, lookback+1)):
            swing_highs.append((i, df['High'].iloc[i]))
        
        if all(df['Low'].iloc[i] <= df['Low'].iloc[i-j] for j in range(1, lookback+1)) and \
           all(df['Low'].iloc[i] <= df['Low'].iloc[i+j] for j in range(1, lookback+1)):
            swing_lows.append((i, df['Low'].iloc[i]))
    
    return swing_highs, swing_lows

def calculate_support_resistance(df, lookback=20):
    """Calculate support and resistance levels"""
    recent_highs = df['High'].tail(lookback)
    recent_lows = df['Low'].tail(lookback)
    
    resistance = recent_highs.max()
    support = recent_lows.min()
    
    return support, resistance

# ========================================
# DATA FETCHING
# ========================================

def fetch_data(ticker, interval, period, mode):
    """Fetch data from yfinance with proper handling"""
    try:
        if mode == "Live Trading":
            delay = random.uniform(1.0, 1.5)
            time.sleep(delay)
        
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if data.empty:
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_cols):
            return None
        
        data.index = pd.to_datetime(data.index)
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            data.index = data.index.tz_convert('Asia/Kolkata')
        
        data = data.reset_index()
        data.rename(columns={'index': 'Datetime', 'Date': 'Datetime'}, inplace=True)
        
        if 'Volume' not in data.columns:
            data['Volume'] = 0
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# ========================================
# STRATEGY IMPLEMENTATIONS
# ========================================

def ema_crossover_strategy(df, config):
    """EMA Crossover Strategy with filters"""
    fast_period = config['ema_fast']
    slow_period = config['ema_slow']
    min_angle = config['min_angle']
    entry_filter = config['entry_filter']
    custom_points = config.get('custom_points', 0)
    atr_multiplier = config.get('atr_multiplier', 1.0)
    use_adx = config.get('use_adx', False)
    adx_threshold = config.get('adx_threshold', 25)
    adx_period = config.get('adx_period', 14)
    
    df['EMA_Fast'] = calculate_ema(df['Close'], fast_period)
    df['EMA_Slow'] = calculate_ema(df['Close'], slow_period)
    
    if entry_filter == "ATR-based Candle":
        df['ATR'] = calculate_atr(df, 14)
    
    if use_adx:
        df['ADX'] = calculate_adx(df, adx_period)
    
    df['Signal'] = 0
    df['Entry_Price'] = 0.0
    df['SL'] = 0.0
    df['Target'] = 0.0
    
    for i in range(1, len(df)):
        ema_fast_angle = calculate_ema_angle(df['EMA_Fast'].iloc[:i+1], lookback=2)
        
        bullish_cross = (df['EMA_Fast'].iloc[i] > df['EMA_Slow'].iloc[i] and 
                        df['EMA_Fast'].iloc[i-1] <= df['EMA_Slow'].iloc[i-1])
        
        bearish_cross = (df['EMA_Fast'].iloc[i] < df['EMA_Slow'].iloc[i] and 
                        df['EMA_Fast'].iloc[i-1] >= df['EMA_Slow'].iloc[i-1])
        
        angle_ok = ema_fast_angle >= min_angle
        
        filter_ok = False
        candle_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
        
        if entry_filter == "Simple Crossover":
            filter_ok = True
        elif entry_filter == "Custom Candle (Points)":
            filter_ok = candle_size >= custom_points
        elif entry_filter == "ATR-based Candle":
            min_candle = df['ATR'].iloc[i] * atr_multiplier
            filter_ok = candle_size >= min_candle
        
        adx_ok = True
        if use_adx:
            adx_ok = df['ADX'].iloc[i] >= adx_threshold
        
        if bullish_cross and angle_ok and filter_ok and adx_ok:
            df.loc[i, 'Signal'] = 1
            df.loc[i, 'Entry_Price'] = df['Close'].iloc[i]
        elif bearish_cross and angle_ok and filter_ok and adx_ok:
            df.loc[i, 'Signal'] = -1
            df.loc[i, 'Entry_Price'] = df['Close'].iloc[i]
    
    return df

def simple_buy_strategy(df, config):
    """Simple Buy Strategy"""
    df['Signal'] = 1
    df['Entry_Price'] = df['Close']
    df['SL'] = 0.0
    df['Target'] = 0.0
    return df

def simple_sell_strategy(df, config):
    """Simple Sell Strategy"""
    df['Signal'] = -1
    df['Entry_Price'] = df['Close']
    df['SL'] = 0.0
    df['Target'] = 0.0
    return df

def price_threshold_strategy(df, config):
    """Price Crosses Threshold Strategy"""
    threshold = config['threshold']
    threshold_type = config['threshold_type']
    
    df['Signal'] = 0
    df['Entry_Price'] = 0.0
    df['SL'] = 0.0
    df['Target'] = 0.0
    
    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        
        if threshold_type == "LONG (Price >= Threshold)":
            if current_price >= threshold:
                df.loc[i, 'Signal'] = 1
                df.loc[i, 'Entry_Price'] = current_price
        elif threshold_type == "SHORT (Price >= Threshold)":
            if current_price >= threshold:
                df.loc[i, 'Signal'] = -1
                df.loc[i, 'Entry_Price'] = current_price
        elif threshold_type == "LONG (Price <= Threshold)":
            if current_price <= threshold:
                df.loc[i, 'Signal'] = 1
                df.loc[i, 'Entry_Price'] = current_price
        elif threshold_type == "SHORT (Price <= Threshold)":
            if current_price <= threshold:
                df.loc[i, 'Signal'] = -1
                df.loc[i, 'Entry_Price'] = current_price
    
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
    df['Entry_Price'] = 0.0
    df['SL'] = 0.0
    df['Target'] = 0.0
    
    for i in range(len(df)):
        if df['RSI'].iloc[i] > 80 and df['ADX'].iloc[i] < 20 and df['EMA1'].iloc[i] < df['EMA2'].iloc[i]:
            df.loc[i, 'Signal'] = -1
            df.loc[i, 'Entry_Price'] = df['Close'].iloc[i]
        elif df['RSI'].iloc[i] < 20 and df['ADX'].iloc[i] > 20 and df['EMA1'].iloc[i] > df['EMA2'].iloc[i]:
            df.loc[i, 'Signal'] = 1
            df.loc[i, 'Entry_Price'] = df['Close'].iloc[i]
    
    return df

def ai_price_action_strategy(df, config):
    """AI Price Action Analysis Strategy"""
    df['EMA_20'] = calculate_ema(df['Close'], 20)
    df['EMA_50'] = calculate_ema(df['Close'], 50)
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    df['ATR'] = calculate_atr(df, 14)
    
    df['Signal'] = 0
    df['Entry_Price'] = 0.0
    df['SL'] = 0.0
    df['Target'] = 0.0
    df['AI_Confidence'] = 0.0
    df['AI_Analysis'] = ''
    
    for i in range(50, len(df)):
        score = 0
        analysis_parts = []
        
        if df['EMA_20'].iloc[i] > df['EMA_50'].iloc[i]:
            score += 2
            analysis_parts.append("Bullish trend (EMA20 > EMA50)")
        elif df['EMA_20'].iloc[i] < df['EMA_50'].iloc[i]:
            score -= 2
            analysis_parts.append("Bearish trend (EMA20 < EMA50)")
        
        if df['RSI'].iloc[i] < 30:
            score += 2
            analysis_parts.append("Oversold RSI (<30)")
        elif df['RSI'].iloc[i] > 70:
            score -= 2
            analysis_parts.append("Overbought RSI (>70)")
        
        if df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]:
            score += 1
            analysis_parts.append("Bullish MACD")
        else:
            score -= 1
            analysis_parts.append("Bearish MACD")
        
        if df['Close'].iloc[i] < df['BB_Lower'].iloc[i]:
            score += 1
            analysis_parts.append("Price below lower BB")
        elif df['Close'].iloc[i] > df['BB_Upper'].iloc[i]:
            score -= 1
            analysis_parts.append("Price above upper BB")
        
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            avg_vol = df['Volume'].iloc[i-20:i].mean()
            if df['Volume'].iloc[i] > avg_vol * 1.5:
                if score > 0:
                    score += 1
                    analysis_parts.append("High volume confirms trend")
                else:
                    score -= 1
        
        confidence = min(abs(score) / 8 * 100, 100)
        df.loc[i, 'AI_Confidence'] = confidence
        df.loc[i, 'AI_Analysis'] = " | ".join(analysis_parts)
        
        if score >= 4:
            df.loc[i, 'Signal'] = 1
            df.loc[i, 'Entry_Price'] = df['Close'].iloc[i]
            df.loc[i, 'SL'] = df['Close'].iloc[i] - (df['ATR'].iloc[i] * 1.5)
            df.loc[i, 'Target'] = df['Close'].iloc[i] + (df['ATR'].iloc[i] * 3)
        elif score <= -4:
            df.loc[i, 'Signal'] = -1
            df.loc[i, 'Entry_Price'] = df['Close'].iloc[i]
            df.loc[i, 'SL'] = df['Close'].iloc[i] + (df['ATR'].iloc[i] * 1.5)
            df.loc[i, 'Target'] = df['Close'].iloc[i] - (df['ATR'].iloc[i] * 3)
    
    return df

def custom_strategy_builder(df, conditions):
    """Custom Strategy Builder"""
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['ADX'] = calculate_adx(df, 14)
    df['EMA_Fast'] = calculate_ema(df['Close'], 9)
    df['EMA_Slow'] = calculate_ema(df['Close'], 21)
    df['EMA_20'] = calculate_ema(df['Close'], 20)
    df['EMA_50'] = calculate_ema(df['Close'], 50)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    df['ATR'] = calculate_atr(df, 14)
    df['SuperTrend'] = calculate_supertrend(df)
    df['VWAP'] = calculate_vwap(df)
    df['KC_Upper'], df['KC_Middle'], df['KC_Lower'] = calculate_keltner_channel(df)
    support, resistance = calculate_support_resistance(df)
    df['Support'] = support
    df['Resistance'] = resistance
    
    df['Signal'] = 0
    df['Entry_Price'] = 0.0
    df['SL'] = 0.0
    df['Target'] = 0.0
    
    buy_conditions = [c for c in conditions if c['action'] == 'BUY' and c['use_condition']]
    sell_conditions = [c for c in conditions if c['action'] == 'SELL' and c['use_condition']]
    
    for i in range(50, len(df)):
        if buy_conditions:
            buy_signal = all(check_condition(df, i, cond) for cond in buy_conditions)
            if buy_signal:
                df.loc[i, 'Signal'] = 1
                df.loc[i, 'Entry_Price'] = df['Close'].iloc[i]
        
        if sell_conditions:
            sell_signal = all(check_condition(df, i, cond) for cond in sell_conditions)
            if sell_signal:
                df.loc[i, 'Signal'] = -1
                df.loc[i, 'Entry_Price'] = df['Close'].iloc[i]
    
    return df

def check_condition(df, i, condition):
    """Check if a custom condition is met"""
    indicator1 = condition.get('indicator1', 'Price')
    operator = condition['operator']
    indicator2 = condition.get('indicator2')
    value = condition.get('value', 0)
    use_indicator = condition.get('use_indicator', False)
    
    # Get first indicator value
    if indicator1 == 'Price':
        val1 = df['Close'].iloc[i]
    elif indicator1 in df.columns:
        val1 = df[indicator1].iloc[i]
    else:
        return False
    
    # Get comparison value
    if use_indicator and indicator2:
        if indicator2 in df.columns:
            val2 = df[indicator2].iloc[i]
        else:
            return False
    else:
        val2 = value
    
    # Check operator
    if operator == '>':
        return val1 > val2
    elif operator == '<':
        return val1 < val2
    elif operator == '>=':
        return val1 >= val2
    elif operator == '<=':
        return val1 <= val2
    elif operator == '==':
        return val1 == val2
    elif operator == 'crosses_above':
        if i > 0:
            prev_val1 = df[indicator1].iloc[i-1] if indicator1 != 'Price' else df['Close'].iloc[i-1]
            return val1 > val2 and prev_val1 <= val2
    elif operator == 'crosses_below':
        if i > 0:
            prev_val1 = df[indicator1].iloc[i-1] if indicator1 != 'Price' else df['Close'].iloc[i-1]
            return val1 < val2 and prev_val1 >= val2
    
    return False

# ========================================
# STOP LOSS & TARGET CALCULATION
# ========================================

def calculate_sl(df, i, signal, sl_type, config, position_state):
    """Calculate stop loss based on type"""
    entry_price = position_state.get('entry_price', df['Close'].iloc[i])
    sl_points = config.get('sl_points', 10)
    atr_multiplier = config.get('atr_multiplier', 1.5)
    min_sl_distance = config.get('min_sl_distance', 10)
    
    if sl_type == "Custom Points":
        if signal == 1:
            sl = entry_price - sl_points
            if abs(entry_price - sl) < min_sl_distance:
                sl = entry_price - min_sl_distance
            return sl
        else:
            sl = entry_price + sl_points
            if abs(entry_price - sl) < min_sl_distance:
                sl = entry_price + min_sl_distance
            return sl
    
    elif sl_type == "ATR-based":
        atr = calculate_atr(df, 14).iloc[i]
        if signal == 1:
            sl = entry_price - (atr * atr_multiplier)
            if abs(entry_price - sl) < min_sl_distance:
                sl = entry_price - min_sl_distance
            return sl
        else:
            sl = entry_price + (atr * atr_multiplier)
            if abs(entry_price - sl) < min_sl_distance:
                sl = entry_price + min_sl_distance
            return sl
    
    elif sl_type == "Current Candle Low/High":
        if signal == 1:
            return df['Low'].iloc[i]
        else:
            return df['High'].iloc[i]
    
    elif sl_type == "Previous Candle Low/High":
        if i > 0:
            if signal == 1:
                return df['Low'].iloc[i-1]
            else:
                return df['High'].iloc[i-1]
        return calculate_sl(df, i, signal, "Custom Points", config, position_state)
    
    elif sl_type == "Current Swing Low/High":
        swing_highs, swing_lows = detect_swing_highs_lows(df.iloc[:i+1])
        if signal == 1 and swing_lows:
            return swing_lows[-1][1]
        elif signal == -1 and swing_highs:
            return swing_highs[-1][1]
        return calculate_sl(df, i, signal, "Custom Points", config, position_state)
    
    elif sl_type == "Previous Swing Low/High":
        swing_highs, swing_lows = detect_swing_highs_lows(df.iloc[:i+1])
        if signal == 1 and len(swing_lows) >= 2:
            return swing_lows[-2][1]
        elif signal == -1 and len(swing_highs) >= 2:
            return swing_highs[-2][1]
        return calculate_sl(df, i, signal, "Custom Points", config, position_state)
    
    else:
        if signal == 1:
            sl = entry_price - sl_points
            if abs(entry_price - sl) < min_sl_distance:
                sl = entry_price - min_sl_distance
            return sl
        else:
            sl = entry_price + sl_points
            if abs(entry_price - sl) < min_sl_distance:
                sl = entry_price + min_sl_distance
            return sl

def calculate_target(df, i, signal, target_type, config, position_state):
    """Calculate target based on type"""
    entry_price = position_state.get('entry_price', df['Close'].iloc[i])
    target_points = config.get('target_points', 20)
    atr_multiplier = config.get('atr_multiplier', 3)
    min_target_distance = config.get('min_target_distance', 15)
    
    if target_type == "Custom Points":
        if signal == 1:
            target = entry_price + target_points
            if abs(target - entry_price) < min_target_distance:
                target = entry_price + min_target_distance
            return target
        else:
            target = entry_price - target_points
            if abs(target - entry_price) < min_target_distance:
                target = entry_price - min_target_distance
            return target
    
    elif target_type == "ATR-based":
        atr = calculate_atr(df, 14).iloc[i]
        if signal == 1:
            target = entry_price + (atr * atr_multiplier)
            if abs(target - entry_price) < min_target_distance:
                target = entry_price + min_target_distance
            return target
        else:
            target = entry_price - (atr * atr_multiplier)
            if abs(target - entry_price) < min_target_distance:
                target = entry_price - min_target_distance
            return target
    
    elif target_type == "Risk-Reward Based":
        sl = position_state.get('sl', entry_price)
        risk = abs(entry_price - sl)
        rr_ratio = config.get('rr_ratio', 2)
        if signal == 1:
            return entry_price + (risk * rr_ratio)
        else:
            return entry_price - (risk * rr_ratio)
    
    elif target_type == "Current Candle Low/High":
        if signal == 1:
            return df['High'].iloc[i]
        else:
            return df['Low'].iloc[i]
    
    elif target_type == "Previous Candle Low/High":
        if i > 0:
            if signal == 1:
                return df['High'].iloc[i-1]
            else:
                return df['Low'].iloc[i-1]
        return calculate_target(df, i, signal, "Custom Points", config, position_state)
    
    elif target_type == "Current Swing Low/High":
        swing_highs, swing_lows = detect_swing_highs_lows(df.iloc[:i+1])
        if signal == 1 and swing_highs:
            return swing_highs[-1][1]
        elif signal == -1 and swing_lows:
            return swing_lows[-1][1]
        return calculate_target(df, i, signal, "Custom Points", config, position_state)
    
    elif target_type == "Previous Swing Low/High":
        swing_highs, swing_lows = detect_swing_highs_lows(df.iloc[:i+1])
        if signal == 1 and len(swing_highs) >= 2:
            return swing_highs[-2][1]
        elif signal == -1 and len(swing_lows) >= 2:
            return swing_lows[-2][1]
        return calculate_target(df, i, signal, "Custom Points", config, position_state)
    
    elif target_type == "Signal-based":
        return 0
    
    else:
        if signal == 1:
            target = entry_price + target_points
            if abs(target - entry_price) < min_target_distance:
                target = entry_price + min_target_distance
            return target
        else:
            target = entry_price - target_points
            if abs(target - entry_price) < min_target_distance:
                target = entry_price - min_target_distance
            return target

def update_trailing_sl(current_price, signal, sl_points, current_sl, trailing_high, trailing_low, trailing_threshold):
    """Update trailing stop loss"""
    if signal == 1:
        if trailing_high is None or current_price > trailing_high:
            if trailing_high is None or (current_price - trailing_high) >= trailing_threshold:
                new_sl = current_price - sl_points
                if new_sl > current_sl:
                    return new_sl, current_price
        return current_sl, trailing_high if trailing_high else current_price
    else:
        if trailing_low is None or current_price < trailing_low:
            if trailing_low is None or (trailing_low - current_price) >= trailing_threshold:
                new_sl = current_price + sl_points
                if new_sl < current_sl:
                    return new_sl, current_price
        return current_sl, trailing_low if trailing_low else current_price

def check_signal_based_exit(df, i, signal):
    """Check for signal-based exit"""
    if i < 1:
        return False
    
    if 'EMA_Fast' not in df.columns or 'EMA_Slow' not in df.columns:
        return False
    
    if signal == 1:
        return (df['EMA_Fast'].iloc[i] < df['EMA_Slow'].iloc[i] and 
                df['EMA_Fast'].iloc[i-1] >= df['EMA_Slow'].iloc[i-1])
    
    elif signal == -1:
        return (df['EMA_Fast'].iloc[i] > df['EMA_Slow'].iloc[i] and 
                df['EMA_Fast'].iloc[i-1] <= df['EMA_Slow'].iloc[i-1])
    
    return False

# ========================================
# BACKTEST ENGINE
# ========================================

def run_backtest(df, strategy, config, quantity):
    """Run backtest on historical data"""
    if strategy == "EMA Crossover":
        df = ema_crossover_strategy(df, config)
    elif strategy == "Simple Buy":
        df = simple_buy_strategy(df, config)
    elif strategy == "Simple Sell":
        df = simple_sell_strategy(df, config)
    elif strategy == "Price Crosses Threshold":
        df = price_threshold_strategy(df, config)
    elif strategy == "RSI-ADX-EMA":
        df = rsi_adx_ema_strategy(df, config)
    elif strategy == "AI Price Action Analysis":
        df = ai_price_action_strategy(df, config)
    elif strategy == "Custom Strategy Builder":
        df = custom_strategy_builder(df, config.get('custom_conditions', []))
    
    trades = []
    position = None
    
    sl_type = config.get('sl_type', 'Custom Points')
    target_type = config.get('target_type', 'Custom Points')
    
    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        
        if position is None and df['Signal'].iloc[i] != 0:
            signal = df['Signal'].iloc[i]
            entry_price = df['Entry_Price'].iloc[i]
            
            position_state = {
                'entry_price': entry_price,
                'signal': signal
            }
            
            sl = calculate_sl(df, i, signal, sl_type, config, position_state)
            target = calculate_target(df, i, signal, target_type, config, position_state)
            
            position = {
                'entry_time': df['Datetime'].iloc[i],
                'entry_price': entry_price,
                'signal': signal,
                'sl': sl,
                'target': target,
                'quantity': quantity,
                'highest_price': current_price,
                'lowest_price': current_price,
                'partial_exit_done': False,
                'breakeven_activated': False,
                'trailing_sl_high': current_price if signal == 1 else None,
                'trailing_sl_low': current_price if signal == -1 else None
            }
            
            position_state['sl'] = sl
            position_state['target'] = target
        
        if position is not None:
            if current_price > position['highest_price']:
                position['highest_price'] = current_price
            if current_price < position['lowest_price']:
                position['lowest_price'] = current_price
            
            exit_reason = None
            exit_price = current_price
            
            if position['signal'] == 1:
                # Update trailing SL
                if sl_type.startswith("Trailing"):
                    sl_points = config.get('sl_points', 10)
                    trailing_threshold = config.get('trailing_threshold', 0)
                    new_sl, new_high = update_trailing_sl(
                        current_price, 1, sl_points, position['sl'],
                        position['trailing_sl_high'], None, trailing_threshold
                    )
                    position['sl'] = new_sl
                    position['trailing_sl_high'] = new_high
                
                if sl_type == "Signal-based":
                    if check_signal_based_exit(df, i, 1):
                        exit_reason = "Signal-based SL"
                elif current_price <= position['sl']:
                    exit_reason = "Stop Loss"
                
                if target_type == "Signal-based":
                    if check_signal_based_exit(df, i, 1):
                        exit_reason = "Signal-based Target"
                elif target_type not in ["Trailing Target (Points)", "Trailing Target + Signal Based"]:
                    if position['target'] > 0 and current_price >= position['target']:
                        if target_type == "50% Exit at Target (Partial)" and not position['partial_exit_done']:
                            position['partial_exit_done'] = True
                        else:
                            exit_reason = "Target Hit"
                
                if config.get('sl_type') == "Break-even After 50% Target" and not position['breakeven_activated']:
                    target_dist = position['target'] - position['entry_price']
                    profit_dist = current_price - position['entry_price']
                    if profit_dist >= target_dist * 0.5:
                        position['sl'] = position['entry_price']
                        position['breakeven_activated'] = True
            
            else:
                # Update trailing SL
                if sl_type.startswith("Trailing"):
                    sl_points = config.get('sl_points', 10)
                    trailing_threshold = config.get('trailing_threshold', 0)
                    new_sl, new_low = update_trailing_sl(
                        current_price, -1, sl_points, position['sl'],
                        None, position['trailing_sl_low'], trailing_threshold
                    )
                    position['sl'] = new_sl
                    position['trailing_sl_low'] = new_low
                
                if sl_type == "Signal-based":
                    if check_signal_based_exit(df, i, -1):
                        exit_reason = "Signal-based SL"
                elif current_price >= position['sl']:
                    exit_reason = "Stop Loss"
                
                if target_type == "Signal-based":
                    if check_signal_based_exit(df, i, -1):
                        exit_reason = "Signal-based Target"
                elif target_type not in ["Trailing Target (Points)", "Trailing Target + Signal Based"]:
                    if position['target'] > 0 and current_price <= position['target']:
                        if target_type == "50% Exit at Target (Partial)" and not position['partial_exit_done']:
                            position['partial_exit_done'] = True
                        else:
                            exit_reason = "Target Hit"
                
                if config.get('sl_type') == "Break-even After 50% Target" and not position['breakeven_activated']:
                    target_dist = position['entry_price'] - position['target']
                    profit_dist = position['entry_price'] - current_price
                    if profit_dist >= target_dist * 0.5:
                        position['sl'] = position['entry_price']
                        position['breakeven_activated'] = True
            
            if exit_reason:
                duration = df['Datetime'].iloc[i] - position['entry_time']
                pnl = (exit_price - position['entry_price']) * position['signal'] * position['quantity']
                
                trade = {
                    'Entry Time': position['entry_time'],
                    'Exit Time': df['Datetime'].iloc[i],
                    'Duration': str(duration),
                    'Signal': 'LONG' if position['signal'] == 1 else 'SHORT',
                    'Entry Price': position['entry_price'],
                    'Exit Price': exit_price,
                    'SL': position['sl'],
                    'Target': position['target'],
                    'Exit Reason': exit_reason,
                    'P&L': pnl,
                    'Highest Price': position['highest_price'],
                    'Lowest Price': position['lowest_price'],
                    'Range': position['highest_price'] - position['lowest_price']
                }
                
                trades.append(trade)
                position = None
    
    if trades:
        trades_df = pd.DataFrame(trades)
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['P&L'] > 0])
        losing_trades = len(trades_df[trades_df['P&L'] < 0])
        accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = trades_df['P&L'].sum()
        
        durations = [pd.Timedelta(d).total_seconds() / 3600 for d in trades_df['Duration']]
        avg_duration = np.mean(durations) if durations else 0
        
        results = {
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Accuracy': accuracy,
            'Total P&L': total_pnl,
            'Avg Duration (hours)': avg_duration,
            'Trades': trades_df
        }
    else:
        results = {
            'Total Trades': 0,
            'Winning Trades': 0,
            'Losing Trades': 0,
            'Accuracy': 0,
            'Total P&L': 0,
            'Avg Duration (hours)': 0,
            'Trades': pd.DataFrame()
        }
    
    return results

# ========================================
# LIVE TRADING WORKER
# ========================================

def live_trading_worker(ticker, interval, period, strategy, config, quantity):
    """Background worker for live trading"""
    while st.session_state.get('trading_active', False):
        try:
            df = fetch_data(ticker, interval, period, "Live Trading")
            
            if df is None or len(df) == 0:
                add_log("Failed to fetch data")
                continue
            
            if strategy == "EMA Crossover":
                df = ema_crossover_strategy(df, config)
            elif strategy == "Simple Buy":
                df = simple_buy_strategy(df, config)
            elif strategy == "Simple Sell":
                df = simple_sell_strategy(df, config)
            elif strategy == "Price Crosses Threshold":
                df = price_threshold_strategy(df, config)
            elif strategy == "RSI-ADX-EMA":
                df = rsi_adx_ema_strategy(df, config)
            elif strategy == "AI Price Action Analysis":
                df = ai_price_action_strategy(df, config)
            elif strategy == "Custom Strategy Builder":
                df = custom_strategy_builder(df, config.get('custom_conditions', []))
            
            st.session_state['current_data'] = df
            st.session_state['last_update'] = get_ist_time()
            
            current_price = df['Close'].iloc[-1]
            current_signal = df['Signal'].iloc[-1]
            
            position = st.session_state.get('position')
            
            if position is None and current_signal != 0:
                entry_price = df['Entry_Price'].iloc[-1]
                
                position_state = {
                    'entry_price': entry_price,
                    'signal': current_signal
                }
                
                sl = calculate_sl(df, len(df)-1, current_signal, config['sl_type'], config, position_state)
                target = calculate_target(df, len(df)-1, current_signal, config['target_type'], config, position_state)
                
                position = {
                    'entry_time': get_ist_time(),
                    'entry_price': entry_price,
                    'signal': current_signal,
                    'sl': sl,
                    'target': target,
                    'quantity': quantity,
                    'highest_price': current_price,
                    'lowest_price': current_price,
                    'partial_exit_done': False,
                    'breakeven_activated': False
                }
                
                st.session_state['position'] = position
                st.session_state['highest_price'] = current_price
                st.session_state['lowest_price'] = current_price
                st.session_state['trailing_sl_high'] = current_price if current_signal == 1 else None
                st.session_state['trailing_sl_low'] = current_price if current_signal == -1 else None
                
                signal_type = "LONG" if current_signal == 1 else "SHORT"
                sl_str = f"{sl:.2f}" if sl != 0 else "Signal Based"
                target_str = f"{target:.2f}" if target != 0 else "Signal Based"
                add_log(f"{signal_type} Entry at {entry_price:.2f} | SL: {sl_str} | Target: {target_str}")
                
                # Placeholder for Dhan order
                # if current_signal == 1:
                #     dhan.place_order(...)
                # else:
                #     dhan.place_order(...)
            
            if position is not None:
                if current_price > st.session_state['highest_price']:
                    st.session_state['highest_price'] = current_price
                if current_price < st.session_state['lowest_price']:
                    st.session_state['lowest_price'] = current_price
                
                position['highest_price'] = st.session_state['highest_price']
                position['lowest_price'] = st.session_state['lowest_price']
                
                exit_reason = None
                exit_price = current_price
                
                if position['signal'] == 1:
                    if config['sl_type'].startswith("Trailing"):
                        sl_points = config.get('sl_points', 10)
                        trailing_threshold = config.get('trailing_threshold', 0)
                        new_sl, new_high = update_trailing_sl(
                            current_price, 1, sl_points, position['sl'],
                            st.session_state.get('trailing_sl_high'), None, trailing_threshold
                        )
                        if new_sl != position['sl']:
                            position['sl'] = new_sl
                            st.session_state['trailing_sl_high'] = new_high
                            add_log(f"Trailing SL updated to {new_sl:.2f}")
                    
                    if config['sl_type'] == "Signal-based":
                        if check_signal_based_exit(df, len(df)-1, 1):
                            exit_reason = "Signal-based SL"
                    elif current_price <= position['sl']:
                        exit_reason = "Stop Loss"
                    
                    if config['target_type'] == "Signal-based":
                        if check_signal_based_exit(df, len(df)-1, 1):
                            exit_reason = "Signal-based Target"
                    elif config['target_type'] not in ["Trailing Target (Points)", "Trailing Target + Signal Based"]:
                        if position['target'] > 0 and current_price >= position['target']:
                            if config['target_type'] == "50% Exit at Target (Partial)" and not position['partial_exit_done']:
                                position['partial_exit_done'] = True
                                add_log("50% position exited - trailing remaining")
                            else:
                                exit_reason = "Target Hit"
                    
                    if config.get('sl_type') == "Break-even After 50% Target" and not position['breakeven_activated']:
                        target_dist = position['target'] - position['entry_price']
                        profit_dist = current_price - position['entry_price']
                        if profit_dist >= target_dist * 0.5:
                            position['sl'] = position['entry_price']
                            position['breakeven_activated'] = True
                            add_log("Break-even SL activated")
                
                else:
                    if config['sl_type'].startswith("Trailing"):
                        sl_points = config.get('sl_points', 10)
                        trailing_threshold = config.get('trailing_threshold', 0)
                        new_sl, new_low = update_trailing_sl(
                            current_price, -1, sl_points, position['sl'],
                            None, st.session_state.get('trailing_sl_low'), trailing_threshold
                        )
                        if new_sl != position['sl']:
                            position['sl'] = new_sl
                            st.session_state['trailing_sl_low'] = new_low
                            add_log(f"Trailing SL updated to {new_sl:.2f}")
                    
                    if config['sl_type'] == "Signal-based":
                        if check_signal_based_exit(df, len(df)-1, -1):
                            exit_reason = "Signal-based SL"
                    elif current_price >= position['sl']:
                        exit_reason = "Stop Loss"
                    
                    if config['target_type'] == "Signal-based":
                        if check_signal_based_exit(df, len(df)-1, -1):
                            exit_reason = "Signal-based Target"
                    elif config['target_type'] not in ["Trailing Target (Points)", "Trailing Target + Signal Based"]:
                        if position['target'] > 0 and current_price <= position['target']:
                            if config['target_type'] == "50% Exit at Target (Partial)" and not position['partial_exit_done']:
                                position['partial_exit_done'] = True
                                add_log("50% position exited - trailing remaining")
                            else:
                                exit_reason = "Target Hit"
                    
                    if config.get('sl_type') == "Break-even After 50% Target" and not position['breakeven_activated']:
                        target_dist = position['entry_price'] - position['target']
                        profit_dist = position['entry_price'] - current_price
                        if profit_dist >= target_dist * 0.5:
                            position['sl'] = position['entry_price']
                            position['breakeven_activated'] = True
                            add_log("Break-even SL activated")
                
                st.session_state['position'] = position
                
                if exit_reason:
                    duration = get_ist_time() - position['entry_time']
                    pnl = (exit_price - position['entry_price']) * position['signal'] * position['quantity']
                    
                    trade = {
                        'Entry Time': position['entry_time'],
                        'Exit Time': get_ist_time(),
                        'Duration': str(duration),
                        'Signal': 'LONG' if position['signal'] == 1 else 'SHORT',
                        'Entry Price': position['entry_price'],
                        'Exit Price': exit_price,
                        'SL': position['sl'],
                        'Target': position['target'],
                        'Exit Reason': exit_reason,
                        'P&L': pnl,
                        'Highest Price': position['highest_price'],
                        'Lowest Price': position['lowest_price'],
                        'Range': position['highest_price'] - position['lowest_price']
                    }
                    
                    st.session_state['trade_history'].append(trade)
                    st.session_state['trade_history'] = st.session_state['trade_history']
                    
                    add_log(f"Exit: {exit_reason} | Price: {exit_price:.2f} | P&L: {pnl:.2f}")
                    
                    # Placeholder for Dhan order
                    # if position['signal'] == 1:
                    #     dhan.place_order(...)
                    # else:
                    #     dhan.place_order(...)
                    
                    reset_position_state()
        
        except Exception as e:
            add_log(f"Error in trading loop: {str(e)}")

# ========================================
# STREAMLIT APP
# ========================================

def main():
    st.set_page_config(page_title="Professional Trading System", layout="wide")
    
    # Initialize session state
    if 'trading_active' not in st.session_state:
        st.session_state['trading_active'] = False
    if 'current_data' not in st.session_state:
        st.session_state['current_data'] = None
    if 'position' not in st.session_state:
        st.session_state['position'] = None
    if 'trade_history' not in st.session_state:
        st.session_state['trade_history'] = []
    if 'trade_logs' not in st.session_state:
        st.session_state['trade_logs'] = []
    if 'trailing_sl_high' not in st.session_state:
        st.session_state['trailing_sl_high'] = None
    if 'trailing_sl_low' not in st.session_state:
        st.session_state['trailing_sl_low'] = None
    if 'trailing_target_high' not in st.session_state:
        st.session_state['trailing_target_high'] = None
    if 'trailing_target_low' not in st.session_state:
        st.session_state['trailing_target_low'] = None
    if 'trailing_profit_points' not in st.session_state:
        st.session_state['trailing_profit_points'] = 0
    if 'threshold_crossed' not in st.session_state:
        st.session_state['threshold_crossed'] = False
    if 'highest_price' not in st.session_state:
        st.session_state['highest_price'] = None
    if 'lowest_price' not in st.session_state:
        st.session_state['lowest_price'] = None
    if 'custom_conditions' not in st.session_state:
        st.session_state['custom_conditions'] = []
    if 'partial_exit_done' not in st.session_state:
        st.session_state['partial_exit_done'] = False
    if 'breakeven_activated' not in st.session_state:
        st.session_state['breakeven_activated'] = False
    if 'last_update' not in st.session_state:
        st.session_state['last_update'] = None
    if 'trading_thread' not in st.session_state:
        st.session_state['trading_thread'] = None
    
    st.title("🚀 Professional Trading System")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("Configuration")
        
        mode = st.selectbox("Mode", ["Backtest", "Live Trading"], key="mode_select")
        
        asset_type = st.selectbox("Asset Type", [
            "Indian Indices", "Crypto", "Forex", "Commodities", "Custom Ticker"
        ], key="asset_type_select")
        
        if asset_type == "Indian Indices":
            ticker = st.selectbox("Select Index", [
                "^NSEI", "^NSEBANK", "^BSESN"
            ], format_func=lambda x: {
                "^NSEI": "NIFTY 50",
                "^NSEBANK": "BANKNIFTY",
                "^BSESN": "SENSEX"
            }[x], key="index_ticker")
        elif asset_type == "Crypto":
            ticker = st.selectbox("Select Crypto", [
                "BTC-USD", "ETH-USD"
            ], format_func=lambda x: x.replace("-USD", ""), key="crypto_ticker")
        elif asset_type == "Forex":
            ticker = st.selectbox("Select Forex Pair", [
                "USDINR=X", "EURUSD=X", "GBPUSD=X"
            ], key="forex_ticker")
        elif asset_type == "Commodities":
            ticker = st.selectbox("Select Commodity", [
                "GC=F", "SI=F"
            ], format_func=lambda x: {
                "GC=F": "Gold",
                "SI=F": "Silver"
            }[x], key="commodity_ticker")
        else:
            ticker = st.text_input("Enter Custom Ticker", "AAPL", key="custom_ticker_input")
        
        interval = st.selectbox("Interval", [
            "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"
        ], key="interval_select")
        
        period_options = {
            "1m": ["1d", "5d"],
            "5m": ["1d", "1mo"],
            "15m": ["1mo"], "30m": ["1mo"], "1h": ["1mo"], "4h": ["1mo"],
            "1d": ["1mo", "1y", "2y", "5y"],
            "1wk": ["1mo", "1y", "5y", "10y", "15y", "20y"],
            "1mo": ["1y", "2y", "5y", "10y", "15y", "20y", "25y", "30y"]
        }
        
        period = st.selectbox("Period", period_options.get(interval, ["1mo"]), key="period_select")
        
        quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="quantity_input")
        
        strategy = st.selectbox("Strategy", [
            "EMA Crossover",
            "Simple Buy",
            "Simple Sell",
            "Price Crosses Threshold",
            "RSI-ADX-EMA",
            "AI Price Action Analysis",
            "Custom Strategy Builder"
        ], key="strategy_select")
        
        config = {}
        
        if strategy == "EMA Crossover":
            st.subheader("EMA Parameters")
            config['ema_fast'] = st.number_input("EMA Fast", min_value=1, value=9, key="ema_fast_input")
            config['ema_slow'] = st.number_input("EMA Slow", min_value=1, value=15, key="ema_slow_input")
            config['min_angle'] = st.number_input("Min Angle (degrees)", min_value=0.0, value=1.0, key="min_angle_input")
            
            st.subheader("Entry Filter")
            config['entry_filter'] = st.selectbox("Entry Filter Type", [
                "Simple Crossover",
                "Custom Candle (Points)",
                "ATR-based Candle"
            ], key="entry_filter_select")
            
            if config['entry_filter'] == "Custom Candle (Points)":
                config['custom_points'] = st.number_input("Custom Points", min_value=0.0, value=10.0, key="custom_points_input")
            elif config['entry_filter'] == "ATR-based Candle":
                config['atr_multiplier'] = st.number_input("ATR Multiplier", min_value=0.1, value=1.0, key="atr_mult_input")
            
            st.subheader("ADX Filter")
            config['use_adx'] = st.checkbox("Use ADX Filter", key="use_adx_check")
            if config['use_adx']:
                config['adx_period'] = st.number_input("ADX Period", min_value=1, value=14, key="adx_period_input")
                config['adx_threshold'] = st.number_input("ADX Threshold", min_value=0.0, value=25.0, key="adx_threshold_input")
        
        elif strategy == "Price Crosses Threshold":
            config['threshold'] = st.number_input("Threshold Price", min_value=0.0, value=100.0, key="threshold_input")
            config['threshold_type'] = st.selectbox("Threshold Type", [
                "LONG (Price >= Threshold)",
                "SHORT (Price >= Threshold)",
                "LONG (Price <= Threshold)",
                "SHORT (Price <= Threshold)"
            ], key="threshold_type_select")
        
        elif strategy == "RSI-ADX-EMA":
            config['rsi_period'] = st.number_input("RSI Period", min_value=1, value=14, key="rsi_period_input")
            config['adx_period'] = st
