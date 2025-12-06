import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import time
from scipy import stats
from scipy.signal import argrelextrema
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Advanced Algo Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
    .signal-buy {background-color: #00ff0020; padding: 1rem; border-radius: 10px; border-left: 5px solid #00ff00;}
    .signal-sell {background-color: #ff000020; padding: 1rem; border-radius: 10px; border-left: 5px solid #ff0000;}
    .signal-hold {background-color: #ffa50020; padding: 1rem; border-radius: 10px; border-left: 5px solid #ffa500;}
    .live-guidance {background-color: #e8f5e9; padding: 2rem; border-radius: 15px; border: 3px solid #4caf50; margin: 1rem 0;}
    .target-box {background-color: #e3f2fd; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #2196f3; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results =                     status_text.text(f"Analyzing {ticker2_name} - {interval}/{period} ({current_analysis}/{total_analyses})")
                    
                    df2 = analyzer.fetch_data_with_retry(ticker2, period, interval)
                    
                    if df2 is not None and not df2.empty:
                        df2 = analyzer.calculate_indicators(df2)
                        
                        if f"{interval}_{period}" in all_results:
                            all_results[f"{interval}_{period}"]['ticker2'] = {'data': df2}
        
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis Complete!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.analysis_results = all_results
        st.session_state.ticker1_name = ticker1_name
        st.session_state.ticker1_symbol = ticker1
        st.session_state.ticker2_name = ticker2_name
        st.session_state.ticker2_symbol = ticker2
        st.success("✅ Analysis completed successfully!")
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        analyzer = TradingAnalyzer()
        
        st.markdown("---")
        st.markdown("## 📊 Comprehensive Analysis Summary")
        
        all_signals = []
        all_confidences = []
        all_reasons = []
        all_zscores = []
        
        for key, result in results.items():
            if 'ticker1' in result and 'signals' in result['ticker1']:
                signal_data = result['ticker1']['signals']
                signal = signal_data['signal']
                confidence = signal_data['confidence']
                
                if signal == 'BUY':
                    all_signals.append(1)
                elif signal == 'SELL':
                    all_signals.append(-1)
                else:
                    all_signals.append(0)
                
                all_confidences.append(confidence)
                all_reasons.extend(signal_data['reasons'])
                
                if 'price_zscore' in signal_data:
                    all_zscores.append(signal_data['price_zscore'])
        
        if len(all_signals) > 0:
            avg_signal = np.mean(all_signals)
            avg_confidence = np.mean(all_confidences)
            avg_zscore = np.mean(all_zscores) if all_zscores else 0
            
            if avg_signal > 0.2:
                final_signal = "BUY"
                signal_class = "signal-buy"
                signal_emoji = "🟢"
            elif avg_signal < -0.2:
                final_signal = "SELL"
                signal_class = "signal-sell"
                signal_emoji = "🔴"
            else:
                final_signal = "HOLD"
                signal_class = "signal-hold"
                signal_emoji = "🟡"
            
            # Calculate targets and stops
            latest_key = [k for k in results.keys() if '1d_' in k][0]
            latest_data = results[latest_key]['ticker1']['data']
            targets_stops = analyzer.calculate_targets_stops(latest_data, final_signal)
            
            st.markdown(f'<div class="{signal_class}">', unsafe_allow_html=True)
            st.markdown(f"### {signal_emoji} Final Recommendation: **{final_signal}**")
            st.markdown(f"**Confidence Level:** {avg_confidence:.1f}%")
            st.markdown(f"**Average Z-Score:** {avg_zscore:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display Entry, Targets, and Stop-Loss
            if final_signal != 'HOLD':
                st.markdown('<div class="target-box">', unsafe_allow_html=True)
                st.markdown("### 🎯 Entry, Targets & Stop-Loss (Based on ATR)")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Entry Price", f"₹{targets_stops['entry']:.2f}")
                col2.metric("Stop Loss", f"₹{targets_stops['stop_loss']:.2f}", f"-{targets_stops['risk']:.2f}")
                col3.metric("Target 1", f"₹{targets_stops['target1']:.2f}", f"+{targets_stops['reward1']:.2f}")
                col4.metric("Target 2", f"₹{targets_stops['target2']:.2f}")
                col5.metric("Target 3", f"₹{targets_stops['target3']:.2f}")
                
                st.markdown(f"""
                **Risk/Reward Ratio:** {targets_stops['rr_ratio1']:.2f}:1  
                **ATR Value:** ₹{targets_stops['atr']:.2f}  
                **Risk per share:** ₹{targets_stops['risk']:.2f}  
                **Reward per share (T1):** ₹{targets_stops['reward1']:.2f}
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("#### 🎯 Key Analysis Points:")
            reason_counts = {}
            for reason in all_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            
            cols = st.columns(2)
            for idx, (reason, count) in enumerate(top_reasons):
                col = cols[idx % 2]
                col.markdown(f"- {reason} *({count} timeframes)*")
            
            # RSI Divergence Analysis
            st.markdown("---")
            st.markdown("## 📊 RSI Divergence Analysis")
            
            divergence_key = [k for k in results.keys() if '1d_' in k][0]
            if divergence_key and 'rsi_divergence' in results[divergence_key]['ticker1']:
                div_info = results[divergence_key]['ticker1']['rsi_divergence']
                
                if div_info['type']:
                    div_color = "green" if div_info['type'] == 'Bullish' else "red"
                    st.markdown(f"### {div_info['type']} Divergence Detected! (Strength: {div_info['strength']:.1f}%)")
                    st.markdown(f"<p style='color:{div_color}'>{div_info['description']}</p>", unsafe_allow_html=True)
                else:
                    st.info(div_info['description'])
                
                # Plot Price and RSI with divergence
                df_for_plot = results[divergence_key]['ticker1']['data'].tail(100)
                fig_rsi = create_price_rsi_chart(df_for_plot, st.session_state.ticker1_name, div_info)
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            # Visual Analysis Section
            st.markdown("---")
            st.markdown("## 📈 Visual Analysis & Charts")
            
            chart_tabs = st.tabs(["Price vs Volatility", "Price vs Fibonacci", "Price vs Indicators", "Technical Summary"])
            
            with chart_tabs[0]:
                st.markdown("### Ticker Price vs Volatility")
                daily_key = [k for k in results.keys() if '1d_' in k][0]
                df_daily = results[daily_key]['ticker1']['data'].tail(100)
                
                fig_vol = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       vertical_spacing=0.03, row_heights=[0.7, 0.3],
                                       subplot_titles=(f'{st.session_state.ticker1_name} Price', 'Volatility %'))
                
                fig_vol.add_trace(go.Scatter(x=df_daily.index, y=df_daily['Close'], name='Price',
                                            line=dict(color='blue', width=2)), row=1, col=1)
                fig_vol.add_trace(go.Scatter(x=df_daily.index, y=df_daily['Volatility'], name='Volatility',
                                            line=dict(color='red', width=2), fill='tozeroy'), row=2, col=1)
                
                fig_vol.update_layout(height=500, showlegend=True)
                st.plotly_chart(fig_vol, use_container_width=True)
                
                st.markdown(f"""
                **Current Volatility:** {df_daily['Volatility'].iloc[-1]:.2f}%  
                **Average Volatility (20 days):** {df_daily['Volatility'].tail(20).mean():.2f}%  
                **Interpretation:** {'High volatility - expect larger price swings' if df_daily['Volatility'].iloc[-1] > 30 else 'Normal volatility - stable price movement'}
                """)
            
            with chart_tabs[1]:
                st.markdown("### Ticker Price vs Fibonacci Levels")
                
                if 'fib_levels' in results[daily_key]['ticker1'] and results[daily_key]['ticker1']['fib_levels']:
                    fib_data = results[daily_key]['ticker1']['fib_levels']
                    
                    fig_fib = go.Figure()
                    fig_fib.add_trace(go.Scatter(x=df_daily.index, y=df_daily['Close'], name='Price',
                                                line=dict(color='blue', width=2)))
                    
                    colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']
                    for idx, (level_name, level_price) in enumerate(fib_data['levels'].items()):
                        fig_fib.add_hline(y=level_price, line_dash="dash", 
                                         line_color=colors[idx % len(colors)],
                                         annotation_text=f"Fib {level_name}: ₹{level_price:.2f}")
                    
                    fig_fib.update_layout(height=500, title=f"{st.session_state.ticker1_name} with Fibonacci Levels",
                                         showlegend=True)
                    st.plotly_chart(fig_fib, use_container_width=True)
                    
                    st.markdown(f"""
                    **Swing High:** ₹{fib_data['swing_high']:.2f}  
                    **Swing Low:** ₹{fib_data['swing_low']:.2f}  
                    **Current Price:** ₹{fib_data['current_price']:.2f}  
                    **Closest Fibonacci Level:** {fib_data['closest_level']} at ₹{fib_data['closest_price']:.2f}  
                    **Distance:** {abs((fib_data['current_price'] - fib_data['closest_price']) / fib_data['current_price'] * 100):.2f}%
                    """)
            
            with chart_tabs[2]:
                st.markdown("### Technical Indicators Overview")
                
                fig_tech = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                        vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25],
                                        subplot_titles=(f'{st.session_state.ticker1_name} with EMAs', 
                                                       'MACD', 'ADX'))
                
                fig_tech.add_trace(go.Scatter(x=df_daily.index, y=df_daily['Close'], name='Price',
                                             line=dict(color='black', width=2)), row=1, col=1)
                fig_tech.add_trace(go.Scatter(x=df_daily.index, y=df_daily['EMA_20'], name='EMA20',
                                             line=dict(color='blue', width=1)), row=1, col=1)
                fig_tech.add_trace(go.Scatter(x=df_daily.index, y=df_daily['EMA_50'], name='EMA50',
                                             line=dict(color='orange', width=1)), row=1, col=1)
                
                fig_tech.add_trace(go.Scatter(x=df_daily.index, y=df_daily['MACD'], name='MACD',
                                             line=dict(color='blue', width=1)), row=2, col=1)
                fig_tech.add_trace(go.Scatter(x=df_daily.index, y=df_daily['MACD_Signal'], name='Signal',
                                             line=dict(color='red', width=1)), row=2, col=1)
                
                fig_tech.add_trace(go.Scatter(x=df_daily.index, y=df_daily['ADX'], name='ADX',
                                             line=dict(color='purple', width=2)), row=3, col=1)
                fig_tech.add_hline(y=25, line_dash="dash", line_color="green", row=3, col=1)
                
                fig_tech.update_layout(height=700, showlegend=True)
                st.plotly_chart(fig_tech, use_container_width=True)
            
            with chart_tabs[3]:
                st.markdown("### Current Technical Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                current_price = df_daily['Close'].iloc[-1]
                current_rsi = df_daily['RSI'].iloc[-1]
                current_adx = df_daily['ADX'].iloc[-1]
                current_vol = df_daily['Volatility'].iloc[-1]
                
                col1.metric("Current Price", f"₹{current_price:.2f}")
                col2.metric("RSI", f"{current_rsi:.1f}", 
                           "Oversold" if current_rsi < 30 else ("Overbought" if current_rsi > 70 else "Neutral"))
                col3.metric("ADX", f"{current_adx:.1f}",
                           "Strong Trend" if current_adx > 25 else "Weak Trend")
                col4.metric("Volatility", f"{current_vol:.2f}%")
                
                st.markdown("**Trend Analysis:**")
                if current_price > df_daily['EMA_20'].iloc[-1] > df_daily['EMA_50'].iloc[-1]:
                    st.success("✅ Strong Uptrend: Price > EMA20 > EMA50")
                elif current_price < df_daily['EMA_20'].iloc[-1] < df_daily['EMA_50'].iloc[-1]:
                    st.error("📉 Strong Downtrend: Price < EMA20 < EMA50")
                else:
                    st.warning("➡️ Sideways/Consolidation: Mixed signals")
            
            # Ratio Analysis Section
            if enable_ratio and ticker2:
                st.markdown("---")
                st.markdown("## 📊 Ratio Analysis & Visualization")
                
                ratio_keys = [k for k in results.keys() if 'ticker2' in results[k] and '1d_' in k]
                if ratio_keys:
                    key = ratio_keys[0]
                    df1 = results[key]['ticker1']['data']
                    df2 = results[key]['ticker2']['data']
                    
                    common_index = df1.index.intersection(df2.index)
                    df1_aligned = df1.loc[common_index]
                    df2_aligned = df2.loc[common_index]
                    
                    ratio_df = pd.DataFrame({
                        'DateTime': common_index,
                        'Ticker1_Price': df1_aligned['Close'].values,
                        'Ticker2_Price': df2_aligned['Close'].values,
                        'Ratio': df1_aligned['Close'].values / df2_aligned['Close'].values,
                        'Ticker1_RSI': df1_aligned['RSI'].values,
                        'Ticker2_RSI': df2_aligned['RSI'].values,
                        'Ticker1_Volatility': df1_aligned['Volatility'].values,
                        'Ticker2_Volatility': df2_aligned['Volatility'].values,
                        'Ticker1_ZScore': df1_aligned['Price_ZScore'].values,
                        'Ticker2_ZScore': df2_aligned['Price_ZScore'].values,
                    })
                    
                    ratio_df['Ratio_Returns'] = ratio_df['Ratio'].pct_change().fillna(0)
                    ratio_df['Ratio_RSI'] = TechnicalIndicators.calculate_rsi(pd.Series(ratio_df['Ratio'].values), 14)
                    ratio_df['Ratio_ZScore'] = TechnicalIndicators.calculate_zscore(pd.Series(ratio_df['Ratio'].values), 20)
                    
                    # Plot Ratio Chart
                    fig_ratio = create_ratio_chart(ratio_df, st.session_state.ticker1_name, st.session_state.ticker2_name)
                    st.plotly_chart(fig_ratio, use_container_width=True)
                    
                    # Ratio metrics
                    col1, col2, col3, col4 = st.columns(4)
                    current_ratio = ratio_df['Ratio'].iloc[-1]
                    ratio_change = ((ratio_df['Ratio'].iloc[-1] - ratio_df['Ratio'].iloc[-2]) / ratio_df['Ratio'].iloc[-2]) * 100
                    
                    col1.metric("Current Ratio", f"{current_ratio:.4f}", f"{ratio_change:+.2f}%")
                    col2.metric("Ratio RSI", f"{ratio_df['Ratio_RSI'].iloc[-1]:.1f}")
                    col3.metric("Ratio Z-Score", f"{ratio_df['Ratio_ZScore'].iloc[-1]:.2f}")
                    col4.metric("Spread Volatility", f"{ratio_df['Ratio_Returns'].std() * 100:.2f}%")
                    
                    # Ticker1 vs Ticker2 comparison chart
                    st.markdown("### Ticker1 vs Ticker2 Price Comparison")
                    
                    fig_comparison = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    fig_comparison.add_trace(
                        go.Scatter(x=ratio_df['DateTime'], y=ratio_df['Ticker1_Price'], 
                                  name=st.session_state.ticker1_name, line=dict(color='blue', width=2)),
                        secondary_y=False
                    )
                    
                    fig_comparison.add_trace(
                        go.Scatter(x=ratio_df['DateTime'], y=ratio_df['Ticker2_Price'],
                                  name=st.session_state.ticker2_name, line=dict(color='red', width=2)),
                        secondary_y=True
                    )
                    
                    fig_comparison.update_xaxes(title_text="Date")
                    fig_comparison.update_yaxes(title_text=st.session_state.ticker1_name, secondary_y=False)
                    fig_comparison.update_yaxes(title_text=st.session_state.ticker2_name, secondary_y=True)
                    fig_comparison.update_layout(height=400, title="Price Comparison")
                    
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Data table
                    st.markdown("#### Detailed Ratio Data (Last 20 rows)")
                    display_df = ratio_df[['DateTime', 'Ticker1_Price', 'Ticker2_Price', 'Ratio', 
                                           'Ticker1_RSI', 'Ticker2_RSI', 'Ratio_RSI', 
                                           'Ticker1_Volatility', 'Ticker2_Volatility',
                                           'Ticker1_ZScore', 'Ticker2_ZScore', 'Ratio_ZScore']].tail(20)
                    st.dataframe(display_df, use_container_width=True)
            
            # Paper Trading Section
            st.markdown("---")
            st.markdown("## 💼 Paper Trading Simulator")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Virtual Capital", f"₹{st.session_state.paper_capital:,.2f}")
            
            with col2:
                if st.button("Execute Recommended Trade", type="primary"):
                    if final_signal != "HOLD":
                        current_price = targets_stops['entry']
                        
                        position_value = st.session_state.paper_capital * 0.1
                        quantity = max(1, int(position_value / current_price))
                        
                        strategy_details = {
                            'zscore': avg_zscore,
                            'volatility': latest_data['Volatility'].iloc[-1],
                            'rsi': latest_data['RSI'].iloc[-1],
                        }
                        
                        trade = {
                            'timestamp': datetime.now(IST),
                            'ticker': st.session_state.ticker1_name,
                            'action': final_signal,
                            'price': current_price,
                            'quantity': quantity,
                            'value': current_price * quantity,
                            'confidence': avg_confidence,
                            'status': 'OPEN',
                            'strategy': strategy_details,
                            'targets': targets_stops,
                            'entry_time': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')
                        }
                        
                        st.session_state.paper_trades.append(trade)
                        st.session_state.live_monitoring = True
                        st.success(f"✅ {final_signal} order placed: {quantity} shares @ ₹{current_price:.2f}")
                    else:
                        st.info("Current recommendation is HOLD - no trade executed")
            
            with col3:
                total_pnl = 0
                for trade in st.session_state.paper_trades:
                    if trade['status'] == 'CLOSED' and 'exit_price' in trade:
                        if trade['action'] == 'BUY':
                            pnl = (trade['exit_price'] - trade['price']) * trade['quantity']
                        else:
                            pnl = (trade['price'] - trade['exit_price']) * trade['quantity']
                        total_pnl += pnl
                
                pnl_color = "green" if total_pnl >= 0 else "red"
                st.markdown(f"**Total P&L:** <span style='color:{pnl_color}'>₹{total_pnl:,.2f}</span>", unsafe_allow_html=True)
            
            # Auto-refresh positions
            auto_refresh_positions(analyzer, st.session_state.ticker1_symbol)
            
            # Position management
            if st.session_state.paper_trades:
                open_trades = [t for t in st.session_state.paper_trades if t['status'] == 'OPEN']
                closed_trades = [t for t in st.session_state.paper_trades if t['status'] == 'CLOSED']
                
                if open_trades:
                    st.markdown("#### 📋 Open Positions")
                    
                    for idx, trade in enumerate(open_trades):
                        with st.expander(f"Position #{idx+1} - {trade['action']} {trade['ticker']}"):
                            if 'targets' in trade:
                                targets = trade['targets']
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Entry:** ₹{trade['price']:.2f}")
                                    st.write(f"**Stop Loss:** ₹{targets['stop_loss']:.2f}")
                                    st.write(f"**Target 1:** ₹{targets['target1']:.2f}")
                                with col2:
                                    st.write(f"**Target 2:** ₹{targets['target2']:.2f}")
                                    st.write(f"**Target 3:** ₹{targets['target3']:.2f}")
                                    st.write(f"**Quantity:** {trade['quantity']}")
                                
                                if st.button(f"Close Position #{idx+1}", key=f"close_{idx}"):
                                    df_current = analyzer.fetch_data_with_retry(st.session_state.ticker1_symbol, '1d', '5m')
                                    if df_current is not None:
                                        df_current = analyzer.calculate_indicators(df_current)
                                        current_price = df_current['Close'].iloc[-1]
                                        
                                        if trade['action'] == 'BUY':
                                            pnl = (current_price - trade['price']) * trade['quantity']
                                        else:
                                            pnl = (trade['price'] - current_price) * trade['quantity']
                                        
                                        trade_idx = len(st.session_state.paper_trades) - len(open_trades) + idx
                                        st.session_state.paper_trades[trade_idx]['status'] = 'CLOSED'
                                        st.session_state.paper_trades[trade_idx]['exit_price'] = current_price
                                        st.session_state.paper_trades[trade_idx]['exit_time'] = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')
                                        st.session_state.paper_capital += pnl
                                        
                                        if len(open_trades) == 1:
                                            st.session_state.live_monitoring = False
                                        
                                        st.rerun()
                
                if closed_trades:
                    st.markdown("#### 📋 Closed Positions")
                    closed_data = []
                    
                    for trade in closed_trades:
                        if trade['quantity'] > 0:
                            if trade['action'] == 'BUY':
                                pnl = (trade['exit_price'] - trade['price']) * trade['quantity']
                            else:
                                pnl = (trade['price'] - trade['exit_price']) * trade['quantity']
                            
                            pnl_pct = (pnl / trade['value']) * 100
                            
                            closed_data.append({
                                'Ticker': trade['ticker'],
                                'Action': trade['action'],
                                'Entry': f"₹{trade['price']:.2f}",
                                'Exit': f"₹{trade['exit_price']:.2f}",
                                'Qty': trade['quantity'],
                                'Entry Time': trade['entry_time'],
                                'Exit Time': trade['exit_time'],
                                'P&L': f"₹{pnl:,.2f}",
                                'P&L %': f"{pnl_pct:+.2f}%"
                            })
                    
                    if closed_data:
                        st.dataframe(pd.DataFrame(closed_data), use_container_width=True)
            
            # Backtesting
            st.markdown("---")
            st.markdown("## 🔬 Strategy Backtesting")
            
            if st.button("Run Backtest", type="secondary"):
                with st.spinner("Running backtest..."):
                    # Use longer timeframe data for backtest
                    daily_keys = [k for k in results.keys() if k.startswith('1d_')]
                    # Find longest period
                    longest_key = max(daily_keys, key=lambda x: ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y'].index(x.split('_')[1]) if x.split('_')[1] in ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y'] else 0)
                    
                    df_bt = results[longest_key]['ticker1']['data'].copy()
                    
                    if len(df_bt) >= 100:
                        trades_bt = []
                        position = None
                        
                        for i in range(50, len(df_bt)):
                            current = df_bt.iloc[i]
                            
                            if position is None and current['RSI'] < 30 and current['Close'] > current['EMA_20']:
                                position = {
                                    'entry_date': current.name.strftime('%Y-%m-%d %H:%M IST'),
                                    'entry_price': current['Close'],
                                    'stop_loss': current['Close'] - (2 * current['ATR']),
                                    'target': current['Close'] + (2 * current['ATR']),
                                }
                            
                            elif position and (current['Close'] >= position['target'] or current['Close'] <= position['stop_loss'] or current['RSI'] > 70):
                                pnl = current['Close'] - position['entry_price']
                                pnl_pct = (pnl / position['entry_price']) * 100
                                
                                exit_reason = "Target" if current['Close'] >= position['target'] else ("Stop Loss" if current['Close'] <= position['stop_loss'] else "RSI Exit")
                                
                                trades_bt.append({
                                    'Entry Date': position['entry_date'],
                                    'Entry Price': f"₹{position['entry_price']:.2f}",
                                    'Stop Loss': f"₹{position['stop_loss']:.2f}",
                                    'Target': f"₹{position['target']:.2f}",
                                    'Exit Date': current.name.strftime('%Y-%m-%d %H:%M IST'),
                                    'Exit Price': f"₹{current['Close']:.2f}",
                                    'Exit Reason': exit_reason,
                                    'P&L': f"₹{pnl:.2f}",
                                    'P&L %': f"{pnl_pct:+.2f}%",
                                    'Result': '✅ WIN' if pnl > 0 else '❌ LOSS'
                                })
                                position = None
                        
                        if trades_bt:
                            st.markdown(f"### Backtest Results ({longest_key.split('_')[1]} period)")
                            
                            bt_df = pd.DataFrame(trades_bt)
                            
                            wins = len(bt_df[bt_df['Result'] == '✅ WIN'])
                            total = len(bt_df)
                            win_rate = (wins / total * 100) if total > 0 else 0
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Total Trades", total)
                            col2.metric("Wins", wins)
                            col3.metric("Win Rate", f"{win_rate:.1f}%")
                            
                
if 'paper_trades' not in st.session_state:
    st.session_state.paper_trades = []
if 'paper_capital' not in st.session_state:
    st.session_state.paper_capital = 100000.0
if 'live_monitoring' not in st.session_state:
    st.session_state.live_monitoring = False
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

TICKER_MAP = {
    'NIFTY 50': '^NSEI',
    'Bank NIFTY': '^NSEBANK',
    'SENSEX': '^BSESN',
    'BTC-USD': 'BTC-USD',
    'ETH-USD': 'ETH-USD',
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'USD/INR': 'INR=X',
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X',
}

TIMEFRAME_PERIODS = {
    '1m': ['1d', '5d'],
    '5m': ['1d', '5d', '1mo'],
    '15m': ['1d', '5d', '1mo'],
    '30m': ['1d', '5d', '1mo'],
    '1h': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
    '1d': ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y'],
    '1wk': ['1y', '2y', '5y', '10y'],
    '1mo': ['1y', '2y', '5y', '10y', '20y']
}

IST = pytz.timezone('Asia/Kolkata')

class TechnicalIndicators:
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False, min_periods=1).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        ema_fast = TechnicalIndicators.calculate_ema(data, fast)
        ema_slow = TechnicalIndicators.calculate_ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return {'macd': macd_line, 'signal': signal_line, 'histogram': histogram}
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> Dict:
        sma = TechnicalIndicators.calculate_sma(data, period)
        std = data.rolling(window=period, min_periods=1).std()
        return {'upper': sma + (std * std_dev), 'middle': sma, 'lower': sma - (std * std_dev)}
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        return atr.fillna(method='bfill')
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)
        
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
        
        atr = TechnicalIndicators.calculate_atr(high, low, close, period)
        atr = atr.replace(0, 1)
        
        plus_di = 100 * (plus_dm.rolling(window=period, min_periods=1).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period, min_periods=1).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period, min_periods=1).mean()
        return adx.fillna(0)
    
    @staticmethod
    def calculate_zscore(data: pd.Series, window: int = 20) -> pd.Series:
        rolling_mean = data.rolling(window=window, min_periods=1).mean()
        rolling_std = data.rolling(window=window, min_periods=1).std()
        rolling_std = rolling_std.replace(0, 1)
        zscore = (data - rolling_mean) / rolling_std
        return zscore.fillna(0)

class TradingAnalyzer:
    def __init__(self):
        self.data_cache = {}
        
    def fetch_data_with_retry(self, ticker: str, period: str, interval: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        cache_key = f"{ticker}_{period}_{interval}"
        
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        for attempt in range(max_retries):
            try:
                time.sleep(np.random.uniform(1.5, 2.5))
                data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
                
                if data.empty:
                    return None
                
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                if data.index.tz is None:
                    data.index = data.index.tz_localize('UTC').tz_convert(IST)
                else:
                    data.index = data.index.tz_convert(IST)
                
                self.data_cache[cache_key] = data
                return data
                
            except Exception as e:
                if attempt == max_retries - 1:
                    st.warning(f"Failed to fetch {ticker} ({interval}/{period}): {str(e)}")
                    return None
                time.sleep(2)
        
        return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        
        df = df.copy()
        df['Returns'] = df['Close'].pct_change().fillna(0)
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
        
        for period in [9, 20, 50, 100, 200]:
            df[f'EMA_{period}'] = TechnicalIndicators.calculate_ema(df['Close'], period)
            df[f'SMA_{period}'] = TechnicalIndicators.calculate_sma(df['Close'], period)
        
        df['RSI'] = TechnicalIndicators.calculate_rsi(df['Close'], 14)
        df['RSI_Oversold'] = df['RSI'] < 30
        df['RSI_Overbought'] = df['RSI'] > 70
        
        macd_data = TechnicalIndicators.calculate_macd(df['Close'])
        df['MACD'] = macd_data['macd']
        df['MACD_Signal'] = macd_data['signal']
        df['MACD_Histogram'] = macd_data['histogram']
        
        bb_data = TechnicalIndicators.calculate_bollinger_bands(df['Close'])
        df['BB_Upper'] = bb_data['upper']
        df['BB_Middle'] = bb_data['middle']
        df['BB_Lower'] = bb_data['lower']
        
        df['ATR'] = TechnicalIndicators.calculate_atr(df['High'], df['Low'], df['Close'])
        df['Volatility'] = df['Returns'].rolling(window=20, min_periods=1).std() * np.sqrt(252) * 100
        df['Volatility'] = df['Volatility'].fillna(0)
        df['ADX'] = TechnicalIndicators.calculate_adx(df['High'], df['Low'], df['Close'])
        
        df['Price_ZScore'] = TechnicalIndicators.calculate_zscore(df['Close'], 20)
        df['Returns_ZScore'] = TechnicalIndicators.calculate_zscore(df['Returns'].fillna(0), 20)
        
        return df
    
    def detect_rsi_divergence(self, df: pd.DataFrame, lookback: int = 30) -> Dict:
        """Detect RSI divergences"""
        if df is None or df.empty or len(df) < lookback:
            return {'type': None, 'strength': 0, 'description': 'Insufficient data'}
        
        recent = df.tail(lookback)
        
        # Find price peaks and troughs
        price_peaks_idx = argrelextrema(recent['Close'].values, np.greater, order=3)[0]
        price_troughs_idx = argrelextrema(recent['Close'].values, np.less, order=3)[0]
        
        # Find RSI peaks and troughs
        rsi_peaks_idx = argrelextrema(recent['RSI'].values, np.greater, order=3)[0]
        rsi_troughs_idx = argrelextrema(recent['RSI'].values, np.less, order=3)[0]
        
        # Bullish divergence: Price making lower lows, RSI making higher lows
        if len(price_troughs_idx) >= 2 and len(rsi_troughs_idx) >= 2:
            last_two_price_troughs = price_troughs_idx[-2:]
            last_two_rsi_troughs = rsi_troughs_idx[-2:]
            
            if len(last_two_price_troughs) == 2 and len(last_two_rsi_troughs) == 2:
                price1, price2 = recent['Close'].iloc[last_two_price_troughs]
                rsi1, rsi2 = recent['RSI'].iloc[last_two_rsi_troughs]
                
                if price2 < price1 and rsi2 > rsi1:
                    strength = min(100, abs((price2 - price1) / price1) * 100 + abs(rsi2 - rsi1))
                    return {
                        'type': 'Bullish',
                        'strength': strength,
                        'description': f'Price made lower low ({price2:.2f} < {price1:.2f}), but RSI made higher low ({rsi2:.1f} > {rsi1:.1f}). Potential upward reversal.'
                    }
        
        # Bearish divergence: Price making higher highs, RSI making lower highs
        if len(price_peaks_idx) >= 2 and len(rsi_peaks_idx) >= 2:
            last_two_price_peaks = price_peaks_idx[-2:]
            last_two_rsi_peaks = rsi_peaks_idx[-2:]
            
            if len(last_two_price_peaks) == 2 and len(last_two_rsi_peaks) == 2:
                price1, price2 = recent['Close'].iloc[last_two_price_peaks]
                rsi1, rsi2 = recent['RSI'].iloc[last_two_rsi_peaks]
                
                if price2 > price1 and rsi2 < rsi1:
                    strength = min(100, abs((price2 - price1) / price1) * 100 + abs(rsi2 - rsi1))
                    return {
                        'type': 'Bearish',
                        'strength': strength,
                        'description': f'Price made higher high ({price2:.2f} > {price1:.2f}), but RSI made lower high ({rsi2:.1f} < {rsi1:.1f}). Potential downward reversal.'
                    }
        
        return {'type': None, 'strength': 0, 'description': 'No significant RSI divergence detected'}
    
    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict:
        if df is None or df.empty or len(df) < 50:
            return {}
        
        recent_data = df.tail(100)
        swing_high = recent_data['High'].max()
        swing_low = recent_data['Low'].min()
        diff = swing_high - swing_low
        
        levels = {
            '0.0': swing_high,
            '0.236': swing_high - 0.236 * diff,
            '0.382': swing_high - 0.382 * diff,
            '0.5': swing_high - 0.5 * diff,
            '0.618': swing_high - 0.618 * diff,
            '0.786': swing_high - 0.786 * diff,
            '1.0': swing_low,
        }
        
        current_price = df['Close'].iloc[-1]
        closest_level = min(levels.items(), key=lambda x: abs(x[1] - current_price))
        
        return {
            'levels': levels,
            'swing_high': swing_high,
            'swing_low': swing_low,
            'closest_level': closest_level[0],
            'closest_price': closest_level[1],
            'current_price': current_price
        }
    
    def calculate_targets_stops(self, df: pd.DataFrame, action: str) -> Dict:
        """Calculate entry, target, and stop-loss levels"""
        current_price = df['Close'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        
        if action == 'BUY':
            stop_loss = current_price - (2 * atr)
            target1 = current_price + (2 * atr)
            target2 = current_price + (3 * atr)
            target3 = current_price + (4 * atr)
        else:  # SELL
            stop_loss = current_price + (2 * atr)
            target1 = current_price - (2 * atr)
            target2 = current_price - (3 * atr)
            target3 = current_price - (4 * atr)
        
        return {
            'entry': current_price,
            'stop_loss': stop_loss,
            'target1': target1,
            'target2': target2,
            'target3': target3,
            'atr': atr,
            'risk': abs(current_price - stop_loss),
            'reward1': abs(target1 - current_price),
            'rr_ratio1': abs(target1 - current_price) / abs(current_price - stop_loss)
        }
    
    def generate_signals(self, df: pd.DataFrame) -> Dict:
        if df is None or df.empty:
            return {'signal': 'HOLD', 'confidence': 0, 'reasons': []}
        
        signals = []
        reasons = []
        
        current_price = df['Close'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        current_adx = df['ADX'].iloc[-1]
        price_zscore = df['Price_ZScore'].iloc[-1]
        
        if price_zscore < -2:
            signals.append(1)
            reasons.append(f"✓ Price Z-Score {price_zscore:.2f} (oversold)")
        elif price_zscore > 2:
            signals.append(-1)
            reasons.append(f"✗ Price Z-Score {price_zscore:.2f} (overbought)")
        
        if current_price > df['EMA_20'].iloc[-1] > df['EMA_50'].iloc[-1]:
            signals.append(1)
            reasons.append("✓ Uptrend (Price > EMA20 > EMA50)")
        elif current_price < df['EMA_20'].iloc[-1] < df['EMA_50'].iloc[-1]:
            signals.append(-1)
            reasons.append("✗ Downtrend (Price < EMA20 < EMA50)")
        
        if current_rsi < 30:
            signals.append(1)
            reasons.append(f"✓ RSI oversold ({current_rsi:.1f})")
        elif current_rsi > 70:
            signals.append(-1)
            reasons.append(f"✗ RSI overbought ({current_rsi:.1f})")
        
        if current_adx > 25:
            if df['EMA_20'].iloc[-1] > df['EMA_50'].iloc[-1]:
                signals.append(1)
                reasons.append(f"✓ Strong uptrend (ADX: {current_adx:.1f})")
            else:
                signals.append(-1)
                reasons.append(f"✗ Strong downtrend (ADX: {current_adx:.1f})")
        
        if len(signals) == 0:
            return {'signal': 'HOLD', 'confidence': 0, 'reasons': ['No clear signals']}
        
        avg_signal = np.mean(signals)
        confidence = abs(avg_signal) * 100
        
        if avg_signal > 0.3:
            signal = 'BUY'
        elif avg_signal < -0.3:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'confidence': min(confidence, 99),
            'reasons': reasons[:8],
            'price_zscore': price_zscore
        }

def create_price_rsi_chart(df: pd.DataFrame, title: str, divergence_info: Dict) -> go.Figure:
    """Create price and RSI chart with divergence highlighting"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3],
                        subplot_titles=(f'{title} - Price', 'RSI'))
    
    # Price candlestick
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                  low=df['Low'], close=df['Close'], name='Price'),
                  row=1, col=1)
    
    # EMAs
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='EMA20', line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], name='EMA50', line=dict(color='orange', width=1)), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True, title=f"{title}<br><sub>{divergence_info['description']}</sub>")
    
    return fig

def create_ratio_chart(ratio_df: pd.DataFrame, ticker1_name: str, ticker2_name: str) -> go.Figure:
    """Create ratio analysis chart"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.7, 0.3],
                        subplot_titles=(f'{ticker1_name}/{ticker2_name} Ratio', 'Ratio RSI'))
    
    fig.add_trace(go.Scatter(x=ratio_df['DateTime'], y=ratio_df['Ratio'], name='Ratio',
                             line=dict(color='blue', width=2)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=ratio_df['DateTime'], y=ratio_df['Ratio_RSI'], name='Ratio RSI',
                             line=dict(color='purple', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(height=500, showlegend=True)
    return fig

def auto_refresh_positions(analyzer: TradingAnalyzer, ticker_symbol: str):
    """Auto-refresh positions every 2 seconds"""
    if st.session_state.live_monitoring and st.session_state.paper_trades:
        current_time = time.time()
        if current_time - st.session_state.last_refresh >= 2:
            st.session_state.last_refresh = current_time
            
            # Fetch latest data
            df_live = analyzer.fetch_data_with_retry(ticker_symbol, '1d', '5m')
            if df_live is not None and not df_live.empty:
                df_live = analyzer.calculate_indicators(df_live)
                current_price = df_live['Close'].iloc[-1]
                
                # Update open positions
                open_trades = [t for t in st.session_state.paper_trades if t['status'] == 'OPEN']
                
                if open_trades:
                    st.markdown("### 🔄 Live Position Updates (Auto-refreshing every 2s)")
                    
                    position_data = []
                    for idx, trade in enumerate(open_trades):
                        if trade['quantity'] > 0:
                            if trade['action'] == 'BUY':
                                unrealized_pnl = (current_price - trade['price']) * trade['quantity']
                                unrealized_pnl_pct = ((current_price - trade['price']) / trade['price']) * 100
                            else:
                                unrealized_pnl = (trade['price'] - current_price) * trade['quantity']
                                unrealized_pnl_pct = ((trade['price'] - current_price) / trade['price']) * 100
                            
                            # Calculate current parameters
                            current_rsi = df_live['RSI'].iloc[-1]
                            current_vol = df_live['Volatility'].iloc[-1]
                            current_zscore = df_live['Price_ZScore'].iloc[-1]
                            
                            position_data.append({
                                'Pos': idx + 1,
                                'Action': trade['action'],
                                'Entry': f"₹{trade['price']:.2f}",
                                'Current': f"₹{current_price:.2f}",
                                'Qty': trade['quantity'],
                                'P&L': f"₹{unrealized_pnl:,.2f}",
                                'P&L %': f"{unrealized_pnl_pct:+.2f}%",
                                'Entry RSI': f"{trade['strategy']['rsi']:.1f}",
                                'Now RSI': f"{current_rsi:.1f}",
                                'Entry Vol': f"{trade['strategy']['volatility']:.1f}%",
                                'Now Vol': f"{current_vol:.1f}%",
                                'Entry Z': f"{trade['strategy']['zscore']:.2f}",
                                'Now Z': f"{current_zscore:.2f}"
                            })
                    
                    if position_data:
                        position_df = pd.DataFrame(position_data)
                        st.dataframe(position_df, use_container_width=True)
                        
                        # Force refresh
                        time.sleep(2)
                        st.rerun()

def main():
    st.markdown('<h1 class="main-header">📈 Advanced Algorithmic Trading System</h1>', unsafe_allow_html=True)
    
    st.sidebar.title("⚙️ Configuration")
    
    ticker1_type = st.sidebar.selectbox("Ticker 1 Type", ["Preset", "Custom"])
    
    if ticker1_type == "Preset":
        ticker1_name = st.sidebar.selectbox("Select Ticker 1", list(TICKER_MAP.keys()))
        ticker1 = TICKER_MAP[ticker1_name]
    else:
        ticker1 = st.sidebar.text_input("Enter Ticker 1 Symbol", "RELIANCE.NS")
        ticker1_name = ticker1
    
    enable_ratio = st.sidebar.checkbox("Enable Ratio Analysis (Ticker 2)")
    
    ticker2 = None
    ticker2_name = None
    if enable_ratio:
        ticker2_type = st.sidebar.selectbox("Ticker 2 Type", ["Preset", "Custom"])
        if ticker2_type == "Preset":
            ticker2_name = st.sidebar.selectbox("Select Ticker 2", list(TICKER_MAP.keys()))
            ticker2 = TICKER_MAP[ticker2_name]
        else:
            ticker2 = st.sidebar.text_input("Enter Ticker 2 Symbol", "TCS.NS")
            ticker2_name = ticker2
    
    st.sidebar.markdown("---")
    analyze_button = st.sidebar.button("🚀 Start Complete Analysis", type="primary", use_container_width=True)
    
    if analyze_button:
        analyzer = TradingAnalyzer()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = {}
        total_analyses = sum(len(periods) for periods in TIMEFRAME_PERIODS.values())
        if enable_ratio and ticker2:
            total_analyses *= 2
        
        current_analysis = 0
        
        st.markdown("### 🔄 Multi-Timeframe Analysis in Progress...")
        
        for interval, periods in TIMEFRAME_PERIODS.items():
            for period in periods:
                current_analysis += 1
                progress = current_analysis / total_analyses
                progress_bar.progress(progress)
                status_text.text(f"Analyzing {ticker1_name} - {interval}/{period} ({current_analysis}/{total_analyses})")
                
                df1 = analyzer.fetch_data_with_retry(ticker1, period, interval)
                
                if df1 is not None and not df1.empty:
                    df1 = analyzer.calculate_indicators(df1)
                    rsi_div = analyzer.detect_rsi_divergence(df1)
                    fib_levels = analyzer.calculate_fibonacci_levels(df1)
                    signals = analyzer.generate_signals(df1)
                    
                    all_results[f"{interval}_{period}"] = {
                        'ticker1': {
                            'data': df1,
                            'rsi_divergence': rsi_div,
                            'fib_levels': fib_levels,
                            'signals': signals
                        }
                    }
                
                if enable_ratio and ticker2:
                    current_analysis += 1
                    progress = current_analysis / total_analyses
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing {ticker2_name} - {interval}/{period} ({current
