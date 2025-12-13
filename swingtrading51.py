import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import pytz
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.signal import argrelextrema

# Page configuration
st.set_page_config(
    page_title="Multi-Strategy Live Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .strategy-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .profit {
        color: #00cc00;
        font-weight: bold;
    }
    .loss {
        color: #ff0000;
        font-weight: bold;
    }
    .status-running {
        color: #00cc00;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .status-stopped {
        color: #ff6b6b;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trading_active' not in st.session_state:
    st.session_state.trading_active = False
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'current_position' not in st.session_state:
    st.session_state.current_position = None     # Check signal-based exit
    if sl_config['type'] == 'Signal Based' or target_config['type'] == 'Signal Based':
        bullish_signal, bearish_signal, _ = self.strategy.generate_signal(data)
            
        if position['type'] == 'LONG' and bearish_signal:
            return True, 'Bearish Signal Exit'
        elif position['type'] == 'SHORT' and bullish_signal:
            return True, 'Bullish Signal Exit'
        
        return False, ''
    
    def analyze_trade_performance(self, trade: Dict) -> str:
        """AI-powered trade analysis"""
        pnl = trade['pnl']
        pnl_pct = trade['pnl_percent']
        duration = trade['duration']
        exit_reason = trade['exit_reason']
        strategy_type = trade.get('strategy', 'Unknown')
        
        analysis = []
        
        if pnl > 0:
            analysis.append(f"✅ **Profitable Trade**: Gained {pnl_pct:.2f}% ({pnl:.2f} points)")
            analysis.append(f"📊 **Strategy**: {strategy_type}")
            
            if exit_reason == 'Target Achieved':
                analysis.append("🎯 **Good Exit**: Target was hit as planned")
            elif 'Signal' in exit_reason:
                analysis.append("📊 **Systematic Exit**: Exited on strategy signal")
            
            if pnl_pct > 1.5:
                analysis.append("💪 **Strong Performance**: Excellent profit capture")
        else:
            analysis.append(f"❌ **Loss Trade**: Lost {abs(pnl_pct):.2f}% ({abs(pnl):.2f} points)")
            analysis.append(f"📊 **Strategy**: {strategy_type}")
            
            if exit_reason == 'Stop Loss Hit':
                analysis.append("🛡️ **Risk Managed**: SL protected from larger loss")
            
            if abs(pnl_pct) > 2:
                analysis.append("⚠️ **Large Loss**: Review strategy parameters")
        
        if duration < 300:
            analysis.append("⚡ **Quick Trade**: Very short duration")
        elif duration > 3600:
            analysis.append("⏰ **Long Hold**: Extended position")
        
        analysis.append("\n**Recommendations:**")
        if pnl > 0:
            analysis.append("• Strategy is working well - maintain discipline")
            analysis.append("• Consider position sizing optimization")
        else:
            analysis.append("• Review entry conditions for this strategy")
            analysis.append("• Ensure proper risk-reward ratio")
            analysis.append("• Consider adjusting strategy parameters")
        
        return "\n".join(analysis)
    
    def get_market_status(self, position: Dict, current_price: float, 
                         signal_data: Dict) -> str:
        """Get current market status"""
        if not position:
            return "No active position"
        
        entry_price = position['entry_price']
        position_type = position['type']
        pnl = current_price - entry_price if position_type == 'LONG' else entry_price - current_price
        pnl_pct = (pnl / entry_price) * 100
        
        status_parts = []
        
        if position_type == 'LONG':
            status_parts.append("📈 **LONG Position Active**")
        else:
            status_parts.append("📉 **SHORT Position Active**")
        
        if pnl > 0:
            status_parts.append(f"✅ In Profit: +{pnl:.2f} pts (+{pnl_pct:.2f}%)")
            if pnl_pct > 0.5:
                status_parts.append("🚀 Strong momentum in favor!")
            else:
                status_parts.append("📊 Moving gradually in favor")
        elif pnl < 0:
            status_parts.append(f"⚠️ In Loss: {pnl:.2f} pts ({pnl_pct:.2f}%)")
            status_parts.append("🔍 Monitoring for reversal or SL hit")
        else:
            status_parts.append("➡️ At entry price")
        
        if position['sl_price']:
            sl_distance = abs(current_price - position['sl_price'])
            status_parts.append(f"🛡️ SL Distance: {sl_distance:.2f} pts")
            if position.get('sl_updated', False):
                status_parts.append("📊 Trailing SL Active")
        
        if position['target_price']:
            target_distance = abs(position['target_price'] - current_price)
            status_parts.append(f"🎯 Target Distance: {target_distance:.2f} pts")
        
        # Add strategy-specific info
        if signal_data:
            status_parts.append(f"\n**Strategy Info**: {signal_data.get('type', 'N/A')}")
        
        return "\n".join(status_parts)


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.markdown('<h1 class="main-header">🚀 Multi-Strategy Live Trading System</h1>', 
            unsafe_allow_html=True)
st.markdown("### Advanced Algorithmic Trading with Multiple Strategy Support")

trading_system = TradingSystem()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Trading Configuration")
    
    # Strategy Selection
    st.markdown('<div class="strategy-header"><h3>📊 Select Strategy</h3></div>', 
                unsafe_allow_html=True)
    
    strategy_type = st.selectbox(
        "Trading Strategy",
        [
            "EMA/SMA Crossover",
            "Pairs Trading (Ratio)",
            "Z-Score Mean Reversion",
            "Support/Resistance + Price Action",
            "Breakout with Volume",
            "Fibonacci Retracement",
            "Multi-Indicator (RSI/MACD/BB)",
            "Market Exhaustion"
        ]
    )
    
    st.markdown("---")
    
    # Asset Selection
    st.subheader("📊 Asset Selection")
    
    if strategy_type == "Pairs Trading (Ratio)":
        st.info("Select two correlated assets for pairs trading")
        ticker1 = st.text_input("Ticker 1", "^NSEI")
        ticker2 = st.text_input("Ticker 2", "^NSEBANK")
    else:
        asset_type = st.selectbox(
            "Asset Type",
            ["Indian Indices", "Crypto", "Forex", "Commodities", "Custom Ticker"]
        )
        
        ticker_map = {
            "Indian Indices": {
                "NIFTY 50": "^NSEI",
                "BANK NIFTY": "^NSEBANK",
                "SENSEX": "^BSESN"
            },
            "Crypto": {
                "Bitcoin": "BTC-USD",
                "Ethereum": "ETH-USD"
            },
            "Forex": {
                "EUR/USD": "EURUSD=X",
                "GBP/USD": "GBPUSD=X",
                "USD/INR": "INR=X"
            },
            "Commodities": {
                "Gold": "GC=F",
                "Silver": "SI=F"
            }
        }
        
        if asset_type == "Custom Ticker":
            ticker = st.text_input("Enter Ticker Symbol", "RELIANCE.NS")
        else:
            asset_name = st.selectbox("Select Asset", list(ticker_map[asset_type].keys()))
            ticker = ticker_map[asset_type][asset_name]
    
    # Timeframe
    st.subheader("⏰ Timeframe")
    col1, col2 = st.columns(2)
    with col1:
        interval = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d"], index=0)
    with col2:
        period_options = {
            "1m": ["1d", "5d"],
            "5m": ["1d", "5d", "1mo"],
            "15m": ["1d", "5d", "1mo"],
            "30m": ["1d", "5d", "1mo"],
            "1h": ["1d", "5d", "1mo", "3mo"],
            "1d": ["1mo", "3mo", "6mo", "1y"]
        }
        period = st.selectbox("Period", period_options.get(interval, ["1d"]))
    
    st.markdown("---")
    
    # Strategy Parameters
    st.subheader("🎯 Strategy Parameters")
    strategy_params = {}
    
    if strategy_type == "EMA/SMA Crossover":
        col1, col2 = st.columns(2)
        with col1:
            fast_type = st.selectbox("Fast Indicator", ["EMA", "SMA"], key="fast")
            fast_period = st.number_input("Fast Period", 1, 200, 9, key="fast_p")
        with col2:
            slow_type = st.selectbox("Slow Indicator", ["EMA", "SMA"], key="slow")
            slow_period = st.number_input("Slow Period", 1, 200, 20, key="slow_p")
        
        strategy_params = {
            'fast_type': fast_type,
            'fast_period': fast_period,
            'slow_type': slow_type,
            'slow_period': slow_period
        }
    
    elif strategy_type == "Pairs Trading (Ratio)":
        window = st.number_input("Lookback Window", 10, 100, 20)
        z_threshold = st.slider("Z-Score Threshold", 1.0, 3.0, 2.0, 0.1)
        strategy_params = {
            'ticker1': ticker1,
            'ticker2': ticker2,
            'window': window,
            'z_threshold': z_threshold
        }
    
    elif strategy_type == "Z-Score Mean Reversion":
        window = st.number_input("Lookback Window", 10, 100, 20)
        entry_threshold = st.slider("Entry Z-Score", 1.0, 3.0, 2.0, 0.1)
        strategy_params = {
            'window': window,
            'entry_threshold': entry_threshold
        }
    
    elif strategy_type == "Support/Resistance + Price Action":
        window = st.number_input("S/R Window", 10, 50, 20)
        tolerance = st.slider("Price Tolerance %", 0.5, 5.0, 2.0, 0.5) / 100
        strategy_params = {
            'window': window,
            'tolerance': tolerance
        }
    
    elif strategy_type == "Breakout with Volume":
        window = st.number_input("Breakout Window", 10, 50, 20)
        strategy_params = {'window': window}
    
    elif strategy_type == "Fibonacci Retracement":
        window = st.number_input("Swing Window", 20, 100, 50)
        tolerance = st.slider("Level Tolerance %", 0.1, 2.0, 0.5, 0.1) / 100
        strategy_params = {
            'window': window,
            'tolerance': tolerance
        }
    
    elif strategy_type == "Multi-Indicator (RSI/MACD/BB)":
        st.info("Uses RSI, MACD, Bollinger Bands, and ADX")
        strategy_params = {}
    
    elif strategy_type == "Market Exhaustion":
        st.info("Detects extreme volume and price moves")
        strategy_params = {}
    
    st.markdown("---")
    
    # Risk Management
    st.subheader("🛡️ Risk Management")
    
    sl_type = st.selectbox(
        "Stop Loss Type",
        ["Custom Points", "Trail SL", "Signal Based"]
    )
    
    sl_value = None
    if sl_type in ["Custom Points", "Trail SL"]:
        sl_value = st.number_input("SL Points", 0.0, 1000.0, 10.0, 0.5)
    
    sl_config = {"type": sl_type, "value": sl_value}
    
    target_type = st.selectbox(
        "Target Type",
        ["Custom Points", "Trail Target", "Signal Based"]
    )
    
    target_value = None
    if target_type in ["Custom Points", "Trail Target"]:
        target_value = st.number_input("Target Points", 0.0, 1000.0, 20.0, 0.5)
    
    target_config = {"type": target_type, "value": target_value}
    
    position_size = st.number_input("Position Size (Qty)", 1, 1000, 1)

# Initialize Strategy
strategy_map = {
    "EMA/SMA Crossover": EMASMACrossoverStrategy,
    "Pairs Trading (Ratio)": PairTradingStrategy,
    "Z-Score Mean Reversion": ZScoreStrategy,
    "Support/Resistance + Price Action": SupportResistanceStrategy,
    "Breakout with Volume": BreakoutStrategy,
    "Fibonacci Retracement": FibonacciStrategy,
    "Multi-Indicator (RSI/MACD/BB)": RSIMomentumStrategy,
    "Market Exhaustion": MarketExhaustionStrategy
}

selected_strategy = strategy_map[strategy_type](strategy_params)
trading_system.set_strategy(selected_strategy)

# Main Tabs
tab1, tab2, tab3 = st.tabs(["📊 Live Trading", "📈 Trade History", "📋 Trade Log"])

with tab1:
    # Control Buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("▶️ Start Trading", type="primary", 
                    disabled=st.session_state.trading_active):
            st.session_state.trading_active = True
            ticker_display = ticker if strategy_type != "Pairs Trading (Ratio)" else f"{ticker1}/{ticker2}"
            st.session_state.trade_log.append({
                'time': datetime.now(trading_system.ist_tz),
                'message': f'Trading started: {strategy_type} on {ticker_display} ({interval})'
            })
            st.rerun()
    
    with col2:
        if st.button("⏸️ Stop Trading", type="secondary", 
                    disabled=not st.session_state.trading_active):
            st.session_state.trading_active = False
            if st.session_state.current_position:
                st.session_state.trade_log.append({
                    'time': datetime.now(trading_system.ist_tz),
                    'message': 'Trading stopped - Position still open'
                })
            st.rerun()
    
    with col3:
        if st.session_state.trading_active:
            st.markdown('<p class="status-running">🟢 Trading Active</p>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-stopped">🔴 Trading Stopped</p>', 
                       unsafe_allow_html=True)
    
    # Trading Info Header
    if st.session_state.trading_active or st.session_state.current_position:
        st.markdown("---")
        info_cols = st.columns(4)
        
        with info_cols[0]:
            if strategy_type == "Pairs Trading (Ratio)":
                st.metric("Assets", f"{ticker1}/{ticker2}")
            else:
                st.metric("Asset", ticker)
        with info_cols[1]:
            st.metric("Strategy", strategy_type.split()[0])
        with info_cols[2]:
            st.metric("Timeframe", interval)
        with info_cols[3]:
            st.metric("Period", period)
    
    # Live Trading Loop
    if st.session_state.trading_active:
        status_container = st.container()
        metrics_container = st.container()
        chart_container = st.container()
        update_container = st.container()
        
        # Fetch data
        if strategy_type == "Pairs Trading (Ratio)":
            data = trading_system.fetch_multi_ticker_data([ticker1, ticker2], interval, period)
        else:
            data = trading_system.fetch_data(ticker, interval, period)
        
        if data is None or len(data) < 20:
            st.error("⚠️ Insufficient data. Waiting...")
            time.sleep(2)
            st.rerun()
        
        # Generate signals
        bullish_signal, bearish_signal, signal_data = selected_strategy.generate_signal(data)
        
        current_price = data['Close'].iloc[-1] if 'Close' in data.columns else data.iloc[-1]['Close']
        current_time = data.index[-1]
        
        # Position Management
        if st.session_state.current_position is None:
            if bullish_signal:
                sl_price, target_price = trading_system.calculate_sl_target(
                    float(current_price), 'LONG', sl_config, target_config
                )
                
                st.session_state.current_position = {
                    'type': 'LONG',
                    'entry_price': float(current_price),
                    'entry_time': current_time,
                    'quantity': position_size,
                    'sl_price': sl_price,
                    'target_price': target_price,
                    'signal_data': signal_data,
                    'sl_updated': False,
                    'strategy': strategy_type
                }
                
                sl_text = f"{float(sl_price):.2f}" if sl_price is not None else "Signal Based"
                target_text = f"{float(target_price):.2f}" if target_price is not None else "Signal Based"
                
                st.session_state.trade_log.append({
                    'time': current_time,
                    'message': f'🟢 LONG Entry at {float(current_price):.2f} | SL: {sl_text} | Target: {target_text}'
                })
            
            elif bearish_signal:
                sl_price, target_price = trading_system.calculate_sl_target(
                    float(current_price), 'SHORT', sl_config, target_config
                )
                
                st.session_state.current_position = {
                    'type': 'SHORT',
                    'entry_price': float(current_price),
                    'entry_time': current_time,
                    'quantity': position_size,
                    'sl_price': sl_price,
                    'target_price': target_price,
                    'signal_data': signal_data,
                    'sl_updated': False,
                    'strategy': strategy_type
                }
                
                sl_text = f"{float(sl_price):.2f}" if sl_price is not None else "Signal Based"
                target_text = f"{float(target_price):.2f}" if target_price is not None else "Signal Based"
                
                st.session_state.trade_log.append({
                    'time': current_time,
                    'message': f'🔴 SHORT Entry at {float(current_price):.2f} | SL: {sl_text} | Target: {target_text}'
                })
        
        else:
            position = st.session_state.current_position
            
            # Update trailing SL
            if sl_config['type'] == 'Trail SL':
                new_sl = trading_system.update_trailing_sl(position, float(current_price), sl_config)
                if new_sl != position['sl_price']:
                    position['sl_price'] = new_sl
                    position['sl_updated'] = True
                    st.session_state.trade_log.append({
                        'time': current_time,
                        'message': f'📊 Trailing SL updated to {float(new_sl):.2f}'
                    })
            
            # Check exit
            should_exit, exit_reason = trading_system.check_exit_conditions(
                position, float(current_price), data, sl_config, target_config
            )
            
            if should_exit:
                entry_price = position['entry_price']
                pnl = (float(current_price) - entry_price) if position['type'] == 'LONG' else (entry_price - float(current_price))
                pnl = pnl * position['quantity']
                pnl_percent = (pnl / (entry_price * position['quantity'])) * 100
                duration = (current_time - position['entry_time']).total_seconds()
                
                trade_record = {
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'type': position['type'],
                    'entry_price': entry_price,
                    'exit_price': float(current_price),
                    'quantity': position['quantity'],
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'duration': duration,
                    'exit_reason': exit_reason,
                    'strategy': strategy_type
                }
                
                trade_record['analysis'] = trading_system.analyze_trade_performance(trade_record)
                
                st.session_state.trade_history.append(trade_record)
                st.session_state.current_position = None
                
                st.session_state.trade_log.append({
                    'time': current_time,
                    'message': f'🏁 Position Closed: {exit_reason} | P&L: {float(pnl):.2f} ({float(pnl_percent):.2f}%)'
                })
        
        # Display Status
        with status_container:
            st.markdown("### 📊 Current Market Status")
            
            if st.session_state.current_position:
                status_html = trading_system.get_market_status(
                    st.session_state.current_position,
                    float(current_price),
                    signal_data
                )
                st.markdown(status_html)
            else:
                st.info(f"⏳ Waiting for {strategy_type} signal...")
                st.markdown(f"**Current Price:** {float(current_price):.2f}")
                if signal_data:
                    st.json(signal_data)
        
        # Display Metrics
        with metrics_container:
            if st.session_state.current_position:
                pos = st.session_state.current_position
                m1, m2, m3, m4, m5 = st.columns(5)
                
                with m1:
                    st.metric("Entry Price", f"{float(pos['entry_price']):.2f}")
                with m2:
                    st.metric("Current Price", f"{float(current_price):.2f}")
                with m3:
                    pnl = (float(current_price) - pos['entry_price']) if pos['type'] == 'LONG' else (pos['entry_price'] - float(current_price))
                    pnl_total = pnl * pos['quantity']
                    st.metric("P&L Points", f"{float(pnl):.2f}", delta=f"{float(pnl_total):.2f}")
                with m4:
                    st.metric("Strategy", pos['strategy'].split()[0])
                with m5:
                    duration = (current_time - pos['entry_time']).total_seconds()
                    st.metric("Duration", f"{int(duration)}s")
        
        # Display Chart
        with chart_container:
            fig = go.Figure()
            
            # Candlestick
            if 'Open' in data.columns:
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'] if 'Close' in data.columns else data.iloc[:, 0],
                    mode='lines',
                    name='Price'
                ))
            
            # Add strategy-specific indicators to chart
            data_with_indicators = selected_strategy.calculate_indicators(data)
            
            if strategy_type == "EMA/SMA Crossover":
                fig.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['fast_ind'],
                    mode='lines',
                    name=f'{strategy_params["fast_type"]}{strategy_params["fast_period"]}',
                    line=dict(color='blue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['slow_ind'],
                    mode='lines',
                    name=f'{strategy_params["slow_type"]}{strategy_params["slow_period"]}',
                    line=dict(color='red', width=2)
                ))
            
            # Mark entry
            if st.session_state.current_position:
                pos = st.session_state.current_position
                fig.add_trace(go.Scatter(
                    x=[pos['entry_time']],
                    y=[pos['entry_price']],
                    mode='markers',
                    name='Entry',
                    marker=dict(
                        size=15,
                        color='green' if pos['type'] == 'LONG' else 'red',
                        symbol='triangle-up' if pos['type'] == 'LONG' else 'triangle-down'
                    )
                ))
            
            ticker_display = ticker if strategy_type != "Pairs Trading (Ratio)" else f"{ticker1}/{ticker2}"
            fig.update_layout(
                title=f'{ticker_display} - {strategy_type}',
                yaxis_title='Price',
                xaxis_title='Time (IST)',
                template='plotly_white',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True, 
                          key=f"live_chart_{current_time.timestamp()}")
        
        # Update timestamp
        with update_container:
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 0.5rem; border-radius: 0.3rem; text-align: center; margin-top: 1rem;'>
                <small>Last updated: {current_time.strftime('%Y-%m-%d %H:%M:%S IST')}</small>
            </div>
            """, unsafe_allow_html=True)
        
        time.sleep(2)
        if st.session_state.trading_active:
            st.rerun()
    
    elif st.session_state.current_position:
        st.warning("⚠️ Trading stopped but position still open")
        
        # Fetch latest data
        if strategy_type == "Pairs Trading (Ratio)":
            data = trading_system.fetch_multi_ticker_data([ticker1, ticker2], interval, period)
        else:
            data = trading_system.fetch_data(ticker, interval, period)
        
        if data is not None:
            current_price = data['Close'].iloc[-1] if 'Close' in data.columns else data.iloc[-1]['Close']
            
            pos = st.session_state.current_position
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                st.metric("Position Type", pos['type'])
            with m2:
                st.metric("Entry Price", f"{float(pos['entry_price']):.2f}")
            with m3:
                st.metric("Current Price", f"{float(current_price):.2f}")
            with m4:
                pnl = (float(current_price) - pos['entry_price']) if pos['type'] == 'LONG' else (pos['entry_price'] - float(current_price))
                pnl_total = pnl * pos['quantity']
                st.metric("P&L", f"{float(pnl_total):.2f}", delta=f"{float(pnl):.2f} pts")
            
            if st.button("🔴 Force Close Position"):
                entry_price = pos['entry_price']
                pnl = (float(current_price) - entry_price) if pos['type'] == 'LONG' else (entry_price - float(current_price))
                pnl = pnl * pos['quantity']
                pnl_percent = (pnl / (entry_price * pos['quantity'])) * 100
                
                trade_record = {
                    'entry_time': pos['entry_time'],
                    'exit_time': datetime.now(trading_system.ist_tz),
                    'type': pos['type'],
                    'entry_price': entry_price,
                    'exit_price': float(current_price),
                    'quantity': pos['quantity'],
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'duration': (datetime.now(trading_system.ist_tz) - pos['entry_time']).total_seconds(),
                    'exit_reason': 'Manual Close',
                    'strategy': pos['strategy']
                }
                
                trade_record['analysis'] = trading_system.analyze_trade_performance(trade_record)
                st.session_state.trade_history.append(trade_record)
                st.session_state.current_position = None
                
                st.success("Position closed!")
                st.rerun()

with tab2:
    st.header("📈 Trade History & Performance")
    
    if not st.session_state.trade_history:
        st.info("No trades executed yet")
    else:
        # Performance Summary
        total_trades = len(st.session_state.trade_history)
        winning_trades = sum(1 for t in st.session_state.trade_history if t['pnl'] > 0)
        losing_trades = total_trades - winning_trades
        total_pnl = sum(t['pnl'] for t in st.session_state.trade_history)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Winning", winning_trades, delta=f"{win_rate:.1f}%")
        with col3:
            st.metric("Losing", losing_trades)
        with col4:
            st.metric("Total P&L", f"{total_pnl:.2f}")
        with col5:
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
            st.metric("Avg P&L", f"{avg_pnl:.2f}")
        
        st.markdown("---")
        
        # Strategy Performance Breakdown
        st.subheader("📊 Performance by Strategy")
        strategy_performance = {}
        for trade in st.session_state.trade_history:
            strat = trade.get('strategy', 'Unknown')
            if strat not in strategy_performance:
                strategy_performance[strat] = {'trades': 0, 'wins': 0, 'pnl': 0}
            strategy_performance[strat]['trades'] += 1
            strategy_performance[strat]['pnl'] += trade['pnl']
            if trade['pnl'] > 0:
                strategy_performance[strat]['wins'] += 1
        
        perf_data = []
        for strat, stats in strategy_performance.items():
            win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
            perf_data.append({
                'Strategy': strat,
                'Trades': stats['trades'],
                'Wins': stats['wins'],
                'Win Rate': f"{win_rate:.1f}%",
                'Total P&L': f"{stats['pnl']:.2f}"
            })
        
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Trade Details Table
        st.subheader("📋 Detailed Trade Records")
        
        trade_df = pd.DataFrame([
            {
                'Entry Time': t['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'Exit Time': t['exit_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'Strategy': t.get('strategy', 'N/A'),
                'Type': t['type'],
                'Entry': f"{t['entry_price']:.2f}",
                'Exit': f"{t['exit_price']:.2f}",
                'Qty': t['quantity'],
                'P&L': f"{t['pnl']:.2f}",
                'P&L %': f"{t['pnl_percent']:.2f}%",
                'Duration (s)': int(t['duration']),
                'Exit Reason': t['exit_reason']
            }
            for t in st.session_state.trade_history
        ])
        
        st.dataframe(trade_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Individual Trade Analysis
        st.subheader("🔍 Trade-by-Trade Analysis")
        
        for idx, trade in enumerate(reversed(st.session_state.trade_history), 1):
            pnl_emoji = "📈" if trade['pnl'] > 0 else "📉"
            with st.expander(f"Trade #{total_trades - idx + 1} {pnl_emoji} {trade.get('strategy', 'N/A')} - {trade['type']} - P&L: {trade['pnl']:.2f} ({trade['pnl_percent']:.2f}%)"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**Trade Details:**")
                    st.markdown(f"- Strategy: {trade.get('strategy', 'N/A')}")
                    st.markdown(f"- Entry: {trade['entry_price']:.2f}")
                    st.markdown(f"- Exit: {trade['exit_price']:.2f}")
                    st.markdown(f"- Quantity: {trade['quantity']}")
                    st.markdown(f"- Duration: {int(trade['duration'])} seconds")
                    st.markdown(f"- Exit Reason: {trade['exit_reason']}")
                
                with col2:
                    st.markdown("**AI Analysis:**")
                    st.markdown(trade['analysis'])
        
        st.markdown("---")
        
        # Performance Charts
        st.subheader("📊 Performance Visualizations")
        
        # Cumulative P&L
        cumulative_pnl = []
        running_total = 0
        for trade in st.session_state.trade_history:
            running_total += trade['pnl']
            cumulative_pnl.append({
                'Trade': len(cumulative_pnl) + 1,
                'Cumulative P&L': running_total,
                'Time': trade['exit_time']
            })
        
        pnl_df = pd.DataFrame(cumulative_pnl)
        
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=pnl_df['Trade'],
            y=pnl_df['Cumulative P&L'],
            mode='lines+markers',
            name='Cumulative P&L',
            line=dict(color='green' if total_pnl >= 0 else 'red', width=3),
            marker=dict(size=8)
        ))
        
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray", 
                         annotation_text="Break Even")
        
        fig_pnl.update_layout(
            title='Cumulative Profit & Loss',
            xaxis_title='Trade Number',
            yaxis_title='Cumulative P&L',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig_pnl, use_container_width=True, key="cumulative_pnl_chart")
        
        # Win/Loss and P&L Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Winning', 'Losing'],
                values=[winning_trades, losing_trades],
                marker_colors=['#00cc00', '#ff6b6b']
            )])
            fig_pie.update_layout(title='Win/Loss Distribution', height=300)
            st.plotly_chart(fig_pie, use_container_width=True, key="win_loss_pie")
        
        with col2:
            pnl_values = [t['pnl'] for t in st.session_state.trade_history]
            fig_hist = go.Figure(data=[go.Histogram(
                x=pnl_values,
                nbinsx=20,
                marker_color='steelblue'
            )])
            fig_hist.update_layout(
                title='P&L Distribution',
                xaxis_title='P&L',
                yaxis_title='Frequency',
                height=300
            )
            st.plotly_chart(fig_hist, use_container_width=True, key="pnl_hist")

with tab3:
    st.header("📋 Real-Time Trade Log")
    
    if not st.session_state.trade_log:
        st.info("No trade activity logged yet")
    else:
        st.markdown("### Recent Activity")
        
        log_entries = []
        for log in reversed(st.session_state.trade_log[-100:]):
            log_entries.append(f"**{log['time'].strftime('%Y-%m-%d %H:%M:%S IST')}** - {log['message']}")
        
        log_html = """
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; 
                    height: 600px; overflow-y: scroll; border: 1px solid #dee2e6;'>
        """
        
        for entry in log_entries:
            log_html += f"<p style='margin: 0.5rem 0; padding: 0.5rem; background: white; border-radius: 0.3rem; border-left: 3px solid #1f77b4;'>{entry}</p>"
        
        log_html += "</div>"
        
        st.markdown(log_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🗑️ Clear Trade Log"):
                st.session_state.trade_log = []
                st.rerun()
        with col2:
            st.metric("Log Entries", len(st.session_state.trade_log))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>⚠️ Risk Warning:</strong> Trading involves substantial risk. This is for educational purposes only.</p>
    <p>Multiple strategies available: EMA/SMA Crossover, Pairs Trading, Z-Score, Support/Resistance, Breakout, Fibonacci, Multi-Indicator, Market Exhaustion</p>
    <p style='margin-top: 1rem; font-size: 0.9rem;'>Powered by yfinance | Real-time with 2-second rate limiting</p>
</div>
""", unsafe_allow_html=True)
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []

# ============================================================================
# STRATEGY CLASSES
# ============================================================================

class BaseStrategy:
    """Base class for all trading strategies"""
    
    def __init__(self, params: Dict):
        self.params = params
        self.ist_tz = pytz.timezone('Asia/Kolkata')
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy-specific indicators"""
        raise NotImplementedError
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        """
        Generate trading signals
        Returns: (bullish_signal, bearish_signal, signal_data)
        """
        raise NotImplementedError
    
    def get_signal_strength(self, data: pd.DataFrame) -> float:
        """Return signal strength from 0 to 1"""
        return 0.5


class EMASMACrossoverStrategy(BaseStrategy):
    """EMA/SMA Crossover Strategy"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        fast_period = self.params['fast_period']
        slow_period = self.params['slow_period']
        fast_type = self.params['fast_type']
        slow_type = self.params['slow_type']
        
        if fast_type == 'EMA':
            data['fast_ind'] = data['Close'].ewm(span=fast_period, adjust=False).mean()
        else:
            data['fast_ind'] = data['Close'].rolling(window=fast_period).mean()
        
        if slow_type == 'EMA':
            data['slow_ind'] = data['Close'].ewm(span=slow_period, adjust=False).mean()
        else:
            data['slow_ind'] = data['Close'].rolling(window=slow_period).mean()
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        data = self.calculate_indicators(data)
        
        if len(data) < 2:
            return False, False, {}
        
        fast = data['fast_ind']
        slow = data['slow_ind']
        
        # Bullish crossover
        bullish = (fast.iloc[-2] <= slow.iloc[-2]) and (fast.iloc[-1] > slow.iloc[-1])
        
        # Bearish crossover
        bearish = (fast.iloc[-2] >= slow.iloc[-2]) and (fast.iloc[-1] < slow.iloc[-1])
        
        signal_data = {
            'fast_value': fast.iloc[-1],
            'slow_value': slow.iloc[-1],
            'type': 'EMA/SMA Crossover'
        }
        
        return bullish, bearish, signal_data


class PairTradingStrategy(BaseStrategy):
    """Pairs Trading / Ratio Strategy"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # Assumes data has both tickers as columns
        ticker1 = self.params['ticker1']
        ticker2 = self.params['ticker2']
        window = self.params.get('window', 20)
        
        data['ratio'] = data[ticker1] / data[ticker2]
        data['ratio_ma'] = data['ratio'].rolling(window=window).mean()
        data['ratio_std'] = data['ratio'].rolling(window=window).std()
        data['z_score'] = (data['ratio'] - data['ratio_ma']) / data['ratio_std']
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        data = self.calculate_indicators(data)
        
        if len(data) < 2:
            return False, False, {}
        
        z_score = data['z_score'].iloc[-1]
        z_threshold = self.params.get('z_threshold', 2.0)
        
        # Long when ratio is significantly below mean
        bullish = z_score < -z_threshold
        
        # Short when ratio is significantly above mean
        bearish = z_score > z_threshold
        
        signal_data = {
            'z_score': z_score,
            'ratio': data['ratio'].iloc[-1],
            'type': 'Pair Trading'
        }
        
        return bullish, bearish, signal_data


class ZScoreStrategy(BaseStrategy):
    """Z-Score Mean Reversion Strategy"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        window = self.params.get('window', 20)
        
        data['price_ma'] = data['Close'].rolling(window=window).mean()
        data['price_std'] = data['Close'].rolling(window=window).std()
        data['z_score'] = (data['Close'] - data['price_ma']) / data['price_std']
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        data = self.calculate_indicators(data)
        
        if len(data) < 2:
            return False, False, {}
        
        z_score = data['z_score'].iloc[-1]
        entry_threshold = self.params.get('entry_threshold', 2.0)
        
        # Buy when price is significantly below mean
        bullish = z_score < -entry_threshold
        
        # Short when price is significantly above mean
        bearish = z_score > entry_threshold
        
        signal_data = {
            'z_score': z_score,
            'mean': data['price_ma'].iloc[-1],
            'type': 'Z-Score Mean Reversion'
        }
        
        return bullish, bearish, signal_data


class SupportResistanceStrategy(BaseStrategy):
    """Support/Resistance with Price Action"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        window = self.params.get('window', 20)
        
        # Find local maxima and minima
        data['resistance'] = data['High'].rolling(window=window, center=True).max()
        data['support'] = data['Low'].rolling(window=window, center=True).min()
        
        # Volume analysis
        data['volume_ma'] = data['Volume'].rolling(window=window).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_ma']
        
        return data
    
    def detect_candlestick_pattern(self, data: pd.DataFrame) -> str:
        """Detect basic candlestick patterns"""
        if len(data) < 3:
            return "none"
        
        last = data.iloc[-1]
        prev = data.iloc[-2]
        
        body = abs(last['Close'] - last['Open'])
        range_size = last['High'] - last['Low']
        
        # Hammer (bullish)
        if (last['Close'] > last['Open'] and 
            (last['High'] - last['Close']) < body * 0.3 and
            (last['Open'] - last['Low']) > body * 2):
            return "hammer"
        
        # Shooting Star (bearish)
        if (last['Open'] > last['Close'] and
            (last['Close'] - last['Low']) < body * 0.3 and
            (last['High'] - last['Open']) > body * 2):
            return "shooting_star"
        
        # Bullish Engulfing
        if (last['Close'] > last['Open'] and
            prev['Open'] > prev['Close'] and
            last['Open'] < prev['Close'] and
            last['Close'] > prev['Open']):
            return "bullish_engulfing"
        
        # Bearish Engulfing
        if (last['Open'] > last['Close'] and
            prev['Close'] > prev['Open'] and
            last['Open'] > prev['Close'] and
            last['Close'] < prev['Open']):
            return "bearish_engulfing"
        
        return "none"
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        data = self.calculate_indicators(data)
        
        if len(data) < 3:
            return False, False, {}
        
        current = data.iloc[-1]
        pattern = self.detect_candlestick_pattern(data)
        
        # Check if price is near support/resistance
        price = current['Close']
        support = current['support']
        resistance = current['resistance']
        
        tolerance = self.params.get('tolerance', 0.02)  # 2% tolerance
        high_volume = current['volume_ratio'] > 1.5
        
        # Bullish: near support with bullish pattern and high volume
        near_support = abs(price - support) / support < tolerance
        bullish_pattern = pattern in ['hammer', 'bullish_engulfing']
        bullish = near_support and bullish_pattern and high_volume
        
        # Bearish: near resistance with bearish pattern and high volume
        near_resistance = abs(price - resistance) / resistance < tolerance
        bearish_pattern = pattern in ['shooting_star', 'bearish_engulfing']
        bearish = near_resistance and bearish_pattern and high_volume
        
        signal_data = {
            'pattern': pattern,
            'support': support,
            'resistance': resistance,
            'volume_ratio': current['volume_ratio'],
            'type': 'Support/Resistance + Price Action'
        }
        
        return bullish, bearish, signal_data


class BreakoutStrategy(BaseStrategy):
    """Breakout Strategy with Retest"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        window = self.params.get('window', 20)
        
        data['high_band'] = data['High'].rolling(window=window).max()
        data['low_band'] = data['Low'].rolling(window=window).min()
        data['atr'] = self.calculate_atr(data, window)
        data['volume_ma'] = data['Volume'].rolling(window=window).mean()
        
        return data
    
    def calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift())
        low_close = abs(data['Low'] - data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(window=period).mean()
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        data = self.calculate_indicators(data)
        
        if len(data) < 3:
            return False, False, {}
        
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Breakout conditions
        bullish_breakout = (prev['Close'] <= prev['high_band'] and 
                           current['Close'] > current['high_band'])
        bearish_breakout = (prev['Close'] >= prev['low_band'] and 
                           current['Close'] < current['low_band'])
        
        # Volume confirmation
        volume_surge = current['Volume'] > current['volume_ma'] * 1.5
        
        # Strong ATR (volatility)
        strong_move = current['atr'] > data['atr'].rolling(20).mean().iloc[-1]
        
        bullish = bullish_breakout and volume_surge and strong_move
        bearish = bearish_breakout and volume_surge and strong_move
        
        signal_data = {
            'high_band': current['high_band'],
            'low_band': current['low_band'],
            'atr': current['atr'],
            'volume_ratio': current['Volume'] / current['volume_ma'],
            'type': 'Breakout with Volume'
        }
        
        return bullish, bearish, signal_data


class FibonacciStrategy(BaseStrategy):
    """Fibonacci Retracement Strategy"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        window = self.params.get('window', 50)
        
        # Find swing high and low
        data['swing_high'] = data['High'].rolling(window=window).max()
        data['swing_low'] = data['Low'].rolling(window=window).min()
        
        # Calculate Fibonacci levels
        diff = data['swing_high'] - data['swing_low']
        data['fib_236'] = data['swing_high'] - 0.236 * diff
        data['fib_382'] = data['swing_high'] - 0.382 * diff
        data['fib_500'] = data['swing_high'] - 0.500 * diff
        data['fib_618'] = data['swing_high'] - 0.618 * diff
        data['fib_786'] = data['swing_high'] - 0.786 * diff
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        data = self.calculate_indicators(data)
        
        if len(data) < 2:
            return False, False, {}
        
        current = data.iloc[-1]
        price = current['Close']
        
        tolerance = self.params.get('tolerance', 0.005)  # 0.5%
        
        # Check if price is near key Fibonacci levels
        fib_levels = [
            ('0.382', current['fib_382']),
            ('0.500', current['fib_500']),
            ('0.618', current['fib_618'])
        ]
        
        near_support = False
        near_resistance = False
        level_name = None
        
        for name, level in fib_levels:
            if abs(price - level) / level < tolerance:
                level_name = name
                # Determine if acting as support or resistance
                if price < current['swing_high'] * 0.7:  # In downtrend
                    near_support = True
                else:  # In uptrend
                    near_resistance = True
                break
        
        # RSI for confirmation
        rsi = self.calculate_rsi(data['Close'])
        
        bullish = near_support and rsi < 40
        bearish = near_resistance and rsi > 60
        
        signal_data = {
            'fib_level': level_name,
            'swing_high': current['swing_high'],
            'swing_low': current['swing_low'],
            'rsi': rsi,
            'type': 'Fibonacci Retracement'
        }
        
        return bullish, bearish, signal_data
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if len(rsi) > 0 else 50


class RSIMomentumStrategy(BaseStrategy):
    """RSI with Multiple Indicators"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # RSI
        data['rsi'] = self.calculate_rsi(data['Close'])
        
        # MACD
        macd_data = self.calculate_macd(data['Close'])
        data['macd'] = macd_data['macd']
        data['signal'] = macd_data['signal']
        data['histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self.calculate_bollinger_bands(data['Close'])
        data['bb_upper'] = bb_data['upper']
        data['bb_middle'] = bb_data['middle']
        data['bb_lower'] = bb_data['lower']
        
        # ADX (Trend Strength)
        data['adx'] = self.calculate_adx(data)
        
        return data
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9) -> Dict:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {'macd': macd, 'signal': signal_line, 'histogram': histogram}
    
    def calculate_bollinger_bands(self, prices: pd.Series, period=20, std=2) -> Dict:
        middle = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return {'upper': upper, 'middle': middle, 'lower': lower}
    
    def calculate_adx(self, data: pd.DataFrame, period=14) -> pd.Series:
        """Simplified ADX calculation"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.concat([high - low, 
                       abs(high - close.shift()), 
                       abs(low - close.shift())], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        data = self.calculate_indicators(data)
        
        if len(data) < 2:
            return False, False, {}
        
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Multiple confirmations required
        rsi_oversold = current['rsi'] < 30 and prev['rsi'] < current['rsi']
        rsi_overbought = current['rsi'] > 70 and prev['rsi'] > current['rsi']
        
        macd_bullish = current['macd'] > current['signal'] and prev['macd'] <= prev['signal']
        macd_bearish = current['macd'] < current['signal'] and prev['macd'] >= prev['signal']
        
        price_below_bb = current['Close'] < current['bb_lower']
        price_above_bb = current['Close'] > current['bb_upper']
        
        strong_trend = current['adx'] > 25
        
        # Bullish: RSI oversold + MACD cross + below BB + strong trend
        bullish = rsi_oversold and macd_bullish and price_below_bb and strong_trend
        
        # Bearish: RSI overbought + MACD cross + above BB + strong trend
        bearish = rsi_overbought and macd_bearish and price_above_bb and strong_trend
        
        signal_data = {
            'rsi': current['rsi'],
            'macd': current['macd'],
            'signal': current['signal'],
            'adx': current['adx'],
            'bb_position': 'below' if price_below_bb else ('above' if price_above_bb else 'middle'),
            'type': 'Multi-Indicator Momentum'
        }
        
        return bullish, bearish, signal_data


class MarketExhaustionStrategy(BaseStrategy):
    """Market Exhaustion / Reversal Strategy"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # Volume analysis
        data['volume_ma'] = data['Volume'].rolling(window=20).mean()
        data['volume_spike'] = data['Volume'] / data['volume_ma']
        
        # Price momentum
        data['roc'] = ((data['Close'] - data['Close'].shift(10)) / 
                       data['Close'].shift(10) * 100)
        
        # Volatility
        data['atr'] = self.calculate_atr(data)
        data['atr_ratio'] = data['atr'] / data['atr'].rolling(20).mean()
        
        # Divergence detection
        data['price_higher_high'] = data['High'] > data['High'].shift(1)
        data['price_lower_low'] = data['Low'] < data['Low'].shift(1)
        
        return data
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift())
        low_close = abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        data = self.calculate_indicators(data)
        
        if len(data) < 3:
            return False, False, {}
        
        current = data.iloc[-1]
        
        # Exhaustion signals
        extreme_volume = current['volume_spike'] > 3.0
        extreme_roc = abs(current['roc']) > 5
        high_volatility = current['atr_ratio'] > 1.5
        
        # Bullish exhaustion: Extreme selling followed by reversal
        selling_exhaustion = (current['roc'] < -5 and 
                             extreme_volume and 
                             current['Close'] > data['Low'].iloc[-2])
        
        # Bearish exhaustion: Extreme buying followed by reversal
        buying_exhaustion = (current['roc'] > 5 and 
                            extreme_volume and 
                            current['Close'] < data['High'].iloc[-2])
        
        bullish = selling_exhaustion and high_volatility
        bearish = buying_exhaustion and high_volatility
        
        signal_data = {
            'volume_spike': current['volume_spike'],
            'roc': current['roc'],
            'atr_ratio': current['atr_ratio'],
            'type': 'Market Exhaustion'
        }
        
        return bullish, bearish, signal_data


# ============================================================================
# TRADING SYSTEM
# ============================================================================

class TradingSystem:
    def __init__(self):
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.strategy = None
    
    def set_strategy(self, strategy: BaseStrategy):
        """Set the active trading strategy"""
        self.strategy = strategy
    
    def fetch_data(self, ticker: str, interval: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch data with proper error handling"""
        try:
            time.sleep(2)
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if data.empty:
                return None
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            data.index = data.index.tz_convert(self.ist_tz)
            
            return data
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    def fetch_multi_ticker_data(self, tickers: List[str], interval: str, 
                                period: str) -> Optional[pd.DataFrame]:
        """Fetch data for multiple tickers (for pair trading)"""
        try:
            time.sleep(2)
            data = yf.download(tickers, period=period, interval=interval, progress=False)
            
            if data.empty:
                return None
            
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            data.index = data.index.tz_convert(self.ist_tz)
            
            return data
        except Exception as e:
            st.error(f"Error fetching multi-ticker data: {str(e)}")
            return None
    
    def calculate_sl_target(self, entry_price: float, position_type: str,
                           sl_config: Dict, target_config: Dict) -> Tuple[float, float]:
        """Calculate SL and target"""
        # Stop Loss
        if sl_config['type'] == 'Custom Points':
            sl_points = sl_config['value']
            sl_price = entry_price - sl_points if position_type == 'LONG' else entry_price + sl_points
        elif sl_config['type'] == 'Trail SL':
            sl_points = sl_config['value']
            sl_price = entry_price - sl_points if position_type == 'LONG' else entry_price + sl_points
        else:  # Signal-based
            sl_price = None
        
        # Target
        if target_config['type'] == 'Custom Points':
            target_points = target_config['value']
            target_price = entry_price + target_points if position_type == 'LONG' else entry_price - target_points
        else:  # Signal-based or trail
            target_price = None
        
        return sl_price, target_price
    
    def update_trailing_sl(self, position: Dict, current_price: float,
                           sl_config: Dict) -> float:
        """Update trailing stop loss"""
        if sl_config['type'] != 'Trail SL':
            return position['sl_price']
        
        trail_points = sl_config['value']
        position_type = position['type']
        current_sl = position['sl_price']
        
        if position_type == 'LONG':
            new_sl = current_price - trail_points
            return max(current_sl, new_sl) if current_sl else new_sl
        else:
            new_sl = current_price + trail_points
            return min(current_sl, new_sl) if current_sl else new_sl
    
    def check_exit_conditions(self, position: Dict, current_price: float,
                             data: pd.DataFrame, sl_config: Dict, 
                             target_config: Dict) -> Tuple[bool, str]:
        """Check exit conditions"""
        # Check SL
        if position['sl_price'] is not None:
            if position['type'] == 'LONG' and current_price <= position['sl_price']:
                return True, 'Stop Loss Hit'
            elif position['type'] == 'SHORT' and current_price >= position['sl_price']:
                return True, 'Stop Loss Hit'
        
        # Check Target
        if position['target_price'] is not None:
            if position['type'] == 'LONG' and current_price >= position['target_price']:
                return True, 'Target Achieved'
            elif position['type'] == 'SHORT' and current_price <= position['target_price']:
                return True, 'Target Achieved'
        
        # Check Signal-Based Exit (NEW - completed section)
        if sl_config['type'] == 'Signal Based' or target_config['type'] == 'Signal Based':
            bullish_signal, bearish_signal, _ = self.strategy.generate_signal(data)
        
        if position['type'] == 'LONG' and bearish_signal:
            return True, 'Bearish Signal Exit'
        elif position['type'] == 'SHORT' and bullish_signal:
            return True, 'Bullish Signal Exit'
    
        return False, ''
