import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import pytz
from typing import Dict, List, Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="Live Algorithmic Trading System",
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
    .trade-status {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    .status-long {
        background-color: #d4edda;
        color: #155724;
    }
    .status-short {
        background-color: #fff3cd;
        color: #856404;
    }
    .status-neutral {
        background-color: #d1ecf1;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trading_active' not in st.session_state:
    st.session_state.trading_active = False
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'current_position' not in st.session_state:
    st.session_state.current_position = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []

class TradingSystem:
    def __init__(self):
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        
    def fetch_data(self, ticker: str, interval: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch data with proper error handling and timezone conversion"""
        try:
            time.sleep(2)  # Rate limiting
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if data.empty:
                return None
            
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Convert to IST
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            data.index = data.index.tz_convert(self.ist_tz)
            
            return data
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    def calculate_indicator(self, data: pd.DataFrame, period: int, ind_type: str) -> pd.Series:
        """Calculate SMA or EMA"""
        if ind_type.upper() == 'SMA':
            return data['Close'].rolling(window=period).mean()
        else:  # EMA
            return data['Close'].ewm(span=period, adjust=False).mean()
    
    def detect_crossover(self, fast: pd.Series, slow: pd.Series) -> Tuple[bool, bool]:
        """Detect bullish or bearish crossover"""
        if len(fast) < 2 or len(slow) < 2:
            return False, False
        
        # Current values
        fast_curr, fast_prev = fast.iloc[-1], fast.iloc[-2]
        slow_curr, slow_prev = slow.iloc[-1], slow.iloc[-2]
        
        # Bullish crossover: fast crosses above slow
        bullish = (fast_prev <= slow_prev) and (fast_curr > slow_curr)
        
        # Bearish crossover: fast crosses below slow
        bearish = (fast_prev >= slow_prev) and (fast_curr < slow_curr)
        
        return bullish, bearish
    
    def calculate_sl_target(self, entry_price: float, position_type: str, 
                           sl_config: Dict, target_config: Dict,
                           current_fast: float, current_slow: float) -> Tuple[float, float]:
        """Calculate stop loss and target levels"""
        
        # Stop Loss Calculation
        if sl_config['type'] == 'Custom Points':
            sl_points = sl_config['value']
            if position_type == 'LONG':
                sl_price = entry_price - sl_points
            else:
                sl_price = entry_price + sl_points
        elif sl_config['type'] == 'EMA/SMA Crossover':
            # SL will be checked dynamically via crossover
            sl_price = None
        else:  # Trail SL
            sl_points = sl_config['value']
            if position_type == 'LONG':
                sl_price = entry_price - sl_points
            else:
                sl_price = entry_price + sl_points
        
        # Target Calculation
        if target_config['type'] == 'Custom Points':
            target_points = target_config['value']
            if position_type == 'LONG':
                target_price = entry_price + target_points
            else:
                target_price = entry_points - target_points
        elif target_config['type'] == 'EMA/SMA Crossover':
            target_price = None  # Will exit on crossover
        else:  # Trail
            target_price = None  # Trailing target
        
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
            # Update SL only if price moved in favor
            new_sl = current_price - trail_points
            return max(current_sl, new_sl) if current_sl else new_sl
        else:  # SHORT
            new_sl = current_price + trail_points
            return min(current_sl, new_sl) if current_sl else new_sl
    
    def check_exit_conditions(self, position: Dict, current_price: float,
                             fast_val: float, slow_val: float,
                             sl_config: Dict, target_config: Dict) -> Tuple[bool, str]:
        """Check if position should be exited"""
        
        # Check SL hit
        if position['sl_price'] is not None:
            if position['type'] == 'LONG' and current_price <= position['sl_price']:
                return True, 'Stop Loss Hit'
            elif position['type'] == 'SHORT' and current_price >= position['sl_price']:
                return True, 'Stop Loss Hit'
        
        # Check target hit
        if position['target_price'] is not None:
            if position['type'] == 'LONG' and current_price >= position['target_price']:
                return True, 'Target Achieved'
            elif position['type'] == 'SHORT' and current_price <= position['target_price']:
                return True, 'Target Achieved'
        
        # Check crossover exit
        if sl_config['type'] == 'EMA/SMA Crossover' or target_config['type'] == 'EMA/SMA Crossover':
            if position['type'] == 'LONG' and fast_val < slow_val:
                return True, 'EMA/SMA Crossover Exit'
            elif position['type'] == 'SHORT' and fast_val > slow_val:
                return True, 'EMA/SMA Crossover Exit'
        
        return False, ''
    
    def analyze_trade_performance(self, trade: Dict) -> str:
        """Provide AI-powered trade analysis"""
        pnl = trade['pnl']
        pnl_pct = trade['pnl_percent']
        duration = trade['duration']
        exit_reason = trade['exit_reason']
        
        analysis = []
        
        # Performance assessment
        if pnl > 0:
            analysis.append(f"✅ **Profitable Trade**: Gained {pnl_pct:.2f}% ({pnl:.2f} points)")
            
            if exit_reason == 'Target Achieved':
                analysis.append("🎯 **Good Exit**: Target was hit as planned")
            elif exit_reason == 'EMA/SMA Crossover Exit':
                analysis.append("📊 **Systematic Exit**: Exited on crossover signal")
            
            if pnl_pct > 1.5:
                analysis.append("💪 **Strong Performance**: Excellent profit capture")
            
        else:
            analysis.append(f"❌ **Loss Trade**: Lost {abs(pnl_pct):.2f}% ({abs(pnl):.2f} points)")
            
            if exit_reason == 'Stop Loss Hit':
                analysis.append("🛡️ **Risk Managed**: SL protected from larger loss")
                analysis.append("💡 **Suggestion**: Consider wider SL or better entry timing")
            
            if abs(pnl_pct) > 2:
                analysis.append("⚠️ **Large Loss**: Review risk management parameters")
        
        # Duration analysis
        if duration < 300:  # Less than 5 minutes
            analysis.append("⚡ **Quick Trade**: Very short duration - consider market volatility")
        elif duration > 3600:  # More than 1 hour
            analysis.append("⏰ **Long Hold**: Extended position - ensure trend alignment")
        
        # Recommendations
        analysis.append("\n**Recommendations:**")
        if pnl > 0:
            analysis.append("• Continue following the strategy discipline")
            analysis.append("• Consider scaling position size on strong setups")
        else:
            analysis.append("• Review entry conditions - wait for stronger signals")
            analysis.append("• Ensure proper risk-reward ratio (minimum 1:1.5)")
            analysis.append("• Consider adding confirmation indicators")
        
        return "\n".join(analysis)
    
    def get_market_status(self, position: Dict, current_price: float) -> str:
        """Get current market status relative to position"""
        if not position:
            return "No active position"
        
        entry_price = position['entry_price']
        position_type = position['type']
        pnl = current_price - entry_price if position_type == 'LONG' else entry_price - current_price
        pnl_pct = (pnl / entry_price) * 100
        
        status_parts = []
        
        # Position direction
        if position_type == 'LONG':
            status_parts.append("📈 **LONG Position Active**")
        else:
            status_parts.append("📉 **SHORT Position Active**")
        
        # P&L status
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
            status_parts.append("➡️ At entry price - waiting for movement")
        
        # SL status
        if position['sl_price']:
            sl_distance = abs(current_price - position['sl_price'])
            status_parts.append(f"🛡️ SL Distance: {sl_distance:.2f} pts")
            
            if position.get('sl_updated', False):
                status_parts.append("📊 Trailing SL Active")
        
        # Target status
        if position['target_price']:
            target_distance = abs(position['target_price'] - current_price)
            status_parts.append(f"🎯 Target Distance: {target_distance:.2f} pts")
        
        return "\n".join(status_parts)

# Main Application
st.markdown('<h1 class="main-header">🚀 Live Algorithmic Trading System</h1>', unsafe_allow_html=True)
st.markdown("### Real-Time EMA/SMA Crossover Strategy with Advanced Risk Management")

# Initialize trading system
trading_system = TradingSystem()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Trading Configuration")
    
    # Asset Selection
    st.subheader("📊 Asset Selection")
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
    
    # Timeframe Selection
    st.subheader("⏰ Timeframe")
    col1, col2 = st.columns(2)
    with col1:
        interval = st.selectbox(
            "Interval",
            ["1m", "5m", "15m", "30m", "1h", "1d"],
            index=0
        )
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
    
    # Indicator Configuration
    st.subheader("📈 Indicators")
    col1, col2 = st.columns(2)
    with col1:
        ind1_type = st.selectbox("Fast Indicator", ["EMA", "SMA"], key="ind1")
        ind1_period = st.number_input("Fast Period", min_value=1, max_value=200, value=9, key="period1")
    with col2:
        ind2_type = st.selectbox("Slow Indicator", ["EMA", "SMA"], key="ind2")
        ind2_period = st.number_input("Slow Period", min_value=1, max_value=200, value=20, key="period2")
    
    # Risk Management
    st.subheader("🛡️ Risk Management")
    
    # Stop Loss Configuration
    sl_type = st.selectbox(
        "Stop Loss Type",
        ["Custom Points", "Trail SL", "EMA/SMA Crossover"]
    )
    
    sl_value = None
    if sl_type in ["Custom Points", "Trail SL"]:
        sl_value = st.number_input(
            "SL Points",
            min_value=0.0,
            value=10.0,
            step=0.5,
            help="Points away from entry price"
        )
    
    sl_config = {"type": sl_type, "value": sl_value}
    
    # Target Configuration
    target_type = st.selectbox(
        "Target Type",
        ["Custom Points", "Trail Target", "EMA/SMA Crossover"]
    )
    
    target_value = None
    if target_type in ["Custom Points", "Trail Target"]:
        target_value = st.number_input(
            "Target Points",
            min_value=0.0,
            value=20.0,
            step=0.5,
            help="Points away from entry price"
        )
    
    target_config = {"type": target_type, "value": target_value}
    
    # Position Size
    position_size = st.number_input(
        "Position Size (Qty)",
        min_value=1,
        value=1,
        help="Number of units to trade"
    )

# Main Content Area
tab1, tab2, tab3 = st.tabs(["📊 Live Trading", "📈 Trade History", "📋 Trade Log"])

with tab1:
    # Control Buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("▶️ Start Trading", type="primary", disabled=st.session_state.trading_active):
            st.session_state.trading_active = True
            st.session_state.trade_log.append({
                'time': datetime.now(trading_system.ist_tz),
                'message': f'Trading started for {ticker} on {interval} timeframe'
            })
            st.rerun()
    
    with col2:
        if st.button("⏸️ Stop Trading", type="secondary", disabled=not st.session_state.trading_active):
            st.session_state.trading_active = False
            
            # Close any open position
            if st.session_state.current_position:
                st.session_state.trade_log.append({
                    'time': datetime.now(trading_system.ist_tz),
                    'message': 'Trading stopped - Closing open position'
                })
                # Add logic to close position here
                st.session_state.current_position = None
            
            st.rerun()
    
    with col3:
        if st.session_state.trading_active:
            st.markdown('<p class="status-running">🟢 Trading Active</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-stopped">🔴 Trading Stopped</p>', unsafe_allow_html=True)
    
    # Trading Information Header
    if st.session_state.trading_active or st.session_state.current_position:
        st.markdown("---")
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        
        with info_col1:
            st.metric("Asset", ticker)
        with info_col2:
            st.metric("Timeframe", interval)
        with info_col3:
            st.metric("Fast Indicator", f"{ind1_type}{ind1_period}")
        with info_col4:
            st.metric("Slow Indicator", f"{ind2_type}{ind2_period}")
        
        # Calculate candles used
        if interval == "1m":
            candles_for_calc = max(ind1_period, ind2_period) + 50
        else:
            candles_for_calc = max(ind1_period, ind2_period) + 20
        
        st.info(f"📊 Using last **{candles_for_calc}** candles for indicator calculation")
    
    # Live Trading Loop
    if st.session_state.trading_active:
        # Create fixed containers for updates
        status_container = st.container()
        metrics_container = st.container()
        chart_container = st.container()
        update_container = st.container()
        
        # Fetch latest data
        data = trading_system.fetch_data(ticker, interval, period)
        
        if data is None or len(data) < max(ind1_period, ind2_period):
            st.error("⚠️ Insufficient data. Waiting for next update...")
            time.sleep(2)
            st.rerun()
        
        # Calculate indicators
        fast_ind = trading_system.calculate_indicator(data, ind1_period, ind1_type)
        slow_ind = trading_system.calculate_indicator(data, ind2_period, ind2_type)
        
        # Get current values
        current_price = data['Close'].iloc[-1]
        current_fast = fast_ind.iloc[-1]
        current_slow = slow_ind.iloc[-1]
        current_time = data.index[-1]
        
        # Check for crossover signals
        bullish_cross, bearish_cross = trading_system.detect_crossover(fast_ind, slow_ind)
        
        # Position Management
        if st.session_state.current_position is None:
            # No position - check for entry signal
            if bullish_cross:
                # Enter LONG
                sl_price, target_price = trading_system.calculate_sl_target(
                    current_price, 'LONG', sl_config, target_config,
                    current_fast, current_slow
                )
                
                st.session_state.current_position = {
                    'type': 'LONG',
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'quantity': position_size,
                    'sl_price': sl_price,
                    'target_price': target_price,
                    'fast_val': current_fast,
                    'slow_val': current_slow,
                    'sl_updated': False
                }
                
                st.session_state.trade_log.append({
                    'time': current_time,
                    'message': f'🟢 LONG Entry at {current_price:.2f} | SL: {sl_price:.2f if sl_price else "Crossover"} | Target: {target_price:.2f if target_price else "Crossover"}'
                })
                
            elif bearish_cross:
                # Enter SHORT
                sl_price, target_price = trading_system.calculate_sl_target(
                    current_price, 'SHORT', sl_config, target_config,
                    current_fast, current_slow
                )
                
                st.session_state.current_position = {
                    'type': 'SHORT',
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'quantity': position_size,
                    'sl_price': sl_price,
                    'target_price': target_price,
                    'fast_val': current_fast,
                    'slow_val': current_slow,
                    'sl_updated': False
                }
                
                st.session_state.trade_log.append({
                    'time': current_time,
                    'message': f'🔴 SHORT Entry at {current_price:.2f} | SL: {sl_price:.2f if sl_price else "Crossover"} | Target: {target_price:.2f if target_price else "Crossover"}'
                })
        
        else:
            # Have position - manage it
            position = st.session_state.current_position
            
            # Update trailing SL
            if sl_config['type'] == 'Trail SL':
                new_sl = trading_system.update_trailing_sl(position, current_price, sl_config)
                if new_sl != position['sl_price']:
                    position['sl_price'] = new_sl
                    position['sl_updated'] = True
                    st.session_state.trade_log.append({
                        'time': current_time,
                        'message': f'📊 Trailing SL updated to {new_sl:.2f}'
                    })
            
            # Check exit conditions
            should_exit, exit_reason = trading_system.check_exit_conditions(
                position, current_price, current_fast, current_slow,
                sl_config, target_config
            )
            
            if should_exit:
                # Close position
                entry_price = position['entry_price']
                pnl = (current_price - entry_price) if position['type'] == 'LONG' else (entry_price - current_price)
                pnl = pnl * position['quantity']
                pnl_percent = (pnl / (entry_price * position['quantity'])) * 100
                duration = (current_time - position['entry_time']).total_seconds()
                
                trade_record = {
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'type': position['type'],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': position['quantity'],
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'duration': duration,
                    'exit_reason': exit_reason,
                    'fast_ind': f"{ind1_type}{ind1_period}",
                    'slow_ind': f"{ind2_type}{ind2_period}"
                }
                
                # Add trade analysis
                trade_record['analysis'] = trading_system.analyze_trade_performance(trade_record)
                
                st.session_state.trade_history.append(trade_record)
                st.session_state.current_position = None
                
                st.session_state.trade_log.append({
                    'time': current_time,
                    'message': f'🏁 Position Closed: {exit_reason} | P&L: {pnl:.2f} ({pnl_percent:.2f}%)'
                })
        
        # Display current status
        with status_container:
            st.markdown("### 📊 Current Market Status")
            
            if st.session_state.current_position:
                status_html = trading_system.get_market_status(
                    st.session_state.current_position, 
                    current_price
                )
                st.markdown(status_html)
            else:
                st.info("⏳ Waiting for crossover signal...")
                st.markdown(f"**Current Price:** {current_price:.2f}")
                st.markdown(f"**Fast {ind1_type}{ind1_period}:** {current_fast:.2f}")
                st.markdown(f"**Slow {ind2_type}{ind2_period}:** {current_slow:.2f}")
        
        # Display metrics
        with metrics_container:
            if st.session_state.current_position:
                pos = st.session_state.current_position
                m1, m2, m3, m4, m5 = st.columns(5)
                
                with m1:
                    st.metric("Entry Price", f"{pos['entry_price']:.2f}")
                with m2:
                    st.metric("Current Price", f"{current_price:.2f}")
                with m3:
                    pnl = (current_price - pos['entry_price']) if pos['type'] == 'LONG' else (pos['entry_price'] - current_price)
                    pnl_total = pnl * pos['quantity']
                    st.metric("P&L Points", f"{pnl:.2f}", delta=f"{pnl_total:.2f}")
                with m4:
                    st.metric(f"{ind1_type}{ind1_period}", f"{current_fast:.2f}")
                with m5:
                    st.metric(f"{ind2_type}{ind2_period}", f"{current_slow:.2f}")
        
        # Display chart
        with chart_container:
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
            
            # Indicators
            fig.add_trace(go.Scatter(
                x=data.index,
                y=fast_ind,
                mode='lines',
                name=f'{ind1_type}{ind1_period}',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=slow_ind,
                mode='lines',
                name=f'{ind2_type}{ind2_period}',
                line=dict(color='red', width=2)
            ))
            
            # Mark entry if position exists
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
            
            fig.update_layout(
                title=f'{ticker} - {interval} Chart',
                yaxis_title='Price',
                xaxis_title='Time (IST)',
                template='plotly_white',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"live_chart_{current_time.timestamp()}")
        
        # Update timestamp in fixed box
        with update_container:
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 0.5rem; border-radius: 0.3rem; text-align: center; margin-top: 1rem;'>
                <small>Last updated: {current_time.strftime('%Y-%m-%d %H:%M:%S IST')}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Rate limiting delay before next update
        time.sleep(2)
        
        # Continue only if still active
        if st.session_state.trading_active:
            st.rerun()
    
    elif st.session_state.current_position:
        # Display current position even when trading is stopped
        st.warning("⚠️ Trading stopped but position still open. Close manually or restart trading.")
        
        # Fetch latest data to show current status
        data = trading_system.fetch_data(ticker, interval, period)
        if data is not None and len(data) > 0:
            current_price = data['Close'].iloc[-1]
            
            pos = st.session_state.current_position
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                st.metric("Position Type", pos['type'])
            with m2:
                st.metric("Entry Price", f"{pos['entry_price']:.2f}")
            with m3:
                st.metric("Current Price", f"{current_price:.2f}")
            with m4:
                pnl = (current_price - pos['entry_price']) if pos['type'] == 'LONG' else (pos['entry_price'] - current_price)
                pnl_total = pnl * pos['quantity']
                st.metric("P&L", f"{pnl_total:.2f}", delta=f"{pnl:.2f} pts")
            
            if st.button("🔴 Force Close Position"):
                # Close position at current price
                entry_price = pos['entry_price']
                pnl = (current_price - entry_price) if pos['type'] == 'LONG' else (entry_price - current_price)
                pnl = pnl * pos['quantity']
                pnl_percent = (pnl / (entry_price * pos['quantity'])) * 100
                
                trade_record = {
                    'entry_time': pos['entry_time'],
                    'exit_time': datetime.now(trading_system.ist_tz),
                    'type': pos['type'],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': pos['quantity'],
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'duration': (datetime.now(trading_system.ist_tz) - pos['entry_time']).total_seconds(),
                    'exit_reason': 'Manual Close',
                    'fast_ind': f"{ind1_type}{ind1_period}",
                    'slow_ind': f"{ind2_type}{ind2_period}"
                }
                
                trade_record['analysis'] = trading_system.analyze_trade_performance(trade_record)
                st.session_state.trade_history.append(trade_record)
                st.session_state.current_position = None
                
                st.success("Position closed successfully!")
                st.rerun()

with tab2:
    st.header("📈 Trade History & Performance")
    
    if not st.session_state.trade_history:
        st.info("No trades executed yet. Start trading to see history.")
    else:
        # Performance Summary
        total_trades = len(st.session_state.trade_history)
        winning_trades = sum(1 for t in st.session_state.trade_history if t['pnl'] > 0)
        losing_trades = sum(1 for t in st.session_state.trade_history if t['pnl'] < 0)
        total_pnl = sum(t['pnl'] for t in st.session_state.trade_history)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Winning Trades", winning_trades, delta=f"{win_rate:.1f}%")
        with col3:
            st.metric("Losing Trades", losing_trades)
        with col4:
            pnl_color = "normal" if total_pnl >= 0 else "inverse"
            st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"{'📈' if total_pnl >= 0 else '📉'}")
        with col5:
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
            st.metric("Avg P&L/Trade", f"{avg_pnl:.2f}")
        
        st.markdown("---")
        
        # Trade Details Table
        st.subheader("📋 Detailed Trade Records")
        
        trade_df = pd.DataFrame([
            {
                'Entry Time': t['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'Exit Time': t['exit_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'Type': t['type'],
                'Entry Price': f"{t['entry_price']:.2f}",
                'Exit Price': f"{t['exit_price']:.2f}",
                'Quantity': t['quantity'],
                'P&L Points': f"{t['pnl'] / t['quantity']:.2f}",
                'P&L Total': f"{t['pnl']:.2f}",
                'P&L %': f"{t['pnl_percent']:.2f}%",
                'Duration (sec)': int(t['duration']),
                'Exit Reason': t['exit_reason']
            }
            for t in st.session_state.trade_history
        ])
        
        st.dataframe(trade_df, use_container_width=True, hide_index=True)
        
        # Individual Trade Analysis
        st.markdown("---")
        st.subheader("🔍 Trade-by-Trade Analysis")
        
        for idx, trade in enumerate(reversed(st.session_state.trade_history), 1):
            with st.expander(f"Trade #{total_trades - idx + 1} - {trade['type']} - P&L: {trade['pnl']:.2f} ({trade['pnl_percent']:.2f}%)"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**Trade Details:**")
                    st.markdown(f"- Entry: {trade['entry_price']:.2f}")
                    st.markdown(f"- Exit: {trade['exit_price']:.2f}")
                    st.markdown(f"- Quantity: {trade['quantity']}")
                    st.markdown(f"- Duration: {int(trade['duration'])} seconds")
                    st.markdown(f"- Exit Reason: {trade['exit_reason']}")
                
                with col2:
                    st.markdown("**AI Analysis:**")
                    st.markdown(trade['analysis'])
        
        # Performance Chart
        st.markdown("---")
        st.subheader("📊 Cumulative P&L Chart")
        
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
        
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break Even")
        
        fig_pnl.update_layout(
            title='Cumulative Profit & Loss',
            xaxis_title='Trade Number',
            yaxis_title='Cumulative P&L',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig_pnl, use_container_width=True, key="cumulative_pnl_chart")
        
        # Win/Loss Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Winning', 'Losing'],
                values=[winning_trades, losing_trades],
                marker_colors=['#00cc00', '#ff6b6b']
            )])
            fig_pie.update_layout(title='Win/Loss Distribution', height=300)
            st.plotly_chart(fig_pie, use_container_width=True, key="win_loss_pie_chart")
        
        with col2:
            # P&L Distribution
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
            st.plotly_chart(fig_hist, use_container_width=True, key="pnl_distribution_chart")

with tab3:
    st.header("📋 Real-Time Trade Log")
    
    if not st.session_state.trade_log:
        st.info("No trade activity logged yet.")
    else:
        # Display log in reverse chronological order with fixed height scrollable container
        st.markdown("### Recent Activity")
        
        # Create scrollable container with fixed height
        log_entries = []
        for log in reversed(st.session_state.trade_log[-100:]):  # Show last 100 entries
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
        
        # Clear log button
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
    <p><strong>⚠️ Risk Warning:</strong> Trading involves substantial risk of loss. This system is for educational purposes only.</p>
    <p>Always use proper risk management and never trade with money you cannot afford to lose.</p>
    <p style='margin-top: 1rem; font-size: 0.9rem;'>Powered by yfinance | Real-time data with 2-second API rate limiting</p>
</div>
""", unsafe_allow_html=True)
