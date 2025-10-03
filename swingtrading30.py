import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.signal import find_peaks, argrelextrema
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Elliott Wave Trading System", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem;}
    .sub-header {font-size: 1.5rem; font-weight: bold; color: #ff7f0e; margin-top: 1rem;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;}
    .trade-signal {font-size: 1.2rem; font-weight: bold; padding: 1rem; border-radius: 0.5rem; text-align: center; margin: 1rem 0;}
    .buy-signal {background-color: #d4edda; color: #155724; border: 2px solid #28a745;}
    .sell-signal {background-color: #f8d7da; color: #721c24; border: 2px solid #dc3545;}
    .hold-signal {background-color: #fff3cd; color: #856404; border: 2px solid #ffc107;}
</style>
""", unsafe_allow_html=True)

class ElliottWaveAnalyzer:
    def __init__(self, data, sensitivity=0.05):
        self.data = data.copy()
        self.sensitivity = sensitivity
        self.waves = []
        self.current_wave = None
        
    def identify_pivots(self):
        """Identify swing highs and lows"""
        high = self.data['High'].values
        low = self.data['Low'].values
        close = self.data['Close'].values
        
        # Dynamic window based on data length
        window = max(5, len(self.data) // 50)
        
        # Find peaks and troughs
        high_peaks_idx = argrelextrema(high, np.greater, order=window)[0]
        low_peaks_idx = argrelextrema(low, np.less, order=window)[0]
        
        # Combine and sort pivots
        pivots = []
        for idx in high_peaks_idx:
            pivots.append({'index': idx, 'price': high[idx], 'type': 'high'})
        for idx in low_peaks_idx:
            pivots.append({'index': idx, 'price': low[idx], 'type': 'low'})
        
        pivots = sorted(pivots, key=lambda x: x['index'])
        
        # Filter pivots by minimum price change
        if len(pivots) > 0:
            filtered_pivots = [pivots[0]]
            for pivot in pivots[1:]:
                last_pivot = filtered_pivots[-1]
                price_change = abs(pivot['price'] - last_pivot['price']) / last_pivot['price']
                if price_change >= self.sensitivity and pivot['type'] != last_pivot['type']:
                    filtered_pivots.append(pivot)
            return filtered_pivots
        return []
    
    def calculate_fibonacci_levels(self, start_price, end_price):
        """Calculate Fibonacci retracement and extension levels"""
        diff = end_price - start_price
        return {
            'level_0': start_price,
            'level_236': start_price + 0.236 * diff,
            'level_382': start_price + 0.382 * diff,
            'level_500': start_price + 0.500 * diff,
            'level_618': start_price + 0.618 * diff,
            'level_786': start_price + 0.786 * diff,
            'level_100': end_price,
            'level_1272': start_price + 1.272 * diff,
            'level_1618': start_price + 1.618 * diff,
            'level_2618': start_price + 2.618 * diff,
        }
    
    def identify_elliott_waves(self):
        """Identify Elliott Wave patterns (5-3 wave structure)"""
        pivots = self.identify_pivots()
        
        if len(pivots) < 8:
            return None
        
        # Look for 5-wave impulse pattern
        waves_5 = []
        for i in range(len(pivots) - 7):
            sequence = pivots[i:i+8]
            
            # Check for alternating high-low pattern
            if self.is_valid_impulse_wave(sequence):
                waves_5.append({
                    'type': 'impulse',
                    'direction': 'bullish' if sequence[0]['type'] == 'low' else 'bearish',
                    'waves': sequence[:6],  # 5 waves + starting point
                    'start_idx': sequence[0]['index'],
                    'end_idx': sequence[5]['index']
                })
        
        # Look for 3-wave corrective pattern
        waves_3 = []
        for i in range(len(pivots) - 5):
            sequence = pivots[i:i+4]
            if self.is_valid_corrective_wave(sequence):
                waves_3.append({
                    'type': 'corrective',
                    'waves': sequence,
                    'start_idx': sequence[0]['index'],
                    'end_idx': sequence[3]['index']
                })
        
        return {'impulse': waves_5, 'corrective': waves_3}
    
    def is_valid_impulse_wave(self, sequence):
        """Validate 5-wave impulse pattern rules"""
        if len(sequence) < 6:
            return False
        
        # Rule 1: Wave 2 cannot retrace more than 100% of wave 1
        wave1_size = abs(sequence[1]['price'] - sequence[0]['price'])
        wave2_retrace = abs(sequence[2]['price'] - sequence[1]['price'])
        if wave2_retrace >= wave1_size:
            return False
        
        # Rule 2: Wave 3 is never the shortest
        wave3_size = abs(sequence[3]['price'] - sequence[2]['price'])
        wave5_size = abs(sequence[5]['price'] - sequence[4]['price'])
        if wave3_size < wave1_size and wave3_size < wave5_size:
            return False
        
        # Rule 3: Wave 4 does not overlap wave 1 price territory
        if sequence[0]['type'] == 'low':  # Bullish
            if sequence[4]['price'] <= sequence[1]['price']:
                return False
        else:  # Bearish
            if sequence[4]['price'] >= sequence[1]['price']:
                return False
        
        return True
    
    def is_valid_corrective_wave(self, sequence):
        """Validate 3-wave corrective pattern (ABC)"""
        if len(sequence) < 4:
            return False
        
        # Wave B should retrace between 50-90% of wave A
        wave_a = abs(sequence[1]['price'] - sequence[0]['price'])
        wave_b = abs(sequence[2]['price'] - sequence[1]['price'])
        retrace_ratio = wave_b / wave_a if wave_a != 0 else 0
        
        return 0.38 <= retrace_ratio <= 1.0
    
    def generate_trading_signals(self):
        """Generate entry, target, and stop-loss levels"""
        wave_patterns = self.identify_elliott_waves()
        current_price = self.data['Close'].iloc[-1]
        signals = []
        
        if not wave_patterns or not wave_patterns['impulse']:
            return None
        
        # Analyze most recent impulse wave
        if wave_patterns['impulse']:
            latest_impulse = wave_patterns['impulse'][-1]
            waves = latest_impulse['waves']
            direction = latest_impulse['direction']
            
            # Determine current position in wave cycle
            last_pivot_idx = waves[-1]['index']
            candles_since_pivot = len(self.data) - last_pivot_idx - 1
            
            # Calculate Fibonacci levels from wave 3 to wave 4
            if len(waves) >= 5:
                wave3_price = waves[3]['price']
                wave4_price = waves[4]['price']
                fib_levels = self.calculate_fibonacci_levels(wave4_price, wave3_price)
                
                signal = {
                    'timestamp': self.data.index[-1],
                    'current_price': current_price,
                    'wave_position': 'Wave 5' if candles_since_pivot < 20 else 'Potential new cycle',
                    'direction': direction,
                    'confidence': self.calculate_confidence(waves, current_price),
                    'candles_since_pivot': candles_since_pivot
                }
                
                if direction == 'bullish':
                    # Bullish setup - waiting for wave 4 pullback to complete
                    signal['action'] = 'BUY' if current_price <= fib_levels['level_618'] else 'WAIT'
                    signal['entry'] = fib_levels['level_618']
                    signal['stop_loss'] = wave4_price * 0.98
                    signal['target1'] = fib_levels['level_1272']
                    signal['target2'] = fib_levels['level_1618']
                    signal['target3'] = fib_levels['level_2618']
                else:
                    # Bearish setup
                    signal['action'] = 'SELL' if current_price >= fib_levels['level_618'] else 'WAIT'
                    signal['entry'] = fib_levels['level_618']
                    signal['stop_loss'] = wave4_price * 1.02
                    signal['target1'] = fib_levels['level_1272']
                    signal['target2'] = fib_levels['level_1618']
                    signal['target3'] = fib_levels['level_2618']
                
                signal['risk_reward'] = abs(signal['target1'] - signal['entry']) / abs(signal['entry'] - signal['stop_loss'])
                signals.append(signal)
        
        return signals[0] if signals else None

    def calculate_confidence(self, waves, current_price):
        """Calculate confidence score based on wave quality"""
        score = 50  # Base score
        
        # Check wave proportions
        wave1 = abs(waves[1]['price'] - waves[0]['price'])
        wave3 = abs(waves[3]['price'] - waves[2]['price'])
        wave5 = abs(waves[5]['price'] - waves[4]['price']) if len(waves) > 5 else 0
        
        # Wave 3 extension is positive
        if wave3 > wave1 * 1.5:
            score += 15
        
        # Wave 5 extension
        if wave5 > 0 and wave1 * 0.8 <= wave5 <= wave1 * 1.2:
            score += 15
        
        # Fibonacci alignment
        if len(waves) >= 5:
            wave2_retrace = abs(waves[2]['price'] - waves[1]['price']) / wave1
            if 0.5 <= wave2_retrace <= 0.618:
                score += 20
        
        return min(score, 95)

def backtest_strategy(data, signals_history):
    """Backtest the Elliott Wave strategy"""
    initial_capital = 100000
    capital = initial_capital
    position = None
    trades = []
    
    for signal in signals_history:
        if signal['action'] in ['BUY', 'SELL'] and position is None:
            # Enter trade
            position = {
                'type': signal['action'],
                'entry_price': signal['entry'],
                'stop_loss': signal['stop_loss'],
                'target': signal['target1'],
                'entry_date': signal['timestamp'],
                'shares': capital * 0.95 / signal['entry']  # 95% capital utilization
            }
        
        elif position:
            # Check exit conditions
            current_price = data.loc[signal['timestamp'], 'Close']
            
            if position['type'] == 'BUY':
                if current_price >= position['target']:
                    # Target hit
                    profit = (current_price - position['entry_price']) * position['shares']
                    capital += profit
                    trades.append({
                        'entry': position['entry_date'],
                        'exit': signal['timestamp'],
                        'type': 'BUY',
                        'profit': profit,
                        'return': (current_price / position['entry_price'] - 1) * 100
                    })
                    position = None
                elif current_price <= position['stop_loss']:
                    # Stop loss hit
                    loss = (position['entry_price'] - current_price) * position['shares']
                    capital -= loss
                    trades.append({
                        'entry': position['entry_date'],
                        'exit': signal['timestamp'],
                        'type': 'BUY',
                        'profit': -loss,
                        'return': (current_price / position['entry_price'] - 1) * 100
                    })
                    position = None
            
            elif position['type'] == 'SELL':
                if current_price <= position['target']:
                    profit = (position['entry_price'] - current_price) * position['shares']
                    capital += profit
                    trades.append({
                        'entry': position['entry_date'],
                        'exit': signal['timestamp'],
                        'type': 'SELL',
                        'profit': profit,
                        'return': (position['entry_price'] / current_price - 1) * 100
                    })
                    position = None
                elif current_price >= position['stop_loss']:
                    loss = (current_price - position['entry_price']) * position['shares']
                    capital -= loss
                    trades.append({
                        'entry': position['entry_date'],
                        'exit': signal['timestamp'],
                        'type': 'SELL',
                        'profit': -loss,
                        'return': (position['entry_price'] / current_price - 1) * 100
                    })
                    position = None
    
    if trades:
        df_trades = pd.DataFrame(trades)
        win_rate = len(df_trades[df_trades['profit'] > 0]) / len(df_trades) * 100
        avg_return = df_trades['return'].mean()
        total_return = (capital - initial_capital) / initial_capital * 100
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'final_capital': capital,
            'trades': df_trades
        }
    
    return None

def plot_elliott_waves(data, analyzer, signal):
    """Create interactive plot with Elliott Waves"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    # Plot pivots
    pivots = analyzer.identify_pivots()
    if pivots:
        pivot_indices = [p['index'] for p in pivots]
        pivot_prices = [p['price'] for p in pivots]
        pivot_dates = [data.index[p['index']] for p in pivots]
        
        fig.add_trace(go.Scatter(
            x=pivot_dates,
            y=pivot_prices,
            mode='markers+lines',
            name='Elliott Wave Pivots',
            marker=dict(size=10, color='purple'),
            line=dict(color='purple', width=2, dash='dot')
        ))
        
        # Label waves
        for i, pivot in enumerate(pivots[-6:]):  # Show last 6 pivots
            fig.add_annotation(
                x=data.index[pivot['index']],
                y=pivot['price'],
                text=f"W{i+1}",
                showarrow=True,
                arrowhead=2,
                bgcolor='yellow',
                opacity=0.8
            )
    
    # Plot signal levels
    if signal:
        fig.add_hline(y=signal['entry'], line_dash="dash", line_color="blue", 
                     annotation_text=f"Entry: {signal['entry']:.2f}")
        fig.add_hline(y=signal['stop_loss'], line_dash="dash", line_color="red", 
                     annotation_text=f"Stop Loss: {signal['stop_loss']:.2f}")
        fig.add_hline(y=signal['target1'], line_dash="dash", line_color="green", 
                     annotation_text=f"Target 1: {signal['target1']:.2f}")
        fig.add_hline(y=signal['target2'], line_dash="dot", line_color="green", 
                     annotation_text=f"Target 2: {signal['target2']:.2f}")
    
    fig.update_layout(
        title='Elliott Wave Analysis',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark',
        height=700,
        xaxis_rangeslider_visible=False
    )
    
    return fig

# Streamlit App
def main():
    st.markdown('<div class="main-header">🌊 Elliott Wave Trading System</div>', unsafe_allow_html=True)
    st.markdown("**Professional Elliott Wave Analysis for Stocks, Indices, Crypto & Forex**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Asset selection
        asset_type = st.selectbox("Asset Type", ["Stock", "Crypto", "Forex", "Index"])
        
        if asset_type == "Stock":
            symbol = st.text_input("Stock Symbol", "AAPL").upper()
        elif asset_type == "Crypto":
            crypto = st.selectbox("Cryptocurrency", ["BTC", "ETH", "BNB", "SOL", "ADA"])
            symbol = f"{crypto}-USD"
        elif asset_type == "Forex":
            symbol = st.selectbox("Forex Pair", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"])
        else:
            symbol = st.selectbox("Index", ["^GSPC", "^DJI", "^IXIC", "^RUT"])
        
        # Timeframe
        timeframe = st.selectbox("Timeframe", ["1d", "1h", "4h", "1wk"])
        period_map = {"1d": "1y", "1h": "60d", "4h": "90d", "1wk": "5y"}
        period = period_map[timeframe]
        
        # Parameters
        sensitivity = st.slider("Wave Sensitivity", 0.01, 0.15, 0.05, 0.01)
        
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📊 Elliott Wave Rules")
        st.info("""
        **5-Wave Impulse:**
        - Wave 2 never retraces >100% of Wave 1
        - Wave 3 never shortest
        - Wave 4 doesn't overlap Wave 1
        
        **3-Wave Correction:**
        - ABC pattern
        - Wave B retraces 38-90% of A
        """)
    
    # Main content
    if analyze_button:
        try:
            with st.spinner(f"Fetching data for {symbol}..."):
                # Download data
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=timeframe)
                
                if data.empty:
                    st.error("❌ No data available for this symbol. Please check the symbol and try again.")
                    return
                
                # Analyze
                analyzer = ElliottWaveAnalyzer(data, sensitivity=sensitivity)
                signal = analyzer.generate_trading_signals()
                
                if signal is None:
                    st.warning("⚠️ Not enough data to identify Elliott Wave patterns. Try increasing the time period or adjusting sensitivity.")
                    return
                
                # Display current signal
                st.markdown('<div class="sub-header">📡 Live Trading Signal</div>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${signal['current_price']:.2f}")
                with col2:
                    st.metric("Wave Position", signal['wave_position'])
                with col3:
                    st.metric("Confidence", f"{signal['confidence']}%")
                with col4:
                    st.metric("R:R Ratio", f"{signal['risk_reward']:.2f}")
                
                # Signal card
                signal_class = f"{signal['action'].lower()}-signal" if signal['action'] != 'WAIT' else "hold-signal"
                st.markdown(f"""
                <div class="trade-signal {signal_class}">
                    <h2>🎯 {signal['action']} Signal</h2>
                    <p style="font-size: 1rem; margin-top: 0.5rem;">
                        Direction: {signal['direction'].upper()} | 
                        Candles Since Pivot: {signal['candles_since_pivot']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Trading levels
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("📍 Entry Level", f"${signal['entry']:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("🛑 Stop Loss", f"${signal['stop_loss']:.2f}")
                    risk = abs(signal['entry'] - signal['stop_loss']) / signal['entry'] * 100
                    st.caption(f"Risk: {risk:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("🎯 Target 1", f"${signal['target1']:.2f}")
                    reward1 = abs(signal['target1'] - signal['entry']) / signal['entry'] * 100
                    st.caption(f"Reward: {reward1:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("🎯 Target 2", f"${signal['target2']:.2f}")
                    reward2 = abs(signal['target2'] - signal['entry']) / signal['entry'] * 100
                    st.caption(f"Reward: {reward2:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Chart
                st.markdown('<div class="sub-header">📈 Elliott Wave Chart</div>', unsafe_allow_html=True)
                fig = plot_elliott_waves(data, analyzer, signal)
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional info
                with st.expander("📋 Trading Plan"):
                    st.markdown(f"""
                    ### {signal['action']} Setup - {signal['direction'].title()}
                    
                    **Entry Strategy:**
                    - Entry Price: ${signal['entry']:.2f}
                    - Wait for price to reach entry level with confirmation
                    - Use limit order at entry price
                    
                    **Risk Management:**
                    - Stop Loss: ${signal['stop_loss']:.2f}
                    - Position Size: Risk 1-2% of capital
                    - Never move stop loss against the trade
                    
                    **Profit Targets:**
                    - Target 1 (50% position): ${signal['target1']:.2f} ({reward1:.1f}% gain)
                    - Target 2 (30% position): ${signal['target2']:.2f} ({reward2:.2f}% gain)
                    - Target 3 (20% position): ${signal['target3']:.2f}
                    
                    **Exit Strategy:**
                    - Move stop to breakeven after Target 1 hit
                    - Trail stop using Wave 4 support/resistance
                    - Exit if wave structure invalidated
                    """)
                
                # Backtesting section
                st.markdown('<div class="sub-header">📊 Strategy Backtest</div>', unsafe_allow_html=True)
                
                with st.spinner("Running backtest..."):
                    # Generate historical signals (simplified for demo)
                    signals_history = []
                    lookback_periods = range(50, len(data), 20)
                    
                    for i in lookback_periods:
                        historical_data = data.iloc[:i]
                        hist_analyzer = ElliottWaveAnalyzer(historical_data, sensitivity=sensitivity)
                        hist_signal = hist_analyzer.generate_trading_signals()
                        if hist_signal:
                            signals_history.append(hist_signal)
                    
                    if signals_history:
                        backtest_results = backtest_strategy(data, signals_history)
                        
                        if backtest_results:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Trades", backtest_results['total_trades'])
                            with col2:
                                st.metric("Win Rate", f"{backtest_results['win_rate']:.1f}%")
                            with col3:
                                st.metric("Avg Return/Trade", f"{backtest_results['avg_return']:.2f}%")
                            with col4:
                                st.metric("Total Return", f"{backtest_results['total_return']:.2f}%")
                            
                            # Trade history
                            with st.expander("📜 Trade History"):
                                st.dataframe(
                                    backtest_results['trades'].style.format({
                                        'profit': '${:.2f}',
                                        'return': '{:.2f}%'
                                    }),
                                    use_container_width=True
                                )
                        else:
                            st.info("Not enough completed trades for backtest statistics")
                    else:
                        st.info("Insufficient historical signals for backtesting")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please check your symbol and try again. For crypto use format: BTC-USD")

if __name__ == "__main__":
    main()
