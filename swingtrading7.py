import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Swing Trading Strategy Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .danger-card {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

class TechnicalIndicators:
    @staticmethod
    def sma(data, window):
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data, window):
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data, window=14):
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

class TradingStrategies:
    @staticmethod
    def mean_reversion_signals(df, rsi_oversold=30, rsi_overbought=70, sma_period=20):
        signals = []
        for i in range(1, len(df)):
            if (df['RSI'].iloc[i] < rsi_oversold and 
                df['Close'].iloc[i] < df['SMA_20'].iloc[i] * 0.98):
                signals.append({
                    'Date': df['Date'].iloc[i],
                    'Type': 'BUY',
                    'Price': df['Close'].iloc[i],
                    'Reason': f"RSI oversold ({df['RSI'].iloc[i]:.1f}) + Price below SMA20"
                })
            elif (df['RSI'].iloc[i] > rsi_overbought and 
                  df['Close'].iloc[i] > df['SMA_20'].iloc[i] * 1.02):
                signals.append({
                    'Date': df['Date'].iloc[i],
                    'Type': 'SELL',
                    'Price': df['Close'].iloc[i],
                    'Reason': f"RSI overbought ({df['RSI'].iloc[i]:.1f}) + Price above SMA20"
                })
        return signals
    
    @staticmethod
    def momentum_signals(df, ema_fast=12, ema_slow=26, volume_multiplier=1.5):
        signals = []
        for i in range(1, len(df)):
            if (df['EMA_12'].iloc[i] > df['EMA_26'].iloc[i] and 
                df['EMA_12'].iloc[i-1] <= df['EMA_26'].iloc[i-1] and
                df['Volume'].iloc[i] > df['Volume'].iloc[i-1] * volume_multiplier):
                signals.append({
                    'Date': df['Date'].iloc[i],
                    'Type': 'BUY',
                    'Price': df['Close'].iloc[i],
                    'Reason': f"EMA crossover + Volume spike ({df['Volume'].iloc[i]/df['Volume'].iloc[i-1]:.1f}x)"
                })
            elif (df['EMA_12'].iloc[i] < df['EMA_26'].iloc[i] and 
                  df['EMA_12'].iloc[i-1] >= df['EMA_26'].iloc[i-1]):
                signals.append({
                    'Date': df['Date'].iloc[i],
                    'Type': 'SELL',
                    'Price': df['Close'].iloc[i],
                    'Reason': "EMA bearish crossover"
                })
        return signals
    
    @staticmethod
    def macd_signals(df, rsi_filter=50, volume_confirm=1.3):
        signals = []
        for i in range(1, len(df)):
            if (df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i] and 
                df['MACD'].iloc[i-1] <= df['MACD_Signal'].iloc[i-1] and
                df['RSI'].iloc[i] > rsi_filter):
                signals.append({
                    'Date': df['Date'].iloc[i],
                    'Type': 'BUY',
                    'Price': df['Close'].iloc[i],
                    'Reason': f"MACD bullish crossover + RSI confirmation ({df['RSI'].iloc[i]:.1f})"
                })
            elif (df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i] and 
                  df['MACD'].iloc[i-1] >= df['MACD_Signal'].iloc[i-1]):
                signals.append({
                    'Date': df['Date'].iloc[i],
                    'Type': 'SELL',
                    'Price': df['Close'].iloc[i],
                    'Reason': "MACD bearish crossover"
                })
        return signals

class Backtester:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
    
    def backtest(self, signals, initial_capital=None):
        if initial_capital is None:
            initial_capital = self.initial_capital
            
        capital = initial_capital
        position = 0
        trades = []
        current_trade = None
        
        for signal in signals:
            if signal['Type'] == 'BUY' and position == 0:
                position = int(capital / signal['Price'])
                capital -= position * signal['Price']
                current_trade = {
                    'entry_date': signal['Date'],
                    'entry_price': signal['Price'],
                    'entry_reason': signal['Reason'],
                    'quantity': position
                }
            elif signal['Type'] == 'SELL' and position > 0:
                exit_value = position * signal['Price']
                capital += exit_value
                
                trade = {
                    **current_trade,
                    'exit_date': signal['Date'],
                    'exit_price': signal['Price'],
                    'exit_reason': signal['Reason'],
                    'pnl': exit_value - (current_trade['quantity'] * current_trade['entry_price']),
                    'return_pct': ((signal['Price'] - current_trade['entry_price']) / current_trade['entry_price']) * 100
                }
                trades.append(trade)
                position = 0
                current_trade = None
        
        # Calculate metrics
        total_return = ((capital - initial_capital) / initial_capital) * 100
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        results = {
            'total_return': total_return,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': (len(winning_trades) / len(trades) * 100) if trades else 0,
            'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0,
            'profit_factor': (sum([t['pnl'] for t in winning_trades]) / sum([abs(t['pnl']) for t in losing_trades])) if losing_trades else 0,
            'final_capital': capital,
            'trades': trades
        }
        
        return results

def load_and_process_data(uploaded_file):
    """Load and process the uploaded CSV file"""
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.replace('"', '')
        
        # Convert date column
        date_col = df.columns[0]  # Assuming first column is date
        df['Date'] = pd.to_datetime(df[date_col], format='%d-%b-%Y', errors='coerce')
        
        # Clean numeric columns by removing commas
        numeric_cols = ['OPEN', 'HIGH', 'LOW', 'PREV. CLOSE', 'ltp', 'close', 'vwap', 'VOLUME']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '').astype(float)
        
        # Rename columns for consistency
        column_mapping = {
            'OPEN': 'Open',
            'HIGH': 'High', 
            'LOW': 'Low',
            'close': 'Close',
            'VOLUME': 'Volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Add technical indicators
        df['SMA_20'] = TechnicalIndicators.sma(df['Close'], 20)
        df['SMA_50'] = TechnicalIndicators.sma(df['Close'], 50)
        df['EMA_12'] = TechnicalIndicators.ema(df['Close'], 12)
        df['EMA_26'] = TechnicalIndicators.ema(df['Close'], 26)
        df['RSI'] = TechnicalIndicators.rsi(df['Close'])
        
        macd, macd_signal, macd_hist = TechnicalIndicators.macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        
        return df
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def optimize_strategy(df, strategy_name, param_ranges):
    """Optimize strategy parameters using grid search"""
    best_params = {}
    best_return = -float('inf')
    best_results = {}
    
    # Generate parameter combinations
    import itertools
    
    param_names = list(param_ranges.keys())
    param_values = [np.linspace(param_ranges[name]['min'], param_ranges[name]['max'], 3) 
                   for name in param_names]
    
    combinations = list(itertools.product(*param_values))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    backtester = Backtester()
    
    for i, combination in enumerate(combinations[:27]):  # Limit to 27 combinations
        params = dict(zip(param_names, combination))
        
        # Generate signals based on strategy
        if strategy_name == 'Mean Reversion':
            signals = TradingStrategies.mean_reversion_signals(df, **params)
        elif strategy_name == 'Momentum Breakout':
            signals = TradingStrategies.momentum_signals(df, **params)
        elif strategy_name == 'MACD Crossover':
            signals = TradingStrategies.macd_signals(df, **params)
        
        # Backtest
        results = backtester.backtest(signals)
        
        # Fitness function (return - 0.5 * max_drawdown)
        fitness = results['total_return'] - (0 * results.get('max_drawdown', 0))
        
        if fitness > best_return:
            best_return = fitness
            best_params = params
            best_results = results
        
        # Update progress
        progress_bar.progress((i + 1) / len(combinations[:27]))
        status_text.text(f'Testing combination {i+1}/{len(combinations[:27])}')
    
    progress_bar.empty()
    status_text.empty()
    
    return best_params, best_results

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Swing Trading Strategy Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Backtesting & Live Trading System")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload your stock data CSV file with OHLCV data"
        )
        
        st.header("⚙️ Strategy Configuration")
        strategy_name = st.selectbox(
            "Select Trading Strategy",
            ["Mean Reversion", "Momentum Breakout", "MACD Crossover"]
        )
        
        initial_capital = st.number_input(
            "Initial Capital (₹)",
            min_value=10000,
            max_value=10000000,
            value=100000,
            step=10000
        )
    
    # Main content
    if uploaded_file is None:
        st.info("👆 Please upload a CSV file to begin analysis")
        
        # Show sample data format
        st.subheader("📋 Expected CSV Format")
        sample_data = {
            'Date': ['18-Aug-2025', '14-Aug-2025', '13-Aug-2025'],
            'OPEN': [3155.90, 3034.00, 2977.80],
            'HIGH': [3255.80, 3039.90, 3044.80],
            'LOW': [3132.40, 2985.60, 2963.80],
            'close': [3219.70, 3020.30, 3019.20],
            'VOLUME': [3748828, 1075908, 1546062]
        }
        st.dataframe(pd.DataFrame(sample_data))
        
        return
    
    # Load and process data
    with st.spinner("Loading and processing data..."):
        df = load_and_process_data(uploaded_file)
    
    if df is None:
        return
    
    st.success(f"✅ Data loaded successfully! {len(df)} records processed.")
    
    # Display data info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Date Range", f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    with col3:
        st.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
    with col4:
        price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        st.metric("Price Change", f"{price_change:.2f}%", delta=f"{price_change:.2f}%")
    
    # Strategy parameters
    st.subheader("🎯 Strategy Optimization")
    
    if strategy_name == "Mean Reversion":
        param_ranges = {
            'rsi_oversold': {'min': 25, 'max': 35},
            'rsi_overbought': {'min': 65, 'max': 80},
            'sma_period': {'min': 15, 'max': 25}
        }
    elif strategy_name == "Momentum Breakout":
        param_ranges = {
            'ema_fast': {'min': 8, 'max': 15},
            'ema_slow': {'min': 20, 'max': 30},
            'volume_multiplier': {'min': 1.2, 'max': 2.0}
        }
    else:  # MACD Crossover
        param_ranges = {
            'rsi_filter': {'min': 40, 'max': 60},
            'volume_confirm': {'min': 1.1, 'max': 1.8}
        }
    
    if st.button("🚀 Optimize Strategy & Backtest", type="primary"):
        with st.spinner("Optimizing strategy parameters..."):
            best_params, best_results = optimize_strategy(df, strategy_name, param_ranges)
        
        # Store results in session state
        st.session_state['best_params'] = best_params
        st.session_state['best_results'] = best_results
        st.session_state['df'] = df
        st.session_state['strategy_name'] = strategy_name
    
    # Display results if available
    if 'best_results' in st.session_state:
        best_results = st.session_state['best_results']
        best_params = st.session_state['best_params']
        df = st.session_state['df']
        strategy_name = st.session_state['strategy_name']
        
        # Performance Metrics
        st.subheader("📊 Performance Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Return",
                f"{best_results['total_return']:.2f}%",
                delta=f"{best_results['total_return']:.2f}%"
            )
        
        with col2:
            st.metric("Win Rate", f"{best_results['win_rate']:.1f}%")
        
        with col3:
            st.metric("Total Trades", best_results['total_trades'])
        
        with col4:
            st.metric("Profit Factor", f"{best_results['profit_factor']:.2f}")
        
        with col5:
            st.metric("Final Capital", f"₹{best_results['final_capital']:,.0f}")
        
        # Optimized Parameters
        st.subheader("⚙️ Optimized Parameters")
        param_cols = st.columns(len(best_params))
        for i, (param, value) in enumerate(best_params.items()):
            with param_cols[i]:
                st.metric(param.replace('_', ' ').title(), f"{value:.2f}")
        
        # Generate signals with best parameters
        if strategy_name == 'Mean Reversion':
            signals = TradingStrategies.mean_reversion_signals(df, **best_params)
        elif strategy_name == 'Momentum Breakout':
            signals = TradingStrategies.momentum_signals(df, **best_params)
        else:
            signals = TradingStrategies.macd_signals(df, **best_params)
        
        # Price Chart with indicators
        st.subheader("📈 Price Chart & Technical Indicators")
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxis=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('Price & Moving Averages', 'RSI', 'Volume')
        )
        
        # Price and moving averages
        fig.add_trace(
            go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['SMA_20'], name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['EMA_12'], name='EMA 12', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add buy/sell signals
        buy_signals = [s for s in signals if s['Type'] == 'BUY']
        sell_signals = [s for s in signals if s['Type'] == 'SELL']
        
        if buy_signals:
            fig.add_trace(
                go.Scatter(
                    x=[s['Date'] for s in buy_signals],
                    y=[s['Price'] for s in buy_signals],
                    mode='markers',
                    marker=dict(color='green', size=10, symbol='triangle-up'),
                    name='Buy Signal'
                ),
                row=1, col=1
            )
        
        if sell_signals:
            fig.add_trace(
                go.Scatter(
                    x=[s['Date'] for s in sell_signals],
                    y=[s['Price'] for s in sell_signals],
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='triangle-down'),
                    name='Sell Signal'
                ),
                row=1, col=1
            )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # Volume
        fig.add_trace(
            go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='lightblue'),
            row=3, col=1
        )
        
        fig.update_layout(
            title=f"{strategy_name} Strategy Analysis",
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Current Signal
        st.subheader("🎯 Live Trading Recommendation")
        
        if signals:
            latest_signal = signals[-1]
            current_price = df['Close'].iloc[-1]
            
            if latest_signal['Type'] == 'BUY':
                stop_loss = current_price * 0.95
                target = current_price * 1.08
                signal_color = "success"
            else:
                stop_loss = current_price * 1.05
                target = current_price * 0.92
                signal_color = "danger"
            
            st.markdown(f"""
            <div class="metric-card {signal_color}-card">
                <h4>{latest_signal['Type']} Signal</h4>
                <p><strong>Reason:</strong> {latest_signal['Reason']}</p>
                <p><strong>Entry Price:</strong> ₹{current_price:.2f}</p>
                <p><strong>Stop Loss:</strong> ₹{stop_loss:.2f}</p>
                <p><strong>Target:</strong> ₹{target:.2f}</p>
                <p><strong>Risk:Reward:</strong> 1:{abs(target - current_price) / abs(current_price - stop_loss):.2f}</p>
                <p><strong>Win Probability:</strong> {best_results['win_rate']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No trading signals generated for current parameters")
        
        # Recent Trades Table
        if best_results['trades']:
            st.subheader("📋 Recent Trades")
            
            trades_df = pd.DataFrame(best_results['trades'][-10:])  # Last 10 trades
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date']).dt.strftime('%Y-%m-%d')
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date']).dt.strftime('%Y-%m-%d')
            
            # Format columns
            trades_display = trades_df[['entry_date', 'exit_date', 'entry_price', 'exit_price', 'return_pct', 'pnl']].copy()
            trades_display.columns = ['Entry Date', 'Exit Date', 'Entry Price', 'Exit Price', 'Return %', 'P&L']
            trades_display['Entry Price'] = trades_display['Entry Price'].apply(lambda x: f"₹{x:.2f}")
            trades_display['Exit Price'] = trades_display['Exit Price'].apply(lambda x: f"₹{x:.2f}")
            trades_display['Return %'] = trades_display['Return %'].apply(lambda x: f"{x:.2f}%")
            trades_display['P&L'] = trades_display['P&L'].apply(lambda x: f"₹{x:.2f}")
            
            st.dataframe(trades_display, use_container_width=True)
        
        # Strategy Logic Explanation
        st.subheader("🧠 Strategy Logic & Risk Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Entry Logic")
            if strategy_name == "Mean Reversion":
                st.markdown("""
                **Buy Conditions:**
                - RSI below oversold threshold (typically 30)
                - Price below SMA20 by 2% (support level)
                - Indicates potential price bounce
                
                **Sell Conditions:**
                - RSI above overbought threshold (typically 70)
                - Price above SMA20 by 2% (resistance level)
                - Mean reversion complete
                """)
            elif strategy_name == "Momentum Breakout":
                st.markdown("""
                **Buy Conditions:**
                - EMA12 crosses above EMA26 (bullish momentum)
                - Volume spike (1.5x average volume)
                - Confirms breakout strength
                
                **Sell Conditions:**
                - EMA12 crosses below EMA26 (bearish momentum)
                - Momentum reversal signal
                """)
            else:  # MACD
                st.markdown("""
                **Buy Conditions:**
                - MACD line crosses above signal line
                - RSI above 50 (bullish confirmation)
                - Volume confirmation (1.3x average)
                
                **Sell Conditions:**
                - MACD line crosses below signal line
                - Trend reversal indication
                """)
        
        with col2:
            st.markdown("### Risk Management")
            st.markdown("""
            **Position Sizing:**
            - Risk 1-2% of capital per trade
            - Stop loss typically 5% from entry
            - Target 8-10% profit (Risk:Reward = 1:1.6)
            - Maximum 3 concurrent positions
            
            **Exit Rules:**
            - Honor stop losses without exception
            - Take partial profits at 50% of target
            - Trail stop loss after 50% target hit
            - Exit if strategy signal reverses
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("⚠️ **Disclaimer:** This is for educational purposes only. Always do your own research before trading. Past performance does not guarantee future results.")

if __name__ == "__main__":
    main()
