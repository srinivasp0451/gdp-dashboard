import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
from sklearn.model_selection import ParameterSampler, ParameterGrid
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Stock Trading System",
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
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

class StockAnalyzer:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.strategy_params = {}
        self.best_strategy = None
        
    def map_columns(self, df):
        """Map column names to standard format"""
        column_mapping = {}
        
        for col in df.columns:
            col_lower = col.lower()
            
            if any(substring in col_lower for substring in ['open']):
                column_mapping[col] = 'Open'
            elif any(substring in col_lower for substring in ['high']):
                column_mapping[col] = 'High'
            elif any(substring in col_lower for substring in ['low']):
                column_mapping[col] = 'Low'
            elif any(substring in col_lower for substring in ['close']):
                column_mapping[col] = 'Close'
            elif any(substring in col_lower for substring in ['volume', 'vol']):
                column_mapping[col] = 'Volume'
            elif any(substring in col_lower for substring in ['date', 'time']):
                column_mapping[col] = 'Date'
        
        df_mapped = df.rename(columns=column_mapping)
        
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']
        missing_cols = [col for col in required_cols if col not in df_mapped.columns]
        
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
            return None
            
        return df_mapped[required_cols]
    
    def preprocess_data(self, df, end_date=None):
        """Preprocess stock data"""
        try:
            # Convert date
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Handle timezone
            ist = pytz.timezone('Asia/Kolkata')
            if df['Date'].dt.tz is None:
                df['Date'] = df['Date'].dt.tz_localize(ist)
            else:
                df['Date'] = df['Date'].dt.tz_convert(ist)
            
            # Sort by date
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Filter by end date
            if end_date:
                if pd.Timestamp(end_date).tz is None:
                    end_date_tz = pd.Timestamp(end_date).tz_localize(ist)
                else:
                    end_date_tz = pd.Timestamp(end_date).tz_convert(ist)
                df = df[df['Date'] <= end_date_tz]
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            return df
            
        except Exception as e:
            st.error(f"Error preprocessing data: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # Moving Averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.DataFrame({'HL': high_low, 'HC': high_close, 'LC': low_close})
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Volume
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
    
    def generate_signals(self, df, params):
        """Generate trading signals"""
        signals = []
        
        for i in range(50, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Long conditions
            long_conditions = []
            
            # MA Crossover
            if current['SMA_10'] > current['SMA_20'] and prev['SMA_10'] <= prev['SMA_20']:
                long_conditions.append('MA_Bullish')
            
            # RSI
            if current['RSI'] < params.get('rsi_oversold', 30):
                long_conditions.append('RSI_Oversold')
            
            # MACD
            if current['MACD'] > current['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
                long_conditions.append('MACD_Bullish')
            
            # Volume
            if current['Volume_Ratio'] > params.get('volume_threshold', 1.5):
                long_conditions.append('High_Volume')
            
            # Generate long signal
            if len(long_conditions) >= params.get('min_conditions', 3):
                entry_price = current['Close']
                target1 = entry_price * (1 + params.get('target1_pct', 0.02))
                target2 = entry_price * (1 + params.get('target2_pct', 0.04))
                stop_loss = entry_price * (1 - params.get('sl_pct', 0.015))
                
                signals.append({
                    'Date': current['Date'],
                    'Type': 'LONG',
                    'Entry_Price': entry_price,
                    'Target1': target1,
                    'Target2': target2,
                    'Stop_Loss': stop_loss,
                    'Conditions': long_conditions,
                    'Risk_Reward': (target1 - entry_price) / (entry_price - stop_loss)
                })
            
            # Short conditions
            short_conditions = []
            
            # MA Crossover
            if current['SMA_10'] < current['SMA_20'] and prev['SMA_10'] >= prev['SMA_20']:
                short_conditions.append('MA_Bearish')
            
            # RSI
            if current['RSI'] > params.get('rsi_overbought', 70):
                short_conditions.append('RSI_Overbought')
            
            # MACD
            if current['MACD'] < current['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
                short_conditions.append('MACD_Bearish')
            
            # Volume
            if current['Volume_Ratio'] > params.get('volume_threshold', 1.5):
                short_conditions.append('High_Volume')
            
            # Generate short signal
            if len(short_conditions) >= params.get('min_conditions', 3):
                entry_price = current['Close']
                target1 = entry_price * (1 - params.get('target1_pct', 0.02))
                target2 = entry_price * (1 - params.get('target2_pct', 0.04))
                stop_loss = entry_price * (1 + params.get('sl_pct', 0.015))
                
                signals.append({
                    'Date': current['Date'],
                    'Type': 'SHORT',
                    'Entry_Price': entry_price,
                    'Target1': target1,
                    'Target2': target2,
                    'Stop_Loss': stop_loss,
                    'Conditions': short_conditions,
                    'Risk_Reward': (entry_price - target1) / (stop_loss - entry_price)
                })
        
        return signals
    
    def backtest_strategy(self, df, signals):
        """Backtest trading strategy"""
        trades = []
        
        for signal in signals:
            # Find future candles for exit
            future_data = df[df['Date'] > signal['Date']].head(20)
            
            if len(future_data) == 0:
                continue
                
            for _, candle in future_data.iterrows():
                exit_price = None
                exit_reason = None
                
                if signal['Type'] == 'LONG':
                    if candle['High'] >= signal['Target1']:
                        exit_price = signal['Target1']
                        exit_reason = 'Target_Hit'
                    elif candle['Low'] <= signal['Stop_Loss']:
                        exit_price = signal['Stop_Loss']
                        exit_reason = 'Stop_Loss'
                
                elif signal['Type'] == 'SHORT':
                    if candle['Low'] <= signal['Target1']:
                        exit_price = signal['Target1']
                        exit_reason = 'Target_Hit'
                    elif candle['High'] >= signal['Stop_Loss']:
                        exit_price = signal['Stop_Loss']
                        exit_reason = 'Stop_Loss'
                
                if exit_price:
                    if signal['Type'] == 'LONG':
                        pnl_pct = (exit_price - signal['Entry_Price']) / signal['Entry_Price'] * 100
                    else:
                        pnl_pct = (signal['Entry_Price'] - exit_price) / signal['Entry_Price'] * 100
                    
                    trades.append({
                        'Entry_Date': signal['Date'],
                        'Exit_Date': candle['Date'],
                        'Type': signal['Type'],
                        'Entry_Price': signal['Entry_Price'],
                        'Exit_Price': exit_price,
                        'Target1': signal['Target1'],
                        'Stop_Loss': signal['Stop_Loss'],
                        'Exit_Reason': exit_reason,
                        'PnL_Pct': pnl_pct,
                        'Conditions': signal['Conditions'],
                        'Hold_Days': (candle['Date'] - signal['Date']).days
                    })
                    break
        
        # Calculate metrics
        if trades:
            trades_df = pd.DataFrame(trades)
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['PnL_Pct'] > 0])
            win_rate = (winning_trades / total_trades) * 100
            avg_win = trades_df[trades_df['PnL_Pct'] > 0]['PnL_Pct'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['PnL_Pct'] < 0]['PnL_Pct'].mean() if total_trades - winning_trades > 0 else 0
            total_return = trades_df['PnL_Pct'].sum()
            
            buy_hold_return = ((df.iloc[-1]['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']) * 100
            
            return {
                'Total_Trades': total_trades,
                'Winning_Trades': winning_trades,
                'Losing_Trades': total_trades - winning_trades,
                'Win_Rate': win_rate,
                'Avg_Win': avg_win,
                'Avg_Loss': avg_loss,
                'Total_Return': total_return,
                'Buy_Hold_Return': buy_hold_return,
                'Avg_Hold_Days': trades_df['Hold_Days'].mean(),
                'Trades_Detail': trades_df
            }
        else:
            return {
                'Total_Trades': 0,
                'Win_Rate': 0,
                'Total_Return': 0,
                'Buy_Hold_Return': ((df.iloc[-1]['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']) * 100,
                'Trades_Detail': pd.DataFrame()
            }
    
    def optimize_strategy(self, df, search_type='random', n_iter=50):
        """Optimize strategy parameters"""
        param_space = {
            'rsi_oversold': [20, 25, 30, 35],
            'rsi_overbought': [65, 70, 75, 80],
            'volume_threshold': [1.2, 1.5, 2.0],
            'target1_pct': [0.015, 0.02, 0.025, 0.03],
            'target2_pct': [0.03, 0.04, 0.05],
            'sl_pct': [0.01, 0.015, 0.02],
            'min_conditions': [2, 3, 4]
        }
        
        best_performance = {'Win_Rate': 0, 'Total_Return': -float('inf')}
        best_params = None
        
        if search_type == 'grid':
            param_combinations = list(ParameterGrid(param_space))
        else:
            param_combinations = list(ParameterSampler(param_space, n_iter=n_iter))
        
        progress_bar = st.progress(0)
        
        for i, params in enumerate(param_combinations[:n_iter]):
            progress = (i + 1) / min(n_iter, len(param_combinations))
            progress_bar.progress(progress)
            
            signals = self.generate_signals(df, params)
            if len(signals) > 3:
                performance = self.backtest_strategy(df, signals)
                
                if (performance['Win_Rate'] > best_performance['Win_Rate'] or 
                    (performance['Win_Rate'] >= best_performance['Win_Rate'] and 
                     performance['Total_Return'] > best_performance['Total_Return'])):
                    best_performance = performance
                    best_params = params
        
        progress_bar.empty()
        return best_params, best_performance
    
    def get_live_recommendation(self, df, params):
        """Get live recommendation"""
        if len(df) < 50:
            return None
        
        signals = self.generate_signals(df, params)
        if signals:
            latest = signals[-1]
            latest['Probability'] = min(95, 60 + len(latest['Conditions']) * 8)
            return latest
        return None

def main():
    st.markdown('<h1 class="main-header">📈 Advanced Stock Trading System</h1>', unsafe_allow_html=True)
    
    analyzer = StockAnalyzer()
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded!")
        
        # Map columns
        mapped_df = analyzer.map_columns(df)
        
        if mapped_df is not None:
            # Date selection
            mapped_df['Date'] = pd.to_datetime(mapped_df['Date'])
            min_date = mapped_df['Date'].min().date()
            max_date = mapped_df['Date'].max().date()
            
            end_date = st.sidebar.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )
            
            trade_side = st.sidebar.selectbox("Trade Side", ["Both", "Long Only", "Short Only"])
            search_type = st.sidebar.selectbox("Optimization", ["Random Search", "Grid Search"])
            n_points = st.sidebar.slider("Optimization Points", 20, 100, 50)
            
            if st.sidebar.button("Run Analysis"):
                # Process data
                processed_df = analyzer.preprocess_data(mapped_df, end_date)
                
                if processed_df is not None and len(processed_df) > 100:
                    # Tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Strategy Optimization", "Backtest Results", "Live Recommendation"])
                    
                    with tab1:
                        st.header("Data Overview")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("First 5 Rows")
                            st.dataframe(processed_df.head())
                        with col2:
                            st.subheader("Last 5 Rows")
                            st.dataframe(processed_df.tail())
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Min Date", processed_df['Date'].min().strftime('%Y-%m-%d'))
                        with col2:
                            st.metric("Max Date", processed_df['Date'].max().strftime('%Y-%m-%d'))
                        with col3:
                            st.metric("Min Price", f"₹{processed_df['Close'].min():.2f}")
                        with col4:
                            st.metric("Max Price", f"₹{processed_df['Close'].max():.2f}")
                        
                        # Chart
                        st.subheader("Price Chart")
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=processed_df['Date'],
                            open=processed_df['Open'],
                            high=processed_df['High'],
                            low=processed_df['Low'],
                            close=processed_df['Close']
                        ))
                        fig.update_layout(title="Stock Price", height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Summary
                        total_return = ((processed_df.iloc[-1]['Close'] - processed_df.iloc[0]['Close']) / processed_df.iloc[0]['Close']) * 100
                        volatility = processed_df['Close'].pct_change().std() * np.sqrt(252) * 100
                        current_rsi = processed_df.iloc[-1]['RSI']
                        
                        summary = f"""
                        **Stock Analysis Summary:**
                        
                        The stock has shown a {total_return:.1f}% return over the period with {volatility:.1f}% annualized volatility. 
                        Current RSI is at {current_rsi:.1f}, indicating {'oversold' if current_rsi < 30 else 'overbought' if current_rsi > 70 else 'neutral'} conditions.
                        The stock is currently trading {'above' if processed_df.iloc[-1]['Close'] > processed_df.iloc[-1]['SMA_20'] else 'below'} its 20-day moving average.
                        """
                        
                        st.markdown(f'<div class="success-card">{summary}</div>', unsafe_allow_html=True)
                    
                    with tab2:
                        st.header("Strategy Optimization")
                        
                        search_method = 'random' if search_type == 'Random Search' else 'grid'
                        
                        with st.spinner("Optimizing strategy..."):
                            best_params, best_performance = analyzer.optimize_strategy(
                                processed_df, search_method, n_points
                            )
                        
                        if best_params:
                            st.success("Optimization Complete!")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Win Rate", f"{best_performance['Win_Rate']:.1f}%")
                            with col2:
                                st.metric("Total Return", f"{best_performance['Total_Return']:.1f}%")
                            with col3:
                                st.metric("Buy & Hold", f"{best_performance['Buy_Hold_Return']:.1f}%")
                            with col4:
                                st.metric("Total Trades", best_performance['Total_Trades'])
                            
                            st.subheader("Best Parameters")
                            st.write(f"**RSI Levels:** {best_params['rsi_oversold']}-{best_params['rsi_overbought']}")
                            st.write(f"**Targets:** {best_params['target1_pct']*100:.1f}%, {best_params['target2_pct']*100:.1f}%")
                            st.write(f"**Stop Loss:** {best_params['sl_pct']*100:.1f}%")
                            st.write(f"**Volume Threshold:** {best_params['volume_threshold']}")
                            st.write(f"**Min Conditions:** {best_params['min_conditions']}")
                            
                            analyzer.best_strategy = best_params
                    
                    with tab3:
                        st.header("Backtest Results")
                        
                        if analyzer.best_strategy:
                            signals = analyzer.generate_signals(processed_df, analyzer.best_strategy)
                            
                            # Filter by trade side
                            if trade_side == "Long Only":
                                signals = [s for s in signals if s['Type'] == 'LONG']
                            elif trade_side == "Short Only":
                                signals = [s for s in signals if s['Type'] == 'SHORT']
                            
                            backtest_results = analyzer.backtest_strategy(processed_df, signals)
                            
                            if backtest_results['Total_Trades'] > 0:
                                col1, col2, col3, col4, col5 = st.columns(5)
                                with col1:
                                    st.metric("Total Trades", backtest_results['Total_Trades'])
                                with col2:
                                    st.metric("Win Rate", f"{backtest_results['Win_Rate']:.1f}%")
                                with col3:
                                    st.metric("Avg Win", f"{backtest_results['Avg_Win']:.1f}%")
                                with col4:
                                    st.metric("Avg Loss", f"{backtest_results['Avg_Loss']:.1f}%")
                                with col5:
                                    st.metric("Total Return", f"{backtest_results['Total_Return']:.1f}%")
                                
                                # Trade details
                                st.subheader("Trade Details")
                                trades_df = backtest_results['Trades_Detail']
                                
                                if not trades_df.empty:
                                    display_trades = trades_df.copy()
                                    display_trades['Entry_Date'] = display_trades['Entry_Date'].dt.strftime('%Y-%m-%d')
                                    display_trades['Exit_Date'] = display_trades['Exit_Date'].dt.strftime('%Y-%m-%d')
                                    
                                    st.dataframe(display_trades[[
                                        'Entry_Date', 'Exit_Date', 'Type', 'Entry_Price', 
                                        'Exit_Price', 'Exit_Reason', 'PnL_Pct', 'Hold_Days'
                                    ]], use_container_width=True)
                                
                                # Summary
                                summary = f"""
                                **Backtest Summary:**
                                
                                Strategy generated {backtest_results['Total_Trades']} trades with {backtest_results['Win_Rate']:.1f}% win rate.
                                Total return: {backtest_results['Total_Return']:.1f}% vs Buy & Hold: {backtest_results['Buy_Hold_Return']:.1f}%.
                                Average winning trade: {backtest_results['Avg_Win']:.1f}%, Average losing trade: {backtest_results['Avg_Loss']:.1f}%.
                                Average holding period: {backtest_results['Avg_Hold_Days']:.1f} days.
                                """
                                
                                st.markdown(f'<div class="success-card">{summary}</div>', unsafe_allow_html=True)
                    
                    with tab4:
                        st.header("Live Recommendation")
                        
                        if analyzer.best_strategy:
                            live_rec = analyzer.get_live_recommendation(processed_df, analyzer.best_strategy)
                            
                            if live_rec:
                                if ((trade_side == "Long Only" and live_rec['Type'] == 'LONG') or
                                    (trade_side == "Short Only" and live_rec['Type'] == 'SHORT') or
                                    trade_side == "Both"):
                                    
                                    signal_type = "🟢 LONG" if live_rec['Type'] == 'LONG' else "🔴 SHORT"
                                    
                                    st.markdown(f"""
                                    <div class="success-card">
                                        <h3>{signal_type} SIGNAL</h3>
                                        <p><strong>Entry:</strong> ₹{live_rec['Entry_Price']:.2f}</p>
                                        <p><strong>Target 1:</strong> ₹{live_rec['Target1']:.2f}</p>
                                        <p><strong>Target 2:</strong> ₹{live_rec['Target2']:.2f}</p>
                                        <p><strong>Stop Loss:</strong> ₹{live_rec['Stop_Loss']:.2f}</p>
                                        <p><strong>Risk:Reward:</strong> {live_rec['Risk_Reward']:.2f}:1</p>
                                        <p><strong>Probability:</strong> {live_rec['Probability']:.0f}%</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.subheader("Signal Conditions")
                                    for condition in live_rec['Conditions']:
                                        st.write(f"• {condition}")
                                    
                                    # Technical snapshot
                                    latest = processed_df.iloc[-1]
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("Current Price", f"₹{latest['Close']:.2f}")
                                    with col2:
                                        st.metric("RSI", f"{latest['RSI']:.1f}")
                                    with col3:
                                        st.metric("Volume Ratio", f"{latest['Volume_Ratio']:.2f}")
                                    with col4:
                                        st.metric("SMA 20", f"₹{latest['SMA_20']:.2f}")
                                    
                                    # Action plan
                                    if live_rec['Type'] == 'LONG':
                                        profit1 = ((live_rec['Target1'] - live_rec['Entry_Price']) / live_rec['Entry_Price']) * 100
                                        loss = ((live_rec['Entry_Price'] - live_rec['Stop_Loss']) / live_rec['Entry_Price']) * 100
                                    else:
                                        profit1 = ((live_rec['Entry_Price'] - live_rec['Target1']) / live_rec['Entry_Price']) * 100
                                        loss = ((live_rec['Stop_Loss'] - live_rec['Entry_Price']) / live_rec['Entry_Price']) * 100
                                    
                                    action_plan = f"""
                                    **Action Plan:**
                                    1. Enter {live_rec['Type']} at ₹{live_rec['Entry_Price']:.2f}
                                    2. Set stop loss at ₹{live_rec['Stop_Loss']:.2f} ({loss:.2f}% risk)
                                    3. Target 1 at ₹{live_rec['Target1']:.2f} ({profit1:.2f}% profit)
                                    4. Risk only 1-2% of capital per trade
                                    5. Exit 50% at target 1, trail remaining position
                                    """
                                    
                                    st.markdown(f'<div class="metric-card">{action_plan}</div>', unsafe_allow_html=True)
                                
                                else:
                                    st.info(f"Current signal is {live_rec['Type']} but you selected {trade_side}")
                            else:
                                st.info("No trading signals at current market conditions")
                        else:
                            st.warning("Run optimization first")
                
                else:
                    st.error("Insufficient data. Need at least 100 candles.")
    
    else:
        st.markdown("""
        ## Welcome to Advanced Stock Trading System
        
        ### Features:
        - **Smart Column Mapping**: Auto-detects OHLCV columns
        - **Technical Analysis**: RSI, MACD, Moving Averages, Bollinger Bands
        - **Strategy Optimization**: Find best parameters automatically  
        - **Backtesting**: Complete performance analysis
        - **Live Recommendations**: Real-time trading signals
        - **Risk Management**: Built-in stop losses and targets
        
        ### Data Requirements:
        Your CSV should have columns containing:
        - Date/Time data
        - Open, High, Low, Close prices
        - Volume data
        
        ### Instructions:
        1. Upload your CSV file
        2. Select end date for backtesting
        3. Choose trade side (Long/Short/Both)
        4. Run analysis to get optimized strategy
        5. Get live trading recommendations
        
        **Important:** This is for educational purposes only. Always do your own research before making investment decisions.
        """)
        
        # Sample data format
        st.subheader("Sample Data Format")
        sample_data = {
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Open': [100.0, 102.0, 101.5],
            'High': [103.0, 105.0, 104.0], 
            'Low': [99.0, 101.0, 100.5],
            'Close': [102.0, 103.5, 103.0],
            'Volume': [1000000, 1200000, 950000]
        }
        st.dataframe(pd.DataFrame(sample_data))

if __name__ == "__main__":
    main()
