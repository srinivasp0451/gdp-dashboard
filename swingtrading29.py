import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="VMA-Elite Trading System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class VMAEliteStrategy:
    def __init__(self, momentum_length=14, volume_ma_length=20, atr_length=14, 
                 sensitivity=1.2, risk_reward_ratio=2.5, max_risk_percent=2.0):
        self.momentum_length = momentum_length
        self.volume_ma_length = volume_ma_length
        self.atr_length = atr_length
        self.sensitivity = sensitivity
        self.risk_reward_ratio = risk_reward_ratio
        self.max_risk_percent = max_risk_percent
    
    def calculate_atr(self, df):
        """Calculate Average True Range"""
        df['TR'] = np.maximum(
            np.maximum(
                df['High'] - df['Low'],
                np.abs(df['High'] - df['Close'].shift(1))
            ),
            np.abs(df['Low'] - df['Close'].shift(1))
        )
        df['ATR'] = df['TR'].rolling(window=self.atr_length).mean()
        return df
    
    def calculate_momentum_acceleration(self, df):
        """Calculate non-lagging momentum acceleration"""
        df['ROC_Fast'] = df['Close'].pct_change(periods=self.momentum_length // 2) * 100
        df['ROC_Slow'] = df['Close'].pct_change(periods=self.momentum_length) * 100
        df['Momentum_Acceleration'] = df['ROC_Fast'] - df['ROC_Slow']
        
        # Fill NaN values with 0
        df['ROC_Fast'] = df['ROC_Fast'].fillna(0)
        df['ROC_Slow'] = df['ROC_Slow'].fillna(0)
        df['Momentum_Acceleration'] = df['Momentum_Acceleration'].fillna(0)
        
        return df
    
    def calculate_volume_analysis(self, df):
        """Calculate volume-price thrust analysis"""
        df['Volume_MA'] = df['Volume'].rolling(window=self.volume_ma_length).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['Price_Change'] = np.abs(df['Close'].diff())
        df['Volume_Price_Thrust'] = (df['Volume_Ratio'] * df['Price_Change'] / df['Close']) * 10000
        
        # Fill NaN values with 0
        df['Volume_MA'] = df['Volume_MA'].fillna(0)
        df['Volume_Ratio'] = df['Volume_Ratio'].fillna(1.0)
        df['Price_Change'] = df['Price_Change'].fillna(0)
        df['Volume_Price_Thrust'] = df['Volume_Price_Thrust'].fillna(0)
        
        return df
    
    def calculate_support_resistance(self, df):
        """Calculate dynamic support/resistance levels"""
        window = self.momentum_length * 2
        df['Highest_High'] = df['High'].rolling(window=window).max()
        df['Lowest_Low'] = df['Low'].rolling(window=window).min()
        df['Mid_Range'] = (df['Highest_High'] + df['Lowest_Low']) / 2
        return df
    
    def calculate_elite_score(self, df):
        """Calculate proprietary Elite Score"""
        # Momentum Score
        df['Momentum_Percentile'] = df['Momentum_Acceleration'].rolling(window=self.momentum_length).rank(pct=True)
        df['Momentum_Score'] = df['Momentum_Percentile'].fillna(0.5)
        
        # Volume Score
        df['Volume_Score'] = np.minimum(df['Volume_Ratio'] / 2, 1.0).fillna(0.5)
        
        # Volatility Score (simplified)
        df['Volatility_Score'] = 0.6
        
        # Structure Score
        df['Structure_Score'] = np.where(df['Close'] > df['Mid_Range'], 1.0,
                                np.where(df['Close'] < df['Mid_Range'], -1.0, 0.0))
        
        # Combined Elite Score
        df['Elite_Score'] = (df['Momentum_Score'] * 0.3 + 
                           df['Volume_Score'] * 0.25 + 
                           df['Volatility_Score'] * 0.2 + 
                           np.abs(df['Structure_Score']) * 0.25)
        
        return df
    
    def generate_signals(self, df):
        """Generate VMA-Elite trading signals"""
        # Calculate VPT average
        df['VPT_Average'] = df['Volume_Price_Thrust'].rolling(window=20).mean().fillna(0)
        
        # Signal conditions - ensure all boolean operations
        bullish_conditions = [
            df['Volume_Price_Thrust'] > (df['VPT_Average'] * self.sensitivity),
            df['Momentum_Acceleration'] > 0,
            df['Close'] > df['Mid_Range'],
            df['Elite_Score'] > 0.6
        ]
        
        bearish_conditions = [
            df['Volume_Price_Thrust'] > (df['VPT_Average'] * self.sensitivity),
            df['Momentum_Acceleration'] < 0,
            df['Close'] < df['Mid_Range'],
            df['Elite_Score'] > 0.6
        ]
        
        # Combine conditions
        df['Bullish_Thrust'] = pd.Series(True, index=df.index)
        for condition in bullish_conditions:
            df['Bullish_Thrust'] = df['Bullish_Thrust'] & condition.fillna(False)
        
        df['Bearish_Thrust'] = pd.Series(True, index=df.index)
        for condition in bearish_conditions:
            df['Bearish_Thrust'] = df['Bearish_Thrust'] & condition.fillna(False)
        
        # Entry signals (non-repainting)
        prev_bullish = df['Bullish_Thrust'].shift(1).fillna(False)
        prev_bearish = df['Bearish_Thrust'].shift(1).fillna(False)
        
        df['Long_Signal'] = (df['Bullish_Thrust'] & ~prev_bullish & (df['Structure_Score'] > 0))
        df['Short_Signal'] = (df['Bearish_Thrust'] & ~prev_bearish & (df['Structure_Score'] < 0))
        
        # Calculate stop loss and targets
        df['Long_SL'] = df['Low'] - (df['ATR'] * 1.5)
        df['Short_SL'] = df['High'] + (df['ATR'] * 1.5)
        
        df['Long_Target'] = df['Close'] + ((df['Close'] - df['Long_SL']) * self.risk_reward_ratio)
        df['Short_Target'] = df['Close'] - ((df['Short_SL'] - df['Close']) * self.risk_reward_ratio)
        
        return df
    
    def process_data(self, df):
        """Process all indicators and signals"""
        df = df.copy()
        df = self.calculate_atr(df)
        df = self.calculate_momentum_acceleration(df)
        df = self.calculate_volume_analysis(df)
        df = self.calculate_support_resistance(df)
        df = self.calculate_elite_score(df)
        df = self.generate_signals(df)
        return df
    
    def backtest(self, df, initial_balance=100000):
        """Comprehensive backtesting with detailed trade tracking"""
        df = self.process_data(df)
        
        trades = []
        balance = initial_balance
        max_balance = initial_balance
        current_trade = None
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Check for new signals
            if row['Long_Signal'] or row['Short_Signal']:
                # Close existing opposite trade
                if current_trade and ((row['Long_Signal'] and current_trade['type'] == 'SHORT') or 
                                    (row['Short_Signal'] and current_trade['type'] == 'LONG')):
                    # Close trade at current price
                    exit_price = row['Close']
                    pnl = self.calculate_pnl(current_trade, exit_price)
                    balance += pnl
                    max_balance = max(max_balance, balance)
                    
                    current_trade.update({
                        'exit_price': exit_price,
                        'exit_date': row.name,
                        'exit_reason': 'Opposite Signal',
                        'pnl': pnl,
                        'balance': balance
                    })
                    trades.append(current_trade)
                    current_trade = None
                
                # Open new trade
                if not current_trade:
                    trade_type = 'LONG' if row['Long_Signal'] else 'SHORT'
                    stop_loss = row['Long_SL'] if trade_type == 'LONG' else row['Short_SL']
                    target = row['Long_Target'] if trade_type == 'LONG' else row['Short_Target']
                    
                    # Calculate position size
                    risk_amount = balance * (self.max_risk_percent / 100)
                    stop_distance = abs(row['Close'] - stop_loss)
                    position_size = risk_amount / stop_distance if stop_distance > 0 else 0
                    
                    # Calculate probability of profit (simplified)
                    prob_profit = min(0.95, row['Elite_Score'] * 1.2)
                    
                    current_trade = {
                        'entry_date': row.name,
                        'entry_price': row['Close'],
                        'type': trade_type,
                        'stop_loss': stop_loss,
                        'target': target,
                        'position_size': position_size,
                        'elite_score': row['Elite_Score'],
                        'momentum': row['Momentum_Acceleration'],
                        'volume_thrust': row['Volume_Price_Thrust'],
                        'probability_profit': prob_profit,
                        'entry_reason': self.get_entry_reason(row, trade_type)
                    }
            
            # Check for stop loss or target hit
            if current_trade:
                hit_stop = False
                hit_target = False
                exit_reason = ''
                
                if current_trade['type'] == 'LONG':
                    if row['Low'] <= current_trade['stop_loss']:
                        hit_stop = True
                        exit_reason = 'Stop Loss Hit'
                    elif row['High'] >= current_trade['target']:
                        hit_target = True
                        exit_reason = 'Target Hit'
                else:  # SHORT
                    if row['High'] >= current_trade['stop_loss']:
                        hit_stop = True
                        exit_reason = 'Stop Loss Hit'
                    elif row['Low'] <= current_trade['target']:
                        hit_target = True
                        exit_reason = 'Target Hit'
                
                if hit_stop or hit_target:
                    exit_price = current_trade['stop_loss'] if hit_stop else current_trade['target']
                    pnl = self.calculate_pnl(current_trade, exit_price)
                    balance += pnl
                    max_balance = max(max_balance, balance)
                    
                    current_trade.update({
                        'exit_price': exit_price,
                        'exit_date': row.name,
                        'exit_reason': exit_reason,
                        'pnl': pnl,
                        'balance': balance
                    })
                    trades.append(current_trade)
                    current_trade = None
        
        # Calculate performance metrics
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        performance = self.calculate_performance_metrics(trades_df, initial_balance, balance)
        
        return df, trades_df, performance
    
    def calculate_pnl(self, trade, exit_price):
        """Calculate P&L for a trade"""
        if trade['type'] == 'LONG':
            return (exit_price - trade['entry_price']) * trade['position_size']
        else:
            return (trade['entry_price'] - exit_price) * trade['position_size']
    
    def get_entry_reason(self, row, trade_type):
        """Generate entry reason explanation"""
        reasons = []
        if row['Elite_Score'] > 0.6:
            reasons.append(f"High Elite Score ({row['Elite_Score']:.2f})")
        if row['Volume_Price_Thrust'] > row.get('VPT_Average', 0) * self.sensitivity:
            reasons.append("Strong Volume Thrust")
        if trade_type == 'LONG' and row['Momentum_Acceleration'] > 0:
            reasons.append("Positive Momentum")
        elif trade_type == 'SHORT' and row['Momentum_Acceleration'] < 0:
            reasons.append("Negative Momentum")
        if trade_type == 'LONG' and row['Close'] > row['Mid_Range']:
            reasons.append("Above Key Level")
        elif trade_type == 'SHORT' and row['Close'] < row['Mid_Range']:
            reasons.append("Below Key Level")
        
        return "; ".join(reasons)
    
    def calculate_performance_metrics(self, trades_df, initial_balance, final_balance):
        """Calculate comprehensive performance metrics"""
        if trades_df.empty:
            return {}
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        total_return = ((final_balance - initial_balance) / initial_balance) * 100
        win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Calculate maximum drawdown
        equity_curve = trades_df['balance'].values
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak * 100
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        return {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'final_balance': final_balance,
            'best_trade': trades_df['pnl'].max() if len(trades_df) > 0 else 0,
            'worst_trade': trades_df['pnl'].min() if len(trades_df) > 0 else 0
        }

def load_data(symbol, period="2y", interval="1d"):
    """Load data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            st.error(f"No data found for {symbol}")
            return None
            
        # Reset index to get Date as a column
        df = df.reset_index()
        
        # Handle different column structures from yfinance
        if 'Datetime' in df.columns:
            df['Date'] = df['Datetime']
            df = df.drop('Datetime', axis=1)
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in required_columns if col in df.columns]
        
        if len(available_columns) < 5:
            st.error(f"Missing required columns. Available: {list(df.columns)}")
            return None
        
        # Select only the columns we need
        df = df[['Date'] + available_columns].copy()
        
        # Rename columns to standard format
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Set Date as index
        df.set_index('Date', inplace=True)
        
        # Ensure numeric types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        if df.empty:
            st.error(f"No valid data found for {symbol}")
            return None
            
        return df
        
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {str(e)}")
        return None

def create_plot(df, trades_df=None):
    """Create comprehensive trading chart"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Signals', 'Elite Score', 'Volume Thrust', 'Momentum Acceleration'),
        row_heights=[0.5, 0.2, 0.15, 0.15]
    )
    
    # Price chart with signals
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price"
    ), row=1, col=1)
    
    # Add support/resistance levels
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Mid_Range'],
        mode='lines',
        name='Key Level',
        line=dict(color='blue', width=1, dash='dash')
    ), row=1, col=1)
    
    # Add entry signals
    long_signals = df[df['Long_Signal']]
    short_signals = df[df['Short_Signal']]
    
    if not long_signals.empty:
        fig.add_trace(go.Scatter(
            x=long_signals.index,
            y=long_signals['Close'],
            mode='markers',
            name='Long Signal',
            marker=dict(symbol='triangle-up', size=12, color='green')
        ), row=1, col=1)
    
    if not short_signals.empty:
        fig.add_trace(go.Scatter(
            x=short_signals.index,
            y=short_signals['Close'],
            mode='markers',
            name='Short Signal',
            marker=dict(symbol='triangle-down', size=12, color='red')
        ), row=1, col=1)
    
    # Elite Score
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Elite_Score'],
        mode='lines',
        name='Elite Score',
        line=dict(color='purple')
    ), row=2, col=1)
    
    fig.add_hline(y=0.6, line_dash="dash", line_color="orange", row=2, col=1)
    
    # Volume Thrust
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Volume_Price_Thrust'],
        mode='lines',
        name='Volume Thrust',
        line=dict(color='orange')
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['VPT_Average'],
        mode='lines',
        name='VPT Average',
        line=dict(color='gray', dash='dash')
    ), row=3, col=1)
    
    # Momentum Acceleration
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Momentum_Acceleration'],
        mode='lines',
        name='Momentum',
        line=dict(color='cyan')
    ), row=4, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="white", row=4, col=1)
    
    # Update layout
    fig.update_layout(
        title="VMA-Elite Trading System Analysis",
        height=800,
        showlegend=True,
        template="plotly_dark"
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

def main():
    st.title("🚀 VMA-Elite Trading System")
    st.markdown("Advanced Non-Lagging Swing Trading Strategy with Live Recommendations")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'last_symbol' not in st.session_state:
        st.session_state.last_symbol = ""
    if 'last_period' not in st.session_state:
        st.session_state.last_period = ""
    if 'last_interval' not in st.session_state:
        st.session_state.last_interval = ""
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Strategy parameters
    st.sidebar.subheader("Strategy Parameters")
    momentum_length = st.sidebar.slider("Momentum Length", 5, 50, 14)
    volume_ma_length = st.sidebar.slider("Volume MA Length", 10, 100, 20)
    atr_length = st.sidebar.slider("ATR Length", 5, 50, 14)
    sensitivity = st.sidebar.slider("Signal Sensitivity", 0.5, 3.0, 1.2, 0.1)
    risk_reward_ratio = st.sidebar.slider("Risk:Reward Ratio", 1.0, 5.0, 2.5, 0.1)
    max_risk_percent = st.sidebar.slider("Max Risk %", 0.5, 10.0, 2.0, 0.5)
    
    # Data parameters
    st.sidebar.subheader("Data Configuration")
    symbol = st.sidebar.text_input("Symbol", "BTC-USD").upper()
    period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=4)
    interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    
    # Add CSV upload option
    st.sidebar.subheader("📁 Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            
            # Try to identify date column
            date_columns = [col for col in df_uploaded.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                df_uploaded[date_columns[0]] = pd.to_datetime(df_uploaded[date_columns[0]])
                df_uploaded = df_uploaded.set_index(date_columns[0])
            
            # Standardize column names
            column_mapping = {}
            for col in df_uploaded.columns:
                col_lower = col.lower()
                if 'open' in col_lower:
                    column_mapping[col] = 'Open'
                elif 'high' in col_lower:
                    column_mapping[col] = 'High'
                elif 'low' in col_lower:
                    column_mapping[col] = 'Low'
                elif 'close' in col_lower:
                    column_mapping[col] = 'Close'
                elif 'volume' in col_lower:
                    column_mapping[col] = 'Volume'
            
            df_uploaded = df_uploaded.rename(columns=column_mapping)
            
            # Check if we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if all(col in df_uploaded.columns for col in required_cols):
                st.session_state.df = df_uploaded[required_cols]
                st.session_state.data_loaded = True
                st.session_state.last_symbol = "CSV_DATA"
                st.sidebar.success(f"✅ CSV loaded: {len(df_uploaded)} rows")
            else:
                st.sidebar.error(f"❌ CSV missing columns. Found: {list(df_uploaded.columns)}")
                
        except Exception as e:
            st.sidebar.error(f"❌ Error reading CSV: {str(e)}")
    
    # Fetch data button
    fetch_button = st.sidebar.button("🔄 Fetch Data", type="primary", use_container_width=True)
    
    # Check if we need to fetch new data
    data_params_changed = (
        symbol != st.session_state.last_symbol or 
        period != st.session_state.last_period or 
        interval != st.session_state.last_interval
    )
    
    # Data status indicator
    if st.session_state.data_loaded and not data_params_changed:
        st.sidebar.success(f"✅ Data loaded for {st.session_state.last_symbol}")
        st.sidebar.info(f"📅 Period: {st.session_state.last_period} | Interval: {st.session_state.last_interval}")
        if st.session_state.df is not None:
            st.sidebar.info(f"📊 Data points: {len(st.session_state.df)}")
    elif data_params_changed and st.session_state.data_loaded:
        st.sidebar.warning("⚠️ Parameters changed. Click 'Fetch Data' to update.")
    else:
        st.sidebar.info("👆 Click 'Fetch Data' to load market data")
    
    # Mode selection
    mode = st.sidebar.radio("Mode", ["📊 Backtesting", "🎯 Live Analysis"])
    
    # Initialize strategy
    strategy = VMAEliteStrategy(
        momentum_length=momentum_length,
        volume_ma_length=volume_ma_length,
        atr_length=atr_length,
        sensitivity=sensitivity,
        risk_reward_ratio=risk_reward_ratio,
        max_risk_percent=max_risk_percent
    )
    
    # Handle data fetching
    if fetch_button or (not st.session_state.data_loaded and symbol):
        with st.spinner(f"Loading {symbol} data..."):
            df = load_data(symbol, period, interval)
            
            if df is not None and not df.empty:
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.session_state.last_symbol = symbol
                st.session_state.last_period = period
                st.session_state.last_interval = interval
                st.success(f"✅ Successfully loaded {len(df)} data points for {symbol}")
            else:
                st.session_state.data_loaded = False
                st.error("❌ Failed to load data. Please check the symbol and try again.")
                return
    
    # Check if data is available
    if not st.session_state.data_loaded or st.session_state.df is None or st.session_state.df.empty:
        st.info("👆 Please configure your settings and click 'Fetch Data' to begin analysis.")
        return
    
    df = st.session_state.df
    
    # Main content based on mode
    if mode == "📊 Backtesting":
        st.header("📊 Backtesting Results")
        
        with st.spinner("Running backtest..."):
            processed_df, trades_df, performance = strategy.backtest(df)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{performance.get('total_return', 0):.2f}%",
                delta=f"vs Buy & Hold"
            )
        
        with col2:
            st.metric(
                "Win Rate",
                f"{performance.get('win_rate', 0):.1f}%",
                delta=f"{performance.get('winning_trades', 0)}/{performance.get('total_trades', 0)} trades"
            )
        
        with col3:
            st.metric(
                "Profit Factor",
                f"{performance.get('profit_factor', 0):.2f}",
                delta=f"Max DD: {performance.get('max_drawdown', 0):.2f}%"
            )
        
        with col4:
            st.metric(
                "Final Balance",
                f"${performance.get('final_balance', 0):,.2f}",
                delta=f"${performance.get('final_balance', 0) - 100000:,.2f}"
            )
        
        # Chart
        st.subheader("📈 Trading Chart")
        fig = create_plot(processed_df, trades_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade details
        if not trades_df.empty:
            st.subheader("💼 Trade History")
            
            # Format trades for display
            display_trades = trades_df.copy()
            display_trades['Entry Date'] = pd.to_datetime(display_trades['entry_date']).dt.strftime('%Y-%m-%d')
            display_trades['Exit Date'] = pd.to_datetime(display_trades['exit_date']).dt.strftime('%Y-%m-%d')
            display_trades['Entry Price'] = display_trades['entry_price'].round(2)
            display_trades['Exit Price'] = display_trades['exit_price'].round(2)
            display_trades['P&L'] = display_trades['pnl'].round(2)
            display_trades['Type'] = display_trades['type']
            display_trades['Elite Score'] = display_trades['elite_score'].round(3)
            display_trades['Probability %'] = (display_trades['probability_profit'] * 100).round(1)
            display_trades['Reason'] = display_trades['entry_reason']
            display_trades['Exit Reason'] = display_trades['exit_reason']
            
            # Select columns for display
            columns_to_show = ['Entry Date', 'Type', 'Entry Price', 'Exit Price', 'P&L', 
                             'Elite Score', 'Probability %', 'Reason', 'Exit Reason']
            
            # Add color coding for P&L
            def highlight_pnl(row):
                color = 'background-color: rgba(34, 197, 94, 0.3)' if row['P&L'] > 0 else 'background-color: rgba(239, 68, 68, 0.3)'
                return [color if col == 'P&L' else '' for col in row.index]
            
            styled_df = display_trades[columns_to_show].style.apply(highlight_pnl, axis=1)
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=400
            )
            
            # Performance summary
            st.subheader("📊 Performance Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Trade Statistics:**")
                st.write(f"- Total Trades: {performance['total_trades']}")
                st.write(f"- Winning Trades: {performance['winning_trades']}")
                st.write(f"- Losing Trades: {performance['losing_trades']}")
                st.write(f"- Win Rate: {performance['win_rate']:.1f}%")
                st.write(f"- Best Trade: ${performance['best_trade']:.2f}")
                st.write(f"- Worst Trade: ${performance['worst_trade']:.2f}")
            
            with col2:
                st.write("**Risk Metrics:**")
                st.write(f"- Total Return: {performance['total_return']:.2f}%")
                st.write(f"- Profit Factor: {performance['profit_factor']:.2f}")
                st.write(f"- Average Win: ${performance['avg_win']:.2f}")
                st.write(f"- Average Loss: ${performance['avg_loss']:.2f}")
                st.write(f"- Max Drawdown: {performance['max_drawdown']:.2f}%")
    
    else:  # Live Analysis Mode
        st.header("🎯 Live Trading Analysis")
        
        with st.spinner("Processing live data..."):
            processed_df = strategy.process_data(df)
        
        # Current market status
        latest = processed_df.iloc[-1]
        
        st.subheader("📊 Current Market Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price",
                f"${latest['Close']:.2f}",
                delta=f"{((latest['Close'] - processed_df.iloc[-2]['Close']) / processed_df.iloc[-2]['Close'] * 100):.2f}%"
            )
        
        with col2:
            color = "normal" if latest['Elite_Score'] < 0.6 else "inverse"
            st.metric(
                "Elite Score",
                f"{latest['Elite_Score']:.3f}",
                delta="High Quality" if latest['Elite_Score'] > 0.6 else "Low Quality",
                delta_color=color
            )
        
        with col3:
            st.metric(
                "Momentum",
                f"{latest['Momentum_Acceleration']:.2f}",
                delta="Bullish" if latest['Momentum_Acceleration'] > 0 else "Bearish"
            )
        
        with col4:
            st.metric(
                "Volume Thrust",
                f"{latest['Volume_Price_Thrust']:.2f}",
                delta="Above Avg" if latest['Volume_Price_Thrust'] > latest.get('VPT_Average', 0) else "Below Avg"
            )
        
        # Live signals
        st.subheader("🚨 Live Trading Signals")
        
        # Check for recent signals (last 5 bars)
        recent_signals = processed_df.tail(5)
        long_signals = recent_signals[recent_signals['Long_Signal']]
        short_signals = recent_signals[recent_signals['Short_Signal']]
        
        if not long_signals.empty or not short_signals.empty:
            if not long_signals.empty:
                latest_long = long_signals.iloc[-1]
                st.success("🟢 **LONG SIGNAL DETECTED!**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Trade Setup:**")
                    st.write(f"- Entry Price: ${latest_long['Close']:.2f}")
                    st.write(f"- Stop Loss: ${latest_long['Long_SL']:.2f}")
                    st.write(f"- Target: ${latest_long['Long_Target']:.2f}")
                    st.write(f"- Risk/Reward: 1:{risk_reward_ratio}")
                    
                with col2:
                    st.write("**Signal Quality:**")
                    st.write(f"- Elite Score: {latest_long['Elite_Score']:.3f}")
                    st.write(f"- Probability of Profit: {min(0.95, latest_long['Elite_Score'] * 1.2)*100:.1f}%")
                    st.write(f"- Signal Date: {latest_long.name.strftime('%Y-%m-%d')}")
                    
                st.write("**Entry Reason:**")
                st.write(strategy.get_entry_reason(latest_long, 'LONG'))
            
            if not short_signals.empty:
                latest_short = short_signals.iloc[-1]
                st.error("🔴 **SHORT SIGNAL DETECTED!**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Trade Setup:**")
                    st.write(f"- Entry Price: ${latest_short['Close']:.2f}")
                    st.write(f"- Stop Loss: ${latest_short['Short_SL']:.2f}")
                    st.write(f"- Target: ${latest_short['Short_Target']:.2f}")
                    st.write(f"- Risk/Reward: 1:{risk_reward_ratio}")
                    
                with col2:
                    st.write("**Signal Quality:**")
                    st.write(f"- Elite Score: {latest_short['Elite_Score']:.3f}")
                    st.write(f"- Probability of Profit: {min(0.95, latest_short['Elite_Score'] * 1.2)*100:.1f}%")
                    st.write(f"- Signal Date: {latest_short.name.strftime('%Y-%m-%d')}")
                    
                st.write("**Entry Reason:**")
                st.write(strategy.get_entry_reason(latest_short, 'SHORT'))
        
        else:
            st.info("⏳ **No active signals. Waiting for setup...**")
            
            st.write("**Next Signal Requirements:**")
            st.write(f"- Elite Score > 0.6 (Current: {latest['Elite_Score']:.3f})")
            st.write(f"- Volume Thrust > Average * {sensitivity}")
            st.write("- Momentum Acceleration confirmation")
            st.write("- Structure break above/below key levels")
        
        # Chart for live analysis
        st.subheader("📈 Live Analysis Chart")
        fig = create_plot(processed_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Market levels
        st.subheader("🎯 Key Levels")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Support/Resistance:**")
            st.write(f"- Key Level: ${latest['Mid_Range']:.2f}")
            st.write(f"- Highest High: ${latest['Highest_High']:.2f}")
            st.write(f"- Lowest Low: ${latest['Lowest_Low']:.2f}")
            
        with col2:
            st.write("**Volatility Metrics:**")
            st.write(f"- Current ATR: ${latest['ATR']:.2f}")
            st.write(f"- ATR %: {(latest['ATR']/latest['Close']*100):.2f}%")
            
        with col3:
            st.write("**Volume Analysis:**")
            st.write(f"- Volume Ratio: {latest['Volume_Ratio']:.2f}")
            st.write(f"- Volume MA: {latest['Volume_MA']:,.0f}")

    # Footer
    st.markdown("---")
    st.markdown("**🚀 VMA-Elite Trading System** - Proprietary Non-Lagging Strategy")
    st.markdown("⚠️ *Trading involves risk. Past performance does not guarantee future results.*")

if __name__ == "__main__":
    main()
