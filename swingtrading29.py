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
                 sensitivity=1.2, risk_reward_ratio=2.5, stop_loss_pct=2.0, 
                 target_pct=5.0, use_volume=True):
        self.momentum_length = momentum_length
        self.volume_ma_length = volume_ma_length
        self.atr_length = atr_length
        self.sensitivity = sensitivity
        self.risk_reward_ratio = risk_reward_ratio
        self.stop_loss_pct = stop_loss_pct
        self.target_pct = target_pct
        self.use_volume = use_volume
    
    def calculate_atr(self, df):
        """Calculate Average True Range"""
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = np.abs(df['High'] - df['Close'].shift(1))
        df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift(1))
        
        df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=self.atr_length).mean().fillna(df['TR'].mean())
        
        # Clean up temporary columns
        df.drop(['High_Low', 'High_Close', 'Low_Close', 'TR'], axis=1, inplace=True)
        return df
    
    def calculate_momentum_acceleration(self, df):
        """Calculate non-lagging momentum acceleration"""
        df['ROC_Fast'] = df['Close'].pct_change(periods=self.momentum_length // 2) * 100
        df['ROC_Slow'] = df['Close'].pct_change(periods=self.momentum_length) * 100
        df['Momentum_Acceleration'] = (df['ROC_Fast'] - df['ROC_Slow']).fillna(0)
        
        # Calculate momentum strength
        df['Momentum_Strength'] = np.abs(df['Momentum_Acceleration'])
        df['Momentum_Direction'] = np.sign(df['Momentum_Acceleration'])
        
        return df
    
    def calculate_volume_analysis(self, df):
        """Calculate volume-price thrust analysis (if volume available)"""
        if self.use_volume and 'Volume' in df.columns and df['Volume'].sum() > 0:
            df['Volume_MA'] = df['Volume'].rolling(window=self.volume_ma_length).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            df['Price_Change'] = np.abs(df['Close'].pct_change()) * 100
            df['Volume_Price_Thrust'] = (df['Volume_Ratio'] * df['Price_Change']).fillna(0)
        else:
            # For indices without volume, use price-based thrust
            df['Price_Change'] = np.abs(df['Close'].pct_change()) * 100
            df['Price_Volatility'] = df['Price_Change'].rolling(window=self.volume_ma_length).std()
            df['Volume_Price_Thrust'] = (df['Price_Change'] / df['Price_Volatility']).fillna(0)
            df['Volume_Ratio'] = 1.0  # Default value
        
        return df
    
    def calculate_support_resistance(self, df):
        """Calculate dynamic support/resistance levels"""
        window = self.momentum_length * 2
        df['Highest_High'] = df['High'].rolling(window=window).max()
        df['Lowest_Low'] = df['Low'].rolling(window=window).min()
        df['Mid_Range'] = (df['Highest_High'] + df['Lowest_Low']) / 2
        
        # Calculate trend strength
        df['Trend_Strength'] = (df['Close'] - df['Mid_Range']) / df['Mid_Range'] * 100
        
        return df
    
    def calculate_elite_score(self, df):
        """Calculate proprietary Elite Score"""
        # Momentum Score (0 to 1)
        df['Momentum_Percentile'] = df['Momentum_Strength'].rolling(window=self.momentum_length*2).rank(pct=True).fillna(0.5)
        
        # Volatility Score (higher volatility = higher score for breakouts)
        df['Volatility_Percentile'] = df['ATR'].rolling(window=self.momentum_length*2).rank(pct=True).fillna(0.5)
        
        # Trend Score
        df['Trend_Score'] = np.where(np.abs(df['Trend_Strength']) > 1, 1.0, 0.5)
        
        # Volume/Price Thrust Score
        df['Thrust_Percentile'] = df['Volume_Price_Thrust'].rolling(window=self.momentum_length).rank(pct=True).fillna(0.5)
        
        # Combined Elite Score with different weights for volume vs non-volume
        if self.use_volume and 'Volume' in df.columns and df['Volume'].sum() > 0:
            # With volume
            df['Elite_Score'] = (df['Momentum_Percentile'] * 0.3 + 
                               df['Volatility_Percentile'] * 0.2 + 
                               df['Trend_Score'] * 0.2 + 
                               df['Thrust_Percentile'] * 0.3)
        else:
            # Without volume (for indices)
            df['Elite_Score'] = (df['Momentum_Percentile'] * 0.4 + 
                               df['Volatility_Percentile'] * 0.3 + 
                               df['Trend_Score'] * 0.3)
        
        return df
    
    def generate_signals(self, df):
        """Generate VMA-Elite trading signals with enhanced conditions"""
        # Calculate thrust average
        df['VPT_Average'] = df['Volume_Price_Thrust'].rolling(window=20).mean().fillna(df['Volume_Price_Thrust'].mean())
        
        # Enhanced signal conditions
        bullish_conditions = [
            df['Volume_Price_Thrust'] > (df['VPT_Average'] * self.sensitivity),
            df['Momentum_Acceleration'] > 0,
            df['Close'] > df['Mid_Range'],
            df['Elite_Score'] > 0.6,
            df['ATR'] > df['ATR'].rolling(window=self.momentum_length).quantile(0.3)  # Minimum volatility
        ]
        
        bearish_conditions = [
            df['Volume_Price_Thrust'] > (df['VPT_Average'] * self.sensitivity),
            df['Momentum_Acceleration'] < 0,
            df['Close'] < df['Mid_Range'],
            df['Elite_Score'] > 0.6,
            df['ATR'] > df['ATR'].rolling(window=self.momentum_length).quantile(0.3)  # Minimum volatility
        ]
        
        # Combine conditions safely
        df['Bullish_Thrust'] = pd.Series(True, index=df.index)
        for condition in bullish_conditions:
            df['Bullish_Thrust'] = df['Bullish_Thrust'] & condition.fillna(False)
        
        df['Bearish_Thrust'] = pd.Series(True, index=df.index)
        for condition in bearish_conditions:
            df['Bearish_Thrust'] = df['Bearish_Thrust'] & condition.fillna(False)
        
        # Entry signals (non-repainting) with cooling period
        prev_bullish = df['Bullish_Thrust'].shift(1).fillna(False)
        prev_bearish = df['Bearish_Thrust'].shift(1).fillna(False)
        
        df['Long_Signal'] = (df['Bullish_Thrust'] & ~prev_bullish & (df['Trend_Strength'] > 0))
        df['Short_Signal'] = (df['Bearish_Thrust'] & ~prev_bearish & (df['Trend_Strength'] < 0))
        
        # Add cooling period (no new signal for N bars after a signal)
        cooling_period = 5
        for i in range(1, cooling_period + 1):
            df['Long_Signal'] = df['Long_Signal'] & ~df['Long_Signal'].shift(i).fillna(False)
            df['Short_Signal'] = df['Short_Signal'] & ~df['Short_Signal'].shift(i).fillna(False)
        
        # Calculate stop loss and targets with percentage-based approach
        df['Long_SL'] = df['Close'] * (1 - self.stop_loss_pct / 100)
        df['Short_SL'] = df['Close'] * (1 + self.stop_loss_pct / 100)
        
        df['Long_Target'] = df['Close'] * (1 + self.target_pct / 100)
        df['Short_Target'] = df['Close'] * (1 - self.target_pct / 100)
        
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
        
        # Buy and hold comparison
        buy_hold_entry = df['Close'].iloc[0]
        buy_hold_exit = df['Close'].iloc[-1]
        buy_hold_return = (buy_hold_exit - buy_hold_entry) / buy_hold_entry * 100
        buy_hold_points = buy_hold_exit - buy_hold_entry
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Check for new signals
            if row['Long_Signal'] or row['Short_Signal']:
                # Close existing opposite trade
                if current_trade and ((row['Long_Signal'] and current_trade['type'] == 'SHORT') or 
                                    (row['Short_Signal'] and current_trade['type'] == 'LONG')):
                    # Close trade at current price
                    exit_price = row['Close']
                    pnl_points = self.calculate_pnl_points(current_trade, exit_price)
                    pnl_pct = self.calculate_pnl_percentage(current_trade, exit_price)
                    pnl_amount = (pnl_pct / 100) * balance
                    balance += pnl_amount
                    max_balance = max(max_balance, balance)
                    
                    current_trade.update({
                        'exit_price': exit_price,
                        'exit_date': row.name,
                        'exit_reason': 'Opposite Signal',
                        'pnl_points': pnl_points,
                        'pnl_percentage': pnl_pct,
                        'pnl_amount': pnl_amount,
                        'balance': balance
                    })
                    trades.append(current_trade)
                    current_trade = None
                
                # Open new trade
                if not current_trade:
                    trade_type = 'LONG' if row['Long_Signal'] else 'SHORT'
                    stop_loss = row['Long_SL'] if trade_type == 'LONG' else row['Short_SL']
                    target = row['Long_Target'] if trade_type == 'LONG' else row['Short_Target']
                    
                    # Calculate probability of profit
                    prob_profit = min(0.95, row['Elite_Score'] * 1.3)
                    
                    current_trade = {
                        'entry_date': row.name,
                        'entry_price': row['Close'],
                        'type': trade_type,
                        'stop_loss': stop_loss,
                        'target': target,
                        'elite_score': row['Elite_Score'],
                        'momentum': row['Momentum_Acceleration'],
                        'volume_thrust': row['Volume_Price_Thrust'],
                        'probability_profit': prob_profit,
                        'entry_reason': self.get_entry_reason(row, trade_type),
                        'atr': row['ATR']
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
                    pnl_points = self.calculate_pnl_points(current_trade, exit_price)
                    pnl_pct = self.calculate_pnl_percentage(current_trade, exit_price)
                    pnl_amount = (pnl_pct / 100) * balance
                    balance += pnl_amount
                    max_balance = max(max_balance, balance)
                    
                    current_trade.update({
                        'exit_price': exit_price,
                        'exit_date': row.name,
                        'exit_reason': exit_reason,
                        'pnl_points': pnl_points,
                        'pnl_percentage': pnl_pct,
                        'pnl_amount': pnl_amount,
                        'balance': balance
                    })
                    trades.append(current_trade)
                    current_trade = None
        
        # Calculate performance metrics
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        performance = self.calculate_performance_metrics(trades_df, initial_balance, balance, buy_hold_return, buy_hold_points)
        
        return df, trades_df, performance
    
    def calculate_pnl_points(self, trade, exit_price):
        """Calculate P&L in points"""
        if trade['type'] == 'LONG':
            return exit_price - trade['entry_price']
        else:
            return trade['entry_price'] - exit_price
    
    def calculate_pnl_percentage(self, trade, exit_price):
        """Calculate P&L as percentage"""
        if trade['type'] == 'LONG':
            return ((exit_price - trade['entry_price']) / trade['entry_price']) * 100
        else:
            return ((trade['entry_price'] - exit_price) / trade['entry_price']) * 100
    
    def get_entry_reason(self, row, trade_type):
        """Generate entry reason explanation"""
        reasons = []
        if row['Elite_Score'] > 0.6:
            reasons.append(f"High Elite Score ({row['Elite_Score']:.2f})")
        if row['Volume_Price_Thrust'] > row.get('VPT_Average', 0) * self.sensitivity:
            reasons.append("Strong Momentum Thrust")
        if trade_type == 'LONG' and row['Momentum_Acceleration'] > 0:
            reasons.append("Positive Momentum")
        elif trade_type == 'SHORT' and row['Momentum_Acceleration'] < 0:
            reasons.append("Negative Momentum")
        if trade_type == 'LONG' and row['Close'] > row['Mid_Range']:
            reasons.append("Above Key Level")
        elif trade_type == 'SHORT' and row['Close'] < row['Mid_Range']:
            reasons.append("Below Key Level")
        
        return "; ".join(reasons)
    
    def calculate_performance_metrics(self, trades_df, initial_balance, final_balance, buy_hold_return, buy_hold_points):
        """Calculate comprehensive performance metrics"""
        if trades_df.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'total_points': 0,
                'avg_win_points': 0,
                'avg_loss_points': 0,
                'avg_win_pct': 0,
                'avg_loss_pct': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'final_balance': initial_balance,
                'best_trade_points': 0,
                'worst_trade_points': 0,
                'best_trade_pct': 0,
                'worst_trade_pct': 0,
                'buy_hold_return': buy_hold_return,
                'buy_hold_points': buy_hold_points,
                'strategy_vs_buyhold_points': 0,
                'strategy_vs_buyhold_pct': 0
            }
        
        winning_trades = trades_df[trades_df['pnl_points'] > 0]
        losing_trades = trades_df[trades_df['pnl_points'] <= 0]
        
        total_return = ((final_balance - initial_balance) / initial_balance) * 100
        win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        
        # Points-based metrics
        total_points = trades_df['pnl_points'].sum()
        avg_win_points = winning_trades['pnl_points'].mean() if len(winning_trades) > 0 else 0
        avg_loss_points = losing_trades['pnl_points'].mean() if len(losing_trades) > 0 else 0
        
        # Percentage-based metrics
        avg_win_pct = winning_trades['pnl_percentage'].mean() if len(winning_trades) > 0 else 0
        avg_loss_pct = losing_trades['pnl_percentage'].mean() if len(losing_trades) > 0 else 0
        
        profit_factor = abs(avg_win_points / avg_loss_points) if avg_loss_points != 0 else 0
        
        # Calculate maximum drawdown
        if 'balance' in trades_df.columns:
            equity_curve = trades_df['balance'].values
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak * 100
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        else:
            max_drawdown = 0
        
        # Strategy vs Buy & Hold comparison
        strategy_vs_buyhold_points = total_points - buy_hold_points
        strategy_vs_buyhold_pct = total_return - buy_hold_return
        
        return {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'total_points': total_points,
            'avg_win_points': avg_win_points,
            'avg_loss_points': avg_loss_points,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'final_balance': final_balance,
            'best_trade_points': trades_df['pnl_points'].max() if len(trades_df) > 0 else 0,
            'worst_trade_points': trades_df['pnl_points'].min() if len(trades_df) > 0 else 0,
            'best_trade_pct': trades_df['pnl_percentage'].max() if len(trades_df) > 0 else 0,
            'worst_trade_pct': trades_df['pnl_percentage'].min() if len(trades_df) > 0 else 0,
            'buy_hold_return': buy_hold_return,
            'buy_hold_points': buy_hold_points,
            'strategy_vs_buyhold_points': strategy_vs_buyhold_points,
            'strategy_vs_buyhold_pct': strategy_vs_buyhold_pct
        }

def load_data_yfinance(symbol, period="2y", interval="1d"):
    """Load data from Yahoo Finance"""
    try:
        # Indian market symbols
        indian_symbols = {
            'NIFTY': '^NSEI',
            'BANKNIFTY': '^NSEBANK',
            'SENSEX': '^BSESN',
            'NIFTY50': '^NSEI',
            'NIFTYBANK': '^NSEBANK'
        }
        
        # Use Indian symbol if available
        yf_symbol = indian_symbols.get(symbol.upper(), symbol)
        
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            st.error(f"No data found for {symbol} ({yf_symbol})")
            return None, False
            
        # Reset index to get Date as a column
        df = df.reset_index()
        
        # Handle different column structures
        if 'Datetime' in df.columns:
            df['Date'] = df['Datetime']
            df = df.drop('Datetime', axis=1)
        
        # Check if volume exists and has meaningful data
        has_volume = 'Volume' in df.columns and df['Volume'].sum() > 0
        
        # Ensure we have OHLC data
        required_columns = ['Open', 'High', 'Low', 'Close']
        available_columns = [col for col in required_columns if col in df.columns]
        
        if len(available_columns) < 4:
            st.error(f"Missing required OHLC columns. Available: {list(df.columns)}")
            return None, False
        
        # Select columns (include Volume if available)
        columns_to_keep = ['Date'] + required_columns
        if has_volume:
            columns_to_keep.append('Volume')
        else:
            df['Volume'] = 0  # Add dummy volume column
            columns_to_keep.append('Volume')
        
        df = df[columns_to_keep].copy()
        df.set_index('Date', inplace=True)
        
        # Ensure numeric types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        if df.empty:
            st.error(f"No valid data found for {symbol}")
            return None, has_volume
            
        return df, has_volume
        
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {str(e)}")
        return None, False

def load_data_csv(uploaded_file):
    """Load data from uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Try to identify date column
        date_columns = [col for col in df.columns if any(word in col.lower() for word in ['date', 'time', 'timestamp'])]
        if date_columns:
            df[date_columns[0]] = pd.to_datetime(df[date_columns[0]])
            df = df.set_index(date_columns[0])
        
        # Standardize column names
        column_mapping = {}
        for col in df.columns:
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
        
        df = df.rename(columns=column_mapping)
        
        # Check volume availability
        has_volume = 'Volume' in df.columns and df['Volume'].sum() > 0
        if not has_volume:
            df['Volume'] = 0
        
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if all(col in df.columns for col in required_cols):
            return df[required_cols], has_volume
        else:
            st.error(f"❌ CSV missing required columns. Found: {list(df.columns)}")
            return None, False
            
    except Exception as e:
        st.error(f"❌ Error reading CSV: {str(e)}")
        return None, False

def create_plot(df, trades_df=None):
    """Create comprehensive trading chart"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Signals', 'Elite Score', 'Volume/Price Thrust', 'Momentum Acceleration'),
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
            marker=dict(symbol='triangle-up', size=15, color='green')
        ), row=1, col=1)
    
    if not short_signals.empty:
        fig.add_trace(go.Scatter(
            x=short_signals.index,
            y=short_signals['Close'],
            mode='markers',
            name='Short Signal',
            marker=dict(symbol='triangle-down', size=15, color='red')
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
    
    # Volume/Price Thrust
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Volume_Price_Thrust'],
        mode='lines',
        name='Price Thrust',
        line=dict(color='orange')
    ), row=3, col=1)
    
    if 'VPT_Average' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['VPT_Average'],
            mode='lines',
            name='Thrust Average',
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
    st.markdown("**Enhanced for Indian Markets** - Nifty, BankNifty, Sensex & Global Assets")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'has_volume' not in st.session_state:
        st.session_state.has_volume = True
    if 'data_source' not in st.session_state:
        st.session_state.data_source = "File Upload"
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Data Source Selection
    st.sidebar.subheader("📊 Data Source")
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["File Upload", "Yahoo Finance"],
        index=0
    )
    
    # Data loading based on source
    uploaded_file = None
    symbol = ""
    period = "2y"
    interval = "1d"
    
    if data_source == "File Upload":
        uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")
    else:
        # Yahoo Finance options
        st.sidebar.subheader("🏛️ Market Selection")
        market_type = st.sidebar.selectbox(
            "Market Type:",
            ["Indian Indices", "Indian Stocks", "US Stocks", "Crypto", "Custom"]
        )
        
        if market_type == "Indian Indices":
            symbol = st.sidebar.selectbox(
                "Select Index:",
                ["NIFTY", "BANKNIFTY", "SENSEX", "NIFTY50", "NIFTYBANK"],
                index=0
            )
        elif market_type == "Indian Stocks":
            symbol = st.sidebar.text_input("Stock Symbol (e.g., TCS.NS, RELIANCE.NS):", "TCS.NS")
        elif market_type == "US Stocks":
            symbol = st.sidebar.text_input("Stock Symbol (e.g., AAPL, MSFT, TSLA):", "AAPL")
        elif market_type == "Crypto":
            symbol = st.sidebar.selectbox(
                "Select Crypto:",
                ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD"],
                index=0
            )
        else:
            symbol = st.sidebar.text_input("Custom Symbol:", "")
        
        period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=4)
        interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    
    # Strategy parameters
    st.sidebar.subheader("⚙️ Strategy Parameters")
    momentum_length = st.sidebar.slider("Momentum Length", 5, 50, 14)
    volume_ma_length = st.sidebar.slider("Volume/Thrust MA Length", 10, 100, 20)
    atr_length = st.sidebar.slider("ATR Length", 5, 50, 14)
    sensitivity = st.sidebar.slider("Signal Sensitivity", 0.5, 3.0, 1.2, 0.1)
    
    # Risk Management
    st.sidebar.subheader("🛡️ Risk Management")
    risk_type = st.sidebar.radio("Risk Management Type:", ["Percentage Based", "Risk-Reward Ratio"])
    
    if risk_type == "Percentage Based":
        stop_loss_pct = st.sidebar.slider("Stop Loss %", 0.5, 10.0, 2.0, 0.1)
        target_pct = st.sidebar.slider("Target %", 1.0, 20.0, 5.0, 0.1)
        risk_reward_ratio = target_pct / stop_loss_pct
    else:
        stop_loss_pct = st.sidebar.slider("Stop Loss %", 0.5, 10.0, 2.0, 0.1)
        risk_reward_ratio = st.sidebar.slider("Risk:Reward Ratio", 1.0, 5.0, 2.5, 0.1)
        target_pct = stop_loss_pct * risk_reward_ratio
    
    st.sidebar.info(f"📊 Current Setup: SL: {stop_loss_pct:.1f}% | Target: {target_pct:.1f}% | R:R = 1:{risk_reward_ratio:.1f}")
    
    # Fetch data button
    fetch_button = st.sidebar.button("🔄 Load Data", type="primary", use_container_width=True)
    
    # Data status
    if st.session_state.data_loaded:
        volume_status = "✅ With Volume" if st.session_state.has_volume else "⚠️ No Volume (Index Mode)"
        st.sidebar.success(f"✅ Data loaded: {volume_status}")
        if st.session_state.df is not None:
            st.sidebar.info(f"📊 Data points: {len(st.session_state.df)}")
    else:
        st.sidebar.info("👆 Configure settings and click 'Load Data'")
    
    # Mode selection
    st.sidebar.subheader("📈 Analysis Mode")
    mode = st.sidebar.radio("Select Mode:", ["📊 Backtesting", "🎯 Live Analysis"])
    
    # Handle data loading
    if fetch_button or (data_source == "File Upload" and uploaded_file is not None):
        if data_source == "File Upload" and uploaded_file is not None:
            with st.spinner("Loading CSV data..."):
                df, has_volume = load_data_csv(uploaded_file)
        else:
            if symbol:
                with st.spinner(f"Loading {symbol} data from Yahoo Finance..."):
                    df, has_volume = load_data_yfinance(symbol, period, interval)
            else:
                st.error("Please enter a symbol for Yahoo Finance data")
                return
        
        if df is not None and not df.empty:
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.session_state.has_volume = has_volume
            st.session_state.data_source = data_source
            
            volume_msg = "with volume data" if has_volume else "without volume (using price-based indicators)"
            st.success(f"✅ Successfully loaded {len(df)} data points {volume_msg}")
        else:
            st.session_state.data_loaded = False
            st.error("❌ Failed to load data. Please check your input.")
            return
    
    # Check if data is available
    if not st.session_state.data_loaded or st.session_state.df is None:
        st.info("👆 Please configure your settings and click 'Load Data' to begin analysis.")
        
        # Show sample Indian market symbols
        st.markdown("### 🇮🇳 Indian Market Examples")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Indices:**")
            st.markdown("- NIFTY")
            st.markdown("- BANKNIFTY") 
            st.markdown("- SENSEX")
        with col2:
            st.markdown("**Stocks (.NS suffix):**")
            st.markdown("- TCS.NS")
            st.markdown("- RELIANCE.NS")
            st.markdown("- HDFCBANK.NS")
        with col3:
            st.markdown("**ETFs:**")
            st.markdown("- NIFTYBEES.NS")
            st.markdown("- BANKBEES.NS")
            st.markdown("- GOLDBEES.NS")
        return
    
    df = st.session_state.df
    has_volume = st.session_state.has_volume
    
    # Initialize strategy
    strategy = VMAEliteStrategy(
        momentum_length=momentum_length,
        volume_ma_length=volume_ma_length,
        atr_length=atr_length,
        sensitivity=sensitivity,
        risk_reward_ratio=risk_reward_ratio,
        stop_loss_pct=stop_loss_pct,
        target_pct=target_pct,
        use_volume=has_volume
    )
    
    # Main analysis based on mode
    if mode == "📊 Backtesting":
        st.header("📊 Comprehensive Backtesting Results")
        
        with st.spinner("Running enhanced backtest..."):
            processed_df, trades_df, performance = strategy.backtest(df)
        
        # Enhanced Performance Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta_color = "normal" if performance['strategy_vs_buyhold_pct'] > 0 else "inverse"
            st.metric(
                "Strategy Return",
                f"{performance['total_return']:.2f}%",
                delta=f"{performance['strategy_vs_buyhold_pct']:+.2f}% vs B&H",
                delta_color=delta_color
            )
        
        with col2:
            st.metric(
                "Total Points",
                f"{performance['total_points']:+.1f}",
                delta=f"{performance['strategy_vs_buyhold_points']:+.1f} vs B&H"
            )
        
        with col3:
            st.metric(
                "Win Rate",
                f"{performance['win_rate']:.1f}%",
                delta=f"{performance['winning_trades']}/{performance['total_trades']} trades"
            )
        
        with col4:
            st.metric(
                "Profit Factor",
                f"{performance['profit_factor']:.2f}",
                delta=f"Max DD: {performance['max_drawdown']:.2f}%"
            )
        
        # Additional metrics
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric("Avg Win", f"{performance['avg_win_points']:+.1f} pts ({performance['avg_win_pct']:+.1f}%)")
        with col6:
            st.metric("Avg Loss", f"{performance['avg_loss_points']:+.1f} pts ({performance['avg_loss_pct']:+.1f}%)")
        with col7:
            st.metric("Best Trade", f"{performance['best_trade_points']:+.1f} pts ({performance['best_trade_pct']:+.1f}%)")
        with col8:
            st.metric("Worst Trade", f"{performance['worst_trade_points']:+.1f} pts ({performance['worst_trade_pct']:+.1f}%)")
        
        # Strategy vs Buy & Hold Comparison
        st.subheader("⚡ Strategy vs Buy & Hold Detailed Comparison")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🚀 VMA-Elite Strategy")
            st.write(f"**Return:** {performance['total_return']:.2f}%")
            st.write(f"**Points:** {performance['total_points']:+.1f}")
            st.write(f"**Trades:** {performance['total_trades']}")
            st.write(f"**Win Rate:** {performance['win_rate']:.1f}%")
            st.write(f"**Max Drawdown:** {performance['max_drawdown']:.2f}%")
        
        with col2:
            st.markdown("### 📈 Buy & Hold")
            st.write(f"**Return:** {performance['buy_hold_return']:.2f}%")
            st.write(f"**Points:** {performance['buy_hold_points']:+.1f}")
            st.write(f"**Trades:** 1 (Buy & Hold)")
            st.write(f"**Win Rate:** {'100%' if performance['buy_hold_return'] > 0 else '0%'}")
            st.write(f"**Max Drawdown:** N/A")
        
        with col3:
            st.markdown("### 🎯 Outperformance")
            outperf_pct = performance['strategy_vs_buyhold_pct']
            outperf_points = performance['strategy_vs_buyhold_points']
            
            color = "🟢" if outperf_pct > 0 else "🔴"
            st.write(f"**Return Diff:** {color} {outperf_pct:+.2f}%")
            st.write(f"**Points Diff:** {color} {outperf_points:+.1f}")
            st.write(f"**Active Management:** {'✅ Better' if outperf_pct > 0 else '❌ Worse'}")
            st.write(f"**Risk Adjusted:** {'✅ Superior' if outperf_pct > 0 and performance['max_drawdown'] < 20 else '⚠️ Review'}")
        
        # Chart
        st.subheader("📈 Trading Chart & Signals")
        fig = create_plot(processed_df, trades_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Trade History
        if not trades_df.empty:
            st.subheader("💼 Detailed Trade History")
            
            # Format trades for display
            display_trades = trades_df.copy()
            display_trades['Entry Date'] = pd.to_datetime(display_trades['entry_date']).dt.strftime('%Y-%m-%d')
            display_trades['Exit Date'] = pd.to_datetime(display_trades['exit_date']).dt.strftime('%Y-%m-%d')
            display_trades['Entry Price'] = display_trades['entry_price'].round(2)
            display_trades['Exit Price'] = display_trades['exit_price'].round(2)
            display_trades['Points P&L'] = display_trades['pnl_points'].round(2)
            display_trades['% P&L'] = display_trades['pnl_percentage'].round(2)
            display_trades['Type'] = display_trades['type']
            display_trades['Elite Score'] = display_trades['elite_score'].round(3)
            display_trades['Probability %'] = (display_trades['probability_profit'] * 100).round(1)
            display_trades['Entry Reason'] = display_trades['entry_reason']
            display_trades['Exit Reason'] = display_trades['exit_reason']
            
            # Color coding function
            def highlight_trades(row):
                if row['Points P&L'] > 0:
                    return ['background-color: rgba(34, 197, 94, 0.2)'] * len(row)
                else:
                    return ['background-color: rgba(239, 68, 68, 0.2)'] * len(row)
            
            columns_to_show = ['Entry Date', 'Type', 'Entry Price', 'Exit Price', 
                             'Points P&L', '% P&L', 'Elite Score', 'Probability %', 
                             'Entry Reason', 'Exit Reason']
            
            styled_df = display_trades[columns_to_show].style.apply(highlight_trades, axis=1)
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Trade Statistics
            st.subheader("📊 Trade Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📈 Winning Trades:**")
                winning_trades = trades_df[trades_df['pnl_points'] > 0]
                st.write(f"- Count: {len(winning_trades)} ({len(winning_trades)/len(trades_df)*100:.1f}%)")
                if len(winning_trades) > 0:
                    st.write(f"- Avg Points: {winning_trades['pnl_points'].mean():.2f}")
                    st.write(f"- Avg %: {winning_trades['pnl_percentage'].mean():.2f}%")
                    st.write(f"- Best Trade: {winning_trades['pnl_points'].max():.2f} points")
                    st.write(f"- Avg Elite Score: {winning_trades['elite_score'].mean():.3f}")
            
            with col2:
                st.write("**📉 Losing Trades:**")
                losing_trades = trades_df[trades_df['pnl_points'] <= 0]
                st.write(f"- Count: {len(losing_trades)} ({len(losing_trades)/len(trades_df)*100:.1f}%)")
                if len(losing_trades) > 0:
                    st.write(f"- Avg Points: {losing_trades['pnl_points'].mean():.2f}")
                    st.write(f"- Avg %: {losing_trades['pnl_percentage'].mean():.2f}%")
                    st.write(f"- Worst Trade: {losing_trades['pnl_points'].min():.2f} points")
                    st.write(f"- Avg Elite Score: {losing_trades['elite_score'].mean():.3f}")
        
        else:
            st.warning("⚠️ No trades generated. Try adjusting sensitivity or risk parameters.")
            st.info("**Suggestions:**")
            st.info("- Lower the Elite Score threshold")
            st.info("- Reduce signal sensitivity")
            st.info("- Adjust momentum length")
    
    else:  # Live Analysis Mode
        st.header("🎯 Live Trading Analysis")
        
        with st.spinner("Processing live market data..."):
            processed_df = strategy.process_data(df)
        
        # Current market status
        latest = processed_df.iloc[-1]
        prev = processed_df.iloc[-2]
        
        st.subheader("📊 Current Market Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price_change = latest['Close'] - prev['Close']
            price_change_pct = (price_change / prev['Close']) * 100
            st.metric(
                "Current Price",
                f"{latest['Close']:.2f}",
                delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
            )
        
        with col2:
            elite_status = "🟢 High Quality" if latest['Elite_Score'] > 0.6 else "🔴 Low Quality"
            st.metric(
                "Elite Score",
                f"{latest['Elite_Score']:.3f}",
                delta=elite_status
            )
        
        with col3:
            momentum_status = "🟢 Bullish" if latest['Momentum_Acceleration'] > 0 else "🔴 Bearish"
            st.metric(
                "Momentum",
                f"{latest['Momentum_Acceleration']:.2f}",
                delta=momentum_status
            )
        
        with col4:
            thrust_avg = latest.get('VPT_Average', 0)
            thrust_status = "🟢 Strong" if latest['Volume_Price_Thrust'] > thrust_avg else "🔴 Weak"
            st.metric(
                "Price Thrust",
                f"{latest['Volume_Price_Thrust']:.2f}",
                delta=thrust_status
            )
        
        # Live signals check
        st.subheader("🚨 Live Trading Signals")
        
        # Check recent signals (last 10 bars)
        recent_signals = processed_df.tail(10)
        long_signals = recent_signals[recent_signals['Long_Signal']]
        short_signals = recent_signals[recent_signals['Short_Signal']]
        
        if not long_signals.empty or not short_signals.empty:
            if not long_signals.empty:
                latest_long = long_signals.iloc[-1]
                st.success("🟢 **LONG SIGNAL DETECTED!**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**📋 Trade Setup:**")
                    st.write(f"- **Entry Price:** {latest_long['Close']:.2f}")
                    st.write(f"- **Stop Loss:** {latest_long['Long_SL']:.2f} ({stop_loss_pct:.1f}%)")
                    st.write(f"- **Target:** {latest_long['Long_Target']:.2f} ({target_pct:.1f}%)")
                    st.write(f"- **Risk/Reward:** 1:{risk_reward_ratio:.1f}")
                    st.write(f"- **Points Risk:** {latest_long['Close'] - latest_long['Long_SL']:.2f}")
                    st.write(f"- **Points Target:** {latest_long['Long_Target'] - latest_long['Close']:.2f}")
                    
                with col2:
                    st.write("**📊 Signal Quality:**")
                    st.write(f"- **Elite Score:** {latest_long['Elite_Score']:.3f}")
                    st.write(f"- **Probability:** {min(0.95, latest_long['Elite_Score'] * 1.3)*100:.1f}%")
                    st.write(f"- **Signal Date:** {latest_long.name.strftime('%Y-%m-%d')}")
                    st.write(f"- **ATR:** {latest_long['ATR']:.2f}")
                    st.write(f"- **Momentum:** {latest_long['Momentum_Acceleration']:.2f}")
                    
                st.write("**💡 Entry Reasons:**")
                st.info(strategy.get_entry_reason(latest_long, 'LONG'))
            
            if not short_signals.empty:
                latest_short = short_signals.iloc[-1]
                st.error("🔴 **SHORT SIGNAL DETECTED!**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**📋 Trade Setup:**")
                    st.write(f"- **Entry Price:** {latest_short['Close']:.2f}")
                    st.write(f"- **Stop Loss:** {latest_short['Short_SL']:.2f} ({stop_loss_pct:.1f}%)")
                    st.write(f"- **Target:** {latest_short['Short_Target']:.2f} ({target_pct:.1f}%)")
                    st.write(f"- **Risk/Reward:** 1:{risk_reward_ratio:.1f}")
                    st.write(f"- **Points Risk:** {latest_short['Short_SL'] - latest_short['Close']:.2f}")
                    st.write(f"- **Points Target:** {latest_short['Close'] - latest_short['Short_Target']:.2f}")
                    
                with col2:
                    st.write("**📊 Signal Quality:**")
                    st.write(f"- **Elite Score:** {latest_short['Elite_Score']:.3f}")
                    st.write(f"- **Probability:** {min(0.95, latest_short['Elite_Score'] * 1.3)*100:.1f}%")
                    st.write(f"- **Signal Date:** {latest_short.name.strftime('%Y-%m-%d')}")
                    st.write(f"- **ATR:** {latest_short['ATR']:.2f}")
                    st.write(f"- **Momentum:** {latest_short['Momentum_Acceleration']:.2f}")
                    
                st.write("**💡 Entry Reasons:**")
                st.info(strategy.get_entry_reason(latest_short, 'SHORT'))
        
        else:
            st.info("⏳ **No Active Signals - Market in Wait Mode**")
            
            st.write("**🔍 Next Signal Requirements:**")
            requirements = []
            if latest['Elite_Score'] <= 0.6:
                requirements.append(f"✗ Elite Score > 0.6 (Current: {latest['Elite_Score']:.3f})")
            else:
                requirements.append(f"✓ Elite Score > 0.6 (Current: {latest['Elite_Score']:.3f})")
            
            thrust_threshold = latest.get('VPT_Average', 0) * sensitivity
            if latest['Volume_Price_Thrust'] <= thrust_threshold:
                requirements.append(f"✗ Price Thrust > {thrust_threshold:.2f} (Current: {latest['Volume_Price_Thrust']:.2f})")
            else:
                requirements.append(f"✓ Price Thrust > {thrust_threshold:.2f} (Current: {latest['Volume_Price_Thrust']:.2f})")
            
            if latest['Momentum_Acceleration'] == 0:
                requirements.append("✗ Clear Momentum Direction Needed")
            elif latest['Momentum_Acceleration'] > 0:
                requirements.append(f"✓ Positive Momentum ({latest['Momentum_Acceleration']:.2f})")
            else:
                requirements.append(f"✓ Negative Momentum ({latest['Momentum_Acceleration']:.2f})")
            
            for req in requirements:
                st.write(f"- {req}")
        
        # Live chart
        st.subheader("📈 Live Market Analysis")
        fig = create_plot(processed_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Key levels for live trading
        st.subheader("🎯 Key Trading Levels")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📊 Support/Resistance:**")
            st.write(f"- **Key Level:** {latest['Mid_Range']:.2f}")
            st.write(f"- **Resistance:** {latest['Highest_High']:.2f}")
            st.write(f"- **Support:** {latest['Lowest_Low']:.2f}")
            st.write(f"- **Trend:** {latest['Trend_Strength']:.2f}%")
            
        with col2:
            st.write("**⚡ Volatility Metrics:**")
            st.write(f"- **ATR:** {latest['ATR']:.2f}")
            st.write(f"- **ATR %:** {(latest['ATR']/latest['Close']*100):.2f}%")
            st.write(f"- **Price Change:** {latest['Price_Change']:.2f}%")
            
        with col3:
            volume_info = "Volume Analysis:" if has_volume else "Price Analysis:"
            st.write(f"**📈 {volume_info}**")
            if has_volume:
                st.write(f"- **Volume Ratio:** {latest.get('Volume_Ratio', 0):.2f}")
                st.write(f"- **Volume Thrust:** {latest['Volume_Price_Thrust']:.2f}")
            else:
                st.write(f"- **Price Thrust:** {latest['Volume_Price_Thrust']:.2f}")
                st.write(f"- **Volatility Mode:** Index Trading")
            st.write(f"- **Momentum Strength:** {latest['Momentum_Strength']:.2f}")

    # Footer
    st.markdown("---")
    volume_mode = "With Volume Analysis" if has_volume else "Index Mode (Price-Based)"
    st.markdown(f"**🚀 VMA-Elite Trading System** - {volume_mode}")
    st.markdown("⚠️ *Trading involves risk. Past performance does not guarantee future results.*")
    st.markdown("🇮🇳 *Optimized for Indian Markets - Nifty, BankNifty, Sensex*")

if __name__ == "__main__":
    main()
