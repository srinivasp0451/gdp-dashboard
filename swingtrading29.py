import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Advanced Trading System",
    page_icon="🚀",
    layout="wide"
)

class TradingStrategy:
    def __init__(self, strategy_type="Momentum Breakout", ema_fast=9, ema_slow=21, 
                 rsi_period=14, stop_loss_pct=2.0, take_profit_pct=6.0):
        self.strategy_type = strategy_type
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators"""
        df = df.copy()
        
        # EMAs
        df['EMA_Fast'] = df['Close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['EMA_Slow'] = df['Close'].ewm(span=self.ema_slow, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = sma_20 + (std_20 * 2)
        df['BB_Lower'] = sma_20 - (std_20 * 2)
        df['BB_Mid'] = sma_20
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        # Volume
        if df['Volume'].sum() > 0:
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        else:
            df['Volume_Ratio'] = 1.0
        
        df = df.fillna(method='bfill')
        return df
    
    def generate_signals(self, df):
        """Generate trading signals"""
        
        if self.strategy_type == "Momentum Breakout":
            # Long conditions
            long_cond1 = df['Close'] > df['EMA_Fast']
            long_cond2 = df['EMA_Fast'] > df['EMA_Slow']
            long_cond3 = df['RSI'] > 45
            long_cond4 = df['RSI'] < 75
            long_cond5 = df['MACD'] > df['MACD_Signal']
            long_cond6 = df['Volume_Ratio'] > 1.0
            
            df['Long_Score'] = (long_cond1.astype(int) + long_cond2.astype(int) + 
                              long_cond3.astype(int) + long_cond4.astype(int) + 
                              long_cond5.astype(int) + long_cond6.astype(int))
            
            # Short conditions
            short_cond1 = df['Close'] < df['EMA_Fast']
            short_cond2 = df['EMA_Fast'] < df['EMA_Slow']
            short_cond3 = df['RSI'] < 55
            short_cond4 = df['RSI'] > 25
            short_cond5 = df['MACD'] < df['MACD_Signal']
            short_cond6 = df['Volume_Ratio'] > 1.0
            
            df['Short_Score'] = (short_cond1.astype(int) + short_cond2.astype(int) + 
                               short_cond3.astype(int) + short_cond4.astype(int) + 
                               short_cond5.astype(int) + short_cond6.astype(int))
            
            # Signals (need 4 out of 6 conditions)
            df['Long_Signal'] = (df['Long_Score'] >= 4) & (df['Long_Score'].shift(1) < 4)
            df['Short_Signal'] = (df['Short_Score'] >= 4) & (df['Short_Score'].shift(1) < 4)
            
        elif self.strategy_type == "Mean Reversion":
            # Oversold for longs
            long_cond1 = df['RSI'] < 30
            long_cond2 = df['Close'] < df['BB_Lower']
            long_cond3 = df['Close'] > df['EMA_Slow']
            
            df['Long_Score'] = long_cond1.astype(int) + long_cond2.astype(int) + long_cond3.astype(int)
            
            # Overbought for shorts
            short_cond1 = df['RSI'] > 70
            short_cond2 = df['Close'] > df['BB_Upper']
            short_cond3 = df['Close'] < df['EMA_Slow']
            
            df['Short_Score'] = short_cond1.astype(int) + short_cond2.astype(int) + short_cond3.astype(int)
            
            # Signals (need 2 out of 3 conditions)
            df['Long_Signal'] = (df['Long_Score'] >= 2) & (df['Long_Score'].shift(1) < 2)
            df['Short_Signal'] = (df['Short_Score'] >= 2) & (df['Short_Score'].shift(1) < 2)
        
        else:  # Trend Following
            # Strong uptrend
            long_cond1 = df['Close'] > df['EMA_Fast']
            long_cond2 = df['EMA_Fast'] > df['EMA_Slow']
            long_cond3 = df['RSI'] > 50
            long_cond4 = df['MACD'] > 0
            
            df['Long_Score'] = (long_cond1.astype(int) + long_cond2.astype(int) + 
                              long_cond3.astype(int) + long_cond4.astype(int))
            
            # Strong downtrend
            short_cond1 = df['Close'] < df['EMA_Fast']
            short_cond2 = df['EMA_Fast'] < df['EMA_Slow']
            short_cond3 = df['RSI'] < 50
            short_cond4 = df['MACD'] < 0
            
            df['Short_Score'] = (short_cond1.astype(int) + short_cond2.astype(int) + 
                               short_cond3.astype(int) + short_cond4.astype(int))
            
            # Signals (all 4 conditions)
            df['Long_Signal'] = (df['Long_Score'] >= 4) & (df['Long_Score'].shift(1) < 4)
            df['Short_Signal'] = (df['Short_Score'] >= 4) & (df['Short_Score'].shift(1) < 4)
        
        return df
    
    def backtest(self, df):
        """Enhanced backtest function"""
        # Prepare data
        df = df.sort_index()
        df = self.calculate_indicators(df)
        df = self.generate_signals(df)
        
        # Initialize tracking
        trades = []
        in_position = False
        position_type = None
        entry_price = 0
        entry_date = None
        stop_loss = 0
        take_profit = 0
        entry_idx = 0
        
        # Buy and hold metrics
        initial_price = df['Close'].iloc[0]
        final_price = df['Close'].iloc[-1]
        buy_hold_return = ((final_price - initial_price) / initial_price) * 100
        buy_hold_points = final_price - initial_price
        
        # Iterate through data
        for i in range(1, len(df)):
            current_bar = df.iloc[i]
            current_date = df.index[i]
            
            # Check for exit if in position
            if in_position:
                exit_triggered = False
                exit_price = 0
                exit_reason = ''
                
                if position_type == 'LONG':
                    # Check stop loss
                    if current_bar['Low'] <= stop_loss:
                        exit_triggered = True
                        exit_price = stop_loss
                        exit_reason = 'Stop Loss'
                    # Check take profit
                    elif current_bar['High'] >= take_profit:
                        exit_triggered = True
                        exit_price = take_profit
                        exit_reason = 'Take Profit'
                    # Check opposite signal
                    elif current_bar['Short_Signal']:
                        exit_triggered = True
                        exit_price = current_bar['Close']
                        exit_reason = 'Opposite Signal'
                
                else:  # SHORT
                    # Check stop loss
                    if current_bar['High'] >= stop_loss:
                        exit_triggered = True
                        exit_price = stop_loss
                        exit_reason = 'Stop Loss'
                    # Check take profit
                    elif current_bar['Low'] <= take_profit:
                        exit_triggered = True
                        exit_price = take_profit
                        exit_reason = 'Take Profit'
                    # Check opposite signal
                    elif current_bar['Long_Signal']:
                        exit_triggered = True
                        exit_price = current_bar['Close']
                        exit_reason = 'Opposite Signal'
                
                # Execute exit
                if exit_triggered:
                    if position_type == 'LONG':
                        pnl_points = exit_price - entry_price
                        pnl_pct = (pnl_points / entry_price) * 100
                    else:
                        pnl_points = entry_price - exit_price
                        pnl_pct = (pnl_points / entry_price) * 100
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'type': position_type,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'pnl_points': pnl_points,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'duration': i - entry_idx
                    })
                    
                    in_position = False
                    position_type = None
            
            # Check for entry if not in position
            if not in_position:
                if current_bar['Long_Signal']:
                    in_position = True
                    position_type = 'LONG'
                    entry_price = current_bar['Close']
                    entry_date = current_date
                    entry_idx = i
                    stop_loss = entry_price * (1 - self.stop_loss_pct / 100)
                    take_profit = entry_price * (1 + self.take_profit_pct / 100)
                
                elif current_bar['Short_Signal']:
                    in_position = True
                    position_type = 'SHORT'
                    entry_price = current_bar['Close']
                    entry_date = current_date
                    entry_idx = i
                    stop_loss = entry_price * (1 + self.stop_loss_pct / 100)
                    take_profit = entry_price * (1 - self.take_profit_pct / 100)
        
        # Close any remaining position
        if in_position:
            exit_price = df['Close'].iloc[-1]
            exit_date = df.index[-1]
            
            if position_type == 'LONG':
                pnl_points = exit_price - entry_price
                pnl_pct = (pnl_points / entry_price) * 100
            else:
                pnl_points = entry_price - exit_price
                pnl_pct = (pnl_points / entry_price) * 100
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'type': position_type,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'pnl_points': pnl_points,
                'pnl_pct': pnl_pct,
                'exit_reason': 'End of Data',
                'duration': len(df) - entry_idx
            })
        
        # Create trades dataframe
        if trades:
            trades_df = pd.DataFrame(trades)
            
            # Calculate metrics
            winning_trades = trades_df[trades_df['pnl_pct'] > 0]
            losing_trades = trades_df[trades_df['pnl_pct'] <= 0]
            
            total_return = trades_df['pnl_pct'].sum()
            total_points = trades_df['pnl_points'].sum()
            win_rate = (len(winning_trades) / len(trades_df)) * 100
            
            avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
            
            gross_profit = winning_trades['pnl_points'].sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades['pnl_points'].sum()) if len(losing_trades) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Drawdown
            cumulative = trades_df['pnl_pct'].cumsum()
            running_max = cumulative.expanding().max()
            drawdown = running_max - cumulative
            max_drawdown = drawdown.max()
            
            performance = {
                'total_trades': len(trades_df),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'total_return': total_return,
                'total_points': total_points,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'avg_duration': trades_df['duration'].mean(),
                'best_trade': trades_df['pnl_pct'].max(),
                'worst_trade': trades_df['pnl_pct'].min(),
                'buy_hold_return': buy_hold_return,
                'buy_hold_points': buy_hold_points,
                'vs_buy_hold': total_return - buy_hold_return
            }
        else:
            trades_df = pd.DataFrame()
            performance = {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0, 'total_return': 0, 'total_points': 0,
                'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0,
                'max_drawdown': 0, 'avg_duration': 0,
                'best_trade': 0, 'worst_trade': 0,
                'buy_hold_return': buy_hold_return,
                'buy_hold_points': buy_hold_points,
                'vs_buy_hold': -buy_hold_return
            }
        
        return df, trades_df, performance

def load_data(symbol, period="1y", interval="1d"):
    """Load data from Yahoo Finance"""
    try:
        indian_symbols = {
            'NIFTY': '^NSEI', 'BANKNIFTY': '^NSEBANK', 'SENSEX': '^BSESN'
        }
        
        yf_symbol = indian_symbols.get(symbol.upper(), symbol)
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return None
        
        df = df.reset_index()
        if 'Datetime' in df.columns:
            df['Date'] = df['Datetime']
            df = df.drop('Datetime', axis=1)
        
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        return df
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def create_chart(df, trades_df=None):
    """Create trading chart"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Signals', 'RSI', 'MACD'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="Price"
    ), row=1, col=1)
    
    # EMAs
    if 'EMA_Fast' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA_Fast'],
            mode='lines', name='EMA Fast',
            line=dict(color='orange', width=1)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA_Slow'],
            mode='lines', name='EMA Slow',
            line=dict(color='blue', width=1)
        ), row=1, col=1)
    
    # Signals
    if trades_df is not None and not trades_df.empty:
        longs = trades_df[trades_df['type'] == 'LONG']
        shorts = trades_df[trades_df['type'] == 'SHORT']
        
        if not longs.empty:
            fig.add_trace(go.Scatter(
                x=longs['entry_date'], y=longs['entry_price'],
                mode='markers', name='Long',
                marker=dict(symbol='triangle-up', size=10, color='green')
            ), row=1, col=1)
        
        if not shorts.empty:
            fig.add_trace(go.Scatter(
                x=shorts['entry_date'], y=shorts['entry_price'],
                mode='markers', name='Short',
                marker=dict(symbol='triangle-down', size=10, color='red')
            ), row=1, col=1)
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'],
            mode='lines', name='RSI',
            line=dict(color='purple')
        ), row=2, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD'],
            mode='lines', name='MACD',
            line=dict(color='blue')
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD_Signal'],
            mode='lines', name='Signal',
            line=dict(color='red')
        ), row=3, col=1)
    
    fig.update_layout(
        height=800,
        template="plotly_dark",
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def main():
    st.title("🚀 Advanced Trading System")
    st.markdown("**Multiple Strategies | All Timeframes | Pure Pandas**")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Strategy
    st.sidebar.subheader("🎯 Strategy")
    strategy_type = st.sidebar.selectbox(
        "Select Strategy:",
        ["Momentum Breakout", "Mean Reversion", "Trend Following"]
    )
    
    # Data
    st.sidebar.subheader("📊 Data")
    
    market = st.sidebar.selectbox(
        "Market:",
        ["Indian Indices", "US Stocks", "Crypto"]
    )
    
    if market == "Indian Indices":
        symbol = st.sidebar.selectbox("Symbol:", ["NIFTY", "BANKNIFTY", "SENSEX"])
    elif market == "US Stocks":
        symbol = st.sidebar.text_input("Symbol:", "AAPL")
    else:
        symbol = st.sidebar.selectbox("Symbol:", ["BTC-USD", "ETH-USD"])
    
    period = st.sidebar.selectbox(
        "Period:",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )
    
    interval = st.sidebar.selectbox(
        "Timeframe:",
        ["1m", "5m", "15m", "30m", "1h", "1d"],
        index=5
    )
    
    # Parameters
    st.sidebar.subheader("⚙️ Parameters")
    ema_fast = st.sidebar.slider("EMA Fast", 5, 20, 9)
    ema_slow = st.sidebar.slider("EMA Slow", 15, 50, 21)
    rsi_period = st.sidebar.slider("RSI Period", 10, 20, 14)
    
    # Risk
    st.sidebar.subheader("🛡️ Risk Management")
    stop_loss = st.sidebar.slider("Stop Loss %", 0.5, 5.0, 2.0, 0.1)
    take_profit = st.sidebar.slider("Take Profit %", 2.0, 15.0, 6.0, 0.5)
    
    st.sidebar.info(f"Risk:Reward = 1:{take_profit/stop_loss:.1f}")
    
    # Load button
    if st.sidebar.button("🔄 Load Data", type="primary"):
        with st.spinner(f"Loading {symbol}..."):
            df = load_data(symbol, period, interval)
            
            if df is not None and not df.empty:
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success(f"✅ Loaded {len(df)} records")
            else:
                st.error("Failed to load data")
                return
    
    if not st.session_state.data_loaded:
        st.info("👆 Configure settings and load data")
        
        # Show features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**✅ Features:**")
            st.markdown("- Pure pandas (no TA-Lib)")
            st.markdown("- All timeframes")
            st.markdown("- Multiple strategies")
            st.markdown("- Optimized backtest")
        
        with col2:
            st.markdown("**📊 Strategies:**")
            st.markdown("- Momentum Breakout")
            st.markdown("- Mean Reversion")
            st.markdown("- Trend Following")
            st.markdown("- More coming soon")
        
        with col3:
            st.markdown("**⏱️ Timeframes:**")
            st.markdown("- 1m, 5m, 15m (intraday)")
            st.markdown("- 30m, 1h (swing)")
            st.markdown("- 1d (daily)")
            st.markdown("- All periods supported")
        
        return
    
    df = st.session_state.df
    
    # Initialize strategy
    strategy = TradingStrategy(
        strategy_type=strategy_type,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_period=rsi_period,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit
    )
    
    # Run backtest
    st.header(f"📊 {strategy_type} Backtest Results")
    
    with st.spinner("Running backtest..."):
        processed_df, trades_df, perf = strategy.backtest(df)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"{perf['total_return']:.2f}%",
            delta=f"{perf['vs_buy_hold']:+.2f}% vs B&H"
        )
    
    with col2:
        st.metric(
            "Trades",
            f"{perf['total_trades']}",
            delta=f"{perf['win_rate']:.1f}% Win Rate"
        )
    
    with col3:
        st.metric(
            "Profit Factor",
            f"{perf['profit_factor']:.2f}",
            delta=f"Avg: {perf['avg_duration']:.0f} bars"
        )
    
    with col4:
        st.metric(
            "Max Drawdown",
            f"{perf['max_drawdown']:.2f}%",
            delta=f"{perf['total_points']:+.1f} pts"
        )
    
    # Assessment
    st.subheader("🎯 Performance Assessment")
    
    score = 0
    if perf['total_return'] > 0:
        score += 30
    if perf['win_rate'] >= 50:
        score += 25
    if perf['total_trades'] >= 10:
        score += 20
    if perf['profit_factor'] >= 1.5:
        score += 15
    if perf['vs_buy_hold'] > 0:
        score += 10
    
    if score >= 80:
        st.success(f"🏆 Excellent ({score}/100)")
    elif score >= 60:
        st.warning(f"⚠️ Good ({score}/100)")
    else:
        st.error(f"❌ Poor ({score}/100)")
    
    # Details
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📈 Strategy Performance:**")
        st.write(f"- Total Return: {perf['total_return']:.2f}%")
        st.write(f"- Total Points: {perf['total_points']:+.1f}")
        st.write(f"- Win Rate: {perf['win_rate']:.1f}%")
        st.write(f"- Profit Factor: {perf['profit_factor']:.2f}")
    
    with col2:
        st.write("**📊 Buy & Hold:**")
        st.write(f"- Return: {perf['buy_hold_return']:.2f}%")
        st.write(f"- Points: {perf['buy_hold_points']:+.1f}")
        st.write(f"- Outperformance: {perf['vs_buy_hold']:+.2f}%")
    
    # Chart
    st.subheader("📈 Trading Chart")
    fig = create_chart(processed_df, trades_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Trades
    if not trades_df.empty:
        st.subheader("💼 Trade History")
        
        display = trades_df.copy()
        display['Entry'] = pd.to_datetime(display['entry_date']).dt.strftime('%Y-%m-%d %H:%M')
        display['Exit'] = pd.to_datetime(display['exit_date']).dt.strftime('%Y-%m-%d %H:%M')
        display['Type'] = display['type']
        display['Entry $'] = display['entry_price'].round(2)
        display['Exit $'] = display['exit_price'].round(2)
        display['P&L %'] = display['pnl_pct'].round(2)
        display['P&L Pts'] = display['pnl_points'].round(1)
        display['Duration'] = display['duration']
        display['Exit Reason'] = display['exit_reason']
        
        # Color coding
        def color_pnl(row):
            colors = []
            for col in row.index:
                if 'P&L' in col:
                    if row[col] > 0:
                        colors.append('background-color: rgba(34, 197, 94, 0.3)')
                    else:
                        colors.append('background-color: rgba(239, 68, 68, 0.3)')
                else:
                    colors.append('')
            return colors
        
        cols = ['Entry', 'Type', 'Entry $', 'Exit $', 'P&L %', 'P&L Pts', 'Duration', 'Exit Reason']
        styled = display[cols].style.apply(color_pnl, axis=1)
        
        st.dataframe(styled, use_container_width=True)
    else:
        st.warning("⚠️ No trades generated!")
        st.info("**Try:**")
        st.info("- Shorter timeframe (1m, 5m)")
        st.info("- Different strategy")
        st.info("- Longer period")
    
    # Footer
    st.markdown("---")
    st.markdown("**🚀 Advanced Trading System** | Pure Pandas | No Dependencies")
    st.markdown("⚠️ *Past performance does not guarantee future results*")

if __name__ == "__main__":
    main()
