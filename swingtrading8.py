import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Stock Trading Recommendation System",
    page_icon="📈",
    layout="wide"
)

class TechnicalIndicators:
    """Calculate technical indicators manually"""
    
    @staticmethod
    def sma(data, window):
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data, window):
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd = ema_fast - ema_slow
        macd_signal = TechnicalIndicators.ema(macd, signal)
        histogram = macd - macd_signal
        return macd, macd_signal, histogram
    
    @staticmethod
    def bollinger_bands(data, window=20, std_dev=2):
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def stochastic(high, low, close, k_window=14, d_window=3):
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    @staticmethod
    def atr(high, low, close, window=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()
    
    @staticmethod
    def williams_r(high, low, close, window=14):
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    @staticmethod
    def momentum(data, window=10):
        return data.pct_change(window) * 100
    
    @staticmethod
    def cci(high, low, close, window=20):
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=window).mean()
        mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (tp - sma_tp) / (0.015 * mad)
    
    @staticmethod
    def obv(close, volume):
        obv = np.where(close > close.shift(1), volume, np.where(close < close.shift(1), -volume, 0))
        return pd.Series(obv, index=close.index).cumsum()

class DataProcessor:
    """Handle data processing and column mapping"""
    
    @staticmethod
    def map_columns(df):
        """Map different column name variations to standard names"""
        column_mapping = {
            'open': ['open', 'open price', 'price open', 'o'],
            'high': ['high', 'high price', 'price high', 'h'],
            'low': ['low', 'low price', 'price low', 'l'],
            'close': ['close', 'close price', 'price close', 'c'],
            'volume': ['volume', 'vol', 'v', 'trading volume']
        }
        
        mapped_df = df.copy()
        column_names = [col.lower().strip() for col in df.columns]
        
        for standard_name, variations in column_mapping.items():
            for col_name in column_names:
                for variation in variations:
                    if variation in col_name:
                        original_col = df.columns[column_names.index(col_name)]
                        mapped_df = mapped_df.rename(columns={original_col: standard_name})
                        break
                if standard_name in mapped_df.columns:
                    break
        
        return mapped_df
    
    @staticmethod
    def validate_data(df):
        """Validate required columns exist"""
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return False
        return True
    
    @staticmethod
    def process_data(df, end_date=None):
        """Process and sort data"""
        # Ensure datetime index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df = df.sort_index()
        
        # Filter by end date if provided
        if end_date:
            df = df[df.index <= end_date]
        
        return df

class StrategyOptimizer:
    """Optimize trading strategy parameters"""
    
    def __init__(self, data, trade_type='both'):
        self.data = data
        self.trade_type = trade_type
        
    def get_parameter_space(self):
        """Define parameter space for optimization"""
        param_space = {
            'rsi_window': [10, 14, 21],
            'rsi_oversold': [20, 25, 30],
            'rsi_overbought': [70, 75, 80],
            'macd_fast': [8, 12, 16],
            'macd_slow': [21, 26, 31],
            'bb_window': [15, 20, 25],
            'bb_std': [1.5, 2.0, 2.5],
            'stoch_k': [10, 14, 18],
            'atr_window': [10, 14, 20],
            'momentum_window': [8, 10, 12],
            'volume_ma': [10, 20, 30],
            'stop_loss_atr': [1.5, 2.0, 2.5],
            'take_profit_atr': [2.0, 3.0, 4.0]
        }
        return param_space
    
    def calculate_indicators(self, params):
        """Calculate all indicators with given parameters"""
        df = self.data.copy()
        
        # Price-based indicators
        df['rsi'] = TechnicalIndicators.rsi(df['close'], params['rsi_window'])
        df['sma_20'] = TechnicalIndicators.sma(df['close'], 20)
        df['ema_12'] = TechnicalIndicators.ema(df['close'], 12)
        df['ema_26'] = TechnicalIndicators.ema(df['close'], 26)
        
        # MACD
        macd, macd_signal, macd_hist = TechnicalIndicators.macd(
            df['close'], params['macd_fast'], params['macd_slow']
        )
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(
            df['close'], params['bb_window'], params['bb_std']
        )
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['bb_middle'] = bb_middle
        
        # Stochastic
        stoch_k, stoch_d = TechnicalIndicators.stochastic(
            df['high'], df['low'], df['close'], params['stoch_k']
        )
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # ATR for volatility
        df['atr'] = TechnicalIndicators.atr(
            df['high'], df['low'], df['close'], params['atr_window']
        )
        
        # Williams %R
        df['williams_r'] = TechnicalIndicators.williams_r(
            df['high'], df['low'], df['close'], 14
        )
        
        # Momentum
        df['momentum'] = TechnicalIndicators.momentum(
            df['close'], params['momentum_window']
        )
        
        # CCI
        df['cci'] = TechnicalIndicators.cci(df['high'], df['low'], df['close'])
        
        # Volume indicators
        if 'volume' in df.columns:
            df['volume_ma'] = TechnicalIndicators.sma(df['volume'], params['volume_ma'])
            df['obv'] = TechnicalIndicators.obv(df['close'], df['volume'])
        
        return df
    
    def generate_signals(self, df, params):
        """Generate buy/sell signals based on indicators"""
        signals = pd.DataFrame(index=df.index)
        signals['position'] = 0
        signals['entry_reason'] = ''
        signals['probability'] = 0.0
        
        # Long signals
        long_conditions = []
        long_reasons = []
        
        # RSI oversold
        if df['rsi'].iloc[-1] < params['rsi_oversold']:
            long_conditions.append(True)
            long_reasons.append(f"RSI oversold ({df['rsi'].iloc[-1]:.1f})")
        
        # MACD bullish crossover
        if (df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] and 
            df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]):
            long_conditions.append(True)
            long_reasons.append("MACD bullish crossover")
        
        # Price near lower Bollinger Band
        if df['close'].iloc[-1] < df['bb_lower'].iloc[-1] * 1.02:
            long_conditions.append(True)
            long_reasons.append("Price near lower BB")
        
        # Stochastic oversold
        if df['stoch_k'].iloc[-1] < 20:
            long_conditions.append(True)
            long_reasons.append(f"Stochastic oversold ({df['stoch_k'].iloc[-1]:.1f})")
        
        # Williams %R oversold
        if df['williams_r'].iloc[-1] < -80:
            long_conditions.append(True)
            long_reasons.append("Williams %R oversold")
        
        # Short signals
        short_conditions = []
        short_reasons = []
        
        # RSI overbought
        if df['rsi'].iloc[-1] > params['rsi_overbought']:
            short_conditions.append(True)
            short_reasons.append(f"RSI overbought ({df['rsi'].iloc[-1]:.1f})")
        
        # MACD bearish crossover
        if (df['macd'].iloc[-1] < df['macd_signal'].iloc[-1] and 
            df['macd'].iloc[-2] >= df['macd_signal'].iloc[-2]):
            short_conditions.append(True)
            short_reasons.append("MACD bearish crossover")
        
        # Price near upper Bollinger Band
        if df['close'].iloc[-1] > df['bb_upper'].iloc[-1] * 0.98:
            short_conditions.append(True)
            short_reasons.append("Price near upper BB")
        
        # Stochastic overbought
        if df['stoch_k'].iloc[-1] > 80:
            short_conditions.append(True)
            short_reasons.append(f"Stochastic overbought ({df['stoch_k'].iloc[-1]:.1f})")
        
        # Williams %R overbought
        if df['williams_r'].iloc[-1] > -20:
            short_conditions.append(True)
            short_reasons.append("Williams %R overbought")
        
        # Generate final signals
        if self.trade_type in ['long', 'both'] and len(long_conditions) >= 2:
            signals.iloc[-1, signals.columns.get_loc('position')] = 1
            signals.iloc[-1, signals.columns.get_loc('entry_reason')] = '; '.join(long_reasons)
            signals.iloc[-1, signals.columns.get_loc('probability')] = min(0.9, len(long_conditions) * 0.15)
        
        if self.trade_type in ['short', 'both'] and len(short_conditions) >= 2:
            signals.iloc[-1, signals.columns.get_loc('position')] = -1
            signals.iloc[-1, signals.columns.get_loc('entry_reason')] = '; '.join(short_reasons)
            signals.iloc[-1, signals.columns.get_loc('probability')] = min(0.9, len(short_conditions) * 0.15)
        
        return signals
    
    def backtest_strategy(self, params):
        """Backtest strategy with given parameters"""
        df = self.calculate_indicators(params)
        
        trades = []
        current_position = 0
        entry_price = 0
        entry_date = None
        stop_loss = 0
        take_profit = 0
        
        for i in range(50, len(df)):  # Start after warm-up period
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Generate signals
            signals = self.generate_signals(df.iloc[:i+1], params)
            signal = signals['position'].iloc[-1]
            
            # Close existing position
            if current_position != 0:
                # Check stop loss and take profit
                current_price = row['close']
                exit_triggered = False
                exit_reason = ""
                
                if current_position == 1:  # Long position
                    if current_price <= stop_loss:
                        exit_triggered = True
                        exit_reason = "Stop Loss"
                    elif current_price >= take_profit:
                        exit_triggered = True
                        exit_reason = "Take Profit"
                    elif signal == -1:
                        exit_triggered = True
                        exit_reason = "Signal Reversal"
                
                elif current_position == -1:  # Short position
                    if current_price >= stop_loss:
                        exit_triggered = True
                        exit_reason = "Stop Loss"
                    elif current_price <= take_profit:
                        exit_triggered = True
                        exit_reason = "Take Profit"
                    elif signal == 1:
                        exit_triggered = True
                        exit_reason = "Signal Reversal"
                
                if exit_triggered:
                    # Calculate PnL
                    if current_position == 1:
                        pnl = (current_price - entry_price) / entry_price * 100
                    else:
                        pnl = (entry_price - current_price) / entry_price * 100
                    
                    # Record trade
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': row.name,
                        'position': current_position,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'hold_days': (row.name - entry_date).days
                    })
                    
                    current_position = 0
            
            # Open new position
            if current_position == 0 and signal != 0:
                if (self.trade_type == 'long' and signal == 1) or \
                   (self.trade_type == 'short' and signal == -1) or \
                   (self.trade_type == 'both'):
                    
                    current_position = signal
                    entry_price = row['close']
                    entry_date = row.name
                    atr_value = row['atr']
                    
                    if signal == 1:  # Long
                        stop_loss = entry_price - (atr_value * params['stop_loss_atr'])
                        take_profit = entry_price + (atr_value * params['take_profit_atr'])
                    else:  # Short
                        stop_loss = entry_price + (atr_value * params['stop_loss_atr'])
                        take_profit = entry_price - (atr_value * params['take_profit_atr'])
        
        return trades
    
    def calculate_performance(self, trades):
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_return': 0,
                'num_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        df_trades = pd.DataFrame(trades)
        
        total_return = df_trades['pnl'].sum()
        num_trades = len(trades)
        
        winning_trades = df_trades[df_trades['pnl'] > 0]
        losing_trades = df_trades[df_trades['pnl'] < 0]
        
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate drawdown
        cumulative_pnl = df_trades['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / running_max * 100
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        # Sharpe ratio (simplified)
        returns_std = df_trades['pnl'].std() if len(df_trades) > 1 else 0
        sharpe_ratio = (df_trades['pnl'].mean() / returns_std) if returns_std > 0 else 0
        
        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate * 100,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_hold_days': df_trades['hold_days'].mean()
        }
    
    def optimize(self, search_type='random', n_iter=100):
        """Optimize strategy parameters"""
        param_space = self.get_parameter_space()
        
        if search_type == 'grid':
            param_combinations = list(ParameterGrid(param_space))
        else:  # random search
            param_combinations = list(ParameterSampler(param_space, n_iter=n_iter, random_state=42))
        
        best_params = None
        best_performance = -float('inf')
        best_trades = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, params in enumerate(param_combinations):
            if i % max(1, len(param_combinations) // 20) == 0:
                progress_bar.progress(i / len(param_combinations))
                status_text.text(f"Optimizing... {i}/{len(param_combinations)}")
            
            try:
                trades = self.backtest_strategy(params)
                performance = self.calculate_performance(trades)
                
                # Score based on total return and other factors
                score = (performance['total_return'] + 
                        performance['win_rate'] * 0.1 + 
                        performance['profit_factor'] * 10 -
                        performance['max_drawdown'] * 0.1)
                
                if score > best_performance:
                    best_performance = score
                    best_params = params
                    best_trades = trades
                    
            except Exception as e:
                continue
        
        progress_bar.progress(1.0)
        status_text.text("Optimization complete!")
        
        return best_params, best_trades

def main():
    st.title("🚀 Advanced Stock Trading Recommendation System")
    st.markdown("Upload your stock data and get AI-powered trading recommendations with backtesting")
    
    # Sidebar for parameters
    st.sidebar.header("Configuration")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Stock Data (CSV/Excel)", 
        type=['csv', 'xlsx', 'xls'],
        help="Upload file with OHLCV data"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Data loaded successfully! Shape: {raw_df.shape}")
            
            # Map columns
            df = DataProcessor.map_columns(raw_df)
            
            # Validate data
            if not DataProcessor.validate_data(df):
                return
            
            # Display column mapping
            st.subheader("📊 Column Mapping")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Columns:**")
                st.write(list(raw_df.columns))
            with col2:
                st.write("**Mapped Columns:**")
                st.write(list(df.columns))
            
            # Process data
            df = DataProcessor.process_data(df)
            
            # Display basic info
            st.subheader("📈 Data Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Date Range", f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
            with col3:
                st.metric("Max Price", f"${float(df['high'].max()):.2f}")
            with col4:
                st.metric("Min Price", f"${float(df['low'].min()):.2f}")
            
            # Display data sample
            st.subheader("📋 Data Sample")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**First 5 rows:**")
                st.dataframe(df.head())
            with col2:
                st.write("**Last 5 rows:**")
                st.dataframe(df.tail())
            
            # Plot raw data
            st.subheader("📊 Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="OHLC"
            ))
            fig.update_layout(
                title="Stock Price Chart",
                yaxis_title="Price ($)",
                xaxis_title="Date",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Exploratory Data Analysis
            if len(df) > 365:  # More than 1 year of data
                st.subheader("🔍 Exploratory Data Analysis")
                
                # Calculate returns
                df['returns'] = df['close'].pct_change()
                df['year'] = df.index.year
                df['month'] = df.index.month
                
                # Monthly returns heatmap
                monthly_returns = df.groupby(['year', 'month'])['returns'].sum().unstack()
                
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(monthly_returns * 100, annot=True, fmt='.1f', 
                           cmap='RdYlBu_r', center=0, ax=ax)
                ax.set_title('Monthly Returns Heatmap (%)')
                st.pyplot(fig)
                
                # Summary statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Return Statistics:**")
                    stats = df['returns'].describe()
                    st.dataframe(stats)
                
                with col2:
                    st.write("**Volatility Analysis:**")
                    volatility_30d = df['returns'].rolling(30).std() * np.sqrt(252) * 100
                    st.metric("30-day Volatility", f"{volatility_30d.iloc[-1]:.1f}%")
                    st.metric("Annual Volatility", f"{df['returns'].std() * np.sqrt(252) * 100:.1f}%")
            
            # Configuration options
            st.sidebar.subheader("Trading Parameters")
            
            # End date selection
            end_date = st.sidebar.date_input(
                "Select End Date for Analysis",
                value=df.index.max().date(),
                min_value=df.index.min().date(),
                max_value=df.index.max().date()
            )
            
            # Trade type
            trade_type = st.sidebar.selectbox(
                "Trade Type",
                ['both', 'long', 'short']
            )
            
            # Optimization method
            search_type = st.sidebar.selectbox(
                "Optimization Method",
                ['random', 'grid'],
                index=0
            )
            
            if st.sidebar.button("🚀 Generate Recommendations", type="primary"):
                # Filter data by end date
                analysis_df = df[df.index <= pd.to_datetime(end_date)]
                
                if len(analysis_df) < 100:
                    st.error("Need at least 100 data points for analysis")
                    return
                
                # Generate summary
                st.subheader("📝 Data Summary")
                current_price = float(analysis_df['close'].iloc[-1])
                prev_price = float(analysis_df['close'].iloc[-2])
                price_change = (current_price - prev_price) / prev_price * 100
                min_price = float(analysis_df['low'].min())
                max_price = float(analysis_df['high'].max())
                avg_volume = float(analysis_df.get('volume', pd.Series([0])).mean()) if 'volume' in analysis_df.columns else 0
                volatility = float(analysis_df['close'].std() / analysis_df['close'].mean()) if len(analysis_df) > 1 else 0
                
                summary_text = f"""
                The stock data shows a current price of ${current_price:.2f} with a recent change of {price_change:.2f}%. 
                The stock has traded between ${min_price:.2f} and ${max_price:.2f} in the analyzed period.
                Average daily volume is {avg_volume:,.0f} shares.
                The stock shows {'high' if volatility > 0.02 else 'moderate'} volatility.
                Recent price action suggests {'bullish' if price_change > 0 else 'bearish'} momentum in the short term.
                Technical analysis indicates potential {'swing trading' if volatility > 0.015 else 'trend following'} opportunities.
                """
                
                st.write(summary_text)
                
                # Initialize optimizer
                optimizer = StrategyOptimizer(analysis_df, trade_type)
                
                # Run optimization
                st.subheader("⚡ Strategy Optimization")
                with st.spinner("Optimizing strategy parameters..."):
                    best_params, best_trades = optimizer.optimize(search_type, n_iter=50)
                
                if best_params is None:
                    st.error("Optimization failed. Try different parameters.")
                    return
                
                # Display best strategy
                st.subheader("🏆 Best Strategy Parameters")
                col1, col2 = st.columns(2)
                
                with col1:
                    for key, value in list(best_params.items())[:len(best_params)//2]:
                        st.metric(key.replace('_', ' ').title(), f"{value}")
                
                with col2:
                    for key, value in list(best_params.items())[len(best_params)//2:]:
                        st.metric(key.replace('_', ' ').title(), f"{value}")
                
                # Calculate performance
                performance = optimizer.calculate_performance(best_trades)
                
                # Display backtest results
                st.subheader("📊 Backtest Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Return", f"{float(performance['total_return']):.2f}%")
                    st.metric("Number of Trades", int(performance['num_trades']))
                
                with col2:
                    st.metric("Win Rate", f"{float(performance['win_rate']):.1f}%")
                    st.metric("Winning Trades", int(performance['winning_trades']))
                
                with col3:
                    st.metric("Average Win", f"{float(performance['avg_win']):.2f}%")
                    st.metric("Losing Trades", int(performance['losing_trades']))
                
                with col4:
                    st.metric("Average Loss", f"{float(performance['avg_loss']):.2f}%")
                    st.metric("Profit Factor", f"{float(performance['profit_factor']):.2f}")
                
                # Additional metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max Drawdown", f"{float(performance['max_drawdown']):.2f}%")
                with col2:
                    st.metric("Sharpe Ratio", f"{float(performance['sharpe_ratio']):.2f}")
                with col3:
                    st.metric("Avg Hold Days", f"{float(performance['avg_hold_days']):.1f}")
                
                # Trade details
                if best_trades:
                    st.subheader("📋 Trade Details")
                    trades_df = pd.DataFrame(best_trades)
                    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date']).dt.strftime('%Y-%m-%d')
                    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date']).dt.strftime('%Y-%m-%d')
                    trades_df['position'] = trades_df['position'].map({1: 'Long', -1: 'Short'})
                    
                    # Format columns
                    display_trades = trades_df[['entry_date', 'exit_date', 'position', 
                                              'entry_price', 'exit_price', 'stop_loss', 
                                              'take_profit', 'pnl', 'exit_reason', 'hold_days']]
                    
                    display_trades.columns = ['Entry Date', 'Exit Date', 'Position', 
                                            'Entry Price', 'Exit Price', 'Stop Loss',
                                            'Take Profit', 'P&L (%)', 'Exit Reason', 'Hold Days']
                    
                    # Format numeric columns
                    numeric_cols = ['Entry Price', 'Exit Price', 'Stop Loss', 'Take Profit']
                    for col in numeric_cols:
                        display_trades[col] = display_trades[col].round(2)
                    display_trades['P&L (%)'] = display_trades['P&L (%)'].round(2)
                    
                    st.dataframe(display_trades, use_container_width=True)
                
                # Generate live recommendation
                st.subheader("🎯 Live Recommendation")
                
                # Use full data for live recommendation
                live_df = optimizer.calculate_indicators(best_params)
                live_signals = optimizer.generate_signals(live_df, best_params)
                
                current_signal = int(live_signals['position'].iloc[-1])
                current_reason = str(live_signals['entry_reason'].iloc[-1])
                current_probability = float(live_signals['probability'].iloc[-1])
                
                if current_signal != 0:
                    # Calculate levels
                    current_price = float(live_df['close'].iloc[-1])
                    current_atr = float(live_df['atr'].iloc[-1])
                    
                    if current_signal == 1:  # Long
                        position_type = "🟢 LONG"
                        stop_loss = current_price - (current_atr * best_params['stop_loss_atr'])
                        take_profit = current_price + (current_atr * best_params['take_profit_atr'])
                    else:  # Short
                        position_type = "🔴 SHORT"
                        stop_loss = current_price + (current_atr * best_params['stop_loss_atr'])
                        take_profit = current_price - (current_atr * best_params['take_profit_atr'])
                    
                    # Calculate risk-reward ratio
                    if current_signal == 1:
                        risk = current_price - stop_loss
                        reward = take_profit - current_price
                    else:
                        risk = stop_loss - current_price
                        reward = current_price - take_profit
                    
                    risk_reward = reward / risk if risk > 0 else 0
                    
                    # Display recommendation
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.success(f"**{position_type} SIGNAL DETECTED**")
                        st.metric("Entry Price", f"${current_price:.2f}")
                        st.metric("Stop Loss", f"${stop_loss:.2f}")
                        st.metric("Take Profit", f"${take_profit:.2f}")
                        st.metric("Risk/Reward Ratio", f"1:{risk_reward:.2f}")
                    
                    with col2:
                        st.metric("Probability of Profit", f"{current_probability*100:.1f}%")
                        st.metric("Entry Date/Time", f"{live_df.index[-1].strftime('%Y-%m-%d %H:%M')}")
                        
                        # Next trading day
                        next_day = live_df.index[-1] + pd.Timedelta(days=1)
                        while next_day.weekday() > 4:  # Skip weekends
                            next_day += pd.Timedelta(days=1)
                        st.metric("Recommended Entry Date", next_day.strftime('%Y-%m-%d'))
                        
                        # Position size calculation (2% risk)
                        account_risk = 0.02  # 2% risk
                        if current_signal == 1:
                            price_risk = current_price - stop_loss
                        else:
                            price_risk = stop_loss - current_price
                        
                        position_size_pct = (account_risk / (price_risk / current_price)) * 100 if price_risk > 0 else 0
                        st.metric("Suggested Position Size", f"{min(position_size_pct, 10):.1f}% of capital")
                    
                    st.write(f"**Entry Logic:** {current_reason}")
                    
                    # Technical analysis summary
                    st.write("**Technical Analysis:**")
                    rsi_current = float(live_df['rsi'].iloc[-1])
                    macd_current = float(live_df['macd'].iloc[-1])
                    macd_signal_current = float(live_df['macd_signal'].iloc[-1])
                    bb_upper_current = float(live_df['bb_upper'].iloc[-1])
                    bb_lower_current = float(live_df['bb_lower'].iloc[-1])
                    bb_position = (current_price - bb_lower_current) / (bb_upper_current - bb_lower_current)
                    
                    analysis_text = f"""
                    - RSI: {rsi_current:.1f} ({'Oversold' if rsi_current < 30 else 'Overbought' if rsi_current > 70 else 'Neutral'})
                    - MACD: {macd_current:.3f} ({'Above signal' if macd_current > macd_signal_current else 'Below signal'})
                    - Bollinger Bands: {bb_position:.1%} position ({'Lower band area' if bb_position < 0.2 else 'Upper band area' if bb_position > 0.8 else 'Middle range'})
                    - Volatility (ATR): ${current_atr:.2f}
                    """
                    st.write(analysis_text)
                    
                else:
                    st.info("🔍 **NO SIGNAL** - Wait for better entry opportunity")
                    st.write("Current market conditions do not meet the strategy criteria for entry.")
                    
                    # Show current levels anyway
                    current_price = float(live_df['close'].iloc[-1])
                    rsi_current = float(live_df['rsi'].iloc[-1])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                        st.metric("Current RSI", f"{rsi_current:.1f}")
                    with col2:
                        st.metric("MACD", f"{float(live_df['macd'].iloc[-1]):.3f}")
                        st.metric("Stochastic %K", f"{float(live_df['stoch_k'].iloc[-1]):.1f}")
                
                # Strategy summary
                st.subheader("📈 Strategy Summary")
                
                # Calculate buy and hold return for comparison
                buy_hold_return = ((float(analysis_df['close'].iloc[-1]) - float(analysis_df['close'].iloc[0])) / float(analysis_df['close'].iloc[0])) * 100
                
                strategy_summary = f"""
                **Backtest Analysis Summary:**
                The optimized strategy generated a total return of {float(performance['total_return']):.2f}% compared to buy-and-hold return of {buy_hold_return:.2f}%.
                The strategy executed {int(performance['num_trades'])} trades with a win rate of {float(performance['win_rate']):.1f}%.
                
                **Strategy Performance:**
                - {'Outperformed' if float(performance['total_return']) > buy_hold_return else 'Underperformed'} buy-and-hold by {abs(float(performance['total_return']) - buy_hold_return):.2f}%
                - Average holding period: {float(performance['avg_hold_days']):.1f} days
                - Risk-adjusted returns (Sharpe): {float(performance['sharpe_ratio']):.2f}
                - Maximum drawdown: {float(performance['max_drawdown']):.2f}%
                
                **Live Trading Recommendation:**
                {'A ' + ('LONG' if current_signal == 1 else 'SHORT') + ' position is recommended' if current_signal != 0 else 'Wait for better entry conditions'}.
                {'The strategy shows ' + str(int(current_probability*100)) + '% probability of profit based on current technical indicators.' if current_signal != 0 else 'Monitor key levels and wait for signal confirmation.'}
                
                **Risk Management:**
                - Use stop-loss orders at calculated levels
                - Position size should not exceed 2-5% risk per trade
                - Monitor market conditions and adjust if needed
                - Consider market volatility and news events
                """
                
                st.write(strategy_summary)
                
                # Performance chart
                if best_trades:
                    st.subheader("📊 Equity Curve")
                    
                    trades_df = pd.DataFrame(best_trades)
                    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trades_df['exit_date'],
                        y=trades_df['cumulative_pnl'],
                        mode='lines+markers',
                        name='Strategy Return',
                        line=dict(color='green', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Strategy Equity Curve",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Return (%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                if best_trades:
                    st.subheader("💾 Download Results")
                    
                    # Prepare download data
                    results_summary = {
                        'Strategy Parameters': best_params,
                        'Performance Metrics': performance,
                        'Live Recommendation': {
                            'Signal': 'Long' if current_signal == 1 else 'Short' if current_signal == -1 else 'No Signal',
                            'Entry Price': f"${live_df['close'].iloc[-1]:.2f}" if current_signal != 0 else 'N/A',
                            'Stop Loss': f"${stop_loss:.2f}" if current_signal != 0 else 'N/A',
                            'Take Profit': f"${take_profit:.2f}" if current_signal != 0 else 'N/A',
                            'Probability': f"{current_probability*100:.1f}%" if current_signal != 0 else 'N/A',
                            'Reason': current_reason if current_signal != 0 else 'No signal conditions met'
                        }
                    }
                    
                    # Convert trades to CSV
                    trades_csv = pd.DataFrame(best_trades).to_csv(index=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download Trade History (CSV)",
                            data=trades_csv,
                            file_name=f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        import json
                        results_json = json.dumps(results_summary, indent=2, default=str)
                        st.download_button(
                            label="📥 Download Strategy Report (JSON)",
                            data=results_json,
                            file_name=f"strategy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.error("Please check your data format and try again.")
    
    else:
        # Instructions
        st.info("👆 Please upload a CSV or Excel file with stock data to get started")
        
        st.markdown("""
        ### 📝 Data Format Requirements:
        
        Your file should contain columns with OHLCV data. The system will automatically detect and map column names like:
        - **Open**: 'open', 'open price', 'price open', 'o'
        - **High**: 'high', 'high price', 'price high', 'h'  
        - **Low**: 'low', 'low price', 'price low', 'l'
        - **Close**: 'close', 'close price', 'price close', 'c'
        - **Volume**: 'volume', 'vol', 'v', 'trading volume' (optional)
        
        ### 🚀 Features:
        
        ✅ **Automated Analysis**: Upload and get instant recommendations  
        ✅ **10+ Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, ATR, Williams %R, Momentum, CCI, OBV  
        ✅ **Strategy Optimization**: Grid search or Random search for best parameters  
        ✅ **Risk Management**: Automatic stop-loss and take-profit calculations  
        ✅ **Backtesting**: Historical performance analysis with detailed trade log  
        ✅ **Live Recommendations**: Real-time entry signals with probability scores  
        ✅ **Visual Analytics**: Interactive charts and performance metrics  
        ✅ **Export Results**: Download trade history and strategy reports  
        
        ### 📊 Example Data Structure:
        ```
        Date       | Open  | High  | Low   | Close | Volume
        2024-01-01 | 100.0 | 102.5 | 99.5  | 101.2 | 1000000
        2024-01-02 | 101.2 | 103.0 | 100.8 | 102.1 | 1200000
        ...
        ```
        """)

if __name__ == "__main__":
    main()
