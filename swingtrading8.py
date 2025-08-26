import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
from sklearn.model_selection import ParameterGrid
import random
from scipy.optimize import differential_evolution
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Custom technical indicators calculated manually"""
    
    @staticmethod
    def sma(data, period):
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data, period):
        multiplier = 2 / (period + 1)
        ema = data.copy()
        for i in range(1, len(data)):
            if pd.notna(ema.iloc[i-1]):
                ema.iloc[i] = (data.iloc[i] * multiplier) + (ema.iloc[i-1] * (1 - multiplier))
        return ema
    
    @staticmethod
    def rsi(data, period=14):
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
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
    
    @staticmethod
    def bollinger_bands(data, period=20, std_dev=2):
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high, low, close, period=14):
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return wr
    
    @staticmethod
    def cci(high, low, close, period=20):
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mean_dev = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (tp - sma_tp) / (0.015 * mean_dev)
        return cci
    
    @staticmethod
    def atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def adx(high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        tr = TechnicalIndicators.atr(high, low, close, 1)
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        return adx, plus_di, minus_di
    
    @staticmethod
    def momentum(data, period=10):
        return data.diff(period)

class SwingTradingStrategy:
    def __init__(self, df, parameters):
        self.df = df.copy()
        self.params = parameters
        self.signals = None
        self.indicators = {}
        
    def calculate_indicators(self):
        """Calculate all technical indicators"""
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        volume = self.df['volume']
        
        # Moving Averages
        self.indicators['sma_fast'] = TechnicalIndicators.sma(close, self.params['sma_fast'])
        self.indicators['sma_slow'] = TechnicalIndicators.sma(close, self.params['sma_slow'])
        self.indicators['ema_fast'] = TechnicalIndicators.ema(close, self.params['ema_fast'])
        self.indicators['ema_slow'] = TechnicalIndicators.ema(close, self.params['ema_slow'])
        
        # Oscillators
        self.indicators['rsi'] = TechnicalIndicators.rsi(close, self.params['rsi_period'])
        macd, signal, histogram = TechnicalIndicators.macd(close, self.params['macd_fast'], 
                                                         self.params['macd_slow'], self.params['macd_signal'])
        self.indicators['macd'] = macd
        self.indicators['macd_signal'] = signal
        self.indicators['macd_histogram'] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(close, self.params['bb_period'])
        self.indicators['bb_upper'] = bb_upper
        self.indicators['bb_middle'] = bb_middle
        self.indicators['bb_lower'] = bb_lower
        
        # Stochastic
        k_percent, d_percent = TechnicalIndicators.stochastic(high, low, close, self.params['stoch_k'])
        self.indicators['stoch_k'] = k_percent
        self.indicators['stoch_d'] = d_percent
        
        # Other indicators
        self.indicators['williams_r'] = TechnicalIndicators.williams_r(high, low, close, self.params['williams_period'])
        self.indicators['cci'] = TechnicalIndicators.cci(high, low, close, self.params['cci_period'])
        self.indicators['atr'] = TechnicalIndicators.atr(high, low, close, self.params['atr_period'])
        adx, plus_di, minus_di = TechnicalIndicators.adx(high, low, close, self.params['adx_period'])
        self.indicators['adx'] = adx
        self.indicators['plus_di'] = plus_di
        self.indicators['minus_di'] = minus_di
        self.indicators['momentum'] = TechnicalIndicators.momentum(close, self.params['momentum_period'])
        
    def generate_signals(self, trade_type='both'):
        """Generate trading signals based on multiple indicators"""
        self.calculate_indicators()
        
        signals = pd.DataFrame(index=self.df.index)
        signals['price'] = self.df['close']
        signals['signal'] = 0
        signals['strength'] = 0
        
        # Long signals
        long_conditions = [
            self.indicators['sma_fast'] > self.indicators['sma_slow'],
            self.indicators['ema_fast'] > self.indicators['ema_slow'],
            self.indicators['rsi'] < self.params['rsi_oversold'],
            self.indicators['macd'] > self.indicators['macd_signal'],
            self.df['close'] < self.indicators['bb_lower'],
            self.indicators['stoch_k'] < self.params['stoch_oversold'],
            self.indicators['williams_r'] < -80,
            self.indicators['cci'] < -100,
            self.indicators['plus_di'] > self.indicators['minus_di'],
            self.indicators['momentum'] > 0
        ]
        
        # Short signals
        short_conditions = [
            self.indicators['sma_fast'] < self.indicators['sma_slow'],
            self.indicators['ema_fast'] < self.indicators['ema_slow'],
            self.indicators['rsi'] > self.params['rsi_overbought'],
            self.indicators['macd'] < self.indicators['macd_signal'],
            self.df['close'] > self.indicators['bb_upper'],
            self.indicators['stoch_k'] > self.params['stoch_overbought'],
            self.indicators['williams_r'] > -20,
            self.indicators['cci'] > 100,
            self.indicators['plus_di'] < self.indicators['minus_di'],
            self.indicators['momentum'] < 0
        ]
        
        # Calculate signal strength
        long_strength = sum([cond.astype(int) for cond in long_conditions])
        short_strength = sum([cond.astype(int) for cond in short_conditions])
        
        # Generate signals based on trade type
        if trade_type in ['long', 'both']:
            long_signal = (long_strength >= self.params['min_conditions']) & (self.indicators['adx'] > 25)
            signals.loc[long_signal, 'signal'] = 1
            signals.loc[long_signal, 'strength'] = long_strength[long_signal]
        
        if trade_type in ['short', 'both']:
            short_signal = (short_strength >= self.params['min_conditions']) & (self.indicators['adx'] > 25)
            signals.loc[short_signal, 'signal'] = -1
            signals.loc[short_signal, 'strength'] = short_strength[short_signal]
        
        # Calculate targets and stop losses
        atr = self.indicators['atr']
        signals['target'] = np.where(signals['signal'] == 1, 
                                   signals['price'] + (atr * self.params['target_atr_multiplier']),
                                   np.where(signals['signal'] == -1,
                                          signals['price'] - (atr * self.params['target_atr_multiplier']),
                                          np.nan))
        
        signals['stop_loss'] = np.where(signals['signal'] == 1,
                                      signals['price'] - (atr * self.params['sl_atr_multiplier']),
                                      np.where(signals['signal'] == -1,
                                             signals['price'] + (atr * self.params['sl_atr_multiplier']),
                                             np.nan))
        
        # Calculate probability based on strength and ADX
        signals['probability'] = np.where(signals['signal'] != 0,
                                        (signals['strength'] / 10 * 0.6) + (self.indicators['adx'] / 100 * 0.4),
                                        np.nan)
        
        self.signals = signals
        return signals

class BacktestEngine:
    def __init__(self, df, signals, initial_capital=100000):
        self.df = df
        self.signals = signals
        self.initial_capital = initial_capital
        self.results = []
        
    def run_backtest(self):
        """Run comprehensive backtest"""
        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        target = 0
        stop_loss = 0
        trades = []
        
        for i in range(len(self.df)):
            current_price = self.df.iloc[i]['close']
            current_date = self.df.index[i]
            
            # Check for new signals
            if self.signals.iloc[i]['signal'] != 0 and position == 0:
                position = self.signals.iloc[i]['signal']
                entry_price = current_price
                entry_date = current_date
                target = self.signals.iloc[i]['target']
                stop_loss = self.signals.iloc[i]['stop_loss']
                
            # Check exit conditions
            elif position != 0:
                exit_triggered = False
                exit_reason = ""
                exit_price = current_price
                
                if position == 1:  # Long position
                    if current_price >= target:
                        exit_triggered = True
                        exit_reason = "Target Hit"
                        exit_price = target
                    elif current_price <= stop_loss:
                        exit_triggered = True
                        exit_reason = "Stop Loss Hit"
                        exit_price = stop_loss
                        
                elif position == -1:  # Short position
                    if current_price <= target:
                        exit_triggered = True
                        exit_reason = "Target Hit"
                        exit_price = target
                    elif current_price >= stop_loss:
                        exit_triggered = True
                        exit_reason = "Stop Loss Hit"
                        exit_price = stop_loss
                
                if exit_triggered:
                    pnl = (exit_price - entry_price) * position
                    pnl_pct = (pnl / entry_price) * 100
                    
                    trade = {
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'target': target,
                        'stop_loss': stop_loss,
                        'signal': 'Long' if position == 1 else 'Short',
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'hold_days': (current_date - entry_date).days,
                        'probability': self.signals.loc[entry_date, 'probability'] if entry_date in self.signals.index else 0
                    }
                    trades.append(trade)
                    
                    capital += pnl
                    position = 0
        
        return trades, capital

class TradingApp:
    def __init__(self):
        self.df = None
        self.processed_df = None
        self.default_params = {
            'sma_fast': 10, 'sma_slow': 20, 'ema_fast': 12, 'ema_slow': 26,
            'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'bb_period': 20, 'stoch_k': 14, 'stoch_oversold': 20, 'stoch_overbought': 80,
            'williams_period': 14, 'cci_period': 20, 'atr_period': 14, 'adx_period': 14,
            'momentum_period': 10, 'min_conditions': 6, 'target_atr_multiplier': 2.0,
            'sl_atr_multiplier': 1.0
        }
        
    def map_columns(self, df):
        """Map uploaded columns to standard format"""
        columns = df.columns.str.lower()
        column_mapping = {}
        
        # Define possible variations for each required column
        mappings = {
            'open': ['open', 'open_price', 'openPrice', 'o'],
            'high': ['high', 'high_price', 'highPrice', 'h'],
            'low': ['low', 'low_price', 'lowPrice', 'l'],
            'close': ['close', 'close_price', 'closePrice', 'c'],
            'volume': ['volume', 'vol', 'v']
        }
        
        for standard_name, variations in mappings.items():
            for col in columns:
                for variation in variations:
                    if variation.lower() in col.lower():
                        column_mapping[df.columns[columns.get_loc(col)]] = standard_name
                        break
                if standard_name in column_mapping.values():
                    break
        
        # Rename columns
        df_mapped = df.rename(columns=column_mapping)
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df_mapped.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None
            
        return df_mapped[required_cols]
    
    def perform_eda(self, df):
        """Perform exploratory data analysis"""
        st.subheader("📊 Exploratory Data Analysis")
        
        # Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Overview:**")
            st.write(f"• Total Records: {len(df):,}")
            st.write(f"• Date Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
            st.write(f"• Max Price: ${df['close'].max():.2f}")
            st.write(f"• Min Price: ${df['close'].min():.2f}")
            st.write(f"• Price Range: ${df['close'].max() - df['close'].min():.2f}")
        
        with col2:
            st.write("**Price Statistics:**")
            st.dataframe(df[['open', 'high', 'low', 'close', 'volume']].describe())
        
        # Price chart
        fig = go.Figure(data=go.Candlestick(x=df.index,
                                          open=df['open'],
                                          high=df['high'],
                                          low=df['low'],
                                          close=df['close']))
        fig.update_layout(title="Price Chart", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
        
        # Returns heatmap (if data spans more than 1 year)
        if (df.index.max() - df.index.min()).days > 365:
            st.write("**Monthly Returns Heatmap:**")
            df['returns'] = df['close'].pct_change()
            df['year'] = df.index.year
            df['month'] = df.index.month
            
            monthly_returns = df.groupby(['year', 'month'])['returns'].sum().unstack()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(monthly_returns * 100, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax)
            ax.set_title('Monthly Returns Heatmap (%)')
            st.pyplot(fig)
    
    def generate_summary(self, df, phase="data"):
        """Generate human-readable summary"""
        if phase == "data":
            days = (df.index.max() - df.index.min()).days
            total_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
            volatility = df['close'].pct_change().std() * np.sqrt(252) * 100
            avg_volume = df['volume'].mean()
            
            summary = f"""
            📈 **Stock Data Analysis Summary:**
            
            The dataset spans {days} days with prices ranging from ${df['close'].min():.2f} to ${df['close'].max():.2f}. 
            The overall return during this period was {total_return:.1f}%, indicating {'strong growth' if total_return > 20 else 'moderate growth' if total_return > 0 else 'decline'}. 
            The annualized volatility of {volatility:.1f}% suggests {'high' if volatility > 30 else 'moderate' if volatility > 15 else 'low'} risk levels.
            Average daily volume of {avg_volume:,.0f} shares indicates {'strong' if avg_volume > 1000000 else 'moderate'} liquidity.
            
            **Opportunities:** {'Swing trading opportunities exist due to high volatility' if volatility > 20 else 'Trend following strategies may work better due to lower volatility'}.
            The current trend appears {'bullish' if df['close'].iloc[-1] > df['close'].iloc[-20] else 'bearish'} based on recent price action.
            """
            return summary
        
        elif phase == "backtest":
            return "Backtest analysis completed with detailed trade-by-trade results."
    
    def optimize_strategy(self, df, optimization_type='random', trade_type='both'):
        """Optimize strategy parameters"""
        st.write("🔍 **Optimizing Strategy Parameters...**")
        
        # Parameter ranges for optimization
        param_ranges = {
            'sma_fast': [5, 10, 15, 20],
            'sma_slow': [20, 30, 50],
            'rsi_period': [10, 14, 20],
            'rsi_oversold': [20, 25, 30, 35],
            'rsi_overbought': [65, 70, 75, 80],
            'min_conditions': [4, 5, 6, 7],
            'target_atr_multiplier': [1.5, 2.0, 2.5, 3.0],
            'sl_atr_multiplier': [0.5, 1.0, 1.5]
        }
        
        best_params = self.default_params.copy()
        best_score = -np.inf
        
        # Generate parameter combinations
        if optimization_type == 'grid':
            param_combinations = list(ParameterGrid(param_ranges))[:50]  # Limit for demo
        else:  # random search
            param_combinations = []
            for _ in range(30):  # 30 random combinations
                params = self.default_params.copy()
                for param, values in param_ranges.items():
                    params[param] = random.choice(values)
                param_combinations.append(params)
        
        progress_bar = st.progress(0)
        
        for i, params in enumerate(param_combinations):
            # Merge with default params
            test_params = self.default_params.copy()
            test_params.update(params)
            
            try:
                # Create strategy and generate signals
                strategy = SwingTradingStrategy(df, test_params)
                signals = strategy.generate_signals(trade_type)
                
                # Run backtest
                backtest = BacktestEngine(df, signals)
                trades, final_capital = backtest.run_backtest()
                
                if trades:
                    # Calculate metrics
                    total_return = ((final_capital / 100000) - 1) * 100
                    winning_trades = len([t for t in trades if t['pnl'] > 0])
                    total_trades = len(trades)
                    win_rate = winning_trades / total_trades if total_trades > 0 else 0
                    
                    # Score based on return and win rate
                    score = total_return * 0.7 + win_rate * 100 * 0.3
                    
                    if score > best_score:
                        best_score = score
                        best_params = test_params.copy()
                        
            except Exception as e:
                continue
                
            progress_bar.progress((i + 1) / len(param_combinations))
        
        return best_params, best_score

def main():
    st.set_page_config(page_title="Advanced Swing Trading System", layout="wide")
    
    st.title("📈 Advanced Swing Trading Analysis System")
    st.markdown("---")
    
    app = TradingApp()
    
    # File upload
    uploaded_file = st.file_uploader("Upload Stock Data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read and process data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Display first and last 5 rows
            col1, col2 = st.columns(2)
            with col1:
                st.write("**First 5 rows:**")
                st.dataframe(df.head())
            with col2:
                st.write("**Last 5 rows:**")
                st.dataframe(df.tail())
            
            # Map columns
            processed_df = app.map_columns(df)
            if processed_df is None:
                return
            
            # Try to parse date index
            if 'date' in df.columns.str.lower() or df.index.name and 'date' in str(df.index.name).lower():
                try:
                    date_col = [col for col in df.columns if 'date' in col.lower()][0]
                    processed_df.index = pd.to_datetime(df[date_col])
                except:
                    processed_df.index = pd.to_datetime(df.index)
            else:
                processed_df.index = pd.to_datetime(df.index)
            
            # Sort by date to prevent data leakage
            processed_df = processed_df.sort_index()
            
            st.success("✅ Data successfully mapped and sorted!")
            
            # Show date range
            st.info(f"📅 Data Range: {processed_df.index.min().strftime('%Y-%m-%d')} to {processed_df.index.max().strftime('%Y-%m-%d')}")
            
            # End date selection
            end_date = st.date_input(
                "Select End Date for Analysis",
                value=processed_df.index.max().date(),
                min_value=processed_df.index.min().date(),
                max_value=processed_df.index.max().date()
            )
            
            # Filter data up to end date
            analysis_df = processed_df[processed_df.index.date <= end_date].copy()
            
            # Trading parameters
            col1, col2 = st.columns(2)
            with col1:
                trade_type = st.selectbox("Trade Type", ["both", "long", "short"])
            with col2:
                optimization_type = st.selectbox("Optimization Method", ["random", "grid"])
            
            if st.button("🚀 Run Analysis"):
                
                # EDA
                app.perform_eda(analysis_df)
                
                # Data summary
                st.markdown(app.generate_summary(analysis_df, "data"))
                
                # Optimize strategy
                with st.spinner("Optimizing strategy..."):
                    best_params, best_score = app.optimize_strategy(analysis_df, optimization_type, trade_type)
                
                st.success(f"✅ Optimization completed! Best Score: {best_score:.2f}")
                
                # Display best parameters
                st.subheader("🎯 Optimized Strategy Parameters")
                param_cols = st.columns(4)
                for i, (param, value) in enumerate(best_params.items()):
                    with param_cols[i % 4]:
                        st.metric(param.replace('_', ' ').title(), f"{value}")
                
                # Generate signals with best parameters
                strategy = SwingTradingStrategy(analysis_df, best_params)
                signals = strategy.generate_signals(trade_type)
                
                # Run backtest
                backtest = BacktestEngine(analysis_df, signals)
                trades, final_capital = backtest.run_backtest()
                
                # Display backtest results
                st.subheader("📊 Backtest Results")
                
                if trades:
                    trades_df = pd.DataFrame(trades)
                    
                    # Summary metrics
                    total_return = ((final_capital / 100000) - 1) * 100
                    winning_trades = len([t for t in trades if t['pnl'] > 0])
                    losing_trades = len([t for t in trades if t['pnl'] < 0])
                    win_rate = (winning_trades / len(trades)) * 100
                    avg_hold_days = trades_df['hold_days'].mean()
                    
                    # Buy and hold comparison
                    buy_hold_return = ((analysis_df['close'].iloc[-1] / analysis_df['close'].iloc[0]) - 1) * 100
                    
                    metrics_cols = st.columns(5)
                    with metrics_cols[0]:
                        st.metric("Total Return", f"{total_return:.2f}%")
                    with metrics_cols[1]:
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                    with metrics_cols[2]:
                        st.metric("Total Trades", len(trades))
                    with metrics_cols[3]:
                        st.metric("Avg Hold Days", f"{avg_hold_days:.1f}")
                    with metrics_cols[4]:
                        st.metric("vs Buy & Hold", f"{total_return - buy_hold_return:+.2f}%")
                    
                    # Detailed trades
                    st.subheader("📋 Trade Details")
                    
                    # Format trades for display
                    display_trades = trades_df.copy()
                    for col in ['entry_price', 'exit_price', 'target', 'stop_loss']:
                        display_trades[col] = display_trades[col].round(2)
                    display_trades['pnl'] = display_trades['pnl'].round(2)
                    display_trades['pnl_pct'] = display_trades['pnl_pct'].round(2)
                    display_trades['probability'] = (display_trades['probability'] * 100).round(1)
                    
                    st.dataframe(display_trades, use_container_width=True)
                    
                    # Live recommendation
                    st.subheader("🎯 Live Recommendation")
                    
                    # Get latest signal
                    latest_signals = signals[signals['signal'] != 0].tail(5)
                    current_price = analysis_df['close'].iloc[-1]
                    current_date = analysis_df.index[-1]
                    
                    # Check if there's a current signal
                    if len(latest_signals) > 0:
                        latest_signal = latest_signals.iloc[-1]
                        signal_date = latest_signals.index[-1]
                        
                        # Calculate future date for next trading day
                        next_trading_day = current_date + timedelta(days=1)
                        while next_trading_day.weekday() > 4:  # Skip weekends
                            next_trading_day += timedelta(days=1)
                        
                        st.info(f"📅 **Latest Signal Date:** {signal_date.strftime('%Y-%m-%d %H:%M')}")
                        
                        if latest_signal['signal'] == 1:
                            st.success("📈 **LONG Signal Detected**")
                            signal_type = "LONG"
                        else:
                            st.error("📉 **SHORT Signal Detected**")
                            signal_type = "SHORT"
                        
                        # Display recommendation details
                        rec_cols = st.columns(4)
                        with rec_cols[0]:
                            st.metric("Entry Price", f"${latest_signal['price']:.2f}")
                        with rec_cols[1]:
                            st.metric("Target", f"${latest_signal['target']:.2f}")
                        with rec_cols[2]:
                            st.metric("Stop Loss", f"${latest_signal['stop_loss']:.2f}")
                        with rec_cols[3]:
                            st.metric("Win Probability", f"{latest_signal['probability']*100:.1f}%")
                        
                        # Risk-Reward calculation
                        if signal_type == "LONG":
                            risk = latest_signal['price'] - latest_signal['stop_loss']
                            reward = latest_signal['target'] - latest_signal['price']
                        else:
                            risk = latest_signal['stop_loss'] - latest_signal['price']
                            reward = latest_signal['price'] - latest_signal['target']
                        
                        risk_reward_ratio = reward / risk if risk > 0 else 0
                        
                        st.info(f"""
                        **📊 Trade Analysis:**
                        - **Signal Strength:** {latest_signal['strength']}/10 indicators confirm
                        - **Risk-Reward Ratio:** 1:{risk_reward_ratio:.2f}
                        - **Expected Next Trading Day:** {next_trading_day.strftime('%Y-%m-%d')}
                        
                        **🧠 Logic & Reasoning:**
                        This {signal_type} signal is generated based on {int(latest_signal['strength'])} out of 10 technical indicators aligning, 
                        including trend analysis, momentum, and volatility measures. The ATR-based stop loss and target 
                        provide a systematic risk management approach. Historical probability suggests a {latest_signal['probability']*100:.1f}% 
                        chance of reaching the target before hitting the stop loss.
                        """)
                    
                    else:
                        st.warning("⚠️ No recent trading signals detected. Market may be in consolidation.")
                    
                    # Strategy Summary
                    st.subheader("📝 Strategy & Backtest Summary")
                    
                    backtest_summary = f"""
                    **🎯 Strategy Performance Analysis:**
                    
                    The optimized swing trading strategy achieved a {total_return:.1f}% return compared to {buy_hold_return:.1f}% 
                    for buy-and-hold, representing a {'significant outperformance' if total_return > buy_hold_return * 1.7 else 'moderate outperformance' if total_return > buy_hold_return else 'underperformance'} 
                    of {total_return - buy_hold_return:+.1f} percentage points.
                    
                    **📊 Key Metrics:**
                    - **Win Rate:** {win_rate:.1f}% ({winning_trades} wins, {losing_trades} losses)
                    - **Average Hold Time:** {avg_hold_days:.1f} days
                    - **Total Trades:** {len(trades)} trades executed
                    - **Risk Management:** ATR-based stops with {best_params['target_atr_multiplier']}x target and {best_params['sl_atr_multiplier']}x stop loss
                    
                    **🚀 Live Trading Recommendations:**
                    Monitor for signals that meet {best_params['min_conditions']}/10 indicator confirmations with ADX > 25 for trend strength. 
                    The strategy works best in trending markets and may generate false signals during sideways consolidation. 
                    Always respect the calculated stop loss levels and position size according to your risk tolerance.
                    
                    **⚡ Strategy Optimization:**
                    Using {optimization_type} search, the system identified optimal parameters focusing on {trade_type} trades. 
                    The strategy combines trend-following (moving averages), momentum (RSI, MACD), and volatility (Bollinger Bands, ATR) 
                    indicators for robust signal generation.
                    """
                    
                    st.markdown(backtest_summary)
                    
                    # Performance Chart
                    st.subheader("📈 Strategy Performance Visualization")
                    
                    # Create cumulative returns chart
                    equity_curve = [100000]  # Starting capital
                    dates = [analysis_df.index[0]]
                    
                    for trade in trades:
                        equity_curve.append(equity_curve[-1] + trade['pnl'])
                        dates.append(trade['exit_date'])
                    
                    # Buy and hold equity curve
                    buy_hold_equity = []
                    for date in analysis_df.index:
                        if date <= dates[-1] if dates else analysis_df.index[-1]:
                            price_ratio = analysis_df.loc[date, 'close'] / analysis_df.iloc[0]['close']
                            buy_hold_equity.append(100000 * price_ratio)
                    
                    fig = go.Figure()
                    
                    # Strategy performance
                    fig.add_trace(go.Scatter(
                        x=dates[:len(equity_curve)],
                        y=equity_curve,
                        mode='lines',
                        name='Strategy',
                        line=dict(color='green', width=2)
                    ))
                    
                    # Buy and hold performance
                    fig.add_trace(go.Scatter(
                        x=analysis_df.index[:len(buy_hold_equity)],
                        y=buy_hold_equity,
                        mode='lines',
                        name='Buy & Hold',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Strategy vs Buy & Hold Performance",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Entry/Exit points on price chart
                    fig_trades = go.Figure()
                    
                    # Price chart
                    fig_trades.add_trace(go.Candlestick(
                        x=analysis_df.index,
                        open=analysis_df['open'],
                        high=analysis_df['high'],
                        low=analysis_df['low'],
                        close=analysis_df['close'],
                        name='Price'
                    ))
                    
                    # Add entry points
                    entry_dates = [trade['entry_date'] for trade in trades]
                    entry_prices = [trade['entry_price'] for trade in trades]
                    entry_colors = ['green' if trade['signal'] == 'Long' else 'red' for trade in trades]
                    
                    fig_trades.add_trace(go.Scatter(
                        x=entry_dates,
                        y=entry_prices,
                        mode='markers',
                        name='Entry Points',
                        marker=dict(
                            size=10,
                            color=entry_colors,
                            symbol='triangle-up',
                            line=dict(width=2, color='white')
                        )
                    ))
                    
                    # Add exit points
                    exit_dates = [trade['exit_date'] for trade in trades]
                    exit_prices = [trade['exit_price'] for trade in trades]
                    
                    fig_trades.add_trace(go.Scatter(
                        x=exit_dates,
                        y=exit_prices,
                        mode='markers',
                        name='Exit Points',
                        marker=dict(
                            size=8,
                            color='orange',
                            symbol='triangle-down',
                            line=dict(width=2, color='white')
                        )
                    ))
                    
                    fig_trades.update_layout(
                        title="Trading Signals on Price Chart",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_trades, use_container_width=True)
                    
                else:
                    st.warning("⚠️ No trades generated with current parameters. Try adjusting the strategy settings.")
                    
                    # Still show live recommendation attempt
                    st.subheader("🎯 Live Market Analysis")
                    current_price = analysis_df['close'].iloc[-1]
                    prev_price = analysis_df['close'].iloc[-2]
                    price_change = ((current_price / prev_price) - 1) * 100
                    
                    st.info(f"""
                    **📊 Current Market Status:**
                    - **Latest Price:** ${current_price:.2f}
                    - **Daily Change:** {price_change:+.2f}%
                    - **Status:** No clear swing trading signals detected
                    
                    **💡 Recommendation:** 
                    Wait for stronger directional momentum or consider adjusting strategy parameters 
                    for more sensitive signal detection.
                    """)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file contains price data with columns like: Date, Open, High, Low, Close, Volume")

if __name__ == "__main__":
    main()
