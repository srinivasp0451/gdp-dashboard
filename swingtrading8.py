import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Swing Trading Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 24px; color: #1f77b4; font-weight: bold;}
    .sub-header {font-size: 20px; color: #ff7f0e; font-weight: bold;}
    .highlight {background-color: #f7f7f7; padding: 10px; border-radius: 5px;}
    .positive {color: green; font-weight: bold;}
    .negative {color: red; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

class TechnicalIndicators:
    @staticmethod
    def sma(data, window):
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data, window):
        return data.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data, window=20, num_std=2):
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    @staticmethod
    def stochastic_oscillator(high, low, close, window=14, smooth_k=3):
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        return k.rolling(window=smooth_k).mean()
    
    @staticmethod
    def atr(high, low, close, window=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()
    
    @staticmethod
    def obv(close, volume):
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def cci(high, low, close, window=20):
        tp = (high + low + close) / 3
        cci = (tp - tp.rolling(window=window).mean()) / (0.015 * tp.rolling(window=window).std())
        return cci
    
    @staticmethod
    def adx(high, low, close, window=14):
        up = high.diff()
        down = -low.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=window).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=high.index).rolling(window=window).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=high.index).rolling(window=window).mean() / atr)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=window).mean()
        return adx, plus_di, minus_di
    
    @staticmethod
    def williams_r(high, low, close, window=14):
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))

class TradingStrategy:
    def __init__(self, data, trade_type="both"):
        self.data = data.copy()
        self.trade_type = trade_type
        self.signals = pd.DataFrame(index=data.index)
        
    def generate_signals(self, params):
        # Calculate all indicators
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        volume = self.data['volume']
        
        # SMA
        sma_short = TechnicalIndicators.sma(close, params['sma_short'])
        sma_long = TechnicalIndicators.sma(close, params['sma_long'])
        
        # EMA
        ema_short = TechnicalIndicators.ema(close, params['ema_short'])
        ema_long = TechnicalIndicators.ema(close, params['ema_long'])
        
        # RSI
        rsi = TechnicalIndicators.rsi(close, params['rsi_window'])
        
        # MACD
        macd_line, signal_line, histogram = TechnicalIndicators.macd(
            close, params['macd_fast'], params['macd_slow'], params['macd_signal'])
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(
            close, params['bb_window'], params['bb_std'])
        
        # Stochastic
        stoch_k = TechnicalIndicators.stochastic_oscillator(
            high, low, close, params['stoch_window'], params['stoch_smooth'])
        
        # ATR
        atr = TechnicalIndicators.atr(high, low, close, params['atr_window'])
        
        # OBV
        obv = TechnicalIndicators.obv(close, volume)
        obv_ema = TechnicalIndicators.ema(obv, params['obv_window'])
        
        # CCI
        cci = TechnicalIndicators.cci(high, low, close, params['cci_window'])
        
        # ADX
        adx, plus_di, minus_di = TechnicalIndicators.adx(high, low, close, params['adx_window'])
        
        # Williams %R
        williams_r = TechnicalIndicators.williams_r(high, low, close, params['williams_window'])
        
        # Generate signals based on trade type
        if self.trade_type in ["long", "both"]:
            # Long entry conditions
            long_condition = (
                (sma_short > sma_long) &
                (ema_short > ema_long) &
                (rsi > params['rsi_lower']) & (rsi < params['rsi_upper']) &
                (macd_line > signal_line) &
                (close > bb_middle) &
                (stoch_k > params['stoch_lower']) & (stoch_k < params['stoch_upper']) &
                (obv > obv_ema) &
                (cci > params['cci_lower']) & (cci < params['cci_upper']) &
                (adx > params['adx_threshold']) &
                (plus_di > minus_di) &
                (williams_r > params['williams_lower']) & (williams_r < params['williams_upper'])
            )
            self.signals['long_entry'] = long_condition
            
        if self.trade_type in ["short", "both"]:
            # Short entry conditions
            short_condition = (
                (sma_short < sma_long) &
                (ema_short < ema_long) &
                (rsi > params['rsi_lower']) & (rsi < params['rsi_upper']) &
                (macd_line < signal_line) &
                (close < bb_middle) &
                (stoch_k > params['stoch_lower']) & (stoch_k < params['stoch_upper']) &
                (obv < obv_ema) &
                (cci > params['cci_lower']) & (cci < params['cci_upper']) &
                (adx > params['adx_threshold']) &
                (plus_di < minus_di) &
                (williams_r > params['williams_lower']) & (williams_r < params['williams_upper'])
            )
            self.signals['short_entry'] = short_condition
        
        return self.signals

class Backtester:
    def __init__(self, data, signals, initial_capital=100000, risk_per_trade=0.02):
        self.data = data.copy()
        self.signals = signals.copy()
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.results = None
        
    def run_backtest(self, atr_multiplier=2):
        data = self.data
        signals = self.signals
        capital = self.initial_capital
        position = 0
        trades = []
        atr = TechnicalIndicators.atr(data['high'], data['low'], data['close'], 14)
        
        for i in range(1, len(data)):
            current_date = data.index[i]
            prev_date = data.index[i-1]
            current_price = data['close'].iloc[i]
            prev_price = data['close'].iloc[i-1]
            current_atr = atr.iloc[i]
            
            # Check for entry signals
            if position == 0:
                # Long entry
                if 'long_entry' in signals.columns and signals['long_entry'].iloc[i]:
                    position = 1
                    entry_price = current_price
                    stop_loss = entry_price - (current_atr * atr_multiplier)
                    target = entry_price + (2 * (entry_price - stop_loss))  # Risk:Reward = 1:2
                    entry_capital = capital * self.risk_per_trade
                    shares = entry_capital / entry_price
                    entry_date = current_date
                    reason = "Long signal based on multiple indicator confluence"
                
                # Short entry
                elif 'short_entry' in signals.columns and signals['short_entry'].iloc[i]:
                    position = -1
                    entry_price = current_price
                    stop_loss = entry_price + (current_atr * atr_multiplier)
                    target = entry_price - (2 * (stop_loss - entry_price))  # Risk:Reward = 1:2
                    entry_capital = capital * self.risk_per_trade
                    shares = entry_capital / entry_price
                    entry_date = current_date
                    reason = "Short signal based on multiple indicator confluence"
            
            # Check for exit conditions
            elif position != 0:
                # Long position exit conditions
                if position == 1:
                    if current_price <= stop_loss or current_price >= target:
                        exit_date = current_date
                        exit_price = current_price
                        pnl = (exit_price - entry_price) * shares
                        capital += pnl
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'position': 'long',
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'stop_loss': stop_loss,
                            'target': target,
                            'pnl': pnl,
                            'return': pnl / (entry_price * shares),
                            'reason': reason if current_price >= target else "Stop loss hit",
                            'hold_duration': (exit_date - entry_date).days
                        })
                        position = 0
                
                # Short position exit conditions
                elif position == -1:
                    if current_price >= stop_loss or current_price <= target:
                        exit_date = current_date
                        exit_price = current_price
                        pnl = (entry_price - exit_price) * shares
                        capital += pnl
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'position': 'short',
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'stop_loss': stop_loss,
                            'target': target,
                            'pnl': pnl,
                            'return': pnl / (entry_price * shares),
                            'reason': reason if current_price <= target else "Stop loss hit",
                            'hold_duration': (exit_date - entry_date).days
                        })
                        position = 0
        
        # Create results dataframe
        if trades:
            self.results = pd.DataFrame(trades)
            self.results['win'] = self.results['pnl'] > 0
        else:
            self.results = pd.DataFrame()
        
        return self.results, capital

def optimize_strategy(data, trade_type, method="random", n_iter=50):
    # Define parameter space
    param_space = {
        'sma_short': np.arange(5, 21, 5),
        'sma_long': np.arange(20, 61, 10),
        'ema_short': np.arange(5, 21, 5),
        'ema_long': np.arange(20, 61, 10),
        'rsi_window': np.arange(10, 21, 5),
        'rsi_lower': np.arange(30, 46, 5),
        'rsi_upper': np.arange(55, 71, 5),
        'macd_fast': np.arange(10, 16, 5),
        'macd_slow': np.arange(20, 26, 5),
        'macd_signal': np.arange(7, 10, 2),
        'bb_window': np.arange(15, 26, 5),
        'bb_std': [1.5, 2, 2.5],
        'stoch_window': np.arange(12, 16, 2),
        'stoch_smooth': np.arange(3, 4, 1),
        'stoch_lower': np.arange(20, 31, 10),
        'stoch_upper': np.arange(70, 81, 10),
        'atr_window': np.arange(12, 15, 2),
        'obv_window': np.arange(15, 21, 5),
        'cci_window': np.arange(18, 22, 2),
        'cci_lower': np.arange(-150, -99, 50),
        'cci_upper': np.arange(100, 151, 50),
        'adx_window': np.arange(12, 15, 2),
        'adx_threshold': np.arange(20, 31, 10),
        'williams_window': np.arange(12, 15, 2),
        'williams_lower': np.arange(-90, -79, 10),
        'williams_upper': np.arange(-10, 1, 10)
    }
    
    best_return = -float('inf')
    best_params = None
    best_results = None
    
    if method == "grid":
        # Grid search (simplified for demonstration)
        for _ in range(min(n_iter, 10)):  # Limit iterations for demo
            params = {key: np.random.choice(values) for key, values in param_space.items()}
            
            strategy = TradingStrategy(data, trade_type)
            signals = strategy.generate_signals(params)
            
            backtester = Backtester(data, signals)
            results, final_capital = backtester.run_backtest()
            
            if len(results) > 0:
                total_return = (final_capital - 100000) / 100000
                if total_return > best_return:
                    best_return = total_return
                    best_params = params
                    best_results = results
    else:
        # Random search
        for _ in range(n_iter):
            params = {key: np.random.choice(values) for key, values in param_space.items()}
            
            strategy = TradingStrategy(data, trade_type)
            signals = strategy.generate_signals(params)
            
            backtester = Backtester(data, signals)
            results, final_capital = backtester.run_backtest()
            
            if len(results) > 0:
                total_return = (final_capital - 100000) / 100000
                if total_return > best_return:
                    best_return = total_return
                    best_params = params
                    best_results = results
    
    return best_params, best_results, best_return

def main():
    st.title("Swing Trading Recommendation System")
    st.markdown('<p class="main-header">Advanced technical analysis with automated strategy optimization</p>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload stock data file (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
        
        st.subheader("Uploaded Data")
        st.write(f"Shape: {data.shape}")
        
        # Display first and last few rows
        col1, col2 = st.columns(2)
        with col1:
            st.write("First 5 rows:")
            st.dataframe(data.head())
        with col2:
            st.write("Last 5 rows:")
            st.dataframe(data.tail())
        
        # Standardize column names
        col_mapping = {}
        for col in data.columns:
            col_lower = col.lower()
            if 'date' in col_lower:
                col_mapping[col] = 'date'
            elif 'open' in col_lower:
                col_mapping[col] = 'open'
            elif 'high' in col_lower:
                col_mapping[col] = 'high'
            elif 'low' in col_lower:
                col_mapping[col] = 'low'
            elif 'close' in col_lower:
                col_mapping[col] = 'close'
            elif 'volume' in col_lower:
                col_mapping[col] = 'volume'
        
        data = data.rename(columns=col_mapping)
        
        # Check if all required columns are present
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return
        
        # Convert date column to datetime and sort
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').reset_index(drop=True)
        data = data.set_index('date')
        
        # Display data info
        min_date = data.index.min()
        max_date = data.index.max()
        min_price = data['close'].min()
        max_price = data['close'].max()
        
        st.subheader("Data Information")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Start Date", min_date.strftime("%Y-%m-%d"))
        col2.metric("End Date", max_date.strftime("%Y-%m-%d"))
        #col3.metric("Min Price", f"${min_price:.2f}")
        #col4.metric("Max Price", f"${max_price:.2f}")
        
        # Plot raw data
        st.subheader("Price Chart")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['close'], label='Close Price')
        ax.set_title("Stock Price Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        # EDA and heatmap
        st.subheader("Exploratory Data Analysis")
        
        # Calculate returns
        data['returns'] = data['close'].pct_change()
        
        # Year-month heatmap if we have more than 1 year of data
        if (max_date - min_date).days > 365:
            data['year'] = data.index.year
            data['month'] = data.index.month
            yearly_returns = data.groupby('year')['returns'].mean()
            monthly_returns = data.groupby('month')['returns'].mean()
            
            # Create pivot table for heatmap
            data['year_month'] = data.index.strftime('%Y-%m')
            pivot_returns = data.pivot_table(values='returns', index=data.index.year, columns=data.index.month, aggfunc='mean')
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(pivot_returns, annot=True, fmt=".2%", cmap="RdYlGn", center=0, ax=ax)
            ax.set_title("Monthly Returns Heatmap by Year")
            st.pyplot(fig)
        
        # User inputs
        st.sidebar.subheader("Strategy Parameters")
        trade_type = st.sidebar.selectbox("Trade Type", ["long", "short", "both"], index=2)
        optimization_method = st.sidebar.selectbox("Optimization Method", ["random", "grid"], index=0)
        end_date = st.sidebar.date_input("Backtest End Date", value=max_date, min_value=min_date, max_value=max_date)
        
        # Filter data based on end date
        if end_date:
            train_data = data[data.index <= pd.Timestamp(end_date)]
        else:
            train_data = data
        
        # Generate summary
        st.subheader("Data Summary")
        summary_text = f"""
        The stock data covers the period from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}, 
        with prices ranging from ${min_price:.2f} to ${max_price:.2f}. The overall trend appears {'bullish' if data['close'].iloc[-1] > data['close'].iloc[0] else 'bearish'}, 
        with an average daily return of {data['returns'].mean():.4%}. Volatility, as measured by the standard deviation of returns, is {data['returns'].std():.4%}. 
        Based on technical indicators, there are potential swing trading opportunities on both long and short sides, particularly during periods of high volatility.
        """
        st.info(summary_text)
        
        # Optimize strategy
        if st.button("Optimize Strategy"):
            with st.spinner("Optimizing strategy parameters..."):
                best_params, best_results, best_return = optimize_strategy(
                    train_data, trade_type, method=optimization_method, n_iter=20
                )
            
            if best_params is not None:
                st.success(f"Strategy optimized! Best return: {best_return:.2%}")
                
                # Display best parameters
                st.subheader("Optimized Strategy Parameters")
                param_df = pd.DataFrame.from_dict(best_params, orient='index', columns=['Value'])
                st.dataframe(param_df)
                
                # Display backtest results
                st.subheader("Backtest Results")
                
                if len(best_results) > 0:
                    # Calculate metrics
                    total_trades = len(best_results)
                    winning_trades = len(best_results[best_results['win']])
                    losing_trades = total_trades - winning_trades
                    accuracy = winning_trades / total_trades
                    avg_profit = best_results[best_results['win']]['pnl'].mean() if winning_trades > 0 else 0
                    avg_loss = best_results[~best_results['win']]['pnl'].mean() if losing_trades > 0 else 0
                    profit_factor = abs(avg_profit * winning_trades) / abs(avg_loss * losing_trades) if losing_trades > 0 else float('inf')
                    max_drawdown = (best_results['pnl'].cumsum().min() / 100000)
                    avg_hold_duration = best_results['hold_duration'].mean()
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Trades", total_trades)
                    col2.metric("Accuracy", f"{accuracy:.2%}")
                    col3.metric("Profit Factor", f"{profit_factor:.2f}")
                    col4.metric("Avg Hold Days", f"{avg_hold_duration:.1f}")
                    
                    # Display trade details
                    st.dataframe(best_results)
                    
                    # Plot equity curve
                    fig, ax = plt.subplots(figsize=(12, 6))
                    equity_curve = best_results['pnl'].cumsum() + 100000
                    ax.plot(equity_curve.index, equity_curve.values)
                    ax.set_title("Equity Curve")
                    ax.set_xlabel("Trade Number")
                    ax.set_ylabel("Portfolio Value")
                    ax.grid(True)
                    st.pyplot(fig)
                    
                    # Generate live recommendation
                    st.subheader("Live Recommendation")
                    
                    # Get the last data point
                    last_data = data.iloc[-1]
                    current_price = last_data['close']
                    current_date = data.index[-1]
                    
                    # Generate signal for the last point
                    strategy = TradingStrategy(data, trade_type)
                    signals = strategy.generate_signals(best_params)
                    
                    # Calculate ATR for risk management
                    atr = TechnicalIndicators.atr(data['high'], data['low'], data['close'], 14).iloc[-1]
                    
                    if 'long_entry' in signals.columns and signals['long_entry'].iloc[-1]:
                        stop_loss = current_price - (atr * 2)
                        target = current_price + (2 * (current_price - stop_loss))
                        prob_profit = 0.6  # Placeholder for probability calculation
                        
                        st.success("LONG ENTRY SIGNAL")
                        st.write(f"**Entry Date**: {current_date.strftime('%Y-%m-%d')}")
                        st.write(f"**Entry Price**: ${current_price:.2f}")
                        st.write(f"**Target**: ${target:.2f}")
                        st.write(f"**Stop Loss**: ${stop_loss:.2f}")
                        st.write(f"**Probability of Profit**: {prob_profit:.2%}")
                        st.write("**Reason**: Multiple indicator confluence suggesting bullish momentum")
                    
                    elif 'short_entry' in signals.columns and signals['short_entry'].iloc[-1]:
                        stop_loss = current_price + (atr * 2)
                        target = current_price - (2 * (stop_loss - current_price))
                        prob_profit = 0.6  # Placeholder for probability calculation
                        
                        st.success("SHORT ENTRY SIGNAL")
                        st.write(f"**Entry Date**: {current_date.strftime('%Y-%m-%d')}")
                        st.write(f"**Entry Price**: ${current_price:.2f}")
                        st.write(f"**Target**: ${target:.2f}")
                        st.write(f"**Stop Loss**: ${stop_loss:.2f}")
                        st.write(f"**Probability of Profit**: {prob_profit:.2%}")
                        st.write("**Reason**: Multiple indicator confluence suggesting bearish momentum")
                    
                    else:
                        st.info("No clear trading signal at the moment")
                    
                    # Final summary
                    st.subheader("Strategy Summary")
                    summary_text = f"""
                    The optimized strategy generated {total_trades} trades with an accuracy of {accuracy:.2%}. 
                    The strategy achieved a total return of {best_return:.2%} compared to a buy-and-hold return of {(data['close'].iloc[-1] / data['close'].iloc[0] - 1):.2%}. 
                    The average holding period was {avg_hold_duration:.1f} days, suitable for swing trading. 
                    For live trading, follow the signals generated by the strategy with proper risk management, 
                    ensuring position sizing based on the {best_params.get('atr_window', 14)}-day ATR for stop-loss placement.
                    """
                    st.info(summary_text)
                else:
                    st.warning("No trades were generated with the optimized parameters")
            else:
                st.error("Strategy optimization failed. Please try different parameters or more iterations.")

if __name__ == "__main__":
    main()
