import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Advanced Stock Trading Strategy Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TechnicalIndicators:
    @staticmethod
    def sma(data, period):
        """Simple Moving Average"""
        if len(data) < period:
            return np.full(len(data), np.nan)
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data, period):
        """Exponential Moving Average"""
        if len(data) < period:
            return np.full(len(data), np.nan)
        return data.ewm(span=period).mean()
    
    @staticmethod
    def rsi(data, period=14):
        """Relative Strength Index"""
        if len(data) < period + 1:
            return np.full(len(data), np.nan)
        
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """MACD Indicator"""
        if len(data) < slow:
            return np.full(len(data), np.nan), np.full(len(data), np.nan), np.full(len(data), np.nan)
        
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data, period=20, std_dev=2):
        """Bollinger Bands"""
        if len(data) < period:
            return np.full(len(data), np.nan), np.full(len(data), np.nan), np.full(len(data), np.nan)
        
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3):
        """Stochastic Oscillator"""
        if len(close) < k_period:
            return np.full(len(close), np.nan), np.full(len(close), np.nan)
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high, low, close, period=14):
        """Williams %R"""
        if len(close) < period:
            return np.full(len(close), np.nan)
        
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    @staticmethod
    def atr(high, low, close, period=14):
        """Average True Range"""
        if len(close) < 2:
            return np.full(len(close), np.nan)
        
        high_low = high - low
        high_close = np.abs(high - close.shift(1))
        low_close = np.abs(low - close.shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def cci(high, low, close, period=20):
        """Commodity Channel Index"""
        if len(close) < period:
            return np.full(len(close), np.nan)
        
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci
    
    @staticmethod
    def adx(high, low, close, period=14):
        """Average Directional Index"""
        if len(close) < period + 1:
            return np.full(len(close), np.nan)
        
        high_diff = high.diff()
        low_diff = -low.diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        plus_dm = pd.Series(plus_dm, index=close.index)
        minus_dm = pd.Series(minus_dm, index=close.index)
        
        tr = TechnicalIndicators.atr(high, low, close, 1)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def obv(close, volume):
        """On-Balance Volume"""
        obv = np.where(close.diff() > 0, volume, np.where(close.diff() < 0, -volume, 0))
        return pd.Series(obv, index=close.index).cumsum()
    
    @staticmethod
    def mfi(high, low, close, volume, period=14):
        """Money Flow Index"""
        if len(close) < period:
            return np.full(len(close), np.nan)
        
        tp = (high + low + close) / 3
        raw_mf = tp * volume
        
        positive_mf = np.where(tp.diff() > 0, raw_mf, 0)
        negative_mf = np.where(tp.diff() < 0, raw_mf, 0)
        
        positive_mf = pd.Series(positive_mf, index=close.index)
        negative_mf = pd.Series(negative_mf, index=close.index)
        
        positive_mf_sum = positive_mf.rolling(window=period).sum()
        negative_mf_sum = negative_mf.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf_sum / negative_mf_sum))
        return mfi

class StrategyBacktester:
    def __init__(self):
        self.data = None
        self.results = None
        self.best_params = None
        
    def clean_data(self, df):
        """Clean and prepare data"""
        # Remove duplicate column names
        df.columns = [col.strip() for col in df.columns]
        
        # Common column mappings
        column_mappings = {
            'Date': ['Date', 'DATE', 'date', 'Datetime', 'DATETIME'],
            'Open': ['OPEN', 'open', 'Open', 'opening'],
            'High': ['HIGH', 'high', 'High'],
            'Low': ['LOW', 'low', 'Low'],
            'Close': ['close', 'Close', 'CLOSE', 'ltp', 'LTP'],
            'Volume': ['VOLUME', 'volume', 'Volume', 'Vol', 'VOL']
        }
        
        # Find matching columns
        final_columns = {}
        for standard_col, possible_names in column_mappings.items():
            for col in df.columns:
                if col in possible_names:
                    final_columns[standard_col] = col
                    break
        
        # Check if we have required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in final_columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.write("Available columns:", list(df.columns))
            return None
        
        # Create clean dataframe
        clean_df = pd.DataFrame()
        for standard_col, original_col in final_columns.items():
            clean_df[standard_col] = df[original_col]
        
        # Handle date column
        if clean_df['Date'].dtype == 'object':
            try:
                clean_df['Date'] = pd.to_datetime(clean_df['Date'], format='%d-%b-%Y')
            except:
                try:
                    clean_df['Date'] = pd.to_datetime(clean_df['Date'])
                except:
                    st.error("Could not parse date column")
                    return None
        
        # Convert numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close']
        if 'Volume' in clean_df.columns:
            numeric_cols.append('Volume')
            
        for col in numeric_cols:
            if col in clean_df.columns:
                # Remove commas and convert to numeric
                clean_df[col] = pd.to_numeric(
                    clean_df[col].astype(str).str.replace(',', ''), 
                    errors='coerce'
                )
        
        # Sort by date
        clean_df = clean_df.sort_values('Date').reset_index(drop=True)
        
        # Remove rows with NaN values in OHLC
        clean_df = clean_df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        # Add Volume column if missing
        if 'Volume' not in clean_df.columns:
            clean_df['Volume'] = 1000000  # Default volume
            
        return clean_df
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators"""
        try:
            # Moving Averages
            df['SMA_10'] = TechnicalIndicators.sma(df['Close'], 10)
            df['SMA_20'] = TechnicalIndicators.sma(df['Close'], 20)
            df['SMA_50'] = TechnicalIndicators.sma(df['Close'], 50)
            df['EMA_10'] = TechnicalIndicators.ema(df['Close'], 10)
            df['EMA_20'] = TechnicalIndicators.ema(df['Close'], 20)
            
            # RSI
            df['RSI'] = TechnicalIndicators.rsi(df['Close'])
            
            # MACD
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = TechnicalIndicators.macd(df['Close'])
            
            # Bollinger Bands
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = TechnicalIndicators.bollinger_bands(df['Close'])
            
            # Stochastic
            df['Stoch_K'], df['Stoch_D'] = TechnicalIndicators.stochastic(df['High'], df['Low'], df['Close'])
            
            # Williams %R
            df['Williams_R'] = TechnicalIndicators.williams_r(df['High'], df['Low'], df['Close'])
            
            # ATR
            df['ATR'] = TechnicalIndicators.atr(df['High'], df['Low'], df['Close'])
            
            # CCI
            df['CCI'] = TechnicalIndicators.cci(df['High'], df['Low'], df['Close'])
            
            # ADX
            df['ADX'] = TechnicalIndicators.adx(df['High'], df['Low'], df['Close'])
            
            # OBV
            df['OBV'] = TechnicalIndicators.obv(df['Close'], df['Volume'])
            
            # MFI
            df['MFI'] = TechnicalIndicators.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # Price-based indicators
            df['Price_Change'] = df['Close'].pct_change()
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Close_SMA20_Ratio'] = df['Close'] / df['SMA_20']
            
            return df
        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
            return None
    
    def generate_signals(self, df, params):
        """Generate buy/sell signals based on parameters"""
        try:
            signals = pd.DataFrame(index=df.index)
            signals['Signal'] = 0
            
            # Initialize conditions
            conditions = []
            
            # RSI conditions
            if params.get('use_rsi', False):
                rsi_oversold = params.get('rsi_oversold', 30)
                rsi_overbought = params.get('rsi_overbought', 70)
                conditions.append(df['RSI'] < rsi_oversold)  # Oversold for buy
                
            # MACD conditions
            if params.get('use_macd', False):
                conditions.append(df['MACD'] > df['MACD_Signal'])  # MACD bullish
                
            # Moving Average conditions
            if params.get('use_ma_cross', False):
                ma_fast = params.get('ma_fast', 10)
                ma_slow = params.get('ma_slow', 20)
                conditions.append(df[f'SMA_{ma_fast}'] > df[f'SMA_{ma_slow}'])  # Fast MA above slow MA
                
            # Bollinger Bands conditions
            if params.get('use_bb', False):
                conditions.append(df['Close'] < df['BB_Lower'])  # Price below lower band
                
            # Stochastic conditions
            if params.get('use_stoch', False):
                stoch_oversold = params.get('stoch_oversold', 20)
                conditions.append(df['Stoch_K'] < stoch_oversold)  # Stochastic oversold
                
            # Volume conditions
            if params.get('use_volume', False):
                volume_ma = df['Volume'].rolling(window=20).mean()
                conditions.append(df['Volume'] > volume_ma * 1.5)  # High volume
                
            # CCI conditions
            if params.get('use_cci', False):
                cci_oversold = params.get('cci_oversold', -100)
                conditions.append(df['CCI'] < cci_oversold)  # CCI oversold
                
            # Williams %R conditions
            if params.get('use_williams', False):
                williams_oversold = params.get('williams_oversold', -80)
                conditions.append(df['Williams_R'] < williams_oversold)  # Williams %R oversold
                
            # MFI conditions
            if params.get('use_mfi', False):
                mfi_oversold = params.get('mfi_oversold', 20)
                conditions.append(df['MFI'] < mfi_oversold)  # MFI oversold
                
            # ADX conditions for trend strength
            if params.get('use_adx', False):
                adx_threshold = params.get('adx_threshold', 25)
                conditions.append(df['ADX'] > adx_threshold)  # Strong trend
            
            # Combine conditions
            if conditions:
                # For long strategy, all conditions should be true
                buy_signal = pd.Series(True, index=df.index)
                for condition in conditions:
                    buy_signal = buy_signal & condition.fillna(False)
                    
                signals.loc[buy_signal, 'Signal'] = 1
                
                # For short strategy, invert the conditions
                if params.get('strategy_type', 'Long') == 'Short':
                    sell_signal = pd.Series(True, index=df.index)
                    for condition in conditions:
                        sell_signal = sell_signal & (~condition).fillna(False)
                    signals.loc[sell_signal, 'Signal'] = -1
                    signals.loc[buy_signal, 'Signal'] = 0
            
            return signals
        except Exception as e:
            st.error(f"Error generating signals: {str(e)}")
            return pd.DataFrame(index=df.index, columns=['Signal'])
    
    def backtest_strategy(self, df, params):
        """Backtest the strategy"""
        try:
            signals = self.generate_signals(df, params)
            
            # Initialize portfolio
            initial_capital = 100000
            capital = initial_capital
            position = 0
            entry_price = 0
            trades = []
            portfolio_values = []
            
            stop_loss_pct = params.get('stop_loss', 0.02)  # 2% default
            take_profit_pct = params.get('take_profit', 0.06)  # 6% default
            
            for i in range(len(df)):
                current_price = df['Close'].iloc[i]
                signal = signals['Signal'].iloc[i]
                date = df['Date'].iloc[i]
                
                # Calculate portfolio value
                if position != 0:
                    unrealized_pnl = position * (current_price - entry_price)
                    portfolio_value = capital + unrealized_pnl
                else:
                    portfolio_value = capital
                
                portfolio_values.append(portfolio_value)
                
                # Check exit conditions first
                if position != 0:
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    # For long positions
                    if position > 0:
                        if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                            # Exit position
                            pnl = position * (current_price - entry_price)
                            capital += pnl
                            
                            exit_reason = "Stop Loss" if pnl_pct <= -stop_loss_pct else "Take Profit"
                            
                            trades.append({
                                'Entry_Date': entry_date,
                                'Exit_Date': date,
                                'Entry_Price': entry_price,
                                'Exit_Price': current_price,
                                'Position_Size': position,
                                'PnL': pnl,
                                'PnL_Pct': pnl_pct * 100,
                                'Exit_Reason': exit_reason,
                                'Direction': 'Long'
                            })
                            
                            position = 0
                            entry_price = 0
                    
                    # For short positions
                    elif position < 0:
                        if pnl_pct >= stop_loss_pct or pnl_pct <= -take_profit_pct:
                            # Exit position
                            pnl = position * (current_price - entry_price)
                            capital += pnl
                            
                            exit_reason = "Stop Loss" if pnl_pct >= stop_loss_pct else "Take Profit"
                            
                            trades.append({
                                'Entry_Date': entry_date,
                                'Exit_Date': date,
                                'Entry_Price': entry_price,
                                'Exit_Price': current_price,
                                'Position_Size': abs(position),
                                'PnL': pnl,
                                'PnL_Pct': -pnl_pct * 100,  # Invert for short
                                'Exit_Reason': exit_reason,
                                'Direction': 'Short'
                            })
                            
                            position = 0
                            entry_price = 0
                
                # Check entry signals
                if position == 0 and signal != 0:
                    position_size = capital * 0.95  # Use 95% of capital
                    shares = position_size / current_price
                    
                    if signal == 1:  # Long position
                        position = shares
                        entry_price = current_price
                        entry_date = date
                        capital -= position_size
                    elif signal == -1:  # Short position
                        position = -shares
                        entry_price = current_price
                        entry_date = date
                        capital -= position_size
            
            # Close any remaining position
            if position != 0:
                pnl = position * (df['Close'].iloc[-1] - entry_price)
                capital += pnl
                
                pnl_pct = (df['Close'].iloc[-1] - entry_price) / entry_price
                if position < 0:
                    pnl_pct = -pnl_pct
                
                trades.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': df['Date'].iloc[-1],
                    'Entry_Price': entry_price,
                    'Exit_Price': df['Close'].iloc[-1],
                    'Position_Size': abs(position),
                    'PnL': pnl,
                    'PnL_Pct': pnl_pct * 100,
                    'Exit_Reason': 'End of Period',
                    'Direction': 'Long' if position > 0 else 'Short'
                })
            
            # Calculate final portfolio value
            final_value = capital
            if position != 0:
                final_value += position * df['Close'].iloc[-1]
            
            portfolio_values[-1] = final_value
            
            # Calculate performance metrics
            trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
            
            if len(trades_df) > 0:
                total_return = (final_value - initial_capital) / initial_capital * 100
                win_rate = len(trades_df[trades_df['PnL'] > 0]) / len(trades_df) * 100
                avg_win = trades_df[trades_df['PnL'] > 0]['PnL_Pct'].mean() if len(trades_df[trades_df['PnL'] > 0]) > 0 else 0
                avg_loss = trades_df[trades_df['PnL'] <= 0]['PnL_Pct'].mean() if len(trades_df[trades_df['PnL'] <= 0]) > 0 else 0
                profit_factor = abs(trades_df[trades_df['PnL'] > 0]['PnL'].sum() / trades_df[trades_df['PnL'] <= 0]['PnL'].sum()) if trades_df[trades_df['PnL'] <= 0]['PnL'].sum() != 0 else float('inf')
                
                # Calculate Sharpe ratio
                returns = pd.Series(portfolio_values).pct_change().dropna()
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
                
                # Calculate maximum drawdown
                peak = pd.Series(portfolio_values).expanding().max()
                drawdown = (pd.Series(portfolio_values) - peak) / peak
                max_drawdown = drawdown.min() * 100
                
                performance = {
                    'Total_Return': total_return,
                    'Total_Trades': len(trades_df),
                    'Win_Rate': win_rate,
                    'Avg_Win': avg_win,
                    'Avg_Loss': avg_loss,
                    'Profit_Factor': profit_factor,
                    'Sharpe_Ratio': sharpe_ratio,
                    'Max_Drawdown': max_drawdown,
                    'Final_Value': final_value
                }
            else:
                performance = {
                    'Total_Return': 0,
                    'Total_Trades': 0,
                    'Win_Rate': 0,
                    'Avg_Win': 0,
                    'Avg_Loss': 0,
                    'Profit_Factor': 0,
                    'Sharpe_Ratio': 0,
                    'Max_Drawdown': 0,
                    'Final_Value': initial_capital
                }
            
            return performance, trades_df, portfolio_values
            
        except Exception as e:
            st.error(f"Error in backtesting: {str(e)}")
            return None, None, None
    
    def optimize_strategy(self, df, strategy_type):
        """Optimize strategy parameters using grid search"""
        try:
            param_grid = {
                'use_rsi': [True, False],
                'rsi_oversold': [20, 25, 30, 35],
                'rsi_overbought': [65, 70, 75, 80],
                'use_macd': [True, False],
                'use_ma_cross': [True, False],
                'ma_fast': [5, 10, 15],
                'ma_slow': [20, 25, 30],
                'use_bb': [True, False],
                'use_stoch': [True, False],
                'stoch_oversold': [15, 20, 25],
                'use_volume': [True, False],
                'use_cci': [True, False],
                'cci_oversold': [-120, -100, -80],
                'use_williams': [True, False],
                'williams_oversold': [-90, -80, -70],
                'use_mfi': [True, False],
                'mfi_oversold': [15, 20, 25],
                'use_adx': [True, False],
                'adx_threshold': [20, 25, 30],
                'stop_loss': [0.01, 0.02, 0.03, 0.05],
                'take_profit': [0.04, 0.06, 0.08, 0.10],
                'strategy_type': [strategy_type]
            }
            
            # Generate parameter combinations (limited to prevent timeout)
            import itertools
            keys = list(param_grid.keys())
            values = list(param_grid.values())
            
            best_performance = -float('inf')
            best_params = None
            best_trades = None
            best_portfolio = None
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Sample combinations to limit computation time
            max_combinations = 1000
            all_combinations = list(itertools.product(*values))
            
            if len(all_combinations) > max_combinations:
                import random
                combinations = random.sample(all_combinations, max_combinations)
            else:
                combinations = all_combinations
            
            total_combinations = len(combinations)
            
            for i, combo in enumerate(combinations):
                params = dict(zip(keys, combo))
                
                # Update progress
                progress = (i + 1) / total_combinations
                progress_bar.progress(progress)
                status_text.text(f"Testing combination {i+1}/{total_combinations}")
                
                performance, trades_df, portfolio_values = self.backtest_strategy(df, params)
                
                if performance is not None:
                    # Score based on multiple criteria
                    score = (
                        performance['Total_Return'] * 0.4 +
                        performance['Win_Rate'] * 0.2 +
                        performance['Profit_Factor'] * 10 * 0.2 +
                        performance['Sharpe_Ratio'] * 20 * 0.1 +
                        (-performance['Max_Drawdown']) * 0.1
                    )
                    
                    if score > best_performance and performance['Total_Trades'] >= 5:
                        best_performance = score
                        best_params = params.copy()
                        best_trades = trades_df
                        best_portfolio = portfolio_values
            
            progress_bar.empty()
            status_text.empty()
            
            return best_params, best_trades, best_portfolio
            
        except Exception as e:
            st.error(f"Error in optimization: {str(e)}")
            return None, None, None

def main():
    st.title("🚀 Advanced Stock Trading Strategy Backtester")
    st.markdown("Upload your stock data and let AI find the most profitable trading strategy!")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Strategy type selection
    strategy_type = st.sidebar.selectbox(
        "Select Strategy Type",
        ["Long", "Short"],
        help="Choose whether to optimize for long or short trading strategies"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Stock Data (CSV/Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="Upload historical stock data with OHLCV columns"
    )
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Initialize backtester
            backtester = StrategyBacktester()
            
            # Clean data
            with st.spinner("🔧 Cleaning and preparing data..."):
                clean_df = backtester.clean_data(df)
            
            if clean_df is not None:
                st.success(f"✅ Data cleaned! Final shape: {clean_df.shape}")
                
                # Calculate indicators
                with st.spinner("📊 Calculating technical indicators..."):
                    df_with_indicators = backtester.calculate_indicators(clean_df)
                
                if df_with_indicators is not None:
                    st.success("✅ Technical indicators calculated!")
                    
                    # Optimize strategy
                    with st.spinner(f"🎯 Optimizing {strategy_type.lower()} strategy... This may take a few minutes."):
                        best_params, best_trades, best_portfolio = backtester.optimize_strategy(df_with_indicators, strategy_type)
                    
                    if best_params is not None:
                        st.success("✅ Strategy optimization completed!")
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🎯 Best Strategy Parameters")
                            param_df = pd.DataFrame(list(best_params.items()), columns=['Parameter', 'Value'])
                            st.dataframe(param_df, use_container_width=True)
                        
                        with col2:
                            st.subheader("📈 Performance Metrics")
                            if len(best_trades) > 0:
                                performance, _, _ = backtester.backtest_strategy(df_with_indicators, best_params)
                                
                                metrics = {
                                    'Total Return': f"{performance['Total_Return']:.2f}%",
                                    'Total Trades': performance['Total_Trades'],
                                    'Win Rate': f"{performance['Win_Rate']:.2f}%",
                                    'Profit Factor': f"{performance['Profit_Factor']:.2f}",
                                    'Sharpe Ratio': f"{performance['Sharpe_Ratio']:.2f}",
                                    'Max Drawdown': f"{performance['Max_Drawdown']:.2f}%"
                                }
                                
                                for metric, value in metrics.items():
                                    st.metric(metric, value)
                        
                        # Performance Summary
                        st.subheader("📋 Performance Summary")
                        if len(best_trades) > 0:
                            total_return = performance['Total_Return']
                            win_rate = performance['Win_Rate']
                            total_trades = performance['Total_Trades']
                            profit_factor = performance['Profit_Factor']
                            
                            # Buy and hold comparison
                            buy_hold_return = ((df_with_indicators['Close'].iloc[-1] - df_with_indicators['Close'].iloc[0]) / df_with_indicators['Close'].iloc[0]) * 100
                            
                            summary = f"""
                            **Strategy Performance vs Buy & Hold:**
                            
                            The optimized {strategy_type.lower()} trading strategy generated a total return of {total_return:.2f}% compared to a buy-and-hold return of {buy_hold_return:.2f}%. 
                            The strategy executed {total_trades} trades with a win rate of {win_rate:.1f}%, achieving a profit factor of {profit_factor:.2f}. 
                            
                            {'✅ The strategy outperformed buy-and-hold!' if total_return > buy_hold_return else '❌ The strategy underperformed buy-and-hold.'}
                            
                            **Risk Assessment:** The maximum drawdown was {performance['Max_Drawdown']:.2f}% with a Sharpe ratio of {performance['Sharpe_Ratio']:.2f}, 
                            indicating {'good' if performance['Sharpe_Ratio'] > 1 else 'moderate' if performance['Sharpe_Ratio'] > 0.5 else 'poor'} risk-adjusted returns.
                            """
                            
                            st.markdown(summary)
                        
                        # Trades Table
                        if len(best_trades) > 0:
                            st.subheader("💼 Trade History")
                            st.dataframe(best_trades.round(2), use_container_width=True)
                            
                            # Download trades
                            csv = best_trades.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Trade History",
                                data=csv,
                                file_name="trade_history.csv",
                                mime="text/csv"
                            )
                        
                        # Portfolio Performance Chart
                        if best_portfolio is not None:
                            st.subheader("📊 Portfolio Performance")
                            
                            fig = make_subplots(
                                rows=2, cols=1,
                                subplot_titles=['Portfolio Value Over Time', 'Price Chart with Signals'],
                                vertical_spacing=0.1,
                                row_heights=[0.4, 0.6]
                            )
                            
                            # Portfolio value chart
                            dates = df_with_indicators['Date'][:len(best_portfolio)]
                            fig.add_trace(
                                go.Scatter(x=dates, y=best_portfolio, name='Portfolio Value', line=dict(color='green')),
                                row=1, col=1
                            )
                            
                            # Buy and hold comparison
                            initial_value = 100000
                            buy_hold_values = initial_value * (df_with_indicators['Close'] / df_with_indicators['Close'].iloc[0])
                            fig.add_trace(
                                go.Scatter(x=df_with_indicators['Date'], y=buy_hold_values, name='Buy & Hold', line=dict(color='blue', dash='dash')),
                                row=1, col=1
                            )
                            
                            # Price chart
                            fig.add_trace(
                                go.Candlestick(
                                    x=df_with_indicators['Date'],
                                    open=df_with_indicators['Open'],
                                    high=df_with_indicators['High'],
                                    low=df_with_indicators['Low'],
                                    close=df_with_indicators['Close'],
                                    name='OHLC'
                                ),
                                row=2, col=1
                            )
                            
                            # Add trade signals
                            signals = backtester.generate_signals(df_with_indicators, best_params)
                            buy_signals = df_with_indicators[signals['Signal'] == 1]
                            sell_signals = df_with_indicators[signals['Signal'] == -1]
                            
                            if len(buy_signals) > 0:
                                fig.add_trace(
                                    go.Scatter(
                                        x=buy_signals['Date'],
                                        y=buy_signals['Close'],
                                        mode='markers',
                                        name='Buy Signals',
                                        marker=dict(symbol='triangle-up', size=10, color='green')
                                    ),
                                    row=2, col=1
                                )
                            
                            if len(sell_signals) > 0:
                                fig.add_trace(
                                    go.Scatter(
                                        x=sell_signals['Date'],
                                        y=sell_signals['Close'],
                                        mode='markers',
                                        name='Sell Signals',
                                        marker=dict(symbol='triangle-down', size=10, color='red')
                                    ),
                                    row=2, col=1
                                )
                            
                            fig.update_layout(
                                title=f"{strategy_type} Strategy Performance",
                                xaxis_title="Date",
                                height=800,
                                showlegend=True
                            )
                            
                            fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
                            fig.update_yaxes(title_text="Price ($)", row=2, col=1)
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Live Trading Recommendation
                        st.subheader("🚨 Live Trading Recommendation")
                        
                        # Get latest data point
                        latest_data = df_with_indicators.iloc[-1]
                        current_signals = backtester.generate_signals(df_with_indicators, best_params)
                        latest_signal = current_signals['Signal'].iloc[-1]
                        
                        # Calculate recommendation
                        current_price = latest_data['Close']
                        atr_value = latest_data['ATR']
                        
                        if latest_signal == 1:  # Buy signal
                            recommendation = "🟢 BUY"
                            entry_price = current_price
                            stop_loss = entry_price * (1 - best_params['stop_loss'])
                            target_price = entry_price * (1 + best_params['take_profit'])
                            direction = "Long"
                            
                        elif latest_signal == -1:  # Sell signal
                            recommendation = "🔴 SELL (Short)"
                            entry_price = current_price
                            stop_loss = entry_price * (1 + best_params['stop_loss'])
                            target_price = entry_price * (1 - best_params['take_profit'])
                            direction = "Short"
                            
                        else:
                            recommendation = "⚪ HOLD"
                            entry_price = current_price
                            stop_loss = None
                            target_price = None
                            direction = "None"
                        
                        # Display recommendation
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Recommendation", recommendation)
                            st.metric("Current Price", f"${current_price:.2f}")
                            
                        with col2:
                            if stop_loss:
                                st.metric("Entry Price", f"${entry_price:.2f}")
                                st.metric("Stop Loss", f"${stop_loss:.2f}")
                                
                        with col3:
                            if target_price:
                                st.metric("Target Price", f"${target_price:.2f}")
                                risk_reward = (target_price - entry_price) / (entry_price - stop_loss) if direction == "Long" else (entry_price - target_price) / (stop_loss - entry_price)
                                st.metric("Risk:Reward", f"1:{risk_reward:.2f}")
                        
                        # Recommendation details
                        if latest_signal != 0:
                            st.subheader("📋 Trade Setup Details")
                            
                            # Calculate probability based on historical performance
                            win_rate = performance['Win_Rate']
                            profit_prob = win_rate / 100
                            
                            trade_details = f"""
                            **Trade Setup:**
                            - **Direction:** {direction}
                            - **Entry Logic:** Based on optimized parameters combining multiple technical indicators
                            - **Stop Loss Logic:** {best_params['stop_loss']*100:.1f}% risk management
                            - **Target Logic:** {best_params['take_profit']*100:.1f}% profit target
                            - **Trailing SL:** Manual trailing recommended as price moves in favor
                            
                            **Probability Analysis:**
                            - **Win Probability:** {win_rate:.1f}% (based on backtesting)
                            - **Expected Return:** {performance['Avg_Win']:.2f}% (average winning trade)
                            - **Risk Assessment:** {abs(performance['Avg_Loss']):.2f}% (average losing trade)
                            
                            **Entry Criteria Met:**
                            """
                            
                            # Show which indicators triggered
                            criteria = []
                            if best_params.get('use_rsi') and not pd.isna(latest_data.get('RSI')):
                                rsi_val = latest_data['RSI']
                                if strategy_type == "Long" and rsi_val < best_params['rsi_oversold']:
                                    criteria.append(f"✅ RSI Oversold ({rsi_val:.1f} < {best_params['rsi_oversold']})")
                                elif strategy_type == "Short" and rsi_val > best_params['rsi_overbought']:
                                    criteria.append(f"✅ RSI Overbought ({rsi_val:.1f} > {best_params['rsi_overbought']})")
                            
                            if best_params.get('use_macd') and not pd.isna(latest_data.get('MACD')):
                                macd_val = latest_data['MACD']
                                macd_signal = latest_data['MACD_Signal']
                                if strategy_type == "Long" and macd_val > macd_signal:
                                    criteria.append("✅ MACD Bullish Crossover")
                                elif strategy_type == "Short" and macd_val < macd_signal:
                                    criteria.append("✅ MACD Bearish Crossover")
                            
                            if best_params.get('use_bb') and not pd.isna(latest_data.get('BB_Lower')):
                                if strategy_type == "Long" and current_price < latest_data['BB_Lower']:
                                    criteria.append("✅ Price Below Lower Bollinger Band")
                                elif strategy_type == "Short" and current_price > latest_data['BB_Upper']:
                                    criteria.append("✅ Price Above Upper Bollinger Band")
                            
                            trade_details += "\n".join(criteria) if criteria else "- Multiple technical indicators aligned"
                            
                            st.markdown(trade_details)
                            
                            # Risk warning
                            st.warning("⚠️ **Risk Disclaimer:** This recommendation is based on historical backtesting. Past performance does not guarantee future results. Always use proper risk management and consider your risk tolerance before trading.")
                        
                        else:
                            st.info("📊 No trading signal detected. Current market conditions do not meet the optimized strategy criteria. Wait for a clear setup.")
                        
                        # Strategy parameters used
                        with st.expander("🔧 View Strategy Parameters"):
                            st.json(best_params)
                    
                    else:
                        st.error("❌ Strategy optimization failed. Please check your data and try again.")
                
                else:
                    st.error("❌ Failed to calculate technical indicators.")
            
            else:
                st.error("❌ Data cleaning failed. Please check your file format.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        # Display sample data format
        st.info("📋 **Sample Data Format Required:**")
        sample_data = {
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Open': [100.0, 102.0, 101.5],
            'High': [103.0, 104.0, 103.0],
            'Low': [99.0, 101.0, 100.0],
            'Close': [102.0, 103.0, 102.5],
            'Volume': [1000000, 1200000, 900000]
        }
        st.dataframe(pd.DataFrame(sample_data))
        
        st.markdown("""
        **Requirements:**
        - ✅ Date column (any common format)
        - ✅ OHLC columns (Open, High, Low, Close)
        - ✅ Volume column (optional, will be estimated if missing)
        - ✅ CSV or Excel format
        - ✅ Historical data (minimum 100+ rows recommended)
        
        **Features:**
        - 🎯 Advanced multi-indicator optimization
        - 📊 Long and short strategy support  
        - 📈 Comprehensive backtesting with risk metrics
        - 🚨 Live trading recommendations
        - 💼 Detailed trade analysis
        - 📋 Performance comparison vs buy-and-hold
        """)

if __name__ == "__main__":
    main()
