import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import ParameterGrid
import random
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Advanced Swing Trading Recommender", layout="wide", page_icon="📈")

class TechnicalIndicators:
    @staticmethod
    def sma(data, period):
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data, period):
        return data.ewm(span=period).mean()
    
    @staticmethod
    def rsi(data, period=14):
        st.write(f"🔢 Calculating RSI with period {period}")
        st.write(f"RSI input data type: {data.dtype}")
        st.write(f"RSI input sample: {data.head().tolist()}")
        
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Handle division by zero
        rs = gain / loss.replace(0, np.nan)
        rsi_result = 100 - (100 / (1 + rs))
        
        st.write(f"RSI calculated successfully. Sample values: {rsi_result.dropna().head().tolist()}")
        return rsi_result
    
    @staticmethod
    def bollinger_bands(data, period=20, std=2):
        sma = data.rolling(window=period).mean()
        std_dev = data.rolling(window=period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper, sma, lower
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3):
        st.write(f"📊 Calculating Stochastic with K={k_period}, D={d_period}")
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        # Handle division by zero
        range_diff = highest_high - lowest_low
        range_diff = range_diff.replace(0, np.nan)
        
        k_percent = 100 * ((close - lowest_low) / range_diff)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        st.write(f"Stochastic calculated. K sample: {k_percent.dropna().head().tolist()}")
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high, low, close, period=14):
        st.write(f"📈 Calculating Williams %R with period {period}")
        
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        # Handle division by zero
        range_diff = highest_high - lowest_low
        range_diff = range_diff.replace(0, np.nan)
        
        wr = -100 * (highest_high - close) / range_diff
        
        st.write(f"Williams %R calculated. Sample: {wr.dropna().head().tolist()}")
        return wr
    
    @staticmethod
    def cci(high, low, close, period=20):
        st.write(f"🎯 Calculating CCI with period {period}")
        
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        # Handle division by zero
        mad_nonzero = mad.replace(0, np.nan)
        cci = (tp - sma_tp) / (0.015 * mad_nonzero)
        
        st.write(f"CCI calculated. Sample: {cci.dropna().head().tolist()}")
        return cci
    
    @staticmethod
    def atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def adx(high, low, close, period=14):
        st.write(f"📊 Calculating ADX with period {period}")
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        # Handle division by zero
        atr_nonzero = atr.replace(0, np.nan)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_nonzero)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_nonzero)
        
        # Handle division by zero for DX calculation
        di_sum = plus_di + minus_di
        di_sum_nonzero = di_sum.replace(0, np.nan)
        dx = 100 * abs(plus_di - minus_di) / di_sum_nonzero
        adx = dx.rolling(window=period).mean()
        
        st.write(f"ADX calculated. Sample ADX: {adx.dropna().head().tolist()}")
        return adx, plus_di, minus_di

class SwingTradingStrategy:
    def __init__(self):
        self.indicators = TechnicalIndicators()
        
    def calculate_all_indicators(self, df, params):
        st.write("🔄 **Starting indicator calculations...**")
        st.write(f"DataFrame shape: {df.shape}")
        st.write(f"DataFrame columns: {list(df.columns)}")
        
        df = df.copy()
        
        # Ensure all price columns are numeric
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in price_cols:
            if col in df.columns:
                st.write(f"Checking data type for {col}: {df[col].dtype}")
                if df[col].dtype == 'object':
                    st.write(f"Converting {col} to numeric...")
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    st.write(f"After conversion {col} dtype: {df[col].dtype}")
        
        try:
            # Moving Averages
            st.write("📈 Calculating Moving Averages...")
            df['SMA_fast'] = self.indicators.sma(df['close'], params['sma_fast'])
            df['SMA_slow'] = self.indicators.sma(df['close'], params['sma_slow'])
            df['EMA_fast'] = self.indicators.ema(df['close'], params['ema_fast'])
            df['EMA_slow'] = self.indicators.ema(df['close'], params['ema_slow'])
            
            # Oscillators
            st.write("🌊 Calculating Oscillators...")
            df['RSI'] = self.indicators.rsi(df['close'], params['rsi_period'])
            df['Stoch_K'], df['Stoch_D'] = self.indicators.stochastic(
                df['high'], df['low'], df['close'], params['stoch_k'], params['stoch_d'])
            df['Williams_R'] = self.indicators.williams_r(
                df['high'], df['low'], df['close'], params['williams_period'])
            df['CCI'] = self.indicators.cci(
                df['high'], df['low'], df['close'], params['cci_period'])
            
            # MACD
            st.write("📊 Calculating MACD...")
            df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = self.indicators.macd(
                df['close'], params['macd_fast'], params['macd_slow'], params['macd_signal'])
            
            # Bollinger Bands
            st.write("📏 Calculating Bollinger Bands...")
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.indicators.bollinger_bands(
                df['close'], params['bb_period'], params['bb_std'])
            
            # Volatility and Trend
            st.write("📈 Calculating ATR and ADX...")
            df['ATR'] = self.indicators.atr(df['high'], df['low'], df['close'], params['atr_period'])
            df['ADX'], df['DI_Plus'], df['DI_Minus'] = self.indicators.adx(
                df['high'], df['low'], df['close'], params['adx_period'])
            
            st.write("✅ All indicators calculated successfully!")
            st.write(f"Final DataFrame shape: {df.shape}")
            
            # Check for any remaining NaN or infinity values
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    nan_count = df[col].isna().sum()
                    inf_count = np.isinf(df[col]).sum()
                    if nan_count > 0 or inf_count > 0:
                        st.write(f"⚠️ {col}: {nan_count} NaN, {inf_count} infinite values")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error calculating indicators: {str(e)}")
            st.write(f"Error details: {type(e).__name__}")
            import traceback
            st.write(f"Traceback: {traceback.format_exc()}")
            raise
    
    def generate_signals(self, df, params, direction='both'):
        df = self.calculate_all_indicators(df, params)
        
        # Long signals
        long_conditions = [
            df['close'] > df['SMA_fast'],
            df['SMA_fast'] > df['SMA_slow'],
            df['EMA_fast'] > df['EMA_slow'],
            df['RSI'] < params['rsi_oversold'],
            df['Stoch_K'] < params['stoch_oversold'],
            df['Williams_R'] < -80,
            df['CCI'] < -100,
            df['MACD'] > df['MACD_Signal'],
            df['close'] < df['BB_Lower'],
            df['ADX'] > params['adx_threshold'],
            df['DI_Plus'] > df['DI_Minus']
        ]
        
        # Short signals
        short_conditions = [
            df['close'] < df['SMA_fast'],
            df['SMA_fast'] < df['SMA_slow'],
            df['EMA_fast'] < df['EMA_slow'],
            df['RSI'] > params['rsi_overbought'],
            df['Stoch_K'] > params['stoch_overbought'],
            df['Williams_R'] > -20,
            df['CCI'] > 100,
            df['MACD'] < df['MACD_Signal'],
            df['close'] > df['BB_Upper'],
            df['ADX'] > params['adx_threshold'],
            df['DI_Minus'] > df['DI_Plus']
        ]
        
        # Combine signals based on minimum required conditions
        long_signal = sum(long_conditions) >= params['min_long_conditions']
        short_signal = sum(short_conditions) >= params['min_short_conditions']
        
        if direction == 'long':
            df['signal'] = np.where(long_signal, 1, 0)
        elif direction == 'short':
            df['signal'] = np.where(short_signal, -1, 0)
        else:  # both
            df['signal'] = np.where(long_signal, 1, np.where(short_signal, -1, 0))
        
        return df

class BacktestEngine:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        
    def run_backtest(self, df, strategy_params, direction='both'):
        st.write(f"🚀 **Starting backtest...**")
        st.write(f"Backtest data shape: {df.shape}")
        st.write(f"Direction: {direction}")
        st.write(f"Initial capital: ${self.initial_capital:,}")
        
        results = []
        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        trades = []
        
        # Ensure we have signals
        if 'signal' not in df.columns:
            st.error("❌ No signal column found in DataFrame")
            return [], capital
        
        signal_counts = df['signal'].value_counts()
        st.write(f"Signal distribution: {dict(signal_counts)}")
        
        for i in range(1, len(df)):
            try:
                current_row = df.iloc[i]
                prev_row = df.iloc[i-1]
                
                # Entry logic
                if position == 0 and current_row['signal'] != 0:
                    position = int(current_row['signal'])  # Ensure integer
                    entry_price = float(current_row['close'])  # Ensure float
                    entry_date = current_row['date']
                    
                    # Calculate targets and stop loss
                    atr = float(current_row['ATR']) if not pd.isna(current_row['ATR']) else entry_price * 0.02
                    
                    if position == 1:  # Long
                        target = entry_price * (1 + strategy_params['long_target_pct'] / 100)
                        stop_loss = entry_price * (1 - strategy_params['long_sl_pct'] / 100)
                    else:  # Short
                        target = entry_price * (1 - strategy_params['short_target_pct'] / 100)
                        stop_loss = entry_price * (1 + strategy_params['short_sl_pct'] / 100)
                    
                    trade_info = {
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'position': position,
                        'target': target,
                        'stop_loss': stop_loss,
                        'atr': atr
                    }
                    
                    if i % 50 == 0:  # Log every 50th entry
                        st.write(f"Entry at index {i}: {position} position at ${entry_price:.2f}")
                
                # Exit logic
                elif position != 0:
                    exit_triggered = False
                    exit_reason = ""
                    exit_price = float(current_row['close'])
                    
                    current_high = float(current_row['high'])
                    current_low = float(current_row['low'])
                    
                    if position == 1:  # Long position
                        if current_high >= target:
                            exit_price = target
                            exit_reason = "Target Hit"
                            exit_triggered = True
                        elif current_low <= stop_loss:
                            exit_price = stop_loss
                            exit_reason = "Stop Loss Hit"
                            exit_triggered = True
                        elif current_row['signal'] == -1:
                            exit_reason = "Signal Reversal"
                            exit_triggered = True
                    
                    elif position == -1:  # Short position
                        if current_low <= target:
                            exit_price = target
                            exit_reason = "Target Hit"
                            exit_triggered = True
                        elif current_high >= stop_loss:
                            exit_price = stop_loss
                            exit_reason = "Stop Loss Hit"
                            exit_triggered = True
                        elif current_row['signal'] == 1:
                            exit_reason = "Signal Reversal"
                            exit_triggered = True
                    
                    if exit_triggered:
                        # Calculate PnL
                        if position == 1:
                            pnl = (exit_price - entry_price) / entry_price
                        else:
                            pnl = (entry_price - exit_price) / entry_price
                        
                        capital = capital * (1 + pnl)
                        hold_days = (current_row['date'] - entry_date).days
                        
                        trade_info.update({
                            'exit_date': current_row['date'],
                            'exit_price': exit_price,
                            'exit_reason': exit_reason,
                            'pnl_pct': pnl * 100,
                            'capital': capital,
                            'hold_days': hold_days
                        })
                        
                        trades.append(trade_info)
                        position = 0
                        
                        if len(trades) % 10 == 0:  # Log every 10th trade
                            st.write(f"Trade #{len(trades)}: {exit_reason}, PnL: {pnl*100:.2f}%")
            
            except Exception as e:
                st.write(f"❌ Error at index {i}: {str(e)}")
                continue
        
        st.write(f"✅ Backtest completed!")
        st.write(f"Total trades executed: {len(trades)}")
        st.write(f"Final capital: ${capital:,.2f}")
        
        return trades, capital
    
    def calculate_metrics(self, trades, df):
        st.write(f"📊 **Calculating performance metrics...**")
        st.write(f"Number of trades: {len(trades)}")
        
        if not trades:
            st.write("⚠️ No trades to analyze")
            return {}
        
        try:
            total_trades = len(trades)
            winning_trades = [t for t in trades if t['pnl_pct'] > 0]
            losing_trades = [t for t in trades if t['pnl_pct'] < 0]
            
            win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
            avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0
            
            total_return = (trades[-1]['capital'] / 100000 - 1) * 100
            
            # Buy and hold return - ensure numeric calculation
            first_price = float(df['close'].iloc[0])
            last_price = float(df['close'].iloc[-1])
            buy_hold_return = (last_price / first_price - 1) * 100
            
            # Average holding period
            avg_hold_days = np.mean([t['hold_days'] for t in trades])
            
            # Profit factor calculation
            if losing_trades and avg_loss != 0:
                profit_factor = abs((avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)))
            else:
                profit_factor = float('inf')
            
            metrics = {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_return': total_return,
                'buy_hold_return': buy_hold_return,
                'avg_hold_days': avg_hold_days,
                'profit_factor': profit_factor
            }
            
            st.write(f"✅ Metrics calculated: Win rate: {win_rate:.1f}%, Total return: {total_return:.2f}%")
            return metrics
            
        except Exception as e:
            st.error(f"❌ Error calculating metrics: {str(e)}")
            import traceback
            st.write(f"Traceback: {traceback.format_exc()}")
            return {}

def clean_numeric_data(value):
    """Clean and convert numeric data to float"""
    if pd.isna(value) or value == '':
        return np.nan
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Remove commas, spaces, and other non-numeric characters except decimal point
        cleaned = str(value).replace(',', '').replace(' ', '').strip()
        # Remove currency symbols and percentage signs
        cleaned = cleaned.replace('$', '').replace('%', '').replace('₹', '').replace('€', '')
        
        try:
            return float(cleaned)
        except ValueError:
            st.write(f"⚠️ Warning: Could not convert '{value}' to float")
            return np.nan
    
    return np.nan

def map_columns(df):
    """Map various column names to standardized format and clean data"""
    st.write("🔍 **Starting column mapping process...**")
    st.write(f"Original columns: {list(df.columns)}")
    
    column_mapping = {}
    
    for col in df.columns:
        col_lower = col.lower().strip()
        
        if 'open' in col_lower:
            column_mapping[col] = 'open'
        elif 'high' in col_lower:
            column_mapping[col] = 'high'
        elif 'low' in col_lower:
            column_mapping[col] = 'low'
        elif 'close' in col_lower:
            column_mapping[col] = 'close'
        elif 'volume' in col_lower or 'vol' in col_lower:
            column_mapping[col] = 'volume'
        elif 'date' in col_lower or 'time' in col_lower or col_lower in ['timestamp', 'datetime']:
            column_mapping[col] = 'date'
    
    st.write(f"Column mapping: {column_mapping}")
    
    df_renamed = df.rename(columns=column_mapping)
    st.write(f"Columns after mapping: {list(df_renamed.columns)}")
    
    # Clean numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df_renamed.columns:
            st.write(f"🧹 Cleaning numeric data in column: {col}")
            st.write(f"Sample original values in {col}: {df_renamed[col].head().tolist()}")
            
            df_renamed[col] = df_renamed[col].apply(clean_numeric_data)
            
            st.write(f"Sample cleaned values in {col}: {df_renamed[col].head().tolist()}")
            st.write(f"Data type after cleaning {col}: {df_renamed[col].dtype}")
            
            # Check for NaN values
            nan_count = df_renamed[col].isna().sum()
            if nan_count > 0:
                st.write(f"⚠️ Warning: {nan_count} NaN values found in {col}")
    
    return df_renamed

def optimize_strategy(df, direction, search_type, n_iterations=50):
    """Optimize strategy parameters using grid search or random search"""
    
    st.write(f"🔧 **Starting optimization with {search_type}...**")
    st.write(f"Data shape for optimization: {df.shape}")
    st.write(f"Direction: {direction}")
    st.write(f"Iterations: {n_iterations}")
    
    param_ranges = {
        'sma_fast': [5, 10, 15, 20],
        'sma_slow': [20, 30, 50, 100],
        'ema_fast': [8, 12, 16, 21],
        'ema_slow': [26, 34, 50, 89],
        'rsi_period': [10, 14, 21],
        'rsi_oversold': [20, 25, 30],
        'rsi_overbought': [70, 75, 80],
        'stoch_k': [10, 14, 21],
        'stoch_d': [3, 5, 8],
        'stoch_oversold': [15, 20, 25],
        'stoch_overbought': [75, 80, 85],
        'williams_period': [10, 14, 21],
        'cci_period': [14, 20, 28],
        'macd_fast': [8, 12, 16],
        'macd_slow': [21, 26, 34],
        'macd_signal': [6, 9, 12],
        'bb_period': [15, 20, 25],
        'bb_std': [1.5, 2, 2.5],
        'atr_period': [10, 14, 21],
        'adx_period': [10, 14, 21],
        'adx_threshold': [20, 25, 30],
        'min_long_conditions': [3, 4, 5, 6],
        'min_short_conditions': [3, 4, 5, 6],
        'long_target_pct': [2, 3, 5, 8],
        'long_sl_pct': [1, 2, 3, 4],
        'short_target_pct': [2, 3, 5, 8],
        'short_sl_pct': [1, 2, 3, 4]
    }
    
    strategy = SwingTradingStrategy()
    backtest = BacktestEngine()
    
    best_return = -float('inf')
    best_params = None
    best_metrics = None
    successful_tests = 0
    failed_tests = 0
    
    if search_type == "Grid Search":
        st.write("🔍 Starting Grid Search...")
        # Reduced parameter space for grid search
        reduced_ranges = {k: v[:2] for k, v in param_ranges.items()}
        param_combinations = list(product(*reduced_ranges.values()))
        random.shuffle(param_combinations)
        param_combinations = param_combinations[:n_iterations]
        
        for i, params_tuple in enumerate(param_combinations):
            if i % 10 == 0:  # Progress update every 10 iterations
                st.write(f"Progress: {i}/{len(param_combinations)} combinations tested")
            
            params = dict(zip(reduced_ranges.keys(), params_tuple))
            
            try:
                st.write(f"Testing parameters: {params}")
                df_with_signals = strategy.generate_signals(df, params, direction.lower())
                trades, final_capital = backtest.run_backtest(df_with_signals, params, direction.lower())
                
                if trades:
                    metrics = backtest.calculate_metrics(trades, df_with_signals)
                    successful_tests += 1
                    st.write(f"✅ Test successful. Return: {metrics['total_return']:.2f}%")
                    
                    if metrics['total_return'] > best_return:
                        best_return = metrics['total_return']
                        best_params = params
                        best_metrics = metrics
                        st.write(f"🏆 New best strategy found! Return: {best_return:.2f}%")
                else:
                    st.write("⚠️ No trades generated")
            except Exception as e:
                failed_tests += 1
                st.write(f"❌ Test failed: {str(e)}")
                continue
    
    else:  # Random Search
        st.write("🎲 Starting Random Search...")
        for i in range(n_iterations):
            if i % 10 == 0:  # Progress update every 10 iterations
                st.write(f"Progress: {i}/{n_iterations} iterations completed")
            
            params = {k: random.choice(v) for k, v in param_ranges.items()}
            
            # Ensure logical constraints
            if params['sma_fast'] >= params['sma_slow']:
                params['sma_slow'] = params['sma_fast'] + 10
            if params['ema_fast'] >= params['ema_slow']:
                params['ema_slow'] = params['ema_fast'] + 10
            
            try:
                df_with_signals = strategy.generate_signals(df, params, direction.lower())
                trades, final_capital = backtest.run_backtest(df_with_signals, params, direction.lower())
                
                if trades:
                    metrics = backtest.calculate_metrics(trades, df_with_signals)
                    successful_tests += 1
                    
                    if metrics['total_return'] > best_return:
                        best_return = metrics['total_return']
                        best_params = params
                        best_metrics = metrics
                        st.write(f"🏆 New best strategy! Iteration {i+1}, Return: {best_return:.2f}%")
            except Exception as e:
                failed_tests += 1
                continue
    
    st.write(f"🎯 **Optimization completed!**")
    st.write(f"Successful tests: {successful_tests}")
    st.write(f"Failed tests: {failed_tests}")
    st.write(f"Best return found: {best_return:.2f}%" if best_return != -float('inf') else "No profitable strategy found")
    
    return best_params, best_metrics(param_combinations)
        param_combinations = param_combinations[:n_iterations]
        
        for params_tuple in param_combinations:
            params = dict(zip(reduced_ranges.keys(), params_tuple))
            
            try:
                df_with_signals = strategy.generate_signals(df, params, direction.lower())
                trades, final_capital = backtest.run_backtest(df_with_signals, params, direction.lower())
                
                if trades:
                    metrics = backtest.calculate_metrics(trades, df_with_signals)
                    if metrics['total_return'] > best_return:
                        best_return = metrics['total_return']
                        best_params = params
                        best_metrics = metrics
            except:
                continue
    
    else:  # Random Search
        for _ in range(n_iterations):
            params = {k: random.choice(v) for k, v in param_ranges.items()}
            
            # Ensure logical constraints
            if params['sma_fast'] >= params['sma_slow']:
                params['sma_slow'] = params['sma_fast'] + 10
            if params['ema_fast'] >= params['ema_slow']:
                params['ema_slow'] = params['ema_fast'] + 10
            
            try:
                df_with_signals = strategy.generate_signals(df, params, direction.lower())
                trades, final_capital = backtest.run_backtest(df_with_signals, params, direction.lower())
                
                if trades:
                    metrics = backtest.calculate_metrics(trades, df_with_signals)
                    if metrics['total_return'] > best_return:
                        best_return = metrics['total_return']
                        best_params = params
                        best_metrics = metrics
            except:
                continue
    
    return best_params, best_metrics

def generate_live_recommendation(df, best_params, direction):
    """Generate live recommendation for the next trading day"""
    strategy = SwingTradingStrategy()
    df_with_signals = strategy.generate_signals(df, best_params, direction.lower())
    
    latest_data = df_with_signals.iloc[-1]
    signal = latest_data['signal']
    
    if signal == 0:
        return None
    
    entry_price = latest_data['close']
    atr = latest_data['ATR']
    next_date = latest_data['date'] + timedelta(days=1)
    
    if signal == 1:  # Long
        target = entry_price * (1 + best_params['long_target_pct'] / 100)
        stop_loss = entry_price * (1 - best_params['long_sl_pct'] / 100)
        direction_text = "LONG"
    else:  # Short
        target = entry_price * (1 - best_params['short_target_pct'] / 100)
        stop_loss = entry_price * (1 + best_params['short_sl_pct'] / 100)
        direction_text = "SHORT"
    
    # Calculate probability based on indicator confluence
    indicators_positive = 0
    total_indicators = 11
    
    conditions = [
        latest_data['close'] > latest_data['SMA_fast'] if signal == 1 else latest_data['close'] < latest_data['SMA_fast'],
        latest_data['SMA_fast'] > latest_data['SMA_slow'] if signal == 1 else latest_data['SMA_fast'] < latest_data['SMA_slow'],
        latest_data['EMA_fast'] > latest_data['EMA_slow'] if signal == 1 else latest_data['EMA_fast'] < latest_data['EMA_slow'],
        latest_data['RSI'] < best_params['rsi_oversold'] if signal == 1 else latest_data['RSI'] > best_params['rsi_overbought'],
        latest_data['Stoch_K'] < best_params['stoch_oversold'] if signal == 1 else latest_data['Stoch_K'] > best_params['stoch_overbought'],
        latest_data['Williams_R'] < -80 if signal == 1 else latest_data['Williams_R'] > -20,
        latest_data['CCI'] < -100 if signal == 1 else latest_data['CCI'] > 100,
        latest_data['MACD'] > latest_data['MACD_Signal'] if signal == 1 else latest_data['MACD'] < latest_data['MACD_Signal'],
        latest_data['close'] < latest_data['BB_Lower'] if signal == 1 else latest_data['close'] > latest_data['BB_Upper'],
        latest_data['ADX'] > best_params['adx_threshold'],
        latest_data['DI_Plus'] > latest_data['DI_Minus'] if signal == 1 else latest_data['DI_Minus'] > latest_data['DI_Plus']
    ]
    
    indicators_positive = sum(conditions)
    probability = (indicators_positive / total_indicators) * 100
    
    logic = f"Based on {indicators_positive}/{total_indicators} confirming indicators: "
    if latest_data['RSI'] < 30 and signal == 1:
        logic += "RSI oversold, "
    if latest_data['close'] < latest_data['BB_Lower'] and signal == 1:
        logic += "Price below BB lower band, "
    if latest_data['MACD'] > latest_data['MACD_Signal'] and signal == 1:
        logic += "MACD bullish crossover, "
    
    return {
        'date': next_date,
        'direction': direction_text,
        'entry_price': entry_price,
        'target': target,
        'stop_loss': stop_loss,
        'probability': probability,
        'logic': logic.rstrip(', '),
        'atr': atr
    }

def create_returns_heatmap(df):
    """Create monthly returns heatmap"""
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['returns'] = df['close'].pct_change()
    
    monthly_returns = df.groupby(['year', 'month'])['returns'].sum() * 100
    
    if len(monthly_returns) > 12:  # More than 1 year of data
        pivot_table = monthly_returns.unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax)
        ax.set_title('Monthly Returns Heatmap (%)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        
        return fig
    return None

def main():
    st.title("🎯 Advanced Swing Trading Recommender")
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Stock Data CSV", type=['csv'])
    
    if uploaded_file is not None:
        # Load and process data
        df = pd.read_csv(uploaded_file)
        st.success(f"Data loaded successfully! Shape: {df.shape}")
        
        # Display original data info
        st.subheader("📊 Data Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**First 5 rows:**")
            st.dataframe(df.head())
        
        with col2:
            st.write("**Last 5 rows:**")
            st.dataframe(df.tail())
        
        # Map columns
        df = map_columns(df)
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'date']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.info("Make sure your CSV contains columns for: Open, High, Low, Close, Volume, Date")
            return
        
        # Convert date column and sort
        st.write("📅 **Processing date column...**")
        st.write(f"Date column before processing: {df['date'].dtype}")
        st.write(f"Sample date values: {df['date'].head().tolist()}")
        
        try:
            df['date'] = pd.to_datetime(df['date'])
            st.write("✅ Date column converted successfully")
        except Exception as e:
            st.error(f"❌ Error converting date column: {e}")
            return
        
        df = df.sort_values('date').reset_index(drop=True)
        st.write(f"📊 Data sorted by date. Shape: {df.shape}")
        
        # Remove rows with NaN values in critical columns
        before_dropna = len(df)
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        after_dropna = len(df)
        if before_dropna != after_dropna:
            st.write(f"🧹 Removed {before_dropna - after_dropna} rows with missing price data")
        
        # Ensure no negative or zero prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            negative_count = (df[col] <= 0).sum()
            if negative_count > 0:
                st.write(f"⚠️ Warning: {negative_count} non-positive values in {col}")
                df = df[df[col] > 0]  # Remove non-positive prices
        
        st.write(f"✅ Final cleaned data shape: {df.shape}")
        
        # Data statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min Date", df['date'].min().strftime('%Y-%m-%d'))
        with col2:
            st.metric("Max Date", df['date'].max().strftime('%Y-%m-%d'))
        with col3:
            st.metric("Min Price", f"${df['close'].min():.2f}")
        with col4:
            st.metric("Max Price", f"${df['close'].max():.2f}")
        
        # Plot raw data
        st.subheader("📈 Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df['date'],
                                   open=df['open'],
                                   high=df['high'],
                                   low=df['low'],
                                   close=df['close'],
                                   name='Price'))
        fig.update_layout(title="Stock Price Chart", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
        
        # Exploratory Data Analysis
        st.subheader("🔍 Exploratory Data Analysis")
        
        # Returns heatmap
        heatmap_fig = create_returns_heatmap(df)
        if heatmap_fig:
            st.pyplot(heatmap_fig)
        else:
            st.info("Need more than 1 year of data for monthly returns heatmap")
        
        # Summary statistics
        st.write("**Price Statistics:**")
        price_stats = df[['open', 'high', 'low', 'close', 'volume']].describe()
        st.dataframe(price_stats)
        
        # Data summary
        total_days = (df['date'].max() - df['date'].min()).days
        avg_volume = df['volume'].mean()
        price_change = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
        volatility = df['close'].pct_change().std() * np.sqrt(252) * 100  # Annualized volatility
        
        summary_text = f"""
        **Data Summary (100 words):**
        
        The dataset contains {len(df)} trading days spanning {total_days} days. The stock shows a {price_change:.1f}% total return 
        over this period with an annualized volatility of {volatility:.1f}%. Average daily volume is {avg_volume:,.0f} shares. 
        The price ranged from ${df['close'].min():.2f} to ${df['close'].max():.2f}. Current market conditions show 
        {'bullish' if price_change > 0 else 'bearish'} momentum. The data quality appears suitable for swing trading analysis 
        with consistent volume patterns and price action. Opportunities exist for both long and short strategies depending 
        on technical indicator alignment and risk management parameters.
        """
        st.markdown(summary_text)
        
        # Strategy Configuration
        st.subheader("⚙️ Strategy Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            end_date = st.date_input("End Date for Backtesting", 
                                   value=df['date'].max().date(),
                                   min_value=df['date'].min().date(),
                                   max_value=df['date'].max().date())
        
        with col2:
            direction = st.selectbox("Trading Direction", ['Both', 'Long', 'Short'])
        
        with col3:
            search_type = st.selectbox("Optimization Method", ['Random Search', 'Grid Search'])
        
        # Filter data based on end date
        backtest_df = df[df['date'] <= pd.to_datetime(end_date)].copy()
        
        if len(backtest_df) < 100:
            st.error("Need at least 100 days of data for backtesting")
            return
        
        if st.button("🚀 Run Strategy Optimization & Backtesting", type="primary"):
            with st.spinner("Optimizing strategy parameters..."):
                
                # Optimize strategy
                best_params, best_metrics = optimize_strategy(
                    backtest_df, direction, search_type, n_iterations=100)
                
                if best_params is None:
                    st.error("No profitable strategy found. Try different parameters or data.")
                    return
                
                # Display best strategy details
                st.subheader("🏆 Best Strategy Found")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Key Parameters:**")
                    st.write(f"- SMA Fast: {best_params['sma_fast']}")
                    st.write(f"- SMA Slow: {best_params['sma_slow']}")
                    st.write(f"- RSI Period: {best_params['rsi_period']}")
                    st.write(f"- RSI Oversold: {best_params['rsi_oversold']}")
                    st.write(f"- RSI Overbought: {best_params['rsi_overbought']}")
                    st.write(f"- MACD Fast: {best_params['macd_fast']}")
                    st.write(f"- MACD Slow: {best_params['macd_slow']}")
                    st.write(f"- MACD Signal: {best_params['macd_signal']}")
                
                with col2:
                    st.write("**Risk Management:**")
                    st.write(f"- Long Target: {best_params['long_target_pct']}%")
                    st.write(f"- Long Stop Loss: {best_params['long_sl_pct']}%")
                    st.write(f"- Short Target: {best_params['short_target_pct']}%")
                    st.write(f"- Short Stop Loss: {best_params['short_sl_pct']}%")
                    st.write(f"- Min Long Conditions: {best_params['min_long_conditions']}")
                    st.write(f"- Min Short Conditions: {best_params['min_short_conditions']}")
                    st.write(f"- ADX Threshold: {best_params['adx_threshold']}")
                
                # Run full backtest with best parameters
                strategy = SwingTradingStrategy()
                backtest = BacktestEngine()
                
                df_with_signals = strategy.generate_signals(backtest_df, best_params, direction.lower())
                trades, final_capital = backtest.run_backtest(df_with_signals, best_params, direction.lower())
                
                # Display backtest results
                st.subheader("📊 Backtest Results")
                
                if trades:
                    metrics = backtest.calculate_metrics(trades, df_with_signals)
                    
                    # Performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Return", f"{metrics['total_return']:.2f}%")
                        st.metric("Buy & Hold Return", f"{metrics['buy_hold_return']:.2f}%")
                    
                    with col2:
                        st.metric("Total Trades", metrics['total_trades'])
                        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                    
                    with col3:
                        st.metric("Winning Trades", metrics['winning_trades'])
                        st.metric("Losing Trades", metrics['losing_trades'])
                    
                    with col4:
                        st.metric("Avg Win", f"{metrics['avg_win']:.2f}%")
                        st.metric("Avg Loss", f"{metrics['avg_loss']:.2f}%")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Avg Hold Days", f"{metrics['avg_hold_days']:.1f}")
                    with col2:
                        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                    
                    # Detailed trades table
                    st.subheader("📋 Detailed Trade History")
                    
                    trades_df = pd.DataFrame(trades)
                    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date']).dt.strftime('%Y-%m-%d')
                    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date']).dt.strftime('%Y-%m-%d')
                    
                    display_df = trades_df[['entry_date', 'exit_date', 'position', 'entry_price', 'exit_price', 
                                          'target', 'stop_loss', 'exit_reason', 'pnl_pct', 'hold_days']].copy()
                    
                    display_df['position'] = display_df['position'].map({1: 'LONG', -1: 'SHORT'})
                    display_df.columns = ['Entry Date', 'Exit Date', 'Direction', 'Entry Price', 'Exit Price', 
                                        'Target', 'Stop Loss', 'Exit Reason', 'PnL %', 'Hold Days']
                    
                    # Format numeric columns
                    for col in ['Entry Price', 'Exit Price', 'Target', 'Stop Loss']:
                        display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
                    display_df['PnL %'] = display_df['PnL %'].apply(lambda x: f"{x:.2f}%")
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Add logic/reason for each trade
                    st.subheader("🧠 Trade Logic & Probability Analysis")
                    
                    for i, trade in enumerate(trades[:5]):  # Show first 5 trades for brevity
                        with st.expander(f"Trade #{i+1} - {trade['position']} on {trade['entry_date'].strftime('%Y-%m-%d')}"):
                            
                            # Find the row for this trade
                            trade_row_idx = df_with_signals[df_with_signals['date'] == trade['entry_date']].index[0]
                            trade_data = df_with_signals.iloc[trade_row_idx]
                            
                            st.write(f"**Entry Logic:**")
                            st.write(f"- Entry Price: ${trade['entry_price']:.2f}")
                            st.write(f"- Target: ${trade['target']:.2f}")
                            st.write(f"- Stop Loss: ${trade['stop_loss']:.2f}")
                            st.write(f"- ATR: {trade['atr']:.2f}")
                            
                            # Calculate probability based on indicators
                            indicators_met = []
                            if trade['position'] == 1:  # Long
                                if trade_data['RSI'] < best_params['rsi_oversold']:
                                    indicators_met.append("RSI Oversold")
                                if trade_data['close'] < trade_data['BB_Lower']:
                                    indicators_met.append("Below BB Lower Band")
                                if trade_data['MACD'] > trade_data['MACD_Signal']:
                                    indicators_met.append("MACD Bullish")
                                if trade_data['Stoch_K'] < best_params['stoch_oversold']:
                                    indicators_met.append("Stochastic Oversold")
                            else:  # Short
                                if trade_data['RSI'] > best_params['rsi_overbought']:
                                    indicators_met.append("RSI Overbought")
                                if trade_data['close'] > trade_data['BB_Upper']:
                                    indicators_met.append("Above BB Upper Band")
                                if trade_data['MACD'] < trade_data['MACD_Signal']:
                                    indicators_met.append("MACD Bearish")
                                if trade_data['Stoch_K'] > best_params['stoch_overbought']:
                                    indicators_met.append("Stochastic Overbought")
                            
                            probability = min(len(indicators_met) * 20, 95)  # Cap at 95%
                            
                            st.write(f"**Confirming Indicators:** {', '.join(indicators_met)}")
                            st.write(f"**Estimated Probability:** {probability}%")
                            st.write(f"**Actual Result:** {trade['exit_reason']} - {trade['pnl_pct']:.2f}% PnL")
                    
                    # Performance chart
                    st.subheader("📈 Strategy Performance Chart")
                    
                    # Calculate cumulative returns
                    equity_curve = [100000]  # Starting capital
                    for trade in trades:
                        equity_curve.append(trade['capital'])
                    
                    trade_dates = [backtest_df['date'].iloc[0]] + [trade['exit_date'] for trade in trades]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=trade_dates, y=equity_curve, 
                                           mode='lines', name='Strategy', line=dict(color='blue', width=2)))
                    
                    # Buy and hold comparison
                    buy_hold_values = []
                    initial_price = backtest_df['close'].iloc[0]
                    for date in trade_dates:
                        if date in backtest_df['date'].values:
                            current_price = backtest_df[backtest_df['date'] == date]['close'].iloc[0]
                            buy_hold_values.append(100000 * (current_price / initial_price))
                        else:
                            buy_hold_values.append(buy_hold_values[-1])
                    
                    fig.add_trace(go.Scatter(x=trade_dates, y=buy_hold_values, 
                                           mode='lines', name='Buy & Hold', line=dict(color='gray', width=2)))
                    
                    fig.update_layout(title="Strategy vs Buy & Hold Performance", 
                                    xaxis_title="Date", yaxis_title="Portfolio Value ($)")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Generate live recommendation
                st.subheader("🎯 Live Recommendation")
                
                # Use full dataset for live recommendation
                live_recommendation = generate_live_recommendation(df, best_params, direction)
                
                if live_recommendation:
                    st.success("📢 **TRADING SIGNAL DETECTED!**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Direction", live_recommendation['direction'])
                        st.metric("Entry Price", f"${live_recommendation['entry_price']:.2f}")
                    
                    with col2:
                        st.metric("Target Price", f"${live_recommendation['target']:.2f}")
                        st.metric("Stop Loss", f"${live_recommendation['stop_loss']:.2f}")
                    
                    with col3:
                        st.metric("Probability", f"{live_recommendation['probability']:.1f}%")
                        st.metric("Expected Date", live_recommendation['date'].strftime('%Y-%m-%d'))
                    
                    st.write("**📝 Entry Logic:**")
                    st.write(live_recommendation['logic'])
                    st.write(f"**📊 ATR:** {live_recommendation['atr']:.2f}")
                    
                    # Risk/Reward calculation
                    if live_recommendation['direction'] == 'LONG':
                        risk = live_recommendation['entry_price'] - live_recommendation['stop_loss']
                        reward = live_recommendation['target'] - live_recommendation['entry_price']
                    else:
                        risk = live_recommendation['stop_loss'] - live_recommendation['entry_price']
                        reward = live_recommendation['entry_price'] - live_recommendation['target']
                    
                    risk_reward = reward / risk if risk > 0 else 0
                    st.write(f"**⚖️ Risk/Reward Ratio:** {risk_reward:.2f}")
                    
                else:
                    st.info("🔍 No trading signals detected for the next session. Continue monitoring.")
                
                # Final Summary
                st.subheader("📝 Strategy Summary & Recommendations")
                
                if trades and metrics['total_return'] > metrics['buy_hold_return'] * 1.7:  # Beat by 70%
                    performance_text = "excellent"
                    beat_margin = ((metrics['total_return'] / metrics['buy_hold_return']) - 1) * 100
                elif trades and metrics['total_return'] > metrics['buy_hold_return']:
                    performance_text = "good"
                    beat_margin = metrics['total_return'] - metrics['buy_hold_return']
                else:
                    performance_text = "underperforming"
                    beat_margin = metrics['total_return'] - metrics['buy_hold_return']
                
                final_summary = f"""
                **📊 Backtest Analysis Summary:**
                
                The optimized swing trading strategy delivered {performance_text} results with a {metrics['total_return']:.1f}% 
                total return compared to {metrics['buy_hold_return']:.1f}% for buy-and-hold, {'outperforming' if beat_margin > 0 else 'underperforming'} 
                by {abs(beat_margin):.1f} percentage points. The strategy executed {metrics['total_trades']} trades with a 
                {metrics['win_rate']:.1f}% success rate and {metrics['avg_hold_days']:.1f} days average holding period.
                
                **🎯 Live Trading Recommendations:**
                
                {'A ' + live_recommendation['direction'] + ' signal is currently active' if live_recommendation else 'No immediate signals detected'}. 
                The strategy uses {best_params['min_long_conditions']} confirming indicators for long positions and 
                {best_params['min_short_conditions']} for short positions. Risk management employs {best_params['long_sl_pct']}% 
                stop losses with {best_params['long_target_pct']}% profit targets for long trades. 
                Continue monitoring daily for new opportunities and strictly follow risk management rules.
                
                **⚠️ Key Success Factors:**
                - Maintain disciplined entry/exit execution
                - Never risk more than predetermined stop loss levels  
                - Allow winners to reach full profit targets when possible
                - Monitor market conditions for strategy effectiveness changes
                """
                
                st.markdown(final_summary)
                
                # Additional metrics in expandable section
                with st.expander("📊 Additional Strategy Details"):
                    st.write("**Complete Parameter Set:**")
                    for key, value in best_params.items():
                        st.write(f"- {key.replace('_', ' ').title()}: {value}")
                    
                    if trades:
                        st.write(f"\n**Performance Metrics:**")
                        st.write(f"- Sharpe Ratio Estimate: {(metrics['total_return'] / (metrics['total_return'] - metrics['buy_hold_return']) if metrics['total_return'] != metrics['buy_hold_return'] else 1):.2f}")
                        st.write(f"- Maximum Drawdown: To be calculated with more data")
                        st.write(f"- Average Trade Duration: {metrics['avg_hold_days']:.1f} days")
                        st.write(f"- Best Trade: {max([t['pnl_pct'] for t in trades]):.2f}%")
                        st.write(f"- Worst Trade: {min([t['pnl_pct'] for t in trades]):.2f}%")

if __name__ == "__main__":
    main()
