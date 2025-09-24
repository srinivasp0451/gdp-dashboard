import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import warnings
warnings.filterwarnings('ignore')
import pytz
from scipy.signal import argrelextrema
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Advanced Swing Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .success-metric {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .danger-metric {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

class TechnicalAnalysis:
    def __init__(self, data):
        self.data = data.copy()
        self.data = self.data.sort_values('date').reset_index(drop=True)
    
    def calculate_sma(self, period):
        return self.data['close'].rolling(window=period).mean()
    
    def calculate_ema(self, period):
        return self.data['close'].ewm(span=period).mean()
    
    def calculate_rsi(self, period=14):
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, period=20, std_dev=2):
        sma = self.calculate_sma(period)
        std = self.data['close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        ema_fast = self.calculate_ema(fast)
        ema_slow = self.calculate_ema(slow)
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def find_support_resistance(self, window=20):
        highs = self.data['high'].rolling(window=window).max()
        lows = self.data['low'].rolling(window=window).min()
        
        # Find local maxima and minima
        local_max = argrelextrema(self.data['high'].values, np.greater, order=window//2)[0]
        local_min = argrelextrema(self.data['low'].values, np.less, order=window//2)[0]
        
        resistance_levels = self.data.iloc[local_max]['high'].values
        support_levels = self.data.iloc[local_min]['low'].values
        
        return support_levels, resistance_levels, local_min, local_max
    
    def identify_chart_patterns(self):
        patterns = []
        
        # Head and Shoulders Pattern
        for i in range(50, len(self.data) - 50):
            window = self.data.iloc[i-25:i+25]
            if self.is_head_shoulders(window):
                patterns.append({
                    'pattern': 'Head and Shoulders',
                    'date': self.data.iloc[i]['date'],
                    'price': self.data.iloc[i]['close'],
                    'type': 'bearish',
                    'confidence': 0.75
                })
        
        # Double Top/Bottom
        support_levels, resistance_levels, _, _ = self.find_support_resistance()
        
        for i in range(1, len(resistance_levels)):
            if abs(resistance_levels[i] - resistance_levels[i-1]) / resistance_levels[i] < 0.02:
                patterns.append({
                    'pattern': 'Double Top',
                    'date': self.data.iloc[-1]['date'],
                    'price': resistance_levels[i],
                    'type': 'bearish',
                    'confidence': 0.7
                })
        
        return patterns
    
    def is_head_shoulders(self, window):
        if len(window) < 25:
            return False
        
        prices = window['high'].values
        mid = len(prices) // 2
        
        # Check if middle peak is highest
        if prices[mid] > max(prices[:mid//2]) and prices[mid] > max(prices[mid + mid//2:]):
            return True
        return False
    
    def calculate_pivot_points(self):
        pivots = []
        for i in range(1, len(self.data) - 1):
            prev_high = self.data.iloc[i-1]['high']
            curr_high = self.data.iloc[i]['high']
            next_high = self.data.iloc[i+1]['high']
            
            prev_low = self.data.iloc[i-1]['low']
            curr_low = self.data.iloc[i]['low']
            next_low = self.data.iloc[i+1]['low']
            
            # Pivot High
            if curr_high > prev_high and curr_high > next_high:
                pivots.append({
                    'type': 'high',
                    'date': self.data.iloc[i]['date'],
                    'price': curr_high,
                    'index': i
                })
            
            # Pivot Low
            if curr_low < prev_low and curr_low < next_low:
                pivots.append({
                    'type': 'low',
                    'date': self.data.iloc[i]['date'],
                    'price': curr_low,
                    'index': i
                })
        
        return pivots
    
    def identify_trend(self, period=20):
        sma_short = self.calculate_sma(10)
        sma_long = self.calculate_sma(period)
        
        if sma_short.iloc[-1] > sma_long.iloc[-1]:
            return "Uptrend"
        elif sma_short.iloc[-1] < sma_long.iloc[-1]:
            return "Downtrend"
        else:
            return "Sideways"

class PatternRecognition:
    def __init__(self, data):
        self.data = data
    
    def find_triangles(self):
        triangles = []
        support_levels, resistance_levels, support_idx, resistance_idx = TechnicalAnalysis(self.data).find_support_resistance()
        
        # Ascending Triangle
        if len(resistance_levels) >= 2 and len(support_levels) >= 2:
            # Check if resistance is relatively flat and support is rising
            resistance_slope = (resistance_levels[-1] - resistance_levels[0]) / len(resistance_levels)
            support_slope = (support_levels[-1] - support_levels[0]) / len(support_levels)
            
            if abs(resistance_slope) < 0.01 and support_slope > 0.01:
                triangles.append({
                    'type': 'Ascending Triangle',
                    'signal': 'bullish',
                    'confidence': 0.8
                })
        
        return triangles
    
    def find_cup_handle(self):
        patterns = []
        if len(self.data) < 100:
            return patterns
        
        # Look for cup and handle pattern in recent data
        recent_data = self.data.tail(100)
        lows = recent_data['low'].values
        
        # Find the deepest point (cup bottom)
        cup_bottom_idx = np.argmin(lows)
        
        if cup_bottom_idx > 20 and cup_bottom_idx < 80:
            patterns.append({
                'type': 'Cup and Handle',
                'signal': 'bullish',
                'confidence': 0.7,
                'cup_bottom': lows[cup_bottom_idx]
            })
        
        return patterns

class SwingTradingStrategy:
    def __init__(self, data, params=None):
        self.data = data.copy()
        self.params = params or {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'sma_short': 10,
            'sma_long': 20,
            'bb_period': 20,
            'bb_std': 2,
            'stop_loss_pct': 2.0,
            'target_pct': 4.0
        }
        self.ta = TechnicalAnalysis(data)
        
    def generate_signals(self, trade_type='both'):
        signals = []
        
        # Calculate indicators
        rsi = self.ta.calculate_rsi(self.params['rsi_period'])
        sma_short = self.ta.calculate_sma(self.params['sma_short'])
        sma_long = self.ta.calculate_sma(self.params['sma_long'])
        bb_upper, bb_middle, bb_lower = self.ta.calculate_bollinger_bands(
            self.params['bb_period'], self.params['bb_std']
        )
        macd, macd_signal, macd_hist = self.ta.calculate_macd()
        
        # Get support/resistance levels
        support_levels, resistance_levels, _, _ = self.ta.find_support_resistance()
        
        # Identify patterns
        patterns = self.ta.identify_chart_patterns()
        
        for i in range(50, len(self.data)):
            current_price = self.data.iloc[i]['close']
            current_rsi = rsi.iloc[i] if not pd.isna(rsi.iloc[i]) else 50
            current_volume = self.data.iloc[i]['volume']
            avg_volume = self.data.iloc[i-20:i]['volume'].mean()
            
            # Long signals
            if trade_type in ['long', 'both']:
                long_conditions = [
                    current_rsi < self.params['rsi_oversold'],
                    sma_short.iloc[i] > sma_long.iloc[i],
                    current_price > bb_lower.iloc[i],
                    current_volume > avg_volume * 1.2,
                    macd.iloc[i] > macd_signal.iloc[i]
                ]
                
                # Support level check
                near_support = any(abs(current_price - level) / current_price < 0.02 for level in support_levels[-5:])
                
                if sum(long_conditions) >= 3 or near_support:
                    entry_price = current_price
                    stop_loss = entry_price * (1 - self.params['stop_loss_pct'] / 100)
                    target = entry_price * (1 + self.params['target_pct'] / 100)
                    
                    # Calculate probability based on conditions met
                    probability = min(0.95, 0.5 + (sum(long_conditions) * 0.1))
                    
                    logic = self.generate_entry_logic(long_conditions, near_support, 'long', current_rsi, current_volume, avg_volume)
                    
                    signals.append({
                        'date': self.data.iloc[i]['date'],
                        'type': 'long',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'target': target,
                        'probability': probability,
                        'logic': logic,
                        'rsi': current_rsi,
                        'volume_ratio': current_volume / avg_volume
                    })
            
            # Short signals
            if trade_type in ['short', 'both']:
                short_conditions = [
                    current_rsi > self.params['rsi_overbought'],
                    sma_short.iloc[i] < sma_long.iloc[i],
                    current_price < bb_upper.iloc[i],
                    current_volume > avg_volume * 1.2,
                    macd.iloc[i] < macd_signal.iloc[i]
                ]
                
                # Resistance level check
                near_resistance = any(abs(current_price - level) / current_price < 0.02 for level in resistance_levels[-5:])
                
                if sum(short_conditions) >= 3 or near_resistance:
                    entry_price = current_price
                    stop_loss = entry_price * (1 + self.params['stop_loss_pct'] / 100)
                    target = entry_price * (1 - self.params['target_pct'] / 100)
                    
                    probability = min(0.95, 0.5 + (sum(short_conditions) * 0.1))
                    
                    logic = self.generate_entry_logic(short_conditions, near_resistance, 'short', current_rsi, current_volume, avg_volume)
                    
                    signals.append({
                        'date': self.data.iloc[i]['date'],
                        'type': 'short',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'target': target,
                        'probability': probability,
                        'logic': logic,
                        'rsi': current_rsi,
                        'volume_ratio': current_volume / avg_volume
                    })
        
        return signals
    
    def generate_entry_logic(self, conditions, near_key_level, trade_type, rsi, volume, avg_volume):
        logic_parts = []
        
        if trade_type == 'long':
            if conditions[0]:  # RSI oversold
                logic_parts.append(f"RSI oversold at {rsi:.1f}")
            if conditions[1]:  # SMA bullish
                logic_parts.append("Short SMA above Long SMA (bullish trend)")
            if conditions[2]:  # Near lower BB
                logic_parts.append("Price above lower Bollinger Band")
            if conditions[3]:  # High volume
                logic_parts.append(f"Volume spike ({volume/avg_volume:.1f}x average)")
            if conditions[4]:  # MACD bullish
                logic_parts.append("MACD bullish crossover")
            if near_key_level:
                logic_parts.append("Price near key support level")
        else:
            if conditions[0]:  # RSI overbought
                logic_parts.append(f"RSI overbought at {rsi:.1f}")
            if conditions[1]:  # SMA bearish
                logic_parts.append("Short SMA below Long SMA (bearish trend)")
            if conditions[2]:  # Near upper BB
                logic_parts.append("Price below upper Bollinger Band")
            if conditions[3]:  # High volume
                logic_parts.append(f"Volume spike ({volume/avg_volume:.1f}x average)")
            if conditions[4]:  # MACD bearish
                logic_parts.append("MACD bearish crossover")
            if near_key_level:
                logic_parts.append("Price near key resistance level")
        
        return " | ".join(logic_parts) if logic_parts else "Multiple confluences detected"

class StrategyOptimizer:
    def __init__(self, data, search_type='random'):
        self.data = data
        self.search_type = search_type
    
    def optimize(self, trade_type='both', n_iter=100, target_accuracy=0.85):
        param_grid = {
            'rsi_period': [10, 12, 14, 16, 18, 20],
            'rsi_oversold': [25, 30, 35, 40],
            'rsi_overbought': [60, 65, 70, 75],
            'sma_short': [5, 8, 10, 12, 15],
            'sma_long': [15, 20, 25, 30],
            'bb_period': [15, 20, 25, 30],
            'bb_std': [1.5, 2.0, 2.5, 3.0],
            'stop_loss_pct': [1.0, 1.5, 2.0, 2.5, 3.0],
            'target_pct': [2.0, 3.0, 4.0, 5.0, 6.0]
        }
        
        best_params = None
        best_score = 0
        best_results = None
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        iterations = n_iter if self.search_type == 'random' else min(1000, n_iter)
        tested_combinations = set()
        
        for i in range(iterations):
            # Generate parameters
            if self.search_type == 'random':
                # Avoid duplicate combinations
                attempts = 0
                while attempts < 10:
                    params = {}
                    for key, values in param_grid.items():
                        params[key] = np.random.choice(values)
                    
                    param_key = tuple(sorted(params.items()))
                    if param_key not in tested_combinations:
                        tested_combinations.add(param_key)
                        break
                    attempts += 1
                
                if attempts >= 10:  # If can't find unique combination, use random
                    params = {}
                    for key, values in param_grid.items():
                        params[key] = np.random.choice(values)
            else:
                # Grid search implementation would go here
                params = self.get_grid_params(param_grid, i)
            
            # Ensure logical parameter relationships
            if params['sma_short'] >= params['sma_long']:
                params['sma_long'] = params['sma_short'] + 5
            
            if params['target_pct'] <= params['stop_loss_pct']:
                params['target_pct'] = params['stop_loss_pct'] * 2
            
            if params['rsi_oversold'] >= params['rsi_overbought']:
                params['rsi_overbought'] = params['rsi_oversold'] + 20
            
            # Test strategy
            strategy = SwingTradingStrategy(self.data, params)
            signals = strategy.generate_signals(trade_type)
            
            if len(signals) > 0:
                backtest_results = self.backtest_strategy(signals, self.data)
                
                # Calculate score based on accuracy, returns, and number of trades
                if backtest_results['total_trades'] >= 5:
                    accuracy_score = min(backtest_results['accuracy'] / 100, 1.0)
                    return_score = max(0, min(backtest_results['total_return'] / 50, 1.0))  # Normalize to 50% max
                    trade_count_score = min(backtest_results['total_trades'] / 30, 1.0)  # Normalize to 30 trades
                    
                    # Weighted score prioritizing accuracy as requested
                    score = (accuracy_score * 0.6 + return_score * 0.25 + trade_count_score * 0.15)
                    
                    # Strong accuracy filter - only accept if meets target
                    if backtest_results['accuracy'] >= target_accuracy and score > best_score:
                        best_score = score
                        best_params = params
                        best_results = backtest_results
            
            # Update progress
            progress = (i + 1) / iterations
            progress_bar.progress(progress)
            status_text.text(f"Optimizing... {i+1}/{iterations} iterations completed | Best Accuracy: {best_results['accuracy']:.1f}% | Target: {target_accuracy:.1f}%" if best_results else f"Optimizing... {i+1}/{iterations} iterations completed | Searching for {target_accuracy:.1f}%+ accuracy")
        
        progress_bar.empty()
        status_text.empty()
        
        return best_params, best_results, best_score
    
    def get_grid_params(self, param_grid, index):
        # Simple grid search implementation
        params = {}
        keys = list(param_grid.keys())
        
        for key in keys:
            params[key] = param_grid[key][0]  # Simplified for demo
        
        return params
    
    def backtest_strategy(self, signals, data):
        if not signals:
            return {'total_trades': 0, 'accuracy': 0, 'total_return': 0}
        
        trades = []
        
        for signal in signals:
            entry_date = signal['date']
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            target = signal['target']
            trade_type = signal['type']
            
            # Find entry index
            entry_idx = data[data['date'] >= entry_date].index
            if len(entry_idx) == 0:
                continue
            
            entry_idx = entry_idx[0]
            
            # Look for exit within next 20 days
            exit_price = None
            exit_date = None
            exit_reason = None
            points_gained = 0
            points_lost = 0
            
            for j in range(entry_idx + 1, min(entry_idx + 21, len(data))):
                current_high = data.iloc[j]['high']
                current_low = data.iloc[j]['low']
                current_close = data.iloc[j]['close']
                current_date = data.iloc[j]['date']
                
                if trade_type == 'long':
                    if current_low <= stop_loss:
                        exit_price = stop_loss
                        exit_date = current_date
                        exit_reason = 'Stop Loss'
                        points_lost = entry_price - exit_price
                        break
                    elif current_high >= target:
                        exit_price = target
                        exit_date = current_date
                        exit_reason = 'Target'
                        points_gained = exit_price - entry_price
                        break
                else:  # short
                    if current_high >= stop_loss:
                        exit_price = stop_loss
                        exit_date = current_date
                        exit_reason = 'Stop Loss'
                        points_lost = exit_price - entry_price
                        break
                    elif current_low <= target:
                        exit_price = target
                        exit_date = current_date
                        exit_reason = 'Target'
                        points_gained = entry_price - exit_price
                        break
            
            # If no exit found, use last available price
            if exit_price is None:
                exit_idx = min(entry_idx + 20, len(data) - 1)
                exit_price = data.iloc[exit_idx]['close']
                exit_date = data.iloc[exit_idx]['date']
                exit_reason = 'Time Exit'
                
                if trade_type == 'long':
                    if exit_price > entry_price:
                        points_gained = exit_price - entry_price
                    else:
                        points_lost = entry_price - exit_price
                else:
                    if exit_price < entry_price:
                        points_gained = entry_price - exit_price
                    else:
                        points_lost = exit_price - entry_price
            
            # Calculate PnL
            if trade_type == 'long':
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'type': trade_type,
                'pnl_pct': pnl_pct,
                'points_gained': points_gained,
                'points_lost': points_lost,
                'net_points': points_gained - points_lost,
                'exit_reason': exit_reason,
                'duration_days': (exit_date - entry_date).days,
                'logic': signal['logic']
            })
        
        if not trades:
            return {'total_trades': 0, 'accuracy': 0, 'total_return': 0}
        
        # Calculate metrics
        total_trades = len(trades)
        positive_trades = sum(1 for t in trades if t['pnl_pct'] > 0)
        accuracy = positive_trades / total_trades * 100
        total_return = sum(t['pnl_pct'] for t in trades)
        avg_return_per_trade = total_return / total_trades
        total_points_gained = sum(t['points_gained'] for t in trades)
        total_points_lost = sum(t['points_lost'] for t in trades)
        net_points = total_points_gained - total_points_lost
        
        # Calculate Buy and Hold return for comparison
        start_price = data.iloc[0]['close']
        end_price = data.iloc[-1]['close']
        buy_hold_return = (end_price - start_price) / start_price * 100
        
        return {
            'trades': trades,
            'total_trades': total_trades,
            'positive_trades': positive_trades,
            'negative_trades': total_trades - positive_trades,
            'accuracy': accuracy,
            'total_return': total_return,
            'avg_return_per_trade': avg_return_per_trade,
            'avg_duration': np.mean([t['duration_days'] for t in trades]),
            'total_points_gained': total_points_gained,
            'total_points_lost': total_points_lost,
            'net_points': net_points,
            'buy_hold_return': buy_hold_return,
            'strategy_vs_buy_hold': total_return - buy_hold_return
        }

def load_data_from_file(uploaded_file):
    """Load and process uploaded data file"""
    try:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def map_columns(df):
    """Map columns to standard format"""
    column_mapping = {}
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Date column mapping
        if any(date_term in col_lower for date_term in ['date', 'time', 'timestamp']):
            column_mapping['date'] = col
        
        # OHLCV mapping
        elif any(open_term in col_lower for open_term in ['open']):
            column_mapping['open'] = col
        elif any(high_term in col_lower for high_term in ['high']):
            column_mapping['high'] = col
        elif any(low_term in col_lower for low_term in ['low']):
            column_mapping['low'] = col
        elif any(close_term in col_lower for close_term in ['close', 'closing', 'close_price', 'closeprice']):
            column_mapping['close'] = col
        elif any(vol_term in col_lower for vol_term in ['volume', 'vol', 'shares', 'traded']):
            column_mapping['volume'] = col
    
    return column_mapping

def standardize_data(df, column_mapping):
    """Standardize column names and data types"""
    # Create new dataframe with standard columns
    standard_df = pd.DataFrame()
    
    required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    for std_col in required_columns:
        if std_col in column_mapping:
            standard_df[std_col] = df[column_mapping[std_col]]
        else:
            if std_col == 'volume' and std_col not in column_mapping:
                # Create dummy volume if not present
                standard_df[std_col] = 100000
            else:
                st.error(f"Required column '{std_col}' not found in the data")
                return None
    
    # Convert date column
    try:
        standard_df['date'] = pd.to_datetime(standard_df['date'])
        
        # Convert to IST timezone
        ist = pytz.timezone('Asia/Kolkata')
        if standard_df['date'].dt.tz is None:
            # Naive datetime - assume UTC and convert to IST
            standard_df['date'] = standard_df['date'].dt.tz_localize('UTC').dt.tz_convert(ist)
        else:
            # Already timezone aware - convert to IST
            standard_df['date'] = standard_df['date'].dt.tz_convert(ist)
        
        # Remove timezone info for easier handling
        standard_df['date'] = standard_df['date'].dt.tz_localize(None)
        
    except Exception as e:
        st.error(f"Error converting date column: {str(e)}")
        return None
    
    # Convert numeric columns
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        standard_df[col] = pd.to_numeric(standard_df[col], errors='coerce')
    
    # Remove rows with NaN values
    standard_df = standard_df.dropna().reset_index(drop=True)
    
    # Sort by date in ascending order
    standard_df = standard_df.sort_values('date').reset_index(drop=True)
    
    return standard_df

def fetch_yfinance_data(symbol, period, interval):
    """Fetch data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Map intervals
        interval_mapping = {
            '1min': '1m',
            '5min': '5m',
            '15min': '15m',
            '1hour': '1h',
            '1day': '1d'
        }
        
        yf_interval = interval_mapping.get(interval, '1d')
        
        # Fetch data
        data = ticker.history(period=period, interval=yf_interval)
        
        if data.empty:
            st.error("No data found for the given symbol")
            return None
        
        # Reset index to get date as column
        data = data.reset_index()
        
        # Rename columns to match our standard
        data = data.rename(columns={
            'Date': 'date',
            'Datetime': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Convert to IST
        ist = pytz.timezone('Asia/Kolkata')
        if data['date'].dt.tz is None:
            data['date'] = data['date'].dt.tz_localize('UTC').dt.tz_convert(ist)
        else:
            data['date'] = data['date'].dt.tz_convert(ist)
        
        data['date'] = data['date'].dt.tz_localize(None)
        
        # Select only required columns
        data = data[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def create_candlestick_chart(data, signals=None, title="Stock Price Chart"):
    """Create an interactive candlestick chart"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price', 'Volume', 'RSI'),
        row_heights=[0.7, 0.2, 0.1]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data['date'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add moving averages
    ta = TechnicalAnalysis(data)
    sma20 = ta.calculate_sma(20)
    sma50 = ta.calculate_sma(50)
    
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=sma20,
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=sma50,
            mode='lines',
            name='SMA 50',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=data['date'],
            y=data['volume'],
            name='Volume',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    # RSI
    rsi = ta.calculate_rsi()
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=rsi,
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=1)
        ),
        row=3, col=1
    )
    
    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Add signals if provided
    if signals:
        for signal in signals:
            color = 'green' if signal['type'] == 'long' else 'red'
            symbol = 'triangle-up' if signal['type'] == 'long' else 'triangle-down'
            
            fig.add_trace(
                go.Scatter(
                    x=[signal['date']],
                    y=[signal['entry_price']],
                    mode='markers',
                    name=f"{signal['type'].title()} Signal",
                    marker=dict(
                        color=color,
                        size=10,
                        symbol=symbol
                    ),
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_returns_heatmap(data):
    """Create a returns heatmap by year and month"""
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['month_name'] = data['date'].dt.strftime('%b')
    
    # Calculate monthly returns
    monthly_data = data.groupby(['year', 'month', 'month_name']).agg({
        'close': ['first', 'last']
    }).reset_index()
    
    monthly_data.columns = ['year', 'month', 'month_name', 'open_price', 'close_price']
    monthly_data['return'] = (monthly_data['close_price'] - monthly_data['open_price']) / monthly_data['open_price'] * 100
    
    # Create pivot table
    heatmap_data = monthly_data.pivot_table(values='return', index='year', columns='month_name', fill_value=0)
    
    # Reorder columns by month
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Only include columns that exist
    existing_months = [month for month in month_order if month in heatmap_data.columns]
    heatmap_data = heatmap_data[existing_months]
    
    fig = px.imshow(
        heatmap_data,
        text_auto=True,
        aspect="auto",
        title="Monthly Returns Heatmap (%)",
        labels=dict(x="Month", y="Year", color="Return %")
    )
    
    return fig

def display_market_summary(data):
    """Generate and display market summary"""
    ta = TechnicalAnalysis(data)
    
    # Calculate key metrics
    current_price = data['close'].iloc[-1]
    price_change = ((current_price - data['close'].iloc[-20]) / data['close'].iloc[-20]) * 100
    trend = ta.identify_trend()
    volatility = data['close'].pct_change().std() * 100
    
    # Generate summary text
    summary = f"""
    **Market Analysis Summary:**
    
    The stock is currently trading at {current_price:.2f}, showing a {price_change:.2f}% change over the past 20 sessions. 
    The overall trend appears to be {trend.lower()}, with a volatility of {volatility:.2f}%. 
    
    Key technical indicators suggest {'bullish' if price_change > 0 else 'bearish'} momentum in the short term. 
    Volume analysis indicates {'strong' if data['volume'].iloc[-5:].mean() > data['volume'].mean() else 'weak'} 
    institutional interest. The current market structure shows 
    {'potential buying opportunities' if trend == 'Downtrend' else 'potential resistance levels'} 
    that swing traders should monitor carefully.
    
    Risk management remains crucial given the current volatility levels, and traders should consider 
    {'waiting for better entry points' if volatility > 3 else 'capitalizing on current momentum'} 
    while maintaining strict stop-loss levels.
    """
    
    return summary

def main():
    st.markdown('<h1 class="main-header">🚀 Advanced Swing Trading Platform</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    if 'analysis_complete' not in st.session_state:
        st.session_state['analysis_complete'] = False
    
    # Sidebar
    st.sidebar.title("📊 Trading Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Select Data Source",
        ["Upload File", "Yahoo Finance"]
    )
    
    data = st.session_state.get('data', None)
    
    if data_source == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your stock data file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV or Excel file with OHLCV data"
        )
        
        if uploaded_file:
            with st.spinner("Loading and processing data..."):
                df = load_data_from_file(uploaded_file)
                
                if df is not None:
                    # Display original column info
                    st.sidebar.write("**Original Columns:**")
                    st.sidebar.write(list(df.columns))
                    
                    # Map columns
                    column_mapping = map_columns(df)
                    st.sidebar.write("**Column Mapping:**")
                    st.sidebar.json(column_mapping)
                    
                    # Standardize data
                    data = standardize_data(df, column_mapping)
                    
                    if data is not None:
                        st.sidebar.success("✅ Data loaded successfully!")
                        st.session_state['data'] = data
    
    else:  # Yahoo Finance
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            symbol = st.text_input(
                "Stock Symbol",
                value="INFY.NS",
                help="Enter stock symbol (e.g., INFY.NS, TCS.NS, ^NSEI)"
            )
        
        with col2:
            period = st.selectbox(
                "Period",
                ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "15y", "max"],
                index=5
            )
        
        interval = st.sidebar.selectbox(
            "Interval",
            ["1min", "5min", "15min", "1hour", "1day"],
            index=4
        )
        
        if st.sidebar.button("Fetch Data"):
            with st.spinner("Fetching data from Yahoo Finance..."):
                data = fetch_yfinance_data(symbol, period, interval)
                
                if data is not None:
                    st.sidebar.success("✅ Data fetched successfully!")
                    # Store data in session state to prevent disappearing
                    st.session_state['data'] = data
                    st.session_state['data_source'] = 'yfinance'
    
    # Update data reference
    data = st.session_state.get('data', None)_source'] = 'yfinance'_source'] = 'yfinance'
                    st.experimental_rerun()
    
    # Check session state for data
    if 'data' in st.session_state and data is None:
        data = st.session_state['data']
    
    # Always show sidebar options regardless of data state
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Trading Parameters")
    
    if data is not None:
        # Display basic data info
        st.subheader("📈 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Date Range", f"{data['date'].min().strftime('%Y-%m-%d')}")
        with col3:
            st.metric("To", f"{data['date'].max().strftime('%Y-%m-%d')}")
        with col4:
            st.metric("Current Price", f"{data['close'].iloc[-1]:.2f}")
        
        # Display top and bottom 5 rows
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**First 5 Rows:**")
            st.dataframe(data.head())
        
        with col2:
            st.write("**Last 5 Rows:**")
            st.dataframe(data.tail())
        
        # Price range
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Min Price", f"{data['close'].min():.2f}")
        with col2:
            st.metric("Max Price", f"{data['close'].max():.2f}")
        with col3:
            st.metric("Avg Volume", f"{data['volume'].mean():,.0f}")
        with col4:
            st.metric("Max Volume", f"{data['volume'].max():,.0f}")
        
        # Raw data plot
        st.subheader("📊 Raw Data Visualization")
        raw_chart = create_candlestick_chart(data, title="Raw Stock Data")
        st.plotly_chart(raw_chart, use_container_width=True)
        
    if data is not None:
        # Display basic data info
        st.subheader("📈 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Date Range", f"{data['date'].min().strftime('%Y-%m-%d')}")
        with col3:
            st.metric("To", f"{data['date'].max().strftime('%Y-%m-%d')}")
        with col4:
            st.metric("Current Price", f"{data['close'].iloc[-1]:.2f}")
        
        # Display top and bottom 5 rows
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**First 5 Rows:**")
            st.dataframe(data.head())
        
        with col2:
            st.write("**Last 5 Rows:**")
            st.dataframe(data.tail())
        
        # Price range
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Min Price", f"{data['close'].min():.2f}")
        with col2:
            st.metric("Max Price", f"{data['close'].max():.2f}")
        with col3:
            st.metric("Avg Volume", f"{data['volume'].mean():,.0f}")
        with col4:
            st.metric("Max Volume", f"{data['volume'].max():,.0f}")
        
        # Raw data plot
        st.subheader("📊 Raw Data Visualization")
        raw_chart = create_candlestick_chart(data, title="Raw Stock Data")
        st.plotly_chart(raw_chart, use_container_width=True)
        
        # Trading Configuration (always show after data is loaded)
        # End date selection for backtesting
        max_date = data['date'].max().date()
        min_date = data['date'].min().date()
        
        end_date = st.sidebar.date_input(
            "End Date for Analysis",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
        
        # Trade type selection
        trade_type = st.sidebar.selectbox(
            "Trade Direction",
            ["both", "long", "short"],
            index=0
        )
        
        # Optimization parameters
        optimization_type = st.sidebar.selectbox(
            "Optimization Method",
            ["random", "grid"],
            index=0
        )
        
        target_accuracy = st.sidebar.slider(
            "Target Accuracy (%)",
            min_value=60,
            max_value=95,
            value=85,
            step=5
        )
        
        min_trades = st.sidebar.slider(
            "Minimum Number of Trades",
            min_value=5,
            max_value=50,
            value=10,
            step=5
        )
        
        # Number of iterations option
        n_iterations = st.sidebar.slider(
            "Optimization Iterations",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="More iterations = better optimization but slower"
        )
        
        # Filter data up to end date
        analysis_data = data[data['date'].dt.date <= end_date].copy()
        
        # EDA Section
        st.subheader("🔍 Exploratory Data Analysis")
        
        # Returns heatmap
        try:
            heatmap_fig = create_returns_heatmap(analysis_data)
            st.plotly_chart(heatmap_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate heatmap: {str(e)}")
        
        # Market summary
        st.subheader("📋 Market Summary")
        market_summary = display_market_summary(analysis_data)
        st.markdown(market_summary)
        
        # Run Analysis Button
        if st.sidebar.button("🚀 Run Analysis", type="primary"):
            st.session_state['analysis_complete'] = False  # Reset analysis state
            
            st.subheader("🔧 Strategy Optimization")
            
            with st.spinner("Optimizing strategy parameters..."):
                optimizer = StrategyOptimizer(analysis_data, optimization_type)
                best_params, best_results, best_score = optimizer.optimize(
                    trade_type=trade_type,
                    n_iter=n_iterations,
                    target_accuracy=target_accuracy
                )
            
            if best_params is None:
                st.error(f"❌ Could not find optimal parameters with {target_accuracy}% accuracy target.")
                st.info("💡 Try lowering the target accuracy or increasing the number of iterations.")
                
                # Show alternative message
                st.warning("🔄 Consider adjusting parameters:")
                st.write("• Lower target accuracy to 70-80%")
                st.write("• Increase iterations to 200-500") 
                st.write("• Try different trade direction")
                st.write("• Check if data has sufficient volatility")
                
                # Still show some basic analysis
                st.subheader("📊 Basic Analysis (Without Optimization)")
                
                # Run with default parameters
                default_params = {
                    'rsi_period': 14,
                    'rsi_oversold': 30,
                    'rsi_overbought': 70,
                    'sma_short': 10,
                    'sma_long': 20,
                    'bb_period': 20,
                    'bb_std': 2,
                    'stop_loss_pct': 2.0,
                    'target_pct': 4.0
                }
                
                strategy = SwingTradingStrategy(analysis_data, default_params)
                signals = strategy.generate_signals(trade_type)
                
                if signals:
                    backtest_results = optimizer.backtest_strategy(signals, analysis_data)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Default Strategy Results:**")
                        st.metric("Accuracy", f"{backtest_results['accuracy']:.1f}%")
                        st.metric("Total Trades", backtest_results['total_trades'])
                    
                    with col2:
                        st.metric("Total Return", f"{backtest_results['total_return']:.2f}%")
                        if 'buy_hold_return' in backtest_results:
                            st.metric("Buy & Hold", f"{backtest_results['buy_hold_return']:.2f}%")
                
                return  # Exit early if optimization failed
            
            # Store results in session state
            st.session_state['best_params'] = best_params
            st.session_state['best_results'] = best_results
            st.session_state['analysis_data'] = analysis_data
            st.session_state['analysis_complete'] = True
        
        # Show results if analysis is complete
        if st.session_state.get('analysis_complete', False):
            best_params = st.session_state['best_params']
            best_results = st.session_state['best_results']
            analysis_data = st.session_state['analysis_data']
            
            # Display best strategy
            st.success("✅ Optimization Complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Best Strategy Parameters:**")
                for key, value in best_params.items():
                    st.write(f"• {key}: {value}")
            
            with col2:
                st.markdown("**Strategy Performance:**")
                st.metric("Accuracy", f"{best_results['accuracy']:.1f}%")
                st.metric("Total Return", f"{best_results['total_return']:.2f}%")
                st.metric("Total Trades", best_results['total_trades'])
                st.metric("Win Rate", f"{(best_results['positive_trades']/best_results['total_trades']*100):.1f}%")
                
                # Add buy and hold comparison
                st.markdown("**vs Buy & Hold:**")
                if 'buy_hold_return' in best_results:
                    st.metric("Buy & Hold Return", f"{best_results['buy_hold_return']:.2f}%")
                    st.metric("Strategy Outperformance", f"{best_results['strategy_vs_buy_hold']:.2f}%")
                
                # Add points summary
                st.markdown("**Points Analysis:**")
                if 'total_points_gained' in best_results:
                    st.metric("Total Points Gained", f"{best_results['total_points_gained']:.1f}")
                    st.metric("Total Points Lost", f"{best_results['total_points_lost']:.1f}")
                    st.metric("Net Points", f"{best_results['net_points']:.1f}")
            
            # Detailed backtest results
            st.subheader("📊 Backtest Results")
            
            if 'trades' in best_results:
                trades_df = pd.DataFrame(best_results['trades'])
                
                # Display trade summary
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
                    st.metric("Winning Trades", best_results['positive_trades'])
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card danger-metric">', unsafe_allow_html=True)
                    st.metric("Losing Trades", best_results['negative_trades'])
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Avg Return/Trade", f"{best_results['avg_return_per_trade']:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Avg Hold Duration", f"{best_results['avg_duration']:.1f} days")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Detailed trades table
                st.write("**Detailed Trade History:**")
                display_trades = trades_df.copy()
                display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
                display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
                display_trades['pnl_pct'] = display_trades['pnl_pct'].round(2)
                display_trades['entry_price'] = display_trades['entry_price'].round(2)
                display_trades['exit_price'] = display_trades['exit_price'].round(2)
                
                # Add points columns if they exist
                if 'points_gained' in display_trades.columns:
                    display_trades['points_gained'] = display_trades['points_gained'].round(2)
                    display_trades['points_lost'] = display_trades['points_lost'].round(2)
                    display_trades['net_points'] = display_trades['net_points'].round(2)
                
                # Reorder columns for better display
                column_order = ['entry_date', 'exit_date', 'type', 'entry_price', 'exit_price', 
                              'pnl_pct', 'exit_reason', 'duration_days']
                
                if 'points_gained' in display_trades.columns:
                    column_order.extend(['points_gained', 'points_lost', 'net_points'])
                
                column_order.append('logic')
                
                # Only include existing columns
                existing_columns = [col for col in column_order if col in display_trades.columns]
                display_trades = display_trades[existing_columns]
                
                st.dataframe(display_trades, use_container_width=True)
            
            # Generate signals for visualization
            strategy = SwingTradingStrategy(analysis_data, best_params)
            all_signals = strategy.generate_signals(trade_type)
            
            # Create chart with signals
            st.subheader("📈 Strategy Signals Visualization")
            signals_chart = create_candlestick_chart(
                analysis_data, 
                all_signals[:20],  # Show first 20 signals for clarity
                "Strategy Signals on Price Chart"
            )
            st.plotly_chart(signals_chart, use_container_width=True)
            
            # Live Recommendation
            st.subheader("🎯 Live Trading Recommendation")
            
            # Get current recommendation using full dataset
            current_data = data.copy()
            live_strategy = SwingTradingStrategy(current_data, best_params)
            live_signals = live_strategy.generate_signals(trade_type)
            
            if live_signals:
                latest_signal = live_signals[-1]  # Get most recent signal
                
                # Check if signal is recent (within last 5 candles)
                last_date = current_data['date'].max()
                signal_date = latest_signal['date']
                
                if (last_date - signal_date).days <= 5:
                    signal_color = "🟢" if latest_signal['type'] == 'long' else "🔴"
                    
                    st.markdown(f"""
                    ### {signal_color} **{latest_signal['type'].upper()} SIGNAL**
                    
                    **📅 Date:** {latest_signal['date'].strftime('%Y-%m-%d %H:%M:%S')}
                    
                    **💰 Entry Price:** ₹{latest_signal['entry_price']:.2f}
                    
                    **🎯 Target:** ₹{latest_signal['target']:.2f}
                    
                    **🛑 Stop Loss:** ₹{latest_signal['stop_loss']:.2f}
                    
                    **📊 Probability:** {latest_signal['probability']*100:.1f}%
                    
                    **🧠 Logic:** {latest_signal['logic']}
                    
                    **📈 RSI:** {latest_signal['rsi']:.1f}
                    
                    **📊 Volume Ratio:** {latest_signal['volume_ratio']:.1f}x
                    """)
                    
                    # Risk-Reward Analysis
                    if latest_signal['type'] == 'long':
                        risk = latest_signal['entry_price'] - latest_signal['stop_loss']
                        reward = latest_signal['target'] - latest_signal['entry_price']
                    else:
                        risk = latest_signal['stop_loss'] - latest_signal['entry_price']
                        reward = latest_signal['entry_price'] - latest_signal['target']
                    
                    risk_reward = reward / risk if risk > 0 else 0
                    
                    st.markdown(f"""
                    **⚖️ Risk-Reward Ratio:** {risk_reward:.2f}:1
                    
                    **💡 Risk:** ₹{risk:.2f} per share
                    
                    **🎁 Potential Reward:** ₹{reward:.2f} per share
                    """)
                
                else:
                    st.info("No recent signals. Current market conditions do not meet entry criteria.")
            else:
                st.info("No signals generated. Market conditions do not currently favor entry.")
            
            # Continue with rest of analysis...
            # Pattern Recognition Summary
            st.subheader("📊 Technical Analysis Summary")
            
            ta = TechnicalAnalysis(current_data)
            patterns = ta.identify_chart_patterns()
            pivots = ta.calculate_pivot_points()
            
            if patterns:
                st.write("**Identified Chart Patterns:**")
                for pattern in patterns[-5:]:  # Show last 5 patterns
                    st.write(f"• {pattern['pattern']} ({pattern['type']}) - {pattern['date'].strftime('%Y-%m-%d')} at ₹{pattern['price']:.2f}")
            
            # Market Structure Analysis
            trend = ta.identify_trend()
            support_levels, resistance_levels, _, _ = ta.find_support_resistance()
            
            st.markdown(f"""
            **📊 Market Structure:**
            - **Current Trend:** {trend}
            - **Key Support Levels:** {', '.join([f'₹{level:.2f}' for level in support_levels[-3:]])}
            - **Key Resistance Levels:** {', '.join([f'₹{level:.2f}' for level in resistance_levels[-3:]])}
            - **Recent Pivot Points:** {len(pivots[-10:])} identified in recent data
            """)
            
            # Final Summary
            st.subheader("📝 Trading Summary & Recommendations")
            
            backtest_summary = f"""
            **Backtest Analysis Complete:**
            
            Our advanced swing trading algorithm analyzed {len(analysis_data)} data points and identified 
            {best_results['total_trades']} trading opportunities with {best_results['accuracy']:.1f}% accuracy.
            The strategy generated {best_results['total_return']:.2f}% total returns with an average 
            holding period of {best_results['avg_duration']:.1f} days per trade.
            
            **Strategy vs Buy & Hold:**
            {'✅ Strategy OUTPERFORMED buy & hold by ' + str(abs(best_results.get('strategy_vs_buy_hold', 0))) + '%' 
             if best_results.get('strategy_vs_buy_hold', 0) > 0 
             else '❌ Strategy UNDERPERFORMED buy & hold by ' + str(abs(best_results.get('strategy_vs_buy_hold', 0))) + '%'}
            
            **Strategy Details:**
            - RSI levels: {best_params['rsi_oversold']}-{best_params['rsi_overbought']}
            - Moving averages: {best_params['sma_short']}/{best_params['sma_long']}
            - Risk management: {best_params['stop_loss_pct']:.1f}% stop loss, {best_params['target_pct']:.1f}% target
            
            **Live Trading Recommendations:**
            Based on current market conditions and the optimized strategy parameters, 
            {'continue monitoring for entry opportunities' if not live_signals or (last_date - live_signals[-1]['date']).days > 5 
            else f'consider the {live_signals[-1]["type"]} position with strict risk management'}.
            
            Always maintain position sizing according to your risk tolerance and never risk more than 
            2% of your capital per trade. Market conditions can change rapidly, so continuous monitoring 
            is essential for successful swing trading.
            """
            
            st.markdown(backtest_summary)
            
            # Buyer/Seller Psychology
            st.subheader("🧠 Market Psychology Analysis")
            
            current_rsi = ta.calculate_rsi().iloc[-1]
            recent_volume = current_data['volume'].iloc[-5:].mean()
            avg_volume = current_data['volume'].mean()
            price_change_5d = ((current_data['close'].iloc[-1] - current_data['close'].iloc[-5]) / current_data['close'].iloc[-5]) * 100
            
            psychology_summary = f"""
            **Current Market Sentiment:**
            
            RSI at {current_rsi:.1f} suggests {'overbought conditions - sellers may step in' if current_rsi > 70 
            else 'oversold conditions - buyers may emerge' if current_rsi < 30 
            else 'neutral momentum - waiting for directional bias'}.
            
            Recent volume activity ({recent_volume/avg_volume:.1f}x average) indicates 
            {'strong institutional interest' if recent_volume/avg_volume > 1.2 
            else 'moderate participation' if recent_volume/avg_volume > 0.8 
            else 'low participation - lack of conviction'}.
            
            The {price_change_5d:.1f}% price move over the last 5 sessions reflects 
            {'bullish sentiment with buyers in control' if price_change_5d > 2
            else 'bearish sentiment with sellers dominating' if price_change_5d < -2
            else 'indecision between buyers and sellers'}.
            
            **Trading Psychology Insights:**
            {'Fear-based selling may create opportunities for patient buyers' if current_rsi < 30
            else 'Greed-driven buying may create shorting opportunities' if current_rsi > 70
            else 'Market in equilibrium - wait for clear directional move'}.
            """
            
            st.markdown(psychology_summary)
    
    else:
        st.info("👆 Please upload a data file or fetch data from Yahoo Finance to get started.")_signals[:20],  # Show first 20 signals for clarity
                "Strategy Signals on Price Chart"
            )
            st.plotly_chart(signals_chart, use_container_width=True)
            
            # Live Recommendation
            st.subheader("🎯 Live Trading Recommendation")
            
            # Get current recommendation
            current_data = data.copy()  # Use full dataset for live recommendation
            live_strategy = SwingTradingStrategy(current_data, best_params)
            live_signals = live_strategy.generate_signals(trade_type)
            
            if live_signals:
                latest_signal = live_signals[-1]  # Get most recent signal
                
                # Check if signal is recent (within last 5 candles)
                last_date = current_data['date'].max()
                signal_date = latest_signal['date']
                
                if (last_date - signal_date).days <= 5:
                    signal_color = "🟢" if latest_signal['type'] == 'long' else "🔴"
                    
                    st.markdown(f"""
                    ### {signal_color} **{latest_signal['type'].upper()} SIGNAL**
                    
                    **📅 Date:** {latest_signal['date'].strftime('%Y-%m-%d %H:%M:%S')}
                    
                    **💰 Entry Price:** ₹{latest_signal['entry_price']:.2f}
                    
                    **🎯 Target:** ₹{latest_signal['target']:.2f}
                    
                    **🛑 Stop Loss:** ₹{latest_signal['stop_loss']:.2f}
                    
                    **📊 Probability:** {latest_signal['probability']*100:.1f}%
                    
                    **🧠 Logic:** {latest_signal['logic']}
                    
                    **📈 RSI:** {latest_signal['rsi']:.1f}
                    
                    **📊 Volume Ratio:** {latest_signal['volume_ratio']:.1f}x
                    """)
                    
                    # Risk-Reward Analysis
                    if latest_signal['type'] == 'long':
                        risk = latest_signal['entry_price'] - latest_signal['stop_loss']
                        reward = latest_signal['target'] - latest_signal['entry_price']
                    else:
                        risk = latest_signal['stop_loss'] - latest_signal['entry_price']
                        reward = latest_signal['entry_price'] - latest_signal['target']
                    
                    risk_reward = reward / risk if risk > 0 else 0
                    
                    st.markdown(f"""
                    **⚖️ Risk-Reward Ratio:** {risk_reward:.2f}:1
                    
                    **💡 Risk:** ₹{risk:.2f} per share
                    
                    **🎁 Potential Reward:** ₹{reward:.2f} per share
                    """)
                
                else:
                    st.info("No recent signals. Current market conditions do not meet entry criteria.")
            else:
                st.info("No signals generated. Market conditions do not currently favor entry.")
            
            # Pattern Recognition Summary
            st.subheader("📊 Technical Analysis Summary")
            
            ta = TechnicalAnalysis(current_data)
            patterns = ta.identify_chart_patterns()
            pivots = ta.calculate_pivot_points()
            
            if patterns:
                st.write("**Identified Chart Patterns:**")
                for pattern in patterns[-5:]:  # Show last 5 patterns
                    st.write(f"• {pattern['pattern']} ({pattern['type']}) - {pattern['date'].strftime('%Y-%m-%d')} at ₹{pattern['price']:.2f}")
            
            # Market Structure Analysis
            trend = ta.identify_trend()
            support_levels, resistance_levels, _, _ = ta.find_support_resistance()
            
            st.markdown(f"""
            **📊 Market Structure:**
            - **Current Trend:** {trend}
            - **Key Support Levels:** {', '.join([f'₹{level:.2f}' for level in support_levels[-3:]])}
            - **Key Resistance Levels:** {', '.join([f'₹{level:.2f}' for level in resistance_levels[-3:]])}
            - **Recent Pivot Points:** {len(pivots[-10:])} identified in recent data
            """)
            
            # Final Summary
            st.subheader("📝 Trading Summary & Recommendations")
            
            backtest_summary = f"""
            **Backtest Analysis Complete:**
            
            Our advanced swing trading algorithm analyzed {len(analysis_data)} data points and identified 
            {best_results['total_trades']} trading opportunities with {best_results['accuracy']:.1f}% accuracy.
            The strategy generated {best_results['total_return']:.2f}% total returns with an average 
            holding period of {best_results['avg_duration']:.1f} days per trade.
            
            **Strategy Details:**
            - RSI levels: {best_params['rsi_oversold']}-{best_params['rsi_overbought']}
            - Moving averages: {best_params['sma_short']}/{best_params['sma_long']}
            - Risk management: {best_params['stop_loss_pct']:.1f}% stop loss, {best_params['target_pct']:.1f}% target
            
            **Live Trading Recommendations:**
            Based on current market conditions and the optimized strategy parameters, 
            {'continue monitoring for entry opportunities' if not live_signals or (last_date - live_signals[-1]['date']).days > 5 
            else f'consider the {live_signals[-1]["type"]} position with strict risk management'}.
            
            Always maintain position sizing according to your risk tolerance and never risk more than 
            2% of your capital per trade. Market conditions can change rapidly, so continuous monitoring 
            is essential for successful swing trading.
            """
            
            st.markdown(backtest_summary)
            
            # Buyer/Seller Psychology
            st.subheader("🧠 Market Psychology Analysis")
            
            current_rsi = ta.calculate_rsi().iloc[-1]
            recent_volume = current_data['volume'].iloc[-5:].mean()
            avg_volume = current_data['volume'].mean()
            price_change_5d = ((current_data['close'].iloc[-1] - current_data['close'].iloc[-5]) / current_data['close'].iloc[-5]) * 100
            
            psychology_summary = f"""
            **Current Market Sentiment:**
            
            RSI at {current_rsi:.1f} suggests {'overbought conditions - sellers may step in' if current_rsi > 70 
            else 'oversold conditions - buyers may emerge' if current_rsi < 30 
            else 'neutral momentum - waiting for directional bias'}.
            
            Recent volume activity ({recent_volume/avg_volume:.1f}x average) indicates 
            {'strong institutional interest' if recent_volume/avg_volume > 1.2 
            else 'moderate participation' if recent_volume/avg_volume > 0.8 
            else 'low participation - lack of conviction'}.
            
            The {price_change_5d:.1f}% price move over the last 5 sessions reflects 
            {'bullish sentiment with buyers in control' if price_change_5d > 2
            else 'bearish sentiment with sellers dominating' if price_change_5d < -2
            else 'indecision between buyers and sellers'}.
            
            **Trading Psychology Insights:**
            {'Fear-based selling may create opportunities for patient buyers' if current_rsi < 30
            else 'Greed-driven buying may create shorting opportunities' if current_rsi > 70
            else 'Market in equilibrium - wait for clear directional move'}.
            """
            
            st.markdown(psychology_summary)

if __name__ == "__main__":
    main()
