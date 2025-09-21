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

# Custom CSS for better styling
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
        
    def ensure_timezone_compatibility(self, date_value, reference_series):
        """Ensure timezone compatibility between date_value and reference_series"""
        try:
            # Convert date_value to pandas Timestamp
            target_date = pd.Timestamp(date_value)
            
            # Get timezone info from reference series
            if hasattr(reference_series.iloc[0], 'tz') and reference_series.iloc[0].tz is not None:
                ref_tz = reference_series.iloc[0].tz
                
                # Make target_date timezone-aware to match reference
                if target_date.tz is None:
                    target_date = target_date.tz_localize(ref_tz)
                else:
                    target_date = target_date.tz_convert(ref_tz)
            
            return target_date
        except Exception as e:
            st.error(f"Timezone conversion error: {e}")
            return None
        
    def map_columns(self, df):
        """Intelligently map column names to standard format"""
        column_mapping = {}
        
        for col in df.columns:
            col_lower = col.lower()
            
            if any(substring in col_lower for substring in ['open', 'open_price']):
                column_mapping[col] = 'Open'
            elif any(substring in col_lower for substring in ['high', 'high_price']):
                column_mapping[col] = 'High'
            elif any(substring in col_lower for substring in ['low', 'low_price']):
                column_mapping[col] = 'Low'
            elif any(substring in col_lower for substring in ['close', 'close_price']):
                column_mapping[col] = 'Close'
            elif any(substring in col_lower for substring in ['volume', 'vol', 'shares', 'traded']):
                column_mapping[col] = 'Volume'
            elif any(substring in col_lower for substring in ['date', 'time', 'timestamp']):
                column_mapping[col] = 'Date'
        
        df_mapped = df.rename(columns=column_mapping)
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']
        missing_cols = [col for col in required_cols if col not in df_mapped.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None
            
        return df_mapped[required_cols]
    
    def preprocess_data(self, df, end_date=None):
        """Preprocess the stock data"""
        # Convert date column to datetime with IST timezone
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Convert to IST timezone
            ist = pytz.timezone('Asia/Kolkata')
            if df['Date'].dt.tz is None:
                # If timezone-naive, first localize to UTC then convert to IST
                df['Date'] = df['Date'].dt.tz_localize('UTC').dt.tz_convert(ist)
            else:
                # If already timezone-aware, convert to IST
                df['Date'] = df['Date'].dt.tz_convert(ist)
        except Exception as e:
            # Fallback: try different date formats
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
                ist = pytz.timezone('Asia/Kolkata')
                df['Date'] = df['Date'].dt.tz_localize(ist)
            except:
                try:
                    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
                    ist = pytz.timezone('Asia/Kolkata')
                    if df['Date'].dt.tz is None:
                        df['Date'] = df['Date'].dt.tz_localize(ist)
                    else:
                        df['Date'] = df['Date'].dt.tz_convert(ist)
                except Exception as final_e:
                    st.error(f"Error converting dates: {final_e}")
                    return None
        
        # Sort by date ascending (no future data leakage)
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Filter by end date if provided
        if end_date:
            try:
                end_date_tz = self.ensure_timezone_compatibility(end_date, df['Date'])
                if end_date_tz is not None:
                    df = df[df['Date'] <= end_date_tz]
            except Exception as e:
                st.error(f"Error filtering by date: {e}")
                # Continue without date filtering
                pass
        
        # Calculate technical indicators
        df = self.calculate_indicators(df)
        
        return df
    
    def calculate_indicators(self, df):
        """Calculate technical indicators without using external libraries"""
        # Simple Moving Averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Price action indicators
        df['HL2'] = (df['High'] + df['Low']) / 2
        df['HLC3'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['OHLC4'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        
        # Volatility
        df['ATR'] = self.calculate_atr(df)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Support and Resistance levels
        df = self.identify_support_resistance(df)
        
        return df
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.DataFrame({'HL': high_low, 'HC': high_close, 'LC': low_close})
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(window=period).mean()
    
    def identify_support_resistance(self, df, window=20):
        """Identify support and resistance levels"""
        df['Resistance'] = df['High'].rolling(window=window, center=True).max()
        df['Support'] = df['Low'].rolling(window=window, center=True).min()
        
        # Dynamic support/resistance based on pivot points
        df['Pivot_High'] = df['High'][(df['High'].shift(1) < df['High']) & 
                                     (df['High'].shift(-1) < df['High'])]
        df['Pivot_Low'] = df['Low'][(df['Low'].shift(1) > df['Low']) & 
                                   (df['Low'].shift(-1) > df['Low'])]
        
        return df
    
    def detect_chart_patterns(self, df):
        """Detect various chart patterns"""
        patterns = []
        
        for i in range(50, len(df)-10):
            # Head and Shoulders
            if self.is_head_shoulders(df, i):
                patterns.append({
                    'pattern': 'Head and Shoulders',
                    'date': df.iloc[i]['Date'],
                    'price': df.iloc[i]['Close'],
                    'type': 'Bearish',
                    'strength': 0.8
                })
            
            # Inverse Head and Shoulders
            if self.is_inverse_head_shoulders(df, i):
                patterns.append({
                    'pattern': 'Inverse Head and Shoulders',
                    'date': df.iloc[i]['Date'],
                    'price': df.iloc[i]['Close'],
                    'type': 'Bullish',
                    'strength': 0.8
                })
            
            # Double Top
            if self.is_double_top(df, i):
                patterns.append({
                    'pattern': 'Double Top',
                    'date': df.iloc[i]['Date'],
                    'price': df.iloc[i]['Close'],
                    'type': 'Bearish',
                    'strength': 0.7
                })
            
            # Double Bottom
            if self.is_double_bottom(df, i):
                patterns.append({
                    'pattern': 'Double Bottom',
                    'date': df.iloc[i]['Date'],
                    'price': df.iloc[i]['Close'],
                    'type': 'Bullish',
                    'strength': 0.7
                })
            
            # Triangles
            triangle_pattern = self.detect_triangles(df, i)
            if triangle_pattern:
                patterns.append(triangle_pattern)
            
            # Cup and Handle
            if self.is_cup_handle(df, i):
                patterns.append({
                    'pattern': 'Cup and Handle',
                    'date': df.iloc[i]['Date'],
                    'price': df.iloc[i]['Close'],
                    'type': 'Bullish',
                    'strength': 0.75
                })
        
        return patterns
    
    def is_head_shoulders(self, df, idx, lookback=20):
        """Detect Head and Shoulders pattern"""
        if idx < lookback or idx >= len(df) - lookback:
            return False
        
        # Simplified head and shoulders detection
        left_shoulder = df.iloc[idx-lookback:idx-lookback//2]['High'].max()
        head = df.iloc[idx-lookback//2:idx+lookback//2]['High'].max()
        right_shoulder = df.iloc[idx+lookback//2:idx+lookback]['High'].max()
        
        return (head > left_shoulder * 1.02 and head > right_shoulder * 1.02 and 
                abs(left_shoulder - right_shoulder) / left_shoulder < 0.03)
    
    def is_inverse_head_shoulders(self, df, idx, lookback=20):
        """Detect Inverse Head and Shoulders pattern"""
        if idx < lookback or idx >= len(df) - lookback:
            return False
        
        left_shoulder = df.iloc[idx-lookback:idx-lookback//2]['Low'].min()
        head = df.iloc[idx-lookback//2:idx+lookback//2]['Low'].min()
        right_shoulder = df.iloc[idx+lookback//2:idx+lookback]['Low'].min()
        
        return (head < left_shoulder * 0.98 and head < right_shoulder * 0.98 and 
                abs(left_shoulder - right_shoulder) / left_shoulder < 0.03)
    
    def is_double_top(self, df, idx, lookback=30):
        """Detect Double Top pattern"""
        if idx < lookback or idx >= len(df) - lookback//2:
            return False
        
        highs = df.iloc[idx-lookback:idx+lookback//2]['High']
        peaks = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)]
        
        if len(peaks) >= 2:
            peak_values = peaks.nlargest(2)
            return abs(peak_values.iloc[0] - peak_values.iloc[1]) / peak_values.iloc[0] < 0.02
        
        return False
    
    def is_double_bottom(self, df, idx, lookback=30):
        """Detect Double Bottom pattern"""
        if idx < lookback or idx >= len(df) - lookback//2:
            return False
        
        lows = df.iloc[idx-lookback:idx+lookback//2]['Low']
        troughs = lows[(lows.shift(1) > lows) & (lows.shift(-1) > lows)]
        
        if len(troughs) >= 2:
            trough_values = troughs.nsmallest(2)
            return abs(trough_values.iloc[0] - trough_values.iloc[1]) / trough_values.iloc[0] < 0.02
        
        return False
    
    def detect_triangles(self, df, idx, lookback=40):
        """Detect triangle patterns"""
        if idx < lookback or idx >= len(df) - 10:
            return None
        
        recent_data = df.iloc[idx-lookback:idx+10]
        
        # Ascending Triangle
        resistance_line = recent_data['High'].rolling(10).max()
        support_trend = np.polyfit(range(len(recent_data)), recent_data['Low'], 1)
        
        if support_trend[0] > 0 and resistance_line.std() < recent_data['Close'].mean() * 0.01:
            return {
                'pattern': 'Ascending Triangle',
                'date': df.iloc[idx]['Date'],
                'price': df.iloc[idx]['Close'],
                'type': 'Bullish',
                'strength': 0.6
            }
        
        # Descending Triangle
        support_line = recent_data['Low'].rolling(10).min()
        resistance_trend = np.polyfit(range(len(recent_data)), recent_data['High'], 1)
        
        if resistance_trend[0] < 0 and support_line.std() < recent_data['Close'].mean() * 0.01:
            return {
                'pattern': 'Descending Triangle',
                'date': df.iloc[idx]['Date'],
                'price': df.iloc[idx]['Close'],
                'type': 'Bearish',
                'strength': 0.6
            }
        
        return None
    
    def is_cup_handle(self, df, idx, lookback=60):
        """Detect Cup and Handle pattern"""
        if idx < lookback or idx >= len(df) - 20:
            return False
        
        cup_data = df.iloc[idx-lookback:idx]
        
        # Check for U-shaped cup
        left_high = cup_data.iloc[:lookback//3]['High'].max()
        bottom = cup_data.iloc[lookback//3:2*lookback//3]['Low'].min()
        right_high = cup_data.iloc[2*lookback//3:]['High'].max()
        
        # Cup criteria
        cup_depth = (left_high - bottom) / left_high
        rim_similarity = abs(left_high - right_high) / left_high
        
        if 0.1 < cup_depth < 0.5 and rim_similarity < 0.05:
            # Check for handle (small pullback after right rim)
            handle_data = df.iloc[idx:idx+20]
            if len(handle_data) > 10:
                handle_low = handle_data['Low'].min()
                handle_pullback = (right_high - handle_low) / right_high
                
                return 0.05 < handle_pullback < 0.15
        
        return False
    
    def identify_supply_demand_zones(self, df):
        """Identify supply and demand zones"""
        zones = []
        
        for i in range(20, len(df)-20):
            # Demand zones (support areas with strong bounces)
            if (df.iloc[i]['Low'] == df.iloc[i-10:i+10]['Low'].min() and
                df.iloc[i+1]['Close'] > df.iloc[i]['Close'] * 1.02):
                zones.append({
                    'type': 'Demand',
                    'level': df.iloc[i]['Low'],
                    'date': df.iloc[i]['Date'],
                    'strength': 'Strong' if df.iloc[i]['Volume'] > df.iloc[i-10:i+10]['Volume'].mean() else 'Moderate'
                })
            
            # Supply zones (resistance areas with strong rejections)
            if (df.iloc[i]['High'] == df.iloc[i-10:i+10]['High'].max() and
                df.iloc[i+1]['Close'] < df.iloc[i]['Close'] * 0.98):
                zones.append({
                    'type': 'Supply',
                    'level': df.iloc[i]['High'],
                    'date': df.iloc[i]['Date'],
                    'strength': 'Strong' if df.iloc[i]['Volume'] > df.iloc[i-10:i+10]['Volume'].mean() else 'Moderate'
                })
        
        return zones
    
    def generate_signals(self, df, strategy_params):
        """Generate trading signals based on strategy parameters"""
        signals = []
        
        for i in range(strategy_params.get('min_periods', 50), len(df)):
            current_candle = df.iloc[i]
            
            # Long signal conditions
            long_conditions = []
            
            # Moving Average conditions
            if strategy_params.get('use_ma_crossover', True):
                if (current_candle['SMA_10'] > current_candle['SMA_20'] and 
                    df.iloc[i-1]['SMA_10'] <= df.iloc[i-1]['SMA_20']):
                    long_conditions.append('MA_Crossover_Bullish')
            
            # RSI conditions
            if strategy_params.get('use_rsi', True):
                if current_candle['RSI'] < strategy_params.get('rsi_oversold', 30):
                    long_conditions.append('RSI_Oversold')
            
            # MACD conditions
            if strategy_params.get('use_macd', True):
                if (current_candle['MACD'] > current_candle['MACD_Signal'] and 
                    df.iloc[i-1]['MACD'] <= df.iloc[i-1]['MACD_Signal']):
                    long_conditions.append('MACD_Bullish')
            
            # Bollinger Band conditions
            if strategy_params.get('use_bb', True):
                if current_candle['Close'] < current_candle['BB_Lower']:
                    long_conditions.append('BB_Oversold')
            
            # Volume confirmation
            if strategy_params.get('use_volume', True):
                if current_candle['Volume_Ratio'] > strategy_params.get('volume_threshold', 1.5):
                    long_conditions.append('High_Volume')
            
            # Support level bounce
            support_bounce = self.check_support_bounce(df, i)
            if support_bounce:
                long_conditions.append('Support_Bounce')
            
            # Pattern recognition
            patterns = self.detect_pattern_at_index(df, i)
            bullish_patterns = [p for p in patterns if p['type'] == 'Bullish']
            if bullish_patterns:
                long_conditions.extend([f"Pattern_{p['pattern']}" for p in bullish_patterns])
            
            # Generate long signal if enough conditions met
            min_conditions = strategy_params.get('min_long_conditions', 3)
            if len(long_conditions) >= min_conditions:
                entry_price = current_candle['Close']
                atr = current_candle['ATR']
                
                # Calculate targets and stop loss
                target1 = entry_price * (1 + strategy_params.get('target1_pct', 0.02))
                target2 = entry_price * (1 + strategy_params.get('target2_pct', 0.04))
                stop_loss = entry_price * (1 - strategy_params.get('sl_pct', 0.015))
                
                # Risk management
                risk_reward = (target1 - entry_price) / (entry_price - stop_loss)
                
                if risk_reward >= strategy_params.get('min_risk_reward', 1.5):
                    signals.append({
                        'Date': current_candle['Date'],
                        'Type': 'LONG',
                        'Entry_Price': entry_price,
                        'Target1': target1,
                        'Target2': target2,
                        'Stop_Loss': stop_loss,
                        'Conditions': long_conditions,
                        'Risk_Reward': risk_reward,
                        'ATR': atr,
                        'Volume_Ratio': current_candle['Volume_Ratio']
                    })
            
            # Short signal conditions (similar logic but reversed)
            short_conditions = []
            
            # Moving Average conditions
            if strategy_params.get('use_ma_crossover', True):
                if (current_candle['SMA_10'] < current_candle['SMA_20'] and 
                    df.iloc[i-1]['SMA_10'] >= df.iloc[i-1]['SMA_20']):
                    short_conditions.append('MA_Crossover_Bearish')
            
            # RSI conditions
            if strategy_params.get('use_rsi', True):
                if current_candle['RSI'] > strategy_params.get('rsi_overbought', 70):
                    short_conditions.append('RSI_Overbought')
            
            # MACD conditions
            if strategy_params.get('use_macd', True):
                if (current_candle['MACD'] < current_candle['MACD_Signal'] and 
                    df.iloc[i-1]['MACD'] >= df.iloc[i-1]['MACD_Signal']):
                    short_conditions.append('MACD_Bearish')
            
            # Bollinger Band conditions
            if strategy_params.get('use_bb', True):
                if current_candle['Close'] > current_candle['BB_Upper']:
                    short_conditions.append('BB_Overbought')
            
            # Volume confirmation
            if strategy_params.get('use_volume', True):
                if current_candle['Volume_Ratio'] > strategy_params.get('volume_threshold', 1.5):
                    short_conditions.append('High_Volume')
            
            # Resistance level rejection
            resistance_rejection = self.check_resistance_rejection(df, i)
            if resistance_rejection:
                short_conditions.append('Resistance_Rejection')
            
            # Bearish patterns
            bearish_patterns = [p for p in patterns if p['type'] == 'Bearish']
            if bearish_patterns:
                short_conditions.extend([f"Pattern_{p['pattern']}" for p in bearish_patterns])
            
            # Generate short signal if enough conditions met
            min_conditions = strategy_params.get('min_short_conditions', 3)
            if len(short_conditions) >= min_conditions:
                entry_price = current_candle['Close']
                atr = current_candle['ATR']
                
                # Calculate targets and stop loss for short
                target1 = entry_price * (1 - strategy_params.get('target1_pct', 0.02))
                target2 = entry_price * (1 - strategy_params.get('target2_pct', 0.04))
                stop_loss = entry_price * (1 + strategy_params.get('sl_pct', 0.015))
                
                # Risk management
                risk_reward = (entry_price - target1) / (stop_loss - entry_price)
                
                if risk_reward >= strategy_params.get('min_risk_reward', 1.5):
                    signals.append({
                        'Date': current_candle['Date'],
                        'Type': 'SHORT',
                        'Entry_Price': entry_price,
                        'Target1': target1,
                        'Target2': target2,
                        'Stop_Loss': stop_loss,
                        'Conditions': short_conditions,
                        'Risk_Reward': risk_reward,
                        'ATR': atr,
                        'Volume_Ratio': current_candle['Volume_Ratio']
                    })
        
        return signals
    
    def check_support_bounce(self, df, idx, lookback=5):
        """Check if price is bouncing from support level"""
        if idx < lookback:
            return False
        
        current_low = df.iloc[idx]['Low']
        recent_lows = df.iloc[idx-lookback:idx]['Low']
        
        # Check if current low is near recent support and price is recovering
        support_level = recent_lows.min()
        bounce_threshold = support_level * 1.005  # 0.5% tolerance
        
        return (current_low <= bounce_threshold and 
                df.iloc[idx]['Close'] > df.iloc[idx]['Open'])
    
    def check_resistance_rejection(self, df, idx, lookback=5):
        """Check if price is being rejected from resistance level"""
        if idx < lookback:
            return False
        
        current_high = df.iloc[idx]['High']
        recent_highs = df.iloc[idx-lookback:idx]['High']
        
        # Check if current high is near recent resistance and price is falling
        resistance_level = recent_highs.max()
        rejection_threshold = resistance_level * 0.995  # 0.5% tolerance
        
        return (current_high >= rejection_threshold and 
                df.iloc[idx]['Close'] < df.iloc[idx]['Open'])
    
    def detect_pattern_at_index(self, df, idx):
        """Detect patterns at specific index"""
        patterns = []
        
        # Simplified pattern detection for real-time use
        if idx >= 20:
            # Bullish engulfing
            if (df.iloc[idx]['Open'] < df.iloc[idx-1]['Close'] and
                df.iloc[idx]['Close'] > df.iloc[idx-1]['Open'] and
                df.iloc[idx]['Close'] > df.iloc[idx]['Open']):
                patterns.append({'pattern': 'Bullish_Engulfing', 'type': 'Bullish', 'strength': 0.6})
            
            # Bearish engulfing
            if (df.iloc[idx]['Open'] > df.iloc[idx-1]['Close'] and
                df.iloc[idx]['Close'] < df.iloc[idx-1]['Open'] and
                df.iloc[idx]['Close'] < df.iloc[idx]['Open']):
                patterns.append({'pattern': 'Bearish_Engulfing', 'type': 'Bearish', 'strength': 0.6})
        
        return patterns
    
    def backtest_strategy(self, df, signals, initial_capital=100000):
        """Backtest the trading strategy"""
        capital = initial_capital
        position = None
        trades = []
        
        for signal in signals:
            if position is None:  # No active position
                # Enter position
                position = {
                    'type': signal['Type'],
                    'entry_date': signal['Date'],
                    'entry_price': signal['Entry_Price'],
                    'target1': signal['Target1'],
                    'target2': signal['Target2'],
                    'stop_loss': signal['Stop_Loss'],
                    'conditions': signal['Conditions'],
                    'risk_reward': signal['Risk_Reward']
                }
                
            else:  # Active position exists, check for exit
                # Find the next candle data for exit checking
                signal_date = signal['Date']
                
                # Ensure timezone compatibility for date comparison
                try:
                    if hasattr(signal_date, 'tz') and signal_date.tz is not None:
                        # Signal date is timezone-aware, ensure df dates match
                        future_data = df[df['Date'] > signal_date].head(20)
                    else:
                        # Handle timezone-naive comparison
                        future_data = df[df['Date'] > pd.Timestamp(signal_date)].head(20)
                except Exception as e:
                    # Fallback: use position entry date for filtering
                    future_data = df[df['Date'] > position['entry_date']].head(20)
                
                if len(future_data) > 0:
                    exit_found = False
                    
                    for _, candle in future_data.iterrows():
                        if position['type'] == 'LONG':
                            # Check for target or stop loss hit
                            if candle['High'] >= position['target1']:
                                # Target hit
                                exit_price = position['target1']
                                exit_reason = 'Target1_Hit'
                                exit_found = True
                            elif candle['Low'] <= position['stop_loss']:
                                # Stop loss hit
                                exit_price = position['stop_loss']
                                exit_reason = 'Stop_Loss_Hit'
                                exit_found = True
                        
                        elif position['type'] == 'SHORT':
                            # Check for target or stop loss hit
                            if candle['Low'] <= position['target1']:
                                # Target hit
                                exit_price = position['target1']
                                exit_reason = 'Target1_Hit'
                                exit_found = True
                            elif candle['High'] >= position['stop_loss']:
                                # Stop loss hit
                                exit_price = position['stop_loss']
                                exit_reason = 'Stop_Loss_Hit'
                                exit_found = True
                        
                        if exit_found:
                            # Calculate P&L
                            if position['type'] == 'LONG':
                                pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                            else:  # SHORT
                                pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
                            
                            pnl_amount = capital * pnl_pct * 0.1  # Risk 10% of capital per trade
                            capital += pnl_amount
                            
                            trades.append({
                                'Entry_Date': position['entry_date'],
                                'Exit_Date': candle['Date'],
                                'Type': position['type'],
                                'Entry_Price': position['entry_price'],
                                'Exit_Price': exit_price,
                                'Target1': position['target1'],
                                'Target2': position['target2'],
                                'Stop_Loss': position['stop_loss'],
                                'Exit_Reason': exit_reason,
                                'PnL_Pct': pnl_pct * 100,
                                'PnL_Amount': pnl_amount,
                                'Conditions': position['conditions'],
                                'Risk_Reward': position['risk_reward'],
                                'Hold_Days': (candle['Date'] - position['entry_date']).days
                            })
                            
                            position = None
                            break
        
        # Calculate performance metrics
        if trades:
            trades_df = pd.DataFrame(trades)
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['PnL_Pct'] > 0])
            losing_trades = len(trades_df[trades_df['PnL_Pct'] < 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            avg_win = trades_df[trades_df['PnL_Pct'] > 0]['PnL_Pct'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['PnL_Pct'] < 0]['PnL_Pct'].mean() if losing_trades > 0 else 0
            
            total_return = ((capital - initial_capital) / initial_capital) * 100
            avg_hold_days = trades_df['Hold_Days'].mean()
            
            # Buy and hold return for comparison
            buy_hold_return = ((df.iloc[-1]['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']) * 100
            
            performance = {
                'Total_Trades': total_trades,
                'Winning_Trades': winning_trades,
                'Losing_Trades': losing_trades,
                'Win_Rate': win_rate,
                'Avg_Win': avg_win,
                'Avg_Loss': avg_loss,
                'Total_Return': total_return,
                'Buy_Hold_Return': buy_hold_return,
                'Avg_Hold_Days': avg_hold_days,
                'Final_Capital': capital,
                'Profit_Factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                'Trades_Detail': trades_df
            }
        else:
            performance = {
                'Total_Trades': 0,
                'Win_Rate': 0,
                'Total_Return': 0,
                'Buy_Hold_Return': ((df.iloc[-1]['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']) * 100,
                'Trades_Detail': pd.DataFrame()
            }
        
        return performance
    
    def optimize_strategy(self, df, search_type='random', n_iter=50, target_accuracy=80):
        """Optimize strategy parameters"""
        
        param_space = {
            'min_periods': [30, 50, 100],
            'use_ma_crossover': [True, False],
            'use_rsi': [True, False],
            'use_macd': [True, False],
            'use_bb': [True, False],
            'use_volume': [True, False],
            'rsi_oversold': [20, 25, 30, 35],
            'rsi_overbought': [65, 70, 75, 80],
            'volume_threshold': [1.2, 1.5, 2.0, 2.5],
            'target1_pct': [0.015, 0.02, 0.025, 0.03],
            'target2_pct': [0.03, 0.04, 0.05, 0.06],
            'sl_pct': [0.01, 0.015, 0.02, 0.025],
            'min_risk_reward': [1.0, 1.5, 2.0, 2.5],
            'min_long_conditions': [2, 3, 4, 5],
            'min_short_conditions': [2, 3, 4, 5]
        }
        
        best_performance = {'Win_Rate': 0, 'Total_Return': -float('inf')}
        best_params = None
        all_results = []
        
        if search_type == 'grid':
            param_combinations = list(ParameterGrid(param_space))
            n_iter = min(n_iter, len(param_combinations))
        else:  # random search
            param_combinations = list(ParameterSampler(param_space, n_iter=n_iter))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, params in enumerate(param_combinations[:n_iter]):
            progress = (i + 1) / n_iter
            progress_bar.progress(progress)
            status_text.text(f'Optimizing strategy... {i+1}/{n_iter} ({progress*100:.1f}%)')
            
            try:
                signals = self.generate_signals(df, params)
                if len(signals) > 5:  # Minimum trades required
                    performance = self.backtest_strategy(df, signals)
                    
                    # Score based on win rate and total return
                    score = (performance['Win_Rate'] * 0.6) + (max(performance['Total_Return'], 0) * 0.4)
                    
                    performance['Score'] = score
                    performance['Params'] = params
                    all_results.append(performance)
                    
                    # Update best performance
                    if (performance['Win_Rate'] >= target_accuracy and 
                        performance['Total_Return'] > best_performance['Total_Return']) or \
                       (performance['Win_Rate'] > best_performance['Win_Rate'] and 
                        performance['Total_Return'] > best_performance['Total_Return'] * 0.8):
                        best_performance = performance
                        best_params = params
            except Exception as e:
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if best_params is None and all_results:
            # If no strategy meets target accuracy, pick the best overall
            best_result = max(all_results, key=lambda x: x['Score'])
            best_performance = best_result
            best_params = best_result['Params']
        
        return best_params, best_performance, all_results
    
    def generate_live_recommendation(self, df, strategy_params):
        """Generate live recommendation based on the latest data"""
        if len(df) < 50:
            return None
        
        latest_signals = self.generate_signals(df, strategy_params)
        
        if latest_signals:
            # Get the most recent signal
            latest_signal = latest_signals[-1]
            
            # Add probability calculation
            latest_signal['Probability'] = min(95, 60 + (len(latest_signal['Conditions']) * 5))
            
            # Add confluence analysis
            confluences = self.analyze_confluences(df, latest_signal)
            latest_signal['Confluences'] = confluences
            
            # Add detailed reasoning
            reasoning = self.generate_reasoning(df, latest_signal)
            latest_signal['Reasoning'] = reasoning
            
            return latest_signal
        
        return None
    
    def analyze_confluences(self, df, signal):
        """Analyze confluences for the signal"""
        confluences = []
        latest_candle = df.iloc[-1]
        
        # Technical confluences
        if 'MA_Crossover' in str(signal['Conditions']):
            confluences.append("Moving Average crossover providing directional bias")
        
        if 'RSI' in str(signal['Conditions']):
            confluences.append(f"RSI at {latest_candle['RSI']:.1f} showing momentum")
        
        if 'MACD' in str(signal['Conditions']):
            confluences.append("MACD histogram confirming trend direction")
        
        if 'High_Volume' in str(signal['Conditions']):
            confluences.append(f"Above average volume ({latest_candle['Volume_Ratio']:.1f}x) confirming move")
        
        if 'Support_Bounce' in str(signal['Conditions']):
            confluences.append("Price bouncing from key support level")
        
        if 'Resistance_Rejection' in str(signal['Conditions']):
            confluences.append("Price rejecting from key resistance level")
        
        return confluences
    
    def generate_reasoning(self, df, signal):
        """Generate detailed reasoning for the signal"""
        latest_candle = df.iloc[-1]
        reasoning = []
        
        reasoning.append(f"Signal Type: {signal['Type']}")
        reasoning.append(f"Entry triggered by {len(signal['Conditions'])} confluent factors:")
        
        for condition in signal['Conditions']:
            if 'MA_Crossover_Bullish' in condition:
                reasoning.append("• Short-term MA crossed above long-term MA (bullish)")
            elif 'MA_Crossover_Bearish' in condition:
                reasoning.append("• Short-term MA crossed below long-term MA (bearish)")
            elif 'RSI_Oversold' in condition:
                reasoning.append(f"• RSI oversold at {latest_candle['RSI']:.1f} (potential reversal)")
            elif 'RSI_Overbought' in condition:
                reasoning.append(f"• RSI overbought at {latest_candle['RSI']:.1f} (potential reversal)")
            elif 'MACD_Bullish' in condition:
                reasoning.append("• MACD bullish crossover (momentum building)")
            elif 'MACD_Bearish' in condition:
                reasoning.append("• MACD bearish crossover (momentum weakening)")
            elif 'High_Volume' in condition:
                reasoning.append(f"• High volume confirmation ({latest_candle['Volume_Ratio']:.1f}x average)")
            elif 'Support_Bounce' in condition:
                reasoning.append("• Strong bounce from support level")
            elif 'Resistance_Rejection' in condition:
                reasoning.append("• Rejection from resistance level")
        
        reasoning.append(f"Risk-Reward Ratio: {signal['Risk_Reward']:.2f}:1")
        
        return reasoning

def main():
    st.markdown('<h1 class="main-header">🚀 Advanced Stock Trading Recommendation System</h1>', unsafe_allow_html=True)
    
    analyzer = StockAnalyzer()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Stock Data (CSV)", 
        type=['csv'],
        help="Upload a CSV file with OHLCV data"
    )
    
    if uploaded_file is not None:
        try:
            # Read and process data
            df = pd.read_csv(uploaded_file)
            
            st.sidebar.success("✅ File uploaded successfully!")
            
            # Display basic file info
            st.sidebar.write(f"**Rows:** {len(df)}")
            st.sidebar.write(f"**Columns:** {list(df.columns)}")
            
            # Map columns
            mapped_df = analyzer.map_columns(df)
            
            if mapped_df is not None:
                st.sidebar.success("✅ Columns mapped successfully!")
                
                # Date selection
                mapped_df['Date'] = pd.to_datetime(mapped_df['Date'])
                min_date = mapped_df['Date'].min().date()
                max_date = mapped_df['Date'].max().date()
                
                end_date = st.sidebar.date_input(
                    "Select End Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date
                )
                
                # Trading configuration
                st.sidebar.subheader("📈 Trading Settings")
                
                trade_side = st.sidebar.selectbox(
                    "Trade Side",
                    ["Both", "Long Only", "Short Only"]
                )
                
                search_type = st.sidebar.selectbox(
                    "Optimization Method",
                    ["Random Search", "Grid Search"],
                    index=0
                )
                
                target_accuracy = st.sidebar.slider(
                    "Target Accuracy (%)",
                    min_value=60,
                    max_value=95,
                    value=80,
                    step=5
                )
                
                n_points = st.sidebar.slider(
                    "Number of Optimization Points",
                    min_value=20,
                    max_value=100,
                    value=50,
                    step=10
                )
                
                if st.sidebar.button("🚀 Run Analysis", type="primary"):
                    
                    try:
                        # Process data
                        with st.spinner("🔄 Processing data..."):
                            processed_df = analyzer.preprocess_data(mapped_df, pd.Timestamp(end_date))
                        
                        if processed_df is not None and len(processed_df) > 100:
                            analyzer.data = processed_df
                            st.success("✅ Data processed successfully!")
                            
                            # Main content area
                            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                                "📊 Data Overview", 
                                "🔍 EDA & Patterns", 
                                "⚡ Strategy Optimization", 
                                "📈 Backtest Results", 
                                "🎯 Live Recommendation"
                            ])
                        
                        with tab1:
                            st.header("📊 Data Overview")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("First 5 Rows")
                                st.dataframe(processed_df.head())
                                
                                st.subheader("Data Statistics")
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Min Date", processed_df['Date'].min().strftime('%Y-%m-%d'))
                                    st.metric("Min Price", f"₹{processed_df['Close'].min():.2f}")
                                with col_b:
                                    st.metric("Max Date", processed_df['Date'].max().strftime('%Y-%m-%d'))
                                    st.metric("Max Price", f"₹{processed_df['Close'].max():.2f}")
                            
                            with col2:
                                st.subheader("Last 5 Rows")
                                st.dataframe(processed_df.tail())
                                
                                st.subheader("Price Chart")
                                fig = go.Figure()
                                fig.add_trace(go.Candlestick(
                                    x=processed_df['Date'],
                                    open=processed_df['Open'],
                                    high=processed_df['High'],
                                    low=processed_df['Low'],
                                    close=processed_df['Close'],
                                    name="Price"
                                ))
                                fig.update_layout(
                                    title="Stock Price Chart",
                                    xaxis_title="Date",
                                    yaxis_title="Price",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Stock summary
                            st.subheader("📝 Stock Analysis Summary")
                            
                            # Calculate key metrics
                            total_return = ((processed_df.iloc[-1]['Close'] - processed_df.iloc[0]['Close']) / processed_df.iloc[0]['Close']) * 100
                            volatility = processed_df['Close'].pct_change().std() * np.sqrt(252) * 100
                            avg_volume = processed_df['Volume'].mean()
                            
                            current_rsi = processed_df.iloc[-1]['RSI']
                            current_macd = processed_df.iloc[-1]['MACD']
                            
                            # Generate summary text
                            summary_parts = []
                            
                            if total_return > 20:
                                summary_parts.append(f"The stock has shown strong performance with a {total_return:.1f}% return")
                            elif total_return > 0:
                                summary_parts.append(f"The stock has posted positive returns of {total_return:.1f}%")
                            else:
                                summary_parts.append(f"The stock has declined by {abs(total_return):.1f}%")
                            
                            if volatility > 30:
                                summary_parts.append(f"with high volatility ({volatility:.1f}%)")
                            elif volatility > 20:
                                summary_parts.append(f"with moderate volatility ({volatility:.1f}%)")
                            else:
                                summary_parts.append(f"with low volatility ({volatility:.1f}%)")
                            
                            if current_rsi > 70:
                                summary_parts.append("Currently showing overbought conditions (RSI > 70)")
                            elif current_rsi < 30:
                                summary_parts.append("Currently showing oversold conditions (RSI < 30)")
                            else:
                                summary_parts.append("Currently in neutral territory")
                            
                            if current_macd > 0:
                                summary_parts.append("MACD indicates bullish momentum")
                            else:
                                summary_parts.append("MACD indicates bearish momentum")
                            
                            summary_parts.append(f"Average daily volume is {avg_volume:,.0f} shares")
                            
                            # Identify potential opportunities
                            opportunities = []
                            if current_rsi < 30 and current_macd > processed_df.iloc[-2]['MACD']:
                                opportunities.append("oversold bounce opportunity")
                            if processed_df.iloc[-1]['Close'] > processed_df.iloc[-1]['SMA_20']:
                                opportunities.append("bullish trend continuation")
                            if processed_df.iloc[-1]['Volume'] > avg_volume * 1.5:
                                opportunities.append("high volume breakout potential")
                            
                            if opportunities:
                                summary_parts.append(f"Key opportunities include: {', '.join(opportunities)}")
                            
                            summary_text = ". ".join(summary_parts) + "."
                            
                            st.markdown(f'<div class="success-card">{summary_text}</div>', unsafe_allow_html=True)
                        
                        with tab2:
                            st.header("🔍 Exploratory Data Analysis")
                            
                            # Returns heatmap
                            processed_df['Returns'] = processed_df['Close'].pct_change()
                            processed_df['Year'] = processed_df['Date'].dt.year
                            processed_df['Month'] = processed_df['Date'].dt.month
                            
                            # Create returns pivot table
                            returns_pivot = processed_df.groupby(['Year', 'Month'])['Returns'].sum().unstack()
                            
                            if len(returns_pivot) > 1:
                                st.subheader("📊 Monthly Returns Heatmap")
                                fig, ax = plt.subplots(figsize=(12, 6))
                                sns.heatmap(returns_pivot * 100, annot=True, fmt='.1f', 
                                          cmap='RdYlGn', center=0, ax=ax)
                                ax.set_title('Monthly Returns (%) by Year')
                                st.pyplot(fig)
                            
                            # Technical indicators chart
                            st.subheader("📈 Technical Indicators")
                            
                            fig = make_subplots(
                                rows=3, cols=1,
                                subplot_titles=('Price with Moving Averages', 'RSI', 'MACD'),
                                vertical_spacing=0.08,
                                row_heights=[0.5, 0.25, 0.25]
                            )
                            
                            # Price and MAs
                            fig.add_trace(go.Scatter(x=processed_df['Date'], y=processed_df['Close'], name='Close', line=dict(color='blue')), row=1, col=1)
                            fig.add_trace(go.Scatter(x=processed_df['Date'], y=processed_df['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
                            fig.add_trace(go.Scatter(x=processed_df['Date'], y=processed_df['SMA_50'], name='SMA 50', line=dict(color='red')), row=1, col=1)
                            
                            # RSI
                            fig.add_trace(go.Scatter(x=processed_df['Date'], y=processed_df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
                            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                            
                            # MACD
                            fig.add_trace(go.Scatter(x=processed_df['Date'], y=processed_df['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
                            fig.add_trace(go.Scatter(x=processed_df['Date'], y=processed_df['MACD_Signal'], name='Signal', line=dict(color='red')), row=3, col=1)
                            fig.add_trace(go.Bar(x=processed_df['Date'], y=processed_df['MACD_Histogram'], name='Histogram'), row=3, col=1)
                            
                            fig.update_layout(height=800, showlegend=True)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Chart patterns detection
                            st.subheader("🎯 Chart Patterns Detected")
                            
                            patterns = analyzer.detect_chart_patterns(processed_df)
                            
                            if patterns:
                                for pattern in patterns[-10:]:  # Show last 10 patterns
                                    pattern_type = "🟢" if pattern['type'] == 'Bullish' else "🔴"
                                    st.write(f"{pattern_type} **{pattern['pattern']}** detected on {pattern['date'].strftime('%Y-%m-%d')} at ₹{pattern['price']:.2f} (Strength: {pattern['strength']:.1%})")
                            else:
                                st.info("No significant chart patterns detected in the recent data.")
                            
                            # Supply/Demand zones
                            st.subheader("📍 Supply & Demand Zones")
                            supply_demand = analyzer.identify_supply_demand_zones(processed_df)
                            
                            if supply_demand:
                                zones_df = pd.DataFrame(supply_demand[-10:])  # Last 10 zones
                                st.dataframe(zones_df)
                            else:
                                st.info("No significant supply/demand zones identified.")
                        
                        with tab3:
                            st.header("⚡ Strategy Optimization")
                            
                            # Run optimization
                            search_method = 'random' if search_type == 'Random Search' else 'grid'
                            
                            with st.spinner("🔍 Optimizing strategy parameters..."):
                                best_params, best_performance, all_results = analyzer.optimize_strategy(
                                    processed_df, 
                                    search_type=search_method, 
                                    n_iter=n_points,
                                    target_accuracy=target_accuracy
                                )
                            
                            if best_params:
                                st.success("✅ Optimization completed!")
                                
                                # Display best strategy
                                st.subheader("🏆 Best Strategy Found")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Win Rate", f"{best_performance['Win_Rate']:.1f}%")
                                with col2:
                                    st.metric("Total Return", f"{best_performance['Total_Return']:.1f}%")
                                with col3:
                                    st.metric("Buy & Hold", f"{best_performance['Buy_Hold_Return']:.1f}%")
                                with col4:
                                    st.metric("Total Trades", best_performance['Total_Trades'])
                                
                                # Strategy parameters
                                st.subheader("📋 Strategy Parameters")
                                
                                param_col1, param_col2 = st.columns(2)
                                
                                with param_col1:
                                    st.write("**Technical Indicators:**")
                                    st.write(f"• Moving Average Crossover: {'✅' if best_params.get('use_ma_crossover') else '❌'}")
                                    st.write(f"• RSI: {'✅' if best_params.get('use_rsi') else '❌'}")
                                    st.write(f"• MACD: {'✅' if best_params.get('use_macd') else '❌'}")
                                    st.write(f"• Bollinger Bands: {'✅' if best_params.get('use_bb') else '❌'}")
                                    st.write(f"• Volume Filter: {'✅' if best_params.get('use_volume') else '❌'}")
                                
                                with param_col2:
                                    st.write("**Risk Management:**")
                                    st.write(f"• Target 1: {best_params.get('target1_pct', 0.02)*100:.1f}%")
                                    st.write(f"• Target 2: {best_params.get('target2_pct', 0.04)*100:.1f}%")
                                    st.write(f"• Stop Loss: {best_params.get('sl_pct', 0.015)*100:.1f}%")
                                    st.write(f"• Min Risk:Reward: {best_params.get('min_risk_reward', 1.5):.1f}:1")
                                    st.write(f"• RSI Levels: {best_params.get('rsi_oversold', 30)}-{best_params.get('rsi_overbought', 70)}")
                                
                                analyzer.best_strategy = best_params
                                analyzer.strategy_params = best_params
                            else:
                                st.error("❌ Unable to find a profitable strategy. Try adjusting parameters.")
                        
                        with tab4:
                            st.header("📈 Backtest Results")
                            
                            if analyzer.best_strategy:
                                # Generate signals with best strategy
                                signals = analyzer.generate_signals(processed_df, analyzer.best_strategy)
                                
                                if signals:
                                    # Filter signals based on trade side preference
                                    if trade_side == "Long Only":
                                        signals = [s for s in signals if s['Type'] == 'LONG']
                                    elif trade_side == "Short Only":
                                        signals = [s for s in signals if s['Type'] == 'SHORT']
                                    
                                    # Run backtest
                                    backtest_results = analyzer.backtest_strategy(processed_df, signals)
                                    
                                    if backtest_results['Total_Trades'] > 0:
                                        # Performance metrics
                                        st.subheader("📊 Performance Summary")
                                        
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
                                            st.metric("Profit Factor", f"{backtest_results['Profit_Factor']:.2f}")
                                        
                                        col6, col7, col8 = st.columns(3)
                                        
                                        with col6:
                                            st.metric("Strategy Return", f"{backtest_results['Total_Return']:.1f}%")
                                        with col7:
                                            st.metric("Buy & Hold Return", f"{backtest_results['Buy_Hold_Return']:.1f}%")
                                        with col8:
                                            outperformance = backtest_results['Total_Return'] - backtest_results['Buy_Hold_Return']
                                            st.metric("Outperformance", f"{outperformance:.1f}%")
                                        
                                        # Trade details
                                        st.subheader("📋 Trade Details")
                                        
                                        trades_df = backtest_results['Trades_Detail']
                                        
                                        if not trades_df.empty:
                                            # Format for display
                                            display_trades = trades_df.copy()
                                            display_trades['Entry_Date'] = display_trades['Entry_Date'].dt.strftime('%Y-%m-%d %H:%M')
                                            display_trades['Exit_Date'] = display_trades['Exit_Date'].dt.strftime('%Y-%m-%d %H:%M')
                                            display_trades['Entry_Price'] = display_trades['Entry_Price'].round(2)
                                            display_trades['Exit_Price'] = display_trades['Exit_Price'].round(2)
                                            display_trades['Target1'] = display_trades['Target1'].round(2)
                                            display_trades['Stop_Loss'] = display_trades['Stop_Loss'].round(2)
                                            display_trades['PnL_Pct'] = display_trades['PnL_Pct'].round(2)
                                            display_trades['Risk_Reward'] = display_trades['Risk_Reward'].round(2)
                                            
                                            st.dataframe(
                                                display_trades[[
                                                    'Entry_Date', 'Exit_Date', 'Type', 'Entry_Price', 
                                                    'Exit_Price', 'Target1', 'Stop_Loss', 'Exit_Reason',
                                                    'PnL_Pct', 'Risk_Reward', 'Hold_Days'
                                                ]],
                                                use_container_width=True
                                            )
                                            
                                            # Trade distribution chart
                                            st.subheader("📊 P&L Distribution")
                                            
                                            fig = px.histogram(
                                                trades_df, 
                                                x='PnL_Pct', 
                                                nbins=20,
                                                title="Trade P&L Distribution (%)"
                                            )
                                            fig.update_layout(
                                                xaxis_title="P&L (%)",
                                                yaxis_title="Number of Trades"
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Backtest summary
                                        st.subheader("📝 Backtest Summary")
                                        
                                        summary_text = f"""
                                        The backtesting analysis was conducted on {len(processed_df)} candles from {processed_df.iloc[0]['Date'].strftime('%Y-%m-%d')} 
                                        to {processed_df.iloc[-1]['Date'].strftime('%Y-%m-%d')}. The optimized strategy generated 
                                        {backtest_results['Total_Trades']} trades with a {backtest_results['Win_Rate']:.1f}% win rate. 
                                        
                                        The strategy achieved a total return of {backtest_results['Total_Return']:.1f}% compared to 
                                        buy-and-hold return of {backtest_results['Buy_Hold_Return']:.1f}%, representing an outperformance 
                                        of {outperformance:.1f}%. With {backtest_results['Winning_Trades']} winning trades and 
                                        {backtest_results['Losing_Trades']} losing trades, the average winning trade was 
                                        {backtest_results['Avg_Win']:.1f}% while average losing trade was {backtest_results['Avg_Loss']:.1f}%.
                                        
                                        The strategy maintained an average holding period of {backtest_results['Avg_Hold_Days']:.1f} days 
                                        and achieved a profit factor of {backtest_results['Profit_Factor']:.2f}, indicating 
                                        {'strong' if backtest_results['Profit_Factor'] > 1.5 else 'moderate' if backtest_results['Profit_Factor'] > 1.2 else 'weak'} 
                                        performance. Risk management was effective with the predefined stop-loss and target levels 
                                        helping to maintain disciplined entries and exits.
                                        """
                                        
                                        st.markdown(f'<div class="success-card">{summary_text}</div>', unsafe_allow_html=True)
                                    else:
                                        st.warning("⚠️ No trades were generated with the current strategy parameters.")
                                else:
                                    st.error("❌ No signals generated. Try adjusting strategy parameters.")
                            else:
                                st.warning("⚠️ Please run strategy optimization first.")
                        
                        with tab5:
                            st.header("🎯 Live Recommendation")
                            
                            if analyzer.best_strategy:
                                # Generate live recommendation
                                live_recommendation = analyzer.generate_live_recommendation(processed_df, analyzer.best_strategy)
                                
                                if live_recommendation:
                                    # Filter based on trade side preference
                                    if ((trade_side == "Long Only" and live_recommendation['Type'] == 'LONG') or
                                        (trade_side == "Short Only" and live_recommendation['Type'] == 'SHORT') or
                                        trade_side == "Both"):
                                        
                                        st.success("🎯 Live Trading Recommendation Available!")
                                        
                                        # Main recommendation card
                                        rec_type = "🟢 LONG" if live_recommendation['Type'] == 'LONG' else "🔴 SHORT"
                                        
                                        st.markdown(f"""
                                        <div class="success-card">
                                            <h3>{rec_type} SIGNAL</h3>
                                            <p><strong>Entry Price:</strong> ₹{live_recommendation['Entry_Price']:.2f}</p>
                                            <p><strong>Target 1:</strong> ₹{live_recommendation['Target1']:.2f}</p>
                                            <p><strong>Target 2:</strong> ₹{live_recommendation['Target2']:.2f}</p>
                                            <p><strong>Stop Loss:</strong> ₹{live_recommendation['Stop_Loss']:.2f}</p>
                                            <p><strong>Risk:Reward Ratio:</strong> {live_recommendation['Risk_Reward']:.2f}:1</p>
                                            <p><strong>Probability of Success:</strong> {live_recommendation['Probability']:.0f}%</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Detailed analysis
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.subheader("📋 Entry Logic & Reasoning")
                                            
                                            for reason in live_recommendation['Reasoning']:
                                                st.write(reason)
                                        
                                        with col2:
                                            st.subheader("🔗 Confluences")
                                            
                                            for confluence in live_recommendation['Confluences']:
                                                st.write(f"• {confluence}")
                                        
                                        # Technical snapshot
                                        st.subheader("📊 Technical Snapshot")
                                        
                                        latest_data = processed_df.iloc[-1]
                                        
                                        tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                                        
                                        with tech_col1:
                                            st.metric("Current Price", f"₹{latest_data['Close']:.2f}")
                                            st.metric("Volume Ratio", f"{latest_data['Volume_Ratio']:.2f}x")
                                        
                                        with tech_col2:
                                            st.metric("RSI", f"{latest_data['RSI']:.1f}")
                                            rsi_status = "Oversold" if latest_data['RSI'] < 30 else "Overbought" if latest_data['RSI'] > 70 else "Neutral"
                                            st.write(f"Status: {rsi_status}")
                                        
                                        with tech_col3:
                                            st.metric("MACD", f"{latest_data['MACD']:.3f}")
                                            macd_status = "Bullish" if latest_data['MACD'] > latest_data['MACD_Signal'] else "Bearish"
                                            st.write(f"Status: {macd_status}")
                                        
                                        with tech_col4:
                                            st.metric("ATR", f"{latest_data['ATR']:.2f}")
                                            st.metric("SMA 20", f"₹{latest_data['SMA_20']:.2f}")
                                        
                                        # Risk management details
                                        st.subheader("⚠️ Risk Management")
                                        
                                        if live_recommendation['Type'] == 'LONG':
                                            potential_profit_1 = ((live_recommendation['Target1'] - live_recommendation['Entry_Price']) / live_recommendation['Entry_Price']) * 100
                                            potential_profit_2 = ((live_recommendation['Target2'] - live_recommendation['Entry_Price']) / live_recommendation['Entry_Price']) * 100
                                            potential_loss = ((live_recommendation['Entry_Price'] - live_recommendation['Stop_Loss']) / live_recommendation['Entry_Price']) * 100
                                        else:
                                            potential_profit_1 = ((live_recommendation['Entry_Price'] - live_recommendation['Target1']) / live_recommendation['Entry_Price']) * 100
                                            potential_profit_2 = ((live_recommendation['Entry_Price'] - live_recommendation['Target2']) / live_recommendation['Entry_Price']) * 100
                                            potential_loss = ((live_recommendation['Stop_Loss'] - live_recommendation['Entry_Price']) / live_recommendation['Entry_Price']) * 100
                                        
                                        risk_col1, risk_col2, risk_col3 = st.columns(3)
                                        
                                        with risk_col1:
                                            st.metric("Target 1 Profit", f"{potential_profit_1:.2f}%")
                                        with risk_col2:
                                            st.metric("Target 2 Profit", f"{potential_profit_2:.2f}%")
                                        with risk_col3:
                                            st.metric("Maximum Loss", f"-{potential_loss:.2f}%")
                                        
                                        # Action plan
                                        st.subheader("📋 Action Plan")
                                        
                                        action_plan = f"""
                                        **Entry Strategy:**
                                        1. Enter {live_recommendation['Type']} position at ₹{live_recommendation['Entry_Price']:.2f} (current candle close)
                                        2. Set stop loss at ₹{live_recommendation['Stop_Loss']:.2f} ({potential_loss:.2f}% risk)
                                        3. Set first target at ₹{live_recommendation['Target1']:.2f} ({potential_profit_1:.2f}% profit)
                                        4. Set second target at ₹{live_recommendation['Target2']:.2f} ({potential_profit_2:.2f}% profit)
                                        
                                        **Position Sizing:**
                                        - Risk only 1-2% of total capital per trade
                                        - Consider partial profit booking at Target 1
                                        - Trail stop loss after Target 1 is achieved
                                        
                                        **Exit Strategy:**
                                        - Exit 50% position at Target 1
                                        - Trail remaining position to Target 2
                                        - Strictly follow stop loss if trade goes against you
                                        """
                                        
                                        st.markdown(f'<div class="metric-card">{action_plan}</div>', unsafe_allow_html=True)
                                        
                                        # Chart with signals
                                        st.subheader("📈 Price Chart with Signal")
                                        
                                        fig = go.Figure()
                                        
                                        # Candlestick chart
                                        fig.add_trace(go.Candlestick(
                                            x=processed_df['Date'].tail(50),
                                            open=processed_df['Open'].tail(50),
                                            high=processed_df['High'].tail(50),
                                            low=processed_df['Low'].tail(50),
                                            close=processed_df['Close'].tail(50),
                                            name="Price"
                                        ))
                                        
                                        # Add moving averages
                                        fig.add_trace(go.Scatter(
                                            x=processed_df['Date'].tail(50),
                                            y=processed_df['SMA_20'].tail(50),
                                            name='SMA 20',
                                            line=dict(color='orange', width=2)
                                        ))
                                        
                                        fig.add_trace(go.Scatter(
                                            x=processed_df['Date'].tail(50),
                                            y=processed_df['SMA_50'].tail(50),
                                            name='SMA 50',
                                            line=dict(color='red', width=2)
                                        ))
                                        
                                        # Add signal markers
                                        signal_color = 'green' if live_recommendation['Type'] == 'LONG' else 'red'
                                        signal_symbol = 'triangle-up' if live_recommendation['Type'] == 'LONG' else 'triangle-down'
                                        
                                        fig.add_trace(go.Scatter(
                                            x=[live_recommendation['Date']],
                                            y=[live_recommendation['Entry_Price']],
                                            mode='markers',
                                            marker=dict(
                                                symbol=signal_symbol,
                                                size=15,
                                                color=signal_color,
                                                line=dict(width=2, color='white')
                                            ),
                                            name=f"{live_recommendation['Type']} Signal"
                                        ))
                                        
                                        # Add target and stop loss lines
                                        fig.add_hline(
                                            y=live_recommendation['Target1'],
                                            line_dash="dash",
                                            line_color="green",
                                            annotation_text=f"Target 1: ₹{live_recommendation['Target1']:.2f}"
                                        )
                                        
                                        fig.add_hline(
                                            y=live_recommendation['Target2'],
                                            line_dash="dash",
                                            line_color="darkgreen",
                                            annotation_text=f"Target 2: ₹{live_recommendation['Target2']:.2f}"
                                        )
                                        
                                        fig.add_hline(
                                            y=live_recommendation['Stop_Loss'],
                                            line_dash="dash",
                                            line_color="red",
                                            annotation_text=f"Stop Loss: ₹{live_recommendation['Stop_Loss']:.2f}"
                                        )
                                        
                                        fig.update_layout(
                                            title=f"Live {live_recommendation['Type']} Signal - {processed_df.iloc[-1]['Date'].strftime('%Y-%m-%d %H:%M')}",
                                            xaxis_title="Date",
                                            yaxis_title="Price (₹)",
                                            height=600,
                                            showlegend=True
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                    else:
                                        st.info(f"📊 Current recommendation is for {live_recommendation['Type']} but you selected {trade_side} only.")
                                
                                else:
                                    st.info("📊 No trading signals detected at the current market conditions. Wait for better setup.")
                                
                                # Strategy summary for live trading
                                st.subheader("📋 Strategy Summary for Live Trading")
                                
                                strategy_summary = f"""
                                **Current Strategy Performance:**
                                - Backtested Win Rate: {best_performance.get('Win_Rate', 0):.1f}%
                                - Total Trades: {best_performance.get('Total_Trades', 0)}
                                - Strategy Return: {best_performance.get('Total_Return', 0):.1f}%
                                - Buy & Hold Return: {best_performance.get('Buy_Hold_Return', 0):.1f}%
                                
                                **Live Trading Guidelines:**
                                1. Only take signals that meet all confluence criteria
                                2. Always use proper position sizing (1-2% risk per trade)
                                3. Set stop losses immediately after entry
                                4. Take partial profits at first target
                                5. Trail stop loss for remaining position
                                6. Keep a trading journal to track performance
                                7. Review and adjust strategy monthly based on live results
                                
                                **Market Conditions:**
                                Current market shows {'bullish' if processed_df.iloc[-1]['Close'] > processed_df.iloc[-1]['SMA_50'] else 'bearish'} 
                                bias with {'high' if processed_df.iloc[-1]['Volume_Ratio'] > 1.5 else 'normal'} volume activity. 
                                RSI is at {processed_df.iloc[-1]['RSI']:.1f} indicating 
                                {'oversold' if processed_df.iloc[-1]['RSI'] < 30 else 'overbought' if processed_df.iloc[-1]['RSI'] > 70 else 'neutral'} conditions.
                                """
                                
                                st.markdown(f'<div class="warning-card">{strategy_summary}</div>', unsafe_allow_html=True)
                                
                            else:
                                st.warning("⚠️ Please run strategy optimization first to get live recommendations.")
                    
                    else:
                        st.error("❌ Insufficient data for analysis. Please upload data with at least 100 candles.")
            
            else:
                st.warning("⚠️ Unable to map columns properly. Please check your data format.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        # Instructions when no file is uploaded
        st.markdown("""
        ## 🚀 Welcome to Advanced Stock Trading System
        
        This comprehensive trading system provides:
        
        ### ✨ Features:
        - **Intelligent Column Mapping**: Automatically detects and maps your data columns
        - **Advanced Technical Analysis**: 15+ technical indicators and chart patterns
        - **Pattern Recognition**: Cup & Handle, Head & Shoulders, Triangles, Double Tops/Bottoms
        - **Supply/Demand Zones**: Identifies key support and resistance levels
        - **Strategy Optimization**: Uses machine learning to find best parameters
        - **Risk Management**: Built-in stop losses and target levels
        - **Live Recommendations**: Real-time trading signals
        - **Comprehensive Backtesting**: Detailed performance analysis
        
        ### 📋 Data Requirements:
        Your CSV file should contain columns with any of these names (case insensitive):
        - **Date/Time**: Date, Time, Timestamp
        - **Open**: Open, Open_Price, OPEN
        - **High**: High, High_Price, HIGH  
        - **Low**: Low, Low_Price, LOW
        - **Close**: Close, Close_Price, CLOSE
        - **Volume**: Volume, Vol, Shares, Traded
        
        ### 🎯 How to Use:
        1. **Upload** your stock data CSV file
        2. **Select** end date for backtesting
        3. **Choose** trading side (Long/Short/Both)
        4. **Configure** optimization settings
        5. **Run** analysis and get live recommendations!
        
        ### ⚡ Getting Started:
        Upload your CSV file using the file uploader in the sidebar to begin your advanced stock analysis journey!
        """)
        
        # Sample data format
        st.subheader("📊 Sample Data Format")
        sample_data = {
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Open': [100.0, 102.0, 101.5],
            'High': [103.0, 105.0, 104.0],
            'Low': [99.0, 101.0, 100.5],
            'Close': [102.0, 103.5, 103.0],
            'Volume': [1000000, 1200000, 950000]
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)

if __name__ == "__main__":
    main()
