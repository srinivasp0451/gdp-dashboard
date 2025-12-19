import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pytz
from abc import ABC, abstractmethod
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, argrelextrema
import math

# Page Configuration
st.set_page_config(page_title="Pro Algo Trading", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 20px;}
    .metric-card {background: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;}
    .profit {color: #00c853; font-weight: bold;}
    .loss {color: #d32f2f; font-weight: bold;}
    .status-active {color: #ff9800; font-weight: bold;}
    .trade-guidance {background: #e3f2fd; padding: 20px; border-radius: 10px; border-left: 5px solid #2196f3; margin: 15px 0;}
    .stButton>button {width: 100%; padding: 10px; font-size: 16px; font-weight: bold;}
    .log-container {height: 600px; overflow-y: auto; background: #fafafa; padding: 15px; border-radius: 5px; font-family: monospace; font-size: 12px;}
</style>
""", unsafe_allow_html=True)

# ==================== UTILITIES ====================

IST = pytz.timezone('Asia/Kolkata')

def convert_to_ist(dt):
    """Convert datetime to IST"""
    if dt is None:
        return None
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    return dt.astimezone(IST)

def log_event(message):
    """Add timestamped event to trade log"""
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    timestamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    st.session_state.trade_log.append(f"[{timestamp}] {message}")
    if len(st.session_state.trade_log) > 100:
        st.session_state.trade_log = st.session_state.trade_log[-100:]

# ==================== BASE STRATEGY CLASS ====================

class BaseStrategy(ABC):
    def __init__(self, params=None):
        self.params = params or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def calculate_indicators(self, data):
        """Calculate strategy-specific indicators and return updated dataframe"""
        pass
    
    @abstractmethod
    def generate_signal(self, data):
        """Generate trading signals. Returns (bullish_signal, bearish_signal, signal_data)"""
        pass
    
    def get_historical_statistics(self, data):
        """Return historical statistics for strategy indicators"""
        return {}

# ==================== STRATEGY IMPLEMENTATIONS ====================

class EMACrossoverStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__(params)
        self.ma1_type = params.get('ma1_type', 'EMA')
        self.ma2_type = params.get('ma2_type', 'EMA')
        self.period1 = params.get('period1', 9)
        self.period2 = params.get('period2', 20)
        self.crossover_type = params.get('crossover_type', 'simple')
        self.custom_candle_size = params.get('custom_candle_size', 50)
        self.atr_multiplier = params.get('atr_multiplier', 1.5)
        self.min_angle = params.get('min_angle', 30)
    
    def calculate_ema(self, data, period):
        return data['Close'].ewm(span=period, adjust=False).mean()
    
    def calculate_sma(self, data, period):
        return data['Close'].rolling(window=period).mean()
    
    def calculate_atr(self, data, period=14):
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def calculate_indicators(self, data):
        if self.ma1_type == 'EMA':
            data['MA1'] = self.calculate_ema(data, self.period1)
        else:
            data['MA1'] = self.calculate_sma(data, self.period1)
        
        if self.ma2_type == 'EMA':
            data['MA2'] = self.calculate_ema(data, self.period2)
        else:
            data['MA2'] = self.calculate_sma(data, self.period2)
        
        data['ATR'] = self.calculate_atr(data)
        data['Candle_Body'] = np.abs(data['Close'] - data['Open'])
        return data
    
    def calculate_crossover_angle(self, data, lookback=5):
        """Calculate angle of crossover in degrees"""
        if len(data) < lookback + 1:
            return 0
        
        ma1_slope = (data['MA1'].iloc[-1] - data['MA1'].iloc[-lookback]) / lookback
        ma2_slope = (data['MA2'].iloc[-1] - data['MA2'].iloc[-lookback]) / lookback
        
        angle = np.degrees(np.arctan(np.abs(ma1_slope - ma2_slope)))
        return angle
    
    def generate_signal(self, data):
        if len(data) < max(self.period1, self.period2) + 5:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        
        # Basic crossover detection
        ma1_curr = data['MA1'].iloc[-1]
        ma1_prev = data['MA1'].iloc[-2]
        ma2_curr = data['MA2'].iloc[-1]
        ma2_prev = data['MA2'].iloc[-2]
        
        bullish_cross = (ma1_prev <= ma2_prev) and (ma1_curr > ma2_curr)
        bearish_cross = (ma1_prev >= ma2_prev) and (ma1_curr < ma2_curr)
        
        # Calculate crossover angle
        angle = self.calculate_crossover_angle(data)
        
        # Angle confirmation
        if angle < self.min_angle:
            bullish_cross = False
            bearish_cross = False
        
        # Additional confirmation based on crossover type
        if self.crossover_type != 'simple':
            candle_body = data['Candle_Body'].iloc[-1]
            
            if self.crossover_type == 'auto_strong_candle':
                avg_body = data['Candle_Body'].rolling(20).mean().iloc[-1]
                strong_candle = candle_body > (avg_body * 1.5)
            elif self.crossover_type == 'atr_strong_candle':
                atr = data['ATR'].iloc[-1]
                strong_candle = candle_body > (atr * self.atr_multiplier)
            elif self.crossover_type == 'custom_candle':
                strong_candle = candle_body > self.custom_candle_size
            
            if not strong_candle:
                bullish_cross = False
                bearish_cross = False
        
        signal_data = {
            'MA1': ma1_curr,
            'MA2': ma2_curr,
            'Candle_Body': data['Candle_Body'].iloc[-1],
            'ATR': data['ATR'].iloc[-1],
            'Crossover_Angle': angle
        }
        
        return bullish_cross, bearish_cross, signal_data
    
    def get_historical_statistics(self, data):
        data = self.calculate_indicators(data)
        return {
            'MA1': {
                'Mean': data['MA1'].mean(),
                'Std': data['MA1'].std(),
                'Min': data['MA1'].min(),
                'Max': data['MA1'].max(),
                '25th': data['MA1'].quantile(0.25),
                '75th': data['MA1'].quantile(0.75)
            },
            'MA2': {
                'Mean': data['MA2'].mean(),
                'Std': data['MA2'].std(),
                'Min': data['MA2'].min(),
                'Max': data['MA2'].max(),
                '25th': data['MA2'].quantile(0.25),
                '75th': data['MA2'].quantile(0.75)
            },
            'ATR': {
                'Mean': data['ATR'].mean(),
                'Current': data['ATR'].iloc[-1]
            }
        }

class PairRatioStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__(params)
        self.ticker2 = params.get('ticker2', '')
        self.lookback = params.get('lookback', 20)
        self.threshold = params.get('threshold', 2.0)
    
    def calculate_indicators(self, data, data2=None):
        if data2 is None or len(data2) == 0:
            return data
        
        # Align data lengths
        min_len = min(len(data), len(data2))
        data = data.iloc[-min_len:]
        data2 = data2.iloc[-min_len:]
        
        data['Ratio'] = data['Close'] / data2['Close'].values
        data['Ratio_Mean'] = data['Ratio'].rolling(window=self.lookback).mean()
        data['Ratio_Std'] = data['Ratio'].rolling(window=self.lookback).std()
        data['Z_Score'] = (data['Ratio'] - data['Ratio_Mean']) / data['Ratio_Std']
        
        return data
    
    def generate_signal(self, data, data2=None):
        if data2 is None or len(data) < self.lookback or len(data2) < self.lookback:
            return False, False, {}
        
        data = self.calculate_indicators(data, data2)
        
        z_score = data['Z_Score'].iloc[-1]
        
        bullish_signal = z_score < -self.threshold
        bearish_signal = z_score > self.threshold
        
        signal_data = {
            'Z_Score': z_score,
            'Ratio': data['Ratio'].iloc[-1],
            'Ratio_Mean': data['Ratio_Mean'].iloc[-1],
            'Ratio_Std': data['Ratio_Std'].iloc[-1]
        }
        
        return bullish_signal, bearish_signal, signal_data
    
    def get_historical_statistics(self, data, data2=None):
        if data2 is None:
            return {}
        data = self.calculate_indicators(data, data2)
        return {
            'Z_Score': {
                'Mean': data['Z_Score'].mean(),
                'Std': data['Z_Score'].std(),
                'Min': data['Z_Score'].min(),
                'Max': data['Z_Score'].max(),
                'Current': data['Z_Score'].iloc[-1]
            },
            'Ratio': {
                'Mean': data['Ratio'].mean(),
                'Current': data['Ratio'].iloc[-1]
            }
        }

class RSIDivergenceStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__(params)
        self.period = params.get('rsi_period', 14)
        self.oversold = params.get('oversold', 30)
        self.overbought = params.get('overbought', 70)
        self.lookback = params.get('lookback', 20)
    
    def calculate_rsi(self, data, period):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_indicators(self, data):
        data['RSI'] = self.calculate_rsi(data, self.period)
        return data
    
    def detect_divergence(self, data):
        if len(data) < self.lookback:
            return False, False
        
        recent_data = data.iloc[-self.lookback:]
        
        # Find peaks and troughs
        price_peaks_idx = argrelextrema(recent_data['Close'].values, np.greater, order=3)[0]
        price_troughs_idx = argrelextrema(recent_data['Close'].values, np.less, order=3)[0]
        rsi_peaks_idx = argrelextrema(recent_data['RSI'].values, np.greater, order=3)[0]
        rsi_troughs_idx = argrelextrema(recent_data['RSI'].values, np.less, order=3)[0]
        
        # Bullish divergence: Price lower low, RSI higher low
        bullish_div = False
        if len(price_troughs_idx) >= 2 and len(rsi_troughs_idx) >= 2:
            last_price_trough = recent_data['Close'].iloc[price_troughs_idx[-1]]
            prev_price_trough = recent_data['Close'].iloc[price_troughs_idx[-2]]
            last_rsi_trough = recent_data['RSI'].iloc[rsi_troughs_idx[-1]]
            prev_rsi_trough = recent_data['RSI'].iloc[rsi_troughs_idx[-2]]
            
            if last_price_trough < prev_price_trough and last_rsi_trough > prev_rsi_trough:
                bullish_div = True
        
        # Bearish divergence: Price higher high, RSI lower high
        bearish_div = False
        if len(price_peaks_idx) >= 2 and len(rsi_peaks_idx) >= 2:
            last_price_peak = recent_data['Close'].iloc[price_peaks_idx[-1]]
            prev_price_peak = recent_data['Close'].iloc[price_peaks_idx[-2]]
            last_rsi_peak = recent_data['RSI'].iloc[rsi_peaks_idx[-1]]
            prev_rsi_peak = recent_data['RSI'].iloc[rsi_peaks_idx[-2]]
            
            if last_price_peak > prev_price_peak and last_rsi_peak < prev_rsi_peak:
                bearish_div = True
        
        return bullish_div, bearish_div
    
    def generate_signal(self, data):
        if len(data) < self.period + self.lookback:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        
        rsi_curr = data['RSI'].iloc[-1]
        
        # Regular RSI signals
        rsi_bullish = rsi_curr < self.oversold
        rsi_bearish = rsi_curr > self.overbought
        
        # Divergence signals
        div_bullish, div_bearish = self.detect_divergence(data)
        
        bullish_signal = rsi_bullish or div_bullish
        bearish_signal = rsi_bearish or div_bearish
        
        signal_data = {
            'RSI': rsi_curr,
            'Divergence_Bullish': div_bullish,
            'Divergence_Bearish': div_bearish
        }
        
        return bullish_signal, bearish_signal, signal_data
    
    def get_historical_statistics(self, data):
        data = self.calculate_indicators(data)
        oversold_count = (data['RSI'] < self.oversold).sum()
        overbought_count = (data['RSI'] > self.overbought).sum()
        
        return {
            'RSI': {
                'Mean': data['RSI'].mean(),
                'Std': data['RSI'].std(),
                'Min': data['RSI'].min(),
                'Max': data['RSI'].max(),
                'Current': data['RSI'].iloc[-1],
                'Oversold_Count': oversold_count,
                'Overbought_Count': overbought_count
            }
        }

class FibonacciStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__(params)
        self.lookback = params.get('lookback', 50)
        self.tolerance = params.get('tolerance', 0.005)
        self.key_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    
    def calculate_indicators(self, data):
        if len(data) < self.lookback:
            return data
        
        recent = data.iloc[-self.lookback:]
        swing_high = recent['High'].max()
        swing_low = recent['Low'].min()
        diff = swing_high - swing_low
        
        data['Swing_High'] = swing_high
        data['Swing_Low'] = swing_low
        
        for level in self.key_levels:
            data[f'Fib_{level}'] = swing_high - (diff * level)
        
        return data
    
    def generate_signal(self, data):
        if len(data) < self.lookback:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        
        curr_price = data['Close'].iloc[-1]
        
        bullish_signal = False
        bearish_signal = False
        near_level = None
        
        for level in [0.382, 0.5, 0.618]:
            fib_price = data[f'Fib_{level}'].iloc[-1]
            if abs(curr_price - fib_price) / fib_price <= self.tolerance:
                near_level = level
                # Bullish if approaching from below
                if data['Close'].iloc[-2] < fib_price and curr_price >= fib_price:
                    bullish_signal = True
                # Bearish if approaching from above
                elif data['Close'].iloc[-2] > fib_price and curr_price <= fib_price:
                    bearish_signal = True
                break
        
        signal_data = {
            'Swing_High': data['Swing_High'].iloc[-1],
            'Swing_Low': data['Swing_Low'].iloc[-1],
            'Near_Level': near_level,
            'Fib_0.382': data['Fib_0.382'].iloc[-1],
            'Fib_0.5': data['Fib_0.5'].iloc[-1],
            'Fib_0.618': data['Fib_0.618'].iloc[-1]
        }
        
        return bullish_signal, bearish_signal, signal_data
    
    def get_historical_statistics(self, data):
        data = self.calculate_indicators(data)
        return {
            'Fibonacci': {
                'Swing_High': data['Swing_High'].iloc[-1],
                'Swing_Low': data['Swing_Low'].iloc[-1],
                'Range': data['Swing_High'].iloc[-1] - data['Swing_Low'].iloc[-1]
            }
        }

class ElliottWaveStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__(params)
        self.lookback = params.get('lookback', 100)
        self.order = params.get('order', 5)
    
    def find_waves(self, data):
        if len(data) < self.lookback:
            return []
        
        recent = data.iloc[-self.lookback:]
        prices = recent['Close'].values
        
        # Find local maxima and minima
        maxima_idx = argrelextrema(prices, np.greater, order=self.order)[0]
        minima_idx = argrelextrema(prices, np.less, order=self.order)[0]
        
        # Combine and sort
        extrema_idx = np.sort(np.concatenate([maxima_idx, minima_idx]))
        
        if len(extrema_idx) < 5:
            return []
        
        waves = []
        for i, idx in enumerate(extrema_idx):
            waves.append({
                'idx': idx,
                'price': prices[idx],
                'wave': i + 1,
                'type': 'peak' if idx in maxima_idx else 'trough'
            })
        
        return waves[-5:] if len(waves) >= 5 else []
    
    def calculate_indicators(self, data):
        waves = self.find_waves(data)
        data['Waves'] = None
        return data
    
    def generate_signal(self, data):
        if len(data) < self.lookback:
            return False, False, {}
        
        waves = self.find_waves(data)
        
        if len(waves) < 5:
            return False, False, {'Waves': 'Insufficient data'}
        
        # Check for 5-wave pattern
        wave5_complete = False
        if waves[-1]['wave'] == 5:
            wave5_complete = True
        
        bullish_signal = wave5_complete and waves[-1]['type'] == 'trough'
        bearish_signal = wave5_complete and waves[-1]['type'] == 'peak'
        
        signal_data = {
            'Waves_Detected': len(waves),
            'Wave5_Complete': wave5_complete,
            'Last_Wave_Type': waves[-1]['type'] if waves else None,
            'Wave_Prices': [w['price'] for w in waves]
        }
        
        return bullish_signal, bearish_signal, signal_data
    
    def get_historical_statistics(self, data):
        waves = self.find_waves(data)
        return {
            'Elliott_Waves': {
                'Waves_Detected': len(waves),
                'Wave_Prices': [w['price'] for w in waves] if waves else []
            }
        }

class ZScoreMeanReversionStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__(params)
        self.lookback = params.get('lookback', 20)
        self.threshold = params.get('threshold', 2.0)
    
    def calculate_indicators(self, data):
        data['Price_Mean'] = data['Close'].rolling(window=self.lookback).mean()
        data['Price_Std'] = data['Close'].rolling(window=self.lookback).std()
        data['Z_Score'] = (data['Close'] - data['Price_Mean']) / data['Price_Std']
        return data
    
    def generate_signal(self, data):
        if len(data) < self.lookback:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        
        z_score = data['Z_Score'].iloc[-1]
        
        bullish_signal = z_score < -self.threshold
        bearish_signal = z_score > self.threshold
        
        signal_data = {
            'Z_Score': z_score,
            'Price_Mean': data['Price_Mean'].iloc[-1],
            'Price_Std': data['Price_Std'].iloc[-1]
        }
        
        return bullish_signal, bearish_signal, signal_data
    
    def get_historical_statistics(self, data):
        data = self.calculate_indicators(data)
        return {
            'Z_Score': {
                'Mean': data['Z_Score'].mean(),
                'Std': data['Z_Score'].std(),
                'Min': data['Z_Score'].min(),
                'Max': data['Z_Score'].max(),
                'Current': data['Z_Score'].iloc[-1]
            }
        }

class BreakoutVolumeStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__(params)
        self.lookback = params.get('lookback', 20)
        self.volume_multiplier = params.get('volume_multiplier', 1.5)
    
    def calculate_indicators(self, data):
        data['Upper_Band'] = data['High'].rolling(window=self.lookback).max()
        data['Lower_Band'] = data['Low'].rolling(window=self.lookback).min()
        if 'Volume' in data.columns:
            data['Avg_Volume'] = data['Volume'].rolling(window=self.lookback).mean()
        return data
    
    def generate_signal(self, data):
        if len(data) < self.lookback:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        
        curr_close = data['Close'].iloc[-1]
        upper_band = data['Upper_Band'].iloc[-2]  # Previous period's band
        lower_band = data['Lower_Band'].iloc[-2]
        
        volume_surge = False
        if 'Volume' in data.columns and 'Avg_Volume' in data.columns:
            curr_vol = data['Volume'].iloc[-1]
            avg_vol = data['Avg_Volume'].iloc[-1]
            volume_surge = curr_vol > (avg_vol * self.volume_multiplier)
        
        bullish_signal = (curr_close > upper_band) and volume_surge
        bearish_signal = (curr_close < lower_band) and volume_surge
        
        signal_data = {
            'Upper_Band': upper_band,
            'Lower_Band': lower_band,
            'Volume_Surge': volume_surge,
            'Current_Volume': data['Volume'].iloc[-1] if 'Volume' in data.columns else 0,
            'Avg_Volume': data['Avg_Volume'].iloc[-1] if 'Avg_Volume' in data.columns else 0
        }
        
        return bullish_signal, bearish_signal, signal_data
    
    def get_historical_statistics(self, data):
        data = self.calculate_indicators(data)
        stats = {
            'Donchian': {
                'Upper_Band': data['Upper_Band'].iloc[-1],
                'Lower_Band': data['Lower_Band'].iloc[-1],
                'Range': data['Upper_Band'].iloc[-1] - data['Lower_Band'].iloc[-1]
            }
        }
        if 'Volume' in data.columns:
            stats['Volume'] = {
                'Mean': data['Volume'].mean(),
                'Current': data['Volume'].iloc[-1]
            }
        return stats

class SimpleBuyStrategy(BaseStrategy):
    def generate_signal(self, data):
        return True, False, {'Signal': 'Always Buy'}
    
    def calculate_indicators(self, data):
        return data

class SimpleSellStrategy(BaseStrategy):
    def generate_signal(self, data):
        return False, True, {'Signal': 'Always Sell'}
    
    def calculate_indicators(self, data):
        return data

# ==================== TRADING SYSTEM ====================

class TradingSystem:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
    
    def fetch_data(self, ticker, interval, period):
        """Fetch data with proper error handling and rate limiting"""
        try:
            time.sleep(1.5)  # Rate limiting
            
            # Handle yfinance compatibility
            valid_combinations = {
                '1m': ['1d', '5d'],
                '5m': ['1d', '5d', '1mo'],
                '15m': ['1d', '5d', '1mo'],
                '30m': ['1d', '5d', '1mo'],
                '1h': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
                '1d': ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', '20y'],
                '1mo': ['1y', '2y', '5y', '10y', '20y', '25y', '30y']
            }
            
            if interval in valid_combinations and period not in valid_combinations[interval]:
                period = valid_combinations[interval][0]
            
            data = yf.download(ticker, interval=interval, period=period, progress=False)
            
            if data.empty:
                return None
            
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # Convert index to IST
            data.index = pd.to_datetime(data.index)
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            data.index = data.index.tz_convert(self.ist)
            
            return data
        
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def calculate_stop_loss(self, entry_price, position_type, sl_type, sl_value, data):
        """Calculate stop loss based on type"""
        if sl_type == 'Custom Points':
            if position_type == 'LONG':
                return entry_price - sl_value
            else:
                return entry_price + sl_value
        
        elif sl_type == 'P&L Based':
            if position_type == 'LONG':
                return entry_price - sl_value
            else:
                return entry_price + sl_value
        
        return entry_price
    
    def calculate_target(self, entry_price, position_type, target_type, target_value, sl_distance=None):
        """Calculate target based on type"""
        if target_type == 'Custom Points':
            if position_type == 'LONG':
                return entry_price + target_value
            else:
                return entry_price - target_value
        
        elif target_type == 'Risk Reward Ratio':
            if sl_distance:
                if position_type == 'LONG':
                    return entry_price + (sl_distance * target_value)
                else:
                    return entry_price - (sl_distance * target_value)
        
        elif target_type == 'Percentage Based':
            pct = target_value / 100
            if position_type == 'LONG':
                return entry_price * (1 + pct)
            else:
                return entry_price * (1 - pct)
        
        return entry_price
    
    def calculate_atr(self, data, period=14):
        """Calculate ATR"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean().iloc[-1]
    
    def find_previous_swing(self, data, position_type):
        """Find previous swing high/low"""
        if position_type == 'LONG':
            return data['Low'].iloc[-20:].min()
        else:
            return data['High'].iloc[-20:].max()
    
    def update_trailing_sl(self, position, current_price, trail_points):
        """Update trailing stop loss"""
        if position['type'] == 'LONG':
            new_sl = current_price - trail_points
            if new_sl > position['sl']:
                position['sl'] = new_sl
                log_event(f"Trailing SL updated to {new_sl:.2f}")
                return True
        else:
            new_sl = current_price + trail_points
            if new_sl < position['sl']:
                position['sl'] = new_sl
                log_event(f"Trailing SL updated to {new_sl:.2f}")
                return True
        return False
    
    def update_trailing_target(self, position, current_price, trail_points, tolerance=0.02):
        """Update trailing target with tolerance for volatility"""
        if position['type'] == 'LONG':
            # Check if we're within tolerance of target
            distance_to_target = position['target'] - current_price
            if distance_to_target <= (position['target'] * tolerance):
                new_target = current_price + trail_points
                if new_target > position['target']:
                    position['target'] = new_target
                    log_event(f"Trailing Target updated to {new_target:.2f}")
                    return True
        else:
            distance_to_target = current_price - position['target']
            if distance_to_target <= (position['target'] * tolerance):
                new_target = current_price - trail_points
                if new_target < position['target']:
                    position['target'] = new_target
                    log_event(f"Trailing Target updated to {new_target:.2f}")
                    return True
        return False
    
    def check_exit_conditions(self, position, current_price, strategy, data, data2=None):
        """Check if any exit conditions are met"""
        # Stop Loss
        if position['type'] == 'LONG':
            if current_price <= position['sl']:
                return True, 'Stop Loss Hit', current_price
        else:
            if current_price >= position['sl']:
                return True, 'Stop Loss Hit', current_price
        
        # Target
        if position['type'] == 'LONG':
            if current_price >= position['target']:
                return True, 'Target Hit', current_price
        else:
            if current_price <= position['target']:
                return True, 'Target Hit', current_price
        
        # Signal Based Exit
        if position.get('signal_based_exit', False):
            if strategy:
                if data2 is not None:
                    bullish, bearish, _ = strategy.generate_signal(data, data2)
                else:
                    bullish, bearish, _ = strategy.generate_signal(data)
                
                if position['type'] == 'LONG' and bearish:
                    return True, 'Opposite Signal', current_price
                elif position['type'] == 'SHORT' and bullish:
                    return True, 'Opposite Signal', current_price
        
        return False, None, None
    
    def get_market_status(self, position, current_price):
        """Get market movement status text"""
        if not position:
            return "Waiting for entry signal..."
        
        entry = position['entry_price']
        pnl_pct = position['pnl_pct']
        
        if position['type'] == 'LONG':
            if pnl_pct > 0.5:
                return f"🟢 Strong upward momentum! +{pnl_pct:.2f}% profit"
            elif pnl_pct > 0:
                return f"📈 Moving gradually in your favor (+{pnl_pct:.2f}%)"
            else:
                return f"🔴 Price moving against position ({pnl_pct:.2f}%). Monitoring for reversal or SL"
        else:
            if pnl_pct > 0.5:
                return f"🟢 Strong downward momentum! +{pnl_pct:.2f}% profit"
            elif pnl_pct > 0:
                return f"📉 Moving gradually in your favor (+{pnl_pct:.2f}%)"
            else:
                return f"🔴 Price moving against position ({pnl_pct:.2f}%). Monitoring for reversal or SL"
    
    def get_trade_guidance(self, position, current_price, strategy_name, signal_data):
        """Provide real-time trade guidance (100 words)"""
        if not position:
            return ""
        
        pnl_pct = position['pnl_pct']
        position_type = position['type']
        
        guidance = f"**Current Trade Status ({strategy_name})**\n\n"
        
        if pnl_pct > 1:
            guidance += f"✅ Excellent! Trade moving as expected with +{pnl_pct:.2f}% profit. "
            guidance += f"{'Trailing SL is active - let winners run.' if position.get('trail_sl') else 'Consider booking partial profits or activating trailing SL.'} "
            guidance += "Stay disciplined, don't exit prematurely due to minor pullbacks. "
        elif pnl_pct > 0:
            guidance += f"✓ Trade in profit (+{pnl_pct:.2f}%). Price moving favorably but slowly. "
            guidance += "Patience is key - avoid premature exits. Monitor for acceleration or reversal signals. "
            guidance += "Don't move SL away from entry in fear. "
        else:
            guidance += f"⚠ Trade in loss ({pnl_pct:.2f}%). This is normal market behavior. "
            guidance += "Trust your SL placement - don't panic exit or move SL further. "
            guidance += "Let the strategy play out. If SL hits, it's protecting your capital as designed. "
        
        guidance += f"Strategy expects: {position_type} movement based on {strategy_name} signals."
        
        return guidance
    
    def analyze_trade_performance(self, trade):
        """AI-powered trade analysis"""
        duration_minutes = (trade['exit_time'] - trade['entry_time']).total_seconds() / 60
        
        analysis = f"**Trade Performance Analysis**\n\n"
        
        # Performance assessment
        if trade['pnl_pct'] > 2:
            analysis += "✅ **Excellent Performance**: Strong profitable trade with good risk-reward execution.\n\n"
        elif trade['pnl_pct'] > 0:
            analysis += "✓ **Profitable**: Trade closed in profit, though there may be room for optimization.\n\n"
        else:
            analysis += "⚠ **Loss Trade**: Trade closed at a loss. This is part of trading - focus on learning.\n\n"
        
        # Exit quality
        if trade['exit_reason'] == 'Target Hit':
            analysis += "🎯 **Exit Quality**: Perfect - target achieved as planned.\n\n"
        elif trade['exit_reason'] == 'Stop Loss Hit':
            analysis += "🛡️ **Exit Quality**: SL protected capital. This is proper risk management.\n\n"
        elif trade['exit_reason'] == 'Opposite Signal':
            analysis += "🔄 **Exit Quality**: Strategy signaled reversal - good disciplined exit.\n\n"
        else:
            analysis += "📊 **Exit Quality**: Manual exit - ensure it was based on valid reasoning.\n\n"
        
        # Duration insights
        if duration_minutes < 5:
            analysis += f"⏱️ **Duration**: Very short ({duration_minutes:.1f}m). Consider if entry was premature.\n\n"
        elif duration_minutes > 240:
            analysis += f"⏱️ **Duration**: Extended hold ({duration_minutes/60:.1f}h). Review if patience was warranted.\n\n"
        else:
            analysis += f"⏱️ **Duration**: {duration_minutes:.1f} minutes - reasonable hold time.\n\n"
        
        # Recommendations
        analysis += "**Recommendations**:\n"
        if trade['pnl_pct'] < 0 and trade['exit_reason'] == 'Stop Loss Hit':
            analysis += "- SL placement was correct - it prevented larger losses\n"
            analysis += "- Review entry timing - was signal strong enough?\n"
            analysis += "- Check if market conditions suited the strategy\n"
        elif trade['pnl_pct'] > 0:
            analysis += "- Good execution - repeat this discipline\n"
            analysis += "- Consider if trailing SL could have captured more profit\n"
            analysis += "- Document what worked for future reference\n"
        
        return analysis

# ==================== STREAMLIT APP ====================

def initialize_session_state():
    """Initialize session state variables"""
    if 'trading_active' not in st.session_state:
        st.session_state.trading_active = False
    if 'current_position' not in st.session_state:
        st.session_state.current_position = None
    if 'trade_history' not in st.session_state:
        st.session_state.trade_history = []
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    if 'iteration_count' not in st.session_state:
        st.session_state.iteration_count = 0
    if 'last_data' not in st.session_state:
        st.session_state.last_data = None
    if 'last_data2' not in st.session_state:
        st.session_state.last_data2 = None

def create_chart(data, strategy, position=None, signal_data=None):
    """Create interactive chart with indicators"""
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    # Add strategy-specific indicators
    if isinstance(strategy, EMACrossoverStrategy):
        if 'MA1' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['MA1'], name=f'{strategy.ma1_type}{strategy.period1}', line=dict(color='blue')))
        if 'MA2' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['MA2'], name=f'{strategy.ma2_type}{strategy.period2}', line=dict(color='red')))
    
    elif isinstance(strategy, FibonacciStrategy):
        for level in strategy.key_levels:
            if f'Fib_{level}' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, 
                    y=data[f'Fib_{level}'], 
                    name=f'Fib {level}',
                    line=dict(dash='dash')
                ))
    
    elif isinstance(strategy, BreakoutVolumeStrategy):
        if 'Upper_Band' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['Upper_Band'], name='Upper Band', line=dict(color='green', dash='dash')))
        if 'Lower_Band' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['Lower_Band'], name='Lower Band', line=dict(color='red', dash='dash')))
    
    # Mark entry point
    if position:
        entry_idx = position.get('entry_time')
        if entry_idx:
            marker_color = 'green' if position['type'] == 'LONG' else 'red'
            marker_symbol = 'triangle-up' if position['type'] == 'LONG' else 'triangle-down'
            fig.add_trace(go.Scatter(
                x=[entry_idx],
                y=[position['entry_price']],
                mode='markers',
                marker=dict(size=15, color=marker_color, symbol=marker_symbol),
                name=f'{position["type"]} Entry'
            ))
        
        # SL and Target lines
        fig.add_hline(y=position['sl'], line_dash="dot", line_color="red", annotation_text="Stop Loss")
        fig.add_hline(y=position['target'], line_dash="dot", line_color="green", annotation_text="Target")
    
    fig.update_layout(
        title='Live Trading Chart',
        yaxis_title='Price',
        xaxis_title='Time (IST)',
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def main():
    initialize_session_state()
    
    st.markdown('<div class="main-header">🚀 Professional Algorithmic Trading System</div>', unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Trading Configuration")
    
    # Asset Selection
    asset_category = st.sidebar.selectbox(
        "Asset Category",
        ["Preset Assets", "Custom Ticker", "Indian Stock"]
    )
    
    if asset_category == "Preset Assets":
        ticker1 = st.sidebar.selectbox(
            "Select Asset",
            ["^NSEI (NIFTY 50)", "^NSEBANK (Bank NIFTY)", "^BSESN (SENSEX)", 
             "BTC-USD", "ETH-USD", "GC=F (Gold)", "SI=F (Silver)", 
             "USDINR=X", "EURUSD=X", "GBPUSD=X"]
        )
        ticker1 = ticker1.split(" ")[0]
    elif asset_category == "Custom Ticker":
        ticker1 = st.sidebar.text_input("Enter Ticker Symbol", "AAPL")
    else:
        stock_name = st.sidebar.text_input("Enter Stock Name", "RELIANCE")
        ticker1 = f"{stock_name}.NS"
    
    # Timeframe and Period
    col1, col2 = st.sidebar.columns(2)
    with col1:
        interval = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1wk", "1mo"])
    with col2:
        period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"])
    
    # Strategy Selection
    st.sidebar.subheader("📊 Strategy Selection")
    strategy_name = st.sidebar.selectbox(
        "Choose Strategy",
        ["EMA/SMA Crossover", "Pair Ratio Trading", "RSI + Divergence", 
         "Fibonacci Retracement", "Elliott Wave", "Z-Score Mean Reversion",
         "Breakout + Volume", "Simple Buy", "Simple Sell"]
    )
    
    # Strategy Parameters
    strategy_params = {}
    ticker2 = None
    
    if strategy_name == "EMA/SMA Crossover":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            strategy_params['ma1_type'] = st.selectbox("MA1 Type", ["EMA", "SMA"])
            strategy_params['period1'] = st.number_input("MA1 Period", value=9, min_value=1)
        with col2:
            strategy_params['ma2_type'] = st.selectbox("MA2 Type", ["EMA", "SMA"])
            strategy_params['period2'] = st.number_input("MA2 Period", value=20, min_value=1)
        
        strategy_params['crossover_type'] = st.sidebar.selectbox(
            "Crossover Type",
            ["simple", "auto_strong_candle", "atr_strong_candle", "custom_candle"]
        )
        
        if strategy_params['crossover_type'] == 'atr_strong_candle':
            strategy_params['atr_multiplier'] = st.sidebar.number_input("ATR Multiplier", value=1.5, min_value=0.1, step=0.1)
        elif strategy_params['crossover_type'] == 'custom_candle':
            strategy_params['custom_candle_size'] = st.sidebar.number_input("Custom Candle Size (points)", value=50, min_value=1)
        
        strategy_params['min_angle'] = st.sidebar.number_input("Min Crossover Angle (degrees)", value=30, min_value=0, max_value=90)
        
        strategy = EMACrossoverStrategy(strategy_params)
    
    elif strategy_name == "Pair Ratio Trading":
        asset_cat2 = st.sidebar.selectbox("Ticker2 Category", ["Preset Assets", "Custom Ticker", "Indian Stock"])
        if asset_cat2 == "Preset Assets":
            ticker2_select = st.sidebar.selectbox("Select Ticker2", ["^NSEI (NIFTY 50)", "^NSEBANK (Bank NIFTY)", "BTC-USD", "ETH-USD"])
            ticker2 = ticker2_select.split(" ")[0]
        elif asset_cat2 == "Custom Ticker":
            ticker2 = st.sidebar.text_input("Enter Ticker2 Symbol", "MSFT")
        else:
            stock2 = st.sidebar.text_input("Enter Ticker2 Stock", "TCS")
            ticker2 = f"{stock2}.NS"
        
        strategy_params['ticker2'] = ticker2
        strategy_params['lookback'] = st.sidebar.number_input("Lookback Period", value=20, min_value=5)
        strategy_params['threshold'] = st.sidebar.number_input("Z-Score Threshold", value=2.0, min_value=0.5, step=0.1)
        strategy = PairRatioStrategy(strategy_params)
    
    elif strategy_name == "RSI + Divergence":
        strategy_params['rsi_period'] = st.sidebar.number_input("RSI Period", value=14, min_value=2)
        strategy_params['oversold'] = st.sidebar.number_input("Oversold Level", value=30, min_value=1, max_value=50)
        strategy_params['overbought'] = st.sidebar.number_input("Overbought Level", value=70, min_value=50, max_value=99)
        strategy_params['lookback'] = st.sidebar.number_input("Divergence Lookback", value=20, min_value=10)
        strategy = RSIDivergenceStrategy(strategy_params)
    
    elif strategy_name == "Fibonacci Retracement":
        strategy_params['lookback'] = st.sidebar.number_input("Swing Lookback", value=50, min_value=20)
        strategy_params['tolerance'] = st.sidebar.number_input("Level Tolerance (%)", value=0.5, min_value=0.1, step=0.1) / 100
        strategy = FibonacciStrategy(strategy_params)
    
    elif strategy_name == "Elliott Wave":
        strategy_params['lookback'] = st.sidebar.number_input("Wave Lookback", value=100, min_value=50)
        strategy_params['order'] = st.sidebar.number_input("Extrema Order", value=5, min_value=3)
        strategy = ElliottWaveStrategy(strategy_params)
    
    elif strategy_name == "Z-Score Mean Reversion":
        strategy_params['lookback'] = st.sidebar.number_input("Lookback Period", value=20, min_value=5)
        strategy_params['threshold'] = st.sidebar.number_input("Z-Score Threshold", value=2.0, min_value=0.5, step=0.1)
        strategy = ZScoreMeanReversionStrategy(strategy_params)
    
    elif strategy_name == "Breakout + Volume":
        strategy_params['lookback'] = st.sidebar.number_input("Channel Lookback", value=20, min_value=5)
        strategy_params['volume_multiplier'] = st.sidebar.number_input("Volume Multiplier", value=1.5, min_value=1.0, step=0.1)
        strategy = BreakoutVolumeStrategy(strategy_params)
    
    elif strategy_name == "Simple Buy":
        strategy = SimpleBuyStrategy()
    
    elif strategy_name == "Simple Sell":
        strategy = SimpleSellStrategy()
    
    # Risk Management
    st.sidebar.subheader("🛡️ Risk Management")
    
    sl_type = st.sidebar.selectbox(
        "Stop Loss Type",
        ["Custom Points", "Trail SL", "Signal Based", "ATR Based", 
         "Previous Swing", "Current Swing", "Current Candle", 
         "Percentage Based", "P&L Based"]
    )
    
    sl_value = 0
    if sl_type in ["Custom Points", "P&L Based"]:
        sl_value = st.sidebar.number_input("SL Points", value=50.0, min_value=1.0)
    elif sl_type == "Trail SL":
        sl_value = st.sidebar.number_input("Trail SL Points", value=30.0, min_value=1.0)
    elif sl_type == "ATR Based":
        sl_value = st.sidebar.number_input("ATR Multiplier", value=2.0, min_value=0.5, step=0.1)
    elif sl_type == "Percentage Based":
        sl_value = st.sidebar.number_input("SL Percentage", value=2.0, min_value=0.1, step=0.1)
    
    target_type = st.sidebar.selectbox(
        "Target Type",
        ["Custom Points", "Trail Target", "Signal Based", "Risk Reward Ratio", "Percentage Based"]
    )
    
    target_value = 0
    if target_type in ["Custom Points"]:
        target_value = st.sidebar.number_input("Target Points", value=100.0, min_value=1.0)
    elif target_type == "Trail Target":
        target_value = st.sidebar.number_input("Trail Target Points", value=50.0, min_value=1.0)
    elif target_type == "Risk Reward Ratio":
        target_value = st.sidebar.number_input("Risk:Reward Ratio", value=2.0, min_value=0.5, step=0.1)
    elif target_type == "Percentage Based":
        target_value = st.sidebar.number_input("Target Percentage", value=3.0, min_value=0.1, step=0.1)
    
    quantity = st.sidebar.number_input("Position Size (Quantity)", value=1, min_value=1)
    
    # Initialize Trading System
    trading_system = TradingSystem()
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Live Trading", "📊 Trade History", "📝 Trade Log"])
    
    with tab1:
        # Control Buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🟢 Start Trading", disabled=st.session_state.trading_active):
                st.session_state.trading_active = True
                st.session_state.iteration_count = 0
                log_event(f"Trading started: {strategy_name} on {ticker1} ({interval}/{period})")
                st.rerun()
        
        with col2:
            if st.button("🔴 Stop Trading", disabled=not st.session_state.trading_active):
                st.session_state.trading_active = False
                if st.session_state.current_position:
                    # Force close position
                    data = st.session_state.last_data
                    if data is not None and len(data) > 0:
                        exit_price = data['Close'].iloc[-1]
                        position = st.session_state.current_position
                        pnl_points = (exit_price - position['entry_price']) * (1 if position['type'] == 'LONG' else -1) * quantity
                        pnl_pct = ((exit_price - position['entry_price']) / position['entry_price'] * 100) * (1 if position['type'] == 'LONG' else -1)
                        
                        trade_record = {
                            'strategy': strategy_name,
                            'ticker': ticker1,
                            'type': position['type'],
                            'entry_price': position['entry_price'],
                            'entry_time': position['entry_time'],
                            'exit_price': exit_price,
                            'exit_time': datetime.now(IST),
                            'exit_reason': 'Manual Stop',
                            'pnl_points': pnl_points,
                            'pnl_pct': pnl_pct,
                            'quantity': quantity,
                            'sl_type': sl_type,
                            'target_type': target_type
                        }
                        
                        st.session_state.trade_history.append(trade_record)
                        log_event(f"Position force closed: {position['type']} @ {exit_price:.2f}, P&L: {pnl_pct:.2f}%")
                        st.session_state.current_position = None
                
                log_event("Trading stopped")
                st.rerun()
        
        with col3:
            if st.button("❌ Force Close Position", disabled=st.session_state.trading_active or not st.session_state.current_position):
                if st.session_state.current_position and st.session_state.last_data is not None:
                    data = st.session_state.last_data
                    exit_price = data['Close'].iloc[-1]
                    position = st.session_state.current_position
                    pnl_points = (exit_price - position['entry_price']) * (1 if position['type'] == 'LONG' else -1) * quantity
                    pnl_pct = ((exit_price - position['entry_price']) / position['entry_price'] * 100) * (1 if position['type'] == 'LONG' else -1)
                    
                    trade_record = {
                        'strategy': strategy_name,
                        'ticker': ticker1,
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'entry_time': position['entry_time'],
                        'exit_price': exit_price,
                        'exit_time': datetime.now(IST),
                        'exit_reason': 'Force Close',
                        'pnl_points': pnl_points,
                        'pnl_pct': pnl_pct,
                        'quantity': quantity,
                        'sl_type': sl_type,
                        'target_type': target_type
                    }
                    
                    st.session_state.trade_history.append(trade_record)
                    log_event(f"Position force closed: {position['type']} @ {exit_price:.2f}, P&L: {pnl_pct:.2f}%")
                    st.session_state.current_position = None
                    st.rerun()
        
        # Live Trading Logic
        if st.session_state.trading_active:
            st.session_state.iteration_count += 1
            
            # Status Header
            st.markdown(f"<div class='status-active'>🔴 LIVE - Auto-refreshing every 1.5-2s | Iteration: {st.session_state.iteration_count}</div>", unsafe_allow_html=True)
            
            # Fetch data
            data = trading_system.fetch_data(ticker1, interval, period)
            st.session_state.last_data = data
            
            data2 = None
            if ticker2:
                data2 = trading_system.fetch_data(ticker2, interval, period)
                st.session_state.last_data2 = data2
            
            if data is None or len(data) < 20:
                st.error("Insufficient data. Please check ticker symbol and try again.")
                st.session_state.trading_active = False
                st.stop()
            
            current_price = data['Close'].iloc[-1]
            
            # Calculate indicators
            data = strategy.calculate_indicators(data) if not isinstance(strategy, PairRatioStrategy) else strategy.calculate_indicators(data, data2)
            
            # Generate signals
            if isinstance(strategy, PairRatioStrategy):
                bullish_signal, bearish_signal, signal_data = strategy.generate_signal(data, data2)
            else:
                bullish_signal, bearish_signal, signal_data = strategy.generate_signal(data)
            
            # Position Management
            position = st.session_state.current_position
            
            if position is None:
                # Look for entry
                if bullish_signal:
                    entry_price = current_price
                    sl_price = trading_system.calculate_stop_loss(entry_price, 'LONG', sl_type, sl_value, data)
                    sl_distance = abs(entry_price - sl_price)
                    target_price = trading_system.calculate_target(entry_price, 'LONG', target_type, target_value, sl_distance)
                    
                    st.session_state.current_position = {
                        'type': 'LONG',
                        'entry_price': entry_price,
                        'entry_time': datetime.now(IST),
                        'sl': sl_price,
                        'target': target_price,
                        'quantity': quantity,
                        'trail_sl': sl_type == 'Trail SL',
                        'trail_target': target_type == 'Trail Target',
                        'signal_based_exit': sl_type == 'Signal Based' or target_type == 'Signal Based',
                        'pnl_points': 0,
                        'pnl_pct': 0
                    }
                    log_event(f"LONG Entry @ {entry_price:.2f}, SL: {sl_price:.2f}, Target: {target_price:.2f}")
                    st.success(f"✅ LONG position entered @ {entry_price:.2f}")
                
                elif bearish_signal:
                    entry_price = current_price
                    sl_price = trading_system.calculate_stop_loss(entry_price, 'SHORT', sl_type, sl_value, data)
                    sl_distance = abs(entry_price - sl_price)
                    target_price = trading_system.calculate_target(entry_price, 'SHORT', target_type, target_value, sl_distance)
                    
                    st.session_state.current_position = {
                        'type': 'SHORT',
                        'entry_price': entry_price,
                        'entry_time': datetime.now(IST),
                        'sl': sl_price,
                        'target': target_price,
                        'quantity': quantity,
                        'trail_sl': sl_type == 'Trail SL',
                        'trail_target': target_type == 'Trail Target',
                        'signal_based_exit': sl_type == 'Signal Based' or target_type == 'Signal Based',
                        'pnl_points': 0,
                        'pnl_pct': 0
                    }
                    log_event(f"SHORT Entry @ {entry_price:.2f}, SL: {sl_price:.2f}, Target: {target_price:.2f}")
                    st.success(f"✅ SHORT position entered @ {entry_price:.2f}")
            
            else:
                # Update P&L
                pnl_points = (current_price - position['entry_price']) * (1 if position['type'] == 'LONG' else -1)
                pnl_pct = ((current_price - position['entry_price']) / position['entry_price'] * 100) * (1 if position['type'] == 'LONG' else -1)
                position['pnl_points'] = pnl_points
                position['pnl_pct'] = pnl_pct
                
                # Update trailing SL
                if position['trail_sl']:
                    trading_system.update_trailing_sl(position, current_price, sl_value)
                
                # Update trailing target
                if position['trail_target']:
                    trading_system.update_trailing_target(position, current_price, target_value)
                
                # Check exit conditions
                should_exit, exit_reason, exit_price = trading_system.check_exit_conditions(
                    position, current_price, strategy, data, data2
                )
                
                if should_exit:
                    trade_record = {
                        'strategy': strategy_name,
                        'ticker': ticker1,
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'entry_time': position['entry_time'],
                        'exit_price': exit_price,
                        'exit_time': datetime.now(IST),
                        'exit_reason': exit_reason,
                        'pnl_points': pnl_points * quantity,
                        'pnl_pct': pnl_pct,
                        'quantity': quantity,
                        'sl_type': sl_type,
                        'target_type': target_type
                    }
                    
                    st.session_state.trade_history.append(trade_record)
                    log_event(f"Position closed: {position['type']} @ {exit_price:.2f}, Reason: {exit_reason}, P&L: {pnl_pct:.2f}%")
                    
                    if pnl_pct > 0:
                        st.success(f"✅ Position closed with profit! P&L: +{pnl_pct:.2f}% ({exit_reason})")
                    else:
                        st.warning(f"⚠ Position closed at loss. P&L: {pnl_pct:.2f}% ({exit_reason})")
                    
                    # Show AI analysis
                    st.markdown("---")
                    st.markdown(trading_system.analyze_trade_performance(trade_record))
                    
                    st.session_state.current_position = None
            
            # Display Current Status
            st.markdown("---")
            st.subheader(f"📊 {strategy_name} | {ticker1} | {interval}/{period}")
            st.caption(f"Analyzing last {len(data)} candles")
            
            # Live Metrics
            position = st.session_state.current_position
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"{current_price:.2f}")
            
            with col2:
                if position:
                    pnl_color = 'profit' if position['pnl_pct'] > 0 else 'loss'
                    st.markdown(f"<div class='metric-card'><b>P&L Points</b><br><span class='{pnl_color}'>{position['pnl_points']:.2f}</span></div>", unsafe_allow_html=True)
                else:
                    st.metric("Position", "No Position")
            
            with col3:
                if position:
                    pnl_color = 'profit' if position['pnl_pct'] > 0 else 'loss'
                    st.markdown(f"<div class='metric-card'><b>P&L %</b><br><span class='{pnl_color}'>{position['pnl_pct']:.2f}%</span></div>", unsafe_allow_html=True)
                else:
                    st.metric("Status", "Waiting for Signal")
            
            with col4:
                if position:
                    st.metric("Position Type", position['type'])
                else:
                    st.metric("Quantity", quantity)
            
            # Strategy Parameters
            st.markdown("### 📈 Current Strategy Parameters")
            param_cols = st.columns(4)
            
            if signal_data:
                items = list(signal_data.items())
                for idx, (key, value) in enumerate(items):
                    with param_cols[idx % 4]:
                        if isinstance(value, (int, float)):
                            st.metric(key, f"{value:.4f}")
                        else:
                            st.metric(key, str(value))
            
            # Position Details
            if position:
                st.markdown("### 💼 Active Position Details")
                pos_col1, pos_col2, pos_col3, pos_col4, pos_col5 = st.columns(5)
                
                with pos_col1:
                    st.metric("Entry Price", f"{position['entry_price']:.2f}")
                with pos_col2:
                    st.metric("Stop Loss", f"{position['sl']:.2f}")
                with pos_col3:
                    st.metric("Target", f"{position['target']:.2f}")
                with pos_col4:
                    dist_to_sl = abs(current_price - position['sl'])
                    st.metric("Distance to SL", f"{dist_to_sl:.2f}")
                with pos_col5:
                    dist_to_target = abs(current_price - position['target'])
                    st.metric("Distance to Target", f"{dist_to_target:.2f}")
                
                # Market Status
                market_status = trading_system.get_market_status(position, current_price)
                st.info(market_status)
                
                # Trade Guidance
                guidance = trading_system.get_trade_guidance(position, current_price, strategy_name, signal_data)
                st.markdown(f"<div class='trade-guidance'>{guidance}</div>", unsafe_allow_html=True)
            
            # Historical Statistics
            st.markdown("### 📊 Historical Statistics")
            if isinstance(strategy, PairRatioStrategy):
                hist_stats = strategy.get_historical_statistics(data, data2)
            else:
                hist_stats = strategy.get_historical_statistics(data)
            
            if hist_stats:
                for category, stats in hist_stats.items():
                    st.markdown(f"**{category}**")
                    stat_cols = st.columns(len(stats))
                    for idx, (stat_name, stat_value) in enumerate(stats.items()):
                        with stat_cols[idx]:
                            if isinstance(stat_value, (int, float)):
                                st.metric(stat_name, f"{stat_value:.4f}")
                            elif isinstance(stat_value, list):
                                st.metric(stat_name, f"{len(stat_value)} items")
                            else:
                                st.metric(stat_name, str(stat_value))
            
            # Chart
            st.markdown("### 📈 Live Chart")
            chart = create_chart(data, strategy, position, signal_data)
            st.plotly_chart(chart, use_container_width=True)
            
            # Auto-refresh
            time.sleep(1.5)
            st.rerun()
        
        else:
            # Preview Mode (Trading Stopped)
            st.info("🔵 Trading is stopped. Configure settings and click 'Start Trading' to begin.")
            
            if st.button("🔄 Fetch Data Preview"):
                data = trading_system.fetch_data(ticker1, interval, period)
                st.session_state.last_data = data
                
                data2 = None
                if ticker2:
                    data2 = trading_system.fetch_data(ticker2, interval, period)
                    st.session_state.last_data2 = data2
                
                if data is not None and len(data) > 0:
                    st.success(f"✅ Data fetched successfully! {len(data)} candles loaded.")
                    
                    # Calculate indicators
                    data = strategy.calculate_indicators(data) if not isinstance(strategy, PairRatioStrategy) else strategy.calculate_indicators(data, data2)
                    
                    # Generate signals
                    if isinstance(strategy, PairRatioStrategy):
                        bullish_signal, bearish_signal, signal_data = strategy.generate_signal(data, data2)
                    else:
                        bullish_signal, bearish_signal, signal_data = strategy.generate_signal(data)
                    
                    # Display current parameters
                    st.markdown("### 📊 Current Strategy Parameters")
                    if signal_data:
                        param_cols = st.columns(4)
                        items = list(signal_data.items())
                        for idx, (key, value) in enumerate(items):
                            with param_cols[idx % 4]:
                                if isinstance(value, (int, float)):
                                    st.metric(key, f"{value:.4f}")
                                else:
                                    st.metric(key, str(value))
                    
                    # Signals
                    col1, col2 = st.columns(2)
                    with col1:
                        if bullish_signal:
                            st.success("🟢 BULLISH SIGNAL DETECTED")
                        else:
                            st.info("No bullish signal")
                    with col2:
                        if bearish_signal:
                            st.error("🔴 BEARISH SIGNAL DETECTED")
                        else:
                            st.info("No bearish signal")
                    
                    # Historical Stats
                    st.markdown("### 📊 Historical Statistics")
                    if isinstance(strategy, PairRatioStrategy):
                        hist_stats = strategy.get_historical_statistics(data, data2)
                    else:
                        hist_stats = strategy.get_historical_statistics(data)
                    
                    if hist_stats:
                        for category, stats in hist_stats.items():
                            st.markdown(f"**{category}**")
                            stat_cols = st.columns(len(stats))
                            for idx, (stat_name, stat_value) in enumerate(stats.items()):
                                with stat_cols[idx]:
                                    if isinstance(stat_value, (int, float)):
                                        st.metric(stat_name, f"{stat_value:.4f}")
                                    elif isinstance(stat_value, list):
                                        st.metric(stat_name, f"{len(stat_value)} items")
                                    else:
                                        st.metric(stat_name, str(stat_value))
                    
                    # Chart Preview
                    st.markdown("### 📈 Chart Preview (Last 20 Candles)")
                    preview_data = data.iloc[-20:]
                    chart = create_chart(preview_data, strategy, None, signal_data)
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.error("Failed to fetch data. Check ticker symbol and try again.")
    
    with tab2:
        st.header("📊 Trade History & Performance")
        
        if not st.session_state.trade_history:
            st.info("No trades yet. Start trading to see your performance here.")
        else:
            trades_df = pd.DataFrame(st.session_state.trade_history)
            
            # Performance Summary
            st.subheader("📈 Performance Summary")
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
            losing_trades = len(trades_df[trades_df['pnl_pct'] <= 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = trades_df['pnl_points'].sum()
            avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].mean() if losing_trades > 0 else 0
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Trades", total_trades)
            with col2:
                st.metric("Wins", winning_trades, delta=f"{win_rate:.1f}%")
            with col3:
                st.metric("Losses", losing_trades)
            with col4:
                pnl_color = 'profit' if total_pnl > 0 else 'loss'
                st.markdown(f"<div class='metric-card'><b>Total P&L</b><br><span class='{pnl_color}'>{total_pnl:.2f} pts</span></div>", unsafe_allow_html=True)
            with col5:
                st.metric("Avg Win %", f"{avg_win:.2f}%")
            
            # Strategy Breakdown
            st.subheader("📊 Performance by Strategy")
            strategy_perf = trades_df.groupby('strategy').agg({
                'pnl_points': 'sum',
                'pnl_pct': 'mean',
                'ticker': 'count'
            }).rename(columns={'ticker': 'trades'})
            st.dataframe(strategy_perf, use_container_width=True)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Win/Loss Pie Chart
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Wins', 'Losses'],
                    values=[winning_trades, losing_trades],
                    marker=dict(colors=['#00c853', '#d32f2f'])
                )])
                fig_pie.update_layout(title='Win/Loss Distribution', height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # P&L Distribution
                fig_hist = go.Figure(data=[go.Histogram(
                    x=trades_df['pnl_pct'],
                    nbinsx=20,
                    marker_color='#1f77b4'
                )])
                fig_hist.update_layout(title='P&L Distribution (%)', xaxis_title='P&L %', yaxis_title='Count', height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Cumulative P&L
            trades_df_sorted = trades_df.sort_values('exit_time')
            trades_df_sorted['cumulative_pnl'] = trades_df_sorted['pnl_points'].cumsum()
            
            fig_cumulative = go.Figure()
            fig_cumulative.add_trace(go.Scatter(
                x=trades_df_sorted['exit_time'],
                y=trades_df_sorted['cumulative_pnl'],
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='#2196f3', width=2)
            ))
            fig_cumulative.update_layout(title='Cumulative P&L Over Time', xaxis_title='Time', yaxis_title='Cumulative P&L (Points)', height=400)
            st.plotly_chart(fig_cumulative, use_container_width=True)
            
            # Detailed Trade Table
            st.subheader("📋 Detailed Trade Log")
            display_df = trades_df.copy()
            display_df['entry_time'] = display_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['exit_time'] = display_df['exit_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['duration'] = (pd.to_datetime(trades_df['exit_time']) - pd.to_datetime(trades_df['entry_time'])).dt.total_seconds() / 60
            display_df['duration'] = display_df['duration'].apply(lambda x: f"{x:.1f} min")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Individual Trade Analysis
            st.subheader("🔍 Individual Trade Analysis")
            trade_idx = st.selectbox("Select Trade to Analyze", range(len(trades_df)), format_func=lambda x: f"Trade #{x+1} - {trades_df.iloc[x]['strategy']} - {trades_df.iloc[x]['type']} - P&L: {trades_df.iloc[x]['pnl_pct']:.2f}%")
            
            if trade_idx is not None:
                selected_trade = trades_df.iloc[trade_idx].to_dict()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Strategy", selected_trade['strategy'])
                    st.metric("Position Type", selected_trade['type'])
                with col2:
                    st.metric("Entry Price", f"{selected_trade['entry_price']:.2f}")
                    st.metric("Exit Price", f"{selected_trade['exit_price']:.2f}")
                with col3:
                    pnl_color = 'profit' if selected_trade['pnl_pct'] > 0 else 'loss'
                    st.markdown(f"<div class='metric-card'><b>P&L</b><br><span class='{pnl_color}'>{selected_trade['pnl_pct']:.2f}%</span></div>", unsafe_allow_html=True)
                    st.metric("Exit Reason", selected_trade['exit_reason'])
                
                # AI Analysis for selected trade
                st.markdown("---")
                st.markdown(trading_system.analyze_trade_performance(selected_trade))
    
    with tab3:
        st.header("📝 Trade Log")
        st.caption("Last 100 events")
        
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🗑️ Clear Log"):
                st.session_state.trade_log = []
                st.rerun()
        
        if st.session_state.trade_log:
            log_html = "<div class='log-container'>"
            for entry in reversed(st.session_state.trade_log):
                log_html += f"<div>{entry}</div>"
            log_html += "</div>"
            st.markdown(log_html, unsafe_allow_html=True)
        else:
            st.info("No log entries yet. Start trading to see events here.")

if __name__ == "__main__":
    main() 'ATR Based':
            atr = self.calculate_atr(data)
            if position_type == 'LONG':
                return entry_price - (atr * sl_value)
            else:
                return entry_price + (atr * sl_value)
        
        elif sl_type == 'Previous Swing':
            swing = self.find_previous_swing(data, position_type)
            return swing
        
        elif sl_type == 'Current Swing':
            if position_type == 'LONG':
                return data['Low'].iloc[-1]
            else:
                return data['High'].iloc[-1]
        
        elif sl_type == 'Current Candle':
            if position_type == 'LONG':
                return data['Low'].iloc[-1]
            else:
                return data['High'].iloc[-1]
        
        elif sl_type == 'Percentage Based':
            pct = sl_value / 100
            if position_type == 'LONG':
                return entry_price * (1 - pct)
            else:
                return entry_price * (1 + pct)
        
        elif sl_type ==
