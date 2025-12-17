import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from abc import ABC, abstractmethod
from scipy.signal import argrelextrema
from scipy.stats import zscore, pearsonr
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
IST = pytz.timezone('Asia/Kolkata')

PRESET_TICKERS = {
    'NIFTY 50': '^NSEI',
    'Bank NIFTY': '^NSEBANK',
    'SENSEX': '^BSESN',
    'BTC-USD': 'BTC-USD',
    'ETH-USD': 'ETH-USD',
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'USD/INR': 'INR=X',
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X'
}

TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1wk', '1mo']
PERIODS = ['1d', '5d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '3y', '5y', '10y', '15y', '20y', '25y', '30y']

TIMEFRAME_PERIOD_COMPATIBILITY = {
    '1m': ['1d', '5d'],
    '5m': ['1d', '5d', '1mo'],
    '15m': ['1d', '5d', '1mo'],
    '30m': ['1d', '5d', '1mo'],
    '1h': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
    '2h': ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y'],
    '4h': ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y'],
    '1d': ['1mo', '3mo', '6mo', '1y', '2y', '3y', '5y', '10y', '15y', '20y', '25y', '30y'],
    '1wk': ['1y', '2y', '3y', '5y', '10y', '15y', '20y', '25y', '30y'],
    '1mo': ['1y', '2y', '3y', '5y', '10y', '15y', '20y', '25y', '30y']
}

# ==================== UTILITY FUNCTIONS ====================
def to_ist(dt):
    """Convert datetime to IST"""
    if dt is None:
        return None
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return dt.astimezone(IST)

def format_ist_time(dt):
    """Format datetime in IST"""
    if dt is None:
        return "N/A"
    ist_dt = to_ist(dt)
    return ist_dt.strftime('%Y-%m-%d %H:%M:%S IST')

def add_log(message):
    """Add log entry with IST timestamp"""
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    
    timestamp = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')
    log_entry = f"[{timestamp}] {message}"
    st.session_state.trade_log.insert(0, log_entry)
    
    if len(st.session_state.trade_log) > 100:
        st.session_state.trade_log = st.session_state.trade_log[:100]

def fetch_data_with_rate_limit(ticker, period, interval):
    """Fetch data with rate limiting and error handling"""
    try:
        time.sleep(1.8)
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if data.empty:
            return None
        
        if data.index.tzinfo is None:
            data.index = pd.to_datetime(data.index).tz_localize('UTC')
        data.index = data.index.tz_convert(IST)
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_atr(data, period=14):
    """Calculate Average True Range"""
    high = data['High']
    low = data['Low']
    close = data['Close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    return atr

def find_support_resistance(data, window=20, num_levels=3):
    """Find support and resistance levels"""
    highs = data['High'].rolling(window, center=True).max()
    lows = data['Low'].rolling(window, center=True).min()
    
    resistance_levels = []
    support_levels = []
    
    for i in range(window, len(data) - window):
        if data['High'].iloc[i] == highs.iloc[i]:
            resistance_levels.append(data['High'].iloc[i])
        if data['Low'].iloc[i] == lows.iloc[i]:
            support_levels.append(data['Low'].iloc[i])
    
    # Cluster nearby levels
    def cluster_levels(levels, tolerance=0.02):
        if not levels:
            return []
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] < tolerance:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        clustered.append(np.mean(current_cluster))
        return clustered[-num_levels:]
    
    return cluster_levels(support_levels), cluster_levels(resistance_levels)

def detect_candlestick_patterns(data):
    """Detect candlestick patterns"""
    patterns = {}
    
    if len(data) < 5:
        return patterns
    
    # Get last few candles
    o = data['Open'].values
    h = data['High'].values
    l = data['Low'].values
    c = data['Close'].values
    
    # Hammer
    body = abs(c[-1] - o[-1])
    lower_shadow = min(o[-1], c[-1]) - l[-1]
    upper_shadow = h[-1] - max(o[-1], c[-1])
    
    if lower_shadow > 2 * body and upper_shadow < body * 0.3 and c[-1] > o[-1]:
        patterns['Bullish_Hammer'] = True
    
    # Shooting Star
    if upper_shadow > 2 * body and lower_shadow < body * 0.3 and c[-1] < o[-1]:
        patterns['Bearish_Shooting_Star'] = True
    
    # Engulfing
    if len(data) >= 2:
        if c[-1] > o[-1] and c[-2] < o[-2] and o[-1] < c[-2] and c[-1] > o[-2]:
            patterns['Bullish_Engulfing'] = True
        if c[-1] < o[-1] and c[-2] > o[-2] and o[-1] > c[-2] and c[-1] < o[-2]:
            patterns['Bearish_Engulfing'] = True
    
    # Doji
    if body < (h[-1] - l[-1]) * 0.1:
        patterns['Doji'] = True
    
    return patterns

# ==================== BASE STRATEGY CLASS ====================
class BaseStrategy(ABC):
    def __init__(self, name, params=None):
        self.name = name
        self.params = params or {}
    
    @abstractmethod
    def calculate_indicators(self, data):
        pass
    
    @abstractmethod
    def generate_signal(self, data, data2=None):
        pass
    
    def get_parameters_display(self, data, data2=None):
        return {}
    
    def add_to_chart(self, fig, data, data2=None):
        """Add strategy-specific overlays to chart"""
        pass

# ==================== STRATEGY IMPLEMENTATIONS ====================

class EMACrossoverStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__("EMA/SMA Crossover", params)
    
    def calculate_indicators(self, data):
        ma_type1 = self.params.get('ma_type1', 'EMA')
        ma_type2 = self.params.get('ma_type2', 'EMA')
        period1 = self.params.get('period1', 9)
        period2 = self.params.get('period2', 20)
        
        if ma_type1 == 'EMA':
            data['MA1'] = data['Close'].ewm(span=period1, adjust=False).mean()
        else:
            data['MA1'] = data['Close'].rolling(window=period1).mean()
        
        if ma_type2 == 'EMA':
            data['MA2'] = data['Close'].ewm(span=period2, adjust=False).mean()
        else:
            data['MA2'] = data['Close'].rolling(window=period2).mean()
        
        return data
    
    def generate_signal(self, data, data2=None):
        data = self.calculate_indicators(data)
        
        if len(data) < 3:
            return False, False, {}
        
        prev_ma1 = data['MA1'].iloc[-2]
        prev_ma2 = data['MA2'].iloc[-2]
        curr_ma1 = data['MA1'].iloc[-1]
        curr_ma2 = data['MA2'].iloc[-1]
        
        crossover_type = self.params.get('crossover_type', 'simple')
        
        bullish = prev_ma1 <= prev_ma2 and curr_ma1 > curr_ma2
        bearish = prev_ma1 >= prev_ma2 and curr_ma1 < curr_ma2
        
        if crossover_type == 'strong_candle':
            curr_close = data['Close'].iloc[-1]
            curr_open = data['Open'].iloc[-1]
            candle_body = abs(curr_close - curr_open)
            avg_body = abs(data['Close'] - data['Open']).tail(20).mean()
            
            strong_candle = candle_body > avg_body * 1.5
            
            if bullish:
                bullish = bullish and strong_candle and curr_close > curr_ma1
            if bearish:
                bearish = bearish and strong_candle and curr_close < curr_ma1
        
        signal_data = {
            'MA1': curr_ma1,
            'MA2': curr_ma2,
            'Spread': abs(curr_ma1 - curr_ma2),
            'Distance_to_Cross': abs(curr_ma1 - curr_ma2)
        }
        
        return bullish, bearish, signal_data
    
    def get_parameters_display(self, data, data2=None):
        data = self.calculate_indicators(data)
        return {
            f"{self.params.get('ma_type1', 'EMA')}{self.params.get('period1', 9)}": data['MA1'].iloc[-1],
            f"{self.params.get('ma_type2', 'EMA')}{self.params.get('period2', 20)}": data['MA2'].iloc[-1],
            'Spread': abs(data['MA1'].iloc[-1] - data['MA2'].iloc[-1])
        }
    
    def add_to_chart(self, fig, data, data2=None):
        data = self.calculate_indicators(data)
        fig.add_trace(go.Scatter(x=data.index, y=data['MA1'],
                                name=f"{self.params.get('ma_type1')} {self.params.get('period1')}",
                                line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA2'],
                                name=f"{self.params.get('ma_type2')} {self.params.get('period2')}",
                                line=dict(color='red', width=2)))

class RSIDivergenceStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__("RSI + Divergence", params)
    
    def calculate_indicators(self, data):
        period = self.params.get('rsi_period', 14)
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        return data
    
    def detect_divergence(self, data, lookback=20):
        if len(data) < lookback:
            return False, False, [], []
        
        recent_data = data.tail(lookback).copy()
        
        price_highs_idx = argrelextrema(recent_data['Close'].values, np.greater, order=3)[0]
        price_lows_idx = argrelextrema(recent_data['Close'].values, np.less, order=3)[0]
        rsi_highs_idx = argrelextrema(recent_data['RSI'].values, np.greater, order=3)[0]
        rsi_lows_idx = argrelextrema(recent_data['RSI'].values, np.less, order=3)[0]
        
        bullish_div = False
        bearish_div = False
        bullish_lines = []
        bearish_lines = []
        
        # Bullish divergence
        if len(price_lows_idx) >= 2 and len(rsi_lows_idx) >= 2:
            for i in range(len(price_lows_idx) - 1):
                pl1, pl2 = price_lows_idx[i], price_lows_idx[i + 1]
                if pl2 < len(rsi_lows_idx):
                    rl1, rl2 = rsi_lows_idx[i], rsi_lows_idx[min(i + 1, len(rsi_lows_idx) - 1)]
                    if (recent_data['Close'].iloc[pl2] < recent_data['Close'].iloc[pl1] and
                        recent_data['RSI'].iloc[rl2] > recent_data['RSI'].iloc[rl1]):
                        bullish_div = True
                        bullish_lines.append((pl1, pl2, rl1, rl2))
        
        # Bearish divergence
        if len(price_highs_idx) >= 2 and len(rsi_highs_idx) >= 2:
            for i in range(len(price_highs_idx) - 1):
                ph1, ph2 = price_highs_idx[i], price_highs_idx[i + 1]
                if ph2 < len(rsi_highs_idx):
                    rh1, rh2 = rsi_highs_idx[i], rsi_highs_idx[min(i + 1, len(rsi_highs_idx) - 1)]
                    if (recent_data['Close'].iloc[ph2] > recent_data['Close'].iloc[ph1] and
                        recent_data['RSI'].iloc[rh2] < recent_data['RSI'].iloc[rh1]):
                        bearish_div = True
                        bearish_lines.append((ph1, ph2, rh1, rh2))
        
        return bullish_div, bearish_div, bullish_lines, bearish_lines
    
    def generate_signal(self, data, data2=None):
        data = self.calculate_indicators(data)
        
        if len(data) < 30:
            return False, False, {}
        
        curr_rsi = data['RSI'].iloc[-1]
        bullish_div, bearish_div, _, _ = self.detect_divergence(data)
        
        rsi_oversold = curr_rsi < 30
        rsi_overbought = curr_rsi > 70
        
        bullish = (rsi_oversold or bullish_div)
        bearish = (rsi_overbought or bearish_div)
        
        signal_data = {
            'RSI': curr_rsi,
            'Bullish_Div': bullish_div,
            'Bearish_Div': bearish_div,
            'Points_to_Oversold': max(0, 30 - curr_rsi),
            'Points_to_Overbought': max(0, curr_rsi - 70)
        }
        
        return bullish, bearish, signal_data
    
    def get_parameters_display(self, data, data2=None):
        data = self.calculate_indicators(data)
        return {
            'RSI': data['RSI'].iloc[-1],
            'RSI_MA_10': data['RSI'].tail(10).mean(),
            'RSI_Min_20': data['RSI'].tail(20).min(),
            'RSI_Max_20': data['RSI'].tail(20).max()
        }
    
    def add_to_chart(self, fig, data, data2=None):
        data = self.calculate_indicators(data)
        bullish_div, bearish_div, bullish_lines, bearish_lines = self.detect_divergence(data)
        
        # Create subplot for RSI
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], 
                           vertical_spacing=0.05,
                           subplot_titles=('Price', 'RSI'))
        
        # Add RSI
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'],
                                name='RSI', line=dict(color='purple')),
                     row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        return fig

class FibonacciStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__("Fibonacci Retracement", params)
    
    def calculate_indicators(self, data):
        lookback = self.params.get('lookback', 50)
        
        recent = data.tail(lookback)
        swing_high = recent['High'].max()
        swing_low = recent['Low'].min()
        
        diff = swing_high - swing_low
        
        data['Fib_0'] = swing_high
        data['Fib_236'] = swing_high - 0.236 * diff
        data['Fib_382'] = swing_high - 0.382 * diff
        data['Fib_50'] = swing_high - 0.5 * diff
        data['Fib_618'] = swing_high - 0.618 * diff
        data['Fib_786'] = swing_high - 0.786 * diff
        data['Fib_100'] = swing_low
        
        return data
    
    def generate_signal(self, data, data2=None):
        data = self.calculate_indicators(data)
        
        if len(data) < 10:
            return False, False, {}
        
        curr_price = data['Close'].iloc[-1]
        tolerance = self.params.get('tolerance', 0.005)
        
        key_levels = {
            '38.2%': data['Fib_382'].iloc[-1],
            '50%': data['Fib_50'].iloc[-1],
            '61.8%': data['Fib_618'].iloc[-1]
        }
        
        nearest_level = None
        nearest_level_name = None
        min_distance = float('inf')
        
        for name, level in key_levels.items():
            distance = abs(curr_price - level) / level
            if distance < tolerance and distance < min_distance:
                nearest_level = level
                nearest_level_name = name
                min_distance = distance
        
        near_level = nearest_level is not None
        
        ma20 = data['Close'].rolling(20).mean().iloc[-1]
        trend_up = curr_price > ma20
        
        bullish = near_level and trend_up and curr_price < data['Fib_618'].iloc[-1]
        bearish = near_level and not trend_up and curr_price > data['Fib_382'].iloc[-1]
        
        signal_data = {
            'Price': curr_price,
            'Nearest_Fib': f"{nearest_level_name}: {nearest_level:.2f}" if nearest_level else 'None',
            'Fib_382': data['Fib_382'].iloc[-1],
            'Fib_50': data['Fib_50'].iloc[-1],
            'Fib_618': data['Fib_618'].iloc[-1]
        }
        
        return bullish, bearish, signal_data
    
    def get_parameters_display(self, data, data2=None):
        data = self.calculate_indicators(data)
        return {
            'Fib_0%': data['Fib_0'].iloc[-1],
            'Fib_38.2%': data['Fib_382'].iloc[-1],
            'Fib_50%': data['Fib_50'].iloc[-1],
            'Fib_61.8%': data['Fib_618'].iloc[-1],
            'Fib_100%': data['Fib_100'].iloc[-1]
        }
    
    def add_to_chart(self, fig, data, data2=None):
        data = self.calculate_indicators(data)
        fib_levels = {
            '0%': data['Fib_0'].iloc[-1],
            '23.6%': data['Fib_236'].iloc[-1],
            '38.2%': data['Fib_382'].iloc[-1],
            '50%': data['Fib_50'].iloc[-1],
            '61.8%': data['Fib_618'].iloc[-1],
            '78.6%': data['Fib_786'].iloc[-1],
            '100%': data['Fib_100'].iloc[-1]
        }
        
        for name, level in fib_levels.items():
            fig.add_hline(y=level, line_dash="dot", line_color="orange",
                         annotation_text=f"Fib {name}")

class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__("Bollinger Bands", params)
    
    def calculate_indicators(self, data):
        period = self.params.get('period', 20)
        std_dev = self.params.get('std_dev', 2)
        
        data['BB_Middle'] = data['Close'].rolling(period).mean()
        data['BB_Std'] = data['Close'].rolling(period).std()
        data['BB_Upper'] = data['BB_Middle'] + (std_dev * data['BB_Std'])
        data['BB_Lower'] = data['BB_Middle'] - (std_dev * data['BB_Std'])
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
        
        return data
    
    def generate_signal(self, data, data2=None):
        data = self.calculate_indicators(data)
        
        if len(data) < 25:
            return False, False, {}
        
        curr_price = data['Close'].iloc[-1]
        upper = data['BB_Upper'].iloc[-1]
        lower = data['BB_Lower'].iloc[-1]
        middle = data['BB_Middle'].iloc[-1]
        
        # Mean reversion signals
        bullish = curr_price < lower
        bearish = curr_price > upper
        
        signal_data = {
            'Price': curr_price,
            'BB_Upper': upper,
            'BB_Middle': middle,
            'BB_Lower': lower,
            'BB_Width': data['BB_Width'].iloc[-1],
            'Distance_to_Lower': abs(curr_price - lower),
            'Distance_to_Upper': abs(curr_price - upper)
        }
        
        return bullish, bearish, signal_data
    
    def get_parameters_display(self, data, data2=None):
        data = self.calculate_indicators(data)
        return {
            'BB_Upper': data['BB_Upper'].iloc[-1],
            'BB_Middle': data['BB_Middle'].iloc[-1],
            'BB_Lower': data['BB_Lower'].iloc[-1],
            'BB_Width': data['BB_Width'].iloc[-1]
        }
    
    def add_to_chart(self, fig, data, data2=None):
        data = self.calculate_indicators(data)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'],
                                name='BB Upper', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Middle'],
                                name='BB Middle', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'],
                                name='BB Lower', line=dict(color='green', dash='dash')))

class VWAPStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__("VWAP Strategy", params)
    
    def calculate_indicators(self, data):
        if 'Volume' in data.columns:
            data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        else:
            data['VWAP'] = data['Close'].expanding().mean()
        
        return data
    
    def generate_signal(self, data, data2=None):
        data = self.calculate_indicators(data)
        
        if len(data) < 10:
            return False, False, {}
        
        curr_price = data['Close'].iloc[-1]
        vwap = data['VWAP'].iloc[-1]
        
        threshold = self.params.get('threshold', 0.002)
        
        bullish = curr_price < vwap * (1 - threshold)
        bearish = curr_price > vwap * (1 + threshold)
        
        signal_data = {
            'Price': curr_price,
            'VWAP': vwap,
            'Distance_%': ((curr_price - vwap) / vwap * 100),
            'Points_to_Signal': abs(curr_price - vwap * (1 - threshold if curr_price < vwap else 1 + threshold))
        }
        
        return bullish, bearish, signal_data
    
    def get_parameters_display(self, data, data2=None):
        data = self.calculate_indicators(data)
        return {
            'VWAP': data['VWAP'].iloc[-1],
            'Price': data['Close'].iloc[-1],
            'Distance_%': ((data['Close'].iloc[-1] - data['VWAP'].iloc[-1]) / data['VWAP'].iloc[-1] * 100)
        }
    
    def add_to_chart(self, fig, data, data2=None):
        data = self.calculate_indicators(data)
        fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'],
                                name='VWAP', line=dict(color='purple', width=2)))

class SupportResistanceStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__("Support/Resistance Breakout", params)
    
    def calculate_indicators(self, data):
        window = self.params.get('window', 20)
        support, resistance = find_support_resistance(data, window)
        
        data['Support_Levels'] = [support] * len(data)
        data['Resistance_Levels'] = [resistance] * len(data)
        
        return data
    
    def generate_signal(self, data, data2=None):
        data = self.calculate_indicators(data)
        
        if len(data) < 25:
            return False, False, {}
        
        curr_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        
        support = data['Support_Levels'].iloc[-1]
        resistance = data['Resistance_Levels'].iloc[-1]
        
        bullish = False
        bearish = False
        
        tolerance = curr_price * 0.002
        
        # Check for breakout with retest
        for res_level in resistance:
            if prev_price < res_level and curr_price > res_level + tolerance:
                bullish = True
        
        for sup_level in support:
            if prev_price > sup_level and curr_price < sup_level - tolerance:
                bearish = True
        
        signal_data = {
            'Price': curr_price,
            'Nearest_Support': min(support, key=lambda x: abs(x - curr_price)) if support else None,
            'Nearest_Resistance': min(resistance, key=lambda x: abs(x - curr_price)) if resistance else None,
            'Num_Support_Levels': len(support),
            'Num_Resistance_Levels': len(resistance)
        }
        
        return bullish, bearish, signal_data
    
    def get_parameters_display(self, data, data2=None):
        data = self.calculate_indicators(data)
        support = data['Support_Levels'].iloc[-1]
        resistance = data['Resistance_Levels'].iloc[-1]
        curr_price = data['Close'].iloc[-1]
        
        return {
            'Nearest_Support': f"{min(support, key=lambda x: abs(x - curr_price)):.2f}" if support else 'N/A',
            'Nearest_Resistance': f"{min(resistance, key=lambda x: abs(x - curr_price)):.2f}" if resistance else 'N/A',
            'Total_Levels': len(support) + len(resistance)
        }
    
    def add_to_chart(self, fig, data, data2=None):
        data = self.calculate_indicators(data)
        support = data['Support_Levels'].iloc[-1]
        resistance = data['Resistance_Levels'].iloc[-1]
        
        for level in support:
            fig.add_hline(y=level, line_dash="dash", line_color="green",
                         annotation_text=f"Support {level:.2f}")
        
        for level in resistance:
            fig.add_hline(y=level, line_dash="dash", line_color="red",
                         annotation_text=f"Resistance {level:.2f}")

class CandlestickPatternStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__("Candlestick Patterns", params)
    
    def calculate_indicators(self, data):
        return data
    
    def generate_signal(self, data, data2=None):
        if len(data) < 5:
            return False, False, {}
        
        patterns = detect_candlestick_patterns(data)
        
        bullish = patterns.get('Bullish_Hammer', False) or patterns.get('Bullish_Engulfing', False)
        bearish = patterns.get('Bearish_Shooting_Star', False) or patterns.get('Bearish_Engulfing', False)
        
        signal_data = {
            'Patterns_Detected': ', '.join(patterns.keys()) if patterns else 'None',
            'Pattern_Count': len(patterns)
        }
        signal_data.update(patterns)
        
        return bullish, bearish, signal_data
    
    def get_parameters_display(self, data, data2=None):
        patterns = detect_candlestick_patterns(data)
        return {
            'Patterns': ', '.join(patterns.keys()) if patterns else 'None',
            'Count': len(patterns)
        }

class ATRBreakoutStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__("ATR Volatility Breakout", params)
    
    def calculate_indicators(self, data):
        period = self.params.get('atr_period', 14)
        data['ATR'] = calculate_atr(data, period)
        data['ATR_MA'] = data['ATR'].rolling(20).mean()
        
        return data
    
    def generate_signal(self, data, data2=None):
        data = self.calculate_indicators(data)
        
        if len(data) < 25:
            return False, False, {}
        
        multiplier = self.params.get('multiplier', 2.0)
        
        curr_atr = data['ATR'].iloc[-1]
        avg_atr = data['ATR_MA'].iloc[-1]
        curr_close = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        
        move = abs(curr_close - prev_close)
        
        volatility_surge = curr_atr > avg_atr * 1.5
        
        bullish = move > curr_atr * multiplier and curr_close > prev_close and volatility_surge
        bearish = move > curr_atr * multiplier and curr_close < prev_close and volatility_surge
        
        signal_data = {
            'ATR': curr_atr,
            'ATR_MA': avg_atr,
            'Move_Size': move,
            'Required_Move': curr_atr * multiplier,
            'Volatility_Surge': volatility_surge
        }
        
        return bullish, bearish, signal_data
    
    def get_parameters_display(self, data, data2=None):
        data = self.calculate_indicators(data)
        return {
            'ATR': data['ATR'].iloc[-1],
            'ATR_MA': data['ATR_MA'].iloc[-1],
            'Volatility_Ratio': data['ATR'].iloc[-1] / data['ATR_MA'].iloc[-1]
        }

class MACDStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__("MACD Strategy", params)
    
    def calculate_indicators(self, data):
        fast = self.params.get('fast', 12)
        slow = self.params.get('slow', 26)
        signal = self.params.get('signal', 9)
        
        ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
        
        data['MACD'] = ema_fast - ema_slow
        data['MACD_Signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
        
        return data
    
    def generate_signal(self, data, data2=None):
        data = self.calculate_indicators(data)
        
        if len(data) < 30:
            return False, False, {}
        
        macd = data['MACD'].iloc[-1]
        macd_signal = data['MACD_Signal'].iloc[-1]
        prev_macd = data['MACD'].iloc[-2]
        prev_signal = data['MACD_Signal'].iloc[-2]
        
        bullish = prev_macd <= prev_signal and macd > macd_signal
        bearish = prev_macd >= prev_signal and macd < macd_signal
        
        signal_data = {
            'MACD': macd,
            'Signal': macd_signal,
            'Histogram': data['MACD_Hist'].iloc[-1],
            'Distance_to_Cross': abs(macd - macd_signal)
        }
        
        return bullish, bearish, signal_data
    
    def get_parameters_display(self, data, data2=None):
        data = self.calculate_indicators(data)
        return {
            'MACD': data['MACD'].iloc[-1],
            'Signal': data['MACD_Signal'].iloc[-1],
            'Histogram': data['MACD_Hist'].iloc[-1]
        }

class StochasticStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__("Stochastic Oscillator", params)
    
    def calculate_indicators(self, data):
        period = self.params.get('period', 14)
        smooth_k = self.params.get('smooth_k', 3)
        smooth_d = self.params.get('smooth_d', 3)
        
        low_min = data['Low'].rolling(period).min()
        high_max = data['High'].rolling(period).max()
        
        data['%K'] = 100 * (data['Close'] - low_min) / (high_max - low_min)
        data['%K'] = data['%K'].rolling(smooth_k).mean()
        data['%D'] = data['%K'].rolling(smooth_d).mean()
        
        return data
    
    def generate_signal(self, data, data2=None):
        data = self.calculate_indicators(data)
        
        if len(data) < 20:
            return False, False, {}
        
        k = data['%K'].iloc[-1]
        d = data['%D'].iloc[-1]
        prev_k = data['%K'].iloc[-2]
        prev_d = data['%D'].iloc[-2]
        
        bullish = k < 20 and prev_k <= prev_d and k > d
        bearish = k > 80 and prev_k >= prev_d and k < d
        
        signal_data = {
            '%K': k,
            '%D': d,
            'Oversold': k < 20,
            'Overbought': k > 80
        }
        
        return bullish, bearish, signal_data
    
    def get_parameters_display(self, data, data2=None):
        data = self.calculate_indicators(data)
        return {
            '%K': data['%K'].iloc[-1],
            '%D': data['%D'].iloc[-1]
        }

class MomentumStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__("Momentum Strategy", params)
    
    def calculate_indicators(self, data):
        period = self.params.get('period', 14)
        data['Momentum'] = data['Close'] - data['Close'].shift(period)
        data['ROC'] = (data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period) * 100
        
        return data
    
    def generate_signal(self, data, data2=None):
        data = self.calculate_indicators(data)
        
        if len(data) < 20:
            return False, False, {}
        
        momentum = data['Momentum'].iloc[-1]
        roc = data['ROC'].iloc[-1]
        
        threshold = self.params.get('threshold', 2.0)
        
        bullish = roc > threshold and momentum > 0
        bearish = roc < -threshold and momentum < 0
        
        signal_data = {
            'Momentum': momentum,
            'ROC': roc,
            'Threshold': threshold
        }
        
        return bullish, bearish, signal_data
    
    def get_parameters_display(self, data, data2=None):
        data = self.calculate_indicators(data)
        return {
            'Momentum': data['Momentum'].iloc[-1],
            'ROC_%': data['ROC'].iloc[-1]
        }

class VolatilityBinningStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__("Volatility Binning", params)
    
    def calculate_indicators(self, data):
        window = self.params.get('window', 20)
        data['Volatility'] = data['Close'].pct_change().rolling(window).std() * np.sqrt(252) * 100
        data['Vol_Percentile'] = data['Volatility'].rolling(100).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() != x.min() else 50
        )
        
        return data
    
    def generate_signal(self, data, data2=None):
        data = self.calculate_indicators(data)
        
        if len(data) < 100:
            return False, False, {}
        
        vol_pct = data['Vol_Percentile'].iloc[-1]
        
        # Low volatility -> expect expansion (breakout setup)
        # High volatility -> expect contraction (mean reversion)
        
        bullish = vol_pct < 20 and data['Close'].iloc[-1] > data['Close'].rolling(20).mean().iloc[-1]
        bearish = vol_pct > 80
        
        signal_data = {
            'Volatility': data['Volatility'].iloc[-1],
            'Vol_Percentile': vol_pct,
            'Regime': 'Low' if vol_pct < 30 else 'High' if vol_pct > 70 else 'Medium'
        }
        
        return bullish, bearish, signal_data
    
    def get_parameters_display(self, data, data2=None):
        data = self.calculate_indicators(data)
        return {
            'Volatility_%': data['Volatility'].iloc[-1],
            'Vol_Percentile': data['Vol_Percentile'].iloc[-1]
        }

class CorrelationStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__("Correlation Strategy", params)
    
    def calculate_indicators(self, data, data2=None):
        if data2 is None or data2.empty:
            return data
        
        common_index = data.index.intersection(data2.index)
        if len(common_index) < 30:
            return data
        
        data = data.loc[common_index]
        data2_aligned = data2.loc[common_index]
        
        window = self.params.get('window', 20)
        
        returns1 = data['Close'].pct_change()
        returns2 = data2_aligned['Close'].pct_change()
        
        data['Correlation'] = returns1.rolling(window).corr(returns2)
        
        return data
    
    def generate_signal(self, data, data2=None):
        data = self.calculate_indicators(data, data2)
        
        if 'Correlation' not in data.columns or len(data) < 30:
            return False, False, {}
        
        corr = data['Correlation'].iloc[-1]
        
        # Trade when correlation breaks down (divergence opportunity)
        threshold = self.params.get('threshold', 0.3)
        
        bullish = abs(corr) < threshold and data['Close'].iloc[-1] < data['Close'].rolling(20).mean().iloc[-1]
        bearish = abs(corr) < threshold and data['Close'].iloc[-1] > data['Close'].rolling(20).mean().iloc[-1]
        
        signal_data = {
            'Correlation': corr,
            'Threshold': threshold,
            'Regime': 'Correlated' if abs(corr) > 0.7 else 'Diverging'
        }
        
        return bullish, bearish, signal_data
    
    def get_parameters_display(self, data, data2=None):
        data = self.calculate_indicators(data, data2)
        if 'Correlation' in data.columns:
            return {
                'Correlation': data['Correlation'].iloc[-1],
                'Corr_MA': data['Correlation'].tail(10).mean()
            }
        return {}

class SeasonalityStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__("Seasonality Strategy", params)
    
    def calculate_indicators(self, data):
        data['Month'] = data.index.month
        data['Day_of_Week'] = data.index.dayofweek
        data['Returns'] = data['Close'].pct_change()
        
        return data
    
    def generate_signal(self, data, data2=None):
        data = self.calculate_indicators(data)
        
        if len(data) < 60:
            return False, False, {}
        
        curr_month = data['Month'].iloc[-1]
        curr_dow = data['Day_of_Week'].iloc[-1]
        
        # Calculate historical performance
        month_returns = data[data['Month'] == curr_month]['Returns'].mean()
        dow_returns = data[data['Day_of_Week'] == curr_dow]['Returns'].mean()
        
        bullish = month_returns > 0 and dow_returns > 0
        bearish = month_returns < 0 and dow_returns < 0
        
        signal_data = {
            'Month': curr_month,
            'Month_Avg_Return_%': month_returns * 100,
            'DOW': curr_dow,
            'DOW_Avg_Return_%': dow_returns * 100
        }
        
        return bullish, bearish, signal_data
    
    def get_parameters_display(self, data, data2=None):
        data = self.calculate_indicators(data)
        curr_month = data['Month'].iloc[-1]
        month_returns = data[data['Month'] == curr_month]['Returns'].mean()
        
        return {
            'Current_Month': curr_month,
            'Month_Avg_Return_%': month_returns * 100
        }

class PairRatioStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__("Pair Ratio Trading", params)
    
    def calculate_indicators(self, data, data2=None):
        if data2 is None or data2.empty:
            return data
        
        common_index = data.index.intersection(data2.index)
        if len(common_index) < 20:
            return data
        
        data = data.loc[common_index]
        data2 = data2.loc[common_index]
        
        data['Ratio'] = data['Close'] / data2['Close']
        data['Ratio_MA'] = data['Ratio'].rolling(50).mean()
        data['Ratio_Std'] = data['Ratio'].rolling(50).std()
        data['Ratio_ZScore'] = (data['Ratio'] - data['Ratio_MA']) / data['Ratio_Std']
        
        return data
    
    def generate_signal(self, data, data2=None):
        data = self.calculate_indicators(data, data2)
        
        if 'Ratio_ZScore' not in data.columns or len(data) < 25:
            return False, False, {}
        
        threshold = self.params.get('threshold', 2.0)
        curr_zscore = data['Ratio_ZScore'].iloc[-1]
        
        bullish = curr_zscore < -threshold
        bearish = curr_zscore > threshold
        
        signal_data = {
            'Ratio': data['Ratio'].iloc[-1],
            'Ratio_ZScore': curr_zscore,
            'Threshold': threshold,
            'Points_to_Signal': min(abs(curr_zscore + threshold), abs(curr_zscore - threshold))
        }
        
        return bullish, bearish, signal_data
    
    def get_parameters_display(self, data, data2=None):
        data = self.calculate_indicators(data, data2)
        if 'Ratio' in data.columns:
            return {
                'Ratio': data['Ratio'].iloc[-1],
                'Ratio_ZScore': data['Ratio_ZScore'].iloc[-1],
                'Ratio_MA': data['Ratio_MA'].iloc[-1]
            }
        return {}

class ZScoreMeanReversionStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__("Z-Score Mean Reversion", params)
    
    def calculate_indicators(self, data):
        window = self.params.get('window', 20)
        data['Price_MA'] = data['Close'].rolling(window).mean()
        data['Price_Std'] = data['Close'].rolling(window).std()
        data['ZScore'] = (data['Close'] - data['Price_MA']) / data['Price_Std']
        
        return data
    
    def generate_signal(self, data, data2=None):
        data = self.calculate_indicators(data)
        
        if len(data) < 25:
            return False, False, {}
        
        threshold = self.params.get('threshold', 2.0)
        curr_zscore = data['ZScore'].iloc[-1]
        
        bullish = curr_zscore < -threshold
        bearish = curr_zscore > threshold
        
        signal_data = {
            'ZScore': curr_zscore,
            'Threshold': threshold,
            'Price_MA': data['Price_MA'].iloc[-1],
            'Points_to_Signal': min(abs(curr_zscore + threshold), abs(curr_zscore - threshold))
        }
        
        return bullish, bearish, signal_data
    
    def get_parameters_display(self, data, data2=None):
        data = self.calculate_indicators(data)
        return {
            'ZScore': data['ZScore'].iloc[-1],
            'Price_MA': data['Price_MA'].iloc[-1],
            'Price_Std': data['Price_Std'].iloc[-1]
        }

class BreakoutVolumeStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__("Breakout with Volume", params)
    
    def calculate_indicators(self, data):
        period = self.params.get('period', 20)
        data['Upper_Band'] = data['High'].rolling(period).max()
        data['Lower_Band'] = data['Low'].rolling(period).min()
        
        if 'Volume' in data.columns:
            data['Avg_Volume'] = data['Volume'].rolling(period).mean()
        
        return data
    
    def generate_signal(self, data, data2=None):
        data = self.calculate_indicators(data)
        
        if len(data) < 25:
            return False, False, {}
        
        curr_close = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        upper = data['Upper_Band'].iloc[-2]
        lower = data['Lower_Band'].iloc[-2]
        
        volume_surge = True
        if 'Volume' in data.columns and 'Avg_Volume' in data.columns:
            curr_vol = data['Volume'].iloc[-1]
            avg_vol = data['Avg_Volume'].iloc[-1]
            volume_surge = curr_vol > avg_vol * 1.5
        
        bullish = curr_close > upper and prev_close <= upper and volume_surge
        bearish = curr_close < lower and prev_close >= lower and volume_surge
        
        signal_data = {
            'Upper_Band': upper,
            'Lower_Band': lower,
            'Current_Price': curr_close,
            'Volume_Surge': volume_surge,
            'Distance_to_Upper': abs(curr_close - upper),
            'Distance_to_Lower': abs(curr_close - lower)
        }
        
        return bullish, bearish, signal_data
    
    def get_parameters_display(self, data, data2=None):
        data = self.calculate_indicators(data)
        params = {
            'Upper_Band': data['Upper_Band'].iloc[-1],
            'Lower_Band': data['Lower_Band'].iloc[-1],
            'Range': data['Upper_Band'].iloc[-1] - data['Lower_Band'].iloc[-1]
        }
        if 'Avg_Volume' in data.columns:
            params['Avg_Volume'] = data['Avg_Volume'].iloc[-1]
        return params

class SimpleBuyStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__("Simple Buy", params)
    
    def calculate_indicators(self, data):
        return data
    
    def generate_signal(self, data, data2=None):
        return True, False, {'Signal': 'Always BUY'}
    
    def get_parameters_display(self, data, data2=None):
        return {'Signal': 'Always BUY'}

class SimpleSellStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__("Simple Sell", params)
    
    def calculate_indicators(self, data):
        return data
    
    def generate_signal(self, data, data2=None):
        return False, True, {'Signal': 'Always SELL'}
    
    def get_parameters_display(self, data, data2=None):
        return {'Signal': 'Always SELL'}

class HybridStrategy(BaseStrategy):
    def __init__(self, params, selected_strategies):
        super().__init__("Hybrid Multi-Strategy", params)
        self.selected_strategies = selected_strategies
        self.min_confirmations = params.get('min_confirmations', 2)
    
    def calculate_indicators(self, data):
        for strategy in self.selected_strategies:
            data = strategy.calculate_indicators(data)
        return data
    
    def generate_signal(self, data, data2=None):
        bullish_count = 0
        bearish_count = 0
        all_signal_data = {}
        
        for strategy in self.selected_strategies:
            bull, bear, sig_data = strategy.generate_signal(data, data2)
            if bull:
                bullish_count += 1
            if bear:
                bearish_count += 1
            all_signal_data[strategy.name] = sig_data
        
        bullish = bullish_count >= self.min_confirmations
        bearish = bearish_count >= self.min_confirmations
        
        signal_data = {
            'Bullish_Confirmations': bullish_count,
            'Bearish_Confirmations': bearish_count,
            'Min_Required': self.min_confirmations,
            'Total_Strategies': len(self.selected_strategies)
        }
        signal_data.update(all_signal_data)
        
        return bullish, bearish, signal_data
    
    def get_parameters_display(self, data, data2=None):
        params = {}
        for strategy in self.selected_strategies:
            strategy_params = strategy.get_parameters_display(data, data2)
            for key, value in strategy_params.items():
                params[f"{strategy.name}_{key}"] = value
        return params

# ==================== TRADING SYSTEM ====================
class TradingSystem:
    def __init__(self, ticker, ticker2, period, interval, strategy, sl_type, sl_value, 
                 target_type, target_value, quantity):
        self.ticker = ticker
        self.ticker2 = ticker2
        self.period = period
        self.interval = interval
        self.strategy = strategy
        self.sl_type = sl_type
        self.sl_value = sl_value
        self.target_type = target_type
        self.target_value = target_value
        self.quantity = quantity
        self.data = None
        self.data2 = None
    
    def fetch_data(self):
        self.data = fetch_data_with_rate_limit(self.ticker, self.period, self.interval)
        
        if self.ticker2:
            self.data2 = fetch_data_with_rate_limit(self.ticker2, self.period, self.interval)
        
        return self.data is not None
    
    def check_entry(self):
        if self.data is None or len(self.data) < 30:
            return None, None
        
        if isinstance(self.strategy, (PairRatioStrategy, CorrelationStrategy)) or hasattr(self.strategy, 'selected_strategies'):
            bullish, bearish, signal_data = self.strategy.generate_signal(self.data, self.data2)
        else:
            bullish, bearish, signal_data = self.strategy.generate_signal(self.data)
        
        if bullish:
            return 'LONG', signal_data
        elif bearish:
            return 'SHORT', signal_data
        
        return None, signal_data
    
    def calculate_sl_target(self, entry_price, position_type):
        sl_price = None
        target_price = None
        
        # Calculate Stop Loss
        if self.sl_type == 'Custom Points':
            if position_type == 'LONG':
                sl_price = entry_price - self.sl_value
            else:
                sl_price = entry_price + self.sl_value
        
        elif self.sl_type == 'ATR Based':
            atr = calculate_atr(self.data).iloc[-1]
            multiplier = self.sl_value
            if position_type == 'LONG':
                sl_price = entry_price - (atr * multiplier)
            else:
                sl_price = entry_price + (atr * multiplier)
        
        elif self.sl_type == 'Previous Swing':
            sl_price = self.find_swing_point(position_type, 'previous')
        
        elif self.sl_type == 'Current Swing':
            sl_price = self.find_swing_point(position_type, 'current')
        
        elif self.sl_type == 'Current Candle':
            if position_type == 'LONG':
                sl_price = self.data['Low'].iloc[-1]
            else:
                sl_price = self.data['High'].iloc[-1]
        
        elif self.sl_type == 'Percentage Based':
            pct = self.sl_value / 100
            if position_type == 'LONG':
                sl_price = entry_price * (1 - pct)
            else:
                sl_price = entry_price * (1 + pct)
        
        # Calculate Target
        if self.target_type == 'Custom Points':
            if position_type == 'LONG':
                target_price = entry_price + self.target_value
            else:
                target_price = entry_price - self.target_value
        
        elif self.target_type == 'Risk Reward Ratio':
            if sl_price:
                sl_distance = abs(entry_price - sl_price)
                rr_ratio = self.target_value
                if position_type == 'LONG':
                    target_price = entry_price + (sl_distance * rr_ratio)
                else:
                    target_price = entry_price - (sl_distance * rr_ratio)
        
        elif self.target_type == 'Percentage Based':
            pct = self.target_value / 100
            if position_type == 'LONG':
                target_price = entry_price * (1 + pct)
            else:
                target_price = entry_price * (1 - pct)
        
        return sl_price, target_price
    
    def find_swing_point(self, position_type, swing_type='previous'):
        lookback = 20 if swing_type == 'previous' else 5
        
        if position_type == 'LONG':
            return self.data['Low'].tail(lookback).min()
        else:
            return self.data['High'].tail(lookback).max()
    
    def update_trailing_sl(self, position, current_price):
        if self.sl_type != 'Trail SL':
            return position['sl_price']
        
        trail_points = self.sl_value
        position_type = position['type']
        
        if position_type == 'LONG':
            new_sl = current_price - trail_points
            if new_sl > position['sl_price']:
                add_log(f"Trailing SL updated: {position['sl_price']:.2f} -> {new_sl:.2f}")
                return new_sl
        else:
            new_sl = current_price + trail_points
            if new_sl < position['sl_price']:
                add_log(f"Trailing SL updated: {position['sl_price']:.2f} -> {new_sl:.2f}")
                return new_sl
        
        return position['sl_price']
    
    def update_trailing_target(self, position, current_price):
        if self.target_type != 'Trail Target':
            return position['target_price']
        
        trail_points = self.target_value
        position_type = position['type']
        tolerance = current_price * 0.01
        
        if position_type == 'LONG':
            effective_target = position['target_price'] - tolerance
            if current_price >= effective_target:
                new_target = current_price + trail_points
                if new_target > position['target_price']:
                    add_log(f"Trailing Target updated: {position['target_price']:.2f} -> {new_target:.2f}")
                    return new_target
        else:
            effective_target = position['target_price'] + tolerance
            if current_price <= effective_target:
                new_target = current_price - trail_points
                if new_target < position['target_price']:
                    add_log(f"Trailing Target updated: {position['target_price']:.2f} -> {new_target:.2f}")
                    return new_target
        
        return position['target_price']
    
    def check_exit(self, position, current_price):
        position_type = position['type']
        
        # Check SL
        if position['sl_price']:
            if position_type == 'LONG' and current_price <= position['sl_price']:
                return True, 'Stop Loss Hit'
            elif position_type == 'SHORT' and current_price >= position['sl_price']:
                return True, 'Stop Loss Hit'
        
        # Check Target with tolerance
        if position['target_price']:
            tolerance = current_price * 0.01
            if position_type == 'LONG' and current_price >= (position['target_price'] - tolerance):
                return True, 'Target Hit'
            elif position_type == 'SHORT' and current_price <= (position['target_price'] + tolerance):
                return True, 'Target Hit'
        
        # Check Signal Based Exit
        if self.sl_type == 'Signal Based' or self.target_type == 'Signal Based':
            if isinstance(self.strategy, (PairRatioStrategy, CorrelationStrategy)) or hasattr(self.strategy, 'selected_strategies'):
                bullish, bearish, _ = self.strategy.generate_signal(self.data, self.data2)
            else:
                bullish, bearish, _ = self.strategy.generate_signal(self.data)
            
            if position_type == 'LONG' and bearish:
                return True, 'Opposite Signal Generated'
            elif position_type == 'SHORT' and bullish:
                return True, 'Opposite Signal Generated'
        
        # Check P&L Based Exit
        if self.sl_type == 'P&L Based':
            pnl = (current_price - position['entry_price']) if position_type == 'LONG' else (position['entry_price'] - current_price)
            if pnl <= -self.sl_value:
                return True, 'P&L Stop Loss Hit'
        
        return False, None
    
    def get_market_status(self, position, current_price):
        if not position:
            return "Waiting for entry signal..."
        
        position_type = position['type']
        entry_price = position['entry_price']
        
        if position_type == 'LONG':
            pnl_points = current_price - entry_price
        else:
            pnl_points = entry_price - current_price
        
        pnl_pct = (pnl_points / entry_price) * 100
        
        status_text = ""
        
        if pnl_pct > 0.5:
            status_text = f"✅ Strong momentum in favor! (+{pnl_pct:.2f}%)"
        elif pnl_pct > 0:
            status_text = f"📈 Moving gradually in favor (+{pnl_pct:.2f}%)"
        else:
            status_text = f"⚠️ In loss - monitoring for reversal or SL hit ({pnl_pct:.2f}%)"
        
        if self.sl_type == 'Trail SL':
            status_text += " | 🔄 Trailing SL active"
        
        if position['sl_price']:
            dist_to_sl = abs(current_price - position['sl_price'])
            status_text += f" | SL Distance: {dist_to_sl:.2f}"
        
        if position['target_price']:
            dist_to_target = abs(current_price - position['target_price'])
            status_text += f" | Target Distance: {dist_to_target:.2f}"
        
        return status_text
    
    def get_trade_guidance(self, position, current_price, signal_data):
        if not position:
            return "No active position. System is monitoring for entry signals based on your selected strategy. Stay patient and wait for the right setup."
        
        position_type = position['type']
        entry_price = position['entry_price']
        
        if position_type == 'LONG':
            pnl_points = current_price - entry_price
        else:
            pnl_points = entry_price - current_price
        
        pnl_pct = (pnl_points / entry_price) * 100
        
        guidance = f"**Trade Management Advice:** "
        
        if pnl_pct > 1.0:
            guidance += f"Excellent! Trade moving well with +{pnl_pct:.2f}% profit. Consider booking partial profits to secure gains or letting it run with trailing SL. "
            guidance += "Avoid greed - secure some gains. Don't move SL away from price; let the strategy work. "
            guidance += f"Strategy expects continued momentum. Stay disciplined and trust your plan. "
        elif pnl_pct > 0:
            guidance += f"Small profit of {pnl_pct:.2f}%. Be patient - premature exits kill profitability. "
            guidance += "Strategy logic suggests more room to move. Hold position unless SL triggers. "
            guidance += "Avoid emotional decisions. Your plan is working - stick to it. Let winners run! "
        elif pnl_pct > -0.5:
            guidance += f"Minor drawdown ({pnl_pct:.2f}%). This is normal price action - don't panic! "
            guidance += "NEVER move SL further away hoping for recovery. Trust your risk management. "
            guidance += "Strategy can still work out. Price fluctuations are expected. Let your SL protect you if needed. "
        else:
            guidance += f"Significant loss ({pnl_pct:.2f}%). Respect your stop loss - this is why it exists! "
            guidance += "Don't hold losing trades hoping for miracles. If SL hits, accept it gracefully and move on. "
            guidance += "NEVER average down or remove SL - that's how accounts blow up. Capital preservation is paramount. "
        
        return guidance

# ==================== STREAMLIT UI ====================
def main():
    st.set_page_config(page_title="Algo Trading System", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #FFFFFF;
    }
    .stApp {
        background-color: #F8F9FA;
    }
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #000000 !important;
    }
    .profit {
        color: #00C853 !important;
        font-weight: bold;
        font-size: 1.2em;
    }
    .loss {
        color: #D32F2F !important;
        font-weight: bold;
        font-size: 1.2em;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .status-box {
        background: #E3F2FD;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 15px 0;
        color: #000000 !important;
    }
    .guidance-box {
        background: #FFF3E0;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #FF9800;
        margin: 15px 0;
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🚀 Professional Algorithmic Trading System - Complete Edition")
    st.markdown("**50+ Strategies | Advanced Risk Management | Real-Time AI Guidance**")
    
    # Initialize session state
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
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Ticker Selection
    ticker_option = st.sidebar.selectbox("Select Asset Type", ['Preset Assets', 'Custom Ticker', 'Indian Stock'])
    
    if ticker_option == 'Preset Assets':
        ticker = st.sidebar.selectbox("Select Asset", list(PRESET_TICKERS.keys()))
        ticker_symbol = PRESET_TICKERS[ticker]
    elif ticker_option == 'Indian Stock':
        stock_name = st.sidebar.text_input("Enter Stock Symbol (without .NS)")
        ticker_symbol = f"{stock_name}.NS" if stock_name else "RELIANCE.NS"
    else:
        ticker_symbol = st.sidebar.text_input("Enter Ticker Symbol", "AAPL")
    
    # Timeframe and Period
    timeframe = st.sidebar.selectbox("Timeframe", TIMEFRAMES, index=4)
    compatible_periods = TIMEFRAME_PERIOD_COMPATIBILITY.get(timeframe, PERIODS)
    period = st.sidebar.selectbox("Period", compatible_periods)
    
    # Strategy Selection
    st.sidebar.subheader("📊 Strategy Configuration")
    
    strategy_options = [
        'EMA/SMA Crossover',
        'RSI + Divergence',
        'Fibonacci Retracement',
        'Bollinger Bands',
        'VWAP Strategy',
        'Support/Resistance Breakout',
        'Candlestick Patterns',
        'ATR Volatility Breakout',
        'MACD Strategy',
        'Stochastic Oscillator',
        'Momentum Strategy',
        'Volatility Binning',
        'Correlation Strategy',
        'Seasonality Strategy',
        'Pair Ratio Trading',
        'Z-Score Mean Reversion',
        'Breakout with Volume',
        'Hybrid Multi-Strategy',
        'Simple Buy',
        'Simple Sell'
    ]
    
    selected_strategy = st.sidebar.selectbox("Select Strategy", strategy_options)
    
    # Strategy Parameters
    strategy_params = {}
    ticker2_symbol = None
    strategy = None
    
    if selected_strategy == 'EMA/SMA Crossover':
        col1, col2 = st.sidebar.columns(2)
        ma_type1 = col1.selectbox("MA Type 1", ['EMA', 'SMA'])
        ma_type2 = col2.selectbox("MA Type 2", ['EMA', 'SMA'])
        
        col3, col4 = st.sidebar.columns(2)
        period1 = col3.number_input("Period 1", min_value=1, value=9)
        period2 = col4.number_input("Period 2", min_value=1, value=20)
        
        crossover_type = st.sidebar.selectbox("Crossover Type", ['simple', 'strong_candle'])
        
        strategy_params = {
            'ma_type1': ma_type1,
            'ma_type2': ma_type2,
            'period1': period1,
            'period2': period2,
            'crossover_type': crossover_type
        }
        strategy = EMACrossoverStrategy(strategy_params)
    
    elif selected_strategy == 'RSI + Divergence':
        rsi_period = st.sidebar.number_input("RSI Period", min_value=5, value=14)
        strategy_params = {'rsi_period': rsi_period}
        strategy = RSIDivergenceStrategy(strategy_params)
    
    elif selected_strategy == 'Fibonacci Retracement':
        lookback = st.sidebar.number_input("Lookback Period", min_value=20, value=50)
        tolerance = st.sidebar.number_input("Level Tolerance (%)", min_value=0.1, value=0.5) / 100
        strategy_params = {'lookback': lookback, 'tolerance': tolerance}
        strategy = FibonacciStrategy(strategy_params)
    
    elif selected_strategy == 'Bollinger Bands':
        period_val = st.sidebar.number_input("Period", min_value=10, value=20)
        std_dev = st.sidebar.number_input("Std Deviation", min_value=1.0, value=2.0, step=0.1)
        strategy_params = {'period': period_val, 'std_dev': std_dev}
        strategy = BollingerBandsStrategy(strategy_params)
    
    elif selected_strategy == 'VWAP Strategy':
        threshold = st.sidebar.number_input("Threshold (%)", min_value=0.1, value=0.2, step=0.1) / 100
        strategy_params = {'threshold': threshold}
        strategy = VWAPStrategy(strategy_params)
    
    elif selected_strategy == 'Support/Resistance Breakout':
        window = st.sidebar.number_input("Window", min_value=10, value=20)
        strategy_params = {'window': window}
        strategy = SupportResistanceStrategy(strategy_params)
    
    elif selected_strategy == 'Candlestick Patterns':
        strategy = CandlestickPatternStrategy({})
    
    elif selected_strategy == 'ATR Volatility Breakout':
        atr_period = st.sidebar.number_input("ATR Period", min_value=5, value=14)
        multiplier = st.sidebar.number_input("Multiplier", min_value=1.0, value=2.0, step=0.1)
        strategy_params = {'atr_period': atr_period, 'multiplier': multiplier}
        strategy = ATRBreakoutStrategy(strategy_params)
    
    elif selected_strategy == 'MACD Strategy':
        fast = st.sidebar.number_input("Fast Period", min_value=5, value=12)
        slow = st.sidebar.number_input("Slow Period", min_value=10, value=26)
        signal = st.sidebar.number_input("Signal Period", min_value=5, value=9)
        strategy_params = {'fast': fast, 'slow': slow, 'signal': signal}
        strategy = MACDStrategy(strategy_params)
    
    elif selected_strategy == 'Stochastic Oscillator':
        period_val = st.sidebar.number_input("Period", min_value=5, value=14)
        smooth_k = st.sidebar.number_input("Smooth K", min_value=1, value=3)
        smooth_d = st.sidebar.number_input("Smooth D", min_value=1, value=3)
        strategy_params = {'period': period_val, 'smooth_k': smooth_k, 'smooth_d': smooth_d}
        strategy = StochasticStrategy(strategy_params)
    
    elif selected_strategy == 'Momentum Strategy':
        period_val = st.sidebar.number_input("Period", min_value=5, value=14)
        threshold = st.sidebar.number_input("Threshold (%)", min_value=0.5, value=2.0, step=0.1)
        strategy_params = {'period': period_val, 'threshold': threshold}
        strategy = MomentumStrategy(strategy_params)
    
    elif selected_strategy == 'Volatility Binning':
        window = st.sidebar.number_input("Window", min_value=10, value=20)
        strategy_params = {'window': window}
        strategy = VolatilityBinningStrategy(strategy_params)
    
    elif selected_strategy == 'Correlation Strategy':
        st.sidebar.markdown("**Second Ticker Selection**")
        ticker2_option = st.sidebar.selectbox("Asset Type 2", ['Preset Assets', 'Custom Ticker', 'Indian Stock'], key='ticker2')
        
        if ticker2_option == 'Preset Assets':
            ticker2 = st.sidebar.selectbox("Select Asset 2", list(PRESET_TICKERS.keys()), key='preset2')
            ticker2_symbol = PRESET_TICKERS[ticker2]
        elif ticker2_option == 'Indian Stock':
            stock_name2 = st.sidebar.text_input("Enter Stock Symbol 2 (without .NS)", key='stock2')
            ticker2_symbol = f"{stock_name2}.NS" if stock_name2 else "TCS.NS"
        else:
            ticker2_symbol = st.sidebar.text_input("Enter Ticker Symbol 2", "GOOGL", key='custom2')
        
        window = st.sidebar.number_input("Window", min_value=10, value=20)
        threshold = st.sidebar.number_input("Correlation Threshold", min_value=0.1, value=0.3, step=0.1)
        strategy_params = {'window': window, 'threshold': threshold}
        strategy = CorrelationStrategy(strategy_params)
    
    elif selected_strategy == 'Seasonality Strategy':
        strategy = SeasonalityStrategy({})
    
    elif selected_strategy == 'Pair Ratio Trading':
        st.sidebar.markdown("**Second Ticker Selection**")
        ticker2_option = st.sidebar.selectbox("Asset Type 2", ['Preset Assets', 'Custom Ticker', 'Indian Stock'], key='ticker2')
        
        if ticker2_option == 'Preset Assets':
            ticker2 = st.sidebar.selectbox("Select Asset 2", list(PRESET_TICKERS.keys()), key='preset2')
            ticker2_symbol = PRESET_TICKERS[ticker2]
        elif ticker2_option == 'Indian Stock':
            stock_name2 = st.sidebar.text_input("Enter Stock Symbol 2 (without .NS)", key='stock2')
            ticker2_symbol = f"{stock_name2}.NS" if stock_name2 else "TCS.NS"
        else:
            ticker2_symbol = st.sidebar.text_input("Enter Ticker Symbol 2", "GOOGL", key='custom2')
        
        threshold = st.sidebar.number_input("Z-Score Threshold", min_value=1.0, value=2.0, step=0.1)
        strategy_params = {'threshold': threshold}
        strategy = PairRatioStrategy(strategy_params)
    
    elif selected_strategy == 'Z-Score Mean Reversion':
        window = st.sidebar.number_input("Window", min_value=10, value=20)
        threshold = st.sidebar.number_input("Z-Score Threshold", min_value=1.0, value=2.0, step=0.1)
        strategy_params = {'window': window, 'threshold': threshold}
        strategy = ZScoreMeanReversionStrategy(strategy_params)
    
    elif selected_strategy == 'Breakout with Volume':
        period_val = st.sidebar.number_input("Donchian Period", min_value=10, value=20)
        strategy_params = {'period': period_val}
        strategy = BreakoutVolumeStrategy(strategy_params)
    
    elif selected_strategy == 'Hybrid Multi-Strategy':
        st.sidebar.markdown("**Select Strategies to Combine**")
        
        # Define available strategies for hybrid
        available_strategies = {
            'EMA Crossover': EMACrossoverStrategy({'ma_type1': 'EMA', 'ma_type2': 'EMA', 'period1': 9, 'period2': 20, 'crossover_type': 'simple'}),
            'RSI': RSIDivergenceStrategy({'rsi_period': 14}),
            'Bollinger Bands': BollingerBandsStrategy({'period': 20, 'std_dev': 2}),
            'MACD': MACDStrategy({'fast': 12, 'slow': 26, 'signal': 9}),
            'Momentum': MomentumStrategy({'period': 14, 'threshold': 2.0}),
            'Support/Resistance': SupportResistanceStrategy({'window': 20})
        }
        
        selected_hybrid_strategies = []
        for strat_name, strat_obj in available_strategies.items():
            if st.sidebar.checkbox(strat_name, value=(strat_name == 'EMA Crossover')):
                selected_hybrid_strategies.append(strat_obj)
        
        min_confirmations = st.sidebar.number_input("Minimum Confirmations", min_value=1, max_value=len(selected_hybrid_strategies), value=2)
        
        strategy_params = {'min_confirmations': min_confirmations}
        strategy = HybridStrategy(strategy_params, selected_hybrid_strategies)
    
    elif selected_strategy == 'Simple Buy':
        strategy = SimpleBuyStrategy({})
    
    elif selected_strategy == 'Simple Sell':
        strategy = SimpleSellStrategy({})
    
    # Risk Management
    st.sidebar.subheader("🛡️ Risk Management")
    
    sl_types = ['Custom Points', 'Trail SL', 'Signal Based', 'ATR Based', 
                'Previous Swing', 'Current Swing', 'Current Candle', 
                'Percentage Based', 'P&L Based']
    sl_type = st.sidebar.selectbox("Stop Loss Type", sl_types)
    
    sl_value = 0
    if sl_type in ['Custom Points', 'Trail SL', 'P&L Based']:
        sl_value = st.sidebar.number_input("SL Points", min_value=1.0, value=50.0)
    elif sl_type == 'ATR Based':
        sl_value = st.sidebar.number_input("ATR Multiplier", min_value=0.5, value=2.0, step=0.5)
    elif sl_type == 'Percentage Based':
        sl_value = st.sidebar.number_input("SL Percentage", min_value=0.1, value=2.0, step=0.1)
    
    target_types = ['Custom Points', 'Trail Target', 'Signal Based', 
                    'Risk Reward Ratio', 'Percentage Based']
    target_type = st.sidebar.selectbox("Target Type", target_types)
    
    target_value = 0
    if target_type in ['Custom Points', 'Trail Target']:
        target_value = st.sidebar.number_input("Target Points", min_value=1.0, value=100.0)
    elif target_type == 'Risk Reward Ratio':
        target_value = st.sidebar.number_input("R:R Ratio", min_value=1.0, value=2.0, step=0.5)
    elif target_type == 'Percentage Based':
        target_value = st.sidebar.number_input("Target Percentage", min_value=0.1, value=3.0, step=0.1)
    
    quantity = st.sidebar.number_input("Position Size (Quantity)", min_value=1, value=1)
    
    # Create Trading System
    trading_system = TradingSystem(
        ticker_symbol, ticker2_symbol, period, timeframe,
        strategy, sl_type, sl_value, target_type, target_value, quantity
    )
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Live Trading", "📊 Trade History", "📝 Trade Log"])
    
    with tab1:
        # Control Buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Start Trading", disabled=st.session_state.trading_active):
                if trading_system.fetch_data():
                    st.session_state.trading_active = True
                    st.session_state.iteration_count = 0
                    add_log("Trading system started")
                    st.rerun()
                else:
                    st.error("Failed to fetch data")
        
        with col2:
            if st.button("⏸️ Stop Trading", disabled=not st.session_state.trading_active):
                st.session_state.trading_active = False
                
                if st.session_state.current_position:
                    trading_system.fetch_data()
                    exit_price = trading_system.data['Close'].iloc[-1]
                    position = st.session_state.current_position
                    
                    if position['type'] == 'LONG':
                        pnl = (exit_price - position['entry_price']) * quantity
                    else:
                        pnl = (position['entry_price'] - exit_price) * quantity
                    
                    trade_record = {
                        'entry_time': position['entry_time'],
                        'exit_time': datetime.now(IST),
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': 'Manual Stop',
                        'strategy': strategy.name,
                        'quantity': quantity
                    }
                    
                    st.session_state.trade_history.append(trade_record)
                    st.session_state.current_position = None
                    add_log(f"Position closed manually at {exit_price:.2f}, P&L: {pnl:.2f}")
                
                add_log("Trading system stopped")
                st.rerun()
        
        with col3:
            if st.button("❌ Force Close Position", 
                        disabled=not st.session_state.current_position or st.session_state.trading_active):
                if st.session_state.current_position:
                    trading_system.fetch_data()
                    exit_price = trading_system.data['Close'].iloc[-1]
                    position = st.session_state.current_position
                    
                    if position['type'] == 'LONG':
                        pnl = (exit_price - position['entry_price']) * quantity
                    else:
                        pnl = (position['entry_price'] - exit_price) * quantity
                    
                    trade_record = {
                        'entry_time': position['entry_time'],
                        'exit_time': datetime.now(IST),
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': 'Force Close',
                        'strategy': strategy.name,
                        'quantity': quantity
                    }
                    
                    st.session_state.trade_history.append(trade_record)
                    st.session_state.current_position = None
                    add_log(f"Position force closed at {exit_price:.2f}, P&L: {pnl:.2f}")
                    st.rerun()
        
        # Live Trading Display
        if st.session_state.trading_active:
            st.session_state.iteration_count += 1
            
            if not trading_system.fetch_data():
                st.error("Failed to fetch data")
                st.session_state.trading_active = False
                st.rerun()
            
            current_price = trading_system.data['Close'].iloc[-1]
            signal_data = {}
            
            # Header Info
            st.markdown(f"""
            <div class='status-box'>
            <h3>🔴 LIVE - Auto-refreshing every 1.8s | Iteration: {st.session_state.iteration_count}</h3>
            <p><strong>Strategy:</strong> {strategy.name} | <strong>Timeframe:</strong> {timeframe} | 
            <strong>Period:</strong> {period} | <strong>Candles Analyzed:</strong> {len(trading_system.data)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Check for entry if no position
            if not st.session_state.current_position:
                position_type, signal_data = trading_system.check_entry()
                
                if position_type:
                    entry_price = current_price
                    sl_price, target_price = trading_system.calculate_sl_target(entry_price, position_type)
                    
                    st.session_state.current_position = {
                        'type': position_type,
                        'entry_price': entry_price,
                        'entry_time': datetime.now(IST),
                        'sl_price': sl_price,
                        'target_price': target_price,
                        'signal_data': signal_data
                    }
                    
                    add_log(f"{position_type} position opened at {entry_price:.2f}")
                    st.success(f"✅ {position_type} Entry at {entry_price:.2f}")
            
            # Monitor existing position
            if st.session_state.current_position:
                position = st.session_state.current_position
                
                # Update trailing SL/Target
                position['sl_price'] = trading_system.update_trailing_sl(position, current_price)
                position['target_price'] = trading_system.update_trailing_target(position, current_price)
                
                # Check exit
                should_exit, exit_reason = trading_system.check_exit(position, current_price)
                
                if should_exit:
                    if position['type'] == 'LONG':
                        pnl = (current_price - position['entry_price']) * quantity
                    else:
                        pnl = (position['entry_price'] - current_price) * quantity
                    
                    trade_record = {
                        'entry_time': position['entry_time'],
                        'exit_time': datetime.now(IST),
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'strategy': strategy.name,
                        'quantity': quantity
                    }
                    
                    st.session_state.trade_history.append(trade_record)
                    st.session_state.current_position = None
                    add_log(f"Position closed: {exit_reason}, P&L: {pnl:.2f}")
                    st.warning(f"⚠️ Exit: {exit_reason} | P&L: {pnl:.2f}")
                else:
                    # Display position info
                    if position['type'] == 'LONG':
                        pnl_points = current_price - position['entry_price']
                    else:
                        pnl_points = position['entry_price'] - current_price
                    
                    #pnl_p
