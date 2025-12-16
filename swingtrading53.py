import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from abc import ABC, abstractmethod
from scipy.signal import argrelextrema
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ======================== CONFIGURATION ========================
IST = pytz.timezone('Asia/Kolkata')

PRESET_ASSETS = {
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

TIMEFRAMES = ['1m', '3m', '5m', '10m', '15m', '30m', '1h', '2h', '4h', '1d', '1wk', '1mo']
PERIODS = ['1d', '5d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '3y', '5y', '6y', '10y', '15y', '20y', '25y', '30y']

TIMEFRAME_PERIOD_COMPATIBILITY = {
    '1m': ['1d', '5d'],
    '3m': ['1d', '5d'],
    '5m': ['1d', '5d', '1mo'],
    '10m': ['1d', '5d', '1mo'],
    '15m': ['1d', '5d', '1mo'],
    '30m': ['1d', '5d', '1mo'],
    '1h': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
    '2h': ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y'],
    '4h': ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '3y'],
    '1d': ['1mo', '3mo', '6mo', '1y', '2y', '3y', '5y', '6y', '10y', '15y', '20y', '25y', '30y'],
    '1wk': ['1y', '2y', '3y', '5y', '6y', '10y', '15y', '20y', '25y', '30y'],
    '1mo': ['1y', '2y', '3y', '5y', '6y', '10y', '15y', '20y', '25y', '30y']
}

# ======================== BASE STRATEGY CLASS ========================
class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy-specific indicators"""
        pass
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        """
        Generate trading signals
        Returns: (bullish_signal, bearish_signal, signal_data)
        """
        pass

# ======================== STRATEGY IMPLEMENTATIONS ========================

class EMACrossoverStrategy(BaseStrategy):
    def __init__(self, indicator1_type='EMA', indicator2_type='EMA', period1=9, period2=20):
        super().__init__(f"{indicator1_type}{period1}/{indicator2_type}{period2} Crossover")
        self.indicator1_type = indicator1_type
        self.indicator2_type = indicator2_type
        self.period1 = period1
        self.period2 = period2
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.indicator1_type == 'EMA':
            data['Indicator1'] = data['Close'].ewm(span=self.period1, adjust=False).mean()
        else:
            data['Indicator1'] = data['Close'].rolling(window=self.period1).mean()
        
        if self.indicator2_type == 'EMA':
            data['Indicator2'] = data['Close'].ewm(span=self.period2, adjust=False).mean()
        else:
            data['Indicator2'] = data['Close'].rolling(window=self.period2).mean()
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        if len(data) < max(self.period1, self.period2) + 2:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        
        prev_ind1 = data['Indicator1'].iloc[-2]
        prev_ind2 = data['Indicator2'].iloc[-2]
        curr_ind1 = data['Indicator1'].iloc[-1]
        curr_ind2 = data['Indicator2'].iloc[-1]
        
        bullish = prev_ind1 <= prev_ind2 and curr_ind1 > curr_ind2
        bearish = prev_ind1 >= prev_ind2 and curr_ind1 < curr_ind2
        
        signal_data = {
            'Indicator1': curr_ind1,
            'Indicator2': curr_ind2,
            'Indicator1_Name': f'{self.indicator1_type}{self.period1}',
            'Indicator2_Name': f'{self.indicator2_type}{self.period2}'
        }
        
        return bullish, bearish, signal_data

class PairRatioStrategy(BaseStrategy):
    def __init__(self, ticker2, zscore_threshold=2.0):
        super().__init__("Pair Ratio Trading")
        self.ticker2 = ticker2
        self.zscore_threshold = zscore_threshold
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # Fetch second ticker data
        try:
            ticker2_data = yf.download(self.ticker2, period='1mo', interval='1d', progress=False)
            if isinstance(ticker2_data.columns, pd.MultiIndex):
                ticker2_data.columns = ticker2_data.columns.droplevel(1)
            
            # Align data by reindexing
            common_dates = data.index.intersection(ticker2_data.index)
            if len(common_dates) < 20:
                return data
            
            data_aligned = data.loc[common_dates]
            ticker2_aligned = ticker2_data.loc[common_dates]
            
            # Calculate ratio
            ratio = data_aligned['Close'] / ticker2_aligned['Close']
            
            # Calculate Z-score
            rolling_mean = ratio.rolling(window=20).mean()
            rolling_std = ratio.rolling(window=20).std()
            zscore = (ratio - rolling_mean) / rolling_std
            
            # Merge back to original data
            data['Ratio'] = ratio
            data['ZScore'] = zscore
            
        except Exception as e:
            st.warning(f"Error fetching pair data: {e}")
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        if 'ZScore' not in data.columns or len(data) < 20:
            return False, False, {}
        
        curr_zscore = data['ZScore'].iloc[-1]
        
        if pd.isna(curr_zscore):
            return False, False, {}
        
        bullish = curr_zscore < -self.zscore_threshold
        bearish = curr_zscore > self.zscore_threshold
        
        signal_data = {
            'ZScore': curr_zscore,
            'Threshold': self.zscore_threshold,
            'Ratio': data['Ratio'].iloc[-1] if 'Ratio' in data.columns else None
        }
        
        return bullish, bearish, signal_data

class RSIDivergenceStrategy(BaseStrategy):
    def __init__(self, rsi_period=14, oversold=30, overbought=70):
        super().__init__("RSI + Divergence")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        if len(data) < self.rsi_period + 10:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        curr_rsi = data['RSI'].iloc[-1]
        
        # Basic RSI signals
        bullish = curr_rsi < self.oversold
        bearish = curr_rsi > self.overbought
        
        # Divergence detection (simplified)
        if len(data) >= 20:
            recent_data = data.tail(20)
            price_lows = argrelextrema(recent_data['Close'].values, np.less, order=3)[0]
            rsi_lows = argrelextrema(recent_data['RSI'].values, np.less, order=3)[0]
            
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                if recent_data['Close'].iloc[price_lows[-1]] < recent_data['Close'].iloc[price_lows[-2]]:
                    if recent_data['RSI'].iloc[rsi_lows[-1]] > recent_data['RSI'].iloc[rsi_lows[-2]]:
                        bullish = True
        
        signal_data = {
            'RSI': curr_rsi,
            'Oversold': self.oversold,
            'Overbought': self.overbought
        }
        
        return bullish, bearish, signal_data

class FibonacciRetracementStrategy(BaseStrategy):
    def __init__(self, lookback=50, tolerance=0.005):
        super().__init__("Fibonacci Retracement")
        self.lookback = lookback
        self.tolerance = tolerance
        self.fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) < self.lookback:
            return data
        
        recent = data.tail(self.lookback)
        swing_high = recent['High'].max()
        swing_low = recent['Low'].min()
        diff = swing_high - swing_low
        
        for level in self.fib_levels:
            data[f'Fib_{level}'] = swing_high - (diff * level)
        
        data['SwingHigh'] = swing_high
        data['SwingLow'] = swing_low
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        if len(data) < self.lookback or 'Fib_0.382' not in data.columns:
            return False, False, {}
        
        curr_price = data['Close'].iloc[-1]
        key_levels = [0.382, 0.5, 0.618]
        
        bullish = False
        bearish = False
        
        for level in key_levels:
            fib_price = data[f'Fib_{level}'].iloc[-1]
            if abs(curr_price - fib_price) / fib_price < self.tolerance:
                if curr_price < data['Close'].iloc[-5]:
                    bullish = True
                else:
                    bearish = True
                break
        
        signal_data = {
            'SwingHigh': data['SwingHigh'].iloc[-1],
            'SwingLow': data['SwingLow'].iloc[-1],
            'Fib_38.2': data['Fib_0.382'].iloc[-1],
            'Fib_50': data['Fib_0.5'].iloc[-1],
            'Fib_61.8': data['Fib_0.618'].iloc[-1]
        }
        
        return bullish, bearish, signal_data

class ElliottWaveStrategy(BaseStrategy):
    def __init__(self, wave_lookback=50):
        super().__init__("Elliott Wave (Simplified)")
        self.wave_lookback = wave_lookback
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) < self.wave_lookback:
            return data
        
        recent = data.tail(self.wave_lookback)
        highs = argrelextrema(recent['High'].values, np.greater, order=5)[0]
        lows = argrelextrema(recent['Low'].values, np.less, order=5)[0]
        
        extrema = sorted(list(highs) + list(lows))
        data['Wave_Extrema'] = 0
        if len(extrema) > 0:
            data.iloc[-self.wave_lookback:].iloc[extrema, data.columns.get_loc('Wave_Extrema')] = 1
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        if len(data) < self.wave_lookback:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        recent = data.tail(self.wave_lookback)
        extrema_indices = recent[recent['Wave_Extrema'] == 1].index
        
        bullish = False
        bearish = False
        
        if len(extrema_indices) >= 5:
            # Simplified wave detection
            wave_prices = recent.loc[extrema_indices, 'Close'].values[-5:]
            if wave_prices[0] < wave_prices[1] > wave_prices[2] < wave_prices[3] > wave_prices[4]:
                bullish = wave_prices[4] < wave_prices[2]
                bearish = wave_prices[4] > wave_prices[2]
        
        signal_data = {
            'Waves_Detected': len(extrema_indices),
            'Wave_Pattern': 'Detected' if len(extrema_indices) >= 5 else 'Insufficient'
        }
        
        return bullish, bearish, signal_data

class ZScoreMeanReversionStrategy(BaseStrategy):
    def __init__(self, lookback=20, threshold=2.0):
        super().__init__("Z-Score Mean Reversion")
        self.lookback = lookback
        self.threshold = threshold
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        data['Rolling_Mean'] = data['Close'].rolling(window=self.lookback).mean()
        data['Rolling_Std'] = data['Close'].rolling(window=self.lookback).std()
        data['ZScore'] = (data['Close'] - data['Rolling_Mean']) / data['Rolling_Std']
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        if len(data) < self.lookback:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        curr_zscore = data['ZScore'].iloc[-1]
        
        if pd.isna(curr_zscore):
            return False, False, {}
        
        bullish = curr_zscore < -self.threshold
        bearish = curr_zscore > self.threshold
        
        signal_data = {
            'ZScore': curr_zscore,
            'Mean': data['Rolling_Mean'].iloc[-1],
            'Threshold': self.threshold
        }
        
        return bullish, bearish, signal_data

class BreakoutVolumeStrategy(BaseStrategy):
    def __init__(self, lookback=20, volume_multiplier=1.5):
        super().__init__("Breakout with Volume")
        self.lookback = lookback
        self.volume_multiplier = volume_multiplier
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        data['Upper_Band'] = data['High'].rolling(window=self.lookback).max()
        data['Lower_Band'] = data['Low'].rolling(window=self.lookback).min()
        
        if 'Volume' in data.columns:
            data['Avg_Volume'] = data['Volume'].rolling(window=self.lookback).mean()
            data['Volume_Surge'] = data['Volume'] > (data['Avg_Volume'] * self.volume_multiplier)
        else:
            data['Volume_Surge'] = True
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        if len(data) < self.lookback:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        
        curr_close = data['Close'].iloc[-1]
        upper_band = data['Upper_Band'].iloc[-2]
        lower_band = data['Lower_Band'].iloc[-2]
        volume_surge = data['Volume_Surge'].iloc[-1]
        
        bullish = curr_close > upper_band and volume_surge
        bearish = curr_close < lower_band and volume_surge
        
        signal_data = {
            'Upper_Band': upper_band,
            'Lower_Band': lower_band,
            'Volume_Surge': volume_surge,
            'Current_Price': curr_close
        }
        
        return bullish, bearish, signal_data

class SimpleBuyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Simple Buy")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        return True, False, {'Signal': 'Always Buy'}

class SimpleSellStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Simple Sell")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        return False, True, {'Signal': 'Always Sell'}

class SupportResistanceStrategy(BaseStrategy):
    def __init__(self, lookback=50, touch_tolerance=0.002):
        super().__init__("Support/Resistance Levels")
        self.lookback = lookback
        self.touch_tolerance = touch_tolerance
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) < self.lookback:
            return data
        
        recent = data.tail(self.lookback)
        
        # Find support/resistance levels
        highs = recent['High'].values
        lows = recent['Low'].values
        
        # Find local maxima and minima
        resistance_indices = argrelextrema(highs, np.greater, order=5)[0]
        support_indices = argrelextrema(lows, np.less, order=5)[0]
        
        resistances = highs[resistance_indices] if len(resistance_indices) > 0 else []
        supports = lows[support_indices] if len(support_indices) > 0 else []
        
        # Cluster nearby levels
        if len(resistances) > 0:
            data['Resistance'] = np.mean(resistances[-3:]) if len(resistances) >= 3 else resistances[-1]
        if len(supports) > 0:
            data['Support'] = np.mean(supports[-3:]) if len(supports) >= 3 else supports[-1]
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        if len(data) < self.lookback or 'Support' not in data.columns:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        curr_price = data['Close'].iloc[-1]
        
        bullish = False
        bearish = False
        
        if 'Support' in data.columns:
            support = data['Support'].iloc[-1]
            if abs(curr_price - support) / support < self.touch_tolerance:
                if curr_price > data['Close'].iloc[-2]:
                    bullish = True
        
        if 'Resistance' in data.columns:
            resistance = data['Resistance'].iloc[-1]
            if abs(curr_price - resistance) / resistance < self.touch_tolerance:
                if curr_price < data['Close'].iloc[-2]:
                    bearish = True
        
        signal_data = {
            'Support': data.get('Support', pd.Series([None])).iloc[-1],
            'Resistance': data.get('Resistance', pd.Series([None])).iloc[-1],
            'Current_Price': curr_price
        }
        
        return bullish, bearish, signal_data

class CandlestickPatternStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Candlestick Patterns")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # Calculate candlestick properties
        data['Body'] = abs(data['Close'] - data['Open'])
        data['UpperWick'] = data['High'] - data[['Close', 'Open']].max(axis=1)
        data['LowerWick'] = data[['Close', 'Open']].min(axis=1) - data['Low']
        data['Range'] = data['High'] - data['Low']
        data['IsBullish'] = data['Close'] > data['Open']
        
        return data
    
    def detect_hammer(self, data: pd.DataFrame, idx: int) -> bool:
        body = data['Body'].iloc[idx]
        lower_wick = data['LowerWick'].iloc[idx]
        upper_wick = data['UpperWick'].iloc[idx]
        return lower_wick > 2 * body and upper_wick < body
    
    def detect_shooting_star(self, data: pd.DataFrame, idx: int) -> bool:
        body = data['Body'].iloc[idx]
        lower_wick = data['LowerWick'].iloc[idx]
        upper_wick = data['UpperWick'].iloc[idx]
        return upper_wick > 2 * body and lower_wick < body
    
    def detect_engulfing(self, data: pd.DataFrame) -> Tuple[bool, bool]:
        if len(data) < 2:
            return False, False
        
        prev_body = data['Body'].iloc[-2]
        curr_body = data['Body'].iloc[-1]
        prev_bullish = data['IsBullish'].iloc[-2]
        curr_bullish = data['IsBullish'].iloc[-1]
        
        bullish_engulfing = not prev_bullish and curr_bullish and curr_body > prev_body
        bearish_engulfing = prev_bullish and not curr_bullish and curr_body > prev_body
        
        return bullish_engulfing, bearish_engulfing
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        if len(data) < 3:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        
        hammer = self.detect_hammer(data, -1)
        shooting_star = self.detect_shooting_star(data, -1)
        bullish_eng, bearish_eng = self.detect_engulfing(data)
        
        bullish = hammer or bullish_eng
        bearish = shooting_star or bearish_eng
        
        pattern = []
        if hammer: pattern.append("Hammer")
        if shooting_star: pattern.append("Shooting Star")
        if bullish_eng: pattern.append("Bullish Engulfing")
        if bearish_eng: pattern.append("Bearish Engulfing")
        
        signal_data = {
            'Pattern': ', '.join(pattern) if pattern else 'None',
            'Body_Size': data['Body'].iloc[-1],
            'Range': data['Range'].iloc[-1]
        }
        
        return bullish, bearish, signal_data

class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, period=20, std_dev=2):
        super().__init__("Bollinger Bands")
        self.period = period
        self.std_dev = std_dev
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        data['BB_Middle'] = data['Close'].rolling(window=self.period).mean()
        data['BB_Std'] = data['Close'].rolling(window=self.period).std()
        data['BB_Upper'] = data['BB_Middle'] + (self.std_dev * data['BB_Std'])
        data['BB_Lower'] = data['BB_Middle'] - (self.std_dev * data['BB_Std'])
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        if len(data) < self.period + 1:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        
        curr_close = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        bb_upper = data['BB_Upper'].iloc[-1]
        bb_lower = data['BB_Lower'].iloc[-1]
        bb_middle = data['BB_Middle'].iloc[-1]
        
        # Mean reversion signals
        bullish = prev_close <= bb_lower and curr_close > bb_lower
        bearish = prev_close >= bb_upper and curr_close < bb_upper
        
        signal_data = {
            'BB_Upper': bb_upper,
            'BB_Middle': bb_middle,
            'BB_Lower': bb_lower,
            'BB_Width': data['BB_Width'].iloc[-1],
            'Distance_to_Upper': ((bb_upper - curr_close) / curr_close) * 100,
            'Distance_to_Lower': ((curr_close - bb_lower) / curr_close) * 100
        }
        
        return bullish, bearish, signal_data

class KeltnerChannelStrategy(BaseStrategy):
    def __init__(self, ema_period=20, atr_period=10, multiplier=2):
        super().__init__("Keltner Channel")
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # EMA
        data['KC_Middle'] = data['Close'].ewm(span=self.ema_period, adjust=False).mean()
        
        # ATR
        data['TR'] = np.maximum(
            data['High'] - data['Low'],
            np.maximum(
                abs(data['High'] - data['Close'].shift()),
                abs(data['Low'] - data['Close'].shift())
            )
        )
        data['ATR'] = data['TR'].rolling(window=self.atr_period).mean()
        
        # Keltner Channels
        data['KC_Upper'] = data['KC_Middle'] + (self.multiplier * data['ATR'])
        data['KC_Lower'] = data['KC_Middle'] - (self.multiplier * data['ATR'])
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        if len(data) < max(self.ema_period, self.atr_period) + 1:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        
        curr_close = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        kc_upper = data['KC_Upper'].iloc[-1]
        kc_lower = data['KC_Lower'].iloc[-1]
        
        bullish = prev_close <= kc_lower and curr_close > kc_lower
        bearish = prev_close >= kc_upper and curr_close < kc_upper
        
        signal_data = {
            'KC_Upper': kc_upper,
            'KC_Middle': data['KC_Middle'].iloc[-1],
            'KC_Lower': kc_lower,
            'ATR': data['ATR'].iloc[-1]
        }
        
        return bullish, bearish, signal_data

class InsideBarStrategy(BaseStrategy):
    def __init__(self, min_bars=2):
        super().__init__("Inside Bar")
        self.min_bars = min_bars
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        data['IsInsideBar'] = (data['High'] < data['High'].shift(1)) & (data['Low'] > data['Low'].shift(1))
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        if len(data) < 3:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        
        # Check if previous bar was inside bar
        is_inside = data['IsInsideBar'].iloc[-2]
        curr_close = data['Close'].iloc[-1]
        mother_high = data['High'].iloc[-3]
        mother_low = data['Low'].iloc[-3]
        
        bullish = is_inside and curr_close > mother_high
        bearish = is_inside and curr_close < mother_low
        
        signal_data = {
            'Inside_Bar_Detected': is_inside,
            'Mother_High': mother_high if len(data) >= 3 else None,
            'Mother_Low': mother_low if len(data) >= 3 else None
        }
        
        return bullish, bearish, signal_data

class VolatilityBreakoutStrategy(BaseStrategy):
    def __init__(self, atr_period=14, breakout_multiplier=1.5):
        super().__init__("Volatility Breakout")
        self.atr_period = atr_period
        self.breakout_multiplier = breakout_multiplier
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        data['TR'] = np.maximum(
            data['High'] - data['Low'],
            np.maximum(
                abs(data['High'] - data['Close'].shift()),
                abs(data['Low'] - data['Close'].shift())
            )
        )
        data['ATR'] = data['TR'].rolling(window=self.atr_period).mean()
        data['VolBreakout_Upper'] = data['Close'].shift(1) + (data['ATR'] * self.breakout_multiplier)
        data['VolBreakout_Lower'] = data['Close'].shift(1) - (data['ATR'] * self.breakout_multiplier)
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        if len(data) < self.atr_period + 1:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        
        curr_close = data['Close'].iloc[-1]
        upper_level = data['VolBreakout_Upper'].iloc[-1]
        lower_level = data['VolBreakout_Lower'].iloc[-1]
        
        bullish = curr_close > upper_level
        bearish = curr_close < lower_level
        
        signal_data = {
            'ATR': data['ATR'].iloc[-1],
            'Upper_Breakout': upper_level,
            'Lower_Breakout': lower_level,
            'Volatility_Pct': (data['ATR'].iloc[-1] / curr_close) * 100
        }
        
        return bullish, bearish, signal_data

class CorrelationStrategy(BaseStrategy):
    def __init__(self, ticker2, correlation_period=20, threshold=0.7):
        super().__init__("Correlation Based")
        self.ticker2 = ticker2
        self.correlation_period = correlation_period
        self.threshold = threshold
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            ticker2_data = yf.download(self.ticker2, period='1mo', interval='1d', progress=False)
            if isinstance(ticker2_data.columns, pd.MultiIndex):
                ticker2_data.columns = ticker2_data.columns.droplevel(1)
            
            common_dates = data.index.intersection(ticker2_data.index)
            if len(common_dates) < self.correlation_period:
                return data
            
            data_aligned = data.loc[common_dates, 'Close']
            ticker2_aligned = ticker2_data.loc[common_dates, 'Close']
            
            # Calculate rolling correlation
            correlation = data_aligned.rolling(window=self.correlation_period).corr(ticker2_aligned)
            data['Correlation'] = correlation
            
        except Exception as e:
            st.warning(f"Error in correlation calculation: {e}")
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        if 'Correlation' not in data.columns or len(data) < self.correlation_period:
            return False, False, {}
        
        curr_corr = data['Correlation'].iloc[-1]
        
        if pd.isna(curr_corr):
            return False, False, {}
        
        # Trade when correlation breaks
        bullish = curr_corr < -self.threshold  # Inverse correlation
        bearish = curr_corr > self.threshold and data['Close'].iloc[-1] < data['Close'].iloc[-5]
        
        signal_data = {
            'Correlation': curr_corr,
            'Threshold': self.threshold,
            'Correlation_Strength': 'Strong' if abs(curr_corr) > 0.8 else 'Moderate' if abs(curr_corr) > 0.5 else 'Weak'
        }
        
        return bullish, bearish, signal_data

class HybridStrategy(BaseStrategy):
    def __init__(self, strategies: List[BaseStrategy], min_confirmations=2):
        super().__init__(f"Hybrid ({min_confirmations}/{len(strategies)} confirmations)")
        self.strategies = strategies
        self.min_confirmations = min_confirmations
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        for strategy in self.strategies:
            data = strategy.calculate_indicators(data)
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        bullish_count = 0
        bearish_count = 0
        all_signals = {}
        
        for i, strategy in enumerate(self.strategies):
            bull, bear, sig_data = strategy.generate_signal(data)
            if bull:
                bullish_count += 1
            if bear:
                bearish_count += 1
            all_signals[f'Strategy_{i+1}'] = strategy.name
            all_signals[f'S{i+1}_Signal'] = 'Bullish' if bull else 'Bearish' if bear else 'Neutral'
        
        bullish = bullish_count >= self.min_confirmations
        bearish = bearish_count >= self.min_confirmations
        
        all_signals['Bullish_Confirmations'] = bullish_count
        all_signals['Bearish_Confirmations'] = bearish_count
        
        return bullish, bearish, all_signals

class SeasonalityStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Seasonality Based")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        data['Month'] = data.index.month
        data['DayOfWeek'] = data.index.dayofweek
        data['Hour'] = data.index.hour
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        if len(data) < 30:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        
        curr_month = data['Month'].iloc[-1]
        curr_day = data['DayOfWeek'].iloc[-1]
        
        # Simple seasonality rules (can be customized)
        bullish = curr_month in [11, 12, 1] or curr_day == 0  # Nov, Dec, Jan, Monday
        bearish = curr_month in [5, 6, 9] or curr_day == 4  # May, June, Sept, Friday
        
        signal_data = {
            'Month': curr_month,
            'Day_of_Week': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][curr_day],
            'Seasonal_Pattern': 'Bullish Season' if bullish else 'Bearish Season' if bearish else 'Neutral'
        }
        
        return bullish, bearish, signal_data

class MomentumReversalStrategy(BaseStrategy):
    def __init__(self, rsi_period=14, roc_period=10):
        super().__init__("Momentum Reversal")
        self.rsi_period = rsi_period
        self.roc_period = roc_period
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Rate of Change
        data['ROC'] = ((data['Close'] - data['Close'].shift(self.roc_period)) / data['Close'].shift(self.roc_period)) * 100
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        if len(data) < max(self.rsi_period, self.roc_period) + 5:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        
        rsi = data['RSI'].iloc[-1]
        roc = data['ROC'].iloc[-1]
        rsi_prev = data['RSI'].iloc[-2]
        
        # Reversal signals
        bullish = rsi < 30 and rsi > rsi_prev and roc < 0
        bearish = rsi > 70 and rsi < rsi_prev and roc > 0
        
        signal_data = {
            'RSI': rsi,
            'ROC': roc,
            'RSI_Trend': 'Rising' if rsi > rsi_prev else 'Falling',
            'Momentum': 'Strong' if abs(roc) > 5 else 'Moderate' if abs(roc) > 2 else 'Weak'
        }
        
        return bullish, bearish, signal_data

class SMCStrategy(BaseStrategy):
    def __init__(self, liquidity_lookback=20):
        super().__init__("Smart Money Concepts (SMC)")
        self.liquidity_lookback = liquidity_lookback
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) < self.liquidity_lookback:
            return data
        
        # Find liquidity zones (swing highs/lows)
        highs = data['High'].rolling(window=self.liquidity_lookback).max()
        lows = data['Low'].rolling(window=self.liquidity_lookback).min()
        
        data['Liquidity_High'] = highs
        data['Liquidity_Low'] = lows
        
        # Order blocks (simplified)
        data['Bullish_OB'] = data['Low'].shift(1)
        data['Bearish_OB'] = data['High'].shift(1)
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        if len(data) < self.liquidity_lookback + 2:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        
        curr_price = data['Close'].iloc[-1]
        liq_high = data['Liquidity_High'].iloc[-1]
        liq_low = data['Liquidity_Low'].iloc[-1]
        
        # Liquidity sweep signals
        bullish = curr_price < liq_low and data['Close'].iloc[-1] > data['Open'].iloc[-1]
        bearish = curr_price > liq_high and data['Close'].iloc[-1] < data['Open'].iloc[-1]
        
        signal_data = {
            'Liquidity_High': liq_high,
            'Liquidity_Low': liq_low,
            'Price_Position': 'Above Liquidity' if curr_price > liq_high else 'Below Liquidity' if curr_price < liq_low else 'Within Range'
        }
        
        return bullish, bearish, signal_data

class VWAPStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("VWAP Based")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'Volume' not in data.columns:
            data['VWAP'] = data['Close']
            return data
        
        data['Typical_Price'] = (data['High'] + data['Low'] + data['Close']) / 3
        data['TP_Volume'] = data['Typical_Price'] * data['Volume']
        data['VWAP'] = data['TP_Volume'].cumsum() / data['Volume'].cumsum()
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        if len(data) < 10:
            return False, False, {}
        
        data = self.calculate_indicators(data)
        
        curr_price = data['Close'].iloc[-1]
        vwap = data['VWAP'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        prev_vwap = data['VWAP'].iloc[-2]
        
        bullish = prev_price < prev_vwap and curr_price > vwap
        bearish = prev_price > prev_vwap and curr_price < vwap
        
        distance_pct = ((curr_price - vwap) / vwap) * 100
        
        signal_data = {
            'VWAP': vwap,
            'Distance_from_VWAP': distance_pct,
            'Position': 'Above VWAP' if curr_price > vwap else 'Below VWAP'
        }
        
        return bullish, bearish, signal_data

# ======================== TRADING SYSTEM ========================

class TradingSystem:
    def __init__(self, ticker: str, timeframe: str, period: str, strategy: BaseStrategy,
                 sl_type: str, sl_value: float, target_type: str, target_value: float, 
                 quantity: int = 1, ticker2: Optional[str] = None):
        self.ticker = ticker
        self.ticker2 = ticker2
        self.timeframe = timeframe
        self.period = period
        self.strategy = strategy
        self.sl_type = sl_type
        self.sl_value = sl_value
        self.target_type = target_type
        self.target_value = target_value
        self.quantity = quantity
        self.data = None
        self.last_fetch = None
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch market data with rate limiting"""
        try:
            # Rate limiting
            if self.last_fetch:
                elapsed = (datetime.now() - self.last_fetch).total_seconds()
                if elapsed < 1.5:
                    time.sleep(1.5 - elapsed)
            
            data = yf.download(self.ticker, period=self.period, interval=self.timeframe, progress=False)
            
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # Convert to IST
            if data.index.tzinfo is None:
                data.index = data.index.tz_localize('UTC').tz_convert(IST)
            else:
                data.index = data.index.tz_convert(IST)
            
            self.data = data
            self.last_fetch = datetime.now()
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
    
    def calculate_sl_target(self, entry_price: float, position_type: str) -> Tuple[float, float]:
        """Calculate stop loss and target prices"""
        data = self.data
        
        # Calculate Stop Loss
        if self.sl_type == 'Custom Points':
            sl_offset = self.sl_value
        elif self.sl_type == 'ATR Based':
            tr = max(
                data['High'].iloc[-1] - data['Low'].iloc[-1],
                abs(data['High'].iloc[-1] - data['Close'].iloc[-2]),
                abs(data['Low'].iloc[-1] - data['Close'].iloc[-2])
            )
            atr = data['Close'].rolling(14).std().iloc[-1] if len(data) >= 14 else tr
            sl_offset = atr * self.sl_value  # sl_value is multiplier here
        elif self.sl_type == 'Previous Swing':
            swings = argrelextrema(data['Low' if position_type == 'LONG' else 'High'].values, 
                                  np.less if position_type == 'LONG' else np.greater, order=5)[0]
            if len(swings) > 0:
                swing_price = data['Low' if position_type == 'LONG' else 'High'].iloc[swings[-1]]
                sl_offset = abs(entry_price - swing_price)
            else:
                sl_offset = self.sl_value
        elif self.sl_type == 'Current Candle':
            sl_offset = abs(entry_price - data['Low' if position_type == 'LONG' else 'High'].iloc[-1])
        elif self.sl_type == 'Percentage Based':
            sl_offset = entry_price * (self.sl_value / 100)
        else:  # Trail SL, Signal Based
            sl_offset = self.sl_value
        
        # Calculate Target
        if self.target_type == 'Custom Points':
            target_offset = self.target_value
        elif self.target_type == 'Risk Reward Ratio':
            target_offset = sl_offset * self.target_value  # target_value is RR ratio
        elif self.target_type == 'Percentage Based':
            target_offset = entry_price * (self.target_value / 100)
        else:  # Trail Target, Signal Based
            target_offset = self.target_value
        
        if position_type == 'LONG':
            sl = entry_price - sl_offset
            target = entry_price + target_offset
        else:  # SHORT
            sl = entry_price + sl_offset
            target = entry_price - target_offset
        
        return sl, target
    
    def update_trailing_sl(self, position: Dict, current_price: float) -> float:
        """Update trailing stop loss"""
        entry_price = position['entry_price']
        current_sl = position['sl']
        position_type = position['type']
        
        if position_type == 'LONG':
            if current_price > entry_price:
                new_sl = current_price - self.sl_value
                if new_sl > current_sl:
                    return new_sl
        else:  # SHORT
            if current_price < entry_price:
                new_sl = current_price + self.sl_value
                if new_sl < current_sl:
                    return new_sl
        
        return current_sl
    
    def check_exit_conditions(self, position: Dict, current_price: float, 
                            bullish_signal: bool, bearish_signal: bool) -> Tuple[bool, str]:
        """Check if exit conditions are met"""
        position_type = position['type']
        sl = position['sl']
        target = position['target']
        
        # Check SL
        if position_type == 'LONG' and current_price <= sl:
            return True, 'Stop Loss Hit'
        elif position_type == 'SHORT' and current_price >= sl:
            return True, 'Stop Loss Hit'
        
        # Check Target
        if position_type == 'LONG' and current_price >= target:
            return True, 'Target Reached'
        elif position_type == 'SHORT' and current_price <= target:
            return True, 'Target Reached'
        
        # Check Signal-based exit
        if self.sl_type == 'Signal Based' or self.target_type == 'Signal Based':
            if position_type == 'LONG' and bearish_signal:
                return True, 'Opposite Signal'
            elif position_type == 'SHORT' and bullish_signal:
                return True, 'Opposite Signal'
        
        return False, ''
    
    def get_market_status(self, position: Dict, current_price: float) -> str:
        """Get market movement status text"""
        entry_price = position['entry_price']
        position_type = position['type']
        
        if position_type == 'LONG':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        sl_distance = abs(current_price - position['sl'])
        target_distance = abs(position['target'] - current_price)
        
        status = ""
        if pnl_pct > 0.5:
            status = "🚀 Strong momentum in favor"
        elif pnl_pct > 0:
            status = "📈 Moving gradually in favor"
        else:
            status = "⚠️ In loss - monitoring for reversal or SL hit"
        
        if 'Trail' in self.sl_type and pnl_pct > 0:
            status += " | 🔄 Trailing SL active"
        
        status += f" | 🛡️ SL Distance: {sl_distance:.2f} | 🎯 Target Distance: {target_distance:.2f}"
        
        return status
    
    def get_trade_guidance(self, position: Dict, current_price: float, signal_data: Dict) -> str:
        """Provide 100-word trade guidance based on market movement"""
        entry_price = position['entry_price']
        position_type = position['type']
        
        if position_type == 'LONG':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            movement = "upward" if current_price > entry_price else "downward"
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
            movement = "downward" if current_price < entry_price else "upward"
        
        duration = (datetime.now(IST) - position['entry_time']).total_seconds() / 60
        
        guidance = f"**Trade Update:** This {position_type} position entered at {entry_price:.2f} is currently "
        
        if pnl_pct > 1:
            guidance += f"performing excellently with {pnl_pct:.2f}% profit. Price is moving {movement} as expected. "
            guidance += "Consider trailing your stop loss to lock in gains. The momentum is in your favor. "
            guidance += "Avoid premature exit; let the trade run toward target unless reversal signals appear. "
            guidance += "Stay patient and disciplined."
        elif pnl_pct > 0:
            guidance += f"showing modest gains ({pnl_pct:.2f}%). Movement is gradual but positive. "
            guidance += "This is typical consolidation before potential expansion. "
            guidance += "Maintain position with original stop loss. Don't exit prematurely due to impatience. "
            guidance += "Watch for acceleration or reversal patterns. Patience is key in trending markets."
        elif pnl_pct > -0.5:
            guidance += f"at breakeven ({pnl_pct:.2f}%). Price is oscillating near entry. "
            guidance += "This is normal market behavior - consolidation before direction. "
            guidance += "Hold your position as per plan. Avoid emotional decisions. "
            guidance += "Your stop loss provides protection. Let the setup play out completely."
        else:
            guidance += f"in drawdown ({pnl_pct:.2f}%). Price moved against entry temporarily. "
            guidance += "This happens in trading. Your stop loss is your safety net. "
            guidance += "Don't panic or average down. If SL hits, accept the loss - it's part of the process. "
            guidance += "Never move stop loss further away. Trust your risk management plan."
        
        # Add strategy-specific context
        if hasattr(self.strategy, 'name'):
            if 'EMA' in self.strategy.name or 'SMA' in self.strategy.name:
                guidance += f" Moving averages show trend alignment is {'favorable' if pnl_pct > 0 else 'weakening'}."
            elif 'RSI' in self.strategy.name:
                rsi_val = signal_data.get('RSI', 50)
                guidance += f" RSI at {rsi_val:.1f} indicates {'overbought' if rsi_val > 70 else 'oversold' if rsi_val < 30 else 'neutral'} conditions."
            elif 'Bollinger' in self.strategy.name:
                guidance += f" Price is {'expanding toward bands' if abs(pnl_pct) > 0.5 else 'within bands'} - volatility context."
        
        return guidance
    
    def get_strategy_parameters_summary(self) -> Dict:
        """Get comprehensive summary of strategy parameters"""
        if self.data is None or len(self.data) == 0:
            return {}
        
        data = self.strategy.calculate_indicators(self.data)
        
        summary = {
            'Strategy_Name': self.strategy.name,
            'Current_Values': {},
            'Historical_Stats': {}
        }
        
        # Collect current indicator values
        if 'Indicator1' in data.columns:
            ind1_val = data['Indicator1'].iloc[-1]
            summary['Current_Values']['Fast_Indicator'] = f"{ind1_val:.2f}"
            summary['Historical_Stats']['Fast_Avg'] = f"{data['Indicator1'].mean():.2f}"
            summary['Historical_Stats']['Fast_Min'] = f"{data['Indicator1'].min():.2f}"
            summary['Historical_Stats']['Fast_Max'] = f"{data['Indicator1'].max():.2f}"
        
        if 'Indicator2' in data.columns:
            ind2_val = data['Indicator2'].iloc[-1]
            summary['Current_Values']['Slow_Indicator'] = f"{ind2_val:.2f}"
            summary['Historical_Stats']['Slow_Avg'] = f"{data['Indicator2'].mean():.2f}"
            summary['Historical_Stats']['Slow_Min'] = f"{data['Indicator2'].min():.2f}"
            summary['Historical_Stats']['Slow_Max'] = f"{data['Indicator2'].max():.2f}"
        
        if 'RSI' in data.columns:
            rsi_val = data['RSI'].iloc[-1]
            summary['Current_Values']['RSI'] = f"{rsi_val:.2f}"
            summary['Historical_Stats']['RSI_Avg'] = f"{data['RSI'].mean():.2f}"
            summary['Historical_Stats']['RSI_Min'] = f"{data['RSI'].min():.2f}"
            summary['Historical_Stats']['RSI_Max'] = f"{data['RSI'].max():.2f}"
        
        if 'ATR' in data.columns:
            atr_val = data['ATR'].iloc[-1]
            summary['Current_Values']['ATR'] = f"{atr_val:.2f}"
            summary['Historical_Stats']['ATR_Avg'] = f"{data['ATR'].mean():.2f}"
            summary['Historical_Stats']['ATR_Min'] = f"{data['ATR'].min():.2f}"
            summary['Historical_Stats']['ATR_Max'] = f"{data['ATR'].max():.2f}"
        
        if 'BB_Upper' in data.columns:
            summary['Current_Values']['BB_Upper'] = f"{data['BB_Upper'].iloc[-1]:.2f}"
            summary['Current_Values']['BB_Middle'] = f"{data['BB_Middle'].iloc[-1]:.2f}"
            summary['Current_Values']['BB_Lower'] = f"{data['BB_Lower'].iloc[-1]:.2f}"
            summary['Current_Values']['BB_Width'] = f"{data['BB_Width'].iloc[-1]:.2f}"
        
        if 'VWAP' in data.columns:
            summary['Current_Values']['VWAP'] = f"{data['VWAP'].iloc[-1]:.2f}"
        
        if 'ZScore' in data.columns:
            summary['Current_Values']['ZScore'] = f"{data['ZScore'].iloc[-1]:.2f}"
        
        # Price statistics
        summary['Current_Values']['Current_Price'] = f"{data['Close'].iloc[-1]:.2f}"
        summary['Historical_Stats']['Price_Avg'] = f"{data['Close'].mean():.2f}"
        summary['Historical_Stats']['Price_Min'] = f"{data['Close'].min():.2f}"
        summary['Historical_Stats']['Price_Max'] = f"{data['Close'].max():.2f}"
        summary['Historical_Stats']['Price_Volatility'] = f"{data['Close'].std():.2f}"
        
        # Volume statistics (if available)
        if 'Volume' in data.columns:
            summary['Current_Values']['Current_Volume'] = f"{data['Volume'].iloc[-1]:,.0f}"
            summary['Historical_Stats']['Volume_Avg'] = f"{data['Volume'].mean():,.0f}"
        
        return summary
    
    def analyze_trade(self, trade: Dict) -> str:
        """AI-powered trade analysis"""
        pnl_pct = trade['pnl_pct']
        duration = trade['duration']
        exit_reason = trade['exit_reason']
        
        analysis = f"**Trade Analysis:**\n\n"
        
        # Performance
        if pnl_pct > 0:
            analysis += f"✅ **Profitable Trade** (+{pnl_pct:.2f}%)\n"
            if pnl_pct > 2:
                analysis += "Excellent execution! Strong profit captured.\n"
            else:
                analysis += "Small profit - consider holding longer for better gains.\n"
        else:
            analysis += f"❌ **Loss Trade** ({pnl_pct:.2f}%)\n"
            analysis += "Loss within acceptable range. Review entry timing.\n"
        
        # Exit quality
        analysis += f"\n**Exit Reason:** {exit_reason}\n"
        if exit_reason == 'Target Reached':
            analysis += "Perfect exit - target achieved as planned.\n"
        elif exit_reason == 'Stop Loss Hit':
            analysis += "SL protected capital. Consider wider SL or better entry.\n"
        elif exit_reason == 'Opposite Signal':
            analysis += "Strategy signaled reversal - good dynamic exit.\n"
        
        # Duration
        analysis += f"\n**Duration:** {duration}\n"
        
        # Recommendations
        analysis += "\n**Recommendations:**\n"
        if pnl_pct < 0:
            analysis += "- Review entry conditions - wait for stronger confirmation\n"
            analysis += "- Consider adjusting SL/Target ratio\n"
        else:
            analysis += "- Maintain current risk management\n"
            analysis += "- Consider scaling up position size gradually\n"
        
        analysis += f"- Strategy: {self.strategy.name} performed {'well' if pnl_pct > 0 else 'poorly'} in this trade\n"
        
        return analysis

# ======================== UI STYLING ========================

def apply_custom_css():
    st.markdown("""
        <style>
        .main {
            background-color: #0e1117;
        }
        .stMetric {
            #background-color: #1e2130;
            padding: 10px;
            border-radius: 5px;
        }
        .profit {
            color: #00ff00;
            font-weight: bold;
        }
        .loss {
            color: #ff0000;
            font-weight: bold;
        }
        .status-box {
            #background-color: #1e2130;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .trade-log {
            height: 600px;
            overflow-y: scroll;
            #background-color: #1e2130;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            color:white
        }
        </style>
    """, unsafe_allow_html=True)

# ======================== STREAMLIT APP ========================

def main():
    st.set_page_config(page_title="Algorithmic Trading System", layout="wide", initial_sidebar_state="expanded")
    apply_custom_css()
    
    st.title("🚀 Professional Algorithmic Trading System")
    
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
    if 'trading_system' not in st.session_state:
        st.session_state.trading_system = None
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Trading Configuration")
        
        # Asset Selection
        st.subheader("Asset Selection")
        asset_type = st.radio("Select Type:", ["Preset Assets", "Custom Ticker", "Indian Stock"])
        
        if asset_type == "Preset Assets":
            selected_asset = st.selectbox("Asset:", list(PRESET_ASSETS.keys()))
            ticker = PRESET_ASSETS[selected_asset]
        elif asset_type == "Custom Ticker":
            ticker = st.text_input("Ticker Symbol:", "AAPL")
        else:
            stock_name = st.text_input("Stock Name (e.g., RELIANCE):", "RELIANCE")
            ticker = f"{stock_name}.NS"
        
        # Timeframe & Period
        st.subheader("Timeframe & Period")
        timeframe = st.selectbox("Timeframe:", TIMEFRAMES, index=TIMEFRAMES.index('1m'))
        
        compatible_periods = TIMEFRAME_PERIOD_COMPATIBILITY.get(timeframe, PERIODS)
        period = st.selectbox("Period:", compatible_periods, index=0)
        
        # Strategy Selection
        st.subheader("Strategy Configuration")
        strategy_type = st.selectbox("Strategy:", [
            "EMA/SMA Crossover",
            "Pair Ratio Trading",
            "RSI + Divergence",
            "Fibonacci Retracement",
            "Elliott Wave",
            "Z-Score Mean Reversion",
            "Breakout with Volume",
            "Support/Resistance Levels",
            "Candlestick Patterns",
            "Bollinger Bands",
            "Keltner Channel",
            "Inside Bar",
            "Volatility Breakout",
            "Correlation Based",
            "Hybrid Strategy",
            "Seasonality Based",
            "Momentum Reversal",
            "Smart Money Concepts (SMC)",
            "VWAP Based",
            "Simple Buy",
            "Simple Sell"
        ])
        
        ticker2 = None
        
        # Strategy-specific parameters
        if strategy_type == "EMA/SMA Crossover":
            ind1_type = st.selectbox("Indicator 1 Type:", ["EMA", "SMA"])
            ind2_type = st.selectbox("Indicator 2 Type:", ["EMA", "SMA"])
            period1 = st.number_input("Period 1:", min_value=2, value=9)
            period2 = st.number_input("Period 2:", min_value=2, value=20)
            strategy = EMACrossoverStrategy(ind1_type, ind2_type, period1, period2)
        
        elif strategy_type == "Pair Ratio Trading":
            asset_type2 = st.radio("Second Asset Type:", ["Preset Assets", "Custom Ticker", "Indian Stock"])
            if asset_type2 == "Preset Assets":
                selected_asset2 = st.selectbox("Second Asset:", list(PRESET_ASSETS.keys()))
                ticker2 = PRESET_ASSETS[selected_asset2]
            elif asset_type2 == "Custom Ticker":
                ticker2 = st.text_input("Second Ticker Symbol:", "^NSEBANK")
            else:
                stock_name2 = st.text_input("Second Stock Name:", "INFY")
                ticker2 = f"{stock_name2}.NS"
            
            zscore_threshold = st.number_input("Z-Score Threshold:", min_value=0.5, value=2.0, step=0.1)
            strategy = PairRatioStrategy(ticker2, zscore_threshold)
        
        elif strategy_type == "RSI + Divergence":
            rsi_period = st.number_input("RSI Period:", min_value=2, value=14)
            oversold = st.number_input("Oversold Level:", min_value=10, max_value=50, value=30)
            overbought = st.number_input("Overbought Level:", min_value=50, max_value=90, value=70)
            strategy = RSIDivergenceStrategy(rsi_period, oversold, overbought)
        
        elif strategy_type == "Fibonacci Retracement":
            lookback = st.number_input("Lookback Period:", min_value=20, value=50)
            tolerance = st.number_input("Level Tolerance (%):", min_value=0.1, value=0.5, step=0.1) / 100
            strategy = FibonacciRetracementStrategy(lookback, tolerance)
        
        elif strategy_type == "Elliott Wave":
            wave_lookback = st.number_input("Wave Lookback:", min_value=30, value=50)
            strategy = ElliottWaveStrategy(wave_lookback)
        
        elif strategy_type == "Z-Score Mean Reversion":
            lookback = st.number_input("Lookback Period:", min_value=10, value=20)
            threshold = st.number_input("Z-Score Threshold:", min_value=0.5, value=2.0, step=0.1)
            strategy = ZScoreMeanReversionStrategy(lookback, threshold)
        
        elif strategy_type == "Breakout with Volume":
            lookback = st.number_input("Lookback Period:", min_value=10, value=20)
            volume_mult = st.number_input("Volume Multiplier:", min_value=1.0, value=1.5, step=0.1)
            strategy = BreakoutVolumeStrategy(lookback, volume_mult)
        
        elif strategy_type == "Support/Resistance Levels":
            lookback = st.number_input("Lookback Period:", min_value=20, value=50)
            tolerance = st.number_input("Touch Tolerance (%):", min_value=0.1, value=0.2, step=0.1) / 100
            strategy = SupportResistanceStrategy(lookback, tolerance)
        
        elif strategy_type == "Candlestick Patterns":
            strategy = CandlestickPatternStrategy()
        
        elif strategy_type == "Bollinger Bands":
            bb_period = st.number_input("BB Period:", min_value=5, value=20)
            bb_std = st.number_input("Standard Deviations:", min_value=1.0, value=2.0, step=0.1)
            strategy = BollingerBandsStrategy(bb_period, bb_std)
        
        elif strategy_type == "Keltner Channel":
            ema_period = st.number_input("EMA Period:", min_value=5, value=20)
            atr_period = st.number_input("ATR Period:", min_value=5, value=10)
            multiplier = st.number_input("ATR Multiplier:", min_value=1.0, value=2.0, step=0.1)
            strategy = KeltnerChannelStrategy(ema_period, atr_period, multiplier)
        
        elif strategy_type == "Inside Bar":
            min_bars = st.number_input("Min Inside Bars:", min_value=1, value=2)
            strategy = InsideBarStrategy(min_bars)
        
        elif strategy_type == "Volatility Breakout":
            atr_period = st.number_input("ATR Period:", min_value=5, value=14)
            breakout_mult = st.number_input("Breakout Multiplier:", min_value=1.0, value=1.5, step=0.1)
            strategy = VolatilityBreakoutStrategy(atr_period, breakout_mult)
        
        elif strategy_type == "Correlation Based":
            asset_type2 = st.radio("Second Asset Type:", ["Preset Assets", "Custom Ticker", "Indian Stock"])
            if asset_type2 == "Preset Assets":
                selected_asset2 = st.selectbox("Second Asset:", list(PRESET_ASSETS.keys()))
                ticker2 = PRESET_ASSETS[selected_asset2]
            elif asset_type2 == "Custom Ticker":
                ticker2 = st.text_input("Second Ticker Symbol:", "^NSEBANK")
            else:
                stock_name2 = st.text_input("Second Stock Name:", "INFY")
                ticker2 = f"{stock_name2}.NS"
            
            corr_period = st.number_input("Correlation Period:", min_value=10, value=20)
            corr_threshold = st.number_input("Correlation Threshold:", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
            strategy = CorrelationStrategy(ticker2, corr_period, corr_threshold)
        
        elif strategy_type == "Hybrid Strategy":
            st.write("Select strategies to combine:")
            use_ema = st.checkbox("EMA Crossover")
            use_rsi = st.checkbox("RSI")
            use_bb = st.checkbox("Bollinger Bands")
            use_vol = st.checkbox("Volume Breakout")
            
            min_conf = st.number_input("Min Confirmations:", min_value=1, value=2)
            
            sub_strategies = []
            if use_ema:
                sub_strategies.append(EMACrossoverStrategy('EMA', 'EMA', 9, 20))
            if use_rsi:
                sub_strategies.append(RSIDivergenceStrategy())
            if use_bb:
                sub_strategies.append(BollingerBandsStrategy())
            if use_vol:
                sub_strategies.append(BreakoutVolumeStrategy())
            
            if len(sub_strategies) == 0:
                sub_strategies.append(EMACrossoverStrategy('EMA', 'EMA', 9, 20))
            
            strategy = HybridStrategy(sub_strategies, min_conf)
        
        elif strategy_type == "Seasonality Based":
            strategy = SeasonalityStrategy()
        
        elif strategy_type == "Momentum Reversal":
            rsi_period = st.number_input("RSI Period:", min_value=5, value=14)
            roc_period = st.number_input("ROC Period:", min_value=5, value=10)
            strategy = MomentumReversalStrategy(rsi_period, roc_period)
        
        elif strategy_type == "Smart Money Concepts (SMC)":
            liq_lookback = st.number_input("Liquidity Lookback:", min_value=10, value=20)
            strategy = SMCStrategy(liq_lookback)
        
        elif strategy_type == "VWAP Based":
            strategy = VWAPStrategy()
        
        elif strategy_type == "Simple Buy":
            strategy = SimpleBuyStrategy()
        
        else:  # Simple Sell
            strategy = SimpleSellStrategy()
        
        # Risk Management
        st.subheader("Risk Management")
        sl_type = st.selectbox("Stop Loss Type:", [
            "Custom Points", 
            "Trail SL", 
            "Signal Based",
            "ATR Based",
            "Previous Swing",
            "Current Candle",
            "Percentage Based"
        ])
        
        if sl_type == "Custom Points":
            sl_value = st.number_input("SL Points:", min_value=1.0, value=50.0, step=5.0)
        elif sl_type == "Trail SL":
            sl_value = st.number_input("Trail Points:", min_value=1.0, value=30.0, step=5.0)
        elif sl_type == "ATR Based":
            sl_value = st.number_input("ATR Multiplier:", min_value=0.5, value=2.0, step=0.1)
        elif sl_type == "Percentage Based":
            sl_value = st.number_input("SL Percentage:", min_value=0.1, value=1.0, step=0.1)
        else:
            sl_value = 0
        
        target_type = st.selectbox("Target Type:", [
            "Custom Points", 
            "Trail Target", 
            "Signal Based",
            "Risk Reward Ratio",
            "Percentage Based"
        ])
        
        if target_type == "Custom Points":
            target_value = st.number_input("Target Points:", min_value=1.0, value=100.0, step=5.0)
        elif target_type == "Trail Target":
            target_value = st.number_input("Trail Points:", min_value=1.0, value=50.0, step=5.0)
        elif target_type == "Risk Reward Ratio":
            target_value = st.number_input("RR Ratio:", min_value=1.0, value=2.0, step=0.1)
        elif target_type == "Percentage Based":
            target_value = st.number_input("Target Percentage:", min_value=0.1, value=2.0, step=0.1)
        else:
            target_value = 0
        
        quantity = st.number_input("Quantity:", min_value=1, value=1)
        
        # Setup button
        if st.button("🔧 Setup Trading System", type="primary"):
            with st.spinner("Setting up trading system..."):
                trading_system = TradingSystem(
                    ticker=ticker,
                    timeframe=timeframe,
                    period=period,
                    strategy=strategy,
                    sl_type=sl_type,
                    sl_value=sl_value,
                    target_type=target_type,
                    target_value=target_value,
                    quantity=quantity,
                    ticker2=ticker2
                )
                
                # Test data fetch
                data = trading_system.fetch_data()
                if data is not None and len(data) > 0:
                    st.session_state.trading_system = trading_system
                    st.success(f"✅ System ready! Loaded {len(data)} candles")
                    log_entry(f"System initialized - {ticker} {timeframe}/{period} - Strategy: {strategy.name}")
                else:
                    st.error("Failed to fetch data. Check ticker/timeframe/period combination.")
    
    # Main content
    if st.session_state.trading_system is None:
        st.info("👈 Configure your trading system in the sidebar and click 'Setup Trading System'")
        return
    
    # Control Buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not st.session_state.trading_active:
            if st.button("▶️ Start Trading", type="primary"):
                st.session_state.trading_active = True
                st.session_state.iteration_count = 0
                log_entry("🟢 Trading started - monitoring for signals")
                st.rerun()
    
    with col2:
        if st.session_state.trading_active:
            if st.button("⏹️ Stop Trading", type="secondary"):
                st.session_state.trading_active = False
                
                # Auto-close position if open
                if st.session_state.current_position:
                    system = st.session_state.trading_system
                    data = system.fetch_data()
                    if data is not None:
                        current_price = data['Close'].iloc[-1]
                        close_position(current_price, "Trading Stopped - Auto Close")
                
                log_entry("🔴 Trading stopped")
                st.rerun()
    
    with col3:
        if not st.session_state.trading_active and st.session_state.current_position:
            if st.button("❌ Force Close Position"):
                system = st.session_state.trading_system
                data = system.fetch_data()
                if data is not None:
                    current_price = data['Close'].iloc[-1]
                    close_position(current_price, "Force Closed by User")
                    st.rerun()
    
    with col4:
        if st.button("🔄 Refresh Data"):
            if st.session_state.trading_system:
                st.session_state.trading_system.fetch_data()
                st.rerun()
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Live Trading", "📈 Trade History", "📝 Trade Log"])
    
    # TAB 1: LIVE TRADING
    with tab1:
        if st.session_state.trading_active:
            st.session_state.iteration_count += 1
            
            system = st.session_state.trading_system
            data = system.fetch_data()
            
            if data is None or len(data) == 0:
                st.error("Unable to fetch data")
                time.sleep(2)
                st.rerun()
                return
            
            # Generate signals
            bullish_signal, bearish_signal, signal_data = system.strategy.generate_signal(data)
            current_price = data['Close'].iloc[-1]
            
            # Display header info
            st.markdown(f"""
            <div class='status-box'>
            <h3>🔴 LIVE - Auto-refreshing every 1.5-2s | Iteration: {st.session_state.iteration_count}</h3>
            <p><strong>Strategy:</strong> {system.strategy.name} | <strong>Asset:</strong> {system.ticker} | 
            <strong>Timeframe:</strong> {system.timeframe} | <strong>Period:</strong> {system.period} | 
            <strong>Candles:</strong> {len(data)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Position management
            if st.session_state.current_position is None:
                # Look for entry signal
                if bullish_signal:
                    enter_position('LONG', current_price, system, signal_data)
                    st.success(f"🟢 LONG Entry at {current_price:.2f}")
                    log_entry(f"🟢 LONG Entry at {current_price:.2f} | Signal: {signal_data}")
                elif bearish_signal:
                    enter_position('SHORT', current_price, system, signal_data)
                    st.warning(f"🔴 SHORT Entry at {current_price:.2f}")
                    log_entry(f"🔴 SHORT Entry at {current_price:.2f} | Signal: {signal_data}")
                else:
                    st.info("⏳ Waiting for entry signal...")
            
            else:
                # Manage existing position
                position = st.session_state.current_position
                
                # Update trailing SL
                if 'Trail' in system.sl_type:
                    old_sl = position['sl']
                    new_sl = system.update_trailing_sl(position, current_price)
                    if new_sl != old_sl:
                        position['sl'] = new_sl
                        log_entry(f"🔄 Trailing SL updated: {old_sl:.2f} → {new_sl:.2f}")
                
                # Check exit conditions
                should_exit, exit_reason = system.check_exit_conditions(
                    position, current_price, bullish_signal, bearish_signal
                )
                
                if should_exit:
                    close_position(current_price, exit_reason)
                    st.success(f"✅ Position closed: {exit_reason}")
                    log_entry(f"✅ Position closed at {current_price:.2f} - {exit_reason}")
                
                else:
                    # Display position status
                    entry_price = position['entry_price']
                    position_type = position['type']
                    
                    if position_type == 'LONG':
                        pnl_points = current_price - entry_price
                    else:
                        pnl_points = entry_price - current_price
                    
                    pnl_pct = (pnl_points / entry_price) * 100
                    pnl_total = pnl_points * system.quantity
                    
                    # Display metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Position", position_type, delta=f"Qty: {system.quantity}")
                    with col2:
                        st.metric("Entry Price", f"{entry_price:.2f}")
                    with col3:
                        st.metric("Current Price", f"{current_price:.2f}")
                    with col4:
                        pnl_color = "profit" if pnl_points > 0 else "loss"
                        st.markdown(f"<div class='stMetric'><p>P&L Points</p><h3 class='{pnl_color}'>{pnl_points:.2f}</h3></div>", unsafe_allow_html=True)
                    with col5:
                        st.markdown(f"<div class='stMetric'><p>P&L %</p><h3 class='{pnl_color}'>{pnl_pct:.2f}%</h3></div>", unsafe_allow_html=True)
                    
                    # Strategy Parameters Summary
                    st.subheader("📊 Strategy Parameters Summary")
                    param_summary = system.get_strategy_parameters_summary()
                    
                    if param_summary:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Current Values:**")
                            for key, value in param_summary.get('Current_Values', {}).items():
                                st.write(f"• {key}: {value}")
                        
                        with col2:
                            st.markdown("**Historical Statistics:**")
                            for key, value in param_summary.get('Historical_Stats', {}).items():
                                st.write(f"• {key}: {value}")
                    
                    # Strategy-specific signal info
                    st.subheader("📈 Current Signal Data")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        for key, value in list(signal_data.items())[:3]:
                            if isinstance(value, (int, float)):
                                st.metric(key, f"{value:.2f}")
                            else:
                                st.metric(key, str(value))
                    
                    with col2:
                        st.metric("Stop Loss", f"{position['sl']:.2f}", 
                                 delta=f"Distance: {abs(current_price - position['sl']):.2f}")
                    
                    with col3:
                        st.metric("Target", f"{position['target']:.2f}",
                                 delta=f"Distance: {abs(position['target'] - current_price):.2f}")
                    
                    # Market status
                    status_text = system.get_market_status(position, current_price)
                    st.markdown(f"<div class='status-box'><h4>{status_text}</h4></div>", unsafe_allow_html=True)
                    
                    # Trade Guidance (100 words)
                    st.subheader("💡 Trade Guidance & Management")
                    guidance = system.get_trade_guidance(position, current_price, signal_data)
                    st.info(guidance)
            
            # Display chart
            display_chart(data, system, st.session_state.current_position)
            
            # Auto-refresh
            time.sleep(1.5)
            st.rerun()
        
        else:
            st.info("Click 'Start Trading' to begin live monitoring")
            
            # Show static chart if system is setup
            if st.session_state.trading_system:
                system = st.session_state.trading_system
                data = system.fetch_data()
                if data is not None:
                    display_chart(data, system, None)
    
    # TAB 2: TRADE HISTORY
    with tab2:
        if len(st.session_state.trade_history) == 0:
            st.info("No trades yet. Start trading to see history.")
        else:
            trades = st.session_state.trade_history
            
            # Summary metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl_points'] > 0])
            losing_trades = total_trades - winning_trades
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            total_pnl = sum([t['pnl_points'] for t in trades])
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Trades", total_trades)
            col2.metric("Wins", winning_trades)
            col3.metric("Losses", losing_trades)
            col4.metric("Win Rate", f"{win_rate:.1f}%")
            
            pnl_color = "profit" if total_pnl > 0 else "loss"
            col5.markdown(f"<div class='stMetric'><p>Total P&L</p><h3 class='{pnl_color}'>{total_pnl:.2f}</h3></div>", unsafe_allow_html=True)
            
            # Performance by strategy
            st.subheader("Performance by Strategy")
            strategy_stats = {}
            for trade in trades:
                strat = trade['strategy']
                if strat not in strategy_stats:
                    strategy_stats[strat] = {'trades': 0, 'wins': 0, 'pnl': 0}
                strategy_stats[strat]['trades'] += 1
                if trade['pnl_points'] > 0:
                    strategy_stats[strat]['wins'] += 1
                strategy_stats[strat]['pnl'] += trade['pnl_points']
            
            for strat, stats in strategy_stats.items():
                wr = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
                st.write(f"**{strat}**: {stats['trades']} trades | Win Rate: {wr:.1f}% | P&L: {stats['pnl']:.2f}")
            
            # Trade table
            st.subheader("Trade Details")
            df_trades = pd.DataFrame(trades)
            st.dataframe(df_trades, use_container_width=True)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Cumulative P&L
                fig = go.Figure()
                cumulative_pnl = np.cumsum([t['pnl_points'] for t in trades])
                fig.add_trace(go.Scatter(
                    y=cumulative_pnl,
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color='cyan', width=2)
                ))
                fig.update_layout(
                    title="Cumulative P&L",
                    xaxis_title="Trade Number",
                    yaxis_title="Points",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Win/Loss pie
                fig = go.Figure(data=[go.Pie(
                    labels=['Wins', 'Losses'],
                    values=[winning_trades, losing_trades],
                    marker=dict(colors=['#00ff00', '#ff0000'])
                )])
                fig.update_layout(title="Win/Loss Distribution", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            
            # Individual trade analysis
            st.subheader("Individual Trade Analysis")
            selected_trade_idx = st.selectbox("Select Trade:", range(len(trades)), 
                                             format_func=lambda x: f"Trade {x+1} - {trades[x]['type']} - {trades[x]['pnl_points']:.2f} pts")
            
            if selected_trade_idx is not None:
                trade = trades[selected_trade_idx]
                st.markdown(f"### Trade #{selected_trade_idx + 1}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Type", trade['type'])
                col2.metric("Entry", f"{trade['entry_price']:.2f}")
                col3.metric("Exit", f"{trade['exit_price']:.2f}")
                col4.metric("P&L", f"{trade['pnl_points']:.2f} ({trade['pnl_pct']:.2f}%)")
                
                st.write(f"**Duration:** {trade['duration']}")
                st.write(f"**Exit Reason:** {trade['exit_reason']}")
                st.write(f"**Strategy:** {trade['strategy']}")
                
                # AI Analysis
                st.markdown("### 🤖 AI Analysis")
                if st.session_state.trading_system:
                    analysis = st.session_state.trading_system.analyze_trade(trade)
                    st.markdown(analysis)
    
    # TAB 3: TRADE LOG
    with tab3:
        st.subheader("Trade Log (Last 100 entries)")
        
        if st.button("🗑️ Clear Log"):
            st.session_state.trade_log = []
            st.rerun()
        
        log_html = "<div class='trade-log'>"
        for entry in reversed(st.session_state.trade_log[-100:]):
            log_html += f"<p>{entry}</p>"
        log_html += "</div>"
        
        st.markdown(log_html, unsafe_allow_html=True)

# ======================== HELPER FUNCTIONS ========================

def log_entry(message: str):
    """Add entry to trade log"""
    timestamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    st.session_state.trade_log.append(f"[{timestamp}] {message}")

def enter_position(position_type: str, entry_price: float, system: TradingSystem, signal_data: Dict):
    """Enter a new position"""
    sl, target = system.calculate_sl_target(entry_price, position_type)
    
    st.session_state.current_position = {
        'type': position_type,
        'entry_price': entry_price,
        'entry_time': datetime.now(IST),
        'sl': sl,
        'target': target,
        'signal_data': signal_data
    }

def close_position(exit_price: float, exit_reason: str):
    """Close current position and record trade"""
    if st.session_state.current_position is None:
        return
    
    position = st.session_state.current_position
    system = st.session_state.trading_system
    
    entry_price = position['entry_price']
    entry_time = position['entry_time']
    exit_time = datetime.now(IST)
    
    if position['type'] == 'LONG':
        pnl_points = exit_price - entry_price
    else:
        pnl_points = entry_price - exit_price
    
    pnl_pct = (pnl_points / entry_price) * 100
    duration = str(exit_time - entry_time).split('.')[0]
    
    trade_record = {
        'entry_time': entry_time.strftime("%Y-%m-%d %H:%M:%S"),
        'exit_time': exit_time.strftime("%Y-%m-%d %H:%M:%S"),
        'duration': duration,
        'type': position['type'],
        'entry_price': entry_price,
        'exit_price': exit_price,
        'pnl_points': pnl_points,
        'pnl_pct': pnl_pct,
        'quantity': system.quantity,
        'exit_reason': exit_reason,
        'strategy': system.strategy.name,
        'ticker': system.ticker
    }
    
    st.session_state.trade_history.append(trade_record)
    st.session_state.current_position = None

def display_chart(data: pd.DataFrame, system: TradingSystem, position: Optional[Dict]):
    """Display interactive candlestick chart with indicators"""
    # Calculate strategy indicators
    data = system.strategy.calculate_indicators(data)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ), row=1, col=1)
    
    # Add strategy-specific indicators
    if 'Indicator1' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Indicator1'],
            mode='lines', name=f"{system.strategy.name.split('/')[0]}",
            line=dict(color='cyan', width=1)
        ), row=1, col=1)
    
    if 'Indicator2' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Indicator2'],
            mode='lines', name=f"{system.strategy.name.split('/')[1] if '/' in system.strategy.name else 'Indicator2'}",
            line=dict(color='magenta', width=1)
        ), row=1, col=1)
    
    # Fibonacci levels
    for col in data.columns:
        if col.startswith('Fib_'):
            level = col.replace('Fib_', '')
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col],
                mode='lines', name=f'Fib {level}',
                line=dict(dash='dot', width=1)
            ), row=1, col=1)
    
    # Bands for breakout strategy
    if 'Upper_Band' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Upper_Band'],
            mode='lines', name='Upper Band',
            line=dict(color='red', width=1, dash='dash')
        ), row=1, col=1)
    
    if 'Lower_Band' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Lower_Band'],
            mode='lines', name='Lower Band',
            line=dict(color='green', width=1, dash='dash')
        ), row=1, col=1)
    
    # Position markers
    if position:
        entry_price = position['entry_price']
        sl = position['sl']
        target = position['target']
        
        # Entry marker
        fig.add_trace(go.Scatter(
            x=[position['entry_time']],
            y=[entry_price],
            mode='markers',
            marker=dict(
                symbol='triangle-up' if position['type'] == 'LONG' else 'triangle-down',
                size=15,
                color='green' if position['type'] == 'LONG' else 'red'
            ),
            name='Entry',
            showlegend=False
        ), row=1, col=1)
        
        # SL and Target lines
        fig.add_hline(y=sl, line_dash="dot", line_color="red", 
                     annotation_text="SL", row=1, col=1)
        fig.add_hline(y=target, line_dash="dot", line_color="green",
                     annotation_text="Target", row=1, col=1)
    
    # Volume (if available)
    if 'Volume' in data.columns:
        colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
                 for i in range(len(data))]
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors
        ), row=2, col=1)
    
    # Layout
    fig.update_layout(
        title=f"{system.ticker} - {system.timeframe} | {system.strategy.name}",
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=700,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

# ======================== RUN APP ========================

if __name__ == "__main__":
    main()
