"""
Production-Grade Quantitative Trading System v2.0
Complete Implementation with All Features
Author: Quantitative Trading Professional
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
from datetime import datetime, timedelta
import pytz
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

ASSET_GROUPS = {
    "Indian Indices": {
        "NIFTY 50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "SENSEX": "^BSESN"
    },
    "Crypto": {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD"
    },
    "Forex": {
        "USDINR": "USDINR=X",
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X"
    },
    "Commodities": {
        "Gold": "GC=F",
        "Silver": "SI=F"
    }
}

TIMEFRAME_PERIODS = {
    "1m": ["1d", "5d"],
    "5m": ["1d", "1mo"],
    "15m": ["1mo"],
    "30m": ["1mo"],
    "1h": ["1mo"],
    "4h": ["1mo"],
    "1d": ["1mo", "1y", "2y", "5y"],
    "1wk": ["1mo", "1y", "5y", "10y", "15y", "20y"],
    "1mo": ["1y", "2y", "5y", "10y", "15y", "20y", "25y", "30y"]
}

IST = pytz.timezone('Asia/Kolkata')

# ==================== DATA FETCHING ====================

def fetch_data(ticker: str, interval: str, period: str, is_live: bool = False) -> pd.DataFrame:
    """Fetch data from yfinance with proper error handling and timezone conversion"""
    try:
        if is_live:
            time.sleep(random.uniform(1.0, 1.5))
        
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if data.empty:
            return pd.DataFrame()
        
        # Flatten multi-index if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip('_') for col in data.columns.values]
            cols_to_keep = []
            for base in ['Open', 'High', 'Low', 'Close', 'Volume']:
                matching = [col for col in data.columns if base in col]
                if matching:
                    cols_to_keep.append((matching[0], base))
            data = data[[col[0] for col in cols_to_keep]]
            data.columns = [col[1] for col in cols_to_keep]
        
        # Ensure required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                return pd.DataFrame()
        
        # Convert timezone to IST
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert(IST)
        else:
            data.index = data.index.tz_convert(IST)
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching {ticker}: {str(e)}")
        return pd.DataFrame()

def align_dataframes_by_timezone(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align two dataframes by matching their common timezone indices - NO GARBAGE ZEROS"""
    if df1.empty or df2.empty:
        return df1, df2
    
    # Convert both to IST if not already
    if df1.index.tz is None:
        df1.index = df1.index.tz_localize('UTC').tz_convert(IST)
    else:
        df1.index = df1.index.tz_convert(IST)
    
    if df2.index.tz is None:
        df2.index = df2.index.tz_localize('UTC').tz_convert(IST)
    else:
        df2.index = df2.index.tz_convert(IST)
    
    # Find common timestamps - ONLY REAL DATA
    common_index = df1.index.intersection(df2.index)
    
    if len(common_index) == 0:
        st.warning("No common timestamps between tickers. Data may be from different markets/timezones.")
        return df1, df2
    
    # Return only matching real data
    df1_aligned = df1.loc[common_index]
    df2_aligned = df2.loc[common_index]
    
    return df1_aligned, df2_aligned

# ==================== TECHNICAL INDICATORS ====================

def calculate_ema(data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return data['Close'].ewm(span=period, adjust=False).mean()

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = pd.Series(true_range).rolling(period).mean()
    atr.index = data.index
    return atr

def calculate_swing_highs_lows(data: pd.DataFrame, lookback: int = 5) -> Tuple[pd.Series, pd.Series]:
    """Calculate swing highs and lows"""
    swing_high = data['High'].rolling(window=lookback*2+1, center=True).max()
    swing_low = data['Low'].rolling(window=lookback*2+1, center=True).min()
    return swing_high, swing_low

def calculate_crossover_angle(ema_fast: pd.Series, ema_slow: pd.Series, lookback: int = 2) -> float:
    """Calculate angle of EMA crossover in degrees"""
    if len(ema_fast) < lookback + 1:
        return 0.0
    
    diff_current = ema_fast.iloc[-1] - ema_slow.iloc[-1]
    diff_previous = ema_fast.iloc[-lookback] - ema_slow.iloc[-lookback]
    
    angle = np.arctan2(diff_current - diff_previous, lookback) * 180 / np.pi
    return abs(angle)

# ==================== RATIO STRATEGY ====================

def calculate_ratio_bins(data1: pd.DataFrame, data2: pd.DataFrame, num_bins: int = 5) -> Dict:
    """Calculate ratio and create bins with expected moves"""
    ratio = data1['Close'] / data2['Close']
    
    # Create bins
    bin_edges = pd.qcut(ratio, q=num_bins, retbins=True, duplicates='drop')[1]
    bins = pd.cut(ratio, bins=bin_edges, include_lowest=True)
    
    bin_info = {}
    for i, bin_range in enumerate(bins.cat.categories):
        mask = bins == bin_range
        if mask.sum() > 0:
            subset = ratio[mask]
            bin_info[i] = {
                'range': f"{bin_range.left:.4f} - {bin_range.right:.4f}",
                'left': bin_range.left,
                'right': bin_range.right,
                'count': mask.sum(),
                'current_in_bin': ratio.iloc[-1] >= bin_range.left and ratio.iloc[-1] <= bin_range.right
            }
    
    return {'ratio': ratio, 'bins': bin_info, 'current_bin': bins.iloc[-1] if len(bins) > 0 else None}

def calculate_expected_moves(data1: pd.DataFrame, data2: pd.DataFrame, ratio_info: Dict, 
                            n_candles: int = 3) -> Dict:
    """Calculate expected moves for each ticker based on bin analysis"""
    ratio = ratio_info['ratio']
    bins_info = ratio_info['bins']
    
    expected_moves = {}
    for bin_id, bin_data in bins_info.items():
        if bin_data['current_in_bin']:
            mask = (ratio >= bin_data['left']) & (ratio <= bin_data['right'])
            
            if mask.sum() > n_candles:
                ticker1_returns = []
                ticker2_returns = []
                
                indices = np.where(mask)[0]
                for idx in indices:
                    if idx + n_candles < len(data1):
                        ret1 = (data1['Close'].iloc[idx + n_candles] - data1['Close'].iloc[idx]) / data1['Close'].iloc[idx] * 100
                        ret2 = (data2['Close'].iloc[idx + n_candles] - data2['Close'].iloc[idx]) / data2['Close'].iloc[idx] * 100
                        ticker1_returns.append(ret1)
                        ticker2_returns.append(ret2)
                
                if ticker1_returns and ticker2_returns:
                    expected_moves[bin_id] = {
                        'ticker1_mean': np.mean(ticker1_returns),
                        'ticker1_std': np.std(ticker1_returns),
                        'ticker2_mean': np.mean(ticker2_returns),
                        'ticker2_std': np.std(ticker2_returns),
                        'observations': len(ticker1_returns)
                    }
    
    return expected_moves

def generate_ratio_signals(data1: pd.DataFrame, data2: pd.DataFrame, ratio_info: Dict,
                          expected_moves: Dict, atr1: pd.Series, atr2: pd.Series,
                          config: Dict) -> Dict:
    """Generate trading signals based on ratio strategy"""
    signal = {
        'ticker1': {'action': 'HOLD', 'entry': 0, 'target': 0, 'sl': 0},
        'ticker2': {'action': 'HOLD', 'entry': 0, 'target': 0, 'sl': 0},
        'reason': 'No signal'
    }
    
    if not expected_moves:
        return signal
    
    current_bin = None
    for bin_id, bin_data in ratio_info['bins'].items():
        if bin_data['current_in_bin']:
            current_bin = bin_id
            break
    
    if current_bin is None or current_bin not in expected_moves:
        return signal
    
    moves = expected_moves[current_bin]
    
    ticker1_price = data1['Close'].iloc[-1]
    ticker2_price = data2['Close'].iloc[-1]
    
    # Ticker 1 signal
    if moves['ticker1_mean'] > 0.2:
        signal['ticker1']['action'] = 'BUY'
        signal['ticker1']['entry'] = ticker1_price
        signal['ticker1']['target'] = calculate_target(data1, ticker1_price, atr1, config, 'BUY')
        signal['ticker1']['sl'] = calculate_stoploss(data1, ticker1_price, atr1, config, 'BUY')
    elif moves['ticker1_mean'] < -0.2:
        signal['ticker1']['action'] = 'SELL'
        signal['ticker1']['entry'] = ticker1_price
        signal['ticker1']['target'] = calculate_target(data1, ticker1_price, atr1, config, 'SELL')
        signal['ticker1']['sl'] = calculate_stoploss(data1, ticker1_price, atr1, config, 'SELL')
    
    # Ticker 2 signal
    if moves['ticker2_mean'] > 0.2:
        signal['ticker2']['action'] = 'BUY'
        signal['ticker2']['entry'] = ticker2_price
        signal['ticker2']['target'] = calculate_target(data2, ticker2_price, atr2, config, 'BUY')
        signal['ticker2']['sl'] = calculate_stoploss(data2, ticker2_price, atr2, config, 'SELL')
    elif moves['ticker2_mean'] < -0.2:
        signal['ticker2']['action'] = 'SELL'
        signal['ticker2']['entry'] = ticker2_price
        signal['ticker2']['target'] = calculate_target(data2, ticker2_price, atr2, config, 'SELL')
        signal['ticker2']['sl'] = calculate_stoploss(data2, ticker2_price, atr2, config, 'SELL')
    
    signal['reason'] = f"Bin {current_bin}: T1 exp={moves['ticker1_mean']:.2f}%, T2 exp={moves['ticker2_mean']:.2f}%"
    
    return signal

# ==================== STOPLOSS & TARGET CALCULATION ====================

def calculate_stoploss(data: pd.DataFrame, entry: float, atr: pd.Series, 
                      config: Dict, action: str) -> float:
    """Calculate stoploss based on configuration"""
    sl_type = config.get('sl_type', 'custom_points')
    
    if sl_type == 'custom_points':
        sl_value = config.get('sl_value', 10)
        return entry - sl_value if action == 'BUY' else entry + sl_value
    
    elif sl_type == 'atr_based':
        atr_multiplier = config.get('atr_multiplier', 2.0)
        atr_val = atr.iloc[-1] if len(atr) > 0 else entry * 0.01
        return entry - (atr_val * atr_multiplier) if action == 'BUY' else entry + (atr_val * atr_multiplier)
    
    elif sl_type == 'previous_candle':
        return data['Low'].iloc[-2] if action == 'BUY' and len(data) > 1 else data['High'].iloc[-2] if len(data) > 1 else entry
    
    elif sl_type == 'current_candle':
        return data['Low'].iloc[-1] if action == 'BUY' else data['High'].iloc[-1]
    
    elif sl_type == 'previous_swing':
        swing_high, swing_low = calculate_swing_highs_lows(data)
        if action == 'BUY':
            prev_swing = swing_low.iloc[-2] if len(swing_low) > 1 else entry * 0.98
        else:
            prev_swing = swing_high.iloc[-2] if len(swing_high) > 1 else entry * 1.02
        return prev_swing
    
    elif sl_type == 'current_swing':
        swing_high, swing_low = calculate_swing_highs_lows(data)
        return swing_low.iloc[-1] if action == 'BUY' else swing_high.iloc[-1]
    
    elif sl_type == 'ema_crossover_reversal':
        # On reverse crossover, exit
        ema1 = calculate_ema(data, 9)
        ema2 = calculate_ema(data, 15)
        if action == 'BUY':
            return ema2.iloc[-1]  # Exit when fast crosses below slow
        else:
            return ema2.iloc[-1]  # Exit when fast crosses above slow
    
    elif sl_type == 'trailing_previous_candle':
        return data['Low'].iloc[-2] if action == 'BUY' and len(data) > 1 else data['High'].iloc[-2] if len(data) > 1 else entry
    
    elif sl_type == 'trailing_current_candle':
        return data['Low'].iloc[-1] if action == 'BUY' else data['High'].iloc[-1]
    
    elif sl_type == 'trailing_previous_swing':
        swing_high, swing_low = calculate_swing_highs_lows(data)
        if action == 'BUY':
            return swing_low.iloc[-2] if len(swing_low) > 1 else entry * 0.98
        else:
            return swing_high.iloc[-2] if len(swing_high) > 1 else entry * 1.02
    
    elif sl_type == 'trailing_current_swing':
        swing_high, swing_low = calculate_swing_highs_lows(data)
        return swing_low.iloc[-1] if action == 'BUY' else swing_high.iloc[-1]
    
    else:
        return entry - 10 if action == 'BUY' else entry + 10

def calculate_target(data: pd.DataFrame, entry: float, atr: pd.Series,
                    config: Dict, action: str) -> float:
    """Calculate target based on configuration"""
    target_type = config.get('target_type', 'custom_points')
    
    if target_type == 'custom_points':
        target_value = config.get('target_value', 20)
        return entry + target_value if action == 'BUY' else entry - target_value
    
    elif target_type == 'atr_based':
        atr_multiplier = config.get('target_atr_multiplier', 3.0)
        atr_val = atr.iloc[-1] if len(atr) > 0 else entry * 0.02
        return entry + (atr_val * atr_multiplier) if action == 'BUY' else entry - (atr_val * atr_multiplier)
    
    elif target_type == 'previous_candle':
        return data['High'].iloc[-2] if action == 'BUY' and len(data) > 1 else data['Low'].iloc[-2] if len(data) > 1 else entry
    
    elif target_type == 'current_candle':
        return data['High'].iloc[-1] if action == 'BUY' else data['Low'].iloc[-1]
    
    elif target_type == 'previous_swing':
        swing_high, swing_low = calculate_swing_highs_lows(data)
        if action == 'BUY':
            return swing_high.iloc[-2] if len(swing_high) > 1 else entry * 1.02
        else:
            return swing_low.iloc[-2] if len(swing_low) > 1 else entry * 0.98
    
    elif target_type == 'current_swing':
        swing_high, swing_low = calculate_swing_highs_lows(data)
        return swing_high.iloc[-1] if action == 'BUY' else swing_low.iloc[-1]
    
    elif target_type == 'ema_crossover_reversal':
        # Target based on EMA projection
        ema1 = calculate_ema(data, 9)
        if action == 'BUY':
            return ema1.iloc[-1] * 1.02
        else:
            return ema1.iloc[-1] * 0.98
    
    elif target_type == 'trailing_previous_candle':
        return data['High'].iloc[-2] if action == 'BUY' and len(data) > 1 else data['Low'].iloc[-2] if len(data) > 1 else entry
    
    elif target_type == 'trailing_current_candle':
        return data['High'].iloc[-1] if action == 'BUY' else data['Low'].iloc[-1]
    
    elif target_type == 'trailing_previous_swing':
        swing_high, swing_low = calculate_swing_highs_lows(data)
        if action == 'BUY':
            return swing_high.iloc[-2] if len(swing_high) > 1 else entry * 1.02
        else:
            return swing_low.iloc[-2] if len(swing_low) > 1 else entry * 0.98
    
    elif target_type == 'trailing_current_swing':
        swing_high, swing_low = calculate_swing_highs_lows(data)
        return swing_high.iloc[-1] if action == 'BUY' else swing_low.iloc[-1]
    
    else:
        return entry + 20 if action == 'BUY' else entry - 20

# ==================== EMA CROSSOVER STRATEGY ====================

def generate_ema_signals(data: pd.DataFrame, ema1_period: int = 9, ema2_period: int = 15,
                        config: Dict = None) -> Dict:
    """Generate EMA crossover signals"""
    if config is None:
        config = {}
    
    ema1 = calculate_ema(data, ema1_period)
    ema2 = calculate_ema(data, ema2_period)
    atr = calculate_atr(data)
    
    signal = {
        'action': 'HOLD', 
        'entry': 0, 
        'target': 0, 
        'sl': 0, 
        'reason': 'No crossover',
        'ema1': ema1.iloc[-1] if len(ema1) > 0 else 0,
        'ema2': ema2.iloc[-1] if len(ema2) > 0 else 0
    }
    
    if len(ema1) < 2 or len(ema2) < 2:
        return signal
    
    current_price = data['Close'].iloc[-1]
    
    # Check for crossover
    bullish_cross = ema1.iloc[-2] <= ema2.iloc[-2] and ema1.iloc[-1] > ema2.iloc[-1]
    bearish_cross = ema1.iloc[-2] >= ema2.iloc[-2] and ema1.iloc[-1] < ema2.iloc[-1]
    
    # Angle check
    if config.get('use_angle', False):
        min_angle = config.get('min_angle', 1.0)
        angle = calculate_crossover_angle(ema1, ema2)
        if angle < min_angle:
            return signal
    
    # Crossover type checks
    crossover_type = config.get('crossover_type', 'simple')
    
    if crossover_type == 'candle_size':
        min_candle_size = config.get('min_candle_size', 10)
        candle_size = abs(data['Close'].iloc[-1] - data['Open'].iloc[-1])
        if candle_size < min_candle_size:
            return signal
    
    elif crossover_type == 'atr_based':
        atr_threshold = config.get('atr_threshold', 0.5)
        if atr.iloc[-1] < atr_threshold:
            return signal
    
    if bullish_cross:
        signal['action'] = 'BUY'
        signal['entry'] = current_price
        signal['target'] = calculate_target(data, current_price, atr, config, 'BUY')
        signal['sl'] = calculate_stoploss(data, current_price, atr, config, 'BUY')
        signal['reason'] = f'Bullish EMA crossover ({ema1_period}/{ema2_period})'
    
    elif bearish_cross:
        signal['action'] = 'SELL'
        signal['entry'] = current_price
        signal['target'] = calculate_target(data, current_price, atr, config, 'SELL')
        signal['sl'] = calculate_stoploss(data, current_price, atr, config, 'SELL')
        signal['reason'] = f'Bearish EMA crossover ({ema1_period}/{ema2_period})'
    
    return signal

# ==================== RSI STRATEGY ====================

def generate_rsi_signals(data: pd.DataFrame, rsi_period: int = 14, 
                        oversold: int = 30, overbought: int = 70,
                        config: Dict = None) -> Dict:
    """Generate RSI-based signals"""
    if config is None:
        config = {}
    
    rsi = calculate_rsi(data, rsi_period)
    atr = calculate_atr(data)
    
    signal = {
        'action': 'HOLD', 
        'entry': 0, 
        'target': 0, 
        'sl': 0, 
        'reason': 'RSI neutral',
        'rsi': rsi.iloc[-1] if len(rsi) > 0 else 50
    }
    
    if len(rsi) < 2:
        return signal
    
    current_price = data['Close'].iloc[-1]
    current_rsi = rsi.iloc[-1]
    
    if current_rsi < oversold:
        signal['action'] = 'BUY'
        signal['entry'] = current_price
        signal['target'] = calculate_target(data, current_price, atr, config, 'BUY')
        signal['sl'] = calculate_stoploss(data, current_price, atr, config, 'BUY')
        signal['reason'] = f'RSI oversold: {current_rsi:.2f}'
    
    elif current_rsi > overbought:
        signal['action'] = 'SELL'
        signal['entry'] = current_price
        signal['target'] = calculate_target(data, current_price, atr, config, 'SELL')
        signal['sl'] = calculate_stoploss(data, current_price, atr, config, 'SELL')
        signal['reason'] = f'RSI overbought: {current_rsi:.2f}'
    
    return signal

# ==================== AI-BASED STRATEGY ====================

def generate_ai_signals(data: pd.DataFrame, config: Dict = None) -> Dict:
    """
    AI-based strategy using simple ML features
    Uses: Price momentum, RSI, ATR, EMA distance
    """
    if config is None:
        config = {}
    
    signal = {
        'action': 'HOLD',
        'entry': 0,
        'target': 0,
        'sl': 0,
        'reason': 'AI analyzing...',
        'confidence': 0
    }
    
    if len(data) < 50:
        signal['reason'] = 'Insufficient data for AI'
        return signal
    
    # Calculate features
    rsi = calculate_rsi(data, 14)
    atr = calculate_atr(data, 14)
    ema9 = calculate_ema(data, 9)
    ema21 = calculate_ema(data, 21)
    
    current_price = data['Close'].iloc[-1]
    
    # Feature 1: Price momentum (5-period rate of change)
    momentum = ((current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6]) * 100 if len(data) >= 6 else 0
    
    # Feature 2: RSI position
    rsi_current = rsi.iloc[-1]
    
    # Feature 3: EMA alignment
    ema_diff = ema9.iloc[-1] - ema21.iloc[-1]
    ema_diff_pct = (ema_diff / ema21.iloc[-1]) * 100
    
    # Feature 4: Volatility (ATR as % of price)
    volatility = (atr.iloc[-1] / current_price) * 100
    
    # Simple ML logic (threshold-based decision tree)
    confidence = 0
    
    # Bullish conditions
    if momentum > 1 and rsi_current < 70 and ema_diff_pct > 0.2:
        confidence += 30
    if rsi_current < 30:
        confidence += 20
    if ema_diff_pct > 0.5:
        confidence += 25
    if volatility > 1:  # High volatility favors trading
        confidence += 15
    if data['Close'].iloc[-1] > data['Close'].iloc[-2]:  # Upward price action
        confidence += 10
    
    # Bearish conditions
    bearish_confidence = 0
    if momentum < -1 and rsi_current > 30 and ema_diff_pct < -0.2:
        bearish_confidence += 30
    if rsi_current > 70:
        bearish_confidence += 20
    if ema_diff_pct < -0.5:
        bearish_confidence += 25
    if volatility > 1:
        bearish_confidence += 15
    if data['Close'].iloc[-1] < data['Close'].iloc[-2]:
        bearish_confidence += 10
    
    # Generate signal based on confidence
    if confidence >= 60:  # Bullish
        signal['action'] = 'BUY'
        signal['entry'] = current_price
        signal['target'] = calculate_target(data, current_price, atr, config, 'BUY')
        signal['sl'] = calculate_stoploss(data, current_price, atr, config, 'BUY')
        signal['reason'] = f'AI Bullish (Momentum: {momentum:.1f}%, RSI: {rsi_current:.1f})'
        signal['confidence'] = confidence
    
    elif bearish_confidence >= 60:  # Bearish
        signal['action'] = 'SELL'
        signal['entry'] = current_price
        signal['target'] = calculate_target(data, current_price, atr, config, 'SELL')
        signal['sl'] = calculate_stoploss(data, current_price, atr, config, 'SELL')
        signal['reason'] = f'AI Bearish (Momentum: {momentum:.1f}%, RSI: {rsi_current:.1f})'
        signal['confidence'] = bearish_confidence
    
    else:
        signal['reason'] = f'AI insufficient confidence (Bull: {confidence}, Bear: {bearish_confidence})'
        signal['confidence'] = max(confidence, bearish_confidence)
    
    return signal

# ==================== CUSTOM STRATEGY ====================

def generate_custom_signals(data: pd.DataFrame, indicators: Dict, config: Dict = None) -> Dict:
    """
    Custom strategy with user-selected indicators
    Allows combining multiple indicators with custom logic
    """
    if config is None:
        config = {}
    
    signal = {
        'action': 'HOLD',
        'entry': 0,
        'target': 0,
        'sl': 0,
        'reason': 'Custom strategy',
        'indicators': {}
    }
    
    if len(data) < 30:
        signal['reason'] = 'Insufficient data'
        return signal
    
    current_price = data['Close'].iloc[-1]
    atr = calculate_atr(data)
    
    # Calculate requested indicators
    buy_signals = 0
    sell_signals = 0
    
    if indicators.get('use_rsi', False):
        rsi = calculate_rsi(data, indicators.get('rsi_period', 14))
        rsi_val = rsi.iloc[-1]
        signal['indicators']['rsi'] = rsi_val
        
        if rsi_val < indicators.get('rsi_oversold', 30):
            buy_signals += 1
        elif rsi_val > indicators.get('rsi_overbought', 70):
            sell_signals += 1
    
    if indicators.get('use_ema', False):
        ema_fast = calculate_ema(data, indicators.get('ema_fast', 9))
        ema_slow = calculate_ema(data, indicators.get('ema_slow', 21))
        signal['indicators']['ema_fast'] = ema_fast.iloc[-1]
        signal['indicators']['ema_slow'] = ema_slow.iloc[-1]
        
        if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
            buy_signals += 1
        else:
            sell_signals += 1
    
    if indicators.get('use_momentum', False):
        momentum_period = indicators.get('momentum_period', 10)
        momentum = ((data['Close'].iloc[-1] - data['Close'].iloc[-momentum_period]) / 
                   data['Close'].iloc[-momentum_period]) * 100
        signal['indicators']['momentum'] = momentum
        
        if momentum > 0:
            buy_signals += 1
        else:
            sell_signals += 1
    
    if indicators.get('use_volume', False):
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        signal['indicators']['volume_ratio'] = volume_ratio
        
        if volume_ratio > 1.5:  # High volume confirms signal
            if buy_signals > sell_signals:
                buy_signals += 1
            else:
                sell_signals += 1
    
    # Generate final signal
    min_signals = indicators.get('min_confirmations', 2)
    
    if buy_signals >= min_signals and buy_signals > sell_signals:
        signal['action'] = 'BUY'
        signal['entry'] = current_price
        signal['target'] = calculate_target(data, current_price, atr, config, 'BUY')
        signal['sl'] = calculate_stoploss(data, current_price, atr, config, 'BUY')
        signal['reason'] = f'Custom: {buy_signals} buy signals confirmed'
    
    elif sell_signals >= min_signals and sell_signals > buy_signals:
        signal['action'] = 'SELL'
        signal['entry'] = current_price
        signal['target'] = calculate_target(data, current_price, atr, config, 'SELL')
        signal['sl'] = calculate_stoploss(data, current_price, atr, config, 'SELL')
        signal['reason'] = f'Custom: {sell_signals} sell signals confirmed'
    
    else:
        signal['reason'] = f'Custom: Insufficient confirmation (Buy: {buy_signals}, Sell: {sell_signals})'
    
    return signal

# ==================== TRADE MANAGEMENT ====================

class TradeManager:
    """Manage trades with entry, exit, PnL tracking"""
    
    def __init__(self):
        self.active_trades = {}
        self.closed_trades = []
        self.trade_id_counter = 0
    
    def open_trade(self, ticker: str, action: str, entry: float, target: float, 
                   sl: float, quantity: int, reason: str) -> int:
        """Open a new trade"""
        self.trade_id_counter += 1
        trade_id = self.trade_id_counter
        
        self.active_trades[trade_id] = {
            'ticker': ticker,
            'action': action,
            'entry_price': entry,
            'entry_time': datetime.now(IST),
            'target': target,
            'sl': sl,
            'quantity': quantity,
            'reason': reason,
            'highest_price': entry,
            'lowest_price': entry
        }
        
        return trade_id
    
    def update_trade(self, trade_id: int, current_price: float, config: Dict):
        """Update active trade with trailing SL/Target"""
        if trade_id not in self.active_trades:
            return
        
        trade = self.active_trades[trade_id]
        
        # Update highest/lowest
        if current_price > trade['highest_price']:
            trade['highest_price'] = current_price
        if current_price < trade['lowest_price']:
            trade['lowest_price'] = current_price
    
    def check_exit(self, trade_id: int, current_price: float) -> Tuple[bool, str]:
        """Check if trade should be exited"""
        if trade_id not in self.active_trades:
            return False, ""
        
        trade = self.active_trades[trade_id]
        
        if trade['action'] == 'BUY':
            if current_price >= trade['target']:
                return True, "Target hit"
            elif current_price <= trade['sl']:
                return True, "Stoploss hit"
        
        elif trade['action'] == 'SELL':
            if current_price <= trade['target']:
                return True, "Target hit"
            elif current_price >= trade['sl']:
                return True, "Stoploss hit"
        
        return False, ""
    
    def close_trade(self, trade_id: int, exit_price: float, exit_reason: str):
        """Close an active trade"""
        if trade_id not in self.active_trades:
            return
        
        trade = self.active_trades[trade_id]
        
        # Calculate PnL
        if trade['action'] == 'BUY':
            pnl_points = exit_price - trade['entry_price']
        else:
            pnl_points = trade['entry_price'] - exit_price
        
        pnl_total = pnl_points * trade['quantity']
        
        closed_trade = {
            **trade,
            'exit_price': exit_price,
            'exit_time': datetime.now(IST),
            'exit_reason': exit_reason,
            'pnl_points': pnl_points,
            'pnl_total': pnl_total,
            'trade_id': trade_id
        }
        
        self.closed_trades.append(closed_trade)
        del self.active_trades[trade_id]
    
    def get_statistics(self) -> Dict:
        """Get trading statistics"""
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'accuracy': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
        
        winning = [t for t in self.closed_trades if t['pnl_total'] > 0]
        losing = [t for t in self.closed_trades if t['pnl_total'] < 0]
        
        total_pnl = sum(t['pnl_total'] for t in self.closed_trades)
        avg_win = np.mean([t['pnl_total'] for t in winning]) if winning else 0
        avg_loss = np.mean([t['pnl_total'] for t in losing]) if losing else 0
        
        return {
            'total_trades': len(self.closed_trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'accuracy': len(winning) / len(self.closed_trades) * 100 if self.closed_trades else 0,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }

# ==================== PLOTTING ====================

def plot_charts(data1: pd.DataFrame, data2: pd.DataFrame, ratio_info: Dict, 
               ticker1_name: str, ticker2_name: str):
    """Create comprehensive charts for ratio strategy"""
    
    rsi1 = calculate_rsi(data1)
    rsi2 = calculate_rsi(data2)
    ratio_series = ratio_info['ratio']
    ratio_df = pd.DataFrame({'Close': ratio_series})
    ratio_rsi = calculate_rsi(ratio_df)
    
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            f'{ticker1_name} Price',
            f'{ticker2_name} Price',
            f'Ratio ({ticker1_name}/{ticker2_name})',
            'RSI Indicators'
        ),
        vertical_spacing=0.08,
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )
    
    # Ticker 1 candlestick
    fig.add_trace(
        go.Candlestick(
            x=data1.index,
            open=data1['Open'],
            high=data1['High'],
            low=data1['Low'],
            close=data1['Close'],
            name=ticker1_name
        ),
        row=1, col=1
    )
    
    # Ticker 2 candlestick
    fig.add_trace(
        go.Candlestick(
            x=data2.index,
            open=data2['Open'],
            high=data2['High'],
            low=data2['Low'],
            close=data2['Close'],
            name=ticker2_name
        ),
        row=2, col=1
    )
    
    # Ratio line
    fig.add_trace(
        go.Scatter(
            x=ratio_series.index,
            y=ratio_series.values,
            mode='lines',
            name='Ratio',
            line=dict(color='purple', width=2)
        ),
        row=3, col=1
    )
    
    # RSI plots
    fig.add_trace(
        go.Scatter(x=rsi1.index, y=rsi1.values, name=f'{ticker1_name} RSI', 
                  line=dict(color='blue')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=rsi2.index, y=rsi2.values, name=f'{ticker2_name} RSI',
                  line=dict(color='green')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=ratio_rsi.index, y=ratio_rsi.values, name='Ratio RSI',
                  line=dict(color='red')),
        row=4, col=1
    )
    
    # Add RSI levels
    for level in [30, 70]:
        fig.add_hline(y=level, line_dash="dash", line_color="gray", row=4, col=1)
    
    fig.update_layout(
        height=1200,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    
    return fig

def plot_single_chart(data: pd.DataFrame, ticker_name: str, strategy: str, signal: Dict = None):
    """Create chart for single ticker strategies"""
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f'{ticker_name} Price',
            'RSI',
            'Volume'
        ),
        vertical_spacing=0.1,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=ticker_name
        ),
        row=1, col=1
    )
    
    # Add EMAs if strategy is EMA Crossover
    if strategy == "EMA Crossover" and signal:
        if 'ema1' in signal and 'ema2' in signal:
            ema1 = calculate_ema(data, 9)
            ema2 = calculate_ema(data, 15)
            fig.add_trace(
                go.Scatter(x=data.index, y=ema1, name='EMA 9', line=dict(color='cyan', width=1)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=ema2, name='EMA 15', line=dict(color='orange', width=1)),
                row=1, col=1
            )
    
    # Add entry/target/sl markers if signal exists
    if signal and signal['action'] != 'HOLD':
        last_time = data.index[-1]
        fig.add_trace(
            go.Scatter(
                x=[last_time], y=[signal['entry']],
                mode='markers',
                marker=dict(color='yellow', size=10, symbol='circle'),
                name='Entry'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[last_time], y=[signal['target']],
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name='Target'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[last_time], y=[signal['sl']],
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name='SL'
            ),
            row=1, col=1
        )
    
    # RSI
    rsi = calculate_rsi(data)
    fig.add_trace(
        go.Scatter(x=data.index, y=rsi, name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    
    # Volume
    colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
             for i in range(len(data))]
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors),
        row=3, col=1
    )
    
    fig.update_layout(
        height=900,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    
    return fig

# ==================== DHAN ORDER PLACEHOLDER ====================

def place_dhan_order(ticker: str, action: str, quantity: int, price: float):
    """
    Placeholder for Dhan API integration
    
    # Uncomment and configure for live trading:
    # from dhanhq import dhanhq
    # 
    # dhan = dhanhq("client_id", "access_token")
    # 
    # order = dhan.place_order(
    #     security_id='ticker_id',
    #     exchange_segment='NSE_EQ',
    #     transaction_type='BUY' if action == 'BUY' else 'SELL',
    #     quantity=quantity,
    #     order_type='LIMIT',
    #     price=price,
    #     product_type='INTRADAY',
    #     validity='DAY'
    # )
    # 
    # return order
    """
    pass

# ==================== STREAMLIT UI ====================

def initialize_session_state():
    """Initialize session state variables"""
    if 'trade_manager' not in st.session_state:
        st.session_state.trade_manager = TradeManager()
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now(IST)
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    if 'live_running' not in st.session_state:
        st.session_state.live_running = False
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    if 'live_iteration' not in st.session_state:
        st.session_state.live_iteration = 0

def add_log(message: str, level: str = "INFO"):
    """Add log entry"""
    timestamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.logs.append({
        'timestamp': timestamp,
        'level': level,
        'message': message
    })
    # Keep only last 500 logs
    if len(st.session_state.logs) > 500:
        st.session_state.logs = st.session_state.logs[-500:]

def main():
    st.set_page_config(
        page_title="Quantitative Trading System v2.0",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    st.title("📈 Production Quantitative Trading System v2.0")
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Strategy Selection
        strategy = st.selectbox(
            "Trading Strategy",
            ["Ratio Strategy", "EMA Crossover", "RSI Strategy", "AI-based", "Custom Strategy"]
        )
        
        st.divider()
        
        # Mode Selection
        mode = st.selectbox("Mode", ["Backtesting", "Live Trading"])
        
        # Ticker Selection
        st.subheader("Asset Selection")
        
        if strategy == "Ratio Strategy":
            asset_group1 = st.selectbox("Ticker 1 Group", list(ASSET_GROUPS.keys()), key='ag1')
            ticker1_name = st.selectbox("Ticker 1", list(ASSET_GROUPS[asset_group1].keys()))
            ticker1 = ASSET_GROUPS[asset_group1][ticker1_name]
            
            asset_group2 = st.selectbox("Ticker 2 Group", list(ASSET_GROUPS.keys()), key='ag2')
            ticker2_name = st.selectbox("Ticker 2", list(ASSET_GROUPS[asset_group2].keys()))
            ticker2 = ASSET_GROUPS[asset_group2][ticker2_name]
        else:
            asset_group = st.selectbox("Asset Group", list(ASSET_GROUPS.keys()))
            ticker_name = st.selectbox("Ticker", list(ASSET_GROUPS[asset_group].keys()))
            ticker = ASSET_GROUPS[asset_group][ticker_name]
        
        # Custom ticker
        use_custom = st.checkbox("Use Custom Ticker")
        if use_custom:
            custom_ticker = st.text_input("Custom ticker (yfinance format)")
            if custom_ticker:
                if strategy == "Ratio Strategy":
                    ticker1 = custom_ticker
                else:
                    ticker = custom_ticker
        
        st.divider()
        
        # Timeframe & Period
        st.subheader("Timeframe")
        interval = st.selectbox("Interval", list(TIMEFRAME_PERIODS.keys()))
        period = st.selectbox("Period", TIMEFRAME_PERIODS[interval])
        
        # Quantity
        quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
        
        st.divider()
        
        # Strategy-specific parameters
        if strategy == "Ratio Strategy":
            st.subheader("Ratio Parameters")
            num_bins = st.number_input("Number of Bins", min_value=3, max_value=20, value=5)
            n_candles = st.number_input("Candles Lookforward", min_value=1, max_value=50, value=3)
        
        elif strategy == "EMA Crossover":
            st.subheader("EMA Parameters")
            ema1_period = st.number_input("Fast EMA", min_value=2, max_value=200, value=9)
            ema2_period = st.number_input("Slow EMA", min_value=2, max_value=200, value=15)
            
            use_angle = st.checkbox("Use Angle Filter")
            min_angle = 1.0
            if use_angle:
                min_angle = st.number_input("Min Angle (degrees)", min_value=0.1, max_value=90.0, value=1.0)
            
            crossover_type = st.selectbox("Crossover Type", 
                                         ["Simple", "Candle Size", "ATR-based"])
            
            min_candle_size = 10.0
            atr_threshold = 0.5
            if crossover_type == "Candle Size":
                min_candle_size = st.number_input("Min Candle Size (points)", value=10.0)
            elif crossover_type == "ATR-based":
                atr_threshold = st.number_input("ATR Threshold", value=0.5)
        
        elif strategy == "RSI Strategy":
            st.subheader("RSI Parameters")
            rsi_period = st.number_input("RSI Period", min_value=2, max_value=50, value=14)
            oversold = st.number_input("Oversold Level", min_value=10, max_value=40, value=30)
            overbought = st.number_input("Overbought Level", min_value=60, max_value=90, value=70)
        
        elif strategy == "Custom Strategy":
            st.subheader("Custom Indicator Selection")
            use_rsi_custom = st.checkbox("Use RSI", value=True)
            use_ema_custom = st.checkbox("Use EMA", value=True)
            use_momentum = st.checkbox("Use Momentum", value=False)
            use_volume = st.checkbox("Use Volume", value=False)
            min_confirmations = st.number_input("Min Confirmations", min_value=1, max_value=4, value=2)
        
        st.divider()
        
        # SL & Target Configuration
        st.subheader("Stop Loss Configuration")
        use_system_sl = st.checkbox("Use System-based SL", value=True)
        
        sl_type = 'custom_points'
        sl_value = 10.0
        atr_multiplier = 2.0
        
        if use_system_sl:
            sl_type = st.selectbox(
                "SL Type",
                ["custom_points", "atr_based", "previous_candle", "current_candle", 
                 "previous_swing", "current_swing", "ema_crossover_reversal",
                 "trailing_previous_candle", "trailing_current_candle",
                 "trailing_previous_swing", "trailing_current_swing"],
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            if sl_type == 'custom_points':
                sl_value = st.number_input("SL Points", value=10.0, disabled=False)
            elif sl_type == 'atr_based':
                atr_multiplier = st.number_input("ATR Multiplier", value=2.0, disabled=False)
            else:
                st.info(f"Using {sl_type.replace('_', ' ')}")
        else:
            sl_value = st.number_input("Manual SL Points", value=10.0)
        
        st.subheader("Target Configuration")
        use_system_target = st.checkbox("Use System-based Target", value=True)
        
        target_type = 'custom_points'
        target_value = 20.0
        target_atr_multiplier = 3.0
        
        if use_system_target:
            target_type = st.selectbox(
                "Target Type",
                ["custom_points", "atr_based", "previous_candle", "current_candle",
                 "previous_swing", "current_swing", "ema_crossover_reversal",
                 "trailing_previous_candle", "trailing_current_candle",
                 "trailing_previous_swing", "trailing_current_swing"],
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            if target_type == 'custom_points':
                target_value = st.number_input("Target Points", value=20.0, disabled=False)
            elif target_type == 'atr_based':
                target_atr_multiplier = st.number_input("Target ATR Multiplier", value=3.0, disabled=False)
            else:
                st.info(f"Using {target_type.replace('_', ' ')}")
        else:
            target_value = st.number_input("Manual Target Points", value=20.0)
    
    # Prepare config
    config = {
        'sl_type': sl_type,
        'sl_value': sl_value,
        'atr_multiplier': atr_multiplier,
        'target_type': target_type,
        'target_value': target_value,
        'target_atr_multiplier': target_atr_multiplier,
    }
    
    if strategy == "EMA Crossover":
        config.update({
            'use_angle': use_angle if 'use_angle' in locals() else False,
            'min_angle': min_angle if 'min_angle' in locals() else 1.0,
            'crossover_type': crossover_type.lower().replace(' ', '_').replace('-', '_') if 'crossover_type' in locals() else 'simple',
            'min_candle_size': min_candle_size if 'min_candle_size' in locals() else 10,
            'atr_threshold': atr_threshold if 'atr_threshold' in locals() else 0.5
        })
    
    # ==================== TABS ====================
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Trading", "📜 Trade History", "📝 Logs", "🔬 Backtesting"])
    
    # ==================== TAB 1: LIVE TRADING ====================
    with tab1:
        st.header("Live Trading Dashboard")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
        
        with col_btn1:
            if st.button("▶️ Start Live", use_container_width=True):
                st.session_state.live_running = True
                add_log("Live trading started", "INFO")
                st.rerun()
        
        with col_btn2:
            if st.button("⏸️ Stop Live", use_container_width=True):
                st.session_state.live_running = False
                add_log("Live trading stopped", "INFO")
                st.rerun()
        
        # Placeholders
        metrics_placeholder = st.empty()
        signal_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        # CONTINUOUS LIVE LOOP
        if st.session_state.live_running:
            try:
                if strategy == "Ratio Strategy":
                    # Fetch data
                    data1 = fetch_data(ticker1, interval, period, is_live=(mode == "Live Trading"))
                    data2 = fetch_data(ticker2, interval, period, is_live=(mode == "Live Trading"))
                    
                    if not data1.empty and not data2.empty:
                        # Align by timezone
                        data1, data2 = align_dataframes_by_timezone(data1, data2)
                        
                        if len(data1) == 0 or len(data2) == 0:
                            st.error("No common data after timezone alignment")
                        else:
                            # Calculate
                            ratio_info = calculate_ratio_bins(data1, data2, num_bins)
                            expected_moves = calculate_expected_moves(data1, data2, ratio_info, n_candles)
                            
                            atr1 = calculate_atr(data1)
                            atr2 = calculate_atr(data2)
                            
                            signals = generate_ratio_signals(data1, data2, ratio_info, expected_moves, 
                                                            atr1, atr2, config)
                            
                            # Update metrics
                            with metrics_placeholder.container():
                                col1, col2, col3, col4, col5 = st.columns(5)
                                
                                col1.metric("Ticker 1", ticker1_name, f"₹{data1['Close'].iloc[-1]:.2f}")
                                col2.metric("Ticker 2", ticker2_name, f"₹{data2['Close'].iloc[-1]:.2f}")
                                col3.metric("Ratio", f"{ratio_info['ratio'].iloc[-1]:.4f}")
                                col4.metric("Active Trades", len(st.session_state.trade_manager.active_trades))
                                stats = st.session_state.trade_manager.get_statistics()
                                col5.metric("Total PnL", f"₹{stats['total_pnl']:.2f}")
                            
                            # Signals
                            with signal_placeholder.container():
                                st.subheader("Current Signals")
                                col_s1, col_s2 = st.columns(2)
                                
                                with col_s1:
                                    sig1 = signals['ticker1']
                                    st.info(f"**{ticker1_name}**: {sig1['action']}")
                                    if sig1['action'] != 'HOLD':
                                        st.write(f"**Entry**: ₹{sig1['entry']:.2f}")
                                        st.write(f"**Target**: ₹{sig1['target']:.2f}")
                                        st.write(f"**SL**: ₹{sig1['sl']:.2f}")
                                
                                with col_s2:
                                    sig2 = signals['ticker2']
                                    st.info(f"**{ticker2_name}**: {sig2['action']}")
                                    if sig2['action'] != 'HOLD':
                                        st.write(f"**Entry**: ₹{sig2['entry']:.2f}")
                                        st.write(f"**Target**: ₹{sig2['target']:.2f}")
                                        st.write(f"**SL**: ₹{sig2['sl']:.2f}")
                                
                                st.caption(f"Reason: {signals['reason']}")
                            
                            # Charts
                            with chart_placeholder.container():
                                fig = plot_charts(data1, data2, ratio_info, ticker1_name, ticker2_name)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            add_log(f"Update #{st.session_state.live_iteration} - {ticker1_name}: {signals['ticker1']['action']}, {ticker2_name}: {signals['ticker2']['action']}", "INFO")
                            st.session_state.live_iteration += 1
                            st.session_state.last_update = datetime.now(IST)
                            
                            # Auto refresh
                            time.sleep(random.uniform(1.0, 1.5))
                            st.rerun()
                
                else:  # Other strategies
                    data = fetch_data(ticker, interval, period, is_live=(mode == "Live Trading"))
                    
                    if not data.empty:
                        # Generate signal
                        if strategy == "EMA Crossover":
                            signal = generate_ema_signals(data, ema1_period, ema2_period, config)
                        elif strategy == "RSI Strategy":
                            signal = generate_rsi_signals(data, rsi_period, oversold, overbought, config)
                        elif strategy == "AI-based":
                            signal = generate_ai_signals(data, config)
                        elif strategy == "Custom Strategy":
                            indicators_config = {
                                'use_rsi': use_rsi_custom,
                                'use_ema': use_ema_custom,
                                'use_momentum': use_momentum,
                                'use_volume': use_volume,
                                'min_confirmations': min_confirmations,
                                'rsi_period': 14,
                                'rsi_oversold': 30,
                                'rsi_overbought': 70,
                                'ema_fast': 9,
                                'ema_slow': 21,
                                'momentum_period': 10
                            }
                            signal = generate_custom_signals(data, indicators_config, config)
                        else:
                            signal = {'action': 'HOLD', 'entry': 0, 'target': 0, 'sl': 0, 'reason': 'Strategy not implemented'}
                        
                        # Metrics
                        with metrics_placeholder.container():
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            col1.metric("Price", f"₹{data['Close'].iloc[-1]:.2f}")
                            col2.metric("Signal", signal['action'])
                            
                            if strategy == "EMA Crossover" and 'ema1' in signal:
                                col3.metric("EMA9", f"₹{signal['ema1']:.2f}")
                                col4.metric("EMA15", f"₹{signal['ema2']:.2f}")
                            elif strategy == "RSI Strategy" and 'rsi' in signal:
                                col3.metric("RSI", f"{signal['rsi']:.1f}")
                            elif strategy == "AI-based" and 'confidence' in signal:
                                col3.metric("AI Confidence", f"{signal['confidence']}")
                            
                            stats = st.session_state.trade_manager.get_statistics()
                            col5.metric("Total PnL", f"₹{stats['total_pnl']:.2f}")
                        
                        # Signal
                        with signal_placeholder.container():
                            if signal['action'] != 'HOLD':
                                st.success(f"**Signal**: {signal['action']}")
                                col_e, col_t, col_s = st.columns(3)
                                col_e.write(f"**Entry**: ₹{signal['entry']:.2f}")
                                col_t.write(f"**Target**: ₹{signal['target']:.2f}")
                                col_s.write(f"**SL**: ₹{signal['sl']:.2f}")
                                st.caption(f"Reason: {signal['reason']}")
                            else:
                                st.info(f"No active signal - {signal['reason']}")
                        
                        # Chart
                        with chart_placeholder.container():
                            fig = plot_single_chart(data, ticker_name, strategy, signal)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        add_log(f"Update #{st.session_state.live_iteration} - Signal: {signal['action']}", "INFO")
                        st.session_state.live_iteration += 1
                        st.session_state.last_update = datetime.now(IST)
                        
                        # Auto refresh
                        time.sleep(random.uniform(1.0, 1.5))
                        st.rerun()
            
            except Exception as e:
                st.error(f"Live trading error: {str(e)}")
                add_log(f"Error: {str(e)}", "ERROR")
                st.session_state.live_running = False
        
        else:
            # Manual refresh when not running
            if st.button("🔄 Refresh Data"):
                st.rerun()
            st.info("Live trading is stopped. Click ▶️ Start Live to begin continuous updates.")
    
    # ==================== TAB 2: TRADE HISTORY ====================
    with tab2:
        st.header("Trade History")
        
        stats = st.session_state.trade_manager.get_statistics()
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total Trades", stats['total_trades'])
        col2.metric("Winning", stats['winning_trades'])
        col3.metric("Losing", stats['losing_trades'])
        col4.metric("Accuracy", f"{stats['accuracy']:.2f}%")
        col5.metric("Total PnL", f"₹{stats['total_pnl']:.2f}")
        col6.metric("Avg Win", f"₹{stats['avg_win']:.2f}")
        
        st.divider()
        
        if st.session_state.trade_manager.closed_trades:
            trades_df = pd.DataFrame(st.session_state.trade_manager.closed_trades)
            
            display_df = trades_df[[
                'trade_id', 'ticker', 'action', 'entry_price', 'exit_price',
                'entry_time', 'exit_time', 'target', 'sl', 'pnl_points',
                'pnl_total', 'exit_reason', 'reason'
            ]].copy()
            
            display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['exit_time'] = pd.to_datetime(display_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            st.dataframe(display_df, use_container_width=True, height=600)
        else:
            st.info("No closed trades yet")
    
    # ==================== TAB 3: LOGS ====================
    with tab3:
        st.header("System Logs")
        
        if st.button("Clear Logs"):
            st.session_state.logs = []
            st.rerun()
        
        if st.session_state.logs:
            logs_df = pd.DataFrame(st.session_state.logs)
            st.dataframe(logs_df, use_container_width=True, height=600)
        else:
            st.info("No logs yet")
    
    # ==================== TAB 4: BACKTESTING ====================
    with tab4:
        st.header("Backtesting")
        
        if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
            with st.spinner("Running backtest..."):
                try:
                    backtest_tm = TradeManager()
                    
                    if strategy == "Ratio Strategy":
                        data1 = fetch_data(ticker1, interval, period, is_live=False)
                        data2 = fetch_data(ticker2, interval, period, is_live=False)
                        
                        if not data1.empty and not data2.empty:
                            data1, data2 = align_dataframes_by_timezone(data1, data2)
                            
                            for i in range(max(50, n_candles), len(data1) - n_candles):
                                window_data1 = data1.iloc[:i+1]
                                window_data2 = data2.iloc[:i+1]
                                
                                ratio_info = calculate_ratio_bins(window_data1, window_data2, num_bins)
                                expected_moves = calculate_expected_moves(window_data1, window_data2, 
                                                                         ratio_info, n_candles)
                                
                                atr1 = calculate_atr(window_data1)
                                atr2 = calculate_atr(window_data2)
                                
                                signals = generate_ratio_signals(window_data1, window_data2, ratio_info,
                                                                expected_moves, atr1, atr2, config)
                                
                                # Process both tickers
                                for ticker_key, ticker_sig_name in [('ticker1', ticker1_name), ('ticker2', ticker2_name)]:
                                    sig = signals[ticker_key]
                                    if sig['action'] != 'HOLD':
                                        trade_id = backtest_tm.open_trade(
                                            ticker_sig_name, sig['action'], sig['entry'],
                                            sig['target'], sig['sl'], quantity, signals['reason']
                                        )
                    
                    else:
                        data = fetch_data(ticker, interval, period, is_live=False)
                        
                        if not data.empty:
                            for i in range(50, len(data)):
                                window_data = data.iloc[:i+1]
                                
                                if strategy == "EMA Crossover":
                                    signal = generate_ema_signals(window_data, ema1_period, ema2_period, config)
                                elif strategy == "RSI Strategy":
                                    signal = generate_rsi_signals(window_data, rsi_period, oversold, overbought, config)
                                elif strategy == "AI-based":
                                    signal = generate_ai_signals(window_data, config)
                                elif strategy == "Custom Strategy":
                                    indicators_config = {
                                        'use_rsi': use_rsi_custom,
                                        'use_ema': use_ema_custom,
                                        'use_momentum': use_momentum,
                                        'use_volume': use_volume,
                                        'min_confirmations': min_confirmations
                                    }
                                    signal = generate_custom_signals(window_data, indicators_config, config)
                                else:
                                    continue
                                
                                if signal['action'] != 'HOLD':
                                    trade_id = backtest_tm.open_trade(
                                        ticker_name, signal['action'], signal['entry'],
                                        signal['target'], signal['sl'], quantity, signal['reason']
                                    )
                    
                    st.session_state.backtest_results = backtest_tm
                    add_log("Backtest completed successfully", "SUCCESS")
                    st.success("Backtest completed!")
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Backtest error: {str(e)}")
                    add_log(f"Backtest error: {str(e)}", "ERROR")
        
        # Display results
        if st.session_state.backtest_results:
            bt_stats = st.session_state.backtest_results.get_statistics()
            
            st.subheader("Backtest Results")
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Total Trades", bt_stats['total_trades'])
            col2.metric("Winning", bt_stats['winning_trades'])
            col3.metric("Losing", bt_stats['losing_trades'])
            col4.metric("Accuracy", f"{bt_stats['accuracy']:.2f}%")
            col5.metric("Total PnL", f"₹{bt_stats['total_pnl']:.2f}")
            col6.metric("Avg Win", f"₹{bt_stats['avg_win']:.2f}")
            
            st.divider()
            
            if st.session_state.backtest_results.closed_trades:
                bt_df = pd.DataFrame(st.session_state.backtest_results.closed_trades)
                
                display_bt_df = bt_df[[
                    'trade_id', 'ticker', 'action', 'entry_price', 'exit_price',
                    'entry_time', 'exit_time', 'target', 'sl', 'pnl_points',
                    'pnl_total', 'exit_reason', 'reason'
                ]].copy()
                
                display_bt_df['entry_time'] = pd.to_datetime(display_bt_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
                display_bt_df['exit_time'] = pd.to_datetime(display_bt_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.dataframe(display_bt_df, use_container_width=True, height=500)
        else:
            st.info("Run a backtest to see results")
    
    # Footer
    st.divider()
    last_update_str = st.session_state.last_update.strftime("%Y-%m-%d %H:%M:%S IST")
    st.caption(f"Last Update: {last_update_str} | Mode: {mode} | Strategy: {strategy} | v2.0")

if __name__ == "__main__":
    main()
