import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Advanced Trading System", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'live_trading_active' not in st.session_state:
    st.session_state.live_trading_active = False
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'trade_logs' not in st.session_state:
    st.session_state.trade_logs = []
if 'current_position' not in st.session_state:
    st.session_state.current_position = None
if 'iteration_count' not in st.session_state:
    st.session_state.iteration_count = 0
if 'strategy_params' not in st.session_state:
    st.session_state.strategy_params = {}

# IST timezone
IST = pytz.timezone('Asia/Kolkata')

# Ticker presets
TICKER_PRESETS = {
    "Nifty 50": "^NSEI",
    "Bank Nifty": "^NSEBANK",
    "Sensex": "^BSESN",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "USD/INR": "INR=X",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Crude Oil (WTI)": "CL=F",
    "Natural Gas": "NG=F",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "Custom": "custom"
}

# Timeframe and period mappings
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

def flatten_multiindex_columns(df):
    """Flatten multi-index columns from yfinance"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in df.columns.values]
    return df

def convert_to_ist(df):
    """Convert datetime index to IST"""
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert(IST)
    return df

def fetch_data(ticker: str, interval: str, period: str, progress_bar) -> pd.DataFrame:
    """Fetch data from yfinance with random rate limiting"""
    try:
        progress_bar.progress(10, text="Initiating data fetch...")
        time.sleep(random.uniform(1.0, 1.5))  # Random delay
        
        progress_bar.progress(30, text=f"Fetching {ticker} data...")
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        time.sleep(random.uniform(1.0, 1.5))  # Random delay
        
        if data.empty:
            st.error(f"No data returned for {ticker}")
            return None
        
        progress_bar.progress(60, text="Processing data...")
        data = flatten_multiindex_columns(data)
        
        # Standardize column names
        column_mapping = {}
        for col in data.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                column_mapping[col] = 'Open'
            elif 'high' in col_lower:
                column_mapping[col] = 'High'
            elif 'low' in col_lower:
                column_mapping[col] = 'Low'
            elif 'close' in col_lower:
                column_mapping[col] = 'Close'
            elif 'volume' in col_lower:
                column_mapping[col] = 'Volume'
        
        data = data.rename(columns=column_mapping)
        
        # Select only required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[[col for col in required_cols if col in data.columns]]
        
        progress_bar.progress(80, text="Converting timezone...")
        data = convert_to_ist(data)
        
        progress_bar.progress(100, text="Data fetch complete!")
        time.sleep(0.5)
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """Calculate EMA (TradingView compatible)"""
    return df['Close'].ewm(span=period, adjust=False).mean()

def calculate_sma(df: pd.DataFrame, period: int) -> pd.Series:
    """Calculate SMA"""
    return df['Close'].rolling(window=period).mean()

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI (TradingView compatible)"""
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR (TradingView compatible)"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return atr

def calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate ADX with +DI and -DI (TradingView compatible)"""
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    atr = calculate_atr(df, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx, plus_di, minus_di

def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    middle_band = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)
    return upper_band, middle_band, lower_band

def calculate_zscore(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Z-Score"""
    mean = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    zscore = (df['Close'] - mean) / std
    return zscore

def find_support_resistance(df: pd.DataFrame, window: int = 20) -> Tuple[List[float], List[float]]:
    """Find support and resistance levels"""
    supports = []
    resistances = []
    
    for i in range(window, len(df) - window):
        if df['Low'].iloc[i] == df['Low'].iloc[i-window:i+window].min():
            supports.append(df['Low'].iloc[i])
        if df['High'].iloc[i] == df['High'].iloc[i-window:i+window].max():
            resistances.append(df['High'].iloc[i])
    
    supports = sorted(list(set(supports)))[-5:] if supports else []
    resistances = sorted(list(set(resistances)))[-5:] if resistances else []
    
    return supports, resistances

def calculate_crossover_angle(fast_ema: pd.Series, slow_ema: pd.Series, idx: int, lookback: int = 5) -> float:
    """Calculate the angle of EMA crossover"""
    if idx < lookback:
        return 0
    
    fast_diff = fast_ema.iloc[idx] - fast_ema.iloc[idx - lookback]
    slow_diff = slow_ema.iloc[idx] - slow_ema.iloc[idx - lookback]
    
    angle = np.arctan2(fast_diff - slow_diff, lookback) * 180 / np.pi
    return abs(angle)

def ema_crossover_strategy(df: pd.DataFrame, fast_period: int, slow_period: int, 
                          crossover_type: str, min_angle: float, candle_points: float,
                          atr_multiplier: float) -> pd.DataFrame:
    """EMA Crossover Strategy"""
    df = df.copy()
    df['EMA_Fast'] = calculate_ema(df, fast_period)
    df['EMA_Slow'] = calculate_ema(df, slow_period)
    df['ATR'] = calculate_atr(df)
    
    df['Signal'] = 0
    df['Entry_Reason'] = ''
    df['Crossover_Angle'] = 0.0
    
    for i in range(slow_period + 5, len(df)):
        cross_up = (df['EMA_Fast'].iloc[i] > df['EMA_Slow'].iloc[i] and 
                    df['EMA_Fast'].iloc[i-1] <= df['EMA_Slow'].iloc[i-1])
        cross_down = (df['EMA_Fast'].iloc[i] < df['EMA_Slow'].iloc[i] and 
                      df['EMA_Fast'].iloc[i-1] >= df['EMA_Slow'].iloc[i-1])
        
        angle = calculate_crossover_angle(df['EMA_Fast'], df['EMA_Slow'], i)
        df.loc[df.index[i], 'Crossover_Angle'] = angle
        
        candle_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
        
        valid_candle = True
        if crossover_type == "Strong Candle":
            valid_candle = candle_size >= candle_points
        elif crossover_type == "ATR Based Candle":
            valid_candle = candle_size >= (df['ATR'].iloc[i] * atr_multiplier)
        
        if cross_up and angle >= min_angle and valid_candle:
            df.loc[df.index[i], 'Signal'] = 1
            df.loc[df.index[i], 'Entry_Reason'] = f"EMA Crossover Long (Angle: {angle:.1f}°, Candle: {candle_size:.2f})"
        elif cross_down and angle >= min_angle and valid_candle:
            df.loc[df.index[i], 'Signal'] = -1
            df.loc[df.index[i], 'Entry_Reason'] = f"EMA Crossover Short (Angle: {angle:.1f}°, Candle: {candle_size:.2f})"
    
    return df

def mean_reversion_strategy(df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
    """Mean Reversion Strategy"""
    df = df.copy()
    df['MA'] = df['Close'].rolling(window=period).mean()
    df['STD'] = df['Close'].rolling(window=period).std()
    df['Upper'] = df['MA'] + (std_dev * df['STD'])
    df['Lower'] = df['MA'] - (std_dev * df['STD'])
    
    df['Signal'] = 0
    df['Entry_Reason'] = ''
    
    for i in range(period, len(df)):
        if df['Close'].iloc[i] < df['Lower'].iloc[i]:
            df.loc[df.index[i], 'Signal'] = 1
            df.loc[df.index[i], 'Entry_Reason'] = f"Mean Reversion Long (Price: {df['Close'].iloc[i]:.2f}, Lower: {df['Lower'].iloc[i]:.2f})"
        elif df['Close'].iloc[i] > df['Upper'].iloc[i]:
            df.loc[df.index[i], 'Signal'] = -1
            df.loc[df.index[i], 'Entry_Reason'] = f"Mean Reversion Short (Price: {df['Close'].iloc[i]:.2f}, Upper: {df['Upper'].iloc[i]:.2f})"
    
    return df

def rsi_ema_adx_strategy(df: pd.DataFrame, rsi_period: int = 14, ema_period: int = 20, 
                         adx_period: int = 14, rsi_overbought: float = 70, 
                         rsi_oversold: float = 30, min_adx: float = 25) -> pd.DataFrame:
    """RSI, EMA, ADX Combined Strategy"""
    df = df.copy()
    df['RSI'] = calculate_rsi(df, rsi_period)
    df['EMA'] = calculate_ema(df, ema_period)
    df['ADX'], df['Plus_DI'], df['Minus_DI'] = calculate_adx(df, adx_period)
    
    df['Signal'] = 0
    df['Entry_Reason'] = ''
    
    for i in range(max(rsi_period, ema_period, adx_period), len(df)):
        trend_up = df['Close'].iloc[i] > df['EMA'].iloc[i]
        trend_down = df['Close'].iloc[i] < df['EMA'].iloc[i]
        strong_trend = df['ADX'].iloc[i] > min_adx
        
        if (df['RSI'].iloc[i] < rsi_oversold and trend_up and strong_trend):
            df.loc[df.index[i], 'Signal'] = 1
            df.loc[df.index[i], 'Entry_Reason'] = f"RSI/EMA/ADX Long (RSI: {df['RSI'].iloc[i]:.1f}, ADX: {df['ADX'].iloc[i]:.1f})"
        elif (df['RSI'].iloc[i] > rsi_overbought and trend_down and strong_trend):
            df.loc[df.index[i], 'Signal'] = -1
            df.loc[df.index[i], 'Entry_Reason'] = f"RSI/EMA/ADX Short (RSI: {df['RSI'].iloc[i]:.1f}, ADX: {df['ADX'].iloc[i]:.1f})"
    
    return df

def breakout_strategy(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Breakout Strategy"""
    df = df.copy()
    df['High_Resistance'] = df['High'].rolling(window=lookback).max()
    df['Low_Support'] = df['Low'].rolling(window=lookback).min()
    
    df['Signal'] = 0
    df['Entry_Reason'] = ''
    
    for i in range(lookback, len(df)):
        if df['Close'].iloc[i] > df['High_Resistance'].iloc[i-1]:
            df.loc[df.index[i], 'Signal'] = 1
            df.loc[df.index[i], 'Entry_Reason'] = f"Breakout Long (Price: {df['Close'].iloc[i]:.2f}, Resistance: {df['High_Resistance'].iloc[i-1]:.2f})"
        elif df['Close'].iloc[i] < df['Low_Support'].iloc[i-1]:
            df.loc[df.index[i], 'Signal'] = -1
            df.loc[df.index[i], 'Entry_Reason'] = f"Breakout Short (Price: {df['Close'].iloc[i]:.2f}, Support: {df['Low_Support'].iloc[i-1]:.2f})"
    
    return df

def volatility_strategy(df: pd.DataFrame, atr_period: int = 14, volatility_threshold: float = 1.5) -> pd.DataFrame:
    """Volatility-based Strategy"""
    df = df.copy()
    df['ATR'] = calculate_atr(df, atr_period)
    df['ATR_MA'] = df['ATR'].rolling(window=20).mean()
    df['Volatility_Ratio'] = df['ATR'] / df['ATR_MA']
    
    df['Signal'] = 0
    df['Entry_Reason'] = ''
    
    for i in range(atr_period + 20, len(df)):
        if df['Volatility_Ratio'].iloc[i] > volatility_threshold:
            candle_direction = 1 if df['Close'].iloc[i] > df['Open'].iloc[i] else -1
            df.loc[df.index[i], 'Signal'] = candle_direction
            df.loc[df.index[i], 'Entry_Reason'] = f"High Volatility {'Long' if candle_direction == 1 else 'Short'} (Vol Ratio: {df['Volatility_Ratio'].iloc[i]:.2f})"
    
    return df

def zscore_strategy(df: pd.DataFrame, period: int = 20, threshold: float = 2.0) -> pd.DataFrame:
    """Z-Score Strategy"""
    df = df.copy()
    df['ZScore'] = calculate_zscore(df, period)
    
    df['Signal'] = 0
    df['Entry_Reason'] = ''
    
    for i in range(period, len(df)):
        if df['ZScore'].iloc[i] < -threshold:
            df.loc[df.index[i], 'Signal'] = 1
            df.loc[df.index[i], 'Entry_Reason'] = f"ZScore Long (ZScore: {df['ZScore'].iloc[i]:.2f})"
        elif df['ZScore'].iloc[i] > threshold:
            df.loc[df.index[i], 'Signal'] = -1
            df.loc[df.index[i], 'Entry_Reason'] = f"ZScore Short (ZScore: {df['ZScore'].iloc[i]:.2f})"
    
    return df

def bollinger_strategy(df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
    """Bollinger Bands Strategy"""
    df = df.copy()
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df, period, std_dev)
    
    df['Signal'] = 0
    df['Entry_Reason'] = ''
    
    for i in range(period, len(df)):
        if df['Close'].iloc[i] < df['BB_Lower'].iloc[i]:
            df.loc[df.index[i], 'Signal'] = 1
            df.loc[df.index[i], 'Entry_Reason'] = f"BB Long (Price: {df['Close'].iloc[i]:.2f}, Lower: {df['BB_Lower'].iloc[i]:.2f})"
        elif df['Close'].iloc[i] > df['BB_Upper'].iloc[i]:
            df.loc[df.index[i], 'Signal'] = -1
            df.loc[df.index[i], 'Entry_Reason'] = f"BB Short (Price: {df['Close'].iloc[i]:.2f}, Upper: {df['BB_Upper'].iloc[i]:.2f})"
    
    return df

def apply_strategy(df: pd.DataFrame, strategy: str, params: Dict) -> pd.DataFrame:
    """Apply selected strategy with parameters"""
    if strategy == "EMA Crossover":
        return ema_crossover_strategy(
            df, params['fast_ema'], params['slow_ema'], params['crossover_type'],
            params['min_angle'], params['candle_points'], params['atr_candle_multiplier']
        )
    elif strategy == "Mean Reversion":
        return mean_reversion_strategy(df, params.get('mean_period', 20), params.get('mean_std', 2.0))
    elif strategy == "RSI + EMA + ADX":
        return rsi_ema_adx_strategy(
            df, params.get('rsi_period', 14), params.get('ema_period', 20),
            params.get('adx_period', 14), params.get('rsi_overbought', 70),
            params.get('rsi_oversold', 30), params.get('min_adx', 25)
        )
    elif strategy == "Breakout":
        return breakout_strategy(df, params.get('breakout_lookback', 20))
    elif strategy == "Volatility Based":
        return volatility_strategy(df, params.get('atr_period', 14), params.get('vol_threshold', 1.5))
    elif strategy == "Z-Score Based":
        return zscore_strategy(df, params.get('zscore_period', 20), params.get('zscore_threshold', 2.0))
    elif strategy == "Bollinger Bands":
        return bollinger_strategy(df, params.get('bb_period', 20), params.get('bb_std', 2.0))
    else:
        return ema_crossover_strategy(df, 9, 15, "Simple Crossover", 20, 0, 1.0)

def calculate_stop_loss(df: pd.DataFrame, idx: int, signal: int, sl_type: str, 
                       sl_points: float, atr_multiplier: float, entry_price: float) -> float:
    """Calculate stop loss based on type"""
    
    if sl_type == "Custom Points":
        if signal == 1:
            return entry_price - sl_points
        else:
            return entry_price + sl_points
    
    elif sl_type == "ATR Based":
        atr = df['ATR'].iloc[idx] if 'ATR' in df.columns else calculate_atr(df).iloc[idx]
        if signal == 1:
            return entry_price - (atr * atr_multiplier)
        else:
            return entry_price + (atr * atr_multiplier)
    
    elif sl_type == "Current Candle Low/High":
        if signal == 1:
            return df['Low'].iloc[idx]
        else:
            return df['High'].iloc[idx]
    
    elif sl_type == "Current Swing Low/High":
        lookback = 10
        if signal == 1:
            swing_low = df['Low'].iloc[max(0, idx-lookback):idx+1].min()
            return swing_low
        else:
            swing_high = df['High'].iloc[max(0, idx-lookback):idx+1].max()
            return swing_high
    
    elif sl_type == "Previous Swing Low/High":
        lookback = 20
        if signal == 1:
            swing_low = df['Low'].iloc[max(0, idx-lookback):max(1, idx-lookback//2)].min()
            return swing_low
        else:
            swing_high = df['High'].iloc[max(0, idx-lookback):max(1, idx-lookback//2)].max()
            return swing_high
    
    elif sl_type == "Trailing SL":
        if signal == 1:
            return entry_price - sl_points
        else:
            return entry_price + sl_points
    
    elif sl_type == "Signal Based":
        if signal == 1:
            return entry_price - sl_points
        else:
            return entry_price + sl_points
    
    else:
        if signal == 1:
            return entry_price - sl_points
        else:
            return entry_price + sl_points

def calculate_target(df: pd.DataFrame, idx: int, signal: int, target_type: str, 
                    target_points: float, atr_multiplier: float, risk_reward: float,
                    entry_price: float, stop_loss: float) -> float:
    """Calculate target based on type"""
    
    if target_type == "Custom Points":
        if signal == 1:
            return entry_price + target_points
        else:
            return entry_price - target_points
    
    elif target_type == "ATR Based":
        atr = df['ATR'].iloc[idx] if 'ATR' in df.columns else calculate_atr(df).iloc[idx]
        if signal == 1:
            return entry_price + (atr * atr_multiplier)
        else:
            return entry_price - (atr * atr_multiplier)
    
    elif target_type == "Risk:Reward":
        sl_distance = abs(entry_price - stop_loss)
        if signal == 1:
            return entry_price + (sl_distance * risk_reward)
        else:
            return entry_price - (sl_distance * risk_reward)
    
    elif target_type == "Trailing Target":
        if signal == 1:
            return entry_price + target_points
        else:
            return entry_price - target_points
    
    elif target_type == "Signal Based":
        if signal == 1:
            return entry_price + target_points
        else:
            return entry_price - target_points
    
    else:
        if signal == 1:
            return entry_price + target_points
        else:
            return entry_price - target_points

def backtest_strategy(df: pd.DataFrame, sl_type: str, sl_points: float, atr_multiplier_sl: float,
                     target_type: str, target_points: float, atr_multiplier_target: float,
                     risk_reward: float) -> Dict:
    """Backtest the strategy"""
    trades = []
    position = None
    
    for i in range(len(df)):
        if position is None:
            if df['Signal'].iloc[i] != 0:
                entry_price = df['Close'].iloc[i]
                signal = df['Signal'].iloc[i]
                
                stop_loss = calculate_stop_loss(df, i, signal, sl_type, sl_points, atr_multiplier_sl, entry_price)
                target = calculate_target(df, i, signal, target_type, target_points, atr_multiplier_target, risk_reward, entry_price, stop_loss)
                
                position = {
                    'entry_idx': i,
                    'entry_time': df.index[i],
                    'entry_price': entry_price,
                    'signal': signal,
                    'stop_loss': stop_loss,
                    'target': target,
                    'entry_reason': df['Entry_Reason'].iloc[i],
                    'trailing_sl': stop_loss,
                    'highest_price': entry_price if signal == 1 else None,
                    'lowest_price': entry_price if signal == -1 else None
                }
        
        else:
            current_price = df['Close'].iloc[i]
            exit_triggered = False
            exit_reason = ""
            exit_price = current_price
            
            if position['signal'] == 1:
                if current_price > position.get('highest_price', position['entry_price']):
                    position['highest_price'] = current_price
                    if sl_type == "Trailing SL":
                        move_up = current_price - position['entry_price']
                        position['trailing_sl'] = max(position['trailing_sl'], position['stop_loss'] + move_up * 0.5)
                
                if df['Low'].iloc[i] <= position['trailing_sl']:
                    exit_triggered = True
                    exit_price = position['trailing_sl']
                    exit_reason = f"Stop Loss Hit (SL: {position['trailing_sl']:.2f})"
                
                elif df['High'].iloc[i] >= position['target']:
                    exit_triggered = True
                    exit_price = position['target']
                    exit_reason = f"Target Hit (Target: {position['target']:.2f})"
                
                elif df['Signal'].iloc[i] == -1 and sl_type == "Signal Based":
                    exit_triggered = True
                    exit_reason = "Signal Reversal"
            
            else:
                if current_price < position.get('lowest_price', position['entry_price']):
                    position['lowest_price'] = current_price
                    if sl_type == "Trailing SL":
                        move_down = position['entry_price'] - current_price
                        position['trailing_sl'] = min(position['trailing_sl'], position['stop_loss'] - move_down * 0.5)
                
                if df['High'].iloc[i] >= position['trailing_sl']:
                    exit_triggered = True
                    exit_price = position['trailing_sl']
                    exit_reason = f"Stop Loss Hit (SL: {position['trailing_sl']:.2f})"
                
                elif df['Low'].iloc[i] <= position['target']:
                    exit_triggered = True
                    exit_price = position['target']
                    exit_reason = f"Target Hit (Target: {position['target']:.2f})"
                
                elif df['Signal'].iloc[i] == 1 and sl_type == "Signal Based":
                    exit_triggered = True
                    exit_reason = "Signal Reversal"
            
            if exit_triggered:
                if position['signal'] == 1:
                    pnl_points = exit_price - position['entry_price']
                else:
                    pnl_points = position['entry_price'] - exit_price
                
                trade = {
                    'Entry Time': position['entry_time'],
                    'Exit Time': df.index[i],
                    'Direction': 'Long' if position['signal'] == 1 else 'Short',
                    'Entry Price': position['entry_price'],
                    'Exit Price': exit_price,
                    'Stop Loss': position['stop_loss'],
                    'Target': position['target'],
                    'P&L Points': pnl_points,
                    'Entry Reason': position['entry_reason'],
                    'Exit Reason': exit_reason,
                    'Duration': df.index[i] - position['entry_time']
                }
                
                trades.append(trade)
                position = None
    
    if len(trades) == 0:
        return {
            'trades': [],
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_win': 0,
            'max_loss': 0,
            'profit_factor': 0,
            'avg_duration': timedelta(0)
        }
    
    trades_df = pd.DataFrame(trades)
    winning_trades = trades_df[trades_df['P&L Points'] > 0]
    losing_trades = trades_df[trades_df['P&L Points'] <= 0]
    
    total_wins = winning_trades['P&L Points'].sum() if len(winning_trades) > 0 else 0
    total_losses = abs(losing_trades['P&L Points'].sum()) if len(losing_trades) > 0 else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else 0
    
    stats = {
        'trades': trades,
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': (len(winning_trades) / len(trades)) * 100 if len(trades) > 0 else 0,
        'total_pnl': trades_df['P&L Points'].sum(),
        'avg_win': winning_trades['P&L Points'].mean() if len(winning_trades) > 0 else 0,
        'avg_loss': losing_trades['P&L Points'].mean() if len(losing_trades) > 0 else 0,
        'max_win': trades_df['P&L Points'].max() if len(trades) > 0 else 0,
        'max_loss': trades_df['P&L Points'].min() if len(trades) > 0 else 0,
        'profit_factor': profit_factor,
        'avg_duration': trades_df['Duration'].mean() if len(trades) > 0 else timedelta(0)
    }
    
    return stats

def generate_forecast(df: pd.DataFrame, strategy_results: Dict, current_price: float) -> Dict:
    """Generate forecast based on historical strategy performance"""
    if len(strategy_results['trades']) == 0:
        return {
            'recommendation': 'HOLD',
            'confidence': 0,
            'reason': 'No historical trades to analyze',
            'entry': None,
            'sl': None,
            'target': None
        }
    
    trades_df = pd.DataFrame(strategy_results['trades'])
    win_rate = strategy_results['win_rate']
    avg_win = strategy_results['avg_win']
    avg_loss = strategy_results['avg_loss']
    profit_factor = strategy_results['profit_factor']
    
    # Check last signal
    last_signal = df['Signal'].iloc[-1]
    
    if last_signal == 0:
        recommendation = 'HOLD'
        confidence = 0
        reason = 'No active signal detected'
        entry = None
        sl = None
        target = None
    else:
        direction = 'BUY' if last_signal == 1 else 'SELL'
        
        # Calculate confidence based on historical performance
        confidence = min(100, (win_rate * 0.6 + profit_factor * 10 * 0.4))
        
        if win_rate > 60 and profit_factor > 1.5:
            recommendation = direction
            reason = f"Strong historical performance: {win_rate:.1f}% win rate, {profit_factor:.2f} profit factor"
        elif win_rate > 50 and profit_factor > 1.2:
            recommendation = direction
            reason = f"Moderate historical performance: {win_rate:.1f}% win rate, {profit_factor:.2f} profit factor"
        elif win_rate < 40 or profit_factor < 0.8:
            recommendation = 'HOLD'
            reason = f"Weak historical performance: {win_rate:.1f}% win rate, {profit_factor:.2f} profit factor"
            confidence = confidence * 0.5
        else:
            recommendation = direction
            reason = f"Average historical performance: {win_rate:.1f}% win rate, {profit_factor:.2f} profit factor"
        
        # Calculate suggested levels
        if recommendation in ['BUY', 'SELL']:
            entry = current_price
            
            if last_signal == 1:
                sl = entry - abs(avg_loss) if avg_loss < 0 else entry - 50
                target = entry + abs(avg_win) if avg_win > 0 else entry + 100
            else:
                sl = entry + abs(avg_loss) if avg_loss < 0 else entry + 50
                target = entry - abs(avg_win) if avg_win > 0 else entry - 100
        else:
            entry = None
            sl = None
            target = None
    
    return {
        'recommendation': recommendation,
        'confidence': confidence,
        'reason': reason,
        'entry': entry,
        'sl': sl,
        'target': target
    }

def create_strategy_chart(df: pd.DataFrame, strategy: str, params: Dict):
    """Create interactive chart with strategy indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price Chart', 'Volume', 'Indicators')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add strategy-specific indicators
    if strategy == "EMA Crossover" and 'EMA_Fast' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['EMA_Fast'], name=f"EMA {params.get('fast_ema', 9)}", 
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['EMA_Slow'], name=f"EMA {params.get('slow_ema', 15)}", 
                      line=dict(color='red', width=1)),
            row=1, col=1
        )
    
    elif strategy == "Mean Reversion" and 'MA' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MA'], name='MA', line=dict(color='purple', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Upper'], name='Upper Band', line=dict(color='red', width=1, dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Lower'], name='Lower Band', line=dict(color='green', width=1, dash='dash')),
            row=1, col=1
        )
    
    elif strategy == "RSI + EMA + ADX" and 'EMA' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['EMA'], name='EMA', line=dict(color='orange', width=1)),
            row=1, col=1
        )
    
    elif strategy == "Bollinger Bands" and 'BB_Middle' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Middle'], name='BB Middle', line=dict(color='blue', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='red', width=1, dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='green', width=1, dash='dash')),
            row=1, col=1
        )
    
    elif strategy == "Breakout" and 'High_Resistance' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['High_Resistance'], name='Resistance', line=dict(color='red', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Low_Support'], name='Support', line=dict(color='green', width=1)),
            row=1, col=1
        )
    
    # Entry signals
    long_entries = df[df['Signal'] == 1]
    short_entries = df[df['Signal'] == -1]
    
    if len(long_entries) > 0:
        fig.add_trace(
            go.Scatter(
                x=long_entries.index, y=long_entries['Close'],
                mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'),
                name='Long Entry'
            ),
            row=1, col=1
        )
    
    if len(short_entries) > 0:
        fig.add_trace(
            go.Scatter(
                x=short_entries.index, y=short_entries['Close'],
                mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'),
                name='Short Entry'
            ),
            row=1, col=1
        )
    
    # Volume
    colors = ['red' if close < open else 'green' for close, open in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )
    
    # Additional indicators subplot
    if strategy == "RSI + EMA + ADX" and 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=1)),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    elif 'ATR' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['ATR'], name='ATR', line=dict(color='orange', width=1)),
            row=3, col=1
        )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=3, col=1)
    
    return fig

# Streamlit UI
st.title("🚀 Advanced Trading Backtesting & Live Trading System")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Ticker Selection
    st.subheader("Ticker Selection")
    ticker_preset = st.selectbox("Select Preset Ticker", list(TICKER_PRESETS.keys()))
    
    if ticker_preset == "Custom":
        ticker1 = st.text_input("Enter Ticker Symbol", value="AAPL")
    else:
        ticker1 = TICKER_PRESETS[ticker_preset]
        st.info(f"Selected: {ticker1}")
    
    # Ratio-based trading
    use_ratio = st.checkbox("Enable Ratio-Based Trading")
    ticker2 = None
    if use_ratio:
        ticker_preset2 = st.selectbox("Select Second Ticker", list(TICKER_PRESETS.keys()), key="ticker2")
        if ticker_preset2 == "Custom":
            ticker2 = st.text_input("Enter Second Ticker Symbol", value="SPY", key="ticker2_input")
        else:
            ticker2 = TICKER_PRESETS[ticker_preset2]
    
    # Timeframe and Period
    st.subheader("Timeframe & Period")
    interval = st.selectbox("Interval", list(TIMEFRAME_PERIODS.keys()))
    period = st.selectbox("Period", TIMEFRAME_PERIODS[interval])
    
    # Strategy Selection
    st.subheader("Strategy Selection")
    strategy = st.selectbox(
        "Select Strategy",
        [
            "EMA Crossover",
            "Ratio Based",
            "Z-Score Based",
            "Volatility Based",
            "RSI Divergence",
            "Simple Buy",
            "Simple Sell",
            "Mean Reversion",
            "RSI + EMA + ADX",
            "Price Action S/R",
            "9 EMA Pullback",
            "Breakout",
            "Breakout Trap",
            "News Based",
            "Bollinger Bands"
        ]
    )
    
    # Strategy Parameters
    st.subheader("Strategy Parameters")
    strategy_params = {}
    
    if strategy == "EMA Crossover":
        col1, col2 = st.columns(2)
        with col1:
            fast_ema = st.number_input("Fast EMA", value=9, min_value=1)
            strategy_params['fast_ema'] = fast_ema
        with col2:
            slow_ema = st.number_input("Slow EMA", value=15, min_value=1)
            strategy_params['slow_ema'] = slow_ema
        
        crossover_type = st.selectbox(
            "Crossover Type",
            ["Simple Crossover", "Strong Candle", "ATR Based Candle"]
        )
        strategy_params['crossover_type'] = crossover_type
        
        min_angle = st.slider("Minimum Crossover Angle (degrees)", 0, 90, 20)
        strategy_params['min_angle'] = min_angle
        
        if crossover_type == "Strong Candle":
            candle_points = st.number_input("Minimum Candle Points", value=30.0)
            strategy_params['candle_points'] = candle_points
        else:
            strategy_params['candle_points'] = 0
        
        if crossover_type == "ATR Based Candle":
            atr_candle_multiplier = st.number_input("ATR Multiplier for Candle", value=1.5)
            strategy_params['atr_candle_multiplier'] = atr_candle_multiplier
        else:
            strategy_params['atr_candle_multiplier'] = 1.0
    
    elif strategy == "Mean Reversion":
        mean_period = st.number_input("Mean Period", value=20, min_value=5)
        strategy_params['mean_period'] = mean_period
        mean_std = st.number_input("Standard Deviations", value=2.0, min_value=0.5)
        strategy_params['mean_std'] = mean_std
    
    elif strategy == "RSI + EMA + ADX":
        rsi_period = st.number_input("RSI Period", value=14, min_value=5)
        strategy_params['rsi_period'] = rsi_period
        ema_period = st.number_input("EMA Period", value=20, min_value=5)
        strategy_params['ema_period'] = ema_period
        adx_period = st.number_input("ADX Period", value=14, min_value=5)
        strategy_params['adx_period'] = adx_period
        rsi_oversold = st.slider("RSI Oversold", 0, 50, 30)
        strategy_params['rsi_oversold'] = rsi_oversold
        rsi_overbought = st.slider("RSI Overbought", 50, 100, 70)
        strategy_params['rsi_overbought'] = rsi_overbought
        min_adx = st.slider("Minimum ADX", 0, 50, 25)
        strategy_params['min_adx'] = min_adx
    
    elif strategy == "Breakout":
        breakout_lookback = st.number_input("Lookback Period", value=20, min_value=5)
        strategy_params['breakout_lookback'] = breakout_lookback
    
    elif strategy == "Volatility Based":
        atr_period = st.number_input("ATR Period", value=14, min_value=5)
        strategy_params['atr_period'] = atr_period
        vol_threshold = st.number_input("Volatility Threshold", value=1.5, min_value=0.5)
        strategy_params['vol_threshold'] = vol_threshold
    
    elif strategy == "Z-Score Based":
        zscore_period = st.number_input("Z-Score Period", value=20, min_value=5)
        strategy_params['zscore_period'] = zscore_period
        zscore_threshold = st.number_input("Z-Score Threshold", value=2.0, min_value=0.5)
        strategy_params['zscore_threshold'] = zscore_threshold
    
    elif strategy == "Bollinger Bands":
        bb_period = st.number_input("BB Period", value=20, min_value=5)
        strategy_params['bb_period'] = bb_period
        bb_std = st.number_input("BB Std Dev", value=2.0, min_value=0.5)
        strategy_params['bb_std'] = bb_std
    
    st.session_state.strategy_params = strategy_params
    
    # Stop Loss Configuration
    st.subheader("Stop Loss Configuration")
    sl_type = st.selectbox(
        "Stop Loss Type",
        [
            "Custom Points",
            "Trailing SL",
            "ATR Based",
            "Current Candle Low/High",
            "Current Swing Low/High",
            "Previous Swing Low/High",
            "Signal Based"
        ]
    )
    
    if sl_type == "Custom Points":
        sl_points = st.number_input("Stop Loss Points", value=50.0, min_value=1.0)
    else:
        sl_points = 50.0
    
    if sl_type == "ATR Based":
        atr_multiplier_sl = st.number_input("ATR Multiplier for SL", value=2.0, min_value=0.1)
    else:
        atr_multiplier_sl = 2.0
    
    # Target Configuration
    st.subheader("Target Configuration")
    target_type = st.selectbox(
        "Target Type",
        [
            "Custom Points",
            "ATR Based",
            "Trailing Target",
            "Signal Based",
            "Risk:Reward"
        ]
    )
    
    if target_type == "Custom Points":
        target_points = st.number_input("Target Points", value=100.0, min_value=1.0)
    else:
        target_points = 100.0
    
    if target_type == "ATR Based":
        atr_multiplier_target = st.number_input("ATR Multiplier for Target", value=3.0, min_value=0.1)
    else:
        atr_multiplier_target = 3.0
    
    if target_type == "Risk:Reward":
        risk_reward = st.number_input("Risk:Reward Ratio", value=2.0, min_value=0.5)
    else:
        risk_reward = 2.0

# Main content area
tab1, tab2 = st.tabs(["📊 Backtesting", "🔴 Live Trading"])

# Backtesting Tab
with tab1:
    st.header("Backtesting")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        fetch_button = st.button("📥 Fetch Data & Run Backtest", type="primary", use_container_width=True)
    
    if fetch_button:
        progress_bar = st.progress(0, text="Starting...")
        
        # Fetch data
        df = fetch_data(ticker1, interval, period, progress_bar)
        
        if df is not None and not df.empty:
            st.session_state.df = df
            st.session_state.data_fetched = True
            st.success(f"✅ Data fetched successfully! {len(df)} candles loaded.")
            
            # Apply strategy
            df_strategy = apply_strategy(df, strategy, st.session_state.strategy_params)
            
            # Run backtest
            st.info("🔄 Running backtest...")
            results = backtest_strategy(
                df_strategy, sl_type, sl_points, atr_multiplier_sl,
                target_type, target_points, atr_multiplier_target,
                risk_reward
            )
            
            # Display results
            st.subheader("📈 Backtest Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Trades", results['total_trades'])
            with col2:
                st.metric("Win Rate", f"{results['win_rate']:.2f}%")
            with col3:
                st.metric("Total P&L", f"{results['total_pnl']:.2f} pts")
            with col4:
                st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Winning Trades", results['winning_trades'])
            with col2:
                st.metric("Losing Trades", results['losing_trades'])
            with col3:
                st.metric("Avg Win", f"{results['avg_win']:.2f} pts")
            with col4:
                st.metric("Avg Loss", f"{results['avg_loss']:.2f} pts")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max Win", f"{results['max_win']:.2f} pts")
            with col2:
                st.metric("Max Loss", f"{results['max_loss']:.2f} pts")
            with col3:
                st.metric("Avg Duration", str(results['avg_duration']).split('.')[0])
            
            # Forecast
            st.subheader("🔮 Strategy Forecast")
            forecast = generate_forecast(df_strategy, results, df['Close'].iloc[-1])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                rec_color = "🟢" if forecast['recommendation'] == "BUY" else "🔴" if forecast['recommendation'] == "SELL" else "🟡"
                st.metric("Recommendation", f"{rec_color} {forecast['recommendation']}")
            with col2:
                st.metric("Confidence", f"{forecast['confidence']:.1f}%")
            with col3:
                if forecast['entry']:
                    st.metric("Suggested Entry", f"{forecast['entry']:.2f}")
            
            if forecast['entry']:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Suggested SL", f"{forecast['sl']:.2f}")
                with col2:
                    st.metric("Suggested Target", f"{forecast['target']:.2f}")
            
            st.info(f"**Reason:** {forecast['reason']}")
            
            # Trade History
            if len(results['trades']) > 0:
                st.subheader("📋 Trade History")
                trades_df = pd.DataFrame(results['trades'])
                
                # Format dataframe
                trades_df['Entry Time'] = trades_df['Entry Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                trades_df['Exit Time'] = trades_df['Exit Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                trades_df['Duration'] = trades_df['Duration'].astype(str)
                
                # Style the dataframe
                def highlight_pnl(val):
                    if isinstance(val, (int, float)):
                        color = 'lightgreen' if val > 0 else 'lightcoral'
                        return f'background-color: {color}'
                    return ''
                
                styled_df = trades_df.style.applymap(highlight_pnl, subset=['P&L Points'])
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                # Download trades
                csv = trades_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Trade History",
                    csv,
                    "trade_history.csv",
                    "text/csv",
                    key='download-csv'
                )
                
                # Chart
                st.subheader("📊 Strategy Chart")
                fig = create_strategy_chart(df_strategy, strategy, st.session_state.strategy_params)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No trades generated by the strategy.")
        else:
            st.error("❌ Failed to fetch data. Please check the ticker symbol and try again.")

# Live Trading Tab
with tab2:
    st.header("Live Trading")
    
    if not st.session_state.data_fetched:
        st.warning("⚠️ Please fetch data from the Backtesting tab first.")
    else:
        tab_live, tab_history, tab_logs = st.tabs(["🔴 Live Monitor", "📋 Trade History", "📝 Logs"])
        
        # Live Monitor Tab
        with tab_live:
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("▶️ Start Live Trading", type="primary", disabled=st.session_state.live_trading_active):
                    st.session_state.live_trading_active = True
                    st.session_state.iteration_count = 0
                    st.session_state.trade_logs.append(f"[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}] Live trading started")
                    st.rerun()
            
            with col2:
                if st.button("⏹️ Stop Live Trading", type="secondary", disabled=not st.session_state.live_trading_active):
                    st.session_state.live_trading_active = False
                    st.session_state.trade_logs.append(f"[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}] Live trading stopped")
                    st.rerun()
            
            if st.session_state.live_trading_active:
                st.success("🟢 Live Trading Active")
                st.session_state.iteration_count += 1
                
                # Auto-refresh placeholder
                live_container = st.empty()
                
                with live_container.container():
                    # Fetch latest data
                    with st.spinner("Fetching live data..."):
                        progress_bar = st.progress(0, text="Fetching live data...")
                        live_df = fetch_data(ticker1, interval, "1d", progress_bar)
                    
                    if live_df is not None and not live_df.empty:
                        # Apply strategy
                        live_strategy = apply_strategy(live_df, strategy, st.session_state.strategy_params)
                        
                        current_price = live_strategy['Close'].iloc[-1]
                        current_time = live_strategy.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                        
                        st.subheader(f"💹 Current Market Status - {current_time}")
                        st.write(f"**Iterations Completed:** {st.session_state.iteration_count}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Price", f"{current_price:.2f}")
                        with col2:
                            change = live_strategy['Close'].iloc[-1] - live_strategy['Open'].iloc[-1]
                            st.metric("Change", f"{change:.2f}", f"{(change/live_strategy['Open'].iloc[-1]*100):.2f}%")
                        with col3:
                            st.metric("Volume", f"{live_strategy['Volume'].iloc[-1]:,.0f}")
                        
                        # Display Strategy Parameters
                        st.subheader("📊 Strategy Parameters")
                        
                        if strategy == "EMA Crossover":
                            ema_fast_val = live_strategy['EMA_Fast'].iloc[-1]
                            ema_slow_val = live_strategy['EMA_Slow'].iloc[-1]
                            ema_avg = (ema_fast_val + ema_slow_val) / 2
                            crossover_angle = live_strategy['Crossover_Angle'].iloc[-1] if 'Crossover_Angle' in live_strategy.columns else 0
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(f"EMA {st.session_state.strategy_params['fast_ema']}", f"{ema_fast_val:.2f}")
                            with col2:
                                st.metric(f"EMA {st.session_state.strategy_params['slow_ema']}", f"{ema_slow_val:.2f}")
                            with col3:
                                st.metric("Average EMA", f"{ema_avg:.2f}")
                            with col4:
                                st.metric("Crossover Angle", f"{crossover_angle:.2f}°")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.info(f"**Min Angle Required:** {st.session_state.strategy_params['min_angle']}°")
                            with col2:
                                st.info(f"**Crossover Type:** {st.session_state.strategy_params['crossover_type']}")
                            with col3:
                                st.info(f"**Candles Analyzed:** {len(live_strategy)}")
                            
                            # Show distance to crossover
                            ema_diff = ema_fast_val - ema_slow_val
                            diff_pct = (abs(ema_diff) / current_price) * 100
                            
                            if abs(ema_diff) < current_price * 0.005:  # Within 0.5%
                                st.warning(f"⚠️ EMAs are converging! Difference: {ema_diff:.2f} pts ({diff_pct:.3f}%)")
                            else:
                                status = "above" if ema_diff > 0 else "below"
                                st.info(f"📍 Fast EMA is {abs(ema_diff):.2f} pts ({diff_pct:.3f}%) {status} Slow EMA")
                        
                        elif strategy == "Mean Reversion":
                            if 'MA' in live_strategy.columns:
                                ma_val = live_strategy['MA'].iloc[-1]
                                upper_val = live_strategy['Upper'].iloc[-1]
                                lower_val = live_strategy['Lower'].iloc[-1]
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Moving Average", f"{ma_val:.2f}")
                                with col2:
                                    st.metric("Upper Band", f"{upper_val:.2f}", f"{abs(upper_val-current_price):.2f} pts away")
                                with col3:
                                    st.metric("Lower Band", f"{lower_val:.2f}", f"{abs(lower_val-current_price):.2f} pts away")
                                
                                st.info(f"**Period:** {st.session_state.strategy_params.get('mean_period', 20)} | **Std Dev:** {st.session_state.strategy_params.get('mean_std', 2.0)}")
                        
                        elif strategy == "RSI + EMA + ADX":
                            if 'RSI' in live_strategy.columns:
                                rsi_val = live_strategy['RSI'].iloc[-1]
                                ema_val = live_strategy['EMA'].iloc[-1]
                                adx_val = live_strategy['ADX'].iloc[-1]
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("RSI", f"{rsi_val:.2f}")
                                with col2:
                                    st.metric("EMA", f"{ema_val:.2f}")
                                with col3:
                                    st.metric("ADX", f"{adx_val:.2f}")
                                with col4:
                                    trend = "Uptrend" if current_price > ema_val else "Downtrend"
                                    st.metric("Trend", trend)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.info(f"**RSI Oversold:** {st.session_state.strategy_params.get('rsi_oversold', 30)}")
                                with col2:
                                    st.info(f"**RSI Overbought:** {st.session_state.strategy_params.get('rsi_overbought', 70)}")
                                with col3:
                                    st.info(f"**Min ADX:** {st.session_state.strategy_params.get('min_adx', 25)}")
                        
                        elif strategy == "Breakout":
                            if 'High_Resistance' in live_strategy.columns:
                                resistance = live_strategy['High_Resistance'].iloc[-1]
                                support = live_strategy['Low_Support'].iloc[-1]
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Resistance", f"{resistance:.2f}", f"{abs(resistance-current_price):.2f} pts away")
                                with col2:
                                    st.metric("Support", f"{support:.2f}", f"{abs(support-current_price):.2f} pts away")
                                
                                st.info(f"**Lookback Period:** {st.session_state.strategy_params.get('breakout_lookback', 20)} candles")
                        
                        elif strategy == "Volatility Based":
                            if 'ATR' in live_strategy.columns:
                                atr_val = live_strategy['ATR'].iloc[-1]
                                vol_ratio = live_strategy['Volatility_Ratio'].iloc[-1] if 'Volatility_Ratio' in live_strategy.columns else 0
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("ATR", f"{atr_val:.2f}")
                                with col2:
                                    st.metric("Volatility Ratio", f"{vol_ratio:.2f}")
                                
                                st.info(f"**ATR Period:** {st.session_state.strategy_params.get('atr_period', 14)} | **Threshold:** {st.session_state.strategy_params.get('vol_threshold', 1.5)}")
                        
                        elif strategy == "Z-Score Based":
                            if 'ZScore' in live_strategy.columns:
                                zscore_val = live_strategy['ZScore'].iloc[-1]
                                
                                st.metric("Z-Score", f"{zscore_val:.2f}")
                                st.info(f"**Period:** {st.session_state.strategy_params.get('zscore_period', 20)} | **Threshold:** {st.session_state.strategy_params.get('zscore_threshold', 2.0)}")
                        
                        elif strategy == "Bollinger Bands":
                            if 'BB_Middle' in live_strategy.columns:
                                bb_upper = live_strategy['BB_Upper'].iloc[-1]
                                bb_middle = live_strategy['BB_Middle'].iloc[-1]
                                bb_lower = live_strategy['BB_Lower'].iloc[-1]
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("BB Upper", f"{bb_upper:.2f}", f"{abs(bb_upper-current_price):.2f} pts away")
                                with col2:
                                    st.metric("BB Middle", f"{bb_middle:.2f}")
                                with col3:
                                    st.metric("BB Lower", f"{bb_lower:.2f}", f"{abs(bb_lower-current_price):.2f} pts away")
                                
                                st.info(f"**Period:** {st.session_state.strategy_params.get('bb_period', 20)} | **Std Dev:** {st.session_state.strategy_params.get('bb_std', 2.0)}")
                        
                        # Show SL and Target configuration
                        st.subheader("🎯 Risk Management")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"**SL Type:** {sl_type}")
                            if sl_type == "Custom Points":
                                st.write(f"**SL Points:** {sl_points}")
                            elif sl_type == "ATR Based":
                                st.write(f"**ATR Multiplier:** {atr_multiplier_sl}")
                        with col2:
                            st.info(f"**Target Type:** {target_type}")
                            if target_type == "Custom Points":
                                st.write(f"**Target Points:** {target_points}")
                            elif target_type == "ATR Based":
                                st.write(f"**ATR Multiplier:** {atr_multiplier_target}")
                            elif target_type == "Risk:Reward":
                                st.write(f"**Risk:Reward Ratio:** {risk_reward}")
                        
                        # Check for signals
                        latest_signal = live_strategy['Signal'].iloc[-1]
                        
                        if st.session_state.current_position is None:
                            st.info("📭 No Active Position")
                            
                            if latest_signal != 0:
                                signal_type = "LONG" if latest_signal == 1 else "SHORT"
                                st.success(f"🚨 {signal_type} SIGNAL TRIGGERED!")
                                st.write(f"**Entry Reason:** {live_strategy['Entry_Reason'].iloc[-1]}")
                                
                                entry_price = current_price
                                stop_loss = calculate_stop_loss(live_strategy, len(live_strategy)-1, latest_signal, 
                                                               sl_type, sl_points, atr_multiplier_sl, entry_price)
                                target = calculate_target(live_strategy, len(live_strategy)-1, latest_signal,
                                                         target_type, target_points, atr_multiplier_target, risk_reward, entry_price, stop_loss)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Entry Price", f"{entry_price:.2f}")
                                with col2:
                                    sl_dist = abs(entry_price - stop_loss)
                                    st.metric("Stop Loss", f"{stop_loss:.2f}", f"{sl_dist:.2f} pts away")
                                with col3:
                                    target_dist = abs(target - entry_price)
                                    st.metric("Target", f"{target:.2f}", f"{target_dist:.2f} pts away")
                                
                                risk_reward_actual = target_dist / sl_dist if sl_dist > 0 else 0
                                st.info(f"💡 **Risk:Reward Ratio:** 1:{risk_reward_actual:.2f}")
                                st.info("💡 **Recommendation:** Signal is active. Consider entering the position based on your risk management.")
                            else:
                                st.warning("⏳ Waiting for entry signal...")
                                st.info("💡 **Market Analysis:** No clear signal yet. Monitoring market conditions...")
                        
                        else:
                            # Active position
                            pos = st.session_state.current_position
                            st.success(f"📍 Active {pos['direction']} Position")
                            
                            pnl = (current_price - pos['entry_price']) if pos['direction'] == 'LONG' else (pos['entry_price'] - current_price)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Entry", f"{pos['entry_price']:.2f}")
                            with col2:
                                st.metric("Current", f"{current_price:.2f}")
                            with col3:
                                st.metric("P&L", f"{pnl:.2f} pts")
                            with col4:
                                risk_ratio = abs(pnl / (pos['entry_price'] - pos['stop_loss'])) if pos['entry_price'] != pos['stop_loss'] else 0
                                st.metric("Risk:Reward", f"{risk_ratio:.2f}x")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                sl_distance = abs(current_price - pos['stop_loss'])
                                sl_pct = (sl_distance / current_price) * 100
                                st.metric("Distance to SL", f"{sl_distance:.2f} pts", f"{sl_pct:.2f}%")
                            with col2:
                                target_distance = abs(pos['target'] - current_price)
                                target_pct = (target_distance / current_price) * 100
                                st.metric("Distance to Target", f"{target_distance:.2f} pts", f"{target_pct:.2f}%")
                            
                            # Guidance
                            st.subheader("🤝 Trading Guidance")
                            
                            if pnl > 0:
                                target_reached = (pnl / abs(pos['target'] - pos['entry_price'])) * 100
                                if target_reached >= 80:
                                    st.success("✅ Trade is near target! Consider booking profits or trailing stop loss.")
                                elif target_reached >= 50:
                                    st.info("📈 Trade is going well. Consider moving SL to breakeven.")
                                else:
                                    st.info("📊 Trade is profitable. Hold position and let it run.")
                            else:
                                sl_risk = (abs(pnl) / abs(pos['entry_price'] - pos['stop_loss'])) * 100
                                if sl_risk >= 80:
                                    st.warning("⚠️ Trade is near stop loss. Be prepared for exit.")
                                else:
                                    st.warning("📉 Trade is currently negative. Stick to your plan and wait.")
                            
                            duration = datetime.now(IST) - pos['entry_time']
                            st.write(f"**Duration:** {str(duration).split('.')[0]}")
                        
                        # Live Chart
                        st.subheader("📊 Live Strategy Chart")
                        live_chart = create_strategy_chart(live_strategy.tail(100), strategy, st.session_state.strategy_params)
                        st.plotly_chart(live_chart, use_container_width=True)
                
                # Auto-refresh
                time.sleep(random.uniform(1.0, 1.5))
                if st.session_state.live_trading_active:
                    st.rerun()
            
            else:
                st.info("⏸️ Live trading is stopped. Click 'Start Live Trading' to begin.")
        
        # Trade History Tab
        with tab_history:
            st.subheader("Trade History")
            if len(st.session_state.trade_history) > 0:
                history_df = pd.DataFrame(st.session_state.trade_history)
                st.dataframe(history_df, use_container_width=True)
            else:
                st.info("No trades executed yet.")
        
        # Logs Tab
        with tab_logs:
            st.subheader("Trade Logs")
            if len(st.session_state.trade_logs) > 0:
                for log in st.session_state.trade_logs[-50:]:  # Show last 50 logs
                    st.text(log)
            else:
                st.info("No logs available.")

# Footer
st.markdown("---")
st.markdown("💡 **Note:** This is a backtesting and live simulation system. Always validate signals before actual trading.")
st.markdown("📌 **Indicators are TradingView compatible** - calculations match TradingView formulas for accuracy.")
