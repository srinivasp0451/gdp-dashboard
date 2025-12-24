import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
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
    """Fetch data from yfinance with rate limiting"""
    try:
        progress_bar.progress(10, text="Initiating data fetch...")
        time.sleep(1.5)  # Rate limiting
        
        progress_bar.progress(30, text=f"Fetching {ticker} data...")
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        time.sleep(1.5)  # Rate limiting
        
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
    """Calculate EMA"""
    return df['Close'].ewm(span=period, adjust=False).mean()

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    return atr

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ADX"""
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    atr = calculate_atr(df, period)
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx

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
        # Check if local minimum (support)
        if df['Low'].iloc[i] == df['Low'].iloc[i-window:i+window].min():
            supports.append(df['Low'].iloc[i])
        # Check if local maximum (resistance)
        if df['High'].iloc[i] == df['High'].iloc[i-window:i+window].max():
            resistances.append(df['High'].iloc[i])
    
    # Remove duplicates and sort
    supports = sorted(list(set(supports)))
    resistances = sorted(list(set(resistances)))
    
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
    
    for i in range(slow_period + 5, len(df)):
        # Check crossover
        cross_up = (df['EMA_Fast'].iloc[i] > df['EMA_Slow'].iloc[i] and 
                    df['EMA_Fast'].iloc[i-1] <= df['EMA_Slow'].iloc[i-1])
        cross_down = (df['EMA_Fast'].iloc[i] < df['EMA_Slow'].iloc[i] and 
                      df['EMA_Fast'].iloc[i-1] >= df['EMA_Slow'].iloc[i-1])
        
        # Calculate angle
        angle = calculate_crossover_angle(df['EMA_Fast'], df['EMA_Slow'], i)
        
        # Check candle conditions
        candle_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
        
        valid_candle = True
        if crossover_type == "Strong Candle":
            valid_candle = candle_size >= candle_points
        elif crossover_type == "ATR Based Candle":
            valid_candle = candle_size >= (df['ATR'].iloc[i] * atr_multiplier)
        
        # Generate signals
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
    df['ADX'] = calculate_adx(df, adx_period)
    
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

def calculate_stop_loss(df: pd.DataFrame, idx: int, signal: int, sl_type: str, 
                       sl_points: float, atr_multiplier: float, trailing_percent: float) -> float:
    """Calculate stop loss based on type"""
    entry_price = df['Close'].iloc[idx]
    
    if sl_type == "Custom Points":
        if signal == 1:  # Long
            return entry_price - sl_points
        else:  # Short
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
    
    else:  # Default to custom points
        if signal == 1:
            return entry_price - sl_points
        else:
            return entry_price + sl_points

def calculate_target(df: pd.DataFrame, idx: int, signal: int, target_type: str, 
                    target_points: float, atr_multiplier: float, risk_reward: float) -> float:
    """Calculate target based on type"""
    entry_price = df['Close'].iloc[idx]
    
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
        # Calculate based on SL distance
        sl_distance = abs(entry_price - calculate_stop_loss(df, idx, signal, "Custom Points", target_points/risk_reward, 0, 0))
        if signal == 1:
            return entry_price + (sl_distance * risk_reward)
        else:
            return entry_price - (sl_distance * risk_reward)
    
    else:  # Default to custom points
        if signal == 1:
            return entry_price + target_points
        else:
            return entry_price - target_points

def backtest_strategy(df: pd.DataFrame, sl_type: str, sl_points: float, atr_multiplier_sl: float,
                     target_type: str, target_points: float, atr_multiplier_target: float,
                     risk_reward: float, initial_capital: float = 100000) -> Dict:
    """Backtest the strategy"""
    trades = []
    capital = initial_capital
    position = None
    
    for i in range(len(df)):
        if position is None:
            # Entry logic
            if df['Signal'].iloc[i] != 0:
                entry_price = df['Close'].iloc[i]
                signal = df['Signal'].iloc[i]
                
                stop_loss = calculate_stop_loss(df, i, signal, sl_type, sl_points, atr_multiplier_sl, 0)
                target = calculate_target(df, i, signal, target_type, target_points, atr_multiplier_target, risk_reward)
                
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
            # Exit logic
            current_price = df['Close'].iloc[i]
            exit_triggered = False
            exit_reason = ""
            exit_price = current_price
            
            # Update trailing stop
            if position['signal'] == 1:  # Long
                if current_price > position.get('highest_price', position['entry_price']):
                    position['highest_price'] = current_price
                    if sl_type == "Trailing SL":
                        move_up = current_price - position['entry_price']
                        position['trailing_sl'] = position['stop_loss'] + move_up * 0.5
                
                # Check SL
                if df['Low'].iloc[i] <= position['trailing_sl']:
                    exit_triggered = True
                    exit_price = position['trailing_sl']
                    exit_reason = f"Stop Loss Hit (SL: {position['trailing_sl']:.2f})"
                
                # Check Target
                elif df['High'].iloc[i] >= position['target']:
                    exit_triggered = True
                    exit_price = position['target']
                    exit_reason = f"Target Hit (Target: {position['target']:.2f})"
                
                # Check signal reversal
                elif df['Signal'].iloc[i] == -1 and sl_type == "Signal Based":
                    exit_triggered = True
                    exit_reason = "Signal Reversal"
            
            else:  # Short
                if current_price < position.get('lowest_price', position['entry_price']):
                    position['lowest_price'] = current_price
                    if sl_type == "Trailing SL":
                        move_down = position['entry_price'] - current_price
                        position['trailing_sl'] = position['stop_loss'] - move_down * 0.5
                
                # Check SL
                if df['High'].iloc[i] >= position['trailing_sl']:
                    exit_triggered = True
                    exit_price = position['trailing_sl']
                    exit_reason = f"Stop Loss Hit (SL: {position['trailing_sl']:.2f})"
                
                # Check Target
                elif df['Low'].iloc[i] <= position['target']:
                    exit_triggered = True
                    exit_price = position['target']
                    exit_reason = f"Target Hit (Target: {position['target']:.2f})"
                
                # Check signal reversal
                elif df['Signal'].iloc[i] == 1 and sl_type == "Signal Based":
                    exit_triggered = True
                    exit_reason = "Signal Reversal"
            
            if exit_triggered:
                # Calculate P&L
                if position['signal'] == 1:
                    pnl_points = exit_price - position['entry_price']
                else:
                    pnl_points = position['entry_price'] - exit_price
                
                pnl_percent = (pnl_points / position['entry_price']) * 100
                
                trade = {
                    'Entry Time': position['entry_time'],
                    'Exit Time': df.index[i],
                    'Direction': 'Long' if position['signal'] == 1 else 'Short',
                    'Entry Price': position['entry_price'],
                    'Exit Price': exit_price,
                    'Stop Loss': position['stop_loss'],
                    'Target': position['target'],
                    'P&L Points': pnl_points,
                    'P&L %': pnl_percent,
                    'Entry Reason': position['entry_reason'],
                    'Exit Reason': exit_reason,
                    'Duration': df.index[i] - position['entry_time']
                }
                
                trades.append(trade)
                capital += (capital * pnl_percent / 100)
                position = None
    
    # Calculate statistics
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
            'final_capital': initial_capital,
            'total_return': 0
        }
    
    trades_df = pd.DataFrame(trades)
    winning_trades = trades_df[trades_df['P&L Points'] > 0]
    losing_trades = trades_df[trades_df['P&L Points'] <= 0]
    
    stats = {
        'trades': trades,
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': (len(winning_trades) / len(trades)) * 100 if len(trades) > 0 else 0,
        'total_pnl': trades_df['P&L Points'].sum(),
        'avg_win': winning_trades['P&L Points'].mean() if len(winning_trades) > 0 else 0,
        'avg_loss': losing_trades['P&L Points'].mean() if len(losing_trades) > 0 else 0,
        'max_win': trades_df['P&L Points'].max(),
        'max_loss': trades_df['P&L Points'].min(),
        'final_capital': capital,
        'total_return': ((capital - initial_capital) / initial_capital) * 100
    }
    
    return stats

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
    
    if strategy == "EMA Crossover":
        col1, col2 = st.columns(2)
        with col1:
            fast_ema = st.number_input("Fast EMA", value=9, min_value=1)
        with col2:
            slow_ema = st.number_input("Slow EMA", value=15, min_value=1)
        
        crossover_type = st.selectbox(
            "Crossover Type",
            ["Simple Crossover", "Strong Candle", "ATR Based Candle"]
        )
        
        min_angle = st.slider("Minimum Crossover Angle (degrees)", 0, 90, 20)
        
        if crossover_type == "Strong Candle":
            candle_points = st.number_input("Minimum Candle Points", value=30.0)
        else:
            candle_points = 0
        
        if crossover_type == "ATR Based Candle":
            atr_candle_multiplier = st.number_input("ATR Multiplier for Candle", value=1.5)
        else:
            atr_candle_multiplier = 1.0
    
    elif strategy == "Mean Reversion":
        mean_period = st.number_input("Mean Period", value=20, min_value=5)
        mean_std = st.number_input("Standard Deviations", value=2.0, min_value=0.5)
    
    elif strategy == "RSI + EMA + ADX":
        rsi_period = st.number_input("RSI Period", value=14, min_value=5)
        ema_period = st.number_input("EMA Period", value=20, min_value=5)
        adx_period = st.number_input("ADX Period", value=14, min_value=5)
        rsi_oversold = st.slider("RSI Oversold", 0, 50, 30)
        rsi_overbought = st.slider("RSI Overbought", 50, 100, 70)
        min_adx = st.slider("Minimum ADX", 0, 50, 25)
    
    elif strategy == "Breakout":
        breakout_lookback = st.number_input("Lookback Period", value=20, min_value=5)
    
    elif strategy == "Volatility Based":
        atr_period = st.number_input("ATR Period", value=14, min_value=5)
        vol_threshold = st.number_input("Volatility Threshold", value=1.5, min_value=0.5)
    
    elif strategy == "Z-Score Based":
        zscore_period = st.number_input("Z-Score Period", value=20, min_value=5)
        zscore_threshold = st.number_input("Z-Score Threshold", value=2.0, min_value=0.5)
    
    elif strategy == "Bollinger Bands":
        bb_period = st.number_input("BB Period", value=20, min_value=5)
        bb_std = st.number_input("BB Std Dev", value=2.0, min_value=0.5)
    
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
    
    # Initial Capital
    initial_capital = st.number_input("Initial Capital", value=100000, min_value=1000)

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
            if strategy == "EMA Crossover":
                df_strategy = ema_crossover_strategy(
                    df, fast_ema, slow_ema, crossover_type, 
                    min_angle, candle_points, atr_candle_multiplier
                )
            elif strategy == "Mean Reversion":
                df_strategy = mean_reversion_strategy(df, mean_period, mean_std)
            elif strategy == "RSI + EMA + ADX":
                df_strategy = rsi_ema_adx_strategy(
                    df, rsi_period, ema_period, adx_period,
                    rsi_overbought, rsi_oversold, min_adx
                )
            elif strategy == "Breakout":
                df_strategy = breakout_strategy(df, breakout_lookback)
            elif strategy == "Volatility Based":
                df_strategy = volatility_strategy(df, atr_period, vol_threshold)
            elif strategy == "Z-Score Based":
                df_strategy = zscore_strategy(df, zscore_period, zscore_threshold)
            elif strategy == "Bollinger Bands":
                df_strategy = bollinger_strategy(df, bb_period, bb_std)
            else:
                df_strategy = ema_crossover_strategy(df, 9, 15, "Simple Crossover", 20, 0, 1.0)
            
            # Run backtest
            st.info("🔄 Running backtest...")
            results = backtest_strategy(
                df_strategy, sl_type, sl_points, atr_multiplier_sl,
                target_type, target_points, atr_multiplier_target,
                risk_reward, initial_capital
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
                st.metric("Total Return", f"{results['total_return']:.2f}%")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Winning Trades", results['winning_trades'])
            with col2:
                st.metric("Losing Trades", results['losing_trades'])
            with col3:
                st.metric("Avg Win", f"{results['avg_win']:.2f} pts")
            with col4:
                st.metric("Avg Loss", f"{results['avg_loss']:.2f} pts")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Max Win", f"{results['max_win']:.2f} pts")
            with col2:
                st.metric("Max Loss", f"{results['max_loss']:.2f} pts")
            
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
                
                styled_df = trades_df.style.applymap(highlight_pnl, subset=['P&L Points', 'P&L %'])
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
                st.subheader("📊 Price Chart with Trades")
                fig = go.Figure()
                
                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=df_strategy.index,
                    open=df_strategy['Open'],
                    high=df_strategy['High'],
                    low=df_strategy['Low'],
                    close=df_strategy['Close'],
                    name='Price'
                ))
                
                # Entry signals
                long_entries = df_strategy[df_strategy['Signal'] == 1]
                short_entries = df_strategy[df_strategy['Signal'] == -1]
                
                fig.add_trace(go.Scatter(
                    x=long_entries.index,
                    y=long_entries['Close'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='green'),
                    name='Long Entry'
                ))
                
                fig.add_trace(go.Scatter(
                    x=short_entries.index,
                    y=short_entries['Close'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='red'),
                    name='Short Entry'
                ))
                
                fig.update_layout(
                    title=f"{ticker1} - {strategy} Strategy",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    height=600,
                    xaxis_rangeslider_visible=False
                )
                
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
                    st.session_state.trade_logs.append(f"[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}] Live trading started")
                    st.rerun()
            
            with col2:
                if st.button("⏹️ Stop Live Trading", type="secondary", disabled=not st.session_state.live_trading_active):
                    st.session_state.live_trading_active = False
                    st.session_state.trade_logs.append(f"[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}] Live trading stopped")
                    st.rerun()
            
            if st.session_state.live_trading_active:
                st.success("🟢 Live Trading Active")
                
                # Auto-refresh placeholder
                live_container = st.empty()
                
                with live_container.container():
                    # Fetch latest data
                    with st.spinner("Fetching live data..."):
                        progress_bar = st.progress(0, text="Fetching live data...")
                        live_df = fetch_data(ticker1, interval, "1d", progress_bar)
                    
                    if live_df is not None and not live_df.empty:
                        current_price = live_df['Close'].iloc[-1]
                        current_time = live_df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                        
                        st.subheader(f"💹 Current Market Status - {current_time}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Price", f"{current_price:.2f}")
                        with col2:
                            change = live_df['Close'].iloc[-1] - live_df['Open'].iloc[-1]
                            st.metric("Change", f"{change:.2f}", f"{(change/live_df['Open'].iloc[-1]*100):.2f}%")
                        with col3:
                            st.metric("Volume", f"{live_df['Volume'].iloc[-1]:,.0f}")
                        
                        # Apply strategy on live data
                        if strategy == "EMA Crossover":
                            live_strategy = ema_crossover_strategy(
                                live_df, fast_ema, slow_ema, crossover_type,
                                min_angle, candle_points, atr_candle_multiplier
                            )
                        elif strategy == "Mean Reversion":
                            live_strategy = mean_reversion_strategy(live_df, mean_period, mean_std)
                        else:
                            live_strategy = ema_crossover_strategy(live_df, 9, 15, "Simple Crossover", 20, 0, 1.0)
                        
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
                                                               sl_type, sl_points, atr_multiplier_sl, 0)
                                target = calculate_target(live_strategy, len(live_strategy)-1, latest_signal,
                                                         target_type, target_points, atr_multiplier_target, risk_reward)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Entry Price", f"{entry_price:.2f}")
                                with col2:
                                    st.metric("Stop Loss", f"{stop_loss:.2f}", f"{abs(entry_price-stop_loss):.2f} pts away")
                                with col3:
                                    st.metric("Target", f"{target:.2f}", f"{abs(target-entry_price):.2f} pts away")
                                
                                st.info("💡 **Recommendation:** Signal is active. Consider entering the position based on your risk management.")
                            else:
                                st.warning("⏳ Waiting for entry signal...")
                                
                                # Show how far from signal
                                if 'EMA_Fast' in live_strategy.columns and 'EMA_Slow' in live_strategy.columns:
                                    ema_diff = live_strategy['EMA_Fast'].iloc[-1] - live_strategy['EMA_Slow'].iloc[-1]
                                    st.write(f"**EMA Status:** Fast EMA is {abs(ema_diff):.2f} points {'above' if ema_diff > 0 else 'below'} Slow EMA")
                                
                                st.info("💡 **Market Analysis:** No clear signal yet. Market is in consolidation or waiting for setup.")
                        
                        else:
                            # Active position
                            pos = st.session_state.current_position
                            st.success(f"📍 Active {pos['direction']} Position")
                            
                            pnl = (current_price - pos['entry_price']) if pos['direction'] == 'LONG' else (pos['entry_price'] - current_price)
                            pnl_pct = (pnl / pos['entry_price']) * 100
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Entry", f"{pos['entry_price']:.2f}")
                            with col2:
                                st.metric("Current", f"{current_price:.2f}")
                            with col3:
                                st.metric("P&L", f"{pnl:.2f}", f"{pnl_pct:.2f}%")
                            with col4:
                                risk_ratio = abs(pnl / (pos['entry_price'] - pos['stop_loss']))
                                st.metric("Risk:Reward", f"{risk_ratio:.2f}x")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                sl_distance = abs(current_price - pos['stop_loss'])
                                st.metric("Distance to SL", f"{sl_distance:.2f} pts", 
                                         f"{(sl_distance/current_price*100):.2f}%")
                            with col2:
                                target_distance = abs(pos['target'] - current_price)
                                st.metric("Distance to Target", f"{target_distance:.2f} pts",
                                         f"{(target_distance/current_price*100):.2f}%")
                            
                            # Guidance
                            st.subheader("🤝 Trading Guidance")
                            
                            if pnl > 0:
                                if pnl >= (pos['target'] - pos['entry_price']) * 0.8:
                                    st.success("✅ Trade is near target! Consider booking profits or trailing stop loss.")
                                elif pnl >= (pos['target'] - pos['entry_price']) * 0.5:
                                    st.info("📈 Trade is going well. Consider moving SL to breakeven.")
                                else:
                                    st.info("📊 Trade is profitable. Hold position and let it run.")
                            else:
                                if abs(pnl) >= abs(pos['entry_price'] - pos['stop_loss']) * 0.8:
                                    st.warning("⚠️ Trade is near stop loss. Be prepared for exit.")
                                else:
                                    st.warning("📉 Trade is currently negative. Stick to your plan and wait.")
                            
                            st.write(f"**Duration:** {datetime.now(IST) - pos['entry_time']}")
                
                # Auto-refresh (simulated with rerun button)
                time.sleep(2)
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
