"""
Production-Grade Quantitative Trading System
Supports: Ratio Strategy, EMA Crossover, RSI, AI-based, Custom Strategies
Features: Live Trading, Backtesting, Advanced SL/Target Management
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
            st.error(f"No data fetched for {ticker}")
            return pd.DataFrame()
        
        # Flatten multi-index if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip('_') for col in data.columns.values]
            # Select first ticker if multiple
            cols_to_keep = []
            for base in ['Open', 'High', 'Low', 'Close', 'Volume']:
                matching = [col for col in data.columns if base in col]
                if matching:
                    cols_to_keep.append((matching[0], base))
            data = data[[col[0] for col in cols_to_keep]]
            data.columns = [col[1] for col in cols_to_keep]
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                st.error(f"Missing column {col} in data")
                return pd.DataFrame()
        
        # Convert timezone
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert(IST)
        else:
            data.index = data.index.tz_convert(IST)
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

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
            # Analyze historical moves when ratio was in this bin
            mask = (ratio >= bin_data['left']) & (ratio <= bin_data['right'])
            
            if mask.sum() > n_candles:
                # Calculate forward returns
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
    
    # Generate signals based on expected moves
    ticker1_price = data1['Close'].iloc[-1]
    ticker2_price = data2['Close'].iloc[-1]
    
    # Ticker 1 signal
    if moves['ticker1_mean'] > 0.2:  # Expected upward move
        signal['ticker1']['action'] = 'BUY'
        signal['ticker1']['entry'] = ticker1_price
        signal['ticker1']['target'] = ticker1_price * (1 + moves['ticker1_mean'] / 100)
        signal['ticker1']['sl'] = calculate_stoploss(data1, ticker1_price, atr1, config, 'BUY')
    elif moves['ticker1_mean'] < -0.2:  # Expected downward move
        signal['ticker1']['action'] = 'SELL'
        signal['ticker1']['entry'] = ticker1_price
        signal['ticker1']['target'] = ticker1_price * (1 + moves['ticker1_mean'] / 100)
        signal['ticker1']['sl'] = calculate_stoploss(data1, ticker1_price, atr1, config, 'SELL')
    
    # Ticker 2 signal
    if moves['ticker2_mean'] > 0.2:
        signal['ticker2']['action'] = 'BUY'
        signal['ticker2']['entry'] = ticker2_price
        signal['ticker2']['target'] = ticker2_price * (1 + moves['ticker2_mean'] / 100)
        signal['ticker2']['sl'] = calculate_stoploss(data2, ticker2_price, atr2, config, 'BUY')
    elif moves['ticker2_mean'] < -0.2:
        signal['ticker2']['action'] = 'SELL'
        signal['ticker2']['entry'] = ticker2_price
        signal['ticker2']['target'] = ticker2_price * (1 + moves['ticker2_mean'] / 100)
        signal['ticker2']['sl'] = calculate_stoploss(data2, ticker2_price, atr2, config, 'SELL')
    
    signal['reason'] = f"Bin {current_bin}: T1 exp={moves['ticker1_mean']:.2f}%, T2 exp={moves['ticker2_mean']:.2f}%"
    
    return signal

# ==================== STOPLOSS & TARGET CALCULATION ====================

def calculate_stoploss(data: pd.DataFrame, entry: float, atr: pd.Series, 
                      config: Dict, action: str) -> float:
    """Calculate stoploss based on configuration"""
    sl_type = config.get('sl_type', 'custom_points')
    sl_value = config.get('sl_value', 10)
    
    if sl_type == 'custom_points':
        if action == 'BUY':
            return entry - sl_value
        else:
            return entry + sl_value
    
    elif sl_type == 'atr_based':
        atr_multiplier = config.get('atr_multiplier', 2.0)
        atr_val = atr.iloc[-1] if len(atr) > 0 else entry * 0.01
        if action == 'BUY':
            return entry - (atr_val * atr_multiplier)
        else:
            return entry + (atr_val * atr_multiplier)
    
    elif sl_type == 'previous_candle':
        if action == 'BUY':
            return data['Low'].iloc[-2] if len(data) > 1 else entry * 0.99
        else:
            return data['High'].iloc[-2] if len(data) > 1 else entry * 1.01
    
    elif sl_type == 'current_candle':
        if action == 'BUY':
            return data['Low'].iloc[-1]
        else:
            return data['High'].iloc[-1]
    
    elif sl_type == 'swing_low_high':
        swing_high, swing_low = calculate_swing_highs_lows(data)
        if action == 'BUY':
            return swing_low.iloc[-1] if len(swing_low) > 0 else entry * 0.98
        else:
            return swing_high.iloc[-1] if len(swing_high) > 0 else entry * 1.02
    
    else:  # Default
        if action == 'BUY':
            return entry - 10
        else:
            return entry + 10

def calculate_target(data: pd.DataFrame, entry: float, atr: pd.Series,
                    config: Dict, action: str) -> float:
    """Calculate target based on configuration"""
    target_type = config.get('target_type', 'custom_points')
    target_value = config.get('target_value', 20)
    
    if target_type == 'custom_points':
        if action == 'BUY':
            return entry + target_value
        else:
            return entry - target_value
    
    elif target_type == 'atr_based':
        atr_multiplier = config.get('target_atr_multiplier', 3.0)
        atr_val = atr.iloc[-1] if len(atr) > 0 else entry * 0.02
        if action == 'BUY':
            return entry + (atr_val * atr_multiplier)
        else:
            return entry - (atr_val * atr_multiplier)
    
    elif target_type == 'previous_candle':
        if action == 'BUY':
            return data['High'].iloc[-2] if len(data) > 1 else entry * 1.01
        else:
            return data['Low'].iloc[-2] if len(data) > 1 else entry * 0.99
    
    elif target_type == 'current_candle':
        if action == 'BUY':
            return data['High'].iloc[-1]
        else:
            return data['Low'].iloc[-1]
    
    elif target_type == 'swing_low_high':
        swing_high, swing_low = calculate_swing_highs_lows(data)
        if action == 'BUY':
            return swing_high.iloc[-1] if len(swing_high) > 0 else entry * 1.02
        else:
            return swing_low.iloc[-1] if len(swing_low) > 0 else entry * 0.98
    
    else:  # Default
        if action == 'BUY':
            return entry + 20
        else:
            return entry - 20

# ==================== EMA CROSSOVER STRATEGY ====================

def generate_ema_signals(data: pd.DataFrame, ema1_period: int = 9, ema2_period: int = 15,
                        config: Dict = None) -> Dict:
    """Generate EMA crossover signals"""
    if config is None:
        config = {}
    
    ema1 = calculate_ema(data, ema1_period)
    ema2 = calculate_ema(data, ema2_period)
    atr = calculate_atr(data)
    
    signal = {'action': 'HOLD', 'entry': 0, 'target': 0, 'sl': 0, 'reason': 'No crossover'}
    
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
    
    signal = {'action': 'HOLD', 'entry': 0, 'target': 0, 'sl': 0, 'reason': 'RSI neutral'}
    
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
            'highest_price': entry if action == 'BUY' else entry,
            'lowest_price': entry if action == 'SELL' else entry
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
        
        # Trailing SL
        if config.get('use_trailing_sl', False):
            trail_type = config.get('trailing_sl_type', 'points')
            
            if trade['action'] == 'BUY':
                if trail_type == 'points':
                    trail_points = config.get('trailing_sl_points', 10)
                    new_sl = current_price - trail_points
                    if new_sl > trade['sl']:
                        trade['sl'] = new_sl
                        
            elif trade['action'] == 'SELL':
                if trail_type == 'points':
                    trail_points = config.get('trailing_sl_points', 10)
                    new_sl = current_price + trail_points
                    if new_sl < trade['sl']:
                        trade['sl'] = new_sl
    
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
    """Create comprehensive charts"""
    
    # Calculate RSI for both
    rsi1 = calculate_rsi(data1)
    rsi2 = calculate_rsi(data2)
    ratio_series = ratio_info['ratio']
    ratio_df = pd.DataFrame({'Close': ratio_series})
    ratio_rsi = calculate_rsi(ratio_df)
    
    # Create subplots
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
    
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

# ==================== DHAN ORDER PLACEHOLDER ====================

def place_dhan_order(ticker: str, action: str, quantity: int, price: float):
    """
    Placeholder for Dhan API integration
    
    # Example integration (commented for future use):
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

def add_log(message: str, level: str = "INFO"):
    """Add log entry"""
    timestamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.logs.append({
        'timestamp': timestamp,
        'level': level,
        'message': message
    })

def main():
    st.set_page_config(
        page_title="Quantitative Trading System",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    st.title("📈 Production-Grade Quantitative Trading System")
    
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
        
        # Custom ticker option
        use_custom = st.checkbox("Use Custom Ticker")
        if use_custom:
            custom_ticker = st.text_input("Enter custom ticker (yfinance format)")
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
            if use_angle:
                min_angle = st.number_input("Min Angle (degrees)", min_value=0.1, max_value=90.0, value=1.0)
            
            crossover_type = st.selectbox("Crossover Type", 
                                         ["Simple", "Candle Size", "ATR-based"])
            
            if crossover_type == "Candle Size":
                min_candle_size = st.number_input("Min Candle Size (points)", value=10.0)
            elif crossover_type == "ATR-based":
                atr_threshold = st.number_input("ATR Threshold", value=0.5)
        
        elif strategy == "RSI Strategy":
            st.subheader("RSI Parameters")
            rsi_period = st.number_input("RSI Period", min_value=2, max_value=50, value=14)
            oversold = st.number_input("Oversold Level", min_value=10, max_value=40, value=30)
            overbought = st.number_input("Overbought Level", min_value=60, max_value=90, value=70)
        
        st.divider()
        
        # SL & Target Configuration
        st.subheader("Stop Loss Configuration")
        use_system_sl = st.checkbox("Use System-based SL", value=True)
        
        if use_system_sl:
            sl_type = st.selectbox(
                "SL Type",
                ["custom_points", "atr_based", "previous_candle", "current_candle", 
                 "swing_low_high", "trailing_sl"]
            )
            
            if sl_type == "custom_points":
                sl_value = st.number_input("SL Points", value=10.0)
            elif sl_type == "atr_based":
                atr_multiplier = st.number_input("ATR Multiplier", value=2.0)
            elif sl_type == "trailing_sl":
                trailing_sl_points = st.number_input("Trailing SL Points", value=10.0)
        
        st.subheader("Target Configuration")
        use_system_target = st.checkbox("Use System-based Target", value=True)
        
        if use_system_target:
            target_type = st.selectbox(
                "Target Type",
                ["custom_points", "atr_based", "previous_candle", "current_candle",
                 "swing_low_high", "trailing_target"]
            )
            
            if target_type == "custom_points":
                target_value = st.number_input("Target Points", value=20.0)
            elif target_type == "atr_based":
                target_atr_multiplier = st.number_input("Target ATR Multiplier", value=3.0)
    
    # Prepare config dictionary
    config = {
        'sl_type': sl_type if use_system_sl else 'custom_points',
        'sl_value': sl_value if sl_type == 'custom_points' else 10,
        'atr_multiplier': atr_multiplier if sl_type == 'atr_based' else 2.0,
        'trailing_sl_points': trailing_sl_points if sl_type == 'trailing_sl' else 10,
        'use_trailing_sl': sl_type == 'trailing_sl',
        'target_type': target_type if use_system_target else 'custom_points',
        'target_value': target_value if target_type == 'custom_points' else 20,
        'target_atr_multiplier': target_atr_multiplier if target_type == 'atr_based' else 3.0,
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
        
        # Control buttons
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
        
        with col_btn1:
            if st.button("▶️ Start Live", use_container_width=True):
                st.session_state.live_running = True
                add_log("Live trading started", "INFO")
        
        with col_btn2:
            if st.button("⏸️ Stop Live", use_container_width=True):
                st.session_state.live_running = False
                add_log("Live trading stopped", "INFO")
        
        # Create placeholder for dynamic updates
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        signal_placeholder = st.empty()
        
        if strategy == "Ratio Strategy":
            if st.session_state.live_running or st.button("Refresh Data"):
                try:
                    # Fetch data
                    with st.spinner("Fetching data..."):
                        data1 = fetch_data(ticker1, interval, period, is_live=(mode == "Live Trading"))
                        data2 = fetch_data(ticker2, interval, period, is_live=(mode == "Live Trading"))
                    
                    if not data1.empty and not data2.empty:
                        # Calculate ratio and bins
                        ratio_info = calculate_ratio_bins(data1, data2, num_bins)
                        
                        # Calculate expected moves
                        expected_moves = calculate_expected_moves(data1, data2, ratio_info, n_candles)
                        
                        # Calculate indicators
                        atr1 = calculate_atr(data1)
                        atr2 = calculate_atr(data2)
                        
                        # Generate signals
                        signals = generate_ratio_signals(data1, data2, ratio_info, expected_moves, 
                                                        atr1, atr2, config)
                        
                        # Update metrics
                        with metrics_placeholder.container():
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            with col1:
                                st.metric("Ticker 1", ticker1_name, f"₹{data1['Close'].iloc[-1]:.2f}")
                            with col2:
                                st.metric("Ticker 2", ticker2_name, f"₹{data2['Close'].iloc[-1]:.2f}")
                            with col3:
                                st.metric("Ratio", f"{ratio_info['ratio'].iloc[-1]:.4f}")
                            with col4:
                                st.metric("Active Trades", len(st.session_state.trade_manager.active_trades))
                            with col5:
                                stats = st.session_state.trade_manager.get_statistics()
                                st.metric("Total PnL", f"₹{stats['total_pnl']:.2f}")
                        
                        # Display signals
                        with signal_placeholder.container():
                            st.subheader("Current Signals")
                            col_s1, col_s2 = st.columns(2)
                            
                            with col_s1:
                                st.info(f"**{ticker1_name}**: {signals['ticker1']['action']}")
                                if signals['ticker1']['action'] != 'HOLD':
                                    st.write(f"Entry: ₹{signals['ticker1']['entry']:.2f}")
                                    st.write(f"Target: ₹{signals['ticker1']['target']:.2f}")
                                    st.write(f"SL: ₹{signals['ticker1']['sl']:.2f}")
                            
                            with col_s2:
                                st.info(f"**{ticker2_name}**: {signals['ticker2']['action']}")
                                if signals['ticker2']['action'] != 'HOLD':
                                    st.write(f"Entry: ₹{signals['ticker2']['entry']:.2f}")
                                    st.write(f"Target: ₹{signals['ticker2']['target']:.2f}")
                                    st.write(f"SL: ₹{signals['ticker2']['sl']:.2f}")
                            
                            st.caption(f"Reason: {signals['reason']}")
                        
                        # Plot charts
                        with chart_placeholder.container():
                            fig = plot_charts(data1, data2, ratio_info, ticker1_name, ticker2_name)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Log update
                        add_log(f"Data updated - {ticker1_name}: {signals['ticker1']['action']}, "
                               f"{ticker2_name}: {signals['ticker2']['action']}", "INFO")
                        
                        st.session_state.last_update = datetime.now(IST)
                
                except Exception as e:
                    st.error(f"Error in live trading: {str(e)}")
                    add_log(f"Error: {str(e)}", "ERROR")
        
        else:  # Other strategies
            if st.session_state.live_running or st.button("Refresh Data"):
                try:
                    with st.spinner("Fetching data..."):
                        data = fetch_data(ticker, interval, period, is_live=(mode == "Live Trading"))
                    
                    if not data.empty:
                        # Generate signals based on strategy
                        if strategy == "EMA Crossover":
                            signal = generate_ema_signals(data, ema1_period, ema2_period, config)
                        elif strategy == "RSI Strategy":
                            signal = generate_rsi_signals(data, rsi_period, oversold, overbought, config)
                        else:
                            signal = {'action': 'HOLD', 'entry': 0, 'target': 0, 'sl': 0, 'reason': 'Strategy not implemented'}
                        
                        # Update metrics
                        with metrics_placeholder.container():
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Price", f"₹{data['Close'].iloc[-1]:.2f}")
                            with col2:
                                st.metric("Signal", signal['action'])
                            with col3:
                                st.metric("Active Trades", len(st.session_state.trade_manager.active_trades))
                            with col4:
                                stats = st.session_state.trade_manager.get_statistics()
                                st.metric("Total PnL", f"₹{stats['total_pnl']:.2f}")
                        
                        # Display signal
                        with signal_placeholder.container():
                            if signal['action'] != 'HOLD':
                                st.success(f"**Signal**: {signal['action']}")
                                st.write(f"Entry: ₹{signal['entry']:.2f}")
                                st.write(f"Target: ₹{signal['target']:.2f}")
                                st.write(f"SL: ₹{signal['sl']:.2f}")
                                st.caption(f"Reason: {signal['reason']}")
                            else:
                                st.info("No active signal")
                        
                        add_log(f"Signal: {signal['action']} - {signal['reason']}", "INFO")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    add_log(f"Error: {str(e)}", "ERROR")
    
    # ==================== TAB 2: TRADE HISTORY ====================
    with tab2:
        st.header("Trade History")
        
        stats = st.session_state.trade_manager.get_statistics()
        
        # Statistics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total Trades", stats['total_trades'])
        col2.metric("Winning", stats['winning_trades'])
        col3.metric("Losing", stats['losing_trades'])
        col4.metric("Accuracy", f"{stats['accuracy']:.2f}%")
        col5.metric("Total PnL", f"₹{stats['total_pnl']:.2f}")
        col6.metric("Avg Win", f"₹{stats['avg_win']:.2f}")
        
        st.divider()
        
        # Trade table
        if st.session_state.trade_manager.closed_trades:
            trades_df = pd.DataFrame(st.session_state.trade_manager.closed_trades)
            
            # Format for display
            display_df = trades_df[[
                'trade_id', 'ticker', 'action', 'entry_price', 'exit_price',
                'entry_time', 'exit_time', 'target', 'sl', 'pnl_points',
                'pnl_total', 'exit_reason', 'reason'
            ]].copy()
            
            display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['exit_time'] = pd.to_datetime(display_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Color code PnL
            def highlight_pnl(val):
                color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
                return f'color: {color}'
            
            styled_df = display_df.style.applymap(highlight_pnl, subset=['pnl_total'])
            st.dataframe(styled_df, use_container_width=True)
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
            
            # Color code by level
            def color_level(val):
                colors = {'INFO': 'blue', 'WARNING': 'orange', 'ERROR': 'red', 'SUCCESS': 'green'}
                return f'color: {colors.get(val, "black")}'
            
            styled_logs = logs_df.style.applymap(color_level, subset=['level'])
            st.dataframe(styled_logs, use_container_width=True, height=600)
        else:
            st.info("No logs yet")
    
    # ==================== TAB 4: BACKTESTING ====================
    with tab4:
        st.header("Backtesting")
        
        if st.button("Run Backtest", type="primary", use_container_width=True):
            try:
                with st.spinner("Running backtest..."):
                    # Initialize new trade manager for backtest
                    backtest_tm = TradeManager()
                    
                    if strategy == "Ratio Strategy":
                        # Fetch data
                        data1 = fetch_data(ticker1, interval, period, is_live=False)
                        data2 = fetch_data(ticker2, interval, period, is_live=False)
                        
                        if not data1.empty and not data2.empty:
                            # Run backtest
                            for i in range(len(data1) - n_candles):
                                window_data1 = data1.iloc[:i+1]
                                window_data2 = data2.iloc[:i+1]
                                
                                ratio_info = calculate_ratio_bins(window_data1, window_data2, num_bins)
                                expected_moves = calculate_expected_moves(window_data1, window_data2, 
                                                                         ratio_info, n_candles)
                                
                                atr1 = calculate_atr(window_data1)
                                atr2 = calculate_atr(window_data2)
                                
                                signals = generate_ratio_signals(window_data1, window_data2, ratio_info,
                                                                expected_moves, atr1, atr2, config)
                                
                                # Process signals
                                for ticker_key, ticker_sig in [('ticker1', ticker1_name), ('ticker2', ticker2_name)]:
                                    sig = signals[ticker_key]
                                    if sig['action'] != 'HOLD':
                                        trade_id = backtest_tm.open_trade(
                                            ticker_sig, sig['action'], sig['entry'],
                                            sig['target'], sig['sl'], quantity, sig.get('reason', '')
                                        )
                                        
                                        # Check exit in subsequent candles
                                        for j in range(i+1, min(i+n_candles+1, len(data1))):
                                            current_price = data1['Close'].iloc[j] if ticker_key == 'ticker1' else data2['Close'].iloc[j]
                                            should_exit, exit_reason = backtest_tm.check_exit(trade_id, current_price)
                                            
                                            if should_exit:
                                                backtest_tm.close_trade(trade_id, current_price, exit_reason)
                                                break
                    
                    else:  # Other strategies
                        data = fetch_data(ticker, interval, period, is_live=False)
                        
                        if not data.empty:
                            for i in range(50, len(data)):  # Start after warmup
                                window_data = data.iloc[:i+1]
                                
                                if strategy == "EMA Crossover":
                                    signal = generate_ema_signals(window_data, ema1_period, ema2_period, config)
                                elif strategy == "RSI Strategy":
                                    signal = generate_rsi_signals(window_data, rsi_period, oversold, overbought, config)
                                else:
                                    continue
                                
                                if signal['action'] != 'HOLD':
                                    trade_id = backtest_tm.open_trade(
                                        ticker_name, signal['action'], signal['entry'],
                                        signal['target'], signal['sl'], quantity, signal['reason']
                                    )
                                    
                                    # Check exit in subsequent candles
                                    for j in range(i+1, len(data)):
                                        current_price = data['Close'].iloc[j]
                                        should_exit, exit_reason = backtest_tm.check_exit(trade_id, current_price)
                                        
                                        if should_exit:
                                            backtest_tm.close_trade(trade_id, current_price, exit_reason)
                                            break
                    
                    st.session_state.backtest_results = backtest_tm
                    add_log("Backtest completed", "SUCCESS")
            
            except Exception as e:
                st.error(f"Backtest error: {str(e)}")
                add_log(f"Backtest error: {str(e)}", "ERROR")
        
        # Display backtest results
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
            
            # Trade table
            if st.session_state.backtest_results.closed_trades:
                bt_trades_df = pd.DataFrame(st.session_state.backtest_results.closed_trades)
                
                display_bt_df = bt_trades_df[[
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
    st.caption(f"Last Update: {last_update_str} | Mode: {mode} | Strategy: {strategy}")

if __name__ == "__main__":
    main()
