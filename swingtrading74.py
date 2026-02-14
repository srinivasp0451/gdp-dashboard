"""
Complete Algorithmic Trading System
=====================================

A comprehensive trading system with:
- Multiple trading strategies (EMA Crossover, RSI-ADX-EMA, etc.)
- 18+ Stop Loss types including Cost-to-Cost trailing
- 12+ Target types
- Real broker integration (Dhan API)
- Live trading and backtesting capabilities
- Advanced position management

Author: Claude
Version: 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import pytz
import traceback
import plotly.graph_objects as go

# ================================
# CONSTANTS & MAPPINGS
# ================================

ASSET_MAPPING = {
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "FINNIFTY": "^NSEI",  # Placeholder
    "MIDCPNIFTY": "^NSEI",  # Placeholder
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "CRUDE OIL": "CL=F",
    "SPY": "SPY",
    "QQQ": "QQQ",
}

INTERVAL_MAPPING = {
    "1 minute": "1m",
    "5 minutes": "5m",
    "15 minutes": "15m",
    "30 minutes": "30m",
    "1 hour": "1h",
    "1 day": "1d",
    "1 week": "1wk",
}

PERIOD_MAPPING = {
    "1 day": "1d",
    "5 days": "5d",
    "1 month": "1mo",
    "3 months": "3mo",
    "6 months": "6mo",
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y",
}

STRATEGY_LIST = [
    "EMA Crossover",
    "Simple Buy",
    "Simple Sell",
    "Price Crosses Threshold",
    "RSI-ADX-EMA Combined",
    "Percentage Change",
    "AI Price Action",
    "Custom Strategy"
]

SL_TYPES = [
    "Custom Points",
    "P&L Based (Rupees)",
    "ATR-based",
    "Current Candle Low/High",
    "Previous Candle Low/High",
    "Current Swing Low/High",
    "Previous Swing Low/High",
    "Signal-based (Reverse Crossover)",
    "Trailing SL (Points)",
    "Trailing Profit (Rupees)",
    "Trailing Loss (Rupees)",
    "Trailing SL + Current Candle",
    "Trailing SL + Previous Candle",
    "Trailing SL + Current Swing",
    "Trailing SL + Previous Swing",
    "Volatility-Adjusted Trailing SL",
    "Break-even After 50% Target",
    "Cost-to-Cost + N Points Trailing SL"
]

TARGET_TYPES = [
    "Custom Points",
    "P&L Based (Rupees)",
    "Trailing Target (Points)",
    "Trailing Target + Signal Based",
    "50% Exit at Target (Partial)",
    "Current Candle Low/High",
    "Previous Candle Low/High",
    "Current Swing Low/High",
    "Previous Swing Low/High",
    "ATR-based",
    "Risk-Reward Based",
    "Signal-based (Reverse Crossover)"
]

EMA_ENTRY_FILTERS = [
    "Simple Crossover",
    "Custom Candle (Points)",
    "ATR-based Candle"
]

# ================================
# DHAN BROKER INTEGRATION CLASS
# ================================

class DhanBrokerIntegration:
    """Handles Dhan API integration for live trading"""
    
    def __init__(self, config):
        """Initialize Dhan broker with configuration"""
        self.config = config
        self.dhan = None
        self.initialized = False
        
        # Try to import and initialize Dhan
        try:
            from dhanhq import dhanhq
            self.dhanhq_module = dhanhq
            
            if config.get('dhan_enabled', False):
                client_id = config.get('dhan_client_id', '')
                access_token = config.get('dhan_access_token', '')
                
                if client_id and access_token:
                    self.dhan = dhanhq(client_id, access_token)
                    self.initialized = True
                    st.success("✅ Dhan API initialized successfully")
                else:
                    st.warning("⚠️ Dhan credentials missing, using simulation mode")
        except ImportError:
            st.warning("⚠️ dhanhq module not installed, using simulation mode")
            self.dhanhq_module = None
        except Exception as e:
            st.error(f"⚠️ Dhan initialization error: {e}")
            
    def _resolve_security(self, signal):
        """
        Resolve security ID based on signal type
        
        Args:
            signal: 'BUY', 'SELL', 'LONG', or 'SHORT'
            
        Returns:
            tuple: (security_id, option_type)
        """
        # Convert to normalized signal
        if signal in ('BUY', 'LONG'):
            # LONG signal → Use CE
            security_id = self.config.get('dhan_ce_security_id', '42568')
            option_type = 'CE'
        else:  # 'SELL' or 'SHORT'
            # SHORT signal → Use PE
            security_id = self.config.get('dhan_pe_security_id', '42569')
            option_type = 'PE'
            
        return security_id, option_type
    
    def _get_exchange_segment(self):
        """Determine exchange segment based on asset"""
        if not self.dhanhq_module:
            return "NSE_FNO"
            
        asset = self.config.get('asset', 'NIFTY 50')
        
        if asset in ('NIFTY 50', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'):
            return self.dhanhq_module.NSE_FNO
        elif asset == 'SENSEX':
            return self.dhanhq_module.BSE_FNO
        else:
            return self.dhanhq_module.NSE_FNO  # default
    
    def place_order(self, transaction_type, security_id, quantity, signal_type=None):
        """
        Place order via Dhan API
        
        Args:
            transaction_type: 'BUY' or 'SELL'
            security_id: Security ID
            quantity: Order quantity
            signal_type: Original signal ('BUY', 'SELL', 'LONG', 'SHORT') for logging
            
        Returns:
            dict: Order response with order_id, status, raw_response
        """
        order_response = {
            'order_id': None,
            'status': 'FAILED',
            'raw_response': None,
            'error': None
        }
        
        try:
            if self.initialized and self.dhan:
                # Real API call
                exchange_segment = self._get_exchange_segment()
                
                response = self.dhan.place_order(
                    security_id=str(security_id),
                    exchange_segment=exchange_segment,
                    transaction_type=transaction_type,
                    quantity=int(quantity),
                    order_type=self.dhanhq_module.MARKET,
                    product_type=self.dhanhq_module.INTRA,
                    price=0
                )
                
                order_response['raw_response'] = response
                
                if response and response.get('status') == 'success':
                    order_response['order_id'] = response.get('data', {}).get('orderId', f"ORDER-{int(time.time())}")
                    order_response['status'] = 'SUCCESS'
                else:
                    order_response['order_id'] = f"ERR-{int(time.time())}"
                    order_response['error'] = response.get('remarks', 'Unknown error')
                    
            else:
                # Simulation mode
                order_response['order_id'] = f"SIM-{int(time.time())}"
                order_response['status'] = 'SIMULATED'
                order_response['raw_response'] = {
                    'mode': 'simulation',
                    'reason': 'API not initialized or dhanhq not installed'
                }
                
        except Exception as e:
            order_response['order_id'] = f"ERR-{int(time.time())}"
            order_response['error'] = str(e)
            order_response['raw_response'] = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
        return order_response
    
    def enter_broker_position(self, signal, price, config, log_func):
        """
        Enter broker position (ALWAYS BUY)
        
        Args:
            signal: 'BUY', 'SELL', 'LONG', or 'SHORT'
            price: Entry price
            config: Configuration dict
            log_func: Logging function
            
        Returns:
            dict: Broker position info
        """
        # Resolve security based on signal
        security_id, option_type = self._resolve_security(signal)
        quantity = config.get('dhan_quantity', 65)
        
        log_func(f"🏦 NEW signal detected: {signal}")
        log_func(f"🏦 {signal} signal → Using {option_type} Security ID: {security_id}")
        
        # ALWAYS BUY to open position
        order_response = self.place_order('BUY', security_id, quantity, signal)
        
        # Create broker position record
        broker_position = {
            'order_id': order_response['order_id'],
            'signal_type': signal,
            'option_type': option_type,
            'security_id': security_id,
            'transaction_type': 'BUY',
            'entry_price': price,
            'quantity': quantity,
            'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')),
            'status': order_response['status'],
            'raw_response': order_response['raw_response']
        }
        
        if order_response['status'] in ('SUCCESS', 'SIMULATED'):
            log_func(f"🏦 ✅ DHAN ORDER PLACED: BUY {quantity} @ {price:.2f}")
        else:
            log_func(f"🏦 ❌ DHAN ORDER FAILED: {order_response.get('error', 'Unknown error')}")
            
        return broker_position
    
    def exit_broker_position(self, broker_position, price, reason, log_func):
        """
        Exit broker position (ALWAYS SELL)
        
        Args:
            broker_position: Existing broker position dict
            price: Exit price
            reason: Exit reason
            log_func: Logging function
            
        Returns:
            dict: Exit order info
        """
        security_id = broker_position['security_id']
        quantity = broker_position['quantity']
        
        log_func(f"🏦 Exiting position: {reason}")
        
        # ALWAYS SELL to close position
        order_response = self.place_order('SELL', security_id, quantity)
        
        # Calculate P&L
        entry_price = broker_position['entry_price']
        signal_type = broker_position['signal_type']
        
        if signal_type in ('BUY', 'LONG'):
            pnl = (price - entry_price) * quantity
        else:  # 'SELL' or 'SHORT'
            pnl = (entry_price - price) * quantity
            
        exit_info = {
            'order_id': order_response['order_id'],
            'transaction_type': 'SELL',
            'exit_price': price,
            'quantity': quantity,
            'pnl': pnl,
            'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')),
            'status': order_response['status'],
            'raw_response': order_response['raw_response']
        }
        
        if order_response['status'] in ('SUCCESS', 'SIMULATED'):
            log_func(f"🏦 ✅ DHAN EXIT ORDER PLACED: SELL {quantity} @ {price:.2f} | P&L: ₹{pnl:.2f}")
        else:
            log_func(f"🏦 ❌ DHAN EXIT ORDER FAILED: {order_response.get('error', 'Unknown error')}")
            
        return exit_info

# ================================
# DATA FETCHING
# ================================

def fetch_data(ticker_symbol, interval, period, is_live_trading=False):
    """
    Fetch historical/live data using yfinance
    
    Args:
        ticker_symbol: Asset ticker
        interval: Time interval
        period: Historical period
        is_live_trading: If True, fetch minimal data for live trading
        
    Returns:
        DataFrame with OHLCV data in IST timezone
    """
    try:
        ticker = ASSET_MAPPING.get(ticker_symbol, ticker_symbol)
        
        if is_live_trading:
            # For live trading, fetch smaller dataset
            if interval in ['1m', '5m', '15m', '30m']:
                period = '1d'
            elif interval in ['1h']:
                period = '5d'
            else:
                period = '1mo'
        
        # Download data
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if df is None or df.empty:
            st.error(f"❌ No data returned for {ticker_symbol}")
            return None
            
        # Ensure proper column names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Reset index to make Datetime a column
        df = df.reset_index()
        
        # Convert to IST timezone
        ist = pytz.timezone('Asia/Kolkata')
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_convert(ist)
        elif 'Date' in df.columns:
            df.rename(columns={'Date': 'Datetime'}, inplace=True)
            df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(ist)
            
        return df
        
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        return None

# ================================
# INDICATOR CALCULATIONS
# ================================

def calculate_ema_angle(ema_series, lookback=3):
    """
    Calculate EMA angle in degrees
    
    Args:
        ema_series: EMA values
        lookback: Number of periods for angle calculation
        
    Returns:
        Series with angle values
    """
    # Calculate slope using linear regression over lookback period
    angles = []
    
    for i in range(len(ema_series)):
        if i < lookback:
            angles.append(np.nan)
        else:
            y = ema_series.iloc[i-lookback:i+1].values
            x = np.arange(len(y))
            
            # Linear regression
            if len(y) > 1 and not np.any(np.isnan(y)):
                slope = np.polyfit(x, y, 1)[0]
                # Convert slope to angle in degrees
                angle = np.degrees(np.arctan(slope))
                angles.append(angle)
            else:
                angles.append(np.nan)
                
    return pd.Series(angles, index=ema_series.index)

def calculate_rsi(series, period=14):
    """Calculate RSI indicator"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_adx(df, period=14):
    """Calculate ADX indicator"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Calculate +DM and -DM
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # Calculate True Range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate smoothed +DI and -DI
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # Calculate DX and ADX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.rolling(window=period).mean()
    
    return adx

def calculate_atr(df, period=14):
    """Calculate ATR indicator"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_all_indicators(df, config):
    """
    Calculate all technical indicators
    
    Args:
        df: DataFrame with OHLCV data
        config: Configuration dict with indicator parameters
        
    Returns:
        DataFrame with all indicators added
    """
    # EMA Fast and Slow
    ema_fast = config.get('ema_fast', 9)
    ema_slow = config.get('ema_slow', 21)
    
    df['EMA_Fast'] = df['Close'].ewm(span=ema_fast, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=ema_slow, adjust=False).mean()
    
    # EMA Angle
    df['EMA_Fast_Angle'] = calculate_ema_angle(df['EMA_Fast'])
    df['EMA_Slow_Angle'] = calculate_ema_angle(df['EMA_Slow'])
    
    # RSI
    df['RSI'] = calculate_rsi(df['Close'], 14)
    
    # ADX
    adx_period = config.get('adx_period', 14)
    df['ADX'] = calculate_adx(df, adx_period)
    
    # ATR
    df['ATR'] = calculate_atr(df, 14)
    
    # Swing Highs and Lows
    df['Swing_High'] = df['High'].rolling(window=5, center=True).max()
    df['Swing_Low'] = df['Low'].rolling(window=5, center=True).min()
    
    # Previous values for reference
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Swing_High'] = df['Swing_High'].shift(1)
    df['Prev_Swing_Low'] = df['Swing_Low'].shift(1)
    
    return df

# ================================
# STRATEGY FUNCTIONS
# ================================

def check_ema_crossover_strategy(df, idx, config, current_position):
    """
    EMA Crossover Strategy with advanced filters
    
    Filters:
    - Minimum Angle (ABSOLUTE value)
    - Entry Filters (Simple/Custom Candle Points/ATR-based)
    - ADX Filter (optional)
    
    Returns:
        tuple: (signal, price) where signal is 'BUY', 'SELL', or None
    """
    if idx < 1:
        return None, None
        
    current = df.iloc[idx]
    previous = df.iloc[idx - 1]
    
    # Check if indicators are ready
    if pd.isna(current['EMA_Fast']) or pd.isna(current['EMA_Slow']):
        return None, None
    
    # Detect crossover
    bullish_cross = (previous['EMA_Fast'] <= previous['EMA_Slow'] and 
                     current['EMA_Fast'] > current['EMA_Slow'])
    bearish_cross = (previous['EMA_Fast'] >= previous['EMA_Slow'] and 
                     current['EMA_Fast'] < current['EMA_Slow'])
    
    if not bullish_cross and not bearish_cross:
        return None, None
    
    # Apply Minimum Angle Filter (ABSOLUTE value)
    min_angle = config.get('ema_min_angle', 0.0)
    fast_angle = abs(current['EMA_Fast_Angle']) if not pd.isna(current['EMA_Fast_Angle']) else 0
    
    if fast_angle < min_angle:
        return None, None
    
    # Apply Entry Filter
    entry_filter = config.get('ema_entry_filter', 'Simple Crossover')
    
    if entry_filter == 'Custom Candle (Points)':
        # Candle body must be >= N points
        min_points = config.get('ema_custom_candle_points', 5)
        candle_body = abs(current['Close'] - current['Open'])
        
        if candle_body < min_points:
            return None, None
            
    elif entry_filter == 'ATR-based Candle':
        # Candle body must be >= ATR × multiplier
        if pd.isna(current['ATR']):
            return None, None
            
        atr_multiplier = config.get('ema_atr_multiplier', 0.3)
        min_body = current['ATR'] * atr_multiplier
        candle_body = abs(current['Close'] - current['Open'])
        
        if candle_body < min_body:
            return None, None
    
    # Apply ADX Filter (optional)
    if config.get('ema_use_adx', False):
        adx_threshold = config.get('ema_adx_threshold', 20)
        
        if pd.isna(current['ADX']) or current['ADX'] < adx_threshold:
            return None, None
    
    # Generate signal
    if bullish_cross:
        return 'BUY', current['Close']
    elif bearish_cross:
        return 'SELL', current['Close']
    
    return None, None

def check_simple_buy_strategy(df, idx, config, current_position):
    """Simple Buy strategy - always returns BUY immediately if no position"""
    if current_position is not None:
        return None, None
    
    # Always signal BUY if no position exists
    return 'BUY', df.iloc[idx]['Close']

def check_simple_sell_strategy(df, idx, config, current_position):
    """Simple Sell strategy - always returns SELL immediately if no position"""
    if current_position is not None:
        return None, None
    
    # Always signal SELL if no position exists
    return 'SELL', df.iloc[idx]['Close']

def check_price_crosses_threshold(df, idx, config, current_position):
    """Price crosses threshold strategy"""
    if current_position is not None:
        return None, None
        
    threshold = config.get('price_threshold', 25000)
    current_price = df.iloc[idx]['Close']
    
    if idx < 1:
        return None, None
        
    prev_price = df.iloc[idx - 1]['Close']
    
    # Bullish cross above threshold
    if prev_price <= threshold and current_price > threshold:
        return 'BUY', current_price
    
    # Bearish cross below threshold
    if prev_price >= threshold and current_price < threshold:
        return 'SELL', current_price
    
    return None, None

def check_rsi_adx_ema_combined(df, idx, config, current_position):
    """Combined RSI-ADX-EMA strategy"""
    if current_position is not None:
        return None, None
        
    if idx < 1:
        return None, None
        
    current = df.iloc[idx]
    
    # Check if indicators are ready
    if pd.isna(current['RSI']) or pd.isna(current['ADX']) or pd.isna(current['EMA_Fast']):
        return None, None
    
    rsi = current['RSI']
    adx = current['ADX']
    price = current['Close']
    ema = current['EMA_Fast']
    
    # BUY: RSI oversold, strong trend, price above EMA
    if rsi < 30 and adx > 25 and price > ema:
        return 'BUY', price
    
    # SELL: RSI overbought, strong trend, price below EMA
    if rsi > 70 and adx > 25 and price < ema:
        return 'SELL', price
    
    return None, None

def check_percentage_change(df, idx, config, current_position):
    """Percentage change strategy"""
    if current_position is not None:
        return None, None
        
    if idx < 1:
        return None, None
        
    current_price = df.iloc[idx]['Close']
    prev_price = df.iloc[idx - 1]['Close']
    
    pct_change = ((current_price - prev_price) / prev_price) * 100
    threshold = config.get('pct_change_threshold', 2.0)
    
    if pct_change >= threshold:
        return 'BUY', current_price
    elif pct_change <= -threshold:
        return 'SELL', current_price
    
    return None, None

def check_ai_price_action(df, idx, config, current_position):
    """AI Price Action - simplified pattern recognition"""
    if current_position is not None:
        return None, None
        
    if idx < 3:
        return None, None
        
    # Look for bullish/bearish patterns in last 3 candles
    candles = df.iloc[idx-2:idx+1]
    
    # Bullish pattern: three consecutive higher closes
    if all(candles['Close'].diff().dropna() > 0):
        return 'BUY', df.iloc[idx]['Close']
    
    # Bearish pattern: three consecutive lower closes
    if all(candles['Close'].diff().dropna() < 0):
        return 'SELL', df.iloc[idx]['Close']
    
    return None, None

def check_custom_strategy(df, idx, config, current_position):
    """Custom strategy placeholder - can be customized by user"""
    # User can implement their own logic here
    return None, None

# Strategy mapping
STRATEGY_FUNCTIONS = {
    'EMA Crossover': check_ema_crossover_strategy,
    'Simple Buy': check_simple_buy_strategy,
    'Simple Sell': check_simple_sell_strategy,
    'Price Crosses Threshold': check_price_crosses_threshold,
    'RSI-ADX-EMA Combined': check_rsi_adx_ema_combined,
    'Percentage Change': check_percentage_change,
    'AI Price Action': check_ai_price_action,
    'Custom Strategy': check_custom_strategy,
}

# ================================
# STOP LOSS CALCULATION
# ================================

def calculate_initial_sl(position_type, entry_price, df, idx, config):
    """
    Calculate initial stop loss based on SL type
    
    Args:
        position_type: 'LONG' or 'SHORT'
        entry_price: Entry price
        df: DataFrame with indicators
        idx: Current index
        config: Configuration dict
        
    Returns:
        float: Stop loss price
    """
    sl_type = config.get('sl_type', 'Custom Points')
    current = df.iloc[idx]
    
    if sl_type == 'Custom Points':
        points = config.get('sl_points', 10)
        if position_type == 'LONG':
            return entry_price - points
        else:  # SHORT
            return entry_price + points
    
    elif sl_type == 'P&L Based (Rupees)':
        rupees = config.get('sl_rupees', 100)
        quantity = config.get('quantity', 1)
        points = rupees / quantity
        
        if position_type == 'LONG':
            return entry_price - points
        else:  # SHORT
            return entry_price + points
    
    elif sl_type == 'ATR-based':
        if pd.isna(current['ATR']):
            # Fallback to custom points
            points = config.get('sl_points', 10)
            if position_type == 'LONG':
                return entry_price - points
            else:
                return entry_price + points
        
        multiplier = config.get('sl_atr_multiplier', 1.5)
        sl_distance = current['ATR'] * multiplier
        
        if position_type == 'LONG':
            return entry_price - sl_distance
        else:  # SHORT
            return entry_price + sl_distance
    
    elif sl_type == 'Current Candle Low/High':
        if position_type == 'LONG':
            return current['Low']
        else:  # SHORT
            return current['High']
    
    elif sl_type == 'Previous Candle Low/High':
        if pd.isna(current['Prev_Low']) or pd.isna(current['Prev_High']):
            # Fallback
            points = config.get('sl_points', 10)
            if position_type == 'LONG':
                return entry_price - points
            else:
                return entry_price + points
        
        if position_type == 'LONG':
            return current['Prev_Low']
        else:  # SHORT
            return current['Prev_High']
    
    elif sl_type == 'Current Swing Low/High':
        if pd.isna(current['Swing_Low']) or pd.isna(current['Swing_High']):
            # Fallback
            points = config.get('sl_points', 10)
            if position_type == 'LONG':
                return entry_price - points
            else:
                return entry_price + points
        
        if position_type == 'LONG':
            return current['Swing_Low']
        else:  # SHORT
            return current['Swing_High']
    
    elif sl_type == 'Previous Swing Low/High':
        if pd.isna(current['Prev_Swing_Low']) or pd.isna(current['Prev_Swing_High']):
            # Fallback
            points = config.get('sl_points', 10)
            if position_type == 'LONG':
                return entry_price - points
            else:
                return entry_price + points
        
        if position_type == 'LONG':
            return current['Prev_Swing_Low']
        else:  # SHORT
            return current['Prev_Swing_High']
    
    elif sl_type == 'Signal-based (Reverse Crossover)':
        # No initial SL, will be set on reverse crossover
        return None
    
    elif sl_type in ['Trailing SL (Points)', 'Trailing Profit (Rupees)', 'Trailing Loss (Rupees)',
                     'Trailing SL + Current Candle', 'Trailing SL + Previous Candle',
                     'Trailing SL + Current Swing', 'Trailing SL + Previous Swing',
                     'Volatility-Adjusted Trailing SL', 'Break-even After 50% Target']:
        # Trailing types start with initial SL
        points = config.get('sl_points', 10)
        if position_type == 'LONG':
            return entry_price - points
        else:  # SHORT
            return entry_price + points
    
    elif sl_type == 'Cost-to-Cost + N Points Trailing SL':
        # CTC starts with initial SL distance
        points = config.get('sl_points', 10)
        if position_type == 'LONG':
            return entry_price - points
        else:  # SHORT
            return entry_price + points
    
    else:
        # Default fallback
        points = config.get('sl_points', 10)
        if position_type == 'LONG':
            return entry_price - points
        else:  # SHORT
            return entry_price + points

# ================================
# TARGET CALCULATION
# ================================

def calculate_initial_target(position_type, entry_price, df, idx, config):
    """
    Calculate initial target based on target type
    
    Args:
        position_type: 'LONG' or 'SHORT'
        entry_price: Entry price
        df: DataFrame with indicators
        idx: Current index
        config: Configuration dict
        
    Returns:
        float: Target price
    """
    target_type = config.get('target_type', 'Custom Points')
    current = df.iloc[idx]
    
    if target_type == 'Custom Points':
        points = config.get('target_points', 20)
        if position_type == 'LONG':
            return entry_price + points
        else:  # SHORT
            return entry_price - points
    
    elif target_type == 'P&L Based (Rupees)':
        rupees = config.get('target_rupees', 200)
        quantity = config.get('quantity', 1)
        points = rupees / quantity
        
        if position_type == 'LONG':
            return entry_price + points
        else:  # SHORT
            return entry_price - points
    
    elif target_type == 'Risk-Reward Based':
        # Calculate based on SL and risk-reward ratio
        sl_price = calculate_initial_sl(position_type, entry_price, df, idx, config)
        rr_ratio = config.get('risk_reward_ratio', 2.0)
        
        if sl_price is None:
            # Fallback
            points = config.get('target_points', 20)
            if position_type == 'LONG':
                return entry_price + points
            else:
                return entry_price - points
        
        risk = abs(entry_price - sl_price)
        reward = risk * rr_ratio
        
        if position_type == 'LONG':
            return entry_price + reward
        else:  # SHORT
            return entry_price - reward
    
    elif target_type == 'ATR-based':
        if pd.isna(current['ATR']):
            # Fallback
            points = config.get('target_points', 20)
            if position_type == 'LONG':
                return entry_price + points
            else:
                return entry_price - points
        
        multiplier = config.get('target_atr_multiplier', 2.0)
        target_distance = current['ATR'] * multiplier
        
        if position_type == 'LONG':
            return entry_price + target_distance
        else:  # SHORT
            return entry_price - target_distance
    
    elif target_type == 'Current Candle Low/High':
        if position_type == 'LONG':
            return current['High']
        else:  # SHORT
            return current['Low']
    
    elif target_type == 'Previous Candle Low/High':
        if pd.isna(current['Prev_Low']) or pd.isna(current['Prev_High']):
            # Fallback
            points = config.get('target_points', 20)
            if position_type == 'LONG':
                return entry_price + points
            else:
                return entry_price - points
        
        if position_type == 'LONG':
            return current['Prev_High']
        else:  # SHORT
            return current['Prev_Low']
    
    elif target_type == 'Current Swing Low/High':
        if pd.isna(current['Swing_Low']) or pd.isna(current['Swing_High']):
            # Fallback
            points = config.get('target_points', 20)
            if position_type == 'LONG':
                return entry_price + points
            else:
                return entry_price - points
        
        if position_type == 'LONG':
            return current['Swing_High']
        else:  # SHORT
            return current['Swing_Low']
    
    elif target_type == 'Previous Swing Low/High':
        if pd.isna(current['Prev_Swing_Low']) or pd.isna(current['Prev_Swing_High']):
            # Fallback
            points = config.get('target_points', 20)
            if position_type == 'LONG':
                return entry_price + points
            else:
                return entry_price - points
        
        if position_type == 'LONG':
            return current['Prev_Swing_High']
        else:  # SHORT
            return current['Prev_Swing_Low']
    
    elif target_type in ['Trailing Target (Points)', 'Trailing Target + Signal Based',
                         '50% Exit at Target (Partial)', 'Signal-based (Reverse Crossover)']:
        # Start with initial target
        points = config.get('target_points', 20)
        if position_type == 'LONG':
            return entry_price + points
        else:  # SHORT
            return entry_price - points
    
    else:
        # Default fallback
        points = config.get('target_points', 20)
        if position_type == 'LONG':
            return entry_price + points
        else:  # SHORT
            return entry_price - points

# ================================
# TRAILING UPDATES
# ================================

def update_trailing_sl(position, current_price, df, idx, config):
    """
    Update trailing stop loss
    
    Implements all trailing SL types including:
    - Standard trailing
    - Cost-to-Cost 3-phase trailing
    - Candle/Swing-based trailing
    - Volatility-adjusted trailing
    
    Args:
        position: Current position dict
        current_price: Current market price
        df: DataFrame with indicators
        idx: Current index
        config: Configuration dict
        
    Returns:
        float: Updated SL price (or existing if no update)
    """
    sl_type = config.get('sl_type', 'Custom Points')
    position_type = position['type']
    current_sl = position['sl_price']
    entry_price = position['entry_price']
    current = df.iloc[idx]
    
    # ============================================
    # COST-TO-COST + N POINTS TRAILING SL
    # ============================================
    if sl_type == 'Cost-to-Cost + N Points Trailing SL':
        # Get parameters
        initial_sl_distance = config.get('sl_points', 10)
        K = config.get('ctc_trigger_points', 3)  # Trigger points
        N = config.get('ctc_offset_points', 2)    # Offset points
        
        # Store entry price in config for reference
        if '_ctc_entry_price' not in config:
            config['_ctc_entry_price'] = entry_price
        
        # LONG Position
        if position_type == 'LONG':
            # Phase 1: Before trigger (normal trailing)
            points_in_favor = current_price - entry_price
            
            if points_in_favor < K:
                # Phase 1: Normal trailing
                new_sl = current_price - initial_sl_distance
                return max(current_sl, new_sl)
            
            # Phase 2: Triggered - lock SL at entry + N
            elif points_in_favor >= K and points_in_favor < initial_sl_distance:
                # Lock at entry + N
                locked_sl = entry_price + N
                return max(current_sl, locked_sl)
            
            # Phase 3: Breakout - resume trailing with reduced distance
            else:  # points_in_favor >= initial_sl_distance
                # New trailing distance = initial_sl_distance - N
                new_trailing_distance = initial_sl_distance - N
                new_sl = current_price - new_trailing_distance
                return max(current_sl, new_sl)
        
        # SHORT Position (mirror reverse)
        else:  # SHORT
            points_in_favor = entry_price - current_price
            
            if points_in_favor < K:
                # Phase 1: Normal trailing
                new_sl = current_price + initial_sl_distance
                return min(current_sl, new_sl)
            
            # Phase 2: Triggered - lock SL at entry - N
            elif points_in_favor >= K and points_in_favor < initial_sl_distance:
                # Lock at entry - N
                locked_sl = entry_price - N
                return min(current_sl, locked_sl)
            
            # Phase 3: Breakout - resume trailing
            else:
                new_trailing_distance = initial_sl_distance - N
                new_sl = current_price + new_trailing_distance
                return min(current_sl, new_sl)
    
    # ============================================
    # TRAILING SL (POINTS)
    # ============================================
    elif sl_type == 'Trailing SL (Points)':
        trail_points = config.get('sl_points', 10)
        
        if position_type == 'LONG':
            new_sl = current_price - trail_points
            return max(current_sl, new_sl)
        else:  # SHORT
            new_sl = current_price + trail_points
            return min(current_sl, new_sl)
    
    # ============================================
    # TRAILING PROFIT (RUPEES)
    # ============================================
    elif sl_type == 'Trailing Profit (Rupees)':
        # Exit if profit drops by X rupees from peak
        quantity = config.get('quantity', 1)
        trail_rupees = config.get('sl_trail_rupees', 50)
        
        # Track highest profit
        if position_type == 'LONG':
            current_profit = (current_price - entry_price) * quantity
        else:  # SHORT
            current_profit = (entry_price - current_price) * quantity
        
        if 'highest_profit' not in position:
            position['highest_profit'] = current_profit
        
        position['highest_profit'] = max(position['highest_profit'], current_profit)
        
        # Check if profit dropped by trail amount
        if position['highest_profit'] - current_profit >= trail_rupees:
            # Exit immediately (set SL at current price)
            return current_price
        
        return current_sl
    
    # ============================================
    # TRAILING LOSS (RUPEES)
    # ============================================
    elif sl_type == 'Trailing Loss (Rupees)':
        # Exit if loss increases by X rupees from lowest
        quantity = config.get('quantity', 1)
        trail_rupees = config.get('sl_trail_rupees', 50)
        
        # Track lowest profit (most negative)
        if position_type == 'LONG':
            current_profit = (current_price - entry_price) * quantity
        else:  # SHORT
            current_profit = (entry_price - current_price) * quantity
        
        if 'lowest_profit' not in position:
            position['lowest_profit'] = current_profit
        
        position['lowest_profit'] = min(position['lowest_profit'], current_profit)
        
        # Check if loss increased by trail amount
        if current_profit - position['lowest_profit'] <= -trail_rupees:
            # Exit immediately
            return current_price
        
        return current_sl
    
    # ============================================
    # TRAILING SL + CURRENT CANDLE
    # ============================================
    elif sl_type == 'Trailing SL + Current Candle':
        if position_type == 'LONG':
            # Trail up to current candle low
            return max(current_sl, current['Low'])
        else:  # SHORT
            # Trail down to current candle high
            return min(current_sl, current['High'])
    
    # ============================================
    # TRAILING SL + PREVIOUS CANDLE
    # ============================================
    elif sl_type == 'Trailing SL + Previous Candle':
        if position_type == 'LONG':
            prev_low = current['Prev_Low']
            if not pd.isna(prev_low):
                return max(current_sl, prev_low)
        else:  # SHORT
            prev_high = current['Prev_High']
            if not pd.isna(prev_high):
                return min(current_sl, prev_high)
        
        return current_sl
    
    # ============================================
    # TRAILING SL + CURRENT SWING
    # ============================================
    elif sl_type == 'Trailing SL + Current Swing':
        if position_type == 'LONG':
            swing_low = current['Swing_Low']
            if not pd.isna(swing_low):
                return max(current_sl, swing_low)
        else:  # SHORT
            swing_high = current['Swing_High']
            if not pd.isna(swing_high):
                return min(current_sl, swing_high)
        
        return current_sl
    
    # ============================================
    # TRAILING SL + PREVIOUS SWING
    # ============================================
    elif sl_type == 'Trailing SL + Previous Swing':
        if position_type == 'LONG':
            prev_swing_low = current['Prev_Swing_Low']
            if not pd.isna(prev_swing_low):
                return max(current_sl, prev_swing_low)
        else:  # SHORT
            prev_swing_high = current['Prev_Swing_High']
            if not pd.isna(prev_swing_high):
                return min(current_sl, prev_swing_high)
        
        return current_sl
    
    # ============================================
    # VOLATILITY-ADJUSTED TRAILING SL
    # ============================================
    elif sl_type == 'Volatility-Adjusted Trailing SL':
        if pd.isna(current['ATR']):
            return current_sl
        
        multiplier = config.get('sl_atr_multiplier', 1.5)
        trail_distance = current['ATR'] * multiplier
        
        if position_type == 'LONG':
            new_sl = current_price - trail_distance
            return max(current_sl, new_sl)
        else:  # SHORT
            new_sl = current_price + trail_distance
            return min(current_sl, new_sl)
    
    # ============================================
    # BREAK-EVEN AFTER 50% TARGET
    # ============================================
    elif sl_type == 'Break-even After 50% Target':
        target_price = position.get('target_price')
        if target_price is None:
            return current_sl
        
        # Calculate halfway point to target
        if position_type == 'LONG':
            halfway = entry_price + (target_price - entry_price) * 0.5
            
            if current_price >= halfway:
                # Move SL to break-even (entry price)
                return max(current_sl, entry_price)
        else:  # SHORT
            halfway = entry_price - (entry_price - target_price) * 0.5
            
            if current_price <= halfway:
                # Move SL to break-even
                return min(current_sl, entry_price)
        
        return current_sl
    
    # No trailing update for other SL types
    return current_sl

def update_trailing_target(position, current_price, df, idx, config):
    """
    Update trailing target
    
    Args:
        position: Current position dict
        current_price: Current market price
        df: DataFrame with indicators
        idx: Current index
        config: Configuration dict
        
    Returns:
        float: Updated target price (or existing if no update)
    """
    target_type = config.get('target_type', 'Custom Points')
    position_type = position['type']
    current_target = position['target_price']
    
    if target_type == 'Trailing Target (Points)':
        trail_points = config.get('target_points', 20)
        
        if position_type == 'LONG':
            # Target trails up
            new_target = current_price + trail_points
            return max(current_target, new_target)
        else:  # SHORT
            # Target trails down
            new_target = current_price - trail_points
            return min(current_target, new_target)
    
    # No trailing update for other target types
    return current_target

# ================================
# BACKTESTING ENGINE
# ================================

def run_backtest(df, config):
    """
    Run backtesting on historical data
    
    Args:
        df: DataFrame with OHLCV and indicators
        config: Configuration dict
        
    Returns:
        tuple: (trades_list, metrics_dict, debug_info)
    """
    trades = []
    position = None
    strategy_name = config.get('strategy', 'EMA Crossover')
    strategy_func = STRATEGY_FUNCTIONS.get(strategy_name, check_ema_crossover_strategy)
    
    # Debug tracking
    total_candles = len(df)
    candles_analyzed = 0
    signals_generated = 0
    trades_entered = 0
    trades_exited = 0
    
    # Start from a reasonable index to ensure indicators are calculated
    start_idx = max(50, config.get('ema_slow', 21) + 10)
    
    for idx in range(start_idx, len(df)):
        candles_analyzed += 1
        current_data = df.iloc[idx]
        current_price = current_data['Close']
        
        if position is None:
            # Check for entry signal
            signal, entry_price = strategy_func(df, idx, config, None)
            
            if signal:
                signals_generated += 1
                
                # Convert signal to position type
                position_type = 'LONG' if signal in ('BUY', 'LONG') else 'SHORT'
                
                # Calculate initial SL and Target
                sl_price = calculate_initial_sl(position_type, entry_price, df, idx, config)
                target_price = calculate_initial_target(position_type, entry_price, df, idx, config)
                
                # Create position
                position = {
                    'type': position_type,
                    'entry_price': entry_price,
                    'entry_time': current_data['Datetime'],
                    'sl_price': sl_price,
                    'target_price': target_price,
                    'quantity': config.get('quantity', 1),
                    'highest_price': entry_price,
                    'lowest_price': entry_price,
                }
                
                trades_entered += 1
        
        else:
            # Update tracking
            position['highest_price'] = max(position['highest_price'], current_price)
            position['lowest_price'] = min(position['lowest_price'], current_price)
            
            # Update trailing SL/Target if applicable
            if position['sl_price'] is not None:
                position['sl_price'] = update_trailing_sl(position, current_price, df, idx, config)
            
            if position['target_price'] is not None:
                position['target_price'] = update_trailing_target(position, current_price, df, idx, config)
            
            # Check exit conditions
            exit_reason = None
            exit_price = current_price
            
            # Check SL hit
            if position['sl_price'] is not None:
                if position['type'] == 'LONG':
                    if current_price <= position['sl_price']:
                        exit_reason = 'SL Hit'
                        exit_price = position['sl_price']
                else:  # SHORT
                    if current_price >= position['sl_price']:
                        exit_reason = 'SL Hit'
                        exit_price = position['sl_price']
            
            # Check Target hit
            if exit_reason is None and position['target_price'] is not None:
                if position['type'] == 'LONG':
                    if current_price >= position['target_price']:
                        exit_reason = 'Target Hit'
                        exit_price = position['target_price']
                else:  # SHORT
                    if current_price <= position['target_price']:
                        exit_reason = 'Target Hit'
                        exit_price = position['target_price']
            
            # Check signal-based exit
            if exit_reason is None and config.get('sl_type') == 'Signal-based (Reverse Crossover)':
                signal, _ = strategy_func(df, idx, config, position)
                if signal:
                    # Reverse signal detected
                    if (position['type'] == 'LONG' and signal in ('SELL', 'SHORT')) or \
                       (position['type'] == 'SHORT' and signal in ('BUY', 'LONG')):
                        exit_reason = 'Signal Exit'
                        exit_price = current_price
            
            # Exit position if conditions met
            if exit_reason:
                # Calculate P&L
                if position['type'] == 'LONG':
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                else:  # SHORT
                    pnl = (position['entry_price'] - exit_price) * position['quantity']
                
                # Record trade
                trade = {
                    'entry_time': position['entry_time'],
                    'exit_time': current_data['Datetime'],
                    'type': position['type'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'sl_price': position['sl_price'],
                    'target_price': position['target_price'],
                    'quantity': position['quantity'],
                    'pnl': pnl,
                    'exit_reason': exit_reason,
                }
                
                trades.append(trade)
                trades_exited += 1
                
                # Clear position
                position = None
    
    # Calculate metrics
    if trades:
        df_trades = pd.DataFrame(trades)
        
        total_trades = len(trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = df_trades['pnl'].sum()
        avg_pnl = df_trades['pnl'].mean()
        
        # Calculate max drawdown
        cumulative_pnl = df_trades['pnl'].cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'max_drawdown': max_drawdown,
        }
    else:
        metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'max_drawdown': 0,
        }
    
    # Debug info
    debug_info = {
        'total_candles': total_candles,
        'candles_analyzed': candles_analyzed,
        'signals_generated': signals_generated,
        'trades_entered': trades_entered,
        'trades_completed': trades_exited,
    }
    
    return trades, metrics, debug_info

# ================================
# LIVE TRADING
# ================================

def add_log(message):
    """Add timestamped log message"""
    timestamp = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    
    if 'live_logs' not in st.session_state:
        st.session_state['live_logs'] = []
    
    st.session_state['live_logs'].append(log_entry)

def live_trading_iteration():
    """
    Single iteration of live trading loop
    
    This function:
    1. Fetches fresh data
    2. Calculates indicators
    3. Checks for entry/exit signals
    4. Manages positions
    5. Places broker orders if enabled
    """
    config = st.session_state.get('config', {})
    
    # Fetch fresh data
    ticker = config.get('asset', 'NIFTY 50')
    interval = INTERVAL_MAPPING.get(config.get('interval', '1 day'), '1d')
    period = PERIOD_MAPPING.get(config.get('period', '1 month'), '1mo')
    
    df = fetch_data(ticker, interval, period, is_live_trading=True)
    
    if df is None or df.empty:
        add_log("❌ Failed to fetch data")
        return
    
    # Calculate indicators
    df = calculate_all_indicators(df, config)
    
    # Get current data
    idx = len(df) - 1
    current_data = df.iloc[idx]
    current_price = current_data['Close']
    
    # Store current data in session
    st.session_state['current_data'] = current_data
    
    # Get strategy function
    strategy_name = config.get('strategy', 'EMA Crossover')
    strategy_func = STRATEGY_FUNCTIONS.get(strategy_name, check_ema_crossover_strategy)
    
    # Get current position
    position = st.session_state.get('position')
    
    if position is None:
        # Check for entry signal
        add_log(f"🔍 Checking for entry signal using {strategy_name}...")
        signal, entry_price = strategy_func(df, idx, config, None)
        
        if signal:
            add_log(f"✅ SIGNAL DETECTED: {signal} at {entry_price:.2f}")
            
            # Convert to position type
            position_type = 'LONG' if signal in ('BUY', 'LONG') else 'SHORT'
            
            # Calculate initial SL and Target
            sl_price = calculate_initial_sl(position_type, entry_price, df, idx, config)
            target_price = calculate_initial_target(position_type, entry_price, df, idx, config)
            
            # Log entry
            add_log(f"📈 ENTERING {position_type} POSITION @ {entry_price:.2f}")
            
            sl_display = f"{sl_price:.2f}" if sl_price is not None else "Not Set"
            target_display = f"{target_price:.2f}" if target_price is not None else "Not Set"
            add_log(f"🛡️ Initial SL: {sl_display} | 🎯 Target: {target_display}")
            
            # Create position
            position = {
                'type': position_type,
                'entry_price': entry_price,
                'entry_time': current_data['Datetime'],
                'sl_price': sl_price,
                'target_price': target_price,
                'quantity': config.get('quantity', 1),
                'highest_price': entry_price,
                'lowest_price': entry_price,
            }
            
            st.session_state['position'] = position
            add_log(f"✅ Position created in session state")
            
            # Place broker order if enabled
            if config.get('dhan_enabled', False):
                add_log(f"🏦 Dhan broker enabled, attempting to place order...")
                dhan_broker = st.session_state.get('dhan_broker')
                if dhan_broker:
                    try:
                        broker_position = dhan_broker.enter_broker_position(signal, entry_price, config, add_log)
                        st.session_state['broker_position'] = broker_position
                    except Exception as e:
                        add_log(f"🏦 ⚠️ Broker order error: {e}")
                        import traceback
                        add_log(f"🏦 Error details: {traceback.format_exc()}")
                else:
                    add_log("🏦 ⚠️ Broker not initialized")
            else:
                add_log("🏦 Dhan broker disabled in config")
        else:
            add_log(f"⏳ No entry signal detected (Price: {current_price:.2f})")
    
    else:
        # Position exists - monitor for exit
        add_log(f"📊 Monitoring {position['type']} position @ {current_price:.2f}")
        
        # Update tracking
        position['highest_price'] = max(position['highest_price'], current_price)
        position['lowest_price'] = min(position['lowest_price'], current_price)
        
        # Calculate current P&L
        if position['type'] == 'LONG':
            current_pnl = (current_price - position['entry_price']) * position['quantity']
        else:  # SHORT
            current_pnl = (position['entry_price'] - current_price) * position['quantity']
        
        add_log(f"💰 Current P&L: ₹{current_pnl:.2f}")
        
        # Update trailing SL/Target
        old_sl = position['sl_price']
        if old_sl is not None:
            position['sl_price'] = update_trailing_sl(position, current_price, df, idx, config)
            
            if position['sl_price'] != old_sl:
                add_log(f"🛡️ SL Updated: {old_sl:.2f} → {position['sl_price']:.2f}")
        
        old_target = position['target_price']
        if old_target is not None:
            position['target_price'] = update_trailing_target(position, current_price, df, idx, config)
            
            if position['target_price'] != old_target:
                add_log(f"🎯 Target Updated: {old_target:.2f} → {position['target_price']:.2f}")
        
        # Check exit conditions
        exit_reason = None
        exit_price = current_price
        
        # Check SL hit
        if position['sl_price'] is not None:
            if position['type'] == 'LONG':
                if current_price <= position['sl_price']:
                    exit_reason = 'SL Hit'
                    exit_price = position['sl_price']
                    add_log(f"🛑 STOP LOSS HIT! Price {current_price:.2f} <= SL {position['sl_price']:.2f}")
            else:  # SHORT
                if current_price >= position['sl_price']:
                    exit_reason = 'SL Hit'
                    exit_price = position['sl_price']
                    add_log(f"🛑 STOP LOSS HIT! Price {current_price:.2f} >= SL {position['sl_price']:.2f}")
        
        # Check Target hit
        if exit_reason is None and position['target_price'] is not None:
            if position['type'] == 'LONG':
                if current_price >= position['target_price']:
                    exit_reason = 'Target Hit'
                    exit_price = position['target_price']
                    add_log(f"🎯 TARGET HIT! Price {current_price:.2f} >= Target {position['target_price']:.2f}")
            else:  # SHORT
                if current_price <= position['target_price']:
                    exit_reason = 'Target Hit'
                    exit_price = position['target_price']
                    add_log(f"🎯 TARGET HIT! Price {current_price:.2f} <= Target {position['target_price']:.2f}")
        
        # Check signal-based exit
        if exit_reason is None and config.get('sl_type') == 'Signal-based (Reverse Crossover)':
            signal, _ = strategy_func(df, idx, config, position)
            if signal:
                if (position['type'] == 'LONG' and signal in ('SELL', 'SHORT')) or \
                   (position['type'] == 'SHORT' and signal in ('BUY', 'LONG')):
                    exit_reason = 'Signal Exit'
                    exit_price = current_price
                    add_log(f"🔄 REVERSE SIGNAL DETECTED: {signal}")
        
        # Exit if conditions met
        if exit_reason:
            # Calculate P&L
            if position['type'] == 'LONG':
                pnl = (exit_price - position['entry_price']) * position['quantity']
            else:  # SHORT
                pnl = (position['entry_price'] - exit_price) * position['quantity']
            
            add_log(f"🚪 EXITING POSITION: {exit_reason} @ {exit_price:.2f}")
            add_log(f"💰 Final P&L: ₹{pnl:.2f}")
            
            # Save to trade history
            trade_record = {
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(pytz.timezone('Asia/Kolkata')),
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'sl_price': position['sl_price'],
                'target_price': position['target_price'],
                'highest_price': position.get('highest_price', exit_price),
                'lowest_price': position.get('lowest_price', exit_price),
                'quantity': position['quantity'],
                'pnl': pnl,
                'exit_reason': exit_reason,
                'price_range': position.get('highest_price', exit_price) - position.get('lowest_price', exit_price)
            }
            
            if 'trade_history' not in st.session_state:
                st.session_state['trade_history'] = []
            st.session_state['trade_history'].append(trade_record)
            add_log(f"📝 Trade saved to history")
            
            # Exit broker position if exists
            if config.get('dhan_enabled', False):
                broker_position = st.session_state.get('broker_position')
                if broker_position:
                    dhan_broker = st.session_state.get('dhan_broker')
                    if dhan_broker:
                        try:
                            exit_info = dhan_broker.exit_broker_position(broker_position, exit_price, exit_reason, add_log)
                            st.session_state['broker_exit'] = exit_info
                        except Exception as e:
                            add_log(f"🏦 ⚠️ Broker exit error: {e}")
            
            add_log("✅ Position closed, session cleared")
            
            # Clear position and session data
            st.session_state['position'] = None
            st.session_state['broker_position'] = None
            if 'current_data' in st.session_state:
                del st.session_state['current_data']
        else:
            add_log(f"⏳ No exit conditions met - holding position")

# ================================
# UI COMPONENTS
# ================================

def render_config_ui():
    """Render configuration sidebar"""
    st.sidebar.header("⚙️ Configuration")
    
    config = {}
    
    # Asset Selection
    config['asset'] = st.sidebar.selectbox("Asset", list(ASSET_MAPPING.keys()), index=0)
    
    # Timeframe
    config['interval'] = st.sidebar.selectbox("Interval", list(INTERVAL_MAPPING.keys()), index=5)
    config['period'] = st.sidebar.selectbox("Period", list(PERIOD_MAPPING.keys()), index=2)
    
    # Quantity
    config['quantity'] = st.sidebar.number_input("Quantity", min_value=1, value=1)
    
    # Strategy Selection
    st.sidebar.subheader("📊 Strategy")
    config['strategy'] = st.sidebar.selectbox("Strategy Type", STRATEGY_LIST, index=0)
    
    # Strategy-specific parameters
    if config['strategy'] == 'EMA Crossover':
        config['ema_fast'] = st.sidebar.number_input("EMA Fast Period", min_value=1, value=9)
        config['ema_slow'] = st.sidebar.number_input("EMA Slow Period", min_value=1, value=21)
        config['ema_min_angle'] = st.sidebar.number_input("Min Angle (ABSOLUTE)", min_value=0.0, value=0.0, step=0.1)
        
        config['ema_entry_filter'] = st.sidebar.selectbox("Entry Filter", EMA_ENTRY_FILTERS, index=0)
        
        if config['ema_entry_filter'] == 'Custom Candle (Points)':
            config['ema_custom_candle_points'] = st.sidebar.number_input("Min Candle Points", min_value=1, value=5)
        elif config['ema_entry_filter'] == 'ATR-based Candle':
            config['ema_atr_multiplier'] = st.sidebar.number_input("ATR Multiplier", min_value=0.1, value=0.3, step=0.1)
        
        config['ema_use_adx'] = st.sidebar.checkbox("Use ADX Filter", value=False)
        if config['ema_use_adx']:
            config['ema_adx_threshold'] = st.sidebar.number_input("ADX Threshold", min_value=1, value=20)
            config['adx_period'] = st.sidebar.number_input("ADX Period", min_value=1, value=14)
    
    elif config['strategy'] == 'Price Crosses Threshold':
        config['price_threshold'] = st.sidebar.number_input("Price Threshold", min_value=0.0, value=25000.0)
    
    elif config['strategy'] == 'Percentage Change':
        config['pct_change_threshold'] = st.sidebar.number_input("% Change Threshold", min_value=0.1, value=2.0, step=0.1)
    
    # Stop Loss Configuration
    st.sidebar.subheader("🛡️ Stop Loss")
    config['sl_type'] = st.sidebar.selectbox("SL Type", SL_TYPES, index=0)
    
    if 'Points' in config['sl_type'] or config['sl_type'] in ['Custom Points', 'ATR-based', 
                                                                'Trailing SL (Points)', 
                                                                'Cost-to-Cost + N Points Trailing SL']:
        config['sl_points'] = st.sidebar.number_input("SL Points", min_value=1, value=10)
    
    if 'Rupees' in config['sl_type'] or config['sl_type'] == 'P&L Based (Rupees)':
        config['sl_rupees'] = st.sidebar.number_input("SL Rupees", min_value=1, value=100)
    
    if 'Trailing Profit' in config['sl_type'] or 'Trailing Loss' in config['sl_type']:
        config['sl_trail_rupees'] = st.sidebar.number_input("Trail Rupees", min_value=1, value=50)
    
    if 'ATR' in config['sl_type']:
        config['sl_atr_multiplier'] = st.sidebar.number_input("SL ATR Multiplier", min_value=0.1, value=1.5, step=0.1)
    
    if config['sl_type'] == 'Cost-to-Cost + N Points Trailing SL':
        config['ctc_trigger_points'] = st.sidebar.number_input("Trigger Points (K)", min_value=1, value=3)
        config['ctc_offset_points'] = st.sidebar.number_input("Offset Points (N)", min_value=1, value=2)
    
    # Target Configuration
    st.sidebar.subheader("🎯 Target")
    config['target_type'] = st.sidebar.selectbox("Target Type", TARGET_TYPES, index=0)
    
    if 'Points' in config['target_type'] or config['target_type'] in ['Custom Points', 'Trailing Target (Points)']:
        config['target_points'] = st.sidebar.number_input("Target Points", min_value=1, value=20)
    
    if config['target_type'] == 'P&L Based (Rupees)':
        config['target_rupees'] = st.sidebar.number_input("Target Rupees", min_value=1, value=200)
    
    if config['target_type'] == 'Risk-Reward Based':
        config['risk_reward_ratio'] = st.sidebar.number_input("Risk:Reward Ratio", min_value=0.1, value=2.0, step=0.1)
    
    if config['target_type'] == 'ATR-based':
        config['target_atr_multiplier'] = st.sidebar.number_input("Target ATR Multiplier", min_value=0.1, value=2.0, step=0.1)
    
    # Dhan Broker Configuration
    st.sidebar.subheader("🏦 Dhan Broker (Optional)")
    config['dhan_enabled'] = st.sidebar.checkbox("Enable Dhan Broker", value=True)
    
    if config['dhan_enabled']:
        config['dhan_client_id'] = st.sidebar.text_input("Client ID", value="1104779876")
        config['dhan_access_token'] = st.sidebar.text_input("Access Token", type="password", value="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzcxMTQzMjM5LCJpYXQiOjE3NzEwNTY4MzksInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTA0Nzc5ODc2In0.qP8kVXDQt-sFa6LWJqd1MRTPESHCCPCqHzEnjsFI2WVbNdywKHXgAKHxVpuH6tP_AJTdqowv9nbqf-2NcGibbQ")
        
        config['dhan_is_options'] = st.sidebar.checkbox("Is Options", value=True)
        
        if config['dhan_is_options']:
            config['dhan_ce_security_id'] = st.sidebar.text_input("CE Security ID", value="48228")
            config['dhan_pe_security_id'] = st.sidebar.text_input("PE Security ID", value="48229")
            config['dhan_strike_price'] = st.sidebar.number_input("Strike Price", min_value=0, value=25000)
            config['dhan_expiry_date'] = st.sidebar.date_input("Expiry Date", value=datetime.now().date())
            config['dhan_quantity'] = st.sidebar.number_input("Dhan Quantity", min_value=1, value=65)
    
    return config

def render_backtest_ui(config):
    """Render backtesting interface"""
    st.header("📈 Backtest Results")
    
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            # Fetch data
            ticker = config.get('asset', 'NIFTY 50')
            interval = INTERVAL_MAPPING.get(config.get('interval', '1 day'), '1d')
            period = PERIOD_MAPPING.get(config.get('period', '1 month'), '1mo')
            
            df = fetch_data(ticker, interval, period)
            
            if df is not None:
                # Calculate indicators
                df = calculate_all_indicators(df, config)
                
                # Run backtest
                trades, metrics, debug_info = run_backtest(df, config)
                
                # Store in session
                st.session_state['backtest_results'] = {
                    'trades': trades,
                    'metrics': metrics,
                    'debug_info': debug_info
                }
    
    # Display results
    if 'backtest_results' in st.session_state:
        results = st.session_state['backtest_results']
        metrics = results['metrics']
        trades = results['trades']
        debug_info = results['debug_info']
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", metrics['total_trades'])
        with col2:
            st.metric("Win Rate", f"{metrics['win_rate']:.2f}%")
        with col3:
            st.metric("Total P&L", f"₹{metrics['total_pnl']:.2f}")
        with col4:
            st.metric("Avg Trade", f"₹{metrics['avg_pnl']:.2f}")
        
        col5, col6 = st.columns(2)
        with col5:
            st.metric("Winning Trades", metrics['winning_trades'])
        with col6:
            st.metric("Losing Trades", metrics['losing_trades'])
        
        st.metric("Max Drawdown", f"₹{metrics['max_drawdown']:.2f}")
        
        # Display trades table
        if trades:
            st.subheader("Trade History")
            df_trades = pd.DataFrame(trades)
            st.dataframe(df_trades, use_container_width=True)
        
        # Display debug info if no trades
        if metrics['total_trades'] == 0:
            st.warning("⚠️ No trades generated")
            
            with st.expander("🔍 Debug Information"):
                st.write("**Backtest Statistics:**")
                st.write(f"- Total Candles: {debug_info['total_candles']}")
                st.write(f"- Candles Analyzed: {debug_info['candles_analyzed']}")
                st.write(f"- Signals Generated: {debug_info['signals_generated']}")
                st.write(f"- Trades Entered: {debug_info['trades_entered']}")
                st.write(f"- Trades Completed: {debug_info['trades_completed']}")
                
                st.write("\n**Suggestions:**")
                st.write("- Try reducing the Minimum Angle filter")
                st.write("- Use a longer historical period")
                st.write("- Adjust entry filter settings")
                st.write("- Check if indicators are calculating correctly")

def render_live_trading_ui(config):
    """Render live trading interface with comprehensive information"""
    st.header("🔴 Live Trading")
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start Trading", type="primary"):
            # Clear all session data
            st.session_state['trading_active'] = True
            st.session_state['position'] = None
            st.session_state['broker_position'] = None
            st.session_state['live_logs'] = []
            
            # Initialize trade history if not exists
            if 'trade_history' not in st.session_state:
                st.session_state['trade_history'] = []
            
            if 'current_data' in st.session_state:
                del st.session_state['current_data']
            
            # Store config
            st.session_state['config'] = config
            
            # Log configuration
            add_log("🚀 Trading started - all sessions cleared")
            add_log(f"📋 Strategy: {config.get('strategy', 'N/A')}")
            add_log(f"📋 Asset: {config.get('asset', 'N/A')} | Interval: {config.get('interval', 'N/A')}")
            add_log(f"📋 SL Type: {config.get('sl_type', 'N/A')} | Target Type: {config.get('target_type', 'N/A')}")
            add_log(f"📋 Quantity: {config.get('quantity', 'N/A')}")
            
            # Initialize broker if enabled
            if config.get('dhan_enabled', False):
                add_log("🏦 Initializing Dhan broker...")
                st.session_state['dhan_broker'] = DhanBrokerIntegration(config)
                add_log("🏦 Broker initialization complete")
            else:
                add_log("🏦 Dhan broker disabled")
    
    with col2:
        if st.button("⏹️ Stop Trading"):
            st.session_state['trading_active'] = False
            add_log("⏹️ Trading stopped")
    
    with col3:
        if st.button("❌ Manual Close"):
            position = st.session_state.get('position')
            if position:
                # Fetch current price
                ticker = config.get('asset', 'NIFTY 50')
                interval = INTERVAL_MAPPING.get(config.get('interval', '1 day'), '1d')
                period = PERIOD_MAPPING.get(config.get('period', '1 month'), '1mo')
                
                df = fetch_data(ticker, interval, period, is_live_trading=True)
                if df is not None:
                    df = calculate_all_indicators(df, config)
                    current_price = df.iloc[-1]['Close']
                    
                    # Calculate P&L
                    if position['type'] == 'LONG':
                        pnl = (current_price - position['entry_price']) * position['quantity']
                    else:  # SHORT
                        pnl = (position['entry_price'] - current_price) * position['quantity']
                    
                    add_log(f"EXIT: Manual Close @ {current_price:.2f} | P&L: ₹{pnl:.2f}")
                    
                    # Save to trade history
                    trade_record = {
                        'entry_time': position['entry_time'],
                        'exit_time': datetime.now(pytz.timezone('Asia/Kolkata')),
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'sl_price': position['sl_price'],
                        'target_price': position['target_price'],
                        'highest_price': position.get('highest_price', current_price),
                        'lowest_price': position.get('lowest_price', current_price),
                        'quantity': position['quantity'],
                        'pnl': pnl,
                        'exit_reason': 'Manual Close',
                        'price_range': position.get('highest_price', current_price) - position.get('lowest_price', current_price)
                    }
                    
                    if 'trade_history' not in st.session_state:
                        st.session_state['trade_history'] = []
                    st.session_state['trade_history'].append(trade_record)
                    
                    # Exit broker position if exists
                    if config.get('dhan_enabled', False):
                        broker_position = st.session_state.get('broker_position')
                        if broker_position:
                            dhan_broker = st.session_state.get('dhan_broker')
                            if dhan_broker:
                                exit_info = dhan_broker.exit_broker_position(broker_position, current_price, 
                                                                             'Manual Close', add_log)
                                st.session_state['broker_exit'] = exit_info
                    
                    add_log("✅ Position closed, session cleared")
                    
                    # Clear position
                    st.session_state['position'] = None
                    st.session_state['broker_position'] = None
                    if 'current_data' in st.session_state:
                        del st.session_state['current_data']
            else:
                st.warning("No active position to close")
    
    # Status indicator
    if st.session_state.get('trading_active', False):
        st.success("🟢 Trading Active")
    else:
        st.info("⚪ Trading Inactive")
    
    # Display strategy parameters
    st.subheader("📋 Strategy Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info(f"**Asset:** {config.get('asset', 'N/A')}")
        st.info(f"**Interval:** {config.get('interval', 'N/A')}")
    with col2:
        st.info(f"**Strategy:** {config.get('strategy', 'N/A')}")
        st.info(f"**Quantity:** {config.get('quantity', 'N/A')}")
    with col3:
        st.info(f"**SL Type:** {config.get('sl_type', 'N/A')}")
        st.info(f"**SL Points:** {config.get('sl_points', 'N/A')}")
    with col4:
        st.info(f"**Target Type:** {config.get('target_type', 'N/A')}")
        st.info(f"**Target Points:** {config.get('target_points', 'N/A')}")
    
    # Display EMA parameters if using EMA strategy
    if config.get('strategy') == 'EMA Crossover':
        st.subheader("📊 EMA Strategy Parameters")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info(f"**Fast EMA:** {config.get('ema_fast', 'N/A')}")
        with col2:
            st.info(f"**Slow EMA:** {config.get('ema_slow', 'N/A')}")
        with col3:
            st.info(f"**Min Angle:** {config.get('ema_min_angle', 'N/A')}")
        with col4:
            st.info(f"**Entry Filter:** {config.get('ema_entry_filter', 'N/A')}")
    
    # Display current indicators and prices
    current_data = st.session_state.get('current_data')
    if current_data is not None:
        st.subheader("📈 Current Market Data")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Current Price", f"₹{current_data['Close']:.2f}")
        with col2:
            if not pd.isna(current_data.get('EMA_Fast')):
                st.metric("Fast EMA", f"₹{current_data['EMA_Fast']:.2f}")
            else:
                st.metric("Fast EMA", "N/A")
        with col3:
            if not pd.isna(current_data.get('EMA_Slow')):
                st.metric("Slow EMA", f"₹{current_data['EMA_Slow']:.2f}")
            else:
                st.metric("Slow EMA", "N/A")
        with col4:
            if not pd.isna(current_data.get('EMA_Fast_Angle')):
                st.metric("EMA Angle", f"{current_data['EMA_Fast_Angle']:.2f}°")
            else:
                st.metric("EMA Angle", "N/A")
        with col5:
            # Determine crossover type
            if not pd.isna(current_data.get('EMA_Fast')) and not pd.isna(current_data.get('EMA_Slow')):
                if current_data['EMA_Fast'] > current_data['EMA_Slow']:
                    crossover = "Bullish ⬆️"
                else:
                    crossover = "Bearish ⬇️"
                st.metric("Crossover", crossover)
            else:
                st.metric("Crossover", "N/A")
    
    # Display current position with all details
    st.subheader("📊 Current Position")
    
    position = st.session_state.get('position')
    
    if position:
        # Main position metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Type", position['type'])
        with col2:
            st.metric("Entry Price", f"₹{position['entry_price']:.2f}")
        with col3:
            if current_data is not None:
                current_price = current_data['Close']
                st.metric("Current Price", f"₹{current_price:.2f}")
            else:
                st.metric("Current Price", "N/A")
        with col4:
            sl_display = f"₹{position['sl_price']:.2f}" if position['sl_price'] is not None else "Not Set"
            st.metric("Stop Loss", sl_display)
        with col5:
            target_display = f"₹{position['target_price']:.2f}" if position['target_price'] is not None else "Not Set"
            st.metric("Target", target_display)
        
        # Additional position details
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Quantity", position['quantity'])
        with col2:
            if current_data is not None:
                current_price = current_data['Close']
                if position['type'] == 'LONG':
                    current_pnl = (current_price - position['entry_price']) * position['quantity']
                else:  # SHORT
                    current_pnl = (position['entry_price'] - current_price) * position['quantity']
                
                pnl_color = "normal" if current_pnl >= 0 else "inverse"
                st.metric("Current P&L", f"₹{current_pnl:.2f}", delta=f"₹{current_pnl:.2f}")
            else:
                st.metric("Current P&L", "N/A")
        with col3:
            st.metric("Highest Price", f"₹{position.get('highest_price', 0):.2f}")
        with col4:
            st.metric("Lowest Price", f"₹{position.get('lowest_price', 0):.2f}")
        with col5:
            price_range = position.get('highest_price', 0) - position.get('lowest_price', 0)
            st.metric("Price Range", f"₹{price_range:.2f}")
        
        # Entry time
        st.info(f"**Entry Time:** {position['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.info("No active position")
    
    # Display broker information if enabled
    if config.get('dhan_enabled', False) and st.session_state.get('broker_position'):
        st.subheader("🏦 Broker Position")
        
        broker_pos = st.session_state['broker_position']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Order ID", broker_pos['order_id'])
        with col2:
            st.metric("Option Type", broker_pos['option_type'])
        with col3:
            st.metric("Security ID", broker_pos['security_id'])
        with col4:
            st.metric("Status", broker_pos['status'])
        
        # Display raw API response
        with st.expander("📄 Raw API Response"):
            st.json(broker_pos['raw_response'])
    
    # Display logs
    st.subheader("📝 Trading Logs")
    
    logs = st.session_state.get('live_logs', [])
    if logs:
        # Display in reverse order (latest first)
        log_text = "\n".join(reversed(logs[-50:]))  # Last 50 logs
        st.text_area("", value=log_text, height=300, disabled=True)
    else:
        st.info("No logs yet")
    
    # Auto-refresh and iteration
    if st.session_state.get('trading_active', False):
        # Run iteration
        live_trading_iteration()
        
        # Auto-refresh every 5 seconds
        time.sleep(1.5)
        st.rerun()

def render_trade_logs_ui():
    """Render comprehensive trade history and statistics"""
    st.header("📊 Trade History & Statistics")
    
    # Get trade history
    trade_history = st.session_state.get('trade_history', [])
    
    if not trade_history:
        st.info("No trades recorded yet. Start live trading to see your trade history here.")
        return
    
    # Convert to DataFrame
    df_trades = pd.DataFrame(trade_history)
    
    # Calculate statistics
    total_trades = len(df_trades)
    profit_trades = len(df_trades[df_trades['pnl'] > 0])
    loss_trades = len(df_trades[df_trades['pnl'] < 0])
    breakeven_trades = len(df_trades[df_trades['pnl'] == 0])
    
    total_pnl = df_trades['pnl'].sum()
    avg_pnl = df_trades['pnl'].mean()
    
    if profit_trades > 0:
        avg_profit = df_trades[df_trades['pnl'] > 0]['pnl'].mean()
    else:
        avg_profit = 0
    
    if loss_trades > 0:
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean()
    else:
        avg_loss = 0
    
    accuracy = (profit_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Display statistics
    st.subheader("📈 Overall Statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Trades", total_trades)
    with col2:
        st.metric("Profit Trades", profit_trades, delta=f"{profit_trades}")
    with col3:
        st.metric("Loss Trades", loss_trades, delta=f"-{loss_trades}")
    with col4:
        st.metric("Accuracy", f"{accuracy:.2f}%")
    with col5:
        st.metric("Total P&L", f"₹{total_pnl:.2f}", delta=f"₹{total_pnl:.2f}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg P&L", f"₹{avg_pnl:.2f}")
    with col2:
        st.metric("Avg Profit", f"₹{avg_profit:.2f}")
    with col3:
        st.metric("Avg Loss", f"₹{avg_loss:.2f}")
    with col4:
        if profit_trades > 0 and loss_trades > 0:
            profit_factor = abs(avg_profit / avg_loss)
            st.metric("Profit Factor", f"{profit_factor:.2f}")
        else:
            st.metric("Profit Factor", "N/A")
    
    # Display trade history table
    st.subheader("📋 Detailed Trade History")
    
    # Format the dataframe for display
    display_df = df_trades.copy()
    
    # Format datetime columns
    if 'entry_time' in display_df.columns:
        display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    if 'exit_time' in display_df.columns:
        display_df['exit_time'] = pd.to_datetime(display_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Format numeric columns
    numeric_cols = ['entry_price', 'exit_price', 'sl_price', 'target_price', 
                    'highest_price', 'lowest_price', 'pnl', 'price_range']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"₹{x:.2f}" if pd.notna(x) else "N/A")
    
    # Reorder columns for better readability
    column_order = ['entry_time', 'exit_time', 'type', 'entry_price', 'exit_price', 
                    'sl_price', 'target_price', 'highest_price', 'lowest_price', 
                    'price_range', 'quantity', 'pnl', 'exit_reason']
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in display_df.columns]
    display_df = display_df[column_order]
    
    # Rename columns for display
    display_df = display_df.rename(columns={
        'entry_time': 'Entry Time',
        'exit_time': 'Exit Time',
        'type': 'Type',
        'entry_price': 'Entry Price',
        'exit_price': 'Exit Price',
        'sl_price': 'Stop Loss',
        'target_price': 'Target',
        'highest_price': 'Highest Price',
        'lowest_price': 'Lowest Price',
        'price_range': 'Price Range',
        'quantity': 'Quantity',
        'pnl': 'P&L',
        'exit_reason': 'Exit Reason'
    })
    
    # Display table
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Add download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Trade History (CSV)",
        data=csv,
        file_name=f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Display P&L chart
    st.subheader("📊 P&L Chart")
    
    # Calculate cumulative P&L
    df_trades_chart = df_trades.copy()
    df_trades_chart['cumulative_pnl'] = df_trades_chart['pnl'].cumsum()
    df_trades_chart['trade_number'] = range(1, len(df_trades_chart) + 1)
    
    # Create chart data
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Add cumulative P&L line
    fig.add_trace(go.Scatter(
        x=df_trades_chart['trade_number'],
        y=df_trades_chart['cumulative_pnl'],
        mode='lines+markers',
        name='Cumulative P&L',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Add individual trade P&L as bars
    colors = ['green' if pnl > 0 else 'red' for pnl in df_trades_chart['pnl']]
    fig.add_trace(go.Bar(
        x=df_trades_chart['trade_number'],
        y=df_trades_chart['pnl'],
        name='Trade P&L',
        marker=dict(color=colors),
        opacity=0.6
    ))
    
    fig.update_layout(
        title='Trade P&L Analysis',
        xaxis_title='Trade Number',
        yaxis_title='P&L (₹)',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trade type distribution
    st.subheader("📊 Trade Type Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Count by type
        type_counts = df_trades['type'].value_counts()
        
        fig_type = go.Figure(data=[go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            hole=0.4
        )])
        
        fig_type.update_layout(
            title='Long vs Short Trades',
            height=300
        )
        
        st.plotly_chart(fig_type, use_container_width=True)
    
    with col2:
        # Count by exit reason
        reason_counts = df_trades['exit_reason'].value_counts()
        
        fig_reason = go.Figure(data=[go.Pie(
            labels=reason_counts.index,
            values=reason_counts.values,
            hole=0.4
        )])
        
        fig_reason.update_layout(
            title='Exit Reason Distribution',
            height=300
        )
        
        st.plotly_chart(fig_reason, use_container_width=True)

# ================================
# MAIN APP
# ================================

def main():
    """Main application"""
    st.set_page_config(
        page_title="Algorithmic Trading System",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Algorithmic Trading System")
    st.markdown("---")
    
    # Render configuration
    config = render_config_ui()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📈 Backtest", "🔴 Live Trading", "📊 Trade History"])
    
    with tab1:
        render_backtest_ui(config)
    
    with tab2:
        render_live_trading_ui(config)
    
    with tab3:
        render_trade_logs_ui()

if __name__ == "__main__":
    main()
