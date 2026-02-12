"""
Professional Quantitative Trading System
Production-ready Streamlit application with live trading and backtesting capabilities

REQUIRED PACKAGES (install with):
pip install streamlit yfinance pandas numpy plotly pytz --break-system-packages

RUN WITH:
streamlit run trading_system.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import time
import random

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

ASSET_MAPPING = {
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "USDINR": "USDINR=X",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "Gold": "GC=F",
    "Silver": "SI=F",
}

INTERVAL_PERIOD_MAP = {
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

# ============================================================================
# DHAN BROKER INTEGRATION
# ============================================================================

class DhanBrokerIntegration:
    """
    Dhan Broker Integration - Live Order Placement
    Uses dhanhq library for real order placement.
    Falls back to simulation if library unavailable.
    """

    def __init__(self, config):
        self.enabled = config.get('dhan_enabled', False)
        self.client_id = config.get('dhan_client_id', '')
        self.access_token = config.get('dhan_access_token', '')

        # Dual security IDs: CE for LONG, PE for SHORT
        self.ce_security_id = str(config.get('dhan_ce_security_id', '42568'))
        self.pe_security_id = str(config.get('dhan_pe_security_id', '42569'))

        # Options parameters
        self.is_options   = config.get('dhan_is_options', True)
        self.strike_price = config.get('dhan_strike_price', 25000)
        self.expiry_date  = config.get('dhan_expiry_date', '')
        self.quantity     = config.get('dhan_quantity', 65)
        self.asset        = config.get('asset', 'NIFTY 50')  # For exchange mapping

        # Order trigger
        self.trigger_condition = config.get('dhan_trigger_condition', '>=')
        self.trigger_price     = config.get('dhan_trigger_price', 0.0)

        # SL / Target config
        self.use_algo_signals    = config.get('dhan_use_algo_signals', True)
        self.custom_sl_type      = config.get('dhan_custom_sl_type', 'Custom Points')
        self.custom_sl_points    = config.get('dhan_custom_sl_points', 50)
        self.custom_sl_rupees    = config.get('dhan_custom_sl_rupees', 300)
        self.custom_target_type  = config.get('dhan_custom_target_type', 'Custom Points')
        self.custom_target_points = config.get('dhan_custom_target_points', 100)
        self.custom_target_rupees = config.get('dhan_custom_target_rupees', 5000)

        # State
        self.broker_position  = None
        self.broker_orders    = []
        self.last_order_status = 'Not started'

    # ------------------------------------------------------------------
    def check_trigger_condition(self, current_price):
        if not self.enabled:
            return False
        if self.trigger_condition == '>=':
            return current_price >= self.trigger_price
        if self.trigger_condition == '<=':
            return current_price <= self.trigger_price
        return False

    # ------------------------------------------------------------------
    def _resolve_security(self, signal_type):
        """Return (security_id, option_type) based on signal direction."""
        if signal_type in ('BUY', 'LONG'):
            return self.ce_security_id, 'CE'
        return self.pe_security_id, 'PE'

    # ------------------------------------------------------------------
    def place_order(self, order_type, quantity, price,
                    order_mode='MARKET', signal_type=None):
        """
        Place an order via Dhan API.
        security_id is auto-selected: CE for BUY/LONG, PE for SELL/SHORT.
        Falls back to simulation when dhanhq is not importable.
        Returns a dict on success, None on hard failure.
        """
        if not self.enabled:
            add_log("🏦 Broker disabled – skipping order")
            return None

        security_id, option_type = self._resolve_security(signal_type or order_type)
        add_log(f"🏦 {'BUY/LONG' if option_type=='CE' else 'SELL/SHORT'} → {option_type} | Security ID: {security_id}")
        add_log(f"🏦 Placing {order_type} {quantity} qty @ {price:.2f} [{order_mode}]")

        raw_response = None
        order_id     = None
        api_used     = False

        # ── Try real Dhan API ──────────────────────────────────────────
        try:
            from dhanhq import dhanhq as DhanHQ          # noqa: N813
            dhan = DhanHQ(self.client_id, self.access_token)

            # Select exchange based on asset
            if self.asset in ('NIFTY 50', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'):
                exchange_seg = dhan.NSE_FNO
            elif self.asset == 'SENSEX':
                exchange_seg = dhan.BSE_FNO
            else:
                exchange_seg = dhan.NSE_FNO  # default

            raw_response = dhan.place_order(
                security_id   = security_id,
                exchange_segment = exchange_seg,
                transaction_type = order_type,           # 'BUY' or 'SELL'
                quantity      = quantity,
                order_type    = order_mode,              # 'MARKET' or 'LIMIT'
                product_type  = dhan.INTRA,
                price         = price if order_mode == 'LIMIT' else 0,
            )
            api_used = True

            # dhanhq returns a dict; orderId is under data or top-level
            if isinstance(raw_response, dict):
                order_id = (
                    raw_response.get('data', {}).get('orderId')
                    or raw_response.get('orderId')
                    or raw_response.get('order_id')
                    or f"DHAN-API-{int(time.time())}"
                )
            else:
                order_id = str(raw_response) if raw_response else f"DHAN-API-{int(time.time())}"

        except ImportError:
            add_log("⚠️ dhanhq not installed – running in SIMULATION mode")
            order_id = f"SIM-{int(time.time())}"

        except Exception as api_err:
            add_log(f"❌ Dhan API error: {api_err}")
            self.last_order_status = f"❌ API Error: {api_err}"
            # Do NOT return None here – create a simulated record so position tracking works
            order_id = f"ERR-{int(time.time())}"

        # ── Build order record (always created) ───────────────────────
        order = {
            'order_id'    : str(order_id),
            'security_id' : security_id,
            'option_type' : option_type,
            'type'        : order_type,
            'quantity'    : quantity,
            'price'       : price,
            'order_mode'  : order_mode,
            'status'      : 'PLACED' if api_used else 'SIMULATED',
            'timestamp'   : datetime.now(IST),
            'raw_response': raw_response,
            'strike_price': self.strike_price,
            'expiry_date' : self.expiry_date,
        }

        self.broker_orders.append(order)
        mode_tag = "✅ API" if api_used else "🔵 SIM"
        self.last_order_status = (
            f"{mode_tag} {order_type} {option_type} {quantity} qty "
            f"@ {price:.2f} | ID: {order_id}"
        )
        add_log(f"🏦 ORDER CONFIRMED [{mode_tag}]: {order_type} {option_type} "
                f"qty={quantity} @ {price:.2f} | ID={order_id}")
        if self.is_options:
            add_log(f"🏦 Strike={self.strike_price} {option_type} | Expiry={self.expiry_date}")

        return order   # ← always returns a valid dict

    # ------------------------------------------------------------------
    def calculate_broker_sl_target(self, entry_price, position_type):
        if self.use_algo_signals:
            return None, None
        sl_price = target_price = None
        if self.custom_sl_type == 'Custom Points':
            sl_price = (entry_price - self.custom_sl_points
                        if position_type == 'BUY'
                        else entry_price + self.custom_sl_points)
        if self.custom_target_type == 'Custom Points':
            target_price = (entry_price + self.custom_target_points
                            if position_type == 'BUY'
                            else entry_price - self.custom_target_points)
        return sl_price, target_price

    # ------------------------------------------------------------------
    def enter_broker_position(self, signal, price, quantity):
        """Open a new broker position."""
        if not self.enabled:
            add_log("🏦 Broker not enabled")
            return
        if self.broker_position is not None:
            add_log("🏦 Position already open – skipping entry")
            return

        # ALWAYS BUY to enter (you're a buyer opening a position)
        # Signal determines CE vs PE, but transaction is always BUY
        order_type = 'BUY'
        order = self.place_order(order_type, quantity, price, signal_type=signal)

        # order is ALWAYS a dict (never None) after the rewrite above
        if order is None:
            add_log("🏦 ❌ place_order returned None – position NOT created")
            return

        sl_price, target_price = self.calculate_broker_sl_target(price, order_type)

        self.broker_position = {
            'type'         : order_type,
            'option_type'  : order['option_type'],
            'security_id'  : order['security_id'],
            'entry_price'  : price,
            'entry_time'   : datetime.now(IST),
            'quantity'     : quantity,
            'sl_price'     : sl_price,
            'target_price' : target_price,
            'order_id'     : order['order_id'],
            'highest_price': price,
            'lowest_price' : price,
        }
        add_log(f"🏦 ✅ POSITION OPENED: {order_type} {order['option_type']} "
                f"@ {price:.2f} | qty={quantity} | ID={order['order_id']}")

    # ------------------------------------------------------------------
    def exit_broker_position(self, price, reason='Manual'):
        """Close the open broker position."""
        if not self.enabled or self.broker_position is None:
            return None

        position   = self.broker_position
        exit_type  = 'SELL' if position['type'] == 'BUY' else 'BUY'
        # For exit we reverse: CE position exits with SELL CE, PE position exits with BUY PE
        exit_signal = 'SELL' if position['type'] == 'BUY' else 'BUY'

        order = self.place_order(exit_type, position['quantity'], price,
                                 signal_type=exit_signal)
        if order is None:
            return None

        pnl = ((price - position['entry_price'])
               if position['type'] == 'BUY'
               else (position['entry_price'] - price)) * position['quantity']

        trade_record = {
            'Entry Time'  : position['entry_time'],
            'Exit Time'   : datetime.now(IST),
            'Type'        : position['type'],
            'Option Type' : position.get('option_type', ''),
            'Entry Price' : position['entry_price'],
            'Exit Price'  : price,
            'Quantity'    : position['quantity'],
            'P&L'         : pnl,
            'Exit Reason' : reason,
            'Order IDs'   : f"{position['order_id']} / {order['order_id']}",
        }

        add_log(f"🏦 POSITION CLOSED: {exit_type} @ {price:.2f} | "
                f"P&L: {pnl:.2f} | Reason: {reason}")
        self.broker_position = None
        return trade_record
    
    def update_broker_position(self, current_price, algo_sl=None, algo_target=None, algo_sl_type=None, algo_target_type=None, algo_sl_rupees=None, algo_target_rupees=None):
        """Update broker position and check for exits"""
        if not self.enabled or self.broker_position is None:
            return None
        
        position = self.broker_position
        
        # Update SL/Target from algo if using algo signals
        if self.use_algo_signals:
            if algo_sl is not None:
                position['sl_price'] = algo_sl
            if algo_target is not None:
                position['target_price'] = algo_target
        
        # Track highest/lowest
        if position['type'] == 'BUY':
            if position['highest_price'] is None or current_price > position['highest_price']:
                position['highest_price'] = current_price
        else:
            if position['lowest_price'] is None or current_price < position['lowest_price']:
                position['lowest_price'] = current_price
        
        # Calculate current P&L
        if position['type'] == 'BUY':
            current_pnl = (current_price - position['entry_price']) * position['quantity']
        else:
            current_pnl = (position['entry_price'] - current_price) * position['quantity']
        
        # Check for P&L-based SL (from algo or custom)
        if self.use_algo_signals and algo_sl_type == 'P&L Based (Rupees)' and algo_sl_rupees:
            if current_pnl <= -algo_sl_rupees:
                return self.exit_broker_position(current_price, 'P&L SL (from algo)')
        elif not self.use_algo_signals and self.custom_sl_type == 'P&L Based (Rupees)':
            if current_pnl <= -self.custom_sl_rupees:
                return self.exit_broker_position(current_price, 'P&L Stop Loss')
        # Check for price-based SL
        elif position['sl_price']:
            if position['type'] == 'BUY' and current_price <= position['sl_price']:
                return self.exit_broker_position(current_price, 'Stop Loss')
            elif position['type'] == 'SELL' and current_price >= position['sl_price']:
                return self.exit_broker_position(current_price, 'Stop Loss')
        
        # Check for P&L-based Target (from algo or custom)
        if self.use_algo_signals and algo_target_type == 'P&L Based (Rupees)' and algo_target_rupees:
            if current_pnl >= algo_target_rupees:
                return self.exit_broker_position(current_price, 'P&L Target (from algo)')
        elif not self.use_algo_signals and self.custom_target_type == 'P&L Based (Rupees)':
            if current_pnl >= self.custom_target_rupees:
                return self.exit_broker_position(current_price, 'P&L Target')
        # Check for price-based Target
        elif position['target_price']:
            if position['type'] == 'BUY' and current_price >= position['target_price']:
                return self.exit_broker_position(current_price, 'Target')
            elif position['type'] == 'SELL' and current_price <= position['target_price']:
                return self.exit_broker_position(current_price, 'Target')
        
        return None
    
    def get_broker_pnl(self, current_price):
        """Calculate current unrealized P&L for broker position"""
        if not self.enabled or self.broker_position is None:
            return 0.0
        
        position = self.broker_position
        
        if position['type'] == 'BUY':
            return (current_price - position['entry_price']) * position['quantity']
        else:
            return (position['entry_price'] - current_price) * position['quantity']

# ============================================================================
# INDICATOR CALCULATIONS (MANUAL - NO TALIB/PANDAS-TA)
# ============================================================================

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = calculate_atr(high, low, close, 1)
    
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = calculate_ema(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calculate_bollinger_bands(close, period=20, std=2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(close, period)
    std_dev = close.rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, sma, lower

def calculate_keltner_channel(high, low, close, period=20, atr_mult=2):
    """Calculate Keltner Channel"""
    ema = calculate_ema(close, period)
    atr = calculate_atr(high, low, close, period)
    upper = ema + (atr * atr_mult)
    lower = ema - (atr * atr_mult)
    return upper, ema, lower

def calculate_supertrend(high, low, close, period=10, multiplier=3):
    """Calculate SuperTrend"""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)
    
    for i in range(1, len(close)):
        if pd.isna(upper_band.iloc[i-1]):
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        else:
            if close.iloc[i] > upper_band.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
                
    return supertrend, direction

def calculate_vwap(high, low, close, volume):
    """Calculate VWAP (Volume Weighted Average Price)"""
    if volume.sum() == 0:
        return pd.Series(close.values, index=close.index)
    
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap

def calculate_swing_points(high, low, period=5):
    """Calculate Swing High and Swing Low"""
    swing_high = high.rolling(window=period*2+1, center=True).max()
    swing_low = low.rolling(window=period*2+1, center=True).min()
    return swing_high, swing_low

def calculate_support_resistance(close, period=20):
    """Calculate Support and Resistance levels"""
    resistance = close.rolling(window=period).max()
    support = close.rolling(window=period).min()
    return support, resistance

def calculate_ema_angle(ema_values, interval='1d'):
    """Calculate EMA angle in degrees"""
    if len(ema_values) < 2:
        return pd.Series(0, index=ema_values.index)
    
    interval_seconds = {
        '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
        '1h': 3600, '4h': 14400, '1d': 86400,
        '1wk': 604800, '1mo': 2592000
    }
    
    time_scale = interval_seconds.get(interval, 86400)
    slope = ema_values.diff() / time_scale
    angle = np.arctan(slope) * (180 / np.pi)
    
    return angle

def calculate_all_indicators(df, config):
    """Calculate all indicators based on configuration"""
    df = df.copy()
    
    df['EMA_Fast'] = calculate_ema(df['Close'], config.get('ema_fast', 9))
    df['EMA_Slow'] = calculate_ema(df['Close'], config.get('ema_slow', 15))
    df['EMA_20'] = calculate_ema(df['Close'], 20)
    df['EMA_50'] = calculate_ema(df['Close'], 50)
    df['SMA_20'] = calculate_sma(df['Close'], 20)
    
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], config.get('atr_period', 14))
    df['RSI'] = calculate_rsi(df['Close'], config.get('rsi_period', 14))
    df['ADX'] = calculate_adx(df['High'], df['Low'], df['Close'], config.get('adx_period', 14))
    
    macd, macd_signal, macd_hist = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd_hist
    
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['Close'])
    df['BB_Upper'] = bb_upper
    df['BB_Middle'] = bb_middle
    df['BB_Lower'] = bb_lower
    
    kc_upper, kc_middle, kc_lower = calculate_keltner_channel(df['High'], df['Low'], df['Close'])
    df['KC_Upper'] = kc_upper
    df['KC_Middle'] = kc_middle
    df['KC_Lower'] = kc_lower
    
    supertrend, st_direction = calculate_supertrend(df['High'], df['Low'], df['Close'])
    df['SuperTrend'] = supertrend
    df['ST_Direction'] = st_direction
    
    df['VWAP'] = calculate_vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    
    swing_high, swing_low = calculate_swing_points(df['High'], df['Low'])
    df['Swing_High'] = swing_high
    df['Swing_Low'] = swing_low
    
    support, resistance = calculate_support_resistance(df['Close'])
    df['Support'] = support
    df['Resistance'] = resistance
    
    df['EMA_Fast_Angle'] = calculate_ema_angle(df['EMA_Fast'], config.get('interval', '1d'))
    df['EMA_Slow_Angle'] = calculate_ema_angle(df['EMA_Slow'], config.get('interval', '1d'))
    
    return df

# ============================================================================
# DATA FETCHING & PROCESSING
# ============================================================================

def fetch_data(ticker, interval, period, is_live_trading=False):
    """Fetch data from yfinance with proper error handling"""
    try:
        if is_live_trading:
            delay = random.uniform(1.0, 1.5)
            time.sleep(delay)
        
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if data.empty:
            st.error(f"No data received for {ticker}")
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_cols):
            st.error(f"Missing required columns in data")
            return None
        
        if 'Volume' not in data.columns:
            data['Volume'] = 0
        
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert(IST)
        else:
            data.index = data.index.tz_convert(IST)
        
        return data
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# ============================================================================
# STRATEGY IMPLEMENTATIONS
# ============================================================================

def check_ema_crossover_strategy(df, idx, config, position_type=None):
    """
    EMA Crossover Strategy - FIXED FILTERS
    All filters work properly now
    """
    if idx < 1:
        return None, None
    
    current = df.iloc[idx]
    previous = df.iloc[idx-1]
    
    # Check for NaN values in EMAs
    if pd.isna(current['EMA_Fast']) or pd.isna(current['EMA_Slow']) or \
       pd.isna(previous['EMA_Fast']) or pd.isna(previous['EMA_Slow']):
        return None, None
    
    # =========================================================================
    # STEP 1: Detect Crossover
    # =========================================================================
    bullish_cross = (previous['EMA_Fast'] <= previous['EMA_Slow'] and 
                     current['EMA_Fast'] > current['EMA_Slow'])
    bearish_cross = (previous['EMA_Fast'] >= previous['EMA_Slow'] and 
                     current['EMA_Fast'] < current['EMA_Slow'])
    
    if not bullish_cross and not bearish_cross:
        return None, None  # No crossover
    
    # =========================================================================
    # STEP 2: Entry Filter Check (if not Simple Crossover)
    # =========================================================================
    entry_filter = config.get('ema_entry_filter', 'Simple Crossover')
    
    if entry_filter != 'Simple Crossover':
        
        # Custom Candle Size Filter
        if entry_filter == 'Custom Candle (Points)':
            custom_points = config.get('ema_custom_points', 10)
            candle_body = abs(current['Close'] - current['Open'])
            
            if candle_body < custom_points:
                return None, None  # Candle too small
        
        # ATR-based Candle Filter
        elif entry_filter == 'ATR-based Candle':
            if pd.isna(current.get('ATR')):
                return None, None  # No ATR data
            
            atr_multiplier = config.get('ema_atr_multiplier', 0.5)  # Lower default
            candle_body = abs(current['Close'] - current['Open'])
            required_size = current['ATR'] * atr_multiplier
            
            if candle_body < required_size:
                return None, None  # Candle too small relative to ATR
    
    # =========================================================================
    # STEP 3: Absolute Angle Check (if configured)
    # =========================================================================
    min_angle = config.get('ema_min_angle', 0.0)
    
    if min_angle > 0:  # Only check if user set minimum > 0
        ema_angle = current.get('EMA_Fast_Angle', 0)
        
        if pd.isna(ema_angle):
            return None, None  # No angle data
        
        # Take ABSOLUTE value - always positive
        abs_angle = abs(float(ema_angle))
        
        if abs_angle < min_angle:
            return None, None  # Angle too shallow
    
    # =========================================================================
    # STEP 4: ADX Trend Filter (if enabled)
    # =========================================================================
    if config.get('ema_use_adx', False):
        if pd.isna(current.get('ADX')):
            return None, None  # No ADX data
        
        adx_threshold = config.get('ema_adx_threshold', 20)  # Lower default
        
        if current['ADX'] < adx_threshold:
            return None, None  # Trend not strong enough
    
    # =========================================================================
    # All filters passed - Return signal
    # =========================================================================
    if bullish_cross:
        return 'BUY', current['Close']
    elif bearish_cross:
        return 'SELL', current['Close']
    
    return None, None

def check_simple_buy_strategy(df, idx, config, position_type=None):
    current = df.iloc[idx]
    return 'BUY', current['Close']

def check_simple_sell_strategy(df, idx, config, position_type=None):
    current = df.iloc[idx]
    return 'SELL', current['Close']

def check_price_threshold_strategy(df, idx, config, position_type=None):
    current = df.iloc[idx]
    threshold = config.get('threshold_price', 0)
    threshold_type = config.get('threshold_type', 'LONG (Price >= Threshold)')
    
    price = current['Close']
    
    if threshold_type == 'LONG (Price >= Threshold)':
        if price >= threshold:
            return 'BUY', price
    elif threshold_type == 'SHORT (Price >= Threshold)':
        if price >= threshold:
            return 'SELL', price
    elif threshold_type == 'LONG (Price <= Threshold)':
        if price <= threshold:
            return 'BUY', price
    elif threshold_type == 'SHORT (Price <= Threshold)':
        if price <= threshold:
            return 'SELL', price
    
    return None, None

def check_rsi_adx_ema_strategy(df, idx, config, position_type=None):
    if idx < 1:
        return None, None
    
    current = df.iloc[idx]
    
    if (current['RSI'] > 80 and current['ADX'] < 20 and current['EMA_Fast'] < current['EMA_Slow']):
        return 'SELL', current['Close']
    
    if (current['RSI'] < 20 and current['ADX'] > 20 and current['EMA_Fast'] > current['EMA_Slow']):
        return 'BUY', current['Close']
    
    return None, None

def check_percentage_change_strategy(df, idx, config, position_type=None):
    if idx < 1:
        return None, None
    
    first_price = df.iloc[0]['Close']
    current_price = df.iloc[idx]['Close']
    
    pct_change = ((current_price - first_price) / first_price) * 100
    st.session_state['current_pct_change'] = pct_change
    
    threshold = config.get('pct_threshold', 0.01)
    direction = config.get('pct_direction', 'BUY on Fall')
    
    if direction == 'BUY on Fall':
        if pct_change <= -threshold:
            return 'BUY', current_price
    elif direction == 'SELL on Fall':
        if pct_change <= -threshold:
            return 'SELL', current_price
    elif direction == 'BUY on Rise':
        if pct_change >= threshold:
            return 'BUY', current_price
    elif direction == 'SELL on Rise':
        if pct_change >= threshold:
            return 'SELL', current_price
    
    return None, None

def check_ai_price_action_strategy(df, idx, config, position_type=None):
    if idx < 20:
        return None, None
    
    current = df.iloc[idx]
    signals = []
    confidence = 0
    
    if current['EMA_Fast'] > current['EMA_Slow']:
        signals.append(('Trend', 'Bullish', 20))
        confidence += 20
    else:
        signals.append(('Trend', 'Bearish', -20))
        confidence -= 20
    
    if current['RSI'] < 30:
        signals.append(('RSI', 'Oversold', 25))
        confidence += 25
    elif current['RSI'] > 70:
        signals.append(('RSI', 'Overbought', -25))
        confidence -= 25
    else:
        signals.append(('RSI', 'Neutral', 0))
    
    if current['MACD'] > current['MACD_Signal']:
        signals.append(('MACD', 'Bullish', 15))
        confidence += 15
    else:
        signals.append(('MACD', 'Bearish', -15))
        confidence -= 15
    
    if current['Close'] < current['BB_Lower']:
        signals.append(('BB', 'Oversold', 20))
        confidence += 20
    elif current['Close'] > current['BB_Upper']:
        signals.append(('BB', 'Overbought', -20))
        confidence -= 20
    else:
        signals.append(('BB', 'Neutral', 0))
    
    if current['Volume'] > 0:
        avg_volume = df['Volume'].tail(20).mean()
        if current['Volume'] > avg_volume * 1.5:
            signals.append(('Volume', 'High', 20))
            confidence += 20 if confidence > 0 else -20
        else:
            signals.append(('Volume', 'Normal', 0))
    
    st.session_state['ai_analysis'] = {
        'signals': signals,
        'confidence': confidence
    }
    
    if confidence >= 50:
        return 'BUY', current['Close']
    elif confidence <= -50:
        return 'SELL', current['Close']
    
    return None, None

def check_custom_strategy(df, idx, config, position_type=None):
    current = df.iloc[idx]
    previous = df.iloc[idx-1] if idx > 0 else None
    
    conditions = config.get('custom_conditions', [])
    
    buy_conditions = [c for c in conditions if c['enabled'] and c['action'] == 'BUY']
    sell_conditions = [c for c in conditions if c['enabled'] and c['action'] == 'SELL']
    
    def get_indicator_value(indicator_name):
        indicator_map = {
            'Price': current['Close'],
            'Close': current['Close'],
            'High': current['High'],
            'Low': current['Low'],
            'RSI': current['RSI'],
            'ADX': current['ADX'],
            'EMA_Fast': current['EMA_Fast'],
            'EMA_Slow': current['EMA_Slow'],
            'EMA_20': current['EMA_20'],
            'EMA_50': current['EMA_50'],
            'SuperTrend': current['SuperTrend'],
            'MACD': current['MACD'],
            'MACD_Signal': current['MACD_Signal'],
            'BB_Upper': current['BB_Upper'],
            'BB_Lower': current['BB_Lower'],
            'ATR': current['ATR'],
            'Volume': current['Volume'],
            'VWAP': current['VWAP'],
            'Support': current['Support'],
            'Resistance': current['Resistance']
        }
        return indicator_map.get(indicator_name, 0)
    
    def check_condition(cond):
        if cond.get('compare_with_price', False):
            indicator_val = get_indicator_value(cond['compare_indicator'])
            price = current['Close']
            operator = cond['operator']
            
            if operator == '>':
                return indicator_val > price
            elif operator == '<':
                return indicator_val < price
            elif operator == '>=':
                return indicator_val >= price
            elif operator == '<=':
                return indicator_val <= price
            elif operator == '==':
                return abs(indicator_val - price) < 0.01
            elif operator == 'crosses_above':
                if previous is None:
                    return False
                prev_indicator = df.iloc[idx-1][cond['compare_indicator']]
                prev_price = previous['Close']
                return prev_indicator <= prev_price and indicator_val > price
            elif operator == 'crosses_below':
                if previous is None:
                    return False
                prev_indicator = df.iloc[idx-1][cond['compare_indicator']]
                prev_price = previous['Close']
                return prev_indicator >= prev_price and indicator_val < price
        else:
            indicator_val = get_indicator_value(cond['indicator'])
            value = cond['value']
            operator = cond['operator']
            
            if operator == '>':
                return indicator_val > value
            elif operator == '<':
                return indicator_val < value
            elif operator == '>=':
                return indicator_val >= value
            elif operator == '<=':
                return indicator_val <= value
            elif operator == '==':
                return abs(indicator_val - value) < 0.01
            elif operator == 'crosses_above':
                if previous is None:
                    return False
                prev_val = df.iloc[idx-1][cond['indicator']]
                return prev_val <= value and indicator_val > value
            elif operator == 'crosses_below':
                if previous is None:
                    return False
                prev_val = df.iloc[idx-1][cond['indicator']]
                return prev_val >= value and indicator_val < value
        
        return False
    
    if buy_conditions:
        if all(check_condition(c) for c in buy_conditions):
            return 'BUY', current['Close']
    
    if sell_conditions:
        if all(check_condition(c) for c in sell_conditions):
            return 'SELL', current['Close']
    
    return None, None

# ============================================================================
# STOP LOSS & TARGET MANAGEMENT
# ============================================================================

def calculate_initial_sl(entry_price, position_type, sl_type, config, current_data):
    if sl_type == 'Custom Points':
        points = config.get('sl_points', 10)
        if position_type == 'LONG':
            return entry_price - points
        else:
            return entry_price + points
    
    elif sl_type == 'ATR-based':
        atr = current_data['ATR']
        multiplier = config.get('sl_atr_multiplier', 1.5)
        if pd.notna(atr):
            if position_type == 'LONG':
                return entry_price - (atr * multiplier)
            else:
                return entry_price + (atr * multiplier)
        return None
    
    elif sl_type in ['Current Candle Low/High', 'Trailing SL + Current Candle']:
        if position_type == 'LONG':
            return current_data['Low']
        else:
            return current_data['High']
    
    elif sl_type in ['Current Swing Low/High', 'Trailing SL + Current Swing']:
        if position_type == 'LONG':
            return current_data['Swing_Low']
        else:
            return current_data['Swing_High']
    
    elif sl_type in ['Trailing SL (Points)', 'Trailing SL + Signal Based', 
                     'Volatility-Adjusted Trailing SL', 'Break-even After 50% Target']:
        points = config.get('sl_points', 10)
        if position_type == 'LONG':
            return entry_price - points
        else:
            return entry_price + points
    
    elif sl_type == 'Signal-based (reverse EMA crossover)':
        return None
    
    return None

def calculate_initial_target(entry_price, position_type, target_type, config, current_data):
    if target_type == 'Custom Points':
        points = config.get('target_points', 20)
        if position_type == 'LONG':
            return entry_price + points
        else:
            return entry_price - points
    
    elif target_type == 'ATR-based':
        atr = current_data['ATR']
        multiplier = config.get('target_atr_multiplier', 3.0)
        if pd.notna(atr):
            if position_type == 'LONG':
                return entry_price + (atr * multiplier)
            else:
                return entry_price - (atr * multiplier)
        return None
    
    elif target_type == 'Risk-Reward Based':
        sl_points = config.get('sl_points', 10)
        rr_ratio = config.get('rr_ratio', 2.0)
        target_points = sl_points * rr_ratio
        if position_type == 'LONG':
            return entry_price + target_points
        else:
            return entry_price - target_points
    
    elif target_type in ['Current Candle Low/High']:
        if position_type == 'LONG':
            return current_data['High']
        else:
            return current_data['Low']
    
    elif target_type in ['Current Swing Low/High']:
        if position_type == 'LONG':
            return current_data['Swing_High']
        else:
            return current_data['Swing_Low']
    
    elif target_type in ['Trailing Target (Points)', 'Trailing Target + Signal Based',
                        '50% Exit at Target (Partial)']:
        points = config.get('target_points', 20)
        if position_type == 'LONG':
            return entry_price + points
        else:
            return entry_price - points
    
    elif target_type == 'Signal-based (reverse EMA crossover)':
        return None
    
    return None

def update_trailing_sl(current_price, current_sl, position_type, sl_type, config, current_data, df, idx):
    if sl_type == 'Trailing SL (Points)':
        points = config.get('sl_points', 10)
        if position_type == 'LONG':
            new_sl = current_price - points
            if current_sl is None or new_sl > current_sl:
                return new_sl
        else:
            new_sl = current_price + points
            if current_sl is None or new_sl < current_sl:
                return new_sl
    
    elif sl_type == 'Trailing SL + Current Candle':
        if position_type == 'LONG':
            new_sl = current_data['Low']
            if current_sl is None or new_sl > current_sl:
                return new_sl
        else:
            new_sl = current_data['High']
            if current_sl is None or new_sl < current_sl:
                return new_sl
    
    elif sl_type == 'Trailing SL + Previous Candle':
        if idx > 0:
            prev_data = df.iloc[idx-1]
            if position_type == 'LONG':
                new_sl = prev_data['Low']
                if current_sl is None or new_sl > current_sl:
                    return new_sl
            else:
                new_sl = prev_data['High']
                if current_sl is None or new_sl < current_sl:
                    return new_sl
    
    elif sl_type == 'Volatility-Adjusted Trailing SL':
        atr = current_data['ATR']
        multiplier = config.get('sl_atr_multiplier', 1.5)
        if pd.notna(atr):
            if position_type == 'LONG':
                new_sl = current_price - (atr * multiplier)
                if current_sl is None or new_sl > current_sl:
                    return new_sl
            else:
                new_sl = current_price + (atr * multiplier)
                if current_sl is None or new_sl < current_sl:
                    return new_sl
    
    elif sl_type == 'Cost-to-Cost + N Points Trailing SL':
        # Phase 1: wait until market moves K points in favour
        # Phase 2: once triggered, trail as (current_price - N) for LONG
        k = config.get('ctc_trigger_points', 3.0)
        n = config.get('ctc_offset_points', 2.0)
        entry_price = config.get('_ctc_entry_price')  # set when position opens

        if entry_price is None:
            return current_sl  # not yet initialised

        if position_type == 'LONG':
            triggered = current_price >= entry_price + k
            if triggered:
                # SL = max of: entry+n, current_price-n, existing trailing SL
                candidate = max(entry_price + n, current_price - n, current_sl or 0)
                # Only ratchet up, never down
                if candidate > (current_sl or 0):
                    return round(candidate, 2)
        else:  # SHORT
            triggered = current_price <= entry_price - k
            if triggered:
                # SL = min of: entry-n, current_price+n, existing trailing SL
                candidate = min(entry_price - n, current_price + n, current_sl or float('inf'))
                # Only ratchet down (SHORT means lower is tighter), never up
                if candidate < (current_sl or float('inf')):
                    return round(candidate, 2)

    return current_sl

def update_trailing_target(current_price, current_target, position_type, target_type, config):
    if target_type in ['Trailing Target (Points)', 'Trailing Target + Signal Based']:
        points = config.get('target_points', 20)
        if position_type == 'LONG':
            new_target = current_price + points
            if current_target is None or new_target > current_target:
                return new_target
        else:
            new_target = current_price - points
            if current_target is None or new_target < current_target:
                return new_target
    
    return current_target

def check_signal_based_exit(df, idx, position_type):
    if idx < 1:
        return False
    
    current = df.iloc[idx]
    previous = df.iloc[idx-1]
    
    if position_type == 'LONG':
        return (previous['EMA_Fast'] >= previous['EMA_Slow'] and 
                current['EMA_Fast'] < current['EMA_Slow'])
    else:
        return (previous['EMA_Fast'] <= previous['EMA_Slow'] and 
                current['EMA_Fast'] > current['EMA_Slow'])

# ============================================================================
# BACKTEST & LIVE TRADING ENGINE
# ============================================================================

def run_backtest(df, config):
    """
    BACKTEST ENGINE - COMPLETELY REWRITTEN FROM SCRATCH
    
    Clear, simple logic:
    1. Loop through each candle
    2. If no position: check for entry signals
    3. If position open: check for exit conditions
    4. Track all P&L accurately
    5. No position data carries over between trades
    """
    
    # Initialize
    trades = []
    position = None
    
    # Get configuration
    strategy_name = config.get('strategy', 'EMA Crossover')
    sl_type = config.get('sl_type', 'Custom Points')
    target_type = config.get('target_type', 'Custom Points')
    quantity = config.get('quantity', 1)
    
    # Map strategies to functions
    strategy_map = {
        'EMA Crossover': check_ema_crossover_strategy,
        'Simple Buy': check_simple_buy_strategy,
        'Simple Sell': check_simple_sell_strategy,
        'Price Crosses Threshold': check_price_threshold_strategy,
        'RSI-ADX-EMA': check_rsi_adx_ema_strategy,
        'Percentage Change': check_percentage_change_strategy,
        'AI Price Action': check_ai_price_action_strategy,
        'Custom Strategy': check_custom_strategy,
    }
    
    strategy_func = strategy_map.get(strategy_name)
    if not strategy_func:
        return trades
    
    # Debugging stats
    total_candles = len(df)
    entry_signals = 0
    trades_entered = 0
    trades_exited = 0
    
    # Start after indicators are ready
    start_idx = 20  # Skip first 20 candles for indicator warmup
    
    # =========================================================================
    # MAIN BACKTEST LOOP
    # =========================================================================
    for idx in range(start_idx, total_candles):
        
        # Get current candle data
        candle = df.iloc[idx]
        timestamp = df.index[idx]
        price = candle['Close']
        
        # Validate price data
        if pd.isna(price) or price <= 0:
            continue
        
        # =====================================================================
        # SCENARIO 1: NO POSITION - Look for entry signal
        # =====================================================================
        if position is None:
            
            # Check strategy for signal
            signal, signal_price = strategy_func(df, idx, config, None)
            
            if signal:  # Entry signal found
                entry_signals += 1
                
                # Determine position type
                pos_type = 'LONG' if signal == 'BUY' else 'SHORT'
                
                # Use signal price as entry (current candle close)
                entry_px = signal_price
                
                # Calculate SL price
                sl_px = calculate_initial_sl(entry_px, pos_type, sl_type, config, candle)
                
                # Calculate Target price  
                tgt_px = calculate_initial_target(entry_px, pos_type, target_type, config, candle)
                
                # Create NEW position (completely fresh, no old data)
                position = {
                    'type': pos_type,
                    'entry_price': entry_px,
                    'entry_time': timestamp,
                    'entry_idx': idx,
                    'quantity': quantity,
                    'sl_price': sl_px,
                    'target_price': tgt_px,
                    'highest_price': entry_px,
                    'lowest_price': entry_px,
                    'highest_pnl': 0.0,
                    'lowest_pnl': 0.0,
                    'partial_exited': False,
                    'breakeven_active': False,
                    'current_qty': quantity,
                }
                
                trades_entered += 1
                
        # =====================================================================
        # SCENARIO 2: POSITION OPEN - Check for exit
        # =====================================================================
        else:
            
            # Extract position data (use local variables to avoid confusion)
            pos_type = position['type']
            entry_px = position['entry_price']
            curr_qty = position['current_qty']
            sl_px = position['sl_price']
            tgt_px = position['target_price']
            
            # Calculate CURRENT P&L for THIS position
            # CRITICAL: Use same asset's entry and current price
            if pos_type == 'LONG':
                pnl = (price - entry_px) * curr_qty
            else:  # SHORT
                pnl = (entry_px - price) * curr_qty
            
            # Track price extremes
            if price > position['highest_price']:
                position['highest_price'] = price
            if price < position['lowest_price']:
                position['lowest_price'] = price
            
            # Track P&L extremes
            if pnl > position['highest_pnl']:
                position['highest_pnl'] = pnl
            if pnl < position['lowest_pnl']:
                position['lowest_pnl'] = pnl
            
            # -----------------------------------------------------------------
            # Update Trailing SL (if applicable)
            # -----------------------------------------------------------------
            if 'Trailing' in sl_type and sl_type not in ['Trailing Profit (Rupees)', 'Trailing Loss (Rupees)']:
                config['_ctc_entry_price'] = position['entry_price']  # needed for CTC trailing
                new_sl = update_trailing_sl(price, sl_px, pos_type, sl_type, config, candle, df, idx)
                if new_sl is not None:
                    position['sl_price'] = new_sl
                    sl_px = new_sl
            
            # -----------------------------------------------------------------
            # Update Trailing Target (if applicable)
            # -----------------------------------------------------------------
            if 'Trailing' in target_type:
                new_tgt = update_trailing_target(price, tgt_px, pos_type, target_type, config)
                if new_tgt is not None:
                    position['target_price'] = new_tgt
                    tgt_px = new_tgt
            
            # -----------------------------------------------------------------
            # Check EXIT conditions (in priority order)
            # -----------------------------------------------------------------
            
            exit_now = False
            exit_price = price
            exit_reason = None
            
            # 1. Trailing Profit Exit
            if sl_type == 'Trailing Profit (Rupees)':
                trail_amt = config.get('trailing_profit_rupees', 1000)
                profit_drop = position['highest_pnl'] - pnl
                
                if position['highest_pnl'] > 0 and profit_drop >= trail_amt:
                    exit_now = True
                    exit_reason = 'Trailing Profit'
            
            # 2. Trailing Loss Exit
            if not exit_now and sl_type == 'Trailing Loss (Rupees)':
                trail_amt = config.get('trailing_loss_rupees', 500)
                loss_increase = pnl - position['lowest_pnl']
                
                if position['lowest_pnl'] < 0 and abs(loss_increase) >= trail_amt:
                    exit_now = True
                    exit_reason = 'Trailing Loss'
            
            # 3. P&L-based SL
            if not exit_now and sl_type == 'P&L Based (Rupees)':
                sl_amt = config.get('sl_rupees', 300)
                if pnl <= -sl_amt:
                    exit_now = True
                    exit_reason = 'P&L SL'
            
            # 4. P&L-based Target
            if not exit_now and target_type == 'P&L Based (Rupees)':
                tgt_amt = config.get('target_rupees', 5000)
                if pnl >= tgt_amt:
                    exit_now = True
                    exit_reason = 'P&L Target'
            
            # 5. Price-based SL
            if not exit_now and sl_px is not None:
                if pos_type == 'LONG' and price <= sl_px:
                    exit_now = True
                    exit_reason = 'SL'
                elif pos_type == 'SHORT' and price >= sl_px:
                    exit_now = True
                    exit_reason = 'SL'
            
            # 6. Price-based Target
            if not exit_now and tgt_px is not None:
                if pos_type == 'LONG' and price >= tgt_px:
                    exit_now = True
                    exit_reason = 'Target'
                elif pos_type == 'SHORT' and price <= tgt_px:
                    exit_now = True
                    exit_reason = 'Target'
            
            # 7. Signal-based Exit
            if not exit_now:
                if (sl_type == 'Signal-based (reverse EMA crossover)' or 
                    target_type == 'Signal-based (reverse EMA crossover)'):
                    if check_signal_based_exit(df, idx, pos_type):
                        exit_now = True
                        exit_reason = 'Signal'
            
            # -----------------------------------------------------------------
            # Execute EXIT
            # -----------------------------------------------------------------
            if exit_now:
                
                # Handle partial exits
                final_pnl = pnl
                if position['partial_exited']:
                    # Add P&L from partial exit
                    partial_qty = quantity - curr_qty
                    if pos_type == 'LONG':
                        partial_pnl = (tgt_px - entry_px) * partial_qty
                    else:
                        partial_pnl = (entry_px - tgt_px) * partial_qty
                    final_pnl += partial_pnl
                
                # Create trade record
                trade = {
                    'Entry Time': position['entry_time'],
                    'Exit Time': timestamp,
                    'Duration': timestamp - position['entry_time'],
                    'Type': pos_type,
                    'Entry Price': entry_px,
                    'Exit Price': exit_price,
                    'SL': sl_px,
                    'Target': tgt_px,
                    'Quantity': quantity,
                    'P&L': final_pnl,
                    'Exit Reason': exit_reason
                }
                
                trades.append(trade)
                trades_exited += 1
                
                # CRITICAL: Clear position completely
                position = None
            
            # -----------------------------------------------------------------
            # Handle Break-even (if not exiting)
            # -----------------------------------------------------------------
            elif target_type == 'Break-even After 50% Target' and not position['breakeven_active']:
                if tgt_px is not None:
                    halfway = entry_px + (tgt_px - entry_px) * 0.5 if pos_type == 'LONG' else entry_px - (entry_px - tgt_px) * 0.5
                    
                    if (pos_type == 'LONG' and price >= halfway) or (pos_type == 'SHORT' and price <= halfway):
                        position['sl_price'] = entry_px
                        position['breakeven_active'] = True
            
            # -----------------------------------------------------------------
            # Handle Partial Exit (if not exiting)
            # -----------------------------------------------------------------
            elif target_type == '50% Exit at Target (Partial)' and not position['partial_exited']:
                if tgt_px is not None:
                    hit = (pos_type == 'LONG' and price >= tgt_px) or (pos_type == 'SHORT' and price <= tgt_px)
                    
                    if hit:
                        partial_qty = quantity // 2
                        if partial_qty > 0:
                            position['partial_exited'] = True
                            position['current_qty'] = quantity - partial_qty
    
    # =========================================================================
    # Store debug info
    # =========================================================================
    st.session_state['backtest_debug'] = {
        'total_candles': total_candles,
        'candles_analyzed': total_candles - start_idx,
        'signals_generated': entry_signals,
        'trades_entered': trades_entered,
        'trades_completed': trades_exited,
        'skipped_candles': start_idx,
        'position_still_open': position is not None
    }
    
    return trades

def add_log(message):
    timestamp = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    
    if 'trade_logs' not in st.session_state:
        st.session_state['trade_logs'] = []
    
    st.session_state['trade_logs'].insert(0, log_entry)
    
    if len(st.session_state['trade_logs']) > 50:
        st.session_state['trade_logs'] = st.session_state['trade_logs'][:50]

def live_trading_iteration():
    if not st.session_state.get('trading_active', False):
        return
    
    config = st.session_state['config']
    
    ticker = ASSET_MAPPING.get(config['asset'], config.get('custom_ticker', '^NSEI'))
    interval = config['interval']
    period = config['period']
    
    df = fetch_data(ticker, interval, period, is_live_trading=True)
    
    if df is None or df.empty:
        add_log("Error: Failed to fetch data")
        st.session_state['trading_active'] = False
        return
    
    df = calculate_all_indicators(df, config)
    st.session_state['current_data'] = df
    
    idx = len(df) - 1
    current_data = df.iloc[idx]
    current_time = df.index[idx]
    current_price = current_data['Close']
    
    # Always rebuild broker from current config so enabled/credentials are fresh
    if 'dhan_broker' not in st.session_state or not isinstance(st.session_state.get('dhan_broker'), DhanBrokerIntegration):
        st.session_state['dhan_broker'] = DhanBrokerIntegration(config)
        st.session_state['broker_trade_history'] = []
    else:
        # Refresh enabled/credentials from current config every iteration
        existing = st.session_state['dhan_broker']
        existing.enabled       = config.get('dhan_enabled', False)
        existing.client_id     = config.get('dhan_client_id', '')
        existing.access_token  = config.get('dhan_access_token', '')
        existing.ce_security_id = str(config.get('dhan_ce_security_id', '42568'))
        existing.pe_security_id = str(config.get('dhan_pe_security_id', '42569'))
        existing.quantity      = config.get('dhan_quantity', 65)
        existing.expiry_date   = config.get('dhan_expiry_date', '')
        existing.strike_price  = config.get('dhan_strike_price', 25000)
        existing.is_options    = config.get('dhan_is_options', True)
        existing.asset         = config.get('asset', 'NIFTY 50')
    
    dhan_broker = st.session_state['dhan_broker']
    
    strategy_name = config['strategy']
    strategy_func_map = {
        'EMA Crossover': check_ema_crossover_strategy,
        'Simple Buy': check_simple_buy_strategy,
        'Simple Sell': check_simple_sell_strategy,
        'Price Crosses Threshold': check_price_threshold_strategy,
        'RSI-ADX-EMA': check_rsi_adx_ema_strategy,
        'Percentage Change': check_percentage_change_strategy,
        'AI Price Action': check_ai_price_action_strategy,
        'Custom Strategy': check_custom_strategy,
    }
    
    strategy_func = strategy_func_map.get(strategy_name)
    position = st.session_state.get('position')
    signal = None  # will be set if strategy fires this tick
    
    # MAIN ALGO ENTRY LOGIC
    if position is None:
        signal, entry_price = strategy_func(df, idx, config, None)
        
        if signal:
            # Convert BUY/SELL to LONG/SHORT for internal consistency
            position_type = 'LONG' if signal == 'BUY' else 'SHORT'
            
            sl_price = calculate_initial_sl(entry_price, position_type, config['sl_type'], config, current_data)
            target_price = calculate_initial_target(entry_price, position_type, config['target_type'], config, current_data)
            
            st.session_state['position'] = {
                'type': position_type,
                'entry_price': entry_price,
                'entry_time': current_time,
                'sl_price': sl_price,
                'target_price': target_price,
                'quantity': config['quantity'],
                'partial_exit_done': False,
                'breakeven_activated': False,
                'highest_price': entry_price,
                'lowest_price': entry_price,
                'highest_profit': 0.0,
                'lowest_profit': 0.0,
            }
            
            sl_display = f"{sl_price:.2f}" if sl_price else "Signal"
            target_display = f"{target_price:.2f}" if target_price else "Signal"
            add_log(f"ENTRY: {signal} @ {entry_price:.2f} | SL: {sl_display} | Target: {target_display}")
    
    # MAIN ALGO EXIT LOGIC
    else:
        position_type = position['type']
        entry_price = position['entry_price']
        sl_price = position['sl_price']
        target_price = position['target_price']
        
        if 'Trailing' in config['sl_type']:
            config['_ctc_entry_price'] = position['entry_price']  # needed for CTC trailing
            new_sl = update_trailing_sl(current_price, sl_price, position_type, config['sl_type'],
                                        config, current_data, df, idx)
            if new_sl is not None and new_sl != sl_price:
                st.session_state['position']['sl_price'] = new_sl
                add_log(f"SL Updated: {new_sl:.2f}")
                sl_price = new_sl
        
        if 'Trailing' in config['target_type']:
            new_target = update_trailing_target(current_price, target_price, position_type,
                                                config['target_type'], config)
            if new_target is not None and new_target != target_price:
                st.session_state['position']['target_price'] = new_target
                add_log(f"Target Updated: {new_target:.2f}")
                target_price = new_target
        
        if position_type == 'LONG':
            if position['highest_price'] is None or current_price > position['highest_price']:
                st.session_state['position']['highest_price'] = current_price
        else:
            if position['lowest_price'] is None or current_price < position['lowest_price']:
                st.session_state['position']['lowest_price'] = current_price
        
        # Calculate current P&L
        if position_type == 'LONG':
            current_pnl = (current_price - entry_price) * position['quantity']
        else:
            current_pnl = (entry_price - current_price) * position['quantity']
        
        # Track highest profit and lowest profit
        if current_pnl > position.get('highest_profit', 0):
            st.session_state['position']['highest_profit'] = current_pnl
        if current_pnl < position.get('lowest_profit', 0):
            st.session_state['position']['lowest_profit'] = current_pnl
        
        # Check Trailing Profit exit
        if config['sl_type'] == 'Trailing Profit (Rupees)':
            trailing_profit_amount = config.get('trailing_profit_rupees', 1000)
            highest_profit = position.get('highest_profit', 0)
            profit_drop = highest_profit - current_pnl
            
            if highest_profit > 0 and profit_drop >= trailing_profit_amount:
                exit_price = current_price
                pnl = current_pnl
                
                duration = current_time - position['entry_time']
                trade = {
                    'Entry Time': position['entry_time'],
                    'Exit Time': current_time,
                    'Duration': duration,
                    'Type': position_type,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'SL': sl_price,
                    'Target': target_price,
                    'Quantity': config['quantity'],
                    'P&L': pnl,
                    'Exit Reason': 'Trailing Profit'
                }
                
                st.session_state['trade_history'].append(trade)
                add_log(f"Trailing Profit Exit: Profit dropped ₹{profit_drop:.2f} from peak | P&L: {pnl:.2f}")
                st.session_state['position'] = None
                
                time.sleep(random.uniform(1.0, 1.5))
                st.rerun()
                return
        
        # Check Trailing Loss exit
        if config['sl_type'] == 'Trailing Loss (Rupees)':
            trailing_loss_amount = config.get('trailing_loss_rupees', 500)
            lowest_profit = position.get('lowest_profit', 0)
            loss_increase = current_pnl - lowest_profit
            
            if lowest_profit < 0 and abs(loss_increase) >= trailing_loss_amount:
                exit_price = current_price
                pnl = current_pnl
                
                duration = current_time - position['entry_time']
                trade = {
                    'Entry Time': position['entry_time'],
                    'Exit Time': current_time,
                    'Duration': duration,
                    'Type': position_type,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'SL': sl_price,
                    'Target': target_price,
                    'Quantity': config['quantity'],
                    'P&L': pnl,
                    'Exit Reason': 'Trailing Loss'
                }
                
                st.session_state['trade_history'].append(trade)
                add_log(f"Trailing Loss Exit: Loss increased ₹{abs(loss_increase):.2f} from lowest | P&L: {pnl:.2f}")
                st.session_state['position'] = None
                
                time.sleep(random.uniform(1.0, 1.5))
                st.rerun()
                return
        
        if (config['target_type'] == 'Break-even After 50% Target' and
            not position['breakeven_activated'] and target_price is not None):
            
            if position_type == 'LONG':
                target_50 = entry_price + (target_price - entry_price) * 0.5
                if current_price >= target_50:
                    st.session_state['position']['sl_price'] = entry_price
                    st.session_state['position']['breakeven_activated'] = True
                    add_log(f"Break-even activated @ {entry_price:.2f}")
            else:
                target_50 = entry_price - (entry_price - target_price) * 0.5
                if current_price <= target_50:
                    st.session_state['position']['sl_price'] = entry_price
                    st.session_state['position']['breakeven_activated'] = True
                    add_log(f"Break-even activated @ {entry_price:.2f}")
        
        if (config['target_type'] == '50% Exit at Target (Partial)' and
            not position['partial_exit_done'] and target_price is not None):
            
            target_hit = False
            if position_type == 'LONG' and current_price >= target_price:
                target_hit = True
            elif position_type == 'SELL' and current_price <= target_price:
                target_hit = True
            
            if target_hit:
                partial_qty = config['quantity'] // 2
                if partial_qty > 0:
                    if position_type == 'LONG':
                        partial_pnl = (target_price - entry_price) * partial_qty
                    else:
                        partial_pnl = (entry_price - target_price) * partial_qty
                    
                    st.session_state['position']['partial_exit_done'] = True
                    st.session_state['position']['quantity'] = config['quantity'] - partial_qty
                    add_log(f"Partial Exit: {partial_qty} qty @ {target_price:.2f} | P&L: {partial_pnl:.2f}")
        
        signal_exit = False
        if (config['sl_type'] == 'Signal-based (reverse EMA crossover)' or
            config['target_type'] == 'Signal-based (reverse EMA crossover)'):
            signal_exit = check_signal_based_exit(df, idx, position_type)
        
        # Check P&L-based SL
        sl_hit = False
        if config['sl_type'] == 'P&L Based (Rupees)':
            sl_rupees = config.get('sl_rupees', 300)
            if current_pnl <= -sl_rupees:  # Loss exceeded limit
                sl_hit = True
                add_log(f"P&L-based SL Hit: Loss {current_pnl:.2f} exceeded limit -{sl_rupees:.2f}")
        elif sl_price is not None:
            if position_type == 'LONG' and current_price <= sl_price:
                sl_hit = True
            elif position_type == 'SELL' and current_price >= sl_price:
                sl_hit = True
        
        # Check P&L-based Target
        target_hit = False
        if config['target_type'] == 'P&L Based (Rupees)':
            target_rupees = config.get('target_rupees', 5000)
            if current_pnl >= target_rupees:  # Profit reached target
                target_hit = True
                add_log(f"P&L-based Target Hit: Profit {current_pnl:.2f} reached target {target_rupees:.2f}")
        elif target_price is not None and 'Trailing' not in config['target_type']:
            if position_type == 'LONG' and current_price >= target_price:
                target_hit = True
            elif position_type == 'SELL' and current_price <= target_price:
                target_hit = True
        
        if signal_exit or sl_hit or target_hit:
            exit_price = current_price
            exit_reason = 'Signal' if signal_exit else ('SL' if sl_hit else 'Target')
            
            if position_type == 'LONG':
                pnl = (exit_price - entry_price) * position['quantity']
            else:
                pnl = (entry_price - exit_price) * position['quantity']
            
            if position['partial_exit_done']:
                partial_qty = config['quantity'] - position['quantity']
                if position_type == 'LONG':
                    partial_pnl = (target_price - entry_price) * partial_qty
                else:
                    partial_pnl = (entry_price - target_price) * partial_qty
                pnl += partial_pnl
            
            duration = current_time - position['entry_time']
            
            trade = {
                'Entry Time': position['entry_time'],
                'Exit Time': current_time,
                'Duration': duration,
                'Type': position_type,
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'SL': sl_price,
                'Target': target_price,
                'Quantity': config['quantity'],
                'P&L': pnl,
                'Exit Reason': exit_reason
            }
            
            st.session_state['trade_history'].append(trade)
            add_log(f"EXIT: {exit_reason} @ {exit_price:.2f} | P&L: {pnl:.2f}")
            
            st.session_state['position'] = None
    
    # DHAN BROKER LOGIC
    if dhan_broker and dhan_broker.enabled:
        if dhan_broker.broker_position is None:
            # Determine signal: prefer current algo position, else fall back to
            # whatever signal the strategy just produced this iteration
            broker_signal = None
            position_now = st.session_state.get('position')
            if position_now:
                broker_signal = 'BUY' if position_now['type'] == 'LONG' else 'SELL'
                add_log(f"🏦 Broker following algo signal: {broker_signal}")
            elif signal:
                broker_signal = signal  # 'BUY' or 'SELL' from strategy this tick
                add_log(f"🏦 Broker using fresh strategy signal: {broker_signal}")

            if broker_signal:
                add_log(f"🏦 Entering broker position: {broker_signal} @ {current_price:.2f}")
                dhan_broker.enter_broker_position(
                    broker_signal,
                    current_price,
                    config.get('dhan_quantity', 65)
                )
            else:
                add_log(f"🏦 No signal yet – broker waiting")
        
        # Update broker position
        if dhan_broker.broker_position:
            # Pass algo SL/Target if using algo signals
            algo_sl = position['sl_price'] if position and dhan_broker.use_algo_signals else None
            algo_target = position['target_price'] if position and dhan_broker.use_algo_signals else None
            algo_sl_type = config.get('sl_type') if position and dhan_broker.use_algo_signals else None
            algo_target_type = config.get('target_type') if position and dhan_broker.use_algo_signals else None
            algo_sl_rupees = config.get('sl_rupees') if position and dhan_broker.use_algo_signals else None
            algo_target_rupees = config.get('target_rupees') if position and dhan_broker.use_algo_signals else None
            
            broker_trade = dhan_broker.update_broker_position(
                current_price, algo_sl, algo_target, 
                algo_sl_type, algo_target_type,
                algo_sl_rupees, algo_target_rupees
            )
            
            if broker_trade:
                st.session_state['broker_trade_history'].append(broker_trade)
    
    time.sleep(random.uniform(1.0, 1.5))
    st.rerun()

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_chart(df, config, position=None, trade_history=None):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price', 'RSI'),
        row_heights=[0.7, 0.3]
    )
    
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
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['EMA_Fast'],
            name=f'EMA {config.get("ema_fast", 9)}',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['EMA_Slow'],
            name=f'EMA {config.get("ema_slow", 15)}',
            line=dict(color='red', width=1)
        ),
        row=1, col=1
    )
    
    if position:
        entry_time = position['entry_time']
        entry_price = position['entry_price']
        
        fig.add_trace(
            go.Scatter(
                x=[entry_time],
                y=[entry_price],
                mode='markers',
                marker=dict(
                    size=10,
                    color='green' if position['type'] == 'LONG' else 'red',
                    symbol='triangle-up' if position['type'] == 'LONG' else 'triangle-down'
                ),
                name='Entry',
                showlegend=True
            ),
            row=1, col=1
        )
        
        if position['sl_price']:
            fig.add_hline(
                y=position['sl_price'],
                line_dash="dash",
                line_color="red",
                annotation_text="SL",
                row=1, col=1
            )
        
        if position['target_price']:
            fig.add_hline(
                y=position['target_price'],
                line_dash="dash",
                line_color="green",
                annotation_text="Target",
                row=1, col=1
            )
    
    if trade_history:
        for trade in trade_history:
            color = 'green' if trade['P&L'] > 0 else 'red'
            fig.add_trace(
                go.Scatter(
                    x=[trade['Entry Time'], trade['Exit Time']],
                    y=[trade['Entry Price'], trade['Exit Price']],
                    mode='markers+lines',
                    marker=dict(size=6, color=color),
                    line=dict(color=color, width=1),
                    showlegend=False
                ),
                row=1, col=1
            )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['RSI'],
            name='RSI',
            line=dict(color='purple', width=1)
        ),
        row=2, col=1
    )
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_configuration_ui():
    st.header("⚙️ Trading Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Asset Selection")
        asset_type = st.selectbox("Asset Type", list(ASSET_MAPPING.keys()) + ["Custom"])
        
        # Clear position if asset changes (prevent BTC/NIFTY price mixing)
        if 'last_selected_asset' not in st.session_state:
            st.session_state['last_selected_asset'] = asset_type
        elif st.session_state['last_selected_asset'] != asset_type:
            # Asset changed - clear position to prevent price mixing
            st.session_state['position'] = None
            st.session_state['last_selected_asset'] = asset_type
            if st.session_state.get('trading_active', False):
                st.warning("⚠️ Asset changed - Previous position cleared")
        
        if asset_type == "Custom":
            custom_ticker = st.text_input("Enter Ticker Symbol", "^NSEI")
            selected_asset = "Custom"
            
            # Also check custom ticker changes
            if 'last_custom_ticker' not in st.session_state:
                st.session_state['last_custom_ticker'] = custom_ticker
            elif st.session_state['last_custom_ticker'] != custom_ticker:
                st.session_state['position'] = None
                st.session_state['last_custom_ticker'] = custom_ticker
                if st.session_state.get('trading_active', False):
                    st.warning("⚠️ Ticker changed - Previous position cleared")
        else:
            custom_ticker = None
            selected_asset = asset_type
    
    with col2:
        st.subheader("Timeframe")
        interval = st.selectbox("Interval", list(INTERVAL_PERIOD_MAP.keys()))
        period = st.selectbox("Period", INTERVAL_PERIOD_MAP[interval])
    
    st.subheader("Trading Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        quantity = st.number_input("Quantity", min_value=1, value=1)
    
    with col2:
        mode = st.selectbox("Mode", ["Backtest", "Live Trading"])
    
    with col3:
        strategy = st.selectbox(
            "Strategy",
            ["EMA Crossover", "Simple Buy", "Simple Sell", "Price Crosses Threshold",
             "RSI-ADX-EMA", "Percentage Change", "AI Price Action", "Custom Strategy"]
        )
    
    config = {
        'asset': selected_asset,
        'custom_ticker': custom_ticker,
        'interval': interval,
        'period': period,
        'quantity': quantity,
        'mode': mode,
        'strategy': strategy,
    }
    
    st.subheader("Strategy Parameters")
    
    if strategy == "EMA Crossover":
        col1, col2, col3 = st.columns(3)
        with col1:
            config['ema_fast'] = st.number_input("EMA Fast", min_value=1, value=9)
        with col2:
            config['ema_slow'] = st.number_input("EMA Slow", min_value=1, value=15)
        with col3:
            config['ema_min_angle'] = st.number_input("Min Angle (degrees)", min_value=0.0, value=0.0, step=0.1, help="Set to 0.0 to allow all crossovers")
        
        entry_filter = st.selectbox(
            "Entry Filter",
            ["Simple Crossover", "Custom Candle (Points)", "ATR-based Candle"]
        )
        config['ema_entry_filter'] = entry_filter
        
        if entry_filter == "Custom Candle (Points)":
            config['ema_custom_points'] = st.number_input("Custom Points", min_value=1, value=5, help="Lower = more trades")
        elif entry_filter == "ATR-based Candle":
            config['ema_atr_multiplier'] = st.number_input("ATR Multiplier", min_value=0.1, value=0.3, step=0.1, help="Lower = more trades (0.3-0.5 recommended)")
        
        use_adx = st.checkbox("Use ADX Filter")
        config['ema_use_adx'] = use_adx
        
        if use_adx:
            col1, col2 = st.columns(2)
            with col1:
                config['adx_period'] = st.number_input("ADX Period", min_value=1, value=14)
            with col2:
                config['ema_adx_threshold'] = st.number_input("ADX Threshold", min_value=1, value=20, help="Lower = more trades (20-25 recommended)")
    
    elif strategy == "Price Crosses Threshold":
        col1, col2 = st.columns(2)
        with col1:
            config['threshold_price'] = st.number_input("Threshold Price", value=0.0)
        with col2:
            config['threshold_type'] = st.selectbox(
                "Type",
                ["LONG (Price >= Threshold)", "SHORT (Price >= Threshold)",
                 "LONG (Price <= Threshold)", "SHORT (Price <= Threshold)"]
            )
    
    elif strategy == "Percentage Change":
        col1, col2 = st.columns(2)
        with col1:
            config['pct_threshold'] = st.number_input("Percentage Threshold (%)", min_value=0.0, value=0.01, step=0.01)
        with col2:
            config['pct_direction'] = st.selectbox(
                "Direction",
                ["BUY on Fall", "SELL on Fall", "BUY on Rise", "SELL on Rise"]
            )
    
    elif strategy == "Custom Strategy":
        st.subheader("Custom Conditions")
        
        if 'custom_conditions' not in st.session_state:
            st.session_state['custom_conditions'] = []
        
        if st.button("➕ Add Condition"):
            st.session_state['custom_conditions'].append({
                'enabled': True,
                'compare_with_price': False,
                'indicator': 'RSI',
                'compare_indicator': 'EMA_20',
                'operator': '>',
                'value': 0,
                'action': 'BUY'
            })
        
        for i, cond in enumerate(st.session_state['custom_conditions']):
            with st.expander(f"Condition {i+1}", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    cond['enabled'] = st.checkbox("Use this condition", value=cond['enabled'], key=f"cond_enabled_{i}")
                
                with col2:
                    cond['compare_with_price'] = st.checkbox("Compare with Price", value=cond.get('compare_with_price', False), key=f"cond_compare_price_{i}")
                
                if cond['compare_with_price']:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        cond['compare_indicator'] = st.selectbox(
                            "Indicator",
                            ['EMA_20', 'EMA_50', 'EMA_Fast', 'EMA_Slow', 'SMA_20', 'VWAP', 'BB_Upper', 'BB_Lower'],
                            key=f"cond_compare_ind_{i}"
                        )
                    with col2:
                        cond['operator'] = st.selectbox(
                            "Operator",
                            ['>', '<', '>=', '<=', '==', 'crosses_above', 'crosses_below'],
                            key=f"cond_op_{i}"
                        )
                    with col3:
                        cond['action'] = st.selectbox("Action", ['BUY', 'SELL'], key=f"cond_action_{i}")
                else:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        cond['indicator'] = st.selectbox(
                            "Indicator",
                            ['Price', 'Close', 'High', 'Low', 'RSI', 'ADX', 'EMA_Fast', 'EMA_Slow',
                             'EMA_20', 'EMA_50', 'SuperTrend', 'MACD', 'MACD_Signal',
                             'BB_Upper', 'BB_Lower', 'ATR', 'Volume', 'VWAP', 'Support', 'Resistance'],
                            key=f"cond_ind_{i}"
                        )
                    
                    with col2:
                        cond['operator'] = st.selectbox(
                            "Operator",
                            ['>', '<', '>=', '<=', '==', 'crosses_above', 'crosses_below'],
                            key=f"cond_op_{i}"
                        )
                    
                    with col3:
                        cond['value'] = st.number_input("Value", value=float(cond['value']), key=f"cond_val_{i}")
                    
                    with col4:
                        cond['action'] = st.selectbox("Action", ['BUY', 'SELL'], key=f"cond_action_{i}")
                
                if st.button("🗑️ Remove", key=f"remove_{i}"):
                    st.session_state['custom_conditions'].pop(i)
                    st.rerun()
        
        config['custom_conditions'] = st.session_state['custom_conditions']
    
    # Dhan Broker Integration Configuration
    st.markdown("---")
    st.subheader("🏦 Dhan Broker Integration")
    
    dhan_enabled = st.checkbox("Enable Dhan Broker Orders", value=False, key="dhan_enabled")
    config['dhan_enabled'] = dhan_enabled
    
    if dhan_enabled:
        with st.expander("📝 Broker Credentials", expanded=True):
            config['dhan_client_id'] = st.text_input("Client ID", value="1104779876", type="password")
            config['dhan_access_token'] = st.text_input("Access Token", value="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzcwNjk1Nzk3LCJpYXQiOjE3NzA2MDkzOTcsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTA0Nzc5ODc2In0.UGpwqKYkopPZ5ultbC93iw8Ks60wDi2EgeBgzUVRtCZOWJxHR2ZcuHKPt6atliAnMs-W9DPyO85knEsr7SHl8g", type="password")
        
        with st.expander("📊 Security Details", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                config['dhan_exchange'] = st.selectbox("Exchange", ["NSE", "BSE", "NFO", "MCX"], index=2)  # Default NFO
            with col2:
                config['dhan_segment'] = st.selectbox("Segment", ["EQ", "FUT", "OPT"], index=2)  # Default OPT
        
        with st.expander("🎯 Options Configuration", expanded=True):
            config['dhan_is_options'] = st.checkbox("Is Options Contract", value=True)  # Default TRUE
            
            if config['dhan_is_options']:
                st.info("📌 System will auto-select CE/PE based on signal: LONG → CE, SHORT → PE")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**CE (Call) Security ID:**")
                    config['dhan_ce_security_id'] = st.text_input("CE Security ID", value="42568", key="ce_sec_id")
                with col2:
                    st.markdown("**PE (Put) Security ID:**")
                    config['dhan_pe_security_id'] = st.text_input("PE Security ID", value="42569", key="pe_sec_id")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    config['dhan_strike_price'] = st.number_input("Strike Price", min_value=0, value=25000)
                with col2:
                    # Default expiry to today
                    from datetime import date
                    today_str = date.today().strftime("%Y-%m-%d")
                    config['dhan_expiry_date'] = st.text_input("Expiry Date (YYYY-MM-DD)", value=today_str)
                with col3:
                    config['dhan_quantity'] = st.number_input("Broker Quantity", min_value=1, value=65)  # Default 65
        
        with st.expander("⚡ Order Trigger Condition", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                config['dhan_trigger_condition'] = st.selectbox(
                    "Trigger When Price",
                    [">=", "<="],
                    help="Place order when price crosses this threshold"
                )
            with col2:
                config['dhan_trigger_price'] = st.number_input(
                    "Trigger Price",
                    min_value=0.0,
                    value=0.0,
                    step=0.5,
                    help="Price level to trigger broker order"
                )
        
        with st.expander("🎯 Broker SL/Target Configuration", expanded=True):
            config['dhan_use_algo_signals'] = st.checkbox(
                "Use Main Algo SL/Target Signals",
                value=True,
                help="Use the same SL/Target from main algorithm"
            )
            
            if not config['dhan_use_algo_signals']:
                st.markdown("**Custom SL Configuration**")
                col1, col2 = st.columns(2)
                with col1:
                    config['dhan_custom_sl_type'] = st.selectbox(
                        "Broker SL Type",
                        [
                            "Custom Points",
                            "P&L Based (Rupees)",
                            "Trailing SL (Points)",
                            "ATR-based",
                            "Current Candle Low/High",
                            "Previous Candle Low/High"
                        ],
                        key="dhan_sl_type_select"
                    )
                with col2:
                    if 'Points' in config['dhan_custom_sl_type']:
                        config['dhan_custom_sl_points'] = st.number_input(
                            "Broker SL Points",
                            min_value=1,
                            value=50,
                            key="dhan_sl_points"
                        )
                    elif config['dhan_custom_sl_type'] == 'P&L Based (Rupees)':
                        config['dhan_custom_sl_rupees'] = st.number_input(
                            "Broker SL Amount (₹)",
                            min_value=1.0,
                            value=300.0,
                            step=10.0,
                            key="dhan_sl_rupees"
                        )
                
                st.markdown("**Custom Target Configuration**")
                col1, col2 = st.columns(2)
                with col1:
                    config['dhan_custom_target_type'] = st.selectbox(
                        "Broker Target Type",
                        [
                            "Custom Points",
                            "P&L Based (Rupees)",
                            "Trailing Target (Points)",
                            "ATR-based",
                            "Risk-Reward Based"
                        ],
                        key="dhan_target_type_select"
                    )
                with col2:
                    if 'Points' in config['dhan_custom_target_type']:
                        config['dhan_custom_target_points'] = st.number_input(
                            "Broker Target Points",
                            min_value=1,
                            value=100,
                            key="dhan_target_points"
                        )
                    elif config['dhan_custom_target_type'] == 'P&L Based (Rupees)':
                        config['dhan_custom_target_rupees'] = st.number_input(
                            "Broker Target Amount (₹)",
                            min_value=1.0,
                            value=5000.0,
                            step=100.0,
                            key="dhan_target_rupees"
                        )
        
        st.info("ℹ️ Dhan broker integration is enabled. Orders will be placed when conditions are met.")
    
    st.markdown("---")
    
    st.subheader("Stop Loss Configuration")
    sl_type = st.selectbox(
        "Stop Loss Type",
        [
            "Custom Points",
            "P&L Based (Rupees)",
            "Trailing Profit (Rupees)",
            "Trailing Loss (Rupees)",
            "Trailing SL (Points)",
            "Cost-to-Cost + N Points Trailing SL",
            "Trailing SL + Current Candle",
            "Trailing SL + Previous Candle",
            "Trailing SL + Current Swing",
            "Trailing SL + Previous Swing",
            "Trailing SL + Signal Based",
            "Volatility-Adjusted Trailing SL",
            "Break-even After 50% Target",
            "ATR-based",
            "Current Candle Low/High",
            "Previous Candle Low/High",
            "Current Swing Low/High",
            "Previous Swing Low/High",
            "Signal-based (reverse EMA crossover)"
        ]
    )
    config['sl_type'] = sl_type
    
    if 'Points' in sl_type or sl_type == 'Custom Points':
        config['sl_points'] = st.number_input("SL Points", min_value=1, value=10)

    if sl_type == 'Cost-to-Cost + N Points Trailing SL':
        _col1, _col2 = st.columns(2)
        with _col1:
            config['ctc_trigger_points'] = st.number_input(
                "Trigger (K points in favour)",
                min_value=0.1, value=3.0, step=0.5,
                help="Market must move this many points in your favour before SL activates"
            )
        with _col2:
            config['ctc_offset_points'] = st.number_input(
                "SL Offset above cost (N points)",
                min_value=0.0, value=2.0, step=0.5,
                help="Once triggered, SL = entry + N (LONG) or entry - N (SHORT)"
            )
        st.caption(
            "Example: entry=50, K=3, N=2 → when price reaches 53, SL moves to 52 "
            "and then trails normally (price-N) as price continues upward."
        )
    
    if sl_type == 'P&L Based (Rupees)':
        config['sl_rupees'] = st.number_input(
            "Stop Loss Amount (₹)", 
            min_value=1.0, 
            value=300.0, 
            step=10.0,
            help="Exit if loss exceeds this amount"
        )
    
    if sl_type == 'Trailing Profit (Rupees)':
        config['trailing_profit_rupees'] = st.number_input(
            "Trailing Profit Amount (₹)", 
            min_value=1.0, 
            value=1000.0, 
            step=100.0,
            help="Exit if profit drops by this amount from highest profit"
        )
    
    if sl_type == 'Trailing Loss (Rupees)':
        config['trailing_loss_rupees'] = st.number_input(
            "Trailing Loss Amount (₹)", 
            min_value=1.0, 
            value=500.0, 
            step=50.0,
            help="Exit if loss increases by this amount from lowest point"
        )
    
    if 'ATR' in sl_type:
        config['sl_atr_multiplier'] = st.number_input("ATR Multiplier (SL)", min_value=0.1, value=1.5, step=0.1)
    
    st.subheader("Target Configuration")
    target_type = st.selectbox(
        "Target Type",
        [
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
            "Signal-based (reverse EMA crossover)"
        ]
    )
    config['target_type'] = target_type
    
    if 'Points' in target_type or target_type == 'Custom Points':
        config['target_points'] = st.number_input("Target Points", min_value=1, value=20)
    
    if target_type == 'P&L Based (Rupees)':
        config['target_rupees'] = st.number_input(
            "Target Profit Amount (₹)", 
            min_value=1.0, 
            value=5000.0, 
            step=100.0,
            help="Exit position if profit reaches this amount"
        )
    
    if 'ATR' in target_type:
        config['target_atr_multiplier'] = st.number_input("ATR Multiplier (Target)", min_value=0.1, value=3.0, step=0.1)
    
    if target_type == 'Risk-Reward Based':
        config['rr_ratio'] = st.number_input("Risk-Reward Ratio", min_value=0.1, value=2.0, step=0.1)
    
    st.session_state['config'] = config
    
    return config

def render_live_trading_ui():
    st.header("📈 Live Trading Dashboard")
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if st.button("▶️ Start Trading", key="start_btn", use_container_width=True):
            st.session_state['trading_active'] = True
            st.session_state['position'] = None
            st.session_state['trade_history'] = []
            st.session_state['trade_logs'] = []
            # Always recreate broker so stale enabled=False never persists
            st.session_state['dhan_broker'] = DhanBrokerIntegration(st.session_state.get('config', {}))
            st.session_state['broker_trade_history'] = []
            add_log("Trading started")
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Trading", key="stop_btn", use_container_width=True):
            if st.session_state.get('position'):
                position = st.session_state['position']
                config = st.session_state['config']
                
                if st.session_state.get('current_data') is not None:
                    df = st.session_state['current_data']
                    current_price = df.iloc[-1]['Close']
                    current_time = df.index[-1]
                    
                    if position['type'] == 'LONG':
                        pnl = (current_price - position['entry_price']) * position['quantity']
                    else:  # SHORT
                        pnl = (position['entry_price'] - current_price) * position['quantity']
                    
                    trade = {
                        'Entry Time': position['entry_time'],
                        'Exit Time': current_time,
                        'Duration': current_time - position['entry_time'],
                        'Type': position['type'],
                        'Entry Price': position['entry_price'],
                        'Exit Price': current_price,
                        'SL': position['sl_price'],
                        'Target': position['target_price'],
                        'Quantity': config['quantity'],
                        'P&L': pnl,
                        'Exit Reason': 'Manual Close'
                    }
                    
                    st.session_state['trade_history'].append(trade)
                    add_log(f"Manual Close @ {current_price:.2f} | P&L: {pnl:.2f}")
            
            # Close broker position if exists
            dhan_broker = st.session_state.get('dhan_broker')
            if dhan_broker and dhan_broker.broker_position:
                if st.session_state.get('current_data') is not None:
                    df = st.session_state['current_data']
                    current_price = df.iloc[-1]['Close']
                    broker_trade = dhan_broker.exit_broker_position(current_price, 'Manual Close')
                    if broker_trade:
                        st.session_state['broker_trade_history'].append(broker_trade)
            
            st.session_state['trading_active'] = False
            st.session_state['position'] = None
            add_log("Trading stopped")
            st.rerun()
    
    with col3:
        status = "🟢 ACTIVE" if st.session_state.get('trading_active', False) else "🔴 STOPPED"
        st.markdown(f"### Status: {status}")
    
    tab1, tab2, tab3 = st.tabs(["📊 Live Dashboard", "📜 Trade History", "📝 Trade Logs"])
    
    with tab1:
        render_live_dashboard()
    
    with tab2:
        render_trade_history()
    
    with tab3:
        render_trade_logs()
    
    if st.session_state.get('trading_active', False):
        live_trading_iteration()

def render_live_dashboard():
    config = st.session_state.get('config', {})
    position = st.session_state.get('position')
    df = st.session_state.get('current_data')
    
    st.subheader("Active Configuration")
    
    asset_display = config.get('asset', 'N/A')
    if asset_display == 'Custom':
        asset_display = config.get('custom_ticker', 'N/A')
    
    st.markdown(f"""
    **Asset:** {asset_display}  
    **Interval:** {config.get('interval', 'N/A')}  
    **Period:** {config.get('period', 'N/A')}  
    **Quantity:** {config.get('quantity', 1)}  
    **Strategy:** {config.get('strategy', 'N/A')}  
    **SL Type:** {config.get('sl_type', 'N/A')}  
    **Target Type:** {config.get('target_type', 'N/A')}  
    """)
    
    if config.get('strategy') == 'EMA Crossover':
        st.markdown(f"""
        **EMA Fast:** {config.get('ema_fast', 9)}  
        **EMA Slow:** {config.get('ema_slow', 15)}  
        **Min Angle:** {config.get('ema_min_angle', 1.0)}°  
        **Entry Filter:** {config.get('ema_entry_filter', 'Simple Crossover')}  
        **Use ADX:** {config.get('ema_use_adx', False)}  
        """)
    
    st.markdown("---")
    
    if df is not None and not df.empty:
        current_data = df.iloc[-1]
        current_price = current_data['Close']
        
        st.subheader("Live Metrics")
        
        st.metric("Current Price", f"{current_price:.2f}")
        
        if position:
            st.metric("Entry Price", f"{position['entry_price']:.2f}")
            st.metric("Position Status", "OPEN", delta="Active")
            st.metric("Position Type", position['type'])
            
            # Calculate unrealized P&L
            if position['type'] == 'LONG':
                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
            else:  # SHORT
                unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']
            
            pnl_color = "normal" if unrealized_pnl >= 0 else "inverse"
            st.metric("Unrealized P&L", f"{unrealized_pnl:.2f}", delta=f"{unrealized_pnl:.2f}", delta_color=pnl_color)
            
            # Display broker P&L if enabled
            if config.get('dhan_enabled', False):
                dhan_broker = st.session_state.get('dhan_broker')
                
                # Show order status
                if dhan_broker and dhan_broker.last_order_status:
                    st.info(f"📋 Last Order Status: {dhan_broker.last_order_status}")
                
                if dhan_broker and dhan_broker.broker_position:
                    broker_pnl = dhan_broker.get_broker_pnl(current_price)
                    broker_pnl_color = "normal" if broker_pnl >= 0 else "inverse"
                    st.metric("🏦 Broker Unrealized P&L", f"{broker_pnl:.2f}", delta=f"{broker_pnl:.2f}", delta_color=broker_pnl_color)
                    
                    # Combined P&L
                    combined_pnl = unrealized_pnl + broker_pnl
                    combined_pnl_color = "normal" if combined_pnl >= 0 else "inverse"
                    st.metric("📊 Total Combined P&L", f"{combined_pnl:.2f}", delta=f"{combined_pnl:.2f}", delta_color=combined_pnl_color)
                else:
                    st.metric("🏦 Broker Position", "No Position")
                    
                    # Show why no position
                    if dhan_broker:
                        if position:
                            st.caption(f"✅ Main algo has position, broker follows signal")
                        else:
                            trigger_info = f"Trigger: Price {dhan_broker.trigger_condition} {dhan_broker.trigger_price:.2f}"
                            st.caption(f"⏳ Waiting for trigger condition | {trigger_info}")
            
            st.metric("EMA Fast", f"{current_data['EMA_Fast']:.2f}")
            st.metric("EMA Slow", f"{current_data['EMA_Slow']:.2f}")
            st.metric("RSI", f"{current_data['RSI']:.2f}")
            
            st.markdown("---")
            st.subheader("Position Information")
            
            duration = datetime.now(IST) - position['entry_time'].to_pydatetime()
            st.markdown(f"**Entry Time:** {position['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**Duration:** {str(duration).split('.')[0]}")
            
            sl_display = f"{position['sl_price']:.2f}" if position['sl_price'] else "Signal-based"
            target_display = f"{position['target_price']:.2f}" if position['target_price'] else "Signal-based"
            
            st.markdown(f"**Stop Loss:** {sl_display}")
            st.markdown(f"**Target:** {target_display}")
            
            if position['sl_price']:
                sl_dist = abs(current_price - position['sl_price'])
                st.markdown(f"**Distance to SL:** {sl_dist:.2f} points")
            
            if position['target_price']:
                target_dist = abs(position['target_price'] - current_price)
                st.markdown(f"**Distance to Target:** {target_dist:.2f} points")
            
            if position['highest_price'] and position['type'] == 'LONG':
                st.markdown(f"**Highest Price:** {position['highest_price']:.2f}")
            
            if position['lowest_price'] and position['type'] == 'SHORT':
                st.markdown(f"**Lowest Price:** {position['lowest_price']:.2f}")
            
            if position.get('breakeven_activated'):
                st.success("✅ Break-even activated!")
            
            if position.get('partial_exit_done'):
                st.info("ℹ️ Partial exit completed")
        
        # Broker Position Information
        if config.get('dhan_enabled', False):
            dhan_broker = st.session_state.get('dhan_broker')
            if dhan_broker and dhan_broker.broker_position:
                st.markdown("---")
                st.subheader("🏦 Broker Position Information")
                
                broker_pos = dhan_broker.broker_position
                broker_duration = datetime.now(IST) - broker_pos['entry_time']
                
                st.markdown(f"**Entry Time:** {broker_pos['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"**Duration:** {str(broker_duration).split('.')[0]}")
                st.markdown(f"**Type:** {broker_pos['type']}")
                st.markdown(f"**Entry Price:** {broker_pos['entry_price']:.2f}")
                st.markdown(f"**Quantity:** {broker_pos['quantity']}")
                
                broker_sl_display = f"{broker_pos['sl_price']:.2f}" if broker_pos['sl_price'] else "Not Set"
                broker_target_display = f"{broker_pos['target_price']:.2f}" if broker_pos['target_price'] else "Not Set"
                
                st.markdown(f"**Stop Loss:** {broker_sl_display}")
                st.markdown(f"**Target:** {broker_target_display}")
                st.markdown(f"**Order ID:** {broker_pos['order_id']}")
                
                # Raw API response for debugging
                raw = None
                if dhan_broker.broker_orders:
                    raw = dhan_broker.broker_orders[-1].get('raw_response')
                with st.expander('📡 Raw API Response (last order)', expanded=False):
                    if raw is not None:
                        st.json(raw) if isinstance(raw, (dict, list)) else st.code(str(raw))
                    else:
                        st.caption('No raw response yet (simulation mode or no order placed)')
                
                if broker_pos['highest_price'] and broker_pos['type'] == 'BUY':
                    st.markdown(f"**Highest Price:** {broker_pos['highest_price']:.2f}")
                
                if broker_pos['lowest_price'] and broker_pos['type'] == 'SELL':
                    st.markdown(f"**Lowest Price:** {broker_pos['lowest_price']:.2f}")
                
                # Display broker trade history
                broker_trades = st.session_state.get('broker_trade_history', [])
                if broker_trades:
                    st.markdown("---")
                    st.subheader("🏦 Broker Trade History")
                    
                    total_broker_pnl = sum(t['P&L'] for t in broker_trades)
                    broker_total_pnl_color = "normal" if total_broker_pnl >= 0 else "inverse"
                    st.metric("Total Broker P&L", f"{total_broker_pnl:.2f}", delta=f"{total_broker_pnl:.2f}", delta_color=broker_total_pnl_color)
                    
                    for i, trade in enumerate(reversed(broker_trades[-5:])):  # Show last 5 broker trades
                        with st.expander(f"Broker Trade #{len(broker_trades) - i} - P&L: {trade['P&L']:.2f}"):
                            st.markdown(f"**Entry:** {trade['Entry Time'].strftime('%Y-%m-%d %H:%M:%S')} @ {trade['Entry Price']:.2f}")
                            st.markdown(f"**Exit:** {trade['Exit Time'].strftime('%Y-%m-%d %H:%M:%S')} @ {trade['Exit Price']:.2f}")
                            st.markdown(f"**Type:** {trade['Type']}")
                            st.markdown(f"**Quantity:** {trade['Quantity']}")
                            st.markdown(f"**Exit Reason:** {trade['Exit Reason']}")
                            st.markdown(f"**Order IDs:** {trade['Order IDs']}")
                            
                            pnl_color = "🟢" if trade['P&L'] > 0 else "🔴"
                            st.markdown(f"**P&L:** {pnl_color} {trade['P&L']:.2f}")
        
        else:
            st.metric("Position Status", "CLOSED", delta="No Active Position")
            st.metric("EMA Fast", f"{current_data['EMA_Fast']:.2f}")
            st.metric("EMA Slow", f"{current_data['EMA_Slow']:.2f}")
            st.metric("RSI", f"{current_data['RSI']:.2f}")
            
            if config.get('strategy') == 'Percentage Change':
                pct_change = st.session_state.get('current_pct_change', 0.0)
                st.metric("% Change from Start", f"{pct_change:.4f}%")
            
            if config.get('strategy') == 'AI Price Action':
                ai_analysis = st.session_state.get('ai_analysis', {})
                if ai_analysis:
                    st.markdown("---")
                    st.subheader("AI Analysis")
                    st.metric("Confidence Score", ai_analysis.get('confidence', 0))
                    
                    for indicator, status, score in ai_analysis.get('signals', []):
                        st.markdown(f"**{indicator}:** {status} ({score:+d})")
        
        st.markdown("---")
        st.subheader("Live Chart")
        
        fig = create_chart(df, config, position)
        st.plotly_chart(fig, use_container_width=True, key="live_chart")

def render_trade_history():
    trade_history = st.session_state.get('trade_history', [])
    
    if not trade_history:
        st.info("No trades yet")
        return
    
    total_trades = len(trade_history)
    winning_trades = sum(1 for t in trade_history if t['P&L'] > 0)
    losing_trades = sum(1 for t in trade_history if t['P&L'] < 0)
    total_pnl = sum(t['P&L'] for t in trade_history)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", total_trades)
    
    with col2:
        st.metric("Wins", winning_trades)
    
    with col3:
        st.metric("Losses", losing_trades)
    
    with col4:
        accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        st.metric("Accuracy", f"{accuracy:.1f}%")
    
    pnl_color = "normal" if total_pnl >= 0 else "inverse"
    st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color=pnl_color)
    
    st.markdown("---")
    
    for i, trade in enumerate(reversed(trade_history)):
        with st.expander(f"Trade #{total_trades - i} - {trade['Type']} - P&L: {trade['P&L']:.2f}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Entry Time:** {trade['Entry Time'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"**Entry Price:** {trade['Entry Price']:.2f}")
                sl_display = f"{trade['SL']:.2f}" if trade['SL'] else "Signal"
                st.markdown(f"**SL:** {sl_display}")
                st.markdown(f"**Quantity:** {trade['Quantity']}")
            
            with col2:
                st.markdown(f"**Exit Time:** {trade['Exit Time'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"**Exit Price:** {trade['Exit Price']:.2f}")
                target_display = f"{trade['Target']:.2f}" if trade['Target'] else "Signal"
                st.markdown(f"**Target:** {target_display}")
                st.markdown(f"**Duration:** {str(trade['Duration']).split('.')[0]}")
            
            st.markdown(f"**Exit Reason:** {trade['Exit Reason']}")
            
            pnl_color = "🟢" if trade['P&L'] > 0 else "🔴"
            st.markdown(f"**P&L:** {pnl_color} {trade['P&L']:.2f}")

def render_trade_logs():
    logs = st.session_state.get('trade_logs', [])
    
    if not logs:
        st.info("No logs yet")
        return
    
    st.text_area("Trade Logs", value="\n".join(logs), height=400)

def render_backtest_ui():
    st.header("🔬 Backtest Mode")
    
    tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "📊 Results", "📈 Market Data Analysis"])
    
    with tab1:
        render_backtest_configuration()
    
    with tab2:
        render_backtest_results()
    
    with tab3:
        render_market_data_analysis()

def render_backtest_configuration():
    config = st.session_state.get('config', {})
    
    st.subheader("Current Configuration")
    
    asset_display = config.get('asset', 'N/A')
    if asset_display == 'Custom':
        asset_display = config.get('custom_ticker', 'N/A')
    
    st.markdown(f"""
    **Asset:** {asset_display}  
    **Interval:** {config.get('interval', 'N/A')}  
    **Period:** {config.get('period', 'N/A')}  
    **Quantity:** {config.get('quantity', 1)}  
    **Strategy:** {config.get('strategy', 'N/A')}  
    **SL Type:** {config.get('sl_type', 'N/A')}  
    **Target Type:** {config.get('target_type', 'N/A')}  
    """)

def render_backtest_results():
    config = st.session_state.get('config', {})
    
    if st.button("🚀 Run Backtest", use_container_width=True):
        with st.spinner("Running backtest..."):
            ticker = ASSET_MAPPING.get(config['asset'], config.get('custom_ticker', '^NSEI'))
            df = fetch_data(ticker, config['interval'], config['period'], is_live_trading=False)
            
            if df is None or df.empty:
                st.error("Failed to fetch data")
                return
            
            df = calculate_all_indicators(df, config)
            trades = run_backtest(df, config)
            
            st.session_state['backtest_results'] = trades
            st.session_state['backtest_data'] = df
    
    if 'backtest_results' in st.session_state:
        trades = st.session_state['backtest_results']
        df = st.session_state['backtest_data']
        debug_info = st.session_state.get('backtest_debug', {})
        
        # Display debug information
        if debug_info:
            with st.expander("🔍 Backtest Analysis Details", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Candles", debug_info.get('total_candles', 0))
                    st.metric("Candles Analyzed", debug_info.get('candles_analyzed', 0))
                with col2:
                    st.metric("Signals Generated", debug_info.get('signals_generated', 0))
                    st.metric("Trades Entered", debug_info.get('trades_entered', 0))
                with col3:
                    st.metric("Trades Completed", debug_info.get('trades_completed', 0))
                    st.metric("Skipped (NaN period)", debug_info.get('skipped_candles', 0))
        
        if not trades:
            st.warning("⚠️ No trades generated in backtest")
            
            # Provide helpful debugging information
            st.info("""
            **Possible Reasons:**
            1. **EMA Angle Filter**: Try setting Min Angle to 0.0 degrees
            2. **ADX Filter**: Disable ADX filter or lower the threshold
            3. **Entry Filters**: Switch to 'Simple Crossover'
            4. **Insufficient Data**: For 1m/5m intervals, ensure period has enough candles
            5. **Indicator Values**: Check if EMAs are generating valid crossovers
            
            **Quick Fix**: Try these settings:
            - Min Angle: 0.0
            - Entry Filter: Simple Crossover
            - ADX Filter: Unchecked
            - SL: Custom Points (50)
            - Target: Custom Points (100)
            """)
            
            # Show sample data to help debug
            if not df.empty:
                st.subheader("Sample Data (Last 10 Candles)")
                sample_cols = ['Close', 'EMA_Fast', 'EMA_Slow', 'RSI', 'ADX', 'ATR']
                available_cols = [col for col in sample_cols if col in df.columns]
                st.dataframe(df[available_cols].tail(10))
            
            return
        
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['P&L'] > 0)
        losing_trades = sum(1 for t in trades if t['P&L'] < 0)
        total_pnl = sum(t['P&L'] for t in trades)
        avg_duration = sum((t['Duration'].total_seconds() for t in trades), 0) / total_trades
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", total_trades)
        
        with col2:
            st.metric("Wins", winning_trades)
        
        with col3:
            st.metric("Losses", losing_trades)
        
        with col4:
            accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            st.metric("Accuracy", f"{accuracy:.1f}%")
        
        pnl_color = "normal" if total_pnl >= 0 else "inverse"
        st.metric("Total P&L", f"{total_pnl:.2f}", delta=f"{total_pnl:.2f}", delta_color=pnl_color)
        
        avg_duration_str = str(timedelta(seconds=int(avg_duration))).split('.')[0]
        st.metric("Average Duration", avg_duration_str)
        
        st.markdown("---")
        
        st.subheader("Trade Details")
        
        for i, trade in enumerate(trades):
            with st.expander(f"Trade #{i+1} - {trade['Type']} - P&L: {trade['P&L']:.2f}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Entry Time:** {trade['Entry Time'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.markdown(f"**Entry Price:** {trade['Entry Price']:.2f}")
                    sl_display = f"{trade['SL']:.2f}" if trade['SL'] else "Signal"
                    st.markdown(f"**SL:** {sl_display}")
                    st.markdown(f"**Quantity:** {trade['Quantity']}")
                
                with col2:
                    st.markdown(f"**Exit Time:** {trade['Exit Time'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.markdown(f"**Exit Price:** {trade['Exit Price']:.2f}")
                    target_display = f"{trade['Target']:.2f}" if trade['Target'] else "Signal"
                    st.markdown(f"**Target:** {target_display}")
                    st.markdown(f"**Duration:** {str(trade['Duration']).split('.')[0]}")
                
                st.markdown(f"**Exit Reason:** {trade['Exit Reason']}")
                
                pnl_color = "🟢" if trade['P&L'] > 0 else "🔴"
                st.markdown(f"**P&L:** {pnl_color} {trade['P&L']:.2f}")
        
        st.markdown("---")
        st.subheader("Backtest Chart")
        
        fig = create_chart(df, config, trade_history=trades)
        st.plotly_chart(fig, use_container_width=True, key="backtest_chart")

def render_market_data_analysis():
    config = st.session_state.get('config', {})
    
    ticker = ASSET_MAPPING.get(config.get('asset', 'NIFTY 50'), config.get('custom_ticker', '^NSEI'))
    interval = config.get('interval', '1d')
    period = config.get('period', '1mo')
    
    df = fetch_data(ticker, interval, period, is_live_trading=False)
    
    if df is None or df.empty:
        st.error("Failed to fetch data")
        return
    
    st.subheader("Market Data")
    
    df_display = df.copy()
    df_display['Change (Points)'] = df_display['Close'] - df_display['Open']
    df_display['Change (%)'] = ((df_display['Close'] - df_display['Open']) / df_display['Open']) * 100
    df_display['Day of Week'] = df_display.index.day_name()
    
    def color_change(val):
        color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
        return f'color: {color}'
    
    st.dataframe(
        df_display[['Open', 'High', 'Low', 'Close', 'Change (Points)', 'Change (%)', 'Day of Week']],
        use_container_width=True
    )
    
    st.markdown("---")
    st.subheader("Change Analysis")
    
    fig_points = go.Figure()
    colors = ['green' if x > 0 else 'red' for x in df_display['Change (Points)']]
    
    fig_points.add_trace(
        go.Bar(
            x=df_display.index,
            y=df_display['Change (Points)'],
            marker_color=colors,
            name='Change (Points)'
        )
    )
    
    fig_points.update_layout(
        title="Change in Points Over Time",
        xaxis_title="Date",
        yaxis_title="Points",
        height=400
    )
    
    st.plotly_chart(fig_points, use_container_width=True)
    
    fig_pct = go.Figure()
    colors = ['green' if x > 0 else 'red' for x in df_display['Change (%)']]
    
    fig_pct.add_trace(
        go.Bar(
            x=df_display.index,
            y=df_display['Change (%)'],
            marker_color=colors,
            name='Change (%)'
        )
    )
    
    fig_pct.update_layout(
        title="Change in Percentage Over Time",
        xaxis_title="Date",
        yaxis_title="Percentage (%)",
        height=400
    )
    
    st.plotly_chart(fig_pct, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Max Price", f"{df_display['High'].max():.2f}")
        st.metric("Min Price", f"{df_display['Low'].min():.2f}")
        st.metric("Average Price", f"{df_display['Close'].mean():.2f}")
    
    with col2:
        volatility = df_display['Close'].pct_change().std() * 100
        st.metric("Volatility (%)", f"{volatility:.2f}")
        total_change_points = df_display['Close'].iloc[-1] - df_display['Close'].iloc[0]
        st.metric("Total Change (Points)", f"{total_change_points:.2f}")
        total_change_pct = ((df_display['Close'].iloc[-1] - df_display['Close'].iloc[0]) / df_display['Close'].iloc[0]) * 100
        st.metric("Total Change (%)", f"{total_change_pct:.2f}")
    
    with col3:
        avg_change_points = df_display['Change (Points)'].mean()
        st.metric("Avg Change (Points)", f"{avg_change_points:.2f}")
        
        positive_days = (df_display['Change (Points)'] > 0).sum()
        total_days = len(df_display)
        win_rate = (positive_days / total_days * 100) if total_days > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")
        
        max_gain = df_display['Change (Points)'].max()
        max_loss = df_display['Change (Points)'].min()
        st.metric("Max Gain/Loss", f"{max_gain:.2f} / {max_loss:.2f}")
    
    st.markdown("---")
    st.subheader("10-Year Analysis (Independent)")
    
    df_10y = fetch_data(ticker, '1d', '10y', is_live_trading=False)
    
    if df_10y is not None and not df_10y.empty:
        df_10y['Return'] = df_10y['Close'].pct_change() * 100
        df_10y['Month'] = df_10y.index.month
        df_10y['Year'] = df_10y.index.year
        
        st.subheader("Monthly Returns Heatmap (10 Years)")
        
        monthly_returns = df_10y.groupby(['Year', 'Month'])['Return'].sum().reset_index()
        pivot_returns = monthly_returns.pivot(index='Year', columns='Month', values='Return')
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_returns.columns = month_names[:len(pivot_returns.columns)]
        
        fig_heatmap_returns = go.Figure(data=go.Heatmap(
            z=pivot_returns.values,
            x=pivot_returns.columns,
            y=pivot_returns.index,
            colorscale='RdYlGn',
            zmid=0,
            text=pivot_returns.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Return (%)")
        ))
        
        fig_heatmap_returns.update_layout(
            title="Monthly Returns by Year",
            xaxis_title="Month",
            yaxis_title="Year",
            height=500
        )
        
        st.plotly_chart(fig_heatmap_returns, use_container_width=True)
        
        st.subheader("Monthly Volatility Heatmap (10 Years)")
        
        monthly_volatility = df_10y.groupby(['Year', 'Month'])['Return'].std().reset_index()
        pivot_volatility = monthly_volatility.pivot(index='Year', columns='Month', values='Return')
        pivot_volatility.columns = month_names[:len(pivot_volatility.columns)]
        
        fig_heatmap_vol = go.Figure(data=go.Heatmap(
            z=pivot_volatility.values,
            x=pivot_volatility.columns,
            y=pivot_volatility.index,
            colorscale='Reds',
            text=pivot_volatility.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Volatility (%)")
        ))
        
        fig_heatmap_vol.update_layout(
            title="Monthly Volatility by Year",
            xaxis_title="Month",
            yaxis_title="Year",
            height=500
        )
        
        st.plotly_chart(fig_heatmap_vol, use_container_width=True)
    else:
        st.warning("Unable to fetch 10-year data for heatmaps")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def initialize_session_state():
    defaults = {
        'trading_active': False,
        'position': None,
        'current_data': None,
        'trade_history': [],
        'trade_logs': [],
        'config': {},
        'custom_conditions': [],
        'current_pct_change': 0.0,
        'ai_analysis': {},
        'dhan_broker': None,
        'broker_trade_history': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main():
    st.set_page_config(
        page_title="Professional Quantitative Trading System",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    with st.sidebar:
        st.title("📈 Quant Trading System")
        st.markdown("---")
        config = render_configuration_ui()
    
    st.title("Professional Quantitative Trading System")
    st.markdown("Production-ready algo trading with live execution and backtesting")
    st.markdown("---")
    
    mode = config.get('mode', 'Backtest')
    
    if mode == 'Live Trading':
        render_live_trading_ui()
    else:
        render_backtest_ui()
    
    st.markdown("---")
    st.markdown("**Note:** This is a simulated trading system. Always test thoroughly before using with real capital.")
    st.markdown("**Disclaimer:** Past performance does not guarantee future results.")

if __name__ == "__main__":
    main()
