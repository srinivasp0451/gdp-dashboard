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
from scipy.signal import argrelextrema

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
    "Custom Ticker": "CUSTOM",  # Placeholder for custom input
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
    "Custom Strategy",
    "SuperTrend AI",
    "VWAP + Volume Spike",
    "Bollinger Squeeze Breakout",
    "Elliott Waves + Ratio Charts"
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
    "Strategy-based Signal",
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
    "Dynamic Trailing SL+Target (Lock Profits)",
    "50% Exit at Target (Partial)",
    "Current Candle Low/High",
    "Previous Candle Low/High",
    "Current Swing Low/High",
    "Previous Swing Low/High",
    "ATR-based",
    "Risk-Reward Based",
    "Signal-based (Reverse Crossover)",
    "Strategy-based Signal"
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
        """Determine exchange segment based on asset and trading type"""
        if not self.dhanhq_module:
            return "NSE_FNO"
        
        is_options    = self.config.get('dhan_is_options', True)
        trading_type  = self.config.get('dhan_trading_type', 'Intraday')
        exchange      = self.config.get('dhan_exchange', 'NSE')

        if is_options:
            # Options → FNO segment
            asset = self.config.get('asset', 'NIFTY 50')
            if asset == 'SENSEX':
                return self.dhanhq_module.BSE_FNO
            return self.dhanhq_module.NSE_FNO
        else:
            # Stocks (Intraday or Delivery) → Equity segment
            if exchange == 'BSE':
                return self.dhanhq_module.BSE  # BSE_EQ
            return self.dhanhq_module.NSE       # NSE_EQ
    
    def place_order(self, transaction_type, security_id, quantity, signal_type=None, order_params=None):
        """
        Place order via Dhan API.
        Supports: Market/Limit orders, CNC/Delivery, Bracket Orders (BO) with SL+Target+Trail.
        """
        order_response = {
            'order_id': None, 'status': 'FAILED', 'raw_response': None, 'error': None
        }
        try:
            if self.initialized and self.dhan:
                exchange_segment = self._get_exchange_segment()
                is_options   = self.config.get('dhan_is_options', True)
                trading_type = self.config.get('dhan_trading_type', 'Intraday')
                use_broker_sl = self.config.get('broker_use_own_sl', False)
                order_type_selection = self.config.get('dhan_order_type', 'Market Order')
                op = order_params or {}

                # Determine order type
                if order_type_selection == 'Limit Order':
                    order_type = self.dhanhq_module.LIMIT
                    limit_price = float(op.get('price', 0)) if op else 0
                else:
                    order_type = self.dhanhq_module.MARKET
                    limit_price = 0

                if use_broker_sl and op:
                    # ── Bracket Order (BO) - always uses LIMIT ──────────────
                    lmt_price = float(op.get('price', 0))
                    bo_profit  = float(op.get('boProfitValue', 0))
                    bo_sl      = float(op.get('boStopLossValue', 0))
                    trail_sl   = float(op.get('trailStopLoss', 0))

                    if is_options or trading_type == 'Intraday':
                        product = self.dhanhq_module.BO
                    else:
                        product = self.dhanhq_module.BO

                    response = self.dhan.place_order(
                        security_id=str(security_id),
                        exchange_segment=exchange_segment,
                        transaction_type=transaction_type,
                        quantity=int(quantity),
                        order_type=self.dhanhq_module.LIMIT,  # BO always LIMIT
                        product_type=product,
                        price=lmt_price,
                        bo_profit_value=bo_profit,
                        bo_stop_loss_value=bo_sl,
                        trailing_stop_loss=trail_sl
                    )

                elif not is_options and trading_type == 'Delivery (CNC)':
                    # ── CNC (Market or Limit) ────────────────────────────────
                    response = self.dhan.place_order(
                        security_id=str(security_id),
                        exchange_segment=exchange_segment,
                        transaction_type=transaction_type,
                        quantity=int(quantity),
                        order_type=order_type,
                        product_type=self.dhanhq_module.CNC,
                        price=limit_price
                    )
                else:
                    # ── Intraday / Options (Market or Limit) ────────────────
                    response = self.dhan.place_order(
                        security_id=str(security_id),
                        exchange_segment=exchange_segment,
                        transaction_type=transaction_type,
                        quantity=int(quantity),
                        order_type=order_type,
                        product_type=self.dhanhq_module.INTRA,
                        price=limit_price
                    )

                order_response['raw_response'] = response
                if response and response.get('status') == 'success':
                    order_response['order_id'] = response.get('data', {}).get('orderId', f"ORDER-{int(time.time())}")
                    order_response['status'] = 'SUCCESS'
                else:
                    order_response['order_id'] = f"ERR-{int(time.time())}"
                    order_response['error'] = str(response.get('remarks', 'Unknown error'))

            else:
                # Simulation mode
                order_response['order_id'] = f"SIM-{int(time.time())}"
                order_response['status'] = 'SIMULATED'
                order_response['raw_response'] = {'mode': 'simulation', 'params': order_params}

        except Exception as e:
            order_response['order_id'] = f"ERR-{int(time.time())}"
            order_response['error'] = str(e)
            order_response['raw_response'] = {'error': str(e), 'traceback': traceback.format_exc()}

        return order_response
    
    def enter_broker_position(self, signal, price, config, log_func):
        """Enter broker position - Options, Intraday, Delivery. Bracket Order when broker SL enabled."""
        is_options   = config.get('dhan_is_options', True)
        quantity     = config.get('dhan_quantity', 10)
        trading_type = config.get('dhan_trading_type', 'Intraday')
        use_broker_sl = config.get('broker_use_own_sl', False)
        log_func(f"🏦 NEW signal detected: {signal}")

        # ── Build bracket order params if broker SL/Target enabled ──────────
        def _build_bo_params(txn, entry_px):
            """Build BO order params (boProfitValue / boStopLossValue)"""
            if not use_broker_sl:
                # Even without BO, pass price for limit orders
                return {'price': entry_px}
            sl_pts  = float(config.get('broker_sl_points', 50))
            tgt_pts = float(config.get('broker_target_points', 100))
            trail   = float(config.get('broker_trailing_jump', 0))
            # bo_profit_value and bo_stop_loss_value are DISTANCES, not absolute prices
            return {
                'price':          entry_px,
                'boProfitValue':  tgt_pts,
                'boStopLossValue': sl_pts,
                'trailStopLoss':  trail
            }

        if is_options:
            security_id, option_type = self._resolve_security(signal)
            log_func(f"🏦 Options [{option_type}] Security ID: {security_id}")
            txn = 'BUY'
            op  = _build_bo_params(txn, price)
            order_response = self.place_order(txn, security_id, quantity, signal, op)
            broker_position = {
                'order_id': order_response['order_id'],
                'signal_type': signal, 'option_type': option_type,
                'security_id': security_id, 'transaction_type': txn,
                'entry_price': price, 'quantity': quantity,
                'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')),
                'status': order_response['status'],
                'raw_response': order_response['raw_response'],
                'is_options': True, 'trading_type': 'Options',
                'broker_sl_active': use_broker_sl
            }

        else:
            security_id = config.get('dhan_security_id', '1234')
            txn = 'BUY' if signal in ('BUY', 'LONG') else 'SELL'
            log_func(f"🏦 {'Delivery' if trading_type=='Delivery (CNC)' else 'Intraday'} → {txn} | Security: {security_id}")
            op = _build_bo_params(txn, price)
            order_response = self.place_order(txn, security_id, quantity, signal, op)
            broker_position = {
                'order_id': order_response['order_id'],
                'signal_type': signal, 'security_id': security_id,
                'transaction_type': txn, 'entry_price': price, 'quantity': quantity,
                'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')),
                'status': order_response['status'],
                'raw_response': order_response['raw_response'],
                'is_options': False, 'trading_type': trading_type,
                'broker_sl_active': use_broker_sl
            }

        if order_response['status'] in ('SUCCESS', 'SIMULATED'):
            bo_info = " [Bracket Order: SL/Target managed by Dhan]" if use_broker_sl else ""
            log_func(f"🏦 ✅ ORDER PLACED: {broker_position['transaction_type']} {quantity} @ {price:.2f}{bo_info}")
        else:
            log_func(f"🏦 ❌ ORDER FAILED: {order_response.get('error', 'Unknown error')}")
        return broker_position
    
    def exit_broker_position(self, broker_position, price, reason, log_func):
        """
        Exit broker position - handles both options and stock trading
        
        Options Trading:
        - Always SELL (sell the option you bought)
        
        Stock Trading:
        - If entered with BUY → Exit with SELL
        - If entered with SELL → Exit with BUY (square off short)
        
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
        is_options = broker_position.get('is_options', True)
        
        log_func(f"🏦 Exiting position: {reason}")
        
        if is_options:
            # Options: Always SELL to close
            exit_transaction = 'SELL'
            log_func(f"🏦 Options Exit → SELL")
        else:
            # Stock: Exit opposite of entry
            entry_transaction = broker_position['transaction_type']
            if entry_transaction == 'BUY':
                exit_transaction = 'SELL'
                log_func(f"🏦 Stock Exit → SELL (close long)")
            else:  # entry was SELL
                exit_transaction = 'BUY'
                log_func(f"🏦 Stock Exit → BUY (square off short)")
        
        order_response = self.place_order(
            exit_transaction, 
            security_id, 
            quantity,
            order_params={'price': price}  # Pass exit price for limit orders
        )
        
        # Calculate P&L
        entry_price = broker_position['entry_price']
        signal_type = broker_position['signal_type']
        
        if signal_type in ('BUY', 'LONG'):
            pnl = (price - entry_price) * quantity
        else:  # 'SELL' or 'SHORT'
            pnl = (entry_price - price) * quantity
            
        exit_info = {
            'order_id': order_response['order_id'],
            'transaction_type': exit_transaction,
            'exit_price': price,
            'quantity': quantity,
            'pnl': pnl,
            'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')),
            'status': order_response['status'],
            'raw_response': order_response['raw_response']
        }
        
        if order_response['status'] in ('SUCCESS', 'SIMULATED'):
            log_func(f"🏦 ✅ DHAN EXIT ORDER PLACED: {exit_transaction} {quantity} @ {price:.2f} | P&L: ₹{pnl:.2f}")
        else:
            log_func(f"🏦 ❌ DHAN EXIT ORDER FAILED: {order_response.get('error', 'Unknown error')}")
            
        return exit_info
    
    def clear_all_positions(self, log_func, convert_to_market=True):
        """
        Clear all positions: cancel/convert pending orders and close open positions
        
        Args:
            log_func: Logging function
            convert_to_market: If True, try to convert pending LIMIT orders to MARKET for faster fill
        
        Returns:
            dict with cleared orders and positions count
        """
        result = {
            'cancelled_orders': 0,
            'converted_orders': 0,
            'closed_positions': 0,
            'errors': [],
            'clearing_complete': False
        }
        
        if not self.initialized or not self.dhan:
            log_func("🏦 ⚠️ Broker not initialized - skipping position clear")
            result['clearing_complete'] = True
            return result
        
        try:
            log_func("🏦 🧹 Starting position clearing process...")
            
            # Get all orders
            order_list = self.dhan.get_order_list()
            
            if order_list and order_list.get('status') == 'success':
                orders = order_list.get('data', [])
                log_func(f"🏦 Found {len(orders)} orders to process")
                
                for order in orders:
                    order_status = order.get('orderStatus', '')
                    order_id = order.get('orderId', '')
                    order_type = order.get('orderType', '')
                    
                    # Handle pending LIMIT orders
                    if order_status == 'PENDING' and order_type == 'LIMIT' and convert_to_market:
                        try:
                            log_func(f"🏦 Converting pending LIMIT order {order_id} to MARKET...")
                            
                            # Cancel the LIMIT order first
                            cancel_response = self.dhan.cancel_order(order_id)
                            
                            if cancel_response and cancel_response.get('status') == 'success':
                                log_func(f"🏦 ✅ Cancelled LIMIT order: {order_id}")
                                
                                # Place new MARKET order with same parameters
                                market_response = self.dhan.place_order(
                                    tag=order.get('tag', ''),
                                    transaction_type=order.get('transactionType'),
                                    exchange_segment=order.get('exchangeSegment'),
                                    product_type=order.get('productType'),
                                    order_type=self.dhanhq_module.MARKET,
                                    security_id=str(order.get('securityId', '')),
                                    quantity=int(order.get('quantity', 0)),
                                    price=0
                                )
                                
                                if market_response and market_response.get('status') == 'success':
                                    result['converted_orders'] += 1
                                    log_func(f"🏦 ✅ Converted to MARKET order: {market_response.get('data', {}).get('orderId', 'N/A')}")
                                else:
                                    # If market order fails, just count as cancelled
                                    result['cancelled_orders'] += 1
                                    log_func(f"🏦 ⚠️ MARKET conversion failed, order cancelled")
                            else:
                                error_msg = f"Failed to cancel order {order_id}: {cancel_response.get('remarks', 'Unknown')}"
                                result['errors'].append(error_msg)
                                log_func(f"🏦 ⚠️ {error_msg}")
                        except Exception as e:
                            error_msg = f"Error converting order {order_id}: {str(e)}"
                            result['errors'].append(error_msg)
                            log_func(f"🏦 ❌ {error_msg}")
                    
                    # Handle other pending orders (non-LIMIT or if convert disabled)
                    elif order_status == 'PENDING':
                        try:
                            cancel_response = self.dhan.cancel_order(order_id)
                            if cancel_response and cancel_response.get('status') == 'success':
                                result['cancelled_orders'] += 1
                                log_func(f"🏦 ✅ Cancelled pending order: {order_id}")
                            else:
                                error_msg = f"Failed to cancel order {order_id}: {cancel_response.get('remarks', 'Unknown')}"
                                result['errors'].append(error_msg)
                                log_func(f"🏦 ⚠️ {error_msg}")
                        except Exception as e:
                            error_msg = f"Error cancelling order {order_id}: {str(e)}"
                            result['errors'].append(error_msg)
                            log_func(f"🏦 ❌ {error_msg}")
                    
                    # Close open positions (TRANSIT/TRADED status)
                    elif order_status in ['TRANSIT', 'TRADED']:
                        try:
                            # Place opposite MARKET order to close immediately
                            opposite_txn = 'SELL' if order.get('transactionType') == 'BUY' else 'BUY'
                            
                            log_func(f"🏦 Closing position {order_id} with {opposite_txn} MARKET order...")
                            
                            close_response = self.dhan.place_order(
                                tag=order.get('tag', ''),
                                transaction_type=opposite_txn,
                                exchange_segment=order.get('exchangeSegment'),
                                product_type=order.get('productType'),
                                order_type=self.dhanhq_module.MARKET,
                                security_id=str(order.get('securityId', '')),
                                quantity=int(order.get('quantity', 0)),
                                price=0
                            )
                            
                            if close_response and close_response.get('status') == 'success':
                                result['closed_positions'] += 1
                                log_func(f"🏦 ✅ Closed position: {order_id} with {opposite_txn}")
                            else:
                                error_msg = f"Failed to close position {order_id}: {close_response.get('remarks', 'Unknown')}"
                                result['errors'].append(error_msg)
                                log_func(f"🏦 ⚠️ {error_msg}")
                        except Exception as e:
                            error_msg = f"Error closing position {order_id}: {str(e)}"
                            result['errors'].append(error_msg)
                            log_func(f"🏦 ❌ {error_msg}")
            
            # Mark clearing as complete
            result['clearing_complete'] = True
            
            summary_msg = f"🏦 🧹 Clearing Complete: {result['cancelled_orders']} cancelled, {result['converted_orders']} converted, {result['closed_positions']} closed"
            log_func(summary_msg)
            
            if result['errors']:
                log_func(f"🏦 ⚠️ {len(result['errors'])} errors during clearing")
            
        except Exception as e:
            error_msg = f"Error in clear_all_positions: {str(e)}"
            result['errors'].append(error_msg)
            log_func(f"🏦 ❌ {error_msg}")
            result['clearing_complete'] = True  # Mark as complete even with error to unblock
        
        return result

# ================================
# DATA FETCHING
# ================================

def fetch_data(ticker_symbol, interval, period, is_live_trading=False, custom_ticker=None):
    """
    Fetch historical/live data using yfinance
    
    Args:
        ticker_symbol: Asset ticker
        interval: Time interval
        period: Historical period
        is_live_trading: If True, fetch minimal data for live trading
        custom_ticker: Custom ticker symbol if using "Custom Ticker" option
        
    Returns:
        DataFrame with OHLCV data in IST timezone
    """
    try:
        # Use custom ticker if provided
        if ticker_symbol == "Custom Ticker" and custom_ticker:
            ticker = custom_ticker
        else:
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
    
    # SMA for custom strategy
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
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
    
    # Bollinger Bands
    bb_period = config.get('custom_bb_period', 20)
    bb_std = config.get('custom_bb_std', 2.0)
    df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
    bb_std_dev = df['Close'].rolling(window=bb_period).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Volume MA (if volume exists)
    if 'Volume' in df.columns:
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    
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
    """
    Price crosses threshold strategy - checks current price state
    
    Checks if current price is above/below threshold and takes action
    No need for actual "crossing" - just checks current state
    
    Combinations:
    - Above Threshold → LONG
    - Above Threshold → SHORT
    - Below Threshold → LONG
    - Below Threshold → SHORT
    """
    if current_position is not None:
        return None, None
        
    threshold = config.get('price_threshold', 25000)
    current_price = df.iloc[idx]['Close']
    cross_type = config.get('price_cross_type', 'Above Threshold')
    position_type = config.get('price_cross_position', 'LONG')
    
    # Check current price state against threshold
    condition_met = False
    
    if cross_type == 'Above Threshold':
        # Check if current price IS above threshold
        if current_price > threshold:
            condition_met = True
    else:  # 'Below Threshold'
        # Check if current price IS below threshold
        if current_price < threshold:
            condition_met = True
    
    # If condition met, return signal based on position type
    if condition_met:
        if position_type == 'LONG':
            return 'BUY', current_price
        else:  # SHORT
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
    """
    Percentage change strategy with full flexibility
    
    Combinations:
    - Positive % → LONG
    - Positive % → SHORT
    - Negative % → LONG
    - Negative % → SHORT
    """
    if current_position is not None:
        return None, None
        
    if idx < 1:
        return None, None
        
    current_price = df.iloc[idx]['Close']
    prev_price = df.iloc[idx - 1]['Close']
    
    pct_change = ((current_price - prev_price) / prev_price) * 100
    threshold = config.get('pct_change_threshold', 2.0)
    change_type = config.get('pct_change_type', 'Positive % (Price Up)')
    position_type = config.get('pct_change_position', 'LONG')
    
    # Determine if condition met
    condition_met = False
    
    if 'Positive' in change_type:
        # Check for positive % change
        if pct_change >= threshold:
            condition_met = True
    else:  # Negative
        # Check for negative % change
        if pct_change <= -threshold:
            condition_met = True
    
    # If condition met, return signal based on position type
    if condition_met:
        if position_type == 'LONG':
            return 'BUY', current_price
        else:  # SHORT
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
    """
    Custom Strategy Builder — multi-indicator, AND/OR combine logic.
    Reads config['custom_conditions'] list (one dict per condition).
    Falls back to legacy single-condition keys if list not present.

    Indicators:
    Price Crosses / Pullback: EMA, SMA, BB Upper/Lower/Middle
    Level Crosses: RSI, MACD, MACD Histogram, ADX, Volume, BB %B,
                   ATR (Volatility), Historical Volatility, Std Dev (Volatility)
    Crossover: Fast EMA x Slow EMA, Fast SMA x Slow SMA, MACD x Signal,
               Price x EMA, Price x SMA, RSI OB/OS
    """
    if current_position is not None:
        return None, None
    if idx < 2:
        return None, None

    current       = df.iloc[idx]
    previous      = df.iloc[idx - 1]
    current_price = float(current["Close"])
    prev_price    = float(previous["Close"])

    def _col(name, period, bb_std=2.0):
        key = f"_CUST_{name}_{period}_{bb_std}"
        if key not in df.columns:
            if name == "EMA":
                df[key] = df["Close"].ewm(span=period, adjust=False).mean()
            elif name == "SMA":
                df[key] = df["Close"].rolling(window=period).mean()
            elif name == "BB_U":
                mid = df["Close"].rolling(window=period).mean()
                df[key] = mid + bb_std * df["Close"].rolling(window=period).std()
            elif name == "BB_L":
                mid = df["Close"].rolling(window=period).mean()
                df[key] = mid - bb_std * df["Close"].rolling(window=period).std()
            elif name == "BB_M":
                df[key] = df["Close"].rolling(window=period).mean()
            elif name == "ATR":
                df[key] = calculate_atr(df, int(period))
            elif name == "HV":
                df[key] = df["Close"].pct_change().rolling(window=int(period)).std() * (252 ** 0.5) * 100
            elif name == "STDDEV":
                df[key] = df["Close"].rolling(window=int(period)).std()
            elif name == "RSI":
                df[key] = calculate_rsi(df["Close"], int(period))
            elif name == "VOL_MA":
                df[key] = df["Volume"].rolling(window=int(period)).mean() if "Volume" in df.columns else float("nan")
        return key

    def _ca(cv, pv, lvl): return float(pv) <= float(lvl) and float(cv) > float(lvl)
    def _cb(cv, pv, lvl): return float(pv) >= float(lvl) and float(cv) < float(lvl)

    def _eval(c):
        stype = c.get("strategy_type", "Price Crosses Indicator")
        ind   = c.get("indicator", "EMA")
        cross = c.get("cross_type", "Above Indicator")
        ptype = c.get("position_type", "LONG")

        if stype == "Price Crosses Indicator":
            bb_std = float(c.get("bb_std", 2.0))
            period = int(c.get("period", c.get("bb_period", 20)))
            imap = {
                "EMA":       ("EMA",   period, 2.0),
                "SMA":       ("SMA",   period, 2.0),
                "BB Upper":  ("BB_U",  int(c.get("bb_period", 20)), bb_std),
                "BB Lower":  ("BB_L",  int(c.get("bb_period", 20)), bb_std),
                "BB Middle": ("BB_M",  int(c.get("bb_period", 20)), 2.0),
            }
            if ind not in imap: return False, ptype
            col = _col(*imap[ind])
            iv, piv = current.get(col, float("nan")), previous.get(col, float("nan"))
            if pd.isna(iv) or pd.isna(piv): return False, ptype
            triggered = _ca(current_price, prev_price, float(iv)) if cross == "Above Indicator"                    else _cb(current_price, prev_price, float(iv))
            return triggered, ptype

        elif stype == "Price Pullback from Indicator":
            bb_std = float(c.get("bb_std", 2.0))
            period = int(c.get("period", c.get("bb_period", 20)))
            imap = {
                "EMA":      ("EMA",  period, 2.0),
                "SMA":      ("SMA",  period, 2.0),
                "BB Upper": ("BB_U", int(c.get("bb_period", 20)), bb_std),
                "BB Lower": ("BB_L", int(c.get("bb_period", 20)), bb_std),
            }
            if ind not in imap: return False, ptype
            col = _col(*imap[ind])
            iv = current.get(col, float("nan"))
            if pd.isna(iv): return False, ptype
            iv = float(iv)
            side = c.get("pullback_side", "Approach from Above")
            triggered = abs(current_price - iv) <= float(c.get("pullback_points", 10)) and (
                current_price >= iv if side == "Approach from Above" else current_price <= iv)
            return triggered, ptype

        elif stype == "Indicator Crosses Level":
            level = float(c.get("level", 50.0))
            chk = _ca if "Above" in cross else _cb

            if ind == "RSI":
                col = _col("RSI", int(c.get("rsi_period", 14)))
                cv, pv = current.get(col, float("nan")), previous.get(col, float("nan"))
                if pd.isna(cv) or pd.isna(pv): return False, ptype
                return chk(cv, pv, level), ptype

            elif ind == "MACD":
                cv, pv = current.get("MACD", float("nan")), previous.get("MACD", float("nan"))
                if pd.isna(cv) or pd.isna(pv): return False, ptype
                return chk(cv, pv, level), ptype

            elif ind == "MACD Histogram":
                cv, pv = current.get("MACD_Hist", float("nan")), previous.get("MACD_Hist", float("nan"))
                if pd.isna(cv) or pd.isna(pv): return False, ptype
                return chk(cv, pv, level), ptype

            elif ind == "ADX":
                cv, pv = current.get("ADX", float("nan")), previous.get("ADX", float("nan"))
                if pd.isna(cv) or pd.isna(pv): return False, ptype
                return chk(cv, pv, level), ptype

            elif ind == "Volume":
                if "Volume" not in df.columns: return False, ptype
                vcol = _col("VOL_MA", int(c.get("volume_ma_period", 20)))
                cv  = float(current.get("Volume", 0))
                pv  = float(previous.get("Volume", 0))
                vma = float(current.get(vcol, 1) or 1)
                thresh = vma * float(c.get("volume_multiplier", 1.5))
                return chk(cv, pv, thresh), ptype

            elif ind == "BB %B":
                bp   = int(c.get("bb_period", 20))
                bstd = float(c.get("bb_std", 2.0))
                mid  = df["Close"].rolling(bp).mean()
                std  = df["Close"].rolling(bp).std()
                pctb_key = f"_CUST_PCTB_{bp}_{bstd}"
                df[pctb_key] = (df["Close"] - (mid - bstd * std)) / (2 * bstd * std)
                cv, pv = df[pctb_key].iloc[idx], df[pctb_key].iloc[idx-1]
                if pd.isna(cv) or pd.isna(pv): return False, ptype
                return chk(cv, pv, level / 100), ptype

            elif ind == "ATR (Volatility)":
                col = _col("ATR", int(c.get("atr_period", 14)))
                cv, pv = current.get(col, float("nan")), previous.get(col, float("nan"))
                if pd.isna(cv) or pd.isna(pv): return False, ptype
                return chk(cv, pv, level), ptype

            elif ind == "Historical Volatility":
                col = _col("HV", int(c.get("hv_period", 20)))
                cv, pv = current.get(col, float("nan")), previous.get(col, float("nan"))
                if pd.isna(cv) or pd.isna(pv): return False, ptype
                return chk(cv, pv, level), ptype

            elif ind == "Std Dev (Volatility)":
                col = _col("STDDEV", int(c.get("stddev_period", 20)))
                cv, pv = current.get(col, float("nan")), previous.get(col, float("nan"))
                if pd.isna(cv) or pd.isna(pv): return False, ptype
                return chk(cv, pv, level), ptype

            return False, ptype

        elif stype == "Indicator Crossover":
            co      = c.get("crossover_type", "Fast EMA x Slow EMA")
            is_bull = "Bullish" in cross

            def bull(fc, fp, sc, sp): return float(fp) <= float(sp) and float(fc) > float(sc)
            def bear(fc, fp, sc, sp): return float(fp) >= float(sp) and float(fc) < float(sc)
            chk2 = bull if is_bull else bear

            if co == "Fast EMA x Slow EMA" or co == "Fast EMA × Slow EMA":
                fc_col = _col("EMA", int(c.get("fast_ema", 9)))
                sc_col = _col("EMA", int(c.get("slow_ema", 21)))
                vals = [current.get(fc_col), previous.get(fc_col), current.get(sc_col), previous.get(sc_col)]
                if any(pd.isna(v) for v in vals): return False, ptype
                return chk2(*vals), ptype

            elif co == "Fast SMA x Slow SMA" or co == "Fast SMA × Slow SMA":
                fc_col = _col("SMA", int(c.get("fast_sma", 20)))
                sc_col = _col("SMA", int(c.get("slow_sma", 50)))
                vals = [current.get(fc_col), previous.get(fc_col), current.get(sc_col), previous.get(sc_col)]
                if any(pd.isna(v) for v in vals): return False, ptype
                return chk2(*vals), ptype

            elif co in ("MACD x Signal", "MACD × Signal"):
                mc, mp = current.get("MACD", float("nan")), previous.get("MACD", float("nan"))
                sc2, sp2 = current.get("MACD_Signal", float("nan")), previous.get("MACD_Signal", float("nan"))
                if any(pd.isna(v) for v in [mc, mp, sc2, sp2]): return False, ptype
                return chk2(mc, mp, sc2, sp2), ptype

            elif co in ("Price x EMA", "Price × EMA"):
                col = _col("EMA", int(c.get("ma_period", 50)))
                ic, ip = current.get(col, float("nan")), previous.get(col, float("nan"))
                if pd.isna(ic) or pd.isna(ip): return False, ptype
                return chk2(current_price, prev_price, float(ic), float(ip)), ptype

            elif co in ("Price x SMA", "Price × SMA"):
                col = _col("SMA", int(c.get("ma_period", 50)))
                ic, ip = current.get(col, float("nan")), previous.get(col, float("nan"))
                if pd.isna(ic) or pd.isna(ip): return False, ptype
                return chk2(current_price, prev_price, float(ic), float(ip)), ptype

            elif co == "RSI Crossover (Overbought/Oversold)":
                rsi_col = _col("RSI", int(c.get("rsi_period", 14)))
                rc, rp  = df[rsi_col].iloc[idx], df[rsi_col].iloc[idx-1]
                if pd.isna(rc) or pd.isna(rp): return False, ptype
                ob, os_lvl = float(c.get("rsi_ob", 70)), float(c.get("rsi_os", 30))
                if is_bull:
                    return (float(rp) <= os_lvl and float(rc) > os_lvl), ptype
                else:
                    return (float(rp) <= ob and float(rc) > ob), ptype

        return False, ptype

    # Gather conditions list; fall back to legacy single-condition keys
    conditions = config.get("custom_conditions", [])
    if not conditions:
        conditions = [{
            "strategy_type":   config.get("custom_strategy_type", "Price Crosses Indicator"),
            "indicator":       config.get("custom_indicator", "EMA"),
            "period":          config.get("custom_indicator_period", 20),
            "bb_period":       config.get("custom_bb_period", 20),
            "bb_std":          config.get("custom_bb_std", 2.0),
            "cross_type":      config.get("custom_cross_type", "Above Indicator"),
            "position_type":   config.get("custom_position_type", "LONG"),
            "pullback_points": config.get("custom_pullback_points", 10),
            "pullback_side":   config.get("custom_pullback_side", "Approach from Above"),
            "crossover_type":  config.get("custom_crossover_type", "Fast EMA x Slow EMA"),
            "fast_ema":        config.get("custom_fast_ema", 9),
            "slow_ema":        config.get("custom_slow_ema", 21),
            "fast_sma":        config.get("custom_fast_sma", 20),
            "slow_sma":        config.get("custom_slow_sma", 50),
            "ma_period":       config.get("custom_ma_period", 50),
            "rsi_period":      config.get("custom_rsi_period", 14),
            "rsi_ob":          config.get("custom_rsi_ob", 70),
            "rsi_os":          config.get("custom_rsi_os", 30),
            "level":           config.get("custom_level", 50.0),
            "volume_ma_period":   config.get("custom_volume_ma_period", 20),
            "volume_multiplier":  config.get("custom_volume_multiplier", 1.5),
            "atr_period":      config.get("custom_atr_period", 14),
            "hv_period":       config.get("custom_hv_period", 20),
            "stddev_period":   config.get("custom_stddev_period", 20),
        }]

    use_and = "AND" in config.get("custom_combine_mode", "AND (all must be true)")
    results  = [_eval(c) for c in conditions]
    flags    = [r[0] for r in results]
    ptypes   = [r[1] for r in results]

    final = all(flags) if use_and else any(flags)
    if not final:
        return None, None

    for flag, ptype in zip(flags, ptypes):
        if flag:
            return ("BUY" if ptype == "LONG" else "SELL"), current_price

    return None, None

def check_supertrend_ai(df, idx, config, current_position):
    """
    SuperTrend AI Strategy - Highly profitable trend-following strategy
    Uses SuperTrend indicator with ADX confirmation and volume filter
    """
    if current_position is not None:
        return None, None
    if idx < 20:
        return None, None
    
    # Parameters
    atr_period = config.get('supertrend_atr_period', 10)
    multiplier = config.get('supertrend_multiplier', 3.0)
    adx_threshold = config.get('supertrend_adx_threshold', 25)
    volume_mult = config.get('supertrend_volume_mult', 1.5)
    
    # Calculate SuperTrend
    if 'SuperTrend' not in df.columns or 'SuperTrend_Direction' not in df.columns:
        # Calculate ATR
        df['ATR_ST'] = calculate_atr(df, atr_period)
        
        # Calculate basic bands
        hl_avg = (df['High'] + df['Low']) / 2
        df['ST_Upper'] = hl_avg + (multiplier * df['ATR_ST'])
        df['ST_Lower'] = hl_avg - (multiplier * df['ATR_ST'])
        
        # Calculate final SuperTrend
        supertrend = []
        direction = []
        
        for i in range(len(df)):
            if i == 0:
                supertrend.append(df['ST_Lower'].iloc[i])
                direction.append(1)
            else:
                prev_st = supertrend[i-1]
                prev_dir = direction[i-1]
                close = df['Close'].iloc[i]
                upper = df['ST_Upper'].iloc[i]
                lower = df['ST_Lower'].iloc[i]
                
                # Determine current SuperTrend
                if prev_dir == 1:
                    if close <= prev_st:
                        supertrend.append(upper)
                        direction.append(-1)
                    else:
                        supertrend.append(max(lower, prev_st))
                        direction.append(1)
                else:
                    if close >= prev_st:
                        supertrend.append(lower)
                        direction.append(1)
                    else:
                        supertrend.append(min(upper, prev_st))
                        direction.append(-1)
        
        df['SuperTrend'] = supertrend
        df['SuperTrend_Direction'] = direction
    
    current = df.iloc[idx]
    previous = df.iloc[idx - 1]
    
    # Check for trend change
    curr_dir = current['SuperTrend_Direction']
    prev_dir = previous['SuperTrend_Direction']
    
    # ADX filter
    adx = current.get('ADX', 0)
    if pd.isna(adx) or adx < adx_threshold:
        return None, None
    
    # Volume filter
    if 'Volume' in df.columns:
        vol_ma = df['Volume'].rolling(20).mean().iloc[idx]
        if current['Volume'] < vol_ma * volume_mult:
            return None, None
    
    # Bullish signal
    if prev_dir == -1 and curr_dir == 1:
        return 'BUY', current['Close']
    
    # Bearish signal
    if prev_dir == 1 and curr_dir == -1:
        return 'SELL', current['Close']
    
    return None, None

def check_vwap_volume_spike(df, idx, config, current_position):
    """
    VWAP + Volume Spike Strategy - High probability reversal/momentum strategy
    Triggers when price crosses VWAP with significant volume spike
    """
    if current_position is not None:
        return None, None
    if idx < 50:
        return None, None
    
    # Parameters
    volume_mult = config.get('vwap_volume_mult', 2.0)
    vwap_distance = config.get('vwap_distance_pct', 0.3)  # % distance from VWAP
    rsi_ob = config.get('vwap_rsi_ob', 70)
    rsi_os = config.get('vwap_rsi_os', 30)
    
    # Calculate VWAP
    if 'VWAP' not in df.columns:
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (df['Typical_Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    current = df.iloc[idx]
    previous = df.iloc[idx - 1]
    current_price = current['Close']
    prev_price = previous['Close']
    vwap = current['VWAP']
    
    # Volume spike check
    vol_ma = df['Volume'].rolling(20).mean().iloc[idx]
    if current['Volume'] < vol_ma * volume_mult:
        return None, None
    
    # Distance check (price within acceptable distance from VWAP)
    distance_pct = abs(current_price - vwap) / vwap * 100
    if distance_pct > vwap_distance:
        return None, None
    
    # RSI confirmation
    rsi = current.get('RSI', 50)
    
    # Bullish: Price crosses above VWAP with volume spike + RSI oversold recovery
    if prev_price < previous['VWAP'] and current_price > vwap:
        if not pd.isna(rsi) and rsi < 55:  # Not overbought
            return 'BUY', current_price
    
    # Bearish: Price crosses below VWAP with volume spike + RSI overbought
    if prev_price > previous['VWAP'] and current_price < vwap:
        if not pd.isna(rsi) and rsi > 45:  # Not oversold
            return 'SELL', current_price
    
    return None, None

def check_bollinger_squeeze_breakout(df, idx, config, current_position):
    """
    Bollinger Band Squeeze Breakout - High probability breakout strategy
    Identifies low volatility squeeze periods followed by explosive breakouts
    """
    if current_position is not None:
        return None, None
    if idx < 30:
        return None, None
    
    # Parameters
    bb_period = config.get('bb_squeeze_period', 20)
    bb_std = config.get('bb_squeeze_std', 2.0)
    squeeze_threshold = config.get('bb_squeeze_threshold', 0.02)  # 2% bandwidth
    volume_mult = config.get('bb_squeeze_volume_mult', 1.8)
    
    # Calculate Bollinger Bands
    if f'BB_Upper_{bb_period}' not in df.columns:
        bb_mid = df['Close'].rolling(bb_period).mean()
        bb_std_val = df['Close'].rolling(bb_period).std()
        df[f'BB_Upper_{bb_period}'] = bb_mid + (bb_std * bb_std_val)
        df[f'BB_Lower_{bb_period}'] = bb_mid - (bb_std * bb_std_val)
        df[f'BB_Mid_{bb_period}'] = bb_mid
        
        # Calculate bandwidth (normalized)
        df[f'BB_Bandwidth_{bb_period}'] = (df[f'BB_Upper_{bb_period}'] - df[f'BB_Lower_{bb_period}']) / df[f'BB_Mid_{bb_period}']
    
    current = df.iloc[idx]
    previous = df.iloc[idx - 1]
    current_price = current['Close']
    prev_price = previous['Close']
    
    bb_upper = current[f'BB_Upper_{bb_period}']
    bb_lower = current[f'BB_Lower_{bb_period}']
    bb_mid = current[f'BB_Mid_{bb_period}']
    bandwidth = current[f'BB_Bandwidth_{bb_period}']
    prev_bandwidth = previous[f'BB_Bandwidth_{bb_period}']
    
    # Check if squeeze is present (low volatility)
    is_squeezed = bandwidth < squeeze_threshold
    was_squeezed = prev_bandwidth < squeeze_threshold
    
    # Volume confirmation
    vol_ma = df['Volume'].rolling(20).mean().iloc[idx]
    volume_surge = current['Volume'] > vol_ma * volume_mult
    
    # Bullish breakout: Price breaks above upper band during/after squeeze
    if (is_squeezed or was_squeezed) and volume_surge:
        if prev_price <= previous[f'BB_Upper_{bb_period}'] and current_price > bb_upper:
            # Additional confirmation: RSI not extremely overbought
            rsi = current.get('RSI', 50)
            if pd.isna(rsi) or rsi < 75:
                return 'BUY', current_price
    
    # Bearish breakout: Price breaks below lower band during/after squeeze
    if (is_squeezed or was_squeezed) and volume_surge:
        if prev_price >= previous[f'BB_Lower_{bb_period}'] and current_price < bb_lower:
            # Additional confirmation: RSI not extremely oversold
            rsi = current.get('RSI', 50)
            if pd.isna(rsi) or rsi > 25:
                return 'SELL', current_price
    
    return None, None

def check_elliott_waves_ratio_charts(df, idx, config, current_position):
    """
    Elliott Waves Strategy (Simplified with argrelextrema)
    Detects extrema points and identifies 5-wave patterns
    """
    if current_position is not None:
        return None, None
    
    # Parameters
    wave_lookback = config.get('elliott_wave_lookback', 50)
    
    if idx < wave_lookback:
        return None, None
    
    # Calculate extrema if not already done
    if 'Wave_Extrema' not in df.columns:
        df['Wave_Extrema'] = 0
        
        # Get recent data for analysis
        if len(df) >= wave_lookback:
            # Find local maxima and minima
            highs_idx = argrelextrema(df['High'].values, np.greater, order=5)[0]
            lows_idx = argrelextrema(df['Low'].values, np.less, order=5)[0]
            
            # Mark extrema points
            for h_idx in highs_idx:
                if h_idx < len(df):
                    df.iloc[h_idx, df.columns.get_loc('Wave_Extrema')] = 1
            
            for l_idx in lows_idx:
                if l_idx < len(df):
                    df.iloc[l_idx, df.columns.get_loc('Wave_Extrema')] = 1
    
    # Get recent window for wave detection
    recent_start = max(0, idx - wave_lookback)
    recent = df.iloc[recent_start:idx+1]
    
    # Find extrema indices in recent window
    extrema_mask = recent['Wave_Extrema'] == 1
    extrema_indices = recent[extrema_mask].index.tolist()
    
    bullish = False
    bearish = False
    
    # Need at least 5 extrema points for wave pattern
    if len(extrema_indices) >= 5:
        # Get last 5 extrema prices
        wave_prices = df.loc[extrema_indices[-5:], 'Close'].values
        
        # Simplified wave detection pattern: High-Low-High-Low-High (or reverse)
        # Bullish: Wave completes with lower low (wave 4 < wave 2)
        if (wave_prices[0] < wave_prices[1] > wave_prices[2] < wave_prices[3] > wave_prices[4]):
            bullish = wave_prices[4] < wave_prices[2]
            bearish = wave_prices[4] > wave_prices[2]
    
    # Generate signals
    if bullish:
        return 'BUY', df.iloc[idx]['Close']
    elif bearish:
        return 'SELL', df.iloc[idx]['Close']
    
    return None, None

# ================================
# HELPER FUNCTIONS
# ================================

def is_within_trade_window(timestamp, config):
    """
    Check if timestamp is within configured trade window.
    Returns True if trade window is disabled or timestamp is within window.
    """
    if not config.get('use_trade_window', False):
        return True
    
    try:
        # Get IST time
        IST = pytz.timezone('Asia/Kolkata')
        if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
            ts_ist = timestamp.astimezone(IST)
        else:
            ts_ist = IST.localize(timestamp) if not hasattr(timestamp, 'tzinfo') else timestamp
        
        start_time = config.get('trade_window_start', datetime.strptime("09:30", "%H:%M").time())
        end_time = config.get('trade_window_end', datetime.strptime("15:00", "%H:%M").time())
        
        current_time = ts_ist.time()
        return start_time <= current_time <= end_time
    except Exception:
        return True  # Default to allowing trade if error

def should_allow_trade_direction(signal, config):
    """
    Check if signal matches allowed trade direction filter.
    Returns True if signal is allowed, False otherwise.
    """
    direction_filter = config.get('trade_direction', 'Both (LONG + SHORT)')
    
    if direction_filter == 'Both (LONG + SHORT)':
        return True
    elif direction_filter == 'LONG Only':
        return signal in ('BUY', 'LONG')
    elif direction_filter == 'SHORT Only':
        return signal in ('SELL', 'SHORT')
    
    return True  # Default to allowing all

def calculate_brokerage(entry_price, exit_price, quantity, config):
    """
    Calculate brokerage and return Net P&L after brokerage.
    Returns brokerage amount.
    """
    if not config.get('include_brokerage', False):
        return 0.0
    
    brokerage_type = config.get('brokerage_type', 'Fixed per Trade')
    
    if brokerage_type == 'Fixed per Trade':
        return float(config.get('brokerage_per_trade', 20.0))
    else:  # Percentage of Turnover
        turnover = (entry_price + exit_price) * quantity
        brokerage_pct = float(config.get('brokerage_percentage', 0.03)) / 100  # Convert % to decimal
        return turnover * brokerage_pct

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
    'SuperTrend AI': check_supertrend_ai,
    'VWAP + Volume Spike': check_vwap_volume_spike,
    'Bollinger Squeeze Breakout': check_bollinger_squeeze_breakout,
    'Elliott Waves + Ratio Charts': check_elliott_waves_ratio_charts,
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
    
    elif sl_type == 'Strategy-based Signal':
        # No price-based SL, exits only on strategy signal
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
    
    elif target_type == 'Dynamic Trailing SL+Target (Lock Profits)':
        # Initial target is based on entry price + target distance
        target_distance = config.get('dynamic_trail_target_points', 20)
        
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
                         '50% Exit at Target (Partial)']:
        # Start with initial target
        points = config.get('target_points', 20)
        if position_type == 'LONG':
            return entry_price + points
        else:  # SHORT
            return entry_price - points
    
    elif target_type == 'Signal-based (Reverse Crossover)':
        # No price-based target - exit only on reverse signal
        # Return None to disable price target checks
        return None
    
    elif target_type == 'Strategy-based Signal':
        # No price-based target - exit only on strategy signal
        # Return None to disable price target checks
        return None
    
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
    - Dynamic Trailing SL+Target (trails with price)
    
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
    target_type = config.get('target_type', 'Custom Points')
    position_type = position['type']
    current_sl = position['sl_price']
    entry_price = position['entry_price']
    current = df.iloc[idx]
    
    # ============================================
    # DYNAMIC TRAILING SL+TARGET
    # ============================================
    if target_type == 'Dynamic Trailing SL+Target (Lock Profits)':
        sl_distance = config.get('dynamic_trail_sl_points', 10)
        
        if position_type == 'LONG':
            # SL trails below current price
            new_sl = current_price - sl_distance
            # Only move SL up, never down
            return max(current_sl, new_sl) if current_sl is not None else new_sl
        else:  # SHORT
            # SL trails above current price
            new_sl = current_price + sl_distance
            # Only move SL down, never up
            return min(current_sl, new_sl) if current_sl is not None else new_sl
    
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
    
    elif target_type == 'Dynamic Trailing SL+Target (Lock Profits)':
        target_distance = config.get('dynamic_trail_target_points', 20)
        
        if position_type == 'LONG':
            # Target trails above current price
            new_target = current_price + target_distance
            # Only move target up, never down
            return max(current_target, new_target) if current_target is not None else new_target
        else:  # SHORT
            # Target trails below current price
            new_target = current_price - target_distance
            # Only move target down, never up
            return min(current_target, new_target) if current_target is not None else new_target
    
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
        tuple: (trades_list, metrics_dict, debug_info, skipped_trades_list)
    """
    trades = []
    skipped_trades = []  # Track overlapping/skipped signals
    position = None
    strategy_name = config.get('strategy', 'EMA Crossover')
    strategy_func = STRATEGY_FUNCTIONS.get(strategy_name, check_ema_crossover_strategy)
    prevent_overlapping = config.get('prevent_overlapping_trades', True)
    
    # Debug tracking
    total_candles = len(df)
    candles_analyzed = 0
    signals_generated = 0
    trades_entered = 0
    trades_exited = 0
    signals_skipped = 0
    
    # Start from a reasonable index to ensure indicators are calculated
    start_idx = max(50, config.get('ema_slow', 21) + 10)
    
    for idx in range(start_idx, len(df)):
        candles_analyzed += 1
        current_data = df.iloc[idx]
        current_price = current_data['Close']
        
        if position is None:
            # Check if within trade window before checking for entry
            if not is_within_trade_window(current_data['Datetime'], config):
                continue  # Skip entry check if outside trade window
            
            # Check for entry signal
            signal, entry_price = strategy_func(df, idx, config, None)
            
            if signal:
                # Check if signal matches trade direction filter
                if not should_allow_trade_direction(signal, config):
                    continue  # Skip this signal if direction not allowed
                
                signals_generated += 1
                
                # Convert signal to position type
                position_type = 'LONG' if signal in ('BUY', 'LONG') else 'SHORT'
                
                # ── Method 2: Use next candle's open as entry (more realistic) ─────
                use_method2 = config.get('use_backtest_method2', False)
                if use_method2:
                    # Check if next candle exists
                    if idx + 1 < len(df):
                        actual_entry_price = df.iloc[idx + 1]['Open']
                        actual_entry_time = df.iloc[idx + 1]['Datetime']
                    else:
                        # No next candle - skip this trade
                        continue
                else:
                    # Method 1: Use signal price (current close)
                    actual_entry_price = entry_price
                    actual_entry_time = current_data['Datetime']
                
                # Calculate initial SL and Target
                sl_price = calculate_initial_sl(position_type, actual_entry_price, df, idx, config)
                target_price = calculate_initial_target(position_type, actual_entry_price, df, idx, config)
                
                # Collect detailed entry metrics
                entry_metrics = {
                    'strategy': strategy_name,
                    'entry_idx': idx,
                }
                
                # EMA-specific metrics (if applicable)
                if strategy_name == 'EMA Crossover' or 'EMA_Fast' in df.columns:
                    ema_fast_period = config.get('ema_fast', 9)
                    ema_slow_period = config.get('ema_slow', 21)
                    
                    ema_fast_val = current_data.get('EMA_Fast', np.nan)
                    ema_slow_val = current_data.get('EMA_Slow', np.nan)
                    ema_angle = current_data.get('EMA_Fast_Angle', np.nan)
                    
                    entry_metrics.update({
                        'ema_fast_period': ema_fast_period,
                        'ema_slow_period': ema_slow_period,
                        'ema_fast_entry': ema_fast_val,
                        'ema_slow_entry': ema_slow_val,
                        'ema_angle_entry': ema_angle,
                        'price_fast_ema_diff_entry': actual_entry_price - ema_fast_val if not pd.isna(ema_fast_val) else np.nan,
                        'price_slow_ema_diff_entry': actual_entry_price - ema_slow_val if not pd.isna(ema_slow_val) else np.nan,
                        'fast_slow_ema_diff_entry': ema_fast_val - ema_slow_val if not pd.isna(ema_fast_val) and not pd.isna(ema_slow_val) else np.nan,
                    })
                
                # Create position
                position = {
                    'type': position_type,
                    'entry_price': actual_entry_price,
                    'entry_time': actual_entry_time,
                    'sl_price': sl_price,
                    'target_price': target_price,
                    'quantity': config.get('quantity', 1),
                    'highest_price': actual_entry_price,
                    'lowest_price': actual_entry_price,
                    'entry_metrics': entry_metrics,
                }
                
                trades_entered += 1
        
        else:
            # Position exists - check if new signal came (overlapping scenario)
            if prevent_overlapping:
                signal, signal_price = strategy_func(df, idx, config, None)
                if signal and should_allow_trade_direction(signal, config):
                    # New signal while position active - track as skipped
                    signals_skipped += 1
                    
                    # Calculate what P&L would have been for this skipped trade
                    # (simulate exit at some future point for analysis)
                    skipped_position_type = 'LONG' if signal in ('BUY', 'LONG') else 'SHORT'
                    skipped_sl = calculate_initial_sl(skipped_position_type, signal_price, df, idx, config)
                    skipped_target = calculate_initial_target(skipped_position_type, signal_price, df, idx, config)
                    
                    # Find when this trade would have exited (next 50 candles or actual signal resolution)
                    skipped_exit_price = None
                    skipped_exit_reason = None
                    skipped_exit_time = None
                    
                    for future_idx in range(idx + 1, min(idx + 100, len(df))):
                        future_price = df.iloc[future_idx]['Close']
                        future_time = df.iloc[future_idx]['Datetime']
                        
                        # Check SL hit
                        if skipped_sl is not None:
                            if skipped_position_type == 'LONG' and future_price <= skipped_sl:
                                skipped_exit_price = skipped_sl
                                skipped_exit_reason = 'SL Hit (Skipped)'
                                skipped_exit_time = future_time
                                break
                            elif skipped_position_type == 'SHORT' and future_price >= skipped_sl:
                                skipped_exit_price = skipped_sl
                                skipped_exit_reason = 'SL Hit (Skipped)'
                                skipped_exit_time = future_time
                                break
                        
                        # Check Target hit
                        if skipped_target is not None:
                            if skipped_position_type == 'LONG' and future_price >= skipped_target:
                                skipped_exit_price = skipped_target
                                skipped_exit_reason = 'Target Hit (Skipped)'
                                skipped_exit_time = future_time
                                break
                            elif skipped_position_type == 'SHORT' and future_price <= skipped_target:
                                skipped_exit_price = skipped_target
                                skipped_exit_reason = 'Target Hit (Skipped)'
                                skipped_exit_time = future_time
                                break
                    
                    # If no exit found, use current+20 candles as exit
                    if skipped_exit_price is None and idx + 20 < len(df):
                        skipped_exit_price = df.iloc[idx + 20]['Close']
                        skipped_exit_reason = 'Simulated Exit (Skipped)'
                        skipped_exit_time = df.iloc[idx + 20]['Datetime']
                    
                    if skipped_exit_price:
                        if skipped_position_type == 'LONG':
                            skipped_pnl = (skipped_exit_price - signal_price) * config.get('quantity', 1)
                        else:
                            skipped_pnl = (signal_price - skipped_exit_price) * config.get('quantity', 1)
                        
                        brokerage = calculate_brokerage(signal_price, skipped_exit_price, config.get('quantity', 1), config)
                        
                        skipped_trades.append({
                            'entry_time': current_data['Datetime'],
                            'exit_time': skipped_exit_time,
                            'type': skipped_position_type,
                            'entry_price': signal_price,
                            'exit_price': skipped_exit_price,
                            'sl_price': skipped_sl,
                            'target_price': skipped_target,
                            'quantity': config.get('quantity', 1),
                            'pnl': skipped_pnl,
                            'brokerage': brokerage,
                            'net_pnl': skipped_pnl - brokerage,
                            'exit_reason': skipped_exit_reason,
                            'note': 'Overlapped with active trade'
                        })
            
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
            
            # Check if outside trade window - force exit
            if not is_within_trade_window(current_data['Datetime'], config):
                exit_reason = 'Trade Window Closed'
                exit_price = current_price
            
            # Check SL hit
            if exit_reason is None and position['sl_price'] is not None:
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
            
            # Check signal-based exit (for both SL and Target types)
            if exit_reason is None and (
                config.get('sl_type') == 'Signal-based (Reverse Crossover)' or 
                config.get('target_type') == 'Signal-based (Reverse Crossover)' or
                config.get('sl_type') == 'Strategy-based Signal' or
                config.get('target_type') == 'Strategy-based Signal'
            ):
                signal, _ = strategy_func(df, idx, config, position)
                if signal:
                    # Check for opposite signal (reverse)
                    if (position['type'] == 'LONG' and signal in ('SELL', 'SHORT')) or \
                       (position['type'] == 'SHORT' and signal in ('BUY', 'LONG')):
                        exit_reason = 'Strategy Signal Exit'
                        exit_price = current_price
            
            # Exit position if conditions met
            if exit_reason:
                # Calculate P&L
                if position['type'] == 'LONG':
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                else:  # SHORT
                    pnl = (position['entry_price'] - exit_price) * position['quantity']
                
                # Calculate brokerage
                brokerage = calculate_brokerage(position['entry_price'], exit_price, position['quantity'], config)
                net_pnl = pnl - brokerage
                
                # Calculate trade duration
                duration_seconds = (current_data['Datetime'] - position['entry_time']).total_seconds()
                duration_minutes = duration_seconds / 60
                
                # Collect exit metrics
                exit_metrics = {}
                if strategy_name == 'EMA Crossover' or 'EMA_Fast' in df.columns:
                    ema_fast_val_exit = current_data.get('EMA_Fast', np.nan)
                    ema_slow_val_exit = current_data.get('EMA_Slow', np.nan)
                    
                    exit_metrics.update({
                        'ema_fast_exit': ema_fast_val_exit,
                        'ema_slow_exit': ema_slow_val_exit,
                        'price_fast_ema_diff_exit': exit_price - ema_fast_val_exit if not pd.isna(ema_fast_val_exit) else np.nan,
                        'price_slow_ema_diff_exit': exit_price - ema_slow_val_exit if not pd.isna(ema_slow_val_exit) else np.nan,
                        'fast_slow_ema_diff_exit': ema_fast_val_exit - ema_slow_val_exit if not pd.isna(ema_fast_val_exit) and not pd.isna(ema_slow_val_exit) else np.nan,
                    })
                
                # Merge entry and exit metrics
                entry_metrics = position.get('entry_metrics', {})
                
                # Record trade with detailed metrics
                trade = {
                    'entry_time': position['entry_time'],
                    'exit_time': current_data['Datetime'],
                    'type': position['type'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'highest_price': position['highest_price'],
                    'lowest_price': position['lowest_price'],
                    'sl_price': position['sl_price'],
                    'target_price': position['target_price'],
                    'quantity': position['quantity'],
                    'pnl': pnl,
                    'brokerage': brokerage,
                    'net_pnl': net_pnl,
                    'exit_reason': exit_reason,
                    'duration_minutes': duration_minutes,
                    'strategy': entry_metrics.get('strategy', strategy_name),
                    'ema_fast_period': entry_metrics.get('ema_fast_period', np.nan),
                    'ema_slow_period': entry_metrics.get('ema_slow_period', np.nan),
                    'ema_angle_entry': entry_metrics.get('ema_angle_entry', np.nan),
                    'ema_fast_entry': entry_metrics.get('ema_fast_entry', np.nan),
                    'ema_slow_entry': entry_metrics.get('ema_slow_entry', np.nan),
                    'price_fast_ema_diff_entry': entry_metrics.get('price_fast_ema_diff_entry', np.nan),
                    'price_slow_ema_diff_entry': entry_metrics.get('price_slow_ema_diff_entry', np.nan),
                    'fast_slow_ema_diff_entry': entry_metrics.get('fast_slow_ema_diff_entry', np.nan),
                    'ema_fast_exit': exit_metrics.get('ema_fast_exit', np.nan),
                    'ema_slow_exit': exit_metrics.get('ema_slow_exit', np.nan),
                    'price_fast_ema_diff_exit': exit_metrics.get('price_fast_ema_diff_exit', np.nan),
                    'price_slow_ema_diff_exit': exit_metrics.get('price_slow_ema_diff_exit', np.nan),
                    'fast_slow_ema_diff_exit': exit_metrics.get('fast_slow_ema_diff_exit', np.nan),
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
        
        # Net P&L calculations
        total_brokerage = df_trades['brokerage'].sum() if 'brokerage' in df_trades.columns else 0
        total_net_pnl = df_trades['net_pnl'].sum() if 'net_pnl' in df_trades.columns else total_pnl
        avg_net_pnl = df_trades['net_pnl'].mean() if 'net_pnl' in df_trades.columns else avg_pnl
        
        # Calculate max drawdown (using net P&L)
        pnl_column = 'net_pnl' if 'net_pnl' in df_trades.columns else 'pnl'
        cumulative_pnl = df_trades[pnl_column].cumsum()
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
            'total_brokerage': total_brokerage,
            'total_net_pnl': total_net_pnl,
            'avg_net_pnl': avg_net_pnl,
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
        'signals_skipped': signals_skipped,
        'overlapping_trades': len(skipped_trades),
    }
    
    return trades, metrics, debug_info, skipped_trades

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
    
    # Check if clearing is in progress - skip iteration
    if st.session_state.get('clearing_in_progress', False):
        add_log(f"🏦 ⏳ Position clearing in progress - skipping iteration")
        return
    
    # Fetch fresh data
    position = st.session_state.get('position')
    
    # Use ticker from active position if exists, otherwise from config
    if position is not None and 'ticker' in position:
        ticker = position['ticker']
        custom_ticker = position.get('custom_ticker')
        add_log(f"📊 Using locked ticker from position: {ticker}")
    else:
        ticker = config.get('asset', 'NIFTY 50')
        custom_ticker = config.get('custom_ticker', None)
    
    interval = INTERVAL_MAPPING.get(config.get('interval', '1 day'), '1d')
    period = PERIOD_MAPPING.get(config.get('period', '1 month'), '1mo')
    
    # Enhanced Live Trading Mode
    enhanced_mode = config.get('enhanced_live_trading', False)
    
    if enhanced_mode:
        # Check if current time is at round interval
        current_time = datetime.now(pytz.timezone('Asia/Kolkata'))
        
        # Determine interval minutes
        interval_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60
        }.get(interval, 1)
        
        # Check if at round time (e.g., 10:30:00, 10:31:00 for 1m)
        is_round_time = (current_time.minute % interval_minutes == 0 and current_time.second < 5)
        
        # Get last candlestick fetch time
        last_candle_fetch = st.session_state.get('last_candle_fetch_time')
        
        # Fetch candlestick data if at round time and haven't fetched recently
        if is_round_time and (last_candle_fetch is None or 
                              (current_time - last_candle_fetch).total_seconds() >= interval_minutes * 60):
            add_log(f"🔄 Enhanced Mode: Fetching candlestick data at round time {current_time.strftime('%H:%M:%S')}")
            
            # Fetch candlestick data for indicators
            df_candles = fetch_data(ticker, interval, period, is_live_trading=True, custom_ticker=custom_ticker)
            
            if df_candles is not None and not df_candles.empty:
                # Calculate indicators on candlestick data
                df_candles = calculate_all_indicators(df_candles, config)
                
                # Store in session for use
                st.session_state['indicator_df'] = df_candles
                st.session_state['last_candle_fetch_time'] = current_time
                add_log(f"✅ Indicators calculated from {len(df_candles)} candles")
        
        # Get indicator dataframe (use stored if available, otherwise fetch new)
        df = st.session_state.get('indicator_df')
        
        if df is None or df.empty:
            add_log("⚠️ No indicator data available, fetching fresh data")
            df = fetch_data(ticker, interval, period, is_live_trading=True, custom_ticker=custom_ticker)
            if df is None or df.empty:
                add_log("❌ Failed to fetch data")
                return
            df = calculate_all_indicators(df, config)
            st.session_state['indicator_df'] = df
        
        # Fetch separate live data for current price (SL/Target checks)
        df_live = fetch_data(ticker, interval, period, is_live_trading=True, custom_ticker=custom_ticker)
        
        if df_live is None or df_live.empty:
            add_log("❌ Failed to fetch live data")
            return
        
        # Use live data for current price
        current_data = df_live.iloc[-1].copy()
        current_price = current_data['Close']
        
        # But use indicator data for strategy checks
        idx = len(df) - 1
        
    else:
        # Standard mode: fetch once and use for everything
        df = fetch_data(ticker, interval, period, is_live_trading=True, custom_ticker=custom_ticker)
        
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
        # Check if within trade window
        if not is_within_trade_window(current_data['Datetime'], config):
            add_log(f"⏰ Outside trade window - no new entries allowed")
            return
        
        # Check cooldown period after previous exit (prevent immediate re-entry)
        last_exit_time = st.session_state.get('last_exit_time')
        cooldown_seconds = config.get('entry_cooldown_seconds', 60)  # Default 60 seconds
        
        if last_exit_time is not None:
            time_since_exit = (datetime.now(pytz.timezone('Asia/Kolkata')) - last_exit_time).total_seconds()
            if time_since_exit < cooldown_seconds:
                add_log(f"⏳ Cooldown active: {int(cooldown_seconds - time_since_exit)}s remaining (prevents duplicate entries)")
                return
        
        # Check if signal is different from last signal (prevent same-signal re-entry)
        last_signal = st.session_state.get('last_signal_type')
        
        # Check for entry signal
        add_log(f"🔍 Checking for entry signal using {strategy_name}...")
        signal, entry_price = strategy_func(df, idx, config, None)
        
        if signal:
            # Prevent re-entry with same signal type immediately after exit
            if last_signal == signal:
                add_log(f"⛔ Same signal {signal} as last trade - waiting for new crossover/condition")
                return
            
            # Check if signal matches trade direction filter
            if not should_allow_trade_direction(signal, config):
                direction_filter = config.get('trade_direction', 'Both (LONG + SHORT)')
                add_log(f"⛔ Signal {signal} filtered out by direction setting: {direction_filter}")
                return
            
            # Check if broker already has an active position - prevent duplicate orders
            if config.get('dhan_enabled', False):
                existing_broker_pos = st.session_state.get('broker_position')
                if existing_broker_pos:
                    add_log(f"⛔ Broker order already active - preventing duplicate order")
                    return
            
            add_log(f"✅ SIGNAL DETECTED: {signal} at {entry_price:.2f}")
            
            # Store current signal type
            st.session_state['last_signal_type'] = signal
            
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
                'ticker': config.get('asset', 'NIFTY 50'),  # Store ticker with position
                'custom_ticker': config.get('custom_ticker'),  # Store custom ticker too
            }
            
            st.session_state['position'] = position
            add_log(f"✅ Position created in session state")
            add_log(f"📊 Locked ticker: {position['ticker']}")
            
            # Place broker order if enabled
            if config.get('dhan_enabled', False):
                add_log(f"🏦 Dhan broker enabled, attempting to place order...")
                dhan_broker = st.session_state.get('dhan_broker')
                if dhan_broker:
                    try:
                        # Clear all positions before new entry if enabled
                        if config.get('clear_positions_before_entry', False):
                            add_log(f"🏦 🧹 Clearing all existing positions before new entry...")
                            
                            # Set clearing flag to block concurrent entries
                            st.session_state['clearing_in_progress'] = True
                            
                            clear_result = dhan_broker.clear_all_positions(add_log, convert_to_market=True)
                            
                            # Check if clearing completed successfully
                            if not clear_result['clearing_complete']:
                                add_log(f"🏦 ⚠️ Position clearing incomplete - aborting new entry")
                                st.session_state['clearing_in_progress'] = False
                                return
                            
                            # Log results
                            total_cleared = clear_result['cancelled_orders'] + clear_result['converted_orders'] + clear_result['closed_positions']
                            if total_cleared > 0:
                                add_log(f"🏦 ✅ Cleared {total_cleared} orders/positions ({clear_result['cancelled_orders']} cancelled, {clear_result['converted_orders']} converted, {clear_result['closed_positions']} closed)")
                            else:
                                add_log(f"🏦 ℹ️ No existing positions to clear")
                            
                            # Clear flag
                            st.session_state['clearing_in_progress'] = False
                        
                        # Now place the new order
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
        
        # Check if outside trade window - force exit
        if not is_within_trade_window(current_data['Datetime'], config):
            exit_reason = 'Trade Window Closed'
            exit_price = current_price
            add_log(f"⏰ TRADE WINDOW CLOSED - Force exiting position")
        
        # Check SL hit
        if exit_reason is None and position['sl_price'] is not None:
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
        
        # Check signal-based exit (for both SL and Target types)
        if exit_reason is None and (
            config.get('sl_type') == 'Signal-based (Reverse Crossover)' or 
            config.get('target_type') == 'Signal-based (Reverse Crossover)' or
            config.get('sl_type') == 'Strategy-based Signal' or
            config.get('target_type') == 'Strategy-based Signal'
        ):
            signal, _ = strategy_func(df, idx, config, position)
            if signal:
                if (position['type'] == 'LONG' and signal in ('SELL', 'SHORT')) or \
                   (position['type'] == 'SHORT' and signal in ('BUY', 'LONG')):
                    exit_reason = 'Strategy Signal Exit'
                    exit_price = current_price
                    add_log(f"🔄 STRATEGY SIGNAL DETECTED: {signal}")
        
        # Exit if conditions met
        if exit_reason:
            # Calculate P&L
            if position['type'] == 'LONG':
                pnl = (exit_price - position['entry_price']) * position['quantity']
            else:  # SHORT
                pnl = (position['entry_price'] - exit_price) * position['quantity']
            
            # Calculate brokerage and net P&L
            brokerage = calculate_brokerage(position['entry_price'], exit_price, position['quantity'], config)
            net_pnl = pnl - brokerage
            
            add_log(f"🚪 EXITING POSITION: {exit_reason} @ {exit_price:.2f}")
            add_log(f"💰 P&L: ₹{pnl:.2f}")
            if config.get('include_brokerage', False):
                add_log(f"💸 Brokerage: ₹{brokerage:.2f}")
                add_log(f"💵 Net P&L: ₹{net_pnl:.2f}")
            
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
                'brokerage': brokerage,
                'net_pnl': net_pnl,
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
            
            # Store exit time for cooldown
            st.session_state['last_exit_time'] = datetime.now(pytz.timezone('Asia/Kolkata'))
            
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
    
    # Custom Ticker Input
    if config['asset'] == 'Custom Ticker':
        config['custom_ticker'] = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL", help="e.g., AAPL, TSLA, MSFT")
    
    # Timeframe
    config['interval'] = st.sidebar.selectbox("Interval", list(INTERVAL_MAPPING.keys()), index=0)  # Default to 1 minute
    config['period'] = st.sidebar.selectbox("Period", list(PERIOD_MAPPING.keys()), index=0)  # Default to 1 day
    
    # Quantity
    config['quantity'] = st.sidebar.number_input("Quantity", min_value=1, value=1)
    
    # ── Trade Window Settings ────────────────────────────────────────────────
    st.sidebar.subheader("⏰ Trade Window")
    config['use_trade_window'] = st.sidebar.checkbox(
        "Enable Trade Window",
        value=False,
        help="Restricts trading to specific hours. Exits positions and blocks new entries outside this window."
    )
    if config['use_trade_window']:
        col_tw1, col_tw2 = st.sidebar.columns(2)
        with col_tw1:
            config['trade_window_start'] = st.sidebar.time_input(
                "Start Time (IST)",
                value=datetime.strptime("09:30", "%H:%M").time(),
                help="Trading starts at this time"
            )
        with col_tw2:
            config['trade_window_end'] = st.sidebar.time_input(
                "End Time (IST)",
                value=datetime.strptime("15:00", "%H:%M").time(),
                help="Trading ends at this time. Positions auto-exit after this."
            )
        st.sidebar.info(f"🕐 Active: {config['trade_window_start'].strftime('%H:%M')} - {config['trade_window_end'].strftime('%H:%M')} IST")
    
    # ── Trade Direction Filter ───────────────────────────────────────────────
    config['trade_direction'] = st.sidebar.selectbox(
        "Trade Direction Filter",
        ["Both (LONG + SHORT)", "LONG Only", "SHORT Only"],
        index=0,
        help="Filters which trade directions the algo will take"
    )
    
    # ── Brokerage Configuration (Available Always) ───────────────────────────
    st.sidebar.subheader("💰 Brokerage & Charges")
    config['include_brokerage'] = st.sidebar.checkbox(
        "Include Brokerage & Charges",
        value=False,
        help="Deducts brokerage from P&L to show Net P&L (works in backtesting and live trading)"
    )
    if config['include_brokerage']:
        col_b1, col_b2 = st.sidebar.columns(2)
        with col_b1:
            config['brokerage_per_trade'] = st.sidebar.number_input(
                "Brokerage per Trade (₹)",
                min_value=0.0,
                value=20.0,
                step=1.0,
                help="Total brokerage + charges per trade (entry + exit)"
            )
        with col_b2:
            config['brokerage_percentage'] = st.sidebar.number_input(
                "Or % of Turnover",
                min_value=0.0,
                value=0.03,
                step=0.01,
                format="%.3f",
                help="Alternative: % of trade value (0.03% typical for intraday)"
            )
        config['brokerage_type'] = st.sidebar.radio(
            "Brokerage Calculation",
            ["Fixed per Trade", "Percentage of Turnover"],
            index=0,
            horizontal=True
        )
    
    # ── Overlapping Trades Prevention ────────────────────────────────────────
    config['prevent_overlapping_trades'] = st.sidebar.checkbox(
        "🚫 Prevent Overlapping Trades",
        value=True,
        help="When enabled, blocks new signals while a position is active. Skipped signals are tracked separately for analysis."
    )
    
    # ── Entry Cooldown (Live Trading) ────────────────────────────────────────
    config['entry_cooldown_seconds'] = st.sidebar.number_input(
        "⏱️ Entry Cooldown (seconds)",
        min_value=0,
        value=60,
        step=5,
        help="Prevents immediate re-entry after exit (Live Trading only). Useful to avoid duplicate orders on same signal."
    )
    st.sidebar.caption(f"Cooldown: {config['entry_cooldown_seconds']}s wait after exit before new entry")
    
    # ── Enhanced Live Trading Mode ────────────────────────────────────────────
    config['enhanced_live_trading'] = st.sidebar.checkbox(
        "🔄 Enhanced Live Trading (TradingView Match)",
        value=False,
        help=(
            "When enabled, fetches candlestick data at round time intervals matching your timeframe "
            "(e.g., 10:30:00, 10:31:00 for 1min; 10:30:00, 10:35:00 for 5min). "
            "Uses candlestick data for indicators and live data for SL/Target checks. "
            "Ensures indicator values match TradingView."
        )
    )
    if config['enhanced_live_trading']:
        st.sidebar.info(
            "📊 Enhanced Mode Active:\n"
            "- Indicators calculated from round-time candles\n"
            "- Live data used for SL/Target checks\n"
            "- Matches TradingView indicator values"
        )
    
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
        
        # Cross direction
        config['price_cross_type'] = st.sidebar.selectbox(
            "Cross Type",
            ["Above Threshold", "Below Threshold"],
            help="When should the signal trigger?"
        )
        
        # Position direction
        config['price_cross_position'] = st.sidebar.selectbox(
            "Position Type",
            ["LONG", "SHORT"],
            help="What position to take when cross occurs?"
        )
    
    elif config['strategy'] == 'Percentage Change':
        config['pct_change_threshold'] = st.sidebar.number_input("% Change Threshold", min_value=0.001, value=2.0, step=0.001, format="%.3f")
        
        # Change direction
        config['pct_change_type'] = st.sidebar.selectbox(
            "Change Type",
            ["Positive % (Price Up)", "Negative % (Price Down)"],
            help="When should the signal trigger?"
        )
        
        # Position direction
        config['pct_change_position'] = st.sidebar.selectbox(
            "Position Type",
            ["LONG", "SHORT"],
            help="What position to take when % change occurs?"
        )
    
    elif config['strategy'] == 'SuperTrend AI':
        st.sidebar.markdown("**SuperTrend AI Parameters**")
        config['supertrend_atr_period'] = st.sidebar.number_input("ATR Period", min_value=5, value=10)
        config['supertrend_multiplier'] = st.sidebar.number_input("Multiplier", min_value=1.0, value=3.0, step=0.5)
        config['supertrend_adx_threshold'] = st.sidebar.number_input("ADX Threshold", min_value=10, value=25)
        config['supertrend_volume_mult'] = st.sidebar.number_input("Volume Multiplier", min_value=1.0, value=1.5, step=0.1)
        st.sidebar.info("📊 Trend-following with SuperTrend + ADX + Volume confirmation")
    
    elif config['strategy'] == 'VWAP + Volume Spike':
        st.sidebar.markdown("**VWAP + Volume Spike Parameters**")
        config['vwap_volume_mult'] = st.sidebar.number_input("Volume Spike Multiplier", min_value=1.5, value=2.0, step=0.1)
        config['vwap_distance_pct'] = st.sidebar.number_input("Max Distance from VWAP (%)", min_value=0.1, value=0.3, step=0.1)
        config['vwap_rsi_ob'] = st.sidebar.number_input("RSI Overbought", min_value=60, max_value=100, value=70)
        config['vwap_rsi_os'] = st.sidebar.number_input("RSI Oversold", min_value=0, max_value=40, value=30)
        st.sidebar.info("📊 Price/VWAP crossover with volume confirmation")
    
    elif config['strategy'] == 'Bollinger Squeeze Breakout':
        st.sidebar.markdown("**Bollinger Squeeze Breakout Parameters**")
        config['bb_squeeze_period'] = st.sidebar.number_input("BB Period", min_value=10, value=20)
        config['bb_squeeze_std'] = st.sidebar.number_input("BB Std Dev", min_value=1.0, value=2.0, step=0.1)
        config['bb_squeeze_threshold'] = st.sidebar.number_input("Squeeze Threshold (%)", min_value=0.01, value=0.02, step=0.01, format="%.3f")
        config['bb_squeeze_volume_mult'] = st.sidebar.number_input("Breakout Volume Mult", min_value=1.0, value=1.8, step=0.1)
        st.sidebar.info("📊 Low volatility squeeze → High probability breakout")
    
    elif config['strategy'] == 'Elliott Waves + Ratio Charts':
        st.sidebar.markdown("**Elliott Waves Parameters**")
        config['elliott_wave_lookback'] = st.sidebar.number_input("Wave Lookback Period", min_value=20, value=50)
        st.sidebar.info("🌊 Detects extrema points and identifies 5-wave patterns using argrelextrema")
    
    elif config['strategy'] == 'Custom Strategy':
        st.sidebar.markdown("**🛠️ Custom Strategy Builder (Multi-Indicator)**")

        # ── Session-state list of indicator conditions ────────────────────────
        if 'custom_indicator_conditions' not in st.session_state:
            st.session_state['custom_indicator_conditions'] = [{}]   # start with 1 slot

        conditions = st.session_state['custom_indicator_conditions']

        # Combine mode when >1 condition
        if len(conditions) > 1:
            config['custom_combine_mode'] = st.sidebar.radio(
                "Combine Conditions With",
                ["AND (all must be true)", "OR (any one true)"],
                index=0,
                help="AND = all conditions trigger  |  OR = any single condition triggers"
            )
        else:
            config['custom_combine_mode'] = "AND (all must be true)"

        # ── Add / Remove buttons ─────────────────────────────────────────────
        col_add, col_clr = st.sidebar.columns(2)
        with col_add:
            if st.button("➕ Add Condition", key="cust_add"):
                st.session_state['custom_indicator_conditions'].append({})
                st.rerun()
        with col_clr:
            if st.button("🗑️ Clear All", key="cust_clr"):
                st.session_state['custom_indicator_conditions'] = [{}]
                st.rerun()

        # ── Render each condition ─────────────────────────────────────────────
        STRATEGY_TYPE_OPTS = [
            "Price Crosses Indicator",
            "Price Pullback from Indicator",
            "Indicator Crosses Level",
            "Indicator Crossover",
        ]

        PRICE_INDICATOR_OPTS = ["EMA", "SMA", "BB Upper", "BB Lower", "BB Middle"]
        PULLBACK_INDICATOR_OPTS = ["EMA", "SMA", "BB Upper", "BB Lower"]
        LEVEL_INDICATOR_OPTS = [
            "RSI", "MACD", "MACD Histogram", "ADX",
            "Volume", "BB %B",
            "ATR (Volatility)", "Historical Volatility", "Std Dev (Volatility)"
        ]
        CROSSOVER_OPTS = [
            "Fast EMA × Slow EMA", "Fast SMA × Slow SMA",
            "MACD × Signal", "Price × EMA", "Price × SMA",
            "RSI Crossover (Overbought/Oversold)"
        ]

        rendered_conditions = []
        for i, cond in enumerate(conditions):
            st.sidebar.markdown(f"---\n**Condition {i+1}**")

            # Delete button for this condition (not for first if only one)
            if len(conditions) > 1:
                if st.sidebar.button(f"🗑️ Delete #{i+1}", key=f"del_cond_{i}"):
                    st.session_state['custom_indicator_conditions'].pop(i)
                    st.rerun()

            c = {}
            c['strategy_type'] = st.sidebar.selectbox(
                f"Type #{i+1}", STRATEGY_TYPE_OPTS,
                index=STRATEGY_TYPE_OPTS.index(cond.get('strategy_type', STRATEGY_TYPE_OPTS[0])),
                key=f"cst_{i}")

            # ── Price Crosses Indicator ──────────────────────────────────────
            if c['strategy_type'] == "Price Crosses Indicator":
                c['indicator'] = st.sidebar.selectbox(
                    f"Indicator #{i+1}", PRICE_INDICATOR_OPTS,
                    index=PRICE_INDICATOR_OPTS.index(cond.get('indicator', 'EMA'))
                    if cond.get('indicator') in PRICE_INDICATOR_OPTS else 0,
                    key=f"ci_{i}")
                if c['indicator'] in ['EMA', 'SMA']:
                    c['period'] = st.sidebar.number_input(f"Period #{i+1}", min_value=1,
                        value=cond.get('period', 20), key=f"cp_{i}")
                elif 'BB' in c['indicator']:
                    c['bb_period'] = st.sidebar.number_input(f"BB Period #{i+1}", min_value=1,
                        value=cond.get('bb_period', 20), key=f"cbp_{i}")
                    c['bb_std']    = st.sidebar.number_input(f"BB Std #{i+1}", min_value=0.1,
                        value=cond.get('bb_std', 2.0), step=0.1, key=f"cbs_{i}")
                c['cross_type']    = st.sidebar.selectbox(f"Cross #{i+1}",
                    ["Above Indicator", "Below Indicator"],
                    index=0 if cond.get('cross_type','Above') == 'Above Indicator' else 1,
                    key=f"cct_{i}")
                c['position_type'] = st.sidebar.selectbox(f"Position #{i+1}", ["LONG","SHORT"],
                    index=0 if cond.get('position_type','LONG') == 'LONG' else 1,
                    key=f"cpt_{i}")

            # ── Price Pullback ───────────────────────────────────────────────
            elif c['strategy_type'] == "Price Pullback from Indicator":
                c['indicator'] = st.sidebar.selectbox(
                    f"Indicator #{i+1}", PULLBACK_INDICATOR_OPTS,
                    index=PULLBACK_INDICATOR_OPTS.index(cond.get('indicator','EMA'))
                    if cond.get('indicator') in PULLBACK_INDICATOR_OPTS else 0,
                    key=f"ci_{i}")
                if c['indicator'] in ['EMA','SMA']:
                    c['period'] = st.sidebar.number_input(f"Period #{i+1}", min_value=1,
                        value=cond.get('period',20), key=f"cp_{i}")
                elif 'BB' in c['indicator']:
                    c['bb_period'] = st.sidebar.number_input(f"BB Period #{i+1}", min_value=1,
                        value=cond.get('bb_period',20), key=f"cbp_{i}")
                    c['bb_std']    = st.sidebar.number_input(f"BB Std #{i+1}", min_value=0.1,
                        value=cond.get('bb_std',2.0), step=0.1, key=f"cbs_{i}")
                c['pullback_points'] = st.sidebar.number_input(f"Pullback Pts #{i+1}", min_value=0.01,
                    value=float(cond.get('pullback_points',10)), step=0.01, key=f"cpp_{i}")
                c['pullback_side']   = st.sidebar.selectbox(f"Approach #{i+1}",
                    ["Approach from Above","Approach from Below"],
                    index=0 if cond.get('pullback_side','Approach from Above')=='Approach from Above' else 1,
                    key=f"cps_{i}")
                c['position_type'] = st.sidebar.selectbox(f"Position #{i+1}", ["LONG","SHORT"],
                    index=0 if cond.get('position_type','LONG')=='LONG' else 1,
                    key=f"cpt_{i}")

            # ── Indicator Crosses Level ──────────────────────────────────────
            elif c['strategy_type'] == "Indicator Crosses Level":
                c['indicator'] = st.sidebar.selectbox(
                    f"Indicator #{i+1}", LEVEL_INDICATOR_OPTS,
                    index=LEVEL_INDICATOR_OPTS.index(cond.get('indicator','RSI'))
                    if cond.get('indicator') in LEVEL_INDICATOR_OPTS else 0,
                    key=f"ci_{i}")

                ind = c['indicator']
                if ind == 'RSI':
                    c['rsi_period'] = st.sidebar.number_input(f"RSI Period #{i+1}", min_value=1,
                        value=cond.get('rsi_period',14), key=f"crsi_{i}")
                    c['level']      = st.sidebar.number_input(f"Level #{i+1}", min_value=0.0, max_value=100.0,
                        value=float(cond.get('level',50.0)), key=f"clv_{i}")
                elif ind in ['MACD','MACD Histogram']:
                    c['level'] = st.sidebar.number_input(f"Level #{i+1}",
                        value=float(cond.get('level',0.0)), key=f"clv_{i}")
                elif ind == 'ADX':
                    c['level'] = st.sidebar.number_input(f"ADX Level #{i+1}", min_value=0.0,
                        value=float(cond.get('level',25.0)), key=f"clv_{i}")
                elif ind == 'Volume':
                    c['volume_ma_period']   = st.sidebar.number_input(f"Vol MA Period #{i+1}", min_value=1,
                        value=cond.get('volume_ma_period',20), key=f"cvmp_{i}")
                    c['volume_multiplier']  = st.sidebar.number_input(f"Vol Mult #{i+1}", min_value=0.1,
                        value=float(cond.get('volume_multiplier',1.5)), step=0.1, key=f"cvm_{i}")
                elif ind == 'BB %B':
                    c['bb_period'] = st.sidebar.number_input(f"BB Period #{i+1}", min_value=1,
                        value=cond.get('bb_period',20), key=f"cbp_{i}")
                    c['bb_std']    = st.sidebar.number_input(f"BB Std #{i+1}", min_value=0.1,
                        value=float(cond.get('bb_std',2.0)), step=0.1, key=f"cbs_{i}")
                    c['level']     = st.sidebar.number_input(f"%%B Level #{i+1} (0–100)", min_value=0.0, max_value=100.0,
                        value=float(cond.get('level',80.0)), key=f"clv_{i}")
                elif ind == 'ATR (Volatility)':
                    c['atr_period'] = st.sidebar.number_input(f"ATR Period #{i+1}", min_value=1,
                        value=cond.get('atr_period',14), key=f"catr_{i}")
                    c['level']      = st.sidebar.number_input(f"ATR Level #{i+1}", min_value=0.0,
                        value=float(cond.get('level',10.0)), step=0.5, key=f"clv_{i}")
                elif ind == 'Historical Volatility':
                    c['hv_period'] = st.sidebar.number_input(f"HV Period #{i+1} (days)", min_value=5,
                        value=cond.get('hv_period',20), key=f"chv_{i}")
                    c['level']     = st.sidebar.number_input(f"HV Level #{i+1} (%)", min_value=0.0,
                        value=float(cond.get('level',20.0)), step=1.0, key=f"clv_{i}")
                elif ind == 'Std Dev (Volatility)':
                    c['stddev_period'] = st.sidebar.number_input(f"StdDev Period #{i+1}", min_value=2,
                        value=cond.get('stddev_period',20), key=f"csd_{i}")
                    c['level']         = st.sidebar.number_input(f"StdDev Level #{i+1}", min_value=0.0,
                        value=float(cond.get('level',5.0)), step=0.5, key=f"clv_{i}")

                c['cross_type']    = st.sidebar.selectbox(f"Cross #{i+1}",
                    ["Above Level","Below Level"],
                    index=0 if cond.get('cross_type','Above Level')=='Above Level' else 1,
                    key=f"cct_{i}")
                c['position_type'] = st.sidebar.selectbox(f"Position #{i+1}", ["LONG","SHORT"],
                    index=0 if cond.get('position_type','LONG')=='LONG' else 1,
                    key=f"cpt_{i}")

            # ── Indicator Crossover ──────────────────────────────────────────
            elif c['strategy_type'] == "Indicator Crossover":
                c['crossover_type'] = st.sidebar.selectbox(
                    f"Crossover #{i+1}", CROSSOVER_OPTS,
                    index=CROSSOVER_OPTS.index(cond.get('crossover_type', CROSSOVER_OPTS[0]))
                    if cond.get('crossover_type') in CROSSOVER_OPTS else 0,
                    key=f"cco_{i}")
                if c['crossover_type'] == "Fast EMA × Slow EMA":
                    c['fast_ema'] = st.sidebar.number_input(f"Fast EMA #{i+1}", min_value=1,
                        value=cond.get('fast_ema',9), key=f"cfe_{i}")
                    c['slow_ema'] = st.sidebar.number_input(f"Slow EMA #{i+1}", min_value=1,
                        value=cond.get('slow_ema',21), key=f"cse_{i}")
                elif c['crossover_type'] == "Fast SMA × Slow SMA":
                    c['fast_sma'] = st.sidebar.number_input(f"Fast SMA #{i+1}", min_value=1,
                        value=cond.get('fast_sma',20), key=f"cfs_{i}")
                    c['slow_sma'] = st.sidebar.number_input(f"Slow SMA #{i+1}", min_value=1,
                        value=cond.get('slow_sma',50), key=f"css_{i}")
                elif c['crossover_type'] in ["Price × EMA","Price × SMA"]:
                    c['ma_period'] = st.sidebar.number_input(f"MA Period #{i+1}", min_value=1,
                        value=cond.get('ma_period',50), key=f"cmap_{i}")
                elif c['crossover_type'] == "RSI Crossover (Overbought/Oversold)":
                    c['rsi_period'] = st.sidebar.number_input(f"RSI Period #{i+1}", min_value=1,
                        value=cond.get('rsi_period',14), key=f"crsi_{i}")
                    c['rsi_ob']     = st.sidebar.number_input(f"Overbought #{i+1}", min_value=50.0, max_value=100.0,
                        value=float(cond.get('rsi_ob',70.0)), key=f"crob_{i}")
                    c['rsi_os']     = st.sidebar.number_input(f"Oversold #{i+1}", min_value=0.0, max_value=50.0,
                        value=float(cond.get('rsi_os',30.0)), key=f"cros_{i}")
                c['cross_type']    = st.sidebar.selectbox(f"Direction #{i+1}",
                    ["Bullish Crossover","Bearish Crossover"],
                    index=0 if cond.get('cross_type','Bullish Crossover')=='Bullish Crossover' else 1,
                    key=f"cct_{i}")
                c['position_type'] = st.sidebar.selectbox(f"Position #{i+1}", ["LONG","SHORT"],
                    index=0 if cond.get('position_type','LONG')=='LONG' else 1,
                    key=f"cpt_{i}")

            rendered_conditions.append(c)

        # Persist updated conditions list
        st.session_state['custom_indicator_conditions'] = rendered_conditions
        config['custom_conditions'] = rendered_conditions
        # Legacy single-condition keys for backward compat
        if rendered_conditions:
            first = rendered_conditions[0]
            config['custom_strategy_type']  = first.get('strategy_type', 'Price Crosses Indicator')
            config['custom_position_type']  = first.get('position_type', 'LONG')
            config['custom_indicator']      = first.get('indicator', 'EMA')
            config['custom_cross_type']     = first.get('cross_type', 'Above Indicator')
            config['custom_indicator_period'] = first.get('period', 20)
    
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
    
    if config['target_type'] == 'Dynamic Trailing SL+Target (Lock Profits)':
        st.sidebar.markdown("**Dynamic Trailing Configuration**")
        st.sidebar.info("Both SL and Target trail together as price moves favorably")
        config['dynamic_trail_sl_points'] = st.sidebar.number_input(
            "SL Distance (Points)", 
            min_value=1, 
            value=10,
            help="Distance from current price to SL (trails with price)"
        )
        config['dynamic_trail_target_points'] = st.sidebar.number_input(
            "Target Distance (Points)", 
            min_value=1, 
            value=20,
            help="Distance from current price to Target (trails with price)"
        )
        st.sidebar.caption(
            f"Example: If price = 50\n"
            f"SL = {50 - config['dynamic_trail_sl_points']:.0f} | Target = {50 + config['dynamic_trail_target_points']:.0f}\n"
            f"Price → 51: SL = {51 - config['dynamic_trail_sl_points']:.0f} | Target = {51 + config['dynamic_trail_target_points']:.0f}"
        )
    
    # Dhan Broker Configuration
    st.sidebar.subheader("🏦 Dhan Broker (Optional)")
    config['dhan_enabled'] = st.sidebar.checkbox("Enable Dhan Broker", value=False)
    
    if config['dhan_enabled']:
        config['dhan_client_id'] = st.sidebar.text_input("Client ID", value="1104779876")
        config['dhan_access_token'] = st.sidebar.text_input("Access Token", type="password", value="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzcxMTQzMjM5LCJpYXQiOjE3NzEwNTY4MzksInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTA0Nzc5ODc2In0.qP8kVXDQt-sFa6LWJqd1MRTPESHCCPCqHzEnjsFI2WVbNdywKHXgAKHxVpuH6tP_AJTdqowv9nbqf-2NcGibbQ")
        
        config['dhan_is_options'] = st.sidebar.checkbox("Is Options", value=True)
        
        if config['dhan_is_options']:
            # ── Options trading ──
            config['dhan_ce_security_id'] = st.sidebar.text_input("CE Security ID", value="48228")
            config['dhan_pe_security_id'] = st.sidebar.text_input("PE Security ID", value="48229")
            config['dhan_strike_price']   = st.sidebar.number_input("Strike Price", min_value=0, value=25000)
            config['dhan_expiry_date']    = st.sidebar.date_input("Expiry Date", value=datetime.now().date())
            config['dhan_quantity']       = st.sidebar.number_input("Dhan Quantity", min_value=1, value=65)
            st.sidebar.info("Order Type: MARKET | Product: INTRA (options)")
        else:
            # ── Stock/Equity trading ──
            config['dhan_trading_type'] = st.sidebar.selectbox(
                "Trading Type",
                ["Intraday", "Delivery (CNC)"],
                help="Intraday = MIS/INTRA  |  Delivery = CNC positional"
            )
            config['dhan_security_id'] = st.sidebar.text_input("Security ID", value="1234")
            config['dhan_exchange']    = st.sidebar.selectbox("Exchange", ["NSE", "BSE"], index=0)
            config['dhan_quantity']    = st.sidebar.number_input("Quantity", min_value=1, value=10)
            if config['dhan_trading_type'] == 'Delivery (CNC)':
                st.sidebar.info("Order Type: MARKET | Product: CNC")
            else:
                st.sidebar.info("Order Type: MARKET | Product: INTRA")

        # ── Broker SL / Target (Bracket Order) — available for ALL order types ──
        st.sidebar.markdown("---")
        
        # Order Type Selection
        config['dhan_order_type'] = st.sidebar.selectbox(
            "Order Type",
            ["Market Order", "Limit Order"],
            index=0,
            help="Market: Executes immediately at best price | Limit: Executes at specified price or better"
        )
        
        config['broker_use_own_sl'] = st.sidebar.checkbox(
            "🎯 Use Broker SL/Target (Bracket Order)",
            value=False,
            help=(
                "Sends a Bracket Order (BO) to Dhan with embedded SL and Target.\n"
                "Works for Options, Intraday, and Delivery orders.\n"
                "Values are DISTANCES (points) from entry price, not absolute levels.\n"
                "When enabled the algo SL/Target still monitors locally; Dhan also manages exits."
            )
        )
        if config['broker_use_own_sl']:
            st.sidebar.markdown("**Bracket Order Parameters**")
            col_a, col_b = st.sidebar.columns(2)
            with col_a:
                config['broker_sl_points']     = st.sidebar.number_input(
                    "SL Points (boStopLossValue)", min_value=0.5, value=50.0, step=0.5,
                    help="Distance in points from entry to place stop-loss")
            with col_b:
                config['broker_target_points'] = st.sidebar.number_input(
                    "Target Points (boProfitValue)", min_value=0.5, value=100.0, step=0.5,
                    help="Distance in points from entry to place profit target")
            config['broker_trailing_jump'] = st.sidebar.number_input(
                "Trail SL Jump (0 = off)", min_value=0.0, value=0.0, step=0.5,
                help="SL trails by this many points for every 1-point favourable move (trailStopLoss)")
            st.sidebar.info(
                f"📌 BO: Entry ± {config['broker_sl_points']}pts SL | "
                f"+{config['broker_target_points']}pts Target"
                + (f" | Trail: {config['broker_trailing_jump']}pts" if config['broker_trailing_jump'] > 0 else "")
            )
        
        # ── Multi-Account Trading ────────────────────────────────────────────
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🔀 Multi-Account Trading**")
        
        # Initialize multi_accounts in session state if not exists
        if 'multi_accounts' not in st.session_state:
            st.session_state['multi_accounts'] = []
        
        # Display existing accounts
        if st.session_state['multi_accounts']:
            st.sidebar.write(f"**Configured Accounts:** {len(st.session_state['multi_accounts'])}")
            for i, acc in enumerate(st.session_state['multi_accounts']):
                col_acc1, col_acc2 = st.sidebar.columns([3, 1])
                with col_acc1:
                    st.sidebar.caption(f"{i+1}. Client: {acc['client_id'][:8]}...")
                with col_acc2:
                    if st.sidebar.button("❌", key=f"del_acc_{i}"):
                        st.session_state['multi_accounts'].pop(i)
                        st.rerun()
        
        # Add new account form
        with st.sidebar.expander("➕ Add Account"):
            new_client_id = st.text_input("Client ID", key="new_client_id")
            new_token = st.text_input("Access Token", type="password", key="new_token")
            if st.button("Add Account", key="add_account_btn"):
                if new_client_id and new_token:
                    st.session_state['multi_accounts'].append({
                        'client_id': new_client_id,
                        'access_token': new_token
                    })
                    st.success(f"Account added! Total: {len(st.session_state['multi_accounts'])}")
                    st.rerun()
                else:
                    st.error("Please provide both Client ID and Token")
        
        config['multi_accounts'] = st.session_state['multi_accounts']
        
        # ── Multi-Strike Options ─────────────────────────────────────────────
        if config['dhan_is_options']:
            st.sidebar.markdown("**📊 Multi-Strike Options**")
            config['multi_strike_enabled'] = st.sidebar.checkbox(
                "Enable Multi-Strike Orders",
                value=False,
                help="Place orders on multiple strike prices simultaneously"
            )
            
            if config['multi_strike_enabled']:
                # Initialize multi_strikes in session state
                if 'multi_strikes_ce' not in st.session_state:
                    st.session_state['multi_strikes_ce'] = []
                if 'multi_strikes_pe' not in st.session_state:
                    st.session_state['multi_strikes_pe'] = []
                
                strike_type = st.sidebar.radio(
                    "Strike Type",
                    ["CE (Call)", "PE (Put)"],
                    horizontal=True
                )
                
                if strike_type == "CE (Call)":
                    # Display existing CE strikes
                    if st.session_state['multi_strikes_ce']:
                        st.sidebar.write(f"**CE Strikes:** {len(st.session_state['multi_strikes_ce'])}")
                        for i, sec_id in enumerate(st.session_state['multi_strikes_ce']):
                            col_s1, col_s2 = st.sidebar.columns([3, 1])
                            with col_s1:
                                st.sidebar.caption(f"{i+1}. {sec_id}")
                            with col_s2:
                                if st.sidebar.button("❌", key=f"del_ce_{i}"):
                                    st.session_state['multi_strikes_ce'].pop(i)
                                    st.rerun()
                    
                    # Add CE strike
                    with st.sidebar.expander("➕ Add CE Strike"):
                        new_ce_id = st.text_input("CE Security ID", key="new_ce_id")
                        if st.button("Add CE", key="add_ce_btn"):
                            if new_ce_id:
                                st.session_state['multi_strikes_ce'].append(new_ce_id)
                                st.success(f"CE Strike added! Total: {len(st.session_state['multi_strikes_ce'])}")
                                st.rerun()
                
                else:  # PE
                    # Display existing PE strikes
                    if st.session_state['multi_strikes_pe']:
                        st.sidebar.write(f"**PE Strikes:** {len(st.session_state['multi_strikes_pe'])}")
                        for i, sec_id in enumerate(st.session_state['multi_strikes_pe']):
                            col_s1, col_s2 = st.sidebar.columns([3, 1])
                            with col_s1:
                                st.sidebar.caption(f"{i+1}. {sec_id}")
                            with col_s2:
                                if st.sidebar.button("❌", key=f"del_pe_{i}"):
                                    st.session_state['multi_strikes_pe'].pop(i)
                                    st.rerun()
                    
                    # Add PE strike
                    with st.sidebar.expander("➕ Add PE Strike"):
                        new_pe_id = st.text_input("PE Security ID", key="new_pe_id")
                        if st.button("Add PE", key="add_pe_btn"):
                            if new_pe_id:
                                st.session_state['multi_strikes_pe'].append(new_pe_id)
                                st.success(f"PE Strike added! Total: {len(st.session_state['multi_strikes_pe'])}")
                                st.rerun()
                
                config['multi_strikes_ce'] = st.session_state.get('multi_strikes_ce', [])
                config['multi_strikes_pe'] = st.session_state.get('multi_strikes_pe', [])
        
        # ── Clear/Close All Positions Before New Entry ───────────────────────
        st.sidebar.markdown("---")
        config['clear_positions_before_entry'] = st.sidebar.checkbox(
            "🧹 Clear All Positions Before New Entry",
            value=False,
            help=(
                "When enabled, cancels all pending orders and closes all open positions "
                "with market orders before placing a new entry order.\n"
                "Prevents messy orders when algo exits but broker order still pending."
            )
        )
    
    return config

def render_backtest_ui(config):
    """Render backtesting interface with EMA plot and IST time filter"""
    st.header("📈 Backtest Results")

    # ── Time-filter checkbox (outside Run button so it persists) ──────────────
    filter_market_hours = st.checkbox(
        "🕐 Filter Same-Day Trades Only (9:15 AM – 3:00 PM IST)",
        value=False,
        help="Shows only trades that were entered AND exited on the same day between 9:15 AM and 3:00 PM IST. Removes gap-up/gap-down and overnight trades."
    )
    
    # ── Backtesting Method 2 (Realistic Entry) ───────────────────────────────
    use_method2 = st.checkbox(
        "🔬 Use Backtesting Method 2 (Realistic Entry)",
        value=False,
        help=(
            "Method 2 fixes look-ahead bias and entry price issues:\n"
            "• Entry price = NEXT candle's OPEN (not current close)\n"
            "• Prevents using future data\n"
            "• More realistic simulation\n"
            "⚠️ Results will be more conservative but accurate"
        )
    )
    
    config['use_backtest_method2'] = use_method2

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            ticker       = config.get('asset', 'NIFTY 50')
            interval     = INTERVAL_MAPPING.get(config.get('interval', '1 day'), '1d')
            period       = PERIOD_MAPPING.get(config.get('period', '1 month'), '1mo')
            custom_ticker = config.get('custom_ticker', None)

            df = fetch_data(ticker, interval, period, custom_ticker=custom_ticker)

            if df is not None:
                df = calculate_all_indicators(df, config)
                trades, metrics, debug_info, skipped_trades = run_backtest(df, config)

                st.session_state['backtest_results'] = {
                    'trades':         trades,
                    'metrics':        metrics,
                    'debug_info':     debug_info,
                    'skipped_trades': skipped_trades,
                    'df':             df          # store for chart
                }

    # ── Display results ───────────────────────────────────────────────────────
    if 'backtest_results' in st.session_state:
        results    = st.session_state['backtest_results']
        all_trades = results['trades']
        debug_info = results['debug_info']
        df_chart   = results.get('df')

        # ── Apply IST time filter if checkbox is on ───────────────────────────
        IST = pytz.timezone('Asia/Kolkata')
        if filter_market_hours and all_trades:
            filtered_trades = []
            for t in all_trades:
                et = t.get('entry_time')
                xt = t.get('exit_time')
                try:
                    # Normalise to IST
                    if et is not None:
                        if hasattr(et, 'tzinfo') and et.tzinfo is not None:
                            et_ist = et.astimezone(IST)
                        else:
                            et_ist = IST.localize(et)
                        # Entry must be between 9:15 AM and 3:00 PM (not 3:15 PM)
                        entry_ok = (et_ist.hour > 9 or (et_ist.hour == 9 and et_ist.minute >= 15)) and \
                                   (et_ist.hour < 15)  # Strictly before 3:00 PM
                    else:
                        entry_ok = False  # Reject if no entry time
                    
                    if xt is not None:
                        if hasattr(xt, 'tzinfo') and xt.tzinfo is not None:
                            xt_ist = xt.astimezone(IST)
                        else:
                            xt_ist = IST.localize(xt)
                        # Exit must be between 9:15 AM and 3:00 PM
                        exit_ok = (xt_ist.hour > 9 or (xt_ist.hour == 9 and xt_ist.minute >= 15)) and \
                                  (xt_ist.hour < 15)  # Strictly before 3:00 PM
                    else:
                        exit_ok = False  # Reject if no exit time
                    
                    # SAME DAY CHECK: entry and exit must be on same date
                    same_day = False
                    if et is not None and xt is not None:
                        if hasattr(et, 'tzinfo') and et.tzinfo is not None:
                            et_ist = et.astimezone(IST)
                        else:
                            et_ist = IST.localize(et) if not hasattr(et, 'tzinfo') or et.tzinfo is None else et
                        if hasattr(xt, 'tzinfo') and xt.tzinfo is not None:
                            xt_ist = xt.astimezone(IST)
                        else:
                            xt_ist = IST.localize(xt) if not hasattr(xt, 'tzinfo') or xt.tzinfo is None else xt
                        
                        same_day = et_ist.date() == xt_ist.date()
                    
                    # All three conditions must be true
                    if entry_ok and exit_ok and same_day:
                        filtered_trades.append(t)
                except Exception:
                    # Skip trades with timestamp errors
                    pass

            trades = filtered_trades
            st.info(f"🕐 Same-day filter (9:15 AM–3:00 PM IST): {len(trades)} / {len(all_trades)} trades shown")
        else:
            trades = all_trades

        # ── Recompute metrics on filtered trades ──────────────────────────────
        if trades:
            df_t = pd.DataFrame(trades)
            total_trades   = len(df_t)
            winning_trades = int((df_t['pnl'] > 0).sum())
            losing_trades  = int((df_t['pnl'] < 0).sum())
            win_rate       = (winning_trades / total_trades * 100) if total_trades else 0
            total_pnl      = float(df_t['pnl'].sum())
            avg_pnl        = float(df_t['pnl'].mean())
            
            # Net P&L calculations
            total_brokerage = float(df_t['brokerage'].sum()) if 'brokerage' in df_t.columns else 0
            total_net_pnl = float(df_t['net_pnl'].sum()) if 'net_pnl' in df_t.columns else total_pnl
            avg_net_pnl = float(df_t['net_pnl'].mean()) if 'net_pnl' in df_t.columns else avg_pnl
            
            # Drawdown calculation
            pnl_column = 'net_pnl' if 'net_pnl' in df_t.columns else 'pnl'
            cum_pnl = df_t[pnl_column].cumsum()
            max_drawdown   = float((cum_pnl - cum_pnl.cummax()).min())
        else:
            total_trades = winning_trades = losing_trades = 0
            win_rate = total_pnl = avg_pnl = max_drawdown = 0.0
            total_brokerage = total_net_pnl = avg_net_pnl = 0.0

        metrics = dict(
            total_trades=total_trades, 
            winning_trades=winning_trades,
            losing_trades=losing_trades, 
            win_rate=win_rate,
            total_pnl=total_pnl, 
            avg_pnl=avg_pnl, 
            max_drawdown=max_drawdown,
            total_brokerage=total_brokerage,
            total_net_pnl=total_net_pnl,
            avg_net_pnl=avg_net_pnl
        )

        # ── Metrics display ───────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Trades",    metrics['total_trades'])
        with col2: st.metric("Win Rate",        f"{metrics['win_rate']:.2f}%")
        with col3: st.metric("Total P&L",       f"₹{metrics['total_pnl']:.2f}")
        with col4: st.metric("Avg Trade",       f"₹{metrics['avg_pnl']:.2f}")
        
        # Display brokerage and net P&L if brokerage is enabled
        if config.get('include_brokerage', False) and 'total_brokerage' in metrics:
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            with col_b1: st.metric("Total Brokerage", f"₹{metrics['total_brokerage']:.2f}")
            with col_b2: st.metric("**Net P&L**",      f"**₹{metrics['total_net_pnl']:.2f}**")
            with col_b3: st.metric("Avg Net P&L",      f"₹{metrics['avg_net_pnl']:.2f}")
            with col_b4: pass
        
        col5, col6, col7 = st.columns(3)
        with col5: st.metric("Winning Trades",  metrics['winning_trades'])
        with col6: st.metric("Losing Trades",   metrics['losing_trades'])
        with col7: st.metric("Max Drawdown",    f"₹{metrics['max_drawdown']:.2f}")

        # ── EMA Crossover chart ───────────────────────────────────────────────
        if df_chart is not None:
            st.subheader("📊 Price Chart with EMA Overlay & Signals")
            strategy = config.get('strategy', '')

            # Determine how many candles to show (cap at 300 for readability)
            plot_df = df_chart.tail(300).copy()

            fig = go.Figure()

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=plot_df['Datetime'],
                open=plot_df['Open'], high=plot_df['High'],
                low=plot_df['Low'],   close=plot_df['Close'],
                name='Price', increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ))

            # EMA overlays (always shown when columns exist)
            if 'EMA_Fast' in plot_df.columns:
                fast_p = config.get('ema_fast', 9)
                fig.add_trace(go.Scatter(
                    x=plot_df['Datetime'], y=plot_df['EMA_Fast'],
                    mode='lines', name=f'EMA {fast_p}',
                    line=dict(color='#FF9800', width=1.5)))

            if 'EMA_Slow' in plot_df.columns:
                slow_p = config.get('ema_slow', 21)
                fig.add_trace(go.Scatter(
                    x=plot_df['Datetime'], y=plot_df['EMA_Slow'],
                    mode='lines', name=f'EMA {slow_p}',
                    line=dict(color='#2196F3', width=1.5)))

            # BB overlay if custom strategy uses BB
            if 'BB_Upper' in plot_df.columns and strategy == 'Custom Strategy':
                fig.add_trace(go.Scatter(x=plot_df['Datetime'], y=plot_df['BB_Upper'],
                    mode='lines', name='BB Upper', line=dict(color='#9C27B0', width=1, dash='dot')))
                fig.add_trace(go.Scatter(x=plot_df['Datetime'], y=plot_df['BB_Lower'],
                    mode='lines', name='BB Lower', line=dict(color='#9C27B0', width=1, dash='dot'),
                    fill='tonexty', fillcolor='rgba(156,39,176,0.05)'))
                fig.add_trace(go.Scatter(x=plot_df['Datetime'], y=plot_df['BB_Middle'],
                    mode='lines', name='BB Mid', line=dict(color='#9C27B0', width=1)))

            # Plot trade entry/exit markers from *filtered* trades
            if trades:
                # Filter markers to the plotted window
                min_dt = plot_df['Datetime'].min()
                for tr in trades:
                    et = tr.get('entry_time')
                    xt = tr.get('exit_time')
                    ep = tr.get('entry_price')
                    xp = tr.get('exit_price')
                    pos = tr.get('type', 'LONG')

                    if et is not None and ep is not None:
                        try:
                            if hasattr(et, 'tzinfo') and et.tzinfo and hasattr(min_dt, 'tzinfo') and min_dt.tzinfo:
                                in_window = et >= min_dt
                            else:
                                in_window = True
                        except Exception:
                            in_window = True
                        if in_window:
                            fig.add_trace(go.Scatter(
                                x=[et], y=[ep],
                                mode='markers',
                                marker=dict(symbol='triangle-up' if pos == 'LONG' else 'triangle-down',
                                            size=12,
                                            color='#00E676' if pos == 'LONG' else '#FF1744',
                                            line=dict(width=1, color='black')),
                                name=f'Entry {pos}', showlegend=False,
                                hovertemplate=f"Entry {pos}<br>Price: {ep:.2f}<extra></extra>"))

                    if xt is not None and xp is not None:
                        try:
                            in_window = True
                        except Exception:
                            in_window = True
                        if in_window:
                            fig.add_trace(go.Scatter(
                                x=[xt], y=[xp],
                                mode='markers',
                                marker=dict(symbol='x', size=10,
                                            color='#FF6F00',
                                            line=dict(width=2, color='black')),
                                name='Exit', showlegend=False,
                                hovertemplate=f"Exit<br>Price: {xp:.2f}<extra></extra>"))

            # Market-hours shading for intraday intervals
            if filter_market_hours and '1d' not in INTERVAL_MAPPING.get(config.get('interval',''), '1d'):
                pass  # shading optional, skip for clarity

            fig.update_layout(
                title=f"{config.get('asset','Asset')} — {config.get('interval','')}"
                      + (" [Market Hours Filter ON]" if filter_market_hours else ""),
                xaxis_title='Time',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False,
                height=500,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                hovermode='x unified',
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Trade table ───────────────────────────────────────────────────────
        if trades:
            st.subheader("✅ Executed Trade History")
            df_trades = pd.DataFrame(trades)
            # Friendly column formatting
            for col in ['entry_price','exit_price','highest_price','lowest_price','sl_price','target_price','pnl','net_pnl','brokerage']:
                if col in df_trades.columns:
                    df_trades[col] = df_trades[col].apply(
                        lambda x: f"₹{x:.2f}" if pd.notna(x) and x == x else "—")  # x == x checks for NaN
            
            # Format duration
            if 'duration_minutes' in df_trades.columns:
                df_trades['duration'] = df_trades['duration_minutes'].apply(
                    lambda x: f"{int(x)} min" if pd.notna(x) else "—")
            
            # Format angle
            if 'ema_angle_entry' in df_trades.columns:
                df_trades['angle'] = df_trades['ema_angle_entry'].apply(
                    lambda x: f"{x:.2f}°" if pd.notna(x) else "—")
            
            # Select columns to display
            display_cols = ['entry_time', 'exit_time', 'type', 'entry_price', 'exit_price', 
                           'highest_price', 'lowest_price', 'pnl', 'net_pnl', 'exit_reason']
            if 'duration' in df_trades.columns:
                display_cols.insert(3, 'duration')
            
            # Show subset for display
            df_display = df_trades[[c for c in display_cols if c in df_trades.columns]]
            st.dataframe(df_display, use_container_width=True)
            
            # Detailed metrics expander
            with st.expander("📊 Detailed Trade Metrics (EMA, Angles, Differences)"):
                detailed_cols = [
                    'entry_time', 'exit_time', 'type', 'strategy', 
                    'pnl', 'net_pnl', 'brokerage',
                    'ema_fast_period', 'ema_slow_period', 'angle',
                    'ema_fast_entry', 'ema_slow_entry', 
                    'price_fast_ema_diff_entry', 'price_slow_ema_diff_entry', 'fast_slow_ema_diff_entry',
                    'ema_fast_exit', 'ema_slow_exit',
                    'price_fast_ema_diff_exit', 'price_slow_ema_diff_exit', 'fast_slow_ema_diff_exit'
                ]
                # Keep formatting for numeric columns
                df_detailed = df_trades.copy()
                for col in ['pnl', 'net_pnl', 'brokerage', 'ema_fast_entry', 'ema_slow_entry', 
                           'price_fast_ema_diff_entry', 'price_slow_ema_diff_entry', 'fast_slow_ema_diff_entry',
                           'ema_fast_exit', 'ema_slow_exit', 'price_fast_ema_diff_exit', 
                           'price_slow_ema_diff_exit', 'fast_slow_ema_diff_exit']:
                    if col in df_detailed.columns and col not in ['pnl', 'net_pnl', 'brokerage']:
                        # Format technical indicators
                        df_detailed[col] = df_detailed[col].apply(
                            lambda x: f"{x:.2f}" if pd.notna(x) and x == x else "—")
                
                df_detailed_display = df_detailed[[c for c in detailed_cols if c in df_detailed.columns]]
                st.dataframe(df_detailed_display, use_container_width=True)
        
        # ── Skipped/Overlapping Trades Table ─────────────────────────────────
        skipped_trades = results.get('skipped_trades', [])
        if skipped_trades and config.get('prevent_overlapping_trades', True):
            st.subheader("⚠️ Skipped/Overlapping Trades (Not Included in P&L)")
            st.info(
                f"**{len(skipped_trades)} signals were skipped** because they occurred while a position was active. "
                "These trades show what *would have happened* if overlapping trades were allowed. "
                "Their P&L is NOT included in total metrics above."
            )
            
            df_skipped = pd.DataFrame(skipped_trades)
            # Format columns
            for col in ['entry_price', 'exit_price', 'sl_price', 'target_price', 'pnl', 'net_pnl', 'brokerage']:
                if col in df_skipped.columns:
                    df_skipped[col] = df_skipped[col].apply(
                        lambda x: f"₹{x:.2f}" if pd.notna(x) else "—")
            
            st.dataframe(df_skipped, use_container_width=True)
            
            # Skipped trades summary
            if len(skipped_trades) > 0:
                df_skip_raw = pd.DataFrame(skipped_trades)
                skip_win = len(df_skip_raw[df_skip_raw['pnl'] > 0])
                skip_loss = len(df_skip_raw[df_skip_raw['pnl'] <= 0])
                skip_total_pnl = df_skip_raw['pnl'].sum()
                
                col_sk1, col_sk2, col_sk3 = st.columns(3)
                with col_sk1:
                    st.metric("Skipped Winning", skip_win)
                with col_sk2:
                    st.metric("Skipped Losing", skip_loss)
                with col_sk3:
                    st.metric("Skipped Total P&L", f"₹{skip_total_pnl:.2f}")

        # ── Debug expander ────────────────────────────────────────────────────
        if metrics['total_trades'] == 0:
            st.warning("⚠️ No trades generated")

        with st.expander("🔍 Debug Information"):
            st.write(f"- Total Candles: {debug_info['total_candles']}")
            st.write(f"- Candles Analyzed: {debug_info['candles_analyzed']}")
            st.write(f"- Signals Generated: {debug_info['signals_generated']}")
            st.write(f"- Trades Entered: {debug_info['trades_entered']}")
            st.write(f"- Trades Completed: {debug_info['trades_completed']}")
            if 'signals_skipped' in debug_info:
                st.write(f"- Signals Skipped (Overlapping): {debug_info['signals_skipped']}")
            if 'overlapping_trades' in debug_info:
                st.write(f"- Overlapping Trades Tracked: {debug_info['overlapping_trades']}")
            if filter_market_hours:
                st.write(f"- Trades after same-day filter (9:15 AM–3:00 PM IST): {len(trades)}")

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
            st.session_state['last_exit_time'] = None
            st.session_state['last_signal_type'] = None
            st.session_state['clearing_in_progress'] = False
            st.session_state['indicator_df'] = None
            st.session_state['last_candle_fetch_time'] = None
            
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
            add_log(f"⏱️ Entry Cooldown: {config.get('entry_cooldown_seconds', 60)}s")
            
            if config.get('enhanced_live_trading', False):
                add_log("🔄 Enhanced Live Trading Mode ENABLED - indicators will match TradingView")
            
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
                custom_ticker = config.get('custom_ticker', None)
                
                df = fetch_data(ticker, interval, period, is_live_trading=True, custom_ticker=custom_ticker)
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

    # ── Live Price Chart with EMA Overlay ────────────────────────────────────
    st.subheader("📊 Live Chart")
    try:
        ticker_sym    = config.get('asset', 'NIFTY 50')
        interval_code = INTERVAL_MAPPING.get(config.get('interval', '1 minute'), '1m')
        period_code   = PERIOD_MAPPING.get(config.get('period', '1 day'), '1d')
        custom_ticker = config.get('custom_ticker', None)

        df_live = fetch_data(ticker_sym, interval_code, period_code, custom_ticker=custom_ticker)
        if df_live is not None and not df_live.empty:
            df_live = calculate_all_indicators(df_live, config)
            plot_df = df_live.tail(150).copy()

            fig_live = go.Figure()

            # Candlestick
            fig_live.add_trace(go.Candlestick(
                x=plot_df['Datetime'],
                open=plot_df['Open'], high=plot_df['High'],
                low=plot_df['Low'],   close=plot_df['Close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ))

            # EMA Fast overlay
            if 'EMA_Fast' in plot_df.columns:
                fig_live.add_trace(go.Scatter(
                    x=plot_df['Datetime'], y=plot_df['EMA_Fast'],
                    mode='lines', name=f"EMA {config.get('ema_fast', 9)}",
                    line=dict(color='#FF9800', width=1.5)))

            # EMA Slow overlay
            if 'EMA_Slow' in plot_df.columns:
                fig_live.add_trace(go.Scatter(
                    x=plot_df['Datetime'], y=plot_df['EMA_Slow'],
                    mode='lines', name=f"EMA {config.get('ema_slow', 21)}",
                    line=dict(color='#2196F3', width=1.5)))

            # BB overlay if custom strategy
            if 'BB_Upper' in plot_df.columns and config.get('strategy') == 'Custom Strategy':
                fig_live.add_trace(go.Scatter(
                    x=plot_df['Datetime'], y=plot_df['BB_Upper'],
                    mode='lines', name='BB Upper',
                    line=dict(color='#9C27B0', width=1, dash='dot')))
                fig_live.add_trace(go.Scatter(
                    x=plot_df['Datetime'], y=plot_df['BB_Lower'],
                    mode='lines', name='BB Lower',
                    line=dict(color='#9C27B0', width=1, dash='dot'),
                    fill='tonexty', fillcolor='rgba(156,39,176,0.05)'))
                fig_live.add_trace(go.Scatter(
                    x=plot_df['Datetime'], y=plot_df['BB_Middle'],
                    mode='lines', name='BB Mid',
                    line=dict(color='#9C27B0', width=1)))

            # Current position marker
            position = st.session_state.get('position')
            if position:
                ep   = position['entry_price']
                etime = position['entry_time']
                ptype = position['type']
                fig_live.add_hline(
                    y=ep, line_dash='dash', line_color='#00E676',
                    annotation_text=f"Entry {ptype} @ {ep:.2f}",
                    annotation_position='right')
                if position.get('sl_price'):
                    fig_live.add_hline(
                        y=position['sl_price'], line_dash='dot', line_color='#FF1744',
                        annotation_text=f"SL @ {position['sl_price']:.2f}",
                        annotation_position='right')
                if position.get('target_price'):
                    fig_live.add_hline(
                        y=position['target_price'], line_dash='dot', line_color='#00BCD4',
                        annotation_text=f"Target @ {position['target_price']:.2f}",
                        annotation_position='right')

            fig_live.update_layout(
                title=f"Live: {config.get('asset', '')} | {config.get('interval', '')}",
                xaxis_rangeslider_visible=False,
                height=420,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                hovermode='x unified',
                template='plotly_dark'
            )
            st.plotly_chart(fig_live, use_container_width=True)
        else:
            st.warning("⚠️ Could not load chart data")
    except Exception as e:
        st.warning(f"⚠️ Chart error: {e}")
    
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
            # Display locked ticker
            locked_ticker = position.get('ticker', 'N/A')
            st.metric("Locked Ticker", locked_ticker)
        with col3:
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
        with col4:
            st.metric("Highest Price", f"₹{position.get('highest_price', 0):.2f}")
        with col5:
            st.metric("Lowest Price", f"₹{position.get('lowest_price', 0):.2f}")
        
        # Show warning about locked ticker
        if position.get('ticker') != config.get('asset'):
            st.warning(f"⚠️ Position locked to {position.get('ticker')}. Config changes won't affect active position.")
        
        # Entry time
        st.info(f"**Entry Time:** {position['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.info("No active position")
    
    # Display broker information if enabled
    if config.get('dhan_enabled', False) and st.session_state.get('broker_position'):
        st.subheader("🏦 Broker Position")
        
        broker_pos = st.session_state['broker_position']
        is_options = broker_pos.get('is_options', True)
        
        if is_options:
            # Options trading display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Order ID", broker_pos['order_id'])
            with col2:
                st.metric("Option Type", broker_pos['option_type'])
            with col3:
                st.metric("Security ID", broker_pos['security_id'])
            with col4:
                st.metric("Status", broker_pos['status'])
        else:
            # Stock trading display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Order ID", broker_pos['order_id'])
            with col2:
                st.metric("Transaction", broker_pos['transaction_type'])
            with col3:
                st.metric("Security ID", broker_pos['security_id'])
            with col4:
                st.metric("Status", broker_pos['status'])
        
        # Display raw API response
        with st.expander("📄 Raw API Response"):
            st.json(broker_pos['raw_response'])
    
    # Display completed trades (immediate update)
    st.subheader("✅ Completed Trades")
    
    trade_history = st.session_state.get('trade_history', [])
    if trade_history:
        df_history = pd.DataFrame(trade_history)
        
        # Format columns
        for col in ['entry_price', 'exit_price', 'pnl', 'net_pnl', 'brokerage', 'sl_price', 'target_price']:
            if col in df_history.columns:
                df_history[col] = df_history[col].apply(
                    lambda x: f"₹{x:.2f}" if pd.notna(x) else "—"
                )
        
        # Calculate duration if available
        if 'entry_time' in df_history.columns and 'exit_time' in df_history.columns:
            df_history['duration'] = df_history.apply(
                lambda row: f"{int((row['exit_time'] - row['entry_time']).total_seconds() / 60)} min" 
                if pd.notna(row['entry_time']) and pd.notna(row['exit_time']) else "—",
                axis=1
            )
        
        # Show summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", len(df_history))
        with col2:
            profitable = len(df_history[df_history['pnl'].str.replace('₹', '').str.replace(',', '').astype(float) > 0])
            st.metric("Winning", profitable)
        with col3:
            if 'pnl' in df_history.columns:
                total_pnl = df_history['pnl'].str.replace('₹', '').str.replace(',', '').astype(float).sum()
                st.metric("Total P&L", f"₹{total_pnl:.2f}")
        with col4:
            if 'net_pnl' in df_history.columns:
                total_net_pnl = df_history['net_pnl'].str.replace('₹', '').str.replace(',', '').astype(float).sum()
                st.metric("Net P&L", f"₹{total_net_pnl:.2f}")
        
        # Display trade table
        display_cols = ['entry_time', 'exit_time', 'type', 'entry_price', 'exit_price', 'pnl', 'net_pnl', 'exit_reason']
        if 'duration' in df_history.columns:
            display_cols.insert(3, 'duration')
        
        df_display = df_history[[c for c in display_cols if c in df_history.columns]]
        st.dataframe(df_display, use_container_width=True, height=200)
    else:
        st.info("No completed trades yet. Trades will appear here immediately after exit.")
    
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
        
        # Auto-refresh every 1.5 seconds
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
    
    # Render configuration first to get the selected ticker
    config = render_config_ui()
    
    # Display selected ticker prominently at the top
    ticker_display = config.get('asset', 'NIFTY 50')
    if ticker_display == 'Custom Ticker':
        custom_ticker = config.get('custom_ticker', 'N/A')
        st.info(f"🎯 **Selected Ticker:** {custom_ticker} (Custom)")
    else:
        ticker_symbol = ASSET_MAPPING.get(ticker_display, ticker_display)
        st.info(f"🎯 **Selected Ticker:** {ticker_display} ({ticker_symbol})")
    
    st.markdown("---")
    
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
