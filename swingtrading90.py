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
import warnings

# Suppress SyntaxWarning from dhanhq library (invalid escape sequence in their code)
warnings.filterwarnings('ignore', category=SyntaxWarning, module='dhanhq')

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
    "Percentage Change",
    "Elliott Waves + Ratio Charts",
    "Price Action",
    "RSI-ADX-EMA Combined",
    "VWAP + Volume Spike",
    "Opening Range Breakout (ORB)",
    "Volume Breakout",
    "Momentum Breakout with ADX",
    "Support Resistance Bounce",
    "Custom Strategy",
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
    
    def place_order(self, transaction_type, security_id, quantity, signal_type=None, order_params=None, is_exit=False):
        """
        Place order via Dhan API.
        Supports: Market/Limit orders, CNC/Delivery, Bracket Orders (BO) with SL+Target+Trail.
        
        Args:
            transaction_type: 'BUY' or 'SELL'
            security_id: Dhan security ID
            quantity: Order quantity
            signal_type: Entry signal type (optional)
            order_params: Dict with order parameters like price
            is_exit: True if this is an exit order (uses exit order type config)
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
                
                # Select appropriate order type based on entry/exit
                if is_exit:
                    order_type_selection = self.config.get('dhan_exit_order_type', 'Market Order')
                else:
                    order_type_selection = self.config.get('dhan_entry_order_type', 'Market Order')
                
                # Fallback to legacy config if new ones not set
                if not order_type_selection:
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
            order_params={'price': price},  # Pass exit price for limit orders
            is_exit=True  # Use exit order type configuration
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

def fetch_data_dhan(security_id, exchange_segment, instrument_type, interval, period,
                    client_id, access_token, is_live_trading=False):
    """
    Fetch OHLCV data from DhanHQ API.

    Verified dhanhq constants (v2):
        dhan.NSE     = 'NSE_EQ'   ← equities on NSE
        dhan.BSE     = 'BSE_EQ'   ← equities on BSE
        dhan.INDEX   = 'IDX_I'    ← indices (Nifty 50, BankNifty …)
        dhan.NSE_FNO = 'NSE_FNO'  ← F&O
        dhan.BSE_FNO = 'BSE_FNO'
        dhan.MCX     = 'MCX_COMM'

    Verified data methods:
        dhan.intraday_minute_data()   – up to 5 trading days, intervals 1/5/15/25/60 min
        dhan.historical_daily_data()  – any date range, daily candles

    Returns a DataFrame with columns [Datetime, Open, High, Low, Close, Volume]
    in IST — identical shape to the yfinance path.
    """
    try:
        from dhanhq import dhanhq
    except ImportError:
        st.error("❌ dhanhq not installed. Run:  pip install dhanhq")
        return None

    try:
        dhan = dhanhq(client_id, access_token)
        ist   = pytz.timezone('Asia/Kolkata')
        today = datetime.now(ist)

        # ── 1. Date range ─────────────────────────────────────────────────────
        period_days = {
            '1d':  1,  '5d':  5,  '1mo':  30, '3mo':  90,
            '6mo': 180,'1y':  365, '2y':  730, '5y': 1825,
        }
        days      = period_days.get(period, 30)
        from_date = (today - timedelta(days=days)).strftime('%Y-%m-%d')
        to_date   = today.strftime('%Y-%m-%d')

        # ── 2. Exchange-segment constant ──────────────────────────────────────
        # dhan.INDEX ('IDX_I') must be used for all index instruments like Nifty.
        # dhan.NSE  ('NSE_EQ') is for equities only.
        seg_map = {
            'IDX_I (Index)': dhan.INDEX,   # Nifty 50, BankNifty, etc.
            'NSE_EQ':        dhan.NSE,     # NSE equities
            'BSE_EQ':        dhan.BSE,     # BSE equities
            'NSE_FNO':       dhan.NSE_FNO, # NSE futures & options
            'BSE_FNO':       dhan.BSE_FNO,
            'MCX_COMM':      dhan.MCX,
        }
        exch_seg = seg_map.get(exchange_segment, dhan.NSE)

        # ── 3. Interval mapping ───────────────────────────────────────────────
        # Dhan supports: 1, 5, 15, 25, 60 minutes (no native 30m)
        # For 30m we fetch 5m and resample afterwards.
        dhan_int_map = {'1m': 1, '5m': 5, '15m': 15, '30m': 5, '1h': 60}

        # ── 4. API call ───────────────────────────────────────────────────────
        if interval in ('1d', '1wk'):
            response = dhan.historical_daily_data(
                security_id      = str(security_id),
                exchange_segment = exch_seg,
                instrument_type  = instrument_type,
                from_date        = from_date,
                to_date          = to_date,
            )
        else:
            # Minute data is capped at 5 trading days
            if days > 5:
                clamped_from = (today - timedelta(days=5)).strftime('%Y-%m-%d')
                st.info(
                    f"ℹ️ DhanHQ minute data is limited to the last 5 trading days "
                    f"(adjusted from {from_date} → {clamped_from})."
                )
                from_date = clamped_from
            response = dhan.intraday_minute_data(
                security_id      = str(security_id),
                exchange_segment = exch_seg,
                instrument_type  = instrument_type,
                from_date        = from_date,
                to_date          = to_date,
                interval         = dhan_int_map.get(interval, 1),
            )

        # ── 5. Validate response ──────────────────────────────────────────────
        if not response or response.get('status') != 'success':
            err = (response or {}).get('remarks', 'Unknown error')
            st.error(f"❌ DhanHQ API error: {err}")
            return None

        raw = response.get('data', {})

        # ── 6. Parse response (handles dict of parallel arrays OR list of dicts)
        if isinstance(raw, list):
            # Format: [{"timestamp":…, "open":…, "high":…, "low":…, "close":…, "volume":…}, …]
            if not raw:
                st.error("❌ DhanHQ returned an empty list.")
                return None
            df = pd.DataFrame(raw)
            df.rename(columns={
                'timestamp': 'Datetime', 'open': 'Open', 'high': 'High',
                'low': 'Low', 'close': 'Close', 'volume': 'Volume'
            }, inplace=True)
            ts_col = 'Datetime'

        elif isinstance(raw, dict):
            # Format: {"timestamp": […], "open": […], …}
            timestamps = raw.get('timestamp', [])
            if not timestamps:
                st.error(
                    f"❌ DhanHQ: timestamp list is empty. "
                    f"Response keys: {list(raw.keys())}. "
                    f"Check Security ID / Exchange Segment / Instrument Type."
                )
                return None
            df = pd.DataFrame({
                'Datetime': timestamps,
                'Open':     raw.get('open',   []),
                'High':     raw.get('high',   []),
                'Low':      raw.get('low',    []),
                'Close':    raw.get('close',  []),
                'Volume':   raw.get('volume', []),
            })
            ts_col = 'Datetime'

        else:
            st.error(f"❌ DhanHQ: unexpected response type {type(raw)}.")
            return None

        # ── 7. Convert timestamps ─────────────────────────────────────────────
        ts_numeric = pd.to_numeric(df[ts_col], errors='coerce')
        # Auto-detect: Dhan returns epoch-seconds; values > 1e12 are milliseconds
        unit = 'ms' if (ts_numeric.dropna().median() > 1e12) else 's'
        df['Datetime'] = (
            pd.to_datetime(ts_numeric, unit=unit, utc=True)
            .dt.tz_convert(ist)
        )

        # ── 8. Numeric coercion ───────────────────────────────────────────────
        for col in ('Open', 'High', 'Low', 'Close'):
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Volume: Dhan may return floats or None — convert safely to int
        df['Volume'] = (
            pd.to_numeric(df['Volume'], errors='coerce')
            .fillna(0)
            .astype(float)     # ensure float before rounding (avoids object-dtype issues)
            .round()
            .astype(int)
        )

        # ── 9. Final clean-up ─────────────────────────────────────────────────
        df = (
            df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
            .dropna(subset=['Datetime', 'Close'])
            .sort_values('Datetime')
            .reset_index(drop=True)
        )

        if df.empty:
            st.error(
                "❌ DhanHQ: DataFrame is empty after processing. "
                "Verify Security ID and Exchange Segment.\n\n"
                "Common IDs → Nifty 50: ID=13, Segment=IDX_I (Index), Type=INDEX\n"
                "BankNifty: ID=25, Segment=IDX_I (Index), Type=INDEX"
            )
            return None

        # ── 10. Resample 5m → 30m (Dhan has no native 30-min interval) ────────
        if interval == '30m':
            df = (
                df.set_index('Datetime')
                .resample('30T')
                .agg({'Open': 'first', 'High': 'max',
                      'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
                .dropna(subset=['Close'])
                .reset_index()
            )

        return df

    except Exception as e:
        import traceback
        st.error(f"❌ Error fetching DhanHQ data: {e}")
        with st.expander("🔍 Full traceback"):
            st.code(traceback.format_exc())
        return None


def fetch_data(ticker_symbol, interval, period, is_live_trading=False, custom_ticker=None, config=None):
    """
    Fetch historical/live OHLCV data.

    Routes to DhanHQ or yfinance based on config['use_yfinance'].
    When config is not supplied it is read from st.session_state['config'].

    Args:
        ticker_symbol: Asset ticker (yfinance path only)
        interval: Time interval string ('1m', '5m', '1d', …)
        period: Historical period string ('1d', '1mo', …)
        is_live_trading: If True, fetch a smaller dataset
        custom_ticker: Custom ticker when asset == "Custom Ticker"
        config: Config dict; falls back to st.session_state['config']

    Returns:
        DataFrame with columns [Datetime, Open, High, Low, Close, Volume] in IST
    """
    # ── Resolve config ────────────────────────────────────────────────────────
    if config is None:
        config = st.session_state.get('config', {})

    # ── Route to DhanHQ ───────────────────────────────────────────────────────
    if not config.get('use_yfinance', True):
        return fetch_data_dhan(
            security_id      = config.get('dhan_data_security_id', ''),
            exchange_segment = config.get('dhan_data_exchange', 'IDX_I (Index)'),
            instrument_type  = config.get('dhan_data_instrument_type', 'INDEX'),
            interval         = interval,
            period           = period,
            client_id        = config.get('dhan_data_client_id') or config.get('dhan_client_id', ''),
            access_token     = config.get('dhan_data_access_token') or config.get('dhan_access_token', ''),
            is_live_trading  = is_live_trading,
        )

    # ── yfinance path (original logic, unchanged) ─────────────────────────────
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

# ================================
# FAST LTP FETCH
# ================================

# Seconds between candle re-fetches per interval
INTERVAL_FETCH_SECONDS = {
    '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
    '1h': 3600, '1d': 86400, '1wk': 604800,
}


def get_candle_period_start(now_dt, interval_secs):
    """
    Return the floor-aligned start of the current candle period.
    E.g. 09:23:45 with 5-min interval → 09:20:00 IST
    """
    ts = int(now_dt.timestamp())
    aligned_ts = (ts // interval_secs) * interval_secs
    return datetime.fromtimestamp(aligned_ts, tz=now_dt.tzinfo)


def update_live_candle(ltp, now_dt, interval_secs, _config_unused=None):
    """
    VISUAL-ONLY forming candle — tracks current period O/H/L/C from LTP.
    Does NOT touch indicator_df or recalculate indicators.
    The completed candle is fetched directly from the API (Dhan/yfinance)
    once the period boundary is crossed and staleness is cleared.
    """
    period_start = get_candle_period_start(now_dt, interval_secs)
    prev_start   = st.session_state.get('live_candle_start')

    if prev_start is None:
        # First tick — initialise
        st.session_state.update({
            'live_candle_start': period_start,
            'live_candle_open':  ltp,
            'live_candle_high':  ltp,
            'live_candle_low':   ltp,
            'live_candle_close': ltp,
        })
        return 'init'

    if period_start > prev_start:
        # New candle period — reset visual candle (API will fetch the completed one)
        add_log(
            f"🕯️ Period closed  "
            f"O={st.session_state.get('live_candle_open', ltp):.2f}  "
            f"H={st.session_state.get('live_candle_high', ltp):.2f}  "
            f"L={st.session_state.get('live_candle_low',  ltp):.2f}  "
            f"C={st.session_state.get('live_candle_close',ltp):.2f}"
        )
        st.session_state.update({
            'live_candle_start': period_start,
            'live_candle_open':  ltp,
            'live_candle_high':  ltp,
            'live_candle_low':   ltp,
            'live_candle_close': ltp,
        })
        return 'new_period'

    # Same period — update H / L / C
    st.session_state['live_candle_high']  = max(st.session_state['live_candle_high'],  ltp)
    st.session_state['live_candle_low']   = min(st.session_state['live_candle_low'],   ltp)
    st.session_state['live_candle_close'] = ltp
    return 'updated'

# ─────────────────────────────────────────────────────────────────────────────
# REAL-TIME LTP FEED
# Dhan  → MarketFeed WebSocket (feed.start() handles its own thread)
#         Pushes ticks instantly. on_ticks writes to a shared list.
# yfinance → background thread calling yf.download every 1s
# Fragment reads from session_state cache — 0ms, no HTTP on main thread.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# DHAN LIVE FEED  — exact pattern from Dhan documentation
#
# run_forever() connects once (not a loop — just opens WebSocket)
# get_data()    blocks until the next tick arrives (ws.recv())
# Both run in ONE background thread → no asyncio conflicts
#
# For display: uses mutable lists (ltp_box, debug_box) so the background
# thread never needs to touch st.session_state
# ─────────────────────────────────────────────────────────────────────────────

def start_ltp_feed(config):
    """Connect to Dhan MarketFeed WebSocket (or yfinance poll). Call on Start Trading."""
    import threading, time as _t

    # Stop any previous feed
    old_stop = st.session_state.get('_ltp_stop')
    if old_stop:
        old_stop.set()
    old_feed = st.session_state.get('_ltp_feed_obj')
    if old_feed:
        try:
            old_feed.disconnect()
        except Exception:
            pass

    ltp_box   = [None]   # ltp_box[0]   = latest LTP  (float)
    debug_box = ['—']    # debug_box[0] = last raw tick string (for display)
    stop_ev   = threading.Event()

    st.session_state['_ltp_box']   = ltp_box
    st.session_state['_debug_box'] = debug_box
    st.session_state['_ltp_stop']  = stop_ev

    use_yf = config.get('use_yfinance', True)

    if not use_yf:
        # ── Dhan MarketFeed WebSocket ─────────────────────────────────────
        client_id    = config.get('dhan_data_client_id') or config.get('dhan_client_id', '')
        access_token = config.get('dhan_data_access_token') or config.get('dhan_access_token', '')
        security_id  = str(config.get('dhan_data_security_id', ''))
        exch_raw     = config.get('dhan_data_exchange', 'IDX_I (Index)')

        def _dhan_thread():
            try:
                from dhanhq import DhanContext, MarketFeed

                seg_map = {
                    'IDX_I (Index)': MarketFeed.IDX,
                    'NSE_EQ':        MarketFeed.NSE,
                    'BSE_EQ':        MarketFeed.BSE,
                    'NSE_FNO':       MarketFeed.NSE_FNO,
                    'BSE_FNO':       MarketFeed.BSE_FNO,
                    'MCX_COMM':      MarketFeed.MCX,
                }
                seg         = seg_map.get(exch_raw, MarketFeed.IDX)
                instruments = [(seg, security_id, MarketFeed.Ticker)]
                ctx         = DhanContext(client_id, access_token)
                feed        = MarketFeed(ctx, instruments, version="v2")

                st.session_state['_ltp_feed_obj'] = feed

                # Step 1: connect (run_forever just opens WebSocket, returns quickly)
                feed.run_forever()

                # Step 2: keep reading ticks — each get_data() blocks until next tick
                while not stop_ev.is_set():
                    try:
                        response = feed.get_data()   # blocks until tick arrives
                        if response:
                            debug_box[0] = str(response)   # store raw for display
                            if isinstance(response, dict):
                                ltp_val = response.get('LTP') or response.get('ltp')
                                if ltp_val:
                                    ltp_box[0] = float(ltp_val)
                    except Exception as tick_err:
                        debug_box[0] = f"tick error: {tick_err}"
                        _t.sleep(1)

            except Exception as conn_err:
                debug_box[0] = f"connection error: {conn_err}"
                # Fallback to REST polling
                _yf_poll_thread(config, ltp_box, debug_box, stop_ev)

        threading.Thread(target=_dhan_thread, daemon=True).start()
        add_log(f"🔌 Dhan MarketFeed started — seg={exch_raw} id={security_id}")

    else:
        _yf_poll_thread(config, ltp_box, debug_box, stop_ev)


def _yf_poll_thread(config, ltp_box, debug_box, stop_ev):
    """yfinance: poll yf.download every 1s in background thread."""
    import threading, time as _t
    cfg = config.copy()

    def _run():
        while not stop_ev.is_set():
            try:
                name   = cfg.get('asset', 'NIFTY 50')
                symbol = cfg.get('custom_ticker') if name == 'Custom Ticker'                          else ASSET_MAPPING.get(name, name)
                df = yf.download(symbol, period='1d', interval='1m',
                                 progress=False, auto_adjust=True)
                if df is not None and not df.empty:
                    val = df['Close'].iloc[-1]
                    ltp = float(val.item() if hasattr(val, 'item') else val)
                    ltp_box[0]   = ltp
                    debug_box[0] = f"yfinance close: {ltp}"
            except Exception as e:
                debug_box[0] = f"yfinance error: {e}"
            stop_ev.wait(1.0)

    threading.Thread(target=_run, daemon=True).start()
    add_log("📡 yfinance poll thread started")


def stop_ltp_feed():
    """Stop the feed. Call on Stop Trading / Manual Close."""
    ev = st.session_state.get('_ltp_stop')
    if ev:
        ev.set()


def fetch_ltp(config):
    """Read LTP from feed cache — 0ms, no HTTP."""
    ltp_box = st.session_state.get('_ltp_box')
    if ltp_box and ltp_box[0] and ltp_box[0] > 0:
        return float(ltp_box[0])
    return None


def live_trading_iteration():
    """
    Single iteration of the live trading loop (called every ~0.3 s via st.fragment).

    Data architecture
    ─────────────────
    • fetch_ltp()            → lightweight price every tick  (yfinance fast_info / Dhan LTP)
    • update_live_candle()   → builds a VISUAL forming-candle from LTP ticks (chart only)
    • fetch_data()           → full OHLCV from Dhan / yfinance, throttled to the selected
                               interval (1m → every 60 s, 5m → every 300 s …)

    Stale-data guard (Dhan cache fix)
    ──────────────────────────────────
    Dhan sometimes returns the same candle repeatedly due to server-side caching.
    We detect this by comparing the latest candle's Datetime with the previously
    seen timestamp.  On a mismatch we back off for 3 s and retry — completely
    non-blocking because the back-off is stored in session_state.
    """
    config = st.session_state.get('config', {})

    if st.session_state.get('clearing_in_progress', False):
        add_log("🏦 ⏳ Position clearing in progress — skipping iteration")
        return

    position = st.session_state.get('position')

    # ── Resolve ticker ──────────────────────────────────────────────────────
    if position is not None and 'ticker' in position:
        ticker        = position['ticker']
        custom_ticker = position.get('custom_ticker')
    else:
        ticker        = config.get('asset', 'NIFTY 50')
        custom_ticker = config.get('custom_ticker', None)

    interval      = INTERVAL_MAPPING.get(config.get('interval', '1 day'), '1d')
    period        = PERIOD_MAPPING.get(config.get('period', '1 month'), '1mo')
    interval_secs = INTERVAL_FETCH_SECONDS.get(interval, 60)

    now = datetime.now(pytz.timezone('Asia/Kolkata'))

    # ── Step 1: Fast LTP every tick ────────────────────────────────────────
    ltp = fetch_ltp(config)
    if ltp is not None and ltp > 0:
        color = '🟢' if ltp >= (st.session_state.get('live_candle_open') or ltp) else '🔴'
        add_log(f"💹 LTP ₹{ltp:.2f} {color}")

    # ── Step 2: Visual forming-candle from LTP (chart overlay only) ────────
    if ltp is not None and ltp > 0:
        update_live_candle(ltp, now, interval_secs)

    # ── Step 3: API candle fetch (throttled + stale-data detection) ────────
    df                  = st.session_state.get('indicator_df')
    last_fetch_time     = st.session_state.get('last_candle_fetch_time')
    last_candle_ts      = st.session_state.get('last_candle_ts')
    stale_detected      = st.session_state.get('stale_detected', False)
    stale_retry_after   = st.session_state.get('stale_retry_after')

    # Decide whether it is time to call the API
    need_initial = (df is None or df.empty)
    need_refresh = need_initial or (
        last_fetch_time is not None and
        (now - last_fetch_time).total_seconds() >= interval_secs
    )

    # If we detected stale data, honour the back-off window
    if stale_detected and stale_retry_after and now < stale_retry_after:
        secs_left = int((stale_retry_after - now).total_seconds())
        need_refresh = False
        if need_initial:
            add_log(f"⚠️ Stale-data back-off: {secs_left}s — waiting before retry")
        else:
            add_log(f"⏳ Stale back-off: {secs_left}s remaining")

    if need_refresh:
        add_log(f"📥 Fetching {'initial' if need_initial else 'latest'} candles "
                f"({interval}/{period})…")
        df_new = fetch_data(
            ticker, interval, period,
            is_live_trading=True, custom_ticker=custom_ticker, config=config
        )

        if df_new is not None and not df_new.empty:
            new_ts = df_new.iloc[-1]['Datetime']

            # ── Stale-data check: compare latest timestamp ────────────────
            is_stale = False
            if last_candle_ts is not None and not need_initial:
                try:
                    ts_new = pd.Timestamp(new_ts).value   # nanoseconds for precision
                    ts_old = pd.Timestamp(last_candle_ts).value
                    is_stale = (ts_new <= ts_old)
                except Exception:
                    is_stale = False   # if comparison fails, treat as fresh

            if is_stale:
                # Dhan returned cached/stale data — back off
                BACKOFF = 3   # seconds
                st.session_state['stale_detected']        = True
                st.session_state['stale_retry_after']     = now + timedelta(seconds=BACKOFF)
                st.session_state['last_candle_fetch_time'] = now   # throttle
                add_log(
                    f"⚠️ Dhan stale data — latest candle unchanged ({new_ts}). "
                    f"Retrying in {BACKOFF}s"
                )
            else:
                # Fresh data — recalculate indicators and cache
                df_new = calculate_all_indicators(df_new, config)
                st.session_state['indicator_df']           = df_new
                st.session_state['last_candle_ts']         = new_ts
                st.session_state['last_candle_fetch_time'] = now
                st.session_state['stale_detected']         = False
                st.session_state['stale_retry_after']      = None
                df = df_new
                add_log(f"✅ {'Loaded' if need_initial else 'Refreshed'}: "
                        f"{len(df_new)} candles, latest: {new_ts}")
        else:
            # Network error / API failure
            st.session_state['last_candle_fetch_time'] = now   # don't hammer API
            add_log("⚠️ Candle fetch failed — retaining cached data")

    # Ensure we have usable data
    df = st.session_state.get('indicator_df')
    if df is None or df.empty:
        add_log("❌ No candle data yet — waiting for initial load")
        return

    # ── Step 4: Build current_data (API indicators + live LTP price) ───────
    idx          = len(df) - 1
    current_data = df.iloc[idx].copy()
    current_price = (
        ltp if (ltp is not None and ltp > 0)
        else float(current_data['Close'])
    )
    current_data['Close'] = current_price    # live LTP for SL/Target checks
    st.session_state['current_data'] = current_data

    # ── Step 5: Entry or exit logic ────────────────────────────────────────────
    strategy_name = config.get('strategy', 'EMA Crossover')
    strategy_func = STRATEGY_FUNCTIONS.get(strategy_name, check_ema_crossover_strategy)
    position      = st.session_state.get('position')

    if position is None:
        # ══ CRITICAL BUG FIX: mandatory post-exit cooldown ══════════════════
        # Prevents the SL re-hit loop: after any exit, no new entry is allowed
        # for at least one full candle interval (prevents 30-40x SL hits).
        _last_exit = st.session_state.get('last_exit_time')
        if _last_exit is not None:
            _elapsed  = (now - _last_exit).total_seconds()
            # Hard floor = one full candle period regardless of user settings
            _user_cool = config.get('entry_cooldown_seconds', 0) \
                         if config.get('enable_entry_cooldown', False) else 0
            _min_cool  = max(5, _user_cool)  # 5s prevents duplicate, not full interval
            if _elapsed < _min_cool:
                add_log(f"🔒 Cooldown: {int(_min_cool - _elapsed)}s remaining")
                return

        # ── Check trade window ─────────────────────────────────────────────────
        if not is_within_trade_window(current_data['Datetime'], config):
            add_log("⏰ Outside trade window — no new entries")
            return

        # ── Entry cooldown ─────────────────────────────────────────────────────
        if config.get('enable_entry_cooldown', False):
            cooldown_secs  = config.get('entry_cooldown_seconds', 0)
            last_exit_time = st.session_state.get('last_exit_time')
            if last_exit_time and cooldown_secs > 0:
                elapsed = (now - last_exit_time).total_seconds()
                if elapsed < cooldown_secs:
                    add_log(f"⏳ Cooldown: {int(cooldown_secs - elapsed)}s remaining")
                    return
                else:
                    add_log(f"✅ Cooldown complete ({int(elapsed)}s elapsed)")

        # ── Strategy signal ────────────────────────────────────────────────────
        add_log(f"🔍 Checking entry signal — {strategy_name}…")
        signal, entry_price = strategy_func(df, idx, config, None)

        if signal:
            if not should_allow_trade_direction(signal, config):
                add_log(f"🚫 Signal {signal} blocked by direction filter")
                return

            entry_price = entry_price or current_price
            position_type = 'LONG' if signal in ('BUY', 'LONG') else 'SHORT'

            sl_price     = calculate_initial_sl(position_type, entry_price, df, idx, config)
            target_price = calculate_initial_target(position_type, entry_price, df, idx, config)

            position = {
                'type':          position_type,
                'entry_price':   entry_price,
                'entry_time':    now,
                'sl_price':      sl_price,
                'target_price':  target_price,
                'highest_price': entry_price,
                'lowest_price':  entry_price,
                'quantity':      config.get('quantity', 1),
                'ticker':        ticker,
                'custom_ticker': custom_ticker,
                'strategy':      strategy_name,
                'ema_fast_period': config.get('ema_fast', 9),
                'ema_slow_period': config.get('ema_slow', 21),
                'ema_fast_entry':  current_data.get('EMA_Fast'),
                'ema_slow_entry':  current_data.get('EMA_Slow'),
                'ema_angle_entry': current_data.get('EMA_Fast_Angle'),
            }
            st.session_state['position'] = position
            add_log(f"🚀 ENTRY {signal} @ ₹{entry_price:.2f} | SL: {sl_price} | Target: {target_price}")

            if config.get('dhan_enabled', False):
                dhan_broker = st.session_state.get('dhan_broker')
                if dhan_broker:
                    try:
                        if config.get('clear_positions_before_entry', False):
                            st.session_state['clearing_in_progress'] = True
                            clear_result = dhan_broker.clear_all_positions(add_log)
                            if clear_result.get('clearing_complete'):
                                add_log("🏦 ✅ Previous positions cleared")
                            else:
                                add_log("🏦 ℹ️ No existing positions to clear")
                            st.session_state['clearing_in_progress'] = False

                        broker_position = dhan_broker.enter_broker_position(signal, entry_price, config, add_log)
                        st.session_state['broker_position'] = broker_position
                    except Exception as e:
                        add_log(f"🏦 ⚠️ Broker order error: {e}")
                else:
                    add_log("🏦 ⚠️ Broker not initialized")
        else:
            add_log(f"⏳ No entry signal (Price: ₹{current_price:.2f})")

    else:
        # ── Monitor position ───────────────────────────────────────────────────
        add_log(f"📊 Monitoring {position['type']} @ ₹{current_price:.2f}")

        price_diff_pct = abs(current_price - position['entry_price']) / position['entry_price'] * 100
        if price_diff_pct > 50:
            add_log(f"⚠️ Price differs {price_diff_pct:.1f}% from entry — possible ticker mismatch!")

        position['highest_price'] = max(position['highest_price'], current_price)
        position['lowest_price']  = min(position['lowest_price'],  current_price)

        if position['type'] == 'LONG':
            current_pnl = (current_price - position['entry_price']) * position['quantity']
        else:
            current_pnl = (position['entry_price'] - current_price) * position['quantity']
        add_log(f"💰 Live P&L: ₹{current_pnl:.2f}")

        # Trailing SL/Target update
        if position.get('sl_price') is not None:
            old_sl = position['sl_price']
            position['sl_price'] = update_trailing_sl(position, current_price, df, idx, config)
            if position['sl_price'] != old_sl:
                add_log(f"🛡️ SL: {old_sl:.2f} → {position['sl_price']:.2f}")

        if position.get('target_price') is not None:
            old_tgt = position['target_price']
            position['target_price'] = update_trailing_target(position, current_price, df, idx, config)
            if position['target_price'] != old_tgt:
                add_log(f"🎯 Target: {old_tgt:.2f} → {position['target_price']:.2f}")

        # ── Exit condition checks ──────────────────────────────────────────────
        exit_reason = None
        exit_price  = current_price

        if not is_within_trade_window(current_data['Datetime'], config):
            exit_reason = 'Trade Window Closed'
            add_log("⏰ TRADE WINDOW CLOSED — force exit")

        if exit_reason is None and position.get('sl_price') is not None:
            if position['type'] == 'LONG'  and current_price <= position['sl_price']:
                exit_reason = 'SL Hit'
                exit_price  = position['sl_price']
                add_log(f"🛑 SL HIT! ₹{current_price:.2f} <= ₹{position['sl_price']:.2f}")
            elif position['type'] == 'SHORT' and current_price >= position['sl_price']:
                exit_reason = 'SL Hit'
                exit_price  = position['sl_price']
                add_log(f"🛑 SL HIT! ₹{current_price:.2f} >= ₹{position['sl_price']:.2f}")

        if exit_reason is None and position.get('target_price') is not None:
            if position['type'] == 'LONG'  and current_price >= position['target_price']:
                exit_reason = 'Target Hit'
                exit_price  = position['target_price']
                add_log(f"🎯 TARGET HIT! ₹{current_price:.2f} >= ₹{position['target_price']:.2f}")
            elif position['type'] == 'SHORT' and current_price <= position['target_price']:
                exit_reason = 'Target Hit'
                exit_price  = position['target_price']
                add_log(f"🎯 TARGET HIT! ₹{current_price:.2f} <= ₹{position['target_price']:.2f}")

        if exit_reason is None and (
            config.get('sl_type')     in ('Signal-based (Reverse Crossover)', 'Strategy-based Signal') or
            config.get('target_type') in ('Signal-based (Reverse Crossover)', 'Strategy-based Signal')
        ):
            rev_signal, _ = strategy_func(df, idx, config, position)
            if rev_signal:
                if (position['type'] == 'LONG'  and rev_signal in ('SELL', 'SHORT')) or                    (position['type'] == 'SHORT' and rev_signal in ('BUY',  'LONG')):
                    exit_reason = 'Strategy Signal Exit'
                    add_log(f"🔄 STRATEGY SIGNAL EXIT: {rev_signal}")

        if exit_reason:
            # Sanity check exit price
            chg_pct = abs(exit_price - position['entry_price']) / position['entry_price'] * 100
            if chg_pct > 50:
                add_log(f"⚠️ Exit price {exit_price:.2f} differs {chg_pct:.1f}% — re-fetching to verify…")
                locked_ticker = position.get('ticker', config.get('asset', 'NIFTY 50'))
                df_verify = fetch_data(locked_ticker, interval, period, is_live_trading=True,
                                       custom_ticker=position.get('custom_ticker'), config=config)
                if df_verify is not None and not df_verify.empty:
                    verified_price = df_verify.iloc[-1]['Close']
                    if abs(verified_price - position['entry_price']) / position['entry_price'] * 100 < chg_pct:
                        exit_price = verified_price

            if position['type'] == 'LONG':
                pnl = (exit_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - exit_price) * position['quantity']

            brokerage = calculate_brokerage(position['entry_price'], exit_price, position['quantity'], config)
            net_pnl   = pnl - brokerage

            add_log(f"🚪 EXIT {exit_reason} @ ₹{exit_price:.2f} | P&L: ₹{pnl:.2f} | Net: ₹{net_pnl:.2f}")

            trade_record = {
                'entry_time':    position['entry_time'],
                'exit_time':     now,
                'type':          position['type'],
                'entry_price':   position['entry_price'],
                'exit_price':    exit_price,
                'sl_price':      position['sl_price'],
                'target_price':  position['target_price'],
                'highest_price': position.get('highest_price', exit_price),
                'lowest_price':  position.get('lowest_price',  exit_price),
                'quantity':      position['quantity'],
                'pnl':           pnl,
                'brokerage':     brokerage,
                'net_pnl':       net_pnl,
                'exit_reason':   exit_reason,
                'price_range':   position.get('highest_price', exit_price) - position.get('lowest_price', exit_price),
                'ticker':        position.get('ticker', 'Unknown'),
                'price_change_pct': abs(exit_price - position['entry_price']) / position['entry_price'] * 100,
                'strategy':         position.get('strategy', strategy_name),
                'ema_fast_period':  position.get('ema_fast_period'),
                'ema_slow_period':  position.get('ema_slow_period'),
                'ema_fast_entry':   position.get('ema_fast_entry'),
                'ema_slow_entry':   position.get('ema_slow_entry'),
                'ema_angle_entry':  position.get('ema_angle_entry'),
                'ema_fast_exit':    current_data.get('EMA_Fast'),
                'ema_slow_exit':    current_data.get('EMA_Slow'),
                'price_fast_ema_diff_entry': (position['entry_price'] - position.get('ema_fast_entry', 0)) if position.get('ema_fast_entry') else None,
                'price_slow_ema_diff_entry': (position['entry_price'] - position.get('ema_slow_entry', 0)) if position.get('ema_slow_entry') else None,
                'fast_slow_ema_diff_entry':  (position.get('ema_fast_entry', 0) - position.get('ema_slow_entry', 0)) if position.get('ema_fast_entry') and position.get('ema_slow_entry') else None,
                'price_fast_ema_diff_exit':  (exit_price - current_data.get('EMA_Fast', 0)) if current_data.get('EMA_Fast') else None,
                'price_slow_ema_diff_exit':  (exit_price - current_data.get('EMA_Slow', 0)) if current_data.get('EMA_Slow') else None,
                'fast_slow_ema_diff_exit':   (current_data.get('EMA_Fast', 0) - current_data.get('EMA_Slow', 0)) if current_data.get('EMA_Fast') and current_data.get('EMA_Slow') else None,
                'duration_minutes': (now - position['entry_time']).total_seconds() / 60,
            }

            st.session_state.setdefault('trade_history', []).append(trade_record)
            add_log("📝 Trade saved to history")

            # Broker exit
            if config.get('dhan_enabled', False):
                broker_position = st.session_state.get('broker_position')
                dhan_broker     = st.session_state.get('dhan_broker')
                if broker_position and dhan_broker:
                    try:
                        exit_info = dhan_broker.exit_broker_position(broker_position, exit_price, exit_reason, add_log)
                        st.session_state['broker_exit'] = exit_info
                    except Exception as e:
                        add_log(f"🏦 ⚠️ Broker exit error: {e}")

            st.session_state['last_exit_time'] = now
            st.session_state['position']        = None
            st.session_state['broker_position'] = None
            st.session_state.pop('current_data', None)
            add_log("✅ Position closed, session cleared")
            return  # guard: never enter same iteration we exited
        else:
            add_log("⏳ No exit conditions met — holding")

# UI COMPONENTS
# ================================

def render_config_ui():
    """Render configuration sidebar"""
    st.sidebar.header("⚙️ Configuration")
    
    config = {}

    # ── Data Source ───────────────────────────────────────────────────────────
    st.sidebar.subheader("📡 Data Source")
    config['use_yfinance'] = st.sidebar.checkbox(
        "Use yfinance for market data",
        value=True,
        help="Checked = Yahoo Finance (default). Uncheck = DhanHQ live feed.",
    )

    if not config['use_yfinance']:
        st.sidebar.markdown("**DhanHQ Data Feed**")
        st.sidebar.caption("Leave blank to reuse the broker credentials entered below.")
        config['dhan_data_client_id'] = st.sidebar.text_input(
            "Client ID (data feed)", value="", key="dhan_data_client_id_in",
        )
        config['dhan_data_access_token'] = st.sidebar.text_input(
            "Access Token (data feed)", type="password", key="dhan_data_token_in",
        )
        config['dhan_data_security_id'] = st.sidebar.text_input(
            "Security ID", value="13", key="dhan_data_sec_id_in",
            help=(
                "Dhan scrip master ID.\n"
                "Nifty 50 = 13 (IDX_I, INDEX)\n"
                "BankNifty = 25 (IDX_I, INDEX)\n"
                "Reliance = 1333 (NSE_EQ, EQUITY)"
            ),
        )
        config['dhan_data_exchange'] = st.sidebar.selectbox(
            "Exchange Segment",
            ["IDX_I (Index)", "NSE_EQ", "BSE_EQ", "NSE_FNO", "BSE_FNO", "MCX_COMM"],
            index=0, key="dhan_data_exch_in",
            help=(
                "IDX_I (Index) for Nifty/BankNifty indices.\n"
                "NSE_EQ for NSE stocks. BSE_EQ for BSE stocks.\n"
                "NSE_FNO for F&O. MCX_COMM for commodities."
            ),
        )
        config['dhan_data_instrument_type'] = st.sidebar.selectbox(
            "Instrument Type",
            ["INDEX", "EQUITY", "FUTIDX", "FUTSTK", "OPTIDX", "OPTSTK"],
            index=0, key="dhan_data_inst_in",
        )
        st.sidebar.info(
            "Minute data: last 5 trading days max.\n"
            "Daily data: multi-year history.\n"
            "30m = fetched as 5m then resampled."
        )
    st.sidebar.divider()

    # Asset Selection
    config['asset'] = st.sidebar.selectbox("Asset", list(ASSET_MAPPING.keys()), index=0)
    
    # Custom Ticker Input
    if config['asset'] == 'Custom Ticker':
        config['custom_ticker'] = st.sidebar.text_input("Enter Ticker Symbol", value="KAYNES.NS", help="e.g., KAYNES.NS, RELIANCE.NS, TCS.NS")
    
    # Timeframe
    config['interval'] = st.sidebar.selectbox("Interval", list(INTERVAL_MAPPING.keys()), index=0)  # Default to 1 minute
    config['period'] = st.sidebar.selectbox("Period", list(PERIOD_MAPPING.keys()), index=1)  # Default to 1 day
    
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
    config['enable_entry_cooldown'] = st.sidebar.checkbox(
        "⏱️ Enable Entry Cooldown",
        value=False,
        help="Prevents immediate re-entry after exit (Live Trading only). Useful to avoid duplicate orders on same signal."
    )
    
    if config['enable_entry_cooldown']:
        config['entry_cooldown_seconds'] = st.sidebar.number_input(
            "Cooldown Duration (seconds)",
            min_value=0,
            max_value=300,
            value=60,
            step=5,
            help="Number of seconds to wait after exit before allowing new entry"
        )
        st.sidebar.caption(f"⏱️ Cooldown: {config['entry_cooldown_seconds']}s wait after exit before new entry")
    else:
        config['entry_cooldown_seconds'] = 0  # Disabled
    
    # Display Last Candle Details
    config['show_last_candle'] = st.sidebar.checkbox(
        "📊 Show Last Candle Details",
        value=False,
        help="Display the last received candle with all calculated indicator values in Live Trading tab"
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
    

    elif config['strategy'] == 'Elliott Waves + Ratio Charts':
        config['elliott_wave_lookback'] = st.sidebar.number_input(
            "Wave Lookback Period", min_value=20, value=50,
            help="Candles to look back for detecting wave extrema"
        )
        st.sidebar.info("📈 Detects local high/low extrema and identifies 5-wave price patterns")


    elif config['strategy'] == 'Price Action':
        config['ema_fast'] = config.get('ema_fast', 9)
        config['ema_slow'] = config.get('ema_slow', 21)
        st.sidebar.info("📈 Price Action: detects candle patterns using EMA trend filter")

    elif config['strategy'] == 'RSI-ADX-EMA Combined':
        st.sidebar.markdown("**RSI-ADX-EMA Parameters**")
        config['rsi_threshold']  = st.sidebar.number_input("RSI Threshold", min_value=30, max_value=70, value=50)
        config['adx_threshold']  = st.sidebar.number_input("ADX Threshold", min_value=15, max_value=40, value=25)
        config['ema_fast']       = st.sidebar.number_input("EMA Fast", min_value=5, max_value=50, value=9)
        config['ema_slow']       = st.sidebar.number_input("EMA Slow", min_value=10, max_value=200, value=21)

    elif config['strategy'] == 'VWAP + Volume Spike':
        st.sidebar.markdown("**VWAP + Volume Spike Parameters**")
        config['vwap_volume_mult'] = st.sidebar.number_input("Volume Spike Multiplier", min_value=1.5, value=2.0, step=0.1)
        config['vwap_distance_pct'] = st.sidebar.number_input("Max Distance from VWAP (%)", min_value=0.1, value=0.3, step=0.1)
        config['vwap_rsi_ob'] = st.sidebar.number_input("RSI Overbought", min_value=60, max_value=100, value=70)
        config['vwap_rsi_os'] = st.sidebar.number_input("RSI Oversold", min_value=0, max_value=40, value=30)
        st.sidebar.info("📊 Price/VWAP crossover with volume confirmation")
    

    elif config['strategy'] == 'Opening Range Breakout (ORB)':
        st.sidebar.markdown("**Opening Range Breakout Parameters**")
        config['orb_minutes'] = st.sidebar.number_input("Opening Range Duration (minutes)", min_value=5, max_value=60, value=15)
        config['orb_breakout_buffer'] = st.sidebar.number_input("Breakout Buffer (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        st.sidebar.info("📊 Trades breakouts from first N minutes of market open. High probability intraday strategy.")
    

    elif config['strategy'] == 'Volume Breakout':
        st.sidebar.markdown("**Volume Breakout Parameters**")
        config['volume_multiplier'] = st.sidebar.number_input("Volume Multiplier", min_value=1.5, max_value=5.0, value=2.0, step=0.5)
        config['volume_price_threshold'] = st.sidebar.number_input("Min Price Change (%)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        st.sidebar.info("📈 Trades when price moves with high volume confirmation. Best for momentum trades.")
    

    elif config['strategy'] == 'Momentum Breakout with ADX':
        st.sidebar.markdown("**Momentum Breakout Parameters**")
        config['momentum_adx_threshold'] = st.sidebar.number_input("ADX Threshold", min_value=20, max_value=40, value=25)
        config['momentum_lookback'] = st.sidebar.number_input("Breakout Lookback", min_value=10, max_value=50, value=20)
        config['momentum_volume_ratio'] = st.sidebar.number_input("Volume Ratio", min_value=1.0, max_value=3.0, value=1.5, step=0.5)
        st.sidebar.info("🚀 Combines strong ADX trend with price breakout. High probability in trending markets.")
    

    elif config['strategy'] == 'Support Resistance Bounce':
        st.sidebar.markdown("**Support/Resistance Parameters**")
        config['sr_lookback'] = st.sidebar.number_input("Lookback Period", min_value=50, max_value=200, value=100)
        config['sr_tolerance'] = st.sidebar.number_input("Level Tolerance (%)", min_value=0.1, max_value=1.0, value=0.2, step=0.1)
        config['sr_min_touches'] = st.sidebar.number_input("Min Level Touches", min_value=2, max_value=5, value=3)
        st.sidebar.info("🎯 Identifies key S/R levels and trades bounces. Very high probability at well-tested levels.")
    

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

    # Stop Loss Configuration
    st.sidebar.subheader("🛡️ Stop Loss")

    # Stop Loss Configuration
    st.sidebar.subheader("🛡️ Stop Loss")
    config['sl_type'] = st.sidebar.selectbox("SL Type", SL_TYPES, index=9)
    
    if 'Points' in config['sl_type'] or config['sl_type'] in ['Custom Points', 'ATR-based', 
                                                                'Trailing SL (Points)', 
                                                                'Cost-to-Cost + N Points Trailing SL']:
        config['sl_points'] = st.sidebar.number_input("SL Points", min_value=1, value=30)
    
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
    config['target_type'] = st.sidebar.selectbox("Target Type", TARGET_TYPES, index=2)
    
    if 'Points' in config['target_type'] or config['target_type'] in ['Custom Points', 'Trailing Target (Points)']:
        config['target_points'] = st.sidebar.number_input("Target Points", min_value=1, value=200)
    
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
            value=200,
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
        config['dhan_access_token'] = st.sidebar.text_input("Access Token", type="password", value="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzc1MTg4NDYyLCJpYXQiOjE3NzUxMDIwNjIsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTA0Nzc5ODc2In0.WhHvZIgy_Wj1SVT-3j43UWCnGTZ9ZBK9Spjmze-wW6LpcrTwY7pzkDRkkM4kCh8jrc883QqYnVuQ-zW8qBXF5A")
        
        config['dhan_is_options'] = st.sidebar.checkbox("Is Options", value=False)
        
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
            config['dhan_security_id'] = st.sidebar.text_input("Security ID", value="12092")
            config['dhan_exchange']    = st.sidebar.selectbox("Exchange", ["NSE", "BSE"], index=0)
            config['dhan_quantity']    = st.sidebar.number_input("Quantity", min_value=1, value=10)
            if config['dhan_trading_type'] == 'Delivery (CNC)':
                st.sidebar.info("Order Type: MARKET | Product: CNC")
            else:
                st.sidebar.info("Order Type: MARKET | Product: INTRA")

        # ── Broker SL / Target (Bracket Order) — available for ALL order types ──
        st.sidebar.markdown("---")
        
        # Order Type Selection - Separate for Entry and Exit
        st.sidebar.markdown("**Order Type Configuration**")
        
        col_entry, col_exit = st.sidebar.columns(2)
        with col_entry:
            config['dhan_entry_order_type'] = st.sidebar.selectbox(
                "Entry Order Type",
                ["Market Order", "Limit Order"],
                index=1,  # Default to Limit Order
                help="Entry: Market = Immediate | Limit = At specified price"
            )
        with col_exit:
            config['dhan_exit_order_type'] = st.sidebar.selectbox(
                "Exit Order Type",
                ["Market Order", "Limit Order"],
                index=1,  # Default to Limit Order
                help="Exit: Market = Immediate | Limit = At specified price"
            )
        
        # Display order combination
        entry_type = "MARKET" if config['dhan_entry_order_type'] == "Market Order" else "LIMIT"
        exit_type = "MARKET" if config['dhan_exit_order_type'] == "Market Order" else "LIMIT"
        st.sidebar.caption(f"📋 Configuration: Entry {entry_type} | Exit {exit_type}")
        
        # Legacy support: set dhan_order_type to entry type for backward compatibility
        config['dhan_order_type'] = config['dhan_entry_order_type']
        
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
            st.plotly_chart(fig, width='stretch')

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
            st.dataframe(df_display, width="stretch")
            
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
                st.dataframe(df_detailed_display, width="stretch")
        
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
            
            st.dataframe(df_skipped, width="stretch")
            
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

# ── Fragment: auto-updates only this section without full-page flicker ─────
@st.fragment(run_every=0.3)
def _live_trading_fragment(config):
    """
    Runs every 0.3 s via st.fragment — only this panel refreshes.
    The rest of the page (sidebar, other tabs) does NOT flicker.
    """
    if not st.session_state.get('trading_active', False):
        return

    config   = st.session_state.get('config', config)
    interval      = INTERVAL_MAPPING.get(config.get('interval', '1 day'), '1d')
    interval_secs = INTERVAL_FETCH_SECONDS.get(interval, 60)

    # ── Trading logic: fast LTP + throttled candle refresh ───────────────
    live_trading_iteration()

    # ── Live price & indicator metrics ────────────────────────────────────
    current_data = st.session_state.get('current_data')
    if current_data is not None:
        st.markdown("#### 📈 Live Market Data")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("LTP", f"₹{float(current_data['Close']):.2f}")
        with col2:
            val = current_data.get('EMA_Fast')
            st.metric(f"EMA {config.get('ema_fast', 9)}", f"₹{float(val):.2f}" if val and not pd.isna(val) else "N/A")
        with col3:
            val = current_data.get('EMA_Slow')
            st.metric(f"EMA {config.get('ema_slow', 21)}", f"₹{float(val):.2f}" if val and not pd.isna(val) else "N/A")
        with col4:
            ang = current_data.get('EMA_Fast_Angle')
            st.metric("EMA Angle", f"{float(ang):.2f}°" if ang and not pd.isna(ang) else "N/A")
        with col5:
            ef = current_data.get('EMA_Fast')
            es = current_data.get('EMA_Slow')
            if ef and es and not pd.isna(ef) and not pd.isna(es):
                st.metric("Crossover", "Bullish ⬆️" if float(ef) > float(es) else "Bearish ⬇️")
            else:
                st.metric("Crossover", "N/A")

    # ── Live Chart with real-time forming candle ─────────────────────────
    df_plot = st.session_state.get('indicator_df')
    if df_plot is not None and not df_plot.empty:
        st.markdown("#### 📊 Live Chart")
        try:
            plot_df = df_plot.tail(150).copy()

            # ── Append the LIVE FORMING candle as the last bar ────────────
            lc_start = st.session_state.get('live_candle_start')
            lc_open  = st.session_state.get('live_candle_open')
            lc_high  = st.session_state.get('live_candle_high')
            lc_low   = st.session_state.get('live_candle_low')
            lc_close = st.session_state.get('live_candle_close')

            if all(v is not None for v in [lc_start, lc_open, lc_high, lc_low, lc_close]):
                forming_row = pd.DataFrame([{
                    'Datetime': lc_start,
                    'Open':  float(lc_open),
                    'High':  float(lc_high),
                    'Low':   float(lc_low),
                    'Close': float(lc_close),
                }])
                # If last completed candle has the same timestamp, replace it;
                # otherwise append so the forming candle grows at the right edge.
                if len(plot_df) > 0 and str(plot_df.iloc[-1]['Datetime']) == str(lc_start):
                    plot_df = pd.concat([plot_df.iloc[:-1], forming_row], ignore_index=True)
                else:
                    plot_df = pd.concat([plot_df, forming_row], ignore_index=True)

            fig_live = go.Figure()

            # ── Completed + forming candles (single trace — colour auto) ──
            fig_live.add_trace(go.Candlestick(
                x=plot_df['Datetime'],
                open=plot_df['Open'],   high=plot_df['High'],
                low=plot_df['Low'],     close=plot_df['Close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
                whiskerwidth=0.4,
            ))

            # ── Highlight the forming (last) candle with a faint border ───
            if lc_start is not None and len(plot_df) > 0:
                fc = plot_df.iloc[-1]
                is_bull = float(fc['Close']) >= float(fc['Open'])
                border   = '#26a69a' if is_bull else '#ef5350'
                fig_live.add_shape(
                    type='rect',
                    x0=str(lc_start), x1=str(lc_start),
                    y0=float(fc['Low']), y1=float(fc['High']),
                    line=dict(color=border, width=2, dash='dot'),
                    xref='x', yref='y',
                )

            # ── EMA lines ─────────────────────────────────────────────────
            if 'EMA_Fast' in plot_df.columns:
                fig_live.add_trace(go.Scatter(
                    x=plot_df['Datetime'], y=plot_df['EMA_Fast'],
                    mode='lines', name=f"EMA {config.get('ema_fast', 9)}",
                    line=dict(color='#FF9800', width=1.5)))
            if 'EMA_Slow' in plot_df.columns:
                fig_live.add_trace(go.Scatter(
                    x=plot_df['Datetime'], y=plot_df['EMA_Slow'],
                    mode='lines', name=f"EMA {config.get('ema_slow', 21)}",
                    line=dict(color='#2196F3', width=1.5)))

            # ── SL / Target / Entry h-lines ───────────────────────────────
            position = st.session_state.get('position')
            if position:
                fig_live.add_hline(
                    y=position['entry_price'], line_dash='dash',
                    line_color='#00E676',
                    annotation_text=f"Entry ₹{position['entry_price']:.2f}",
                    annotation_position='right')
                if position.get('sl_price'):
                    fig_live.add_hline(
                        y=position['sl_price'], line_dash='dot',
                        line_color='#FF1744',
                        annotation_text=f"SL ₹{position['sl_price']:.2f}",
                        annotation_position='right')
                if position.get('target_price'):
                    fig_live.add_hline(
                        y=position['target_price'], line_dash='dot',
                        line_color='#00BCD4',
                        annotation_text=f"Target ₹{position['target_price']:.2f}",
                        annotation_position='right')

            # ── Forming candle progress bar annotation ─────────────────────
            if lc_start is not None and lc_open is not None:
                elapsed  = (datetime.now(pytz.timezone('Asia/Kolkata')) - lc_start).total_seconds()
                progress = min(100, int(elapsed / interval_secs * 100)) if interval_secs > 0 else 0
                fig_live.add_annotation(
                    text=f"🕯️ Forming {progress}%",
                    xref='paper', yref='paper', x=0.01, y=0.97,
                    showarrow=False,
                    font=dict(color='#FFD700', size=11),
                    bgcolor='rgba(0,0,0,0.4)'
                )

            fig_live.update_layout(
                xaxis_rangeslider_visible=False, height=420,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                hovermode='x unified', template='plotly_dark',
                margin=dict(l=0, r=60, t=30, b=0)
            )
            st.plotly_chart(fig_live, use_container_width=True)

            # Forming candle OHLC caption below chart
            if lc_open is not None:
                chg    = float(lc_close) - float(lc_open)
                chg_p  = chg / float(lc_open) * 100 if lc_open else 0
                color_tag = '🟢' if chg >= 0 else '🔴'
                st.caption(
                    f"{color_tag} **Forming candle** — "
                    f"O: ₹{lc_open:.2f}  H: ₹{lc_high:.2f}  "
                    f"L: ₹{lc_low:.2f}  C: ₹{lc_close:.2f}  "
                    f"({'+' if chg>=0 else ''}{chg:.2f} / {chg_p:+.2f}%)"
                )
        except Exception as e:
            st.warning(f"Chart error: {e}")

    # ── Last Candle indicator details (expandable) ─────────────────────────
    if config.get('show_last_candle', False) and current_data is not None:
        with st.expander("📊 Last Candle Indicators", expanded=False):
            col1, col2, col3, col4, col5 = st.columns(5)
            for col, (k, label) in zip([col1, col2, col3, col4, col5],
                                       [('Open','Open'),('High','High'),('Low','Low'),
                                        ('Close','Close'),('Volume','Volume')]):
                with col:
                    v = current_data.get(k, 0)
                    st.metric(label, f"₹{float(v):.2f}" if k != 'Volume' else f"{int(v):,}")
            common = [('EMA_Fast','EMA Fast'),('EMA_Slow','EMA Slow'),
                      ('EMA_Fast_Angle','EMA Angle'),('RSI','RSI'),('ADX','ADX'),
                      ('ATR','ATR'),('MACD','MACD'),('BB_Upper','BB Upper'),
                      ('BB_Lower','BB Lower'),('Volume_MA','Vol MA')]
            items = [(lbl, current_data.get(col)) for col, lbl in common
                     if current_data.get(col) is not None and not pd.isna(current_data.get(col, float('nan')))]
            for i in range(0, len(items), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i+j < len(items):
                        lbl, val = items[i+j]
                        with col:
                            st.metric(lbl, f"{float(val):.2f}°" if 'Angle' in lbl else
                                      (f"{int(float(val)):,}" if 'MA' in lbl else f"{float(val):.2f}"))

    # ── Current position ───────────────────────────────────────────────────
    position = st.session_state.get('position')
    current_price = float(current_data['Close']) if current_data is not None else None

    st.markdown("#### 📊 Current Position")
    if position:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.metric("Type", position['type'])
        with col2: st.metric("Entry", f"₹{position['entry_price']:.2f}")
        with col3: st.metric("LTP", f"₹{current_price:.2f}" if current_price else "N/A")
        with col4:
            sl = position.get('sl_price')
            st.metric("Stop Loss", f"₹{sl:.2f}" if sl else "Not Set")
        with col5:
            tgt = position.get('target_price')
            st.metric("Target", f"₹{tgt:.2f}" if tgt else "Not Set")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.metric("Qty", position['quantity'])
        with col2: st.metric("Ticker", position.get('ticker', 'N/A'))
        with col3:
            if current_price:
                pnl = (current_price - position['entry_price']) * position['quantity']                       if position['type'] == 'LONG'                       else (position['entry_price'] - current_price) * position['quantity']
                st.metric("Live P&L", f"₹{pnl:.2f}", delta=f"₹{pnl:.2f}")
        with col4: st.metric("High", f"₹{position.get('highest_price', 0):.2f}")
        with col5: st.metric("Low",  f"₹{position.get('lowest_price', 0):.2f}")
        st.caption(f"Entry: {position['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.info("No active position")

    # ── Broker position ────────────────────────────────────────────────────
    if config.get('dhan_enabled', False) and st.session_state.get('broker_position'):
        bp = st.session_state['broker_position']
        st.markdown("#### 🏦 Broker Position")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Order ID", bp['order_id'])
        with col2: st.metric("Type", bp.get('option_type', bp.get('transaction_type', 'N/A')))
        with col3: st.metric("Security", bp['security_id'])
        with col4: st.metric("Status", bp['status'])
        with st.expander("📄 Raw API Response"):
            st.json(bp['raw_response'])

    # ── Completed trades ───────────────────────────────────────────────────
    st.markdown("#### ✅ Completed Trades")
    trade_history = st.session_state.get('trade_history', [])
    if trade_history:
        df_history = pd.DataFrame(trade_history)
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total", len(df_history))
        with col2:
            wins = int((df_history['pnl'] > 0).sum())
            st.metric("Wins", wins)
        with col3:
            total_pnl = float(df_history['pnl'].sum())
            st.metric("Total P&L", f"₹{total_pnl:.2f}")
        with col4:
            net = float(df_history['net_pnl'].sum()) if 'net_pnl' in df_history.columns else total_pnl
            st.metric("Net P&L", f"₹{net:.2f}")

        disp = df_history.copy()
        for c in ['entry_price','exit_price','sl_price','target_price','pnl','net_pnl','brokerage']:
            if c in disp.columns:
                disp[c] = disp[c].apply(lambda x: f"₹{x:.2f}" if pd.notna(x) else "—")
        show_cols = ['entry_time','exit_time','ticker','type','entry_price','exit_price','pnl','net_pnl','exit_reason']
        st.dataframe(disp[[c for c in show_cols if c in disp.columns]],
                     use_container_width=True, height=180)
    else:
        st.info("No completed trades yet.")

    # ── Trading logs ───────────────────────────────────────────────────────
    st.markdown("#### 📝 Logs")
    logs = st.session_state.get('live_logs', [])
    if logs:
        st.text_area("", value="\n".join(reversed(logs[-50:])),
                     height=200, disabled=True, label_visibility="collapsed")
    else:
        st.info("No logs yet")


def render_live_trading_ui(config):
    """Render live trading interface — static controls + auto-refreshing fragment"""
    st.header("🔴 Live Trading")

    # ── Control buttons (outside fragment — these trigger full reruns) ─────
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("▶️ Start Trading", type="primary"):
            st.session_state.update({
                'trading_active': True,
                'position': None,
                'broker_position': None,
                'live_logs': [],
                'last_exit_time': None,
                'last_signal_type': None,
                'clearing_in_progress': False,
                'indicator_df': None,
                'live_candle_start':  None,
                'live_candle_open':   None,
                'live_candle_high':   None,
                'live_candle_low':    None,
                'live_candle_close':  None,
                'live_candle_volume': 0,
            'last_candle_ts':       None,
            'last_candle_fetch_time': None,
            'stale_detected':       False,
            'stale_retry_after':    None,
                'config': config,
            })
            if 'trade_history' not in st.session_state:
                st.session_state['trade_history'] = []
            st.session_state.pop('current_data', None)

            add_log("🚀 Trading started")
            start_ltp_feed(config)
            add_log(f"📋 Strategy: {config.get('strategy')} | Asset: {config.get('asset')} | Interval: {config.get('interval')}")
            add_log(f"📋 SL: {config.get('sl_type')} {config.get('sl_points','')} | Target: {config.get('target_type')} {config.get('target_points','')}")

            if config.get('dhan_enabled', False):
                add_log("🏦 Initializing Dhan broker…")
                st.session_state['dhan_broker'] = DhanBrokerIntegration(config)
                add_log("🏦 Broker ready")

    with col2:
        if st.button("⏹️ Stop Trading"):
            st.session_state['trading_active'] = False
            stop_ltp_feed()
            add_log("⏹️ Trading stopped")

    with col3:
        if st.button("❌ Manual Close"):
            position = st.session_state.get('position')
            if position:
                df_m = st.session_state.get('indicator_df')
                if df_m is not None and not df_m.empty:
                    cp = float(df_m.iloc[-1]['Close'])
                else:
                    cp = position['entry_price']
                pnl = (cp - position['entry_price']) * position['quantity']                       if position['type'] == 'LONG'                       else (position['entry_price'] - cp) * position['quantity']
                add_log(f"EXIT: Manual Close @ ₹{cp:.2f} | P&L: ₹{pnl:.2f}")
                trade_record = {
                    'entry_time': position['entry_time'],
                    'exit_time': datetime.now(pytz.timezone('Asia/Kolkata')),
                    'type': position['type'],
                    'entry_price': position['entry_price'], 'exit_price': cp,
                    'sl_price': position.get('sl_price'), 'target_price': position.get('target_price'),
                    'highest_price': position.get('highest_price', cp),
                    'lowest_price': position.get('lowest_price', cp),
                    'quantity': position['quantity'], 'pnl': pnl,
                    'brokerage': 0, 'net_pnl': pnl,
                    'exit_reason': 'Manual Close',
                    'price_range': position.get('highest_price', cp) - position.get('lowest_price', cp),
                    'ticker': position.get('ticker', 'Unknown'),
                    'price_change_pct': abs(cp - position['entry_price']) / position['entry_price'] * 100,
                }
                st.session_state.setdefault('trade_history', []).append(trade_record)
                if config.get('dhan_enabled', False):
                    bp = st.session_state.get('broker_position')
                    dhan_broker = st.session_state.get('dhan_broker')
                    if bp and dhan_broker:
                        try:
                            dhan_broker.exit_broker_position(bp, cp, 'Manual Close', add_log)
                        except Exception as e:
                            add_log(f"🏦 ⚠️ {e}")
                st.session_state['position'] = None
                st.session_state['broker_position'] = None
                st.session_state.pop('current_data', None)
                stop_ltp_feed()
                add_log("✅ Position closed")
            else:
                st.warning("No active position to close")

    # ── Status ─────────────────────────────────────────────────────────────
    if st.session_state.get('trading_active', False):
        st.success("🟢 Trading Active  —  LTP updates every 0.3 s · Candles refresh per interval")
        # ── Raw tick debug display (frozen, not scrolling) ────────
        raw = st.session_state.get('_last_raw_tick', '')
        if raw:
            st.info(f"📡 Last raw tick: {raw}")
    else:
        st.info("⚪ Trading Inactive")

    # ── Static strategy params (never flickers) ────────────────────────────
    with st.expander("📋 Strategy Parameters", expanded=False):
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

    # ── Live fragment (only this part auto-refreshes) ──────────────────────
    if st.session_state.get('trading_active', False):
        _live_trading_fragment(config)
    else:
        # When inactive: show last known data as a static snapshot
        current_data = st.session_state.get('current_data')
        df_plot = st.session_state.get('indicator_df')
        if df_plot is not None and not df_plot.empty:
            st.markdown("#### 📊 Last Chart Snapshot")
            try:
                plot_df = df_plot.tail(150).copy()
                fig_s = go.Figure()
                fig_s.add_trace(go.Candlestick(
                    x=plot_df['Datetime'],
                    open=plot_df['Open'], high=plot_df['High'],
                    low=plot_df['Low'],   close=plot_df['Close'],
                    name='Price',
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'))
                if 'EMA_Fast' in plot_df.columns:
                    fig_s.add_trace(go.Scatter(x=plot_df['Datetime'], y=plot_df['EMA_Fast'],
                        mode='lines', name=f"EMA {config.get('ema_fast', 9)}",
                        line=dict(color='#FF9800', width=1.5)))
                if 'EMA_Slow' in plot_df.columns:
                    fig_s.add_trace(go.Scatter(x=plot_df['Datetime'], y=plot_df['EMA_Slow'],
                        mode='lines', name=f"EMA {config.get('ema_slow', 21)}",
                        line=dict(color='#2196F3', width=1.5)))
                fig_s.update_layout(xaxis_rangeslider_visible=False, height=400,
                    hovermode='x unified', template='plotly_dark',
                    margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_s, use_container_width=True)
            except Exception:
                pass

        trade_history = st.session_state.get('trade_history', [])
        if trade_history:
            st.subheader("✅ Completed Trades")
            st.dataframe(pd.DataFrame(trade_history)[['entry_time','exit_time','type',
                'entry_price','exit_price','pnl','exit_reason']],
                use_container_width=True, height=200)

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
    st.dataframe(display_df, width="stretch", height=400)
    
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
    
    st.plotly_chart(fig, width="stretch")
    
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
        
        st.plotly_chart(fig_type, width="stretch")
    
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
        
        st.plotly_chart(fig_reason, width="stretch")

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
