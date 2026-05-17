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

def fetch_ltp(config):
    """
    Fetch Last Traded Price only — fast, lightweight call.

    yfinance path : Ticker.fast_info.last_price  (no full OHLCV download)
    Dhan path     : get_ltp_data()  or falls back to intraday_minute_data last row

    Returns float price, or None on any failure.
    """
    try:
        if not config.get('use_yfinance', True):
            # ── Dhan LTP ─────────────────────────────────────────────────
            try:
                from dhanhq import dhanhq
                client_id    = config.get('dhan_data_client_id') or config.get('dhan_client_id', '')
                access_token = config.get('dhan_data_access_token') or config.get('dhan_access_token', '')
                security_id  = str(config.get('dhan_data_security_id', ''))
                exch_raw     = config.get('dhan_data_exchange', 'IDX_I (Index)')

                if not client_id or not access_token or not security_id:
                    return None

                seg_map = {
                    'IDX_I (Index)': 'IDX_I', 'NSE_EQ': 'NSE_EQ', 'BSE_EQ': 'BSE_EQ',
                    'NSE_FNO': 'NSE_FNO', 'BSE_FNO': 'BSE_FNO', 'MCX_COMM': 'MCX_COMM',
                }
                exch_seg = seg_map.get(exch_raw, 'NSE_EQ')
                dhan = dhanhq(client_id, access_token)

                # Try native LTP endpoint if available in installed version
                if hasattr(dhan, 'get_ltp_data'):
                    resp = dhan.get_ltp_data(securities={exch_seg: [security_id]})
                    if resp and resp.get('status') == 'success':
                        data = resp.get('data', {})
                        for seg_val in data.values():
                            if isinstance(seg_val, dict):
                                for sec_val in seg_val.values():
                                    if isinstance(sec_val, dict):
                                        ltp = (sec_val.get('last_price')
                                               or sec_val.get('ltp')
                                               or sec_val.get('LTP'))
                                        if ltp:
                                            return float(ltp)

                # Fallback: use last row of latest 1-min candle (still faster than full history)
                try:
                    ist   = pytz.timezone('Asia/Kolkata')
                    today = datetime.now(ist)
                    from_date = (today - timedelta(days=1)).strftime('%Y-%m-%d')
                    to_date   = today.strftime('%Y-%m-%d')
                    instr     = config.get('dhan_data_instrument_type', 'INDEX')
                    resp2 = dhan.intraday_minute_data(
                        security_id=security_id, exchange_segment=exch_seg,
                        instrument_type=instr, from_date=from_date,
                        to_date=to_date, interval=1,
                    )
                    if resp2 and resp2.get('status') == 'success':
                        raw = resp2.get('data', {})
                        closes = raw.get('close', []) if isinstance(raw, dict) else [r.get('close') for r in raw]
                        if closes:
                            return float(closes[-1])
                except Exception:
                    pass
                return None
            except Exception:
                return None
        else:
            # ── yfinance fast_info ────────────────────────────────────────
            ticker_name   = config.get('asset', 'NIFTY 50')
            custom_ticker = config.get('custom_ticker', None)
            symbol = custom_ticker if (ticker_name == 'Custom Ticker' and custom_ticker)                      else ASSET_MAPPING.get(ticker_name, ticker_name)
            try:
                ticker_obj = yf.Ticker(symbol)
                ltp = getattr(ticker_obj.fast_info, 'last_price', None)
                if ltp and float(ltp) > 0:
                    return float(ltp)
            except Exception:
                pass
            return None
    except Exception:
        return None

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
    # EMA Fast and Slow (TradingView-compatible calculation)
    ema_fast = config.get('ema_fast', 9)
    ema_slow = config.get('ema_slow', 21)
    
    # Calculate EMA with exact TradingView parameters
    # TradingView uses: alpha = 2/(length+1), which is pandas span parameter
    # adjust=False ensures we use the recursive formula like TradingView
    # min_periods=ema_fast ensures we start calculation after sufficient data
    df['EMA_Fast'] = df['Close'].ewm(span=ema_fast, adjust=False, min_periods=ema_fast).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=ema_slow, adjust=False, min_periods=ema_slow).mean()
    
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
    'EMA Crossover':              check_ema_crossover_strategy,
    'Simple Buy':                 check_simple_buy_strategy,
    'Simple Sell':                check_simple_sell_strategy,
    'Price Crosses Threshold':    check_price_crosses_threshold,
    'Percentage Change':          check_percentage_change,
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
    Single iteration of live trading loop.

    Architecture (always active — no toggle needed):
    • fetch_ltp()     → fast price every refresh (~0.3 s)
    • fetch_data()    → OHLCV + indicators, throttled to the selected timeframe
                        (1m → every 60 s, 5m → every 300 s, etc.)

    SL / 
