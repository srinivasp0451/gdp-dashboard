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

Version: 2.0

What changed in v2.0
--------------------
1. Indicators are now calculated exactly like TradingView (Pine Script):
   - ta.ema  -> SMA-seeded EMA          - ta.rsi -> Wilder RMA
   - ta.atr  -> RMA of True Range       - ta.dmi/ADX -> TradingView DMI formula
   - ta.bb   -> population std-dev      - SuperTrend -> ta.supertrend logic
   - VWAP    -> session anchored (hlc3) - Ichimoku cloud displacement 26 (offset 25)
   Extra history is downloaded as warm-up so values exist from the very first
   candle of the session (no more "NA" till 9:29 for EMA-9 on 1 min).
2. Manual square-off rewritten (shared close_position() helper, robust price fetch).
3. Swing / candle based SL use the real swing/candle level (points are only a
   fallback when no valid level exists). Swings are now detected without
   look-ahead (confirmed pivots, like ta.pivothigh / ta.pivotlow).
4. Trailing SL AND Trailing Target for current/previous candle and
   current/previous swing high/low.
5. Signals are evaluated on CLOSED candles by default and each signal candle can
   trigger only one entry (no repeated entries while fast EMA > slow EMA).
   Several strategy bugs fixed (see check_* functions).
6. 27 optional entry filters (RSI, MACD, Volume, BB, Regime, SMC BOS/CHoCH/
   Inducement/OB/Mitigation/FVG, Liquidity sweep, Fibonacci, EMA pullback,
   Multi EMA/SMA, Volatility, ATR, ORB, Pivot, HH/HL, VWAP, SuperTrend,
   ROC, Stochastic, Candle strength) - all disabled by default.
7. New defaults, new periods (7 days, 60 days, 3 years, 10 years, Max), monthly
   timeframe, automatic yfinance limit handling, full config display in the
   Live Trading tab.
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

IST = pytz.timezone('Asia/Kolkata')

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
    "1 month": "1mo",
}

# Minutes per candle (used to detect the still-forming candle)
INTERVAL_MINUTES = {
    '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60,
    '1d': 1440, '1wk': 7 * 1440, '1mo': 31 * 1440,
}

INTRADAY_INTERVALS = {'1m', '5m', '15m', '30m', '1h'}

PERIOD_MAPPING = {
    "1 day": "1d",
    "5 days": "5d",
    "7 days": "7d",
    "1 month": "1mo",
    "60 days": "60d",
    "3 months": "3mo",
    "6 months": "6mo",
    "1 year": "1y",
    "2 years": "2y",
    "3 years": "3y",
    "5 years": "5y",
    "10 years": "10y",
    "Max": "max",
}

# Calendar days represented by each period code (None = all available history)
PERIOD_DAYS = {
    '1d': 1, '5d': 5, '7d': 7, '1mo': 30, '60d': 60, '3mo': 91, '6mo': 182,
    '1y': 365, '2y': 730, '3y': 1095, '5y': 1826, '10y': 3652, 'max': None,
}

# Period strings Yahoo accepts directly
YF_STANDARD_PERIODS = {'1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'}

# yfinance / Yahoo intraday history limits (calendar days back from today)
#   1m  : last ~30 days, max 8 days per request (fetched in chunks automatically)
#   5m-30m : last 60 days
#   1h  : last 730 days
INTERVAL_MAX_DAYS = {'1m': 29, '5m': 59, '15m': 59, '30m': 59, '1h': 729}

# Extra history downloaded before the requested period so that every indicator
# is fully "warmed up" (same values as TradingView from the first candle shown)
WARMUP_EXTRA_DAYS = {
    '1m': 4, '5m': 15, '15m': 40, '30m': 59, '1h': 180,
    '1d': 1500, '1wk': 7000, '1mo': 15000,
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
    "Elliott Waves + Ratio Charts",
    "Opening Range Breakout (ORB)",
    "Pivot Point Reversal",
    "Ichimoku Cloud",
    "Volume Breakout",
    "Gap Trading Strategy",
    "Mean Reversion with Bollinger Bands",
    "Momentum Breakout with ADX",
    "Support Resistance Bounce"
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
    "Trailing Target + Current Candle",
    "Trailing Target + Previous Candle",
    "Trailing Target + Current Swing",
    "Trailing Target + Previous Swing",
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

# SL / Target types that are based on a candle or swing level
SL_LEVEL_TOKENS = {
    'Current Candle Low/High':       ['cur_candle', 'prev_candle', 'cur_swing'],
    'Trailing SL + Current Candle':  ['cur_candle', 'prev_candle', 'cur_swing'],
    'Previous Candle Low/High':      ['prev_candle', 'cur_candle', 'cur_swing'],
    'Trailing SL + Previous Candle': ['prev_candle', 'cur_candle', 'cur_swing'],
    'Current Swing Low/High':        ['cur_swing', 'prev_candle'],
    'Trailing SL + Current Swing':   ['cur_swing', 'prev_candle'],
    'Previous Swing Low/High':       ['prev_swing', 'cur_swing', 'prev_candle'],
    'Trailing SL + Previous Swing':  ['prev_swing', 'cur_swing', 'prev_candle'],
}
TRAIL_SL_LEVEL_TOKEN = {
    'Trailing SL + Current Candle': 'cur_candle',
    'Trailing SL + Previous Candle': 'prev_candle',
    'Trailing SL + Current Swing': 'cur_swing',
    'Trailing SL + Previous Swing': 'prev_swing',
}
TARGET_LEVEL_TOKENS = {
    'Current Candle Low/High':           ['cur_candle', 'cur_swing'],
    'Trailing Target + Current Candle':  ['cur_candle', 'cur_swing'],
    'Previous Candle Low/High':          ['prev_candle', 'cur_candle', 'cur_swing'],
    'Trailing Target + Previous Candle': ['prev_candle', 'cur_candle', 'cur_swing'],
    'Current Swing Low/High':            ['cur_swing', 'prev_candle'],
    'Trailing Target + Current Swing':   ['cur_swing', 'prev_candle'],
    'Previous Swing Low/High':           ['prev_swing', 'cur_swing'],
    'Trailing Target + Previous Swing':  ['prev_swing', 'cur_swing'],
}
TRAIL_TARGET_LEVEL_TOKEN = {
    'Trailing Target + Current Candle': 'cur_candle',
    'Trailing Target + Previous Candle': 'prev_candle',
    'Trailing Target + Current Swing': 'cur_swing',
    'Trailing Target + Previous Swing': 'prev_swing',
}

SIGNAL_EXIT_TYPES = {
    'Signal-based (Reverse Crossover)',
    'Strategy-based Signal',
    'Trailing Target + Signal Based',
}

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
        self.dhanhq_module = None
        
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

                if use_broker_sl and op and not is_exit:
                    # ── Bracket Order (BO) - always uses LIMIT ──────────────
                    lmt_price = float(op.get('price', 0))
                    bo_profit  = float(op.get('boProfitValue', 0))
                    bo_sl      = float(op.get('boStopLossValue', 0))
                    trail_sl   = float(op.get('trailStopLoss', 0))

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
                    order_response['error'] = str((response or {}).get('remarks', 'Unknown error'))

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
                'timestamp': datetime.now(IST),
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
                'timestamp': datetime.now(IST),
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
            'timestamp': datetime.now(IST),
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
                                error_msg = f"Failed to cancel order {order_id}: {(cancel_response or {}).get('remarks', 'Unknown')}"
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
                                error_msg = f"Failed to cancel order {order_id}: {(cancel_response or {}).get('remarks', 'Unknown')}"
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
                                error_msg = f"Failed to close position {order_id}: {(close_response or {}).get('remarks', 'Unknown')}"
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

def _resolve_yf_ticker(ticker_symbol, custom_ticker=None):
    """Map UI asset name to a yfinance ticker"""
    if ticker_symbol == "Custom Ticker" and custom_ticker:
        return str(custom_ticker).strip()
    return ASSET_MAPPING.get(ticker_symbol, ticker_symbol)


def _normalize_yf_df(df):
    """Flatten yfinance output into Datetime/Open/High/Low/Close/Volume in IST"""
    if df is None or df.empty:
        return None

    # ── Handle newer yfinance MultiIndex columns e.g. ("Close","^NSEI") → "Close"
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]

    # ── Move date index into a column named "Datetime"
    df = df.reset_index()
    if 'Datetime' not in df.columns:
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'Datetime'})
        else:
            df = df.rename(columns={df.columns[0]: 'Datetime'})

    # ── Timezone: intraday data is tz-aware (UTC/exchange) → convert;
    #    daily/weekly/monthly data can be tz-naive → localize (fixes old crash)
    dt = pd.to_datetime(df['Datetime'])
    if getattr(dt.dt, 'tz', None) is None:
        dt = dt.dt.tz_localize(IST)
    else:
        dt = dt.dt.tz_convert(IST)
    df['Datetime'] = dt

    if 'Adj Close' in df.columns:
        df = df.drop(columns=['Adj Close'])
    if 'Volume' not in df.columns:
        df['Volume'] = 0.0

    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    df = df.drop_duplicates(subset='Datetime', keep='last')
    df = df.sort_values('Datetime').reset_index(drop=True)
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
    df['Volume'] = df['Volume'].fillna(0.0)
    return df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]


def _yf_download(ticker, interval, period=None, start=None, end=None):
    """Single yfinance request (raw, unadjusted prices like TradingView)"""
    kwargs = dict(interval=interval, progress=False, auto_adjust=False, threads=False)
    if period is not None:
        raw = yf.download(ticker, period=period, **kwargs)
    else:
        raw = yf.download(ticker, start=start, end=end, **kwargs)
    return _normalize_yf_df(raw)


def _plan_fetch_days(interval, period, is_live_trading):
    """
    Decide how many calendar days to download.
    Returns (requested_days, download_days); None means 'max history'.
    Backtest: requested period + warm-up.  Live: at least the warm-up.
    Clamped to Yahoo's intraday limits.
    """
    req_days = PERIOD_DAYS.get(period, None) if period != 'max' else None
    if period not in PERIOD_DAYS:
        req_days = 30
    extra = WARMUP_EXTRA_DAYS.get(interval, 0)
    if req_days is None:
        days = None
    elif is_live_trading:
        days = max(req_days, extra)
    else:
        days = req_days + extra
    cap = INTERVAL_MAX_DAYS.get(interval)
    if cap is not None and (days is None or days > cap):
        days = cap
    return req_days, days


def fetch_data(ticker_symbol, interval, period, is_live_trading=False, custom_ticker=None, quiet=False):
    """
    Fetch historical/live data using yfinance

    - Downloads extra warm-up history so every indicator has a TradingView-
      identical value from the first candle of the requested period.
    - Respects yfinance limits (1m: 30 days in 7-day chunks, 5m-30m: 60 days,
      1h: 730 days) and tells the user when a period was clamped.
    - Adds 'In_Period' column: True for candles inside the requested period
      (the rest is warm-up history, used only for indicator calculation).

    Returns:
        DataFrame with OHLCV data in IST timezone (or None)
    """
    try:
        ticker = _resolve_yf_ticker(ticker_symbol, custom_ticker)
        req_days, days = _plan_fetch_days(interval, period, is_live_trading)
        now = datetime.now(IST)

        # Inform user about clamped periods
        cap = INTERVAL_MAX_DAYS.get(interval)
        note = None
        if cap is not None and (req_days is None or req_days > cap):
            note = (f"ℹ️ yfinance allows only ~{cap + 1} days of {interval} data. "
                    f"Period reduced to the last {cap} days.")
        try:
            st.session_state['data_fetch_note'] = note
        except Exception:
            pass

        end = now + (timedelta(minutes=30) if interval in INTRADAY_INTERVALS else timedelta(days=1))

        if days is None:
            df = _yf_download(ticker, interval, period='max')
        elif days == req_days and period in YF_STANDARD_PERIODS and not (interval == '1m' and days > 7):
            df = _yf_download(ticker, interval, period=period)
        elif interval == '1m' and days > 6:
            # 1-minute data: Yahoo allows max 8 days per request → chunk it
            frames = []
            s = now - timedelta(days=days)
            while s < now:
                e = min(s + timedelta(days=6), end)
                part = _yf_download(ticker, interval, start=s, end=e)
                if part is not None and not part.empty:
                    frames.append(part)
                s = e
            if frames:
                df = pd.concat(frames, ignore_index=True)
                df = df.drop_duplicates(subset='Datetime', keep='last').sort_values('Datetime').reset_index(drop=True)
            else:
                df = None
        else:
            df = _yf_download(ticker, interval, start=now - timedelta(days=days), end=end)

        if df is None or df.empty:
            if not quiet:
                st.error(f"❌ No data returned for {ticker_symbol} ({ticker})")
            return None

        # Mark the requested period (everything before = warm-up only)
        if req_days is None:
            df['In_Period'] = True
        elif req_days <= 1 and interval in INTRADAY_INTERVALS:
            last_date = df['Datetime'].iloc[-1].date()
            df['In_Period'] = df['Datetime'].dt.date >= last_date
        else:
            df['In_Period'] = df['Datetime'] >= (now - timedelta(days=req_days))
        if not df['In_Period'].any():
            df['In_Period'] = True

        return df

    except Exception as e:
        if not quiet:
            st.error(f"❌ Error fetching data: {e}")
        return None


def get_latest_price(ticker_symbol, custom_ticker=None):
    """Light-weight latest price fetch (used by manual square-off)"""
    try:
        ticker = _resolve_yf_ticker(ticker_symbol, custom_ticker)
        df = _yf_download(ticker, '1m', period='1d')
        if df is None or df.empty:
            df = _yf_download(ticker, '1d', period='5d')
        if df is not None and not df.empty:
            return float(df['Close'].iloc[-1])
    except Exception:
        pass
    return None


def has_volume(df):
    """True if the data has real volume (indices on yfinance have Volume = 0)"""
    return 'Volume' in df.columns and float(df['Volume'].fillna(0).abs().sum()) > 0

# ================================
# TRADINGVIEW-EXACT INDICATOR PRIMITIVES
# ================================

def _seeded_recursive(values, length, alpha):
    """
    Pine Script style recursive average:
        first value = SMA(length) of the first `length` valid values,
        then  out = alpha * src + (1 - alpha) * out[1]
    Identical to ta.ema (alpha = 2/(len+1)) and ta.rma (alpha = 1/len).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    length = max(1, int(length))
    valid = np.where(~np.isnan(values))[0]
    if len(valid) == 0:
        return out
    f = valid[0]
    s = f + length - 1
    if s >= n:
        return out
    prev = float(np.nanmean(values[f:s + 1]))
    out[s] = prev
    for i in range(s + 1, n):
        v = values[i]
        if not np.isnan(v):
            prev = alpha * v + (1.0 - alpha) * prev
        out[i] = prev
    return out


def tv_ema(series, length):
    """ta.ema - TradingView EMA (SMA seeded)"""
    series = pd.Series(series, dtype=float)
    length = max(1, int(length))
    return pd.Series(_seeded_recursive(series.values, length, 2.0 / (length + 1)), index=series.index)


def tv_rma(series, length):
    """ta.rma - Wilder's moving average (used by RSI, ATR, ADX)"""
    series = pd.Series(series, dtype=float)
    length = max(1, int(length))
    return pd.Series(_seeded_recursive(series.values, length, 1.0 / length), index=series.index)


def tv_sma(series, length):
    """ta.sma"""
    return pd.Series(series, dtype=float).rolling(window=max(1, int(length))).mean()


def tv_stdev(series, length):
    """ta.stdev (population standard deviation, like TradingView)"""
    return pd.Series(series, dtype=float).rolling(window=max(1, int(length))).std(ddof=0)


def tv_true_range(df, handle_na=True):
    """ta.tr(handle_na)"""
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    if len(tr) > 0:
        tr.iloc[0] = (high.iloc[0] - low.iloc[0]) if handle_na else np.nan
    return tr

# ================================
# INDICATOR CALCULATIONS
# ================================

def calculate_ema_angle(ema_series, lookback=3):
    """
    Calculate EMA angle in degrees (slope of linear regression over lookback+1
    points, same result as the previous np.polyfit loop but vectorized)
    """
    ema_series = pd.Series(ema_series, dtype=float)
    n = lookback + 1
    y = ema_series.values
    out = np.full(len(y), np.nan)
    if len(y) >= n:
        x = np.arange(n, dtype=float)
        xm = x.mean()
        w = (x - xm) / ((x - xm) ** 2).sum()
        slopes = np.convolve(y, w[::-1], mode='valid')
        out[n - 1:] = np.degrees(np.arctan(slopes))
    return pd.Series(out, index=ema_series.index)


def calculate_rsi(series, period=14):
    """ta.rsi - TradingView RSI (Wilder RMA)"""
    series = pd.Series(series, dtype=float)
    change = series.diff()
    up = tv_rma(change.clip(lower=0), period)
    down = tv_rma((-change).clip(lower=0), period)
    rsi = 100 - (100 / (1 + up / down))
    rsi = rsi.where(down != 0, 100.0)
    rsi = rsi.where(up != 0, 0.0)
    rsi[up.isna() | down.isna()] = np.nan
    return rsi


def calculate_dmi(df, period=14, adx_smoothing=None):
    """ta.dmi - TradingView +DI, -DI, ADX"""
    adx_smoothing = adx_smoothing or period
    up = df['High'].diff()
    down = -df['Low'].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    plus_dm[up.isna()] = np.nan
    minus_dm[down.isna()] = np.nan
    trur = tv_rma(tv_true_range(df, handle_na=False), period)
    plus_di = 100 * tv_rma(plus_dm, period) / trur
    minus_di = 100 * tv_rma(minus_dm, period) / trur
    di_sum = plus_di + minus_di
    dx = (plus_di - minus_di).abs() / di_sum.where(di_sum != 0, 1.0)
    adx = 100 * tv_rma(dx, adx_smoothing)
    return plus_di, minus_di, adx


def calculate_adx(df, period=14):
    """Calculate ADX indicator (TradingView formula)"""
    return calculate_dmi(df, period)[2]


def calculate_atr(df, period=14):
    """ta.atr - RMA of True Range"""
    return tv_rma(tv_true_range(df, handle_na=True), period)


def compute_supertrend(df, period=10, multiplier=3.0):
    """
    ta.supertrend(factor, atrPeriod) - exact Pine logic.
    Returns (supertrend_line, direction) with direction 1 = uptrend, -1 = downtrend.
    """
    n = len(df)
    hl2 = ((df['High'] + df['Low']) / 2).values
    atr = calculate_atr(df, period).values
    close = df['Close'].values
    final_up = np.full(n, np.nan)
    final_lo = np.full(n, np.nan)
    st_line = np.full(n, np.nan)
    pine_dir = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(atr[i]):
            continue
        ub = hl2[i] + multiplier * atr[i]
        lb = hl2[i] - multiplier * atr[i]
        prev_lo = final_lo[i - 1] if i > 0 and not np.isnan(final_lo[i - 1]) else 0.0
        prev_up = final_up[i - 1] if i > 0 and not np.isnan(final_up[i - 1]) else 0.0
        pc = close[i - 1] if i > 0 else np.nan
        lb = lb if (lb > prev_lo or (not np.isnan(pc) and pc < prev_lo)) else prev_lo
        ub = ub if (ub < prev_up or (not np.isnan(pc) and pc > prev_up)) else prev_up
        final_lo[i], final_up[i] = lb, ub
        if i == 0 or np.isnan(atr[i - 1]):
            d = 1
        elif st_line[i - 1] == prev_up:
            d = -1 if close[i] > ub else 1
        else:
            d = 1 if close[i] < lb else -1
        st_line[i] = lb if d == -1 else ub
        pine_dir[i] = d
    return pd.Series(st_line, index=df.index), pd.Series(-pine_dir, index=df.index)


def compute_pivots(df, left=3, right=3):
    """
    ta.pivothigh / ta.pivotlow equivalent.
    Returns arrays with the pivot value at the pivot bar (NaN elsewhere).
    NOTE: a pivot at bar i is only KNOWN at bar i + right (confirmation).
    """
    left, right = max(1, int(left)), max(1, int(right))
    h = df['High'].astype(float)
    l = df['Low'].astype(float)
    left_max = h.shift(1).rolling(left).max()
    right_max = h[::-1].shift(1).rolling(right).max()[::-1]
    left_min = l.shift(1).rolling(left).min()
    right_min = l[::-1].shift(1).rolling(right).min()[::-1]
    ph = np.where((h > left_max) & (h >= right_max), h, np.nan)
    pl = np.where((l < left_min) & (l <= right_min), l, np.nan)
    return ph, pl


def add_swing_columns(df, config):
    """
    Causal swing levels (no look-ahead):
      Swing_High / Swing_Low           = most recent CONFIRMED pivot
      Prev_Swing_High / Prev_Swing_Low = the confirmed pivot before that
      PH_Raw / PL_Raw                  = pivot value at pivot bar (for searches)
    """
    left = int(config.get('swing_left', 3))
    right = int(config.get('swing_right', 3))
    ph, pl = compute_pivots(df, left, right)
    df['PH_Raw'] = ph
    df['PL_Raw'] = pl
    pos = np.arange(len(df), dtype=float)

    for raw, name in ((ph, 'High'), (pl, 'Low')):
        conf = pd.Series(raw, index=df.index).shift(right)          # known at i+right
        conf_pos = pd.Series(np.where(~np.isnan(raw), pos, np.nan), index=df.index).shift(right)
        df[f'Swing_{name}'] = conf.ffill()
        df[f'Swing_{name}_Pos'] = conf_pos.ffill()
        nn = conf.dropna()
        prev = nn.shift(1).reindex(df.index)
        # carry the 'previous' value forward until a new pivot is confirmed
        prev_full = pd.Series(np.nan, index=df.index)
        prev_full.loc[nn.index] = prev.loc[nn.index]
        df[f'Prev_Swing_{name}'] = prev_full.ffill()
    return df


def compute_market_structure(df, config):
    """
    Causal Smart-Money-Concepts structure tracker.
    Columns:
      MS_Trend  : 1 bullish / -1 bearish / 0 unknown
      MS_Event  : 'BOS_UP', 'CHOCH_UP', 'BOS_DN', 'CHOCH_DN' or ''
      MS_BullOB_Top/Bot, MS_BearOB_Top/Bot : active (unmitigated-invalidated) order block
    """
    right = int(config.get('swing_right', 3))
    ph = df['PH_Raw'].values
    pl = df['PL_Raw'].values
    o, h, l, c = (df[x].values for x in ('Open', 'High', 'Low', 'Close'))
    n = len(df)
    trend = np.zeros(n)
    events = np.array([''] * n, dtype=object)
    bull_top = np.full(n, np.nan); bull_bot = np.full(n, np.nan)
    bear_top = np.full(n, np.nan); bear_bot = np.full(n, np.nan)

    last_sh, sh_broken = np.nan, True
    last_sl, sl_broken = np.nan, True
    cur = 0
    bull_ob = None
    bear_ob = None
    for t in range(n):
        j = t - right
        if j >= 0:
            if not np.isnan(ph[j]):
                last_sh, sh_broken = ph[j], False
            if not np.isnan(pl[j]):
                last_sl, sl_broken = pl[j], False
        if not sh_broken and not np.isnan(last_sh) and c[t] > last_sh:
            events[t] = 'CHOCH_UP' if cur == -1 else 'BOS_UP'
            cur, sh_broken = 1, True
            for k in range(t - 1, max(-1, t - 31), -1):   # last bearish candle before the break
                if c[k] < o[k]:
                    bull_ob = (h[k], l[k])
                    break
        elif not sl_broken and not np.isnan(last_sl) and c[t] < last_sl:
            events[t] = 'CHOCH_DN' if cur == 1 else 'BOS_DN'
            cur, sl_broken = -1, True
            for k in range(t - 1, max(-1, t - 31), -1):   # last bullish candle before the break
                if c[k] > o[k]:
                    bear_ob = (h[k], l[k])
                    break
        if bull_ob is not None and c[t] < bull_ob[1]:
            bull_ob = None
        if bear_ob is not None and c[t] > bear_ob[0]:
            bear_ob = None
        trend[t] = cur
        if bull_ob is not None:
            bull_top[t], bull_bot[t] = bull_ob
        if bear_ob is not None:
            bear_top[t], bear_bot[t] = bear_ob

    df['MS_Trend'] = trend
    df['MS_Event'] = events
    df['MS_BullOB_Top'] = bull_top
    df['MS_BullOB_Bot'] = bull_bot
    df['MS_BearOB_Top'] = bear_top
    df['MS_BearOB_Bot'] = bear_bot
    return df


def add_session_columns(df):
    """Session date, session open and previous-day OHLC (for gaps, pivots, VWAP)"""
    d = df['Datetime'].dt.date
    df['Session_Date'] = d
    g = df.groupby(d)
    df['Session_Open'] = g['Open'].transform('first')
    daily = g.agg(H=('High', 'max'), L=('Low', 'min'), C=('Close', 'last'))
    prev = daily.shift(1)
    df['Prev_Day_High'] = d.map(prev['H']).astype(float)
    df['Prev_Day_Low'] = d.map(prev['L']).astype(float)
    df['Prev_Session_Close'] = d.map(prev['C']).astype(float)
    return df


def add_session_vwap(df):
    """ta.vwap (session anchored, hlc3). NaN when the ticker has no volume."""
    if has_volume(df):
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        pv = (tp * df['Volume']).groupby(df['Session_Date']).cumsum()
        vv = df['Volume'].groupby(df['Session_Date']).cumsum()
        df['VWAP'] = pv / vv.replace(0, np.nan)
    else:
        df['VWAP'] = np.nan
    return df


def ensure_daily_pivots(df):
    """Classic floor pivots from previous session's H/L/C"""
    if 'PIV_P' in df.columns:
        return df
    if 'Prev_Day_High' not in df.columns:
        add_session_columns(df)
    H, L, C = df['Prev_Day_High'], df['Prev_Day_Low'], df['Prev_Session_Close']
    P = (H + L + C) / 3
    df['PIV_P'] = P
    df['PIV_R1'] = 2 * P - L
    df['PIV_S1'] = 2 * P - H
    df['PIV_R2'] = P + (H - L)
    df['PIV_S2'] = P - (H - L)
    df['PIV_R3'] = H + 2 * (P - L)
    df['PIV_S3'] = L - 2 * (H - P)
    return df


def ensure_orb_cols(df, minutes=15, session_start='09:15'):
    """
    Opening range high/low for every session (no look-ahead: use only when *_READY).
    Returns the column prefix.
    """
    key = f"_ORB_{int(minutes)}_{str(session_start).replace(':', '')}"
    if key + '_H' in df.columns:
        return key
    if 'Session_Date' not in df.columns:
        add_session_columns(df)
    try:
        hh, mm = [int(x) for x in str(session_start).split(':')[:2]]
    except Exception:
        hh, mm = 9, 15
    dt = df['Datetime']
    sess_start = dt.dt.normalize() + pd.Timedelta(hours=hh, minutes=mm)
    mins = (dt - sess_start).dt.total_seconds() / 60.0
    in_rng = (mins >= 0) & (mins < minutes)
    d = df['Session_Date']
    df[key + '_H'] = df['High'].where(in_rng).groupby(d).transform('max')
    df[key + '_L'] = df['Low'].where(in_rng).groupby(d).transform('min')
    df[key + '_READY'] = (mins >= minutes) & df[key + '_H'].notna()
    return key


def get_confirmed_pivots(df, idx, kind, config, max_scan=1500):
    """Most-recent-first list of pivot values (kind 'high'/'low') confirmed by bar idx"""
    right = int(config.get('swing_right', 3))
    col = 'PH_Raw' if kind == 'high' else 'PL_Raw'
    if col not in df.columns:
        add_swing_columns(df, config)
    last = idx - right
    if last < 0:
        return []
    start = max(0, last - max_scan)
    vals = df[col].values[start:last + 1][::-1]
    return [float(v) for v in vals if not np.isnan(v)]


def pick_swing(df, idx, kind, which, ref_price, side, config):
    """
    Pick swing level relative to ref_price.
      kind  : 'high' or 'low'
      which : 'current' (most recent valid) or 'previous' (the one before)
      side  : 'below' or 'above' ref_price (the level must be on this side)
    """
    piv = get_confirmed_pivots(df, idx, kind, config)
    if side == 'below':
        cands = [v for v in piv if v < ref_price]
    else:
        cands = [v for v in piv if v > ref_price]
    k = 0 if which == 'current' else 1
    return cands[k] if len(cands) > k else None


def calculate_all_indicators(df, config):
    """
    Calculate all technical indicators (TradingView-identical formulas)
    """
    df = df.copy()
    ema_fast = int(config.get('ema_fast', 9))
    ema_slow = int(config.get('ema_slow', 20))

    # EMA (ta.ema – SMA seeded, exactly like TradingView)
    df['EMA_Fast'] = tv_ema(df['Close'], ema_fast)
    df['EMA_Slow'] = tv_ema(df['Close'], ema_slow)

    # SMA for custom strategy
    df['SMA_20'] = tv_sma(df['Close'], 20)
    df['SMA_50'] = tv_sma(df['Close'], 50)

    # EMA Angle
    df['EMA_Fast_Angle'] = calculate_ema_angle(df['EMA_Fast'])
    df['EMA_Slow_Angle'] = calculate_ema_angle(df['EMA_Slow'])

    # RSI (Wilder)
    df['RSI'] = calculate_rsi(df['Close'], 14)

    # ADX / DMI
    adx_period = int(config.get('adx_period', 14))
    plus_di, minus_di, adx = calculate_dmi(df, adx_period)
    df['ADX'] = adx
    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di

    # ATR
    df['ATR'] = calculate_atr(df, 14)

    # Bollinger Bands (population std like TradingView)
    bb_period = int(config.get('custom_bb_period', 20))
    bb_std = float(config.get('custom_bb_std', 2.0))
    df['BB_Middle'] = tv_sma(df['Close'], bb_period)
    bb_std_dev = tv_stdev(df['Close'], bb_period)
    df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)

    # MACD (ta.macd)
    df['MACD'] = tv_ema(df['Close'], 12) - tv_ema(df['Close'], 26)
    df['MACD_Signal'] = tv_ema(df['MACD'], 9)
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Volume MA
    if 'Volume' in df.columns:
        df['Volume_MA'] = tv_sma(df['Volume'], 20)

    # Sessions, VWAP, SuperTrend
    add_session_columns(df)
    add_session_vwap(df)
    st_line, st_dir = compute_supertrend(df, int(config.get('supertrend_atr_period', 10)),
                                         float(config.get('supertrend_multiplier', 3.0)))
    df['SuperTrend'] = st_line
    df['SuperTrend_Direction'] = st_dir

    # Swing highs / lows (confirmed pivots, no look-ahead) + market structure
    add_swing_columns(df, config)
    compute_market_structure(df, config)

    # Previous values for reference
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    df['Prev_Close'] = df['Close'].shift(1)

    if 'In_Period' not in df.columns:
        df['In_Period'] = True

    return df

# ================================
# STRATEGY FUNCTIONS
# ================================
# Signal types:
#   EVENT  -> fires only on the candle where the condition becomes true
#             (crossover, breakout, flip).  e.g. EMA Crossover
#   STATE  -> fires on every candle where the condition is true; in live
#             trading each signal candle can still trigger only ONE entry.

def _isnan(x):
    try:
        return x is None or bool(pd.isna(x))
    except Exception:
        return True


def check_ema_crossover_strategy(df, idx, config, current_position):
    """
    EMA Crossover Strategy with advanced filters  [EVENT]

    BUY  only when: previous Fast <= previous Slow AND current Fast > current Slow
    SELL only when: previous Fast >= previous Slow AND current Fast < current Slow

    Filters:
    - Minimum Angle (ABSOLUTE value)
    - Entry Filters (Simple/Custom Candle Points/ATR-based)
    - ADX Filter (optional)
    """
    if idx < 1:
        return None, None
        
    current = df.iloc[idx]
    previous = df.iloc[idx - 1]
    
    # Check if indicators are ready (current AND previous)
    if (_isnan(current['EMA_Fast']) or _isnan(current['EMA_Slow']) or
            _isnan(previous['EMA_Fast']) or _isnan(previous['EMA_Slow'])):
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
    fast_angle = abs(current['EMA_Fast_Angle']) if not _isnan(current['EMA_Fast_Angle']) else 0
    
    if fast_angle < min_angle:
        return None, None
    
    # Apply Entry Filter
    entry_filter = config.get('ema_entry_filter', 'Simple Crossover')
    
    if entry_filter == 'Custom Candle (Points)':
        min_points = config.get('ema_custom_candle_points', 5)
        candle_body = abs(current['Close'] - current['Open'])
        if candle_body < min_points:
            return None, None
            
    elif entry_filter == 'ATR-based Candle':
        if _isnan(current['ATR']):
            return None, None
        atr_multiplier = config.get('ema_atr_multiplier', 0.3)
        min_body = current['ATR'] * atr_multiplier
        candle_body = abs(current['Close'] - current['Open'])
        if candle_body < min_body:
            return None, None
    
    # Apply ADX Filter (optional)
    if config.get('ema_use_adx', False):
        adx_threshold = config.get('ema_adx_threshold', 20)
        if _isnan(current['ADX']) or current['ADX'] < adx_threshold:
            return None, None
    
    if bullish_cross:
        return 'BUY', current['Close']
    elif bearish_cross:
        return 'SELL', current['Close']
    
    return None, None

def check_simple_buy_strategy(df, idx, config, current_position):
    """Simple Buy strategy - always returns BUY immediately if no position  [STATE]"""
    if current_position is not None:
        return None, None
    return 'BUY', df.iloc[idx]['Close']

def check_simple_sell_strategy(df, idx, config, current_position):
    """Simple Sell strategy - always returns SELL immediately if no position  [STATE]"""
    if current_position is not None:
        return None, None
    return 'SELL', df.iloc[idx]['Close']

def check_price_crosses_threshold(df, idx, config, current_position):
    """
    Price vs threshold  [STATE by design: checks current price state, no actual crossing needed]
    """
    if current_position is not None:
        return None, None
        
    threshold = config.get('price_threshold', 25000)
    current_price = df.iloc[idx]['Close']
    cross_type = config.get('price_cross_type', 'Above Threshold')
    position_type = config.get('price_cross_position', 'LONG')
    
    if cross_type == 'Above Threshold':
        condition_met = current_price > threshold
    else:
        condition_met = current_price < threshold
    
    if condition_met:
        return ('BUY' if position_type == 'LONG' else 'SELL'), current_price
    return None, None

def check_rsi_adx_ema_combined(df, idx, config, current_position):
    """Combined RSI-ADX-EMA strategy  [STATE]"""
    if current_position is not None:
        return None, None
    if idx < 1:
        return None, None
        
    current = df.iloc[idx]
    if _isnan(current['RSI']) or _isnan(current['ADX']) or _isnan(current['EMA_Fast']):
        return None, None
    
    rsi = current['RSI']
    adx = current['ADX']
    price = current['Close']
    ema = current['EMA_Fast']
    
    if rsi < 30 and adx > 25 and price > ema:
        return 'BUY', price
    if rsi > 70 and adx > 25 and price < ema:
        return 'SELL', price
    return None, None

def check_percentage_change(df, idx, config, current_position):
    """Percentage change strategy (candle-to-candle % change)  [EVENT per candle]"""
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
    
    if 'Positive' in change_type:
        condition_met = pct_change >= threshold
    else:
        condition_met = pct_change <= -threshold
    
    if condition_met:
        return ('BUY' if position_type == 'LONG' else 'SELL'), current_price
    return None, None

def check_ai_price_action(df, idx, config, current_position):
    """AI Price Action - three consecutive higher/lower closes  [STATE]"""
    if current_position is not None:
        return None, None
    if idx < 3:
        return None, None
        
    candles = df.iloc[idx-2:idx+1]
    if all(candles['Close'].diff().dropna() > 0):
        return 'BUY', df.iloc[idx]['Close']
    if all(candles['Close'].diff().dropna() < 0):
        return 'SELL', df.iloc[idx]['Close']
    return None, None

def check_custom_strategy(df, idx, config, current_position):
    """
    Custom Strategy Builder — multi-indicator, AND/OR combine logic.
    Reads config['custom_conditions'] list (one dict per condition).
    Falls back to legacy single-condition keys if list not present.
    Crosses/crossovers are EVENTS; pullback is a STATE.
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
                df[key] = tv_ema(df["Close"], int(period))
            elif name == "SMA":
                df[key] = tv_sma(df["Close"], int(period))
            elif name == "BB_U":
                df[key] = tv_sma(df["Close"], int(period)) + bb_std * tv_stdev(df["Close"], int(period))
            elif name == "BB_L":
                df[key] = tv_sma(df["Close"], int(period)) - bb_std * tv_stdev(df["Close"], int(period))
            elif name == "BB_M":
                df[key] = tv_sma(df["Close"], int(period))
            elif name == "ATR":
                df[key] = calculate_atr(df, int(period))
            elif name == "HV":
                df[key] = df["Close"].pct_change().rolling(window=int(period)).std(ddof=0) * (252 ** 0.5) * 100
            elif name == "STDDEV":
                df[key] = tv_stdev(df["Close"], int(period))
            elif name == "RSI":
                df[key] = calculate_rsi(df["Close"], int(period))
            elif name == "VOL_MA":
                df[key] = tv_sma(df["Volume"], int(period)) if "Volume" in df.columns else float("nan")
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
            iv, piv = df[col].iloc[idx], df[col].iloc[idx - 1]
            if pd.isna(iv) or pd.isna(piv): return False, ptype
            # price crosses the indicator (indicator value of each bar)
            if cross == "Above Indicator":
                triggered = prev_price <= float(piv) and current_price > float(iv)
            else:
                triggered = prev_price >= float(piv) and current_price < float(iv)
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
            iv = df[col].iloc[idx]
            if pd.isna(iv): return False, ptype
            iv = float(iv)
            side = c.get("pullback_side", "Approach from Above")
            triggered = abs(current_price - iv) <= float(c.get("pullback_points", 10)) and (
                current_price >= iv if side == "Approach from Above" else current_price <= iv)
            return triggered, ptype

        elif stype == "Indicator Crosses Level":
            level = float(c.get("level", 50.0))
            chk = _ca if "Above" in cross else _cb

            def _pair(colname):
                cv, pv = df[colname].iloc[idx], df[colname].iloc[idx - 1]
                return cv, pv

            if ind == "RSI":
                cv, pv = _pair(_col("RSI", int(c.get("rsi_period", 14))))
            elif ind == "MACD":
                cv, pv = _pair("MACD")
            elif ind == "MACD Histogram":
                cv, pv = _pair("MACD_Hist")
            elif ind == "ADX":
                cv, pv = _pair("ADX")
            elif ind == "Volume":
                if not has_volume(df): return False, ptype
                vcol = _col("VOL_MA", int(c.get("volume_ma_period", 20)))
                cv  = float(current.get("Volume", 0))
                pv  = float(previous.get("Volume", 0))
                vma = float(df[vcol].iloc[idx] or 1)
                thresh = vma * float(c.get("volume_multiplier", 1.5))
                return chk(cv, pv, thresh), ptype
            elif ind == "BB %B":
                bp   = int(c.get("bb_period", 20))
                bstd = float(c.get("bb_std", 2.0))
                pctb_key = f"_CUST_PCTB_{bp}_{bstd}"
                if pctb_key not in df.columns:
                    mid = tv_sma(df["Close"], bp)
                    std = tv_stdev(df["Close"], bp)
                    df[pctb_key] = (df["Close"] - (mid - bstd * std)) / (2 * bstd * std)
                cv, pv = df[pctb_key].iloc[idx], df[pctb_key].iloc[idx-1]
                if pd.isna(cv) or pd.isna(pv): return False, ptype
                return chk(cv, pv, level / 100), ptype
            elif ind == "ATR (Volatility)":
                cv, pv = _pair(_col("ATR", int(c.get("atr_period", 14))))
            elif ind == "Historical Volatility":
                cv, pv = _pair(_col("HV", int(c.get("hv_period", 20))))
            elif ind == "Std Dev (Volatility)":
                cv, pv = _pair(_col("STDDEV", int(c.get("stddev_period", 20))))
            else:
                return False, ptype
            if pd.isna(cv) or pd.isna(pv): return False, ptype
            return chk(cv, pv, level), ptype

        elif stype == "Indicator Crossover":
            co      = c.get("crossover_type", "Fast EMA x Slow EMA")
            is_bull = "Bullish" in cross

            def bull(fc, fp, sc, sp): return float(fp) <= float(sp) and float(fc) > float(sc)
            def bear(fc, fp, sc, sp): return float(fp) >= float(sp) and float(fc) < float(sc)
            chk2 = bull if is_bull else bear

            def _vals(fc_col, sc_col):
                return [df[fc_col].iloc[idx], df[fc_col].iloc[idx-1], df[sc_col].iloc[idx], df[sc_col].iloc[idx-1]]

            if co in ("Fast EMA x Slow EMA", "Fast EMA × Slow EMA"):
                vals = _vals(_col("EMA", int(c.get("fast_ema", 9))), _col("EMA", int(c.get("slow_ema", 21))))
                if any(pd.isna(v) for v in vals): return False, ptype
                return chk2(*vals), ptype

            elif co in ("Fast SMA x Slow SMA", "Fast SMA × Slow SMA"):
                vals = _vals(_col("SMA", int(c.get("fast_sma", 20))), _col("SMA", int(c.get("slow_sma", 50))))
                if any(pd.isna(v) for v in vals): return False, ptype
                return chk2(*vals), ptype

            elif co in ("MACD x Signal", "MACD × Signal"):
                vals = _vals("MACD", "MACD_Signal")
                if any(pd.isna(v) for v in vals): return False, ptype
                return chk2(*vals), ptype

            elif co in ("Price x EMA", "Price × EMA", "Price x SMA", "Price × SMA"):
                col = _col("EMA" if "EMA" in co else "SMA", int(c.get("ma_period", 50)))
                ic, ip = df[col].iloc[idx], df[col].iloc[idx-1]
                if pd.isna(ic) or pd.isna(ip): return False, ptype
                return chk2(current_price, prev_price, float(ic), float(ip)), ptype

            elif co == "RSI Crossover (Overbought/Oversold)":
                rsi_col = _col("RSI", int(c.get("rsi_period", 14)))
                rc, rp  = df[rsi_col].iloc[idx], df[rsi_col].iloc[idx-1]
                if pd.isna(rc) or pd.isna(rp): return False, ptype
                ob, os_lvl = float(c.get("rsi_ob", 70)), float(c.get("rsi_os", 30))
                if is_bull:
                    # RSI crosses back ABOVE oversold
                    return (float(rp) <= os_lvl and float(rc) > os_lvl), ptype
                else:
                    # FIX: RSI crosses back BELOW overbought (was "crosses above OB")
                    return (float(rp) >= ob and float(rc) < ob), ptype

        return False, ptype

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
    SuperTrend AI Strategy  [EVENT: SuperTrend direction flip]
    SuperTrend = exact TradingView ta.supertrend, + ADX + volume confirmation
    """
    if current_position is not None:
        return None, None
    if idx < 20:
        return None, None
    
    atr_period = int(config.get('supertrend_atr_period', 10))
    multiplier = float(config.get('supertrend_multiplier', 3.0))
    adx_threshold = config.get('supertrend_adx_threshold', 25)
    volume_mult = config.get('supertrend_volume_mult', 1.5)
    
    dir_col = f'_ST_DIR_{atr_period}_{multiplier}'
    if dir_col not in df.columns:
        _, d = compute_supertrend(df, atr_period, multiplier)
        df[dir_col] = d
    
    current = df.iloc[idx]
    curr_dir = df[dir_col].iloc[idx]
    prev_dir = df[dir_col].iloc[idx - 1]
    if _isnan(curr_dir) or _isnan(prev_dir):
        return None, None
    
    # ADX filter
    adx = current.get('ADX', np.nan)
    if _isnan(adx) or adx < adx_threshold:
        return None, None
    
    # Volume filter (skipped automatically for indices which have no volume)
    if has_volume(df):
        vol_ma = df['Volume'].rolling(20).mean().iloc[idx]
        if current['Volume'] < vol_ma * volume_mult:
            return None, None
    
    if prev_dir == -1 and curr_dir == 1:
        return 'BUY', current['Close']
    if prev_dir == 1 and curr_dir == -1:
        return 'SELL', current['Close']
    return None, None

def check_vwap_volume_spike(df, idx, config, current_position):
    """
    VWAP + Volume Spike Strategy  [EVENT: price crosses session VWAP]
    NOTE: needs real volume (does not work on indices like ^NSEI where Volume = 0)
    """
    if current_position is not None:
        return None, None
    if idx < 50:
        return None, None
    if not has_volume(df):
        return None, None
    
    volume_mult = config.get('vwap_volume_mult', 2.0)
    vwap_distance = config.get('vwap_distance_pct', 0.3)
    
    if 'VWAP' not in df.columns:
        if 'Session_Date' not in df.columns:
            add_session_columns(df)
        add_session_vwap(df)
    
    current = df.iloc[idx]
    previous = df.iloc[idx - 1]
    current_price = current['Close']
    prev_price = previous['Close']
    vwap = current['VWAP']
    if _isnan(vwap) or _isnan(previous['VWAP']):
        return None, None
    
    vol_ma = df['Volume'].rolling(20).mean().iloc[idx]
    if current['Volume'] < vol_ma * volume_mult:
        return None, None
    
    distance_pct = abs(current_price - vwap) / vwap * 100
    if distance_pct > vwap_distance:
        return None, None
    
    rsi = current.get('RSI', 50)
    
    if prev_price < previous['VWAP'] and current_price > vwap:
        if not _isnan(rsi) and rsi < 55:
            return 'BUY', current_price
    if prev_price > previous['VWAP'] and current_price < vwap:
        if not _isnan(rsi) and rsi > 45:
            return 'SELL', current_price
    return None, None

def check_bollinger_squeeze_breakout(df, idx, config, current_position):
    """
    Bollinger Band Squeeze Breakout  [EVENT: close crosses outside band after squeeze]
    """
    if current_position is not None:
        return None, None
    if idx < 30:
        return None, None
    
    bb_period = int(config.get('bb_squeeze_period', 20))
    bb_std = config.get('bb_squeeze_std', 2.0)
    squeeze_threshold = config.get('bb_squeeze_threshold', 0.02)
    volume_mult = config.get('bb_squeeze_volume_mult', 1.8)
    
    if f'BB_Upper_{bb_period}' not in df.columns:
        bb_mid = tv_sma(df['Close'], bb_period)
        bb_std_val = tv_stdev(df['Close'], bb_period)
        df[f'BB_Upper_{bb_period}'] = bb_mid + (bb_std * bb_std_val)
        df[f'BB_Lower_{bb_period}'] = bb_mid - (bb_std * bb_std_val)
        df[f'BB_Mid_{bb_period}'] = bb_mid
        df[f'BB_Bandwidth_{bb_period}'] = (df[f'BB_Upper_{bb_period}'] - df[f'BB_Lower_{bb_period}']) / df[f'BB_Mid_{bb_period}']
    
    current = df.iloc[idx]
    previous = df.iloc[idx - 1]
    current_price = current['Close']
    prev_price = previous['Close']
    
    bb_upper = current[f'BB_Upper_{bb_period}']
    bb_lower = current[f'BB_Lower_{bb_period}']
    bandwidth = current[f'BB_Bandwidth_{bb_period}']
    prev_bandwidth = previous[f'BB_Bandwidth_{bb_period}']
    if _isnan(bandwidth) or _isnan(prev_bandwidth):
        return None, None
    
    is_squeezed = bandwidth < squeeze_threshold
    was_squeezed = prev_bandwidth < squeeze_threshold
    
    if has_volume(df):
        vol_ma = df['Volume'].rolling(20).mean().iloc[idx]
        volume_surge = current['Volume'] > vol_ma * volume_mult
    else:
        volume_surge = True   # no volume data (index) → volume condition skipped
    
    if (is_squeezed or was_squeezed) and volume_surge:
        if prev_price <= previous[f'BB_Upper_{bb_period}'] and current_price > bb_upper:
            rsi = current.get('RSI', 50)
            if _isnan(rsi) or rsi < 75:
                return 'BUY', current_price
        if prev_price >= previous[f'BB_Lower_{bb_period}'] and current_price < bb_lower:
            rsi = current.get('RSI', 50)
            if _isnan(rsi) or rsi > 25:
                return 'SELL', current_price
    return None, None

def check_elliott_waves_ratio_charts(df, idx, config, current_position):
    """
    Elliott Waves Strategy (Simplified with argrelextrema)  [EVENT]
    FIX: only extrema already confirmed at bar idx are used (no look-ahead),
         and a signal fires only on the bar where the 5th extreme gets confirmed.
    """
    if current_position is not None:
        return None, None
    
    wave_lookback = config.get('elliott_wave_lookback', 50)
    order = 5
    if idx < wave_lookback:
        return None, None
    
    if 'Wave_Extrema' not in df.columns:
        df['Wave_Extrema'] = 0
        if len(df) >= wave_lookback:
            highs_idx = argrelextrema(df['High'].values, np.greater, order=order)[0]
            lows_idx = argrelextrema(df['Low'].values, np.less, order=order)[0]
            col = df.columns.get_loc('Wave_Extrema')
            for h_idx in highs_idx:
                df.iloc[h_idx, col] = 1
            for l_idx in lows_idx:
                df.iloc[l_idx, col] = 1
    
    recent_start = max(0, idx - wave_lookback)
    recent = df.iloc[recent_start:idx+1]
    extrema_indices = [i for i in recent[recent['Wave_Extrema'] == 1].index.tolist() if i <= idx - order]
    
    bullish = False
    bearish = False
    if len(extrema_indices) >= 5 and extrema_indices[-1] == idx - order:
        wave_prices = df.loc[extrema_indices[-5:], 'Close'].values
        if (wave_prices[0] < wave_prices[1] > wave_prices[2] < wave_prices[3] > wave_prices[4]):
            bullish = wave_prices[4] < wave_prices[2]
            bearish = wave_prices[4] > wave_prices[2]
    
    if bullish:
        return 'BUY', df.iloc[idx]['Close']
    elif bearish:
        return 'SELL', df.iloc[idx]['Close']
    return None, None

def check_opening_range_breakout(df, idx, config, current_position):
    """
    Opening Range Breakout (ORB)  [EVENT: close crosses ORB high/low + buffer]
    FIX: opening range excludes the candle that starts at the range end time
    """
    if current_position is not None:
        return None, None
    if idx < 2 or 'Datetime' not in df.columns:
        return None, None
    
    orb_minutes = int(config.get('orb_minutes', 15))
    breakout_buffer = config.get('orb_breakout_buffer', 0.1)
    key = ensure_orb_cols(df, orb_minutes, config.get('orb_session_start', '09:15'))
    
    cur = df.iloc[idx]
    if not bool(cur[key + '_READY']):
        return None, None
    if df['Session_Date'].iloc[idx - 1] != df['Session_Date'].iloc[idx] and not bool(df[key + '_READY'].iloc[idx - 1]):
        pass
    orb_high = cur[key + '_H']
    orb_low = cur[key + '_L']
    if _isnan(orb_high) or _isnan(orb_low):
        return None, None
    
    breakout_high = orb_high * (1 + breakout_buffer / 100)
    breakout_low = orb_low * (1 - breakout_buffer / 100)
    current_price = cur['Close']
    previous_price = df.iloc[idx - 1]['Close']
    
    if previous_price <= breakout_high and current_price > breakout_high:
        return 'BUY', current_price
    if previous_price >= breakout_low and current_price < breakout_low:
        return 'SELL', current_price
    return None, None

def check_pivot_point_reversal(df, idx, config, current_position):
    """
    Pivot Point Reversal Strategy  [STATE near level + reversal candle]
    Uses pivot points (R1, R2, S1, S2) of the last `lookback` candles
    """
    if current_position is not None:
        return None, None
    lookback = int(config.get('pivot_lookback', 24))
    if idx < max(2, lookback):
        return None, None
    
    prev_data = df.iloc[idx - lookback:idx]
    pivot_high = prev_data['High'].max()
    pivot_low = prev_data['Low'].min()
    pivot_close = prev_data['Close'].iloc[-1]
    
    pivot = (pivot_high + pivot_low + pivot_close) / 3
    r1 = 2 * pivot - pivot_low
    r2 = pivot + (pivot_high - pivot_low)
    s1 = 2 * pivot - pivot_high
    s2 = pivot - (pivot_high - pivot_low)
    
    current_price = df.iloc[idx]['Close']
    previous_price = df.iloc[idx - 1]['Close']
    rsi = df.iloc[idx].get('RSI', 50)
    if _isnan(rsi):
        return None, None
    tolerance = current_price * 0.001
    
    if rsi < 40:
        if abs(current_price - s1) < tolerance or abs(current_price - s2) < tolerance:
            if current_price > previous_price:
                return 'BUY', current_price
    if rsi > 60:
        if abs(current_price - r1) < tolerance or abs(current_price - r2) < tolerance:
            if current_price < previous_price:
                return 'SELL', current_price
    return None, None

def check_ichimoku_cloud(df, idx, config, current_position):
    """
    Ichimoku Cloud Strategy  [EVENT: Tenkan/Kijun cross + price vs cloud]
    FIX: cloud displacement now matches TradingView (offset = 26 - 1)
    """
    if current_position is not None:
        return None, None
    if idx < 52:
        return None, None
    
    if 'Ichimoku_Tenkan' not in df.columns:
        disp = 26 - 1
        high_9 = df['High'].rolling(window=9).max()
        low_9 = df['Low'].rolling(window=9).min()
        df['Ichimoku_Tenkan'] = (high_9 + low_9) / 2
        high_26 = df['High'].rolling(window=26).max()
        low_26 = df['Low'].rolling(window=26).min()
        df['Ichimoku_Kijun'] = (high_26 + low_26) / 2
        df['Ichimoku_SpanA'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(disp)
        high_52 = df['High'].rolling(window=52).max()
        low_52 = df['Low'].rolling(window=52).min()
        df['Ichimoku_SpanB'] = ((high_52 + low_52) / 2).shift(disp)
        df['Ichimoku_Chikou'] = df['Close'].shift(-disp)   # display only, not used for signals
    
    current = df.iloc[idx]
    current_price = current['Close']
    tenkan = current.get('Ichimoku_Tenkan')
    kijun = current.get('Ichimoku_Kijun')
    span_a = current.get('Ichimoku_SpanA')
    span_b = current.get('Ichimoku_SpanB')
    if _isnan(tenkan) or _isnan(kijun) or _isnan(span_a) or _isnan(span_b):
        return None, None
    
    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)
    previous = df.iloc[idx - 1]
    prev_tenkan = previous.get('Ichimoku_Tenkan')
    prev_kijun = previous.get('Ichimoku_Kijun')
    
    if not _isnan(prev_tenkan) and not _isnan(prev_kijun):
        if prev_tenkan <= prev_kijun and tenkan > kijun and current_price > cloud_top:
            return 'BUY', current_price
        if prev_tenkan >= prev_kijun and tenkan < kijun and current_price < cloud_bottom:
            return 'SELL', current_price
    return None, None

def check_volume_breakout(df, idx, config, current_position):
    """
    Volume Breakout Strategy  [STATE per candle]
    NOTE: needs real volume (indices on yfinance have none)
    """
    if current_position is not None:
        return None, None
    if idx < 20 or not has_volume(df):
        return None, None
    
    volume_multiplier = config.get('volume_multiplier', 2.0)
    price_change_threshold = config.get('volume_price_threshold', 0.5)
    
    if 'Volume_MA' not in df.columns:
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    
    current = df.iloc[idx]
    current_price = current['Close']
    current_volume = current['Volume']
    volume_ma = current.get('Volume_MA', 0)
    if _isnan(volume_ma) or volume_ma == 0:
        return None, None
    if current_volume < volume_ma * volume_multiplier:
        return None, None
    
    open_price = current['Open']
    price_change_pct = abs(current_price - open_price) / open_price * 100
    if price_change_pct < price_change_threshold:
        return None, None
    
    rsi = current.get('RSI', 50)
    if _isnan(rsi):
        rsi = 50
    prev_close = df.iloc[idx - 1]['Close']
    if current_price > open_price and rsi < 70 and current_price > prev_close:
        return 'BUY', current_price
    if current_price < open_price and rsi > 30 and current_price < prev_close:
        return 'SELL', current_price
    return None, None

def check_gap_trading(df, idx, config, current_position):
    """
    Gap Trading Strategy  [EVENT: price crosses back through the session open]
    FIX: gap is now measured between today's session OPEN and the PREVIOUS
         SESSION CLOSE (earlier code compared with the previous 1-min candle).
         Volume confirmation is skipped when the ticker has no volume.
    """
    if current_position is not None:
        return None, None
    if idx < 2:
        return None, None
    if 'Session_Open' not in df.columns:
        add_session_columns(df)
    
    min_gap_percent = config.get('gap_min_percent', 0.5)
    max_gap_percent = config.get('gap_max_percent', 3.0)
    
    current = df.iloc[idx]
    sess_open = current['Session_Open']
    prev_close = current['Prev_Session_Close']
    if _isnan(sess_open) or _isnan(prev_close) or prev_close == 0:
        return None, None
    current_price = current['Close']
    
    gap_percent = abs(sess_open - prev_close) / prev_close * 100
    if gap_percent < min_gap_percent or gap_percent > max_gap_percent:
        return None, None
    
    # Only trade gaps in the first hour (intraday data)
    current_time = current['Datetime']
    if hasattr(current_time, 'time'):
        market_open = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
        if current_time > market_open + timedelta(hours=1):
            return None, None
    
    vol_ok = True
    if has_volume(df) and idx >= 20:
        vol_ok = current['Volume'] > df['Volume'].iloc[idx - 20:idx].mean()
    if not vol_ok:
        return None, None
    
    first_bar = df['Session_Date'].iloc[idx - 1] != df['Session_Date'].iloc[idx]
    prev_c = df.iloc[idx - 1]['Close']
    
    if sess_open > prev_close:   # Gap up → fade (SHORT) when price falls back below open
        if (first_bar and current_price < sess_open) or (not first_bar and prev_c >= sess_open and current_price < sess_open):
            return 'SELL', current_price
    elif sess_open < prev_close:  # Gap down → LONG when price climbs back above open
        if (first_bar and current_price > sess_open) or (not first_bar and prev_c <= sess_open and current_price > sess_open):
            return 'BUY', current_price
    return None, None

def check_mean_reversion_bollinger(df, idx, config, current_position):
    """
    Mean Reversion with Bollinger Bands  [EVENT: close re-enters band]
    FIX: the bands are now calculated (earlier the code looked for
         'Bollinger_Upper' columns that never existed → never fired)
    """
    if current_position is not None:
        return None, None
    if idx < 2:
        return None, None
    
    bb_period = int(config.get('bb_period', 20))
    bb_std = float(config.get('bb_std', 2.0))
    rsi_oversold = config.get('mr_rsi_oversold', 30)
    rsi_overbought = config.get('mr_rsi_overbought', 70)
    
    key = f'_MR_BB_{bb_period}_{bb_std}'
    if key + '_U' not in df.columns:
        mid = tv_sma(df['Close'], bb_period)
        sd = tv_stdev(df['Close'], bb_period)
        df[key + '_M'] = mid
        df[key + '_U'] = mid + bb_std * sd
        df[key + '_L'] = mid - bb_std * sd
    
    current = df.iloc[idx]
    previous = df.iloc[idx - 1]
    current_price = current['Close']
    prev_price = previous['Close']
    bb_upper, bb_lower = current[key + '_U'], current[key + '_L']
    p_upper, p_lower = previous[key + '_U'], previous[key + '_L']
    rsi = current.get('RSI', 50)
    if any(_isnan(v) for v in (bb_upper, bb_lower, p_upper, p_lower, rsi)):
        return None, None
    
    if prev_price <= p_lower and current_price > bb_lower and rsi < rsi_oversold and current_price > prev_price:
        return 'BUY', current_price
    if prev_price >= p_upper and current_price < bb_upper and rsi > rsi_overbought and current_price < prev_price:
        return 'SELL', current_price
    return None, None

def check_momentum_breakout_adx(df, idx, config, current_position):
    """
    Momentum Breakout with ADX  [EVENT: close breaks N-bar high/low]
    """
    if current_position is not None:
        return None, None
    if idx < 50:
        return None, None
    
    adx_threshold = config.get('momentum_adx_threshold', 25)
    breakout_lookback = int(config.get('momentum_lookback', 20))
    min_volume_ratio = config.get('momentum_volume_ratio', 1.5)
    
    current = df.iloc[idx]
    current_price = current['Close']
    adx = current.get('ADX', np.nan)
    rsi = current.get('RSI', 50)
    if _isnan(adx) or adx < adx_threshold:
        return None, None
    if _isnan(rsi):
        rsi = 50
    
    recent_data = df.iloc[idx - breakout_lookback:idx]
    recent_high = recent_data['High'].max()
    recent_low = recent_data['Low'].min()
    
    if has_volume(df):
        avg_volume = recent_data['Volume'].mean()
        if current['Volume'] < avg_volume * min_volume_ratio:
            return None, None
    
    prev_close = df.iloc[idx - 1]['Close']
    if current_price > recent_high and rsi < 70 and prev_close <= recent_high:
        return 'BUY', current_price
    if current_price < recent_low and rsi > 30 and prev_close >= recent_low:
        return 'SELL', current_price
    return None, None

def check_support_resistance_bounce(df, idx, config, current_position):
    """
    Support Resistance Bounce Strategy  [STATE at level]
    """
    if current_position is not None:
        return None, None
    if idx < 100:
        return None, None
    
    sr_lookback = int(config.get('sr_lookback', 100))
    sr_tolerance = config.get('sr_tolerance', 0.2)
    # UI gives % (0.2 = 0.2%) → fraction
    sr_tolerance = sr_tolerance / 100 if sr_tolerance >= 0.05 else sr_tolerance
    min_touches = config.get('sr_min_touches', 3)
    
    current = df.iloc[idx]
    current_price = current['Close']
    current_high = current['High']
    current_low = current['Low']
    
    lookback_data = df.iloc[max(0, idx - sr_lookback):idx]
    highs = lookback_data['High'].values
    lows = lookback_data['Low'].values
    swing_highs, swing_lows = [], []
    for i in range(2, len(lookback_data) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append(lows[i])
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None, None
    
    def find_key_level(levels, current_level, tolerance):
        clusters = [lv for lv in levels if abs(lv - current_level) / current_level <= tolerance]
        if len(clusters) >= min_touches:
            return sum(clusters) / len(clusters)
        return None
    
    rsi = current.get('RSI', 50)
    if _isnan(rsi):
        rsi = 50
    support_level = find_key_level(swing_lows, current_low, sr_tolerance)
    if support_level is not None:
        if current_low <= support_level * (1 + sr_tolerance) and current_price > current_low and rsi < 50:
            return 'BUY', current_price
    resistance_level = find_key_level(swing_highs, current_high, sr_tolerance)
    if resistance_level is not None:
        if current_high >= resistance_level * (1 - sr_tolerance) and current_price < current_high and rsi > 50:
            return 'SELL', current_price
    return None, None


# Human-readable description of how each strategy fires (shown in the UI)
STRATEGY_SIGNAL_TYPE = {
    'EMA Crossover': 'EVENT – only on the candle where Fast EMA crosses Slow EMA',
    'Simple Buy': 'STATE – BUY whenever flat',
    'Simple Sell': 'STATE – SELL whenever flat',
    'Price Crosses Threshold': 'STATE – while price is above/below threshold',
    'RSI-ADX-EMA Combined': 'STATE – while RSI/ADX/EMA conditions hold',
    'Percentage Change': 'EVENT – candle with ≥ N% change vs previous close',
    'AI Price Action': 'STATE – 3 consecutive higher / lower closes',
    'Custom Strategy': 'EVENT for crosses/crossovers, STATE for pullback',
    'SuperTrend AI': 'EVENT – SuperTrend direction flip',
    'VWAP + Volume Spike': 'EVENT – price crosses session VWAP (needs volume)',
    'Bollinger Squeeze Breakout': 'EVENT – close breaks band after squeeze',
    'Elliott Waves + Ratio Charts': 'EVENT – 5th extreme confirmed',
    'Opening Range Breakout (ORB)': 'EVENT – close crosses ORB high / low',
    'Pivot Point Reversal': 'STATE – reversal candle at pivot level',
    'Ichimoku Cloud': 'EVENT – Tenkan/Kijun cross outside cloud',
    'Volume Breakout': 'STATE – high-volume directional candle (needs volume)',
    'Gap Trading Strategy': 'EVENT – price crosses back through session open',
    'Mean Reversion with Bollinger Bands': 'EVENT – close re-enters the band',
    'Momentum Breakout with ADX': 'EVENT – close breaks N-bar high/low',
    'Support Resistance Bounce': 'STATE – bounce at tested S/R level',
}

# ================================
# ADDITIONAL ENTRY FILTERS (all OFF by default)
# ================================
# Every filter receives the raw strategy signal direction (bull=True for
# BUY/LONG) and the SIGNAL candle index. When enabled, the entry happens only
# if the strategy fires AND every enabled filter passes.
# Each returns (passed: bool, detail: str).

def _num(x):
    try:
        v = float(x)
        return v
    except Exception:
        return np.nan


def _cache(df, key, fn):
    if key not in df.columns:
        df[key] = fn()
    return df[key]


def _get_dmi_cols(df, period):
    key = f'_F_DMI_{int(period)}'
    if key + '_ADX' not in df.columns:
        p, m, a = calculate_dmi(df, int(period))
        df[key + '_P'], df[key + '_M'], df[key + '_ADX'] = p, m, a
    return key


def _flt_adx(df, idx, bull, c):
    p = int(c.get('f_adx_period', 14)); thr = float(c.get('f_adx_threshold', 20))
    key = _get_dmi_cols(df, p)
    adx = _num(df[key + '_ADX'].iloc[idx])
    if np.isnan(adx):
        return False, 'ADX not available'
    ok = adx >= thr
    detail = f'ADX {adx:.1f} (min {thr})'
    if ok and c.get('f_adx_di', False):
        pdi, mdi = _num(df[key + '_P'].iloc[idx]), _num(df[key + '_M'].iloc[idx])
        ok = (pdi > mdi) if bull else (mdi > pdi)
        detail += f', +DI {pdi:.1f} / -DI {mdi:.1f}'
    return ok, detail


def _flt_rsi(df, idx, bull, c):
    p = int(c.get('f_rsi_period', 14))
    s = _cache(df, f'_F_RSI_{p}', lambda: calculate_rsi(df['Close'], p))
    r = _num(s.iloc[idx])
    if np.isnan(r):
        return False, 'RSI not available'
    if bull:
        lo, hi = float(c.get('f_rsi_bull_min', 50)), float(c.get('f_rsi_bull_max', 70))
    else:
        lo, hi = float(c.get('f_rsi_bear_min', 30)), float(c.get('f_rsi_bear_max', 50))
    return lo <= r <= hi, f'RSI {r:.1f} (need {lo:g}–{hi:g})'


def _flt_macd(df, idx, bull, c):
    f, s, g = int(c.get('f_macd_fast', 12)), int(c.get('f_macd_slow', 26)), int(c.get('f_macd_signal', 9))
    key = f'_F_MACD_{f}_{s}_{g}'
    if key not in df.columns:
        m = tv_ema(df['Close'], f) - tv_ema(df['Close'], s)
        df[key] = m
        df[key + '_S'] = tv_ema(m, g)
    m, sg = _num(df[key].iloc[idx]), _num(df[key + '_S'].iloc[idx])
    if np.isnan(m) or np.isnan(sg):
        return False, 'MACD not available'
    ok = m > sg if bull else m < sg
    if c.get('f_macd_zero', False):
        ok = ok and (m > 0 if bull else m < 0)
    return ok, f'MACD {m:.2f} / Signal {sg:.2f}'


def _flt_volume(df, idx, bull, c):
    if not has_volume(df):
        return True, 'no volume data for this ticker (filter skipped)'
    p, mult = int(c.get('f_volume_period', 20)), float(c.get('f_volume_mult', 1.5))
    vma = _cache(df, f'_F_VMA_{p}', lambda: tv_sma(df['Volume'], p))
    v, a = _num(df['Volume'].iloc[idx]), _num(vma.iloc[idx])
    if np.isnan(a):
        return False, 'Volume MA not available'
    return v >= mult * a, f'Vol {v:,.0f} vs {mult}× avg {a:,.0f}'


def _flt_bb(df, idx, bull, c):
    p, k = int(c.get('f_bb_period', 20)), float(c.get('f_bb_std', 2.0))
    mode = c.get('f_bb_mode', 'Price on trend side of Middle Band')
    mid = _cache(df, f'_F_BBM_{p}', lambda: tv_sma(df['Close'], p))
    sd = _cache(df, f'_F_BBS_{p}', lambda: tv_stdev(df['Close'], p))
    m, s_ = _num(mid.iloc[idx]), _num(sd.iloc[idx])
    if np.isnan(m) or np.isnan(s_):
        return False, 'BB not available'
    up, lo, cl = m + k * s_, m - k * s_, _num(df['Close'].iloc[idx])
    if mode.startswith('Breakout'):
        ok = cl > up if bull else cl < lo
    elif mode.startswith('Inside'):
        ok = lo < cl < up
    else:
        ok = cl > m if bull else cl < m
    return ok, f'Close {cl:.2f} | BB {lo:.2f}/{m:.2f}/{up:.2f}'


def _flt_regime(df, idx, bull, c):
    ln = int(c.get('f_regime_ema', 200)); adx_min = float(c.get('f_regime_adx', 20))
    mode = c.get('f_regime_mode', 'Trend Direction (Price vs EMA + slope)')
    ema = _cache(df, f'_F_EMA_{ln}', lambda: tv_ema(df['Close'], ln))
    e, e5 = _num(ema.iloc[idx]), _num(ema.iloc[max(0, idx - 5)])
    cl = _num(df['Close'].iloc[idx])
    adx = _num(df[_get_dmi_cols(df, 14) + '_ADX'].iloc[idx])
    trend_ok = False
    if not np.isnan(e) and not np.isnan(e5):
        trend_ok = (cl > e and e > e5) if bull else (cl < e and e < e5)
    if mode.startswith('Trending Market'):
        ok = not np.isnan(adx) and adx >= adx_min
    elif mode.startswith('Ranging'):
        ok = not np.isnan(adx) and adx < adx_min
    elif mode.startswith('Trend Direction +'):
        ok = trend_ok and not np.isnan(adx) and adx >= adx_min
    else:
        ok = trend_ok
    return ok, f'EMA{ln} {e:.2f}, ADX {adx:.1f}'


def _window(idx, n):
    return max(0, idx - int(n)), idx + 1


def _flt_bos(df, idx, bull, c):
    s, e = _window(idx, c.get('f_bos_lookback', 20))
    wanted = {'BOS_UP'} if bull else {'BOS_DN'}
    if c.get('f_bos_include_choch', True):
        wanted |= {'CHOCH_UP'} if bull else {'CHOCH_DN'}
    ev = df['MS_Event'].values[s:e]
    tr = df['MS_Trend'].iloc[idx]
    ok = tr == (1 if bull else -1) and any(x in wanted for x in ev)
    return ok, f'trend {int(tr)}, recent breaks: {[x for x in ev if x]}'


def _flt_choch(df, idx, bull, c):
    s, e = _window(idx, c.get('f_choch_lookback', 20))
    wanted = 'CHOCH_UP' if bull else 'CHOCH_DN'
    ev = df['MS_Event'].values[s:e]
    ok = df['MS_Trend'].iloc[idx] == (1 if bull else -1) and wanted in list(ev)
    return ok, f'{wanted} within {e - s} bars: {wanted in list(ev)}'


def _ensure_sweep_cols(df, kind):
    key = f'_F_SWEEP_{kind}'
    if key + '_BULL' in df.columns:
        return key
    if kind == 'minor':
        ph, pl = compute_pivots(df, 1, 1)
        lvl_low = pd.Series(pl, index=df.index).shift(2).ffill()
        lvl_high = pd.Series(ph, index=df.index).shift(2).ffill()
    else:
        lvl_low = df['Swing_Low'].shift(1)
        lvl_high = df['Swing_High'].shift(1)
    df[key + '_BULL'] = (df['Low'] < lvl_low) & (df['Close'] > lvl_low)
    df[key + '_BEAR'] = (df['High'] > lvl_high) & (df['Close'] < lvl_high)
    return key


def _flt_inducement(df, idx, bull, c):
    key = _ensure_sweep_cols(df, 'minor')
    s, e = _window(idx, c.get('f_ind_lookback', 10))
    swept = bool(df[key + ('_BULL' if bull else '_BEAR')].values[s:e].any())
    ok = swept
    if c.get('f_ind_trend', True):
        ok = ok and df['MS_Trend'].iloc[idx] == (1 if bull else -1)
    return ok, f'internal liquidity swept: {swept}'


def _flt_liquidity(df, idx, bull, c):
    key = _ensure_sweep_cols(df, 'major')
    s, e = _window(idx, c.get('f_liq_lookback', 20))
    swept = bool(df[key + ('_BULL' if bull else '_BEAR')].values[s:e].any())
    side = 'sell-side (below swing low)' if bull else 'buy-side (above swing high)'
    return swept, f'{side} liquidity swept: {swept}'


def _flt_ob(df, idx, bull, c):
    maxd = float(c.get('f_ob_max_dist', 0.5))
    cl = _num(df['Close'].iloc[idx])
    if bull:
        top, bot = _num(df['MS_BullOB_Top'].iloc[idx]), _num(df['MS_BullOB_Bot'].iloc[idx])
        if np.isnan(top):
            return False, 'no active bullish order block'
        ok = cl >= bot and (cl - top) / cl * 100 <= maxd
    else:
        top, bot = _num(df['MS_BearOB_Top'].iloc[idx]), _num(df['MS_BearOB_Bot'].iloc[idx])
        if np.isnan(top):
            return False, 'no active bearish order block'
        ok = cl <= top and (bot - cl) / cl * 100 <= maxd
    return ok, f'OB {bot:.2f}–{top:.2f}, close {cl:.2f}'


def _flt_mitigation(df, idx, bull, c):
    s, e = _window(idx, c.get('f_mit_lookback', 10))
    lo, hi, cl = df['Low'].values, df['High'].values, df['Close'].values
    if bull:
        top, bot = df['MS_BullOB_Top'].values, df['MS_BullOB_Bot'].values
        tapped = (~np.isnan(top[s:e])) & (lo[s:e] <= top[s:e]) & (cl[s:e] >= bot[s:e])
        ok = bool(tapped.any()) and not np.isnan(top[idx]) and cl[idx] > top[idx]
    else:
        top, bot = df['MS_BearOB_Top'].values, df['MS_BearOB_Bot'].values
        tapped = (~np.isnan(bot[s:e])) & (hi[s:e] >= bot[s:e]) & (cl[s:e] <= top[s:e])
        ok = bool(tapped.any()) and not np.isnan(bot[idx]) and cl[idx] < bot[idx]
    return ok, f'OB mitigated & respected: {ok}'


def _flt_fvg(df, idx, bull, c):
    n = int(c.get('f_fvg_lookback', 20))
    mode = c.get('f_fvg_mode', 'Unfilled FVG present')
    lo, hi = df['Low'].values, df['High'].values
    for k in range(idx, max(1, idx - n), -1):
        if k < 2:
            break
        if bull and lo[k] > hi[k - 2]:
            bottom, top = hi[k - 2], lo[k]
            after = lo[k + 1:idx + 1]
            mn = after.min() if len(after) else np.inf
            if mn <= bottom:
                continue  # filled
            if mode.startswith('Unfilled') or mn <= top:
                return True, f'bullish FVG {bottom:.2f}–{top:.2f}'
        if (not bull) and hi[k] < lo[k - 2]:
            top, bottom = lo[k - 2], hi[k]
            after = hi[k + 1:idx + 1]
            mx = after.max() if len(after) else -np.inf
            if mx >= top:
                continue
            if mode.startswith('Unfilled') or mx >= bottom:
                return True, f'bearish FVG {bottom:.2f}–{top:.2f}'
    return False, 'no qualifying FVG'


def _flt_fib(df, idx, bull, c):
    fib = float(c.get('f_fib_level', 0.618)); tol = float(c.get('f_fib_tol', 0.1))
    sh, sl = _num(df['Swing_High'].iloc[idx]), _num(df['Swing_Low'].iloc[idx])
    shp, slp = _num(df['Swing_High_Pos'].iloc[idx]), _num(df['Swing_Low_Pos'].iloc[idx])
    if any(np.isnan(v) for v in (sh, sl, shp, slp)) or sh <= sl:
        return False, 'no swing leg'
    rng = sh - sl
    cl = _num(df['Close'].iloc[idx])
    if bull:
        if not slp < shp:
            return False, 'last leg is not an up-leg'
        lvl = sh - rng * fib
        touched = df['Low'].iloc[int(shp):idx + 1].min() <= lvl * (1 + tol / 100)
        ok = cl > lvl and (touched or not c.get('f_fib_touch', True))
    else:
        if not shp < slp:
            return False, 'last leg is not a down-leg'
        lvl = sl + rng * fib
        touched = df['High'].iloc[int(slp):idx + 1].max() >= lvl * (1 - tol / 100)
        ok = cl < lvl and (touched or not c.get('f_fib_touch', True))
    return ok, f'Fib {fib} level {lvl:.2f}, touched {touched}'


def _flt_ema_pullback(df, idx, bull, c):
    fp, sp, n = int(c.get('f_pb_fast', 9)), int(c.get('f_pb_slow', 20)), int(c.get('f_pb_lookback', 15))
    ef = _cache(df, f'_F_EMA_{fp}', lambda: tv_ema(df['Close'], fp)).values
    es = _cache(df, f'_F_EMA_{sp}', lambda: tv_ema(df['Close'], sp)).values
    diff = ef - es
    if idx < 2 or np.isnan(diff[idx]):
        return False, 'EMAs not available'
    if (bull and diff[idx] <= 0) or ((not bull) and diff[idx] >= 0):
        return False, 'EMAs not aligned'
    cross = None
    for k in range(idx, max(0, idx - n), -1):
        if np.isnan(diff[k - 1]):
            break
        if bull and diff[k - 1] <= 0 < diff[k]:
            cross = k; break
        if (not bull) and diff[k - 1] >= 0 > diff[k]:
            cross = k; break
    if cross is None:
        return False, f'no crossover in last {n} bars'
    lo, hi, cl = df['Low'].values, df['High'].values, df['Close'].values
    rng = range(cross + 1, idx + 1)
    if bull:
        pulled = any(lo[k] <= ef[k] for k in rng)
        brk = cl[idx] > hi[idx - 1]
    else:
        pulled = any(hi[k] >= ef[k] for k in rng)
        brk = cl[idx] < lo[idx - 1]
    return pulled and brk, f'cross {idx - cross} bars ago, pullback {pulled}, prev-candle break {brk}'


def _parse_periods(txt, default):
    try:
        vals = sorted({int(x.strip()) for x in str(txt).split(',') if x.strip()})
        return vals or default
    except Exception:
        return default


def _flt_multi_ma(df, idx, bull, c, kind):
    periods = _parse_periods(c.get(f'f_multi_{kind}_periods', '9,20,50' if kind == 'ema' else '20,50,200'), [20, 50])
    fn = tv_ema if kind == 'ema' else tv_sma
    vals = [_num(_cache(df, f'_F_{kind.upper()}_{p}', lambda p=p: fn(df['Close'], p)).iloc[idx]) for p in periods]
    if any(np.isnan(v) for v in vals):
        return False, f'{kind.upper()}s not available'
    cl = _num(df['Close'].iloc[idx])
    ok = all(cl > v for v in vals) if bull else all(cl < v for v in vals)
    if c.get(f'f_multi_{kind}_stacked', True):
        pairs = list(zip(vals, vals[1:]))
        ok = ok and (all(a > b for a, b in pairs) if bull else all(a < b for a, b in pairs))
    return ok, ', '.join(f'{kind.upper()}{p}={v:.2f}' for p, v in zip(periods, vals))


def _flt_volatility(df, idx, bull, c):
    p = int(c.get('f_vol_atr_period', 14))
    atr = _cache(df, f'_F_ATR_{p}', lambda: calculate_atr(df, p))
    a, cl = _num(atr.iloc[idx]), _num(df['Close'].iloc[idx])
    if np.isnan(a):
        return False, 'ATR not available'
    pct = a / cl * 100
    lo, hi = float(c.get('f_vol_min_pct', 0.02)), float(c.get('f_vol_max_pct', 2.0))
    return lo <= pct <= hi, f'ATR% {pct:.3f} (range {lo}–{hi})'


def _flt_atr(df, idx, bull, c):
    p = int(c.get('f_atr_period', 14))
    atr = _cache(df, f'_F_ATR_{p}', lambda: calculate_atr(df, p))
    a = _num(atr.iloc[idx])
    if np.isnan(a):
        return False, 'ATR not available'
    mn, mx = float(c.get('f_atr_min', 5.0)), float(c.get('f_atr_max', 0.0))
    ok = a >= mn and (mx <= 0 or a <= mx)
    if ok and c.get('f_atr_rising', False):
        avg = _num(_cache(df, f'_F_ATRSMA_{p}', lambda: tv_sma(atr, p)).iloc[idx])
        ok = not np.isnan(avg) and a > avg
    return ok, f'ATR {a:.2f} pts (min {mn}, max {mx or "∞"})'


def _flt_orb(df, idx, bull, c):
    key = ensure_orb_cols(df, int(c.get('f_orb_minutes', 15)), c.get('f_orb_start', '09:15'))
    if not bool(df[key + '_READY'].iloc[idx]):
        return False, 'opening range not complete yet'
    h, l, cl = _num(df[key + '_H'].iloc[idx]), _num(df[key + '_L'].iloc[idx]), _num(df['Close'].iloc[idx])
    buf = float(c.get('f_orb_buffer', 0.0))
    ok = cl > h + buf if bull else cl < l - buf
    return ok, f'ORB {l:.2f}–{h:.2f}, close {cl:.2f}'


PIVOT_LEVEL_COLS = {'Pivot (P)': 'PIV_P', 'R1': 'PIV_R1', 'R2': 'PIV_R2', 'R3': 'PIV_R3',
                    'S1': 'PIV_S1', 'S2': 'PIV_S2', 'S3': 'PIV_S3'}


def _flt_pivot(df, idx, bull, c):
    ensure_daily_pivots(df)
    name = c.get('f_piv_level', 'Nearest Level')
    k = max(1, int(c.get('f_piv_sustain', 3))); n = int(c.get('f_piv_lookback', 20))
    cl = df['Close'].values
    levels = {nm: _num(df[col].iloc[idx]) for nm, col in PIVOT_LEVEL_COLS.items()}
    if name == 'Nearest Level':
        if bull:
            cands = [v for v in levels.values() if not np.isnan(v) and v < cl[idx]]
            L = max(cands) if cands else np.nan
        else:
            cands = [v for v in levels.values() if not np.isnan(v) and v > cl[idx]]
            L = min(cands) if cands else np.nan
    else:
        L = levels.get(name, np.nan)
    if np.isnan(L) or idx - k + 1 < 0:
        return False, 'pivot level not available'
    recent = cl[idx - k + 1:idx + 1]
    sustain = all(recent > L) if bull else all(recent < L)
    ok = sustain
    if c.get('f_piv_break', True):
        prior = cl[max(0, idx - k - n + 1):idx - k + 1]
        brk = bool(len(prior)) and (any(prior <= L) if bull else any(prior >= L))
        ok = ok and brk
    return ok, f'level {L:.2f}, sustained {k} bars: {sustain}'


def _flt_hhhl(df, idx, bull, c):
    sh, psh = _num(df['Swing_High'].iloc[idx]), _num(df['Prev_Swing_High'].iloc[idx])
    sl, psl = _num(df['Swing_Low'].iloc[idx]), _num(df['Prev_Swing_Low'].iloc[idx])
    if any(np.isnan(v) for v in (sh, psh, sl, psl)):
        return False, 'not enough swings'
    ok = (sh > psh and sl > psl) if bull else (sh < psh and sl < psl)
    return ok, f'highs {psh:.2f}→{sh:.2f}, lows {psl:.2f}→{sl:.2f}'


def _flt_vwap(df, idx, bull, c):
    v = _num(df['VWAP'].iloc[idx]) if 'VWAP' in df.columns else np.nan
    if np.isnan(v):
        return True, 'no VWAP (ticker has no volume) – filter skipped'
    cl = _num(df['Close'].iloc[idx])
    return (cl > v) if bull else (cl < v), f'close {cl:.2f} vs VWAP {v:.2f}'


def _flt_supertrend(df, idx, bull, c):
    p, m = int(c.get('f_st_period', 10)), float(c.get('f_st_mult', 3.0))
    key = f'_F_ST_{p}_{m}'
    if key not in df.columns:
        _, d = compute_supertrend(df, p, m)
        df[key] = d
    d = _num(df[key].iloc[idx])
    return d == (1 if bull else -1), f'SuperTrend direction {d:+.0f}'


def _flt_roc(df, idx, bull, c):
    p, thr = int(c.get('f_roc_period', 10)), float(c.get('f_roc_threshold', 0.05))
    roc = _cache(df, f'_F_ROC_{p}', lambda: (df['Close'] - df['Close'].shift(p)) / df['Close'].shift(p) * 100)
    r = _num(roc.iloc[idx])
    if np.isnan(r):
        return False, 'ROC not available'
    return (r >= thr) if bull else (r <= -thr), f'ROC {r:.3f}% (thr ±{thr})'


def _flt_stoch(df, idx, bull, c):
    kp, sk, dp = int(c.get('f_stoch_k', 14)), int(c.get('f_stoch_smooth', 3)), int(c.get('f_stoch_d', 3))
    ob, os_ = float(c.get('f_stoch_ob', 80)), float(c.get('f_stoch_os', 20))
    key = f'_F_STOCH_{kp}_{sk}_{dp}'
    if key + '_K' not in df.columns:
        ll = df['Low'].rolling(kp).min(); hh = df['High'].rolling(kp).max()
        raw = 100 * (df['Close'] - ll) / (hh - ll).replace(0, np.nan)
        k = tv_sma(raw, sk)
        df[key + '_K'], df[key + '_D'] = k, tv_sma(k, dp)
    k, d = _num(df[key + '_K'].iloc[idx]), _num(df[key + '_D'].iloc[idx])
    if np.isnan(k) or np.isnan(d):
        return False, 'Stochastic not available'
    ok = (k > d and k < ob) if bull else (k < d and k > os_)
    return ok, f'%K {k:.1f} / %D {d:.1f}'


def _flt_candle(df, idx, bull, c):
    o, h, l, cl = (_num(df[x].iloc[idx]) for x in ('Open', 'High', 'Low', 'Close'))
    rng = h - l
    if rng <= 0:
        return False, 'zero-range candle'
    body_pct = abs(cl - o) / rng * 100
    mn = float(c.get('f_candle_body_pct', 60))
    ok = body_pct >= mn and ((cl > o) if bull else (cl < o))
    return ok, f'body {body_pct:.0f}% of range (min {mn:g}%)'


# key → (label, function)
ENTRY_FILTERS = {
    'f_adx':          ('ADX Trend-Strength Filter', _flt_adx),
    'f_rsi':          ('RSI Filter', _flt_rsi),
    'f_macd':         ('MACD Filter', _flt_macd),
    'f_volume':       ('Volume Filter (N× average)', _flt_volume),
    'f_bb':           ('Bollinger Band Filter', _flt_bb),
    'f_regime':       ('Market Regime Filter', _flt_regime),
    'f_bos':          ('SMC: Break of Structure (BOS)', _flt_bos),
    'f_choch':        ('SMC: Change of Character (CHoCH)', _flt_choch),
    'f_inducement':   ('SMC: Inducement (internal liquidity sweep)', _flt_inducement),
    'f_ob':           ('SMC: Order Block', _flt_ob),
    'f_mitigation':   ('SMC: Order Block Mitigation', _flt_mitigation),
    'f_fvg':          ('SMC: Fair Value Gap (FVG)', _flt_fvg),
    'f_liquidity':    ('Sell-side / Buy-side Liquidity Swept', _flt_liquidity),
    'f_fib':          ('Fibonacci Level Sustain', _flt_fib),
    'f_ema_pullback': ('EMA Pullback after Crossover + Prev-High Break', _flt_ema_pullback),
    'f_multi_ema':    ('Multiple EMA Alignment', lambda df, i, b, c: _flt_multi_ma(df, i, b, c, 'ema')),
    'f_multi_sma':    ('Multiple SMA Alignment', lambda df, i, b, c: _flt_multi_ma(df, i, b, c, 'sma')),
    'f_volatility':   ('Volatility Filter (ATR % of price)', _flt_volatility),
    'f_atr':          ('ATR Filter (points)', _flt_atr),
    'f_orb':          ('Opening Range Breakout Filter', _flt_orb),
    'f_pivot':        ('Pivot Level Break & Sustain', _flt_pivot),
    'f_hhhl':         ('Market Structure: HH-HL / LH-LL', _flt_hhhl),
    'f_vwap':         ('VWAP Side Filter', _flt_vwap),
    'f_supertrend':   ('SuperTrend Direction Filter', _flt_supertrend),
    'f_roc':          ('Momentum: Rate of Change (ROC)', _flt_roc),
    'f_stoch':        ('Momentum: Stochastic', _flt_stoch),
    'f_candle':       ('Momentum: Strong Candle Body', _flt_candle),
}


def check_entry_filters(df, idx, signal, config):
    """
    Evaluate every ENABLED additional entry filter for the raw strategy signal.
    Returns (all_passed, [(label, passed, detail), ...])
    """
    bull = signal in ('BUY', 'LONG')
    results = []
    all_ok = True
    for key, (label, fn) in ENTRY_FILTERS.items():
        if not config.get(key, False):
            continue
        try:
            ok, detail = fn(df, idx, bull, config)
        except Exception as e:
            ok, detail = False, f'error: {e}'
        ok = bool(ok)
        results.append((label, ok, detail))
        all_ok = all_ok and ok
    return all_ok, results


def enabled_filter_labels(config):
    return [label for key, (label, _) in ENTRY_FILTERS.items() if config.get(key, False)]

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
        if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
            ts_ist = timestamp.astimezone(IST)
        else:
            ts_ist = IST.localize(timestamp)
        
        start_time = config.get('trade_window_start', datetime.strptime("09:30", "%H:%M").time())
        end_time = config.get('trade_window_end', datetime.strptime("15:00", "%H:%M").time())
        
        current_time = ts_ist.time()
        return start_time <= current_time <= end_time
    except Exception:
        return True  # Default to allowing trade if error

def should_allow_trade_direction(signal, config):
    """
    Check if signal matches allowed trade direction filter.
    """
    direction_filter = config.get('trade_direction', 'Both (LONG + SHORT)')
    
    if direction_filter == 'Both (LONG + SHORT)':
        return True
    elif direction_filter == 'LONG Only':
        return signal in ('BUY', 'LONG')
    elif direction_filter == 'SHORT Only':
        return signal in ('SELL', 'SHORT')
    
    return True

def get_entry_position_type(signal, config):
    """
    Convert raw signal into the entry position type ('LONG' / 'SHORT').
    If "Reverse Entry" is enabled, the entry is flipped.
    """
    position_type = 'LONG' if signal in ('BUY', 'LONG') else 'SHORT'
    
    if config.get('reverse_entry', False):
        position_type = 'SHORT' if position_type == 'LONG' else 'LONG'
    
    return position_type

def calculate_brokerage(entry_price, exit_price, quantity, config):
    """
    Calculate brokerage amount.
    """
    if not config.get('include_brokerage', False):
        return 0.0
    
    brokerage_type = config.get('brokerage_type', 'Fixed per Trade')
    
    if brokerage_type == 'Fixed per Trade':
        return float(config.get('brokerage_per_trade', 20.0))
    else:  # Percentage of Turnover
        turnover = (entry_price + exit_price) * quantity
        brokerage_pct = float(config.get('brokerage_percentage', 0.03)) / 100
        return turnover * brokerage_pct

def uses_signal_exit(config):
    """True if SL or Target type exits on an opposite strategy signal"""
    return config.get('sl_type') in SIGNAL_EXIT_TYPES or config.get('target_type') in SIGNAL_EXIT_TYPES

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
    'Opening Range Breakout (ORB)': check_opening_range_breakout,
    'Pivot Point Reversal': check_pivot_point_reversal,
    'Ichimoku Cloud': check_ichimoku_cloud,
    'Volume Breakout': check_volume_breakout,
    'Gap Trading Strategy': check_gap_trading,
    'Mean Reversion with Bollinger Bands': check_mean_reversion_bollinger,
    'Momentum Breakout with ADX': check_momentum_breakout_adx,
    'Support Resistance Bounce': check_support_resistance_bounce,
}

# ================================
# CANDLE / SWING LEVEL HELPERS
# ================================

def _to_float(x):
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except Exception:
        return None


def _resolve_level(df, idx, position_type, token, ref_price, config, for_target):
    """
    Return the raw level for a token:
      cur_candle / prev_candle  -> candle low/high
      cur_swing  / prev_swing   -> most recent / previous confirmed swing that
                                   lies on the correct side of ref_price
    SL  : LONG uses lows (below price),  SHORT uses highs (above price)
    TGT : LONG uses highs (above price), SHORT uses lows (below price)
    """
    is_long = position_type == 'LONG'
    want_high = (is_long and for_target) or ((not is_long) and (not for_target))
    if token in ('cur_candle', 'prev_candle'):
        row = df.iloc[idx]
        if token == 'cur_candle':
            return _to_float(row['High'] if want_high else row['Low'])
        return _to_float(row['Prev_High'] if want_high else row['Prev_Low'])
    which = 'current' if token == 'cur_swing' else 'previous'
    kind = 'high' if want_high else 'low'
    side = 'above' if want_high else 'below'
    return pick_swing(df, idx, kind, which, ref_price, side, config)


def _valid_sl(level, position_type, ref_price, buffer=0.0):
    """SL must be below price for LONG, above price for SHORT"""
    level = _to_float(level)
    if level is None:
        return None
    if position_type == 'LONG':
        lv = level - buffer
        return lv if lv < ref_price else None
    lv = level + buffer
    return lv if lv > ref_price else None


def _valid_target(level, position_type, ref_price, buffer=0.0):
    """Target must be above price for LONG, below price for SHORT (exit `buffer` points early)"""
    level = _to_float(level)
    if level is None:
        return None
    if position_type == 'LONG':
        lv = level - buffer
        return lv if lv > ref_price else None
    lv = level + buffer
    return lv if lv < ref_price else None


def _points_sl(position_type, entry_price, config):
    points = float(config.get('sl_points', 10))
    return entry_price - points if position_type == 'LONG' else entry_price + points


def _points_target(position_type, entry_price, config):
    points = float(config.get('target_points', 20))
    return entry_price + points if position_type == 'LONG' else entry_price - points

# ================================
# STOP LOSS CALCULATION
# ================================

def calculate_initial_sl(position_type, entry_price, df, idx, config):
    """
    Calculate initial stop loss based on SL type

    Candle / Swing based types (incl. their trailing variants) use the REAL
    candle/swing level. sl_points is used only as the last fallback when no
    valid level exists on the correct side of the entry price.
    """
    sl_type = config.get('sl_type', 'Custom Points')
    current = df.iloc[idx]
    buffer = float(config.get('swing_buffer_points', 0.0))
    
    if sl_type == 'Custom Points':
        return _points_sl(position_type, entry_price, config)
    
    elif sl_type == 'P&L Based (Rupees)':
        rupees = config.get('sl_rupees', 100)
        quantity = config.get('quantity', 1)
        points = rupees / quantity
        return entry_price - points if position_type == 'LONG' else entry_price + points
    
    elif sl_type == 'ATR-based':
        if _isnan(current['ATR']):
            return _points_sl(position_type, entry_price, config)
        sl_distance = current['ATR'] * config.get('sl_atr_multiplier', 1.5)
        return entry_price - sl_distance if position_type == 'LONG' else entry_price + sl_distance
    
    elif sl_type in SL_LEVEL_TOKENS:
        for token in SL_LEVEL_TOKENS[sl_type]:
            lvl = _valid_sl(_resolve_level(df, idx, position_type, token, entry_price, config, False),
                            position_type, entry_price, buffer)
            if lvl is not None:
                return lvl
        return _points_sl(position_type, entry_price, config)
    
    elif sl_type in ('Signal-based (Reverse Crossover)', 'Strategy-based Signal'):
        # No price-based SL, exits only on signal
        return None
    
    elif sl_type == 'Volatility-Adjusted Trailing SL':
        if not _isnan(current['ATR']):
            d = current['ATR'] * config.get('sl_atr_multiplier', 1.5)
            return entry_price - d if position_type == 'LONG' else entry_price + d
        return _points_sl(position_type, entry_price, config)
    
    else:
        # Trailing SL (Points), Trailing Profit/Loss, Break-even, Cost-to-Cost, default
        return _points_sl(position_type, entry_price, config)

# ================================
# TARGET CALCULATION
# ================================

def calculate_initial_target(position_type, entry_price, df, idx, config):
    """
    Calculate initial target based on target type
    """
    target_type = config.get('target_type', 'Custom Points')
    current = df.iloc[idx]
    buffer = float(config.get('swing_buffer_points', 0.0))
    
    if target_type == 'Custom Points':
        return _points_target(position_type, entry_price, config)
    
    elif target_type == 'P&L Based (Rupees)':
        rupees = config.get('target_rupees', 200)
        quantity = config.get('quantity', 1)
        points = rupees / quantity
        return entry_price + points if position_type == 'LONG' else entry_price - points
    
    elif target_type == 'Risk-Reward Based':
        sl_price = calculate_initial_sl(position_type, entry_price, df, idx, config)
        rr_ratio = config.get('risk_reward_ratio', 2.0)
        if sl_price is None:
            return _points_target(position_type, entry_price, config)
        reward = abs(entry_price - sl_price) * rr_ratio
        return entry_price + reward if position_type == 'LONG' else entry_price - reward
    
    elif target_type == 'ATR-based':
        if _isnan(current['ATR']):
            return _points_target(position_type, entry_price, config)
        d = current['ATR'] * config.get('target_atr_multiplier', 2.0)
        return entry_price + d if position_type == 'LONG' else entry_price - d
    
    elif target_type == 'Dynamic Trailing SL+Target (Lock Profits)':
        d = config.get('dynamic_trail_target_points', 20)
        return entry_price + d if position_type == 'LONG' else entry_price - d
    
    elif target_type in TARGET_LEVEL_TOKENS:
        for token in TARGET_LEVEL_TOKENS[target_type]:
            lvl = _valid_target(_resolve_level(df, idx, position_type, token, entry_price, config, True),
                                position_type, entry_price, buffer)
            if lvl is not None:
                return lvl
        return _points_target(position_type, entry_price, config)
    
    elif target_type in ('Trailing Target (Points)', 'Trailing Target + Signal Based',
                         '50% Exit at Target (Partial)'):
        return _points_target(position_type, entry_price, config)
    
    elif target_type in ('Signal-based (Reverse Crossover)', 'Strategy-based Signal'):
        # No price-based target - exit only on signal
        return None
    
    else:
        return _points_target(position_type, entry_price, config)

# ================================
# TRAILING UPDATES
# ================================

def _ratchet(old, new, position_type):
    """Move level only in the favourable direction (up for LONG, down for SHORT)"""
    if new is None:
        return old
    if old is None:
        return new
    return max(old, new) if position_type == 'LONG' else min(old, new)


def update_trailing_sl(position, current_price, df, idx, config):
    """
    Update trailing stop loss
    
    Implements all trailing SL types including:
    - Standard trailing
    - Cost-to-Cost 3-phase trailing
    - Candle/Swing-based trailing (real levels)
    - Volatility-adjusted trailing
    - Dynamic Trailing SL+Target (trails with price)
    """
    sl_type = config.get('sl_type', 'Custom Points')
    target_type = config.get('target_type', 'Custom Points')
    position_type = position['type']
    current_sl = position['sl_price']
    entry_price = position['entry_price']
    current = df.iloc[idx]
    buffer = float(config.get('swing_buffer_points', 0.0))
    
    # ── DYNAMIC TRAILING SL+TARGET ────────────────────────────────────────
    if target_type == 'Dynamic Trailing SL+Target (Lock Profits)':
        sl_distance = config.get('dynamic_trail_sl_points', 10)
        if position_type == 'LONG':
            return _ratchet(current_sl, current_price - sl_distance, 'LONG')
        return _ratchet(current_sl, current_price + sl_distance, 'SHORT')
    
    if current_sl is None:
        return current_sl
    
    # ── COST-TO-COST + N POINTS TRAILING SL ───────────────────────────────
    if sl_type == 'Cost-to-Cost + N Points Trailing SL':
        initial_sl_distance = config.get('sl_points', 10)
        K = config.get('ctc_trigger_points', 3)
        N = config.get('ctc_offset_points', 2)
        
        if position_type == 'LONG':
            points_in_favor = current_price - entry_price
            if points_in_favor < K:
                return max(current_sl, current_price - initial_sl_distance)
            elif points_in_favor < initial_sl_distance:
                return max(current_sl, entry_price + N)
            else:
                return max(current_sl, current_price - (initial_sl_distance - N))
        else:
            points_in_favor = entry_price - current_price
            if points_in_favor < K:
                return min(current_sl, current_price + initial_sl_distance)
            elif points_in_favor < initial_sl_distance:
                return min(current_sl, entry_price - N)
            else:
                return min(current_sl, current_price + (initial_sl_distance - N))
    
    # ── TRAILING SL (POINTS) ──────────────────────────────────────────────
    elif sl_type == 'Trailing SL (Points)':
        trail_points = config.get('sl_points', 10)
        if position_type == 'LONG':
            return max(current_sl, current_price - trail_points)
        return min(current_sl, current_price + trail_points)
    
    # ── TRAILING PROFIT (RUPEES) ──────────────────────────────────────────
    elif sl_type == 'Trailing Profit (Rupees)':
        quantity = config.get('quantity', 1)
        trail_rupees = config.get('sl_trail_rupees', 50)
        if position_type == 'LONG':
            current_profit = (current_price - entry_price) * quantity
        else:
            current_profit = (entry_price - current_price) * quantity
        position['highest_profit'] = max(position.get('highest_profit', current_profit), current_profit)
        if position['highest_profit'] - current_profit >= trail_rupees:
            return current_price
        return current_sl
    
    # ── TRAILING LOSS (RUPEES) ────────────────────────────────────────────
    elif sl_type == 'Trailing Loss (Rupees)':
        quantity = config.get('quantity', 1)
        trail_rupees = config.get('sl_trail_rupees', 50)
        if position_type == 'LONG':
            current_profit = (current_price - entry_price) * quantity
        else:
            current_profit = (entry_price - current_price) * quantity
        position['lowest_profit'] = min(position.get('lowest_profit', current_profit), current_profit)
        if current_profit - position['lowest_profit'] <= -trail_rupees:
            return current_price
        return current_sl
    
    # ── TRAILING SL + CURRENT/PREVIOUS CANDLE, CURRENT/PREVIOUS SWING ────
    elif sl_type in TRAIL_SL_LEVEL_TOKEN:
        token = TRAIL_SL_LEVEL_TOKEN[sl_type]
        lvl = _resolve_level(df, idx, position_type, token, current_price, config, for_target=False)
        new_sl = _valid_sl(lvl, position_type, current_price, buffer)
        return _ratchet(current_sl, new_sl, position_type)
    
    # ── VOLATILITY-ADJUSTED TRAILING SL ───────────────────────────────────
    elif sl_type == 'Volatility-Adjusted Trailing SL':
        if _isnan(current['ATR']):
            return current_sl
        trail_distance = current['ATR'] * config.get('sl_atr_multiplier', 1.5)
        if position_type == 'LONG':
            return max(current_sl, current_price - trail_distance)
        return min(current_sl, current_price + trail_distance)
    
    # ── BREAK-EVEN AFTER 50% TARGET ───────────────────────────────────────
    elif sl_type == 'Break-even After 50% Target':
        target_price = position.get('target_price')
        if target_price is None:
            return current_sl
        if position_type == 'LONG':
            halfway = entry_price + (target_price - entry_price) * 0.5
            if current_price >= halfway:
                return max(current_sl, entry_price)
        else:
            halfway = entry_price - (entry_price - target_price) * 0.5
            if current_price <= halfway:
                return min(current_sl, entry_price)
        return current_sl
    
    return current_sl

def update_trailing_target(position, current_price, df, idx, config):
    """
    Update trailing target
    """
    target_type = config.get('target_type', 'Custom Points')
    position_type = position['type']
    current_target = position['target_price']
    buffer = float(config.get('swing_buffer_points', 0.0))
    
    if target_type == 'Trailing Target (Points)':
        trail_points = config.get('target_points', 20)
        if position_type == 'LONG':
            return _ratchet(current_target, current_price + trail_points, 'LONG')
        return _ratchet(current_target, current_price - trail_points, 'SHORT')
    
    elif target_type == 'Dynamic Trailing SL+Target (Lock Profits)':
        target_distance = config.get('dynamic_trail_target_points', 20)
        if position_type == 'LONG':
            return _ratchet(current_target, current_price + target_distance, 'LONG')
        return _ratchet(current_target, current_price - target_distance, 'SHORT')
    
    elif target_type in TRAIL_TARGET_LEVEL_TOKEN:
        # Target follows the candle / swing level beyond current price (only moves further away)
        token = TRAIL_TARGET_LEVEL_TOKEN[target_type]
        lvl = _resolve_level(df, idx, position_type, token, current_price, config, for_target=True)
        new_t = _valid_target(lvl, position_type, current_price, buffer)
        return _ratchet(current_target, new_t, position_type)
    
    return current_target

# ================================
# BACKTESTING ENGINE
# ================================

def run_backtest(df, config):
    """
    Run backtesting on historical data
    
    Signals are evaluated at candle close (Method 1: entry at signal close;
    Method 2: entry at next candle open). Candles before the requested period
    ('In_Period' == False) are only used to warm up indicators.

    Returns:
        tuple: (trades_list, metrics_dict, debug_info, skipped_trades_list)
    """
    trades = []
    skipped_trades = []
    position = None
    strategy_name = config.get('strategy', 'EMA Crossover')
    strategy_func = STRATEGY_FUNCTIONS.get(strategy_name, check_ema_crossover_strategy)
    prevent_overlapping = config.get('prevent_overlapping_trades', True)
    signal_exit = uses_signal_exit(config)
    
    total_candles = len(df)
    candles_analyzed = 0
    signals_generated = 0
    signals_filtered = 0
    filter_rejections = {}
    trades_entered = 0
    trades_exited = 0
    signals_skipped = 0
    
    # Start after indicator warm-up AND inside the requested period
    start_idx = max(50, int(config.get('ema_slow', 20)) + 10)
    if 'In_Period' in df.columns and df['In_Period'].any():
        start_idx = max(start_idx, int(np.argmax(df['In_Period'].values)))
    
    for idx in range(start_idx, len(df)):
        candles_analyzed += 1
        current_data = df.iloc[idx]
        current_price = current_data['Close']
        
        if position is None:
            if not is_within_trade_window(current_data['Datetime'], config):
                continue
            
            signal, entry_price = strategy_func(df, idx, config, None)
            
            if signal:
                if not should_allow_trade_direction(signal, config):
                    continue
                
                signals_generated += 1
                
                # Additional entry filters (all must pass)
                ok, f_results = check_entry_filters(df, idx, signal, config)
                if not ok:
                    signals_filtered += 1
                    for label, passed, _ in f_results:
                        if not passed:
                            filter_rejections[label] = filter_rejections.get(label, 0) + 1
                    continue
                
                position_type = get_entry_position_type(signal, config)
                
                use_method2 = config.get('use_backtest_method2', False)
                if use_method2:
                    if idx + 1 < len(df):
                        actual_entry_price = df.iloc[idx + 1]['Open']
                        actual_entry_time = df.iloc[idx + 1]['Datetime']
                    else:
                        continue
                else:
                    actual_entry_price = entry_price
                    actual_entry_time = current_data['Datetime']
                
                sl_price = calculate_initial_sl(position_type, actual_entry_price, df, idx, config)
                target_price = calculate_initial_target(position_type, actual_entry_price, df, idx, config)
                
                entry_metrics = {
                    'strategy': strategy_name,
                    'entry_idx': idx,
                }
                
                if strategy_name == 'EMA Crossover' or 'EMA_Fast' in df.columns:
                    ema_fast_val = current_data.get('EMA_Fast', np.nan)
                    ema_slow_val = current_data.get('EMA_Slow', np.nan)
                    ema_angle = current_data.get('EMA_Fast_Angle', np.nan)
                    entry_metrics.update({
                        'ema_fast_period': config.get('ema_fast', 9),
                        'ema_slow_period': config.get('ema_slow', 20),
                        'ema_fast_entry': ema_fast_val,
                        'ema_slow_entry': ema_slow_val,
                        'ema_angle_entry': ema_angle,
                        'price_fast_ema_diff_entry': actual_entry_price - ema_fast_val if not pd.isna(ema_fast_val) else np.nan,
                        'price_slow_ema_diff_entry': actual_entry_price - ema_slow_val if not pd.isna(ema_slow_val) else np.nan,
                        'fast_slow_ema_diff_entry': ema_fast_val - ema_slow_val if not pd.isna(ema_fast_val) and not pd.isna(ema_slow_val) else np.nan,
                    })
                
                position = {
                    'type': position_type,
                    'entry_price': actual_entry_price,
                    'entry_time': actual_entry_time,
                    'sl_price': sl_price,
                    'target_price': target_price,
                    'initial_sl': sl_price,
                    'initial_target': target_price,
                    'quantity': config.get('quantity', 1),
                    'highest_price': actual_entry_price,
                    'lowest_price': actual_entry_price,
                    'entry_metrics': entry_metrics,
                    'entry_idx': idx,
                }
                trades_entered += 1
        
        else:
            # Position exists - check if new signal came (overlapping scenario)
            if prevent_overlapping:
                signal, signal_price = strategy_func(df, idx, config, None)
                if signal and should_allow_trade_direction(signal, config) and \
                        check_entry_filters(df, idx, signal, config)[0]:
                    signals_skipped += 1
                    skipped_position_type = get_entry_position_type(signal, config)
                    skipped_sl = calculate_initial_sl(skipped_position_type, signal_price, df, idx, config)
                    skipped_target = calculate_initial_target(skipped_position_type, signal_price, df, idx, config)
                    
                    skipped_exit_price = None
                    skipped_exit_reason = None
                    skipped_exit_time = None
                    
                    for future_idx in range(idx + 1, min(idx + 100, len(df))):
                        future_price = df.iloc[future_idx]['Close']
                        future_time = df.iloc[future_idx]['Datetime']
                        if skipped_sl is not None:
                            if (skipped_position_type == 'LONG' and future_price <= skipped_sl) or \
                               (skipped_position_type == 'SHORT' and future_price >= skipped_sl):
                                skipped_exit_price, skipped_exit_reason, skipped_exit_time = skipped_sl, 'SL Hit (Skipped)', future_time
                                break
                        if skipped_target is not None:
                            if (skipped_position_type == 'LONG' and future_price >= skipped_target) or \
                               (skipped_position_type == 'SHORT' and future_price <= skipped_target):
                                skipped_exit_price, skipped_exit_reason, skipped_exit_time = skipped_target, 'Target Hit (Skipped)', future_time
                                break
                    
                    if skipped_exit_price is None and idx + 20 < len(df):
                        skipped_exit_price = df.iloc[idx + 20]['Close']
                        skipped_exit_reason = 'Simulated Exit (Skipped)'
                        skipped_exit_time = df.iloc[idx + 20]['Datetime']
                    
                    if skipped_exit_price:
                        qty = config.get('quantity', 1)
                        if skipped_position_type == 'LONG':
                            skipped_pnl = (skipped_exit_price - signal_price) * qty
                        else:
                            skipped_pnl = (signal_price - skipped_exit_price) * qty
                        brokerage = calculate_brokerage(signal_price, skipped_exit_price, qty, config)
                        skipped_trades.append({
                            'entry_time': current_data['Datetime'],
                            'exit_time': skipped_exit_time,
                            'type': skipped_position_type,
                            'entry_price': signal_price,
                            'exit_price': skipped_exit_price,
                            'sl_price': skipped_sl,
                            'target_price': skipped_target,
                            'quantity': qty,
                            'pnl': skipped_pnl,
                            'brokerage': brokerage,
                            'net_pnl': skipped_pnl - brokerage,
                            'exit_reason': skipped_exit_reason,
                            'note': 'Overlapped with active trade'
                        })
            
            # Method 2: the entry candle (signal candle) is not traded yet
            if config.get('use_backtest_method2', False) and idx == position.get('entry_idx'):
                continue
            
            position['highest_price'] = max(position['highest_price'], current_price)
            position['lowest_price'] = min(position['lowest_price'], current_price)
            
            if position['sl_price'] is not None or config.get('target_type') == 'Dynamic Trailing SL+Target (Lock Profits)':
                position['sl_price'] = update_trailing_sl(position, current_price, df, idx, config)
            if position['target_price'] is not None:
                position['target_price'] = update_trailing_target(position, current_price, df, idx, config)
            
            exit_reason = None
            exit_price = current_price
            
            if not is_within_trade_window(current_data['Datetime'], config):
                exit_reason = 'Trade Window Closed'
                exit_price = current_price
            
            if exit_reason is None and position['sl_price'] is not None:
                if (position['type'] == 'LONG' and current_price <= position['sl_price']) or \
                   (position['type'] == 'SHORT' and current_price >= position['sl_price']):
                    exit_reason = 'SL Hit'
                    exit_price = position['sl_price']
            
            if exit_reason is None and position['target_price'] is not None:
                if (position['type'] == 'LONG' and current_price >= position['target_price']) or \
                   (position['type'] == 'SHORT' and current_price <= position['target_price']):
                    exit_reason = 'Target Hit'
                    exit_price = position['target_price']
            
            # Signal-based exit (FIX: strategy is asked for a fresh signal; earlier the
            # open position was passed in, so most strategies never returned an exit signal)
            if exit_reason is None and signal_exit:
                signal, _ = strategy_func(df, idx, config, None)
                if signal:
                    raw_side = 'LONG' if signal in ('BUY', 'LONG') else 'SHORT'
                    entry_side = get_entry_position_type(signal, config)
                    if entry_side != position['type']:
                        exit_reason = 'Strategy Signal Exit'
                        exit_price = current_price
            
            if exit_reason:
                if position['type'] == 'LONG':
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                else:
                    pnl = (position['entry_price'] - exit_price) * position['quantity']
                
                brokerage = calculate_brokerage(position['entry_price'], exit_price, position['quantity'], config)
                net_pnl = pnl - brokerage
                
                duration_minutes = (current_data['Datetime'] - position['entry_time']).total_seconds() / 60
                
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
                
                entry_metrics = position.get('entry_metrics', {})
                
                trade = {
                    'entry_time': position['entry_time'],
                    'exit_time': current_data['Datetime'],
                    'type': position['type'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'highest_price': position['highest_price'],
                    'lowest_price': position['lowest_price'],
                    'initial_sl': position.get('initial_sl'),
                    'initial_target': position.get('initial_target'),
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
                position = None
    
    metrics = compute_trade_metrics(trades)
    
    debug_info = {
        'total_candles': total_candles,
        'warmup_candles': int(start_idx),
        'candles_analyzed': candles_analyzed,
        'signals_generated': signals_generated,
        'signals_filtered': signals_filtered,
        'filter_rejections': filter_rejections,
        'trades_entered': trades_entered,
        'trades_completed': trades_exited,
        'signals_skipped': signals_skipped,
        'overlapping_trades': len(skipped_trades),
    }
    
    return trades, metrics, debug_info, skipped_trades


def compute_trade_metrics(trades):
    """Metrics from a list of trade dicts (uses numeric pnl / net_pnl)"""
    if not trades:
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0.0,
            'total_pnl': 0.0, 'avg_pnl': 0.0, 'max_drawdown': 0.0,
            'total_brokerage': 0.0, 'total_net_pnl': 0.0, 'avg_net_pnl': 0.0,
        }
    df_t = pd.DataFrame(trades)
    pnl = pd.to_numeric(df_t['pnl'], errors='coerce').fillna(0.0)
    brk = pd.to_numeric(df_t.get('brokerage', pd.Series(0.0, index=df_t.index)), errors='coerce').fillna(0.0)
    net = pd.to_numeric(df_t.get('net_pnl', pnl - brk), errors='coerce')
    net = net.fillna(pnl - brk)
    total = len(df_t)
    wins = int((pnl > 0).sum())
    cum = net.cumsum()
    return {
        'total_trades': total,
        'winning_trades': wins,
        'losing_trades': int((pnl < 0).sum()),
        'win_rate': wins / total * 100 if total else 0.0,
        'total_pnl': float(pnl.sum()),
        'avg_pnl': float(pnl.mean()),
        'total_brokerage': float(brk.sum()),
        'total_net_pnl': float(net.sum()),
        'avg_net_pnl': float(net.mean()),
        'max_drawdown': float((cum - cum.cummax()).min()),
    }

# ================================
# LIVE TRADING
# ================================

def add_log(message):
    """Add timestamped log message"""
    timestamp = datetime.now(IST).strftime('%H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    if 'live_logs' not in st.session_state:
        st.session_state['live_logs'] = []
    st.session_state['live_logs'].append(log_entry)
    # keep memory bounded
    if len(st.session_state['live_logs']) > 2000:
        st.session_state['live_logs'] = st.session_state['live_logs'][-1000:]


def get_signal_index(df, interval, config, now=None):
    """
    Index of the candle used for signal evaluation.
    'Signal on candle close' (default): the last CLOSED candle, so a signal is
    final (no repaint) – exactly like TradingView 'once per bar close' alerts.
    Otherwise: the forming (last) candle.
    """
    last = len(df) - 1
    if not config.get('signal_on_candle_close', True) or last < 1:
        return last
    now = now or datetime.now(IST)
    t = df['Datetime'].iloc[-1]
    if interval in INTRADAY_INTERVALS:
        candle_end = t + timedelta(minutes=INTERVAL_MINUTES.get(interval, 1))
        # small grace: yfinance may publish the new candle a few seconds late
        return last - 1 if now < candle_end else last
    if interval == '1d':
        close_t = t.replace(hour=15, minute=30, second=0, microsecond=0)
        return last - 1 if (now.date() == t.date() and now < close_t) else last
    candle_end = t + timedelta(minutes=INTERVAL_MINUTES.get(interval, 1440))
    return last - 1 if now < candle_end else last


def close_position(position, exit_price, exit_reason, config, log=True):
    """
    Close the current position (shared by automatic exits and manual square-off).
    Records the trade with numeric P&L / brokerage / net P&L, exits the broker
    position, and clears session state. Returns the trade record.
    """
    qty = position.get('quantity', 1)
    entry = float(position['entry_price'])
    exit_price = float(exit_price)
    if position['type'] == 'LONG':
        pnl = (exit_price - entry) * qty
    else:
        pnl = (entry - exit_price) * qty
    brokerage = calculate_brokerage(entry, exit_price, qty, config)
    net_pnl = pnl - brokerage
    now = datetime.now(IST)

    trade_record = {
        'entry_time': position.get('entry_time'),
        'exit_time': now,
        'ticker': position.get('ticker', config.get('asset', 'Unknown')),
        'type': position['type'],
        'entry_price': entry,
        'exit_price': exit_price,
        'sl_price': position.get('sl_price'),
        'target_price': position.get('target_price'),
        'highest_price': position.get('highest_price', exit_price),
        'lowest_price': position.get('lowest_price', exit_price),
        'quantity': qty,
        'pnl': pnl,
        'brokerage': brokerage,
        'net_pnl': net_pnl,
        'exit_reason': exit_reason,
        'price_range': position.get('highest_price', exit_price) - position.get('lowest_price', exit_price),
        'price_change_pct': abs(exit_price - entry) / entry * 100 if entry else 0.0,
    }

    if log:
        add_log(f"🚪 EXITING POSITION: {exit_reason} @ {exit_price:.2f}")
        add_log(f"💰 P&L: ₹{pnl:.2f} | Entry: ₹{entry:.2f} | Exit: ₹{exit_price:.2f}")
        if config.get('include_brokerage', False):
            add_log(f"💸 Brokerage: ₹{brokerage:.2f} | 💵 Net P&L: ₹{net_pnl:.2f}")

    st.session_state.setdefault('trade_history', []).append(trade_record)
    add_log("📝 Trade saved to history")

    # Exit broker position if exists
    if config.get('dhan_enabled', False):
        broker_position = st.session_state.get('broker_position')
        dhan_broker = st.session_state.get('dhan_broker')
        if broker_position and dhan_broker:
            try:
                exit_info = dhan_broker.exit_broker_position(broker_position, exit_price, exit_reason, add_log)
                st.session_state['broker_exit'] = exit_info
            except Exception as e:
                add_log(f"🏦 ⚠️ Broker exit error: {e}")
        elif broker_position and not dhan_broker:
            add_log("🏦 ⚠️ Broker position exists but broker is not initialized – close it manually in Dhan!")

    st.session_state['last_exit_time'] = now
    st.session_state['position'] = None
    st.session_state['broker_position'] = None
    add_log("✅ Position closed, session cleared")
    return trade_record


def manual_square_off(config):
    """
    Manual square-off: close the active position at the latest price.
    Price source priority: fresh 1-minute quote of the LOCKED ticker →
    last live price seen by the algo → entry price (with a warning).
    """
    position = st.session_state.get('position')
    if not position:
        st.warning("No active position to close")
        return None

    active_cfg = st.session_state.get('config') or config
    price, source = None, ''
    try:
        price = get_latest_price(position.get('ticker', active_cfg.get('asset', 'NIFTY 50')),
                                 position.get('custom_ticker'))
        source = 'latest quote'
    except Exception:
        price = None
    if price is None:
        cd = st.session_state.get('current_data')
        if cd is not None:
            try:
                price = float(cd['Close'])
                source = 'last live price'
            except Exception:
                price = None
    if price is None:
        price = float(position['entry_price'])
        source = 'entry price (no quote available!)'
        st.warning("⚠️ Could not fetch a live price – squared off at entry price for records. Verify in broker.")

    add_log(f"✋ MANUAL SQUARE-OFF requested ({source}: {price:.2f})")
    record = close_position(position, price, 'Manual Close', active_cfg)
    st.success(f"✅ Position squared off @ ₹{price:.2f} | P&L ₹{record['pnl']:.2f}")
    return record


def live_trading_iteration():
    """
    Single iteration of live trading loop
    
    1. Fetches fresh data (with indicator warm-up history)
    2. Calculates indicators (TradingView-identical)
    3. Checks entry signal on the signal candle (last closed candle by default)
       – each signal candle can create only ONE entry
    4. Applies direction filter + additional entry filters
    5. Manages position (trailing SL/target on live price) and broker orders
    """
    config = st.session_state.get('config', {})
    
    if st.session_state.get('clearing_in_progress', False):
        add_log("🏦 ⏳ Position clearing in progress - skipping iteration")
        return
    
    position = st.session_state.get('position')
    
    if position is not None and 'ticker' in position:
        ticker = position['ticker']
        custom_ticker = position.get('custom_ticker')
    else:
        ticker = config.get('asset', 'NIFTY 50')
        custom_ticker = config.get('custom_ticker', None)
    
    interval = INTERVAL_MAPPING.get(config.get('interval', '1 minute'), '1m')
    period = PERIOD_MAPPING.get(config.get('period', '7 days'), '7d')
    enhanced_mode = config.get('enhanced_live_trading', False)
    
    if enhanced_mode:
        current_time = datetime.now(IST)
        interval_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60}.get(interval, 1)
        is_round_time = (current_time.minute % interval_minutes == 0 and current_time.second < 5)
        last_candle_fetch = st.session_state.get('last_candle_fetch_time')
        
        if is_round_time and (last_candle_fetch is None or
                              (current_time - last_candle_fetch).total_seconds() >= interval_minutes * 60 - 10):
            add_log(f"🔄 Enhanced Mode: Fetching candlestick data at round time {current_time.strftime('%H:%M:%S')}")
            df_candles = fetch_data(ticker, interval, period, is_live_trading=True, custom_ticker=custom_ticker)
            if df_candles is not None and not df_candles.empty:
                df_candles = calculate_all_indicators(df_candles, config)
                st.session_state['indicator_df'] = df_candles
                st.session_state['last_candle_fetch_time'] = current_time
                add_log(f"✅ Indicators calculated from {len(df_candles)} candles")
        
        df = st.session_state.get('indicator_df')
        if df is None or df.empty:
            add_log("⚠️ No indicator data available, fetching fresh data")
            df = fetch_data(ticker, interval, period, is_live_trading=True, custom_ticker=custom_ticker)
            if df is None or df.empty:
                add_log("❌ Failed to fetch data")
                return
            df = calculate_all_indicators(df, config)
            st.session_state['indicator_df'] = df
        
        df_live = fetch_data(ticker, interval, '1d', is_live_trading=True, custom_ticker=custom_ticker, quiet=True)
        if df_live is None or df_live.empty:
            add_log("❌ Failed to fetch live data")
            return
        current_data = df.iloc[-1].copy()
        live_row = df_live.iloc[-1]
        for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Datetime']:
            current_data[col] = live_row[col]
        current_price = float(live_row['Close'])
    else:
        df = fetch_data(ticker, interval, period, is_live_trading=True, custom_ticker=custom_ticker)
        if df is None or df.empty:
            add_log("❌ Failed to fetch data")
            return
        df = calculate_all_indicators(df, config)
        current_data = df.iloc[-1]
        current_price = float(current_data['Close'])
    
    live_idx = len(df) - 1
    sig_idx = get_signal_index(df, interval, config)
    signal_candle_time = df['Datetime'].iloc[sig_idx]
    
    st.session_state['current_data'] = current_data
    st.session_state['live_df'] = df
    st.session_state['signal_candle'] = df.iloc[sig_idx]
    
    strategy_name = config.get('strategy', 'EMA Crossover')
    strategy_func = STRATEGY_FUNCTIONS.get(strategy_name, check_ema_crossover_strategy)
    position = st.session_state.get('position')
    
    if position is None:
        if not is_within_trade_window(current_data['Datetime'], config):
            add_log("⏰ Outside trade window - no new entries allowed")
            return
        
        cooldown_enabled = config.get('enable_entry_cooldown', False)
        cooldown_seconds = config.get('entry_cooldown_seconds', 0)
        if cooldown_enabled and cooldown_seconds > 0:
            last_exit_time = st.session_state.get('last_exit_time')
            if last_exit_time is not None:
                time_since_exit = (datetime.now(IST) - last_exit_time).total_seconds()
                if time_since_exit < cooldown_seconds:
                    add_log(f"⏳ Entry Cooldown Active: {int(cooldown_seconds - time_since_exit)}s remaining")
                    return
        
        # One entry per signal candle (prevents repeated entries on the same signal)
        if st.session_state.get('last_signal_candle_time') == signal_candle_time:
            add_log(f"⏳ Waiting for next candle (signal candle {signal_candle_time.strftime('%H:%M')} already processed) | Price {current_price:.2f}")
            return
        
        signal, signal_close = strategy_func(df, sig_idx, config, None)
        
        if signal:
            if cooldown_enabled and cooldown_seconds > 0:
                if st.session_state.get('last_signal_type') == signal:
                    add_log(f"⛔ Same signal {signal} as last trade - waiting for signal change (cooldown protection)")
                    st.session_state['last_signal_candle_time'] = signal_candle_time
                    return
            
            if not should_allow_trade_direction(signal, config):
                add_log(f"⛔ Signal {signal} filtered out by direction setting: {config.get('trade_direction')}")
                st.session_state['last_signal_candle_time'] = signal_candle_time
                return
            
            ok, f_results = check_entry_filters(df, sig_idx, signal, config)
            for label, passed, detail in f_results:
                add_log(f"   {'✅' if passed else '❌'} {label}: {detail}")
            if not ok:
                add_log(f"⛔ Signal {signal} rejected by additional entry filters")
                if config.get('signal_on_candle_close', True):
                    st.session_state['last_signal_candle_time'] = signal_candle_time
                return
            
            if config.get('dhan_enabled', False) and st.session_state.get('broker_position'):
                add_log("⛔ Broker order already active - preventing duplicate order")
                return
            
            entry_price = current_price  # actual fill happens now at live price
            add_log(f"✅ SIGNAL DETECTED: {signal} on candle {signal_candle_time.strftime('%Y-%m-%d %H:%M')} "
                    f"(close {signal_close:.2f}) → entry @ live {entry_price:.2f}")
            st.session_state['last_signal_type'] = signal
            st.session_state['last_signal_candle_time'] = signal_candle_time
            
            position_type = get_entry_position_type(signal, config)
            sl_price = calculate_initial_sl(position_type, entry_price, df, sig_idx, config)
            target_price = calculate_initial_target(position_type, entry_price, df, sig_idx, config)
            
            add_log(f"📈 ENTERING {position_type} POSITION @ {entry_price:.2f}")
            sl_display = f"{sl_price:.2f}" if sl_price is not None else "Not Set"
            target_display = f"{target_price:.2f}" if target_price is not None else "Not Set"
            add_log(f"🛡️ Initial SL: {sl_display} | 🎯 Target: {target_display}")
            
            position = {
                'type': position_type,
                'signal': signal,
                'entry_price': entry_price,
                'entry_time': datetime.now(IST),
                'signal_candle_time': signal_candle_time,
                'sl_price': sl_price,
                'target_price': target_price,
                'quantity': config.get('quantity', 1),
                'highest_price': entry_price,
                'lowest_price': entry_price,
                'ticker': config.get('asset', 'NIFTY 50'),
                'custom_ticker': config.get('custom_ticker'),
            }
            st.session_state['position'] = position
            add_log(f"📊 Locked ticker: {position['ticker']} | Entry Price: ₹{entry_price:.2f}")
            
            if config.get('dhan_enabled', False):
                add_log("🏦 Dhan broker enabled, attempting to place order...")
                dhan_broker = st.session_state.get('dhan_broker')
                if dhan_broker:
                    try:
                        if config.get('clear_positions_before_entry', False):
                            add_log("🏦 🧹 Clearing all existing positions before new entry...")
                            st.session_state['clearing_in_progress'] = True
                            clear_result = dhan_broker.clear_all_positions(add_log, convert_to_market=True)
                            st.session_state['clearing_in_progress'] = False
                            if not clear_result['clearing_complete']:
                                add_log("🏦 ⚠️ Position clearing incomplete - aborting new entry")
                                return
                            total_cleared = clear_result['cancelled_orders'] + clear_result['converted_orders'] + clear_result['closed_positions']
                            if total_cleared > 0:
                                add_log(f"🏦 ✅ Cleared {total_cleared} orders/positions")
                            else:
                                add_log("🏦 ℹ️ No existing positions to clear")
                        
                        # Broker order follows the actual position side, so Reverse Entry
                        # also applies to the broker (options: LONG → CE, SHORT → PE)
                        broker_signal = 'BUY' if position_type == 'LONG' else 'SELL'
                        broker_position = dhan_broker.enter_broker_position(broker_signal, entry_price, config, add_log)
                        st.session_state['broker_position'] = broker_position
                    except Exception as e:
                        st.session_state['clearing_in_progress'] = False
                        add_log(f"🏦 ⚠️ Broker order error: {e}")
                        add_log(f"🏦 Error details: {traceback.format_exc()}")
                else:
                    add_log("🏦 ⚠️ Broker not initialized")
        else:
            add_log(f"⏳ No entry signal on candle {signal_candle_time.strftime('%H:%M')} (Price: {current_price:.2f})")
    
    else:
        # Position exists - monitor for exit
        add_log(f"📊 Monitoring {position['type']} position @ {current_price:.2f}")
        
        price_diff_pct = abs(current_price - position['entry_price']) / position['entry_price'] * 100
        if price_diff_pct > 50:
            add_log(f"⚠️ WARNING: Current price ₹{current_price:.2f} differs by {price_diff_pct:.1f}% from entry - possible ticker mismatch")
        
        position['highest_price'] = max(position['highest_price'], current_price)
        position['lowest_price'] = min(position['lowest_price'], current_price)
        
        if position['type'] == 'LONG':
            current_pnl = (current_price - position['entry_price']) * position['quantity']
        else:
            current_pnl = (position['entry_price'] - current_price) * position['quantity']
        add_log(f"💰 Current P&L: ₹{current_pnl:.2f}")
        
        # Trailing SL/Target use the live price and the latest candle
        old_sl = position['sl_price']
        if old_sl is not None or config.get('target_type') == 'Dynamic Trailing SL+Target (Lock Profits)':
            position['sl_price'] = update_trailing_sl(position, current_price, df, live_idx, config)
            if position['sl_price'] != old_sl and position['sl_price'] is not None:
                add_log(f"🛡️ SL Updated: {old_sl if old_sl is None else round(old_sl, 2)} → {position['sl_price']:.2f}")
        
        old_target = position['target_price']
        if old_target is not None:
            position['target_price'] = update_trailing_target(position, current_price, df, live_idx, config)
            if position['target_price'] != old_target:
                add_log(f"🎯 Target Updated: {old_target:.2f} → {position['target_price']:.2f}")
        
        exit_reason = None
        exit_price = current_price
        
        if not is_within_trade_window(current_data['Datetime'], config):
            exit_reason = 'Trade Window Closed'
            add_log("⏰ TRADE WINDOW CLOSED - Force exiting position")
        
        if exit_reason is None and position['sl_price'] is not None:
            if (position['type'] == 'LONG' and current_price <= position['sl_price']) or \
               (position['type'] == 'SHORT' and current_price >= position['sl_price']):
                exit_reason = 'SL Hit'
                exit_price = position['sl_price']
                add_log(f"🛑 STOP LOSS HIT! Price {current_price:.2f} vs SL {position['sl_price']:.2f}")
        
        if exit_reason is None and position['target_price'] is not None:
            if (position['type'] == 'LONG' and current_price >= position['target_price']) or \
               (position['type'] == 'SHORT' and current_price <= position['target_price']):
                exit_reason = 'Target Hit'
                exit_price = position['target_price']
                add_log(f"🎯 TARGET HIT! Price {current_price:.2f} vs Target {position['target_price']:.2f}")
        
        if exit_reason is None and uses_signal_exit(config) and \
                signal_candle_time != position.get('signal_candle_time'):
            signal, _ = strategy_func(df, sig_idx, config, None)
            if signal and get_entry_position_type(signal, config) != position['type']:
                exit_reason = 'Strategy Signal Exit'
                exit_price = current_price
                add_log(f"🔄 OPPOSITE STRATEGY SIGNAL: {signal}")
                st.session_state['last_signal_candle_time'] = signal_candle_time
        
        if exit_reason:
            if abs(exit_price - position['entry_price']) / position['entry_price'] * 100 > 50:
                add_log("⚠️ Possible ticker mismatch! Re-fetching locked ticker price...")
                verified = get_latest_price(position.get('ticker'), position.get('custom_ticker'))
                if verified is not None and abs(verified - position['entry_price']) < abs(exit_price - position['entry_price']):
                    add_log(f"✅ Using verified price {verified:.2f} instead of {exit_price:.2f}")
                    exit_price = verified
            close_position(position, exit_price, exit_reason, config)
        else:
            add_log("⏳ No exit conditions met - holding position")

# ================================
# UI COMPONENTS
# ================================

def render_entry_filters_ui(config):
    """Sidebar: additional entry filters (every checkbox OFF by default)"""
    with st.sidebar.expander("🧰 Additional Entry Filters (all OFF by default)", expanded=False):
        st.caption("When enabled, an entry happens only if the strategy signal fires AND every "
                   "enabled filter is true (checked on the signal candle, in the direction of the "
                   "raw signal).")

        def chk(key, help_txt=None):
            config[key] = st.checkbox(ENTRY_FILTERS[key][0], value=False, key=f"chk_{key}", help=help_txt)
            return config[key]

        st.markdown("**📈 Trend / Oscillators**")
        if chk('f_adx', "ADX must be ≥ threshold (trend strength)."):
            c1, c2 = st.columns(2)
            config['f_adx_period'] = c1.number_input("ADX Period", 1, 100, 14, key="f_adx_p")
            config['f_adx_threshold'] = c2.number_input("Min ADX", 0.0, 100.0, 20.0, key="f_adx_t")
            config['f_adx_di'] = st.checkbox("Require +DI > -DI (BUY) / -DI > +DI (SELL)", False, key="f_adx_di")
        if chk('f_rsi', "BUY needs RSI in the bull range, SELL needs RSI in the bear range."):
            config['f_rsi_period'] = st.number_input("RSI Period", 1, 100, 14, key="f_rsi_p")
            c1, c2 = st.columns(2)
            config['f_rsi_bull_min'] = c1.number_input("BUY RSI min", 0.0, 100.0, 50.0, key="f_rsi_bmin")
            config['f_rsi_bull_max'] = c2.number_input("BUY RSI max", 0.0, 100.0, 70.0, key="f_rsi_bmax")
            config['f_rsi_bear_min'] = c1.number_input("SELL RSI min", 0.0, 100.0, 30.0, key="f_rsi_smin")
            config['f_rsi_bear_max'] = c2.number_input("SELL RSI max", 0.0, 100.0, 50.0, key="f_rsi_smax")
        if chk('f_macd', "BUY: MACD > Signal. SELL: MACD < Signal."):
            c1, c2, c3 = st.columns(3)
            config['f_macd_fast'] = c1.number_input("Fast", 1, 100, 12, key="f_macd_f")
            config['f_macd_slow'] = c2.number_input("Slow", 1, 200, 26, key="f_macd_s")
            config['f_macd_signal'] = c3.number_input("Signal", 1, 50, 9, key="f_macd_g")
            config['f_macd_zero'] = st.checkbox("Also require MACD above/below zero", False, key="f_macd_z")
        if chk('f_volume', "Signal candle volume ≥ N × average volume. Skipped for indices (no volume)."):
            c1, c2 = st.columns(2)
            config['f_volume_period'] = c1.number_input("Avg Period", 1, 200, 20, key="f_vol_p")
            config['f_volume_mult'] = c2.number_input("N × Avg", 0.1, 20.0, 1.5, step=0.1, key="f_vol_m")
        if chk('f_bb'):
            c1, c2 = st.columns(2)
            config['f_bb_period'] = c1.number_input("BB Period", 2, 200, 20, key="f_bb_p")
            config['f_bb_std'] = c2.number_input("BB Std", 0.5, 5.0, 2.0, step=0.1, key="f_bb_s")
            config['f_bb_mode'] = st.selectbox("BB Condition", ["Price on trend side of Middle Band",
                                                                "Breakout beyond Band",
                                                                "Inside Bands (avoid extremes)"], key="f_bb_mode")
        if chk('f_regime', "Trade only in the right market regime."):
            config['f_regime_mode'] = st.selectbox("Regime", ["Trend Direction (Price vs EMA + slope)",
                                                              "Trending Market Only (ADX ≥ min)",
                                                              "Ranging Market Only (ADX < min)",
                                                              "Trend Direction + ADX ≥ min"], key="f_reg_mode")
            c1, c2 = st.columns(2)
            config['f_regime_ema'] = c1.number_input("Regime EMA", 5, 500, 200, key="f_reg_ema")
            config['f_regime_adx'] = c2.number_input("ADX level", 0.0, 100.0, 20.0, key="f_reg_adx")
        if chk('f_supertrend', "BUY only in SuperTrend uptrend, SELL only in downtrend."):
            c1, c2 = st.columns(2)
            config['f_st_period'] = c1.number_input("ATR Period", 1, 100, 10, key="f_st_p")
            config['f_st_mult'] = c2.number_input("Factor", 0.5, 10.0, 3.0, step=0.5, key="f_st_m")
        if chk('f_vwap', "BUY above session VWAP, SELL below. Skipped when no volume."):
            pass
        if chk('f_multi_ema', "Price above all EMAs (BUY) / below all (SELL)."):
            config['f_multi_ema_periods'] = st.text_input("EMA periods (comma)", "9,20,50", key="f_mema_p")
            config['f_multi_ema_stacked'] = st.checkbox("Require stacked order (fast > slow for BUY)", True, key="f_mema_s")
        if chk('f_multi_sma'):
            config['f_multi_sma_periods'] = st.text_input("SMA periods (comma)", "20,50,200", key="f_msma_p")
            config['f_multi_sma_stacked'] = st.checkbox("Require stacked order", True, key="f_msma_s")

        st.markdown("**🏛️ Market Structure / SMC** (uses Swing Detection settings)")
        if chk('f_bos', "Structure is bullish and price broke a swing high recently (mirror for SELL)."):
            config['f_bos_lookback'] = st.number_input("BOS within last N bars", 1, 500, 20, key="f_bos_n")
            config['f_bos_include_choch'] = st.checkbox("Count CHoCH as a structure break too", True, key="f_bos_c")
        if chk('f_choch', "A change of character (trend flip) in signal direction happened recently."):
            config['f_choch_lookback'] = st.number_input("CHoCH within last N bars", 1, 500, 20, key="f_choch_n")
        if chk('f_inducement', "Internal (minor) liquidity was swept: wick beyond a minor swing and close back."):
            config['f_ind_lookback'] = st.number_input("Sweep within last N bars", 1, 200, 10, key="f_ind_n")
            config['f_ind_trend'] = st.checkbox("Require structure in signal direction", True, key="f_ind_t")
        if chk('f_ob', "An active (not invalidated) order block exists behind price, and price is near it."):
            config['f_ob_max_dist'] = st.number_input("Max distance from OB (%)", 0.0, 10.0, 0.5, step=0.05, key="f_ob_d")
        if chk('f_mitigation', "Price tapped into the order block recently and reacted away from it."):
            config['f_mit_lookback'] = st.number_input("Mitigation within last N bars", 1, 200, 10, key="f_mit_n")
        if chk('f_fvg'):
            config['f_fvg_lookback'] = st.number_input("FVG within last N bars", 3, 500, 20, key="f_fvg_n")
            config['f_fvg_mode'] = st.selectbox("FVG condition", ["Unfilled FVG present",
                                                                  "Price retesting FVG"], key="f_fvg_mode")
        if chk('f_liquidity', "BUY: sell-side liquidity (below a swing low) swept. SELL: buy-side swept."):
            config['f_liq_lookback'] = st.number_input("Sweep within last N bars", 1, 500, 20, key="f_liq_n")
        if chk('f_hhhl', "BUY: higher high + higher low. SELL: lower high + lower low."):
            pass
        if chk('f_fib', "Price retraced to the Fibonacci level of the last swing leg and is holding it."):
            config['f_fib_level'] = st.selectbox("Fib level", [0.236, 0.382, 0.5, 0.618, 0.786], index=3, key="f_fib_l")
            config['f_fib_tol'] = st.number_input("Touch tolerance (%)", 0.0, 5.0, 0.1, step=0.05, key="f_fib_t")
            config['f_fib_touch'] = st.checkbox("Require price to have touched the level", True, key="f_fib_req")
        if chk('f_ema_pullback', "After a fresh crossover, price pulled back to the fast EMA and the signal "
                                 "candle closes above the previous candle high (below low for SELL). "
                                 "Best combined with non-crossover strategies."):
            c1, c2, c3 = st.columns(3)
            config['f_pb_fast'] = c1.number_input("Fast", 1, 200, 9, key="f_pb_f")
            config['f_pb_slow'] = c2.number_input("Slow", 1, 400, 20, key="f_pb_s")
            config['f_pb_lookback'] = c3.number_input("Bars", 2, 200, 15, key="f_pb_n")
        if chk('f_orb', "BUY above opening-range high, SELL below low (after the range completes)."):
            c1, c2 = st.columns(2)
            config['f_orb_minutes'] = c1.number_input("Range (min)", 1, 240, 15, key="f_orb_m")
            config['f_orb_start'] = c2.text_input("Session start", "09:15", key="f_orb_s")
            config['f_orb_buffer'] = st.number_input("Buffer (points)", 0.0, 1000.0, 0.0, key="f_orb_b")
        if chk('f_pivot', "Price broke a daily pivot level and has sustained beyond it."):
            config['f_piv_level'] = st.selectbox("Level", ["Nearest Level", "Pivot (P)", "R1", "R2", "R3",
                                                           "S1", "S2", "S3"], key="f_piv_l")
            c1, c2 = st.columns(2)
            config['f_piv_sustain'] = c1.number_input("Sustain bars", 1, 50, 3, key="f_piv_k")
            config['f_piv_lookback'] = c2.number_input("Break lookback", 1, 500, 20, key="f_piv_n")
            config['f_piv_break'] = st.checkbox("Require the break (cross) before sustaining", True, key="f_piv_b")

        st.markdown("**🌪️ Volatility / Momentum**")
        if chk('f_volatility', "ATR as % of price must be inside the range."):
            config['f_vol_atr_period'] = st.number_input("ATR Period", 1, 100, 14, key="f_vola_p")
            c1, c2 = st.columns(2)
            config['f_vol_min_pct'] = c1.number_input("Min ATR %", 0.0, 50.0, 0.02, step=0.01, format="%.3f", key="f_vola_min")
            config['f_vol_max_pct'] = c2.number_input("Max ATR %", 0.0, 50.0, 2.0, step=0.1, key="f_vola_max")
        if chk('f_atr', "ATR in points must be ≥ min (and ≤ max if set)."):
            config['f_atr_period'] = st.number_input("ATR Period", 1, 100, 14, key="f_atr_p")
            c1, c2 = st.columns(2)
            config['f_atr_min'] = c1.number_input("Min ATR (pts)", 0.0, 100000.0, 5.0, key="f_atr_min")
            config['f_atr_max'] = c2.number_input("Max ATR (0=off)", 0.0, 100000.0, 0.0, key="f_atr_max")
            config['f_atr_rising'] = st.checkbox("Require ATR above its average (expanding)", False, key="f_atr_r")
        if chk('f_roc', "Rate of change over N bars ≥ +threshold % (BUY) / ≤ -threshold % (SELL)."):
            c1, c2 = st.columns(2)
            config['f_roc_period'] = c1.number_input("ROC Period", 1, 200, 10, key="f_roc_p")
            config['f_roc_threshold'] = c2.number_input("Threshold %", 0.0, 50.0, 0.05, step=0.01, format="%.3f", key="f_roc_t")
        if chk('f_stoch', "BUY: %K > %D and not overbought. SELL: %K < %D and not oversold."):
            c1, c2, c3 = st.columns(3)
            config['f_stoch_k'] = c1.number_input("%K", 1, 100, 14, key="f_sto_k")
            config['f_stoch_smooth'] = c2.number_input("Smooth", 1, 20, 3, key="f_sto_s")
            config['f_stoch_d'] = c3.number_input("%D", 1, 20, 3, key="f_sto_d")
            config['f_stoch_ob'] = c1.number_input("OB", 50.0, 100.0, 80.0, key="f_sto_ob")
            config['f_stoch_os'] = c2.number_input("OS", 0.0, 50.0, 20.0, key="f_sto_os")
        if chk('f_candle', "Signal candle body ≥ X% of its range and in the signal direction."):
            config['f_candle_body_pct'] = st.number_input("Min body % of range", 0.0, 100.0, 60.0, key="f_cdl")

        n_on = len(enabled_filter_labels(config))
        st.caption(f"✅ {n_on} filter(s) enabled" if n_on else "No additional filters enabled")


def _default_dhan_credentials():
    """Credentials are read from Streamlit secrets if present (never hard-code tokens)"""
    try:
        return (str(st.secrets.get('DHAN_CLIENT_ID', '')), str(st.secrets.get('DHAN_ACCESS_TOKEN', '')))
    except Exception:
        return '', ''


def render_config_ui():
    """Render configuration sidebar"""
    st.sidebar.header("⚙️ Configuration")
    
    config = {}
    
    # Asset Selection (default NIFTY 50)
    config['asset'] = st.sidebar.selectbox("Asset", list(ASSET_MAPPING.keys()), index=0)
    
    if config['asset'] == 'Custom Ticker':
        config['custom_ticker'] = st.sidebar.text_input("Enter Ticker Symbol", value="KAYNES.NS", help="e.g., KAYNES.NS, RELIANCE.NS, TCS.NS")
    
    # Timeframe (defaults: 1 minute, 7 days)
    config['interval'] = st.sidebar.selectbox("Interval", list(INTERVAL_MAPPING.keys()), index=0)
    config['period'] = st.sidebar.selectbox("Period", list(PERIOD_MAPPING.keys()),
                                            index=list(PERIOD_MAPPING.keys()).index("7 days"))
    _iv = INTERVAL_MAPPING[config['interval']]
    if _iv in INTERVAL_MAX_DAYS:
        st.sidebar.caption(f"ℹ️ yfinance limit for {config['interval']}: last ~{INTERVAL_MAX_DAYS[_iv] + 1} days "
                           f"(longer periods are reduced automatically; extra warm-up candles are "
                           f"downloaded so indicators match TradingView from the first candle).")
    
    config['quantity'] = st.sidebar.number_input("Quantity", min_value=1, value=1)
    
    # ── Signal evaluation ────────────────────────────────────────────────────
    config['signal_on_candle_close'] = st.sidebar.checkbox(
        "🕯️ Signal on Candle Close (no repaint)",
        value=True,
        help="Live trading evaluates the strategy on the last CLOSED candle (TradingView 'once per bar "
             "close'). A crossover is final and each signal candle can trigger only ONE entry. "
             "Untick to evaluate on the forming candle (faster but can repaint)."
    )
    
    # ── Trade Window Settings ────────────────────────────────────────────────
    st.sidebar.subheader("⏰ Trade Window")
    config['use_trade_window'] = st.sidebar.checkbox(
        "Enable Trade Window",
        value=False,
        help="Restricts trading to specific hours. Exits positions and blocks new entries outside this window."
    )
    if config['use_trade_window']:
        config['trade_window_start'] = st.sidebar.time_input(
            "Start Time (IST)", value=datetime.strptime("09:30", "%H:%M").time())
        config['trade_window_end'] = st.sidebar.time_input(
            "End Time (IST)", value=datetime.strptime("15:00", "%H:%M").time())
        st.sidebar.info(f"🕐 Active: {config['trade_window_start'].strftime('%H:%M')} - {config['trade_window_end'].strftime('%H:%M')} IST")
    
    config['trade_direction'] = st.sidebar.selectbox(
        "Trade Direction Filter",
        ["Both (LONG + SHORT)", "LONG Only", "SHORT Only"],
        index=0,
        help="Filters which trade directions the algo will take"
    )
    
    config['reverse_entry'] = st.sidebar.checkbox(
        "Reverse Entry (Flip Signal Direction)",
        value=False,
        help="When enabled, entries are flipped opposite to the raw signal: "
             "a LONG/BUY signal enters SHORT and a SHORT/SELL signal enters LONG."
    )
    
    # ── Brokerage ────────────────────────────────────────────────────────────
    st.sidebar.subheader("💰 Brokerage & Charges")
    config['include_brokerage'] = st.sidebar.checkbox(
        "Include Brokerage & Charges", value=False,
        help="Deducts brokerage from P&L to show Net P&L (backtesting and live trading)"
    )
    if config['include_brokerage']:
        config['brokerage_per_trade'] = st.sidebar.number_input(
            "Brokerage per Trade (₹)", min_value=0.0, value=20.0, step=1.0)
        config['brokerage_percentage'] = st.sidebar.number_input(
            "Or % of Turnover", min_value=0.0, value=0.03, step=0.01, format="%.3f")
        config['brokerage_type'] = st.sidebar.radio(
            "Brokerage Calculation", ["Fixed per Trade", "Percentage of Turnover"], index=0, horizontal=True)
    
    config['prevent_overlapping_trades'] = st.sidebar.checkbox(
        "🚫 Prevent Overlapping Trades", value=True,
        help="Blocks new signals while a position is active. Skipped signals are tracked separately."
    )
    
    config['enable_entry_cooldown'] = st.sidebar.checkbox(
        "⏱️ Enable Entry Cooldown", value=False,
        help="Prevents immediate re-entry after exit (Live Trading only)."
    )
    if config['enable_entry_cooldown']:
        config['entry_cooldown_seconds'] = st.sidebar.number_input(
            "Cooldown Duration (seconds)", min_value=0, max_value=300, value=60, step=5)
        st.sidebar.caption(f"⏱️ Cooldown: {config['entry_cooldown_seconds']}s wait after exit before new entry")
    else:
        config['entry_cooldown_seconds'] = 0
    
    config['enhanced_live_trading'] = st.sidebar.checkbox(
        "🔄 Enhanced Live Trading (TradingView Match)", value=False,
        help="Fetches candlestick data at round time intervals; indicators from candles, live data for SL/Target."
    )
    if config['enhanced_live_trading']:
        st.sidebar.info("📊 Enhanced Mode Active:\n- Indicators calculated from round-time candles\n"
                        "- Live data used for SL/Target checks")
    
    config['show_last_candle'] = st.sidebar.checkbox(
        "📊 Show Last Candle Details", value=False,
        help="Display the last candle with all calculated indicator values in Live Trading tab"
    )
    
    # ── Strategy ─────────────────────────────────────────────────────────────
    st.sidebar.subheader("📊 Strategy")
    config['strategy'] = st.sidebar.selectbox("Strategy Type", STRATEGY_LIST, index=0)
    st.sidebar.caption(f"⚡ Fires: {STRATEGY_SIGNAL_TYPE.get(config['strategy'], '')}")
    
    if config['strategy'] == 'EMA Crossover':
        config['ema_fast'] = st.sidebar.number_input("EMA Fast Period", min_value=1, value=9)
        config['ema_slow'] = st.sidebar.number_input("EMA Slow Period", min_value=1, value=20)
        config['ema_min_angle'] = st.sidebar.number_input("Min Angle (ABSOLUTE)", min_value=0.0, value=30.0, step=0.1)
        config['ema_entry_filter'] = st.sidebar.selectbox("Entry Filter", EMA_ENTRY_FILTERS, index=0)
        if config['ema_entry_filter'] == 'Custom Candle (Points)':
            config['ema_custom_candle_points'] = st.sidebar.number_input("Min Candle Points", min_value=1, value=5)
        elif config['ema_entry_filter'] == 'ATR-based Candle':
            config['ema_atr_multiplier'] = st.sidebar.number_input("ATR Multiplier", min_value=0.1, value=0.3, step=0.1)
        config['ema_use_adx'] = st.sidebar.checkbox("Use ADX Filter", value=True)
        if config['ema_use_adx']:
            config['ema_adx_threshold'] = st.sidebar.number_input("ADX Threshold", min_value=1, value=20)
            config['adx_period'] = st.sidebar.number_input("ADX Period", min_value=1, value=14)
    
    elif config['strategy'] == 'Price Crosses Threshold':
        config['price_threshold'] = st.sidebar.number_input("Price Threshold", min_value=0.0, value=25000.0)
        config['price_cross_type'] = st.sidebar.selectbox("Cross Type", ["Above Threshold", "Below Threshold"])
        config['price_cross_position'] = st.sidebar.selectbox("Position Type", ["LONG", "SHORT"])
    
    elif config['strategy'] == 'Percentage Change':
        config['pct_change_threshold'] = st.sidebar.number_input("% Change Threshold", min_value=0.001, value=2.0, step=0.001, format="%.3f")
        config['pct_change_type'] = st.sidebar.selectbox("Change Type", ["Positive % (Price Up)", "Negative % (Price Down)"])
        config['pct_change_position'] = st.sidebar.selectbox("Position Type", ["LONG", "SHORT"])
    
    elif config['strategy'] == 'SuperTrend AI':
        st.sidebar.markdown("**SuperTrend AI Parameters**")
        config['supertrend_atr_period'] = st.sidebar.number_input("ATR Period", min_value=1, value=10)
        config['supertrend_multiplier'] = st.sidebar.number_input("Multiplier", min_value=0.5, value=3.0, step=0.5)
        config['supertrend_adx_threshold'] = st.sidebar.number_input("ADX Threshold", min_value=0, value=25)
        config['supertrend_volume_mult'] = st.sidebar.number_input("Volume Multiplier", min_value=0.0, value=1.5, step=0.1)
        st.sidebar.info("📊 SuperTrend flip + ADX + Volume (volume skipped for indices)")
    
    elif config['strategy'] == 'VWAP + Volume Spike':
        st.sidebar.markdown("**VWAP + Volume Spike Parameters**")
        config['vwap_volume_mult'] = st.sidebar.number_input("Volume Spike Multiplier", min_value=1.0, value=2.0, step=0.1)
        config['vwap_distance_pct'] = st.sidebar.number_input("Max Distance from VWAP (%)", min_value=0.1, value=0.3, step=0.1)
        config['vwap_rsi_ob'] = st.sidebar.number_input("RSI Overbought", min_value=60, max_value=100, value=70)
        config['vwap_rsi_os'] = st.sidebar.number_input("RSI Oversold", min_value=0, max_value=40, value=30)
        st.sidebar.warning("Needs real volume – indices (^NSEI, ^NSEBANK) have no volume on yfinance.")
    
    elif config['strategy'] == 'Bollinger Squeeze Breakout':
        st.sidebar.markdown("**Bollinger Squeeze Breakout Parameters**")
        config['bb_squeeze_period'] = st.sidebar.number_input("BB Period", min_value=5, value=20)
        config['bb_squeeze_std'] = st.sidebar.number_input("BB Std Dev", min_value=1.0, value=2.0, step=0.1)
        config['bb_squeeze_threshold'] = st.sidebar.number_input("Squeeze Threshold (bandwidth ratio)", min_value=0.001, value=0.02, step=0.001, format="%.3f")
        config['bb_squeeze_volume_mult'] = st.sidebar.number_input("Breakout Volume Mult", min_value=1.0, value=1.8, step=0.1)
    
    elif config['strategy'] == 'Elliott Waves + Ratio Charts':
        config['elliott_wave_lookback'] = st.sidebar.number_input("Wave Lookback Period", min_value=20, value=50)
    
    elif config['strategy'] == 'Opening Range Breakout (ORB)':
        config['orb_minutes'] = st.sidebar.number_input("Opening Range Duration (minutes)", min_value=1, max_value=240, value=15)
        config['orb_session_start'] = st.sidebar.text_input("Session Start (IST)", value="09:15")
        config['orb_breakout_buffer'] = st.sidebar.number_input("Breakout Buffer (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    
    elif config['strategy'] == 'Pivot Point Reversal':
        config['pivot_lookback'] = st.sidebar.number_input("Lookback Period", min_value=12, max_value=500, value=24)
    
    elif config['strategy'] == 'Ichimoku Cloud':
        st.sidebar.info("☁️ Tenkan/Kijun crossovers with cloud confirmation (9/26/52, displacement 26).")
    
    elif config['strategy'] == 'Volume Breakout':
        config['volume_multiplier'] = st.sidebar.number_input("Volume Multiplier", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
        config['volume_price_threshold'] = st.sidebar.number_input("Min Price Change (%)", min_value=0.01, max_value=5.0, value=0.5, step=0.05)
        st.sidebar.warning("Needs real volume – indices have no volume on yfinance.")
    
    elif config['strategy'] == 'Gap Trading Strategy':
        config['gap_min_percent'] = st.sidebar.number_input("Minimum Gap (%)", min_value=0.05, max_value=5.0, value=0.5, step=0.05)
        config['gap_max_percent'] = st.sidebar.number_input("Maximum Gap (%)", min_value=0.5, max_value=15.0, value=3.0, step=0.5)
    
    elif config['strategy'] == 'Mean Reversion with Bollinger Bands':
        config['bb_period'] = st.sidebar.number_input("Bollinger Period", min_value=5, max_value=100, value=20)
        config['bb_std'] = st.sidebar.number_input("Standard Deviations", min_value=1.0, max_value=4.0, value=2.0, step=0.5)
        config['mr_rsi_oversold'] = st.sidebar.number_input("RSI Oversold", min_value=5, max_value=50, value=30)
        config['mr_rsi_overbought'] = st.sidebar.number_input("RSI Overbought", min_value=50, max_value=95, value=70)
    
    elif config['strategy'] == 'Momentum Breakout with ADX':
        config['momentum_adx_threshold'] = st.sidebar.number_input("ADX Threshold", min_value=5, max_value=60, value=25)
        config['momentum_lookback'] = st.sidebar.number_input("Breakout Lookback", min_value=5, max_value=200, value=20)
        config['momentum_volume_ratio'] = st.sidebar.number_input("Volume Ratio", min_value=0.0, max_value=5.0, value=1.5, step=0.5)
    
    elif config['strategy'] == 'Support Resistance Bounce':
        config['sr_lookback'] = st.sidebar.number_input("Lookback Period", min_value=50, max_value=500, value=100)
        config['sr_tolerance'] = st.sidebar.number_input("Level Tolerance (%)", min_value=0.05, max_value=2.0, value=0.2, step=0.05)
        config['sr_min_touches'] = st.sidebar.number_input("Min Level Touches", min_value=2, max_value=10, value=3)
    
    elif config['strategy'] == 'Custom Strategy':
        render_custom_strategy_ui(config)
    
    # ── Additional entry filters ─────────────────────────────────────────────
    render_entry_filters_ui(config)
    
    # ── Swing detection (used by swing SL/Target, SMC filters) ───────────────
    st.sidebar.subheader("📐 Swing Detection")
    c1, c2 = st.sidebar.columns(2)
    config['swing_left'] = c1.number_input("Left bars", min_value=1, max_value=50, value=3,
                                           help="Pivot must be the extreme of N bars on the left")
    config['swing_right'] = c2.number_input("Right bars", min_value=1, max_value=50, value=3,
                                            help="Confirmation bars on the right (swing known only after these close)")
    config['swing_buffer_points'] = st.sidebar.number_input(
        "Swing/Candle SL-Target buffer (points)", min_value=0.0, value=0.0, step=0.5,
        help="SL placed this many points beyond the level; target exits this many points before the level")
    
    # ── Stop Loss ────────────────────────────────────────────────────────────
    st.sidebar.subheader("🛡️ Stop Loss")
    config['sl_type'] = st.sidebar.selectbox("SL Type", SL_TYPES, index=0)
    sl_type = config['sl_type']
    
    if sl_type in SL_LEVEL_TOKENS:
        st.sidebar.caption("SL = actual candle/swing level (trailing variants move it only in your favour).")
        config['sl_points'] = st.sidebar.number_input(
            "Fallback SL Points", min_value=1, value=10,
            help="Used ONLY if no valid candle/swing level exists on the correct side of price")
    elif sl_type in ('Custom Points', 'ATR-based', 'Trailing SL (Points)', 'Cost-to-Cost + N Points Trailing SL',
                     'Break-even After 50% Target', 'Trailing Profit (Rupees)', 'Trailing Loss (Rupees)',
                     'Volatility-Adjusted Trailing SL'):
        config['sl_points'] = st.sidebar.number_input("SL Points", min_value=1, value=10)
    
    if sl_type == 'P&L Based (Rupees)':
        config['sl_rupees'] = st.sidebar.number_input("SL Rupees", min_value=1, value=100)
    if sl_type in ('Trailing Profit (Rupees)', 'Trailing Loss (Rupees)'):
        config['sl_trail_rupees'] = st.sidebar.number_input("Trail Rupees", min_value=1, value=50)
    if sl_type in ('ATR-based', 'Volatility-Adjusted Trailing SL'):
        config['sl_atr_multiplier'] = st.sidebar.number_input("SL ATR Multiplier", min_value=0.1, value=1.5, step=0.1)
    if sl_type == 'Cost-to-Cost + N Points Trailing SL':
        config['ctc_trigger_points'] = st.sidebar.number_input("Trigger Points (K)", min_value=1, value=3)
        config['ctc_offset_points'] = st.sidebar.number_input("Offset Points (N)", min_value=1, value=2)
    
    # ── Target ───────────────────────────────────────────────────────────────
    st.sidebar.subheader("🎯 Target")
    config['target_type'] = st.sidebar.selectbox("Target Type", TARGET_TYPES, index=0)
    tt = config['target_type']
    
    if tt in TARGET_LEVEL_TOKENS:
        st.sidebar.caption("Target = actual candle/swing level beyond price "
                           "(trailing variants push it further as new levels form).")
        config['target_points'] = st.sidebar.number_input(
            "Fallback Target Points", min_value=1, value=20,
            help="Used ONLY if no valid candle/swing level exists beyond price")
    elif tt in ('Custom Points', 'Trailing Target (Points)', 'Trailing Target + Signal Based',
                '50% Exit at Target (Partial)'):
        config['target_points'] = st.sidebar.number_input("Target Points", min_value=1, value=20)
    if tt == 'P&L Based (Rupees)':
        config['target_rupees'] = st.sidebar.number_input("Target Rupees", min_value=1, value=200)
    if tt == 'Risk-Reward Based':
        config['risk_reward_ratio'] = st.sidebar.number_input("Risk:Reward Ratio", min_value=0.1, value=2.0, step=0.1)
    if tt == 'ATR-based':
        config['target_atr_multiplier'] = st.sidebar.number_input("Target ATR Multiplier", min_value=0.1, value=2.0, step=0.1)
    if tt == 'Dynamic Trailing SL+Target (Lock Profits)':
        st.sidebar.info("Both SL and Target trail together as price moves favorably")
        config['dynamic_trail_sl_points'] = st.sidebar.number_input("SL Distance (Points)", min_value=1, value=10)
        config['dynamic_trail_target_points'] = st.sidebar.number_input("Target Distance (Points)", min_value=1, value=200)
    
    # ── Dhan Broker ──────────────────────────────────────────────────────────
    render_dhan_ui(config)
    
    return config


def render_custom_strategy_ui(config):
    """Custom Strategy Builder (multi-indicator) – unchanged behaviour"""
    st.sidebar.markdown("**🛠️ Custom Strategy Builder (Multi-Indicator)**")

    if 'custom_indicator_conditions' not in st.session_state:
        st.session_state['custom_indicator_conditions'] = [{}]
    conditions = st.session_state['custom_indicator_conditions']

    if len(conditions) > 1:
        config['custom_combine_mode'] = st.sidebar.radio(
            "Combine Conditions With", ["AND (all must be true)", "OR (any one true)"], index=0)
    else:
        config['custom_combine_mode'] = "AND (all must be true)"

    col_add, col_clr = st.sidebar.columns(2)
    with col_add:
        if st.button("➕ Add Condition", key="cust_add"):
            st.session_state['custom_indicator_conditions'].append({})
            st.rerun()
    with col_clr:
        if st.button("🗑️ Clear All", key="cust_clr"):
            st.session_state['custom_indicator_conditions'] = [{}]
            st.rerun()

    STRATEGY_TYPE_OPTS = ["Price Crosses Indicator", "Price Pullback from Indicator",
                          "Indicator Crosses Level", "Indicator Crossover"]
    PRICE_INDICATOR_OPTS = ["EMA", "SMA", "BB Upper", "BB Lower", "BB Middle"]
    PULLBACK_INDICATOR_OPTS = ["EMA", "SMA", "BB Upper", "BB Lower"]
    LEVEL_INDICATOR_OPTS = ["RSI", "MACD", "MACD Histogram", "ADX", "Volume", "BB %B",
                            "ATR (Volatility)", "Historical Volatility", "Std Dev (Volatility)"]
    CROSSOVER_OPTS = ["Fast EMA × Slow EMA", "Fast SMA × Slow SMA", "MACD × Signal",
                      "Price × EMA", "Price × SMA", "RSI Crossover (Overbought/Oversold)"]

    def _idx(opts, val, default=0):
        return opts.index(val) if val in opts else default

    rendered_conditions = []
    for i, cond in enumerate(conditions):
        st.sidebar.markdown(f"---\n**Condition {i+1}**")
        if len(conditions) > 1:
            if st.sidebar.button(f"🗑️ Delete #{i+1}", key=f"del_cond_{i}"):
                st.session_state['custom_indicator_conditions'].pop(i)
                st.rerun()

        c = {}
        c['strategy_type'] = st.sidebar.selectbox(f"Type #{i+1}", STRATEGY_TYPE_OPTS,
            index=_idx(STRATEGY_TYPE_OPTS, cond.get('strategy_type')), key=f"cst_{i}")

        if c['strategy_type'] == "Price Crosses Indicator":
            c['indicator'] = st.sidebar.selectbox(f"Indicator #{i+1}", PRICE_INDICATOR_OPTS,
                index=_idx(PRICE_INDICATOR_OPTS, cond.get('indicator')), key=f"ci_{i}")
            if c['indicator'] in ['EMA', 'SMA']:
                c['period'] = st.sidebar.number_input(f"Period #{i+1}", min_value=1, value=int(cond.get('period', 20)), key=f"cp_{i}")
            elif 'BB' in c['indicator']:
                c['bb_period'] = st.sidebar.number_input(f"BB Period #{i+1}", min_value=1, value=int(cond.get('bb_period', 20)), key=f"cbp_{i}")
                c['bb_std'] = st.sidebar.number_input(f"BB Std #{i+1}", min_value=0.1, value=float(cond.get('bb_std', 2.0)), step=0.1, key=f"cbs_{i}")
            c['cross_type'] = st.sidebar.selectbox(f"Cross #{i+1}", ["Above Indicator", "Below Indicator"],
                index=0 if cond.get('cross_type', 'Above Indicator') == 'Above Indicator' else 1, key=f"cct_{i}")
            c['position_type'] = st.sidebar.selectbox(f"Position #{i+1}", ["LONG", "SHORT"],
                index=0 if cond.get('position_type', 'LONG') == 'LONG' else 1, key=f"cpt_{i}")

        elif c['strategy_type'] == "Price Pullback from Indicator":
            c['indicator'] = st.sidebar.selectbox(f"Indicator #{i+1}", PULLBACK_INDICATOR_OPTS,
                index=_idx(PULLBACK_INDICATOR_OPTS, cond.get('indicator')), key=f"ci_{i}")
            if c['indicator'] in ['EMA', 'SMA']:
                c['period'] = st.sidebar.number_input(f"Period #{i+1}", min_value=1, value=int(cond.get('period', 20)), key=f"cp_{i}")
            elif 'BB' in c['indicator']:
                c['bb_period'] = st.sidebar.number_input(f"BB Period #{i+1}", min_value=1, value=int(cond.get('bb_period', 20)), key=f"cbp_{i}")
                c['bb_std'] = st.sidebar.number_input(f"BB Std #{i+1}", min_value=0.1, value=float(cond.get('bb_std', 2.0)), step=0.1, key=f"cbs_{i}")
            c['pullback_points'] = st.sidebar.number_input(f"Pullback Pts #{i+1}", min_value=0.01,
                value=float(cond.get('pullback_points', 10)), step=0.01, key=f"cpp_{i}")
            c['pullback_side'] = st.sidebar.selectbox(f"Approach #{i+1}", ["Approach from Above", "Approach from Below"],
                index=0 if cond.get('pullback_side', 'Approach from Above') == 'Approach from Above' else 1, key=f"cps_{i}")
            c['position_type'] = st.sidebar.selectbox(f"Position #{i+1}", ["LONG", "SHORT"],
                index=0 if cond.get('position_type', 'LONG') == 'LONG' else 1, key=f"cpt_{i}")

        elif c['strategy_type'] == "Indicator Crosses Level":
            c['indicator'] = st.sidebar.selectbox(f"Indicator #{i+1}", LEVEL_INDICATOR_OPTS,
                index=_idx(LEVEL_INDICATOR_OPTS, cond.get('indicator')), key=f"ci_{i}")
            ind = c['indicator']
            if ind == 'RSI':
                c['rsi_period'] = st.sidebar.number_input(f"RSI Period #{i+1}", min_value=1, value=int(cond.get('rsi_period', 14)), key=f"crsi_{i}")
                c['level'] = st.sidebar.number_input(f"Level #{i+1}", min_value=0.0, max_value=100.0, value=float(cond.get('level', 50.0)), key=f"clv_{i}")
            elif ind in ['MACD', 'MACD Histogram']:
                c['level'] = st.sidebar.number_input(f"Level #{i+1}", value=float(cond.get('level', 0.0)), key=f"clv_{i}")
            elif ind == 'ADX':
                c['level'] = st.sidebar.number_input(f"ADX Level #{i+1}", min_value=0.0, value=float(cond.get('level', 25.0)), key=f"clv_{i}")
            elif ind == 'Volume':
                c['volume_ma_period'] = st.sidebar.number_input(f"Vol MA Period #{i+1}", min_value=1, value=int(cond.get('volume_ma_period', 20)), key=f"cvmp_{i}")
                c['volume_multiplier'] = st.sidebar.number_input(f"Vol Mult #{i+1}", min_value=0.1, value=float(cond.get('volume_multiplier', 1.5)), step=0.1, key=f"cvm_{i}")
            elif ind == 'BB %B':
                c['bb_period'] = st.sidebar.number_input(f"BB Period #{i+1}", min_value=1, value=int(cond.get('bb_period', 20)), key=f"cbp_{i}")
                c['bb_std'] = st.sidebar.number_input(f"BB Std #{i+1}", min_value=0.1, value=float(cond.get('bb_std', 2.0)), step=0.1, key=f"cbs_{i}")
                c['level'] = st.sidebar.number_input(f"%B Level #{i+1} (0–100)", min_value=0.0, max_value=100.0, value=float(cond.get('level', 80.0)), key=f"clv_{i}")
            elif ind == 'ATR (Volatility)':
                c['atr_period'] = st.sidebar.number_input(f"ATR Period #{i+1}", min_value=1, value=int(cond.get('atr_period', 14)), key=f"catr_{i}")
                c['level'] = st.sidebar.number_input(f"ATR Level #{i+1}", min_value=0.0, value=float(cond.get('level', 10.0)), step=0.5, key=f"clv_{i}")
            elif ind == 'Historical Volatility':
                c['hv_period'] = st.sidebar.number_input(f"HV Period #{i+1} (days)", min_value=5, value=int(cond.get('hv_period', 20)), key=f"chv_{i}")
                c['level'] = st.sidebar.number_input(f"HV Level #{i+1} (%)", min_value=0.0, value=float(cond.get('level', 20.0)), step=1.0, key=f"clv_{i}")
            elif ind == 'Std Dev (Volatility)':
                c['stddev_period'] = st.sidebar.number_input(f"StdDev Period #{i+1}", min_value=2, value=int(cond.get('stddev_period', 20)), key=f"csd_{i}")
                c['level'] = st.sidebar.number_input(f"StdDev Level #{i+1}", min_value=0.0, value=float(cond.get('level', 5.0)), step=0.5, key=f"clv_{i}")
            c['cross_type'] = st.sidebar.selectbox(f"Cross #{i+1}", ["Above Level", "Below Level"],
                index=0 if cond.get('cross_type', 'Above Level') == 'Above Level' else 1, key=f"cct_{i}")
            c['position_type'] = st.sidebar.selectbox(f"Position #{i+1}", ["LONG", "SHORT"],
                index=0 if cond.get('position_type', 'LONG') == 'LONG' else 1, key=f"cpt_{i}")

        elif c['strategy_type'] == "Indicator Crossover":
            c['crossover_type'] = st.sidebar.selectbox(f"Crossover #{i+1}", CROSSOVER_OPTS,
                index=_idx(CROSSOVER_OPTS, cond.get('crossover_type')), key=f"cco_{i}")
            if c['crossover_type'] == "Fast EMA × Slow EMA":
                c['fast_ema'] = st.sidebar.number_input(f"Fast EMA #{i+1}", min_value=1, value=int(cond.get('fast_ema', 9)), key=f"cfe_{i}")
                c['slow_ema'] = st.sidebar.number_input(f"Slow EMA #{i+1}", min_value=1, value=int(cond.get('slow_ema', 21)), key=f"cse_{i}")
            elif c['crossover_type'] == "Fast SMA × Slow SMA":
                c['fast_sma'] = st.sidebar.number_input(f"Fast SMA #{i+1}", min_value=1, value=int(cond.get('fast_sma', 20)), key=f"cfs_{i}")
                c['slow_sma'] = st.sidebar.number_input(f"Slow SMA #{i+1}", min_value=1, value=int(cond.get('slow_sma', 50)), key=f"css_{i}")
            elif c['crossover_type'] in ["Price × EMA", "Price × SMA"]:
                c['ma_period'] = st.sidebar.number_input(f"MA Period #{i+1}", min_value=1, value=int(cond.get('ma_period', 50)), key=f"cmap_{i}")
            elif c['crossover_type'] == "RSI Crossover (Overbought/Oversold)":
                c['rsi_period'] = st.sidebar.number_input(f"RSI Period #{i+1}", min_value=1, value=int(cond.get('rsi_period', 14)), key=f"crsi_{i}")
                c['rsi_ob'] = st.sidebar.number_input(f"Overbought #{i+1}", min_value=50.0, max_value=100.0, value=float(cond.get('rsi_ob', 70.0)), key=f"crob_{i}")
                c['rsi_os'] = st.sidebar.number_input(f"Oversold #{i+1}", min_value=0.0, max_value=50.0, value=float(cond.get('rsi_os', 30.0)), key=f"cros_{i}")
            c['cross_type'] = st.sidebar.selectbox(f"Direction #{i+1}", ["Bullish Crossover", "Bearish Crossover"],
                index=0 if cond.get('cross_type', 'Bullish Crossover') == 'Bullish Crossover' else 1, key=f"cct_{i}")
            c['position_type'] = st.sidebar.selectbox(f"Position #{i+1}", ["LONG", "SHORT"],
                index=0 if cond.get('position_type', 'LONG') == 'LONG' else 1, key=f"cpt_{i}")

        rendered_conditions.append(c)

    st.session_state['custom_indicator_conditions'] = rendered_conditions
    config['custom_conditions'] = rendered_conditions
    if rendered_conditions:
        first = rendered_conditions[0]
        config['custom_strategy_type'] = first.get('strategy_type', 'Price Crosses Indicator')
        config['custom_position_type'] = first.get('position_type', 'LONG')
        config['custom_indicator'] = first.get('indicator', 'EMA')
        config['custom_cross_type'] = first.get('cross_type', 'Above Indicator')
        config['custom_indicator_period'] = first.get('period', 20)


def render_dhan_ui(config):
    """Dhan broker sidebar section"""
    st.sidebar.subheader("🏦 Dhan Broker (Optional)")
    config['dhan_enabled'] = st.sidebar.checkbox("Enable Dhan Broker", value=False)
    if not config['dhan_enabled']:
        return

    def_id, def_token = _default_dhan_credentials()
    config['dhan_client_id'] = st.sidebar.text_input("Client ID", value=def_id)
    config['dhan_access_token'] = st.sidebar.text_input(
        "Access Token", type="password", value=def_token,
        help="Tip: store it in .streamlit/secrets.toml as DHAN_ACCESS_TOKEN instead of typing it")

    config['dhan_is_options'] = st.sidebar.checkbox("Is Options", value=False)

    if config['dhan_is_options']:
        config['dhan_ce_security_id'] = st.sidebar.text_input("CE Security ID", value="48228")
        config['dhan_pe_security_id'] = st.sidebar.text_input("PE Security ID", value="48229")
        config['dhan_strike_price'] = st.sidebar.number_input("Strike Price", min_value=0, value=25000)
        config['dhan_expiry_date'] = st.sidebar.date_input("Expiry Date", value=datetime.now().date())
        config['dhan_quantity'] = st.sidebar.number_input("Dhan Quantity", min_value=1, value=65)
        st.sidebar.info("Product: INTRA (options)")
    else:
        config['dhan_trading_type'] = st.sidebar.selectbox(
            "Trading Type", ["Intraday", "Delivery (CNC)"],
            help="Intraday = MIS/INTRA  |  Delivery = CNC positional")
        config['dhan_security_id'] = st.sidebar.text_input("Security ID", value="12092")
        config['dhan_exchange'] = st.sidebar.selectbox("Exchange", ["NSE", "BSE"], index=0)
        config['dhan_quantity'] = st.sidebar.number_input("Dhan Order Quantity", min_value=1, value=10)
        st.sidebar.info(f"Product: {'CNC' if config['dhan_trading_type'] == 'Delivery (CNC)' else 'INTRA'}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Order Type Configuration**")
    config['dhan_entry_order_type'] = st.sidebar.selectbox(
        "Entry Order Type", ["Market Order", "Limit Order"], index=1,
        help="Entry: Market = Immediate | Limit = At specified price")
    config['dhan_exit_order_type'] = st.sidebar.selectbox(
        "Exit Order Type", ["Market Order", "Limit Order"], index=1,
        help="Exit: Market = Immediate | Limit = At specified price")
    entry_type = "MARKET" if config['dhan_entry_order_type'] == "Market Order" else "LIMIT"
    exit_type = "MARKET" if config['dhan_exit_order_type'] == "Market Order" else "LIMIT"
    st.sidebar.caption(f"📋 Configuration: Entry {entry_type} | Exit {exit_type}")
    config['dhan_order_type'] = config['dhan_entry_order_type']

    config['broker_use_own_sl'] = st.sidebar.checkbox(
        "🎯 Use Broker SL/Target (Bracket Order)", value=False,
        help="Sends a Bracket Order (BO) to Dhan with embedded SL and Target (distances in points).")
    if config['broker_use_own_sl']:
        config['broker_sl_points'] = st.sidebar.number_input(
            "SL Points (boStopLossValue)", min_value=0.5, value=50.0, step=0.5)
        config['broker_target_points'] = st.sidebar.number_input(
            "Target Points (boProfitValue)", min_value=0.5, value=100.0, step=0.5)
        config['broker_trailing_jump'] = st.sidebar.number_input(
            "Trail SL Jump (0 = off)", min_value=0.0, value=0.0, step=0.5)
        st.sidebar.info(
            f"📌 BO: Entry ± {config['broker_sl_points']}pts SL | +{config['broker_target_points']}pts Target"
            + (f" | Trail: {config['broker_trailing_jump']}pts" if config['broker_trailing_jump'] > 0 else ""))

    # ── Multi-Account Trading ────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔀 Multi-Account Trading**")
    if 'multi_accounts' not in st.session_state:
        st.session_state['multi_accounts'] = []
    if st.session_state['multi_accounts']:
        st.sidebar.write(f"**Configured Accounts:** {len(st.session_state['multi_accounts'])}")
        for i, acc in enumerate(st.session_state['multi_accounts']):
            col_acc1, col_acc2 = st.sidebar.columns([3, 1])
            col_acc1.caption(f"{i+1}. Client: {acc['client_id'][:8]}...")
            if col_acc2.button("❌", key=f"del_acc_{i}"):
                st.session_state['multi_accounts'].pop(i)
                st.rerun()
    with st.sidebar.expander("➕ Add Account"):
        new_client_id = st.text_input("Client ID", key="new_client_id")
        new_token = st.text_input("Access Token", type="password", key="new_token")
        if st.button("Add Account", key="add_account_btn"):
            if new_client_id and new_token:
                st.session_state['multi_accounts'].append({'client_id': new_client_id, 'access_token': new_token})
                st.success(f"Account added! Total: {len(st.session_state['multi_accounts'])}")
                st.rerun()
            else:
                st.error("Please provide both Client ID and Token")
    config['multi_accounts'] = st.session_state['multi_accounts']

    # ── Multi-Strike Options ─────────────────────────────────────────────────
    if config['dhan_is_options']:
        st.sidebar.markdown("**📊 Multi-Strike Options**")
        config['multi_strike_enabled'] = st.sidebar.checkbox(
            "Enable Multi-Strike Orders", value=False,
            help="Place orders on multiple strike prices simultaneously")
        if config['multi_strike_enabled']:
            st.session_state.setdefault('multi_strikes_ce', [])
            st.session_state.setdefault('multi_strikes_pe', [])
            strike_type = st.sidebar.radio("Strike Type", ["CE (Call)", "PE (Put)"], horizontal=True)
            side = 'ce' if strike_type == "CE (Call)" else 'pe'
            lst = st.session_state[f'multi_strikes_{side}']
            if lst:
                st.sidebar.write(f"**{side.upper()} Strikes:** {len(lst)}")
                for i, sec_id in enumerate(lst):
                    col_s1, col_s2 = st.sidebar.columns([3, 1])
                    col_s1.caption(f"{i+1}. {sec_id}")
                    if col_s2.button("❌", key=f"del_{side}_{i}"):
                        lst.pop(i)
                        st.rerun()
            with st.sidebar.expander(f"➕ Add {side.upper()} Strike"):
                new_id = st.text_input(f"{side.upper()} Security ID", key=f"new_{side}_id")
                if st.button(f"Add {side.upper()}", key=f"add_{side}_btn") and new_id:
                    lst.append(new_id)
                    st.success(f"{side.upper()} Strike added! Total: {len(lst)}")
                    st.rerun()
            config['multi_strikes_ce'] = st.session_state.get('multi_strikes_ce', [])
            config['multi_strikes_pe'] = st.session_state.get('multi_strikes_pe', [])

    st.sidebar.markdown("---")
    config['clear_positions_before_entry'] = st.sidebar.checkbox(
        "🧹 Clear All Positions Before New Entry", value=False,
        help="Cancels all pending orders and closes all open positions with market orders before a new entry.")

# ================================
# CONFIG SUMMARY (Live Trading tab)
# ================================

CONFIG_GROUPS = [
    ('📌 General', lambda k: k in ('asset', 'custom_ticker', 'interval', 'period', 'quantity', 'signal_on_candle_close',
                                   'trade_direction', 'reverse_entry', 'prevent_overlapping_trades',
                                   'enable_entry_cooldown', 'entry_cooldown_seconds', 'enhanced_live_trading',
                                   'show_last_candle', 'use_trade_window', 'trade_window_start', 'trade_window_end',
                                   'include_brokerage', 'brokerage_per_trade', 'brokerage_percentage', 'brokerage_type')),
    ('🧰 Entry Filters', lambda k: k.startswith('f_')),
    ('📐 Swing Detection', lambda k: k.startswith('swing_')),
    ('🛡️ Stop Loss', lambda k: k.startswith('sl_') or k.startswith('ctc_')),
    ('🎯 Target', lambda k: k.startswith('target_') or k.startswith('dynamic_') or k == 'risk_reward_ratio'),
    ('🏦 Broker', lambda k: k.startswith('dhan_') or k.startswith('broker_') or k.startswith('multi_')
                            or k == 'clear_positions_before_entry'),
]


def _fmt_cfg_value(k, v):
    if 'token' in k.lower():
        return ('••••' + str(v)[-4:]) if v else '(not set)'
    if isinstance(v, bool):
        return '✅ ON' if v else '❌ OFF'
    if k == 'multi_accounts':
        return f"{len(v)} account(s)"
    if hasattr(v, 'strftime'):
        try:
            return v.strftime('%H:%M') if not hasattr(v, 'year') else v.strftime('%Y-%m-%d')
        except Exception:
            return str(v)
    if isinstance(v, (list, dict)):
        return str(v)
    return str(v)


def render_config_summary(cfg):
    """Show EVERY sidebar setting currently in effect, grouped"""
    groups = {name: [] for name, _ in CONFIG_GROUPS}
    groups['📊 Strategy'] = []
    filter_param_rows = []
    enabled_filters = [k for k in ENTRY_FILTERS if cfg.get(k, False)]

    for k, v in cfg.items():
        if k.startswith('_'):
            continue
        placed = False
        for name, pred in CONFIG_GROUPS:
            if pred(k):
                if name == '🧰 Entry Filters':
                    if k in ENTRY_FILTERS:
                        groups[name].append({'Setting': ENTRY_FILTERS[k][0], 'Value': _fmt_cfg_value(k, v)})
                    else:
                        # parameters exist only for filters that are switched on
                        filter_param_rows.append({'Setting': f"   ↳ {k}", 'Value': _fmt_cfg_value(k, v)})
                else:
                    groups[name].append({'Setting': k, 'Value': _fmt_cfg_value(k, v)})
                placed = True
                break
        if not placed:
            groups['📊 Strategy'].append({'Setting': k, 'Value': _fmt_cfg_value(k, v)})

    if enabled_filters:
        st.success("🧰 Enabled entry filters: " + ", ".join(ENTRY_FILTERS[k][0] for k in enabled_filters))
    else:
        st.info("🧰 No additional entry filters enabled")

    order = ['📌 General', '📊 Strategy', '🧰 Entry Filters', '📐 Swing Detection', '🛡️ Stop Loss', '🎯 Target', '🏦 Broker']
    tabs = st.tabs(order)
    for tab, name in zip(tabs, order):
        with tab:
            rows = groups.get(name, [])
            if name == '🧰 Entry Filters':
                rows = [r for r in rows] + filter_param_rows
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                st.caption("Nothing configured in this group")


def _fmt_rupee_cols(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"₹{x:.2f}" if isinstance(x, (int, float, np.floating)) and pd.notna(x) else "—")
    return df


def _fmt_time(x):
    try:
        return x.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return str(x) if x is not None else "—"


def _duration_min(row):
    try:
        return f"{int((pd.Timestamp(row['exit_time']) - pd.Timestamp(row['entry_time'])).total_seconds() / 60)} min"
    except Exception:
        return "—"


def render_backtest_ui(config):
    """Render backtesting interface with EMA plot and IST time filter"""
    st.header("📈 Backtest Results")

    filter_market_hours = st.checkbox(
        "🕐 Filter Same-Day Trades Only (9:15 AM – 3:00 PM IST)",
        value=False,
        help="Shows only trades entered AND exited on the same day between 9:15 AM and 3:00 PM IST."
    )
    
    use_method2 = st.checkbox(
        "🔬 Use Backtesting Method 2 (Realistic Entry)",
        value=False,
        help="Entry price = NEXT candle's OPEN (not the signal candle close)."
    )
    config['use_backtest_method2'] = use_method2

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            ticker = config.get('asset', 'NIFTY 50')
            interval = INTERVAL_MAPPING.get(config.get('interval', '1 minute'), '1m')
            period = PERIOD_MAPPING.get(config.get('period', '7 days'), '7d')
            custom_ticker = config.get('custom_ticker', None)

            df = fetch_data(ticker, interval, period, custom_ticker=custom_ticker)
            if st.session_state.get('data_fetch_note'):
                st.info(st.session_state['data_fetch_note'])

            if df is not None:
                df = calculate_all_indicators(df, config)
                trades, metrics, debug_info, skipped_trades = run_backtest(df, config)
                st.session_state['backtest_results'] = {
                    'trades': trades, 'metrics': metrics, 'debug_info': debug_info,
                    'skipped_trades': skipped_trades, 'df': df,
                }

    if 'backtest_results' not in st.session_state:
        return

    results = st.session_state['backtest_results']
    all_trades = results['trades']
    debug_info = results['debug_info']
    df_chart = results.get('df')

    def _in_hours(t):
        t = t.astimezone(IST) if getattr(t, 'tzinfo', None) else IST.localize(t)
        return (t.hour > 9 or (t.hour == 9 and t.minute >= 15)) and t.hour < 15, t

    if filter_market_hours and all_trades:
        trades = []
        for t in all_trades:
            try:
                e_ok, et = _in_hours(t['entry_time'])
                x_ok, xt = _in_hours(t['exit_time'])
                if e_ok and x_ok and et.date() == xt.date():
                    trades.append(t)
            except Exception:
                pass
        st.info(f"🕐 Same-day filter (9:15 AM–3:00 PM IST): {len(trades)} / {len(all_trades)} trades shown")
    else:
        trades = all_trades

    metrics = compute_trade_metrics(trades)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", metrics['total_trades'])
    col2.metric("Win Rate", f"{metrics['win_rate']:.2f}%")
    col3.metric("Total P&L", f"₹{metrics['total_pnl']:.2f}")
    col4.metric("Avg Trade", f"₹{metrics['avg_pnl']:.2f}")
    if config.get('include_brokerage', False):
        col_b1, col_b2, col_b3, _ = st.columns(4)
        col_b1.metric("Total Brokerage", f"₹{metrics['total_brokerage']:.2f}")
        col_b2.metric("Net P&L", f"₹{metrics['total_net_pnl']:.2f}")
        col_b3.metric("Avg Net P&L", f"₹{metrics['avg_net_pnl']:.2f}")
    col5, col6, col7 = st.columns(3)
    col5.metric("Winning Trades", metrics['winning_trades'])
    col6.metric("Losing Trades", metrics['losing_trades'])
    col7.metric("Max Drawdown", f"₹{metrics['max_drawdown']:.2f}")

    # ── Chart (requested period only, last 300 candles) ──────────────────────
    if df_chart is not None:
        st.subheader("📊 Price Chart with EMA Overlay & Signals")
        plot_src = df_chart[df_chart['In_Period']] if 'In_Period' in df_chart.columns else df_chart
        plot_df = plot_src.tail(300).copy()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=plot_df['Datetime'], open=plot_df['Open'], high=plot_df['High'],
            low=plot_df['Low'], close=plot_df['Close'], name='Price',
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
        fig.add_trace(go.Scatter(x=plot_df['Datetime'], y=plot_df['EMA_Fast'], mode='lines',
                                 name=f"EMA {config.get('ema_fast', 9)}", line=dict(color='#FF9800', width=1.5)))
        fig.add_trace(go.Scatter(x=plot_df['Datetime'], y=plot_df['EMA_Slow'], mode='lines',
                                 name=f"EMA {config.get('ema_slow', 20)}", line=dict(color='#2196F3', width=1.5)))
        if config.get('strategy') == 'Custom Strategy':
            fig.add_trace(go.Scatter(x=plot_df['Datetime'], y=plot_df['BB_Upper'], mode='lines', name='BB Upper',
                                     line=dict(color='#9C27B0', width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=plot_df['Datetime'], y=plot_df['BB_Lower'], mode='lines', name='BB Lower',
                                     line=dict(color='#9C27B0', width=1, dash='dot'),
                                     fill='tonexty', fillcolor='rgba(156,39,176,0.05)'))
        if trades and len(plot_df):
            min_dt = plot_df['Datetime'].min()
            for tr in trades:
                et, xt, ep, xp, pos = tr.get('entry_time'), tr.get('exit_time'), tr.get('entry_price'), tr.get('exit_price'), tr.get('type', 'LONG')
                try:
                    if et is not None and et >= min_dt:
                        fig.add_trace(go.Scatter(
                            x=[et], y=[ep], mode='markers', showlegend=False,
                            marker=dict(symbol='triangle-up' if pos == 'LONG' else 'triangle-down', size=12,
                                        color='#00E676' if pos == 'LONG' else '#FF1744', line=dict(width=1, color='black')),
                            hovertemplate=f"Entry {pos}<br>Price: {ep:.2f}<extra></extra>"))
                    if xt is not None and xt >= min_dt:
                        fig.add_trace(go.Scatter(
                            x=[xt], y=[xp], mode='markers', showlegend=False,
                            marker=dict(symbol='x', size=10, color='#FF6F00', line=dict(width=2, color='black')),
                            hovertemplate=f"Exit ({tr.get('exit_reason')})<br>Price: {xp:.2f}<extra></extra>"))
                except Exception:
                    pass
        fig.update_layout(
            title=f"{config.get('asset','Asset')} — {config.get('interval','')}"
                  + (" [Market Hours Filter ON]" if filter_market_hours else ""),
            xaxis_title='Time', yaxis_title='Price', xaxis_rangeslider_visible=False, height=500,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='x unified', template='plotly_dark')
        st.plotly_chart(fig, width='stretch')

    # ── Trade table ──────────────────────────────────────────────────────────
    if trades:
        st.subheader("✅ Executed Trade History")
        df_trades = pd.DataFrame(trades)
        df_trades['duration'] = df_trades['duration_minutes'].apply(lambda x: f"{int(x)} min" if pd.notna(x) else "—")
        df_trades['angle'] = df_trades['ema_angle_entry'].apply(lambda x: f"{x:.2f}°" if pd.notna(x) else "—")
        _fmt_rupee_cols(df_trades, ['entry_price', 'exit_price', 'highest_price', 'lowest_price', 'initial_sl',
                                    'initial_target', 'sl_price', 'target_price', 'pnl', 'net_pnl', 'brokerage'])
        display_cols = ['entry_time', 'exit_time', 'type', 'duration', 'entry_price', 'exit_price',
                        'initial_sl', 'initial_target', 'sl_price', 'target_price',
                        'highest_price', 'lowest_price', 'pnl', 'net_pnl', 'exit_reason']
        st.dataframe(df_trades[[c for c in display_cols if c in df_trades.columns]], width="stretch")

        with st.expander("📊 Detailed Trade Metrics (EMA, Angles, Differences)"):
            detailed_cols = ['entry_time', 'exit_time', 'type', 'strategy', 'pnl', 'net_pnl', 'brokerage',
                             'ema_fast_period', 'ema_slow_period', 'angle',
                             'ema_fast_entry', 'ema_slow_entry', 'price_fast_ema_diff_entry',
                             'price_slow_ema_diff_entry', 'fast_slow_ema_diff_entry',
                             'ema_fast_exit', 'ema_slow_exit', 'price_fast_ema_diff_exit',
                             'price_slow_ema_diff_exit', 'fast_slow_ema_diff_exit']
            df_detailed = df_trades.copy()
            for col in detailed_cols[10:] + ['ema_fast_entry', 'ema_slow_entry']:
                if col in df_detailed.columns:
                    df_detailed[col] = df_detailed[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float, np.floating)) and pd.notna(x) else "—")
            st.dataframe(df_detailed[[c for c in detailed_cols if c in df_detailed.columns]], width="stretch")

    # ── Skipped/Overlapping Trades ───────────────────────────────────────────
    skipped_trades = results.get('skipped_trades', [])
    if skipped_trades and config.get('prevent_overlapping_trades', True):
        st.subheader("⚠️ Skipped/Overlapping Trades (Not Included in P&L)")
        st.info(f"**{len(skipped_trades)} signals were skipped** because they occurred while a position was active.")
        df_skip_raw = pd.DataFrame(skipped_trades)
        df_skipped = _fmt_rupee_cols(df_skip_raw.copy(), ['entry_price', 'exit_price', 'sl_price', 'target_price',
                                                          'pnl', 'net_pnl', 'brokerage'])
        st.dataframe(df_skipped, width="stretch")
        col_sk1, col_sk2, col_sk3 = st.columns(3)
        col_sk1.metric("Skipped Winning", int((df_skip_raw['pnl'] > 0).sum()))
        col_sk2.metric("Skipped Losing", int((df_skip_raw['pnl'] <= 0).sum()))
        col_sk3.metric("Skipped Total P&L", f"₹{df_skip_raw['pnl'].sum():.2f}")

    if metrics['total_trades'] == 0:
        st.warning("⚠️ No trades generated")

    with st.expander("🔍 Debug Information"):
        st.write(f"- Total Candles (incl. warm-up): {debug_info['total_candles']}")
        st.write(f"- Warm-up candles (indicator history, not traded): {debug_info.get('warmup_candles', 0)}")
        st.write(f"- Candles Analyzed: {debug_info['candles_analyzed']}")
        st.write(f"- Signals Generated: {debug_info['signals_generated']}")
        st.write(f"- Signals Rejected by Entry Filters: {debug_info.get('signals_filtered', 0)}")
        for lbl, cnt in debug_info.get('filter_rejections', {}).items():
            st.write(f"    • {lbl}: {cnt} rejection(s)")
        st.write(f"- Trades Entered: {debug_info['trades_entered']}")
        st.write(f"- Trades Completed: {debug_info['trades_completed']}")
        st.write(f"- Signals Skipped (Overlapping): {debug_info.get('signals_skipped', 0)}")
        if filter_market_hours:
            st.write(f"- Trades after same-day filter: {len(trades)}")


def render_live_trading_ui(config):
    """Render live trading interface with comprehensive information"""
    st.header("🔴 Live Trading")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start Trading", type="primary"):
            st.session_state['trading_active'] = True
            st.session_state['position'] = None
            st.session_state['broker_position'] = None
            st.session_state['live_logs'] = []
            st.session_state['last_exit_time'] = None
            st.session_state['last_signal_type'] = None
            st.session_state['last_signal_candle_time'] = None
            st.session_state['clearing_in_progress'] = False
            st.session_state['indicator_df'] = None
            st.session_state['last_candle_fetch_time'] = None
            st.session_state['live_df'] = None
            st.session_state.setdefault('trade_history', [])
            st.session_state.pop('current_data', None)
            st.session_state['config'] = dict(config)
            
            add_log("🚀 Trading started - all sessions cleared")
            add_log(f"📋 Strategy: {config.get('strategy', 'N/A')} ({STRATEGY_SIGNAL_TYPE.get(config.get('strategy'), '')})")
            add_log(f"📋 Asset: {config.get('asset', 'N/A')} | Interval: {config.get('interval', 'N/A')} | Period: {config.get('period')}")
            add_log(f"📋 SL Type: {config.get('sl_type', 'N/A')} | Target Type: {config.get('target_type', 'N/A')}")
            add_log(f"📋 Quantity: {config.get('quantity', 'N/A')} | Signal on candle close: {config.get('signal_on_candle_close', True)}")
            flts = enabled_filter_labels(config)
            add_log(f"🧰 Entry filters: {', '.join(flts) if flts else 'none'}")
            if config.get('enhanced_live_trading', False):
                add_log("🔄 Enhanced Live Trading Mode ENABLED")
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
            try:
                manual_square_off(config)
            except Exception as e:
                st.error(f"Manual square-off failed: {e}")
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
    
    trading_active = st.session_state.get('trading_active', False)
    if trading_active:
        st.success("🟢 Trading Active")
    else:
        st.info("⚪ Trading Inactive")
    
    # Settings in effect: snapshot taken at Start while trading, else current sidebar
    active_cfg = st.session_state.get('config', config) if trading_active else config
    
    st.subheader("📋 Strategy Parameters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info(f"**Asset:** {active_cfg.get('asset', 'N/A')}")
        st.info(f"**Interval:** {active_cfg.get('interval', 'N/A')} | **Period:** {active_cfg.get('period', 'N/A')}")
    with col2:
        st.info(f"**Strategy:** {active_cfg.get('strategy', 'N/A')}")
        st.info(f"**Quantity:** {active_cfg.get('quantity', 'N/A')}")
    with col3:
        st.info(f"**SL Type:** {active_cfg.get('sl_type', 'N/A')}")
        st.info(f"**SL Points:** {active_cfg.get('sl_points', 'N/A')}")
    with col4:
        st.info(f"**Target Type:** {active_cfg.get('target_type', 'N/A')}")
        st.info(f"**Target Points:** {active_cfg.get('target_points', 'N/A')}")
    
    if active_cfg.get('strategy') == 'EMA Crossover':
        st.subheader("📊 EMA Strategy Parameters")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.info(f"**Fast EMA:** {active_cfg.get('ema_fast', 'N/A')}")
        col2.info(f"**Slow EMA:** {active_cfg.get('ema_slow', 'N/A')}")
        col3.info(f"**Min Angle:** {active_cfg.get('ema_min_angle', 'N/A')}")
        col4.info(f"**Entry Filter:** {active_cfg.get('ema_entry_filter', 'N/A')}")
        col5.info(f"**ADX:** {'ON ≥ ' + str(active_cfg.get('ema_adx_threshold')) if active_cfg.get('ema_use_adx') else 'OFF'}")
    
    with st.expander("⚙️ All Selected Configurations (every sidebar setting)"
                     + (" – snapshot at Start" if trading_active else ""), expanded=False):
        if trading_active and st.session_state.get('config') != config:
            st.warning("Sidebar was changed after Start. Changes apply after Stop → Start.")
        render_config_summary(active_cfg)

    # ── Live Price Chart ─────────────────────────────────────────────────────
    st.subheader("📊 Live Chart")
    try:
        df_live = st.session_state.get('live_df') if trading_active else None
        if df_live is None:
            df_live = fetch_data(active_cfg.get('asset', 'NIFTY 50'),
                                 INTERVAL_MAPPING.get(active_cfg.get('interval', '1 minute'), '1m'),
                                 PERIOD_MAPPING.get(active_cfg.get('period', '7 days'), '7d'),
                                 is_live_trading=True, custom_ticker=active_cfg.get('custom_ticker'), quiet=True)
            if df_live is not None and not df_live.empty:
                df_live = calculate_all_indicators(df_live, active_cfg)
        if df_live is not None and not df_live.empty:
            plot_df = df_live.tail(150).copy()
            fig_live = go.Figure()
            fig_live.add_trace(go.Candlestick(
                x=plot_df['Datetime'], open=plot_df['Open'], high=plot_df['High'],
                low=plot_df['Low'], close=plot_df['Close'], name='Price',
                increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
            fig_live.add_trace(go.Scatter(x=plot_df['Datetime'], y=plot_df['EMA_Fast'], mode='lines',
                                          name=f"EMA {active_cfg.get('ema_fast', 9)}", line=dict(color='#FF9800', width=1.5)))
            fig_live.add_trace(go.Scatter(x=plot_df['Datetime'], y=plot_df['EMA_Slow'], mode='lines',
                                          name=f"EMA {active_cfg.get('ema_slow', 20)}", line=dict(color='#2196F3', width=1.5)))
            position = st.session_state.get('position')
            if position:
                fig_live.add_hline(y=position['entry_price'], line_dash='dash', line_color='#00E676',
                                   annotation_text=f"Entry {position['type']} @ {position['entry_price']:.2f}")
                if position.get('sl_price') is not None:
                    fig_live.add_hline(y=position['sl_price'], line_dash='dot', line_color='#FF1744',
                                       annotation_text=f"SL @ {position['sl_price']:.2f}")
                if position.get('target_price') is not None:
                    fig_live.add_hline(y=position['target_price'], line_dash='dot', line_color='#00BCD4',
                                       annotation_text=f"Target @ {position['target_price']:.2f}")
            fig_live.update_layout(
                title=f"Live: {active_cfg.get('asset', '')} | {active_cfg.get('interval', '')}",
                xaxis_rangeslider_visible=False, height=420,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                hovermode='x unified', template='plotly_dark')
            st.plotly_chart(fig_live, width="stretch")
        else:
            st.warning("⚠️ Could not load chart data")
    except Exception as e:
        st.warning(f"⚠️ Chart error: {e}")
    
    # ── Current indicators ───────────────────────────────────────────────────
    current_data = st.session_state.get('current_data')
    if current_data is not None:
        st.subheader("📈 Current Market Data (forming candle – same as TradingView last bar)")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Current Price", f"₹{current_data['Close']:.2f}")
        ef, es, ang = current_data.get('EMA_Fast'), current_data.get('EMA_Slow'), current_data.get('EMA_Fast_Angle')
        col2.metric("Fast EMA", f"₹{ef:.2f}" if not _isnan(ef) else "N/A")
        col3.metric("Slow EMA", f"₹{es:.2f}" if not _isnan(es) else "N/A")
        col4.metric("EMA Angle", f"{ang:.2f}°" if not _isnan(ang) else "N/A")
        if not _isnan(ef) and not _isnan(es):
            col5.metric("EMA State", "Bullish ⬆️" if ef > es else "Bearish ⬇️")
        else:
            col5.metric("EMA State", "N/A")
        sc = st.session_state.get('signal_candle')
        if sc is not None:
            st.caption(f"🕯️ Signal candle (last closed): {sc['Datetime']} | Close {sc['Close']:.2f} | "
                       f"EMA Fast {sc.get('EMA_Fast', np.nan):.2f} | EMA Slow {sc.get('EMA_Slow', np.nan):.2f}")
    
    if active_cfg.get('show_last_candle', False) and current_data is not None:
        st.subheader("📊 Last Candle Details")
        with st.expander("🔍 Click to view complete candle data with all indicators", expanded=False):
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Open", f"₹{current_data.get('Open', 0):.2f}")
            col2.metric("High", f"₹{current_data.get('High', 0):.2f}")
            col3.metric("Low", f"₹{current_data.get('Low', 0):.2f}")
            col4.metric("Close", f"₹{current_data.get('Close', 0):.2f}")
            col5.metric("Volume", f"{int(current_data.get('Volume', 0) or 0):,}")
            if 'Datetime' in current_data.index:
                st.caption(f"🕐 Candle Time: {current_data['Datetime']}")
            st.markdown("---")
            st.markdown(f"**📊 Indicators ({active_cfg.get('strategy', '')})**")
            common_indicators = [
                ('EMA_Fast', 'EMA Fast'), ('EMA_Slow', 'EMA Slow'), ('EMA_Fast_Angle', 'EMA Fast Angle'),
                ('EMA_Slow_Angle', 'EMA Slow Angle'), ('SMA_20', 'SMA 20'), ('SMA_50', 'SMA 50'),
                ('RSI', 'RSI'), ('ADX', 'ADX'), ('Plus_DI', '+DI'), ('Minus_DI', '-DI'), ('ATR', 'ATR'),
                ('MACD', 'MACD'), ('MACD_Signal', 'MACD Signal'), ('MACD_Hist', 'MACD Histogram'),
                ('BB_Upper', 'Bollinger Upper'), ('BB_Middle', 'Bollinger Middle'), ('BB_Lower', 'Bollinger Lower'),
                ('VWAP', 'VWAP'), ('SuperTrend', 'SuperTrend'), ('SuperTrend_Direction', 'SuperTrend Direction'),
                ('Swing_High', 'Swing High'), ('Swing_Low', 'Swing Low'),
                ('Prev_Swing_High', 'Prev Swing High'), ('Prev_Swing_Low', 'Prev Swing Low'),
                ('MS_Trend', 'Structure Direction'), ('Volume_MA', 'Volume MA'),
            ]
            items = [(name, current_data[col]) for col, name in common_indicators
                     if col in current_data.index and not _isnan(current_data.get(col))]
            if items:
                for i in range(0, len(items), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(items):
                            name, value = items[i + j]
                            with col:
                                if 'Angle' in name:
                                    st.metric(name, f"{value:.2f}°")
                                elif name in ('RSI', 'ADX', '+DI', '-DI'):
                                    st.metric(name, f"{value:.2f}")
                                elif 'Direction' in name:
                                    st.metric(name, "LONG ⬆️" if value == 1 else ("SHORT ⬇️" if value == -1 else "—"))
                                elif name == 'Volume MA':
                                    st.metric(name, f"{int(value):,}")
                                else:
                                    st.metric(name, f"₹{value:.2f}")
            else:
                st.info("No indicators calculated yet.")
    
    # ── Current position ─────────────────────────────────────────────────────
    st.subheader("📊 Current Position")
    position = st.session_state.get('position')
    if position:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Type", position['type'])
        col2.metric("Entry Price", f"₹{position['entry_price']:.2f}")
        cp = float(current_data['Close']) if current_data is not None else None
        col3.metric("Current Price", f"₹{cp:.2f}" if cp is not None else "N/A")
        col4.metric("Stop Loss", f"₹{position['sl_price']:.2f}" if position['sl_price'] is not None else "Not Set")
        col5.metric("Target", f"₹{position['target_price']:.2f}" if position['target_price'] is not None else "Not Set")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Quantity", position['quantity'])
        col2.metric("Locked Ticker", position.get('ticker', 'N/A'))
        if cp is not None:
            pnl = (cp - position['entry_price']) * position['quantity'] if position['type'] == 'LONG' \
                else (position['entry_price'] - cp) * position['quantity']
            col3.metric("Current P&L", f"₹{pnl:.2f}", delta=f"₹{pnl:.2f}")
        else:
            col3.metric("Current P&L", "N/A")
        col4.metric("Highest Price", f"₹{position.get('highest_price', 0):.2f}")
        col5.metric("Lowest Price", f"₹{position.get('lowest_price', 0):.2f}")
        if position.get('ticker') != config.get('asset'):
            st.warning(f"⚠️ Position locked to {position.get('ticker')}. Config changes won't affect active position.")
        st.info(f"**Entry Time:** {_fmt_time(position['entry_time'])} | **Signal Candle:** {_fmt_time(position.get('signal_candle_time'))}")
    else:
        st.info("No active position")
    
    if active_cfg.get('dhan_enabled', False) and st.session_state.get('broker_position'):
        st.subheader("🏦 Broker Position")
        broker_pos = st.session_state['broker_position']
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Order ID", broker_pos['order_id'])
        if broker_pos.get('is_options', True):
            col2.metric("Option Type", broker_pos.get('option_type', '—'))
        else:
            col2.metric("Transaction", broker_pos['transaction_type'])
        col3.metric("Security ID", broker_pos['security_id'])
        col4.metric("Status", broker_pos['status'])
        with st.expander("📄 Raw API Response"):
            st.json(broker_pos['raw_response'] or {})
    
    # ── Completed trades (numeric stats computed BEFORE formatting) ──────────
    st.subheader("✅ Completed Trades")
    trade_history = st.session_state.get('trade_history', [])
    if trade_history:
        df_raw = pd.DataFrame(trade_history)
        for c in ('pnl', 'brokerage', 'net_pnl', 'price_change_pct'):
            if c not in df_raw.columns:
                df_raw[c] = np.nan
            df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')
        df_raw['brokerage'] = df_raw['brokerage'].fillna(0.0)
        df_raw['net_pnl'] = df_raw['net_pnl'].fillna(df_raw['pnl'] - df_raw['brokerage'])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", len(df_raw))
        col2.metric("Winning", int((df_raw['pnl'] > 0).sum()))
        col3.metric("Total P&L", f"₹{df_raw['pnl'].sum():.2f}")
        col4.metric("Net P&L", f"₹{df_raw['net_pnl'].sum():.2f}")
        
        df_history = df_raw.copy()
        df_history['duration'] = df_history.apply(_duration_min, axis=1)
        df_history['entry_time'] = df_history['entry_time'].apply(_fmt_time)
        df_history['exit_time'] = df_history['exit_time'].apply(_fmt_time)
        df_history['⚠️'] = df_raw['price_change_pct'].apply(lambda x: '⚠️ CHECK' if pd.notna(x) and x > 50 else '')
        _fmt_rupee_cols(df_history, ['entry_price', 'exit_price', 'highest_price', 'lowest_price', 'pnl',
                                     'net_pnl', 'brokerage', 'sl_price', 'target_price'])
        display_cols = ['entry_time', 'exit_time', 'ticker', 'duration', 'type', 'entry_price', 'exit_price',
                        'highest_price', 'lowest_price', 'pnl', 'net_pnl', 'exit_reason', '⚠️']
        st.dataframe(df_history[[c for c in display_cols if c in df_history.columns]], width="stretch", height=200)
        suspicious = int((df_raw['price_change_pct'] > 50).sum())
        if suspicious:
            st.warning(f"⚠️ {suspicious} trade(s) have large price changes (>50%). Check logs.")
    else:
        st.info("No completed trades yet. Trades will appear here immediately after exit.")
    
    st.subheader("📝 Trading Logs")
    logs = st.session_state.get('live_logs', [])
    if logs:
        st.text_area("Trading Logs", value="\n".join(reversed(logs[-80:])), height=300,
                     disabled=True, label_visibility="collapsed")
    else:
        st.info("No logs yet")
    
    if trading_active:
        try:
            live_trading_iteration()
        except Exception as e:
            add_log(f"❌ Iteration error: {e}")
        time.sleep(1.5)
        st.rerun()


def render_trade_logs_ui():
    """Render comprehensive trade history and statistics"""
    st.header("📊 Trade History & Statistics")
    trade_history = st.session_state.get('trade_history', [])
    if not trade_history:
        st.info("No trades recorded yet. Start live trading to see your trade history here.")
        return
    
    df_trades = pd.DataFrame(trade_history)
    df_trades['pnl'] = pd.to_numeric(df_trades['pnl'], errors='coerce').fillna(0.0)
    if 'net_pnl' not in df_trades.columns:
        df_trades['net_pnl'] = df_trades['pnl']
    df_trades['net_pnl'] = pd.to_numeric(df_trades['net_pnl'], errors='coerce').fillna(df_trades['pnl'])
    
    total_trades = len(df_trades)
    profit_trades = int((df_trades['pnl'] > 0).sum())
    loss_trades = int((df_trades['pnl'] < 0).sum())
    total_pnl = df_trades['pnl'].sum()
    avg_pnl = df_trades['pnl'].mean()
    avg_profit = df_trades.loc[df_trades['pnl'] > 0, 'pnl'].mean() if profit_trades else 0
    avg_loss = df_trades.loc[df_trades['pnl'] < 0, 'pnl'].mean() if loss_trades else 0
    accuracy = profit_trades / total_trades * 100 if total_trades else 0
    
    st.subheader("📈 Overall Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Trades", total_trades)
    col2.metric("Profit Trades", profit_trades)
    col3.metric("Loss Trades", loss_trades)
    col4.metric("Accuracy", f"{accuracy:.2f}%")
    col5.metric("Total P&L", f"₹{total_pnl:.2f}", delta=f"₹{total_pnl:.2f}")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Avg P&L", f"₹{avg_pnl:.2f}")
    col2.metric("Avg Profit", f"₹{avg_profit:.2f}")
    col3.metric("Avg Loss", f"₹{avg_loss:.2f}")
    col4.metric("Profit Factor", f"{abs(avg_profit / avg_loss):.2f}" if profit_trades and loss_trades else "N/A")
    col5.metric("Net P&L", f"₹{df_trades['net_pnl'].sum():.2f}")
    
    st.subheader("📋 Detailed Trade History")
    display_df = df_trades.copy()
    display_df['entry_time'] = display_df['entry_time'].apply(_fmt_time)
    display_df['exit_time'] = display_df['exit_time'].apply(_fmt_time)
    _fmt_rupee_cols(display_df, ['entry_price', 'exit_price', 'sl_price', 'target_price', 'highest_price',
                                 'lowest_price', 'pnl', 'net_pnl', 'brokerage', 'price_range'])
    column_order = ['entry_time', 'exit_time', 'ticker', 'type', 'entry_price', 'exit_price', 'sl_price',
                    'target_price', 'highest_price', 'lowest_price', 'price_range', 'quantity', 'pnl',
                    'brokerage', 'net_pnl', 'exit_reason']
    display_df = display_df[[c for c in column_order if c in display_df.columns]]
    display_df = display_df.rename(columns={
        'entry_time': 'Entry Time', 'exit_time': 'Exit Time', 'ticker': 'Ticker', 'type': 'Type',
        'entry_price': 'Entry Price', 'exit_price': 'Exit Price', 'sl_price': 'Stop Loss', 'target_price': 'Target',
        'highest_price': 'Highest Price', 'lowest_price': 'Lowest Price', 'price_range': 'Price Range',
        'quantity': 'Quantity', 'pnl': 'P&L', 'brokerage': 'Brokerage', 'net_pnl': 'Net P&L',
        'exit_reason': 'Exit Reason'})
    st.dataframe(display_df, width="stretch", height=400)
    st.download_button(
        label="📥 Download Trade History (CSV)", data=display_df.to_csv(index=False),
        file_name=f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
    
    st.subheader("📊 P&L Chart")
    chart = df_trades.copy()
    chart['cumulative_pnl'] = chart['pnl'].cumsum()
    chart['trade_number'] = range(1, len(chart) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart['trade_number'], y=chart['cumulative_pnl'], mode='lines+markers',
                             name='Cumulative P&L', line=dict(color='blue', width=2), marker=dict(size=6)))
    fig.add_trace(go.Bar(x=chart['trade_number'], y=chart['pnl'], name='Trade P&L',
                         marker=dict(color=['green' if p > 0 else 'red' for p in chart['pnl']]), opacity=0.6))
    fig.update_layout(title='Trade P&L Analysis', xaxis_title='Trade Number', yaxis_title='P&L (₹)',
                      hovermode='x unified', height=400)
    st.plotly_chart(fig, width="stretch")
    
    st.subheader("📊 Trade Type Distribution")
    col1, col2 = st.columns(2)
    with col1:
        tc = df_trades['type'].value_counts()
        f1 = go.Figure(data=[go.Pie(labels=tc.index, values=tc.values, hole=0.4)])
        f1.update_layout(title='Long vs Short Trades', height=300)
        st.plotly_chart(f1, width="stretch")
    with col2:
        rc = df_trades['exit_reason'].value_counts()
        f2 = go.Figure(data=[go.Pie(labels=rc.index, values=rc.values, hole=0.4)])
        f2.update_layout(title='Exit Reason Distribution', height=300)
        st.plotly_chart(f2, width="stretch")

# ================================
# MAIN APP
# ================================

def main():
    """Main application"""
    st.set_page_config(page_title="Algorithmic Trading System", page_icon="📈", layout="wide")
    st.title("📈 Algorithmic Trading System")
    
    config = render_config_ui()
    
    ticker_display = config.get('asset', 'NIFTY 50')
    if ticker_display == 'Custom Ticker':
        st.info(f"🎯 **Selected Ticker:** {config.get('custom_ticker', 'N/A')} (Custom)")
    else:
        st.info(f"🎯 **Selected Ticker:** {ticker_display} ({ASSET_MAPPING.get(ticker_display, ticker_display)})")
    
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📈 Backtest", "🔴 Live Trading", "📊 Trade History"])
    with tab1:
        render_backtest_ui(config)
    with tab2:
        render_live_trading_ui(config)
    with tab3:
        render_trade_logs_ui()

if __name__ == "__main__":
    main()
