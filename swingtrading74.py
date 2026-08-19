"""
Complete Algorithmic Trading System  (v2.0)
===========================================

A comprehensive trading system with:
- Multiple trading strategies (EMA Crossover, Donchian, Keltner, Heikin Ashi, Ichimoku, ...)
- 18+ Stop Loss types including Cost-to-Cost trailing
- 12+ Target types
- Real broker integration (Dhan API)
- Live trading and backtesting capabilities
- TradingView-compatible indicator maths (EMA/RSI/ATR/ADX/BB seeded exactly like TV)
- Email alerts (Gmail app-password) + Groq AI chatbot on every tab

WHAT CHANGED IN v2.0  (search the tags in CAPS to jump around)
--------------------------------------------------------------
1. "USER DEFAULTS"        -> single block at top of file to change SL/Target/strategy defaults
2. "CROSSOVER FIX"        -> live entries only on a *freshly closed* candle crossover (no repeats)
3. "MANUAL SQUAREOFF FIX" -> manual close no longer crashes / no longer stops live trading
4. "WARMUP FIX"           -> enough history is always downloaded so EMA9/EMA21 are never NaN at 09:15
5. "TRADINGVIEW MATH"     -> EMA/RSI/ATR/ADX/BB/VWAP computed exactly like TradingView
6. "BACKTEST ENTRY N+1"   -> signal on candle N => entry at OPEN of candle N+1
7. "CONSERVATIVE EXIT"    -> SL checked against Low/High FIRST, then Target
8. "NEW STRATEGIES"       -> Donchian, Keltner, Heikin Ashi, HA+EMA, MACD crossover (+ existing Ichimoku)
9. "EMAIL ALERTS"         -> optional Gmail notifications (works in paper AND broker mode)
10. "GROQ CHATBOT"        -> optional AI chat on Backtest / Live / Trade-History tabs

Author: Claude
Version: 2.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from datetime import time as dt_time
import time
import pytz
import traceback
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import plotly.graph_objects as go
from scipy.signal import argrelextrema
import warnings

# Suppress SyntaxWarning from dhanhq library (invalid escape sequence in their code)
warnings.filterwarnings('ignore', category=SyntaxWarning, module='dhanhq')

try:
    import requests
except ImportError:              # requests ships with streamlit, but just in case
    requests = None


# =============================================================================
# =============================================================================
# ##                        USER DEFAULTS                                    ##
# ##   >>>>>>  EDIT EVERYTHING HERE. NOTHING ELSE NEEDS TOUCHING.  <<<<<<     ##
# ##   (search the file for "USER DEFAULTS" to come back to this block)       ##
# =============================================================================
# =============================================================================

# ---- CHANGE #1: default Stop-Loss ------------------------------------------
# Valid values are ANY string from the SL_TYPES list further down the file.
DEFAULT_SL_TYPE   = "Trailing SL (Points)"        # <-- change SL type here
DEFAULT_SL_POINTS = 10                            # <-- change SL points here
DEFAULT_SL_RUPEES = 100                           # used only by "P&L Based (Rupees)"
DEFAULT_SL_TRAIL_RUPEES = 50                      # used by Trailing Profit/Loss (Rupees)
DEFAULT_SL_ATR_MULT = 1.5                         # used by ATR-based SL types

# ---- CHANGE #2: default Target ---------------------------------------------
# Valid values are ANY string from the TARGET_TYPES list further down the file.
DEFAULT_TARGET_TYPE   = "Signal-based (Reverse Crossover)"   # <-- change Target type here
DEFAULT_TARGET_POINTS = 20                        # used by point-based targets
DEFAULT_TARGET_RUPEES = 200                       # used by "P&L Based (Rupees)"
DEFAULT_TARGET_ATR_MULT = 2.0
DEFAULT_RISK_REWARD = 2.0

# ---- CHANGE #3: default Strategy / instrument / timeframe -------------------
DEFAULT_STRATEGY = "EMA Crossover"                # <-- change default strategy here
DEFAULT_ASSET    = "NIFTY 50"
DEFAULT_INTERVAL = "1 minute"
DEFAULT_PERIOD   = "5 days"
DEFAULT_QUANTITY = 1

# ---- CHANGE #4: EMA Crossover defaults -------------------------------------
DEFAULT_EMA_FAST      = 9
DEFAULT_EMA_SLOW      = 21
DEFAULT_EMA_MIN_ANGLE = 0.0      # 0 = angle filter OFF (was 30 which blocked most signals)
DEFAULT_EMA_USE_ADX   = False    # ADX confirmation filter OFF by default
DEFAULT_EMA_ADX_THRESHOLD = 20
DEFAULT_ADX_PERIOD    = 14

# ---- CHANGE #5: engine behaviour -------------------------------------------
# "CROSSOVER FIX": in LIVE trading, evaluate the strategy on the last *CLOSED*
# candle (index -2) instead of the still-forming candle (index -1).
# This is exactly the prev(-3)/curr(-2) logic you described and it stops the
# same crossover from firing on every 1.5-second refresh.
LIVE_USE_CLOSED_CANDLE_ONLY = True

# One entry per candle maximum (extra safety against duplicate live orders)
LIVE_ONE_ENTRY_PER_BAR = True

# "BACKTEST ENTRY N+1": signal on candle N -> entry at OPEN of candle N+1
DEFAULT_BACKTEST_NEXT_CANDLE_ENTRY = True

# "CONSERVATIVE EXIT": inside a candle assume the WORST case ->
# check SL against the candle LOW (long) / HIGH (short) FIRST, target second.
DEFAULT_CONSERVATIVE_INTRABAR_EXIT = True

# "WARMUP FIX": minimum number of candles we insist on downloading so that
# EMA/RSI/ADX are already "warm" at 09:15 AM instead of showing NaN.
MIN_WARMUP_BARS = 400

# ---- CHANGE #6: Email alert defaults ---------------------------------------
DEFAULT_EMAIL_ENABLED   = False                    # checkbox starts DISABLED
DEFAULT_EMAIL_FROM      = "srinivasp451@gmail.com"
DEFAULT_EMAIL_TO        = "srinivasp451@gmail.com"
DEFAULT_EMAIL_SMTP_HOST = "smtp.gmail.com"
DEFAULT_EMAIL_SMTP_PORT = 465                      # SSL

# ---- CHANGE #7: Groq chatbot defaults --------------------------------------
DEFAULT_GROQ_ENABLED = False                       # checkbox starts DISABLED
DEFAULT_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# NOTE: Groq has DECOMMISSIONED llama3-70b-8192, llama3-8b-8192,
# mixtral-8x7b-32768, gemma-7b-it and gemma2-9b-it.
# The list below only contains models that are currently servable.
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct",
    "deepseek-r1-distill-llama-70b",
]
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# ---- CHANGE #8: Dhan defaults (leave token blank -> paste it in the UI) -----
DEFAULT_DHAN_CLIENT_ID    = ""     # <- put your client id here if you want it pre-filled
DEFAULT_DHAN_ACCESS_TOKEN = ""     # <- SECURITY: keep tokens OUT of source code

# =============================================================================
# ##                    END OF USER DEFAULTS BLOCK                           ##
# =============================================================================


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

# "WARMUP FIX" helper tables ---------------------------------------------------
PERIOD_DAYS = {
    '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
    '6mo': 180, '1y': 365, '2y': 730, '5y': 1825
}
# approx number of candles produced per CALENDAR day (NSE ~ 375 min session)
BARS_PER_DAY = {
    '1m': 260, '5m': 52, '15m': 18, '30m': 9, '1h': 5, '1d': 0.7, '1wk': 0.15
}
# yfinance hard limits for intraday history
INTERVAL_MAX_DAYS = {
    '1m': 7, '5m': 59, '15m': 59, '30m': 59, '1h': 700, '1d': 20000, '1wk': 20000
}
INTRADAY_INTERVALS = ('1m', '5m', '15m', '30m', '1h')

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
    "Support Resistance Bounce",
    # ---- "NEW STRATEGIES" (v2.0) ----
    "Donchian Channel Breakout",
    "Keltner Channel Breakout",
    "Heikin Ashi Trend Flip",
    "Heikin Ashi + EMA Confirmation",
    "MACD Signal Crossover",
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

IST = pytz.timezone('Asia/Kolkata')


def safe_index(options, value, fallback=0):
    """Return the index of `value` inside `options`, else `fallback`.
    Used so the USER DEFAULTS block can never crash the selectboxes."""
    try:
        return options.index(value)
    except (ValueError, AttributeError):
        return fallback


def now_ist():
    return datetime.now(IST)


# =============================================================================
# "EMAIL ALERTS"  -  Gmail notifications (works in paper trading AND live/Dhan)
# =============================================================================

def send_email_notification(subject, body, config, log_func=None):
    """
    Send an email alert via Gmail SMTP-SSL using an App Password.

    Enabled by the "Enable Email Notifications" checkbox (OFF by default).
    Works for paper trading as well as when the Dhan broker is enabled.

    Returns True on success, False otherwise (never raises).
    """
    if not config.get('enable_email_alerts', False):
        return False

    sender = (config.get('email_from') or DEFAULT_EMAIL_FROM).strip()
    receiver = (config.get('email_to') or DEFAULT_EMAIL_TO).strip()
    app_password = (config.get('email_app_password') or '').strip()

    if not app_password:
        if log_func:
            log_func("📧 ⚠️ Email enabled but App Password is empty - skipping mail")
        return False

    try:
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = receiver
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        host = config.get('email_smtp_host', DEFAULT_EMAIL_SMTP_HOST)
        port = int(config.get('email_smtp_port', DEFAULT_EMAIL_SMTP_PORT))

        with smtplib.SMTP_SSL(host, port, timeout=15) as server:
            server.login(sender, app_password)
            server.sendmail(sender, [r.strip() for r in receiver.split(',') if r.strip()],
                            msg.as_string())

        if log_func:
            log_func(f"📧 ✅ Email sent: {subject}")
        return True

    except Exception as e:
        if log_func:
            log_func(f"📧 ❌ Email failed: {e}")
        return False


def email_trade_event(event, position, config, log_func=None, extra=None):
    """Convenience wrapper: builds a readable mail body for ENTRY / EXIT events."""
    if not config.get('enable_email_alerts', False):
        return False

    mode = "LIVE (Dhan)" if config.get('dhan_enabled', False) else "PAPER"
    lines = [
        f"Event      : {event}",
        f"Mode       : {mode}",
        f"Time (IST) : {now_ist().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Asset      : {config.get('asset', 'N/A')} ({config.get('interval', 'N/A')})",
        f"Strategy   : {config.get('strategy', 'N/A')}",
        "",
    ]
    if position:
        lines += [
            f"Direction  : {position.get('type', 'N/A')}",
            f"Entry      : {position.get('entry_price', 0):.2f}",
            f"Stop Loss  : {position.get('sl_price')}",
            f"Target     : {position.get('target_price')}",
            f"Quantity   : {position.get('quantity', 0)}",
        ]
    if extra:
        lines.append("")
        for k, v in extra.items():
            lines.append(f"{k:<11}: {v}")

    subject = f"[ALGO {event}] {config.get('asset', '')} {position.get('type', '') if position else ''}"
    return send_email_notification(subject, "\n".join(lines), config, log_func)


# =============================================================================
# "GROQ CHATBOT"  -  optional AI assistant available on every tab
# =============================================================================

def groq_chat_completion(messages, config):
    """
    Call the Groq OpenAI-compatible chat endpoint.
    Returns (text, error). Never raises.
    """
    if requests is None:
        return None, "The `requests` package is not installed."

    api_key = (config.get('groq_api_key') or '').strip()
    if not api_key:
        return None, "Groq API key is empty. Paste it in the sidebar."

    model = config.get('groq_model', DEFAULT_GROQ_MODEL)

    try:
        resp = requests.post(
            DEFAULT_GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 2048,
            },
            timeout=90,
        )
    except Exception as e:
        return None, f"Network error talking to Groq: {e}"

    if resp.status_code != 200:
        # Groq returns a helpful JSON error body (e.g. model decommissioned)
        try:
            detail = resp.json().get('error', {}).get('message', resp.text[:400])
        except Exception:
            detail = resp.text[:400]
        return None, f"Groq API error {resp.status_code}: {detail}"

    try:
        data = resp.json()
        return data['choices'][0]['message']['content'], None
    except Exception as e:
        return None, f"Unexpected Groq response: {e}"


def _df_to_context(df, max_rows=40, float_fmt="%.2f"):
    """Compact CSV-ish representation of a DataFrame for the LLM context."""
    if df is None or len(df) == 0:
        return "(no rows)"
    sub = df.tail(max_rows).copy()
    for c in sub.columns:
        if pd.api.types.is_float_dtype(sub[c]):
            sub[c] = sub[c].round(4)
    try:
        return sub.to_csv(index=False)
    except Exception:
        return str(sub)


def render_groq_chat(tab_key, context_text, config, title="🤖 Ask the AI about these results"):
    """
    Renders a chat panel. `context_text` is the tab-specific data snapshot
    (metrics, trade table, chart series, logs) that gets injected as system
    context so the model can reason about YOUR numbers.

    Chart *images* cannot be sent to a text model, so the chart's underlying
    OHLC / indicator series are serialised instead - the model sees the same
    information the chart is drawn from.
    """
    if not config.get('enable_groq_chat', False):
        return

    st.markdown("---")
    st.subheader(title)

    hist_key = f"groq_history_{tab_key}"
    if hist_key not in st.session_state:
        st.session_state[hist_key] = []

    col_a, col_b = st.columns([4, 1])
    with col_a:
        st.caption(f"Model: `{config.get('groq_model', DEFAULT_GROQ_MODEL)}` · "
                   f"context: {len(context_text):,} chars from this tab")
    with col_b:
        if st.button("🧹 Clear chat", key=f"clear_chat_{tab_key}"):
            st.session_state[hist_key] = []
            st.rerun()

    with st.expander("👁️ View exactly what is sent to the model", expanded=False):
        st.code(context_text[:12000] + ("\n...[truncated]" if len(context_text) > 12000 else ""),
                language="text")

    # Replay history
    for msg in st.session_state[hist_key]:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    prompt = st.chat_input("Ask about your trades, P&L, drawdown, indicator values...",
                           key=f"chat_input_{tab_key}")

    if prompt:
        st.session_state[hist_key].append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        system_prompt = (
            "You are a quantitative trading assistant embedded in a Python/Streamlit "
            "algo-trading application. The user will ask about the data snapshot below, "
            "which comes from their own backtest / live session. "
            "Be concise, quantitative and honest. If something cannot be determined from "
            "the snapshot, say so instead of guessing. Never claim a strategy is guaranteed "
            "to be profitable; note that past performance does not predict future results.\n\n"
            f"=== DATA SNAPSHOT ({tab_key}) ===\n{context_text[:24000]}\n=== END SNAPSHOT ==="
        )

        messages = [{'role': 'system', 'content': system_prompt}]
        messages += st.session_state[hist_key][-10:]

        with st.chat_message('assistant'):
            with st.spinner("Thinking..."):
                answer, err = groq_chat_completion(messages, config)
            if err:
                st.error(err)
                answer = f"⚠️ {err}"
            else:
                st.markdown(answer)

        st.session_state[hist_key].append({'role': 'assistant', 'content': answer})


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
            self.dhanhq_module = None

    def _resolve_security(self, signal):
        """
        Resolve security ID based on signal type

        Args:
            signal: 'BUY', 'SELL', 'LONG', or 'SHORT'

        Returns:
            tuple: (security_id, option_type)
        """
        if signal in ('BUY', 'LONG'):
            security_id = self.config.get('dhan_ce_security_id', '42568')
            option_type = 'CE'
        else:  # 'SELL' or 'SHORT'
            security_id = self.config.get('dhan_pe_security_id', '42569')
            option_type = 'PE'

        return security_id, option_type

    def _get_exchange_segment(self):
        """Determine exchange segment based on asset and trading type"""
        if not self.dhanhq_module:
            return "NSE_FNO"

        is_options = self.config.get('dhan_is_options', True)
        exchange = self.config.get('dhan_exchange', 'NSE')

        if is_options:
            asset = self.config.get('asset', 'NIFTY 50')
            if asset == 'SENSEX':
                return self.dhanhq_module.BSE_FNO
            return self.dhanhq_module.NSE_FNO
        else:
            if exchange == 'BSE':
                return self.dhanhq_module.BSE  # BSE_EQ
            return self.dhanhq_module.NSE      # NSE_EQ

    def place_order(self, transaction_type, security_id, quantity, signal_type=None,
                    order_params=None, is_exit=False):
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
                is_options = self.config.get('dhan_is_options', True)
                trading_type = self.config.get('dhan_trading_type', 'Intraday')
                use_broker_sl = self.config.get('broker_use_own_sl', False)

                if is_exit:
                    order_type_selection = self.config.get('dhan_exit_order_type', 'Market Order')
                else:
                    order_type_selection = self.config.get('dhan_entry_order_type', 'Market Order')

                if not order_type_selection:
                    order_type_selection = self.config.get('dhan_order_type', 'Market Order')

                op = order_params or {}

                if order_type_selection == 'Limit Order':
                    order_type = self.dhanhq_module.LIMIT
                    limit_price = float(op.get('price', 0)) if op else 0
                else:
                    order_type = self.dhanhq_module.MARKET
                    limit_price = 0

                if use_broker_sl and op:
                    # ── Bracket Order (BO) - always uses LIMIT ──────────────
                    lmt_price = float(op.get('price', 0))
                    bo_profit = float(op.get('boProfitValue', 0))
                    bo_sl = float(op.get('boStopLossValue', 0))
                    trail_sl = float(op.get('trailStopLoss', 0))

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
                    order_response['order_id'] = response.get('data', {}).get(
                        'orderId', f"ORDER-{int(time.time())}")
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
        is_options = config.get('dhan_is_options', True)
        quantity = config.get('dhan_quantity', 10)
        trading_type = config.get('dhan_trading_type', 'Intraday')
        use_broker_sl = config.get('broker_use_own_sl', False)
        log_func(f"🏦 NEW signal detected: {signal}")

        def _build_bo_params(txn, entry_px):
            if not use_broker_sl:
                return {'price': entry_px}
            sl_pts = float(config.get('broker_sl_points', 50))
            tgt_pts = float(config.get('broker_target_points', 100))
            trail = float(config.get('broker_trailing_jump', 0))
            return {
                'price': entry_px,
                'boProfitValue': tgt_pts,
                'boStopLossValue': sl_pts,
                'trailStopLoss': trail
            }

        if is_options:
            security_id, option_type = self._resolve_security(signal)
            log_func(f"🏦 Options [{option_type}] Security ID: {security_id}")
            txn = 'BUY'
            op = _build_bo_params(txn, price)
            order_response = self.place_order(txn, security_id, quantity, signal, op)
            broker_position = {
                'order_id': order_response['order_id'],
                'signal_type': signal, 'option_type': option_type,
                'security_id': security_id, 'transaction_type': txn,
                'entry_price': price, 'quantity': quantity,
                'timestamp': now_ist(),
                'status': order_response['status'],
                'raw_response': order_response['raw_response'],
                'is_options': True, 'trading_type': 'Options',
                'broker_sl_active': use_broker_sl
            }

        else:
            security_id = config.get('dhan_security_id', '1234')
            txn = 'BUY' if signal in ('BUY', 'LONG') else 'SELL'
            log_func(f"🏦 {'Delivery' if trading_type == 'Delivery (CNC)' else 'Intraday'} "
                     f"→ {txn} | Security: {security_id}")
            op = _build_bo_params(txn, price)
            order_response = self.place_order(txn, security_id, quantity, signal, op)
            broker_position = {
                'order_id': order_response['order_id'],
                'signal_type': signal, 'security_id': security_id,
                'transaction_type': txn, 'entry_price': price, 'quantity': quantity,
                'timestamp': now_ist(),
                'status': order_response['status'],
                'raw_response': order_response['raw_response'],
                'is_options': False, 'trading_type': trading_type,
                'broker_sl_active': use_broker_sl,
                'option_type': '-'
            }

        if order_response['status'] in ('SUCCESS', 'SIMULATED'):
            bo_info = " [Bracket Order: SL/Target managed by Dhan]" if use_broker_sl else ""
            log_func(f"🏦 ✅ ORDER PLACED: {broker_position['transaction_type']} "
                     f"{quantity} @ {price:.2f}{bo_info}")
        else:
            log_func(f"🏦 ❌ ORDER FAILED: {order_response.get('error', 'Unknown error')}")
        return broker_position

    def exit_broker_position(self, broker_position, price, reason, log_func):
        """Exit broker position - handles both options and stock trading"""
        security_id = broker_position['security_id']
        quantity = broker_position['quantity']
        is_options = broker_position.get('is_options', True)

        log_func(f"🏦 Exiting position: {reason}")

        if is_options:
            exit_transaction = 'SELL'
            log_func("🏦 Options Exit → SELL")
        else:
            entry_transaction = broker_position['transaction_type']
            if entry_transaction == 'BUY':
                exit_transaction = 'SELL'
                log_func("🏦 Stock Exit → SELL (close long)")
            else:
                exit_transaction = 'BUY'
                log_func("🏦 Stock Exit → BUY (square off short)")

        order_response = self.place_order(
            exit_transaction,
            security_id,
            quantity,
            order_params={'price': price},
            is_exit=True
        )

        entry_price = broker_position['entry_price']
        signal_type = broker_position['signal_type']

        if signal_type in ('BUY', 'LONG'):
            pnl = (price - entry_price) * quantity
        else:
            pnl = (entry_price - price) * quantity

        exit_info = {
            'order_id': order_response['order_id'],
            'transaction_type': exit_transaction,
            'exit_price': price,
            'quantity': quantity,
            'pnl': pnl,
            'timestamp': now_ist(),
            'status': order_response['status'],
            'raw_response': order_response['raw_response']
        }

        if order_response['status'] in ('SUCCESS', 'SIMULATED'):
            log_func(f"🏦 ✅ DHAN EXIT ORDER PLACED: {exit_transaction} {quantity} "
                     f"@ {price:.2f} | P&L: ₹{pnl:.2f}")
        else:
            log_func(f"🏦 ❌ DHAN EXIT ORDER FAILED: {order_response.get('error', 'Unknown error')}")

        return exit_info

    def clear_all_positions(self, log_func, convert_to_market=True):
        """Clear all positions: cancel/convert pending orders and close open positions"""
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

            order_list = self.dhan.get_order_list()

            if order_list and order_list.get('status') == 'success':
                orders = order_list.get('data', [])
                log_func(f"🏦 Found {len(orders)} orders to process")

                for order in orders:
                    order_status = order.get('orderStatus', '')
                    order_id = order.get('orderId', '')
                    order_type = order.get('orderType', '')

                    if order_status == 'PENDING' and order_type == 'LIMIT' and convert_to_market:
                        try:
                            log_func(f"🏦 Converting pending LIMIT order {order_id} to MARKET...")
                            cancel_response = self.dhan.cancel_order(order_id)

                            if cancel_response and cancel_response.get('status') == 'success':
                                log_func(f"🏦 ✅ Cancelled LIMIT order: {order_id}")
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
                                    log_func("🏦 ✅ Converted to MARKET order: "
                                             f"{market_response.get('data', {}).get('orderId', 'N/A')}")
                                else:
                                    result['cancelled_orders'] += 1
                                    log_func("🏦 ⚠️ MARKET conversion failed, order cancelled")
                            else:
                                error_msg = (f"Failed to cancel order {order_id}: "
                                             f"{(cancel_response or {}).get('remarks', 'Unknown')}")
                                result['errors'].append(error_msg)
                                log_func(f"🏦 ⚠️ {error_msg}")
                        except Exception as e:
                            error_msg = f"Error converting order {order_id}: {str(e)}"
                            result['errors'].append(error_msg)
                            log_func(f"🏦 ❌ {error_msg}")

                    elif order_status == 'PENDING':
                        try:
                            cancel_response = self.dhan.cancel_order(order_id)
                            if cancel_response and cancel_response.get('status') == 'success':
                                result['cancelled_orders'] += 1
                                log_func(f"🏦 ✅ Cancelled pending order: {order_id}")
                            else:
                                error_msg = (f"Failed to cancel order {order_id}: "
                                             f"{(cancel_response or {}).get('remarks', 'Unknown')}")
                                result['errors'].append(error_msg)
                                log_func(f"🏦 ⚠️ {error_msg}")
                        except Exception as e:
                            error_msg = f"Error cancelling order {order_id}: {str(e)}"
                            result['errors'].append(error_msg)
                            log_func(f"🏦 ❌ {error_msg}")

                    elif order_status in ['TRANSIT', 'TRADED']:
                        try:
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
                                error_msg = (f"Failed to close position {order_id}: "
                                             f"{(close_response or {}).get('remarks', 'Unknown')}")
                                result['errors'].append(error_msg)
                                log_func(f"🏦 ⚠️ {error_msg}")
                        except Exception as e:
                            error_msg = f"Error closing position {order_id}: {str(e)}"
                            result['errors'].append(error_msg)
                            log_func(f"🏦 ❌ {error_msg}")

            result['clearing_complete'] = True

            log_func(f"🏦 🧹 Clearing Complete: {result['cancelled_orders']} cancelled, "
                     f"{result['converted_orders']} converted, {result['closed_positions']} closed")

            if result['errors']:
                log_func(f"🏦 ⚠️ {len(result['errors'])} errors during clearing")

        except Exception as e:
            error_msg = f"Error in clear_all_positions: {str(e)}"
            result['errors'].append(error_msg)
            log_func(f"🏦 ❌ {error_msg}")
            result['clearing_complete'] = True

        return result


# =============================================================================
# DATA FETCHING   ("WARMUP FIX")
# =============================================================================

def _required_days_for_warmup(interval_code, requested_period_code):
    """
    Work out how many CALENDAR days of history we must download so that
    every indicator is already warmed up on the very first candle of today.

    THIS IS THE FIX for "at 09:15 EMA9 shows NA, at 09:24 it appears".
    The old code forced period='1d' during live trading, so at 09:15 the
    dataframe only had 1-2 candles and EMA(9)/EMA(21) were genuinely NaN.
    Now we always keep at least MIN_WARMUP_BARS candles of history, capped
    by whatever yfinance actually allows for that interval.
    """
    requested_days = PERIOD_DAYS.get(requested_period_code, 5)
    bars_per_day = BARS_PER_DAY.get(interval_code, 100)
    # +2 days of slack for weekends / holidays
    warmup_days = int(np.ceil(MIN_WARMUP_BARS / max(bars_per_day, 0.1))) + 2
    days = max(requested_days, warmup_days)
    return int(min(days, INTERVAL_MAX_DAYS.get(interval_code, 3650)))


def fetch_data(ticker_symbol, interval, period, is_live_trading=False, custom_ticker=None):
    """
    Fetch historical/live data using yfinance.

    v2.0: never shrinks the history below the indicator warm-up requirement,
    so EMA/RSI/ADX values are valid from the first candle of the session and
    they carry over across the previous close -> today's open gap exactly the
    way TradingView does (EMA is continuous, it is NOT reset each day, so
    gap-ups / gap-downs are handled identically).
    """
    try:
        if ticker_symbol == "Custom Ticker" and custom_ticker:
            ticker = custom_ticker
        else:
            ticker = ASSET_MAPPING.get(ticker_symbol, ticker_symbol)

        days = _required_days_for_warmup(interval, period)

        if interval in INTRADAY_INTERVALS:
            # Date-range download: lets us ask for e.g. 7 days of 1-minute data,
            # which is impossible with the standard yfinance `period` strings.
            end_dt = datetime.now(IST) + timedelta(days=1)
            start_dt = end_dt - timedelta(days=days + 1)
            df = yf.download(
                ticker,
                interval=interval,
                start=start_dt.strftime('%Y-%m-%d'),
                end=end_dt.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=False,   # raw prices => matches TradingView levels
                prepost=False,       # exclude pre/post market so bars align with TV
            )
        else:
            df = yf.download(
                ticker,
                interval=interval,
                period=period,
                progress=False,
                auto_adjust=False,
            )

        if df is None or df.empty:
            st.error(f"❌ No data returned for {ticker_symbol}")
            return None

        # ── Handle newer yfinance MultiIndex columns ──────────────────
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df = df.loc[:, ~df.columns.duplicated()]

        # ── Move date index into a column named "Datetime" ────────────
        df = df.reset_index()
        if 'Datetime' not in df.columns:
            if 'Date' in df.columns:
                df.rename(columns={'Date': 'Datetime'}, inplace=True)
            else:
                df.rename(columns={df.columns[0]: 'Datetime'}, inplace=True)

        # ── Timezone: always end up in IST ────────────────────────────
        dt = pd.to_datetime(df['Datetime'])
        try:
            if dt.dt.tz is None:
                dt = dt.dt.tz_localize(IST)
            else:
                dt = dt.dt.tz_convert(IST)
        except Exception:
            dt = pd.to_datetime(df['Datetime'], utc=True).dt.tz_convert(IST)
        df['Datetime'] = dt

        # Drop rows with no close (yfinance sometimes returns empty padding bars)
        df = df.dropna(subset=['Close']).reset_index(drop=True)

        if 'Volume' not in df.columns:
            df['Volume'] = 0

        return df

    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        return None


# =============================================================================
# "TRADINGVIEW MATH"  -  indicator primitives that match Pine Script exactly
# =============================================================================
# TradingView seeds ta.ema() with an SMA of the first `length` values and then
# applies the recursive formula. pandas' .ewm(adjust=False) does NOT do the SMA
# seed, which is why plain pandas EMAs drift a few points away from TV.
# ta.rsi / ta.atr / ta.adx all use Wilder's RMA (alpha = 1/length), NOT a
# simple rolling mean - the original code used rolling means, hence mismatch.
# ta.stdev uses the POPULATION standard deviation (ddof=0), pandas defaults to
# the sample stdev (ddof=1) - that shifts Bollinger Bands too.
# =============================================================================

def _recursive_smooth(series, length, alpha):
    """Generic Wilder/EMA style recursion seeded with an SMA of the first `length` values."""
    s = pd.Series(series).astype(float)
    arr = s.to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)

    if n == 0 or length is None or length <= 0:
        return pd.Series(out, index=s.index)

    valid_idx = np.where(~np.isnan(arr))[0]
    if len(valid_idx) < length:
        return pd.Series(out, index=s.index)

    seed_pos = valid_idx[length - 1]
    out[seed_pos] = float(np.mean(arr[valid_idx[:length]]))

    for i in range(seed_pos + 1, n):
        x = arr[i]
        if np.isnan(x):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * x + (1.0 - alpha) * out[i - 1]

    return pd.Series(out, index=s.index)


def tv_ema(series, length):
    """TradingView ta.ema() - SMA seed + alpha = 2/(length+1)."""
    length = int(length)
    return _recursive_smooth(series, length, 2.0 / (length + 1.0))


def tv_rma(series, length):
    """TradingView ta.rma() / Wilder smoothing - alpha = 1/length."""
    length = int(length)
    return _recursive_smooth(series, length, 1.0 / length)


def tv_sma(series, length):
    return pd.Series(series).astype(float).rolling(int(length)).mean()


def tv_stdev(series, length):
    """TradingView ta.stdev() -> population stdev (ddof=0)."""
    return pd.Series(series).astype(float).rolling(int(length)).std(ddof=0)


def true_range(df):
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    # first bar: TradingView uses high-low
    tr.iloc[0] = (high.iloc[0] - low.iloc[0]) if len(tr) else np.nan
    return tr


def calculate_ema_angle(ema_series, lookback=3):
    """
    Calculate EMA angle in degrees (slope of the last `lookback` points).
    Vectorised version of the original loop (same numbers, ~100x faster).
    """
    s = pd.Series(ema_series).astype(float)
    slope = (s - s.shift(lookback)) / float(lookback)
    return np.degrees(np.arctan(slope))


def calculate_rsi(series, period=14):
    """TradingView-compatible RSI (Wilder RMA based)."""
    s = pd.Series(series).astype(float)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = tv_rma(gain, period)
    avg_loss = tv_rma(loss, period)

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, rsi.where(avg_loss == 0, 0.0))
    return rsi


def calculate_atr(df, period=14):
    """TradingView-compatible ATR = rma(TR, period)."""
    return tv_rma(true_range(df), period)


def calculate_adx(df, period=14):
    """
    TradingView-compatible ADX / DI.
        up      = high - high[1]
        down    = low[1] - low
        +DM     = up   if (up > down  and up   > 0) else 0
        -DM     = down if (down > up  and down > 0) else 0
        +DI     = 100 * rma(+DM, len) / rma(TR, len)
        ADX     = rma(DX, len)
    """
    high = df['High'].astype(float)
    low = df['Low'].astype(float)

    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    trur = tv_rma(true_range(df), period)
    plus_di = 100 * tv_rma(plus_dm, period) / trur.replace(0, np.nan)
    minus_di = 100 * tv_rma(minus_dm, period) / trur.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return tv_rma(dx, period)


def calculate_heikin_ashi(df):
    """
    Heikin Ashi candles (TradingView identical).
        HA_Close = (O+H+L+C)/4
        HA_Open  = (prev HA_Open + prev HA_Close)/2   [first bar: (O+C)/2]
        HA_High  = max(High, HA_Open, HA_Close)
        HA_Low   = min(Low,  HA_Open, HA_Close)
    """
    o = df['Open'].astype(float).to_numpy()
    h = df['High'].astype(float).to_numpy()
    low_ = df['Low'].astype(float).to_numpy()
    c = df['Close'].astype(float).to_numpy()
    n = len(df)

    ha_close = (o + h + low_ + c) / 4.0
    ha_open = np.full(n, np.nan)
    if n:
        ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([low_, ha_open, ha_close])
    return ha_open, ha_high, ha_low, ha_close


def calculate_session_vwap(df):
    """Session-anchored VWAP (resets every trading day) - this is what TradingView shows."""
    tp = (df['High'].astype(float) + df['Low'].astype(float) + df['Close'].astype(float)) / 3.0
    vol = df['Volume'].astype(float).fillna(0)
    try:
        session = df['Datetime'].dt.date
    except Exception:
        session = pd.Series(0, index=df.index)

    pv = (tp * vol).groupby(session).cumsum()
    cv = vol.groupby(session).cumsum().replace(0, np.nan)
    vwap = pv / cv
    return vwap.fillna(tp)


# ================================
# INDICATOR CALCULATIONS
# ================================

def calculate_all_indicators(df, config):
    """
    Calculate all technical indicators (TradingView-compatible).
    """
    ema_fast = int(config.get('ema_fast', DEFAULT_EMA_FAST))
    ema_slow = int(config.get('ema_slow', DEFAULT_EMA_SLOW))

    # ── EMAs: TradingView ta.ema (SMA seed + recursion) ───────────────────
    df['EMA_Fast'] = tv_ema(df['Close'], ema_fast)
    df['EMA_Slow'] = tv_ema(df['Close'], ema_slow)

    # SMA for custom strategy
    df['SMA_20'] = tv_sma(df['Close'], 20)
    df['SMA_50'] = tv_sma(df['Close'], 50)

    # EMA Angle
    df['EMA_Fast_Angle'] = calculate_ema_angle(df['EMA_Fast'])
    df['EMA_Slow_Angle'] = calculate_ema_angle(df['EMA_Slow'])

    # RSI / ADX / ATR
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['ADX'] = calculate_adx(df, int(config.get('adx_period', DEFAULT_ADX_PERIOD)))
    df['ATR'] = calculate_atr(df, 14)

    # Bollinger Bands (population stdev -> matches TV)
    bb_period = int(config.get('custom_bb_period', 20))
    bb_std = float(config.get('custom_bb_std', 2.0))
    df['BB_Middle'] = tv_sma(df['Close'], bb_period)
    bb_std_dev = tv_stdev(df['Close'], bb_period)
    df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)
    # aliases used by the Mean-Reversion strategy (previously missing -> never fired)
    df['Bollinger_Upper'] = df['BB_Upper']
    df['Bollinger_Middle'] = df['BB_Middle']
    df['Bollinger_Lower'] = df['BB_Lower']

    # MACD (TradingView default 12/26/9 on EMA)
    df['MACD'] = tv_ema(df['Close'], 12) - tv_ema(df['Close'], 26)
    df['MACD_Signal'] = tv_ema(df['MACD'], 9)
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Volume MA + session VWAP
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['VWAP'] = calculate_session_vwap(df)

    # ── Heikin Ashi ("NEW STRATEGIES") ────────────────────────────────────
    ha_o, ha_h, ha_l, ha_c = calculate_heikin_ashi(df)
    df['HA_Open'], df['HA_High'], df['HA_Low'], df['HA_Close'] = ha_o, ha_h, ha_l, ha_c
    df['HA_Bull'] = (df['HA_Close'] > df['HA_Open']).astype(int)

    # ── Donchian Channel ──────────────────────────────────────────────────
    dc_period = int(config.get('donchian_period', 20))
    df['DC_Upper'] = df['High'].rolling(dc_period).max()
    df['DC_Lower'] = df['Low'].rolling(dc_period).min()
    df['DC_Middle'] = (df['DC_Upper'] + df['DC_Lower']) / 2.0

    # ── Keltner Channel (EMA basis + ATR bands, TradingView default) ──────
    kc_period = int(config.get('keltner_period', 20))
    kc_mult = float(config.get('keltner_multiplier', 2.0))
    kc_atr_p = int(config.get('keltner_atr_period', 10))
    df['KC_Middle'] = tv_ema(df['Close'], kc_period)
    kc_atr = calculate_atr(df, kc_atr_p)
    df['KC_Upper'] = df['KC_Middle'] + kc_mult * kc_atr
    df['KC_Lower'] = df['KC_Middle'] - kc_mult * kc_atr

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


# =============================================================================
# STRATEGY FUNCTIONS
# =============================================================================
# All strategy functions share the signature (df, idx, config, current_position)
# and evaluate the candle at `idx` against the candle at `idx-1`.
#
# "CROSSOVER FIX":
#   * In BACKTESTING  idx walks over closed candles only, so idx / idx-1 is
#     exactly your prev/curr pair.
#   * In LIVE TRADING we now pass idx = len(df) - 2 (the last CLOSED candle)
#     instead of len(df) - 1 (the candle still being built). That is identical
#     to your df.iloc[-3] / df.iloc[-2] snippet and it is what stops the same
#     crossover from re-firing on every refresh.
#   * On top of that, live entries are de-duplicated per candle timestamp.
# =============================================================================

def detect_ema_crossover(df, idx, fast_col='EMA_Fast', slow_col='EMA_Slow'):
    """
    Return (is_bullish_cross, is_bearish_cross) for the candle at `idx`.

    BOTH conditions are always required - the "just now" crossover:
        bullish : prev_fast <= prev_slow  AND  curr_fast >  curr_slow
        bearish : prev_fast >= prev_slow  AND  curr_fast <  curr_slow

    Using only `curr_fast > curr_slow` (a state check instead of an event
    check) is what produced repeated entries in live trading.
    """
    if idx < 1:
        return False, False

    prev_fast = df[fast_col].iloc[idx - 1]
    prev_slow = df[slow_col].iloc[idx - 1]
    curr_fast = df[fast_col].iloc[idx]
    curr_slow = df[slow_col].iloc[idx]

    if pd.isna(prev_fast) or pd.isna(prev_slow) or pd.isna(curr_fast) or pd.isna(curr_slow):
        return False, False

    bullish = (prev_fast <= prev_slow) and (curr_fast > curr_slow)
    bearish = (prev_fast >= prev_slow) and (curr_fast < curr_slow)
    return bool(bullish), bool(bearish)


def check_ema_crossover_strategy(df, idx, config, current_position):
    """
    EMA Crossover Strategy with advanced filters.

    Filters:
    - Minimum Angle (ABSOLUTE value)
    - Entry Filters (Simple / Custom Candle Points / ATR-based)
    - ADX Filter (optional)
    """
    if idx < 1:
        return None, None

    current = df.iloc[idx]

    if pd.isna(current['EMA_Fast']) or pd.isna(current['EMA_Slow']):
        return None, None

    bullish_cross, bearish_cross = detect_ema_crossover(df, idx)

    if not bullish_cross and not bearish_cross:
        return None, None

    # Minimum Angle Filter (ABSOLUTE value)
    min_angle = float(config.get('ema_min_angle', DEFAULT_EMA_MIN_ANGLE))
    if min_angle > 0:
        fast_angle = abs(current['EMA_Fast_Angle']) if not pd.isna(current['EMA_Fast_Angle']) else 0
        if fast_angle < min_angle:
            return None, None

    # Entry Filter
    entry_filter = config.get('ema_entry_filter', 'Simple Crossover')

    if entry_filter == 'Custom Candle (Points)':
        min_points = config.get('ema_custom_candle_points', 5)
        candle_body = abs(current['Close'] - current['Open'])
        if candle_body < min_points:
            return None, None

    elif entry_filter == 'ATR-based Candle':
        if pd.isna(current['ATR']):
            return None, None
        atr_multiplier = config.get('ema_atr_multiplier', 0.3)
        min_body = current['ATR'] * atr_multiplier
        candle_body = abs(current['Close'] - current['Open'])
        if candle_body < min_body:
            return None, None

    # ADX Filter (optional)
    if config.get('ema_use_adx', DEFAULT_EMA_USE_ADX):
        adx_threshold = config.get('ema_adx_threshold', DEFAULT_EMA_ADX_THRESHOLD)
        if pd.isna(current['ADX']) or current['ADX'] < adx_threshold:
            return None, None

    if bullish_cross:
        return 'BUY', current['Close']
    if bearish_cross:
        return 'SELL', current['Close']

    return None, None


def check_simple_buy_strategy(df, idx, config, current_position):
    """Simple Buy strategy - always returns BUY immediately if no position"""
    if current_position is not None:
        return None, None
    return 'BUY', df.iloc[idx]['Close']


def check_simple_sell_strategy(df, idx, config, current_position):
    """Simple Sell strategy - always returns SELL immediately if no position"""
    if current_position is not None:
        return None, None
    return 'SELL', df.iloc[idx]['Close']


def check_price_crosses_threshold(df, idx, config, current_position):
    """Price crosses threshold strategy - checks current price state"""
    if current_position is not None:
        return None, None

    threshold = config.get('price_threshold', 25000)
    current_price = df.iloc[idx]['Close']
    cross_type = config.get('price_cross_type', 'Above Threshold')
    position_type = config.get('price_cross_position', 'LONG')

    condition_met = False
    if cross_type == 'Above Threshold':
        if current_price > threshold:
            condition_met = True
    else:
        if current_price < threshold:
            condition_met = True

    if condition_met:
        return ('BUY' if position_type == 'LONG' else 'SELL'), current_price

    return None, None


def check_rsi_adx_ema_combined(df, idx, config, current_position):
    """Combined RSI-ADX-EMA strategy"""
    if current_position is not None:
        return None, None
    if idx < 1:
        return None, None

    current = df.iloc[idx]

    if pd.isna(current['RSI']) or pd.isna(current['ADX']) or pd.isna(current['EMA_Fast']):
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
    """Percentage change strategy with full flexibility"""
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

    condition_met = False
    if 'Positive' in change_type:
        if pct_change >= threshold:
            condition_met = True
    else:
        if pct_change <= -threshold:
            condition_met = True

    if condition_met:
        return ('BUY' if position_type == 'LONG' else 'SELL'), current_price

    return None, None


def check_ai_price_action(df, idx, config, current_position):
    """AI Price Action - simplified pattern recognition"""
    if current_position is not None:
        return None, None
    if idx < 3:
        return None, None

    candles = df.iloc[idx - 2:idx + 1]

    if all(candles['Close'].diff().dropna() > 0):
        return 'BUY', df.iloc[idx]['Close']
    if all(candles['Close'].diff().dropna() < 0):
        return 'SELL', df.iloc[idx]['Close']

    return None, None


def check_custom_strategy(df, idx, config, current_position):
    """
    Custom Strategy Builder - multi-indicator, AND/OR combine logic.
    Reads config['custom_conditions'] list (one dict per condition).
    """
    if current_position is not None:
        return None, None
    if idx < 2:
        return None, None

    current = df.iloc[idx]
    previous = df.iloc[idx - 1]
    current_price = float(current["Close"])
    prev_price = float(previous["Close"])

    def _col(name, period, bb_std=2.0):
        key = f"_CUST_{name}_{period}_{bb_std}"
        if key not in df.columns:
            if name == "EMA":
                df[key] = tv_ema(df["Close"], int(period))
            elif name == "SMA":
                df[key] = tv_sma(df["Close"], int(period))
            elif name == "BB_U":
                mid = tv_sma(df["Close"], int(period))
                df[key] = mid + bb_std * tv_stdev(df["Close"], int(period))
            elif name == "BB_L":
                mid = tv_sma(df["Close"], int(period))
                df[key] = mid - bb_std * tv_stdev(df["Close"], int(period))
            elif name == "BB_M":
                df[key] = tv_sma(df["Close"], int(period))
            elif name == "ATR":
                df[key] = calculate_atr(df, int(period))
            elif name == "HV":
                df[key] = df["Close"].pct_change().rolling(window=int(period)).std() * (252 ** 0.5) * 100
            elif name == "STDDEV":
                df[key] = tv_stdev(df["Close"], int(period))
            elif name == "RSI":
                df[key] = calculate_rsi(df["Close"], int(period))
            elif name == "VOL_MA":
                df[key] = df["Volume"].rolling(window=int(period)).mean() if "Volume" in df.columns else float("nan")
        return key

    def _ca(cv, pv, lvl):
        return float(pv) <= float(lvl) and float(cv) > float(lvl)

    def _cb(cv, pv, lvl):
        return float(pv) >= float(lvl) and float(cv) < float(lvl)

    def _eval(c):
        stype = c.get("strategy_type", "Price Crosses Indicator")
        ind = c.get("indicator", "EMA")
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
            if ind not in imap:
                return False, ptype
            col = _col(*imap[ind])
            iv, piv = current.get(col, float("nan")), previous.get(col, float("nan"))
            if pd.isna(iv) or pd.isna(piv):
                return False, ptype
            triggered = _ca(current_price, prev_price, float(iv)) if cross == "Above Indicator" \
                else _cb(current_price, prev_price, float(iv))
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
            if ind not in imap:
                return False, ptype
            col = _col(*imap[ind])
            iv = current.get(col, float("nan"))
            if pd.isna(iv):
                return False, ptype
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
                if pd.isna(cv) or pd.isna(pv):
                    return False, ptype
                return chk(cv, pv, level), ptype

            elif ind == "MACD":
                cv, pv = current.get("MACD", float("nan")), previous.get("MACD", float("nan"))
                if pd.isna(cv) or pd.isna(pv):
                    return False, ptype
                return chk(cv, pv, level), ptype

            elif ind == "MACD Histogram":
                cv, pv = current.get("MACD_Hist", float("nan")), previous.get("MACD_Hist", float("nan"))
                if pd.isna(cv) or pd.isna(pv):
                    return False, ptype
                return chk(cv, pv, level), ptype

            elif ind == "ADX":
                cv, pv = current.get("ADX", float("nan")), previous.get("ADX", float("nan"))
                if pd.isna(cv) or pd.isna(pv):
                    return False, ptype
                return chk(cv, pv, level), ptype

            elif ind == "Volume":
                if "Volume" not in df.columns:
                    return False, ptype
                vcol = _col("VOL_MA", int(c.get("volume_ma_period", 20)))
                cv = float(current.get("Volume", 0))
                pv = float(previous.get("Volume", 0))
                vma = float(current.get(vcol, 1) or 1)
                thresh = vma * float(c.get("volume_multiplier", 1.5))
                return chk(cv, pv, thresh), ptype

            elif ind == "BB %B":
                bp = int(c.get("bb_period", 20))
                bstd = float(c.get("bb_std", 2.0))
                mid = tv_sma(df["Close"], bp)
                std = tv_stdev(df["Close"], bp)
                pctb_key = f"_CUST_PCTB_{bp}_{bstd}"
                df[pctb_key] = (df["Close"] - (mid - bstd * std)) / (2 * bstd * std)
                cv, pv = df[pctb_key].iloc[idx], df[pctb_key].iloc[idx - 1]
                if pd.isna(cv) or pd.isna(pv):
                    return False, ptype
                return chk(cv, pv, level / 100), ptype

            elif ind == "ATR (Volatility)":
                col = _col("ATR", int(c.get("atr_period", 14)))
                cv, pv = current.get(col, float("nan")), previous.get(col, float("nan"))
                if pd.isna(cv) or pd.isna(pv):
                    return False, ptype
                return chk(cv, pv, level), ptype

            elif ind == "Historical Volatility":
                col = _col("HV", int(c.get("hv_period", 20)))
                cv, pv = current.get(col, float("nan")), previous.get(col, float("nan"))
                if pd.isna(cv) or pd.isna(pv):
                    return False, ptype
                return chk(cv, pv, level), ptype

            elif ind == "Std Dev (Volatility)":
                col = _col("STDDEV", int(c.get("stddev_period", 20)))
                cv, pv = current.get(col, float("nan")), previous.get(col, float("nan"))
                if pd.isna(cv) or pd.isna(pv):
                    return False, ptype
                return chk(cv, pv, level), ptype

            return False, ptype

        elif stype == "Indicator Crossover":
            co = c.get("crossover_type", "Fast EMA x Slow EMA")
            is_bull = "Bullish" in cross

            def bull(fc, fp, sc, sp):
                return float(fp) <= float(sp) and float(fc) > float(sc)

            def bear(fc, fp, sc, sp):
                return float(fp) >= float(sp) and float(fc) < float(sc)

            chk2 = bull if is_bull else bear

            if co in ("Fast EMA x Slow EMA", "Fast EMA × Slow EMA"):
                fc_col = _col("EMA", int(c.get("fast_ema", 9)))
                sc_col = _col("EMA", int(c.get("slow_ema", 21)))
                vals = [current.get(fc_col), previous.get(fc_col), current.get(sc_col), previous.get(sc_col)]
                if any(pd.isna(v) for v in vals):
                    return False, ptype
                return chk2(*vals), ptype

            elif co in ("Fast SMA x Slow SMA", "Fast SMA × Slow SMA"):
                fc_col = _col("SMA", int(c.get("fast_sma", 20)))
                sc_col = _col("SMA", int(c.get("slow_sma", 50)))
                vals = [current.get(fc_col), previous.get(fc_col), current.get(sc_col), previous.get(sc_col)]
                if any(pd.isna(v) for v in vals):
                    return False, ptype
                return chk2(*vals), ptype

            elif co in ("MACD x Signal", "MACD × Signal"):
                mc, mp = current.get("MACD", float("nan")), previous.get("MACD", float("nan"))
                sc2, sp2 = current.get("MACD_Signal", float("nan")), previous.get("MACD_Signal", float("nan"))
                if any(pd.isna(v) for v in [mc, mp, sc2, sp2]):
                    return False, ptype
                return chk2(mc, mp, sc2, sp2), ptype

            elif co in ("Price x EMA", "Price × EMA"):
                col = _col("EMA", int(c.get("ma_period", 50)))
                ic, ip = current.get(col, float("nan")), previous.get(col, float("nan"))
                if pd.isna(ic) or pd.isna(ip):
                    return False, ptype
                return chk2(current_price, prev_price, float(ic), float(ip)), ptype

            elif co in ("Price x SMA", "Price × SMA"):
                col = _col("SMA", int(c.get("ma_period", 50)))
                ic, ip = current.get(col, float("nan")), previous.get(col, float("nan"))
                if pd.isna(ic) or pd.isna(ip):
                    return False, ptype
                return chk2(current_price, prev_price, float(ic), float(ip)), ptype

            elif co == "RSI Crossover (Overbought/Oversold)":
                rsi_col = _col("RSI", int(c.get("rsi_period", 14)))
                rc, rp = df[rsi_col].iloc[idx], df[rsi_col].iloc[idx - 1]
                if pd.isna(rc) or pd.isna(rp):
                    return False, ptype
                ob, os_lvl = float(c.get("rsi_ob", 70)), float(c.get("rsi_os", 30))
                if is_bull:
                    return (float(rp) <= os_lvl and float(rc) > os_lvl), ptype
                else:
                    return (float(rp) <= ob and float(rc) > ob), ptype

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
    results = [_eval(c) for c in conditions]
    flags = [r[0] for r in results]
    ptypes = [r[1] for r in results]

    final = all(flags) if use_and else any(flags)
    if not final:
        return None, None

    for flag, ptype in zip(flags, ptypes):
        if flag:
            return ("BUY" if ptype == "LONG" else "SELL"), current_price

    return None, None


def check_supertrend_ai(df, idx, config, current_position):
    """SuperTrend AI Strategy - trend-following with ADX + volume confirmation"""
    if current_position is not None:
        return None, None
    if idx < 20:
        return None, None

    atr_period = config.get('supertrend_atr_period', 10)
    multiplier = config.get('supertrend_multiplier', 3.0)
    adx_threshold = config.get('supertrend_adx_threshold', 25)
    volume_mult = config.get('supertrend_volume_mult', 1.5)

    if 'SuperTrend' not in df.columns or 'SuperTrend_Direction' not in df.columns:
        df['ATR_ST'] = calculate_atr(df, atr_period)

        hl_avg = (df['High'] + df['Low']) / 2
        df['ST_Upper'] = hl_avg + (multiplier * df['ATR_ST'])
        df['ST_Lower'] = hl_avg - (multiplier * df['ATR_ST'])

        supertrend = []
        direction = []

        for i in range(len(df)):
            if i == 0:
                supertrend.append(df['ST_Lower'].iloc[i])
                direction.append(1)
            else:
                prev_st = supertrend[i - 1]
                prev_dir = direction[i - 1]
                close = df['Close'].iloc[i]
                upper = df['ST_Upper'].iloc[i]
                lower = df['ST_Lower'].iloc[i]

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

    curr_dir = current['SuperTrend_Direction']
    prev_dir = previous['SuperTrend_Direction']

    adx = current.get('ADX', 0)
    if pd.isna(adx) or adx < adx_threshold:
        return None, None

    if 'Volume' in df.columns:
        vol_ma = df['Volume'].rolling(20).mean().iloc[idx]
        if pd.notna(vol_ma) and vol_ma > 0 and current['Volume'] < vol_ma * volume_mult:
            return None, None

    if prev_dir == -1 and curr_dir == 1:
        return 'BUY', current['Close']
    if prev_dir == 1 and curr_dir == -1:
        return 'SELL', current['Close']

    return None, None


def check_vwap_volume_spike(df, idx, config, current_position):
    """VWAP + Volume Spike Strategy"""
    if current_position is not None:
        return None, None
    if idx < 50:
        return None, None

    volume_mult = config.get('vwap_volume_mult', 2.0)
    vwap_distance = config.get('vwap_distance_pct', 0.3)

    if 'VWAP' not in df.columns:
        df['VWAP'] = calculate_session_vwap(df)

    current = df.iloc[idx]
    previous = df.iloc[idx - 1]
    current_price = current['Close']
    prev_price = previous['Close']
    vwap = current['VWAP']

    vol_ma = df['Volume'].rolling(20).mean().iloc[idx]
    if pd.isna(vol_ma) or vol_ma <= 0 or current['Volume'] < vol_ma * volume_mult:
        return None, None

    distance_pct = abs(current_price - vwap) / vwap * 100
    if distance_pct > vwap_distance:
        return None, None

    rsi = current.get('RSI', 50)

    if prev_price < previous['VWAP'] and current_price > vwap:
        if not pd.isna(rsi) and rsi < 55:
            return 'BUY', current_price

    if prev_price > previous['VWAP'] and current_price < vwap:
        if not pd.isna(rsi) and rsi > 45:
            return 'SELL', current_price

    return None, None


def check_bollinger_squeeze_breakout(df, idx, config, current_position):
    """Bollinger Band Squeeze Breakout"""
    if current_position is not None:
        return None, None
    if idx < 30:
        return None, None

    bb_period = config.get('bb_squeeze_period', 20)
    bb_std = config.get('bb_squeeze_std', 2.0)
    squeeze_threshold = config.get('bb_squeeze_threshold', 0.02)
    volume_mult = config.get('bb_squeeze_volume_mult', 1.8)

    if f'BB_Upper_{bb_period}' not in df.columns:
        bb_mid = tv_sma(df['Close'], bb_period)
        bb_std_val = tv_stdev(df['Close'], bb_period)
        df[f'BB_Upper_{bb_period}'] = bb_mid + (bb_std * bb_std_val)
        df[f'BB_Lower_{bb_period}'] = bb_mid - (bb_std * bb_std_val)
        df[f'BB_Mid_{bb_period}'] = bb_mid
        df[f'BB_Bandwidth_{bb_period}'] = (
            df[f'BB_Upper_{bb_period}'] - df[f'BB_Lower_{bb_period}']) / df[f'BB_Mid_{bb_period}']

    current = df.iloc[idx]
    previous = df.iloc[idx - 1]
    current_price = current['Close']
    prev_price = previous['Close']

    bb_upper = current[f'BB_Upper_{bb_period}']
    bb_lower = current[f'BB_Lower_{bb_period}']
    bandwidth = current[f'BB_Bandwidth_{bb_period}']
    prev_bandwidth = previous[f'BB_Bandwidth_{bb_period}']

    if pd.isna(bandwidth) or pd.isna(prev_bandwidth):
        return None, None

    is_squeezed = bandwidth < squeeze_threshold
    was_squeezed = prev_bandwidth < squeeze_threshold

    vol_ma = df['Volume'].rolling(20).mean().iloc[idx]
    volume_surge = pd.notna(vol_ma) and vol_ma > 0 and current['Volume'] > vol_ma * volume_mult

    if (is_squeezed or was_squeezed) and volume_surge:
        if prev_price <= previous[f'BB_Upper_{bb_period}'] and current_price > bb_upper:
            rsi = current.get('RSI', 50)
            if pd.isna(rsi) or rsi < 75:
                return 'BUY', current_price

    if (is_squeezed or was_squeezed) and volume_surge:
        if prev_price >= previous[f'BB_Lower_{bb_period}'] and current_price < bb_lower:
            rsi = current.get('RSI', 50)
            if pd.isna(rsi) or rsi > 25:
                return 'SELL', current_price

    return None, None


def check_elliott_waves_ratio_charts(df, idx, config, current_position):
    """Elliott Waves Strategy (simplified with argrelextrema)"""
    if current_position is not None:
        return None, None

    wave_lookback = config.get('elliott_wave_lookback', 50)

    if idx < wave_lookback:
        return None, None

    if 'Wave_Extrema' not in df.columns:
        df['Wave_Extrema'] = 0
        if len(df) >= wave_lookback:
            highs_idx = argrelextrema(df['High'].values, np.greater, order=5)[0]
            lows_idx = argrelextrema(df['Low'].values, np.less, order=5)[0]

            for h_idx in highs_idx:
                if h_idx < len(df):
                    df.iloc[h_idx, df.columns.get_loc('Wave_Extrema')] = 1
            for l_idx in lows_idx:
                if l_idx < len(df):
                    df.iloc[l_idx, df.columns.get_loc('Wave_Extrema')] = 1

    recent_start = max(0, idx - wave_lookback)
    recent = df.iloc[recent_start:idx + 1]

    extrema_mask = recent['Wave_Extrema'] == 1
    extrema_indices = recent[extrema_mask].index.tolist()

    bullish = False
    bearish = False

    if len(extrema_indices) >= 5:
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
    """Opening Range Breakout (ORB) Strategy"""
    if current_position is not None:
        return None, None
    if idx < 30:
        return None, None

    orb_minutes = config.get('orb_minutes', 15)
    breakout_buffer = config.get('orb_breakout_buffer', 0.1)

    if 'Datetime' not in df.columns:
        return None, None

    current_time = df.iloc[idx]['Datetime']
    if not hasattr(current_time, 'time'):
        return None, None

    market_open_time = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
    orb_end_time = market_open_time + timedelta(minutes=orb_minutes)

    if current_time < orb_end_time:
        return None, None

    orb_mask = (df['Datetime'] >= market_open_time) & (df['Datetime'] <= orb_end_time)
    orb_candles = df[orb_mask]

    if len(orb_candles) == 0:
        return None, None

    orb_high = orb_candles['High'].max()
    orb_low = orb_candles['Low'].min()

    breakout_high = orb_high * (1 + breakout_buffer / 100)
    breakout_low = orb_low * (1 - breakout_buffer / 100)

    current_price = df.iloc[idx]['Close']
    previous_price = df.iloc[idx - 1]['Close']

    if previous_price <= orb_high and current_price > breakout_high:
        return 'BUY', current_price
    if previous_price >= orb_low and current_price < breakout_low:
        return 'SELL', current_price

    return None, None


def check_pivot_point_reversal(df, idx, config, current_position):
    """Pivot Point Reversal Strategy"""
    if current_position is not None:
        return None, None
    if idx < 2:
        return None, None

    lookback = config.get('pivot_lookback', 24)
    if idx < lookback:
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
    """Ichimoku Cloud Strategy - TK cross + cloud confirmation"""
    if current_position is not None:
        return None, None
    if idx < 52:
        return None, None

    if 'Ichimoku_Tenkan' not in df.columns:
        t_len = int(config.get('ichimoku_tenkan', 9))
        k_len = int(config.get('ichimoku_kijun', 26))
        b_len = int(config.get('ichimoku_senkou_b', 52))
        disp = int(config.get('ichimoku_displacement', 26))

        high_t = df['High'].rolling(window=t_len).max()
        low_t = df['Low'].rolling(window=t_len).min()
        df['Ichimoku_Tenkan'] = (high_t + low_t) / 2

        high_k = df['High'].rolling(window=k_len).max()
        low_k = df['Low'].rolling(window=k_len).min()
        df['Ichimoku_Kijun'] = (high_k + low_k) / 2

        df['Ichimoku_SpanA'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(disp)

        high_b = df['High'].rolling(window=b_len).max()
        low_b = df['Low'].rolling(window=b_len).min()
        df['Ichimoku_SpanB'] = ((high_b + low_b) / 2).shift(disp)

        df['Ichimoku_Chikou'] = df['Close'].shift(-disp)

    current = df.iloc[idx]
    current_price = current['Close']

    tenkan = current.get('Ichimoku_Tenkan')
    kijun = current.get('Ichimoku_Kijun')
    span_a = current.get('Ichimoku_SpanA')
    span_b = current.get('Ichimoku_SpanB')

    if pd.isna(tenkan) or pd.isna(kijun) or pd.isna(span_a) or pd.isna(span_b):
        return None, None

    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)

    previous = df.iloc[idx - 1]
    prev_tenkan = previous.get('Ichimoku_Tenkan')
    prev_kijun = previous.get('Ichimoku_Kijun')

    if not pd.isna(prev_tenkan) and not pd.isna(prev_kijun):
        if prev_tenkan <= prev_kijun and tenkan > kijun:
            if current_price > cloud_top:
                return 'BUY', current_price
        if prev_tenkan >= prev_kijun and tenkan < kijun:
            if current_price < cloud_bottom:
                return 'SELL', current_price

    return None, None


def check_volume_breakout(df, idx, config, current_position):
    """Volume Breakout Strategy"""
    if current_position is not None:
        return None, None
    if idx < 20:
        return None, None

    volume_multiplier = config.get('volume_multiplier', 2.0)
    price_change_threshold = config.get('volume_price_threshold', 0.5)

    if 'Volume_MA' not in df.columns:
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()

    current = df.iloc[idx]
    current_price = current['Close']
    current_volume = current['Volume']
    volume_ma = current.get('Volume_MA', 0)

    if volume_ma == 0 or pd.isna(volume_ma):
        return None, None

    if current_volume < volume_ma * volume_multiplier:
        return None, None

    open_price = current['Open']
    price_change_pct = abs(current_price - open_price) / open_price * 100

    if price_change_pct < price_change_threshold:
        return None, None

    rsi = current.get('RSI', 50)
    prev_close = df.iloc[idx - 1]['Close']

    if current_price > open_price and rsi < 70 and current_price > prev_close:
        return 'BUY', current_price
    if current_price < open_price and rsi > 30 and current_price < prev_close:
        return 'SELL', current_price

    return None, None


def check_gap_trading(df, idx, config, current_position):
    """Gap Trading Strategy - gap fill trades in the first hour"""
    if current_position is not None:
        return None, None
    if idx < 2:
        return None, None

    min_gap_percent = config.get('gap_min_percent', 0.5)
    max_gap_percent = config.get('gap_max_percent', 3.0)

    current = df.iloc[idx]
    previous = df.iloc[idx - 1]

    current_open = current['Open']
    prev_close = previous['Close']
    current_price = current['Close']

    gap_percent = abs(current_open - prev_close) / prev_close * 100

    if gap_percent < min_gap_percent or gap_percent > max_gap_percent:
        return None, None

    if 'Datetime' in df.columns:
        current_time = current['Datetime']
        if hasattr(current_time, 'time'):
            market_open = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
            first_hour_end = market_open + timedelta(hours=1)
            if current_time > first_hour_end:
                return None, None

    if current_open > prev_close:
        gap_up_percent = (current_open - prev_close) / prev_close * 100
        if min_gap_percent <= gap_up_percent <= max_gap_percent:
            if current_price < current_open and idx >= 20:
                volume_ma = df.iloc[idx - 20:idx]['Volume'].mean()
                if current['Volume'] > volume_ma:
                    return 'SELL', current_price

    elif current_open < prev_close:
        gap_down_percent = (prev_close - current_open) / prev_close * 100
        if min_gap_percent <= gap_down_percent <= max_gap_percent:
            if current_price > current_open and idx >= 20:
                volume_ma = df.iloc[idx - 20:idx]['Volume'].mean()
                if current['Volume'] > volume_ma:
                    return 'BUY', current_price

    return None, None


def check_mean_reversion_bollinger(df, idx, config, current_position):
    """Mean Reversion with Bollinger Bands"""
    if current_position is not None:
        return None, None
    if idx < 2:
        return None, None

    rsi_oversold = config.get('mr_rsi_oversold', 30)
    rsi_overbought = config.get('mr_rsi_overbought', 70)

    if 'Bollinger_Upper' not in df.columns or 'Bollinger_Lower' not in df.columns:
        return None, None

    current = df.iloc[idx]
    previous = df.iloc[idx - 1]

    current_price = current['Close']
    bb_upper = current.get('Bollinger_Upper')
    bb_lower = current.get('Bollinger_Lower')
    bb_middle = current.get('Bollinger_Middle')
    rsi = current.get('RSI', 50)

    if pd.isna(bb_upper) or pd.isna(bb_lower) or pd.isna(bb_middle):
        return None, None

    prev_price = previous['Close']

    if prev_price <= previous['Bollinger_Lower'] and current_price > bb_lower and rsi < rsi_oversold:
        if current_price > prev_price:
            return 'BUY', current_price

    if prev_price >= previous['Bollinger_Upper'] and current_price < bb_upper and rsi > rsi_overbought:
        if current_price < prev_price:
            return 'SELL', current_price

    return None, None


def check_momentum_breakout_adx(df, idx, config, current_position):
    """Momentum Breakout with ADX"""
    if current_position is not None:
        return None, None
    if idx < 50:
        return None, None

    adx_threshold = config.get('momentum_adx_threshold', 25)
    breakout_lookback = config.get('momentum_lookback', 20)
    min_volume_ratio = config.get('momentum_volume_ratio', 1.5)

    current = df.iloc[idx]
    current_price = current['Close']
    adx = current.get('ADX', 0)
    rsi = current.get('RSI', 50)

    if pd.isna(adx) or adx < adx_threshold:
        return None, None

    recent_data = df.iloc[idx - breakout_lookback:idx]
    recent_high = recent_data['High'].max()
    recent_low = recent_data['Low'].min()

    current_volume = current['Volume']
    avg_volume = recent_data['Volume'].mean()

    if pd.notna(avg_volume) and avg_volume > 0 and current_volume < avg_volume * min_volume_ratio:
        return None, None

    prev_close = df.iloc[idx - 1]['Close']

    if current_price > recent_high and rsi < 70 and prev_close <= recent_high:
        return 'BUY', current_price
    if current_price < recent_low and rsi > 30 and prev_close >= recent_low:
        return 'SELL', current_price

    return None, None


def check_support_resistance_bounce(df, idx, config, current_position):
    """Support Resistance Bounce Strategy"""
    if current_position is not None:
        return None, None
    if idx < 100:
        return None, None

    sr_lookback = config.get('sr_lookback', 100)
    sr_tolerance = config.get('sr_tolerance', 0.002)
    min_touches = config.get('sr_min_touches', 3)

    current = df.iloc[idx]
    current_price = current['Close']
    current_high = current['High']
    current_low = current['Low']

    lookback_data = df.iloc[max(0, idx - sr_lookback):idx]

    swing_highs = []
    swing_lows = []

    for i in range(2, len(lookback_data) - 2):
        high = lookback_data.iloc[i]['High']
        low = lookback_data.iloc[i]['Low']

        if (high > lookback_data.iloc[i - 1]['High'] and
                high > lookback_data.iloc[i - 2]['High'] and
                high > lookback_data.iloc[i + 1]['High'] and
                high > lookback_data.iloc[i + 2]['High']):
            swing_highs.append(high)

        if (low < lookback_data.iloc[i - 1]['Low'] and
                low < lookback_data.iloc[i - 2]['Low'] and
                low < lookback_data.iloc[i + 1]['Low'] and
                low < lookback_data.iloc[i + 2]['Low']):
            swing_lows.append(low)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None, None

    def find_key_level(levels, current_level, tolerance):
        clusters = [lv for lv in levels if abs(lv - current_level) / current_level <= tolerance]
        if len(clusters) >= min_touches:
            return sum(clusters) / len(clusters)
        return None

    support_level = find_key_level(swing_lows, current_low, sr_tolerance)
    if support_level is not None:
        if current_low <= support_level * (1 + sr_tolerance) and current_price > current_low:
            rsi = current.get('RSI', 50)
            if rsi < 50:
                return 'BUY', current_price

    resistance_level = find_key_level(swing_highs, current_high, sr_tolerance)
    if resistance_level is not None:
        if current_high >= resistance_level * (1 - sr_tolerance) and current_price < current_high:
            rsi = current.get('RSI', 50)
            if rsi > 50:
                return 'SELL', current_price

    return None, None


# =============================================================================
# "NEW STRATEGIES"  (v2.0)
# =============================================================================

def check_donchian_channel(df, idx, config, current_position):
    """
    Donchian Channel Breakout ("Turtle" style).

    Entry rule (event based, not state based):
        LONG  : close crosses ABOVE the highest high of the previous N bars
        SHORT : close crosses BELOW the lowest  low  of the previous N bars

    The channel is measured on bars [idx-N .. idx-1] (i.e. EXCLUDING the
    current bar), which is how Turtle/Donchian breakouts are actually defined -
    including the current bar makes a breakout impossible to detect.

    Optional "Mean Reversion" mode fades the breakout instead of following it.
    """
    if current_position is not None:
        return None, None

    period = int(config.get('donchian_period', 20))
    if idx < period + 1:
        return None, None

    window = df.iloc[idx - period:idx]
    upper = window['High'].max()
    lower = window['Low'].min()

    if pd.isna(upper) or pd.isna(lower):
        return None, None

    current = df.iloc[idx]
    previous = df.iloc[idx - 1]
    price = current['Close']
    prev_price = previous['Close']

    mode = config.get('donchian_mode', 'Breakout (Trend Following)')

    breakout_up = (prev_price <= upper) and (price > upper)
    breakout_dn = (prev_price >= lower) and (price < lower)

    # optional ADX trend filter
    if config.get('donchian_use_adx', False):
        adx = current.get('ADX', np.nan)
        if pd.isna(adx) or adx < float(config.get('donchian_adx_threshold', 20)):
            return None, None

    if mode.startswith('Breakout'):
        if breakout_up:
            return 'BUY', price
        if breakout_dn:
            return 'SELL', price
    else:  # Mean Reversion - fade the extremes
        if breakout_up:
            return 'SELL', price
        if breakout_dn:
            return 'BUY', price

    return None, None


def check_keltner_channel(df, idx, config, current_position):
    """
    Keltner Channel Breakout.

        basis = EMA(close, period)          (TradingView default)
        band  = basis +/- multiplier * ATR(atr_period)

    LONG  : close crosses ABOVE the upper band
    SHORT : close crosses BELOW the lower band
    ("Mean Reversion" mode trades the snap-back to the basis instead.)
    """
    if current_position is not None:
        return None, None
    if idx < 2:
        return None, None

    if 'KC_Upper' not in df.columns:
        period = int(config.get('keltner_period', 20))
        mult = float(config.get('keltner_multiplier', 2.0))
        atr_p = int(config.get('keltner_atr_period', 10))
        df['KC_Middle'] = tv_ema(df['Close'], period)
        kc_atr = calculate_atr(df, atr_p)
        df['KC_Upper'] = df['KC_Middle'] + mult * kc_atr
        df['KC_Lower'] = df['KC_Middle'] - mult * kc_atr

    current = df.iloc[idx]
    previous = df.iloc[idx - 1]

    up, lo = current.get('KC_Upper'), current.get('KC_Lower')
    p_up, p_lo = previous.get('KC_Upper'), previous.get('KC_Lower')

    if pd.isna(up) or pd.isna(lo) or pd.isna(p_up) or pd.isna(p_lo):
        return None, None

    price = current['Close']
    prev_price = previous['Close']

    cross_up = (prev_price <= p_up) and (price > up)
    cross_dn = (prev_price >= p_lo) and (price < lo)

    mode = config.get('keltner_mode', 'Breakout (Trend Following)')

    if mode.startswith('Breakout'):
        if cross_up:
            return 'BUY', price
        if cross_dn:
            return 'SELL', price
    else:
        # Mean reversion: buy when price re-enters from below the lower band
        re_enter_up = (prev_price < p_lo) and (price > lo)
        re_enter_dn = (prev_price > p_up) and (price < up)
        if re_enter_up:
            return 'BUY', price
        if re_enter_dn:
            return 'SELL', price

    return None, None


def check_heikin_ashi_trend(df, idx, config, current_position):
    """
    Heikin Ashi Trend Flip.

    LONG  : HA candle flips from red to green and stays green for
            `ha_confirm_bars` consecutive candles.
    SHORT : mirror image.

    Optional "strong candle" filter requires a flat-bottom (no lower wick) for
    longs and a flat-top (no upper wick) for shorts - the classic HA
    trend-continuation signal.
    """
    if current_position is not None:
        return None, None

    confirm = int(config.get('ha_confirm_bars', 2))
    if idx < confirm + 1:
        return None, None

    if 'HA_Bull' not in df.columns:
        ha_o, ha_h, ha_l, ha_c = calculate_heikin_ashi(df)
        df['HA_Open'], df['HA_High'], df['HA_Low'], df['HA_Close'] = ha_o, ha_h, ha_l, ha_c
        df['HA_Bull'] = (df['HA_Close'] > df['HA_Open']).astype(int)

    bulls = df['HA_Bull'].iloc[idx - confirm:idx + 1].tolist()
    prior = int(df['HA_Bull'].iloc[idx - confirm - 1])

    current = df.iloc[idx]
    price = current['Close']

    strong_only = config.get('ha_strong_candle_only', False)

    # bullish flip: prior candle bearish, then `confirm+1` bullish candles
    if prior == 0 and all(b == 1 for b in bulls):
        if strong_only:
            if not np.isclose(current['HA_Open'], current['HA_Low'], rtol=0, atol=1e-6):
                return None, None
        return 'BUY', price

    if prior == 1 and all(b == 0 for b in bulls):
        if strong_only:
            if not np.isclose(current['HA_Open'], current['HA_High'], rtol=0, atol=1e-6):
                return None, None
        return 'SELL', price

    return None, None


def check_heikin_ashi_ema(df, idx, config, current_position):
    """
    Heikin Ashi + EMA Confirmation.

    Takes the HA colour flip but only in the direction of the EMA trend:
        LONG  : HA flips green AND EMA_Fast > EMA_Slow AND HA_Close > EMA_Fast
        SHORT : HA flips red   AND EMA_Fast < EMA_Slow AND HA_Close < EMA_Fast
    A much lower-frequency, higher-quality version of the plain HA flip.
    """
    if current_position is not None:
        return None, None
    if idx < 2:
        return None, None

    if 'HA_Bull' not in df.columns:
        ha_o, ha_h, ha_l, ha_c = calculate_heikin_ashi(df)
        df['HA_Open'], df['HA_High'], df['HA_Low'], df['HA_Close'] = ha_o, ha_h, ha_l, ha_c
        df['HA_Bull'] = (df['HA_Close'] > df['HA_Open']).astype(int)

    current = df.iloc[idx]
    previous = df.iloc[idx - 1]

    ef, es = current.get('EMA_Fast'), current.get('EMA_Slow')
    if pd.isna(ef) or pd.isna(es):
        return None, None

    flipped_bull = (previous['HA_Bull'] == 0) and (current['HA_Bull'] == 1)
    flipped_bear = (previous['HA_Bull'] == 1) and (current['HA_Bull'] == 0)

    price = current['Close']

    if flipped_bull and ef > es and current['HA_Close'] > ef:
        return 'BUY', price
    if flipped_bear and ef < es and current['HA_Close'] < ef:
        return 'SELL', price

    return None, None


def check_macd_crossover(df, idx, config, current_position):
    """
    MACD line x Signal line crossover (TradingView 12/26/9 by default).

    Same "event, not state" discipline as the EMA crossover:
        bullish : prev MACD <= prev Signal AND curr MACD > curr Signal
    Optional filter: only take crosses that happen below zero for longs /
    above zero for shorts (classic momentum-reset entries).
    """
    if current_position is not None:
        return None, None
    if idx < 2:
        return None, None

    fast = int(config.get('macd_fast', 12))
    slow = int(config.get('macd_slow', 26))
    sig = int(config.get('macd_signal', 9))

    col_m = f"_MACD_{fast}_{slow}"
    col_s = f"_MACDSIG_{fast}_{slow}_{sig}"
    if col_m not in df.columns:
        df[col_m] = tv_ema(df['Close'], fast) - tv_ema(df['Close'], slow)
        df[col_s] = tv_ema(df[col_m], sig)

    m_c, m_p = df[col_m].iloc[idx], df[col_m].iloc[idx - 1]
    s_c, s_p = df[col_s].iloc[idx], df[col_s].iloc[idx - 1]

    if pd.isna(m_c) or pd.isna(m_p) or pd.isna(s_c) or pd.isna(s_p):
        return None, None

    price = df.iloc[idx]['Close']
    zero_filter = config.get('macd_zero_filter', False)

    bullish = (m_p <= s_p) and (m_c > s_c)
    bearish = (m_p >= s_p) and (m_c < s_c)

    if bullish and (not zero_filter or m_c < 0):
        return 'BUY', price
    if bearish and (not zero_filter or m_c > 0):
        return 'SELL', price

    return None, None


# ================================
# HELPER FUNCTIONS
# ================================

def is_within_trade_window(timestamp, config):
    """True if trade window is disabled or timestamp lies inside the window."""
    if not config.get('use_trade_window', False):
        return True

    try:
        if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
            ts_ist = timestamp.astimezone(IST)
        else:
            ts_ist = IST.localize(timestamp)

        start_time = config.get('trade_window_start', dt_time(9, 30))
        end_time = config.get('trade_window_end', dt_time(15, 0))

        current_time = ts_ist.time()
        return start_time <= current_time <= end_time
    except Exception:
        return True


def should_allow_trade_direction(signal, config):
    """Check if signal matches allowed trade direction filter."""
    direction_filter = config.get('trade_direction', 'Both (LONG + SHORT)')

    if direction_filter == 'Both (LONG + SHORT)':
        return True
    elif direction_filter == 'LONG Only':
        return signal in ('BUY', 'LONG')
    elif direction_filter == 'SHORT Only':
        return signal in ('SELL', 'SHORT')

    return True


def get_entry_position_type(signal, config):
    """Convert a raw strategy signal into the actual entry position type."""
    position_type = 'LONG' if signal in ('BUY', 'LONG') else 'SHORT'

    if config.get('reverse_entry', False):
        position_type = 'SHORT' if position_type == 'LONG' else 'LONG'

    return position_type


def calculate_brokerage(entry_price, exit_price, quantity, config):
    """Calculate brokerage for one round trip."""
    if not config.get('include_brokerage', False):
        return 0.0

    brokerage_type = config.get('brokerage_type', 'Fixed per Trade')

    if brokerage_type == 'Fixed per Trade':
        return float(config.get('brokerage_per_trade', 20.0))
    else:
        turnover = (entry_price + exit_price) * quantity
        brokerage_pct = float(config.get('brokerage_percentage', 0.03)) / 100
        return turnover * brokerage_pct


def build_trade_record(position, exit_price, exit_reason, config, exit_time=None):
    """
    "MANUAL SQUAREOFF FIX"
    Single source of truth for a completed-trade dict.

    Previously the manual-close button wrote a record that was MISSING the
    'brokerage' / 'net_pnl' / 'ticker' keys. The Completed-Trades table then
    tried `df['net_pnl'].str.replace(...).astype(float)` on a column that
    contained an em-dash for that row -> ValueError -> the whole Live Trading
    tab crashed and the auto-refresh loop died. Every exit path now builds the
    record through this function, so the columns are always consistent.
    """
    exit_time = exit_time or now_ist()
    qty = position.get('quantity', 1)
    entry_price = position['entry_price']

    if position['type'] == 'LONG':
        pnl = (exit_price - entry_price) * qty
    else:
        pnl = (entry_price - exit_price) * qty

    brokerage = calculate_brokerage(entry_price, exit_price, qty, config)

    try:
        duration_minutes = (exit_time - position['entry_time']).total_seconds() / 60.0
    except Exception:
        duration_minutes = np.nan

    highest = position.get('highest_price', exit_price)
    lowest = position.get('lowest_price', exit_price)

    return {
        'entry_time': position['entry_time'],
        'exit_time': exit_time,
        'ticker': position.get('ticker', config.get('asset', 'Unknown')),
        'type': position['type'],
        'entry_price': float(entry_price),
        'exit_price': float(exit_price),
        'sl_price': position.get('sl_price'),
        'target_price': position.get('target_price'),
        'highest_price': highest,
        'lowest_price': lowest,
        'price_range': (highest - lowest) if (highest is not None and lowest is not None) else np.nan,
        'quantity': qty,
        'pnl': float(pnl),
        'brokerage': float(brokerage),
        'net_pnl': float(pnl - brokerage),
        'exit_reason': exit_reason,
        'duration_minutes': duration_minutes,
        'price_change_pct': abs(exit_price - entry_price) / entry_price * 100 if entry_price else 0.0,
    }


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
    # ---- NEW STRATEGIES ----
    'Donchian Channel Breakout': check_donchian_channel,
    'Keltner Channel Breakout': check_keltner_channel,
    'Heikin Ashi Trend Flip': check_heikin_ashi_trend,
    'Heikin Ashi + EMA Confirmation': check_heikin_ashi_ema,
    'MACD Signal Crossover': check_macd_crossover,
}


# ================================
# STOP LOSS CALCULATION
# ================================

def calculate_initial_sl(position_type, entry_price, df, idx, config):
    """Calculate initial stop loss based on SL type."""
    sl_type = config.get('sl_type', DEFAULT_SL_TYPE)
    current = df.iloc[idx]

    def _fallback():
        points = config.get('sl_points', DEFAULT_SL_POINTS)
        return entry_price - points if position_type == 'LONG' else entry_price + points

    if sl_type == 'Custom Points':
        points = config.get('sl_points', DEFAULT_SL_POINTS)
        return entry_price - points if position_type == 'LONG' else entry_price + points

    elif sl_type == 'P&L Based (Rupees)':
        rupees = config.get('sl_rupees', DEFAULT_SL_RUPEES)
        quantity = config.get('quantity', 1)
        points = rupees / max(quantity, 1)
        return entry_price - points if position_type == 'LONG' else entry_price + points

    elif sl_type == 'ATR-based':
        if pd.isna(current['ATR']):
            return _fallback()
        multiplier = config.get('sl_atr_multiplier', DEFAULT_SL_ATR_MULT)
        sl_distance = current['ATR'] * multiplier
        return entry_price - sl_distance if position_type == 'LONG' else entry_price + sl_distance

    elif sl_type == 'Current Candle Low/High':
        return current['Low'] if position_type == 'LONG' else current['High']

    elif sl_type == 'Previous Candle Low/High':
        if pd.isna(current['Prev_Low']) or pd.isna(current['Prev_High']):
            return _fallback()
        return current['Prev_Low'] if position_type == 'LONG' else current['Prev_High']

    elif sl_type == 'Current Swing Low/High':
        if pd.isna(current['Swing_Low']) or pd.isna(current['Swing_High']):
            return _fallback()
        return current['Swing_Low'] if position_type == 'LONG' else current['Swing_High']

    elif sl_type == 'Previous Swing Low/High':
        if pd.isna(current['Prev_Swing_Low']) or pd.isna(current['Prev_Swing_High']):
            return _fallback()
        return current['Prev_Swing_Low'] if position_type == 'LONG' else current['Prev_Swing_High']

    elif sl_type == 'Signal-based (Reverse Crossover)':
        return None

    elif sl_type == 'Strategy-based Signal':
        return None

    elif sl_type in ['Trailing SL (Points)', 'Trailing Profit (Rupees)', 'Trailing Loss (Rupees)',
                     'Trailing SL + Current Candle', 'Trailing SL + Previous Candle',
                     'Trailing SL + Current Swing', 'Trailing SL + Previous Swing',
                     'Volatility-Adjusted Trailing SL', 'Break-even After 50% Target',
                     'Cost-to-Cost + N Points Trailing SL']:
        return _fallback()

    return _fallback()


# ================================
# TARGET CALCULATION
# ================================

def calculate_initial_target(position_type, entry_price, df, idx, config):
    """Calculate initial target based on target type."""
    target_type = config.get('target_type', DEFAULT_TARGET_TYPE)
    current = df.iloc[idx]

    def _fallback():
        points = config.get('target_points', DEFAULT_TARGET_POINTS)
        return entry_price + points if position_type == 'LONG' else entry_price - points

    if target_type == 'Custom Points':
        return _fallback()

    elif target_type == 'P&L Based (Rupees)':
        rupees = config.get('target_rupees', DEFAULT_TARGET_RUPEES)
        quantity = config.get('quantity', 1)
        points = rupees / max(quantity, 1)
        return entry_price + points if position_type == 'LONG' else entry_price - points

    elif target_type == 'Risk-Reward Based':
        sl_price = calculate_initial_sl(position_type, entry_price, df, idx, config)
        rr_ratio = config.get('risk_reward_ratio', DEFAULT_RISK_REWARD)
        if sl_price is None:
            return _fallback()
        risk = abs(entry_price - sl_price)
        reward = risk * rr_ratio
        return entry_price + reward if position_type == 'LONG' else entry_price - reward

    elif target_type == 'ATR-based':
        if pd.isna(current['ATR']):
            return _fallback()
        multiplier = config.get('target_atr_multiplier', DEFAULT_TARGET_ATR_MULT)
        target_distance = current['ATR'] * multiplier
        return entry_price + target_distance if position_type == 'LONG' else entry_price - target_distance

    elif target_type == 'Dynamic Trailing SL+Target (Lock Profits)':
        target_distance = config.get('dynamic_trail_target_points', 20)
        return entry_price + target_distance if position_type == 'LONG' else entry_price - target_distance

    elif target_type == 'Current Candle Low/High':
        return current['High'] if position_type == 'LONG' else current['Low']

    elif target_type == 'Previous Candle Low/High':
        if pd.isna(current['Prev_Low']) or pd.isna(current['Prev_High']):
            return _fallback()
        return current['Prev_High'] if position_type == 'LONG' else current['Prev_Low']

    elif target_type == 'Current Swing Low/High':
        if pd.isna(current['Swing_Low']) or pd.isna(current['Swing_High']):
            return _fallback()
        return current['Swing_High'] if position_type == 'LONG' else current['Swing_Low']

    elif target_type == 'Previous Swing Low/High':
        if pd.isna(current['Prev_Swing_Low']) or pd.isna(current['Prev_Swing_High']):
            return _fallback()
        return current['Prev_Swing_High'] if position_type == 'LONG' else current['Prev_Swing_Low']

    elif target_type in ['Trailing Target (Points)', 'Trailing Target + Signal Based',
                         '50% Exit at Target (Partial)']:
        return _fallback()

    elif target_type == 'Signal-based (Reverse Crossover)':
        # No price target - exit ONLY on the reverse strategy signal
        return None

    elif target_type == 'Strategy-based Signal':
        return None

    return _fallback()


# ================================
# TRAILING UPDATES
# ================================

def update_trailing_sl(position, current_price, df, idx, config):
    """Update trailing stop loss (all trailing SL flavours)."""
    sl_type = config.get('sl_type', DEFAULT_SL_TYPE)
    target_type = config.get('target_type', DEFAULT_TARGET_TYPE)
    position_type = position['type']
    current_sl = position['sl_price']
    entry_price = position['entry_price']
    current = df.iloc[idx]

    if current_sl is None:
        return None

    # ============================================
    # DYNAMIC TRAILING SL+TARGET
    # ============================================
    if target_type == 'Dynamic Trailing SL+Target (Lock Profits)':
        sl_distance = config.get('dynamic_trail_sl_points', 10)
        if position_type == 'LONG':
            new_sl = current_price - sl_distance
            return max(current_sl, new_sl)
        else:
            new_sl = current_price + sl_distance
            return min(current_sl, new_sl)

    # ============================================
    # COST-TO-COST + N POINTS TRAILING SL
    # ============================================
    if sl_type == 'Cost-to-Cost + N Points Trailing SL':
        initial_sl_distance = config.get('sl_points', DEFAULT_SL_POINTS)
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

    # ============================================
    # TRAILING SL (POINTS)   <- default SL type
    # ============================================
    elif sl_type == 'Trailing SL (Points)':
        trail_points = config.get('sl_points', DEFAULT_SL_POINTS)
        if position_type == 'LONG':
            return max(current_sl, current_price - trail_points)
        else:
            return min(current_sl, current_price + trail_points)

    # ============================================
    # TRAILING PROFIT (RUPEES)
    # ============================================
    elif sl_type == 'Trailing Profit (Rupees)':
        quantity = config.get('quantity', 1)
        trail_rupees = config.get('sl_trail_rupees', DEFAULT_SL_TRAIL_RUPEES)

        if position_type == 'LONG':
            current_profit = (current_price - entry_price) * quantity
        else:
            current_profit = (entry_price - current_price) * quantity

        if 'highest_profit' not in position:
            position['highest_profit'] = current_profit

        position['highest_profit'] = max(position['highest_profit'], current_profit)

        if position['highest_profit'] - current_profit >= trail_rupees:
            return current_price

        return current_sl

    # ============================================
    # TRAILING LOSS (RUPEES)
    # ============================================
    elif sl_type == 'Trailing Loss (Rupees)':
        quantity = config.get('quantity', 1)
        trail_rupees = config.get('sl_trail_rupees', DEFAULT_SL_TRAIL_RUPEES)

        if position_type == 'LONG':
            current_profit = (current_price - entry_price) * quantity
        else:
            current_profit = (entry_price - current_price) * quantity

        if 'lowest_profit' not in position:
            position['lowest_profit'] = current_profit

        position['lowest_profit'] = min(position['lowest_profit'], current_profit)

        if current_profit - position['lowest_profit'] <= -trail_rupees:
            return current_price

        return current_sl

    elif sl_type == 'Trailing SL + Current Candle':
        if position_type == 'LONG':
            return max(current_sl, current['Low'])
        else:
            return min(current_sl, current['High'])

    elif sl_type == 'Trailing SL + Previous Candle':
        if position_type == 'LONG':
            prev_low = current['Prev_Low']
            if not pd.isna(prev_low):
                return max(current_sl, prev_low)
        else:
            prev_high = current['Prev_High']
            if not pd.isna(prev_high):
                return min(current_sl, prev_high)
        return current_sl

    elif sl_type == 'Trailing SL + Current Swing':
        if position_type == 'LONG':
            swing_low = current['Swing_Low']
            if not pd.isna(swing_low):
                return max(current_sl, swing_low)
        else:
            swing_high = current['Swing_High']
            if not pd.isna(swing_high):
                return min(current_sl, swing_high)
        return current_sl

    elif sl_type == 'Trailing SL + Previous Swing':
        if position_type == 'LONG':
            prev_swing_low = current['Prev_Swing_Low']
            if not pd.isna(prev_swing_low):
                return max(current_sl, prev_swing_low)
        else:
            prev_swing_high = current['Prev_Swing_High']
            if not pd.isna(prev_swing_high):
                return min(current_sl, prev_swing_high)
        return current_sl

    elif sl_type == 'Volatility-Adjusted Trailing SL':
        if pd.isna(current['ATR']):
            return current_sl
        multiplier = config.get('sl_atr_multiplier', DEFAULT_SL_ATR_MULT)
        trail_distance = current['ATR'] * multiplier
        if position_type == 'LONG':
            return max(current_sl, current_price - trail_distance)
        else:
            return min(current_sl, current_price + trail_distance)

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
    """Update trailing target."""
    target_type = config.get('target_type', DEFAULT_TARGET_TYPE)
    position_type = position['type']
    current_target = position['target_price']

    if current_target is None:
        return None

    if target_type == 'Trailing Target (Points)':
        trail_points = config.get('target_points', DEFAULT_TARGET_POINTS)
        if position_type == 'LONG':
            return max(current_target, current_price + trail_points)
        else:
            return min(current_target, current_price - trail_points)

    elif target_type == 'Dynamic Trailing SL+Target (Lock Profits)':
        target_distance = config.get('dynamic_trail_target_points', 20)
        if position_type == 'LONG':
            return max(current_target, current_price + target_distance)
        else:
            return min(current_target, current_price - target_distance)

    return current_target


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================
# "BACKTEST ENTRY N+1"   : signal on candle N  ->  entry at the OPEN of N+1
# "CONSERVATIVE EXIT"    : inside one candle we assume the worst sequence ->
#                          SL is tested against the candle LOW  (for LONG)
#                          / HIGH (for SHORT) FIRST, and only if the SL was not
#                          touched do we test the target against the HIGH/LOW.
#                          Gaps are handled too: if the candle OPENS beyond the
#                          stop, the fill is taken at the OPEN, not at the stop.
# Exits are evaluated with the stop/target that existed BEFORE this candle, and
# trailing is applied only AFTER the candle survived - that is the correct
# chronological order and removes look-ahead bias.
# =============================================================================

def _exit_fill_price(level, open_price, position_type, is_stop):
    """Realistic fill: a gap through the level fills at the open, not the level."""
    if level is None:
        return None
    if position_type == 'LONG':
        if is_stop:
            return min(level, open_price)      # gapped below the stop
        return max(level, open_price)          # gapped above the target
    else:
        if is_stop:
            return max(level, open_price)      # gapped above the stop
        return min(level, open_price)          # gapped below the target


def _check_price_exit(position, candle, conservative=True):
    """
    Returns (exit_reason, exit_price) or (None, None) for one candle,
    using High/Low instead of Close so intrabar touches are not missed.
    """
    ptype = position['type']
    sl = position.get('sl_price')
    tgt = position.get('target_price')
    high = float(candle['High'])
    low = float(candle['Low'])
    op = float(candle['Open'])

    sl_hit = False
    tgt_hit = False

    if sl is not None:
        sl_hit = (low <= sl) if ptype == 'LONG' else (high >= sl)
    if tgt is not None:
        tgt_hit = (high >= tgt) if ptype == 'LONG' else (low <= tgt)

    if conservative:
        # SL always wins when both are touched inside the same candle
        if sl_hit:
            return 'SL Hit', _exit_fill_price(sl, op, ptype, is_stop=True)
        if tgt_hit:
            return 'Target Hit', _exit_fill_price(tgt, op, ptype, is_stop=False)
    else:
        if tgt_hit:
            return 'Target Hit', _exit_fill_price(tgt, op, ptype, is_stop=False)
        if sl_hit:
            return 'SL Hit', _exit_fill_price(sl, op, ptype, is_stop=True)

    return None, None


def run_backtest(df, config):
    """
    Run backtesting on historical data.

    Returns:
        tuple: (trades_list, metrics_dict, debug_info, skipped_trades_list)
    """
    trades = []
    skipped_trades = []
    position = None
    strategy_name = config.get('strategy', DEFAULT_STRATEGY)
    strategy_func = STRATEGY_FUNCTIONS.get(strategy_name, check_ema_crossover_strategy)
    prevent_overlapping = config.get('prevent_overlapping_trades', True)
    conservative = config.get('conservative_intrabar_exit', DEFAULT_CONSERVATIVE_INTRABAR_EXIT)
    next_candle_entry = config.get('use_backtest_method2', DEFAULT_BACKTEST_NEXT_CANDLE_ENTRY)
    trail_on_extremes = config.get('backtest_trail_on_extremes', True)
    quantity = config.get('quantity', 1)

    total_candles = len(df)
    candles_analyzed = 0
    signals_generated = 0
    trades_entered = 0
    trades_exited = 0
    signals_skipped = 0

    start_idx = max(50, int(config.get('ema_slow', DEFAULT_EMA_SLOW)) + 10)

    for idx in range(start_idx, len(df)):
        candles_analyzed += 1
        current_data = df.iloc[idx]
        current_price = float(current_data['Close'])

        # =====================================================================
        # NO POSITION -> look for an entry signal on THIS (closed) candle
        # =====================================================================
        if position is None:
            if not is_within_trade_window(current_data['Datetime'], config):
                continue

            signal, entry_price = strategy_func(df, idx, config, None)

            if not signal:
                continue

            if not should_allow_trade_direction(signal, config):
                continue

            signals_generated += 1
            position_type = get_entry_position_type(signal, config)

            # ---- "BACKTEST ENTRY N+1" --------------------------------------
            if next_candle_entry:
                if idx + 1 >= len(df):
                    continue                      # no next candle -> skip
                entry_idx = idx + 1
                actual_entry_price = float(df.iloc[entry_idx]['Open'])
                actual_entry_time = df.iloc[entry_idx]['Datetime']
            else:
                entry_idx = idx
                actual_entry_price = float(entry_price)
                actual_entry_time = current_data['Datetime']

            # SL / Target are computed from the SIGNAL candle's indicators but
            # measured from the ACTUAL (next-candle-open) entry price.
            sl_price = calculate_initial_sl(position_type, actual_entry_price, df, idx, config)
            target_price = calculate_initial_target(position_type, actual_entry_price, df, idx, config)

            entry_metrics = {'strategy': strategy_name, 'entry_idx': entry_idx,
                             'signal_idx': idx}

            if strategy_name == 'EMA Crossover' or 'EMA_Fast' in df.columns:
                ema_fast_val = current_data.get('EMA_Fast', np.nan)
                ema_slow_val = current_data.get('EMA_Slow', np.nan)
                entry_metrics.update({
                    'ema_fast_period': config.get('ema_fast', DEFAULT_EMA_FAST),
                    'ema_slow_period': config.get('ema_slow', DEFAULT_EMA_SLOW),
                    'ema_fast_entry': ema_fast_val,
                    'ema_slow_entry': ema_slow_val,
                    'ema_angle_entry': current_data.get('EMA_Fast_Angle', np.nan),
                    'price_fast_ema_diff_entry': actual_entry_price - ema_fast_val
                    if not pd.isna(ema_fast_val) else np.nan,
                    'price_slow_ema_diff_entry': actual_entry_price - ema_slow_val
                    if not pd.isna(ema_slow_val) else np.nan,
                    'fast_slow_ema_diff_entry': ema_fast_val - ema_slow_val
                    if not pd.isna(ema_fast_val) and not pd.isna(ema_slow_val) else np.nan,
                })

            position = {
                'type': position_type,
                'entry_price': actual_entry_price,
                'entry_time': actual_entry_time,
                'entry_idx': entry_idx,
                'sl_price': sl_price,
                'target_price': target_price,
                'quantity': quantity,
                'highest_price': actual_entry_price,
                'lowest_price': actual_entry_price,
                'entry_metrics': entry_metrics,
                'ticker': config.get('asset', 'N/A'),
            }
            trades_entered += 1
            continue   # exits start from the NEXT loop iteration

        # =====================================================================
        # POSITION OPEN
        # =====================================================================
        # If the entry happens on this candle's open we still evaluate it - the
        # stop can legitimately be hit on the entry candle itself.
        if idx < position['entry_idx']:
            continue

        # ---- track overlapping (skipped) signals for analysis --------------
        if prevent_overlapping:
            signal, signal_price = strategy_func(df, idx, config, None)
            if signal and should_allow_trade_direction(signal, config):
                signals_skipped += 1
                skipped_position_type = get_entry_position_type(signal, config)
                skipped_sl = calculate_initial_sl(skipped_position_type, signal_price, df, idx, config)
                skipped_target = calculate_initial_target(skipped_position_type, signal_price, df, idx, config)

                skipped_exit_price = None
                skipped_exit_reason = None
                skipped_exit_time = None

                sim_pos = {'type': skipped_position_type, 'sl_price': skipped_sl,
                           'target_price': skipped_target}

                for future_idx in range(idx + 1, min(idx + 100, len(df))):
                    fc = df.iloc[future_idx]
                    reason, px = _check_price_exit(sim_pos, fc, conservative)
                    if reason:
                        skipped_exit_price = px
                        skipped_exit_reason = f"{reason} (Skipped)"
                        skipped_exit_time = fc['Datetime']
                        break

                if skipped_exit_price is None and idx + 20 < len(df):
                    skipped_exit_price = float(df.iloc[idx + 20]['Close'])
                    skipped_exit_reason = 'Simulated Exit (Skipped)'
                    skipped_exit_time = df.iloc[idx + 20]['Datetime']

                if skipped_exit_price:
                    if skipped_position_type == 'LONG':
                        skipped_pnl = (skipped_exit_price - signal_price) * quantity
                    else:
                        skipped_pnl = (signal_price - skipped_exit_price) * quantity

                    brokerage = calculate_brokerage(signal_price, skipped_exit_price, quantity, config)

                    skipped_trades.append({
                        'entry_time': current_data['Datetime'],
                        'exit_time': skipped_exit_time,
                        'type': skipped_position_type,
                        'entry_price': signal_price,
                        'exit_price': skipped_exit_price,
                        'sl_price': skipped_sl,
                        'target_price': skipped_target,
                        'quantity': quantity,
                        'pnl': skipped_pnl,
                        'brokerage': brokerage,
                        'net_pnl': skipped_pnl - brokerage,
                        'exit_reason': skipped_exit_reason,
                        'note': 'Overlapped with active trade'
                    })

        # ---- 1. EXIT CHECKS (using the SL/Target as of the previous candle) --
        exit_reason = None
        exit_price = current_price

        if not is_within_trade_window(current_data['Datetime'], config):
            exit_reason = 'Trade Window Closed'
            exit_price = current_price

        if exit_reason is None:
            exit_reason, exit_price = _check_price_exit(position, current_data, conservative)

        # ---- signal based exit ---------------------------------------------
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
                    # reverse-signal exits fill on the NEXT candle open, same
                    # N -> N+1 rule that governs entries
                    if next_candle_entry and idx + 1 < len(df):
                        exit_price = float(df.iloc[idx + 1]['Open'])
                    else:
                        exit_price = current_price

        # ---- 2. record the exit ---------------------------------------------
        if exit_reason:
            position['highest_price'] = max(position['highest_price'], float(current_data['High']))
            position['lowest_price'] = min(position['lowest_price'], float(current_data['Low']))

            trade = build_trade_record(position, exit_price, exit_reason, config,
                                       exit_time=current_data['Datetime'])

            exit_metrics = {}
            if strategy_name == 'EMA Crossover' or 'EMA_Fast' in df.columns:
                ef, es = current_data.get('EMA_Fast', np.nan), current_data.get('EMA_Slow', np.nan)
                exit_metrics = {
                    'ema_fast_exit': ef,
                    'ema_slow_exit': es,
                    'price_fast_ema_diff_exit': exit_price - ef if not pd.isna(ef) else np.nan,
                    'price_slow_ema_diff_exit': exit_price - es if not pd.isna(es) else np.nan,
                    'fast_slow_ema_diff_exit': ef - es
                    if not pd.isna(ef) and not pd.isna(es) else np.nan,
                }

            em = position.get('entry_metrics', {})
            trade.update({
                'strategy': em.get('strategy', strategy_name),
                'ema_fast_period': em.get('ema_fast_period', np.nan),
                'ema_slow_period': em.get('ema_slow_period', np.nan),
                'ema_angle_entry': em.get('ema_angle_entry', np.nan),
                'ema_fast_entry': em.get('ema_fast_entry', np.nan),
                'ema_slow_entry': em.get('ema_slow_entry', np.nan),
                'price_fast_ema_diff_entry': em.get('price_fast_ema_diff_entry', np.nan),
                'price_slow_ema_diff_entry': em.get('price_slow_ema_diff_entry', np.nan),
                'fast_slow_ema_diff_entry': em.get('fast_slow_ema_diff_entry', np.nan),
                'ema_fast_exit': exit_metrics.get('ema_fast_exit', np.nan),
                'ema_slow_exit': exit_metrics.get('ema_slow_exit', np.nan),
                'price_fast_ema_diff_exit': exit_metrics.get('price_fast_ema_diff_exit', np.nan),
                'price_slow_ema_diff_exit': exit_metrics.get('price_slow_ema_diff_exit', np.nan),
                'fast_slow_ema_diff_exit': exit_metrics.get('fast_slow_ema_diff_exit', np.nan),
            })

            trades.append(trade)
            trades_exited += 1
            position = None
            continue

        # ---- 3. survived the candle -> update tracking + trailing ------------
        position['highest_price'] = max(position['highest_price'], float(current_data['High']))
        position['lowest_price'] = min(position['lowest_price'], float(current_data['Low']))

        if trail_on_extremes:
            trail_price = float(current_data['High']) if position['type'] == 'LONG' \
                else float(current_data['Low'])
        else:
            trail_price = current_price

        if position['sl_price'] is not None:
            position['sl_price'] = update_trailing_sl(position, trail_price, df, idx, config)

        if position['target_price'] is not None:
            position['target_price'] = update_trailing_target(position, trail_price, df, idx, config)

    # =========================================================================
    # METRICS
    # =========================================================================
    if trades:
        df_trades = pd.DataFrame(trades)

        total_trades = len(trades)
        winning_trades = int((df_trades['pnl'] > 0).sum())
        losing_trades = int((df_trades['pnl'] < 0).sum())

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = float(df_trades['pnl'].sum())
        avg_pnl = float(df_trades['pnl'].mean())

        total_brokerage = float(df_trades['brokerage'].sum())
        total_net_pnl = float(df_trades['net_pnl'].sum())
        avg_net_pnl = float(df_trades['net_pnl'].mean())

        cumulative_pnl = df_trades['net_pnl'].cumsum()
        drawdown = cumulative_pnl - cumulative_pnl.cummax()
        max_drawdown = float(drawdown.min())

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
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0, 'max_drawdown': 0,
            'total_brokerage': 0, 'total_net_pnl': 0, 'avg_net_pnl': 0,
        }

    debug_info = {
        'total_candles': total_candles,
        'candles_analyzed': candles_analyzed,
        'signals_generated': signals_generated,
        'trades_entered': trades_entered,
        'trades_completed': trades_exited,
        'signals_skipped': signals_skipped,
        'overlapping_trades': len(skipped_trades),
        'entry_rule': 'Next candle OPEN (N+1)' if next_candle_entry else 'Signal candle CLOSE (N)',
        'exit_rule': 'SL checked against Low/High first (conservative)'
        if conservative else 'Target checked first (optimistic)',
    }

    return trades, metrics, debug_info, skipped_trades


# =============================================================================
# LIVE TRADING
# =============================================================================

def add_log(message):
    """Add timestamped log message"""
    timestamp = now_ist().strftime('%H:%M:%S')
    log_entry = f"[{timestamp}] {message}"

    if 'live_logs' not in st.session_state:
        st.session_state['live_logs'] = []

    st.session_state['live_logs'].append(log_entry)
    # keep memory bounded during long sessions
    if len(st.session_state['live_logs']) > 2000:
        st.session_state['live_logs'] = st.session_state['live_logs'][-1500:]


def get_live_signal_index(df):
    """
    "CROSSOVER FIX"
    Which row should the strategy be evaluated on?

        LIVE_USE_CLOSED_CANDLE_ONLY = True  ->  len(df) - 2  (last CLOSED candle)
        LIVE_USE_CLOSED_CANDLE_ONLY = False ->  len(df) - 1  (forming candle)

    With the closed candle, df.iloc[idx-1] / df.iloc[idx] is exactly the
    prev(-3) / curr(-2) pair from your snippet, so a crossover is reported
    once and only once - on the candle where it actually happened.
    """
    if LIVE_USE_CLOSED_CANDLE_ONLY and len(df) >= 3:
        return len(df) - 2
    return len(df) - 1


def finalize_exit(position, exit_price, exit_reason, config, log_func=None):
    """
    "MANUAL SQUAREOFF FIX"
    One exit path for EVERYTHING: SL, target, signal, trade-window and the
    Manual Square-Off button. It records the trade, fires the broker exit,
    sends the email and cleans the session - but it NEVER touches
    st.session_state['trading_active'], so live trading keeps running after a
    manual close.
    """
    log_func = log_func or add_log

    trade_record = build_trade_record(position, exit_price, exit_reason, config)

    log_func(f"🚪 EXITING POSITION: {exit_reason} @ {exit_price:.2f}")
    log_func(f"💰 P&L: ₹{trade_record['pnl']:.2f} | Entry: ₹{position['entry_price']:.2f} "
             f"| Exit: ₹{exit_price:.2f}")
    if config.get('include_brokerage', False):
        log_func(f"💸 Brokerage: ₹{trade_record['brokerage']:.2f}")
        log_func(f"💵 Net P&L: ₹{trade_record['net_pnl']:.2f}")

    if 'trade_history' not in st.session_state:
        st.session_state['trade_history'] = []
    st.session_state['trade_history'].append(trade_record)
    log_func("📝 Trade saved to history")

    # ---- broker exit ----------------------------------------------------
    if config.get('dhan_enabled', False):
        broker_position = st.session_state.get('broker_position')
        if broker_position:
            dhan_broker = st.session_state.get('dhan_broker')
            if dhan_broker:
                try:
                    exit_info = dhan_broker.exit_broker_position(
                        broker_position, exit_price, exit_reason, log_func)
                    st.session_state['broker_exit'] = exit_info
                except Exception as e:
                    log_func(f"🏦 ⚠️ Broker exit error: {e}")

    # ---- email ----------------------------------------------------------
    if config.get('email_on_exit', True):
        email_trade_event(
            "EXIT", position, config, log_func,
            extra={
                'Exit Px': f"{exit_price:.2f}",
                'Reason': exit_reason,
                'P&L': f"{trade_record['pnl']:.2f}",
                'Net P&L': f"{trade_record['net_pnl']:.2f}",
            }
        )

    # ---- session cleanup (trading stays ACTIVE) --------------------------
    st.session_state['last_exit_time'] = now_ist()
    st.session_state['position'] = None
    st.session_state['broker_position'] = None
    st.session_state['last_entry_bar_time'] = None
    log_func("✅ Position closed, session cleared (live trading still running)")

    return trade_record


def manual_square_off(config):
    """
    Manual close button handler.
    Fetches a fresh price for the LOCKED ticker of the open position and exits.
    Fully guarded - any failure is logged, never raised, and never stops the
    live-trading loop.
    """
    position = st.session_state.get('position')
    if not position:
        st.warning("No active position to close")
        return False

    try:
        ticker = position.get('ticker', config.get('asset', DEFAULT_ASSET))
        custom_ticker = position.get('custom_ticker', config.get('custom_ticker'))
        interval = INTERVAL_MAPPING.get(config.get('interval', DEFAULT_INTERVAL), '1m')
        period = PERIOD_MAPPING.get(config.get('period', DEFAULT_PERIOD), '5d')

        df = fetch_data(ticker, interval, period, is_live_trading=True, custom_ticker=custom_ticker)

        if df is None or df.empty:
            add_log("❌ Manual close: could not fetch a price - using last known price")
            exit_price = float(st.session_state.get('live_price', position['entry_price']))
        else:
            exit_price = float(df.iloc[-1]['Close'])

        finalize_exit(position, exit_price, 'Manual Close', config)
        st.success(f"Position closed manually @ ₹{exit_price:.2f}")
        return True

    except Exception as e:
        add_log(f"❌ Manual close error: {e}")
        add_log(traceback.format_exc().splitlines()[-1])
        st.error(f"Manual close failed: {e}")
        return False


def live_trading_iteration():
    """
    Single iteration of the live trading loop.
    Wrapped by the caller in try/except so one bad tick can never stop trading.
    """
    config = st.session_state.get('config', {})

    if st.session_state.get('clearing_in_progress', False):
        add_log("🏦 ⏳ Position clearing in progress - skipping iteration")
        return

    position = st.session_state.get('position')

    # Use the ticker locked into the active position (if any)
    if position is not None and 'ticker' in position:
        ticker = position['ticker']
        custom_ticker = position.get('custom_ticker')
    else:
        ticker = config.get('asset', DEFAULT_ASSET)
        custom_ticker = config.get('custom_ticker', None)

    interval = INTERVAL_MAPPING.get(config.get('interval', DEFAULT_INTERVAL), '1m')
    period = PERIOD_MAPPING.get(config.get('period', DEFAULT_PERIOD), '5d')

    enhanced_mode = config.get('enhanced_live_trading', False)

    # =====================================================================
    # DATA
    # =====================================================================
    if enhanced_mode:
        current_time = now_ist()
        interval_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60}.get(interval, 1)
        is_round_time = (current_time.minute % interval_minutes == 0 and current_time.second < 5)
        last_candle_fetch = st.session_state.get('last_candle_fetch_time')

        if is_round_time and (last_candle_fetch is None or
                              (current_time - last_candle_fetch).total_seconds() >= interval_minutes * 60):
            add_log(f"🔄 Enhanced Mode: fetching candles at {current_time.strftime('%H:%M:%S')}")
            df_candles = fetch_data(ticker, interval, period, is_live_trading=True,
                                    custom_ticker=custom_ticker)
            if df_candles is not None and not df_candles.empty:
                df_candles = calculate_all_indicators(df_candles, config)
                st.session_state['indicator_df'] = df_candles
                st.session_state['last_candle_fetch_time'] = current_time
                add_log(f"✅ Indicators calculated from {len(df_candles)} candles")

        df = st.session_state.get('indicator_df')

        if df is None or df.empty:
            df = fetch_data(ticker, interval, period, is_live_trading=True, custom_ticker=custom_ticker)
            if df is None or df.empty:
                add_log("❌ Failed to fetch data")
                return
            df = calculate_all_indicators(df, config)
            st.session_state['indicator_df'] = df

        df_live = fetch_data(ticker, interval, period, is_live_trading=True, custom_ticker=custom_ticker)
        live_price = float(df_live.iloc[-1]['Close']) if (df_live is not None and not df_live.empty) \
            else float(df.iloc[-1]['Close'])
    else:
        df = fetch_data(ticker, interval, period, is_live_trading=True, custom_ticker=custom_ticker)
        if df is None or df.empty:
            add_log("❌ Failed to fetch data")
            return
        df = calculate_all_indicators(df, config)
        live_price = float(df.iloc[-1]['Close'])

    if len(df) < 3:
        add_log(f"⚠️ Only {len(df)} candles available - waiting for more data")
        return

    # ---- "CROSSOVER FIX": evaluate on the last CLOSED candle -------------
    idx = get_live_signal_index(df)
    current_data = df.iloc[idx]
    bar_time = current_data['Datetime']

    st.session_state['current_data'] = current_data
    st.session_state['live_price'] = live_price
    st.session_state['signal_bar_time'] = bar_time
    st.session_state['candle_count'] = len(df)

    strategy_name = config.get('strategy', DEFAULT_STRATEGY)
    strategy_func = STRATEGY_FUNCTIONS.get(strategy_name, check_ema_crossover_strategy)

    position = st.session_state.get('position')

    # =====================================================================
    # NO POSITION -> ENTRY
    # =====================================================================
    if position is None:
        if not is_within_trade_window(bar_time, config):
            add_log("⏰ Outside trade window - no new entries allowed")
            return

        cooldown_enabled = config.get('enable_entry_cooldown', False)
        cooldown_seconds = config.get('entry_cooldown_seconds', 0)

        if cooldown_enabled and cooldown_seconds > 0:
            last_exit_time = st.session_state.get('last_exit_time')
            if last_exit_time is not None:
                time_since_exit = (now_ist() - last_exit_time).total_seconds()
                if time_since_exit < cooldown_seconds:
                    add_log(f"⏳ Entry Cooldown: {int(cooldown_seconds - time_since_exit)}s remaining")
                    return

        add_log(f"🔍 Checking {strategy_name} on CLOSED candle "
                f"{pd.Timestamp(bar_time).strftime('%H:%M:%S')} (idx {idx} of {len(df) - 1})")

        signal, entry_price = strategy_func(df, idx, config, None)

        if not signal:
            add_log(f"⏳ No entry signal (live price: {live_price:.2f})")
            return

        # ---- one entry per candle, ever ---------------------------------
        if LIVE_ONE_ENTRY_PER_BAR:
            if st.session_state.get('last_entry_bar_time') == bar_time:
                add_log("⛔ Signal already acted on for this candle - ignoring duplicate")
                return

        if cooldown_enabled and cooldown_seconds > 0:
            if st.session_state.get('last_signal_type') == signal:
                add_log(f"⛔ Same signal {signal} as last trade - waiting for a direction change")
                return

        if not should_allow_trade_direction(signal, config):
            add_log(f"⛔ Signal {signal} filtered out by "
                    f"{config.get('trade_direction', 'Both (LONG + SHORT)')}")
            return

        if config.get('dhan_enabled', False) and st.session_state.get('broker_position'):
            add_log("⛔ Broker order already active - preventing duplicate order")
            return

        add_log(f"✅ FRESH CROSSOVER / SIGNAL: {signal} at {entry_price:.2f}")

        st.session_state['last_signal_type'] = signal
        st.session_state['last_entry_bar_time'] = bar_time

        position_type = get_entry_position_type(signal, config)

        # Entry is filled at the CURRENT live price (the candle that produced
        # the signal has already closed - this is the live equivalent of the
        # backtest's "enter on candle N+1").
        actual_entry_price = live_price

        sl_price = calculate_initial_sl(position_type, actual_entry_price, df, idx, config)
        target_price = calculate_initial_target(position_type, actual_entry_price, df, idx, config)

        add_log(f"📈 ENTERING {position_type} POSITION @ {actual_entry_price:.2f}")
        add_log(f"🛡️ Initial SL: {f'{sl_price:.2f}' if sl_price is not None else 'Not Set'} | "
                f"🎯 Target: {f'{target_price:.2f}' if target_price is not None else 'Not Set'}")

        position = {
            'type': position_type,
            'entry_price': actual_entry_price,
            'entry_time': now_ist(),
            'entry_bar_time': bar_time,
            'sl_price': sl_price,
            'target_price': target_price,
            'quantity': config.get('quantity', DEFAULT_QUANTITY),
            'highest_price': actual_entry_price,
            'lowest_price': actual_entry_price,
            'ticker': config.get('asset', DEFAULT_ASSET),
            'custom_ticker': config.get('custom_ticker'),
        }

        st.session_state['position'] = position
        add_log(f"📊 Locked ticker: {position['ticker']} | Entry: ₹{actual_entry_price:.2f}")

        # ---- broker order ------------------------------------------------
        if config.get('dhan_enabled', False):
            dhan_broker = st.session_state.get('dhan_broker')
            if dhan_broker:
                try:
                    if config.get('clear_positions_before_entry', False):
                        add_log("🏦 🧹 Clearing existing positions before new entry...")
                        st.session_state['clearing_in_progress'] = True
                        clear_result = dhan_broker.clear_all_positions(add_log, convert_to_market=True)
                        st.session_state['clearing_in_progress'] = False

                        if not clear_result['clearing_complete']:
                            add_log("🏦 ⚠️ Position clearing incomplete - aborting new entry")
                            st.session_state['position'] = None
                            return

                    broker_position = dhan_broker.enter_broker_position(
                        signal, actual_entry_price, config, add_log)
                    st.session_state['broker_position'] = broker_position
                except Exception as e:
                    st.session_state['clearing_in_progress'] = False
                    add_log(f"🏦 ⚠️ Broker order error: {e}")
            else:
                add_log("🏦 ⚠️ Broker not initialized")

        # ---- email -------------------------------------------------------
        if config.get('email_on_entry', True):
            email_trade_event("ENTRY", position, config, add_log,
                              extra={'Signal': signal, 'Bar': str(bar_time)})
        return

    # =====================================================================
    # POSITION OPEN -> MONITOR
    # =====================================================================
    add_log(f"📊 Monitoring {position['type']} position @ {live_price:.2f}")

    price_diff_pct = abs(live_price - position['entry_price']) / position['entry_price'] * 100
    if price_diff_pct > 50:
        add_log(f"⚠️ Live price ₹{live_price:.2f} differs {price_diff_pct:.1f}% from entry "
                f"₹{position['entry_price']:.2f} - possible ticker mismatch "
                f"(locked: {position.get('ticker')})")

    position['highest_price'] = max(position['highest_price'], live_price)
    position['lowest_price'] = min(position['lowest_price'], live_price)

    if position['type'] == 'LONG':
        current_pnl = (live_price - position['entry_price']) * position['quantity']
    else:
        current_pnl = (position['entry_price'] - live_price) * position['quantity']

    add_log(f"💰 Current P&L: ₹{current_pnl:.2f}")

    # ---- exit checks BEFORE trailing (correct chronological order) -------
    exit_reason = None
    exit_price = live_price

    if not is_within_trade_window(bar_time, config):
        exit_reason = 'Trade Window Closed'
        add_log("⏰ TRADE WINDOW CLOSED - force exiting position")

    if exit_reason is None and position['sl_price'] is not None:
        if position['type'] == 'LONG' and live_price <= position['sl_price']:
            exit_reason = 'SL Hit'
            exit_price = live_price
            add_log(f"🛑 STOP LOSS HIT! {live_price:.2f} <= SL {position['sl_price']:.2f}")
        elif position['type'] == 'SHORT' and live_price >= position['sl_price']:
            exit_reason = 'SL Hit'
            exit_price = live_price
            add_log(f"🛑 STOP LOSS HIT! {live_price:.2f} >= SL {position['sl_price']:.2f}")

    if exit_reason is None and position['target_price'] is not None:
        if position['type'] == 'LONG' and live_price >= position['target_price']:
            exit_reason = 'Target Hit'
            exit_price = live_price
            add_log(f"🎯 TARGET HIT! {live_price:.2f} >= Target {position['target_price']:.2f}")
        elif position['type'] == 'SHORT' and live_price <= position['target_price']:
            exit_reason = 'Target Hit'
            exit_price = live_price
            add_log(f"🎯 TARGET HIT! {live_price:.2f} <= Target {position['target_price']:.2f}")

    if exit_reason is None and (
        config.get('sl_type') == 'Signal-based (Reverse Crossover)' or
        config.get('target_type') == 'Signal-based (Reverse Crossover)' or
        config.get('sl_type') == 'Strategy-based Signal' or
        config.get('target_type') == 'Strategy-based Signal'
    ):
        # Reverse-crossover exit - also evaluated on the last CLOSED candle,
        # so it fires exactly once per genuine crossover.
        signal, _ = strategy_func(df, idx, config, position)
        if signal:
            if (position['type'] == 'LONG' and signal in ('SELL', 'SHORT')) or \
               (position['type'] == 'SHORT' and signal in ('BUY', 'LONG')):
                if position.get('entry_bar_time') == bar_time:
                    add_log("↩️ Reverse signal on the same candle as entry - ignored")
                else:
                    exit_reason = 'Strategy Signal Exit'
                    exit_price = live_price
                    add_log(f"🔄 REVERSE SIGNAL DETECTED: {signal}")

    if exit_reason:
        finalize_exit(position, exit_price, exit_reason, config)
        return

    # ---- no exit -> update trailing SL / Target ---------------------------
    old_sl = position['sl_price']
    if old_sl is not None:
        position['sl_price'] = update_trailing_sl(position, live_price, df, idx, config)
        if position['sl_price'] != old_sl:
            add_log(f"🛡️ SL Updated: {old_sl:.2f} → {position['sl_price']:.2f}")

    old_target = position['target_price']
    if old_target is not None:
        position['target_price'] = update_trailing_target(position, live_price, df, idx, config)
        if position['target_price'] != old_target:
            add_log(f"🎯 Target Updated: {old_target:.2f} → {position['target_price']:.2f}")

    st.session_state['position'] = position
    add_log("⏳ No exit conditions met - holding position")


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_config_ui():
    """Render configuration sidebar"""
    st.sidebar.header("⚙️ Configuration")

    config = {}

    # Asset Selection
    config['asset'] = st.sidebar.selectbox(
        "Asset", list(ASSET_MAPPING.keys()),
        index=safe_index(list(ASSET_MAPPING.keys()), DEFAULT_ASSET))

    if config['asset'] == 'Custom Ticker':
        config['custom_ticker'] = st.sidebar.text_input(
            "Enter Ticker Symbol", value="KAYNES.NS",
            help="e.g., KAYNES.NS, RELIANCE.NS, TCS.NS")

    # Timeframe
    config['interval'] = st.sidebar.selectbox(
        "Interval", list(INTERVAL_MAPPING.keys()),
        index=safe_index(list(INTERVAL_MAPPING.keys()), DEFAULT_INTERVAL))
    config['period'] = st.sidebar.selectbox(
        "Period", list(PERIOD_MAPPING.keys()),
        index=safe_index(list(PERIOD_MAPPING.keys()), DEFAULT_PERIOD))

    _iv = INTERVAL_MAPPING.get(config['interval'], '1m')
    _days = _required_days_for_warmup(_iv, PERIOD_MAPPING.get(config['period'], '5d'))
    st.sidebar.caption(
        f"📥 Will download ~{_days} calendar days of {config['interval']} data "
        f"(warm-up guaranteed, so EMA/RSI/ADX are valid from 09:15)."
    )

    config['quantity'] = st.sidebar.number_input("Quantity", min_value=1, value=DEFAULT_QUANTITY)

    # ── Trade Window Settings ────────────────────────────────────────────
    st.sidebar.subheader("⏰ Trade Window")
    config['use_trade_window'] = st.sidebar.checkbox(
        "Enable Trade Window", value=False,
        help="Restricts trading to specific hours. Exits positions and blocks new entries outside this window.")
    if config['use_trade_window']:
        config['trade_window_start'] = st.sidebar.time_input("Start Time (IST)", value=dt_time(9, 30))
        config['trade_window_end'] = st.sidebar.time_input("End Time (IST)", value=dt_time(15, 0))
        st.sidebar.info(f"🕐 Active: {config['trade_window_start'].strftime('%H:%M')} - "
                        f"{config['trade_window_end'].strftime('%H:%M')} IST")

    # ── Trade Direction Filter ───────────────────────────────────────────
    config['trade_direction'] = st.sidebar.selectbox(
        "Trade Direction Filter",
        ["Both (LONG + SHORT)", "LONG Only", "SHORT Only"], index=0,
        help="Filters which trade directions the algo will take")

    config['reverse_entry'] = st.sidebar.checkbox(
        "Reverse Entry (Flip Signal Direction)", value=False,
        help="A LONG/BUY signal enters SHORT and vice-versa. Useful for index option buying.")

    # ── Brokerage ────────────────────────────────────────────────────────
    st.sidebar.subheader("💰 Brokerage & Charges")
    config['include_brokerage'] = st.sidebar.checkbox(
        "Include Brokerage & Charges", value=False,
        help="Deducts brokerage from P&L to show Net P&L (backtest + live)")
    if config['include_brokerage']:
        config['brokerage_per_trade'] = st.sidebar.number_input(
            "Brokerage per Trade (₹)", min_value=0.0, value=20.0, step=1.0)
        config['brokerage_percentage'] = st.sidebar.number_input(
            "Or % of Turnover", min_value=0.0, value=0.03, step=0.01, format="%.3f")
        config['brokerage_type'] = st.sidebar.radio(
            "Brokerage Calculation", ["Fixed per Trade", "Percentage of Turnover"],
            index=0, horizontal=True)

    # ── Overlap / cooldown ───────────────────────────────────────────────
    config['prevent_overlapping_trades'] = st.sidebar.checkbox(
        "🚫 Prevent Overlapping Trades", value=True,
        help="Blocks new signals while a position is active. Skipped signals are tracked separately.")

    config['enable_entry_cooldown'] = st.sidebar.checkbox(
        "⏱️ Enable Entry Cooldown", value=False,
        help="Prevents immediate re-entry after an exit (Live Trading only).")

    if config['enable_entry_cooldown']:
        config['entry_cooldown_seconds'] = st.sidebar.number_input(
            "Cooldown Duration (seconds)", min_value=0, max_value=300, value=60, step=5)
    else:
        config['entry_cooldown_seconds'] = 0

    config['enhanced_live_trading'] = st.sidebar.checkbox(
        "🔄 Enhanced Live Trading (TradingView Match)", value=False,
        help=("Fetches candlestick data at round time intervals matching your timeframe and uses "
              "a separate live feed for SL/Target checks."))

    config['show_last_candle'] = st.sidebar.checkbox(
        "📊 Show Last Candle Details", value=False,
        help="Display the last received candle with all calculated indicator values")

    # =====================================================================
    # "EMAIL ALERTS"
    # =====================================================================
    st.sidebar.subheader("📧 Email Notifications")
    config['enable_email_alerts'] = st.sidebar.checkbox(
        "Enable Email Notifications", value=DEFAULT_EMAIL_ENABLED,
        help="Works in PAPER trading and when the Dhan broker is enabled.")

    if config['enable_email_alerts']:
        config['email_from'] = st.sidebar.text_input("From (Gmail)", value=DEFAULT_EMAIL_FROM)
        config['email_to'] = st.sidebar.text_input(
            "To (comma separated)", value=DEFAULT_EMAIL_TO)
        config['email_app_password'] = st.sidebar.text_input(
            "Gmail App Password", type="password", value="",
            help=("NOT your Gmail password. Create one at "
                  "Google Account → Security → 2-Step Verification → App passwords. "
                  "16 characters, spaces optional."))
        config['email_smtp_host'] = DEFAULT_EMAIL_SMTP_HOST
        config['email_smtp_port'] = DEFAULT_EMAIL_SMTP_PORT

        col_e1, col_e2 = st.sidebar.columns(2)
        with col_e1:
            config['email_on_entry'] = st.sidebar.checkbox("Mail on ENTRY", value=True)
        with col_e2:
            config['email_on_exit'] = st.sidebar.checkbox("Mail on EXIT", value=True)

        if st.sidebar.button("✉️ Send test email"):
            ok = send_email_notification(
                "[ALGO] Test email",
                f"Test message from the Algo Trading System at "
                f"{now_ist().strftime('%Y-%m-%d %H:%M:%S')} IST.",
                config)
            if ok:
                st.sidebar.success("Test email sent ✅")
            else:
                st.sidebar.error("Test email failed - check the App Password.")
    else:
        config['email_on_entry'] = False
        config['email_on_exit'] = False

    # =====================================================================
    # "GROQ CHATBOT"
    # =====================================================================
    st.sidebar.subheader("🤖 Groq AI Chatbot")
    config['enable_groq_chat'] = st.sidebar.checkbox(
        "Enable Groq Chatbot", value=DEFAULT_GROQ_ENABLED,
        help="Adds an AI chat panel to the Backtest, Live Trading and Trade History tabs.")

    if config['enable_groq_chat']:
        config['groq_api_key'] = st.sidebar.text_input(
            "Groq API Key", type="password", value="",
            help="Get one free at console.groq.com → API Keys")
        config['groq_model'] = st.sidebar.selectbox(
            "Groq Model", GROQ_MODELS, index=safe_index(GROQ_MODELS, DEFAULT_GROQ_MODEL))
        st.sidebar.caption(
            "⚠️ Groq has decommissioned llama3-70b-8192, mixtral-8x7b-32768, "
            "gemma-7b-it and gemma2-9b-it - only live models are listed above.")
        config['groq_context_rows'] = st.sidebar.number_input(
            "Rows of data sent to the model", min_value=10, max_value=200, value=50, step=10)
    else:
        config['groq_context_rows'] = 50

    # =====================================================================
    # STRATEGY
    # =====================================================================
    st.sidebar.subheader("📊 Strategy")
    config['strategy'] = st.sidebar.selectbox(
        "Strategy Type", STRATEGY_LIST, index=safe_index(STRATEGY_LIST, DEFAULT_STRATEGY))

    if config['strategy'] == 'EMA Crossover':
        config['ema_fast'] = st.sidebar.number_input("EMA Fast Period", min_value=1,
                                                     value=DEFAULT_EMA_FAST)
        config['ema_slow'] = st.sidebar.number_input("EMA Slow Period", min_value=1,
                                                     value=DEFAULT_EMA_SLOW)
        config['ema_min_angle'] = st.sidebar.number_input(
            "Min Angle (ABSOLUTE, 0 = off)", min_value=0.0,
            value=float(DEFAULT_EMA_MIN_ANGLE), step=0.1)

        config['ema_entry_filter'] = st.sidebar.selectbox("Entry Filter", EMA_ENTRY_FILTERS, index=0)

        if config['ema_entry_filter'] == 'Custom Candle (Points)':
            config['ema_custom_candle_points'] = st.sidebar.number_input(
                "Min Candle Points", min_value=1, value=5)
        elif config['ema_entry_filter'] == 'ATR-based Candle':
            config['ema_atr_multiplier'] = st.sidebar.number_input(
                "ATR Multiplier", min_value=0.1, value=0.3, step=0.1)

        config['ema_use_adx'] = st.sidebar.checkbox("Use ADX Filter", value=DEFAULT_EMA_USE_ADX)
        if config['ema_use_adx']:
            config['ema_adx_threshold'] = st.sidebar.number_input(
                "ADX Threshold", min_value=1, value=DEFAULT_EMA_ADX_THRESHOLD)
        config['adx_period'] = st.sidebar.number_input(
            "ADX Period", min_value=1, value=DEFAULT_ADX_PERIOD)

        st.sidebar.info(
            "✅ Entry requires BOTH conditions:\n"
            "prev_fast ≤ prev_slow AND curr_fast > curr_slow (bullish).\n"
            "In live trading the check runs on the last CLOSED candle, so one "
            "crossover = one entry.")

    elif config['strategy'] == 'Price Crosses Threshold':
        config['price_threshold'] = st.sidebar.number_input("Price Threshold", min_value=0.0, value=25000.0)
        config['price_cross_type'] = st.sidebar.selectbox(
            "Cross Type", ["Above Threshold", "Below Threshold"])
        config['price_cross_position'] = st.sidebar.selectbox("Position Type", ["LONG", "SHORT"])

    elif config['strategy'] == 'Percentage Change':
        config['pct_change_threshold'] = st.sidebar.number_input(
            "% Change Threshold", min_value=0.001, value=2.0, step=0.001, format="%.3f")
        config['pct_change_type'] = st.sidebar.selectbox(
            "Change Type", ["Positive % (Price Up)", "Negative % (Price Down)"])
        config['pct_change_position'] = st.sidebar.selectbox("Position Type", ["LONG", "SHORT"])

    elif config['strategy'] == 'SuperTrend AI':
        st.sidebar.markdown("**SuperTrend AI Parameters**")
        config['supertrend_atr_period'] = st.sidebar.number_input("ATR Period", min_value=5, value=10)
        config['supertrend_multiplier'] = st.sidebar.number_input("Multiplier", min_value=1.0, value=3.0, step=0.5)
        config['supertrend_adx_threshold'] = st.sidebar.number_input("ADX Threshold", min_value=10, value=25)
        config['supertrend_volume_mult'] = st.sidebar.number_input("Volume Multiplier", min_value=1.0, value=1.5, step=0.1)

    elif config['strategy'] == 'VWAP + Volume Spike':
        st.sidebar.markdown("**VWAP + Volume Spike Parameters**")
        config['vwap_volume_mult'] = st.sidebar.number_input("Volume Spike Multiplier", min_value=1.5, value=2.0, step=0.1)
        config['vwap_distance_pct'] = st.sidebar.number_input("Max Distance from VWAP (%)", min_value=0.1, value=0.3, step=0.1)
        st.sidebar.info("VWAP is session-anchored (resets daily) exactly like TradingView.")

    elif config['strategy'] == 'Bollinger Squeeze Breakout':
        st.sidebar.markdown("**Bollinger Squeeze Breakout Parameters**")
        config['bb_squeeze_period'] = st.sidebar.number_input("BB Period", min_value=10, value=20)
        config['bb_squeeze_std'] = st.sidebar.number_input("BB Std Dev", min_value=1.0, value=2.0, step=0.1)
        config['bb_squeeze_threshold'] = st.sidebar.number_input("Squeeze Threshold (%)", min_value=0.01, value=0.02, step=0.01, format="%.3f")
        config['bb_squeeze_volume_mult'] = st.sidebar.number_input("Breakout Volume Mult", min_value=1.0, value=1.8, step=0.1)

    elif config['strategy'] == 'Elliott Waves + Ratio Charts':
        config['elliott_wave_lookback'] = st.sidebar.number_input("Wave Lookback Period", min_value=20, value=50)

    elif config['strategy'] == 'Opening Range Breakout (ORB)':
        config['orb_minutes'] = st.sidebar.number_input("Opening Range Duration (minutes)", min_value=5, max_value=60, value=15)
        config['orb_breakout_buffer'] = st.sidebar.number_input("Breakout Buffer (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    elif config['strategy'] == 'Pivot Point Reversal':
        config['pivot_lookback'] = st.sidebar.number_input("Lookback Period", min_value=12, max_value=48, value=24)

    elif config['strategy'] == 'Ichimoku Cloud':
        st.sidebar.markdown("**Ichimoku Parameters**")
        config['ichimoku_tenkan'] = st.sidebar.number_input("Tenkan (Conversion)", min_value=2, value=9)
        config['ichimoku_kijun'] = st.sidebar.number_input("Kijun (Base)", min_value=2, value=26)
        config['ichimoku_senkou_b'] = st.sidebar.number_input("Senkou Span B", min_value=2, value=52)
        config['ichimoku_displacement'] = st.sidebar.number_input("Displacement", min_value=1, value=26)
        st.sidebar.info("☁️ TK cross confirmed by price being outside the Kumo cloud.")

    elif config['strategy'] == 'Volume Breakout':
        config['volume_multiplier'] = st.sidebar.number_input("Volume Multiplier", min_value=1.5, max_value=5.0, value=2.0, step=0.5)
        config['volume_price_threshold'] = st.sidebar.number_input("Min Price Change (%)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

    elif config['strategy'] == 'Gap Trading Strategy':
        config['gap_min_percent'] = st.sidebar.number_input("Minimum Gap (%)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        config['gap_max_percent'] = st.sidebar.number_input("Maximum Gap (%)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)

    elif config['strategy'] == 'Mean Reversion with Bollinger Bands':
        config['bb_period'] = st.sidebar.number_input("Bollinger Period", min_value=10, max_value=50, value=20)
        config['bb_std'] = st.sidebar.number_input("Standard Deviations", min_value=1.5, max_value=3.0, value=2.0, step=0.5)
        config['custom_bb_period'] = config['bb_period']
        config['custom_bb_std'] = config['bb_std']
        config['mr_rsi_oversold'] = st.sidebar.number_input("RSI Oversold", min_value=20, max_value=40, value=30)
        config['mr_rsi_overbought'] = st.sidebar.number_input("RSI Overbought", min_value=60, max_value=80, value=70)

    elif config['strategy'] == 'Momentum Breakout with ADX':
        config['momentum_adx_threshold'] = st.sidebar.number_input("ADX Threshold", min_value=20, max_value=40, value=25)
        config['momentum_lookback'] = st.sidebar.number_input("Breakout Lookback", min_value=10, max_value=50, value=20)
        config['momentum_volume_ratio'] = st.sidebar.number_input("Volume Ratio", min_value=1.0, max_value=3.0, value=1.5, step=0.5)

    elif config['strategy'] == 'Support Resistance Bounce':
        config['sr_lookback'] = st.sidebar.number_input("Lookback Period", min_value=50, max_value=200, value=100)
        config['sr_tolerance'] = st.sidebar.number_input("Level Tolerance (%)", min_value=0.1, max_value=1.0, value=0.2, step=0.1) / 100
        config['sr_min_touches'] = st.sidebar.number_input("Min Level Touches", min_value=2, max_value=5, value=3)

    # ---- NEW STRATEGY PARAMETERS ---------------------------------------
    elif config['strategy'] == 'Donchian Channel Breakout':
        st.sidebar.markdown("**Donchian Channel Parameters**")
        config['donchian_period'] = st.sidebar.number_input("Channel Period (N)", min_value=5, max_value=200, value=20)
        config['donchian_mode'] = st.sidebar.selectbox(
            "Mode", ["Breakout (Trend Following)", "Mean Reversion (Fade)"], index=0)
        config['donchian_use_adx'] = st.sidebar.checkbox("Use ADX Filter", value=False)
        if config['donchian_use_adx']:
            config['donchian_adx_threshold'] = st.sidebar.number_input("ADX Threshold", min_value=5, value=20)
        st.sidebar.info("📦 Channel is measured on the PREVIOUS N bars (current bar excluded), "
                        "so a close beyond it is a genuine breakout.")

    elif config['strategy'] == 'Keltner Channel Breakout':
        st.sidebar.markdown("**Keltner Channel Parameters**")
        config['keltner_period'] = st.sidebar.number_input("EMA Basis Period", min_value=5, max_value=100, value=20)
        config['keltner_multiplier'] = st.sidebar.number_input("ATR Multiplier", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
        config['keltner_atr_period'] = st.sidebar.number_input("ATR Period", min_value=2, max_value=50, value=10)
        config['keltner_mode'] = st.sidebar.selectbox(
            "Mode", ["Breakout (Trend Following)", "Mean Reversion (Band Re-entry)"], index=0)
        st.sidebar.info("📈 basis = EMA(close, N), bands = basis ± mult × ATR (TradingView default).")

    elif config['strategy'] == 'Heikin Ashi Trend Flip':
        st.sidebar.markdown("**Heikin Ashi Parameters**")
        config['ha_confirm_bars'] = st.sidebar.number_input(
            "Confirmation Candles", min_value=1, max_value=5, value=2,
            help="How many consecutive HA candles of the new colour before entering")
        config['ha_strong_candle_only'] = st.sidebar.checkbox(
            "Only 'strong' candles (no opposite wick)", value=False)
        st.sidebar.info("🕯️ HA_Close=(O+H+L+C)/4, HA_Open=(prevHAOpen+prevHAClose)/2 - identical to TradingView.")

    elif config['strategy'] == 'Heikin Ashi + EMA Confirmation':
        st.sidebar.markdown("**HA + EMA Parameters**")
        config['ema_fast'] = st.sidebar.number_input("EMA Fast Period", min_value=1, value=DEFAULT_EMA_FAST)
        config['ema_slow'] = st.sidebar.number_input("EMA Slow Period", min_value=1, value=DEFAULT_EMA_SLOW)
        st.sidebar.info("🕯️ HA colour flip taken only in the direction of the EMA trend.")

    elif config['strategy'] == 'MACD Signal Crossover':
        st.sidebar.markdown("**MACD Parameters**")
        config['macd_fast'] = st.sidebar.number_input("Fast Length", min_value=2, value=12)
        config['macd_slow'] = st.sidebar.number_input("Slow Length", min_value=3, value=26)
        config['macd_signal'] = st.sidebar.number_input("Signal Length", min_value=2, value=9)
        config['macd_zero_filter'] = st.sidebar.checkbox(
            "Only take crosses on the correct side of zero", value=False)

    elif config['strategy'] == 'Custom Strategy':
        st.sidebar.markdown("**🛠️ Custom Strategy Builder (Multi-Indicator)**")

        if 'custom_indicator_conditions' not in st.session_state:
            st.session_state['custom_indicator_conditions'] = [{}]

        conditions = st.session_state['custom_indicator_conditions']

        if len(conditions) > 1:
            config['custom_combine_mode'] = st.sidebar.radio(
                "Combine Conditions With",
                ["AND (all must be true)", "OR (any one true)"], index=0)
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

        STRATEGY_TYPE_OPTS = [
            "Price Crosses Indicator",
            "Price Pullback from Indicator",
            "Indicator Crosses Level",
            "Indicator Crossover",
        ]
        PRICE_INDICATOR_OPTS = ["EMA", "SMA", "BB Upper", "BB Lower", "BB Middle"]
        PULLBACK_INDICATOR_OPTS = ["EMA", "SMA", "BB Upper", "BB Lower"]
        LEVEL_INDICATOR_OPTS = [
            "RSI", "MACD", "MACD Histogram", "ADX", "Volume", "BB %B",
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

            if len(conditions) > 1:
                if st.sidebar.button(f"🗑️ Delete #{i+1}", key=f"del_cond_{i}"):
                    st.session_state['custom_indicator_conditions'].pop(i)
                    st.rerun()

            c = {}
            c['strategy_type'] = st.sidebar.selectbox(
                f"Type #{i+1}", STRATEGY_TYPE_OPTS,
                index=safe_index(STRATEGY_TYPE_OPTS, cond.get('strategy_type', STRATEGY_TYPE_OPTS[0])),
                key=f"cst_{i}")

            if c['strategy_type'] == "Price Crosses Indicator":
                c['indicator'] = st.sidebar.selectbox(
                    f"Indicator #{i+1}", PRICE_INDICATOR_OPTS,
                    index=safe_index(PRICE_INDICATOR_OPTS, cond.get('indicator', 'EMA')), key=f"ci_{i}")
                if c['indicator'] in ['EMA', 'SMA']:
                    c['period'] = st.sidebar.number_input(f"Period #{i+1}", min_value=1,
                                                          value=cond.get('period', 20), key=f"cp_{i}")
                elif 'BB' in c['indicator']:
                    c['bb_period'] = st.sidebar.number_input(f"BB Period #{i+1}", min_value=1,
                                                             value=cond.get('bb_period', 20), key=f"cbp_{i}")
                    c['bb_std'] = st.sidebar.number_input(f"BB Std #{i+1}", min_value=0.1,
                                                          value=cond.get('bb_std', 2.0), step=0.1, key=f"cbs_{i}")
                c['cross_type'] = st.sidebar.selectbox(
                    f"Cross #{i+1}", ["Above Indicator", "Below Indicator"],
                    index=0 if cond.get('cross_type', 'Above Indicator') == 'Above Indicator' else 1,
                    key=f"cct_{i}")
                c['position_type'] = st.sidebar.selectbox(
                    f"Position #{i+1}", ["LONG", "SHORT"],
                    index=0 if cond.get('position_type', 'LONG') == 'LONG' else 1, key=f"cpt_{i}")

            elif c['strategy_type'] == "Price Pullback from Indicator":
                c['indicator'] = st.sidebar.selectbox(
                    f"Indicator #{i+1}", PULLBACK_INDICATOR_OPTS,
                    index=safe_index(PULLBACK_INDICATOR_OPTS, cond.get('indicator', 'EMA')), key=f"ci_{i}")
                if c['indicator'] in ['EMA', 'SMA']:
                    c['period'] = st.sidebar.number_input(f"Period #{i+1}", min_value=1,
                                                          value=cond.get('period', 20), key=f"cp_{i}")
                elif 'BB' in c['indicator']:
                    c['bb_period'] = st.sidebar.number_input(f"BB Period #{i+1}", min_value=1,
                                                             value=cond.get('bb_period', 20), key=f"cbp_{i}")
                    c['bb_std'] = st.sidebar.number_input(f"BB Std #{i+1}", min_value=0.1,
                                                          value=cond.get('bb_std', 2.0), step=0.1, key=f"cbs_{i}")
                c['pullback_points'] = st.sidebar.number_input(
                    f"Pullback Pts #{i+1}", min_value=0.01,
                    value=float(cond.get('pullback_points', 10)), step=0.01, key=f"cpp_{i}")
                c['pullback_side'] = st.sidebar.selectbox(
                    f"Approach #{i+1}", ["Approach from Above", "Approach from Below"],
                    index=0 if cond.get('pullback_side', 'Approach from Above') == 'Approach from Above' else 1,
                    key=f"cps_{i}")
                c['position_type'] = st.sidebar.selectbox(
                    f"Position #{i+1}", ["LONG", "SHORT"],
                    index=0 if cond.get('position_type', 'LONG') == 'LONG' else 1, key=f"cpt_{i}")

            elif c['strategy_type'] == "Indicator Crosses Level":
                c['indicator'] = st.sidebar.selectbox(
                    f"Indicator #{i+1}", LEVEL_INDICATOR_OPTS,
                    index=safe_index(LEVEL_INDICATOR_OPTS, cond.get('indicator', 'RSI')), key=f"ci_{i}")

                ind = c['indicator']
                if ind == 'RSI':
                    c['rsi_period'] = st.sidebar.number_input(f"RSI Period #{i+1}", min_value=1,
                                                              value=cond.get('rsi_period', 14), key=f"crsi_{i}")
                    c['level'] = st.sidebar.number_input(f"Level #{i+1}", min_value=0.0, max_value=100.0,
                                                         value=float(cond.get('level', 50.0)), key=f"clv_{i}")
                elif ind in ['MACD', 'MACD Histogram']:
                    c['level'] = st.sidebar.number_input(f"Level #{i+1}",
                                                         value=float(cond.get('level', 0.0)), key=f"clv_{i}")
                elif ind == 'ADX':
                    c['level'] = st.sidebar.number_input(f"ADX Level #{i+1}", min_value=0.0,
                                                         value=float(cond.get('level', 25.0)), key=f"clv_{i}")
                elif ind == 'Volume':
                    c['volume_ma_period'] = st.sidebar.number_input(f"Vol MA Period #{i+1}", min_value=1,
                                                                    value=cond.get('volume_ma_period', 20), key=f"cvmp_{i}")
                    c['volume_multiplier'] = st.sidebar.number_input(f"Vol Mult #{i+1}", min_value=0.1,
                                                                     value=float(cond.get('volume_multiplier', 1.5)), step=0.1, key=f"cvm_{i}")
                elif ind == 'BB %B':
                    c['bb_period'] = st.sidebar.number_input(f"BB Period #{i+1}", min_value=1,
                                                             value=cond.get('bb_period', 20), key=f"cbp_{i}")
                    c['bb_std'] = st.sidebar.number_input(f"BB Std #{i+1}", min_value=0.1,
                                                          value=float(cond.get('bb_std', 2.0)), step=0.1, key=f"cbs_{i}")
                    c['level'] = st.sidebar.number_input(f"%B Level #{i+1} (0-100)", min_value=0.0, max_value=100.0,
                                                         value=float(cond.get('level', 80.0)), key=f"clv_{i}")
                elif ind == 'ATR (Volatility)':
                    c['atr_period'] = st.sidebar.number_input(f"ATR Period #{i+1}", min_value=1,
                                                              value=cond.get('atr_period', 14), key=f"catr_{i}")
                    c['level'] = st.sidebar.number_input(f"ATR Level #{i+1}", min_value=0.0,
                                                         value=float(cond.get('level', 10.0)), step=0.5, key=f"clv_{i}")
                elif ind == 'Historical Volatility':
                    c['hv_period'] = st.sidebar.number_input(f"HV Period #{i+1} (days)", min_value=5,
                                                             value=cond.get('hv_period', 20), key=f"chv_{i}")
                    c['level'] = st.sidebar.number_input(f"HV Level #{i+1} (%)", min_value=0.0,
                                                         value=float(cond.get('level', 20.0)), step=1.0, key=f"clv_{i}")
                elif ind == 'Std Dev (Volatility)':
                    c['stddev_period'] = st.sidebar.number_input(f"StdDev Period #{i+1}", min_value=2,
                                                                 value=cond.get('stddev_period', 20), key=f"csd_{i}")
                    c['level'] = st.sidebar.number_input(f"StdDev Level #{i+1}", min_value=0.0,
                                                         value=float(cond.get('level', 5.0)), step=0.5, key=f"clv_{i}")

                c['cross_type'] = st.sidebar.selectbox(
                    f"Cross #{i+1}", ["Above Level", "Below Level"],
                    index=0 if cond.get('cross_type', 'Above Level') == 'Above Level' else 1, key=f"cct_{i}")
                c['position_type'] = st.sidebar.selectbox(
                    f"Position #{i+1}", ["LONG", "SHORT"],
                    index=0 if cond.get('position_type', 'LONG') == 'LONG' else 1, key=f"cpt_{i}")

            elif c['strategy_type'] == "Indicator Crossover":
                c['crossover_type'] = st.sidebar.selectbox(
                    f"Crossover #{i+1}", CROSSOVER_OPTS,
                    index=safe_index(CROSSOVER_OPTS, cond.get('crossover_type', CROSSOVER_OPTS[0])),
                    key=f"cco_{i}")
                if c['crossover_type'] == "Fast EMA × Slow EMA":
                    c['fast_ema'] = st.sidebar.number_input(f"Fast EMA #{i+1}", min_value=1,
                                                            value=cond.get('fast_ema', 9), key=f"cfe_{i}")
                    c['slow_ema'] = st.sidebar.number_input(f"Slow EMA #{i+1}", min_value=1,
                                                            value=cond.get('slow_ema', 21), key=f"cse_{i}")
                elif c['crossover_type'] == "Fast SMA × Slow SMA":
                    c['fast_sma'] = st.sidebar.number_input(f"Fast SMA #{i+1}", min_value=1,
                                                            value=cond.get('fast_sma', 20), key=f"cfs_{i}")
                    c['slow_sma'] = st.sidebar.number_input(f"Slow SMA #{i+1}", min_value=1,
                                                            value=cond.get('slow_sma', 50), key=f"css_{i}")
                elif c['crossover_type'] in ["Price × EMA", "Price × SMA"]:
                    c['ma_period'] = st.sidebar.number_input(f"MA Period #{i+1}", min_value=1,
                                                             value=cond.get('ma_period', 50), key=f"cmap_{i}")
                elif c['crossover_type'] == "RSI Crossover (Overbought/Oversold)":
                    c['rsi_period'] = st.sidebar.number_input(f"RSI Period #{i+1}", min_value=1,
                                                              value=cond.get('rsi_period', 14), key=f"crsi_{i}")
                    c['rsi_ob'] = st.sidebar.number_input(f"Overbought #{i+1}", min_value=50.0, max_value=100.0,
                                                          value=float(cond.get('rsi_ob', 70.0)), key=f"crob_{i}")
                    c['rsi_os'] = st.sidebar.number_input(f"Oversold #{i+1}", min_value=0.0, max_value=50.0,
                                                          value=float(cond.get('rsi_os', 30.0)), key=f"cros_{i}")
                c['cross_type'] = st.sidebar.selectbox(
                    f"Direction #{i+1}", ["Bullish Crossover", "Bearish Crossover"],
                    index=0 if cond.get('cross_type', 'Bullish Crossover') == 'Bullish Crossover' else 1,
                    key=f"cct_{i}")
                c['position_type'] = st.sidebar.selectbox(
                    f"Position #{i+1}", ["LONG", "SHORT"],
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

    # EMA periods are always needed (indicator panel + charts)
    config.setdefault('ema_fast', DEFAULT_EMA_FAST)
    config.setdefault('ema_slow', DEFAULT_EMA_SLOW)
    config.setdefault('adx_period', DEFAULT_ADX_PERIOD)

    # =====================================================================
    # STOP LOSS   (defaults come from the USER DEFAULTS block at the top)
    # =====================================================================
    st.sidebar.subheader("🛡️ Stop Loss")
    config['sl_type'] = st.sidebar.selectbox(
        "SL Type", SL_TYPES, index=safe_index(SL_TYPES, DEFAULT_SL_TYPE))

    if 'Points' in config['sl_type'] or config['sl_type'] in [
            'Custom Points', 'ATR-based', 'Trailing SL (Points)',
            'Cost-to-Cost + N Points Trailing SL', 'Trailing SL + Current Candle',
            'Trailing SL + Previous Candle', 'Trailing SL + Current Swing',
            'Trailing SL + Previous Swing', 'Break-even After 50% Target']:
        config['sl_points'] = st.sidebar.number_input(
            "SL Points", min_value=1, value=DEFAULT_SL_POINTS)

    if 'Rupees' in config['sl_type'] or config['sl_type'] == 'P&L Based (Rupees)':
        config['sl_rupees'] = st.sidebar.number_input(
            "SL Rupees", min_value=1, value=DEFAULT_SL_RUPEES)

    if 'Trailing Profit' in config['sl_type'] or 'Trailing Loss' in config['sl_type']:
        config['sl_trail_rupees'] = st.sidebar.number_input(
            "Trail Rupees", min_value=1, value=DEFAULT_SL_TRAIL_RUPEES)

    if 'ATR' in config['sl_type'] or 'Volatility' in config['sl_type']:
        config['sl_atr_multiplier'] = st.sidebar.number_input(
            "SL ATR Multiplier", min_value=0.1, value=DEFAULT_SL_ATR_MULT, step=0.1)

    if config['sl_type'] == 'Cost-to-Cost + N Points Trailing SL':
        config['ctc_trigger_points'] = st.sidebar.number_input("Trigger Points (K)", min_value=1, value=3)
        config['ctc_offset_points'] = st.sidebar.number_input("Offset Points (N)", min_value=1, value=2)

    config.setdefault('sl_points', DEFAULT_SL_POINTS)

    # =====================================================================
    # TARGET
    # =====================================================================
    st.sidebar.subheader("🎯 Target")
    config['target_type'] = st.sidebar.selectbox(
        "Target Type", TARGET_TYPES, index=safe_index(TARGET_TYPES, DEFAULT_TARGET_TYPE))

    if 'Points' in config['target_type'] or config['target_type'] in [
            'Custom Points', 'Trailing Target (Points)', 'Trailing Target + Signal Based',
            '50% Exit at Target (Partial)']:
        config['target_points'] = st.sidebar.number_input(
            "Target Points", min_value=1, value=DEFAULT_TARGET_POINTS)

    if config['target_type'] == 'P&L Based (Rupees)':
        config['target_rupees'] = st.sidebar.number_input(
            "Target Rupees", min_value=1, value=DEFAULT_TARGET_RUPEES)

    if config['target_type'] == 'Risk-Reward Based':
        config['risk_reward_ratio'] = st.sidebar.number_input(
            "Risk:Reward Ratio", min_value=0.1, value=DEFAULT_RISK_REWARD, step=0.1)

    if config['target_type'] == 'ATR-based':
        config['target_atr_multiplier'] = st.sidebar.number_input(
            "Target ATR Multiplier", min_value=0.1, value=DEFAULT_TARGET_ATR_MULT, step=0.1)

    if config['target_type'] == 'Dynamic Trailing SL+Target (Lock Profits)':
        st.sidebar.info("Both SL and Target trail together as price moves favourably")
        config['dynamic_trail_sl_points'] = st.sidebar.number_input(
            "SL Distance (Points)", min_value=1, value=10)
        config['dynamic_trail_target_points'] = st.sidebar.number_input(
            "Target Distance (Points)", min_value=1, value=200)

    if config['target_type'] in ('Signal-based (Reverse Crossover)', 'Strategy-based Signal'):
        st.sidebar.info("🎯 No price target - the position is held until the strategy "
                        "produces the OPPOSITE signal (reverse crossover).")

    config.setdefault('target_points', DEFAULT_TARGET_POINTS)

    # =====================================================================
    # DHAN BROKER
    # =====================================================================
    st.sidebar.subheader("🏦 Dhan Broker (Optional)")
    config['dhan_enabled'] = st.sidebar.checkbox("Enable Dhan Broker", value=False)

    if config['dhan_enabled']:
        config['dhan_client_id'] = st.sidebar.text_input("Client ID", value=DEFAULT_DHAN_CLIENT_ID)
        config['dhan_access_token'] = st.sidebar.text_input(
            "Access Token", type="password", value=DEFAULT_DHAN_ACCESS_TOKEN,
            help="Paste your Dhan access token here. Never commit it to source control.")

        config['dhan_is_options'] = st.sidebar.checkbox("Is Options", value=False)

        if config['dhan_is_options']:
            config['dhan_ce_security_id'] = st.sidebar.text_input("CE Security ID", value="48228")
            config['dhan_pe_security_id'] = st.sidebar.text_input("PE Security ID", value="48229")
            config['dhan_strike_price'] = st.sidebar.number_input("Strike Price", min_value=0, value=25000)
            config['dhan_expiry_date'] = st.sidebar.date_input("Expiry Date", value=datetime.now().date())
            config['dhan_quantity'] = st.sidebar.number_input("Dhan Quantity", min_value=1, value=65)
        else:
            config['dhan_trading_type'] = st.sidebar.selectbox(
                "Trading Type", ["Intraday", "Delivery (CNC)"])
            config['dhan_security_id'] = st.sidebar.text_input("Security ID", value="12092")
            config['dhan_exchange'] = st.sidebar.selectbox("Exchange", ["NSE", "BSE"], index=0)
            config['dhan_quantity'] = st.sidebar.number_input("Quantity", min_value=1, value=10)

        st.sidebar.markdown("---")
        st.sidebar.markdown("**Order Type Configuration**")
        config['dhan_entry_order_type'] = st.sidebar.selectbox(
            "Entry Order Type", ["Market Order", "Limit Order"], index=1)
        config['dhan_exit_order_type'] = st.sidebar.selectbox(
            "Exit Order Type", ["Market Order", "Limit Order"], index=1)
        config['dhan_order_type'] = config['dhan_entry_order_type']

        config['broker_use_own_sl'] = st.sidebar.checkbox(
            "🎯 Use Broker SL/Target (Bracket Order)", value=False,
            help="Sends a Bracket Order with embedded SL and Target (values are DISTANCES in points).")
        if config['broker_use_own_sl']:
            config['broker_sl_points'] = st.sidebar.number_input(
                "SL Points (boStopLossValue)", min_value=0.5, value=50.0, step=0.5)
            config['broker_target_points'] = st.sidebar.number_input(
                "Target Points (boProfitValue)", min_value=0.5, value=100.0, step=0.5)
            config['broker_trailing_jump'] = st.sidebar.number_input(
                "Trail SL Jump (0 = off)", min_value=0.0, value=0.0, step=0.5)

        # ── Multi-Account Trading ────────────────────────────────────────
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🔀 Multi-Account Trading**")

        if 'multi_accounts' not in st.session_state:
            st.session_state['multi_accounts'] = []

        if st.session_state['multi_accounts']:
            st.sidebar.write(f"**Configured Accounts:** {len(st.session_state['multi_accounts'])}")
            for i, acc in enumerate(st.session_state['multi_accounts']):
                st.sidebar.caption(f"{i+1}. Client: {acc['client_id'][:8]}...")
                if st.sidebar.button("❌ remove", key=f"del_acc_{i}"):
                    st.session_state['multi_accounts'].pop(i)
                    st.rerun()

        with st.sidebar.expander("➕ Add Account"):
            new_client_id = st.text_input("Client ID", key="new_client_id")
            new_token = st.text_input("Access Token", type="password", key="new_token")
            if st.button("Add Account", key="add_account_btn"):
                if new_client_id and new_token:
                    st.session_state['multi_accounts'].append(
                        {'client_id': new_client_id, 'access_token': new_token})
                    st.rerun()
                else:
                    st.error("Please provide both Client ID and Token")

        config['multi_accounts'] = st.session_state['multi_accounts']

        # ── Multi-Strike Options ─────────────────────────────────────────
        if config['dhan_is_options']:
            st.sidebar.markdown("**📊 Multi-Strike Options**")
            config['multi_strike_enabled'] = st.sidebar.checkbox(
                "Enable Multi-Strike Orders", value=False)

            if config['multi_strike_enabled']:
                if 'multi_strikes_ce' not in st.session_state:
                    st.session_state['multi_strikes_ce'] = []
                if 'multi_strikes_pe' not in st.session_state:
                    st.session_state['multi_strikes_pe'] = []

                strike_type = st.sidebar.radio("Strike Type", ["CE (Call)", "PE (Put)"], horizontal=True)

                if strike_type == "CE (Call)":
                    if st.session_state['multi_strikes_ce']:
                        st.sidebar.write(f"**CE Strikes:** {len(st.session_state['multi_strikes_ce'])}")
                        for i, sec_id in enumerate(st.session_state['multi_strikes_ce']):
                            st.sidebar.caption(f"{i+1}. {sec_id}")
                            if st.sidebar.button("❌ remove", key=f"del_ce_{i}"):
                                st.session_state['multi_strikes_ce'].pop(i)
                                st.rerun()
                    with st.sidebar.expander("➕ Add CE Strike"):
                        new_ce_id = st.text_input("CE Security ID", key="new_ce_id")
                        if st.button("Add CE", key="add_ce_btn") and new_ce_id:
                            st.session_state['multi_strikes_ce'].append(new_ce_id)
                            st.rerun()
                else:
                    if st.session_state['multi_strikes_pe']:
                        st.sidebar.write(f"**PE Strikes:** {len(st.session_state['multi_strikes_pe'])}")
                        for i, sec_id in enumerate(st.session_state['multi_strikes_pe']):
                            st.sidebar.caption(f"{i+1}. {sec_id}")
                            if st.sidebar.button("❌ remove", key=f"del_pe_{i}"):
                                st.session_state['multi_strikes_pe'].pop(i)
                                st.rerun()
                    with st.sidebar.expander("➕ Add PE Strike"):
                        new_pe_id = st.text_input("PE Security ID", key="new_pe_id")
                        if st.button("Add PE", key="add_pe_btn") and new_pe_id:
                            st.session_state['multi_strikes_pe'].append(new_pe_id)
                            st.rerun()

                config['multi_strikes_ce'] = st.session_state.get('multi_strikes_ce', [])
                config['multi_strikes_pe'] = st.session_state.get('multi_strikes_pe', [])

        st.sidebar.markdown("---")
        config['clear_positions_before_entry'] = st.sidebar.checkbox(
            "🧹 Clear All Positions Before New Entry", value=False,
            help="Cancels pending orders and closes open positions before placing a new entry.")

    return config


def _format_money_cols(df, cols):
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—")
    return out


def build_price_chart(plot_df, config, trades=None, position=None, title=""):
    """Candles + EMA/BB/Donchian/Keltner overlays + trade markers."""
    fig = go.Figure()
    strategy = config.get('strategy', '')

    fig.add_trace(go.Candlestick(
        x=plot_df['Datetime'],
        open=plot_df['Open'], high=plot_df['High'],
        low=plot_df['Low'], close=plot_df['Close'],
        name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))

    if 'EMA_Fast' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df['Datetime'], y=plot_df['EMA_Fast'], mode='lines',
                                 name=f"EMA {config.get('ema_fast', DEFAULT_EMA_FAST)}",
                                 line=dict(color='#FF9800', width=1.5)))
    if 'EMA_Slow' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df['Datetime'], y=plot_df['EMA_Slow'], mode='lines',
                                 name=f"EMA {config.get('ema_slow', DEFAULT_EMA_SLOW)}",
                                 line=dict(color='#2196F3', width=1.5)))

    if strategy in ('Custom Strategy', 'Mean Reversion with Bollinger Bands',
                    'Bollinger Squeeze Breakout') and 'BB_Upper' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df['Datetime'], y=plot_df['BB_Upper'], mode='lines',
                                 name='BB Upper', line=dict(color='#9C27B0', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=plot_df['Datetime'], y=plot_df['BB_Lower'], mode='lines',
                                 name='BB Lower', line=dict(color='#9C27B0', width=1, dash='dot'),
                                 fill='tonexty', fillcolor='rgba(156,39,176,0.05)'))

    if strategy == 'Donchian Channel Breakout' and 'DC_Upper' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df['Datetime'], y=plot_df['DC_Upper'], mode='lines',
                                 name='Donchian Upper', line=dict(color='#00BCD4', width=1)))
        fig.add_trace(go.Scatter(x=plot_df['Datetime'], y=plot_df['DC_Lower'], mode='lines',
                                 name='Donchian Lower', line=dict(color='#00BCD4', width=1),
                                 fill='tonexty', fillcolor='rgba(0,188,212,0.05)'))

    if strategy == 'Keltner Channel Breakout' and 'KC_Upper' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df['Datetime'], y=plot_df['KC_Upper'], mode='lines',
                                 name='Keltner Upper', line=dict(color='#8BC34A', width=1)))
        fig.add_trace(go.Scatter(x=plot_df['Datetime'], y=plot_df['KC_Middle'], mode='lines',
                                 name='Keltner Basis', line=dict(color='#8BC34A', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=plot_df['Datetime'], y=plot_df['KC_Lower'], mode='lines',
                                 name='Keltner Lower', line=dict(color='#8BC34A', width=1),
                                 fill='tonexty', fillcolor='rgba(139,195,74,0.05)'))

    if strategy in ('Heikin Ashi Trend Flip', 'Heikin Ashi + EMA Confirmation') \
            and 'HA_Close' in plot_df.columns:
        fig.add_trace(go.Candlestick(
            x=plot_df['Datetime'], open=plot_df['HA_Open'], high=plot_df['HA_High'],
            low=plot_df['HA_Low'], close=plot_df['HA_Close'], name='Heikin Ashi',
            increasing_line_color='#7CB342', decreasing_line_color='#C62828', opacity=0.55))

    if trades:
        for tr in trades:
            et, xt = tr.get('entry_time'), tr.get('exit_time')
            ep, xp = tr.get('entry_price'), tr.get('exit_price')
            pos = tr.get('type', 'LONG')
            if et is not None and ep is not None:
                fig.add_trace(go.Scatter(
                    x=[et], y=[ep], mode='markers',
                    marker=dict(symbol='triangle-up' if pos == 'LONG' else 'triangle-down',
                                size=12, color='#00E676' if pos == 'LONG' else '#FF1744',
                                line=dict(width=1, color='black')),
                    name=f'Entry {pos}', showlegend=False,
                    hovertemplate=f"Entry {pos}<br>Price: {ep:.2f}<extra></extra>"))
            if xt is not None and xp is not None:
                fig.add_trace(go.Scatter(
                    x=[xt], y=[xp], mode='markers',
                    marker=dict(symbol='x', size=10, color='#FF6F00',
                                line=dict(width=2, color='black')),
                    name='Exit', showlegend=False,
                    hovertemplate=f"Exit<br>Price: {xp:.2f}<extra></extra>"))

    if position:
        fig.add_hline(y=position['entry_price'], line_dash='dash', line_color='#00E676',
                      annotation_text=f"Entry {position['type']} @ {position['entry_price']:.2f}",
                      annotation_position='right')
        if position.get('sl_price'):
            fig.add_hline(y=position['sl_price'], line_dash='dot', line_color='#FF1744',
                          annotation_text=f"SL @ {position['sl_price']:.2f}",
                          annotation_position='right')
        if position.get('target_price'):
            fig.add_hline(y=position['target_price'], line_dash='dot', line_color='#00BCD4',
                          annotation_text=f"Target @ {position['target_price']:.2f}",
                          annotation_position='right')

    fig.update_layout(
        title=title, xaxis_title='Time', yaxis_title='Price',
        xaxis_rangeslider_visible=False, height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified', template='plotly_dark')
    return fig


def build_chart_context(df, config, rows=40):
    """
    Serialise the chart into text for the chatbot.
    (A text LLM cannot see a picture, so the model receives the exact OHLC and
    indicator series the chart is drawn from.)
    """
    if df is None or len(df) == 0:
        return "(no chart data)"
    cols = [c for c in ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'EMA_Fast', 'EMA_Slow', 'EMA_Fast_Angle', 'RSI', 'ADX', 'ATR',
                        'BB_Upper', 'BB_Lower', 'VWAP', 'DC_Upper', 'DC_Lower',
                        'KC_Upper', 'KC_Lower', 'HA_Close'] if c in df.columns]
    return _df_to_context(df[cols], max_rows=rows)


# =============================================================================
# BACKTEST TAB
# =============================================================================

def render_backtest_ui(config):
    """Render backtesting interface with EMA plot, IST time filter and AI chat"""
    st.header("📈 Backtest Results")

    col_o1, col_o2 = st.columns(2)
    with col_o1:
        filter_market_hours = st.checkbox(
            "🕐 Filter Same-Day Trades Only (9:15 AM – 3:00 PM IST)", value=False,
            help="Shows only trades entered AND exited on the same day inside market hours.")
        use_method2 = st.checkbox(
            "🔬 Entry on NEXT candle open (N ➜ N+1)", value=DEFAULT_BACKTEST_NEXT_CANDLE_ENTRY,
            help=("Signal on candle N ➜ fill at the OPEN of candle N+1. "
                  "Removes look-ahead bias. This is the default now."))
    with col_o2:
        conservative = st.checkbox(
            "🛡️ Conservative intrabar exit (SL checked first)", value=DEFAULT_CONSERVATIVE_INTRABAR_EXIT,
            help=("Inside one candle the SL is tested against the LOW (long) / HIGH (short) FIRST; "
                  "the target is only tested if the SL was not touched. Gaps fill at the open."))
        trail_extremes = st.checkbox(
            "📉 Trail using candle extremes", value=True,
            help="Trailing stops move with the candle High (long) / Low (short) instead of the Close.")

    config['use_backtest_method2'] = use_method2
    config['conservative_intrabar_exit'] = conservative
    config['backtest_trail_on_extremes'] = trail_extremes

    st.caption(f"⚙️ Active rules — SL: **{config.get('sl_type')}** "
               f"({config.get('sl_points')} pts) · Target: **{config.get('target_type')}** · "
               f"Entry: **{'N+1 open' if use_method2 else 'N close'}** · "
               f"Exit priority: **{'SL first' if conservative else 'Target first'}**")

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            ticker = config.get('asset', DEFAULT_ASSET)
            interval = INTERVAL_MAPPING.get(config.get('interval', DEFAULT_INTERVAL), '1m')
            period = PERIOD_MAPPING.get(config.get('period', DEFAULT_PERIOD), '5d')
            custom_ticker = config.get('custom_ticker', None)

            df = fetch_data(ticker, interval, period, custom_ticker=custom_ticker)

            if df is not None:
                df = calculate_all_indicators(df, config)
                trades, metrics, debug_info, skipped_trades = run_backtest(df, config)

                st.session_state['backtest_results'] = {
                    'trades': trades,
                    'metrics': metrics,
                    'debug_info': debug_info,
                    'skipped_trades': skipped_trades,
                    'df': df
                }

    if 'backtest_results' not in st.session_state:
        st.info("Configure the strategy in the sidebar and press **Run Backtest**.")
        return

    results = st.session_state['backtest_results']
    all_trades = results['trades']
    debug_info = results['debug_info']
    df_chart = results.get('df')

    # ── same-day IST filter ───────────────────────────────────────────────
    if filter_market_hours and all_trades:
        filtered_trades = []
        for t in all_trades:
            et, xt = t.get('entry_time'), t.get('exit_time')
            try:
                if et is None or xt is None:
                    continue
                et_ist = et.astimezone(IST) if et.tzinfo else IST.localize(et)
                xt_ist = xt.astimezone(IST) if xt.tzinfo else IST.localize(xt)

                entry_ok = (et_ist.hour > 9 or (et_ist.hour == 9 and et_ist.minute >= 15)) and et_ist.hour < 15
                exit_ok = (xt_ist.hour > 9 or (xt_ist.hour == 9 and xt_ist.minute >= 15)) and xt_ist.hour < 15
                same_day = et_ist.date() == xt_ist.date()

                if entry_ok and exit_ok and same_day:
                    filtered_trades.append(t)
            except Exception:
                pass
        trades = filtered_trades
        st.info(f"🕐 Same-day filter: {len(trades)} / {len(all_trades)} trades shown")
    else:
        trades = all_trades

    # ── metrics on the filtered set ───────────────────────────────────────
    if trades:
        df_t = pd.DataFrame(trades)
        total_trades = len(df_t)
        winning_trades = int((df_t['pnl'] > 0).sum())
        losing_trades = int((df_t['pnl'] < 0).sum())
        win_rate = (winning_trades / total_trades * 100) if total_trades else 0
        total_pnl = float(df_t['pnl'].sum())
        avg_pnl = float(df_t['pnl'].mean())
        total_brokerage = float(df_t['brokerage'].sum())
        total_net_pnl = float(df_t['net_pnl'].sum())
        avg_net_pnl = float(df_t['net_pnl'].mean())
        cum_pnl = df_t['net_pnl'].cumsum()
        max_drawdown = float((cum_pnl - cum_pnl.cummax()).min())
    else:
        df_t = pd.DataFrame()
        total_trades = winning_trades = losing_trades = 0
        win_rate = total_pnl = avg_pnl = max_drawdown = 0.0
        total_brokerage = total_net_pnl = avg_net_pnl = 0.0

    metrics = dict(total_trades=total_trades, winning_trades=winning_trades,
                   losing_trades=losing_trades, win_rate=win_rate, total_pnl=total_pnl,
                   avg_pnl=avg_pnl, max_drawdown=max_drawdown, total_brokerage=total_brokerage,
                   total_net_pnl=total_net_pnl, avg_net_pnl=avg_net_pnl)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", metrics['total_trades'])
    col2.metric("Win Rate", f"{metrics['win_rate']:.2f}%")
    col3.metric("Total P&L", f"₹{metrics['total_pnl']:,.2f}")
    col4.metric("Avg Trade", f"₹{metrics['avg_pnl']:,.2f}")

    if config.get('include_brokerage', False):
        cb1, cb2, cb3, _ = st.columns(4)
        cb1.metric("Total Brokerage", f"₹{metrics['total_brokerage']:,.2f}")
        cb2.metric("Net P&L", f"₹{metrics['total_net_pnl']:,.2f}")
        cb3.metric("Avg Net P&L", f"₹{metrics['avg_net_pnl']:,.2f}")

    col5, col6, col7 = st.columns(3)
    col5.metric("Winning Trades", metrics['winning_trades'])
    col6.metric("Losing Trades", metrics['losing_trades'])
    col7.metric("Max Drawdown", f"₹{metrics['max_drawdown']:,.2f}")

    # ── chart ─────────────────────────────────────────────────────────────
    if df_chart is not None:
        st.subheader("📊 Price Chart with Indicator Overlay & Signals")
        plot_df = df_chart.tail(300).copy()
        min_dt = plot_df['Datetime'].min()
        vis_trades = [t for t in trades if t.get('entry_time') is not None
                      and t['entry_time'] >= min_dt] if trades else []
        fig = build_price_chart(
            plot_df, config, trades=vis_trades,
            title=f"{config.get('asset', 'Asset')} — {config.get('interval', '')}"
                  + (" [Same-day filter ON]" if filter_market_hours else ""))
        st.plotly_chart(fig, width="stretch")

    # ── trade table ───────────────────────────────────────────────────────
    if trades:
        st.subheader("✅ Executed Trade History")
        df_trades = pd.DataFrame(trades)
        df_show = _format_money_cols(df_trades, [
            'entry_price', 'exit_price', 'highest_price', 'lowest_price',
            'sl_price', 'target_price', 'pnl', 'net_pnl', 'brokerage'])

        if 'duration_minutes' in df_show.columns:
            df_show['duration'] = df_trades['duration_minutes'].apply(
                lambda x: f"{int(x)} min" if pd.notna(x) else "—")
        if 'ema_angle_entry' in df_show.columns:
            df_show['angle'] = df_trades['ema_angle_entry'].apply(
                lambda x: f"{x:.2f}°" if pd.notna(x) else "—")

        display_cols = ['entry_time', 'exit_time', 'type', 'duration', 'entry_price', 'exit_price',
                        'highest_price', 'lowest_price', 'sl_price', 'target_price',
                        'pnl', 'net_pnl', 'exit_reason']
        st.dataframe(df_show[[c for c in display_cols if c in df_show.columns]], width="stretch")

        with st.expander("📊 Detailed Trade Metrics (EMA, Angles, Differences)"):
            detailed_cols = ['entry_time', 'exit_time', 'type', 'strategy', 'pnl', 'net_pnl',
                             'brokerage', 'ema_fast_period', 'ema_slow_period', 'angle',
                             'ema_fast_entry', 'ema_slow_entry', 'price_fast_ema_diff_entry',
                             'price_slow_ema_diff_entry', 'fast_slow_ema_diff_entry',
                             'ema_fast_exit', 'ema_slow_exit', 'price_fast_ema_diff_exit',
                             'price_slow_ema_diff_exit', 'fast_slow_ema_diff_exit']
            df_det = df_show.copy()
            for col in ['ema_fast_entry', 'ema_slow_entry', 'price_fast_ema_diff_entry',
                        'price_slow_ema_diff_entry', 'fast_slow_ema_diff_entry',
                        'ema_fast_exit', 'ema_slow_exit', 'price_fast_ema_diff_exit',
                        'price_slow_ema_diff_exit', 'fast_slow_ema_diff_exit']:
                if col in df_trades.columns:
                    df_det[col] = df_trades[col].apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "—")
            st.dataframe(df_det[[c for c in detailed_cols if c in df_det.columns]], width="stretch")

    # ── skipped trades ────────────────────────────────────────────────────
    skipped_trades = results.get('skipped_trades', [])
    if skipped_trades and config.get('prevent_overlapping_trades', True):
        st.subheader("⚠️ Skipped/Overlapping Trades (Not Included in P&L)")
        st.info(f"**{len(skipped_trades)} signals were skipped** because a position was already "
                "active. Their P&L is NOT part of the metrics above.")
        df_skipped = pd.DataFrame(skipped_trades)
        st.dataframe(_format_money_cols(df_skipped, [
            'entry_price', 'exit_price', 'sl_price', 'target_price', 'pnl', 'net_pnl', 'brokerage']),
            width="stretch")

        sk1, sk2, sk3 = st.columns(3)
        sk1.metric("Skipped Winning", int((df_skipped['pnl'] > 0).sum()))
        sk2.metric("Skipped Losing", int((df_skipped['pnl'] <= 0).sum()))
        sk3.metric("Skipped Total P&L", f"₹{df_skipped['pnl'].sum():,.2f}")

    if metrics['total_trades'] == 0:
        st.warning("⚠️ No trades generated. Loosen the filters (angle / ADX) or widen the period.")

    with st.expander("🔍 Debug Information"):
        for k, v in debug_info.items():
            st.write(f"- **{k.replace('_', ' ').title()}**: {v}")

    # ── AI chat ───────────────────────────────────────────────────────────
    rows = int(config.get('groq_context_rows', 50))
    context = (
        f"TAB: BACKTEST\n"
        f"Asset={config.get('asset')} Interval={config.get('interval')} Period={config.get('period')}\n"
        f"Strategy={config.get('strategy')}  SL={config.get('sl_type')} ({config.get('sl_points')} pts)  "
        f"Target={config.get('target_type')}\n"
        f"EntryRule={debug_info.get('entry_rule')}  ExitRule={debug_info.get('exit_rule')}\n\n"
        f"METRICS:\n{metrics}\n\n"
        f"DEBUG:\n{debug_info}\n\n"
        f"TRADES (last {rows}, CSV):\n{_df_to_context(df_t, max_rows=rows)}\n\n"
        f"CHART SERIES (last {min(rows, 60)} candles of the plotted data, CSV):\n"
        f"{build_chart_context(df_chart, config, rows=min(rows, 60))}"
    )
    render_groq_chat("backtest", context, config,
                     title="🤖 Ask the AI about this backtest")


# =============================================================================
# LIVE TRADING TAB
# =============================================================================

def render_live_trading_ui(config):
    """Render live trading interface"""
    st.header("🔴 Live Trading")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("▶️ Start Trading", type="primary"):
            st.session_state['trading_active'] = True
            st.session_state['position'] = None
            st.session_state['broker_position'] = None
            st.session_state['live_logs'] = []
            st.session_state['last_exit_time'] = None
            st.session_state['last_signal_type'] = None
            st.session_state['last_entry_bar_time'] = None
            st.session_state['clearing_in_progress'] = False
            st.session_state['indicator_df'] = None
            st.session_state['last_candle_fetch_time'] = None

            if 'trade_history' not in st.session_state:
                st.session_state['trade_history'] = []
            st.session_state.pop('current_data', None)

            st.session_state['config'] = config

            add_log("🚀 Trading started - all sessions cleared")
            add_log(f"📋 Strategy: {config.get('strategy')} | Asset: {config.get('asset')} "
                    f"| Interval: {config.get('interval')}")
            add_log(f"📋 SL: {config.get('sl_type')} ({config.get('sl_points')} pts) "
                    f"| Target: {config.get('target_type')}")
            add_log(f"📋 Quantity: {config.get('quantity')}")
            add_log("🕯️ Signals evaluated on the LAST CLOSED candle "
                    f"({'ON' if LIVE_USE_CLOSED_CANDLE_ONLY else 'OFF'}) - one crossover = one entry")

            if config.get('enable_email_alerts', False):
                add_log(f"📧 Email alerts ON → {config.get('email_to')}")

            if config.get('dhan_enabled', False):
                add_log("🏦 Initializing Dhan broker...")
                st.session_state['dhan_broker'] = DhanBrokerIntegration(config)
            else:
                add_log("🏦 Dhan broker disabled (PAPER trading)")

    with col2:
        if st.button("⏹️ Stop Trading"):
            st.session_state['trading_active'] = False
            add_log("⏹️ Trading stopped")

    with col3:
        # "MANUAL SQUAREOFF FIX" - closes the position, keeps trading running
        if st.button("❌ Manual Square Off"):
            manual_square_off(config)

    with col4:
        pause = st.checkbox("⏸️ Pause auto-refresh", value=False,
                            key="pause_autorefresh",
                            help="Stops the 1.5s refresh loop so you can type in the AI chat "
                                 "or read the logs. Trading resumes when you untick it.")

    if st.session_state.get('trading_active', False):
        if pause:
            st.warning("🟡 Trading ACTIVE but auto-refresh is PAUSED (no new iterations)")
        else:
            st.success("🟢 Trading Active")
    else:
        st.info("⚪ Trading Inactive")

    # ── parameters ────────────────────────────────────────────────────────
    st.subheader("📋 Strategy Parameters")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.info(f"**Asset:** {config.get('asset', 'N/A')}")
        st.info(f"**Interval:** {config.get('interval', 'N/A')}")
    with p2:
        st.info(f"**Strategy:** {config.get('strategy', 'N/A')}")
        st.info(f"**Quantity:** {config.get('quantity', 'N/A')}")
    with p3:
        st.info(f"**SL Type:** {config.get('sl_type', 'N/A')}")
        st.info(f"**SL Points:** {config.get('sl_points', 'N/A')}")
    with p4:
        st.info(f"**Target Type:** {config.get('target_type', 'N/A')}")
        st.info(f"**Target Points:** {config.get('target_points', 'N/A')}")

    if config.get('strategy') == 'EMA Crossover':
        st.subheader("📊 EMA Strategy Parameters")
        e1, e2, e3, e4 = st.columns(4)
        e1.info(f"**Fast EMA:** {config.get('ema_fast', 'N/A')}")
        e2.info(f"**Slow EMA:** {config.get('ema_slow', 'N/A')}")
        e3.info(f"**Min Angle:** {config.get('ema_min_angle', 'N/A')}")
        e4.info(f"**Entry Filter:** {config.get('ema_entry_filter', 'N/A')}")

    # ── live chart ────────────────────────────────────────────────────────
    st.subheader("📊 Live Chart")
    df_live = None
    try:
        ticker_sym = config.get('asset', DEFAULT_ASSET)
        interval_code = INTERVAL_MAPPING.get(config.get('interval', DEFAULT_INTERVAL), '1m')
        period_code = PERIOD_MAPPING.get(config.get('period', DEFAULT_PERIOD), '5d')
        custom_ticker = config.get('custom_ticker', None)

        df_live = fetch_data(ticker_sym, interval_code, period_code, custom_ticker=custom_ticker)
        if df_live is not None and not df_live.empty:
            df_live = calculate_all_indicators(df_live, config)
            plot_df = df_live.tail(150).copy()
            fig_live = build_price_chart(
                plot_df, config, position=st.session_state.get('position'),
                title=f"Live: {config.get('asset', '')} | {config.get('interval', '')}")
            st.plotly_chart(fig_live, width="stretch")

            valid_fast = int(plot_df['EMA_Fast'].notna().sum())
            valid_slow = int(plot_df['EMA_Slow'].notna().sum())
            st.caption(f"🧮 Candles loaded: {len(df_live)} · EMA_Fast valid: {valid_fast}/{len(plot_df)} "
                       f"· EMA_Slow valid: {valid_slow}/{len(plot_df)} "
                       "(warm-up history is downloaded automatically, so these are never NaN at 09:15)")
        else:
            st.warning("⚠️ Could not load chart data")
    except Exception as e:
        st.warning(f"⚠️ Chart error: {e}")

    # ── current market data ───────────────────────────────────────────────
    current_data = st.session_state.get('current_data')
    live_price = st.session_state.get('live_price')

    if current_data is not None:
        st.subheader("📈 Current Market Data (indicators from the last CLOSED candle)")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Live Price", f"₹{live_price:,.2f}" if live_price else "N/A")
        m2.metric("Closed Candle", f"₹{current_data['Close']:,.2f}")
        m3.metric("Fast EMA", f"₹{current_data['EMA_Fast']:,.2f}"
                  if not pd.isna(current_data.get('EMA_Fast')) else "N/A")
        m4.metric("Slow EMA", f"₹{current_data['EMA_Slow']:,.2f}"
                  if not pd.isna(current_data.get('EMA_Slow')) else "N/A")
        m5.metric("EMA Angle", f"{current_data['EMA_Fast_Angle']:.2f}°"
                  if not pd.isna(current_data.get('EMA_Fast_Angle')) else "N/A")
        if not pd.isna(current_data.get('EMA_Fast')) and not pd.isna(current_data.get('EMA_Slow')):
            m6.metric("State", "Bullish ⬆️" if current_data['EMA_Fast'] > current_data['EMA_Slow']
                      else "Bearish ⬇️")
        else:
            m6.metric("State", "N/A")

        bt = st.session_state.get('signal_bar_time')
        if bt is not None:
            st.caption(f"🕯️ Signal candle: {pd.Timestamp(bt).strftime('%Y-%m-%d %H:%M:%S')} IST "
                       f"· total candles in memory: {st.session_state.get('candle_count', 0)}")

    # ── last candle details ───────────────────────────────────────────────
    if config.get('show_last_candle', False) and current_data is not None:
        st.subheader("📊 Last Candle Details")
        with st.expander("🔍 Complete candle data with all indicators", expanded=False):
            o1, o2, o3, o4, o5 = st.columns(5)
            o1.metric("Open", f"₹{current_data.get('Open', 0):,.2f}")
            o2.metric("High", f"₹{current_data.get('High', 0):,.2f}")
            o3.metric("Low", f"₹{current_data.get('Low', 0):,.2f}")
            o4.metric("Close", f"₹{current_data.get('Close', 0):,.2f}")
            o5.metric("Volume", f"{int(current_data.get('Volume', 0)):,}")

            if 'Datetime' in current_data.index:
                st.caption(f"🕐 Candle Time: {current_data['Datetime']}")

            st.markdown("---")
            common_indicators = [
                ('EMA_Fast', 'EMA Fast'), ('EMA_Slow', 'EMA Slow'),
                ('EMA_Fast_Angle', 'EMA Fast Angle'), ('EMA_Slow_Angle', 'EMA Slow Angle'),
                ('SMA_20', 'SMA 20'), ('SMA_50', 'SMA 50'), ('RSI', 'RSI'), ('ADX', 'ADX'),
                ('ATR', 'ATR'), ('MACD', 'MACD'), ('MACD_Signal', 'MACD Signal'),
                ('MACD_Hist', 'MACD Histogram'), ('BB_Upper', 'Bollinger Upper'),
                ('BB_Middle', 'Bollinger Middle'), ('BB_Lower', 'Bollinger Lower'),
                ('VWAP', 'VWAP'), ('DC_Upper', 'Donchian Upper'), ('DC_Lower', 'Donchian Lower'),
                ('KC_Upper', 'Keltner Upper'), ('KC_Middle', 'Keltner Basis'),
                ('KC_Lower', 'Keltner Lower'), ('HA_Open', 'HA Open'), ('HA_Close', 'HA Close'),
                ('SuperTrend', 'SuperTrend'), ('SuperTrend_Direction', 'SuperTrend Direction'),
                ('Volume_MA', 'Volume MA'),
            ]
            items = [(name, current_data[col]) for col, name in common_indicators
                     if col in current_data.index and not pd.isna(current_data.get(col))]

            if items:
                for i in range(0, len(items), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(items):
                            name, value = items[i + j]
                            with col:
                                if 'Angle' in name:
                                    st.metric(name, f"{value:.2f}°")
                                elif name in ['RSI', 'ADX']:
                                    st.metric(name, f"{value:.2f}")
                                elif 'Direction' in name:
                                    st.metric(name, "LONG ⬆️" if value == 1 else "SHORT ⬇️")
                                elif name == 'Volume MA':
                                    st.metric(name, f"{int(value):,}")
                                else:
                                    st.metric(name, f"₹{value:,.2f}")
            else:
                st.info("No indicators calculated yet.")

    # ── current position ──────────────────────────────────────────────────
    st.subheader("📊 Current Position")
    position = st.session_state.get('position')

    if position:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Type", position['type'])
        c2.metric("Entry Price", f"₹{position['entry_price']:,.2f}")
        c3.metric("Live Price", f"₹{live_price:,.2f}" if live_price else "N/A")
        c4.metric("Stop Loss", f"₹{position['sl_price']:,.2f}"
                  if position['sl_price'] is not None else "Not Set")
        c5.metric("Target", f"₹{position['target_price']:,.2f}"
                  if position['target_price'] is not None else "Not Set")

        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Quantity", position['quantity'])
        d2.metric("Locked Ticker", position.get('ticker', 'N/A'))
        if live_price:
            if position['type'] == 'LONG':
                cur_pnl = (live_price - position['entry_price']) * position['quantity']
            else:
                cur_pnl = (position['entry_price'] - live_price) * position['quantity']
            d3.metric("Current P&L", f"₹{cur_pnl:,.2f}", delta=f"₹{cur_pnl:,.2f}")
        else:
            d3.metric("Current P&L", "N/A")
        d4.metric("Highest Price", f"₹{position.get('highest_price', 0):,.2f}")
        d5.metric("Lowest Price", f"₹{position.get('lowest_price', 0):,.2f}")

        if position.get('ticker') != config.get('asset'):
            st.warning(f"⚠️ Position locked to {position.get('ticker')}. "
                       "Config changes won't affect the active position.")

        st.info(f"**Entry Time:** {position['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.info("No active position")

    # ── broker panel ──────────────────────────────────────────────────────
    if config.get('dhan_enabled', False) and st.session_state.get('broker_position'):
        st.subheader("🏦 Broker Position")
        bp = st.session_state['broker_position']
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Order ID", str(bp['order_id']))
        b2.metric("Option Type" if bp.get('is_options') else "Transaction",
                  bp.get('option_type') if bp.get('is_options') else bp.get('transaction_type'))
        b3.metric("Security ID", str(bp['security_id']))
        b4.metric("Status", bp['status'])
        with st.expander("📄 Raw API Response"):
            st.json(bp['raw_response'])

    # ── completed trades (numeric-safe) ───────────────────────────────────
    st.subheader("✅ Completed Trades")
    trade_history = st.session_state.get('trade_history', [])

    if trade_history:
        df_hist = pd.DataFrame(trade_history)

        # metrics computed from the RAW numeric frame (never from formatted strings)
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Total Trades", len(df_hist))
        h2.metric("Winning", int((df_hist['pnl'] > 0).sum()))
        h3.metric("Total P&L", f"₹{df_hist['pnl'].sum():,.2f}")
        h4.metric("Net P&L", f"₹{df_hist['net_pnl'].sum():,.2f}")

        df_disp = _format_money_cols(df_hist, [
            'entry_price', 'exit_price', 'highest_price', 'lowest_price',
            'pnl', 'net_pnl', 'brokerage', 'sl_price', 'target_price'])
        if 'duration_minutes' in df_hist.columns:
            df_disp['duration'] = df_hist['duration_minutes'].apply(
                lambda x: f"{int(x)} min" if pd.notna(x) else "—")
        if 'price_change_pct' in df_hist.columns:
            df_disp['⚠️'] = df_hist['price_change_pct'].apply(
                lambda x: '⚠️ CHECK' if pd.notna(x) and x > 50 else '')

        display_cols = ['entry_time', 'exit_time', 'ticker', 'type', 'duration', 'entry_price',
                        'exit_price', 'highest_price', 'lowest_price', 'pnl', 'net_pnl',
                        'exit_reason', '⚠️']
        st.dataframe(df_disp[[c for c in display_cols if c in df_disp.columns]],
                     width="stretch", height=240)
    else:
        st.info("No completed trades yet. Trades appear here immediately after exit.")

    # ── logs ──────────────────────────────────────────────────────────────
    st.subheader("📝 Trading Logs")
    logs = st.session_state.get('live_logs', [])
    if logs:
        st.text_area("Trading Logs", value="\n".join(reversed(logs[-60:])), height=300,
                     disabled=True, label_visibility="collapsed")
    else:
        st.info("No logs yet")

    # ── AI chat ───────────────────────────────────────────────────────────
    rows = int(config.get('groq_context_rows', 50))
    hist_df = pd.DataFrame(trade_history) if trade_history else pd.DataFrame()
    context = (
        f"TAB: LIVE TRADING\n"
        f"trading_active={st.session_state.get('trading_active', False)} "
        f"paused={st.session_state.get('pause_autorefresh', False)}\n"
        f"Asset={config.get('asset')} Interval={config.get('interval')} "
        f"Strategy={config.get('strategy')}\n"
        f"SL={config.get('sl_type')} ({config.get('sl_points')} pts) Target={config.get('target_type')}\n"
        f"Broker={'Dhan LIVE' if config.get('dhan_enabled') else 'PAPER'}\n\n"
        f"OPEN POSITION:\n{position if position else '(flat)'}\n\n"
        f"LIVE PRICE: {live_price}\n"
        f"LAST CLOSED CANDLE INDICATORS:\n"
        f"{current_data.to_dict() if current_data is not None else '(none)'}\n\n"
        f"COMPLETED TRADES (CSV):\n{_df_to_context(hist_df, max_rows=rows)}\n\n"
        f"RECENT LOGS:\n" + "\n".join(logs[-80:]) + "\n\n"
        f"CHART SERIES (CSV):\n{build_chart_context(df_live, config, rows=min(rows, 60))}"
    )
    render_groq_chat("live", context, config, title="🤖 Ask the AI about this live session")

    # ── auto-refresh + iteration ──────────────────────────────────────────
    if st.session_state.get('trading_active', False) and not pause:
        try:
            live_trading_iteration()
        except Exception as e:
            # Never let one bad iteration kill the loop
            add_log(f"❌ Iteration error: {e}")
            add_log(f"❌ {traceback.format_exc().splitlines()[-1]}")

        time.sleep(1.5)
        st.rerun()


# =============================================================================
# TRADE HISTORY TAB
# =============================================================================

def render_trade_logs_ui(config):
    """Render comprehensive trade history and statistics"""
    st.header("📊 Trade History & Statistics")

    trade_history = st.session_state.get('trade_history', [])

    if not trade_history:
        st.info("No trades recorded yet. Start live trading to see your trade history here.")
        render_groq_chat("history", "TAB: TRADE HISTORY\n(no trades recorded yet)", config,
                         title="🤖 Ask the AI about your trade history")
        return

    df_trades = pd.DataFrame(trade_history)

    total_trades = len(df_trades)
    profit_trades = int((df_trades['pnl'] > 0).sum())
    loss_trades = int((df_trades['pnl'] < 0).sum())

    total_pnl = float(df_trades['pnl'].sum())
    avg_pnl = float(df_trades['pnl'].mean())
    total_net = float(df_trades['net_pnl'].sum()) if 'net_pnl' in df_trades.columns else total_pnl

    avg_profit = float(df_trades[df_trades['pnl'] > 0]['pnl'].mean()) if profit_trades else 0.0
    avg_loss = float(df_trades[df_trades['pnl'] < 0]['pnl'].mean()) if loss_trades else 0.0
    accuracy = (profit_trades / total_trades * 100) if total_trades else 0.0

    cum = df_trades['pnl'].cumsum()
    max_dd = float((cum - cum.cummax()).min())

    st.subheader("📈 Overall Statistics")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Total Trades", total_trades)
    s2.metric("Profit Trades", profit_trades)
    s3.metric("Loss Trades", loss_trades)
    s4.metric("Accuracy", f"{accuracy:.2f}%")
    s5.metric("Total P&L", f"₹{total_pnl:,.2f}", delta=f"₹{total_pnl:,.2f}")

    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Avg P&L", f"₹{avg_pnl:,.2f}")
    t2.metric("Avg Profit", f"₹{avg_profit:,.2f}")
    t3.metric("Avg Loss", f"₹{avg_loss:,.2f}")
    t4.metric("Profit Factor", f"{abs(avg_profit / avg_loss):.2f}"
              if profit_trades and loss_trades and avg_loss else "N/A")
    t5.metric("Max Drawdown", f"₹{max_dd:,.2f}")

    if 'net_pnl' in df_trades.columns and config.get('include_brokerage', False):
        st.metric("Net P&L (after brokerage)", f"₹{total_net:,.2f}")

    # ── table ─────────────────────────────────────────────────────────────
    st.subheader("📋 Detailed Trade History")
    display_df = df_trades.copy()

    for col in ['entry_time', 'exit_time']:
        if col in display_df.columns:
            display_df[col] = pd.to_datetime(display_df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')

    numeric_cols = ['entry_price', 'exit_price', 'sl_price', 'target_price',
                    'highest_price', 'lowest_price', 'pnl', 'net_pnl', 'brokerage', 'price_range']
    display_df = _format_money_cols(display_df, numeric_cols)

    column_order = ['entry_time', 'exit_time', 'ticker', 'type', 'entry_price', 'exit_price',
                    'sl_price', 'target_price', 'highest_price', 'lowest_price',
                    'price_range', 'quantity', 'pnl', 'brokerage', 'net_pnl', 'exit_reason']
    display_df = display_df[[c for c in column_order if c in display_df.columns]]

    display_df = display_df.rename(columns={
        'entry_time': 'Entry Time', 'exit_time': 'Exit Time', 'ticker': 'Ticker',
        'type': 'Type', 'entry_price': 'Entry Price', 'exit_price': 'Exit Price',
        'sl_price': 'Stop Loss', 'target_price': 'Target', 'highest_price': 'Highest Price',
        'lowest_price': 'Lowest Price', 'price_range': 'Price Range', 'quantity': 'Quantity',
        'pnl': 'P&L', 'brokerage': 'Brokerage', 'net_pnl': 'Net P&L', 'exit_reason': 'Exit Reason'})

    st.dataframe(display_df, width="stretch", height=400)

    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Trade History (CSV)", data=csv,
        file_name=f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv")

    # ── charts ────────────────────────────────────────────────────────────
    st.subheader("📊 P&L Chart")
    chart_df = df_trades.copy()
    chart_df['cumulative_pnl'] = chart_df['pnl'].cumsum()
    chart_df['trade_number'] = range(1, len(chart_df) + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df['trade_number'], y=chart_df['cumulative_pnl'],
                             mode='lines+markers', name='Cumulative P&L',
                             line=dict(color='#2196F3', width=2), marker=dict(size=6)))
    fig.add_trace(go.Bar(x=chart_df['trade_number'], y=chart_df['pnl'], name='Trade P&L',
                         marker=dict(color=['green' if p > 0 else 'red' for p in chart_df['pnl']]),
                         opacity=0.6))
    fig.update_layout(title='Trade P&L Analysis', xaxis_title='Trade Number',
                      yaxis_title='P&L (₹)', hovermode='x unified', height=400,
                      template='plotly_dark')
    st.plotly_chart(fig, width="stretch")

    st.subheader("📊 Trade Type Distribution")
    d1, d2 = st.columns(2)
    with d1:
        type_counts = df_trades['type'].value_counts()
        fig_type = go.Figure(data=[go.Pie(labels=type_counts.index, values=type_counts.values, hole=0.4)])
        fig_type.update_layout(title='Long vs Short Trades', height=320, template='plotly_dark')
        st.plotly_chart(fig_type, width="stretch")
    with d2:
        reason_counts = df_trades['exit_reason'].value_counts()
        fig_reason = go.Figure(data=[go.Pie(labels=reason_counts.index, values=reason_counts.values, hole=0.4)])
        fig_reason.update_layout(title='Exit Reason Distribution', height=320, template='plotly_dark')
        st.plotly_chart(fig_reason, width="stretch")

    # ── AI chat ───────────────────────────────────────────────────────────
    rows = int(config.get('groq_context_rows', 50))
    stats = {
        'total_trades': total_trades, 'profit_trades': profit_trades, 'loss_trades': loss_trades,
        'accuracy_pct': round(accuracy, 2), 'total_pnl': round(total_pnl, 2),
        'net_pnl': round(total_net, 2), 'avg_pnl': round(avg_pnl, 2),
        'avg_profit': round(avg_profit, 2), 'avg_loss': round(avg_loss, 2),
        'max_drawdown': round(max_dd, 2),
    }
    equity_curve = chart_df[['trade_number', 'pnl', 'cumulative_pnl']]
    context = (
        f"TAB: TRADE HISTORY\n"
        f"Strategy={config.get('strategy')} Asset={config.get('asset')} "
        f"Interval={config.get('interval')}\n"
        f"SL={config.get('sl_type')} Target={config.get('target_type')}\n\n"
        f"STATS:\n{stats}\n\n"
        f"EXIT REASON COUNTS:\n{reason_counts.to_dict()}\n\n"
        f"DIRECTION COUNTS:\n{type_counts.to_dict()}\n\n"
        f"TRADES (CSV):\n{_df_to_context(df_trades, max_rows=rows)}\n\n"
        f"EQUITY CURVE (the P&L chart, as data):\n{_df_to_context(equity_curve, max_rows=rows)}"
    )
    render_groq_chat("history", context, config,
                     title="🤖 Ask the AI about your trade history")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application"""
    st.set_page_config(page_title="Algorithmic Trading System", page_icon="📈", layout="wide")

    st.title("📈 Algorithmic Trading System")

    config = render_config_ui()

    ticker_display = config.get('asset', DEFAULT_ASSET)
    if ticker_display == 'Custom Ticker':
        st.info(f"🎯 **Selected Ticker:** {config.get('custom_ticker', 'N/A')} (Custom)")
    else:
        st.info(f"🎯 **Selected Ticker:** {ticker_display} "
                f"({ASSET_MAPPING.get(ticker_display, ticker_display)})")

    badges = []
    badges.append(f"SL: {config.get('sl_type')} ({config.get('sl_points')} pts)")
    badges.append(f"Target: {config.get('target_type')}")
    badges.append("📧 Email ON" if config.get('enable_email_alerts') else "📧 Email OFF")
    badges.append("🤖 Groq ON" if config.get('enable_groq_chat') else "🤖 Groq OFF")
    badges.append("🏦 Dhan LIVE" if config.get('dhan_enabled') else "🧻 Paper")
    st.caption(" · ".join(badges))

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Backtest", "🔴 Live Trading", "📊 Trade History"])

    with tab1:
        render_backtest_ui(config)

    with tab2:
        render_live_trading_ui(config)

    with tab3:
        render_trade_logs_ui(config)


if __name__ == "__main__":
    main()
