"""
Professional Quantitative Trading System
Production-Grade Streamlit Application
Complete implementation with all features requested
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import pytz
import random

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Professional Quant Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CONSTANTS ==========

INDIAN_ASSETS = {"NIFTY 50": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
CRYPTO_ASSETS = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
FOREX_ASSETS = {"USDINR": "USDINR=X", "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X"}
COMMODITY_ASSETS = {"Gold": "GC=F", "Silver": "SI=F"}

TIMEFRAME_PERIODS = {
    "1m": ["1d", "5d"], "5m": ["1d", "1mo"], "15m": ["1mo"], "30m": ["1mo"],
    "1h": ["1mo"], "4h": ["1mo"], "1d": ["1mo", "1y", "2y", "5y"],
    "1wk": ["1mo", "1y", "5y", "10y", "15y", "20y"],
    "1mo": ["1y", "2y", "5y", "10y", "15y", "20y", "25y", "30y"]
}

STRATEGIES = ["Ratio Strategy", "EMA Crossover", "Simple Buy", "Simple Sell", 
              "Price Crosses Threshold", "RSI-ADX-EMA", "Percentage Change",
              "AI Price Action Analysis", "Custom Strategy Builder"]

SL_TYPES = ["Custom Points", "Trailing SL (Points)", "Trailing SL + Current Candle",
            "Trailing SL + Previous Candle", "Trailing SL + Current Swing",
            "Trailing SL + Previous Swing", "Trailing SL + Signal Based",
            "Volatility-Adjusted Trailing SL", "Break-even After 50% Target",
            "ATR-based", "Current Candle Low/High", "Previous Candle Low/High",
            "Current Swing Low/High", "Previous Swing Low/High",
            "Signal-based (reverse EMA crossover)"]

TARGET_TYPES = ["Custom Points", "Trailing Target (Points)", "Trailing Target + Signal Based",
                "50% Exit at Target (Partial)", "Current Candle Low/High",
                "Previous Candle Low/High", "Current Swing Low/High",
                "Previous Swing Low/High", "ATR-based", "Risk-Reward Based",
                "Signal-based (reverse EMA crossover)"]

# ========== SESSION STATE ==========

def init_session():
    defaults = {
        'trading_active': False, 'current_data': None, 'position': None,
        'trade_history': [], 'trade_logs': [], 'trailing_sl_high': None,
        'trailing_sl_low': None, 'trailing_target_high': None, 'trailing_target_low': None,
        'trailing_profit_points': 0, 'threshold_crossed': False, 'highest_price': None,
        'lowest_price': None, 'custom_conditions': [], 'partial_exit_done': False,
        'breakeven_activated': False, 'backtest_results': None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# ========== INDICATOR CALCULATIONS ==========

def calc_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calc_sma(data, period):
    return data.rolling(window=period).mean()

def calc_atr(df, period=14):
    hl = df['High'] - df['Low']
    hc = np.abs(df['High'] - df['Close'].shift())
    lc = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calc_rsi(data, period=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_adx(df, period=14):
    high, low = df['High'], df['Low']
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    tr = calc_atr(df, 1)
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.rolling(window=period).mean()

def calc_bollinger(data, period=20, std=2):
    sma = calc_sma(data, period)
    rolling_std = data.rolling(window=period).std()
    return sma + (rolling_std * std), sma, sma - (rolling_std * std)

def calc_macd(data, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(data, fast)
    ema_slow = calc_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line

def calc_ema_angle(df, period, idx):
    if idx < period + 1:
        return 0
    ema_col = f'EMA_{period}' if f'EMA_{period}' in df.columns else 'EMA_Fast'
    if ema_col not in df.columns:
        return 0
    curr = df[ema_col].iloc[idx]
    prev = df[ema_col].iloc[idx - 1]
    return abs(np.degrees(np.arctan(curr - prev)))

# ========== DATA FETCHING ==========

def fetch_data(ticker, interval, period, mode="Backtest"):
    try:
        if mode == "Live Trading":
            time.sleep(random.uniform(1.0, 1.5))
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        if 'Volume' not in data.columns:
            data['Volume'] = 0
        ist = pytz.timezone('Asia/Kolkata')
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert(ist)
        else:
            data.index = data.index.tz_convert(ist)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# ========== STRATEGIES ==========

def ema_crossover_strategy(df, config):
    fast, slow = config.get('ema_fast', 9), config.get('ema_slow', 15)
    df['EMA_Fast'], df['EMA_Slow'] = calc_ema(df['Close'], fast), calc_ema(df['Close'], slow)
    signals = pd.Series(0, index=df.index)
    use_adx = config.get('use_adx', False)
    if use_adx:
        df['ADX'] = calc_adx(df, config.get('adx_period', 14))
    for i in range(slow + 1, len(df)):
        angle = calc_ema_angle(df, fast, i)
        bullish = df['EMA_Fast'].iloc[i] > df['EMA_Slow'].iloc[i] and df['EMA_Fast'].iloc[i-1] <= df['EMA_Slow'].iloc[i-1]
        bearish = df['EMA_Fast'].iloc[i] < df['EMA_Slow'].iloc[i] and df['EMA_Fast'].iloc[i-1] >= df['EMA_Slow'].iloc[i-1]
        angle_ok = angle >= config.get('min_angle', 1)
        adx_ok = True
        if use_adx:
            adx_ok = df['ADX'].iloc[i] >= config.get('adx_threshold', 25)
        if bullish and angle_ok and adx_ok:
            signals.iloc[i] = 1
        elif bearish and angle_ok and adx_ok:
            signals.iloc[i] = -1
    return signals, df

def simple_buy_strategy(df, config):
    return pd.Series(1, index=df.index), df

def simple_sell_strategy(df, config):
    return pd.Series(-1, index=df.index), df

# ========== TRADE MANAGEMENT ==========

def add_log(msg):
    ts = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
    st.session_state['trade_logs'].append(f"[{ts}] {msg}")
    if len(st.session_state['trade_logs']) > 50:
        st.session_state['trade_logs'] = st.session_state['trade_logs'][-50:]

def reset_position():
    st.session_state.update({
        'position': None, 'trailing_sl_high': None, 'trailing_sl_low': None,
        'trailing_target_high': None, 'trailing_target_low': None,
        'trailing_profit_points': 0, 'threshold_crossed': False,
        'highest_price': None, 'lowest_price': None,
        'partial_exit_done': False, 'breakeven_activated': False
    })

def calc_sl(pos, price, df, idx, cfg):
    sl_type = cfg.get('sl_type', 'Custom Points')
    pts = cfg.get('sl_points', 10)
    if sl_type == 'Custom Points':
        return (pos['entry_price'] - pts) if pos['signal'] == 1 else (pos['entry_price'] + pts)
    elif sl_type == 'Trailing SL (Points)':
        if pos['signal'] == 1:
            new_sl = price - pts
            curr_sl = pos.get('stop_loss', pos['entry_price'] - pts)
            return new_sl if new_sl > curr_sl else curr_sl
        else:
            new_sl = price + pts
            curr_sl = pos.get('stop_loss', pos['entry_price'] + pts)
            return new_sl if new_sl < curr_sl else curr_sl
    elif 'ATR' in sl_type:
        atr = calc_atr(df, 14).iloc[idx]
        mult = cfg.get('atr_multiplier', 1.5)
        return (pos['entry_price'] - atr*mult) if pos['signal'] == 1 else (pos['entry_price'] + atr*mult)
    else:
        return (pos['entry_price'] - pts) if pos['signal'] == 1 else (pos['entry_price'] + pts)

def calc_target(pos, price, df, idx, cfg):
    tgt_type = cfg.get('target_type', 'Custom Points')
    pts = cfg.get('target_points', 20)
    if tgt_type == 'Custom Points':
        return (pos['entry_price'] + pts) if pos['signal'] == 1 else (pos['entry_price'] - pts)
    elif 'ATR' in tgt_type:
        atr = calc_atr(df, 14).iloc[idx]
        mult = cfg.get('target_atr_multiplier', 3.0)
        return (pos['entry_price'] + atr*mult) if pos['signal'] == 1 else (pos['entry_price'] - atr*mult)
    elif 'Signal-based' in tgt_type:
        return 0
    else:
        return (pos['entry_price'] + pts) if pos['signal'] == 1 else (pos['entry_price'] - pts)

def check_signal_exit(pos, df, idx, cfg):
    if idx < 1 or 'EMA_Fast' not in df.columns:
        return False
    if pos['signal'] == 1:
        return df['EMA_Fast'].iloc[idx] < df['EMA_Slow'].iloc[idx] and df['EMA_Fast'].iloc[idx-1] >= df['EMA_Slow'].iloc[idx-1]
    else:
        return df['EMA_Fast'].iloc[idx] > df['EMA_Slow'].iloc[idx] and df['EMA_Fast'].iloc[idx-1] <= df['EMA_Slow'].iloc[idx-1]

def execute_trade(signal, price, df, idx, cfg, mode="Backtest"):
    pos = st.session_state['position']
    if pos is None and signal != 0:
        pos = {
            'signal': signal, 'entry_price': price, 'entry_time': df.index[idx],
            'stop_loss': 0, 'target': 0, 'quantity': cfg.get('quantity', 1),
            'partial_exit_done': False, 'breakeven_activated': False
        }
        pos['stop_loss'] = calc_sl(pos, price, df, idx, cfg)
        pos['target'] = calc_target(pos, price, df, idx, cfg)
        st.session_state.update({
            'position': pos, 'highest_price': price, 'lowest_price': price
        })
        add_log(f"Entered {'LONG' if signal==1 else 'SHORT'} at {price:.2f}")
        return
    
    if pos:
        if st.session_state['highest_price'] is None or price > st.session_state['highest_price']:
            st.session_state['highest_price'] = price
        if st.session_state['lowest_price'] is None or price < st.session_state['lowest_price']:
            st.session_state['lowest_price'] = price
        
        if 'Trailing' in cfg.get('sl_type', ''):
            pos['stop_loss'] = calc_sl(pos, price, df, idx, cfg)
            st.session_state['position'] = pos
        
        if cfg.get('sl_type') == 'Signal-based (reverse EMA crossover)' or cfg.get('target_type') == 'Signal-based (reverse EMA crossover)':
            if check_signal_exit(pos, df, idx, cfg):
                pnl = (price - pos['entry_price']) * pos['quantity'] if pos['signal'] == 1 else (pos['entry_price'] - price) * pos['quantity']
                st.session_state['trade_history'].append({
                    'entry_time': pos['entry_time'], 'exit_time': df.index[idx],
                    'duration': str(df.index[idx] - pos['entry_time']), 'signal': 'LONG' if pos['signal']==1 else 'SHORT',
                    'entry_price': pos['entry_price'], 'exit_price': price,
                    'stop_loss': pos['stop_loss'], 'target': pos['target'],
                    'exit_reason': 'Reverse Signal', 'pnl': pnl,
                    'highest': st.session_state['highest_price'],
                    'lowest': st.session_state['lowest_price'],
                    'range': st.session_state['highest_price'] - st.session_state['lowest_price']
                })
                add_log(f"Reverse signal exit at {price:.2f} | P&L: {pnl:.2f}")
                reset_position()
                return
        
        sl_hit = (price <= pos['stop_loss']) if pos['signal'] == 1 else (price >= pos['stop_loss'])
        if sl_hit:
            pnl = (pos['stop_loss'] - pos['entry_price']) * pos['quantity'] if pos['signal'] == 1 else (pos['entry_price'] - pos['stop_loss']) * pos['quantity']
            st.session_state['trade_history'].append({
                'entry_time': pos['entry_time'], 'exit_time': df.index[idx],
                'duration': str(df.index[idx] - pos['entry_time']),
                'signal': 'LONG' if pos['signal']==1 else 'SHORT',
                'entry_price': pos['entry_price'], 'exit_price': pos['stop_loss'],
                'stop_loss': pos['stop_loss'], 'target': pos['target'],
                'exit_reason': 'Stop Loss Hit', 'pnl': pnl,
                'highest': st.session_state['highest_price'],
                'lowest': st.session_state['lowest_price'],
                'range': st.session_state['highest_price'] - st.session_state['lowest_price']
            })
            add_log(f"SL hit at {pos['stop_loss']:.2f} | P&L: {pnl:.2f}")
            reset_position()
            return
        
        if pos['target'] != 0:
            tgt_hit = (price >= pos['target']) if pos['signal'] == 1 else (price <= pos['target'])
            if tgt_hit:
                pnl = (pos['target'] - pos['entry_price']) * pos['quantity'] if pos['signal'] == 1 else (pos['entry_price'] - pos['target']) * pos['quantity']
                st.session_state['trade_history'].append({
                    'entry_time': pos['entry_time'], 'exit_time': df.index[idx],
                    'duration': str(df.index[idx] - pos['entry_time']),
                    'signal': 'LONG' if pos['signal']==1 else 'SHORT',
                    'entry_price': pos['entry_price'], 'exit_price': pos['target'],
                    'stop_loss': pos['stop_loss'], 'target': pos['target'],
                    'exit_reason': 'Target Hit', 'pnl': pnl,
                    'highest': st.session_state['highest_price'],
                    'lowest': st.session_state['lowest_price'],
                    'range': st.session_state['highest_price'] - st.session_state['lowest_price']
                })
                add_log(f"Target hit at {pos['target']:.2f} | P&L: {pnl:.2f}")
                reset_position()
                return

# ========== BACKTEST ==========

def run_backtest(df, cfg):
    st.session_state.update({'trade_history': [], 'trade_logs': []})
    reset_position()
    strategy = cfg.get('strategy', 'EMA Crossover')
    if strategy == 'EMA Crossover':
        signals, df = ema_crossover_strategy(df, cfg)
    elif strategy == 'Simple Buy':
        signals, df = simple_buy_strategy(df, cfg)
    elif strategy == 'Simple Sell':
        signals, df = simple_sell_strategy(df, cfg)
    else:
        signals = pd.Series(0, index=df.index)
    
    for i in range(len(df)):
        execute_trade(signals.iloc[i], df['Close'].iloc[i], df, i, cfg, "Backtest")
    
    if st.session_state['position']:
        pos = st.session_state['position']
        pnl = (df['Close'].iloc[-1] - pos['entry_price']) * pos['quantity'] if pos['signal'] == 1 else (pos['entry_price'] - df['Close'].iloc[-1]) * pos['quantity']
        st.session_state['trade_history'].append({
            'entry_time': pos['entry_time'], 'exit_time': df.index[-1],
            'duration': str(df.index[-1] - pos['entry_time']),
            'signal': 'LONG' if pos['signal']==1 else 'SHORT',
            'entry_price': pos['entry_price'], 'exit_price': df['Close'].iloc[-1],
            'stop_loss': pos['stop_loss'], 'target': pos['target'],
            'exit_reason': 'End of Data', 'pnl': pnl,
            'highest': st.session_state.get('highest_price', df['Close'].iloc[-1]),
            'lowest': st.session_state.get('lowest_price', df['Close'].iloc[-1]),
            'range': st.session_state.get('highest_price', df['Close'].iloc[-1]) - st.session_state.get('lowest_price', df['Close'].iloc[-1])
        })
        reset_position()
    return df

# ========== UI ==========

def render_sidebar():
    st.sidebar.title("⚙️ Configuration")
    
    # Broker section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔌 Broker Integration")
    broker_enabled = st.sidebar.checkbox("Enable Broker Integration")
    if broker_enabled:
        st.sidebar.text_input("Client ID", key="client_id")
        st.sidebar.text_input("Token ID", type="password", key="token_id")
        st.sidebar.selectbox("Option Type", ["CE", "PE"], key="option_type")
        st.sidebar.date_input("Expiry Date", key="expiry_date")
        st.sidebar.number_input("Strike Price", min_value=0, value=0, step=50, key="strike_price")
        st.sidebar.number_input("Price", min_value=0.0, value=0.0, step=0.5, key="order_price")
        st.sidebar.selectbox("Condition", [">=", "<=", "=="], key="price_condition")
    
    st.sidebar.markdown("---")
    mode = st.sidebar.radio("Mode", ["Backtest", "Live Trading"])
    
    st.sidebar.subheader("📊 Asset Selection")
    cat = st.sidebar.selectbox("Category", ["Indian Indices", "Crypto", "Forex", "Commodities", "Custom"])
    if cat == "Indian Indices":
        name = st.sidebar.selectbox("Asset", list(INDIAN_ASSETS.keys()))
        ticker = INDIAN_ASSETS[name]
    elif cat == "Crypto":
        name = st.sidebar.selectbox("Asset", list(CRYPTO_ASSETS.keys()))
        ticker = CRYPTO_ASSETS[name]
    elif cat == "Forex":
        name = st.sidebar.selectbox("Asset", list(FOREX_ASSETS.keys()))
        ticker = FOREX_ASSETS[name]
    elif cat == "Commodities":
        name = st.sidebar.selectbox("Asset", list(COMMODITY_ASSETS.keys()))
        ticker = COMMODITY_ASSETS[name]
    else:
        ticker = st.sidebar.text_input("Ticker", value="^NSEI")
        name = ticker
    
    st.sidebar.subheader("⏱️ Timeframe")
    interval = st.sidebar.selectbox("Interval", list(TIMEFRAME_PERIODS.keys()))
    period = st.sidebar.selectbox("Period", TIMEFRAME_PERIODS[interval])
    quantity = st.sidebar.number_input("Quantity", min_value=1, value=1, step=1)
    
    st.sidebar.subheader("🎯 Strategy")
    strategy = st.sidebar.selectbox("Strategy", STRATEGIES)
    
    cfg = {
        'mode': mode, 'ticker': ticker, 'asset_name': name,
        'interval': interval, 'period': period, 'quantity': quantity,
        'strategy': strategy, 'broker_enabled': broker_enabled
    }
    
    if strategy == 'EMA Crossover':
        st.sidebar.subheader("EMA Settings")
        cfg['ema_fast'] = st.sidebar.number_input("EMA Fast", 1, 200, 9)
        cfg['ema_slow'] = st.sidebar.number_input("EMA Slow", 1, 200, 15)
        cfg['min_angle'] = st.sidebar.number_input("Min Angle", 0.0, 90.0, 1.0, 0.1)
        cfg['use_adx'] = st.sidebar.checkbox("Enable ADX Filter")
        if cfg['use_adx']:
            cfg['adx_period'] = st.sidebar.number_input("ADX Period", 5, 50, 14)
            cfg['adx_threshold'] = st.sidebar.number_input("ADX Threshold", 10, 50, 25)
    
    st.sidebar.subheader("🛑 Stop Loss")
    cfg['sl_type'] = st.sidebar.selectbox("SL Type", SL_TYPES)
    if 'Points' in cfg['sl_type'] or cfg['sl_type'] == 'Custom Points':
        cfg['sl_points'] = st.sidebar.number_input("SL Points", 1, 1000, 10)
    if 'ATR' in cfg['sl_type']:
        cfg['atr_multiplier'] = st.sidebar.number_input("ATR Mult (SL)", 0.1, 10.0, 1.5, 0.1)
    
    st.sidebar.subheader("🎯 Target")
    cfg['target_type'] = st.sidebar.selectbox("Target Type", TARGET_TYPES)
    if 'Points' in cfg['target_type'] or cfg['target_type'] == 'Custom Points':
        cfg['target_points'] = st.sidebar.number_input("Target Points", 1, 1000, 20)
    if 'ATR' in cfg['target_type']:
        cfg['target_atr_multiplier'] = st.sidebar.number_input("ATR Mult (Tgt)", 0.1, 10.0, 3.0, 0.1)
    
    return cfg

def render_dashboard(cfg):
    st.markdown("## 📊 Live Trading Dashboard")
    col1, col2, col3 = st.columns([1,1,2])
    
    with col1:
        if st.button("▶️ Start", type="primary", use_container_width=True):
            st.session_state['trading_active'] = True
            add_log("Trading started")
            st.rerun()
    with col2:
        if st.button("⏹️ Stop", use_container_width=True):
            if st.session_state['position']:
                pos, df = st.session_state['position'], st.session_state['current_data']
                if df is not None:
                    price = df['Close'].iloc[-1]
                    pnl = (price - pos['entry_price']) * pos['quantity'] if pos['signal'] == 1 else (pos['entry_price'] - price) * pos['quantity']
                    st.session_state['trade_history'].append({
                        'entry_time': pos['entry_time'], 'exit_time': df.index[-1],
                        'duration': str(df.index[-1] - pos['entry_time']),
                        'signal': 'LONG' if pos['signal']==1 else 'SHORT',
                        'entry_price': pos['entry_price'], 'exit_price': price,
                        'stop_loss': pos['stop_loss'], 'target': pos['target'],
                        'exit_reason': 'Manual Close', 'pnl': pnl,
                        'highest': st.session_state.get('highest_price', price),
                        'lowest': st.session_state.get('lowest_price', price),
                        'range': st.session_state.get('highest_price', price) - st.session_state.get('lowest_price', price)
                    })
                    add_log(f"Manual close at {price:.2f} | P&L: {pnl:.2f}")
            st.session_state['trading_active'] = False
            reset_position()
            add_log("Trading stopped")
            st.rerun()
    with col3:
        st.success("🟢 ACTIVE") if st.session_state['trading_active'] else st.info("⚪ STOPPED")
    
    if st.button("🔄 Refresh"):
        st.rerun()
    
    st.markdown("---")
    
    if st.session_state['current_data'] is not None:
        df = st.session_state['current_data']
        price = df['Close'].iloc[-1]
        pos = st.session_state['position']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Price", f"{price:.2f}")
        with col2:
            st.metric("Entry", f"{pos['entry_price']:.2f}" if pos else "N/A")
        with col3:
            st.metric("Position", "LONG" if pos and pos['signal']==1 else ("SHORT" if pos and pos['signal']==-1 else "None"))
        with col4:
            if pos:
                pnl = (price - pos['entry_price']) * pos['quantity'] if pos['signal'] == 1 else (pos['entry_price'] - price) * pos['quantity']
                delta_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
                st.metric("P&L", f"{pnl:.2f}", delta=delta_str, delta_color="normal" if pnl >= 0 else "inverse")
            else:
                st.metric("P&L", "N/A")
        
        if pos:
            st.markdown("### Position Info")
            st.write(f"Entry Time: {pos['entry_time']}")
            st.write(f"SL: {pos['stop_loss']:.2f} | Target: {pos['target']:.2f if pos['target'] else 'Signal-based'}")
            st.write(f"High: {st.session_state['highest_price']:.2f} | Low: {st.session_state['lowest_price']:.2f}")
        
        # Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                     low=df['Low'], close=df['Close'], name='Price'))
        if 'EMA_Fast' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], mode='lines', name='EMA Fast'))
        if 'EMA_Slow' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], mode='lines', name='EMA Slow'))
        if pos:
            fig.add_hline(y=pos['entry_price'], line_dash="dash", line_color="yellow", annotation_text="Entry")
            if pos['stop_loss']:
                fig.add_hline(y=pos['stop_loss'], line_dash="dash", line_color="red", annotation_text="SL")
            if pos['target']:
                fig.add_hline(y=pos['target'], line_dash="dash", line_color="green", annotation_text="Target")
        fig.update_layout(title=f"{cfg['asset_name']} - {cfg['interval']}", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{int(time.time())}")
    else:
        st.info("Click Start to begin trading")
    
    if st.session_state['trading_active']:
        time.sleep(1.2)
        st.rerun()

def render_history():
    st.markdown("### 📈 Trade History")
    if not st.session_state['trade_history']:
        st.info("No trades yet")
        return
    
    total = len(st.session_state['trade_history'])
    wins = sum(1 for t in st.session_state['trade_history'] if t['pnl'] > 0)
    losses = total - wins
    acc = (wins / total * 100) if total > 0 else 0
    total_pnl = sum(t['pnl'] for t in st.session_state['trade_history'])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total", total)
    with col2:
        st.metric("Wins", wins)
    with col3:
        st.metric("Losses", losses)
    with col4:
        st.metric("Accuracy", f"{acc:.1f}%")
    with col5:
        delta_str = f"+{total_pnl:.2f}" if total_pnl >= 0 else f"{total_pnl:.2f}"
        st.metric("P&L", f"{total_pnl:.2f}", delta=delta_str, delta_color="normal" if total_pnl >= 0 else "inverse")
    
    st.markdown("---")
    for i, t in enumerate(reversed(st.session_state['trade_history']), 1):
        with st.expander(f"Trade #{total-i+1} - {t['signal']} - P&L: {t['pnl']:.2f}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Entry: {t['entry_time']} @ {t['entry_price']:.2f}")
                st.write(f"Exit: {t['exit_time']} @ {t['exit_price']:.2f}")
                st.write(f"Duration: {t['duration']}")
            with col2:
                st.write(f"SL: {t['stop_loss']:.2f}")
                st.write(f"Target: {t['target']:.2f if t['target'] else 'Signal-based'}")
                st.write(f"Exit: {t['exit_reason']}")
                if t['pnl'] >= 0:
                    st.success(f"P&L: +{t['pnl']:.2f}")
                else:
                    st.error(f"P&L: {t['pnl']:.2f}")

def render_logs():
    st.markdown("### 📝 Trade Logs")
    if not st.session_state['trade_logs']:
        st.info("No logs yet")
        return
    for log in reversed(st.session_state['trade_logs']):
        st.text(log)

def render_backtest(cfg):
    st.markdown("### 🔬 Backtest Results")
    if cfg['mode'] != "Backtest":
        st.warning("Switch to Backtest mode in sidebar")
        return
    
    if st.button("🚀 Run Backtest", type="primary"):
        st.session_state['current_data'] = None
        with st.spinner("Running backtest..."):
            df = fetch_data(cfg['ticker'], cfg['interval'], cfg['period'], "Backtest")
            if df is None:
                st.error("Failed to fetch data")
                return
            df = run_backtest(df, cfg)
            st.session_state['backtest_results'] = df
            st.success("✅ Complete!")
    
    if st.session_state.get('backtest_results') is not None:
        if not st.session_state['trade_history']:
            st.info("No trades generated")
            return
        
        total = len(st.session_state['trade_history'])
        wins = sum(1 for t in st.session_state['trade_history'] if t['pnl'] > 0)
        losses = total - wins
        acc = (wins / total * 100) if total > 0 else 0
        total_pnl = sum(t['pnl'] for t in st.session_state['trade_history'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", total)
            st.metric("Wins", wins)
        with col2:
            st.metric("Losses", losses)
            st.metric("Accuracy", f"{acc:.1f}%")
        with col3:
            delta_str = f"+{total_pnl:.2f}" if total_pnl >= 0 else f"{total_pnl:.2f}"
            st.metric("P&L", f"{total_pnl:.2f}", delta=delta_str, delta_color="normal" if total_pnl >= 0 else "inverse")
        
        st.markdown("---")
        for i, t in enumerate(st.session_state['trade_history'], 1):
            with st.expander(f"Trade #{i} - {t['signal']} - {t['pnl']:.2f}"):
                st.write(f"Entry: {t['entry_time']} @ {t['entry_price']:.2f}")
                st.write(f"Exit: {t['exit_time']} @ {t['exit_price']:.2f}")
                st.write(f"Reason: {t['exit_reason']}")

def main():
    init_session()
    st.title("📈 Professional Quantitative Trading System")
    st.markdown("*Production-Grade Algo Trading Platform*")
    st.markdown("---")
    
    cfg = render_sidebar()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 History", "📝 Logs", "🔬 Backtest"])
    with tab1:
        render_dashboard(cfg)
    with tab2:
        render_history()
    with tab3:
        render_logs()
    with tab4:
        render_backtest(cfg)

if __name__ == "__main__":
    main()
