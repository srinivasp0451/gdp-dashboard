import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import pytz
import time
import random

def log_message(msg):
    ts = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
    st.session_state['trade_logs'].append(f"[{ts}] {msg}")
    if len(st.session_state['trade_logs']) > 50:
        st.session_state['trade_logs'] = st.session_state['trade_logs'][-50:]

def reset_position_state():
    st.session_state['position'] = None
    st.session_state['highest_price'] = None
    st.session_state['lowest_price'] = None
    st.session_state['partial_exit_done'] = False
    st.session_state['breakeven_activated'] = False

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    hl = df['High'] - df['Low']
    hc = np.abs(df['High'] - df['Close'].shift())
    lc = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_adx(df, period=14):
    high = df['High']
    low = df['Low']
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    atr = calculate_atr(df, period)
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.ewm(span=period, adjust=False).mean()

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line

def calculate_bollinger_bands(data, period=20, std_dev=2):
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    return sma + (std * std_dev), sma, sma - (std * std_dev)

def calculate_ema_angle(df, column, period=9):
    ema = calculate_ema(df[column], period)
    angles = []
    for i in range(1, len(ema)):
        if pd.isna(ema.iloc[i]) or pd.isna(ema.iloc[i-1]):
            angles.append(0)
        else:
            slope = ema.iloc[i] - ema.iloc[i-1]
            angles.append(abs(np.degrees(np.arctan(slope))))
    return pd.Series([0] + angles, index=df.index)

def generate_ema_crossover_signal(df, config):
    df['EMA_Fast'] = calculate_ema(df['Close'], config['ema_fast'])
    df['EMA_Slow'] = calculate_ema(df['Close'], config['ema_slow'])
    df['EMA_Angle'] = calculate_ema_angle(df, 'Close', config['ema_fast'])
    
    if config.get('use_adx'):
        df['ADX'] = calculate_adx(df, config.get('adx_period', 14))
    if config['entry_filter'] == 'ATR-based Candle':
        df['ATR'] = calculate_atr(df, 14)
    
    signals = []
    for i in range(1, len(df)):
        signal = 0
        if (df['EMA_Fast'].iloc[i] > df['EMA_Slow'].iloc[i] and 
            df['EMA_Fast'].iloc[i-1] <= df['EMA_Slow'].iloc[i-1]):
            if df['EMA_Angle'].iloc[i] >= config['min_angle']:
                candle_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
                if config['entry_filter'] == 'Simple Crossover':
                    signal = 1
                elif config['entry_filter'] == 'Custom Candle (Points)':
                    signal = 1 if candle_size >= config.get('custom_points', 10) else 0
                elif config['entry_filter'] == 'ATR-based Candle':
                    signal = 1 if candle_size >= df['ATR'].iloc[i] * config.get('atr_multiplier', 1.5) else 0
                if signal == 1 and config.get('use_adx') and df['ADX'].iloc[i] < config.get('adx_threshold', 25):
                    signal = 0
        elif (df['EMA_Fast'].iloc[i] < df['EMA_Slow'].iloc[i] and 
              df['EMA_Fast'].iloc[i-1] >= df['EMA_Slow'].iloc[i-1]):
            if df['EMA_Angle'].iloc[i] >= config['min_angle']:
                candle_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
                if config['entry_filter'] == 'Simple Crossover':
                    signal = -1
                elif config['entry_filter'] == 'Custom Candle (Points)':
                    signal = -1 if candle_size >= config.get('custom_points', 10) else 0
                elif config['entry_filter'] == 'ATR-based Candle':
                    signal = -1 if candle_size >= df['ATR'].iloc[i] * config.get('atr_multiplier', 1.5) else 0
                if signal == -1 and config.get('use_adx') and df['ADX'].iloc[i] < config.get('adx_threshold', 25):
                    signal = 0
        signals.append(signal)
    df['Signal'] = [0] + signals
    return df

def generate_simple_buy_signal(df):
    df['Signal'] = 1
    df['EMA_Fast'] = calculate_ema(df['Close'], 9)
    df['EMA_Slow'] = calculate_ema(df['Close'], 21)
    return df

def generate_simple_sell_signal(df):
    df['Signal'] = -1
    df['EMA_Fast'] = calculate_ema(df['Close'], 9)
    df['EMA_Slow'] = calculate_ema(df['Close'], 21)
    return df

def generate_price_threshold_signal(df, config):
    df['EMA_Fast'] = calculate_ema(df['Close'], 9)
    df['EMA_Slow'] = calculate_ema(df['Close'], 21)
    signals = []
    for i in range(len(df)):
        signal = 0
        p = df['Close'].iloc[i]
        t = config['threshold']
        d = config['direction']
        if d == 'LONG (Price >= Threshold)' and p >= t:
            signal = 1
        elif d == 'SHORT (Price >= Threshold)' and p >= t:
            signal = -1
        elif d == 'LONG (Price <= Threshold)' and p <= t:
            signal = 1
        elif d == 'SHORT (Price <= Threshold)' and p <= t:
            signal = -1
        signals.append(signal)
    df['Signal'] = signals
    return df

def generate_rsi_adx_ema_signal(df, config):
    df['RSI'] = calculate_rsi(df['Close'], config.get('rsi_period', 14))
    df['ADX'] = calculate_adx(df, config.get('adx_period', 14))
    df['EMA_Fast'] = calculate_ema(df['Close'], config.get('ema1_period', 9))
    df['EMA_Slow'] = calculate_ema(df['Close'], config.get('ema2_period', 21))
    signals = []
    for i in range(len(df)):
        if df['RSI'].iloc[i] > 80 and df['ADX'].iloc[i] < 20 and df['EMA_Fast'].iloc[i] < df['EMA_Slow'].iloc[i]:
            signals.append(-1)
        elif df['RSI'].iloc[i] < 20 and df['ADX'].iloc[i] > 20 and df['EMA_Fast'].iloc[i] > df['EMA_Slow'].iloc[i]:
            signals.append(1)
        else:
            signals.append(0)
    df['Signal'] = signals
    return df

def generate_percentage_change_signal(df, config):
    df['EMA_Fast'] = calculate_ema(df['Close'], 9)
    df['EMA_Slow'] = calculate_ema(df['Close'], 21)
    first_price = df['Close'].iloc[0]
    signals = []
    for i in range(len(df)):
        pct = ((df['Close'].iloc[i] - first_price) / first_price) * 100
        signal = 0
        d = config['direction']
        t = config['percentage_threshold']
        if d == 'BUY on Fall' and pct <= -t:
            signal = 1
        elif d == 'SELL on Fall' and pct <= -t:
            signal = -1
        elif d == 'BUY on Rise' and pct >= t:
            signal = 1
        elif d == 'SELL on Rise' and pct >= t:
            signal = -1
        signals.append(signal)
    df['Signal'] = signals
    df['PctChange'] = ((df['Close'] - first_price) / first_price) * 100
    return df

def generate_ai_analysis_signal(df):
    df['EMA_20'] = calculate_ema(df['Close'], 20)
    df['EMA_50'] = calculate_ema(df['Close'], 50)
    df['EMA_Fast'] = calculate_ema(df['Close'], 9)
    df['EMA_Slow'] = calculate_ema(df['Close'], 21)
    df['RSI'] = calculate_rsi(df['Close'], 14)
    macd, macd_sig, _ = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macd_sig
    bb_u, _, bb_l = calculate_bollinger_bands(df['Close'])
    df['BB_Upper'] = bb_u
    df['BB_Lower'] = bb_l
    
    has_vol = 'Volume' in df.columns and df['Volume'].sum() > 0
    signals = []
    analysis_list = []
    
    for i in range(50, len(df)):
        score = 0
        analysis = {}
        
        if df['EMA_20'].iloc[i] > df['EMA_50'].iloc[i]:
            score += 2
            analysis['trend'] = 'Bullish'
        else:
            score -= 2
            analysis['trend'] = 'Bearish'
        
        if df['RSI'].iloc[i] < 30:
            score += 2
            analysis['rsi'] = 'Oversold'
        elif df['RSI'].iloc[i] > 70:
            score -= 2
            analysis['rsi'] = 'Overbought'
        else:
            analysis['rsi'] = 'Neutral'
        
        if df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]:
            score += 1
            analysis['macd'] = 'Bullish'
        else:
            score -= 1
            analysis['macd'] = 'Bearish'
        
        if df['Close'].iloc[i] < df['BB_Lower'].iloc[i]:
            score += 1
            analysis['bb'] = 'Oversold'
        elif df['Close'].iloc[i] > df['BB_Upper'].iloc[i]:
            score -= 1
            analysis['bb'] = 'Overbought'
        else:
            analysis['bb'] = 'Neutral'
        
        if has_vol:
            avg = df['Volume'].iloc[i-20:i].mean()
            analysis['volume'] = 'High' if df['Volume'].iloc[i] > avg * 1.5 else 'Normal'
            if df['Volume'].iloc[i] > avg * 1.5:
                score += 1
        else:
            analysis['volume'] = 'N/A'
        
        analysis['score'] = score
        signals.append(1 if score >= 3 else (-1 if score <= -3 else 0))
        analysis_list.append(analysis)
    
    df['Signal'] = [0] * 50 + signals
    df['AI_Analysis'] = [{}] * 50 + analysis_list
    return df

def calculate_stop_loss(df, i, entry, signal, sl_type, config):
    pts = config.get('sl_points', 10)
    if sl_type == 'Custom Points':
        return entry - pts if signal == 1 else entry + pts
    elif sl_type == 'Trailing SL (Points)':
        cp = df['Close'].iloc[i]
        return cp - pts if signal == 1 else cp + pts
    elif sl_type == 'ATR-based':
        atr = calculate_atr(df, 14).iloc[i]
        mult = config.get('atr_multiplier', 2.0)
        return entry - (atr * mult) if signal == 1 else entry + (atr * mult)
    elif sl_type == 'Break-even After 50% Target':
        return entry - pts if signal == 1 else entry + pts
    elif sl_type == 'Signal-based (reverse EMA crossover)':
        return 0
    else:
        return entry - pts if signal == 1 else entry + pts

def update_trailing_sl(cp, csl, signal, sl_type, config, pos):
    if sl_type not in ['Trailing SL (Points)', 'Trailing SL + Signal Based']:
        return csl
    pts = config.get('sl_points', 10)
    thresh = config.get('trailing_threshold', 0)
    if signal == 1:
        nsl = cp - pts
        profit = cp - pos['entry_price']
        return nsl if profit >= thresh and nsl > csl else csl
    else:
        nsl = cp + pts
        profit = pos['entry_price'] - cp
        return nsl if profit >= thresh and nsl < csl else csl

def calculate_target(df, i, entry, signal, tgt_type, config, sl_val=0):
    pts = config.get('target_points', 20)
    if tgt_type == 'Custom Points':
        return entry + pts if signal == 1 else entry - pts
    elif tgt_type in ['Trailing Target (Points)', '50% Exit at Target (Partial)']:
        return entry + pts if signal == 1 else entry - pts
    elif tgt_type == 'ATR-based':
        atr = calculate_atr(df, 14).iloc[i]
        mult = config.get('target_atr_multiplier', 3.0)
        return entry + (atr * mult) if signal == 1 else entry - (atr * mult)
    elif tgt_type == 'Risk-Reward Based':
        rr = config.get('rr_ratio', 2.0)
        dist = abs(entry - sl_val)
        return entry + (dist * rr) if signal == 1 else entry - (dist * rr)
    elif tgt_type == 'Signal-based (reverse EMA crossover)':
        return 0
    else:
        return entry + pts if signal == 1 else entry - pts

def fetch_data(symbol, interval, period, mode='Backtest'):
    try:
        if mode == 'Live Trading':
            time.sleep(random.uniform(1.0, 1.5))
        ticker = yf.Ticker(symbol)
        df = ticker.history(interval=interval, period=period)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            df.index = df.index.tz_convert('Asia/Kolkata')
        return df
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def run_backtest(df, strategy, config, qty):
    res = {'trades': [], 'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 
           'total_pnl': 0, 'accuracy': 0, 'avg_duration': 0}
    pos = None
    
    for i in range(1, len(df)):
        cp = df['Close'].iloc[i]
        
        if pos:
            exit_price = None
            exit_reason = None
            
            if config['sl_type'] == 'Signal-based (reverse EMA crossover)' or \
               config['target_type'] == 'Signal-based (reverse EMA crossover)':
                if pos['signal'] == 1:
                    if df['EMA_Fast'].iloc[i] < df['EMA_Slow'].iloc[i] and df['EMA_Fast'].iloc[i-1] >= df['EMA_Slow'].iloc[i-1]:
                        exit_reason = 'Reverse Signal'
                        exit_price = cp
                elif pos['signal'] == -1:
                    if df['EMA_Fast'].iloc[i] > df['EMA_Slow'].iloc[i] and df['EMA_Fast'].iloc[i-1] <= df['EMA_Slow'].iloc[i-1]:
                        exit_reason = 'Reverse Signal'
                        exit_price = cp
            
            if pos['signal'] == 1:
                if pos['highest_price'] is None or cp > pos['highest_price']:
                    pos['highest_price'] = cp
            else:
                if pos['lowest_price'] is None or cp < pos['lowest_price']:
                    pos['lowest_price'] = cp
            
            if config['sl_type'] in ['Trailing SL (Points)', 'Trailing SL + Signal Based']:
                pos['sl'] = update_trailing_sl(cp, pos['sl'], pos['signal'], config['sl_type'], config, pos)
            
            if config['sl_type'] == 'Break-even After 50% Target' and not pos.get('breakeven_activated'):
                if pos['signal'] == 1:
                    prof = cp - pos['entry_price']
                    tdist = pos['target'] - pos['entry_price']
                    if tdist > 0 and prof >= tdist * 0.5:
                        pos['sl'] = pos['entry_price']
                        pos['breakeven_activated'] = True
                else:
                    prof = pos['entry_price'] - cp
                    tdist = pos['entry_price'] - pos['target']
                    if tdist > 0 and prof >= tdist * 0.5:
                        pos['sl'] = pos['entry_price']
                        pos['breakeven_activated'] = True
            
            if config['sl_type'] != 'Signal-based (reverse EMA crossover)' and not exit_reason:
                if pos['signal'] == 1 and cp <= pos['sl']:
                    exit_reason = 'Stop Loss Hit'
                    exit_price = pos['sl']
                elif pos['signal'] == -1 and cp >= pos['sl']:
                    exit_reason = 'Stop Loss Hit'
                    exit_price = pos['sl']
            
            if config['target_type'] not in ['Trailing Target (Points)', 'Trailing Target + Signal Based', 
                                             'Signal-based (reverse EMA crossover)'] and not exit_reason:
                if config['target_type'] == '50% Exit at Target (Partial)':
                    if not pos.get('partial_exit_done'):
                        if pos['signal'] == 1 and cp >= pos['target']:
                            pos['partial_exit_done'] = True
                        elif pos['signal'] == -1 and cp <= pos['target']:
                            pos['partial_exit_done'] = True
                else:
                    if pos['signal'] == 1 and cp >= pos['target']:
                        exit_reason = 'Target Hit'
                        exit_price = pos['target']
                    elif pos['signal'] == -1 and cp <= pos['target']:
                        exit_reason = 'Target Hit'
                        exit_price = pos['target']
            
            if exit_reason:
                dur = (df.index[i] - pos['entry_time']).total_seconds() / 3600
                pnl = (exit_price - pos['entry_price']) * qty if pos['signal'] == 1 else (pos['entry_price'] - exit_price) * qty
                
                trade = {
                    'entry_time': pos['entry_time'], 'exit_time': df.index[i], 'duration': dur,
                    'signal': 'LONG' if pos['signal'] == 1 else 'SHORT',
                    'entry_price': pos['entry_price'], 'exit_price': exit_price,
                    'sl': pos['sl'], 'target': pos['target'], 'exit_reason': exit_reason, 'pnl': pnl,
                    'highest': pos['highest_price'], 'lowest': pos['lowest_price'],
                    'range': pos['highest_price'] - pos['lowest_price']
                }
                res['trades'].append(trade)
                res['total_trades'] += 1
                res['total_pnl'] += pnl
                if pnl > 0:
                    res['winning_trades'] += 1
                else:
                    res['losing_trades'] += 1
                pos = None
        
        if not pos and df['Signal'].iloc[i] != 0:
            sig = df['Signal'].iloc[i]
            entry = cp
            sl = calculate_stop_loss(df, i, entry, sig, config['sl_type'], config)
            tgt = calculate_target(df, i, entry, sig, config['target_type'], config, sl)
            
            min_sl = config.get('min_sl_distance', 10)
            min_tgt = config.get('min_target_distance', 15)
            
            if sig == 1:
                if abs(entry - sl) < min_sl:
                    sl = entry - min_sl
                if abs(tgt - entry) < min_tgt:
                    tgt = entry + min_tgt
            else:
                if abs(sl - entry) < min_sl:
                    sl = entry + min_sl
                if abs(entry - tgt) < min_tgt:
                    tgt = entry - min_tgt
            
            pos = {'entry_time': df.index[i], 'entry_price': entry, 'signal': sig,
                   'sl': sl, 'target': tgt, 'highest_price': entry, 'lowest_price': entry,
                   'partial_exit_done': False, 'breakeven_activated': False}
    
    if res['total_trades'] > 0:
        res['accuracy'] = (res['winning_trades'] / res['total_trades']) * 100
        res['avg_duration'] = sum(t['duration'] for t in res['trades']) / res['total_trades']
    return res

def process_live_trading(df, strategy, config, qty):
    if df is None or len(df) == 0:
        return
    
    cp = df['Close'].iloc[-1]
    pos = st.session_state.get('position')
    
    if pos:
        if pos['signal'] == 1:
            if not st.session_state['highest_price'] or cp > st.session_state['highest_price']:
                st.session_state['highest_price'] = cp
        else:
            if not st.session_state['lowest_price'] or cp < st.session_state['lowest_price']:
                st.session_state['lowest_price'] = cp
        
        pos['highest_price'] = st.session_state['highest_price']
        pos['lowest_price'] = st.session_state['lowest_price']
        
        exit_price = None
        exit_reason = None
        
        if config['sl_type'] == 'Signal-based (reverse EMA crossover)' or \
           config['target_type'] == 'Signal-based (reverse EMA crossover)':
            if pos['signal'] == 1:
                if df['EMA_Fast'].iloc[-1] < df['EMA_Slow'].iloc[-1] and df['EMA_Fast'].iloc[-2] >= df['EMA_Slow'].iloc[-2]:
                    exit_reason = 'Reverse Signal'
                    exit_price = cp
            elif pos['signal'] == -1:
                if df['EMA_Fast'].iloc[-1] > df['EMA_Slow'].iloc[-1] and df['EMA_Fast'].iloc[-2] <= df['EMA_Slow'].iloc[-2]:
                    exit_reason = 'Reverse Signal'
                    exit_price = cp
        
        if config['sl_type'] in ['Trailing SL (Points)', 'Trailing SL + Signal Based']:
            nsl = update_trailing_sl(cp, pos['sl'], pos['signal'], config['sl_type'], config, pos)
            if nsl != pos['sl']:
                log_message(f"Trailing SL: {pos['sl']:.2f} -> {nsl:.2f}")
                pos['sl'] = nsl
                st.session_state['position'] = pos
        
        if config['sl_type'] == 'Break-even After 50% Target' and not st.session_state.get('breakeven_activated'):
            if pos['signal'] == 1:
                prof = cp - pos['entry_price']
                tdist = pos['target'] - pos['entry_price']
                if tdist > 0 and prof >= tdist * 0.5:
                    pos['sl'] = pos['entry_price']
                    st.session_state['breakeven_activated'] = True
                    st.session_state['position'] = pos
                    log_message("Break-even activated")
            else:
                prof = pos['entry_price'] - cp
                tdist = pos['entry_price'] - pos['target']
                if tdist > 0 and prof >= tdist * 0.5:
                    pos['sl'] = pos['entry_price']
                    st.session_state['breakeven_activated'] = True
                    st.session_state['position'] = pos
                    log_message("Break-even activated")
        
        if config['sl_type'] != 'Signal-based (reverse EMA crossover)' and not exit_reason:
            if pos['signal'] == 1 and cp <= pos['sl']:
                exit_reason = 'Stop Loss Hit'
                exit_price = pos['sl']
            elif pos['signal'] == -1 and cp >= pos['sl']:
                exit_reason = 'Stop Loss Hit'
                exit_price = pos['sl']
        
        if config['target_type'] not in ['Trailing Target (Points)', 'Trailing Target + Signal Based', 
                                         'Signal-based (reverse EMA crossover)'] and not exit_reason:
            if config['target_type'] == '50% Exit at Target (Partial)':
                if not st.session_state.get('partial_exit_done'):
                    if pos['signal'] == 1 and cp >= pos['target']:
                        st.session_state['partial_exit_done'] = True
                        log_message("50% exited")
                    elif pos['signal'] == -1 and cp <= pos['target']:
                        st.session_state['partial_exit_done'] = True
                        log_message("50% exited")
            else:
                if pos['signal'] == 1 and cp >= pos['target']:
                    exit_reason = 'Target Hit'
                    exit_price = pos['target']
                elif pos['signal'] == -1 and cp <= pos['target']:
                    exit_reason = 'Target Hit'
                    exit_price = pos['target']
        
        if exit_reason:
            et = datetime.now(pytz.timezone('Asia/Kolkata'))
            dur = (et - pos['entry_time']).total_seconds() / 3600
            pnl = (exit_price - pos['entry_price']) * qty if pos['signal'] == 1 else (pos['entry_price'] - exit_price) * qty
            
            trade = {
                'entry_time': pos['entry_time'], 'exit_time': et, 'duration': dur,
                'signal': 'LONG' if pos['signal'] == 1 else 'SHORT',
                'entry_price': pos['entry_price'], 'exit_price': exit_price,
                'sl': pos['sl'], 'target': pos['target'], 'exit_reason': exit_reason, 'pnl': pnl,
                'highest': st.session_state['highest_price'], 'lowest': st.session_state['lowest_price'],
                'range': st.session_state['highest_price'] - st.session_state['lowest_price']
            }
            st.session_state['trade_history'].append(trade)
            log_message(f"CLOSED: {exit_reason} | PnL: {pnl:.2f}")
            reset_position_state()
    
    if not st.session_state['position'] and df['Signal'].iloc[-1] != 0:
        sig = df['Signal'].iloc[-1]
        entry = cp
        sl = calculate_stop_loss(df, len(df)-1, entry, sig, config['sl_type'], config)
        tgt = calculate_target(df, len(df)-1, entry, sig, config['target_type'], config, sl)
        
        min_sl = config.get('min_sl_distance', 10)
        min_tgt = config.get('min_target_distance', 15)
        
        if sig == 1:
            if abs(entry - sl) < min_sl:
                sl = entry - min_sl
            if abs(tgt - entry) < min_tgt:
                tgt = entry + min_tgt
        else:
            if abs(sl - entry) < min_sl:
                sl = entry + min_sl
            if abs(entry - tgt) < min_tgt:
                tgt = entry - min_tgt
        
        pos = {'entry_time': datetime.now(pytz.timezone('Asia/Kolkata')), 'entry_price': entry,
               'signal': sig, 'sl': sl, 'target': tgt, 'highest_price': entry, 'lowest_price': entry}
        
        st.session_state['position'] = pos
        st.session_state['highest_price'] = entry
        st.session_state['lowest_price'] = entry
        
        stype = 'LONG' if sig == 1 else 'SHORT'
        log_message(f"OPENED: {stype} at {entry:.2f} | SL: {sl:.2f} | Target: {tgt:.2f}")

def main():
    st.set_page_config(page_title="Quant Trading System", layout="wide")
    st.title("🚀 Professional Quantitative Trading System")
    
    for key in ['trading_active', 'current_data', 'position', 'trade_history', 'trade_logs',
                'highest_price', 'lowest_price', 'partial_exit_done', 'breakeven_activated']:
        if key not in st.session_state:
            st.session_state[key] = False if key == 'trading_active' else ([] if 'history' in key or 'logs' in key else None)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        asset_type = st.selectbox("Asset Type", ["Indian Indices", "Crypto", "Forex", "Commodities", "Custom"])
        
        if asset_type == "Indian Indices":
            symbol = st.selectbox("Symbol", ["^NSEI", "^NSEBANK", "^BSESN"])
        elif asset_type == "Crypto":
            symbol = st.selectbox("Symbol", ["BTC-USD", "ETH-USD"])
        elif asset_type == "Forex":
            symbol = st.selectbox("Symbol", ["USDINR=X", "EURUSD=X", "GBPUSD=X"])
        elif asset_type == "Commodities":
            symbol = st.selectbox("Symbol", ["GC=F", "SI=F"])
        else:
            symbol = st.text_input("Custom Ticker", value="AAPL")
        
        interval = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"])
        
        period_opts = {
            "1m": ["1d", "5d"], "5m": ["1d", "1mo"], "15m": ["1mo"], "30m": ["1mo"],
            "1h": ["1mo"], "4h": ["1mo"], "1d": ["1mo", "1y", "2y", "5y"],
            "1wk": ["1mo", "1y", "5y", "10y", "15y", "20y"],
            "1mo": ["1y", "2y", "5y", "10y", "15y", "20y", "25y", "30y"]
        }
        
        period = st.selectbox("Period", period_opts.get(interval, ["1mo"]))
        qty = st.number_input("Quantity", min_value=1, value=1, step=1)
        mode = st.selectbox("Mode", ["Backtest", "Live Trading"])
        
        st.markdown("---")
        
        strategy = st.selectbox("Strategy", [
            "EMA Crossover", "Simple Buy", "Simple Sell", "Price Crosses Threshold",
            "RSI-ADX-EMA", "Percentage Change", "AI Price Action Analysis"
        ])
        
        config = {}
        
        if strategy == "EMA Crossover":
            st.subheader("EMA Parameters")
            config['ema_fast'] = st.number_input("EMA Fast", min_value=1, value=9, step=1)
            config['ema_slow'] = st.number_input("EMA Slow", min_value=1, value=15, step=1)
            config['min_angle'] = st.number_input("Min Angle", min_value=0.0, value=1.0, step=0.1)
            
            config['entry_filter'] = st.selectbox("Entry Filter", 
                ["Simple Crossover", "Custom Candle (Points)", "ATR-based Candle"])
            
            if config['entry_filter'] == "Custom Candle (Points)":
                config['custom_points'] = st.number_input("Custom Points", min_value=1.0, value=10.0, step=1.0)
            elif config['entry_filter'] == "ATR-based Candle":
                config['atr_multiplier'] = st.number_input("ATR Multiplier", min_value=0.1, value=1.5, step=0.1)
            
            config['use_adx'] = st.checkbox("Use ADX Filter")
            if config['use_adx']:
                config['adx_period'] = st.number_input("ADX Period", min_value=1, value=14, step=1)
                config['adx_threshold'] = st.number_input("ADX Threshold", min_value=1.0, value=25.0, step=1.0)
        
        elif strategy == "Price Crosses Threshold":
            config['threshold'] = st.number_input("Threshold", min_value=0.0, value=100.0, step=1.0)
            config['direction'] = st.selectbox("Direction", [
                "LONG (Price >= Threshold)", "SHORT (Price >= Threshold)",
                "LONG (Price <= Threshold)", "SHORT (Price <= Threshold)"
            ])
        
        elif strategy == "RSI-ADX-EMA":
            config['rsi_period'] = st.number_input("RSI Period", min_value=1, value=14, step=1)
            config['adx_period'] = st.number_input("ADX Period", min_value=1, value=14, step=1)
            config['ema1_period'] = st.number_input("EMA1 Period", min_value=1, value=9, step=1)
            config['ema2_period'] = st.number_input("EMA2 Period", min_value=1, value=21, step=1)
        
        elif strategy == "Percentage Change":
            config['percentage_threshold'] = st.number_input("% Threshold", 
                min_value=0.001, value=0.01, step=0.001, format="%.3f")
            config['direction'] = st.selectbox("Direction", 
                ["BUY on Fall", "SELL on Fall", "BUY on Rise", "SELL on Rise"])
        
        st.markdown("---")
        st.subheader("Stop Loss")
        config['sl_type'] = st.selectbox("SL Type", [
            "Custom Points", "Trailing SL (Points)", "Trailing SL + Signal Based",
            "Break-even After 50% Target", "ATR-based", "Signal-based (reverse EMA crossover)"
        ])
        
        if config['sl_type'] != "Signal-based (reverse EMA crossover)":
            config['sl_points'] = st.number_input("SL Points", min_value=1.0, value=10.0, step=1.0)
        else:
            config['sl_points'] = 0
        
        if 'Trailing' in config['sl_type']:
            config['trailing_threshold'] = st.number_input("Trailing Threshold", min_value=0.0, value=0.0, step=1.0)
        
        if 'ATR' in config['sl_type']:
            config['atr_multiplier'] = st.number_input("ATR Multiplier (SL)", min_value=0.1, value=2.0, step=0.1)
        
        config['min_sl_distance'] = st.number_input("Min SL Distance", min_value=0.0, value=10.0, step=1.0)
        
        st.markdown("---")
        st.subheader("Target")
        config['target_type'] = st.selectbox("Target Type", [
            "Custom Points", "Trailing Target (Points)", "50% Exit at Target (Partial)",
            "ATR-based", "Risk-Reward Based", "Signal-based (reverse EMA crossover)"
        ])
        
        if config['target_type'] != "Signal-based (reverse EMA crossover)":
            config['target_points'] = st.number_input("Target Points", min_value=1.0, value=20.0, step=1.0)
        else:
            config['target_points'] = 0
        
        if config['target_type'] == 'ATR-based':
            config['target_atr_multiplier'] = st.number_input("ATR Multiplier (Target)", min_value=0.1, value=3.0, step=0.1)
        
        if config['target_type'] == 'Risk-Reward Based':
            config['rr_ratio'] = st.number_input("RR Ratio", min_value=0.1, value=2.0, step=0.1)
        
        config['min_target_distance'] = st.number_input("Min Target Distance", min_value=0.0, value=15.0, step=1.0)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Dashboard", "📈 History", "📝 Logs", "🔬 Backtest"])
    
    with tab1:
        if mode == "Live Trading":
            st.markdown("### 🎛️ Controls")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("▶️ Start", type="primary", use_container_width=True):
                    st.session_state['trading_active'] = True
                    log_message("Trading started")
            
            with col2:
                if st.button("⏸️ Stop", use_container_width=True):
                    if st.session_state['trading_active']:
                        st.session_state['trading_active'] = False
                        if st.session_state['position'] and st.session_state['current_data'] is not None:
                            pos = st.session_state['position']
                            cp = st.session_state['current_data']['Close'].iloc[-1]
                            et = datetime.now(pytz.timezone('Asia/Kolkata'))
                            dur = (et - pos['entry_time']).total_seconds() / 3600
                            pnl = (cp - pos['entry_price']) * qty if pos['signal'] == 1 else (pos['entry_price'] - cp) * qty
                            
                            trade = {
                                'entry_time': pos['entry_time'], 'exit_time': et, 'duration': dur,
                                'signal': 'LONG' if pos['signal'] == 1 else 'SHORT',
                                'entry_price': pos['entry_price'], 'exit_price': cp,
                                'sl': pos['sl'], 'target': pos['target'], 'exit_reason': 'Manual Close', 'pnl': pnl,
                                'highest': st.session_state.get('highest_price', pos['entry_price']),
                                'lowest': st.session_state.get('lowest_price', pos['entry_price']),
                                'range': st.session_state.get('highest_price', pos['entry_price']) - 
                                        st.session_state.get('lowest_price', pos['entry_price'])
                            }
                            st.session_state['trade_history'].append(trade)
                            log_message(f"Manual close | PnL: {pnl:.2f}")
                        reset_position_state()
                        log_message("Stopped")
            
            with col3:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()
            
            if st.session_state['trading_active']:
                st.success("🟢 ACTIVE")
            else:
                st.info("⚪ STOPPED")
            
            st.markdown("---")
            
            if st.session_state['trading_active']:
                pb = st.progress(0)
                st_txt = st.empty()
                
                while st.session_state['trading_active']:
                    st_txt.text("Fetching...")
                    pb.progress(30)
                    
                    df = fetch_data(symbol, interval, period, mode)
                    
                    if df is not None:
                        st.session_state['current_data'] = df
                        pb.progress(60)
                        st_txt.text("Signals...")
                        
                        if strategy == "EMA Crossover":
                            df = generate_ema_crossover_signal(df, config)
                        elif strategy == "Simple Buy":
                            df = generate_simple_buy_signal(df)
                        elif strategy == "Simple Sell":
                            df = generate_simple_sell_signal(df)
                        elif strategy == "Price Crosses Threshold":
                            df = generate_price_threshold_signal(df, config)
                        elif strategy == "RSI-ADX-EMA":
                            df = generate_rsi_adx_ema_signal(df, config)
                        elif strategy == "Percentage Change":
                            df = generate_percentage_change_signal(df, config)
                        elif strategy == "AI Price Action Analysis":
                            df = generate_ai_analysis_signal(df)
                        
                        pb.progress(80)
                        st_txt.text("Processing...")
                        process_live_trading(df, strategy, config, qty)
                        pb.progress(100)
                        st_txt.text("Updated")
                        st.session_state['current_data'] = df
                    else:
                        st.error("Failed to fetch")
                    
                    time.sleep(random.uniform(1.0, 1.5))
                    st.rerun()
            
            if st.session_state['current_data'] is not None:
                df = st.session_state['current_data']
                cp = df['Close'].iloc[-1]
                pos = st.session_state['position']
                
                st.markdown("### 📈 Metrics")
                m1, m2, m3, m4 = st.columns(4)
                
                with m1:
                    st.metric("Price", f"{cp:.2f}")
                with m2:
                    st.metric("Entry", f"{pos['entry_price']:.2f}" if pos else "N/A")
                with m3:
                    st.metric("Position", ("LONG" if pos['signal'] == 1 else "SHORT") if pos else "None")
                with m4:
                    if pos:
                        upnl = (cp - pos['entry_price']) * qty if pos['signal'] == 1 else (pos['entry_price'] - cp) * qty
                        if upnl >= 0:
                            st.metric("P&L", f"{upnl:.2f}", delta=f"+{upnl:.2f}")
                        else:
                            st.metric("P&L", f"{upnl:.2f}", delta=f"{upnl:.2f}", delta_color="inverse")
                    else:
                        st.metric("P&L", "0.00")
                
                sig = df['Signal'].iloc[-1]
                if sig == 1:
                    st.success("🟢 BUY")
                elif sig == -1:
                    st.error("🔴 SELL")
                else:
                    st.info("⚪ NONE")
                
                if pos:
                    st.markdown("### 💼 Position")
                    p1, p2, p3 = st.columns(3)
                    with p1:
                        st.write(f"**Entry:** {pos['entry_time'].strftime('%H:%M:%S')}")
                        st.write(f"**Price:** {pos['entry_price']:.2f}")
                    with p2:
                        st.write(f"**SL:** {pos['sl']:.2f if pos['sl'] != 0 else 'Signal'}")
                        st.write(f"**Target:** {pos['target']:.2f if pos['target'] != 0 else 'Signal'}")
                    with p3:
                        if st.session_state['highest_price']:
                            st.write(f"**High:** {st.session_state['highest_price']:.2f}")
                        if st.session_state['lowest_price']:
                            st.write(f"**Low:** {st.session_state['lowest_price']:.2f}")
                
                st.markdown("### 📊 Chart")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                            low=df['Low'], close=df['Close'], name='Price'))
                
                if 'EMA_Fast' in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], mode='lines',
                                            name='Fast', line=dict(color='blue', width=1)))
                if 'EMA_Slow' in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], mode='lines',
                                            name='Slow', line=dict(color='red', width=1)))
                
                if pos:
                    fig.add_hline(y=pos['entry_price'], line_dash="dash", line_color="yellow", annotation_text="Entry")
                    if pos['sl'] != 0:
                        fig.add_hline(y=pos['sl'], line_dash="dash", line_color="red", annotation_text="SL")
                    if pos['target'] != 0:
                        fig.add_hline(y=pos['target'], line_dash="dash", line_color="green", annotation_text="Tgt")
                
                fig.update_layout(title=f"{symbol} - {interval}", xaxis_title="Time",
                                 yaxis_title="Price", height=600, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True, key=f"c_{int(time.time())}")
        else:
            st.info("Live Dashboard only in Live Trading mode")
    
    with tab2:
        st.markdown("### 📈 Trade History")
        
        if len(st.session_state['trade_history']) == 0:
            st.info("No trades yet")
        else:
            tt = len(st.session_state['trade_history'])
            wt = sum(1 for t in st.session_state['trade_history'] if t['pnl'] > 0)
            lt = sum(1 for t in st.session_state['trade_history'] if t['pnl'] <= 0)
            tp = sum(t['pnl'] for t in st.session_state['trade_history'])
            acc = (wt / tt * 100) if tt > 0 else 0
            
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Total", tt)
            with m2:
                st.metric("Wins", wt)
            with m3:
                st.metric("Losses", lt)
            with m4:
                st.metric("Accuracy", f"{acc:.2f}%")
            with m5:
                if tp >= 0:
                    st.metric("P&L", f"{tp:.2f}", delta=f"+{tp:.2f}")
                else:
                    st.metric("P&L", f"{tp:.2f}", delta=f"{tp:.2f}", delta_color="inverse")
            
            st.markdown("---")
            
            for idx, t in enumerate(reversed(st.session_state['trade_history'])):
                with st.expander(f"#{tt - idx} - {t['signal']} - {t['pnl']:.2f}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"**Entry:** {t['entry_time'].strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"**Exit:** {t['exit_time'].strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"**Duration:** {t['duration']:.2f}h")
                        st.write(f"**Entry Price:** {t['entry_price']:.2f}")
                    with c2:
                        st.write(f"**Exit Price:** {t['exit_price']:.2f}")
                        sl_display = f"{t['sl']:.2f}" if t['sl'] != 0 else "Signal"
                        st.write(f"**SL:** {sl_display}")
                        tgt_display = f"{t['target']:.2f}" if t['target'] != 0 else "Signal"
                        st.write(f"**Target:** {tgt_display}")
                        st.write(f"**Reason:** {t['exit_reason']}")
                    if t['pnl'] >= 0:
                        st.success(f"**P&L:** +{t['pnl']:.2f}")
                    else:
                        st.error(f"**P&L:** {t['pnl']:.2f}")
    
    with tab3:
        st.markdown("### 📝 Logs")
        if len(st.session_state['trade_logs']) == 0:
            st.info("No logs")
        else:
            for log in reversed(st.session_state['trade_logs']):
                st.text(log)
    
    with tab4:
        if mode == "Backtest":
            st.markdown("### 🔬 Backtest")
            
            if st.button("▶️ Run", type="primary"):
                with st.spinner("Running..."):
                    if 'backtest_results' in st.session_state:
                        del st.session_state['backtest_results']
                    
                    df = fetch_data(symbol, interval, period, 'Backtest')
                    
                    if df is not None:
                        if strategy == "EMA Crossover":
                            df = generate_ema_crossover_signal(df, config)
                        elif strategy == "Simple Buy":
                            df = generate_simple_buy_signal(df)
                        elif strategy == "Simple Sell":
                            df = generate_simple_sell_signal(df)
                        elif strategy == "Price Crosses Threshold":
                            df = generate_price_threshold_signal(df, config)
                        elif strategy == "RSI-ADX-EMA":
                            df = generate_rsi_adx_ema_signal(df, config)
                        elif strategy == "Percentage Change":
                            df = generate_percentage_change_signal(df, config)
                        elif strategy == "AI Price Action Analysis":
                            df = generate_ai_analysis_signal(df)
                        
                        res = run_backtest(df, strategy, config, qty)
                        st.session_state['backtest_results'] = res
                        st.success("✅ Done!")
                    else:
                        st.error("Failed")
            
            if 'backtest_results' in st.session_state:
                res = st.session_state['backtest_results']
                
                st.markdown("### 📊 Results")
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Total", res['total_trades'])
                with m2:
                    st.metric("Wins", res['winning_trades'])
                with m3:
                    st.metric("Losses", res['losing_trades'])
                with m4:
                    st.metric("Accuracy", f"{res['accuracy']:.2f}%")
                with m5:
                    if res['total_pnl'] >= 0:
                        st.metric("P&L", f"{res['total_pnl']:.2f}", delta=f"+{res['total_pnl']:.2f}")
                    else:
                        st.metric("P&L", f"{res['total_pnl']:.2f}", delta=f"{res['total_pnl']:.2f}", delta_color="inverse")
                
                st.metric("Avg Duration", f"{res['avg_duration']:.2f}h")
                
                st.markdown("---")
                st.markdown("### 📋 Trades")
                
                if len(res['trades']) == 0:
                    st.info("No trades")
                else:
                    for idx, t in enumerate(res['trades']):
                        with st.expander(f"#{idx + 1} - {t['signal']} - {t['pnl']:.2f}"):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.write(f"**Entry:** {t['entry_time'].strftime('%Y-%m-%d %H:%M')}")
                                st.write(f"**Exit:** {t['exit_time'].strftime('%Y-%m-%d %H:%M')}")
                                st.write(f"**Duration:** {t['duration']:.2f}h")
                                st.write(f"**Entry Price:** {t['entry_price']:.2f}")
                    with c2:
                        st.write(f"**Exit Price:** {t['exit_price']:.2f}")
                        sl_str = f"{t['sl']:.2f}" if t['sl'] != 0 else "Signal"
                        st.write(f"**SL:** {sl_str}")
                        tgt_str = f"{t['target']:.2f}" if t['target'] != 0 else "Signal"
                        st.write(f"**Target:** {tgt_str}")
                        st.write(f"**Reason:** {t['exit_reason']}")
                        if t['pnl'] >= 0:
                            st.success(f"**P&L:** +{t['pnl']:.2f}")
                        else:
                            st.error(f"**P&L:** {t['pnl']:.2f}")
                            st.info(f"H: {t.get('highest', 0):.2f} | L: {t.get('lowest', 0):.2f} | R: {t.get('range', 0):.2f}")
        else:
            st.info("Backtest only in Backtest mode")

if __name__ == "__main__":
    main()
