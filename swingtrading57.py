import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
import pytz

# --- INITIALIZATION FIX ---
if 'trading_active' not in st.session_state:
    st.session_state.update({
        'trading_active': False,
        'current_position': None,
        'trade_history': [],
        'trade_log': [],
        'iteration_count': 0,
        'last_api_call': 0
    })

IST = pytz.timezone('Asia/Kolkata')

# --- REFINED TRADING LOGIC ---
def run_trading_engine(df, strat_obj, sl_type, tp_type):
    # Calculate indicators
    df_processed = strat_obj.calculate_indicators(df)
    # Get signals: returns (bullish, bearish, metadata)
    bull, bear, sig_data = strat_obj.generate_signal(df_processed)
    
    curr_price = df_processed['Close'].iloc[-1]
    curr_time = df_processed.index[-1]

    if st.session_state.current_position is None:
        # ENTRY LOGIC
        if bull:
            entry_data = {'type': 'LONG', 'entry': curr_price, 'time': curr_time}
            st.session_state.current_position = entry_data
            add_log(f"🚀 LONG Entry @ {curr_price:.2f}")
        elif bear:
            entry_data = {'type': 'SHORT', 'entry': curr_price, 'time': curr_time}
            st.session_state.current_position = entry_data
            add_log(f"🔻 SHORT Entry @ {curr_price:.2f}")
            
    else:
        # EXIT LOGIC
        pos = st.session_state.current_position
        # FIX: Only allow signal exit if the signal is from a NEW candle
        is_new_candle = curr_time > pos['time']
        
        exit_triggered = False
        reason = ""

        # 1. Signal Based Exit Logic
        if sl_type == "Signal Based" or tp_type == "Signal Based":
            if is_new_candle:
                if pos['type'] == 'LONG' and bear:
                    exit_triggered, reason = True, "Opposite Signal (Bearish)"
                elif pos['type'] == 'SHORT' and bull:
                    exit_triggered, reason = True, "Opposite Signal (Bullish)"

        # 2. Standard SL/TP (Example for Custom Points)
        pnl = curr_price - pos['entry'] if pos['type'] == 'LONG' else pos['entry'] - curr_price
        
        # [Additional SL/TP logic here...]

        if exit_triggered:
            trade_record = {
                'entry': pos['entry'],
                'exit': curr_price,
                'pnl': pnl,
                'reason': reason,
                'type': pos['type']
            }
            st.session_state.trade_history.append(trade_record)
            st.session_state.current_position = None
            add_log(f"🛑 EXIT @ {curr_price:.2f} | Reason: {reason}")

    return df_processed, sig_data
