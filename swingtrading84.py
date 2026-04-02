import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import time
from datetime import datetime
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Constants & Setup ---
st.set_page_config(page_title="Smart Wealth", layout="wide")
IST = pytz.timezone('Asia/Kolkata')

TICKERS = {
    "NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN",
    "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD", "GOLD": "GC=F", "SILVER": "SI=F", "Custom": ""
}

TIMEFRAME_PERIODS = {
    "1m": ["1d", "5d", "7d"],
    "5m": ["1d", "5d", "7d", "1mo"],
    "15m": ["1d", "5d", "7d", "1mo"],
    "1h": ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y"],
    "1d": ["5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
    "1wk": ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"]
}

# --- State Management ---
if 'live_running' not in st.session_state: st.session_state.live_running = False
if 'trade_history' not in st.session_state: st.session_state.trade_history = []
if 'current_position' not in st.session_state: st.session_state.current_position = None
if 'last_signal_time' not in st.session_state: st.session_state.last_signal_time = None

# --- Mock Dhan Class ---
class MockDhan:
    NSE, BSE, INTRADAY, DELIVERY, MARKET, LIMIT, BUY, SELL = "NSE", "BSE", "INTRADAY", "DELIVERY", "MARKET", "LIMIT", "BUY", "SELL"
    def place_order(self, **kwargs): return {"status": "success", "order_id": "MOCK_" + str(int(time.time()))}

dhan = MockDhan()

def place_order(algo_signal, is_options, config, ltp):
    if not config['enable_dhan']: return "Paper Trade"
    return {"status": "Order Placed"}

# --- Utility Functions ---
def fetch_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        df.index = df.index.tz_convert(IST) if df.index.tzinfo else df.index.tz_localize('UTC').tz_convert(IST)
        return df
    except: return None

def apply_indicators(df, fast_ema, slow_ema):
    if df is not None and not df.empty:
        df['EMA_Fast'] = df['Close'].ewm(span=fast_ema, adjust=False).mean()
        df['EMA_Slow'] = df['Close'].ewm(span=slow_ema, adjust=False).mean()
    return df

def get_ltp_stats(ticker):
    try:
        tkr = yf.Ticker(ticker)
        data = tkr.history(period='2d')
        if len(data) >= 2:
            prev_close = float(data['Close'].iloc[-2])
            ltp = float(data['Close'].iloc[-1])
            diff = ltp - prev_close
            return ltp, diff, (diff / prev_close) * 100
    except: pass
    return None, None, None

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ Smart Wealth Config")
selected_asset = st.sidebar.selectbox("Select Asset", list(TICKERS.keys()))
current_ticker = st.sidebar.text_input("Custom Ticker") if selected_asset == "Custom" else TICKERS[selected_asset]
tf = st.sidebar.selectbox("Timeframe", list(TIMEFRAME_PERIODS.keys()))
period = st.sidebar.selectbox("Period", TIMEFRAME_PERIODS[tf])

strategy = st.sidebar.selectbox("Strategy", ["EMA Crossover", "Simple Buy", "Simple Sell"])
fast_ema = st.sidebar.number_input("Fast EMA", value=9)
slow_ema = st.sidebar.number_input("Slow EMA", value=15)
sl_val = st.sidebar.number_input("SL Points", value=10.0)
tgt_val = st.sidebar.number_input("Target Points", value=20.0)

cooldown_enabled = st.sidebar.checkbox("Enable Cooldown", value=True)
cooldown_sec = st.sidebar.number_input("Cooldown (s)", value=5)
no_overlap = st.sidebar.checkbox("Prevent Overlap", value=True)
enable_dhan = st.sidebar.checkbox("Enable Dhan", value=False)
opt_enabled = st.sidebar.checkbox("Options Trading", value=False)

config = {'enable_dhan': enable_dhan, 'opt_enabled': opt_enabled}

# --- Header ---
st.title("📈 Smart Wealth")
ltp_val, diff_val, pct_val = get_ltp_stats(current_ticker)
if ltp_val:
    color = "green" if diff_val >= 0 else "red"
    st.markdown(f"**{selected_asset} LTP:** :blue[{ltp_val:.2f}] | <span style='color:{color}'>{'▲' if diff_val >= 0 else '▼'} {abs(diff_val):.2f} ({pct_val:.2f}%)</span>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🧪 Backtesting", "⚡ Live Trading", "📜 Trade History"])

# --- Tab 1: Backtesting ---
with tab1:
    if st.button("Run Backtest", key="btn_run_bt"):
        df = fetch_data(current_ticker, period, tf)
        if df is not None:
            df = apply_indicators(df, fast_ema, slow_ema)
            trades, in_trade, violation_count, last_exit_time = [], False, 0, None
            
            for i in range(1, len(df)):
                row, prev_row = df.iloc[i], df.iloc[i-1]
                c_close, c_high, c_low = float(row['Close']), float(row['High']), float(row['Low'])
                p_f, p_s, c_f, c_s = float(prev_row['EMA_Fast']), float(prev_row['EMA_Slow']), float(row['EMA_Fast']), float(row['EMA_Slow'])
                
                if in_trade:
                    h_sl = (c_low <= sl if t_type == 'Buy' else c_high >= sl)
                    h_tg = (c_high >= tg if t_type == 'Buy' else c_low <= tg)
                    if h_sl and h_tg: violation_count += 1; h_tg = False
                    if h_sl or h_tg:
                        ex_p = sl if h_sl else tg
                        pnl = (ex_p - en_p) if t_type == 'Buy' else (en_p - ex_p)
                        trades.append({
                            'Entry Time': en_time, 
                            'Exit Time': df.index[i],
                            'Type': t_type, 'Entry': en_p, 'Exit': ex_p, 'PnL': pnl, 'Reason': 'SL' if h_sl else 'Tgt'
                        })
                        in_trade, last_exit_time = False, df.index[i]
                else:
                    if cooldown_enabled and last_exit_time and (df.index[i] - last_exit_time).total_seconds() < cooldown_sec: continue
                    
                    buy_sig = (p_f < p_s and c_f > c_s) or strategy == "Simple Buy"
                    sell_sig = (p_f > p_s and c_f < c_s) or strategy == "Simple Sell"

                    if buy_sig:
                        in_trade, t_type, en_p, en_time = True, 'Buy', c_close, df.index[i]
                        sl, tg = en_p - sl_val, en_p + tgt_val
                    elif sell_sig:
                        in_trade, t_type, en_p, en_time = True, 'Sell', c_close, df.index[i]
                        sl, tg = en_p + sl_val, en_p - tgt_val

            if trades:
                tdf = pd.DataFrame(trades)
                won = tdf[tdf['PnL'] > 0]
                lost = tdf[tdf['PnL'] <= 0]
                
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Trades", len(tdf))
                m2.metric("Won / Lost", f"{len(won)} / {len(lost)}")
                m3.metric("Accuracy", f"{(len(won)/len(tdf))*100:.1f}%")
                m4.metric("Total PnL", f"{tdf['PnL'].sum():.2f}")
                m5.metric("Avg PnL/Trade", f"{(tdf['PnL'].mean()):.2f}")

                st.dataframe(tdf, use_container_width=True)
                
                fig_bt = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
                fig_bt.update_layout(title="Backtest Price Action", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig_bt, use_container_width=True, key="backtest_static_chart")
            else: st.info("No trades matched strategy criteria.")

# --- Tab 2: Live Trading ---
with tab2:
    l_c1, l_c2 = st.columns(2)
    if l_c1.button("▶ START LIVE", use_container_width=True, key="live_start"):
        st.session_state.live_running = True
        # IMMEDIATE ENTRY FOR SIMPLE STRATEGIES
        if not st.session_state.current_position:
            if strategy == "Simple Buy" or strategy == "Simple Sell":
                temp_df = fetch_data(current_ticker, '1d', tf)
                if temp_df is not None:
                    current_ltp = float(temp_df['Close'].iloc[-1])
                    t_type = "Buy" if strategy == "Simple Buy" else "Sell"
                    sl = current_ltp - sl_val if t_type == "Buy" else current_ltp + sl_val
                    tg = current_ltp + tgt_val if t_type == "Buy" else current_ltp - tgt_val
                    
                    st.session_state.current_position = {
                        'type': t_type, 'entry_price': current_ltp,
                        'sl': sl, 'tgt': tg, 'entry_time': datetime.now(IST)
                    }
                    place_order(t_type, config['opt_enabled'], config, current_ltp)
                    st.toast(f"Immediate {t_type} Entry Executed!")

    if l_c2.button("🛑 STOP LIVE", use_container_width=True, key="live_stop"): 
        st.session_state.live_running = False
    
    live_stat_container = st.empty()
    live_chart_container = st.empty()

    if st.session_state.live_running:
        while st.session_state.live_running:
            df_live = fetch_data(current_ticker, '1d', tf)
            if df_live is not None:
                df_live = apply_indicators(df_live, fast_ema, slow_ema)
                last_candle = df_live.iloc[-1]
                ltp = float(last_candle['Close'])
                f_val, s_val = float(last_candle['EMA_Fast']), float(last_candle['EMA_Slow'])
                
                # Check Exit Logic
                if st.session_state.current_position:
                    pos = st.session_state.current_position
                    live_pnl = (ltp - pos['entry_price']) if pos['type'] == 'Buy' else (pos['entry_price'] - ltp)
                    
                    exit_flag, reason = False, ""
                    if pos['type'] == 'Buy':
                        if ltp <= pos['sl']: exit_flag, reason = True, "SL"
                        elif ltp >= pos['tgt']: exit_flag, reason = True, "Target"
                    else:
                        if ltp >= pos['sl']: exit_flag, reason = True, "SL"
                        elif ltp <= pos['tgt']: exit_flag, reason = True, "Target"
                    
                    if exit_flag:
                        st.session_state.trade_history.append({
                            'Entry Time': pos['entry_time'], 'Exit Time': datetime.now(IST),
                            'Type': pos['type'], 'Entry': pos['entry_price'], 'Exit': ltp, 'PnL': live_pnl, 'Reason': reason
                        })
                        st.session_state.current_position = None
                        st.toast(f"Position Closed: {reason}")
                    
                    live_stat_container.warning(f"ACTIVE TRADE: {pos['type']} @ {pos['entry_price']:.2f} | LIVE PnL: {live_pnl:.2f}")
                else:
                    live_stat_container.info(f"SCANNING: {current_ticker} | LTP: {ltp:.2f} | EMA({fast_ema}): {f_val:.2f}")

                # Live Chart Generation
                view_df = df_live.iloc[-50:]
                fig_l = go.Figure()
                fig_l.add_trace(go.Scatter(x=view_df.index, y=view_df['Close'], name="LTP", line=dict(color='white')))
                fig_l.add_trace(go.Scatter(x=view_df.index, y=view_df['EMA_Fast'], name=f"EMA {fast_ema}", line=dict(color='cyan')))
                fig_l.add_trace(go.Scatter(x=view_df.index, y=view_df['EMA_Slow'], name=f"EMA {slow_ema}", line=dict(color='magenta')))
                
                # Plot active trade levels
                if st.session_state.current_position:
                    cp = st.session_state.current_position
                    fig_l.add_hline(y=cp['entry_price'], line_dash="dot", line_color="yellow", annotation_text="ENTRY")
                    fig_l.add_hline(y=cp['sl'], line_dash="dash", line_color="red", annotation_text="SL")
                    fig_l.add_hline(y=cp['tgt'], line_dash="dash", line_color="green", annotation_text="TGT")

                fig_l.add_annotation(x=view_df.index[-1], y=f_val, text=f"EMA {fast_ema}: {f_val:.2f}", showarrow=True, font=dict(color="cyan"))
                fig_l.add_annotation(x=view_df.index[-1], y=s_val, text=f"EMA {slow_ema}: {s_val:.2f}", showarrow=True, font=dict(color="magenta"))
                
                fig_l.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600, uirevision='constant')
                live_chart_container.plotly_chart(fig_l, use_container_width=True, key=f"live_chart_{time.time()}")
                
            time.sleep(1.5)

# --- Tab 3: History ---
with tab3:
    st.subheader("Executed Live Trades")
    if st.session_state.trade_history: 
        st.dataframe(pd.DataFrame(st.session_state.trade_history), use_container_width=True)
    else: 
        st.info("No live trades recorded in this session.")
