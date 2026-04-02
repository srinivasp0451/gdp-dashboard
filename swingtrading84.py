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

# --- Broker API Mock Setup (Dhan) ---
class MockDhan:
    NSE, BSE, INTRADAY, DELIVERY, MARKET, LIMIT, BUY, SELL = "NSE", "BSE", "INTRADAY", "DELIVERY", "MARKET", "LIMIT", "BUY", "SELL"
    def place_order(self, **kwargs): return {"status": "success", "order_id": "MOCK_" + str(int(time.time()))}

dhan = MockDhan()

def place_order(algo_signal, is_options, config, ltp):
    try:
        if not config['enable_dhan']: return "Paper Trade Success"
        transaction_type = dhan.BUY if (is_options or algo_signal == 'Buy') else dhan.SELL
        security_id = config['ce_id'] if (is_options and algo_signal == 'Buy') else (config['pe_id'] if is_options else config['eq_id'])
        res = dhan.place_order(security_id=security_id, quantity=config['opt_qty'] if is_options else config['eq_qty'])
        return res
    except Exception as e: return f"Order failed: {e}"

# --- Data Engine ---
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
        todays_data = tkr.history(period='2d')
        if len(todays_data) >= 2:
            prev_close = float(np.squeeze(todays_data['Close'].iloc[-2]))
            ltp = float(np.squeeze(todays_data['Close'].iloc[-1]))
            diff = ltp - prev_close
            return ltp, diff, (diff / prev_close) * 100
    except: pass
    return None, None, None

# --- UI Sidebar ---
st.sidebar.title("⚙️ Smart Wealth Config")
selected_asset = st.sidebar.selectbox("Select Asset", list(TICKERS.keys()))
current_ticker = st.sidebar.text_input("Custom Ticker") if selected_asset == "Custom" else TICKERS[selected_asset]
tf = st.sidebar.selectbox("Timeframe", list(TIMEFRAME_PERIODS.keys()))
period = st.sidebar.selectbox("Period", TIMEFRAME_PERIODS[tf])

st.sidebar.markdown("### Strategy")
strategy = st.sidebar.selectbox("Strategy", ["EMA Crossover", "Simple Buy", "Simple Sell"])
fast_ema = st.sidebar.number_input("Fast EMA", value=9)
slow_ema = st.sidebar.number_input("Slow EMA", value=15)
sl_val = st.sidebar.number_input("SL Points", value=10.0)
tgt_val = st.sidebar.number_input("Target Points", value=20.0)

st.sidebar.markdown("### Execution")
cooldown_enabled = st.sidebar.checkbox("Enable Cooldown", value=True)
cooldown_sec = st.sidebar.number_input("Cooldown (s)", value=5)
no_overlap = st.sidebar.checkbox("Prevent Overlap", value=True)

st.sidebar.markdown("### Broker (Dhan)")
enable_dhan = st.sidebar.checkbox("Enable Dhan", value=False)
opt_enabled = st.sidebar.checkbox("Options Trading", value=False)

# Configuration mapping
config = {'enable_dhan': enable_dhan, 'opt_enabled': opt_enabled, 'ce_id': "123", 'pe_id': "456", 'eq_id': "1594", 'opt_qty': 65, 'eq_qty': 1}

# --- Header ---
st.title("📈 Smart Wealth")
ltp_val, diff_val, pct_val = get_ltp_stats(current_ticker)
if ltp_val:
    color = "green" if diff_val >= 0 else "red"
    st.markdown(f"**{selected_asset} LTP:** :blue[{ltp_val:.2f}] | <span style='color:{color}'>{'▲' if diff_val >= 0 else '▼'} {abs(diff_val):.2f} ({pct_val:.2f}%)</span>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🧪 Backtesting", "⚡ Live Trading", "📜 Trade History"])

# --- Tab 1: Backtesting ---
with tab1:
    if st.button("Run Backtest", key="run_bt"):
        df = fetch_data(current_ticker, period, tf)
        if df is not None:
            df = apply_indicators(df, fast_ema, slow_ema)
            trades, in_trade, violation_count, last_exit_time = [], False, 0, None
            
            for i in range(1, len(df)):
                row, prev_row = df.iloc[i], df.iloc[i-1]
                c_close, c_high, c_low = float(np.squeeze(row['Close'])), float(np.squeeze(row['High'])), float(np.squeeze(row['Low']))
                p_f, p_s, c_f, c_s = [float(np.squeeze(x)) for x in [prev_row['EMA_Fast'], prev_row['EMA_Slow'], row['EMA_Fast'], row['EMA_Slow']]]
                
                if in_trade:
                    h_sl, h_tg = (c_low <= sl if t_type == 'Buy' else c_high >= sl), (c_high >= tg if t_type == 'Buy' else c_low <= tg)
                    if h_sl and h_tg: violation_count += 1; h_tg = False
                    if h_sl or h_tg:
                        ex_p = sl if h_sl else tg
                        pnl = (ex_p - en_p) if t_type == 'Buy' else (en_p - ex_p)
                        trades.append({'Type': t_type, 'Entry': en_p, 'Exit': ex_p, 'PnL': pnl, 'Time': df.index[i], 'Reason': 'SL' if h_sl else 'Tgt'})
                        in_trade, last_exit_time = False, df.index[i]
                else:
                    if cooldown_enabled and last_exit_time and (df.index[i] - last_exit_time).total_seconds() < cooldown_sec: continue
                    if (p_f < p_s and c_f > c_s) or strategy == "Simple Buy":
                        in_trade, t_type, en_p = True, 'Buy', c_close
                        sl, tg = en_p - sl_val, en_p + tgt_val
                    elif (p_f > p_s and c_f < c_s) or strategy == "Simple Sell":
                        in_trade, t_type, en_p = True, 'Sell', c_close
                        sl, tg = en_p + sl_val, en_p - tgt_val

            if trades:
                tdf = pd.DataFrame(trades)
                won = tdf[tdf['PnL'] > 0]
                lost = tdf[tdf['PnL'] <= 0]
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Trades", len(tdf))
                c1.metric("Accuracy", f"{(len(won)/len(tdf))*100:.2f}%")
                c2.metric("Trades Won", len(won))
                c2.metric("Trades Lost", len(lost))
                c3.metric("Total Points", f"{tdf['PnL'].sum():.2f}")
                c4.metric("Points Won", f"{won['PnL'].sum():.2f}")
                c4.metric("Points Lost", f"{abs(lost['PnL'].sum()):.2f}")
                
                st.dataframe(tdf, use_container_width=True)
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
                st.plotly_chart(fig_bt, use_container_width=True, key="bt_chart")
            else: st.info("No trades found.")

# --- Tab 2: Live Trading ---
with tab2:
    c1, c2 = st.columns(2)
    if c1.button("▶ Start", use_container_width=True): st.session_state.live_running = True
    if c2.button("🛑 Stop", use_container_width=True): st.session_state.live_running = False
    
    stat_placeholder = st.empty()
    chart_placeholder = st.empty()

    if st.session_state.live_running:
        while st.session_state.live_running:
            df = fetch_data(current_ticker, '1d', tf)
            if df is not None:
                df = apply_indicators(df, fast_ema, slow_ema)
                last = df.iloc[-1]
                ltp = float(np.squeeze(last['Close']))
                f_ema, s_ema = float(np.squeeze(last['EMA_Fast'])), float(np.squeeze(last['EMA_Slow']))
                
                stat_placeholder.info(f"LTP: {ltp:.2f} | EMA({fast_ema}): {f_ema:.2f} | EMA({slow_ema}): {s_ema:.2f}")
                
                fig_live = go.Figure()
                fig_live.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].iloc[-60:], name="LTP"))
                fig_live.add_trace(go.Scatter(x=df.index[-60:], y=df['EMA_Fast'].iloc[-60:], name=f"EMA {fast_ema}", line=dict(color='blue')))
                fig_live.add_trace(go.Scatter(x=df.index[-60:], y=df['EMA_Slow'].iloc[-60:], name=f"EMA {slow_ema}", line=dict(color='orange')))
                
                # EMA Labels on plot
                fig_live.add_annotation(x=df.index[-1], y=f_ema, text=f"EMA{fast_ema}:{f_ema:.1f}", showarrow=False, xanchor="left", font=dict(color="blue"))
                fig_live.add_annotation(x=df.index[-1], y=s_ema, text=f"EMA{slow_ema}:{s_ema:.1f}", showarrow=False, xanchor="left", font=dict(color="orange"))
                
                fig_live.update_layout(margin=dict(l=0, r=50, t=30, b=0), height=500, uirevision='true')
                # FIX: Manual key to prevent ID conflicts
                chart_placeholder.plotly_chart(fig_live, use_container_width=True, key="live_plot_canvas")
                
            time.sleep(1.5)

# --- Tab 3: History ---
with tab3:
    if st.session_state.trade_history: st.dataframe(pd.DataFrame(st.session_state.trade_history))
    else: st.info("History will appear here.")
