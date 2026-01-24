import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import pytz
import time
import random

st.set_page_config(page_title="Quantitative Trading System", layout="wide")

# Initialize session state
for key in ['trading_active', 'current_data', 'position', 'trade_history', 'trade_logs', 
            'trailing_sl_high', 'trailing_sl_low', 'trailing_target_high', 'trailing_target_low',
            'trailing_profit_points', 'threshold_crossed', 'highest_price', 'lowest_price',
            'partial_exit_done', 'breakeven_activated']:
    if key not in st.session_state:
        if 'history' in key or 'logs' in key:
            st.session_state[key] = []
        elif 'active' in key:
            st.session_state[key] = False
        elif 'points' in key:
            st.session_state[key] = 0
        elif 'crossed' in key or 'done' in key or 'activated' in key:
            st.session_state[key] = False
        else:
            st.session_state[key] = None

# Utility functions
def get_ist_time():
    return datetime.now(pytz.timezone('Asia/Kolkata'))

def format_ist_time(dt):
    if dt is None or isinstance(dt, str):
        return dt if dt else "N/A"
    return dt.strftime("%Y-%m-%d %H:%M:%S IST")

def add_log(msg):
    log = f"[{format_ist_time(get_ist_time())}] {msg}"
    st.session_state['trade_logs'].append(log)
    if len(st.session_state['trade_logs']) > 50:
        st.session_state['trade_logs'] = st.session_state['trade_logs'][-50:]

def reset_position():
    for key in ['position', 'trailing_sl_high', 'trailing_sl_low', 'trailing_target_high',
                'trailing_target_low', 'threshold_crossed', 'highest_price', 'lowest_price',
                'partial_exit_done', 'breakeven_activated']:
        if 'points' in key:
            st.session_state[key] = 0
        elif 'crossed' in key or 'done' in key or 'activated' in key:
            st.session_state[key] = False
        else:
            st.session_state[key] = None

# Indicators
def calc_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calc_rsi(data, period=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_atr(h, l, c, period=14):
    tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_adx(h, l, c, period=14):
    pdm = h.diff()
    mdm = -l.diff()
    pdm[pdm < 0] = 0
    mdm[mdm < 0] = 0
    tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    pdi = 100 * (pdm.rolling(period).mean() / atr)
    mdi = 100 * (mdm.rolling(period).mean() / atr)
    dx = 100 * abs(pdi - mdi) / (pdi + mdi)
    return dx.rolling(period).mean()

def calc_angle(ema_vals, period=2):
    if len(ema_vals) < period + 1:
        return pd.Series([0] * len(ema_vals), index=ema_vals.index)
    slope = ema_vals.diff(period)
    return np.degrees(np.arctan(slope))

# Data fetch
def fetch_data(ticker, interval, period, mode):
    try:
        if mode == "Live Trading":
            time.sleep(random.uniform(1.0, 1.5))
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in data.columns]
        data = data[cols]
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            data.index = data.index.tz_convert('Asia/Kolkata')
        return data
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Sidebar
st.sidebar.header("⚙️ Configuration")

asset_map = {"NIFTY 50": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN",
             "BTC": "BTC-USD", "ETH": "ETH-USD", "USDINR": "USDINR=X",
             "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "GOLD": "GC=F", "SILVER": "SI=F"}

asset = st.sidebar.selectbox("Asset", list(asset_map.keys()) + ["Custom"])
ticker = st.sidebar.text_input("Ticker", asset_map.get(asset, "^NSEI")) if asset == "Custom" else asset_map[asset]

interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"])

period_map = {"1m": ["1d", "5d"], "5m": ["1d", "1mo"],
              "15m": ["1mo"], "30m": ["1mo"], "1h": ["1mo"], "4h": ["1mo"],
              "1d": ["1mo", "1y", "2y", "5y"],
              "1wk": ["1mo", "1y", "5y", "10y", "15y", "20y"],
              "1mo": ["1y", "2y", "5y", "10y", "15y", "20y", "25y", "30y"]}

period = st.sidebar.selectbox("Period", period_map.get(interval, ["1mo"]))
mode = st.sidebar.selectbox("Mode", ["Live Trading", "Backtest"])
quantity = st.sidebar.number_input("Quantity", min_value=1, value=1)

strategy = st.sidebar.selectbox("Strategy", ["EMA Crossover", "Simple Buy", "Simple Sell",
                                              "Price Threshold", "Percentage Change"])

cfg = {}
if strategy == "EMA Crossover":
    st.sidebar.subheader("EMA Settings")
    cfg['ema_fast'] = st.sidebar.number_input("Fast", value=9, key="ef")
    cfg['ema_slow'] = st.sidebar.number_input("Slow", value=15, key="es")
    cfg['min_angle'] = st.sidebar.number_input("Min Angle", value=1.0, key="ma")
    cfg['use_adx'] = st.sidebar.checkbox("Use ADX Filter")
    if cfg['use_adx']:
        cfg['adx_threshold'] = st.sidebar.number_input("ADX Threshold", value=25.0, key="at")

elif strategy == "Price Threshold":
    cfg['threshold'] = st.sidebar.number_input("Threshold", value=100.0, key="th")
    cfg['direction'] = st.sidebar.selectbox("Direction", ["LONG (>=)", "SHORT (>=)", "LONG (<=)", "SHORT (<=)"])

elif strategy == "Percentage Change":
    cfg['pct_threshold'] = st.sidebar.number_input("% Threshold", value=0.01, step=0.01, key="pt")
    cfg['pct_direction'] = st.sidebar.selectbox("Direction", ["BUY on Fall", "SELL on Fall", "BUY on Rise", "SELL on Rise"])

st.sidebar.subheader("Stop Loss")
sl_type = st.sidebar.selectbox("SL Type", ["Custom Points", "Trailing SL (Points)", "ATR-based", "Signal-based"])
cfg['sl_type'] = sl_type
if "Points" in sl_type:
    cfg['sl_points'] = st.sidebar.number_input("SL Points", value=10.0, key="slp")
if "ATR" in sl_type:
    cfg['sl_atr'] = st.sidebar.number_input("ATR Mult (SL)", value=1.5, key="sla")

st.sidebar.subheader("Target")
tgt_type = st.sidebar.selectbox("Target Type", ["Custom Points", "Trailing Target", "ATR-based", "Signal-based"])
cfg['target_type'] = tgt_type
if "Points" in tgt_type:
    cfg['target_points'] = st.sidebar.number_input("Target Points", value=20.0, key="tp")
if "ATR" in tgt_type:
    cfg['tgt_atr'] = st.sidebar.number_input("ATR Mult (Tgt)", value=3.0, key="ta")

st.sidebar.subheader("Dhan (Placeholder)")
use_dhan = st.sidebar.checkbox("Enable Dhan")
if use_dhan:
    st.sidebar.text_input("Client ID", key="dci")
    st.sidebar.text_input("Token", type="password", key="dtk")
    st.sidebar.info("Placeholder only")

# Main UI
st.title("🎯 Professional Quantitative Trading System")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Dashboard", "📈 Trade History", "📝 Logs", "🔬 Backtest"])

with tab1:
    st.header("Live Trading Dashboard")
    
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("▶️ Start", type="primary", use_container_width=True):
            st.session_state['trading_active'] = True
            add_log("Trading started")
            st.rerun()
    with c2:
        if st.button("⏹️ Stop", type="secondary", use_container_width=True):
            st.session_state['trading_active'] = False
            if st.session_state['position']:
                pos = st.session_state['position']
                ep = pos['entry_price']
                xp = st.session_state['current_data']['Close'].iloc[-1] if st.session_state['current_data'] is not None else ep
                pnl = (xp - ep) * quantity if pos['signal'] == 1 else (ep - xp) * quantity
                st.session_state['trade_history'].append({
                    'entry_time': pos['entry_time'], 'exit_time': get_ist_time(),
                    'signal': 'LONG' if pos['signal'] == 1 else 'SHORT',
                    'entry_price': ep, 'exit_price': xp, 'sl': pos.get('sl', 0),
                    'target': pos.get('target', 0), 'exit_reason': 'Manual Close',
                    'pnl': pnl, 'highest': pos.get('highest', xp), 'lowest': pos.get('lowest', xp)
                })
                add_log(f"Manual close. PnL: {pnl:.2f}")
            reset_position()
            add_log("Trading stopped")
            st.rerun()
    with c3:
        st.success("🟢 ACTIVE") if st.session_state['trading_active'] else st.info("⚪ STOPPED")
    
    if st.button("🔄 Refresh"):
        st.rerun()
    
    st.subheader("Configuration")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.write(f"**Asset:** {asset} ({ticker})")
        st.write(f"**Interval:** {interval} | **Period:** {period}")
    with cc2:
        st.write(f"**Qty:** {quantity} | **Strategy:** {strategy}")
        st.write(f"**Mode:** {mode}")
    with cc3:
        st.write(f"**SL:** {sl_type}")
        st.write(f"**Target:** {tgt_type}")
    
    if st.session_state['trading_active'] and mode == "Live Trading":
        ph = st.empty()
        
        while st.session_state['trading_active']:
            with ph.container():
                df = fetch_data(ticker, interval, period, mode)
                if df is None or len(df) == 0:
                    st.error("No data")
                    time.sleep(2)
                    continue
                
                st.session_state['current_data'] = df
                
                df['EMA_F'] = calc_ema(df['Close'], cfg.get('ema_fast', 9))
                df['EMA_S'] = calc_ema(df['Close'], cfg.get('ema_slow', 15))
                df['Angle'] = calc_angle(df['EMA_F'])
                df['RSI'] = calc_rsi(df['Close'])
                df['ADX'] = calc_adx(df['High'], df['Low'], df['Close'])
                df['ATR'] = calc_atr(df['High'], df['Low'], df['Close'])
                
                cp = df['Close'].iloc[-1]
                ef = df['EMA_F'].iloc[-1]
                es = df['EMA_S'].iloc[-1]
                ang = abs(df['Angle'].iloc[-1])
                rsi = df['RSI'].iloc[-1]
                adx = df['ADX'].iloc[-1]
                
                st.subheader("Live Metrics")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Price", f"{cp:.2f}")
                m2.metric("EMA Fast", f"{ef:.2f}")
                m3.metric("EMA Slow", f"{es:.2f}")
                m4.metric("RSI", f"{rsi:.2f}")
                m5.metric("ADX", f"{adx:.2f}")
                
                m6, m7, m8 = st.columns(3)
                m6.metric("Angle", f"{ang:.2f}°")
                m7.metric("Updated", format_ist_time(get_ist_time()))
                
                pos = st.session_state['position']
                
                if pos is None:
                    sig = 0
                    
                    if strategy == "EMA Crossover":
                        if ef > es and df['EMA_F'].iloc[-2] <= df['EMA_S'].iloc[-2]:
                            if ang >= cfg.get('min_angle', 1.0):
                                if not cfg.get('use_adx') or adx >= cfg.get('adx_threshold', 25):
                                    sig = 1
                        elif ef < es and df['EMA_F'].iloc[-2] >= df['EMA_S'].iloc[-2]:
                            if ang >= cfg.get('min_angle', 1.0):
                                if not cfg.get('use_adx') or adx >= cfg.get('adx_threshold', 25):
                                    sig = -1
                    
                    elif strategy == "Simple Buy":
                        sig = 1
                    elif strategy == "Simple Sell":
                        sig = -1
                    
                    elif strategy == "Price Threshold":
                        th = cfg['threshold']
                        dr = cfg['direction']
                        if "LONG (>=" in dr and cp >= th:
                            sig = 1
                        elif "SHORT (>=" in dr and cp >= th:
                            sig = -1
                        elif "LONG (<=" in dr and cp <= th:
                            sig = 1
                        elif "SHORT (<=" in dr and cp <= th:
                            sig = -1
                    
                    elif strategy == "Percentage Change":
                        fp = df['Close'].iloc[0]
                        pct = ((cp - fp) / fp) * 100
                        pth = cfg['pct_threshold']
                        pdr = cfg['pct_direction']
                        if "BUY on Fall" in pdr and pct <= -pth:
                            sig = 1
                        elif "SELL on Fall" in pdr and pct <= -pth:
                            sig = -1
                        elif "BUY on Rise" in pdr and pct >= pth:
                            sig = 1
                        elif "SELL on Rise" in pdr and pct >= pth:
                            sig = -1
                    
                    if sig != 0:
                        ep = cp
                        slp = cfg.get('sl_points', 10)
                        tgp = cfg.get('target_points', 20)
                        
                        sl = ep - slp if sig == 1 else ep + slp
                        tg = ep + tgp if sig == 1 else ep - tgp
                        
                        st.session_state['position'] = {
                            'entry_time': get_ist_time(), 'entry_price': ep,
                            'signal': sig, 'sl': sl, 'target': tg,
                            'highest': ep, 'lowest': ep
                        }
                        add_log(f"{'LONG' if sig == 1 else 'SHORT'} @ {ep:.2f}, SL: {sl:.2f}, Tgt: {tg:.2f}")
                        m8.success("✅ Entered")
                
                else:
                    ep = pos['entry_price']
                    sl = pos['sl']
                    tg = pos['target']
                    sig = pos['signal']
                    
                    if pos['highest'] is None or cp > pos['highest']:
                        pos['highest'] = cp
                    if pos['lowest'] is None or cp < pos['lowest']:
                        pos['lowest'] = cp
                    
                    pnl = (cp - ep) * quantity if sig == 1 else (ep - cp) * quantity
                    
                    st.subheader("Position")
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("Entry", f"{ep:.2f}")
                    p1.metric("Type", "LONG" if sig == 1 else "SHORT")
                    p2.metric("SL", f"{sl:.2f}")
                    p2.metric("Dist SL", f"{abs(cp - sl):.2f}")
                    p3.metric("Target", f"{tg:.2f}")
                    p3.metric("Dist Tgt", f"{abs(tg - cp):.2f}")
                    p4.metric("P&L", f"{pnl:.2f}", delta=f"{'+' if pnl >= 0 else ''}{pnl:.2f}",
                             delta_color="normal" if pnl >= 0 else "inverse")
                    
                    exit_trig = False
                    exit_rsn = ""
                    
                    if sig == 1 and cp <= sl:
                        exit_trig, exit_rsn = True, "SL Hit"
                    elif sig == -1 and cp >= sl:
                        exit_trig, exit_rsn = True, "SL Hit"
                    
                    if sig == 1 and cp >= tg:
                        exit_trig, exit_rsn = True, "Target Hit"
                    elif sig == -1 and cp <= tg:
                        exit_trig, exit_rsn = True, "Target Hit"
                    
                    if "Signal-based" in sl_type or "Signal-based" in tgt_type:
                        if sig == 1 and ef < es and df['EMA_F'].iloc[-2] >= df['EMA_S'].iloc[-2]:
                            exit_trig, exit_rsn = True, "Reverse Signal"
                        elif sig == -1 and ef > es and df['EMA_F'].iloc[-2] <= df['EMA_S'].iloc[-2]:
                            exit_trig, exit_rsn = True, "Reverse Signal"
                    
                    if exit_trig:
                        st.session_state['trade_history'].append({
                            'entry_time': pos['entry_time'], 'exit_time': get_ist_time(),
                            'signal': 'LONG' if sig == 1 else 'SHORT',
                            'entry_price': ep, 'exit_price': cp, 'sl': sl, 'target': tg,
                            'exit_reason': exit_rsn, 'pnl': pnl,
                            'highest': pos['highest'], 'lowest': pos['lowest']
                        })
                        add_log(f"Exit: {exit_rsn}. PnL: {pnl:.2f}")
                        reset_position()
                        st.success(f"✅ {exit_rsn}")
                
                st.subheader("Chart")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                            low=df['Low'], close=df['Close'], name='Price'))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_F'], name='Fast', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_S'], name='Slow', line=dict(color='red')))
                
                if pos:
                    fig.add_hline(y=pos['entry_price'], line_dash="dash", line_color="yellow", annotation_text="Entry")
                    fig.add_hline(y=pos['sl'], line_dash="dash", line_color="red", annotation_text="SL")
                    fig.add_hline(y=pos['target'], line_dash="dash", line_color="green", annotation_text="Target")
                
                fig.update_layout(xaxis_rangeslider_visible=False, height=500)
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{int(time.time())}")
            
            if not st.session_state['trading_active']:
                break
            time.sleep(random.uniform(1.0, 1.5))

with tab2:
    st.markdown("### 📈 Trade History")
    
    if len(st.session_state['trade_history']) == 0:
        st.info("No trades yet")
    else:
        total = len(st.session_state['trade_history'])
        wins = sum(1 for t in st.session_state['trade_history'] if t['pnl'] > 0)
        losses = total - wins
        acc = (wins / total * 100) if total > 0 else 0
        tot_pnl = sum(t['pnl'] for t in st.session_state['trade_history'])
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total", total)
        m2.metric("Wins", wins)
        m3.metric("Losses", losses)
        m4.metric("Accuracy", f"{acc:.1f}%")
        m5.metric("Total P&L", f"{tot_pnl:.2f}",
                 delta=f"{'+' if tot_pnl >= 0 else ''}{tot_pnl:.2f}",
                 delta_color="normal" if tot_pnl >= 0 else "inverse")
        
        for i, t in enumerate(reversed(st.session_state['trade_history'])):
            with st.expander(f"Trade #{total - i}: {t['signal']} | P&L: {t['pnl']:.2f}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Entry:** {format_ist_time(t['entry_time'])}")
                    st.write(f"**Entry Price:** {t['entry_price']:.2f}")
                    st.write(f"**SL:** {t['sl']:.2f}")
                    st.write(f"**Highest:** {t['highest']:.2f}")
                with c2:
                    st.write(f"**Exit:** {format_ist_time(t['exit_time'])}")
                    st.write(f"**Exit Price:** {t['exit_price']:.2f}")
                    st.write(f"**Target:** {t['target']:.2f}")
                    st.write(f"**Lowest:** {t['lowest']:.2f}")
                st.write(f"**Exit Reason:** {t['exit_reason']}")
                pnl_color = "green" if t['pnl'] >= 0 else "red"
                st.markdown(f"**P&L:** :<color>{pnl_color}</color>[**{t['pnl']:.2f}**]")

with tab3:
    st.markdown("### 📝 Trade Logs")
    
    if len(st.session_state['trade_logs']) == 0:
        st.info("No logs yet")
    else:
        for log in reversed(st.session_state['trade_logs']):
            st.text(log)

with tab4:
    st.markdown("### 🔬 Backtest Results")
    
    if mode != "Backtest":
        st.warning("Switch to Backtest mode in sidebar")
    else:
        if st.button("▶️ Run Backtest"):
            st.info("Backtest functionality: Fetching data and simulating trades...")
            st.info("This is a simplified placeholder. Full backtest engine requires additional code.")
