import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats, signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from datetime import datetime
import time
import warnings

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="TradeFlow Pro AI v3.1", page_icon="📈")
warnings.filterwarnings('ignore')
IST = pytz.timezone('Asia/Kolkata')

# --- CONSTANTS ---
ANALYSIS_INTERVALS = ["15m", "1h", "1d"] # Timeframes for automatic analysis

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2127; padding: 10px; border-radius: 5px; border: 1px solid #333; }
    .guidance-box { background-color: #161b22; padding: 20px; border-left: 5px solid #00ff00; border-radius: 5px; margin-bottom: 20px; }
    .ratio-info { background-color: #1a1a2e; padding: 10px; border-radius: 5px; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'paper_trades' not in st.session_state: st.session_state.paper_trades = []
if 'capital' not in st.session_state: st.session_state.capital = 100000.0
if 'live_monitoring' not in st.session_state: st.session_state.live_monitoring = False
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = {} # Stores MTF results

# --- ASSETS ---
ASSETS = {
    "NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", 
    "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD", 
    "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS"
}

# --- DATA FETCHING (FIXED) ---
@st.cache_data(ttl=60) 
def fetch_data(ticker, interval, period='5y'):
    time.sleep(0.5)
    try:
        # Adjust period based on interval for efficiency
        if interval in ['1m', '5m', '15m']: period = '60d'
        elif interval == '1h': period = '730d' # 2 years
        
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty: return None
        
        # 1. Handle MultiIndex Columns
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
            
        # 2. Reset Index and Standardize Date Column Name (Addresses ValueError ambiguity)
        df = df.reset_index()
        # Rename the new index column (which could be 'Date', 'Datetime', or 'index') to 'Date'
        df = df.rename(columns=lambda x: 'Date' if x in ['index', 'level_0', 'Datetime', 'Date'] else x, errors='ignore')
        df = df.set_index('Date')
        
        # 3. Explicit Column Selection
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[required_cols]

        # 4. Handle Timezone
        if df.index.tz is None: df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert(IST)
        
        return df
    except Exception as e:
        st.error(f"Data Fetch Error for {ticker} ({interval}): {e}")
        return None

# --- TECHNICAL ANALYSIS ENGINE ---
class Analyzer:
    def __init__(self, df):
        self.df = df.copy()
        
    def add_indicators(self):
        df = self.df
        close = df['Close']
        for p in [20, 50, 200]: df[f'EMA_{p}'] = close.ewm(span=p, adjust=False).mean()
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        returns = close.pct_change()
        returns_std = returns.rolling(20).std()
        df['Z_Score'] = (returns - returns.rolling(20).mean()) / returns_std.replace(0, np.nan) 
        return df.fillna(method='bfill')

    def get_fibonacci(self):
        sub = self.df.iloc[-100:]
        high, low = sub['High'].max(), sub['Low'].min()
        diff = high - low
        levels = {0: high, 0.382: high - 0.382*diff, 0.5: high - 0.5*diff, 0.618: high - 0.618*diff, 1: low}
        curr = self.df['Close'].iloc[-1]
        closest_name, closest_price = min(levels.items(), key=lambda x: abs(x[1] - curr))
        dist_pct = (curr - closest_price) / closest_price * 100
        return (closest_name, closest_price, dist_pct)

    def get_support_resistance(self):
        closes = self.df['Close'].values
        peaks = signal.argrelextrema(closes, np.greater, order=5)[0]
        troughs = signal.argrelextrema(closes, np.less, order=5)[0]
        levels = list(closes[peaks]) + list(closes[troughs])
        levels.sort()
        clusters = []
        if not levels: return []
        
        temp = [levels[0]]
        for i in range(1, len(levels)):
            if (levels[i] - temp[-1])/temp[-1] < 0.003: temp.append(levels[i])
            else:
                if len(temp) >= 3: clusters.append(np.mean(temp))
                temp = [levels[i]]
        if len(temp) >= 3: clusters.append(np.mean(temp))
        return clusters

def calculate_ratio_analysis(df1, df2):
    if df1 is None or df2 is None or df1.empty or df2.empty: return None
    
    common_index = df1.index.intersection(df2.index)
    if common_index.empty: return None
    
    ratio_df = pd.DataFrame({
        'Ratio': df1['Close'].loc[common_index] / df2['Close'].loc[common_index]
    })
    
    ratio_df['Ratio_EMA_50'] = ratio_df['Ratio'].ewm(span=50, adjust=False).mean()
    
    current_ratio = ratio_df['Ratio'].iloc[-1]
    ratio_ema = ratio_df['Ratio_EMA_50'].iloc[-1]
    
    if current_ratio > ratio_ema * 1.01:
        return {"status": "Strong", "desc": f"The ratio ({current_ratio:.4f}) is running >1% above its 50-EMA, showing **strong relative strength**.", "diff": current_ratio - ratio_ema}
    elif current_ratio < ratio_ema * 0.99:
        return {"status": "Weak", "desc": f"The ratio ({current_ratio:.4f}) is running <1% below its 50-EMA, showing **relative weakness**.", "diff": current_ratio - ratio_ema}
    else:
        return {"status": "Neutral", "desc": "The ratio is consolidating around its 50-EMA, indicating **balanced relative strength**.", "diff": current_ratio - ratio_ema}

# --- NARRATIVE GENERATION ---
def get_entry_reason(curr):
    if curr['RSI'] < 30 and curr['Close'] > curr['EMA_20']:
        return "RSI Oversold Bounce (Buy)"
    if curr['RSI'] > 70 and curr['Close'] < curr['EMA_20']:
        return "RSI Overbought Rejection (Sell)"
    return "Manual Entry (No Signal)"

def manage_trade_narrative(trade, current_price, current_rsi, current_z):
    pnl_pct = (current_price - trade['price']) / trade['price'] * 100
    if trade['type'] == 'SELL': pnl_pct *= -1
    
    advice = ""
    if pnl_pct > 2.0: advice = "💰 **Take Profit Alert:** You are up >2%. Consider closing half."
    elif pnl_pct < -1.5: advice = "🛑 **Stop Loss Alert:** Position down >1.5%. Validate thesis."
    
    if trade['type'] == 'BUY' and current_rsi > 70: 
        advice += " ⚠️ **Warning:** RSI is Overbought (Invalidating BUY thesis)."
    elif trade['type'] == 'SELL' and current_rsi < 30:
        advice += " ⚠️ **Warning:** RSI is Oversold (Invalidating SELL thesis)."
    
    status = f"""
    **Trade Monitor ({trade['symbol']}):**
    * **Entry Signal:** {trade['entry_reason']}
    * **Entry Price:** {trade['price']:.2f} | **Current Price:** {current_price:.2f}
    * **P&L:** {pnl_pct:.2f}%
    * **Thesis Check:** Entry Z-Score was {trade['entry_z']:.2f}, now {current_z:.2f}.
    * **AI Advice:** {advice if advice else "✅ Hold Position. Trends align."}
    """
    return status

def generate_multi_timeframe_narrative(ticker, results, ratio_data):
    bull_count, bear_count = 0, 0
    total_score = 0
    
    report = "### 🕰️ Multi-Timeframe Summary\n"
    
    for interval, res in results.items():
        if res is None or res['df'].empty:
            report += f"* **{interval}:** Data unavailable.\n"
            continue
            
        df, fib_data = res['df'], res['fib_data']
        curr = df.iloc[-1]
        trend = "Bullish" if curr['Close'] > curr['EMA_50'] else "Bearish"
        
        if trend == "Bullish": bull_count += 1; total_score += 1
        else: bear_count += 1; total_score -= 1
        
        if curr['RSI'] < 30: total_score += 2
        elif curr['RSI'] > 70: total_score -= 2
        
        if abs(fib_data[2]) < 0.2: total_score += 1 
        
        report += f"* **{interval}:** {trend} trend. RSI {curr['RSI']:.1f}. Fib {fib_data[0]} test. Z-Score {curr['Z_Score']:.2f}.\n"

    final_rec = "WAIT / NEUTRAL 🟡"
    if total_score >= 3 and bull_count >= bear_count: final_rec = "STRONG BUY 🟢"
    elif total_score <= -3 and bear_count >= bull_count: final_rec = "STRONG SELL 🔴"

    confidence = min(100, 50 + abs(total_score) * 5)
    
    final_narrative = f"""
    # {final_rec}
    **Confidence:** **{confidence}%** (Based on {bull_count} Bullish vs {bear_count} Bearish signals across all analyzed timeframes: {', '.join(ANALYSIS_INTERVALS)}).

    **Ratio Analysis ({ratio_data['status']}):** {ratio_data['desc']}

    {report}
    """
    return final_narrative

# --- BACKTESTING LOGIC (INCREASED TRADES) ---
def run_backtest(df):
    if len(df) < 100: return pd.DataFrame()
    trades = []
    in_pos, entry_p, entry_d, sl, tp, pos_type = False, 0, None, 0, 0, None
    
    for i in range(50, len(df)):
        curr = df.iloc[i]
        
        if not in_pos:
            # Loosened condition: RSI 35/65 instead of 30/70 for more trades
            if curr['RSI'] < 35 and curr['Close'] > curr['EMA_20']:
                in_pos, entry_p, entry_d, pos_type = True, curr['Close'], curr.name, 'BUY'
                sl = entry_p - (1 * curr['ATR'])
                tp = entry_p + (2 * curr['ATR']) # Increased R/R slightly
            elif curr['RSI'] > 65 and curr['Close'] < curr['EMA_20']:
                in_pos, entry_p, entry_d, pos_type = True, curr['Close'], curr.name, 'SELL'
                sl = entry_p + (1 * curr['ATR'])
                tp = entry_p - (2 * curr['ATR'])

        else:
            exit_p, reason = None, None
            
            if pos_type == 'BUY':
                if curr['Low'] <= sl: exit_p, reason = sl, "SL"
                elif curr['High'] >= tp: exit_p, reason = tp, "TP"
            
            elif pos_type == 'SELL':
                if curr['High'] >= sl: exit_p, reason = sl, "SL"
                elif curr['Low'] <= tp: exit_p, reason = tp, "TP"

            if reason:
                pnl = (exit_p - entry_p) if pos_type == 'BUY' else (entry_p - exit_p)
                trades.append({'Entry': entry_d, 'Exit': curr.name, 'Type': pos_type, 'PnL': pnl, 'Reason': reason})
                in_pos = False
    
    return pd.DataFrame(trades)

# --- MAIN APP ---
def main():
    st.title("⚡ TradeFlow Pro AI: Multi-Timeframe Analyst")

    with st.sidebar:
        st.header("⚙️ Configuration")
        ticker = st.selectbox("Primary Asset", list(ASSETS.keys()))
        ratio_base = st.selectbox("Ratio Analysis Base Asset", ["None"] + list(ASSETS.keys()), index=1)
        
        st.info(f"Automatically analyzing timeframes: **{', '.join(ANALYSIS_INTERVALS)}**")
        
        st.divider()
        st.header("💼 Paper Trading")
        st.metric("Buying Power", f"₹{st.session_state.capital:,.2f}")
        
        add_capital = st.number_input("Increase Capital By", min_value=0.0, value=0.0, step=1000.0)
        if st.button("Add Funds"):
            st.session_state.capital += add_capital
            st.success(f"Added ₹{add_capital:,.2f} to capital.")
            st.rerun()

        # Manual Trade Input
        st.subheader("Manual Order")
        order_type = st.selectbox("Type", ["BUY", "SELL"], key="order_type")
        qty = st.number_input("Quantity", min_value=0.001, value=1.0, step=0.1, key="order_qty")
        
        if st.button("Submit Order", key="submit_order"):
            res_15m = st.session_state.analysis_results.get('15m')
            
            if res_15m and not res_15m['df'].empty:
                df_15m = res_15m['df']
                curr = df_15m.iloc[-1]
                curr_price = curr['Close']
                
                curr_z = curr['Z_Score'] if not np.isnan(curr['Z_Score']) else 0.0
                curr_rsi = curr['RSI']
                entry_reason = get_entry_reason(curr)
                
                cost = curr_price * qty
                
                if st.session_state.capital >= cost or order_type == "SELL":
                    st.session_state.paper_trades.append({
                        'time': datetime.now(IST), 'symbol': ticker, 'type': order_type,
                        'qty': qty, 'price': curr_price, 
                        'entry_z': curr_z, 
                        'entry_rsi': curr_rsi,
                        'entry_reason': entry_reason
                    })
                    if order_type == "BUY": st.session_state.capital -= cost
                    else: st.session_state.capital += cost
                    st.success(f"Order Placed: {order_type} {qty} @ {curr_price:.2f}")
                else:
                    st.error("Insufficient Funds")
            else:
                 st.warning("Analysis for 15m is not yet complete. Please wait for data to load.")
        
        st.divider()
        live_mode = st.toggle("🔴 Live Analysis Loop (5s)")
        if live_mode: st.session_state.live_monitoring = True
        else: st.session_state.live_monitoring = False

    # --- MAIN PAGE EXECUTION ---
    
    # 1. MTF Analysis Loop
    st.session_state.analysis_results = {}
    
    df_base = None
    if ratio_base != "None":
        df_base = fetch_data(ASSETS[ratio_base], '1d')

    for interval in ANALYSIS_INTERVALS:
        df = fetch_data(ASSETS[ticker], interval)
        if df is not None and not df.empty:
            analyzer = Analyzer(df)
            df = analyzer.add_indicators()
            st.session_state.analysis_results[interval] = {
                'df': df,
                'fib_data': analyzer.get_fibonacci(),
                'sr_levels': analyzer.get_support_resistance()
            }
        else:
            st.session_state.analysis_results[interval] = None

    # Ratio Analysis
    df_1d_primary = st.session_state.analysis_results.get('1d', {}).get('df')
    if df_1d_primary is not None and df_base is not None:
        ratio_analysis = calculate_ratio_analysis(df_1d_primary, df_base)
    else:
        ratio_analysis = {"status": "Disabled", "desc": "Ratio analysis disabled or base asset data not found.", "diff": 0}

    # 2. UI Layout
    res_15m = st.session_state.analysis_results.get('15m')
    
    if res_15m and not res_15m['df'].empty:
        df_15m = res_15m['df']
        current = df_15m.iloc[-1]
        
        t1, t2, t3, t4 = st.tabs(["AI Guidance", "Charts (15m)", "Paper Trade Manager", "Backtest"])
        
        with t1:
            narrative = generate_multi_timeframe_narrative(ticker, st.session_state.analysis_results, ratio_analysis)
            st.markdown(f"<div class='guidance-box'>{narrative}</div>", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"{current['Close']:.2f}", f"{df_15m['Close'].pct_change().iloc[-1]*100:.2f}%")
            c2.metric("15m RSI", f"{current['RSI']:.2f}")
            c3.metric("15m Z-Score", f"{current['Z_Score']:.2f}")

        with t2:
            st.subheader(f"{ticker} Price Action - 15 Minute Interval")
            # Chart logic using 15m data
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df_15m.index, open=df_15m['Open'], high=df_15m['High'], low=df_15m['Low'], close=df_15m['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_15m.index, y=df_15m['EMA_50'], line=dict(color='cyan'), name='EMA 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_15m.index, y=df_15m['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_color='red', row=2, col=1); fig.add_hline(y=30, line_color='green', row=2, col=1)
            fig.update_layout(height=600, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with t3:
            st.subheader(f"Active Trades for {ticker}")
            my_trades = [t for t in st.session_state.paper_trades if t['symbol'] == ticker]
            
            if my_trades and my_trades[-1]['qty'] > 0:
                last_trade = my_trades[-1]
                curr_z = current['Z_Score'] if not np.isnan(current['Z_Score']) else 0.0
                advice = manage_trade_narrative(last_trade, current['Close'], current['RSI'], curr_z)
                st.markdown(f"<div class='trade-panel'>{advice}</div>", unsafe_allow_html=True)
                
                if st.button("Close Last Position"):
                    pnl = (current['Close'] - last_trade['price']) * last_trade['qty']
                    if last_trade['type'] == 'SELL': pnl *= -1
                    st.session_state.capital += (last_trade['qty'] * current['Close']) + pnl 
                    st.session_state.paper_trades.remove(last_trade)
                    st.success(f"Closed P&L: {pnl:.2f}")
                    st.rerun()

            st.subheader("Global Portfolio Dashboard")
            if st.session_state.paper_trades:
                p_df = pd.DataFrame(st.session_state.paper_trades)
                p_df['Current Price'] = current['Close']
                
                def calc_pnl(row):
                    val = (row['Current Price'] - row['price']) * row['qty']
                    return val if row['type'] == 'BUY' else -val
                    
                p_df['Unrealized P&L'] = p_df.apply(calc_pnl, axis=1)
                
                st.dataframe(p_df[['time', 'symbol', 'type', 'qty', 'price', 'Current Price', 'entry_reason', 'Unrealized P&L']], use_container_width=True)
            else:
                st.info("No trades in portfolio.")

        with t4:
            st.subheader("Backtesting Results (1D Interval)")
            df_1d = st.session_state.analysis_results.get('1d', {}).get('df')
            if df_1d is not None:
                bt_df = run_backtest(df_1d)
                
                if not bt_df.empty:
                    total_pnl = bt_df['PnL'].sum()
                    win_rate = (bt_df['PnL'] > 0).sum() / len(bt_df) * 100
                    total_trades = len(bt_df)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Trades", total_trades)
                    c2.metric("Total PnL", f"₹{total_pnl:,.2f}")
                    c3.metric("Win Rate", f"{win_rate:.1f}%")
    
                    st.markdown(f"***Strategy based on RSI & EMA20, using 1D timeframe over {len(df_1d)} candles.***")
                    st.dataframe(bt_df.sort_values('Entry', ascending=False), use_container_width=True)
                else:
                    st.warning("Backtest found no profitable signals for the 1D timeframe using current strategy.")
            else:
                st.warning("1D data required for backtesting is not available.")

    if st.session_state.live_monitoring:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
