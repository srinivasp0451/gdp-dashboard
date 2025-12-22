import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import zscore
from scipy.signal import argrelextrema
import pytz
from datetime import datetime, timedelta
import time
import warnings

# Suppress warnings for cleaner UI
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="ProAlgo Quant Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Dark Theme CSS
st.markdown("""
<style>
    .main {background-color: #0E1117;}
    .stMetric {background-color: #1E1E1E; border: 1px solid #333; padding: 10px; border-radius: 5px;}
    .big-font {font-size: 1.5rem !important; font-weight: 600;}
    .success-text {color: #00FF00;}
    .danger-text {color: #FF0000;}
    .warning-text {color: #FFA500;}
    /* Table Styling */
    thead tr th:first-child {display:none}
    tbody th {display:none}
</style>
""", unsafe_allow_html=True)

TICKER_MAP = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "GOLD (Comex)": "GC=F",
    "USD/INR": "INR=X",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFCBANK": "HDFCBANK.NS"
}

# Auto-map periods to timeframes for batch processing
TIMEFRAME_CONFIG = [
    {"interval": "5m", "period": "5d"},   # Intraday
    {"interval": "15m", "period": "1mo"}, # Swing
    {"interval": "1h", "period": "1y"},   # Short Term
    {"interval": "1d", "period": "2y"},   # Long Term
]

IST = pytz.timezone('Asia/Kolkata')

# ==========================================
# 2. DATA ENGINE
# ==========================================

def get_readable_time(dt_val):
    if pd.isna(dt_val): return "-"
    if dt_val.tzinfo is None: dt_val = dt_val.replace(tzinfo=IST)
    now = datetime.now(IST)
    diff = now - dt_val
    seconds = diff.total_seconds()
    
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    elif seconds < 86400:
        return f"{int(seconds // 3600)} hours ago"
    elif seconds < 2592000: # 30 days
        return f"{int(seconds // 86400)} days ago"
    else:
        return dt_val.strftime("%Y-%m-%d")

@st.cache_data(ttl=600)
def fetch_ticker_data(symbol, period, interval):
    """Robust data fetcher with rate limiting"""
    time.sleep(1.5) # Anti-ban delay
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        if df.empty: return None
        
        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Clean & Localize
        df = df.dropna(subset=['Close'])
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)
            
        return df
    except Exception as e:
        return None

# ==========================================
# 3. ANALYTICS CORE (The Brain)
# ==========================================

class QuantAnalyzer:
    def __init__(self, df, timeframe):
        self.df = df.copy()
        self.tf = timeframe
        self.prepare_indicators()
        self.run_statistical_analysis()

    def prepare_indicators(self):
        # Basic Moving Averages
        self.df['EMA_9'] = self.df['Close'].ewm(span=9).mean()
        self.df['EMA_20'] = self.df['Close'].ewm(span=20).mean()
        self.df['EMA_50'] = self.df['Close'].ewm(span=50).mean()
        self.df['EMA_200'] = self.df['Close'].ewm(span=200).mean()
        
        # RSI
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = self.df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = exp1 - exp2
        self.df['Signal_Line'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Volatility & Z-Score
        self.df['Log_Ret'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        # Annualized volatility factor depends on timeframe, approximated here
        factor = 252 if 'd' in self.tf else 252*6 if 'h' in self.tf else 252*75
        self.df['Volatility'] = self.df['Log_Ret'].rolling(20).std() * np.sqrt(factor) * 100
        
        # Z-Score of Price relative to 20SMA (Bollinger Z)
        self.df['Z_Score'] = (self.df['Close'] - self.df['EMA_20']) / (self.df['Close'].rolling(20).std())
        
        # ATR for SL/Target
        self.df['TR'] = np.maximum(self.df['High'] - self.df['Low'], 
                                   np.maximum(abs(self.df['High'] - self.df['Close'].shift()), 
                                              abs(self.df['Low'] - self.df['Close'].shift())))
        self.df['ATR'] = self.df['TR'].rolling(14).mean()

    def run_statistical_analysis(self):
        """
        Calculates Binning Logic:
        If current Z-Score is X, what happened historically when Z-Score was X?
        """
        # Create Bins for Z-Score
        try:
            self.df['Z_Bin'] = pd.cut(self.df['Z_Score'], bins=[-np.inf, -2, -1, 0, 1, 2, np.inf], 
                                      labels=['Extreme Low', 'Low', 'Mild Low', 'Mild High', 'High', 'Extreme High'])
            
            # Forward Returns (Next 5 candles)
            self.df['Fwd_Ret_5'] = self.df['Close'].shift(-5) - self.df['Close']
            self.df['Fwd_Ret_Pct'] = (self.df['Fwd_Ret_5'] / self.df['Close']) * 100
        except:
            pass # Handle short data errors

    def get_bin_stats(self):
        """Returns statistical probability for current market state"""
        if 'Z_Bin' not in self.df.columns: return None
        
        curr_bin = self.df['Z_Bin'].iloc[-1]
        
        # Filter historical occurrences of this bin
        history = self.df[self.df['Z_Bin'] == curr_bin]
        if len(history) < 5: return None
        
        # Calculate stats
        avg_move = history['Fwd_Ret_Pct'].mean()
        win_rate = len(history[history['Fwd_Ret_Pct'] > 0]) / len(history) * 100
        
        return {
            'Current_Bin': curr_bin,
            'Occurrences': len(history),
            'Avg_Next_5_Bars_Move': avg_move,
            'Bullish_Probability': win_rate,
            'Bearish_Probability': 100 - win_rate
        }

    def detect_patterns(self):
        """Returns active support/resistance and divergences"""
        curr_price = self.df['Close'].iloc[-1]
        
        # S/R via Histogram
        counts, bins = np.histogram(self.df['Close'], bins=20)
        top_idx = np.argsort(counts)[-3:][::-1] # Top 3 zones
        zones = [(bins[i] + bins[i+1])/2 for i in top_idx]
        
        nearest_zone = min(zones, key=lambda x: abs(x - curr_price))
        dist = curr_price - nearest_zone
        
        # RSI Divergence
        div = "None"
        if len(self.df) > 30:
            price_lows = argrelextrema(self.df['Close'].values, np.less, order=5)[0]
            if len(price_lows) >= 2:
                last_2_idx = price_lows[-2:]
                if self.df['Close'].iloc[last_2_idx[1]] < self.df['Close'].iloc[last_2_idx[0]]:
                    if self.df['RSI'].iloc[last_2_idx[1]] > self.df['RSI'].iloc[last_2_idx[0]]:
                        div = "Bullish Divergence"
                        
        return {
            'Nearest_Zone': nearest_zone,
            'Distance': dist,
            'Divergence': div
        }

    def generate_signal(self):
        """Composite Logic Signal"""
        row = self.df.iloc[-1]
        trend = "Bullish" if row['Close'] > row['EMA_200'] else "Bearish"
        momentum = "Bullish" if row['MACD'] > row['Signal_Line'] else "Bearish"
        rsi_state = "Overbought" if row['RSI'] > 70 else "Oversold" if row['RSI'] < 30 else "Neutral"
        
        score = 0
        if trend == "Bullish": score += 1
        if momentum == "Bullish": score += 1
        if row['RSI'] < 60 and row['RSI'] > 40: score += 0.5 # Sustainable zone
        
        signal = "NEUTRAL"
        if score >= 2: signal = "BUY"
        elif score <= 0.5: signal = "SELL"
        
        return signal, trend, momentum, rsi_state

# ==========================================
# 4. ENHANCED BACKTESTER (MACD + Trend)
# ==========================================

def run_smart_backtest(df):
    """
    Strategy: 
    1. Trend Filter: Price > EMA 200 (Long) / Price < EMA 200 (Short)
    2. Entry: MACD Crossover
    3. Exit: MACD Crossunder OR Stop Loss (2 * ATR) OR Target (3 * ATR)
    """
    df = df.dropna().copy()
    if len(df) < 200: return None
    
    trades = []
    position = 0 # 0=None, 1=Long, -1=Short
    entry_price = 0
    sl = 0
    tp = 0
    
    # Pre-calculate crossover boolean series for speed
    df['MACD_Cross_Up'] = (df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) < df['Signal_Line'].shift(1))
    df['MACD_Cross_Down'] = (df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) > df['Signal_Line'].shift(1))
    
    for i in range(200, len(df)):
        row = df.iloc[i]
        date = df.index[i]
        
        # --- EXIT LOGIC ---
        if position == 1:
            # Check SL/TP
            if row['Low'] <= sl:
                trades.append({'Date': date, 'Type': 'SL Hit', 'P/L': sl - entry_price, 'Side': 'Long'})
                position = 0
            elif row['High'] >= tp:
                trades.append({'Date': date, 'Type': 'TP Hit', 'P/L': tp - entry_price, 'Side': 'Long'})
                position = 0
            # Indicator Exit
            elif row['MACD_Cross_Down']:
                trades.append({'Date': date, 'Type': 'Exit Signal', 'P/L': row['Close'] - entry_price, 'Side': 'Long'})
                position = 0
                
        elif position == -1:
            if row['High'] >= sl:
                trades.append({'Date': date, 'Type': 'SL Hit', 'P/L': entry_price - sl, 'Side': 'Short'})
                position = 0
            elif row['Low'] <= tp:
                trades.append({'Date': date, 'Type': 'TP Hit', 'P/L': entry_price - tp, 'Side': 'Short'})
                position = 0
            elif row['MACD_Cross_Up']:
                trades.append({'Date': date, 'Type': 'Exit Signal', 'P/L': entry_price - row['Close'], 'Side': 'Short'})
                position = 0

        # --- ENTRY LOGIC ---
        if position == 0:
            atr = row['ATR']
            
            # Long Setup
            if row['Close'] > row['EMA_200'] and row['MACD_Cross_Up']:
                position = 1
                entry_price = row['Close']
                sl = entry_price - (2.0 * atr)
                tp = entry_price + (3.0 * atr)
                trades.append({'Date': date, 'Type': 'Entry', 'P/L': 0, 'Side': 'Long'})
                
            # Short Setup
            elif row['Close'] < row['EMA_200'] and row['MACD_Cross_Down']:
                position = -1
                entry_price = row['Close']
                sl = entry_price + (2.0 * atr)
                tp = entry_price - (3.0 * atr)
                trades.append({'Date': date, 'Type': 'Entry', 'P/L': 0, 'Side': 'Short'})

    # Summary Stats
    if not trades: return None
    
    trade_df = pd.DataFrame(trades)
    closed_trades = trade_df[trade_df['Type'] != 'Entry']
    
    if closed_trades.empty: return None
    
    wins = len(closed_trades[closed_trades['P/L'] > 0])
    total = len(closed_trades)
    accuracy = (wins / total) * 100
    total_pnl = closed_trades['P/L'].sum()
    
    return {
        'trades': total,
        'accuracy': accuracy,
        'pnl': total_pnl,
        'log': trade_df
    }

# ==========================================
# 5. UI LAYOUT & EXECUTION
# ==========================================

st.sidebar.title("🎛️ Control Panel")
selected_ticker = st.sidebar.selectbox("Select Asset", list(TICKER_MAP.keys()))
custom_ticker = st.sidebar.text_input("Or Custom Ticker (e.g., MSFT)")
symbol = custom_ticker if custom_ticker else TICKER_MAP[selected_ticker]

enable_ratio = st.sidebar.checkbox("Compare with Asset 2?")
symbol_2 = None
if enable_ratio:
    symbol_2_key = st.sidebar.selectbox("Select Asset 2", list(TICKER_MAP.keys()), index=1)
    symbol_2 = TICKER_MAP[symbol_2_key]

if st.sidebar.button("RUN FULL ANALYSIS"):
    
    master_container = st.container()
    prog_bar = master_container.progress(0)
    status = master_container.empty()
    
    results_store = {}
    
    # --- AUTOMATIC LOOP THROUGH TIMEFRAMES ---
    total_steps = len(TIMEFRAME_CONFIG)
    
    for i, cfg in enumerate(TIMEFRAME_CONFIG):
        tf = cfg['interval']
        period = cfg['period']
        
        status.markdown(f"**Processing {tf} Timeframe ({period})...**")
        
        # Data Fetch
        df = fetch_ticker_data(symbol, period, tf)
        
        if df is not None and len(df) > 50:
            # Analysis
            analyzer = QuantAnalyzer(df, tf)
            stats = analyzer.get_bin_stats()
            patterns = analyzer.detect_patterns()
            signal, trend, mom, rsi_st = analyzer.generate_signal()
            backtest = run_smart_backtest(analyzer.df)
            
            results_store[tf] = {
                'data': analyzer.df,
                'stats': stats,
                'patterns': patterns,
                'signal': signal,
                'trend': trend,
                'backtest': backtest,
                'metrics': {
                    'price': df['Close'].iloc[-1],
                    'rsi': df['RSI'].iloc[-1],
                    'zscore': df['Z_Score'].iloc[-1],
                    'vol': df['Volatility'].iloc[-1]
                }
            }
        
        prog_bar.progress((i + 1) / total_steps)
    
    status.empty()
    prog_bar.empty()
    
    if not results_store:
        st.error("Failed to fetch sufficient data. Please check ticker or try again.")
        st.stop()

    # ==========================================
    # DISPLAY: EXECUTIVE SUMMARY
    # ==========================================
    st.markdown("## 🧠 AI Executive Summary")
    
    # Synthesize the text
    summary_text = ""
    bias_score = 0
    for tf, res in results_store.items():
        if res['signal'] == "BUY": bias_score += 1
        elif res['signal'] == "SELL": bias_score -= 1
        
    overall_bias = "STRONG BUY" if bias_score >= 2 else "STRONG SELL" if bias_score <= -2 else "NEUTRAL"
    color = "green" if "BUY" in overall_bias else "red" if "SELL" in overall_bias else "orange"
    
    st.markdown(f"""
    <div style='background-color: #1E1E1E; padding: 20px; border-left: 5px solid {color}; border-radius: 5px;'>
        <h3 style='margin:0; color: {color}'>{overall_bias} RECOMMENDATION</h3>
        <p>Market Analysis Report for <b>{symbol}</b> completed across {len(results_store)} timeframes.</p>
        <ul>
    """, unsafe_allow_html=True)
    
    cols = st.columns(len(results_store))
    for idx, (tf, res) in enumerate(results_store.items()):
        with cols[idx]:
            s_col = "green" if res['signal'] == "BUY" else "red" if res['signal'] == "SELL" else "grey"
            st.markdown(f"**{tf.upper()}**: <span style='color:{s_col}'>{res['signal']}</span>", unsafe_allow_html=True)
            st.caption(f"Trend: {res['trend']} | RSI: {res['metrics']['rsi']:.1f}")
            
            # Insight sentence
            if res['stats']:
                prob_dir = "Rise" if res['stats']['Bullish_Probability'] > 50 else "Fall"
                prob_val = max(res['stats']['Bullish_Probability'], res['stats']['Bearish_Probability'])
                st.info(f"Stats: {prob_val:.0f}% chance of {prob_dir} based on Z-Bin.")

    st.markdown("</div>", unsafe_allow_html=True)
    
    # ==========================================
    # DISPLAY: DEEP DIVE TABS
    # ==========================================
    
    tab_bins, tab_bt, tab_chart = st.tabs(["📊 Statistical Bins & Tables", "🧪 Backtesting Results", "📈 Technical Charts"])
    
    with tab_bins:
        tf_select = st.selectbox("Select Timeframe for Deep Dive", list(results_store.keys()))
        data = results_store[tf_select]
        df_curr = data['data']
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Z-Score Probability Matrix")
            st.markdown("Analyzing how price reacts when it enters specific statistical deviations (Z-Scores).")
            
            # Create a summary table of bins
            if 'Z_Bin' in df_curr.columns:
                bin_summary = df_curr.groupby('Z_Bin')['Fwd_Ret_Pct'].agg(['count', 'mean', lambda x: (x>0).mean()*100])
                bin_summary.columns = ['Occurrences', 'Avg Return %', 'Win Rate %']
                st.dataframe(bin_summary.style.background_gradient(cmap="RdYlGn", subset=['Avg Return %', 'Win Rate %']), use_container_width=True)
                
                curr_z = data['metrics']['zscore']
                st.caption(f"Current Z-Score: {curr_z:.2f} (Highlighted bin above)")

        with c2:
            st.subheader("Volatility Regimes")
            st.markdown("Impact of volatility on future returns.")
            try:
                df_curr['Vol_Bin'] = pd.qcut(df_curr['Volatility'], 4, labels=["Low Vol", "Normal", "High Vol", "Extreme"])
                vol_stats = df_curr.groupby('Vol_Bin')['Fwd_Ret_Pct'].agg(['mean', 'std'])
                vol_stats.columns = ['Avg Return', 'Risk (Std Dev)']
                st.dataframe(vol_stats, use_container_width=True)
            except:
                st.warning("Not enough data for Volatility Binning")
                
        st.markdown("---")
        st.subheader("Complete Data Table")
        st.dataframe(df_curr[['Close', 'RSI', 'Z_Score', 'Volatility', 'MACD', 'EMA_200']].sort_index(ascending=False).head(50), use_container_width=True)

    with tab_bt:
        st.subheader("Strategy Performance (MACD Trend Follower)")
        
        bt_cols = st.columns(len(results_store))
        
        for idx, (tf, res) in enumerate(results_store.items()):
            bt = res['backtest']
            with bt_cols[idx]:
                st.markdown(f"#### {tf}")
                if bt:
                    acc_color = "green" if bt['accuracy'] > 50 else "red"
                    st.markdown(f"**Accuracy**: :{acc_color}[{bt['accuracy']:.1f}%]")
                    st.markdown(f"**Trades**: {bt['trades']}")
                    st.markdown(f"**Total PnL**: {bt['pnl']:.2f}")
                    
                    with st.expander("Trade Log"):
                        st.dataframe(bt['log'], hide_index=True)
                else:
                    st.warning("No trades generated")

    with tab_chart:
        chart_tf = st.selectbox("Select Chart Timeframe", list(results_store.keys()), key="chart_tf")
        chart_df = results_store[chart_tf]['data']
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
        
        # Price
        fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_200'], line=dict(color='blue', width=2), name="EMA 200"), row=1, col=1)
        
        # Add Patterns
        pat = results_store[chart_tf]['patterns']
        fig.add_hline(y=pat['Nearest_Zone'], line_dash="dash", line_color="white", row=1, col=1, annotation_text="Strong Zone")
        
        # RSI
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['RSI'], line=dict(color='purple'), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_color="red", line_dash="dot", row=2, col=1)
        fig.add_hline(y=30, line_color="green", line_dash="dot", row=2, col=1)
        
        # Z-Score
        colors = ['red' if val > 2 else 'green' if val < -2 else 'gray' for val in chart_df['Z_Score']]
        fig.add_trace(go.Bar(x=chart_df.index, y=chart_df['Z_Score'], marker_color=colors, name="Z-Score"), row=3, col=1)
        
        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
