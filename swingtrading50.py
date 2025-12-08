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

# --- CONFIGURATION & SETUP ---
st.set_page_config(layout="wide", page_title="TradeFlow Pro AI", page_icon="📈")
warnings.filterwarnings('ignore')
IST = pytz.timezone('Asia/Kolkata')

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #fafafa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #262730; color: white; border: 1px solid #4b4b4b; }
    .stButton>button:hover { border-color: #00ff00; color: #00ff00; }
    .metric-card { background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #464e5f; }
    h1, h2, h3 { color: #00d4ff; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'data_cache' not in st.session_state: st.session_state.data_cache = {}
if 'paper_trades' not in st.session_state: st.session_state.paper_trades = []
if 'capital' not in st.session_state: st.session_state.capital = 100000.0
if 'live_monitoring' not in st.session_state: st.session_state.live_monitoring = False
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None

# --- ASSET MAPPING ---
ASSETS = {
    "NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN",
    "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD", "GOLD": "GC=F", "SILVER": "SI=F",
    "USD/INR": "INR=X", "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X",
    "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "INFY": "INFY.NS", "HDFC BANK": "HDFCBANK.NS"
}

# --- DATA FETCHING ---
def get_valid_period(interval):
    if interval == '1m': return '5d'
    if interval in ['5m', '15m', '30m']: return '1mo'
    if interval == '1h': return '1y'
    if interval == '1d': return '5y'
    return 'max'

@st.cache_data(ttl=300)
def fetch_data(ticker, interval, period):
    time.sleep(1.5)
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if data.empty: return None
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        data.index = data.index.tz_convert(IST)
        return data
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# --- TECHNICAL INDICATORS ---
class TechnicalAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        
    def add_all_indicators(self):
        df = self.df
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Moving Averages
        for p in [9, 20, 50, 100, 200]:
            df[f'SMA_{p}'] = close.rolling(window=p, min_periods=1).mean()
            df[f'EMA_{p}'] = close.ewm(span=p, adjust=False, min_periods=1).mean()
            
        # RSI (14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14, min_periods=1).mean()
        
        # ADX
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        tr_smooth = tr.rolling(14).sum()
        pdm_smooth = plus_dm.rolling(14).sum()
        mdm_smooth = minus_dm.rolling(14).sum()
        plus_di = 100 * (pdm_smooth / tr_smooth)
        minus_di = 100 * (mdm_smooth / tr_smooth)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(window=14).mean()
        
        # Z-Score & Volatility
        returns = close.pct_change()
        df['Z_Score'] = (returns - returns.rolling(20).mean()) / returns.rolling(20).std()
        df['Volatility'] = returns.rolling(20).std() * np.sqrt(252) * 100
        
        return df.fillna(0)

# --- ANALYSIS FUNCTIONS ---
def get_support_resistance(df, order=5):
    close_arr = df['Close'].values
    high_idx = signal.argrelextrema(close_arr, np.greater, order=order)[0]
    low_idx = signal.argrelextrema(close_arr, np.less, order=order)[0]
    
    levels = []
    for i in high_idx: levels.append({'type': 'Resistance', 'price': close_arr[i]})
    for i in low_idx: levels.append({'type': 'Support', 'price': close_arr[i]})
    
    if not levels: return []
    
    # Cluster levels
    df_levels = pd.DataFrame(levels).sort_values('price')
    clusters = []
    temp_cluster = [df_levels.iloc[0]]
    
    for i in range(1, len(df_levels)):
        prev = temp_cluster[-1]['price']
        curr = df_levels.iloc[i]['price']
        if abs(curr - prev) / prev <= 0.005:
            temp_cluster.append(df_levels.iloc[i])
        else:
            avg_price = np.mean([x['price'] for x in temp_cluster])
            clusters.append({
                'type': temp_cluster[0]['type'], 'price': avg_price, 
                'touches': len(temp_cluster),
                'distance_pct': (avg_price - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100
            })
            temp_cluster = [df_levels.iloc[i]]
            
    # Add last
    if temp_cluster:
        avg_price = np.mean([x['price'] for x in temp_cluster])
        clusters.append({
            'type': temp_cluster[0]['type'], 'price': avg_price, 
            'touches': len(temp_cluster),
            'distance_pct': (avg_price - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100
        })
        
    return sorted(clusters, key=lambda x: abs(x['distance_pct']))[:8]

def calculate_fibonacci(df):
    subset = df.iloc[-100:]
    swing_high = subset['High'].max()
    swing_low = subset['Low'].min()
    diff = swing_high - swing_low
    
    levels = {
        0.0: swing_high, 0.236: swing_high - 0.236*diff,
        0.382: swing_high - 0.382*diff, 0.5: swing_high - 0.5*diff,
        0.618: swing_high - 0.618*diff, 1.0: swing_low
    }
    closest = min(levels.items(), key=lambda x: abs(x[1] - df['Close'].iloc[-1]))
    return levels, closest

def check_divergence(df):
    if len(df) < 30: return "None", 0
    subset = df.iloc[-30:]
    price_lows = signal.argrelextrema(subset['Low'].values, np.less)[0]
    
    if len(price_lows) >= 2:
        l1, l2 = price_lows[-1], price_lows[-2]
        if subset['Low'].iloc[l1] < subset['Low'].iloc[l2]:
            if subset['RSI'].iloc[l1] > subset['RSI'].iloc[l2]:
                return "Bullish", round(subset['RSI'].iloc[l1] - subset['RSI'].iloc[l2], 2)
    return "None", 0

def find_historical_patterns(df):
    if len(df) < 200: return None
    curr = (df['Close'].iloc[-20:].values - df['Close'].iloc[-20:].mean()) / df['Close'].iloc[-20:].std()
    
    best_corr, best_idx = 0, 0
    for i in range(len(df) - 220):
        hist = df['Close'].iloc[i:i+20].values
        if np.std(hist) == 0: continue
        hist = (hist - hist.mean()) / hist.std()
        corr = np.corrcoef(curr, hist)[0, 1]
        if corr > best_corr: best_corr, best_idx = corr, i
            
    if best_corr > 0.85:
        return {
            'correlation': round(best_corr * 100, 2),
            'date': df.index[best_idx],
            'outcome_pct': round((df['Close'].iloc[best_idx+25] - df['Close'].iloc[best_idx+20])/df['Close'].iloc[best_idx+20]*100, 2)
        }
    return None

# --- VISUALIZATION (FIXED) ---
def create_charts(df, ticker, levels, fibs):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                        row_heights=[0.6, 0.2, 0.2], subplot_titles=(f"{ticker} Price", "RSI", "MACD"))

    # Panel 1: Price
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='orange'), name='EMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='blue'), name='EMA 50'), row=1, col=1)

    # Support/Resistance
    for level in levels:
        c = 'green' if level['type'] == 'Support' else 'red'
        fig.add_hline(y=level['price'], line_dash="dash", line_color=c, row=1, col=1)

    # Fibonacci (FIXED ITERATION)
    # fibs is a dictionary {0.0: price, 0.5: price...}
    for name, price in fibs.items():
        fig.add_hline(y=price, line_color="gray", opacity=0.3, row=1, col=1, annotation_text=f"Fib {name}")

    # Panel 2: RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_color="green", row=2, col=1)

    # Panel 3: MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue'), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='orange'), name='Signal'), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color='gray', name='Hist'), row=3, col=1)

    fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark")
    return fig

# --- BACKTESTING ---
def run_backtest(df):
    if len(df) < 100: return pd.DataFrame()
    trades = []
    in_pos, entry_p, entry_d, sl, tp = False, 0, None, 0, 0
    
    for i in range(50, len(df)):
        curr = df.iloc[i]
        if not in_pos:
            if curr['RSI'] < 30 and curr['Close'] > curr['EMA_20']:
                in_pos, entry_p, entry_d = True, curr['Close'], curr.name
                sl = entry_p - (2 * curr['ATR'])
                tp = entry_p + (3 * curr['ATR'])
        else:
            exit_p, reason = None, None
            if curr['Low'] <= sl: exit_p, reason = sl, "Stop Loss"
            elif curr['High'] >= tp: exit_p, reason = tp, "Target"
            elif curr['RSI'] > 70: exit_p, reason = curr['Close'], "RSI Overbought"
            elif (i - df.index.get_loc(entry_d)) > 20: exit_p, reason = curr['Close'], "Time Exit"
            
            if reason:
                trades.append({'Entry': entry_d, 'Exit': curr.name, 'PnL': exit_p - entry_p, 'Reason': reason})
                in_pos = False
    return pd.DataFrame(trades)

# --- MAIN APP ---
def main():
    st.title("🤖 TradeFlow Pro AI")
    
    with st.sidebar:
        st.header("Settings")
        ticker = st.selectbox("Asset", list(ASSETS.keys()))
        interval = st.selectbox("Interval", ["1m","5m","15m","1h","1d"], index=3)
        period = st.selectbox("Period", ["1d","5d","1mo","1y","5y"], index=3)
        
        if st.button("🚀 Run Analysis"):
            with st.spinner("Analyzing..."):
                df = fetch_data(ASSETS[ticker], interval, period)
                if df is not None:
                    an = TechnicalAnalyzer(df)
                    st.session_state.analysis_results = {
                        'df': an.add_all_indicators(), 'ticker': ticker, 'interval': interval
                    }
        
        st.markdown("---")
        monitor = st.toggle("Live Monitor (5s)")
        if monitor: st.session_state.live_monitoring = True
        else: st.session_state.live_monitoring = False

    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        df = res['df']
        current = df.iloc[-1]
        
        levels = get_support_resistance(df)
        fibs, fib_close = calculate_fibonacci(df)
        div_type, div_str = check_divergence(df)
        hist_pattern = find_historical_patterns(df)
        
        t1, t2, t3, t4, t5 = st.tabs(["Dashboard", "Charts", "AI Insights", "Backtest", "Paper Trade"])
        
        with t1:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"{current['Close']:.2f}", f"{df['Close'].pct_change().iloc[-1]*100:.2f}%")
            c2.metric("RSI", f"{current['RSI']:.2f}")
            c3.metric("Z-Score", f"{current['Z_Score']:.2f}")
            c4.metric("Volatility", f"{current['Volatility']:.2f}%")
            
            st.subheader("Support & Resistance")
            if levels:
                st.dataframe(pd.DataFrame(levels)[['type','price','touches','distance_pct']], hide_index=True)
            else:
                st.info("No clear levels found nearby.")
                
        with t2:
            st.plotly_chart(create_charts(df, res['ticker'], levels, fibs), use_container_width=True)
            
        with t3:
            st.info(f"💡 **AI Guidance:** Price is at {current['Close']:.2f}. "
                    f"Nearest Fibonacci level is {fib_close[0]} at {fib_close[1]:.2f}. "
                    f"Z-Score is {current['Z_Score']:.2f}. "
                    f"{'⚠️ Oversold!' if current['Z_Score'] < -2 else ''}")
            if hist_pattern:
                st.success(f"History Match: {hist_pattern['correlation']}% correlation with pattern on {hist_pattern['date']}.")
                
        with t4:
            bt = run_backtest(df)
            if not bt.empty:
                st.metric("Total PnL", f"{bt['PnL'].sum():.2f}")
                st.dataframe(bt)
            else:
                st.warning("No trades found in this period.")

        with t5:
            st.write(f"Capital: ₹{st.session_state.capital:.2f}")
            if st.button("Buy 1 Share"):
                st.session_state.paper_trades.append({'time': datetime.now(), 'price': current['Close']})
                st.session_state.capital -= current['Close']
                st.success("Bought!")
                
    if st.session_state.live_monitoring:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
