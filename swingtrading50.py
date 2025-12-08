import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats, signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from datetime import datetime, timedelta
import time
import warnings

# --- CONFIGURATION & SETUP ---
st.set_page_config(layout="wide", page_title="TradeFlow Pro AI", page_icon="📈")
warnings.filterwarnings('ignore')
IST = pytz.timezone('Asia/Kolkata')

# Custom CSS for Professional UI
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #fafafa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #262730; color: white; border: 1px solid #4b4b4b; }
    .stButton>button:hover { border-color: #00ff00; color: #00ff00; }
    .metric-card { background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #464e5f; }
    h1, h2, h3 { color: #00d4ff; }
    .bullish { color: #00ff00; font-weight: bold; }
    .bearish { color: #ff4b4b; font-weight: bold; }
    .neutral { color: #ffa500; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'data_cache' not in st.session_state: st.session_state.data_cache = {}
if 'paper_trades' not in st.session_state: st.session_state.paper_trades = []
if 'capital' not in st.session_state: st.session_state.capital = 100000.0
if 'live_monitoring' not in st.session_state: st.session_state.live_monitoring = False
if 'last_refresh' not in st.session_state: st.session_state.last_refresh = time.time()
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None

# --- ASSET MAPPING ---
ASSETS = {
    "NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN",
    "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD", "GOLD": "GC=F", "SILVER": "SI=F",
    "USD/INR": "INR=X", "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X",
    "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "INFY": "INFY.NS", "HDFC BANK": "HDFCBANK.NS"
}

# --- 1. DATA FETCHING & MANAGEMENT ---

def get_valid_period(interval):
    """Enforces yfinance interval/period limitations."""
    if interval == '1m': return '5d'
    if interval in ['5m', '15m', '30m']: return '1mo'
    if interval == '1h': return '1y'
    if interval == '1d': return '5y'
    return 'max'

@st.cache_data(ttl=300)
def fetch_data(ticker, interval, period):
    """Fetches data with rate limiting and error handling."""
    time.sleep(1.5)  # Rate limiting
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        # Handle MultiIndex columns (yfinance update)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        if data.empty:
            return None
            
        # Timezone Conversion
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        data.index = data.index.tz_convert(IST)
        
        return data
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# --- 2. TECHNICAL INDICATORS (MANUAL CALCULATION) ---

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
        
        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands (20, 2)
        sma20 = close.rolling(window=20, min_periods=1).mean()
        std20 = close.rolling(window=20, min_periods=1).std()
        df['BB_Upper'] = sma20 + (std20 * 2)
        df['BB_Lower'] = sma20 - (std20 * 2)
        
        # ATR (14)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14, min_periods=1).mean()
        
        # ADX (14)
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0  # Should be positive logic for calculation
        minus_dm = abs(minus_dm)
        
        tr_smooth = tr.rolling(14).sum()
        pdm_smooth = plus_dm.rolling(14).sum()
        mdm_smooth = minus_dm.rolling(14).sum()
        
        plus_di = 100 * (pdm_smooth / tr_smooth)
        minus_di = 100 * (mdm_smooth / tr_smooth)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(window=14).mean()
        
        # Z-Score (Returns)
        returns = close.pct_change()
        df['Z_Score'] = (returns - returns.rolling(20).mean()) / returns.rolling(20).std()
        
        # Volatility (Historical)
        df['Volatility'] = returns.rolling(20).std() * np.sqrt(252) * 100
        
        return df.fillna(0) # Basic fill for initial NaNs

# --- 3, 4, 5, 6, 7. ADVANCED ANALYSIS FUNCTIONS ---

def get_support_resistance(df, order=5):
    """Finds local minima (support) and maxima (resistance)."""
    # Using iloc to get numpy array for argrelextrema
    close_arr = df['Close'].values
    
    # Find local peaks/valleys
    high_idx = signal.argrelextrema(close_arr, np.greater, order=order)[0]
    low_idx = signal.argrelextrema(close_arr, np.less, order=order)[0]
    
    levels = []
    
    # Process Resistance
    for i in high_idx:
        levels.append({'type': 'Resistance', 'price': close_arr[i], 'date': df.index[i]})
        
    # Process Support
    for i in low_idx:
        levels.append({'type': 'Support', 'price': close_arr[i], 'date': df.index[i]})
        
    # Cluster levels within 0.5%
    df_levels = pd.DataFrame(levels)
    if df_levels.empty: return []
    
    clusters = []
    current_price = df['Close'].iloc[-1]
    
    # Sort by price to cluster
    df_levels = df_levels.sort_values('price')
    
    if not df_levels.empty:
        temp_cluster = [df_levels.iloc[0]]
        
        for i in range(1, len(df_levels)):
            prev = temp_cluster[-1]['price']
            curr = df_levels.iloc[i]['price']
            
            if abs(curr - prev) / prev <= 0.005: # 0.5% tolerance
                temp_cluster.append(df_levels.iloc[i])
            else:
                # Process completed cluster
                avg_price = np.mean([x['price'] for x in temp_cluster])
                touches = len(temp_cluster)
                dist_pct = (avg_price - current_price) / current_price * 100
                strength = "Strong" if touches >= 3 else "Moderate"
                l_type = temp_cluster[0]['type'] # Simplification
                clusters.append({
                    'type': l_type, 'price': avg_price, 'touches': touches,
                    'strength': strength, 'distance_pct': dist_pct
                })
                temp_cluster = [df_levels.iloc[i]]
                
        # Add last cluster
        if temp_cluster:
            avg_price = np.mean([x['price'] for x in temp_cluster])
            touches = len(temp_cluster)
            dist_pct = (avg_price - current_price) / current_price * 100
            strength = "Strong" if touches >= 3 else "Moderate"
            clusters.append({
                'type': temp_cluster[0]['type'], 'price': avg_price, 
                'touches': touches, 'strength': strength, 'distance_pct': dist_pct
            })
            
    # Return top 8 sorted by absolute distance
    clusters.sort(key=lambda x: abs(x['distance_pct']))
    return clusters[:8]

def calculate_fibonacci(df):
    """Calculates Fib levels based on last 100 candles."""
    subset = df.iloc[-100:]
    swing_high = subset['High'].max()
    swing_low = subset['Low'].min()
    diff = swing_high - swing_low
    
    levels = {
        0.0: swing_high,
        0.236: swing_high - 0.236 * diff,
        0.382: swing_high - 0.382 * diff,
        0.5: swing_high - 0.5 * diff,
        0.618: swing_high - 0.618 * diff,
        0.786: swing_high - 0.786 * diff,
        1.0: swing_low
    }
    
    current_price = df['Close'].iloc[-1]
    closest_level = min(levels.items(), key=lambda x: abs(x[1] - current_price))
    
    return levels, closest_level

def check_divergence(df):
    """Detects simple RSI divergence (last 20 candles)."""
    # Simplified Logic:
    # Bullish: Price Low < Prev Price Low AND RSI Low > Prev RSI Low
    # Bearish: Price High > Prev Price High AND RSI High < Prev RSI High
    
    if len(df) < 30: return "None", 0
    
    subset = df.iloc[-30:]
    
    # Find local minima/maxima in subset
    price_lows = signal.argrelextrema(subset['Low'].values, np.less)[0]
    price_highs = signal.argrelextrema(subset['High'].values, np.greater)[0]
    
    div_type = "None"
    strength = 0
    
    if len(price_lows) >= 2:
        last_low_idx = price_lows[-1]
        prev_low_idx = price_lows[-2]
        
        if subset['Low'].iloc[last_low_idx] < subset['Low'].iloc[prev_low_idx]:
            if subset['RSI'].iloc[last_low_idx] > subset['RSI'].iloc[prev_low_idx]:
                div_type = "Bullish"
                strength = subset['RSI'].iloc[last_low_idx] - subset['RSI'].iloc[prev_low_idx] # Crude strength
                
    if len(price_highs) >= 2:
        last_high_idx = price_highs[-1]
        prev_high_idx = price_highs[-2]
        
        if subset['High'].iloc[last_high_idx] > subset['High'].iloc[prev_high_idx]:
            if subset['RSI'].iloc[last_high_idx] < subset['RSI'].iloc[prev_high_idx]:
                div_type = "Bearish"
                strength = subset['RSI'].iloc[prev_high_idx] - subset['RSI'].iloc[last_high_idx]

    return div_type, round(abs(strength), 2)

def find_historical_patterns(df):
    """Finds correlation patterns (Last 20 candles vs History)."""
    if len(df) < 200: return None
    
    current_pattern = df['Close'].iloc[-20:].values
    # Normalize
    current_pattern = (current_pattern - np.mean(current_pattern)) / np.std(current_pattern)
    
    best_corr = 0
    best_idx = 0
    
    # Look back
    for i in range(len(df) - 220): # Avoid overlapping exactly with current
        hist_pattern = df['Close'].iloc[i:i+20].values
        if np.std(hist_pattern) == 0: continue
        
        hist_pattern = (hist_pattern - np.mean(hist_pattern)) / np.std(hist_pattern)
        
        corr = np.corrcoef(current_pattern, hist_pattern)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_idx = i
            
    if best_corr > 0.85:
        # Determine outcome after pattern
        future_price = df['Close'].iloc[best_idx + 20 + 5] # 5 candles later
        past_price = df['Close'].iloc[best_idx + 20]
        change = (future_price - past_price) / past_price * 100
        return {
            'correlation': round(best_corr * 100, 2),
            'date': df.index[best_idx],
            'outcome_pct': round(change, 2),
            'candles_later': 5
        }
    return None

# --- 11. VISUALIZATION ---

def create_charts(df, ticker, levels, fibs):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=(f"{ticker} Price Action", "RSI", "MACD"))

    # Price & EMAs
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='orange', width=1), name='EMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='blue', width=1), name='EMA 50'), row=1, col=1)

    # Support/Resistance Lines
    for level in levels:
        color = 'green' if level['type'] == 'Support' else 'red'
        fig.add_hline(y=level['price'], line_dash="dash", line_color=color, row=1, col=1, annotation_text=f"{level['type']} ({level['touches']}x)")

    # Fibonacci
    for name, price in fibs[0].items():
        fig.add_hline(y=price, line_color="gray", opacity=0.3, row=1, col=1, annotation_text=f"Fib {name}")

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue'), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='orange'), name='Signal'), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color='gray', name='Hist'), row=3, col=1)

    fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark")
    return fig

# --- 13. BACKTESTING ---

def run_backtest(df):
    """Backtest: RSI < 30 & Price > EMA 20 (Mean Reversion)."""
    trades = []
    in_position = False
    entry_price = 0
    entry_date = None
    stop_loss = 0
    target = 0
    
    # Needs at least 100 rows
    if len(df) < 100: return pd.DataFrame()

    for i in range(50, len(df)):
        current = df.iloc[i]
        
        if not in_position:
            # Entry Condition
            if current['RSI'] < 30 and current['Close'] > current['EMA_20']:
                in_position = True
                entry_price = current['Close']
                entry_date = current.name
                atr = current['ATR']
                stop_loss = entry_price - (2 * atr)
                target = entry_price + (3 * atr) # 1:1.5 RR
        else:
            # Exit Conditions
            exit_reason = None
            if current['Low'] <= stop_loss:
                exit_price = stop_loss
                exit_reason = "Stop Loss"
            elif current['High'] >= target:
                exit_price = target
                exit_reason = "Target Hit"
            elif current['RSI'] > 70:
                exit_price = current['Close']
                exit_reason = "RSI Overbought"
            elif (i - df.index.get_loc(entry_date)) > 20: # Time exit
                exit_price = current['Close']
                exit_reason = "Time Exit"
                
            if exit_reason:
                pnl = exit_price - entry_price
                pnl_pct = (pnl / entry_price) * 100
                trades.append({
                    'Entry Date': entry_date, 'Entry Price': entry_price,
                    'Exit Date': current.name, 'Exit Price': exit_price,
                    'Reason': exit_reason, 'PnL': pnl, 'PnL %': pnl_pct
                })
                in_position = False
                
    return pd.DataFrame(trades)

# --- MAIN UI LAYOUT ---

def main():
    st.title("🤖 TradeFlow Pro AI - Algorithmic Analysis Dashboard")
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Asset Selection
        asset_type = st.radio("Asset Class", ["Indices", "Crypto", "Forex", "Stocks"])
        
        if asset_type == "Indices":
            ticker_options = ["NIFTY 50", "BANK NIFTY", "SENSEX"]
        elif asset_type == "Crypto":
            ticker_options = ["BTC-USD", "ETH-USD"]
        elif asset_type == "Forex":
            ticker_options = ["USD/INR", "EUR/USD", "GBP/USD", "GOLD", "SILVER"]
        else:
            ticker_options = ["RELIANCE", "TCS", "INFY", "HDFC BANK"]
            custom_ticker = st.text_input("Or Custom Ticker (e.g., TSLA)")
            if custom_ticker: ticker_options = [custom_ticker]

        selected_ticker_label = st.selectbox("Select Asset", ticker_options)
        selected_ticker = ASSETS.get(selected_ticker_label, selected_ticker_label)
        
        # Timeframe
        col1, col2 = st.columns(2)
        with col1:
            interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d", "1wk"], index=3)
        with col2:
            period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"], index=5)
            
        # Ratio Analysis Toggle
        enable_ratio = st.checkbox("Enable Ratio Analysis")
        ratio_ticker = None
        if enable_ratio:
            ratio_ticker_label = st.selectbox("Compare With", [t for t in ASSETS.keys() if t != selected_ticker_label])
            ratio_ticker = ASSETS.get(ratio_ticker_label)

        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner(f"Fetching data for {selected_ticker}..."):
                # Validate period
                valid_p = get_valid_period(interval)
                if period == "max": period = valid_p # Fallback
                
                df = fetch_data(selected_ticker, interval, period)
                
                if df is not None:
                    analyzer = TechnicalAnalyzer(df)
                    df_analyzed = analyzer.add_all_indicators()
                    st.session_state.analysis_results = {
                        'df': df_analyzed,
                        'ticker': selected_ticker_label,
                        'interval': interval,
                        'ratio_data': None
                    }
                    
                    # Handle Ratio
                    if enable_ratio and ratio_ticker:
                        df2 = fetch_data(ratio_ticker, interval, period)
                        if df2 is not None:
                            # Align data
                            common = df.index.intersection(df2.index)
                            if not common.empty:
                                df1_c = df.loc[common]['Close']
                                df2_c = df2.loc[common]['Close']
                                ratio_series = df1_c / df2_c
                                st.session_state.analysis_results['ratio_data'] = {
                                    'ticker2': ratio_ticker_label,
                                    'series': ratio_series
                                }
                            else:
                                st.warning("Timestamps do not align for ratio analysis.")
                                
        # Paper Trading Controls
        st.markdown("---")
        st.header("💼 Paper Trading")
        st.metric("Capital", f"₹{st.session_state.capital:,.2f}")
        
        monitor = st.toggle("🔴 Live Monitoring (5s Refresh)")
        if monitor:
            st.session_state.live_monitoring = True
        else:
            st.session_state.live_monitoring = False

    # --- MAIN CONTENT AREA ---
    
    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        df = res['df']
        current = df.iloc[-1]
        
        # Calculate Advanced Metrics
        levels = get_support_resistance(df)
        fibs, fib_close = calculate_fibonacci(df)
        div_type, div_str = check_divergence(df)
        hist_pattern = find_historical_patterns(df)
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "📈 Charts", "🧠 AI Analysis", "🔙 Backtest", "📝 Paper Trade"])
        
        with tab1:
            # Top Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Price", f"₹{current['Close']:.2f}", f"{df['Close'].pct_change().iloc[-1]*100:.2f}%")
            c2.metric("RSI (14)", f"{current['RSI']:.2f}", delta=None, delta_color="off")
            
            z_score = current['Z_Score']
            z_col = "normal"
            if z_score < -2: z_col = "inverse" # Oversold
            elif z_score > 2: z_col = "normal" # Overbought
            c3.metric("Z-Score", f"{z_score:.2f}", "Mean Reversion" if abs(z_score)>2 else "Neutral", delta_color=z_col)
            
            c4.metric("Volatility", f"{current['Volatility']:.2f}%")

            # Signal Strength
            st.subheader("Signal Strength Analysis")
            
            signals = []
            if current['RSI'] < 30: signals.append("✅ RSI Oversold (Bullish)")
            elif current['RSI'] > 70: signals.append("❌ RSI Overbought (Bearish)")
            
            if current['Close'] > current['EMA_50']: signals.append("✅ Price > EMA 50 (Uptrend)")
            else: signals.append("❌ Price < EMA 50 (Downtrend)")
            
            if current['ADX'] > 25: signals.append(f"✅ Strong Trend (ADX {current['ADX']:.1f})")
            
            if div_type == "Bullish": signals.append(f"✅ Bullish Divergence ({div_str}%)")
            
            # Recommendation Logic
            score = 0
            if current['Close'] > current['EMA_20']: score += 1
            if current['RSI'] < 30: score += 2
            if current['MACD'] > current['MACD_Signal']: score += 1
            if z_score < -2: score += 2
            
            rec = "HOLD 🟡"
            if score >= 3: rec = "BUY 🟢"
            elif score <= -1: rec = "SELL 🔴" # Simplified logic
            
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.info(f"Final Recommendation: **{rec}**")
                st.write("**What's Working:**")
                for s in signals:
                    if "✅" in s: st.write(s)
                st.write("**Risks:**")
                for s in signals:
                    if "❌" in s: st.write(s)
                    
            with col_b:
                st.write("#### Nearest Levels")
                lvl_df = pd.DataFrame(levels)
                if not lvl_df.empty:
                    st.dataframe(lvl_df[['type', 'price', 'touches', 'distance_pct']], hide_index=True)
                else:
                    st.write("No strong levels found nearby.")

        with tab2:
            fig = create_charts(df, res['ticker'], levels, fibs)
            st.plotly_chart(fig, use_container_width=True)
            
            if res['ratio_data']:
                st.subheader(f"Ratio: {res['ticker']} / {res['ratio_data']['ticker2']}")
                st.line_chart(res['ratio_data']['series'])

        with tab3:
            st.markdown("### 🤖 AI Friend Guidance")
            
            guidance = f"""
            **Hey there! Let's look at {res['ticker']} on the {res['interval']} timeframe.**
            
            Right now, the price is **₹{current['Close']:.2f}**.
            """
            
            if z_score < -2:
                guidance += f"\n\n**Opportunity Alert:** The Z-Score is **{z_score:.2f}**, which is quite low! Historically, this suggests the market is oversold and might bounce back (mean reversion). Be on the lookout for a reversal pattern."
            elif z_score > 2:
                guidance += f"\n\n**Caution:** The Z-Score is **{z_score:.2f}**, suggesting things are a bit heated (overbought). Don't FOMO in right now."
            
            guidance += f"\n\n**Fibonacci Check:** We are currently closest to the **Fib {fib_close[0]}** level at ₹{fib_close[1]:.2f}. Keep an eye on price action here."
            
            if hist_pattern:
                guidance += f"\n\n🔮 **History Repeats?** I found a pattern from {hist_pattern['date']} that looks **{hist_pattern['correlation']}%** similar to now. Back then, the price moved **{hist_pattern['outcome_pct']}%** in the next few candles."
            
            st.info(guidance)
            
            st.subheader("Data Table")
            st.dataframe(df.tail(10).sort_index(ascending=False))

        with tab4:
            st.subheader(f"Backtest Results: Mean Reversion ({res['interval']})")
            bt_results = run_backtest(df)
            
            if not bt_results.empty:
                b1, b2, b3 = st.columns(3)
                win_trades = bt_results[bt_results['PnL'] > 0]
                win_rate = len(win_trades) / len(bt_results) * 100
                total_pnl = bt_results['PnL'].sum()
                
                b1.metric("Total Trades", len(bt_results))
                b2.metric("Win Rate", f"{win_rate:.1f}%")
                b3.metric("Total PnL Points", f"{total_pnl:.2f}")
                
                st.dataframe(bt_results)
                
                # CSV Export
                csv = bt_results.to_csv().encode('utf-8')
                st.download_button("Download CSV", csv, "backtest_results.csv", "text/csv")
            else:
                st.warning("Not enough trades generated with current strategy parameters.")

        with tab5:
            st.subheader("Simulated Trading Floor")
            
            c_entry, c_action = st.columns([3, 1])
            with c_entry:
                st.write(f"Current Price: **₹{current['Close']:.2f}** | Available Capital: **₹{st.session_state.capital:,.2f}**")
            
            with c_action:
                qty = max(1, int((st.session_state.capital * 0.1) / current['Close']))
                if st.button(f"BUY {qty} Qty"):
                    trade = {
                        'entry_time': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S'),
                        'symbol': res['ticker'],
                        'price': current['Close'],
                        'qty': qty,
                        'sl': current['Close'] * 0.99,
                        'target': current['Close'] * 1.02,
                        'entry_rsi': current['RSI']
                    }
                    st.session_state.paper_trades.append(trade)
                    st.session_state.capital -= (current['Close'] * qty)
                    st.success(f"Executed BUY for {qty} shares at {current['Close']}")

            # Active Positions
            if st.session_state.paper_trades:
                st.write("### Active Positions")
                active_df = pd.DataFrame(st.session_state.paper_trades)
                
                # Update current price in table logic would go here in a real DB app
                # For now, we compare against the currently analyzed ticker if matches
                if not active_df.empty:
                    active_df['Current Price'] = active_df.apply(
                        lambda x: current['Close'] if x['symbol'] == res['ticker'] else x['price'], axis=1
                    )
                    active_df['Unrealized P&L'] = (active_df['Current Price'] - active_df['price']) * active_df['qty']
                    
                    st.table(active_df[['entry_time', 'symbol', 'price', 'qty', 'Current Price', 'Unrealized P&L']])

            # Auto-Refresh Logic (The "Live" part)
            if st.session_state.live_monitoring:
                time.sleep(5)
                st.rerun()

if __name__ == "__main__":
    main()
