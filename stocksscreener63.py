import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Professional Stock Screener", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
    .sub-header {font-size: 1.5rem; font-weight: bold; color: #2ca02c; margin-top: 1rem;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;}
    .stTabs [data-baseweb="tab-list"] {gap: 2rem;}
    .stTabs [data-baseweb="tab"] {height: 3rem; white-space: pre-wrap; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'last_run_time' not in st.session_state:
    st.session_state.last_run_time = None
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []

# Stock universes
NIFTY_50 = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
            'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS',
            'BAJFINANCE.NS', 'HCLTECH.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS']

NIFTY_MIDCAP = ['ADANIENT.NS', 'GODREJCP.NS', 'PIIND.NS', 'BOSCHLTD.NS', 'HAVELLS.NS',
                'LUPIN.NS', 'GLAND.NS', 'BIOCON.NS', 'MUTHOOTFIN.NS', 'COFORGE.NS']

NIFTY_SMALLCAP = ['AFFLE.NS', 'ROUTE.NS', 'ANGELONE.NS', 'RVNL.NS', 'IRFC.NS',
                  'POLICYBZR.NS', 'KAYNES.NS', 'CAMS.NS', 'MANKIND.NS', 'RAINBOW.NS']

def flatten_multiindex_columns(df):
    """Flatten multi-index columns from yfinance"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in df.columns.values]
    return df

def fetch_stock_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch stock data with rate limiting and error handling"""
    try:
        time.sleep(1.2)  # Rate limiting
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            return None
            
        df = flatten_multiindex_columns(df)
        df.reset_index(inplace=True)
        
        # Ensure required columns exist
        required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return None
            
        return df
    except Exception as e:
        st.warning(f"Error fetching {symbol}: {str(e)}")
        return None

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD"""
    ema12 = calculate_ema(data, 12)
    ema26 = calculate_ema(data, 26)
    macd = ema12 - ema26
    signal = calculate_ema(macd, 9)
    histogram = macd - signal
    return macd, signal, histogram

def calculate_bollinger_bands(data: pd.Series, period: int = 20, std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    sma = calculate_sma(data, period)
    rolling_std = data.rolling(window=period).std()
    upper = sma + (rolling_std * std)
    lower = sma - (rolling_std * std)
    return upper, sma, lower

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index"""
    tr = calculate_atr(high, low, close, 1)
    
    up = high.diff()
    down = -low.diff()
    
    plus_dm = up.where((up > down) & (up > 0), 0)
    minus_dm = down.where((down > up) & (down > 0), 0)
    
    atr = calculate_atr(high, low, close, period)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On Balance Volume"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume Weighted Average Price"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    return (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

def detect_buildup(df: pd.DataFrame) -> Dict[str, float]:
    """Detect long/short buildup based on price and OI changes"""
    # Using volume as proxy for OI since OI data requires derivatives data
    recent_price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]) * 100
    recent_volume_change = ((df['Volume'].iloc[-5:].mean() - df['Volume'].iloc[-20:-5].mean()) / 
                           df['Volume'].iloc[-20:-5].mean()) * 100
    
    buildup_score = 0
    buildup_type = "Neutral"
    
    if recent_price_change > 1 and recent_volume_change > 20:
        buildup_type = "Long Buildup"
        buildup_score = min(recent_price_change + recent_volume_change/2, 100)
    elif recent_price_change < -1 and recent_volume_change > 20:
        buildup_type = "Short Buildup"
        buildup_score = min(abs(recent_price_change) + recent_volume_change/2, 100)
    elif recent_price_change > 1 and recent_volume_change < -20:
        buildup_type = "Long Unwinding"
        buildup_score = min(recent_price_change - recent_volume_change/2, 100)
    elif recent_price_change < -1 and recent_volume_change < -20:
        buildup_type = "Short Covering"
        buildup_score = min(abs(recent_price_change) - recent_volume_change/2, 100)
    
    return {
        'type': buildup_type,
        'score': buildup_score,
        'price_change': recent_price_change,
        'volume_change': recent_volume_change
    }

def calculate_technical_indicators(df: pd.DataFrame, symbol: str) -> Dict:
    """Calculate all technical indicators for a stock"""
    if df is None or len(df) < 200:
        return None
    
    try:
        latest = df.iloc[-1]
        
        # Moving Averages
        df['EMA9'] = calculate_ema(df['Close'], 9)
        df['EMA15'] = calculate_ema(df['Close'], 15)
        df['EMA20'] = calculate_ema(df['Close'], 20)
        df['EMA50'] = calculate_ema(df['Close'], 50)
        df['EMA100'] = calculate_ema(df['Close'], 100)
        df['EMA200'] = calculate_ema(df['Close'], 200)
        df['SMA20'] = calculate_sma(df['Close'], 20)
        df['SMA50'] = calculate_sma(df['Close'], 50)
        df['SMA200'] = calculate_sma(df['Close'], 200)
        
        # Momentum Indicators
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
        
        # Volatility
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
        df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
        
        # Trend
        df['ADX'] = calculate_adx(df['High'], df['Low'], df['Close'])
        
        # Volume
        df['OBV'] = calculate_obv(df['Close'], df['Volume'])
        df['VWAP'] = calculate_vwap(df)
        
        # Volume Analysis
        avg_volume_20 = df['Volume'].iloc[-20:].mean()
        volume_ratio = latest['Volume'] / avg_volume_20 if avg_volume_20 > 0 else 0
        
        # Buildup Detection
        buildup = detect_buildup(df)
        
        # EMA Crossover Analysis
        ema9_val = df['EMA9'].iloc[-1]
        ema15_val = df['EMA15'].iloc[-1]
        ema_diff = abs(ema9_val - ema15_val)
        ema_diff_pct = (ema_diff / ema15_val) * 100 if ema15_val > 0 else 100
        
        # Support and Resistance
        recent_high = df['High'].iloc[-20:].max()
        recent_low = df['Low'].iloc[-20:].min()
        
        # Calculate Stop Loss and Targets
        atr_value = df['ATR'].iloc[-1]
        entry_price = latest['Close']
        stop_loss = entry_price - (1.5 * atr_value)
        target1 = entry_price + (2 * atr_value)
        target2 = entry_price + (3 * atr_value)
        risk_reward = (target1 - entry_price) / (entry_price - stop_loss) if (entry_price - stop_loss) > 0 else 0
        
        # Trend Determination
        trend = "Neutral"
        if latest['Close'] > df['EMA50'].iloc[-1] > df['EMA200'].iloc[-1]:
            trend = "Strong Uptrend"
        elif latest['Close'] > df['EMA50'].iloc[-1]:
            trend = "Uptrend"
        elif latest['Close'] < df['EMA50'].iloc[-1] < df['EMA200'].iloc[-1]:
            trend = "Strong Downtrend"
        elif latest['Close'] < df['EMA50'].iloc[-1]:
            trend = "Downtrend"
        
        # Signal Generation
        signals = []
        signal_strength = 0
        
        # Bullish Signals
        if df['RSI'].iloc[-1] < 40 and df['RSI'].iloc[-1] > 30:
            signals.append("RSI Oversold Recovery")
            signal_strength += 15
        
        if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] and df['MACD'].iloc[-2] <= df['MACD_Signal'].iloc[-2]:
            signals.append("MACD Bullish Cross")
            signal_strength += 20
        
        if latest['Close'] > df['EMA20'].iloc[-1] and df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1]:
            signals.append("Price Above EMAs")
            signal_strength += 15
        
        if volume_ratio > 1.5:
            signals.append("High Volume")
            signal_strength += 10
        
        if df['ADX'].iloc[-1] > 25:
            signals.append("Strong Trend (ADX)")
            signal_strength += 10
        
        # Bearish Signals
        if df['RSI'].iloc[-1] > 70:
            signals.append("RSI Overbought")
            signal_strength -= 15
        
        if df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1] and df['MACD'].iloc[-2] >= df['MACD_Signal'].iloc[-2]:
            signals.append("MACD Bearish Cross")
            signal_strength -= 20
        
        return {
            'symbol': symbol.replace('.NS', ''),
            'price': entry_price,
            'change_pct': ((entry_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100,
            'volume': latest['Volume'],
            'volume_ratio': volume_ratio,
            'rsi': df['RSI'].iloc[-1],
            'macd': df['MACD'].iloc[-1],
            'macd_signal': df['MACD_Signal'].iloc[-1],
            'adx': df['ADX'].iloc[-1],
            'atr': atr_value,
            'ema9': ema9_val,
            'ema15': ema15_val,
            'ema20': df['EMA20'].iloc[-1],
            'ema50': df['EMA50'].iloc[-1],
            'ema200': df['EMA200'].iloc[-1],
            'sma20': df['SMA20'].iloc[-1],
            'sma50': df['SMA50'].iloc[-1],
            'sma200': df['SMA200'].iloc[-1],
            'vwap': df['VWAP'].iloc[-1],
            'bb_upper': df['BB_Upper'].iloc[-1],
            'bb_lower': df['BB_Lower'].iloc[-1],
            'support': recent_low,
            'resistance': recent_high,
            'trend': trend,
            'signals': signals,
            'signal_strength': signal_strength,
            'stop_loss': stop_loss,
            'target1': target1,
            'target2': target2,
            'risk_reward': risk_reward,
            'buildup_type': buildup['type'],
            'buildup_score': buildup['score'],
            'buildup_price_change': buildup['price_change'],
            'buildup_volume_change': buildup['volume_change'],
            'ema_diff_pct': ema_diff_pct,
            'ema_crossover_potential': ema_diff_pct < 0.5,
            'df': df  # Store for detailed analysis
        }
    except Exception as e:
        st.warning(f"Error calculating indicators for {symbol}: {str(e)}")
        return None

def analyze_stocks(stock_list: List[str], progress_bar, status_text) -> pd.DataFrame:
    """Analyze all stocks in the list"""
    results = []
    total = len(stock_list)
    
    for idx, symbol in enumerate(stock_list):
        status_text.text(f"Analyzing {symbol.replace('.NS', '')} ({idx+1}/{total})...")
        progress_bar.progress((idx + 1) / total)
        
        df = fetch_stock_data(symbol, period="1y", interval="1d")
        if df is not None:
            indicators = calculate_technical_indicators(df, symbol)
            if indicators:
                results.append(indicators)
    
    status_text.text("Analysis Complete!")
    return results

def create_buildup_chart(df: pd.DataFrame):
    """Create bar chart for buildup analysis"""
    # Filter stocks with significant buildup
    buildup_df = df[df['buildup_score'] > 20].copy()
    
    if buildup_df.empty:
        st.info("No significant buildup detected in current stocks")
        return
    
    # Sort by buildup score
    buildup_df = buildup_df.sort_values('buildup_score', ascending=True)
    
    # Create color mapping
    color_map = {
        'Long Buildup': '#00CC00',
        'Short Buildup': '#FF4444',
        'Long Unwinding': '#FFA500',
        'Short Covering': '#4169E1'
    }
    
    buildup_df['color'] = buildup_df['buildup_type'].map(color_map)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=buildup_df['symbol'],
        x=buildup_df['buildup_score'],
        orientation='h',
        marker=dict(color=buildup_df['color']),
        text=buildup_df['buildup_type'],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>' +
                      'Score: %{x:.1f}<br>' +
                      'Price Change: %{customdata[0]:.2f}%<br>' +
                      'Volume Change: %{customdata[1]:.2f}%<br>' +
                      '<extra></extra>',
        customdata=buildup_df[['buildup_price_change', 'buildup_volume_change']].values
    ))
    
    fig.update_layout(
        title="Long/Short Buildup Analysis",
        xaxis_title="Buildup Strength Score",
        yaxis_title="Stock",
        height=max(400, len(buildup_df) * 30),
        showlegend=False,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed table
    st.subheader("Buildup Details")
    display_df = buildup_df[['symbol', 'price', 'buildup_type', 'buildup_score', 
                             'buildup_price_change', 'buildup_volume_change']].copy()
    display_df.columns = ['Symbol', 'Price', 'Buildup Type', 'Score', 'Price Change %', 'Volume Change %']
    display_df = display_df.round(2)
    st.dataframe(display_df, use_container_width=True)

def display_ema_crossover_stocks(df: pd.DataFrame):
    """Display stocks with potential EMA crossover"""
    crossover_df = df[df['ema_crossover_potential'] == True].copy()
    
    if crossover_df.empty:
        st.info("No stocks showing imminent EMA crossover (9 & 15)")
        return
    
    crossover_df = crossover_df.sort_values('ema_diff_pct')
    
    st.subheader("🎯 Stocks with Potential EMA (9/15) Crossover")
    st.write("These stocks show EMA 9 and EMA 15 converging - potential breakout opportunities")
    
    # Create display dataframe
    display_df = crossover_df[['symbol', 'price', 'ema9', 'ema15', 'ema_diff_pct', 
                               'rsi', 'trend', 'signal_strength']].copy()
    
    # Add crossover direction
    display_df['crossover_direction'] = display_df.apply(
        lambda x: '↑ Bullish' if x['ema9'] > x['ema15'] else '↓ Bearish', axis=1
    )
    
    display_df.columns = ['Symbol', 'Price', 'EMA 9', 'EMA 15', 'Diff %', 
                         'RSI', 'Trend', 'Signal Strength', 'Direction']
    display_df = display_df.round(2)
    
    # Color code by direction
    def highlight_direction(row):
        if '↑' in str(row['Direction']):
            return ['background-color: #90EE90'] * len(row)
        elif '↓' in str(row['Direction']):
            return ['background-color: #FFB6C1'] * len(row)
        return [''] * len(row)
    
    st.dataframe(display_df.style.apply(highlight_direction, axis=1), use_container_width=True)
    
    # Visualization
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=crossover_df['symbol'],
        y=crossover_df['ema9'],
        mode='markers+lines',
        name='EMA 9',
        marker=dict(size=10, color='blue'),
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=crossover_df['symbol'],
        y=crossover_df['ema15'],
        mode='markers+lines',
        name='EMA 15',
        marker=dict(size=10, color='red'),
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=crossover_df['symbol'],
        y=crossover_df['price'],
        mode='markers',
        name='Current Price',
        marker=dict(size=12, color='green', symbol='diamond')
    ))
    
    fig.update_layout(
        title="EMA Convergence Visualization",
        xaxis_title="Stock",
        yaxis_title="Price",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Main App
st.markdown('<div class="main-header">📊 Professional Stock Screener</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Stock Universe Selection
    st.subheader("Select Universe")
    include_nifty50 = st.checkbox("Nifty 50", value=True)
    include_midcap = st.checkbox("Nifty Midcap", value=False)
    include_smallcap = st.checkbox("Nifty Smallcap", value=False)
    
    # Custom stocks
    st.subheader("Custom Stocks")
    custom_stocks_input = st.text_area(
        "Enter stock symbols (one per line, with .NS suffix)",
        placeholder="TATAMOTORS.NS\nWIPRO.NS\nTATASTEEL.NS"
    )
    
    # Trading Style
    st.subheader("Trading Style")
    trading_style = st.selectbox(
        "Select Style",
        ["Intraday", "Swing Trading", "Positional", "Long-term Investment"]
    )
    
    # Filters
    st.subheader("Filters")
    min_signal_strength = st.slider("Min Signal Strength", -50, 50, 0)
    min_volume_ratio = st.slider("Min Volume Ratio", 0.5, 3.0, 1.0, 0.1)
    
    st.markdown("---")
    
    # Run Analysis Button
    run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    if st.session_state.last_run_time:
        st.caption(f"Last run: {st.session_state.last_run_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Build stock list
stock_list = []
if include_nifty50:
    stock_list.extend(NIFTY_50)
if include_midcap:
    stock_list.extend(NIFTY_MIDCAP)
if include_smallcap:
    stock_list.extend(NIFTY_SMALLCAP)

# Add custom stocks
if custom_stocks_input:
    custom_list = [s.strip() for s in custom_stocks_input.split('\n') if s.strip()]
    stock_list.extend(custom_list)

# Remove duplicates
stock_list = list(set(stock_list))

# Run analysis when button is clicked
if run_button:
    if not stock_list:
        st.error("Please select at least one stock universe or add custom stocks")
    else:
        st.info(f"Analyzing {len(stock_list)} stocks... This may take a few minutes.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Fetching and analyzing data..."):
            results = analyze_stocks(stock_list, progress_bar, status_text)
            
            if results:
                st.session_state.analysis_data = pd.DataFrame(results)
                st.session_state.last_run_time = datetime.now()
                progress_bar.empty()
                status_text.empty()
                st.success(f"✅ Analysis complete! Found {len(results)} stocks with valid data.")
                st.rerun()
            else:
                st.error("No data could be retrieved. Please check your internet connection and try again.")

# Display results if analysis has been run
if st.session_state.analysis_data is not None:
    df_results = st.session_state.analysis_data
    
    # Apply filters
    df_filtered = df_results[
        (df_results['signal_strength'] >= min_signal_strength) &
        (df_results['volume_ratio'] >= min_volume_ratio)
    ].copy()
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Stocks", len(df_filtered))
    with col2:
        bullish = len(df_filtered[df_filtered['signal_strength'] > 20])
        st.metric("Bullish Signals", bullish)
    with col3:
        bearish = len(df_filtered[df_filtered['signal_strength'] < -20])
        st.metric("Bearish Signals", bearish)
    with col4:
        avg_rsi = df_filtered['rsi'].mean()
        st.metric("Avg RSI", f"{avg_rsi:.1f}")
    with col5:
        high_volume = len(df_filtered[df_filtered['volume_ratio'] > 1.5])
        st.metric("High Volume", high_volume)
    
    st.markdown("---")
    
    # Tabs for different views
    tabs = st.tabs([
        "📋 Overview",
        "🎯 Top Opportunities",
        "📈 Long/Short Buildup",
        "🔄 EMA Crossover Potential",
        "📊 Intraday Signals",
        "🌊 Swing Trading",
        "📍 Positional Trading",
        "💎 Long-term Investment",
        "🔍 Detailed Analysis"
    ])
    
    # Tab 1: Overview
    with tabs[0]:
        st.subheader("Market Overview")
        
        # Display main dataframe
        display_cols = ['symbol', 'price', 'change_pct', 'volume_ratio', 'rsi', 
                       'trend', 'signal_strength', 'risk_reward']
        overview_df = df_filtered[display_cols].copy()
        overview_df.columns = ['Symbol', 'Price', 'Change %', 'Vol Ratio', 'RSI', 
                              'Trend', 'Signal', 'R:R']
        overview_df = overview_df.sort_values('Signal', ascending=False)
        overview_df = overview_df.round(2)
        
        st.dataframe(overview_df, use_container_width=True, height=400)
        
        # Trend distribution
        col1, col2 = st.columns(2)
        
        with col1:
            trend_counts = df_filtered['trend'].value_counts()
            fig = px.pie(values=trend_counts.values, names=trend_counts.index, 
                        title="Trend Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df_filtered, x='signal_strength', nbins=30,
                             title="Signal Strength Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Top Opportunities
    with tabs[1]:
        st.subheader("🎯 Top Trading Opportunities")
        
        # Sort by signal strength and risk-reward
        top_stocks = df_filtered.sort_values('signal_strength', ascending=False).head(15)
        
        for idx, row in top_stocks.iterrows():
            with st.expander(f"🔥 {row['symbol']} - Signal Strength: {row['signal_strength']:.0f}"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"₹{row['price']:.2f}")
                    st.metric("Trend", row['trend'])
                
                with col2:
                    st.metric("Stop Loss", f"₹{row['stop_loss']:.2f}")
                    st.metric("RSI", f"{row['rsi']:.1f}")
                
                with col3:
                    st.metric("Target 1", f"₹{row['target1']:.2f}")
                    st.metric("Target 2", f"₹{row['target2']:.2f}")
                
                with col4:
                    st.metric("Risk:Reward", f"1:{row['risk_reward']:.2f}")
                    st.metric("Volume Ratio", f"{row['volume_ratio']:.2f}x")
                
                # Signals
                if row['signals']:
                    st.write("**Active Signals:**")
                    for signal in row['signals']:
                        st.write(f"• {signal}")
                
                # Key levels
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Support:**", f"₹{row['support']:.2f}")
                with col2:
                    st.write("**Resistance:**", f"₹{row['resistance']:.2f}")
    
    # Tab 3: Long/Short Buildup
    with tabs[2]:
        st.subheader("📈 Long/Short Buildup Analysis")
        st.write("Stocks showing significant price and volume changes indicating position buildup")
        create_buildup_chart(df_filtered)
    
    # Tab 4: EMA Crossover Potential
    with tabs[3]:
        display_ema_crossover_stocks(df_filtered)
    
    # Tab 5: Intraday Signals
    with tabs[4]:
        st.subheader("📊 Intraday Trading Signals")
        st.write("High volume breakouts and momentum plays for day trading")
        
        intraday_stocks = df_filtered[
            (df_filtered['volume_ratio'] > 1.5) &
            (df_filtered['signal_strength'] > 15) &
            (df_filtered['atr'] > 0)
        ].copy()
        
        intraday_stocks = intraday_stocks.sort_values('volume_ratio', ascending=False)
        
        if intraday_stocks.empty:
            st.info("No intraday signals found with current filters")
        else:
            display_cols = ['symbol', 'price', 'change_pct', 'volume_ratio', 'rsi', 
                          'vwap', 'atr', 'stop_loss', 'target1', 'risk_reward']
            intraday_df = intraday_stocks[display_cols].copy()
            intraday_df.columns = ['Symbol', 'Price', 'Change %', 'Vol Ratio', 'RSI', 
                                  'VWAP', 'ATR', 'Stop Loss', 'Target', 'R:R']
            intraday_df = intraday_df.round(2)
            st.dataframe(intraday_df, use_container_width=True)
            
            # VWAP Analysis
            st.subheader("VWAP Analysis")
            intraday_stocks['vwap_position'] = intraday_stocks.apply(
                lambda x: 'Above VWAP' if x['price'] > x['vwap'] else 'Below VWAP', axis=1
            )
            
            col1, col2 = st.columns(2)
            with col1:
                vwap_counts = intraday_stocks['vwap_position'].value_counts()
                fig = px.pie(values=vwap_counts.values, names=vwap_counts.index,
                           title="Price Position vs VWAP")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(intraday_stocks, x='volume_ratio', y='change_pct',
                               size='atr', color='rsi', hover_data=['symbol'],
                               title="Volume vs Price Change")
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 6: Swing Trading
    with tabs[5]:
        st.subheader("🌊 Swing Trading Opportunities")
        st.write("Stocks with pullback setups and trend continuation patterns")
        
        swing_stocks = df_filtered[
            (df_filtered['rsi'] > 40) & (df_filtered['rsi'] < 70) &
            (df_filtered['adx'] > 20) &
            (df_filtered['signal_strength'] > 10)
        ].copy()
        
        swing_stocks = swing_stocks.sort_values('signal_strength', ascending=False)
        
        if swing_stocks.empty:
            st.info("No swing trading signals found with current filters")
        else:
            display_cols = ['symbol', 'price', 'trend', 'rsi', 'adx', 'ema20', 'ema50',
                          'stop_loss', 'target1', 'target2', 'risk_reward']
            swing_df = swing_stocks[display_cols].copy()
            swing_df.columns = ['Symbol', 'Price', 'Trend', 'RSI', 'ADX', 'EMA20', 'EMA50',
                               'Stop Loss', 'Target 1', 'Target 2', 'R:R']
            swing_df = swing_df.round(2)
            st.dataframe(swing_df, use_container_width=True)
            
            # EMA Analysis for Swing
            st.subheader("Moving Average Analysis")
            swing_stocks['ma_position'] = swing_stocks.apply(
                lambda x: 'Bullish' if x['price'] > x['ema20'] > x['ema50'] 
                else 'Bearish' if x['price'] < x['ema20'] < x['ema50']
                else 'Neutral', axis=1
            )
            
            col1, col2 = st.columns(2)
            with col1:
                ma_counts = swing_stocks['ma_position'].value_counts()
                fig = px.bar(x=ma_counts.index, y=ma_counts.values,
                           title="MA Position Distribution",
                           labels={'x': 'Position', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(swing_stocks, x='rsi', y='adx',
                               size='signal_strength', color='trend',
                               hover_data=['symbol'],
                               title="RSI vs ADX (Trend Strength)")
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 7: Positional Trading
    with tabs[6]:
        st.subheader("📍 Positional Trading Setups")
        st.write("Strong trends with good fundamentals for medium-term holds")
        
        positional_stocks = df_filtered[
            (df_filtered['adx'] > 25) &
            ((df_filtered['trend'] == 'Strong Uptrend') | (df_filtered['trend'] == 'Uptrend')) &
            (df_filtered['signal_strength'] > 15)
        ].copy()
        
        positional_stocks = positional_stocks.sort_values('adx', ascending=False)
        
        if positional_stocks.empty:
            st.info("No positional trading signals found with current filters")
        else:
            display_cols = ['symbol', 'price', 'trend', 'adx', 'ema50', 'ema200',
                          'stop_loss', 'target1', 'target2', 'risk_reward', 'signal_strength']
            pos_df = positional_stocks[display_cols].copy()
            pos_df.columns = ['Symbol', 'Price', 'Trend', 'ADX', 'EMA50', 'EMA200',
                             'Stop Loss', 'Target 1', 'Target 2', 'R:R', 'Signal']
            pos_df = pos_df.round(2)
            st.dataframe(pos_df, use_container_width=True)
            
            # Trend Strength Analysis
            st.subheader("Trend Strength Indicators")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(positional_stocks.head(10), x='symbol', y='adx',
                           title="Top 10 Stocks by ADX (Trend Strength)",
                           color='adx', color_continuous_scale='Greens')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(positional_stocks, x='ema50', y='ema200',
                               size='signal_strength', color='adx',
                               hover_data=['symbol'],
                               title="EMA50 vs EMA200 Position")
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 8: Long-term Investment
    with tabs[7]:
        st.subheader("💎 Long-term Investment Opportunities")
        st.write("Quality stocks trading above long-term averages")
        
        longterm_stocks = df_filtered[
            (df_filtered['price'] > df_filtered['ema200']) &
            (df_filtered['ema50'] > df_filtered['ema200']) &
            (df_filtered['trend'].isin(['Strong Uptrend', 'Uptrend']))
        ].copy()
        
        longterm_stocks = longterm_stocks.sort_values('signal_strength', ascending=False)
        
        if longterm_stocks.empty:
            st.info("No long-term investment signals found with current filters")
        else:
            display_cols = ['symbol', 'price', 'trend', 'ema50', 'ema200', 'sma200',
                          'rsi', 'adx', 'support', 'resistance']
            lt_df = longterm_stocks[display_cols].copy()
            lt_df.columns = ['Symbol', 'Price', 'Trend', 'EMA50', 'EMA200', 'SMA200',
                            'RSI', 'ADX', 'Support', 'Resistance']
            lt_df = lt_df.round(2)
            st.dataframe(lt_df, use_container_width=True)
            
            # Long-term Health Check
            st.subheader("Long-term Trend Health")
            
            longterm_stocks['ema_alignment'] = (
                (longterm_stocks['price'] > longterm_stocks['ema50']) &
                (longterm_stocks['ema50'] > longterm_stocks['ema200'])
            )
            
            col1, col2 = st.columns(2)
            with col1:
                aligned_count = longterm_stocks['ema_alignment'].sum()
                total_count = len(longterm_stocks)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=aligned_count,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "EMA Aligned Stocks"},
                    delta={'reference': total_count},
                    gauge={'axis': {'range': [None, total_count]},
                          'bar': {'color': "darkgreen"},
                          'steps': [
                              {'range': [0, total_count/2], 'color': "lightgray"},
                              {'range': [total_count/2, total_count], 'color': "gray"}],
                          'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': total_count}}))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(longterm_stocks, x='ema200', y='price',
                               size='adx', color='rsi',
                               hover_data=['symbol'],
                               title="Price vs 200 EMA")
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 9: Detailed Analysis
    with tabs[8]:
        st.subheader("🔍 Detailed Stock Analysis")
        
        # Stock selector
        selected_symbol = st.selectbox(
            "Select Stock for Detailed Analysis",
            options=df_filtered['symbol'].tolist()
        )
        
        if selected_symbol:
            stock_data = df_filtered[df_filtered['symbol'] == selected_symbol].iloc[0]
            stock_df = stock_data['df']
            
            # Stock header
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Price", f"₹{stock_data['price']:.2f}", 
                         f"{stock_data['change_pct']:.2f}%")
            with col2:
                st.metric("RSI", f"{stock_data['rsi']:.1f}")
            with col3:
                st.metric("ADX", f"{stock_data['adx']:.1f}")
            with col4:
                st.metric("Volume Ratio", f"{stock_data['volume_ratio']:.2f}x")
            with col5:
                st.metric("Signal", f"{stock_data['signal_strength']:.0f}")
            
            # Price Chart
            st.subheader("Price Chart with Indicators")
            
            fig = go.Figure()
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=stock_df.index[-100:],
                open=stock_df['Open'][-100:],
                high=stock_df['High'][-100:],
                low=stock_df['Low'][-100:],
                close=stock_df['Close'][-100:],
                name='Price'
            ))
            
            # EMAs
            fig.add_trace(go.Scatter(x=stock_df.index[-100:], y=stock_df['EMA20'][-100:],
                                    mode='lines', name='EMA 20', line=dict(color='blue', width=1)))
            fig.add_trace(go.Scatter(x=stock_df.index[-100:], y=stock_df['EMA50'][-100:],
                                    mode='lines', name='EMA 50', line=dict(color='orange', width=1)))
            fig.add_trace(go.Scatter(x=stock_df.index[-100:], y=stock_df['EMA200'][-100:],
                                    mode='lines', name='EMA 200', line=dict(color='red', width=1)))
            
            # Bollinger Bands
            fig.add_trace(go.Scatter(x=stock_df.index[-100:], y=stock_df['BB_Upper'][-100:],
                                    mode='lines', name='BB Upper', 
                                    line=dict(color='gray', width=1, dash='dash')))
            fig.add_trace(go.Scatter(x=stock_df.index[-100:], y=stock_df['BB_Lower'][-100:],
                                    mode='lines', name='BB Lower',
                                    line=dict(color='gray', width=1, dash='dash'),
                                    fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))
            
            fig.update_layout(
                title=f"{selected_symbol} - Technical Chart",
                yaxis_title="Price (₹)",
                xaxis_title="Date",
                height=600,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Indicator Panels
            col1, col2 = st.columns(2)
            
            with col1:
                # RSI Chart
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=stock_df.index[-100:], y=stock_df['RSI'][-100:],
                                            mode='lines', name='RSI', line=dict(color='purple', width=2)))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.update_layout(title="RSI (14)", yaxis_title="RSI", height=300)
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col2:
                # MACD Chart
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=stock_df.index[-100:], y=stock_df['MACD'][-100:],
                                             mode='lines', name='MACD', line=dict(color='blue', width=2)))
                fig_macd.add_trace(go.Scatter(x=stock_df.index[-100:], y=stock_df['MACD_Signal'][-100:],
                                             mode='lines', name='Signal', line=dict(color='red', width=2)))
                fig_macd.add_trace(go.Bar(x=stock_df.index[-100:], y=stock_df['MACD_Hist'][-100:],
                                         name='Histogram', marker_color='gray'))
                fig_macd.update_layout(title="MACD", yaxis_title="MACD", height=300)
                st.plotly_chart(fig_macd, use_container_width=True)
            
            # Volume Chart
            fig_vol = go.Figure()
            colors = ['red' if stock_df['Close'].iloc[i] < stock_df['Open'].iloc[i] 
                     else 'green' for i in range(len(stock_df[-100:]))]
            fig_vol.add_trace(go.Bar(x=stock_df.index[-100:], y=stock_df['Volume'][-100:],
                                    name='Volume', marker_color=colors))
            avg_vol = stock_df['Volume'][-100:].mean()
            fig_vol.add_hline(y=avg_vol, line_dash="dash", line_color="orange",
                            annotation_text="Avg Volume")
            fig_vol.update_layout(title="Volume Analysis", yaxis_title="Volume", height=300)
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # Key Levels and Signals
            st.subheader("Key Levels & Trading Plan")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📍 Support & Resistance**")
                st.write(f"Resistance: ₹{stock_data['resistance']:.2f}")
                st.write(f"Current: ₹{stock_data['price']:.2f}")
                st.write(f"Support: ₹{stock_data['support']:.2f}")
                st.write(f"VWAP: ₹{stock_data['vwap']:.2f}")
            
            with col2:
                st.markdown("**🎯 Trading Levels**")
                st.write(f"Entry: ₹{stock_data['price']:.2f}")
                st.write(f"Stop Loss: ₹{stock_data['stop_loss']:.2f}")
                st.write(f"Target 1: ₹{stock_data['target1']:.2f}")
                st.write(f"Target 2: ₹{stock_data['target2']:.2f}")
            
            with col3:
                st.markdown("**📊 Risk Management**")
                st.write(f"Risk: ₹{abs(stock_data['price'] - stock_data['stop_loss']):.2f}")
                st.write(f"Reward: ₹{abs(stock_data['target1'] - stock_data['price']):.2f}")
                st.write(f"R:R Ratio: 1:{stock_data['risk_reward']:.2f}")
                st.write(f"ATR: ₹{stock_data['atr']:.2f}")
            
            # Active Signals
            st.subheader("🚨 Active Signals")
            if stock_data['signals']:
                for signal in stock_data['signals']:
                    st.success(f"✓ {signal}")
            else:
                st.info("No active signals at the moment")
            
            # Technical Summary
            st.subheader("📈 Technical Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Moving Averages**")
                st.write(f"EMA 9: ₹{stock_data['ema9']:.2f}")
                st.write(f"EMA 15: ₹{stock_data['ema15']:.2f}")
                st.write(f"EMA 20: ₹{stock_data['ema20']:.2f}")
                st.write(f"EMA 50: ₹{stock_data['ema50']:.2f}")
                st.write(f"EMA 200: ₹{stock_data['ema200']:.2f}")
            
            with col2:
                st.markdown("**Momentum & Volatility**")
                st.write(f"RSI: {stock_data['rsi']:.1f}")
                st.write(f"MACD: {stock_data['macd']:.2f}")
                st.write(f"ADX: {stock_data['adx']:.1f}")
                st.write(f"BB Upper: ₹{stock_data['bb_upper']:.2f}")
                st.write(f"BB Lower: ₹{stock_data['bb_lower']:.2f}")
            
            # Overall Assessment
            st.subheader("💡 Overall Assessment")
            st.write(f"**Trend:** {stock_data['trend']}")
            st.write(f"**Signal Strength:** {stock_data['signal_strength']:.0f}")
            st.write(f"**Buildup Type:** {stock_data['buildup_type']}")
            
            if stock_data['signal_strength'] > 30:
                st.success("🟢 Strong Buy Signal - Multiple confirmations")
            elif stock_data['signal_strength'] > 15:
                st.info("🔵 Moderate Buy Signal - Some confirmations")
            elif stock_data['signal_strength'] < -30:
                st.error("🔴 Strong Sell Signal - Multiple bearish indicators")
            elif stock_data['signal_strength'] < -15:
                st.warning("🟠 Moderate Sell Signal - Some bearish signs")
            else:
                st.info("⚪ Neutral - Wait for clearer signals")

else:
    st.info("👆 Configure your preferences in the sidebar and click 'Run Analysis' to start scanning stocks.")
    
    # Display sample information
    st.markdown("---")
    st.markdown("### 📚 Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Technical Analysis**
        - 15+ Technical Indicators
        - Multi-timeframe Analysis
        - Chart Pattern Detection
        - Support/Resistance Levels
        - Volume Analysis
        """)
    
    with col2:
        st.markdown("""
        **Trading Strategies**
        - Intraday Signals
        - Swing Trading Setups
        - Positional Opportunities
        - Long-term Investment Ideas
        - Risk-Reward Calculation
        """)
    
    with col3:
        st.markdown("""
        **Advanced Features**
        - Long/Short Buildup Detection
        - EMA Crossover Alerts
        - Signal Strength Scoring
        - Professional Entry/Exit Levels
        - Comprehensive Screening
        """)

# Footer
st.markdown("---")
st.caption("⚠️ Disclaimer: This tool is for educational purposes only. Always do your own research before making investment decisions.")
st.caption("📊 Data provided by Yahoo Finance with rate-limited requests for stability.")
