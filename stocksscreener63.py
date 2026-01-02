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
    .stTabs [data-baseweb="tab-list"] {gap: 1rem;}
    .stTabs [data-baseweb="tab"] {
        height: auto;
        white-space: pre-wrap;
        font-weight: 600;
        padding: 10px 20px;
        font-size: 14px;
    }
    .timeframe-badge {
        background-color: #e1f5ff;
        padding: 5px 15px;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'all_stocks_data' not in st.session_state:
    st.session_state.all_stocks_data = None
if 'last_run_time' not in st.session_state:
    st.session_state.last_run_time = None
if 'timeframe_info' not in st.session_state:
    st.session_state.timeframe_info = {'period': '1y', 'interval': '1d'}

# Complete Stock universes
NIFTY_50 = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
    'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS',
    'BAJFINANCE.NS', 'HCLTECH.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS',
    'WIPRO.NS', 'ADANIENT.NS', 'TATAMOTORS.NS', 'ONGC.NS', 'NTPC.NS',
    'POWERGRID.NS', 'M&M.NS', 'JSWSTEEL.NS', 'TATASTEEL.NS', 'INDUSINDBK.NS',
    'COALINDIA.NS', 'BAJAJFINSV.NS', 'HDFCLIFE.NS', 'SBILIFE.NS', 'TECHM.NS',
    'EICHERMOT.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'BRITANNIA.NS', 'DIVISLAB.NS',
    'CIPLA.NS', 'DRREDDY.NS', 'APOLLOHOSP.NS', 'HINDALCO.NS', 'ADANIPORTS.NS',
    'GRASIM.NS', 'BPCL.NS', 'TRENT.NS', 'SHRIRAMFIN.NS', 'LTIM.NS'
]

NIFTY_MIDCAP = ['ADANIENT.NS', 'GODREJCP.NS', 'PIIND.NS', 'BOSCHLTD.NS', 'HAVELLS.NS',
                'LUPIN.NS', 'GLAND.NS', 'BIOCON.NS', 'MUTHOOTFIN.NS', 'COFORGE.NS',
                'TORNTPHARM.NS', 'BANDHANBNK.NS', 'COLPAL.NS', 'MARICO.NS', 'BERGEPAINT.NS']

NIFTY_SMALLCAP = ['AFFLE.NS', 'ROUTE.NS', 'ANGELONE.NS', 'RVNL.NS', 'IRFC.NS',
                  'POLICYBZR.NS', 'KAYNES.NS', 'CAMS.NS', 'MANKIND.NS', 'RAINBOW.NS',
                  'TANLA.NS', 'CLEAN.NS', 'HAPPSTMNDS.NS', 'NAZARA.NS', 'ZOMATO.NS']

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

def calculate_ema_angle(ema_series: pd.Series, lookback: int = 5) -> float:
    """Calculate the angle of EMA in degrees"""
    if len(ema_series) < lookback + 1:
        return 0
    
    recent_values = ema_series.iloc[-lookback:].values
    x = np.arange(len(recent_values))
    
    # Linear regression to find slope
    slope = np.polyfit(x, recent_values, 1)[0]
    
    # Convert slope to angle
    angle = np.degrees(np.arctan(slope / recent_values[-1] * 100))
    return angle

def classify_candle(open_price: float, close_price: float, high: float, low: float) -> Dict:
    """Classify candle strength and type"""
    body = abs(close_price - open_price)
    total_range = high - low
    
    if total_range == 0:
        return {'type': 'Doji', 'strength': 'Neutral', 'body_pct': 0}
    
    body_pct = (body / total_range) * 100
    
    if body_pct < 10:
        candle_type = 'Doji'
        strength = 'Neutral'
    elif close_price > open_price:
        if body_pct > 70:
            candle_type = 'Bullish Marubozu'
            strength = 'Strong Bullish'
        elif body_pct > 50:
            strength = 'Bullish'
            candle_type = 'Bullish'
        else:
            strength = 'Weak Bullish'
            candle_type = 'Weak Bullish'
    else:
        if body_pct > 70:
            candle_type = 'Bearish Marubozu'
            strength = 'Strong Bearish'
        elif body_pct > 50:
            strength = 'Bearish'
            candle_type = 'Bearish'
        else:
            strength = 'Weak Bearish'
            candle_type = 'Weak Bearish'
    
    return {
        'type': candle_type,
        'strength': strength,
        'body_pct': body_pct,
        'body_size': body
    }

def detect_buildup(df: pd.DataFrame) -> Dict[str, float]:
    """Detect long/short buildup based on price, volume and RSI changes"""
    if len(df) < 20:
        return {
            'type': 'Insufficient Data',
            'score': 0,
            'price_change': 0,
            'volume_change': 0,
            'avg_volume_5d': 0,
            'avg_volume_20d': 0,
            'rsi_change': 0,
            'oi_change': 0  # Placeholder
        }
    
    # Price changes
    price_5d_ago = df['Close'].iloc[-6]
    current_price = df['Close'].iloc[-1]
    recent_price_change = ((current_price - price_5d_ago) / price_5d_ago) * 100
    
    # Volume changes
    avg_volume_5d = df['Volume'].iloc[-5:].mean()
    avg_volume_20d = df['Volume'].iloc[-20:-5].mean()
    recent_volume_change = ((avg_volume_5d - avg_volume_20d) / avg_volume_20d) * 100 if avg_volume_20d > 0 else 0
    
    # RSI change
    rsi_5d_ago = df['RSI'].iloc[-6] if 'RSI' in df.columns and len(df) > 6 else 50
    current_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
    rsi_change = current_rsi - rsi_5d_ago
    
    # OI change (using volume as proxy since we don't have actual OI data for stocks)
    oi_change = recent_volume_change  # Placeholder
    
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
        'volume_change': recent_volume_change,
        'avg_volume_5d': avg_volume_5d,
        'avg_volume_20d': avg_volume_20d,
        'rsi_change': rsi_change,
        'oi_change': oi_change
    }

def calculate_technical_indicators(df: pd.DataFrame, symbol: str) -> Dict:
    """Calculate all technical indicators for a stock"""
    if df is None or len(df) < 200:
        return {
            'symbol': symbol.replace('.NS', ''),
            'status': 'Insufficient Data',
            'reason': f'Only {len(df) if df is not None else 0} candles available. Need at least 200 for proper analysis.',
            'data_available': False
        }
    
    try:
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
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
        
        # EMA Angles
        ema9_angle = calculate_ema_angle(df['EMA9'])
        ema15_angle = calculate_ema_angle(df['EMA15'])
        
        # Candle Analysis
        candle_info = classify_candle(latest['Open'], latest['Close'], latest['High'], latest['Low'])
        
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
        
        # Signal Generation and Missing Criteria
        signals = []
        missing_criteria = []
        signal_strength = 0
        
        # Bullish Signals
        if df['RSI'].iloc[-1] < 40 and df['RSI'].iloc[-1] > 30:
            signals.append("RSI Oversold Recovery")
            signal_strength += 15
        elif df['RSI'].iloc[-1] <= 30:
            missing_criteria.append("RSI too oversold (<30) - wait for recovery")
        elif df['RSI'].iloc[-1] >= 70:
            missing_criteria.append("RSI overbought (>70) - risk of reversal")
        
        if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] and df['MACD'].iloc[-2] <= df['MACD_Signal'].iloc[-2]:
            signals.append("MACD Bullish Cross")
            signal_strength += 20
        elif df['MACD'].iloc[-1] <= df['MACD_Signal'].iloc[-1]:
            missing_criteria.append("MACD below signal line - no momentum")
        
        if latest['Close'] > df['EMA20'].iloc[-1] and df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1]:
            signals.append("Price Above EMAs")
            signal_strength += 15
        else:
            missing_criteria.append("Price not aligned above EMAs - weak trend")
        
        if volume_ratio > 1.5:
            signals.append("High Volume")
            signal_strength += 10
        else:
            missing_criteria.append(f"Volume ratio low ({volume_ratio:.2f}x) - need >1.5x")
        
        if df['ADX'].iloc[-1] > 25:
            signals.append("Strong Trend (ADX)")
            signal_strength += 10
        else:
            missing_criteria.append(f"ADX weak ({df['ADX'].iloc[-1]:.1f}) - need >25 for strong trend")
        
        # Bearish Signals
        if df['RSI'].iloc[-1] > 70:
            signals.append("RSI Overbought")
            signal_strength -= 15
        
        if df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1] and df['MACD'].iloc[-2] >= df['MACD_Signal'].iloc[-2]:
            signals.append("MACD Bearish Cross")
            signal_strength -= 20
        
        # Risk assessment
        if risk_reward < 1.5:
            missing_criteria.append(f"Poor R:R ratio ({risk_reward:.2f}) - need >1.5")
        
        return {
            'symbol': symbol.replace('.NS', ''),
            'price': entry_price,
            'change_pct': ((entry_price - previous['Close']) / previous['Close']) * 100,
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
            'missing_criteria': missing_criteria,
            'signal_strength': signal_strength,
            'stop_loss': stop_loss,
            'target1': target1,
            'target2': target2,
            'risk_reward': risk_reward,
            'buildup_type': buildup['type'],
            'buildup_score': buildup['score'],
            'buildup_price_change': buildup['price_change'],
            'buildup_volume_change': buildup['volume_change'],
            'buildup_avg_volume_5d': buildup['avg_volume_5d'],
            'buildup_avg_volume_20d': buildup['avg_volume_20d'],
            'buildup_rsi_change': buildup['rsi_change'],
            'buildup_oi_change': buildup['oi_change'],
            'ema_diff_pct': ema_diff_pct,
            'ema9_angle': ema9_angle,
            'ema15_angle': ema15_angle,
            'ema_crossover_potential': ema_diff_pct < 0.5,
            'candle_type': candle_info['type'],
            'candle_strength': candle_info['strength'],
            'candle_body_pct': candle_info['body_pct'],
            'data_available': True,
            'df': df
        }
    except Exception as e:
        return {
            'symbol': symbol.replace('.NS', ''),
            'status': 'Error',
            'reason': f'Error calculating indicators: {str(e)}',
            'data_available': False
        }

def analyze_stocks(stock_list: List[str], progress_bar, status_text) -> Tuple[List, List]:
    """Analyze all stocks in the list"""
    successful_results = []
    all_results = []
    total = len(stock_list)
    
    for idx, symbol in enumerate(stock_list):
        status_text.text(f"Analyzing {symbol.replace('.NS', '')} ({idx+1}/{total})...")
        progress_bar.progress((idx + 1) / total)
        
        df = fetch_stock_data(symbol, period=st.session_state.timeframe_info['period'], 
                             interval=st.session_state.timeframe_info['interval'])
        indicators = calculate_technical_indicators(df, symbol)
        all_results.append(indicators)
        
        if indicators.get('data_available', False):
            successful_results.append(indicators)
    
    status_text.text("Analysis Complete!")
    return successful_results, all_results

def create_enhanced_buildup_chart(df: pd.DataFrame):
    """Create enhanced horizontal bar chart for buildup analysis"""
    buildup_df = df[df['buildup_score'] > 20].copy()
    
    if buildup_df.empty:
        st.info("No significant buildup detected in current stocks")
        return
    
    buildup_df = buildup_df.sort_values('buildup_score', ascending=True)
    
    color_map = {
        'Long Buildup': '#00CC00',
        'Short Buildup': '#FF4444',
        'Long Unwinding': '#FFA500',
        'Short Covering': '#4169E1'
    }
    
    buildup_df['color'] = buildup_df['buildup_type'].map(color_map)
    
    # Create subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=['Buildup Strength Score']
    )
    
    fig.add_trace(go.Bar(
        y=buildup_df['symbol'],
        x=buildup_df['buildup_score'],
        orientation='h',
        marker=dict(color=buildup_df['color']),
        text=buildup_df['buildup_type'],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>' +
                      'Score: %{x:.1f}<br>' +
                      'Type: %{text}<br>' +
                      '<extra></extra>',
        name='Buildup Score'
    ))
    
    fig.update_layout(
        title="Long/Short Buildup Strength Analysis",
        xaxis_title="Buildup Score",
        yaxis_title="Stock",
        height=max(400, len(buildup_df) * 35),
        showlegend=False,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("📊 Detailed Buildup Metrics")
    
    display_df = buildup_df[[
        'symbol', 'price', 'buildup_type', 'buildup_score',
        'buildup_price_change', 'buildup_volume_change',
        'buildup_avg_volume_5d', 'buildup_avg_volume_20d',
        'buildup_rsi_change', 'buildup_oi_change', 'rsi'
    ]].copy()
    
    display_df.columns = [
        'Symbol', 'Price (₹)', 'Buildup Type', 'Score',
        'Price Change %', 'Volume Change %',
        'Avg Vol (5D)', 'Avg Vol (20D)',
        'RSI Change', 'OI Change %', 'Current RSI'
    ]
    
    display_df = display_df.round(2)
    
    # Format large numbers
    for col in ['Avg Vol (5D)', 'Avg Vol (20D)']:
        display_df[col] = display_df[col].apply(lambda x: f"{x/1000:.0f}K" if x < 1000000 else f"{x/1000000:.2f}M")
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Individual metric charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_price = go.Figure()
        fig_price.add_trace(go.Bar(
            y=buildup_df['symbol'],
            x=buildup_df['buildup_price_change'],
            orientation='h',
            marker=dict(color=buildup_df['buildup_price_change'],
                       colorscale='RdYlGn',
                       showscale=True),
            name='Price Change %'
        ))
        fig_price.update_layout(
            title="Price Change % (5-day)",
            xaxis_title="Change %",
            height=max(300, len(buildup_df) * 25)
        )
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            y=buildup_df['symbol'],
            x=buildup_df['buildup_volume_change'],
            orientation='h',
            marker=dict(color=buildup_df['buildup_volume_change'],
                       colorscale='Blues',
                       showscale=True),
            name='Volume Change %'
        ))
        fig_vol.update_layout(
            title="Volume Change % (5D vs 20D Avg)",
            xaxis_title="Change %",
            height=max(300, len(buildup_df) * 25)
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    
    # RSI Change chart
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Bar(
        y=buildup_df['symbol'],
        x=buildup_df['buildup_rsi_change'],
        orientation='h',
        marker=dict(color=buildup_df['buildup_rsi_change'],
                   colorscale='RdYlGn',
                   showscale=True),
        name='RSI Change'
    ))
    fig_rsi.update_layout(
        title="RSI Change (5-day)",
        xaxis_title="RSI Change",
        height=max(300, len(buildup_df) * 25)
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

def display_ema_crossover_stocks(df: pd.DataFrame):
    """Display stocks with potential EMA crossover with enhanced details"""
    crossover_df = df[df['ema_crossover_potential'] == True].copy()
    
    if crossover_df.empty:
        st.info("No stocks showing imminent EMA crossover (9 & 15)")
        return
    
    crossover_df = crossover_df.sort_values('ema_diff_pct')
    
    st.subheader("🎯 Stocks with Potential EMA (9/15) Crossover")
    st.write("These stocks show EMA 9 and EMA 15 converging - potential breakout opportunities")
    
    # Create detailed display dataframe
    display_df = crossover_df[[
        'symbol', 'price', 'ema9', 'ema15', 'ema_diff_pct',
        'ema9_angle', 'ema15_angle', 'candle_strength',
        'rsi', 'trend', 'signal_strength'
    ]].copy()
    
    # Add crossover direction
    display_df['crossover_direction'] = display_df.apply(
        lambda x: '↑ Bullish' if x['ema9'] > x['ema15'] else '↓ Bearish', axis=1
    )
    
    # Calculate angle difference
    display_df['angle_convergence'] = abs(display_df['ema9_angle'] - display_df['ema15_angle'])
    
    display_df.columns = [
        'Symbol', 'Price', 'EMA 9', 'EMA 15', 'Diff %',
        'EMA9 Angle°', 'EMA15 Angle°', 'Candle Type',
        'RSI', 'Trend', 'Signal', 'Direction', 'Angle Conv°'
    ]
    display_df = display_df.round(2)
    
    # Color code by direction
    def highlight_direction(row):
        if '↑' in str(row['Direction']):
            return ['background-color: #90EE90'] * len(row)
        elif '↓' in str(row['Direction']):
            return ['background-color: #FFB6C1'] * len(row)
        return [''] * len(row)
    
    st.dataframe(display_df.style.apply(highlight_direction, axis=1), use_container_width=True, height=400)
    
    # EMA Values and Angles Visualization
    st.subheader("📐 EMA Values & Angle Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=crossover_df['symbol'],
            y=crossover_df['ema9'],
            mode='markers+lines',
            name='EMA 9',
            marker=dict(size=12, color='blue'),
            line=dict(color='blue', width=2),
            text=crossover_df['ema9_angle'].apply(lambda x: f"Angle: {x:.1f}°"),
            hovertemplate='<b>%{x}</b><br>EMA 9: ₹%{y:.2f}<br>%{text}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=crossover_df['symbol'],
            y=crossover_df['ema15'],
            mode='markers+lines',
            name='EMA 15',
            marker=dict(size=12, color='red'),
            line=dict(color='red', width=2),
            text=crossover_df['ema15_angle'].apply(lambda x: f"Angle: {x:.1f}°"),
            hovertemplate='<b>%{x}</b><br>EMA 15: ₹%{y:.2f}<br>%{text}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=crossover_df['symbol'],
            y=crossover_df['price'],
            mode='markers',
            name='Current Price',
            marker=dict(size=14, color='green', symbol='diamond')
        ))
        
        fig.update_layout(
            title="EMA Convergence - Price vs EMAs",
            xaxis_title="Stock",
            yaxis_title="Price (₹)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig_angle = go.Figure()
        
        fig_angle.add_trace(go.Bar(
            x=crossover_df['symbol'],
            y=crossover_df['ema9_angle'],
            name='EMA 9 Angle',
            marker_color='blue',
            text=crossover_df['ema9_angle'].apply(lambda x: f"{x:.1f}°"),
            textposition='outside'
        ))
        
        fig_angle.add_trace(go.Bar(
            x=crossover_df['symbol'],
            y=crossover_df['ema15_angle'],
            name='EMA 15 Angle',
            marker_color='red',
            text=crossover_df['ema15_angle'].apply(lambda x: f"{x:.1f}°"),
            textposition='outside'
        ))
        
        fig_angle.update_layout(
            title="EMA Slope Angles (Trend Direction)",
            xaxis_title="Stock",
            yaxis_title="Angle (degrees)",
            height=500,
            barmode='group'
        )
        
        st.plotly_chart(fig_angle, use_container_width=True)
    
    # Candle Strength Analysis
    st.subheader("🕯️ Current Candle Strength")
    
    candle_colors = {
        'Strong Bullish': '#006400',
        'Bullish': '#90EE90',
        'Weak Bullish': '#D3F9D8',
        'Strong Bearish': '#8B0000',
        'Bearish': '#FFB6C1',
        'Weak Bearish': '#FFE4E1',
        'Neutral': '#D3D3D3'
    }
    
    crossover_df['candle_color'] = crossover_df['candle_strength'].map(candle_colors)
    
    fig_candle = go.Figure()
    
    fig_candle.add_trace(go.Bar(
        x=crossover_df['symbol'],
        y=[1] * len(crossover_df),
        marker=dict(color=crossover_df['candle_color']),
        text=crossover_df['candle_strength'],
        textposition='inside',
        hovertemplate='<b>%{x}</b><br>Candle: %{text}<br>Type: ' + 
                      crossover_df['candle_type'].astype(str) + '<extra></extra>',
        showlegend=False
    ))
    
    fig_candle.update_layout(
        title="Current Candle Strength Classification",
        xaxis_title="Stock",
        yaxis_title="",
        height=300,
        yaxis_visible=False
    )
    
    st.plotly_chart(fig_candle, use_container_width=True)

def display_rejected_stocks(all_stocks_df: List[Dict]):
    """Display stocks that didn't pass screening with reasons"""
    st.subheader("🔍 All Analyzed Stocks - Complete Analysis")
    
    # Separate successful and failed analyses
    failed_stocks = [s for s in all_stocks_df if not s.get('data_available', False)]
    successful_stocks = [s for s in all_stocks_df if s.get('data_available', False)]
    
    # Display failed stocks
    if failed_stocks:
        st.markdown("### ❌ Stocks with Data Issues")
        failed_df = pd.DataFrame(failed_stocks)
        failed_display = failed_df[['symbol', 'status', 'reason']].copy()
        failed_display.columns = ['Symbol', 'Status', 'Reason']
        st.dataframe(failed_display, use_container_width=True)
        st.markdown("---")
    
    # Display successful stocks with missing criteria
    if successful_stocks:
        st.markdown("### 📋 All Screened Stocks with Analysis")
        
        stocks_df = pd.DataFrame(successful_stocks)
        
        # Create comprehensive display
        for idx, row in stocks_df.iterrows():
            with st.expander(f"{'✅' if row['signal_strength'] > 15 else '⚠️'} {row['symbol']} - Signal: {row['signal_strength']:.0f} | Trend: {row['trend']}"):
                
                # Metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Price", f"₹{row['price']:.2f}", f"{row['change_pct']:.2f}%")
                with col2:
                    st.metric("RSI", f"{row['rsi']:.1f}")
                with col3:
                    st.metric("ADX", f"{row['adx']:.1f}")
                with col4:
                    st.metric("Volume Ratio", f"{row['volume_ratio']:.2f}x")
                with col5:
                    st.metric("R:R", f"1:{row['risk_reward']:.2f}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**✅ Active Signals:**")
                    if row['signals']:
                        for signal in row['signals']:
                            st.success(f"✓ {signal}")
                    else:
                        st.info("No active signals")
                
                with col2:
                    st.markdown("**❌ Missing Criteria for Trade:**")
                    if row['missing_criteria']:
                        for criteria in row['missing_criteria']:
                            st.warning(f"⚠ {criteria}")
                    else:
                        st.success("All criteria met! ✅")
                
                # Additional details
                st.markdown("**📊 Technical Details:**")
                detail_col1, detail_col2, detail_col3 = st.columns(3)
                
                with detail_col1:
                    st.write(f"**EMA 9:** ₹{row['ema9']:.2f} (Angle: {row['ema9_angle']:.1f}°)")
                    st.write(f"**EMA 15:** ₹{row['ema15']:.2f} (Angle: {row['ema15_angle']:.1f}°)")
                    st.write(f"**EMA 20:** ₹{row['ema20']:.2f}")
                
                with detail_col2:
                    st.write(f"**Support:** ₹{row['support']:.2f}")
                    st.write(f"**Resistance:** ₹{row['resistance']:.2f}")
                    st.write(f"**VWAP:** ₹{row['vwap']:.2f}")
                
                with detail_col3:
                    st.write(f"**Candle:** {row['candle_strength']}")
                    st.write(f"**Buildup:** {row['buildup_type']}")
                    st.write(f"**Stop Loss:** ₹{row['stop_loss']:.2f}")

def display_timeframe_badge(period: str, interval: str):
    """Display current timeframe being analyzed"""
    timeframe_map = {
        '1d': 'Daily',
        '1h': 'Hourly',
        '15m': '15-Min',
        '5m': '5-Min',
        '1m': '1-Min'
    }
    
    period_map = {
        '1d': '1 Day',
        '5d': '5 Days',
        '1mo': '1 Month',
        '3mo': '3 Months',
        '6mo': '6 Months',
        '1y': '1 Year',
        '2y': '2 Years',
        '5y': '5 Years'
    }
    
    interval_display = timeframe_map.get(interval, interval)
    period_display = period_map.get(period, period)
    
    st.markdown(f"""
    <div style='background-color: #e1f5ff; padding: 10px; border-radius: 10px; text-align: center; margin: 10px 0;'>
        <span style='font-size: 16px; font-weight: bold; color: #0066cc;'>
            📊 Timeframe: {interval_display} | Period: {period_display}
        </span>
    </div>
    """, unsafe_allow_html=True)

# Main App
st.markdown('<div class="main-header">📊 Professional Stock Screener</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Timeframe Selection
    st.subheader("📈 Timeframe Settings")
    
    trading_style = st.selectbox(
        "Select Trading Style",
        ["Intraday", "Swing Trading", "Positional", "Long-term Investment"]
    )
    
    # Set default timeframes based on trading style
    if trading_style == "Intraday":
        default_period = "5d"
        default_interval = "15m"
    elif trading_style == "Swing Trading":
        default_period = "3mo"
        default_interval = "1d"
    elif trading_style == "Positional":
        default_period = "1y"
        default_interval = "1d"
    else:  # Long-term
        default_period = "5y"
        default_interval = "1d"
    
    period = st.selectbox(
        "Historical Period",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"].index(default_period)
    )
    
    interval = st.selectbox(
        "Candle Interval",
        ["1m", "5m", "15m", "1h", "1d"],
        index=["1m", "5m", "15m", "1h", "1d"].index(default_interval)
    )
    
    st.session_state.timeframe_info = {'period': period, 'interval': interval}
    
    st.markdown("---")
    
    # Stock Universe Selection
    st.subheader("📊 Select Universe")
    include_nifty50 = st.checkbox("Nifty 50 (All 50 stocks)", value=True)
    include_midcap = st.checkbox("Nifty Midcap", value=False)
    include_smallcap = st.checkbox("Nifty Smallcap", value=False)
    
    # Custom stocks
    st.subheader("➕ Custom Stocks")
    custom_stocks_input = st.text_area(
        "Enter symbols (one per line)",
        placeholder="TATAMOTORS.NS\nWIPRO.NS\nTATASTEEL.NS",
        height=100
    )
    
    st.markdown("---")
    
    # Filters
    st.subheader("🔍 Filters")
    min_signal_strength = st.slider("Min Signal Strength", -50, 50, 0)
    min_volume_ratio = st.slider("Min Volume Ratio", 0.5, 3.0, 1.0, 0.1)
    
    st.markdown("---")
    
    # Run Analysis Button
    run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    if st.session_state.last_run_time:
        st.caption(f"Last run: {st.session_state.last_run_time.strftime('%H:%M:%S')}")
        st.caption(f"Timeframe: {interval} / {period}")

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
        st.info(f"Analyzing {len(stock_list)} stocks with {interval} candles over {period} period...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Fetching and analyzing data..."):
            successful_results, all_results = analyze_stocks(stock_list, progress_bar, status_text)
            
            if successful_results:
                st.session_state.analysis_data = pd.DataFrame(successful_results)
                st.session_state.all_stocks_data = all_results
                st.session_state.last_run_time = datetime.now()
                progress_bar.empty()
                status_text.empty()
                st.success(f"✅ Analysis complete! {len(successful_results)} stocks analyzed successfully.")
                st.rerun()
            else:
                st.error("No valid data retrieved. Please check symbols and try again.")

# Display results if analysis has been run
if st.session_state.analysis_data is not None:
    df_results = st.session_state.analysis_data
    
    # Display timeframe badge
    display_timeframe_badge(
        st.session_state.timeframe_info['period'],
        st.session_state.timeframe_info['interval']
    )
    
    # Apply filters
    df_filtered = df_results[
        (df_results['signal_strength'] >= min_signal_strength) &
        (df_results['volume_ratio'] >= min_volume_ratio)
    ].copy()
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Analyzed", len(df_results))
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
    tab_names = [
        "📋 Overview",
        "🎯 Top Opportunities",
        "📈 Long/Short Buildup",
        "🔄 EMA Crossover",
        "📊 Intraday",
        "🌊 Swing",
        "📍 Positional",
        "💎 Long-term",
        "🔍 Detailed",
        "📑 All Stocks"
    ]
    
    tabs = st.tabs(tab_names)
    
    # Tab 0: Overview
    with tabs[0]:
        display_timeframe_badge(
            st.session_state.timeframe_info['period'],
            st.session_state.timeframe_info['interval']
        )
        
        st.subheader("Market Overview")
        
        display_cols = ['symbol', 'price', 'change_pct', 'volume_ratio', 'rsi', 
                       'trend', 'signal_strength', 'risk_reward', 'candle_strength']
        overview_df = df_filtered[display_cols].copy()
        overview_df.columns = ['Symbol', 'Price', 'Change %', 'Vol Ratio', 'RSI', 
                              'Trend', 'Signal', 'R:R', 'Candle']
        overview_df = overview_df.sort_values('Signal', ascending=False)
        overview_df = overview_df.round(2)
        
        st.dataframe(overview_df, use_container_width=True, height=400)
        
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
    
    # Tab 1: Top Opportunities
    with tabs[1]:
        display_timeframe_badge(
            st.session_state.timeframe_info['period'],
            st.session_state.timeframe_info['interval']
        )
        
        st.subheader("🎯 Top Trading Opportunities")
        
        top_stocks = df_filtered.sort_values('signal_strength', ascending=False).head(15)
        
        for idx, row in top_stocks.iterrows():
            with st.expander(f"🔥 {row['symbol']} - Signal: {row['signal_strength']:.0f} | Candle: {row['candle_strength']}"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Price", f"₹{row['price']:.2f}")
                    st.metric("Trend", row['trend'])
                
                with col2:
                    st.metric("Stop Loss", f"₹{row['stop_loss']:.2f}")
                    st.metric("RSI", f"{row['rsi']:.1f}")
                
                with col3:
                    st.metric("Target 1", f"₹{row['target1']:.2f}")
                    st.metric("Target 2", f"₹{row['target2']:.2f}")
                
                with col4:
                    st.metric("R:R", f"1:{row['risk_reward']:.2f}")
                    st.metric("Vol Ratio", f"{row['volume_ratio']:.2f}x")
                
                if row['signals']:
                    st.write("**✅ Active Signals:**")
                    for signal in row['signals']:
                        st.success(f"• {signal}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Support:** ₹{row['support']:.2f}")
                with col2:
                    st.write(f"**Resistance:** ₹{row['resistance']:.2f}")
    
    # Tab 2: Long/Short Buildup
    with tabs[2]:
        display_timeframe_badge(
            st.session_state.timeframe_info['period'],
            st.session_state.timeframe_info['interval']
        )
        
        st.subheader("📈 Long/Short Buildup Analysis")
        st.write("Stocks showing significant position buildup based on price, volume, and RSI changes")
        create_enhanced_buildup_chart(df_filtered)
    
    # Tab 3: EMA Crossover
    with tabs[3]:
        display_timeframe_badge(
            st.session_state.timeframe_info['period'],
            st.session_state.timeframe_info['interval']
        )
        display_ema_crossover_stocks(df_filtered)
    
    # Tab 4: Intraday
    with tabs[4]:
        display_timeframe_badge(
            st.session_state.timeframe_info['period'],
            st.session_state.timeframe_info['interval']
        )
        
        st.subheader("📊 Intraday Trading Signals")
        
        intraday_stocks = df_filtered[
            (df_filtered['volume_ratio'] > 1.5) &
            (df_filtered['signal_strength'] > 15)
        ].copy()
        
        intraday_stocks = intraday_stocks.sort_values('volume_ratio', ascending=False)
        
        if intraday_stocks.empty:
            st.info("No intraday signals found")
        else:
            display_cols = ['symbol', 'price', 'change_pct', 'volume_ratio', 'rsi', 
                          'vwap', 'atr', 'candle_strength', 'stop_loss', 'target1']
            intraday_df = intraday_stocks[display_cols].copy()
            intraday_df.columns = ['Symbol', 'Price', 'Change %', 'Vol Ratio', 'RSI', 
                                  'VWAP', 'ATR', 'Candle', 'SL', 'Target']
            st.dataframe(intraday_df.round(2), use_container_width=True)
    
    # Tab 5: Swing
    with tabs[5]:
        display_timeframe_badge(
            st.session_state.timeframe_info['period'],
            st.session_state.timeframe_info['interval']
        )
        
        st.subheader("🌊 Swing Trading")
        
        swing_stocks = df_filtered[
            (df_filtered['rsi'] > 40) & (df_filtered['rsi'] < 70) &
            (df_filtered['adx'] > 20) &
            (df_filtered['signal_strength'] > 10)
        ].sort_values('signal_strength', ascending=False)
        
        if not swing_stocks.empty:
            st.dataframe(swing_stocks[['symbol', 'price', 'trend', 'rsi', 'adx', 
                                       'ema20', 'stop_loss', 'target1']].round(2), 
                        use_container_width=True)
    
    # Tab 6: Positional
    with tabs[6]:
        display_timeframe_badge(
            st.session_state.timeframe_info['period'],
            st.session_state.timeframe_info['interval']
        )
        
        st.subheader("📍 Positional Trading")
        
        positional = df_filtered[
            (df_filtered['adx'] > 25) &
            (df_filtered['trend'].isin(['Strong Uptrend', 'Uptrend']))
        ].sort_values('adx', ascending=False)
        
        if not positional.empty:
            st.dataframe(positional[['symbol', 'price', 'trend', 'adx', 
                                     'ema50', 'ema200', 'signal_strength']].round(2),
                        use_container_width=True)
    
    # Tab 7: Long-term
    with tabs[7]:
        display_timeframe_badge(
            st.session_state.timeframe_info['period'],
            st.session_state.timeframe_info['interval']
        )
        
        st.subheader("💎 Long-term Investment")
        
        longterm = df_filtered[
            (df_filtered['price'] > df_filtered['ema200']) &
            (df_filtered['ema50'] > df_filtered['ema200'])
        ].sort_values('signal_strength', ascending=False)
        
        if not longterm.empty:
            st.dataframe(longterm[['symbol', 'price', 'trend', 'ema50', 
                                   'ema200', 'rsi', 'adx']].round(2),
                        use_container_width=True)
    
    # Tab 8: Detailed Analysis
    with tabs[8]:
        display_timeframe_badge(
            st.session_state.timeframe_info['period'],
            st.session_state.timeframe_info['interval']
        )
        
        st.subheader("🔍 Detailed Stock Analysis")
        
        selected_symbol = st.selectbox(
            "Select Stock",
            options=df_filtered['symbol'].tolist()
        )
        
        if selected_symbol:
            stock_data = df_filtered[df_filtered['symbol'] == selected_symbol].iloc[0]
            stock_df = stock_data['df']
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Price", f"₹{stock_data['price']:.2f}", f"{stock_data['change_pct']:.2f}%")
            with col2:
                st.metric("RSI", f"{stock_data['rsi']:.1f}")
            with col3:
                st.metric("ADX", f"{stock_data['adx']:.1f}")
            with col4:
                st.metric("Candle", stock_data['candle_strength'])
            with col5:
                st.metric("Signal", f"{stock_data['signal_strength']:.0f}")
            
            # Price Chart
            st.subheader("📈 Price Chart with Indicators")
            
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
            fig.add_trace(go.Scatter(x=stock_df.index[-100:], y=stock_df['EMA9'][-100:],
                                    mode='lines', name='EMA 9', line=dict(color='purple', width=1)))
            fig.add_trace(go.Scatter(x=stock_df.index[-100:], y=stock_df['EMA20'][-100:],
                                    mode='lines', name='EMA 20', line=dict(color='blue', width=1)))
            fig.add_trace(go.Scatter(x=stock_df.index[-100:], y=stock_df['EMA50'][-100:],
                                    mode='lines', name='EMA 50', line=dict(color='orange', width=1)))
            fig.add_trace(go.Scatter(x=stock_df.index[-100:], y=stock_df['EMA200'][-100:],
                                    mode='lines', name='EMA 200', line=dict(color='red', width=1)))
            
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
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=stock_df.index[-100:], y=stock_df['RSI'][-100:],
                                            mode='lines', name='RSI', line=dict(color='purple', width=2)))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig_rsi.update_layout(title="RSI (14)", yaxis_title="RSI", height=300)
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col2:
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=stock_df.index[-100:], y=stock_df['MACD'][-100:],
                                             mode='lines', name='MACD', line=dict(color='blue')))
                fig_macd.add_trace(go.Scatter(x=stock_df.index[-100:], y=stock_df['MACD_Signal'][-100:],
                                             mode='lines', name='Signal', line=dict(color='red')))
                fig_macd.add_trace(go.Bar(x=stock_df.index[-100:], y=stock_df['MACD_Hist'][-100:],
                                         name='Histogram', marker_color='gray'))
                fig_macd.update_layout(title="MACD", yaxis_title="MACD", height=300)
                st.plotly_chart(fig_macd, use_container_width=True)
            
            # Key Details
            st.subheader("📊 Key Levels & Signals")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**EMAs & Angles**")
                st.write(f"EMA 9: ₹{stock_data['ema9']:.2f} ({stock_data['ema9_angle']:.1f}°)")
                st.write(f"EMA 15: ₹{stock_data['ema15']:.2f} ({stock_data['ema15_angle']:.1f}°)")
                st.write(f"EMA 20: ₹{stock_data['ema20']:.2f}")
                st.write(f"EMA 50: ₹{stock_data['ema50']:.2f}")
            
            with col2:
                st.markdown("**Trading Levels**")
                st.write(f"Entry: ₹{stock_data['price']:.2f}")
                st.write(f"Stop Loss: ₹{stock_data['stop_loss']:.2f}")
                st.write(f"Target 1: ₹{stock_data['target1']:.2f}")
                st.write(f"Target 2: ₹{stock_data['target2']:.2f}")
            
            with col3:
                st.markdown("**Support/Resistance**")
                st.write(f"Resistance: ₹{stock_data['resistance']:.2f}")
                st.write(f"Support: ₹{stock_data['support']:.2f}")
                st.write(f"VWAP: ₹{stock_data['vwap']:.2f}")
                st.write(f"ATR: ₹{stock_data['atr']:.2f}")
            
            # Signals
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**✅ Active Signals**")
                if stock_data['signals']:
                    for signal in stock_data['signals']:
                        st.success(f"✓ {signal}")
                else:
                    st.info("No active signals")
            
            with col2:
                st.markdown("**⚠️ Missing Criteria**")
                if stock_data['missing_criteria']:
                    for criteria in stock_data['missing_criteria']:
                        st.warning(f"• {criteria}")
                else:
                    st.success("All criteria met!")
    
    # Tab 9: All Stocks Analysis
    with tabs[9]:
        display_timeframe_badge(
            st.session_state.timeframe_info['period'],
            st.session_state.timeframe_info['interval']
        )
        
        if st.session_state.all_stocks_data:
            display_rejected_stocks(st.session_state.all_stocks_data)
        else:
            st.info("No analysis data available")

else:
    st.info("👆 Configure your preferences and click 'Run Analysis' to start")
    
    # Information display
    st.markdown("---")
    st.markdown("### 📚 Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Technical Analysis**
        - 15+ Indicators
        - Multi-timeframe Support
        - EMA Angle Calculation
        - Candle Strength Analysis
        - Support/Resistance Levels
        """)
    
    with col2:
        st.markdown("""
        **Trading Strategies**
        - Intraday Signals
        - Swing Trading Setups
        - Positional Opportunities
        - Long-term Investment
        - Risk-Reward Analysis
        """)
    
    with col3:
        st.markdown("""
        **Advanced Features**
        - Long/Short Buildup (with metrics)
        - EMA Crossover Detection
        - Missing Criteria Analysis
        - All Stocks Review
        - Complete Transparency
        """)
    
    st.markdown("---")
    st.markdown("### 📊 How to Use")
    
    st.markdown("""
    1. **Select Trading Style** - Choose your preferred timeframe
    2. **Configure Timeframe** - Set period and interval for analysis
    3. **Choose Universe** - Select Nifty 50 (all 50 stocks), Midcap, Smallcap, or add custom stocks
    4. **Set Filters** - Adjust signal strength and volume filters
    5. **Run Analysis** - Click the button and wait for data (includes 1.2s delay per stock for rate limiting)
    6. **Explore Results** - Use tabs to view different analysis perspectives
    7. **Check "All Stocks" Tab** - See why stocks were rejected and what's missing
    
    **Key Tabs:**
    - **Long/Short Buildup**: Horizontal bar charts with price change, volume change, RSI change, OI proxy
    - **EMA Crossover**: Shows EMA 9 & 15 values, angles, and candle strength
    - **All Stocks**: Complete transparency on every analyzed stock with missing criteria
    """)

# Footer
st.markdown("---")
st.caption("⚠️ Disclaimer: Educational purposes only. Not financial advice. Always do your own research.")
st.caption(f"📊 Data: Yahoo Finance | Rate Limited: 1.2s/request | Total Nifty 50 Stocks: {len(NIFTY_50)}")
