import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import time
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Trading Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .positive {
        color: #00ff00;
        font-weight: bold;
    }
    .negative {
        color: #ff4444;
        font-weight: bold;
    }
    .neutral {
        color: #ffa500;
        font-weight: bold;
    }
    .signal-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 1.1rem;
    }
    .buy-signal {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .sell-signal {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .hold-signal {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'ticker1_data' not in st.session_state:
    st.session_state.ticker1_data = None
if 'ticker2_data' not in st.session_state:
    st.session_state.ticker2_data = None

# Ticker mappings
TICKER_MAP = {
    "NIFTY 50": "^NSEI",
    "Bank NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "USD/INR": "INR=X",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X"
}

# Timeframe and period options
TIMEFRAMES = ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "2h", "4h", "1d"]
PERIODS = ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "6y", "10y", "15y", "20y", "25y", "30y"]

# IST timezone
IST = pytz.timezone('Asia/Kolkata')

def get_ticker_symbol(ticker_input):
    """Convert ticker input to yfinance symbol"""
    return TICKER_MAP.get(ticker_input, ticker_input)

def fetch_data_with_retry(ticker, period, interval, max_retries=3):
    """Fetch data with retry mechanism and rate limiting"""
    for attempt in range(max_retries):
        try:
            time.sleep(np.random.uniform(1.5, 3.0))  # Rate limiting
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if data.empty:
                return None
            
            # Convert to IST timezone
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            data.index = data.index.tz_convert(IST)
            
            return data
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Failed to fetch {ticker}: {str(e)}")
                return None
            time.sleep(2)
    return None

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_adx(high, low, close, period=14):
    """Calculate ADX indicator"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * pd.Series(plus_dm, index=close.index).rolling(window=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(window=period).mean() / atr
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def find_support_resistance(data, window=20, prominence=0.02):
    """Find strong support and resistance levels"""
    highs = data['High'].values
    lows = data['Low'].values
    closes = data['Close'].values
    
    # Find peaks (resistance) and troughs (support)
    resistance_indices, _ = find_peaks(highs, prominence=np.std(highs) * prominence)
    support_indices, _ = find_peaks(-lows, prominence=np.std(lows) * prominence)
    
    # Get resistance levels
    resistances = []
    for idx in resistance_indices:
        price = highs[idx]
        date = data.index[idx]
        # Count touches within 0.5% range
        touches = sum((abs(highs - price) / price < 0.005).astype(int))
        resistances.append({'level': price, 'touches': touches, 'date': date, 'type': 'resistance'})
    
    # Get support levels
    supports = []
    for idx in support_indices:
        price = lows[idx]
        date = data.index[idx]
        touches = sum((abs(lows - price) / price < 0.005).astype(int))
        supports.append({'level': price, 'touches': touches, 'date': date, 'type': 'support'})
    
    # Combine and sort by touches
    levels = sorted(supports + resistances, key=lambda x: x['touches'], reverse=True)
    
    return levels[:10]  # Return top 10 levels

def calculate_fibonacci_levels(data):
    """Calculate Fibonacci retracement levels"""
    high = data['High'].max()
    low = data['Low'].min()
    diff = high - low
    
    levels = {
        '0.0': high,
        '0.236': high - 0.236 * diff,
        '0.382': high - 0.382 * diff,
        '0.5': high - 0.5 * diff,
        '0.618': high - 0.618 * diff,
        '0.786': high - 0.786 * diff,
        '1.0': low
    }
    
    return levels

def detect_elliott_wave(data):
    """Simplified Elliott Wave detection"""
    closes = data['Close'].values
    peaks, _ = find_peaks(closes, distance=5)
    troughs, _ = find_peaks(-closes, distance=5)
    
    # Combine and sort
    turns = sorted(list(peaks) + list(troughs))
    
    if len(turns) < 5:
        return None
    
    # Take last 5 turning points
    wave_points = turns[-5:]
    waves = []
    
    for i, idx in enumerate(wave_points):
        waves.append({
            'wave': i + 1,
            'price': closes[idx],
            'date': data.index[idx],
            'type': 'peak' if idx in peaks else 'trough'
        })
    
    return waves

def detect_rsi_divergence(price, rsi, window=14):
    """Detect RSI divergence"""
    price_peaks, _ = find_peaks(price.values, distance=window)
    price_troughs, _ = find_peaks(-price.values, distance=window)
    rsi_peaks, _ = find_peaks(rsi.values, distance=window)
    rsi_troughs, _ = find_peaks(-rsi.values, distance=window)
    
    divergences = []
    
    # Bullish divergence (price lower low, RSI higher low)
    if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
        last_price_trough = price_troughs[-1]
        prev_price_trough = price_troughs[-2]
        
        if price.iloc[last_price_trough] < price.iloc[prev_price_trough]:
            # Check RSI
            rsi_near_last = rsi_troughs[rsi_troughs >= prev_price_trough - 5]
            rsi_near_last = rsi_near_last[rsi_near_last <= last_price_trough + 5]
            
            if len(rsi_near_last) >= 2:
                if rsi.iloc[rsi_near_last[-1]] > rsi.iloc[rsi_near_last[-2]]:
                    divergences.append({
                        'type': 'bullish',
                        'price': price.iloc[last_price_trough],
                        'date': price.index[last_price_trough]
                    })
    
    # Bearish divergence (price higher high, RSI lower high)
    if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
        last_price_peak = price_peaks[-1]
        prev_price_peak = price_peaks[-2]
        
        if price.iloc[last_price_peak] > price.iloc[prev_price_peak]:
            rsi_near_last = rsi_peaks[rsi_peaks >= prev_price_peak - 5]
            rsi_near_last = rsi_near_last[rsi_near_last <= last_price_peak + 5]
            
            if len(rsi_near_last) >= 2:
                if rsi.iloc[rsi_near_last[-1]] < rsi.iloc[rsi_near_last[-2]]:
                    divergences.append({
                        'type': 'bearish',
                        'price': price.iloc[last_price_peak],
                        'date': price.index[last_price_peak]
                    })
    
    return divergences if divergences else None

def calculate_volatility(returns, window=20):
    """Calculate rolling volatility"""
    return returns.rolling(window=window).std() * np.sqrt(252) * 100

def calculate_zscore(data, window=20):
    """Calculate Z-score for mean reversion"""
    mean = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    zscore = (data - mean) / std
    return zscore

def generate_trading_signals(data, ticker_name, timeframe):
    """Generate comprehensive trading signals"""
    if data is None or len(data) < 50:
        return None
    
    close = data['Close']
    high = data['High']
    low = data['Low']
    
    # Calculate indicators
    rsi = calculate_rsi(close)
    ema_9 = calculate_ema(close, 9)
    ema_20 = calculate_ema(close, 20)
    ema_50 = calculate_ema(close, 50)
    adx = calculate_adx(high, low, close)
    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(close)
    macd, signal, histogram = calculate_macd(close)
    
    returns = close.pct_change()
    volatility = calculate_volatility(returns)
    zscore = calculate_zscore(close)
    
    # Get latest values
    latest_price = close.iloc[-1]
    latest_rsi = rsi.iloc[-1]
    latest_adx = adx.iloc[-1]
    latest_volatility = volatility.iloc[-1]
    latest_zscore = zscore.iloc[-1]
    
    # Support and resistance
    sr_levels = find_support_resistance(data)
    fib_levels = calculate_fibonacci_levels(data)
    elliott = detect_elliott_wave(data)
    divergence = detect_rsi_divergence(close, rsi)
    
    # Signal generation
    signals = []
    score = 0
    
    # RSI signals
    if latest_rsi < 30:
        signals.append(("RSI Oversold", "bullish", f"RSI at {latest_rsi:.1f}, potential bounce"))
        score += 2
    elif latest_rsi > 70:
        signals.append(("RSI Overbought", "bearish", f"RSI at {latest_rsi:.1f}, potential reversal"))
        score -= 2
    
    # Trend signals
    if latest_price > ema_20.iloc[-1] > ema_50.iloc[-1]:
        signals.append(("Trend", "bullish", "Price above 20 & 50 EMA"))
        score += 1
    elif latest_price < ema_20.iloc[-1] < ema_50.iloc[-1]:
        signals.append(("Trend", "bearish", "Price below 20 & 50 EMA"))
        score -= 1
    
    # ADX strength
    if latest_adx > 25:
        signals.append(("ADX", "strong", f"Strong trend (ADX: {latest_adx:.1f})"))
        score += 1 if latest_price > ema_20.iloc[-1] else -1
    
    # Divergence signals
    if divergence:
        for div in divergence:
            if div['type'] == 'bullish':
                signals.append(("RSI Divergence", "bullish", f"Bullish divergence detected"))
                score += 2
            else:
                signals.append(("RSI Divergence", "bearish", f"Bearish divergence detected"))
                score -= 2
    
    # Z-score mean reversion
    if latest_zscore < -2:
        signals.append(("Z-Score", "bullish", f"Oversold (Z: {latest_zscore:.2f}), mean reversion likely"))
        score += 2
    elif latest_zscore > 2:
        signals.append(("Z-Score", "bearish", f"Overbought (Z: {latest_zscore:.2f}), mean reversion likely"))
        score -= 2
    
    # Support/Resistance proximity
    for level in sr_levels[:3]:
        price_diff = abs(latest_price - level['level'])
        pct_diff = (price_diff / latest_price) * 100
        
        if pct_diff < 0.5:  # Within 0.5%
            if level['type'] == 'support':
                signals.append(("Support", "bullish", 
                              f"Near strong support {level['level']:.2f} ({level['touches']} touches)"))
                score += 1
            else:
                signals.append(("Resistance", "bearish", 
                              f"Near strong resistance {level['level']:.2f} ({level['touches']} touches)"))
                score -= 1
    
    # Bollinger Bands
    if latest_price < lower_bb.iloc[-1]:
        signals.append(("Bollinger", "bullish", "Price below lower band"))
        score += 1
    elif latest_price > upper_bb.iloc[-1]:
        signals.append(("Bollinger", "bearish", "Price above upper band"))
        score -= 1
    
    # MACD crossover
    if len(histogram) > 1:
        if histogram.iloc[-2] < 0 and histogram.iloc[-1] > 0:
            signals.append(("MACD", "bullish", "Bullish MACD crossover"))
            score += 1
        elif histogram.iloc[-2] > 0 and histogram.iloc[-1] < 0:
            signals.append(("MACD", "bearish", "Bearish MACD crossover"))
            score -= 1
    
    # Final signal
    if score >= 3:
        final_signal = "STRONG BUY"
        signal_class = "buy-signal"
    elif score >= 1:
        final_signal = "BUY"
        signal_class = "buy-signal"
    elif score <= -3:
        final_signal = "STRONG SELL"
        signal_class = "sell-signal"
    elif score <= -1:
        final_signal = "SELL"
        signal_class = "sell-signal"
    else:
        final_signal = "HOLD"
        signal_class = "hold-signal"
    
    # Calculate entry, SL, and targets
    atr = (high - low).rolling(window=14).mean().iloc[-1]
    
    if "BUY" in final_signal:
        entry = latest_price
        sl = entry - (1.5 * atr)
        target1 = entry + (2 * atr)
        target2 = entry + (3 * atr)
    elif "SELL" in final_signal:
        entry = latest_price
        sl = entry + (1.5 * atr)
        target1 = entry - (2 * atr)
        target2 = entry - (3 * atr)
    else:
        entry = sl = target1 = target2 = latest_price
    
    return {
        'signal': final_signal,
        'signal_class': signal_class,
        'score': score,
        'signals': signals,
        'entry': entry,
        'sl': sl,
        'target1': target1,
        'target2': target2,
        'rsi': latest_rsi,
        'adx': latest_adx,
        'volatility': latest_volatility,
        'zscore': latest_zscore,
        'sr_levels': sr_levels,
        'fib_levels': fib_levels,
        'elliott': elliott,
        'divergence': divergence
    }

def multi_timeframe_analysis(ticker, period, timeframes):
    """Analyze across multiple timeframes"""
    results = {}
    
    for tf in timeframes:
        try:
            data = fetch_data_with_retry(ticker, period, tf)
            if data is not None and len(data) > 0:
                signals = generate_trading_signals(data, ticker, tf)
                if signals:
                    results[tf] = signals
        except Exception as e:
            st.warning(f"Could not analyze {tf} timeframe: {str(e)}")
    
    return results

def create_summary_report(mtf_analysis, ticker_name):
    """Create comprehensive summary report"""
    if not mtf_analysis:
        return "Insufficient data for analysis"
    
    # Aggregate signals across timeframes
    buy_count = sum(1 for tf, sig in mtf_analysis.items() if "BUY" in sig['signal'])
    sell_count = sum(1 for tf, sig in mtf_analysis.items() if "SELL" in sig['signal'])
    hold_count = sum(1 for tf, sig in mtf_analysis.items() if sig['signal'] == "HOLD")
    
    total = len(mtf_analysis)
    
    # Determine consensus
    if buy_count / total > 0.6:
        consensus = "BULLISH"
        color = "🟢"
    elif sell_count / total > 0.6:
        consensus = "BEARISH"
        color = "🔴"
    else:
        consensus = "NEUTRAL"
        color = "🟡"
    
    # Get key levels from daily timeframe
    daily_signals = mtf_analysis.get('1d', mtf_analysis.get(list(mtf_analysis.keys())[-1]))
    
    summary = f"""
    **{color} {ticker_name} - {consensus} OUTLOOK**
    
    **Multi-Timeframe Consensus:** {buy_count} Bullish | {sell_count} Bearish | {hold_count} Neutral (across {total} timeframes)
    
    **Key Technical Levels:**
    • RSI: {daily_signals['rsi']:.1f} {'(Oversold)' if daily_signals['rsi'] < 30 else '(Overbought)' if daily_signals['rsi'] > 70 else '(Neutral)'}
    • ADX: {daily_signals['adx']:.1f} {'(Strong Trend)' if daily_signals['adx'] > 25 else '(Weak Trend)'}
    • Volatility: {daily_signals['volatility']:.1f}%
    • Z-Score: {daily_signals['zscore']:.2f} {'(Mean Reversion Expected)' if abs(daily_signals['zscore']) > 2 else ''}
    
    **Signal Strength:** {daily_signals['score']}/10
    
    **Working Strategies:**
    """
    
    # List working signals
    bullish_signals = [s for s in daily_signals['signals'] if s[1] == 'bullish']
    bearish_signals = [s for s in daily_signals['signals'] if s[1] == 'bearish']
    
    if bullish_signals:
        summary += "\n✅ **Bullish Indicators:**\n"
        for sig in bullish_signals[:3]:
            summary += f"• {sig[0]}: {sig[2]}\n"
    
    if bearish_signals:
        summary += "\n❌ **Bearish Indicators:**\n"
        for sig in bearish_signals[:3]:
            summary += f"• {sig[0]}: {sig[2]}\n"
    
    # Add top support/resistance
    if daily_signals['sr_levels']:
        summary += f"\n**Critical Levels:**\n"
        for level in daily_signals['sr_levels'][:2]:
            summary += f"• {level['type'].title()}: {level['level']:.2f} ({level['touches']} touches)\n"
    
    return summary

# Sidebar
st.sidebar.title("⚙️ Trading Configuration")

ticker1_input = st.sidebar.selectbox(
    "Select Ticker 1",
    list(TICKER_MAP.keys()) + ["Custom"],
    index=0
)

if ticker1_input == "Custom":
    ticker1_custom = st.sidebar.text_input("Enter Custom Ticker 1", "AAPL")
    ticker1 = ticker1_custom
else:
    ticker1 = get_ticker_symbol(ticker1_input)

enable_ratio = st.sidebar.checkbox("Enable Ratio Analysis (Ticker 2)")

ticker2 = None
if enable_ratio:
    ticker2_input = st.sidebar.selectbox(
        "Select Ticker 2",
        list(TICKER_MAP.keys()) + ["Custom"],
        index=1
    )
    
    if ticker2_input == "Custom":
        ticker2_custom = st.sidebar.text_input("Enter Custom Ticker 2", "MSFT")
        ticker2 = ticker2_custom
    else:
        ticker2 = get_ticker_symbol(ticker2_input)

period = st.sidebar.selectbox("Period", PERIODS, index=PERIODS.index("3mo"))
interval = st.sidebar.selectbox("Primary Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("1d"))

# Multi-timeframe selection
st.sidebar.subheader("Multi-Timeframe Analysis")
selected_timeframes = st.sidebar.multiselect(
    "Select Timeframes",
    TIMEFRAMES,
    default=["15m", "1h", "1d"]
)

# Fetch data button
if st.sidebar.button("🔄 Fetch Data", type="primary"):
    with st.spinner("Fetching market data..."):
        st.session_state.ticker1_data = fetch_data_with_retry(ticker1, period, interval)
        
        if enable_ratio and ticker2:
            st.session_state.ticker2_data = fetch_data_with_retry(ticker2, period, interval)
        
        st.session_state.data_fetched = True
        st.success("Data fetched successfully!")

# Main content
st.markdown('<div class="main-header">📈 Advanced Algorithmic Trading Analysis</div>', unsafe_allow_html=True)

if st.session_state.data_fetched and st.session_state.ticker1_data is not None:
    data1 = st.session_state.ticker1_data
    
    # Current metrics
    st.subheader("📊 Current Market Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = data1['Close'].iloc[-1]
    prev_price = data1['Close'].iloc[-2] if len(data1) > 1 else current_price
    price_change = current_price - prev_price
    pct_change = (price_change / prev_price) * 100
    
    rsi = calculate_rsi(data1['Close']).iloc[-1]
    
    with col1:
        color_class = "positive" if price_change > 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <h4>{ticker1}</h4>
            <h2 class="{color_class}">{current_price:.2f}</h2>
            <p class="{color_class}">{price_change:+.2f} ({pct_change:+.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        rsi_color = "positive" if 30 <= rsi <= 70 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <h4>RSI</h4>
            <h2 class="{rsi_color}">{rsi:.1f}</h2>
            <p>{'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        volatility = calculate_volatility(data1['Close'].pct_change()).iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <h4>Volatility</h4>
            <h2>{volatility:.1f}%</h2>
            <p>Annualized</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        zscore_val = calculate_zscore(data1['Close']).iloc[-1]
        zscore_color = "positive" if abs(zscore_val) < 2 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Z-Score</h4>
            <h2 class="{zscore_color}">{zscore_val:.2f}</h2>
            <p>Mean Reversion</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Multi-timeframe analysis
    if selected_timeframes:
        st.subheader("🔍 Multi-Timeframe Analysis")
        
        with st.spinner("Analyzing across timeframes..."):
            mtf_results = multi_timeframe_analysis(ticker1, period, selected_timeframes)
        
        if mtf_results:
            # Summary report
            st.subheader("📋 Executive Summary")
            summary = create_summary_report(mtf_results, ticker1)
            st.markdown(summary)
            
            # Detailed signals by timeframe
            st.subheader("📈 Detailed Timeframe Analysis")
            
            for tf, signals in mtf_results.items():
                with st.expander(f"{tf} Timeframe - {signals['signal']}"):
                    st.markdown(f"""
                    <div class="signal-box {signals['signal_class']}">
                        <strong>Signal:</strong> {signals['signal']} (Score: {signals['score']}/10)
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("**📍 Trading Levels:**")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Entry", f"{signals['entry']:.2f}")
                    col2.metric("Stop Loss", f"{signals['sl']:.2f}")
                    col3.metric("Target 1", f"{signals['target1']:.2f}")
                    col4.metric("Target 2", f"{signals['target2']:.2f}")
                    
                    st.write("**🎯 Active Signals:**")
                    for sig_name, sig_type, sig_desc in signals['signals']:
                        emoji = "🟢" if sig_type == "bullish" else "🔴" if sig_type == "bearish" else "🔵"
                        st.write(f"{emoji} **{sig_name}:** {sig_desc}")
                    
                    # Support/Resistance levels
                    if signals['sr_levels']:
                        st.write("**🎚️ Key Support & Resistance:**")
                        sr_df = pd.DataFrame(signals['sr_levels'][:5])
                        sr_df['date'] = sr_df['date'].dt.strftime('%Y-%m-%d %H:%M IST')
                        st.dataframe(sr_df, use_container_width=True)
                    
                    # Fibonacci levels
                    if signals['fib_levels']:
                        st.write("**📐 Fibonacci Retracement:**")
                        fib_df = pd.DataFrame([
                            {'Level': k, 'Price': f"{v:.2f}"} 
                            for k, v in signals['fib_levels'].items()
                        ])
                        st.dataframe(fib_df, use_container_width=True)
                    
                    # Elliott Wave
                    if signals['elliott']:
                        st.write("**🌊 Elliott Wave Analysis:**")
                        wave_df = pd.DataFrame(signals['elliott'])
                        wave_df['date'] = wave_df['date'].dt.strftime('%Y-%m-%d %H:%M IST')
                        st.dataframe(wave_df, use_container_width=True)
                    
                    # RSI Divergence
                    if signals['divergence']:
                        st.write("**⚡ RSI Divergence:**")
                        for div in signals['divergence']:
                            div_type = div['type'].title()
                            div_emoji = "🟢" if div['type'] == 'bullish' else "🔴"
                            st.write(f"{div_emoji} **{div_type}** at {div['price']:.2f} on {div['date'].strftime('%Y-%m-%d %H:%M IST')}")
    
    # Ratio Analysis
    if enable_ratio and st.session_state.ticker2_data is not None:
        st.subheader("🔄 Ratio Analysis")
        
        data2 = st.session_state.ticker2_data
        
        # Align data
        common_index = data1.index.intersection(data2.index)
        data1_aligned = data1.loc[common_index]
        data2_aligned = data2.loc[common_index]
        
        # Calculate ratio
        ratio = data1_aligned['Close'] / data2_aligned['Close']
        
        # Create comprehensive comparison table
        comparison_df = pd.DataFrame({
            'DateTime (IST)': common_index.strftime('%Y-%m-%d %H:%M:%S'),
            f'{ticker1} Price': data1_aligned['Close'].values,
            f'{ticker2} Price': data2_aligned['Close'].values,
            'Ratio': ratio.values,
            f'{ticker1} RSI': calculate_rsi(data1_aligned['Close']).values,
            f'{ticker2} RSI': calculate_rsi(data2_aligned['Close']).values,
            'Ratio RSI': calculate_rsi(ratio).values,
            f'{ticker1} Volatility': calculate_volatility(data1_aligned['Close'].pct_change()).values,
            f'{ticker2} Volatility': calculate_volatility(data2_aligned['Close'].pct_change()).values,
            f'{ticker1} Z-Score': calculate_zscore(data1_aligned['Close']).values,
            f'{ticker2} Z-Score': calculate_zscore(data2_aligned['Close']).values
        })
        
        st.dataframe(comparison_df.tail(50), use_container_width=True)
        
        # Export functionality
        col1, col2 = st.columns(2)
        with col1:
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"ratio_analysis_{ticker1}_{ticker2}_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Convert to Excel
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                comparison_df.to_excel(writer, index=False, sheet_name='Ratio Analysis')
            excel_data = output.getvalue()
            
            st.download_button(
                label="📥 Download Excel",
                data=excel_data,
                file_name=f"ratio_analysis_{ticker1}_{ticker2}_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Ratio chart
        st.subheader("📊 Ratio Visualization")
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{ticker1} vs {ticker2} Price', 'Ratio', 'Ratio RSI'),
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Price comparison
        fig.add_trace(
            go.Scatter(x=common_index, y=data1_aligned['Close'], name=ticker1, line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=common_index, y=data2_aligned['Close'], name=ticker2, line=dict(color='red'), yaxis='y2'),
            row=1, col=1
        )
        
        # Ratio
        fig.add_trace(
            go.Scatter(x=common_index, y=ratio, name='Ratio', line=dict(color='purple')),
            row=2, col=1
        )
        
        # Ratio RSI
        ratio_rsi = calculate_rsi(ratio)
        fig.add_trace(
            go.Scatter(x=common_index, y=ratio_rsi, name='Ratio RSI', line=dict(color='orange')),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(height=800, showlegend=True, hovermode='x unified')
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Price chart with indicators
    st.subheader("📈 Technical Analysis Chart")
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{ticker1} Price & Indicators', 'RSI', 'MACD', 'Volume'),
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data1.index,
            open=data1['Open'],
            high=data1['High'],
            low=data1['Low'],
            close=data1['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # EMAs
    ema_9 = calculate_ema(data1['Close'], 9)
    ema_20 = calculate_ema(data1['Close'], 20)
    ema_50 = calculate_ema(data1['Close'], 50)
    
    fig.add_trace(go.Scatter(x=data1.index, y=ema_9, name='EMA 9', line=dict(color='yellow', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data1.index, y=ema_20, name='EMA 20', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data1.index, y=ema_50, name='EMA 50', line=dict(color='red', width=1)), row=1, col=1)
    
    # Bollinger Bands
    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(data1['Close'])
    fig.add_trace(go.Scatter(x=data1.index, y=upper_bb, name='BB Upper', line=dict(color='gray', dash='dash', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data1.index, y=lower_bb, name='BB Lower', line=dict(color='gray', dash='dash', width=1)), row=1, col=1)
    
    # Support/Resistance levels
    sr_levels = find_support_resistance(data1)
    for level in sr_levels[:5]:
        color = 'green' if level['type'] == 'support' else 'red'
        fig.add_hline(
            y=level['level'],
            line_dash="dot",
            line_color=color,
            annotation_text=f"{level['type'][:3].upper()} {level['level']:.2f}",
            row=1, col=1
        )
    
    # RSI
    rsi_series = calculate_rsi(data1['Close'])
    fig.add_trace(go.Scatter(x=data1.index, y=rsi_series, name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)
    
    # MACD
    macd, signal, histogram = calculate_macd(data1['Close'])
    fig.add_trace(go.Scatter(x=data1.index, y=macd, name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=data1.index, y=signal, name='Signal', line=dict(color='red')), row=3, col=1)
    fig.add_trace(go.Bar(x=data1.index, y=histogram, name='Histogram', marker_color='gray'), row=3, col=1)
    
    # Volume
    colors = ['green' if data1['Close'].iloc[i] > data1['Open'].iloc[i] else 'red' for i in range(len(data1))]
    fig.add_trace(go.Bar(x=data1.index, y=data1['Volume'], name='Volume', marker_color=colors), row=4, col=1)
    
    fig.update_layout(height=1200, showlegend=True, hovermode='x unified', xaxis_rangeslider_visible=False)
    fig.update_xaxes(title_text="Date", row=4, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("📊 Price Data Table")
    
    display_data = data1.copy()
    display_data['Returns (%)'] = display_data['Close'].pct_change() * 100
    display_data['RSI'] = calculate_rsi(display_data['Close'])
    display_data['Volatility'] = calculate_volatility(display_data['Close'].pct_change())
    display_data.index = display_data.index.strftime('%Y-%m-%d %H:%M:%S IST')
    
    st.dataframe(display_data.tail(100), use_container_width=True)
    
    # Backtesting section
    st.subheader("🔬 Strategy Backtesting")
    
    with st.expander("Configure & Run Backtest"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            strategy_type = st.selectbox(
                "Strategy Type",
                ["RSI Mean Reversion", "Moving Average Crossover", "Breakout", "Combined Signals"]
            )
        
        with col2:
            initial_capital = st.number_input("Initial Capital", value=100000, step=10000)
        
        with col3:
            position_size = st.slider("Position Size (%)", 10, 100, 20)
        
        if st.button("🚀 Run Backtest"):
            with st.spinner("Running backtest..."):
                # Simple backtesting logic
                capital = initial_capital
                positions = []
                trades = []
                
                close_prices = data1['Close'].values
                rsi_values = calculate_rsi(data1['Close']).values
                
                position = None
                
                for i in range(50, len(data1)):
                    if strategy_type == "RSI Mean Reversion":
                        # Buy when RSI < 30, sell when RSI > 70
                        if position is None and rsi_values[i] < 30:
                            shares = (capital * position_size / 100) / close_prices[i]
                            position = {'entry': close_prices[i], 'shares': shares, 'entry_date': data1.index[i]}
                        
                        elif position is not None and rsi_values[i] > 70:
                            exit_price = close_prices[i]
                            profit = (exit_price - position['entry']) * position['shares']
                            capital += profit
                            
                            trades.append({
                                'Entry': position['entry'],
                                'Exit': exit_price,
                                'Profit': profit,
                                'Return %': (profit / (position['entry'] * position['shares'])) * 100,
                                'Entry Date': position['entry_date'],
                                'Exit Date': data1.index[i]
                            })
                            position = None
                    
                    elif strategy_type == "Moving Average Crossover":
                        if i >= 50:
                            ema_20_val = calculate_ema(data1['Close'].iloc[:i+1], 20).iloc[-1]
                            ema_50_val = calculate_ema(data1['Close'].iloc[:i+1], 50).iloc[-1]
                            
                            if position is None and ema_20_val > ema_50_val:
                                shares = (capital * position_size / 100) / close_prices[i]
                                position = {'entry': close_prices[i], 'shares': shares, 'entry_date': data1.index[i]}
                            
                            elif position is not None and ema_20_val < ema_50_val:
                                exit_price = close_prices[i]
                                profit = (exit_price - position['entry']) * position['shares']
                                capital += profit
                                
                                trades.append({
                                    'Entry': position['entry'],
                                    'Exit': exit_price,
                                    'Profit': profit,
                                    'Return %': (profit / (position['entry'] * position['shares'])) * 100,
                                    'Entry Date': position['entry_date'],
                                    'Exit Date': data1.index[i]
                                })
                                position = None
                
                # Calculate metrics
                if trades:
                    trades_df = pd.DataFrame(trades)
                    total_return = ((capital - initial_capital) / initial_capital) * 100
                    winning_trades = len(trades_df[trades_df['Profit'] > 0])
                    losing_trades = len(trades_df[trades_df['Profit'] < 0])
                    win_rate = (winning_trades / len(trades_df)) * 100 if len(trades_df) > 0 else 0
                    
                    avg_win = trades_df[trades_df['Profit'] > 0]['Profit'].mean() if winning_trades > 0 else 0
                    avg_loss = trades_df[trades_df['Profit'] < 0]['Profit'].mean() if losing_trades > 0 else 0
                    
                    # Calculate annualized return
                    days = (data1.index[-1] - data1.index[0]).days
                    years = days / 365.25
                    annualized_return = (((capital / initial_capital) ** (1 / years)) - 1) * 100 if years > 0 else 0
                    
                    st.success("✅ Backtest Complete!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Return", f"{total_return:.2f}%")
                    col2.metric("Annualized Return", f"{annualized_return:.2f}%")
                    col3.metric("Win Rate", f"{win_rate:.1f}%")
                    col4.metric("Total Trades", len(trades_df))
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Final Capital", f"₹{capital:,.2f}")
                    col2.metric("Winning Trades", winning_trades)
                    col3.metric("Losing Trades", losing_trades)
                    col4.metric("Avg Win/Loss", f"₹{avg_win:.2f} / ₹{avg_loss:.2f}")
                    
                    st.write("**Trade History:**")
                    trades_df['Entry Date'] = pd.to_datetime(trades_df['Entry Date']).dt.strftime('%Y-%m-%d %H:%M IST')
                    trades_df['Exit Date'] = pd.to_datetime(trades_df['Exit Date']).dt.strftime('%Y-%m-%d %H:%M IST')
                    st.dataframe(trades_df, use_container_width=True)
                    
                    # Equity curve
                    equity_curve = [initial_capital]
                    for trade in trades:
                        equity_curve.append(equity_curve[-1] + trade['Profit'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=equity_curve,
                        mode='lines',
                        name='Equity Curve',
                        line=dict(color='green', width=2)
                    ))
                    fig.update_layout(
                        title="Equity Curve",
                        xaxis_title="Trade Number",
                        yaxis_title="Capital (₹)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No trades generated with current strategy parameters.")

else:
    st.info("👆 Configure your analysis parameters in the sidebar and click 'Fetch Data' to begin.")
    
    st.markdown("""
    ### 🎯 Features:
    
    - **Multi-Asset Support:** NIFTY, Bank NIFTY, SENSEX, Crypto, Forex, Stocks
    - **Multi-Timeframe Analysis:** From 1-minute to daily charts
    - **Advanced Technical Indicators:** RSI, EMA, ADX, MACD, Bollinger Bands
    - **Support & Resistance:** Automatic detection with touch count
    - **Fibonacci Retracement:** Complete level analysis
    - **Elliott Wave Detection:** Wave pattern identification
    - **RSI Divergence:** Bullish and bearish divergence detection
    - **Z-Score Analysis:** Mean reversion signals
    - **AI-Powered Signals:** Multi-factor signal generation
    - **Ratio Analysis:** Compare two assets comprehensively
    - **Backtesting Engine:** Test your strategies with historical data
    - **Export Functionality:** Download analysis as CSV/Excel
    
    ### 📊 How to Use:
    
    1. Select your ticker(s) from the sidebar
    2. Choose timeframe and period
    3. Select multiple timeframes for comprehensive analysis
    4. Click "Fetch Data" to load market data
    5. Review signals, levels, and recommendations
    6. Run backtests to validate strategies
    7. Export data for further analysis
    
    ### ⚠️ Disclaimer:
    
    This tool is for educational and informational purposes only. Trading involves risk of loss. Always do your own research and consult with a financial advisor before making investment decisions.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Advanced Algorithmic Trading Analysis Tool | Built with Streamlit & Python</p>
    <p>⚠️ For Educational Purposes Only | Not Financial Advice</p>
</div>
""", unsafe_allow_html=True)
