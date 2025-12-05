import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import time
from scipy import stats
from scipy.signal import argrelextrema
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Advanced Algo Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
    .metric-positive {color: #00ff00; font-weight: bold; font-size: 1.2rem;}
    .metric-negative {color: #ff0000; font-weight: bold; font-size: 1.2rem;}
    .metric-neutral {color: #ffa500; font-weight: bold; font-size: 1.2rem;}
    .signal-buy {background-color: #00ff0020; padding: 1rem; border-radius: 10px; border-left: 5px solid #00ff00;}
    .signal-sell {background-color: #ff000020; padding: 1rem; border-radius: 10px; border-left: 5px solid #ff0000;}
    .signal-hold {background-color: #ffa50020; padding: 1rem; border-radius: 10px; border-left: 5px solid #ffa500;}
    .live-guidance {background-color: #e3f2fd; padding: 1.5rem; border-radius: 10px; border: 2px solid #2196f3; margin: 1rem 0;}
    .stProgress > div > div > div > div {background-color: #1f77b4;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'paper_trades' not in st.session_state:
    st.session_state.paper_trades = []
if 'paper_capital' not in st.session_state:
    st.session_state.paper_capital = 100000
if 'live_monitoring' not in st.session_state:
    st.session_state.live_monitoring = False

# Ticker mappings
TICKER_MAP = {
    'NIFTY 50': '^NSEI',
    'Bank NIFTY': '^NSEBANK',
    'SENSEX': '^BSESN',
    'BTC-USD': 'BTC-USD',
    'ETH-USD': 'ETH-USD',
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'USD/INR': 'INR=X',
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X',
}

# Timeframe configurations
TIMEFRAME_PERIODS = {
    '1m': ['1d', '5d'],
    '5m': ['1d', '5d', '1mo'],
    '15m': ['1d', '5d', '1mo'],
    '30m': ['1d', '5d', '1mo'],
    '1h': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
    '1d': ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y'],
    '1wk': ['1y', '2y', '5y', '10y'],
    '1mo': ['1y', '2y', '5y', '10y', '20y']
}

IST = pytz.timezone('Asia/Kolkata')

class TechnicalIndicators:
    """Manual calculation of all technical indicators"""
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """MACD Indicator"""
        ema_fast = TechnicalIndicators.calculate_ema(data, fast)
        ema_slow = TechnicalIndicators.calculate_ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> Dict:
        """Bollinger Bands"""
        sma = TechnicalIndicators.calculate_sma(data, period)
        std = data.rolling(window=period).std()
        
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)
        
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
        
        atr = TechnicalIndicators.calculate_atr(high, low, close, period)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, smooth_k: int = 3) -> Dict:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_smooth = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_smooth.rolling(window=3).mean()
        
        return {
            'k': k_smooth,
            'd': d_percent
        }
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        obv = pd.Series(0.0, index=close.index)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def calculate_zscore(data: pd.Series, window: int = 20) -> pd.Series:
        """Rolling Z-Score"""
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        zscore = (data - rolling_mean) / rolling_std
        return zscore

class TradingAnalyzer:
    """Comprehensive trading analysis engine"""
    
    def __init__(self):
        self.data_cache = {}
        
    def fetch_data_with_retry(self, ticker: str, period: str, interval: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch data with retry logic and rate limiting"""
        cache_key = f"{ticker}_{period}_{interval}"
        
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        for attempt in range(max_retries):
            try:
                time.sleep(np.random.uniform(1.5, 2.5))
                data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
                
                if data.empty:
                    return None
                
                # Handle multi-level columns from yfinance
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # Convert to IST
                if data.index.tz is None:
                    data.index = data.index.tz_localize('UTC').tz_convert(IST)
                else:
                    data.index = data.index.tz_convert(IST)
                
                self.data_cache[cache_key] = data
                return data
                
            except Exception as e:
                if attempt == max_retries - 1:
                    st.warning(f"Failed to fetch {ticker} ({interval}/{period}): {str(e)}")
                    return None
                time.sleep(2)
        
        return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if df is None or df.empty:
            return df
        
        df = df.copy()
        
        # Price-based indicators
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving Averages
        for period in [9, 20, 50, 100, 200]:
            df[f'EMA_{period}'] = TechnicalIndicators.calculate_ema(df['Close'], period)
            df[f'SMA_{period}'] = TechnicalIndicators.calculate_sma(df['Close'], period)
        
        # RSI
        df['RSI'] = TechnicalIndicators.calculate_rsi(df['Close'], 14)
        df['RSI_Oversold'] = df['RSI'] < 30
        df['RSI_Overbought'] = df['RSI'] > 70
        
        # MACD
        macd_data = TechnicalIndicators.calculate_macd(df['Close'])
        df['MACD'] = macd_data['macd']
        df['MACD_Signal'] = macd_data['signal']
        df['MACD_Histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = TechnicalIndicators.calculate_bollinger_bands(df['Close'])
        df['BB_Upper'] = bb_data['upper']
        df['BB_Middle'] = bb_data['middle']
        df['BB_Lower'] = bb_data['lower']
        
        # ATR & Volatility
        df['ATR'] = TechnicalIndicators.calculate_atr(df['High'], df['Low'], df['Close'])
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252) * 100
        
        # ADX
        df['ADX'] = TechnicalIndicators.calculate_adx(df['High'], df['Low'], df['Close'])
        
        # Stochastic
        stoch_data = TechnicalIndicators.calculate_stochastic(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch_data['k']
        df['Stoch_D'] = stoch_data['d']
        
        # Volume indicators
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['OBV'] = TechnicalIndicators.calculate_obv(df['Close'], df['Volume'])
        
        # Z-Score calculations
        df['Price_ZScore'] = TechnicalIndicators.calculate_zscore(df['Close'], 20)
        df['Returns_ZScore'] = TechnicalIndicators.calculate_zscore(df['Returns'].fillna(0), 20)
        df['Volume_ZScore'] = TechnicalIndicators.calculate_zscore(df['Volume'], 20) if 'Volume' in df.columns else 0
        
        return df
    
    def find_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """Find strong support and resistance levels"""
        if df is None or df.empty or len(df) < window * 2:
            return {'support': [], 'resistance': [], 'analysis': []}
        
        df = df.copy()
        
        local_min_idx = argrelextrema(df['Close'].values, np.less_equal, order=window)[0]
        local_max_idx = argrelextrema(df['Close'].values, np.greater_equal, order=window)[0]
        
        support_levels = df['Close'].iloc[local_min_idx].values
        resistance_levels = df['Close'].iloc[local_max_idx].values
        
        def cluster_levels(levels, tolerance=0.005):
            if len(levels) == 0:
                return []
            
            clustered = []
            sorted_levels = np.sort(levels)
            current_cluster = [sorted_levels[0]]
            
            for level in sorted_levels[1:]:
                if (level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                    current_cluster.append(level)
                else:
                    clustered.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            clustered.append(np.mean(current_cluster))
            return clustered
        
        support_clustered = cluster_levels(support_levels)
        resistance_clustered = cluster_levels(resistance_levels)
        
        current_price = df['Close'].iloc[-1]
        analysis = []
        
        for level in support_clustered[-5:]:
            touches = np.sum(np.abs(df['Low'] - level) / level < 0.01)
            sustained = np.sum((df['Close'] > level) & (df['Low'] <= level * 1.01))
            
            if touches >= 2:
                distance_pct = ((current_price - level) / level) * 100
                analysis.append({
                    'type': 'Support',
                    'level': level,
                    'touches': int(touches),
                    'sustained': int(sustained),
                    'distance_pct': distance_pct,
                    'strength': 'Strong' if touches >= 3 else 'Moderate'
                })
        
        for level in resistance_clustered[-5:]:
            touches = np.sum(np.abs(df['High'] - level) / level < 0.01)
            sustained = np.sum((df['Close'] < level) & (df['High'] >= level * 0.99))
            
            if touches >= 2:
                distance_pct = ((level - current_price) / current_price) * 100
                analysis.append({
                    'type': 'Resistance',
                    'level': level,
                    'touches': int(touches),
                    'sustained': int(sustained),
                    'distance_pct': distance_pct,
                    'strength': 'Strong' if touches >= 3 else 'Moderate'
                })
        
        return {
            'support': sorted(support_clustered, reverse=True)[:5],
            'resistance': sorted(resistance_clustered)[:5],
            'analysis': sorted(analysis, key=lambda x: abs(x['distance_pct']))[:8]
        }
    
    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict:
        """Calculate Fibonacci retracement levels"""
        if df is None or df.empty or len(df) < 50:
            return {}
        
        recent_data = df.tail(100)
        swing_high = recent_data['High'].max()
        swing_low = recent_data['Low'].min()
        
        diff = swing_high - swing_low
        
        levels = {
            '0.0': swing_high,
            '0.236': swing_high - 0.236 * diff,
            '0.382': swing_high - 0.382 * diff,
            '0.5': swing_high - 0.5 * diff,
            '0.618': swing_high - 0.618 * diff,
            '0.786': swing_high - 0.786 * diff,
            '1.0': swing_low,
            '1.272': swing_high - 1.272 * diff,
            '1.618': swing_high - 1.618 * diff,
        }
        
        current_price = df['Close'].iloc[-1]
        closest_level = min(levels.items(), key=lambda x: abs(x[1] - current_price))
        
        return {
            'levels': levels,
            'swing_high': swing_high,
            'swing_low': swing_low,
            'closest_level': closest_level[0],
            'closest_price': closest_level[1],
            'current_price': current_price
        }
    
    def detect_elliott_wave(self, df: pd.DataFrame) -> Dict:
        """Simplified Elliott Wave detection"""
        if df is None or df.empty or len(df) < 50:
            return {'wave': 'Unknown', 'confidence': 0}
        
        recent = df.tail(50)
        peaks_idx = argrelextrema(recent['Close'].values, np.greater, order=5)[0]
        troughs_idx = argrelextrema(recent['Close'].values, np.less, order=5)[0]
        
        if len(peaks_idx) >= 3 and len(troughs_idx) >= 2:
            return {'wave': 'Wave 3 (Impulse)', 'confidence': 65}
        elif len(peaks_idx) >= 2 and len(troughs_idx) >= 2:
            return {'wave': 'Wave 2 (Correction)', 'confidence': 55}
        else:
            return {'wave': 'Wave 1 (Starting)', 'confidence': 45}
    
    def generate_signals(self, df: pd.DataFrame, sr_levels: Dict, fib_levels: Dict) -> Dict:
        """Generate trading signals with Z-score analysis"""
        if df is None or df.empty:
            return {'signal': 'HOLD', 'confidence': 0, 'reasons': []}
        
        signals = []
        reasons = []
        
        current_price = df['Close'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
        current_adx = df['ADX'].iloc[-1] if 'ADX' in df.columns and not pd.isna(df['ADX'].iloc[-1]) else 0
        price_zscore = df['Price_ZScore'].iloc[-1] if not pd.isna(df['Price_ZScore'].iloc[-1]) else 0
        returns_zscore = df['Returns_ZScore'].iloc[-1] if not pd.isna(df['Returns_ZScore'].iloc[-1]) else 0
        
        # Z-Score signals (mean reversion)
        if price_zscore < -2:
            signals.append(1)
            reasons.append(f"✓ Price Z-Score {price_zscore:.2f} (oversold, mean reversion expected)")
        elif price_zscore > 2:
            signals.append(-1)
            reasons.append(f"✗ Price Z-Score {price_zscore:.2f} (overbought, mean reversion expected)")
        
        # Trend signals
        if not pd.isna(df['EMA_20'].iloc[-1]) and not pd.isna(df['EMA_50'].iloc[-1]):
            if current_price > df['EMA_20'].iloc[-1] > df['EMA_50'].iloc[-1]:
                signals.append(1)
                reasons.append("✓ Price above EMA20 and EMA50 (Uptrend)")
            elif current_price < df['EMA_20'].iloc[-1] < df['EMA_50'].iloc[-1]:
                signals.append(-1)
                reasons.append("✗ Price below EMA20 and EMA50 (Downtrend)")
        
        # RSI signals
        if current_rsi < 30:
            signals.append(1)
            reasons.append(f"✓ RSI oversold at {current_rsi:.1f}")
        elif current_rsi > 70:
            signals.append(-1)
            reasons.append(f"✗ RSI overbought at {current_rsi:.1f}")
        
        # ADX signals
        if current_adx > 25:
            if not pd.isna(df['EMA_20'].iloc[-1]) and not pd.isna(df['EMA_50'].iloc[-1]):
                if df['EMA_20'].iloc[-1] > df['EMA_50'].iloc[-1]:
                    signals.append(1)
                    reasons.append(f"✓ Strong uptrend (ADX: {current_adx:.1f})")
                else:
                    signals.append(-1)
                    reasons.append(f"✗ Strong downtrend (ADX: {current_adx:.1f})")
        
        # Support/Resistance signals
        if sr_levels.get('analysis'):
            nearest = sr_levels['analysis'][0]
            if nearest['type'] == 'Support' and abs(nearest['distance_pct']) < 2:
                signals.append(1)
                reasons.append(f"✓ Near strong support at {nearest['level']:.2f} ({nearest['touches']} touches)")
            elif nearest['type'] == 'Resistance' and abs(nearest['distance_pct']) < 2:
                signals.append(-1)
                reasons.append(f"✗ Near strong resistance at {nearest['level']:.2f} ({nearest['touches']} touches)")
        
        # Fibonacci signals
        if fib_levels and 'closest_level' in fib_levels:
            fib_key = fib_levels['closest_level']
            if fib_key in ['0.618', '0.786']:
                signals.append(1)
                reasons.append(f"✓ Near Fibonacci {fib_key} level (bounce expected)")
        
        # MACD signals
        if 'MACD' in df.columns and len(df) >= 2:
            if not pd.isna(df['MACD'].iloc[-1]) and not pd.isna(df['MACD_Signal'].iloc[-1]):
                if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] and df['MACD'].iloc[-2] <= df['MACD_Signal'].iloc[-2]:
                    signals.append(1)
                    reasons.append("✓ MACD bullish crossover")
                elif df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1] and df['MACD'].iloc[-2] >= df['MACD_Signal'].iloc[-2]:
                    signals.append(-1)
                    reasons.append("✗ MACD bearish crossover")
        
        if len(signals) == 0:
            return {'signal': 'HOLD', 'confidence': 0, 'reasons': ['No clear signals']}
        
        avg_signal = np.mean(signals)
        confidence = abs(avg_signal) * 100
        
        if avg_signal > 0.3:
            signal = 'BUY'
        elif avg_signal < -0.3:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'confidence': min(confidence, 99),
            'reasons': reasons[:8],
            'price_zscore': price_zscore,
            'returns_zscore': returns_zscore
        }

def generate_live_guidance(analyzer: TradingAnalyzer, ticker: str, results: Dict) -> str:
    """Generate live trading guidance"""
    guidance = []
    
    # Get latest 5m data for real-time analysis
    df_5m = analyzer.fetch_data_with_retry(ticker, '1d', '5m')
    
    if df_5m is not None and not df_5m.empty:
        df_5m = analyzer.calculate_indicators(df_5m)
        sr_levels = analyzer.find_support_resistance(df_5m)
        fib_levels = analyzer.calculate_fibonacci_levels(df_5m)
        elliott = analyzer.detect_elliott_wave(df_5m)
        
        current_price = df_5m['Close'].iloc[-1]
        current_rsi = df_5m['RSI'].iloc[-1]
        current_adx = df_5m['ADX'].iloc[-1]
        current_vol = df_5m['Volatility'].iloc[-1]
        price_zscore = df_5m['Price_ZScore'].iloc[-1]
        
        guidance.append(f"**Current Price:** ₹{current_price:.2f}")
        guidance.append(f"**RSI:** {current_rsi:.1f} {'(Oversold)' if current_rsi < 30 else '(Overbought)' if current_rsi > 70 else '(Neutral)'}")
        guidance.append(f"**ADX:** {current_adx:.1f} {'(Strong Trend)' if current_adx > 25 else '(Weak Trend)'}")
        guidance.append(f"**Volatility:** {current_vol:.2f}%")
        guidance.append(f"**Price Z-Score:** {price_zscore:.2f} {'(Oversold)' if price_zscore < -2 else '(Overbought)' if price_zscore > 2 else '(Normal)'}")
        
        # Support/Resistance
        if sr_levels.get('analysis'):
            nearest = sr_levels['analysis'][0]
            guidance.append(f"**Nearest Level:** {nearest['type']} at ₹{nearest['level']:.2f} ({nearest['distance_pct']:.2f}% away)")
        
        # Fibonacci
        if fib_levels and 'closest_level' in fib_levels:
            guidance.append(f"**Fibonacci Level:** {fib_levels['closest_level']} at ₹{fib_levels['closest_price']:.2f}")
        
        # Elliott Wave
        guidance.append(f"**Elliott Wave:** {elliott['wave']} (Confidence: {elliott['confidence']}%)")
        
        # Trading recommendation
        signals = analyzer.generate_signals(df_5m, sr_levels, fib_levels)
        
        if signals['signal'] == 'BUY':
            guidance.append("\n**💡 RECOMMENDATION: CONSIDER ENTRY (BUY)**")
            guidance.append("Set stop-loss below nearest support")
        elif signals['signal'] == 'SELL':
            guidance.append("\n**💡 RECOMMENDATION: CONSIDER EXIT/SHORT (SELL)**")
            guidance.append("Set stop-loss above nearest resistance")
        else:
            guidance.append("\n**💡 RECOMMENDATION: HOLD/WAIT**")
            guidance.append("Wait for clearer signals or better entry point")
    
    return "\n".join(guidance)

def main():
    st.markdown('<h1 class="main-header">📈 Advanced Algorithmic Trading System</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    ticker1_type = st.sidebar.selectbox("Ticker 1 Type", ["Preset", "Custom"])
    
    if ticker1_type == "Preset":
        ticker1_name = st.sidebar.selectbox("Select Ticker 1", list(TICKER_MAP.keys()))
        ticker1 = TICKER_MAP[ticker1_name]
    else:
        ticker1 = st.sidebar.text_input("Enter Ticker 1 Symbol", "RELIANCE.NS")
        ticker1_name = ticker1
    
    enable_ratio = st.sidebar.checkbox("Enable Ratio Analysis (Ticker 2)")
    
    ticker2 = None
    ticker2_name = None
    if enable_ratio:
        ticker2_type = st.sidebar.selectbox("Ticker 2 Type", ["Preset", "Custom"])
        if ticker2_type == "Preset":
            ticker2_name = st.sidebar.selectbox("Select Ticker 2", list(TICKER_MAP.keys()))
            ticker2 = TICKER_MAP[ticker2_name]
        else:
            ticker2 = st.sidebar.text_input("Enter Ticker 2 Symbol", "TCS.NS")
            ticker2_name = ticker2
    
    st.sidebar.markdown("---")
    analyze_button = st.sidebar.button("🚀 Start Complete Analysis", type="primary", use_container_width=True)
    
    if analyze_button:
        analyzer = TradingAnalyzer()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = {}
        total_analyses = 0
        
        for interval, periods in TIMEFRAME_PERIODS.items():
            total_analyses += len(periods)
        
        if enable_ratio and ticker2:
            total_analyses *= 2
        
        current_analysis = 0
        
        st.markdown("### 🔄 Multi-Timeframe Analysis in Progress...")
        
        for interval, periods in TIMEFRAME_PERIODS.items():
            for period in periods:
                current_analysis += 1
                progress = current_analysis / total_analyses
                progress_bar.progress(progress)
                status_text.text(f"Analyzing {ticker1_name} - {interval}/{period} ({current_analysis}/{total_analyses})")
                
                df1 = analyzer.fetch_data_with_retry(ticker1, period, interval)
                
                if df1 is not None and not df1.empty:
                    df1 = analyzer.calculate_indicators(df1)
                    sr_levels = analyzer.find_support_resistance(df1)
                    fib_levels = analyzer.calculate_fibonacci_levels(df1)
                    elliott = analyzer.detect_elliott_wave(df1)
                    signals = analyzer.generate_signals(df1, sr_levels, fib_levels)
                    
                    all_results[f"{interval}_{period}"] = {
                        'ticker1': {
                            'data': df1,
                            'sr_levels': sr_levels,
                            'fib_levels': fib_levels,
                            'elliott': elliott,
                            'signals': signals
                        }
                    }
                
                if enable_ratio and ticker2:
                    current_analysis += 1
                    progress = current_analysis / total_analyses
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing {ticker2_name} - {interval}/{period} ({current_analysis}/{total_analyses})")
                    
                    df2 = analyzer.fetch_data_with_retry(ticker2, period, interval)
                    
                    if df2 is not None and not df2.empty:
                        df2 = analyzer.calculate_indicators(df2)
                        
                        if f"{interval}_{period}" in all_results:
                            all_results[f"{interval}_{period}"]['ticker2'] = {
                                'data': df2
                            }
        
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis Complete!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.analysis_results = all_results
        st.session_state.ticker1_name = ticker1_name
        st.session_state.ticker1_symbol = ticker1
        st.session_state.ticker2_name = ticker2_name
        st.session_state.ticker2_symbol = ticker2
        st.success("✅ Analysis completed successfully!")
    
    # Display results
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        analyzer = TradingAnalyzer()
        
        st.markdown("---")
        st.markdown("## 📊 Comprehensive Analysis Summary")
        
        all_signals = []
        all_confidences = []
        all_reasons = []
        all_zscores = []
        
        for key, result in results.items():
            if 'ticker1' in result and 'signals' in result['ticker1']:
                signal_data = result['ticker1']['signals']
                signal = signal_data['signal']
                confidence = signal_data['confidence']
                
                if signal == 'BUY':
                    all_signals.append(1)
                elif signal == 'SELL':
                    all_signals.append(-1)
                else:
                    all_signals.append(0)
                
                all_confidences.append(confidence)
                all_reasons.extend(signal_data['reasons'])
                
                if 'price_zscore' in signal_data:
                    all_zscores.append(signal_data['price_zscore'])
        
        if len(all_signals) > 0:
            avg_signal = np.mean(all_signals)
            avg_confidence = np.mean(all_confidences)
            avg_zscore = np.mean(all_zscores) if all_zscores else 0
            
            if avg_signal > 0.2:
                final_signal = "BUY"
                signal_class = "signal-buy"
                signal_emoji = "🟢"
            elif avg_signal < -0.2:
                final_signal = "SELL"
                signal_class = "signal-sell"
                signal_emoji = "🔴"
            else:
                final_signal = "HOLD"
                signal_class = "signal-hold"
                signal_emoji = "🟡"
            
            st.markdown(f'<div class="{signal_class}">', unsafe_allow_html=True)
            st.markdown(f"### {signal_emoji} Final Recommendation: **{final_signal}**")
            st.markdown(f"**Confidence Level:** {avg_confidence:.1f}%")
            st.markdown(f"**Average Z-Score:** {avg_zscore:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("#### 🎯 Key Analysis Points:")
            reason_counts = {}
            for reason in all_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            
            cols = st.columns(2)
            for idx, (reason, count) in enumerate(top_reasons):
                col = cols[idx % 2]
                col.markdown(f"- {reason} *({count} timeframes)*")
            
            st.markdown("---")
            st.markdown("### 📈 Current Market Metrics")
            
            latest_key = [k for k in results.keys() if '1d_' in k]
            if latest_key:
                latest_data = results[latest_key[0]]['ticker1']['data']
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                current_price = latest_data['Close'].iloc[-1]
                price_change = ((latest_data['Close'].iloc[-1] - latest_data['Close'].iloc[-2]) / latest_data['Close'].iloc[-2]) * 100
                
                col1.metric("Current Price", f"₹{current_price:.2f}", f"{price_change:+.2f}%")
                
                current_rsi = latest_data['RSI'].iloc[-1]
                rsi_status = "Oversold" if current_rsi < 30 else ("Overbought" if current_rsi > 70 else "Neutral")
                col2.metric("RSI", f"{current_rsi:.1f}", rsi_status)
                
                current_vol = latest_data['Volatility'].iloc[-1]
                col3.metric("Volatility", f"{current_vol:.2f}%")
                
                current_adx = latest_data['ADX'].iloc[-1]
                col4.metric("ADX", f"{current_adx:.1f}")
                
                price_zscore = latest_data['Price_ZScore'].iloc[-1]
                if not pd.isna(price_zscore):
                    col5.metric("Price Z-Score", f"{price_zscore:.2f}")
            
            st.markdown("---")
            st.markdown("### 🎯 Support & Resistance Analysis")
            
            sr_key = [k for k in results.keys() if '1d_' in k]
            if sr_key and 'sr_levels' in results[sr_key[0]]['ticker1']:
                sr_data = results[sr_key[0]]['ticker1']['sr_levels']
                
                if sr_data.get('analysis'):
                    sr_df = pd.DataFrame(sr_data['analysis'])
                    st.dataframe(sr_df, use_container_width=True)
            
            fib_key = [k for k in results.keys() if '1d_' in k]
            if fib_key and 'fib_levels' in results[fib_key[0]]['ticker1']:
                fib_data = results[fib_key[0]]['ticker1']['fib_levels']
                
                if fib_data and 'levels' in fib_data:
                    st.markdown("### 📐 Fibonacci Retracement Levels")
                    fib_df = pd.DataFrame(list(fib_data['levels'].items()), columns=['Level', 'Price'])
                    fib_df['Distance from Current'] = ((fib_df['Price'] - fib_data['current_price']) / fib_data['current_price'] * 100).round(2)
                    st.dataframe(fib_df, use_container_width=True)
            
            if 'elliott' in results[sr_key[0]]['ticker1']:
                elliott_data = results[sr_key[0]]['ticker1']['elliott']
                st.markdown(f"### 🌊 Elliott Wave Analysis: {elliott_data['wave']} (Confidence: {elliott_data['confidence']}%)")
        
        # Paper Trading Section with Live Guidance
        st.markdown("---")
        st.markdown("## 💼 Paper Trading Simulator with Live Guidance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Virtual Capital", f"₹{st.session_state.paper_capital:,.2f}")
        
        with col2:
            if st.button("Execute Recommended Trade", type="primary"):
                if final_signal != "HOLD":
                    latest_key = [k for k in results.keys() if '1d_' in k][0]
                    latest_data = results[latest_key]['ticker1']['data']
                    current_price = latest_data['Close'].iloc[-1]
                    
                    position_value = st.session_state.paper_capital * 0.1
                    quantity = int(position_value / current_price)
                    
                    # Get strategy details
                    strategy_details = {
                        'zscore': avg_zscore,
                        'volatility': latest_data['Volatility'].iloc[-1],
                        'rsi': latest_data['RSI'].iloc[-1],
                        'support_resistance': results[latest_key]['ticker1']['sr_levels']['analysis'][0] if results[latest_key]['ticker1']['sr_levels']['analysis'] else None
                    }
                    
                    trade = {
                        'timestamp': datetime.now(IST),
                        'ticker': st.session_state.ticker1_name,
                        'action': final_signal,
                        'price': current_price,
                        'quantity': quantity,
                        'value': current_price * quantity,
                        'confidence': avg_confidence,
                        'status': 'OPEN',
                        'strategy': strategy_details,
                        'entry_time': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')
                    }
                    
                    st.session_state.paper_trades.append(trade)
                    st.session_state.live_monitoring = True
                    st.success(f"✅ {final_signal} order placed: {quantity} shares @ ₹{current_price:.2f}")
                else:
                    st.info("Current recommendation is HOLD - no trade executed")
        
        with col3:
            total_pnl = 0
            for trade in st.session_state.paper_trades:
                if trade['status'] == 'CLOSED' and 'exit_price' in trade:
                    if trade['action'] == 'BUY':
                        pnl = (trade['exit_price'] - trade['price']) * trade['quantity']
                    else:
                        pnl = (trade['price'] - trade['exit_price']) * trade['quantity']
                    total_pnl += pnl
            
            pnl_color = "green" if total_pnl >= 0 else "red"
            st.markdown(f"**Total P&L:** <span style='color:{pnl_color}'>₹{total_pnl:,.2f}</span>", unsafe_allow_html=True)
        
        # Live Guidance Panel
        if st.session_state.live_monitoring and st.session_state.paper_trades:
            st.markdown("---")
            st.markdown("### 🔴 LIVE TRADING GUIDANCE")
            
            refresh_btn = st.button("🔄 Refresh Live Data", type="secondary")
            
            if refresh_btn or st.session_state.live_monitoring:
                guidance = generate_live_guidance(analyzer, st.session_state.ticker1_symbol, results)
                
                st.markdown('<div class="live-guidance">', unsafe_allow_html=True)
                st.markdown(guidance)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Display trade positions
        if st.session_state.paper_trades:
            st.markdown("#### 📋 Active Positions & Trade History")
            
            open_trades = [t for t in st.session_state.paper_trades if t['status'] == 'OPEN']
            closed_trades = [t for t in st.session_state.paper_trades if t['status'] == 'CLOSED']
            
            if open_trades:
                st.markdown("**Open Positions:**")
                
                # Create detailed position table
                position_data = []
                for idx, trade in enumerate(open_trades):
                    # Get current price
                    latest_key = [k for k in results.keys() if '1d_' in k][0]
                    latest_data = results[latest_key]['ticker1']['data']
                    current_price = latest_data['Close'].iloc[-1]
                    
                    if trade['action'] == 'BUY':
                        unrealized_pnl = (current_price - trade['price']) * trade['quantity']
                        unrealized_pnl_pct = ((current_price - trade['price']) / trade['price']) * 100
                    else:
                        unrealized_pnl = (trade['price'] - current_price) * trade['quantity']
                        unrealized_pnl_pct = ((trade['price'] - current_price) / trade['price']) * 100
                    
                    strategy_str = f"Z-Score: {trade['strategy']['zscore']:.2f}, Vol: {trade['strategy']['volatility']:.2f}%, RSI: {trade['strategy']['rsi']:.1f}"
                    
                    position_data.append({
                        'Position': idx + 1,
                        'Ticker': trade['ticker'],
                        'Action': trade['action'],
                        'Entry Price': f"₹{trade['price']:.2f}",
                        'Current Price': f"₹{current_price:.2f}",
                        'Quantity': trade['quantity'],
                        'Entry Time': trade['entry_time'],
                        'Unrealized P&L': f"₹{unrealized_pnl:,.2f}",
                        'P&L %': f"{unrealized_pnl_pct:+.2f}%",
                        'Strategy Used': strategy_str,
                        'Confidence': f"{trade['confidence']:.1f}%"
                    })
                
                if position_data:
                    position_df = pd.DataFrame(position_data)
                    st.dataframe(position_df, use_container_width=True)
                
                # Individual position details with close button
                for idx, trade in enumerate(open_trades):
                    with st.expander(f"Position #{idx+1} - {trade['action']} {trade['ticker']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Entry Price:** ₹{trade['price']:.2f}")
                            st.write(f"**Quantity:** {trade['quantity']}")
                            st.write(f"**Position Value:** ₹{trade['value']:,.2f}")
                        
                        with col2:
                            st.write(f"**Entry Time:** {trade['entry_time']}")
                            st.write(f"**Confidence:** {trade['confidence']:.1f}%")
                        
                        st.write("**Strategy Metrics:**")
                        st.write(f"- Z-Score at Entry: {trade['strategy']['zscore']:.2f}")
                        st.write(f"- Volatility at Entry: {trade['strategy']['volatility']:.2f}%")
                        st.write(f"- RSI at Entry: {trade['strategy']['rsi']:.1f}")
                        
                        # Get current metrics
                        latest_key = [k for k in results.keys() if '1d_' in k][0]
                        latest_data = results[latest_key]['ticker1']['data']
                        current_price = latest_data['Close'].iloc[-1]
                        
                        if trade['action'] == 'BUY':
                            unrealized_pnl = (current_price - trade['price']) * trade['quantity']
                            unrealized_pnl_pct = ((current_price - trade['price']) / trade['price']) * 100
                        else:
                            unrealized_pnl = (trade['price'] - current_price) * trade['quantity']
                            unrealized_pnl_pct = ((trade['price'] - current_price) / trade['price']) * 100
                        
                        pnl_color = "green" if unrealized_pnl >= 0 else "red"
                        st.markdown(f"**Current Price:** ₹{current_price:.2f}")
                        st.markdown(f"**Unrealized P&L:** <span style='color:{pnl_color}'>₹{unrealized_pnl:,.2f} ({unrealized_pnl_pct:+.2f}%)</span>", unsafe_allow_html=True)
                        
                        if st.button(f"Close Position #{idx+1}", key=f"close_{idx}"):
                            trade_idx = len(st.session_state.paper_trades) - len(open_trades) + idx
                            st.session_state.paper_trades[trade_idx]['status'] = 'CLOSED'
                            st.session_state.paper_trades[trade_idx]['exit_price'] = current_price
                            st.session_state.paper_trades[trade_idx]['exit_time'] = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')
                            st.session_state.paper_capital += unrealized_pnl
                            
                            if len(open_trades) == 1:
                                st.session_state.live_monitoring = False
                            
                            st.rerun()
            
            if closed_trades:
                st.markdown("**Closed Positions:**")
                closed_df_data = []
                
                for trade in closed_trades:
                    if trade['action'] == 'BUY':
                        pnl = (trade['exit_price'] - trade['price']) * trade['quantity']
                    else:
                        pnl = (trade['price'] - trade['exit_price']) * trade['quantity']
                    
                    pnl_pct = (pnl / trade['value']) * 100
                    
                    strategy_str = f"Z:{trade['strategy']['zscore']:.2f}, Vol:{trade['strategy']['volatility']:.1f}%, RSI:{trade['strategy']['rsi']:.0f}"
                    
                    closed_df_data.append({
                        'Ticker': trade['ticker'],
                        'Action': trade['action'],
                        'Entry Price': f"₹{trade['price']:.2f}",
                        'Exit Price': f"₹{trade['exit_price']:.2f}",
                        'Quantity': trade['quantity'],
                        'Entry Time': trade['entry_time'],
                        'Exit Time': trade['exit_time'],
                        'P&L': f"₹{pnl:,.2f}",
                        'P&L %': f"{pnl_pct:+.2f}%",
                        'Strategy': strategy_str
                    })
                
                if closed_df_data:
                    closed_df = pd.DataFrame(closed_df_data)
                    st.dataframe(closed_df, use_container_width=True)
        
        # Detailed Analysis Tabs
        st.markdown("---")
        st.markdown("## 📊 Detailed Multi-Timeframe Analysis")
        
        analysis_tabs = st.tabs(["1 Day", "1 Hour", "15 Min", "Summary Table"])
        
        with analysis_tabs[0]:
            daily_keys = [k for k in results.keys() if k.startswith('1d_')]
            if daily_keys:
                for key in daily_keys[:3]:
                    period = key.split('_')[1]
                    st.markdown(f"### Daily - {period} Period")
                    
                    result = results[key]['ticker1']
                    df = result['data']
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Signal", result['signals']['signal'])
                    col2.metric("Confidence", f"{result['signals']['confidence']:.1f}%")
                    col3.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
                    col4.metric("ADX", f"{df['ADX'].iloc[-1]:.1f}")
                    col5.metric("Z-Score", f"{df['Price_ZScore'].iloc[-1]:.2f}")
                    
                    st.markdown("**Key Indicators:**")
                    for reason in result['signals']['reasons'][:5]:
                        st.markdown(f"- {reason}")
                    
                    st.markdown("---")
        
        with analysis_tabs[1]:
            hourly_keys = [k for k in results.keys() if k.startswith('1h_')]
            if hourly_keys:
                for key in hourly_keys[:2]:
                    period = key.split('_')[1]
                    st.markdown(f"### Hourly - {period} Period")
                    
                    result = results[key]['ticker1']
                    df = result['data']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Signal", result['signals']['signal'])
                    col2.metric("Confidence", f"{result['signals']['confidence']:.1f}%")
                    col3.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
                    col4.metric("Volatility", f"{df['Volatility'].iloc[-1]:.2f}%")
                    
                    st.markdown("**Key Indicators:**")
                    for reason in result['signals']['reasons'][:5]:
                        st.markdown(f"- {reason}")
                    
                    st.markdown("---")
        
        with analysis_tabs[2]:
            min15_keys = [k for k in results.keys() if k.startswith('15m_')]
            if min15_keys:
                for key in min15_keys[:2]:
                    period = key.split('_')[1]
                    st.markdown(f"### 15-Minute - {period} Period")
                    
                    result = results[key]['ticker1']
                    df = result['data']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Signal", result['signals']['signal'])
                    col2.metric("Confidence", f"{result['signals']['confidence']:.1f}%")
                    col3.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
                    col4.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")
                    
                    st.markdown("**Key Indicators:**")
                    for reason in result['signals']['reasons'][:5]:
                        st.markdown(f"- {reason}")
                    
                    st.markdown("---")
        
        with analysis_tabs[3]:
            st.markdown("### All Timeframes Summary")
            
            summary_data = []
            for key, result in results.items():
                if 'ticker1' in result and 'signals' in result['ticker1']:
                    interval, period = key.split('_')
                    signal_data = result['ticker1']['signals']
                    df = result['ticker1']['data']
                    
                    summary_data.append({
                        'Timeframe': interval,
                        'Period': period,
                        'Signal': signal_data['signal'],
                        'Confidence': f"{signal_data['confidence']:.1f}%",
                        'RSI': f"{df['RSI'].iloc[-1]:.1f}",
                        'ADX': f"{df['ADX'].iloc[-1]:.1f}",
                        'Volatility': f"{df['Volatility'].iloc[-1]:.2f}%",
                        'Z-Score': f"{df['Price_ZScore'].iloc[-1]:.2f}",
                        'Data Points': len(df)
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, height=400)
        
        # Ratio Analysis
        if enable_ratio and ticker2:
            st.markdown("---")
            st.markdown("## 📊 Ratio Analysis")
            
            ratio_keys = [k for k in results.keys() if 'ticker2' in results[k] and '1d_' in k]
            if ratio_keys:
                key = ratio_keys[0]
                df1 = results[key]['ticker1']['data']
                df2 = results[key]['ticker2']['data']
                
                common_index = df1.index.intersection(df2.index)
                df1_aligned = df1.loc[common_index]
                df2_aligned = df2.loc[common_index]
                
                ratio_df = pd.DataFrame({
                    'DateTime': common_index,
                    'Ticker1_Price': df1_aligned['Close'].values,
                    'Ticker2_Price': df2_aligned['Close'].values,
                    'Ratio': df1_aligned['Close'].values / df2_aligned['Close'].values,
                    'Ticker1_RSI': df1_aligned['RSI'].values,
                    'Ticker2_RSI': df2_aligned['RSI'].values,
                    'Ticker1_Volatility': df1_aligned['Volatility'].values,
                    'Ticker2_Volatility': df2_aligned['Volatility'].values,
                    'Ticker1_ZScore': df1_aligned['Price_ZScore'].values,
                    'Ticker2_ZScore': df2_aligned['Price_ZScore'].values,
                })
                
                ratio_df['Ratio_Returns'] = ratio_df['Ratio'].pct_change()
                ratio_df['Ratio_RSI'] = TechnicalIndicators.calculate_rsi(pd.Series(ratio_df['Ratio'].values), 14)
                ratio_df['Ratio_ZScore'] = TechnicalIndicators.calculate_zscore(pd.Series(ratio_df['Ratio'].values), 20)
                
                st.markdown(f"### {st.session_state.ticker1_name} / {st.session_state.ticker2_name} Ratio Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                current_ratio = ratio_df['Ratio'].iloc[-1]
                ratio_change = ((ratio_df['Ratio'].iloc[-1] - ratio_df['Ratio'].iloc[-2]) / ratio_df['Ratio'].iloc[-2]) * 100
                
                col1.metric("Current Ratio", f"{current_ratio:.4f}", f"{ratio_change:+.2f}%")
                col2.metric("Ratio RSI", f"{ratio_df['Ratio_RSI'].iloc[-1]:.1f}")
                col3.metric("Ratio Z-Score", f"{ratio_df['Ratio_ZScore'].iloc[-1]:.2f}")
                col4.metric("Spread Volatility", f"{ratio_df['Ratio_Returns'].std() * 100:.2f}%")
                
                st.markdown("#### Detailed Ratio Data (Last 20 rows)")
                display_df = ratio_df[['DateTime', 'Ticker1_Price', 'Ticker2_Price', 'Ratio', 
                                       'Ticker1_RSI', 'Ticker2_RSI', 'Ratio_RSI', 
                                       'Ticker1_Volatility', 'Ticker2_Volatility',
                                       'Ticker1_ZScore', 'Ticker2_ZScore', 'Ratio_ZScore']].tail(20)
                st.dataframe(display_df, use_container_width=True)
                
                csv = ratio_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Ratio Analysis (CSV)",
                    data=csv,
                    file_name=f"ratio_analysis_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Backtesting
        st.markdown("---")
        st.markdown("## 🔬 Strategy Backtesting")
        
        if st.button("Run Backtest", type="secondary"):
            with st.spinner("Running backtest..."):
                daily_keys = [k for k in results.keys() if k.startswith('1d_') and ('1y' in k or '2y' in k)]
                if daily_keys:
                    daily_key = daily_keys[0]
                    df = results[daily_key]['ticker1']['data'].copy()
                    
                    df['Signal'] = 0
                    df.loc[(df['RSI'] < 30) & (df['Close'] > df['EMA_20']), 'Signal'] = 1
                    df.loc[(df['RSI'] > 70), 'Signal'] = -1
                    
                    df['Position'] = df['Signal'].shift(1)
                    df['Strategy_Returns'] = df['Returns'] * df['Position']
                    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()
                    df['Cumulative_Market_Returns'] = (1 + df['Returns']).cumprod()
                    
                    total_return = (df['Cumulative_Strategy_Returns'].iloc[-1] - 1) * 100
                    market_return = (df['Cumulative_Market_Returns'].iloc[-1] - 1) * 100
                    
                    strategy_std = df['Strategy_Returns'].std()
                    sharpe_ratio = (df['Strategy_Returns'].mean() / strategy_std) * np.sqrt(252) if strategy_std > 0 else 0
                    
                    max_drawdown = ((df['Cumulative_Strategy_Returns'].cummax() - df['Cumulative_Strategy_Returns']) / df['Cumulative_Strategy_Returns'].cummax()).max() * 100
                    
                    st.markdown("### Backtest Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Strategy Return", f"{total_return:.2f}%")
                    col2.metric("Buy & Hold Return", f"{market_return:.2f}%")
                    col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    col4.metric("Max Drawdown", f"-{max_drawdown:.2f}%")
                    
                    performance_df = pd.DataFrame({
                        'Date': df.index,
                        'Strategy': df['Cumulative_Strategy_Returns'].values * 100000,
                        'Market': df['Cumulative_Market_Returns'].values * 100000
                    }).set_index('Date')
                    
                    st.line_chart(performance_df)
                    
                    trades = df[df['Signal'] != 0].copy()
                    num_trades = len(trades)
                    winning_trades = len(trades[trades['Strategy_Returns'] > 0])
                    win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0
                    
                    avg_return = df['Strategy_Returns'][df['Strategy_Returns'] != 0].mean() * 100
                    
                    st.markdown(f"""
                    **Trade Statistics:**
                    - Total Trades: {num_trades}
                    - Winning Trades: {winning_trades}
                    - Win Rate: {win_rate:.1f}%
                    - Average Return per Trade: {avg_return:.2f}%
                    """)
                    
                    if total_return < 0:
                        st.warning("⚠️ Strategy showing negative returns. Consider optimization or different parameters.")
                        st.info("💡 Tip: Try adjusting RSI thresholds, adding stop-loss levels, or combining with other indicators.")
                    else:
                        st.success(f"✅ Strategy generated positive returns of {total_return:.2f}% over the backtest period!")
                else:
                    st.error("No suitable data available for backtesting.")

    else:
        st.info("👆 Configure your analysis parameters in the sidebar and click 'Start Complete Analysis' to begin.")
        
        st.markdown("""
        ### 🎯 Features:
        
        - **Multi-Timeframe Analysis**: Analyzes all available timeframes from 1-minute to monthly
        - **Comprehensive Indicators**: RSI, MACD, ADX, Bollinger Bands, EMA/SMA, Fibonacci, Stochastic, ATR
        - **Z-Score Analysis**: Price and returns Z-score for mean reversion detection
        - **Support/Resistance Detection**: Identifies strong price levels with historical validation
        - **Elliott Wave Detection**: Simplified wave pattern recognition
        - **AI-Powered Signals**: Generates BUY/SELL/HOLD recommendations with confidence levels
        - **Ratio Analysis**: Compare two assets for spread trading opportunities with Z-scores
        - **Paper Trading with Live Guidance**: Test recommendations with real-time market analysis
        - **Live Trading Guidance**: Continuous monitoring with entry/exit/hold recommendations
        - **Detailed Position Tracking**: Track all metrics including entry/exit prices, strategy used, P&L
        - **Backtesting Engine**: Validate strategies with historical performance data
        - **Real-time Analysis**: Rate-limited API calls to ensure reliability
        
        ### 📝 Instructions:
        
        1. Select your asset(s) from the sidebar
        2. Optionally enable ratio analysis for pair trading
        3. Click "Start Complete Analysis" to run comprehensive analysis
        4. Review multi-timeframe signals, Z-scores, and confidence levels
        5. Execute paper trades to test recommendations
        6. Monitor positions with live guidance and track performance
        7. Use "Refresh Live Data" button to get updated market conditions
        
        ### 💡 Live Trading Guidance Features:
        
        - **Real-time Price Updates**: Current price with RSI, ADX, volatility
        - **Z-Score Monitoring**: Track mean reversion opportunities
        - **Support/Resistance Levels**: Distance from key levels
        - **Fibonacci Levels**: Proximity to important retracement levels
        - **Elliott Wave Status**: Current wave pattern and confidence
        - **Entry/Exit Recommendations**: Clear guidance on when to trade
        - **Strategy Tracking**: All positions show strategy metrics (Z-score, volatility, RSI at entry)
        
        ### ⚠️ Disclaimer:
        
        This tool is for educational and research purposes only. Not financial advice.
        Past performance does not guarantee future results. Trade at your own risk.
        """)

if __name__ == "__main__":
    main()
