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

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
    .signal-buy {background-color: #00ff0020; padding: 1rem; border-radius: 10px; border-left: 5px solid #00ff00;}
    .signal-sell {background-color: #ff000020; padding: 1rem; border-radius: 10px; border-left: 5px solid #ff0000;}
    .signal-hold {background-color: #ffa50020; padding: 1rem; border-radius: 10px; border-left: 5px solid #ffa500;}
    .live-guidance {background-color: #e8f5e9; padding: 2rem; border-radius: 15px; border: 3px solid #4caf50; margin: 1rem 0;}
    .warning-box {background-color: #fff3e0; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #ff9800;}
    .success-box {background-color: #e8f5e9; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #4caf50;}
    .danger-box {background-color: #ffebee; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #f44336;}
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
    st.session_state.paper_capital = 100000.0
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
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        ema_fast = TechnicalIndicators.calculate_ema(data, fast)
        ema_slow = TechnicalIndicators.calculate_ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {'macd': macd_line, 'signal': signal_line, 'histogram': histogram}
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> Dict:
        sma = TechnicalIndicators.calculate_sma(data, period)
        std = data.rolling(window=period).std()
        return {'upper': sma + (std * std_dev), 'middle': sma, 'lower': sma - (std * std_dev)}
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
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
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_smooth = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_smooth.rolling(window=3).mean()
        return {'k': k_smooth, 'd': d_percent}
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
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
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        zscore = (data - rolling_mean) / rolling_std
        return zscore

class TradingAnalyzer:
    """Comprehensive trading analysis engine"""
    
    def __init__(self):
        self.data_cache = {}
        
    def fetch_data_with_retry(self, ticker: str, period: str, interval: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        cache_key = f"{ticker}_{period}_{interval}"
        
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        for attempt in range(max_retries):
            try:
                time.sleep(np.random.uniform(1.5, 2.5))
                data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
                
                if data.empty:
                    return None
                
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
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
        if df is None or df.empty:
            return df
        
        df = df.copy()
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        for period in [9, 20, 50, 100, 200]:
            df[f'EMA_{period}'] = TechnicalIndicators.calculate_ema(df['Close'], period)
            df[f'SMA_{period}'] = TechnicalIndicators.calculate_sma(df['Close'], period)
        
        df['RSI'] = TechnicalIndicators.calculate_rsi(df['Close'], 14)
        df['RSI_Oversold'] = df['RSI'] < 30
        df['RSI_Overbought'] = df['RSI'] > 70
        
        macd_data = TechnicalIndicators.calculate_macd(df['Close'])
        df['MACD'] = macd_data['macd']
        df['MACD_Signal'] = macd_data['signal']
        df['MACD_Histogram'] = macd_data['histogram']
        
        bb_data = TechnicalIndicators.calculate_bollinger_bands(df['Close'])
        df['BB_Upper'] = bb_data['upper']
        df['BB_Middle'] = bb_data['middle']
        df['BB_Lower'] = bb_data['lower']
        
        df['ATR'] = TechnicalIndicators.calculate_atr(df['High'], df['Low'], df['Close'])
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252) * 100
        df['ADX'] = TechnicalIndicators.calculate_adx(df['High'], df['Low'], df['Close'])
        
        stoch_data = TechnicalIndicators.calculate_stochastic(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch_data['k']
        df['Stoch_D'] = stoch_data['d']
        
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['OBV'] = TechnicalIndicators.calculate_obv(df['Close'], df['Volume'])
        
        df['Price_ZScore'] = TechnicalIndicators.calculate_zscore(df['Close'], 20)
        df['Returns_ZScore'] = TechnicalIndicators.calculate_zscore(df['Returns'].fillna(0), 20)
        df['Volume_ZScore'] = TechnicalIndicators.calculate_zscore(df['Volume'], 20) if 'Volume' in df.columns else 0
        
        return df
    
    def find_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict:
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
    
    def find_similar_patterns(self, df: pd.DataFrame, lookback: int = 100) -> List[Dict]:
        """Find similar historical patterns"""
        if df is None or df.empty or len(df) < lookback + 20:
            return []
        
        current_pattern = df['Close'].tail(10).pct_change().fillna(0).values
        similar_patterns = []
        
        for i in range(lookback, len(df) - 20):
            historical_pattern = df['Close'].iloc[i:i+10].pct_change().fillna(0).values
            correlation = np.corrcoef(current_pattern, historical_pattern)[0, 1]
            
            if correlation > 0.85:
                future_returns = ((df['Close'].iloc[i+10:i+20].max() - df['Close'].iloc[i+9]) / df['Close'].iloc[i+9]) * 100
                pattern_date = df.index[i+9]
                pattern_price = df['Close'].iloc[i+9]
                
                similar_patterns.append({
                    'date': pattern_date,
                    'price': pattern_price,
                    'correlation': correlation,
                    'future_move': future_returns,
                    'candles_to_peak': 10
                })
        
        return sorted(similar_patterns, key=lambda x: x['correlation'], reverse=True)[:3]
    
    def generate_signals(self, df: pd.DataFrame, sr_levels: Dict, fib_levels: Dict) -> Dict:
        if df is None or df.empty:
            return {'signal': 'HOLD', 'confidence': 0, 'reasons': []}
        
        signals = []
        reasons = []
        
        current_price = df['Close'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
        current_adx = df['ADX'].iloc[-1] if 'ADX' in df.columns and not pd.isna(df['ADX'].iloc[-1]) else 0
        price_zscore = df['Price_ZScore'].iloc[-1] if not pd.isna(df['Price_ZScore'].iloc[-1]) else 0
        
        if price_zscore < -2:
            signals.append(1)
            reasons.append(f"✓ Price Z-Score {price_zscore:.2f} (oversold, mean reversion expected)")
        elif price_zscore > 2:
            signals.append(-1)
            reasons.append(f"✗ Price Z-Score {price_zscore:.2f} (overbought, mean reversion expected)")
        
        if not pd.isna(df['EMA_20'].iloc[-1]) and not pd.isna(df['EMA_50'].iloc[-1]):
            if current_price > df['EMA_20'].iloc[-1] > df['EMA_50'].iloc[-1]:
                signals.append(1)
                reasons.append("✓ Price above EMA20 and EMA50 (Uptrend)")
            elif current_price < df['EMA_20'].iloc[-1] < df['EMA_50'].iloc[-1]:
                signals.append(-1)
                reasons.append("✗ Price below EMA20 and EMA50 (Downtrend)")
        
        if current_rsi < 30:
            signals.append(1)
            reasons.append(f"✓ RSI oversold at {current_rsi:.1f}")
        elif current_rsi > 70:
            signals.append(-1)
            reasons.append(f"✗ RSI overbought at {current_rsi:.1f}")
        
        if current_adx > 25:
            if not pd.isna(df['EMA_20'].iloc[-1]) and not pd.isna(df['EMA_50'].iloc[-1]):
                if df['EMA_20'].iloc[-1] > df['EMA_50'].iloc[-1]:
                    signals.append(1)
                    reasons.append(f"✓ Strong uptrend (ADX: {current_adx:.1f})")
                else:
                    signals.append(-1)
                    reasons.append(f"✗ Strong downtrend (ADX: {current_adx:.1f})")
        
        if sr_levels.get('analysis'):
            nearest = sr_levels['analysis'][0]
            if nearest['type'] == 'Support' and abs(nearest['distance_pct']) < 2:
                signals.append(1)
                reasons.append(f"✓ Near strong support at {nearest['level']:.2f} ({nearest['touches']} touches)")
            elif nearest['type'] == 'Resistance' and abs(nearest['distance_pct']) < 2:
                signals.append(-1)
                reasons.append(f"✗ Near strong resistance at {nearest['level']:.2f} ({nearest['touches']} touches)")
        
        if fib_levels and 'closest_level' in fib_levels:
            fib_key = fib_levels['closest_level']
            if fib_key in ['0.618', '0.786']:
                signals.append(1)
                reasons.append(f"✓ Near Fibonacci {fib_key} level (bounce expected)")
        
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
            'price_zscore': price_zscore
        }

def generate_friendly_guidance(analyzer: TradingAnalyzer, ticker: str, trade: Dict, results: Dict) -> str:
    """Generate friend-like trading guidance"""
    guidance_parts = []
    
    # Fetch latest 5m data
    df_5m = analyzer.fetch_data_with_retry(ticker, '1d', '5m')
    
    if df_5m is None or df_5m.empty:
        return "Unable to fetch live data. Please try refreshing again."
    
    df_5m = analyzer.calculate_indicators(df_5m)
    sr_levels = analyzer.find_support_resistance(df_5m)
    fib_levels = analyzer.calculate_fibonacci_levels(df_5m)
    elliott = analyzer.detect_elliott_wave(df_5m)
    similar_patterns = analyzer.find_similar_patterns(df_5m, lookback=200)
    
    current_price = df_5m['Close'].iloc[-1]
    current_rsi = df_5m['RSI'].iloc[-1]
    current_adx = df_5m['ADX'].iloc[-1]
    current_vol = df_5m['Volatility'].iloc[-1]
    current_zscore = df_5m['Price_ZScore'].iloc[-1]
    
    # Entry analysis
    guidance_parts.append("### 📊 Trade Entry Analysis (5min/1day timeframe)")
    guidance_parts.append(f"**You entered this {trade['action']} position at ₹{trade['price']:.2f} on {trade['entry_time']}**\n")
    
    guidance_parts.append("**Entry Conditions:**")
    guidance_parts.append(f"- Z-Score was: {trade['strategy']['zscore']:.2f}")
    guidance_parts.append(f"- Volatility was: {trade['strategy']['volatility']:.2f}%")
    guidance_parts.append(f"- RSI was: {trade['strategy']['rsi']:.1f}")
    
    # Current situation
    guidance_parts.append("\n### 📈 Current Market Status (5min/1day timeframe)")
    guidance_parts.append(f"**Current Price: ₹{current_price:.2f}** (moved {((current_price - trade['price']) / trade['price'] * 100):+.2f}% from entry)")
    guidance_parts.append(f"- Z-Score now: {current_zscore:.2f} (was {trade['strategy']['zscore']:.2f})")
    guidance_parts.append(f"- Volatility now: {current_vol:.2f}% (was {trade['strategy']['volatility']:.2f}%)")
    guidance_parts.append(f"- RSI now: {current_rsi:.1f} (was {trade['strategy']['rsi']:.1f})")
    guidance_parts.append(f"- ADX: {current_adx:.1f} {'(Strong trend)' if current_adx > 25 else '(Weak trend)'}")
    
    # Parameter comparison and interpretation
    guidance_parts.append("\n### 🔍 How Parameters Have Changed:")
    
    zscore_change = current_zscore - trade['strategy']['zscore']
    if trade['action'] == 'BUY':
        if zscore_change > 0.5:
            guidance_parts.append("✅ **Z-Score improving** - Price moving toward mean, good for your long position")
        elif zscore_change < -0.5:
            guidance_parts.append("⚠️ **Z-Score deteriorating** - Price moving further from mean")
        else:
            guidance_parts.append("➡️ **Z-Score stable** - No significant mean reversion yet")
    else:
        if zscore_change < -0.5:
            guidance_parts.append("✅ **Z-Score improving** - Price moving toward mean, good for your short position")
        elif zscore_change > 0.5:
            guidance_parts.append("⚠️ **Z-Score deteriorating** - Price moving further from mean")
        else:
            guidance_parts.append("➡️ **Z-Score stable** - No significant mean reversion yet")
    
    vol_change = current_vol - trade['strategy']['volatility']
    if vol_change > 5:
        guidance_parts.append(f"⚠️ **Volatility increased by {vol_change:.1f}%** - Market becoming more volatile")
    elif vol_change < -5:
        guidance_parts.append(f"✅ **Volatility decreased by {abs(vol_change):.1f}%** - Market stabilizing")
    
    # Support/Resistance check
    if sr_levels.get('analysis'):
        nearest = sr_levels['analysis'][0]
        guidance_parts.append(f"\n**Nearest {nearest['type']}: ₹{nearest['level']:.2f}** ({abs(nearest['distance_pct']):.2f}% away, {nearest['touches']} touches)")
        
        if trade['action'] == 'BUY':
            if nearest['type'] == 'Resistance' and abs(nearest['distance_pct']) < 1:
                guidance_parts.append("🚨 **WARNING:** Approaching strong resistance! Consider partial profit booking.")
            elif nearest['type'] == 'Support' and abs(nearest['distance_pct']) < 2:
                guidance_parts.append("✅ **GOOD:** Strong support below providing safety net.")
        else:
            if nearest['type'] == 'Support' and abs(nearest['distance_pct']) < 1:
                guidance_parts.append("🚨 **WARNING:** Approaching strong support! Consider covering position.")
            elif nearest['type'] == 'Resistance' and abs(nearest['distance_pct']) < 2:
                guidance_parts.append("✅ **GOOD:** Strong resistance above providing safety net.")
    
    # Fibonacci analysis
    if fib_levels and 'closest_level' in fib_levels:
        fib_dist = abs((fib_levels['closest_price'] - current_price) / current_price * 100)
        guidance_parts.append(f"\n**Fibonacci Level:** Near {fib_levels['closest_level']} at ₹{fib_levels['closest_price']:.2f} ({fib_dist:.2f}% away)")
        if fib_levels['closest_level'] in ['0.618', '0.5', '0.382']:
            guidance_parts.append("✅ Key Fibonacci level nearby - potential bounce/reversal zone")
    
    # Elliott Wave
    guidance_parts.append(f"\n**Elliott Wave:** Currently in {elliott['wave']} (Confidence: {elliott['confidence']}%)")
    
    # Historical pattern matching
    if similar_patterns:
        guidance_parts.append("\n### 🔮 Historical Pattern Match (5min/1day timeframe)")
        best_match = similar_patterns[0]
        guidance_parts.append(f"**Found similar pattern from {best_match['date'].strftime('%Y-%m-%d %H:%M IST')}!**")
        guidance_parts.append(f"- Price then: ₹{best_match['price']:.2f}")
        guidance_parts.append(f"- Pattern correlation: {best_match['correlation']*100:.1f}%")
        guidance_parts.append(f"- After {best_match['candles_to_peak']} candles, price moved {best_match['future_move']:+.2f}%")
        
        if trade['action'] == 'BUY' and best_match['future_move'] > 2:
            guidance_parts.append(f"✅ **Historical pattern suggests upside potential of ~{best_match['future_move']:.1f}%**")
        elif trade['action'] == 'SELL' and best_match['future_move'] < -2:
            guidance_parts.append(f"✅ **Historical pattern suggests downside move of ~{abs(best_match['future_move']):.1f}%**")
        elif (trade['action'] == 'BUY' and best_match['future_move'] < -2) or (trade['action'] == 'SELL' and best_match['future_move'] > 2):
            guidance_parts.append(f"⚠️ **Historical pattern went against current position!** Be cautious.")
    
    # EMA status check
    guidance_parts.append("\n### 📉 Trend Analysis (5min/1day timeframe)")
    ema_20 = df_5m['EMA_20'].iloc[-1]
    ema_50 = df_5m['EMA_50'].iloc[-1]
    
    if not pd.isna(ema_20) and not pd.isna(ema_50):
        if current_price > ema_20 > ema_50:
            guidance_parts.append("✅ **Strong Uptrend:** Price > EMA20 > EMA50")
            if trade['action'] == 'SELL':
                guidance_parts.append("⚠️ **WARNING:** You're SHORT in an uptrend - risky!")
        elif current_price < ema_20 < ema_50:
            guidance_parts.append("📉 **Strong Downtrend:** Price < EMA20 < EMA50")
            if trade['action'] == 'BUY':
                guidance_parts.append("⚠️ **WARNING:** You're LONG in a downtrend - risky!")
        else:
            guidance_parts.append("➡️ **Sideways/Consolidation:** Mixed EMA signals")
    
    # What's working and what's not
    guidance_parts.append("\n### ✅ What's Working in Current Analysis:")
    working_factors = []
    not_working_factors = []
    
    if trade['action'] == 'BUY':
        if current_price > trade['price']:
            working_factors.append(f"✓ Position in profit (+{((current_price - trade['price']) / trade['price'] * 100):.2f}%)")
        else:
            not_working_factors.append(f"✗ Position in loss ({((current_price - trade['price']) / trade['price'] * 100):.2f}%)")
        
        if current_rsi < 70:
            working_factors.append(f"✓ RSI at {current_rsi:.1f} - not overbought yet")
        else:
            not_working_factors.append(f"✗ RSI at {current_rsi:.1f} - overbought territory")
        
        if current_zscore < 2:
            working_factors.append(f"✓ Z-Score {current_zscore:.2f} - room for upside")
        else:
            not_working_factors.append(f"✗ Z-Score {current_zscore:.2f} - price extended")
        
        if current_adx > 25 and ema_20 > ema_50:
            working_factors.append(f"✓ Strong uptrend confirmed (ADX {current_adx:.1f})")
        elif current_adx > 25 and ema_20 < ema_50:
            not_working_factors.append(f"✗ Strong downtrend active (ADX {current_adx:.1f})")
    
    else:  # SELL position
        if current_price < trade['price']:
            working_factors.append(f"✓ Position in profit (+{((trade['price'] - current_price) / trade['price'] * 100):.2f}%)")
        else:
            not_working_factors.append(f"✗ Position in loss ({((trade['price'] - current_price) / trade['price'] * 100):.2f}%)")
        
        if current_rsi > 30:
            working_factors.append(f"✓ RSI at {current_rsi:.1f} - not oversold yet")
        else:
            not_working_factors.append(f"✗ RSI at {current_rsi:.1f} - oversold territory")
        
        if current_zscore > -2:
            working_factors.append(f"✓ Z-Score {current_zscore:.2f} - room for downside")
        else:
            not_working_factors.append(f"✗ Z-Score {current_zscore:.2f} - price oversold")
        
        if current_adx > 25 and ema_20 < ema_50:
            working_factors.append(f"✓ Strong downtrend confirmed (ADX {current_adx:.1f})")
        elif current_adx > 25 and ema_20 > ema_50:
            not_working_factors.append(f"✗ Strong uptrend active (ADX {current_adx:.1f})")
    
    for factor in working_factors:
        guidance_parts.append(factor)
    
    if not_working_factors:
        guidance_parts.append("\n### ⚠️ What's NOT Working:")
        for factor in not_working_factors:
            guidance_parts.append(factor)
    
    # Final recommendation
    guidance_parts.append("\n### 🎯 My Recommendation:")
    
    danger_signals = len(not_working_factors)
    positive_signals = len(working_factors)
    
    if danger_signals >= 3:
        guidance_parts.append('<div class="danger-box">', )
        guidance_parts.append("🚨 **EXIT NOW!** Too many factors working against your position.")
        guidance_parts.append("Multiple warning signs detected. Better to take a small loss than a big one.")
        guidance_parts.append('</div>')
    elif danger_signals >= 2:
        guidance_parts.append('<div class="warning-box">')
        guidance_parts.append("⚠️ **CAUTION:** Several concerns detected. Consider:")
        guidance_parts.append("- Book partial profits if in gain")
        guidance_parts.append("- Tighten stop-loss")
        guidance_parts.append("- Monitor closely for next 2-3 candles")
        guidance_parts.append('</div>')
    elif positive_signals >= 4:
        guidance_parts.append('<div class="success-box">')
        guidance_parts.append("✅ **HOLD STRONG!** Trade setup working perfectly.")
        guidance_parts.append("All major factors supporting your position. Stay patient!")
        if trade['action'] == 'BUY':
            if sr_levels.get('analysis') and sr_levels['analysis'][0]['type'] == 'Resistance':
                target = sr_levels['analysis'][0]['level']
                guidance_parts.append(f"Target: ₹{target:.2f} (next resistance)")
        else:
            if sr_levels.get('analysis') and sr_levels['analysis'][0]['type'] == 'Support':
                target = sr_levels['analysis'][0]['level']
                guidance_parts.append(f"Target: ₹{target:.2f} (next support)")
        guidance_parts.append('</div>')
    else:
        guidance_parts.append('<div class="warning-box">')
        guidance_parts.append("➡️ **MONITOR:** Mixed signals, no clear direction yet.")
        guidance_parts.append("Wait for 2-3 more candles for clarity. Don't panic, don't be greedy.")
        guidance_parts.append('</div>')
    
    return "\n\n".join(guidance_parts)

def run_detailed_backtest(df: pd.DataFrame, ticker_name: str) -> pd.DataFrame:
    """Run detailed backtesting with entry/exit logic"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    
    # Strategy: Buy when RSI < 30 and price > EMA20, Sell when RSI > 70 or 3% profit/1.5% loss
    trades = []
    position = None
    
    for i in range(50, len(df)):
        current_row = df.iloc[i]
        current_price = current_row['Close']
        current_rsi = current_row['RSI']
        current_ema20 = current_row['EMA_20']
        
        # Entry logic
        if position is None:
            if current_rsi < 30 and current_price > current_ema20:
                # Calculate targets
                entry_price = current_price
                atr = current_row['ATR']
                stop_loss = entry_price - (1.5 * atr)
                target = entry_price + (2 * atr)
                
                position = {
                    'entry_date': current_row.name,
                    'entry_price': entry_price,
                    'entry_rsi': current_rsi,
                    'stop_loss': stop_loss,
                    'target': target,
                    'entry_reason': f"RSI oversold ({current_rsi:.1f}), Price > EMA20"
                }
        
        # Exit logic
        elif position is not None:
            pnl_pct = ((current_price - position['entry_price']) / position['entry_price']) * 100
            
            exit_reason = None
            if current_price >= position['target']:
                exit_reason = f"Target hit (₹{position['target']:.2f})"
            elif current_price <= position['stop_loss']:
                exit_reason = f"Stop-loss hit (₹{position['stop_loss']:.2f})"
            elif current_rsi > 70:
                exit_reason = f"RSI overbought ({current_rsi:.1f})"
            elif i - df.index.get_loc(position['entry_date']) > 20:
                exit_reason = "Time-based exit (20 candles)"
            
            if exit_reason:
                pnl_points = current_price - position['entry_price']
                
                trades.append({
                    'Entry Date': position['entry_date'].strftime('%Y-%m-%d %H:%M IST'),
                    'Entry Price': f"₹{position['entry_price']:.2f}",
                    'Entry RSI': f"{position['entry_rsi']:.1f}",
                    'Stop Loss': f"₹{position['stop_loss']:.2f}",
                    'Target': f"₹{position['target']:.2f}",
                    'Exit Date': current_row.name.strftime('%Y-%m-%d %H:%M IST'),
                    'Exit Price': f"₹{current_price:.2f}",
                    'Exit RSI': f"{current_rsi:.1f}",
                    'Entry Reason': position['entry_reason'],
                    'Exit Reason': exit_reason,
                    'P&L Points': f"{pnl_points:+.2f}",
                    'P&L %': f"{pnl_pct:+.2f}%",
                    'Result': '✅ WIN' if pnl_points > 0 else '❌ LOSS'
                })
                
                position = None
    
    return pd.DataFrame(trades)

def main():
    st.markdown('<h1 class="main-header">📈 Advanced Algorithmic Trading System</h1>', unsafe_allow_html=True)
    
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
        total_analyses = sum(len(periods) for periods in TIMEFRAME_PERIODS.values())
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
                            all_results[f"{interval}_{period}"]['ticker2'] = {'data': df2}
        
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
            
            # Paper Trading Section
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
                        
                        # Fix: Ensure proper quantity calculation
                        position_value = st.session_state.paper_capital * 0.1
                        quantity = max(1, int(position_value / current_price))  # At least 1 share
                        
                        strategy_details = {
                            'zscore': avg_zscore,
                            'volatility': latest_data['Volatility'].iloc[-1],
                            'rsi': latest_data['RSI'].iloc[-1],
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
                        st.success(f"✅ {final_signal} order placed: {quantity} shares @ ₹{current_price:.2f} (Total: ₹{current_price * quantity:,.2f})")
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
                open_trades = [t for t in st.session_state.paper_trades if t['status'] == 'OPEN']
                
                if open_trades:
                    st.markdown("---")
                    st.markdown("### 🔴 LIVE TRADING GUIDANCE - Your Personal Trading Assistant")
                    
                    refresh_btn = st.button("🔄 Refresh Live Analysis", type="secondary", key="refresh_live")
                    
                    if refresh_btn:
                        for trade in open_trades:
                            st.markdown(f"## Position: {trade['action']} {trade['ticker']}")
                            guidance = generate_friendly_guidance(analyzer, st.session_state.ticker1_symbol, trade, results)
                            st.markdown('<div class="live-guidance">', unsafe_allow_html=True)
                            st.markdown(guidance, unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown("---")
            
            # Display trade positions
            if st.session_state.paper_trades:
                st.markdown("#### 📋 Active Positions & Trade History")
                
                open_trades = [t for t in st.session_state.paper_trades if t['status'] == 'OPEN']
                closed_trades = [t for t in st.session_state.paper_trades if t['status'] == 'CLOSED']
                
                if open_trades:
                    st.markdown("**Open Positions:**")
                    
                    position_data = []
                    for idx, trade in enumerate(open_trades):
                        latest_key = [k for k in results.keys() if '1d_' in k][0]
                        latest_data = results[latest_key]['ticker1']['data']
                        current_price = latest_data['Close'].iloc[-1]
                        
                        if trade['quantity'] > 0:
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
                    
                    for idx, trade in enumerate(open_trades):
                        with st.expander(f"Position #{idx+1} Details"):
                            latest_key = [k for k in results.keys() if '1d_' in k][0]
                            latest_data = results[latest_key]['ticker1']['data']
                            current_price = latest_data['Close'].iloc[-1]
                            
                            if trade['quantity'] > 0:
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
                        if trade['quantity'] > 0:
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
            
            # Backtesting Section
            st.markdown("---")
            st.markdown("## 🔬 Detailed Strategy Backtesting")
            
            if st.button("Run Detailed Backtest", type="secondary"):
                with st.spinner("Running comprehensive backtest with detailed trade analysis..."):
                    daily_keys = [k for k in results.keys() if k.startswith('1d_') and ('1y' in k or '2y' in k)]
                    if daily_keys:
                        daily_key = daily_keys[0]
                        df = results[daily_key]['ticker1']['data'].copy()
                        
                        # Run detailed backtest
                        backtest_results = run_detailed_backtest(df, st.session_state.ticker1_name)
                        
                        if not backtest_results.empty:
                            st.markdown("### 📊 Backtest Summary")
                            
                            # Calculate summary statistics
                            total_trades = len(backtest_results)
                            winning_trades = len(backtest_results[backtest_results['Result'] == '✅ WIN'])
                            losing_trades = len(backtest_results[backtest_results['Result'] == '❌ LOSS'])
                            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                            
                            # Extract P&L values
                            pnl_values = []
                            for pnl_str in backtest_results['P&L %']:
                                pnl_values.append(float(pnl_str.replace('%', '').replace('+', '')))
                            
                            avg_win = np.mean([p for p in pnl_values if p > 0]) if any(p > 0 for p in pnl_values) else 0
                            avg_loss = np.mean([p for p in pnl_values if p < 0]) if any(p < 0 for p in pnl_values) else 0
                            total_return = sum(pnl_values)
                            
                            col1, col2, col3, col4, col5 = st.columns(5)
                            col1.metric("Total Trades", total_trades)
                            col2.metric("Win Rate", f"{win_rate:.1f}%")
                            col3.metric("Avg Win", f"{avg_win:.2f}%")
                            col4.metric("Avg Loss", f"{avg_loss:.2f}%")
                            col5.metric("Total Return", f"{total_return:.2f}%")
                            
                            # Display detailed trade table
                            st.markdown("### 📋 Detailed Trade History")
                            st.markdown(f"**Timeframe Used:** 1 Day / {daily_key.split('_')[1]}")
                            st.markdown("**Strategy:** Buy when RSI < 30 and Price > EMA20, Exit at target/stop-loss/RSI > 70")
                            
                            st.dataframe(backtest_results, use_container_width=True, height=400)
                            
                            # Download button
                            csv = backtest_results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Backtest Results (CSV)",
                                data=csv,
                                file_name=f"backtest_{st.session_state.ticker1_name}_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                            # Analysis of results
                            st.markdown("### 💡 Backtest Analysis")
                            
                            if win_rate >= 60:
                                st.success(f"✅ **Excellent Strategy!** {win_rate:.1f}% win rate indicates strong performance.")
                            elif win_rate >= 50:
                                st.info(f"✓ **Good Strategy.** {win_rate:.1f}% win rate with room for optimization.")
                            else:
                                st.warning(f"⚠️ **Strategy needs improvement.** {win_rate:.1f}% win rate suggests optimization needed.")
                            
                            if avg_win > abs(avg_loss) * 1.5:
                                st.success(f"✅ **Excellent Risk/Reward!** Average wins ({avg_win:.2f}%) significantly exceed average losses ({avg_loss:.2f}%).")
                            elif avg_win > abs(avg_loss):
                                st.info(f"✓ **Positive Risk/Reward.** Average wins ({avg_win:.2f}%) exceed losses ({avg_loss:.2f}%).")
                            else:
                                st.warning(f"⚠️ **Poor Risk/Reward.** Average wins ({avg_win:.2f}%) don't sufficiently exceed losses ({avg_loss:.2f}%).")
                            
                        else:
                            st.info("Not enough data to generate detailed backtest results.")
                    else:
                        st.error("No suitable daily data available for backtesting.")
            
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
                    st.markdown(f"**Timeframe:** 1 Day / {key.split('_')[1]}")
                    
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

    else:
        st.info("👆 Configure your analysis parameters in the sidebar and click 'Start Complete Analysis' to begin.")
        
        st.markdown("""
        ### 🎯 Enhanced Features:
        
        - **Multi-Timeframe Analysis**: Analyzes all available timeframes with proper labeling
        - **Comprehensive Indicators**: RSI, MACD, ADX, Bollinger Bands, EMA/SMA, Fibonacci, Stochastic, ATR
        - **Z-Score Analysis**: Price and returns Z-score for mean reversion detection
        - **Support/Resistance Detection**: Identifies strong price levels with historical validation
        - **Elliott Wave Detection**: Simplified wave pattern recognition
        - **Historical Pattern Matching**: Finds similar patterns from past with outcome predictions
        - **Friend-like Live Guidance**: Personalized trading advice comparing entry vs current conditions
        - **What's Working Analysis**: Clear indication of which factors support or oppose your trade
        - **Detailed Backtesting**: Complete trade history with entry/exit levels, reasons, and P&L
        - **Paper Trading**: Test recommendations with real-time guidance and position tracking
        
        ### 💡 Live Trading Guidance Features:
        
        - **Entry Analysis**: Review why you entered and at what parameter values
        - **Current vs Entry Comparison**: See how Z-score, volatility, RSI have changed
        - **Historical Pattern Match**: Find similar past patterns with 85%+ correlation
        - **Support/Resistance Proximity**: Distance and strength of nearby levels
        - **What's Working/Not Working**: Clear breakdown of supporting vs opposing factors
        - **Friend-like Recommendations**: EXIT NOW / CAUTION / HOLD STRONG based on analysis
        - **Timeframe Labels**: Every analysis clearly shows which timeframe/period used
        
        ### 📊 Detailed Backtesting:
        
        - Entry/Exit datetime with IST timezone
        - Entry/Exit price levels and RSI values
        - Stop-loss and target calculations (1.5x ATR for SL, 2x ATR for target)
        - Entry and exit reasons clearly explained
        - P&L in both points and percentage
        - Win/Loss indicator for each trade
        - Summary statistics with win rate and average returns
        
        ### ⚠️ Disclaimer:
        
        This tool is for educational and research purposes only. Not financial advice.
        Past performance does not guarantee future results. Trade at your own risk.
        """)

if __name__ == "__main__":
    main()
