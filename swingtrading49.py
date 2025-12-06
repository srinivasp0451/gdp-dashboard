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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

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
    '5m': ['1d', '5d'],
    '15m': ['1d', '5d'],
    '1h': ['1d', '5d', '1mo'],
    '1d': ['1mo', '3mo', '6mo', '1y', '2y'],
}

IST = pytz.timezone('Asia/Kolkata')

class TechnicalIndicators:
    """Manual calculation of all technical indicators"""
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False, min_periods=period).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral RSI
    
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
        std = data.rolling(window=period, min_periods=period).std()
        return {'upper': sma + (std * std_dev), 'middle': sma, 'lower': sma - (std * std_dev)}
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        return atr.fillna(method='bfill')
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)
        
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
        
        atr = TechnicalIndicators.calculate_atr(high, low, close, period)
        plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(window=period, min_periods=period).mean()
        return adx.fillna(0)
    
    @staticmethod
    def calculate_zscore(data: pd.Series, window: int = 20) -> pd.Series:
        rolling_mean = data.rolling(window=window, min_periods=window).mean()
        rolling_std = data.rolling(window=window, min_periods=window).std()
        zscore = (data - rolling_mean) / rolling_std.replace(0, np.nan)
        return zscore.fillna(0)

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
        df['Returns'] = df['Close'].pct_change().fillna(0)
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
        
        for period in [9, 20, 50, 100, 200]:
            if len(df) >= period:
                df[f'EMA_{period}'] = TechnicalIndicators.calculate_ema(df['Close'], period)
                df[f'SMA_{period}'] = TechnicalIndicators.calculate_sma(df['Close'], period)
        
        if len(df) >= 14:
            df['RSI'] = TechnicalIndicators.calculate_rsi(df['Close'], 14)
            df['RSI_Oversold'] = df['RSI'] < 30
            df['RSI_Overbought'] = df['RSI'] > 70
        
        if len(df) >= 26:
            macd_data = TechnicalIndicators.calculate_macd(df['Close'])
            df['MACD'] = macd_data['macd']
            df['MACD_Signal'] = macd_data['signal']
            df['MACD_Histogram'] = macd_data['histogram']
        
        if len(df) >= 20:
            bb_data = TechnicalIndicators.calculate_bollinger_bands(df['Close'])
            df['BB_Upper'] = bb_data['upper']
            df['BB_Middle'] = bb_data['middle']
            df['BB_Lower'] = bb_data['lower']
        
        if len(df) >= 14:
            df['ATR'] = TechnicalIndicators.calculate_atr(df['High'], df['Low'], df['Close'])
            df['Volatility'] = df['Returns'].rolling(window=min(20, len(df)), min_periods=5).std() * np.sqrt(252) * 100
            df['ADX'] = TechnicalIndicators.calculate_adx(df['High'], df['Low'], df['Close'])
        
        if len(df) >= 20:
            df['Price_ZScore'] = TechnicalIndicators.calculate_zscore(df['Close'], min(20, len(df)))
            df['Returns_ZScore'] = TechnicalIndicators.calculate_zscore(df['Returns'], min(20, len(df)))
        
        # Fill any remaining NaN values
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return df
    
    def detect_rsi_divergence(self, df: pd.DataFrame, lookback: int = 30) -> Dict:
        """Detect RSI divergences"""
        if df is None or df.empty or len(df) < lookback or 'RSI' not in df.columns:
            return {'type': None, 'description': 'Insufficient data'}
        
        recent = df.tail(lookback)
        
        # Find price peaks and troughs
        price_peaks_idx = argrelextrema(recent['Close'].values, np.greater, order=3)[0]
        price_troughs_idx = argrelextrema(recent['Close'].values, np.less, order=3)[0]
        
        # Find RSI peaks and troughs
        rsi_peaks_idx = argrelextrema(recent['RSI'].values, np.greater, order=3)[0]
        rsi_troughs_idx = argrelextrema(recent['RSI'].values, np.less, order=3)[0]
        
        # Bullish divergence: Price making lower lows, RSI making higher lows
        if len(price_troughs_idx) >= 2 and len(rsi_troughs_idx) >= 2:
            last_price_trough = recent['Close'].iloc[price_troughs_idx[-1]]
            prev_price_trough = recent['Close'].iloc[price_troughs_idx[-2]]
            last_rsi_trough = recent['RSI'].iloc[rsi_troughs_idx[-1]]
            prev_rsi_trough = recent['RSI'].iloc[rsi_troughs_idx[-2]]
            
            if last_price_trough < prev_price_trough and last_rsi_trough > prev_rsi_trough:
                return {
                    'type': 'Bullish',
                    'description': f'Price lower low ({prev_price_trough:.2f} → {last_price_trough:.2f}), RSI higher low ({prev_rsi_trough:.1f} → {last_rsi_trough:.1f})',
                    'strength': 'Strong' if (last_rsi_trough - prev_rsi_trough) > 5 else 'Moderate'
                }
        
        # Bearish divergence: Price making higher highs, RSI making lower highs
        if len(price_peaks_idx) >= 2 and len(rsi_peaks_idx) >= 2:
            last_price_peak = recent['Close'].iloc[price_peaks_idx[-1]]
            prev_price_peak = recent['Close'].iloc[price_peaks_idx[-2]]
            last_rsi_peak = recent['RSI'].iloc[rsi_peaks_idx[-1]]
            prev_rsi_peak = recent['RSI'].iloc[rsi_peaks_idx[-2]]
            
            if last_price_peak > prev_price_peak and last_rsi_peak < prev_rsi_peak:
                return {
                    'type': 'Bearish',
                    'description': f'Price higher high ({prev_price_peak:.2f} → {last_price_peak:.2f}), RSI lower high ({prev_rsi_peak:.1f} → {last_rsi_peak:.1f})',
                    'strength': 'Strong' if (prev_rsi_peak - last_rsi_peak) > 5 else 'Moderate'
                }
        
        return {'type': None, 'description': 'No divergence detected'}
    
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
    
    def calculate_targets_stopless(self, entry_price: float, action: str, atr: float, sr_levels: Dict) -> Dict:
        """Calculate reasonable targets and stop-loss"""
        if action == 'BUY':
            stop_loss = entry_price - (1.5 * atr)
            target1 = entry_price + (2 * atr)
            target2 = entry_price + (3 * atr)
            
            # Adjust target based on resistance
            if sr_levels.get('resistance'):
                nearest_resistance = min(sr_levels['resistance'], key=lambda x: abs(x - entry_price))
                if nearest_resistance > entry_price and nearest_resistance < target2:
                    target1 = nearest_resistance * 0.98  # Just before resistance
        else:  # SELL
            stop_loss = entry_price + (1.5 * atr)
            target1 = entry_price - (2 * atr)
            target2 = entry_price - (3 * atr)
            
            # Adjust target based on support
            if sr_levels.get('support'):
                nearest_support = min(sr_levels['support'], key=lambda x: abs(x - entry_price))
                if nearest_support < entry_price and nearest_support > target2:
                    target1 = nearest_support * 1.02  # Just above support
        
        return {
            'stop_loss': stop_loss,
            'target1': target1,
            'target2': target2,
            'risk': abs(entry_price - stop_loss),
            'reward1': abs(target1 - entry_price),
            'reward2': abs(target2 - entry_price),
            'rr_ratio1': abs(target1 - entry_price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0,
            'rr_ratio2': abs(target2 - entry_price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
        }
    
    def generate_signals(self, df: pd.DataFrame, sr_levels: Dict, fib_levels: Dict) -> Dict:
        if df is None or df.empty:
            return {'signal': 'HOLD', 'confidence': 0, 'reasons': []}
        
        signals = []
        reasons = []
        
        current_price = df['Close'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        current_adx = df['ADX'].iloc[-1] if 'ADX' in df.columns else 0
        price_zscore = df['Price_ZScore'].iloc[-1] if 'Price_ZScore' in df.columns else 0
        
        if price_zscore < -2:
            signals.append(1)
            reasons.append(f"✓ Price Z-Score {price_zscore:.2f} (oversold)")
        elif price_zscore > 2:
            signals.append(-1)
            reasons.append(f"✗ Price Z-Score {price_zscore:.2f} (overbought)")
        
        if 'EMA_20' in df.columns and 'EMA_50' in df.columns:
            ema20 = df['EMA_20'].iloc[-1]
            ema50 = df['EMA_50'].iloc[-1]
            if not pd.isna(ema20) and not pd.isna(ema50):
                if current_price > ema20 > ema50:
                    signals.append(1)
                    reasons.append("✓ Price above EMA20 and EMA50 (Uptrend)")
                elif current_price < ema20 < ema50:
                    signals.append(-1)
                    reasons.append("✗ Price below EMA20 and EMA50 (Downtrend)")
        
        if current_rsi < 30:
            signals.append(1)
            reasons.append(f"✓ RSI oversold at {current_rsi:.1f}")
        elif current_rsi > 70:
            signals.append(-1)
            reasons.append(f"✗ RSI overbought at {current_rsi:.1f}")
        
        if current_adx > 25:
            if 'EMA_20' in df.columns and 'EMA_50' in df.columns:
                ema20 = df['EMA_20'].iloc[-1]
                ema50 = df['EMA_50'].iloc[-1]
                if not pd.isna(ema20) and not pd.isna(ema50):
                    if ema20 > ema50:
                        signals.append(1)
                        reasons.append(f"✓ Strong uptrend (ADX: {current_adx:.1f})")
                    else:
                        signals.append(-1)
                        reasons.append(f"✗ Strong downtrend (ADX: {current_adx:.1f})")
        
        if sr_levels.get('analysis'):
            nearest = sr_levels['analysis'][0]
            if nearest['type'] == 'Support' and abs(nearest['distance_pct']) < 2:
                signals.append(1)
                reasons.append(f"✓ Near strong support at {nearest['level']:.2f}")
            elif nearest['type'] == 'Resistance' and abs(nearest['distance_pct']) < 2:
                signals.append(-1)
                reasons.append(f"✗ Near strong resistance at {nearest['level']:.2f}")
        
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

def create_price_rsi_plot(df: pd.DataFrame, title: str, divergence_info: Dict) -> go.Figure:
    """Create price and RSI subplot with divergence annotation"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, subplot_titles=(f'{title} - Price', 'RSI'),
                        row_heights=[0.7, 0.3])
    
    # Price chart
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                  low=df['Low'], close=df['Close'], name='Price'))
    
    if fib_levels and 'levels' in fib_levels:
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']
        for idx, (level, price) in enumerate(fib_levels['levels'].items()):
            fig.add_hline(y=price, line_dash="dash", 
                         line_color=colors[idx % len(colors)],
                         annotation_text=f"Fib {level}: ₹{price:.2f}",
                         annotation_position="right")
    
    fig.update_layout(title=f'{title} - Fibonacci Levels', height=500,
                     xaxis_rangeslider_visible=False)
    return fig

def create_technical_indicators_plot(df: pd.DataFrame, title: str) -> go.Figure:
    """Create comprehensive technical indicators plot"""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=(f'{title} - Price & EMAs', 'MACD', 'RSI', 'ADX'),
                        row_heights=[0.4, 0.2, 0.2, 0.2])
    
    # Price and EMAs
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price',
                            line=dict(color='black', width=2)), row=1, col=1)
    
    for ema in [9, 20, 50]:
        if f'EMA_{ema}' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA_{ema}'], name=f'EMA{ema}',
                                    line=dict(width=1)), row=1, col=1)
    
    # MACD
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                                line=dict(color='blue', width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                                line=dict(color='red', width=1)), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram',
                            marker_color='gray'), row=2, col=1)
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                                line=dict(color='purple', width=2)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # ADX
    if 'ADX' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX',
                                line=dict(color='orange', width=2)), row=4, col=1)
        fig.add_hline(y=25, line_dash="dash", line_color="blue", row=4, col=1)
    
    fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
    return fig

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
                
                if df1 is not None and not df1.empty and len(df1) >= 50:
                    df1 = analyzer.calculate_indicators(df1)
                    sr_levels = analyzer.find_support_resistance(df1)
                    fib_levels = analyzer.calculate_fibonacci_levels(df1)
                    rsi_divergence = analyzer.detect_rsi_divergence(df1)
                    signals = analyzer.generate_signals(df1, sr_levels, fib_levels)
                    
                    all_results[f"{interval}_{period}"] = {
                        'ticker1': {
                            'data': df1,
                            'sr_levels': sr_levels,
                            'fib_levels': fib_levels,
                            'rsi_divergence': rsi_divergence,
                            'signals': signals
                        }
                    }
                
                if enable_ratio and ticker2:
                    current_analysis += 1
                    progress = current_analysis / total_analyses
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing {ticker2_name} - {interval}/{period} ({current_analysis}/{total_analyses})")
                    
                    df2 = analyzer.fetch_data_with_retry(ticker2, period, interval)
                    
                    if df2 is not None and not df2.empty and len(df2) >= 50:
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
            
            # Get latest data for targets/SL calculation
            latest_key = [k for k in results.keys() if '1d_' in k][0]
            latest_data = results[latest_key]['ticker1']['data']
            current_price = latest_data['Close'].iloc[-1]
            current_atr = latest_data['ATR'].iloc[-1]
            sr_levels_latest = results[latest_key]['ticker1']['sr_levels']
            
            # Calculate targets and stop-loss
            targets_sl = analyzer.calculate_targets_stopless(
                current_price, final_signal, current_atr, sr_levels_latest
            )
            
            st.markdown(f'<div class="{signal_class}">', unsafe_allow_html=True)
            st.markdown(f"### {signal_emoji} Final Recommendation: **{final_signal}**")
            st.markdown(f"**Confidence Level:** {avg_confidence:.1f}%")
            st.markdown(f"**Average Z-Score:** {avg_zscore:.2f}")
            
            if final_signal != "HOLD":
                st.markdown(f"\n**Entry Price:** ₹{current_price:.2f}")
                st.markdown(f"**Stop Loss:** ₹{targets_sl['stop_loss']:.2f} (Risk: ₹{targets_sl['risk']:.2f})")
                st.markdown(f"**Target 1:** ₹{targets_sl['target1']:.2f} (Reward: ₹{targets_sl['reward1']:.2f}, R:R = 1:{targets_sl['rr_ratio1']:.2f})")
                st.markdown(f"**Target 2:** ₹{targets_sl['target2']:.2f} (Reward: ₹{targets_sl['reward2']:.2f}, R:R = 1:{targets_sl['rr_ratio2']:.2f})")
            
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
            
            # Current Metrics
            st.markdown("---")
            st.markdown("### 📈 Current Market Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
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
            col5.metric("Price Z-Score", f"{price_zscore:.2f}")
            
            # RSI Divergence Analysis
            st.markdown("---")
            st.markdown("### 🔍 RSI Divergence Analysis")
            
            rsi_div = results[latest_key]['ticker1']['rsi_divergence']
            if rsi_div['type']:
                div_color = "green" if rsi_div['type'] == 'Bullish' else "red"
                st.markdown(f"**{rsi_div['type']} Divergence Detected ({rsi_div['strength']})**", unsafe_allow_html=True)
                st.markdown(f"<span style='color:{div_color}'>{rsi_div['description']}</span>", unsafe_allow_html=True)
            else:
                st.info(rsi_div['description'])
            
            # Paper Trading Section
            st.markdown("---")
            st.markdown("## 💼 Paper Trading Simulator")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Virtual Capital", f"₹{st.session_state.paper_capital:,.2f}")
            
            with col2:
                if st.button("Execute Recommended Trade", type="primary"):
                    if final_signal != "HOLD":
                        position_value = st.session_state.paper_capital * 0.1
                        quantity = max(1, int(position_value / current_price))
                        
                        strategy_details = {
                            'zscore': avg_zscore,
                            'volatility': current_vol,
                            'rsi': current_rsi,
                            'targets_sl': targets_sl
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
                        st.success(f"Stop Loss: ₹{targets_sl['stop_loss']:.2f} | Target: ₹{targets_sl['target1']:.2f}")
                    else:
                        st.info("Current recommendation is HOLD")
            
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
            
            # Auto-refresh for live monitoring
            if st.session_state.live_monitoring:
                open_trades = [t for t in st.session_state.paper_trades if t['status'] == 'OPEN']
                
                if open_trades and time.time() - st.session_state.last_refresh > 2:
                    st.session_state.last_refresh = time.time()
                    st.rerun()
            
            # Display positions
            if st.session_state.paper_trades:
                st.markdown("#### 📋 Active Positions")
                
                open_trades = [t for t in st.session_state.paper_trades if t['status'] == 'OPEN']
                
                if open_trades:
                    position_placeholder = st.empty()
                    
                    # Fetch latest price for P&L calculation
                    latest_df = analyzer.fetch_data_with_retry(st.session_state.ticker1_symbol, '1d', '5m')
                    if latest_df is not None and not latest_df.empty:
                        current_live_price = latest_df['Close'].iloc[-1]
                    else:
                        current_live_price = current_price
                    
                    position_data = []
                    for idx, trade in enumerate(open_trades):
                        if trade['quantity'] > 0:
                            if trade['action'] == 'BUY':
                                unrealized_pnl = (current_live_price - trade['price']) * trade['quantity']
                                unrealized_pnl_pct = ((current_live_price - trade['price']) / trade['price']) * 100
                            else:
                                unrealized_pnl = (trade['price'] - current_live_price) * trade['quantity']
                                unrealized_pnl_pct = ((trade['price'] - current_live_price) / trade['price']) * 100
                            
                            targets_info = trade['strategy']['targets_sl']
                            
                            position_data.append({
                                'Position': idx + 1,
                                'Ticker': trade['ticker'],
                                'Action': trade['action'],
                                'Entry': f"₹{trade['price']:.2f}",
                                'Current': f"₹{current_live_price:.2f}",
                                'Qty': trade['quantity'],
                                'Stop Loss': f"₹{targets_info['stop_loss']:.2f}",
                                'Target': f"₹{targets_info['target1']:.2f}",
                                'Unrealized P&L': f"₹{unrealized_pnl:,.2f}",
                                'P&L %': f"{unrealized_pnl_pct:+.2f}%",
                                'Entry Time': trade['entry_time']
                            })
                    
                    if position_data:
                        position_df = pd.DataFrame(position_data)
                        position_placeholder.dataframe(position_df, use_container_width=True)
                    
                    # Close buttons
                    cols = st.columns(len(open_trades))
                    for idx, trade in enumerate(open_trades):
                        with cols[idx]:
                            if st.button(f"Close Position #{idx+1}", key=f"close_{idx}"):
                                trade_idx = len(st.session_state.paper_trades) - len(open_trades) + idx
                                
                                if trade['quantity'] > 0:
                                    if trade['action'] == 'BUY':
                                        pnl = (current_live_price - trade['price']) * trade['quantity']
                                    else:
                                        pnl = (trade['price'] - current_live_price) * trade['quantity']
                                    
                                    st.session_state.paper_trades[trade_idx]['status'] = 'CLOSED'
                                    st.session_state.paper_trades[trade_idx]['exit_price'] = current_live_price
                                    st.session_state.paper_trades[trade_idx]['exit_time'] = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')
                                    st.session_state.paper_capital += pnl
                                    
                                    if len(open_trades) == 1:
                                        st.session_state.live_monitoring = False
                                    
                                    st.rerun()
                
                closed_trades = [t for t in st.session_state.paper_trades if t['status'] == 'CLOSED']
                if closed_trades:
                    st.markdown("#### 📊 Closed Positions")
                    closed_df_data = []
                    
                    for trade in closed_trades:
                        if trade['quantity'] > 0:
                            if trade['action'] == 'BUY':
                                pnl = (trade['exit_price'] - trade['price']) * trade['quantity']
                            else:
                                pnl = (trade['price'] - trade['exit_price']) * trade['quantity']
                            
                            pnl_pct = (pnl / trade['value']) * 100
                            
                            closed_df_data.append({
                                'Ticker': trade['ticker'],
                                'Action': trade['action'],
                                'Entry': f"₹{trade['price']:.2f}",
                                'Exit': f"₹{trade['exit_price']:.2f}",
                                'Qty': trade['quantity'],
                                'Entry Time': trade['entry_time'],
                                'Exit Time': trade['exit_time'],
                                'P&L': f"₹{pnl:,.2f}",
                                'P&L %': f"{pnl_pct:+.2f}%"
                            })
                    
                    if closed_df_data:
                        st.dataframe(pd.DataFrame(closed_df_data), use_container_width=True)
            
            # Visualization Section
            st.markdown("---")
            st.markdown("## 📊 Advanced Visualization & Analysis")
            
            viz_tabs = st.tabs(["Price & RSI", "Ratio Analysis", "Volatility", "Fibonacci", "Technical Indicators"])
            
            with viz_tabs[0]:
                st.markdown("### Price and RSI with Divergence Detection")
                daily_key = [k for k in results.keys() if '1d_' in k][0]
                df_daily = results[daily_key]['ticker1']['data']
                rsi_div_daily = results[daily_key]['ticker1']['rsi_divergence']
                
                fig_rsi = create_price_rsi_plot(df_daily.tail(100), st.session_state.ticker1_name, rsi_div_daily)
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                if rsi_div_daily['type']:
                    st.success(f"**{rsi_div_daily['type']} Divergence ({rsi_div_daily['strength']})**")
                    st.write(rsi_div_daily['description'])
                else:
                    st.info("No RSI divergence detected in the current timeframe")
            
            with viz_tabs[1]:
                if enable_ratio and ticker2:
                    st.markdown(f"### {st.session_state.ticker1_name} / {st.session_state.ticker2_name} Ratio Analysis")
                    ratio_key = [k for k in results.keys() if 'ticker2' in results[k] and '1d_' in k]
                    if ratio_key:
                        df1_ratio = results[ratio_key[0]]['ticker1']['data']
                        df2_ratio = results[ratio_key[0]]['ticker2']['data']
                        
                        fig_ratio = create_ratio_plot(df1_ratio, df2_ratio, st.session_state.ticker1_name, st.session_state.ticker2_name)
                        st.plotly_chart(fig_ratio, use_container_width=True)
                        
                        common_index = df1_ratio.index.intersection(df2_ratio.index)
                        ratio = df1_ratio.loc[common_index, 'Close'] / df2_ratio.loc[common_index, 'Close']
                        current_ratio = ratio.iloc[-1]
                        ratio_mean = ratio.mean()
                        ratio_std = ratio.std()
                        
                        st.write(f"**Current Ratio:** {current_ratio:.4f}")
                        st.write(f"**Mean Ratio:** {ratio_mean:.4f}")
                        st.write(f"**Std Dev:** {ratio_std:.4f}")
                        
                        if current_ratio > ratio_mean + ratio_std:
                            st.warning(f"{st.session_state.ticker1_name} is expensive relative to {st.session_state.ticker2_name}")
                        elif current_ratio < ratio_mean - ratio_std:
                            st.success(f"{st.session_state.ticker1_name} is cheap relative to {st.session_state.ticker2_name}")
                else:
                    st.info("Enable Ratio Analysis to see this chart")
            
            with viz_tabs[2]:
                st.markdown("### Price vs Volatility")
                fig_vol = create_volatility_plot(df_daily.tail(100), st.session_state.ticker1_name)
                st.plotly_chart(fig_vol, use_container_width=True)
                
                current_vol = df_daily['Volatility'].iloc[-1]
                avg_vol = df_daily['Volatility'].tail(100).mean()
                st.write(f"**Current Volatility:** {current_vol:.2f}%")
                st.write(f"**Average Volatility (100 days):** {avg_vol:.2f}%")
                
                if current_vol > avg_vol * 1.5:
                    st.warning("⚠️ Volatility is significantly higher than average - exercise caution")
                elif current_vol < avg_vol * 0.7:
                    st.info("📉 Volatility is lower than average - market is calm")
            
            with viz_tabs[3]:
                st.markdown("### Fibonacci Retracement Levels")
                fib_daily = results[daily_key]['ticker1']['fib_levels']
                
                fig_fib = create_fibonacci_plot(df_daily.tail(100), fib_daily, st.session_state.ticker1_name)
                st.plotly_chart(fig_fib, use_container_width=True)
                
                if fib_daily and 'closest_level' in fib_daily:
                    st.write(f"**Closest Fibonacci Level:** {fib_daily['closest_level']} at ₹{fib_daily['closest_price']:.2f}")
                    distance = abs((fib_daily['closest_price'] - current_price) / current_price * 100)
                    st.write(f"**Distance:** {distance:.2f}%")
                    
                    if fib_daily['closest_level'] in ['0.618', '0.5', '0.382']:
                        st.success("✅ Near key Fibonacci level - potential bounce/reversal zone")
            
            with viz_tabs[4]:
                st.markdown("### Complete Technical Indicators")
                fig_tech = create_technical_indicators_plot(df_daily.tail(100), st.session_state.ticker1_name)
                st.plotly_chart(fig_tech, use_container_width=True)
                
                st.write("**Current Technical Status:**")
                st.write(f"- RSI: {current_rsi:.1f} ({'Oversold' if current_rsi < 30 else 'Overbought' if current_rsi > 70 else 'Neutral'})")
                st.write(f"- ADX: {current_adx:.1f} ({'Strong trend' if current_adx > 25 else 'Weak trend'})")
                
                if 'MACD' in df_daily.columns:
                    macd_val = df_daily['MACD'].iloc[-1]
                    macd_signal = df_daily['MACD_Signal'].iloc[-1]
                    st.write(f"- MACD: {'Bullish' if macd_val > macd_signal else 'Bearish'} ({macd_val:.2f})")

    else:
        st.info("👆 Configure your analysis and click 'Start Complete Analysis'")

if __name__ == "__main__":
    main()
    df['High'],low=df['Low'], close=df['Close'], name='Price'),row=1, col=1)
    
    if 'EMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='EMA20',
                                line=dict(color='orange', width=1)), row=1, col=1)
    
    # RSI chart
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                                line=dict(color='purple', width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Add divergence annotation
    if divergence_info['type']:
        fig.add_annotation(text=f"<b>{divergence_info['type']} Divergence</b><br>{divergence_info['description']}",
                          xref="paper", yref="paper", x=0.5, y=1.15,
                          showarrow=False, font=dict(size=12, color="red" if divergence_info['type'] == 'Bearish' else "green"),
                          bgcolor="rgba(255,255,255,0.8)")
    
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, showlegend=True)
    return fig

def create_ratio_plot(df1: pd.DataFrame, df2: pd.DataFrame, ticker1_name: str, ticker2_name: str) -> go.Figure:
    """Create ratio plot"""
    common_index = df1.index.intersection(df2.index)
    ratio = df1.loc[common_index, 'Close'] / df2.loc[common_index, 'Close']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=common_index, y=ratio, name=f'{ticker1_name}/{ticker2_name}',
                            line=dict(color='blue', width=2)))
    
    # Add moving average
    ratio_ma = ratio.rolling(window=20).mean()
    fig.add_trace(go.Scatter(x=common_index, y=ratio_ma, name='20-MA',
                            line=dict(color='orange', width=1, dash='dash')))
    
    fig.update_layout(title=f'{ticker1_name} / {ticker2_name} Ratio', height=400,
                     xaxis_title='Date', yaxis_title='Ratio')
    return fig

def create_volatility_plot(df: pd.DataFrame, title: str) -> go.Figure:
    """Create volatility plot"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f'{title} - Price', 'Volatility'),
                        row_heights=[0.6, 0.4])
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price',
                            line=dict(color='blue', width=2)), row=1, col=1)
    
    if 'Volatility' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['Volatility'], name='Volatility (%)',
                                line=dict(color='red', width=2), fill='tozeroy'), row=2, col=1)
    
    fig.update_layout(height=500, showlegend=True)
    return fig

def create_fibonacci_plot(df: pd.DataFrame, fib_levels: Dict, title: str) -> go.Figure:
    """Create Fibonacci retracement plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=
