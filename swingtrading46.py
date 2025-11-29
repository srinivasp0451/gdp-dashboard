import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import pytz
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Install vaderSentiment if not available
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "vaderSentiment"])
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Page Configuration
st.set_page_config(
    page_title="Professional Trading System",
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
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .signal-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
        font-size: 1.2rem;
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
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Data Classes
@dataclass
class TradingSignal:
    action: str  # BUY, SELL, HOLD
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    timeframe: str
    strategy: str
    reasoning: str
    risk_reward: float
    sentiment_score: float = 0
    sentiment_summary: str = ""
    strong_support: float = 0
    strong_resistance: float = 0
    support_strength: str = ""
    resistance_strength: str = ""
    zscore: float = 0
    zscore_interpretation: str = ""
    timeframe_signals: Dict = None
    detailed_summary: str = ""
    signal_confluence: str = ""

@dataclass
class BacktestResult:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_profit: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    total_return: float

# Instrument Mappings
INSTRUMENTS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "USD/INR": "INR=X",
    "EUR/USD": "EURUSD=X",
    "Custom Ticker": "CUSTOM"
}

TIMEFRAMES = ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "2h", "4h", "1d"]
PERIODS = ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "15y", "20y"]

# Session State Initialization
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

class SentimentAnalyzer:
    """News Sentiment Analysis using VADER"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.vader = SentimentIntensityAnalyzer()
    
    def analyze(self) -> Dict:
        """Analyze news sentiment"""
        try:
            news = yf.Ticker(self.ticker).news
            
            if not news:
                return {"score": 0, "summary": "No Recent News", "details": []}
            
            scores = []
            news_details = []
            
            for n in news[:5]:
                try:
                    title = n.get('title', '')
                    if not title:
                        continue
                    
                    sentiment_score = self.vader.polarity_scores(title)['compound']
                    scores.append(sentiment_score)
                    
                    news_details.append({
                        'title': title,
                        'score': sentiment_score,
                        'link': n.get('link', '#')
                    })
                    
                except Exception as e:
                    continue
            
            if not scores:
                return {"score": 0, "summary": "No Valid News", "details": []}
            
            avg_score = sum(scores) / len(scores)
            
            # Determine sentiment
            if avg_score > 0.25:
                sentiment = "VERY POSITIVE (Strong Bullish Catalyst)"
            elif avg_score > 0.15:
                sentiment = "POSITIVE (Mild Bullish News)"
            elif avg_score < -0.25:
                sentiment = "VERY NEGATIVE (Strong Bearish Catalyst)"
            elif avg_score < -0.15:
                sentiment = "NEGATIVE (Mild Bearish News)"
            else:
                sentiment = "NEUTRAL (No Significant News Impact)"
            
            return {
                "score": avg_score,
                "summary": sentiment,
                "details": news_details
            }
            
        except Exception as e:
            return {"score": 0, "summary": f"Error analyzing sentiment: {str(e)}", "details": []}


class SupportResistanceAnalyzer:
    """Advanced Support and Resistance Analysis"""
    
    @staticmethod
    def find_strong_levels(df: pd.DataFrame) -> Dict:
        """Identify strong support and resistance levels"""
        
        if len(df) < 50:
            return {
                'support': df['Low'].min(),
                'resistance': df['High'].max(),
                'support_strength': 'Insufficient data',
                'resistance_strength': 'Insufficient data',
                'support_touches': 0,
                'resistance_touches': 0
            }
        
        # Use multiple timeframe approach
        close_prices = df['Close'].values
        high_prices = df['High'].values
        low_prices = df['Low'].values
        
        # Find pivot points
        window = 10
        resistance_levels = []
        support_levels = []
        
        # Identify local maxima and minima
        for i in range(window, len(df) - window):
            # Resistance (local maxima)
            if high_prices[i] == max(high_prices[i-window:i+window+1]):
                resistance_levels.append(high_prices[i])
            
            # Support (local minima)
            if low_prices[i] == min(low_prices[i-window:i+window+1]):
                support_levels.append(low_prices[i])
        
        current_price = close_prices[-1]
        
        # Find strongest support (closest below current price with most touches)
        supports_below = [s for s in support_levels if s < current_price]
        if supports_below:
            # Cluster nearby levels
            support_clusters = SupportResistanceAnalyzer._cluster_levels(supports_below)
            strong_support = max(support_clusters.keys())
            support_touches = support_clusters[strong_support]
        else:
            strong_support = df['Low'].min()
            support_touches = 1
        
        # Find strongest resistance (closest above current price with most touches)
        resistances_above = [r for r in resistance_levels if r > current_price]
        if resistances_above:
            resistance_clusters = SupportResistanceAnalyzer._cluster_levels(resistances_above)
            strong_resistance = min(resistance_clusters.keys())
            resistance_touches = resistance_clusters[strong_resistance]
        else:
            strong_resistance = df['High'].max()
            resistance_touches = 1
        
        # Calculate strength based on touches and volume
        volume_at_support = SupportResistanceAnalyzer._volume_at_level(
            df, strong_support, tolerance=0.02
        )
        volume_at_resistance = SupportResistanceAnalyzer._volume_at_level(
            df, strong_resistance, tolerance=0.02
        )
        
        # Strength interpretation
        support_strength = SupportResistanceAnalyzer._interpret_strength(
            support_touches, volume_at_support, "Support"
        )
        resistance_strength = SupportResistanceAnalyzer._interpret_strength(
            resistance_touches, volume_at_resistance, "Resistance"
        )
        
        return {
            'support': strong_support,
            'resistance': strong_resistance,
            'support_strength': support_strength,
            'resistance_strength': resistance_strength,
            'support_touches': support_touches,
            'resistance_touches': resistance_touches,
            'support_distance': ((current_price - strong_support) / current_price) * 100,
            'resistance_distance': ((strong_resistance - current_price) / current_price) * 100
        }
    
    @staticmethod
    def _cluster_levels(levels: List[float], tolerance: float = 0.02) -> Dict:
        """Cluster nearby price levels and count touches"""
        clusters = {}
        
        for level in levels:
            found_cluster = False
            for cluster_level in list(clusters.keys()):
                if abs(level - cluster_level) / cluster_level < tolerance:
                    clusters[cluster_level] += 1
                    found_cluster = True
                    break
            
            if not found_cluster:
                clusters[level] = 1
        
        return clusters
    
    @staticmethod
    def _volume_at_level(df: pd.DataFrame, level: float, tolerance: float = 0.02) -> float:
        """Calculate average volume when price was near this level"""
        mask = (df['Close'] >= level * (1 - tolerance)) & (df['Close'] <= level * (1 + tolerance))
        if mask.sum() > 0:
            return df.loc[mask, 'Volume'].mean()
        return df['Volume'].mean()
    
    @staticmethod
    def _interpret_strength(touches: int, volume: float, level_type: str) -> str:
        """Interpret the strength of support/resistance"""
        
        strength_score = touches * 2
        
        if touches >= 5:
            strength = "VERY STRONG"
            reason = f"Tested {touches} times and held. This is a critical {level_type.lower()} zone."
        elif touches >= 3:
            strength = "STRONG"
            reason = f"Tested {touches} times. Significant {level_type.lower()} level."
        elif touches >= 2:
            strength = "MODERATE"
            reason = f"Tested {touches} times. Established {level_type.lower()}."
        else:
            strength = "WEAK"
            reason = f"Only tested {touches} time. Unconfirmed {level_type.lower()}."
        
        importance = f"{strength} - {reason} High volume activity at this level increases its reliability."
        
        return importance


class ZScoreAnalyzer:
    """Z-Score Analysis for Mean Reversion and Extreme Conditions"""
    
    @staticmethod
    def calculate_zscore(df: pd.DataFrame, window: int = 20) -> Dict:
        """Calculate Z-Score and interpret its impact"""
        
        if len(df) < window:
            return {
                'current_zscore': 0,
                'interpretation': 'Insufficient data',
                'historical_impact': 'N/A',
                'future_outlook': 'N/A'
            }
        
        # Calculate rolling Z-Score
        close_prices = df['Close']
        rolling_mean = close_prices.rolling(window=window).mean()
        rolling_std = close_prices.rolling(window=window).std()
        
        zscore = (close_prices - rolling_mean) / rolling_std
        df['ZScore'] = zscore
        
        current_zscore = zscore.iloc[-1]
        
        # Historical analysis
        historical_analysis = ZScoreAnalyzer._analyze_historical_zscore(df, window)
        
        # Current interpretation
        if current_zscore > 2:
            interpretation = "EXTREMELY OVERBOUGHT"
            signal = "Strong mean reversion expected - consider selling"
        elif current_zscore > 1.5:
            interpretation = "OVERBOUGHT"
            signal = "Price extended above mean - potential pullback"
        elif current_zscore > 1:
            interpretation = "MODERATELY OVERBOUGHT"
            signal = "Price above average - watch for reversal"
        elif current_zscore < -2:
            interpretation = "EXTREMELY OVERSOLD"
            signal = "Strong mean reversion expected - consider buying"
        elif current_zscore < -1.5:
            interpretation = "OVERSOLD"
            signal = "Price extended below mean - potential bounce"
        elif current_zscore < -1:
            interpretation = "MODERATELY OVERSOLD"
            signal = "Price below average - watch for recovery"
        else:
            interpretation = "NEUTRAL"
            signal = "Price near historical average - no extreme condition"
        
        # Future outlook
        future_outlook = ZScoreAnalyzer._forecast_based_on_zscore(
            current_zscore, historical_analysis
        )
        
        return {
            'current_zscore': current_zscore,
            'interpretation': f"{interpretation} (Z={current_zscore:.2f})",
            'signal': signal,
            'historical_impact': historical_analysis,
            'future_outlook': future_outlook
        }
    
    @staticmethod
    def _analyze_historical_zscore(df: pd.DataFrame, window: int) -> str:
        """Analyze how Z-Score behaved historically"""
        
        zscore = df['ZScore'].dropna()
        
        if len(zscore) < 50:
            return "Limited historical data for Z-Score analysis"
        
        # Count extreme events
        extreme_high = (zscore > 2).sum()
        extreme_low = (zscore < -2).sum()
        
        # Check mean reversion success rate
        reversion_success = 0
        total_extremes = 0
        
        for i in range(len(zscore) - 10):
            if zscore.iloc[i] > 2:  # Overbought
                # Check if price reverted in next 10 periods
                future_prices = df['Close'].iloc[i+1:i+11]
                if future_prices.min() < df['Close'].iloc[i]:
                    reversion_success += 1
                total_extremes += 1
            
            elif zscore.iloc[i] < -2:  # Oversold
                future_prices = df['Close'].iloc[i+1:i+11]
                if future_prices.max() > df['Close'].iloc[i]:
                    reversion_success += 1
                total_extremes += 1
        
        success_rate = (reversion_success / total_extremes * 100) if total_extremes > 0 else 0
        
        analysis = f"Historical Z-Score analysis: {extreme_high} overbought events, "
        analysis += f"{extreme_low} oversold events. Mean reversion success rate: {success_rate:.1f}%. "
        
        if success_rate > 70:
            analysis += "Strong historical tendency to revert to mean."
        elif success_rate > 50:
            analysis += "Moderate mean reversion tendency."
        else:
            analysis += "Weak mean reversion - trending market."
        
        return analysis
    
    @staticmethod
    def _forecast_based_on_zscore(current_zscore: float, historical: str) -> str:
        """Forecast future price movement based on Z-Score"""
        
        if current_zscore > 2:
            forecast = "HIGH PROBABILITY of price decline in coming sessions. "
            forecast += "Z-Score above 2 indicates extreme deviation from mean. "
            forecast += "Historical data suggests mean reversion is likely. "
            forecast += "Consider taking profits or waiting for pullback before entering longs."
        
        elif current_zscore > 1:
            forecast = "MODERATE PROBABILITY of consolidation or mild pullback. "
            forecast += "Price is stretched but not at extreme levels. "
            forecast += "Could continue higher with momentum, but risk/reward favors caution."
        
        elif current_zscore < -2:
            forecast = "HIGH PROBABILITY of price recovery in coming sessions. "
            forecast += "Z-Score below -2 indicates extreme undervaluation relative to recent mean. "
            forecast += "Historical patterns suggest strong bounce potential. "
            forecast += "Risk/reward favors buying at these oversold levels."
        
        elif current_zscore < -1:
            forecast = "MODERATE PROBABILITY of upward move. "
            forecast += "Price below average presents opportunity. "
            forecast += "Could decline further, but statistical edge favors buyers."
        
        else:
            forecast = "NEUTRAL OUTLOOK - Price near equilibrium. "
            forecast += "No statistical edge from Z-Score. "
            forecast += "Rely on trend, momentum, and other technical indicators."
        
        return forecast
    """Advanced Technical Analysis Engine"""
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        df = df.copy()
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # ADX (Average Directional Index)
        df['ADX'] = TechnicalAnalyzer.calculate_adx(df)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Support and Resistance
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        
        return df
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift()),
            'lc': abs(low - close.shift())
        }).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx

class StrategyEngine:
    """Multi-Strategy Trading Engine"""
    
    def __init__(self):
        self.strategies = {
            'trend_following': self.trend_following_strategy,
            'mean_reversion': self.mean_reversion_strategy,
            'momentum': self.momentum_strategy,
            'breakout': self.breakout_strategy,
            'scalping': self.scalping_strategy
        }
    
    def analyze_multiple_timeframes(self, ticker: str, period: str, 
                                   trading_style: str) -> Dict:
        """Analyze multiple timeframes and generate signals"""
        
        timeframe_map = {
            'Scalping': ['1m', '5m', '15m'],
            'Day Trading': ['5m', '15m', '1h'],
            'Swing Trading': ['1h', '4h', '1d'],
            'Positional Trading': ['1d', '1d', '1d']  # Same for all
        }
        
        timeframes = timeframe_map.get(trading_style, ['15m', '1h', '4h'])
        timeframe_results = {}
        
        for tf in timeframes:
            try:
                time.sleep(1.5)  # Rate limiting
                df = DataFetcher.fetch_data(ticker, period, tf)
                
                if df.empty:
                    continue
                
                df = TechnicalAnalyzer.calculate_indicators(df)
                market_structure = self.detect_market_structure(df)
                strategy = self.select_best_strategy(df, trading_style)
                result = self.strategies[strategy](df)
                
                # Determine signal
                score = result['score']
                if score > 2:
                    signal = "BUY"
                elif score < -2:
                    signal = "SELL"
                else:
                    signal = "HOLD"
                
                timeframe_results[tf] = {
                    'signal': signal,
                    'score': score,
                    'strategy': strategy,
                    'market_structure': market_structure,
                    'signals_list': result['signals'][:3]  # Top 3 signals
                }
                
            except Exception as e:
                continue
        
        return timeframe_results
    
    def detect_market_structure(self, df: pd.DataFrame) -> str:
        """Detect current market structure"""
        if len(df) < 50:
            return 'ranging'
        
        latest = df.iloc[-1]
        sma_20 = latest['SMA_20']
        sma_50 = latest['SMA_50']
        adx = latest['ADX']
        atr = latest['ATR']
        close = latest['Close']
        
        # Volatility measure
        volatility = (atr / close) * 100
        
        # Trend strength
        if adx > 25 and sma_20 > sma_50:
            if volatility > 2:
                return 'strong_uptrend'
            return 'uptrend'
        elif adx > 25 and sma_20 < sma_50:
            if volatility > 2:
                return 'strong_downtrend'
            return 'downtrend'
        elif adx < 20:
            return 'ranging'
        else:
            return 'ranging'
    
    def select_best_strategy(self, df: pd.DataFrame, trading_style: str) -> str:
        """Select optimal strategy based on market structure"""
        market_structure = self.detect_market_structure(df)
        
        # Strategy selection based on market and trading style
        if trading_style == "Scalping":
            return 'scalping'
        elif trading_style == "Day Trading":
            if market_structure in ['strong_uptrend', 'strong_downtrend']:
                return 'momentum'
            elif market_structure == 'ranging':
                return 'mean_reversion'
            else:
                return 'breakout'
        elif trading_style == "Swing Trading":
            if market_structure in ['uptrend', 'downtrend']:
                return 'trend_following'
            else:
                return 'breakout'
        else:  # Positional
            return 'trend_following'
    
    def trend_following_strategy(self, df: pd.DataFrame) -> Dict:
        """Trend Following Strategy"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        score = 0
        signals = []
        
        # Moving Average Alignment
        if latest['SMA_20'] > latest['SMA_50'] > latest['SMA_200']:
            score += 2
            signals.append("Strong upward MA alignment")
        elif latest['SMA_20'] < latest['SMA_50'] < latest['SMA_200']:
            score -= 2
            signals.append("Strong downward MA alignment")
        
        # MACD
        if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            score += 1.5
            signals.append("MACD bullish crossover")
        elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            score -= 1.5
            signals.append("MACD bearish crossover")
        
        # ADX Trend Strength
        if latest['ADX'] > 25:
            signals.append(f"Strong trend (ADX: {latest['ADX']:.1f})")
            score = score * 1.2 if score != 0 else score
        
        # Price vs SMA
        if latest['Close'] > latest['SMA_20']:
            score += 0.5
        else:
            score -= 0.5
        
        return {'score': score, 'signals': signals}
    
    def mean_reversion_strategy(self, df: pd.DataFrame) -> Dict:
        """Mean Reversion Strategy"""
        latest = df.iloc[-1]
        score = 0
        signals = []
        
        # Bollinger Bands
        if latest['Close'] < latest['BB_Lower']:
            score += 2
            signals.append("Price below lower BB - oversold")
        elif latest['Close'] > latest['BB_Upper']:
            score -= 2
            signals.append("Price above upper BB - overbought")
        
        # RSI
        if latest['RSI'] < 30:
            score += 1.5
            signals.append(f"RSI oversold ({latest['RSI']:.1f})")
        elif latest['RSI'] > 70:
            score -= 1.5
            signals.append(f"RSI overbought ({latest['RSI']:.1f})")
        
        # Stochastic
        if latest['Stoch_K'] < 20:
            score += 1
            signals.append("Stochastic oversold")
        elif latest['Stoch_K'] > 80:
            score -= 1
            signals.append("Stochastic overbought")
        
        return {'score': score, 'signals': signals}
    
    def momentum_strategy(self, df: pd.DataFrame) -> Dict:
        """Momentum Strategy"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0
        signals = []
        
        # RSI Momentum
        if 50 < latest['RSI'] < 70:
            score += 1.5
            signals.append("Bullish RSI momentum zone")
        elif 30 < latest['RSI'] < 50:
            score -= 1.5
            signals.append("Bearish RSI momentum zone")
        
        # MACD Histogram
        if latest['MACD_Hist'] > prev['MACD_Hist'] > 0:
            score += 1
            signals.append("Increasing bullish MACD histogram")
        elif latest['MACD_Hist'] < prev['MACD_Hist'] < 0:
            score -= 1
            signals.append("Increasing bearish MACD histogram")
        
        # Volume
        if latest['Volume_Ratio'] > 1.5:
            signals.append("High volume confirmation")
            score = score * 1.3 if score != 0 else score
        
        # Price momentum
        price_change = (latest['Close'] - prev['Close']) / prev['Close'] * 100
        if price_change > 1:
            score += 1
            signals.append(f"Strong upward momentum ({price_change:.2f}%)")
        elif price_change < -1:
            score -= 1
            signals.append(f"Strong downward momentum ({price_change:.2f}%)")
        
        return {'score': score, 'signals': signals}
    
    def breakout_strategy(self, df: pd.DataFrame) -> Dict:
        """Breakout Strategy"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0
        signals = []
        
        # Resistance breakout
        if latest['Close'] > latest['Resistance'] and prev['Close'] <= prev['Resistance']:
            score += 2.5
            signals.append("Bullish resistance breakout")
        
        # Support breakdown
        if latest['Close'] < latest['Support'] and prev['Close'] >= prev['Support']:
            score -= 2.5
            signals.append("Bearish support breakdown")
        
        # Volume confirmation
        if latest['Volume_Ratio'] > 1.5:
            signals.append("Breakout with high volume")
            score = score * 1.4 if score != 0 else score
        
        # ATR expansion
        if len(df) > 20:
            atr_avg = df['ATR'].iloc[-20:-1].mean()
            if latest['ATR'] > atr_avg * 1.2:
                signals.append("Volatility expansion")
                score = score * 1.2 if score != 0 else score
        
        return {'score': score, 'signals': signals}
    
    def scalping_strategy(self, df: pd.DataFrame) -> Dict:
        """Scalping Strategy - Quick entries/exits"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0
        signals = []
        
        # Quick EMA crossovers
        if latest['EMA_12'] > latest['EMA_26'] and prev['EMA_12'] <= prev['EMA_26']:
            score += 2
            signals.append("Fast EMA crossover - bullish")
        elif latest['EMA_12'] < latest['EMA_26'] and prev['EMA_12'] >= prev['EMA_26']:
            score -= 2
            signals.append("Fast EMA crossover - bearish")
        
        # Stochastic quick signals
        if latest['Stoch_K'] < 30 and latest['Stoch_K'] > prev['Stoch_K']:
            score += 1.5
            signals.append("Stochastic turning up from oversold")
        elif latest['Stoch_K'] > 70 and latest['Stoch_K'] < prev['Stoch_K']:
            score -= 1.5
            signals.append("Stochastic turning down from overbought")
        
        # Tight price action
        bb_width = (latest['BB_Upper'] - latest['BB_Lower']) / latest['BB_Middle']
        if bb_width < 0.02:
            signals.append("Tight consolidation - breakout pending")
        
        return {'score': score, 'signals': signals}
    
    def generate_signal(self, df: pd.DataFrame, strategy_name: str, 
                       trading_style: str, ticker: str, 
                       timeframe_signals: Dict = None) -> TradingSignal:
        """Generate final trading signal with comprehensive analysis"""
        
        # Execute strategy
        strategy_func = self.strategies[strategy_name]
        result = strategy_func(df)
        
        score = result['score']
        signals = result['signals']
        
        latest = df.iloc[-1]
        atr = latest['ATR']
        close = latest['Close']
        
        # Sentiment Analysis
        sentiment_analyzer = SentimentAnalyzer(ticker)
        sentiment_result = sentiment_analyzer.analyze()
        
        # Adjust score based on sentiment
        sentiment_adjustment = sentiment_result['score'] * 1.5
        adjusted_score = score + sentiment_adjustment
        
        # Support/Resistance Analysis
        sr_analysis = SupportResistanceAnalyzer.find_strong_levels(df)
        
        # Z-Score Analysis
        zscore_analysis = ZScoreAnalyzer.calculate_zscore(df)
        
        # Adjust score based on Z-Score
        zscore = zscore_analysis['current_zscore']
        if zscore > 2:
            adjusted_score -= 1.5  # Overbought, reduce buy signals
        elif zscore < -2:
            adjusted_score += 1.5  # Oversold, boost buy signals
        
        # Determine action and confidence
        if adjusted_score > 2:
            action = "BUY"
            confidence = min(adjusted_score / 5 * 100, 95)
        elif adjusted_score < -2:
            action = "SELL"
            confidence = min(abs(adjusted_score) / 5 * 100, 95)
        else:
            action = "HOLD"
            confidence = 50 - abs(adjusted_score) * 10
        
        # Calculate risk management levels using S/R
        if action == "BUY":
            entry_price = close
            stop_loss = max(sr_analysis['support'] * 0.99, close - (2 * atr))
            target_price = min(sr_analysis['resistance'] * 1.01, close + (3 * atr))
        elif action == "SELL":
            entry_price = close
            stop_loss = min(sr_analysis['resistance'] * 1.01, close + (2 * atr))
            target_price = max(sr_analysis['support'] * 0.99, close - (3 * atr))
        else:
            entry_price = close
            stop_loss = close - (1.5 * atr)
            target_price = close + (1.5 * atr)
        
        # Calculate risk-reward ratio
        if action in ["BUY", "SELL"]:
            risk = abs(entry_price - stop_loss)
            reward = abs(target_price - entry_price)
            risk_reward = reward / risk if risk > 0 else 0
        else:
            risk_reward = 1.0
        
        # Generate comprehensive reasoning
        reasoning = self._generate_reasoning(df, action, signals, 
                                             strategy_name, trading_style)
        
        # Generate detailed summary
        detailed_summary = self._generate_detailed_summary(
            df, action, adjusted_score, sentiment_result, 
            sr_analysis, zscore_analysis, timeframe_signals
        )
        
        # Generate signal confluence explanation
        confluence_explanation = self._generate_confluence_explanation(
            timeframe_signals, action, trading_style
        )
        
        return TradingSignal(
            action=action,
            confidence=confidence,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            timeframe="Combined",
            strategy=strategy_name.replace('_', ' ').title(),
            reasoning=reasoning,
            risk_reward=risk_reward,
            sentiment_score=sentiment_result['score'],
            sentiment_summary=sentiment_result['summary'],
            strong_support=sr_analysis['support'],
            strong_resistance=sr_analysis['resistance'],
            support_strength=sr_analysis['support_strength'],
            resistance_strength=sr_analysis['resistance_strength'],
            zscore=zscore,
            zscore_interpretation=zscore_analysis['interpretation'],
            timeframe_signals=timeframe_signals,
            detailed_summary=detailed_summary,
            signal_confluence=confluence_explanation
        )
    
    def _generate_detailed_summary(self, df: pd.DataFrame, action: str, 
                                  score: float, sentiment: Dict, 
                                  sr_analysis: Dict, zscore_analysis: Dict,
                                  timeframe_signals: Dict) -> str:
        """Generate detailed 100-word summary with values"""
        
        latest = df.iloc[-1]
        prev_week = df.iloc[-5] if len(df) > 5 else df.iloc[0]
        
        price_change = ((latest['Close'] - prev_week['Close']) / prev_week['Close']) * 100
        
        summary = f"**Comprehensive Market Summary:**\n\n"
        
        # Past Structure
        summary += f"**Past 7 Days:** Price moved from ₹{prev_week['Close']:.2f} to ₹{latest['Close']:.2f} "
        summary += f"({price_change:+.2f}%). "
        
        if price_change > 5:
            summary += "Strong bullish momentum observed. "
        elif price_change < -5:
            summary += "Strong bearish pressure evident. "
        else:
            summary += "Consolidation phase with limited directional movement. "
        
        # Current Structure
        summary += f"\n\n**Current Structure:** RSI at {latest['RSI']:.1f} "
        if latest['RSI'] > 70:
            summary += "(overbought territory), "
        elif latest['RSI'] < 30:
            summary += "(oversold territory), "
        else:
            summary += "(neutral zone), "
        
        summary += f"ADX at {latest['ADX']:.1f} "
        if latest['ADX'] > 25:
            summary += "indicating strong trend. "
        else:
            summary += "suggesting weak/ranging market. "
        
        summary += f"Z-Score at {zscore_analysis['current_zscore']:.2f} "
        if abs(zscore_analysis['current_zscore']) > 2:
            summary += "shows extreme deviation from mean - high mean reversion probability. "
        else:
            summary += "indicates price near statistical equilibrium. "
        
        # Support/Resistance Context
        summary += f"\n\n**Key Levels:** Strong support at ₹{sr_analysis['support']:.2f} "
        summary += f"({sr_analysis['support_distance']:.2f}% below), "
        summary += f"resistance at ₹{sr_analysis['resistance']:.2f} "
        summary += f"({sr_analysis['resistance_distance']:.2f}% above). "
        
        # Sentiment
        summary += f"News sentiment is {sentiment['summary']} (score: {sentiment['score']:.2f}). "
        
        # Future Forecast
        summary += f"\n\n**Forecast:** {action} signal generated with {score:.1f} points. "
        
        if action == "BUY":
            summary += f"Expect upward move toward ₹{sr_analysis['resistance']:.2f} resistance. "
            summary += "Entry recommended near current levels with stop below support. "
        elif action == "SELL":
            summary += f"Expect downward move toward ₹{sr_analysis['support']:.2f} support. "
            summary += "Short positions viable with stop above resistance. "
        else:
            summary += "No clear directional bias. Wait for better setup. "
        
        summary += f"{zscore_analysis['future_outlook'][:100]}"
        
        return summary
    
    def _generate_confluence_explanation(self, timeframe_signals: Dict, 
                                        final_action: str, 
                                        trading_style: str) -> str:
        """Explain how different timeframes contributed to final decision"""
        
        if not timeframe_signals:
            return "Single timeframe analysis used for signal generation."
        
        explanation = f"**Multi-Timeframe Analysis for {trading_style}:**\n\n"
        
        buy_count = 0
        sell_count = 0
        hold_count = 0
        
        for tf, data in timeframe_signals.items():
            signal = data['signal']
            score = data['score']
            strategy = data['strategy'].replace('_', ' ').title()
            market = data['market_structure'].replace('_', ' ').title()
            
            if signal == "BUY":
                buy_count += 1
                emoji = "🟢"
            elif signal == "SELL":
                sell_count += 1
                emoji = "🔴"
            else:
                hold_count += 1
                emoji = "🟡"
            
            explanation += f"{emoji} **{tf} Timeframe:** {signal} signal (Score: {score:.2f})\n"
            explanation += f"   • Strategy: {strategy}\n"
            explanation += f"   • Market: {market}\n"
            explanation += f"   • Key Factors: {', '.join(data['signals_list'][:2])}\n\n"
        
        # Confluence Decision
        explanation += f"**Decision Logic:**\n"
        explanation += f"• BUY signals: {buy_count}/{len(timeframe_signals)}\n"
        explanation += f"• SELL signals: {sell_count}/{len(timeframe_signals)}\n"
        explanation += f"• HOLD signals: {hold_count}/{len(timeframe_signals)}\n\n"
        
        if final_action == "BUY":
            explanation += f"**Why BUY Will Work:** "
            if buy_count >= 2:
                explanation += f"Strong confluence across {buy_count} timeframes. "
                explanation += "Multiple timeframe alignment significantly increases probability of success. "
            else:
                explanation += "Primary timeframe shows strong bullish setup. "
            
            explanation += "Lower timeframes provide entry precision while higher timeframes confirm trend direction. "
            explanation += "Combined with positive sentiment and oversold conditions, risk/reward favors long positions."
        
        elif final_action == "SELL":
            explanation += f"**Why SELL Will Work:** "
            if sell_count >= 2:
                explanation += f"Strong confluence across {sell_count} timeframes. "
                explanation += "Multiple timeframe alignment creates high-confidence short setup. "
            else:
                explanation += "Primary timeframe shows strong bearish setup. "
            
            explanation += "Lower timeframes identify reversal points while higher timeframes confirm downtrend. "
            explanation += "Combined with negative sentiment and overbought readings, downside risk is elevated."
        
        else:
            explanation += f"**Why HOLD is Prudent:** "
            explanation += "Conflicting signals across timeframes indicate market indecision. "
            explanation += "Trading in uncertain conditions reduces edge and increases risk. "
            explanation += "Patience is key - wait for clearer multi-timeframe alignment before committing capital."
        
        return explanation
    
    def _generate_reasoning(self, df: pd.DataFrame, action: str, 
                           signals: List[str], strategy: str, 
                           trading_style: str) -> str:
        """Generate human-readable reasoning"""
        latest = df.iloc[-1]
        market_structure = self.detect_market_structure(df)
        
        reasoning = f"**Market Analysis ({trading_style})**\n\n"
        reasoning += f"• Market Structure: {market_structure.replace('_', ' ').title()}\n"
        reasoning += f"• Selected Strategy: {strategy.replace('_', ' ').title()}\n"
        reasoning += f"• Current Price: ₹{latest['Close']:.2f}\n\n"
        
        reasoning += "**Key Indicators:**\n"
        reasoning += f"• RSI: {latest['RSI']:.1f} "
        if latest['RSI'] < 30:
            reasoning += "(Oversold)\n"
        elif latest['RSI'] > 70:
            reasoning += "(Overbought)\n"
        else:
            reasoning += "(Neutral)\n"
        
        reasoning += f"• ADX: {latest['ADX']:.1f} "
        if latest['ADX'] > 25:
            reasoning += "(Strong Trend)\n"
        else:
            reasoning += "(Weak Trend)\n"
        
        reasoning += f"• Volume Ratio: {latest['Volume_Ratio']:.2f}x average\n\n"
        
        reasoning += "**Signal Triggers:**\n"
        for signal in signals[:5]:  # Top 5 signals
            reasoning += f"• {signal}\n"
        
        reasoning += f"\n**Recommendation: {action}**\n"
        
        # Psychology considerations
        reasoning += "\n**⚠️ Trading Psychology Reminder:**\n"
        if action == "BUY":
            reasoning += "• Don't chase the price - wait for your entry\n"
            reasoning += "• Set stop-loss BEFORE entering trade\n"
            reasoning += "• Fear of missing out (FOMO) clouds judgment\n"
        elif action == "SELL":
            reasoning += "• Don't panic sell - follow your plan\n"
            reasoning += "• Protect profits with trailing stops\n"
            reasoning += "• Greed can turn winners into losers\n"
        else:
            reasoning += "• Patience is a position - not every moment needs action\n"
            reasoning += "• Overtrading reduces profitability\n"
            reasoning += "• Wait for high-probability setups\n"
        
        return reasoning

class BacktestEngine:
    """Backtesting Engine for Strategy Validation"""
    
    @staticmethod
    def run_backtest(df: pd.DataFrame, strategy_name: str, 
                     initial_capital: float = 100000) -> BacktestResult:
        """Run backtest on historical data"""
        
        if len(df) < 100:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        strategy_engine = StrategyEngine()
        strategy_func = strategy_engine.strategies[strategy_name]
        
        trades = []
        capital = initial_capital
        position = None
        equity_curve = [initial_capital]
        
        for i in range(50, len(df) - 1):
            window_df = df.iloc[:i+1].copy()
            latest = window_df.iloc[-1]
            
            # Generate signal
            result = strategy_func(window_df)
            score = result['score']
            
            # Entry logic
            if position is None:
                if score > 2:  # Buy signal
                    position = {
                        'type': 'LONG',
                        'entry_price': latest['Close'],
                        'entry_idx': i,
                        'stop_loss': latest['Close'] - (2 * latest['ATR']),
                        'target': latest['Close'] + (3 * latest['ATR'])
                    }
                elif score < -2:  # Sell signal
                    position = {
                        'type': 'SHORT',
                        'entry_price': latest['Close'],
                        'entry_idx': i,
                        'stop_loss': latest['Close'] + (2 * latest['ATR']),
                        'target': latest['Close'] - (3 * latest['ATR'])
                    }
            
            # Exit logic
            elif position is not None:
                current_price = latest['Close']
                entry_price = position['entry_price']
                
                exit_trade = False
                exit_reason = None
                
                if position['type'] == 'LONG':
                    if current_price >= position['target']:
                        exit_trade = True
                        exit_reason = 'TARGET'
                    elif current_price <= position['stop_loss']:
                        exit_trade = True
                        exit_reason = 'STOP_LOSS'
                    elif score < -1:
                        exit_trade = True
                        exit_reason = 'SIGNAL_REVERSAL'
                else:  # SHORT
                    if current_price <= position['target']:
                        exit_trade = True
                        exit_reason = 'TARGET'
                    elif current_price >= position['stop_loss']:
                        exit_trade = True
                        exit_reason = 'STOP_LOSS'
                    elif score > 1:
                        exit_trade = True
                        exit_reason = 'SIGNAL_REVERSAL'
                
                if exit_trade:
                    if position['type'] == 'LONG':
                        pnl = current_price - entry_price
                    else:
                        pnl = entry_price - current_price
                    
                    pnl_pct = (pnl / entry_price) * 100
                    capital += (capital * pnl_pct / 100)
                    
                    trades.append({
                        'entry': entry_price,
                        'exit': current_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'type': position['type'],
                        'reason': exit_reason
                    })
                    
                    position = None
            
            equity_curve.append(capital)
        
        # Calculate statistics
        if not trades:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = len([t for t in trades if t['pnl'] <= 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        profits = [t['pnl_pct'] for t in trades if t['pnl'] > 0]
        losses = [abs(t['pnl_pct']) for t in trades if t['pnl'] <= 0]
        
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        
        total_profit = sum(profits)
        total_loss = sum(losses)
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
        
        # Max drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100
        max_drawdown = abs(np.min(drawdown))
        
        total_return = ((capital - initial_capital) / initial_capital) * 100
        
        return BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            total_return=total_return
        )

class DataFetcher:
    """Handles data fetching with rate limiting"""
    
    @staticmethod
    def fetch_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch data with rate limiting and error handling"""
        
        cache_key = f"{ticker}_{period}_{interval}"
        current_time = time.time()
        
        # Check cache (5 minutes validity)
        if cache_key in st.session_state.data_cache:
            cached_data, cache_time = st.session_state.data_cache[cache_key]
            if current_time - cache_time < 300:  # 5 minutes
                return cached_data
        
        # Rate limiting
        if cache_key in st.session_state.last_fetch_time:
            time_since_last = current_time - st.session_state.last_fetch_time[cache_key]
            if time_since_last < 1.5:
                time.sleep(1.5 - time_since_last)
        
        try:
            # Fetch data
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                st.error(f"No data available for {ticker}")
                return pd.DataFrame()
            
            # Convert to IST
            if df.index.tz is not None:
                df.index = df.index.tz_convert('Asia/Kolkata')
            else:
                df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
            
            # Fetch news
            try:
                news = stock.news[:5] if hasattr(stock, 'news') else []
                df.attrs['news'] = news
            except:
                df.attrs['news'] = []
            
            # Cache data
            st.session_state.data_cache[cache_key] = (df, current_time)
            st.session_state.last_fetch_time[cache_key] = current_time
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()

def plot_advanced_chart(df: pd.DataFrame, signal: TradingSignal):
    """Create advanced trading chart with indicators"""
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=('Price & Moving Averages', 'MACD', 'RSI', 'Volume'),
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ), row=1, col=1)
    
    # Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
                            line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
                            line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200',
                            line=dict(color='purple', width=1)), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                            line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                            line=dict(color='gray', width=1, dash='dash'),
                            fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
    
    # Entry, Target, Stop Loss lines
    latest_time = df.index[-1]
    if signal.action in ["BUY", "SELL"]:
        fig.add_hline(y=signal.entry_price, line_dash="solid", line_color="blue",
                     annotation_text="Entry", row=1, col=1)
        fig.add_hline(y=signal.target_price, line_dash="dash", line_color="green",
                     annotation_text="Target", row=1, col=1)
        fig.add_hline(y=signal.stop_loss, line_dash="dash", line_color="red",
                     annotation_text="Stop Loss", row=1, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                            line=dict(color='blue', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                            line=dict(color='red', width=1)), row=2, col=1)
    
    colors = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram',
                        marker_color=colors), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                            line=dict(color='purple', width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.1, row=3, col=1)
    
    # Volume
    volume_colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] 
                    else 'red' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                        marker_color=volume_colors), row=4, col=1)
    
    # Layout
    fig.update_layout(
        title=f"Technical Analysis Chart - {signal.strategy}",
        xaxis_rangeslider_visible=False,
        height=1000,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    
    return fig

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Professional Multi-Timeframe Trading System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Instrument Selection
        instrument = st.selectbox("Select Instrument", list(INSTRUMENTS.keys()))
        
        if instrument == "Custom Ticker":
            custom_ticker = st.text_input("Enter Ticker Symbol", "RELIANCE.NS")
            ticker = custom_ticker
        else:
            ticker = INSTRUMENTS[instrument]
        
        st.markdown("---")
        
        # Trading Style
        trading_style = st.selectbox(
            "Trading Style",
            ["Day Trading", "Swing Trading", "Scalping", "Positional Trading"]
        )
        
        # Timeframe
        timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=4)
        
        # Period
        period = st.selectbox("Period", PERIODS, index=5)
        
        st.markdown("---")
        
        # API Rate Limiting
        api_delay = st.slider("API Delay (seconds)", 1.0, 5.0, 1.5, 0.5)
        
        st.markdown("---")
        
        # Fetch Data Button
        fetch_button = st.button("🔄 Fetch & Analyze", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.info("⚠️ **Risk Disclaimer**: This tool is for educational purposes. Always do your own research and consult with financial advisors.")
    
    # Main Content
    if fetch_button:
        with st.spinner("🔍 Fetching market data..."):
            # Fetch data with delay
            time.sleep(api_delay)
            df = DataFetcher.fetch_data(ticker, period, timeframe)
            
            if df.empty:
                st.error("Unable to fetch data. Please check the ticker symbol and try again.")
                return
            
            st.success(f"✅ Data fetched successfully! ({len(df)} candles)")
        
        with st.spinner("🧮 Calculating technical indicators..."):
            # Calculate indicators
            analyzer = TechnicalAnalyzer()
            df = analyzer.calculate_indicators(df)
        
        with st.spinner("📊 Analyzing multiple timeframes..."):
            # Multi-timeframe analysis
            strategy_engine = StrategyEngine()
            timeframe_signals = strategy_engine.analyze_multiple_timeframes(
                ticker, period, trading_style
            )
        
        with st.spinner("🎯 Generating trading signals..."):
            # Generate signals
            best_strategy = strategy_engine.select_best_strategy(df, trading_style)
            signal = strategy_engine.generate_signal(
                df, best_strategy, trading_style, ticker, timeframe_signals
            )
            
            # Store in session state
            st.session_state.analysis_results = {
                'df': df,
                'signal': signal,
                'strategy': best_strategy,
                'ticker': ticker,
                'instrument': instrument,
                'timeframe_signals': timeframe_signals
            }
        
        st.success("✅ Analysis complete!")
    
    # Display Results
    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        df = results['df']
        signal = results['signal']
        strategy = results['strategy']
        ticker = results['ticker']
        instrument = results['instrument']
        timeframe_signals = results.get('timeframe_signals', {})
        
        # Signal Box
        signal_class = {
            'BUY': 'buy-signal',
            'SELL': 'sell-signal',
            'HOLD': 'hold-signal'
        }[signal.action]
        
        st.markdown(f"""
        <div class="signal-box {signal_class}">
            🎯 Signal: {signal.action} | Confidence: {signal.confidence:.1f}% | Strategy: {signal.strategy}
        </div>
        """, unsafe_allow_html=True)
        
        # Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Current Price", f"₹{signal.entry_price:.2f}")
        
        with col2:
            target_change = ((signal.target_price - signal.entry_price) / signal.entry_price) * 100
            st.metric("Target Price", f"₹{signal.target_price:.2f}", 
                     f"{target_change:+.2f}%")
        
        with col3:
            sl_change = ((signal.stop_loss - signal.entry_price) / signal.entry_price) * 100
            st.metric("Stop Loss", f"₹{signal.stop_loss:.2f}", 
                     f"{sl_change:.2f}%")
        
        with col4:
            st.metric("Risk:Reward", f"1:{signal.risk_reward:.2f}")
        
        with col5:
            latest = df.iloc[-1]
            st.metric("RSI", f"{latest['RSI']:.1f}")
        
        # Detailed Summary
        st.markdown("---")
        st.subheader("📋 Detailed Market Summary")
        st.markdown(signal.detailed_summary)
        
        # Multi-Timeframe Confluence
        if timeframe_signals:
            st.markdown("---")
            st.subheader("⏱️ Multi-Timeframe Analysis & Signal Confluence")
            st.markdown(signal.signal_confluence)
        
        # Support & Resistance Analysis
        st.markdown("---")
        st.subheader("🎚️ Strong Support & Resistance Levels")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🟢 Strong Support Level**")
            st.metric("Support Price", f"₹{signal.strong_support:.2f}")
            st.info(f"**Analysis:** {signal.support_strength}")
        
        with col2:
            st.markdown("**🔴 Strong Resistance Level**")
            st.metric("Resistance Price", f"₹{signal.strong_resistance:.2f}")
            st.info(f"**Analysis:** {signal.resistance_strength}")
        
        # Z-Score Analysis
        st.markdown("---")
        st.subheader("📊 Z-Score Analysis (Mean Reversion)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            zscore_color = "green" if abs(signal.zscore) < 1 else ("orange" if abs(signal.zscore) < 2 else "red")
            st.markdown(f"**Current Z-Score:** <span style='color:{zscore_color};font-size:24px;font-weight:bold'>{signal.zscore:.2f}</span>", unsafe_allow_html=True)
            st.write(f"**Interpretation:** {signal.zscore_interpretation}")
        
        with col2:
            # Z-Score visualization
            z_fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = signal.zscore,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [-3, 3]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-3, -2], 'color': "lightgreen"},
                        {'range': [-2, -1], 'color': "lightblue"},
                        {'range': [-1, 1], 'color': "lightyellow"},
                        {'range': [1, 2], 'color': "lightcoral"},
                        {'range': [2, 3], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': signal.zscore
                    }
                },
                title = {'text': "Z-Score Gauge"}
            ))
            z_fig.update_layout(height=250)
            st.plotly_chart(z_fig, use_container_width=True)
        
        # Sentiment Analysis
        st.markdown("---")
        st.subheader("📰 News Sentiment Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            sentiment_color = "green" if signal.sentiment_score > 0.15 else ("red" if signal.sentiment_score < -0.15 else "gray")
            st.markdown(f"**Sentiment Score:** <span style='color:{sentiment_color};font-size:24px;font-weight:bold'>{signal.sentiment_score:.3f}</span>", unsafe_allow_html=True)
            st.write(f"**Summary:** {signal.sentiment_summary}")
        
        with col2:
            st.markdown("**Recent News Headlines:**")
            sentiment_analyzer = SentimentAnalyzer(ticker)
            sentiment_data = sentiment_analyzer.analyze()
            
            if sentiment_data.get('details'):
                for item in sentiment_data['details'][:3]:
                    score = item['score']
                    emoji = "🟢" if score > 0.1 else ("🔴" if score < -0.1 else "🟡")
                    st.markdown(f"{emoji} **{item['title']}** (Sentiment: {score:.2f})")
            else:
                st.info("No recent news available")
        
        # Analysis Reasoning
        st.markdown("---")
        st.subheader("📊 Technical Analysis Reasoning")
        st.markdown(signal.reasoning)
        
        # Chart
        st.markdown("---")
        st.subheader("📈 Technical Chart")
        chart = plot_advanced_chart(df, signal)
        st.plotly_chart(chart, use_container_width=True)
        
        # Backtesting Section
        st.markdown("---")
        st.subheader("🔬 Backtesting Results")
        
        with st.spinner("Running backtest..."):
            backtest_engine = BacktestEngine()
            backtest_result = backtest_engine.run_backtest(df, strategy)
        
        if backtest_result.total_trades > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Trades", backtest_result.total_trades)
                st.metric("Win Rate", f"{backtest_result.win_rate:.1f}%")
                st.metric("Total Return", f"{backtest_result.total_return:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Winning Trades", backtest_result.winning_trades)
                st.metric("Losing Trades", backtest_result.losing_trades)
                st.metric("Profit Factor", f"{backtest_result.profit_factor:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Avg Profit", f"{backtest_result.avg_profit:.2f}%")
                st.metric("Avg Loss", f"{backtest_result.avg_loss:.2f}%")
                st.metric("Max Drawdown", f"{backtest_result.max_drawdown:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Backtest interpretation
            st.markdown("---")
            st.subheader("🎓 Backtest Interpretation")
            
            if backtest_result.win_rate >= 60:
                st.success(f"✅ **Strong Strategy**: {backtest_result.win_rate:.1f}% win rate indicates reliable performance")
            elif backtest_result.win_rate >= 50:
                st.warning(f"⚠️ **Moderate Strategy**: {backtest_result.win_rate:.1f}% win rate - use with caution")
            else:
                st.error(f"❌ **Weak Strategy**: {backtest_result.win_rate:.1f}% win rate - not recommended")
            
            if backtest_result.profit_factor >= 2:
                st.success(f"✅ **Excellent Risk/Reward**: Profit factor of {backtest_result.profit_factor:.2f}")
            elif backtest_result.profit_factor >= 1.5:
                st.info(f"ℹ️ **Good Risk/Reward**: Profit factor of {backtest_result.profit_factor:.2f}")
            else:
                st.warning(f"⚠️ **Poor Risk/Reward**: Profit factor of {backtest_result.profit_factor:.2f}")
            
            if backtest_result.max_drawdown < 10:
                st.success(f"✅ **Low Risk**: Maximum drawdown of {backtest_result.max_drawdown:.2f}%")
            elif backtest_result.max_drawdown < 20:
                st.warning(f"⚠️ **Moderate Risk**: Maximum drawdown of {backtest_result.max_drawdown:.2f}%")
            else:
                st.error(f"❌ **High Risk**: Maximum drawdown of {backtest_result.max_drawdown:.2f}%")
        else:
            st.warning("⚠️ Not enough data for backtesting. Need at least 100 candles.")
        
        # Market Conditions
        st.markdown("---")
        st.subheader("🌐 Current Market Conditions")
        
        latest = df.iloc[-1]
        market_structure = strategy_engine.detect_market_structure(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Technical Indicators:**")
            st.write(f"• Market Structure: **{market_structure.replace('_', ' ').title()}**")
            st.write(f"• RSI: **{latest['RSI']:.2f}**")
            st.write(f"• MACD: **{latest['MACD']:.4f}**")
            st.write(f"• ADX: **{latest['ADX']:.2f}**")
            st.write(f"• ATR: **{latest['ATR']:.2f}**")
        
        with col2:
            st.markdown("**Price Levels:**")
            st.write(f"• Support: **₹{latest['Support']:.2f}**")
            st.write(f"• Resistance: **₹{latest['Resistance']:.2f}**")
            st.write(f"• BB Upper: **₹{latest['BB_Upper']:.2f}**")
            st.write(f"• BB Lower: **₹{latest['BB_Lower']:.2f}**")
            st.write(f"• Volume Ratio: **{latest['Volume_Ratio']:.2f}x**")
        
        # Trading Psychology Section
        st.markdown("---")
        st.subheader("🧠 Trading Psychology Guide")
        
        with st.expander("📚 Essential Trading Psychology Tips"):
            st.markdown("""
            **1. Emotional Control:**
            - Fear and greed are your biggest enemies
            - Stick to your trading plan, don't deviate
            - Accept that losses are part of trading
            
            **2. Risk Management:**
            - Never risk more than 1-2% per trade
            - Always use stop losses
            - Position sizing is crucial
            
            **3. Discipline:**
            - Wait for high-probability setups
            - Don't overtrade - patience is key
            - Keep a trading journal
            
            **4. Avoid Common Mistakes:**
            - Revenge trading after losses
            - Moving stop losses when price goes against you
            - Holding losing positions hoping for recovery
            - FOMO (Fear of Missing Out)
            
            **5. Success Mindset:**
            - Focus on the process, not just profits
            - Continuous learning and improvement
            - Accept responsibility for all trades
            - Stay humble and respect the market
            """)
        
        # Download Report
        st.markdown("---")
        
        report = f"""
TRADING ANALYSIS REPORT
Generated: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S IST')}

Instrument: {instrument} ({ticker})
Trading Style: {trading_style}
Timeframe: {timeframe}
Period: {period}

=== SIGNAL ===
Action: {signal.action}
Confidence: {signal.confidence:.1f}%
Strategy: {signal.strategy}

=== PRICE LEVELS ===
Entry Price: ₹{signal.entry_price:.2f}
Target Price: ₹{signal.target_price:.2f}
Stop Loss: ₹{signal.stop_loss:.2f}
Risk:Reward: 1:{signal.risk_reward:.2f}

=== BACKTEST RESULTS ===
Total Trades: {backtest_result.total_trades}
Win Rate: {backtest_result.win_rate:.1f}%
Total Return: {backtest_result.total_return:.2f}%
Profit Factor: {backtest_result.profit_factor:.2f}
Max Drawdown: {backtest_result.max_drawdown:.2f}%

=== REASONING ===
{signal.reasoning}

DISCLAIMER: This is for educational purposes only. Not financial advice.
"""
        
        st.download_button(
            label="📥 Download Analysis Report",
            data=report,
            file_name=f"trading_analysis_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    else:
        # Welcome screen
        st.info("👈 Configure your settings in the sidebar and click 'Fetch & Analyze' to begin")
        
        st.markdown("""
        ### Welcome to the Professional Trading System
        
        This advanced trading application provides:
        
        ✅ **Multi-timeframe Analysis** - Analyze across different timeframes
        
        ✅ **Intelligent Strategy Selection** - Automatically selects best strategy based on market structure
        
        ✅ **Comprehensive Indicators** - RSI, MACD, Bollinger Bands, ADX, Stochastic, and more
        
        ✅ **Backtesting Validation** - Verify signals with historical performance
        
        ✅ **Risk Management** - Clear entry, target, and stop-loss levels
        
        ✅ **News Integration** - Stay updated with latest market news
        
        ✅ **Psychology Guidance** - Overcome emotional trading pitfalls
        
        #### Supported Instruments:
        - Indian Indices (NIFTY 50, Bank NIFTY, SENSEX)
        - Cryptocurrencies (Bitcoin, Ethereum)
        - Commodities (Gold, Silver)
        - Forex (USD/INR, EUR/USD)
        - Custom stocks and tickers
        
        #### Trading Styles:
        - **Scalping**: Quick in-and-out trades (minutes)
        - **Day Trading**: Intraday positions (hours)
        - **Swing Trading**: Multi-day positions (days to weeks)
        - **Positional Trading**: Long-term positions (weeks to months)
        """)

if __name__ == "__main__":
    main()
