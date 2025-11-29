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

class TechnicalAnalyzer:
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
                       trading_style: str) -> TradingSignal:
        """Generate final trading signal with risk management"""
        
        # Execute strategy
        strategy_func = self.strategies[strategy_name]
        result = strategy_func(df)
        
        score = result['score']
        signals = result['signals']
        
        latest = df.iloc[-1]
        atr = latest['ATR']
        close = latest['Close']
        
        # Determine action and confidence
        if score > 2:
            action = "BUY"
            confidence = min(score / 5 * 100, 95)
        elif score < -2:
            action = "SELL"
            confidence = min(abs(score) / 5 * 100, 95)
        else:
            action = "HOLD"
            confidence = 50 - abs(score) * 10
        
        # Calculate risk management levels
        if action == "BUY":
            entry_price = close
            # Support-based stop loss
            stop_loss = max(latest['Support'], close - (2 * atr))
            # Resistance-based target
            target_price = min(latest['Resistance'], close + (3 * atr))
        elif action == "SELL":
            entry_price = close
            # Resistance-based stop loss
            stop_loss = min(latest['Resistance'], close + (2 * atr))
            # Support-based target
            target_price = max(latest['Support'], close - (3 * atr))
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
        
        # Generate reasoning
        reasoning = self._generate_reasoning(df, action, signals, 
                                             strategy_name, trading_style)
        
        return TradingSignal(
            action=action,
            confidence=confidence,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            timeframe="Combined",
            strategy=strategy_name.replace('_', ' ').title(),
            reasoning=reasoning,
            risk_reward=risk_reward
        )
    
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
        
        with st.spinner("🎯 Generating trading signals..."):
            # Generate signals
            strategy_engine = StrategyEngine()
            best_strategy = strategy_engine.select_best_strategy(df, trading_style)
            signal = strategy_engine.generate_signal(df, best_strategy, trading_style)
            
            # Store in session state
            st.session_state.analysis_results = {
                'df': df,
                'signal': signal,
                'strategy': best_strategy,
                'ticker': ticker,
                'instrument': instrument
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
        
        # News Section
        st.markdown("---")
        st.subheader("📰 Latest News")
        
        news = df.attrs.get('news', [])
        if news:
            for item in news:
                with st.expander(f"📌 {item.get('title', 'News Item')}"):
                    st.write(f"**Publisher:** {item.get('publisher', 'Unknown')}")
                    st.write(f"**Link:** [{item.get('link', '#')}]({item.get('link', '#')})")
                    if 'providerPublishTime' in item:
                        pub_time = datetime.fromtimestamp(item['providerPublishTime'])
                        st.write(f"**Published:** {pub_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.info("No recent news available")
        
        # Analysis Reasoning
        st.markdown("---")
        st.subheader("📊 Analysis Summary")
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
