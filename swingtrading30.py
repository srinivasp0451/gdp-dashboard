"""
Streamlit AI Trading Agent - Advanced UI
No auto-refresh, persistent state, detailed trade logs
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


# Page config
st.set_page_config(
    page_title="AI Trading Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 5px;
        font-weight: bold;
    }
    .buy-signal {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .sell-signal {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .hold-signal {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@dataclass
class TradeSignal:
    symbol: str
    action: str
    signal_strength: float
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    timestamp: datetime
    strategy_type: str
    reasons: List[str]
    entry_date: datetime = None
    exit_date: datetime = None
    pnl_points: float = 0
    pnl_percent: float = 0


class TechnicalIndicators:
    """Custom technical indicators"""
    
    @staticmethod
    def sma(data: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(data).rolling(window=period).mean().values
    
    @staticmethod
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(data).ewm(span=period, adjust=False).mean().values
    
    @staticmethod
    def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        delta = pd.Series(data).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).values
    
    @staticmethod
    def macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple:
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple:
        sma = TechnicalIndicators.sma(data, period)
        std = pd.Series(data).rolling(window=period).std().values
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        tr[0] = high_low[0]
        return pd.Series(tr).rolling(window=period).mean().values
    
    @staticmethod
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple:
        lowest_low = pd.Series(low).rolling(window=period).min().values
        highest_high = pd.Series(high).rolling(window=period).max().values
        k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        k = pd.Series(k).rolling(window=3).mean().values
        d = pd.Series(k).rolling(window=3).mean().values
        return k, d
    
    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        high_diff = np.diff(high, prepend=high[0])
        low_diff = -np.diff(low, prepend=low[0])
        pos_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        neg_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        atr = TechnicalIndicators.atr(high, low, close, period)
        pos_di = 100 * pd.Series(pos_dm).rolling(window=period).mean().values / (atr + 1e-10)
        neg_di = 100 * pd.Series(neg_dm).rolling(window=period).mean().values / (atr + 1e-10)
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di + 1e-10)
        return pd.Series(dx).rolling(window=period).mean().values


class AITradingAgent:
    def __init__(self, initial_capital: float = 100000, risk_per_trade: float = 0.02):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.positions = {}
        self.trade_history = []
        self.ml_model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.ta = TechnicalIndicators()
        
    def fetch_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
        
        df.columns = [col.lower() for col in df.columns]
        
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            df['volume'] = 0
        
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values if 'volume' in df.columns else np.zeros_like(close)
        
        df['sma_20'] = self.ta.sma(close, 20)
        df['sma_50'] = self.ta.sma(close, 50)
        df['sma_200'] = self.ta.sma(close, 200)
        df['ema_12'] = self.ta.ema(close, 12)
        df['ema_26'] = self.ta.ema(close, 26)
        
        macd, macd_signal, macd_hist = self.ta.macd(close)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        df['rsi'] = self.ta.rsi(close, 14)
        
        bb_upper, bb_middle, bb_lower = self.ta.bollinger_bands(close, 20, 2)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
        
        df['atr'] = self.ta.atr(high, low, close, 14)
        df['atr_percent'] = df['atr'] / close * 100
        
        stoch_k, stoch_d = self.ta.stochastic(high, low, close, 14)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        df['adx'] = self.ta.adx(high, low, close, 14)
        
        df['volume_sma'] = self.ta.sma(volume, 20)
        df['volume_ratio'] = volume / (df['volume_sma'] + 1e-10)
        
        return df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
    
    def create_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ML features"""
        features = pd.DataFrame(index=df.index)
        
        features['returns_1d'] = df['close'].pct_change(1)
        features['returns_5d'] = df['close'].pct_change(5)
        features['returns_10d'] = df['close'].pct_change(10)
        features['volatility_5d'] = features['returns_1d'].rolling(5).std()
        features['volatility_20d'] = features['returns_1d'].rolling(20).std()
        
        features['rsi'] = df['rsi']
        features['rsi_change'] = df['rsi'].diff()
        features['macd_hist'] = df['macd_hist']
        features['adx'] = df['adx']
        features['bb_position'] = df['bb_position']
        features['atr_percent'] = df['atr_percent']
        
        features['sma_trend'] = (df['sma_20'] > df['sma_50']).astype(int)
        features['price_above_sma20'] = (df['close'] > df['sma_20']).astype(int)
        features['price_above_sma50'] = (df['close'] > df['sma_50']).astype(int)
        
        features['stoch_k'] = df['stoch_k']
        features['volume_ratio'] = df['volume_ratio']
        
        return features.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
    
    def train_ml_model(self, df: pd.DataFrame, lookforward: int = 5):
        """Train ML model"""
        df = self.calculate_technical_indicators(df)
        features_df = self.create_ml_features(df)
        
        df['future_return'] = df['close'].shift(-lookforward) / df['close'] - 1
        df['target'] = (df['future_return'] > 0.02).astype(int)
        
        X = features_df.iloc[:-lookforward]
        y = df['target'].iloc[:-lookforward]
        
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 100:
            return None
        
        X_scaled = self.scaler.fit_transform(X)
        
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                         min_samples_split=10, random_state=42)
        gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, 
                                             learning_rate=0.1, random_state=42)
        
        tscv = TimeSeriesSplit(n_splits=5)
        rf_scores = []
        gb_scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            rf_model.fit(X_train, y_train)
            gb_model.fit(X_train, y_train)
            
            rf_scores.append(rf_model.score(X_val, y_val))
            gb_scores.append(gb_model.score(X_val, y_val))
        
        if np.mean(rf_scores) > np.mean(gb_scores):
            self.ml_model = rf_model
        else:
            self.ml_model = gb_model
        
        self.ml_model.fit(X_scaled, y)
        self.feature_importance = dict(zip(X.columns, self.ml_model.feature_importances_))
        
        return self.ml_model
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate position size"""
        risk_amount = self.capital * self.risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        position_size = risk_amount / risk_per_share
        max_position_value = self.capital * 0.20
        position_size = min(position_size, max_position_value / entry_price)
        
        return int(position_size)
    
    def generate_live_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """Generate live signal"""
        if self.ml_model is None:
            self.train_ml_model(df)
        
        df = self.calculate_technical_indicators(df)
        features_df = self.create_ml_features(df)
        
        latest_features = features_df.iloc[-1:].values
        latest_scaled = self.scaler.transform(latest_features)
        
        ml_proba = self.ml_model.predict_proba(latest_scaled)[0]
        ml_confidence = ml_proba[1]
        
        latest = df.iloc[-1]
        current_price = latest['close']
        current_date = latest['date'] if 'date' in latest else datetime.now()
        
        reasons = []
        bull_score = 0
        bear_score = 0
        
        # Bullish conditions
        if 30 < latest['rsi'] < 70:
            bull_score += 1
            reasons.append(f"RSI neutral zone ({latest['rsi']:.1f})")
        
        if latest['macd'] > latest['macd_signal']:
            bull_score += 1
            reasons.append("MACD bullish crossover")
        
        if latest['close'] > latest['sma_20']:
            bull_score += 1
            reasons.append("Price above SMA20")
        
        if latest['close'] > latest['sma_50']:
            bull_score += 1
            reasons.append("Price above SMA50")
        
        if latest['adx'] > 25:
            bull_score += 1
            reasons.append(f"Strong trend (ADX: {latest['adx']:.1f})")
        
        if latest['stoch_k'] > latest['stoch_d'] and latest['stoch_k'] < 80:
            bull_score += 1
            reasons.append("Stochastic bullish")
        
        if ml_confidence > 0.6:
            bull_score += 2
            reasons.append(f"ML bullish prediction ({ml_confidence*100:.1f}%)")
        
        # Bearish conditions
        bear_reasons = []
        if latest['rsi'] > 70:
            bear_score += 1
            bear_reasons.append(f"RSI overbought ({latest['rsi']:.1f})")
        
        if latest['macd'] < latest['macd_signal']:
            bear_score += 1
            bear_reasons.append("MACD bearish")
        
        if latest['close'] < latest['sma_20']:
            bear_score += 1
            bear_reasons.append("Price below SMA20")
        
        if ml_confidence < 0.4:
            bear_score += 2
            bear_reasons.append(f"ML bearish prediction ({ml_confidence*100:.1f}%)")
        
        total_score = bull_score + bear_score
        bull_percentage = bull_score / total_score if total_score > 0 else 0
        
        atr = latest['atr']
        
        if bull_score >= 5 and bull_percentage > 0.65:
            stop_loss = current_price - (2 * atr)
            take_profit = current_price + (3 * atr)
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            return TradeSignal(
                symbol=symbol,
                action='BUY',
                signal_strength=bull_percentage,
                confidence=ml_confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                timestamp=datetime.now(),
                strategy_type='SWING',
                reasons=reasons,
                entry_date=current_date
            )
        
        elif bear_score >= 4:
            return TradeSignal(
                symbol=symbol,
                action='SELL',
                signal_strength=1 - bull_percentage,
                confidence=1 - ml_confidence,
                entry_price=current_price,
                stop_loss=0,
                take_profit=0,
                position_size=0,
                timestamp=datetime.now(),
                strategy_type='SWING',
                reasons=bear_reasons,
                entry_date=current_date
            )
        
        else:
            return TradeSignal(
                symbol=symbol,
                action='HOLD',
                signal_strength=0.5,
                confidence=0.5,
                entry_price=current_price,
                stop_loss=0,
                take_profit=0,
                position_size=0,
                timestamp=datetime.now(),
                strategy_type='SWING',
                reasons=["Mixed signals - waiting for clarity"],
                entry_date=current_date
            )
    
    def run_backtest(self, df: pd.DataFrame, symbol: str):
        """Run backtest"""
        train_size = int(len(df) * 0.7)
        train_data = df.iloc[:train_size].copy()
        
        self.train_ml_model(train_data)
        
        if self.ml_model is None:
            return []
        
        test_data = df.iloc[train_size:].copy()
        detailed_trades = []
        
        for i in range(50, len(test_data)):
            window_df = test_data.iloc[max(0, i-200):i+1].copy()
            signal = self.generate_live_signal(window_df, symbol)
            
            if signal.action == 'BUY':
                self.execute_trade(signal, detailed_trades)
            elif signal.action == 'SELL' and symbol in self.positions:
                self.close_position(signal, symbol, detailed_trades)
        
        # Close remaining positions
        if self.positions:
            for sym in list(self.positions.keys()):
                last_price = test_data.iloc[-1]['close']
                last_date = test_data.iloc[-1]['date'] if 'date' in test_data.iloc[-1] else datetime.now()
                close_signal = TradeSignal(
                    symbol=sym, action='SELL', signal_strength=1.0,
                    confidence=1.0, entry_price=last_price,
                    stop_loss=0, take_profit=0, position_size=0,
                    timestamp=datetime.now(), strategy_type='SWING',
                    reasons=["End of backtest"], entry_date=last_date
                )
                self.close_position(close_signal, sym, detailed_trades)
        
        return detailed_trades
    
    def execute_trade(self, signal: TradeSignal, detailed_trades: list):
        """Execute trade"""
        if signal.action == 'BUY' and signal.position_size > 0:
            cost = signal.entry_price * signal.position_size
            
            if cost <= self.capital * 0.95:
                self.positions[signal.symbol] = {
                    'quantity': signal.position_size,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'entry_date': signal.entry_date,
                    'reasons': signal.reasons,
                    'confidence': signal.confidence
                }
                self.capital -= cost
    
    def close_position(self, signal: TradeSignal, symbol: str, detailed_trades: list):
        """Close position"""
        if symbol in self.positions:
            position = self.positions[symbol]
            pnl_points = signal.entry_price - position['entry_price']
            pnl_percent = (pnl_points / position['entry_price']) * 100
            
            detailed_trades.append({
                'Entry Date': position['entry_date'],
                'Exit Date': signal.entry_date,
                'Symbol': symbol,
                'Action': 'BUY → SELL',
                'Entry Price': position['entry_price'],
                'Exit Price': signal.entry_price,
                'Stop Loss': position['stop_loss'],
                'Take Profit': position['take_profit'],
                'Position Size': position['quantity'],
                'PnL Points': pnl_points,
                'PnL %': pnl_percent,
                'Profit Probability': position['confidence'] * 100,
                'Reasons': ', '.join(position['reasons'][:3])
            })
            
            revenue = signal.entry_price * position['quantity']
            self.capital += revenue
            del self.positions[symbol]


# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'live_signal' not in st.session_state:
    st.session_state.live_signal = None


# Sidebar
st.sidebar.title("⚙️ Configuration")

# Predefined symbols
symbol_categories = {
    "Indian Indices": ["^NSEI", "^BSESN", "^NSEBANK", "NIFTY50.NS", "BANKNIFTY.NS"],
    "Indian Stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", 
                      "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS"],
    "Crypto": ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD"],
    "Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"],
    "US Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM"],
    "Custom": ["Enter custom symbol"]
}

category = st.sidebar.selectbox("📊 Select Category", list(symbol_categories.keys()))

if category == "Custom":
    symbol = st.sidebar.text_input("Enter Symbol", "AAPL")
else:
    symbol = st.sidebar.selectbox("Select Symbol", symbol_categories[category])

# Timeframe and Period
col1, col2 = st.sidebar.columns(2)
with col1:
    interval = st.selectbox("⏱️ Timeframe", 
                           ["1m", "5m", "10m", "15m", "30m", "1h", "4h", "1d"],
                           index=7)
with col2:
    period = st.selectbox("📅 Period",
                         ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "3y", 
                          "5y", "10y", "15y", "20y", "25y", "30y"],
                         index=6)

# Risk parameters
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Risk Management")
initial_capital = st.sidebar.number_input("Initial Capital", value=100000, step=10000)
risk_per_trade = st.sidebar.slider("Risk Per Trade (%)", 1, 5, 2) / 100

# Fetch Data Button
st.sidebar.markdown("---")
if st.sidebar.button("📥 FETCH DATA", type="primary"):
    with st.spinner(f"Fetching data for {symbol}..."):
        try:
            agent = AITradingAgent(initial_capital=initial_capital, risk_per_trade=risk_per_trade)
            df = agent.fetch_data(symbol, period=period, interval=interval)
            
            st.session_state.df = df
            st.session_state.agent = agent
            st.session_state.data_loaded = True
            st.session_state.backtest_results = None
            st.session_state.live_signal = None
            
            st.sidebar.success(f"✅ Loaded {len(df)} candles")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")

# Main Content
st.title("🤖 AI Trading Agent")
st.markdown("### Advanced Swing & Options Trading with Machine Learning")

if not st.session_state.data_loaded:
    st.info("👈 Configure settings and click 'FETCH DATA' to begin")
    st.markdown("""
    ### Features:
    - 🎯 **Live Trading Signals** with ML confidence
    - 📊 **Complete Backtesting** with detailed trade logs
    - 🔄 **Support for Stocks, Indices, Crypto, Forex**
    - 📈 **Multiple Timeframes** from 1m to 1d
    - 🛡️ **Advanced Risk Management**
    """)
else:
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Live Signal", "📈 Backtest", "📉 Data Overview"])
    
    # Tab 1: Live Signal
    with tab1:
        st.subheader(f"🎯 Live Trading Recommendation - {symbol}")
        
        if st.button("🔍 GENERATE LIVE SIGNAL", type="primary"):
            with st.spinner("Analyzing market conditions..."):
                try:
                    df = st.session_state.df
                    agent = st.session_state.agent
                    
                    # Train model
                    agent.train_ml_model(df)
                    
                    # Generate signal
                    signal = agent.generate_live_signal(df, symbol)
                    st.session_state.live_signal = signal
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Display live signal
        if st.session_state.live_signal:
            signal = st.session_state.live_signal
            
            # Signal card
            if signal.action == 'BUY':
                st.markdown('<div class="buy-signal">', unsafe_allow_html=True)
                st.markdown(f"### 🟢 BUY SIGNAL")
            elif signal.action == 'SELL':
                st.markdown('<div class="sell-signal">', unsafe_allow_html=True)
                st.markdown(f"### 🔴 SELL SIGNAL")
            else:
                st.markdown('<div class="hold-signal">', unsafe_allow_html=True)
                st.markdown(f"### 🟡 HOLD - Wait for Better Setup")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Entry Price", f"${signal.entry_price:.2f}")
            with col2:
                st.metric("Signal Strength", f"{signal.signal_strength*100:.1f}%")
            with col3:
                st.metric("ML Confidence", f"{signal.confidence*100:.1f}%")
            with col4:
                st.metric("Probability of Profit", f"{signal.confidence*100:.1f}%")
            
            if signal.action == 'BUY':
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Stop Loss", f"${signal.stop_loss:.2f}",
                             f"{((signal.stop_loss/signal.entry_price-1)*100):.2f}%")
                with col2:
                    st.metric("Take Profit", f"${signal.take_profit:.2f}",
                             f"{((signal.take_profit/signal.entry_price-1)*100):.2f}%")
                with col3:
                    risk_reward = (signal.take_profit - signal.entry_price) / (signal.entry_price - signal.stop_loss)
                    st.metric("Risk/Reward", f"1:{risk_reward:.2f}")
                with col4:
                    st.metric("Position Size", f"{signal.position_size} shares")
                
                st.markdown("---")
                st.markdown(f"**💰 Investment Required:** ${signal.entry_price * signal.position_size:,.2f}")
                st.markdown(f"**🎯 Potential Profit:** ${(signal.take_profit - signal.entry_price) * signal.position_size:,.2f}")
                st.markdown(f"**🛑 Maximum Risk:** ${(signal.entry_price - signal.stop_loss) * signal.position_size:,.2f}")
            
            st.markdown("---")
            st.markdown("### 📋 Reasoning & Logic")
            for i, reason in enumerate(signal.reasons, 1):
                st.markdown(f"**{i}.** {reason}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown(f"**⏰ Generated at:** {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Tab 2: Backtest
    with tab2:
        st.subheader(f"📈 Backtest Results - {symbol}")
        
        if st.button("🧪 RUN BACKTEST", type="primary"):
            with st.spinner("Running backtest... This may take a minute."):
                try:
                    df = st.session_state.df
                    agent = AITradingAgent(initial_capital=initial_capital, risk_per_trade=risk_per_trade)
                    
                    detailed_trades = agent.run_backtest(df, symbol)
                    st.session_state.backtest_results = {
                        'trades': detailed_trades,
                        'agent': agent
                    }
                    
                    st.success("✅ Backtest completed!")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Display backtest results
        if st.session_state.backtest_results:
            results = st.session_state.backtest_results
            trades_df = pd.DataFrame(results['trades'])
            agent = results['agent']
            
            if len(trades_df) > 0:
                # Performance metrics
                st.markdown("### 📊 Performance Metrics")
                
                total_return = ((agent.capital - agent.initial_capital) / agent.initial_capital) * 100
                winning_trades = trades_df[trades_df['PnL %'] > 0]
                losing_trades = trades_df[trades_df['PnL %'] < 0]
                
                win_rate = len(winning_trades) / len(trades_df) * 100
                avg_win = winning_trades['PnL %'].mean() if len(winning_trades) > 0 else 0
                avg_loss = losing_trades['PnL %'].mean() if len(losing_trades) > 0 else 0
                
                # Metrics row 1
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Return", f"{total_return:.2f}%")
                with col2:
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                with col3:
                    st.metric("Total Trades", len(trades_df))
                with col4:
                    st.metric("Winning Trades", len(winning_trades), delta=f"+{len(winning_trades)}")
                with col5:
                    st.metric("Losing Trades", len(losing_trades), delta=f"-{len(losing_trades)}")
                
                # Metrics row 2
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Avg Win", f"{avg_win:.2f}%")
                with col2:
                    st.metric("Avg Loss", f"{avg_loss:.2f}%")
                with col3:
                    profit_factor = abs(winning_trades['PnL %'].sum() / losing_trades['PnL %'].sum()) if len(losing_trades) > 0 else 0
                    st.metric("Profit Factor", f"{profit_factor:.2f}")
                with col4:
                    st.metric("Final Capital", f"${agent.capital:,.0f}")
                with col5:
                    total_pnl = trades_df['PnL %'].sum()
                    st.metric("Total PnL", f"{total_pnl:.2f}%")
                
                st.markdown("---")
                
                # Detailed trades table
                st.markdown("### 📝 Detailed Trade Log")
                
                # Format the dataframe
                display_df = trades_df.copy()
                
                # Format dates
                if 'Entry Date' in display_df.columns:
                    display_df['Entry Date'] = pd.to_datetime(display_df['Entry Date']).dt.strftime('%Y-%m-%d %H:%M')
                if 'Exit Date' in display_df.columns:
                    display_df['Exit Date'] = pd.to_datetime(display_df['Exit Date']).dt.strftime('%Y-%m-%d %H:%M')
                
                # Format prices
                for col in ['Entry Price', 'Exit Price', 'Stop Loss', 'Take Profit']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
                
                # Format percentages
                for col in ['PnL %', 'Profit Probability']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
                
                # Format PnL Points
                if 'PnL Points' in display_df.columns:
                    display_df['PnL Points'] = display_df['PnL Points'].apply(lambda x: f"{x:.2f}")
                
                # Style the dataframe
                def highlight_pnl(row):
                    pnl_str = row['PnL %']
                    pnl_val = float(pnl_str.replace('%', ''))
                    
                    if pnl_val > 0:
                        return ['background-color: #d4edda'] * len(row)
                    elif pnl_val < 0:
                        return ['background-color: #f8d7da'] * len(row)
                    else:
                        return [''] * len(row)
                
                styled_df = display_df.style.apply(highlight_pnl, axis=1)
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                # Download button
                csv = trades_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Trade Log (CSV)",
                    data=csv,
                    file_name=f"{symbol}_backtest_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                # Trade distribution
                st.markdown("---")
                st.markdown("### 📊 Trade Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # PnL histogram
                    st.markdown("#### PnL Distribution")
                    pnl_data = trades_df['PnL %'].values
                    
                    hist_df = pd.DataFrame({
                        'PnL %': pnl_data,
                        'Count': 1
                    })
                    
                    st.bar_chart(hist_df.set_index('PnL %'))
                
                with col2:
                    # Win/Loss pie chart
                    st.markdown("#### Win/Loss Ratio")
                    pie_data = pd.DataFrame({
                        'Type': ['Winning', 'Losing'],
                        'Count': [len(winning_trades), len(losing_trades)]
                    })
                    st.bar_chart(pie_data.set_index('Type'))
                
                # Best and worst trades
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🏆 Best Trade")
                    best_trade = trades_df.loc[trades_df['PnL %'].idxmax()]
                    st.success(f"""
                    **Entry:** {best_trade['Entry Date']}  
                    **Exit:** {best_trade['Exit Date']}  
                    **Entry Price:** {best_trade['Entry Price']}  
                    **Exit Price:** {best_trade['Exit Price']}  
                    **PnL:** {best_trade['PnL Points']} points ({best_trade['PnL %']})  
                    **Reason:** {best_trade['Reasons']}
                    """)
                
                with col2:
                    st.markdown("### 💔 Worst Trade")
                    worst_trade = trades_df.loc[trades_df['PnL %'].idxmin()]
                    st.error(f"""
                    **Entry:** {worst_trade['Entry Date']}  
                    **Exit:** {worst_trade['Exit Date']}  
                    **Entry Price:** {worst_trade['Entry Price']}  
                    **Exit Price:** {worst_trade['Exit Price']}  
                    **PnL:** {worst_trade['PnL Points']} points ({worst_trade['PnL %']})  
                    **Reason:** {worst_trade['Reasons']}
                    """)
            else:
                st.warning("No trades were executed during the backtest period.")
    
    # Tab 3: Data Overview
    with tab3:
        st.subheader(f"📉 Data Overview - {symbol}")
        
        df = st.session_state.df
        
        # Basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Candles", len(df))
        with col2:
            st.metric("Start Date", df['date'].iloc[0].strftime('%Y-%m-%d'))
        with col3:
            st.metric("End Date", df['date'].iloc[-1].strftime('%Y-%m-%d'))
        with col4:
            current_price = df['close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
        
        st.markdown("---")
        
        # Price chart
        st.markdown("### 📈 Price Chart")
        chart_data = df.set_index('date')[['close']]
        st.line_chart(chart_data)
        
        # Volume chart
        if df['volume'].sum() > 0:
            st.markdown("### 📊 Volume Chart")
            volume_data = df.set_index('date')[['volume']]
            st.bar_chart(volume_data)
        
        # Recent data
        st.markdown("### 📋 Recent Data (Last 10 candles)")
        recent_df = df.tail(10).copy()
        
        # Format for display
        display_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        recent_display = recent_df[display_cols].copy()
        
        for col in ['open', 'high', 'low', 'close']:
            recent_display[col] = recent_display[col].apply(lambda x: f"${x:.2f}")
        
        recent_display['volume'] = recent_display['volume'].apply(lambda x: f"{int(x):,}")
        
        st.dataframe(recent_display, use_container_width=True)
        
        # Statistics
        st.markdown("### 📊 Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Highest Price", f"${df['high'].max():.2f}")
        with col2:
            st.metric("Lowest Price", f"${df['low'].min():.2f}")
        with col3:
            volatility = df['close'].pct_change().std() * 100
            st.metric("Volatility", f"{volatility:.2f}%")
        with col4:
            avg_volume = df['volume'].mean()
            st.metric("Avg Volume", f"{int(avg_volume):,}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. 
    Past performance does not guarantee future results. Always do your own research and 
    consult with a financial advisor before making investment decisions.</p>
    <p>📚 Powered by Machine Learning | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)


# Instructions in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 How to Use:
1. Select category and symbol
2. Choose timeframe and period
3. Set risk parameters
4. Click **FETCH DATA**
5. Generate live signal or run backtest

### 🎯 Features:
- ✅ Live trading signals
- ✅ Complete backtesting
- ✅ Detailed trade logs
- ✅ ML-based predictions
- ✅ Risk management

### 💡 Tips:
- Use longer periods for better ML training
- Lower risk per trade for safer trading
- Check multiple timeframes
- Review backtest before live trading
""")

# Install instructions
if not YFINANCE_AVAILABLE:
    st.sidebar.error("""
    ### ⚠️ Installation Required
    ```bash
    pip install yfinance
    pip install pandas numpy
    pip install scikit-learn
    pip install streamlit
    ```
    """)
