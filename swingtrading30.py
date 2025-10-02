"""
Advanced AI Trading Agent - No TA-Lib Required
Features: Live signals, auto data fetch, handles indices without volume
"""

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
    print("⚠️  yfinance not installed. Install with: pip install yfinance")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class TradeSignal:
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    signal_strength: float
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    timestamp: datetime
    strategy_type: str
    reasons: List[str]


class TechnicalIndicators:
    """Custom technical indicators without TA-Lib"""
    
    @staticmethod
    def sma(data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        return pd.Series(data).rolling(window=period).mean().values
    
    @staticmethod
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        return pd.Series(data).ewm(span=period, adjust=False).mean().values
    
    @staticmethod
    def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        delta = pd.Series(data).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).values
    
    @staticmethod
    def macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple:
        """MACD indicator"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, period)
        std = pd.Series(data).rolling(window=period).std().values
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        tr[0] = high_low[0]  # First value
        
        atr = pd.Series(tr).rolling(window=period).mean().values
        return atr
    
    @staticmethod
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                   period: int = 14, smooth_k: int = 3) -> Tuple:
        """Stochastic Oscillator"""
        lowest_low = pd.Series(low).rolling(window=period).min().values
        highest_high = pd.Series(high).rolling(window=period).max().values
        
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k = pd.Series(k).rolling(window=smooth_k).mean().values
        d = pd.Series(k).rolling(window=3).mean().values
        
        return k, d
    
    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Average Directional Index"""
        high_diff = np.diff(high, prepend=high[0])
        low_diff = -np.diff(low, prepend=low[0])
        
        pos_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        neg_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        atr = TechnicalIndicators.atr(high, low, close, period)
        
        pos_di = 100 * pd.Series(pos_dm).rolling(window=period).mean().values / atr
        neg_di = 100 * pd.Series(neg_dm).rolling(window=period).mean().values / atr
        
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di + 1e-10)
        adx = pd.Series(dx).rolling(window=period).mean().values
        
        return adx
    
    @staticmethod
    def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """On Balance Volume - handles missing volume"""
        if volume is None or np.all(volume == 0):
            return np.zeros_like(close)
        
        direction = np.sign(np.diff(close, prepend=close[0]))
        obv = np.cumsum(direction * volume)
        return obv


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
        
    def fetch_data(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        if not YFINANCE_AVAILABLE:
            raise ImportError("Please install yfinance: pip install yfinance")
        
        print(f"📊 Fetching data for {symbol}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
        
        # Standardize column names
        df.columns = [col.lower() for col in df.columns]
        
        # Handle missing volume (common for indices)
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            print(f"⚠️  No volume data for {symbol} (likely an index)")
            df['volume'] = 0
        
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        
        print(f"✓ Loaded {len(df)} candles from {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values if 'volume' in df.columns else np.zeros_like(close)
        
        # Moving Averages
        df['sma_20'] = self.ta.sma(close, 20)
        df['sma_50'] = self.ta.sma(close, 50)
        df['sma_200'] = self.ta.sma(close, 200)
        df['ema_12'] = self.ta.ema(close, 12)
        df['ema_26'] = self.ta.ema(close, 26)
        df['ema_9'] = self.ta.ema(close, 9)
        
        # MACD
        macd, macd_signal, macd_hist = self.ta.macd(close)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # RSI
        df['rsi'] = self.ta.rsi(close, 14)
        df['rsi_sma'] = self.ta.sma(df['rsi'].values, 14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.ta.bollinger_bands(close, 20, 2)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
        
        # ATR (Volatility)
        df['atr'] = self.ta.atr(high, low, close, 14)
        df['atr_percent'] = df['atr'] / close * 100
        
        # Stochastic
        stoch_k, stoch_d = self.ta.stochastic(high, low, close, 14)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # ADX (Trend Strength)
        df['adx'] = self.ta.adx(high, low, close, 14)
        
        # Volume indicators (handle zero volume)
        has_volume = volume.sum() > 0
        if has_volume:
            df['obv'] = self.ta.obv(close, volume)
            df['volume_sma'] = self.ta.sma(volume, 20)
            df['volume_ratio'] = volume / (df['volume_sma'] + 1e-10)
        else:
            df['obv'] = 0
            df['volume_sma'] = 0
            df['volume_ratio'] = 1
        
        # Price momentum
        df['roc'] = ((close / np.roll(close, 10)) - 1) * 100
        df['mom'] = close - np.roll(close, 10)
        
        # Support/Resistance
        df['pivot'] = (high + low + close) / 3
        df['r1'] = 2 * df['pivot'] - low
        df['s1'] = 2 * df['pivot'] - high
        
        # Custom features
        df['price_to_sma20'] = close / (df['sma_20'] + 1e-10)
        df['price_to_sma50'] = close / (df['sma_50'] + 1e-10)
        df['price_change'] = close / np.roll(close, 1) - 1
        
        return df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
    
    def create_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for ML model"""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns_1d'] = df['close'].pct_change(1)
        features['returns_5d'] = df['close'].pct_change(5)
        features['returns_10d'] = df['close'].pct_change(10)
        features['returns_20d'] = df['close'].pct_change(20)
        
        # Volatility features
        features['volatility_5d'] = features['returns_1d'].rolling(5).std()
        features['volatility_20d'] = features['returns_1d'].rolling(20).std()
        
        # Technical indicator features
        features['rsi'] = df['rsi']
        features['rsi_change'] = df['rsi'].diff()
        features['rsi_normalized'] = (df['rsi'] - 50) / 50
        
        features['macd_hist'] = df['macd_hist']
        features['macd_hist_change'] = df['macd_hist'].diff()
        
        features['adx'] = df['adx']
        features['adx_change'] = df['adx'].diff()
        
        features['bb_position'] = df['bb_position']
        features['bb_width'] = df['bb_width']
        features['atr_percent'] = df['atr_percent']
        
        # Trend features
        features['sma_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        features['price_above_sma20'] = (df['close'] > df['sma_20']).astype(int)
        features['price_above_sma50'] = (df['close'] > df['sma_50']).astype(int)
        features['price_above_sma200'] = (df['close'] > df['sma_200']).astype(int)
        
        # Momentum features
        features['roc'] = df['roc']
        features['stoch_k'] = df['stoch_k']
        features['stoch_d'] = df['stoch_d']
        features['stoch_cross'] = (df['stoch_k'] > df['stoch_d']).astype(int)
        
        # Volume features (if available)
        if df['volume'].sum() > 0:
            features['volume_ratio'] = df['volume_ratio']
        else:
            features['volume_ratio'] = 1
        
        # Price position
        features['price_to_sma20'] = df['price_to_sma20']
        features['price_to_sma50'] = df['price_to_sma50']
        
        return features.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
    
    def train_ml_model(self, df: pd.DataFrame, lookforward: int = 5):
        """Train ML model to predict profitable trades"""
        df = self.calculate_technical_indicators(df)
        features_df = self.create_ml_features(df)
        
        # Create target: 1 if price increases by >2% in next 5 days
        df['future_return'] = df['close'].shift(-lookforward) / df['close'] - 1
        df['target'] = (df['future_return'] > 0.02).astype(int)
        
        # Align features with target
        X = features_df.iloc[:-lookforward]
        y = df['target'].iloc[:-lookforward]
        
        # Remove NaN
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 100:
            print("⚠️  Insufficient data for training")
            return None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train ensemble model
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                         min_samples_split=10, random_state=42)
        gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, 
                                             learning_rate=0.1, random_state=42)
        
        # Time series cross-validation
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
        
        # Use best model
        if np.mean(rf_scores) > np.mean(gb_scores):
            self.ml_model = rf_model
            print(f"✓ Random Forest selected (Accuracy: {np.mean(rf_scores):.3f})")
        else:
            self.ml_model = gb_model
            print(f"✓ Gradient Boosting selected (Accuracy: {np.mean(gb_scores):.3f})")
        
        # Final fit on all data
        self.ml_model.fit(X_scaled, y)
        
        # Feature importance
        self.feature_importance = dict(zip(X.columns, self.ml_model.feature_importances_))
        
        return self.ml_model
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk management"""
        risk_amount = self.capital * self.risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        position_size = risk_amount / risk_per_share
        max_position_value = self.capital * 0.20  # Max 20% per position
        
        position_size = min(position_size, max_position_value / entry_price)
        return int(position_size)
    
    def generate_live_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """Generate LIVE signal for the last candle"""
        if self.ml_model is None:
            print("⚠️  Model not trained. Training now...")
            self.train_ml_model(df)
        
        df = self.calculate_technical_indicators(df)
        features_df = self.create_ml_features(df)
        
        # Get latest features
        latest_features = features_df.iloc[-1:].values
        latest_scaled = self.scaler.transform(latest_features)
        
        # ML prediction
        ml_proba = self.ml_model.predict_proba(latest_scaled)[0]
        ml_confidence = ml_proba[1]  # Probability of bullish
        
        # Get latest data
        latest = df.iloc[-1]
        current_price = latest['close']
        
        # Signal conditions
        reasons = []
        bull_score = 0
        bear_score = 0
        
        # Bullish conditions
        if latest['rsi'] > 30 and latest['rsi'] < 70:
            bull_score += 1
            reasons.append(f"RSI neutral ({latest['rsi']:.1f})")
        
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
            bull_score += 2  # ML gets double weight
            reasons.append(f"ML bullish ({ml_confidence*100:.1f}% confidence)")
        
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
        
        if latest['stoch_k'] < latest['stoch_d']:
            bear_score += 1
            bear_reasons.append("Stochastic bearish")
        
        if ml_confidence < 0.4:
            bear_score += 2
            bear_reasons.append(f"ML bearish ({ml_confidence*100:.1f}% confidence)")
        
        # Determine signal
        total_score = bull_score + bear_score
        bull_percentage = bull_score / total_score if total_score > 0 else 0
        
        atr = latest['atr']
        
        if bull_score >= 5 and bull_percentage > 0.65:  # Strong bullish
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
                reasons=reasons
            )
        
        elif bear_score >= 4:  # Bearish signal
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
                reasons=bear_reasons
            )
        
        else:  # No clear signal
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
                reasons=["Mixed signals - waiting for clarity"]
            )
    
    def display_live_signal(self, signal: TradeSignal):
        """Display live signal in formatted output"""
        print(f"\n{'='*70}")
        print(f"🎯 LIVE TRADING SIGNAL - {signal.symbol}")
        print(f"{'='*70}")
        print(f"⏰ Time: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📍 Current Price: ${signal.entry_price:.2f}")
        print(f"")
        
        if signal.action == 'BUY':
            print(f"🟢 ACTION: {signal.action} ✓")
            print(f"💪 Signal Strength: {signal.signal_strength*100:.1f}%")
            print(f"🎓 ML Confidence: {signal.confidence*100:.1f}%")
            print(f"📊 Position Size: {signal.position_size} shares")
            print(f"🛑 Stop Loss: ${signal.stop_loss:.2f} ({((signal.stop_loss/signal.entry_price-1)*100):.2f}%)")
            print(f"🎯 Take Profit: ${signal.take_profit:.2f} ({((signal.take_profit/signal.entry_price-1)*100):.2f}%)")
            print(f"💰 Investment: ${signal.entry_price * signal.position_size:,.2f}")
            risk_reward = (signal.take_profit - signal.entry_price) / (signal.entry_price - signal.stop_loss)
            print(f"⚖️  Risk/Reward: 1:{risk_reward:.2f}")
        elif signal.action == 'SELL':
            print(f"🔴 ACTION: {signal.action}")
            print(f"💪 Signal Strength: {signal.signal_strength*100:.1f}%")
            print(f"🎓 ML Confidence: {signal.confidence*100:.1f}%")
        else:
            print(f"🟡 ACTION: {signal.action}")
            print(f"⏳ Waiting for better setup...")
        
        print(f"\n📋 Reasons:")
        for i, reason in enumerate(signal.reasons, 1):
            print(f"   {i}. {reason}")
        
        print(f"{'='*70}\n")
    
    def run_backtest(self, df: pd.DataFrame, symbol: str):
        """Run backtest on historical data"""
        print(f"\n{'='*70}")
        print(f"📈 BACKTESTING {symbol}")
        print(f"{'='*70}\n")
        
        # Train model on first 70% of data
        train_size = int(len(df) * 0.7)
        train_data = df.iloc[:train_size].copy()
        
        print(f"Training on {len(train_data)} candles...")
        self.train_ml_model(train_data)
        
        if self.ml_model is None:
            print("❌ Model training failed")
            return
        
        print(f"\n🔝 Top 5 Important Features:")
        sorted_features = sorted(self.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        for feat, importance in sorted_features:
            print(f"   {feat}: {importance:.3f}")
        
        # Backtest on remaining data
        test_data = df.iloc[train_size:].copy()
        print(f"\n🧪 Testing on {len(test_data)} candles...\n")
        
        # Simulate trading
        for i in range(50, len(test_data)):
            window_df = test_data.iloc[max(0, i-200):i+1].copy()
            signal = self.generate_live_signal(window_df, symbol)
            
            if signal.action in ['BUY', 'SELL']:
                self.execute_trade(signal)
        
        # Close remaining positions
        if self.positions:
            print("\n📤 Closing remaining positions...")
            for sym in list(self.positions.keys()):
                last_price = test_data.iloc[-1]['close']
                close_signal = TradeSignal(
                    symbol=sym, action='SELL', signal_strength=1.0,
                    confidence=1.0, entry_price=last_price,
                    stop_loss=0, take_profit=0, position_size=0,
                    timestamp=datetime.now(), strategy_type='SWING',
                    reasons=["End of backtest"]
                )
                self.execute_trade(close_signal)
        
        # Display results
        self.display_backtest_results()
    
    def execute_trade(self, signal: TradeSignal):
        """Execute trade based on signal"""
        if signal.action == 'BUY' and signal.position_size > 0:
            cost = signal.entry_price * signal.position_size
            
            if cost <= self.capital * 0.95:
                self.positions[signal.symbol] = {
                    'quantity': signal.position_size,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'entry_date': signal.timestamp
                }
                self.capital -= cost
                self.trade_history.append({
                    'action': 'BUY',
                    'symbol': signal.symbol,
                    'quantity': signal.position_size,
                    'price': signal.entry_price,
                    'timestamp': signal.timestamp
                })
        
        elif signal.action == 'SELL' and signal.symbol in self.positions:
            position = self.positions[signal.symbol]
            revenue = signal.entry_price * position['quantity']
            profit = revenue - (position['entry_price'] * position['quantity'])
            profit_pct = (profit / (position['entry_price'] * position['quantity'])) * 100
            
            self.capital += revenue
            del self.positions[signal.symbol]
            
            self.trade_history.append({
                'action': 'SELL',
                'symbol': signal.symbol,
                'quantity': position['quantity'],
                'price': signal.entry_price,
                'profit': profit,
                'profit_pct': profit_pct,
                'timestamp': signal.timestamp
            })
    
    def display_backtest_results(self):
        """Display comprehensive backtest results"""
        if not self.trade_history:
            print("❌ No trades executed")
            return
        
        df = pd.DataFrame(self.trade_history)
        closed_trades = df[df['action'] == 'SELL']
        
        if len(closed_trades) == 0:
            print("❌ No closed trades")
            return
        
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        winning_trades = closed_trades[closed_trades['profit'] > 0]
        losing_trades = closed_trades[closed_trades['profit'] < 0]
        
        win_rate = len(winning_trades) / len(closed_trades) * 100
        avg_win = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['profit'].mean()) if len(losing_trades) > 0 else 0
        
        profit_factor = abs(winning_trades['profit'].sum() / losing_trades['profit'].sum()) \
                       if len(losing_trades) > 0 and losing_trades['profit'].sum() != 0 else float('inf')
        
        avg_risk_reward = avg_win / avg_loss if avg_loss != 0 else float('inf')
        
        # Sharpe Ratio
        returns = closed_trades['profit_pct'].values
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = (1 + closed_trades['profit_pct'] / 100).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Display results
        print(f"\n{'='*70}")
        print(f"📊 BACKTEST RESULTS")
        print(f"{'='*70}")
        print(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        print(f"💵 Final Capital: ${self.capital:,.2f}")
        print(f"📈 Total Return: {total_return:.2f}%")
        print(f"")
        print(f"📋 Trading Statistics:")
        print(f"   Total Trades: {len(closed_trades)}")
        print(f"   🟢 Winning Trades: {len(winning_trades)}")
        print(f"   🔴 Losing Trades: {len(losing_trades)}")
        print(f"   🎯 Win Rate: {win_rate:.2f}%")
        print(f"")
        print(f"💵 Profit/Loss:")
        print(f"   Average Win: ${avg_win:,.2f}")
        print(f"   Average Loss: ${avg_loss:,.2f}")
        print(f"   Total Profit: ${winning_trades['profit'].sum():,.2f}")
        print(f"   Total Loss: ${losing_trades['profit'].sum():,.2f}")
        print(f"")
        print(f"📊 Performance Metrics:")
        print(f"   Profit Factor: {profit_factor:.2f}")
        print(f"   Risk/Reward Ratio: {avg_risk_reward:.2f}:1")
        print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {max_drawdown:.2f}%")
        print(f"{'='*70}\n")
        
        # Best and worst trades
        best_trade = closed_trades.loc[closed_trades['profit'].idxmax()]
        worst_trade = closed_trades.loc[closed_trades['profit'].idxmin()]
        
        print(f"🏆 Best Trade: ${best_trade['profit']:.2f} ({best_trade['profit_pct']:.2f}%)")
        print(f"💔 Worst Trade: ${worst_trade['profit']:.2f} ({worst_trade['profit_pct']:.2f}%)")
        print(f"{'='*70}\n")


def main():
    """Main function with examples"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          AI Trading Agent - Swing & Options Trading              ║
║              No TA-Lib Required | Live Signals                   ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check if yfinance is available
    if not YFINANCE_AVAILABLE:
        print("❌ Please install required package:")
        print("   pip install yfinance")
        return
    
    # Initialize agent
    agent = AITradingAgent(initial_capital=100000, risk_per_trade=0.02)
    
    # Example 1: Get live signal for a stock
    print("\n" + "="*70)
    print("EXAMPLE 1: LIVE SIGNAL FOR STOCK")
    print("="*70)
    
    symbol = "AAPL"  # Change to any symbol
    
    try:
        # Fetch recent data
        df = agent.fetch_data(symbol, period="1y", interval="1d")
        
        # Train model
        print(f"\n🎓 Training ML model on {symbol}...")
        agent.train_ml_model(df)
        
        # Get live signal
        print(f"\n🔍 Analyzing current market conditions...")
        live_signal = agent.generate_live_signal(df, symbol)
        agent.display_live_signal(live_signal)
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 2: Backtest
    print("\n" + "="*70)
    print("EXAMPLE 2: BACKTEST ON HISTORICAL DATA")
    print("="*70)
    
    try:
        # Reset agent for backtest
        agent = AITradingAgent(initial_capital=100000, risk_per_trade=0.02)
        
        # Fetch more data for backtest
        df = agent.fetch_data(symbol, period="2y", interval="1d")
        
        # Run backtest
        agent.run_backtest(df, symbol)
        
    except Exception as e:
        print(f"❌ Backtest Error: {e}")
    
    # Example 3: Index without volume (like ^GSPC, ^DJI)
    print("\n" + "="*70)
    print("EXAMPLE 3: LIVE SIGNAL FOR INDEX (NO VOLUME)")
    print("="*70)
    
    index_symbol = "^GSPC"  # S&P 500
    
    try:
        # Reset agent
        agent = AITradingAgent(initial_capital=100000, risk_per_trade=0.02)
        
        # Fetch index data
        df = agent.fetch_data(index_symbol, period="1y", interval="1d")
        
        # Train and get signal
        agent.train_ml_model(df)
        live_signal = agent.generate_live_signal(df, index_symbol)
        agent.display_live_signal(live_signal)
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 4: Multiple symbols
    print("\n" + "="*70)
    print("EXAMPLE 4: SCAN MULTIPLE SYMBOLS")
    print("="*70)
    
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    
    print(f"\n🔍 Scanning {len(symbols)} symbols for opportunities...\n")
    
    buy_signals = []
    
    for sym in symbols:
        try:
            agent_temp = AITradingAgent(initial_capital=100000, risk_per_trade=0.02)
            df = agent_temp.fetch_data(sym, period="6mo", interval="1d")
            agent_temp.train_ml_model(df)
            signal = agent_temp.generate_live_signal(df, sym)
            
            if signal.action == 'BUY':
                buy_signals.append(signal)
                print(f"✓ {sym}: BUY signal (Strength: {signal.signal_strength*100:.1f}%)")
            elif signal.action == 'SELL':
                print(f"✗ {sym}: SELL signal")
            else:
                print(f"○ {sym}: HOLD (waiting)")
                
        except Exception as e:
            print(f"⚠️  {sym}: Error - {e}")
    
    if buy_signals:
        print(f"\n{'='*70}")
        print(f"🎯 TOP BUY OPPORTUNITIES")
        print(f"{'='*70}")
        
        # Sort by signal strength
        buy_signals.sort(key=lambda x: x.signal_strength, reverse=True)
        
        for i, signal in enumerate(buy_signals[:3], 1):
            print(f"\n#{i} {signal.symbol}")
            print(f"   Signal Strength: {signal.signal_strength*100:.1f}%")
            print(f"   ML Confidence: {signal.confidence*100:.1f}%")
            print(f"   Entry: ${signal.entry_price:.2f}")
            print(f"   Stop Loss: ${signal.stop_loss:.2f}")
            print(f"   Take Profit: ${signal.take_profit:.2f}")
            rr = (signal.take_profit - signal.entry_price) / (signal.entry_price - signal.stop_loss)
            print(f"   R/R Ratio: 1:{rr:.2f}")
    
    print(f"\n{'='*70}")
    print("✓ Analysis Complete!")
    print(f"{'='*70}\n")
    
    # Usage instructions
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                     HOW TO USE THIS CODE                         ║
╚══════════════════════════════════════════════════════════════════╝

1. INSTALL DEPENDENCIES:
   pip install yfinance pandas numpy scikit-learn

2. GET LIVE SIGNAL FOR ANY SYMBOL:
   
   agent = AITradingAgent(initial_capital=100000, risk_per_trade=0.02)
   df = agent.fetch_data("AAPL", period="1y", interval="1d")
   agent.train_ml_model(df)
   signal = agent.generate_live_signal(df, "AAPL")
   agent.display_live_signal(signal)

3. RUN BACKTEST:
   
   agent = AITradingAgent(initial_capital=100000, risk_per_trade=0.02)
   df = agent.fetch_data("AAPL", period="2y", interval="1d")
   agent.run_backtest(df, "AAPL")

4. SCAN MULTIPLE STOCKS:
   
   symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
   for symbol in symbols:
       agent = AITradingAgent()
       df = agent.fetch_data(symbol, period="1y")
       agent.train_ml_model(df)
       signal = agent.generate_live_signal(df, symbol)
       agent.display_live_signal(signal)

5. CUSTOMIZATION:
   - Change risk_per_trade (default 0.02 = 2%)
   - Adjust initial_capital
   - Modify period/interval for different timeframes
   - Add your own technical indicators
   - Tune ML model parameters

📋 SUPPORTED SYMBOLS:
   - Stocks: AAPL, MSFT, GOOGL, TSLA, etc.
   - Indices: ^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (Nasdaq)
   - ETFs: SPY, QQQ, IWM, etc.
   - Forex: EURUSD=X, GBPUSD=X, etc.
   - Crypto: BTC-USD, ETH-USD, etc.

⚠️  NOTES:
   - Indices may not have volume data (handled automatically)
   - Requires internet connection to fetch data
   - Past performance doesn't guarantee future results
   - Always use proper risk management
   - This is for educational purposes only

╚══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
