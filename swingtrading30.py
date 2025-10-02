"""
Advanced AI Trading Agent for Swing Trading & Options
Features: ML-based signals, risk management, backtesting, options strategies
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML and Technical Analysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import talib

@dataclass
class TradeSignal:
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    signal_strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    timestamp: datetime
    strategy_type: str  # 'SWING' or 'OPTIONS'
    
@dataclass
class OptionsStrategy:
    strategy_name: str
    underlying_price: float
    strike_prices: List[float]
    option_types: List[str]  # 'CALL' or 'PUT'
    positions: List[int]  # +1 for long, -1 for short
    max_profit: float
    max_loss: float
    breakeven: List[float]

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
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Trend Indicators
        df['sma_20'] = talib.SMA(close, timeperiod=20)
        df['sma_50'] = talib.SMA(close, timeperiod=50)
        df['sma_200'] = talib.SMA(close, timeperiod=200)
        df['ema_12'] = talib.EMA(close, timeperiod=12)
        df['ema_26'] = talib.EMA(close, timeperiod=26)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)
        
        # RSI
        df['rsi'] = talib.RSI(close, timeperiod=14)
        df['rsi_sma'] = talib.SMA(df['rsi'].values, timeperiod=14)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR (Volatility)
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)
        df['atr_percent'] = df['atr'] / close * 100
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, close)
        
        # ADX (Trend Strength)
        df['adx'] = talib.ADX(high, low, close, timeperiod=14)
        
        # Volume indicators
        df['obv'] = talib.OBV(close, volume)
        df['ad'] = talib.AD(high, low, close, volume)
        
        # Price momentum
        df['roc'] = talib.ROC(close, timeperiod=10)
        df['mom'] = talib.MOM(close, timeperiod=10)
        
        # Support/Resistance levels
        df['pivot'] = (high + low + close) / 3
        df['r1'] = 2 * df['pivot'] - low
        df['s1'] = 2 * df['pivot'] - high
        
        # Custom features
        df['price_to_sma20'] = close / df['sma_20']
        df['price_to_sma50'] = close / df['sma_50']
        df['volume_sma'] = talib.SMA(volume, timeperiod=20)
        df['volume_ratio'] = volume / df['volume_sma']
        
        return df.dropna()
    
    def create_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for ML model"""
        features = pd.DataFrame()
        
        # Price-based features
        features['returns_1d'] = df['close'].pct_change(1)
        features['returns_5d'] = df['close'].pct_change(5)
        features['returns_10d'] = df['close'].pct_change(10)
        
        # Volatility features
        features['volatility_5d'] = df['returns_1d'].rolling(5).std()
        features['volatility_20d'] = df['returns_1d'].rolling(20).std()
        
        # Technical indicator features
        features['rsi'] = df['rsi']
        features['rsi_change'] = df['rsi'].diff()
        features['macd_hist'] = df['macd_hist']
        features['adx'] = df['adx']
        features['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        features['atr_percent'] = df['atr_percent']
        
        # Trend features
        features['sma_trend'] = (df['sma_20'] > df['sma_50']).astype(int)
        features['price_above_sma20'] = (df['close'] > df['sma_20']).astype(int)
        features['price_above_sma50'] = (df['close'] > df['sma_50']).astype(int)
        
        # Momentum features
        features['roc'] = df['roc']
        features['stoch_k'] = df['stoch_k']
        features['volume_ratio'] = df['volume_ratio']
        
        return features.dropna()
    
    def train_ml_model(self, df: pd.DataFrame, lookforward: int = 5):
        """Train ML model to predict profitable trades"""
        df = self.calculate_technical_indicators(df)
        features_df = self.create_ml_features(df)
        
        # Create target: 1 if price increases by >2% in next 5 days, 0 otherwise
        df['future_return'] = df['close'].shift(-lookforward) / df['close'] - 1
        df['target'] = (df['future_return'] > 0.02).astype(int)
        
        # Align features with target
        X = features_df.iloc[:-lookforward]
        y = df['target'].iloc[len(df) - len(features_df):-lookforward]
        
        # Remove NaN
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train ensemble model
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        
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
            print(f"Random Forest selected (Accuracy: {np.mean(rf_scores):.3f})")
        else:
            self.ml_model = gb_model
            print(f"Gradient Boosting selected (Accuracy: {np.mean(gb_scores):.3f})")
        
        # Final fit on all data
        self.ml_model.fit(X_scaled, y)
        
        # Feature importance
        self.feature_importance = dict(zip(X.columns, self.ml_model.feature_importances_))
        
        return self.ml_model
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        risk_amount = self.capital * self.risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        position_size = risk_amount / risk_per_share
        max_position_value = self.capital * 0.20  # Max 20% per position
        
        position_size = min(position_size, max_position_value / entry_price)
        return int(position_size)
    
    def generate_swing_signal(self, df: pd.DataFrame, symbol: str) -> Optional[TradeSignal]:
        """Generate swing trading signal using ML and technical analysis"""
        if self.ml_model is None:
            return None
        
        df = self.calculate_technical_indicators(df)
        features_df = self.create_ml_features(df)
        
        if len(features_df) == 0:
            return None
        
        # Get latest features
        latest_features = features_df.iloc[-1:].values
        latest_scaled = self.scaler.transform(latest_features)
        
        # ML prediction
        ml_prediction = self.ml_model.predict_proba(latest_scaled)[0][1]
        
        # Technical confirmation
        latest = df.iloc[-1]
        current_price = latest['close']
        
        # Bull signal conditions
        bull_conditions = [
            latest['rsi'] < 70 and latest['rsi'] > 30,  # Not overbought/oversold
            latest['macd'] > latest['macd_signal'],      # MACD bullish
            latest['close'] > latest['sma_20'],          # Above SMA20
            latest['adx'] > 25,                          # Strong trend
            latest['stoch_k'] > latest['stoch_d'],       # Stochastic bullish
            ml_prediction > 0.6                           # ML confidence
        ]
        
        # Bear signal conditions
        bear_conditions = [
            latest['rsi'] < 70 and latest['rsi'] > 30,
            latest['macd'] < latest['macd_signal'],
            latest['close'] < latest['sma_20'],
            latest['adx'] > 25,
            latest['stoch_k'] < latest['stoch_d'],
            ml_prediction < 0.4
        ]
        
        bull_score = sum(bull_conditions) / len(bull_conditions)
        bear_score = sum(bear_conditions) / len(bear_conditions)
        
        # Generate signal
        if bull_score > 0.70:
            atr = latest['atr']
            stop_loss = current_price - (2 * atr)
            take_profit = current_price + (3 * atr)  # 3:1 reward-risk
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            return TradeSignal(
                symbol=symbol,
                action='BUY',
                signal_strength=bull_score,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                timestamp=datetime.now(),
                strategy_type='SWING'
            )
        
        elif bear_score > 0.70 and symbol in self.positions:
            return TradeSignal(
                symbol=symbol,
                action='SELL',
                signal_strength=bear_score,
                entry_price=current_price,
                stop_loss=0,
                take_profit=0,
                position_size=self.positions[symbol]['quantity'],
                timestamp=datetime.now(),
                strategy_type='SWING'
            )
        
        return None
    
    def analyze_options_strategy(self, symbol: str, current_price: float, 
                                 volatility: float, days_to_expiry: int = 30) -> OptionsStrategy:
        """Analyze and recommend options strategy"""
        
        # Simplified Black-Scholes for demonstration
        # In production, use actual options pricing library
        
        if volatility > 0.40:  # High volatility - sell premium
            # Iron Condor strategy
            strike_distance = current_price * 0.05
            
            return OptionsStrategy(
                strategy_name="Iron Condor",
                underlying_price=current_price,
                strike_prices=[
                    current_price - 2*strike_distance,  # Put buy
                    current_price - strike_distance,     # Put sell
                    current_price + strike_distance,     # Call sell
                    current_price + 2*strike_distance    # Call buy
                ],
                option_types=['PUT', 'PUT', 'CALL', 'CALL'],
                positions=[1, -1, -1, 1],
                max_profit=strike_distance * 0.3,
                max_loss=strike_distance * 0.7,
                breakeven=[current_price - strike_distance * 0.7, 
                          current_price + strike_distance * 0.7]
            )
        
        elif volatility < 0.25:  # Low volatility - buy options
            # Bull Call Spread
            strike_distance = current_price * 0.03
            
            return OptionsStrategy(
                strategy_name="Bull Call Spread",
                underlying_price=current_price,
                strike_prices=[
                    current_price,
                    current_price + strike_distance
                ],
                option_types=['CALL', 'CALL'],
                positions=[1, -1],
                max_profit=strike_distance * 0.5,
                max_loss=strike_distance * 0.5,
                breakeven=[current_price + strike_distance * 0.5]
            )
        
        else:  # Medium volatility - directional play
            # Protective Put
            strike_distance = current_price * 0.05
            
            return OptionsStrategy(
                strategy_name="Protective Put",
                underlying_price=current_price,
                strike_prices=[current_price - strike_distance],
                option_types=['PUT'],
                positions=[1],
                max_profit=float('inf'),
                max_loss=strike_distance,
                breakeven=[current_price]
            )
    
    def execute_trade(self, signal: TradeSignal):
        """Execute trade based on signal"""
        if signal.action == 'BUY' and signal.position_size > 0:
            cost = signal.entry_price * signal.position_size
            
            if cost <= self.capital * 0.95:  # Keep 5% cash buffer
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
                    'timestamp': signal.timestamp,
                    'strategy': signal.strategy_type
                })
                
                print(f"✓ BUY {signal.position_size} {signal.symbol} @ ${signal.entry_price:.2f}")
                print(f"  Stop Loss: ${signal.stop_loss:.2f} | Take Profit: ${signal.take_profit:.2f}")
        
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
                'timestamp': signal.timestamp,
                'strategy': signal.strategy_type
            })
            
            print(f"✓ SELL {position['quantity']} {signal.symbol} @ ${signal.entry_price:.2f}")
            print(f"  Profit: ${profit:.2f} ({profit_pct:.2f}%)")
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trade_history:
            return {}
        
        df = pd.DataFrame(self.trade_history)
        closed_trades = df[df['action'] == 'SELL']
        
        if len(closed_trades) == 0:
            return {}
        
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        winning_trades = closed_trades[closed_trades['profit'] > 0]
        losing_trades = closed_trades[closed_trades['profit'] < 0]
        
        win_rate = len(winning_trades) / len(closed_trades) * 100
        avg_win = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['profit'].mean()) if len(losing_trades) > 0 else 0
        
        profit_factor = abs(winning_trades['profit'].sum() / losing_trades['profit'].sum()) \
                       if len(losing_trades) > 0 and losing_trades['profit'].sum() != 0 else float('inf')
        
        avg_risk_reward = avg_win / avg_loss if avg_loss != 0 else float('inf')
        
        # Sharpe Ratio (simplified)
        returns = closed_trades['profit_pct'].values
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = (1 + closed_trades['profit_pct'] / 100).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        return {
            'Total Return (%)': round(total_return, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Total Trades': len(closed_trades),
            'Winning Trades': len(winning_trades),
            'Losing Trades': len(losing_trades),
            'Avg Win ($)': round(avg_win, 2),
            'Avg Loss ($)': round(avg_loss, 2),
            'Profit Factor': round(profit_factor, 2),
            'Risk/Reward Ratio': round(avg_risk_reward, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Current Capital ($)': round(self.capital, 2)
        }
    
    def run_backtest(self, df: pd.DataFrame, symbol: str):
        """Run backtest on historical data"""
        print(f"\n{'='*60}")
        print(f"Starting Backtest for {symbol}")
        print(f"{'='*60}\n")
        
        # Train model on first 70% of data
        train_size = int(len(df) * 0.7)
        train_data = df.iloc[:train_size].copy()
        test_data = df.iloc[train_size:].copy()
        
        print("Training ML model...")
        self.train_ml_model(train_data)
        
        print(f"\nTop 5 Important Features:")
        sorted_features = sorted(self.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        for feat, importance in sorted_features:
            print(f"  {feat}: {importance:.3f}")
        
        print(f"\nRunning backtest on {len(test_data)} days...")
        
        # Simulate trading
        for i in range(50, len(test_data)):
            window_df = test_data.iloc[max(0, i-200):i+1].copy()
            signal = self.generate_swing_signal(window_df, symbol)
            
            if signal:
                self.execute_trade(signal)
        
        # Close remaining positions
        if self.positions:
            print("\nClosing remaining positions...")
            for symbol in list(self.positions.keys()):
                last_price = test_data.iloc[-1]['close']
                signal = TradeSignal(
                    symbol=symbol,
                    action='SELL',
                    signal_strength=1.0,
                    entry_price=last_price,
                    stop_loss=0,
                    take_profit=0,
                    position_size=0,
                    timestamp=datetime.now(),
                    strategy_type='SWING'
                )
                self.execute_trade(signal)
        
        # Display results
        metrics = self.calculate_performance_metrics()
        
        print(f"\n{'='*60}")
        print("BACKTEST RESULTS")
        print(f"{'='*60}")
        for key, value in metrics.items():
            print(f"{key:.<40} {value}")
        print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    # Generate sample data (replace with real data from yfinance, alpaca, etc.)
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
    
    # Simulate realistic price data
    returns = np.random.randn(len(dates)) * 0.02
    price = 100 * (1 + returns).cumprod()
    
    df = pd.DataFrame({
        'date': dates,
        'open': price * (1 + np.random.randn(len(dates)) * 0.005),
        'high': price * (1 + abs(np.random.randn(len(dates)) * 0.01)),
        'low': price * (1 - abs(np.random.randn(len(dates)) * 0.01)),
        'close': price,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # Initialize and run agent
    agent = AITradingAgent(initial_capital=100000, risk_per_trade=0.02)
    agent.run_backtest(df, 'AAPL')
    
    # Example options analysis
    print("\n" + "="*60)
    print("OPTIONS STRATEGY RECOMMENDATION")
    print("="*60)
    options_strategy = agent.analyze_options_strategy(
        symbol='AAPL',
        current_price=df.iloc[-1]['close'],
        volatility=0.35,
        days_to_expiry=30
    )
    
    print(f"Strategy: {options_strategy.strategy_name}")
    print(f"Underlying Price: ${options_strategy.underlying_price:.2f}")
    print(f"Max Profit: ${options_strategy.max_profit:.2f}")
    print(f"Max Loss: ${options_strategy.max_loss:.2f}")
    print(f"Breakeven: {[f'${x:.2f}' for x in options_strategy.breakeven]}")
    print("="*60)
