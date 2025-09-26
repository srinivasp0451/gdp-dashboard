import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="VMA-Elite Trading System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class VMAEliteStrategy:
    def __init__(self, momentum_length=21, volume_ma_length=20, atr_length=14, 
                 sensitivity=1.5, risk_reward_ratio=3.0, stop_loss_pct=1.5, 
                 target_pct=4.5, use_volume=True, min_elite_score=0.75):
        self.momentum_length = momentum_length
        self.volume_ma_length = volume_ma_length
        self.atr_length = atr_length
        self.sensitivity = sensitivity
        self.risk_reward_ratio = risk_reward_ratio
        self.stop_loss_pct = stop_loss_pct
        self.target_pct = target_pct
        self.use_volume = use_volume
        self.min_elite_score = min_elite_score
    
    def sort_and_clean_data(self, df):
        """Sort data in ascending order and clean"""
        df = df.copy()
        
        # Sort by date (ascending)
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Forward fill missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Check if volume column exists but has all zeros
        if 'Volume' in df.columns:
            volume_sum = df['Volume'].sum()
            if volume_sum == 0:
                self.use_volume = False
                st.warning("⚠️ Volume column detected but all values are zero. Switching to price-only mode.")
            else:
                # Check if volume has meaningful variation
                volume_std = df['Volume'].std()
                volume_mean = df['Volume'].mean()
                if volume_mean > 0 and volume_std / volume_mean < 0.1:  # Low variation
                    self.use_volume = False
                    st.warning("⚠️ Volume data has low variation. Switching to price-only mode.")
        
        return df
    
    def calculate_advanced_momentum(self, df):
        """Enhanced momentum calculation with multiple timeframes"""
        # Multi-period ROC for better momentum detection
        df['ROC_Short'] = df['Close'].pct_change(periods=5) * 100
        df['ROC_Medium'] = df['Close'].pct_change(periods=self.momentum_length) * 100
        df['ROC_Long'] = df['Close'].pct_change(periods=self.momentum_length * 2) * 100
        
        # Momentum acceleration with better smoothing
        df['Momentum_Raw'] = (df['ROC_Short'] * 0.5 + df['ROC_Medium'] * 0.3 + df['ROC_Long'] * 0.2)
        df['Momentum_Acceleration'] = df['Momentum_Raw'].rolling(window=3).mean().fillna(0)
        
        # Momentum strength and direction
        df['Momentum_Strength'] = np.abs(df['Momentum_Acceleration'])
        df['Momentum_Direction'] = np.sign(df['Momentum_Acceleration'])
        
        # Momentum consistency (how consistent the direction is)
        df['Momentum_Consistency'] = df['Momentum_Direction'].rolling(window=5).apply(
            lambda x: np.abs(x.sum()) / len(x), raw=True
        ).fillna(0)
        
        return df
    
    def calculate_volatility_breakout(self, df):
        """Enhanced volatility and breakout detection"""
        # True Range and ATR
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = np.abs(df['High'] - df['Close'].shift(1))
        df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=self.atr_length).mean()
        
        # Volatility expansion detection
        df['ATR_MA'] = df['ATR'].rolling(window=self.atr_length).mean()
        df['Volatility_Expansion'] = df['ATR'] / df['ATR_MA']
        
        # Price breakout detection
        df['High_20'] = df['High'].rolling(window=20).max()
        df['Low_20'] = df['Low'].rolling(window=20).min()
        df['Price_Position'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'])
        
        # Bollinger Band squeeze detection
        df['BB_Upper'] = df['Close'].rolling(window=20).mean() + (df['Close'].rolling(window=20).std() * 2)
        df['BB_Lower'] = df['Close'].rolling(window=20).mean() - (df['Close'].rolling(window=20).std() * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close']
        df['BB_Squeeze'] = df['BB_Width'] < df['BB_Width'].rolling(window=20).quantile(0.2)
        
        # Clean up temporary columns
        df.drop(['High_Low', 'High_Close', 'Low_Close', 'TR'], axis=1, inplace=True)
        return df
    
    def calculate_trend_structure(self, df):
        """Enhanced trend and market structure analysis"""
        # Multiple EMA for trend detection
        df['EMA_9'] = df['Close'].ewm(span=9).mean()
        df['EMA_21'] = df['Close'].ewm(span=21).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        
        # Trend alignment score
        conditions = [
            df['Close'] > df['EMA_9'],
            df['EMA_9'] > df['EMA_21'],
            df['EMA_21'] > df['EMA_50'],
            df['Close'] > df['Close'].shift(1)
        ]
        
        df['Trend_Bullish_Score'] = sum(conditions)
        
        conditions = [
            df['Close'] < df['EMA_9'],
            df['EMA_9'] < df['EMA_21'],
            df['EMA_21'] < df['EMA_50'],
            df['Close'] < df['Close'].shift(1)
        ]
        
        df['Trend_Bearish_Score'] = sum(conditions)
        
        # Support and resistance levels
        window = 20
        df['Resistance'] = df['High'].rolling(window=window).max()
        df['Support'] = df['Low'].rolling(window=window).min()
        df['Key_Level'] = (df['Resistance'] + df['Support']) / 2
        
        # Breakout confirmation
        df['Bullish_Breakout'] = (df['Close'] > df['Resistance'].shift(1)) & (df['Volume'] > df['Volume'].rolling(20).mean() * 1.2 if self.use_volume else True)
        df['Bearish_Breakdown'] = (df['Close'] < df['Support'].shift(1)) & (df['Volume'] > df['Volume'].rolling(20).mean() * 1.2 if self.use_volume else True)
        
        return df
    
    def calculate_volume_analysis(self, df):
        """Enhanced volume analysis or price-based alternative"""
        if self.use_volume and 'Volume' in df.columns and df['Volume'].sum() > 0:
            # Volume-based analysis
            df['Volume_MA'] = df['Volume'].rolling(window=self.volume_ma_length).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            df['Volume_Surge'] = df['Volume_Ratio'] > 1.5
            
            # On Balance Volume
            df['OBV'] = (df['Volume'] * np.sign(df['Close'].diff())).cumsum()
            df['OBV_MA'] = df['OBV'].rolling(window=20).mean()
            df['OBV_Bullish'] = df['OBV'] > df['OBV_MA']
            
            # Volume Price Trend
            df['VPT'] = ((df['Close'].diff() / df['Close'].shift(1)) * df['Volume']).cumsum()
            df['VPT_MA'] = df['VPT'].rolling(window=20).mean()
            df['VPT_Signal'] = df['VPT'] > df['VPT_MA']
            
            df['Volume_Confirmation'] = df['Volume_Surge'] & df['OBV_Bullish'] & df['VPT_Signal']
            
        else:
            # Price-based analysis for indices without volume
            df['Price_Change'] = df['Close'].pct_change() * 100
            df['Price_Volatility'] = df['Price_Change'].rolling(window=self.volume_ma_length).std()
            df['Price_Surge'] = np.abs(df['Price_Change']) > df['Price_Volatility'] * 1.5
            
            # Price momentum as volume substitute
            df['Price_Momentum'] = df['Close'].diff().rolling(window=5).sum()
            df['Price_Momentum_MA'] = df['Price_Momentum'].rolling(window=20).mean()
            df['Price_Momentum_Strong'] = np.abs(df['Price_Momentum']) > np.abs(df['Price_Momentum_MA']) * 1.2
            
            df['Volume_Confirmation'] = df['Price_Surge'] & df['Price_Momentum_Strong']
            df['Volume_Ratio'] = 1.0  # Default for non-volume analysis
        
        return df
    
    def calculate_elite_score(self, df):
        """Enhanced Elite Score calculation"""
        # Normalize components to 0-1 range
        df['Momentum_Score'] = self.normalize_score(df['Momentum_Strength'], 50)
        df['Trend_Score'] = np.maximum(df['Trend_Bullish_Score'], df['Trend_Bearish_Score']) / 4.0
        df['Volatility_Score'] = self.normalize_score(df['Volatility_Expansion'], 50)
        df['Consistency_Score'] = df['Momentum_Consistency']
        df['Position_Score'] = np.where(df['Price_Position'] > 0.8, 1.0, 
                                      np.where(df['Price_Position'] < 0.2, 1.0, 0.5))
        
        # Breakout bonus
        df['Breakout_Score'] = np.where(df['Bullish_Breakout'] | df['Bearish_Breakdown'], 1.0, 0.5)
        
        # Weighted Elite Score
        weights = {
            'momentum': 0.25,
            'trend': 0.20,
            'volatility': 0.15,
            'consistency': 0.15,
            'position': 0.15,
            'breakout': 0.10
        }
        
        df['Elite_Score'] = (
            df['Momentum_Score'] * weights['momentum'] +
            df['Trend_Score'] * weights['trend'] +
            df['Volatility_Score'] * weights['volatility'] +
            df['Consistency_Score'] * weights['consistency'] +
            df['Position_Score'] * weights['position'] +
            df['Breakout_Score'] * weights['breakout']
        )
        
        return df
    
    def normalize_score(self, series, window):
        """Normalize series to 0-1 range using rolling percentile"""
        return series.rolling(window=window).rank(pct=True).fillna(0.5)
    
    def generate_enhanced_signals(self, df):
        """Generate high-quality trading signals"""
        
        # Enhanced signal conditions
        # Long conditions (ALL must be true)
        long_conditions = [
            df['Elite_Score'] > self.min_elite_score,  # High quality setup
            df['Trend_Bullish_Score'] >= 3,  # Strong bullish trend
            df['Momentum_Acceleration'] > 0,  # Positive momentum
            df['Momentum_Consistency'] > 0.6,  # Consistent direction
            df['Volume_Confirmation'],  # Volume/price confirmation
            df['Volatility_Expansion'] > 1.1,  # Some volatility expansion
            df['Close'] > df['EMA_21'],  # Above key moving average
            ~df['BB_Squeeze']  # Not in squeeze (avoid choppy markets)
        ]
        
        # Short conditions (ALL must be true)
        short_conditions = [
            df['Elite_Score'] > self.min_elite_score,  # High quality setup
            df['Trend_Bearish_Score'] >= 3,  # Strong bearish trend
            df['Momentum_Acceleration'] < 0,  # Negative momentum
            df['Momentum_Consistency'] > 0.6,  # Consistent direction
            df['Volume_Confirmation'],  # Volume/price confirmation
            df['Volatility_Expansion'] > 1.1,  # Some volatility expansion
            df['Close'] < df['EMA_21'],  # Below key moving average
            ~df['BB_Squeeze']  # Not in squeeze
        ]
        
        # Combine all conditions
        df['Long_Setup'] = pd.Series(True, index=df.index)
        for condition in long_conditions:
            df['Long_Setup'] = df['Long_Setup'] & condition.fillna(False)
        
        df['Short_Setup'] = pd.Series(True, index=df.index)
        for condition in short_conditions:
            df['Short_Setup'] = df['Short_Setup'] & condition.fillna(False)
        
        # Signal generation with proper timing (use CURRENT bar, not future)
        # Signal is generated at the END of the current bar
        df['Long_Signal'] = df['Long_Setup'] & ~df['Long_Setup'].shift(1).fillna(False)
        df['Short_Signal'] = df['Short_Setup'] & ~df['Short_Setup'].shift(1).fillna(False)
        
        # Add cooling period to avoid overtrading
        cooling_period = 10
        for i in range(1, cooling_period + 1):
            df['Long_Signal'] = df['Long_Signal'] & ~df['Long_Signal'].shift(i).fillna(False)
            df['Short_Signal'] = df['Short_Signal'] & ~df['Short_Signal'].shift(i).fillna(False)
        
        # Calculate stop loss and targets based on CURRENT price (not future)
        # Use more conservative stops based on volatility
        df['Dynamic_Stop_Pct'] = np.maximum(self.stop_loss_pct, (df['ATR'] / df['Close']) * 100 * 1.5)
        df['Dynamic_Target_Pct'] = df['Dynamic_Stop_Pct'] * self.risk_reward_ratio
        
        df['Long_SL'] = df['Close'] * (1 - df['Dynamic_Stop_Pct'] / 100)
        df['Short_SL'] = df['Close'] * (1 + df['Dynamic_Stop_Pct'] / 100)
        
        df['Long_Target'] = df['Close'] * (1 + df['Dynamic_Target_Pct'] / 100)
        df['Short_Target'] = df['Close'] * (1 - df['Dynamic_Target_Pct'] / 100)
        
        return df
    
    def process_data(self, df):
        """Process all indicators and signals"""
        df = self.sort_and_clean_data(df)
        df = self.calculate_advanced_momentum(df)
        df = self.calculate_volatility_breakout(df)
        df = self.calculate_trend_structure(df)
        df = self.calculate_volume_analysis(df)
        df = self.calculate_elite_score(df)
        df = self.generate_enhanced_signals(df)
        return df
    
    def backtest(self, df, initial_balance=100000):
        """Enhanced backtesting with proper entry timing"""
        df = self.process_data(df)
        
        trades = []
        balance = initial_balance
        max_balance = initial_balance
        current_trade = None
        
        # Buy and hold comparison
        buy_hold_entry = df['Close'].iloc[0]
        buy_hold_exit = df['Close'].iloc[-1]
        buy_hold_return = (buy_hold_exit - buy_hold_entry) / buy_hold_entry * 100
        buy_hold_points = buy_hold_exit - buy_hold_entry
        
        # Iterate through data with proper indexing
        for i in range(1, len(df)):  # Start from 1 to have previous data
            current_row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Entry signals - use CURRENT bar close price
            if current_row['Long_Signal'] or current_row['Short_Signal']:
                # Close existing opposite trade
                if current_trade:
                    if ((current_row['Long_Signal'] and current_trade['type'] == 'SHORT') or 
                        (current_row['Short_Signal'] and current_trade['type'] == 'LONG')):
                        
                        # Exit at current close price
                        exit_price = current_row['Close']
                        pnl_points = self.calculate_pnl_points(current_trade, exit_price)
                        pnl_pct = self.calculate_pnl_percentage(current_trade, exit_price)
                        pnl_amount = (pnl_pct / 100) * balance
                        balance += pnl_amount
                        max_balance = max(max_balance, balance)
                        
                        current_trade.update({
                            'exit_price': exit_price,
                            'exit_date': current_row.name,
                            'exit_reason': 'Opposite Signal',
                            'pnl_points': pnl_points,
                            'pnl_percentage': pnl_pct,
                            'pnl_amount': pnl_amount,
                            'balance': balance
                        })
                        trades.append(current_trade)
                        current_trade = None
                
                # Open new trade at CURRENT close price
                if not current_trade:
                    trade_type = 'LONG' if current_row['Long_Signal'] else 'SHORT'
                    entry_price = current_row['Close']  # Use current close, not future
                    
                    # Calculate stops and targets based on entry price
                    stop_pct = current_row['Dynamic_Stop_Pct']
                    target_pct = current_row['Dynamic_Target_Pct']
                    
                    stop_loss = entry_price * (1 - stop_pct/100) if trade_type == 'LONG' else entry_price * (1 + stop_pct/100)
                    target = entry_price * (1 + target_pct/100) if trade_type == 'LONG' else entry_price * (1 - target_pct/100)
                    
                    prob_profit = min(0.95, current_row['Elite_Score'] * 1.1)
                    
                    current_trade = {
                        'entry_date': current_row.name,
                        'entry_price': entry_price,
                        'type': trade_type,
                        'stop_loss': stop_loss,
                        'target': target,
                        'stop_pct': stop_pct,
                        'target_pct': target_pct,
                        'elite_score': current_row['Elite_Score'],
                        'momentum': current_row['Momentum_Acceleration'],
                        'trend_score': current_row['Trend_Bullish_Score'] if trade_type == 'LONG' else current_row['Trend_Bearish_Score'],
                        'probability_profit': prob_profit,
                        'entry_reason': self.get_entry_reason(current_row, trade_type)
                    }
            
            # Check for stop loss or target hit on NEXT candles
            if current_trade:
                hit_stop = False
                hit_target = False
                exit_reason = ''
                exit_price = 0
                
                if current_trade['type'] == 'LONG':
                    if current_row['Low'] <= current_trade['stop_loss']:
                        hit_stop = True
                        exit_price = current_trade['stop_loss']
                        exit_reason = 'Stop Loss Hit'
                    elif current_row['High'] >= current_trade['target']:
                        hit_target = True
                        exit_price = current_trade['target']
                        exit_reason = 'Target Hit'
                else:  # SHORT
                    if current_row['High'] >= current_trade['stop_loss']:
                        hit_stop = True
                        exit_price = current_trade['stop_loss']
                        exit_reason = 'Stop Loss Hit'
                    elif current_row['Low'] <= current_trade['target']:
                        hit_target = True
                        exit_price = current_trade['target']
                        exit_reason = 'Target Hit'
                
                if hit_stop or hit_target:
                    pnl_points = self.calculate_pnl_points(current_trade, exit_price)
                    pnl_pct = self.calculate_pnl_percentage(current_trade, exit_price)
                    pnl_amount = (pnl_pct / 100) * balance
                    balance += pnl_amount
                    max_balance = max(max_balance, balance)
                    
                    current_trade.update({
                        'exit_price': exit_price,
                        'exit_date': current_row.name,
                        'exit_reason': exit_reason,
                        'pnl_points': pnl_points,
                        'pnl_percentage': pnl_pct,
                        'pnl_amount': pnl_amount,
                        'balance': balance
                    })
                    trades.append(current_trade)
                    current_trade = None
        
        # Close any remaining trade at the end
        if current_trade:
            exit_price = df['Close'].iloc[-1]
            pnl_points = self.calculate_pnl_points(current_trade, exit_price)
            pnl_pct = self.calculate_pnl_percentage(current_trade, exit_price)
            pnl_amount = (pnl_pct / 100) * balance
            balance += pnl_amount
            
            current_trade.update({
                'exit_price': exit_price,
                'exit_date': df.index[-1],
                'exit_reason': 'End of Data',
                'pnl_points': pnl_points,
                'pnl_percentage': pnl_pct,
                'pnl_amount': pnl_amount,
                'balance': balance
            })
            trades.append(current_trade)
        
        # Calculate performance metrics
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        performance = self.calculate_performance_metrics(trades_df, initial_balance, balance, buy_hold_return, buy_hold_points)
        
        return df, trades_df, performance
    
    def calculate_pnl_points(self, trade, exit_price):
        """Calculate P&L in points"""
        if trade['type'] == 'LONG':
            return exit_price - trade['entry_price']
        else:
            return trade['entry_price'] - exit_price
    
    def calculate_pnl_percentage(self, trade, exit_price):
        """Calculate P&L as percentage"""
        if trade['type'] == 'LONG':
            return ((exit_price - trade['entry_price']) / trade['entry_price']) * 100
        else:
            return ((trade['entry_price'] - exit_price) / trade['entry_price']) * 100
    
    def get_entry_reason(self, row, trade_type):
        """Generate detailed entry reason"""
        reasons = []
        reasons.append(f"Elite Score: {row['Elite_Score']:.3f}")
        
        if trade_type == 'LONG':
            reasons.append(f"Bullish Trend Score: {row['Trend_Bullish_Score']}/4")
        else:
            reasons.append(f"Bearish Trend Score: {row['Trend_Bearish_Score']}/4")
        
        reasons.append(f"Momentum: {row['Momentum_Acceleration']:.2f}")
        reasons.append(f"Consistency: {row['Momentum_Consistency']:.2f}")
        
        if row['Volume_Confirmation']:
            reasons.append("Volume Confirmed")
        
        if row['Volatility_Expansion'] > 1.1:
            reasons.append("Volatility Expansion")
        
        return "; ".join(reasons)
    
    def calculate_performance_metrics(self, trades_df, initial_balance, final_balance, buy_hold_return, buy_hold_points):
        """Calculate comprehensive performance metrics"""
        if trades_df.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'total_points': 0,
                'avg_win_points': 0,
                'avg_loss_points': 0,
                'avg_win_pct': 0,
                'avg_loss_pct': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'final_balance': initial_balance,
                'best_trade_points': 0,
                'worst_trade_points': 0,
                'best_trade_pct': 0,
                'worst_trade_pct': 0,
                'buy_hold_return': buy_hold_return,
                'buy_hold_points': buy_hold_points,
                'strategy_vs_buyhold_points': -buy_hold_points,
                'strategy_vs_buyhold_pct': -buy_hold_return,
                'sharpe_ratio': 0,
                'max_consecutive_losses': 0
            }
        
        winning_trades = trades_df[trades_df['pnl_points'] > 0]
        losing_trades = trades_df[trades_df['pnl_points'] <= 0]
        
        total_return = ((final_balance - initial_balance) / initial_balance) * 100
        win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        
        # Points and percentage metrics
        total_points = trades_df['pnl_points'].sum()
        avg_win_points = winning_trades['pnl_points'].mean() if len(winning_trades) > 0 else 0
        avg_loss_points = losing_trades['pnl_points'].mean() if len(losing_trades) > 0 else 0
        avg_win_pct = winning_trades['pnl_percentage'].mean() if len(winning_trades) > 0 else 0
        avg_loss_pct = losing_trades['pnl_percentage'].mean() if len(losing_trades) > 0 else 0
        
        # Profit factor and other metrics
        gross_profit = winning_trades['pnl_points'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl_points'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Maximum drawdown
        if 'balance' in trades_df.columns and len(trades_df) > 0:
            equity_curve = trades_df['balance'].values
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak * 100
            max_drawdown = np.max(drawdown)
        else:
            max_drawdown = 0
        
        # Sharpe ratio (simplified)
        if len(trades_df) > 1:
            returns = trades_df['pnl_percentage'].values
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Maximum consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for _, trade in trades_df.iterrows():
            if trade['pnl_points'] <= 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        # Strategy vs Buy & Hold
        strategy_vs_buyhold_points = total_points - buy_hold_points
        strategy_vs_buyhold_pct = total_return - buy_hold_return
        
        return {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'total_points': total_points,
            'avg_win_points': avg_win_points,
            'avg_loss_points': avg_loss_points,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'final_balance': final_balance,
            'best_trade_points': trades_df['pnl_points'].max() if len(trades_df) > 0 else 0,
            'worst_trade_points': trades_df['pnl_points'].min() if len(trades_df) > 0 else 0,
            'best_trade_pct': trades_df['pnl_percentage'].max() if len(trades_df) > 0 else 0,
            'worst_trade_pct': trades_df['pnl_percentage'].min() if len(trades_df) > 0 else 0,
            'buy_hold_return': buy_hold_return,
            'buy_hold_points': buy_hold_points,
            'strategy_vs_buyhold_points': strategy_vs_buyhold_points,
            'strategy_vs_buyhold_pct': strategy_vs_buyhold_pct,
            'sharpe_ratio': sharpe_ratio,
            'max_consecutive_losses': max_consecutive_losses,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }

def load_data_yfinance(symbol, period="2y", interval="1d"):
    """Load data from Yahoo Finance with proper sorting"""
    try:
        # Indian market symbols
        indian_symbols = {
            'NIFTY': '^NSEI',
            'BANKNIFTY': '^NSEBANK', 
            'SENSEX': '^BSESN',
            'NIFTY50': '^NSEI',
            'NIFTYBANK': '^NSEBANK'
        }
        
        yf_symbol = indian_symbols.get(symbol.upper(), symbol)
        
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            st.error(f"No data found for {symbol} ({yf_symbol})")
            return None, False
        
        # Reset index and sort by date ascending
        df = df.reset_index()
        
        if 'Datetime' in df.columns:
            df['Date'] = df['Datetime'] 
            df = df.drop('Datetime', axis=1)
        
        # Ensure we have required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Missing required OHLC columns")
            return None, False
        
        # Handle volume
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        
        # Select and clean data
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.set_index('Date', inplace=True)
        
        # Sort by date ascending
        df = df.sort_index()
        
        # Convert to numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove NaN values
        df = df.dropna()
        
        if df.empty:
            st.error(f"No valid data after cleaning")
            return None, False
        
        # Check volume validity
        has_volume = df['Volume'].sum() > 0 and df['Volume'].std() > 0
        
        return df, has_volume
        
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {str(e)}")
        return None, False

def load_data_csv(uploaded_file):
    """Load and sort CSV data properly"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Find date column
        date_columns = [col for col in df.columns if any(word in col.lower() for word in ['date', 'time', 'timestamp'])]
        if date_columns:
            df[date_columns[0]] = pd.to_datetime(df[date_columns[0]], errors='coerce')
            df = df.set_index(date_columns[0])
            df = df.sort_index()  # Sort by date ascending
        
        # Standardize column names
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'open' in col_lower:
                column_mapping[col] = 'Open'
            elif 'high' in col_lower:
                column_mapping[col] = 'High'
            elif 'low' in col_lower:
                column_mapping[col] = 'Low'
            elif 'close' in col_lower:
                column_mapping[col] = 'Close'
            elif 'volume' in col_lower:
                column_mapping[col] = 'Volume'
        
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required OHLC columns. Found: {list(df.columns)}")
            return None, False
        
        # Add volume if missing
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        
        # Convert to numeric and clean
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        # Check volume validity  
        has_volume = df['Volume'].sum() > 0 and df['Volume'].std() > 0
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']], has_volume
        
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        return None, False

def create_enhanced_plot(df, trades_df=None):
    """Create comprehensive trading chart"""
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            'Price & Signals', 
            'Elite Score & Trend Scores', 
            'Momentum & Consistency',
            'Volume/Price Analysis',
            'Volatility & ATR'
        ),
        row_heights=[0.4, 0.2, 0.15, 0.15, 0.1]
    )
    
    # 1. Price chart with signals and EMAs
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'], 
        low=df['Low'],
        close=df['Close'],
        name="Price"
    ), row=1, col=1)
    
    # Add EMAs
    if 'EMA_21' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA_21'],
            mode='lines', name='EMA 21',
            line=dict(color='orange', width=1)
        ), row=1, col=1)
    
    if 'EMA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA_50'], 
            mode='lines', name='EMA 50',
            line=dict(color='blue', width=1)
        ), row=1, col=1)
    
    # Support/Resistance
    if 'Support' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Support'],
            mode='lines', name='Support',
            line=dict(color='green', width=1, dash='dash')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Resistance'],
            mode='lines', name='Resistance', 
            line=dict(color='red', width=1, dash='dash')
        ), row=1, col=1)
    
    # Entry signals
    long_signals = df[df['Long_Signal']] if 'Long_Signal' in df.columns else pd.DataFrame()
    short_signals = df[df['Short_Signal']] if 'Short_Signal' in df.columns else pd.DataFrame()
    
    if not long_signals.empty:
        fig.add_trace(go.Scatter(
            x=long_signals.index,
            y=long_signals['Close'],
            mode='markers',
            name='Long Entry',
            marker=dict(symbol='triangle-up', size=12, color='lime', line=dict(color='darkgreen', width=1))
        ), row=1, col=1)
    
    if not short_signals.empty:
        fig.add_trace(go.Scatter(
            x=short_signals.index,
            y=short_signals['Close'],
            mode='markers',
            name='Short Entry',
            marker=dict(symbol='triangle-down', size=12, color='red', line=dict(color='darkred', width=1))
        ), row=1, col=1)
    
    # 2. Elite Score and Trend Scores
    if 'Elite_Score' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Elite_Score'],
            mode='lines', name='Elite Score',
            line=dict(color='purple', width=2)
        ), row=2, col=1)
        
        fig.add_hline(y=0.75, line_dash="dash", line_color="orange", row=2, col=1)
    
    if 'Trend_Bullish_Score' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Trend_Bullish_Score'],
            mode='lines', name='Bullish Score',
            line=dict(color='green', width=1)
        ), row=2, col=1)
    
    if 'Trend_Bearish_Score' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Trend_Bearish_Score'],
            mode='lines', name='Bearish Score',
            line=dict(color='red', width=1)
        ), row=2, col=1)
    
    # 3. Momentum and Consistency
    if 'Momentum_Acceleration' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Momentum_Acceleration'],
            mode='lines', name='Momentum',
            line=dict(color='cyan', width=1)
        ), row=3, col=1)
        
        fig.add_hline(y=0, line_dash="dash", line_color="white", row=3, col=1)
    
    if 'Momentum_Consistency' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Momentum_Consistency'],
            mode='lines', name='Consistency',
            line=dict(color='yellow', width=1)
        ), row=3, col=1)
    
    # 4. Volume/Price Analysis
    if 'Volume_Confirmation' in df.columns:
        confirmation_points = df[df['Volume_Confirmation']]
        if not confirmation_points.empty:
            fig.add_trace(go.Scatter(
                x=confirmation_points.index,
                y=[1] * len(confirmation_points),
                mode='markers',
                name='Volume Confirmation',
                marker=dict(symbol='circle', size=6, color='orange')
            ), row=4, col=1)
    
    if 'Price_Change' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=np.abs(df['Price_Change']),
            mode='lines', name='Price Change %',
            line=dict(color='orange', width=1)
        ), row=4, col=1)
    
    # 5. Volatility
    if 'ATR' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['ATR'],
            mode='lines', name='ATR',
            line=dict(color='gray', width=1)
        ), row=5, col=1)
    
    if 'Volatility_Expansion' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Volatility_Expansion'],
            mode='lines', name='Volatility Expansion',
            line=dict(color='red', width=1)
        ), row=5, col=1)
        
        fig.add_hline(y=1.1, line_dash="dash", line_color="orange", row=5, col=1)
    
    # Update layout
    fig.update_layout(
        title="Enhanced VMA-Elite Trading System Analysis",
        height=1000,
        showlegend=True,
        template="plotly_dark"
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

def main():
    st.title("🚀 VMA-Elite Trading System v2.0")
    st.markdown("**Enhanced & Fixed** - Higher Accuracy with Proper Entry Logic")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'has_volume' not in st.session_state:
        st.session_state.has_volume = True
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Enhanced Configuration")
    
    # Data source
    st.sidebar.subheader("📊 Data Source")
    data_source = st.sidebar.radio("Select Source:", ["File Upload", "Yahoo Finance"], index=0)
    
    # Data loading
    uploaded_file = None
    symbol = ""
    period = "2y"
    interval = "1d"
    
    if data_source == "File Upload":
        uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")
    else:
        st.sidebar.subheader("🏛️ Market Selection")
        market_type = st.sidebar.selectbox(
            "Market Type:",
            ["Indian Indices", "Indian Stocks", "US Stocks", "Crypto", "Custom"]
        )
        
        if market_type == "Indian Indices":
            symbol = st.sidebar.selectbox(
                "Select Index:",
                ["NIFTY", "BANKNIFTY", "SENSEX"],
                index=0
            )
        elif market_type == "Indian Stocks":
            symbol = st.sidebar.text_input("Stock Symbol (e.g., TCS.NS):", "TCS.NS")
        elif market_type == "US Stocks":
            symbol = st.sidebar.text_input("Stock Symbol (e.g., AAPL):", "AAPL")
        elif market_type == "Crypto":
            symbol = st.sidebar.selectbox(
                "Select Crypto:",
                ["BTC-USD", "ETH-USD", "BNB-USD"],
                index=0
            )
        else:
            symbol = st.sidebar.text_input("Custom Symbol:", "")
        
        period = st.sidebar.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=2)
        interval = st.sidebar.selectbox("Interval", ["1d", "1wk"], index=0)
    
    # Enhanced strategy parameters
    st.sidebar.subheader("⚙️ Strategy Parameters") 
    momentum_length = st.sidebar.slider("Momentum Length", 14, 35, 21)
    sensitivity = st.sidebar.slider("Signal Sensitivity", 1.0, 2.5, 1.5, 0.1)
    min_elite_score = st.sidebar.slider("Min Elite Score", 0.6, 0.9, 0.75, 0.05)
    
    # Risk management
    st.sidebar.subheader("🛡️ Risk Management")
    stop_loss_pct = st.sidebar.slider("Stop Loss %", 0.8, 3.0, 1.5, 0.1)
    risk_reward_ratio = st.sidebar.slider("Risk:Reward Ratio", 2.0, 5.0, 3.0, 0.5)
    target_pct = stop_loss_pct * risk_reward_ratio
    
    st.sidebar.info(f"📊 Setup: SL: {stop_loss_pct:.1f}% | Target: {target_pct:.1f}% | R:R = 1:{risk_reward_ratio:.1f}")
    
    # Load data button
    fetch_button = st.sidebar.button("🔄 Load Data", type="primary", use_container_width=True)
    
    # Data status
    if st.session_state.data_loaded:
        volume_status = "✅ With Volume" if st.session_state.has_volume else "📊 Index Mode" 
        st.sidebar.success(f"✅ Data loaded: {volume_status}")
        if st.session_state.df is not None:
            st.sidebar.info(f"📊 Records: {len(st.session_state.df)}")
    else:
        st.sidebar.info("👆 Configure and click 'Load Data'")
    
    # Mode selection
    st.sidebar.subheader("📈 Analysis Mode")
    mode = st.sidebar.radio("Select Mode:", ["📊 Backtesting", "🎯 Live Analysis"])
    
    # Handle data loading
    if fetch_button or (data_source == "File Upload" and uploaded_file is not None):
        if data_source == "File Upload" and uploaded_file is not None:
            with st.spinner("Loading and sorting CSV data..."):
                df, has_volume = load_data_csv(uploaded_file)
        else:
            if symbol:
                with st.spinner(f"Loading {symbol} data..."):
                    df, has_volume = load_data_yfinance(symbol, period, interval)
            else:
                st.error("Please enter a symbol")
                return
        
        if df is not None and not df.empty:
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.session_state.has_volume = has_volume
            
            volume_msg = "with volume data" if has_volume else "in index mode (price-only)"
            st.success(f"✅ Loaded {len(df)} records {volume_msg}")
            st.info(f"📅 Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        else:
            st.session_state.data_loaded = False
            return
    
    # Check data availability
    if not st.session_state.data_loaded or st.session_state.df is None:
        st.info("👆 Configure settings and click 'Load Data' to start")
        
        # Show examples
        st.markdown("### 🇮🇳 Indian Market Examples")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Indices:**")
            st.markdown("- NIFTY (^NSEI)")
            st.markdown("- BANKNIFTY (^NSEBANK)")
            st.markdown("- SENSEX (^BSESN)")
        with col2:
            st.markdown("**Stocks:**")
            st.markdown("- TCS.NS")
            st.markdown("- RELIANCE.NS")
            st.markdown("- HDFCBANK.NS")
        with col3:
            st.markdown("**Features:**")
            st.markdown("- ✅ Fixed entry logic")
            st.markdown("- ✅ Proper data sorting") 
            st.markdown("- ✅ Enhanced accuracy")
        return
    
    df = st.session_state.df
    has_volume = st.session_state.has_volume
    
    # Initialize enhanced strategy
    strategy = VMAEliteStrategy(
        momentum_length=momentum_length,
        sensitivity=sensitivity,
        risk_reward_ratio=risk_reward_ratio,
        stop_loss_pct=stop_loss_pct,
        target_pct=target_pct,
        use_volume=has_volume,
        min_elite_score=min_elite_score
    )
    
    # Analysis based on mode
    if mode == "📊 Backtesting":
        st.header("📊 Enhanced Backtesting Results")
        
        with st.spinner("Running enhanced backtest with proper entry logic..."):
            processed_df, trades_df, performance = strategy.backtest(df)
        
        # Key performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            return_color = "normal" if performance['total_return'] > 0 else "inverse"
            vs_bh_color = "normal" if performance['strategy_vs_buyhold_pct'] > 0 else "inverse"
            st.metric(
                "Strategy Return",
                f"{performance['total_return']:.2f}%",
                delta=f"{performance['strategy_vs_buyhold_pct']:+.2f}% vs B&H",
                delta_color=vs_bh_color
            )
        
        with col2:
            st.metric(
                "Total Points",
                f"{performance['total_points']:+.1f}",
                delta=f"{performance['strategy_vs_buyhold_points']:+.1f} vs B&H"
            )
        
        with col3:
            win_color = "normal" if performance['win_rate'] >= 60 else "inverse"
            st.metric(
                "Win Rate",
                f"{performance['win_rate']:.1f}%",
                delta=f"{performance['winning_trades']}/{performance['total_trades']} trades"
            )
        
        with col4:
            pf_color = "normal" if performance['profit_factor'] >= 1.5 else "inverse"
            st.metric(
                "Profit Factor",
                f"{performance['profit_factor']:.2f}",
                delta=f"Sharpe: {performance['sharpe_ratio']:.2f}"
            )
        
        # Performance quality assessment
        st.subheader("🎯 Performance Quality Assessment")
        
        quality_score = 0
        quality_items = []
        
        if performance['total_return'] > 0:
            quality_score += 25
            quality_items.append("✅ Positive Returns")
        else:
            quality_items.append("❌ Negative Returns")
        
        if performance['win_rate'] >= 60:
            quality_score += 25  
            quality_items.append("✅ High Win Rate (≥60%)")
        elif performance['win_rate'] >= 40:
            quality_score += 15
            quality_items.append("⚠️ Moderate Win Rate (40-60%)")
        else:
            quality_items.append("❌ Low Win Rate (<40%)")
        
        if performance['profit_factor'] >= 1.5:
            quality_score += 25
            quality_items.append("✅ Good Profit Factor (≥1.5)")
        elif performance['profit_factor'] >= 1.0:
            quality_score += 15
            quality_items.append("⚠️ Break-even Profit Factor")
        else:
            quality_items.append("❌ Poor Profit Factor (<1.0)")
        
        if performance['strategy_vs_buyhold_pct'] > 0:
            quality_score += 25
            quality_items.append("✅ Outperforms Buy & Hold")
        else:
            quality_items.append("❌ Underperforms Buy & Hold")
        
        # Display quality assessment
        if quality_score >= 80:
            st.success(f"🏆 Excellent Strategy Performance ({quality_score}/100)")
        elif quality_score >= 60:
            st.warning(f"⚠️ Good Strategy Performance ({quality_score}/100)")
        else:
            st.error(f"❌ Poor Strategy Performance ({quality_score}/100) - Needs Improvement")
        
        for item in quality_items:
            st.write(f"  {item}")
        
        # Detailed comparison
        st.subheader("⚡ Strategy vs Buy & Hold Detailed Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🚀 VMA-Elite Strategy")
            st.write(f"**Final Return:** {performance['total_return']:.2f}%")
            st.write(f"**Total Points:** {performance['total_points']:+.1f}")
            st.write(f"**Number of Trades:** {performance['total_trades']}")
            st.write(f"**Winning Trades:** {performance['winning_trades']} ({performance['win_rate']:.1f}%)")
            st.write(f"**Losing Trades:** {performance['losing_trades']}")
            st.write(f"**Average Win:** {performance['avg_win_points']:+.1f} pts ({performance['avg_win_pct']:+.1f}%)")
            st.write(f"**Average Loss:** {performance['avg_loss_points']:+.1f} pts ({performance['avg_loss_pct']:+.1f}%)")
            st.write(f"**Max Drawdown:** {performance['max_drawdown']:.2f}%")
            st.write(f"**Profit Factor:** {performance['profit_factor']:.2f}")
        
        with col2:
            st.markdown("### 📈 Buy & Hold")
            st.write(f"**Final Return:** {performance['buy_hold_return']:.2f}%")
            st.write(f"**Total Points:** {performance['buy_hold_points']:+.1f}")
            st.write(f"**Number of Trades:** 1")
            st.write(f"**Strategy:** Buy and Hold")
            st.write(f"**Risk:** Market exposure")
            st.write(f"**Management:** None")
            st.write(f"**Drawdown:** Market dependent")
            st.write(f"**Discipline:** Passive")
        
        with col3:
            st.markdown("### 🎯 Performance Comparison")
            outperformance_pct = performance['strategy_vs_buyhold_pct']
            outperformance_pts = performance['strategy_vs_buyhold_points']
            
            if outperformance_pct > 0:
                st.success(f"🟢 Strategy OUTPERFORMS by {outperformance_pct:.2f}%")
                st.success(f"🟢 Point advantage: {outperformance_pts:+.1f}")
            else:
                st.error(f"🔴 Strategy UNDERPERFORMS by {abs(outperformance_pct):.2f}%")
                st.error(f"🔴 Point disadvantage: {outperformance_pts:+.1f}")
            
            st.write("**Strategy Benefits:**")
            if performance['total_trades'] > 0:
                st.write(f"- Active risk management")
                st.write(f"- {performance['max_consecutive_losses']} max consecutive losses")
                st.write(f"- Defined entry/exit rules")
                st.write(f"- Emotion-free trading")
            
        # Enhanced chart
        st.subheader("📈 Detailed Trading Analysis")
        fig = create_enhanced_plot(processed_df, trades_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade history
        if not trades_df.empty:
            st.subheader("💼 Complete Trade History")
            
            # Format for display
            display_trades = trades_df.copy()
            display_trades['Entry Date'] = pd.to_datetime(display_trades['entry_date']).dt.strftime('%Y-%m-%d')
            display_trades['Exit Date'] = pd.to_datetime(display_trades['exit_date']).dt.strftime('%Y-%m-%d')
            display_trades['Type'] = display_trades['type']
            display_trades['Entry'] = display_trades['entry_price'].round(2)
            display_trades['Exit'] = display_trades['exit_price'].round(2)
            display_trades['Points'] = display_trades['pnl_points'].round(1)
            display_trades['Return %'] = display_trades['pnl_percentage'].round(2)
            display_trades['Elite Score'] = display_trades['elite_score'].round(3)
            display_trades['Win Prob %'] = (display_trades['probability_profit'] * 100).round(1)
            display_trades['Stop %'] = display_trades['stop_pct'].round(2)
            display_trades['Target %'] = display_trades['target_pct'].round(2)
            display_trades['Entry Reason'] = display_trades['entry_reason']
            display_trades['Exit Reason'] = display_trades['exit_reason']
            
            # Color coding
            def color_trades(row):
                if row['Points'] > 0:
                    return ['background-color: rgba(34, 197, 94, 0.3)'] * len(row)
                else:
                    return ['background-color: rgba(239, 68, 68, 0.3)'] * len(row)
            
            display_columns = ['Entry Date', 'Type', 'Entry', 'Exit', 'Points', 'Return %',
                              'Elite Score', 'Win Prob %', 'Stop %', 'Target %', 'Entry Reason', 'Exit Reason']
            
            styled_df = display_trades[display_columns].style.apply(color_trades, axis=1)
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Trade analysis
            st.subheader("📊 Trade Performance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🎯 Winning Trades Analysis:**")
                winning = trades_df[trades_df['pnl_points'] > 0]
                if len(winning) > 0:
                    st.write(f"- Count: {len(winning)} trades")
                    st.write(f"- Win Rate: {len(winning)/len(trades_df)*100:.1f}%")
                    st.write(f"- Avg Points: {winning['pnl_points'].mean():.1f}")
                    st.write(f"- Avg Return: {winning['pnl_percentage'].mean():.1f}%")
                    st.write(f"- Best Trade: {winning['pnl_points'].max():.1f} points")
                    st.write(f"- Avg Elite Score: {winning['elite_score'].mean():.3f}")
                    st.write(f"- Target Hits: {len(winning[winning['exit_reason'] == 'Target Hit'])}")
                else:
                    st.write("- No winning trades")
            
            with col2:
                st.write("**❌ Losing Trades Analysis:**")
                losing = trades_df[trades_df['pnl_points'] <= 0]
                if len(losing) > 0:
                    st.write(f"- Count: {len(losing)} trades")
                    st.write(f"- Loss Rate: {len(losing)/len(trades_df)*100:.1f}%")
                    st.write(f"- Avg Points: {losing['pnl_points'].mean():.1f}")
                    st.write(f"- Avg Return: {losing['pnl_percentage'].mean():.1f}%")
                    st.write(f"- Worst Trade: {losing['pnl_points'].min():.1f} points")
                    st.write(f"- Avg Elite Score: {losing['elite_score'].mean():.3f}")
                    st.write(f"- Stop Loss Hits: {len(losing[losing['exit_reason'] == 'Stop Loss Hit'])}")
                    st.write(f"- Max Consecutive: {performance['max_consecutive_losses']}")
                else:
                    st.write("- No losing trades")
        
        else:
            st.warning("⚠️ No trades generated with current parameters!")
            st.info("**Try adjusting these settings:**")
            st.info("- Reduce minimum Elite Score (currently {:.2f})".format(min_elite_score))
            st.info("- Increase signal sensitivity (currently {:.1f})".format(sensitivity))
            st.info("- Reduce momentum length")
            st.info("- Check if data has sufficient volatility")
    
    else:  # Live Analysis Mode
        st.header("🎯 Live Trading Analysis & Recommendations")
        
        with st.spinner("Processing live data with enhanced indicators..."):
            processed_df = strategy.process_data(df)
        
        # Current market status
        if len(processed_df) < 2:
            st.error("Insufficient data for analysis")
            return
            
        latest = processed_df.iloc[-1]
        prev = processed_df.iloc[-2]
        
        st.subheader("📊 Current Market Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price_change = latest['Close'] - prev['Close']
            price_change_pct = (price_change / prev['Close']) * 100
            st.metric(
                "Current Price",
                f"{latest['Close']:.2f}",
                delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
            )
        
        with col2:
            elite_status = "🟢 High Quality" if latest['Elite_Score'] > min_elite_score else "🔴 Low Quality"
            st.metric(
                "Elite Score",
                f"{latest['Elite_Score']:.3f}",
                delta=f"Min: {min_elite_score}"
            )
        
        with col3:
            momentum_status = "🟢 Bullish" if latest['Momentum_Acceleration'] > 0 else "🔴 Bearish"
            st.metric(
                "Momentum",
                f"{latest['Momentum_Acceleration']:.2f}",
                delta=momentum_status
            )
        
        with col4:
            trend_bull = latest.get('Trend_Bullish_Score', 0)
            trend_bear = latest.get('Trend_Bearish_Score', 0)
            trend_status = f"🟢 Bullish ({trend_bull}/4)" if trend_bull >= 3 else f"🔴 Bearish ({trend_bear}/4)"
            st.metric(
                "Trend Strength",
                f"{max(trend_bull, trend_bear)}/4",
                delta=trend_status
            )
        
        # Live signal detection
        st.subheader("🚨 Live Signal Status")
        
        # Check recent signals (last 5 bars)
        recent_data = processed_df.tail(5)
        long_signals = recent_data[recent_data.get('Long_Signal', False)] if 'Long_Signal' in recent_data.columns else pd.DataFrame()
        short_signals = recent_data[recent_data.get('Short_Signal', False)] if 'Short_Signal' in recent_data.columns else pd.DataFrame()
        
        has_signals = not long_signals.empty or not short_signals.empty
        
        if has_signals:
            if not long_signals.empty:
                signal_data = long_signals.iloc[-1]
                st.success("🟢 **LONG SIGNAL DETECTED!**")
                
                # Calculate proper entry details
                entry_price = signal_data['Close']
                stop_loss = signal_data.get('Long_SL', entry_price * (1 - stop_loss_pct/100))
                target = signal_data.get('Long_Target', entry_price * (1 + target_pct/100))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📋 LONG Trade Setup:**")
                    st.write(f"- **Entry Price:** {entry_price:.2f}")
                    st.write(f"- **Stop Loss:** {stop_loss:.2f} ({stop_loss_pct:.1f}%)")
                    st.write(f"- **Target:** {target:.2f} ({target_pct:.1f}%)")
                    st.write(f"- **Risk/Reward:** 1:{risk_reward_ratio:.1f}")
                    st.write(f"- **Risk Points:** {entry_price - stop_loss:.1f}")
                    st.write(f"- **Reward Points:** {target - entry_price:.1f}")
                    st.write(f"- **Signal Date:** {signal_data.name.strftime('%Y-%m-%d')}")
                
                with col2:
                    st.write("**📊 Signal Quality:**")
                    st.write(f"- **Elite Score:** {signal_data['Elite_Score']:.3f} (Min: {min_elite_score})")
                    st.write(f"- **Success Probability:** {min(95, signal_data['Elite_Score'] * 110):.1f}%")
                    st.write(f"- **Trend Score:** {signal_data.get('Trend_Bullish_Score', 0)}/4")
                    st.write(f"- **Momentum:** {signal_data['Momentum_Acceleration']:.2f}")
                    st.write(f"- **Consistency:** {signal_data.get('Momentum_Consistency', 0):.2f}")
                    st.write(f"- **Volatility:** {signal_data.get('Volatility_Expansion', 1):.2f}x")
                
                st.write("**💡 Entry Reasoning:**")
                reason = strategy.get_entry_reason(signal_data, 'LONG')
                st.info(reason)
                
                # Position sizing recommendation
                st.write("**💰 Position Sizing Recommendation:**")
                account_sizes = [50000, 100000, 500000, 1000000]
                risk_per_trade = 1.0  # 1% risk
                
                cols = st.columns(len(account_sizes))
                for i, account_size in enumerate(account_sizes):
                    with cols[i]:
                        risk_amount = account_size * (risk_per_trade / 100)
                        position_size = risk_amount / (entry_price - stop_loss)
                        st.metric(
                            f"₹{account_size/1000:.0f}K Account",
                            f"{int(position_size)} shares",
                            delta=f"₹{risk_amount:.0f} risk"
                        )
            
            if not short_signals.empty:
                signal_data = short_signals.iloc[-1]
                st.error("🔴 **SHORT SIGNAL DETECTED!**")
                
                # Calculate proper entry details
                entry_price = signal_data['Close']
                stop_loss = signal_data.get('Short_SL', entry_price * (1 + stop_loss_pct/100))
                target = signal_data.get('Short_Target', entry_price * (1 - target_pct/100))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📋 SHORT Trade Setup:**")
                    st.write(f"- **Entry Price:** {entry_price:.2f}")
                    st.write(f"- **Stop Loss:** {stop_loss:.2f} ({stop_loss_pct:.1f}%)")
                    st.write(f"- **Target:** {target:.2f} ({target_pct:.1f}%)")
                    st.write(f"- **Risk/Reward:** 1:{risk_reward_ratio:.1f}")
                    st.write(f"- **Risk Points:** {stop_loss - entry_price:.1f}")
                    st.write(f"- **Reward Points:** {entry_price - target:.1f}")
                    st.write(f"- **Signal Date:** {signal_data.name.strftime('%Y-%m-%d')}")
                
                with col2:
                    st.write("**📊 Signal Quality:**")
                    st.write(f"- **Elite Score:** {signal_data['Elite_Score']:.3f} (Min: {min_elite_score})")
                    st.write(f"- **Success Probability:** {min(95, signal_data['Elite_Score'] * 110):.1f}%")
                    st.write(f"- **Trend Score:** {signal_data.get('Trend_Bearish_Score', 0)}/4")
                    st.write(f"- **Momentum:** {signal_data['Momentum_Acceleration']:.2f}")
                    st.write(f"- **Consistency:** {signal_data.get('Momentum_Consistency', 0):.2f}")
                    st.write(f"- **Volatility:** {signal_data.get('Volatility_Expansion', 1):.2f}x")
                
                st.write("**💡 Entry Reasoning:**")
                reason = strategy.get_entry_reason(signal_data, 'SHORT')
                st.info(reason)
        
        else:
            st.info("⏳ **No Active Signals - Market Analysis**")
            
            # Show what's missing for a signal
            st.write("**🔍 Signal Requirements Analysis:**")
            
            requirements = []
            
            # Elite Score check
            if latest['Elite_Score'] > min_elite_score:
                requirements.append(f"✅ Elite Score: {latest['Elite_Score']:.3f} > {min_elite_score}")
            else:
                requirements.append(f"❌ Elite Score: {latest['Elite_Score']:.3f} ≤ {min_elite_score}")
            
            # Trend checks
            trend_bull = latest.get('Trend_Bullish_Score', 0)
            trend_bear = latest.get('Trend_Bearish_Score', 0)
            
            if trend_bull >= 3:
                requirements.append(f"✅ Bullish Trend: {trend_bull}/4")
            elif trend_bear >= 3:
                requirements.append(f"✅ Bearish Trend: {trend_bear}/4")
            else:
                requirements.append(f"❌ Weak Trend: Bull {trend_bull}/4, Bear {trend_bear}/4")
            
            # Momentum check
            if abs(latest['Momentum_Acceleration']) > 0.5:
                direction = "Bullish" if latest['Momentum_Acceleration'] > 0 else "Bearish"
                requirements.append(f"✅ Strong Momentum: {latest['Momentum_Acceleration']:.2f} ({direction})")
            else:
                requirements.append(f"❌ Weak Momentum: {latest['Momentum_Acceleration']:.2f}")
            
            # Consistency check
            consistency = latest.get('Momentum_Consistency', 0)
            if consistency > 0.6:
                requirements.append(f"✅ Good Consistency: {consistency:.2f}")
            else:
                requirements.append(f"❌ Poor Consistency: {consistency:.2f}")
            
            # Volume confirmation
            vol_conf = latest.get('Volume_Confirmation', False)
            if vol_conf:
                requirements.append("✅ Volume/Price Confirmed")
            else:
                requirements.append("❌ No Volume/Price Confirmation")
            
            # Volatility
            vol_exp = latest.get('Volatility_Expansion', 1)
            if vol_exp > 1.1:
                requirements.append(f"✅ Volatility Expansion: {vol_exp:.2f}x")
            else:
                requirements.append(f"❌ Low Volatility: {vol_exp:.2f}x")
            
            for req in requirements:
                st.write(f"  {req}")
            
            # Next action suggestions
            st.write("**💡 Recommendations:**")
            if latest['Elite_Score'] < min_elite_score:
                st.write("- Wait for higher quality setup")
            if max(trend_bull, trend_bear) < 3:
                st.write("- Wait for clearer trend direction")
            if abs(latest['Momentum_Acceleration']) < 0.5:
                st.write("- Wait for stronger momentum")
            if consistency < 0.6:
                st.write("- Wait for more consistent price action")
        
        # Live chart
        st.subheader("📈 Live Market Chart")
        fig = create_enhanced_plot(processed_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Key levels and market structure
        st.subheader("🎯 Key Trading Levels & Market Structure")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📊 Price Levels:**")
            st.write(f"- **Current Price:** {latest['Close']:.2f}")
            if 'Resistance' in latest:
                st.write(f"- **Resistance:** {latest['Resistance']:.2f}")
            if 'Support' in latest:
                st.write(f"- **Support:** {latest['Support']:.2f}")
            if 'EMA_21' in latest:
                st.write(f"- **EMA 21:** {latest['EMA_21']:.2f}")
            if 'EMA_50' in latest:
                st.write(f"- **EMA 50:** {latest['EMA_50']:.2f}")
        
        with col2:
            st.write("**⚡ Volatility & Risk:**")
            if 'ATR' in latest:
                st.write(f"- **ATR:** {latest['ATR']:.2f}")
                atr_pct = (latest['ATR'] / latest['Close']) * 100
                st.write(f"- **ATR %:** {atr_pct:.2f}%")
            if 'Volatility_Expansion' in latest:
                st.write(f"- **Vol Expansion:** {latest['Volatility_Expansion']:.2f}x")
            st.write(f"- **Dynamic Stop:** {latest['Close'] * (1 - stop_loss_pct/100):.2f}")
            st.write(f"- **Dynamic Target:** {latest['Close'] * (1 + target_pct/100):.2f}")
        
        with col3:
            st.write(f"**📈 Market Analysis:**")
            if has_volume:
                st.write("- **Mode:** Volume + Price Analysis")
                if 'Volume_Ratio' in latest:
                    st.write(f"- **Volume Ratio:** {latest.get('Volume_Ratio', 1):.2f}")
            else:
                st.write("- **Mode:** Price-Only Analysis")
            
            if 'Price_Position' in latest:
                pos = latest['Price_Position'] * 100
                st.write(f"- **Price Position:** {pos:.1f}% of range")
            
            if 'BB_Squeeze' in latest:
                squeeze_status = "Yes" if latest['BB_Squeeze'] else "No"
                st.write(f"- **BB Squeeze:** {squeeze_status}")
        
        # Market timing advice
        st.subheader("⏰ Market Timing Advice")
        
        # Calculate overall market score
        market_score = 0
        scoring_factors = []
        
        if latest['Elite_Score'] > min_elite_score:
            market_score += 20
            scoring_factors.append("Elite Score ✅")
        else:
            scoring_factors.append("Elite Score ❌")
        
        if max(trend_bull, trend_bear) >= 3:
            market_score += 20
            scoring_factors.append("Trend Strength ✅")
        else:
            scoring_factors.append("Trend Strength ❌")
        
        if abs(latest['Momentum_Acceleration']) > 0.5:
            market_score += 20
            scoring_factors.append("Momentum ✅")
        else:
            scoring_factors.append("Momentum ❌")
        
        if latest.get('Momentum_Consistency', 0) > 0.6:
            market_score += 20
            scoring_factors.append("Consistency ✅")
        else:
            scoring_factors.append("Consistency ❌")
        
        if latest.get('Volume_Confirmation', False):
            market_score += 20
            scoring_factors.append("Volume/Price ✅")
        else:
            scoring_factors.append("Volume/Price ❌")
        
        # Market timing recommendation
        if market_score >= 80:
            st.success(f"🟢 **EXCELLENT MARKET CONDITIONS** ({market_score}/100)")
            st.success("💡 **Recommendation:** Ready to trade on next signal!")
        elif market_score >= 60:
            st.warning(f"⚠️ **GOOD MARKET CONDITIONS** ({market_score}/100)")
            st.warning("💡 **Recommendation:** Consider trading with smaller position size")
        elif market_score >= 40:
            st.info(f"🔵 **MODERATE MARKET CONDITIONS** ({market_score}/100)")
            st.info("💡 **Recommendation:** Wait for better setup or paper trade")
        else:
            st.error(f"🔴 **POOR MARKET CONDITIONS** ({market_score}/100)")
            st.error("💡 **Recommendation:** Avoid trading, wait for improvement")
        
        # Show scoring breakdown
        with st.expander("📊 Market Scoring Breakdown"):
            for factor in scoring_factors:
                st.write(f"• {factor}")

    # Footer with enhanced information
    st.markdown("---")
    st.markdown("### 🚀 VMA-Elite Trading System v2.0 - Enhanced Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**✅ Fixed Issues:**")
        st.markdown("- Proper entry timing")
        st.markdown("- Correct data sorting")
        st.markdown("- Enhanced signal quality")
        st.markdown("- Volume detection improved")
    
    with col2:
        st.markdown("**🎯 Performance Goals:**")
        st.markdown("- Target 60%+ win rate")
        st.markdown("- Positive absolute returns")
        st.markdown("- Outperform buy & hold")
        st.markdown("- Controlled drawdowns")
    
    with col3:
        st.markdown("**🛡️ Risk Management:**")
        st.markdown("- Dynamic stop losses")
        st.markdown("- Position sizing guidance")
        st.markdown("- Quality-based filtering")
        st.markdown("- Market condition assessment")
    
    volume_mode = "With Volume Analysis" if has_volume else "Index Mode (Price-Based)"
    st.markdown(f"**Current Mode:** {volume_mode}")
    st.markdown("⚠️ *Trading involves risk. Past performance does not guarantee future results.*")
    st.markdown("🇮🇳 *Enhanced for Indian Markets - Nifty, BankNifty, Sensex*")

if __name__ == "__main__":
    main()
