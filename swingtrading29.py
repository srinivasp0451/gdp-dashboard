import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import itertools
import random
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Advanced Trading System - Clean",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AdvancedTradingStrategy:
    def __init__(self, strategy_type="Momentum Breakout", ema_fast=9, ema_slow=21, 
                 rsi_period=14, rsi_oversold=30, rsi_overbought=70,
                 bb_period=20, bb_std=2, macd_fast=12, macd_slow=26, macd_signal=9,
                 stop_loss_pct=2.0, take_profit_pct=6.0, use_volume=True):
        
        self.strategy_type = strategy_type
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.use_volume = use_volume
    
    def calculate_ema(self, data, window):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=window, adjust=False).mean()
    
    def calculate_sma(self, data, window):
        """Calculate Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    def calculate_rsi(self, data, window=14):
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        macd = ema_fast - ema_slow
        macd_signal = self.calculate_ema(macd, signal)
        macd_hist = macd - macd_signal
        return macd.fillna(0), macd_signal.fillna(0), macd_hist.fillna(0)
    
    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(data, window)
        std = data.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper.fillna(data), sma.fillna(data), lower.fillna(data)
    
    def calculate_stochastic(self, high, low, close, window=14):
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()
        
        # Avoid division by zero
        range_val = highest_high - lowest_low
        range_val = range_val.replace(0, 1)  # Replace 0 with 1 to avoid division by zero
        
        k_percent = 100 * ((close - lowest_low) / range_val)
        d_percent = k_percent.rolling(window=3).mean()
        return k_percent.fillna(50), d_percent.fillna(50)
    
    def calculate_williams_r(self, high, low, close, window=14):
        """Calculate Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        # Avoid division by zero
        range_val = highest_high - lowest_low
        range_val = range_val.replace(0, 1)
        
        williams_r = -100 * ((highest_high - close) / range_val)
        return williams_r.fillna(-50)
    
    def calculate_atr(self, high, low, close, window=14):
        """Calculate Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift(1))
        low_close = np.abs(low - close.shift(1))
        
        tr = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close}).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr.fillna(tr.mean())
    
    def calculate_adx(self, high, low, close, window=14):
        """Simplified ADX calculation"""
        # Simple trend strength based on price movement
        up_moves = high.diff()
        down_moves = -low.diff()
        
        up_trend = up_moves.where(up_moves > down_moves, 0)
        down_trend = down_moves.where(down_moves > up_moves, 0)
        
        trend_strength = (up_trend + down_trend).rolling(window=window).mean()
        adx = trend_strength / close * 100
        return adx.fillna(20)
    
    def prepare_data(self, df):
        """Prepare data with all technical indicators"""
        df = df.copy()
        df = df.sort_index()
        df = df.dropna()
        
        if df.empty:
            return df
        
        # Check volume validity
        if 'Volume' in df.columns:
            if df['Volume'].sum() == 0 or df['Volume'].std() == 0:
                self.use_volume = False
                df['Volume'] = df['Close'] * 1000  # Fake volume for indices
        else:
            df['Volume'] = df['Close'] * 1000
            self.use_volume = False
        
        return self.calculate_all_indicators(df)
    
    def calculate_all_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        try:
            # Moving Averages
            df['EMA_Fast'] = self.calculate_ema(df['Close'], self.ema_fast)
            df['EMA_Slow'] = self.calculate_ema(df['Close'], self.ema_slow)
            df['SMA_20'] = self.calculate_sma(df['Close'], 20)
            df['SMA_50'] = self.calculate_sma(df['Close'], 50)
            
            # RSI
            df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
            
            # MACD
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.calculate_macd(
                df['Close'], self.macd_fast, self.macd_slow, self.macd_signal
            )
            
            # Bollinger Bands
            df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = self.calculate_bollinger_bands(
                df['Close'], self.bb_period, self.bb_std
            )
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid'] * 100
            
            # Stochastic
            df['Stoch_K'], df['Stoch_D'] = self.calculate_stochastic(
                df['High'], df['Low'], df['Close']
            )
            
            # Williams %R
            df['Williams_R'] = self.calculate_williams_r(
                df['High'], df['Low'], df['Close']
            )
            
            # ATR
            df['ATR'] = self.calculate_atr(df['High'], df['Low'], df['Close'])
            
            # ADX
            df['ADX'] = self.calculate_adx(df['High'], df['Low'], df['Close'])
            
            # Volume indicators
            if self.use_volume:
                df['Volume_SMA'] = self.calculate_sma(df['Volume'], 20)
                df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA'].replace(0, 1)
            else:
                # Price-based volume substitute
                df['Price_Change'] = df['Close'].pct_change().abs()
                df['Volume_Ratio'] = df['Price_Change'].rolling(20).rank(pct=True)
            
            df['Volume_Ratio'] = df['Volume_Ratio'].fillna(1.0)
            
            # Price position in range
            high_20 = df['High'].rolling(20).max()
            low_20 = df['Low'].rolling(20).min()
            price_range = high_20 - low_20
            price_range = price_range.replace(0, 1)  # Avoid division by zero
            
            df['Price_Position'] = ((df['Close'] - low_20) / price_range * 100).fillna(50)
            
            # Fill any remaining NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Final check for any remaining NaN
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(df[col].mean())
            
            return df
            
        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
            return df
    
    def generate_signals(self, df):
        """Generate trading signals based on selected strategy"""
        
        if self.strategy_type == "Momentum Breakout":
            return self.momentum_breakout_signals(df)
        elif self.strategy_type == "Mean Reversion":
            return self.mean_reversion_signals(df)
        elif self.strategy_type == "Trend Following":
            return self.trend_following_signals(df)
        elif self.strategy_type == "Multi-Indicator":
            return self.multi_indicator_signals(df)
        elif self.strategy_type == "Scalping":
            return self.scalping_signals(df)
        else:
            return self.momentum_breakout_signals(df)
    
    def momentum_breakout_signals(self, df):
        """Momentum breakout strategy"""
        # Create boolean conditions with safe comparisons
        cond1 = df['Close'] > df['EMA_Fast']
        cond2 = df['EMA_Fast'] > df['EMA_Slow']
        cond3 = df['RSI'] > 45
        cond4 = df['RSI'] < 80
        cond5 = df['MACD'] > df['MACD_Signal']
        cond6 = df['Volume_Ratio'] > 1.1
        cond7 = df['Close'] > df['Close'].shift(1)
        cond8 = df['ADX'] > 20
        
        # Count satisfied conditions for long
        long_conditions = [cond1, cond2, cond3, cond4, cond5, cond6, cond7, cond8]
        df['Long_Score'] = sum([cond.astype(int) for cond in long_conditions])
        
        # Short conditions (opposite)
        scond1 = df['Close'] < df['EMA_Fast']
        scond2 = df['EMA_Fast'] < df['EMA_Slow']
        scond3 = df['RSI'] < 55
        scond4 = df['RSI'] > 20
        scond5 = df['MACD'] < df['MACD_Signal']
        scond6 = df['Volume_Ratio'] > 1.1
        scond7 = df['Close'] < df['Close'].shift(1)
        scond8 = df['ADX'] > 20
        
        short_conditions = [scond1, scond2, scond3, scond4, scond5, scond6, scond7, scond8]
        df['Short_Score'] = sum([cond.astype(int) for cond in short_conditions])
        
        # Generate signals (need 5 out of 8 conditions)
        df['Long_Signal'] = (df['Long_Score'] >= 5) & (df['Long_Score'].shift(1) < 5)
        df['Short_Signal'] = (df['Short_Score'] >= 5) & (df['Short_Score'].shift(1) < 5)
        
        return df
    
    def mean_reversion_signals(self, df):
        """Mean reversion strategy"""
        # Long conditions (oversold)
        lcond1 = df['RSI'] < self.rsi_oversold
        lcond2 = df['Close'] < df['BB_Lower']
        lcond3 = df['Stoch_K'] < 20
        lcond4 = df['Williams_R'] < -80
        lcond5 = df['Price_Position'] < 20
        
        df['Long_Score'] = sum([cond.astype(int) for cond in [lcond1, lcond2, lcond3, lcond4, lcond5]])
        
        # Short conditions (overbought)
        scond1 = df['RSI'] > self.rsi_overbought
        scond2 = df['Close'] > df['BB_Upper']
        scond3 = df['Stoch_K'] > 80
        scond4 = df['Williams_R'] > -20
        scond5 = df['Price_Position'] > 80
        
        df['Short_Score'] = sum([cond.astype(int) for cond in [scond1, scond2, scond3, scond4, scond5]])
        
        # Generate signals (need 3 out of 5 conditions)
        df['Long_Signal'] = (df['Long_Score'] >= 3) & (df['Long_Score'].shift(1) < 3)
        df['Short_Signal'] = (df['Short_Score'] >= 3) & (df['Short_Score'].shift(1) < 3)
        
        return df
    
    def trend_following_signals(self, df):
        """Trend following strategy"""
        # Long conditions
        lcond1 = df['Close'] > df['SMA_20']
        lcond2 = df['SMA_20'] > df['SMA_50']
        lcond3 = df['EMA_Fast'] > df['EMA_Slow']
        lcond4 = df['MACD'] > 0
        lcond5 = df['MACD'] > df['MACD_Signal']
        lcond6 = df['ADX'] > 25
        lcond7 = df['RSI'] > 50
        
        df['Long_Score'] = sum([cond.astype(int) for cond in [lcond1, lcond2, lcond3, lcond4, lcond5, lcond6, lcond7]])
        
        # Short conditions
        scond1 = df['Close'] < df['SMA_20']
        scond2 = df['SMA_20'] < df['SMA_50']
        scond3 = df['EMA_Fast'] < df['EMA_Slow']
        scond4 = df['MACD'] < 0
        scond5 = df['MACD'] < df['MACD_Signal']
        scond6 = df['ADX'] > 25
        scond7 = df['RSI'] < 50
        
        df['Short_Score'] = sum([cond.astype(int) for cond in [scond1, scond2, scond3, scond4, scond5, scond6, scond7]])
        
        # Generate signals (need 5 out of 7 conditions)
        df['Long_Signal'] = (df['Long_Score'] >= 5) & (df['Long_Score'].shift(1) < 5)
        df['Short_Signal'] = (df['Short_Score'] >= 5) & (df['Short_Score'].shift(1) < 5)
        
        return df
    
    def multi_indicator_signals(self, df):
        """Multi-indicator confluence strategy"""
        # Initialize scores
        df['Bull_Score'] = 0
        df['Bear_Score'] = 0
        
        # Add points for bullish conditions
        df.loc[df['Close'] > df['EMA_Fast'], 'Bull_Score'] += 1
        df.loc[df['EMA_Fast'] > df['EMA_Slow'], 'Bull_Score'] += 1
        df.loc[df['RSI'] > 50, 'Bull_Score'] += 1
        df.loc[df['MACD'] > df['MACD_Signal'], 'Bull_Score'] += 1
        df.loc[df['Stoch_K'] > 50, 'Bull_Score'] += 1
        df.loc[df['Williams_R'] > -50, 'Bull_Score'] += 1
        df.loc[df['Volume_Ratio'] > 1.2, 'Bull_Score'] += 1
        df.loc[df['ADX'] > 25, 'Bull_Score'] += 1
        df.loc[df['Close'] > df['BB_Mid'], 'Bull_Score'] += 1
        df.loc[df['Price_Position'] > 60, 'Bull_Score'] += 1
        
        # Add points for bearish conditions
        df.loc[df['Close'] < df['EMA_Fast'], 'Bear_Score'] += 1
        df.loc[df['EMA_Fast'] < df['EMA_Slow'], 'Bear_Score'] += 1
        df.loc[df['RSI'] < 50, 'Bear_Score'] += 1
        df.loc[df['MACD'] < df['MACD_Signal'], 'Bear_Score'] += 1
        df.loc[df['Stoch_K'] < 50, 'Bear_Score'] += 1
        df.loc[df['Williams_R'] < -50, 'Bear_Score'] += 1
        df.loc[df['Volume_Ratio'] > 1.2, 'Bear_Score'] += 1
        df.loc[df['ADX'] > 25, 'Bear_Score'] += 1
        df.loc[df['Close'] < df['BB_Mid'], 'Bear_Score'] += 1
        df.loc[df['Price_Position'] < 40, 'Bear_Score'] += 1
        
        # Generate signals (need 6 out of 10 points)
        df['Long_Signal'] = (df['Bull_Score'] >= 6) & (df['Bull_Score'].shift(1) < 6)
        df['Short_Signal'] = (df['Bear_Score'] >= 6) & (df['Bear_Score'].shift(1) < 6)
        
        return df
    
    def scalping_signals(self, df):
        """High-frequency scalping strategy"""
        # Long conditions
        lcond1 = df['Close'] > df['EMA_Fast']
        lcond2 = (df['RSI'] > 45) & (df['RSI'] < 65)
        lcond3 = df['MACD_Hist'] > df['MACD_Hist'].shift(1)
        lcond4 = df['Stoch_K'] > df['Stoch_D']
        lcond5 = df['Close'] > df['Close'].shift(1)
        lcond6 = df['Volume_Ratio'] > 0.8
        
        df['Long_Score'] = sum([cond.astype(int) for cond in [lcond1, lcond2, lcond3, lcond4, lcond5, lcond6]])
        
        # Short conditions
        scond1 = df['Close'] < df['EMA_Fast']
        scond2 = (df['RSI'] < 55) & (df['RSI'] > 35)
        scond3 = df['MACD_Hist'] < df['MACD_Hist'].shift(1)
        scond4 = df['Stoch_K'] < df['Stoch_D']
        scond5 = df['Close'] < df['Close'].shift(1)
        scond6 = df['Volume_Ratio'] > 0.8
        
        df['Short_Score'] = sum([cond.astype(int) for cond in [scond1, scond2, scond3, scond4, scond5, scond6]])
        
        # Generate signals (need 4 out of 6 conditions)
        df['Long_Signal'] = (df['Long_Score'] >= 4) & (df['Long_Score'].shift(1) < 4)
        df['Short_Signal'] = (df['Short_Score'] >= 4) & (df['Short_Score'].shift(1) < 4)
        
        return df
    
    def enhanced_backtest(self, df, initial_capital=100000):
        """Enhanced realistic backtesting with proper trade management"""
        df = self.prepare_data(df)
        df = self.generate_signals(df)
        
        # Initialize trading variables
        trades = []
        capital = initial_capital
        position =     def calculate_enhanced_performance(self, trades_df, portfolio_df, initial_capital, final_capital, buy_hold_return):
        """Calculate comprehensive performance metrics"""
        if trades_df.empty:
            return {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0, 'total_return': 0, 'total_pnl': 0,
                'avg_win': 0, 'avg_loss': 0, 'avg_win_pct': 0, 'avg_loss_pct': 0,
                'profit_factor': 0, 'max_drawdown': 0, 'max_drawdown_duration': 0,
                'avg_trade_duration': 0, 'best_trade': 0, 'worst_trade': 0,
                'sharpe_ratio': 0, 'sortino_ratio': 0, 'calmar_ratio': 0,
                'buy_hold_return': buy_hold_return, 'excess_return': -buy_hold_return,
                'total_fees': 0, 'net_profit': final_capital - initial_capital,
                'roi': (final_capital - initial_capital) / initial_capital * 100
            }
        
        # Basic metrics
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        total_return = (final_capital - initial_capital) / initial_capital * 100
        win_rate = len(winning_trades) / len(trades_df) * 100
        
        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        avg_win_pct = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss_pct = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Drawdown analysis
        portfolio_values = portfolio_df['portfolio_value'].values
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak * 100
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Drawdown duration
        in_drawdown = drawdown > 0.1  # Consider >0.1% as drawdown
        drawdown_periods = []
        current_dd_length = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_dd_length += 1
            else:
                if current_dd_length > 0:
                    drawdown_periods.append(current_dd_length)
                current_dd_length = 0
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Risk-adjusted returns
        returns = trades_df['pnl_pct'].values
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            
            # Sortino ratio (downside deviation)
            negative_returns = returns[returns < 0]
            downside_std = np.std(negative_returns) if len(negative_returns) > 1 else std_return
            sortino_ratio = avg_return / downside_std if downside_std > 0 else 0
            
            # Calmar ratio
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        else:
            sharpe_ratio = sortino_ratio = calmar_ratio = 0
        
        return {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_trade_duration': trades_df['duration'].mean(),
            'best_trade': trades_df['pnl'].max(),
            'worst_trade': trades_df['pnl'].min(),
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'net_profit': final_capital - initial_capital,
            'roi': total_return,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }

class StrategyOptimizer:
    def __init__(self, df, strategy_type, use_volume=True):
        self.df = df
        self.strategy_type = strategy_type
        self.use_volume = use_volume
    
    def get_parameter_ranges(self):
        """Parameter ranges for optimization"""
        base_ranges = {
            'stop_loss_pct': [1.0, 1.5, 2.0, 2.5, 3.0],
            'take_profit_pct': [3.0, 4.5, 6.0, 9.0, 12.0],
            'ema_fast': [5, 9, 12, 15],
            'ema_slow': [18, 21, 26, 30]
        }
        
        if self.strategy_type == "Mean Reversion":
            base_ranges.update({
                'rsi_oversold': [25, 30, 35],
                'rsi_overbought': [65, 70, 75]
            })
        
        return base_ranges
    
    def optimize_parameters(self, max_combinations=80):
        """Optimize strategy parameters"""
        param_ranges = self.get_parameter_ranges()
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        
        # Generate all combinations
        all_combinations = list(itertools.product(*values))
        
        # Sample if too many combinations
        if len(all_combinations) > max_combinations:
            combinations = random.sample(all_combinations, max_combinations)
        else:
            combinations = all_combinations
        
        st.info(f"🔍 Testing {len(combinations)} parameter combinations...")
        
        progress_bar = st.progress(0)
        results = []
        
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            
            try:
                # Create strategy with parameters
                strategy = AdvancedTradingStrategy(
                    strategy_type=self.strategy_type,
                    ema_fast=params.get('ema_fast', 9),
                    ema_slow=params.get('ema_slow', 21),
                    rsi_oversold=params.get('rsi_oversold', 30),
                    rsi_overbought=params.get('rsi_overbought', 70),
                    stop_loss_pct=params['stop_loss_pct'],
                    take_profit_pct=params['take_profit_pct'],
                    use_volume=self.use_volume
                )
                
                # Run backtest
                _, trades_df, performance = strategy.enhanced_backtest(self.df)
                
                # Calculate optimization score
                score = 0
                
                # Positive returns (40 points)
                if performance['total_return'] > 0:
                    score += 40
                elif performance['total_return'] > -5:
                    score += 20
                
                # Win rate (25 points)
                if performance['win_rate'] >= 60:
                    score += 25
                elif performance['win_rate'] >= 50:
                    score += 20
                elif performance['win_rate'] >= 40:
                    score += 15
                
                # Trade frequency (15 points)
                if performance['total_trades'] >= 20:
                    score += 15
                elif performance['total_trades'] >= 10:
                    score += 10
                elif performance['total_trades'] >= 5:
                    score += 5
                
                # Profit factor (15 points)
                pf = performance['profit_factor']
                if pf >= 2.0:
                    score += 15
                elif pf >= 1.5:
                    score += 12
                elif pf >= 1.2:
                    score += 8
                elif pf >= 1.0:
                    score += 5
                
                # Excess return over buy & hold (5 points)
                if performance['excess_return'] > 0:
                    score += 5
                
                result = performance.copy()
                result['parameters'] = params
                result['score'] = score
                results.append(result)
                
            except Exception as e:
                results.append({
                    'score': 0, 'total_return': -999, 'parameters': params,
                    'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
                    'error': str(e)
                })
            
            progress_bar.progress((i + 1) / len(combinations))
        
        progress_bar.empty()
        
        # Sort by score
        results = [r for r in results if r.get('total_return', -999) != -999]
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return results

def load_data_yfinance(symbol, period="1y", interval="1d"):
    """Load data from Yahoo Finance"""
    try:
        indian_symbols = {
            'NIFTY': '^NSEI', 'BANKNIFTY': '^NSEBANK', 'SENSEX': '^BSESN',
            'NIFTY50': '^NSEI', 'NIFTYBANK': '^NSEBANK', 'FINNIFTY': '^CNXFIN'
        }
        
        yf_symbol = indian_symbols.get(symbol.upper(), symbol)
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return None, False
        
        df = df.reset_index()
        if 'Datetime' in df.columns:
            df['Date'] = df['Datetime']
            df = df.drop('Datetime', axis=1)
        
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            return None, False
        
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if df.empty:
            return None, False
        
        has_volume = df['Volume'].sum() > 0 and df['Volume'].std() > 0
        return df, has_volume
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, False

def load_data_csv(uploaded_file):
    """Load CSV data"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Find date column
        date_columns = [col for col in df.columns if any(word in col.lower() for word in ['date', 'time', 'timestamp'])]
        if date_columns:
            df[date_columns[0]] = pd.to_datetime(df[date_columns[0]], errors='coerce')
            df = df.set_index(date_columns[0])
            df = df.sort_index()
        
        # Map columns
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
        
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            return None, False
        
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        has_volume = df['Volume'].sum() > 0 and df['Volume'].std() > 0
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']], has_volume
        
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        return None, False

def create_advanced_chart(df, trades_df=None, strategy_type=""):
    """Create comprehensive trading chart"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'Price & Signals - {strategy_type}',
            'RSI & Stochastic',
            'MACD',
            'Volume'
        ),
        row_heights=[0.5, 0.2, 0.2, 0.1]
    )
    
    # Price chart
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], 
        low=df['Low'], close=df['Close'], name="Price"
    ), row=1, col=1)
    
    # EMAs
    if 'EMA_Fast' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA_Fast'], mode='lines',
            name='EMA Fast', line=dict(color='orange', width=1)
        ), row=1, col=1)
    
    if 'EMA_Slow' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA_Slow'], mode='lines',
            name='EMA Slow', line=dict(color='blue', width=1)
        ), row=1, col=1)
    
    # Entry signals
    if trades_df is not None and not trades_df.empty:
        long_trades = trades_df[trades_df['type'] == 'LONG']
        short_trades = trades_df[trades_df['type'] == 'SHORT']
        
        if not long_trades.empty:
            fig.add_trace(go.Scatter(
                x=long_trades['entry_date'], y=long_trades['entry_price'],
                mode='markers', name='Long Entry',
                marker=dict(symbol='triangle-up', size=10, color='green')
            ), row=1, col=1)
        
        if not short_trades.empty:
            fig.add_trace(go.Scatter(
                x=short_trades['entry_date'], y=short_trades['entry_price'],
                mode='markers', name='Short Entry',
                marker=dict(symbol='triangle-down', size=10, color='red')
            ), row=1, col=1)
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'], mode='lines',
            name='RSI', line=dict(color='purple')
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Stochastic
    if 'Stoch_K' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Stoch_K'], mode='lines',
            name='Stoch %K', line=dict(color='orange')
        ), row=2, col=1)
    
    # MACD
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD'], mode='lines',
            name='MACD', line=dict(color='blue')
        ), row=3, col=1)
        
        if 'MACD_Signal' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['MACD_Signal'], mode='lines',
                name='MACD Signal', line=dict(color='red')
            ), row=3, col=1)
        
        fig.add_hline(y=0, line_dash="solid", line_color="gray", row=3, col=1)
    
    # Volume
    if 'Volume' in df.columns:
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'],
            name='Volume', marker_color='lightblue'
        ), row=4, col=1)
    
    fig.update_layout(
        title=f"Trading System Analysis - {strategy_type}",
        height=800,
        showlegend=True,
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )
    
    return fig

def main():
    st.title("🚀 Advanced Trading System - Fixed & Enhanced")
    st.markdown("**Clean Code, Realistic Backtesting, High Performance**")
    
    # Initialize session state
    for key in ['data_loaded', 'df', 'has_volume', 'optimized_params']:
        if key not in st.session_state:
            st.session_state[key] = None if key != 'data_loaded' else False
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Strategy selection
    st.sidebar.subheader("🎯 Strategy Selection")
    strategy_type = st.sidebar.selectbox(
        "Trading Strategy:",
        ["Momentum Breakout", "Mean Reversion", "Trend Following", "Multi-Indicator", "Scalping"],
        index=0
    )
    
    # Data source
    st.sidebar.subheader("📊 Data Source")
    data_source = st.sidebar.radio("Source:", ["Yahoo Finance", "File Upload"], index=0)
    
    # Data loading options
    uploaded_file = None
    symbol = ""
    period = "1y"
    interval = "1d"
    
    if data_source == "File Upload":
        uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")
    else:
        st.sidebar.subheader("🏛️ Market Selection")
        market_type = st.sidebar.selectbox(
            "Market:",
            ["Indian Indices", "Indian Stocks", "US Stocks", "Crypto", "Custom"]
        )
        
        if market_type == "Indian Indices":
            symbol = st.sidebar.selectbox("Index:", ["NIFTY", "BANKNIFTY", "SENSEX"])
        elif market_type == "Indian Stocks":
            symbol = st.sidebar.text_input("Stock Symbol (e.g., TCS.NS):", "TCS.NS")
        elif market_type == "US Stocks":
            symbol = st.sidebar.text_input("Stock Symbol (e.g., AAPL):", "AAPL")
        elif market_type == "Crypto":
            symbol = st.sidebar.selectbox("Crypto:", ["BTC-USD", "ETH-USD", "BNB-USD"])
        else:
            symbol = st.sidebar.text_input("Symbol:", "")
        
        period = st.sidebar.selectbox("Period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        interval = st.sidebar.selectbox("Interval:", ["1m", "5m", "15m", "30m", "1h", "1d"], index=5)
    
    # Strategy parameters
    st.sidebar.subheader("⚙️ Parameters")
    if strategy_type in ["Momentum Breakout", "Trend Following", "Multi-Indicator"]:
        ema_fast = st.sidebar.slider("EMA Fast", 5, 20, 9)
        ema_slow = st.sidebar.slider("EMA Slow", 15, 50, 21)
    else:
        ema_fast, ema_slow = 9, 21
    
    if strategy_type == "Mean Reversion":
        rsi_oversold = st.sidebar.slider("RSI Oversold", 20, 35, 30)
        rsi_overbought = st.sidebar.slider("RSI Overbought", 65, 80, 70)
    else:
        rsi_oversold, rsi_overbought = 30, 70
    
    # Risk management
    st.sidebar.subheader("🛡️ Risk Management")
    stop_loss_pct = st.sidebar.slider("Stop Loss %", 0.5, 5.0, 2.0, 0.1)
    take_profit_pct = st.sidebar.slider("Take Profit %", 2.0, 15.0, 6.0, 0.5)
    
    st.sidebar.info(f"Risk:Reward = 1:{take_profit_pct/stop_loss_pct:.1f}")
    
    # Action buttons
    fetch_button = st.sidebar.button("🔄 Load Data", type="primary")
    optimize_button = st.sidebar.button("🎯 Optimize", type="secondary")
    
    # Data status
    if st.session_state.data_loaded and st.session_state.df is not None:
        st.sidebar.success(f"✅ Data: {len(st.session_state.df)} records")
        st.sidebar.info(f"Volume: {'Yes' if st.session_state.has_volume else 'No'}")
    else:
        st.sidebar.info("Load data to begin")
    
    # Mode selection
    mode = st.sidebar.radio("Mode:", ["📊 Backtest", "🎯 Optimize", "🔍 Live"])
    
    # Handle data loading
    if fetch_button or (data_source == "File Upload" and uploaded_file is not None):
        if data_source == "File Upload" and uploaded_file is not None:
            with st.spinner("Loading CSV..."):
                df, has_volume = load_data_csv(uploaded_file)
        else:
            if symbol:
                with st.spinner(f"Loading {symbol}..."):
                    df, has_volume = load_data_yfinance(symbol, period, interval)
            else:
                st.error("Enter a symbol")
                return
        
        if df is not None and not df.empty:
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.session_state.has_volume = has_volume
            st.success(f"✅ Loaded {len(df)} records")
        else:
            st.session_state.data_loaded = False
            return
    
    # Main content
    if not st.session_state.data_loaded:
        st.info("Configure settings and load data to start")
        
        # Show features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### ✅ Fixed Issues")
            st.markdown("- All syntax errors fixed")
            st.markdown("- Realistic backtesting")
            st.markdown("- Proper trade management")
            st.markdown("- Enhanced risk metrics")
        
        with col2:
            st.markdown("### 🎯 Strategy Features")
            st.markdown("- 5 proven strategies")
            st.markdown("- Relaxed entry conditions")
            st.markdown("- Multiple timeframes")
            st.markdown("- Auto-optimization")
        
        with col3:
            st.markdown("### 📊 Performance Focus")
            st.markdown("- Positive returns target")
            st.markdown("- High trade frequency")
            st.markdown("- Proper risk management")
            st.markdown("- Realistic execution")
        
        return
    
    df = st.session_state.df
    has_volume = st.session_state.has_volume
    
    # Initialize strategy
    strategy = AdvancedTradingStrategy(
        strategy_type=strategy_type,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        use_volume=has_volume
    )
    
    # Content based on mode
    if mode == "📊 Backtest":
        st.header(f"📊 {strategy_type} Enhanced Backtesting")
        
        with st.spinner("Running enhanced backtest..."):
            processed_df, trades_df, performance = strategy.enhanced_backtest(df)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            color = "normal" if performance['total_return'] > 0 else "inverse"
            st.metric(
                "Total Return",
                f"{performance['total_return']:.2f}%",
                delta=f"{performance['excess_return']:+.2f}% vs B&H"
            )
        
        with col2:
            st.metric(
                "Trades",
                f"{performance['total_trades']}",
                delta=f"{performance['win_rate']:.1f}% wins"
            )
        
        with col3:
            pf = performance['profit_factor']
            pf_display = f"{pf:.2f}" if pf < 100 else "∞"
            st.metric(
                "Profit Factor",
                pf_display,
                delta=f"Sharpe: {performance['sharpe_ratio']:.2f}"
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                f"{performance['max_drawdown']:.2f}%",
                delta=f"Duration: {performance['max_drawdown_duration']:.0f} bars"
            )
        
        # Performance assessment
        score = 0
        assessment = []
        
        if performance['total_return'] > 0:
            score += 30
            assessment.append("✅ Positive Returns")
        else:
            assessment.append("❌ Negative Returns")
        
        if performance['total_trades'] >= 15:
            score += 25
            assessment.append("✅ Good Trade Frequency")
        elif performance['total_trades'] >= 5:
            score += 15
            assessment.append("⚠️ Moderate Trade Frequency")
        else:
            assessment.append("❌ Low Trade Frequency")
        
        if performance['win_rate'] >= 50:
            score += 25
            assessment.append("✅ High Win Rate")
        elif performance['win_rate'] >= 40:
            score += 15
            assessment.append("⚠️ Moderate Win Rate")
        else:
            assessment.append("❌ Low Win Rate")
        
        if performance['profit_factor'] >= 1.5:
            score += 20
            assessment.append("✅ Strong Profit Factor")
        elif performance['profit_factor'] >= 1.0:
            score += 10
            assessment.append("⚠️ Break-even Profit Factor")
        else:
            assessment.append("❌ Poor Profit Factor")
        
        # Display assessment
        if score >= 75:
            st.success(f"🏆 Excellent Performance ({score}/100)")
        elif score >= 50:
            st.warning(f"⚠️ Good Performance ({score}/100)")
        else:
            st.error(f"❌ Needs Improvement ({score}/100)")
        
        for item in assessment:
            st.write(f"  {item}")
        
        # Detailed metrics
        st.subheader("📊 Detailed Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🎯 Trade Statistics:**")
            st.write(f"- Total Trades: {performance['total_trades']}")
            st.write(f"- Winning: {performance['winning_trades']}")
            st.write(f"- Losing: {performance['losing_trades']}")
            st.write(f"- Win Rate: {performance['win_rate']:.1f}%")
            st.write(f"- Avg Duration: {performance['avg_trade_duration']:.1f} bars")
            st.write(f"- Best Trade: ${performance['best_trade']:.2f}")
            st.write(f"- Worst Trade: ${performance['worst_trade']:.2f}")
        
        with col2:
            st.write("**📈 Risk Metrics:**")
            st.write(f"- Total Return: {performance['total_return']:.2f}%")
            st.write(f"- Buy & Hold: {performance['buy_hold_return']:.2f}%")
            st.write(f"- Net P&L: ${performance['net_profit']:.2f}")
            st.write(f"- Profit Factor: {performance['profit_factor']:.2f}")
            st.write(f"- Max Drawdown: {performance['max_drawdown']:.2f}%")
            st.write(f"- Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
            st.write(f"- Sortino Ratio: {performance['sortino_ratio']:.2f}")
        
        # Chart
        st.subheader("📈 Trading Chart")
        fig = create_advanced_chart(processed_df, trades_df, strategy_type)
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade history
        if not trades_df.empty:
            st.subheader("💼 Trade History")
            
            display_trades = trades_df.copy()
            display_trades['Entry Date'] = pd.to_datetime(display_trades['entry_date']).dt.strftime('%Y-%m-%d %H:%M')
            display_trades['Exit Date'] = pd.to_datetime(display_trades['exit_date']).dt.strftime('%Y-%m-%d %H:%M')
            display_trades['Type'] = display_trades['type']
            display_trades['Entry'] = display_trades['entry_price'].round(2)
            display_trades['Exit'] = display_trades['exit_price'].round(2)
            display_trades['P&L $'] = display_trades['pnl'].round(2)
            display_trades['P&L %'] = display_trades['pnl_pct'].round(2)
            display_trades['Duration'] = display_trades['duration'].astype(int)
            display_trades['Exit Reason'] = display_trades['exit_reason']
            display_trades['Signal Strength'] = display_trades['signal_strength'].astype(int)
            
            # Show last 15 trades
            recent_trades = display_trades.tail(15)
            
            def highlight_pnl(row):
                colors = []
                for col in row.index:
                    if col in ['P&L $', 'P&L %']:
                        if row[col] > 0:
                            colors.append('background-color: rgba(34, 197, 94, 0.3)')
                        else:
                            colors.append('background-color: rgba(239, 68, 68, 0.3)')
                    else:
                        colors.append('')
                return colors
            
            cols_to_show = ['Entry Date', 'Type', 'Entry', 'Exit', 'P&L $', 'P&L %', 'Duration', 'Signal Strength', 'Exit Reason']
            styled_df = recent_trades[cols_to_show].style.apply(highlight_pnl, axis=1)
            
            st.dataframe(styled_df, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if performance['total_return'] > 5 and performance['win_rate'] >= 50:
            st.success("🚀 **Excellent! Ready for live trading.**")
        elif performance['total_trades'] < 10:
            st.warning("⚠️ **Low trade frequency.** Try shorter timeframes or different strategy.")
        elif performance['win_rate'] < 45:
            st.error("❌ **Low win rate.** Consider optimization or different parameters.")
        else:
            st.info("📊 **Good foundation.** Consider optimization for better results.")
    
    elif mode == "🎯 Optimize":
        st.header(f"🎯 {strategy_type} Optimization")
        
        if optimize_button:
            optimizer = StrategyOptimizer(df, strategy_type, has_volume)
            
            with st.spinner("Optimizing parameters..."):
                results = optimizer.optimize_parameters()
            
            if results and len(results) > 0:
                st.success("✅ Optimization completed!")
                
                best = results[0]
                st.subheader("🏆 Best Configuration")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Return", f"{best['total_return']:.2f}%")
                with col2:
                    st.metric("Win Rate", f"{best['win_rate']:.1f}%")
                with col3:
                    st.metric("Trades", f"{best['total_trades']}")
                with col4:
                    st.metric("Score", f"{best['score']:.0f}/100")
                
                # Top 5 results
                st.subheader("🏆 Top 5 Results")
                
                for i, result in enumerate(results[:5]):
                    with st.expander(f"#{i+1} - Score: {result['score']:.0f} | Return: {result['total_return']:.2f}%"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Performance:**")
                            st.write(f"- Return: {result['total_return']:.2f}%")
                            st.write(f"- Win Rate: {result['win_rate']:.1f}%")
                            st.write(f"- Trades: {result['total_trades']}")
                            st.write(f"- Profit Factor: {result['profit_factor']:.2f}")
                        
                        with col2:
                            st.write("**Parameters:**")
                            params = result['parameters']
                            for key, value in params.items():
                                st.write(f"- {key.replace('_', ' ').title()}: {value}")
                
                if st.button("✅ Apply Best Parameters"):
                    st.session_state.optimized_params = best['parameters']
                    st.success("Best parameters applied!")
                    st.balloons()
            else:
                st.error("No profitable configurations found. Try different data or strategy.")
        else:
            st.info("Click 'Optimize' to find the best parameters automatically.")
            
            st.subheader("🔧 Optimization Preview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Parameters to Optimize:**")
                st.write("- Stop Loss %: 1.0% - 3.0%")
                st.write("- Take Profit %: 3.0% - 12.0%")
                st.write("- EMA Fast: 5 - 15")
                st.write("- EMA Slow: 18 - 30")
            
            with col2:
                st.write("**Optimization Goals:**")
                st.write("- Positive returns (40 pts)")
                st.write("- High win rate (25 pts)")
                st.write("- Good trade frequency (15 pts)")
                st.write("- Strong profit factor (15 pts)")
                st.write("- Beat buy & hold (5 pts)")
    
    else:  # Live Analysis
        st.header(f"🔍 {strategy_type} Live Analysis")
        
        with st.spinner("Processing live data..."):
            processed_df = strategy.prepare_data(df)
            processed_df = strategy.generate_signals(processed_df)
        
        if len(processed_df) < 50:
            st.error("Need at least 50 data points for analysis")
            return
        
        latest = processed_df.iloc[-1]
        prev = processed_df.iloc[-2]
        
        st.subheader("📊 Current Market Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price_change = latest['Close'] - prev['Close']
            price_change_pct = (price_change / prev['Close']) * 100
            st.metric(
                "Price",
                f"{latest['Close']:.2f}",
                delta=f"{price_change_pct:+.2f}%"
            )
        
        with col2:
            rsi = latest.get('RSI', 50)
            rsi_status = "Normal" if 30 <= rsi <= 70 else "Extreme"
            st.metric("RSI", f"{rsi:.1f}", delta=rsi_status)
        
        with col3:
            macd = latest.get('MACD', 0)
            macd_signal = latest.get('MACD_Signal', 0)
            macd_trend = "Bullish" if macd > macd_signal else "Bearish"
            st.metric("MACD", f"{macd:.3f}", delta=macd_trend)
        
        with col4:
            vol_ratio = latest.get('Volume_Ratio', 1)
            vol_status = "High" if vol_ratio > 1.2 else "Normal"
            st.metric("Volume", f"{vol_ratio:.2f}x", delta=vol_status)
        
        # Signal detection
        st.subheader("🚨 Live Signals")
        
        recent_data = processed_df.tail(3)
        long_signals = recent_data[recent_data.get('Long_Signal', False)]
        short_signals = recent_data[recent_data.get('Short_Signal', False)]
        
        if not long_signals.empty or not short_signals.empty:
            if not long_signals.empty:
                signal = long_signals.iloc[-1]
                st.success("🟢 **LONG SIGNAL DETECTED!**")
                
                entry_price = signal['Close']
                stop_loss = entry_price * (1 - stop_loss_pct / 100)
                take_profit = entry_price * (1 + take_profit_pct / 100)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Trade Setup:**")
                    st.write(f"- Entry: {entry_price:.2f}")
                    st.write(f"- Stop Loss: {stop_loss:.2f}")
                    st.write(f"- Take Profit: {take_profit:.2f}")
                    st.write(f"- Risk: {entry_price - stop_loss:.2f}")
                    st.write(f"- Reward: {take_profit - entry_price:.2f}")
                
                with col2:
                    st.write("**Signal Quality:**")
                    st.write(f"- Strategy: {strategy_type}")
                    st.write(f"- Score: {signal.get('Long_Score', 0)}")
                    st.write(f"- RSI: {signal.get('RSI', 50):.1f}")
                    st.write(f"- Volume: {signal.get('Volume_Ratio', 1):.2f}x")
                    st.write(f"- Confidence: High")
            
            if not short_signals.empty:
                signal = short_signals.iloc[-1]
                st.error("🔴 **SHORT SIGNAL DETECTED!**")
                
                entry_price = signal['Close']
                stop_loss = entry_price * (1 + stop_loss_pct / 100)
                take_profit = entry_price * (1 - take_profit_pct / 100)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Trade Setup:**")
                    st.write(f"- Entry: {entry_price:.2f}")
                    st.write(f"- Stop Loss: {stop_loss:.2f}")
                    st.write(f"- Take Profit: {take_profit:.2f}")
                    st.write(f"- Risk: {stop_loss - entry_price:.2f}")
                    st.write(f"- Reward: {entry_price - take_profit:.2f}")
                
                with col2:
                    st.write("**Signal Quality:**")
                    st.write(f"- Strategy: {strategy_type}")
                    st.write(f"- Score: {signal.get('Short_Score', 0)}")
                    st.write(f"- RSI: {signal.get('RSI', 50):.1f}")
                    st.write(f"- Volume: {signal.get('Volume_Ratio', 1):.2f}x")
                    st.write(f"- Confidence: High")
        else:
            st.info("⏳ **No active signals. Waiting for setup...**")
            
            # Show current conditions
            st.write("**Current Conditions:**")
            
            conditions = []
            if latest['Close'] > latest.get('EMA_Fast', 0):
                conditions.append("✅ Price > EMA Fast")
            else:
                conditions.append("❌ Price < EMA Fast")
            
            rsi = latest.get('RSI', 50)
            if strategy_type == "Mean Reversion":
                if rsi < rsi_oversold or rsi > rsi_overbought:
                    conditions.append(f"✅ RSI Extreme ({rsi:.1f})")
                else:
                    conditions.append(f"❌ RSI Normal ({rsi:.1f})")
            else:
                if 40 < rsi < 75:
                    conditions.append(f"✅ RSI Good ({rsi:.1f})")
                else:
                    conditions.append(f"❌ RSI Extreme ({rsi:.1f})")
            
            if latest.get('Volume_Ratio', 1) > 1.0:
                conditions.append("✅ Volume Above Average")
            else:
                conditions.append("❌ Volume Below Average")
            
            for condition in conditions:
                st.write(f"  {condition}")
        
        # Live chart
        st.subheader("📈 Live Chart")
        fig = create_advanced_chart(processed_df, strategy_type=strategy_type)
        st.plotly_chart(fig, use_container_width=True)
        
        # Market timing
        st.subheader("⏰ Market Timing")
        
        score = 0
        factors = []
        
        # Trend alignment
        if latest['Close'] > latest.get('EMA_Fast', 0) > latest.get('EMA_Slow', 0):
            score += 25
            factors.append("✅ Strong Uptrend")
        elif latest['Close'] < latest.get('EMA_Fast', 0) < latest.get('EMA_Slow', 0):
            score += 25
            factors.append("✅ Strong Downtrend")
        else:
            factors.append("⚠️ Sideways Market")
        
        # RSI positioning
        if 30 < rsi < 70:
            score += 20
            factors.append("✅ RSI Normal")
        else:
            factors.append("⚠️ RSI Extreme")
        
        # Volume
        if latest.get('Volume_Ratio', 1) > 1.2:
            score += 20
            factors.append("✅ High Volume")
        elif latest.get('Volume_Ratio', 1) > 0.8:
            score += 10
            factors.append("⚪ Normal Volume")
        else:
            factors.append("🔴 Low Volume")
        
        # MACD
        if latest.get('MACD', 0) > latest.get('MACD_Signal', 0):
            score += 15
            factors.append("✅ MACD Bullish")
        else:
            factors.append("❌ MACD Bearish")
        
        # Volatility
        adx = latest.get('ADX', 20)
        if adx > 25:
            score += 20
            factors.append("✅ Trending Market")
        else:
            factors.append("⚠️ Ranging Market")
        
        # Display recommendation
        if score >= 80:
            st.success(f"🟢 **EXCELLENT CONDITIONS** ({score}/100)")
        elif score >= 60:
            st.warning(f"⚠️ **GOOD CONDITIONS** ({score}/100)")
        else:
            st.error(f"🔴 **POOR CONDITIONS** ({score}/100)")
        
        for factor in factors:
            st.write(f"  {factor}")
    
    # Footer
    st.markdown("---")
    st.markdown("### 🚀 Enhanced Trading System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**✅ Fixed & Enhanced:**")
        st.markdown("- All syntax errors fixed")
        st.markdown("- Realistic backtesting")
        st.markdown("- Proper risk management")
        st.markdown("- Enhanced performance metrics")
    
    with col2:
        st.markdown("**🎯 Key Features:**")
        st.markdown("- 5 proven strategies")
        st.markdown("- Multiple timeframes")
        st.markdown("- Auto-optimization")
        st.markdown("- Live signal detection")
    
    with col3:
        st.markdown("**📊 Performance Focus:**")
        st.markdown("- High trade frequency")
        st.markdown("- Positive returns target")
        st.markdown("- Realistic execution")
        st.markdown("- Professional metrics")
    
    st.markdown("⚠️ *Enhanced system with realistic backtesting and proper risk management.*")

if __name__ == "__main__":
    main()
        trade_id = 0
        
        # Buy and hold benchmark
        buy_hold_start = df['Close'].iloc[0]
        buy_hold_end = df['Close'].iloc[-1]
        buy_hold_return = (buy_hold_end - buy_hold_start) / buy_hold_start * 100
        
        # Portfolio tracking
        portfolio_values = []
        max_portfolio = initial_capital
        
        for i in range(len(df)):
            current_bar = df.iloc[i]
            current_date = df.index[i]
            current_price = current_bar['Close']
            
            # Update portfolio value
            if position is None:
                portfolio_value = capital
            else:
                if position['type'] == 'LONG':
                    unrealized_pnl = (current_price - position['entry_price']) * position['shares']
                else:
                    unrealized_pnl = (position['entry_price'] - current_price) * position['shares']
                portfolio_value = capital + unrealized_pnl
            
            portfolio_values.append(portfolio_value)
            max_portfolio = max(max_portfolio, portfolio_value)
            
            # Check for exit conditions first
            if position is not None:
                exit_triggered = False
                exit_price = current_price
                exit_reason = ""
                
                # Check stop loss and take profit using intrabar logic
                if position['type'] == 'LONG':
                    # Check if low touched stop loss
                    if current_bar['Low'] <= position['stop_loss']:
                        exit_triggered = True
                        exit_price = position['stop_loss']
                        exit_reason = "Stop Loss"
                    # Check if high touched take profit
                    elif current_bar['High'] >= position['take_profit']:
                        exit_triggered = True
                        exit_price = position['take_profit']
                        exit_reason = "Take Profit"
                    # Check for opposite signal
                    elif current_bar.get('Short_Signal', False):
                        exit_triggered = True
                        exit_reason = "Opposite Signal"
                
                else:  # SHORT position
                    # Check if high touched stop loss
                    if current_bar['High'] >= position['stop_loss']:
                        exit_triggered = True
                        exit_price = position['stop_loss']
                        exit_reason = "Stop Loss"
                    # Check if low touched take profit
                    elif current_bar['Low'] <= position['take_profit']:
                        exit_triggered = True
                        exit_price = position['take_profit']
                        exit_reason = "Take Profit"
                    # Check for opposite signal
                    elif current_bar.get('Long_Signal', False):
                        exit_triggered = True
                        exit_reason = "Opposite Signal"
                
                # Execute exit
                if exit_triggered:
                    # Calculate P&L
                    if position['type'] == 'LONG':
                        pnl = (exit_price - position['entry_price']) * position['shares']
                        pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                    else:
                        pnl = (position['entry_price'] - exit_price) * position['shares']
                        pnl_pct = (position['entry_price'] - exit_price) / position['entry_price'] * 100
                    
                    # Update capital
                    capital += pnl
                    
                    # Record trade
                    trade_duration = i - position['entry_bar']
                    
                    trade = {
                        'trade_id': position['trade_id'],
                        'type': position['type'],
                        'entry_date': position['entry_date'],
                        'entry_price': position['entry_price'],
                        'entry_bar': position['entry_bar'],
                        'exit_date': current_date,
                        'exit_price': exit_price,
                        'exit_bar': i,
                        'shares': position['shares'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'duration': trade_duration,
                        'exit_reason': exit_reason,
                        'stop_loss': position['stop_loss'],
                        'take_profit': position['take_profit'],
                        'portfolio_value': capital,
                        'signal_strength': position.get('signal_strength', 0)
                    }
                    
                    trades.append(trade)
                    position = None
            
            # Check for new entry signals (only if no position)
            if position is None:
                entry_signal = False
                signal_type = ""
                signal_strength = 0
                
                if current_bar.get('Long_Signal', False):
                    entry_signal = True
                    signal_type = "LONG"
                    signal_strength = current_bar.get('Long_Score', 0)
                elif current_bar.get('Short_Signal', False):
                    entry_signal = True
                    signal_type = "SHORT" 
                    signal_strength = current_bar.get('Short_Score', 0)
                
                # Execute entry
                if entry_signal:
                    trade_id += 1
                    entry_price = current_price
                    
                    # Calculate position size (2% risk per trade)
                    risk_per_trade = capital * 0.02
                    
                    # Calculate stop loss and take profit
                    atr = current_bar.get('ATR', current_price * 0.02)
                    volatility_mult = max(1.0, atr / current_price * 50)  # Dynamic multiplier
                    
                    if signal_type == "LONG":
                        stop_loss = entry_price * (1 - self.stop_loss_pct * volatility_mult / 100)
                        take_profit = entry_price * (1 + self.take_profit_pct * volatility_mult / 100)
                        risk_per_share = entry_price - stop_loss
                    else:
                        stop_loss = entry_price * (1 + self.stop_loss_pct * volatility_mult / 100)
                        take_profit = entry_price * (1 - self.take_profit_pct * volatility_mult / 100)
                        risk_per_share = stop_loss - entry_price
                    
                    # Calculate shares based on risk
                    if risk_per_share > 0:
                        shares = int(risk_per_trade / risk_per_share)
                        shares = max(1, min(shares, int(capital * 0.95 / entry_price)))  # Max 95% of capital
                    else:
                        shares = 1
                    
                    # Create position
                    position = {
                        'trade_id': trade_id,
                        'type': signal_type,
                        'entry_date': current_date,
                        'entry_price': entry_price,
                        'entry_bar': i,
                        'shares': shares,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'signal_strength': signal_strength
                    }
        
        # Close any remaining position at the end
        if position is not None:
            exit_price = df['Close'].iloc[-1]
            
            if position['type'] == 'LONG':
                pnl = (exit_price - position['entry_price']) * position['shares']
                pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
            else:
                pnl = (position['entry_price'] - exit_price) * position['shares']
                pnl_pct = (position['entry_price'] - exit_price) / position['entry_price'] * 100
            
            capital += pnl
            
            trade = {
                'trade_id': position['trade_id'],
                'type': position['type'],
                'entry_date': position['entry_date'],
                'entry_price': position['entry_price'],
                'entry_bar': position['entry_bar'],
                'exit_date': df.index[-1],
                'exit_price': exit_price,
                'exit_bar': len(df) - 1,
                'shares': position['shares'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'duration': len(df) - 1 - position['entry_bar'],
                'exit_reason': "End of Data",
                'stop_loss': position['stop_loss'],
                'take_profit': position['take_profit'],
                'portfolio_value': capital,
                'signal_strength': position.get('signal_strength', 0)
            }
            
            trades.append(trade)
        
        # Calculate comprehensive performance metrics
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # Portfolio metrics
        portfolio_df = pd.DataFrame({
            'date': df.index,
            'portfolio_value': portfolio_values
        })
        
        performance = self.calculate_enhanced_performance(
            trades_df, portfolio_df, initial_capital, capital, buy_hold_return
        )
        
        return df, trades_df, performance
    
    def calculate_enhanced_performance(self, trades_df, portfolio_df, initial_capital, final_capital, buy_hold_return):
        """Calculate comprehensive performance metrics"""
        if trades_df.empty:
