import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Stock Screener & Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AdvancedStockScreener:
    def __init__(self):
        # Nifty 50 stock symbols
        self.nifty50_stocks = {
            'RELIANCE': 'RELIANCE.NS', 'TCS': 'TCS.NS', 'HDFCBANK': 'HDFCBANK.NS',
            'INFY': 'INFY.NS', 'HINDUNILVR': 'HINDUNILVR.NS', 'ICICIBANK': 'ICICIBANK.NS',
            'SBIN': 'SBIN.NS', 'BHARTIARTL': 'BHARTIARTL.NS', 'ITC': 'ITC.NS',
            'KOTAKBANK': 'KOTAKBANK.NS', 'LT': 'LT.NS', 'AXISBANK': 'AXISBANK.NS',
            'ASIANPAINT': 'ASIANPAINT.NS', 'MARUTI': 'MARUTI.NS', 'SUNPHARMA': 'SUNPHARMA.NS',
            'TITAN': 'TITAN.NS', 'ULTRACEMCO': 'ULTRACEMCO.NS', 'NESTLEIND': 'NESTLEIND.NS',
            'BAJFINANCE': 'BAJFINANCE.NS', 'POWERGRID': 'POWERGRID.NS', 'M&M': 'M&M.NS',
            'NTPC': 'NTPC.NS', 'TECHM': 'TECHM.NS', 'HCLTECH': 'HCLTECH.NS',
            'WIPRO': 'WIPRO.NS', 'TATAMOTORS': 'TATAMOTORS.NS', 'COALINDIA': 'COALINDIA.NS',
            'JSWSTEEL': 'JSWSTEEL.NS', 'GRASIM': 'GRASIM.NS', 'HINDALCO': 'HINDALCO.NS',
            'DRREDDY': 'DRREDDY.NS', 'TATASTEEL': 'TATASTEEL.NS', 'ADANIPORTS': 'ADANIPORTS.NS',
            'ONGC': 'ONGC.NS', 'EICHERMOT': 'EICHERMOT.NS', 'SBILIFE': 'SBILIFE.NS',
            'BPCL': 'BPCL.NS', 'BAJAJFINSV': 'BAJAJFINSV.NS', 'DIVISLAB': 'DIVISLAB.NS',
            'CIPLA': 'CIPLA.NS', 'BRITANNIA': 'BRITANNIA.NS', 'HEROMOTOCO': 'HEROMOTOCO.NS',
            'APOLLOHOSP': 'APOLLOHOSP.NS', 'UPL': 'UPL.NS', 'INDUSINDBK': 'INDUSINDBK.NS',
            'TATACONSUM': 'TATACONSUM.NS', 'BAJAJ-AUTO': 'BAJAJ-AUTO.NS',
            'HDFCLIFE': 'HDFCLIFE.NS', 'ADANIENT': 'ADANIENT.NS'
        }
    
    @st.cache_data
    def fetch_stock_data(_self, stock_name, period="10y"):
        """Fetch stock data with caching"""
        try:
            if stock_name.upper() in _self.nifty50_stocks:
                symbol = _self.nifty50_stocks[stock_name.upper()]
            else:
                symbol = stock_name + '.NS'
            
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if data.empty:
                st.error(f"No data found for {stock_name}")
                return None
            return data
        except Exception as e:
            st.error(f"Error fetching data for {stock_name}: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate all technical indicators"""
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Price momentum
        df['Price_Change_5'] = df['Close'].pct_change(5)
        df['Price_Change_10'] = df['Close'].pct_change(10)
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        return df
    
    def generate_trading_signals(self, df):
        """Generate enhanced trading signals"""
        df = self.calculate_technical_indicators(df)
        
        # Initialize signals
        df['Signal'] = 0
        df['Signal_Strength'] = 0
        df['Entry_Logic'] = ""
        df['Exit_Logic'] = ""
        
        for i in range(50, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Entry conditions with scoring
            entry_conditions = {
                'Price_Above_SMA50': current['Close'] > current['SMA_50'],
                'SMA50_Above_SMA200': current['SMA_50'] > current['SMA_200'],
                'RSI_Optimal': 40 <= current['RSI'] <= 65,
                'High_Volume': current['Volume_Ratio'] > 1.2,
                'Positive_Momentum': current['Price_Change_5'] > 0.01,
                'Above_BB_Lower': current['Close'] > current['BB_Lower'],
                'MACD_Bullish': current['MACD'] > current['MACD_Signal']
            }
            
            # Exit conditions
            exit_conditions = {
                'Price_Below_SMA50': current['Close'] < current['SMA_50'],
                'RSI_Overbought': current['RSI'] > 75,
                'RSI_Oversold': current['RSI'] < 30,
                'Below_BB_Lower': current['Close'] < current['BB_Lower'],
                'MACD_Bearish': current['MACD'] < current['MACD_Signal']
            }
            
            # Calculate entry score
            entry_score = sum(entry_conditions.values())
            df.loc[df.index[i], 'Signal_Strength'] = entry_score
            
            # Create logic strings
            entry_logic = ", ".join([k for k, v in entry_conditions.items() if v])
            exit_logic = ", ".join([k for k, v in exit_conditions.items() if v])
            
            df.loc[df.index[i], 'Entry_Logic'] = entry_logic
            df.loc[df.index[i], 'Exit_Logic'] = exit_logic
            
            # Generate signals
            if entry_score >= 5:  # Need at least 5 out of 7 conditions
                df.loc[df.index[i], 'Signal'] = 1
            elif any(exit_conditions.values()):
                df.loc[df.index[i], 'Signal'] = -1
        
        return df
    
    def advanced_backtest(self, df, initial_capital=100000):
        """Enhanced backtesting with detailed logging"""
        capital = initial_capital
        cash = initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        stop_loss = 0
        target = 0
        trades = []
        portfolio_values = []
        max_position_size = 0.1  # 10% per trade
        
        for i in range(len(df)):
            current_price = df['Close'].iloc[i]
            current_date = df.index[i]
            current_data = df.iloc[i]
            
            # Check stop loss
            if position > 0 and current_price <= stop_loss:
                exit_value = position * current_price
                pnl = exit_value - (position * entry_price)
                pnl_points = current_price - entry_price
                pnl_percent = (pnl / (position * entry_price)) * 100
                cash += exit_value
                
                trades.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': current_date,
                    'Entry_Price': entry_price,
                    'Exit_Price': current_price,
                    'Quantity': position,
                    'PnL_Amount': pnl,
                    'PnL_Points': pnl_points,
                    'PnL_Percent': pnl_percent,
                    'Exit_Reason': 'Stop Loss',
                    'Entry_Logic': df.loc[entry_date, 'Entry_Logic'] if entry_date in df.index else '',
                    'Exit_Logic': 'Price hit stop loss level',
                    'Trade_Duration': (current_date - entry_date).days if entry_date else 0
                })
                position = 0
                entry_price = 0
                stop_loss = 0
                target = 0
                entry_date = None
            
            # Check target
            elif position > 0 and current_price >= target:
                exit_value = position * current_price
                pnl = exit_value - (position * entry_price)
                pnl_points = current_price - entry_price
                pnl_percent = (pnl / (position * entry_price)) * 100
                cash += exit_value
                
                trades.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': current_date,
                    'Entry_Price': entry_price,
                    'Exit_Price': current_price,
                    'Quantity': position,
                    'PnL_Amount': pnl,
                    'PnL_Points': pnl_points,
                    'PnL_Percent': pnl_percent,
                    'Exit_Reason': 'Target Hit',
                    'Entry_Logic': df.loc[entry_date, 'Entry_Logic'] if entry_date in df.index else '',
                    'Exit_Logic': 'Price hit target level',
                    'Trade_Duration': (current_date - entry_date).days if entry_date else 0
                })
                position = 0
                entry_price = 0
                stop_loss = 0
                target = 0
                entry_date = None
            
            # Process buy signals
            elif current_data['Signal'] == 1 and position == 0 and cash > 0:
                max_investment = cash * max_position_size
                position = int(max_investment // current_price)
                
                if position > 0:
                    investment = position * current_price
                    entry_price = current_price
                    entry_date = current_date
                    stop_loss = entry_price * 0.98  # 2% stop loss
                    target = entry_price * 1.04     # 4% target
                    cash -= investment
            
            # Process sell signals
            elif current_data['Signal'] == -1 and position > 0:
                exit_value = position * current_price
                pnl = exit_value - (position * entry_price)
                pnl_points = current_price - entry_price
                pnl_percent = (pnl / (position * entry_price)) * 100
                cash += exit_value
                
                trades.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': current_date,
                    'Entry_Price': entry_price,
                    'Exit_Price': current_price,
                    'Quantity': position,
                    'PnL_Amount': pnl,
                    'PnL_Points': pnl_points,
                    'PnL_Percent': pnl_percent,
                    'Exit_Reason': 'Signal Exit',
                    'Entry_Logic': df.loc[entry_date, 'Entry_Logic'] if entry_date in df.index else '',
                    'Exit_Logic': current_data['Exit_Logic'],
                    'Trade_Duration': (current_date - entry_date).days if entry_date else 0
                })
                position = 0
                entry_price = 0
                stop_loss = 0
                target = 0
                entry_date = None
            
            # Calculate portfolio value
            portfolio_value = cash + (position * current_price if position > 0 else 0)
            portfolio_values.append(portfolio_value)
        
        # Close remaining position
        if position > 0:
            current_price = df['Close'].iloc[-1]
            current_date = df.index[-1]
            exit_value = position * current_price
            pnl = exit_value - (position * entry_price)
            pnl_points = current_price - entry_price
            pnl_percent = (pnl / (position * entry_price)) * 100
            cash += exit_value
            
            trades.append({
                'Entry_Date': entry_date,
                'Exit_Date': current_date,
                'Entry_Price': entry_price,
                'Exit_Price': current_price,
                'Quantity': position,
                'PnL_Amount': pnl,
                'PnL_Points': pnl_points,
                'PnL_Percent': pnl_percent,
                'Exit_Reason': 'End of Period',
                'Entry_Logic': df.loc[entry_date, 'Entry_Logic'] if entry_date in df.index else '',
                'Exit_Logic': 'Backtest period ended',
                'Trade_Duration': (current_date - entry_date).days if entry_date else 0
            })
        
        return self.calculate_performance_metrics(trades, cash, initial_capital, portfolio_values)
    
    def calculate_performance_metrics(self, trades, final_capital, initial_capital, portfolio_values):
        """Calculate comprehensive performance metrics"""
        trades_df = pd.DataFrame(trades)
        
        if trades_df.empty:
            return {
                'trades_df': trades_df,
                'total_trades': 0,
                'profitable_trades': 0,
                'loss_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'max_profit': 0,
                'max_loss': 0,
                'total_return': 0,
                'final_capital': initial_capital,
                'profit_factor': 0,
                'avg_trade_duration': 0,
                'portfolio_values': portfolio_values
            }
        
        # Basic metrics
        total_trades = len(trades_df)
        profitable_trades = len(trades_df[trades_df['PnL_Amount'] > 0])
        loss_trades = total_trades - profitable_trades
        win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = trades_df['PnL_Amount'].sum()
        avg_profit = trades_df[trades_df['PnL_Amount'] > 0]['PnL_Amount'].mean() if profitable_trades > 0 else 0
        avg_loss = trades_df[trades_df['PnL_Amount'] < 0]['PnL_Amount'].mean() if loss_trades > 0 else 0
        max_profit = trades_df['PnL_Amount'].max()
        max_loss = trades_df['PnL_Amount'].min()
        
        # Returns
        total_return = ((final_capital - initial_capital) / initial_capital) * 100
        
        # Advanced metrics
        gross_profit = trades_df[trades_df['PnL_Amount'] > 0]['PnL_Amount'].sum()
        gross_loss = abs(trades_df[trades_df['PnL_Amount'] < 0]['PnL_Amount'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade duration
        avg_trade_duration = trades_df['Trade_Duration'].mean()
        
        return {
            'trades_df': trades_df,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'loss_trades': loss_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'total_return': total_return,
            'final_capital': final_capital,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_trade_duration,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'portfolio_values': portfolio_values
        }
    
    def get_live_recommendation(self, stock_name):
        """Get live trading recommendation"""
        data = self.fetch_stock_data(stock_name, period="1y")
        if data is None:
            return None
        
        # Apply strategy
        strategy_data = self.generate_trading_signals(data)
        
        # Get latest data
        latest = strategy_data.iloc[-1]
        current_price = latest['Close']
        
        # Calculate recommendation
        if latest['Signal'] == 1:
            recommendation = "🟢 BUY"
            stop_loss = current_price * 0.98
            target = current_price * 1.04
        elif latest['Signal'] == -1:
            recommendation = "🔴 SELL"
            stop_loss = current_price * 1.02
            target = current_price * 0.96
        else:
            recommendation = "🟡 HOLD"
            stop_loss = current_price * 0.98
            target = current_price * 1.04
        
        # Calculate entry score
        entry_score = latest['Signal_Strength']
        
        return {
            'stock': stock_name.upper(),
            'current_price': current_price,
            'recommendation': recommendation,
            'entry_score': f"{entry_score}/7",
            'stop_loss': stop_loss,
            'target': target,
            'rsi': latest['RSI'],
            'volume_ratio': latest['Volume_Ratio'],
            'entry_logic': latest['Entry_Logic'],
            'exit_logic': latest['Exit_Logic'],
            'sma_50': latest['SMA_50'],
            'sma_200': latest['SMA_200']
        }

def main():
    st.title("📈 Advanced Stock Screener & Trading System")
    st.markdown("---")
    
    # Initialize screener
    screener = AdvancedStockScreener()
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Main navigation
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["📊 Complete Stock Analysis", "⚡ Live Trading Signals", "🔍 Portfolio Scanner"]
    )
    
    if page == "📊 Complete Stock Analysis":
        st.header("📊 Complete Stock Analysis (10-Year Data)")
        
        # Stock selection
        col1, col2 = st.columns([2, 1])
        with col1:
            stock_name = st.selectbox(
                "Select Nifty 50 Stock:",
                options=list(screener.nifty50_stocks.keys()),
                index=0
            )
        
        with col2:
            st.markdown("### Quick Stats")
            if st.button("📈 Analyze Stock", type="primary"):
                with st.spinner(f"Analyzing {stock_name}... Please wait"):
                    # Fetch data
                    data = screener.fetch_stock_data(stock_name)
                    if data is not None:
                        # Apply strategy
                        strategy_data = screener.generate_trading_signals(data)
                        
                        # Backtest
                        results = screener.advanced_backtest(strategy_data)
                        
                        # Display results
                        st.success(f"✅ Analysis Complete for {stock_name}")
                        
                        # Performance Overview
                        st.subheader("🎯 Performance Overview")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Trades", results['total_trades'])
                        with col2:
                            st.metric("Win Rate", f"{results['win_rate']:.1f}%")
                        with col3:
                            st.metric("Total Return", f"{results['total_return']:.2f}%")
                        with col4:
                            st.metric("Final Capital", f"₹{results['final_capital']:,.0f}")
                        
                        # Detailed Metrics
                        st.subheader("📈 Detailed Performance Metrics")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📊 Trade Statistics:**")
                            st.write(f"• Total Trades: {results['total_trades']}")
                            st.write(f"• Profitable Trades: {results['profitable_trades']}")
                            st.write(f"• Loss Trades: {results['loss_trades']}")
                            st.write(f"• Win Rate: {results['win_rate']:.2f}%")
                            st.write(f"• Average Trade Duration: {results['avg_trade_duration']:.1f} days")
                        
                        with col2:
                            st.markdown("**💰 Financial Metrics:**")
                            st.write(f"• Total P&L: ₹{results['total_pnl']:,.2f}")
                            st.write(f"• Average Profit: ₹{results['avg_profit']:,.2f}")
                            st.write(f"• Average Loss: ₹{results['avg_loss']:,.2f}")
                            st.write(f"• Max Profit: ₹{results['max_profit']:,.2f}")
                            st.write(f"• Max Loss: ₹{results['max_loss']:,.2f}")
                            st.write(f"• Profit Factor: {results['profit_factor']:.2f}")
                        
                        # Trade Log
                        if not results['trades_df'].empty:
                            st.subheader("📋 Detailed Trade Log")
                            
                            # Format the dataframe for display
                            display_df = results['trades_df'].copy()
                            display_df['Entry_Date'] = display_df['Entry_Date'].dt.strftime('%Y-%m-%d')
                            display_df['Exit_Date'] = display_df['Exit_Date'].dt.strftime('%Y-%m-%d')
                            display_df['Entry_Price'] = display_df['Entry_Price'].round(2)
                            display_df['Exit_Price'] = display_df['Exit_Price'].round(2)
                            display_df['PnL_Amount'] = display_df['PnL_Amount'].round(2)
                            display_df['PnL_Points'] = display_df['PnL_Points'].round(2)
                            display_df['PnL_Percent'] = display_df['PnL_Percent'].round(2)
                            
                            # Color coding for P&L
                            def color_pnl(val):
                                color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                                return f'color: {color}'
                            
                            styled_df = display_df.style.applymap(color_pnl, subset=['PnL_Amount', 'PnL_Points', 'PnL_Percent'])
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # Download button for trade log
                            csv = display_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Trade Log",
                                data=csv,
                                file_name=f"{stock_name}_trade_log.csv",
                                mime="text/csv"
                            )
                        
                        # Charts
                        st.subheader("📈 Price Charts with Signals")
                        
                        # Create price chart with signals
                        fig = make_subplots(
                            rows=3, cols=1,
                            subplot_titles=['Price & Moving Averages', 'RSI', 'Volume'],
                            vertical_spacing=0.08,
                            row_weights=[0.6, 0.2, 0.2]
                        )
                        
                        # Price and SMAs
                        fig.add_trace(go.Scatter(x=strategy_data.index, y=strategy_data['Close'], name='Close', line=dict(color='blue')), row=1, col=1)
                        fig.add_trace(go.Scatter(x=strategy_data.index, y=strategy_data['SMA_50'], name='SMA 50', line=dict(color='orange')), row=1, col=1)
                        fig.add_trace(go.Scatter(x=strategy_data.index, y=strategy_data['SMA_200'], name='SMA 200', line=dict(color='red')), row=1, col=1)
                        
                        # Buy/Sell signals
                        buy_signals = strategy_data[strategy_data['Signal'] == 1]
                        sell_signals = strategy_data[strategy_data['Signal'] == -1]
                        
                        fig.add_trace(go.Scatter(
                            x=buy_signals.index, y=buy_signals['Close'],
                            mode='markers', name='Buy Signal',
                            marker=dict(color='green', size=10, symbol='triangle-up')
                        ), row=1, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=sell_signals.index, y=sell_signals['Close'],
                            mode='markers', name='Sell Signal',
                            marker=dict(color='red', size=10, symbol='triangle-down')
                        ), row=1, col=1)
                        
                        # RSI
                        fig.add_trace(go.Scatter(x=strategy_data.index, y=strategy_data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                        
                        # Volume
                        fig.add_trace(go.Bar(x=strategy_data.index, y=strategy_data['Volume'], name='Volume', marker_color='lightblue'), row=3, col=1)
                        
                        fig.update_layout(height=800, title_text=f"{stock_name} - Technical Analysis")
                        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "⚡ Live Trading Signals":
        st.header("⚡ Live Trading Signals")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            stock_input = st.text_input("Enter Stock Symbol (e.g., RELIANCE, TCS):", "RELIANCE")
        
        with col2:
            if st.button("🎯 Get Live Signal", type="primary"):
                if stock_input:
                    with st.spinner("Fetching live data..."):
                        recommendation = screener.get_live_recommendation(stock_input)
                        
                        if recommendation:
                            st.success("✅ Live Signal Generated!")
                            
                            # Display recommendation
                            st.subheader(f"📊 {recommendation['stock']} - Live Analysis")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current Price", f"₹{recommendation['current_price']:.2f}")
                            with col2:
                                st.metric("Signal", recommendation['recommendation'])
                            with col3:
                                st.metric("Entry Score", recommendation['entry_score'])
                            
                            # Trading levels
                            st.subheader("🎯 Trading Levels")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Stop Loss", f"₹{recommendation['stop_loss']:.2f}")
                            with col2:
                                st.metric("Target", f"₹{recommendation['target']:.2f}")
                            with col3:
                                st.metric("RSI", f"{recommendation['rsi']:.1f}")
                            with col4:
                                st.metric("Volume Ratio", f"{recommendation['volume_ratio']:.2f}")
                            
                            # Logic explanation
                            st.subheader("🧠 Strategy Logic")
                            st.info(f"**Entry Logic:** {recommendation['entry_logic']}")
                            if recommendation['exit_logic']:
                                st.warning(f"**Exit Logic:** {recommendation['exit_logic']}")
                            
                            # Technical Summary
                            st.subheader("📈 Technical Summary")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Moving Averages:**")
                                st.write(f"• SMA 50: ₹{recommendation['sma_50']:.2f}")
                                st.write(f"• SMA 200: ₹{recommendation['sma_200']:.2f}")
                                trend = "🟢 Uptrend" if recommendation['sma_50'] > recommendation['sma_200'] else "🔴 Downtrend"
                                st.write(f"• Trend: {trend}")
                            
                            with col2:
                                st.write("**Key Indicators:**")
                                rsi_status = "🟢 Normal" if 30 <= recommendation['rsi'] <= 70 else "🟡 Extreme"
                                st.write(f"• RSI Status: {rsi_status}")
                                vol_status = "🟢 High" if recommendation['volume_ratio'] > 1.2 else "🟡 Normal"
                                st.write(f"• Volume: {vol_status}")
                        else:
                            st.error("❌ Could not fetch data for this stock")
                else:
                    st.warning("⚠️ Please enter a stock symbol")
        
        # Quick scanner for all Nifty 50
        st.markdown("---")
        st.subheader("🔍 Quick Nifty 50 Scanner")
        
        if st.button("🚀 Scan All Nifty 50 Stocks", type="secondary"):
            progress_bar = st.progress(0)
            results_container = st.empty()
            
            scanner_results = []
            total_stocks = len(screener.nifty50_stocks)
            
            for i, stock in enumerate(list(screener.nifty50_stocks.keys())[:10]):  # Limit to first 10 for demo
                progress_bar.progress((i + 1) / 10)
                
                with st.spinner(f"Scanning {stock}..."):
                    recommendation = screener.get_live_recommendation(stock)
                    if recommendation:
                        scanner_results.append(recommendation)
            
            progress_bar.empty()
            
            # Display scanner results
            if scanner_results:
                st.success(f"✅ Scanned {len(scanner_results)} stocks successfully!")
                
                # Create DataFrame for display
                scanner_df = pd.DataFrame([
                    {
                        'Stock': r['stock'],
                        'Price': f"₹{r['current_price']:.2f}",
                        'Signal': r['recommendation'],
                        'Score': r['entry_score'],
                        'Stop Loss': f"₹{r['stop_loss']:.2f}",
                        'Target': f"₹{r['target']:.2f}",
                        'RSI': f"{r['rsi']:.1f}",
                        'Volume': f"{r['volume_ratio']:.2f}x"
                    }
                    for r in scanner_results
                ])
                
                # Color code signals
                def highlight_signals(s):
                    if 'BUY' in str(s):
                        return 'background-color: lightgreen'
                    elif 'SELL' in str(s):
                        return 'background-color: lightcoral'
                    else:
                        return 'background-color: lightyellow'
                
                styled_scanner = scanner_df.style.applymap(highlight_signals, subset=['Signal'])
                st.dataframe(styled_scanner, use_container_width=True)
                
                # Download scanner results
                csv = scanner_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Scanner Results",
                    data=csv,
                    file_name=f"nifty50_scanner_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
    
    elif page == "🔍 Portfolio Scanner":
        st.header("🔍 Portfolio Performance Scanner")
        
        # Portfolio input
        st.subheader("📝 Build Your Portfolio")
        
        # Multi-select for stocks
        selected_stocks = st.multiselect(
            "Select stocks for portfolio analysis:",
            options=list(screener.nifty50_stocks.keys()),
            default=['RELIANCE', 'TCS', 'INFY']
        )
        
        if selected_stocks:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                initial_capital = st.number_input("Initial Capital (₹):", value=100000, step=10000)
            
            with col2:
                analysis_period = st.selectbox("Analysis Period:", ["1y", "2y", "3y", "5y"], index=1)
            
            if st.button("📊 Analyze Portfolio", type="primary"):
                st.subheader("🎯 Portfolio Analysis Results")
                
                portfolio_results = []
                progress_bar = st.progress(0)
                
                for i, stock in enumerate(selected_stocks):
                    progress_bar.progress((i + 1) / len(selected_stocks))
                    
                    with st.spinner(f"Analyzing {stock}..."):
                        data = screener.fetch_stock_data(stock, period=analysis_period)
                        if data is not None:
                            strategy_data = screener.generate_trading_signals(data)
                            results = screener.advanced_backtest(strategy_data, initial_capital)
                            
                            portfolio_results.append({
                                'Stock': stock,
                                'Total_Trades': results['total_trades'],
                                'Win_Rate': results['win_rate'],
                                'Total_Return': results['total_return'],
                                'Final_Capital': results['final_capital'],
                                'Max_Profit': results['max_profit'],
                                'Max_Loss': results['max_loss'],
                                'Profit_Factor': results['profit_factor']
                            })
                
                progress_bar.empty()
                
                if portfolio_results:
                    # Portfolio summary
                    portfolio_df = pd.DataFrame(portfolio_results)
                    
                    # Summary metrics
                    st.subheader("📈 Portfolio Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_return = portfolio_df['Total_Return'].mean()
                        st.metric("Average Return", f"{avg_return:.2f}%")
                    
                    with col2:
                        avg_win_rate = portfolio_df['Win_Rate'].mean()
                        st.metric("Average Win Rate", f"{avg_win_rate:.1f}%")
                    
                    with col3:
                        total_trades = portfolio_df['Total_Trades'].sum()
                        st.metric("Total Trades", total_trades)
                    
                    with col4:
                        best_stock = portfolio_df.loc[portfolio_df['Total_Return'].idxmax(), 'Stock']
                        st.metric("Best Performer", best_stock)
                    
                    # Detailed table
                    st.subheader("📊 Detailed Portfolio Results")
                    
                    # Format for display
                    display_portfolio = portfolio_df.copy()
                    display_portfolio['Win_Rate'] = display_portfolio['Win_Rate'].round(1).astype(str) + '%'
                    display_portfolio['Total_Return'] = display_portfolio['Total_Return'].round(2).astype(str) + '%'
                    display_portfolio['Final_Capital'] = '₹' + display_portfolio['Final_Capital'].round(0).astype(str)
                    display_portfolio['Max_Profit'] = '₹' + display_portfolio['Max_Profit'].round(2).astype(str)
                    display_portfolio['Max_Loss'] = '₹' + display_portfolio['Max_Loss'].round(2).astype(str)
                    display_portfolio['Profit_Factor'] = display_portfolio['Profit_Factor'].round(2)
                    
                    # Color coding for returns
                    def color_returns(val):
                        if '%' in str(val):
                            num_val = float(str(val).replace('%', ''))
                            return 'color: green' if num_val > 0 else 'color: red' if num_val < 0 else 'color: black'
                        return 'color: black'
                    
                    styled_portfolio = display_portfolio.style.applymap(color_returns, subset=['Total_Return'])
                    st.dataframe(styled_portfolio, use_container_width=True)
                    
                    # Portfolio visualization
                    st.subheader("📈 Portfolio Performance Visualization")
                    
                    # Returns chart
                    fig_returns = px.bar(
                        portfolio_df, 
                        x='Stock', 
                        y='Total_Return',
                        title='Stock Returns Comparison',
                        color='Total_Return',
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig_returns, use_container_width=True)
                    
                    # Win rate vs Returns scatter
                    fig_scatter = px.scatter(
                        portfolio_df,
                        x='Win_Rate',
                        y='Total_Return',
                        size='Total_Trades',
                        hover_name='Stock',
                        title='Win Rate vs Returns Analysis',
                        labels={'Win_Rate': 'Win Rate (%)', 'Total_Return': 'Total Return (%)'}
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Download portfolio results
                    csv = display_portfolio.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Portfolio Analysis",
                        data=csv,
                        file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
    
    # Strategy explanation sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("🧠 Strategy Logic")
        
        st.markdown("""
        **Entry Conditions (5/7 required):**
        - Price > 50-day SMA
        - SMA 50 > SMA 200 (Uptrend)
        - RSI between 40-65
        - Volume > 1.2x average
        - Positive 5-day momentum
        - Price > Lower Bollinger Band
        - MACD bullish crossover
        
        **Exit Conditions (Any 1):**
        - Price < 50-day SMA
        - RSI > 75 (Overbought)
        - RSI < 30 (Oversold)
        - Price < Lower Bollinger Band
        - MACD bearish crossover
        
        **Risk Management:**
        - 2% Stop Loss
        - 4% Target
        - 10% position size per trade
        """)
        
        st.markdown("---")
        st.subheader("📞 Support")
        st.info("""
        For technical support or feature requests, 
        please contact the development team.
        
        **Version:** 2.0.0
        **Last Updated:** 2024
        """)

if __name__ == "__main__":
    main()
