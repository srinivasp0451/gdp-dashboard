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

# Set page config
st.set_page_config(
    page_title="Nifty50 Swing Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class Nifty50SwingTrader:
    def __init__(self):
        # Nifty50 stock symbols with .NS suffix for Yahoo Finance
        self.nifty50_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS',
            'MARUTI.NS', 'BAJFINANCE.NS', 'HCLTECH.NS', 'AXISBANK.NS', 'LT.NS',
            'DMART.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'ONGC.NS',
            'WIPRO.NS', 'NESTLEIND.NS', 'POWERGRID.NS', 'NTPC.NS', 'TECHM.NS',
            'M&M.NS', 'DIVISLAB.NS', 'TATAMOTORS.NS', 'JSWSTEEL.NS', 'HINDALCO.NS',
            'INDUSINDBK.NS', 'ADANIENT.NS', 'COALINDIA.NS', 'TATACONSUM.NS',
            'GRASIM.NS', 'BAJAJFINSV.NS', 'CIPLA.NS', 'HEROMOTOCO.NS', 'DRREDDY.NS',
            'EICHERMOT.NS', 'UPL.NS', 'APOLLOHOSP.NS', 'BAJAJ-AUTO.NS', 'BPCL.NS',
            'BRITANNIA.NS', 'KOTAKBANK.NS', 'HDFCLIFE.NS', 'SBILIFE.NS', 'TATASTEEL.NS',
            'ADANIPORTS.NS'
        ]
        
        # Simple mapping for user input
        self.stock_mapping = {
            stock.replace('.NS', '').lower(): stock for stock in self.nifty50_stocks
        }
        
        # Friendly names for dropdown
        self.stock_names = [stock.replace('.NS', '') for stock in self.nifty50_stocks]
    
    @st.cache_data
    def fetch_data(_self, symbol, period='10y'):
        """Fetch historical data for a stock"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if data.empty:
                return None
            return data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_returns(self, data):
        """Calculate monthly and quarterly returns"""
        # Monthly returns
        monthly_data = data.resample('M').last()
        monthly_returns = monthly_data['Close'].pct_change().dropna()
        
        # Quarterly returns
        quarterly_data = data.resample('Q').last()
        quarterly_returns = quarterly_data['Close'].pct_change().dropna()
        
        return monthly_returns, quarterly_returns
    
    def create_heatmaps(self, data, symbol):
        """Create interactive heatmaps using Plotly"""
        monthly_returns, quarterly_returns = self.calculate_returns(data)
        
        # Monthly returns heatmap
        monthly_pivot = pd.DataFrame({
            'Returns': monthly_returns,
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month
        })
        monthly_heatmap_data = monthly_pivot.pivot_table(
            values='Returns', index='Month', columns='Year'
        )
        
        # Quarterly returns heatmap
        quarterly_pivot = pd.DataFrame({
            'Returns': quarterly_returns,
            'Year': quarterly_returns.index.year,
            'Quarter': quarterly_returns.index.quarter
        })
        quarterly_heatmap_data = quarterly_pivot.pivot_table(
            values='Returns', index='Quarter', columns='Year'
        )
        
        return monthly_heatmap_data, quarterly_heatmap_data
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators for swing trading"""
        df = data.copy()
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
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
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Support and Resistance levels
        df['Support'] = df['Low'].rolling(window=20, min_periods=1).min()
        df['Resistance'] = df['High'].rolling(window=20, min_periods=1).max()
        
        return df
    
    def swing_trading_strategy(self, data):
        """Implement swing trading strategy with institutional participation"""
        df = self.calculate_technical_indicators(data)
        
        # Initialize signals
        df['Signal'] = 0
        df['Position'] = 0
        df['Entry_Price'] = 0
        df['Exit_Price'] = 0
        df['Stop_Loss'] = 0
        
        # Strategy parameters
        rsi_oversold = 30
        rsi_overbought = 70
        volume_threshold = 1.5  # Institutional participation indicator
        
        for i in range(50, len(df)):
            current_price = df['Close'].iloc[i]
            
            # Entry conditions (Long)
            if (df['Position'].iloc[i-1] == 0 and
                df['RSI'].iloc[i] < rsi_oversold and
                df['Close'].iloc[i] > df['SMA_20'].iloc[i] and
                df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i] and
                df['Volume_Ratio'].iloc[i] > volume_threshold and  # Institutional participation
                df['Close'].iloc[i] > df['BB_Lower'].iloc[i]):
                
                df.loc[df.index[i], 'Signal'] = 1
                df.loc[df.index[i], 'Position'] = 1
                df.loc[df.index[i], 'Entry_Price'] = current_price
                df.loc[df.index[i], 'Stop_Loss'] = current_price * 0.95  # 5% stop loss
            
            # Exit conditions
            elif (df['Position'].iloc[i-1] == 1 and
                  (df['RSI'].iloc[i] > rsi_overbought or
                   df['Close'].iloc[i] < df['Stop_Loss'].iloc[i-1] or
                   df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i])):
                
                df.loc[df.index[i], 'Signal'] = -1
                df.loc[df.index[i], 'Position'] = 0
                df.loc[df.index[i], 'Exit_Price'] = current_price
            
            # Carry forward position
            elif df['Position'].iloc[i-1] == 1:
                df.loc[df.index[i], 'Position'] = 1
                df.loc[df.index[i], 'Stop_Loss'] = df['Stop_Loss'].iloc[i-1]
        
        return df
    
    def backtest_strategy(self, df, symbol):
        """Backtest the swing trading strategy"""
        trades = []
        entry_price = 0
        entry_date = None
        
        for i, row in df.iterrows():
            if row['Signal'] == 1:  # Entry
                entry_price = row['Close']
                entry_date = i
            elif row['Signal'] == -1 and entry_price > 0:  # Exit
                exit_price = row['Close']
                exit_date = i
                profit_points = exit_price - entry_price
                profit_pct = (profit_points / entry_price) * 100
                holding_days = (exit_date - entry_date).days
                
                trades.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': exit_date,
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'Profit_Points': profit_points,
                    'Profit_Pct': profit_pct,
                    'Holding_Days': holding_days
                })
                entry_price = 0
        
        trades_df = pd.DataFrame(trades)
        
        if not trades_df.empty:
            # Performance metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['Profit_Points'] > 0])
            losing_trades = total_trades - winning_trades
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            avg_profit = trades_df['Profit_Points'].mean()
            total_profit = trades_df['Profit_Points'].sum()
            max_profit = trades_df['Profit_Points'].max()
            max_loss = trades_df['Profit_Points'].min()
            
            metrics = {
                'Total Trades': total_trades,
                'Winning Trades': winning_trades,
                'Losing Trades': losing_trades,
                'Win Rate (%)': round(win_rate, 2),
                'Total Profit Points': round(total_profit, 2),
                'Average Profit per Trade': round(avg_profit, 2),
                'Maximum Profit': round(max_profit, 2),
                'Maximum Loss': round(max_loss, 2),
                'Meets Profit Criteria': "✅ Yes" if total_profit >= 100 else "❌ No"
            }
            
            return trades_df, metrics
        
        return pd.DataFrame(), {}
    
    def create_interactive_plots(self, df, symbol):
        """Create interactive plots using Plotly"""
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                f'{symbol} - Price Action with Trading Signals',
                'RSI (Relative Strength Index)',
                'MACD',
                'Volume Analysis'
            ),
            vertical_spacing=0.08,
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Price chart with Bollinger Bands
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='blue', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='red', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', 
                      name='Bollinger Bands', fillcolor='rgba(128,128,128,0.2)'),
            row=1, col=1
        )
        
        # Buy and Sell signals
        entries = df[df['Signal'] == 1]
        exits = df[df['Signal'] == -1]
        
        if not entries.empty:
            fig.add_trace(
                go.Scatter(x=entries.index, y=entries['Close'], mode='markers', 
                          marker=dict(symbol='triangle-up', size=10, color='green'),
                          name='Buy Signal'),
                row=1, col=1
            )
        
        if not exits.empty:
            fig.add_trace(
                go.Scatter(x=exits.index, y=exits['Close'], mode='markers',
                          marker=dict(symbol='triangle-down', size=10, color='red'),
                          name='Sell Signal'),
                row=1, col=1
            )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='orange')),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", row=2, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal Line', line=dict(color='red')),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram', opacity=0.3),
            row=3, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume', opacity=0.3),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Volume_SMA'], name='Volume SMA', line=dict(color='red')),
            row=4, col=1
        )
        
        fig.update_layout(height=1000, showlegend=True, title_text=f"Technical Analysis for {symbol}")
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        
        return fig
    
    def get_recommendation(self, symbol):
        """Get swing trading recommendation for a stock"""
        data = self.fetch_data(symbol)
        if data is None:
            return None
        
        df = self.swing_trading_strategy(data)
        
        # Get latest data point
        latest = df.iloc[-1]
        
        # Calculate recommendation score
        score = 0
        reasons = []
        
        # RSI analysis
        if latest['RSI'] < 35:
            score += 1
            reasons.append("RSI oversold")
        elif latest['RSI'] > 65:
            score -= 1
            reasons.append("RSI overbought")
        
        # MACD analysis
        if latest['MACD'] > latest['MACD_Signal']:
            score += 1
            reasons.append("MACD bullish")
        else:
            score -= 1
            reasons.append("MACD bearish")
        
        # Moving average analysis
        if latest['Close'] > latest['SMA_20']:
            score += 1
            reasons.append("Above SMA20")
        
        # Volume analysis (institutional participation)
        if latest['Volume_Ratio'] > 1.5:
            score += 1
            reasons.append("High institutional volume")
        
        # Bollinger Bands
        if latest['Close'] < latest['BB_Lower']:
            score += 1
            reasons.append("Near lower Bollinger Band")
        
        # Normalize score to 1-5 scale
        recommendation_score = max(1, min(5, score + 3))
        
        # Determine action
        if recommendation_score >= 4:
            action = "STRONG BUY"
        elif recommendation_score >= 3:
            action = "BUY"
        elif recommendation_score >= 2:
            action = "HOLD"
        else:
            action = "SELL"
        
        # Calculate entry, exit, and stop loss
        entry_price = latest['Close']
        target_price = entry_price * 1.1  # 10% target
        stop_loss = entry_price * 0.95   # 5% stop loss
        
        return {
            'Stock': symbol.replace('.NS', ''),
            'Action': action,
            'Entry_Price': round(entry_price, 2),
            'Target_Price': round(target_price, 2),
            'Stop_Loss': round(stop_loss, 2),
            'Score': recommendation_score,
            'Reasons': ', '.join(reasons),
            'Current_RSI': round(latest['RSI'], 2),
            'Volume_Ratio': round(latest['Volume_Ratio'], 2)
        }

# Initialize the trader
@st.cache_resource
def get_trader():
    return Nifty50SwingTrader()

def main():
    st.title("📈 Nifty50 Swing Trading System")
    st.markdown("### Advanced Technical Analysis & Swing Trading Recommendations")
    
    trader = get_trader()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Stock Recommendation", "Complete Technical Analysis", "Bulk Screening"]
    )
    
    if analysis_type == "Stock Recommendation":
        st.header("🎯 Get Trading Recommendation")
        
        # Stock selection
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_stock = st.selectbox(
                "Select a Nifty50 Stock",
                trader.stock_names,
                index=0
            )
        
        with col2:
            if st.button("Get Recommendation", type="primary"):
                with st.spinner("Analyzing stock..."):
                    symbol = f"{selected_stock}.NS"
                    recommendation = trader.get_recommendation(symbol)
                    
                    if recommendation:
                        # Display recommendation in a nice format
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Recommendation", recommendation['Action'])
                            st.metric("Score", f"{recommendation['Score']}/5")
                        
                        with col2:
                            st.metric("Entry Price", f"₹{recommendation['Entry_Price']}")
                            st.metric("Target Price", f"₹{recommendation['Target_Price']}")
                        
                        with col3:
                            st.metric("Stop Loss", f"₹{recommendation['Stop_Loss']}")
                            st.metric("Current RSI", recommendation['Current_RSI'])
                        
                        # Additional details
                        st.subheader("Analysis Details")
                        st.write(f"**Reasons:** {recommendation['Reasons']}")
                        st.write(f"**Volume Ratio:** {recommendation['Volume_Ratio']}")
                        
                        # Create a DataFrame for tabular view
                        df_rec = pd.DataFrame([recommendation])
                        st.subheader("Detailed Recommendation Table")
                        st.dataframe(df_rec, use_container_width=True)
    
    elif analysis_type == "Complete Technical Analysis":
        st.header("📊 Complete Technical Analysis")
        
        # Stock selection
        selected_stock = st.selectbox(
            "Select a Nifty50 Stock for Complete Analysis",
            trader.stock_names,
            index=0
        )
        
        if st.button("Run Complete Analysis", type="primary"):
            symbol = f"{selected_stock}.NS"
            
            with st.spinner("Fetching data and running analysis..."):
                # Fetch data
                data = trader.fetch_data(symbol)
                
                if data is not None:
                    # Create tabs for different analyses
                    tab1, tab2, tab3, tab4 = st.tabs(["Returns Heatmaps", "Technical Analysis", "Backtesting Results", "Trading Signals"])
                    
                    with tab1:
                        st.subheader("📅 Returns Heatmaps")
                        
                        # Create heatmaps
                        monthly_heatmap, quarterly_heatmap = trader.create_heatmaps(data, symbol)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Monthly Returns Heatmap**")
                            fig_monthly = px.imshow(
                                monthly_heatmap.values,
                                x=monthly_heatmap.columns,
                                y=monthly_heatmap.index,
                                color_continuous_scale='RdYlGn',
                                aspect="auto",
                                title="Monthly Returns"
                            )
                            fig_monthly.update_layout(
                                xaxis_title="Year",
                                yaxis_title="Month"
                            )
                            st.plotly_chart(fig_monthly, use_container_width=True)
                        
                        with col2:
                            st.write("**Quarterly Returns Heatmap**")
                            fig_quarterly = px.imshow(
                                quarterly_heatmap.values,
                                x=quarterly_heatmap.columns,
                                y=quarterly_heatmap.index,
                                color_continuous_scale='RdYlGn',
                                aspect="auto",
                                title="Quarterly Returns"
                            )
                            fig_quarterly.update_layout(
                                xaxis_title="Year",
                                yaxis_title="Quarter"
                            )
                            st.plotly_chart(fig_quarterly, use_container_width=True)
                    
                    with tab2:
                        st.subheader("📈 Technical Analysis Charts")
                        
                        # Run strategy to get technical indicators
                        df = trader.swing_trading_strategy(data)
                        
                        # Create interactive plots
                        fig = trader.create_interactive_plots(df, symbol)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        st.subheader("🔍 Backtesting Results")
                        
                        # Run backtesting
                        trades_df, metrics = trader.backtest_strategy(df, symbol)
                        
                        if metrics:
                            # Display metrics in columns
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Trades", metrics['Total Trades'])
                                st.metric("Win Rate", f"{metrics['Win Rate (%)']}%")
                            
                            with col2:
                                st.metric("Winning Trades", metrics['Winning Trades'])
                                st.metric("Losing Trades", metrics['Losing Trades'])
                            
                            with col3:
                                st.metric("Total Profit Points", metrics['Total Profit Points'])
                                st.metric("Average Profit", metrics['Average Profit per Trade'])
                            
                            with col4:
                                st.metric("Max Profit", metrics['Maximum Profit'])
                                st.metric("Max Loss", metrics['Maximum Loss'])
                            
                            # Profit criteria check
                            if metrics['Total Profit Points'] >= 100:
                                st.success("✅ Strategy meets profit criteria (≥100 points)")
                            else:
                                st.warning(f"❌ Strategy needs improvement (Total: {metrics['Total Profit Points']} points)")
                            
                            # Display individual trades
                            if not trades_df.empty:
                                st.subheader("Individual Trades")
                                st.dataframe(trades_df, use_container_width=True)
                        else:
                            st.warning("No trades found in the backtesting period.")
                    
                    with tab4:
                        st.subheader("🎯 Current Trading Signals")
                        
                        # Get latest recommendation
                        recommendation = trader.get_recommendation(symbol)
                        
                        if recommendation:
                            # Current status
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.info(f"**Current Recommendation:** {recommendation['Action']}")
                                st.info(f"**Entry Price:** ₹{recommendation['Entry_Price']}")
                                st.info(f"**Target Price:** ₹{recommendation['Target_Price']}")
                            
                            with col2:
                                st.info(f"**Stop Loss:** ₹{recommendation['Stop_Loss']}")
                                st.info(f"**Score:** {recommendation['Score']}/5")
                                st.info(f"**Current RSI:** {recommendation['Current_RSI']}")
                            
                            st.write(f"**Analysis:** {recommendation['Reasons']}")
                
                else:
                    st.error("Failed to fetch data for the selected stock.")
    
    elif analysis_type == "Bulk Screening":
        st.header("🔍 Bulk Stock Screening")
        st.write("Screen multiple Nifty50 stocks for swing trading opportunities")
        
        # Multi-select for stocks
        selected_stocks = st.multiselect(
            "Select stocks to screen",
            trader.stock_names,
            default=trader.stock_names[:5]  # Default to first 5 stocks
        )
        
        # Filtering options
        col1, col2, col3 = st.columns(3)
        with col1:
            min_score = st.slider("Minimum Score", 1, 5, 3)
        with col2:
            action_filter = st.selectbox("Action Filter", ["All", "STRONG BUY", "BUY", "HOLD", "SELL"])
        with col3:
            max_rsi = st.slider("Maximum RSI", 30, 100, 70)
        
        if st.button("Screen Selected Stocks", type="primary"):
            results = []
            
            progress_bar = st.progress(0)
            
            for i, stock in enumerate(selected_stocks):
                symbol = f"{stock}.NS"
                
                with st.spinner(f"Analyzing {stock}..."):
                    recommendation = trader.get_recommendation(symbol)
                    
                    if recommendation:
                        # Apply filters
                        if (recommendation['Score'] >= min_score and
                            recommendation['Current_RSI'] <= max_rsi and
                            (action_filter == "All" or recommendation['Action'] == action_filter)):
                            results.append(recommendation)
                
                progress_bar.progress((i + 1) / len(selected_stocks))
            
            if results:
                st.success(f"Found {len(results)} stocks matching your criteria:")
                
                # Create DataFrame and display
                results_df = pd.DataFrame(results)
                
                # Sort by score descending
                results_df = results_df.sort_values('Score', ascending=False)
                
                # Display results
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    column_config={
                        "Action": st.column_config.TextColumn("Action", width="small"),
                        "Score": st.column_config.NumberColumn("Score", format="%d/5"),
                        "Entry_Price": st.column_config.NumberColumn("Entry Price", format="₹%.2f"),
                        "Target_Price": st.column_config.NumberColumn("Target Price", format="₹%.2f"),
                        "Stop_Loss": st.column_config.NumberColumn("Stop Loss", format="₹%.2f"),
                    }
                )
                
                # Download option
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"swing_trading_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No stocks match your filtering criteria.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p><strong>Nifty50 Swing Trading System</strong> - Advanced Technical Analysis & Trading Recommendations</p>
            <p><em>⚠️ This is for educational purposes only. Please consult with a financial advisor before making investment decisions.</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
