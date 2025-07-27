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
                return         with col2:
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
                        
                        # Heat Maps Section
                        st.subheader("🔥 Returns Heat Maps (10-Year Analysis)")
                        
                        # Calculate returns for heatmaps
                        monthly_returns, quarterly_returns = screener.calculate_returns_for_heatmap(data)
                        
                        if monthly_returns is not None and quarterly_returns is not None:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**📅 Monthly Returns Heatmap**")
                                monthly_fig = screener.create_monthly_heatmap(monthly_returns, stock_name)
                                st.plotly_chart(monthly_fig, use_container_width=True)
                            
                            with col2:
                                st.markdown("**📊 Quarterly Returns Heatmap**")
                                quarterly_fig = screener.create_quarterly_heatmap(quarterly_returns, stock_name)
                                st.plotly_chart(quarterly_fig, use_container_width=True)
                            
                            # Seasonality Analysis
                            st.subheader("📈 Seasonality Analysis")
                            seasonality_fig = screener.create_seasonality_analysis(monthly_returns, stock_name)
                            st.plotly_chart(seasonality_fig, use_container_width=True)
                        
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
                            specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
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
                        recommendation = screener.enhanced_live_recommendation(stock_input)
                        
                        if recommendation:
                            st.success("✅ Live Signal Generated!")
                            
                            # Display recommendation with rating
                            st.subheader(f"📊 {recommendation['stock']} - Live Analysis")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Current Price", f"₹{recommendation['current_price']:.2f}")
                            with col2:
                                st.metric("Signal", recommendation['recommendation'])
                            with col3:
                                st.metric("Rating", f"{recommendation['rating_score']}/10")
                            with col4:
                                st.metric("Entry Score", recommendation['entry_score'])
                            
                            # Rating recommendation
                            st.subheader("🎯 AI Rating & Recommendation")
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                rating_color = "green" if recommendation['rating_score'] >= 7 else "orange" if recommendation['rating_score'] >= 5 else "red"
                                st.markdown(f"<h3 style='color: {rating_color}'>{recommendation['rating_recommendation']}</h3>", unsafe_allow_html=True)
                            
                            with col2:
                                st.info(f"**Reason:** {recommendation['rating_reason']}")
                            
                            # Trading levels
                            st.subheader("🎯 Trading Levels & Risk Management")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            with col1:
                                st.metric("Stop Loss", f"₹{recommendation['stop_loss']:.2f}")
                            with col2:
                                st.metric("Target", f"₹{recommendation['target']:.2f}")
                            with col3:
                                st.metric("Risk:Reward", f"1:{recommendation['risk_reward_ratio']:.2f}")
                            with col4:
                                st.metric("RSI", f"{recommendation['rsi']:.1f}")
                            with col5:
                                st.metric("Volume Ratio", f"{recommendation['volume_ratio']:.2f}x")
                            
                            # Technical Summary with more details
                            st.subheader("📈 Comprehensive Technical Summary")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write("**📊 Moving Averages:**")
                                st.write(f"• Current Price: ₹{recommendation['current_price']:.2f}")
                                st.write(f"• SMA 50: ₹{recommendation['sma_50']:.2f}")
                                st.write(f"• SMA 200: ₹{recommendation['sma_200']:.2f}")
                                trend = "🟢 Bullish" if recommendation['sma_50'] > recommendation['sma_200'] else "🔴 Bearish"
                                st.write(f"• Long-term Trend: {trend}")
                            
                            with col2:
                                st.write("**⚡ Momentum Analysis:**")
                                st.write(f"• 5-day Change: {recommendation['price_change_5']:.2f}%")
                                st.write(f"• 10-day Change: {recommendation['price_change_10']:.2f}%")
                                momentum = "🟢 Positive" if recommendation['price_change_5'] > 0 else "🔴 Negative"
                                st.write(f"• Short-term Momentum: {momentum}")
                                rsi_status = "🟢 Normal" if 30 <= recommendation['rsi'] <= 70 else "🟡 Extreme"
                                st.write(f"• RSI Status: {rsi_status}")
                            
                            with col3:
                                st.write("**📈 Volume & Signals:**")
                                vol_status = "🟢 High" if recommendation['volume_ratio'] > 1.2 else "🟡 Normal" if recommendation['volume_ratio'] > 0.8 else "🔴 Low"
                                st.write(f"• Volume Status: {vol_status}")
                                st.write(f"• Signal Strength: {recommendation['entry_score']}")
                                st.write(f"• Overall Rating: {recommendation['rating_score']}/10")
                            
                            # Strategy Logic
                            st.subheader("🧠 Strategy Logic & Analysis")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if recommendation['entry_logic']:
                                    st.success(f"**✅ Entry Conditions Met:**\n{recommendation['entry_logic']}")
                                else:
                                    st.info("**Entry Conditions:** No strong entry signals currently")
                            
                            with col2:
                                if recommendation['exit_logic']:
                                    st.warning(f"**⚠️ Exit Conditions Present:**\n{recommendation['exit_logic']}")
                                else:
                                    st.success("**Exit Conditions:** No exit signals currently")
                        else:
                            st.error("❌ Could not fetch data for this stock")
                else:
                    st.warning("⚠️ Please enter a stock symbol")
        
        # Enhanced Quick scanner for all Nifty 50
        st.markdown("---")
        st.subheader("🔍 Advanced Nifty 50 Scanner")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            scan_type = st.selectbox(
                "Scanner Type:",
                ["Quick Scan (Top 10)", "Full Scan (All 48)", "High Rating Only (8+)"]
            )
        
        with col2:
            min_rating = st.slider("Minimum Rating Filter:", 0, 10, 5)
        
        with col3:
            sort_by = st.selectbox(
                "Sort Results By:",
                ["Rating Score", "Entry Score", "Volume Ratio", "Price Change"]
            )
        
        if st.button("🚀 Run Advanced Scanner", type="secondary"):
            # Determine stocks to scan
            if scan_type == "Quick Scan (Top 10)":
                stocks_to_scan = list(screener.nifty50_stocks.keys())[:10]
            elif scan_type == "High Rating Only (8+)":
                stocks_to_scan = list(screener.nifty50_stocks.keys())
                min_rating = 8
            else:  # Full scan
                stocks_to_scan = list(screener.nifty50_stocks.keys())
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            scanner_results = []
            total_stocks = len(stocks_to_scan)
            
            for i, stock in enumerate(stocks_to_scan):
                progress_bar.progress((i + 1) / total_stocks)
                status_text.text(f"Scanning {stock}... ({i+1}/{total_stocks})")
                
                try:
                    recommendation = screener.enhanced_live_recommendation(stock)
                    if recommendation and recommendation['rating_score'] >= min_rating:
                        scanner_results.append(recommendation)
                except Exception as e:
                    st.warning(f"Error scanning {stock}: {str(e)}")
                    continue
            
            progress_bar.empty()
            status_text.empty()
            
            # Display enhanced scanner results
            if scanner_results:
                st.success(f"✅ Found {len(scanner_results)} stocks meeting criteria!")
                
                # Sort results
                if sort_by == "Rating Score":
                    scanner_results.sort(key=lambda x: x['rating_score'], reverse=True)
                elif sort_by == "Entry Score":
                    scanner_results.sort(key=lambda x: int(x['entry_score'].split('/')[0]), reverse=True)
                elif sort_by == "Volume Ratio":
                    scanner_results.sort(key=lambda x: x['volume_ratio'], reverse=True)
                elif sort_by == "Price Change":
                    scanner_results.sort(key=lambda x: x['price_change_5'], reverse=True)
                
                # Create enhanced DataFrame for display
                scanner_df = pd.DataFrame([
                    {
                        'Stock': r['stock'],
                        'Price': f"₹{r['current_price']:.2f}",
                        'Signal': r['recommendation'],
                        'Rating': f"{r['rating_score']}/10",
                        'AI Recommendation': r['rating_recommendation'],
                        'Entry Score': r['entry_score'],
                        'Stop Loss': f"₹{r['stop_loss']:.2f}",
                        'Target': f"₹{r['target']:.2f}",
                        'Risk:Reward': f"1:{r['risk_reward_ratio']:.2f}",
                        'RSI': f"{r['rsi']:.1f}",
                        'Volume': f"{r['volume_ratio']:.2f}x",
                        '5D Change': f"{r['price_change_5']:.2f}%"
                    }
                    for r in scanner_results
                ])
                
                # Color coding function
                def highlight_recommendations(s):
                    if 'STRONG BUY' in str(s):
                        return 'background-color: darkgreen; color: white'
                    elif 'BUY' in str(s):
                        return 'background-color: lightgreen'
                    elif 'STRONG SELL' in str(s) or 'SELL' in str(s):
                        return 'background-color: lightcoral'
                    elif 'AVOID' in str(s):
                        return 'background-color: red; color: white'
                    else:
                        return 'background-color: lightyellow'
                
                def highlight_rating(val):
                    try:
                        rating = int(val.split('/')[0])
                        if rating >= 8:
                            return 'background-color: darkgreen; color: white'
                        elif rating >= 6:
                            return 'background-color: lightgreen'
                        elif rating >= 4:
                            return 'background-color: lightyellow'
                        else:
                            return 'background-color: lightcoral'
                    except:
                        return ''
                
                styled_scanner = scanner_df.style.applymap(highlight_recommendations, subset=['AI Recommendation']) \
                                                  .applymap(highlight_rating, subset=['Rating'])
                
                st.dataframe(styled_scanner, use_container_width=True)
                
                # Summary statistics
                st.subheader("📊 Scanner Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    strong_buys = sum(1 for r in scanner_results if 'STRONG BUY' in r['rating_recommendation'])
                    st.metric("Strong Buy Signals", strong_buys)
                
                with col2:
                    avg_rating = np.mean([r['rating_score'] for r in scanner_results])
                    st.metric("Average Rating", f"{avg_rating:.1f}/10")
                
                with col3:
                    high_volume = sum(1 for r in scanner_results if r['volume_ratio'] > 1.5)
                    st.metric("High Volume Stocks", high_volume)
                
                with col4:
                    positive_momentum = sum(1 for r in scanner_results if r['price_change_5'] > 2)
                    st.metric("Strong Momentum", positive_momentum)
                
                # Download enhanced scanner results
                csv = scanner_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Enhanced Scanner Results",
                    data=csv,
                    file_name=f"enhanced_nifty50_scanner_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning(f"⚠️ No stocks found meeting the criteria (Rating >= {min_rating})")
                st.info("Try lowering the minimum rating filter or changing the scan type.")
    
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
        st.subheader("🧠 Enhanced Strategy Logic")
        
        st.markdown("""
        **Entry Conditions (5/7 required):**
        - ✅ Price > 50-day SMA
        - ✅ SMA 50 > SMA 200 (Uptrend)
        - ✅ RSI between 40-65
        - ✅ Volume > 1.2x average
        - ✅ Positive 5-day momentum
        - ✅ Price > Lower Bollinger Band
        - ✅ MACD bullish crossover
        
        **Exit Conditions (Any 1):**
        - ❌ Price < 50-day SMA
        - ❌ RSI > 75 (Overbought)
        - ❌ RSI < 30 (Oversold)
        - ❌ Price < Lower Bollinger Band
        - ❌ MACD bearish crossover
        
        **Risk Management:**
        - 🛡️ 2% Stop Loss
        - 🎯 4% Target
        - 💰 10% position size per trade
        
        **Rating System (Out of 10):**
        - 📊 Technical Score: 4 points
        - ⚡ Momentum Score: 2 points
        - 🎯 Signal Strength: 2 points
        - ⚖️ Risk-Reward Ratio: 2 points
        """)
        
        st.markdown("---")
        st.subheader("📈 Heat Map Features")
        st.info("""
        **Monthly Returns Heatmap:**
        - 10-year historical analysis
        - Month-wise performance patterns
        - Color-coded returns (Green=Positive, Red=Negative)
        
        **Quarterly Returns Heatmap:**
        - Quarter-wise performance analysis
        - Seasonal trends identification
        
        **Seasonality Analysis:**
        - Average monthly returns with volatility
        - Best/worst performing months
        - Statistical significance indicators
        """)
        
        st.markdown("---")
        st.subheader("🎯 Rating System Guide")
        
        st.markdown("""
        **Rating Interpretation:**
        - 🟢 **8-10:** Strong Buy/Sell signal
        - 🟡 **6-7:** Moderate signal, good setup
        - 🟠 **4-5:** Neutral, wait for clarity
        - 🔴 **0-3:** Weak signal, avoid entry
        
        **Scanner Features:**
        - Full Nifty 50 coverage (48 stocks)
        - Rating-based filtering
        - Multiple sorting options
        - Risk-reward analysis
        - Volume and momentum filters
        """)
        
        st.markdown("---")
        st.subheader("📞 Support & Info")
        st.success("""
        🚀 **Enhanced Features:**
        - 10-year heat map analysis
        - AI-powered rating system
        - Full portfolio scanner
        - Advanced risk management
        - Real-time data integration
        
        **Version:** 3.0.0  
        **Last Updated:** 2024
        **Data Source:** Yahoo Finance
        """)
        
        # Performance metrics explanation
        st.markdown("---")
        st.subheader("📊 Performance Metrics")
        with st.expander("Click to understand metrics"):
            st.markdown("""
            st.markdown("""
            **Win Rate:** Percentage of profitable trades
            
            **Profit Factor:** Gross Profit ÷ Gross Loss
            - >2.0: Excellent
            - 1.5-2.0: Good  
            - 1.0-1.5: Average
            - <1.0: Poor
            
            **Sharpe Ratio:** Risk-adjusted returns
            - >1.0: Good
            - >2.0: Excellent
            
            **Max Drawdown:** Largest peak-to-trough decline
            
            **Average Trade Duration:** Days per trade
            
            **Risk-Reward Ratio:** Target ÷ Stop Loss
            - Minimum 1:2 recommended
            """)

if __name__ == "__main__":
    main()
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
    
    def calculate_returns_for_heatmap(self, data):
        """Calculate monthly and quarterly returns for heatmap visualization"""
        try:
            # Monthly returns
            monthly_data = data['Close'].resample('M').last()
            monthly_returns = monthly_data.pct_change().dropna() * 100
            
            # Quarterly returns  
            quarterly_data = data['Close'].resample('Q').last()
            quarterly_returns = quarterly_data.pct_change().dropna() * 100
            
            return monthly_returns, quarterly_returns
        except Exception as e:
            st.error(f"Error calculating returns: {e}")
            return None, None
    
    def create_monthly_heatmap(self, monthly_returns, stock_name):
        """Create monthly returns heatmap"""
        try:
            # Create DataFrame with year and month
            heatmap_data = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month_name(),
                'Returns': monthly_returns.values
            })
            
            # Pivot for heatmap
            pivot_data = heatmap_data.pivot_table(
                values='Returns', 
                index='Year', 
                columns='Month', 
                aggfunc='mean'
            )
            
            # Reorder months correctly
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            pivot_data = pivot_data.reindex(columns=month_order)
            
            # Create Plotly heatmap
            fig = px.imshow(
                pivot_data,
                labels=dict(x="Month", y="Year", color="Returns (%)"),
                title=f"{stock_name} - Monthly Returns Heatmap",
                color_continuous_scale='RdYlGn',
                aspect="auto"
            )
            
            fig.update_layout(
                height=400,
                xaxis_title="Month",
                yaxis_title="Year"
            )
            
            return fig
        except Exception as e:
            st.error(f"Error creating monthly heatmap: {e}")
            return px.scatter(title="Error creating heatmap")
    
    def create_quarterly_heatmap(self, quarterly_returns, stock_name):
        """Create quarterly returns heatmap"""
        try:
            # Create DataFrame with year and quarter
            heatmap_data = pd.DataFrame({
                'Year': quarterly_returns.index.year,
                'Quarter': 'Q' + quarterly_returns.index.quarter.astype(str),
                'Returns': quarterly_returns.values
            })
            
            # Pivot for heatmap
            pivot_data = heatmap_data.pivot_table(
                values='Returns', 
                index='Year', 
                columns='Quarter', 
                aggfunc='mean'
            )
            
            # Ensure correct quarter order
            quarter_order = ['Q1', 'Q2', 'Q3', 'Q4']
            pivot_data = pivot_data.reindex(columns=quarter_order)
            
            # Create Plotly heatmap
            fig = px.imshow(
                pivot_data,
                labels=dict(x="Quarter", y="Year", color="Returns (%)"),
                title=f"{stock_name} - Quarterly Returns Heatmap",
                color_continuous_scale='RdYlGn',
                aspect="auto"
            )
            
            fig.update_layout(
                height=400,
                xaxis_title="Quarter",
                yaxis_title="Year"
            )
            
            return fig
        except Exception as e:
            st.error(f"Error creating quarterly heatmap: {e}")
            return px.scatter(title="Error creating heatmap")
    
    def create_seasonality_analysis(self, monthly_returns, stock_name):
        """Create seasonality analysis chart"""
        try:
            # Calculate average returns by month
            seasonality_data = pd.DataFrame({
                'Month': monthly_returns.index.month_name(),
                'Returns': monthly_returns.values
            })
            
            avg_monthly_returns = seasonality_data.groupby('Month')['Returns'].agg(['mean', 'std']).reset_index()
            
            # Reorder months
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            avg_monthly_returns['Month'] = pd.Categorical(avg_monthly_returns['Month'], categories=month_order, ordered=True)
            avg_monthly_returns = avg_monthly_returns.sort_values('Month')
            
            # Create bar chart with error bars
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=avg_monthly_returns['Month'],
                y=avg_monthly_returns['mean'],
                error_y=dict(type='data', array=avg_monthly_returns['std']),
                marker_color=['green' if x > 0 else 'red' for x in avg_monthly_returns['mean']],
                name='Average Monthly Returns'
            ))
            
            fig.update_layout(
                title=f"{stock_name} - Seasonality Analysis (Average Monthly Returns)",
                xaxis_title="Month",
                yaxis_title="Average Returns (%)",
                height=400
            )
            
            return fig
        except Exception as e:
            st.error(f"Error creating seasonality analysis: {e}")
            return px.scatter(title="Error creating seasonality chart")
    
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
    
    def calculate_rating_score(self, recommendation_data):
        """Calculate comprehensive rating score out of 10"""
        score = 0
        max_score = 10
        
        # Technical Score (4 points)
        technical_conditions = [
            recommendation_data['current_price'] > recommendation_data['sma_50'],  # 1 point
            recommendation_data['sma_50'] > recommendation_data['sma_200'],       # 1 point  
            30 <= recommendation_data['rsi'] <= 70,                               # 1 point
            recommendation_data['volume_ratio'] > 1.0                            # 1 point
        ]
        score += sum(technical_conditions)
        
        # Momentum Score (2 points)
        if hasattr(recommendation_data, 'price_change_5'):
            if recommendation_data.get('price_change_5', 0) > 0.02:  # Strong positive momentum
                score += 2
            elif recommendation_data.get('price_change_5', 0) > 0:   # Positive momentum
                score += 1
        
        # Signal Strength Score (2 points)
        entry_score_num = int(recommendation_data['entry_score'].split('/')[0])
        if entry_score_num >= 6:
            score += 2
        elif entry_score_num >= 4:
            score += 1
        
        # Risk-Reward Score (2 points)
        current_price = recommendation_data['current_price']
        risk = abs(current_price - recommendation_data['stop_loss'])
        reward = abs(recommendation_data['target'] - current_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        if risk_reward_ratio >= 2:
            score += 2
        elif risk_reward_ratio >= 1.5:
            score += 1
        
        return min(score, max_score)
    
    def get_rating_recommendation(self, rating_score, signal_type):
        """Get recommendation based on rating score"""
        if rating_score >= 8:
            if 'BUY' in signal_type:
                return "🟢 STRONG BUY", "Excellent technical setup with strong momentum"
            else:
                return "🔴 STRONG SELL", "Poor technical setup, avoid or exit"
        elif rating_score >= 6:
            if 'BUY' in signal_type:
                return "🟢 BUY", "Good technical setup, favorable entry"
            else:
                return "🟡 HOLD", "Mixed signals, wait for better opportunity"
        elif rating_score >= 4:
            return "🟡 HOLD", "Neutral setup, wait for clearer signals"
        else:
            if 'SELL' in signal_type:
                return "🔴 SELL", "Weak technical setup, consider exit"
            else:
                return "🔴 AVOID", "Poor setup, avoid entry"
    
    def enhanced_live_recommendation(self, stock_name):
        """Enhanced live recommendation with rating system"""
        data = self.fetch_stock_data(stock_name, period="1y")
        if data is None:
            return None
        
        # Apply strategy
        strategy_data = self.generate_trading_signals(data)
        
        # Get latest data
        latest = strategy_data.iloc[-1]
        current_price = latest['Close']
        
        # Calculate additional metrics for rating
        price_change_5 = strategy_data['Close'].pct_change(5).iloc[-1]
        price_change_10 = strategy_data['Close'].pct_change(10).iloc[-1]
        
        # Basic recommendation
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
        
        # Create recommendation data for rating
        rec_data = {
            'current_price': current_price,
            'sma_50': latest['SMA_50'],
            'sma_200': latest['SMA_200'],
            'rsi': latest['RSI'],
            'volume_ratio': latest['Volume_Ratio'],
            'entry_score': f"{latest['Signal_Strength']}/7",
            'stop_loss': stop_loss,
            'target': target,
            'price_change_5': price_change_5,
            'price_change_10': price_change_10
        }
        
        # Calculate rating
        rating_score = self.calculate_rating_score(rec_data)
        rating_recommendation, rating_reason = self.get_rating_recommendation(rating_score, recommendation)
        
        return {
            'stock': stock_name.upper(),
            'current_price': current_price,
            'recommendation': recommendation,
            'rating_recommendation': rating_recommendation,
            'rating_score': rating_score,
            'rating_reason': rating_reason,
            'entry_score': f"{latest['Signal_Strength']}/7",
            'stop_loss': stop_loss,
            'target': target,
            'rsi': latest['RSI'],
            'volume_ratio': latest['Volume_Ratio'],
            'entry_logic': latest['Entry_Logic'],
            'exit_logic': latest['Exit_Logic'],
            'sma_50': latest['SMA_50'],
            'sma_200': latest['SMA_200'],
            'price_change_5': price_change_5 * 100,  # Convert to percentage
            'price_change_10': price_change_10 * 100,
            'risk_reward_ratio': (abs(target - current_price) / abs(current_price - stop_loss)) if abs(current_price - stop_loss) > 0 else 0
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
                        
                        # Heat Maps Section
                        st.subheader("🔥 Returns Heat Maps (10-Year Analysis)")
                        
                        # Calculate returns for heatmaps
                        monthly_returns, quarterly_returns = screener.calculate_returns_for_heatmap(data)
                        
                        if monthly_returns is not None and quarterly_returns is not None:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**📅 Monthly Returns Heatmap**")
                                monthly_fig = screener.create_monthly_heatmap(monthly_returns, stock_name)
                                st.plotly_chart(monthly_fig, use_container_width=True)
                            
                            with col2:
                                st.markdown("**📊 Quarterly Returns Heatmap**")
                                quarterly_fig = screener.create_quarterly_heatmap(quarterly_returns, stock_name)
                                st.plotly_chart(quarterly_fig, use_container_width=True)
                            
                            # Seasonality Analysis
                            st.subheader("📈 Seasonality Analysis")
                            seasonality_fig = screener.create_seasonality_analysis(monthly_returns, stock_name)
                            st.plotly_chart(seasonality_fig, use_container_width=True)
                        
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
                            specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
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
                        recommendation = screener.enhanced_live_recommendation(stock_input)
                        
                        if recommendation:
                            st.success("✅ Live Signal Generated!")
                            
                            # Display recommendation with rating
                            st.subheader(f"📊 {recommendation['stock']} - Live Analysis")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Current Price", f"₹{recommendation['current_price']:.2f}")
                            with col2:
                                st.metric("Signal", recommendation['recommendation'])
                            with col3:
                                st.metric("Rating", f"{recommendation['rating_score']}/10")
                            with col4:
                                st.metric("Entry Score", recommendation['entry_score'])
                            
                            # Rating recommendation
                            st.subheader("🎯 AI Rating & Recommendation")
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                rating_color = "green" if recommendation['rating_score'] >= 7 else "orange" if recommendation['rating_score'] >= 5 else "red"
                                st.markdown(f"<h3 style='color: {rating_color}'>{recommendation['rating_recommendation']}</h3>", unsafe_allow_html=True)
                            
                            with col2:
                                st.info(f"**Reason:** {recommendation['rating_reason']}")
                            
                            # Trading levels
                            st.subheader("🎯 Trading Levels & Risk Management")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            with col1:
                                st.metric("Stop Loss", f"₹{recommendation['stop_loss']:.2f}")
                            with col2:
                                st.metric("Target", f"₹{recommendation['target']:.2f}")
                            with col3:
                                st.metric("Risk:Reward", f"1:{recommendation['risk_reward_ratio']:.2f}")
                            with col4:
                                st.metric("RSI", f"{recommendation['rsi']:.1f}")
                            with col5:
                                st.metric("Volume Ratio", f"{recommendation['volume_ratio']:.2f}x")
                            
                            # Technical Summary with more details
                            st.subheader("📈 Comprehensive Technical Summary")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write("**📊 Moving Averages:**")
                                st.write(f"• Current Price: ₹{recommendation['current_price']:.2f}")
                                st.write(f"• SMA 50: ₹{recommendation['sma_50']:.2f}")
                                st.write(f"• SMA 200: ₹{recommendation['sma_200']:.2f}")
                                trend = "🟢 Bullish" if recommendation['sma_50'] > recommendation['sma_200'] else "🔴 Bearish"
                                st.write(f"• Long-term Trend: {trend}")
                            
                            with col2:
                                st.write("**⚡ Momentum Analysis:**")
                                st.write(f"• 5-day Change: {recommendation['price_change_5']:.2f}%")
                                st.write(f"• 10-day Change: {recommendation['price_change_10']:.2f}%")
                                momentum = "🟢 Positive" if recommendation['price_change_5'] > 0 else "🔴 Negative"
                                st.write(f"• Short-term Momentum: {momentum}")
                                rsi_status = "🟢 Normal" if 30 <= recommendation['rsi'] <= 70 else "🟡 Extreme"
                                st.write(f"• RSI Status: {rsi_status}")
                            
                            with col3:
                                st.write("**📈 Volume & Signals:**")
                                vol_status = "🟢 High" if recommendation['volume_ratio'] > 1.2 else "🟡 Normal" if recommendation['volume_ratio'] > 0.8 else "🔴 Low"
                                st.write(f"• Volume Status: {vol_status}")
                                st.write(f"• Signal Strength: {recommendation['entry_score']}")
                                st.write(f"• Overall Rating: {recommendation['rating_score']}/10")
                            
                            # Strategy Logic
                            st.subheader("🧠 Strategy Logic & Analysis")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if recommendation['entry_logic']:
                                    st.success(f"**✅ Entry Conditions Met:**\n{recommendation['entry_logic']}")
                                else:
                                    st.info("**Entry Conditions:** No strong entry signals currently")
                            
                            with col2:
                                if recommendation['exit_logic']:
                                    st.warning(f"**⚠️ Exit Conditions Present:**\n{recommendation['exit_logic']}")
                                else:
                                    st.success("**Exit Conditions:** No exit signals currently")
                        else:
                            st.error("❌ Could not fetch data for this stock")
                else:
                    st.warning("⚠️ Please enter a stock symbol")
        
        # Enhanced Quick scanner for all Nifty 50
        st.markdown("---")
        st.subheader("🔍 Advanced Nifty 50 Scanner")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            scan_type = st.selectbox(
                "Scanner Type:",
                ["Quick Scan (Top 10)", "Full Scan (All 48)", "High Rating Only (8+)"]
            )
        
        with col2:
            min_rating = st.slider("Minimum Rating Filter:", 0, 10, 5)
        
        with col3:
            sort_by = st.selectbox(
                "Sort Results By:",
                ["Rating Score", "Entry Score", "Volume Ratio", "Price Change"]
            )
        
        if st.button("🚀 Run Advanced Scanner", type="secondary"):
            # Determine stocks to scan
            if scan_type == "Quick Scan (Top 10)":
                stocks_to_scan = list(screener.nifty50_stocks.keys())[:10]
            elif scan_type == "High Rating Only (8+)":
                stocks_to_scan = list(screener.nifty50_stocks.keys())
                min_rating = 8
            else:  # Full scan
                stocks_to_scan = list(screener.nifty50_stocks.keys())
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            scanner_results = []
            total_stocks = len(stocks_to_scan)
            
            for i, stock in enumerate(stocks_to_scan):
                progress_bar.progress((i + 1) / total_stocks)
                status_text.text(f"Scanning {stock}... ({i+1}/{total_stocks})")
                
                try:
                    recommendation = screener.enhanced_live_recommendation(stock)
                    if recommendation and recommendation['rating_score'] >= min_rating:
                        scanner_results.append(recommendation)
                except Exception as e:
                    st.warning(f"Error scanning {stock}: {str(e)}")
                    continue
            
            progress_bar.empty()
            status_text.empty()
            
            # Display enhanced scanner results
            if scanner_results:
                st.success(f"✅ Found {len(scanner_results)} stocks meeting criteria!")
                
                # Sort results
                if sort_by == "Rating Score":
                    scanner_results.sort(key=lambda x: x['rating_score'], reverse=True)
                elif sort_by == "Entry Score":
                    scanner_results.sort(key=lambda x: int(x['entry_score'].split('/')[0]), reverse=True)
                elif sort_by == "Volume Ratio":
                    scanner_results.sort(key=lambda x: x['volume_ratio'], reverse=True)
                elif sort_by == "Price Change":
                    scanner_results.sort(key=lambda x: x['price_change_5'], reverse=True)
                
                # Create enhanced DataFrame for display
                scanner_df = pd.DataFrame([
                    {
                        'Stock': r['stock'],
                        'Price': f"₹{r['current_price']:.2f}",
                        'Signal': r['recommendation'],
                        'Rating': f"{r['rating_score']}/10",
                        'AI Recommendation': r['rating_recommendation'],
                        'Entry Score': r['entry_score'],
                        'Stop Loss': f"₹{r['stop_loss']:.2f}",
                        'Target': f"₹{r['target']:.2f}",
                        'Risk:Reward': f"1:{r['risk_reward_ratio']:.2f}",
                        'RSI': f"{r['rsi']:.1f}",
                        'Volume': f"{r['volume_ratio']:.2f}x",
                        '5D Change': f"{r['price_change_5']:.2f}%"
                    }
                    for r in scanner_results
                ])
                
                # Color coding function
                def highlight_recommendations(s):
                    if 'STRONG BUY' in str(s):
                        return 'background-color: darkgreen; color: white'
                    elif 'BUY' in str(s):
                        return 'background-color: lightgreen'
                    elif 'STRONG SELL' in str(s) or 'SELL' in str(s):
                        return 'background-color: lightcoral'
                    elif 'AVOID' in str(s):
                        return 'background-color: red; color: white'
                    else:
                        return 'background-color: lightyellow'
                
                def highlight_rating(val):
                    try:
                        rating = int(val.split('/')[0])
                        if rating >= 8:
                            return 'background-color: darkgreen; color: white'
                        elif rating >= 6:
                            return 'background-color: lightgreen'
                        elif rating >= 4:
                            return 'background-color: lightyellow'
                        else:
                            return 'background-color: lightcoral'
                    except:
                        return ''
                
                styled_scanner = scanner_df.style.applymap(highlight_recommendations, subset=['AI Recommendation']) \
                                                  .applymap(highlight_rating, subset=['Rating'])
                
                st.dataframe(styled_scanner, use_container_width=True)
                
                # Summary statistics
                st.subheader("📊 Scanner Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    strong_buys = sum(1 for r in scanner_results if 'STRONG BUY' in r['rating_recommendation'])
                    st.metric("Strong Buy Signals", strong_buys)
                
                with col2:
                    avg_rating = np.mean([r['rating_score'] for r in scanner_results])
                    st.metric("Average Rating", f"{avg_rating:.1f}/10")
                
                with col3:
                    high_volume = sum(1 for r in scanner_results if r['volume_ratio'] > 1.5)
                    st.metric("High Volume Stocks", high_volume)
                
                with col4:
                    positive_momentum = sum(1 for r in scanner_results if r['price_change_5'] > 2)
                    st.metric("Strong Momentum", positive_momentum)
                
                # Download enhanced scanner results
                csv = scanner_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Enhanced Scanner Results",
                    data=csv,
                    file_name=f"enhanced_nifty50_scanner_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning(f"⚠️ No stocks found meeting the criteria (Rating >= {min_rating})")
                st.info("Try lowering the minimum rating filter or changing the scan type.")

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
        st.subheader("🧠 Enhanced Strategy Logic")
        
        st.markdown("""
        **Entry Conditions (5/7 required):**
        - ✅ Price > 50-day SMA
        - ✅ SMA 50 > SMA 200 (Uptrend)
        - ✅ RSI between 40-65
        - ✅ Volume > 1.2x average
        - ✅ Positive 5-day momentum
        - ✅ Price > Lower Bollinger Band
        - ✅ MACD bullish crossover
        
        **Exit Conditions (Any 1):**
        - ❌ Price < 50-day SMA
        - ❌ RSI > 75 (Overbought)
        - ❌ RSI < 30 (Oversold)
        - ❌ Price < Lower Bollinger Band
        - ❌ MACD bearish crossover
        
        **Risk Management:**
        - 🛡️ 2% Stop Loss
        - 🎯 4% Target
        - 💰 10% position size per trade
        
        **Rating System (Out of 10):**
        - 📊 Technical Score: 4 points
        - ⚡ Momentum Score: 2 points
        - 🎯 Signal Strength: 2 points
        - ⚖️ Risk-Reward Ratio: 2 points
        """)
        
        st.markdown("---")
        st.subheader("📈 Heat Map Features")
        st.info("""
        **Monthly Returns Heatmap:**
        - 10-year historical analysis
        - Month-wise performance patterns
        - Color-coded returns (Green=Positive, Red=Negative)
        
        **Quarterly Returns Heatmap:**
        - Quarter-wise performance analysis
        - Seasonal trends identification
        
        **Seasonality Analysis:**
        - Average monthly returns with volatility
        - Best/worst performing months
        - Statistical significance indicators
        """)
        
        st.markdown("---")
        st.subheader("🎯 Rating System Guide")
        
        st.markdown("""
        **Rating Interpretation:**
        - 🟢 **8-10:** Strong Buy/Sell signal
        - 🟡 **6-7:** Moderate signal, good setup
        - 🟠 **4-5:** Neutral, wait for clarity
        - 🔴 **0-3:** Weak signal, avoid entry
        
        **Scanner Features:**
        - Full Nifty 50 coverage (48 stocks)
        - Rating-based filtering
        - Multiple sorting options
        - Risk-reward analysis
        - Volume and momentum filters
        """)
        
        st.markdown("---")
        st.subheader("📞 Support & Info")
        st.success("""
        🚀 **Enhanced Features:**
        - 10-year heat map analysis
        - AI-powered rating system
        - Full portfolio scanner
        - Advanced risk management
        - Real-time data integration
        
        **Version:** 3.0.0  
        **Last Updated:** 2024
        **Data Source:** Yahoo Finance
        """)
        
        # Performance metrics explanation
        st.markdown("---")
        st.subheader("📊 Performance Metrics")
        with st.expander("Click to understand metrics"):
            st.markdown("""
            **Win Rate:** Percentage of profitable trades
            
            **Profit Factor:** Gross Profit ÷ Gross Loss
            - >2.0: Excellent
            - 1.5-2.0: Good  
            - 1.0-1.5: Average
            - <1.0: Poor
            
            **Sharpe Ratio:** Risk-adjusted returns
            - >1.0: Good
            - >2.0: Excellent
            
            **Max Drawdown:** Largest peak-to-trough decline
            
            **Average Trade Duration:** Days per trade
            
            **Risk-Reward Ratio:** Target ÷ Stop Loss
            - Minimum 1:2 recommended
            """)

if __name__ == "__main__":
    main()
