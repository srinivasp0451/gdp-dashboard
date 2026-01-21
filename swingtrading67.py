import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Options Chain Momentum Predictor",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Zero to Hero Options Chain Momentum Predictor")
st.markdown("""
**Predict Big Momentum Moves using Options Chain Analysis**
- Put-Call Ratio (PCR) Analysis
- Open Interest Analysis
- Max Pain Calculation
- Implied Volatility Analysis
- Support/Resistance from Options Data
""")

# Helper Functions
def get_nse_option_chain(symbol):
    """Fetch NSE options data using NSEpy or web scraping"""
    try:
        # For demonstration - using mock data structure
        # In production, use NSEpy or official NSE API
        st.warning("⚠️ NSE real-time data requires authentication. Using demo structure.")
        return None
    except Exception as e:
        st.error(f"Error fetching NSE data: {e}")
        return None

def get_yahoo_options(ticker):
    """Fetch options data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        exp_dates = stock.options
        
        if not exp_dates:
            return None, None
        
        # Get nearest expiry
        exp_date = exp_dates[0]
        opt_chain = stock.option_chain(exp_date)
        
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        return calls, puts, exp_date
    except Exception as e:
        st.error(f"Error fetching options data: {e}")
        return None, None, None

def calculate_pcr(calls, puts):
    """Calculate Put-Call Ratio"""
    try:
        put_oi = puts['openInterest'].sum()
        call_oi = calls['openInterest'].sum()
        
        put_vol = puts['volume'].sum()
        call_vol = calls['volume'].sum()
        
        pcr_oi = put_oi / call_oi if call_oi > 0 else 0
        pcr_vol = put_vol / call_vol if call_vol > 0 else 0
        
        return pcr_oi, pcr_vol
    except:
        return 0, 0

def calculate_max_pain(calls, puts):
    """Calculate Max Pain - price where most options expire worthless"""
    try:
        strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
        pain_values = []
        
        for strike in strikes:
            call_pain = calls[calls['strike'] < strike]['openInterest'].sum() * (strike - calls[calls['strike'] < strike]['strike']).sum()
            put_pain = puts[puts['strike'] > strike]['openInterest'].sum() * (puts[puts['strike'] > strike]['strike'] - strike).sum()
            total_pain = call_pain + put_pain
            pain_values.append(total_pain)
        
        max_pain_strike = strikes[pain_values.index(min(pain_values))]
        return max_pain_strike
    except:
        return 0

def analyze_oi_buildup(calls, puts, current_price):
    """Analyze Open Interest buildup for support/resistance"""
    try:
        # Call OI analysis
        calls_sorted = calls.sort_values('openInterest', ascending=False).head(5)
        puts_sorted = puts.sort_values('openInterest', ascending=False).head(5)
        
        # Find strong resistance (high call OI above current price)
        resistance = calls_sorted[calls_sorted['strike'] > current_price]['strike'].min()
        
        # Find strong support (high put OI below current price)
        support = puts_sorted[puts_sorted['strike'] < current_price]['strike'].max()
        
        return support, resistance, calls_sorted, puts_sorted
    except:
        return 0, 0, None, None

def calculate_momentum_signal(pcr_oi, pcr_vol, max_pain, current_price, support, resistance):
    """Calculate momentum signal based on multiple factors"""
    signals = []
    score = 0
    
    # PCR Analysis
    if pcr_oi > 1.2:
        signals.append("🟢 High PCR (OI): Bullish - More puts than calls")
        score += 2
    elif pcr_oi < 0.8:
        signals.append("🔴 Low PCR (OI): Bearish - More calls than puts")
        score -= 2
    else:
        signals.append("🟡 Neutral PCR (OI)")
    
    # Max Pain Analysis
    if current_price < max_pain * 0.98:
        signals.append(f"🟢 Price below Max Pain (${max_pain:.2f}): Potential upside")
        score += 1
    elif current_price > max_pain * 1.02:
        signals.append(f"🔴 Price above Max Pain (${max_pain:.2f}): Potential downside")
        score -= 1
    else:
        signals.append(f"🟡 Price near Max Pain (${max_pain:.2f})")
    
    # Support/Resistance Analysis
    if support and resistance:
        range_pct = ((resistance - support) / current_price) * 100
        signals.append(f"📊 Trading Range: ${support:.2f} - ${resistance:.2f} ({range_pct:.1f}%)")
        
        # Position in range
        if current_price <= support * 1.01:
            signals.append("🟢 Near Support: Potential bounce")
            score += 1
        elif current_price >= resistance * 0.99:
            signals.append("🔴 Near Resistance: Potential rejection")
            score -= 1
    
    # Final Signal
    if score >= 3:
        momentum = "🚀 STRONG BULLISH"
    elif score >= 1:
        momentum = "📈 BULLISH"
    elif score <= -3:
        momentum = "📉 STRONG BEARISH"
    elif score <= -1:
        momentum = "📉 BEARISH"
    else:
        momentum = "➡️ NEUTRAL"
    
    return momentum, signals, score

def plot_oi_chart(calls, puts, current_price, max_pain):
    """Create Open Interest visualization"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Open Interest Distribution', 'Volume Distribution'),
        vertical_spacing=0.15
    )
    
    # OI Chart
    fig.add_trace(
        go.Bar(x=calls['strike'], y=calls['openInterest'], 
               name='Call OI', marker_color='green', opacity=0.6),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=puts['strike'], y=puts['openInterest'], 
               name='Put OI', marker_color='red', opacity=0.6),
        row=1, col=1
    )
    
    # Volume Chart
    fig.add_trace(
        go.Bar(x=calls['strike'], y=calls['volume'], 
               name='Call Vol', marker_color='lightgreen', opacity=0.6),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=puts['strike'], y=puts['volume'], 
               name='Put Vol', marker_color='lightcoral', opacity=0.6),
        row=2, col=1
    )
    
    # Add current price line
    fig.add_vline(x=current_price, line_dash="dash", line_color="blue", 
                  annotation_text="Current Price", row=1, col=1)
    fig.add_vline(x=current_price, line_dash="dash", line_color="blue", row=2, col=1)
    
    # Add max pain line
    if max_pain > 0:
        fig.add_vline(x=max_pain, line_dash="dot", line_color="purple", 
                      annotation_text="Max Pain", row=1, col=1)
    
    fig.update_layout(height=700, showlegend=True, title_text="Options Chain Analysis")
    fig.update_xaxes(title_text="Strike Price", row=2, col=1)
    fig.update_yaxes(title_text="Open Interest", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def plot_price_levels(ticker, support, resistance, max_pain):
    """Plot price with support/resistance levels"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='Price'
        ))
        
        # Add support/resistance lines
        if support > 0:
            fig.add_hline(y=support, line_dash="dash", line_color="green", 
                         annotation_text="Support")
        if resistance > 0:
            fig.add_hline(y=resistance, line_dash="dash", line_color="red", 
                         annotation_text="Resistance")
        if max_pain > 0:
            fig.add_hline(y=max_pain, line_dash="dot", line_color="purple", 
                         annotation_text="Max Pain")
        
        fig.update_layout(
            title=f"{ticker} Price with Key Levels",
            yaxis_title="Price",
            xaxis_title="Date",
            height=400
        )
        
        return fig
    except:
        return None

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Asset Class Selection
asset_class = st.sidebar.selectbox(
    "Select Asset Class",
    ["Indian Indices (NSE)", "US Stocks", "Crypto", "Forex", "Commodities"]
)

# Ticker input based on asset class
if asset_class == "Indian Indices (NSE)":
    ticker_options = ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY"]
    ticker = st.sidebar.selectbox("Select Index", ticker_options)
    st.sidebar.info("Note: NSE real-time data requires authentication. Use Yahoo Finance alternatives below.")
    use_nse = False
    
elif asset_class == "US Stocks":
    popular_stocks = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "SPY", "QQQ"]
    ticker = st.sidebar.selectbox("Select Stock", popular_stocks + ["Custom"])
    if ticker == "Custom":
        ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
    use_nse = False
    
elif asset_class == "Crypto":
    crypto_options = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"]
    ticker = st.sidebar.selectbox("Select Crypto", crypto_options)
    st.sidebar.warning("⚠️ Limited options data available for crypto")
    use_nse = False
    
elif asset_class == "Forex":
    forex_options = ["USDINR=X", "EURUSD=X", "GBPUSD=X", "JPYUSD=X"]
    ticker = st.sidebar.selectbox("Select Forex Pair", forex_options)
    st.sidebar.warning("⚠️ Limited options data available for forex")
    use_nse = False
    
else:  # Commodities
    commodity_options = ["GC=F", "SI=F", "CL=F", "NG=F"]  # Gold, Silver, Crude, Natural Gas
    commodity_names = ["Gold", "Silver", Crude Oil", "Natural Gas"]
    selection = st.sidebar.selectbox("Select Commodity", commodity_names)
    ticker = commodity_options[commodity_names.index(selection)]
    st.sidebar.warning("⚠️ Limited options data available for commodities")
    use_nse = False

# Analysis button
analyze_button = st.sidebar.button("🔍 Analyze Options Chain", type="primary")

# Main Analysis
if analyze_button:
    with st.spinner(f"Fetching options data for {ticker}..."):
        
        # Get current price
        try:
            stock = yf.Ticker(ticker)
            current_price = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice')
            if not current_price:
                hist = stock.history(period="1d")
                current_price = hist['Close'].iloc[-1] if not hist.empty else 0
        except:
            current_price = 0
        
        if current_price == 0:
            st.error("❌ Unable to fetch current price. Please check ticker symbol.")
        else:
            # Display current price
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            
            # Fetch options data
            calls, puts, exp_date = get_yahoo_options(ticker)
            
            if calls is not None and puts is not None:
                st.success(f"✅ Options data fetched successfully! Expiry: {exp_date}")
                
                # Calculate metrics
                pcr_oi, pcr_vol = calculate_pcr(calls, puts)
                max_pain = calculate_max_pain(calls, puts)
                support, resistance, top_calls, top_puts = analyze_oi_buildup(calls, puts, current_price)
                
                # Display key metrics
                with col2:
                    st.metric("PCR (OI)", f"{pcr_oi:.2f}")
                with col3:
                    st.metric("Max Pain", f"${max_pain:.2f}")
                
                col4, col5 = st.columns(2)
                with col4:
                    st.metric("Support Level", f"${support:.2f}" if support > 0 else "N/A")
                with col5:
                    st.metric("Resistance Level", f"${resistance:.2f}" if resistance > 0 else "N/A")
                
                # Calculate momentum signal
                momentum, signals, score = calculate_momentum_signal(
                    pcr_oi, pcr_vol, max_pain, current_price, support, resistance
                )
                
                # Display momentum signal
                st.markdown("---")
                st.subheader("🎯 Momentum Signal")
                st.markdown(f"## {momentum}")
                st.markdown(f"**Signal Strength Score: {score}**")
                
                st.markdown("### Analysis Details:")
                for signal in signals:
                    st.markdown(f"- {signal}")
                
                # Trading recommendation
                st.markdown("---")
                st.subheader("💡 Trading Insights")
                
                if score >= 2:
                    st.success("""
                    **Bullish Setup Detected:**
                    - Consider LONG positions or CALL options
                    - Watch for breakout above resistance
                    - Target: Next resistance level
                    - Stop Loss: Below support
                    """)
                elif score <= -2:
                    st.warning("""
                    **Bearish Setup Detected:**
                    - Consider SHORT positions or PUT options
                    - Watch for breakdown below support
                    - Target: Next support level
                    - Stop Loss: Above resistance
                    """)
                else:
                    st.info("""
                    **Neutral Zone:**
                    - Wait for clear directional move
                    - Consider range-bound strategies
                    - Monitor key levels for breakout
                    """)
                
                # Visualizations
                st.markdown("---")
                st.subheader("📊 Options Chain Visualization")
                
                oi_chart = plot_oi_chart(calls, puts, current_price, max_pain)
                st.plotly_chart(oi_chart, use_container_width=True)
                
                # Price chart with levels
                price_chart = plot_price_levels(ticker, support, resistance, max_pain)
                if price_chart:
                    st.plotly_chart(price_chart, use_container_width=True)
                
                # Detailed OI Tables
                st.markdown("---")
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.subheader("📞 Top Call OI")
                    if top_calls is not None:
                        display_calls = top_calls[['strike', 'openInterest', 'volume', 'impliedVolatility']].copy()
                        display_calls.columns = ['Strike', 'Open Interest', 'Volume', 'IV']
                        st.dataframe(display_calls, use_container_width=True)
                
                with col_right:
                    st.subheader("📉 Top Put OI")
                    if top_puts is not None:
                        display_puts = top_puts[['strike', 'openInterest', 'volume', 'impliedVolatility']].copy()
                        display_puts.columns = ['Strike', 'Open Interest', 'Volume', 'IV']
                        st.dataframe(display_puts, use_container_width=True)
                
            else:
                st.error(f"""
                ❌ No options data available for {ticker}
                
                **Possible reasons:**
                - Options not traded for this ticker
                - Ticker symbol incorrect
                - Data not available on Yahoo Finance
                
                **Try:**
                - Use tickers with active options (e.g., AAPL, TSLA, SPY)
                - Check ticker symbol format
                """)

else:
    # Landing page
    st.info("👆 Select an asset class and ticker from the sidebar, then click 'Analyze Options Chain'")
    
    st.markdown("---")
    st.subheader("📚 How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Key Indicators:**
        
        1. **Put-Call Ratio (PCR)**
           - > 1.2: Bullish (more puts, fear)
           - < 0.8: Bearish (more calls, greed)
        
        2. **Max Pain**
           - Price where most options expire worthless
           - Market tends to gravitate here
        
        3. **Open Interest Analysis**
           - High Call OI = Resistance
           - High Put OI = Support
        """)
    
    with col2:
        st.markdown("""
        **Trading Strategy:**
        
        1. **Strong Bullish Signal (Score ≥ 3)**
           - Enter long positions
           - Buy calls near support
        
        2. **Strong Bearish Signal (Score ≤ -3)**
           - Enter short positions
           - Buy puts near resistance
        
        3. **Neutral (Score -1 to 1)**
           - Range-bound strategies
           - Wait for breakout
        """)
    
    st.markdown("---")
    st.subheader("⚠️ Important Notes")
    st.warning("""
    - This tool is for educational purposes only
    - Options trading involves significant risk
    - Always do your own research
    - Use proper risk management
    - Past performance doesn't guarantee future results
    - Real-time NSE data requires authentication/subscription
    """)
    
    st.markdown("---")
    st.subheader("🔧 Data Sources")
    st.markdown("""
    - **Yahoo Finance**: US stocks, ETFs, some crypto
    - **yfinance Python library**: Free, no API key needed
    - **NSE Data**: Requires NSEpy or official API subscription
    
    For Indian markets (NIFTY, BANKNIFTY), consider:
    - NSEpy library (unofficial)
    - Official NSE API (requires registration)
    - Broker APIs (Zerodha Kite, etc.)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | Data from Yahoo Finance (yfinance)</p>
    <p><strong>Disclaimer:</strong> Not financial advice. Trade at your own risk.</p>
</div>
""", unsafe_allow_html=True)
