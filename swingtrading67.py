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

try:
    from scipy.stats import norm
    import math
    SCIPY_AVAILABLE = True
except:
    SCIPY_AVAILABLE = False
    st.warning("⚠️ scipy not installed. Greeks calculation will be limited. Install with: pip install scipy")

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
def get_yahoo_options(ticker):
    """Fetch options data from Yahoo Finance with error handling"""
    try:
        stock = yf.Ticker(ticker)
        exp_dates = stock.options
        
        if not exp_dates or len(exp_dates) == 0:
            return None, None, None, "No options data available for this ticker"
        
        # Get nearest expiry
        exp_date = exp_dates[0]
        opt_chain = stock.option_chain(exp_date)
        
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        # Validate data
        if calls.empty or puts.empty:
            return None, None, None, "Options chain is empty"
        
        return calls, puts, exp_date, None
    except Exception as e:
        return None, None, None, str(e)

def get_nse_options_web(symbol):
    """Fetch NSE options data via web scraping (for NIFTY, BANKNIFTY)"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        # Create session
        session = requests.Session()
        
        # Get cookies first
        session.get('https://www.nseindia.com', headers=headers, timeout=10)
        
        # Fetch option chain
        if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
            url = f'https://www.nseindia.com/api/option-chain-indices?symbol={symbol}'
        else:
            url = f'https://www.nseindia.com/api/option-chain-equities?symbol={symbol}'
        
        response = session.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            records = data.get('records', {}).get('data', [])
            
            if not records:
                return None, None, None, "No options data in NSE response"
            
            # Parse into calls and puts dataframes
            calls_data = []
            puts_data = []
            
            for record in records:
                strike = record.get('strikePrice', 0)
                
                # Parse Call data
                if 'CE' in record:
                    ce = record['CE']
                    calls_data.append({
                        'strike': strike,
                        'lastPrice': ce.get('lastPrice', 0),
                        'change': ce.get('change', 0),
                        'pChange': ce.get('pchangeinOpenInterest', 0),
                        'openInterest': ce.get('openInterest', 0),
                        'changeinOpenInterest': ce.get('changeinOpenInterest', 0),
                        'volume': ce.get('totalTradedVolume', 0),
                        'impliedVolatility': ce.get('impliedVolatility', 0),
                        'bidQty': ce.get('bidQty', 0),
                        'bidprice': ce.get('bidprice', 0),
                        'askPrice': ce.get('askPrice', 0),
                        'askQty': ce.get('askQty', 0),
                    })
                
                # Parse Put data
                if 'PE' in record:
                    pe = record['PE']
                    puts_data.append({
                        'strike': strike,
                        'lastPrice': pe.get('lastPrice', 0),
                        'change': pe.get('change', 0),
                        'pChange': pe.get('pchangeinOpenInterest', 0),
                        'openInterest': pe.get('openInterest', 0),
                        'changeinOpenInterest': pe.get('changeinOpenInterest', 0),
                        'volume': pe.get('totalTradedVolume', 0),
                        'impliedVolatility': pe.get('impliedVolatility', 0),
                        'bidQty': pe.get('bidQty', 0),
                        'bidprice': pe.get('bidprice', 0),
                        'askPrice': pe.get('askPrice', 0),
                        'askQty': pe.get('askQty', 0),
                    })
            
            calls_df = pd.DataFrame(calls_data)
            puts_df = pd.DataFrame(puts_data)
            
            # Get expiry date
            expiry = data.get('records', {}).get('expiryDates', [''])[0]
            
            return calls_df, puts_df, expiry, None
        else:
            return None, None, None, f"NSE API returned status code {response.status_code}"
            
    except Exception as e:
        return None, None, None, f"NSE fetch error: {str(e)}"

def calculate_greeks_approximate(option_type, S, K, T, r, sigma):
    """Approximate Black-Scholes Greeks calculation"""
    if not SCIPY_AVAILABLE:
        return 0, 0
    
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0, 0
        
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:  # put
            delta = norm.cdf(d1) - 1
        
        # Gamma is same for calls and puts
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        
        return delta, gamma
    except Exception as e:
        return 0, 0

def enrich_options_with_greeks(calls, puts, current_price, exp_date_str):
    """Add Greeks to options dataframe"""
    try:
        from datetime import datetime
        
        # Calculate time to expiry
        exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d')
        today = datetime.now()
        T = max((exp_date - today).days / 365.0, 0.001)  # Time in years
        r = 0.05  # Risk-free rate (5%)
        
        # Add Greeks to calls
        calls['delta'] = 0.0
        calls['gamma'] = 0.0
        
        for idx, row in calls.iterrows():
            sigma = row.get('impliedVolatility', 0.3)
            if sigma > 0:
                delta, gamma = calculate_greeks_approximate('call', current_price, row['strike'], T, r, sigma)
                calls.at[idx, 'delta'] = delta
                calls.at[idx, 'gamma'] = gamma
        
        # Add Greeks to puts
        puts['delta'] = 0.0
        puts['gamma'] = 0.0
        
        for idx, row in puts.iterrows():
            sigma = row.get('impliedVolatility', 0.3)
            if sigma > 0:
                delta, gamma = calculate_greeks_approximate('put', current_price, row['strike'], T, r, sigma)
                puts.at[idx, 'delta'] = delta
                puts.at[idx, 'gamma'] = gamma
        
        return calls, puts
    except:
        return calls, puts

def get_strikes_near_spot(calls, puts, current_price, num_strikes=10):
    """Get N strikes closest to current price"""
    all_strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
    
    # Find strikes around current price
    strikes_near = sorted(all_strikes, key=lambda x: abs(x - current_price))[:num_strikes * 2]
    strikes_near = sorted(strikes_near)
    
    # Filter calls and puts for these strikes
    calls_near = calls[calls['strike'].isin(strikes_near)].copy()
    puts_near = puts[puts['strike'].isin(strikes_near)].copy()
    
    return calls_near, puts_near

def get_current_price(ticker):
    """Get current price with multiple fallback methods"""
    try:
        stock = yf.Ticker(ticker)
        
        # Try different methods to get current price
        try:
            info = stock.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            if current_price and current_price > 0:
                return current_price, None
        except:
            pass
        
        # Fallback to history
        hist = stock.history(period="5d")
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            return current_price, None
        
        return None, "Unable to fetch current price"
    except Exception as e:
        return None, str(e)

def calculate_pcr(calls, puts):
    """Calculate Put-Call Ratio"""
    try:
        put_oi = puts['openInterest'].sum()
        call_oi = calls['openInterest'].sum()
        
        put_vol = puts['volume'].fillna(0).sum()
        call_vol = calls['volume'].fillna(0).sum()
        
        pcr_oi = put_oi / call_oi if call_oi > 0 else 0
        pcr_vol = put_vol / call_vol if call_vol > 0 else 0
        
        return pcr_oi, pcr_vol
    except Exception as e:
        return 0, 0

def calculate_max_pain(calls, puts):
    """Calculate Max Pain - price where most options expire worthless"""
    try:
        # Get all unique strikes
        all_strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
        
        if not all_strikes:
            return 0
        
        pain_values = []
        
        for strike in all_strikes:
            # Calculate pain for calls (ITM calls lose money for writers)
            call_pain = sum(
                (strike - s) * oi 
                for s, oi in zip(calls['strike'], calls['openInterest']) 
                if s < strike
            )
            
            # Calculate pain for puts (ITM puts lose money for writers)
            put_pain = sum(
                (s - strike) * oi 
                for s, oi in zip(puts['strike'], puts['openInterest']) 
                if s > strike
            )
            
            total_pain = call_pain + put_pain
            pain_values.append(total_pain)
        
        if pain_values:
            max_pain_strike = all_strikes[pain_values.index(min(pain_values))]
            return max_pain_strike
        return 0
    except Exception as e:
        st.error(f"Max pain calculation error: {e}")
        return 0

def analyze_oi_buildup(calls, puts, current_price):
    """Analyze Open Interest buildup for support/resistance"""
    try:
        # Sort by OI
        calls_sorted = calls.sort_values('openInterest', ascending=False).head(10)
        puts_sorted = puts.sort_values('openInterest', ascending=False).head(10)
        
        # Find strong resistance (high call OI above current price)
        resistance_levels = calls_sorted[calls_sorted['strike'] > current_price]['strike']
        resistance = resistance_levels.min() if not resistance_levels.empty else 0
        
        # Find strong support (high put OI below current price)
        support_levels = puts_sorted[puts_sorted['strike'] < current_price]['strike']
        support = support_levels.max() if not support_levels.empty else 0
        
        return support, resistance, calls_sorted.head(5), puts_sorted.head(5)
    except Exception as e:
        st.error(f"OI analysis error: {e}")
        return 0, 0, None, None

def calculate_momentum_signal(pcr_oi, pcr_vol, max_pain, current_price, support, resistance):
    """Calculate momentum signal based on multiple factors"""
    signals = []
    score = 0
    
    # PCR Analysis
    if pcr_oi > 1.2:
        signals.append("🟢 High PCR (OI): Bullish - More puts than calls (Fear indicator)")
        score += 2
    elif pcr_oi < 0.8:
        signals.append("🔴 Low PCR (OI): Bearish - More calls than puts (Greed indicator)")
        score -= 2
    else:
        signals.append(f"🟡 Neutral PCR (OI): {pcr_oi:.2f}")
    
    # Max Pain Analysis
    if max_pain > 0:
        pain_diff_pct = ((current_price - max_pain) / max_pain) * 100
        
        if current_price < max_pain * 0.98:
            signals.append(f"🟢 Price {abs(pain_diff_pct):.1f}% below Max Pain (${max_pain:.2f}): Potential upside pull")
            score += 1
        elif current_price > max_pain * 1.02:
            signals.append(f"🔴 Price {pain_diff_pct:.1f}% above Max Pain (${max_pain:.2f}): Potential downside pull")
            score -= 1
        else:
            signals.append(f"🟡 Price near Max Pain (${max_pain:.2f}): Consolidation zone")
    
    # Support/Resistance Analysis
    if support and resistance and support > 0 and resistance > 0:
        range_pct = ((resistance - support) / current_price) * 100
        signals.append(f"📊 Key Trading Range: ${support:.2f} - ${resistance:.2f} ({range_pct:.1f}% range)")
        
        # Position in range
        dist_from_support = ((current_price - support) / support) * 100
        dist_from_resistance = ((resistance - current_price) / current_price) * 100
        
        if dist_from_support <= 1.5:
            signals.append(f"🟢 Near Support ({dist_from_support:.1f}% away): Strong bounce zone")
            score += 1.5
        elif dist_from_resistance <= 1.5:
            signals.append(f"🔴 Near Resistance ({dist_from_resistance:.1f}% away): Potential rejection zone")
            score -= 1.5
        else:
            signals.append(f"🟡 Mid-range: {dist_from_support:.1f}% from support, {dist_from_resistance:.1f}% from resistance")
    
    # Volume PCR (confirmation)
    if pcr_vol > 1.5:
        signals.append("🟢 High Volume PCR: Active put buying (Bullish confirmation)")
        score += 0.5
    elif pcr_vol < 0.6:
        signals.append("🔴 Low Volume PCR: Active call buying (Bearish confirmation)")
        score -= 0.5
    
    # Final Signal
    if score >= 3:
        momentum = "🚀 STRONG BULLISH MOMENTUM"
        confidence = "Very High"
    elif score >= 1.5:
        momentum = "📈 BULLISH BIAS"
        confidence = "High"
    elif score <= -3:
        momentum = "💥 STRONG BEARISH MOMENTUM"
        confidence = "Very High"
    elif score <= -1.5:
        momentum = "📉 BEARISH BIAS"
        confidence = "High"
    else:
        momentum = "➡️ NEUTRAL / RANGE-BOUND"
        confidence = "Low"
    
    return momentum, signals, score, confidence

def plot_oi_chart(calls, puts, current_price, max_pain):
    """Create Open Interest visualization"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Open Interest Distribution', 'Volume Distribution'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # OI Chart
    fig.add_trace(
        go.Bar(x=calls['strike'], y=calls['openInterest'], 
               name='Call OI', marker_color='rgba(0, 255, 0, 0.6)'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=puts['strike'], y=puts['openInterest'], 
               name='Put OI', marker_color='rgba(255, 0, 0, 0.6)'),
        row=1, col=1
    )
    
    # Volume Chart
    fig.add_trace(
        go.Bar(x=calls['strike'], y=calls['volume'].fillna(0), 
               name='Call Vol', marker_color='rgba(144, 238, 144, 0.6)', showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=puts['strike'], y=puts['volume'].fillna(0), 
               name='Put Vol', marker_color='rgba(255, 160, 122, 0.6)', showlegend=False),
        row=2, col=1
    )
    
    # Add current price line
    fig.add_vline(x=current_price, line_dash="dash", line_color="blue", line_width=2,
                  annotation_text=f"Current: ${current_price:.2f}", annotation_position="top")
    
    # Add max pain line
    if max_pain > 0:
        fig.add_vline(x=max_pain, line_dash="dot", line_color="purple", line_width=2,
                      annotation_text=f"Max Pain: ${max_pain:.2f}", annotation_position="bottom")
    
    fig.update_layout(
        height=700, 
        showlegend=True, 
        title_text="Options Chain Analysis",
        hovermode='x unified'
    )
    fig.update_xaxes(title_text="Strike Price", row=2, col=1)
    fig.update_yaxes(title_text="Open Interest", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def plot_price_levels(ticker, support, resistance, max_pain):
    """Plot price with support/resistance levels"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        
        if hist.empty:
            return None
        
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
            fig.add_hline(y=support, line_dash="dash", line_color="green", line_width=2,
                         annotation_text=f"Support: ${support:.2f}", annotation_position="right")
        if resistance > 0:
            fig.add_hline(y=resistance, line_dash="dash", line_color="red", line_width=2,
                         annotation_text=f"Resistance: ${resistance:.2f}", annotation_position="right")
        if max_pain > 0:
            fig.add_hline(y=max_pain, line_dash="dot", line_color="purple", line_width=2,
                         annotation_text=f"Max Pain: ${max_pain:.2f}", annotation_position="right")
        
        fig.update_layout(
            title=f"{ticker} - 1 Month Price Action with Key Levels",
            yaxis_title="Price ($)",
            xaxis_title="Date",
            height=450,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    except Exception as e:
        st.error(f"Chart error: {e}")
        return None

# Ticker mapping for better compatibility
TICKER_MAP = {
    # Indian Indices - Yahoo Finance format
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "SENSEX": "^BSESN",
    "NIFTY IT": "^CNXIT",
    "NIFTY FIN": "NIFTYBEES.NS",
    
    # Crypto
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "BNB": "BNB-USD",
    "Solana": "SOL-USD",
    "XRP": "XRP-USD",
    
    # Forex
    "USD/INR": "USDINR=X",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    
    # Commodities
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Crude Oil": "CL=F",
    "Natural Gas": "NG=F"
}

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Asset Class Selection
asset_class = st.sidebar.selectbox(
    "Select Asset Class",
    ["US Stocks & ETFs (Best Options Data)", "Indian Indices", "Crypto", "Forex", "Commodities", "Custom Ticker"]
)

# Ticker input based on asset class
ticker = None

if asset_class == "US Stocks & ETFs (Best Options Data)":
    popular_stocks = {
        "S&P 500 ETF (SPY)": "SPY",
        "Nasdaq ETF (QQQ)": "QQQ",
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Tesla": "TSLA",
        "NVIDIA": "NVDA",
        "Amazon": "AMZN",
        "Google": "GOOGL",
        "Meta": "META",
        "Netflix": "NFLX"
    }
    selected = st.sidebar.selectbox("Select Stock/ETF", list(popular_stocks.keys()))
    ticker = popular_stocks[selected]
    
elif asset_class == "Indian Indices":
    st.sidebar.warning("⚠️ Indian indices have LIMITED options data on Yahoo Finance. Use US stocks for best results.")
    indices = list(TICKER_MAP.keys())[:5]
    selected = st.sidebar.selectbox("Select Index", indices)
    ticker = TICKER_MAP[selected]
    st.sidebar.info(f"Using ticker: {ticker}")
    
elif asset_class == "Crypto":
    cryptos = ["Bitcoin", "Ethereum", "BNB", "Solana", "XRP"]
    selected = st.sidebar.selectbox("Select Crypto", cryptos)
    ticker = TICKER_MAP[selected]
    st.sidebar.warning("⚠️ Limited options data for crypto. Try BTC-USD or ETH-USD")
    
elif asset_class == "Forex":
    forex = ["USD/INR", "EUR/USD", "GBP/USD"]
    selected = st.sidebar.selectbox("Select Pair", forex)
    ticker = TICKER_MAP[selected]
    st.sidebar.warning("⚠️ Very limited options data for forex pairs")
    
elif asset_class == "Commodities":
    commodities = ["Gold", "Silver", "Crude Oil", "Natural Gas"]
    selected = st.sidebar.selectbox("Select Commodity", commodities)
    ticker = TICKER_MAP[selected]
    st.sidebar.warning("⚠️ Limited options data for commodities")
    
else:  # Custom Ticker
    ticker = st.sidebar.text_input("Enter Ticker Symbol", "AAPL").upper()
    st.sidebar.info("Examples: AAPL, TSLA, SPY, MSFT")

# Analysis button
analyze_button = st.sidebar.button("🔍 Analyze Options Chain", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Quick Tips")
st.sidebar.info("""
**✅ Working Now:**
- US Stocks: AAPL, TSLA, SPY
- US ETFs: QQQ, IWM, DIA

**⚠️ Limited/Not Working:**
- Indian Indices (NSE blocking)
- Crypto (limited options)
- Forex (very limited)

**Best Results:** Use US stocks/ETFs
""")

# Main Analysis
if analyze_button and ticker:
    with st.spinner(f"Fetching options data for {ticker}..."):
        
        # Determine if we should use NSE API
        use_nse_api = use_nse if 'use_nse' in locals() else False
        
        # Get current price
        if use_nse_api:
            # For NSE, try to get spot price from Yahoo with correct ticker
            spot_ticker_map = {
                'NIFTY': '^NSEI',
                'BANKNIFTY': '^NSEBANK',
                'FINNIFTY': '^CNXFINANCE',
                'SENSEX': '^BSESN'
            }
            spot_ticker = spot_ticker_map.get(ticker, ticker)
            current_price, price_error = get_current_price(spot_ticker)
        else:
            current_price, price_error = get_current_price(ticker)
        
        if price_error or not current_price:
            st.error(f"❌ Error fetching price for {ticker}: {price_error}")
            st.info("Try US stocks (AAPL, TSLA, SPY) for best data")
        else:
            # Display current price
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Ticker", ticker)
            with col2:
                st.metric("💰 Current Price", f"${current_price:.2f}")
            
            # Fetch options data
            if use_nse_api:
                calls, puts, exp_date, opt_error = get_nse_options_web(ticker)
            else:
                calls, puts, exp_date, opt_error = get_yahoo_options(ticker)
            
            if opt_error or calls is None:
                st.error(f"❌ {opt_error or 'Unable to fetch options data'}")
                
                if use_nse_api:
                    st.warning(f"""
                    **NSE Options Data Not Available**
                    
                    The NSE website may be blocking automated requests or the API format has changed.
                    
                    **Alternative Solutions for Indian Markets:**
                    
                    1. **Use Broker APIs (Recommended):**
                       - Zerodha Kite Connect (₹2000/month)
                       - Upstox API (Free tier)
                       - Angel Broking SmartAPI
                    
                    2. **Install NSEpy Library:**
                       ```bash
                       pip install nsepy
                       ```
                       Then modify code to use nsepy for data fetching
                    
                    3. **Manual Data Entry:**
                       - Visit NSE website manually
                       - Export options chain CSV
                       - Upload to this app (feature coming soon)
                    
                    4. **Try US Markets Instead:**
                       - Select "US Stocks & ETFs" from sidebar
                       - Try: SPY, QQQ, AAPL (excellent data availability)
                    
                    **Why NSE is Difficult:**
                    - NSE blocks automated scraping
                    - Requires cookies/session management
                    - API access needs authentication
                    - Rate limiting on requests
                    """)
                else:
                    st.warning(f"""
                    **No Options Data Available for {ticker}**
                    
                    This could mean:
                    - Options are not traded for this ticker
                    - Ticker format is incorrect
                    - Data not available on Yahoo Finance
                    
                    **✅ Best Tickers for Options Analysis:**
                    
                    **US Stocks (100% Working):**
                    - **AAPL** - Apple
                    - **TSLA** - Tesla  
                    - **MSFT** - Microsoft
                    - **NVDA** - NVIDIA
                    - **AMZN** - Amazon
                    
                    **ETFs (Excellent Data):**
                    - **SPY** - S&P 500
                    - **QQQ** - Nasdaq 100
                    - **IWM** - Russell 2000
                    - **DIA** - Dow Jones
                    
                    **High Volume Options:**
                    - **AMD** - Advanced Micro Devices
                    - **PLTR** - Palantir
                    - **SOFI** - SoFi
                    - **F** - Ford
                    
                    Try selecting "US Stocks & ETFs" from the dropdown!
                    """)
            else:
                with col3:
                    st.metric("📅 Expiry Date", exp_date)
                
                st.success(f"✅ Options data loaded successfully!")
                
                # Enrich with Greeks if possible
                try:
                    calls, puts = enrich_options_with_greeks(calls, puts, current_price, exp_date)
                except:
                    pass
                
                # Calculate metrics
                pcr_oi, pcr_vol = calculate_pcr(calls, puts)
                max_pain = calculate_max_pain(calls, puts)
                support, resistance, top_calls, top_puts = analyze_oi_buildup(calls, puts, current_price)
                
                # Display key metrics
                st.markdown("---")
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    pcr_color = "🟢" if pcr_oi > 1.2 else "🔴" if pcr_oi < 0.8 else "🟡"
                    st.metric("PCR (OI)", f"{pcr_oi:.2f} {pcr_color}")
                
                with metric_cols[1]:
                    st.metric("PCR (Vol)", f"{pcr_vol:.2f}")
                
                with metric_cols[2]:
                    pain_diff = ((current_price - max_pain) / max_pain * 100) if max_pain > 0 else 0
                    st.metric("Max Pain", f"${max_pain:.2f}", f"{pain_diff:+.1f}%")
                
                with metric_cols[3]:
                    range_size = ((resistance - support) / current_price * 100) if (support and resistance) else 0
                    st.metric("Range Size", f"{range_size:.1f}%")
                
                metric_cols2 = st.columns(2)
                with metric_cols2[0]:
                    if support > 0:
                        support_dist = ((current_price - support) / support * 100)
                        st.metric("🟢 Support", f"${support:.2f}", f"{support_dist:.1f}% away")
                    else:
                        st.metric("🟢 Support", "N/A")
                
                with metric_cols2[1]:
                    if resistance > 0:
                        resist_dist = ((resistance - current_price) / current_price * 100)
                        st.metric("🔴 Resistance", f"${resistance:.2f}", f"{resist_dist:.1f}% away")
                    else:
                        st.metric("🔴 Resistance", "N/A")
                
                # Calculate momentum signal
                momentum, signals, score, confidence = calculate_momentum_signal(
                    pcr_oi, pcr_vol, max_pain, current_price, support, resistance
                )
                
                # Display momentum signal
                st.markdown("---")
                st.subheader("🎯 Momentum Signal & Analysis")
                
                signal_col1, signal_col2 = st.columns([2, 1])
                
                with signal_col1:
                    st.markdown(f"## {momentum}")
                    st.markdown(f"**Signal Strength Score: {score:.1f} | Confidence: {confidence}**")
                
                with signal_col2:
                    # Score gauge
                    if score >= 3:
                        st.success("⬆️ HIGH CONVICTION LONG")
                    elif score >= 1.5:
                        st.success("↗️ MODERATE LONG")
                    elif score <= -3:
                        st.error("⬇️ HIGH CONVICTION SHORT")
                    elif score <= -1.5:
                        st.error("↘️ MODERATE SHORT")
                    else:
                        st.info("↔️ WAIT & WATCH")
                
                st.markdown("### 📋 Signal Breakdown:")
                for signal in signals:
                    st.markdown(f"- {signal}")
                
                # Trading recommendation
                st.markdown("---")
                st.subheader("💡 Trading Strategy & Action Plan")
                
                if score >= 2:
                    st.success(f"""
                    ### 🚀 BULLISH SETUP - GO LONG
                    
                    **Entry Strategy:**
                    - 🎯 Entry Zone: ${current_price * 0.995:.2f} - ${current_price * 1.005:.2f}
                    - 📞 Call Options: Consider ATM or slightly OTM calls
                    - 📈 Stock: Buy on dips toward support at ${support:.2f}
                    
                    **Exit Targets:**
                    - 🎯 Target 1: ${resistance:.2f} ({((resistance/current_price - 1) * 100):.1f}% gain)
                    - 🎯 Target 2: ${max_pain * 1.05:.2f} (if momentum sustains)
                    - 🛑 Stop Loss: ${support * 0.98:.2f} ({((support * 0.98/current_price - 1) * 100):.1f}% risk)
                    
                    **Risk Management:**
                    - Risk/Reward: ~1:{((resistance - current_price)/(current_price - support * 0.98)):.1f}
                    - Position Size: Limit to 2-5% of portfolio
                    - Time Decay: Monitor theta if using options
                    """)
                    
                elif score <= -2:
                    st.warning(f"""
                    ### 📉 BEARISH SETUP - GO SHORT
                    
                    **Entry Strategy:**
                    - 🎯 Entry Zone: ${current_price * 0.995:.2f} - ${current_price * 1.005:.2f}
                    - 📉 Put Options: Consider ATM or slightly OTM puts
                    - 📊 Short Stock: Consider shorting near resistance at ${resistance:.2f}
                    
                    **Exit Targets:**
                    - 🎯 Target 1: ${support:.2f} ({((support/current_price - 1) * 100):.1f}% gain)
                    - 🎯 Target 2: ${max_pain * 0.95:.2f} (if momentum sustains)
                    - 🛑 Stop Loss: ${resistance * 1.02:.2f} ({((resistance * 1.02/current_price - 1) * 100):.1f}% risk)
                    
                    **Risk Management:**
                    - Risk/Reward: ~1:{abs((support - current_price)/(resistance * 1.02 - current_price)):.1f}
                    - Position Size: Limit to 2-5% of portfolio
                    - Cover rallies: Be ready to exit on strength
                    """)
                    
                else:
                    st.info(f"""
                    ### ↔️ NEUTRAL ZONE - RANGE STRATEGY
                    
                    **Current Status:**
                    - Market is in consolidation/indecision phase
                    - No clear directional bias detected
                    
                    **Recommended Strategies:**
                    - 🎯 Range Trading: Buy near ${support:.2f}, Sell near ${resistance:.2f}
                    - 🦋 Options: Consider Iron Condor or Butterfly spreads
                    - ⏳ Wait for Breakout: Monitor for clear break above ${resistance:.2f} or below ${support:.2f}
                    
                    **Watch For:**
                    - Volume spike with directional move
                    - PCR shifting above 1.2 or below 0.8
                    - Break and hold above/below key levels
                    
                    **Risk Management:**
                    - Reduce position size in uncertain markets
                    - Use tight stops if trading the range
                    - Consider staying cash until clear signal emerges
                    """)
                
                # Visualizations
                st.markdown("---")
                st.subheader("📊 Visual Analysis")
                
                # Price chart with levels
                price_chart = plot_price_levels(ticker, support, resistance, max_pain)
                if price_chart:
                    st.plotly_chart(price_chart, use_container_width=True)
                
                # OI chart
                oi_chart = plot_oi_chart(calls, puts, current_price, max_pain)
                st.plotly_chart(oi_chart, use_container_width=True)
                
                # Detailed OI Tables
                st.markdown("---")
                st.subheader("📊 DETAILED STRIKE PRICE ANALYSIS - 10 Strikes Near Spot")
                
                # Get 10 strikes near current price
                calls_near, puts_near = get_strikes_near_spot(calls, puts, current_price, 10)
                
                st.markdown(f"**Spot Price: ${current_price:.2f}**")
                
                # Create comprehensive table
                col_ce, col_pe = st.columns(2)
                
                with col_ce:
                    st.markdown("### 📞 CALL OPTIONS (CE)")
                    
                    if not calls_near.empty:
                        # Prepare display dataframe
                        ce_display = pd.DataFrame()
                        ce_display['Strike'] = calls_near['strike'].round(2)
                        ce_display['LTP'] = pd.to_numeric(calls_near.get('lastPrice', 0), errors='coerce').fillna(0).round(2)
                        ce_display['Chg'] = pd.to_numeric(calls_near.get('change', 0), errors='coerce').fillna(0).round(2)
                        ce_display['Chg%'] = pd.to_numeric(calls_near.get('pChange', 0), errors='coerce').fillna(0).round(2)
                        ce_display['OI'] = pd.to_numeric(calls_near['openInterest'], errors='coerce').fillna(0).astype(int)
                        ce_display['OI Chg'] = pd.to_numeric(calls_near.get('changeinOpenInterest', 0), errors='coerce').fillna(0).astype(int)
                        
                        # Calculate OI change percentage
                        prev_oi = ce_display['OI'] - ce_display['OI Chg']
                        prev_oi = prev_oi.replace(0, 1)  # Avoid division by zero
                        ce_display['OI Chg%'] = ((ce_display['OI Chg'] / prev_oi) * 100).round(2)
                        
                        ce_display['Vol'] = pd.to_numeric(calls_near.get('volume', 0), errors='coerce').fillna(0).astype(int)
                        
                        # Delta and Gamma
                        if 'delta' in calls_near.columns:
                            ce_display['Delta'] = pd.to_numeric(calls_near['delta'], errors='coerce').fillna(0).round(3)
                            ce_display['Gamma'] = pd.to_numeric(calls_near['gamma'], errors='coerce').fillna(0).round(4)
                        
                        # IV
                        ce_display['IV%'] = (pd.to_numeric(calls_near.get('impliedVolatility', 0), errors='coerce').fillna(0) * 100).round(2)
                        
                        # Highlight ATM
                        def highlight_atm(row):
                            try:
                                if abs(row['Strike'] - current_price) < 50:
                                    return ['background-color: #90EE90'] * len(row)
                            except:
                                pass
                            return [''] * len(row)
                        
                        styled_ce = ce_display.style.apply(highlight_atm, axis=1)
                        st.dataframe(styled_ce, use_container_width=True, hide_index=True, height=400)
                    else:
                        st.warning("No call data available")
                
                with col_pe:
                    st.markdown("### 📉 PUT OPTIONS (PE)")
                    
                    if not puts_near.empty:
                        # Prepare display dataframe
                        pe_display = pd.DataFrame()
                        pe_display['Strike'] = puts_near['strike'].round(2)
                        pe_display['LTP'] = pd.to_numeric(puts_near.get('lastPrice', 0), errors='coerce').fillna(0).round(2)
                        pe_display['Chg'] = pd.to_numeric(puts_near.get('change', 0), errors='coerce').fillna(0).round(2)
                        pe_display['Chg%'] = pd.to_numeric(puts_near.get('pChange', 0), errors='coerce').fillna(0).round(2)
                        pe_display['OI'] = pd.to_numeric(puts_near['openInterest'], errors='coerce').fillna(0).astype(int)
                        pe_display['OI Chg'] = pd.to_numeric(puts_near.get('changeinOpenInterest', 0), errors='coerce').fillna(0).astype(int)
                        
                        # Calculate OI change percentage
                        prev_oi = pe_display['OI'] - pe_display['OI Chg']
                        prev_oi = prev_oi.replace(0, 1)  # Avoid division by zero
                        pe_display['OI Chg%'] = ((pe_display['OI Chg'] / prev_oi) * 100).round(2)
                        
                        pe_display['Vol'] = pd.to_numeric(puts_near.get('volume', 0), errors='coerce').fillna(0).astype(int)
                        
                        # Delta and Gamma
                        if 'delta' in puts_near.columns:
                            pe_display['Delta'] = pd.to_numeric(puts_near['delta'], errors='coerce').fillna(0).round(3)
                            pe_display['Gamma'] = pd.to_numeric(puts_near['gamma'], errors='coerce').fillna(0).round(4)
                        
                        # IV
                        pe_display['IV%'] = (pd.to_numeric(puts_near.get('impliedVolatility', 0), errors='coerce').fillna(0) * 100).round(2)
                        
                        # Highlight ATM
                        def highlight_atm(row):
                            try:
                                if abs(row['Strike'] - current_price) < 50:
                                    return ['background-color: #FFB6C6'] * len(row)
                            except:
                                pass
                            return [''] * len(row)
                        
                        styled_pe = pe_display.style.apply(highlight_atm, axis=1)
                        st.dataframe(styled_pe, use_container_width=True, hide_index=True, height=400)
                    else:
                        st.warning("No put data available")
                
                # Legend
                st.markdown("""
                **Legend:**
                - **LTP:** Last Traded Price | **Chg:** Price Change | **Chg%:** Price Change %
                - **OI:** Open Interest | **OI Chg:** Change in OI | **OI Chg%:** OI Change %
                - **Vol:** Volume | **Delta:** Rate of change | **Gamma:** Delta acceleration
                - **IV:** Implied Volatility (%)
                - 🟢 **Green Background:** Near ATM (money) strikes
                """)
                
                st.markdown("---")
                st.subheader("📋 Top 5 by Open Interest")
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("### 📞 Top 5 Call Open Interest")
                    if top_calls is not None and not top_calls.empty:
                        display_calls = top_calls[['strike', 'openInterest', 'volume', 'impliedVolatility']].copy()
                        display_calls.columns = ['Strike', 'Open Interest', 'Volume', 'IV (%)']
                        display_calls['IV (%)'] = (display_calls['IV (%)'] * 100).round(2)
                        st.dataframe(display_calls, use_container_width=True, hide_index=True)
                
                with col_right:
                    st.markdown("### 📉 Top 5 Put Open Interest")
                    if top_puts is not None and not top_puts.empty:
                        display_puts = top_puts[['strike', 'openInterest', 'volume', 'impliedVolatility']].copy()
                        display_puts.columns = ['Strike', 'Open Interest', 'Volume', 'IV (%)']
                        display_puts['IV (%)'] = (display_puts['IV (%)'] * 100).round(2)
                        st.dataframe(display_puts, use_container_width=True, hide_index=True)
                
                # STRIKE PRICE RECOMMENDATIONS
                st.markdown("---")
                st.subheader("🎯 STRIKE PRICE RECOMMENDATIONS")
                
                # Calculate recommended strikes
                if score >= 2:  # Bullish
                    # Call recommendations
                    atm_call = round(current_price / 5) * 5  # Round to nearest 5
                    otm_call_1 = atm_call + 5
                    otm_call_2 = atm_call + 10
                    
                    # Calculate targets and stops
                    call_target_1 = resistance if resistance > 0 else current_price * 1.03
                    call_target_2 = call_target_1 * 1.05
                    call_stop = support if support > 0 else current_price * 0.97
                    
                    risk_pct = ((current_price - call_stop) / current_price) * 100
                    reward_pct = ((call_target_1 - current_price) / current_price) * 100
                    rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
                    
                    st.success("### 🟢 CALL (CE) BUY RECOMMENDATIONS")
                    
                    rec_col1, rec_col2, rec_col3 = st.columns(3)
                    
                    with rec_col1:
                        st.markdown(f"""
                        **🎯 CONSERVATIVE (ATM)**
                        - **Strike:** ${atm_call:.0f} CE
                        - **Entry:** Current price zone
                        - **Target 1:** ${call_target_1:.2f} (+{reward_pct:.1f}%)
                        - **Target 2:** ${call_target_2:.2f} (+{((call_target_2/current_price - 1)*100):.1f}%)
                        - **Stop Loss:** ${call_stop:.2f} (-{risk_pct:.1f}%)
                        - **Risk/Reward:** 1:{rr_ratio:.1f}
                        
                        **Logic:**
                        - ATM = Highest delta, moves with stock
                        - Best for directional conviction
                        - Lower risk, moderate reward
                        """)
                    
                    with rec_col2:
                        otm_reward_pct = ((call_target_1 - current_price) / current_price) * 150  # Higher leverage
                        st.markdown(f"""
                        **⚡ AGGRESSIVE (OTM)**
                        - **Strike:** ${otm_call_1:.0f} CE
                        - **Entry:** On dips/pullbacks
                        - **Target 1:** ${call_target_1:.2f} (~{otm_reward_pct:.0f}% option gain)
                        - **Target 2:** ${call_target_2:.2f}
                        - **Stop Loss:** ${call_stop:.2f} or 30% of premium
                        - **Risk/Reward:** High risk, high reward
                        
                        **Logic:**
                        - OTM = Higher gamma, explosive moves
                        - Cheaper premium, higher % gains
                        - Best when expecting big move
                        """)
                    
                    with rec_col3:
                        deep_otm_reward = ((call_target_2 - current_price) / current_price) * 200
                        st.markdown(f"""
                        **🚀 LOTTERY (Deep OTM)**
                        - **Strike:** ${otm_call_2:.0f} CE
                        - **Entry:** Small position only
                        - **Target:** ${call_target_2:.2f} (~{deep_otm_reward:.0f}% option gain)
                        - **Stop Loss:** 50% of premium or worthless
                        - **Position Size:** Max 1% of capital
                        
                        **Logic:**
                        - Deep OTM = Very cheap, lottery ticket
                        - Only for strong momentum
                        - High probability of 100% loss
                        - Can return 5-10x if hits
                        """)
                    
                    st.markdown("---")
                    st.info(f"""
                    **📊 TRADE RATIONALE:**
                    - **Signal Score:** {score:.1f} (Strong Bullish)
                    - **PCR:** {pcr_oi:.2f} (High put activity = contrarian bullish)
                    - **Max Pain:** ${max_pain:.2f} (Price likely to rise toward/above it)
                    - **Support:** ${support:.2f} (Strong put OI acting as floor)
                    - **Resistance:** ${resistance:.2f} (Initial target zone)
                    - **Strategy:** Buy calls on dips near support, target resistance breakout
                    """)
                
                elif score <= -2:  # Bearish
                    # Put recommendations
                    atm_put = round(current_price / 5) * 5
                    otm_put_1 = atm_put - 5
                    otm_put_2 = atm_put - 10
                    
                    # Calculate targets and stops
                    put_target_1 = support if support > 0 else current_price * 0.97
                    put_target_2 = put_target_1 * 0.95
                    put_stop = resistance if resistance > 0 else current_price * 1.03
                    
                    risk_pct = ((put_stop - current_price) / current_price) * 100
                    reward_pct = ((current_price - put_target_1) / current_price) * 100
                    rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
                    
                    st.error("### 🔴 PUT (PE) BUY RECOMMENDATIONS")
                    
                    rec_col1, rec_col2, rec_col3 = st.columns(3)
                    
                    with rec_col1:
                        st.markdown(f"""
                        **🎯 CONSERVATIVE (ATM)**
                        - **Strike:** ${atm_put:.0f} PE
                        - **Entry:** Current price zone
                        - **Target 1:** ${put_target_1:.2f} (-{reward_pct:.1f}%)
                        - **Target 2:** ${put_target_2:.2f} (-{((1 - put_target_2/current_price)*100):.1f}%)
                        - **Stop Loss:** ${put_stop:.2f} (+{risk_pct:.1f}%)
                        - **Risk/Reward:** 1:{rr_ratio:.1f}
                        
                        **Logic:**
                        - ATM put = Best delta for downside
                        - Moves 1:1 with stock decline
                        - Lower risk, steady gains
                        """)
                    
                    with rec_col2:
                        otm_reward_pct = ((current_price - put_target_1) / current_price) * 150
                        st.markdown(f"""
                        **⚡ AGGRESSIVE (OTM)**
                        - **Strike:** ${otm_put_1:.0f} PE
                        - **Entry:** On rallies/bounces
                        - **Target 1:** ${put_target_1:.2f} (~{otm_reward_pct:.0f}% option gain)
                        - **Target 2:** ${put_target_2:.2f}
                        - **Stop Loss:** ${put_stop:.2f} or 30% of premium
                        - **Risk/Reward:** High risk, high reward
                        
                        **Logic:**
                        - OTM put = Cheaper, higher leverage
                        - Best for sharp declines
                        - Can double/triple quickly
                        """)
                    
                    with rec_col3:
                        deep_otm_reward = ((current_price - put_target_2) / current_price) * 200
                        st.markdown(f"""
                        **💥 LOTTERY (Deep OTM)**
                        - **Strike:** ${otm_put_2:.0f} PE
                        - **Entry:** Small position only
                        - **Target:** ${put_target_2:.2f} (~{deep_otm_reward:.0f}% option gain)
                        - **Stop Loss:** 50% of premium or worthless
                        - **Position Size:** Max 1% of capital
                        
                        **Logic:**
                        - Deep OTM = Crash protection
                        - Cheap lottery ticket
                        - Only for major breakdown
                        - 10x+ potential if crashes
                        """)
                    
                    st.markdown("---")
                    st.info(f"""
                    **📊 TRADE RATIONALE:**
                    - **Signal Score:** {score:.1f} (Strong Bearish)
                    - **PCR:** {pcr_oi:.2f} (Low PCR = too many calls = contrarian bearish)
                    - **Max Pain:** ${max_pain:.2f} (Price likely to fall toward it)
                    - **Resistance:** ${resistance:.2f} (Heavy call OI acting as ceiling)
                    - **Support:** ${support:.2f} (Initial downside target)
                    - **Strategy:** Buy puts on rallies near resistance, target support breakdown
                    """)
                
                else:  # Neutral
                    st.warning("### ↔️ NEUTRAL ZONE - NO DIRECTIONAL TRADE")
                    
                    st.markdown("""
                    **Current Market Status:**
                    - No clear bullish or bearish signal
                    - Range-bound price action expected
                    - Low conviction for directional trades
                    
                    **Recommended Strategies:**
                    """)
                    
                    strat_col1, strat_col2 = st.columns(2)
                    
                    with strat_col1:
                        iron_condor_call_sell = round((current_price * 1.02) / 5) * 5
                        iron_condor_call_buy = iron_condor_call_sell + 5
                        iron_condor_put_sell = round((current_price * 0.98) / 5) * 5
                        iron_condor_put_buy = iron_condor_put_sell - 5
                        
                        st.markdown(f"""
                        **🦋 IRON CONDOR (Neutral Strategy)**
                        - **Sell Call:** ${iron_condor_call_sell:.0f} CE
                        - **Buy Call:** ${iron_condor_call_buy:.0f} CE
                        - **Sell Put:** ${iron_condor_put_sell:.0f} PE
                        - **Buy Put:** ${iron_condor_put_buy:.0f} PE
                        - **Max Profit:** Premium collected
                        - **Max Loss:** Width of spread - premium
                        - **Best For:** Low volatility, range-bound
                        
                        **Logic:**
                        - Profit if price stays between sold strikes
                        - Time decay works in your favor
                        - Defined risk, defined reward
                        """)
                    
                    with strat_col2:
                        st.markdown(f"""
                        **⏳ WAIT FOR SETUP**
                        
                        **Watch For:**
                        - PCR moving above 1.2 or below 0.8
                        - Volume spike with direction
                        - Break above ${resistance:.2f} (bullish)
                        - Break below ${support:.2f} (bearish)
                        - Signal score reaching ±2 or more
                        
                        **Action Plan:**
                        - Stay in cash for now
                        - Set price alerts at key levels
                        - Re-analyze when signal improves
                        - Don't force trades in unclear markets
                        
                        **Remember:** No trade is better than a bad trade
                        """)
                    
                    st.markdown("---")
                    st.info(f"""
                    **📊 CURRENT STATUS:**
                    - **Signal Score:** {score:.1f} (Neutral/Indecisive)
                    - **PCR:** {pcr_oi:.2f} (Neither fear nor greed extreme)
                    - **Price Position:** Between support (${support:.2f}) and resistance (${resistance:.2f})
                    - **Recommendation:** Wait for clearer signal or use range-bound strategies
                    """)

else:
    # Landing page
    st.info("👆 **Get Started:** Select an asset class and ticker from the sidebar, then click 'Analyze Options Chain'")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 How It Works")
        st.markdown("---")
    
    st.markdown("---")
    st.subheader("🔧 Data Sources & Setup")
    
    tab1, tab2, tab3 = st.tabs(["📡 Data Sources", "💻 Installation", "🇮🇳 Indian Markets"])
    
    with tab1:
        st.markdown("""
        ### Free Data Sources Used:
        
        **Yahoo Finance (yfinance library)**
        - ✅ Free, no API key needed
        - ✅ US stocks, ETFs, major indices
        - ✅ Options chain data with OI, volume, IV
        - ⚠️ 15-minute delay on real-time data
        - ⚠️ Limited coverage for Indian indices
        
        **Best Coverage:**
        - 🇺🇸 US Stocks: AAPL, TSLA, MSFT, NVDA, AMD
        - 📊 ETFs: SPY, QQQ, IWM, DIA, VIX
        - 💹 High Volume: PLTR, SOFI, F, AAL
        
        **Limited Coverage:**
        - 🇮🇳 Indian Indices (use alternatives below)
        - 💰 Crypto (BTC-USD, ETH-USD only)
        - 🌍 Forex pairs (very limited)
        - 📦 Commodities futures
        """)
    
    with tab2:
        st.markdown("""
        ### 🐍 Python Installation:
        
        ```bash
        # Install required packages
        pip install streamlit pandas numpy yfinance plotly requests
        
        # Run the app
        streamlit run app.py
        ```
        
        ### 📦 Requirements.txt:
        ```
        streamlit>=1.28.0
        pandas>=2.0.0
        numpy>=1.24.0
        yfinance>=0.2.28
        plotly>=5.17.0
        requests>=2.31.0
        ```
        
        ### 🚀 Quick Start:
        ```python
        # Save the code as app.py
        # Open terminal/command prompt
        # Navigate to the folder
        # Run: streamlit run app.py
        # App opens in browser at localhost:8501
        ```
        """)
    
    with tab3:
        st.markdown("""
        ### 🇮🇳 For Indian Markets (NSE/BSE):
        
        **Option 1: NSEpy (Unofficial Library)**
        ```bash
        pip install nsepy
        ```
        
        ```python
        from nsepy import get_history
        from nsepy.derivatives import get_expiry_date
        from datetime import date
        
        # Get NIFTY options data
        expiry = get_expiry_date(year=2026, month=1)
        nifty_options = get_history(
            symbol="NIFTY",
            start=date(2026,1,1),
            end=date(2026,1,22),
            option_type="CE",  # Call
            strike_price=23000,
            expiry_date=expiry
        )
        ```
        
        **Option 2: Official NSE API**
        - Requires registration at nseindia.com
        - Need to handle cookies/headers
        - More reliable but complex setup
        
        **Option 3: Broker APIs (Recommended)**
        
        **Zerodha Kite Connect:**
        ```bash
        pip install kiteconnect
        ```
        - Paid API (₹2000/month)
        - Real-time data, order placement
        - Best for serious traders
        
        **Upstox API:**
        - Free tier available
        - Good documentation
        - Real-time market data
        
        **Angel Broking SmartAPI:**
        - Free for clients
        - Historical + real-time data
        - WebSocket support
        
        **Sample NSE Integration:**
        ```python
        # Using NSE official website scraping
        import requests
        
        def get_nse_option_chain(symbol="NIFTY"):
            url = "https://www.nseindia.com/api/option-chain-indices"
            params = {"symbol": symbol}
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json"
            }
            
            session = requests.Session()
            session.get("https://www.nseindia.com", headers=headers)
            response = session.get(url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                return data['records']['data']
            return None
        ```
        
        **Note:** NSE blocks automated requests. Use broker APIs for production.
        """)
    
    st.markdown("---")
    st.subheader("🎓 Advanced Features (Coming Soon)")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
        ### 📈 Technical Enhancements:
        - [ ] Greeks calculation (Delta, Gamma, Theta, Vega)
        - [ ] IV Rank & IV Percentile
        - [ ] Historical volatility comparison
        - [ ] Volume profile analysis
        - [ ] Unusual options activity alerts
        - [ ] Multi-timeframe analysis
        - [ ] Backtesting engine
        """)
    
    with feature_col2:
        st.markdown("""
        ### 🔔 Smart Features:
        - [ ] Real-time alerts via email/SMS
        - [ ] Portfolio tracker integration
        - [ ] Risk calculator
        - [ ] Strategy builder (spreads, straddles)
        - [ ] Earnings calendar integration
        - [ ] Market sentiment dashboard
        - [ ] AI-powered pattern recognition
        """)
    
    st.markdown("---")
    st.subheader("💡 Pro Trading Tips")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
        ### ✅ DO's:
        
        - ✅ **Trade liquid options** (tight bid-ask spreads)
        - ✅ **Check IV before buying** (avoid high IV)
        - ✅ **Use stop losses** (mental or hard stops)
        - ✅ **Size positions properly** (2-5% max)
        - ✅ **Combine with technicals** (RSI, MACD, support/resistance)
        - ✅ **Paper trade first** (practice without risk)
        - ✅ **Keep a trading journal** (learn from mistakes)
        - ✅ **Respect earnings dates** (avoid pre-earnings if new)
        - ✅ **Monitor theta decay** (time is money in options)
        - ✅ **Take profits** (greed kills accounts)
        """)
    
    with tips_col2:
        st.markdown("""
        ### ❌ DON'Ts:
        
        - ❌ **Don't trade illiquid options** (wide spreads)
        - ❌ **Don't ignore IV** (buying high IV = expensive)
        - ❌ **Don't trade without stops** (recipe for disaster)
        - ❌ **Don't overtrade** (quality > quantity)
        - ❌ **Don't revenge trade** (emotions kill)
        - ❌ **Don't ignore max pain** (especially near expiry)
        - ❌ **Don't hold to expiry** (unless planned)
        - ❌ **Don't bet the farm** (position sizing matters)
        - ❌ **Don't fight the trend** (trend is your friend)
        - ❌ **Don't ignore news** (catalysts matter)
        """)
    
    st.markdown("---")
    st.subheader("📊 Real Example Walkthrough")
    
    with st.expander("🔍 Click to see AAPL example analysis"):
        st.markdown("""
        ### Apple (AAPL) Options Analysis Example
        
        **Scenario:** AAPL trading at $185.00
        
        **Options Chain Data:**
        - PCR (OI): 1.35 → 🟢 Bullish (more puts than calls)
        - Max Pain: $182.50 → 🟢 Current price above max pain
        - Support: $180.00 (high put OI)
        - Resistance: $190.00 (high call OI)
        - PCR (Vol): 1.45 → 🟢 Active put buying
        
        **Signal Breakdown:**
        1. High PCR = +2 points (bullish)
        2. Above max pain = +1 point (upward pull)
        3. Distance from resistance = 0 points (mid-range)
        4. High volume PCR = +0.5 points
        
        **Total Score: +3.5 → 🚀 STRONG BULLISH**
        
        **Trade Setup:**
        - **Strategy:** Buy ATM calls ($185 strike)
        - **Entry:** $185.00 ± $1.00
        - **Target 1:** $190.00 (resistance) = 2.7% gain
        - **Target 2:** $195.00 (next level) = 5.4% gain
        - **Stop Loss:** $180.00 (support) = 2.7% risk
        - **Risk/Reward:** 1:2 (good setup)
        
        **Options Strategy:**
        - Buy $185 Call expiring in 2-3 weeks
        - Or buy $190 Call (cheaper, higher risk)
        - Or sell $180 Put (if bullish + want premium)
        
        **Risk Management:**
        - Position size: 3% of portfolio
        - Exit 50% at Target 1, hold for Target 2
        - Move stop to breakeven after +2%
        - Close position if PCR drops below 1.0
        """)
    
    st.markdown("---")
    st.subheader("🔗 Useful Resources")
    
    st.markdown("""
    ### 📚 Learning Resources:
    - [Options Playbook](https://www.optionsplaybook.com/) - Strategy guides
    - [CBOE Education](https://www.cboe.com/education/) - Options basics
    - [Investopedia Options](https://www.investopedia.com/options-basics-tutorial-4583012) - Comprehensive guide
    - [TastyTrade](https://www.tastytrade.com/learn) - Video tutorials
    
    ### 🛠️ Tools & Platforms:
    - [TradingView](https://www.tradingview.com/) - Charting platform
    - [OptionStrat](https://optionstrat.com/) - Strategy visualizer
    - [Market Chameleon](https://marketchameleon.com/) - Options flow
    - [Barchart](https://www.barchart.com/options) - Unusual activity
    
    ### 📰 Market Data:
    - [Yahoo Finance](https://finance.yahoo.com/) - Free real-time quotes
    - [NSE India](https://www.nseindia.com/) - Indian market data
    - [Investing.com](https://www.investing.com/) - Global markets
    - [Finviz](https://finviz.com/) - Stock screener
    
    ### 🤖 APIs for Developers:
    - [Alpha Vantage](https://www.alphavantage.co/) - Free stock API
    - [Polygon.io](https://polygon.io/) - Market data API
    - [IEX Cloud](https://iexcloud.io/) - Financial data API
    - [Kite Connect](https://kite.trade/) - Zerodha API (India)
    """)
    
    st.markdown("---")
    st.subheader("❓ FAQ - Frequently Asked Questions")
    
    with st.expander("Q1: Why can't I see data for NIFTY or BANKNIFTY?"):
        st.markdown("""
        Yahoo Finance uses different ticker formats for Indian indices:
        - **NIFTY 50:** Use ^NSEI (limited options data)
        - **NIFTY BANK:** Use ^NSEBANK (limited options data)
        - **SENSEX:** Use ^BSESN (limited options data)
        
        **Better Solution:** Use NSE official API or broker APIs (Zerodha, Upstox) for real NSE options data.
        
        **Best Alternative:** Focus on US stocks (SPY, QQQ, AAPL) which have excellent options data coverage.
        """)
    
    with st.expander("Q2: What's the difference between PCR OI and PCR Volume?"):
        st.markdown("""
        **PCR (Open Interest):**
        - Shows accumulated positions over time
        - More reliable for longer-term sentiment
        - Changes slowly (established positions)
        
        **PCR (Volume):**
        - Shows today's trading activity
        - Better for short-term sentiment
        - Changes quickly (new positions)
        
        **Best Practice:** Use both together. If both are high/low, it's a stronger signal.
        """)
    
    with st.expander("Q3: When is the best time to use this tool?"):
        st.markdown("""
        **Best Times:**
        - 📅 **Weekly Options Expiry** (Thursday/Friday) - Max pain more relevant
        - 🕐 **Market Open** (9:30 AM ET) - Fresh options activity
        - 📊 **Before Major Events** - Earnings, Fed meetings, economic data
        - 🎯 **At Key Support/Resistance** - Confirm breakout/breakdown
        
        **Avoid:**
        - ❌ During low volume periods (lunch hour, holidays)
        - ❌ Far from expiry (max pain less relevant)
        - ❌ On illiquid stocks (unreliable OI data)
        """)
    
    with st.expander("Q4: How accurate is the Max Pain theory?"):
        st.markdown("""
        **Accuracy depends on:**
        - ✅ **High accuracy:** Weekly options, day before expiry, liquid stocks
        - ⚠️ **Medium accuracy:** Monthly options, 1 week to expiry
        - ❌ **Low accuracy:** Quarterly options, >3 weeks to expiry
        
        **Important:** Max pain is ONE indicator. Always combine with:
        - Price action & technical analysis
        - Volume confirmation
        - Market sentiment
        - News/catalysts
        
        **Reality:** Markets don't always go to max pain, but awareness of it helps understand potential price magnets.
        """)
    
    with st.expander("Q5: Can I use this for day trading?"):
        st.markdown("""
        **Yes, but with caveats:**
        
        **Good for:**
        - 🎯 Identifying intraday support/resistance
        - 📊 Confirming trend direction
        - ⚡ Spotting unusual options activity
        - 🎲 Quick scalps near key levels
        
        **Not ideal for:**
        - ❌ Pure scalping (too slow to update)
        - ❌ News-driven volatility (fundamentals override)
        - ❌ Low float stocks (options data unreliable)
        
        **Best Use:** Swing trading (2-7 days) where options dynamics have more time to play out.
        """)
    
    with st.expander("Q6: What's a good PCR value?"):
        st.markdown("""
        **PCR Interpretation (as contrarian indicator):**
        
        **Bullish Signals:**
        - PCR > 1.3 = Extreme fear → Potential bounce
        - PCR 1.1-1.3 = Moderate fear → Cautiously bullish
        
        **Neutral:**
        - PCR 0.8-1.1 = Balanced → No clear signal
        
        **Bearish Signals:**
        - PCR 0.6-0.8 = Moderate greed → Cautiously bearish
        - PCR < 0.6 = Extreme greed → Potential reversal down
        
        **Context Matters:** 
        - Compare to historical PCR for that ticker
        - Consider overall market conditions
        - Look for extremes, not absolute values
        """)
    
    st.markdown("---")

# Footer
st.markdown("---")
