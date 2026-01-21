import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="Zero-Hero Algo Trader Pro", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .metric-container { background-color: #121212; padding: 10px; border-radius: 5px; border: 1px solid #444; }
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
    .buy-signal { color: #00ff00; font-weight: bold; font-size: 20px; animation: blinker 1s linear infinite; }
    @keyframes blinker { 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DHAN API PLACEHOLDER (INTEGRATION READY)
# ==========================================
class OrderManager:
    """
    Placeholder class for Dhan API integration.
    Uncomment and fill credentials when ready.
    """
    def __init__(self):
        # self.client_id = "YOUR_CLIENT_ID"
        # self.access_token = "YOUR_ACCESS_TOKEN"
        # self.dhan = dhanhq(self.client_id, self.access_token)
        pass

    def place_buy_order(self, symbol, quantity, price):
        st.toast(f"🚀 SIGNAL: Placing BUY Order for {symbol} at {price}", icon="🟢")
        # try:
        #     order = self.dhan.place_order(
        #         security_id=symbol,  # You need to map Symbol to Security ID
        #         exchange_segment=dhan.NSE_FNO,
        #         transaction_type=dhan.BUY,
        #         quantity=quantity,
        #         order_type=dhan.MARKET,
        #         product_type=dhan.INTRADAY,
        #         price=0
        #     )
        #     return order
        # except Exception as e:
        #     st.error(f"Dhan Order Failed: {e}")

    def place_sell_order(self, symbol, quantity):
        st.toast(f"🛑 SIGNAL: Placing SELL Order for {symbol}", icon="🔴")
        # Implementation similar to buy

order_manager = OrderManager()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
@st.cache_data(ttl=60)
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # fast_info is efficient for single price checks
        price = stock.fast_info['last_price']
        return price
    except:
        return 0

def get_option_chain_data(ticker, expiry):
    stock = yf.Ticker(ticker)
    opt = stock.option_chain(expiry)
    return opt.calls, opt.puts

def calculate_levels(entry_price):
    """Calculates Zero-Hero Risk Management Levels"""
    sl = entry_price * 0.70  # 30% Stop Loss (High risk for Zero Hero)
    target1 = entry_price * 1.50 # 50% Gain
    target2 = entry_price * 2.00 # 100% Gain (The "Hero" move)
    return sl, target1, target2

# ==========================================
# 4. MAIN APPLICATION
# ==========================================
st.title("⚡ Zero-Hero Algo Trader Pro")

# Create Tabs
tab_analysis, tab_live = st.tabs(["🔍 Market Scanner", "🤖 Live Trading Bot"])

# -----------------------------------------------------------------------------
# TAB 1: MARKET SCANNER (Logic from previous request)
# -----------------------------------------------------------------------------
with tab_analysis:
    st.header("Option Chain Scanner")
    col1, col2, col3 = st.columns(3)
    ticker_input = col1.text_input("Ticker Symbol", value="SPY", help="e.g., SPY, AAPL, BTC-USD").upper()
    budget = col2.number_input("Max Budget ($)", value=2.0)
    vol_min = col3.number_input("Min Volume", value=500)

    if st.button("Scan Markets"):
        try:
            stock = yf.Ticker(ticker_input)
            spot_price = stock.fast_info['last_price']
            st.metric("Spot Price", f"{spot_price:.2f}")

            # Get Expirations
            exps = stock.options
            if exps:
                # Default to nearest expiry
                calls, puts = get_option_chain_data(ticker_input, exps[0])
                
                # Zero Hero Logic: Cheap + High Volume
                # 1. Filter Calls
                hero_calls = calls[(calls['lastPrice'] <= budget) & (calls['volume'] > vol_min)].copy()
                hero_calls['HeroScore'] = hero_calls['volume'] / (hero_calls['openInterest'] + 1)
                hero_calls = hero_calls.sort_values(by='HeroScore', ascending=False)
                
                # 2. Filter Puts
                hero_puts = puts[(puts['lastPrice'] <= budget) & (puts['volume'] > vol_min)].copy()
                hero_puts['HeroScore'] = hero_puts['volume'] / (hero_puts['openInterest'] + 1)
                hero_puts = hero_puts.sort_values(by='HeroScore', ascending=False)

                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("🚀 Call Candidates")
                    st.dataframe(hero_calls[['contractSymbol', 'strike', 'lastPrice', 'volume', 'HeroScore']].head(5), hide_index=True)
                with c2:
                    st.subheader("🐻 Put Candidates")
                    st.dataframe(hero_puts[['contractSymbol', 'strike', 'lastPrice', 'volume', 'HeroScore']].head(5), hide_index=True)
                
                st.info("Copy the 'contractSymbol' of your chosen option to the Live Trading tab.")
            else:
                st.error("No options data found.")
        except Exception as e:
            st.error(f"Error scanning: {e}")

# -----------------------------------------------------------------------------
# TAB 2: LIVE TRADING BOT (New Feature)
# -----------------------------------------------------------------------------
with tab_live:
    st.header("🤖 Live Zero-Hero Monitor")
    st.caption("Real-time tracking with Rate Limits & Auto-Trade Execution")

    # Inputs
    col_l1, col_l2, col_l3 = st.columns(3)
    contract_symbol = col_l1.text_input("Contract Symbol", value="", placeholder="e.g. SPY231117C00440000")
    refresh_rate = col_l2.slider("Refresh Rate (Seconds)", 1.0, 5.0, 1.5)
    auto_trade = col_l3.checkbox("Enable Auto-Orders (Dhan)", value=False)

    # Session State for tracking trade status
    if 'in_trade' not in st.session_state:
        st.session_state.in_trade = False
        st.session_state.entry_price = 0.0
        st.session_state.sl = 0.0
        st.session_state.target = 0.0

    # Placeholders for live data updates
    status_ph = st.empty()
    metric_ph = st.empty()
    chart_ph = st.empty()
    log_ph = st.empty()

    # Start Button
    start_btn = st.button("🔴 START LIVE TRACKING")
    
    if start_btn and contract_symbol:
        price_history = []
        
        # SIMULATED LIVE LOOP
        # Note: In a real server, this would be a background thread. 
        # In Streamlit, this loops until user stops or error.
        st.toast("System Initialized. Fetching Data...")
        
        while True:
            try:
                # 1. Fetch Data with Rate Limit
                # We use yfinance Ticker for the specific OPTION symbol
                opt = yf.Ticker(contract_symbol)
                
                # Fetch live price (fast_info is best for this)
                current_price = opt.fast_info['last_price']
                prev_close = opt.fast_info['previous_close']
                
                # If data is missing (common in yfinance for illiquid options), skip
                if current_price is None:
                    time.sleep(refresh_rate)
                    continue

                price_history.append(current_price)
                if len(price_history) > 50: price_history.pop(0)

                # 2. Logic: Momentum Trigger
                # Simple Logic: If Price > Previous Close + 10% AND not already in trade
                pct_change = ((current_price - prev_close) / prev_close) * 100
                
                # 3. Update UI
                with metric_ph.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Live Premium", f"${current_price:.2f}")
                    m2.metric("Change", f"{pct_change:.2f}%", delta_color="normal")
                    m3.metric("Status", "IN TRADE" if st.session_state.in_trade else "SCANNING")

                # 4. Charting
                fig = go.Figure(data=go.Scatter(y=price_history, mode='lines+markers', line=dict(color='#00ffca')))
                fig.update_layout(title="Live Premium Momentum", height=300, margin=dict(l=0, r=0, t=30, b=0))
                chart_ph.plotly_chart(fig, use_container_width=True)

                # 5. TRADE EXECUTION LOGIC
                # ENTRY CONDITION: Premium jumps > 15% and we are not in trade
                if not st.session_state.in_trade and pct_change > 15.0:
                    st.session_state.in_trade = True
                    st.session_state.entry_price = current_price
                    st.session_state.sl, t1, st.session_state.target = calculate_levels(current_price)
                    
                    msg = f"⚔️ ENTRY TRIGGERED @ {current_price} | SL: {st.session_state.sl:.2f} | TGT: {st.session_state.target:.2f}"
                    log_ph.warning(msg)
                    
                    # ----------------------
                    # DHAN API: BUY ORDER
                    # ----------------------
                    if auto_trade:
                        order_manager.place_buy_order(contract_symbol, 25, current_price) # Qty 25 example

                # EXIT CONDITIONS
                if st.session_state.in_trade:
                    # Check Stop Loss
                    if current_price <= st.session_state.sl:
                        msg = f"🛑 STOP LOSS HIT @ {current_price}. Exiting..."
                        st.session_state.in_trade = False
                        log_ph.error(msg)
                        if auto_trade: order_manager.place_sell_order(contract_symbol, 25)

                    # Check Target
                    elif current_price >= st.session_state.target:
                        msg = f"💰 TARGET HIT @ {current_price}. Profit Booked!"
                        st.session_state.in_trade = False
                        log_ph.success(msg)
                        if auto_trade: order_manager.place_sell_order(contract_symbol, 25)
                        
                    # Trailing SL Logic (Simple)
                    # If price moves 10% above entry, move SL to Break Even
                    elif current_price > st.session_state.entry_price * 1.10 and st.session_state.sl < st.session_state.entry_price:
                        st.session_state.sl = st.session_state.entry_price
                        st.toast("Trailing SL moved to Break Even")

                # 6. Rate Limit Sleep
                time.sleep(refresh_rate)

            except Exception as e:
                # Handle API crashes gracefully
                status_ph.error(f"API Limit/Error: {e}. Retrying...")
                time.sleep(5) # Longer wait on error
