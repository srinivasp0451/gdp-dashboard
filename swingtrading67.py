import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta

# ==========================================
# 0. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="Zero-Hero Algo Pro", layout="wide", page_icon="📈")

# Custom CSS for Trading Terminal Look
st.markdown("""
<style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .success { color: #00ff00; }
    .fail { color: #ff0000; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. DHAN API PLACEHOLDER (INTEGRATION READY)
# ==========================================
class DhanExecution:
    """
    Dhan API Wrapper for Order Execution.
    Uncomment 'self.dhan' lines when you have API Keys.
    """
    def __init__(self, client_id="", access_token=""):
        self.client_id = client_id
        # from dhanhq import dhanhq
        # self.dhan = dhanhq(client_id, access_token)
        self.is_connected = False # Set True if connection succeeds

    def place_order(self, symbol, transaction_type, quantity, price=0):
        # transaction_type: "BUY" or "SELL"
        st.toast(f"🚀 DHAN API: Sending {transaction_type} Order for {symbol} Qty: {quantity}", icon="⚡")
        
        # Real Implementation:
        # try:
        #     order = self.dhan.place_order(
        #         security_id=symbol, # You must map Symbol -> SecurityID (e.g. 45000)
        #         exchange_segment=self.dhan.NSE_FNO,
        #         transaction_type=self.dhan.BUY if transaction_type == "BUY" else self.dhan.SELL,
        #         quantity=quantity,
        #         order_type=self.dhan.MARKET,
        #         product_type=self.dhan.INTRADAY,
        #         price=price
        #     )
        #     return order
        # except Exception as e:
        #     st.error(f"Order Failed: {e}")

bot = DhanExecution()

# ==========================================
# 2. SESSION STATE (MEMORY)
# ==========================================
# Controls the Start/Stop loop
if 'bot_active' not in st.session_state: st.session_state.bot_active = False
# Stores trade details
if 'trade' not in st.session_state: 
    st.session_state.trade = {"status": "IDLE", "entry": 0.0, "sl": 0.0, "tgt": 0.0, "pnl": 0.0, "qty": 0}
# Stores price history for chart
if 'history' not in st.session_state: st.session_state.history = []

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def fetch_option_chain(ticker_symbol):
    """
    Robust fetcher that handles yfinance failures gracefully.
    """
    try:
        tk = yf.Ticker(ticker_symbol)
        # Check if options exist
        exps = tk.options
        if not exps:
            return None, "No Expiry Dates Found. Ticker might be invalid or data blocked."
        
        # Get nearest expiry
        chain = tk.option_chain(exps[0])
        calls = chain.calls
        puts = chain.puts
        
        # Add timestamp for data freshness check
        calls['fetch_time'] = datetime.now()
        puts['fetch_time'] = datetime.now()
        
        return (calls, puts, exps[0]), None
    except Exception as e:
        return None, str(e)

def get_live_price(symbol):
    """
    Fetches single live price point.
    """
    try:
        # yfinance caching workaround
        ticker = yf.Ticker(symbol)
        # fast_info is much faster/stable
        price = ticker.fast_info['last_price']
        return price
    except:
        return None

# ==========================================
# 4. MAIN APP TABS
# ==========================================
tab_scanner, tab_trader = st.tabs(["🔍 1. Option Chain Scanner", "⚡ 2. Live Zero-Hero Bot"])

# ----------------------------------------------------------------------
# TAB 1: SCANNER (FIND THE HERO)
# ----------------------------------------------------------------------
with tab_scanner:
    st.header("Find Momentum Candidates")
    c1, c2, c3 = st.columns(3)
    
    # Input with examples
    input_ticker = c1.text_input("Underlying Ticker", value="SPY", help="Use SPY, AAPL for US. Nifty is often blocked on Yahoo free tier.")
    budget = c2.number_input("Max Premium ($/₹)", value=5.0)
    min_vol = c3.number_input("Min Volume", value=1000)
    
    if st.button("🔎 Scan Option Chain"):
        with st.spinner(f"Fetching data for {input_ticker}..."):
            data, error = fetch_option_chain(input_ticker)
            
            if error:
                st.error(f"Failed to fetch: {error}")
                st.info("💡 Tip: Yahoo Finance blocks automated requests for Indian Indices (^NSEI) often. Try 'SPY' to verify code works.")
            else:
                calls, puts, expiry = data
                st.success(f"Data Fetched! Expiry: {expiry}")
                
                # Zero Hero Logic
                # 1. Filter by Budget and Volume
                hero_calls = calls[(calls['lastPrice'] <= budget) & (calls['volume'] > min_vol)].copy()
                hero_puts = puts[(puts['lastPrice'] <= budget) & (puts['volume'] > min_vol)].copy()
                
                # 2. Add 'Hero Score' (Volume / OpenInterest)
                # High score means fresh momentum
                hero_calls['Hero_Score'] = hero_calls['volume'] / (hero_calls['openInterest'].replace(0, 1))
                hero_puts['Hero_Score'] = hero_puts['volume'] / (hero_puts['openInterest'].replace(0, 1))
                
                st.markdown("### 🚀 Top Call Candidates (Bullish)")
                if not hero_calls.empty:
                    st.dataframe(hero_calls[['contractSymbol', 'strike', 'lastPrice', 'change', 'volume', 'Hero_Score']].sort_values('Hero_Score', ascending=False).head(5), use_container_width=True)
                else:
                    st.warning("No Calls matched your budget/volume criteria.")

                st.markdown("### 🐻 Top Put Candidates (Bearish)")
                if not hero_puts.empty:
                    st.dataframe(hero_puts[['contractSymbol', 'strike', 'lastPrice', 'change', 'volume', 'Hero_Score']].sort_values('Hero_Score', ascending=False).head(5), use_container_width=True)
                else:
                    st.warning("No Puts matched your budget/volume criteria.")
                
                st.info("📋 **Step 2:** Copy the `contractSymbol` (e.g., SPY231215C00460000) and paste it in the **Live Bot** tab.")

# ----------------------------------------------------------------------
# TAB 2: LIVE TRADING BOT
# ----------------------------------------------------------------------
with tab_trader:
    st.header("🤖 Auto-Trading Terminal")
    
    # -- SETTINGS SIDEBAR --
    st.sidebar.markdown("## ⚙️ Algo Settings")
    target_contract = st.sidebar.text_input("Contract Symbol", value="", placeholder="Paste Symbol Here")
    trade_qty = st.sidebar.number_input("Quantity", value=50)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛡️ Risk Management")
    sl_pct = st.sidebar.slider("Stop Loss (%)", 5, 50, 30, help="If price drops 30%, exit.")
    tgt_pct = st.sidebar.slider("Target (%)", 10, 300, 100, help="If price doubles (100%), exit.")
    trail_trigger = st.sidebar.slider("Trail SL Trigger (%)", 10, 100, 20, help="If price moves up 20%, move SL to Cost.")
    
    # -- CONTROL PANEL --
    c_start, c_stop, c_status = st.columns([1, 1, 2])
    
    start = c_start.button("🟢 START BOT")
    stop = c_stop.button("🔴 STOP BOT")
    
    if start:
        if not target_contract:
            st.error("❌ Please enter a Contract Symbol first.")
        else:
            st.session_state.bot_active = True
            st.toast("System Armed. Waiting for data...", icon="🤖")

    if stop:
        st.session_state.bot_active = False
        st.warning("Bot Stopped.")

    # Status Indicator
    status_color = "green" if st.session_state.bot_active else "red"
    status_text = "RUNNING" if st.session_state.bot_active else "STOPPED"
    c_status.markdown(f"### Status: <span style='color:{status_color}'>{status_text}</span>", unsafe_allow_html=True)

    # -- LIVE DISPLAY AREAS --
    placeholder_metrics = st.empty()
    placeholder_chart = st.empty()
    placeholder_logs = st.empty()

    # -- THE TRADING LOOP --
    if st.session_state.bot_active:
        
        # Initialize loop variables
        log_messages = []
        
        while st.session_state.bot_active:
            # 1. FETCH DATA
            current_price = get_live_price(target_contract)
            
            if current_price is None:
                placeholder_logs.error("⚠️ Error fetching data. Retrying in 2s...")
                time.sleep(2)
                continue
                
            # Update History for Chart
            st.session_state.history.append(current_price)
            if len(st.session_state.history) > 50: st.session_state.history.pop(0)
            
            # 2. TRADING LOGIC
            trade = st.session_state.trade
            
            # --- ENTRY LOGIC ---
            if trade["status"] == "IDLE":
                # For this demo, we enter IMMEDIATELY upon Start if not in trade
                # In real strategy, you might wait for RSI > 60 or Price > VWAP
                trade["status"] = "ACTIVE"
                trade["entry"] = current_price
                trade["qty"] = trade_qty
                
                # Calculate Levels
                trade["sl"] = current_price * (1 - sl_pct/100)
                trade["tgt"] = current_price * (1 + tgt_pct/100)
                
                bot.place_order(target_contract, "BUY", trade_qty, current_price)
                log_messages.append(f"✅ ENTRY Filled @ {current_price:.2f} | SL: {trade['sl']:.2f} | TGT: {trade['tgt']:.2f}")

            # --- MANAGEMENT LOGIC (If in trade) ---
            elif trade["status"] == "ACTIVE":
                # Calculate PnL
                pnl = (current_price - trade["entry"]) * trade["qty"]
                pnl_pct = ((current_price - trade["entry"]) / trade["entry"]) * 100
                
                # A. Check Target
                if current_price >= trade["tgt"]:
                    trade["status"] = "CLOSED"
                    bot.place_order(target_contract, "SELL", trade_qty, current_price)
                    log_messages.append(f"🏆 TARGET HIT @ {current_price:.2f} | Profit: {pnl:.2f}")
                    st.session_state.bot_active = False # Stop after target
                
                # B. Check Stop Loss
                elif current_price <= trade["sl"]:
                    trade["status"] = "CLOSED"
                    bot.place_order(target_contract, "SELL", trade_qty, current_price)
                    log_messages.append(f"💀 SL HIT @ {current_price:.2f} | Loss: {pnl:.2f}")
                    st.session_state.bot_active = False # Stop after SL
                
                # C. Trailing SL Logic
                elif pnl_pct >= trail_trigger:
                    # If price is up X%, move SL to Break Even
                    if trade["sl"] < trade["entry"]:
                        trade["sl"] = trade["entry"]
                        log_messages.append(f"🛡️ Trailing SL Moved to Breakeven: {trade['sl']:.2f}")
                        st.toast("Trailing SL Activated!")

            # 3. RENDER UI
            with placeholder_metrics.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Live Price", f"{current_price:.2f}")
                
                if trade["status"] == "ACTIVE":
                    m2.metric("Entry", f"{trade['entry']:.2f}")
                    m3.metric("Stop Loss", f"{trade['sl']:.2f}", delta=f"{current_price - trade['sl']:.2f}")
                    m4.metric("Target", f"{trade['tgt']:.2f}", delta=f"{trade['tgt'] - current_price:.2f}")
                else:
                    m2.metric("Status", "Scanning...")
                    
            # 4. RENDER CHART
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=st.session_state.history, mode='lines+markers', name='Price', line=dict(color='#00ff00')))
            
            # Add horizontal lines if in trade
            if trade["status"] == "ACTIVE":
                fig.add_hline(y=trade["entry"], line_dash="dot", annotation_text="Entry", line_color="gray")
                fig.add_hline(y=trade["sl"], line_dash="dash", annotation_text="SL", line_color="red")
                fig.add_hline(y=trade["tgt"], line_dash="dash", annotation_text="Target", line_color="green")
            
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0), template="plotly_dark")
            placeholder_chart.plotly_chart(fig, use_container_width=True)
            
            # 5. RENDER LOGS
            if log_messages:
                placeholder_logs.info(log_messages[-1])

            # 6. SLEEP (Rate Limit Prevention)
            time.sleep(1.5)

