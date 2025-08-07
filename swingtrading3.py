import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import io
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Universal Trading Strategy Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card { background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%); padding: 1rem;
border-radius: 10px; border-left: 4px solid #007bff; margin-bottom: 10px; }
.signal-buy { background: linear-gradient(90deg, #d4edda 0%, #c3e6cb 100%);
border-left: 4px solid #28a745; padding: 1rem; border-radius: 10px; margin: 10px 0; }
.signal-sell { background: linear-gradient(90deg, #f8d7da 0%, #f5c6cb 100%);
border-left: 4px solid #dc3545; padding: 1rem; border-radius: 10px; margin: 10px 0; }
.signal-hold { background: linear-gradient(90deg, #fff3cd 0%, #ffeaa7 100%);
border-left: 4px solid #ffc107; padding: 1rem; border-radius: 10px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# Your earlier functions: detect_csv_format, load_and_process_data, add_technical_indicators,
# generate_strategy_signals should be placed here (already included in your snippet)

# Continue advanced_backtest from where you left off
def advanced_backtest(df, initial_capital=100000, transaction_cost_pct=0.1, stop_loss_pct=5, take_profit_pct=10):
    df = df.copy()
    cash = initial_capital
    shares = 0
    portfolio_value = []
    trades = []
    position_history = []
    current_position = None
    entry_date = None
    entry_price = None
    entry_reason = None
    days_in_position = 0

    for i, row in df.iterrows():
        current_price = row['Close']
        signal = row['Signal']
        signal_strength = row.get('Signal_Strength', 0)
        current_portfolio_value = cash + (shares * current_price)
        portfolio_value.append(current_portfolio_value)
        days_in_position = days_in_position + 1 if shares > 0 else 0

        # Check stop loss / take profit
        if shares > 0 and current_position:
            stop_loss_price = current_position.get('stop_loss', 0)
            take_profit_price = current_position.get('take_profit', float('inf'))

            if current_price <= stop_loss_price:
                signal = 'SELL'
                exit_reason = f"Stop Loss @ {current_price:.2f}"
            elif current_price >= take_profit_price:
                signal = 'SELL'
                exit_reason = f"Take Profit @ {current_price:.2f}"
            else:
                exit_reason = row.get('Exit_Reason', 'Strategy exit')

        if signal == 'BUY' and shares == 0 and cash > current_price:
            position_pct = min(0.95, signal_strength / 100) if signal_strength > 0 else 0.95
            available_cash = cash * position_pct
            shares_to_buy = int(available_cash / current_price)

            if shares_to_buy > 0:
                cost_pre_fee = shares_to_buy * current_price
                transaction_fee = cost_pre_fee * (transaction_cost_pct / 100)
                total_cost = cost_pre_fee + transaction_fee

                if cash >= total_cost:
                    shares = shares_to_buy
                    cash -= total_cost
                    entry_date = row['Date']
                    entry_price = current_price
                    entry_reason = row.get('Entry_Reason', 'Buy signal')
                    stop_loss_price = current_price * (1 - stop_loss_pct / 100)
                    take_profit_price = current_price * (1 + take_profit_pct / 100)

                    if not pd.isna(row.get('Stop_Loss')):
                        stop_loss_price = row['Stop_Loss']
                    if not pd.isna(row.get('Take_Profit')):
                        take_profit_price = row['Take_Profit']

                    current_position = {
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'shares': shares,
                        'stop_loss': stop_loss_price,
                        'take_profit': take_profit_price,
                        'entry_reason': entry_reason
                    }

                    trades.append({
                        'Date': row['Date'],
                        'Type': 'BUY',
                        'Price': current_price,
                        'Shares': shares,
                        'Cost': total_cost,
                        'Transaction_Fee': transaction_fee,
                        'Reason': entry_reason,
                        'Stop_Loss': stop_loss_price,
                        'Take_Profit': take_profit_price,
                        'Portfolio_Value': current_portfolio_value
                    })

        elif signal == 'SELL' and shares > 0:
            proceeds = shares * current_price
            fee = proceeds * (transaction_cost_pct / 100)
            net_proceeds = proceeds - fee
            cash += net_proceeds

            if current_position:
                entry_value = current_position['entry_price'] * shares
                profit = net_proceeds - entry_value
                profit_pct = (profit / entry_value) * 100
                days_held = (row['Date'] - current_position['entry_date']).days

                trades.append({
                    'Date': row['Date'],
                    'Type': 'SELL',
                    'Price': current_price,
                    'Shares': shares,
                    'Proceeds': net_proceeds,
                    'Transaction_Fee': fee,
                    'Profit_Loss': profit,
                    'Profit_Loss_Pct': profit_pct,
                    'Holding_Days': days_held,
                    'Entry_Price': current_position['entry_price'],
                    'Entry_Date': current_position['entry_date'],
                    'Entry_Reason': current_position['entry_reason'],
                    'Exit_Reason': exit_reason,
                    'Portfolio_Value': current_portfolio_value
                })

            shares = 0
            current_position = None
            days_in_position = 0

        position_history.append({
            'Date': row['Date'],
            'Shares': shares,
            'Cash': cash,
            'Equity_Value': shares * current_price,
            'Portfolio_Value': current_portfolio_value
        })

    df['Portfolio_Value'] = portfolio_value
    trades_df = pd.DataFrame(trades)
    positions_df = pd.DataFrame(position_history)

    return df, trades_df, positions_df

# === Streamlit App ===

st.title("📈 Universal Trading Strategy Backtester")

uploaded_file = st.file_uploader("Upload your stock CSV", type=['csv'])

if uploaded_file:
    df = load_and_process_data(uploaded_file)
    
    if df is not None:
        st.success("✅ Data loaded successfully!")
        df = add_technical_indicators(df)

        strategy = st.selectbox("Select Strategy", [
            "Buy & Hold",
            "Enhanced Contrarian",
            "Smart Momentum",
            "Mean Reversion Pro"
        ])
        
        st.markdown("Customize strategy parameters (optional):")
        drop_thresh = st.slider("Drop % for Contrarian Buy", -10.0, 0.0, -2.0)
        gain_thresh = st.slider("Gain % for Contrarian Sell", 0.0, 10.0, 1.5)

        df = generate_strategy_signals(df, strategy, {
            'drop_threshold': drop_thresh,
            'gain_threshold': gain_thresh
        })

        df_bt, trades_df, positions_df = advanced_backtest(df)

        st.subheader("🧠 Strategy Signals")
        st.dataframe(df_bt[['Date', 'Close', 'Signal', 'Entry_Reason', 'Exit_Reason']].tail(20))

        st.subheader("💹 Trade History")
        if not trades_df.empty:
            st.dataframe(trades_df)

            # Show cumulative P&L chart
            trades_df['Cumulative_PnL'] = trades_df['Profit_Loss'].cumsum()
            fig = px.line(trades_df, x='Date', y='Cumulative_PnL', title="📊 Cumulative Profit/Loss")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trades executed based on this strategy.")

        st.subheader("📊 Portfolio Value Over Time")
        st.plotly_chart(
            px.line(df_bt, x='Date', y='Portfolio_Value', title="Portfolio Value"), 
            use_container_width=True
        )

        with st.expander("🔎 Technical Indicator Snapshot"):
            st.dataframe(df_bt.tail(20))
