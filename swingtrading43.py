import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
import plotly.graph_objects as go
import time

# --- GLOBAL CONSTANTS (Re-added to resolve NameError) ---
STANDARD_TICKERS = [
    "^NSEI", "^BANKNIFTY", "XLE", "XOM", "KO", "PEP"
]
TIME_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
PERIODS = ["1d", "5d", "1mo", "6mo", "1y", "5y", "10y"]
IST_TIMEZONE = 'Asia/Kolkata' 

# --- CONFIG ---
st.set_page_config(
    page_title="🤝 Professional Pairs Trading (Z-Score) Backtest",
    layout="wide"
)

# --- HYPERPARAMETERS ---
TICKER_A = 'XLE' # Energy Sector ETF (Stock A)
TICKER_B = 'XOM' # Exxon Mobil (Stock B)
WINDOW = 252     # Rolling window for calculating Z-Score (approx 1 trading year)
ENTRY_Z_SCORE = 2.0
EXIT_Z_SCORE = 0.5 

# --- CORE FUNCTIONS ---

@st.cache_data(ttl=3600)
def fetch_and_prepare_data(ticker_a, ticker_b, period):
    """Fetches and merges data for the two stocks."""
    try:
        data_a = yf.download(ticker_a, period=period, progress=False)['Adj Close']
        data_b = yf.download(ticker_b, period=period, progress=False)['Adj Close']
        
        df = pd.DataFrame({'A': data_a, 'B': data_b}).dropna()
        if df.empty or len(df) < WINDOW:
            return pd.DataFrame(), f"Insufficient data (need > {WINDOW} points)."
            
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"Error fetching data: {e}"

def calculate_spread_and_zscore(df):
    """Calculates the hedge ratio, spread, and Z-Score using a rolling window."""
    
    # 1. Rolling Hedge Ratio (Beta) Calculation using OLS
    def get_beta(series):
        # Rolling regression of A on B
        y = series['A']
        X = series['B']
        # Add a constant term for the regression (intercept)
        X = add_constant(X, prepend=False) 
        try:
            model = OLS(y, X).fit()
            # The hedge ratio is the coefficient of B (index 0 if we prepend constant)
            return model.params[0] 
        except:
             return np.nan

    # Apply OLS over a rolling window to get the time-varying hedge ratio
    # We shift(1) to prevent look-ahead bias: the beta for today is calculated using yesterday's data
    df['Beta'] = df[['A', 'B']].rolling(WINDOW).apply(get_beta, raw=False).shift(1)
    df.dropna(subset=['Beta'], inplace=True)
    
    # 2. Calculate the Stationary Spread
    # Spread = Price_A - Beta * Price_B (This is the stationary portfolio)
    df['Spread'] = df['A'] - df['Beta'] * df['B']
    
    # 3. Calculate Rolling Z-Score
    # Shift(1) again to ensure mean and std dev are calculated using only past data
    df['Mean_Spread'] = df['Spread'].rolling(WINDOW).mean().shift(1)
    df['Std_Spread'] = df['Spread'].rolling(WINDOW).std().shift(1)
    
    df['Z_Score'] = (df['Spread'] - df['Mean_Spread']) / df['Std_Spread']
    df.dropna(subset=['Z_Score', 'Spread'], inplace=True)

    return df

def run_pairs_backtest(df, entry_z, exit_z):
    """Executes the Z-Score strategy backtest."""
    df['Signal'] = 0  
    
    # Strategy Logic: Generate entry signals based on the Z-Score
    
    # Short Spread (Sell A, Buy B): Z-Score is high (Spread is overbought)
    df.loc[df['Z_Score'] >= entry_z, 'Signal'] = -1  
    
    # Long Spread (Buy A, Sell B): Z-Score is low (Spread is oversold)
    df.loc[df['Z_Score'] <= -entry_z, 'Signal'] = 1   
    
    # Exit/Close Trade Logic: If the Z-Score is between -EXIT_Z_SCORE and +EXIT_Z_SCORE, set signal to 0
    df.loc[df['Z_Score'].abs() <= exit_z, 'Signal'] = 0
    
    # Position tracking: Forward-fill the last signal to maintain the position until a new signal (entry or exit)
    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill').fillna(0)
    
    # 4. Calculate P&L 
    # The spread is the P&L of the stationary portfolio (A - Beta*B).
    # We calculate P&L by taking the change in the spread and multiplying by the previous day's position.
    df['Daily_Spread_Change'] = df['Spread'].diff()
    df['P_L'] = df['Position'].shift(1) * df['Daily_Spread_Change']
    df['Cumulative_P_L'] = df['P_L'].cumsum().fillna(0)
    
    return df

def display_results(df_final):
    """Displays key metrics and chart using Streamlit."""
    
    # --- Performance Metrics ---
    total_return = df_final['Cumulative_P_L'].iloc[-1]
    
    # Annualized Sharpe Ratio = sqrt(Trading Days per Year) * Mean Daily Return / Std Dev Daily Return
    sharpe_ratio = np.sqrt(WINDOW) * df_final['P_L'].mean() / df_final['P_L'].std() if df_final['P_L'].std() != 0 else 0

    # Calculate Trade Metrics (using position changes)
    # Total trades is half the number of times the position changes from non-zero to zero (entry + exit)
    # A cleaner way is counting entries (Signal != 0)
    trades_df = df_final[df_final['Position'].diff() != 0].copy()
    
    # Count how many times we ENTERED a trade (Position went from 0 to non-zero)
    entry_count = (trades_df['Position'] != 0).sum()
    
    # This strategy often involves partial trades/re-entries, so a simple P&L aggregation is better.
    # The P/L for each position change from a closed trade is:
    trade_profits = trades_df[(trades_df['Position'] == 0) & (trades_df['Position'].shift(1) != 0)]
    
    if not trade_profits.empty:
        # Calculate P/L for each completed trade (simplified win rate)
        completed_trades = len(trade_profits)
        winning_trades = (trade_profits['P_L'] > 0).sum()
        win_rate = (winning_trades / completed_trades) * 100 if completed_trades > 0 else 0
    else:
        completed_trades = 0
        win_rate = 0
    
    st.markdown("---")
    st.markdown("### 📊 Backtest Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total P&L (USD)", f"${total_return:,.2f}")
    col2.metric("Annualized Sharpe Ratio", f"{sharpe_ratio:.2f}")
    col3.metric("Trade Entries", f"{entry_count}")
    col4.metric("Win Rate (Completed Trades)", f"{win_rate:.2f}%" if completed_trades > 0 else "N/A")
    
    # --- Charting the Z-Score and Trades ---
    st.markdown("### 📉 Z-Score and Trading Signals")
    
    # Calculate Z-Score min/max for chart range
    y_min = df_final['Z_Score'].min() - 0.5
    y_max = df_final['Z_Score'].max() + 0.5

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_final.index, y=df_final['Z_Score'], name='Z-Score', line=dict(color='blue', width=2)))
    
    # Entry Lines
    fig.add_hline(y=st.session_state.entry_z_score, line_dash="dash", line_color="red", annotation_text="Short Entry", opacity=0.8)
    fig.add_hline(y=-st.session_state.entry_z_score, line_dash="dash", line_color="green", annotation_text="Long Entry", opacity=0.8)
    
    # Exit Lines
    fig.add_hline(y=st.session_state.exit_z_score, line_dash="dot", line_color="gray", annotation_text="Exit", opacity=0.5)
    fig.add_hline(y=-st.session_state.exit_z_score, line_dash="dot", line_color="gray", opacity=0.5)
    
    # Highlight Long/Short Positions in the background
    fig.add_vrect(x0=df_final[df_final['Position'] == 1].index.min(), x1=df_final[df_final['Position'] == 1].index.max(), 
                  fillcolor="green", opacity=0.1, layer="below", line_width=0, name="Long Spread")
    fig.add_vrect(x0=df_final[df_final['Position'] == -1].index.min(), x1=df_final[df_final['Position'] == -1].index.max(), 
                  fillcolor="red", opacity=0.1, layer="below", line_width=0, name="Short Spread")

    fig.update_layout(title=f'{st.session_state.ticker_a} / {st.session_state.ticker_b} Pairs Trading Z-Score',
                      yaxis_title='Z-Score', xaxis_title='Date', height=500,
                      yaxis_range=[y_min, y_max])
    st.plotly_chart(fig, use_container_width=True)
    

    # --- P&L Chart ---
    st.markdown("### 💰 Cumulative Profit and Loss")
    fig_pl = go.Figure()
    fig_pl.add_trace(go.Scatter(x=df_final.index, y=df_final['Cumulative_P_L'], fill='tozeroy', 
                                line=dict(color='darkblue'), name='Equity Curve'))
    fig_pl.update_layout(title='Strategy Equity Curve', yaxis_title='Cumulative P&L', 
                         xaxis_title='Date', height=300)
    st.plotly_chart(fig_pl, use_container_width=True)

# --- MAIN EXECUTION ---
def main_pairs_dashboard():
    st.title("🤝 Professional Pairs Trading Strategy (Z-Score Mean Reversion)")
    st.markdown("This strategy exploits the **stationary relationship (Cointegration)** between two related assets by trading the **Z-Score** of their spread.")

    # --- Session State Initialization ---
    if 'run_backtest' not in st.session_state:
        st.session_state.run_backtest = False
        st.session_state.ticker_a = TICKER_A
        st.session_state.ticker_b = TICKER_B
        st.session_state.period = PERIODS[5] # Default to 5y
        st.session_state.window = WINDOW
        st.session_state.entry_z_score = ENTRY_Z_SCORE
        st.session_state.exit_z_score = EXIT_Z_SCORE
        
    with st.sidebar:
        st.header("Stock Pair & Parameters")
        
        # Ticker A input with standard suggestions
        col_a_sel, col_a_text = st.columns([1, 2])
        base_a = col_a_sel.selectbox("Stock A (Base)", STANDARD_TICKERS, index=2, key='sb_a')
        st.session_state.ticker_a = col_a_text.text_input("Custom A (Override)", value=base_a).upper()

        # Ticker B input with standard suggestions
        col_b_sel, col_b_text = st.columns([1, 2])
        base_b = col_b_sel.selectbox("Stock B (Hedge)", STANDARD_TICKERS, index=3, key='sb_b')
        st.session_state.ticker_b = col_b_text.text_input("Custom B (Override)", value=base_b).upper()

        st.session_state.period = st.selectbox("Data Period", PERIODS, index=5)
        
        st.session_state.window = st.number_input("Lookback Window (Days)", min_value=30, value=WINDOW, step=10, 
                                                 help="The number of past days used to calculate the rolling Mean and Std Dev. (252 ≈ 1 year)")
        st.session_state.entry_z_score = st.slider("Entry Z-Score (Threshold)", 1.5, 3.5, ENTRY_Z_SCORE, 0.1, 
                                                  help="Enter when the Z-Score is above this magnitude.")
        st.session_state.exit_z_score = st.slider("Exit Z-Score (Mean Reversion)", 0.0, 1.0, EXIT_Z_SCORE, 0.1, 
                                                 help="Exit when the Z-Score returns to this magnitude near zero.")
        
        if st.button("🚀 Run Backtest"):
            st.session_state.run_backtest = True
            st.rerun()

    if st.session_state.run_backtest:
        
        df_raw, error = fetch_and_prepare_data(
            st.session_state.ticker_a, 
            st.session_state.ticker_b, 
            st.session_state.period
        )

        if error:
            st.error(error)
        elif not df_raw.empty:
            
            # --- Analysis Execution ---
            with st.status(f"Running backtest for {st.session_state.ticker_a} / {st.session_state.ticker_b}...", expanded=True) as status:
                st.write("1. Calculating Rolling Beta (Hedge Ratio) using OLS...")
                df_spread = calculate_spread_and_zscore(df_raw.copy())
                
                st.write("2. Calculating Z-Score and generating entry/exit signals...")
                df_final = run_pairs_backtest(df_spread, st.session_state.entry_z_score, st.session_state.exit_z_score)
                
                st.write("3. Compiling results and generating charts...")
                time.sleep(1) # Visual delay for status update
                status.update(label="✅ Backtest Complete!", state="complete", expanded=False)

            display_results(df_final)
            
            st.markdown("---")
            st.subheader("Statistical Data Sample (Z-Score & Spread)")
            st.dataframe(df_final[['A', 'B', 'Beta', 'Spread', 'Z_Score', 'Position', 'P_L']].tail(10), use_container_width=True)

    else:
        st.info("Select your stock pair and parameters in the sidebar and click **'Run Backtest'** to begin.")
        
if __name__ == "__main__":
    main_pairs_dashboard()
