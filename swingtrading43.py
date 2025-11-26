import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant # Used for Cointegration/Beta
import plotly.graph_objects as go

# --- CONFIG ---
st.set_page_config(
    page_title="Pairs Trading (Z-Score) Backtest",
    layout="wide"
)

# --- HYPERPARAMETERS ---
TICKER_A = 'XLE' # Energy Sector ETF (Stock A)
TICKER_B = 'XOM' # Exxon Mobil (Stock B)
PERIOD = '5y'    # Lookback period for yfinance data
WINDOW = 252     # Rolling window for calculating Z-Score (approx 1 trading year)
ENTRY_Z_SCORE = 2.0
EXIT_Z_SCORE = 0.5 

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
        X = add_constant(X, prepend=False)
        model = OLS(y, X).fit()
        return model.params[0] # The slope (beta/hedge ratio)

    # Apply OLS over a rolling window to get the time-varying hedge ratio
    df['Beta'] = df[['A', 'B']].rolling(WINDOW).apply(get_beta, raw=False).shift(1)
    df.dropna(subset=['Beta'], inplace=True)
    
    # 2. Calculate the Stationary Spread
    df['Spread'] = df['A'] - df['Beta'] * df['B']
    
    # 3. Calculate Rolling Z-Score
    df['Mean_Spread'] = df['Spread'].rolling(WINDOW).mean().shift(1)
    df['Std_Spread'] = df['Spread'].rolling(WINDOW).std().shift(1)
    
    df['Z_Score'] = (df['Spread'] - df['Mean_Spread']) / df['Std_Spread']
    df.dropna(subset=['Z_Score'], inplace=True)

    return df

def run_pairs_backtest(df):
    """Executes the Z-Score strategy backtest."""
    df['Signal'] = 0  # 0: Hold/Neutral, 1: Long Spread, -1: Short Spread
    
    # Strategy Logic
    df.loc[df['Z_Score'] >= ENTRY_Z_SCORE, 'Signal'] = -1  # Short Spread (Sell A, Buy B)
    df.loc[df['Z_Score'] <= -ENTRY_Z_SCORE, 'Signal'] = 1   # Long Spread (Buy A, Sell B)
    
    # Exit/Close Trade Logic: If the Z-Score is between -EXIT_Z_SCORE and +EXIT_Z_SCORE
    df.loc[df['Z_Score'].abs() <= EXIT_Z_SCORE, 'Signal'] = 0
    
    # Ensure positions are maintained until an exit or opposite entry signal
    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill').fillna(0)
    
    # 4. Calculate P&L (based on the stationary portfolio Spread)
    df['Daily_Spread_Change'] = df['Spread'].diff()
    df['P_L'] = df['Position'].shift(1) * df['Daily_Spread_Change']
    df['Cumulative_P_L'] = df['P_L'].cumsum().fillna(0)
    
    return df

def display_results(df_final):
    """Displays key metrics and chart using Streamlit."""
    
    total_return = df_final['Cumulative_P_L'].iloc[-1]
    sharpe_ratio = np.sqrt(WINDOW) * df_final['P_L'].mean() / df_final['P_L'].std()

    # Calculate Trade Metrics (simplified as the strategy trades the spread)
    trades_df = df_final[df_final['Position'].diff() != 0].copy()
    total_trades = len(trades_df[trades_df['Position'] != 0])
    
    st.markdown("---")
    st.markdown("### 📊 Backtest Performance Summary")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total P&L (USD)", f"${total_return:,.2f}")
    col2.metric("Annualized Sharpe Ratio", f"{sharpe_ratio:.2f}", help="Sharpe Ratio > 1.0 is considered good.")
    col3.metric("Trade Signals Generated", f"{total_trades}")
    
    # --- Charting the Z-Score and Trades ---
    st.markdown("### 📉 Z-Score and Trading Signals")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_final.index, y=df_final['Z_Score'], name='Z-Score', line=dict(color='blue', width=2)))
    
    # Add Entry/Exit Lines
    fig.add_hline(y=ENTRY_Z_SCORE, line_dash="dash", line_color="red", annotation_text="Short Entry / Mean Exit")
    fig.add_hline(y=-ENTRY_Z_SCORE, line_dash="dash", line_color="green", annotation_text="Long Entry / Mean Exit")
    fig.add_hline(y=EXIT_Z_SCORE, line_dash="dot", line_color="gray")
    fig.add_hline(y=-EXIT_Z_SCORE, line_dash="dot", line_color="gray")
    
    # Highlight Long/Short Positions on the Z-Score Chart
    fig.add_trace(go.Scatter(x=df_final[df_final['Position'] == 1].index, 
                             y=df_final[df_final['Position'] == 1]['Z_Score'], 
                             mode='markers', marker=dict(symbol='triangle-up', size=8, color='green'), 
                             name='Long Spread Entry'))
    fig.add_trace(go.Scatter(x=df_final[df_final['Position'] == -1].index, 
                             y=df_final[df_final['Position'] == -1]['Z_Score'], 
                             mode='markers', marker=dict(symbol='triangle-down', size=8, color='red'), 
                             name='Short Spread Entry'))

    fig.update_layout(title=f'{TICKER_A} / {TICKER_B} Pairs Trading Z-Score',
                      yaxis_title='Z-Score', xaxis_title='Date', height=500)
    st.plotly_chart(fig, use_container_width=True)
    

    # --- P&L Chart ---
    st.markdown("### 💰 Cumulative Profit and Loss")
    fig_pl = go.Figure()
    fig_pl.add_trace(go.Scatter(x=df_final.index, y=df_final['Cumulative_P_L'], fill='tozeroy', name='Cumulative P&L'))
    fig_pl.update_layout(title='Strategy Equity Curve', yaxis_title='Cumulative P&L', xaxis_title='Date', height=300)
    st.plotly_chart(fig_pl, use_container_width=True)

# --- MAIN EXECUTION ---
st.title("🤝 Professional Pairs Trading Strategy (Z-Score Mean Reversion)")
st.markdown("This strategy exploits the stationary relationship (Cointegration) between two related assets.")

with st.sidebar:
    st.header("Stock Pair & Parameters")
    ticker_a = st.text_input("Stock A (Base)", value=TICKER_A).upper()
    ticker_b = st.text_input("Stock B (Hedge)", value=TICKER_B).upper()
    period = st.selectbox("Data Period", PERIODS, index=5)
    
    window = st.number_input("Lookback Window (Days)", min_value=10, value=WINDOW, step=10, 
                             help="The historical period used to calculate the Mean and Std Dev.")
    entry_z_score = st.slider("Entry Z-Score (Volatility Threshold)", 1.5, 3.0, ENTRY_Z_SCORE, 0.1)
    exit_z_score = st.slider("Exit Z-Score (Mean Reversion Point)", 0.0, 1.0, EXIT_Z_SCORE, 0.1)
    
    if st.button("🚀 Run Backtest"):
        st.session_state.run_backtest = True
        st.session_state.ticker_a = ticker_a
        st.session_state.ticker_b = ticker_b
        st.session_state.period = period
        st.session_state.window = window
        st.session_state.entry_z_score = entry_z_score
        st.session_state.exit_z_score = exit_z_score

if 'run_backtest' in st.session_state and st.session_state.run_backtest:
    
    df_raw, error = fetch_and_prepare_data(st.session_state.ticker_a, st.session_state.ticker_b, st.session_state.period)

    if error:
        st.error(error)
    elif not df_raw.empty:
        with st.spinner("Calculating Rolling Beta and Z-Scores..."):
            df_spread = calculate_spread_and_zscore(df_raw.copy())
            
        with st.spinner("Running Strategy Backtest and P&L Calculation..."):
            df_final = run_pairs_backtest(df_spread)

        display_results(df_final)
    
    st.session_state.run_backtest = False # Prevent endless loop if button isn't pressed
else:
    st.info("Select your parameters in the sidebar and click 'Run Backtest' to execute the strategy.")

