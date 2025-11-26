import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
import plotly.graph_objects as go
import time

# --- GLOBAL CONSTANTS (Necessary for Streamlit inputs) ---
STANDARD_TICKERS = [
    "XLE", "XOM", "KO", "PEP", "TCS.NS", "INFY.NS"
]
TIME_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
PERIODS = ["1d", "5d", "1mo", "6mo", "1y", "5y", "10y"]
IST_TIMEZONE = 'Asia/Kolkata' 

# --- HYPERPARAMETERS ---
TICKER_A = 'KO' # Coca-Cola (Suggested reliable pair)
TICKER_B = 'PEP' # PepsiCo
WINDOW = 252     # Rolling window for calculating Z-Score (approx 1 trading year)
ENTRY_Z_SCORE = 2.0
EXIT_Z_SCORE = 0.5 

# --- CORE FUNCTIONS ---

def get_price_series(ticker, period):
    """
    Fetches data and ensures a time-series Series is returned, 
    handling the 'Adj Close' column selection robustly.
    """
    # Fetch data
    data = yf.download(ticker, period=period, progress=False)
    
    if data.empty:
        return None
    
    # Identify the target column ('Adj Close' or 'Close')
    if 'Adj Close' in data.columns:
        price_series = data['Adj Close']
    elif 'Close' in data.columns:
        # Use 'Close' as a fallback if 'Adj Close' is missing (e.g., for short intervals)
        price_series = data['Close']
    else:
        # No usable price data found
        return None
        
    # Ensure the result is a pandas Series with an index. This fixes the 'scalar values' error.
    if isinstance(price_series, pd.Series):
        return price_series.dropna()
    else:
        return None

@st.cache_data(ttl=3600)
def fetch_and_prepare_data(ticker_a, ticker_b, period):
    """Fetches and merges data for the two stocks."""
    try:
        data_a = get_price_series(ticker_a, period)
        data_b = get_price_series(ticker_b, period)
        
        if data_a is None or data_b is None:
            return pd.DataFrame(), "Could not fetch valid price data for one or both tickers."

        # Combine and rename columns, ensuring the index is aligned (fixing the scalar error indirectly)
        df = pd.DataFrame({'A': data_a, 'B': data_b}).dropna()
        
        if df.empty or len(df) < WINDOW:
            return pd.DataFrame(), f"Insufficient overlapping data (need > {WINDOW} points after dropping NaNs)."
            
        return df, None
        
    except Exception as e:
        # Catch any unexpected errors during the DataFrame creation
        return pd.DataFrame(), f"Error during data processing: {e}"

def calculate_spread_and_zscore(df):
    """Calculates the hedge ratio, spread, and Z-Score using a rolling window."""
    
    # 1. Rolling Hedge Ratio (Beta) Calculation using OLS
    def get_beta(series):
        y = series['A']
        X = series['B']
        X = add_constant(X, prepend=False) 
        try:
            model = OLS(y, X).fit()
            return model.params[0] 
        except:
             return np.nan

    df['Beta'] = df[['A', 'B']].rolling(WINDOW).apply(get_beta, raw=False).shift(1)
    df.dropna(subset=['Beta'], inplace=True)
    
    # 2. Calculate the Stationary Spread
    df['Spread'] = df['A'] - df['Beta'] * df['B']
    
    # 3. Calculate Rolling Z-Score
    df['Mean_Spread'] = df['Spread'].rolling(WINDOW).mean().shift(1)
    df['Std_Spread'] = df['Spread'].rolling(WINDOW).std().shift(1)
    
    df['Z_Score'] = (df['Spread'] - df['Mean_Spread']) / df['Std_Spread']
    df.dropna(subset=['Z_Score', 'Spread'], inplace=True)

    return df

def run_pairs_backtest(df, entry_z, exit_z):
    """Executes the Z-Score strategy backtest."""
    df['Signal'] = 0  
    
    # Strategy Logic: Generate entry signals based on the Z-Score
    df.loc[df['Z_Score'] >= entry_z, 'Signal'] = -1  # Short Spread (Sell A, Buy B)
    df.loc[df['Z_Score'] <= -entry_z, 'Signal'] = 1   # Long Spread (Buy A, Sell B)
    
    # Exit/Close Trade Logic: If the Z-Score is between -EXIT_Z_SCORE and +EXIT_Z_SCORE
    df.loc[df['Z_Score'].abs() <= exit_z, 'Signal'] = 0
    
    # Position tracking: Forward-fill the last signal to maintain the position
    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill').fillna(0)
    
    # 4. Calculate P&L 
    df['Daily_Spread_Change'] = df['Spread'].diff()
    # P&L = Previous Day's Position * Today's change in Spread value
    df['P_L'] = df['Position'].shift(1) * df['Daily_Spread_Change']
    df['Cumulative_P_L'] = df['P_L'].cumsum().fillna(0)
    
    return df

def display_results(df_final):
    """Displays key metrics and chart using Streamlit."""
    
    total_return = df_final['Cumulative_P_L'].iloc[-1]
    
    # Annualized Sharpe Ratio 
    sharpe_ratio = np.sqrt(WINDOW) * df_final['P_L'].mean() / df_final['P_L'].std() if df_final['P_L'].std() != 0 else 0

    trades_df = df_final[df_final['Position'].diff() != 0].copy()
    entry_count = (trades_df['Position'] != 0).sum()
    
    st.markdown("---")
    st.markdown("### 📊 Backtest Performance Summary")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total P&L (USD)", f"${total_return:,.2f}")
    col2.metric("Annualized Sharpe Ratio", f"{sharpe_ratio:.2f}")
    col3.metric("Trade Entries", f"{entry_count}")

    # --- Charting the Z-Score and Trades ---
    st.markdown("### 📉 Z-Score and Trading Signals")
    
    y_min = df_final['Z_Score'].min() - 0.5
    y_max = df_final['Z_Score'].max() + 0.5

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_final.index, y=df_final['Z_Score'], name='Z-Score', line=dict(color='blue', width=2)))
    
    # Entry/Exit Lines
    fig.add_hline(y=st.session_state.entry_z_score, line_dash="dash", line_color="red", annotation_text="Short Entry", opacity=0.8)
    fig.add_hline(y=-st.session_state.entry_z_score, line_dash="dash", line_color="green", annotation_text="Long Entry", opacity=0.8)
    fig.add_hline(y=st.session_state.exit_z_score, line_dash="dot", line_color="gray", annotation_text="Exit", opacity=0.5)
    fig.add_hline(y=-st.session_state.exit_z_score, line_dash="dot", line_color="gray", opacity=0.5)
    
    # Highlight Long/Short Positions in the background
    long_dates = df_final[df_final['Position'] == 1].index
    short_dates = df_final[df_final['Position'] == -1].index
    
    if not long_dates.empty:
        fig.add_vrect(x0=long_dates.min(), x1=long_dates.max(), fillcolor="green", opacity=0.1, layer="below", line_width=0, name="Long Spread")
    if not short_dates.empty:
        fig.add_vrect(x0=short_dates.min(), x1=short_dates.max(), fillcolor="red", opacity=0.1, layer="below", line_width=0, name="Short Spread")

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
        st.session_state.period = PERIODS[5] 
        st.session_state.window = WINDOW
        st.session_state.entry_z_score = ENTRY_Z_SCORE
        st.session_state.exit_z_score = EXIT_Z_SCORE
        
    with st.sidebar:
        st.header("Stock Pair & Parameters")
        
        # Ticker A input
        col_a_sel, col_a_text = st.columns([1, 2])
        # Default index set to KO (index 2)
        base_a = col_a_sel.selectbox("Stock A (Base)", STANDARD_TICKERS, index=2, key='sb_a')
        st.session_state.ticker_a = col_a_text.text_input("Custom A (Override)", value=base_a).upper()

        # Ticker B input
        col_b_sel, col_b_text = st.columns([1, 2])
        # Default index set to PEP (index 3)
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
                time.sleep(1) 
                status.update(label="✅ Backtest Complete!", state="complete", expanded=False)

            display_results(df_final)
            
            st.markdown("---")
            st.subheader("Statistical Data Sample (Z-Score & Spread)")
            st.dataframe(df_final[['A', 'B', 'Beta', 'Spread', 'Z_Score', 'Position', 'P_L']].tail(10), use_container_width=True)

    else:
        st.info("Select your stock pair (e.g., KO/PEP) and parameters in the sidebar and click **'Run Backtest'** to begin.")
        
if __name__ == "__main__":
    main_pairs_dashboard()
