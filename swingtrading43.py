import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
import plotly.graph_objects as go
import time

# --- GLOBAL CONSTANTS ---
STANDARD_TICKERS = [
    "AAPL", "MSFT", "KO", "PEP", "TCS.NS", "INFY.NS"
]
PERIODS = ["1d", "5d", "1mo", "6mo", "1y", "5y", "10y"]
IST_TIMEZONE = 'Asia/Kolkata' 

# --- HYPERPARAMETERS ---
TICKER_A = 'AAPL' 
TICKER_B = 'MSFT' 
WINDOW = 252     
ENTRY_Z_SCORE = 2.0
EXIT_Z_SCORE = 0.5 

# --- UTILITY FUNCTIONS ---

def convert_to_ist(data):
    """Converts the DataFrame index to IST/Asia/Kolkata timezone."""
    try:
        if data.index.tz is None:
            # Assume data is UTC if timezone naive, then localize and convert
            data.index = data.index.tz_localize('UTC').tz_convert(IST_TIMEZONE)
        else:
            # Convert directly if timezone aware
            data.index = data.index.tz_convert(IST_TIMEZONE)
    except Exception:
        # Pass silently if conversion fails
        pass 
    return data

# --- DATA FETCHING WITH RETRY LOGIC ---

def fetch_data_with_retry(ticker, period, interval='1d', max_retries=3, delay=3):
    """Fetch data with retry logic and error handling to bypass rate limits."""
    
    for attempt in range(max_retries):
        try:
            time.sleep(delay)  
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if not data.empty:
                # 1. Flatten multi-index columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # 2. Ensure standard column names
                data.columns = [col.strip().title() for col in data.columns]
                
                # 3. Convert to IST
                return convert_to_ist(data)
            else:
                st.warning(f"No data returned for {ticker} (Attempt {attempt + 1}).")
        
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}. Retrying in {delay * (attempt + 2)}s...")
                time.sleep(delay * (attempt + 2)) 
            else:
                st.error(f"Failed to fetch data for {ticker} after {max_retries} attempts.")
                return pd.DataFrame()
                
    return pd.DataFrame() 

# --- INTEGRATED FETCH AND PREPARE ---

@st.cache_data(ttl=3600)
def fetch_and_prepare_data(ticker_a, ticker_b, period):
    """Fetches, cleans, and merges data using the robust retry mechanism."""
    
    data_a_raw = fetch_data_with_retry(ticker_a, period, interval='1d')
    data_b_raw = fetch_data_with_retry(ticker_b, period, interval='1d')

    if data_a_raw.empty or data_b_raw.empty:
        return pd.DataFrame(), f"Could not fetch valid data for one or both tickers ({ticker_a}, {ticker_b})."

    def extract_price_series(data, ticker_name):
        if 'Adj Close' in data.columns:
            return data['Adj Close']
        elif 'Close' in data.columns:
            return data['Close']
        else:
            return None

    data_a = extract_price_series(data_a_raw, ticker_a)
    data_b = extract_price_series(data_b_raw, ticker_b)

    if data_a is None or data_b is None:
        return pd.DataFrame(), "Price column (Adj Close or Close) not found in fetched data."

    df = pd.DataFrame({'A': data_a, 'B': data_b}).dropna()
    
    if len(df) < WINDOW:
        return pd.DataFrame(), f"Insufficient overlapping data (need > {WINDOW} points, got {len(df)}). Reduce the Lookback Window or increase the Data Period."
        
    return df, None

# --- TRADING LOGIC WITH KEYERROR FIX ---

def get_beta(series):
    """Calculates the hedge ratio (Beta) using Ordinary Least Squares (OLS)."""
    # FIX for KeyError: 'A' -> Defensive access using .loc 
    # and ensuring the series is not empty before regression.
    if series.empty:
        return np.nan
        
    try:
        # Accessing columns 'A' and 'B' from the passed rolling slice
        y = series.loc[:, 'A']
        X = series.loc[:, 'B']
        
        X = add_constant(X, prepend=False) 
        model = OLS(y, X).fit()
        return model.params[0] 
    except Exception:
         return np.nan

def calculate_spread_and_zscore(df):
    """Calculates the spread and Z-Score using a rolling window."""

    # We apply the rolling function to the entire DataFrame slice df[['A', 'B']]
    # This guarantees the 'A' and 'B' columns exist in the slice passed to get_beta.
    df['Beta'] = df[['A', 'B']].rolling(WINDOW).apply(get_beta, raw=False).shift(1)
    df.dropna(subset=['Beta'], inplace=True)
    
    df['Spread'] = df['A'] - df['Beta'] * df['B']
    
    df['Mean_Spread'] = df['Spread'].rolling(WINDOW).mean().shift(1)
    df['Std_Spread'] = df['Spread'].rolling(WINDOW).std().shift(1)
    
    df['Z_Score'] = (df['Spread'] - df['Mean_Spread']) / df['Std_Spread']
    df.dropna(subset=['Z_Score', 'Spread'], inplace=True)

    return df

def run_pairs_backtest(df, entry_z, exit_z):
    """Executes the Z-Score strategy backtest."""
    df['Signal'] = 0  
    
    df.loc[df['Z_Score'] >= entry_z, 'Signal'] = -1  
    df.loc[df['Z_Score'] <= -entry_z, 'Signal'] = 1   
    
    df.loc[df['Z_Score'].abs() <= exit_z, 'Signal'] = 0
    
    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill').fillna(0)
    
    df['Daily_Spread_Change'] = df['Spread'].diff()
    df['P_L'] = df['Position'].shift(1) * df['Daily_Spread_Change']
    df['Cumulative_P_L'] = df['P_L'].cumsum().fillna(0)
    
    return df

def display_results(df_final):
    """Displays key metrics and chart using Streamlit."""
    
    total_return = df_final['Cumulative_P_L'].iloc[-1]
    sharpe_ratio = np.sqrt(WINDOW) * df_final['P_L'].mean() / df_final['P_L'].std() if df_final['P_L'].std() != 0 else 0
    trades_df = df_final[df_final['Position'].diff() != 0].copy()
    entry_count = (trades_df['Position'] != 0).sum()
    
    st.markdown("---")
    st.markdown("### 📊 Backtest Performance Summary")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total P&L (USD)", f"${total_return:,.2f}")
    col2.metric("Annualized Sharpe Ratio", f"{sharpe_ratio:.2f}")
    col3.metric("Trade Entries", f"{entry_count}")
    

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
    
    fig.update_layout(title=f'{st.session_state.ticker_a} / {st.session_state.ticker_b} Pairs Trading Z-Score',
                      yaxis_title='Z-Score', xaxis_title='Date', height=500,
                      yaxis_range=[y_min, y_max])
    st.plotly_chart(fig, use_container_width=True)
    

# --- MAIN EXECUTION ---
def main_pairs_dashboard():
    st.title("🤝 Professional Pairs Trading Strategy (Z-Score Mean Reversion)")
    st.markdown("This version uses **Retry Logic** and **Defensive Programming** for high reliability.")

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
        base_a = col_a_sel.selectbox("Stock A (Base)", STANDARD_TICKERS, index=0, key='sb_a')
        st.session_state.ticker_a = col_a_text.text_input("Custom A (Override)", value=base_a).upper()

        # Ticker B input
        col_b_sel, col_b_text = st.columns([1, 2])
        base_b = col_b_sel.selectbox("Stock B (Hedge)", STANDARD_TICKERS, index=1, key='sb_b')
        st.session_state.ticker_b = col_b_text.text_input("Custom B (Override)", value=base_b).upper()

        st.session_state.period = st.selectbox("Data Period", PERIODS, index=5)
        
        st.session_state.window = st.number_input("Lookback Window (Days)", min_value=30, value=WINDOW, step=10)
        st.session_state.entry_z_score = st.slider("Entry Z-Score (Threshold)", 1.5, 3.5, ENTRY_Z_SCORE, 0.1)
        st.session_state.exit_z_score = st.slider("Exit Z-Score (Mean Reversion)", 0.0, 1.0, EXIT_Z_SCORE, 0.1)
        
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
            
            with st.status(f"Running backtest for {st.session_state.ticker_a} / {st.session_state.ticker_b}...", expanded=True) as status:
                st.write("1. Calculating Rolling Beta and Spread...")
                df_spread = calculate_spread_and_zscore(df_raw.copy())
                
                st.write("2. Calculating Z-Score and generating signals...")
                df_final = run_pairs_backtest(df_spread, st.session_state.entry_z_score, st.session_state.exit_z_score)
                
                st.write("3. Compiling results and generating charts...")
                time.sleep(1) 
                status.update(label="✅ Backtest Complete!", state="complete", expanded=False)

            display_results(df_final)
            
            st.markdown("---")
            st.subheader("Statistical Data Sample (Z-Score & Spread)")
            st.dataframe(df_final[['A', 'B', 'Beta', 'Spread', 'Z_Score', 'Position', 'P_L']].tail(10), use_container_width=True)

    else:
        st.info("The tickers are defaulted to **AAPL/MSFT**. Click **'Run Backtest'** to begin.")
        
if __name__ == "__main__":
    main_pairs_dashboard()
