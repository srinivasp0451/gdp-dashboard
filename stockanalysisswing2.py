# Advanced Nifty 50 Stock Screener & Swing Trading System
# No TA-Lib or pandas-ta dependencies
# Streamlit app with custom indicators, backtest & AI rating

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# ==================== CONFIG ======================

NIFTY50 = [
    'ADANIPORTS.NS','APOLLOHOSP.NS','ASIANPAINT.NS','AXISBANK.NS','BAJAJ-AUTO.NS','BAJFINANCE.NS',
    'BAJAJFINSV.NS','BPCL.NS','BHARTIARTL.NS','BRITANNIA.NS','CIPLA.NS','COALINDIA.NS','DIVISLAB.NS',
    'DRREDDY.NS','EICHERMOT.NS','GRASIM.NS','HCLTECH.NS','HDFCBANK.NS','HDFCLIFE.NS','HEROMOTOCO.NS',
    'HINDALCO.NS','HINDUNILVR.NS','ICICIBANK.NS','ITC.NS','INDUSINDBK.NS','INFY.NS','JSWSTEEL.NS','KOTAKBANK.NS',
    'LTIM.NS','LT.NS','M&M.NS','MARUTI.NS','NTPC.NS','NESTLEIND.NS','ONGC.NS','POWERGRID.NS','PIDILITIND.NS',
    'RELIANCE.NS','SBILIFE.NS','SBIN.NS','SUNPHARMA.NS','TCS.NS','TATACONSUM.NS','TATAMOTORS.NS',
    'TATASTEEL.NS','TECHM.NS','TITAN.NS','UPL.NS','ULTRACEMCO.NS','WIPRO.NS'
]

START_DATE = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')

# Adjustable target and stop loss in points
MIN_PROFIT_TARGET_POINTS = 50
MAX_PROFIT_TARGET_POINTS = 200
STOP_LOSS_POINTS = 50

# Capital and position sizing
INITIAL_CAPITAL = 100000
POSITION_SIZE_PCT = 0.1  # 10% per trade


# ======================== INDICATOR CALCULATIONS ===========================

def sma(series, period):
    """Simple Moving Average"""
    return series.rolling(window=period, min_periods=period).mean()

def rsi(series, period=14):
    """Relative Strength Index calculation"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def macd(series):
    """MACD and signal line"""
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line

def bollinger_bands(series, period=20, n_std=2):
    """Bollinger Bands"""
    sma_ = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()
    upper = sma_ + (n_std * std)
    lower = sma_ - (n_std * std)
    return lower, sma_, upper

def momentum(series, period):
    """Momentum % over specified period"""
    return 100 * (series / series.shift(period) - 1)


# ======================= FETCHING AND CACHING DATA =========================

@st.cache_data(show_spinner=False)
def fetch_history(ticker, start, end):
    """Fetch historical OHLCV data from Yahoo Finance"""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return None
        # Remove duplicate dates if any
        df = df.loc[~df.index.duplicated()]
        df = df[['Open','High','Low','Close','Volume']]
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


# ======================= CALCULATE INDICATORS =========================

def add_indicators(df):
    """Add all required technical indicators to DataFrame"""
    df = df.copy()
    df['SMA20'] = sma(df['Close'], 20)
    df['SMA50'] = sma(df['Close'], 50)
    df['SMA200'] = sma(df['Close'], 200)
    df['RSI14'] = rsi(df['Close'], 14)
    df['MACD'], df['MACD_SIGNAL'] = macd(df['Close'])
    df['BB_LOWER'], df['BB_MID'], df['BB_UPPER'] = bollinger_bands(df['Close'], 20, 2)
    df['VOL_MA20'] = sma(df['Volume'], 20)
    # This assignment produces no error: VOLUME_RATIO is a single Series
    df['VOLUME_RATIO'] = df['Volume'] / df['VOL_MA20']
    df['MOMENTUM5'] = momentum(df['Close'], 5)
    df['MOMENTUM10'] = momentum(df['Close'], 10)
    return df


# ======================= TRADING STRATEGY CONDITIONS ========================

def get_entry_conditions(row):
    """7 entry conditions dictionary"""
    return {
        'Price > 50 SMA': row['Close'] > row['SMA50'],
        '50 SMA > 200 SMA': row['SMA50'] > row['SMA200'],
        'RSI 40-65': 40 <= row['RSI14'] <= 65,
        'Volume > 1.2x avg': row['VOLUME_RATIO'] > 1.2,
        '5D Momentum > 1%': row['MOMENTUM5'] > 1,
        'Price > Lower BB': row['Close'] > row['BB_LOWER'],
        'MACD > Signal': row['MACD'] > row['MACD_SIGNAL'],
    }


def get_exit_conditions(row):
    """Exit conditions dictionary"""
    return {
        'Price < 50 SMA': row['Close'] < row['SMA50'],
        'RSI > 75': row['RSI14'] > 75,
        'RSI < 30': row['RSI14'] < 30,
        'Price < Lower BB': row['Close'] < row['BB_LOWER'],
        'MACD < Signal': row['MACD'] < row['MACD_SIGNAL'],
    }


# ======================= AI RATING SYSTEM ===============================

def ai_rating(row):
    """Calculate AI rating (0-10) with reasons"""
    score = 0
    reasons = []

    # Technical Score (4 points)
    if row['Close'] > row['SMA50']:
        score += 1
        reasons.append("Price > 50 SMA")
    if row['SMA50'] > row['SMA200']:
        score += 1
        reasons.append("50 SMA > 200 SMA")
    if 30 <= row['RSI14'] <= 70:
        score += 1
        reasons.append("RSI in 30-70 range")
    if row['VOLUME_RATIO'] > 1.2:
        score += 1
        reasons.append("Volume > 1.2x avg")

    # Momentum Score (2 points)
    if row['MOMENTUM5'] > 2:
        score += 2
        reasons.append("Strong Momentum (>2%)")
    elif row['MOMENTUM5'] > 0:
        score += 1
        reasons.append("Moderate Momentum (>0%)")

    # Signal Strength Score (2 points)
    entries_met = sum(get_entry_conditions(row).values())
    if entries_met >= 6:
        score += 2
        reasons.append("6+ Entry Conditions Met")
    elif 4 <= entries_met <= 5:
        score += 1
        reasons.append("4-5 Entry Conditions Met")

    # Risk-Reward Score (2 points)
    # Using fixed 100 pts target / 50 pts stop loss => 2:1 R:R
    rr = MAX_PROFIT_TARGET_POINTS / STOP_LOSS_POINTS
    if rr >= 2:
        score += 2
        reasons.append("Risk-Reward ≥ 2:1")
    elif rr >= 1.5:
        score += 1
        reasons.append("Risk-Reward ≥ 1.5:1")

    return min(score, 10), "; ".join(reasons)


# ======================= BACKTESTING ENGINE ===============================

def backtest(df, capital=INITIAL_CAPITAL, position_pct=POSITION_SIZE_PCT,
             sl_points=STOP_LOSS_POINTS, min_tp_points=MIN_PROFIT_TARGET_POINTS,
             max_tp_points=MAX_PROFIT_TARGET_POINTS, verbose=False):
    """
    Backtest the strategy:
    - Entry when 5 or more entry conditions met.
    - Exit when any exit condition met, or stop loss, or profit target hit.
    - Positions opened at next day's open price.
    - Stop loss and profit target based on fixed points (min-max range).
    - Position size = position_pct * cash.
    Returns trade log DataFrame and performance dict.
    """
    df = df.copy()
    trades = []
    position = 0
    entry_price = None
    entry_index = None
    cash = capital

    for i in range(50, len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        if position == 0:
            # Check entry
            entry_conds = get_entry_conditions(row)
            if sum(entry_conds.values()) >= 5:
                # Open trade at next day open price
                open_price = next_row['Open']
                max_units = int((cash * position_pct) // open_price)
                if max_units == 0:
                    continue

                position = max_units
                entry_price = open_price
                entry_index = i + 1

                # Random target points from min to max for variability
                import random
                tp_points = random.randint(min_tp_points, max_tp_points)
                stop_loss_price = entry_price - sl_points
                target_price = entry_price + tp_points

                if verbose:
                    print(f"Entry at {entry_price} on {df.index[entry_index]} units: {position}")

                continue
        else:
            # Position open, check exit conditions on next row
            exit_conds = get_exit_conditions(next_row)
            low_next = next_row['Low']
            high_next = next_row['High']

            exit_signal = any(exit_conds.values()) or \
                          (low_next <= stop_loss_price) or \
                          (high_next >= target_price) or \
                          (i + 2 == len(df))

            reason = ""
            if any(exit_conds.values()):
                reason = "Exit Signal"
            if low_next <= stop_loss_price:
                reason = "Stop Loss Hit"
            if high_next >= target_price:
                reason = "Profit Target Hit"
            if (i + 2 == len(df)):
                reason = "End of Data"

            if exit_signal:
                exit_price = next_row['Close']
                pnl = (exit_price - entry_price) * position
                ret_pct = (exit_price - entry_price) / entry_price * 100
                hold_days = (df.index[i + 1] - df.index[entry_index]).days
                cash += pnl

                trades.append({
                    'Entry Date': df.index[entry_index],
                    'Exit Date': df.index[i + 1],
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Units': position,
                    'P/L': pnl,
                    'P/L %': ret_pct,
                    'Reason': reason,
                    'Duration (days)': hold_days
                })

                if verbose:
                    print(f"Exit at {exit_price} on {df.index[i+1]} reason: {reason} P&L: {pnl}")

                position = 0
                entry_price = None
                entry_index = None

    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        wins = trades_df[trades_df['P/L'] > 0]
        losses = trades_df[trades_df['P/L'] <= 0]
        win_rate = len(wins) / len(trades_df) * 100
        total_return = (cash - capital) / capital * 100
        avg_pl = trades_df['P/L'].mean()
        max_profit = trades_df['P/L'].max()
        max_loss = trades_df['P/L'].min()
        avg_duration = trades_df['Duration (days)'].mean()
        gross_profit = wins['P/L'].sum()
        gross_loss = losses['P/L'].sum()
        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else np.nan

        perf = {
            'Trades': len(trades_df),
            'Win Rate': f"{win_rate:.1f}%",
            'Total Return (%)': f"{total_return:.1f}%",
            'Avg P/L': f"{avg_pl:.2f}",
            'Max Profit': f"{max_profit:.2f}",
            'Max Loss': f"{max_loss:.2f}",
            'Avg Duration (days)': f"{avg_duration:.1f}",
            'Profit Factor': f"{profit_factor:.2f}",
            'Final Capital': int(cash)
        }
    else:
        perf = {
            'Trades': 0,
            'Win Rate': "0%",
            'Total Return (%)': "0%",
            'Avg P/L': "0",
            'Max Profit': "0",
            'Max Loss': "0",
            'Avg Duration (days)': "0",
            'Profit Factor': "0",
            'Final Capital': int(cash)
        }

    return trades_df, perf


# ======================== VISUALIZATIONS =============================

def monthly_returns_heatmap(df):
    """Plot monthly returns heatmap using seaborn"""
    mrets = df['Close'].resample('M').last().pct_change()
    mrets = mrets.to_frame('Returns')
    mrets['Year'] = mrets.index.year
    mrets['Month'] = mrets.index.strftime('%b')
    pivot = mrets.pivot(index='Year', columns='Month', values='Returns').fillna(0)
    months_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    pivot = pivot[months_order]

    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot*100, annot=True, fmt=".1f", cmap='RdYlGn', center=0, cbar_kws={'label':'% Return'})
    plt.title("Monthly Returns Heatmap")
    plt.ylabel("Year")
    plt.xlabel("Month")
    st.pyplot(plt)

def quarterly_returns_heatmap(df):
    """Plot quarterly returns heatmap using seaborn"""
    qrets = df['Close'].resample('Q').last().pct_change()
    qrets = qrets.to_frame('Returns')
    qrets['Year'] = qrets.index.year
    qrets['Quarter'] = qrets.index.quarter
    pivot = qrets.pivot(index='Year', columns='Quarter', values='Returns').fillna(0)

    plt.figure(figsize=(7, 5))
    sns.heatmap(pivot*100, annot=True, fmt=".1f", cmap='RdYlGn', center=0, cbar_kws={'label':'% Return'})
    plt.title("Quarterly Returns Heatmap")
    plt.ylabel("Year")
    plt.xlabel("Quarter")
    st.pyplot(plt)

def plot_interactive_chart(df, trades=None):
    """Plot interactive candlestick chart with SMAs and Trade markers"""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price'))

    # Add SMA lines
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='blue', width=1), name='SMA20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='orange', width=1), name='SMA50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], line=dict(color='green', width=1), name='SMA200'))

    if trades is not None and not trades.empty:
        # Entry markers
        fig.add_trace(go.Scatter(x=trades['Entry Date'], y=trades['Entry Price'], mode='markers',
                                 marker=dict(symbol='triangle-up', color='green', size=10), name='Buy'))
        # Exit markers
        fig.add_trace(go.Scatter(x=trades['Exit Date'], y=trades['Exit Price'], mode='markers',
                                 marker=dict(symbol='triangle-down', color='red', size=10), name='Sell'))

    fig.update_layout(title='Price Chart with SMAs and Trades',
                      xaxis_title='Date', yaxis_title='Price', height=550)
    st.plotly_chart(fig, use_container_width=True)

def plot_rsi(df):
    """Plot RSI with overbought and oversold levels"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI14'], line=dict(color='purple'), name='RSI 14'))
    fig.add_hline(y=70, line_dash='dash', line_color='red', annotation_text='Overbought', annotation_position='top left')
    fig.add_hline(y=30, line_dash='dash', line_color='green', annotation_text='Oversold', annotation_position='bottom left')
    fig.update_layout(title='RSI (14)', yaxis_range=[0, 100], height=250)
    st.plotly_chart(fig, use_container_width=True)

def plot_volume(df):
    """Plot volume bars and 20-day volume moving average"""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'))
    fig.add_trace(go.Scatter(x=df.index, y=df['VOL_MA20'], line=dict(color='orange', width=2), name='Volume MA 20'))
    fig.update_layout(title='Volume', height=250)
    st.plotly_chart(fig, use_container_width=True)


# ======================== STREAMLIT APP UI ==============================

st.set_page_config(layout="wide", page_title="Nifty 50 Advanced Stock Screener")

sidebar = st.sidebar
sidebar.title("Nifty 50 Screener & Swing Trading System")

# Sidebar documentation/help 
with sidebar.expander("ℹ️ Strategy Logic & Help", expanded=True):
    st.markdown("""
    ### Strategy Logic
    - **Entry:** 5 of 7 conditions required:
      - Price > 50 SMA
      - 50 SMA > 200 SMA
      - RSI between 40-65
      - Volume > 1.2x 20-day avg
      - 5-Day Momentum > 1%
      - Price above lower Bollinger Band
      - MACD above Signal Line
    - **Exit:** Any 1 condition triggers exit:
      - Price < 50 SMA
      - RSI > 75 (overbought)
      - RSI < 30 (oversold)
      - Price < lower Bollinger Band
      - MACD below Signal Line
    - **Risk Management:**
      - Fixed stop loss: 50 points
      - Profit target: 50 to 200 points (varies per trade)
      - Position size: 10% capital
    - **AI Rating (0-10):**
      - Based on technicals, momentum, entry conditions & risk-reward
    """)

# Tabs for main modules
tab1, tab2, tab3 = st.tabs(["Complete Stock Analysis", "Live Trading Signals", "Portfolio Scanner"])

# ================== TAB 1: Complete Stock Analysis ==========================

with tab1:
    st.header("Single Stock 10-Year Analysis & Backtesting")

    stock = st.selectbox("Select Stock for Analysis", NIFTY50, index=0)

    df = fetch_history(stock, START_DATE, END_DATE)

    if df is None:
        st.warning("Failed to fetch data for the selected stock.")
    else:
        df = add_indicators(df)

        st.subheader("Monthly Returns Heatmap")
        monthly_returns_heatmap(df)

        st.subheader("Quarterly Returns Heatmap")
        quarterly_returns_heatmap(df)

        st.subheader("Backtesting with Strategy")

        trades, perf = backtest(df)

        plot_interactive_chart(df, trades)

        with st.expander("RSI and Volume Charts"):
            plot_rsi(df)
            plot_volume(df)

        st.markdown("**Backtest Performance Summary:**")
        st.table(pd.DataFrame([perf]))

        if not trades.empty:
            st.markdown("**Trade Log:**")
            st.dataframe(trades)

            csv_trades = trades.to_csv(index=False).encode('utf-8')
            st.download_button("Download Trade Log CSV", csv_trades, file_name=f"{stock}_trade_log.csv")


# ================== TAB 2: Live Trading Signals & Scanner ======================

with tab2:
    st.header("Live Trading Signals & Scanner - Nifty 50")

    scan_mode = st.radio("Select Scan Mode", ['Quick (10 Stocks)', 'Full (50 Stocks)', 'High Rating Only (8+)'], index=1)

    min_rating = st.slider("Minimum AI Rating Filter", min_value=0, max_value=10, value=0, step=1)

    if scan_mode == "Quick (10 Stocks)":
        tickers = NIFTY50[:10]
    else:
        tickers = NIFTY50

    st.info("Scanning stocks... This may take some time depending on internet speed.")

    progress_bar = st.progress(0)
    results = []

    for idx, ticker in enumerate(tickers):
        df = fetch_history(ticker, START_DATE, END_DATE)
        if df is None or df.empty:
            progress_bar.progress((idx + 1) / len(tickers))
            continue

        df = add_indicators(df)
        latest = df.iloc[-1]
        rating, reasons = ai_rating(latest)
        entry_signal_count = sum(get_entry_conditions(latest).values())
        signal = "ENTRY!" if entry_signal_count >= 5 else ""

        # Filter stocks in High Rating mode
        if scan_mode == "High Rating Only" and rating < 8:
            progress_bar.progress((idx + 1) / len(tickers))
            continue

        results.append({
            "Stock": ticker,
            "Price": round(latest['Close'], 2),
            "AI Rating": rating,
            "Entry Signals": entry_signal_count,
            "Signal": signal,
            "Volume Ratio": round(latest['VOLUME_RATIO'], 2),
            "5D Momentum (%)": round(latest['MOMENTUM5'], 2),
            "Reasons": reasons
        })
        progress_bar.progress((idx + 1) / len(tickers))

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df[results_df["AI Rating"] >= min_rating]
        results_df = results_df.sort_values(by="AI Rating", ascending=False)

        # Color code high rating rows
        def highlight_rating(val):
            color = ''
            if isinstance(val, (int, float)):
                if val >= 8:
                    color = 'background-color: #b6d7a8'  # light green
                elif val >= 6:
                    color = 'background-color: #fff2cc'  # light yellow
                elif val <= 3:
                    color = 'background-color: #f4cccc'  # light red
            return color

        st.dataframe(results_df.style.applymap(highlight_rating, subset=['AI Rating']))

        csv_scanner = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Scanner Results CSV", csv_scanner, file_name="scanner_results.csv")
    else:
        st.info("No stocks matched the criteria.")


# ================== TAB 3: Portfolio Scanner & Comparison =====================

with tab3:
    st.header("Portfolio Scanner & Multi-Stock Performance Comparison")

    selected_stocks = st.multiselect("Select Stocks for Portfolio", NIFTY50, default=NIFTY50[:5])

    capital_input = st.number_input("Initial Capital (₹)", min_value=10000, value=INITIAL_CAPITAL, step=10000)

    if st.button("Run Portfolio Backtest") and selected_stocks:
        st.info("Running backtest for portfolio...")

        portfolio_results = []
        for s in selected_stocks:
            df = fetch_history(s, START_DATE, END_DATE)
            if df is None or df.empty:
                continue
            df = add_indicators(df)
            trades, perf = backtest(df, capital=capital_input / len(selected_stocks))
            perf['Stock'] = s
            portfolio_results.append(perf)

        if portfolio_results:
            portdf = pd.DataFrame(portfolio_results)
            st.table(portdf.set_index('Stock'))

            fig = go.Figure()
            fig.add_trace(go.Bar(x=portdf['Stock'], y=portdf['Total Return (%)'].astype(float), name='Total Return (%)'))
            fig.update_layout(title="Portfolio Total Returns (%)", yaxis_title="Return %")
            st.plotly_chart(fig, use_container_width=True)

            csv_portfolio = portdf.to_csv(index=False).encode('utf-8')
            st.download_button("Download Portfolio Results CSV", csv_portfolio, file_name="portfolio_results.csv")

# =========================== END OF SCRIPT ==============================
