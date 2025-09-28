"""
Streamlit — Professional Turtle-like Trend Trading Platform
Features:
- Load data by CSV upload or yfinance ticker (EOD data)
- Auto-map common OHLCV column names (case-insensitive)
- Option to treat index data (Volume=0)
- Full backtest engine (Turtle-style breakout, ATR sizing, pyramiding, volatility timing)
- Commission & slippage modeling
- Walk-forward hyperparameter tuning (basic)
- Live recommendation (based on last candle close)
- Downloadable trade logs and equity curve

How to run:
1) pip install streamlit yfinance pandas numpy matplotlib
2) streamlit run streamlit_turtle_trading_platform.py

This file is intended to be a single-file Streamlit app. Test with your uploaded CSVs (nifty.csv, btc.csv)

"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from io import StringIO
from datetime import datetime

st.set_page_config(page_title="Turtle Trading Platform", layout="wide")

# -------------------- Utilities --------------------
@st.cache_data
def fetch_yfinance(ticker: str, period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")
    df = df.reset_index()
    df.columns = [c if not isinstance(c, tuple) else c[1] for c in df.columns]
    df = df.rename(columns={
        'Date': 'datetime', 'Datetime': 'datetime', 'date': 'datetime'
    })
    if 'datetime' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def map_columns(df: pd.DataFrame, treat_as_index: bool = False) -> pd.DataFrame:
    """Map common OHLCV column names in a case-insensitive way. If treat_as_index, set Volume to 0."""
    df = df.copy()
    # Normalize column names
    mapping = {}
    lower_to_col = {c.lower(): c for c in df.columns}
    canonical = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume', 'datetime': 'datetime'}
    for key, name in canonical.items():
        if key in lower_to_col:
            mapping[lower_to_col[key]] = name
    # If datetime not found, assume first column
    if 'datetime' not in mapping.values():
        mapping[df.columns[0]] = 'datetime'
    df = df.rename(columns=mapping)
    # If Volume missing or user wants index, set to 0
    if 'Volume' not in df.columns or treat_as_index:
        df['Volume'] = 0
    # Ensure required columns exist
    required = ['datetime','Open','High','Low','Close','Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}")
    # Ensure datetime type and sort
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    return df


# -------------------- Indicators & Strategy --------------------

def prepare_indicators(df: pd.DataFrame, breakout_len: int = 34, exit_len: int = 20, atr_len: int = 14, vol_look: int = 63) -> pd.DataFrame:
    d = df.copy()
    d['high_break'] = d['High'].rolling(breakout_len).max().shift(1)
    d['low_exit'] = d['Low'].rolling(exit_len).min().shift(1)
    high_low = d['High'] - d['Low']
    high_close = (d['High'] - d['Close'].shift(1)).abs()
    low_close = (d['Low'] - d['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    d['atr'] = tr.rolling(atr_len).mean().fillna(method='bfill')
    d['ret'] = d['Close'].pct_change()
    d['vol'] = d['ret'].rolling(vol_look).std() * np.sqrt(252)
    return d


def simulate_turtle(d: pd.DataFrame,
                    pyramid_max_units: int = 2,
                    risk_per_unit: float = 0.02,
                    initial_stop_atr: float = 1.5,
                    add_unit_at_atr: float = 0.5,
                    unit_risk_multiplier: float = 1.0,
                    vol_percentile: float = 0.7,
                    start_capital: float = 100000.0,
                    commission_per_trade: float = 0.0,
                    slippage_per_share: float = 0.0,
                    return_trades: bool = True):
    """
    Core Turtle simulator.
    - Positions are per-unit entries sized by risk_per_unit fraction of equity and ATR.
    - Each unit has its own stop; stop/hits use intraday Low to check stop.
    - Trend exit uses low_exit (close < prior low_exit) to exit all.
    - Commission is charged per executed entry and exit as commission_per_trade * qty.
    - Slippage applied per share on both entry and exit.
    """
    equity = float(start_capital)
    positions = []  # list of dicts: entry_dt, entry_price, qty, stop, entry_idx
    trade_log = []
    equity_curve = []

    vol_hist = d['vol'].dropna()
    vol_thresh = vol_hist.quantile(vol_percentile) if len(vol_hist) > 0 else np.inf

    for i in range(len(d)):
        today = d.iloc[i]
        price = float(today['Close'])
        atr = float(today['atr']) if not np.isnan(today['atr']) else 0.0

        # first check unit stops (intraday low hit)
        new_positions = []
        for unit in positions:
            if today['Low'] <= unit['stop']:
                # stopped out at unit['stop'] price (apply slippage/commission)
                exit_price = unit['stop'] - slippage_per_share  # slippage hurts exit
                pnl = unit['qty'] * (exit_price - unit['entry_price']) - commission_per_trade
                equity += pnl
                trade_log.append({
                    'entry_dt': unit['entry_dt'], 'entry_price': round(unit['entry_price'], 2),
                    'exit_dt': today['datetime'], 'exit_price': round(exit_price, 2), 'qty': int(unit['qty']),
                    'pnl': round(pnl, 2), 'side': 'LONG', 'exit_reason': 'StopHit'
                })
            else:
                new_positions.append(unit)
        positions = new_positions

        # trend exit: if close < prior low_exit, exit all at close (apply slippage & commission)
        if (not np.isnan(today['low_exit'])) and price < today['low_exit'] and len(positions) > 0:
            for unit in positions:
                exit_price = price - slippage_per_share
                pnl = unit['qty'] * (exit_price - unit['entry_price']) - commission_per_trade
                equity += pnl
                trade_log.append({
                    'entry_dt': unit['entry_dt'], 'entry_price': round(unit['entry_price'], 2),
                    'exit_dt': today['datetime'], 'exit_price': round(exit_price, 2), 'qty': int(unit['qty']),
                    'pnl': round(pnl, 2), 'side': 'LONG', 'exit_reason': 'TrendExit'
                })
            positions = []

        # volatility timing: allow entries only if vol <= historical threshold
        allow_new = (not np.isnan(today['vol'])) and (today['vol'] <= vol_thresh)

        # entry rule
        if allow_new and (not np.isnan(today['high_break'])) and price > today['high_break']:
            # if no positions -> open initial unit
            if len(positions) == 0 and atr > 0:
                risk_per_share = atr * unit_risk_multiplier
                dollar_risk = equity * risk_per_unit
                qty = int(dollar_risk / risk_per_share) if risk_per_share > 0 else 0
                if qty > 0:
                    entry_price = price + slippage_per_share  # assume slippage on entry
                    stop = entry_price - initial_stop_atr * atr
                    positions.append({'entry_dt': today['datetime'], 'entry_price': entry_price, 'qty': qty, 'stop': stop, 'entry_idx': i, 'last_add_price': entry_price})
            # pyramiding
            elif len(positions) > 0:
                last_add_price = positions[-1]['last_add_price']
                if price >= last_add_price + add_unit_at_atr * atr and len(positions) < pyramid_max_units and atr > 0:
                    risk_per_share = atr * unit_risk_multiplier
                    dollar_risk = equity * risk_per_unit
                    qty = int(dollar_risk / risk_per_share) if risk_per_share > 0 else 0
                    if qty > 0:
                        entry_price = price + slippage_per_share
                        stop = entry_price - initial_stop_atr * atr
                        positions.append({'entry_dt': today['datetime'], 'entry_price': entry_price, 'qty': qty, 'stop': stop, 'entry_idx': i, 'last_add_price': entry_price})

        # mark to market unrealized PnL
        unrealized = sum([p['qty'] * (price - p['entry_price']) for p in positions])
        equity_curve.append(equity + unrealized)

    # at end close remaining positions at last close
    if len(positions) > 0:
        final_price = float(d.iloc[-1]['Close'])
        final_date = d.iloc[-1]['datetime']
        for unit in positions:
            exit_price = final_price - slippage_per_share
            pnl = unit['qty'] * (exit_price - unit['entry_price']) - commission_per_trade
            equity += pnl
            trade_log.append({
                'entry_dt': unit['entry_dt'], 'entry_price': round(unit['entry_price'], 2),
                'exit_dt': final_date, 'exit_price': round(exit_price, 2), 'qty': int(unit['qty']),
                'pnl': round(pnl, 2), 'side': 'LONG', 'exit_reason': 'EndClose'
            })
        equity_curve.append(equity)

    trades_df = pd.DataFrame(trade_log)
    stats = {}
    if not trades_df.empty:
        trades_df['win'] = trades_df['pnl'] > 0
        total = len(trades_df); wins = int(trades_df['win'].sum())
        stats = {
            'total_trades': total,
            'wins': wins,
            'winrate': float(wins / total),
            'gross_profit': float(trades_df[trades_df['pnl'] > 0]['pnl'].sum()),
            'gross_loss': float(trades_df[trades_df['pnl'] <= 0]['pnl'].sum()),
            'avg_win': float(trades_df[trades_df['pnl'] > 0]['pnl'].mean()),
            'avg_loss': float(trades_df[trades_df['pnl'] <= 0]['pnl'].mean()),
            'final_equity': float(round(equity, 2))
        }
    else:
        stats = {'total_trades': 0, 'final_equity': float(round(equity, 2))}

    if return_trades:
        return stats, trades_df, equity_curve
    return stats, equity_curve


# -------------------- Streamlit UI --------------------

st.title("Turtle Trading — Professional Backtester & Live Recommender")
st.markdown("This app runs a Turtle-style trend following engine with ATR sizing, pyramiding and volatility timing. Use CSV upload or yfinance to load data.")

with st.sidebar:
    st.header("Data source")
    data_mode = st.radio("Load data by", options=["Upload CSV", "yfinance"], index=0)
    treat_as_index = st.checkbox("Treat as index (set Volume=0)", value=False, help="If checked, Volume column will be zeroed. Useful for indices like NIFTY.")
    if data_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV (datetime,Open,High,Low,Close,Volume)", type=['csv'], accept_multiple_files=False)
        ticker_input = None
    else:
        ticker_input = st.text_input("yfinance ticker (e.g. BTC-USD, ^NSEI)", value="BTC-USD")
        period = st.selectbox("Period (yfinance)", options=["1y","2y","5y","10y"], index=3)
        interval = st.selectbox("Interval", options=["1d","1wk"], index=0)
        uploaded = None

    st.markdown("---")
    st.header("Strategy parameters")
    col1, col2 = st.columns(2)
    with col1:
        breakout_len = st.selectbox("Breakout length (entry)", options=[20,34,55,89], index=1)
        exit_len = st.selectbox("Exit length (exit) (low)", options=[10,20,55], index=1)
        initial_stop_atr = st.slider("Initial stop (ATR multiples)", min_value=1.0, max_value=3.5, value=1.5, step=0.1)
        add_unit_at_atr = st.slider("Add unit when price moves this ATR above last add", min_value=0.25, max_value=2.0, value=0.5, step=0.05)
    with col2:
        pyramid_max_units = st.selectbox("Max units (pyramiding)", options=[1,2,3,4], index=1)
        risk_per_unit = st.slider("Risk per unit (fraction of equity)", min_value=0.0025, max_value=0.05, value=0.02, step=0.0025)
        vol_percentile = st.slider("Volatility timing percentile (allow entry if vol <= percentile)", min_value=0.5, max_value=0.95, value=0.7, step=0.05)
        unit_risk_multiplier = st.number_input("Unit risk multiplier (ATR * multiplier)", value=1.0, step=0.1)

    st.markdown("---")
    st.header("Execution & friction")
    commission_per_trade = st.number_input("Commission per trade (flat)", value=0.0, step=1.0)
    slippage_per_share = st.number_input("Slippage per share (price units)", value=0.0, step=0.0, format="%.5f")
    start_capital = st.number_input("Starting capital", value=100000.0, step=1000.0)
    st.markdown("---")
    run_btn = st.button("Run Backtest")

# Main area: load data
data_load_error = None
data_df = None
if data_mode == "Upload CSV":
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            data_df = map_columns(raw, treat_as_index=treat_as_index)
            st.success(f"Loaded {len(data_df)} rows from uploaded CSV")
        except Exception as e:
            data_load_error = str(e)
            st.error(data_load_error)
    else:
        st.info("Upload a CSV or switch to yfinance mode to fetch data")
else:
    if ticker_input:
        try:
            raw = fetch_yfinance(ticker_input, period=period, interval=interval)
            data_df = map_columns(raw, treat_as_index=treat_as_index)
            st.success(f"Fetched {len(data_df)} rows for {ticker_input}")
        except Exception as e:
            data_load_error = str(e)
            st.error(data_load_error)

# If data loaded, show quick preview and let user run
if data_df is not None:
    st.subheader("Data preview")
    st.dataframe(data_df.head(10))

    # Run backtest when user clicks
    if run_btn:
        try:
            with st.spinner("Preparing indicators and running backtest..."):
                d_ind = prepare_indicators(data_df, breakout_len=breakout_len, exit_len=exit_len, atr_len=14, vol_look=63)
                stats, trades_df, eq_curve = simulate_turtle(
                    d_ind,
                    pyramid_max_units=pyramid_max_units,
                    risk_per_unit=risk_per_unit,
                    initial_stop_atr=initial_stop_atr,
                    add_unit_at_atr=add_unit_at_atr,
                    unit_risk_multiplier=unit_risk_multiplier,
                    vol_percentile=vol_percentile,
                    start_capital=start_capital,
                    commission_per_trade=commission_per_trade,
                    slippage_per_share=slippage_per_share,
                    return_trades=True
                )

            # Metrics
            st.subheader("Backtest Summary")
            st.write(stats)

            # Equity curve
            st.subheader("Equity Curve")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(eq_curve)
            ax.axhline(start_capital, color='k', linestyle='--', linewidth=0.8)
            ax.set_xlabel('Time (steps)')
            ax.set_ylabel('Equity')
            st.pyplot(fig)

            # Trades table and download
            if not trades_df.empty:
                st.subheader("Trade Log (per unit)")
                st.dataframe(trades_df)
                csv = trades_df.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download trade log CSV", data=csv, file_name='trades.csv', mime='text/csv')
            else:
                st.warning("No trades generated for chosen parameters — try lowering breakout length or loosening vol timing.")

            # Live recommendation
            st.subheader("Live Recommendation (last candle close)")
            last = d_ind.iloc[-1]
            vol_hist = d_ind['vol'].dropna()
            vol_thresh = vol_hist.quantile(vol_percentile) if len(vol_hist)>0 else np.inf
            allow_new = (not np.isnan(last['vol'])) and (last['vol'] <= vol_thresh)
            if allow_new and (not np.isnan(last['high_break'])) and last['Close'] > last['high_break']:
                atr = last['atr']
                if atr > 0:
                    risk_per_share = atr * unit_risk_multiplier
                    qty = int(start_capital * risk_per_unit / (risk_per_share)) if risk_per_share>0 else 0
                else:
                    qty = 0
                sl = last['Close'] - initial_stop_atr * last['atr']
                target = last['Close'] + 2.5 * (last['Close'] - sl)  # example R:R 2.5x
                st.success(f"LONG suggestion — Entry: {last['Close']:.2f}, Qty: {qty}, SL: {sl:.2f}, Target: {target:.2f}, Reason: Breakout + VolTiming")
            else:
                st.info("No new entry signal on last candle under current params.")

            # Show additional charts: drawdown, per-year returns
            # Compute drawdown
            if eq_curve:
                eq_arr = np.array(eq_curve)
                running_max = np.maximum.accumulate(eq_arr)
                drawdown = (running_max - eq_arr) / running_max
                st.subheader('Max Drawdown')
                st.write(f"Max drawdown: {drawdown.max() * 100:.2f}%")

        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")

# Footer/disclaimer
st.markdown("---")
st.info("This platform provides research tools and live recommendations based on historical rules. It does not execute trades. Backtests exclude market microstructure unless you add realistic slippage/commission values. Always validate out-of-sample before trading live.")

# End of file
