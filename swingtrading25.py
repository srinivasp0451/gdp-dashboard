# streamlit_swing_recommender.py
# Single-file Streamlit app for swing trading recommendations based on uploaded OHLCV data.
# - Automatically maps messy column names
# - Performs EDA and heatmap of returns
# - Detects simple price-action features, pivot points, support/resistance, patterns
# - Runs randomized/grid search over strategy parameters (no TA libs)
# - Backtests without future data leakage: entries executed at the closing price of the signal candle
# - Produces a live recommendation (entry at last candle close) using best backtested strategy
# Note: This is a heavy app; progress bar is shown during optimization/backtest.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import io
import random
from itertools import product

# ---------------------------- Helper functions ----------------------------

def map_columns(df):
    """Map messy column names to Date, Open, High, Low, Close, Volume."""
    col_map = {"date": None, "open": None, "high": None, "low": None, "close": None, "volume": None}
    cols = list(df.columns)
    lower_cols = [c.lower() for c in cols]
    # date column: look for 'date' or 'time'
    for i, c in enumerate(lower_cols):
        if 'date' in c or 'time' in c:
            col_map['date'] = cols[i]
            break
    # If no explicit date, check index
    if col_map['date'] is None and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        cols = list(df.columns)
        lower_cols = [c.lower() for c in cols]
        for i, c in enumerate(lower_cols):
            if 'date' in c or 'time' in c:
                col_map['date'] = cols[i]
                break
    # Map OHLCV using substring matching
    for key in ['open','high','low','close','volume']:
        for i, c in enumerate(lower_cols):
            if key in c or (key=='volume' and ('vol' in c or 'share' in c or 'turnover' in c or 'qty' in c)):
                # prefer exact matches earlier
                if col_map[key] is None:
                    col_map[key] = cols[i]
    # Fallback: attempt common alternatives
    if col_map['close'] is None:
        for i, c in enumerate(lower_cols):
            if c.strip() in ['last','lastprice','closeprice','close_price','cls']:
                col_map['close'] = cols[i]
                break
    # If still missing, try using the last numeric column
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if col_map['close'] is None and len(numeric_cols) > 0:
        col_map['close'] = numeric_cols[-1]
    # Ensure mappings exist
    return col_map, df


def ensure_datetime_to_ist(df, date_col):
    """Convert date column to pandas datetime and convert to Asia/Kolkata timezone safely."""
    tz = 'Asia/Kolkata'
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if df[date_col].isnull().all():
        # try epoch
        try:
            df[date_col] = pd.to_datetime(df[date_col].astype(float), unit='s')
        except Exception:
            pass
    # Drop rows where date couldn't be parsed
    df = df.dropna(subset=[date_col])
    # If timezone-naive, assume UTC then convert to IST (safer than assuming local)
    if df[date_col].dt.tz is None:
        df[date_col] = df[date_col].dt.tz_localize('UTC').dt.tz_convert(tz)
    else:
        df[date_col] = df[date_col].dt.tz_convert(tz)
    return df


def standardize_ohlcv(df, col_map):
    """Return df with standardized columns: Date, Open, High, Low, Close, Volume"""
    df = df.copy()
    # Create columns if mapping exists
    for std, orig in col_map.items():
        if orig is None:
            if std == 'volume':
                df['Volume'] = np.nan
            else:
                df[std.capitalize() if std!='date' else 'Date'] = np.nan
        else:
            if std=='date':
                df['Date'] = df[orig]
            else:
                df[std.capitalize()] = pd.to_numeric(df[orig], errors='coerce')
    # rename to uniform
    cols_needed = ['Date','Open','High','Low','Close','Volume']
    df = df[cols_needed]
    return df


def show_basic_info(df):
    st.subheader('Data preview & basic stats')
    st.write('Top 5 rows')
    st.dataframe(df.head())
    st.write('Bottom 5 rows')
    st.dataframe(df.tail())
    st.write(f"Min date: {df['Date'].min()}  |  Max date: {df['Date'].max()}")
    st.write(f"Min price (Close): {df['Close'].min()}  |  Max price (Close): {df['Close'].max()}")


def plot_price(df):
    st.subheader('Price Chart')
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='price'))
    fig.update_layout(height=500, margin={'t':20,'b':20})
    st.plotly_chart(fig, use_container_width=True)


def plot_returns_heatmap(df):
    st.subheader('Heatmap: Year vs Month returns')
    df = df.copy()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Return'] = df['Close'].pct_change()
    pivot = df.groupby(['Year','Month'])['Return'].sum().unstack(level=0).fillna(0)
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10,4))
    im = ax.imshow(pivot.values, aspect='auto')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_title('Month (rows) vs Year (cols) - Sum of returns')
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)


def generate_summary_text(df):
    """Generate ~100 word human-readable summary of what the data says."""
    start = df['Date'].min()
    end = df['Date'].max()
    days = (end - start).days
    mean_return = df['Close'].pct_change().mean()
    vol = df['Close'].pct_change().std()
    trend = 'uptrend' if df['Close'].iloc[-1] > df['Close'].iloc[0] else 'downtrend'
    recent = df['Close'].pct_change().tail(20).mean()
    summary = (f"Data from {start.date()} to {end.date()} ({days} days). The overall price shows a {trend} over the period. "
               f"Average daily return is {mean_return:.4f} with volatility {vol:.4f}. Over the last 20 candles the average return is {recent:.4f}, "
               f"suggesting {'recent strength' if recent>0 else 'recent weakness'}. Potential opportunities include trading breakouts from recent ranges, buying at identified demand zones or selling at supply zones, and using pattern-based entries with disciplined stop-loss and profit-target management.")
    return summary

# ---------------------------- Price-action utilities ----------------------------

def rolling_pivots(df, window=5):
    """Identify pivot highs and lows (simple)."""
    highs = df['High'].values
    lows = df['Low'].values
    pivot_high = np.zeros(len(df), dtype=bool)
    pivot_low = np.zeros(len(df), dtype=bool)
    w = window
    for i in range(w, len(df)-w):
        seg_high = highs[i-w:i+w+1]
        seg_low = lows[i-w:i+w+1]
        if highs[i] == seg_high.max() and list(seg_high).count(highs[i])==1:
            pivot_high[i] = True
        if lows[i] == seg_low.min() and list(seg_low).count(lows[i])==1:
            pivot_low[i] = True
    df['pivot_high'] = pivot_high
    df['pivot_low'] = pivot_low
    return df


def get_support_resistance_zones(df, lookback=50, width_pct=0.01):
    """Build simple SR zones from recent pivot points."""
    pivots = df[(df['pivot_high']) | (df['pivot_low'])]
    zones = []
    for idx, r in pivots.iterrows():
        if r['pivot_high']:
            price = r['High']
            zones.append({'type':'supply','price':price,'low':price*(1-width_pct),'high':price*(1+width_pct),'idx':idx})
        else:
            price = r['Low']
            zones.append({'type':'demand','price':price,'low':price*(1-width_pct),'high':price*(1+width_pct),'idx':idx})
    return zones


def detect_basic_patterns(df):
    """Detect a few patterns using pivot points heuristics: double top/bottom, head&shoulders approximation."""
    patterns = []
    piv = df[df['pivot_high'] | df['pivot_low']]
    piv_idx = list(piv.index)
    for i in range(2, len(piv_idx)):
        i2 = piv_idx[i-2]
        i1 = piv_idx[i-1]
        i0 = piv_idx[i]
        # double top: two pivot_high close in price
        if df.loc[i2,'pivot_high'] and df.loc[i1,'pivot_high']:
            p1 = df.loc[i2,'High']; p2 = df.loc[i1,'High']
            if abs(p1-p2)/max(p1,p2) < 0.03:
                patterns.append({'type':'double_top','idxs':[i2,i1],'date':df.loc[i1,'Date'],'level':(p1+p2)/2})
        # double bottom
        if df.loc[i2,'pivot_low'] and df.loc[i1,'pivot_low']:
            p1 = df.loc[i2,'Low']; p2 = df.loc[i1,'Low']
            if abs(p1-p2)/max(p1,p2) < 0.03:
                patterns.append({'type':'double_bottom','idxs':[i2,i1],'date':df.loc[i1,'Date'],'level':(p1+p2)/2})
        # head and shoulders: pivot_high pivot_low pivot_high pivot_high where middle higher
        # simple check for three highs with middle higher
    # Head and shoulders rough detection (look for three highs)
    highs = piv[piv['pivot_high']]
    hi = list(highs.index)
    for j in range(2, len(hi)):
        a, b, c = hi[j-2], hi[j-1], hi[j]
        ha, hb, hc = df.loc[a,'High'], df.loc[b,'High'], df.loc[c,'High']
        if hb>ha and hb>hc and abs(ha-hc)/max(ha,hc) < 0.05:
            patterns.append({'type':'head_and_shoulders','idxs':[a,b,c],'date':df.loc[b,'Date'],'level':hb})
    return patterns

# ---------------------------- Strategy & Backtesting ----------------------------

class SimplePriceActionStrategy:
    """A parameterized strategy using breakout + volume confirmation + stop/target multipliers."""
    def __init__(self, lookback=20, breakout_pct=0.005, vol_mult=1.5, stop_mul=1.0, target_mul=2.0, max_hold=20, direction='both'):
        self.lookback = int(lookback)
        self.breakout_pct = float(breakout_pct)
        self.vol_mult = float(vol_mult)
        self.stop_mul = float(stop_mul)
        self.target_mul = float(target_mul)
        self.max_hold = int(max_hold)
        self.direction = direction

    def generate_signal(self, df, i):
        # i is index position in df (0-based). Use only data up to i inclusive.
        if i < self.lookback:
            return None
        window = df.iloc[i-self.lookback+1:i+1]
        prev_high = window['High'].max()
        prev_low = window['Low'].min()
        close = df.iloc[i]['Close']
        vol = df.iloc[i]['Volume'] if 'Volume' in df.columns else 0
        avg_vol = window['Volume'].mean() if 'Volume' in df.columns else 1
        reason = []
        # Long breakout
        if (self.direction in ('long','both')):
            if close > prev_high*(1+self.breakout_pct) and vol >= avg_vol*self.vol_mult:
                # compute stops/targets using rolling std
                std = window['Close'].pct_change().std() if window['Close'].pct_change().std() > 0 else 0.001
                sl = close*(1 - self.stop_mul*std*10)
                target = close*(1 + self.target_mul*std*10)
                reason.append(f"breakout above {prev_high:.2f} with volume spike {vol/avg_vol:.2f}x")
                return {'side':'long','entry':close,'sl':sl,'target':target,'reason':'; '.join(reason)}
        # Short breakout
        if (self.direction in ('short','both')):
            if close < prev_low*(1-self.breakout_pct) and vol >= avg_vol*self.vol_mult:
                std = window['Close'].pct_change().std() if window['Close'].pct_change().std() > 0 else 0.001
                sl = close*(1 + self.stop_mul*std*10)
                target = close*(1 - self.target_mul*std*10)
                reason.append(f"breakdown below {prev_low:.2f} with volume spike {vol/avg_vol:.2f}x")
                return {'side':'short','entry':close,'sl':sl,'target':target,'reason':'; '.join(reason)}
        return None


def run_backtest(df, strategy, start_idx=0, end_idx=None, progress_callback=None):
    """Run backtest scanning sequentially and executing trades at close of signal candle.
       Returns trades list and performance metrics."""
    trades = []
    if end_idx is None:
        end_idx = len(df)-1
    i = start_idx + strategy.lookback
    total = end_idx - i + 1
    count = 0
    while i <= end_idx:
        if progress_callback and total>0:
            progress_callback(int((count/total)*100))
        sig = strategy.generate_signal(df, i)
        if sig is not None:
            # open trade at close price on candle i
            entry_price = sig['entry']
            sl = sig['sl']
            target = sig['target']
            side = sig['side']
            entry_date = df.iloc[i]['Date']
            exit_date = None
            exit_price = None
            pnl = None
            reason = sig['reason']
            # Monitor forward candles for exit
            j = i+1
            hold = 0
            exited = False
            while j <= min(len(df)-1, end_idx) and hold < strategy.max_hold:
                high = df.iloc[j]['High']
                low = df.iloc[j]['Low']
                price_close = df.iloc[j]['Close']
                # Check target/stop
                if side=='long':
                    if low <= sl:
                        exit_price = sl
                        exit_date = df.iloc[j]['Date']
                        pnl = exit_price - entry_price
                        exited = True
                        break
                    if high >= target:
                        exit_price = target
                        exit_date = df.iloc[j]['Date']
                        pnl = exit_price - entry_price
                        exited = True
                        break
                else:
                    if high >= sl:
                        exit_price = sl
                        exit_date = df.iloc[j]['Date']
                        pnl = entry_price - exit_price
                        exited = True
                        break
                    if low <= target:
                        exit_price = target
                        exit_date = df.iloc[j]['Date']
                        pnl = entry_price - exit_price
                        exited = True
                        break
                j += 1
                hold += 1
            if not exited:
                # exit at last available close or at end_idx close
                exit_price = df.iloc[min(j, end_idx)]['Close']
                exit_date = df.iloc[min(j, end_idx)]['Date']
                pnl = (exit_price - entry_price) if side=='long' else (entry_price - exit_price)
            trades.append({'entry_date':entry_date,'entry_price':entry_price,'exit_date':exit_date,'exit_price':exit_price,'pnl':pnl,'side':side,'sl':sl,'target':target,'reason':reason})
            # move index to j to avoid overlapping trades (conservative)
            i = j
        else:
            i += 1
        count += 1
    # Metrics
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        total_pnl = trades_df['pnl'].sum()
        wins = trades_df[trades_df['pnl']>0]
        losses = trades_df[trades_df['pnl']<=0]
        accuracy = len(wins)/(len(trades_df)) if len(trades_df)>0 else 0
        summary = {'total_trades':len(trades_df),'total_pnl':total_pnl,'accuracy':accuracy,'wins':len(wins),'losses':len(losses),'avg_pnl':trades_df['pnl'].mean()}
    else:
        summary = {'total_trades':0,'total_pnl':0,'accuracy':0,'wins':0,'losses':0,'avg_pnl':0}
    if progress_callback:
        progress_callback(100)
    return trades_df, summary

# ---------------------------- Parameter search ----------------------------

def param_search(df, direction, search_type='random', n_iter=20, desired_accuracy=0.7, progress=None):
    """Search for good strategy parameters. Returns best strategy and its backtest results."""
    # Define parameter grid
    grid = {
        'lookback':[10,15,20,30,40],
        'breakout_pct':[0.002,0.003,0.005,0.007,0.01],
        'vol_mult':[1.0,1.3,1.5,2.0],
        'stop_mul':[0.5,1.0,1.5,2.0],
        'target_mul':[1.0,1.5,2.0,3.0],
        'max_hold':[5,10,20,40]
    }
    # Create list of all combos for grid
    combos = list(product(grid['lookback'], grid['breakout_pct'], grid['vol_mult'], grid['stop_mul'], grid['target_mul'], grid['max_hold']))
    random.shuffle(combos)
    if search_type=='random':
        combos = combos[:n_iter]
    best = None
    best_score = -np.inf
    total = len(combos)
    for idx, c in enumerate(combos):
        if progress:
            progress(int((idx/total)*100))
        params = {'lookback':c[0],'breakout_pct':c[1],'vol_mult':c[2],'stop_mul':c[3],'target_mul':c[4],'max_hold':c[5]}
        strat = SimplePriceActionStrategy(direction=direction, **params)
        trades, summary = run_backtest(df, strat, progress_callback=None)
        # scoring: prefer higher total_pnl and high accuracy above desired threshold
        if summary['total_trades']<=0:
            score = -1000
        else:
            # favor accuracy and pnl scaled
            acc = summary['accuracy']
            pnl = summary['total_pnl']
            score = pnl + (acc - desired_accuracy)*1000
        if score > best_score:
            best_score = score
            best = {'params':params,'summary':summary,'trades':trades}
    if progress:
        progress(100)
    return best

# ---------------------------- UI / App ----------------------------

st.set_page_config(page_title='Swing Recommender', layout='wide')
st.title('Automated Swing Trading Recommender (Streamlit)')
st.markdown('Upload OHLCV data (CSV/XLSX). The app maps messy column names automatically and runs automated optimization and backtest. No TA libs used.')

uploaded = st.file_uploader('Upload CSV or Excel', type=['csv','xls','xlsx'])

if uploaded is not None:
    # Read file
    try:
        if uploaded.type == 'text/csv' or str(uploaded).lower().endswith('.csv'):
            df_raw = pd.read_csv(uploaded)
        else:
            df_raw = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f'Failed to read file: {e}')
        st.stop()

    col_map, df_raw = map_columns(df_raw)
    # Find date column if missing: ask user to pick
    if col_map['date'] is None:
        st.warning('Could not auto-detect date column. Please select it from dropdown.')
        date_col = st.selectbox('Select date/time column', options=list(df_raw.columns))
        col_map['date'] = date_col
    # Standardize
    df = standardize_ohlcv(df_raw, col_map)
    df = ensure_datetime_to_ist(df, 'Date')
    # Sort ascending
    df = df.sort_values('Date').reset_index(drop=True)
    # Allow user to set end date for backtest
    st.sidebar.subheader('Backtest / Optimization Options')
    default_end = df['Date'].max().date()
    end_date = st.sidebar.date_input('Select end date for backtest (data after this is ignored for backtest)', value=default_end, min_value=df['Date'].min().date(), max_value=df['Date'].max().date())
    # Convert selected end_date to timezone-aware timestamp at end of day IST
    end_timestamp = pd.to_datetime(datetime.combine(end_date, datetime.max.time())).tz_localize('Asia/Kolkata')
    end_idx = df[df['Date']<=end_timestamp].index.max()
    if pd.isna(end_idx):
        st.error('Selected end date is before available data. Please choose a later date.')
        st.stop()

    # Display basic info
    show_basic_info(df)
    plot_price(df[df['Date']<=end_timestamp])
    plot_returns_heatmap(df[df['Date']<=end_timestamp])
    st.markdown('**100-word summary of dataset**')
    st.write(generate_summary_text(df[df['Date']<=end_timestamp]))

    # Options: direction, search type, desired accuracy, number of points
    direction = st.sidebar.selectbox('Select side', options=['both','long','short'], index=0)
    search_type = st.sidebar.selectbox('Search type', options=['random','grid'], index=0)
    n_points = st.sidebar.number_input('Number of parameter combinations (random/grid search size)', min_value=5, max_value=500, value=30, step=5)
    desired_accuracy = st.sidebar.slider('Desired minimum accuracy (used in scoring)', min_value=0.0, max_value=1.0, value=0.6)

    # Run optimization
    if st.sidebar.button('Run Optimization & Backtest'):
        with st.spinner('Running parameter search and backtest. This may take time depending on data size and n_points...'):
            prog = st.progress(0)
            def p(x):
                prog.progress(int(x))
            best = param_search(df[df['Date']<=end_timestamp].reset_index(drop=True), direction=direction, search_type=search_type, n_iter=n_points, desired_accuracy=desired_accuracy, progress=p)
            prog.empty()
        if best is None:
            st.error('No valid strategy found.')
        else:
            st.success('Optimization completed')
            st.subheader('Best Strategy Summary')
            st.write(best['params'])
            st.write('Backtest summary:')
            st.write(best['summary'])
            st.subheader('Trades (backtest)')
            if not best['trades'].empty:
                trades_disp = best['trades'].copy()
                trades_disp['entry_date'] = trades_disp['entry_date'].astype(str)
                trades_disp['exit_date'] = trades_disp['exit_date'].astype(str)
                st.dataframe(trades_disp)
            else:
                st.write('No trades generated by best strategy on the chosen horizon.')

            # Human readable analysis of trades
            total_pnl = best['summary']['total_pnl']
            accuracy = best['summary']['accuracy']
            if accuracy >= desired_accuracy:
                verdict = 'Meets desired accuracy'
            else:
                verdict = 'Below desired accuracy'
            st.markdown(f"**Backtest verdict:** {verdict}. Total PnL: {total_pnl:.2f}. Accuracy: {accuracy:.2%}.")
            # Live recommendation on last available candle using best strategy
            st.subheader('Live Recommendation (using best strategy, executed at last candle close)')
            best_strat = SimplePriceActionStrategy(direction=direction, **best['params'])
            last_i = len(df)-1
            sig = best_strat.generate_signal(df, last_i)
            if sig is None:
                st.write('No live signal on last candle based on best strategy.')
            else:
                # Estimate probability of profit by comparing with historical similar signals
                # Find historical signals with same sign and similar breakout_pct using a sliding window
                hist_signals = []
                for i in range(best_strat.lookback, last_i):
                    s = best_strat.generate_signal(df, i)
                    if s and s['side']==sig['side']:
                        hist_signals.append(s)
                hist_df = pd.DataFrame(hist_signals)
                prob = 0.0
                if not hist_df.empty:
                    # approximate: historical win rate from trades dataframe
                    trades_df = best['trades']
                    if not trades_df.empty:
                        prob = len(trades_df[trades_df['pnl']>0])/len(trades_df)
                st.write(f"Signal side: {sig['side'].upper()}")
                st.write(f"Entry (last close): {sig['entry']:.4f}")
                st.write(f"Suggested Stop-loss: {sig['sl']:.4f}")
                st.write(f"Suggested Target: {sig['target']:.4f}")
                st.write(f"Estimated probability of profit (historical): {prob:.2%}")
                st.write('Reason / logic:')
                st.write(sig['reason'])

            # Final human-readable summary
            st.subheader('Conclusion (human readable)')
            conclusion = (f"Backtest using optimized parameters found total trades {best['summary']['total_trades']} with accuracy {best['summary']['accuracy']:.2%} and total PnL {best['summary']['total_pnl']:.2f}. "
                          f"Live recommendation: {('No signal' if sig is None else f'signal {sig["side"]} at {sig["entry"]:.2f} SL {sig["sl"]:.2f} TG {sig["target"]:.2f} with est. prob {prob:.2%}')}.")
            st.write(conclusion)

    st.sidebar.markdown('---')
    st.sidebar.write('Notes: Entries are placed at the close of the signal candle (no future data leakage). Date/times are converted to IST. This app uses heuristic pattern detection and simple price-action rules; further customization may improve results.')

else:
    st.info('Upload a CSV or Excel file containing OHLCV data to begin.')


# End of file
