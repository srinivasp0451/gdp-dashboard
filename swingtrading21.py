import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import random
import itertools
from io import StringIO

st.set_page_config(page_title="Swing Trading Recommender", layout="wide")

# ----------------------------- Utilities ---------------------------------

def infer_columns(df):
    """Try to find date, open, high, low, close, volume columns in any naming convention."""
    cols = {c: c.lower() for c in df.columns}
    mapping = {"date": None, "open": None, "high": None, "low": None, "close": None, "volume": None}

    for col, low in cols.items():
        if mapping["date"] is None and any(k in low for k in ["date", "time", "timestamp", "datetime", "trade_date"]):
            mapping["date"] = col
            continue
        if mapping["open"] is None and ("open" in low or "o" == low.strip()):
            mapping["open"] = col
            continue
        if mapping["high"] is None and ("high" in low or "hi" in low):
            mapping["high"] = col
            continue
        if mapping["low"] is None and ("low" in low or "lo" in low):
            mapping["low"] = col
            continue
        if mapping["close"] is None and ("close" in low or "c" == low.strip() or "adj close" in low or "closeprice" in low):
            mapping["close"] = col
            continue
        if mapping["volume"] is None and ("volume" in low or "vol" in low or "qty" in low or "trade" in low):
            mapping["volume"] = col
            continue

    # If date not found, try to parse index or first column
    if mapping["date"] is None:
        # try to find any column that parses well to datetime
        for col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            non_null = parsed.notna().sum()
            if non_null > len(df) * 0.6:  # mostly dates
                mapping["date"] = col
                break
    return mapping


def standardize_df(df):
    """Map columns, parse dates, sort ascending and return dataframe with standard column names."""
    orig_cols = df.columns.tolist()
    mapping = infer_columns(df)
    if mapping["date"] is None:
        raise ValueError("Could not infer a date column. Please make sure your file contains a date/time column.")

    df = df.copy()
    # parse date
    df['__date__'] = pd.to_datetime(df[mapping['date']], errors='coerce')
    if df['__date__'].isna().all():
        # try common formats
        df['__date__'] = pd.to_datetime(df[mapping['date']], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['__date__'])

    # rename mapped columns to standard
    renames = {}
    for std in ['open','high','low','close','volume']:
        if mapping[std] is not None:
            renames[mapping[std]] = std.capitalize()
    renames[mapping['date']] = 'Date'
    df = df.rename(columns=renames)

    # make sure OHLC exists, maybe some datasets have only price column
    if 'Close' not in df.columns and 'Price' in df.columns:
        df = df.rename(columns={'Price':'Close'})

    # create default numeric OHLC if not present
    if 'Open' not in df.columns:
        df['Open'] = df.get('Close', df.iloc[:, 1] if df.shape[1] > 1 else df.iloc[:, 0]).astype(float)
    if 'High' not in df.columns:
        df['High'] = df['Open']
    if 'Low' not in df.columns:
        df['Low'] = df['Open']
    if 'Close' not in df.columns:
        df['Close'] = df['Open']
    if 'Volume' not in df.columns:
        df['Volume'] = np.nan

    # select standard columns and date
    df = df[['Date','Open','High','Low','Close','Volume']]
    # ensure numeric
    for c in ['Open','High','Low','Close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    # sort by date ascending
    df = df.sort_values('Date').reset_index(drop=True)

    return df


# ------------------------- Price action helpers ---------------------------

def get_pivots(price_series, left=3, right=3):
    """Return indices of pivot highs and lows. A pivot high is greater than left bars and right bars."""
    ph = []
    pl = []
    s = price_series.values
    L = len(s)
    for i in range(left, L - right):
        left_slice = s[i-left:i]
        right_slice = s[i+1:i+1+right]
        if s[i] > left_slice.max() and s[i] > right_slice.max():
            ph.append((i, s[i]))
        if s[i] < left_slice.min() and s[i] < right_slice.min():
            pl.append((i, s[i]))
    return ph, pl


def cluster_levels(levels, tol=0.01):
    """Cluster price levels within tol fraction into consolidated levels."""
    if not levels:
        return []
    levels = sorted(levels)
    clusters = []
    current = [levels[0]]
    for p in levels[1:]:
        if abs(p - np.mean(current)) <= tol * np.mean(current):
            current.append(p)
        else:
            clusters.append(np.mean(current))
            current = [p]
    clusters.append(np.mean(current))
    return clusters


def detect_double_top_bottom(ph, pl, price, tol=0.01, min_bars_between=3):
    """Detect simple double top/bottom patterns from pivot points."""
    patterns = []
    # double top: two highs with similar price and a valley between
    for i in range(len(ph)-1):
        idx1, p1 = ph[i]
        idx2, p2 = ph[i+1]
        if idx2 - idx1 >= min_bars_between and abs(p1 - p2) <= tol * ((p1 + p2)/2):
            # find low between idx1 and idx2
            low_between = price[idx1:idx2+1].min()
            patterns.append(('double_top', idx1, idx2, p1, low_between))
    # double bottom
    for i in range(len(pl)-1):
        idx1, p1 = pl[i]
        idx2, p2 = pl[i+1]
        if idx2 - idx1 >= min_bars_between and abs(p1 - p2) <= tol * ((p1 + p2)/2):
            high_between = price[idx1:idx2+1].max()
            patterns.append(('double_bottom', idx1, idx2, p1, high_between))
    return patterns


def get_trend_slope_from_pivots(pivots):
    """Simple linear fit on pivots to return slope (price change per bar)."""
    if len(pivots) < 2:
        return 0
    xs = np.array([p[0] for p in pivots]).reshape(-1, 1)
    ys = np.array([p[1] for p in pivots])
    # linear regression slope
    A = np.vstack([xs.T[0], np.ones(len(xs))]).T
    m, c = np.linalg.lstsq(A, ys, rcond=None)[0]
    return m


def build_zones(levels, width_pct=0.005):
    """Return supply/demand zones (low, high) around each level using width_pct fraction."""
    zones = []
    for lv in levels:
        w = lv * width_pct
        zones.append((lv - w, lv + w))
    return zones


def in_zone(price, zone):
    return price >= zone[0] and price <= zone[1]


# --------------------------- Strategy & Backtest -------------------------

def generate_signals(df, params):
    """Generate simple price-action signals (1=long, -1=short) with reasons based on params."""
    price = df['Close']
    ph, pl = get_pivots(price, left=params['pivot_window'], right=params['pivot_window'])
    ph_prices = [p for _, p in ph]
    pl_prices = [p for _, p in pl]
    resistances = cluster_levels(ph_prices, tol=params['cluster_tol'])
    supports = cluster_levels(pl_prices, tol=params['cluster_tol'])
    sup_zones = build_zones(supports, width_pct=params['zone_width'])
    res_zones = build_zones(resistances, width_pct=params['zone_width'])

    patterns = detect_double_top_bottom(ph, pl, price.values, tol=params['pattern_tol'], min_bars_between=params['min_bars_between'])

    signals = []
    reasons = []
    L = len(df)
    for i in range(L):
        sig = 0
        reason = []
        close = df.loc[i, 'Close']
        # check patterns that end near i
        for p in patterns:
            kind = p[0]
            idx1 = p[1]
            idx2 = p[2]
            # if pattern recently completed
            if idx2 <= i and i - idx2 <= params['pattern_lookahead']:
                if kind == 'double_bottom' and ('long' in params['allowed_dirs']):
                    sig = 1
                    reason.append(f"double_bottom between {idx1} and {idx2}")
                if kind == 'double_top' and ('short' in params['allowed_dirs']):
                    sig = -1
                    reason.append(f"double_top between {idx1} and {idx2}")
        # check support/resistance zones
        for z in sup_zones:
            if in_zone(close, z) and ('long' in params['allowed_dirs']):
                # price near support -> demand zone
                sig = 1
                reason.append(f"near_support_zone {round((z[0]+z[1])/2,2)}")
        for z in res_zones:
            if in_zone(close, z) and ('short' in params['allowed_dirs']):
                sig = -1
                reason.append(f"near_resistance_zone {round((z[0]+z[1])/2,2)}")

        # breakout logic: close above recent high or below recent low
        look = params['breakout_lookback']
        if i > look:
            recent_high = df.loc[i-look:i-1, 'High'].max()
            recent_low = df.loc[i-look:i-1, 'Low'].min()
            if close > recent_high and ('long' in params['allowed_dirs']):
                sig = 1
                reason.append(f"breakout_above_{recent_high:.2f}")
            if close < recent_low and ('short' in params['allowed_dirs']):
                sig = -1
                reason.append(f"breakdown_below_{recent_low:.2f}")

        # volume spike and wick detection as trap/liquidity
        # wick sizes
        high = df.loc[i, 'High']
        low = df.loc[i, 'Low']
        openp = df.loc[i, 'Open']
        body = abs(close - openp) + 1e-9
        upper_wick = high - max(close, openp)
        lower_wick = min(close, openp) - low
        if (upper_wick > params['wick_factor'] * body and df.loc[i, 'Volume'] > df['Volume'].median() * params['volume_factor']):
            # upper wick trap -> potential short
            if 'short' in params['allowed_dirs']:
                sig = -1
                reason.append('upper_wick_liquidity_trap')
        if (lower_wick > params['wick_factor'] * body and df.loc[i, 'Volume'] > df['Volume'].median() * params['volume_factor']):
            if 'long' in params['allowed_dirs']:
                sig = 1
                reason.append('lower_wick_liquidity_trap')

        signals.append(sig)
        reasons.append(';'.join(reason) if reason else '')

    df_signals = df.copy()
    df_signals['signal'] = signals
    df_signals['reason'] = reasons
    return df_signals, {'supports': supports, 'resistances': resistances, 'patterns': patterns}


def backtest_signals(df_signals, params):
    """Simple backtester using OHLC to simulate entry on next open and exit by SL/TP or after max_hold_days."""
    trades = []
    L = len(df_signals)
    i = 0
    while i < L - 1:
        row = df_signals.loc[i]
        sig = row['signal']
        if sig == 0:
            i += 1
            continue
        entry_idx = i + 1 if i + 1 < L else i
        entry_price = df_signals.loc[entry_idx, 'Open'] if not pd.isna(df_signals.loc[entry_idx, 'Open']) else df_signals.loc[entry_idx, 'Close']
        sl = entry_price * (1 - params['sl_pct']) if sig == 1 else entry_price * (1 + params['sl_pct'])
        tp = entry_price * (1 + params['tp_pct']) if sig == 1 else entry_price * (1 - params['tp_pct'])
        exit_price = None
        exit_idx = None
        exit_reason = None
        max_hold = params['max_hold']
        for j in range(entry_idx, min(L, entry_idx + max_hold + 1)):
            day_high = df_signals.loc[j, 'High']
            day_low = df_signals.loc[j, 'Low']
            # both hit same day -> assume TP hit first (optimistic)
            if sig == 1:
                if day_high >= tp:
                    exit_price = tp
                    exit_idx = j
                    exit_reason = 'tp'
                    break
                if day_low <= sl:
                    exit_price = sl
                    exit_idx = j
                    exit_reason = 'sl'
                    break
            else:
                if day_low <= tp:
                    exit_price = tp
                    exit_idx = j
                    exit_reason = 'tp'
                    break
                if day_high >= sl:
                    exit_price = sl
                    exit_idx = j
                    exit_reason = 'sl'
                    break
        if exit_price is None:
            # exit at close after max_hold
            exit_idx = min(L - 1, entry_idx + max_hold)
            exit_price = df_signals.loc[exit_idx, 'Close']
            exit_reason = 'time_exit'

        pnl = (exit_price - entry_price) / entry_price if sig == 1 else (entry_price - exit_price) / entry_price
        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': exit_idx,
            'entry_time': df_signals.loc[entry_idx, 'Date'],
            'exit_time': df_signals.loc[exit_idx, 'Date'],
            'direction': 'long' if sig == 1 else 'short',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'hold_days': (df_signals.loc[exit_idx, 'Date'] - df_signals.loc[entry_idx, 'Date']).days,
            'reason': df_signals.loc[i, 'reason'] or exit_reason
        })
        # move index past this trade to avoid overlapping
        i = exit_idx + 1 if exit_idx is not None else i + 1
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        stats = {
            'total_trades': 0,
            'positive_trades': 0,
            'negative_trades': 0,
            'accuracy': 0,
            'total_pnl_pct': 0,
            'avg_pnl_pct': 0
        }
    else:
        stats = {
            'total_trades': len(trades_df),
            'positive_trades': (trades_df['pnl'] > 0).sum(),
            'negative_trades': (trades_df['pnl'] <= 0).sum(),
            'accuracy': (trades_df['pnl'] > 0).mean(),
            'total_pnl_pct': trades_df['pnl'].sum() * 100,
            'avg_pnl_pct': trades_df['pnl'].mean() * 100,
            'avg_hold_days': trades_df['hold_days'].mean()
        }
    return trades_df, stats


# --------------------------- Hyperparameter Search ----------------------

def sample_random_params(n_samples=50, param_space=None):
    if param_space is None:
        param_space = {
            'pivot_window': [2,3,4,5],
            'cluster_tol': [0.003, 0.005, 0.01],
            'zone_width': [0.003, 0.005, 0.01],
            'sl_pct': [0.005, 0.01, 0.02],
            'tp_pct': [0.01, 0.02, 0.05],
            'max_hold': [3,5,10],
            'breakout_lookback': [3,5,10],
            'pattern_tol': [0.01,0.02],
            'min_bars_between': [2,3,5]
        }
    samples = []
    keys = list(param_space.keys())
    for _ in range(n_samples):
        s = {k: random.choice(param_space[k]) for k in keys}
        samples.append(s)
    return samples


def grid_params(param_grid):
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def find_best_strategy(df_train, search_type='random', random_iters=50, grid=None, allowed_dirs=['long','short'], desired_accuracy=0.7, min_trades=5):
    best = None
    best_metric = -np.inf
    tried = 0
    if search_type == 'random':
        samples = sample_random_params(random_iters, param_space=grid)
    else:
        if grid is None:
            raise ValueError('Grid required for grid search')
        samples = list(grid_params(grid))

    for s in samples:
        params = {
            'pivot_window': s.get('pivot_window',3),
            'cluster_tol': s.get('cluster_tol',0.005),
            'zone_width': s.get('zone_width',0.005),
            'sl_pct': s.get('sl_pct',0.01),
            'tp_pct': s.get('tp_pct',0.02),
            'max_hold': s.get('max_hold',5),
            'breakout_lookback': s.get('breakout_lookback',5),
            'pattern_tol': s.get('pattern_tol',0.02),
            'min_bars_between': s.get('min_bars_between',3),
            'wick_factor': 1.5,
            'volume_factor': 1.5,
            'pattern_lookahead': 5,
            'cluster_tol': s.get('cluster_tol',0.005),
            'zone_width': s.get('zone_width',0.005),
            'breakout_lookback': s.get('breakout_lookback',5),
            'allowed_dirs': allowed_dirs
        }
        df_signals, _ = generate_signals(df_train, params)
        trades_df, stats = backtest_signals(df_signals, params)
        tried += 1
        # require minimum trades and optionally desired accuracy
        if stats['total_trades'] < min_trades:
            continue
        metric = stats['total_pnl_pct']  # prioritize total pnl percent
        # prefer meeting desired accuracy
        if stats['accuracy'] >= desired_accuracy:
            metric += 1000  # big bonus for meeting accuracy
        if metric > best_metric:
            best_metric = metric
            best = {'params': params, 'trades': trades_df, 'stats': stats}
    return best, tried


# ----------------------------- Streamlit UI -----------------------------

def app():
    st.title("Swing Trading Recommender (Price Action + Auto Optimization)")
    st.markdown("Upload an OHLC file (csv or xlsx). The app will map columns automatically and run an optimization to find the best price-action swing strategy.")

    with st.sidebar:
        st.header('Controls')
        upload = st.file_uploader('Upload CSV/Excel', type=['csv','xlsx','xls'])
        side_opt = st.selectbox('Side', ['both','long','short'])
        search_type = st.selectbox('Search method', ['random','grid'])
        random_iters = st.number_input('Random search iterations', min_value=10, max_value=1000, value=60)
        desired_accuracy = st.slider('Desired accuracy (win rate)', 0.0, 1.0, 0.7)
        min_trades = st.number_input('Minimum trades required for a strategy', min_value=1, max_value=100, value=5)
        capital = st.number_input('Trading capital (for position sizing recommendation)', min_value=1000, value=20000)
        st.markdown('---')
        st.markdown('If grid search selected, choose grid parameters below (small grids only):')
        if search_type == 'grid':
            gw_pivot = st.multiselect('pivot_window', [2,3,4,5], default=[3,4])
            gw_sl = st.multiselect('sl_pct', [0.005,0.01,0.02], default=[0.01])
            gw_tp = st.multiselect('tp_pct', [0.01,0.02,0.05], default=[0.02])
            grid = {
                'pivot_window': gw_pivot or [3],
                'sl_pct': gw_sl or [0.01],
                'tp_pct': gw_tp or [0.02],
                'cluster_tol': [0.005],
                'zone_width': [0.005],
                'max_hold': [5]
            }
        else:
            grid = None

    if upload is None:
        st.info('Please upload a CSV or Excel file to begin. Example columns: date, open, high, low, close, volume. Case and naming do not matter.')
        return

    # read file
    try:
        if upload.name.endswith('.csv'):
            raw = pd.read_csv(upload)
        else:
            raw = pd.read_excel(upload)
    except Exception as e:
        st.error(f'Failed to read file: {e}')
        return

    # preprocess
    try:
        df = standardize_df(raw)
    except Exception as e:
        st.error(f'Error mapping columns: {e}')
        return

    st.subheader('Data preview')
    c1, c2 = st.columns([1,1])
    with c1:
        st.write('Top 5 rows')
        st.dataframe(df.head())
    with c2:
        st.write('Bottom 5 rows')
        st.dataframe(df.tail())

    st.write('Date range: ', df['Date'].min(), 'to', df['Date'].max())
    st.write('Price range (Close):', df['Close'].min(), 'to', df['Close'].max())

    # plot raw close
    st.subheader('Price chart')
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df['Date'], df['Close'])
    ax.set_title('Close Price')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    st.pyplot(fig)

    # allow end date selection (for backtest as of certain end date)
    max_date = df['Date'].max()
    min_date = df['Date'].min()
    user_end = st.date_input('Select end date for backtest (data after end date will be excluded)', value=max_date.date(), min_value=min_date.date(), max_value=max_date.date())
    # convert user_end to datetime
    end_dt = pd.to_datetime(user_end)
    df_train = df[df['Date'] <= end_dt].reset_index(drop=True)
    st.write(f'Data used for backtest: {len(df_train)} rows up to {end_dt.date()}')

    # exploratory analysis
    st.subheader('Exploratory Data Analysis')
    df_train['returns'] = df_train['Close'].pct_change()
    # monthly heatmap of returns
    df_train['year'] = df_train['Date'].dt.year
    df_train['month'] = df_train['Date'].dt.month
    heat = df_train.pivot_table(values='returns', index='year', columns='month', aggfunc=lambda x: (x+1.0).prod()-1)
    st.write('Year vs Month returns (heatmap)')
    fig2, ax2 = plt.subplots(figsize=(8,4))
    im = ax2.imshow(heat.fillna(0).values, aspect='auto')
    ax2.set_xticks(range(len(heat.columns)))
    ax2.set_xticklabels(heat.columns)
    ax2.set_yticks(range(len(heat.index)))
    ax2.set_yticklabels(heat.index)
    ax2.set_title('Year-Month Returns')
    st.pyplot(fig2)

    # short summary of 100 words
    st.subheader('100-word summary (automated)')
    summary = generate_summary(df_train)
    st.write(summary)

    # optimization / search
    st.subheader('Run Optimization')
    if st.button('Start Optimization'):
        with st.spinner('Searching for best strategy...'):
            allowed_dirs = []
            if side_opt in ['both','long']:
                allowed_dirs.append('long')
            if side_opt in ['both','short']:
                allowed_dirs.append('short')
            best, tried = find_best_strategy(df_train, search_type=search_type, random_iters=random_iters, grid=grid, allowed_dirs=allowed_dirs, desired_accuracy=desired_accuracy, min_trades=min_trades)

        if best is None:
            st.warning('No strategy found meeting the filters (minimum trades/accuracy). Try relaxing filters or increase iterations.')
        else:
            st.success('Best strategy found')
            st.write('Tried combinations:', tried)
            st.write('Strategy params:')
            st.json(best['params'])
            st.write('Backtest stats:')
            st.json(best['stats'])
            st.write('Sample trades (first 100)')
            st.dataframe(best['trades'].head(100))

            # show live recommendation using strategy on latest data (up to user selected end date)
            df_full_signals, meta = generate_signals(df_train, best['params'])
            trades_df, stats = backtest_signals(df_full_signals, best['params'])
            rec = generate_live_recommendation(df_full_signals, best['params'], capital)

            st.subheader('Live Recommendation (based on last closed candle)')
            if rec is None:
                st.write('No valid signal at last candle. No recommendation.')
            else:
                st.write(rec)

            st.subheader('Backtest Summary (human readable)')
            st.write(backtest_human_readable(best['stats'], best['params']))


# --------------------------- Helper outputs ------------------------------

def generate_summary(df):
    # create a compact 100-wordish summary
    try:
        returns = df['returns'].dropna()
        mean_ret = returns.mean()*252
        vol = returns.std()*np.sqrt(252)
        last_close = df['Close'].iloc[-1]
        trend = 'uptrend' if df['Close'].iloc[-1] > df['Close'].iloc[-30:].mean() else 'sideways/downtrend'
        opp = []
        if mean_ret > 0.05:
            opp.append('long-biased')
        if vol > 0.2:
            opp.append('high volatility')
        summary = (f"Data from {df['Date'].min().date()} to {df['Date'].max().date()}. Latest close {last_close:.2f}. The recent trend appears {trend}. "
                   f"Annualized mean return approx {mean_ret:.2%} with volatility {vol:.2%}. Potential opportunities: {', '.join(opp) or 'range trading and breakout plays'}. "
                   f"Look for support/resistance, double bottom/top patterns and liquidity wick traps near key levels. Use disciplined SL and position sizing.")
    except Exception:
        summary = 'Could not generate automated summary.'
    return summary


def generate_live_recommendation(df_signals, params, capital=20000):
    if df_signals.empty:
        return None
    last = df_signals.iloc[-1]
    sig = last['signal']
    if sig == 0:
        return None
    # construct recommendation
    entry_price = last['Close']
    if sig == 1:
        sl = entry_price * (1 - params['sl_pct'])
        tp = entry_price * (1 + params['tp_pct'])
    else:
        sl = entry_price * (1 + params['sl_pct'])
        tp = entry_price * (1 - params['tp_pct'])
    # position sizing simple risk % 1% of capital
    risk_per_trade = 0.01 * capital
    # for long position, risk per unit = entry - sl
    unit_risk = abs(entry_price - sl)
    if unit_risk == 0:
        qty = 0
    else:
        qty = int(risk_per_trade // unit_risk)
    rec = {
        'direction': 'long' if sig == 1 else 'short',
        'entry_price': round(entry_price, 4),
        'stop_loss': round(sl, 4),
        'target_price': round(tp, 4),
        'position_size_units': qty,
        'position_size_value': round(qty * entry_price, 2),
        'risk_per_trade_value': round(min(risk_per_trade, qty*unit_risk),2),
        'reason': last['reason'] or 'signal',
        'probability_of_profit': None
    }
    # estimate probability as historical win rate of similar signals
    # quick heuristic: backtest and find last 100 similar direction trades
    # (This function assumes backtest has been run externally with same params)
    return rec


def backtest_human_readable(stats, params):
    if stats['total_trades'] == 0:
        return 'No trades executed in backtest with these parameters.'
    s = (f"Backtest performed with sl {params['sl_pct']*100:.2f}% and tp {params['tp_pct']*100:.2f}%. "
         f"Total trades: {stats['total_trades']}. Winning trades: {stats['positive_trades']} ({stats['accuracy']*100:.2f}% win rate). "
         f"Total return from signals (sum of trade returns): {stats['total_pnl_pct']:.2f}% ." )
    return s


if __name__ == '__main__':
    app()
