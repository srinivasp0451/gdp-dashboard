import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import random
import itertools

st.set_page_config(page_title="Swing Trading Recommender", layout="wide")

# ----------------------------- Utilities ---------------------------------

def infer_columns(df):
    """Robust inference of common OHLCV and date column names using tokenized matching.
    Returns mapping for keys: date, open, high, low, close, volume (original column names or None).
    """
    mapping = {"date": None, "open": None, "high": None, "low": None, "close": None, "volume": None}

    for col in df.columns:
        low = str(col).lower()
        tokens = re.findall(r"[a-z]+", low)
        token_set = set(tokens)

        # Date detection
        if mapping['date'] is None and any(t in token_set for t in ("date", "time", "timestamp", "datetime", "trade", "trade_date")):
            mapping['date'] = col
            continue

        # Open
        if mapping['open'] is None and ("open" in token_set or low.strip() in ("op","openprice")):
            mapping['open'] = col
            continue

        # High
        if mapping['high'] is None and ("high" in token_set or low.strip() in ("h","hi","highprice")):
            mapping['high'] = col
            continue

        # Low
        if mapping['low'] is None and ("low" in token_set or low.strip() in ("l","lo","lowprice")):
            mapping['low'] = col
            continue

        # Close
        if mapping['close'] is None and ("close" in token_set or ("adj" in token_set and "close" in token_set) or low.strip() in ("c","cl","last","price","closeprice")):
            mapping['close'] = col
            continue

        # Volume
        if mapping['volume'] is None and any(t in token_set for t in ("volume", "vol", "qty", "quantity", "tradevolume")):
            mapping['volume'] = col
            continue

    # fallback: detect date-like columns by parsing
    if mapping['date'] is None:
        for col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                non_null = parsed.notna().sum()
                if non_null > len(df) * 0.6:
                    mapping['date'] = col
                    break
            except Exception:
                continue

    return mapping


def standardize_df(df):
    """Map columns, parse dates, sort ascending and return (dataframe, mapping).
    Ensures the returned DataFrame always has ['Date','Open','High','Low','Close','Volume'] and Date is tz-aware in IST.
    """
    mapping = infer_columns(df)
    if mapping['date'] is None:
        raise ValueError("Could not infer a date column. Please include a date/time column.")

    df = df.copy()
    df['Date'] = pd.to_datetime(df[mapping['date']], errors='coerce')
    if df['Date'].isna().all():
        df['Date'] = pd.to_datetime(df[mapping['date']], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['Date']).reset_index(drop=True)

    # normalize to Asia/Kolkata tz
    try:
        # if tz-naive, assume it's local IST and localize; if timezone aware, convert
        if df['Date'].dt.tz is None:
            df['Date'] = df['Date'].dt.tz_localize('Asia/Kolkata')
        else:
            df['Date'] = df['Date'].dt.tz_convert('Asia/Kolkata')
    except Exception:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize('Asia/Kolkata')

    renames = {}
    for std in ['open', 'high', 'low', 'close', 'volume']:
        if mapping.get(std) is not None:
            renames[mapping[std]] = std.capitalize()
    df = df.rename(columns=renames)

    if 'Close' not in df.columns:
        possible_price_cols = [c for c in df.columns if re.search(r"price", str(c).lower())]
        if possible_price_cols:
            df = df.rename(columns={possible_price_cols[0]: 'Close'})

    if 'Close' not in df.columns:
        candidates = [c for c in df.columns if c != 'Date']
        if candidates:
            df['Close'] = pd.to_numeric(df[candidates[0]], errors='coerce')
        else:
            raise ValueError('No price/close column could be found or inferred.')

    if 'Open' not in df.columns:
        df['Open'] = df['Close']
    if 'High' not in df.columns:
        df['High'] = df[['Open','Close']].max(axis=1)
    if 'Low' not in df.columns:
        df['Low'] = df[['Open','Close']].min(axis=1)
    if 'Volume' not in df.columns:
        df['Volume'] = np.nan

    for c in ['Open','High','Low','Close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    df = df[['Date','Open','High','Low','Close','Volume']]
    df = df.sort_values('Date').reset_index(drop=True)

    return df, mapping


# ------------------------- Price action helpers ---------------------------

def get_pivots(price_series, left=3, right=3):
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
    patterns = []
    for i in range(len(ph)-1):
        idx1, p1 = ph[i]
        idx2, p2 = ph[i+1]
        if idx2 - idx1 >= min_bars_between and abs(p1 - p2) <= tol * ((p1 + p2)/2):
            low_between = price[idx1:idx2+1].min()
            patterns.append(('double_top', idx1, idx2, p1, low_between))
    for i in range(len(pl)-1):
        idx1, p1 = pl[i]
        idx2, p2 = pl[i+1]
        if idx2 - idx1 >= min_bars_between and abs(p1 - p2) <= tol * ((p1 + p2)/2):
            high_between = price[idx1:idx2+1].max()
            patterns.append(('double_bottom', idx1, idx2, p1, high_between))
    return patterns


def detect_triple_top_bottom(ph, pl, price, tol=0.01, min_bars_between=3):
    patterns = []
    for i in range(len(ph)-2):
        idx1, p1 = ph[i]
        idx2, p2 = ph[i+1]
        idx3, p3 = ph[i+2]
        if idx2 - idx1 >= min_bars_between and idx3 - idx2 >= min_bars_between:
            if abs(p1 - p2) <= tol * ((p1 + p2)/2) and abs(p2 - p3) <= tol * ((p2 + p3)/2):
                low_between = price[idx1:idx3+1].min()
                patterns.append(('triple_top', idx1, idx2, idx3, p1, p2, p3, low_between))
    for i in range(len(pl)-2):
        idx1, p1 = pl[i]
        idx2, p2 = pl[i+1]
        idx3, p3 = pl[i+2]
        if idx2 - idx1 >= min_bars_between and idx3 - idx2 >= min_bars_between:
            if abs(p1 - p2) <= tol * ((p1 + p2)/2) and abs(p2 - p3) <= tol * ((p2 + p3)/2):
                high_between = price[idx1:idx3+1].max()
                patterns.append(('triple_bottom', idx1, idx2, idx3, p1, p2, p3, high_between))
    return patterns


def detect_head_shoulders(ph, pl, tol=0.03, min_spacing=3, max_spacing=60):
    patterns = []
    for i in range(len(ph)-2):
        l_idx, l_price = ph[i]
        m_idx, m_price = ph[i+1]
        r_idx, r_price = ph[i+2]
        if (m_idx - l_idx >= min_spacing and r_idx - m_idx >= min_spacing and m_idx - l_idx <= max_spacing and r_idx - m_idx <= max_spacing):
            shoulders_avg = (l_price + r_price) / 2
            if m_price > shoulders_avg and abs(l_price - r_price) <= tol * shoulders_avg:
                patterns.append(('head_and_shoulders', l_idx, m_idx, r_idx, l_price, m_price, r_price))
    # inverse head & shoulders based on pivot lows
    for i in range(len(pl)-2):
        l_idx, l_price = pl[i]
        m_idx, m_price = pl[i+1]
        r_idx, r_price = pl[i+2]
        if (m_idx - l_idx >= min_spacing and r_idx - m_idx >= min_spacing and m_idx - l_idx <= max_spacing and r_idx - m_idx <= max_spacing):
            shoulders_avg = (l_price + r_price) / 2
            if m_price < shoulders_avg and abs(l_price - r_price) <= tol * shoulders_avg:
                patterns.append(('inverse_head_and_shoulders', l_idx, m_idx, r_idx, l_price, m_price, r_price))
    return patterns


def detect_engulfing(df):
    patterns = []
    for i in range(1, len(df)):
        prev_o = df.loc[i-1, 'Open']
        prev_c = df.loc[i-1, 'Close']
        o = df.loc[i, 'Open']
        c = df.loc[i, 'Close']
        if prev_c < prev_o and c > o and (c - o) > (prev_o - prev_c):
            patterns.append(('bullish_engulfing', i-1, i))
        if prev_c > prev_o and c < o and (o - c) > (prev_c - prev_o):
            patterns.append(('bearish_engulfing', i-1, i))
    return patterns


def detect_triangles(ph, pl, min_points=3, tol=0.01):
    """Detect simple triangle patterns where highs form descending slope and lows form ascending slope in an interleaved manner."""
    patterns = []
    # merge pivot indices
    pivots = sorted([(i, p, 'H') for i,p in ph] + [(i,p,'L') for i,p in pl], key=lambda x: x[0])
    # sliding window
    for w in range(0, max(0, len(pivots)- (2*min_points))):
        window = pivots[w:w+2*min_points]
        highs = [p for i,p,t in window if t == 'H']
        lows = [p for i,p,t in window if t == 'L']
        if len(highs) >= min_points and len(lows) >= min_points:
            # check descending highs and ascending lows
            highs_desc = all(highs[i] >= highs[i+1] * (1 - tol) for i in range(len(highs)-1))
            lows_asc = all(lows[i] <= lows[i+1] * (1 + tol) for i in range(len(lows)-1))
            if highs_desc and lows_asc:
                start_idx = window[0][0]
                end_idx = window[-1][0]
                patterns.append(('sym_triangle', start_idx, end_idx))
    return patterns


def detect_flags(df, trend_lookback=20, flag_lookback=8, range_ratio_threshold=0.5):
    patterns = []
    if len(df) < trend_lookback + flag_lookback + 1:
        return patterns
    for i in range(trend_lookback + flag_lookback, len(df)):
        prev = df['Close'].iloc[i-trend_lookback-flag_lookback:i-flag_lookback]
        flag = df['Close'].iloc[i-flag_lookback:i]
        prev_range = prev.max() - prev.min()
        flag_range = flag.max() - flag.min()
        if prev_range > 0 and flag_range < prev_range * range_ratio_threshold:
            # breakout
            if df['Close'].iloc[i] > flag.max():
                patterns.append(('flag_breakout', i-flag_lookback, i))
            if df['Close'].iloc[i] < flag.min():
                patterns.append(('flag_breakdown', i-flag_lookback, i))
    return patterns


def build_zones(levels, width_pct=0.005):
    zones = []
    for lv in levels:
        w = lv * width_pct
        zones.append((lv - w, lv + w))
    return zones


def in_zone(price, zone):
    return price >= zone[0] and price <= zone[1]


# --------------------------- Strategy & Backtest -------------------------

def generate_signals(df, params):
    price = df['Close']
    ph, pl = get_pivots(price, left=params['pivot_window'], right=params['pivot_window'])
    ph_prices = [p for _, p in ph]
    pl_prices = [p for _, p in pl]

    resistances = cluster_levels(ph_prices, tol=params['cluster_tol'])
    supports = cluster_levels(pl_prices, tol=params['cluster_tol'])
    sup_zones = build_zones(supports, width_pct=params['zone_width'])
    res_zones = build_zones(resistances, width_pct=params['zone_width'])

    patterns_db = detect_double_top_bottom(ph, pl, price.values, tol=params['pattern_tol'], min_bars_between=params['min_bars_between'])
    patterns_triple = detect_triple_top_bottom(ph, pl, price.values, tol=params['pattern_tol'], min_bars_between=params['min_bars_between'])
    hs_patterns = detect_head_shoulders(ph, pl, tol=params.get('hs_tol',0.03), min_spacing=params.get('min_bars_between',3))
    engulf = detect_engulfing(df)
    triangles = detect_triangles(ph, pl, min_points=3, tol=params.get('triangle_tol',0.01))
    flags = detect_flags(df, trend_lookback=params.get('flag_trend_lookback',20), flag_lookback=params.get('flag_lookback',8), range_ratio_threshold=0.5)

    # weights for pattern types (positive => long, negative => short)
    weights = {
        'head_and_shoulders': -3.0,
        'inverse_head_and_shoulders': 3.0,
        'double_top': -2.0,
        'double_bottom': 2.0,
        'triple_top': -2.5,
        'triple_bottom': 2.5,
        'bullish_engulfing': 1.5,
        'bearish_engulfing': -1.5,
        'sym_triangle': 1.8,
        'flag_breakout': 1.3,
        'flag_breakdown': -1.3,
        'breakout': 1.5,
        'support_zone': 1.0,
        'resistance_zone': -1.0,
        'upper_wick_liquidity_trap': -1.2,
        'lower_wick_liquidity_trap': 1.2
    }

    signals = []
    reasons = []
    L = len(df)

    vol_median = df['Volume'].median()
    if np.isnan(vol_median):
        vol_median = 0

    # precompute quick lookup structures for patterns by end index
    pattern_by_index = {}
    def add_pattern(idx, name):
        pattern_by_index.setdefault(idx, []).append(name)

    for p in patterns_db:
        kind = p[0]
        end_idx = p[2]
        add_pattern(end_idx, kind)
    for p in patterns_triple:
        kind = p[0]
        end_idx = p[3]
        add_pattern(end_idx, kind)
    for p in hs_patterns:
        kind = p[0]
        end_idx = p[2]
        add_pattern(end_idx, kind)
    for p in engulf:
        kind = p[0]
        end_idx = p[2]
        add_pattern(end_idx, kind)
    for p in triangles:
        kind = p[0]
        end_idx = p[2]
        add_pattern(end_idx, kind)
    for p in flags:
        kind = p[0]
        end_idx = p[2]
        add_pattern(end_idx, kind)

    for i in range(L):
        score = 0.0
        reason_list = []
        close = df.loc[i, 'Close']

        # patterns completed recently
        recent_patterns = []
        for j in range(max(0, i - params['pattern_lookahead']), i+1):
            recent_patterns += pattern_by_index.get(j, [])
        for rp in recent_patterns:
            w = weights.get(rp, 0)
            score += w
            reason_list.append(f"{rp}({w:+.1f})")

        # support/resistance zones
        for z in sup_zones:
            if in_zone(close, z):
                score += weights['support_zone']
                reason_list.append(f"near_support_zone {round((z[0]+z[1])/2,2)}")
        for z in res_zones:
            if in_zone(close, z):
                score += weights['resistance_zone']
                reason_list.append(f"near_resistance_zone {round((z[0]+z[1])/2,2)}")

        # breakout detection (simple)
        look = params['breakout_lookback']
        if i > look:
            recent_high = df.loc[i-look:i-1, 'High'].max()
            recent_low = df.loc[i-look:i-1, 'Low'].min()
            if close > recent_high:
                score += weights['breakout']
                reason_list.append(f"breakout_above_{recent_high:.2f}")
            if close < recent_low:
                score += weights['breakout'] * -1
                reason_list.append(f"breakdown_below_{recent_low:.2f}")

        # wick/liquidity traps
        high = df.loc[i, 'High']
        low = df.loc[i, 'Low']
        openp = df.loc[i, 'Open']
        body = abs(close - openp) + 1e-9
        upper_wick = high - max(close, openp)
        lower_wick = min(close, openp) - low
        if (upper_wick > params['wick_factor'] * body and df.loc[i, 'Volume'] > vol_median * params['volume_factor']):
            score += weights['upper_wick_liquidity_trap']
            reason_list.append('upper_wick_liquidity_trap')
        if (lower_wick > params['wick_factor'] * body and df.loc[i, 'Volume'] > vol_median * params['volume_factor']):
            score += weights['lower_wick_liquidity_trap']
            reason_list.append('lower_wick_liquidity_trap')

        # final decision threshold
        thresh = params.get('signal_threshold', 1.0)
        sig = 1 if score >= thresh and 'long' in params['allowed_dirs'] else (-1 if score <= -thresh and 'short' in params['allowed_dirs'] else 0)

        signals.append(sig)
        reasons.append(';'.join(reason_list) if reason_list else '')

    df_signals = df.copy()
    df_signals['signal'] = signals
    df_signals['reason'] = reasons
    meta = {'supports': supports, 'resistances': resistances, 'patterns': {'double': patterns_db, 'triple': patterns_triple, 'hs': hs_patterns, 'engulf': engulf, 'triangles': triangles, 'flags': flags}}
    return df_signals, meta


def backtest_signals(df_signals, params):
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

        # stop loss remains pct-based; take profit can be pct-based or absolute points
        if params.get('use_target_points') and params.get('target_points') is not None:
            tp_points = params['target_points']
            tp = entry_price + tp_points if sig == 1 else entry_price - tp_points
        else:
            tp = entry_price * (1 + params['tp_pct']) if sig == 1 else entry_price * (1 - params['tp_pct'])

        sl = entry_price * (1 - params['sl_pct']) if sig == 1 else entry_price * (1 + params['sl_pct'])

        exit_price = None
        exit_idx = None
        exit_reason = None
        max_hold = params['max_hold']
        for j in range(entry_idx, min(L, entry_idx + max_hold + 1)):
            day_high = df_signals.loc[j, 'High']
            day_low = df_signals.loc[j, 'Low']
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
            exit_idx = min(L - 1, entry_idx + max_hold)
            exit_price = df_signals.loc[exit_idx, 'Close']
            exit_reason = 'time_exit'

        pnl = (exit_price - entry_price) / entry_price if sig == 1 else (entry_price - exit_price) / entry_price
        pnl_points = (exit_price - entry_price) if sig == 1 else (entry_price - exit_price)
        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': exit_idx,
            'entry_time': df_signals.loc[entry_idx, 'Date'],
            'exit_time': df_signals.loc[exit_idx, 'Date'],
            'direction': 'long' if sig == 1 else 'short',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_points': pnl_points,
            'hold_days': (df_signals.loc[exit_idx, 'Date'] - df_signals.loc[entry_idx, 'Date']).days,
            'reason': df_signals.loc[i, 'reason'] or exit_reason
        })
        i = exit_idx + 1 if exit_idx is not None else i + 1
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        stats = {'total_trades': 0, 'positive_trades': 0, 'negative_trades': 0, 'accuracy': 0, 'total_pnl_pct': 0, 'avg_pnl_pct': 0, 'avg_hold_days': 0, 'total_points': 0}
    else:
        stats = {
            'total_trades': len(trades_df),
            'positive_trades': (trades_df['pnl'] > 0).sum(),
            'negative_trades': (trades_df['pnl'] <= 0).sum(),
            'accuracy': (trades_df['pnl'] > 0).mean(),
            'total_pnl_pct': trades_df['pnl'].sum() * 100,
            'avg_pnl_pct': trades_df['pnl'].mean() * 100,
            'avg_hold_days': trades_df['hold_days'].mean(),
            'total_points': trades_df['pnl_points'].sum()
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
    # if user passed a grid-like dict, use it as param_space
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


def find_best_strategy(df_train, search_type='random', random_iters=50, grid=None, allowed_dirs=['long','short'], desired_accuracy=0.7, min_trades=5, progress_callback=None, use_points=False, target_points=None):
    best = None
    best_metric = -np.inf
    tried = 0
    if search_type == 'random':
        samples = sample_random_params(random_iters, param_space=grid)
    else:
        if grid is None:
            raise ValueError('Grid required for grid search')
        samples = list(grid_params(grid))

    total = len(samples)
    for s in samples:
        tried += 1
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
            'hs_tol': 0.03,
            'signal_threshold': s.get('signal_threshold', 1.0),
            'triangle_tol': s.get('triangle_tol', 0.01),
            'allowed_dirs': allowed_dirs,
            'use_target_points': use_points,
            'target_points': target_points
        }
        df_signals, _ = generate_signals(df_train, params)
        trades_df, stats = backtest_signals(df_signals, params)
        if stats['total_trades'] < min_trades:
            metric = -np.inf
        else:
            metric = stats['total_pnl_pct']
            if stats['accuracy'] >= desired_accuracy:
                metric += 1000
        if metric > best_metric:
            best_metric = metric
            best = {'params': params, 'trades': trades_df, 'stats': stats}

        if progress_callback is not None:
            try:
                progress_callback(tried, total)
            except Exception:
                pass

    return best, tried


# ----------------------------- Streamlit UI -----------------------------

def app():
    st.title("Swing Trading Recommender (Price Action + Auto Optimization) — Advanced Patterns")
    st.markdown("Upload an OHLC file (csv or xlsx). The app maps columns automatically and finds the best price-action swing strategy.")

    with st.sidebar:
        st.header('Controls')
        upload = st.file_uploader('Upload CSV/Excel', type=['csv','xlsx','xls'])
        side_opt = st.selectbox('Side', ['both','long','short'])
        search_type = st.selectbox('Search method', ['random','grid'])
        random_iters = st.number_input('Random search iterations', min_value=10, max_value=2000, value=60)
        desired_accuracy = st.slider('Desired accuracy (win rate)', 0.0, 1.0, 0.7)
        min_trades = st.number_input('Minimum trades required for a strategy', min_value=1, max_value=500, value=5)
        target_points = st.number_input('Target points per trade (absolute price points)', min_value=1, value=50)
        use_points_for_backtest = st.checkbox('Use target points for backtest (instead of percent TP)', value=False)
        capital = st.number_input('Optional capital for position sizing (0 = off)', min_value=0, value=0)
        risk_pct = st.number_input('Risk % per trade (only if capital provided)', min_value=0.1, max_value=10.0, value=1.0)
        st.markdown('---')
        st.markdown('If grid search selected, choose grid parameters (keep grid small):')
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
        st.info('Please upload a CSV or Excel file to begin. Example columns: date, open, high, low, close, volume.')
        return

    try:
        if upload.name.endswith('.csv'):
            raw = pd.read_csv(upload)
        else:
            raw = pd.read_excel(upload)
    except Exception as e:
        st.error(f'Failed to read file: {e}')
        return

    try:
        df, mapping = standardize_df(raw)
    except Exception as e:
        st.error(f'Error mapping columns: {e}')
        return

    st.subheader('Mapped columns (detected)')
    st.json(mapping)

    st.subheader('Data preview')
    df_display = df.copy()
    try:
        df_display['Date'] = df_display['Date'].dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S %Z')
    except Exception:
        df_display['Date'] = df_display['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    c1, c2 = st.columns([1,1])
    with c1:
        st.write('Top 5 rows')
        st.dataframe(df_display.head())
    with c2:
        st.write('Bottom 5 rows')
        st.dataframe(df_display.tail())

    st.write('Date range: ', df['Date'].min(), 'to', df['Date'].max())
    st.write('Price range (Close):', df['Close'].min(), 'to', df['Close'].max())

    st.subheader('Price chart')
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df['Date'], df['Close'])
    ax.set_title('Close Price')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    st.pyplot(fig)

    max_date = df['Date'].max()
    min_date = df['Date'].min()
    user_end = st.date_input('Select end date for backtest (data after end date will be excluded)', value=max_date.date(), min_value=min_date.date(), max_value=max_date.date())
    end_dt = pd.to_datetime(user_end)
    end_dt = pd.Timestamp(end_dt)
    if end_dt.tzinfo is None:
        end_dt = end_dt.tz_localize('Asia/Kolkata')
    else:
        end_dt = end_dt.tz_convert('Asia/Kolkata')
    end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    df_train = df[df['Date'] <= end_dt].reset_index(drop=True)
    st.write(f'Data used for backtest: {len(df_train)} rows up to {end_dt.date()}')

    st.subheader('Exploratory Data Analysis')
    df_train['returns'] = df_train['Close'].pct_change()
    df_train['year'] = df_train['Date'].dt.year
    df_train['month'] = df_train['Date'].dt.month

    st.write('Year vs Month returns (heatmap) — values shown as %')
    try:
        heat = df_train.pivot_table(values='returns', index='year', columns='month', aggfunc=lambda x: (x+1.0).prod()-1)
        heat_pct = heat * 100
        fig2, ax2 = plt.subplots(figsize=(10, max(4, 0.5 * len(heat.index))))
        im = ax2.imshow(heat_pct.fillna(0).values, aspect='auto', interpolation='nearest')
        ax2.set_xticks(range(len(heat_pct.columns)))
        ax2.set_xticklabels(heat_pct.columns)
        ax2.set_yticks(range(len(heat_pct.index)))
        ax2.set_yticklabels(heat_pct.index)
        ax2.set_title('Year-Month Returns (%)')
        # annotate cells
        for (j,i), val in np.ndenumerate(heat_pct.fillna(0).values):
            ax2.text(i, j, f"{val:.2f}%", ha='center', va='center', fontsize=8)
        fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        st.pyplot(fig2)
    except Exception as e:
        st.warning('Could not generate heatmap: ' + str(e))
        summary_table = df_train.groupby(['year','month'])['returns'].apply(lambda x: (x+1.0).prod()-1).unstack(fill_value=0) * 100
        st.dataframe(summary_table)

    st.subheader('100-word summary (automated)')
    summary = generate_summary(df_train)
    st.write(summary)

    st.subheader('Run Optimization')
    if st.button('Start Optimization'):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_cb(done, total):
            try:
                pct = int(done / total * 100)
            except Exception:
                pct = 0
            progress_bar.progress(pct)
            status_text.text(f"Optimization progress: {pct}% ({done}/{total})")

        with st.spinner('Searching for best strategy...'):
            allowed_dirs = []
            if side_opt in ['both','long']:
                allowed_dirs.append('long')
            if side_opt in ['both','short']:
                allowed_dirs.append('short')
            best, tried = find_best_strategy(df_train, search_type=search_type, random_iters=random_iters, grid=grid, allowed_dirs=allowed_dirs, desired_accuracy=desired_accuracy, min_trades=min_trades, progress_callback=progress_cb, use_points=use_points_for_backtest, target_points=target_points if use_points_for_backtest else None)

        progress_bar.progress(100)
        status_text.success('Optimization completed')

        if best is None:
            st.warning('No strategy found meeting the filters. Try relaxing filters or increase iterations.')
            return

        st.success('Best strategy found')
        st.write('Tried combinations:', tried)
        st.write('Strategy params:')
        st.json(best['params'])
        st.write('Backtest stats:')
        st.json(best['stats'])

        trades_df = best['trades'].copy()
        if not trades_df.empty:
            trades_df['entry_time'] = trades_df['entry_time'].dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            trades_df['exit_time'] = trades_df['exit_time'].dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S %Z')

        st.write('Sample trades (first 100)')
        st.dataframe(trades_df.head(100))

        if len(df_train) >= 2:
            buy_hold_points = df_train['Close'].iloc[-1] - df_train['Close'].iloc[0]
            buy_hold_return_pct = (df_train['Close'].iloc[-1] / df_train['Close'].iloc[0] - 1) * 100
        else:
            buy_hold_points = 0
            buy_hold_return_pct = 0

        strategy_points = best['stats'].get('total_points', 0)
        pct_more_points = (strategy_points - buy_hold_points) / (abs(buy_hold_points) if buy_hold_points != 0 else 1) * 100

        st.metric('Buy and hold points', f"{buy_hold_points:.2f}")
        st.metric('Buy and hold return %', f"{buy_hold_return_pct:.2f}%")
        st.metric('Strategy total points', f"{strategy_points:.2f}")
        st.write(f"Strategy gave {pct_more_points:.2f}% more points vs buy-and-hold (relative to absolute buy-and-hold points)")

        df_full_signals, meta = generate_signals(df_train, best['params'])
        trades_df2, stats = backtest_signals(df_full_signals, best['params'])
        rec = generate_live_recommendation(df_full_signals, best['params'], target_points=target_points, backtest_stats=best['stats'], capital=capital, risk_pct=risk_pct)

        st.subheader('Live Recommendation (based on last closed candle)')
        if rec is None:
            st.write('No valid signal at last candle. No recommendation.')
        else:
            # format any datetime fields
            st.json(rec)

        st.subheader('Detected patterns summary')
        st.write('Counts:')
        counts = {k: len(v) for k,v in meta['patterns'].items()}
        st.json(counts)

        st.subheader('Backtest Summary (human readable)')
        st.write(backtest_human_readable(best['stats'], best['params'], buy_hold_points))


# --------------------------- Helper outputs ------------------------------

def generate_summary(df):
    try:
        returns = df['returns'].dropna()
        mean_ret = returns.mean() * 252
        vol = returns.std() * np.sqrt(252)
        last_close = df['Close'].iloc[-1]
        trend = 'uptrend' if (len(df) >= 30 and df['Close'].iloc[-1] > df['Close'].iloc[-30:].mean()) else 'sideways/downtrend'
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


def generate_live_recommendation(df_signals, params, target_points=50, backtest_stats=None, capital=0, risk_pct=1.0):
    if df_signals.empty:
        return None
    last = df_signals.iloc[-1]
    sig = last['signal']
    if sig == 0:
        return None
    entry_price = last['Close']

    # TP calculation: follow params first (backtest alignment), else target_points
    if params.get('use_target_points') and params.get('target_points') is not None:
        tp = entry_price + params['target_points'] if sig == 1 else entry_price - params['target_points']
    else:
        tp = entry_price * (1 + params['tp_pct']) if sig == 1 else entry_price * (1 - params['tp_pct'])

    sl = entry_price * (1 - params['sl_pct']) if sig == 1 else entry_price * (1 + params['sl_pct'])

    unit_risk = abs(entry_price - sl)
    suggested_units = None
    if capital and capital > 0 and unit_risk > 0:
        risk_amount = capital * (risk_pct / 100.0)
        suggested_units = int(risk_amount // unit_risk)

    rec = {
        'direction': 'long' if sig == 1 else 'short',
        'entry_price': round(float(entry_price), 4),
        'stop_loss': round(float(sl), 4),
        'target_price': round(float(tp), 4),
        'unit_risk': round(float(unit_risk), 4),
        'suggested_units_by_capital': int(suggested_units) if suggested_units is not None else None,
        'reason': last['reason'] or 'signal',
        'probability_of_profit': float(backtest_stats['accuracy']) if backtest_stats and backtest_stats.get('total_trades',0) > 0 else None
    }
    return rec


def backtest_human_readable(stats, params, buy_hold_points=None):
    if stats['total_trades'] == 0:
        return 'No trades executed in backtest with these parameters.'
    s = (f"Backtest performed with sl {params['sl_pct']*100:.2f}% and tp {params['tp_pct']*100:.2f}%. "
         f"Total trades: {stats['total_trades']}. Winning trades: {stats['positive_trades']} ({stats['accuracy']*100:.2f}% win rate). "
         f"Total return from signals (sum of trade returns): {stats['total_pnl_pct']:.2f}% . ")
    if buy_hold_points is not None:
        s += f"Buy-and-hold points during period: {buy_hold_points:.2f}. Strategy total points: {stats.get('total_points',0):.2f}."
    return s


if __name__ == '__main__':
    app()
