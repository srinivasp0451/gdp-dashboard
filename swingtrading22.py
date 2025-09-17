import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import random
import itertools

st.set_page_config(page_title="Swing Trading Recommender — Robust", layout="wide")

# ----------------------------- Utilities ---------------------------------

def infer_columns(df):
    mapping = {"date": None, "open": None, "high": None, "low": None, "close": None, "volume": None}
    for col in df.columns:
        low = str(col).lower()
        tokens = re.findall(r"[a-z]+", low)
        token_set = set(tokens)
        if mapping['date'] is None and any(t in token_set for t in ("date", "time", "timestamp", "datetime", "trade", "trade_date")):
            mapping['date'] = col
            continue
        if mapping['open'] is None and ("open" in token_set or low.strip() in ("op","openprice")):
            mapping['open'] = col
            continue
        if mapping['high'] is None and ("high" in token_set or low.strip() in ("h","hi","highprice")):
            mapping['high'] = col
            continue
        if mapping['low'] is None and ("low" in token_set or low.strip() in ("l","lo","lowprice")):
            mapping['low'] = col
            continue
        if mapping['close'] is None and ("close" in token_set or ("adj" in token_set and "close" in token_set) or low.strip() in ("c","cl","last","price","closeprice")):
            mapping['close'] = col
            continue
        if mapping['volume'] is None and any(t in token_set for t in ("volume", "vol", "qty", "quantity", "tradevolume")):
            mapping['volume'] = col
            continue
    # fallback date detection
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
    mapping = infer_columns(df)
    if mapping['date'] is None:
        raise ValueError("Could not infer a date column. Please include a date/time column.")
    df = df.copy()
    df['Date'] = pd.to_datetime(df[mapping['date']], errors='coerce')
    if df['Date'].isna().all():
        df['Date'] = pd.to_datetime(df[mapping['date']], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['Date']).reset_index(drop=True)
    # normalize to Asia/Kolkata
    try:
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


# ------------------------ Technical Indicators ---------------------------

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def atr(df, n=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ------------------------- Price action helpers --------------------------

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


def detect_patterns(df, ph, pl, params):
    price = df['Close'].values
    patterns = {'double': [], 'triple': [], 'hs': [], 'inverse_hs': [], 'engulf': [], 'triangle': [], 'flag': [], 'hammer': [], 'doji': []}
    # double
    for i in range(len(ph)-1):
        idx1, p1 = ph[i]
        idx2, p2 = ph[i+1]
        if idx2 - idx1 >= params['min_bars_between'] and abs(p1 - p2) <= params['pattern_tol'] * ((p1 + p2)/2):
            low_between = price[idx1:idx2+1].min()
            patterns['double'].append(('double_top', idx1, idx2, p1, low_between))
    for i in range(len(pl)-1):
        idx1, p1 = pl[i]
        idx2, p2 = pl[i+1]
        if idx2 - idx1 >= params['min_bars_between'] and abs(p1 - p2) <= params['pattern_tol'] * ((p1 + p2)/2):
            high_between = price[idx1:idx2+1].max()
            patterns['double'].append(('double_bottom', idx1, idx2, p1, high_between))
    # triple
    for i in range(len(ph)-2):
        idx1, p1 = ph[i]
        idx2, p2 = ph[i+1]
        idx3, p3 = ph[i+2]
        if idx2 - idx1 >= params['min_bars_between'] and idx3 - idx2 >= params['min_bars_between']:
            if abs(p1 - p2) <= params['pattern_tol'] * ((p1 + p2)/2) and abs(p2 - p3) <= params['pattern_tol'] * ((p2 + p3)/2):
                low_between = price[idx1:idx3+1].min()
                patterns['triple'].append(('triple_top', idx1, idx2, idx3, p1, p2, p3, low_between))
    for i in range(len(pl)-2):
        idx1, p1 = pl[i]
        idx2, p2 = pl[i+1]
        idx3, p3 = pl[i+2]
        if idx2 - idx1 >= params['min_bars_between'] and idx3 - idx2 >= params['min_bars_between']:
            if abs(p1 - p2) <= params['pattern_tol'] * ((p1 + p2)/2) and abs(p2 - p3) <= params['pattern_tol'] * ((p2 + p3)/2):
                high_between = price[idx1:idx3+1].max()
                patterns['triple'].append(('triple_bottom', idx1, idx2, idx3, p1, p2, p3, high_between))
    # head & shoulders
    for i in range(len(ph)-2):
        l_idx, l_price = ph[i]
        m_idx, m_price = ph[i+1]
        r_idx, r_price = ph[i+2]
        if (m_idx - l_idx >= params['min_bars_between'] and r_idx - m_idx >= params['min_bars_between']):
            shoulders_avg = (l_price + r_price) / 2
            if m_price > shoulders_avg and abs(l_price - r_price) <= params.get('hs_tol', 0.03) * shoulders_avg:
                patterns['hs'].append(('head_and_shoulders', l_idx, m_idx, r_idx, l_price, m_price, r_price))
    for i in range(len(pl)-2):
        l_idx, l_price = pl[i]
        m_idx, m_price = pl[i+1]
        r_idx, r_price = pl[i+2]
        if (m_idx - l_idx >= params['min_bars_between'] and r_idx - m_idx >= params['min_bars_between']):
            shoulders_avg = (l_price + r_price) / 2
            if m_price < shoulders_avg and abs(l_price - r_price) <= params.get('hs_tol', 0.03) * shoulders_avg:
                patterns['inverse_hs'].append(('inverse_head_and_shoulders', l_idx, m_idx, r_idx, l_price, m_price, r_price))
    # engulfing
    for i in range(1, len(df)):
        prev_o = df.loc[i-1, 'Open']
        prev_c = df.loc[i-1, 'Close']
        o = df.loc[i, 'Open']
        c = df.loc[i, 'Close']
        if prev_c < prev_o and c > o and (c - o) > (prev_o - prev_c):
            patterns['engulf'].append(('bullish_engulfing', i-1, i))
        if prev_c > prev_o and c < o and (o - c) > (prev_c - prev_o):
            patterns['engulf'].append(('bearish_engulfing', i-1, i))
    # simple candle patterns
    for i in range(len(df)):
        o = df.loc[i, 'Open']
        c = df.loc[i, 'Close']
        h = df.loc[i, 'High']
        l = df.loc[i, 'Low']
        body = abs(c - o)
        upper = h - max(c, o)
        lower = min(c, o) - l
        # hammer: small body, long lower wick
        if body < (h - l) * 0.3 and lower > 2 * body:
            patterns['hammer'].append(('hammer', i))
        # doji
        if body < (h - l) * 0.1:
            patterns['doji'].append(('doji', i))
    # triangles and flags -> simplified
    # triangles: require sequence of pivots with contracting highs/lows
    # flags handled in detect_flags previously - keep simple here
    return patterns


def build_zones(levels, width_pct=0.005):
    zones = []
    for lv in levels:
        w = lv * width_pct
        zones.append((lv - w, lv + w))
    return zones


def in_zone(price, zone):
    return price >= zone[0] and price <= zone[1]


# --------------------------- Signal generation --------------------------

def generate_robust_signals(df, params):
    # compute indicators
    df = df.copy()
    df['ema_short'] = ema(df['Close'], params.get('ema_short', 9))
    df['ema_long'] = ema(df['Close'], params.get('ema_long', 21))
    df['ema_trend'] = ema(df['Close'], params.get('ema_trend', 50))
    df['rsi'] = rsi(df['Close'], period=params.get('rsi_period', 14))
    df['atr'] = atr(df, n=params.get('atr_period', 14))

    ph, pl = get_pivots(df['Close'], left=params.get('pivot_window',3), right=params.get('pivot_window',3))
    ph_prices = [p for _, p in ph]
    pl_prices = [p for _, p in pl]
    resistances = cluster_levels(ph_prices, tol=params.get('cluster_tol',0.005))
    supports = cluster_levels(pl_prices, tol=params.get('cluster_tol',0.005))
    sup_zones = build_zones(supports, width_pct=params.get('zone_width',0.005))
    res_zones = build_zones(resistances, width_pct=params.get('zone_width',0.005))

    patterns = detect_patterns(df, ph, pl, params)

    # weights tuned conservatively (favor precision)
    weights = params.get('weights') or {
        'head_and_shoulders': -4.0,
        'inverse_head_and_shoulders': 4.0,
        'double_top': -3.0,
        'double_bottom': 3.0,
        'triple_top': -3.5,
        'triple_bottom': 3.5,
        'bullish_engulfing': 2.0,
        'bearish_engulfing': -2.0,
        'hammer': 1.8,
        'doji': 0.5,
        'breakout': 1.5,
        'support_zone': 1.2,
        'resistance_zone': -1.2,
        'upper_wick_liquidity_trap': -1.5,
        'lower_wick_liquidity_trap': 1.5
    }

    L = len(df)
    signals = [0] * L
    reasons = [''] * L

    # build quick lookup for pattern ends
    pattern_lookup = {}
    for k, v in patterns.items():
        for p in v:
            end_idx = p[-2] if len(p) >= 3 else p[1]
            pattern_lookup.setdefault(end_idx, []).append((k, p))

    vol_median = df['Volume'].median()
    if np.isnan(vol_median):
        vol_median = 0

    for i in range(L):
        score = 0.0
        reason_list = []
        close = df.loc[i, 'Close']
        ema_trend = df.loc[i, 'ema_trend']
        ema_short = df.loc[i, 'ema_short']
        ema_long = df.loc[i, 'ema_long']
        rsi_val = df.loc[i, 'rsi']
        atr_val = df.loc[i, 'atr'] if not np.isnan(df.loc[i, 'atr']) else 0

        # trend filter
        trend_long = close > ema_trend
        trend_short = close < ema_trend
        if trend_long:
            score += 0.5
            reason_list.append('trend_long')
        if trend_short:
            score -= 0.5
            reason_list.append('trend_short')

        # pattern-based score
        recent_patterns = pattern_lookup.get(i, [])
        for k, p in recent_patterns:
            if k == 'double' or k == 'triple':
                # p[0] contains type
                ptype = p[0]
                weight = weights.get(ptype.split('_')[0] + '_' + ptype.split('_')[-1], 0) if '_' in ptype else weights.get(ptype, 0)
                if weight == 0:
                    weight = weights.get(ptype, 0)
                score += weight
                reason_list.append(f"{ptype}:{weight:+.2f}")
            else:
                # for other pattern categories, use mapping
                # p[0] often string like 'head_and_shoulders' or 'bullish_engulfing'
                patname = p[0]
                w = weights.get(patname, 0)
                score += w
                reason_list.append(f"{patname}:{w:+.2f}")

        # zone proximity
        for z in sup_zones:
            if in_zone(close, z):
                score += weights['support_zone']
                reason_list.append('near_support')
        for z in res_zones:
            if in_zone(close, z):
                score += weights['resistance_zone']
                reason_list.append('near_resistance')

        # momentum confirmation
        if rsi_val is not None and not np.isnan(rsi_val):
            if rsi_val > params.get('rsi_long_thresh', 55):
                score += 0.5
                reason_list.append(f'rsi_high:{rsi_val:.1f}')
            if rsi_val < params.get('rsi_short_thresh', 45):
                score -= 0.5
                reason_list.append(f'rsi_low:{rsi_val:.1f}')

        # volume confirmation
        vol = df.loc[i, 'Volume']
        if not np.isnan(vol) and vol_median > 0:
            if vol > vol_median * params.get('volume_factor', 1.2):
                score += 0.5
                reason_list.append('volume_spike')

        # wick/liquidity trap
        o = df.loc[i, 'Open']
        h = df.loc[i, 'High']
        l = df.loc[i, 'Low']
        body = abs(close - o) + 1e-9
        upper_wick = h - max(close, o)
        lower_wick = min(close, o) - l
        if (upper_wick > params.get('wick_factor', 1.5) * body and vol > vol_median * params.get('volume_factor', 1.2)):
            score += weights['upper_wick_liquidity_trap']
            reason_list.append('upper_wick_trap')
        if (lower_wick > params.get('wick_factor', 1.5) * body and vol > vol_median * params.get('volume_factor', 1.2)):
            score += weights['lower_wick_liquidity_trap']
            reason_list.append('lower_wick_trap')

        # ATR volatility filter: prefer trades when ATR is above threshold
        if atr_val > params.get('atr_min', 0):
            if atr_val > params.get('atr_min', 0):
                score += 0.2

        # final thresholding – precision mode increases threshold
        threshold = params.get('signal_threshold', 3.0) if params.get('precision_mode', True) else params.get('signal_threshold', 2.0)
        sig = 0
        if score >= threshold and 'long' in params['allowed_dirs']:
            # require trend alignment for longs
            if trend_long or (ema_short > ema_long):
                sig = 1
        if score <= -threshold and 'short' in params['allowed_dirs']:
            if trend_short or (ema_short < ema_long):
                sig = -1

        signals[i] = sig
        reasons[i] = ';'.join(reason_list)

    df_out = df.copy()
    df_out['signal'] = signals
    df_out['reason'] = reasons
    meta = {'supports': supports, 'resistances': resistances, 'patterns': patterns}
    return df_out, meta


# --------------------------- Backtester ---------------------------------

def backtest_signals(df_signals, params):
    trades = []
    L = len(df_signals)
    i = 0
    while i < L - 1:
        row = df_signals.loc[i]
        sig = int(row['signal'])
        if sig == 0:
            i += 1
            continue
        entry_idx = i + 1 if i + 1 < L else i
        entry_price = df_signals.loc[entry_idx, 'Open'] if not pd.isna(df_signals.loc[entry_idx, 'Open']) else df_signals.loc[entry_idx, 'Close']
        if params.get('use_target_points') and params.get('target_points'):
            tp = entry_price + params['target_points'] if sig == 1 else entry_price - params['target_points']
        else:
            tp = entry_price * (1 + params['tp_pct']) if sig == 1 else entry_price * (1 - params['tp_pct'])
        sl = entry_price * (1 - params['sl_pct']) if sig == 1 else entry_price * (1 + params['sl_pct'])
        exit_price = None
        exit_idx = None
        exit_reason = None
        max_hold = params.get('max_hold', 5)
        for j in range(entry_idx, min(L, entry_idx + max_hold + 1)):
            day_high = df_signals.loc[j, 'High']
            day_low = df_signals.loc[j, 'Low']
            # conservative fill: assume SL hit before TP in same candle (safer)
            if sig == 1:
                if day_low <= sl:
                    exit_price = sl
                    exit_idx = j
                    exit_reason = 'sl'
                    break
                if day_high >= tp:
                    exit_price = tp
                    exit_idx = j
                    exit_reason = 'tp'
                    break
            else:
                if day_high >= sl:
                    exit_price = sl
                    exit_idx = j
                    exit_reason = 'sl'
                    break
                if day_low <= tp:
                    exit_price = tp
                    exit_idx = j
                    exit_reason = 'tp'
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
        stats = {'total_trades': 0, 'positive_trades': 0, 'negative_trades': 0, 'accuracy': 0.0, 'total_pnl_pct': 0.0, 'avg_pnl_pct': 0.0, 'avg_hold_days': 0.0, 'total_points': 0.0}
    else:
        stats = {
            'total_trades': len(trades_df),
            'positive_trades': (trades_df['pnl'] > 0).sum(),
            'negative_trades': (trades_df['pnl'] <= 0).sum(),
            'accuracy': float((trades_df['pnl'] > 0).mean()),
            'total_pnl_pct': float(trades_df['pnl'].sum() * 100),
            'avg_pnl_pct': float(trades_df['pnl'].mean() * 100),
            'avg_hold_days': float(trades_df['hold_days'].mean()),
            'total_points': float(trades_df['pnl_points'].sum())
        }
    return trades_df, stats


# ------------------------ Hyperparameter search -------------------------

def sample_random_params(n_samples=50, param_space=None):
    default_space = {
        'pivot_window': [2,3,4],
        'cluster_tol': [0.003,0.005,0.01],
        'zone_width': [0.003,0.005,0.01],
        'sl_pct': [0.005,0.0075,0.01],
        'tp_pct': [0.01,0.02,0.03],
        'max_hold': [3,5,7],
        'breakout_lookback': [3,5,8],
        'pattern_tol': [0.01,0.02],
        'min_bars_between': [2,3,4],
        'signal_threshold': [2.5,3.0,3.5]
    }
    space = param_space if param_space is not None else default_space
    samples = []
    keys = list(space.keys())
    for _ in range(n_samples):
        s = {k: random.choice(space[k]) for k in keys}
        samples.append(s)
    return samples


def grid_params(param_grid):
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def find_best_strategy(df_train, search_type='random', random_iters=100, grid=None, allowed_dirs=['long','short'], desired_accuracy=0.9, min_trades=5, progress_callback=None, use_points=False, target_points=None, optimize_for='accuracy'):
    best = None
    best_metric = -np.inf
    tried = 0
    if search_type == 'random':
        param_space = grid if isinstance(grid, dict) else None
        samples = sample_random_params(random_iters, param_space=param_space)
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
            'volume_factor': 1.2,
            'pattern_lookahead': 5,
            'hs_tol': 0.03,
            'signal_threshold': s.get('signal_threshold',3.0),
            'allowed_dirs': allowed_dirs,
            'use_target_points': use_points,
            'target_points': target_points
        }
        df_signals, _ = generate_robust_signals(df_train, params)
        trades_df, stats = backtest_signals(df_signals, params)
        if stats['total_trades'] < min_trades:
            metric = -np.inf
        else:
            if optimize_for == 'accuracy':
                metric = stats['accuracy']
            else:
                metric = stats['total_pnl_pct']
            # big bonus for meeting desired accuracy
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
    st.title("Swing Trading Recommender — Robust Strategy")
    st.markdown("This version prioritizes **high accuracy and precision**. Note: 90%+ accuracy is difficult to guarantee across all datasets; the tool optimizes aggressively for accuracy which reduces trade frequency.")

    with st.sidebar:
        st.header('Controls')
        upload = st.file_uploader('Upload CSV/Excel', type=['csv','xlsx','xls'])
        side_opt = st.selectbox('Side', ['both','long','short'])
        search_type = st.selectbox('Search method', ['random','grid'])
        random_iters = st.number_input('Random search iterations', min_value=10, max_value=2000, value=200)
        desired_accuracy = st.slider('Desired accuracy (win rate)', 0.5, 0.99, 0.9, step=0.01)
        min_trades = st.number_input('Minimum trades required for a strategy', min_value=1, max_value=500, value=5)
        target_points = st.number_input('Target points per trade (absolute price points)', min_value=1, value=50)
        use_points_for_backtest = st.checkbox('Use target points for backtest (instead of percent TP)', value=False)
        precision_mode = st.checkbox('Precision mode (stricter filters, fewer trades, higher accuracy)', value=True)
        optimize_for = st.selectbox('Optimize for', ['accuracy','points'])
        st.markdown('Optional position sizing')
        capital = st.number_input('Capital (0=off)', min_value=0, value=0)
        risk_pct = st.number_input('Risk % per trade', min_value=0.1, max_value=10.0, value=1.0)
        st.markdown('Grid (for grid search)')
        if search_type == 'grid':
            gw_pivot = st.multiselect('pivot_window', [2,3,4], default=[3])
            gw_sl = st.multiselect('sl_pct', [0.005,0.01,0.02], default=[0.01])
            gw_tp = st.multiselect('tp_pct', [0.01,0.02,0.03], default=[0.02])
            grid = {
                'pivot_window': gw_pivot or [3],
                'sl_pct': gw_sl or [0.01],
                'tp_pct': gw_tp or [0.02],
                'cluster_tol': [0.005],
                'zone_width': [0.005],
                'max_hold': [5],
                'signal_threshold': [2.5,3.0]
            }
        else:
            grid = None

    if upload is None:
        st.info('Please upload a CSV or Excel file to begin.')
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
    st.subheader('Mapped columns')
    st.json(mapping)
    st.subheader('Data preview')
    df_display = df.copy()
    try:
        df_display['Date'] = df_display['Date'].dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S %Z')
    except Exception:
        df_display['Date'] = df_display['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    c1,c2 = st.columns([1,1])
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
    user_end = st.date_input('Select end date for backtest', value=max_date.date(), min_value=min_date.date(), max_value=max_date.date())
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
            best, tried = find_best_strategy(df_train, search_type=search_type, random_iters=random_iters, grid=grid, allowed_dirs=allowed_dirs, desired_accuracy=desired_accuracy, min_trades=min_trades, progress_callback=progress_cb, use_points=use_points_for_backtest, target_points=target_points if use_points_for_backtest else None, optimize_for=optimize_for)
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
        df_full_signals, meta = generate_robust_signals(df_train, best['params'])
        trades_df2, stats = backtest_signals(df_full_signals, best['params'])
        rec = generate_live_recommendation(df_full_signals, best['params'], target_points=target_points if use_points_for_backtest else None, backtest_stats=best['stats'], capital=capital, risk_pct=risk_pct)
        st.subheader('Live Recommendation (based on last closed candle)')
        if rec is None:
            st.write('No valid signal at last candle. No recommendation.')
        else:
            st.json(rec)
        st.subheader('Detected patterns summary')
        st.write({k: len(v) for k, v in meta['patterns'].items()})
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


def generate_live_recommendation(df_signals, params, target_points=None, backtest_stats=None, capital=0, risk_pct=1.0):
    if df_signals.empty:
        return None
    last = df_signals.iloc[-1]
    sig = int(last['signal'])
    if sig == 0:
        return None
    entry_price = float(last['Close'])
    if params.get('use_target_points') and params.get('target_points'):
        tp = entry_price + params['target_points'] if sig == 1 else entry_price - params['target_points']
    elif target_points is not None:
        tp = entry_price + target_points if sig == 1 else entry_price - target_points
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
        'entry_price': round(entry_price, 4),
        'stop_loss': round(sl, 4),
        'target_price': round(tp, 4),
        'unit_risk': round(unit_risk, 4),
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
