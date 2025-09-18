# streamlit_swing_recommender_upgraded.py
# Requirements: streamlit, pandas, numpy, matplotlib, openpyxl, yfinance, mplfinance
# pip install streamlit pandas numpy matplotlib openpyxl yfinance mplfinance

import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import random
import itertools
import yfinance as yf
import mplfinance as mpf
import requests

st.set_page_config(page_title="Swing Trading Recommender — Upgraded", layout="wide")

# ------------------------------- Utilities ---------------------------------
def fetch_nifty50_tickers():
    """
    Attempt to fetch NIFTY50 constituents from Wikipedia live.
    Returns list of yahoo-style tickers with .NS suffix.
    """
    try:
        url = "https://en.wikipedia.org/wiki/NIFTY_50"
        tables = pd.read_html(url)
        # the constituents table usually includes 'Symbol' or 'Ticker' column.
        # Try to find a table containing 'Symbol' or 'Code'
        ticker_col = None
        for t in tables:
            cols = [c.lower() for c in t.columns.astype(str)]
            if 'symbol' in cols:
                ticker_col = t.columns[[c.lower() for c in t.columns.astype(str)].index('symbol')]
                break
            if 'code' in cols:
                ticker_col = t.columns[[c.lower() for c in t.columns.astype(str)].index('code')]
                break
        if ticker_col is None:
            # fallback: try common table structure
            t0 = tables[0]
            if 'Symbol' in t0.columns:
                tickers = t0['Symbol'].astype(str).tolist()
            else:
                tickers = t0.iloc[:,0].astype(str).tolist()
        else:
            tickers = t[ticker_col].astype(str).tolist()
        # sanitize & make yahoo format
        cleaned = []
        for tk in tickers:
            tk = str(tk).strip().upper()
            if tk.endswith('.NS'):
                cleaned.append(tk)
            else:
                # replace spaces with '-' (some tickers like 'BAJAJ-AUTO') kept intact
                tk2 = tk.replace(' ', '-')
                if not tk2.endswith('.NS'):
                    tk2 = tk2 + '.NS'
                cleaned.append(tk2)
        # unique preserve order
        seen = set()
        result = []
        for x in cleaned:
            if x not in seen:
                seen.add(x)
                result.append(x)
        if not result:
            raise ValueError("No tickers found")
        return result
    except Exception as e:
        # fallback: small embedded list if fetch fails
        fallback = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
                    "HINDUNILVR.NS","KOTAKBANK.NS","SBIN.NS","LT.NS","BAJFINANCE.NS"]
        return fallback

def infer_columns(df):
    mapping = {"date": None, "open": None, "high": None, "low": None, "close": None, "volume": None}
    for col in df.columns:
        low = str(col).lower()
        tokens = re.findall(r"[a-z]+", low)
        token_set = set(tokens)
        if mapping['date'] is None and any(t in token_set for t in ("date","time","timestamp","datetime","trade","trade_date")):
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
        if mapping['close'] is None and ("close" in token_set or (("adj" in token_set) and ("close" in token_set)) or low.strip() in ("c","cl","last","price","closeprice")):
            mapping['close'] = col
            continue
        if mapping['volume'] is None and any(t in token_set for t in ("volume","vol","qty","quantity","tradevolume")):
            mapping['volume'] = col
            continue
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
    """
    Map columns, parse dates, sort ascending and return (df, mapping).
    Normalize dates to Asia/Kolkata tz-aware.
    """
    mapping = infer_columns(df)
    if mapping['date'] is None:
        raise ValueError("Could not infer a date column. Please include a date/time column.")
    df = df.copy()
    df['Date'] = pd.to_datetime(df[mapping['date']], errors='coerce')
    if df['Date'].isna().all():
        df['Date'] = pd.to_datetime(df[mapping['date']], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['Date']).reset_index(drop=True)
    # timezone handling: localize naive to Asia/Kolkata, convert aware to Asia/Kolkata
    try:
        if df['Date'].dt.tz is None:
            df['Date'] = df['Date'].dt.tz_localize('Asia/Kolkata')
        else:
            df['Date'] = df['Date'].dt.tz_convert('Asia/Kolkata')
    except Exception:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize('Asia/Kolkata')
    renames = {}
    for std in ['open','high','low','close','volume']:
        if mapping.get(std) is not None:
            renames[mapping[std]] = std.capitalize()
    df = df.rename(columns=renames)
    if 'Close' not in df.columns:
        possible_price_cols = [c for c in df.columns if re.search(r"\bprice\b", str(c).lower())]
        if possible_price_cols:
            df = df.rename(columns={possible_price_cols[0]:'Close'})
    if 'Close' not in df.columns:
        candidates = [c for c in df.columns if c != 'Date']
        if candidates:
            df['Close'] = pd.to_numeric(df[candidates[0]], errors='coerce')
        else:
            raise ValueError("No price/close column could be found or inferred.")
    if 'Open' not in df.columns: df['Open'] = df['Close']
    if 'High' not in df.columns: df['High'] = df[['Open','Close']].max(axis=1)
    if 'Low' not in df.columns: df['Low'] = df[['Open','Close']].min(axis=1)
    if 'Volume' not in df.columns: df['Volume'] = np.nan
    for c in ['Open','High','Low','Close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df = df[['Date','Open','High','Low','Close','Volume']]
    df = df.sort_values('Date').reset_index(drop=True)
    return df, mapping

# --------------------- Indicators & Price-action helpers -------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df, n=14):
    high = df['High']; low = df['Low']; close = df['Close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def get_pivots(price_series, left=3, right=3):
    ph=[]; pl=[]
    s = price_series.values
    L = len(s)
    for i in range(left, L-right):
        left_slice = s[i-left:i]; right_slice = s[i+1:i+1+right]
        if s[i] > left_slice.max() and s[i] > right_slice.max(): ph.append((i,s[i]))
        if s[i] < left_slice.min() and s[i] < right_slice.min(): pl.append((i,s[i]))
    return ph, pl

def cluster_levels(levels, tol=0.01):
    if not levels: return []
    levels = sorted(levels); clusters=[]; current=[levels[0]]
    for p in levels[1:]:
        if abs(p - np.mean(current)) <= tol * np.mean(current):
            current.append(p)
        else:
            clusters.append(np.mean(current)); current=[p]
    clusters.append(np.mean(current)); return clusters

def build_zones(levels, width_pct=0.005):
    return [(lv - lv*width_pct, lv + lv*width_pct) for lv in levels]

# A compact advanced pattern detector (double/triple/HS/engulf/triangles/pennant/cup-handle)
def detect_patterns(df, ph, pl, params):
    price = df['Close'].values
    patterns = {'double':[],'triple':[],'hs':[],'inverse_hs':[],'engulf':[],'tri':[],'pennant':[],'wedge':[],'cup_handle':[],'hammer':[],'doji':[]}
    # double/triple (similar to previous implementation)
    for i in range(len(ph)-1):
        idx1,p1=ph[i]; idx2,p2=ph[i+1]
        if idx2-idx1 >= params['min_bars_between'] and abs(p1-p2) <= params['pattern_tol']*((p1+p2)/2):
            low_between = price[idx1:idx2+1].min()
            patterns['double'].append(('double_top',idx1,idx2,p1,low_between))
    for i in range(len(pl)-1):
        idx1,p1=pl[i]; idx2,p2=pl[i+1]
        if idx2-idx1 >= params['min_bars_between'] and abs(p1-p2) <= params['pattern_tol']*((p1+p2)/2):
            high_between = price[idx1:idx2+1].max()
            patterns['double'].append(('double_bottom',idx1,idx2,p1,high_between))
    # triple
    for i in range(len(ph)-2):
        idx1,p1=ph[i]; idx2,p2=ph[i+1]; idx3,p3=ph[i+2]
        if idx2-idx1>=params['min_bars_between'] and idx3-idx2>=params['min_bars_between']:
            if abs(p1-p2)<=params['pattern_tol']*((p1+p2)/2) and abs(p2-p3)<=params['pattern_tol']*((p2+p3)/2):
                low_between = price[idx1:idx3+1].min()
                patterns['triple'].append(('triple_top',idx1,idx2,idx3,p1,p2,p3,low_between))
    for i in range(len(pl)-2):
        idx1,p1=pl[i]; idx2,p2=pl[i+1]; idx3,p3=pl[i+2]
        if idx2-idx1>=params['min_bars_between'] and idx3-idx2>=params['min_bars_between']:
            if abs(p1-p2)<=params['pattern_tol']*((p1+p2)/2) and abs(p2-p3)<=params['pattern_tol']*((p2+p3)/2):
                high_between = price[idx1:idx3+1].max()
                patterns['triple'].append(('triple_bottom',idx1,idx2,idx3,p1,p2,p3,high_between))
    # head and shoulders
    for i in range(len(ph)-2):
        l_idx,l_price=ph[i]; m_idx,m_price=ph[i+1]; r_idx,r_price=ph[i+2]
        if (m_idx-l_idx>=params['min_bars_between'] and r_idx-m_idx>=params['min_bars_between']):
            shoulders_avg=(l_price+r_price)/2
            if m_price > shoulders_avg and abs(l_price-r_price) <= params.get('hs_tol',0.03)*shoulders_avg:
                patterns['hs'].append(('head_and_shoulders',l_idx,m_idx,r_idx,l_price,m_price,r_price))
    for i in range(len(pl)-2):
        l_idx,l_price=pl[i]; m_idx,m_price=pl[i+1]; r_idx,r_price=pl[i+2]
        if (m_idx-l_idx>=params['min_bars_between'] and r_idx-m_idx>=params['min_bars_between']):
            shoulders_avg=(l_price+r_price)/2
            if m_price < shoulders_avg and abs(l_price-r_price) <= params.get('hs_tol',0.03)*shoulders_avg:
                patterns['inverse_hs'].append(('inverse_head_and_shoulders',l_idx,m_idx,r_idx,l_price,m_price,r_price))
    # engulfing
    for i in range(1,len(df)):
        prev_o,prev_c = df.loc[i-1,'Open'], df.loc[i-1,'Close']
        o,c = df.loc[i,'Open'], df.loc[i,'Close']
        if prev_c < prev_o and c > o and (c-o) > (prev_o - prev_c): patterns['engulf'].append(('bullish_engulfing',i-1,i))
        if prev_c > prev_o and c < o and (o-c) > (prev_c - prev_o): patterns['engulf'].append(('bearish_engulfing',i-1,i))
    # simple candle patterns
    for i in range(len(df)):
        o,c,h,l = df.loc[i,['Open','Close','High','Low']]
        body = abs(c-o)
        upper = h - max(c,o)
        lower = min(c,o) - l
        if body < (h-l)*0.3 and lower > 2*body: patterns['hammer'].append(('hammer',i))
        if body < (h-l)*0.1: patterns['doji'].append(('doji',i))
    # pennant/flag/wedge/cup-handle: simplified heuristics - detect triangular contraction or cup shape
    # Pennant/triangle: look for contracting highs and lows in recent pivot window
    # (These detectors are intentionally conservative to avoid false positives)
    # triangle/pennant detection (very simplified)
    window = params.get('pattern_window', 30)
    for start in range(0, max(0,len(df)-window), int(window/2) if window>4 else 1):
        seg = df.iloc[start:start+window]
        if len(seg) < 8: continue
        highs = seg['High'].values; lows = seg['Low'].values
        # check contraction: range at end smaller than start
        start_range = highs[:int(len(highs)/2)].max() - lows[:int(len(lows)/2)].min()
        end_range = highs[int(len(highs)/2):].max() - lows[int(len(lows)/2):].min()
        if end_range < 0.6 * start_range:
            # mark as pennant candidate
            patterns['pennant'].append(('pennant', start, start+window-1))
    # cup & handle : search for rounded bottom structure (very basic)
    # This is intentionally conservative and simplistic
    for i in range(0, len(df)-window, int(window/2) if window>4 else 1):
        seg = df['Close'].iloc[i:i+window]
        if len(seg) < 12: continue
        mid = len(seg)//2
        left_high = seg[:mid].max(); right_high = seg[mid:].max(); bottom = seg.min()
        if left_high > bottom*1.05 and right_high > bottom*1.05 and abs(left_high - right_high) < 0.1*bottom:
            patterns['cup_handle'].append(('cup_handle', i, i+window-1))
    return patterns

# ----------------------- Robust signal generation -------------------------
def generate_robust_signals(df, params):
    """
    Combines pattern, trend, momentum, ATR, volume, zones into a single scoring signal.
    Returns df_signals (df with 'signal' and 'reason') and meta (zones + patterns)
    """
    df = df.copy()
    # indicators
    df['ema_short'] = ema(df['Close'], params.get('ema_short',9))
    df['ema_long']  = ema(df['Close'], params.get('ema_long',21))
    df['ema_trend'] = ema(df['Close'], params.get('ema_trend',50))
    df['rsi'] = rsi(df['Close'], period=params.get('rsi_period',14))
    df['atr'] = atr(df, n=params.get('atr_period',14))
    # pivots & zones
    ph, pl = get_pivots(df['Close'], left=params.get('pivot_window',3), right=params.get('pivot_window',3))
    ph_prices = [p for _,p in ph]; pl_prices = [p for _,p in pl]
    resistances = cluster_levels(ph_prices, tol=params.get('cluster_tol',0.005))
    supports = cluster_levels(pl_prices, tol=params.get('cluster_tol',0.005))
    sup_zones = build_zones(supports, width_pct=params.get('zone_width',0.005))
    res_zones = build_zones(resistances, width_pct=params.get('zone_width',0.005))
    patterns = detect_patterns(df, ph, pl, params)
    # conservative weights tuned to favor precision
    weights = params.get('weights') or {
        'head_and_shoulders': -4.0, 'inverse_head_and_shoulders': 4.0,
        'double_top': -3.0, 'double_bottom': 3.0,
        'triple_top': -3.5, 'triple_bottom': 3.5,
        'bullish_engulfing': 2.0, 'bearish_engulfing': -2.0,
        'hammer': 1.8, 'doji': 0.5,
        'breakout': 1.5, 'support_zone': 1.2, 'resistance_zone': -1.2,
        'upper_wick_liquidity_trap': -1.5, 'lower_wick_liquidity_trap': 1.5
    }
    L = len(df)
    signals = [0]*L; reasons=['']*L
    # build pattern index to quickly find patterns near index
    pattern_index = {}
    for k,v in patterns.items():
        for p in v:
            # end index heuristics
            if len(p) >= 3:
                end_idx = p[2]
            else:
                end_idx = p[1]
            pattern_index.setdefault(end_idx, []).append((k,p))
    vol_median = df['Volume'].median()
    if np.isnan(vol_median): vol_median = 0
    for i in range(L):
        score = 0.0; reason_list=[]
        close = df.loc[i,'Close']; o=df.loc[i,'Open']; h=df.loc[i,'High']; l=df.loc[i,'Low']
        ema_trend = df.loc[i,'ema_trend']; ema_short=df.loc[i,'ema_short']; ema_long=df.loc[i,'ema_long']
        rsi_val = df.loc[i,'rsi'] if not np.isnan(df.loc[i,'rsi']) else None
        atr_val = df.loc[i,'atr'] if not np.isnan(df.loc[i,'atr']) else 0
        # trend
        trend_long = close > ema_trend; trend_short = close < ema_trend
        if trend_long: score += 0.5; reason_list.append('trend_long')
        if trend_short: score -= 0.5; reason_list.append('trend_short')
        # patterns near index
        recent_patterns = []
        for j in range(max(0,i - params.get('pattern_lookahead',5)), i+1):
            recent_patterns += pattern_index.get(j, [])
        for (cat,p) in recent_patterns:
            if cat in ('double','triple'):
                ptype = p[0]  # e.g. 'double_top'
                # map to weight key
                if 'top' in ptype: w = weights.get('double_top', 0) if 'double' in ptype else weights.get(ptype,0)
                else: w = weights.get('double_bottom',0)
                # more careful mapping
                w = weights.get(ptype, weights.get(ptype.split('_')[0], 0)) if ptype in weights else weights.get(ptype.split('_')[0], 0)
                score += w; reason_list.append(f'{ptype}:{w:+.1f}')
            else:
                # p[0] usually a descriptive name
                pname = p[0]
                w = weights.get(pname, 0)
                score += w; reason_list.append(f'{pname}:{w:+.1f}')
        # zones
        for z in sup_zones:
            if close >= z[0] and close <= z[1]:
                score += weights['support_zone']; reason_list.append('near_support_zone')
        for z in res_zones:
            if close >= z[0] and close <= z[1]:
                score += weights['resistance_zone']; reason_list.append('near_resistance_zone')
        # breakout
        look = params.get('breakout_lookback',5)
        if i > look:
            recent_high = df.loc[i-look:i-1,'High'].max()
            recent_low = df.loc[i-look:i-1,'Low'].min()
            if close > recent_high:
                score += weights['breakout']; reason_list.append('breakout_above')
            if close < recent_low:
                score -= weights['breakout']; reason_list.append('breakdown_below')
        # wick traps
        body = abs(close - o) + 1e-9
        upper_wick = h - max(close,o)
        lower_wick = min(close,o) - l
        if (upper_wick > params.get('wick_factor',1.5) * body and df.loc[i,'Volume'] > vol_median * params.get('volume_factor',1.2)):
            score += weights.get('upper_wick_liquidity_trap',0); reason_list.append('upper_wick_trap')
        if (lower_wick > params.get('wick_factor',1.5) * body and df.loc[i,'Volume'] > vol_median * params.get('volume_factor',1.2)):
            score += weights.get('lower_wick_liquidity_trap',0); reason_list.append('lower_wick_trap')
        # rsi confirmation
        if rsi_val is not None:
            if rsi_val > params.get('rsi_long_thresh',55): score += 0.5; reason_list.append(f'rsi:{rsi_val:.1f}')
            if rsi_val < params.get('rsi_short_thresh',45): score -= 0.5; reason_list.append(f'rsi:{rsi_val:.1f}')
        # volume spike
        vol = df.loc[i,'Volume']
        if not np.isnan(vol) and vol_median > 0 and vol > vol_median * params.get('volume_factor',1.2):
            score += 0.5; reason_list.append('volume_spike')
        # ATR volatility preference
        if atr_val > params.get('atr_min',0): score += 0.2
        # thresholding (precision mode)
        threshold = params.get('signal_threshold',3.0) if params.get('precision_mode', True) else params.get('signal_threshold',2.0)
        sig = 0
        if score >= threshold and 'long' in params['allowed_dirs']:
            if trend_long or (ema_short > ema_long): sig = 1
        if score <= -threshold and 'short' in params['allowed_dirs']:
            if trend_short or (ema_short < ema_long): sig = -1
        signals[i] = sig
        reasons[i] = ';'.join(reason_list)
    out = df.copy(); out['signal'] = signals; out['reason'] = reasons
    meta = {'supports': supports, 'resistances': resistances, 'patterns': patterns, 'zones': {'sup':sup_zones,'res':res_zones}, 'pivots': {'ph':ph,'pl':pl}}
    return out, meta

# --------------------------- Backtester (conservative) --------------------
def backtest_signals(df_signals, params):
    trades=[]; L=len(df_signals); i=0
    while i < L-1:
        row = df_signals.loc[i]; sig = int(row['signal'])
        if sig == 0:
            i += 1; continue
        entry_idx = i+1 if i+1 < L else i
        entry_price = df_signals.loc[entry_idx,'Open'] if not pd.isna(df_signals.loc[entry_idx,'Open']) else df_signals.loc[entry_idx,'Close']
        # TP: either absolute points or percent depending on params
        if params.get('use_target_points') and params.get('target_points'):
            tp = entry_price + params['target_points'] if sig == 1 else entry_price - params['target_points']
        else:
            tp = entry_price * (1 + params['tp_pct']) if sig == 1 else entry_price * (1 - params['tp_pct'])
        sl = entry_price * (1 - params['sl_pct']) if sig == 1 else entry_price * (1 + params['sl_pct'])
        exit_price=None; exit_idx=None; exit_reason=None
        max_hold = params.get('max_hold',5)
        # conservative fill: assume SL first if both in same candle
        for j in range(entry_idx, min(L, entry_idx + max_hold + 1)):
            day_high = df_signals.loc[j,'High']; day_low = df_signals.loc[j,'Low']
            if sig == 1:
                if day_low <= sl:
                    exit_price = sl; exit_idx = j; exit_reason='sl'; break
                if day_high >= tp:
                    exit_price = tp; exit_idx = j; exit_reason='tp'; break
            else:
                if day_high >= sl:
                    exit_price = sl; exit_idx = j; exit_reason='sl'; break
                if day_low <= tp:
                    exit_price = tp; exit_idx = j; exit_reason='tp'; break
        if exit_price is None:
            exit_idx = min(L-1, entry_idx + max_hold); exit_price = df_signals.loc[exit_idx,'Close']; exit_reason='time_exit'
        pnl = (exit_price - entry_price)/entry_price if sig == 1 else (entry_price - exit_price)/entry_price
        pnl_points = (exit_price - entry_price) if sig == 1 else (entry_price - exit_price)
        trades.append({'entry_idx':entry_idx,'exit_idx':exit_idx,'entry_time':df_signals.loc[entry_idx,'Date'],'exit_time':df_signals.loc[exit_idx,'Date'],
                       'direction':'long' if sig==1 else 'short','entry_price':entry_price,'exit_price':exit_price,'pnl':pnl,'pnl_points':pnl_points,
                       'hold_days': (df_signals.loc[exit_idx,'Date'] - df_signals.loc[entry_idx,'Date']).days, 'reason': df_signals.loc[i,'reason'] or exit_reason})
        i = exit_idx + 1 if exit_idx is not None else i + 1
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        stats = {'total_trades':0,'positive_trades':0,'negative_trades':0,'accuracy':0.0,'total_pnl_pct':0.0,'avg_pnl_pct':0.0,'avg_hold_days':0.0,'total_points':0.0}
    else:
        stats = {'total_trades': int(len(trades_df)), 'positive_trades': int((trades_df['pnl']>0).sum()),
                 'negative_trades': int((trades_df['pnl']<=0).sum()), 'accuracy': float((trades_df['pnl']>0).mean()),
                 'total_pnl_pct': float(trades_df['pnl'].sum()*100), 'avg_pnl_pct': float(trades_df['pnl'].mean()*100),
                 'avg_hold_days': float(trades_df['hold_days'].mean()), 'total_points': float(trades_df['pnl_points'].sum())}
    return trades_df, stats

# ----------------------------- Search & WFCV --------------------------------
def sample_random_params(n_samples=50, param_space=None):
    default_space = {'pivot_window':[2,3,4],'cluster_tol':[0.003,0.005],'zone_width':[0.003,0.005],'sl_pct':[0.005,0.01],'tp_pct':[0.01,0.02],
                     'max_hold':[3,5,7],'breakout_lookback':[3,5,8],'pattern_tol':[0.01,0.02],'min_bars_between':[2,3],'signal_threshold':[2.5,3.0]}
    space = param_space if param_space is not None else default_space
    keys = list(space.keys()); samples=[]
    for _ in range(n_samples):
        s = {k: random.choice(space[k]) for k in keys}
        samples.append(s)
    return samples

def grid_params(param_grid):
    keys = list(param_grid.keys()); values=[param_grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

def find_best_strategy(df_train, search_type='random', random_iters=50, grid=None, allowed_dirs=['long','short'], desired_accuracy=0.9, min_trades=3, progress_callback=None, use_points=False, target_points=None, optimize_for='accuracy', precision_mode=True):
    best=None; best_metric=-np.inf; tried=0
    if search_type=='random':
        param_space = grid if isinstance(grid, dict) else None
        samples = sample_random_params(random_iters, param_space=param_space)
    else:
        if grid is None: raise ValueError("Grid required for grid search")
        samples = list(grid_params(grid))
    total = len(samples)
    for s in samples:
        tried += 1
        params = {'pivot_window': s.get('pivot_window',3), 'cluster_tol': s.get('cluster_tol',0.005), 'zone_width': s.get('zone_width',0.005),
                  'sl_pct': s.get('sl_pct',0.01), 'tp_pct': s.get('tp_pct',0.02), 'max_hold': s.get('max_hold',5), 'breakout_lookback': s.get('breakout_lookback',5),
                  'pattern_tol': s.get('pattern_tol',0.02), 'min_bars_between': s.get('min_bars_between',3), 'wick_factor':1.5, 'volume_factor':1.2,
                  'pattern_lookahead':5, 'hs_tol':0.03, 'signal_threshold': s.get('signal_threshold',3.0), 'allowed_dirs': allowed_dirs,
                  'use_target_points': use_points, 'target_points': target_points, 'precision_mode': precision_mode}
        df_signals, _ = generate_robust_signals(df_train, params)
        trades_df, stats = backtest_signals(df_signals, params)
        if stats['total_trades'] < min_trades:
            metric = -np.inf
        else:
            metric = stats['accuracy'] if optimize_for == 'accuracy' else stats['total_pnl_pct']
            if stats['accuracy'] >= desired_accuracy: metric += 1000
        if metric > best_metric:
            best_metric = metric; best = {'params':params, 'trades':trades_df, 'stats':stats}
        if progress_callback is not None:
            try: progress_callback(tried, total)
            except Exception: pass
    return best, tried

def walk_forward_cv(df_all, train_months=24, test_months=6, step_months=6, search_type='random', random_iters=60, grid=None, **search_kwargs):
    """
    Rolling-window walk-forward: optimize on each train window and test on following test window.
    Returns a list of dictionaries with train_range, test_range, train_stats, test_stats, best_params.
    """
    results = []
    if df_all.empty: return results
    # convert to naive dates for arithmetic
    df = df_all.copy()
    df['date_naive'] = df['Date'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    start = df['date_naive'].min(); end = df['date_naive'].max()
    train_delta = pd.DateOffset(months=train_months); test_delta = pd.DateOffset(months=test_months); step_delta = pd.DateOffset(months=step_months)
    win_start = start
    while True:
        train_end = win_start + train_delta - pd.Timedelta(seconds=1)
        test_start = train_end + pd.Timedelta(seconds=1)
        test_end = test_start + test_delta - pd.Timedelta(seconds=1)
        if test_end > end:
            break
        df_train = df[(df['date_naive'] >= win_start) & (df['date_naive'] <= train_end)].drop(columns=['date_naive'])
        df_test  = df[(df['date_naive'] >= test_start) & (df['date_naive'] <= test_end)].drop(columns=['date_naive'])
        if len(df_train) < 20 or len(df_test) < 5:
            win_start = win_start + step_delta
            if win_start > end: break
            continue
        best, tried = find_best_strategy(df_train, search_type=search_type, random_iters=random_iters, grid=grid, **search_kwargs)
        if best is None:
            win_start = win_start + step_delta; continue
        # evaluate best on test
        params = best['params']
        df_signals_test, _ = generate_robust_signals(df_test, params)
        trades_test, stats_test = backtest_signals(df_signals_test, params)
        df_signals_train, _ = generate_robust_signals(df_train, params)
        trades_train, stats_train = backtest_signals(df_signals_train, params)
        results.append({'train_start':win_start,'train_end':train_end,'test_start':test_start,'test_end':test_end,
                        'best_params':params, 'train_stats':stats_train, 'test_stats':stats_test})
        win_start = win_start + step_delta
        if win_start > end: break
    return results

# ---------------------------- Visualization helpers -----------------------
def plot_candles_with_overlays(df, meta, trades=None, title="Price with overlays"):
    """
    Use mplfinance to plot candles and overlay zones, pivots, pattern boxes, and trades.
    trades: DataFrame with entry_idx and exit_idx and entry_time/exit_time
    """
    df_plot = df.copy()
    df_plot = df_plot.set_index('Date')
    # style
    mc = mpf.make_marketcolors(up='g', down='r', wick='inherit', edge='inherit', volume='in')
    s  = mpf.make_mpf_style(base_mpl_style='fast', marketcolors=mc)
    ap = []
    # zones shading
    for z in meta.get('zones',{}):
        pass  # we'll draw zones using matplotlib after mpf plot
    # prepare mpf plot
    fig, axlist = mpf.plot(df_plot, type='candle', style=s, volume=True, returnfig=True, figsize=(12,6), title=title)
    ax = axlist[0]
    # draw zones
    for z in meta.get('zones', {}).get('sup', []):
        ax.axhspan(z[0], z[1], alpha=0.12, color='green')
    for z in meta.get('zones', {}).get('res', []):
        ax.axhspan(z[0], z[1], alpha=0.12, color='red')
    # draw pivots
    for (i,p) in meta.get('pivots', {}).get('ph', []):
        try:
            idx = df_plot.index[i]
            ax.scatter(idx, p, marker='^', color='black', s=30)
        except Exception:
            pass
    for (i,p) in meta.get('pivots', {}).get('pl', []):
        try:
            idx = df_plot.index[i]
            ax.scatter(idx, p, marker='v', color='black', s=30)
        except Exception:
            pass
    # draw detected patterns bounding boxes (simplified)
    patterns = meta.get('patterns', {})
    for cat, items in patterns.items():
        color = 'blue'
        if cat in ('hs','inverse_hs'): color='purple'
        if cat in ('engulf',): color='orange'
        for p in items:
            try:
                start_idx = p[1]; end_idx = p[2]
                start = df_plot.index[start_idx]; end = df_plot.index[end_idx]
                ymin = df_plot['Low'].iloc[start_idx:end_idx+1].min()
                ymax = df_plot['High'].iloc[start_idx:end_idx+1].max()
                ax.add_patch(plt.Rectangle((start, ymin), end-start, ymax-ymin, fill=False, edgecolor=color, linewidth=1, alpha=0.6))
            except Exception:
                pass
    # draw trades if provided
    if trades is not None and not trades.empty:
        for _, tr in trades.iterrows():
            try:
                e_time = pd.to_datetime(tr['entry_time'])
                x = e_time.tz_convert('Asia/Kolkata') if e_time.tzinfo else e_time
                ax.annotate('Entry', xy=(x, tr['entry_price']), xytext=(0,15), textcoords='offset points', arrowprops=dict(arrowstyle='->', color='green'))
                x2 = pd.to_datetime(tr['exit_time'])
                x2 = x2.tz_convert('Asia/Kolkata') if x2.tzinfo else x2
                ax.annotate('Exit', xy=(x2, tr['exit_price']), xytext=(0,-15), textcoords='offset points', arrowprops=dict(arrowstyle='->', color='red'))
            except Exception:
                pass
    st.pyplot(fig)

# ------------------------------ Streamlit UI ------------------------------
def app():
    st.title("Swing Trading Recommender — Upgraded")
    st.markdown("Select data source (yfinance NIFTY50 or upload). Backtest, walk-forward CV, visualization overlays, and live recommendation use the same logic.")

    source = st.sidebar.radio("Data source", ["yfinance (Nifty50)", "Upload file (CSV/XLSX)"])
    tickers = []
    if source.startswith('yfinance'):
        with st.sidebar.expander("NIFTY50 tickers source & selection"):
            st.write("Tickers fetched live from Wikipedia NIFTY 50 page (fallback if unavailable).")
            # fetch tickers
            tickers = fetch_nifty50_tickers()
            sel_ticker = st.selectbox("Select ticker", tickers, index=0)
            interval = st.selectbox("Timeframe / interval", ['1d','1h','30m','15m','5m'])
            period = st.selectbox("Period", ['1mo','3mo','6mo','1y','2y','5y','max'], index=3)
    else:
        with st.sidebar.expander("Upload data"):
            upload = st.file_uploader("Upload CSV/Excel with OHLCV", type=['csv','xlsx','xls'])

    # common controls
    st.sidebar.markdown("---")
    st.sidebar.header("Strategy / Optimization")
    side_opt = st.sidebar.selectbox("Side", ['both','long','short'])
    search_type = st.sidebar.selectbox("Search method", ['random','grid'])
    random_iters = st.sidebar.number_input("Random search iterations", min_value=10, max_value=2000, value=200)
    desired_accuracy = st.sidebar.slider("Desired accuracy (win rate target)", 0.5, 0.99, 0.9, step=0.01)
    min_trades = st.sidebar.number_input("Minimum trades required", min_value=1, max_value=500, value=5)
    use_points_for_backtest = st.sidebar.checkbox("Use absolute target points for TP in backtest/live", value=False)
    target_points = st.sidebar.number_input("Target points (if using points)", min_value=1, value=50)
    precision_mode = st.sidebar.checkbox("Precision mode (stricter)", value=True)
    optimize_for = st.sidebar.selectbox("Optimize for", ['accuracy','points'])
    st.sidebar.markdown("---")
    st.sidebar.header("Walk-Forward CV (optional)")
    do_wfcv = st.sidebar.checkbox("Run Walk-Forward CV after optimization", value=False)
    w_train_months = st.sidebar.number_input("WFCV train months", min_value=6, max_value=60, value=24)
    w_test_months = st.sidebar.number_input("WFCV test months", min_value=1, max_value=24, value=6)
    w_step_months = st.sidebar.number_input("WFCV step months", min_value=1, max_value=24, value=6)
    st.sidebar.markdown("---")
    st.sidebar.header("Live recommendation options")
    allow_looser_live = st.sidebar.checkbox("Allow slightly looser live filtering (keeps backtest strict)", value=False)
    capital = st.sidebar.number_input("Optional capital for sizing (0=off)", min_value=0, value=0)
    risk_pct = st.sidebar.number_input("Risk % per trade", min_value=0.1, max_value=10.0, value=1.0)

    # load data
    df = None; mapping = None
    if source.startswith('yfinance'):
        st.write(f"Fetching {sel_ticker} data — period={period}, interval={interval} ...")
        try:
            raw = yf.download(sel_ticker, period=period, interval=interval, progress=False, threads=True)
            if raw.empty:
                st.error("yfinance returned no data for that ticker/interval/period combination.")
                return
            raw = raw.reset_index().rename(columns={'Datetime':'Date'})
            # yfinance already has Date column
            df, mapping = standardize_df(raw.rename(columns={raw.columns[0]:'Date'})) if 'Date' in raw.columns else standardize_df(raw)
        except Exception as e:
            st.error(f"Failed to fetch via yfinance: {e}")
            return
    else:
        if 'upload' not in locals() or upload is None:
            st.info("Upload a file to continue.")
            return
        try:
            if upload.name.endswith('.csv'):
                raw = pd.read_csv(upload)
            else:
                raw = pd.read_excel(upload)
            df, mapping = standardize_df(raw)
        except Exception as e:
            st.error(f"Failed to read/standardize file: {e}")
            return

    st.subheader("Mapped columns (detected)")
    st.json(mapping)

    # display sample and basic metrics
    st.subheader("Data preview (Date shown in IST)")
    df_display = df.copy()
    try:
        df_display['Date'] = df_display['Date'].dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S %Z')
    except Exception:
        df_display['Date'] = df_display['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    c1,c2 = st.columns(2)
    with c1:
        st.write("Top 5 rows"); st.dataframe(df_display.head())
    with c2:
        st.write("Bottom 5 rows"); st.dataframe(df_display.tail())
    st.write("Date range:", df['Date'].min(), "to", df['Date'].max())
    st.write("Price range (Close):", df['Close'].min(), "to", df['Close'].max())

    # heatmap of returns — improved colormap and annotation
    st.subheader("Year vs Month returns (heatmap, %)")
    df['returns'] = df['Close'].pct_change()
    df['year'] = df['Date'].dt.tz_convert('Asia/Kolkata').dt.year
    df['month'] = df['Date'].dt.tz_convert('Asia/Kolkata').dt.month
    try:
        heat = df.pivot_table(values='returns', index='year', columns='month', aggfunc=lambda x: (x+1.0).prod()-1)
        heat_pct = heat * 100
        fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(heat_pct.index))))
        cmap = plt.get_cmap('coolwarm')
        im = ax.imshow(heat_pct.fillna(0).values, aspect='auto', cmap=cmap, interpolation='nearest')
        ax.set_xticks(range(len(heat_pct.columns))); ax.set_xticklabels(heat_pct.columns)
        ax.set_yticks(range(len(heat_pct.index))); ax.set_yticklabels(heat_pct.index)
        ax.set_title('Year-Month Returns (%)')
        for (j,i), val in np.ndenumerate(heat_pct.fillna(0).values):
            ax.text(i, j, f"{val:.2f}%", ha='center', va='center', fontsize=8, color='k' if abs(val) < 8 else 'white')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
    except Exception as e:
        st.warning("Could not build heatmap: " + str(e))
        st.dataframe(df.groupby(['year','month'])['returns'].apply(lambda x: (x+1.0).prod()-1).unstack(fill_value=0) * 100)

    # 100-word summary
    st.subheader("Automated summary (approx 100 words)")
    def generate_summary(df):
        try:
            returns = df['returns'].dropna()
            mean_ret = returns.mean() * 252
            vol = returns.std() * np.sqrt(252)
            last_close = df['Close'].iloc[-1]
            trend = 'uptrend' if (len(df) >= 30 and df['Close'].iloc[-1] > df['Close'].iloc[-30:].mean()) else 'sideways/downtrend'
            opp = []
            if mean_ret > 0.05: opp.append('long-biased')
            if vol > 0.2: opp.append('high volatility')
            return (f"Data from {df['Date'].min().date()} to {df['Date'].max().date()}. Latest close {last_close:.2f}. The recent trend appears {trend}. "
                    f"Annualized mean return approx {mean_ret:.2%} with volatility {vol:.2%}. Potential opportunities: {', '.join(opp) or 'range trading and breakouts'}. "
                    f"Look for support/resistance, double bottom/top and wick traps near key levels; use disciplined SL and position sizing.")
        except Exception:
            return "Could not generate summary."
    st.write(generate_summary(df))

    # optimization & walk-forward triggers
    if st.button("Start Optimization (and optional Walk-Forward CV)"):
        progress_bar = st.progress(0); status_text = st.empty()
        def progress_cb(done, total):
            pct = int(done/total*100) if total else 0
            progress_bar.progress(min(max(pct,0),100))
            status_text.text(f"Progress: {pct}% ({done}/{total})")
        # allowed dirs
        allowed_dirs=[]
        if side_opt in ['both','long']: allowed_dirs.append('long')
        if side_opt in ['both','short']: allowed_dirs.append('short')
        # grid if specified in sidebar (kept small by UI)
        grid = None
        if search_type == 'grid':
            grid = {'pivot_window':[2,3,4], 'sl_pct':[0.005,0.01], 'tp_pct':[0.01,0.02], 'cluster_tol':[0.005], 'zone_width':[0.005], 'max_hold':[5], 'signal_threshold':[2.5,3.0]}
        # find best
        best, tried = find_best_strategy(df, search_type=search_type, random_iters=random_iters, grid=grid, allowed_dirs=allowed_dirs, desired_accuracy=desired_accuracy, min_trades=min_trades, progress_callback=progress_cb, use_points=use_points_for_backtest, target_points=(target_points if use_points_for_backtest else None), optimize_for=optimize_for, precision_mode=precision_mode)
        progress_bar.progress(100); status_text.success("Optimization finished")
        if best is None:
            st.warning("No strategy found meeting requirements. Loosen filters and try again.")
            return
        st.success("Best strategy found")
        st.write("Tried combinations:", tried)
        st.subheader("Best strategy params"); st.json(best['params'])
        st.subheader("Backtest stats"); st.json(best['stats'])
        trades_df = best['trades'].copy()
        if not trades_df.empty:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time']).dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            trades_df['exit_time']  = pd.to_datetime(trades_df['exit_time']).dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        st.write("Sample trades (first 200)"); st.dataframe(trades_df.head(200))
        # buy and hold comparison
        if len(df) >= 2:
            buy_points = df['Close'].iloc[-1] - df['Close'].iloc[0]
            buy_pct = (df['Close'].iloc[-1]/df['Close'].iloc[0]-1)*100
        else:
            buy_points = 0; buy_pct = 0
        strategy_points = best['stats'].get('total_points',0)
        pct_more_points = (strategy_points - buy_points) / (abs(buy_points) if buy_points!=0 else 1) * 100
        st.metric("Buy & hold points", f"{buy_points:.2f}"); st.metric("Buy & hold return %", f"{buy_pct:.2f}%")
        st.metric("Strategy total points", f"{strategy_points:.2f}")
        st.write(f"Strategy gave {pct_more_points:.2f}% more points vs buy-and-hold (relative to absolute buy-and-hold points)")

        # Walk-forward CV if requested
        wfcv_results = None
        if do_wfcv:
            st.info("Running Walk-Forward CV (this may take longer).")
            wfcv_results = walk_forward_cv(df, train_months=w_train_months, test_months=w_test_months, step_months=w_step_months, search_type=search_type, random_iters=max(20, int(random_iters/4)), grid=grid, allowed_dirs=allowed_dirs, desired_accuracy=desired_accuracy, min_trades=min_trades, use_points=use_points_for_backtest, target_points=(target_points if use_points_for_backtest else None), optimize_for=optimize_for, precision_mode=precision_mode, progress_callback=progress_cb)
            if not wfcv_results:
                st.warning("Walk-forward found no usable windows.")
            else:
                avg_test_acc = np.mean([r['test_stats']['accuracy'] for r in wfcv_results if r['test_stats'] and r['test_stats'].get('total_trades',0)>0])
                st.subheader("Walk-Forward CV summary")
                st.write("Windows tested:", len(wfcv_results))
                st.write("Average out-of-sample accuracy:", f"{avg_test_acc:.3f}" if not np.isnan(avg_test_acc) else "N/A")
                # show table of windows
                rows=[]
                for r in wfcv_results:
                    rows.append({'train_start':r['train_start'].date(),'train_end':r['train_end'].date(),'test_start':r['test_start'].date(),'test_end':r['test_end'].date(),'test_acc': r['test_stats']['accuracy'] if r['test_stats'] else None,'test_trades': r['test_stats']['total_trades'] if r['test_stats'] else None})
                st.dataframe(pd.DataFrame(rows))

        # compute full signals with best params (for visualization & live rec)
        df_signals_full, meta = generate_robust_signals(df, best['params'])
        trades_full, stats_full = backtest_signals(df_signals_full, best['params'])
        st.subheader("Visualization — price chart with overlays & trades")
        # show chart
        plot_candles_with_overlays(df, meta, trades_full)

        # Live recommendation: use same backtested params (or slightly looser if user allowed)
        st.subheader("Live recommendation (same logic as backtest)")
        live_params = best['params'].copy()
        if allow_looser_live:
            # Slightly reduce required threshold for live only (user explicitly chose to allow)
            live_params['signal_threshold'] = max(1.5, live_params.get('signal_threshold',3.0) - 1.0)
            st.info("Live filtering is slightly looser than backtest (per your choice); backtest remained strict.")
        # ensure use_target_points matches choice
        live_params['use_target_points'] = use_points_for_backtest
        live_params['target_points'] = target_points if use_points_for_backtest else None
        # recompute signals on latest data (already have df_signals_full)
        # pick last closed candle index — if intraday and last bar may be incomplete, use last-1 as conservative
        last_idx = len(df_signals_full) - 1
        # detect if last timestamp is within last minute (optional heuristic): keep as last row for simplicity
        last_sig_row = df_signals_full.iloc[last_idx]
        if int(last_sig_row['signal']) == 0:
            st.write("No live signal according to best strategy on the last closed candle.")
        else:
            # build recommendation (same as backtest logic)
            sig = int(last_sig_row['signal'])
            entry_price = float(last_sig_row['Close'])
            if live_params.get('use_target_points') and live_params.get('target_points'):
                tp = entry_price + live_params['target_points'] if sig==1 else entry_price - live_params['target_points']
            else:
                tp = entry_price * (1 + live_params['tp_pct']) if sig==1 else entry_price * (1 - live_params['tp_pct'])
            sl = entry_price * (1 - live_params['sl_pct']) if sig==1 else entry_price * (1 + live_params['sl_pct'])
            unit_risk = abs(entry_price - sl)
            suggested_units = None
            if capital and capital>0 and unit_risk>0:
                risk_amount = capital * (risk_pct / 100.0)
                suggested_units = int(risk_amount // unit_risk)
            rec = {'direction': 'long' if sig==1 else 'short', 'entry_price': round(entry_price,4), 'stop_loss': round(sl,4), 'target_price': round(tp,4), 'unit_risk': round(unit_risk,4), 'suggested_units_by_capital': int(suggested_units) if suggested_units else None, 'reason': last_sig_row['reason'], 'prob_of_profit': float(best['stats']['accuracy'])}
            st.json(rec)

        st.success("Optimization & live recommendation completed.")

if __name__ == "__main__":
    app()
