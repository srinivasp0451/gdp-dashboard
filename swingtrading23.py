import streamlit as st
import pandas as pd
import numpy as np
import re
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime
import random
import itertools

st.set_page_config(page_title="Swing Trading Recommender — Live + WFCV + Visuals (Fixed)", layout="wide")

# ---------------------------- Session defaults ---------------------------
if 'fetched_df' not in st.session_state:
    st.session_state['fetched_df'] = None
if 'raw_source' not in st.session_state:
    st.session_state['raw_source'] = None
if 'best_strategy' not in st.session_state:
    st.session_state['best_strategy'] = None

# -------------------- Helper: column mapping & standardize -----------------

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
        if mapping['close'] is None and ("close" in token_set or ("adj" in token_set and "close" in token_set) or low.strip() in ("c","cl","last","price","closeprice")):
            mapping['close'] = col
            continue
        if mapping['volume'] is None and any(t in token_set for t in ("volume","vol","qty","quantity","tradevolume")):
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


def standardize_df_from_file(raw):
    df = raw.copy()
    mapping = infer_columns(df)
    if mapping['date'] is None:
        raise ValueError('No date column found')
    df['Date'] = pd.to_datetime(df[mapping['date']], errors='coerce')
    if df['Date'].isna().all():
        df['Date'] = pd.to_datetime(df[mapping['date']], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['Date']).reset_index(drop=True)
    # localize/convert to IST
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
        possibles = [c for c in df.columns if re.search(r"price", str(c).lower())]
        if possibles:
            df = df.rename(columns={possibles[0]:'Close'})
    if 'Close' not in df.columns:
        candidates = [c for c in df.columns if c!='Date']
        if candidates:
            df['Close'] = pd.to_numeric(df[candidates[0]], errors='coerce')
        else:
            raise ValueError('No price column')
    for col in ['Open','High','Low','Close']:
        if col not in df.columns:
            df[col] = df['Close']
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'Volume' not in df.columns:
        df['Volume'] = np.nan
    else:
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df = df[['Date','Open','High','Low','Close','Volume']]
    df = df.sort_values('Date').reset_index(drop=True)
    return df, mapping


def standardize_df_from_yf(df):
    d = df.copy()
    if isinstance(d.index, pd.DatetimeIndex):
        d = d.reset_index().rename(columns={'index':'Date'})
    for col in ['Open','High','Low','Close','Volume']:
        if col not in d.columns:
            d[col] = np.nan
    d['Date'] = pd.to_datetime(d['Date'], errors='coerce')
    try:
        if d['Date'].dt.tz is None:
            d['Date'] = d['Date'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        else:
            d['Date'] = d['Date'].dt.tz_convert('Asia/Kolkata')
    except Exception:
        d['Date'] = pd.to_datetime(d['Date']).dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    for col in ['Open','High','Low','Close','Volume']:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    d = d[['Date','Open','High','Low','Close','Volume']]
    d = d.dropna(subset=['Date']).reset_index(drop=True)
    d = d.sort_values('Date').reset_index(drop=True)
    return d


# ---------------------- Technical indicators ------------------------------

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
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev = close.shift(1)
    tr = pd.concat([high-low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ---------------------- price-action & patterns ---------------------------

def get_pivots(price_series, left=3, right=3):
    ph = []
    pl = []
    s = price_series.values
    L = len(s)
    for i in range(left, L-right):
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
    cur = [levels[0]]
    for p in levels[1:]:
        if abs(p - np.mean(cur)) <= tol * np.mean(cur):
            cur.append(p)
        else:
            clusters.append(np.mean(cur))
            cur = [p]
    clusters.append(np.mean(cur))
    return clusters


def detect_basic_patterns(df, ph, pl, params):
    patterns = []
    price = df['Close'].values
    for i in range(len(ph)-1):
        idx1, p1 = ph[i]
        idx2, p2 = ph[i+1]
        if idx2 - idx1 >= params['min_bars_between'] and abs(p1 - p2) <= params['pattern_tol'] * ((p1 + p2)/2):
            patterns.append(('double_top', idx1, idx2, p1))
    for i in range(len(pl)-1):
        idx1, p1 = pl[i]
        idx2, p2 = pl[i+1]
        if idx2 - idx1 >= params['min_bars_between'] and abs(p1 - p2) <= params['pattern_tol'] * ((p1 + p2)/2):
            patterns.append(('double_bottom', idx1, idx2, p1))
    for i in range(len(ph)-2):
        l_idx, l_p = ph[i]
        m_idx, m_p = ph[i+1]
        r_idx, r_p = ph[i+2]
        if (m_idx - l_idx >= params['min_bars_between'] and r_idx - m_idx >= params['min_bars_between']):
            shoulders = (l_p + r_p) / 2
            if m_p > shoulders and abs(l_p - r_p) <= params.get('hs_tol',0.03) * shoulders:
                patterns.append(('head_and_shoulders', l_idx, m_idx, r_idx))
    for i in range(1, len(df)):
        po = df.loc[i-1,'Open']; pc = df.loc[i-1,'Close']; o = df.loc[i,'Open']; c = df.loc[i,'Close']
        if pc < po and c > o and (c - o) > (po - pc):
            patterns.append(('bullish_engulf', i-1, i))
        if pc > po and c < o and (o - c) > (pc - po):
            patterns.append(('bearish_engulf', i-1, i))
    return patterns


# ---------------------- robust signal generation -------------------------

def generate_signals_and_meta(df, params):
    df = df.copy()
    df['ema9'] = ema(df['Close'], 9)
    df['ema21'] = ema(df['Close'], 21)
    df['ema50'] = ema(df['Close'], 50)
    df['rsi'] = rsi(df['Close'], 14)
    df['atr'] = atr(df, 14)

    ph, pl = get_pivots(df['Close'], left=params['pivot_window'], right=params['pivot_window'])
    ph_prices = [p for _,p in ph]
    pl_prices = [p for _,p in pl]
    supports = cluster_levels(pl_prices, tol=params['cluster_tol'])
    resistances = cluster_levels(ph_prices, tol=params['cluster_tol'])
    sup_zones = [(lv - lv*params['zone_width'], lv + lv*params['zone_width']) for lv in supports]
    res_zones = [(lv - lv*params['zone_width'], lv + lv*params['zone_width']) for lv in resistances]

    patterns = detect_basic_patterns(df, ph, pl, params)

    weights = params.get('weights') or {
        'double_top': -3.0, 'double_bottom': 3.0,
        'head_and_shoulders': -4.0, 'bullish_engulf': 2.0, 'bearish_engulf': -2.0
    }

    signals = [0]*len(df)
    reasons = ['']*len(df)
    vol_med = df['Volume'].median() if not df['Volume'].isnull().all() else 0

    pat_map = {}
    for p in patterns:
        end = p[2] if len(p)>=3 else p[1]
        pat_map.setdefault(end, []).append(p[0])

    for i in range(len(df)):
        score = 0.0
        fired = []
        price = df.loc[i,'Close']
        # pattern score
        for p in pat_map.get(i, []):
            score += weights.get(p, 0)
            fired.append(p)
        # trend
        if price > df.loc[i,'ema50']:
            score += 0.5; fired.append('trend_long')
        else:
            score -= 0.5; fired.append('trend_short')
        # rsi
        if df.loc[i,'rsi'] > 55:
            score += 0.5; fired.append('rsi_long')
        if df.loc[i,'rsi'] < 45:
            score -= 0.5; fired.append('rsi_short')
        # volume
        if vol_med>0 and df.loc[i,'Volume'] > vol_med*params['volume_factor']:
            score += 0.5; fired.append('vol_spike')
        # wick traps
        o,h,l,c = df.loc[i,['Open','High','Low','Close']]
        body = abs(c-o)+1e-9
        if h - max(c,o) > params['wick_factor']*body:
            score -= 1.0; fired.append('upper_wick')
        if min(c,o) - l > params['wick_factor']*body:
            score += 1.0; fired.append('lower_wick')

        thr = params['signal_threshold']
        sig = 0
        if score >= thr and 'long' in params['allowed_dirs']:
            sig = 1
        if score <= -thr and 'short' in params['allowed_dirs']:
            sig = -1
        signals[i]=sig
        reasons[i] = ','.join(fired)

    df_out = df.copy()
    df_out['signal'] = signals
    df_out['reason'] = reasons
    meta = {'supports': supports, 'resistances': resistances, 'sup_zones': sup_zones, 'res_zones': res_zones, 'patterns': patterns}
    return df_out, meta


# ------------------------ backtest (same logic used for live) -------------

def backtest(df_signals, params):
    trades = []
    i = 0
    L = len(df_signals)
    while i < L-1:
        sig = int(df_signals.loc[i,'signal'])
        if sig == 0:
            i += 1
            continue
        entry_idx = i+1 if i+1 < L else i
        entry_price = df_signals.loc[entry_idx,'Open'] if not pd.isna(df_signals.loc[entry_idx,'Open']) else df_signals.loc[entry_idx,'Close']
        if params['use_target_points'] and params.get('target_points'):
            tp = entry_price + params['target_points'] if sig==1 else entry_price - params['target_points']
        else:
            tp = entry_price*(1+params['tp_pct']) if sig==1 else entry_price*(1-params['tp_pct'])
        sl = entry_price*(1-params['sl_pct']) if sig==1 else entry_price*(1+params['sl_pct'])
        exit_price=None; exit_idx=None; exit_reason=None
        for j in range(entry_idx, min(L, entry_idx+params['max_hold']+1)):
            day_high = df_signals.loc[j,'High']; day_low = df_signals.loc[j,'Low']
            if sig==1:
                if day_low <= sl:
                    exit_price=sl; exit_idx=j; exit_reason='sl'; break
                if day_high >= tp:
                    exit_price=tp; exit_idx=j; exit_reason='tp'; break
            else:
                if day_high >= sl:
                    exit_price=sl; exit_idx=j; exit_reason='sl'; break
                if day_low <= tp:
                    exit_price=tp; exit_idx=j; exit_reason='tp'; break
        if exit_price is None:
            exit_idx = min(L-1, entry_idx+params['max_hold'])
            exit_price = df_signals.loc[exit_idx,'Close']
            exit_reason='time_exit'
        pnl = (exit_price-entry_price)/entry_price if sig==1 else (entry_price-exit_price)/entry_price
        pnl_points = (exit_price-entry_price) if sig==1 else (entry_price-exit_price)
        trades.append({'entry_idx':entry_idx,'exit_idx':exit_idx,'entry_time':df_signals.loc[entry_idx,'Date'],'exit_time':df_signals.loc[exit_idx,'Date'],'direction':'long' if sig==1 else 'short','entry_price':entry_price,'exit_price':exit_price,'pnl':pnl,'pnl_points':pnl_points,'reason':df_signals.loc[i,'reason'] or exit_reason,'hold_days':(df_signals.loc[exit_idx,'Date']-df_signals.loc[entry_idx,'Date']).days})
        i = exit_idx+1 if exit_idx is not None else i+1
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        stats = {'total_trades':0,'positive_trades':0,'negative_trades':0,'accuracy':0.0,'total_pnl_pct':0.0,'avg_pnl_pct':0.0,'total_points':0.0}
    else:
        stats = {'total_trades':len(trades_df),'positive_trades':int((trades_df['pnl']>0).sum()),'negative_trades':int((trades_df['pnl']<=0).sum()),'accuracy':float((trades_df['pnl']>0).mean()),'total_pnl_pct':float(trades_df['pnl'].sum()*100),'avg_pnl_pct':float(trades_df['pnl'].mean()*100),'total_points':float(trades_df['pnl_points'].sum())}
    return trades_df, stats


# --------------------- Visualization helpers ----------------------------

def plot_with_overlays(df, meta, trades=None, title='Price with overlays'):
    df_plot = df.copy()
    df_plot = df_plot.set_index('Date')
    # prepare hlines and colors
    hlines = []
    colors = []
    for s in meta.get('supports', []):
        hlines.append(s); colors.append('green')
    for r in meta.get('resistances', []):
        # avoid duplicate
        if r not in meta.get('supports',[]):
            hlines.append(r); colors.append('red')
    hlines_dict = {'hlines': hlines, 'colors': colors, 'linewidths': 1, 'alpha':0.6} if hlines else None

    # create entry/exit markers aligned to df_plot index
    addplots = []
    if trades is not None and not trades.empty:
        entries = pd.Series(np.nan, index=df_plot.index)
        exits = pd.Series(np.nan, index=df_plot.index)
        # map trade times to nearest index in df_plot
        for _,t in trades.iterrows():
            try:
                et = t['entry_time']
                xt = t['exit_time']
                # find nearest index (by date)
                nearest_entry_idx = df_plot.index.get_indexer([et], method='nearest')[0]
                nearest_exit_idx = df_plot.index.get_indexer([xt], method='nearest')[0]
                if nearest_entry_idx >= 0 and nearest_entry_idx < len(df_plot):
                    entries.iloc[nearest_entry_idx] = t['entry_price']
                if nearest_exit_idx >= 0 and nearest_exit_idx < len(df_plot):
                    exits.iloc[nearest_exit_idx] = t['exit_price']
            except Exception:
                continue
        if entries.notna().any():
            addplots.append(mpf.make_addplot(entries, type='scatter', markersize=50, marker='^'))
        if exits.notna().any():
            addplots.append(mpf.make_addplot(exits, type='scatter', markersize=50, marker='v'))

    try:
        fig, axlist = mpf.plot(df_plot, type='candle', style='charles', title=title, returnfig=True, addplot=addplots, hlines=hlines_dict)
        return fig
    except Exception as e:
        # fallback: simple close-line plot
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df_plot.index, df_plot['Close'])
        ax.set_title(title)
        for lvl in hlines:
            ax.axhline(lvl, color='green' if lvl in meta.get('supports',[]) else 'red', alpha=0.4)
        return fig


# ---------------------------- Streamlit app ------------------------------

def app():
    st.title('Swing Trading Recommender — Fixed (Fetch + Optimize)')
    st.markdown('This version fixes the TypeError seen in the heatmap and adds a separate **Fetch data** button (to avoid yfinance rate limits). The live recommendation uses identical logic as the backtester.')

    # Sidebar: data source and controls
    with st.sidebar:
        st.header('Data source')
        source = st.radio('Choose source', ['Nifty50 dropdown (yfinance)','Upload file'])
        nifty50 = ['RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','HDFC.NS','ICICIBANK.NS','KOTAKBANK.NS','SBIN.NS','LT.NS','AXISBANK.NS']
        if source.startswith('Nifty'):
            ticker_choice = st.selectbox('Select ticker', nifty50 + ['Other'])
            if ticker_choice == 'Other':
                ticker = st.text_input('Enter ticker (yfinance format, e.g. RELIANCE.NS or AAPL)', value='RELIANCE.NS')
            else:
                ticker = ticker_choice
            timeframe = st.selectbox('Interval', ['1d','1wk','1mo','60m','30m'])
            period = st.selectbox('Period', ['6mo','1y','2y','5y','max'])
            st.markdown('Click **Fetch data** to download from yfinance (avoids automatic repeated fetches and rate limits).')
        else:
            upload = st.file_uploader('Upload CSV/Excel', type=['csv','xlsx','xls'])
            ticker = None
            timeframe = '1d'
            period = 'all'
        st.markdown('---')
        st.header('Strategy & Optimization')
        side_opt = st.selectbox('Side', ['both','long','short'])
        search_type = st.selectbox('Search', ['random','grid'])
        random_iters = st.number_input('Random iterations', min_value=20, max_value=2000, value=200)
        desired_accuracy = st.slider('Desired accuracy', 0.5, 0.99, 0.9, step=0.01)
        min_trades = st.number_input('Min trades for strategy', min_value=1, max_value=200, value=5)
        use_points = st.checkbox('Use absolute target points in backtest & live', value=False)
        target_points = st.number_input('Target points (if using points)', min_value=1, value=50)
        precision_mode = st.checkbox('Precision mode (rarer but more precise)', value=True)
        run_wf = st.checkbox('Enable Walk-Forward CV (slower)', value=False)
        st.markdown('Optional position sizing')
        capital = st.number_input('Capital (0 = off)', min_value=0, value=0)
        risk_pct = st.number_input('Risk % per trade', min_value=0.1, max_value=10.0, value=1.0)
        st.markdown('Heatmap colors')
        heat_cmap = st.selectbox('Heatmap colormap', ['RdYlGn','coolwarm','viridis','bwr'])

    # Fetch data button (for yfinance) or process upload
    if source.startswith('Nifty'):
        if st.button('Fetch data'):
            st.info('Fetching data — yfinance rate limits may apply. If you hit rate-limit, wait a minute or use a smaller period.')
            try:
                raw = yf.download(tickers=ticker, period=period, interval=timeframe, progress=False)
            except Exception as e:
                st.error(f'Failed to download: {e}')
                return
            if raw is None or raw.empty:
                st.error('No data returned by yfinance. Try another interval/period or ticker.')
                return
            df_fetched = standardize_df_from_yf(raw)
            st.session_state['fetched_df'] = df_fetched
            st.session_state['raw_source'] = {'source':'yfinance','ticker':ticker,'period':period,'interval':timeframe}
            st.success(f'Data fetched: {len(df_fetched)} rows')
    else:
        # file upload path
        if 'upload' in locals() and upload is not None:
            try:
                if upload.name.endswith('.csv'):
                    raw = pd.read_csv(upload)
                else:
                    raw = pd.read_excel(upload)
            except Exception as e:
                st.error(f'Failed to read file: {e}')
                return
            try:
                df_fetched, mapping = standardize_df_from_file(raw)
            except Exception as e:
                st.error(f'Error mapping file: {e}'); return
            st.session_state['fetched_df'] = df_fetched
            st.session_state['raw_source'] = {'source':'file','name':upload.name}
            st.success(f'File processed: {len(df_fetched)} rows')

    # show fetched data preview
    if st.session_state['fetched_df'] is None:
        st.info('No data loaded yet. Use Fetch data or upload a file.')
        return

    df = st.session_state['fetched_df']
    st.subheader('Data preview (IST times)')
    df_display = df.copy()
    try:
        df_display['Date'] = df_display['Date'].dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S %Z')
    except Exception:
        df_display['Date'] = df_display['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    c1,c2 = st.columns([1,1])
    with c1:
        st.write('Top 5')
        st.dataframe(df_display.head())
    with c2:
        st.write('Bottom 5')
        st.dataframe(df_display.tail())

    st.write('Date range:', df['Date'].min(), 'to', df['Date'].max())

    # heatmap of returns (safe conversion to numeric 2D array)
    st.subheader('Year vs Month returns heatmap')
    df['returns'] = df['Close'].pct_change()
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    heat = df.pivot_table(values='returns', index='year', columns='month', aggfunc=lambda x: (x+1.0).prod()-1) * 100
    if heat.empty:
        st.warning('Not enough data to build heatmap')
    else:
        try:
            heat_vals = np.asarray(heat.fillna(0).astype(float).values)
            fig_h, axh = plt.subplots(figsize=(10, max(3, 0.5*len(heat.index))))
            cmap = plt.get_cmap(heat_cmap)
            im = axh.imshow(heat_vals, aspect='auto', cmap=cmap, interpolation='nearest')
            axh.set_xticks(range(len(heat.columns))); axh.set_xticklabels([str(x) for x in heat.columns])
            axh.set_yticks(range(len(heat.index))); axh.set_yticklabels([str(x) for x in heat.index])
            # annotated text with readable contrast
            norm = plt.Normalize(np.nanmin(heat_vals), np.nanmax(heat_vals))
            for (j,i), val in np.ndenumerate(heat_vals):
                rgba = cmap(norm(val))
                luminance = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                txt_color = 'white' if luminance < 0.5 else 'black'
                axh.text(i, j, f"{val:.2f}%", ha='center', va='center', fontsize=8, color=txt_color)
            fig_h.colorbar(im, ax=axh)
            st.pyplot(fig_h)
        except Exception as e:
            st.warning('Heatmap plotting failed — showing table instead: ' + str(e))
            st.dataframe(heat * 100)

    # prepare base params
    base_params = {'pivot_window':3,'cluster_tol':0.005,'zone_width':0.005,'sl_pct':0.01,'tp_pct':0.02,'max_hold':5,'breakout_lookback':5,'pattern_tol':0.02,'min_bars_between':3,'signal_threshold':3.0,'allowed_dirs':[],'use_target_points':use_points,'target_points':target_points,'volume_factor':1.2,'wick_factor':1.5}
    if side_opt in ['both','long']:
        base_params['allowed_dirs'].append('long')
    if side_opt in ['both','short']:
        base_params['allowed_dirs'].append('short')

    # optimization control: Run Optimization button
    st.subheader('Optimization (find best strategy)')
    run_opt = st.button('Run Optimization (use fetched data)')
    if run_opt:
        samples = []
        if search_type == 'random':
            samples = sample_random_params(random_iters, None)
        else:
            # small grid default if none provided
            grid = {'pivot_window':[3],'sl_pct':[0.01],'tp_pct':[0.02]}
            samples = list(grid_params(grid))
        total = len(samples)
        progress = st.progress(0)
        status = st.empty()
        best = None
        for i,s in enumerate(samples):
            params = base_params.copy(); params.update(s)
            df_signals, meta = generate_signals_and_meta(df, params)
            trades, stats = backtest(df_signals, params)
            if stats['total_trades'] < min_trades:
                metric = -np.inf
            else:
                metric = stats['accuracy'] if st.session_state.get('optimize_for','accuracy')=='accuracy' else stats['total_points']
                if stats['accuracy'] >= desired_accuracy:
                    metric += 1000
            if best is None or metric > best.get('metric', -np.inf):
                best = {'params':params,'trades':trades,'stats':stats,'meta':meta,'metric':metric}
            progress.progress(int((i+1)/total*100))
            status.text(f'Progress: {i+1}/{total}')
        progress.progress(100)
        status.success('Optimization finished')
        if best is None:
            st.warning('No suitable strategy found')
        else:
            st.session_state['best_strategy'] = best
            st.write('Best params:'); st.json(best['params'])
            st.write('Backtest stats:'); st.json(best['stats'])
            trades_df = best['trades']
            if not trades_df.empty:
                trades_df['entry_time'] = trades_df['entry_time'].dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S %Z')
                trades_df['exit_time'] = trades_df['exit_time'].dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            st.write('Sample trades'); st.dataframe(trades_df.head(50))

    # Walk-forward CV
    if run_wf:
        if st.button('Run Walk-Forward CV'):
            with st.spinner('Running walk-forward...'):
                wf_stats = walk_forward_cv(df, train_days=365*2, test_days=90, step_days=90, search_iters=60, params_base=base_params)
                st.write('Completed windows:', len(wf_stats))
                for i,w in enumerate(wf_stats):
                    st.write(f"Window {i+1}")
                    st.write('Train accuracy:', w['train_stats']['accuracy'])
                    st.write('Test accuracy:', w['test_stats']['accuracy'])

    # Live recommendation (use best if exists)
    st.subheader('Live recommendation (same logic as backtest)')
    if st.button('Generate Live Recommendation'):
        strategy = st.session_state.get('best_strategy')
        if strategy is not None:
            params = strategy['params']
        else:
            params = base_params
        params['use_target_points'] = use_points
        params['target_points'] = target_points if use_points else None
        df_signals, meta = generate_signals_and_meta(df, params)
        trades_bt, stats_bt = backtest(df_signals, params)
        last_sig_row = df_signals.iloc[-1]
        sig = int(last_sig_row['signal'])
        if sig==0:
            st.write('No signal on last closed candle.')
        else:
            next_open_est = float(df_signals['Close'].iloc[-1])
            if params['use_target_points'] and params.get('target_points'):
                tp = next_open_est + params['target_points'] if sig==1 else next_open_est - params['target_points']
            else:
                tp = next_open_est*(1+params['tp_pct']) if sig==1 else next_open_est*(1-params['tp_pct'])
            sl = next_open_est*(1-params['sl_pct']) if sig==1 else next_open_est*(1+params['sl_pct'])
            unit_risk = abs(next_open_est - sl)
            suggested_units = None
            if capital>0 and unit_risk>0:
                suggested_units = int((capital*(risk_pct/100.0))//unit_risk)
            rec = {'direction': 'long' if sig==1 else 'short','entry_next_open_est':round(next_open_est,4),'stop_loss':round(sl,4),'target':round(tp,4),'reason': last_sig_row['reason'],'probability_of_profit': stats_bt.get('accuracy',None),'suggested_units': suggested_units}
            st.json(rec)
            st.write('Strategy backtest stats:'); st.json(stats_bt)

    # Chart with overlays
    st.subheader('Chart with overlays (last 200 bars)')
    best = st.session_state.get('best_strategy')
    if best is not None:
        params = best['params']
        df_signals, meta = generate_signals_and_meta(df, params)
        trades_df = best['trades']
    else:
        params = base_params
        df_signals, meta = generate_signals_and_meta(df, params)
        trades_df = pd.DataFrame()
    N = min(200, len(df))
    df_plot = df.tail(N).reset_index(drop=True)
    fig = plot_with_overlays(df_plot, meta, trades=trades_df, title=f"{st.session_state.get('raw_source',{}).get('ticker','TICKER')} — last {N} bars")
    st.pyplot(fig)


if __name__ == '__main__':
    app()
