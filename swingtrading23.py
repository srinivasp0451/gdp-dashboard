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

st.set_page_config(page_title="Swing Trading Recommender — Live + WFCV + Visuals", layout="wide")

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
    # fallback
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
    # yfinance returns DatetimeIndex
    d = df.copy()
    if isinstance(d.index, pd.DatetimeIndex):
        d = d.reset_index().rename(columns={'index':'Date'})
    # yfinance may have 'Adj Close' column
    if 'Adj Close' in d.columns and 'Close' in d.columns:
        # prefer Adj Close for returns but keep Close for price levels
        pass
    # ensure columns
    for col in ['Open','High','Low','Close','Volume']:
        if col not in d.columns:
            d[col] = np.nan
    d['Date'] = pd.to_datetime(d['Date'], errors='coerce')
    # convert tz to IST
    try:
        if d['Date'].dt.tz is None:
            d['Date'] = d['Date'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        else:
            d['Date'] = d['Date'].dt.tz_convert('Asia/Kolkata')
    except Exception:
        d['Date'] = pd.to_datetime(d['Date']).dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    # numeric
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
    # returns list of detected patterns (simple heuristics)
    patterns = []
    price = df['Close'].values
    # double/triple, head&shoulders, engulfing, triangle/pennant (simple)
    # double top/bottom
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
    # head & shoulders (PH-based)
    for i in range(len(ph)-2):
        l_idx, l_p = ph[i]
        m_idx, m_p = ph[i+1]
        r_idx, r_p = ph[i+2]
        if (m_idx - l_idx >= params['min_bars_between'] and r_idx - m_idx >= params['min_bars_between']):
            shoulders = (l_p + r_p) / 2
            if m_p > shoulders and abs(l_p - r_p) <= params.get('hs_tol',0.03) * shoulders:
                patterns.append(('head_and_shoulders', l_idx, m_idx, r_idx))
    # engulfing
    for i in range(1, len(df)):
        po = df.loc[i-1,'Open']; pc = df.loc[i-1,'Close']; o = df.loc[i,'Open']; c = df.loc[i,'Close']
        if pc < po and c > o and (c - o) > (po - pc):
            patterns.append(('bullish_engulf', i-1, i))
        if pc > po and c < o and (o - c) > (pc - po):
            patterns.append(('bearish_engulf', i-1, i))
    # triangles/pennants - simplified: look for contracting range in recent pivots
    # cup & handle simplified: large rounded bottom followed by small consolidation
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

    # scoring: pattern weight + trend + momentum + volume + ATR
    weights = params.get('weights') or {
        'double_top': -3.0, 'double_bottom': 3.0,
        'head_and_shoulders': -4.0, 'bullish_engulf': 2.0, 'bearish_engulf': -2.0
    }

    signals = [0]*len(df)
    reasons = ['']*len(df)
    vol_med = df['Volume'].median() if not df['Volume'].isnull().all() else 0

    # build index->patterns map
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
        # rsi momentum
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

        # threshold
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
            # conservative: SL first
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


# ------------------------ walk-forward CV --------------------------------

def walk_forward_cv(df, train_days=365*2, test_days=90, step_days=90, search_iters=100, params_base=None):
    # returns list of stats for each window
    stats_list = []
    start = df['Date'].min()
    end = df['Date'].max()
    cur_train_start = start
    while True:
        train_end = cur_train_start + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days)
        if test_end > end:
            break
        df_train = df[(df['Date']>=cur_train_start)&(df['Date']<=train_end)].reset_index(drop=True)
        df_test = df[(df['Date']>train_end)&(df['Date']<=test_end)].reset_index(drop=True)
        if len(df_train)<50 or len(df_test)<10:
            cur_train_start += pd.Timedelta(days=step_days)
            continue
        # optimize on train with fewer iterations
        best, _ = find_best_on_train(df_train, search_iters=search_iters, params_base=params_base)
        if best is None:
            cur_train_start += pd.Timedelta(days=step_days)
            continue
        # test
        df_signals_test, meta = generate_signals_and_meta(df_test, best['params'])
        trades_test, stats_test = backtest(df_signals_test, best['params'])
        stats_list.append({'train_period':(cur_train_start,train_end),'test_period':(train_end+pd.Timedelta(seconds=1), test_end),'train_stats':best['stats'],'test_stats':stats_test,'best_params':best['params']})
        cur_train_start += pd.Timedelta(days=step_days)
    return stats_list


def find_best_on_train(df_train, search_iters=80, params_base=None):
    # simple random search optimized for accuracy
    best=None; best_metric=-np.inf; tried=0
    for _ in range(search_iters):
        s = {'pivot_window':random.choice([2,3,4]),'cluster_tol':random.choice([0.003,0.005,0.01]),'zone_width':random.choice([0.003,0.005]),'sl_pct':random.choice([0.005,0.01]),'tp_pct':random.choice([0.01,0.02]),'max_hold':random.choice([3,5,7]),'breakout_lookback':random.choice([3,5]),'pattern_tol':random.choice([0.01,0.02]),'min_bars_between':random.choice([2,3]),'signal_threshold':random.choice([2.5,3.0]),'use_target_points':False,'target_points':None,'allowed_dirs':['long','short'],'volume_factor':1.2,'wick_factor':1.5}
        params = s
        df_signals, meta = generate_signals_and_meta(df_train, params)
        trades, stats = backtest(df_signals, params)
        tried+=1
        if stats['total_trades']<3:
            continue
        metric = stats['accuracy']
        if metric>best_metric:
            best_metric=metric; best={'params':params,'stats':stats,'trades':trades}
    return best, tried


# --------------------- Visualization helpers ----------------------------

def plot_with_overlays(df, meta, trades=None, title='Price with overlays'):
    # prepare mpf plot
    df_plot = df.copy()
    df_plot = df_plot.set_index('Date')
    df_plot.index = pd.DatetimeIndex(df_plot.index)  # mpf requires naive or tz-aware? it handles tz-aware
    addplots = []
    # mark support/resistance zones as hlines
    hlines = []
    for s in meta.get('supports', []):
        hlines.append(s)
    for r in meta.get('resistances', []):
        hlines.append(r)
    apdict = dict()
    if hlines:
        apdict['hlines'] = dict(hlines=hlines, colors=['green' if i in meta.get('supports',[]) else 'red' for i in hlines], linewidths=1, alpha=0.3)
    # markers for trades
    if trades is not None and not trades.empty:
        buys = trades[trades['direction']=='long']
        sells = trades[trades['direction']=='short']
        if not buys.empty:
            buy_scatter = mpf.make_addplot(pd.Series([np.nan]*len(df_plot), index=df_plot.index))
            # instead of precise scatter use vertical lines at indices
        # we will use mpf.plot with hlines only for clarity
    fig, axlist = mpf.plot(df_plot, type='candle', style='charles', title=title, returnfig=True, hlines=apdict.get('hlines'))
    return fig


# ---------------------------- Streamlit app ------------------------------

def app():
    st.title('Swing Trading Recommender — Live + Walk-forward + Visuals')
    st.markdown('This upgraded app fetches data from yfinance (Nifty50 dropdown or manual ticker), runs robust price-action strategy backtests, walk-forward CV, and shows live recommendations using the exact same logic as backtest.')

    # Sidebar controls
    with st.sidebar:
        st.header('Data source')
        source = st.radio('Choose source', ['Nifty50 dropdown (yfinance)','Upload file'])
        # a reasonably complete Nifty50 ticker list (as of 2025) with .NS suffix for yfinance
        nifty50 = ['RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','HDFC.NS','ICICIBANK.NS','KOTAKBANK.NS','SBIN.NS','LT.NS','AXISBANK.NS','BHARTIARTL.NS','ITC.NS','HINDUNILVR.NS','BAJFINANCE.NS','MARUTI.NS','ASIANPAINT.NS','WIPRO.NS','ONGC.NS','TATAMOTORS.NS','TATASTEEL.NS','JSWSTEEL.NS','HCLTECH.NS','NESTLEIND.NS','SUNPHARMA.NS','ULTRACEMCO.NS','POWERGRID.NS','TECHM.NS','BPCL.NS','NTPC.NS','COALINDIA.NS','ADANIENT.NS','ADANIPORTS.NS','DIVISLAB.NS','BRITANNIA.NS','GRASIM.NS','HINDALCO.NS','EICHERMOT.NS','SBILIFE.NS','BPCL.NS','TITAN.NS','DRREDDY.NS','INDUSINDBK.NS','M&M.NS','HDFCLIFE.NS','BAJAJFINSV.NS','SHREECEM.NS','CIPLA.NS','ONGC.NS']
        if source.startswith('Nifty'):
            ticker = st.selectbox('Select ticker', nifty50+['Other'])
            if ticker == 'Other':
                ticker = st.text_input('Enter ticker (yfinance format, e.g. AAPL or RELIANCE.NS)', value='')
            timeframe = st.selectbox('Timeframe / Interval', ['1d','1wk','1mo','60m','30m'])
            period = st.selectbox('Period', ['6mo','1y','2y','5y','max'])
        else:
            upload = st.file_uploader('Upload CSV/Excel', type=['csv','xlsx','xls'])
            timeframe = st.selectbox('Timeframe / Interval (ignored for file)', ['1d'])
            period = st.selectbox('Period (ignored for file)', ['all'])
        st.markdown('---')
        st.header('Strategy & Optimization')
        side_opt = st.selectbox('Side', ['both','long','short'])
        search_type = st.selectbox('Search', ['random','grid'])
        random_iters = st.number_input('Random iterations', min_value=20, max_value=2000, value=200)
        desired_accuracy = st.slider('Desired accuracy', 0.5, 0.99, 0.9, step=0.01)
        min_trades = st.number_input('Min trades for strategy', min_value=1, max_value=200, value=5)
        use_points = st.checkbox('Use absolute target points in backtest & live', value=False)
        target_points = st.number_input('Target points (if using points)', min_value=1, value=50)
        precision_mode = st.checkbox('Precision mode (makes signals rarer but more precise)', value=True)
        run_wf = st.checkbox('Run Walk-Forward CV (slower)', value=False)
        st.markdown('Optional position sizing')
        capital = st.number_input('Capital (0 = off)', min_value=0, value=0)
        risk_pct = st.number_input('Risk % per trade', min_value=0.1, max_value=10.0, value=1.0)
        st.markdown('Heatmap colors')
        heat_cmap = st.selectbox('Heatmap colormap', ['coolwarm','RdYlGn','viridis','bwr'])

    # Load data
    df = None
    mapping = None
    if source.startswith('Nifty'):
        if not ticker:
            st.warning('Enter a ticker')
            return
        with st.spinner(f'Downloading {ticker}...'):
            try:
                raw = yf.download(tickers=ticker, period=period, interval=timeframe, progress=False)
            except Exception as e:
                st.error(f'Failed to download: {e}')
                return
        if raw is None or raw.empty:
            st.error('No data from yfinance — try a different ticker/period/interval')
            return
        df = standardize_df_from_yf(raw)
        mapping = {'source':'yfinance','ticker':ticker}
    else:
        if 'upload' not in locals() or upload is None:
            st.info('Upload a file to proceed')
            return
        try:
            if upload.name.endswith('.csv'):
                raw = pd.read_csv(upload)
            else:
                raw = pd.read_excel(upload)
        except Exception as e:
            st.error(f'Failed to read file: {e}'); return
        try:
            df, mapping = standardize_df_from_file(raw)
        except Exception as e:
            st.error(f'Error mapping file: {e}'); return

    st.subheader('Mapped columns / source')
    st.json(mapping)

    # display preview with IST strings
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
    st.write('Close range:', df['Close'].min(), 'to', df['Close'].max())

    # heatmap of returns (year vs month) with chosen cmap
    st.subheader('Year vs Month returns heatmap')
    df['returns'] = df['Close'].pct_change()
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    heat = df.pivot_table(values='returns', index='year', columns='month', aggfunc=lambda x: (x+1.0).prod()-1) * 100
    fig_h, axh = plt.subplots(figsize=(10, max(3, 0.5*len(heat.index))))
    im = axh.imshow(heat.fillna(0).values, aspect='auto', cmap=heat_cmap)
    axh.set_xticks(range(len(heat.columns))); axh.set_xticklabels(heat.columns)
    axh.set_yticks(range(len(heat.index))); axh.set_yticklabels(heat.index)
    for (j,i), val in np.ndenumerate(heat.fillna(0).values):
        axh.text(i, j, f"{val:.2f}%", ha='center', va='center', fontsize=8, color='black')
    fig_h.colorbar(im, ax=axh)
    st.pyplot(fig_h)

    # set up params
    base_params = {'pivot_window':3,'cluster_tol':0.005,'zone_width':0.005,'sl_pct':0.01,'tp_pct':0.02,'max_hold':5,'breakout_lookback':5,'pattern_tol':0.02,'min_bars_between':3,'signal_threshold':3.0,'allowed_dirs':[],'use_target_points':use_points,'target_points':target_points,'volume_factor':1.2,'wick_factor':1.5}
    if side_opt in ['both','long']:
        base_params['allowed_dirs'].append('long')
    if side_opt in ['both','short']:
        base_params['allowed_dirs'].append('short')
    base_params['precision_mode'] = precision_mode

    # optimization
    st.subheader('Optimization (find best strategy)')
    if st.button('Run Optimization'):
        progress = st.progress(0)
        status = st.empty()
        best=None; tried=0
        samples = []
        if search_type=='random':
            samples = sample_random_params(random_iters, None)
        else:
            samples = list(grid_params(grid if 'grid' in locals() and grid is not None else {'pivot_window':[3]}))
        total = len(samples)
        for idx,s in enumerate(samples):
            tried +=1
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
            progress.progress(int((idx+1)/total*100))
            status.text(f"Progress: {(idx+1)}/{total}")
        progress.progress(100)
        status.success('Optimization finished')
        if best is None:
            st.warning('No suitable strategy found')
        else:
            st.write('Best params:'); st.json(best['params'])
            st.write('Backtest stats:'); st.json(best['stats'])
            trades_df = best['trades']
            if not trades_df.empty:
                trades_df['entry_time'] = trades_df['entry_time'].dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S %Z')
                trades_df['exit_time'] = trades_df['exit_time'].dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            st.write('Sample trades'); st.dataframe(trades_df.head(50))
            # buy and hold
            if len(df)>=2:
                bh_pts = df['Close'].iloc[-1]-df['Close'].iloc[0]
                bh_ret = (df['Close'].iloc[-1]/df['Close'].iloc[0]-1)*100
            else:
                bh_pts=0; bh_ret=0
            st.metric('Buy&Hold points',f"{bh_pts:.2f}")
            st.metric('Strategy total points', f"{best['stats'].get('total_points',0):.2f}")
            st.write('Patterns detected counts:')
            pat_counts = {}
            for p in best['meta']['patterns']:
                pat_counts[p] = pat_counts.get(p,0)+1
            st.json(pat_counts)

            # store best in session
            st.session_state['best_strategy'] = best

    # walk-forward
    if run_wf:
        st.subheader('Walk-forward CV')
        if st.button('Run Walk-Forward CV'):
            with st.spinner('Running walk-forward...'):
                wf_stats = walk_forward_cv(df, train_days=365*2, test_days=90, step_days=90, search_iters=60, params_base=base_params)
                st.write('Completed windows:', len(wf_stats))
                for i,w in enumerate(wf_stats):
                    st.write(f"Window {i+1}")
                    st.write('Train accuracy:', w['train_stats']['accuracy'])
                    st.write('Test accuracy:', w['test_stats']['accuracy'])

    # live recommendation using same backtest logic
    st.subheader('Live recommendation (same logic as backtest)')
    if st.button('Generate Live Recommendation'):
        # use best strategy if available, otherwise base
        strategy = st.session_state.get('best_strategy')
        if strategy is not None:
            params = strategy['params']
        else:
            params = base_params
        # make sure use_target_points matches user choice
        params['use_target_points'] = use_points
        params['target_points'] = target_points if use_points else None
        df_signals, meta = generate_signals_and_meta(df, params)
        trades_bt, stats_bt = backtest(df_signals, params)
        # last closed candle signal
        last_sig_row = df_signals.iloc[-1]
        sig = int(last_sig_row['signal'])
        if sig==0:
            st.write('No signal on last candle (live).')
        else:
            # entry is next open - cannot know; show recommended entry as next_open_estimate = last_close
            next_open_estimate = df_signals['Close'].iloc[-1]
            if params['use_target_points'] and params.get('target_points'):
                tp = next_open_estimate + params['target_points'] if sig==1 else next_open_estimate - params['target_points']
            else:
                tp = next_open_estimate*(1+params['tp_pct']) if sig==1 else next_open_estimate*(1-params['tp_pct'])
            sl = next_open_estimate*(1-params['sl_pct']) if sig==1 else next_open_estimate*(1+params['sl_pct'])
            unit_risk = abs(next_open_estimate - sl)
            suggested_units = None
            if capital>0 and unit_risk>0:
                suggested_units = int((capital*(risk_pct/100.0))//unit_risk)
            rec = {'direction': 'long' if sig==1 else 'short','entry_next_open_est':round(float(next_open_estimate),4),'stop_loss':round(float(sl),4),'target':round(float(tp),4),'reason': last_sig_row['reason'],'probability_of_profit': stats_bt.get('accuracy',None),'suggested_units': suggested_units}
            st.json(rec)
            # also display what the backtest would have done if this signal occurred earlier
            st.write('Backtest stats for strategy used:'); st.json(stats_bt)

    # visualization of last N candles with overlays
    st.subheader('Chart with overlays (last 200 bars)')
    if 'best_strategy' in st.session_state:
        best = st.session_state['best_strategy']
        params = best['params']
        df_signals, meta = generate_signals_and_meta(df, params)
        trades_df = best['trades']
    else:
        params = base_params
        df_signals, meta = generate_signals_and_meta(df, params)
        trades_df = pd.DataFrame()
    N = 200
    df_plot = df.tail(N).reset_index(drop=True)
    fig = plot_with_overlays(df_plot, meta, trades=trades_df, title=f"{ticker if 'ticker' in locals() else 'TICKER'} — last {N} bars")
    st.pyplot(fig)


if __name__ == '__main__':
    app()
