import streamlit as st
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

from datetime import datetime

# -----------------------------
# Helper: Flexible Column Mapping
# -----------------------------
def normalize_columns(df):
    col_map = {}
    for col in df.columns:
        lower = col.strip().lower()
        if 'date' in lower: col_map[col] = 'date'
        elif 'open' in lower: col_map[col] = 'open'
        elif 'high' in lower: col_map[col] = 'high'
        elif 'low' in lower: col_map[col] = 'low'
        elif 'close' in lower: col_map[col] = 'close'
        elif 'volume' in lower: col_map[col] = 'volume'
    df = df.rename(columns=col_map)
    return df

# -----------------------------
# Manual Indicator Calculations
# -----------------------------
def SMA(series, period): return series.rolling(period).mean()
def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    gain, loss = np.where(delta > 0, delta, 0), np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def MACD(close, fast=12, slow=26, signal=9):
    macd = EMA(close, fast) - EMA(close, slow)
    signal_line = EMA(macd, signal)
    hist = macd - signal_line
    return macd, signal_line, hist

def Bollinger(series, period=20, dev=2):
    mid = SMA(series, period)
    std = series.rolling(period).std()
    upper, lower = mid + dev * std, mid - dev * std
    return upper, mid, lower

def ATR(df, period=14):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def Stochastic(df, k=14, d=3):
    low_min = df['low'].rolling(k).min()
    high_max = df['high'].rolling(k).max()
    stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d

def CCI(df, period=20):
    tp = (df['high'] + df['low'] + df['close'])/3
    sma = tp.rolling(period).mean()
    mad = (tp - sma).abs().rolling(period).mean()
    return (tp - sma) / (0.015 * mad)

def OBV(close, volume):
    obv = [0]
    for i in range(1, len(close)):
        change = 1 if close.iloc[i] > close.iloc[i-1] else -1 if close.iloc[i] < close.iloc[i-1] else 0
        obv.append(obv[-1] + change*volume.iloc[i])
    return pd.Series(obv, index=close.index)

def MFI(df, period=14):
    tp = (df['high']+df['low']+df['close'])/3
    mf = tp * df['volume']
    pos_mf, neg_mf = [], []
    for i in range(1,len(tp)):
        if tp.iloc[i] > tp.iloc[i-1]: pos_mf.append(mf.iloc[i])
        else: pos_mf.append(0)
        if tp.iloc[i] < tp.iloc[i-1]: neg_mf.append(mf.iloc[i])
        else: neg_mf.append(0)
    pos_mf = pd.Series(+pos_mf).rolling(period).sum()
    neg_mf = pd.Series(+neg_mf).rolling(period).sum()
    mfi = 100 * (pos_mf / (pos_mf+neg_mf))
    return mfi

def ROC(series, period=12): return series.diff(period)/series.shift(period)*100
def VWAP(df):
    return (df['close']*df['volume']).cumsum() / df['volume'].cumsum()

# -----------------------------
# Signal Generation
# -----------------------------
def generate_signals(df, params):
    df = df.copy()
    df['signal'] = 0
    
    # Trend filter (EMA crossover)
    df['ema_fast'] = EMA(df['close'], params['ema_fast'])
    df['ema_slow'] = EMA(df['close'], params['ema_slow'])
    df['trend'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
    
    # Indicators
    df['rsi'] = RSI(df['close'], params['rsi_period'])
    df['stoch_k'], df['stoch_d'] = Stochastic(df, 14,3)
    df['cci'] = CCI(df, 20)
    df['obv'] = OBV(df['close'], df['volume'])
    df['mfi'] = MFI(df)
    df['roc'] = ROC(df['close'])
    df['vwap'] = VWAP(df)
    df['bb_up'], df['bb_mid'], df['bb_low'] = Bollinger(df['close'], 20, 2)
    
    # Entry condition
    long_cond = (df['trend']==1) & (df['rsi']<params['rsi_buy']) & (df['close']<df['bb_low'])
    short_cond= (df['trend']==-1)& (df['rsi']>params['rsi_sell']) & (df['close']>df['bb_up'])
    df.loc[long_cond, 'signal'] = 1
    df.loc[short_cond,'signal'] = -1
    return df

# -----------------------------
# Backtest
# -----------------------------
def backtest(df, params, sl_percent=0.01, tp_percent=0.02):
    df = generate_signals(df, params)
    trades, position, entry_price = [], 0, 0
    
    for i in range(len(df)-1):
        if position == 0:
            if df['signal'].iloc[i] == 1:
                position, entry_price = 1, df['open'].iloc[i+1]
                trades.append({'date':df.index[i],'type':'BUY','entry':entry_price})
            elif df['signal'].iloc[i] == -1:
                position, entry_price = -1, df['open'].iloc[i+1]
                trades.append({'date':df.index[i],'type':'SELL','entry':entry_price})
        else:
            high, low, close = df['high'].iloc[i], df['low'].iloc[i], df['close'].iloc[i]
            target, stop = entry_price*(1+tp_percent*position), entry_price*(1-sl_percent*position)
            exit_flag, exit_price = False, close
            if position==1:
                if low<=stop: exit_flag,exit_price=True,stop
                elif high>=target: exit_flag,exit_price=True,target
                elif close>=df['bb_mid'].iloc[i]: exit_flag,exit_price=True,close
            else:
                if high>=stop: exit_flag,exit_price=True,stop
                elif low<=target: exit_flag,exit_price=True,target
                elif close<=df['bb_mid'].iloc[i]: exit_flag,exit_price=True,close
            if exit_flag:
                trades[-1].update({'exit_date':df.index[i],'exit':exit_price,
                                   'pnl':round((exit_price-entry_price)*position,2)})
                position=0
    return trades

def evaluate_trades(trades):
    if not trades: return {}
    df=pd.DataFrame(trades)
    df['return_%']=df['pnl']/df['entry']*100
    winrate=(df['pnl']>0).mean()*100
    return {
        'Total Trades': len(df),
        'Winning %': round(winrate,2),
        'Total Return %': round(df['return_%'].sum(),2),
        'Avg Return %': round(df['return_%'].mean(),2),
        'Profit Factor': round(df[df.pnl>0]['pnl'].sum()/abs(df[df.pnl<0]['pnl'].sum()) if (df.pnl<0).any() else np.inf,2)
    }, df

# -----------------------------
# Streamlit Frontend
# -----------------------------
st.title("📊 Ultra-Robust Swing Trading Optimizer")

uploaded = st.file_uploader("Upload Stock CSV/Excel", type=['csv','xlsx'])
if uploaded:
    df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
    df = normalize_columns(df)
    df['date']=pd.to_datetime(df['date'],errors='coerce'); df=df.set_index('date').sort_index()
    
    # Grid Parameters
    param_grid = {
        'ema_fast':[10,20,50], 'ema_slow':[100,150,200],
        'rsi_period':[10,14], 'rsi_buy':[25,30,35], 'rsi_sell':[65,70,75]
    }
    
    best_score,best_params,best_trades=-np.inf,None,None
    for ef,es,rp,rb,rs in itertools.product(
        param_grid['ema_fast'],param_grid['ema_slow'],
        param_grid['rsi_period'],param_grid['rsi_buy'],param_grid['rsi_sell']):
        
        params={'ema_fast':ef,'ema_slow':es,'rsi_period':rp,'rsi_buy':rb,'rsi_sell':rs}
        trades = backtest(df, params)
        stats,_ = evaluate_trades(trades)
        score=stats.get('Total Return %',-999)
        if score>best_score:
            best_score, best_params,best_trades=score,params,trades
    
    st.subheader("✅ Best Optimized Parameters")
    st.json(best_params)
    stats,log=evaluate_trades(best_trades)
    st.subheader("📈 Backtest Results")
    st.json(stats)
    st.write(log)
    
    # Live Recommendation
    st.subheader("📌 Live Recommendation (based on last candle)")
    df_last = generate_signals(df, best_params).iloc[-1]
    if df_last['signal']==1:
        entry=df_last['close']; sl=entry*(1-0.01); target=entry*(1+0.02)
        st.success(f"BUY Signal | Entry={entry:.2f} SL={sl:.2f} Target={target:.2f}")
    elif df_last['signal']==-1:
        entry=df_last['close']; sl=entry*(1+0.01); target=entry*(1-0.02)
        st.error(f"SELL Signal | Entry={entry:.2f} SL={sl:.2f} Target={target:.2f}")
    else: st.info("No signal as per strategy")
