import streamlit as st
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import os

# -----------------------------
# Utility Helpers
# -----------------------------

def clean_columns(df):
    """Standardize and deduplicate column names"""
    df.columns = df.columns.str.strip().str.lower()
    df = df.loc[:, ~df.columns.duplicated()]
    col_map = {
        'date': 'date', 'timestamp': 'date',
        'open': 'open', 'high': 'high', 'low': 'low',
        'close': 'close', 'adj close': 'close', 'ltp': 'close',
        'volume': 'volume', 'shares': 'volume'
    }
    df.rename(columns={c: col_map.get(c, c) for c in df.columns}, inplace=True)
    return df

def load_data(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1]
    if ext in [".xls", ".xlsx"]:
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    df = clean_columns(df)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['date'])
        df = df.sort_values('date')
        df.set_index('date', inplace=True)
    numeric_cols = ['open','high','low','close','volume']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['close'])
    return df

# -----------------------------
# Technical Indicators (Manual)
# -----------------------------

def SMA(series, period): return series.rolling(period).mean()

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.rolling(period).mean() / down.rolling(period).mean()
    return 100 - (100 / (1 + rs))

def MACD(series, fast=12, slow=26, signal=9):
    fast_ema, slow_ema = EMA(series, fast), EMA(series, slow)
    macd = fast_ema - slow_ema
    signal_line = EMA(macd, signal)
    hist = macd - signal_line
    return macd, signal_line, hist

def Bollinger(series, period=20, std_factor=2):
    sma = SMA(series, period)
    std = series.rolling(period).std()
    return sma + std_factor*std, sma, sma - std_factor*std

def ATR(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = high_low.combine(high_close, max).combine(low_close, max)
    return SMA(tr, period)

def Stochastic(df, k_period=14, d_period=3):
    low_min = df['low'].rolling(k_period).min()
    high_max = df['high'].rolling(k_period).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    d = k.rolling(d_period).mean()
    return k, d

def CCI(df, period=20):
    tp = (df['high']+df['low']+df['close'])/3
    sma = SMA(tp, period)
    mad = (tp - sma).abs().rolling(period).mean()
    return (tp - sma) / (0.015 * mad)

def OBV(df):
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv

def MFI(df, period=14):
    tp = (df['high']+df['low']+df['close'])/3
    mf = tp * df['volume']
    pos, neg = [], []
    for i in range(1,len(df)):
        if tp.iloc[i] > tp.iloc[i-1]:
            pos.append(mf.iloc[i]); neg.append(0)
        else:
            pos.append(0); neg.append(mf.iloc[i])
    pos, neg = pd.Series([0]+pos,index=df.index), pd.Series(+neg,index=df.index)
    mr = pos.rolling(period).sum() / neg.rolling(period).sum()
    return 100 - 100/(1+mr)

def ADX(df, period=14):
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff() * -1
    plus_dm[plus_dm<0]=0; minus_dm[minus_dm<0]=0
    atr = ATR(df,period)
    plus_di = 100*(plus_dm.ewm(alpha=1/period).mean()/atr)
    minus_di = 100*(minus_dm.ewm(alpha=1/period).mean()/atr)
    dx = (abs(plus_di-minus_di)/(plus_di+minus_di))*100
    return dx.ewm(alpha=1/period).mean()

# -----------------------------
# Feature Engineering
# -----------------------------

def add_indicators(df):
    df['sma50'] = SMA(df['close'],50)
    df['ema20'] = EMA(df['close'],20)
    df['ema200'] = EMA(df['close'],200)
    df['rsi14'] = RSI(df['close'],14)
    bb_up, bb_mid, bb_low = Bollinger(df['close'])
    df['bb_up'],df['bb_mid'],df['bb_low'] = bb_up,bb_mid,bb_low
    df['atr14'] = ATR(df,14)
    macd,macd_sig,macd_hist=MACD(df['close'])
    df['macd'],df['macd_sig'],df['macd_hist']=macd,macd_sig,macd_hist
    k,d = Stochastic(df); df['stoch_k'],df['stoch_d']=k,d
    df['cci20']=CCI(df,20)
    df['obv']=OBV(df)
    df['mfi14']=MFI(df)
    df['adx14']=ADX(df)
    return df

# -----------------------------
# Strategy Backtesting
# -----------------------------

def generate_signals(df, params):
    df = df.copy()
    df['signal']=0

    # Hybrid conditions
    if params['trend']=='ema':
        df['trend_flag']=np.where(df['ema20']>df['ema200'],1,-1)
    else:
        df['trend_flag']=np.where(df['sma50']>df['ema200'],1,-1)

    for i in range(len(df)):
        if df['trend_flag'].iloc[i]==1 and df['rsi14'].iloc[i]<params['rsi_buy'] and df['close'].iloc[i]<df['bb_low'].iloc[i]:
            df.at[df.index[i],'signal']=1
        elif df['trend_flag'].iloc[i]==-1 and df['rsi14'].iloc[i]>params['rsi_sell'] and df['close'].iloc[i]>df['bb_up'].iloc[i]:
            df.at[df.index[i],'signal']=-1

    return df

def backtest(df, params, sl_factor=1.0, tp_factor=2.0):
    df = generate_signals(df,params)
    trades=[]; pos=0; entry=0
    for i in range(1,len(df)-1):
        if pos==0 and df['signal'].iloc[i]!=0:
            pos=df['signal'].iloc[i]
            entry=df['open'].iloc[i+1]
            trades.append({"date":df.index[i],"type":"LONG" if pos==1 else "SHORT","entry":entry})
        elif pos!=0:
            sl = entry*(1-pos*sl_factor*0.01)
            tp = entry*(1+pos*tp_factor*0.01)
            price=df['close'].iloc[i]
            hit_exit=False
            if (pos==1 and (df['low'].iloc[i]<=sl or df['high'].iloc[i]>=tp)):
                exitp=tp if df['high'].iloc[i]>=tp else sl
                hit_exit=True
            elif (pos==-1 and (df['high'].iloc[i]>=sl or df['low'].iloc[i]<=tp)):
                exitp=tp if df['low'].iloc[i]<=tp else sl
                hit_exit=True
            elif (pos==1 and df['close'].iloc[i]>=df['bb_mid'].iloc[i]) or (pos==-1 and df['close'].iloc[i]<=df['bb_mid'].iloc[i]):
                exitp=price; hit_exit=True
            if hit_exit:
                trades[-1].update({"exit":exitp,"exit_date":df.index[i],"pnl":(exitp-entry)*pos})
                pos=0
    return trades

def evaluate_trades(trades):
    if not trades: return {"Total Trades":0},pd.DataFrame()
    df=pd.DataFrame(trades)
    df['return%']=df['pnl']/df['entry']*100
    winrate=(df['pnl']>0).mean()*100
    return {
        "Total Trades":len(df),
        "Win %":round(winrate,2),
        "Total Return %":round(df['return%'].sum(),2),
        "Avg Return %":round(df['return%'].mean(),2),
        "Profit Factor":round(df[df.pnl>0]['pnl'].sum()/abs(df[df.pnl<0]['pnl'].sum()),2) if any(df['pnl']<0) else np.inf
    },df

# -----------------------------
# Streamlit App
# -----------------------------

st.title("📊 Ultra-Robust Strategy Optimizer & Live Signal App")

file=st.file_uploader("Upload Stock file (CSV/Excel)",type=["csv","xls","xlsx"])
if file:
    df=load_data(file)
    df=add_indicators(df)

    # Parameter Grid
    grid={
        'rsi_buy':[20,25,30,35],
        'rsi_sell':[65,70,75,80],
        'trend':['ema','sma']
    }

    best_score=-1;best_params=None;best_trades=[]
    for rsi_b,rsi_s,tr in itertools.product(grid['rsi_buy'],grid['rsi_sell'],grid['trend']):
        params={'rsi_buy':rsi_b,'rsi_sell':rsi_s,'trend':tr}
        trades=backtest(df,params)
        stats,_=evaluate_trades(trades)
        score=stats.get('Total Return %',-999)
        if score>best_score:
            best_score=score;best_params=params;best_trades=trades;best_stats=stats

    st.subheader("Best Parameters Found")
    st.json(best_params)
    st.subheader("Backtest Performance")
    st.json(best_stats)
    st.write(pd.DataFrame(best_trades))

    # Live Recommendation from last candle
    last_df=generate_signals(df,best_params)
    last_signal=last_df['signal'].iloc[-1]
    live_msg="No trade setup right now."
    live_rec={}
    if last_signal!=0:
        entry=df['close'].iloc[-1]
        sl=entry*(1-last_signal*0.01)
        tp=entry*(1+last_signal*0.02)
        direction="BUY (LONG)" if last_signal==1 else "SELL (SHORT)"
        live_rec={"Direction":direction,"Entry":entry,"SL":sl,"TP":tp,"Prob(Win%)":best_stats['Win %']}
        live_msg=f"Recommendation: {direction} at {entry} with SL {sl}, TP {tp}. Historical win probability {best_stats['Win %']}%."

    st.subheader("Live Recommendation")
    st.write(live_msg)
    st.json(live_rec)

    # Human-readable summary
    st.subheader("📘 Summary")
    summary=f"Over the past year, the optimizer tested multiple indicator combinations and found the best swing-trading parameters {best_params}. "\
            f"Backtesting produced {best_stats['Total Trades']} trades with {best_stats['Win %']}% accuracy and "\
            f"a total return of {best_stats['Total Return %']}%. The system adapts to both up and down trends. "\
            f"Currently, based on the last candle, {live_msg if live_signal!=0 else 'there is no active trade setup'}. "\
            "This strategy uses a mix of EMA trend filter, RSI oversold/overbought levels, and Bollinger band mean reversion for robust swing trades."
    st.write(summary)
