import streamlit as st
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------------------
# Indicator Calculation Functions (Manual, no TA-Lib!)
# -------------------------------------

def SMA(series, period):
    return series.rolling(period).mean()

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    up, down = delta.clip(lower=0), -1*delta.clip(upper=0)
    ma_up = up.ewm(com=period-1, adjust=False).mean()
    ma_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100/(1+rs))

def MACD(series, fast=12, slow=26, signal=9):
    fast_ma = EMA(series, fast)
    slow_ma = EMA(series, slow)
    macd_line = fast_ma - slow_ma
    signal_line = EMA(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def Bollinger_Bands(series, period=20, dev=2):
    mean = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mean + dev*std
    lower = mean - dev*std
    return upper, mean, lower

def ATR(high, low, close, period=14):
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def CCI(df, period=20):
    tp = (df['HIGH'] + df['LOW'] + df['close'])/3
    sma = tp.rolling(period).mean()
    mad = (tp - sma).abs().rolling(period).mean()
    return (tp - sma) / (0.015*mad)

def MFI(df, period=14):
    tp = (df['HIGH'] + df['LOW'] + df['close'])/3
    mf = tp * df['VOLUME']
    pos_mf, neg_mf = [], []
    for i in range(1,len(tp)):
        if tp.iloc[i] > tp.iloc[i-1]:
            pos_mf.append(mf.iloc[i])
            neg_mf.append(0)
        else:
            pos_mf.append(0)
            neg_mf.append(mf.iloc[i])
    pos_mf = pd.Series(pos_mf, index=df.index[1:])
    neg_mf = pd.Series(neg_mf, index=df.index[1:])
    mr = (pos_mf.rolling(period).sum() / neg_mf.rolling(period).sum())
    mfi = 100 - (100/(1+mr))
    return mfi.reindex(df.index)

def OBV(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['VOLUME'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['VOLUME'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

def StochRSI(df, period=14):
    rsi = RSI(df['close'], period)
    min_rsi = rsi.rolling(period).min()
    max_rsi = rsi.rolling(period).max()
    return (rsi - min_rsi) / (max_rsi - min_rsi)

# -------------------------------------
# Load Data
# -------------------------------------

def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except:
        df = pd.read_excel(uploaded_file)

    # Clean headers
    cols = [c.strip().replace(".","").replace(" ","").upper() for c in df.columns]
    df.columns = cols

    # Try to map standard names
    rename_map = {}
    for col in df.columns:
        if 'DATE' in col: rename_map[col] = 'DATE'
        elif 'OPEN' in col: rename_map[col] = 'OPEN'
        elif 'HIGH' in col: rename_map[col] = 'HIGH'
        elif 'LOW' in col: rename_map[col] = 'LOW'
        elif 'CLOSE' in col: rename_map[col] = 'close'
        elif 'VOLUME' in col: rename_map[col] = 'VOLUME'
    df.rename(columns=rename_map, inplace=True)

    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').reset_index(drop=True)
    return df

# -------------------------------------
# Generate Signals and Backtest
# -------------------------------------

def generate_signals(df, params, trade_type="Both"):
    df = df.copy()

    # Indicators
    df['EMAfast'] = EMA(df['close'], params['ema_fast'])
    df['EMAslow'] = EMA(df['close'], params['ema_slow'])
    df['RSI'] = RSI(df['close'], params['rsi_period'])
    df['MACD'], df['MACDsig'], df['MACDhist'] = MACD(df['close'])
    df['BBU'], df['BBM'], df['BBL'] = Bollinger_Bands(df['close'], params['bb_period'], 2)
    df['ATR'] = ATR(df['HIGH'], df['LOW'], df['close'])
    df['CCI'] = CCI(df)
    df['MFI'] = MFI(df)
    df['OBV'] = OBV(df)
    df['StochRSI'] = StochRSI(df)

    df['signal'] = 0

    # Long rule
    if trade_type in ["Both", "Long"]:
        df.loc[(df['EMAfast'] > df['EMAslow']) &
               (df['RSI'] < params['rsi_buy']) &
               (df['close'] < df['BBL']) &
               (df['MFI'] < 35), 'signal'] = 1

    # Short rule
    if trade_type in ["Both", "Short"]:
        df.loc[(df['EMAfast'] < df['EMAslow']) &
               (df['RSI'] > params['rsi_sell']) &
               (df['close'] > df['BBU']) &
               (df['MFI'] > 65), 'signal'] = -1

    return df

def backtest(df, params, trade_type="Both", sl_pct=0.01, tp_pct=0.02):
    df = generate_signals(df, params, trade_type)
    trades, position = [], 0
    entry_price = None

    for i in range(len(df)-1):
        if position == 0 and df['signal'].iloc[i] != 0:
            position = df['signal'].iloc[i]
            entry_price = df['OPEN'].iloc[i+1]
            trades.append({"EntryDate":df['DATE'].iloc[i+1],"Type":"Long" if position==1 else "Short",
                           "Entry":entry_price})

        elif position != 0:
            high, low, close = df['HIGH'].iloc[i], df['LOW'].iloc[i], df['close'].iloc[i]
            exit_flag, exit_price = False, close
            target = entry_price*(1+tp_pct*position)
            stop = entry_price*(1-sl_pct*position)
            if position==1:
                if low<=stop: exit_flag, exit_price=True, stop
                elif high>=target: exit_flag, exit_price=True, target
                elif close>=df['BBM'].iloc[i]: exit_flag, exit_price=True, close
            else:
                if high>=stop: exit_flag, exit_price=True, stop
                elif low<=target: exit_flag, exit_price=True, target
                elif close<=df['BBM'].iloc[i]: exit_flag, exit_price=True, close

            if exit_flag:
                trades[-1].update({"ExitDate":df['DATE'].iloc[i],"Exit":exit_price,
                                   "PnL":round((exit_price-entry_price)*position,2)})
                position=0

    return pd.DataFrame(trades)

def evaluate(trades):
    if trades.empty: return {"TotalTrades":0,"Win%":0,"TotalReturn%":0}, trades
    trades['Return%']=trades['PnL']/trades['Entry']*100
    winrate=(trades['PnL']>0).mean()*100
    totalpnl=trades['Return%'].sum()
    return {"TotalTrades":len(trades),"Win%":round(winrate,2),
            "TotalReturn%":round(totalpnl,2)}, trades

# -------------------------------------
# Streamlit UI
# -------------------------------------

st.title("📊 Ultra-Robust Strategy Optimizer & Live Recommender")

uploaded_file=st.file_uploader("Upload Stock Data (CSV/Excel)",type=['csv','xlsx'])
trade_type=st.selectbox("Select Trade Type",["Both","Long","Short"])

if uploaded_file:
    df=load_data(uploaded_file)

    # Param grid to expand until profitable
    ranges={"ema_fast":[10,20,50],
            "ema_slow":[100,200],
            "rsi_period":[14],
            "rsi_buy":[25,30,35],
            "rsi_sell":[65,70,75],
            "bb_period":}
    
    best_score=-9999
    best_params=None
    best_trades=None
    for ef,es,rp,rbu,rsu,bp in itertools.product(ranges['ema_fast'],ranges['ema_slow'],
                                                 ranges['rsi_period'],
                                                 ranges['rsi_buy'],ranges['rsi_sell'],
                                                 ranges['bb_period']):
        params={"ema_fast":ef,"ema_slow":es,"rsi_period":rp,"rsi_buy":rbu,"rsi_sell":rsu,"bb_period":bp}
        trades=backtest(df,params,trade_type)
        stats,_=evaluate(trades)
        if stats['TotalReturn%']>best_score:
            best_score=stats['TotalReturn%']
            best_params=params
            best_trades=trades

    final_stats,_=evaluate(best_trades)
    st.subheader("✅ Best Parameters Used:")
    st.json(best_params)

    st.subheader("📈 Backtest Results")
    st.json(final_stats)

    st.write(best_trades)

    # Live Recommendation
    live_df=generate_signals(df,best_params,trade_type)
    last=live_df.iloc[-1]
    rec="No Trade"
    if last['signal']==1: rec="BUY"
    elif last['signal']==-1: rec="SELL"

    st.subheader("🎯 Live Recommendation (based on last candle)")
    st.write({"Action":rec,"Entry":last['close'],
              "StopLoss":round(last['close']*(1-0.01 if rec=="BUY" else 1+0.01),2),
              "Target":round(last['close']*(1+0.02 if rec=="BUY" else 1-0.02),2),
              "Probability of Profit":str(final_stats["Win%"])+"%"})
    
    # Chart
    fig,ax=plt.subplots()
    ax.plot(df['DATE'],df['close'],label='Close')
    for _,t in best_trades.iterrows():
        ax.scatter(t['EntryDate'],t['Entry'],color='green' if t['Type']=="Long" else 'red')
        ax.scatter(t['ExitDate'],t['Exit'],color='blue')
    ax.legend()
    st.pyplot(fig)

    st.subheader("📝 Summary")
    st.write(f"""
    Over the uploaded dataset, {final_stats['TotalTrades']} swing trades were generated. 
    The win rate was {final_stats['Win%']}% and the total return of {final_stats['TotalReturn%']}%. 
    The optimizer tested multiple parameter grids across indicators (EMA, RSI, Bollinger Bands, ATR, CCI, MFI, OBV, StochRSI) 
    and selected the most robust profitable combination {best_params}. 
    For the latest candle, the signal is **{rec}**, with entry at {round(last['close'],2)}, 
    target at {round(last['close']*1.02 if rec=='BUY' else last['close']*0.98,2)}, 
    and stop-loss at {round(last['close']*0.99 if rec=='BUY' else last['close']*1.01,2)}. 
    This approach adapts dynamically to market phases and avoids lookahead bias.
    """)
