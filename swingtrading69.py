import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time, pytz, random

st.set_page_config(page_title="Quant Trading", layout="wide")

ASSETS = {"NIFTY 50": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN", "BTC": "BTC-USD", "ETH": "ETH-USD", 
          "USDINR": "USDINR=X", "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "Gold": "GC=F", "Silver": "SI=F"}
INTERVALS = {"1m": ["1d","5d"], "5m": ["1d","1mo"], "15m": ["1mo"], "30m": ["1mo"], "1h": ["1mo"], "4h": ["1mo"],
             "1d": ["1mo","1y","2y","5y"], "1wk": ["1mo","1y","5y","10y","15y","20y"], "1mo": ["1y","2y","5y","10y","15y","20y","25y","30y"]}

if 'trading_active' not in st.session_state:
    for k,v in {'trading_active': False, 'position': None, 'trade_history': [], 'trade_logs': [], 'current_data': None,
                'highest_price': None, 'lowest_price': None, 'partial_exit_done': False, 'breakeven_activated': False}.items():
        st.session_state[k] = v

def ema(d, p): return d.ewm(span=p, adjust=False).mean()
def atr(df, p=14):
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    return tr.rolling(p).mean()
def rsi(d, p=14):
    delta = d.diff(); gain = delta.where(delta>0, 0).rolling(p).mean(); loss = -delta.where(delta<0, 0).rolling(p).mean()
    return 100 - 100/(1 + gain/loss)
def adx(df, p=14):
    pdm, mdm = df['High'].diff(), -df['Low'].diff(); pdm[pdm<0]=0; mdm[mdm<0]=0
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    atr_v = tr.rolling(p).mean(); pdi = 100*(pdm.rolling(p).mean()/atr_v); mdi = 100*(mdm.rolling(p).mean()/atr_v)
    return (100*abs(pdi-mdi)/(pdi+mdi)).rolling(p).mean()
def ema_angle(es, p=2): return np.degrees(np.arctan(es.diff(p)/p))

def fetch_data(ticker, interval, period):
    try:
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = [col[0] for col in data.columns.values]
        data = data[['Open','High','Low','Close','Volume']].copy()
        data.index = data.index.tz_localize('UTC').tz_convert('Asia/Kolkata') if data.index.tz is None else data.index.tz_convert('Asia/Kolkata')
        return data
    except: return None

def ema_cross_strat(df, cfg):
    fast, slow = cfg.get('ema_fast',9), cfg.get('ema_slow',15); min_ang = cfg.get('min_angle',1)
    entry_filt = cfg.get('entry_filter','Simple Crossover'); cpts = cfg.get('custom_points',10); atr_m = cfg.get('atr_multiplier',1.5)
    use_adx, adx_th = cfg.get('use_adx',False), cfg.get('adx_threshold',25)
    df['EMA_Fast'], df['EMA_Slow'] = ema(df['Close'], fast), ema(df['Close'], slow)
    df['EMA_Angle'], df['ATR'] = ema_angle(df['EMA_Fast'], 2), atr(df, 14)
    if use_adx: df['ADX'] = adx(df, 14)
    sigs = pd.Series(index=df.index, data=0)
    for i in range(1, len(df)):
        bc = df['EMA_Fast'].iloc[i]>df['EMA_Slow'].iloc[i] and df['EMA_Fast'].iloc[i-1]<=df['EMA_Slow'].iloc[i-1]
        brc = df['EMA_Fast'].iloc[i]<df['EMA_Slow'].iloc[i] and df['EMA_Fast'].iloc[i-1]>=df['EMA_Slow'].iloc[i-1]
        ang_ok = abs(df['EMA_Angle'].iloc[i]) >= min_ang; adx_ok = not use_adx or df['ADX'].iloc[i]>=adx_th
        csz = abs(df['Close'].iloc[i]-df['Open'].iloc[i])
        fok = csz>=cpts if entry_filt=='Custom Candle (Points)' else (csz>=df['ATR'].iloc[i]*atr_m if entry_filt=='ATR-based Candle' else True)
        if bc and ang_ok and fok and adx_ok: sigs.iloc[i]=1
        elif brc and ang_ok and fok and adx_ok: sigs.iloc[i]=-1
    df['Signal'] = sigs; return df

def pct_change_strat(df, cfg):
    pth, direc = cfg.get('pct_threshold',0.01), cfg.get('pct_direction','BUY on Fall')
    fp = df['Close'].iloc[0]; df['PctChange'] = ((df['Close']-fp)/fp)*100; df['Signal']=0
    for i in range(len(df)):
        pc = df['PctChange'].iloc[i]
        if direc=='BUY on Fall' and pc<=-pth: df['Signal'].iloc[i]=1
        elif direc=='SELL on Fall' and pc<=-pth: df['Signal'].iloc[i]=-1
        elif direc=='BUY on Rise' and pc>=pth: df['Signal'].iloc[i]=1
        elif direc=='SELL on Rise' and pc>=pth: df['Signal'].iloc[i]=-1
    return df

def calc_sl(df, i, pos, slt, slp, cfg):
    ent, sig, cur = pos['entry_price'], pos['signal'], df['Close'].iloc[i]
    if slt=='Custom Points': return ent-slp if sig==1 else ent+slp
    elif slt=='Trailing SL (Points)':
        nsl = cur-slp if sig==1 else cur+slp
        return nsl if pos['sl'] is None or ((sig==1 and nsl>pos['sl']) or (sig==-1 and nsl<pos['sl'])) else pos['sl']
    elif slt=='ATR-based':
        av = df['ATR'].iloc[i] if 'ATR' in df.columns else atr(df.iloc[:i+1],14).iloc[-1]
        return ent-av*cfg.get('atr_sl_multiplier',1.5) if sig==1 else ent+av*cfg.get('atr_sl_multiplier',1.5)
    elif slt=='Break-even After 50% Target':
        if pos.get('breakeven_activated'): return ent
        if pos['target']:
            td, cp = abs(pos['target']-ent), abs(cur-ent)
            if cp>=td*0.5: pos['breakeven_activated']=True; return ent
        return ent-slp if sig==1 else ent+slp
    else: return ent-slp if sig==1 else ent+slp

def calc_tgt(df, i, pos, tgtt, tgtp, cfg):
    ent, sig = pos['entry_price'], pos['signal']
    if tgtt=='Custom Points': return ent+tgtp if sig==1 else ent-tgtp
    elif tgtt=='ATR-based':
        av = df['ATR'].iloc[i] if 'ATR' in df.columns else atr(df.iloc[:i+1],14).iloc[-1]
        return ent+av*cfg.get('atr_target_multiplier',3) if sig==1 else ent-av*cfg.get('atr_target_multiplier',3)
    elif tgtt in ['Trailing Target (Points)','Signal-based (reverse EMA crossover)']: return None
    else: return ent+tgtp if sig==1 else ent-tgtp

def check_sig_exit(df, i, pos):
    if 'EMA_Fast' not in df.columns or i<1: return False
    if pos['signal']==1: return df['EMA_Fast'].iloc[i]<df['EMA_Slow'].iloc[i] and df['EMA_Fast'].iloc[i-1]>=df['EMA_Slow'].iloc[i-1]
    else: return df['EMA_Fast'].iloc[i]>df['EMA_Slow'].iloc[i] and df['EMA_Fast'].iloc[i-1]<=df['EMA_Slow'].iloc[i-1]

def add_log(m):
    ts = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
    st.session_state['trade_logs'].append(f"[{ts}] {m}"); st.session_state['trade_logs']=st.session_state['trade_logs'][-50:]

def reset_pos():
    for k in ['position','highest_price','lowest_price','partial_exit_done','breakeven_activated']: st.session_state[k]=None if 'price' in k or k=='position' else False

def run_backtest(df, strat_func, cfg, slt, slp, tgtt, tgtp, qty):
    trades, pos = [], None
    for i in range(1, len(df)):
        if pos is None:
            if df['Signal'].iloc[i]!=0:
                pos = {'entry_time': df.index[i], 'entry_price': df['Close'].iloc[i], 'signal': df['Signal'].iloc[i],
                       'sl': None, 'target': None, 'highest': df['Close'].iloc[i], 'lowest': df['Close'].iloc[i],
                       'partial_exit_done': False, 'breakeven_activated': False}
                pos['sl'], pos['target'] = calc_sl(df,i,pos,slt,slp,cfg), calc_tgt(df,i,pos,tgtt,tgtp,cfg)
        else:
            cp = df['Close'].iloc[i]; pos['highest'], pos['lowest'] = max(pos['highest'],cp), min(pos['lowest'],cp)
            exp, exr = None, None
            if slt=='Signal-based (reverse EMA crossover)' or tgtt=='Signal-based (reverse EMA crossover)':
                if check_sig_exit(df,i,pos): exp, exr = cp, "Reverse Signal"
            if exp is None and pos['sl']:
                if (pos['signal']==1 and cp<=pos['sl']) or (pos['signal']==-1 and cp>=pos['sl']): exp, exr = pos['sl'], "Stop Loss"
            if exp is None and pos['target']:
                if (pos['signal']==1 and cp>=pos['target']) or (pos['signal']==-1 and cp<=pos['target']):
                    if tgtt=='50% Exit at Target (Partial)' and not pos['partial_exit_done']: pos['partial_exit_done']=True
                    else: exp, exr = pos['target'], "Target Hit"
            if exp is None and slt in ['Trailing SL (Points)','Break-even After 50% Target']: pos['sl']=calc_sl(df,i,pos,slt,slp,cfg)
            if exp:
                pnl = (exp-pos['entry_price'])*qty if pos['signal']==1 else (pos['entry_price']-exp)*qty
                dur = (df.index[i]-pos['entry_time']).total_seconds()/3600
                trades.append({'entry_time': pos['entry_time'], 'exit_time': df.index[i], 'duration': dur,
                              'signal': 'LONG' if pos['signal']==1 else 'SHORT', 'entry_price': pos['entry_price'],
                              'exit_price': exp, 'sl': pos['sl'], 'target': pos['target'], 'exit_reason': exr,
                              'pnl': pnl, 'highest': pos['highest'], 'lowest': pos['lowest']})
                pos = None
    tt = len(trades); win = sum(1 for t in trades if t['pnl']>0); los = tt-win
    acc = win/tt*100 if tt>0 else 0; tpnl = sum(t['pnl'] for t in trades); avgd = sum(t['duration'] for t in trades)/tt if tt>0 else 0
    return {'trades': trades, 'total_trades': tt, 'winning_trades': win, 'losing_trades': los, 'accuracy': acc, 'total_pnl': tpnl, 'avg_duration': avgd}

st.title("🎯 Professional Quantitative Trading System")

st.sidebar.header("Configuration")
asset_name = st.sidebar.selectbox("Select Asset", list(ASSETS.keys())+["Custom"])
ticker = st.sidebar.text_input("Enter Ticker", "AAPL") if asset_name=="Custom" else ASSETS[asset_name]
interval = st.sidebar.selectbox("Interval", list(INTERVALS.keys()))
period = st.sidebar.selectbox("Period", INTERVALS[interval])
mode = st.sidebar.radio("Mode", ["Backtest","Live Trading"])
quantity = st.sidebar.number_input("Quantity", min_value=1, value=1, step=1, key="qty")
strategy = st.sidebar.selectbox("Strategy", ["EMA Crossover","Simple Buy","Simple Sell","Price Threshold","RSI-ADX-EMA","Percentage Change","AI Price Action","Custom Builder"])

cfg = {}
if strategy=="EMA Crossover":
    cfg['ema_fast'] = st.sidebar.number_input("EMA Fast", value=9, min_value=1, key="ef")
    cfg['ema_slow'] = st.sidebar.number_input("EMA Slow", value=15, min_value=1, key="es")
    cfg['min_angle'] = st.sidebar.number_input("Min Angle", value=1.0, min_value=0.0, key="ma")
    cfg['entry_filter'] = st.sidebar.selectbox("Entry Filter", ["Simple Crossover","Custom Candle (Points)","ATR-based Candle"])
    if cfg['entry_filter']=="Custom Candle (Points)": cfg['custom_points']=st.sidebar.number_input("Custom Points", value=10.0, key="cp")
    elif cfg['entry_filter']=="ATR-based Candle": cfg['atr_multiplier']=st.sidebar.number_input("ATR Multiplier", value=1.5, key="am")
    cfg['use_adx'] = st.sidebar.checkbox("Use ADX Filter")
    if cfg['use_adx']:
        cfg['adx_period'], cfg['adx_threshold'] = st.sidebar.number_input("ADX Period", value=14, min_value=1, key="ap"), st.sidebar.number_input("ADX Threshold", value=25.0, key="at")
elif strategy=="Percentage Change":
    cfg['pct_threshold'] = st.sidebar.number_input("% Threshold", value=0.01, min_value=0.001, step=0.001, key="pt")
    cfg['pct_direction'] = st.sidebar.selectbox("Direction", ["BUY on Fall","SELL on Fall","BUY on Rise","SELL on Rise"])

sl_type = st.sidebar.selectbox("Stop Loss Type", ["Custom Points","Trailing SL (Points)","ATR-based","Signal-based (reverse EMA crossover)","Break-even After 50% Target","Current Candle Low/High"])
sl_points = 10.0
if sl_type not in ["Signal-based (reverse EMA crossover)"]: sl_points = st.sidebar.number_input("SL Points", value=10.0, min_value=1.0, key="slp")
if sl_type=="ATR-based": cfg['atr_sl_multiplier'] = st.sidebar.number_input("ATR SL Multiplier", value=1.5, key="asm")

target_type = st.sidebar.selectbox("Target Type", ["Custom Points","Trailing Target (Points)","ATR-based","Signal-based (reverse EMA crossover)","50% Exit at Target (Partial)"])
target_points = 20.0
if target_type not in ["Signal-based (reverse EMA crossover)","Trailing Target (Points)"]: target_points = st.sidebar.number_input("Target Points", value=20.0, min_value=1.0, key="tp")
if target_type=="ATR-based": cfg['atr_target_multiplier'] = st.sidebar.number_input("ATR Target Multiplier", value=3.0, key="atm")

st.sidebar.markdown("---")
st.sidebar.subheader("Dhan Integration (Placeholder)")
use_dhan = st.sidebar.checkbox("Enable Dhan Brokerage")
if use_dhan:
    dhan_client_id, dhan_token = st.sidebar.text_input("Client ID"), st.sidebar.text_input("Token", type="password")
    dhan_strike, dhan_option = st.sidebar.number_input("Strike Price", value=0.0), st.sidebar.selectbox("Option Type", ["CE","PE"])
    dhan_expiry, dhan_lots = st.sidebar.date_input("Expiry Date"), st.sidebar.number_input("Lots", value=1, min_value=1)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Trading","📈 Trade History","📝 Trade Logs","🔬 Backtest"])

with tab1:
    c1,c2,c3 = st.columns([2,2,6])
    with c1:
        if st.button("▶️ Start Trading", type="primary", use_container_width=True):
            st.session_state['trading_active']=True; reset_pos(); add_log("Trading started"); st.rerun()
    with c2:
        if st.button("⏹️ Stop Trading", use_container_width=True):
            if st.session_state['trading_active']:
                if st.session_state['position']:
                    pos = st.session_state['position']
                    cp = st.session_state['current_data']['Close'].iloc[-1] if st.session_state['current_data'] is not None else pos['entry_price']
                    pnl = (cp-pos['entry_price'])*quantity if pos['signal']==1 else (pos['entry_price']-cp)*quantity
                    dur = (datetime.now(pytz.timezone('Asia/Kolkata'))-pos['entry_time']).total_seconds()/3600
                    st.session_state['trade_history'].append({'entry_time': pos['entry_time'], 'exit_time': datetime.now(pytz.timezone('Asia/Kolkata')),
                                                               'duration': dur, 'signal': 'LONG' if pos['signal']==1 else 'SHORT', 'entry_price': pos['entry_price'],
                                                               'exit_price': cp, 'sl': pos.get('sl'), 'target': pos.get('target'), 'exit_reason': 'Manual Close',
                                                               'pnl': pnl, 'highest': pos.get('highest',cp), 'lowest': pos.get('lowest',cp)})
                    add_log(f"Position manually closed - PnL: {pnl:.2f}")
                st.session_state['trading_active']=False; reset_pos(); add_log("Trading stopped"); st.rerun()
    with c3:
        st.success("🟢 Trading is ACTIVE") if st.session_state['trading_active'] else st.info("⚪ Trading is STOPPED")
    
    if st.button("🔄 Manual Refresh"): st.rerun()
    
    st.markdown("### Configuration")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Asset", asset_name if asset_name!="Custom" else ticker); st.metric("Interval", interval)
    with c2: st.metric("Period", period); st.metric("Quantity", quantity)
    with c3: st.metric("Strategy", strategy); st.metric("SL Type", sl_type)
    with c4:
        sl_d = f"{sl_points:.2f}" if sl_type!="Signal-based (reverse EMA crossover)" else "Signal Based"
        tgt_d = f"{target_points:.2f}" if target_type not in ["Signal-based (reverse EMA crossover)","Trailing Target (Points)"] else "Signal Based"
        st.metric("SL Points", sl_d); st.metric("Target Points", tgt_d)
    
    met_cont, chart_cont = st.container(), st.container()
    
    if st.session_state['trading_active']:
        if mode=="Live Trading": time.sleep(random.uniform(1.0,1.5))
        df = fetch_data(ticker, interval, period)
        if df is None or df.empty: st.error("Unable to fetch data"); st.session_state['trading_active']=False; st.rerun()
        st.session_state['current_data'] = df
        
        if strategy=="EMA Crossover": df = ema_cross_strat(df, cfg)
        elif strategy=="Percentage Change": df = pct_change_strat(df, cfg)
        else: df['Signal'] = 0
        
        df['EMA_Fast'], df['EMA_Slow'] = ema(df['Close'], cfg.get('ema_fast',9)), ema(df['Close'], cfg.get('ema_slow',15))
        df['EMA_Angle'], df['RSI'], df['ADX'], df['ATR'] = ema_angle(df['EMA_Fast'],2), rsi(df['Close'],14), adx(df,14), atr(df,14)
        
        cp, cs = df['Close'].iloc[-1], df['Signal'].iloc[-1]
        pos = st.session_state['position']
        
        if pos is None and cs!=0:
            pos = {'entry_time': df.index[-1], 'entry_price': cp, 'signal': cs, 'sl': None, 'target': None,
                   'highest': cp, 'lowest': cp, 'partial_exit_done': False, 'breakeven_activated': False}
            pos['sl'], pos['target'] = calc_sl(df,len(df)-1,pos,sl_type,sl_points,cfg), calc_tgt(df,len(df)-1,pos,target_type,target_points,cfg)
            st.session_state['position'], st.session_state['highest_price'], st.session_state['lowest_price'] = pos, cp, cp
            add_log(f"{'LONG' if cs==1 else 'SHORT'} entry at {cp:.2f}")
        
        elif pos:
            pos['highest'], pos['lowest'] = max(pos.get('highest',cp),cp), min(pos.get('lowest',cp),cp)
            st.session_state['highest_price'], st.session_state['lowest_price'] = pos['highest'], pos['lowest']
            exp, exr = None, None
            
            if sl_type=='Signal-based (reverse EMA crossover)' or target_type=='Signal-based (reverse EMA crossover)':
                if check_sig_exit(df,len(df)-1,pos): exp, exr = cp, "Reverse Signal"
            if exp is None and pos['sl']:
                if (pos['signal']==1 and cp<=pos['sl']) or (pos['signal']==-1 and cp>=pos['sl']): exp, exr = pos['sl'], "Stop Loss"
            if exp is None and pos['target']:
                if (pos['signal']==1 and cp>=pos['target']) or (pos['signal']==-1 and cp<=pos['target']):
                    if target_type=='50% Exit at Target (Partial)' and not pos['partial_exit_done']:
                        pos['partial_exit_done']=True; add_log("50% position exited")
                    else: exp, exr = pos['target'], "Target Hit"
            if exp is None and sl_type in ['Trailing SL (Points)','Break-even After 50% Target']: pos['sl']=calc_sl(df,len(df)-1,pos,sl_type,sl_points,cfg)
            
            if exp:
                pnl = (exp-pos['entry_price'])*quantity if pos['signal']==1 else (pos['entry_price']-exp)*quantity
                dur = (df.index[-1]-pos['entry_time']).total_seconds()/3600
                st.session_state['trade_history'].append({'entry_time': pos['entry_time'], 'exit_time': df.index[-1], 'duration': dur,
                                                           'signal': 'LONG' if pos['signal']==1 else 'SHORT', 'entry_price': pos['entry_price'],
                                                           'exit_price': exp, 'sl': pos['sl'], 'target': pos['target'], 'exit_reason': exr,
                                                           'pnl': pnl, 'highest': pos['highest'], 'lowest': pos['lowest']})
                add_log(f"Exit: {exr} at {exp:.2f}, PnL: {pnl:.2f}"); reset_pos()
            else: st.session_state['position'] = pos
        
        with met_cont:
            st.markdown("### Live Metrics")
            m1,m2,m3,m4,m5 = st.columns(5)
            with m1: st.metric("Current Price", f"{cp:.2f}"); st.metric("EMA Fast", f"{df['EMA_Fast'].iloc[-1]:.2f}")
            with m2:
                ep = pos['entry_price'] if pos else 0
                st.metric("Entry Price", f"{ep:.2f}" if pos else "—")
                st.metric("EMA Slow", f"{df['EMA_Slow'].iloc[-1]:.2f}")
            with m3:
                sig_txt = "LONG" if cs==1 else ("SHORT" if cs==-1 else "NONE")
                st.metric("Signal", sig_txt)
                st.metric("Crossover Angle", f"{abs(df['EMA_Angle'].iloc[-1]):.2f}°")
            with m4:
                if pos:
                    pnl = (cp-ep)*quantity if pos['signal']==1 else (ep-cp)*quantity
                    if pnl>=0: st.metric("Unrealized P&L", f"{pnl:.2f}", delta=f"+{pnl:.2f}")
                    else: st.metric("Unrealized P&L", f"{pnl:.2f}", delta=f"{pnl:.2f}", delta_color="inverse")
                else: st.metric("Unrealized P&L", "—")
                st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
            with m5:
                pos_stat = "IN POSITION" if pos else "NO POSITION"
                st.metric("Position Status", pos_stat)
                st.metric("ADX", f"{df['ADX'].iloc[-1]:.2f}")
            
            if pos:
                st.markdown("### Position Details")
                p1,p2,p3,p4 = st.columns(4)
                with p1:
                    dur = (datetime.now(pytz.timezone('Asia/Kolkata'))-pos['entry_time']).total_seconds()/60
                    st.metric("Duration", f"{dur:.1f} min")
                    sl_val = f"{pos['sl']:.2f}" if pos['sl'] else "—"
                    st.metric("Stop Loss", sl_val)
                with p2:
                    st.metric("Entry Price", f"{pos['entry_price']:.2f}")
                    tgt_val = f"{pos['target']:.2f}" if pos['target'] else "—"
                    st.metric("Target", tgt_val)
                with p3:
                    sl_dist = abs(cp-pos['sl']) if pos['sl'] else 0
                    st.metric("Distance to SL", f"{sl_dist:.2f}")
                    st.metric("Highest", f"{pos['highest']:.2f}")
                with p4:
                    tgt_dist = abs(pos['target']-cp) if pos['target'] else 0
                    st.metric("Distance to Target", f"{tgt_dist:.2f}")
                    st.metric("Lowest", f"{pos['lowest']:.2f}")
        
        with chart_cont:
            st.markdown("### Live Chart")
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price')])
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], name='EMA Fast', line=dict(color='blue', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], name='EMA Slow', line=dict(color='red', width=1)))
            if pos:
                fig.add_hline(y=pos['entry_price'], line_dash="dash", line_color="yellow", annotation_text="Entry")
                if pos['sl']: fig.add_hline(y=pos['sl'], line_dash="dash", line_color="red", annotation_text="SL")
                if pos['target']: fig.add_hline(y=pos['target'], line_dash="dash", line_color="green", annotation_text="Target")
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True, key=f"live_chart_{int(time.time())}")
        
        time.sleep(0.1); st.rerun()

with tab2:
    st.markdown("### 📈 Trade History")
    if len(st.session_state['trade_history'])==0: st.info("No trades yet")
    else:
        tt = len(st.session_state['trade_history']); win = sum(1 for t in st.session_state['trade_history'] if t['pnl']>0)
        los = tt-win; acc = win/tt*100 if tt>0 else 0; tpnl = sum(t['pnl'] for t in st.session_state['trade_history'])
        h1,h2,h3,h4,h5 = st.columns(5)
        with h1: st.metric("Total Trades", tt)
        with h2: st.metric("Winning", win)
        with h3: st.metric("Losing", los)
        with h4: st.metric("Accuracy", f"{acc:.1f}%")
        with h5:
            if tpnl>=0: st.metric("Total P&L", f"{tpnl:.2f}", delta=f"+{tpnl:.2f}")
            else: st.metric("Total P&L", f"{tpnl:.2f}", delta=f"{tpnl:.2f}", delta_color="inverse")
        
        st.markdown("---")
        for idx, t in enumerate(reversed(st.session_state['trade_history'])):
            with st.expander(f"Trade #{tt-idx} - {t.get('signal','N/A')} - P&L: {t.get('pnl',0):.2f}"):
                tc1,tc2 = st.columns(2)
                with tc1:
                    st.write(f"**Entry Time:** {t.get('entry_time','N/A')}")
                    st.write(f"**Entry Price:** {t.get('entry_price',0):.2f}")
                    st.write(f"**Stop Loss:** {t.get('sl',0):.2f}" if t.get('sl') else "**Stop Loss:** —")
                    st.write(f"**Highest:** {t.get('highest',0):.2f}")
                with tc2:
                    st.write(f"**Exit Time:** {t.get('exit_time','N/A')}")
                    st.write(f"**Exit Price:** {t.get('exit_price',0):.2f}")
                    st.write(f"**Target:** {t.get('target',0):.2f}" if t.get('target') else "**Target:** —")
                    st.write(f"**Lowest:** {t.get('lowest',0):.2f}")
                st.write(f"**Duration:** {t.get('duration',0):.2f} hours")
                st.write(f"**Exit Reason:** {t.get('exit_reason','N/A')}")
                pnl_val = t.get('pnl',0)
                pnl_str = f"{pnl_val:.2f}"
                if pnl_val>=0: st.success(f"**P&L:** +{pnl_str}")
                else: st.error(f"**P&L:** {pnl_str}")

with tab3:
    st.markdown("### 📝 Trade Logs")
    if len(st.session_state['trade_logs'])==0: st.info("No logs yet")
    else:
        for log in reversed(st.session_state['trade_logs']): st.text(log)

with tab4:
    st.markdown("### 🔬 Backtest Results")
    if mode!="Backtest": st.warning("Switch to Backtest mode to run backtests")
    else:
        if st.button("▶️ Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                df = fetch_data(ticker, interval, period)
                if df is None or df.empty: st.error("Unable to fetch data")
                else:
                    if strategy=="EMA Crossover": df = ema_cross_strat(df, cfg)
                    elif strategy=="Percentage Change": df = pct_change_strat(df, cfg)
                    else: df['Signal'] = 0
                    
                    results = run_backtest(df, None, cfg, sl_type, sl_points, target_type, target_points, quantity)
                    
                    st.success("Backtest completed!")
                    b1,b2,b3,b4,b5,b6 = st.columns(6)
                    with b1: st.metric("Total Trades", results['total_trades'])
                    with b2: st.metric("Winning", results['winning_trades'])
                    with b3: st.metric("Losing", results['losing_trades'])
                    with b4: st.metric("Accuracy", f"{results['accuracy']:.1f}%")
                    with b5:
                        tpnl = results['total_pnl']
                        if tpnl>=0: st.metric("Total P&L", f"{tpnl:.2f}", delta=f"+{tpnl:.2f}")
                        else: st.metric("Total P&L", f"{tpnl:.2f}", delta=f"{tpnl:.2f}", delta_color="inverse")
                    with b6: st.metric("Avg Duration", f"{results['avg_duration']:.2f}h")
                    
                    st.markdown("---")
                    st.markdown("### All Trades")
                    for idx, t in enumerate(results['trades']):
                        with st.expander(f"Trade #{idx+1} - {t['signal']} - P&L: {t['pnl']:.2f}"):
                            bc1,bc2 = st.columns(2)
                            with bc1:
                                st.write(f"**Entry Time:** {t['entry_time']}")
                                st.write(f"**Entry Price:** {t['entry_price']:.2f}")
                                st.write(f"**Stop Loss:** {t['sl']:.2f}" if t['sl'] else "**Stop Loss:** —")
                                st.write(f"**Highest:** {t['highest']:.2f}")
                            with bc2:
                                st.write(f"**Exit Time:** {t['exit_time']}")
                                st.write(f"**Exit Price:** {t['exit_price']:.2f}")
                                st.write(f"**Target:** {t['target']:.2f}" if t['target'] else "**Target:** —")
                                st.write(f"**Lowest:** {t['lowest']:.2f}")
                            st.write(f"**Duration:** {t['duration']:.2f} hours")
                            st.write(f"**Exit Reason:** {t['exit_reason']}")
                            st.write(f"**Range:** {t['highest']-t['lowest']:.2f}")
                            if t['pnl']>=0: st.success(f"**P&L:** +{t['pnl']:.2f}")
                            else: st.error(f"**P&L:** {t['pnl']:.2f}")
