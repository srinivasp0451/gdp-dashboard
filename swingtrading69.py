import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import pytz
import time
import random

st.set_page_config(page_title="Quantitative Trading System", layout="wide")

# Session state
for k in ['trading_active','current_data','position','trade_history','trade_logs','trailing_sl_high','trailing_sl_low','trailing_target_high','trailing_target_low','trailing_profit_points','threshold_crossed','highest_price','lowest_price','partial_exit_done','breakeven_activated']:
    if k not in st.session_state:
        st.session_state[k]=[]if'history'in k or'logs'in k else(False if'active'in k or'crossed'in k or'done'in k or'activated'in k else(0 if'points'in k else None))

def ist():return datetime.now(pytz.timezone('Asia/Kolkata'))
def fmt(d):return d.strftime("%Y-%m-%d %H:%M:%S IST")if d and not isinstance(d,str)else(d if d else"N/A")
def log(m):st.session_state['trade_logs'].append(f"[{fmt(ist())}] {m}");st.session_state['trade_logs']=st.session_state['trade_logs'][-50:]
def rst():
    for k in['position','trailing_sl_high','trailing_sl_low','trailing_target_high','trailing_target_low','threshold_crossed','highest_price','lowest_price','partial_exit_done','breakeven_activated']:
        st.session_state[k]=0 if'points'in k else(False if any(x in k for x in['crossed','done','activated'])else None)

def ema(d,p):return d.ewm(span=p,adjust=False).mean()
def rsi(d,p=14):g=(d.diff().where(d.diff()>0,0)).rolling(p).mean();l=(-d.diff().where(d.diff()<0,0)).rolling(p).mean();return 100-(100/(1+g/l))
def atr(h,l,c,p=14):return pd.concat([h-l,abs(h-c.shift(1)),abs(l-c.shift(1))],axis=1).max(axis=1).rolling(p).mean()
def adx(h,l,c,p=14):pd1,md=h.diff(),-l.diff();pd1[pd1<0],md[md<0]=0,0;tr=pd.concat([h-l,abs(h-c.shift(1)),abs(l-c.shift(1))],axis=1).max(axis=1);a=tr.rolling(p).mean();pi,mi=100*pd1.rolling(p).mean()/a,100*md.rolling(p).mean()/a;return(100*abs(pi-mi)/(pi+mi)).rolling(p).mean()
def ang(e,p=2):return np.degrees(np.arctan(e.diff(p)))if len(e)>=p+1 else pd.Series([0]*len(e),index=e.index)
def swg(h,l,p=5):sh,sl=pd.Series(index=h.index,dtype=float),pd.Series(index=l.index,dtype=float);[sh.__setitem__(i,h.iloc[i])if h.iloc[i]==h.iloc[i-p:i+p+1].max()else None for i in range(p,len(h)-p)];[sl.__setitem__(i,l.iloc[i])if l.iloc[i]==l.iloc[i-p:i+p+1].min()else None for i in range(p,len(l)-p)];return sh,sl

def fch(t,i,p,m):
    try:
        m=="Live Trading"and time.sleep(random.uniform(1.0,1.5));d=yf.download(t,interval=i,period=p,progress=False)
        if d.empty:return None
        isinstance(d.columns,pd.MultiIndex)and setattr(d,'columns',d.columns.droplevel(1));d=d[[c for c in['Open','High','Low','Close','Volume']if c in d.columns]]
        d.index=d.index.tz_localize('UTC').tz_convert('Asia/Kolkata')if d.index.tz is None else d.index.tz_convert('Asia/Kolkata');return d
    except:return None

def csl(e,s,t,c,d,i,p=None):
    sp,md=c.get('sl_points',10),c.get('min_sl_distance',10)
    if t=="Custom Points":sl=e-sp if s==1 else e+sp
    elif"Trailing"in t and"Points"in t:sl=(p-sp if s==1 else p+sp)if p else(e-sp if s==1 else e+sp)
    elif"ATR"in t or"Volatility"in t:a=d['ATR'].iloc[i]*c.get('sl_atr',1.5);sl=e-a if s==1 else e+a
    elif"Swing"in t:sh,sl1=swg(d['High'],d['Low']);r=sl1[:i+1 if"Current"in t else i].dropna()if s==1 else sh[:i+1 if"Current"in t else i].dropna();sl=r.iloc[-1]if len(r)>0 else(e-sp if s==1 else e+sp)
    elif"Candle"in t:j=i if"Current"in t else(i-1 if i>0 else i);sl=d['Low'].iloc[j]if s==1 else d['High'].iloc[j]
    else:sl=e-sp if s==1 else e+sp
    return min(sl,e-md)if s==1 else max(sl,e+md)

def ctg(e,s,t,c,d,i,sp):
    tp,md=c.get('target_points',20),c.get('min_target_distance',15)
    if"Custom"in t or"Trailing"in t:tg=e+tp if s==1 else e-tp
    elif"ATR"in t:a=d['ATR'].iloc[i]*c.get('tgt_atr',3.0);tg=e+a if s==1 else e-a
    elif"Risk-Reward"in t:tg=e+(abs(e-sp)*c.get('rr_ratio',2.0))if s==1 else e-(abs(e-sp)*c.get('rr_ratio',2.0))
    elif"Swing"in t:sh,sl=swg(d['High'],d['Low']);r=sh[:i+1 if"Current"in t else i].dropna()if s==1 else sl[:i+1 if"Current"in t else i].dropna();tg=r.iloc[-1]if len(r)>0 else(e+tp if s==1 else e-tp)
    elif"Candle"in t:j=i if"Current"in t else(i-1 if i>0 else i);tg=d['High'].iloc[j]if s==1 else d['Low'].iloc[j]
    else:tg=e+tp if s==1 else e-tp
    return max(tg,e+md)if s==1 else min(tg,e-md)

def utsl(p,cs,s,t,c,d,i):
    if"Trailing"not in t:return cs
    sp,th=c.get('sl_points',10),c.get('trailing_threshold',0)
    if"Points"in t:n=p-sp if s==1 else p+sp;return n if(s==1 and n>cs+th)or(s==-1 and n<cs-th)else cs
    elif"Candle"in t:n=d['Low'].iloc[i]if s==1 else d['High'].iloc[i];return n if(s==1 and n>cs)or(s==-1 and n<cs)else cs
    return cs

def bkt(t,i,p,st,c,q):
    d=fch(t,i,p,"Backtest")
    if d is None or len(d)<50:return None,"Insufficient data"
    d['EF'],d['ES'],d['A'],d['R'],d['X'],d['T']=ema(d['Close'],c.get('ema_fast',9)),ema(d['Close'],c.get('ema_slow',15)),ang(ema(d['Close'],c.get('ema_fast',9))),rsi(d['Close']),adx(d['High'],d['Low'],d['Close']),atr(d['High'],d['Low'],d['Close'])
    tr,po=[],None
    for j in range(50,len(d)):
        cp,ef,es,an,ax=d['Close'].iloc[j],d['EF'].iloc[j],d['ES'].iloc[j],abs(d['A'].iloc[j]),d['X'].iloc[j]
        if po is None:
            sg=0
            if st=="EMA Crossover"and ef>es and d['EF'].iloc[j-1]<=d['ES'].iloc[j-1]and an>=c.get('min_angle',1.0)and(not c.get('use_adx')or ax>=c.get('adx_threshold',25)):sg=1
            elif st=="EMA Crossover"and ef<es and d['EF'].iloc[j-1]>=d['ES'].iloc[j-1]and an>=c.get('min_angle',1.0)and(not c.get('use_adx')or ax>=c.get('adx_threshold',25)):sg=-1
            elif st=="Simple Buy":sg=1
            elif st=="Simple Sell":sg=-1
            elif st=="Price Threshold":th,dr=c['threshold'],c['direction'];sg=1 if("LONG (>="in dr and cp>=th)or("LONG (<="in dr and cp<=th)else(-1 if("SHORT (>="in dr and cp>=th)or("SHORT (<="in dr and cp<=th)else 0)
            elif st=="Percentage Change":pc=((cp-d['Close'].iloc[0])/d['Close'].iloc[0])*100;pt,pd=c['pct_threshold'],c['pct_direction'];sg=1 if(("BUY on Fall"in pd and pc<=-pt)or("BUY on Rise"in pd and pc>=pt))else(-1 if(("SELL on Fall"in pd and pc<=-pt)or("SELL on Rise"in pd and pc>=pt))else 0)
            if sg!=0:sl=csl(cp,sg,c['sl_type'],c,d,j);tg=ctg(cp,sg,c['target_type'],c,d,j,sl);po={'et':d.index[j],'ep':cp,'sg':sg,'sl':sl,'tg':tg,'hi':cp,'lo':cp}
        else:
            sg,ep,sl,tg=po['sg'],po['ep'],po['sl'],po['tg'];po['hi'],po['lo']=max(po['hi'],cp),min(po['lo'],cp);sl=utsl(cp,sl,sg,c['sl_type'],c,d,j);po['sl']=sl
            xt,xr,xp=False,"",cp
            if(sg==1 and cp<=sl)or(sg==-1 and cp>=sl):xt,xr,xp=True,"SL Hit",sl
            elif(sg==1 and cp>=tg)or(sg==-1 and cp<=tg):xt,xr,xp=True,"Target Hit",tg
            elif"Signal-based"in c['sl_type']or"Signal-based"in c['target_type']:
                if(sg==1 and ef<es and d['EF'].iloc[j-1]>=d['ES'].iloc[j-1])or(sg==-1 and ef>es and d['EF'].iloc[j-1]<=d['ES'].iloc[j-1]):xt,xr,xp=True,"Reverse Signal",cp
            if xt:pnl=(xp-ep)*q if sg==1 else(ep-xp)*q;tr.append({'entry_time':po['et'],'exit_time':d.index[j],'duration':(d.index[j]-po['et']).total_seconds()/3600,'signal':'LONG'if sg==1 else'SHORT','entry_price':ep,'exit_price':xp,'sl':po['sl'],'target':tg,'exit_reason':xr,'pnl':pnl,'highest':po['hi'],'lowest':po['lo'],'range':po['hi']-po['lo']});po=None
    return tr,None

# Sidebar
st.sidebar.header("⚙️ Configuration")
am={"NIFTY 50":"^NSEI","BANKNIFTY":"^NSEBANK","SENSEX":"^BSESN","BTC":"BTC-USD","ETH":"ETH-USD","USDINR":"USDINR=X","EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","GOLD":"GC=F","SILVER":"SI=F"}
a=st.sidebar.selectbox("Asset",list(am.keys())+["Custom"])
tk=st.sidebar.text_input("Ticker",am.get(a,"^NSEI"))if a=="Custom"else am[a]
iv=st.sidebar.selectbox("Interval",["1m","5m","15m","30m","1h","4h","1d","1wk","1mo"])
pm={"1m":["1d","5d"],"5m":["1d","1mo"],"15m":["1mo"],"30m":["1mo"],"1h":["1mo"],"4h":["1mo"],"1d":["1mo","1y","2y","5y"],"1wk":["1mo","1y","5y","10y","15y","20y"],"1mo":["1y","2y","5y","10y","15y","20y","25y","30y"]}
pr=st.sidebar.selectbox("Period",pm.get(iv,["1mo"]))
md=st.sidebar.selectbox("Mode",["Live Trading","Backtest"])
qty=st.sidebar.number_input("Quantity",min_value=1,value=1)
strat=st.sidebar.selectbox("Strategy",["EMA Crossover","Simple Buy","Simple Sell","Price Threshold","Percentage Change"])

cfg={}
if strat=="EMA Crossover":
    st.sidebar.subheader("EMA Settings");cfg['ema_fast'],cfg['ema_slow'],cfg['min_angle']=st.sidebar.number_input("Fast",value=9,key="ef"),st.sidebar.number_input("Slow",value=15,key="es"),st.sidebar.number_input("Min Angle",value=1.0,key="ma")
    cfg['use_adx']=st.sidebar.checkbox("Use ADX Filter")
    cfg['use_adx']and setattr(cfg,'adx_threshold',st.sidebar.number_input("ADX Threshold",value=25.0,key="at"))or None
elif strat=="Price Threshold":cfg['threshold'],cfg['direction']=st.sidebar.number_input("Threshold",value=100.0,key="th"),st.sidebar.selectbox("Direction",["LONG (>=)","SHORT (>=)","LONG (<=)","SHORT (<=)"])
elif strat=="Percentage Change":cfg['pct_threshold'],cfg['pct_direction']=st.sidebar.number_input("% Threshold",value=0.01,step=0.01,key="pt"),st.sidebar.selectbox("Direction",["BUY on Fall","SELL on Fall","BUY on Rise","SELL on Rise"])

st.sidebar.subheader("Stop Loss")
slt=st.sidebar.selectbox("SL Type",["Custom Points","Trailing SL (Points)","Trailing SL + Current Candle","Trailing SL + Previous Candle","Trailing SL + Current Swing","Trailing SL + Previous Swing","Trailing SL + Signal Based","Volatility-Adjusted Trailing SL","Break-even After 50% Target","ATR-based","Current Candle Low/High","Previous Candle Low/High","Current Swing Low/High","Previous Swing Low/High","Signal-based (reverse EMA crossover)"])
cfg['sl_type']=slt
("Points"in slt or"Custom"in slt)and setattr(cfg,'sl_points',st.sidebar.number_input("SL Points",value=10.0,key="slp"))or None
("ATR"in slt or"Volatility"in slt)and setattr(cfg,'sl_atr',st.sidebar.number_input("ATR Mult (SL)",value=1.5,key="sla"))or None
"Trailing"in slt and setattr(cfg,'trailing_threshold',st.sidebar.number_input("Trailing Threshold",value=0.0,key="tth"))or None
cfg['min_sl_distance']=st.sidebar.number_input("Min SL Distance",value=10.0,key="msd")

st.sidebar.subheader("Target")
tgt=st.sidebar.selectbox("Target Type",["Custom Points","Trailing Target (Points)","Trailing Target + Signal Based","50% Exit at Target (Partial)","Current Candle Low/High","Previous Candle Low/High","Current Swing Low/High","Previous Swing Low/High","ATR-based","Risk-Reward Based","Signal-based (reverse EMA crossover)"])
cfg['target_type']=tgt
("Points"in tgt or"Custom"in tgt)and setattr(cfg,'target_points',st.sidebar.number_input("Target Points",value=20.0,key="tp"))or None
"ATR"in tgt and setattr(cfg,'tgt_atr',st.sidebar.number_input("ATR Mult (Tgt)",value=3.0,key="ta"))or None
"Risk-Reward"in tgt and setattr(cfg,'rr_ratio',st.sidebar.number_input("R:R Ratio",value=2.0,key="rr"))or None
cfg['min_target_distance']=st.sidebar.number_input("Min Target Distance",value=15.0,key="mtd")

st.sidebar.subheader("Dhan (Placeholder)")
ud=st.sidebar.checkbox("Enable Dhan")
ud and(st.sidebar.text_input("Client ID",key="dci"),st.sidebar.text_input("Token",type="password",key="dtk"),st.sidebar.info("Placeholder only"))or None

st.title("🎯 Professional Quantitative Trading System")

t1,t2,t3,t4=st.tabs(["📊 Live Dashboard","📈 Trade History","📝 Logs","🔬 Backtest"])

with t1:
    st.header("Live Trading Dashboard")
    c1,c2,c3=st.columns([1,1,2])
    with c1:
        st.button("▶️ Start",type="primary",use_container_width=True)and(setattr(st.session_state,'trading_active',True),log("Trading started"),st.rerun())or None
    with c2:
        if st.button("⏹️ Stop",type="secondary",use_container_width=True):
            st.session_state['trading_active']=False
            if st.session_state['position']:
                po=st.session_state['position'];xp=st.session_state['current_data']['Close'].iloc[-1]if st.session_state['current_data']is not None else po['entry_price'];pnl=(xp-po['entry_price'])*qty if po['signal']==1 else(po['entry_price']-xp)*qty
                st.session_state['trade_history'].append({'entry_time':po['entry_time'],'exit_time':ist(),'signal':'LONG'if po['signal']==1 else'SHORT','entry_price':po['entry_price'],'exit_price':xp,'sl':po.get('sl',0),'target':po.get('target',0),'exit_reason':'Manual Close','pnl':pnl,'highest':po.get('highest',xp),'lowest':po.get('lowest',xp),'range':po.get('highest',xp)-po.get('lowest',xp),'duration':(ist()-po['entry_time']).total_seconds()/3600})
                log(f"Manual close. PnL: {pnl:.2f}")
            rst();log("Trading stopped");st.rerun()
    with c3:st.success("🟢 ACTIVE")if st.session_state['trading_active']else st.info("⚪ STOPPED")
    
    st.button("🔄 Refresh")and st.rerun()or None
    
    c1,c2,c3=st.columns(3)
    c1.write(f"**Asset:** {a} ({tk})\n**Interval:** {iv} | **Period:** {pr}")
    c2.write(f"**Qty:** {qty} | **Strategy:** {strat}\n**Mode:** {md}")
    c3.write(f"**SL:** {slt}\n**Target:** {tgt}")
    
    if st.session_state['trading_active']and md=="Live Trading":
        ph=st.empty()
        while st.session_state['trading_active']:
            with ph.container():
                df=fch(tk,iv,pr,md)
                if df is None or len(df)==0:st.error("No data");time.sleep(2);continue
                st.session_state['current_data']=df;df['EF'],df['ES'],df['A'],df['R'],df['X'],df['T']=ema(df['Close'],cfg.get('ema_fast',9)),ema(df['Close'],cfg.get('ema_slow',15)),ang(ema(df['Close'],cfg.get('ema_fast',9))),rsi(df['Close']),adx(df['High'],df['Low'],df['Close']),atr(df['High'],df['Low'],df['Close'])
                cp,ef,es,an,rs,ax=df['Close'].iloc[-1],df['EF'].iloc[-1],df['ES'].iloc[-1],abs(df['A'].iloc[-1]),df['R'].iloc[-1],df['X'].iloc[-1]
                
                m1,m2,m3,m4,m5=st.columns(5)
                m1.metric("Price",f"{cp:.2f}");m2.metric("EMA Fast",f"{ef:.2f}");m3.metric("EMA Slow",f"{es:.2f}");m4.metric("RSI",f"{rs:.2f}");m5.metric("ADX",f"{ax:.2f}")
                m6,m7,m8=st.columns(3)
                m6.metric("Angle",f"{an:.2f}°");m7.metric("Updated",fmt(ist()))
                
                po=st.session_state['position']
                if po is None:
                    sg=0
                    if strat=="EMA Crossover"and ef>es and df['EF'].iloc[-2]<=df['ES'].iloc[-2]and an>=cfg.get('min_angle',1.0)and(not cfg.get('use_adx')or ax>=cfg.get('adx_threshold',25)):sg=1
                    elif strat=="EMA Crossover"and ef<es and df['EF'].iloc[-2]>=df['ES'].iloc[-2]and an>=cfg.get('min_angle',1.0)and(not cfg.get('use_adx')or ax>=cfg.get('adx_threshold',25)):sg=-1
                    elif strat=="Simple Buy":sg=1
                    elif strat=="Simple Sell":sg=-1
                    elif strat=="Price Threshold":th,dr=cfg['threshold'],cfg['direction'];sg=1 if("LONG (>="in dr and cp>=th)or("LONG (<="in dr and cp<=th)else(-1 if("SHORT (>="in dr and cp>=th)or("SHORT (<="in dr and cp<=th)else 0)
                    elif strat=="Percentage Change":pc=((cp-df['Close'].iloc[0])/df['Close'].iloc[0])*100;pt,pd=cfg['pct_threshold'],cfg['pct_direction'];sg=1 if(("BUY on Fall"in pd and pc<=-pt)or("BUY on Rise"in pd and pc>=pt))else(-1 if(("SELL on Fall"in pd and pc<=-pt)or("SELL on Rise"in pd and pc>=pt))else 0)
                    
                    if sg!=0:sl=csl(cp,sg,cfg['sl_type'],cfg,df,len(df)-1);tg=ctg(cp,sg,cfg['target_type'],cfg,df,len(df)-1,sl);st.session_state['position']={'entry_time':ist(),'entry_price':cp,'signal':sg,'sl':sl,'target':tg,'highest':cp,'lowest':cp};log(f"{'LONG'if sg==1 else'SHORT'} @ {cp:.2f}, SL: {sl:.2f}, Tgt: {tg:.2f}");m8.success("✅ Entered")
                else:
                    sg,ep,sl,tg=po['signal'],po['entry_price'],po['sl'],po['target'];po['highest'],po['lowest']=max(po['highest'],cp),min(po['lowest'],cp);pnl=(cp-ep)*qty if sg==1 else(ep-cp)*qty
                    
                    p1,p2,p3,p4=st.columns(4)
                    p1.metric("Entry",f"{ep:.2f}");p1.metric("Type","LONG"if sg==1 else"SHORT");p2.metric("SL",f"{sl:.2f}");p2.metric("Dist SL",f"{abs(cp-sl):.2f}");p3.metric("Target",f"{tg:.2f}");p3.metric("Dist Tgt",f"{abs(tg-cp):.2f}");p4.metric("P&L",f"{pnl:.2f}",delta=f"{'+'if pnl>=0 else''}{pnl:.2f}",delta_color="normal"if pnl>=0 else"inverse")
                    
                    sl=utsl(cp,sl,sg,cfg['sl_type'],cfg,df,len(df)-1);po['sl']=sl
                    xt,xr=False,""
                    if(sg==1 and cp<=sl)or(sg==-1 and cp>=sl):xt,xr=True,"SL Hit"
                    elif(sg==1 and cp>=tg)or(sg==-1 and cp<=tg):xt,xr=True,"Target Hit"
                    elif"Signal-based"in cfg['sl_type']or"Signal-based"in cfg['target_type']:
                        if(sg==1 and ef<es and df['EF'].iloc[-2]>=df['ES'].iloc[-2])or(sg==-1 and ef>es and df['EF'].iloc[-2]<=df['ES'].iloc[-2]):xt,xr=True,"Reverse Signal"
                    
                    if xt:st.session_state['trade_history'].append({'entry_time':po['entry_time'],'exit_time':ist(),'signal':'LONG'if sg==1 else'SHORT','entry_price':ep,'exit_price':cp,'sl':sl,'target':tg,'exit_reason':xr,'pnl':pnl,'highest':po['highest'],'lowest':po['lowest'],'range':po['highest']-po['lowest'],'duration':(ist()-po['entry_time']).total_seconds()/3600});log(f"Exit: {xr}. PnL: {pnl:.2f}");rst();st.success(f"✅ {xr}")
                
                fig=go.Figure()
                fig.add_trace(go.Candlestick(x=df.index,open=df['Open'],high=df['High'],low=df['Low'],close=df['Close'],name='Price'))
                fig.add_trace(go.Scatter(x=df.index,y=df['EF'],name='Fast',line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=df.index,y=df['ES'],name='Slow',line=dict(color='red')))
                po and(fig.add_hline(y=po['entry_price'],line_dash="dash",line_color="yellow",annotation_text="Entry"),fig.add_hline(y=po['sl'],line_dash="dash",line_color="red",annotation_text="SL"),fig.add_hline(y=po['target'],line_dash="dash",line_color="green",annotation_text="Target"))or None
                fig.update_layout(xaxis_rangeslider_visible=False,height=500)
                st.plotly_chart(fig,use_container_width=True,key=f"ch_{int(time.time())}")
            
            if not st.session_state['trading_active']:break
            time.sleep(random.uniform(1.0,1.5))

with t2:
    st.markdown("### 📈 Trade History")
    if len(st.session_state['trade_history'])==0:st.info("No trades yet")
    else:
        tot=len(st.session_state['trade_history']);wins=sum(1 for t in st.session_state['trade_history']if t['pnl']>0);loss=tot-wins;acc=(wins/tot*100)if tot>0 else 0;tpnl=sum(t['pnl']for t in st.session_state['trade_history'])
        m1,m2,m3,m4,m5=st.columns(5);m1.metric("Total",tot);m2.metric("Wins",wins);m3.metric("Losses",loss);m4.metric("Accuracy",f"{acc:.1f}%");m5.metric("Total P&L",f"{tpnl:.2f}",delta=f"{'+'if tpnl>=0 else''}{tpnl:.2f}",delta_color="normal"if tpnl>=0 else"inverse")
        for i,t in enumerate(reversed(st.session_state['trade_history'])):
            with st.expander(f"Trade #{tot-i}: {t['signal']} | P&L: {t['pnl']:.2f}"):
                c1,c2=st.columns(2)
                c1.write(f"**Entry:** {fmt(t['entry_time'])}\n**Entry Price:** {t['entry_price']:.2f}\n**SL:** {t['sl']:.2f}\n**Highest:** {t['highest']:.2f}")
                c2.write(f"**Exit:** {fmt(t['exit_time'])}\n**Exit Price:** {t['exit_price']:.2f}\n**Target:** {t['target']:.2f}\n**Lowest:** {t['lowest']:.2f}")
                st.write(f"**Exit Reason:** {t['exit_reason']}")
                st.markdown(f"**P&L:** <span style='color:{'green'if t['pnl']>=0 else'red'
