# =============================================================================
# SMART INVESTING PLATFORM  v2.0
# Professional Algorithmic Trading | NSE • BSE • Crypto • Forex • Commodities
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time, threading, datetime, traceback, warnings
from typing import Optional, Dict, List, Tuple, Any

warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Investing Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container{padding-top:1rem}
  .metric-box{background:#12192c;border:1px solid #1e3a5f;border-radius:10px;
              padding:14px;text-align:center;margin-bottom:6px}
  .metric-label{color:#6b7280;font-size:11px;margin-bottom:4px}
  .metric-val{font-size:22px;font-weight:700}
  .green{color:#00e676} .red{color:#ff5252} .yellow{color:#ffca28} .white{color:#fff}
  div[data-testid="stTabs"] button{font-size:15px;font-weight:600}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# THREAD-SAFE PERSISTENT STORE
# ─────────────────────────────────────────────────────────────────────────────
class _Store:
    _inst = None
    _cls_lock = threading.Lock()

    def __new__(cls):
        if cls._inst is None:
            with cls._cls_lock:
                if cls._inst is None:
                    o = super().__new__(cls)
                    o._lock = threading.Lock()
                    o._d = dict(
                        live_running=False, live_thread=None,
                        live_position=None, live_ltp=None,
                        live_pnl=0.0, live_signal="HOLD",
                        live_log=[], trade_history=[],
                        backtest_trades=[], backtest_result=None,
                        live_df=None, error=None,
                        trailing_sl=None, trailing_target=None,
                        last_fetch=0.0,
                    )
                    cls._inst = o
        return cls._inst

    def get(self, k, default=None):
        with self._lock: return self._d.get(k, default)

    def set(self, k, v):
        with self._lock: self._d[k] = v

    def upd(self, d):
        with self._lock: self._d.update(d)

    def clear_position(self):
        with self._lock:
            self._d.update(dict(live_position=None, live_pnl=0.0,
                                trailing_sl=None, trailing_target=None))

STORE = _Store()

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = {
    "NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN",
    "NIFTY IT": "^CNXIT", "BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD",
    "USD/INR": "USDINR=X", "GOLD": "GC=F", "SILVER": "SI=F",
    "CRUDE OIL": "CL=F", "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X",
    "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "INFY": "INFY.NS",
    "HDFC BANK": "HDFCBANK.NS", "Custom ✏️": "__custom__",
}

INTERVALS = ["1m","5m","15m","30m","1h","4h","1d","1wk"]

# yfinance max periods per interval
IV_MAX = {"1m":"5d","2m":"5d","5m":"60d","15m":"60d","30m":"60d",
          "60m":"730d","1h":"730d","4h":"730d","1d":"max","1wk":"max"}

PERIODS = ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","max"]

STRATEGIES = [
    "🏆 Smart Algo (Recommended)",
    "SuperTrend + EMA Cloud",
    "EMA Crossover + ADX",
    "MACD Momentum",
    "Bollinger Squeeze",
    "Heikin Ashi Reversal",
    "Smart Money Concepts (SMC)",
    "Opening Range Breakout",
    "Donchian Breakout",
]

SL_TYPES = [
    "ATR Based",
    "Custom Points",
    "Trailing – Fixed Points",
    "Trailing – Candle Low/High",
    "Trailing – Pivot Low/High",
    "Risk Reward Based",
    "Volatility Based (BB)",
    "Signal Reversal Exit",
]

TGT_TYPES = [
    "ATR Based",
    "Custom Points",
    "Risk Reward Ratio",
    "Volatility Based (BB)",
    "Fibonacci Extension",
    "Signal Reversal Exit",
    "Trailing Target (Display Only)",
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING  (1.5 s rate-limit guard)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_data(ticker: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    elapsed = time.time() - STORE.get("last_fetch", 0.0)
    if elapsed < 1.5:
        time.sleep(1.5 - elapsed)
    STORE.set("last_fetch", time.time())

    fetch_iv = "1h" if interval == "4h" else interval
    # Clamp period to what yfinance supports for that interval
    allowed = IV_MAX.get(fetch_iv, "max")
    if allowed != "max":
        period_days = {"1d":1,"5d":5,"1mo":30,"3mo":90,"6mo":180,
                       "1y":365,"2y":730,"5y":1825,"10y":3650,"max":99999,
                       "60d":60,"730d":730}
        if period_days.get(period, 9999) > period_days.get(allowed, 9999):
            period = allowed

    try:
        raw = yf.download(ticker, interval=fetch_iv, period=period,
                          progress=False, auto_adjust=True, timeout=15)
    except Exception as e:
        STORE.set("error", f"yfinance error: {e}")
        return None

    if raw is None or raw.empty:
        return None

    # Flatten MultiIndex if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    df = raw.copy()

    # Resample 1h → 4h
    if interval == "4h":
        df = df.resample("4h", label="right", closed="right").agg(
            {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
        ).dropna(subset=["Close"])

    df = df[~df.index.duplicated(keep="last")].dropna(subset=["Close"])
    return df


def has_volume(df: pd.DataFrame) -> bool:
    return "Volume" in df.columns and df["Volume"].fillna(0).sum() > 0

# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────
def ema(s, n): return s.ewm(span=n, adjust=False, min_periods=1).mean()
def sma(s, n): return s.rolling(n, min_periods=1).mean()

def calc_rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n, min_periods=1).mean()
    l = (-d.clip(upper=0)).rolling(n, min_periods=1).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def calc_macd(s, fast=12, slow=26, sig=9):
    m = ema(s, fast) - ema(s, slow)
    sl = ema(m, sig)
    return m, sl, m - sl

def calc_atr(df, n=14):
    h, l, pc = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False, min_periods=1).mean()

def calc_adx(df, n=14):
    h, l, pc = df["High"], df["Low"], df["Close"].shift(1)
    pdm = (h - h.shift(1)).clip(lower=0)
    ndm = (l.shift(1) - l).clip(lower=0)
    pdm = pdm.where(pdm > ndm, 0)
    ndm = ndm.where(ndm > pdm, 0)
    atr_ = calc_atr(df, n)
    pdi = 100 * ema(pdm, n) / atr_.replace(0, np.nan)
    ndi = 100 * ema(ndm, n) / atr_.replace(0, np.nan)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx_ = ema(dx, n)
    return adx_, pdi, ndi

def calc_bb(s, n=20, k=2.0):
    m = sma(s, n)
    std = s.rolling(n, min_periods=1).std()
    return m + k*std, m, m - k*std

def calc_supertrend(df, n=10, mult=3.0):
    atr_ = calc_atr(df, n).values
    hl2 = ((df["High"] + df["Low"]) / 2).values
    cl = df["Close"].values
    N = len(df)
    ub = hl2 + mult * atr_
    lb = hl2 - mult * atr_
    st = np.full(N, np.nan)
    dr = np.zeros(N, int)
    st[0] = ub[0]; dr[0] = -1
    for i in range(1, N):
        lb[i] = lb[i] if lb[i] > lb[i-1] or cl[i-1] < lb[i-1] else lb[i-1]
        ub[i] = ub[i] if ub[i] < ub[i-1] or cl[i-1] > ub[i-1] else ub[i-1]
        if np.isnan(st[i-1]) or st[i-1] == ub[i-1]:
            if cl[i] > ub[i]: dr[i]=1; st[i]=lb[i]
            else:              dr[i]=-1; st[i]=ub[i]
        else:
            if cl[i] < lb[i]: dr[i]=-1; st[i]=ub[i]
            else:              dr[i]=1;  st[i]=lb[i]
    idx = df.index
    return pd.Series(st,idx), pd.Series(dr,idx), pd.Series(ub,idx), pd.Series(lb,idx)

def calc_ha(df):
    ha = pd.DataFrame(index=df.index)
    ha["Close"] = (df["Open"]+df["High"]+df["Low"]+df["Close"])/4
    ha["Open"]  = (df["Open"].shift(1)+df["Close"].shift(1))/2
    ha.loc[ha.index[0],"Open"] = df["Open"].iloc[0]
    ha["High"]  = pd.concat([df["High"],ha["Open"],ha["Close"]],axis=1).max(axis=1)
    ha["Low"]   = pd.concat([df["Low"], ha["Open"],ha["Close"]],axis=1).min(axis=1)
    return ha

def detect_order_blocks(df, lb=5):
    ob = pd.Series(0, index=df.index)
    for i in range(lb, len(df)):
        if df["Open"].iloc[i-lb]>df["Close"].iloc[i-lb] and df["Close"].iloc[i]>df["High"].iloc[i-lb]:
            ob.iloc[i]=1
        elif df["Close"].iloc[i-lb]>df["Open"].iloc[i-lb] and df["Close"].iloc[i]<df["Low"].iloc[i-lb]:
            ob.iloc[i]=-1
    return ob

def detect_liq_hunt(df, lb=10):
    lh = pd.Series(0, index=df.index)
    for i in range(lb+1, len(df)):
        rh = df["High"].iloc[i-lb:i].max()
        rl = df["Low"].iloc[i-lb:i].min()
        c,o = df["Close"].iloc[i], df["Open"].iloc[i]
        lo, hi = df["Low"].iloc[i], df["High"].iloc[i]
        if lo < rl and c > o and c > rl: lh.iloc[i]=1
        if hi > rh and c < o and c < rh: lh.iloc[i]=-1
    return lh

def fib_levels(df, lb=50):
    r = df.tail(lb)
    hi, lo = r["High"].max(), r["Low"].min()
    d = hi - lo
    return {"0":hi,"0.236":hi-0.236*d,"0.382":hi-0.382*d,"0.5":hi-0.5*d,
            "0.618":hi-0.618*d,"1":lo,"1.272":lo-0.272*d,"1.618":lo-0.618*d}

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    df["ema9"]  = ema(c,9);  df["ema21"] = ema(c,21)
    df["ema50"] = ema(c,50); df["ema200"]= ema(c,200)
    df["sma20"] = sma(c,20); df["sma50"] = sma(c,50)
    df["rsi"]   = calc_rsi(c)
    df["macd"], df["macd_sig"], df["macd_hist"] = calc_macd(c)
    df["atr"]   = calc_atr(df)
    df["adx"], df["pdi"], df["ndi"] = calc_adx(df)
    df["bb_u"], df["bb_m"], df["bb_l"] = calc_bb(c)
    df["bb_w"]  = (df["bb_u"]-df["bb_l"]) / df["bb_m"]
    df["bb_sq"] = df["bb_w"] < df["bb_w"].rolling(50,min_periods=10).quantile(0.2)
    df["st"], df["st_dir"], df["st_ub"], df["st_lb"] = calc_supertrend(df)
    ha = calc_ha(df)
    df["ha_c"]=ha["Close"]; df["ha_o"]=ha["Open"]
    df["ha_bull"] = df["ha_c"] > df["ha_o"]
    if has_volume(df):
        df["vol_ma"] = sma(df["Volume"],20)
        df["vol_r"]  = df["Volume"] / df["vol_ma"].replace(0,1)
    else:
        df["vol_ma"]=0; df["vol_r"]=1.0
    df["ob"]  = detect_order_blocks(df)
    df["lh"]  = detect_liq_hunt(df)
    df["ema_bull"] = (df["ema9"]>df["ema21"]) & (df["ema21"]>df["ema50"])
    df["ema_bear"] = (df["ema9"]<df["ema21"]) & (df["ema21"]<df["ema50"])
    df["mc_up"]  = (df["macd"]>df["macd_sig"]) & (df["macd"].shift(1)<=df["macd_sig"].shift(1))
    df["mc_dn"]  = (df["macd"]<df["macd_sig"]) & (df["macd"].shift(1)>=df["macd_sig"].shift(1))
    df["regime"] = np.where(df["adx"]>25,"TREND",np.where(df["adx"]<15,"CHOPPY","NORMAL"))
    return df

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY SIGNAL GENERATORS  → +1 BUY  /  -1 SELL  /  0 HOLD
# ─────────────────────────────────────────────────────────────────────────────
def _ind_ok(df, i, side, ena):
    """Return False to veto a signal via optional indicator filters."""
    if not ena: return True
    r = df.iloc[i]
    if side=="BUY":
        if ena.get("rsi")    and r["rsi"]>75:          return False
        if ena.get("macd")   and r["macd_hist"]<0:     return False
        if ena.get("adx")    and r["adx"]<18:          return False
        if ena.get("ema")    and not r["ema_bull"]:    return False
        if ena.get("volume") and has_volume(df) and r["vol_r"]<1.0: return False
        if ena.get("bb")     and r["Close"]>r["bb_u"]: return False
        if ena.get("ob")     and r["ob"]==-1:          return False
    else:
        if ena.get("rsi")    and r["rsi"]<25:          return False
        if ena.get("macd")   and r["macd_hist"]>0:    return False
        if ena.get("adx")    and r["adx"]<18:          return False
        if ena.get("ema")    and not r["ema_bear"]:    return False
        if ena.get("volume") and has_volume(df) and r["vol_r"]<1.0: return False
        if ena.get("bb")     and r["Close"]<r["bb_l"]: return False
        if ena.get("ob")     and r["ob"]==1:           return False
    return True

def sig_smart(df, use_ind=False, ena=None):
    S = pd.Series(0, index=df.index)
    for i in range(55, len(df)):
        r, p = df.iloc[i], df.iloc[i-1]
        vol_ok = (not has_volume(df)) or r["vol_r"] >= 1.1
        if r["adx"] >= 20:                          # ── TRENDING ──
            flip_up   = r["st_dir"]==1  and p["st_dir"]==-1
            flip_dn   = r["st_dir"]==-1 and p["st_dir"]==1
            rsi_b = 45 < r["rsi"] < 76
            rsi_s = 24 < r["rsi"] < 56
            b = (flip_up or (r["st_dir"]==1 and r["mc_up"])) and r["ema_bull"] and rsi_b
            s = (flip_dn or (r["st_dir"]==-1 and r["mc_dn"])) and r["ema_bear"] and rsi_s
        else:                                       # ── RANGING ──
            b = (r["Close"]<=r["bb_l"]*1.003) and r["rsi"]<38 and r["ha_bull"]
            s = (r["Close"]>=r["bb_u"]*0.997) and r["rsi"]>62 and not r["ha_bull"]
            # liquidity-hunt bonus
            if r["lh"]==1:  b = True
            if r["lh"]==-1: s = True
        if b and vol_ok and _ind_ok(df,i,"BUY",ena if use_ind else None):   S.iloc[i]=1
        if s and vol_ok and _ind_ok(df,i,"SELL",ena if use_ind else None):  S.iloc[i]=-1
    return S

def sig_supertrend(df, use_ind=False, ena=None):
    S = pd.Series(0, index=df.index)
    for i in range(55, len(df)):
        r, p = df.iloc[i], df.iloc[i-1]
        if r["st_dir"]==1 and p["st_dir"]==-1 and r["Close"]>r["ema50"]:
            if _ind_ok(df,i,"BUY",ena if use_ind else None): S.iloc[i]=1
        elif r["st_dir"]==-1 and p["st_dir"]==1 and r["Close"]<r["ema50"]:
            if _ind_ok(df,i,"SELL",ena if use_ind else None): S.iloc[i]=-1
    return S

def sig_ema_cross(df, use_ind=False, ena=None):
    S = pd.Series(0, index=df.index)
    for i in range(55, len(df)):
        r, p = df.iloc[i], df.iloc[i-1]
        cross_u = r["ema9"]>r["ema21"] and p["ema9"]<=p["ema21"]
        cross_d = r["ema9"]<r["ema21"] and p["ema9"]>=p["ema21"]
        if cross_u and r["adx"]>20 and r["pdi"]>r["ndi"] and _ind_ok(df,i,"BUY",ena if use_ind else None):  S.iloc[i]=1
        if cross_d and r["adx"]>20 and r["ndi"]>r["pdi"] and _ind_ok(df,i,"SELL",ena if use_ind else None): S.iloc[i]=-1
    return S

def sig_macd(df, use_ind=False, ena=None):
    S = pd.Series(0, index=df.index)
    for i in range(35, len(df)):
        r = df.iloc[i]
        if r["mc_up"] and r["ema9"]>r["ema21"] and _ind_ok(df,i,"BUY",ena if use_ind else None):  S.iloc[i]=1
        if r["mc_dn"] and r["ema9"]<r["ema21"] and _ind_ok(df,i,"SELL",ena if use_ind else None): S.iloc[i]=-1
    return S

def sig_bb_squeeze(df, use_ind=False, ena=None):
    S = pd.Series(0, index=df.index)
    for i in range(55, len(df)):
        r, p = df.iloc[i], df.iloc[i-1]
        sq_rel = p["bb_sq"] and not r["bb_sq"]
        if sq_rel:
            if r["Close"]>r["bb_m"] and r["macd_hist"]>0 and _ind_ok(df,i,"BUY",ena if use_ind else None):  S.iloc[i]=1
            if r["Close"]<r["bb_m"] and r["macd_hist"]<0 and _ind_ok(df,i,"SELL",ena if use_ind else None): S.iloc[i]=-1
        # Pure BB breakout
        bo_u = r["Close"]>r["bb_u"] and p["Close"]<=p["bb_u"] and r["adx"]>20
        bo_d = r["Close"]<r["bb_l"] and p["Close"]>=p["bb_l"] and r["adx"]>20
        if bo_u and _ind_ok(df,i,"BUY",ena if use_ind else None):  S.iloc[i]=1
        if bo_d and _ind_ok(df,i,"SELL",ena if use_ind else None): S.iloc[i]=-1
    return S

def sig_ha(df, use_ind=False, ena=None):
    S = pd.Series(0, index=df.index)
    for i in range(30, len(df)):
        r, p = df.iloc[i], df.iloc[i-1]
        if r["ha_bull"] and not p["ha_bull"] and r["Close"]>r["ema21"] and _ind_ok(df,i,"BUY",ena if use_ind else None):  S.iloc[i]=1
        if not r["ha_bull"] and p["ha_bull"] and r["Close"]<r["ema21"] and _ind_ok(df,i,"SELL",ena if use_ind else None): S.iloc[i]=-1
    return S

def sig_smc(df, use_ind=False, ena=None):
    S = pd.Series(0, index=df.index)
    for i in range(20, len(df)):
        r = df.iloc[i]
        if (r["lh"]==1 or r["ob"]==1) and r["Close"]>r["ema21"] and _ind_ok(df,i,"BUY",ena if use_ind else None):   S.iloc[i]=1
        if (r["lh"]==-1 or r["ob"]==-1) and r["Close"]<r["ema21"] and _ind_ok(df,i,"SELL",ena if use_ind else None): S.iloc[i]=-1
    return S

def sig_orb(df, use_ind=False, ena=None):
    S = pd.Series(0, index=df.index)
    if not hasattr(df.index,"date"): return S
    tmp = df.copy(); tmp["_d"]=tmp.index.date
    OR = 5
    for d, g in tmp.groupby("_d"):
        if len(g)<OR+2: continue
        orh = g["High"].iloc[:OR].max(); orl = g["Low"].iloc[:OR].min()
        for j in range(OR, len(g)):
            idx = g.index[j]; pr = g["Close"].iloc[j]; pp = g["Close"].iloc[j-1]
            if pr>orh and pp<=orh: S[idx]=1
            elif pr<orl and pp>=orl: S[idx]=-1
    return S

def sig_donchian(df, use_ind=False, ena=None):
    S = pd.Series(0, index=df.index)
    n=20
    dh = df["High"].rolling(n, min_periods=n).max()
    dl = df["Low"].rolling(n, min_periods=n).min()
    for i in range(n+1, len(df)):
        r, p = df.iloc[i], df.iloc[i-1]
        if r["Close"]>dh.iloc[i-1] and p["Close"]<=dh.iloc[i-1] and r["adx"]>20: S.iloc[i]=1
        if r["Close"]<dl.iloc[i-1] and p["Close"]>=dl.iloc[i-1] and r["adx"]>20: S.iloc[i]=-1
    return S

STRAT_FN = {
    "🏆 Smart Algo (Recommended)": sig_smart,
    "SuperTrend + EMA Cloud": sig_supertrend,
    "EMA Crossover + ADX": sig_ema_cross,
    "MACD Momentum": sig_macd,
    "Bollinger Squeeze": sig_bb_squeeze,
    "Heikin Ashi Reversal": sig_ha,
    "Smart Money Concepts (SMC)": sig_smc,
    "Opening Range Breakout": sig_orb,
    "Donchian Breakout": sig_donchian,
}

def get_signals(df, strategy, use_ind, ena):
    fn = STRAT_FN.get(strategy, sig_smart)
    return fn(df, use_ind, ena)

# ─────────────────────────────────────────────────────────────────────────────
# SL / TARGET HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def init_sl(df, i, side, sl_type, sl_val, entry):
    r = df.iloc[i]; atr = r["atr"]
    if side=="LONG":
        if sl_type=="ATR Based":           return entry - sl_val*atr
        if sl_type=="Custom Points":       return entry - sl_val
        if "Trailing" in sl_type:          return entry - sl_val*atr   # initial
        if sl_type=="Risk Reward Based":   return entry - sl_val
        if sl_type=="Volatility Based (BB)": return r["bb_l"]
        if sl_type=="Signal Reversal Exit":  return entry - 2.5*atr
        return entry - sl_val*atr
    else:
        if sl_type=="ATR Based":           return entry + sl_val*atr
        if sl_type=="Custom Points":       return entry + sl_val
        if "Trailing" in sl_type:          return entry + sl_val*atr
        if sl_type=="Risk Reward Based":   return entry + sl_val
        if sl_type=="Volatility Based (BB)": return r["bb_u"]
        if sl_type=="Signal Reversal Exit":  return entry + 2.5*atr
        return entry + sl_val*atr

def init_tgt(df, i, side, tgt_type, tgt_val, entry, sl):
    r = df.iloc[i]; atr = r["atr"]; risk = abs(entry-sl)
    if side=="LONG":
        if tgt_type=="ATR Based":               return entry + tgt_val*atr
        if tgt_type=="Custom Points":           return entry + tgt_val
        if tgt_type=="Risk Reward Ratio":       return entry + tgt_val*risk
        if tgt_type=="Volatility Based (BB)":   return r["bb_u"]
        if tgt_type=="Fibonacci Extension":
            fi = fib_levels(df, 50); return fi.get("1.618", entry+3*atr)
        if tgt_type in ("Signal Reversal Exit","Trailing Target (Display Only)"): return entry+3*atr
        return entry + tgt_val*atr
    else:
        if tgt_type=="ATR Based":               return entry - tgt_val*atr
        if tgt_type=="Custom Points":           return entry - tgt_val
        if tgt_type=="Risk Reward Ratio":       return entry - tgt_val*risk
        if tgt_type=="Volatility Based (BB)":   return r["bb_l"]
        if tgt_type=="Fibonacci Extension":
            fi = fib_levels(df, 50); return fi.get("1.618", entry-3*atr)
        if tgt_type in ("Signal Reversal Exit","Trailing Target (Display Only)"): return entry-3*atr
        return entry - tgt_val*atr

def trail_sl(cur_sl, price, df, i, side, sl_type, sl_val):
    r = df.iloc[i]; atr = r["atr"]
    if side=="LONG":
        if sl_type=="Trailing – Fixed Points":     return max(cur_sl, price - sl_val)
        if sl_type=="Trailing – Candle Low/High":  return max(cur_sl, r["Low"])
        if sl_type=="Trailing – Pivot Low/High":
            if i>=3: return max(cur_sl, df["Low"].iloc[i-3:i].min())
        if sl_type=="ATR Based":                   return max(cur_sl, price - sl_val*atr)
        if sl_type=="Volatility Based (BB)":       return max(cur_sl, r["bb_l"])
    else:
        if sl_type=="Trailing – Fixed Points":     return min(cur_sl, price + sl_val)
        if sl_type=="Trailing – Candle Low/High":  return min(cur_sl, r["High"])
        if sl_type=="Trailing – Pivot Low/High":
            if i>=3: return min(cur_sl, df["High"].iloc[i-3:i].max())
        if sl_type=="ATR Based":                   return min(cur_sl, price + sl_val*atr)
        if sl_type=="Volatility Based (BB)":       return min(cur_sl, r["bb_u"])
    return cur_sl

_TRAIL_SL_TYPES = {"ATR Based","Trailing – Fixed Points",
                    "Trailing – Candle Low/High","Trailing – Pivot Low/High",
                    "Volatility Based (BB)"}

# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df, signals, sl_type, sl_val, tgt_type, tgt_val, capital=100000.0):
    """
    Entry  : candle N signal  →  candle N+1 Open
    SL     : checked against candle Low  FIRST  (conservative)
    Target : checked against candle High SECOND
    """
    trades, pos = [], None
    equity = capital
    eq_curve = [capital]

    for i in range(1, len(df)):
        row   = df.iloc[i]
        psig  = signals.iloc[i-1]

        # ── ENTRY ──
        if pos is None:
            if psig not in (1,-1): continue
            side = "LONG" if psig==1 else "SHORT"
            ep   = float(row["Open"])
            sl   = init_sl(df, i-1, side, sl_type, sl_val, ep)
            tgt  = init_tgt(df, i-1, side, tgt_type, tgt_val, ep, sl)

            # Validate
            if side=="LONG"  and (sl>=ep or tgt<=ep): continue
            if side=="SHORT" and (sl<=ep or tgt>=ep): continue

            risk = abs(ep-sl)
            if risk<=0: continue
            qty = max(1, int((equity*0.02)/risk))

            pos = dict(side=side, ep=ep, sl=sl, tgt=tgt, isl=sl, itgt=tgt,
                       qty=qty, etime=row.name, sig=psig)
            continue

        # ── TRAILING SL UPDATE (before exit check) ──
        if sl_type in _TRAIL_SL_TYPES:
            pos["sl"] = trail_sl(pos["sl"],
                                  float(row["High"]) if pos["side"]=="LONG" else float(row["Low"]),
                                  df, i, pos["side"], sl_type, sl_val)

        # ── TRAILING TARGET (display only, no hit) ──
        if tgt_type=="Trailing Target (Display Only)":
            pos["tgt"] = (float(row["Close"]) + 2*float(row["atr"])) if pos["side"]=="LONG" \
                    else (float(row["Close"]) - 2*float(row["atr"]))

        # ── EXIT CHECK ──
        xp, xr = None, None
        if pos["side"]=="LONG":
            if float(row["Low"])  <= pos["sl"]:
                xp, xr = pos["sl"], "SL"
            elif tgt_type not in ("Trailing Target (Display Only)","Signal Reversal Exit") \
                    and float(row["High"]) >= pos["tgt"]:
                xp, xr = pos["tgt"], "TARGET"
            elif tgt_type=="Signal Reversal Exit" or sl_type=="Signal Reversal Exit":
                if signals.iloc[i]==-1: xp, xr = float(row["Close"]), "SIGNAL"
        else:
            if float(row["High"]) >= pos["sl"]:
                xp, xr = pos["sl"], "SL"
            elif tgt_type not in ("Trailing Target (Display Only)","Signal Reversal Exit") \
                    and float(row["Low"]) <= pos["tgt"]:
                xp, xr = pos["tgt"], "TARGET"
            elif tgt_type=="Signal Reversal Exit" or sl_type=="Signal Reversal Exit":
                if signals.iloc[i]==1: xp, xr = float(row["Close"]), "SIGNAL"

        if xr:
            pnl = (xp-pos["ep"])*pos["qty"] if pos["side"]=="LONG" else (pos["ep"]-xp)*pos["qty"]
            equity += pnl
            trades.append(dict(
                EntryTime=pos["etime"], ExitTime=row.name,
                Side=pos["side"], EntryPrice=round(pos["ep"],2), ExitPrice=round(xp,2),
                SL=round(pos["isl"],2), Target=round(pos["itgt"],2),
                Qty=pos["qty"], PnL=round(pnl,2),
                PnLpct=round(pnl/(pos["ep"]*pos["qty"])*100,2),
                ExitReason=xr, Equity=round(equity,2)
            ))
            eq_curve.append(equity)
            pos = None

    # Close open position at last close
    if pos:
        xp = float(df["Close"].iloc[-1])
        pnl = (xp-pos["ep"])*pos["qty"] if pos["side"]=="LONG" else (pos["ep"]-xp)*pos["qty"]
        equity += pnl
        trades.append(dict(
            EntryTime=pos["etime"], ExitTime=df.index[-1],
            Side=pos["side"], EntryPrice=round(pos["ep"],2), ExitPrice=round(xp,2),
            SL=round(pos["isl"],2), Target=round(pos["itgt"],2),
            Qty=pos["qty"], PnL=round(pnl,2),
            PnLpct=round(pnl/(pos["ep"]*pos["qty"])*100,2),
            ExitReason="OPEN", Equity=round(equity,2)
        ))

    tdf = pd.DataFrame(trades)
    m   = metrics(tdf, capital, equity, eq_curve)
    return dict(trades=tdf, eq=eq_curve, final=equity, metrics=m, df=df, sigs=signals)

def metrics(tdf, cap, final, eq):
    if tdf.empty:
        return dict(n=0,wr=0,pnl=0,pf=0,dd=0,ret=0,aw=0,al=0,rr=0,exp=0,wins=0,loss=0,sh=0)
    w = tdf[tdf.PnL>0]; l = tdf[tdf.PnL<=0]
    wr   = len(w)/len(tdf)*100
    gp   = w.PnL.sum() if not w.empty else 0
    gl   = abs(l.PnL.sum()) if not l.empty else 1e-9
    pf   = gp/gl
    aw   = w.PnL.mean() if not w.empty else 0
    al   = abs(l.PnL.mean()) if not l.empty else 0
    eq_s = pd.Series(eq)
    dd   = abs(((eq_s - eq_s.cummax())/eq_s.cummax()*100).min())
    ret  = (final-cap)/cap*100
    rr   = aw/al if al>0 else 0
    exp  = (wr/100*aw) - ((1-wr/100)*al)
    arr  = np.diff(np.array(eq))/np.array(eq[:-1])
    sh   = arr.mean()/arr.std()*np.sqrt(252) if len(arr)>1 and arr.std()>0 else 0
    return dict(n=len(tdf),wr=round(wr,1),pnl=round(gp-gl+gp*0+tdf.PnL.sum(),2),
                pf=round(pf,2),dd=round(dd,1),ret=round(ret,1),
                aw=round(aw,2),al=round(al,2),rr=round(rr,2),exp=round(exp,2),
                wins=len(w),loss=len(l),sh=round(sh,2))

# ─────────────────────────────────────────────────────────────────────────────
# DHAN BROKER INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────
def dhan_equity_order(cfg, sec_id, txn, qty, prod, exch, otype, price=0):
    """Place intraday/delivery order via pydhan."""
    try:
        from pydhan import pydhan
        dhan = pydhan(client_id=cfg["client_id"], access_token=cfg["access_token"])
        exch_seg = dhan.NSE if exch=="NSE" else dhan.BSE
        res = dhan.place_order(
            security_id=str(sec_id),
            exchange_segment=exch_seg,
            transaction_type=dhan.BUY if txn=="BUY" else dhan.SELL,
            quantity=int(qty),
            order_type=dhan.MARKET if otype=="MARKET" else dhan.LIMIT,
            product_type=dhan.INTRADAY if prod=="INTRADAY" else dhan.DELIVERY,
            price=float(price) if otype=="LIMIT" else 0,
        )
        return {"ok":True,"resp":res}
    except ImportError:
        return {"ok":False,"err":"pydhan not installed (pip install pydhan)"}
    except Exception as e:
        return {"ok":False,"err":str(e)}

def dhan_options_order(cfg, sec_id, txn, qty, exch_seg, otype, price=0):
    """Place options order via dhanhq."""
    try:
        from dhanhq import dhanhq
        dhan = dhanhq(cfg["client_id"], cfg["access_token"])
        res = dhan.place_order(
            transactionType=txn,
            exchangeSegment=exch_seg,
            productType="INTRADAY",
            orderType=otype,
            validity="DAY",
            securityId=str(sec_id),
            quantity=int(qty),
            price=float(price) if otype=="LIMIT" else 0,
            triggerPrice=0,
        )
        return {"ok":True,"resp":res}
    except ImportError:
        return {"ok":False,"err":"dhanhq not installed (pip install dhanhq)"}
    except Exception as e:
        return {"ok":False,"err":str(e)}

def dhan_entry(signal: str, ltp: float, cfg: dict):
    """signal: 'BUY' or 'SELL'"""
    if not cfg.get("enabled"): return {"ok":False,"err":"Dhan disabled"}
    if not cfg.get("client_id") or not cfg.get("access_token"):
        return {"ok":False,"err":"Missing credentials"}
    price = ltp if cfg.get("entry_otype")=="LIMIT" else 0

    if cfg.get("options"):
        # BUY signal → Buy CE   |   SELL signal → Buy PE
        sid = cfg["ce_id"] if signal=="BUY" else cfg["pe_id"]
        if not sid: return {"ok":False,"err":"Security ID not set"}
        return dhan_options_order(cfg, sid, "BUY", cfg.get("opt_qty",65),
                                   cfg.get("opt_exch","NSE_FNO"),
                                   cfg.get("entry_otype","MARKET"), price)
    else:
        txn = "BUY" if signal=="BUY" else "SELL"
        return dhan_equity_order(cfg, cfg.get("sec_id","1594"), txn,
                                  cfg.get("qty",1), cfg.get("prod","INTRADAY"),
                                  cfg.get("exch","NSE"), cfg.get("entry_otype","MARKET"), price)

def dhan_exit(entry_signal: str, ltp: float, cfg: dict):
    if not cfg.get("enabled"): return {"ok":False,"err":"Dhan disabled"}
    price = ltp if cfg.get("exit_otype")=="MARKET" else ltp  # always use LTP for exit price

    if cfg.get("options"):
        sid = cfg["ce_id"] if entry_signal=="BUY" else cfg["pe_id"]
        if not sid: return {"ok":False,"err":"Security ID not set"}
        return dhan_options_order(cfg, sid, "SELL", cfg.get("opt_qty",65),
                                   cfg.get("opt_exch","NSE_FNO"),
                                   cfg.get("exit_otype","MARKET"), price)
    else:
        txn = "SELL" if entry_signal=="BUY" else "BUY"
        return dhan_equity_order(cfg, cfg.get("sec_id","1594"), txn,
                                  cfg.get("qty",1), cfg.get("prod","INTRADAY"),
                                  cfg.get("exch","NSE"), cfg.get("exit_otype","MARKET"), price)

# ─────────────────────────────────────────────────────────────────────────────
# LIVE TRADING THREAD
# ─────────────────────────────────────────────────────────────────────────────
def live_loop(ticker, interval, period, strategy, sl_type, sl_val,
              tgt_type, tgt_val, use_ind, ena, dhan_cfg):

    def log(msg, lvl="INFO"):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}][{lvl}] {msg}"
        lg = STORE.get("live_log", [])
        lg.append(entry)
        STORE.set("live_log", lg[-120:])

    log(f"Started: {ticker} | {interval} | {strategy}")

    while STORE.get("live_running", False):
        try:
            df = fetch_data(ticker, interval, period)
            if df is None or len(df) < 60:
                log("Insufficient data – retrying", "WARN"); time.sleep(3); continue

            df  = add_indicators(df)
            sig = get_signals(df, strategy, use_ind, ena)

            ltp     = float(df["Close"].iloc[-1])
            atr_now = float(df["atr"].iloc[-1])
            cur_sig = int(sig.iloc[-1])
            prv_sig = int(sig.iloc[-2]) if len(sig)>=2 else 0

            STORE.upd(dict(live_ltp=ltp, live_df=df, error=None,
                           live_signal="BUY" if cur_sig==1 else "SELL" if cur_sig==-1 else "HOLD"))

            pos = STORE.get("live_position")

            # ── ENTRY ──
            if pos is None:
                if prv_sig not in (1,-1): continue
                side = "LONG" if prv_sig==1 else "SHORT"
                sl_  = init_sl(df, -2, side, sl_type, sl_val, ltp)
                tgt_ = init_tgt(df, -2, side, tgt_type, tgt_val, ltp, sl_)
                valid = (side=="LONG" and sl_<ltp<tgt_) or (side=="SHORT" and tgt_<ltp<sl_)
                if not valid: continue

                new_pos = dict(side=side, sig="BUY" if prv_sig==1 else "SELL",
                               ep=ltp, sl=sl_, tgt=tgt_, isl=sl_, itgt=tgt_,
                               qty=1, etime=datetime.datetime.now())
                STORE.set("live_position", new_pos)
                STORE.set("trailing_sl",   sl_)
                STORE.set("trailing_target", tgt_)
                log(f"ENTRY {side} @ {ltp:.2f}  SL={sl_:.2f}  TGT={tgt_:.2f}")

                if dhan_cfg.get("enabled"):
                    r = dhan_entry("BUY" if prv_sig==1 else "SELL", ltp, dhan_cfg)
                    log(f"Dhan entry: {r}")

            else:
                # ── TRAIL SL ──
                if sl_type in _TRAIL_SL_TYPES:
                    t_sl = STORE.get("trailing_sl", pos["sl"])
                    ref  = ltp
                    t_sl = trail_sl(t_sl, ref, df, -1, pos["side"], sl_type, sl_val)
                    STORE.set("trailing_sl", t_sl)
                    pos["sl"] = t_sl
                    STORE.set("live_position", pos)

                # ── TRAIL TARGET (display) ──
                if tgt_type=="Trailing Target (Display Only)":
                    tt = ltp+2*atr_now if pos["side"]=="LONG" else ltp-2*atr_now
                    STORE.set("trailing_target", tt)

                # ── PNL ──
                pnl = (ltp-pos["ep"])*pos["qty"] if pos["side"]=="LONG" else (pos["ep"]-ltp)*pos["qty"]
                STORE.set("live_pnl", pnl)

                # ── EXIT CHECK (LTP based) ──
                xp, xr = None, None
                if pos["side"]=="LONG":
                    if ltp<=pos["sl"]: xp,xr=ltp,"SL"
                    elif tgt_type not in ("Trailing Target (Display Only)","Signal Reversal Exit") \
                            and ltp>=pos["tgt"]: xp,xr=ltp,"TARGET"
                    elif sl_type=="Signal Reversal Exit" or tgt_type=="Signal Reversal Exit":
                        if cur_sig==-1: xp,xr=ltp,"SIGNAL"
                else:
                    if ltp>=pos["sl"]: xp,xr=ltp,"SL"
                    elif tgt_type not in ("Trailing Target (Display Only)","Signal Reversal Exit") \
                            and ltp<=pos["tgt"]: xp,xr=ltp,"TARGET"
                    elif sl_type=="Signal Reversal Exit" or tgt_type=="Signal Reversal Exit":
                        if cur_sig==1: xp,xr=ltp,"SIGNAL"

                if xr:
                    final_pnl = (xp-pos["ep"])*pos["qty"] if pos["side"]=="LONG" \
                                else (pos["ep"]-xp)*pos["qty"]
                    rec = dict(EntryTime=pos["etime"], ExitTime=datetime.datetime.now(),
                               Side=pos["side"], EntryPrice=round(pos["ep"],2),
                               ExitPrice=round(xp,2), SL=round(pos["isl"],2),
                               Target=round(pos["itgt"],2), Qty=pos["qty"],
                               PnL=round(final_pnl,2),
                               PnLpct=round(final_pnl/(pos["ep"]*pos["qty"])*100,2),
                               ExitReason=xr, Source="LIVE")
                    h = STORE.get("trade_history", []); h.append(rec); STORE.set("trade_history",h)
                    log(f"EXIT {pos['side']} @ {xp:.2f} | {xr} | PnL={final_pnl:+.2f}")
                    if dhan_cfg.get("enabled"):
                        r2 = dhan_exit(pos["sig"], xp, dhan_cfg)
                        log(f"Dhan exit: {r2}")
                    STORE.clear_position()

        except Exception:
            STORE.set("error", traceback.format_exc()); time.sleep(5)

    log("Live trading stopped.")

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def make_chart(df, sigs, trades_df=None):
    vol = has_volume(df)
    rows = 4 if vol else 3
    heights = [0.55,0.15,0.15,0.15] if vol else [0.60,0.20,0.20]
    specs  = [[{"secondary_y":True}]]*rows
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                         vertical_spacing=0.03, row_heights=heights,
                         specs=specs,
                         subplot_titles=["Price + Indicators","RSI","MACD",
                                          "ADX + Volume"][:rows])

    # Candles
    fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High,
                                  low=df.Low, close=df.Close, name="Price",
                                  increasing_line_color="#26a69a",
                                  decreasing_line_color="#ef5350"),
                  row=1,col=1,secondary_y=False)

    # EMAs
    for p,col,dash in [(9,"#ff9800","solid"),(21,"#2196f3","solid"),(50,"#ba68c8","dash")]:
        cn = f"ema{p}"
        if cn in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df[cn],name=f"EMA{p}",
                                      line=dict(color=col,width=1,dash=dash),opacity=0.85),
                          row=1,col=1,secondary_y=False)

    # SuperTrend
    if "st" in df.columns:
        bull = df["st"].where(df["st_dir"]==1); bear = df["st"].where(df["st_dir"]==-1)
        fig.add_trace(go.Scatter(x=df.index,y=bull,name="ST↑",
                                  line=dict(color="#00e676",width=1.8),connectgaps=False),
                      row=1,col=1,secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index,y=bear,name="ST↓",
                                  line=dict(color="#ff5252",width=1.8),connectgaps=False),
                      row=1,col=1,secondary_y=False)

    # Bollinger Bands
    if "bb_u" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df.bb_u,name="BB↑",
                                  line=dict(color="rgba(100,181,246,0.4)",width=1,dash="dot")),
                      row=1,col=1,secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index,y=df.bb_l,name="BB↓",
                                  line=dict(color="rgba(100,181,246,0.4)",width=1,dash="dot"),
                                  fill="tonexty",fillcolor="rgba(100,181,246,0.05)"),
                      row=1,col=1,secondary_y=False)

    # Trade markers
    if trades_df is not None and not trades_df.empty:
        L = trades_df[trades_df.Side=="LONG"]
        S = trades_df[trades_df.Side=="SHORT"]
        for sub, col, sym, nm in [
            (L,"#00e676","triangle-up","Buy Entry"),
            (L,"#ffca28","triangle-down","Buy Exit"),
            (S,"#ff5252","triangle-down","Sell Entry"),
        ]:
            if sub.empty: continue
            xt = sub.EntryTime if nm!="Buy Exit" else sub.ExitTime
            yt = sub.EntryPrice if nm!="Buy Exit" else sub.ExitPrice
            fig.add_trace(go.Scatter(x=xt,y=yt,mode="markers",name=nm,
                                      marker=dict(symbol=sym,size=11,color=col,
                                                  line=dict(color="white",width=1))),
                          row=1,col=1,secondary_y=False)

    # RSI
    fig.add_trace(go.Scatter(x=df.index,y=df.rsi,name="RSI",
                              line=dict(color="#ff9800",width=1.5)),row=2,col=1,secondary_y=False)
    for y,c in [(70,"rgba(255,82,82,0.5)"),(30,"rgba(0,230,118,0.5)"),(50,"rgba(150,150,150,0.3)")]:
        fig.add_hline(y=y,line_dash="dash",line_color=c,row=2,col=1)

    # MACD
    if "macd" in df.columns:
        hcol = ["#26a69a" if v>=0 else "#ef5350" for v in df.macd_hist]
        fig.add_trace(go.Bar(x=df.index,y=df.macd_hist,name="Hist",
                              marker_color=hcol,opacity=0.7),row=3,col=1,secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index,y=df.macd,name="MACD",
                                  line=dict(color="#2196f3",width=1.4)),row=3,col=1,secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index,y=df.macd_sig,name="Sig",
                                  line=dict(color="#ff9800",width=1.4)),row=3,col=1,secondary_y=False)

    # ADX + Volume
    if "adx" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df.adx,name="ADX",
                                  line=dict(color="#ba68c8",width=1.5)),row=4,col=1,secondary_y=False)
        fig.add_hline(y=25,line_dash="dash",line_color="rgba(255,202,40,0.5)",row=4,col=1)
    if vol:
        vc = ["#26a69a" if c>=o else "#ef5350" for c,o in zip(df.Close,df.Open)]
        fig.add_trace(go.Bar(x=df.index,y=df.Volume,name="Vol",
                              marker_color=vc,opacity=0.45),row=4,col=1,secondary_y=True)

    fig.update_layout(template="plotly_dark",height=920,
                       showlegend=True,
                       legend=dict(orientation="h",y=1.01,x=0,font=dict(size=10)),
                       xaxis_rangeslider_visible=False,
                       plot_bgcolor="#0d1117",paper_bgcolor="#0d1117",
                       margin=dict(l=0,r=0,t=28,b=0))
    return fig

def eq_chart(eq, cap):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=eq,mode="lines",name="Equity",
                              fill="tozeroy",fillcolor="rgba(0,230,118,0.08)",
                              line=dict(color="#00e676",width=2)))
    fig.add_hline(y=cap,line_dash="dash",line_color="rgba(200,200,200,0.4)")
    fig.update_layout(template="plotly_dark",height=260,title="Equity Curve",
                       showlegend=False,plot_bgcolor="#0d1117",paper_bgcolor="#0d1117",
                       margin=dict(l=0,r=0,t=30,b=0))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # ── Market ──
        st.markdown("### 📊 Market")
        tn = st.selectbox("Asset", list(TICKERS.keys()))
        ticker = st.text_input("Ticker symbol", TICKERS[tn]) if tn=="Custom ✏️" else TICKERS[tn]
        if tn!="Custom ✏️": st.caption(f"`{ticker}`")

        c1,c2 = st.columns(2)
        with c1: interval = st.selectbox("Timeframe", INTERVALS, index=4)
        with c2: period   = st.selectbox("Period",    PERIODS,   index=4)

        # ── Strategy ──
        st.markdown("### 🧠 Strategy")
        strategy = st.selectbox("Strategy", STRATEGIES)
        DESCS = {
            "🏆 Smart Algo (Recommended)":
                "Regime-aware: Trending → SuperTrend+EMA+MACD. Ranging → BB reversal+RSI+Liquidity Hunt. Best all-rounder.",
            "SuperTrend + EMA Cloud": "SuperTrend flip filtered by EMA50 position.",
            "EMA Crossover + ADX":    "9/21 EMA cross only when ADX>20.",
            "MACD Momentum":          "MACD crossovers with EMA trend filter.",
            "Bollinger Squeeze":      "Squeeze release breakout + BB boundary breakout.",
            "Heikin Ashi Reversal":   "HA candle color flip with EMA21 confirmation.",
            "Smart Money Concepts (SMC)":"Order blocks + liquidity hunt entries.",
            "Opening Range Breakout": "First-5-candle range breakout (best for 1m–5m).",
            "Donchian Breakout":      "20-period Donchian channel breakout with ADX filter.",
        }
        with st.expander("ℹ️ Strategy Info"):
            st.info(DESCS.get(strategy,""))

        capital = st.number_input("Backtest Capital (₹)", 10000, 10000000, 100000, 10000)

        # ── Risk Mgmt ──
        st.markdown("### 🎯 Risk Management")
        c1,c2 = st.columns(2)
        with c1: sl_type  = st.selectbox("SL Type",     SL_TYPES)
        with c2: tgt_type = st.selectbox("Target Type", TGT_TYPES)
        c3,c4 = st.columns(2)
        with c3: sl_val  = st.number_input("SL Value",     0.1, 500.0, 1.5, 0.1)
        with c4: tgt_val = st.number_input("Target Value", 0.1, 500.0, 3.0, 0.1)

        sl_note = ("× ATR" if "ATR" in sl_type else
                   "Points" if "Custom" in sl_type or "Pivot" in sl_type or "Fixed" in sl_type else "")
        tgt_note = ("× ATR" if "ATR" in tgt_type else
                    "× Risk" if "Ratio" in tgt_type else "Points" if "Custom" in tgt_type else "")
        if sl_note:  st.caption(f"SL unit: **{sl_note}**  |  Tgt unit: **{tgt_note}**")

        # ── Indicators Filter ──
        st.markdown("### 📈 Indicator Filters")
        use_ind = st.checkbox("Enable Filters (adds extra conditions)", value=False)
        ena = {}
        if use_ind:
            with st.expander("Select Filters"):
                c1,c2 = st.columns(2)
                with c1:
                    ena["rsi"]     = st.checkbox("RSI",          True)
                    ena["macd"]    = st.checkbox("MACD",         True)
                    ena["adx"]     = st.checkbox("ADX",          True)
                    ena["ema"]     = st.checkbox("EMA Align",    True)
                    ena["ob"]      = st.checkbox("Order Block",  False)
                with c2:
                    ena["volume"]  = st.checkbox("Volume",       False)
                    ena["bb"]      = st.checkbox("BB Extreme",   False)
                    ena["lh"]      = st.checkbox("Liq Hunt",     False)
                    ena["atr"]     = st.checkbox("ATR Filter",   False)

        # ── Dhan Broker ──
        st.markdown("### 🏦 Dhan Broker")
        dhan_en = st.checkbox("Enable Dhan Broker", value=False)
        dcfg = {"enabled": dhan_en}
        if dhan_en:
            with st.expander("🔑 Credentials", expanded=True):
                dcfg["client_id"]    = st.text_input("Client ID",     type="password")
                dcfg["access_token"] = st.text_input("Access Token",  type="password")

            opts = st.checkbox("Options Trading", value=False)
            dcfg["options"] = opts
            if opts:
                dcfg["opt_exch"]  = st.selectbox("F&O Exchange", ["NSE_FNO","BSE_FNO"])
                dcfg["ce_id"]     = st.text_input("CE Security ID", "")
                dcfg["pe_id"]     = st.text_input("PE Security ID", "")
                dcfg["opt_qty"]   = st.number_input("Lot Qty", 1, 10000, 65)
                dcfg["entry_otype"]= st.selectbox("Entry Order", ["MARKET","LIMIT"])
                dcfg["exit_otype"] = st.selectbox("Exit Order",  ["MARKET","LIMIT"])
                st.success("BUY signal → CE Buy  |  SELL signal → PE Buy")
            else:
                dcfg["prod"]       = st.selectbox("Product",      ["INTRADAY","DELIVERY"])
                dcfg["exch"]       = st.selectbox("Exchange",     ["NSE","BSE"])
                dcfg["sec_id"]     = st.text_input("Security ID", "1594")
                dcfg["qty"]        = st.number_input("Quantity",  1, 100000, 1)
                dcfg["entry_otype"]= st.selectbox("Entry Order",  ["LIMIT","MARKET"])
                dcfg["exit_otype"] = st.selectbox("Exit Order",   ["MARKET","LIMIT"])
                st.caption("LIMIT orders use current LTP as price.")

        return dict(ticker=ticker, ticker_name=tn, interval=interval, period=period,
                    strategy=strategy, capital=capital,
                    sl_type=sl_type, sl_val=sl_val, tgt_type=tgt_type, tgt_val=tgt_val,
                    use_ind=use_ind, ena=ena, dcfg=dcfg)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – BACKTESTING
# ─────────────────────────────────────────────────────────────────────────────
def tab_backtest(cfg):
    st.markdown("## 📊 Strategy Backtester")
    c1,c2,c3 = st.columns([3,1,1])
    with c1:
        st.markdown(f"**{cfg['ticker_name']}** `{cfg['ticker']}` | `{cfg['interval']}` | "
                    f"`{cfg['period']}` | **{cfg['strategy']}**")
    with c2:
        run  = st.button("▶️ Run Backtest", type="primary", use_container_width=True)
    with c3:
        clr  = st.button("🗑️ Clear",        use_container_width=True)

    if clr:
        STORE.set("backtest_result", None); STORE.set("backtest_trades",[]); st.rerun()

    if run:
        with st.spinner("Fetching data & running backtest…"):
            df = fetch_data(cfg["ticker"], cfg["interval"], cfg["period"])
            if df is None or df.empty:
                st.error("❌ Could not fetch data – check ticker / period."); return
            if len(df) < 65:
                st.warning(f"⚠️ Only {len(df)} candles. Use a longer period."); return
            df  = add_indicators(df)
            sig = get_signals(df, cfg["strategy"], cfg["use_ind"], cfg["ena"])
            res = run_backtest(df, sig, cfg["sl_type"], cfg["sl_val"],
                               cfg["tgt_type"], cfg["tgt_val"], cfg["capital"])
            STORE.set("backtest_result", res)
            STORE.set("backtest_trades",
                      res["trades"].to_dict("records") if not res["trades"].empty else [])

    res = STORE.get("backtest_result")
    if res is None:
        st.info("👆 Press **Run Backtest** to start."); return

    m = res["metrics"]; tdf = res["trades"]

    # ── Metric cards ──
    st.markdown("### 📈 Performance")
    def mbox(label, val, good=None):
        c = "#00e676" if good is True else "#ff5252" if good is False else "#ffca28"
        return (f'<div class="metric-box"><div class="metric-label">{label}</div>'
                f'<div class="metric-val" style="color:{c}">{val}</div></div>')

    cols = st.columns(6)
    cards = [
        ("Trades",          str(m["n"]),              None),
        ("Win Rate",        f"{m['wr']}%",             m["wr"]>=50),
        ("Total PnL",       f"₹{m['pnl']:,.0f}",      m["pnl"]>=0),
        ("Profit Factor",   str(m["pf"]),              m["pf"]>=1.5),
        ("Max Drawdown",    f"{m['dd']}%",             m["dd"]<=15),
        ("Return",          f"{m['ret']}%",            m["ret"]>=0),
    ]
    for col,(lbl,val,g) in zip(cols,cards):
        col.markdown(mbox(lbl,val,g), unsafe_allow_html=True)

    st.markdown("")
    cols2 = st.columns(5)
    cards2=[("Avg Win",f"₹{m['aw']:,.0f}"),("Avg Loss",f"₹{m['al']:,.0f}"),
            ("Avg R:R",str(m["rr"])),("Expectancy",f"₹{m['exp']:,.0f}"),
            ("Sharpe",str(m["sh"]))]
    for col,(lbl,val) in zip(cols2,cards2):
        col.markdown(f'<div class="metric-box"><div class="metric-label">{lbl}</div>'
                     f'<div class="metric-val white">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    t1,t2,t3 = st.tabs(["📉 Chart","💰 Equity Curve","📋 Trades"])
    with t1:
        st.plotly_chart(make_chart(res["df"], res["sigs"], tdf), use_container_width=True)
    with t2:
        st.plotly_chart(eq_chart(res["eq"], cfg["capital"]), use_container_width=True)
        ca,cb,cc = st.columns(3)
        ca.metric("Initial", f"₹{cfg['capital']:,.0f}")
        cb.metric("Final",   f"₹{res['final']:,.0f}",
                  delta=f"₹{res['final']-cfg['capital']:,.0f}")
        cc.metric("W / L",   f"{m['wins']} / {m['loss']}")
    with t3:
        if not tdf.empty:
            def colour(v):
                try: return "background-color:#00e67622" if float(v)>0 else "background-color:#ff525222"
                except: return ""
            st.dataframe(tdf.style.map(colour, subset=["PnL"]),
                         use_container_width=True, height=420)
            st.download_button("📥 CSV", tdf.to_csv(index=False), "backtest_trades.csv","text/csv")
        else:
            st.info("No trades generated – try adjusting parameters.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – LIVE TRADING
# ─────────────────────────────────────────────────────────────────────────────
def tab_live(cfg):
    st.markdown("## ⚡ Live Trading")
    running = STORE.get("live_running", False)

    c1,c2,c3 = st.columns([3,1,1])
    with c1:
        dot = "🟢" if running else "🔴"
        st.markdown(f"{dot} **{'LIVE' if running else 'STOPPED'}** &nbsp;|&nbsp; "
                    f"**{cfg['ticker_name']}** `{cfg['ticker']}` | `{cfg['interval']}` | {cfg['strategy']}")
    with c2:
        if not running:
            if st.button("▶️ Start Live", type="primary", use_container_width=True):
                STORE.upd(dict(live_running=True, live_log=[], error=None))
                t = threading.Thread(target=live_loop, daemon=True,
                                     args=(cfg["ticker"],cfg["interval"],cfg["period"],
                                           cfg["strategy"],cfg["sl_type"],cfg["sl_val"],
                                           cfg["tgt_type"],cfg["tgt_val"],
                                           cfg["use_ind"],cfg["ena"],cfg["dcfg"]))
                t.start(); STORE.set("live_thread",t)
                st.success("Started!"); st.rerun()
        else:
            if st.button("⏹️ Stop", use_container_width=True):
                STORE.upd(dict(live_running=False)); STORE.clear_position()
                st.warning("Stopping…"); st.rerun()
    with c3:
        if st.button("🔄 Refresh", use_container_width=True): st.rerun()

    err = STORE.get("error")
    if err: st.error(err[:300])

    st.divider()

    # ── Live metrics ──
    ltp    = STORE.get("live_ltp")
    pnl    = STORE.get("live_pnl", 0.0)
    sig    = STORE.get("live_signal","HOLD")
    pos    = STORE.get("live_position")
    t_sl   = STORE.get("trailing_sl")
    t_tgt  = STORE.get("trailing_target")

    def mbox2(label, val, colour="#fff"):
        return (f'<div class="metric-box"><div class="metric-label">{label}</div>'
                f'<div class="metric-val" style="color:{colour}">{val}</div></div>')

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(mbox2("LTP", f"₹{ltp:,.2f}" if ltp else "—"), unsafe_allow_html=True)
    sig_col = {"BUY":"#00e676","SELL":"#ff5252","HOLD":"#ffca28"}.get(sig,"#fff")
    c2.markdown(mbox2("Signal", sig, sig_col), unsafe_allow_html=True)
    pnl_col = "#00e676" if pnl>=0 else "#ff5252"
    c3.markdown(mbox2("Open PnL", f"₹{pnl:+,.2f}" if pnl!=0 else "—", pnl_col), unsafe_allow_html=True)
    pos_txt = f"{pos['side']} @ ₹{pos['ep']:,.2f}" if pos else "Flat"
    pos_col = "#00e676" if pos and pos["side"]=="LONG" else "#ff5252" if pos else "#6b7280"
    c4.markdown(mbox2("Position", pos_txt, pos_col), unsafe_allow_html=True)

    # ── Position detail ──
    if pos and ltp:
        st.markdown("")
        sl_now  = t_sl  or pos["sl"]
        tgt_now = t_tgt or pos["tgt"]
        ca,cb,cc,cd = st.columns(4)
        ca.metric("Entry",  f"₹{pos['ep']:,.2f}")
        cb.metric("SL",     f"₹{sl_now:,.2f}",  delta=f"{ltp-sl_now:+.2f}")
        cc.metric("Target", f"₹{tgt_now:,.2f}", delta=f"{tgt_now-ltp:+.2f}")
        risk = abs(pos["ep"]-sl_now); rew = abs(tgt_now-pos["ep"])
        cd.metric("R:R", f"{rew/risk:.2f}" if risk>0 else "—")

        # Progress bar
        if pos["side"]=="LONG":
            prog = (ltp-sl_now)/(tgt_now-sl_now) if tgt_now>sl_now else 0
        else:
            prog = (sl_now-ltp)/(sl_now-tgt_now) if sl_now>tgt_now else 0
        prog = max(0.0, min(1.0, prog))
        st.progress(prog, text=f"Trade progress: {prog*100:.1f}%")

        # Manual exit
        if st.button("🚨 Manual Exit", type="secondary"):
            ep_exit = ltp
            fpnl = (ep_exit-pos["ep"])*pos["qty"] if pos["side"]=="LONG" \
                   else (pos["ep"]-ep_exit)*pos["qty"]
            rec = dict(EntryTime=pos["etime"], ExitTime=datetime.datetime.now(),
                       Side=pos["side"], EntryPrice=round(pos["ep"],2),
                       ExitPrice=round(ep_exit,2), SL=round(pos["isl"],2),
                       Target=round(pos["itgt"],2), Qty=pos["qty"],
                       PnL=round(fpnl,2),
                       PnLpct=round(fpnl/(pos["ep"]*pos["qty"])*100,2),
                       ExitReason="MANUAL", Source="LIVE")
            h = STORE.get("trade_history",[]); h.append(rec); STORE.set("trade_history",h)
            if cfg["dcfg"].get("enabled"):
                r = dhan_exit(pos["sig"], ep_exit, cfg["dcfg"])
                st.info(f"Dhan: {r}")
            STORE.clear_position()
            st.success(f"Exited @ ₹{ep_exit:,.2f}  PnL: ₹{fpnl:+,.2f}"); st.rerun()

    st.divider()

    # ── Live chart ──
    ldf = STORE.get("live_df")
    if ldf is not None:
        with st.expander("📈 Live Chart", expanded=True):
            fc = make_chart(ldf, pd.Series(0,index=ldf.index))
            if pos:
                sl_now  = t_sl  or pos["sl"]
                tgt_now = t_tgt or pos["tgt"]
                fc.add_hline(y=pos["ep"],  line_color="#ffca28", line_dash="solid",
                              annotation_text=f"Entry {pos['ep']:.2f}", row=1, col=1)
                fc.add_hline(y=sl_now,     line_color="#ff5252", line_dash="dash",
                              annotation_text=f"SL {sl_now:.2f}", row=1, col=1)
                fc.add_hline(y=tgt_now,    line_color="#00e676", line_dash="dash",
                              annotation_text=f"Tgt {tgt_now:.2f}", row=1, col=1)
            st.plotly_chart(fc, use_container_width=True)

    # ── Activity log ──
    with st.expander("📋 Activity Log"):
        log = STORE.get("live_log",[])
        if log: st.code("\n".join(reversed(log[-40:])), language=None)
        else: st.caption("No activity yet.")

    if running: time.sleep(2); st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – TRADE HISTORY
# ─────────────────────────────────────────────────────────────────────────────
def tab_history():
    st.markdown("## 📚 Trade History")
    live_h = STORE.get("trade_history", [])
    bt_h   = STORE.get("backtest_trades", [])

    t1,t2,t3 = st.tabs(["🔴 Live Trades","📊 Backtest Trades","📈 Combined Stats"])

    with t1:
        if live_h:
            ldf = pd.DataFrame(live_h)
            st.dataframe(ldf, use_container_width=True, height=400)
            w = sum(1 for t in live_h if t.get("PnL",0)>0)
            tp = sum(t.get("PnL",0) for t in live_h)
            ca,cb,cc = st.columns(3)
            ca.metric("Trades",   len(live_h))
            cb.metric("Win Rate", f"{w/len(live_h)*100:.1f}%")
            cc.metric("Total PnL",f"₹{tp:+,.2f}")
            if st.button("🗑️ Clear Live History"):
                STORE.set("trade_history",[]); st.rerun()
        else: st.info("No live trades yet.")

    with t2:
        if bt_h:
            bdf = pd.DataFrame(bt_h)
            st.dataframe(bdf, use_container_width=True, height=400)
            st.download_button("📥 CSV", bdf.to_csv(index=False),"bt_trades.csv","text/csv")
        else: st.info("Run a backtest first.")

    with t3:
        all_t = []
        for t in live_h: all_t.append({**t,"Source":"LIVE"})
        for t in bt_h:   all_t.append({**t,"Source":"BACKTEST"})
        if not all_t: st.info("No trades yet."); return

        adf = pd.DataFrame(all_t)
        if "PnL" in adf.columns:
            tp  = adf.PnL.sum()
            w   = len(adf[adf.PnL>0])
            tot = len(adf)
            ca,cb,cc,cd = st.columns(4)
            ca.metric("Total",     tot)
            cb.metric("Win Rate",  f"{w/tot*100:.1f}%")
            cc.metric("Total PnL", f"₹{tp:+,.2f}")
            cd.metric("W/L",       f"{w}/{tot-w}")

            fig = go.Figure()
            cols = ["#00e676" if p>0 else "#ff5252" for p in adf.PnL]
            fig.add_trace(go.Bar(y=adf.PnL, marker_color=cols, name="PnL/Trade"))
            fig.update_layout(template="plotly_dark",height=280,
                               title="Per-Trade PnL Distribution",
                               plot_bgcolor="#0d1117",paper_bgcolor="#0d1117",
                               margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown(
        '<div style="background:linear-gradient(90deg,#0a1628,#0f2952);'
        'padding:14px 20px;border-radius:10px;margin-bottom:18px;">'
        '<span style="color:#00e676;font-size:22px;font-weight:700;">🧠 Smart Investing Platform</span>'
        '<span style="color:#6b7280;font-size:13px;margin-left:16px;">'
        'NSE · BSE · Crypto · Forex · Commodities</span></div>',
        unsafe_allow_html=True)

    cfg = sidebar()
    tb1, tb2, tb3 = st.tabs(["📊 Backtesting", "⚡ Live Trading", "📚 Trade History"])
    with tb1: tab_backtest(cfg)
    with tb2: tab_live(cfg)
    with tb3: tab_history()

if __name__ == "__main__":
    main()
