"""
╔══════════════════════════════════════════════════════════════════════╗
║          SMART INVESTING — Advanced Algo Trading Platform            ║
║   Backtesting · Live Trading · Dhan Broker · EMA Strategies          ║
╚══════════════════════════════════════════════════════════════════════╝
Install: pip install streamlit yfinance pandas numpy plotly pytz dhanhq requests
Run    : streamlit run smart_investing.py
"""

# ─── IMPORTS ────────────────────────────────────────────────────────────────
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time, datetime, threading, math, warnings, requests, json
from datetime import timedelta

try:
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
except ImportError:
    IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Smart Investing", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container{padding-top:.5rem!important}
[data-testid="stSidebarContent"]{background:#0d1117}
.app-hdr{font-size:1.85rem;font-weight:900;letter-spacing:-1px;
  background:linear-gradient(90deg,#64ffda,#00b4d8,#a78bfa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0}
.ltp-wrap{display:flex;align-items:center;gap:18px;
  background:linear-gradient(135deg,#1a1f35,#20263e);
  border:1px solid #2d3560;border-radius:12px;padding:10px 20px;margin-bottom:12px}
.ltp-sym{color:#8892b0;font-size:.75rem;font-weight:700;letter-spacing:1px;text-transform:uppercase}
.ltp-px{color:#e0e6f0;font-size:1.85rem;font-weight:700;line-height:1.1}
.ltp-up{color:#00d4aa;font-weight:700}
.ltp-dn{color:#ff4b5c;font-weight:700}
.ltp-meta{margin-left:auto;color:#8892b0;font-size:.75rem;text-align:right}
.stTabs [data-baseweb="tab-list"]{gap:5px;background:#0d1117;
  border-radius:10px;padding:3px;border:1px solid #1e2340}
.stTabs [data-baseweb="tab"]{border-radius:7px;color:#8892b0;
  font-weight:600;font-size:.87rem;height:34px}
.stTabs [aria-selected="true"]{background:#1e2b50!important;color:#64ffda!important}
.mrow{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}
.mc{background:#1a1f35;border:1px solid #2d3258;border-radius:10px;
  padding:10px 14px;min-width:100px;flex:1;text-align:center}
.mc-v{font-size:1.2rem;font-weight:700;margin:0}
.mc-l{font-size:.7rem;color:#8892b0;margin-top:2px}
.c-g .mc-v{color:#00d4aa} .c-r .mc-v{color:#ff4b5c}
.c-b .mc-v{color:#64b5f6} .c-gold .mc-v{color:#ffd166}
.cfg-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:7px;margin:7px 0}
.cfg-item{background:#151a2d;border-left:3px solid #64ffda;
  border-radius:7px;padding:7px 11px;font-size:.8rem}
.cfg-k{color:#8892b0;font-size:.7rem;text-transform:uppercase;letter-spacing:.5px}
.cfg-v{color:#ccd6f6;font-weight:600}
.pos-card{background:linear-gradient(135deg,#13273f,#1a2040);
  border:1px solid #2d6a4f;border-radius:12px;padding:14px 18px;margin:8px 0}
.pos-buy{border-left:4px solid #00d4aa} .pos-sell{border-left:4px solid #ff4b5c}
.pos-detail{display:flex;flex-wrap:wrap;gap:14px;margin-top:7px}
.pos-f{font-size:.81rem}
.pos-fl{color:#8892b0} .pos-fv{color:#ccd6f6;font-weight:600}
.badge-live{background:#00d4aa22;color:#00d4aa;border:1px solid #00d4aa55;
  border-radius:20px;padding:3px 14px;font-size:.8rem;font-weight:700;display:inline-block}
.badge-idle{background:#8892b022;color:#8892b0;border:1px solid #8892b044;
  border-radius:20px;padding:3px 14px;font-size:.8rem;display:inline-block}
.sbs{color:#64ffda;font-size:.76rem;font-weight:700;letter-spacing:1px;
  text-transform:uppercase;margin:10px 0 3px 0;
  border-bottom:1px solid #1e2340;padding-bottom:3px}
.viol-box{background:#2a0f0f;border:1px solid #ff4b5c55;
  border-radius:8px;padding:10px 14px;margin:6px 0}
[data-testid="stDataFrame"]{border-radius:8px;overflow:hidden}
</style>""", unsafe_allow_html=True)


# ─── CONSTANTS ───────────────────────────────────────────────────────────────
TICKER_MAP = {
    "Nifty 50":"^NSEI","Bank Nifty":"^NSEBANK","Sensex":"^BSESN",
    "Bitcoin (BTC)":"BTC-USD","Ethereum (ETH)":"ETH-USD",
    "Gold":"GC=F","Silver":"SI=F","Custom":None
}
TF_PERIODS = {
    "1m":  ["1d","5d","7d"],
    "5m":  ["1d","5d","7d","1mo"],
    "15m": ["1d","5d","7d","1mo"],
    "1h":  ["1d","5d","7d","1mo","3mo","6mo","1y","2y"],
    "1d":  ["5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"],
    "1wk": ["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"],
}
PERIOD_DAYS = {"1d":1,"5d":5,"7d":7,"1mo":30,"3mo":90,"6mo":180,
               "1y":365,"2y":730,"5y":1825,"10y":3650,"20y":7300}
TF_MIN = {"1m":1,"5m":5,"15m":15,"1h":60,"1d":1440,"1wk":10080}
STRATEGIES  = ["EMA Crossover","Simple Buy","Simple Sell"]
SL_TYPES    = ["Custom Points","ATR Based","Trailing SL","Reverse EMA Crossover","Risk Reward Based"]
TGT_TYPES   = ["Custom Points","Trailing Target","EMA Crossover","Risk Reward Based"]
CROSS_TYPES = ["Simple Crossover","Custom Candle Size","ATR Based Candle Size"]

C = {"cup":"#00d4aa","cdn":"#ff4b5c","ef":"#ffd166","es":"#a78bfa",
     "bm":"#00d4aa","sm":"#ff4b5c","sl":"#ff4b5c","tg":"#00d4aa",
     "bg":"#0d1117","gr":"rgba(255,255,255,0.05)"}

# ─── GLOBAL LIVE STATE ───────────────────────────────────────────────────────
# CRITICAL: @st.cache_resource makes these survive Streamlit reruns.
# Plain module-level dicts get RE-CREATED on every rerun → resets active=False
# → live loop appears to stop after first refresh. cache_resource fixes this.
@st.cache_resource
def _get_live_state():
    return {
        "active":           False,
        "position":         None,
        "completed_trades": [],
        "log":              [],
        "last_ltp":         None,
        "prev_close":       None,
        "last_candle":      None,
        "df":               None,
        "config":           None,
        "_thread":          None,
        "_stop_evt":        None,
    }

@st.cache_resource
def _get_lock():
    return threading.Lock()

_LIVE = _get_live_state()
_LOCK = _get_lock()

# ─── SESSION STATE ────────────────────────────────────────────────────────────
def _init():
    defs = {"bt_results": None, "bt_cfg": None}
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()


# ─── EMA / ATR CALCULATIONS (TradingView-accurate) ───────────────────────────
def ema_tv(series: pd.Series, period: int) -> pd.Series:
    """
    Exact TradingView EMA:
      Seed = SMA of first `period` non-NaN bars, then k=2/(period+1).
      Gaps (NaN) are carried forward so that a big gapup/gapdown never
      breaks the calculation mid-series.
    """
    result = pd.Series(np.nan, index=series.index, dtype=float)
    if len(series) < period:
        return result
    vals = series.values.astype(float)
    n    = len(vals)
    k    = 2.0 / (period + 1)
    streak_start = None
    streak_count = 0
    for i in range(n):
        if not np.isnan(vals[i]):
            if streak_start is None: streak_start = i
            streak_count += 1
        else:
            streak_start = None
            streak_count = 0
        if streak_count == period:
            result.iloc[i] = float(np.nanmean(vals[streak_start:i+1]))
            for j in range(i+1, n):
                prev = result.iloc[j-1]
                if np.isnan(vals[j]):
                    result.iloc[j] = prev
                elif np.isnan(prev):
                    result.iloc[j] = vals[j]
                else:
                    result.iloc[j] = vals[j]*k + prev*(1.0-k)
            break
    return result

def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR via RMA (same as TradingView ATR)."""
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    p = df["Close"].astype(float).shift(1)
    tr = pd.concat([h-l,(h-p).abs(),(l-p).abs()],axis=1).max(axis=1)
    return ema_tv(tr, period)

def build_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    cl = df["Close"].astype(float)
    df["EMA_Fast"] = ema_tv(cl, int(cfg["fast_ema"]))
    df["EMA_Slow"] = ema_tv(cl, int(cfg["slow_ema"]))
    df["ATR"]      = atr_series(df, 14)
    return df

def cross_angle(fast: pd.Series, idx: int) -> float:
    if idx < 1 or idx >= len(fast): return 0.0
    p, c = fast.iloc[idx-1], fast.iloc[idx]
    if pd.isna(p) or pd.isna(c) or p == 0: return 0.0
    return math.degrees(math.atan((c-p)/p*5000))

# ─── DATA FETCHING ────────────────────────────────────────────────────────────
def _fetch_raw(ticker: str, tf: str, period: str, slow_ema: int = 50) -> pd.DataFrame:
    """
    Fetches OHLCV with warmup candles so EMA seed is always accurate,
    even after a big gapup/gapdown.
    '7d' handled via explicit start/end since yfinance doesn't support it natively.
    """
    warmup  = max(int(slow_ema) * 4, 300)
    tf_min  = TF_MIN.get(tf, 60)
    end_dt  = datetime.datetime.now()
    days    = PERIOD_DAYS.get(period, 30)
    st_dt   = end_dt - timedelta(days=days)
    wu_dt   = st_dt  - timedelta(minutes=tf_min*warmup)
    fmt     = "%Y-%m-%d"
    try:
        df = yf.download(
            ticker,
            start       = wu_dt.strftime(fmt),
            end         = (end_dt+timedelta(days=1)).strftime(fmt),
            interval    = tf,
            progress    = False,
            auto_adjust = True,
            prepost     = False,
        )
    except Exception as e:
        st.error(f"yfinance error: {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df = df[cols].copy()
    df.dropna(subset=["Open","High","Low","Close"], inplace=True)
    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)
    except Exception:
        pass
    return df

@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(ticker: str, tf: str, period: str, slow_ema: int = 50) -> pd.DataFrame:
    """Cached wrapper (60 s TTL) for UI / backtest use.
    The live trading thread calls _fetch_raw() directly to always get fresh data."""
    return _fetch_raw(ticker, tf, period, slow_ema)


@st.cache_data(ttl=12, show_spinner=False)
def get_ltp(ticker: str) -> tuple:
    try:
        fi   = yf.Ticker(ticker).fast_info
        ltp  = fi.last_price
        prev = fi.previous_close
        if ltp is None or prev is None:
            h = yf.Ticker(ticker).history(period="5d",interval="1d")
            if len(h) >= 2:
                prev,ltp = float(h["Close"].iloc[-2]),float(h["Close"].iloc[-1])
            elif len(h)==1:
                ltp=prev=float(h["Close"].iloc[-1])
        ltp,prev = float(ltp),float(prev)
        chg  = ltp-prev
        pct  = chg/prev*100 if prev else 0.0
        return ltp,prev,chg,pct
    except Exception:
        return None,None,0.0,0.0


# ─── SIGNAL / SL / TARGET LOGIC ──────────────────────────────────────────────
def _cross(fast, slow, i, bull):
    if i < 1: return False
    f0,f1,s0,s1 = fast[i-1],fast[i],slow[i-1],slow[i]
    if any(np.isnan(v) for v in [f0,f1,s0,s1]): return False
    # Exact TradingView ta.crossover / ta.crossunder (Pine Script source):
    #   crossover(a,b)  = a[1] < b[1]  and  a[0] > b[0]   -- STRICT both sides
    #   crossunder(a,b) = a[1] > b[1]  and  a[0] < b[0]   -- STRICT both sides
    return (f0 < s0 and f1 > s1) if bull else (f0 > s0 and f1 < s1)

def get_signal(df: pd.DataFrame, i: int, cfg: dict):
    """
    Returns 'BUY', 'SELL', or None.
    EMA Crossover  : signal on bar i, caller must enter on bar i+1 open.
    Simple Buy/Sell: signal and entry on bar i close immediately.

    # ── COMMENTED STRATEGIES (not in dropdown) ──────────────────────────────
    # strategy = "Price Crosses Above Threshold"
    #   thr    = cfg.get("price_threshold", 0)
    #   action = cfg.get("thresh_action_above","BUY")   # dropdown: BUY or SELL
    #   prev_c = df["Close"].iloc[i-1] if i>0 else 0
    #   if df["Close"].iloc[i] > thr and prev_c <= thr:
    #       return action
    #
    # strategy = "Price Crosses Below Threshold"
    #   thr    = cfg.get("price_threshold", 0)
    #   action = cfg.get("thresh_action_below","SELL")  # dropdown: BUY or SELL
    #   prev_c = df["Close"].iloc[i-1] if i>0 else 0
    #   if df["Close"].iloc[i] < thr and prev_c >= thr:
    #       return action
    # ─────────────────────────────────────────────────────────────────────────
    """
    strat = cfg["strategy"]
    if strat == "Simple Buy":  return "BUY"
    if strat == "Simple Sell": return "SELL"

    if strat == "EMA Crossover":
        if "EMA_Fast" not in df.columns: return None
        fast = df["EMA_Fast"].values.astype(float)
        slow = df["EMA_Slow"].values.astype(float)
        bull = _cross(fast,slow,i,True)
        bear = _cross(fast,slow,i,False)
        if not bull and not bear: return None
        sig = "BUY" if bull else "SELL"

        # crossover type filter
        ct = cfg.get("crossover_type","Simple Crossover")
        if ct == "Custom Candle Size":
            body = abs(float(df["Close"].iloc[i])-float(df["Open"].iloc[i]))
            if body < cfg.get("custom_candle_size",10.0): return None
        elif ct == "ATR Based Candle Size":
            if "ATR" not in df.columns or pd.isna(df["ATR"].iloc[i]): return None
            body = abs(float(df["Close"].iloc[i])-float(df["Open"].iloc[i]))
            if body < float(df["ATR"].iloc[i]): return None

        # angle filter
        if cfg.get("use_min_angle",False) and cfg.get("min_angle",0)>0:
            if abs(cross_angle(df["EMA_Fast"],i)) < cfg["min_angle"]: return None
        return sig
    return None

def _compute_sl(entry, sig, df, idx, cfg):
    t   = cfg["sl_type"]
    pts = cfg.get("custom_sl_pts",10.0)
    sgn = -1.0 if sig=="BUY" else 1.0
    if t=="Custom Points":       return entry + sgn*pts
    if t=="ATR Based":
        av = float(df["ATR"].iloc[idx]) if "ATR" in df.columns and not pd.isna(df["ATR"].iloc[idx]) else pts
        return entry + sgn*av
    if t=="Trailing SL":         return entry + sgn*pts
    if t=="Reverse EMA Crossover": return entry + sgn*pts
    if t=="Risk Reward Based":
        tp = cfg.get("custom_tgt_pts",20.0)
        return entry + sgn*(tp / max(cfg.get("rr_ratio",2.0),0.01))
    return entry + sgn*pts

def _compute_tgt(entry, sig, df, idx, cfg, sl):
    t   = cfg["tgt_type"]
    pts = cfg.get("custom_tgt_pts",20.0)
    sgn = 1.0 if sig=="BUY" else -1.0
    if t in ("Custom Points","Trailing Target","EMA Crossover"):
        return entry + sgn*pts
    if t=="Risk Reward Based":
        return entry + sgn*abs(entry-sl)*cfg.get("rr_ratio",2.0)
    return entry + sgn*pts

def _update_trail(pos, bar_close, cfg):
    trail = cfg.get("custom_sl_pts",10.0)
    sl    = pos["sl_price"]
    if pos["signal"]=="BUY":  return max(sl, bar_close-trail)
    else:                      return min(sl, bar_close+trail)

def _rev_ema(df, idx, sig):
    if "EMA_Fast" not in df.columns or idx<1: return False
    fast = df["EMA_Fast"].values.astype(float)
    slow = df["EMA_Slow"].values.astype(float)
    return _cross(fast,slow,idx, sig=="SELL")   # reverse of current signal


# ─── DHAN BROKER ─────────────────────────────────────────────────────────────
try:
    from dhanhq import dhanhq as DhanHQ
    DHAN_OK = True
except ImportError:
    DHAN_OK = False; DhanHQ = None

def _dhan_client(cid, tok):
    if not DHAN_OK: return None
    try: return DhanHQ(cid, tok)
    except: return None

def register_ip(cid, tok):
    out = {}
    try:
        out["public_ip"] = requests.get("https://api.ipify.org",timeout=5).text.strip()
    except: out["public_ip"]="unavailable"
    try:
        r = requests.post("https://api.dhan.co/edis/tpin",
            headers={"access-token":tok,"client-id":cid,"Content-Type":"application/json"},
            timeout=10)
        out["status"]=r.status_code; out["body"]=r.text[:300]
    except Exception as e: out["err"]=str(e)
    return out

def _equity_order(dhan, cfg, sig, ltp):
    if dhan is None: return {"error":"Dhan not init"}
    try:
        tx  = "BUY" if sig=="BUY" else "SELL"
        prd = "INTRADAY" if cfg.get("dhan_product","Intraday")=="Intraday" else "CNC"
        ot  = "MARKET" if "Market" in cfg.get("dhan_entry_order","Limit Order") else "LIMIT"
        ex  = "NSE" if cfg.get("dhan_exchange","NSE")=="NSE" else "BSE"
        px  = round(ltp,2) if ot=="LIMIT" else 0
        return dhan.place_order(security_id=str(cfg.get("dhan_security_id","1594")),
            exchange_segment=ex, transaction_type=tx,
            quantity=int(cfg.get("dhan_qty",1)), order_type=ot,
            product_type=prd, price=px)
    except Exception as e: return {"error":str(e)}

def _option_order(dhan, cfg, sig, ltp):
    """BUY CE when algo says BUY; BUY PE when algo says SELL (buyer only)."""
    if dhan is None: return {"error":"Dhan not init"}
    try:
        sid = str(cfg.get("dhan_ce_id","")) if sig=="BUY" else str(cfg.get("dhan_pe_id",""))
        ot  = "MARKET" if "Market" in cfg.get("dhan_opt_entry_order","Market Order") else "LIMIT"
        px  = round(ltp,2) if ot=="LIMIT" else 0
        return dhan.place_order(
            transactionType="BUY",
            exchangeSegment=cfg.get("dhan_opt_exchange","NSE_FNO"),
            productType="INTRADAY", orderType=ot, validity="DAY",
            securityId=sid, quantity=int(cfg.get("dhan_opt_qty",65)),
            price=px, triggerPrice=0)
    except Exception as e: return {"error":str(e)}

def _exit_order(dhan, cfg, pos, ltp):
    if dhan is None: return {"error":"Dhan not init"}
    sig = pos.get("signal","BUY")
    if cfg.get("dhan_options_enabled",False):
        try:
            sid = str(cfg.get("dhan_ce_id","")) if sig=="BUY" else str(cfg.get("dhan_pe_id",""))
            ot  = "MARKET" if "Market" in cfg.get("dhan_opt_exit_order","Market Order") else "LIMIT"
            px  = round(ltp,2) if ot=="LIMIT" else 0
            return dhan.place_order(
                transactionType="SELL",
                exchangeSegment=cfg.get("dhan_opt_exchange","NSE_FNO"),
                productType="INTRADAY", orderType=ot, validity="DAY",
                securityId=sid, quantity=int(cfg.get("dhan_opt_qty",65)),
                price=px, triggerPrice=0)
        except Exception as e: return {"error":str(e)}
    else:
        c2 = {**cfg,"dhan_entry_order":cfg.get("dhan_exit_order","Market Order")}
        return _equity_order(dhan, c2, "SELL" if sig=="BUY" else "BUY", ltp)


# ─── BACKTESTING ENGINE ───────────────────────────────────────────────────────
def run_backtest(df_raw, cfg, bt_start_ts):
    """
    Conservative SL-first backtest engine.
    BUY  : check candle LOW  vs SL first → then candle HIGH vs Target.
    SELL : check candle HIGH vs SL first → then candle LOW  vs Target.
    Violation = both SL & Target hit on same candle (SL wins).
    EMA Crossover signal on bar N → entry on bar N+1 OPEN.
    Simple Buy/Sell → immediate entry on bar close.
    No cooldown in backtesting.
    """
    if df_raw.empty: return {"trades":[],"violations":[],"stats":{},"df":df_raw}
    df      = build_indicators(df_raw, cfg)
    strat   = cfg["strategy"]
    trades  = []; violations = []
    pos     = None  # open trade
    pending = None  # EMA signal waiting for N+1 open

    for i in range(1, len(df)):
        bar  = df.iloc[i]
        ts   = df.index[i]

        # ── Skip warmup bars — stay BEFORE pending-entry so warmup
        #    signals don't open positions with wrong prices.
        # NOTE: a signal on the very last warmup bar IS valid — its
        #       pending is preserved and fires on the first real bar below.
        if ts < bt_start_ts:
            continue

        # ── Enter pending EMA trade at N+1 open (within real window) ────
        # This fires on the bar immediately after the crossover bar.
        # Because we skip above, the earliest this can fire is the first
        # bar of the actual requested period — correct and consistent.
        if pending and pos is None:
            ep   = float(bar["Open"])
            sig  = pending["signal"]
            sl   = _compute_sl(ep, sig, df, i, cfg)
            tgt  = _compute_tgt(ep, sig, df, i, cfg, sl)
            pos  = {"signal":sig,"entry_time":ts,"entry_price":ep,
                    "sl_price":sl,"tgt_price":tgt,
                    "init_sl":sl,"init_tgt":tgt,"reason_in":pending["reason"]}
            pending = None

        h = float(bar["High"]); l = float(bar["Low"])

        # ── Manage open position ──────────────────────────────────────
        if pos:
            sig  = pos["signal"]
            sl   = pos["sl_price"]
            tgt  = pos["tgt_price"]
            ex_p = ex_r = None; viol = False

            if sig == "BUY":
                sl_hit  = l <= sl
                tgt_hit = h >= tgt and cfg["tgt_type"] != "Trailing Target"
                if sl_hit and tgt_hit: viol=True; ex_p,ex_r=sl,"Stop Loss (Violation ⚠)"
                elif sl_hit:           ex_p,ex_r=sl,"Stop Loss"
                elif tgt_hit:          ex_p,ex_r=tgt,"Target"
            else:  # SELL
                sl_hit  = h >= sl
                tgt_hit = l <= tgt and cfg["tgt_type"] != "Trailing Target"
                if sl_hit and tgt_hit: viol=True; ex_p,ex_r=sl,"Stop Loss (Violation ⚠)"
                elif sl_hit:           ex_p,ex_r=sl,"Stop Loss"
                elif tgt_hit:          ex_p,ex_r=tgt,"Target"

            # Trailing SL
            if cfg["sl_type"]=="Trailing SL" and ex_p is None:
                pos["sl_price"] = _update_trail(pos, float(bar["Close"]), cfg)

            # Reverse EMA exit
            if ex_p is None:
                # EMA Crossover strategy: ALWAYS exit on reverse cross regardless of SL/Target type.
                # This makes every crossover produce a trade (matches TradingView behaviour).
                # SL / Target still fire first if hit before the reverse cross arrives.
                if cfg["strategy"] == "EMA Crossover" and _rev_ema(df, i, sig):
                    ex_p, ex_r = float(bar["Close"]), "EMA Reverse Crossover"
                elif cfg["sl_type"]=="Reverse EMA Crossover" and _rev_ema(df,i,sig):
                    ex_p,ex_r=float(bar["Close"]),"Reverse EMA Crossover"
                elif cfg["tgt_type"]=="EMA Crossover" and _rev_ema(df,i,sig):
                    ex_p,ex_r=float(bar["Close"]),"EMA Target (Reverse Cross)"

            if ex_p is not None:
                raw = ex_p - pos["entry_price"]
                pnl = raw*cfg["qty"] if sig=="BUY" else -raw*cfg["qty"]
                t = {"Type":sig,
                     "Entry Time":pos["entry_time"],"Entry Price":round(pos["entry_price"],2),
                     "Exit Time":ts,"Exit Price":round(ex_p,2),
                     "SL":round(pos["init_sl"],2),"Target":round(pos["init_tgt"],2),
                     "Candle High":round(h,2),"Candle Low":round(l,2),
                     "Entry Reason":pos["reason_in"],"Exit Reason":ex_r,
                     "PnL":round(pnl,2),"Result":"✅ Win" if pnl>0 else "❌ Loss",
                     "Violation":viol,"Qty":cfg["qty"]}
                trades.append(t)
                if viol: violations.append(t)
                pos = None

        # ── Look for new entry ────────────────────────────────────────
        if pos is None and pending is None:
            sig = get_signal(df, i, cfg)
            if sig:
                if strat in ("Simple Buy","Simple Sell"):
                    ep  = float(bar["Close"])
                    sl  = _compute_sl(ep, sig, df, i, cfg)
                    tgt = _compute_tgt(ep, sig, df, i, cfg, sl)
                    pos = {"signal":sig,"entry_time":ts,"entry_price":ep,
                           "sl_price":sl,"tgt_price":tgt,
                           "init_sl":sl,"init_tgt":tgt,"reason_in":strat}
                elif strat=="EMA Crossover" and i < len(df)-1:
                    pending = {"signal":sig,
                               "reason":f"EMA {cfg['fast_ema']}/{cfg['slow_ema']} "
                                        f"{'Bullish' if sig=='BUY' else 'Bearish'} Crossover"}

    total = len(trades)
    if total==0: return {"trades":[],"violations":[],"stats":{"Total Trades":0},"df":df}
    wins  = sum(1 for t in trades if t["PnL"]>0)
    tpnl  = sum(t["PnL"] for t in trades)
    pnls  = [t["PnL"] for t in trades]
    stats = {"Total Trades":total,"Wins":wins,"Losses":total-wins,
             "Accuracy (%)":round(wins/total*100,1),"Total PnL":round(tpnl,2),
             "Avg PnL":round(tpnl/total,2),"Best Trade":round(max(pnls),2),
             "Worst Trade":round(min(pnls),2),"Violations":len(violations)}
    return {"trades":trades,"violations":violations,"stats":stats,"df":df}


# ─── CHART LAYOUT TEMPLATE ──────────────────────────────────────────────────
_LAY = dict(
    template="plotly_dark",
    paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
    font=dict(color="#ccd6f6", size=11),
    margin=dict(l=8, r=8, t=28, b=8),
    legend=dict(orientation="h", y=1.04, x=0, bgcolor="rgba(0,0,0,0)"),
    xaxis_rangeslider_visible=False,
)

# ─── LIVE TRADING THREAD ─────────────────────────────────────────────────────
def _log(msg):
    ts = datetime.datetime.now(IST).strftime("%H:%M:%S")
    with _LOCK:
        _LIVE["log"].append(f"[{ts}] {msg}")
        if len(_LIVE["log"])>200: _LIVE["log"]=_LIVE["log"][-200:]


def _base(df, cfg, h=640):
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,
                        row_heights=[0.76,0.24],vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(
        x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],
        name="Price",
        increasing_line_color=C["cup"],decreasing_line_color=C["cdn"],
        increasing_fillcolor=C["cup"],decreasing_fillcolor=C["cdn"]),row=1,col=1)
    if "EMA_Fast" in df.columns:
        last_ef = df["EMA_Fast"].dropna().iloc[-1] if not df["EMA_Fast"].dropna().empty else None
        ef_label = f"EMA {cfg['fast_ema']}  {last_ef:,.2f}" if last_ef is not None else f"EMA {cfg['fast_ema']}"
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_Fast"],
            name=ef_label,
            line=dict(color=C["ef"], width=1.7), connectgaps=False,
            hovertemplate="EMA " + str(cfg["fast_ema"]) + ": %{y:,.2f}<extra></extra>"),
            row=1, col=1)
        if last_ef is not None:
            fig.add_annotation(
                x=df.index[-1], y=float(last_ef),
                text=f" {last_ef:,.2f}",
                font=dict(color=C["ef"], size=10),
                showarrow=False, xanchor="left",
                bgcolor="rgba(13,17,23,0.7)",
                row=1, col=1)
    if "EMA_Slow" in df.columns:
        last_es = df["EMA_Slow"].dropna().iloc[-1] if not df["EMA_Slow"].dropna().empty else None
        es_label = f"EMA {cfg['slow_ema']}  {last_es:,.2f}" if last_es is not None else f"EMA {cfg['slow_ema']}"
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_Slow"],
            name=es_label,
            line=dict(color=C["es"], width=1.7), connectgaps=False,
            hovertemplate="EMA " + str(cfg["slow_ema"]) + ": %{y:,.2f}<extra></extra>"),
            row=1, col=1)
        if last_es is not None:
            fig.add_annotation(
                x=df.index[-1], y=float(last_es),
                text=f" {last_es:,.2f}",
                font=dict(color=C["es"], size=10),
                showarrow=False, xanchor="left",
                bgcolor="rgba(13,17,23,0.7)",
                row=1, col=1)
    if "Volume" in df.columns:
        VOL_UP = "rgba(0,212,170,0.45)"
        VOL_DN = "rgba(255,75,92,0.45)"
        is_up  = (df["Close"].values >= df["Open"].values)   # vectorised, no per-row cast
        vc     = [VOL_UP if u else VOL_DN for u in is_up]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Vol",
            marker_color=vc, showlegend=False), row=2, col=1)
    fig.update_layout(**_LAY,height=h,
        yaxis=dict(showgrid=True,gridcolor=C["gr"]),
        yaxis2=dict(showgrid=True,gridcolor=C["gr"]),
        xaxis2=dict(showgrid=True,gridcolor=C["gr"]))
    return fig

def bt_chart(df, trades, cfg):
    fig = _base(df, cfg, 700)
    for t in trades:
        ib = t["Type"]=="BUY"
        ce = C["bm"] if ib else C["sm"]
        cx = C["cup"] if t["PnL"]>0 else C["cdn"]
        sym= "triangle-up" if ib else "triangle-down"
        fig.add_trace(go.Scatter(
            x=[t["Entry Time"]],y=[t["Entry Price"]],mode="markers+text",
            marker=dict(symbol=sym,size=13,color=ce,line=dict(width=1,color="#fff")),
            text=[t["Type"]],textposition="bottom center" if ib else "top center",
            textfont=dict(color=ce,size=8),showlegend=False,
            hovertemplate=f"<b>{t['Type']} Entry</b><br>@ {t['Entry Price']}<br>"
                          f"SL:{t['SL']} | Tgt:{t['Target']}<br>{t['Entry Reason']}"),row=1,col=1)
        for y,col in [(t["SL"],C["sl"]),(t["Target"],C["tg"])]:
            fig.add_shape(type="line",x0=t["Entry Time"],x1=t["Exit Time"],
                y0=y,y1=y,line=dict(color=col,width=1,dash="dot"),row=1,col=1)
        fig.add_trace(go.Scatter(
            x=[t["Exit Time"]],y=[t["Exit Price"]],mode="markers",
            marker=dict(symbol="x",size=10,color=cx,line=dict(width=2,color=cx)),
            showlegend=False,
            hovertemplate=f"<b>Exit</b><br>@ {t['Exit Price']}<br>"
                          f"PnL:{t['PnL']:+.2f}<br>{t['Exit Reason']}"),row=1,col=1)
    return fig

def live_chart(df, pos, cfg):
    fig = _base(df, cfg, 560)
    if pos:
        sig=pos["signal"]; ep=pos["entry_price"]
        sl=pos["sl_price"]; tgt=pos["tgt_price"]
        ce=C["bm"] if sig=="BUY" else C["sm"]
        sym="triangle-up" if sig=="BUY" else "triangle-down"
        x0=pos["entry_time"]; x1=df.index[-1]
        fig.add_trace(go.Scatter(x=[x0],y=[ep],mode="markers+text",
            marker=dict(symbol=sym,size=15,color=ce,line=dict(width=1.5,color="#fff")),
            text=[sig],textposition="bottom center" if sig=="BUY" else "top center",
            textfont=dict(color=ce,size=9),showlegend=False,
            hovertemplate=f"Entry:{ep} | SL:{sl} | Tgt:{tgt}"),row=1,col=1)
        for y,col,lbl in [(sl,C["sl"],f"SL {sl:.2f}"),(tgt,C["tg"],f"TGT {tgt:.2f}")]:
            fig.add_shape(type="line",x0=x0,x1=x1,y0=y,y1=y,
                line=dict(color=col,width=1.5,dash="dashdot"),row=1,col=1)
            fig.add_annotation(x=x1,y=y,text=f" {lbl}",
                font=dict(color=col,size=10),showarrow=False,xanchor="left",row=1,col=1)
    return fig


# ─── UI HELPERS ──────────────────────────────────────────────────────────────
def ltp_card(ticker, ph=None):
    ltp,prev,chg,pct = get_ltp(ticker)
    if ltp is None:
        html='<div class="ltp-wrap"><span style="color:#8892b0">LTP unavailable</span></div>'
    else:
        arrow="▲" if chg>=0 else "▼"; cls="ltp-up" if chg>=0 else "ltp-dn"
        html=f"""<div class="ltp-wrap">
  <div><div class="ltp-sym">{ticker}</div>
       <div class="ltp-px">{ltp:,.2f}</div></div>
  <div class="{cls}" style="margin-left:10px;font-size:1rem">
    {arrow} {abs(chg):,.2f} &nbsp;({abs(pct):.2f}%)</div>
  <div class="ltp-meta">Prev Close <b>{prev:,.2f}</b><br>
    <span style="font-size:.7rem">{datetime.datetime.now(IST).strftime('%H:%M:%S IST')}</span>
  </div></div>"""
    (ph or st).markdown(html,unsafe_allow_html=True)
    return ltp

def _mc(v,l,cls=""):
    return f'<div class="mc {cls}"><div class="mc-v">{v}</div><div class="mc-l">{l}</div></div>'

def stats_bar(stats):
    if not stats or stats.get("Total Trades",0)==0:
        st.info("No trades for this configuration."); return
    a=stats.get("Accuracy (%)",0); p=stats.get("Total PnL",0); v=stats.get("Violations",0)
    html=(_mc(stats["Total Trades"],"Total Trades","c-b")
         +_mc(f"{stats['Wins']} / {stats['Losses']}","Wins / Losses","c-g")
         +_mc(f"{a}%","Accuracy","c-gold" if a>=50 else "c-r")
         +_mc(f"₹{p:+,.2f}","Total PnL","c-g" if p>=0 else "c-r")
         +_mc(f"₹{stats.get('Avg PnL',0):+,.2f}","Avg PnL",
              "c-g" if stats.get("Avg PnL",0)>=0 else "c-r")
         +_mc(f"₹{stats.get('Best Trade',0):+,.2f}","Best Trade","c-g")
         +_mc(f"₹{stats.get('Worst Trade',0):+,.2f}","Worst Trade","c-r")
         +_mc(v,"Violations ⚠","c-r" if v>0 else ""))
    st.markdown(f'<div class="mrow">{html}</div>',unsafe_allow_html=True)

def cfg_cards(cfg):
    items=[("Ticker",cfg.get("ticker_name",cfg.get("ticker","—"))),
           ("Timeframe",cfg.get("timeframe","—")),("Period",cfg.get("period","—")),
           ("Strategy",cfg.get("strategy","—")),
           ("Fast EMA",cfg.get("fast_ema","—")),("Slow EMA",cfg.get("slow_ema","—")),
           ("SL Type",cfg.get("sl_type","—")),("Target",cfg.get("tgt_type","—")),
           ("SL Pts",cfg.get("custom_sl_pts","—")),("Tgt Pts",cfg.get("custom_tgt_pts","—")),
           ("Qty",cfg.get("qty","—")),("R:R",cfg.get("rr_ratio","—")),
           ("Crossover",cfg.get("crossover_type","—")),
           ("Dhan","✅ ON" if cfg.get("dhan_enabled") else "❌ OFF")]
    html="".join(f'<div class="cfg-item"><div class="cfg-k">{k}</div>'
                 f'<div class="cfg-v">{v}</div></div>' for k,v in items)
    st.markdown(f'<div class="cfg-grid">{html}</div>',unsafe_allow_html=True)

def ema_pill_row(df, cfg):
    """Render a compact EMA value pill strip. Call after stats_bar or cfg_cards."""
    if df is None or df.empty:
        return
    strat = cfg.get("strategy","")
    # Only meaningful for EMA Crossover; for Simple Buy/Sell still show for info
    fe = cfg.get("fast_ema", 9)
    se = cfg.get("slow_ema", 15)
    ef_val = es_val = None
    if "EMA_Fast" in df.columns:
        s = df["EMA_Fast"].dropna()
        if not s.empty: ef_val = float(s.iloc[-1])
    if "EMA_Slow" in df.columns:
        s = df["EMA_Slow"].dropna()
        if not s.empty: es_val = float(s.iloc[-1])
    if ef_val is None and es_val is None:
        return
    # Determine crossover state for colour hint
    cross_lbl = ""
    cross_col = "#8892b0"
    if ef_val is not None and es_val is not None:
        if ef_val > es_val:
            cross_lbl = "▲ Bullish"; cross_col = "#00d4aa"
        elif ef_val < es_val:
            cross_lbl = "▼ Bearish"; cross_col = "#ff4b5c"
        else:
            cross_lbl = "◆ Crossing"; cross_col = "#ffd166"
    parts = []
    if ef_val is not None:
        parts.append(
            f'<span style="background:#1a1f35;border:1px solid {C["ef"]}33;'
            f'border-radius:6px;padding:4px 12px;font-size:.82rem;">'
            f'<span style="color:#8892b0;font-size:.7rem;margin-right:4px;">EMA {fe}</span>'
            f'<span style="color:{C["ef"]};font-weight:700;">{ef_val:,.2f}</span>'
            f'</span>')
    if es_val is not None:
        parts.append(
            f'<span style="background:#1a1f35;border:1px solid {C["es"]}33;'
            f'border-radius:6px;padding:4px 12px;font-size:.82rem;">'
            f'<span style="color:#8892b0;font-size:.7rem;margin-right:4px;">EMA {se}</span>'
            f'<span style="color:{C["es"]};font-weight:700;">{es_val:,.2f}</span>'
            f'</span>')
    if cross_lbl:
        parts.append(
            f'<span style="background:#1a1f35;border:1px solid {cross_col}44;'
            f'border-radius:6px;padding:4px 12px;font-size:.82rem;">'
            f'<span style="color:{cross_col};font-weight:700;">{cross_lbl}</span>'
            f'</span>')
    html = (
        '<div style="display:flex;flex-wrap:wrap;gap:8px;'
        'align-items:center;margin:6px 0 10px 0;">'
        '<span style="color:#8892b0;font-size:.75rem;font-weight:600;'
        'letter-spacing:.5px;text-transform:uppercase;">EMA Values</span>'
        + "".join(parts) + "</div>")
    st.markdown(html, unsafe_allow_html=True)

def pos_card(pos, ltp):
    if not pos: return
    sig=pos["signal"]; ep=pos["entry_price"]; sl=pos["sl_price"]; tgt=pos["tgt_price"]
    et=pos["entry_time"]
    unr="—"
    if ltp:
        raw=(ltp-ep) if sig=="BUY" else (ep-ltp)
        col="color:#00d4aa" if raw>=0 else "color:#ff4b5c"
        unr=f'<span style="{col};font-weight:700">₹{raw:+.2f}</span>'
    cls="pos-buy" if sig=="BUY" else "pos-sell"
    clr="#00d4aa" if sig=="BUY" else "#ff4b5c"
    ts_str = et.strftime("%H:%M:%S IST") if hasattr(et,"strftime") else str(et)
    st.markdown(f"""<div class="pos-card {cls}">
<div style="color:{clr};font-weight:700;font-size:1rem">
  {'▲' if sig=='BUY' else '▼'} {sig} Position &nbsp;
  <span style="color:#8892b0;font-size:.82rem">| {ts_str}</span>
</div>
<div class="pos-detail">
  <div class="pos-f"><div class="pos-fl">Entry</div><div class="pos-fv">₹{ep:,.2f}</div></div>
  <div class="pos-f"><div class="pos-fl">LTP</div>
    <div class="pos-fv">{'₹{:,.2f}'.format(ltp) if ltp else '—'}</div></div>
  <div class="pos-f"><div class="pos-fl">Stop Loss</div>
    <div class="pos-fv" style="color:#ff4b5c">₹{sl:,.2f}</div></div>
  <div class="pos-f"><div class="pos-fl">Target</div>
    <div class="pos-fv" style="color:#00d4aa">₹{tgt:,.2f}</div></div>
  <div class="pos-f"><div class="pos-fl">Unrealised</div>
    <div class="pos-fv">{unr}</div></div>
  <div class="pos-f"><div class="pos-fl">Reason</div>
    <div class="pos-fv">{pos.get('reason_in','—')}</div></div>
</div></div>""",unsafe_allow_html=True)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
def sidebar() -> dict:
    with st.sidebar:
        st.markdown("""<div style="text-align:center;padding:8px 0">
<span style="font-size:2rem">📈</span><br>
<span style="font-size:1.1rem;font-weight:900;
  background:linear-gradient(90deg,#64ffda,#00b4d8);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent">
  Smart Investing</span><br>
<span style="color:#8892b0;font-size:.7rem">Advanced Algo Trading Platform</span>
</div>""",unsafe_allow_html=True)
        st.divider()

        # Asset & TF
        st.markdown('<div class="sbs">📊 Asset & Timeframe</div>',unsafe_allow_html=True)
        tn  = st.selectbox("Ticker",list(TICKER_MAP.keys()),index=0)
        if tn=="Custom":
            ct = st.text_input("Yahoo Symbol","RELIANCE.NS",placeholder="e.g. TCS.NS, AAPL")
            tk = ct.strip() or "^NSEI"
        else:
            tk = TICKER_MAP[tn]
        tf     = st.selectbox("Timeframe",list(TF_PERIODS.keys()),index=1)
        period = st.selectbox("Period",TF_PERIODS[tf],index=0)

        # Strategy
        st.markdown('<div class="sbs">🎯 Strategy</div>',unsafe_allow_html=True)
        strat = st.selectbox("Strategy",STRATEGIES,index=0)
        fe=9; se=15; ct2="Simple Crossover"; ccs=10.0; uma=False; ma=0.0

        if strat=="EMA Crossover":
            c1,c2=st.columns(2)
            fe=c1.number_input("Fast EMA",2,500,9)
            se=c2.number_input("Slow EMA",2,500,15)
            if fe>=se: st.warning("⚠️ Fast EMA must be < Slow EMA")
            use_adv = st.checkbox(
                "🔬 Advanced Crossover Filters", value=False,
                help="Enable min-angle and candle-size filters on top of the crossover signal")
            if use_adv:
                ct2 = st.selectbox("Crossover Filter", CROSS_TYPES, index=0)
                if ct2 == "Custom Candle Size":
                    ccs = st.number_input("Min Candle Body", 0.01, 1e6, 10.0, 1.0)
                elif ct2 == "ATR Based Candle Size":
                    st.caption("ℹ️ Candle body ≥ ATR(14) required")
                uma = st.checkbox("Minimum Angle Filter", value=False,
                                  help="Ignore weak crossovers below this EMA angle")
                if uma:
                    ma = st.slider("Min Angle (°)", 0.0, 89.0, 0.0, 0.5)

        # Stop Loss
        st.markdown('<div class="sbs">🛡️ Stop Loss</div>',unsafe_allow_html=True)
        sl_t=st.selectbox("SL Type",SL_TYPES,index=0)
        sl_pts=10.0; rr=2.0
        if sl_t=="Custom Points":
            sl_pts=st.number_input("SL Points",0.01,1e6,10.0,1.0)
        elif sl_t=="Trailing SL":
            sl_pts=st.number_input("Trail Distance (pts)",0.01,1e6,10.0,1.0)
        elif sl_t=="ATR Based":
            st.caption("SL = Entry ± ATR(14)")
        elif sl_t=="Reverse EMA Crossover":
            sl_pts=st.number_input("Fallback SL pts",0.01,1e6,10.0,1.0)
            st.caption("Exits on reverse EMA cross. Fallback for non-EMA strategies.")
        elif sl_t=="Risk Reward Based":
            st.caption("SL = Target ÷ R:R")
            rr=st.number_input("R:R Ratio",0.1,100.0,2.0,0.5,key="rr_sl")

        # Target
        st.markdown('<div class="sbs">🎯 Target</div>',unsafe_allow_html=True)
        tgt_t=st.selectbox("Target Type",TGT_TYPES,index=0)
        tgt_pts=20.0
        if tgt_t=="Custom Points":
            tgt_pts=st.number_input("Target Points",0.01,1e6,20.0,1.0)
        elif tgt_t=="Trailing Target":
            tgt_pts=st.number_input("Trailing Distance (display)",0.01,1e6,20.0,1.0)
            st.caption("ℹ️ Displayed on chart — does NOT trigger exit.")
        elif tgt_t=="EMA Crossover":
            tgt_pts=st.number_input("Display Target pts",0.01,1e6,20.0,1.0)
            st.caption("Exits on reverse EMA crossover.")
        elif tgt_t=="Risk Reward Based":
            tgt_pts=st.number_input("Base Target pts",0.01,1e6,20.0,1.0)
            if sl_t!="Risk Reward Based":
                rr=st.number_input("R:R Ratio",0.1,100.0,2.0,0.5,key="rr_tgt")

        # Trade settings
        st.markdown('<div class="sbs">⚙️ Trade Settings</div>',unsafe_allow_html=True)
        qty=st.number_input("Quantity",min_value=1,value=1,step=1)
        use_cd=st.checkbox("Cooldown Between Trades (Live only)",value=True,
                           help="Not applied in backtesting")
        cd_sec=5
        if use_cd: cd_sec=st.number_input("Cooldown (sec)",1,600,5,1)
        prev_ol=st.checkbox("Prevent Overlapping Trades",value=True,
                            help="One position must fully close before the next opens")

        # Dhan
        st.divider()
        st.markdown('<div class="sbs">🏦 Dhan Broker</div>',unsafe_allow_html=True)
        dhan_en=st.checkbox("Enable Dhan Broker",value=False)
        dc: dict = {"dhan_enabled":dhan_en}

        if dhan_en:
            if not DHAN_OK:
                st.error("dhanhq not installed. Run: pip install dhanhq")
            cid = st.text_input("Client ID",  type="password")
            tok = st.text_input("Access Token",type="password")
            dc.update({"dhan_client_id":cid,"dhan_access_token":tok})

            if st.button("🔐 Register IP (SEBI Mandate)",use_container_width=True):
                with st.spinner("Detecting IP…"):
                    res=register_ip(cid,tok)
                st.info(f"**Public IP:** `{res.get('public_ip','N/A')}`\n\n"
                        f"Also whitelist in [Dhan Developer Portal](https://developer.dhan.co/).\n\n"
                        f"API: `{res.get('status','N/A')}` — {res.get('body','')[:100]}")

            opt_en=st.checkbox("Options Trading",value=False)
            dc["dhan_options_enabled"]=opt_en

            if not opt_en:
                prd=st.selectbox("Product",["Intraday","Delivery"],index=0)
                exch=st.selectbox("Exchange",["NSE","BSE"],index=0)
                sid=st.text_input("Security ID","1594")
                dq=st.number_input("Dhan Qty",min_value=1,value=1,step=1)
                eno=st.selectbox("Entry Order",["Limit Order","Market Order"],index=0)
                exo=st.selectbox("Exit Order", ["Market Order","Limit Order"],index=0)
                dc.update({"dhan_product":prd,"dhan_exchange":exch,
                           "dhan_security_id":sid,"dhan_qty":dq,
                           "dhan_entry_order":eno,"dhan_exit_order":exo})
            else:
                oe=st.selectbox("Options Exchange",["NSE_FNO","BSE_FNO"],index=0)
                ce=st.text_input("CE Security ID","")
                pe=st.text_input("PE Security ID","")
                oq=st.number_input("Qty/Lots",min_value=1,value=65,step=1)
                oeno=st.selectbox("Entry Order",["Market Order","Limit Order"],index=0)
                oexo=st.selectbox("Exit Order", ["Market Order","Limit Order"],index=0)
                dc.update({"dhan_opt_exchange":oe,"dhan_ce_id":ce,"dhan_pe_id":pe,
                           "dhan_opt_qty":oq,"dhan_opt_entry_order":oeno,
                           "dhan_opt_exit_order":oexo})

        return {"ticker":tk,"ticker_name":tn,"timeframe":tf,"period":period,
                "strategy":strat,"fast_ema":fe,"slow_ema":se,
                "crossover_type":ct2,"custom_candle_size":ccs,
                "use_min_angle":uma,"min_angle":ma,
                "sl_type":sl_t,"tgt_type":tgt_t,
                "custom_sl_pts":sl_pts,"custom_tgt_pts":tgt_pts,
                "rr_ratio":rr,"qty":qty,"use_cooldown":use_cd,
                "cooldown_secs":cd_sec,"prevent_overlap":prev_ol,**dc}


# ─── BACKTEST TAB ─────────────────────────────────────────────────────────────
def tab_backtest(cfg):
    ph = st.empty()
    ltp_card(cfg["ticker"], ph)

    c1,c2 = st.columns([2,6])
    run = c1.button("▶️  Run Backtest",type="primary",use_container_width=True)
    c2.caption("Conservative SL-first: For BUY, candle LOW checked vs SL before HIGH vs Target. "
               "For SELL, candle HIGH vs SL before LOW vs Target. "
               "Violations = same candle hits both SL & Target (SL wins).")

    if run:
        with st.spinner("⏳ Fetching data & running backtest…"):
            df_raw = fetch_data(cfg["ticker"],cfg["timeframe"],cfg["period"],cfg["slow_ema"])
            if df_raw.empty:
                st.error("❌ No data returned. Check ticker symbol."); return
            days = PERIOD_DAYS.get(cfg["period"],30)
            try:
                bt_s = datetime.datetime.now(IST)-timedelta(days=days)
                if df_raw.index.tz:
                    bts = pd.Timestamp(bt_s).tz_convert(df_raw.index.tz)
                else:
                    bts = pd.Timestamp(bt_s.replace(tzinfo=None))
            except Exception:
                bts = df_raw.index[0]
            res = run_backtest(df_raw, cfg, bts)
            st.session_state.bt_results    = res
            st.session_state.bt_cfg        = cfg.copy()
            st.session_state["bt_chart_stale"] = True   # force chart rebuild

    res = st.session_state.get("bt_results")
    if not res:
        st.markdown("""<div style="text-align:center;padding:55px 0;color:#8892b0">
<div style="font-size:3rem">📊</div>
<div style="font-size:1rem;margin-top:8px">
  Configure parameters in the sidebar and click <b>Run Backtest</b></div>
</div>""",unsafe_allow_html=True)
        return

    stats  = res["stats"]
    trades = res["trades"]
    viols  = res["violations"]
    df_ind = res["df"]

    # Stats bar
    st.markdown("#### 📊 Backtest Summary")
    stats_bar(stats)
    ema_pill_row(df_ind, cfg)   # EMA values right below summary cards

    if not trades:
        st.info("No trades generated. Adjust strategy or timeframe."); return

    # Chart — built once per backtest run, cached in session_state
    st.markdown("#### 📈 Trade Chart")
    if "bt_chart_fig" not in st.session_state or st.session_state.get("bt_chart_stale", True):
        disp_df = df_ind.iloc[-500:] if len(df_ind) > 500 else df_ind
        st.session_state["bt_chart_fig"]   = bt_chart(disp_df, trades, cfg)
        st.session_state["bt_chart_stale"] = False
    st.plotly_chart(st.session_state["bt_chart_fig"],
                    use_container_width=True, config={"scrollZoom": True})

    # Trade Table
    st.markdown("#### 📋 Trade Log")
    df_trades = pd.DataFrame(trades)
    if not df_trades.empty:
        display_cols = ["Type","Entry Time","Entry Price","Exit Time","Exit Price",
                        "SL","Target","Candle High","Candle Low",
                        "Entry Reason","Exit Reason","PnL","Result","Qty","Violation"]
        df_show = df_trades[[c for c in display_cols if c in df_trades.columns]]
        # Fast styling: only apply Styler when ≤200 rows to keep renders snappy
        if len(df_show) <= 200:
            def _color(row):
                if row.get("PnL", 0) > 0:
                    return ["background-color:#0d2620;color:#00d4aa"] * len(row)
                elif row.get("PnL", 0) < 0:
                    return ["background-color:#2a0f0f;color:#ff4b5c"] * len(row)
                return [""] * len(row)
            st.dataframe(df_show.style.apply(_color, axis=1),
                         use_container_width=True, height=400)
        else:
            st.dataframe(df_show, use_container_width=True, height=400)

    # Violations section
    st.markdown("#### ⚠️ Violation Analysis")
    vcount = len(viols)
    if vcount==0:
        st.success("✅ No violations — SL and Target were never hit on the same candle.")
    else:
        st.markdown(f"""<div class="viol-box">
<b style="color:#ff4b5c">⚠️ {vcount} violation(s) detected</b><br>
<span style="color:#ccd6f6;font-size:.85rem">
These are trades where both SL and Target were hit on the same candle.
In live trading these would be resolved tick-by-tick (SL hit first with conservative approach).
This is the fundamental gap between backtesting and live trading.</span>
</div>""",unsafe_allow_html=True)
        df_v = pd.DataFrame(viols)
        if not df_v.empty:
            st.dataframe(df_v[[c for c in ["Type","Entry Time","Entry Price",
                "Exit Price","SL","Target","Candle High","Candle Low","PnL","Exit Reason"]
                if c in df_v.columns]],
                use_container_width=True)

    # Equity curve
    if len(trades)>1:
        st.markdown("#### 💰 Equity Curve")
        pnls = [t["PnL"] for t in trades]
        cum  = pd.Series(pnls).cumsum().tolist()
        times= [str(t["Exit Time"]) for t in trades]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=times,y=cum,mode="lines",
            fill="tozeroy",
            line=dict(color=C["cup"] if cum[-1]>=0 else C["cdn"],width=2),
            fillcolor="rgba(0,212,170,0.1)" if cum[-1]>=0 else "rgba(255,75,92,0.1)"))
        fig2.update_layout(**_LAY,height=260,
            yaxis=dict(title="Cumulative PnL ₹",showgrid=True,gridcolor=C["gr"]),
            xaxis=dict(showgrid=True,gridcolor=C["gr"]))
        st.plotly_chart(fig2,use_container_width=True)


# ─── LIVE TRADING THREAD ─────────────────────────────────────────────────────
def _log(msg: str):
    ts = datetime.datetime.now(IST).strftime("%H:%M:%S")
    with _LOCK:
        _LIVE["log"].append(f"[{ts}] {msg}")
        if len(_LIVE["log"]) > 200:
            _LIVE["log"] = _LIVE["log"][-200:]

def live_loop(stop_evt: threading.Event):
    """
    Background thread — runs every ~2 s.
    Tick : SL/Target checked against live LTP on every iteration.
    Bar  : New signal checked only when the last closed bar changes.
           EMA Crossover: signal on closed bar N → entry at forming bar open.
           Simple Buy/Sell: entry at last closed bar close.
    """
    with _LOCK:
        cfg = dict(_LIVE["config"])
    ticker   = cfg["ticker"]
    tf       = cfg["timeframe"]
    last_bt  = None
    last_fetch = 0.0
    cooldown_until = 0.0

    # Dhan init
    dhan = None
    if cfg.get("dhan_enabled") and cfg.get("dhan_client_id"):
        dhan = _dhan_client(cfg["dhan_client_id"], cfg["dhan_access_token"])
        _log("✅ Dhan connected" if dhan else "⚠️ Dhan init failed")

    _log(f"▶ Started | {ticker} | {tf} | {cfg['strategy']}")

    while not stop_evt.is_set():
        try:
            now = time.time()
            # Rate-limit: one fetch per 2 s
            if now - last_fetch < 2.0:
                time.sleep(0.2)
                continue
            last_fetch = now

            # ── Fetch OHLCV + build indicators ──────────────────────────
            df = _fetch_raw(ticker, tf, cfg["period"], cfg["slow_ema"])  # always fresh
            if df is None or df.empty:
                _log("⚠️ Empty data, retrying…")
                time.sleep(3)
                continue
            df_i = build_indicators(df, cfg)

            # ── Get LTP ─────────────────────────────────────────────────
            ltp, prev, _, _ = get_ltp(ticker)
            if ltp is None:
                ltp = float(df_i["Close"].iloc[-1])

            with _LOCK:
                _LIVE["last_ltp"]    = ltp
                _LIVE["prev_close"]  = prev
                _LIVE["df"]          = df_i
                row = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer))
                           else str(v))
                       for k, v in df_i.iloc[-1].items()}
                row["Time"] = str(df_i.index[-1])
                _LIVE["last_candle"] = row

            # ── Tick: check SL/Target against LTP (not OHLC) ────────────
            with _LOCK:
                pos = _LIVE["position"]

            if pos:
                sig = pos["signal"]
                sl  = pos["sl_price"]
                tgt = pos["tgt_price"]
                ex_p = ex_r = None

                if sig == "BUY":
                    if ltp <= sl:
                        ex_p, ex_r = ltp, "Stop Loss"
                    elif ltp >= tgt and cfg["tgt_type"] != "Trailing Target":
                        ex_p, ex_r = ltp, "Target"
                else:
                    if ltp >= sl:
                        ex_p, ex_r = ltp, "Stop Loss"
                    elif ltp <= tgt and cfg["tgt_type"] != "Trailing Target":
                        ex_p, ex_r = ltp, "Target"

                # Trailing SL update
                if cfg["sl_type"] == "Trailing SL" and ex_p is None:
                    trail = cfg.get("custom_sl_pts", 10.0)
                    with _LOCK:
                        cur = _LIVE["position"]["sl_price"]
                        new_sl = (ltp - trail) if sig == "BUY" else (ltp + trail)
                        _LIVE["position"]["sl_price"] = max(cur, new_sl) if sig == "BUY" else min(cur, new_sl)

                # Reverse EMA on last closed bar
                if ex_p is None and len(df_i) >= 2:
                    ci = len(df_i) - 2
                    # EMA Crossover strategy: ALWAYS exit on reverse cross (same logic as backtest)
                    if cfg["strategy"] == "EMA Crossover" and _rev_ema(df_i, ci, sig):
                        ex_p, ex_r = ltp, "EMA Reverse Crossover"
                    elif cfg["sl_type"] == "Reverse EMA Crossover" and _rev_ema(df_i, ci, sig):
                        ex_p, ex_r = ltp, "Reverse EMA Crossover"
                    elif cfg["tgt_type"] == "EMA Crossover" and _rev_ema(df_i, ci, sig):
                        ex_p, ex_r = ltp, "EMA Target (Reverse Cross)"

                # Close trade
                if ex_p is not None:
                    with _LOCK:
                        p2  = _LIVE["position"]
                        raw = ex_p - p2["entry_price"]
                        pnl = raw * cfg["qty"] if sig == "BUY" else -raw * cfg["qty"]
                        done = {
                            **p2,
                            "exit_time":   datetime.datetime.now(IST),
                            "exit_price":  round(ex_p, 2),
                            "exit_reason": ex_r,
                            "pnl":         round(pnl, 2),
                            "result":      "✅ Win" if pnl > 0 else "❌ Loss",
                        }
                        _LIVE["completed_trades"].append(done)
                        _LIVE["position"] = None
                    _log(f"🔴 EXIT {sig} @ {ex_p:.2f} | {ex_r} | PnL: {pnl:+.2f}")
                    if dhan and cfg.get("dhan_enabled"):
                        resp = _exit_order(dhan, cfg, p2, ex_p)
                        _log(f"📤 Dhan exit: {str(resp)[:80]}")
                    if cfg.get("use_cooldown", True):
                        cooldown_until = time.time() + cfg.get("cooldown_secs", 5)

            # ── Bar: new entry signal on closed bar ──────────────────────
            if len(df_i) >= 2:
                closed_bt = df_i.index[-2]
                if last_bt is None or closed_bt != last_bt:
                    last_bt = closed_bt
                    ci = len(df_i) - 2
                    with _LOCK:
                        has_pos = _LIVE["position"] is not None
                    if not has_pos and time.time() >= cooldown_until:
                        sig = get_signal(df_i, ci, cfg)
                        if sig:
                            if cfg["strategy"] == "EMA Crossover":
                                ep  = float(df_i["Open"].iloc[-1])
                                rsn = (f"EMA {cfg['fast_ema']}/{cfg['slow_ema']} "
                                       f"{'Bullish' if sig == 'BUY' else 'Bearish'} Cross")
                            else:
                                ep  = float(df_i["Close"].iloc[-2])
                                rsn = cfg["strategy"]
                            sl  = _compute_sl(ep, sig, df_i, ci, cfg)
                            tgt = _compute_tgt(ep, sig, df_i, ci, cfg, sl)
                            new_pos = {
                                "signal":      sig,
                                "entry_time":  datetime.datetime.now(IST),
                                "entry_price": round(ep, 2),
                                "sl_price":    round(sl, 2),
                                "tgt_price":   round(tgt, 2),
                                "init_sl":     round(sl, 2),
                                "init_tgt":    round(tgt, 2),
                                "reason_in":   rsn,
                            }
                            with _LOCK:
                                _LIVE["position"] = new_pos
                            _log(f"🟢 ENTRY {sig} @ {ep:.2f} | SL:{sl:.2f} | Tgt:{tgt:.2f}")
                            if dhan and cfg.get("dhan_enabled"):
                                fn   = _option_order if cfg.get("dhan_options_enabled") else _equity_order
                                resp = fn(dhan, cfg, sig, ep)
                                _log(f"📤 Dhan entry: {str(resp)[:80]}")

            time.sleep(2.0)

        except Exception as e:
            _log(f"⚠️ {type(e).__name__}: {e}")
            time.sleep(3)

    with _LOCK:
        _LIVE["active"] = False
    _log("⏹ Stopped")


# ─── LIVE TRADING TAB ─────────────────────────────────────────────────────────
def tab_live(cfg):
    # ── LTP header ──────────────────────────────────────────────────────────
    ltp_ph = st.empty()
    ltp_card(cfg["ticker"], ltp_ph)

    # ── Read shared state once at top of this render ─────────────────────────
    with _LOCK:
        is_active = _LIVE["active"]
        pos       = _LIVE["position"]
        ltp_now   = _LIVE["last_ltp"]
        live_cfg  = _LIVE["config"] or cfg
        logs      = list(_LIVE["log"])
        lc        = _LIVE["last_candle"]
        df_live   = _LIVE["df"]

    # ── Status ───────────────────────────────────────────────────────────────
    if is_active:
        st.markdown('<span class="badge-live">🟢 LIVE — refreshing every 3 s</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-idle">⚫ IDLE</span>', unsafe_allow_html=True)

    # ── Control buttons ──────────────────────────────────────────────────────
    bc1, bc2, bc3 = st.columns(3)
    start_btn = bc1.button("▶️  Start",      type="primary",    use_container_width=True,
                           disabled=is_active)
    stop_btn  = bc2.button("⏹  Stop",        type="secondary",  use_container_width=True,
                           disabled=not is_active)
    sq_btn    = bc3.button("⚡ Square Off",                       use_container_width=True,
                           disabled=not is_active)

    # ── START ────────────────────────────────────────────────────────────────
    if start_btn:
        with _LOCK:
            # Clean slate for new run
            _LIVE["active"]            = True
            _LIVE["config"]            = cfg.copy()
            _LIVE["position"]          = None
            _LIVE["log"]               = []
            _LIVE["last_ltp"]          = None
            _LIVE["prev_close"]        = None
            _LIVE["last_candle"]       = None
            _LIVE["df"]                = None
        # Start thread
        stop_evt = threading.Event()
        t = threading.Thread(target=live_loop, args=(stop_evt,), daemon=True)
        t.start()
        with _LOCK:
            _LIVE["_thread"]   = t
            _LIVE["_stop_evt"] = stop_evt
        st.rerun()

    # ── STOP ─────────────────────────────────────────────────────────────────
    if stop_btn:
        with _LOCK:
            ev = _LIVE.get("_stop_evt")
        if ev:
            ev.set()
        with _LOCK:
            _LIVE["active"] = False
        st.rerun()

    # ── SQUARE OFF ───────────────────────────────────────────────────────────
    if sq_btn:
        with _LOCK:
            pos2    = _LIVE["position"]
            lcfg2   = _LIVE["config"] or cfg
        if pos2:
            ltp2, _, _, _ = get_ltp(cfg["ticker"])
            ltp2 = ltp2 or pos2["entry_price"]
            sig  = pos2["signal"]
            raw  = ltp2 - pos2["entry_price"]
            pnl  = raw * lcfg2["qty"] if sig == "BUY" else -raw * lcfg2["qty"]
            with _LOCK:
                done = {**pos2,
                        "exit_time":   datetime.datetime.now(IST),
                        "exit_price":  round(ltp2, 2),
                        "exit_reason": "Manual Square Off",
                        "pnl":         round(pnl, 2),
                        "result":      "✅ Win" if pnl > 0 else "❌ Loss"}
                _LIVE["completed_trades"].append(done)
                _LIVE["position"] = None
            _log(f"⚡ SQUARE OFF {sig} @ {ltp2:.2f} | PnL: {pnl:+.2f}")
            if lcfg2.get("dhan_enabled"):
                dh = _dhan_client(lcfg2.get("dhan_client_id", ""), lcfg2.get("dhan_access_token", ""))
                if dh:
                    _exit_order(dh, lcfg2, pos2, ltp2)
            st.success(f"Position squared off @ ₹{ltp2:.2f}")
            st.rerun()
        else:
            st.warning("No open position.")

    # ── IDLE STATE ───────────────────────────────────────────────────────────
    if not is_active:
        st.markdown("""
<div style="text-align:center;padding:55px 0;color:#8892b0">
  <div style="font-size:3rem">🚀</div>
  <div style="font-size:1rem;margin-top:8px">
    Configure in the sidebar, then click <b>Start</b>.</div>
</div>""", unsafe_allow_html=True)
        return

    # ── Active: show everything ──────────────────────────────────────────────
    st.divider()

    # Config cards
    st.markdown("#### ⚙️ Active Configuration")
    cfg_cards(live_cfg)
    with _LOCK: _df_for_ema = _LIVE["df"]
    ema_pill_row(_df_for_ema, live_cfg)   # EMA values right below config cards

    # Open position
    st.markdown("#### 📌 Open Position")
    if pos:
        pos_card(pos, ltp_now)
    else:
        st.info("💤 No open position — scanning for signal…")

    # Chart
    st.markdown("#### 📈 Live Chart")
    if df_live is not None and not df_live.empty:
        disp = df_live.iloc[-300:] if len(df_live) > 300 else df_live
        st.plotly_chart(live_chart(disp, pos, live_cfg),
                        use_container_width=True, config={"scrollZoom": True})
    else:
        st.caption("⏳ Waiting for first data fetch from background thread…")

    # Last fetched candle
    st.markdown("#### 🕯️ Last Fetched Candle (yfinance data)")
    if lc:
        st.dataframe(pd.DataFrame([lc]), use_container_width=True)
    else:
        st.caption("No candle yet.")

    # Log
    st.markdown("#### 📝 Activity Log")
    log_text = "\n".join(reversed(logs[-60:]))
    st.code(log_text or "Waiting for events…", language="")

    # ── Auto-refresh: sleep 3 s then rerun ──────────────────────────────────
    # sleep(3) means the UI is locked for 3 s — acceptable for a trading dashboard.
    # Do NOT use a shorter value; Streamlit's rate limiter will kill the session.
    time.sleep(3)
    st.rerun()


# ─── TRADE HISTORY TAB ────────────────────────────────────────────────────────
def tab_history(cfg):
    ltp_ph = st.empty()
    ltp_card(cfg["ticker"], ltp_ph)

    with _LOCK:
        completed = list(_LIVE["completed_trades"])
        is_active = _LIVE["active"]

    st.markdown("#### 📚 Completed Trades")
    if is_active:
        st.markdown('<span class="badge-live">🟢 LIVE — history updates in real-time</span>',
                    unsafe_allow_html=True)

    if not completed:
        st.markdown("""<div style="text-align:center;padding:50px 0;color:#8892b0">
<div style="font-size:3rem">📂</div>
<div style="font-size:1rem;margin-top:8px">
  No completed trades yet. Run live trading or backtest to see history here.</div>
</div>""",unsafe_allow_html=True)
        return

    # Summary
    wins   = sum(1 for t in completed if t.get("pnl",0)>0)
    total  = len(completed)
    tpnl   = sum(t.get("pnl",0) for t in completed)
    acc    = round(wins/total*100,1) if total else 0

    html=(_mc(total,"Total Trades","c-b")
         +_mc(f"{wins} / {total-wins}","Wins / Losses","c-g")
         +_mc(f"{acc}%","Accuracy","c-gold" if acc>=50 else "c-r")
         +_mc(f"₹{tpnl:+,.2f}","Total PnL","c-g" if tpnl>=0 else "c-r"))
    st.markdown(f'<div class="mrow">{html}</div>',unsafe_allow_html=True)

    # Table
    rows=[]
    for t in reversed(completed):
        et = t.get("entry_time","—")
        xt = t.get("exit_time","—")
        rows.append({
            "Type":         t.get("signal","—"),
            "Entry Time":   et.strftime("%Y-%m-%d %H:%M:%S") if hasattr(et,"strftime") else str(et),
            "Entry Price":  t.get("entry_price","—"),
            "Exit Time":    xt.strftime("%Y-%m-%d %H:%M:%S") if hasattr(xt,"strftime") else str(xt),
            "Exit Price":   t.get("exit_price","—"),
            "SL":           t.get("init_sl","—"),
            "Target":       t.get("init_tgt","—"),
            "Entry Reason": t.get("reason_in","—"),
            "Exit Reason":  t.get("exit_reason","—"),
            "PnL":          t.get("pnl","—"),
            "Result":       t.get("result","—"),
        })

    df_h = pd.DataFrame(rows)
    if not df_h.empty:
        if len(df_h) <= 200:
            def _hcol(row):
                try:
                    pnl = float(row["PnL"])
                    if pnl > 0:   return ["background-color:#0d2620;color:#00d4aa"] * len(row)
                    elif pnl < 0: return ["background-color:#2a0f0f;color:#ff4b5c"] * len(row)
                except: pass
                return [""] * len(row)
            st.dataframe(df_h.style.apply(_hcol, axis=1),
                         use_container_width=True, height=450)
        else:
            st.dataframe(df_h, use_container_width=True, height=450)

        # Equity curve
        if len(completed)>1:
            st.markdown("#### 💰 Live Equity Curve")
            pnls = [t.get("pnl",0) for t in completed]
            cum  = pd.Series(pnls).cumsum().tolist()
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=list(range(1,len(cum)+1)),y=cum,mode="lines+markers",
                line=dict(color=C["cup"] if cum[-1]>=0 else C["cdn"],width=2),
                marker=dict(size=6,
                    color=[C["cup"] if p>=0 else C["cdn"] for p in pnls]),
                fill="tozeroy",
                fillcolor="rgba(0,212,170,0.08)" if cum[-1]>=0 else "rgba(255,75,92,0.08)"))
            fig3.update_layout(**_LAY,height=260,
                xaxis=dict(title="Trade #",showgrid=True,gridcolor=C["gr"]),
                yaxis=dict(title="Cumulative PnL ₹",showgrid=True,gridcolor=C["gr"]))
            st.plotly_chart(fig3,use_container_width=True)

        # Export
        csv = df_h.to_csv(index=False)
        st.download_button("⬇️ Export as CSV",data=csv,
                           file_name="smart_investing_trades.csv",
                           mime="text/csv",use_container_width=True)

    # Auto refresh while live (3 s — same guard as live tab)
    if is_active:
        time.sleep(3)
        st.rerun()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
  <span style="font-size:2rem">📈</span>
  <div>
    <div class="app-hdr">Smart Investing</div>
    <div style="color:#8892b0;font-size:.78rem">
      Advanced Algorithmic Trading · EMA Strategies · Backtesting · Live Trading
    </div>
  </div>
</div>
""",unsafe_allow_html=True)

    cfg = sidebar()

    tab1, tab2, tab3 = st.tabs([
        "📊  Backtesting",
        "🚀  Live Trading",
        "📚  Trade History",
    ])

    with tab1: tab_backtest(cfg)
    with tab2: tab_live(cfg)
    with tab3: tab_history(cfg)

if __name__ == "__main__":
    main()
