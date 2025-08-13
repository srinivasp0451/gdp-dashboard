# nifty_option_chain_friend.py
# Streamlit app: friendly, long-only option-chain guide (NIFTY)
# Features:
# - Reads NSE option chain CSV in standard "CALLS ... STRIKE ... PUTS" table format (like NSE site download).
# - Human-readable summary: ATM, OI walls, max-pain proxy, key build-ups & volumes.
# - Long-only recommendations (no shorting): clear Entry, Target, SL, Probability (proxy), and simple logic.
# - Straddles premium per strike with % change colored (green up / red down).
# - OI & ΔOI charts (bars go up; negative ΔOI shows below zero only if truly negative).
# - Volume bars per strike (CE & PE).
# - Payoff chart at expiry for buying any strike (CALL or PUT).
# - “Coach mode” to guide you like a friend.

import streamlit as st
import pandas as pd
import numpy as np
import csv, io
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="NIFTY Option Chain — Friendly Long-Only Guide", layout="wide")
st.title("🧭 NIFTY Option Chain — Your Friendly Long-Only Guide")

with st.sidebar:
    st.markdown("## Coach Mode")
    st.write(
        "I’ll guide you step-by-step:\n"
        "1) Upload option chain CSV (same format as NSE table download).\n"
        "2) I’ll detect ATM, OI walls, and build a clear summary.\n"
        "3) I’ll propose long-only buys with Entry / Target / SL and a POP (probability proxy).\n"
        "4) Use the charts to confirm the move and manage risk.\n"
        "5) Use the payoff chart before you hit Buy.\n"
    )
    risk = st.radio("Your risk comfort", ["Conservative", "Balanced", "Aggressive"], index=1)
    st.caption("Affects the default Target/SL multiples used below.")
    st.divider()
    st.markdown("**Pro tips:**")
    st.markdown(
        "- Prefer strikes near ATM for quick scalps.\n"
        "- Enter **with momentum** (price moving in your direction).\n"
        "- Respect **SL**. Small loss > big regret.\n"
        "- If flat / chop → avoid overtrading; time decay hurts buyers."
    )

def parse_nse_option_chain(file) -> pd.DataFrame:
    """
    Parse the NSE-like option-chain CSV:
    Row0: headers with 'CALLS', 'STRIKE', 'PUTS' (sometimes)
    Row1: actual column names
    Data: rows start at index 2
    """
    txt = file.getvalue().decode("utf-8", "ignore")
    reader = csv.reader(io.StringIO(txt))
    rows = [r for r in reader if any(str(c).strip() for c in r)]
    if len(rows) < 3:
        raise ValueError("This file doesn't look like an NSE option-chain table.")

    header = rows[1]
    data_rows = rows[2:]
    df = pd.DataFrame(data_rows, columns=header)

    # Build prefixed columns to avoid collisions
    call_cols = ['OI','CHNG IN OI','VOLUME','IV','LTP','CHNG','BID QTY','BID','ASK','ASK QTY']
    put_cols  = ['BID QTY','BID','ASK','ASK QTY','CHNG','LTP','IV','VOLUME','CHNG IN OI','OI']
    cols = list(df.columns)
    if 'STRIKE' not in cols:
        # Sometimes strike may be lowercase or with stray space
        # Attempt to locate the strike column by name
        strike_idx = None
        for i, c in enumerate(cols):
            if c.strip().lower() == 'strike':
                strike_idx = i
                break
        if strike_idx is None:
            raise ValueError("Couldn't find STRIKE column in header row.")
    else:
        strike_idx = cols.index('STRIKE')

    new_cols = []
    seen_put = 0
    for i, c in enumerate(cols):
        if i == strike_idx:
            new_cols.append('strike')
        elif c in call_cols and i < strike_idx:
            new_cols.append(f"C_{c.lower().replace(' ', '_')}")
        elif c in put_cols and i > strike_idx:
            new_cols.append(f"P_{put_cols[seen_put].lower().replace(' ', '_')}")
            seen_put += 1
        else:
            new_cols.append(c.lower().replace(' ', '_'))
    df.columns = [c.lower() for c in new_cols]

    def to_num(x):
        s = str(x).strip().replace(',', '')
        if s in ['', '-', '—', 'nan', 'None']:
            return np.nan
        try:
            return float(s)
        except:
            return np.nan

    for c in df.columns:
        if c == 'strike':
            df[c] = df[c].apply(to_num).astype(float)
        elif c.startswith('c_') or c.startswith('p_'):
            df[c] = df[c].apply(to_num)

    df = df[~df['strike'].isna()].copy()
    df.sort_values('strike', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def pop_proxy_distance(df: pd.DataFrame) -> pd.Series:
    """
    Probability-of-profit proxy:
    - If no greeks, estimate ITM probability via distance to ATM normalized by ATM straddle premium.
    - This is a **heuristic** (ballpark only).
    """
    df = df.copy()
    df['abs_cp_diff'] = (df['c_ltp'] - df['p_ltp']).abs()
    if df['abs_cp_diff'].isna().all():
        return pd.Series([np.nan] * len(df), index=df.index)

    atm_row = df.loc[df['abs_cp_diff'].idxmin()]
    atm_strike = atm_row['strike']
    atm_sp = (atm_row['c_ltp'] or 0) + (atm_row['p_ltp'] or 0)
    if pd.isna(atm_sp) or atm_sp <= 0:
        return pd.Series([np.nan] * len(df), index=df.index)

    def score(s):
        dist = abs(s - atm_strike)
        # simple exponential decay: center ~60%, decays with distance
        return float(max(5, min(90, 50 * np.exp(-dist / max(1.0, atm_sp)) + 10)))

    return df['strike'].apply(score)

def pick_multipliers(risk_mode: str):
    if risk_mode == "Conservative":
        return 1.10, 0.93  # +10% target, -7% SL
    if risk_mode == "Aggressive":
        return 1.25, 0.90  # +25% target, -10% SL
    return 1.15, 0.92      # Balanced default

def friendly_logic_text(side, strike, coi, vol, dltp, near_level):
    tip = (
        f"**{side} {int(strike)}** because we see **fresh interest** (ΔOI≈{coi:.0f}) "
        f"with **solid volume** ({vol:.0f}) and **price uptick** (ΔLTP≈{dltp:.2f}). "
    )
    if near_level:
        tip += f"It’s also near the **key level {near_level}**, so small moves can expand premium quickly. "
    tip += "Enter **with momentum** (price moving your way), avoid chop, and respect the SL."
    return tip

st.markdown("### 1) Upload your NSE option chain CSV")
uploaded = st.file_uploader("Drop the CSV (same structure as NSE option chain table).", type=["csv"])

if not uploaded:
    st.info("Waiting for a CSV… Tip: Download from NSE option chain and upload here.")
    st.stop()

# Parse file
try:
    df = parse_nse_option_chain(uploaded)
except Exception as e:
    st.error(f"Could not parse your file: {e}")
    st.stop()

# Derived metrics
df['straddle_premium'] = df['c_ltp'].fillna(0) + df['p_ltp'].fillna(0)
df['straddle_change']  = df['c_chng'].fillna(0) + df['p_chng'].fillna(0)
df['straddle_pct']     = np.where(df['straddle_premium'] > 0,
                                  df['straddle_change'] / df['straddle_premium'] * 100,
                                  np.nan)

df['total_oi']        = df['c_oi'].fillna(0) + df['p_oi'].fillna(0)
df['total_oi_change'] = df['c_chng_in_oi'].fillna(0) + df['p_chng_in_oi'].fillna(0)
df['total_volume']    = df['c_volume'].fillna(0) + df['p_volume'].fillna(0)

# ATM estimate
df['abs_cp_diff'] = (df['c_ltp'] - df['p_ltp']).abs()
atm_row = df.loc[df['abs_cp_diff'].idxmin()] if not df['abs_cp_diff'].isna().all() else None
atm_strike = int(atm_row['strike']) if atm_row is not None else None

# OI walls & max-pain proxy
call_oi_wall = int(df.loc[df['c_oi'].idxmax(), 'strike']) if df['c_oi'].notna().any() else None
put_oi_wall  = int(df.loc[df['p_oi'].idxmax(), 'strike']) if df['p_oi'].notna().any() else None
max_oi_strike = int(df.loc[df['total_oi'].idxmax(), 'strike']) if df['total_oi'].notna().any() else None

# POP proxy (we use the same for both sides if no greeks)
df['pop_proxy'] = pop_proxy_distance(df)

# ======= SUMMARY (friendly) =======
st.markdown("### 2) What’s happening (plain English)")

colA, colB, colC = st.columns(3)
colA.metric("ATM strike (approx)", f"{atm_strike}" if atm_strike else "—")
colB.metric("Call OI wall (resistance)", f"{call_oi_wall}" if call_oi_wall else "—")
colC.metric("Put OI wall (support)", f"{put_oi_wall}" if put_oi_wall else "—")

st.caption("Max-pain proxy (highest combined OI) often acts like a magnet near expiry.")
st.metric("Max combined OI (proxy)", f"{max_oi_strike}" if max_oi_strike else "—")

# Top ΔOI and Volume hotspots
hot_call = df.sort_values('c_chng_in_oi', ascending=False).head(3)[['strike','c_chng_in_oi','c_oi','c_volume','c_ltp']]
hot_put  = df.sort_values('p_chng_in_oi', ascending=False).head(3)[['strike','p_chng_in_oi','p_oi','p_volume','p_ltp']]
vol_call = df.sort_values('c_volume', ascending=False).head(3)[['strike','c_volume','c_ltp','c_chng','c_chng_in_oi']]
vol_put  = df.sort_values('p_volume', ascending=False).head(3)[['strike','p_volume','p_ltp','p_chng','p_chng_in_oi']]

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Call: biggest fresh build (ΔOI)**")
    st.dataframe(hot_call, use_container_width=True)
with c2:
    st.markdown("**Put: biggest fresh build (ΔOI)**")
    st.dataframe(hot_put, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.markdown("**Call: heaviest volume**")
    st.dataframe(vol_call, use_container_width=True)
with c4:
    st.markdown("**Put: heaviest volume**")
    st.dataframe(vol_put, use_container_width=True)

st.info(
    "Takeaway: Calls cluster near resistance (writers defend), Puts near support (writers defend). "
    "For **buyers**, near-ATM momentum bursts give the best odds for quick gains."
)

# ======= RECOMMENDATIONS (long-only) =======
st.markdown("### 3) Long-only opportunities (friend-style guidance)")

# Thresholds from quartiles (adaptive to your file)
def qsafe(s, q, default=0):
    s = s.dropna()
    return float(np.nanpercentile(s, q)) if len(s) else default

q_c_oi   = qsafe(df['c_chng_in_oi'], 75, 0)
q_c_vol  = qsafe(df['c_volume'], 75, 0)
q_p_oi   = qsafe(df['p_chng_in_oi'], 75, 0)
q_p_vol  = qsafe(df['p_volume'], 75, 0)

long_call_pool = df[(df['c_chng_in_oi'] >= q_c_oi) & (df['c_volume'] >= q_c_vol) & (df['c_chng'] >= 0)].copy()
long_put_pool  = df[(df['p_chng_in_oi'] >= q_p_oi) & (df['p_volume'] >= q_p_vol) & (df['p_chng'] >= 0)].copy()

# If too strict, fallback to positive ΔOI & volume > 0
if long_call_pool.empty:
    lc = df[(df['c_chng_in_oi'] > 0) & (df['c_volume'] > 0)].copy()
    lc['score'] = lc['c_chng_in_oi'].rank(pct=True)*0.6 + lc['c_volume'].rank(pct=True)*0.4
    long_call_pool = lc.sort_values('score', ascending=False).head(5)

if long_put_pool.empty:
    lp = df[(df['p_chng_in_oi'] > 0) & (df['p_volume'] > 0)].copy()
    lp['score'] = lp['p_chng_in_oi'].rank(pct=True)*0.6 + lp['p_volume'].rank(pct=True)*0.4
    long_put_pool = lp.sort_values('score', ascending=False).head(5)

t_mult, sl_mult = pick_multipliers(risk)

def make_reco_rows(sub: pd.DataFrame, side: str):
    rows = []
    for _, r in sub.iterrows():
        if side == 'CALL':
            ltp, dltp, iv, vol, coi = r['c_ltp'], r['c_chng'], r['c_iv'], r['c_volume'], r['c_chng_in_oi']
            pop = r['pop_proxy']
        else:
            ltp, dltp, iv, vol, coi = r['p_ltp'], r['p_chng'], r['p_iv'], r['p_volume'], r['p_chng_in_oi']
            pop = r['pop_proxy']
        if pd.isna(ltp) or ltp <= 0:
            continue
        entry  = round(ltp, 2)
        target = round(ltp * t_mult, 2)
        sl     = round(ltp * sl_mult, 2)
        near_key = None
        if atm_strike and abs(r['strike'] - atm_strike) <= 50:
            near_key = f"ATM {atm_strike}"
        elif call_oi_wall and abs(r['strike'] - call_oi_wall) <= 50:
            near_key = f"Call OI wall {call_oi_wall}"
        elif put_oi_wall and abs(r['strike'] - put_oi_wall) <= 50:
            near_key = f"Put OI wall {put_oi_wall}"
        logic_txt = friendly_logic_text(side, r['strike'], coi or 0, vol or 0, dltp or 0, near_key)
        rows.append({
            "Strike": int(r['strike']),
            "Type": side,
            "Entry": entry,
            "Target": target,
            "StopLoss": sl,
            "POP(Proxy)%": round(pop, 1) if pd.notna(pop) else None,
            "LTP": round(ltp, 2),
            "ΔLTP": round(dltp, 2) if pd.notna(dltp) else None,
            "IV": round(iv, 2) if pd.notna(iv) else None,
            "Volume": int(vol) if pd.notna(vol) else None,
            "ΔOI": int(coi) if pd.notna(coi) else None,
            "Why (human)": logic_txt
        })
    return rows

call_recos = make_reco_rows(long_call_pool, "CALL")
put_recos  = make_reco_rows(long_put_pool, "PUT")

# Show top 3 each by POP then Volume then ΔOI
def rank_recos(recos):
    if not recos: return []
    dfR = pd.DataFrame(recos)
    # Some POP may be None; fill for sorting
    dfR['_pop'] = dfR['POP(Proxy)%'].fillna(-1)
    dfR['_vol'] = dfR['Volume'].fillna(0)
    dfR['_oi']  = dfR['ΔOI'].fillna(0)
    dfR = dfR.sort_values(by=['_pop','_vol','_oi'], ascending=[False, False, False]).head(3)
    return dfR.drop(columns=['_pop','_vol','_oi'])

colL, colR = st.columns(2)
with colL:
    st.subheader("Buy CALL — top picks")
    df_call = rank_recos(call_recos)
    if len(df_call):
        st.dataframe(df_call, use_container_width=True)
    else:
        st.info("No strong call setups by the current heuristics.")
with colR:
    st.subheader("Buy PUT — top picks")
    df_put = rank_recos(put_recos)
    if len(df_put):
        st.dataframe(df_put, use_container_width=True)
    else:
        st.info("No strong put setups by the current heuristics.")

st.caption("POP is a **rough** probability proxy derived from ATM distance and straddle premium. Use it as guidance, not a guarantee.")

# ======= STRADDLES PREMIUM (chart + % colored) =======
st.markdown("### 4) Straddles premium by strike (with % change)")

# Table with colored % (green up / red down)
str_tbl = df[['strike','c_ltp','p_ltp','straddle_premium','straddle_change','straddle_pct']].copy()
styler = (
    str_tbl.style
    .format({'c_ltp':'{:.2f}','p_ltp':'{:.2f}','straddle_premium':'{:.2f}','straddle_change':'{:.2f}','straddle_pct':'{:.2f}'})
    .apply(lambda s: [
        'background-color: #e6ffed' if (pd.notna(v) and v>0) else (
        'background-color: #ffecec' if (pd.notna(v) and v<0) else '')
        for v in s
    ], subset=['straddle_pct'])
)
st.dataframe(styler, use_container_width=True)

# Bar chart of straddle premiums per strike
fig_str = px.bar(
    df, x="strike", y="straddle_premium",
    title="Straddle Premium per Strike",
    labels={"strike":"Strike", "straddle_premium":"CE+PE LTP"}
)
st.plotly_chart(fig_str, use_container_width=True)

# ======= OI & CHANGE IN OI (always upward; negative only if truly negative) =======
st.markdown("### 5) OI and Change in OI")

# OI are non-negative — bars point upward
fig_oi = go.Figure()
fig_oi.add_bar(x=df['strike'], y=df['c_oi'], name='Call OI')
fig_oi.add_bar(x=df['strike'], y=df['p_oi'], name='Put OI')
fig_oi.update_layout(barmode='group', title="Open Interest (upward bars)", xaxis_title="Strike", yaxis_title="OI")
st.plotly_chart(fig_oi, use_container_width=True)

# ΔOI can be negative — allow below zero only when it's negative
fig_doi = go.Figure()
fig_doi.add_bar(x=df['strike'], y=df['c_chng_in_oi'], name='Call ΔOI')
fig_doi.add_bar(x=df['strike'], y=df['p_chng_in_oi'], name='Put ΔOI')
fig_doi.update_layout(barmode='group', title="Change in OI (negative shown only if actual decrease)", xaxis_title="Strike", yaxis_title="ΔOI")
st.plotly_chart(fig_doi, use_container_width=True)

# ======= VOLUME (separate bars per strike) =======
st.markdown("### 6) Volume per strike (CE & PE)")
fig_vol = go.Figure()
fig_vol.add_bar(x=df['strike'], y=df['c_volume'], name='Call Volume')
fig_vol.add_bar(x=df['strike'], y=df['p_volume'], name='Put Volume')
fig_vol.update_layout(barmode='group', title="Volume per Strike", xaxis_title="Strike", yaxis_title="Contracts")
st.plotly_chart(fig_vol, use_container_width=True)

# ======= PAYOFF CHART (Buy a strike) =======
st.markdown("### 7) Payoff at Expiry — for buyers")

colP1, colP2, colP3 = st.columns(3)
opt_side = colP1.selectbox("Option Type", ["CALL","PUT"])
strike_sel = colP2.selectbox("Strike", df['strike'].astype(int).tolist())
if opt_side == "CALL":
    premium = float(df.loc[df['strike']==strike_sel, 'c_ltp'].values[0])
else:
    premium = float(df.loc[df['strike']==strike_sel, 'p_ltp'].values[0])

qty = colP3.number_input("Quantity (lots or units)", min_value=1, value=1, step=1)
spot_center = int(df['strike'].median())
spot_center = st.number_input("Spot range center", value=spot_center, step=50)

rng = np.arange(spot_center - 1000, spot_center + 1001, 50)

if opt_side == "CALL":
    intrinsic = np.maximum(rng - strike_sel, 0)
else:
    intrinsic = np.maximum(strike_sel - rng, 0)
payoff = (intrinsic - premium) * qty

fig_pay = px.line(x=rng, y=payoff, labels={'x':'Spot at Expiry', 'y':'Payoff'}, title=f"Payoff — Buy {opt_side} {strike_sel} @ {premium:.2f}")
fig_pay.add_hline(y=0, line_dash="dash")
st.plotly_chart(fig_pay, use_container_width=True)

breakeven = strike_sel + premium if opt_side == "CALL" else strike_sel - premium
st.success(f"**Breakeven**: {breakeven:.2f} | **Max Loss**: {premium*qty:.2f} | **Max Gain**: unlimited (CALL) / up to {strike_sel}−0−premium (PUT)")

# ======= FINAL FRIENDLY REMARKS =======
st.divider()
st.markdown("#### Friend’s checklist before you buy")
st.markdown(
    "- 🔍 Is price **moving in your direction now**? (Don’t buy during chop.)\n"
    "- 🧱 Are we **near ATM/support/resistance**? (Closer = quicker premium reaction.)\n"
    "- 🔊 Do **ΔOI and Volume** support the move?\n"
    "- 🧮 Are you comfortable with **Target / SL**? (Adjust via sidebar risk mode.)\n"
    "- ⏳ Remember **theta**—time decay hurts buyers in slow markets.\n"
)
st.caption("This tool is educational and not financial advice. Trade thoughtfully, manage risk, and size positions responsibly.")
