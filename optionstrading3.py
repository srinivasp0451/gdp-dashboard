# nifty_option_chain_final.py
import streamlit as st
import pandas as pd
import numpy as np
import csv, io, re
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="NIFTY Option Chain — Robust Final", layout="wide")
st.title("NIFTY Option Chain — Friendly Long-only Analyzer (Robust)")

# -------------------- helpers --------------------
def detect_delimiter(sample_text: str):
    """Detect delimiter using csv.Sniffer: fallback to tab or comma."""
    try:
        dialect = csv.Sniffer().sniff(sample_text[:2048])
        return dialect.delimiter
    except Exception:
        # simple heuristics
        if '\t' in sample_text:
            return '\t'
        if '|' in sample_text:
            return '|'
        return ','

def normalize_colname(s):
    s = "" if s is None else str(s).strip()
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    s = re.sub(r'[^0-9A-Za-z _\-]', '', s)  # remove weird chars
    s = s.lower().replace(' ', '_')
    if s == "":
        s = "unnamed"
    return s

def make_unique(cols):
    out = []
    seen = {}
    for c in cols:
        base = c
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        out.append(base)
    return out

def to_float_safe(x):
    if x is None: return np.nan
    s = str(x).strip()
    if s == '' or s in ['-', '—', 'NA', 'na', 'NaN', 'nan', 'None']:
        return np.nan
    s = s.replace(',', '').replace('%','')
    # parentheses negative like (123)
    if re.match(r'^\([0-9\.\-]+$', s):
        s = s.replace('(', '-').replace(')', '')
    try:
        return float(s)
    except:
        return np.nan

def find_header_row_and_rows(text, delim):
    """Return header row index and parsed rows (skip empty lines)."""
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    raw_rows = [r for r in reader]
    # remove fully empty rows
    rows = [r for r in raw_rows if any(cell.strip() for cell in r)]
    # try to find row containing 'strike' word
    for i, r in enumerate(rows[:10]):  # examine first 10 rows for header
        joined = ' '.join(r).lower()
        if 'strike' in joined:
            return i, rows
    # fallback second row
    return min(1, len(rows)-1), rows

def parse_uploaded(file) -> (pd.DataFrame, float):
    text = file.getvalue().decode('utf-8', 'ignore')
    delim = detect_delimiter(text)
    header_idx, rows = find_header_row_and_rows(text, delim)
    header = rows[header_idx]
    data_rows = rows[header_idx+1:]
    # normalize lengths
    max_len = len(header)
    norm_rows = []
    for r in data_rows:
        if len(r) < max_len:
            r = r + [''] * (max_len - len(r))
        elif len(r) > max_len:
            r = r[:max_len]
        norm_rows.append(r)
    # normalize header names and make unique
    norm_header = [normalize_colname(h) for h in header]
    norm_header = [h if h!='' else 'unnamed' for h in norm_header]
    uniq_header = make_unique(norm_header)
    df = pd.DataFrame(norm_rows, columns=uniq_header)
    # try to detect underlying/spot from rows above header
    underlying = None
    for r in rows[:header_idx+1]:
        for cell in r:
            if isinstance(cell, str) and re.search(r'(underlying|spot|index value|underlying value|ltp of underlying)', cell, re.I):
                # find numeric in same row
                for c in r:
                    v = to_float_safe(c)
                    if not np.isnan(v):
                        underlying = v
                        break
                if underlying is not None:
                    break
        if underlying is not None:
            break
    return df, underlying

def map_ce_pe_by_strike(df):
    """
    Map normalized parsed df columns into standard columns:
      c_* for columns to left of strike, p_* for columns to right of strike.
    Also convert numeric strings to floats safely.
    """
    cols = list(df.columns)
    # find strike column index by column name containing 'strike' OR numeric-looking column with many numeric rows
    strike_idx = None
    for i,c in enumerate(cols):
        if 'strike' in c:
            strike_idx = i
            break
    if strike_idx is None:
        # fallback: choose column where many values look numeric and likely increasing
        best_i, best_cnt = None, -1
        for i,c in enumerate(cols):
            vals = df[c].dropna().astype(str).tolist()[:50]
            cnt = sum(1 for v in vals if not np.isnan(to_float_safe(v)))
            if cnt > best_cnt:
                best_cnt = cnt; best_i = i
        strike_idx = best_i

    if strike_idx is None:
        raise KeyError("Strike column not found in uploaded file.")

    # build new column names
    new_cols = []
    for i,c in enumerate(cols):
        if i == strike_idx:
            new_cols.append('strike')
        elif i < strike_idx:
            new_cols.append('c_' + c)
        else:
            new_cols.append('p_' + c)

    # rename dataframe
    df2 = df.copy()
    df2.columns = new_cols

    # convert numeric-like columns using to_float_safe
    for c in df2.columns:
        if c == 'strike':
            df2[c] = df2[c].apply(to_float_safe)
        else:
            # convert all CE/PE side numbers to floats; leave non-numeric as NaN
            df2[c] = df2[c].apply(to_float_safe)
    # remove rows where strike is NaN
    df2 = df2[~df2['strike'].isna()].reset_index(drop=True)
    return df2

def compute_pop_proxy(df, atm_strike):
    # if deltas exist use them
    if 'c_delta' in df.columns and 'p_delta' in df.columns:
        call_pop = df['c_delta'].abs()*100
        put_pop  = df['p_delta'].abs()*100
        note = "POP from delta columns (when available)"
    else:
        # use distance-to-atm normalized by atm straddle premium
        df = df.copy()
        df['c_ltp_f'] = df.get('c_ltp', 0).fillna(0)
        df['p_ltp_f'] = df.get('p_ltp', 0).fillna(0)
        atm_sp = None
        if atm_strike is not None and (atm_strike in df['strike'].values):
            atm_sp = float(df.loc[df['strike']==atm_strike, 'c_ltp_f'].iloc[0] + df.loc[df['strike']==atm_strike, 'p_ltp_f'].iloc[0])
        def f(s):
            if atm_strike is None or atm_sp is None or atm_sp <= 0:
                return np.nan
            dist = abs(s - atm_strike)
            return float(max(5, min(90, 50 * np.exp(-dist / max(1.0, atm_sp)) + 10)))
        call_pop = df['strike'].apply(f)
        put_pop  = df['strike'].apply(f)
        note = "POP proxy from distance-to-ATM normalized by ATM straddle premium (heuristic)"
    return call_pop, put_pop, note

# -------------------- UI --------------------
st.markdown("Upload the NSE option-chain CSV/TSV pasted or downloaded from NSE. This parser tolerates empty headers and duplicate names.")

uploaded = st.file_uploader("Upload option-chain file (CSV/TSV)", type=["csv","txt"])

if not uploaded:
    st.info("Upload an option-chain file (CSV/TSV). Example row format (calls ... strike ... puts) is supported.")
    st.stop()

# Parse file robustly
try:
    parsed_raw, underlying = parse_uploaded(uploaded)
except Exception as e:
    st.error(f"Failed to parse file: {e}")
    st.stop()

# Map to CE/PE structured table
try:
    df = map_ce_pe_by_strike(parsed_raw)
except Exception as e:
    st.error(f"Failed to map CE/PE by strike: {e}")
    st.stop()

# Normalize commonly used column names to friendly keys if present
# We'll attempt to guess columns like OI, chng_in_oi, ltp, chng, iv, volume, bid, ask
def guess_col(df, patterns):
    for p in patterns:
        if p in df.columns:
            return p
    # try fuzzy: check substrings
    lowercols = {c:c for c in df.columns}
    for c in df.columns:
        for p in patterns:
            if p in c:
                return c
    return None

# rename important CE/PE columns if they exist
colmap = {}
# CE side patterns
colmap['c_oi'] = guess_col(df, ['c_oi','c_oi_','c_oi_0','c_oi_1','c_oi_2','c_open_interest','c_openinterest','c_oi'])
colmap['c_chng_in_oi'] = guess_col(df, ['c_chng_in_oi','c_chng','c_change_in_oi','c_change_in_open_interest','c_chng_in_oi'])
colmap['c_volume'] = guess_col(df, ['c_volume','c_vol','c_tradevolume','c_volume_'])
colmap['c_iv'] = guess_col(df, ['c_iv','c_implied_volatility','c_impliedvolatility'])
colmap['c_ltp'] = guess_col(df, ['c_ltp','c_last_traded_price','c_last_price','c_ltp_'])
colmap['c_chng'] = guess_col(df, ['c_chng','c_change','c_chg','c_chng_'])

# PE side patterns (columns already prefixed with p_)
colmap['p_oi'] = guess_col(df, ['p_oi','p_open_interest','p_oi_'])
colmap['p_chng_in_oi'] = guess_col(df, ['p_chng_in_oi','p_chng','p_change_in_oi'])
colmap['p_volume'] = guess_col(df, ['p_volume','p_vol','p_tradevolume'])
colmap['p_iv'] = guess_col(df, ['p_iv','p_implied_volatility'])
colmap['p_ltp'] = guess_col(df, ['p_ltp','p_last_traded_price','p_last_price'])
colmap['p_chng'] = guess_col(df, ['p_chng','p_change','p_chg'])

# Apply renames where found
rename_lookup = {}
for std, found in colmap.items():
    if found is not None and found != std:
        rename_lookup[found] = std
if rename_lookup:
    df = df.rename(columns=rename_lookup)

# Ensure numeric dtype for the mapped columns
for col in ['strike','c_oi','c_chng_in_oi','c_volume','c_iv','c_ltp','c_chng','p_ltp','p_chng','p_iv','p_volume','p_chng_in_oi','p_oi']:
    if col in df.columns:
        # if it's a scalar or int/float already cast safe series; ensure it's a Series
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Make sure there is a numeric strike column
if 'strike' not in df.columns:
    st.error("Could not detect numeric strike column after parsing. Please paste a small sample (first 8 rows) and I'll adapt parser.")
    st.stop()

# Drop rows where strike is NaN
df = df[~df['strike'].isna()].copy()
df['strike'] = df['strike'].astype(float)

# Derived columns
df['straddle_premium'] = df.get('c_ltp', 0).fillna(0) + df.get('p_ltp', 0).fillna(0)
df['straddle_change']  = df.get('c_chng', 0).fillna(0) + df.get('p_chng', 0).fillna(0)
df['straddle_pct']     = np.where(df['straddle_premium']>0, df['straddle_change'] / df['straddle_premium'] * 100.0, np.nan)
df['total_oi']         = df.get('c_oi', 0).fillna(0) + df.get('p_oi', 0).fillna(0)
df['total_oi_change']  = df.get('c_chng_in_oi', 0).fillna(0) + df.get('p_chng_in_oi', 0).fillna(0)
df['total_volume']     = df.get('c_volume', 0).fillna(0) + df.get('p_volume', 0).fillna(0)

# ATM detection using smallest abs(c_ltp - p_ltp) difference or nearest to underlying if detected
atm_strike = None
if ('c_ltp' in df.columns) and ('p_ltp' in df.columns):
    df['abs_cp_diff'] = (df['c_ltp'].fillna(0) - df['p_ltp'].fillna(0)).abs()
    if df['abs_cp_diff'].notna().any():
        atm_strike = int(df.loc[df['abs_cp_diff'].idxmin(), 'strike'])
# If we found underlying from header, pick nearest strike to underlying
if atm_strike is None and (underlying is not None) and (not np.isnan(underlying)):
    # find nearest strike
    atm_strike = int(df.iloc[(df['strike'] - underlying).abs().argsort()[:1]]['strike'].iloc[0])

# ---------- SUMMARY (friendly) ----------
st.subheader("1) Plain-English summary")
c1, c2, c3 = st.columns(3)
c1.metric("Detected ATM (approx)", f"{atm_strike}" if atm_strike is not None else "—")
c2.metric("Underlying (if found)", f"{underlying:.2f}" if (underlying is not None and not np.isnan(underlying)) else "—")
max_oi_strike = int(df.loc[df['total_oi'].idxmax(), 'strike']) if df['total_oi'].notna().any() else "—"
c3.metric("Max combined OI (proxy)", f"{max_oi_strike}")

st.markdown("""
**Interpretation (friendly):**  
- The app detected ATM and the strikes where OI is big.  
- Large **ΔOI + Volume** in CE suggests fresh bullish interest (or call writing depending on price movement).  
- Large **ΔOI + Volume** in PE suggests fresh bearish interest (or put writing depending on price movement).  
We produce **buy-only** recommendations: buy CE when upward momentum + CE ΔOI/volume confirms; buy PE when downward momentum + PE ΔOI/volume confirms.
""")

# show hotspots
st.subheader("Top activity hotspots (calls / puts)")
hot_call = df.sort_values('c_chng_in_oi', ascending=False).head(5)[['strike','c_chng_in_oi','c_oi','c_volume','c_ltp']].fillna('-')
hot_put  = df.sort_values('p_chng_in_oi', ascending=False).head(5)[['strike','p_chng_in_oi','p_oi','p_volume','p_ltp']].fillna('-')
hc, hp = st.columns(2)
with hc:
    st.markdown("🔺 Calls (largest ΔOI)")
    st.dataframe(hot_call, use_container_width=True)
with hp:
    st.markdown("🔻 Puts (largest ΔOI)")
    st.dataframe(hot_put, use_container_width=True)

# ---------- Recommendations (long-only) ----------
st.subheader("2) Long-only buy recommendations (human friendly)")

# percentile thresholds
def pctile_safe(series, q):
    if series is None or series.dropna().empty:
        return 0.0
    return float(np.nanpercentile(series.dropna(), q))

q_c_oi = pctile_safe(df.get('c_chng_in_oi', pd.Series([], dtype=float)), 75)
q_c_vol = pctile_safe(df.get('c_volume', pd.Series([], dtype=float)), 75)
q_p_oi = pctile_safe(df.get('p_chng_in_oi', pd.Series([], dtype=float)), 75)
q_p_vol = pctile_safe(df.get('p_volume', pd.Series([], dtype=float)), 75)

# compute POP proxy
call_pop, put_pop, pop_note = compute_pop_proxy(df, atm_strike)
df['call_pop'] = call_pop
df['put_pop'] = put_pop

recs = []
# Candidate CE buys
candidates_call = df[(df.get('c_chng_in_oi',0) >= q_c_oi) & (df.get('c_volume',0) >= q_c_vol)].copy()
if candidates_call.empty:
    candidates_call = df[(df.get('c_chng_in_oi',0) > 0) & (df.get('c_volume',0) > 0)].sort_values(['c_chng_in_oi','c_volume'], ascending=False).head(5)

for idx, r in candidates_call.iterrows():
    ltp = r.get('c_ltp', np.nan)
    if pd.isna(ltp) or ltp <= 0:
        continue
    entry = round(ltp,2)
    target = round(entry * 1.15, 2)
    stop = round(entry * 0.92, 2)
    pop = r.get('call_pop', np.nan)
    reason = f"CE ΔOI={int(r.get('c_chng_in_oi',0))}, Volume={int(r.get('c_volume',0) if not np.isnan(r.get('c_volume', np.nan)) else 0)}, ΔLTP={r.get('c_chng',0)}"
    human = f"Buy CE {int(r['strike'])} if you see an upward momentum candle — {reason}."
    recs.append({
        "Strike": int(r['strike']),
        "Type": "CALL",
        "Entry": entry,
        "Target": target,
        "StopLoss": stop,
        "POP(Proxy%)": round(float(pop),1) if not pd.isna(pop) else np.nan,
        "LTP": entry,
        "ΔLTP": r.get('c_chng', np.nan),
        "IV": r.get('c_iv', np.nan),
        "Volume": int(r.get('c_volume',0)) if not np.isnan(r.get('c_volume', np.nan)) else None,
        "ΔOI": int(r.get('c_chng_in_oi',0)) if not np.isnan(r.get('c_chng_in_oi', np.nan)) else None,
        "Why": human
    })

# Candidate PE buys
candidates_put = df[(df.get('p_chng_in_oi',0) >= q_p_oi) & (df.get('p_volume',0) >= q_p_vol)].copy()
if candidates_put.empty:
    candidates_put = df[(df.get('p_chng_in_oi',0) > 0) & (df.get('p_volume',0) > 0)].sort_values(['p_chng_in_oi','p_volume'], ascending=False).head(5)

for idx, r in candidates_put.iterrows():
    ltp = r.get('p_ltp', np.nan)
    if pd.isna(ltp) or ltp <= 0:
        continue
    entry = round(ltp,2)
    target = round(entry * 1.15, 2)
    stop = round(entry * 0.92, 2)
    pop = r.get('put_pop', np.nan)
    reason = f"PE ΔOI={int(r.get('p_chng_in_oi',0))}, Volume={int(r.get('p_volume',0) if not np.isnan(r.get('p_volume', np.nan)) else 0)}, ΔLTP={r.get('p_chng',0)}"
    human = f"Buy PE {int(r['strike'])} if you see a downward momentum candle — {reason}."
    recs.append({
        "Strike": int(r['strike']),
        "Type": "PUT",
        "Entry": entry,
        "Target": target,
        "StopLoss": stop,
        "POP(Proxy%)": round(float(pop),1) if not pd.isna(pop) else np.nan,
        "LTP": entry,
        "ΔLTP": r.get('p_chng', np.nan),
        "IV": r.get('p_iv', np.nan),
        "Volume": int(r.get('p_volume',0)) if not np.isnan(r.get('p_volume', np.nan)) else None,
        "ΔOI": int(r.get('p_chng_in_oi',0)) if not np.isnan(r.get('p_chng_in_oi', np.nan)) else None,
        "Why": human
    })

reco_df = pd.DataFrame(recs)
if reco_df.empty:
    st.info("No strong long-only buys found by the heuristics (try relaxing thresholds or upload a different snapshot).")
else:
    # sort by POP then Volume then ΔOI
    reco_df['_pop'] = reco_df['POP(Proxy%)'].fillna(-1)
    reco_df['_vol'] = reco_df['Volume'].fillna(0)
    reco_df['_oi']  = reco_df['ΔOI'].fillna(0)
    reco_df = reco_df.sort_values(['_pop','_vol','_oi'], ascending=[False, False, False]).drop(columns=['_pop','_vol','_oi'])
    st.dataframe(reco_df.reset_index(drop=True), use_container_width=True)

st.caption(f"POP note: {pop_note} — POP is a heuristic, treat it as guidance, not certainty.")

# ---------- Charts ----------
st.subheader("3) Straddle premium per strike (with % change)")

# prepare straddle table
strad = df[['strike']].copy()
strad['c_ltp'] = df.get('c_ltp', np.nan)
strad['p_ltp'] = df.get('p_ltp', np.nan)
strad['straddle'] = strad['c_ltp'].fillna(0) + strad['p_ltp'].fillna(0)
strad['straddle_change'] = df.get('c_chng', 0).fillna(0) + df.get('p_chng', 0).fillna(0)
strad['straddle_pct'] = np.where(strad['straddle']>0, strad['straddle_change'] / strad['straddle'] * 100.0, np.nan)
strad = strad.sort_values('strike')

# bar colored by sign of straddle_pct
fig = px.bar(strad, x='strike', y='straddle', color=(strad['straddle_pct']>0),
             color_discrete_map={True: 'green', False: 'red'},
             title='Straddle premium (CE+PE) per strike — green = increase, red = decrease')
fig.update_layout(xaxis_title='Strike', yaxis_title='CE+PE LTP')
st.plotly_chart(fig, use_container_width=True)

st.markdown("Percent column (increase / decrease):")
st.dataframe(strad[['strike','c_ltp','p_ltp','straddle','straddle_pct']].rename(columns={
    'strike':'Strike','c_ltp':'CE LTP','p_ltp':'PE LTP','straddle':'Straddle Premium','straddle_pct':'Straddle %'
}).style.format({'CE LTP':'{:.2f}','PE LTP':'{:.2f}','Straddle Premium':'{:.2f}','Straddle %':'{:.2f}'}).applymap(lambda v: 'color: green' if (isinstance(v,(int,float)) and v>0) else ('color: red' if (isinstance(v,(int,float)) and v<0) else ''), subset=['Straddle %']), use_container_width=True)

# OI and ΔOI charts
st.subheader("4) OI and ΔOI charts")
oi_fig = go.Figure()
if 'c_oi' in df.columns:
    oi_fig.add_trace(go.Bar(x=df['strike'], y=df['c_oi'], name='Call OI'))
if 'p_oi' in df.columns:
    oi_fig.add_trace(go.Bar(x=df['strike'], y=df['p_oi'], name='Put OI'))
oi_fig.update_layout(barmode='group', xaxis_title='Strike', yaxis_title='Open Interest')
st.plotly_chart(oi_fig, use_container_width=True)

doi_fig = go.Figure()
if 'c_chng_in_oi' in df.columns:
    doi_fig.add_trace(go.Bar(x=df['strike'], y=df['c_chng_in_oi'], name='Call ΔOI'))
if 'p_chng_in_oi' in df.columns:
    doi_fig.add_trace(go.Bar(x=df['strike'], y=df['p_chng_in_oi'], name='Put ΔOI'))
doi_fig.update_layout(barmode='group', xaxis_title='Strike', yaxis_title='Change in OI')
st.plotly_chart(doi_fig, use_container_width=True)

# Volume chart
st.subheader("5) Volume per strike (CE & PE)")
vol_fig = go.Figure()
if 'c_volume' in df.columns:
    vol_fig.add_trace(go.Bar(x=df['strike'], y=df['c_volume'], name='Call Volume'))
if 'p_volume' in df.columns:
    vol_fig.add_trace(go.Bar(x=df['strike'], y=df['p_volume'], name='Put Volume'))
vol_fig.update_layout(barmode='group', xaxis_title='Strike', yaxis_title='Volume')
st.plotly_chart(vol_fig, use_container_width=True)

# ---------- Payoff chart ----------
st.subheader("6) Payoff chart (buy a strike) — use before placing order")
# build strike choices
strike_choices = sorted(df['strike'].dropna().unique().astype(int).tolist())
if not strike_choices:
    st.info("No strikes present to chart payoff.")
else:
    sel_method = st.radio("Choose strike from", ["Recommendations", "All strikes"], index=0)
    chosen_strike = None
    chosen_side = None
    chosen_prem = None
    if sel_method == "Recommendations" and (not reco_df.empty):
        # show top recs to select
        sel = st.selectbox("Pick recommendation", [f"{r['Type']} {r['Strike']}" for _, r in reco_df.reset_index(drop=True).iterrows()])
        if sel:
            typ, s = sel.split()
            chosen_side = typ
            chosen_strike = int(s)
            row = reco_df[(reco_df['Strike']==chosen_strike) & (reco_df['Type']==chosen_side)]
            if not row.empty:
                chosen_prem = float(row.iloc[0]['Entry'])
    else:
        chosen_strike = st.selectbox("Pick strike", ["-- none --"] + strike_choices)
        chosen_side = st.selectbox("Option side", ["CALL","PUT"])
        if chosen_strike != "-- none --":
            chosen_strike = int(chosen_strike)
            # premium from df
            if chosen_side == "CALL":
                chosen_prem = None
                if 'c_ltp' in df.columns:
                    tmp = df.loc[df['strike']==chosen_strike, 'c_ltp']
                    if not tmp.dropna().empty:
                        chosen_prem = float(tmp.dropna().iloc[0])
            else:
                chosen_prem = None
                if 'p_ltp' in df.columns:
                    tmp = df.loc[df['strike']==chosen_strike, 'p_ltp']
                    if not tmp.dropna().empty:
                        chosen_prem = float(tmp.dropna().iloc[0])

    if chosen_strike is None or chosen_strike == "-- none --":
        st.info("Select a strike from recommendations or the strike list.")
    else:
        if chosen_prem is None or np.isnan(chosen_prem):
            st.warning("Premium not available for chosen strike & side in this file. You can still plot with a manual premium.")
            chosen_prem = st.number_input("Enter premium to use for payoff", min_value=0.0, value=0.0)
        else:
            st.write(f"Using premium = {chosen_prem:.2f} (from file)")

        center = underlying if (underlying is not None and not np.isnan(underlying)) else chosen_strike
        rng = np.arange(int(center-800), int(center+801), 25)
        if chosen_side == "CALL":
            payoff = np.maximum(rng - chosen_strike, 0) - chosen_prem
        else:
            payoff = np.maximum(chosen_strike - rng, 0) - chosen_prem
        pay_fig = go.Figure()
        pay_fig.add_trace(go.Scatter(x=rng, y=payoff, mode='lines', name='Payoff'))
        pay_fig.add_hline(y=0, line_dash='dash')
        pay_fig.update_layout(title=f"Payoff at expiry — Buy {chosen_side} {chosen_strike} @ {chosen_prem:.2f}",
                              xaxis_title='Spot at expiry', yaxis_title='PnL')
        st.plotly_chart(pay_fig, use_container_width=True)
        breakeven = chosen_strike + chosen_prem if chosen_side == "CALL" else chosen_strike - chosen_prem
        st.success(f"Breakeven: {breakeven:.2f} | Max loss: premium ({chosen_prem:.2f})")

# final tips
st.markdown("---")
st.markdown("**Friendly checklist before executing a buy:**")
st.markdown("- Confirm momentum in price (don’t buy in chop).")
st.markdown("- ΔOI and Volume rising together adds conviction.")
st.markdown("- Use displayed Entry/Target/StopLoss and size accordingly.")
st.caption("This tool provides heuristics and guidance only — not financial advice. Trade responsibly.")
