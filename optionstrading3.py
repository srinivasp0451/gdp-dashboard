# nifty_option_chain_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
import csv, io, re
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="NIFTY Option Chain — Fixed Parser", layout="wide")

# ---------- Helpers ----------
def to_float_safe(x):
    if x is None: return np.nan
    s = str(x).strip()
    if s in ['','-','—','NA','NaN','nan', 'None']:
        return np.nan
    # remove commas and percentage signs and parentheses
    s = s.replace(',', '').replace('%','')
    s = s.replace('(', '-').replace(')', '')
    try:
        return float(s)
    except:
        return np.nan

def normalize_colname(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r'[^0-9a-zA-Z_ ]+', '', s)          # drop odd chars
    s = re.sub(r'\s+', ' ', s).strip()             # normalize spaces
    s = s.replace(' ', '_')
    if s == '':
        return 'unnamed'
    return s

def make_unique(cols):
    out = []
    seen = {}
    for i,c in enumerate(cols):
        base = c
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        out.append(base)
    return out

def find_header_row(rows):
    # find the row index that contains a cell like 'strike' (case-insensitive)
    for i, r in enumerate(rows):
        for c in r:
            if isinstance(c, str) and re.search(r'\bstrike\b', c, re.I):
                return i
    # fallback: if file has at least 2 rows return index 1 else 0
    return 1 if len(rows) > 1 else 0

def parse_nse_style_csv(file_bytes):
    text = file_bytes.getvalue().decode('utf-8', 'ignore')
    reader = csv.reader(io.StringIO(text))
    raw_rows = [row for row in reader]
    # remove completely empty rows
    rows = [r for r in raw_rows if any(str(c).strip() for c in r)]
    if not rows:
        raise ValueError("Uploaded file appears empty or unreadable.")
    header_idx = find_header_row(rows)
    header_row = rows[header_idx]
    data_rows = rows[header_idx+1:]
    # normalize row length
    max_len = len(header_row)
    norm_rows = []
    for r in data_rows:
        if len(r) < max_len:
            r = r + ['']*(max_len - len(r))
        elif len(r) > max_len:
            r = r[:max_len]
        norm_rows.append(r)

    # make header unique & normalized
    raw_headers = [h if (h is not None) else "" for h in header_row]
    norm_headers = [normalize_colname(h) for h in raw_headers]
    norm_headers = [h if h!='' else f'unnamed' for h in norm_headers]
    unique_headers = make_unique(norm_headers)

    df = pd.DataFrame(norm_rows, columns=unique_headers)

    # try to find an "underlying/spot" value in pre-header lines (search first few raw rows)
    underlying = None
    for r in rows[:min(6, header_idx+1)]:
        for i,cell in enumerate(r):
            if isinstance(cell, str) and re.search(r'(underlying|spot|underlying_value|index|underlying value|underlyingindex)', cell, re.I):
                # try to find numeric neighbor
                if i+1 < len(r):
                    val = to_float_safe(r[i+1])
                    if not np.isnan(val):
                        underlying = val
                        break
                # scan row for any number
                for c in r:
                    val = to_float_safe(c)
                    if not np.isnan(val):
                        underlying = val
                        break
        if underlying is not None:
            break

    return df, header_idx, underlying

# ---------- Column mapping helper ----------
def map_columns_to_ce_pe(df, header_idx):
    """
    After parsing the CSV into a DataFrame with normalized unique headers,
    map columns to a standardized set:
    outputs dataframe with renamed columns such as:
      strike, c_oi, c_chng_in_oi, c_volume, c_iv, c_ltp, c_chng, p_ltp, p_chng, p_iv, p_volume, p_chng_in_oi, p_oi
    The left-of-strike columns get 'c_' prefix, right-of-strike get 'p_' prefix.
    """
    cols = list(df.columns)
    # find index of column whose original name contains 'strike'
    strike_idx = None
    for i,c in enumerate(cols):
        if re.search(r'\bstrike\b', c, re.I) or re.search(r'\bstrike_price\b', c, re.I):
            strike_idx = i
            break
    if strike_idx is None:
        # try looking for exact numeric-like column by sampling rows
        for i,c in enumerate(cols):
            # check if column values look numeric in many rows
            sample_vals = df[c].dropna().astype(str).tolist()[:10]
            numeric_count = sum(1 for v in sample_vals if to_float_safe(v) is not np.nan)
            if numeric_count >= max(1, len(sample_vals)//2):
                strike_idx = i
                break
    if strike_idx is None:
        raise KeyError("Could not find the strike column in the CSV.")

    # Build mapped columns
    new_names = []
    for i,c in enumerate(cols):
        if i == strike_idx:
            new_names.append('strike')
        elif i < strike_idx:
            # Calls side
            nn = 'c_' + normalize_colname(c)
            new_names.append(nn)
        else:
            # Puts side
            nn = 'p_' + normalize_colname(c)
            new_names.append(nn)

    df = df.copy()
    df.columns = new_names

    # convert numeric-like CE/PE columns
    for col in df.columns:
        if col == 'strike':
            df[col] = df[col].apply(to_float_safe).astype(float)
        elif col.startswith('c_') or col.startswith('p_'):
            df[col] = df[col].apply(to_float_safe)

    return df

# ---------- POP proxy ----------
def compute_pop_proxy(df, atm_strike):
    # if delta columns exist, use them
    if 'c_delta' in df.columns and 'p_delta' in df.columns:
        call_pop = df['c_delta'].abs() * 100
        put_pop  = df['p_delta'].abs() * 100
        note = "POP from available option deltas (proxy)."
    else:
        # distance to ATM normalized by ATM straddle premium heuristic
        df['c_ltp_f'] = df['c_ltp'].fillna(0)
        df['p_ltp_f'] = df['p_ltp'].fillna(0)
        df['straddle_atm'] = df['c_ltp_f'] + df['p_ltp_f']
        atm_row = df.loc[df['strike'] == atm_strike] if atm_strike in df['strike'].values else None
        atm_sp = float(atm_row['straddle_atm'].iloc[0]) if (isinstance(atm_row, pd.DataFrame) and not atm_row.empty) else np.nan
        def f(s):
            if pd.isna(atm_strike) or pd.isna(atm_sp) or atm_sp <= 0:
                return np.nan
            dist = abs(s - atm_strike)
            return float(max(5, min(90, 50 * np.exp(-dist / max(1.0, atm_sp)) + 10)))
        call_pop = df['strike'].apply(f)
        put_pop  = df['strike'].apply(f)
        note = "POP proxy from distance-to-ATM normalized by ATM straddle premium (heuristic)."
    return call_pop, put_pop, note

# ---------- Recommendation logic ----------
def recommend_long_trades(df, atm_strike, risk='Balanced'):
    # thresholds based on percentiles
    q_c_oi = float(np.nanpercentile(df['c_chng_in_oi'].dropna(), 75)) if df['c_chng_in_oi'].notna().sum() else 0.0
    q_c_vol = float(np.nanpercentile(df['c_volume'].dropna(), 75)) if df['c_volume'].notna().sum() else 0.0
    q_p_oi = float(np.nanpercentile(df['p_chng_in_oi'].dropna(), 75)) if df['p_chng_in_oi'].notna().sum() else 0.0
    q_p_vol = float(np.nanpercentile(df['p_volume'].dropna(), 75)) if df['p_volume'].notna().sum() else 0.0

    # risk multipliers
    if risk == 'Conservative':
        t_mult, sl_mult = 1.10, 0.94
    elif risk == 'Aggressive':
        t_mult, sl_mult = 1.30, 0.88
    else:
        t_mult, sl_mult = 1.15, 0.92

    call_pop, put_pop, pop_note = compute_pop_proxy(df, atm_strike)

    recs = []
    # find call candidates
    calls = df[(df['c_chng_in_oi'] >= q_c_oi) & (df['c_volume'] >= q_c_vol)].copy()
    puts  = df[(df['p_chng_in_oi'] >= q_p_oi) & (df['p_volume'] >= q_p_vol)].copy()

    # Fallback: if empty, relax filters
    if calls.empty:
        calls = df[(df['c_chng_in_oi'] > 0) & (df['c_volume'] > 0)].copy().sort_values(['c_chng_in_oi','c_volume'], ascending=False).head(5)
    if puts.empty:
        puts = df[(df['p_chng_in_oi'] > 0) & (df['p_volume'] > 0)].copy().sort_values(['p_chng_in_oi','p_volume'], ascending=False).head(5)

    for _, r in calls.iterrows():
        ltp = r.get('c_ltp', np.nan)
        if pd.isna(ltp) or ltp <= 0:
            continue
        entry = round(ltp,2)
        target = round(ltp * t_mult,2)
        sl = round(ltp * sl_mult,2)
        pop = call_pop.loc[r.name] if hasattr(call_pop, 'loc') else np.nan
        reason = f"Call ΔOI={int(r.get('c_chng_in_oi',0))}, Volume={int(r.get('c_volume',0))}, ΔLTP={r.get('c_chng',0)}."
        human = f"Buy CE {int(r['strike'])} if you see momentum upwards — {reason}"
        recs.append({
            'Strike': int(r['strike']),
            'Type': 'CALL',
            'Entry': entry,
            'Target': target,
            'StopLoss': sl,
            'POP(Proxy%)': round(float(pop),1) if not pd.isna(pop) else np.nan,
            'LTP': ltp,
            'ΔLTP': r.get('c_chng', np.nan),
            'IV': r.get('c_iv', np.nan),
            'Volume': int(r.get('c_volume',0)) if not pd.isna(r.get('c_volume')) else None,
            'ΔOI': int(r.get('c_chng_in_oi',0)) if not pd.isna(r.get('c_chng_in_oi')) else None,
            'Why': human
        })

    for _, r in puts.iterrows():
        ltp = r.get('p_ltp', np.nan)
        if pd.isna(ltp) or ltp <= 0:
            continue
        entry = round(ltp,2)
        target = round(ltp * t_mult,2)
        sl = round(ltp * sl_mult,2)
        pop = put_pop.loc[r.name] if hasattr(put_pop, 'loc') else np.nan
        reason = f"Put ΔOI={int(r.get('p_chng_in_oi',0))}, Volume={int(r.get('p_volume',0))}, ΔLTP={r.get('p_chng',0)}."
        human = f"Buy PE {int(r['strike'])} if you see momentum down — {reason}"
        recs.append({
            'Strike': int(r['strike']),
            'Type': 'PUT',
            'Entry': entry,
            'Target': target,
            'StopLoss': sl,
            'POP(Proxy%)': round(float(pop),1) if not pd.isna(pop) else np.nan,
            'LTP': ltp,
            'ΔLTP': r.get('p_chng', np.nan),
            'IV': r.get('p_iv', np.nan),
            'Volume': int(r.get('p_volume',0)) if not pd.isna(r.get('p_volume')) else None,
            'ΔOI': int(r.get('p_chng_in_oi',0)) if not pd.isna(r.get('p_chng_in_oi')) else None,
            'Why': human
        })

    # sort recs by POP then volume then ΔOI
    rec_df = pd.DataFrame(recs)
    if not rec_df.empty:
        rec_df['_pop'] = rec_df['POP(Proxy%)'].fillna(-1)
        rec_df['_vol'] = rec_df['Volume'].fillna(0)
        rec_df['_oi']  = rec_df['ΔOI'].fillna(0)
        rec_df = rec_df.sort_values(['_pop','_vol','_oi'], ascending=[False,False,False]).drop(columns=['_pop','_vol','_oi'])
    return rec_df

# ---------- STREAMLIT UI ----------
st.title("NIFTY Option Chain — Robust Analyzer (fixed)")

st.markdown(
    "Upload the NSE option chain CSV (download from NSE or your broker). "
    "The app auto-detects the header row and column layout. It outputs a friendly summary, charts, and long-only buys."
)

uploaded = st.file_uploader("Upload option-chain CSV (NSE format recommended)", type=["csv"])

if not uploaded:
    st.info("Please upload an option-chain CSV to begin. (If you already uploaded earlier, re-upload here.)")
    st.stop()

# Parse CSV with tolerant parser
try:
    raw_df, header_idx, underlying = parse_nse_style_csv(uploaded)
except Exception as e:
    st.error(f"Failed to parse file: {e}")
    st.stop()

# Map to c_/p_ columns using detected strike index
try:
    df = map_columns_to_ce_pe(raw_df, header_idx)
except Exception as e:
    st.error(f"Failed to map columns: {e}")
    st.stop()

# basic info
st.subheader("Detected column mapping (sample)")
st.write("If you want to inspect raw parsed columns, expand below.")
with st.expander("Show parsed DataFrame columns"):
    st.write(list(df.columns))

# Try find ATM strike (where abs(c_ltp - p_ltp) smallest) or nearest to underlying if found
df['c_ltp'] = df.get('c_ltp', pd.Series([np.nan]*len(df)))
df['p_ltp'] = df.get('p_p_ltp', df.get('p_ltp', pd.Series([np.nan]*len(df))))

# compute straddle (safe)
df['straddle_premium'] = df['c_ltp'].fillna(0) + df['p_ltp'].fillna(0)
# compute abs diff for ATM detection
df['abs_cp_diff'] = (df['c_ltp'].fillna(0) - df['p_ltp'].fillna(0)).abs()
atm_strike = None
if not df['abs_cp_diff'].isna().all():
    try:
        atm_strike = int(df.loc[df['abs_cp_diff'].idxmin(), 'strike'])
    except Exception:
        atm_strike = None

# Show high-level summary
st.subheader("1) Plain-English summary")
col1, col2, col3 = st.columns(3)
col1.metric("Detected ATM (approx)", f"{atm_strike}" if atm_strike else "—")
col2.metric("Underlying (if detected)", f"{underlying:.2f}" if (underlying is not None and not np.isnan(underlying)) else "—")
# max combined OI (proxy max-pain)
df['total_oi'] = df.get('c_oi',0).fillna(0) + df.get('p_oi',0).fillna(0)
max_oi_strike = int(df.loc[df['total_oi'].idxmax(), 'strike']) if df['total_oi'].notna().any() else None
col3.metric("Max combined OI (proxy)", f"{max_oi_strike}" if max_oi_strike else "—")

st.markdown("""
**Interpretation (friendly):**  
- Look for momentum near the ATM / support / resistance bands.  
- Large ΔOI + volume in CE → traders are building calls (momentum or resistance test).  
- Large ΔOI + volume in PE → traders building puts (support or downside momentum).  
We only propose **long buys** — buy CE on upward momentum and buy PE on downward momentum.
""")

# show top ΔOI and volume hotspots
st.subheader("Top activity hotspots")
hot_call = df.sort_values('c_chng_in_oi', ascending=False).head(5)[['strike','c_chng_in_oi','c_oi','c_volume','c_ltp']].fillna('-')
hot_put  = df.sort_values('p_chng_in_oi', ascending=False).head(5)[['strike','p_chng_in_oi','p_oi','p_volume','p_ltp']].fillna('-')
c1, c2 = st.columns(2)
with c1:
    st.markdown("🔺 Calls: biggest ΔOI")
    st.dataframe(hot_call, use_container_width=True)
with c2:
    st.markdown("🔻 Puts: biggest ΔOI")
    st.dataframe(hot_put, use_container_width=True)

# Recommendation block
st.subheader("2) Long-only recommendations (data-backed)")
risk_choice = st.selectbox("Risk profile (affects target/SL)", ["Conservative", "Balanced", "Aggressive"], index=1)
rec_df = recommend_long_trades(df, atm_strike, risk=risk_choice)
if rec_df.empty:
    st.info("No strong long-only setups found using the heuristics. Try another CSV or relax risk profile.")
else:
    st.dataframe(rec_df.reset_index(drop=True), use_container_width=True)

# Straddle premium table + colored % bar
st.subheader("3) Straddle premium (CE+PE) per strike with % move")
# compute % move as combined LTP change if available else NaN
df['c_chng'] = df.get('c_chng', pd.Series([0]*len(df))).fillna(0)
df['p_chng'] = df.get('p_chng', pd.Series([0]*len(df))).fillna(0)
df['straddle_change'] = df['c_chng'] + df['p_chng']
df['straddle_pct'] = np.where(df['straddle_premium']>0, df['straddle_change'] / df['straddle_premium'] * 100.0, np.nan)
strad = df[['strike','c_ltp','p_ltp','straddle_premium','straddle_change','straddle_pct']].copy()
strad = strad.sort_values('strike')

# Bar chart colored by sign
fig = px.bar(strad, x='strike', y='straddle_premium', color=strad['straddle_pct']>0,
             labels={'color':'% change >0'}, title="Straddle premium by strike (green up / red down)")
st.plotly_chart(fig, use_container_width=True)

# show the table with pct column
def color_pct(val):
    if pd.isna(val):
        return ""
    return "color: green" if val>0 else "color: red"
st.write("Percent column shows direction (green = increase, red = decrease).")
st.dataframe(strad.style.format({'c_ltp':'{:.2f}','p_ltp':'{:.2f}','straddle_premium':'{:.2f}','straddle_pct':'{:.2f}'}).applymap(lambda v: 'color: green' if (isinstance(v,(int,float)) and v>0) else ('color: red' if (isinstance(v,(int,float)) and v<0) else ''), subset=['straddle_pct']), use_container_width=True)

# OI & ΔOI charts (bars upward; negative ΔOI shows below 0 automatically)
st.subheader("4) OI and ΔOI charts")
oi_fig = go.Figure()
if 'c_oi' in df.columns:
    oi_fig.add_trace(go.Bar(x=df['strike'], y=df['c_oi'], name='Call OI'))
if 'p_oi' in df.columns:
    oi_fig.add_trace(go.Bar(x=df['strike'], y=df['p_oi'], name='Put OI'))
oi_fig.update_layout(barmode='group', xaxis_title='Strike', yaxis_title='OI')
st.plotly_chart(oi_fig, use_container_width=True)

doi_fig = go.Figure()
if 'c_chng_in_oi' in df.columns:
    doi_fig.add_trace(go.Bar(x=df['strike'], y=df['c_chng_in_oi'], name='Call ΔOI'))
if 'p_chng_in_oi' in df.columns:
    doi_fig.add_trace(go.Bar(x=df['strike'], y=df['p_chng_in_oi'], name='Put ΔOI'))
doi_fig.update_layout(barmode='group', xaxis_title='Strike', yaxis_title='ΔOI')
st.plotly_chart(doi_fig, use_container_width=True)

# Volume charts
st.subheader("5) Volume per strike (CE & PE)")
vol_fig = go.Figure()
if 'c_volume' in df.columns:
    vol_fig.add_trace(go.Bar(x=df['strike'], y=df['c_volume'], name='Call Volume'))
if 'p_volume' in df.columns:
    vol_fig.add_trace(go.Bar(x=df['strike'], y=df['p_volume'], name='Put Volume'))
vol_fig.update_layout(barmode='group', xaxis_title='Strike', yaxis_title='Volume')
st.plotly_chart(vol_fig, use_container_width=True)

# Payoff chart for selected recommended trade or manual strike
st.subheader("6) Payoff chart (buy a strike)")
# build strike list from df
strike_list = sorted(df['strike'].dropna().unique().astype(int).tolist())
if strike_list:
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        buy_choice = st.selectbox("Choose recommended (or manual) strike", ["-- none --"] + [f"{r}:{t}" for r,t in zip(rec_df['Strike'].astype(str) if not rec_df.empty else [], rec_df['Type'] if not rec_df.empty else [])])
    with colB:
        manual_strike = st.selectbox("Or pick any strike", ["-- none --"] + strike_list)
    with colC:
        opt_side = st.selectbox("Option type for payoff", ["CALL","PUT"])
    # decide strike and premium
    chosen_strike = None
    prem = None
    if buy_choice and buy_choice != "-- none --":
        # parse like "24650:CALL"
        parts = buy_choice.split(':')
        chosen_strike = int(parts[0])
        # find premium from rec_df
        if not rec_df.empty:
            rowmatch = rec_df[(rec_df['Strike']==chosen_strike) & (rec_df['Type'].str.contains(parts[1], na=False))]
            if not rowmatch.empty:
                prem = float(rowmatch.iloc[0]['Entry'])
    if chosen_strike is None and manual_strike and manual_strike != "-- none --":
        chosen_strike = int(manual_strike)
    if chosen_strike is not None and prem is None:
        # try find premium in df
        if opt_side == "CALL":
            prem = float(df.loc[df['strike']==chosen_strike, 'c_ltp'].iloc[0]) if not df.loc[df['strike']==chosen_strike, 'c_ltp'].dropna().empty else np.nan
        else:
            prem = float(df.loc[df['strike']==chosen_strike, 'p_ltp'].iloc[0]) if not df.loc[df['strike']==chosen_strike, 'p_ltp'].dropna().empty else np.nan

    if chosen_strike is None:
        st.info("Choose a strike (from recommendations or manual) to see payoff.")
    else:
        if np.isnan(prem):
            st.warning("Premium not available for the chosen strike/side in this file.")
        else:
            # build spot range around ATM or chosen strike
            center = underlying if (underlying is not None and not np.isnan(underlying)) else chosen_strike
            rng = np.arange(int(center-1000), int(center+1001), 25)
            if opt_side == "CALL":
                payoff = np.maximum(rng - chosen_strike, 0) - prem
            else:
                payoff = np.maximum(chosen_strike - rng, 0) - prem
            pay_fig = go.Figure()
            pay_fig.add_trace(go.Scatter(x=rng, y=payoff, mode='lines', name='Payoff'))
            pay_fig.add_hline(y=0, line_dash='dash')
            pay_fig.update_layout(title=f"Payoff at expiry — Buy {opt_side} {chosen_strike} @ {prem:.2f}", xaxis_title='Spot at expiry', yaxis_title='PnL')
            st.plotly_chart(pay_fig, use_container_width=True)
            breakeven = chosen_strike + prem if opt_side=='CALL' else chosen_strike - prem
            st.success(f"Breakeven: {breakeven:.2f} | Max loss = premium ({prem:.2f})")

else:
    st.info("No strikes found in file to show payoff chart.")

# final friendly tips
st.markdown("---")
st.markdown("**Friendly final checklist before entering any buy:**")
st.markdown("- Wait for momentum (price must move in the direction of your buy).")
st.markdown("- Check ΔOI + Volume confirm the move (we prefer both rising).")
st.markdown("- Use the displayed Entry/Target/StopLoss and size per your account risk.")
st.caption("This tool provides heuristics and does not replace risk management or your judgement.")
