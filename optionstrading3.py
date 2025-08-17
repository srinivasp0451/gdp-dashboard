streamlit_app.py

BankNifty Option Chain Analyzer - Greeks, LTP, PLOI

---------------------------------------------------

Features

- Two file uploads: CURRENT snapshot (required) and PREVIOUS snapshot (optional) for change calculations

- Auto column mapping (case-insensitive) for common names

- Configurable ATM window (+/- N strikes)

- Directional bias (PCR + Notional OI build-up)

- Data-backed long opportunities (Entry/Target/SL + Probability of Profit + reasons)

- Interactive plots: OI, Change in OI, Volume, IV, Greeks (Delta, Gamma, Theta, Vega), Notional OI

- Export recommendations to CSV

Notes: POP is approximated from |Delta| and adjusted by IV z-score. Use with risk management.

import io import math from datetime import datetime

import numpy as np import pandas as pd import plotly.graph_objects as go import streamlit as st

st.set_page_config(page_title="BankNifty Option Chain Analyzer", layout="wide") st.title("BankNifty Option Chain Analyzer - Greeks, OI and Opportunities")

----------------------

Sidebar Controls

----------------------

st.sidebar.header("Controls") window_strikes = st.sidebar.number_input("Strikes around ATM", min_value=5, max_value=60, value=15, step=1) min_volume = st.sidebar.number_input("Minimum Volume filter", min_value=0, value=100, step=50) max_iv = st.sidebar.number_input("Max IV for Buying (optional)", min_value=0.0, value=200.0, step=0.5) score_weights = { 'chg_oi': st.sidebar.slider("Weight: Change in OI", 0.0, 3.0, 1.0, 0.1), 'volume': st.sidebar.slider("Weight: Volume", 0.0, 3.0, 1.0, 0.1), 'iv': st.sidebar.slider("Weight: (Lower) IV", 0.0, 3.0, 0.6, 0.1), 'delta': st.sidebar.slider("Weight: Delta closeness", 0.0, 3.0, 0.8, 0.1), 'price_momentum': st.sidebar.slider("Weight: Premium momentum", 0.0, 3.0, 0.6, 0.1), 'notional_oi': st.sidebar.slider("Weight: Notional OI", 0.0, 3.0, 0.6, 0.1), }

risk_reward = st.sidebar.selectbox("Risk/Reward preset", ["1:1", "1:1.5", "1:2", "Custom"], index=1) if risk_reward == "Custom": rr_risk_pct = st.sidebar.number_input("Stop Loss %", min_value=2.0, max_value=80.0, value=20.0, step=1.0) rr_reward_pct = st.sidebar.number_input("Target %", min_value=2.0, max_value=200.0, value=30.0, step=1.0) else: rr_map = {"1:1": (20.0, 20.0), "1:1.5": (20.0, 30.0), "1:2": (20.0, 40.0)} rr_risk_pct, rr_reward_pct = rr_map[risk_reward]

st.sidebar.markdown("---")

def uploader(label, key): return st.file_uploader(label, type=["csv", "xlsx"], key=key)

cur_file = uploader("Upload CURRENT option chain (with Greeks, LTP/PLOI)", key="cur") prev_file = uploader("Upload PREVIOUS option chain (optional, for change calc)", key="prev")

st.sidebar.markdown("---") show_raw = st.sidebar.checkbox("Show raw normalized data", value=False)

----------------------

Helpers

----------------------

COMMON_MAP = { 'strike': ['strike', 'strike_price', 'strikeprice', 'stk', 'strik'], 'option_type': ['option_type', 'type', 'o_type', 'cp', 'opt_type', 'option', 'cepe', 'call_put'], 'expiry': ['expiry', 'expiry_date', 'exp', 'exp_date'], 'ltp': ['ltp', 'last', 'last_traded_price', 'premium', 'price'], 'iv': ['iv', 'implied_volatility', 'imp_vol'], 'oi': ['oi', 'open_interest', 'openint'], 'change_oi': ['change_in_oi', 'chng_in_oi', 'oi_change', 'chg_oi', 'delta_oi', 'd_oi'], 'volume': ['volume', 'vol'], 'delta': ['delta'], 'gamma': ['gamma'], 'theta': ['theta'], 'vega': ['vega'], 'underlying': ['underlying', 'spot', 'underlying_price', 'index_price', 'bnf_spot'], 'ploi': ['ploi', 'price_oi', 'p_oi', 'notional', 'notional_oi', 'oi_value'] }

CE_ALIASES = {"c", "ce", "call"} PE_ALIASES = {"p", "pe", "put"}

@st.cache_data(show_spinner=False) def load_any(file): if file is None: return None name = file.name.lower() if name.endswith(".xlsx"): df = pd.read_excel(file) else: content = file.getvalue() df = None for enc in ["utf-8", "utf-16", "cp1252", "latin-1"]: try: df = pd.read_csv(io.BytesIO(content), encoding=enc) break except Exception: df = None if df is None: df = pd.read_csv(io.BytesIO(content)) return df

def map_columns(df: pd.DataFrame) -> pd.DataFrame: if df is None: return None lower = {c.lower().strip().replace(" ", "_"): c for c in df.columns}

mapped = {}
for std, aliases in COMMON_MAP.items():
    found = None
    for al in aliases:
        al = al.lower()
        if al in lower:
            found = lower[al]
            break
    if found is None:
        for k, orig in lower.items():
            if std in k:
                found = orig
                break
    mapped[std] = found

out = pd.DataFrame()
for std, src in mapped.items():
    if src is not None and src in df.columns:
        out[std] = df[src]
    else:
        out[std] = np.nan

# Option type normalization
if out['option_type'].notna().any():
    out['option_type'] = out['option_type'].astype(str).str.lower().str.strip()
    def norm_cp(x):
        x = str(x).lower()
        if x in CE_ALIASES or any(x == a for a in CE_ALIASES):
            return 'CE'
        if x in PE_ALIASES or any(x == a for a in PE_ALIASES):
            return 'PE'
        if 'c' == x or 'call' in x:
            return 'CE'
        if 'p' == x or 'put' in x:
            return 'PE'
        return np.nan
    out['option_type'] = out['option_type'].map(norm_cp)

# Numerics
for c in ['strike', 'ltp', 'iv', 'oi', 'change_oi', 'volume', 'delta', 'gamma', 'theta', 'vega', 'ploi', 'underlying']:
    if c in out.columns:
        out[c] = pd.to_numeric(out[c], errors='coerce')

# Expiry to datetime if present
if out['expiry'].notna().any():
    out['expiry'] = pd.to_datetime(out['expiry'], errors='coerce')

# Clean
out = out.dropna(subset=['strike', 'option_type'])
out = out.sort_values('strike').reset_index(drop=True)
return out

----------------------

Load Data

----------------------

cur_df_raw = load_any(cur_file) prev_df_raw = load_any(prev_file)

if cur_df_raw is None: st.info("Upload the CURRENT option chain file to begin.") st.stop()

cur = map_columns(cur_df_raw) prev = map_columns(prev_df_raw) if prev_df_raw is not None else None

if cur is None or cur.empty: st.error("Could not map required columns from the CURRENT file. Ensure it has strike, option_type, and ltp at minimum.") st.stop()

If change_oi missing, derive using previous snapshot OI

if (cur['change_oi'].isna().all() or 'change_oi' not in cur.columns) and prev is not None and 'oi' in prev.columns: base = prev.groupby(['strike','option_type'], as_index=False)['oi'].sum().rename(columns={'oi':'oi_prev'}) cur = cur.merge(base, on=['strike','option_type'], how='left') cur['change_oi'] = cur['oi'] - cur['oi_prev'] else: if 'change_oi' not in cur.columns: cur['change_oi'] = np.nan cur['oi_prev'] = np.nan

Detect underlying spot

spot_candidates = cur['underlying'].dropna() if 'underlying' in cur.columns else pd.Series([], dtype=float) spot = float(spot_candidates.iloc[0]) if len(spot_candidates) else np.nan

ATM detection: nearest strike to spot if spot available, else minimize |CE LTP - PE LTP|, else median strike

if np.isfinite(spot): atm_strike = float(cur.loc[(cur['strike'] - spot).abs().idxmin(), 'strike']) else: ce = cur[cur.option_type=='CE'][['strike','ltp']].rename(columns={'ltp':'ce_ltp'}) pe = cur[cur.option_type=='PE'][['strike','ltp']].rename(columns={'ltp':'pe_ltp'}) both = ce.merge(pe, on='strike', how='inner') if not both.empty: idx = (both['ce_ltp'] - both['pe_ltp']).abs().idxmin() atm_strike = float(both.loc[idx, 'strike']) else: atm_strike = float(cur['strike'].median())

st.subheader("Snapshot and ATM") colA, colB = st.columns(2) colA.metric("Detected ATM Strike", f"{int(round(atm_strike))}") colB.metric("Underlying (if provided)", f"{spot:.2f}" if np.isfinite(spot) else "NA")

Window by strike index (safer than raw value distance)

unique_strikes = np.sort(cur['strike'].unique()) if len(unique_strikes) == 0: st.error("No strikes detected in data.") st.stop()

atm_index = int(np.argmin(np.abs(unique_strikes - atm_strike))) start_idx = max(0, atm_index - int(window_strikes)) end_idx = min(len(unique_strikes) - 1, atm_index + int(window_strikes)) lowS, highS = unique_strikes[start_idx], unique_strikes[end_idx]

win = cur[(cur['strike'] >= lowS) & (cur['strike'] <= highS)].copy().reset_index(drop=True)

Derived fields

win['notional_oi'] = win['oi'] * win['ltp'] win['notional_chg_oi'] = win['change_oi'].fillna(0) * win['ltp'].fillna(0)

PCR

call_oi = win.loc[win.option_type=='CE','oi'].sum() put_oi = win.loc[win.option_type=='PE','oi'].sum() pcr = (put_oi / call_oi) if call_oi else np.nan

Bias via notional change

call_chg_notional = win.loc[win.option_type=='CE','notional_chg_oi'].sum() put_chg_notional = win.loc[win.option_type=='PE','notional_chg_oi'].sum()

bias = "Neutral" if np.isfinite(call_chg_notional) and np.isfinite(put_chg_notional): if put_chg_notional > call_chg_notional * 1.1: bias = "Bullish (Put build-up > Call)" elif call_chg_notional > put_chg_notional * 1.1: bias = "Bearish (Call build-up > Put)"

col1, col2, col3 = st.columns(3) col1.metric("PCR (OI)", f"{pcr:.2f}" if np.isfinite(pcr) else "NA") col2.metric("Change in Notional OI - Calls", f"{call_chg_notional:,.0f}") col3.metric("Change in Notional OI - Puts", f"{put_chg_notional:,.0f}") st.info(f"Directional Bias: {bias}")

----------------------

Plots

----------------------

st.subheader("Exploration - OI, Change in OI, Volume, IV and Greeks")

ce_win = win[win.option_type=='CE'] pe_win = win[win.option_type=='PE']

fig_oi = go.Figure() fig_oi.add_trace(go.Bar(x=ce_win['strike'], y=ce_win['oi'], name='CE OI')) fig_oi.add_trace(go.Bar(x=pe_win['strike'], y=pe_win['oi'], name='PE OI')) fig_oi.update_layout(barmode='group', title='Open Interest by Strike', xaxis_title='Strike', yaxis_title='OI') st.plotly_chart(fig_oi, use_container_width=True)

fig_chg = go.Figure() fig_chg.add_trace(go.Bar(x=ce_win['strike'], y=ce_win['change_oi'], name='CE Change in OI')) fig_chg.add_trace(go.Bar(x=pe_win['strike'], y=pe_win['change_oi'], name='PE Change in OI')) fig_chg.update_layout(barmode='group', title='Change in OI by Strike', xaxis_title='Strike', yaxis_title='Change in OI') st.plotly_chart(fig_chg, use_container_width=True)

fig_vol = go.Figure() fig_vol.add_trace(go.Bar(x=ce_win['strike'], y=ce_win['volume'], name='CE Volume')) fig_vol.add_trace(go.Bar(x=pe_win['strike'], y=pe_win['volume'], name='PE Volume')) fig_vol.update_layout(barmode='group', title='Volume by Strike', xaxis_title='Strike', yaxis_title='Volume') st.plotly_chart(fig_vol, use_container_width=True)

fig_iv = go.Figure() fig_iv.add_trace(go.Scatter(x=ce_win['strike'], y=ce_win['iv'], mode='lines+markers', name='CE IV')) fig_iv.add_trace(go.Scatter(x=pe_win['strike'], y=pe_win['iv'], mode='lines+markers', name='PE IV')) fig_iv.update_layout(title='IV by Strike', xaxis_title='Strike', yaxis_title='IV (%)') st.plotly_chart(fig_iv, use_container_width=True)

for greek in ['delta','gamma','theta','vega']: if greek in win.columns and win[greek].notna().any(): fig = go.Figure() fig.add_trace(go.Scatter(x=ce_win['strike'], y=ce_win[greek], mode='lines+markers', name=f'CE {greek.title()}')) fig.add_trace(go.Scatter(x=pe_win['strike'], y=pe_win[greek], mode='lines+markers', name=f'PE {greek.title()}')) fig.update_layout(title=f'{greek.title()} by Strike', xaxis_title='Strike', yaxis_title=greek.title()) st.plotly_chart(fig, use_container_width=True)

fig_noti = go.Figure() fig_noti.add_trace(go.Bar(x=ce_win['strike'], y=ce_win['notional_oi'], name='CE Notional OI')) fig_noti.add_trace(go.Bar(x=pe_win['strike'], y=pe_win['notional_oi'], name='PE Notional OI')) fig_noti.update_layout(barmode='group', title='Notional OI (OI x LTP)', xaxis_title='Strike', yaxis_title='Notional (Rs)') st.plotly_chart(fig_noti, use_container_width=True)

----------------------

Opportunity Scoring

----------------------

Momentum proxy (if prev provided with LTP)

if prev is not None and 'ltp' in prev.columns and prev['ltp'].notna().any(): mom_base = prev.groupby(['strike','option_type'], as_index=False)['ltp'].mean().rename(columns={'ltp':'ltp_prev'}) win = win.merge(mom_base, on=['strike','option_type'], how='left') win['prem_momentum'] = (win['ltp'] - win['ltp_prev']) / win['ltp_prev'].replace(0, np.nan) else: win['prem_momentum'] = np.nan

Z-scores for ranking

for c in ['change_oi','volume','iv','notional_oi','prem_momentum']: s = win[c].astype(float) std = s.std(skipna=True) if std is None or std == 0 or np.isnan(std): z = s * 0 else: z = (s - s.mean(skipna=True)) / std win[c + '_z'] = z.replace([np.inf, -np.inf], 0).fillna(0)

Delta closeness: prefer ~0.45 for long options

def delta_closeness(row): d = row.get('delta', np.nan) if pd.isna(d): return 0.0 target = 0.45 if row['option_type'] == 'CE': return max(0.0, 1.0 - min(abs(d - target) / target, 1.0)) else: return max(0.0, 1.0 - min(abs(abs(d) - target) / target, 1.0))

win['delta_closeness'] = win.apply(delta_closeness, axis=1)

Base filter

flt = (win['volume'].fillna(0) >= min_volume) & (win['iv'].fillna(0) <= max_iv) win_f = win[flt].copy()

Score

win_f['score'] = ( score_weights['chg_oi'] * win_f['change_oi_z'] + score_weights['volume'] * win_f['volume_z'] + score_weights['iv'] * (-win_f['iv_z']) + score_weights['delta'] * win_f['delta_closeness'] + score_weights['price_momentum'] * win_f['prem_momentum_z'] + score_weights['notional_oi'] * win_f['notional_oi_z'] )

Directional filter by bias

if 'Bullish' in bias: long_side = 'CE' elif 'Bearish' in bias: long_side = 'PE' else: long_side = None

if long_side: cand = win_f[win_f.option_type == long_side] else: # neutral - prefer near ATM both sides (within +/- 6 strikes) near_mask = (win_f['strike'] >= lowS) & (win_f['strike'] <= highS) cand = win_f[near_mask]

reco = cand.sort_values(['score'], ascending=False).copy()

Probability of Profit approximation

def pop_est(row): d = abs(row.get('delta', 0.0)) ivz = row.get('iv_z', 0.0) adj = 1.0 / (1.0 + max(ivz, -0.9)) pop = np.clip(d * adj, 0.05, 0.95) return float(pop)

reco['POP'] = reco.apply(pop_est, axis=1)

Entry/Target/SL

reco['Entry'] = reco['ltp'] reco['SL'] = reco['Entry'] * (1 - rr_risk_pct/100.0) reco['Target'] = reco['Entry'] * (1 + rr_reward_pct/100.0)

Reason string

def reason(row): bits = [] if pd.notna(row.get('change_oi')) and row.get('change_oi') > 0: bits.append('OI build-up') if pd.notna(row.get('prem_momentum')) and row.get('prem_momentum') > 0: bits.append('Premium rising') if row.get('iv_z', 0) < 0: bits.append('IV relatively low') if row.get('delta_closeness', 0) > 0.6: bits.append('Favourable Delta (~0.45)') noti = row.get('notional_oi', 0) if pd.notna(noti): bits.append(f"Notional OI {noti:,.0f}") return ", ".join(bits)

reco['Reason'] = reco.apply(reason, axis=1)

Display top recommendations

st.subheader("Data-backed Buying Opportunities") cols_to_show = ['option_type','strike','ltp','iv','oi','change_oi','volume','delta','gamma','theta','vega','Entry','Target','SL','POP','score','Reason'] show = reco[cols_to_show].head(6).copy() show = show.rename(columns={'ltp':'LTP'})

st.dataframe( show.style.format({ 'LTP':'{:.2f}','iv':'{:.2f}','oi':'{:,.0f}','change_oi':'{:,.0f}','volume':'{:,.0f}', 'delta':'{:.2f}','gamma':'{:.4f}','theta':'{:.2f}','vega':'{:.2f}', 'Entry':'{:.2f}','Target':'{:.2f}','SL':'{:.2f}','POP':'{:.1%}','score':'{:.2f}' }), use_container_width=True )

csv_bytes = show.to_csv(index=False).encode('utf-8') st.download_button( "Download Recommendations (CSV)", data=csv_bytes, file_name=f"bnf_recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime='text/csv' )

----------------------

Summary Narrative

----------------------

st.subheader("What is happening - Auto Summary")

summary = [] summary.append(f"ATM detected at {int(round(atm_strike))} with window +/- {int(window_strikes)} strikes.") if np.isfinite(pcr): if pcr > 1.2: summary.append(f"PCR = {pcr:.2f} (Put heavy) -> Bullish tilt.") elif pcr < 0.8: summary.append(f"PCR = {pcr:.2f} (Call heavy) -> Bearish tilt.") else: summary.append(f"PCR = {pcr:.2f} -> Neutral.")

summary.append(f"Change in Notional OI - Calls: {call_chg_notional:,.0f}, Puts: {put_chg_notional:,.0f} -> {bias}.")

OI walls

def top_walls(df, side, n=3): tmp = df[df.option_type==side].sort_values('oi', ascending=False) return ", ".join([f"{int(r.strike)} ({int(r.oi):,})" for _,r in tmp.head(n).iterrows()])

walls_ce = top_walls(win, 'CE') walls_pe = top_walls(win, 'PE') summary.append(f"Top OI walls - CE: {walls_ce}; PE: {walls_pe}.")

iv_med = win['iv'].median(skipna=True) if np.isfinite(iv_med): summary.append(f"Median IV approx {iv_med:.2f}. Limit set {max_iv}.")

pm = win['prem_momentum'].median(skipna=True) if np.isfinite(pm): summary.append(f"Median premium momentum approx {pm*100:.1f}% vs previous snapshot.")

st.markdown("\n".join([f"- {s}" for s in summary]))

----------------------

Raw Data (optional)

----------------------

if show_raw: st.subheader("Normalized Current Data (window)") st.dataframe(win, use_container_width=True)

st.caption("This tool uses approximations (for example, POP via Delta and IV). Use prudent position sizing.")

