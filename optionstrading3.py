streamlit_app.py

BankNifty Option Chain Analyser with Greeks + LTP/PLOI

------------------------------------------------------

Features

- Two file uploads: CURRENT snapshot (mandatory) and PREVIOUS snapshot (optional) for delta/changes

- Auto column mapping (case-insensitive) for common names

- Configurable ATM window (± N strikes)

- Directional bias (PCR + Notional OI Build-up)

- Data-backed long opportunities (CE for bullish, PE for bearish) with Entry/Target/SL & POP (probability of profit)

- Interactive plots: OI, Change in OI, Volume, IV, Greeks (Δ Γ Θ ν), Notional OI

- Exports recommendations to CSV



NOTE: POP is an approximation using |Delta| as proxy for probability of expiring ITM and adjusted by IV z-score.

Always validate with your risk management.

import io import math import numpy as np import pandas as pd import streamlit as st import plotly.express as px import plotly.graph_objects as go from datetime import datetime

st.set_page_config(page_title="BankNifty Option Chain Analyzer", layout="wide") st.title("📈 BankNifty Option Chain Analyzer — Greeks, OI & Opportunities")

----------------------

Sidebar Controls

----------------------

st.sidebar.header("⚙️ Controls") window_strikes = st.sidebar.number_input("Strikes around ATM", min_value=5, max_value=60, value=15, step=1) min_volume = st.sidebar.number_input("Minimum Volume filter", min_value=0, value=100, step=50) max_iv = st.sidebar.number_input("Max IV for Buying (optional)", min_value=0.0, value=200.0, step=0.5) score_weights = { 'chg_oi': st.sidebar.slider("Weight: ΔOI", 0.0, 3.0, 1.0, 0.1), 'volume': st.sidebar.slider("Weight: Volume", 0.0, 3.0, 1.0, 0.1), 'iv': st.sidebar.slider("Weight: (Lower) IV", 0.0, 3.0, 0.6, 0.1), 'delta': st.sidebar.slider("Weight: Delta closeness", 0.0, 3.0, 0.8, 0.1), 'price_momentum': st.sidebar.slider("Weight: Premium momentum", 0.0, 3.0, 0.6, 0.1), 'notional_oi': st.sidebar.slider("Weight: Notional OI", 0.0, 3.0, 0.6, 0.1), }

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

@st.cache_data(show_spinner=False) def load_any(file): if file is None: return None name = file.name.lower() if name.endswith(".xlsx"): df = pd.read_excel(file) else: # try multiple encodings content = file.getvalue() for enc in ["utf-8", "utf-16", "cp1252", "latin-1"]: try: df = pd.read_csv(io.BytesIO(content), encoding=enc) break except Exception: df = None if df is None: df = pd.read_csv(io.BytesIO(content)) return df

def map_columns(df: pd.DataFrame) -> pd.DataFrame: cols = {c: c for c in df.columns} lower = {c.lower().strip().replace(" ", "_"): c for c in df.columns}

mapped = {}
for std, aliases in COMMON_MAP.items():
    found = None
    for al in aliases:
        al = al.lower()
        if al in lower:
            found = lower[al]
            break
    if found is None:
        # Try contains logic
        for k, orig in lower.items():
            if std in k:
                found = orig
                break
    mapped[std] = found

# Build normalized df with available columns
out = pd.DataFrame()
for std, src in mapped.items():
    if src is not None:
        out[std] = df[src]
    else:
        out[std] = np.nan

# Option type normalization
if out['option_type'].notna().any():
    out['option_type'] = out['option_type'].astype(str).str.lower().str.strip()
    def norm_cp(x):
        x = str(x).lower()
        if any(a == x or a in x.split('/') for a in CE_ALIASES):
            return 'CE'
        if any(a == x or a in x.split('/') for a in PE_ALIASES):
            return 'PE'
        return np.nan
    out['option_type'] = out['option_type'].map(norm_cp)
else:
    # Try to infer from column split (e.g., dataset with CE_* and PE_* columns not supported here)
    pass

# Numerics
for c in ['strike', 'ltp', 'iv', 'oi', 'change_oi', 'volume', 'delta', 'gamma', 'theta', 'vega', 'ploi', 'underlying']:
    if c in out:
        out[c] = pd.to_numeric(out[c], errors='coerce')

# Expiry to datetime if present
if out['expiry'].notna().any():
    out['expiry'] = pd.to_datetime(out['expiry'], errors='coerce')

# Basic clean
out = out.dropna(subset=['strike', 'option_type'])
out = out.sort_values('strike')
return out

----------------------

Load Data

----------------------

cur_df_raw = load_any(cur_file) prev_df_raw = load_any(prev_file)

if cur_df_raw is None: st.info("👆 Upload the CURRENT option chain file to begin.") st.stop()

cur = map_columns(cur_df_raw) prev = map_columns(prev_df_raw) if prev_df_raw is not None else None

Merge previous for change calc if current lacks change_oi

if cur['change_oi'].isna().all() and prev is not None: # compute ΔOI by matching on ['strike','option_type'] base = prev.groupby(['strike','option_type'], as_index=False)['oi'].sum().rename(columns={'oi':'oi_prev'}) cur = cur.merge(base, on=['strike','option_type'], how='left') cur['change_oi'] = cur['oi'] - cur['oi_prev'] else: cur['oi_prev'] = np.nan

Detect underlying

spot_candidates = cur['underlying'].dropna() spot = float(spot_candidates.iloc[0]) if len(spot_candidates) else None

ATM detection (robust): 1) nearest strike to spot if available; otherwise 2) minimize |CE_ltp - PE_ltp|

if spot is not None and not math.isnan(spot): atm_strike = cur['strike'].iloc[(cur['strike'] - spot).abs().argmin()] else: ce = cur[cur.option_type=='CE'][['strike','ltp']].rename(columns={'ltp':'ce_ltp'}) pe = cur[cur.option_type=='PE'][['strike','ltp']].rename(columns={'ltp':'pe_ltp'}) both = ce.merge(pe, on='strike', how='inner') if not both.empty: atm_strike = both.iloc[(both['ce_ltp'] - both['pe_ltp']).abs().argmin()]['strike'] else: atm_strike = cur['strike'].median()

st.subheader("Snapshot & ATM") colA, colB, colC = st.columns(3) colA.metric("Detected ATM Strike", f"{int(atm_strike)}") colB.metric("Underlying (if provided)", f"{spot:.2f}" if spot is not None and not math.isnan(spot) else "—")

Filter to window around ATM

lowS = atm_strike - window_strikes * (cur['strike'].diff().median() or 100) highS = atm_strike + window_strikes * (cur['strike'].diff().median() or 100) win = cur[(cur['strike']>=lowS) & (cur['strike']<=highS)].copy()

Basic derived fields

win['notional_oi'] = win['oi'] * win['ltp'] win['notional_chg_oi'] = win['change_oi'].fillna(0) * win['ltp'].fillna(0)

PCR (around window)

call_oi = win.loc[win.option_type=='CE','oi'].sum() put_oi  = win.loc[win.option_type=='PE','oi'].sum() pcr = (put_oi / call_oi) if call_oi else np.nan

Bias via Δ Notional OI

call_chg_notional = win.loc[win.option_type=='CE','notional_chg_oi'].sum() put_chg_notional  = win.loc[win.option_type=='PE','notional_chg_oi'].sum()

bias = "Neutral" if np.isfinite(call_chg_notional) and np.isfinite(put_chg_notional): if put_chg_notional > call_chg_notional * 1.1: bias = "Bullish (Put build-up > Call)" elif call_chg_notional > put_chg_notional * 1.1: bias = "Bearish (Call build-up > Put)"

col1, col2, col3 = st.columns(3) col1.metric("PCR (OI)", f"{pcr:.2f}" if np.isfinite(pcr) else "—") col2.metric("Δ Notional OI — Calls", f"{call_chg_notional:,.0f}") col3.metric("Δ Notional OI — Puts",  f"{put_chg_notional:,.0f}") st.info(f"Directional Bias: {bias}")

----------------------

Plots

----------------------

st.subheader("Exploration — OI, ΔOI, Volume, IV & Greeks")

Base pivot helpers

ce_win = win[win.option_type=='CE'] pe_win = win[win.option_type=='PE']

fig_oi = go.Figure() fig_oi.add_trace(go.Bar(x=ce_win['strike'], y=ce_win['oi'], name='CE OI')) fig_oi.add_trace(go.Bar(x=pe_win['strike'], y=pe_win['oi'], name='PE OI')) fig_oi.update_layout(barmode='group', title='Open Interest by Strike', xaxis_title='Strike', yaxis_title='OI') st.plotly_chart(fig_oi, use_container_width=True)

fig_chg = go.Figure() fig_chg.add_trace(go.Bar(x=ce_win['strike'], y=ce_win['change_oi'], name='CE ΔOI')) fig_chg.add_trace(go.Bar(x=pe_win['strike'], y=pe_win['change_oi'], name='PE ΔOI')) fig_chg.update_layout(barmode='group', title='Change in OI by Strike', xaxis_title='Strike', yaxis_title='ΔOI') st.plotly_chart(fig_chg, use_container_width=True)

fig_vol = go.Figure() fig_vol.add_trace(go.Bar(x=ce_win['strike'], y=ce_win['volume'], name='CE Volume')) fig_vol.add_trace(go.Bar(x=pe_win['strike'], y=pe_win['volume'], name='PE Volume')) fig_vol.update_layout(barmode='group', title='Volume by Strike', xaxis_title='Strike', yaxis_title='Volume') st.plotly_chart(fig_vol, use_container_width=True)

fig_iv = go.Figure() fig_iv.add_trace(go.Scatter(x=ce_win['strike'], y=ce_win['iv'], mode='lines+markers', name='CE IV')) fig_iv.add_trace(go.Scatter(x=pe_win['strike'], y=pe_win['iv'], mode='lines+markers', name='PE IV')) fig_iv.update_layout(title='IV by Strike', xaxis_title='Strike', yaxis_title='IV (%)') st.plotly_chart(fig_iv, use_container_width=True)

Greeks lines

for greek in ['delta','gamma','theta','vega']: if greek in win.columns and win[greek].notna().any(): fig = go.Figure() fig.add_trace(go.Scatter(x=ce_win['strike'], y=ce_win[greek], mode='lines+markers', name=f'CE {greek.title()}')) fig.add_trace(go.Scatter(x=pe_win['strike'], y=pe_win[greek], mode='lines+markers', name=f'PE {greek.title()}')) fig.update_layout(title=f'{greek.title()} by Strike', xaxis_title='Strike', yaxis_title=greek.title()) st.plotly_chart(fig, use_container_width=True)

Notional OI

fig_noti = go.Figure() fig_noti.add_trace(go.Bar(x=ce_win['strike'], y=ce_win['notional_oi'], name='CE Notional OI')) fig_noti.add_trace(go.Bar(x=pe_win['strike'], y=pe_win['notional_oi'], name='PE Notional OI')) fig_noti.update_layout(barmode='group', title='Notional OI (OI × LTP)', xaxis_title='Strike', yaxis_title='₹ Notional') st.plotly_chart(fig_noti, use_container_width=True)

----------------------

Opportunity Scoring

----------------------

Momentum proxy (if prev provided with LTP)

if prev is not None and 'ltp' in prev.columns and prev['ltp'].notna().any(): mom_base = prev.groupby(['strike','option_type'], as_index=False)['ltp'].mean().rename(columns={'ltp':'ltp_prev'}) win = win.merge(mom_base, on=['strike','option_type'], how='left') win['prem_momentum'] = (win['ltp'] - win['ltp_prev']) / win['ltp_prev'].replace(0, np.nan) else: win['prem_momentum'] = np.nan

Z-scores for ranking

for c in ['change_oi','volume','iv','notional_oi','prem_momentum']: s = win[c].astype(float) z = (s - s.mean(skipna=True)) / (s.std(skipna=True) if s.std(skipna=True) not in [0, None, np.nan] else 1) win[c+"_z"] = z.replace([np.inf,-np.inf], 0).fillna(0)

Delta closeness target: for CE prefer 0.3–0.6, for PE −0.6–−0.3

def delta_closeness(row): d = row.get('delta', np.nan) if pd.isna(d): return 0.0 if row['option_type']=='CE': return 1.0 - min(abs(d-0.45)/0.45, 1.0) else: return 1.0 - min(abs(abs(d)-0.45)/0.45, 1.0)

win['delta_closeness'] = win.apply(delta_closeness, axis=1)

Base filter

flt = (win['volume'] >= min_volume) & (win['iv'] <= max_iv) win_f = win[flt].copy()

Score

win_f['score'] = ( score_weights['chg_oi'] * win_f['change_oi_z'] + score_weights['volume'] * win_f['volume_z'] + score_weights['iv'] * (-win_f['iv_z']) + score_weights['delta'] * win_f['delta_closeness'] + score_weights['price_momentum'] * win_f['prem_momentum_z'] + score_weights['notional_oi'] * win_f['notional_oi_z'] )

Directional filter by bias

if 'Bullish' in bias: long_side = 'CE' elif 'Bearish' in bias: long_side = 'PE' else: # Neutral: pick closer to ATM both sides but keep top by score long_side = None

if long_side: cand = win_f[win_f.option_type==long_side] else: # both sides, but prefer within ±6 strikes from ATM spacing = int(round(win['strike'].diff().median() or 100)) near = (win_f['strike'] >= atm_strike-6spacing) & (win_f['strike'] <= atm_strike+6spacing) cand = win_f[near]

Rank within each side

reco = cand.sort_values(['score'], ascending=False).copy()

POP approximation: |Delta| adjusted by IV z-score

def pop_est(row): d = abs(row.get('delta', 0.0)) adj = 1.0 / (1.0 + max(row.get('iv_z',0.0), -0.9)) # clamp pop = np.clip(d * adj, 0.05, 0.95) return pop

reco['POP'] = reco.apply(pop_est, axis=1)

Entry/Target/SL

reco['Entry'] = reco['ltp'] reco['SL'] = reco['Entry'] * (1 - rr_risk_pct/100.0) reco['Target'] = reco['Entry'] * (1 + rr_reward_pct/100.0)

Reason string

def reason(row): bits = [] if row.get('change_oi', np.nan) > 0: bits.append("OI build-up") if row.get('prem_momentum', np.nan) > 0: bits.append("Premium rising") if row.get('iv_z', 0) < 0: bits.append("IV relatively low") if row.get('delta_closeness', 0) > 0.6: bits.append("Favorable Delta (~0.45)") bits.append(f"Notional OI: {row.get('notional_oi',0):.0f}") return ", ".join(bits)

reco['Reason'] = reco.apply(reason, axis=1)

Display top recommendations

st.subheader("🎯 Data-backed Buying Opportunities")

cols_to_show = ['option_type','strike','ltp','iv','oi','change_oi','volume','delta','gamma','theta','vega','Entry','Target','SL','POP','score','Reason'] show = reco[cols_to_show].head(6).copy() show = show.rename(columns={'ltp':'LTP'})

st.dataframe(show.style.format({ 'LTP':'{:.2f}','iv':'{:.2f}','oi':'{:,.0f}','change_oi':'{:,.0f}','volume':'{:,.0f}', 'delta':'{:.2f}','gamma':'{:.4f}','theta':'{:.2f}','vega':'{:.2f}', 'Entry':'{:.2f}','Target':'{:.2f}','SL':'{:.2f}','POP':'{:.1%}','score':'{:.2f}' }).highlight_max(subset=['score','POP'], color='#e6ffe6'), use_container_width=True)

Export

csv_bytes = show.to_csv(index=False).encode('utf-8') st.download_button("💾 Download Recommendations (CSV)", data=csv_bytes, file_name=f"bnf_recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime='text/csv')

----------------------

Summary Narrative

----------------------

st.subheader("🧭 What’s happening — Auto Summary")

summary = [] summary.append(f"ATM detected at {int(atm_strike)} with window ±{window_strikes} strikes.") if np.isfinite(pcr): if pcr > 1.2: summary.append(f"PCR = {pcr:.2f} (Put-heavy) → Bullish tilt.") elif pcr < 0.8: summary.append(f"PCR = {pcr:.2f} (Call-heavy) → Bearish tilt.") else: summary.append(f"PCR = {pcr:.2f} → Neutral.")

summary.append(f"Δ Notional OI — Calls: {call_chg_notional:,.0f}, Puts: {put_chg_notional:,.0f} → {bias}.")

OI walls

def top_walls(df, side, n=3): tmp = df[df.option_type==side].sort_values('oi', ascending=False) return ", ".join([f"{int(r.strike)} ({int(r.oi):,})" for _,r in tmp.head(n).iterrows()])

walls_ce = top_walls(win, 'CE') walls_pe = top_walls(win, 'PE') summary.append(f"Top OI walls — CE: {walls_ce}; PE: {walls_pe}.")

IV regime

iv_med = win['iv'].median(skipna=True) if np.isfinite(iv_med): summary.append(f"Median IV ≈ {iv_med:.2f}. {'Rich' if iv_med>max_iv else 'Within preset'} vs your limit {max_iv}.")

Momentum color

pm = win['prem_momentum'].median(skipna=True) if np.isfinite(pm): summary.append(f"Median premium momentum ≈ {pm*100:.1f}% (vs previous snapshot).")

st.markdown("\n".join([f"- {s}" for s in summary]))

----------------------

Raw Data (optional)

----------------------

if show_raw: st.subheader("Normalized Current Data (window)") st.dataframe(win, use_container_width=True)

st.caption("⚠️ This tool provides an analytical view using approximations (e.g., POP via Delta & IV). Use with prudent position sizing and your own discretion.")

