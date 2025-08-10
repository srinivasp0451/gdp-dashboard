# app.py
"""
Nifty50 Options Recommender (auto-days-to-expiry + auto Rf rate)
- Two file uploads retained:
    1) Main raw option-chain CSV (CALLS-STRIKE-PUTS)
    2) Optional secondary upload (price history / alt chain)
- Auto-fetch RBI 91-day T-Bill yield (fallback to default if fetch fails)
- Expiry date selected by user via date_input (auto -> days to expiry)
- All other logic (cleaning, scoring, POP) preserved
"""

import streamlit as st
import pandas as pd
import numpy as np
import csv, re, requests, io
from math import log, sqrt, erf
from datetime import date, datetime
from functools import lru_cache

# -------------------------
# Utilities
# -------------------------
def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def bs_d2_prob_itm(spot, strike, iv_pct, days, r=0.06):
    if iv_pct is None or np.isnan(iv_pct) or iv_pct <= 0 or days <= 0:
        return np.nan, np.nan
    sigma = iv_pct / 100.0
    T = days / 365.0
    denom = sigma * sqrt(T)
    if denom == 0:
        return np.nan, np.nan
    d2 = (log(spot / strike) + (r - 0.5 * sigma * sigma) * T) / denom
    p_call = norm_cdf(d2)
    p_put = 1.0 - p_call
    return float(p_call), float(p_put)

def clean_numeric(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if s in ['', '-', 'NaN', 'nan', 'None']:
        return np.nan
    s = re.sub(r'[^\d\.\-]', '', s)
    try:
        if '.' in s:
            return float(s)
        else:
            return int(s)
    except:
        return np.nan

# -------------------------
# Robust parser for messy option chain
# -------------------------
def parse_messy_option_chain(file_obj):
    """Parse NSE-style CALLS-STRIKE-PUTS messy CSV into structured DataFrame."""
    # read with csv.reader - support file-like bytes or text
    # ensure we have strings
    if isinstance(file_obj, (bytes, bytearray)):
        s = io.StringIO(file_obj.decode('utf-8', errors='replace'))
        reader = csv.reader(s)
    else:
        # file_obj may be an UploadedFile from Streamlit
        try:
            # streamlit gives a buffer-like object; decode lines if bytes present
            sample = file_obj.read()
            if isinstance(sample, bytes):
                s = io.StringIO(sample.decode('utf-8', errors='replace'))
            else:
                s = io.StringIO(sample)
            reader = csv.reader(s)
        finally:
            try:
                file_obj.seek(0)
            except:
                pass

    rows = list(reader)
    if len(rows) < 3:
        raise ValueError("CSV looks too short / unexpected format")

    structured = []
    for r in rows[2:]:
        r = list(r)
        if len(r) < 22:
            r = r + [''] * (22 - len(r))
        ce = r[1:11]
        strike = r[11]
        pe = r[12:22]
        structured.append(ce + [strike] + pe)

    cols = [
        'CE_OI','CE_CHNG_IN_OI','CE_VOLUME','CE_IV','CE_LTP','CE_CHNG',
        'CE_BID_QTY','CE_BID','CE_ASK','CE_ASK_QTY',
        'Strike',
        'PE_BID_QTY','PE_BID','PE_ASK','PE_ASK_QTY','PE_CHNG','PE_LTP','PE_IV','PE_VOLUME','PE_CHNG_IN_OI','PE_OI'
    ]
    df = pd.DataFrame(structured, columns=cols)

    # Clean numerics
    for c in df.columns:
        if c == 'Strike':
            df['Strike'] = df['Strike'].apply(clean_numeric)
        else:
            df[c] = df[c].apply(clean_numeric)
    df = df.dropna(subset=['Strike']).reset_index(drop=True)
    try:
        df['Strike'] = df['Strike'].astype(int)
    except:
        df['Strike'] = pd.to_numeric(df['Strike'], errors='coerce').round().astype('Int64')
    return df

# -------------------------
# RBI T-Bill fetch (cached)
# -------------------------
@lru_cache(maxsize=1)
def fetch_rbi_91d_yield():
    """Try to fetch RBI tender or summary page for a recent 91-day T-Bill yield."""
    # NOTE: website structure can change; this is a pragmatic attempt with fallback
    try:
        url = "https://www.rbi.org.in/Scripts/BS_ViewTenders.aspx"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            raise RuntimeError("RBI fetch returned status " + str(resp.status_code))
        text = resp.text
        # Look for patterns like '91 day' or '91-day' and a nearby percent number
        import re
        m = re.search(r"91-?day.*?([0-9]+\.[0-9]+)%", text, re.I | re.S)
        if m:
            return float(m.group(1))
        # fallback: find the first percent-looking number on the page (risky)
        m2 = re.search(r"([0-9]+\.[0-9]+)%", text)
        if m2:
            return float(m2.group(1))
        raise RuntimeError("Could not parse RBI page")
    except Exception as e:
        # Fail quietly — caller should fallback
        return None

# -------------------------
# Scoring & recommendation (core logic)
# -------------------------
def score_and_recommend(df, spot, days_to_expiry=7, side='CE', top_n=3,
                        tp_pct=0.15, sl_pct=0.10, atm_distance=500, r=0.06,
                        vol_weight=0.2, oi_weight=0.4, chng_oi_weight=0.3, dist_weight=0.1):
    df = df.copy()
    prefix = 'CE' if side == 'CE' else 'PE'
    oi_col = f"{prefix}_OI"
    chng_col = f"{prefix}_CHNG_IN_OI"
    ltp_col = f"{prefix}_LTP"
    iv_col = f"{prefix}_IV"

    df['dist_atm'] = np.abs(df['Strike'] - spot)
    df_near = df[df['dist_atm'] <= atm_distance].copy()
    if df_near.empty:
        return pd.DataFrame(), "No strikes in ATM range."

    # normalization functions
    def norm(x):
        x = np.array(x, dtype=float)
        if np.nanmax(x) - np.nanmin(x) == 0:
            return np.zeros_like(x, dtype=float)
        return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-12)

    df_near['oi_norm'] = norm(df_near[oi_col].fillna(0).astype(float))
    df_near['chng_norm'] = norm(df_near[chng_col].fillna(0).astype(float))
    df_near['dist_norm'] = 1.0 - norm(df_near['dist_atm'].astype(float))
    df_near['iv_norm'] = 1.0 - norm(df_near[iv_col].fillna(df_near[iv_col].median()).astype(float))

    df_near['score'] = (
        oi_weight * df_near['oi_norm'] +
        chng_oi_weight * df_near['chng_norm'] +
        dist_weight * df_near['dist_norm'] +
        vol_weight * df_near['iv_norm']
    )

    # compute POP (BS d2)
    p_call_list, p_put_list = [], []
    for _, row in df_near.iterrows():
        p_call, p_put = bs_d2_prob_itm(spot, row['Strike'], row.get(iv_col, np.nan) or np.nan, days_to_expiry, r=r)
        p_call_list.append(p_call)
        p_put_list.append(p_put)
    df_near['p_call_ITM'] = p_call_list
    df_near['p_put_ITM'] = p_put_list
    df_near['prob_ITM'] = df_near['p_call_ITM'] if side == 'CE' else df_near['p_put_ITM']

    df_sorted = df_near.sort_values(by=['score', 'prob_ITM'], ascending=False)

    recs = []
    for _, row in df_sorted.head(top_n).iterrows():
        ltp = float(row.get(ltp_col) or np.nan)
        entry = np.nan if np.isnan(ltp) else ltp
        target = np.nan if np.isnan(ltp) else round(entry * (1 + tp_pct), 2)
        sl = np.nan if np.isnan(ltp) else round(entry * (1 - sl_pct), 2)

        reasons = []
        if (row.get(oi_col) or 0) >= df_near[oi_col].median():
            reasons.append(f"High OI ({int(row.get(oi_col) or 0)})")
        if (row.get(chng_col) or 0) > 0:
            reasons.append(f"OI build-up ({int(row.get(chng_col) or 0)})")
        if (row.get(iv_col) or 0) < (df_near[iv_col].median() if not df_near[iv_col].isna().all() else 999):
            reasons.append(f"Relatively lower IV ({row.get(iv_col)})")
        if row['dist_atm'] <= 50:
            reasons.append("Very close to ATM (<=50 pts)")
        else:
            reasons.append(f"{int(row['dist_atm'])} pts from ATM")

        prob_text = f"{(row['prob_ITM']*100):.1f}%" if not np.isnan(row['prob_ITM']) else "N/A"
        recs.append({
            'Strike': int(row['Strike']),
            'Type': side,
            'LTP': entry,
            'Entry (LTP)': entry,
            'Target': target,
            'Stop Loss': sl,
            'OI': int(row.get(oi_col) or 0),
            'Change_in_OI': int(row.get(chng_col) or 0),
            'IV': float(row.get(iv_col) or np.nan),
            'Prob_ITM': prob_text,
            'Confidence': round(float(row['score']), 3),
            'Reasons': "; ".join(reasons)
        })
    return pd.DataFrame(recs), f"Found {len(df_sorted)} candidates within ±{atm_distance} pts."

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Nifty Options Recommender (Auto)", layout="wide")
st.title("Nifty50 Options — Auto days-to-expiry & Auto Risk-free rate")

st.markdown("""
Upload the raw option chain CSV (CALLS-STRIKE-PUTS).  
Since your CSV doesn't include expiry, **pick the expiry date** below — the app will compute days-to-expiry automatically.  
Risk-free rate is fetched from RBI (91-day T-Bill) by default; you may override it.
""")

# Two file uploaders kept intact
col1, col2 = st.columns([2,1])
with col1:
    uploaded_main = st.file_uploader("Upload main option-chain CSV (raw)", type=['csv'])
with col2:
    uploaded_secondary = st.file_uploader("Upload optional secondary file (price history / alt data)", type=['csv'])

# Expiry input (user picks date since CSV lacks expiry)
st.sidebar.header("Expiry & RF rate")
expiry_date = st.sidebar.date_input("Select expiry date (required)", value=(date.today()), help="Choose the option expiry date (e.g., next Thursday/Friday). Days-to-expiry will be computed from today.")
# compute days to expiry (auto)
today = date.today()
days_to_expiry_auto = (expiry_date - today).days
days_override = st.sidebar.checkbox("Override days-to-expiry manually?", value=False)
days_to_expiry = st.sidebar.number_input("Days to expiry (if override)", value=max(1, days_to_expiry_auto), min_value=1, max_value=365, step=1) if days_override else max(1, days_to_expiry_auto)

# Risk-free rate auto-fetch with override
rf_auto = None
with st.spinner("Fetching RBI 91-day T-Bill yield..."):
    try:
        rf_auto = fetch_rbi_91d_yield()
    except Exception:
        rf_auto = None

default_rf = rf_auto if (rf_auto is not None and 0 < rf_auto < 50) else 6.75
st.sidebar.write(f"Auto-detected 91-day T-Bill yield: {rf_auto if rf_auto else 'N/A'}%")
rf_override = st.sidebar.checkbox("Override risk-free rate?", value=False)
rf_input = st.sidebar.number_input("Risk-free rate (annual %, e.g., 6.75)", value=float(default_rf), min_value=0.0, max_value=15.0, step=0.01) if rf_override else float(default_rf)
rf_used = rf_input

# tuning & parameters
st.sidebar.markdown("### Scoring weights")
oi_weight = st.sidebar.slider("OI weight", 0.0, 1.0, 0.4, 0.05)
chng_oi_weight = st.sidebar.slider("ΔOI weight", 0.0, 1.0, 0.3, 0.05)
dist_weight = st.sidebar.slider("Proximity weight", 0.0, 1.0, 0.1, 0.05)
vol_weight = st.sidebar.slider("IV (lower better) weight", 0.0, 1.0, 0.2, 0.05)

# other inputs
spot_price = st.number_input("Enter current Nifty spot price", value=24363.30, step=0.01)
top_n = st.number_input("Top N per side", min_value=1, max_value=10, value=3, step=1)
atm_distance = st.number_input("ATM range (pts) to consider", min_value=50, max_value=2000, value=500, step=10)
tp_pct = st.number_input("Target % (from premium)", value=15.0, step=0.5) / 100.0
sl_pct = st.number_input("Stop Loss % (from premium)", value=10.0, step=0.5) / 100.0

st.markdown("---")

# Process uploads
if uploaded_main is not None:
    try:
        raw_bytes = uploaded_main.read()
        df_raw = parse_messy_option_chain(raw_bytes)
        st.success("Parsed option chain successfully.")
        st.subheader("Cleaned Option Chain (sample)")
        st.dataframe(df_raw.head(20))

        # Run recommendations
        ce_recs, ce_msg = score_and_recommend(df_raw, float(spot_price), days_to_expiry=int(days_to_expiry),
                                              side='CE', top_n=int(top_n), tp_pct=tp_pct, sl_pct=sl_pct,
                                              atm_distance=int(atm_distance), r=float(rf_used),
                                              vol_weight=vol_weight, oi_weight=oi_weight,
                                              chng_oi_weight=chng_oi_weight, dist_weight=dist_weight)

        pe_recs, pe_msg = score_and_recommend(df_raw, float(spot_price), days_to_expiry=int(days_to_expiry),
                                              side='PE', top_n=int(top_n), tp_pct=tp_pct, sl_pct=sl_pct,
                                              atm_distance=int(atm_distance), r=float(rf_used),
                                              vol_weight=vol_weight, oi_weight=oi_weight,
                                              chng_oi_weight=chng_oi_weight, dist_weight=dist_weight)

        st.markdown("### Recommendations")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🟢 CE (Calls)")
            if ce_recs.empty:
                st.warning(ce_msg)
            else:
                st.dataframe(ce_recs)
                st.download_button("Download CE CSV", ce_recs.to_csv(index=False).encode('utf-8'), file_name="ce_recommendations.csv")
        with c2:
            st.subheader("🔴 PE (Puts)")
            if pe_recs.empty:
                st.warning(pe_msg)
            else:
                st.dataframe(pe_recs)
                st.download_button("Download PE CSV", pe_recs.to_csv(index=False).encode('utf-8'), file_name="pe_recommendations.csv")

        st.markdown("### Explanation & details")
        st.write(f"- Days to expiry used: **{int(days_to_expiry)}** (expiry date: **{expiry_date.isoformat()}**).")
        st.write(f"- Risk-free rate used: **{rf_used:.2f}%** (source: RBI 91-day T-Bill auto-fetch with fallback).")
        st.write("- `Prob_ITM` is a Black-Scholes d2-based approximation of finishing ITM at expiry (risk-neutral).")
        st.write("- `Confidence` is a composite score (OI, ΔOI, proximity, IV). Tune weights in the sidebar.")

        def show_explanations(df_recs):
            for _, row in df_recs.iterrows():
                st.markdown(f"**{row['Type']} {row['Strike']}** — Entry(LTP): {row['Entry (LTP)']}, TP: {row['Target']}, SL: {row['Stop Loss']}")
                st.write(f"- Confidence: {row['Confidence']}, Prob ITM (expiry): {row['Prob_ITM']}")
                st.write(f"- Reasons: {row['Reasons']}")
                st.write("---")
        if not ce_recs.empty:
            st.markdown("#### CE explanations")
            show_explanations(ce_recs)
        if not pe_recs.empty:
            st.markdown("#### PE explanations")
            show_explanations(pe_recs)

        st.success("Done — recommendations generated. Use position sizing & risk rules before trading.")
    except Exception as e:
        st.error("Failed to parse or analyze file: " + str(e))
else:
    st.info("Upload your main option-chain CSV to start. The optional secondary upload is for price history or extra checks.")
