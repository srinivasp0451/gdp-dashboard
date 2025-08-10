# app.py
import streamlit as st
import pandas as pd
import numpy as np
import csv
import re
from math import log, sqrt, exp, erf

# -------------------------
# Utility helpers
# -------------------------
def norm_cdf(x):
    # normal CDF using erf
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def bs_d2_prob_itm(spot, strike, iv_pct, days, r=0.06):
    """
    Approximate risk-neutral probability of finishing ITM.
    iv_pct: implied volatility in percent (e.g. 20 for 20%)
    days: days to expiry
    returns: prob_call_ITM, prob_put_ITM
    """
    if iv_pct is None or np.isnan(iv_pct) or iv_pct <= 0 or days <= 0:
        return np.nan, np.nan
    sigma = iv_pct / 100.0
    T = days / 365.0
    # avoid zero sigma
    denom = sigma * sqrt(T)
    if denom == 0:
        return np.nan, np.nan
    d2 = (log(spot / strike) + (r - 0.5 * sigma * sigma) * T) / denom
    p_call = norm_cdf(d2)   # probability call finishes ITM
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
# Robust CSV parser for NSE option chain
# -------------------------
def parse_messy_option_chain(file_obj):
    """
    Accept a file-like object (or path). This will parse the mixed-layout CSV where header
    is like: ['', 'OI','CHNG IN OI',...,'STRIKE',...,'CHNG IN OI','OI','']
    and rows contain CALLS ... STRIKE ... PUTS columns.
    Returns a cleaned pandas DataFrame with columns:
      Strike, CE_OI, CE_CHNG_IN_OI, CE_VOLUME, CE_IV, CE_LTP, CE_CHNG, ... (and PE_*)
    """
    # read raw using csv.reader
    reader = csv.reader((line.decode('utf-8', errors='replace') if isinstance(line, bytes) else line for line in file_obj))
    rows = list(reader)
    if len(rows) < 3:
        raise ValueError("File seems too short / invalid format")
    header = rows[1]
    # expected pattern: idx 1..10 are CE, 11 is STRIKE, 12..21 are PE
    # We'll build rows accordingly and then clean numeric columns.
    structured = []
    for r in rows[2:]:
        # pad/truncate to at least 22 tokens
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
    # clean numeric columns
    for c in df.columns:
        if c == 'Strike':
            df['Strike'] = df['Strike'].apply(clean_numeric)
        else:
            df[c] = df[c].apply(clean_numeric)
    # drop rows without strike
    df = df.dropna(subset=['Strike']).reset_index(drop=True)
    # cast strike to integer if reasonable
    try:
        df['Strike'] = df['Strike'].astype(int)
    except:
        df['Strike'] = pd.to_numeric(df['Strike'], errors='coerce').round().astype('Int64')
    return df

# -------------------------
# Recommendation logic
# -------------------------
def score_and_recommend(df, spot, days_to_expiry=7, side='CE', top_n=3,
                        tp_pct=0.15, sl_pct=0.10, atm_distance=500, r=0.06,
                        vol_weight=0.2, oi_weight=0.4, chng_oi_weight=0.3, dist_weight=0.1):
    """
    side: 'CE' or 'PE'
    Returns DataFrame of top recommendations and explanation strings.
    """
    df = df.copy()
    # columns we expect: CE_OI, CE_CHNG_IN_OI, CE_LTP, CE_IV, PE_* etc.
    prefix = 'CE' if side == 'CE' else 'PE'
    oi_col = f"{prefix}_OI"
    chng_col = f"{prefix}_CHNG_IN_OI"
    ltp_col = f"{prefix}_LTP"
    iv_col = f"{prefix}_IV"

    # Keep strikes within atm_distance
    df['dist_atm'] = np.abs(df['Strike'] - spot)
    df_near = df[df['dist_atm'] <= atm_distance].copy()
    if df_near.empty:
        return pd.DataFrame(), "No strikes found within ATM distance."

    # normalize factors
    # avoid divide by zero
    def norm(x):
        if np.nanmax(x) - np.nanmin(x) == 0:
            return np.zeros_like(x, dtype=float)
        return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-12)

    # we want higher OI, higher chng_in_OI for buy candidates; lower IV is better for buying
    df_near['oi_norm'] = norm(df_near[oi_col].fillna(0).astype(float))
    df_near['chng_norm'] = norm(df_near[chng_col].fillna(0).astype(float))
    # inverse instrument for distance: closer to ATM => higher score
    df_near['dist_norm'] = 1.0 - norm(df_near['dist_atm'].astype(float))
    # lower IV -> higher score
    df_near['iv_norm'] = 1.0 - norm(df_near[iv_col].fillna(df_near[iv_col].median()).astype(float))

    # composite score
    df_near['score'] = (
        oi_weight * df_near['oi_norm'] +
        chng_oi_weight * df_near['chng_norm'] +
        dist_weight * df_near['dist_norm'] +
        vol_weight * df_near['iv_norm']
    )

    # probability of ITM from IV
    p_call_list = []
    p_put_list = []
    for _, row in df_near.iterrows():
        p_call, p_put = bs_d2_prob_itm(spot, row['Strike'], row.get(iv_col, np.nan) or np.nan, days_to_expiry, r=r)
        p_call_list.append(p_call)
        p_put_list.append(p_put)
    df_near['p_call_ITM'] = p_call_list
    df_near['p_put_ITM'] = p_put_list

    # choose appropriate probability
    if side == 'CE':
        df_near['prob_ITM'] = df_near['p_call_ITM']
    else:
        df_near['prob_ITM'] = df_near['p_put_ITM']

    df_sorted = df_near.sort_values(by=['score', 'prob_ITM'], ascending=False)

    # build recommendations
    recs = []
    for _, row in df_sorted.head(top_n).iterrows():
        ltp = float(row.get(ltp_col) or np.nan)
        if np.isnan(ltp):
            entry = np.nan
            target = np.nan
            sl = np.nan
        else:
            entry = ltp
            target = round(entry * (1 + tp_pct), 2)
            sl = round(entry * (1 - sl_pct), 2)

        # Reasoning bullets
        reasons = []
        if (row.get(oi_col) or 0) >= df_near[oi_col].median():
            reasons.append(f"High OI ({int(row.get(oi_col) or 0)}) near this strike")
        if (row.get(chng_col) or 0) > 0:
            reasons.append(f"Positive OI change ({int(row.get(chng_col) or 0)}) — fresh build-up")
        if (row.get(iv_col) or 0) < (df_near[iv_col].median() if not df_near[iv_col].isna().all() else 999):
            reasons.append(f"Relatively lower IV ({row.get(iv_col)}) → cheaper premium")
        if row['dist_atm'] <= 50:
            reasons.append("Very close to ATM (<=50) — better liquidity & delta")
        else:
            reasons.append(f"{int(row['dist_atm'])} pts from ATM")

        prob_text = f"{(row['prob_ITM']*100):.1f}%" if not np.isnan(row['prob_ITM']) else "N/A"
        confidence = float(row['score'])  # 0..1 normalized-ish
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
            'Confidence': round(confidence, 3),
            'Reasons': "; ".join(reasons)
        })

    return pd.DataFrame(recs), f"Found {len(df_sorted)} candidate strikes within ±{atm_distance} pts."

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Nifty50 Options Recommender", layout="wide")
st.title("🧠 Nifty50 Options — Auto-clean + Trade Recommendations")

st.markdown("""
Upload the raw option-chain CSV (same layout as NSE/Upstox mobile screenshots export).
The app will auto-clean it and suggest CE / PE buying opportunities with Entry / Target / SL,
confidence score, probability of finishing ITM (approx from IV) and human-readable logic.
""")

colA, colB = st.columns([2,1])
with colA:
    uploaded = st.file_uploader("Upload raw option-chain CSV (CALLS-STRIKE-PUTS format)", type=['csv'])
with colB:
    spot_price = st.number_input("Spot price (Nifty)", value=24363.30, step=0.01)
    days_to_expiry = st.number_input("Days to expiry", value=7, step=1)
    top_n = st.number_input("Top N per side", value=3, min_value=1, max_value=10, step=1)
    atm_distance = st.number_input("ATM range (pts) to consider", value=500, step=10)
    tp_pct = st.number_input("Target (TP) % (from premium)", value=15.0, step=0.5) / 100.0
    sl_pct = st.number_input("Stop Loss % (from premium)", value=10.0, step=0.5) / 100.0
    r = st.number_input("Risk-free rate (annual, e.g. 0.06)", value=0.06, step=0.005)

st.markdown("---")
st.sidebar.header("Scoring weights (modify to tune)")
oi_weight = st.sidebar.slider("OI weight", 0.0, 1.0, 0.4, 0.05)
chng_oi_weight = st.sidebar.slider("Chg-in-OI weight", 0.0, 1.0, 0.3, 0.05)
dist_weight = st.sidebar.slider("Proximity weight", 0.0, 1.0, 0.1, 0.05)
vol_weight = st.sidebar.slider("IV (lower better) weight", 0.0, 1.0, 0.2, 0.05)

# Optional: Upload price history CSV to compute SMA/trend
st.sidebar.markdown("### Price confirmation (optional)")
price_file = st.sidebar.file_uploader("Upload price CSV (cols: datetime, close)", type=['csv'])
trend_confirm = False
if price_file is not None:
    try:
        price_df = pd.read_csv(price_file, parse_dates=[0])
        price_df = price_df.dropna(subset=[price_df.columns[1]])
        price_col = price_df.columns[1]
        sma_short = st.sidebar.number_input("SMA short (period)", value=20, step=1)
        sma_long = st.sidebar.number_input("SMA long (period)", value=50, step=1)
        price_df['sma_short'] = price_df[price_col].rolling(sma_short).mean()
        price_df['sma_long'] = price_df[price_col].rolling(sma_long).mean()
        last_close = float(price_df[price_col].iloc[-1])
        trend_confirm = last_close > price_df['sma_long'].iloc[-1]
        st.sidebar.write(f"Last close: {last_close:.2f}. SMA{ sma_long }: {price_df['sma_long'].iloc[-1]:.2f}")
    except Exception as e:
        st.sidebar.error("Price CSV parsing failed: " + str(e))

if uploaded is not None:
    try:
        df_raw = parse_messy_option_chain(uploaded)
        st.success("Option-chain parsed successfully.")
        st.subheader("Sample of cleaned option chain")
        st.dataframe(df_raw.head(20))

        # run recommendation for both sides
        ce_recs, ce_msg = score_and_recommend(df_raw, spot_price, days_to_expiry, side='CE', top_n=top_n,
                                             tp_pct=tp_pct, sl_pct=sl_pct, atm_distance=atm_distance, r=r,
                                             vol_weight=vol_weight, oi_weight=oi_weight,
                                             chng_oi_weight=chng_oi_weight, dist_weight=dist_weight)
        pe_recs, pe_msg = score_and_recommend(df_raw, spot_price, days_to_expiry, side='PE', top_n=top_n,
                                             tp_pct=tp_pct, sl_pct=sl_pct, atm_distance=atm_distance, r=r,
                                             vol_weight=vol_weight, oi_weight=oi_weight,
                                             chng_oi_weight=chng_oi_weight, dist_weight=dist_weight)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🟢 CE (Call) Buying Opportunities")
            if ce_recs.empty:
                st.warning(ce_msg)
            else:
                st.dataframe(ce_recs)
        with col2:
            st.subheader("🔴 PE (Put) Buying Opportunities")
            if pe_recs.empty:
                st.warning(pe_msg)
            else:
                st.dataframe(pe_recs)

        st.markdown("### Selected trade logic & explanations")
        st.write("""
        Each recommendation includes:
        - **Prob_ITM**: approximate chance the option finishes ITM (BS d2 with IV).
        - **Confidence**: composite score (0..1) combining OI, change-in-OI, proximity to ATM and IV.
        - **Reasons**: human-readable bullets derived from the data (high OI, fresh OI build-up, proximity).
        """)
        # Show detail for the top recs with natural language
        def show_explanations(df_recs):
            for i, row in df_recs.iterrows():
                st.markdown(f"**{row['Type']} {row['Strike']}**  — Entry(LTP): {row['Entry (LTP)']}, TP: {row['Target']}, SL: {row['Stop Loss']}")
                st.write(f"- Confidence: {row['Confidence']}, Prob ITM: {row['Prob_ITM']}")
                st.write(f"- Reasons: {row['Reasons']}")
                # trend confirmation note
                if price_file is not None:
                    if row['Type'] == 'CE' and trend_confirm:
                        st.write("  - Price trend confirms bullish bias (price above long SMA).")
                    if row['Type'] == 'PE' and not trend_confirm:
                        st.write("  - Price trend confirms bearish bias (price below long SMA).")
                st.write("---")

        if not ce_recs.empty:
            st.markdown("#### CE explanations")
            show_explanations(ce_recs)
        if not pe_recs.empty:
            st.markdown("#### PE explanations")
            show_explanations(pe_recs)

        st.success("Recommendations ready. Use appropriate position sizing; these are algorithmic signals, not investment advice.")
        st.info("Tip: adjust scoring weights on the sidebar to tune aggressiveness (give more weight to change-in-OI to favor fresh build-ups).")

        # allow CSV download of recs
        def df_to_csv_download(df):
            return df.to_csv(index=False).encode('utf-8')
        if not ce_recs.empty:
            st.download_button("Download CE recommendations CSV", df_to_csv_download(ce_recs), file_name="ce_recommendations.csv")
        if not pe_recs.empty:
            st.download_button("Download PE recommendations CSV", df_to_csv_download(pe_recs), file_name="pe_recommendations.csv")

    except Exception as e:
        st.error("Failed to parse / analyze file: " + str(e))
else:
    st.info("Upload a raw option-chain CSV to begin. If you don't have one, upload the cleaned CSV I produced earlier.")
