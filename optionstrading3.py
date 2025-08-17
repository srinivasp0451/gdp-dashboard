# streamlit_app.py
# BankNifty Option Chain Analyzer – Two uploads, manual spot, plots, heatmaps, and recommendations.

import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ========================
# Page setup & sidebar
# ========================
st.set_page_config(page_title="BankNifty Option Chain Analyzer", layout="wide")
st.title("BankNifty Option Chain Analyzer (Greeks, OI, PLOI)")

st.sidebar.header("Controls")
n_window = st.sidebar.number_input("Strikes around ATM (+/- N)", min_value=5, max_value=60, value=15, step=1)
min_volume = st.sidebar.number_input("Minimum Volume filter", min_value=0, value=100, step=50)
max_iv = st.sidebar.number_input("Max IV for Buying (optional)", min_value=0.0, value=200.0, step=0.5)

risk_reward = st.sidebar.selectbox("Risk/Reward preset", ["1:1", "1:1.5", "1:2", "Custom"], index=1)
if risk_reward == "Custom":
    risk_pct = st.sidebar.number_input("Stop Loss %", min_value=2.0, max_value=80.0, value=20.0, step=1.0)
    reward_pct = st.sidebar.number_input("Target %", min_value=2.0, max_value=200.0, value=30.0, step=1.0)
else:
    rr_map = {"1:1": (20.0, 20.0), "1:1.5": (20.0, 30.0), "1:2": (20.0, 40.0)}
    risk_pct, reward_pct = rr_map[risk_reward]

st.sidebar.markdown("---")
w_chg_oi = st.sidebar.slider("Weight: Change in OI", 0.0, 3.0, 1.0, 0.1)
w_vol    = st.sidebar.slider("Weight: Volume", 0.0, 3.0, 1.0, 0.1)
w_iv     = st.sidebar.slider("Weight: (Lower) IV", 0.0, 3.0, 0.6, 0.1)
w_delta  = st.sidebar.slider("Weight: Delta closeness", 0.0, 3.0, 0.8, 0.1)
w_momo   = st.sidebar.slider("Weight: Premium momentum", 0.0, 3.0, 0.6, 0.1)
w_notioi = st.sidebar.slider("Weight: Notional OI", 0.0, 3.0, 0.6, 0.1)

st.sidebar.markdown("---")
manual_spot = st.sidebar.number_input("Manual Spot (leave 0 to auto)", min_value=0.0, value=0.0, step=50.0)
show_raw = st.sidebar.checkbox("Show normalized raw window data", value=False)

# ========================
# Uploads
# ========================
st.subheader("Upload Files")
col_up1, col_up2 = st.columns(2)
cur_file = col_up1.file_uploader("CURRENT option chain (with Greeks/LTP/PLOI)", type=["csv", "xlsx"], key="cur")
prev_file = col_up2.file_uploader("PREVIOUS snapshot (optional, for change/momentum)", type=["csv", "xlsx"], key="prev")

# ========================
# Helpers
# ========================
CE_ALIASES = {"c", "ce", "call"}
PE_ALIASES = {"p", "pe", "put"}

COMMON_MAP = {
    "strike":      ["strike", "strike_price", "strikeprice", "strikeprice "],
    "option_type": ["option_type", "type", "opt_type", "cp", "option", "call_put", "cepe"],
    "expiry":      ["expiry", "expiry_date", "exp", "expdate"],
    "ltp":         ["ltp", "last", "last_traded_price", "premium", "price", "c_ltp", "p_ltp"], # wide handled later
    "iv":          ["iv", "implied_volatility", "imp_vol", "c_iv", "p_iv"],
    "oi":          ["oi", "open_interest", "openint", "c_oi", "p_oi"],
    "change_oi":   ["change_in_oi", "chng_in_oi", "oi_change", "chg_oi", "delta_oi", "d_oi", "c_chng_in_oi", "p_chng_in_oi"],
    "volume":      ["volume", "vol", "c_volume", "p_volume"],
    "delta":       ["delta", "c_delta", "p_delta"],
    "gamma":       ["gamma", "c_gamma", "p_gamma"],
    "theta":       ["theta", "c_theta", "p_theta"],
    "vega":        ["vega", "c_vega", "p_vega"],
    "underlying":  ["underlying", "spot", "underlying_price", "index_price", "bnf_spot", "underlyingvalue"],
    "ploi":        ["ploi", "price_oi", "p_oi", "notional", "notional_oi", "oi_value"],
}

def read_any(uploaded):
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded)
    # try encodings for csv
    content = uploaded.getvalue()
    for enc in ["utf-8", "utf-16", "cp1252", "latin-1"]:
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc)
        except Exception:
            pass
    return pd.read_csv(io.BytesIO(content))

def normalize_cols(df):
    df2 = df.copy()
    df2.columns = [c.strip().lower().replace(" ", "_") for c in df2.columns]
    return df2

def wide_to_long(df):
    """
    Accepts two possible shapes:
    1) Long: has 'option_type' (CE/PE) and one column per metric.
    2) Wide: columns prefixed with c_ and p_ (e.g., c_oi, p_oi). We split into long.
    Returns standardized long columns:
    ['strike','option_type','ltp','iv','oi','change_oi','volume','delta','gamma','theta','vega','expiry','underlying','ploi']
    """
    d = normalize_cols(df)
    cols = d.columns.tolist()

    # candidate strike column
    strike_col = None
    for k in ["strike_price", "strikeprice", "strike"]:
        if k in cols:
            strike_col = k
            break
    if strike_col is None:
        # last resort: a column containing "strike"
        strike_col = next((c for c in cols if "strike" in c), None)
    if strike_col is None:
        raise ValueError("No strike column found. Expected like strike_price / strikePrice / strike.")

    # do we have option_type?
    has_opt_type = any(c in cols for c in ["option_type", "type", "cp", "option", "call_put", "cepe"])

    # detect if wide (c_/p_ columns present)
    is_wide = any(c.startswith("c_") for c in cols) and any(c.startswith("p_") for c in cols)

    # underlying if present
    und_col = next((c for c in cols if c in ["underlying", "spot", "underlying_price", "index_price", "bnf_spot", "underlyingvalue"]), None)

    if not is_wide and has_opt_type:
        # Already long; map to standard names
        out = pd.DataFrame()
        out["strike"] = pd.to_numeric(d[strike_col], errors="coerce")
        # map option type to CE/PE
        def norm_cp(x):
            s = str(x).lower().strip()
            if s in CE_ALIASES or "call" in s or s == "c":
                return "CE"
            if s in PE_ALIASES or "put" in s or s == "p":
                return "PE"
            return np.nan
        optcol = next(c for c in ["option_type", "type", "cp", "option", "call_put", "cepe"] if c in cols)
        out["option_type"] = d[optcol].map(norm_cp)

        # colonize metrics
        for tgt, aliases in COMMON_MAP.items():
            if tgt in ["strike", "option_type"]:
                continue
            found = None
            for a in aliases:
                a2 = a.lower().replace(" ", "_")
                if a2 in d.columns:
                    found = a2
                    break
            if found is None:
                # also try contains
                for c in d.columns:
                    if tgt in c:
                        found = c
                        break
            out[tgt] = pd.to_numeric(d.get(found, np.nan), errors="coerce")
        if und_col:
            out["underlying"] = pd.to_numeric(d[und_col], errors="coerce")
        return out.dropna(subset=["strike", "option_type"])

    # Wide shape: split CE/PE rows
    base_cols = [c for c in cols if not (c.startswith("c_") or c.startswith("p_"))]
    base = d[base_cols].copy()
    base = base.rename(columns={strike_col: "strike"})
    base["strike"] = pd.to_numeric(base["strike"], errors="coerce")

    def side_frame(prefix, side):
        fcols = [c for c in cols if c.startswith(prefix)]
        sub = base.copy()
        for c in fcols:
            metric = c[len(prefix):]  # e.g., "oi" from "c_oi"
            sub[metric] = pd.to_numeric(d[c], errors="coerce")
        sub["option_type"] = side
        return sub

    ce = side_frame("c_", "CE")
    pe = side_frame("p_", "PE")
    out = pd.concat([ce, pe], ignore_index=True)

    # rename common
    rename_map = {
        "strike": "strike",
        "ltp": "ltp", "iv": "iv", "oi": "oi", "chng_in_oi": "change_oi", "change_in_oi": "change_oi",
        "volume": "volume", "delta": "delta", "gamma": "gamma", "theta": "theta", "vega": "vega",
        "ploi": "ploi"
    }
    for old, new in list(rename_map.items()):
        if old in out.columns and new not in out.columns:
            out[new] = out[old]
    if "change_oi" not in out.columns and "chnginoi" in out.columns:
        out["change_oi"] = out["chnginoi"]

    if und_col and "underlying" not in out.columns:
        out["underlying"] = pd.to_numeric(d[und_col], errors="coerce")

    needed = ["strike", "option_type", "ltp", "iv", "oi", "change_oi", "volume", "delta", "gamma", "theta", "vega", "ploi", "underlying"]
    for c in needed:
        if c not in out.columns:
            out[c] = np.nan

    return out.dropna(subset=["strike", "option_type"]).sort_values(["strike", "option_type"]).reset_index(drop=True)

def zscore(series):
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], 0).fillna(0)

# ========================
# Load & normalize
# ========================
if cur_file is None:
    st.info("Upload the CURRENT option chain file to begin.")
    st.stop()

cur_raw = read_any(cur_file)
prev_raw = read_any(prev_file) if prev_file is not None else None

try:
    cur = wide_to_long(cur_raw)
except Exception as e:
    st.error(f"Could not normalize CURRENT file: {e}")
    st.stop()

prev = None
if prev_raw is not None:
    try:
        prev = wide_to_long(prev_raw)
    except Exception as e:
        st.warning(f"PREVIOUS file could not be normalized ({e}). Momentum/ΔOI from previous will be skipped.")
        prev = None

# If change_oi missing but previous available, derive it
if ("change_oi" not in cur.columns or cur["change_oi"].isna().all()) and prev is not None and "oi" in prev.columns:
    base_prev = prev.groupby(["strike", "option_type"], as_index=False)["oi"].sum().rename(columns={"oi": "oi_prev"})
    cur = cur.merge(base_prev, on=["strike", "option_type"], how="left")
    cur["change_oi"] = cur["oi"] - cur["oi_prev"]

# ========================
# Detect spot & ATM
# ========================
spot = None
if manual_spot > 0:
    spot = float(manual_spot)
elif cur["underlying"].notna().any():
    spot = float(cur["underlying"].dropna().iloc[0])
else:
    # fallback: parity of CE/PE LTP
    ce = cur[cur.option_type == "CE"][["strike", "ltp"]].rename(columns={"ltp": "ce_ltp"})
    pe = cur[cur.option_type == "PE"][["strike", "ltp"]].rename(columns={"ltp": "pe_ltp"})
    both = ce.merge(pe, on="strike", how="inner")
    if not both.empty:
        idx = (both["ce_ltp"] - both["pe_ltp"]).abs().idxmin()
        spot = float(both.loc[idx, "strike"])

if spot is None or not np.isfinite(spot):
    st.error("Could not detect spot. Enter a Manual Spot value in the sidebar.")
    st.stop()

# ATM = nearest strike to spot
uniq_strikes = np.sort(cur["strike"].unique())
atm_strike = float(uniq_strikes[np.argmin(np.abs(uniq_strikes - spot))])

# Window
atm_idx = int(np.argmin(np.abs(uniq_strikes - atm_strike)))
low_idx = max(0, atm_idx - int(n_window))
high_idx = min(len(uniq_strikes) - 1, atm_idx + int(n_window))
lowS, highS = uniq_strikes[low_idx], uniq_strikes[high_idx]
win = cur[(cur["strike"] >= lowS) & (cur["strike"] <= highS)].copy().reset_index(drop=True)

# Derived metrics
win["notional_oi"] = win["oi"] * win["ltp"]
win["notional_chg_oi"] = win["change_oi"].fillna(0) * win["ltp"].fillna(0)

# PCR & bias
call_oi = win.loc[win.option_type == "CE", "oi"].sum()
put_oi  = win.loc[win.option_type == "PE", "oi"].sum()
pcr = (put_oi / call_oi) if call_oi else np.nan

call_chg_notional = win.loc[win.option_type == "CE", "notional_chg_oi"].sum()
put_chg_notional  = win.loc[win.option_type == "PE", "notional_chg_oi"].sum()

bias = "Neutral"
if np.isfinite(call_chg_notional) and np.isfinite(put_chg_notional):
    if put_chg_notional > call_chg_notional * 1.1:
        bias = "Bullish (Put build-up > Call)"
    elif call_chg_notional > put_chg_notional * 1.1:
        bias = "Bearish (Call build-up > Put)"

# ========================
# Header metrics
# ========================
st.subheader("Snapshot")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Spot (manual/auto)", f"{spot:,.2f}")
m2.metric("ATM Strike", f"{int(round(atm_strike))}")
m3.metric("PCR (OI)", f"{pcr:.2f}" if np.isfinite(pcr) else "NA")
m4.metric("Bias", bias)

# ========================
# 150-word layman summary (data-aware)
# ========================
st.subheader("Auto Summary (Layman)")
summary_bits = []
summary_bits.append(f"Spot is near {int(round(spot))} with ATM at {int(round(atm_strike))}.")
if np.isfinite(pcr):
    if pcr > 1.2:
        summary_bits.append(f"PCR is {pcr:.2f}, showing heavier Put positioning, often supportive/bullish.")
    elif pcr < 0.8:
        summary_bits.append(f"PCR is {pcr:.2f}, showing heavier Call positioning, often caution/bearish.")
    else:
        summary_bits.append(f"PCR is {pcr:.2f}, suggesting a balanced/neutral stance.")
summary_bits.append(f"Change in notional OI suggests: {bias}.")
iv_med = win["iv"].median(skipna=True)
if np.isfinite(iv_med):
    summary_bits.append(f"Median IV is around {iv_med:.1f} — {'elevated' if iv_med>25 else 'moderate to low'}, affecting premium behaviour.")
# word-trim to ~150 words:
summary_text = (" ".join(summary_bits))[:1100]
st.write(summary_text)

# ========================
# Plots (OI, ΔOI, Volume, IV, Greeks, Notional OI)
# ========================
st.subheader("Exploration – OI, Change in OI, Volume, IV, Greeks")

def bar_dual(df, y_ce, y_pe, title, ylab):
    ce = df[df.option_type == "CE"]
    pe = df[df.option_type == "PE"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ce["strike"], y=ce[y_ce], name="CE"))
    fig.add_trace(go.Bar(x=pe["strike"], y=pe[y_pe], name="PE"))
    fig.update_layout(barmode="group", title=title, xaxis_title="Strike", yaxis_title=ylab)
    st.plotly_chart(fig, use_container_width=True)

bar_dual(win, "oi", "oi", "Open Interest by Strike", "OI")
bar_dual(win, "change_oi", "change_oi", "Change in OI by Strike", "Change in OI")
bar_dual(win, "volume", "volume", "Volume by Strike", "Volume")

def line_dual(df, col, title):
    ce = df[df.option_type == "CE"]
    pe = df[df.option_type == "PE"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ce["strike"], y=ce[col], mode="lines+markers", name=f"CE {col}"))
    fig.add_trace(go.Scatter(x=pe["strike"], y=pe[col], mode="lines+markers", name=f"PE {col}"))
    fig.update_layout(title=title, xaxis_title="Strike", yaxis_title=col.title())
    st.plotly_chart(fig, use_container_width=True)

for g in ["iv", "delta", "gamma", "theta", "vega"]:
    if g in win.columns and win[g].notna().any():
        line_dual(win, g, f"{g.title()} by Strike")

bar_dual(win, "notional_oi", "notional_oi", "Notional OI (OI x LTP)", "Notional (Rs)")

# ========================
# Heatmaps (Change in OI, IV)
# ========================
st.subheader("Heatmaps")
def heat_matrix(df, value_col, title):
    mat = df.pivot_table(index="option_type", columns="strike", values=value_col, aggfunc="sum")
    mat = mat.sort_index()  # CE first then PE
    fig = px.imshow(mat, aspect="auto", origin="lower", title=title,
                    labels=dict(x="Strike", y="Side", color=value_col))
    st.plotly_chart(fig, use_container_width=True)

if "change_oi" in win.columns:
    heat_matrix(win, "change_oi", "Heatmap: Change in OI (by Side vs Strike)")
if "iv" in win.columns:
    heat_matrix(win, "iv", "Heatmap: Implied Volatility (by Side vs Strike)")

# ========================
# Momentum (optional from previous)
# ========================
if prev is not None and "ltp" in prev.columns and prev["ltp"].notna().any():
    base_prev = prev.groupby(["strike", "option_type"], as_index=False)["ltp"].mean().rename(columns={"ltp": "ltp_prev"})
    win = win.merge(base_prev, on=["strike", "option_type"], how="left")
    win["prem_momentum"] = (win["ltp"] - win["ltp_prev"]) / win["ltp_prev"].replace(0, np.nan)
else:
    win["prem_momentum"] = np.nan

# ========================
# Opportunity scoring & recommendations
# ========================
# Z-scores
for c in ["change_oi", "volume", "iv", "notional_oi", "prem_momentum"]:
    if c not in win.columns:
        win[c] = np.nan
    win[c + "_z"] = zscore(win[c])

# Delta closeness (~0.45 absolute for long options)
def delta_closeness(row):
    d = row.get("delta", np.nan)
    if pd.isna(d):
        return 0.0
    target = 0.45
    return max(0.0, 1.0 - min(abs(abs(d) - target) / target, 1.0))

win["delta_closeness"] = win.apply(delta_closeness, axis=1)

# Filters
flt = (win["volume"].fillna(0) >= min_volume) & (win["iv"].fillna(0) <= max_iv)
cand = win[flt].copy()

# Directional bias selection
long_side = None
if "Bullish" in bias:
    long_side = "CE"
elif "Bearish" in bias:
    long_side = "PE"

if long_side:
    cand = cand[cand.option_type == long_side]

# Scoring
cand["score"] = (
    w_chg_oi * cand["change_oi_z"] +
    w_vol    * cand["volume_z"] +
    w_iv     * (-cand["iv_z"]) +
    w_delta  * cand["delta_closeness"] +
    w_momo   * cand["prem_momentum_z"] +
    w_notioi * cand["notional_oi_z"]
)

# Probability of Profit approximation from |Delta| adjusted by IV z
def pop_est(row):
    d = abs(row.get("delta", 0.0))
    ivz = row.get("iv_z", 0.0)
    adj = 1.0 / (1.0 + max(ivz, -0.9))  # penalize high IV
    return float(np.clip(d * adj, 0.05, 0.95))

cand["POP"] = cand.apply(pop_est, axis=1)

# Entry/Target/SL
cand["Entry"] = cand["ltp"]
cand["SL"] = cand["Entry"] * (1 - risk_pct / 100.0)
cand["Target"] = cand["Entry"] * (1 + reward_pct / 100.0)

# Reason string
def reason(row):
    parts = []
    if pd.notna(row.get("change_oi")) and row.get("change_oi") > 0:
        parts.append("OI build-up")
    if pd.notna(row.get("prem_momentum")) and row.get("prem_momentum") > 0:
        parts.append("Premium rising")
    if row.get("iv_z", 0) < 0:
        parts.append("IV relatively low")
    if row.get("delta_closeness", 0) > 0.6:
        parts.append("Favourable Delta (~0.45 abs)")
    noti = row.get("notional_oi", 0)
    if pd.notna(noti):
        parts.append(f"Notional OI {noti:,.0f}")
    return ", ".join(parts)

cand["Reason"] = cand.apply(reason, axis=1)

# Top opportunities
st.subheader("Data-backed Buying Opportunities")
cols_show = ["option_type", "strike", "ltp", "iv", "oi", "change_oi", "volume", "delta", "gamma", "theta", "vega",
             "Entry", "Target", "SL", "POP", "score", "Reason"]
recos = cand.sort_values("score", ascending=False)[cols_show].head(8).rename(columns={"ltp": "LTP"})
st.dataframe(
    recos.style.format({
        "LTP": "{:.2f}", "iv": "{:.2f}", "oi": "{:,.0f}", "change_oi": "{:,.0f}", "volume": "{:,.0f}",
        "delta": "{:.2f}", "gamma": "{:.4f}", "theta": "{:.2f}", "vega": "{:.2f}",
        "Entry": "{:.2f}", "Target": "{:.2f}", "SL": "{:.2f}", "POP": "{:.1%}", "score": "{:.2f}"
    }),
    use_container_width=True
)

# Export recos
csv_bytes = recos.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Recommendations (CSV)",
    data=csv_bytes,
    file_name=f"bnf_recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)

# Raw window (optional)
if show_raw:
    st.subheader("Normalized Current Data (Window)")
    st.dataframe(win, use_container_width=True)

st.caption("Notes: POP is approximated using |Delta| with IV adjustment. Use prudent position sizing and independent risk management.")
