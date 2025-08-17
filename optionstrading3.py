# bnf_analyzer.py
# BankNifty Option Chain Analyzer (single file upload, default spot=55341)
# - Robust column detection (wide or long), graceful handling of missing columns
# - ATM window, plots, heatmaps, 150-word summary, data-backed recommendations

import io
from datetime import datetime
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="BankNifty Option Chain Analyzer", layout="wide")
st.title("BankNifty Option Chain Analyzer — Greeks, OI, Heatmaps & Recommendations")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")
n_window = st.sidebar.number_input("ATM window (± N strikes)", min_value=5, max_value=60, value=15, step=1)
min_volume = st.sidebar.number_input("Min volume to consider (rows)", min_value=0, value=50, step=10)

risk_reward = st.sidebar.selectbox("Risk/Reward preset", ["1:1", "1:1.5", "1:2", "Custom"], index=1)
if risk_reward == "Custom":
    stop_pct = st.sidebar.number_input("Stop loss %", min_value=1.0, max_value=80.0, value=20.0)
    tgt_pct = st.sidebar.number_input("Target %", min_value=1.0, max_value=300.0, value=30.0)
else:
    rr_map = {"1:1": (20.0, 20.0), "1:1.5": (20.0, 30.0), "1:2": (20.0, 40.0)}
    stop_pct, tgt_pct = rr_map[risk_reward]

st.sidebar.markdown("---")
st.sidebar.subheader("Scoring weights (for recommendations)")
w_chg_oi = st.sidebar.slider("Weight: Change in OI", 0.0, 3.0, 1.0, 0.1)
w_vol = st.sidebar.slider("Weight: Volume", 0.0, 3.0, 1.0, 0.1)
w_iv = st.sidebar.slider("Weight: (Lower) IV", 0.0, 3.0, 0.6, 0.1)
w_delta = st.sidebar.slider("Weight: Delta closeness", 0.0, 3.0, 0.8, 0.1)
w_mom = st.sidebar.slider("Weight: Premium momentum", 0.0, 3.0, 0.6, 0.1)
w_notioi = st.sidebar.slider("Weight: Notional OI", 0.0, 3.0, 0.6, 0.1)

st.sidebar.markdown("---")
manual_spot = st.sidebar.number_input("Manual spot price (default 55341)", value=55341)
show_raw = st.sidebar.checkbox("Show normalized window data", value=False)

# -------------------------
# File upload
# -------------------------
st.subheader("Upload option chain (CSV or XLSX)")
uploaded = st.file_uploader("Upload one option chain file (wide or long format)", type=["csv", "xlsx"])

# -------------------------
# Helpers: read & normalize
# -------------------------
def read_file(f):
    if f is None:
        return None
    name = f.name.lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(f)
    content = f.getvalue()
    for enc in ["utf-8", "utf-16", "cp1252", "latin-1"]:
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc)
        except Exception:
            pass
    return pd.read_csv(io.BytesIO(content))  # last resort

# Normalize columns: lower, strip, replace spaces -> underscores
def norm_cols(df):
    df2 = df.copy()
    df2.columns = [str(c).strip().lower().replace(" ", "_").replace(".", "_") for c in df2.columns]
    return df2

# Try to standardize either wide (c_/p_) or long
def to_long(df):
    df0 = norm_cols(df)
    cols = df0.columns.tolist()

    # find strike-like column
    strike_candidates = [c for c in cols if "strike" in c]
    if not strike_candidates:
        raise ValueError("No strike column found (need strike / strike_price / strikeprice).")
    strike_col = strike_candidates[0]

    # find if 'option_type' present
    option_type_col = next((c for c in cols if c in ("option_type", "type", "cp", "call_put", "option")), None)
    # detect wide prefixes
    has_c = any(c.startswith("c_") for c in cols)
    has_p = any(c.startswith("p_") for c in cols)

    # standard long target
    target = ["strike", "option_type", "ltp", "iv", "oi", "change_oi", "volume", "delta", "gamma", "theta", "vega", "underlying", "ploi"]

    if option_type_col and not (has_c and has_p):
        # long format: map columns by key words
        out = pd.DataFrame()
        out["strike"] = pd.to_numeric(df0[strike_col], errors="coerce")
        def map_opt(x):
            s = str(x).lower()
            if "call" in s or s == "c":
                return "CE"
            if "put" in s or s == "p":
                return "PE"
            return np.nan
        out["option_type"] = df0[option_type_col].map(map_opt)

        # search for metric columns (best-effort)
        def find_col_for(key_aliases):
            for alias in key_aliases:
                if alias in df0.columns:
                    return alias
            for c in df0.columns:
                for alias in key_aliases:
                    if alias in c:
                        return c
            return None

        aliases = {
            "ltp": ["ltp", "last_traded_price", "last", "premium", "price"],
            "iv": ["iv", "implied_volatility", "imp_vol"],
            "oi": ["oi", "open_interest", "openint"],
            "change_oi": ["change_in_oi", "chng_in_oi", "chg_oi", "change_oi"],
            "volume": ["volume", "vol"],
            "delta": ["delta"],
            "gamma": ["gamma"],
            "theta": ["theta"],
            "vega": ["vega"],
            "underlying": ["underlying", "spot", "underlying_price", "underlyingvalue", "index_price"],
            "ploi": ["ploi", "price_oi", "notional", "notional_oi", "oi_value"]
        }
        for k, a in aliases.items():
            col_found = find_col_for(a)
            out[k] = pd.to_numeric(df0.get(col_found, np.nan), errors="coerce")
        return out.dropna(subset=["strike", "option_type"]).reset_index(drop=True)

    if has_c and has_p:
        # Wide format: split into CE and PE rows
        base = df0[[c for c in df0.columns if not (c.startswith("c_") or c.startswith("p_"))]].copy()
        # ensure strike column present in base
        if strike_col not in base.columns:
            # maybe the strike is in a prefixed column, try to find it globally
            pass
        base = base.rename(columns={strike_col: "strike"})
        base["strike"] = pd.to_numeric(base["strike"], errors="coerce")
        def extract_side(prefix, side):
            s = base.copy()
            for c in df0.columns:
                if c.startswith(prefix):
                    metric = c[len(prefix):]
                    s[metric] = pd.to_numeric(df0[c], errors="coerce")
            s["option_type"] = side
            return s
        ce = extract_side("c_", "CE")
        pe = extract_side("p_", "PE")
        out = pd.concat([ce, pe], ignore_index=True, sort=False)
        # normalize some names
        rename_map = {
            "chng_in_oi": "change_oi", "change_in_oi": "change_oi",
            "ltp": "ltp", "iv": "iv", "oi": "oi", "volume": "volume",
            "delta": "delta", "gamma": "gamma", "theta": "theta", "vega": "vega", "ploi": "ploi"
        }
        for old, new in rename_map.items():
            if old in out.columns and new not in out.columns:
                out[new] = out[old]
        # underlying detection (if present)
        und_col = next((c for c in df0.columns if c in ("underlying", "spot", "underlying_price", "underlyingvalue", "index_price")), None)
        if und_col:
            out["underlying"] = pd.to_numeric(df0[und_col], errors="coerce")
        # ensure columns exist
        for c in target:
            if c not in out.columns:
                out[c] = np.nan
        return out[["strike", "option_type", "ltp", "iv", "oi", "change_oi", "volume", "delta", "gamma", "theta", "vega", "underlying", "ploi"]].dropna(subset=["strike", "option_type"]).reset_index(drop=True)

    # fallback: raise
    raise ValueError("Could not parse file: unknown layout (neither long with option_type nor wide with c_/p_ prefixes).")

def zscore(series):
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd == 0 or pd.isna(sd):
        return pd.Series(0.0, index=s.index)
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], 0).fillna(0)

# -------------------------
# Main flow
# -------------------------
if uploaded is None:
    st.info("Upload one option chain file (CSV or XLSX). Supported: wide (c_/p_ columns) or long (option_type column).")
    st.stop()

try:
    raw = read_file(uploaded)
except Exception as e:
    st.error(f"Failed to read upload: {e}")
    st.stop()

# Normalize to long table
try:
    df_long = to_long(raw)
except Exception as e:
    st.error(f"Failed to normalize upload: {e}")
    st.stop()

# show which columns are present / missing
expected = ["strike", "option_type", "ltp", "iv", "oi", "change_oi", "volume", "delta", "gamma", "theta", "vega", "underlying"]
present = [c for c in expected if c in df_long.columns and df_long[c].notna().any()]
missing = [c for c in expected if c not in df_long.columns or df_long[c].isna().all()]
if missing:
    st.warning(f"Note: some fields are missing or all-null and features depending on them will be skipped: {missing}")

# detect spot
spot = None
if manual_spot and manual_spot > 0:
    spot = float(manual_spot)
elif "underlying" in df_long.columns and df_long["underlying"].notna().any():
    spot = float(df_long["underlying"].dropna().iloc[0])
else:
    # parity fallback: find strike where CE LTP ~= PE LTP
    try:
        ce = df_long[df_long.option_type == "CE"][["strike", "ltp"]].rename(columns={"ltp": "ce_ltp"})
        pe = df_long[df_long.option_type == "PE"][["strike", "ltp"]].rename(columns={"ltp": "pe_ltp"})
        both = ce.merge(pe, on="strike", how="inner")
        if not both.empty:
            idx = (both["ce_ltp"] - both["pe_ltp"]).abs().idxmin()
            spot = float(both.loc[idx, "strike"])
    except Exception:
        pass

if spot is None or not np.isfinite(spot):
    st.error("Spot price could not be detected automatically. Enter a Manual spot price in the sidebar.")
    st.stop()

# atm strike (nearest)
uniq_strikes = np.sort(df_long["strike"].unique())
atm_strike = float(uniq_strikes[np.argmin(np.abs(uniq_strikes - spot))])

# window around ATM based on strike index
atm_idx = int(np.argmin(np.abs(uniq_strikes - atm_strike)))
low_idx = max(0, atm_idx - n_window)
high_idx = min(len(uniq_strikes) - 1, atm_idx + n_window)
lowS, highS = uniq_strikes[low_idx], uniq_strikes[high_idx]

win = df_long[(df_long["strike"] >= lowS) & (df_long["strike"] <= highS)].copy().reset_index(drop=True)

# derived columns
win["notional_oi"] = win["oi"].fillna(0) * win["ltp"].fillna(0)
win["notional_chg_oi"] = win["change_oi"].fillna(0) * win["ltp"].fillna(0)

# compute PCR (OI) if CE & PE present
ce_oi = win.loc[win.option_type == "CE", "oi"].sum() if "oi" in win.columns else np.nan
pe_oi = win.loc[win.option_type == "PE", "oi"].sum() if "oi" in win.columns else np.nan
pcr = (pe_oi / ce_oi) if ce_oi and not math.isnan(ce_oi) else np.nan

# directional bias by notional change
call_chg_notional = win.loc[win.option_type == "CE", "notional_chg_oi"].sum()
put_chg_notional = win.loc[win.option_type == "PE", "notional_chg_oi"].sum()
bias = "Neutral"
if np.isfinite(call_chg_notional) and np.isfinite(put_chg_notional):
    if put_chg_notional > call_chg_notional * 1.1:
        bias = "Bullish (Put build-up > Call)"
    elif call_chg_notional > put_chg_notional * 1.1:
        bias = "Bearish (Call build-up > Put)"

# header metrics
st.subheader("Snapshot")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Spot (manual/auto)", f"{spot:,.2f}")
c2.metric("ATM Strike", f"{int(round(atm_strike))}")
c3.metric("PCR (OI)", f"{pcr:.2f}" if np.isfinite(pcr) else "NA")
c4.metric("Bias", bias)

# 150-word layman summary (simple, data-aware)
summary_lines = []
summary_lines.append(f"Spot is near {int(round(spot))} and ATM is {int(round(atm_strike))}.")
if np.isfinite(pcr):
    if pcr > 1.2:
        summary_lines.append(f"PCR is {pcr:.2f}, indicating more Put interest than Calls — often a bullish tilt.")
    elif pcr < 0.8:
        summary_lines.append(f"PCR is {pcr:.2f}, indicating more Call interest — often cautionary/bearish.")
    else:
        summary_lines.append(f"PCR is {pcr:.2f} — broadly balanced between Calls and Puts.")
if "change_oi" in win.columns:
    summary_lines.append("Change-in-OI highlights where traders are building or trimming positions; strikes with rising OI show conviction.")
if "iv" in win.columns:
    ivm = win["iv"].median(skipna=True)
    summary_lines.append(f"Median IV ≈ {ivm:.1f}%, which shows the market's expectation of near-term movement.")
summary_lines.append("Greeks show sensitivity: Delta tracks directional exposure, Gamma shows Delta acceleration, Theta is time decay and Vega shows volatility sensitivity.")
# join and limit ~150 words
summary_text = " ".join(summary_lines)
summary_text = " ".join(summary_text.split()[:160])  # approx 150 words cap
st.subheader("Auto Summary (Layman)")
st.write(summary_text)

# -------------------------
# Exploration plots
# -------------------------
st.subheader("Exploration — OI, Change in OI, Volume, IV & Greeks")

def bar_combined(df, col, title, ytitle):
    ce = df[df.option_type == "CE"]
    pe = df[df.option_type == "PE"]
    fig = go.Figure()
    if not ce.empty:
        fig.add_trace(go.Bar(x=ce["strike"], y=ce[col], name="CE"))
    if not pe.empty:
        fig.add_trace(go.Bar(x=pe["strike"], y=pe[col], name="PE"))
    fig.update_layout(barmode="group", title=title, xaxis_title="Strike", yaxis_title=ytitle)
    st.plotly_chart(fig, use_container_width=True)

if "oi" in win.columns:
    bar_combined(win, "oi", "Open Interest by Strike", "OI")
if "change_oi" in win.columns:
    bar_combined(win, "change_oi", "Change in OI by Strike", "ΔOI")
if "volume" in win.columns:
    bar_combined(win, "volume", "Volume by Strike", "Volume")
if "iv" in win.columns:
    line_combined = lambda col, t: st.plotly_chart(go.Figure().add_trace(go.Scatter(x=win[win.option_type=="CE"]["strike"], y=win[win.option_type=="CE"][col], mode="lines+markers", name=f"CE {col}")).add_trace(go.Scatter(x=win[win.option_type=="PE"]["strike"], y=win[win.option_type=="PE"][col], mode="lines+markers", name=f"PE {col}")).update_layout(title=t, xaxis_title="Strike"), use_container_width=True)
    line_combined("iv", "IV by Strike")

for g in ["delta", "gamma", "theta", "vega"]:
    if g in win.columns and win[g].notna().any():
        fig = go.Figure()
        ce = win[win.option_type == "CE"]
        pe = win[win.option_type == "PE"]
        if not ce.empty:
            fig.add_trace(go.Scatter(x=ce["strike"], y=ce[g], mode="lines+markers", name=f"CE {g}"))
        if not pe.empty:
            fig.add_trace(go.Scatter(x=pe["strike"], y=pe[g], mode="lines+markers", name=f"PE {g}"))
        fig.update_layout(title=f"{g.title()} by Strike", xaxis_title="Strike", yaxis_title=g.title())
        st.plotly_chart(fig, use_container_width=True)

# Notional OI
if "notional_oi" in win.columns:
    bar_combined(win, "notional_oi", "Notional OI (OI × LTP) by Strike", "Notional (Rs)")

# -------------------------
# Heatmaps
# -------------------------
st.subheader("Heatmaps")

def heatshow(df, value_col, title):
    # pivot: rows = side, cols = strike
    mat = df.pivot_table(index="option_type", columns="strike", values=value_col, aggfunc="sum").fillna(0)
    if mat.size == 0:
        st.info(f"No data for heatmap {value_col}")
        return
    fig = px.imshow(mat, aspect="auto", origin="lower", title=title, labels=dict(x="Strike", y="Side", color=value_col))
    st.plotly_chart(fig, use_container_width=True)

if "change_oi" in win.columns:
    heatshow(win, "change_oi", "Heatmap: Change in OI (side × strike)")
if "iv" in win.columns:
    heatshow(win, "iv", "Heatmap: IV (side × strike)")

# -------------------------
# Momentum (if previous snapshot provided in same file shape? optional)
# -------------------------
# We'll skip external prev unless user uploads separate file in future versions.

# -------------------------
# Recommendation engine
# -------------------------
st.subheader("Data-backed Recommendations")

# create z-scores and features
for c in ["change_oi", "volume", "iv", "notional_oi"]:
    win[c + "_z"] = zscore(win[c]) if c in win.columns else 0.0
win["prem_momentum"] = np.nan  # placeholder

# delta closeness: prefer ~0.45 absolute
def delta_closeness(v):
    try:
        d = abs(float(v))
    except Exception:
        return 0.0
    target = 0.45
    return max(0.0, 1.0 - min(abs(d - target) / target, 1.0))

win["delta_closeness"] = win["delta"].apply(delta_closeness) if "delta" in win.columns else 0.0

# base filter
cand = win.copy()
cand = cand[cand["volume"].fillna(0) >= min_volume] if "volume" in cand.columns else cand

# optional directional bias filtering
if "Bullish" in bias:
    cand = cand[cand.option_type == "CE"]
elif "Bearish" in bias:
    cand = cand[cand.option_type == "PE"]

# scoring
cand["score"] = (
    w_chg_oi * cand.get("change_oi_z", 0) +
    w_vol * cand.get("volume_z", 0) +
    w_iv * ( - cand.get("iv_z", 0)) +
    w_delta * cand.get("delta_closeness", 0) +
    w_mom * cand.get("prem_momentum_z", 0) +
    w_notioi * cand.get("notional_oi_z", 0)
)

# POP estimate from |delta| adjusted by IV z
def estimate_pop(row):
    d = abs(row.get("delta", 0.0)) if not pd.isna(row.get("delta", np.nan)) else 0.0
    ivz = row.get("iv_z", 0.0) if not pd.isna(row.get("iv_z", 0.0)) else 0.0
    adj = 1.0 / (1.0 + max(ivz, -0.9))
    pop = np.clip(d * adj, 0.03, 0.95)
    return float(pop)

cand["POP"] = cand.apply(estimate_pop, axis=1)

# Entry/Target/SL
cand["Entry"] = cand["ltp"]
cand["SL"] = cand["Entry"] * (1 - stop_pct / 100.0)
cand["Target"] = cand["Entry"] * (1 + tgt_pct / 100.0)

# Reason builder
def build_reason(r):
    parts = []
    if pd.notna(r.get("change_oi")) and r.get("change_oi") > 0:
        parts.append("OI build-up")
    if pd.notna(r.get("volume")) and r.get("volume") > 0:
        parts.append("Active volume")
    if pd.notna(r.get("iv_z")) and r.get("iv_z") < 0:
        parts.append("IV relatively low")
    if r.get("delta_closeness", 0) > 0.6:
        parts.append("Favourable Delta (~0.45)")
    parts.append(f"Notional OI {int(r.get('notional_oi',0)):,}")
    return ", ".join(parts)

cand["Reason"] = cand.apply(build_reason, axis=1)

# show top N
top = cand.sort_values("score", ascending=False).head(8)
if top.empty:
    st.info("No candidate recommendations found with current filters and available data.")
else:
    display_cols = ["option_type", "strike", "ltp", "iv", "oi", "change_oi", "volume", "delta", "gamma", "theta", "vega", "Entry", "Target", "SL", "POP", "score", "Reason"]
    # keep only existing columns
    display_cols = [c for c in display_cols if c in top.columns]
    top_display = top[display_cols].copy()
    top_display = top_display.rename(columns={"ltp": "LTP"})
    st.dataframe(top_display.style.format({
        **({ "LTP":"{:.2f}"} if "LTP" in top_display.columns else {}),
        **({ "iv":"{:.2f}"} if "iv" in top_display.columns else {}),
        **({ "oi":"{:,.0f}"} if "oi" in top_display.columns else {}),
        **({ "change_oi":"{:,.0f}"} if "change_oi" in top_display.columns else {}),
        **({ "volume":"{:,.0f}"} if "volume" in top_display.columns else {}),
        **({ "delta":"{:.2f}"} if "delta" in top_display.columns else {}),
        **({ "gamma":"{:.4f}"} if "gamma" in top_display.columns else {}),
        **({ "theta":"{:.2f}"} if "theta" in top_display.columns else {}),
        **({ "vega":"{:.2f}"} if "vega" in top_display.columns else {}),
        **({ "Entry":"{:.2f}", "Target":"{:.2f}", "SL":"{:.2f}", "POP":"{:.1%}", "score":"{:.2f}"})
    }), use_container_width=True)

    # download
    csv_bytes = top_display.to_csv(index=False).encode("utf-8")
    st.download_button("Download recommendations CSV", data=csv_bytes, file_name=f"bnf_recos_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")

# Raw window optionally
if show_raw:
    st.subheader("Normalized window data (raw)")
    st.dataframe(win)

st.caption("Notes: This tool uses approximations (POP uses |delta| and IV z-score). Use with prudent position sizing.")
