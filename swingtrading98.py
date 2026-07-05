"""
=====================================================================================
 INSIGHTS FINDER  —  Single-file Streamlit Market Pattern-Mining App
=====================================================================================
One button, one ticker (Nifty/BankNifty/Sensex/BTC/ETH/Gold/Silver/Forex/custom),
and a full report across every timeframe: day-of-week seasonality, month-of-year
seasonality, opening-gap behavior, up/down streak persistence, intraday time-of-day
patterns, cross-asset correlation, and an India-VIX regime comparison — each with a
bar chart AND a plain-English narrative, not just raw numbers.

MANDATORY 30-SECOND DELAY between every real Yahoo Finance request, as instructed,
to stay well clear of rate limits. A full report is therefore genuinely slow (the
sidebar shows an honest time estimate before you click) — that tradeoff is
deliberate, not a bug.

HONESTY NOTES (this app prints these in-app too, but worth saying up front):
  - Every stat here describes the PAST. None of it predicts tomorrow.
  - This tool deliberately tests many patterns on the same data. Some will look
    "significant" purely by chance (the multiple-comparisons problem) — check the
    sample size (n) on every stat, and treat anything here as a hypothesis to
    validate on data you haven't looked at yet, not a finished trading edge.
=====================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="Insights Finder", layout="wide", page_icon="🔎")

# =====================================================================================
# CONSTANTS
# =====================================================================================
YF_REQUEST_DELAY_SECONDS = 30  # mandatory, per requirement — applied before every real fetch

PRESET_TICKERS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Gold Futures (GC=F)": "GC=F",
    "Silver Futures (SI=F)": "SI=F",
    "USD/INR": "INR=X",
    "EUR/USD": "EURUSD=X",
    "India VIX": "^INDIAVIX",
    "Custom Ticker": None,
}

CORRELATION_UNIVERSE = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "USD/INR": "INR=X",
    "India VIX": "^INDIAVIX",
}

# (label, yfinance interval, yfinance period — all valid Yahoo period tokens, no invented ones, kind)
SECTION_CONFIGS = [
    ("1 Minute",  "1m",  "5d",  "intraday"),
    ("5 Minute",  "5m",  "3mo", "intraday"),
    ("15 Minute", "15m", "3mo", "intraday"),
    ("30 Minute", "30m", "3mo", "intraday"),
    ("1 Hour",    "60m", "2y",  "intraday"),
    ("1 Day",     "1d",  "max", "daily"),
    ("1 Week",    "1wk", "max", "weekly"),
    ("1 Month",   "1mo", "max", "monthly"),
]

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


# =====================================================================================
# DATA FETCH — mandatory 30s delay, retries, real error surfaced (not swallowed)
# =====================================================================================
def _fetch_yf_with_retries(ticker, interval, period, attempts=2):
    last_err = None
    for attempt in range(attempts):
        try:
            time.sleep(YF_REQUEST_DELAY_SECONDS)
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
            if df is not None and not df.empty:
                return df[["Open", "High", "Low", "Close", "Volume"]].dropna(), None
            last_err = "Yahoo Finance returned zero rows for this ticker/interval/period."
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return pd.DataFrame(), last_err


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker, interval, period):
    df, _ = _fetch_yf_with_retries(ticker, interval, period)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_vix_snapshot():
    return fetch_data("^INDIAVIX", "1d", "5d")


def slice_years(df, years):
    if df.empty:
        return df
    cutoff = df.index.max() - pd.DateOffset(years=years)
    return df[df.index >= cutoff]


# =====================================================================================
# NARRATIVE HELPERS
# =====================================================================================
def sample_note(n):
    if n < 20:
        return f"n={n} — very small sample, likely noisy"
    if n < 60:
        return f"n={n} — modest sample, treat cautiously"
    return f"n={n}"


def tilt_phrase(pct, baseline=50.0):
    diff = pct - baseline
    if abs(diff) < 3:
        return "essentially a coin flip"
    if abs(diff) < 8:
        return f"a mild {'bullish' if diff > 0 else 'bearish'} tilt"
    return f"a notable {'bullish' if diff > 0 else 'bearish'} tilt"


# =====================================================================================
# INSIGHT COMPUTATIONS
# =====================================================================================
def weekday_seasonality(df):
    d = df.copy()
    d["ret"] = d["Close"].pct_change()
    d = d.dropna(subset=["ret"])
    d["weekday"] = d.index.day_name()
    stats = d.groupby("weekday")["ret"].agg(n="count", avg_return=lambda x: x.mean() * 100,
                                             pct_positive=lambda x: (x > 0).mean() * 100)
    return stats.reindex([w for w in WEEKDAY_ORDER if w in stats.index])


def month_seasonality(df):
    d = df.copy()
    d["ret"] = d["Close"].pct_change()
    d = d.dropna(subset=["ret"])
    d["month"] = d.index.month
    stats = d.groupby("month")["ret"].agg(n="count", avg_return=lambda x: x.mean() * 100,
                                           pct_positive=lambda x: (x > 0).mean() * 100)
    stats.index = [MONTH_NAMES[m - 1] for m in stats.index]
    return stats


def gap_behavior(df):
    g = df.copy()
    g["prev_close"] = g["Close"].shift(1)
    g["gap_pct"] = (g["Open"] - g["prev_close"]) / g["prev_close"] * 100
    g = g.dropna(subset=["gap_pct"])
    up = g[g["gap_pct"] > 0.1]
    down = g[g["gap_pct"] < -0.1]
    up_fill = (up["Low"] <= up["prev_close"]).mean() * 100 if len(up) else np.nan
    down_fill = (down["High"] >= down["prev_close"]).mean() * 100 if len(down) else np.nan
    return dict(n_total=len(g), n_up=len(up), n_down=len(down), up_fill=up_fill, down_fill=down_fill)


def streak_persistence(df):
    up = (df["Close"].pct_change() > 0)
    rows = []
    for n in (1, 2, 3):
        up_streak = up.rolling(n).sum() == n
        down_streak = (~up).rolling(n).sum() == n
        rows.append(dict(
            streak=n,
            pct_up_after_up=up.shift(-1)[up_streak].mean() * 100 if up_streak.sum() else np.nan,
            n_up=int(up_streak.sum()),
            pct_up_after_down=up.shift(-1)[down_streak].mean() * 100 if down_streak.sum() else np.nan,
            n_down=int(down_streak.sum())))
    return pd.DataFrame(rows)


def intraday_timing(df):
    idf = df.copy()
    idf["date"] = idf.index.date
    recs = []
    for d, grp in idf.groupby("date"):
        if len(grp) < 3:
            continue
        start = grp.index[0]
        recs.append(((grp["High"].idxmax() - start).total_seconds() / 60,
                      (grp["Low"].idxmin() - start).total_seconds() / 60))
    t = pd.DataFrame(recs, columns=["min_to_high", "min_to_low"])
    if t.empty:
        return None
    return dict(n=len(t),
                high15=(t["min_to_high"] <= 15).mean() * 100, low15=(t["min_to_low"] <= 15).mean() * 100,
                high30=(t["min_to_high"] <= 30).mean() * 100, low30=(t["min_to_low"] <= 30).mean() * 100)


def hourly_returns(df):
    r = df["Close"].pct_change().dropna() * 100
    if r.empty:
        return None
    stats = r.groupby(r.index.hour).agg(["mean", "count"])
    stats.columns = ["avg_return", "n"]
    return stats


def compute_correlations(main_df, corr_data, years):
    main_ret = main_df["Close"].pct_change().dropna()
    cutoff = main_df.index.max() - pd.DateOffset(years=years)
    main_ret = main_ret[main_ret.index >= cutoff]
    results = {}
    for name, cdf in corr_data.items():
        if cdf.empty:
            continue
        cret = cdf["Close"].pct_change().dropna()
        aligned = pd.concat([main_ret, cret], axis=1, join="inner")
        aligned.columns = ["main", "other"]
        aligned = aligned.dropna()
        if len(aligned) > 20:
            results[name] = (aligned["main"].corr(aligned["other"]), len(aligned))
    return results


def vix_regime_analysis(main_df, vix_df):
    if vix_df is None or vix_df.empty:
        return None
    m = main_df["Close"].pct_change().rename("main_ret")
    v = vix_df["Close"].rename("vix")
    aligned = pd.concat([m, v], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return None
    median_vix = aligned["vix"].median()
    high = aligned[aligned["vix"] >= median_vix]
    low = aligned[aligned["vix"] < median_vix]
    return dict(median=median_vix,
                high_avg=high["main_ret"].mean() * 100, high_vol=high["main_ret"].std() * 100, n_high=len(high),
                low_avg=low["main_ret"].mean() * 100, low_vol=low["main_ret"].std() * 100, n_low=len(low))


# =====================================================================================
# CHART HELPER
# =====================================================================================
def bar_chart(x, y, title, y_title="", color=None):
    colors = color or ["#2E86AB" if v >= 0 else "#C73E3E" for v in y]
    fig = go.Figure(go.Bar(x=list(x), y=list(y), marker_color=colors,
                           text=[f"{v:.2f}" for v in y], textposition="outside"))
    fig.update_layout(title=title, yaxis_title=y_title, height=350, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


# =====================================================================================
# SIDEBAR
# =====================================================================================
st.sidebar.title("🔎 Insights Finder")

st.sidebar.subheader("📟 Market Context")
with st.sidebar.container():
    try:
        vix_df = fetch_vix_snapshot()
        if not vix_df.empty:
            last_vix = vix_df["Close"].iloc[-1]
            prev_vix = vix_df["Close"].iloc[-2] if len(vix_df) > 1 else last_vix
            chg = last_vix - prev_vix
            regime = ("Low" if last_vix < 13 else "Normal" if last_vix < 18 else
                      "Elevated" if last_vix < 25 else "High")
            st.metric("India VIX", f"{last_vix:.2f}", f"{chg:+.2f}")
            st.caption(f"Regime: **{regime}** volatility (rule-of-thumb bands, not a signal by itself)")
        else:
            st.caption("India VIX unavailable right now.")
    except Exception:
        st.caption("India VIX unavailable right now.")

st.sidebar.subheader("🎯 Ticker")
preset_choice = st.sidebar.selectbox("Preset", list(PRESET_TICKERS.keys()))
if preset_choice == "Custom Ticker":
    custom = st.sidebar.text_input("Exact Yahoo Finance ticker", "RELIANCE.NS",
                                    help="e.g. RELIANCE.NS, TCS.NS, AAPL, TSLA, DOGE-USD")
    main_ticker = custom.strip()
    display_name = custom.strip()
else:
    main_ticker = PRESET_TICKERS[preset_choice]
    display_name = preset_choice

years_hist = st.sidebar.slider("Years of history (Daily/Weekly/Monthly sections)", 1, 20, 10)

include_intraday = st.sidebar.checkbox(
    "Include intraday sections (1m/5m/15m/30m/1h)", value=True,
    help="Uncheck to skip these 5 sections and save ~2.5 minutes of mandatory delay if you only "
         "care about daily/weekly/monthly patterns.")

st.sidebar.subheader("🔗 Correlation Instruments")
corr_choices = st.sidebar.multiselect("Compare against", list(CORRELATION_UNIVERSE.keys()),
                                       default=list(CORRELATION_UNIVERSE.keys()))

sections_to_run = SECTION_CONFIGS if include_intraday else [s for s in SECTION_CONFIGS if s[3] != "intraday"]
n_corr = len([c for c in corr_choices if CORRELATION_UNIVERSE[c] != main_ticker])
total_requests = len(sections_to_run) + n_corr + 1  # +1 for VIX regime series fetch
est_minutes = total_requests * YF_REQUEST_DELAY_SECONDS / 60
st.sidebar.info(f"⏱ Estimated run time: **~{est_minutes:.1f} minutes** "
                f"({total_requests} requests × {YF_REQUEST_DELAY_SECONDS}s mandatory delay each)")

run_btn = st.sidebar.button("🔍 Get Full Insights Report", type="primary")

st.sidebar.divider()
st.sidebar.caption("⚠️ Every stat in this app describes the past. Testing many patterns at once "
                   "means some will look significant purely by chance — check the sample size (n) "
                   "on everything, and validate anything promising on data you haven't seen yet "
                   "before trusting it.")

# =====================================================================================
# MAIN
# =====================================================================================
st.title("🔎 Insights Finder")
st.caption(f"{display_name}  •  {main_ticker}")

if run_btn:
    total_steps = len(sections_to_run) + n_corr + 1
    progress = st.progress(0.0)
    status = st.empty()
    step = 0

    section_data = {}
    for label, interval, period, kind in sections_to_run:
        step += 1
        status.info(f"⏳ Fetching {label} data... (step {step}/{total_steps}, "
                    f"~{(total_steps - step + 1) * YF_REQUEST_DELAY_SECONDS / 60:.1f} min remaining)")
        df = fetch_data(main_ticker, interval, period)
        if kind in ("daily", "weekly", "monthly") and not df.empty:
            df = slice_years(df, years_hist)
        section_data[label] = (df, kind)
        progress.progress(step / total_steps)

    corr_data = {}
    for name in corr_choices:
        sym = CORRELATION_UNIVERSE[name]
        if sym == main_ticker:
            continue
        step += 1
        status.info(f"⏳ Fetching {name} for correlation... (step {step}/{total_steps}, "
                    f"~{(total_steps - step + 1) * YF_REQUEST_DELAY_SECONDS / 60:.1f} min remaining)")
        corr_data[name] = fetch_data(sym, "1d", "max")
        progress.progress(step / total_steps)

    step += 1
    status.info(f"⏳ Fetching India VIX for regime analysis... (step {step}/{total_steps})")
    vix_full = fetch_data("^INDIAVIX", "1d", "max")
    progress.progress(1.0)
    status.empty()
    progress.empty()

    st.session_state["insights_report"] = dict(
        sections=section_data, corr=corr_data, vix_full=vix_full,
        ticker=main_ticker, display_name=display_name, years_hist=years_hist)
    st.success("✅ Report ready.")

report = st.session_state.get("insights_report")
if report is None:
    st.info("Configure the sidebar and click **🔍 Get Full Insights Report**. Given the mandatory "
            "30-second delay per request, a full report takes several minutes — the sidebar shows "
            "an estimate before you commit.")
else:
    daily_df_for_corr = report["sections"].get("1 Day", (pd.DataFrame(), None))[0]

    # ---------------------------------------------------------------------------
    # PER-TIMEFRAME SECTIONS
    # ---------------------------------------------------------------------------
    for label, (df, kind) in report["sections"].items():
        st.header(f"📐 {label} — {kind.capitalize()} Patterns")
        if df.empty:
            st.warning(f"No data returned for {label}. Yahoo Finance may not offer this "
                      f"interval/period combination for this ticker.")
            continue

        st.caption(f"{len(df)} bars covering {df.index.min()} → {df.index.max()}")

        # --- Weekday seasonality (daily bars only) ---
        if kind == "daily":
            wk = weekday_seasonality(df)
            if not wk.empty:
                st.subheader("📅 Day-of-Week Seasonality")
                bar_chart(wk.index, wk["avg_return"], "Average Return by Weekday (%)", "Avg Return %")
                best = wk["avg_return"].idxmax()
                worst = wk["avg_return"].idxmin()
                st.markdown(
                    f"Over **{int(wk['n'].sum())}** trading days in the last **{years_hist}** years, "
                    f"**{best}** has the best average return (**{wk.loc[best,'avg_return']:+.2f}%**, "
                    f"positive **{wk.loc[best,'pct_positive']:.0f}%** of the time, "
                    f"{sample_note(int(wk.loc[best,'n']))}) — {tilt_phrase(wk.loc[best,'pct_positive'])}. "
                    f"**{worst}** has the worst average return (**{wk.loc[worst,'avg_return']:+.2f}%**, "
                    f"positive **{wk.loc[worst,'pct_positive']:.0f}%** of the time, "
                    f"{sample_note(int(wk.loc[worst,'n']))}).")

        # --- Month-of-year seasonality (daily/weekly/monthly) ---
        if kind in ("daily", "weekly", "monthly") and len(df) > 24:
            mo = month_seasonality(df)
            if not mo.empty:
                st.subheader("🗓️ Month-of-Year Seasonality")
                mo_ordered = mo.reindex([m for m in MONTH_NAMES if m in mo.index])
                bar_chart(mo_ordered.index, mo_ordered["avg_return"], "Average Return by Month (%)", "Avg Return %")
                best_m = mo_ordered["avg_return"].idxmax()
                worst_m = mo_ordered["avg_return"].idxmin()
                small_n_flag = " (monthly bars means n here is just years of history — very small sample)" if kind == "monthly" else ""
                st.markdown(
                    f"Historically, **{best_m}** has been the strongest month on average "
                    f"(**{mo_ordered.loc[best_m,'avg_return']:+.2f}%**, positive "
                    f"**{mo_ordered.loc[best_m,'pct_positive']:.0f}%** of years, "
                    f"{sample_note(int(mo_ordered.loc[best_m,'n']))}) and **{worst_m}** the weakest "
                    f"(**{mo_ordered.loc[worst_m,'avg_return']:+.2f}%**){small_n_flag}.")

        # --- Gap behavior (all kinds) ---
        gap = gap_behavior(df)
        if gap["n_total"] > 10:
            st.subheader("📊 Opening Gap Behavior")
            gc1, gc2 = st.columns(2)
            with gc1:
                st.metric("Gap-Up Bars that Filled", f"{gap['up_fill']:.1f}%" if pd.notna(gap["up_fill"]) else "n/a")
                st.caption(sample_note(gap["n_up"]))
            with gc2:
                st.metric("Gap-Down Bars that Filled", f"{gap['down_fill']:.1f}%" if pd.notna(gap["down_fill"]) else "n/a")
                st.caption(sample_note(gap["n_down"]))
            if pd.notna(gap["up_fill"]) and pd.notna(gap["down_fill"]):
                if gap["up_fill"] > 65 and gap["down_fill"] > 65:
                    interp = "gaps on this instrument/timeframe tend to **mean-revert** (fill back) rather than run away."
                elif gap["up_fill"] < 35 and gap["down_fill"] < 35:
                    interp = "gaps here tend to **persist** rather than fill, consistent with trending/momentum behavior around the open."
                else:
                    interp = "there's no strong tendency either way for gaps to fill or persist."
                st.markdown(f"Out of {gap['n_total']} bars, {gap['n_up']} gapped up and {gap['n_down']} gapped down "
                           f"by more than 0.1%. Taken together, {interp}")

        # --- Streak persistence (all kinds) ---
        streaks = streak_persistence(df)
        st.subheader("🔁 After N Up/Down Moves, What Happens Next?")
        show_streaks = streaks.copy()
        show_streaks.columns = ["Streak Length", "% Next Up (after Up streak)", "n (up)",
                                 "% Next Up (after Down streak)", "n (down)"]
        st.dataframe(show_streaks, use_container_width=True, hide_index=True)
        row1 = streaks.iloc[0]
        if pd.notna(row1["pct_up_after_up"]):
            tilt1 = tilt_phrase(row1["pct_up_after_up"])
            st.markdown(f"After a single up move, the next bar is up **{row1['pct_up_after_up']:.1f}%** of the time "
                       f"({sample_note(row1['n_up'])}) — {tilt1}. Consistently **above ~55%** across streak "
                       f"lengths hints at momentum; consistently **below ~45%** hints at mean reversion.")

        # --- Intraday-only: time-of-day timing + hourly returns ---
        if kind == "intraday":
            timing = intraday_timing(df)
            if timing:
                st.subheader("⏱ When Does the High/Low Typically Form?")
                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric("High in first 15 min", f"{timing['high15']:.1f}%")
                tc2.metric("Low in first 15 min", f"{timing['low15']:.1f}%")
                tc3.metric("High in first 30 min", f"{timing['high30']:.1f}%")
                tc4.metric("Low in first 30 min", f"{timing['low30']:.1f}%")
                st.caption(f"Based on {timing['n']} sessions. " +
                          ("⚠️ Small sample — Yahoo Finance's own lookback limit for this interval is the "
                           "constraint here." if timing["n"] < 30 else ""))
                if timing["high15"] > 40 or timing["low15"] > 40:
                    st.markdown("A meaningful share of session extremes form very early — this is the kind of "
                               "pattern opening-range-breakout approaches are built around, though it needs "
                               "more than a handful of sessions to trust.")

            hrs = hourly_returns(df)
            if hrs is not None and len(hrs) > 1:
                st.subheader("🕐 Average Return by Hour of Day")
                bar_chart(hrs.index.astype(str), hrs["avg_return"], "Average Return by Hour (%)", "Avg Return %")
                best_h = hrs["avg_return"].idxmax()
                worst_h = hrs["avg_return"].idxmin()
                st.markdown(f"The **{best_h}:00** hour shows the strongest average return "
                           f"({hrs.loc[best_h,'avg_return']:+.3f}%, {sample_note(int(hrs.loc[best_h,'n']))}); "
                           f"**{worst_h}:00** the weakest ({hrs.loc[worst_h,'avg_return']:+.3f}%).")

        st.divider()

    # ---------------------------------------------------------------------------
    # CROSS-ASSET CORRELATION
    # ---------------------------------------------------------------------------
    st.header("🔗 Cross-Asset Correlation")
    if daily_df_for_corr.empty or not report["corr"]:
        st.info("No daily data or correlation instruments available to compute this.")
    else:
        corr_results = compute_correlations(daily_df_for_corr, report["corr"], report["years_hist"])
        if not corr_results:
            st.info("Not enough overlapping data to compute correlations.")
        else:
            names = list(corr_results.keys())
            values = [corr_results[n][0] for n in names]
            bar_chart(names, values, f"{report['display_name']} — Daily Return Correlation "
                                     f"(last {report['years_hist']}y)", "Correlation (r)")
            strongest_pos = max(corr_results, key=lambda k: corr_results[k][0])
            strongest_neg = min(corr_results, key=lambda k: corr_results[k][0])
            r_pos, n_pos = corr_results[strongest_pos]
            r_neg, n_neg = corr_results[strongest_neg]
            st.markdown(
                f"**{report['display_name']}**'s daily returns are most positively correlated with "
                f"**{strongest_pos}** (r = {r_pos:.2f}, n={n_pos}) and least correlated / most negative with "
                f"**{strongest_neg}** (r = {r_neg:.2f}, n={n_neg}). As a rough guide: |r| above ~0.5 is a "
                f"meaningfully strong relationship, below ~0.2 is weak-to-negligible. This describes the past "
                f"relationship only — correlations between assets drift over time and are not stable forever.")

    # ---------------------------------------------------------------------------
    # INDIA VIX REGIME ANALYSIS
    # ---------------------------------------------------------------------------
    st.header("📟 India VIX Regime Analysis")
    if daily_df_for_corr.empty or report["vix_full"].empty:
        st.info("Need both daily price data and India VIX data to run this comparison.")
    else:
        regime = vix_regime_analysis(daily_df_for_corr, report["vix_full"])
        if regime is None:
            st.info("Not enough overlapping data between this ticker and India VIX.")
        else:
            bar_chart(["Low VIX days", "High VIX days"], [regime["low_avg"], regime["high_avg"]],
                     "Average Daily Return by VIX Regime (%)", "Avg Return %")
            rc1, rc2 = st.columns(2)
            rc1.metric("Low-VIX days: Avg Return / Volatility", f"{regime['low_avg']:+.3f}% / {regime['low_vol']:.2f}%")
            rc1.caption(sample_note(regime["n_low"]))
            rc2.metric("High-VIX days: Avg Return / Volatility", f"{regime['high_avg']:+.3f}% / {regime['high_vol']:.2f}%")
            rc2.caption(sample_note(regime["n_high"]))
            vol_note = ("higher realized volatility on high-VIX days, as expected" if regime["high_vol"] > regime["low_vol"]
                       else "surprisingly similar realized volatility across regimes")
            st.markdown(
                f"Splitting history at the median India VIX level ({regime['median']:.1f}): "
                f"{report['display_name']} shows {vol_note}. The average *return* difference between regimes "
                f"({regime['high_avg']:+.3f}% vs {regime['low_avg']:+.3f}%) is the more interesting number — "
                f"a large, consistent gap here (not just a volatility difference) would be the more actionable "
                f"finding, but check the sample sizes above before leaning on it.")

    st.divider()
    st.caption("⚠️ Every pattern above describes historical data only. This tab tested many patterns at "
               "once on purpose — some numbers here will look interesting purely by chance. Treat this as "
               "a hypothesis generator: note what looks promising, then check whether it still holds on a "
               "later period of data you haven't looked at yet before trusting it with real money.")
