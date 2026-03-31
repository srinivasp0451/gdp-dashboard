"""
\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
\u2551    NIFTY EXPIRY DAY \u2014 LIVE ROLLING ATM STRADDLE TABLE               \u2551
\u2551    Data Source : NSE India Live Option Chain (no yfinance)          \u2551
\u2551    Author      : AlgoTrader Pro                                      \u2551
\u2551    Run         : streamlit run nifty_expiry_straddle.py             \u2551
\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime, date, time as dtime
import pytz
import time

# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# CONFIG
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
IST = pytz.timezone('Asia/Kolkata')
AUTO_REFRESH_SECONDS = 60

SESSION_BINS = [
    {"bin": 1, "label": "Opening",      "start": (9, 15),  "end": (9, 45)},
    {"bin": 2, "label": "Morning",      "start": (9, 45),  "end": (11, 0)},
    {"bin": 3, "label": "Mid-Session",  "start": (11, 0),  "end": (13, 0)},
    {"bin": 4, "label": "Afternoon",    "start": (13, 0),  "end": (14, 30)},
    {"bin": 5, "label": "Expiry Close", "start": (14, 30), "end": (15, 31)},
]

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/option-chain",
    "Connection": "keep-alive",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
}

# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# NSE SESSION \u2014 fresh session each call to avoid stale cookies
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def create_nse_session() -> requests.Session:
    """Bootstrap an NSE session with proper cookies."""
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    warm_urls = [
        "https://www.nseindia.com",
        "https://www.nseindia.com/option-chain",
    ]
    for url in warm_urls:
        try:
            session.get(url, timeout=10)
            time.sleep(0.5)
        except Exception:
            pass
    return session


def safe_get(session: requests.Session, url: str) -> dict | None:
    """GET with retry logic, returns parsed JSON or None."""
    for attempt in range(3):
        try:
            r = session.get(url, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.JSONDecodeError:
            time.sleep(1)
        except Exception as e:
            if attempt == 2:
                st.warning(f"\u26a0\ufe0f NSE fetch failed: {e}")
            time.sleep(1 + attempt)
    return None


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# DATA FETCH
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def fetch_all_data() -> dict:
    """
    Single function that fetches:
      - Nifty spot price
      - India VIX
      - Full option chain for nearest expiry
    Returns a dict with all values, or error flags.
    """
    result = {
        "ok": False,
        "spot": None,
        "vix": None,
        "expiry": None,
        "records": [],
        "error": "",
        "timestamp": datetime.now(IST).strftime("%H:%M:%S"),
    }

    session = create_nse_session()

    # \u2500\u2500 India VIX \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    vix_data = safe_get(session, "https://www.nseindia.com/api/allIndices")
    if vix_data:
        for idx in vix_data.get("data", []):
            name = idx.get("index", "")
            if "VIX" in name.upper():
                try:
                    result["vix"] = round(float(idx.get("last", 0)), 2)
                except Exception:
                    pass
                break

    # \u2500\u2500 Option Chain \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    oc_data = safe_get(
        session,
        "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    )
    if oc_data is None:
        result["error"] = "Option chain fetch failed. NSE may be blocking \u2014 try again."
        return result

    records_root = oc_data.get("records", {})
    spot = records_root.get("underlyingValue", 0)
    if not spot:
        result["error"] = "Spot price missing in NSE response."
        return result

    result["spot"] = round(float(spot), 2)
    expiry_dates = records_root.get("expiryDates", [])
    nearest = _nearest_expiry(expiry_dates)
    result["expiry"] = nearest

    all_records = records_root.get("data", [])
    result["records"] = [r for r in all_records if r.get("expiryDate") == nearest]
    result["ok"] = True
    return result


def _nearest_expiry(expiry_dates: list[str]) -> str | None:
    """Pick the earliest expiry >= today."""
    today = date.today()
    for exp in expiry_dates:
        try:
            d = datetime.strptime(exp, "%d-%b-%Y").date()
            if d >= today:
                return exp
        except Exception:
            pass
    return expiry_dates[0] if expiry_dates else None


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# OPTION CHAIN CALCULATIONS
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def find_atm(records: list, spot: float) -> int:
    strikes = [r["strikePrice"] for r in records if "strikePrice" in r]
    if not strikes:
        return 0
    return min(strikes, key=lambda x: abs(x - spot))


def straddle_premium(records: list, atm: int) -> tuple[float, float, float]:
    """Returns (total, ce_price, pe_price) for ATM strike."""
    for r in records:
        if r.get("strikePrice") == atm:
            ce = float(r.get("CE", {}).get("lastPrice", 0) or 0)
            pe = float(r.get("PE", {}).get("lastPrice", 0) or 0)
            return round(ce + pe, 2), round(ce, 2), round(pe, 2)
    return 0.0, 0.0, 0.0


def calc_pcr(records: list) -> float:
    ce_oi = sum(r.get("CE", {}).get("openInterest", 0) or 0 for r in records)
    pe_oi = sum(r.get("PE", {}).get("openInterest", 0) or 0 for r in records)
    if ce_oi == 0:
        return 0.0
    return round(pe_oi / ce_oi, 2)


def calc_total_oi(records: list) -> str:
    total = sum(
        (r.get("CE", {}).get("openInterest", 0) or 0)
        + (r.get("PE", {}).get("openInterest", 0) or 0)
        for r in records
    )
    if total >= 1_00_00_000:
        return f"{total / 1_00_00_000:.2f}Cr"
    return f"{total / 1_00_000:.1f}L"


def calc_max_pain(records: list) -> int:
    """
    Max pain = strike at which total open interest writer loss is minimum.
    CE writers lose when spot > strike, PE writers lose when spot < strike.
    """
    strikes = sorted({r["strikePrice"] for r in records if "strikePrice" in r})
    if not strikes:
        return 0

    # Pre-compute OI per strike
    ce_oi = {r["strikePrice"]: r.get("CE", {}).get("openInterest", 0) or 0 for r in records}
    pe_oi = {r["strikePrice"]: r.get("PE", {}).get("openInterest", 0) or 0 for r in records}

    min_loss = float("inf")
    pain_strike = 0

    for test in strikes:
        loss = 0
        for s in strikes:
            if test > s:
                loss += (test - s) * ce_oi.get(s, 0)
            if test < s:
                loss += (s - test) * pe_oi.get(s, 0)
        if loss < min_loss:
            min_loss = loss
            pain_strike = test

    return pain_strike


def calc_atm_ce_oi_pe_oi(records: list, atm: int) -> tuple[str, str]:
    """Return CE OI and PE OI at ATM strike."""
    for r in records:
        if r.get("strikePrice") == atm:
            ce = r.get("CE", {}).get("openInterest", 0) or 0
            pe = r.get("PE", {}).get("openInterest", 0) or 0
            return f"{ce/100:.0f}K", f"{pe/100:.0f}K"
    return "\u2014", "\u2014"


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# TIME BIN LOGIC
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def get_current_bin() -> int:
    """
    Returns:
      0  \u2192 Pre-market (before 9:15)
      1\u20135 \u2192 Active bin
      6  \u2192 Post-market (after 15:31)
    """
    now = datetime.now(IST).time()
    for b in SESSION_BINS:
        s = dtime(*b["start"])
        e = dtime(*b["end"])
        if s <= now < e:
            return b["bin"]
    if now >= dtime(15, 31):
        return 6
    return 0


def market_status() -> str:
    now = datetime.now(IST).time()
    if now < dtime(9, 15):
        return "\ud83d\udd34 Pre-Market"
    if now >= dtime(15, 30):
        return "\ud83d\udd34 Market Closed"
    return "\ud83d\udfe2 Market Live"


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# OBSERVATIONS ENGINE
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def generate_observations(
    snapshots: dict,
    spot: float,
    vix: float,
    max_pain: int,
    pcr: float,
    straddle: float,
    spot_open: float,
    expiry: str,
) -> list[str]:
    """Returns a list of human-readable markdown strings."""
    obs = []

    # \u2500\u2500 VIX Analysis \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if vix:
        if vix >= 28:
            obs.append(
                f"\ud83d\udd34 **VIX CRITICAL at {vix}**: Far above the normal expiry-day range of 12\u201318. "
                f"This level of VIX signals extreme fear and inflates straddle premiums massively \u2014 "
                f"option sellers are at high risk, buyers benefit from elevated IV."
            )
        elif vix >= 22:
            obs.append(
                f"\ud83d\udfe0 **VIX Elevated at {vix}**: Market is fearful. Straddle premiums are significantly "
                f"higher than a calm expiry. Expect wider moves and possible IV crush post-event."
            )
        elif vix >= 16:
            obs.append(
                f"\ud83d\udfe1 **VIX Moderate at {vix}**: Slightly above comfort zone. Premiums are mildly inflated. "
                f"Expiry-day theta decay will be normal."
            )
        else:
            obs.append(
                f"\ud83d\udfe2 **VIX Calm at {vix}**: Normal expiry conditions. Straddle premiums are in the expected "
                f"range. Theta decay is the dominant force today."
            )

    # \u2500\u2500 Spot vs Max Pain \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if spot and max_pain:
        diff = spot - max_pain
        abs_diff = abs(diff)
        pct_diff = round(abs_diff / max_pain * 100, 2)
        if abs_diff < 80:
            obs.append(
                f"\ud83d\udccd **Spot ({spot:.0f}) is within {abs_diff:.0f} pts of Max Pain ({max_pain})**: "
                f"The market is gravitating toward Max Pain \u2014 this is where options writers lose the least. "
                f"Expect resistance to big moves from here."
            )
        elif diff < 0:
            obs.append(
                f"\ud83d\udcc9 **Spot ({spot:.0f}) is {abs_diff:.0f} pts BELOW Max Pain ({max_pain}) [{pct_diff}%]**: "
                f"Bears are fully in control, significantly overriding max pain magnetism. "
                f"This indicates strong institutional selling pressure. "
                f"A pullback toward {max_pain} is possible but NOT guaranteed \u2014 "
                f"large FII short positions can keep spot pinned below."
            )
        else:
            obs.append(
                f"\ud83d\udcc8 **Spot ({spot:.0f}) is {abs_diff:.0f} pts ABOVE Max Pain ({max_pain}) [{pct_diff}%]**: "
                f"Bulls are pushing above max pain. CE writers are under pressure. "
                f"Watch for profit-taking / rollover activity near max pain level."
            )

    # \u2500\u2500 PCR Sentiment \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if pcr:
        if pcr < 0.65:
            obs.append(
                f"\ud83d\udc3b **PCR at {pcr} \u2014 Extremely Bearish**: Calls are dominating put OI significantly. "
                f"Either aggressive put buying OR call writing at scale. High probability of continued bearish pressure."
            )
        elif pcr < 0.85:
            obs.append(
                f"\ud83d\udd34 **PCR at {pcr} \u2014 Bearish Bias**: More puts than calls in the market, "
                f"indicating fear and downside expectation."
            )
        elif pcr <= 1.15:
            obs.append(
                f"\u2696\ufe0f **PCR at {pcr} \u2014 Neutral**: Call and Put OI are balanced. "
                f"Market is undecided on direction \u2014 range-bound action likely near ATM."
            )
        else:
            obs.append(
                f"\ud83d\udfe2 **PCR at {pcr} \u2014 Bullish**: Strong put writing activity signals confidence "
                f"in holding current levels or moving up. Contrarian signal \u2014 watch for reversal if PCR spikes."
            )

    # \u2500\u2500 Straddle Decay Trend \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    bins_with_data = sorted(snapshots.keys())
    premiums = [snapshots[b]["straddle"] for b in bins_with_data if snapshots[b].get("straddle")]
    if len(premiums) >= 2:
        first_p = premiums[0]
        last_p = premiums[-1]
        decay = first_p - last_p
        decay_pct = round(decay / first_p * 100, 1) if first_p > 0 else 0
        if decay_pct >= 60:
            intensity = "\ud83d\udd25 Aggressive"
            comment = "Theta is dominating \u2014 straddle is melting fast. Option buyers losing money rapidly."
        elif decay_pct >= 30:
            intensity = "\u26a1 Moderate"
            comment = "Normal expiry-day decay pattern. Straddle is losing value steadily."
        else:
            intensity = "\ud83d\udc22 Slow"
            comment = "Premium is sticky \u2014 possibly due to high VIX or ongoing directional move."
        obs.append(
            f"\u23f1\ufe0f **Theta Decay \u2014 {intensity}**: "
            f"Straddle premium dropped from \u20b9{first_p} (opening) \u2192 \u20b9{last_p} now. "
            f"That's \u20b9{decay:.0f} ({decay_pct}%) eroded. {comment}"
        )

    # \u2500\u2500 Spot Movement \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if spot_open and spot:
        chg = spot - spot_open
        pct = round(chg / spot_open * 100, 2)
        direction = "up" if chg > 0 else "down"
        emoji = "\ud83d\udcc8" if chg > 0 else "\ud83d\udcc9"
        bias = (
            "Bullish momentum. Watch for call writers coming under pressure."
            if chg > 0
            else "Bearish momentum. Put writers are bleeding. OTM puts may have turned ITM."
        )
        obs.append(
            f"{emoji} **Spot moved {direction} {abs(chg):.0f} pts ({abs(pct)}%)** "
            f"from day-open of {spot_open:.0f} to current {spot:.0f}. {bias}"
        )

    # \u2500\u2500 ATM Shift \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if len(bins_with_data) >= 2:
        first_atm = snapshots[bins_with_data[0]]["atm"]
        last_atm = snapshots[bins_with_data[-1]]["atm"]
        atm_shift = abs(last_atm - first_atm)
        if atm_shift > 0:
            direction = "lower" if last_atm < first_atm else "higher"
            obs.append(
                f"\ud83c\udfaf **Rolling ATM shifted {direction} by {atm_shift} pts** "
                f"({first_atm} \u2192 {last_atm}). "
                f"This means a straddle bought at opening would now be {'deep ITM on PE side' if direction == 'lower' else 'deep ITM on CE side'} \u2014 "
                f"{'CE is worthless, full loss on that leg' if direction == 'lower' else 'PE is worthless, full loss on that leg'}."
            )

    # \u2500\u2500 Trading Strategy Signal \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    current_bin = get_current_bin()
    if current_bin in [1, 2] and straddle:
        if vix and vix > 22:
            obs.append(
                f"\u26a0\ufe0f **Strategy Alert \u2014 Do NOT buy straddle at open with VIX {vix}**: "
                f"High VIX = inflated premium = IV crush will destroy your straddle even if market moves. "
                f"Better approach: Wait for VIX to spike further, then sell OTM strangles OR "
                f"wait for a sharp directional move and buy only the directional leg."
            )
        else:
            obs.append(
                f"\ud83d\udca1 **Strategy Signal (Bin {current_bin})**: "
                f"Straddle at \u20b9{straddle} with VIX {vix}. "
                f"If spot moves more than \u20b9{round(straddle * 1.5, 0):.0f} pts from ATM, "
                f"the straddle buyer profits. Otherwise, theta wins."
            )

    if not obs:
        obs.append("\ud83d\udcca Waiting for market to open (9:15 IST) to begin capturing data...")

    return obs


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# SESSION STATE INIT
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
if "snapshots" not in st.session_state:
    st.session_state.snapshots = {}       # {bin_num: snapshot_dict}
if "spot_open" not in st.session_state:
    st.session_state.spot_open = None
if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0
if "last_fetch_time" not in st.session_state:
    st.session_state.last_fetch_time = None


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# PAGE LAYOUT
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
st.set_page_config(
    page_title="Nifty Expiry Straddle Table",
    page_icon="\ud83d\udcca",
    layout="wide",
)

st.markdown("""
<style>
    .main-title { font-size: 26px; font-weight: 700; color: #1a1a2e; }
    .sub-title  { font-size: 13px; color: #555; margin-top: -10px; }
    .bin-badge  { padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
    .bin-live   { background: #fff3cd; color: #856404; }
    .bin-done   { background: #d1e7dd; color: #0f5132; }
    .bin-wait   { background: #e2e3e5; color: #41464b; }
    .obs-box    { background: #f8f9fa; border-left: 4px solid #0d6efd;
                  padding: 16px; border-radius: 6px; margin: 8px 0; }
    div[data-testid="stMetric"] { background: #f8f9fa; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# \u2500\u2500 Header \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
now_ist = datetime.now(IST)
st.markdown('<div class="main-title">\ud83d\udcca Nifty Expiry Day \u2014 Live Rolling ATM Straddle Table</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">Captures live NSE data across 5 session bins | '
            f'IST: {now_ist.strftime("%d-%b-%Y %H:%M:%S")} | {market_status()}</div>', unsafe_allow_html=True)

# \u2500\u2500 Control Bar \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
st.markdown("---")
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 3])
with ctrl1:
    auto_ref = st.toggle("\ud83d\udd04 Auto Refresh", value=True)
with ctrl2:
    refresh_interval = st.selectbox("Interval", [30, 60, 120], index=1, label_visibility="collapsed")
with ctrl3:
    manual_refresh = st.button("\u26a1 Fetch Now", type="primary")
with ctrl4:
    st.caption(f"Refresh #{st.session_state.refresh_count} | Last fetch: {st.session_state.last_fetch_time or 'Never'}")

if manual_refresh:
    st.session_state.refresh_count += 1
    st.rerun()

# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# FETCH LIVE DATA
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
with st.spinner("\ud83d\udd17 Connecting to NSE India \u2014 fetching live option chain..."):
    data = fetch_all_data()

if not data["ok"]:
    st.error(f"\u274c {data['error']}")
    st.info("\ud83d\udca1 NSE blocks rapid requests. Wait 30\u201360s and retry. "
            "If persistent, it may be outside market hours or NSE is under maintenance.")
    st.stop()

# \u2500\u2500 Unpack \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
spot      = data["spot"]
vix       = data["vix"]
records   = data["records"]
expiry    = data["expiry"]
fetch_ts  = data["timestamp"]

atm             = find_atm(records, spot)
straddle, ce_px, pe_px = straddle_premium(records, atm)
pcr             = calc_pcr(records)
total_oi        = calc_total_oi(records)
max_pain        = calc_max_pain(records)
atm_ce_oi, atm_pe_oi = calc_atm_ce_oi_pe_oi(records, atm)

st.session_state.last_fetch_time = fetch_ts

# \u2500\u2500 Record spot at open (Bin 1) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
current_bin = get_current_bin()
if st.session_state.spot_open is None and current_bin in [1, 2]:
    st.session_state.spot_open = spot

# \u2500\u2500 Store snapshot for current bin \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
snapshot = {
    "vix":      vix,
    "spot":     spot,
    "atm":      atm,
    "straddle": straddle,
    "ce_px":    ce_px,
    "pe_px":    pe_px,
    "max_pain": max_pain,
    "pcr":      pcr,
    "total_oi": total_oi,
    "atm_ce":   atm_ce_oi,
    "atm_pe":   atm_pe_oi,
    "time":     now_ist.strftime("%H:%M"),
}

if current_bin in range(1, 7):
    # Always update the current bin with latest live data
    st.session_state.snapshots[current_bin] = snapshot

    # If bin 6 (post-market), also lock bin 5 if not already captured
    if current_bin == 6 and 5 not in st.session_state.snapshots:
        st.session_state.snapshots[5] = snapshot


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# LIVE METRICS BAR
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
st.subheader(f"\ud83d\udce1 Live Data \u2014 Expiry: {expiry}")
m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)

open_str = f"{st.session_state.spot_open:.0f}" if st.session_state.spot_open else "\u2014"
spot_delta = f"{spot - st.session_state.spot_open:+.0f} from open" if st.session_state.spot_open else None

m1.metric("Nifty Spot",     f"{spot:.2f}",   spot_delta)
m2.metric("India VIX",      f"{vix}",         None)
m3.metric("ATM Strike",     f"{atm}")
m4.metric("CE (ATM)",       f"\u20b9{ce_px}")
m5.metric("PE (ATM)",       f"\u20b9{pe_px}")
m6.metric("Straddle",       f"\u20b9{straddle}")
m7.metric("Max Pain",       f"{max_pain}")
m8.metric("PCR (OI)",       f"{pcr}")


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# 5-BIN TABLE
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
st.markdown("---")
st.subheader("\ud83d\uddc2\ufe0f 5-Bin Expiry Session Table (Rolling ATM Straddle)")

rows = []
for b in SESSION_BINS:
    bn     = b["bin"]
    label  = f"Bin {bn} \u2014 {b['label']}\n{b['start'][0]:02d}:{b['start'][1]:02d} \u2013 {b['end'][0]:02d}:{b['end'][1]:02d}"
    is_live = (bn == current_bin)
    is_past = (bn < current_bin) or (current_bin == 6)

    if bn in st.session_state.snapshots:
        s   = st.session_state.snapshots[bn]
        tag = "\ud83d\udd34 LIVE" if is_live else "\u2705 Captured"
        rows.append({
            "Session Bin":   label,
            "Status":        tag,
            "India VIX":     s["vix"],
            "Nifty Spot":    s["spot"],
            "ATM Strike":    s["atm"],
            "CE (ATM) \u20b9":    s["ce_px"],
            "PE (ATM) \u20b9":    s["pe_px"],
            "Straddle \u20b9":    s["straddle"],
            "Max Pain":      s["max_pain"],
            "PCR (OI)":      s["pcr"],
            "Total OI":      s["total_oi"],
            "ATM CE OI":     s["atm_ce"],
            "ATM PE OI":     s["atm_pe"],
            "Captured At":   s["time"],
        })
    elif is_past and bn not in st.session_state.snapshots:
        rows.append({
            "Session Bin":   label,
            "Status":        "\u26a0\ufe0f Missed",
            "India VIX":     "\u2014", "Nifty Spot": "\u2014", "ATM Strike": "\u2014",
            "CE (ATM) \u20b9":    "\u2014", "PE (ATM) \u20b9": "\u2014",
