"""
TutorNest: single-file version.  Run:  streamlit run tutornest_single.py
Settings (price, DB, payments...) come from Streamlit secrets or env vars;
see the CONFIG section below.  DATABASE_URL examples:
  sqlite:///tutornest.db | mysql://user:pass@host:3306/db | postgresql://user:pass@host:5432/db
Drivers: pip install pymysql  (MySQL)  /  psycopg2-binary  (Postgres)
"""
import hashlib
import hmac
import html
import os
import re
import secrets
from datetime import datetime, timedelta

import requests
import streamlit as st
from sqlalchemy import (
    Column, DateTime, ForeignKey, Integer, MetaData, String, Table, Text,
    UniqueConstraint, create_engine, desc, func, insert, select, update,
)


# ==========================================================================
# CONFIG
# ==========================================================================
def _get(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


# ---- Branding -------------------------------------------------------------
SITE_NAME = _get("SITE_NAME", "TutorNest")
TAGLINE = _get("TAGLINE", "Verified tuition requirements, delivered to tutors")
SUPPORT_PHONE = _get("SUPPORT_PHONE", "+91 90000 00000")
SUPPORT_EMAIL = _get("SUPPORT_EMAIL", "support@example.com")

# ---- Plan (configurable) --------------------------------------------------
PLAN_PRICE_INR = int(_get("PLAN_PRICE_INR", 1200))
PLAN_LEADS = int(_get("PLAN_LEADS", 10))
PLAN_VALIDITY_DAYS = int(_get("PLAN_VALIDITY_DAYS", 365))

# ---- Lead privacy ----------------------------------------------------------
# These fields are masked until a tutor spends one credit to unlock the lead.
LOCKED_FIELDS = ["parent_name", "phone", "email", "address"]
# How many characters of the free-text requirement are shown before unlocking.
DETAILS_PREVIEW_CHARS = int(_get("DETAILS_PREVIEW_CHARS", 70))

# ---- Payments ---------------------------------------------------------------
# Comma separated: "cash", "online" (Razorpay: cards, netbanking, UPI/GPay/PhonePe)
PAYMENT_METHODS = [m.strip() for m in str(_get("PAYMENT_METHODS", "cash")).split(",") if m.strip()]
# For testing: cash payments activate instantly. Set to false in production so
# the admin has to confirm that cash was really received.
CASH_AUTO_APPROVE = str(_get("CASH_AUTO_APPROVE", "true")).lower() == "true"
RAZORPAY_KEY_ID = _get("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = _get("RAZORPAY_KEY_SECRET", "")
APP_URL = _get("APP_URL", "")  # your public streamlit URL, used as payment return link

# ---- Infrastructure ---------------------------------------------------------
# Any SQLAlchemy URL works:
#   sqlite:///tutornest.db                         (local file; lost on Streamlit Cloud restarts)
#   mysql://USER:PASSWORD@HOST:3306/DBNAME         (MySQL / MariaDB)
#   postgresql://USER:PASSWORD@HOST:5432/DBNAME    (Postgres)
# Streamlit Cloud wipes local files, so use a hosted MySQL/Postgres there.
DATABASE_URL = _get("DATABASE_URL", "sqlite:///tutornest_dev.db")
ADMIN_PASSWORD = _get("ADMIN_PASSWORD", "")  # empty = admin panel disabled

CLASS_LEVELS = [
    "Pre-school / KG", "Class 1-5", "Class 6-8", "Class 9-10", "Class 11-12",
    "Degree / College", "Competitive exams", "Languages", "Music / Arts / Hobby",
]
MODES = ["Home tuition", "Online", "Either"]
GENDER_PREF = ["No preference", "Female tutor", "Male tutor"]


# ==========================================================================
# DATABASE
# ==========================================================================
md = MetaData()

users = Table(
    "users", md,
    Column("id", Integer, primary_key=True),
    Column("name", String(120)),
    Column("email", String(200), unique=True, nullable=False),
    Column("phone", String(20)),
    Column("pw_hash", String(200)),
    Column("salt", String(64)),
    Column("created_at", DateTime, default=datetime.utcnow),
)
sessions = Table(
    "sessions", md,
    Column("token", String(64), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("expires_at", DateTime),
)
leads = Table(
    "leads", md,
    Column("id", Integer, primary_key=True),
    Column("parent_name", String(120)),
    Column("phone", String(20)),
    Column("email", String(200)),
    Column("address", Text),
    Column("class_level", String(60)),
    Column("subject", String(200)),
    Column("mode", String(30)),
    Column("city", String(80)),
    Column("area", String(120)),
    Column("pincode", String(10)),
    Column("budget", String(60)),
    Column("gender_pref", String(30)),
    Column("details", Text),
    Column("status", String(20), default="open"),
    Column("created_at", DateTime, default=datetime.utcnow),
)
payments = Table(
    "payments", md,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("amount", Integer),
    Column("method", String(20)),
    Column("provider_ref", String(100)),
    Column("pay_url", Text),
    Column("status", String(20), default="pending"),  # pending | paid
    Column("created_at", DateTime, default=datetime.utcnow),
)
subscriptions = Table(
    "subscriptions", md,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("payment_id", Integer, ForeignKey("payments.id")),
    Column("credits_total", Integer),
    Column("credits_left", Integer),
    Column("starts_at", DateTime),
    Column("expires_at", DateTime),
)
unlocks = Table(
    "unlocks", md,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("lead_id", Integer, ForeignKey("leads.id")),
    Column("created_at", DateTime, default=datetime.utcnow),
    UniqueConstraint("user_id", "lead_id", name="uq_unlock"),
)


_DRIVER_PREFIXES = (
    ("postgres://", "postgresql+psycopg2://"),
    ("postgresql://", "postgresql+psycopg2://"),
    ("mysql://", "mysql+pymysql://"),
    ("mariadb://", "mysql+pymysql://"),
)


def normalize_url(url):
    """Accept plain URLs (mysql://, postgres://, sqlite:///) and add the driver."""
    url = (url or "").strip()
    for old, new in _DRIVER_PREFIXES:
        if url.startswith(old):
            return new + url[len(old):]
    return url


def backend_name():
    return normalize_url(DATABASE_URL).split(":", 1)[0].split("+", 1)[0]


@st.cache_resource
def engine():
    url = normalize_url(DATABASE_URL)
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_recycle"] = 280  # MySQL drops idle connections
    try:
        e = create_engine(url, **kwargs)
        md.create_all(e)
    except ModuleNotFoundError as ex:
        st.error(f"Database driver missing: {ex}. Install it with pip "
                 "(pymysql for MySQL, psycopg2-binary for Postgres).")
        st.stop()
    except Exception as ex:
        st.error(f"Could not connect to the database ({backend_name()}). "
                 f"Check DATABASE_URL. Details: {type(ex).__name__}")
        st.stop()
    return e


def _now():
    return datetime.utcnow()


def _rows(res):
    return [dict(r._mapping) for r in res]


# ---- auth -------------------------------------------------------------------
def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 120_000).hex()


def create_user(name, email, phone, password):
    email = email.strip().lower()
    salt = secrets.token_hex(16)
    with engine().begin() as c:
        if c.execute(select(users.c.id).where(users.c.email == email)).first():
            return None
        r = c.execute(insert(users).values(
            name=name.strip(), email=email, phone=phone.strip(),
            pw_hash=_hash(password, salt), salt=salt, created_at=_now()))
        return r.inserted_primary_key[0]


def verify_user(email, password):
    with engine().connect() as c:
        row = c.execute(select(users).where(users.c.email == email.strip().lower())).first()
    if not row:
        return None
    u = dict(row._mapping)
    if hmac.compare_digest(u["pw_hash"], _hash(password, u["salt"])):
        return {k: u[k] for k in ("id", "name", "email", "phone")}
    return None


def create_session(user_id, days=30):
    token = secrets.token_urlsafe(32)
    with engine().begin() as c:
        c.execute(insert(sessions).values(token=token, user_id=user_id,
                                          expires_at=_now() + timedelta(days=days)))
    return token


def get_session_user(token):
    with engine().connect() as c:
        row = c.execute(
            select(users.c.id, users.c.name, users.c.email, users.c.phone)
            .select_from(sessions.join(users, users.c.id == sessions.c.user_id))
            .where(sessions.c.token == token, sessions.c.expires_at > _now())
        ).first()
    return dict(row._mapping) if row else None


def delete_session(token):
    with engine().begin() as c:
        c.execute(sessions.delete().where(sessions.c.token == token))


# ---- leads ------------------------------------------------------------------
def add_lead(**data):
    data["created_at"] = _now()
    data["status"] = "open"
    with engine().begin() as c:
        return c.execute(insert(leads).values(**data)).inserted_primary_key[0]


def list_leads(class_level=None, search=None, limit=60, include_closed=False):
    q = select(leads).order_by(desc(leads.c.created_at)).limit(limit)
    if not include_closed:
        q = q.where(leads.c.status == "open")
    if class_level and class_level != "All":
        q = q.where(leads.c.class_level == class_level)
    if search:
        s = f"%{search.strip().lower()}%"
        q = q.where(
            func.lower(leads.c.area).like(s) | func.lower(leads.c.city).like(s)
            | func.lower(leads.c.subject).like(s) | leads.c.pincode.like(s))
    with engine().connect() as c:
        return _rows(c.execute(q))


def set_lead_status(lead_id, status):
    with engine().begin() as c:
        c.execute(update(leads).where(leads.c.id == lead_id).values(status=status))


# ---- payments & subscriptions ----------------------------------------------
def create_payment(user_id, amount, method, provider_ref=None, pay_url=None, status="pending"):
    with engine().begin() as c:
        return c.execute(insert(payments).values(
            user_id=user_id, amount=amount, method=method, provider_ref=provider_ref,
            pay_url=pay_url, status=status, created_at=_now())).inserted_primary_key[0]


def update_payment_link(payment_id, provider_ref, pay_url):
    with engine().begin() as c:
        c.execute(update(payments).where(payments.c.id == payment_id)
                  .values(provider_ref=provider_ref, pay_url=pay_url))


def mark_paid(payment_id):
    """Idempotent: marks payment paid and activates exactly one subscription."""
    with engine().begin() as c:
        p = c.execute(select(payments).where(payments.c.id == payment_id)).first()
        if not p or p._mapping["status"] == "paid":
            return False
        c.execute(update(payments).where(payments.c.id == payment_id).values(status="paid"))
        now = _now()
        c.execute(insert(subscriptions).values(
            user_id=p._mapping["user_id"], payment_id=payment_id,
            credits_total=PLAN_LEADS, credits_left=PLAN_LEADS,
            starts_at=now, expires_at=now + timedelta(days=PLAN_VALIDITY_DAYS)))
        return True


def pending_payments(user_id):
    with engine().connect() as c:
        return _rows(c.execute(select(payments).where(
            payments.c.user_id == user_id, payments.c.status == "pending")
            .order_by(desc(payments.c.id))))


def all_pending_cash():
    with engine().connect() as c:
        return _rows(c.execute(
            select(payments.c.id, payments.c.amount, payments.c.created_at,
                   users.c.name, users.c.phone, users.c.email)
            .select_from(payments.join(users, users.c.id == payments.c.user_id))
            .where(payments.c.status == "pending", payments.c.method == "cash")
            .order_by(payments.c.id)))


def credits_summary(user_id):
    """Total remaining credits across non-expired plans and the latest expiry."""
    with engine().connect() as c:
        r = c.execute(select(func.coalesce(func.sum(subscriptions.c.credits_left), 0),
                             func.max(subscriptions.c.expires_at))
                      .where(subscriptions.c.user_id == user_id,
                             subscriptions.c.expires_at > _now(),
                             subscriptions.c.credits_left > 0)).first()
        total_paid = c.execute(select(func.count(subscriptions.c.id))
                               .where(subscriptions.c.user_id == user_id)).scalar()
    return {"credits": int(r[0] or 0), "expires": r[1], "plans_bought": int(total_paid or 0)}


# ---- unlocks ----------------------------------------------------------------
def unlocked_ids(user_id):
    with engine().connect() as c:
        return {r[0] for r in c.execute(select(unlocks.c.lead_id).where(unlocks.c.user_id == user_id))}


def unlock_lead(user_id, lead_id):
    """Spend one credit (earliest-expiring plan first). Returns True on success."""
    with engine().begin() as c:
        if c.execute(select(unlocks.c.id).where(unlocks.c.user_id == user_id,
                                                unlocks.c.lead_id == lead_id)).first():
            return True
        sub = c.execute(select(subscriptions.c.id).where(
            subscriptions.c.user_id == user_id, subscriptions.c.credits_left > 0,
            subscriptions.c.expires_at > _now()).order_by(subscriptions.c.expires_at)).first()
        if not sub:
            return False
        res = c.execute(update(subscriptions)
                        .where(subscriptions.c.id == sub[0], subscriptions.c.credits_left > 0)
                        .values(credits_left=subscriptions.c.credits_left - 1))
        if res.rowcount != 1:
            return False
        c.execute(insert(unlocks).values(user_id=user_id, lead_id=lead_id, created_at=_now()))
        return True


def my_unlocked_leads(user_id):
    with engine().connect() as c:
        return _rows(c.execute(
            select(leads).select_from(unlocks.join(leads, leads.c.id == unlocks.c.lead_id))
            .where(unlocks.c.user_id == user_id).order_by(desc(unlocks.c.created_at))))


# ---- admin stats --------------------------------------------------------------
def stats():
    with engine().connect() as c:
        return {
            "tutors": c.execute(select(func.count(users.c.id))).scalar(),
            "leads": c.execute(select(func.count(leads.c.id))).scalar(),
            "revenue": c.execute(select(func.coalesce(func.sum(payments.c.amount), 0))
                                 .where(payments.c.status == "paid")).scalar(),
            "unlocks": c.execute(select(func.count(unlocks.c.id))).scalar(),
        }


# ==========================================================================
# PAYMENTS
# ==========================================================================
RZP = "https://api.razorpay.com/v1/payment_links"


def online_configured():
    return bool(RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET)


def available_methods():
    out = []
    for m in PAYMENT_METHODS:
        if m == "online" and not online_configured():
            continue
        if m in ("cash", "online"):
            out.append(m)
    return out or ["cash"]


def start_checkout(user, method):
    """Returns (payment_id, message). For online, payment row carries pay_url."""
    amount = PLAN_PRICE_INR
    if method == "cash":
        pid = create_payment(user["id"], amount, "cash")
        if CASH_AUTO_APPROVE:
            mark_paid(pid)
            return pid, "Cash payment recorded and plan activated (test mode)."
        return pid, "Cash payment recorded. Your plan activates once we confirm the cash."

    pid = create_payment(user["id"], amount, "online")
    payload = {
        "amount": amount * 100,
        "currency": "INR",
        "description": f"{SITE_NAME}: {PLAN_LEADS} leads, {PLAN_VALIDITY_DAYS} days",
        "reference_id": f"pay_{pid}",
        "customer": {"name": user["name"], "email": user["email"], "contact": user.get("phone") or ""},
        "notify": {"sms": False, "email": False},
        "reminder_enable": False,
    }
    if APP_URL:
        payload["callback_url"] = APP_URL
        payload["callback_method"] = "get"
    r = requests.post(RZP, json=payload,
                      auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET), timeout=20)
    r.raise_for_status()
    d = r.json()
    update_payment_link(pid, d["id"], d["short_url"])
    return pid, "Complete the payment on the secure page, then come back and tap 'I've paid'."


def refresh_online(payment):
    """Check a pending online payment with Razorpay; activate plan if paid."""
    if payment["method"] != "online" or not payment.get("provider_ref"):
        return False
    try:
        r = requests.get(f"{RZP}/{payment['provider_ref']}",
                         auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET), timeout=20)
        r.raise_for_status()
        if r.json().get("status") == "paid":
            return mark_paid(payment["id"])
    except Exception:
        return False
    return False


# ==========================================================================
# UI / STYLING
# ==========================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Plus Jakarta Sans', sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.stApp { background: linear-gradient(180deg,#f3f7f6 0%,#eef2fb 100%); }
.block-container { padding-top: 1rem; padding-bottom: 4rem; max-width: 760px; }

.hero {
  background: radial-gradient(1200px 400px at 90% -10%, #6c63ff55, transparent),
              linear-gradient(135deg,#0f766e 0%,#115e59 45%,#312e81 100%);
  color:#fff; border-radius: 26px; padding: 28px 24px 24px; margin-bottom: 18px;
  box-shadow: 0 18px 40px -18px #115e59aa;
}
.hero h1 { margin:0; font-size: 1.9rem; font-weight: 800; letter-spacing:-.5px; color:#fff; }
.hero p  { margin:.4rem 0 0; opacity:.9; font-size:.98rem; }
.hero .pills { margin-top:16px; display:flex; gap:8px; flex-wrap:wrap; }
.hero .pill { background:#ffffff22; border:1px solid #ffffff33; padding:6px 12px;
              border-radius:999px; font-size:.78rem; font-weight:600; backdrop-filter: blur(6px); }

.card { background:#fff; border-radius:20px; padding:18px 18px 14px; margin: 14px 0 4px;
        border:1px solid #e6ecf0; box-shadow: 0 10px 30px -18px #0f172a55; }
.card .top { display:flex; justify-content:space-between; align-items:flex-start; gap:10px; }
.card h3 { margin:0; font-size:1.08rem; font-weight:700; color:#0f172a; line-height:1.3; }
.card .when { font-size:.72rem; color:#64748b; white-space:nowrap; margin-top:3px; }
.chips { display:flex; flex-wrap:wrap; gap:6px; margin:10px 0 8px; }
.chip { background:#ecfdf5; color:#047857; border-radius:999px; padding:4px 10px;
        font-size:.74rem; font-weight:600; }
.chip.blue { background:#eef2ff; color:#4338ca; }
.chip.amber { background:#fffbeb; color:#b45309; }
.chip.rose { background:#fff1f2; color:#be123c; }
.desc { color:#475569; font-size:.9rem; margin:6px 0 10px; }

.locked { border:1.5px dashed #c7d2fe; background: linear-gradient(135deg,#f8faff,#f1f5ff);
          border-radius:14px; padding:10px 14px; font-size:.86rem; color:#475569; }
.locked b { color:#312e81; }
.locked .blur { filter: blur(4px); user-select:none; letter-spacing:1px; }
.unlocked { border:1.5px solid #86efac; background:#f0fdf4; border-radius:14px;
            padding:12px 14px; font-size:.92rem; color:#14532d; }
.unlocked div { margin: 2px 0; }

.plan { background:linear-gradient(135deg,#312e81,#4f46e5 60%,#7c3aed); color:#fff;
        border-radius:24px; padding:24px; box-shadow: 0 20px 40px -20px #4f46e5; }
.plan .price { font-size:2.4rem; font-weight:800; letter-spacing:-1px; }
.plan .price small { font-size:.9rem; font-weight:500; opacity:.85; }
.plan ul { margin:12px 0 0; padding-left:1.1rem; line-height:1.9; font-size:.95rem; }

.stat { background:#fff; border:1px solid #e6ecf0; border-radius:18px; padding:14px 16px;
        box-shadow: 0 8px 24px -18px #0f172a66; }
.stat .n { font-size:1.6rem; font-weight:800; color:#0f766e; }
.stat .l { font-size:.78rem; color:#64748b; font-weight:600; }

.stButton>button, .stFormSubmitButton>button, .stLinkButton>a {
  border-radius: 14px !important; font-weight:700 !important; border:0 !important;
  background: linear-gradient(135deg,#0f766e,#14b8a6) !important; color:#fff !important;
  padding:.55rem 1.1rem !important; box-shadow: 0 10px 20px -12px #0f766e; }
.stButton>button:hover, .stFormSubmitButton>button:hover { filter:brightness(1.06); transform: translateY(-1px); }
.stTabs [data-baseweb="tab-list"] { gap:6px; background:#fff; padding:6px; border-radius:16px;
                                     border:1px solid #e6ecf0; }
.stTabs [data-baseweb="tab"] { border-radius:12px; padding:8px 12px; font-weight:700; height:auto; }
.stTabs [aria-selected="true"] { background:#ecfdf5; color:#047857; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
.foot { text-align:center; color:#94a3b8; font-size:.78rem; margin-top:30px; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def e(x):
    return html.escape(str(x or ""))


def hero():
    st.markdown(f"""
    <div class="hero">
      <h1>🎓 {e(SITE_NAME)}</h1>
      <p>{e(TAGLINE)}</p>
      <div class="pills">
        <span class="pill">✅ Verified parents</span>
        <span class="pill">🔒 Parents never pay</span>
        <span class="pill">⚡ Fresh leads daily</span>
      </div>
    </div>""", unsafe_allow_html=True)


def ago(dt):
    if not dt:
        return ""
    s = (datetime.utcnow() - dt).total_seconds()
    if s < 3600:
        return f"{max(1, int(s // 60))} min ago"
    if s < 86400:
        return f"{int(s // 3600)} h ago"
    d = int(s // 86400)
    return "yesterday" if d == 1 else f"{d} days ago"


def mask_phone(p):
    p = "".join(ch for ch in str(p or "") if ch.isdigit())
    return (p[:2] + "••••••" + p[-2:]) if len(p) >= 6 else "••••••••••"


def mask_name(n):
    parts = str(n or "").split()
    return " ".join((w[0] + "•••••") for w in parts) if parts else "••••• •••••"


def mask_email(m):
    if "@" not in str(m or ""):
        return "••••@••••.com"
    u, d = m.split("@", 1)
    return u[:1] + "•••@" + d[:1] + "•••"


def preview(text):
    t = str(text or "").strip()
    n = DETAILS_PREVIEW_CHARS
    return t if len(t) <= n else t[:n].rstrip() + "… 🔒"


def lead_card(l, unlocked=False):
    chips = [
        f'<span class="chip">{e(l["class_level"])}</span>',
        f'<span class="chip blue">{e(l["mode"])}</span>',
    ]
    if l.get("budget"):
        chips.append(f'<span class="chip amber">₹ {e(l["budget"])}</span>')
    if l.get("gender_pref") and l["gender_pref"] != "No preference":
        chips.append(f'<span class="chip rose">{e(l["gender_pref"])}</span>')
    where = ", ".join(x for x in [l.get("area"), l.get("city")] if x)
    details = l.get("details") if unlocked else preview(l.get("details"))

    if unlocked:
        box = f"""<div class="unlocked">
          <div>👤 <b>{e(l['parent_name'])}</b></div>
          <div>📞 <b>{e(l['phone'])}</b></div>
          {f"<div>✉️ {e(l['email'])}</div>" if l.get('email') else ""}
          {f"<div>📍 {e(l['address'])}</div>" if l.get('address') else ""}
        </div>"""
    else:
        box = f"""<div class="locked">🔒 <b>Contact locked</b> · unlock to see full details<br>
          <span class="blur">{e(mask_name(l['parent_name']))} · {e(mask_phone(l['phone']))}</span></div>"""

    return f"""
    <div class="card">
      <div class="top">
        <h3>{e(l['subject'])}<br><span style="font-weight:500;color:#64748b;font-size:.85rem">📍 {e(where)} {e(l.get('pincode'))}</span></h3>
        <span class="when">{e(ago(l.get('created_at')))}</span>
      </div>
      <div class="chips">{''.join(chips)}</div>
      <div class="desc">{e(details)}</div>
      {box}
    </div>"""


def plan_card():
    st.markdown(f"""
    <div class="plan">
      <div style="opacity:.85;font-weight:600">Tutor Plan</div>
      <div class="price">₹{PLAN_PRICE_INR:,} <small>/ {PLAN_VALIDITY_DAYS} days</small></div>
      <ul>
        <li>Unlock <b>{PLAN_LEADS} student leads</b> of your choice</li>
        <li>Full parent name, phone &amp; address</li>
        <li>Valid for <b>{PLAN_VALIDITY_DAYS} days</b> from purchase</li>
        <li>No commission on your tuition fees</li>
      </ul>
    </div>""", unsafe_allow_html=True)


def stat(n, label):
    return f'<div class="stat"><div class="n">{e(n)}</div><div class="l">{e(label)}</div></div>'


def footer():
    st.markdown(f'<div class="foot">© {datetime.now().year} {e(SITE_NAME)} · '
                f'{e(SUPPORT_PHONE)} · {e(SUPPORT_EMAIL)}</div>',
                unsafe_allow_html=True)


# ==========================================================================
# APP PAGES
# ==========================================================================
st.set_page_config(page_title=f"{SITE_NAME} · Home tuition leads", page_icon="🎓",
                   layout="centered", initial_sidebar_state="collapsed")
inject_css()


# ---------------------------------------------------------------- session ---
def current_user():
    if st.session_state.get("user"):
        return st.session_state["user"]
    tok = st.query_params.get("s")
    if tok:
        u = get_session_user(tok)
        if u:
            st.session_state["user"] = u
            return u
    return None


def login(user):
    st.session_state["user"] = user
    st.query_params["s"] = create_session(user["id"])


def logout():
    tok = st.query_params.get("s")
    if tok:
        delete_session(tok)
    st.session_state.pop("user", None)
    st.query_params.clear()


def valid_phone(p):
    return bool(re.fullmatch(r"[6-9]\d{9}", re.sub(r"[\s+-]|^91", "", p.strip())))


# ------------------------------------------------------------ browse leads ---
def tab_find(user):
    c1, c2 = st.columns([1, 1])
    level = c1.selectbox("Class / category", ["All"] + CLASS_LEVELS, label_visibility="collapsed")
    search = c2.text_input("Search area, pincode, subject", placeholder="🔎 Area, pincode, subject",
                           label_visibility="collapsed")
    items = list_leads(level, search)
    unlocked = unlocked_ids(user["id"]) if user else set()
    credits = credits_summary(user["id"])["credits"] if user else 0

    if not items:
        st.info("No tuition requirements match yet. Try clearing the filters.")
    for l in items:
        is_un = l["id"] in unlocked
        st.markdown(lead_card(l, is_un), unsafe_allow_html=True)
        if is_un:
            continue
        if not user:
            st.caption("🔑 Sign in from the **Tutor account** tab to unlock contact details.")
        elif credits <= 0:
            st.caption("💳 Buy a plan in the **Tutor account** tab to unlock contacts.")
        elif st.button(f"🔓 Unlock contact · uses 1 of {credits} credits", key=f"u{l['id']}"):
            if unlock_lead(user["id"], l["id"]):
                st.rerun()
            else:
                st.error("Could not unlock. Your plan may have expired.")


# ------------------------------------------------------------- parent form ---
def tab_post():
    st.markdown("### Need a tutor? It's free for parents")
    st.caption("Tell us what you need. Verified tutors in your area will contact you.")
    with st.form("post", clear_on_submit=True):
        name = st.text_input("Your name *")
        phone = st.text_input("Mobile number *", max_chars=15, placeholder="10-digit number")
        email = st.text_input("Email (optional)")
        c1, c2 = st.columns(2)
        level = c1.selectbox("Class / category *", CLASS_LEVELS)
        subject = c2.text_input("Subject(s) *", placeholder="Maths, Science")
        c3, c4 = st.columns(2)
        mode = c3.selectbox("Tuition type", MODES)
        gender = c4.selectbox("Tutor preference", GENDER_PREF)
        c5, c6 = st.columns(2)
        city = c5.text_input("City *", value="Hyderabad")
        area = c6.text_input("Area / locality *")
        c7, c8 = st.columns(2)
        pin = c7.text_input("Pincode", max_chars=6)
        budget = c8.text_input("Budget per month", placeholder="e.g. 3000-4000")
        address = st.text_area("Full address (shared only with tutors who unlock)", height=70)
        details = st.text_area("Requirement details", height=90,
                               placeholder="Timings, board (CBSE/State), goals…")
        ok = st.checkbox("I agree to share these details with verified tutors *")
        if st.form_submit_button("Post requirement", use_container_width=True):
            if not (name.strip() and subject.strip() and city.strip() and area.strip()):
                st.error("Please fill all required fields.")
            elif not valid_phone(phone):
                st.error("Enter a valid 10-digit Indian mobile number.")
            elif not ok:
                st.error("Please accept consent to continue.")
            else:
                add_lead(parent_name=name.strip(), phone=re.sub(r"\D", "", phone)[-10:],
                            email=email.strip(), address=address.strip(), class_level=level,
                            subject=subject.strip(), mode=mode, city=city.strip(), area=area.strip(),
                            pincode=pin.strip(), budget=budget.strip(), gender_pref=gender,
                            details=details.strip())
                st.success("Posted! Tutors will contact you soon.")
                st.balloons()


# ------------------------------------------------------------ tutor account --
def auth_forms():
    t1, t2 = st.tabs(["Sign in", "Create account"])
    with t1:
        with st.form("login"):
            em = st.text_input("Email")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", use_container_width=True):
                u = verify_user(em, pw)
                if u:
                    login(u)
                    st.rerun()
                else:
                    st.error("Wrong email or password.")
    with t2:
        with st.form("register"):
            nm = st.text_input("Full name")
            em = st.text_input("Email ")
            ph = st.text_input("Mobile number", max_chars=15)
            pw = st.text_input("Password (min 6 chars)", type="password")
            if st.form_submit_button("Create account", use_container_width=True):
                if not (nm.strip() and "@" in em and len(pw) >= 6 and valid_phone(ph)):
                    st.error("Fill all fields correctly (valid email, mobile and 6+ char password).")
                else:
                    uid = create_user(nm, em, ph, pw)
                    if uid is None:
                        st.error("This email is already registered. Please sign in.")
                    else:
                        login({"id": uid, "name": nm.strip(), "email": em.strip().lower(),
                               "phone": ph.strip()})
                        st.rerun()


def tab_account(user):
    if not user:
        st.markdown("### Tutor sign in")
        auth_forms()
        return

    # auto-verify any pending online payment
    for p in pending_payments(user["id"]):
        if p["method"] == "online":
            refresh_online(p)

    summ = credits_summary(user["id"])
    st.markdown(f"### Hi {user['name'].split()[0]} 👋")
    a, b, c = st.columns(3)
    a.markdown(stat(summ["credits"], "Credits left"), unsafe_allow_html=True)
    b.markdown(stat(summ["expires"].strftime("%d %b %Y") if summ["expires"] else "—",
                       "Valid till"), unsafe_allow_html=True)
    c.markdown(stat(len(unlocked_ids(user["id"])), "Unlocked"), unsafe_allow_html=True)

    st.write("")
    if summ["credits"] <= 0:
        plan_card()
        st.write("")
        pend = pending_payments(user["id"])
        if pend:
            p = pend[0]
            st.warning(f"Payment #{p['id']} ({p['method']}) is pending.")
            if p["method"] == "online" and p.get("pay_url"):
                st.link_button("Open payment page", p["pay_url"], use_container_width=True)
                if st.button("I've paid, verify now", use_container_width=True):
                    if refresh_online(p):
                        st.success("Payment confirmed!")
                    st.rerun()
            else:
                st.caption("Waiting for admin to confirm your cash payment.")
        else:
            methods = available_methods()
            labels = {"cash": "💵 Cash", "online": "💳 Card / Net banking / UPI (GPay, PhonePe)"}
            m = st.radio("Pay with", methods, format_func=labels.get)
            if st.button(f"Pay ₹{PLAN_PRICE_INR:,}", use_container_width=True):
                try:
                    _, msg = start_checkout(user, m)
                    st.info(msg)
                    st.rerun() if m == "online" or CASH_AUTO_APPROVE else None
                except Exception as ex:
                    st.error(f"Payment could not be started: {ex}")
    else:
        st.success("Your plan is active. Browse leads in the **Find tuitions** tab and unlock.")

    mine = my_unlocked_leads(user["id"])
    if mine:
        st.markdown("#### Your unlocked leads")
        for l in mine:
            st.markdown(lead_card(l, True), unsafe_allow_html=True)
    st.write("")
    if st.button("Sign out"):
        logout()
        st.rerun()


# ------------------------------------------------------------------- admin ---
def tab_admin():
    if not ADMIN_PASSWORD:
        st.info("Admin panel is disabled. Set ADMIN_PASSWORD in secrets to enable it.")
        return
    if not st.session_state.get("is_admin"):
        pw = st.text_input("Admin password", type="password")
        if st.button("Enter") and pw == ADMIN_PASSWORD:
            st.session_state["is_admin"] = True
            st.rerun()
        return
    backend = backend_name()
    if backend == "sqlite":
        st.warning("Using SQLite. On Streamlit Cloud this file is wiped on restart. "
                   "Use a hosted MySQL/Postgres DATABASE_URL for real data.")
    else:
        st.caption(f"Database: {backend}")
    s = stats()
    cols = st.columns(4)
    for col, (k, lab) in zip(cols, [("tutors", "Tutors"), ("leads", "Leads"),
                                    ("unlocks", "Unlocks"), ("revenue", "Revenue ₹")]):
        col.markdown(stat(s[k], lab), unsafe_allow_html=True)

    st.markdown("#### Pending cash payments")
    pend = all_pending_cash()
    if not pend:
        st.caption("Nothing pending.")
    for p in pend:
        st.write(f"**#{p['id']}** · {p['name']} · {p['phone']} · ₹{p['amount']}")
        if st.button(f"Confirm cash received #{p['id']}", key=f"ap{p['id']}"):
            mark_paid(p["id"])
            st.rerun()

    st.markdown("#### Manage leads")
    for l in list_leads(include_closed=True, limit=15):
        c1, c2 = st.columns([3, 1])
        c1.write(f"#{l['id']} {l['subject']} · {l['area']} · {l['status']}")
        new = "closed" if l["status"] == "open" else "open"
        if c2.button(new.title(), key=f"st{l['id']}"):
            set_lead_status(l["id"], new)
            st.rerun()

    if st.button("Add 5 demo leads (testing)"):
        demo = [("Meera R", "9876501234", "Class 6-8", "All subjects", "AC Guards", "500004", "3000", "Female tutor"),
                ("Rahul K", "9812345670", "Class 9-10", "Maths, Physics", "Banjara Hills", "500034", "4500", "No preference"),
                ("Sana F", "9900112233", "Class 1-5", "English, EVS", "Tolichowki", "500008", "2500", "Female tutor"),
                ("Anil V", "9123456780", "Class 11-12", "Chemistry", "Madhapur", "500081", "5000", "No preference"),
                ("Divya S", "9988776655", "Competitive exams", "Quant & Reasoning", "Kukatpally", "500072", "6000", "Male tutor")]
        for n, ph, lv, sub, ar, pin, bud, g in demo:
            add_lead(parent_name=n, phone=ph, email="", address=f"{ar}, Hyderabad", class_level=lv,
                        subject=sub, mode="Home tuition", city="Hyderabad", area=ar, pincode=pin,
                        budget=bud, gender_pref=g, details="Looking for an experienced tutor, "
                        "evening slots preferred, 5 days a week.")
        st.rerun()
    if st.button("Log out of admin"):
        st.session_state["is_admin"] = False
        st.rerun()


# -------------------------------------------------------------------- main ---
user = current_user()
hero()
t_find, t_post, t_acc, t_admin = st.tabs(["🔎 Find tuitions", "📝 Need a tutor", "👤 Tutor account", "⚙️ Admin"])
with t_find:
    tab_find(user)
with t_post:
    tab_post()
with t_acc:
    tab_account(user)
with t_admin:
    tab_admin()
footer()
