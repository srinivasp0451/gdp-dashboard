"""
DocMind — Production-Grade RAG Chatbot
======================================
Features:
  • Hybrid retrieval  : FAISS (dense) + BM25 (lexical) + Cross-encoder reranking
  • Multi-query       : Generates N query variants for richer recall
  • HyDE              : Hypothetical Document Embedding (toggle via ENABLE_HYDE)
  • RAGAS eval        : Faithfulness + Answer Relevancy → console + sidebar + DB
  • SQLite history    : Full persistence — sessions, messages, feedback, metrics
  • Like / Dislike    : Per-message feedback stored to DB
  • st.status loader  : Step-by-step live progress while bot responds
  • Dark theme        : White readable fonts, indigo accents

Install:
    pip install streamlit langchain langchain-community langchain-openai
               sentence-transformers rank-bm25 faiss-cpu tiktoken ragas datasets
               python-dotenv

Run:
    streamlit run rag_chatbot.py
"""

import os, time, re, uuid, json, sqlite3
import streamlit as st
import tiktoken
from datetime import datetime
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH         = "data"
FAISS_PATH        = "faiss_index"
SQLITE_PATH       = "chat_history.db"
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY",  "abcd")
OPENAI_API_BASE   = os.getenv("OPENAI_API_BASE", "https://api.groq.com/openai/v1")
MODEL_NAME        = os.getenv("MODEL_NAME",      "meta-llama/llama-4-scout-17b-16e-instruct")
MAX_HISTORY_TURNS = 6
TOP_K_RETRIEVE    = 5
ENABLE_MULTI_QUERY = True
ENABLE_HYDE        = False   # extra LLM call — enable for better semantic match
ENABLE_RAGAS       = True    # set False to skip evaluation (faster responses)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind · RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS  — dark theme, white fonts, components
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base reset ─────────────────────────────────── */
html, body, [class*="css"], .stApp {
    font-family: 'Sora', sans-serif !important;
    background: #070c18 !important;
    color: #dde4f0 !important;
}

/* ── Force readable text throughout ────────────── */
p, li, h1, h2, h3, h4, h5, span, div, label, a {
    color: #dde4f0;
}

/* ── Chat message area ──────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 4px 0 !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em,
[data-testid="stChatMessage"] td,
[data-testid="stChatMessage"] th {
    color: #dde4f0 !important;
}
[data-testid="stChatMessage"] code {
    background: #111c30 !important;
    color: #93c5fd !important;
    border-radius: 4px;
    padding: 1px 5px;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}
[data-testid="stChatMessage"] pre {
    background: #0c1728 !important;
    border: 1px solid #1e3050;
    border-radius: 8px;
    padding: 12px !important;
}
[data-testid="stChatMessage"] pre code {
    background: transparent !important;
    color: #bae6fd !important;
    font-size: 12px !important;
}

/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #090e1c !important;
    border-right: 1px solid #131f38 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #6a8ab0 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #0e1828 !important;
    color: #6a8ab0 !important;
    border: 1px solid #1a2e48 !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 11px !important;
    width: 100% !important;
    transition: all .2s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #152240 !important;
    color: #b8d0f8 !important;
    border-color: #3b5bdb !important;
}

/* ── Header ─────────────────────────────────────── */
.dm-header {
    display:flex; align-items:center; gap:14px;
    padding: 14px 0 12px 0;
    border-bottom: 1px solid #131f38;
    margin-bottom: 18px;
}
.dm-logo {
    width:46px; height:46px;
    background: linear-gradient(135deg,#3b5bdb 0%,#7048e8 100%);
    border-radius:14px;
    display:flex; align-items:center; justify-content:center;
    font-size:24px; flex-shrink:0;
    box-shadow: 0 4px 22px rgba(59,91,219,.45);
}
.dm-title {
    font-size:22px; font-weight:700;
    background: linear-gradient(90deg,#c5d5ff,#a78bfa);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.dm-sub { font-size:11px; color:#2d4060; margin-top:2px; }

/* ── Metric chips ────────────────────────────────── */
.metric-row { display:flex; gap:7px; flex-wrap:wrap; margin:7px 0 2px 0; }
.mc {
    background:#0c1422; border:1px solid #172840;
    border-radius:6px; padding:2px 9px;
    font-family:'JetBrains Mono',monospace; font-size:10px; color:#3d5270;
}
.mc span { color:#60a5fa; font-weight:600; }
.mc.good span { color:#34d399; }
.mc.warn span { color:#fbbf24; }
.mc.bad  span { color:#f87171; }

/* ── Info block (sidebar) ────────────────────────── */
.ib {
    background:#0c1422; border:1px solid #172840;
    border-radius:8px; padding:9px 12px; margin:5px 0; font-size:11px;
}
.ib .ibl {
    font-size:9px; font-weight:600; letter-spacing:.08em;
    text-transform:uppercase; color:#3b5bdb !important; margin-bottom:3px;
}
.ib .ibv {
    color:#93c5fd !important;
    font-family:'JetBrains Mono',monospace; font-size:11px;
}

/* ── Source card ─────────────────────────────────── */
.src-card {
    background:#0a1020; border:1px solid #162436;
    border-radius:8px; padding:8px 11px; margin:4px 0;
    border-left:3px solid #3b5bdb;
}
.src-card .sf {
    font-family:'JetBrains Mono',monospace;
    color:#93c5fd !important; font-size:10px; margin-bottom:3px;
}
.src-card .ss { color:#3d5270 !important; font-size:11px; line-height:1.5; }

/* ── Suggestion chips ─────────────────────────────── */
.sugg-label {
    font-size:10px; font-weight:600; color:#2d4060;
    letter-spacing:.08em; text-transform:uppercase; margin:14px 0 8px 0;
}
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: #0c1828 !important;
    color: #93c5fd !important;
    border: 1px solid #1a3a5f !important;
    border-radius: 20px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 11px !important;
    padding: 5px 13px !important;
    transition: all .2s !important;
    height: auto !important;
    line-height: 1.45 !important;
    white-space: normal !important;
    text-align: left !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: #132648 !important;
    color: #dbeafe !important;
    border-color: #3b82f6 !important;
    box-shadow: 0 4px 14px rgba(59,130,246,.2) !important;
    transform: translateY(-1px);
}

/* ── Like / dislike buttons ──────────────────────── */
[data-testid^="stButton-like_"] > button,
[data-testid^="stButton-dislike_"] > button {
    background: #0c1422 !important;
    border: 1px solid #172840 !important;
    border-radius: 6px !important;
    font-size: 14px !important;
    padding: 2px 8px !important;
    transition: all .18s !important;
    color: #4a6080 !important;
}
[data-testid^="stButton-like_"] > button:hover    { background:#0d2210 !important; border-color:#34d399 !important; }
[data-testid^="stButton-dislike_"] > button:hover { background:#200d0d !important; border-color:#f87171 !important; }

/* ── Empty state ─────────────────────────────────── */
.empty-state {
    text-align:center; padding:70px 20px; color:#1a2540;
}
.empty-state .ei { font-size:52px; margin-bottom:16px; opacity:.5; }
.empty-state h3  { color:#243652; font-size:17px; font-weight:600; }
.empty-state p   { color:#1a2540; font-size:13px; }

/* ── Chat input ──────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    background: #0c1422 !important;
    border: 1px solid #172840 !important;
    color: #dde4f0 !important;
    border-radius: 12px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 13px !important;
}
[data-testid="stChatInput"] textarea::placeholder { color:#2d4060 !important; }
[data-testid="stChatInput"] textarea:focus {
    border-color: #3b5bdb !important;
    box-shadow: 0 0 0 3px rgba(59,91,219,.12) !important;
}

/* ── Status widget (loader) ──────────────────────── */
[data-testid="stStatusWidget"],
details[data-testid="stStatus"] {
    background: #0c1828 !important;
    border: 1px solid #1a2e48 !important;
    border-radius: 10px !important;
    color: #6a8ab0 !important;
}
details[data-testid="stStatus"] * { color:#6a8ab0 !important; }

/* ── Expander ─────────────────────────────────────── */
details summary {
    background: #0c1422 !important;
    border: 1px solid #172840 !important;
    border-radius: 8px !important;
    color: #4a6080 !important;
    font-size: 11px !important;
}
details[open] summary { border-radius:8px 8px 0 0 !important; }
details > div {
    background: #080e1c !important;
    border: 1px solid #172840 !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    color: #6a8ab0 !important;
}
details > div * { color: #6a8ab0 !important; }

/* ── Divider ─────────────────────────────────────── */
hr { border-color: #172840 !important; margin: 12px 0 !important; }

/* ── Scrollbar ───────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #070c18; }
::-webkit-scrollbar-thumb { background: #172840; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #243654; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SQLITE  — persistence layer
# ─────────────────────────────────────────────────────────────────────────────
def _conn():
    return sqlite3.connect(SQLITE_PATH, check_same_thread=False)

def init_sqlite():
    c = _conn()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id            TEXT PRIMARY KEY,
            title         TEXT    DEFAULT 'New Conversation',
            created_at    TEXT,
            updated_at    TEXT,
            total_queries INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS messages (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id          TEXT NOT NULL,
            role                TEXT NOT NULL,
            content             TEXT NOT NULL,
            timestamp           TEXT NOT NULL,
            latency             REAL    DEFAULT 0,
            tokens              INTEGER DEFAULT 0,
            sources             TEXT    DEFAULT '[]',
            feedback            INTEGER DEFAULT 0,
            ragas_faithfulness  REAL    DEFAULT NULL,
            ragas_relevancy     REAL    DEFAULT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
    """)
    c.commit(); c.close()

def db_new_session(sid: str, title: str = "New Conversation"):
    c = _conn(); now = datetime.now().isoformat()
    c.execute("INSERT OR IGNORE INTO sessions (id,title,created_at,updated_at) VALUES(?,?,?,?)",
              (sid, title, now, now))
    c.commit(); c.close()

def db_save_msg(sid, role, content, latency=0, tokens=0, sources=None) -> int:
    srcs_json = json.dumps([
        {"source": d.metadata.get("source","?"),
         "page":   d.metadata.get("page","?"),
         "snippet":d.page_content[:200]}
        for d in (sources or [])
    ])
    c = _conn(); cur = c.cursor()
    cur.execute(
        "INSERT INTO messages(session_id,role,content,timestamp,latency,tokens,sources)"
        " VALUES(?,?,?,?,?,?,?)",
        (sid, role, content, datetime.now().isoformat(), latency, tokens, srcs_json)
    )
    mid = cur.lastrowid
    if role == "assistant":
        c.execute("UPDATE sessions SET updated_at=?,total_queries=total_queries+1 WHERE id=?",
                  (datetime.now().isoformat(), sid))
    c.commit(); c.close(); return mid

def db_set_ragas(mid: int, faith: float, relev: float):
    c = _conn()
    c.execute("UPDATE messages SET ragas_faithfulness=?,ragas_relevancy=? WHERE id=?",
              (faith, relev, mid))
    c.commit(); c.close()

def db_set_feedback(mid: int, fb: int):
    c = _conn()
    c.execute("UPDATE messages SET feedback=? WHERE id=?", (fb, mid))
    c.commit(); c.close()

def db_set_title(sid: str, title: str):
    c = _conn()
    c.execute("UPDATE sessions SET title=? WHERE id=?", (title[:60], sid))
    c.commit(); c.close()

def db_list_sessions(limit: int = 15):
    c = _conn()
    rows = c.execute(
        "SELECT id,title,created_at,total_queries FROM sessions"
        " ORDER BY updated_at DESC LIMIT ?", (limit,)
    ).fetchall()
    c.close(); return rows

def db_load_session(sid: str):
    c = _conn()
    rows = c.execute(
        "SELECT role,content,latency,tokens,feedback,"
        "ragas_faithfulness,ragas_relevancy,id"
        " FROM messages WHERE session_id=? ORDER BY id", (sid,)
    ).fetchall()
    c.close(); return rows


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADERS  (cached once per process)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource(show_spinner=False)
def _embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )

reranker        = _reranker()
embedding_model = _embeddings()


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_session():
    init_sqlite()
    if "session_id" not in st.session_state:
        sid = str(uuid.uuid4())
        st.session_state.session_id = sid
        db_new_session(sid)
    for k, v in {
        "chat_history"  : [],
        "pending_query" : None,
        "suggestions"   : [],
        "last_sources"  : [],
        "last_latency"  : 0.0,
        "last_tokens"   : 0,
        "feedback"      : {},
        "ragas_cache"   : {},
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
#  LLM FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def llm(temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        model_name=MODEL_NAME,
        temperature=temperature,
    )

def count_tokens(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


# ─────────────────────────────────────────────────────────────────────────────
#  PDF loading & splitting
# ─────────────────────────────────────────────────────────────────────────────
def load_pdfs():
    docs = []
    if not os.path.isdir(DATA_PATH):
        return docs
    for fname in os.listdir(DATA_PATH):
        if fname.lower().endswith(".pdf"):
            pages = PyPDFLoader(os.path.join(DATA_PATH, fname)).load()
            for p in pages:
                p.metadata["source"] = fname
            docs.extend(pages)
    return docs

def split_docs(docs):
    # Smaller chunks (450) + mild overlap (80) → higher precision
    return RecursiveCharacterTextSplitter(
        chunk_size=450, chunk_overlap=80
    ).split_documents(docs)


# ─────────────────────────────────────────────────────────────────────────────
#  VECTOR DB
# ─────────────────────────────────────────────────────────────────────────────
def _db_ready() -> bool:
    return os.path.exists(os.path.join(FAISS_PATH, "index.faiss"))

def build_faiss():
    db = FAISS.from_documents(split_docs(load_pdfs()), embedding_model)
    db.save_local(FAISS_PATH)
    return db

@st.cache_resource(show_spinner=False)
def load_faiss():
    return FAISS.load_local(
        FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
    )


# ─────────────────────────────────────────────────────────────────────────────
#  BM25  lexical retrieval
# ─────────────────────────────────────────────────────────────────────────────
def bm25_score(docs, query: str):
    if not docs:
        return []
    scores = BM25Okapi([d.page_content.lower().split() for d in docs]
                       ).get_scores(query.lower().split())
    return [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]


# ─────────────────────────────────────────────────────────────────────────────
#  CROSS-ENCODER reranker
# ─────────────────────────────────────────────────────────────────────────────
def rerank(query: str, docs):
    if not docs:
        return []
    scores = reranker.predict([(query, d.page_content) for d in docs])
    return [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]


# ─────────────────────────────────────────────────────────────────────────────
#  MULTI-QUERY expansion  — richer recall
# ─────────────────────────────────────────────────────────────────────────────
def query_variants(query: str, n: int = 3) -> list:
    try:
        prompt = (
            f"Rephrase this search query in {n} different ways to maximise document recall.\n"
            "Return ONLY the queries, one per line.\n"
            f"Query: {query}"
        )
        resp = llm(0.4).invoke(prompt).content.strip()
        variants = [l.strip() for l in resp.split("\n") if len(l.strip()) > 5][:n]
        return variants + [query]
    except Exception:
        return [query]


# ─────────────────────────────────────────────────────────────────────────────
#  HyDE  — Hypothetical Document Embedding
# ─────────────────────────────────────────────────────────────────────────────
def hyde_doc(query: str) -> str:
    try:
        return llm(0.5).invoke(
            f"Write a concise 3-sentence paragraph that directly answers: \"{query}\""
        ).content
    except Exception:
        return query


# ─────────────────────────────────────────────────────────────────────────────
#  HYBRID RETRIEVAL  pipeline
# ─────────────────────────────────────────────────────────────────────────────
def retrieve(query: str, db, status_ref=None) -> list:
    def log(msg):
        if status_ref:
            status_ref.write(msg)

    # 1. Multi-query variants
    queries = query_variants(query) if ENABLE_MULTI_QUERY else [query]
    log(f"   ↳ {len(queries)} query variants")

    # 2. Optional HyDE
    if ENABLE_HYDE:
        log("   ↳ HyDE hypothetical doc…")
        queries.append(hyde_doc(query))

    # 3. Dense retrieval (FAISS)
    seen, candidates = set(), []
    for q in queries:
        for d in db.similarity_search(q, k=10):
            if d.page_content not in seen:
                candidates.append(d)
                seen.add(d.page_content)
    log(f"   ↳ {len(candidates)} dense candidates")

    # 4. BM25 re-score on candidate pool
    bm25_ranked = bm25_score(candidates, query)
    for d in bm25_ranked:
        if d.page_content not in seen:
            candidates.append(d)
            seen.add(d.page_content)

    # 5. Cross-encoder reranking
    log("   ↳ Cross-encoder reranking…")
    return rerank(query, candidates)[:TOP_K_RETRIEVE]


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_history(history, n: int = MAX_HISTORY_TURNS) -> str:
    lines = []
    for role, msg, *_ in history[-(n * 2):]:
        lines.append(f"{'User' if role == 'user' else 'Assistant'}: {msg}")
    return "\n".join(lines)

def clean_qs(text: str) -> list:
    out = []
    for l in text.strip().split("\n"):
        l = re.sub(r"^[-•\d.)\s]+", "", l.strip())
        if len(l) > 8 and "question" not in l.lower():
            out.append(l)
    return out[:3]


# ─────────────────────────────────────────────────────────────────────────────
#  ANSWER GENERATION
# ─────────────────────────────────────────────────────────────────────────────
_SYS = """You are DocMind, a precise and expert RAG assistant.
Rules:
- Answer using ONLY the provided context. Cite source filenames when useful.
- Use markdown formatting (bullets, bold, tables, code blocks) where it aids clarity.
- If context is insufficient, say: "The documents don't contain enough info on this."
- Never fabricate facts or hallucinate citations."""

def answer_query(query: str, history, db, status_ref=None):
    status_ref and status_ref.write("📚 Running hybrid retrieval…")
    docs = retrieve(query, db, status_ref)

    context = "\n\n---\n\n".join([
        f"[{d.metadata.get('source','?')} p.{d.metadata.get('page','?')}]\n{d.page_content}"
        for d in docs
    ])
    prompt = (
        f"{_SYS}\n\n"
        f"### Conversation History\n{fmt_history(history)}\n\n"
        f"### Retrieved Context\n{context}\n\n"
        f"### Question\n{query}"
    )
    tokens = count_tokens(prompt)

    status_ref and status_ref.write("🧠 Generating answer…")
    t0   = time.time()
    resp = llm().invoke(prompt)
    return resp.content, docs, time.time() - t0, tokens


# ─────────────────────────────────────────────────────────────────────────────
#  RAGAS  evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_ragas(query: str, answer: str, docs) -> dict:
    """Faithfulness + Answer Relevancy → printed to console + returned."""
    if not ENABLE_RAGAS:
        return {}
    try:
        from ragas import evaluate as _eval
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset

        ds = Dataset.from_dict({
            "question": [query],
            "answer":   [answer],
            "contexts": [[d.page_content for d in docs]],
        })
        res = _eval(
            ds,
            metrics=[faithfulness, answer_relevancy],
            llm=llm(),
            embeddings=embedding_model,
            raise_exceptions=False,
        )
        f = round(float(res["faithfulness"]),     4)
        r = round(float(res["answer_relevancy"]), 4)

        # ── Console output ────────────────────────────────────────────────
        bar = "=" * 58
        print(f"\n{bar}")
        print(f"  🔬  RAGAS  [{datetime.now().strftime('%H:%M:%S')}]")
        print(f"  Q              : {query[:65]}")
        print(f"  Faithfulness   : {f:.4f}  {'✅ OK' if f >= 0.7 else '⚠️  LOW'}")
        print(f"  Ans Relevancy  : {r:.4f}  {'✅ OK' if r >= 0.7 else '⚠️  LOW'}")
        print(bar)

        return {"faithfulness": f, "answer_relevancy": r}

    except ImportError:
        print("[RAGAS] Not installed — pip install ragas datasets")
        return {}
    except Exception as e:
        print(f"[RAGAS] Error: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
#  SUGGESTIONS
# ─────────────────────────────────────────────────────────────────────────────
def suggest(query: str, answer: str) -> list:
    try:
        resp = llm(0.45).invoke(
            "Generate exactly 3 short follow-up questions based on this Q&A.\n"
            "Return ONLY the questions, one per line, no numbers.\n\n"
            f"Q: {query}\nA: {answer}"
        ).content
        return clean_qs(resp)
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  RAGAS score → CSS class helper
# ─────────────────────────────────────────────────────────────────────────────
def _cls(v) -> str:
    if v is None:
        return ""
    return "good" if v >= 0.75 else ("warn" if v >= 0.5 else "bad")


# ═════════════════════════════════════════════════════════════════════════════
#  APP  STARTS  HERE
# ═════════════════════════════════════════════════════════════════════════════
init_session()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:10px 0 14px'>
      <div style='font-size:17px;font-weight:700;
                  background:linear-gradient(90deg,#c5d5ff,#a78bfa);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        🧠 DocMind
      </div>
      <div style='font-size:10px;color:#1e2d45;margin-top:2px'>
        RAG-Powered Document Assistant
      </div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Controls ──────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ New Chat", use_container_width=True):
            sid = str(uuid.uuid4())
            db_new_session(sid)
            for k in ("chat_history","suggestions","last_sources","feedback","ragas_cache"):
                st.session_state[k] = [] if k == "chat_history" else {}
                if k in ("suggestions","last_sources"):
                    st.session_state[k] = []
            st.session_state.session_id   = sid
            st.session_state.pending_query = None
            st.rerun()
    with c2:
        if st.button("🔄 Rebuild DB", use_container_width=True):
            with st.spinner("Indexing PDFs…"):
                build_faiss(); load_faiss.clear()
            st.success("Done!")

    st.divider()

    # ── Session stats ──────────────────────────────────────────────────────
    n_msgs = len([1 for r, *_ in st.session_state.chat_history if r == "assistant"])
    st.markdown(f"""
    <div class='ib'><div class='ibl'>Queries this session</div>
      <div class='ibv'>{n_msgs}</div></div>
    <div class='ib'><div class='ibl'>Last response time</div>
      <div class='ibv'>{st.session_state.last_latency:.2f}s</div></div>
    <div class='ib'><div class='ibl'>Last prompt tokens</div>
      <div class='ibv'>{st.session_state.last_tokens:,}</div></div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Last retrieved sources ─────────────────────────────────────────────
    if st.session_state.last_sources:
        st.markdown("<div class='ibl' style='font-size:9px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#3b5bdb;margin-bottom:6px'>Last Sources</div>",
                    unsafe_allow_html=True)
        for doc in st.session_state.last_sources:
            snippet = doc.page_content[:105].replace("\n", " ")
            st.markdown(f"""
            <div class='src-card'>
              <div class='sf'>📄 {doc.metadata.get('source','?')} · p.{doc.metadata.get('page','?')}</div>
              <div class='ss'>{snippet}…</div>
            </div>""", unsafe_allow_html=True)
        st.divider()

    # ── Past sessions ──────────────────────────────────────────────────────
    sessions = db_list_sessions()
    if sessions:
        st.markdown("<div class='ibl' style='font-size:9px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#3b5bdb;margin-bottom:6px'>Past Sessions</div>",
                    unsafe_allow_html=True)
        for sid, title, created, nq in sessions:
            is_cur   = sid == st.session_state.session_id
            date_str = (created or "")[:10]
            label    = f"{'🟢 ' if is_cur else ''}{title or 'Untitled'} · {nq}q · {date_str}"
            if st.button(label, key=f"sess_{sid}", use_container_width=True):
                if not is_cur:
                    rows = db_load_session(sid)
                    hist, fb_map, rg_map = [], {}, {}
                    for role, content, lat, tok, fb, rf, rr, mid in rows:
                        meta = {"latency": lat, "tokens": tok, "msg_id": mid, "feedback": fb}
                        if rf is not None:
                            meta["ragas"] = {"faithfulness": rf, "answer_relevancy": rr}
                            rg_map[mid]   = meta["ragas"]
                        hist.append([role, content, meta])
                        if fb: fb_map[mid] = fb
                    st.session_state.update({
                        "session_id": sid, "chat_history": hist,
                        "feedback": fb_map, "ragas_cache": rg_map,
                        "suggestions": [], "last_sources": [],
                    })
                    st.rerun()

    st.divider()
    st.markdown(
        f"<div style='font-size:9px;color:#172438;text-align:center;line-height:1.8'>"
        f"Model · {MODEL_NAME.split('/')[-1]}<br>"
        "Embed · BAAI/bge-base-en-v1.5<br>"
        "Rerank · ms-marco-MiniLM-L-6</div>",
        unsafe_allow_html=True,
    )

# ─── Main header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='dm-header'>
  <div class='dm-logo'>🧠</div>
  <div>
    <div class='dm-title'>DocMind RAG Assistant</div>
    <div class='dm-sub'>Hybrid retrieval · Multi-query · Cross-encoder reranking · RAGAS evaluation</div>
  </div>
</div>""", unsafe_allow_html=True)

# ─── Ensure FAISS DB ───────────────────────────────────────────────────────────
if not _db_ready():
    with st.spinner("⚙️ Building FAISS index from PDFs in `data/`…"):
        build_faiss()

try:
    vec_db = load_faiss()
except Exception as e:
    st.error(f"❌ Cannot load index: {e}  →  Click **Rebuild DB** in the sidebar.")
    st.stop()

# ─── Capture input ─────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything about your documents…")
query = None

if st.session_state.pending_query:           # suggestion click → takes priority
    query = st.session_state.pending_query
    st.session_state.pending_query = None
elif user_input:
    query = user_input

# ─── Process ───────────────────────────────────────────────────────────────────
if query:
    with st.status("⚙️ Processing your question…", expanded=True) as status:
        status.write("🔍 Expanding query…")
        resp, docs, latency, tokens = answer_query(
            query, st.session_state.chat_history, vec_db, status_ref=status
        )
        status.write("💡 Generating follow-up suggestions…")
        suggestions = suggest(query, resp)
        status.update(label="✅ Answer ready!", state="complete", expanded=False)

    # persist to SQLite
    db_save_msg(st.session_state.session_id, "user", query)
    asst_id = db_save_msg(
        st.session_state.session_id, "assistant", resp,
        latency=latency, tokens=tokens, sources=docs
    )

    # auto-title on first query
    if not any(r == "user" for r, *_ in st.session_state.chat_history):
        db_set_title(st.session_state.session_id, query[:55])

    # update in-memory state
    st.session_state.chat_history.append(["user",      query, {}])
    st.session_state.chat_history.append(["assistant", resp,  {
        "latency": latency, "tokens": tokens, "msg_id": asst_id
    }])
    st.session_state.last_sources = docs
    st.session_state.last_latency = latency
    st.session_state.last_tokens  = tokens
    st.session_state.suggestions  = suggestions

    # ── RAGAS evaluation (synchronous; output to console + store to DB) ──
    if ENABLE_RAGAS:
        with st.spinner("🔬 Evaluating answer quality (RAGAS)…"):
            metrics = run_ragas(query, resp, docs)
        if metrics:
            db_set_ragas(asst_id, metrics["faithfulness"], metrics["answer_relevancy"])
            st.session_state.ragas_cache[asst_id] = metrics
            for entry in st.session_state.chat_history:
                if entry[0] == "assistant" and entry[2].get("msg_id") == asst_id:
                    entry[2]["ragas"] = metrics

# ─── Render chat ───────────────────────────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown("""
    <div class='empty-state'>
      <div class='ei'>📂</div>
      <h3>No conversation yet</h3>
      <p>Place PDF files in the <code>data/</code> folder, then ask a question below.</p>
    </div>""", unsafe_allow_html=True)
else:
    for idx, (role, msg, meta) in enumerate(st.session_state.chat_history):
        with st.chat_message(role):
            st.markdown(msg)

            if role == "assistant":
                mid     = meta.get("msg_id")
                latency = meta.get("latency", 0)
                tokens  = meta.get("tokens",  0)
                ragas   = meta.get("ragas") or st.session_state.ragas_cache.get(mid, {})
                fb      = st.session_state.feedback.get(mid, meta.get("feedback", 0))

                # metrics row
                ragas_html = ""
                if ragas:
                    ragas_html = (
                        f"<div class='mc {_cls(ragas.get('faithfulness'))}'>🎯 Faith "
                        f"<span>{ragas['faithfulness']:.2f}</span></div>"
                        f"<div class='mc {_cls(ragas.get('answer_relevancy'))}'>📐 Relev "
                        f"<span>{ragas['answer_relevancy']:.2f}</span></div>"
                    )
                st.markdown(
                    f"<div class='metric-row'>"
                    f"<div class='mc'>⚡ <span>{latency:.2f}s</span></div>"
                    f"<div class='mc'>🔤 <span>{tokens:,}</span> tk</div>"
                    f"{ragas_html}</div>",
                    unsafe_allow_html=True,
                )

                # like / dislike  (toggle behaviour)
                if mid:
                    fb_col1, fb_col2, _ = st.columns([1, 1, 12])
                    with fb_col1:
                        label = "👍✓" if fb == 1 else "👍"
                        if st.button(label, key=f"like_{mid}_{idx}", help="Helpful"):
                            new = 0 if fb == 1 else 1
                            st.session_state.feedback[mid] = new
                            db_set_feedback(mid, new)
                            for e in st.session_state.chat_history:
                                if e[0]=="assistant" and e[2].get("msg_id")==mid:
                                    e[2]["feedback"] = new
                            st.rerun()
                    with fb_col2:
                        label = "👎✓" if fb == -1 else "👎"
                        if st.button(label, key=f"dislike_{mid}_{idx}", help="Not helpful"):
                            new = 0 if fb == -1 else -1
                            st.session_state.feedback[mid] = new
                            db_set_feedback(mid, new)
                            for e in st.session_state.chat_history:
                                if e[0]=="assistant" and e[2].get("msg_id")==mid:
                                    e[2]["feedback"] = new
                            st.rerun()

# ─── Suggestion chips  ─────────────────────────────────────────────────────────
# Rendered OUTSIDE `if query:` — this is the critical fix that makes clicks work
if st.session_state.suggestions:
    st.markdown("<div class='sugg-label'>💡 Suggested follow-ups</div>",
                unsafe_allow_html=True)
    cols = st.columns(len(st.session_state.suggestions))
    for i, (col, q) in enumerate(zip(cols, st.session_state.suggestions)):
        with col:
            if st.button(q, key=f"sugg_{i}", use_container_width=True):
                st.session_state.pending_query = q
                st.session_state.suggestions   = []
                st.rerun()
