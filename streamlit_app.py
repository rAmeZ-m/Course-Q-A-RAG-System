import os, re, csv
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import chromadb
from docx import Document
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "openai/gpt-oss-120b"
DATA_DIR = "data/raw_formats"
CHROMA_DIR = "/tmp/chroma"
COLLECTION = "courses_v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

COURSES = {
    "python": ("Python 3 Tutorial", "python_tutorial.txt"),
    "mdl": ("Multimodal Deep Learning", "mdl_course.docx"),
    "cv": ("Computer Vision (Szeliski)", "cv_course.csv"),
}
HEAD = re.compile(r"CHUNK \| id=(.*?) \| section=(.*?) \| pages=(.*)$")
TOKEN_RE = re.compile(r"[a-z0-9_]+")


# ---------------- Loaders ----------------
def _mk(cid, course, source, section, pages, text):
    return {"id": cid, "course": course, "source": source,
            "section": section or "", "pages": pages or "", "text": text.strip()}


def load_txt(path, course):
    chunks, cur, buf = [], None, []

    def flush():
        if cur:
            chunks.append(_mk(cur[0], course, os.path.basename(path), cur[1], cur[2], "\n".join(buf)))

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("### CHUNK | "):
                flush()
                cur = HEAD.match(line[len("### "):]).groups()
                buf = []
            else:
                buf.append(line)
    flush()
    return chunks


def load_docx(path, course):
    chunks, cur, buf = [], None, []

    def flush():
        if cur:
            chunks.append(_mk(cur[0], course, os.path.basename(path), cur[1], cur[2], "\n".join(buf)))

    for p in Document(path).paragraphs:
        if p.style.name == "Heading 2" and p.text.startswith("CHUNK | "):
            flush()
            cur = HEAD.match(p.text).groups()
            buf = []
        elif cur is not None:
            buf.append(p.text)
    flush()
    return chunks


def load_csv(path, course):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [_mk(r["id"], course, os.path.basename(path), r["section"], r["pages"], r["text"])
                for r in csv.DictReader(f)]


LOADERS = {".txt": load_txt, ".docx": load_docx, ".csv": load_csv}


def load_file(path, course):
    return LOADERS[os.path.splitext(path)[1].lower()](path, course)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


# ---------------- Index (built once, cached) ----------------
@st.cache_resource(show_spinner="Building the index (first run only)...")
def build_index():
    chunks = []
    for _, (course, fname) in COURSES.items():
        chunks.extend(load_file(os.path.join(DATA_DIR, fname), course))
    id2idx = {c["id"]: i for i, c in enumerate(chunks)}
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    col = db.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    if col.count() != len(chunks):
        db.delete_collection(COLLECTION)
        col = db.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        emb = embedder.encode([c["text"] for c in chunks], batch_size=32,
                              normalize_embeddings=True)
        B = 500
        for i in range(0, len(chunks), B):
            b = chunks[i:i + B]
            col.add(
                ids=[c["id"] for c in b],
                embeddings=emb[i:i + B].tolist(),
                documents=[c["text"] for c in b],
                metadatas=[{"course": c["course"], "source": c["source"],
                            "section": c["section"], "pages": str(c["pages"])} for c in b],
            )
    return chunks, id2idx, bm25, embedder, col


chunks, id2idx, bm25, embedder, col = build_index()


# ---------------- Retrieval ----------------
def dense_ids(q, k=30):
    qe = embedder.encode([QUERY_PREFIX + q], normalize_embeddings=True).tolist()
    return col.query(query_embeddings=qe, n_results=k)["ids"][0]


def bm25_ids(q, k=30):
    scores = bm25.get_scores(tokenize(q))
    top = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i]["id"] for i in top]


def rrf_w(rank_lists, weights, k=60):
    s = {}
    for lst, w in zip(rank_lists, weights):
        for rank, cid in enumerate(lst, start=1):
            s[cid] = s.get(cid, 0) + w / (k + rank)
    return sorted(s.items(), key=lambda x: x[1], reverse=True)


def search(q, top_k=8, pool=30, w_dense=1.0, w_bm25=2.0):
    fused = rrf_w([dense_ids(q, pool), bm25_ids(q, pool)], [w_dense, w_bm25])
    return [cid for cid, _ in fused[:top_k]]


# ---------------- LLM ----------------
def get_client():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        try:
            key = st.secrets["GROQ_API_KEY"]
        except Exception:
            key = None
    return Groq(api_key=key)


def llm(messages, temperature=0.2, max_tokens=1024):
    r = get_client().chat.completions.create(
        model=LLM_MODEL, messages=messages,
        temperature=temperature, max_tokens=max_tokens,
    )
    return r.choices[0].message.content.strip()


def translate_to_en(q):
    return llm([
        {"role": "system", "content": "Translate the user's question to English. If it is already English, return it unchanged. Keep technical terms as they are. Output only the translation."},
        {"role": "user", "content": q},
    ], temperature=0, max_tokens=500)


def is_arabic(text):
    ar = len(re.findall(r"[\u0600-\u06FF]", text))
    return ar > 0.3 * max(1, len(re.findall(r"[A-Za-z\u0600-\u06FF]", text)))


SYSTEM = """You are a Q&A assistant over three books: Multimodal Deep Learning, Python 3 Tutorial, and Computer Vision (Szeliski).
Rules:
- Answer ONLY from the provided passages. Never add outside knowledge.
- Some passages may be irrelevant to the question. Ignore them, and use only the ones that actually answer it.
- If NONE of the passages answer the question, say so clearly and do not guess.
- Cite a passage number like [1] only if that exact passage contains the fact you are stating. Never cite a passage you did not use.
- Do NOT write a "Sources" section or list; the system adds it automatically.
- Do not write code examples that are not in the passages.
- Be concise and well organized.
- {lang_rule}"""

LANG_AR = ("اكتب الإجابة بالعامية المصرية بس، ممنوع الفصحى. "
           "مثال على الأسلوب: 'ده بيعمل كذا عشان يوصل لكذا، ويعني إن الـ model بيتعلم ازاي...'. "
           "سيب المصطلحات التقنية بالإنجليزي. "
           "لو الإجابة مش في المقاطع، قول بالنص: مش موجود في المصادر اللي معايا.")
LANG_EN = "Reply in English. If the answer is not in the passages, say exactly: Not found in the provided sources."


def answer(question, top_k=8):
    ar = is_arabic(question)
    q_en = translate_to_en(question) if ar else question
    ids = search(q_en, top_k=top_k)

    ctx, srcs = [], []
    for n, cid in enumerate(ids, 1):
        c = chunks[id2idx[cid]]
        label = f"{c['course']} | {c['section'] or '-'} | p.{c['pages']}"
        ctx.append(f"[{n}] ({label})\n{c['text']}")
        srcs.append(f"[{n}] {label} ({cid})")

    user = "Passages:\n\n" + "\n\n".join(ctx) + f"\n\nQuestion: {question}"
    system = SYSTEM.format(lang_rule=LANG_AR if ar else LANG_EN)
    ans = llm([{"role": "system", "content": system},
               {"role": "user", "content": user}])
    return q_en, ans, srcs


# ---------------- UI ----------------
st.set_page_config(page_title="Course Q&A RAG", page_icon="📚")
st.markdown("""
<style>
.stMarkdown p, .stMarkdown li {
    unicode-bidi: plaintext;
    text-align: start;
}
.stApp {
    background:
        radial-gradient(circle at 12% 8%, rgba(0, 140, 255, 0.28), transparent 38%),
        radial-gradient(circle at 88% 92%, rgba(0, 230, 190, 0.22), transparent 40%),
        #05070D;
}
h1 {
    color: #FFFFFF;
    font-weight: 800;
    letter-spacing: -0.5px;
}
h1::first-line { color: #C6F432; }
.stButton > button {
    background: #C6F432;
    color: #05070D;
    border: none;
    border-radius: 999px;
    font-weight: 700;
    padding: 0.5rem 1.6rem;
}
.stButton > button:hover {
    background: #D8FF55;
    color: #05070D;
    box-shadow: 0 0 18px rgba(198, 244, 50, 0.45);
}
.stTextArea textarea {
    background: rgba(14, 22, 38, 0.85);
    border: 1px solid rgba(0, 200, 255, 0.35);
    border-radius: 14px;
}
.stTextArea textarea:focus {
    border-color: #C6F432;
    box-shadow: 0 0 0 1px #C6F432;
}
</style>
""", unsafe_allow_html=True)
st.title("Course Q&A RAG")
st.caption("Ask about Multimodal Deep Learning, Python 3, or Computer Vision. "
           "Answers come from the course materials only. / اسأل بالعربي أو بالإنجليزي.")

question = st.text_area("Question / السؤال", height=80,
                        placeholder="Ask in Arabic or English...")

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Searching the course materials..."):
            try:
                _, ans, srcs = answer(question)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
        st.markdown(ans)
        with st.expander("Sources / المصادر"):
            for s in srcs:
                st.text(s)