"""
RAG Doc QA App
===============

Collect → Preprocess → Index → Query/Retrieve → Generate → UI

Tech:
- Streamlit UI
- Custom token-aware chunking (~500 tokens) with line-number metadata
- Embeddings: OpenAI (default) or Sentence-Transformers (local)
- Vector store: FAISS (in-memory with optional persistence)
- LLM: OpenAI via LangChain
- Simple web scraper (requests + BeautifulSoup) and ZIP doc import

Setup
-----
1) `pip install -U streamlit faiss-cpu tiktoken beautifulsoup4 langchain-core langchain-openai openai`
   Optionally for local embeddings: `pip install -U sentence-transformers`
2) Set `OPENAI_API_KEY` in your environment or Streamlit secrets.

Run
---
`streamlit run app.py`

Notes
-----
- This single-file app keeps dependencies lean and avoids heavy loaders.
- URL crawl is constrained and same-origin by default.
- Persistence: toggle "Persist index" to save/load FAISS and metadata.
"""

from __future__ import annotations
import os
import io
import re
import json
import time
import zipfile
import tempfile
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import streamlit as st
import requests
from bs4 import BeautifulSoup

import numpy as np
import faiss  # faiss-cpu
import tiktoken

# LangChain (LLM + Embeddings)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Chunk:
    text: str
    source_id: str  # path or URL
    start_line: int
    end_line: int
    doc_hash: str   # checksum to tie back to original


# ----------------------------
# Utilities
# ----------------------------

def get_openai_key() -> Optional[str]:
    # Priority: st.secrets → env
    try:
        return st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        return os.getenv("OPENAI_API_KEY")


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style/nav/footer
    for bad in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        bad.decompose()
    text = soup.get_text("\n")
    # collapse
    return "\n".join([ln.strip() for ln in text.splitlines() if ln.strip()])


def fetch_url(url: str, timeout: int = 15) -> Optional[str]:
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200 and resp.text:
            return text_from_html(resp.text)
        return None
    except Exception:
        return None


def same_origin(url: str, root: str) -> bool:
    try:
        from urllib.parse import urlparse
        a, b = urlparse(url), urlparse(root)
        return (a.scheme, a.netloc) == (b.scheme, b.netloc)
    except Exception:
        return False


def crawl_urls(start_url: str, max_pages: int = 15) -> Dict[str, str]:
    """Breadth-first crawl on same origin; returns {url: text}."""
    from collections import deque
    seen, out = set(), {}
    q = deque([start_url])
    while q and len(out) < max_pages:
        url = q.popleft()
        if url in seen:
            continue
        seen.add(url)
        txt = fetch_url(url)
        if not txt:
            continue
        out[url] = txt
        # discover links
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a["href"].strip()
                    from urllib.parse import urljoin
                    nxt = urljoin(url, href)
                    if same_origin(nxt, start_url) and nxt not in seen and len(out) + len(q) < max_pages:
                        q.append(nxt)
        except Exception:
            pass
    return out


def read_zip_texts(uploaded: io.BytesIO) -> Dict[str, str]:
    """Extract .md/.txt/.py/.rst/.json files from a ZIP into memory."""
    result: Dict[str, str] = {}
    with zipfile.ZipFile(uploaded) as z:
        for info in z.infolist():
            name = info.filename
            if info.is_dir():
                continue
            if not re.search(r"\.(md|markdown|txt|py|rst|json)$", name, re.I):
                continue
            try:
                with z.open(info) as f:
                    data = f.read()
                text = data.decode("utf-8", errors="ignore")
                result[name] = text
            except Exception:
                continue
    return result


def doc_checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def approx_token_count(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def line_token_count(lines: List[str], model: str = "gpt-3.5-turbo") -> List[int]:
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return [len(enc.encode(ln)) for ln in lines]


def chunk_by_tokens_with_lines(text: str, source_id: str, target_tokens: int = 500, model: str = "gpt-3.5-turbo") -> List[Chunk]:
    """Greedy line-based packing to ~target_tokens with recorded line spans."""
    lines = text.splitlines()
    tokens_per_line = line_token_count(lines, model=model)
    chunks: List[Chunk] = []
    i = 0
    h = doc_checksum(text)
    while i < len(lines):
        cur_tokens = 0
        start = i
        buff: List[str] = []
        while i < len(lines) and (cur_tokens + tokens_per_line[i] <= target_tokens or not buff):
            buff.append(lines[i])
            cur_tokens += tokens_per_line[i]
            i += 1
        chunk_text = "\n".join(buff).strip()
        if chunk_text:
            chunks.append(Chunk(text=chunk_text, source_id=source_id, start_line=start + 1, end_line=i, doc_hash=h))
    return chunks


# ----------------------------
# Embeddings & Vector store
# ----------------------------
class EmbeddingBackend:
    def __init__(self, provider: str = "openai", model: str = "text-embedding-3-small"):
        self.provider = provider
        self.model = model
        self._st_model: Optional[SentenceTransformer] = None
        if provider == "local":
            if not _HAS_ST:
                raise RuntimeError("sentence-transformers not installed; run `pip install sentence-transformers` or switch to OpenAI embeddings.")
            # lightweight default
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.provider == "openai":
            emb = OpenAIEmbeddings(model=self.model)
            vecs = emb.embed_documents(texts)
            return np.array(vecs, dtype=np.float32)
        else:
            assert self._st_model is not None
            vecs = self._st_model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
            return np.array(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        if self.provider == "openai":
            emb = OpenAIEmbeddings(model=self.model)
            v = emb.embed_query(text)
            return np.array(v, dtype=np.float32)
        else:
            assert self._st_model is not None
            v = self._st_model.encode([text], normalize_embeddings=True)[0]
            return np.array(v, dtype=np.float32)


class FaissIndex:
    def __init__(self, dim: int, normalize: bool = True):
        self.normalize = normalize
        self.index = faiss.IndexFlatIP(dim)  # cosine with normalized vectors
        self.vectors: Optional[np.ndarray] = None
        self.meta: List[Chunk] = []

    def add(self, vecs: np.ndarray, meta: List[Chunk]):
        # Normalize for cosine
        if self.normalize:
            faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.vectors = vecs if self.vectors is None else np.vstack([self.vectors, vecs])
        self.meta.extend(meta)

    def search(self, q: np.ndarray, k: int = 5) -> List[Tuple[float, Chunk, int]]:
        xq = q.reshape(1, -1).astype(np.float32)
        if self.normalize:
            faiss.normalize_L2(xq)
        D, I = self.index.search(xq, k)
        out = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            out.append((float(score), self.meta[idx], int(idx)))
        return out

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        # Save FAISS
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        # Save meta
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump([chunk.__dict__ for chunk in self.meta], f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str, dim: int, normalize: bool = True) -> "FaissIndex":
        idx = FaissIndex(dim, normalize=normalize)
        idx.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
            meta_list = json.load(f)
        idx.meta = [Chunk(**m) for m in meta_list]
        return idx


# ----------------------------
# Generation (LLM via LangChain)
# ----------------------------
GEN_SYSTEM = (
    "You are a precise coding assistant. Answer the user question using the provided context chunks. "
    "Cite sources as [S{i}] where i is the source number you are referencing. If code is requested, provide a minimal, correct snippet. "
    "If the answer is uncertain, say so and suggest how to validate."
)

GEN_HUMAN = (
    "Question:\n{question}\n\n"
    "Context chunks (with ids):\n{context}\n\n"
    "Instructions:\n- Use only the context to answer when possible.\n- Provide references like [S1], [S2] where appropriate.\n- If multiple interpretations exist, enumerate them briefly.\n- Keep the answer concise and well-formatted."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", GEN_SYSTEM),
    ("human", GEN_HUMAN),
])


def run_llm(question: str, contexts: List[Tuple[str, str]], model_name: str = "gpt-4o-mini", temperature: float = 0.1) -> str:
    """
    contexts: list of (sid, text)
    """
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    ctx_text = "\n\n".join([f"[S{i+1}] (id={sid})\n{normalize_whitespace(txt)[:4000]}" for i, (sid, txt) in enumerate(contexts)])
    msg = prompt.invoke({"question": question, "context": ctx_text})
    out = llm.invoke(msg)
    return out.content if hasattr(out, "content") else str(out)


# ----------------------------
# UI Helpers
# ----------------------------
PERSIST_DIR = "/tmp/rag_faiss"


def ensure_state():
    if "docs" not in st.session_state:
        st.session_state.docs = {}  # {source_id: text}
    if "chunks" not in st.session_state:
        st.session_state.chunks = []  # List[Chunk]
    if "faiss" not in st.session_state:
        st.session_state.faiss = None  # FaissIndex
    if "embed_backend" not in st.session_state:
        st.session_state.embed_backend = None
    if "chat" not in st.session_state:
        st.session_state.chat = []  # [(user, assistant)]


def highlight_lines(original: str, start_line: int, end_line: int, window: int = 6) -> Tuple[str, Tuple[int, int]]:
    """Return a small window of text with the target lines marked."""
    lines = original.splitlines()
    i0 = max(0, start_line - 1 - window)
    i1 = min(len(lines), end_line + window)
    block = []
    for i in range(i0, i1):
        prefix = "▶ " if (start_line - 1) <= i < end_line else "  "
        block.append(f"{prefix}{i+1:>4}: {lines[i]}")
    return "\n".join(block), (i0 + 1, i1)


def build_index(backend: EmbeddingBackend, persist: bool = False) -> Tuple[FaissIndex, int]:
    chunks: List[Chunk] = st.session_state.chunks
    if not chunks:
        raise RuntimeError("No chunks to index. Add documents first.")
    texts = [c.text for c in chunks]
    vecs = backend.embed(texts)
    dim = vecs.shape[1]
    idx = FaissIndex(dim)
    idx.add(vecs, chunks)
    if persist:
        idx.save(PERSIST_DIR)
    return idx, dim


def load_index_if_available(dim_guess: int = 384) -> Optional[FaissIndex]:
    try:
        if os.path.exists(os.path.join(PERSIST_DIR, "index.faiss")):
            return FaissIndex.load(PERSIST_DIR, dim=dim_guess)
    except Exception:
        return None
    return None


# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="RAG Doc QA", layout="wide")
ensure_state()

st.title("🔎 RAG Doc QA — Docs → Answers")
st.caption("Collect docs → 500-token chunks → Embeddings → FAISS → Retrieve → Generate with LangChain")

with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("Embedding provider", ["openai", "local"], index=0, help="OpenAI requires API key; Local uses sentence-transformers (if installed).")
    if provider == "openai":
        st.text_input("OpenAI API Key", type="password", value=get_openai_key() or "", help="Leave blank to use environment/secrets.")
        emb_model = st.text_input("OpenAI Embedding model", value="text-embedding-3-small")
    else:
        emb_model = st.text_input("Local embedding model", value="all-MiniLM-L6-v2")
    persist = st.checkbox("Persist index to disk", value=False, help=f"Save to {PERSIST_DIR}")
    max_pages = st.number_input("Max crawl pages", min_value=1, max_value=200, value=15)
    chunk_tokens = st.slider("Target tokens / chunk", min_value=200, max_value=1000, value=500, step=50)
    top_k = st.slider("Top-K chunks", min_value=2, max_value=12, value=5)
    llm_name = st.text_input("LLM model", value="gpt-4o-mini")

st.subheader("1) Collect Docs")
col1, col2 = st.columns(2)

with col1:
    url = st.text_input("Seed URL to crawl (same origin)")
    if st.button("🕷️ Crawl URL") and url:
        with st.spinner("Crawling..."):
            pages = crawl_urls(url, max_pages=int(max_pages))
            added = 0
            for sid, txt in pages.items():
                st.session_state.docs[sid] = txt
                added += 1
        st.success(f"Added {added} pages from crawl.")

with col2:
    up = st.file_uploader("Or upload a ZIP of docs (.md/.txt/.py/.rst/.json)", type=["zip"])
    if up is not None and st.button("📦 Ingest ZIP"):
        with st.spinner("Extracting and reading ZIP..."):
            items = read_zip_texts(up)
            for sid, txt in items.items():
                st.session_state.docs[sid] = txt
        st.success(f"Added {len(items)} files from ZIP.")

if st.session_state.docs:
    with st.expander("Preview collected sources"):
        for i, (sid, txt) in enumerate(list(st.session_state.docs.items())[:5], start=1):
            st.markdown(f"**{i}. {sid}** — {len(txt)} chars, ~{approx_token_count(txt)} tokens")
            st.code(txt[:1000] + ("\n..." if len(txt) > 1000 else ""))

st.subheader("2) Preprocess → 500-token Chunks")
if st.button("✂️ Chunk Docs"):
    with st.spinner("Chunking documents..."):
        chunks: List[Chunk] = []
        for sid, txt in st.session_state.docs.items():
            parts = chunk_by_tokens_with_lines(txt, sid, target_tokens=int(chunk_tokens))
            chunks.extend(parts)
        st.session_state.chunks = chunks
    st.success(f"Created {len(st.session_state.chunks)} chunks from {len(st.session_state.docs)} docs.")

if st.session_state.chunks:
    with st.expander("Preview chunks"):
        for i, ch in enumerate(st.session_state.chunks[:5], start=1):
            st.markdown(f"**Chunk {i}** — {ch.source_id} (lines {ch.start_line}–{ch.end_line})")
            st.code(ch.text[:800] + ("\n..." if len(ch.text) > 800 else ""))

st.subheader("3) Index → FAISS")
if st.button("🧱 Build/Load Index"):
    with st.spinner("Preparing embeddings and FAISS index..."):
        backend = EmbeddingBackend(provider=provider, model=emb_model)
        st.session_state.embed_backend = backend
        # Try loading if persistence enabled
        idx = load_index_if_available()
        if idx is not None and not st.session_state.chunks:
            st.session_state.faiss = idx
        else:
            st.session_state.faiss, _ = build_index(backend, persist=persist)
    st.success("Index ready.")

# Chat / Query Section
st.subheader("4) Ask Questions → Retrieve → Generate")
q = st.text_input("Ask a code or docs question")
ask = st.button("🔎 Answer")

if ask and q:
    if st.session_state.faiss is None or st.session_state.embed_backend is None:
        st.error("Please build the index first.")
    else:
        with st.spinner("Retrieving relevant chunks..."):
            qv = st.session_state.embed_backend.embed_query(q)
            hits = st.session_state.faiss.search(qv, k=int(top_k))
        if not hits:
            st.warning("No results found. Try a different query.")
        else:
            # Prepare contexts and a source map for display
            contexts: List[Tuple[str, str]] = []
            source_map: List[Tuple[str, Chunk]] = []
            for rank, (score, ch, idx) in enumerate(hits, start=1):
                sid = f"{rank}:{os.path.basename(ch.source_id) or ch.source_id}#{ch.start_line}-{ch.end_line}"
                contexts.append((sid, ch.text))
                source_map.append((sid, ch))

            answer = run_llm(q, contexts, model_name=llm_name)
            st.markdown("### 🧠 Answer")
            st.write(answer)

            st.markdown("### 📚 Sources & Highlights")
            for sid, ch in source_map:
                full = st.session_state.docs.get(ch.source_id, ch.text)
                highlighted, (w0, w1) = highlight_lines(full, ch.start_line, ch.end_line)
                with st.expander(f"{sid} — {ch.source_id} (lines {ch.start_line}–{ch.end_line})"):
                    st.code(highlighted)

            # Maintain chat history (lightweight)
            st.session_state.chat.append((q, answer))

if st.session_state.chat:
    st.markdown("---")
    st.markdown("### 💬 Follow-ups")
    for i, (uq, ua) in enumerate(st.session_state.chat[-5:]):
        with st.chat_message("user"):
            st.write(uq)
        with st.chat_message("assistant"):
            st.write(ua)

st.markdown("---")
st.caption("Tip: Enable persistence to cache the FAISS index and metadata between runs. Use ZIP ingest for local Markdown/API docs.")
