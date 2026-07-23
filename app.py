"""
AI Document Q&A System — RAG Pipeline
--------------------------------------
Upload PDFs, TXT, Markdown, or DOCX files, ask questions, get answers grounded
in the source text with page/section-level citations.

Stack: LangChain + ChromaDB (persisted) + Google Gemini (embeddings + LLM) + Streamlit

Run:
    streamlit run app.py
"""

import os
import json
import shutil
import tempfile
import traceback
import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
PERSIST_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sessions")
os.makedirs(PERSIST_ROOT, exist_ok=True)

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"  # if this 404s later, Google renamed it again — check aistudio.google.com/models

st.set_page_config(page_title="AI Document Q&A (RAG)", page_icon="📄", layout="wide")
st.title("📄 AI Document Q&A — RAG Pipeline")
st.caption(
    "Upload documents, ask questions in plain English, and get answers grounded "
    "in the source text with citations."
)

# --------------------------------------------------------------------------
# Session state defaults
# --------------------------------------------------------------------------
defaults = {
    "vectorstore": None,
    "chat_history": [],      # list of (question, answer, sources)
    "num_chunks": 0,
    "reranker": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# --------------------------------------------------------------------------
# Sidebar: API key + settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        help="Get a free key at https://aistudio.google.com/app/apikey. "
             "This field is masked and never written to disk.",
    )
    st.markdown("---")

    st.subheader("Chunking")
    chunking_mode = st.radio(
        "Strategy",
        ["Character-based", "Smart (headers/paragraphs)"],
        help="Character-based splits every N characters regardless of structure. "
             "Smart mode splits on markdown headers / paragraph breaks first, which "
             "keeps tables and code blocks more intact.",
    )
    chunk_size = st.slider("Chunk size (characters)", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)

    st.markdown("---")
    st.subheader("Retrieval")
    top_k = st.slider("Chunks to retrieve per question (k)", 1, 10, 3)
    use_reranker = st.checkbox(
        "Enable reranking (more accurate, slower first run)",
        value=False,
        help="Retrieves more candidates (2x k), then reorders them with a small "
             "cross-encoder model for better relevance. Downloads a ~90MB model "
             "the first time it's used.",
    )
    use_history_context = st.checkbox(
        "Use conversation history for follow-ups",
        value=True,
        help='Lets you ask "what about that section?" — includes your last few '
             "turns as context for the current question.",
    )

    st.markdown("---")
    st.subheader("Session persistence")
    session_name = st.text_input("Session name", value="default", help="Save/load a knowledge base by name so you don't have to re-index after refreshing.")
    col_save, col_load = st.columns(2)
    save_clicked = col_save.button("💾 Save", use_container_width=True)
    load_clicked = col_load.button("📂 Load", use_container_width=True)

    existing_sessions = sorted(
        [d for d in os.listdir(PERSIST_ROOT) if os.path.isdir(os.path.join(PERSIST_ROOT, d))]
    )
    if existing_sessions:
        st.caption(f"Saved sessions: {', '.join(existing_sessions)}")

    st.markdown("---")
    st.markdown(
        "**How it works**\n"
        "1. Documents are split into chunks\n"
        "2. Each chunk is embedded into a vector\n"
        "3. Your question is embedded and matched to the closest chunks "
        "(optionally reranked)\n"
        "4. Gemini answers using only those chunks, and cites the source"
    )

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to get started. It's free at aistudio.google.com.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def session_dir(name: str) -> str:
    safe = "".join(c for c in name if c.isalnum() or c in ("-", "_")) or "default"
    return os.path.join(PERSIST_ROOT, safe)


def load_documents(uploaded_files):
    """Load each uploaded file with the right loader. Returns (docs, warnings)."""
    all_docs, warnings = [], []

    for uf in uploaded_files:
        suffix = os.path.splitext(uf.name)[1].lower()
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name

            if suffix == ".pdf":
                docs = PyPDFLoader(tmp_path).load()
            elif suffix in (".txt", ".md"):
                docs = TextLoader(tmp_path, encoding="utf-8").load()
            elif suffix == ".docx":
                docs = Docx2txtLoader(tmp_path).load()
            else:
                warnings.append(f"⚠️ Skipped **{uf.name}** — unsupported file type ({suffix}).")
                os.unlink(tmp_path)
                continue

            # flag image-only / empty-extraction PDFs early
            total_chars = sum(len(d.page_content.strip()) for d in docs)
            if total_chars < 20:
                warnings.append(
                    f"⚠️ **{uf.name}** produced almost no extractable text. "
                    "It may be a scanned/image-only file — OCR would be needed first."
                )

            for d in docs:
                d.metadata["source_file"] = uf.name
            all_docs.extend(docs)
            os.unlink(tmp_path)

        except Exception as e:
            warnings.append(f"❌ Failed to read **{uf.name}**: {e}")

    return all_docs, warnings


def split_documents(docs, mode, size, overlap):
    if mode == "Smart (headers/paragraphs)":
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = []
        for d in docs:
            try:
                sub_chunks = header_splitter.split_text(d.page_content)
                for sc in sub_chunks:
                    sc.metadata.update(d.metadata)
                chunks.extend(char_splitter.split_documents(sub_chunks))
            except Exception:
                # fall back to plain paragraph splitting if no headers found
                chunks.extend(char_splitter.split_documents([d]))
        return chunks
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        return splitter.split_documents(docs)


def get_reranker():
    """Lazily load a small cross-encoder reranker, only if the user enabled it."""
    if st.session_state.reranker is not None:
        return st.session_state.reranker
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        st.session_state.reranker = model
        return model
    except ImportError:
        st.warning(
            "Reranking needs `sentence-transformers`. Install it with:\n\n"
            "`pip install sentence-transformers`\n\nFalling back to plain similarity search for now."
        )
        return None
    except Exception as e:
        st.warning(f"Couldn't load reranker ({e}). Falling back to plain similarity search.")
        return None


def rerank(question, docs, top_k):
    model = get_reranker()
    if model is None:
        return docs[:top_k]
    pairs = [(question, d.page_content) for d in docs]
    scores = model.predict(pairs)
    ranked = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return ranked[:top_k]


PROMPT_TEMPLATE = """You are a precise research assistant. Answer the question using ONLY
the context below. If the answer isn't in the context, say you don't know —
do not make anything up.

{history_block}
Context:
{{context}}

Question: {{question}}

Answer clearly and concisely:"""


# --------------------------------------------------------------------------
# File upload
# --------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload one or more documents",
    type=["pdf", "txt", "md", "docx"],
    accept_multiple_files=True,
)

build_col, clear_col = st.columns([1, 1])
build_clicked = build_col.button("🔨 Build knowledge base", type="primary", use_container_width=True)
clear_clicked = clear_col.button("🗑️ Clear", use_container_width=True)

if clear_clicked:
    st.session_state.vectorstore = None
    st.session_state.chat_history = []
    st.session_state.num_chunks = 0
    st.rerun()

# --------------------------------------------------------------------------
# Save / Load persisted sessions
# --------------------------------------------------------------------------
if save_clicked:
    if st.session_state.vectorstore is None:
        st.sidebar.warning("Nothing to save yet — build a knowledge base first.")
    else:
        try:
            target = session_dir(session_name)
            if os.path.exists(target):
                shutil.rmtree(target)
            embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
            Chroma.from_documents(
                st.session_state.vectorstore.get()["documents"] and [],  # placeholder, replaced below
                embedding=embeddings,
            ) if False else None
            # Persist by re-creating a Chroma store with persist_directory from current docs
            data = st.session_state.vectorstore.get(include=["documents", "metadatas"])
            from langchain_core.documents import Document
            docs = [
                Document(page_content=t, metadata=m)
                for t, m in zip(data["documents"], data["metadatas"])
            ]
            persisted = Chroma.from_documents(docs, embedding=embeddings, persist_directory=target)
            persisted.persist()
            with open(os.path.join(target, "meta.json"), "w") as f:
                json.dump({"num_chunks": len(docs)}, f)
            st.sidebar.success(f"Saved session '{session_name}' ({len(docs)} chunks).")
        except Exception as e:
            st.sidebar.error(f"Save failed: {e}")

if load_clicked:
    try:
        target = session_dir(session_name)
        if not os.path.exists(target):
            st.sidebar.warning(f"No saved session named '{session_name}' found.")
        else:
            embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
            vectorstore = Chroma(persist_directory=target, embedding_function=embeddings)
            st.session_state.vectorstore = vectorstore
            meta_path = os.path.join(target, "meta.json")
            st.session_state.num_chunks = json.load(open(meta_path))["num_chunks"] if os.path.exists(meta_path) else "?"
            st.session_state.chat_history = []
            st.sidebar.success(f"Loaded session '{session_name}'.")
    except Exception as e:
        st.sidebar.error(f"Load failed: {e}")

# --------------------------------------------------------------------------
# Ingest -> chunk -> embed -> store
# --------------------------------------------------------------------------
if build_clicked:
    if not uploaded_files:
        st.warning("Upload at least one document first.")
    else:
        try:
            with st.spinner("Reading documents..."):
                all_docs, warnings = load_documents(uploaded_files)

            for w in warnings:
                st.warning(w)

            if not all_docs:
                st.error("No readable content found in the uploaded file(s). Nothing to index.")
            else:
                with st.spinner("Splitting into chunks..."):
                    chunks = split_documents(all_docs, chunking_mode, chunk_size, chunk_overlap)

                with st.spinner("Embedding chunks and building the vector index (this calls the Gemini API)..."):
                    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
                    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)

                st.session_state.vectorstore = vectorstore
                st.session_state.num_chunks = len(chunks)
                st.session_state.chat_history = []
                st.success(f"Indexed {len(uploaded_files)} file(s) into {len(chunks)} chunks. Ask away below.")

        except Exception as e:
            st.error(
                "Something went wrong while building the knowledge base. "
                "This is usually an API key issue, a rate limit, or a Gemini outage."
            )
            with st.expander("Technical details"):
                st.code(f"{e}\n\n{traceback.format_exc()}")

# --------------------------------------------------------------------------
# Q&A
# --------------------------------------------------------------------------
if st.session_state.vectorstore is not None:
    st.markdown("---")
    st.subheader("Ask a question")
    if isinstance(st.session_state.num_chunks, int) and st.session_state.num_chunks:
        st.caption(f"Knowledge base ready — {st.session_state.num_chunks} chunks indexed.")

    question = st.text_input("Your question", placeholder="e.g. What was the total revenue mentioned in the report?")
    ask_clicked = st.button("Ask")

    if ask_clicked and question:
        try:
            with st.spinner("Retrieving relevant chunks and generating an answer..."):
                fetch_k = top_k * 2 if use_reranker else top_k
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": fetch_k})
                candidates = retriever.invoke(question)

                sources = rerank(question, candidates, top_k) if use_reranker else candidates[:top_k]

                history_block = ""
                if use_history_context and st.session_state.chat_history:
                    recent = st.session_state.chat_history[-3:]
                    history_lines = "\n".join(f"Q: {q}\nA: {a}" for q, a, _ in recent)
                    history_block = f"Previous conversation (for follow-up context):\n{history_lines}\n\n"

                prompt = PromptTemplate(
                    template=PROMPT_TEMPLATE.format(history_block=history_block),
                    input_variables=["context", "question"],
                )

                llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
                context_text = "\n\n---\n\n".join(d.page_content for d in sources)
                chain_input = prompt.format(context=context_text, question=question)
                raw_content = llm.invoke(chain_input).content
                # Newer Gemini responses can come back as a list of content
                # blocks (e.g. [{"type": "text", "text": "...", "extras": {...}}])
                # instead of a plain string. Extract just the text in either case.
                if isinstance(raw_content, str):
                    answer = raw_content
                elif isinstance(raw_content, list):
                    parts = []
                    for block in raw_content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            parts.append(block)
                    answer = "\n".join(p for p in parts if p) or str(raw_content)
                else:
                    answer = str(raw_content)

                st.session_state.chat_history.append((question, answer, sources))

        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                st.error("Gemini rate limit hit. Wait a bit and try again, or reduce k / chunk count.")
            elif "404" in msg:
                st.error(
                    f"Model not found ({CHAT_MODEL}). Google may have renamed/retired it — "
                    "check https://aistudio.google.com/models for the current model name and "
                    "update CHAT_MODEL near the top of app.py."
                )
            else:
                st.error("Something went wrong answering this question.")
            with st.expander("Technical details"):
                st.code(f"{e}\n\n{traceback.format_exc()}")

    # Render chat history, most recent first
    for q, a, sources in reversed(st.session_state.chat_history):
        st.markdown(f"**Q: {q}**")
        st.write(a)
        with st.expander(f"📎 Sources ({len(sources)} chunk(s))"):
            for i, doc in enumerate(sources, 1):
                fname = doc.metadata.get("source_file", "unknown")
                page = doc.metadata.get("page")
                label = f"page {page}" if page is not None else "section"
                st.markdown(f"**[{i}] {fname} — {label}**")
                st.caption(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))
        st.markdown("---")
else:
    st.info("Upload documents and click **Build knowledge base** to get started — or load a saved session from the sidebar.")
