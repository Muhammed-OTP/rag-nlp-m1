"""Streamlit interface for the NLP/ML RAG assistant (Phase 4).

Run with: streamlit run app.py
"""

import json
import os
import re
import time
from io import BytesIO
from typing import TypedDict

import chromadb
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from matplotlib.backends.backend_pdf import PdfPages
from sentence_transformers import SentenceTransformer

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, GROQ_MODEL, TOP_K
from src.prepare_data import OUTPUT_FILE as CHUNKS_FILE
from src.prepare_data import RAW_DIR, SKIP_FILES, chunk_text, clean_text
from src.rag_pipeline import CHROMA_DIR, COLLECTION_NAME, build_prompt, load_collection

load_dotenv()

LLM_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]
if GROQ_MODEL not in LLM_MODELS:
    LLM_MODELS.insert(0, GROQ_MODEL)


class Chunk(TypedDict):
    id: str
    text: str
    source: str
    similarity: float


# --------------------------------------------------------------------------- #
# Page setup & styling
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="RAG Assistant - NLP/ML Concepts",
    page_icon=":material/local_library:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
        background-color: var(--secondary-background-color);
    }
    .badge-row { display: flex; gap: 0.5rem; margin: 0.4rem 0 0.6rem 0; flex-wrap: wrap; }
    .badge {
        display: inline-block;
        padding: 0.18rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: var(--primary-color);
        color: #ffffff;
        opacity: 0.88;
    }
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border-radius: 14px;
        padding: 0.75rem 1rem;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px !important;
        transition: transform 0.06s ease-in-out, box-shadow 0.06s ease-in-out;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.12);
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }
    div[data-testid="stExpander"] {
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Backend hooks
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer used for embeddings."""
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Loading vector index...")
def get_vectorstore():
    """Load and cache the persistent ChromaDB collection.

    Raises if the collection has not been built yet (see Corpus Management tab).
    """
    return load_collection()


@st.cache_resource
def get_groq_client():
    """Load and cache the Groq client, or None if no API key is configured."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def vectorstore_available() -> bool:
    try:
        get_vectorstore()
        return True
    except Exception:
        return False


def retrieve(question: str, k: int, threshold: float) -> list[Chunk]:
    """Embed the question, query Chroma for the top-k chunks, and filter by similarity."""
    model = get_embedding_model()
    collection = get_vectorstore()
    query_embedding = model.encode([question])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=k)

    ids = results["ids"][0]
    documents = (results["documents"] or [[]])[0]
    metadatas = (results["metadatas"] or [[]])[0]
    distances = (results["distances"] or [[]])[0]

    chunks: list[Chunk] = []
    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        similarity = 1.0 / (1.0 + distance)
        if similarity >= threshold:
            chunks.append({"id": chunk_id, "text": text, "source": str(meta["source"]), "similarity": similarity})
    return chunks


def generate_response(question: str, chunks: list[Chunk], model_name: str) -> str:
    """Build the prompt from retrieved chunks and call the Groq LLM."""
    client = get_groq_client()
    if client is None:
        return "Error: GROQ_API_KEY is not set. Add it to your .env file and restart the app."

    if chunks:
        prompt = build_prompt(question, chunks)
    else:
        prompt = build_prompt(question, [{"text": "(no context passed the similarity threshold)"}])

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        return f"Error calling the LLM ({model_name}): {exc}"


def lexical_overlap(answer: str, context: str) -> float:
    """Fraction of the answer's words that also appear in the retrieved context.

    A simple, dependency-free proxy for grounding/faithfulness.
    """
    answer_tokens = set(re.findall(r"\w+", answer.lower()))
    context_tokens = set(re.findall(r"\w+", context.lower()))
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


@st.cache_data
def load_chunks_df() -> pd.DataFrame:
    """Read data/processed/chunks.jsonl into a DataFrame."""
    if not os.path.exists(CHUNKS_FILE):
        return pd.DataFrame(columns=["id", "source", "text"])
    records = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def corpus_stats(chunks_df: pd.DataFrame) -> pd.DataFrame:
    """Per-source chunk counts and average chunk size."""
    if chunks_df.empty:
        return pd.DataFrame(columns=["source", "num_chunks", "avg_chars"])
    stats = (
        chunks_df.groupby("source")["text"]
        .agg(num_chunks="count", avg_chars=lambda s: s.str.len().mean())
        .reset_index()
        .sort_values("source")
    )
    stats["avg_chars"] = stats["avg_chars"].round(1)
    return stats


def reprocess_corpus() -> int:
    """Re-clean and re-chunk every .txt file in data/raw into chunks.jsonl."""
    filenames = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".txt") and f not in SKIP_FILES)
    total = 0
    with open(CHUNKS_FILE, "w", encoding="utf-8") as out:
        for filename in filenames:
            source = filename[:-4].replace("_", " ")
            with open(os.path.join(RAW_DIR, filename), encoding="utf-8") as f:
                raw_text = f.read()
            cleaned = clean_text(raw_text)
            for i, chunk in enumerate(chunk_text(cleaned, CHUNK_SIZE, CHUNK_OVERLAP)):
                record = {"id": f"{filename[:-4]}_{i}", "source": source, "text": chunk}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
    load_chunks_df.clear()
    return total


def rebuild_index(selected_sources: list[str]) -> int:
    """Re-embed the chunks of the selected sources and recreate the Chroma collection."""
    chunks_df = load_chunks_df()
    filtered = chunks_df[chunks_df["source"].isin(selected_sources)]

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    if not filtered.empty:
        model = get_embedding_model()
        embeddings = model.encode(filtered["text"].tolist(), show_progress_bar=False, batch_size=64)
        collection.add(
            ids=filtered["id"].tolist(),
            embeddings=embeddings.tolist(),
            documents=filtered["text"].tolist(),
            metadatas=[{"source": s} for s in filtered["source"]],
        )

    get_vectorstore.clear()
    return collection.count()


def save_uploaded_file(uploaded_file) -> str:
    """Save an uploaded PDF or TXT into data/raw, converting PDFs to plain text."""
    name = uploaded_file.name
    if name.lower().endswith(".pdf"):
        import fitz  # PyMuPDF, imported lazily since only needed for PDF uploads

        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "\n".join(str(page.get_text("text")) for page in pdf)
        out_name = os.path.splitext(name)[0] + ".txt"
        with open(os.path.join(RAW_DIR, out_name), "w", encoding="utf-8") as f:
            f.write(text)
        return out_name

    if name.lower().endswith(".txt"):
        with open(os.path.join(RAW_DIR, name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return name

    raise ValueError(f"Unsupported file type: {name}")


def load_test_set(uploaded_file) -> list[dict]:
    """Load an evaluation test set from a JSON or CSV file.

    Expected fields: question, expected_sources (list or ';'-separated string),
    optional expected_answer.
    """
    if uploaded_file.name.lower().endswith(".json"):
        data = json.load(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
        data = df.to_dict("records")
        for item in data:
            sources = item.get("expected_sources")
            if isinstance(sources, str):
                item["expected_sources"] = [s.strip() for s in sources.split(";") if s.strip()]
    return data


def evaluate(test_set: list[dict], k: int, threshold: float, model_name: str, progress_callback=None) -> pd.DataFrame:
    """Run the RAG pipeline on each test question and compute retrieval/answer metrics."""
    rows = []
    for i, item in enumerate(test_set):
        question = item["question"]
        expected_sources = item.get("expected_sources") or []

        start = time.perf_counter()
        chunks = retrieve(question, k, threshold)
        answer = generate_response(question, chunks, model_name)
        elapsed = time.perf_counter() - start

        retrieved_sources = [c["source"] for c in chunks]
        relevant = [s for s in retrieved_sources if s in expected_sources]
        precision = len(relevant) / len(retrieved_sources) if retrieved_sources else 0.0
        recall = (len(set(relevant)) / len(set(expected_sources))) if expected_sources else float("nan")
        context = " ".join(c["text"] for c in chunks)

        rows.append(
            {
                "question": question,
                "expected_sources": "; ".join(expected_sources),
                "retrieved_sources": "; ".join(retrieved_sources),
                "answer": answer,
                "ground_truth": item.get("expected_answer", ""),
                f"precision@{k}": round(precision, 3),
                f"recall@{k}": round(recall, 3) if recall == recall else None,
                "faithfulness": round(lexical_overlap(answer, context), 3),
                "response_time_s": round(elapsed, 2),
            }
        )
        if progress_callback:
            progress_callback((i + 1) / len(test_set))
    return pd.DataFrame(rows)


def build_pdf_report(results_df: pd.DataFrame, k: int) -> BytesIO:
    """Render a small multi-page PDF summary of an evaluation run."""
    precision_col, recall_col = f"precision@{k}", f"recall@{k}"
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 3))
        ax.axis("off")
        ax.set_title("RAG Evaluation Summary", fontsize=14, fontweight="bold")
        summary_cols = [precision_col, recall_col, "faithfulness", "response_time_s"]
        summary = results_df[summary_cols].mean(numeric_only=True).round(3)
        table_data = [[c, str(summary.get(c, "N/A"))] for c in summary_cols]
        ax.table(cellText=table_data, colLabels=["Metric", "Average"], loc="center", cellLoc="left")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 4))
        results_df[[precision_col, recall_col]].plot(kind="bar", ax=ax)
        ax.set_title("Precision / Recall per question")
        ax.set_xlabel("Question index")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 4))
        results_df["response_time_s"].plot(kind="hist", ax=ax, bins=10)
        ax.set_title("Response time distribution (s)")
        pdf.savefig(fig)
        plt.close(fig)
    buffer.seek(0)
    return buffer


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

st.session_state.setdefault("messages", [])

with st.sidebar:
    st.markdown("## :material/tune: Configuration")

    st.markdown("##### Generation model")
    llm_model = st.selectbox(
        "Groq model",
        LLM_MODELS,
        index=LLM_MODELS.index(GROQ_MODEL) if GROQ_MODEL in LLM_MODELS else 0,
        help="LLM used to generate the final answer from the retrieved context.",
        label_visibility="collapsed",
    )

    st.markdown("##### Embedding model")
    st.selectbox(
        "Embedding model",
        [EMBEDDING_MODEL],
        disabled=True,
        help="Fixed by the vector index. Change it in config.py and rebuild the index.",
        label_visibility="collapsed",
    )

    st.markdown("##### Retrieval settings")
    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=TOP_K)
    similarity_threshold = st.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Retrieved chunks scoring below this similarity are discarded.",
    )

    st.markdown("##### Vector store status")
    if vectorstore_available():
        collection = get_vectorstore()
        chunks_df = load_chunks_df()
        n_docs = chunks_df["source"].nunique() if not chunks_df.empty else 0
        st.success(f":material/database: Index loaded - {collection.count()} chunks / {n_docs} documents")
    else:
        st.error(":material/error: No index found. Build it from the Corpus Management tab.")

    if get_groq_client() is None:
        st.warning(":material/key_off: GROQ_API_KEY not set in .env")

    st.markdown("##### Add documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files and st.button(":material/upload_file: Save to corpus", use_container_width=True):
        for uploaded in uploaded_files:
            try:
                saved_name = save_uploaded_file(uploaded)
                st.toast(f"Saved {saved_name}", icon=":material/check_circle:")
            except ValueError as exc:
                st.toast(str(exc), icon=":material/error:")
        st.info("Now reprocess the corpus and rebuild the index from the Corpus Management tab.")

    st.divider()
    if st.button(":material/delete_sweep: Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.expander(":material/monitoring: System info"):
        last = st.session_state.get("last_query_stats")
        if last:
            st.write(f"Last response time: {last['time']:.2f} s")
            st.write(f"Chunks retrieved: {last['num_chunks']}")
            st.write(f"Faithfulness (lexical overlap): {last['faithfulness']:.2f}")
        else:
            st.caption("No queries yet.")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.markdown("# :material/local_library: RAG Assistant")
st.caption("Retrieval-Augmented Generation over NLP & ML concepts - Wikipedia corpus, ChromaDB, Groq LLM")

tab_chat, tab_eval, tab_corpus = st.tabs(
    [":material/chat: Chat", ":material/query_stats: Evaluation", ":material/folder_managed: Corpus Management"]
)


# --------------------------------------------------------------------------- #
# Chat tab
# --------------------------------------------------------------------------- #

with tab_chat:
    ready = vectorstore_available()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=msg["role"]):
            st.markdown(msg["content"])
            meta = msg.get("meta")
            if meta:
                st.markdown(
                    f"""<div class="badge-row">
                        <span class="badge">Time: {meta['time']:.2f}s</span>
                        <span class="badge">Chunks used: {meta['num_chunks']}</span>
                        <span class="badge">Faithfulness: {meta['faithfulness']:.2f}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            if "chunks" in msg:
                with st.expander(f"Sources ({len(msg['chunks'])})"):
                    if not msg["chunks"]:
                        st.caption("No chunks passed the similarity threshold.")
                    for chunk in msg["chunks"]:
                        st.markdown(f"**{chunk['source']}** - similarity {chunk['similarity']:.3f}")
                        st.caption(chunk["text"])

    if not ready:
        st.error("The vector index has not been built yet. Go to the Corpus Management tab and click 'Rebuild vector index'.")

    question = st.chat_input("Ask a question about NLP/ML concepts...", disabled=not ready)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Retrieving context and generating answer..."):
            start = time.perf_counter()
            chunks = retrieve(question, top_k, similarity_threshold)
            answer = generate_response(question, chunks, llm_model)
            elapsed = time.perf_counter() - start

        faithfulness = lexical_overlap(answer, " ".join(c["text"] for c in chunks))
        meta = {"time": elapsed, "num_chunks": len(chunks), "faithfulness": faithfulness}
        st.session_state.messages.append({"role": "assistant", "content": answer, "chunks": chunks, "meta": meta})
        st.session_state.last_query_stats = meta
        st.rerun()


# --------------------------------------------------------------------------- #
# Evaluation tab
# --------------------------------------------------------------------------- #

with tab_eval:
    st.markdown("### Batch evaluation")
    st.caption(
        "Upload a JSON or CSV test set with fields `question`, `expected_sources` "
        "(a list, or a `;`-separated string of source document names), and optionally `expected_answer`."
    )

    template = json.dumps(
        [
            {
                "question": "What is attention in machine learning?",
                "expected_sources": ["Attention (machine learning)"],
                "expected_answer": "A mechanism that weighs the importance of each part of the input.",
            }
        ],
        indent=2,
    )
    st.download_button(":material/download: Download template (JSON)", template, file_name="eval_template.json", mime="application/json")

    eval_file = st.file_uploader("Test set (JSON or CSV)", type=["json", "csv"], key="eval_uploader")

    if eval_file is not None:
        test_set = load_test_set(eval_file)
        st.write(f"Loaded {len(test_set)} test question(s).")
        run_disabled = not vectorstore_available()
        if run_disabled:
            st.error("The vector index has not been built yet.")
        if st.button(":material/play_arrow: Run evaluation", type="primary", disabled=run_disabled):
            progress = st.progress(0.0)
            st.session_state.eval_results = evaluate(
                test_set, top_k, similarity_threshold, llm_model, progress_callback=progress.progress
            )
            progress.empty()

    results_df = st.session_state.get("eval_results")
    if results_df is not None and not results_df.empty:
        precision_col, recall_col = f"precision@{top_k}", f"recall@{top_k}"

        st.markdown("#### Aggregate metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Avg Precision@{top_k}", f"{results_df[precision_col].mean():.2f}")
        recall_mean = results_df[recall_col].mean()
        c2.metric(f"Avg Recall@{top_k}", f"{recall_mean:.2f}" if recall_mean == recall_mean else "N/A")
        c3.metric("Avg Faithfulness", f"{results_df['faithfulness'].mean():.2f}")
        c4.metric("Avg Response Time", f"{results_df['response_time_s'].mean():.2f} s")

        st.markdown("#### Visualizations")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            bar_fig = px.bar(
                results_df.reset_index(),
                x="index",
                y=[precision_col, recall_col],
                barmode="group",
                title="Precision / Recall per question",
                labels={"index": "Question #", "value": "Score", "variable": "Metric"},
            )
            st.plotly_chart(bar_fig, use_container_width=True)
        with chart_col2:
            hist_fig = px.histogram(
                results_df, x="response_time_s", nbins=10, title="Response time distribution (s)"
            )
            st.plotly_chart(hist_fig, use_container_width=True)

        heatmap_fig = px.imshow(
            results_df[[precision_col, recall_col, "faithfulness"]].T,
            labels=dict(x="Question #", y="Metric", color="Score"),
            x=[str(i) for i in results_df.index],
            color_continuous_scale="Blues",
            title="Per-question metric heatmap",
            aspect="auto",
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)

        st.markdown("#### Detailed results")
        st.dataframe(results_df, use_container_width=True)

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                ":material/download: Export CSV",
                results_df.to_csv(index=False),
                file_name="eval_results.csv",
                mime="text/csv",
            )
        with dl_col2:
            st.download_button(
                ":material/picture_as_pdf: Export PDF report",
                build_pdf_report(results_df, top_k),
                file_name="eval_report.pdf",
                mime="application/pdf",
            )


# --------------------------------------------------------------------------- #
# Corpus management tab
# --------------------------------------------------------------------------- #

with tab_corpus:
    chunks_df = load_chunks_df()
    stats = corpus_stats(chunks_df)

    st.markdown("### Corpus overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("Documents", int(stats["source"].nunique()) if not stats.empty else 0)
    m2.metric("Total chunks", len(chunks_df))
    avg_chars = chunks_df["text"].str.len().mean() if not chunks_df.empty else 0
    m3.metric("Avg chunk size (chars)", f"{avg_chars:.0f}")

    st.markdown("### Documents")
    st.caption("Choose which sources to include the next time the vector index is rebuilt.")

    if "selected_sources" not in st.session_state:
        st.session_state.selected_sources = set(stats["source"]) if not stats.empty else set()

    with st.expander(f"Source documents ({len(stats)})", expanded=False):
        selected = []
        for _, row in stats.iterrows():
            checked = st.checkbox(
                f"{row['source']} - {row['num_chunks']} chunks, avg {row['avg_chars']:.0f} chars",
                value=row["source"] in st.session_state.selected_sources,
                key=f"src_{row['source']}",
            )
            if checked:
                selected.append(row["source"])
        st.session_state.selected_sources = set(selected)

    st.divider()
    st.markdown("### Index management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            ":material/refresh: Reprocess corpus from data/raw",
            use_container_width=True,
            help="Re-cleans and re-chunks every .txt file in data/raw/ into chunks.jsonl.",
        ):
            n_chunks = reprocess_corpus()
            st.toast(f"Re-chunked corpus into {n_chunks} chunks.", icon=":material/check_circle:")
            st.rerun()
    with col2:
        if st.button(
            ":material/sync: Rebuild vector index",
            type="primary",
            use_container_width=True,
            help="Embeds the selected sources and recreates the ChromaDB collection.",
        ):
            if not st.session_state.selected_sources:
                st.error("Select at least one document before rebuilding.")
            else:
                with st.spinner("Embedding chunks and rebuilding index..."):
                    n_vectors = rebuild_index(list(st.session_state.selected_sources))
                st.toast(f"Rebuilt index with {n_vectors} chunks.", icon=":material/check_circle:")
                st.rerun()
