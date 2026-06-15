# Technical Documentation — RAG System over NLP/ML Concepts

This document is a complete reference for the project: every file, every important
function, every package, and every question a jury could reasonably ask — each
answered twice: **first like you're explaining it to a 5-year-old (ELI5)**, then
with the **technical detail** you'd give in an oral defense.

---

## 1. The Big Picture (ELI5 first)

**ELI5:** Imagine you have a giant pile of 34 little books about AI/NLP topics
(Transformers, BERT, Word2vec, etc.). Someone asks you a question. Instead of
reading all 34 books, you have a smart librarian (the *retriever*) who instantly
finds the 3 most relevant pages. Then you hand those 3 pages to a very smart
friend (the *LLM*, Llama 3.3 via Groq) and say "answer the question using ONLY
these pages." The friend reads them and writes the answer. That's it — that's RAG
(Retrieval-Augmented Generation).

**Technical:** This is a classic two-stage RAG pipeline:
1. **Offline indexing pipeline** (run once, or whenever the corpus changes):
   `collect_corpus.py` → `src/prepare_data.py` → `src/build_index.py`
2. **Online query pipeline** (run every time a user asks a question):
   embed question → vector similarity search in ChromaDB → build prompt with
   retrieved chunks → call Groq LLM → return grounded answer + sources

```
Wikipedia articles --> clean & chunk --> embed (MiniLM) --> ChromaDB index
                                                                  |
                                                                  v
                User question --> embed --> retrieve top-k chunks --> Groq LLM --> answer
```

---

## 2. Repository Map

```
rag-nlp-m1/
├── app.py                    # Streamlit UI (Chat / Evaluation / Corpus Management)
├── collect_corpus.py         # Step 1: download Wikipedia articles -> data/raw/
├── config.py                 # Shared constants (single source of truth)
├── src/
│   ├── prepare_data.py       # Step 2: clean & chunk -> data/processed/chunks.jsonl
│   ├── build_index.py        # Step 3: embed chunks -> chroma_db/ (ChromaDB)
│   └── rag_pipeline.py        # Step 4: retriever + prompt + Groq call + CLI
├── evaluation/
│   ├── eval_questions.json   # 34 test questions w/ expected source docs
│   ├── evaluate.py           # Batch eval -> precision/recall/faithfulness/time
│   └── results.csv           # Saved results (mean: P@3=0.922, R@3=0.985)
├── visualizations/
│   └── plot_results.py       # results.csv -> 5 PNG charts
├── data/
│   ├── raw/                  # 36 raw Wikipedia .txt files (gitignored)
│   └── processed/chunks.jsonl # 1443 cleaned/chunked passages
├── chroma_db/                 # Persistent vector index (gitignored, rebuilt locally)
├── report/overleaf/           # LaTeX report (French) for Overleaf
├── requirements.txt
├── README.md
├── PHASES.md                  # Project roadmap / phase log
└── DEMO.md                    # Oral defense script
```

---

## 3. File-by-File Walkthrough

### 3.1 `config.py` — Shared parameters (the "single source of truth")

```python
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
GROQ_MODEL = "llama-3.3-70b-versatile"
```

**ELI5:** This is the project's settings page — like the options menu in a video
game. Every other file reads its numbers from here instead of making up its own,
so everything stays consistent.

**Technical:**
- `CHUNK_SIZE = 512` — each text chunk stored in the vector DB is at most 512
  characters.
- `CHUNK_OVERLAP = 50` — consecutive chunks share 50 characters, so a sentence
  that straddles a chunk boundary isn't fully lost in either chunk.
- `EMBEDDING_MODEL = "all-MiniLM-L6-v2"` — a small (≈80MB), fast sentence-
  transformer model producing 384-dimensional embeddings. Used both when
  building the index AND when embedding the user's question — this consistency
  is essential, otherwise similarity scores would be meaningless.
- `TOP_K = 3` — number of chunks retrieved per query.
- `GROQ_MODEL = "llama-3.3-70b-versatile"` — the LLM used to generate the final
  answer, called through Groq's API (fast inference, free tier).

**Likely question — "Why these specific values?"**
- Chunk size 512 chars ≈ a couple of sentences — small enough to be focused
  (good for precision) but big enough to carry context (good for recall).
- Overlap 50 chars avoids cutting a key sentence exactly at a chunk boundary.
- Top-K=3 balances enough context for the LLM vs. prompt size/latency/noise.

---

### 3.2 `collect_corpus.py` — Phase 0: Corpus collection

**ELI5:** A robot that goes to Wikipedia, downloads 36 articles about AI/NLP
topics, and saves each one as a plain `.txt` file.

**Technical — main pieces:**
- `ARTICLES`: a hardcoded list of 36 Wikipedia page titles (Transformer, BERT,
  GPT, Word2vec, TF-IDF, BLEU, ROUGE, RAG, LangChain, Vector database, etc.) —
  chosen to cover the whole "NLP/ML concepts" domain required by the assignment.
- `fetch_article(title)`:
  - Calls the **Wikipedia API** (`action=query&prop=extracts&explaintext=True`)
    to get the plain-text extract of an article.
  - Handles **HTTP 429 (rate limiting)** by reading the `Retry-After` header and
    sleeping.
  - Retries up to `MAX_RETRIES = 5` times with **exponential backoff**
    (`2**attempt * 2` seconds).
  - Raises `ValueError` if the page is missing or the extract is empty.
- Main loop: for each article, fetch it, write to `data/raw/<Title>.txt`
  (spaces → underscores), sleep `DELAY = 2.0` seconds between requests (politeness
  to Wikipedia's servers), and print a per-article success/failure log + final
  summary.

**Likely questions:**
- *"Why a 2-second delay and retries?"* — To respect Wikipedia's API usage policy
  and avoid getting rate-limited/blocked mid-run.
- *"What if a download fails?"* — It's caught, logged as `FAILED`, and the script
  continues with the next article — the run isn't all-or-nothing.
- *"This is gitignored — how do I get the data?"* — Either re-run this script
  (needs network) or use the already-committed `data/processed/chunks.jsonl`.

---

### 3.3 `src/prepare_data.py` — Phase 1: Clean & chunk

**ELI5:** Takes the raw downloaded articles, throws away the boring parts
(like "See Also" link lists), cuts the rest into bite-sized pieces (chunks), and
writes them all into one file — like cutting a long book into flashcards.

**Technical — main pieces:**
- `RAW_DIR`, `PROCESSED_DIR`, `OUTPUT_FILE` — path constants (`data/raw/`,
  `data/processed/`, `data/processed/chunks.jsonl`).
- `SKIP_FILES = {"BM25.txt", "Tokenization.txt"}` — these two downloaded pages
  are **Wikipedia disambiguation pages** (just lists of links to other pages, no
  real content), so they're excluded. (`Named-entity_recognition.txt` is also
  mentioned as a quirk in `CLAUDE.md`, but `Tokenization_(lexical_analysis).txt`
  already covers tokenization content, keeping the corpus at 34 usable docs —
  above the 30-document minimum required by the assignment.)
- `DROP_SECTIONS = {"see also", "references", "further reading", "external links", "notes"}`
  — section headers (Wikipedia `== Header ==` syntax) whose content is dropped.
- **`clean_text(text)`**:
  - Iterates line by line. When it sees a line matching `^==+\s*(.+?)\s*==+$`
    (a Wikipedia section header), it checks if the header name (lowercased) is in
    `DROP_SECTIONS`. If yes, it starts "skipping" — every following line is
    discarded until the next header.
  - Collapses 3+ consecutive newlines down to 2 (`re.sub(r"\n{3,}", "\n\n", text)`).
  - Returns the cleaned text, stripped of leading/trailing whitespace.
- **`chunk_text(text, chunk_size, overlap)`**:
  - A simple sliding-window splitter. `start = 0`; while `start < len(text)`:
    take `text[start:start+chunk_size]`, strip it, append if non-empty.
    If the chunk reached the end of the text, stop. Otherwise advance
    `start = end - overlap` (so the next chunk overlaps the previous one by
    `overlap` characters).
  - **Note:** this is a *character-based* chunker (not word/sentence-aware) —
    simple, deterministic, and good enough given the small corpus.
- **`main()`**: lists all `.txt` files in `data/raw/` except `SKIP_FILES`, sorts
  them, for each file: derive `source` name (filename without `.txt`, underscores
  → spaces, e.g. `"BERT (language model)"`), read raw text, `clean_text()`,
  `chunk_text()`, and write one JSON object per chunk to `chunks.jsonl`:
  `{"id": "<filename>_<index>", "source": "<source name>", "text": "<chunk>"}`.

**Output:** `data/processed/chunks.jsonl` — 1443 chunks from 34 documents.

**Likely questions:**
- *"Why character-based chunking instead of sentence/token-based?"* — Simple,
  fast, deterministic, no extra dependency (no need for a sentence tokenizer),
  and works fine for Wikipedia prose at this scale. A trade-off mentioned as a
  possible improvement (sentence-aware chunking) in the report's discussion.
- *"What happens to a chunk that's the last partial piece of a doc?"* — It's still
  included as long as it's non-empty after stripping.
- *"How is the `source` field used?"* — It's the human-readable document name
  stored as ChromaDB metadata, used later for precision/recall (does the
  retrieved chunk's source match the expected document?).

---

### 3.4 `src/build_index.py` — Phase 2: Build the vector index

**ELI5:** Takes every flashcard (chunk) from Phase 1, turns it into a list of
384 numbers that captures its *meaning* (an "embedding"), and stores all those
number-lists in a searchable database (ChromaDB) — like turning every flashcard
into a GPS coordinate on a "meaning map", so similar ideas end up near each other.

**Technical — main pieces:**
- `CHUNKS_FILE`, `CHROMA_DIR = "../chroma_db"`, `COLLECTION_NAME = "nlp_corpus"`.
- **`load_chunks()`**: reads `chunks.jsonl` line by line, `json.loads` each line
  into a dict, returns a list of `{id, source, text}`.
- **`main()`**:
  1. Loads all chunks, prints the count.
  2. Loads `SentenceTransformer(EMBEDDING_MODEL)` (i.e. `all-MiniLM-L6-v2`).
  3. `model.encode(texts, show_progress_bar=True, batch_size=64)` — encodes all
     1443 chunk texts into embeddings (NumPy array, shape `(1443, 384)`).
  4. Creates a `chromadb.PersistentClient(path=CHROMA_DIR)` — a ChromaDB instance
     that writes to disk at `chroma_db/`.
  5. If a collection named `nlp_corpus` already exists, **deletes it** (so
     re-running this script is idempotent / rebuilds from scratch), then
     `create_collection(COLLECTION_NAME)`.
  6. `collection.add(ids=..., embeddings=..., documents=..., metadatas=[{"source": ...}])`
     — stores, for each chunk: its id, its 384-dim embedding vector, its raw
     text, and metadata `{"source": "<doc name>"}`.
  7. **Sanity check**: runs a sample query ("What is attention in machine
     learning?"), embeds it, calls `collection.query(query_embeddings=..., n_results=3)`,
     and prints the top-3 chunk ids + sources + distances — to visually confirm
     the index makes sense before moving on.

**Likely questions:**
- *"What distance metric does ChromaDB use by default?"* — Cosine-similarity-based
  (squared L2 / cosine, ChromaDB default is `l2` but for normalized
  sentence-transformer embeddings this is monotonically related to cosine
  similarity — practically, "smaller distance = more similar").
- *"Why delete and recreate the collection instead of upserting?"* — Simplicity
  and correctness: guarantees the index always exactly matches the current
  `chunks.jsonl`, no stale/duplicate entries.
- *"Where is the index stored?"* — `chroma_db/` — a local folder with SQLite +
  binary HNSW index files (`data_level0.bin`, `header.bin`, etc.), gitignored
  because it's large and 100% reproducible from `chunks.jsonl`.

---

### 3.5 `src/rag_pipeline.py` — Phase 3: Retriever + LLM + CLI

**ELI5:** This is the "brain" of the project. Given a question, it (1) turns the
question into the same kind of "meaning numbers" used for the chunks, (2) asks
ChromaDB "which 3 chunks are closest to this question on the meaning map?",
(3) writes a note to the AI saying "here's the question and these 3 snippets,
answer using ONLY them", and (4) sends that note to Llama 3.3 via Groq and prints
the reply.

**Technical — main pieces:**
- `CHROMA_DIR`, `COLLECTION_NAME = "nlp_corpus"`.
- **`PROMPT_TEMPLATE`** (critical for anti-hallucination):
  ```
  Answer the question using only the context below. If the context does not
  contain the answer, say you don't know.

  Context:
  {context}

  Question: {question}

  Answer:
  ```
- **`load_collection()`**: `chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION_NAME)`
  — opens the existing persisted index (raises if it doesn't exist yet, which
  the Streamlit app catches via `vectorstore_available()`).
- **`retrieve(question, model, collection, top_k=TOP_K)`**:
  - `model.encode([question])` → query embedding.
  - `collection.query(query_embeddings=..., n_results=top_k)`.
  - Zips `ids`, `documents`, `metadatas` from the first (only) query result into
    a list of `{"id", "text", "source"}` dicts.
- **`build_prompt(question, chunks)`**: joins `chunk["text"]` for all chunks with
  `"\n\n"` as `context`, and `.format()`s it into `PROMPT_TEMPLATE`.
- **`generate_answer(question, model, collection, groq_client)`**: calls
  `retrieve()`, `build_prompt()`, then
  `groq_client.chat.completions.create(model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}])`,
  returns `(answer_text, chunks)`.
- **`main()`**: CLI loop — `load_dotenv()`, load the embedding model, load the
  collection, create the Groq client from `GROQ_API_KEY`, then loop reading
  questions from stdin (`exit`/`quit` to stop), printing the answer and the
  list of source chunks (source doc name + chunk id).

**Likely questions:**
- *"Why give the LLM the chunks instead of asking it directly?"* — Grounding:
  the LLM's training data may be outdated/wrong/hallucinated; giving it
  retrieved, verifiable text makes the answer traceable to a source and reduces
  hallucination. This is the core idea of RAG.
- *"What if no chunks are relevant?"* — The prompt explicitly instructs the model
  to say "I don't know" rather than make something up — though this is only a
  soft instruction, not a hard guarantee (an LLM can still ignore it).
- *"Why same embedding model for indexing and querying?"* — Embeddings from
  different models live in different vector spaces; comparing them would be
  meaningless. Consistency is mandatory.
- *"Single-turn or multi-turn?"* — Single-turn: each question is independent,
  no conversation memory is sent to the LLM (the CLI loop is just a UX
  convenience; Streamlit's chat history is also not fed back into the prompt).

---

### 3.6 `app.py` — Phase 4: Streamlit UI

**ELI5:** This is the website you click around on. It has three tabs: a chat
window to ask questions, an "Evaluation" tab to test how good the system is on
many questions at once, and a "Corpus Management" tab to add new documents and
rebuild the index — all without touching the terminal.

**Technical — structure and key functions:**

#### Setup
- Imports config constants, `prepare_data` helpers (`clean_text`, `chunk_text`,
  `RAW_DIR`, `SKIP_FILES`, `OUTPUT_FILE`), and `rag_pipeline` helpers
  (`CHROMA_DIR`, `COLLECTION_NAME`, `build_prompt`, `load_collection`).
- `LLM_MODELS` — a dropdown list of selectable Groq models
  (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `gemma2-9b-it`), with the
  configured `GROQ_MODEL` inserted first if not already present.
- `Chunk` — a `TypedDict` describing a retrieved chunk shape: `{id, text, source, similarity}`.
- `st.set_page_config(...)` + custom CSS injected via `st.markdown(..., unsafe_allow_html=True)`
  for styling chat bubbles, badges, metrics, buttons.

#### Cached backend resources (`@st.cache_resource` — loaded once, reused across reruns)
- **`get_embedding_model()`** → `SentenceTransformer(EMBEDDING_MODEL)`.
- **`get_vectorstore()`** → `load_collection()` (raises if no index built yet).
- **`get_groq_client()`** → `Groq(api_key=...)` or `None` if `GROQ_API_KEY` is unset.

#### Helper functions
- **`vectorstore_available()`**: tries `get_vectorstore()`, returns `True`/`False`
  — used to disable chat/eval UI when no index exists yet.
- **`retrieve(question, k, threshold)`**: like `rag_pipeline.retrieve`, but also
  computes **`similarity = 1 / (1 + distance)`** for each result and **filters
  out chunks below `threshold`** — this is the app's extra knob beyond the CLI.
- **`generate_response(question, chunks, model_name)`**: builds the prompt (or a
  placeholder "(no context passed the similarity threshold)" if `chunks` is
  empty), calls Groq, returns the text or a friendly error string (e.g. missing
  API key, or any exception from the Groq call wrapped in a message).
- **`lexical_overlap(answer, context)`**: the **faithfulness metric** —
  `len(answer_tokens & context_tokens) / len(answer_tokens)`, where tokens are
  `\w+` regex matches, lowercased. i.e., "what fraction of the words in the
  answer also appear somewhere in the retrieved context?" A simple,
  dependency-free proxy for "is the model grounding its answer in the
  retrieved text, or making stuff up?"
- **`load_chunks_df()`** (`@st.cache_data`): reads `chunks.jsonl` into a pandas
  DataFrame (`id, source, text`), or an empty DataFrame if the file doesn't
  exist.
- **`corpus_stats(chunks_df)`**: groups by `source`, computes `num_chunks` (count)
  and `avg_chars` (mean text length, rounded to 1 decimal) — feeds the Corpus
  Management overview table.
- **`reprocess_corpus()`**: re-runs the cleaning/chunking logic from
  `prepare_data` (same `clean_text`/`chunk_text`) over every `.txt` in
  `data/raw/` (excluding `SKIP_FILES`), rewrites `chunks.jsonl`, clears the
  `load_chunks_df` cache, returns the new total chunk count. Lets the user
  re-chunk from the UI after uploading new documents.
- **`rebuild_index(selected_sources)`**: filters `chunks_df` to only the
  user-selected `source` documents, deletes & recreates the `nlp_corpus`
  ChromaDB collection, encodes the filtered texts with the cached embedding
  model, `collection.add(...)`, clears the `get_vectorstore` cache, returns the
  new `collection.count()`. This lets the user pick a *subset* of documents to
  index (e.g. exclude one if it's noisy).
- **`save_uploaded_file(uploaded_file)`**: handles file uploads.
  - `.pdf` → opens with **PyMuPDF (`fitz`)** (imported lazily, only when needed),
    extracts text page by page with `page.get_text("text")`, joins with `\n`,
    writes as `<name>.txt` into `data/raw/`.
  - `.txt` → writes the raw bytes directly into `data/raw/`.
  - anything else → raises `ValueError("Unsupported file type: ...")`.
- **`load_test_set(uploaded_file)`**: loads an evaluation test set from `.json`
  (a list of dicts) or `.csv` (via pandas, converting a `;`-separated
  `expected_sources` string column into a list).
- **`evaluate(test_set, k, threshold, model_name, progress_callback=None)`**: the
  in-app equivalent of `evaluation/evaluate.py` — for each test item: `retrieve()`,
  `generate_response()`, time it, compute `precision@k`, `recall@k`, `faithfulness`,
  `response_time_s`; returns a DataFrame; calls `progress_callback(fraction_done)`
  after each item (drives the Streamlit progress bar).
- **`build_pdf_report(results_df, k)`**: uses **matplotlib's `PdfPages`** to build
  a 3-page in-memory PDF (`BytesIO`): (1) a table of average metrics, (2) a bar
  chart of precision/recall per question, (3) a histogram of response times.
  Returned for download via `st.download_button`.

#### UI layout
- **Sidebar**: model selector (`llm_model`), embedding model display (disabled —
  fixed by the built index), `top_k` slider (1–10), `similarity_threshold` slider
  (0.0–1.0), vector store status (chunk/doc counts or error), Groq API key
  warning, file uploader + "Save to corpus" button, "Clear chat history" button,
  and a "System info" expander showing the last query's time/chunks/faithfulness.
- **Chat tab**: renders message history (`st.session_state.messages`), each
  assistant message shows badges (time, chunks used, faithfulness) and an
  expander listing source chunks with similarity scores. `st.chat_input(...)`
  is disabled if the vector store isn't built yet. On a new question: retrieve →
  generate → compute faithfulness → append to history → `st.rerun()`.
- **Evaluation tab**: download a JSON template, upload a test set, "Run
  evaluation" button (disabled if no index), then aggregate metrics (`st.metric`),
  Plotly bar chart (precision/recall per question), Plotly histogram (response
  times), Plotly heatmap (precision/recall/faithfulness per question), detailed
  results table, and CSV/PDF export buttons.
- **Corpus Management tab**: overview metrics (doc count, total chunks, avg chunk
  size), an expander listing every source document with a checkbox (selection
  used by `rebuild_index`), "Reprocess corpus from data/raw" button, and
  "Rebuild vector index" button (disabled if no source selected).

**Likely questions:**
- *"What's the difference between the CLI (`rag_pipeline.main`) and the Streamlit
  app's retrieve/generate?"* — Same underlying logic (same embedding model,
  same ChromaDB collection, same prompt template), but the app adds a
  **similarity threshold filter**, model selection, and UI/metrics on top.
- *"Why `@st.cache_resource` vs `@st.cache_data`?"* — `cache_resource` is for
  non-serializable objects that should be created once and reused (ML models,
  DB clients, API clients) — Streamlit doesn't try to hash/copy them.
  `cache_data` is for serializable data (DataFrames) — Streamlit hashes inputs
  and can safely return copies.
- *"What happens if `GROQ_API_KEY` is missing?"* — `get_groq_client()` returns
  `None`; `generate_response()` then returns an explanatory error string instead
  of crashing; the sidebar also shows a warning.
- *"How does the similarity threshold work exactly?"* — `similarity = 1/(1+distance)`
  converts ChromaDB's distance (0 = identical, increasing means less similar)
  into a 0–1 "similarity" score (1 = identical, decreasing means less similar);
  chunks below the user-chosen threshold are dropped before being sent to the LLM.

---

### 3.7 `evaluation/eval_questions.json` & `evaluation/evaluate.py` — Phase 5

**ELI5:** A teacher gives the system 34 quiz questions, each with an "answer key"
saying which Wikipedia article should contain the answer. The script runs every
question through the pipeline, checks whether the right article was actually
retrieved, and how good/fast the answer was. The results (scores) are saved to a
spreadsheet (`results.csv`).

**Technical:**
- `eval_questions.json`: list of `{"question": ..., "expected_sources": [...]}`
  — 34 questions, one (or more) per corpus document, covering the whole corpus.
- `evaluate.py main()`:
  1. `load_dotenv()`, load `SentenceTransformer("all-MiniLM-L6-v2")`,
     `load_collection()`, create `Groq` client.
  2. For each question:
     - `retrieve(question, model, collection, top_k=TOP_K)`.
     - `build_prompt(...)`, call Groq, get `answer`.
     - `retrieved_sources = [c["source"] for c in chunks]`.
     - `relevant = [s for s in retrieved_sources if s in expected_sources]`.
     - **`precision = len(relevant) / len(retrieved_sources)`** — of the chunks
       we retrieved, what fraction came from an expected document?
     - **`recall = len(set(relevant)) / len(set(expected_sources))`** — of the
       expected documents, how many were found among the retrieved chunks
       (counted as distinct documents, not chunk counts)?
     - `faithfulness = lexical_overlap(answer, context)` (same function as in
       `app.py`).
     - `response_time_s` via `time.perf_counter()`.
  3. Collects all rows into a `pandas.DataFrame`, writes `evaluation/results.csv`,
     prints per-question results and column means.

**Results obtained (`results.csv` summary):**

| Metric | Mean |
|---|---|
| Precision@3 | 0.922 |
| Recall@3 | 0.985 |
| Faithfulness | 0.887 |
| Response time | 0.811 s |

**Likely questions:**
- *"What does Precision@3 = 0.922 actually mean?"* — On average, 92.2% of the
  top-3 retrieved chunks came from a document the question was "about" — i.e.
  retrieval is rarely fetching irrelevant documents.
- *"What does Recall@3 = 0.985 mean?"* — 98.5% of the time, the expected source
  document was present among the top-3 retrieved chunks — i.e. the right
  information is almost always reachable.
- *"Why would precision be < 1 even if recall is high?"* — If `top_k=3` but only
  1 expected source exists, and 2 of the 3 retrieved chunks come from a
  *related-but-different* article (e.g. FastText query also pulling Word2vec
  chunks because they're semantically close), recall can still be 1.0 (the right
  doc was found) while precision drops (2/3 chunks were "off-target" by this
  strict document-matching definition). The DEMO.md explicitly calls out the
  FastText vs Word2vec case (precision drops to 0.33) as an honest example.
- *"Is faithfulness a perfect metric?"* — No — it's a *lexical overlap* proxy
  (word-level set intersection), not semantic. A paraphrased correct answer that
  reuses few exact words from the context would score low even if accurate. A
  possible improvement: LLM-as-judge.
- *"Why is `evaluate.py` somewhat duplicated in `app.py`?"* — `app.py`'s
  `evaluate()` is the same logic but adds the similarity-threshold filter and
  feeds a Streamlit progress bar — kept separate because the Streamlit app
  needs to be self-contained and interactive (re-runs on every UI event).

---

### 3.8 `visualizations/plot_results.py` — Phase 6

**ELI5:** Turns the spreadsheet of scores into pictures (bar charts, histograms)
so it's easy to *see* how well the system did.

**Technical — 5 functions, each producing one PNG (saved to `visualizations/`,
dpi=150, using `matplotlib` + `seaborn`):**
- `plot_precision_recall(df)` → `precision_recall_per_question.png` — grouped
  bar chart of `precision@3`/`recall@3` per question.
- `plot_response_time(df)` → `response_time_distribution.png` — histogram + KDE
  of `response_time_s`.
- `plot_faithfulness(df)` → `faithfulness_distribution.png` — histogram + KDE of
  `faithfulness`, x-axis clamped to [0, 1.05].
- `plot_summary(df)` → `average_metrics.png` — bar chart of the column means of
  `[precision@3, recall@3, faithfulness]`, with value labels on top of each bar.
- `plot_time_vs_faithfulness(df)` → `time_vs_faithfulness.png` — scatter plot of
  response time vs faithfulness, colored by `precision@3`.

`main()` reads `evaluation/results.csv`, calls all five, prints a confirmation.
These PNGs are copied into `report/overleaf/figures/` for the LaTeX report.

---

### 3.9 `report/` — LaTeX report (French)

- `report/overleaf/main.tex` + `sections/01_introduction.tex` … `10_references.tex`
  + `annexe_code.tex`, `annexe_exemples.tex` — a full academic report (≥10 pages):
  introduction, état d'avancement, architecture, données, implémentation,
  évaluation, résultats, discussion, conclusion, références, code appendices.
- `report/overleaf/figures/` — contains the architecture diagram and all 5
  evaluation PNGs, plus the university logo.
- `report/architecture_diagram.py` — generates `architecture.png` (the pipeline
  diagram used in the report and README).
- `report/generate_report.py` — a **legacy** ReportLab-based PDF generator, kept
  for history but superseded by the LaTeX/Overleaf report.
- `Rapport_RAG_NLP.pdf` — the compiled report PDF.

---

### 3.10 `CLAUDE.md`, `README.md`, `PHASES.md`, `DEMO.md`

- **`CLAUDE.md`** — internal working agreement (deadlines, tech stack lock-in,
  code style rules, known data quirks). Not part of the deliverable, but useful
  context for "why was it built this way, session by session."
- **`README.md`** — the canonical setup/usage doc: architecture diagram, tech
  stack table, repo structure, setup instructions, first-run pipeline commands,
  usage (Streamlit/CLI), evaluation results table, report instructions.
- **`PHASES.md`** — phase-by-phase project log (Phase 0–8, all marked done) with
  a "Result" note per phase — a great source for "what was done when and why."
- **`DEMO.md`** — the oral defense script: checklist, talking points (French),
  demo questions, code walkthrough picks, evaluation explanation, Q&A table,
  fallback plan if the API fails, and a bank of sample questions.

---

## 4. Packages Used (`requirements.txt`) — what & why

| Package | ELI5 | Technical role |
|---|---|---|
| **langchain** / **langchain-community** | A toolbox of LEGO bricks for building AI apps that use documents + LLMs. | Listed as a dependency but the final implementation uses **direct** ChromaDB + SentenceTransformers + Groq calls rather than LangChain's abstractions (simpler, fewer moving parts, easier to explain/debug for the project's scope). |
| **chromadb** | The "filing cabinet" that stores the meaning-numbers (embeddings) and can quickly find the closest ones. | Persistent local vector database (`chroma_db/`). Stores embeddings + documents + metadata in collection `nlp_corpus`; `query()` does approximate nearest-neighbor search (HNSW index under the hood). |
| **sentence-transformers** | Turns sentences into lists of numbers that capture meaning. | Loads `all-MiniLM-L6-v2` — 6-layer MiniLM distilled model, 384-dim output, ~80MB, runs fast on CPU. Used for both indexing (`build_index.py`) and querying (`rag_pipeline.py`, `app.py`). |
| **streamlit** | A way to turn a Python script into a clickable website without writing HTML/JS. | Powers `app.py` — widgets (`st.chat_input`, `st.slider`, `st.tabs`, `st.file_uploader`, `st.dataframe`, etc.), caching decorators (`st.cache_resource`, `st.cache_data`), session state. |
| **groq** | The "phone line" to a very fast AI chat model (Llama 3.3) hosted by Groq. | `Groq(api_key=...)` client; `client.chat.completions.create(model=..., messages=[...])` — OpenAI-compatible chat completion API, used to generate the final answer from the prompt. |
| **pymupdf** (imported as `fitz`) | Lets the program "read" PDF files and pull out the text. | Used in `app.py`'s `save_uploaded_file()` to convert uploaded PDFs into `.txt` files for the corpus (`page.get_text("text")` per page). |
| **python-dotenv** | Reads secret values (like API keys) from a `.env` file instead of hardcoding them. | `load_dotenv()` loads `GROQ_API_KEY` from `.env` into environment variables — keeps secrets out of source code / git. |
| **matplotlib** | Draws charts/graphs. | Used in `visualizations/plot_results.py` and `app.py`'s `build_pdf_report()` (via `matplotlib.backends.backend_pdf.PdfPages`). |
| **seaborn** | A "skin" on top of matplotlib that makes nicer-looking statistical charts with less code. | `sns.histplot`, `sns.barplot`, `sns.scatterplot`, `sns.set_theme(style="whitegrid")` in `plot_results.py`. |
| **pandas** | Spreadsheet-like tables in Python. | `DataFrame` for `chunks.jsonl` (corpus stats), `results.csv` (evaluation), used throughout `app.py` and `evaluate.py`. |
| **tqdm** | A progress bar so you know a long task is still running. | Used implicitly via `show_progress_bar=True` in `SentenceTransformer.encode()`. |
| **wikipedia** | A simple wrapper to fetch Wikipedia content (listed, though `collect_corpus.py` actually calls the raw Wikipedia HTTP API via `requests` directly for finer control over retries/rate-limits). | Listed as a dependency; the actual corpus collection uses `requests` against `https://en.wikipedia.org/w/api.php` for full control over retries and rate-limit handling. |
| **plotly** | Interactive charts (hover, zoom) for web apps. | `plotly.express` (`px.bar`, `px.histogram`, `px.imshow`) — used in the Streamlit Evaluation tab for interactive charts (vs. the static PNGs from `plot_results.py`). |

**Why two charting libraries (matplotlib/seaborn AND plotly)?**
- matplotlib/seaborn → static, high-quality PNGs for the **LaTeX report** (which
  can't embed interactive widgets).
- plotly → interactive charts inside the **Streamlit app** (hover tooltips,
  zoom) for live demos.

---

## 5. End-to-End Data Flow (concrete numbers)

1. `collect_corpus.py` → 36 Wikipedia articles → `data/raw/*.txt`
2. `src/prepare_data.py` → drops 2 disambiguation pages (BM25, Tokenization) →
   cleans 34 docs (drops See Also/References/etc. sections) → chunks into
   512-char pieces with 50-char overlap → **1443 chunks** → `data/processed/chunks.jsonl`
3. `src/build_index.py` → embeds all 1443 chunks with `all-MiniLM-L6-v2` (384-dim
   vectors) → stores in ChromaDB collection `nlp_corpus` → `chroma_db/`
4. User asks a question (CLI or Streamlit):
   - Question → `all-MiniLM-L6-v2` → 384-dim query vector
   - ChromaDB `query()` → top-`TOP_K` (default 3) nearest chunks (id, text, source, distance)
   - (Streamlit only) convert distance → similarity `1/(1+d)`, filter by threshold
   - `build_prompt()` → fills `PROMPT_TEMPLATE` with joined chunk texts + question
   - Groq `llama-3.3-70b-versatile` → generates answer
   - (Streamlit) `lexical_overlap(answer, context)` → faithfulness score
5. Evaluation (`evaluation/evaluate.py`): repeats step 4 for 34 questions, computes
   precision@3/recall@3/faithfulness/time → `evaluation/results.csv`
6. `visualizations/plot_results.py` → 5 PNG charts → copied to `report/overleaf/figures/`
7. `report/overleaf/` → compiled on Overleaf → `Rapport_RAG_NLP.pdf`

---

## 6. General Q&A Bank (beyond DEMO.md)

**Q: What is RAG and why use it instead of just asking the LLM directly?**
ELI5: Instead of trusting a friend's memory (which can be wrong or made up), you
hand them the exact pages of a book first, then ask the question — they answer
from the book, not from guessing.
Technical: RAG = Retrieval-Augmented Generation. It grounds LLM outputs in
retrieved documents, reducing hallucination, enabling up-to-date/domain-specific
knowledge without fine-tuning, and providing traceable sources.

**Q: Why ChromaDB and not FAISS/Pinecone/Weaviate?**
ChromaDB is embeddable, persistent to a local folder, requires no server/account,
and has a tiny, simple Python API (`PersistentClient`, `create_collection`,
`.add`, `.query`) — ideal for a local academic project with no infra budget.

**Q: Why `all-MiniLM-L6-v2`?**
Small (~80MB), CPU-friendly, fast to encode 1443 chunks, and a strong baseline
for sentence-level semantic similarity — a standard choice for lightweight RAG.

**Q: Why Groq and Llama 3.3 70B specifically?**
Groq's LPU inference is extremely fast (good for live demo latency, ~0.8s mean
response time) and has a generous free tier; Llama 3.3 70B is a strong
general-purpose open model good at instruction-following ("answer only from
context").

**Q: What happens if I change `CHUNK_SIZE` in config.py?**
You must re-run `src/prepare_data.py` (re-chunk) AND `src/build_index.py`
(re-embed + rebuild ChromaDB), since the chunk boundaries and counts change. The
Streamlit Corpus Management tab can also do this (Reprocess corpus → Rebuild
index).

**Q: How is the vector index versioned / shared with teammates?**
It isn't — `chroma_db/` is gitignored because it's large and 100% derivable from
`chunks.jsonl` (which IS committed). Anyone cloning the repo runs
`python -m src.build_index` once to regenerate it locally.

**Q: What's the difference between `src/rag_pipeline.py`'s `retrieve()` and
`app.py`'s `retrieve()`?**
Same embedding + ChromaDB query, but `app.py` additionally converts distance to a
0-1 similarity score and filters out chunks below a user-adjustable threshold.

**Q: How would you scale this to a bigger corpus?**
Swap `all-MiniLM-L6-v2` for a larger embedding model if needed, move ChromaDB to
a hosted/server mode or another vector DB (Pinecone/Qdrant) for concurrent
access, add metadata filtering, consider hybrid search (BM25 + embeddings) and a
re-ranker (cross-encoder) for precision.

**Q: What are the system's main limitations / future work?**
- Character-based (not semantic) chunking.
- Faithfulness metric is lexical overlap, not true semantic faithfulness
  (improvement: LLM-as-judge).
- No re-ranking step after retrieval (improvement: cross-encoder re-ranker).
- English-only corpus (improvement: multilingual embeddings/corpus).
- No conversation memory (each question is independent).
- Precision can drop for semantically-overlapping topics (e.g. FastText vs
  Word2vec) due to strict document-level metric definition.

**Q: Where do the API keys live and how are they protected?**
`.env` file at the project root, holding `GROQ_API_KEY=...`, loaded via
`python-dotenv`'s `load_dotenv()`. `.env` is gitignored (check `.gitignore`) so
the key never reaches version control.

**Q: How do you add a new document to the corpus live during the demo?**
Streamlit sidebar → "Add documents" → upload a PDF or TXT (PyMuPDF extracts text
from PDFs) → saved to `data/raw/` → go to Corpus Management tab → "Reprocess
corpus from data/raw" (re-chunks everything into `chunks.jsonl`) → select the new
document's checkbox → "Rebuild vector index" (re-embeds selected sources and
recreates the ChromaDB collection).

**Q: What does "1.0 / (1.0 + distance)" mean intuitively?**
ChromaDB returns a *distance* (0 = identical vectors, larger = more different).
This formula maps distance 0 → similarity 1.0, and as distance grows, similarity
asymptotically approaches 0 — a convenient bounded (0,1] "similarity-like" score
for a UI threshold slider, without needing to know the true max possible distance.
