# Project Phases

Each phase = one session. Mark `[x]` and add a one-line "Result" note when done.
Do not start a phase until the previous one is checked off.

- [x] **Phase 0 — Project setup & corpus collection**
  Result: repo scaffolded (`src/`, `evaluation/`, `notebooks/`, `visualizations/`,
  `data/raw`, `data/processed`); `collect_corpus.py` downloaded 36 Wikipedia
  articles into `data/raw/`; `config.py` with shared params; `.env` with
  `GROQ_API_KEY`; deps in `requirements.txt`.

- [x] **Phase 1 — Clean & chunk the corpus**
  Drop disambiguation pages, strip Wikipedia boilerplate, split remaining docs
  into overlapping chunks (`CHUNK_SIZE`/`CHUNK_OVERLAP` from `config.py`).
  Output: `data/processed/chunks.jsonl` (one chunk per line: id, source doc,
  text). Script: `src/prepare_data.py`.
  Result: dropped 2 disambiguation pages (BM25, Tokenization), cleaned the
  remaining 34 docs (stripped "See also/References/Notes/External links"
  sections), produced 1443 chunks of 512 chars (50 overlap) in
  `data/processed/chunks.jsonl`.

- [x] **Phase 2 — Build the vector index**
  Embed every chunk with `all-MiniLM-L6-v2` and store in a persistent ChromaDB
  collection (`chroma_db/`). Script: `src/build_index.py`.
  Result: embedded all 1443 chunks with `all-MiniLM-L6-v2` and stored them in
  a persistent ChromaDB collection `nlp_corpus` at `chroma_db/` (gitignored,
  rebuilt by running `src/build_index.py`). Sanity-checked with a sample
  query ("What is attention in machine learning?") which correctly retrieved
  chunks from the "Attention (machine learning)" article.

- [ ] **Phase 3 — RAG pipeline (retriever + LLM)**
  Given a question: embed it, retrieve top-k chunks from Chroma, fill a prompt
  template with the retrieved context, call the Groq LLM, return the answer +
  source chunks. Script: `src/rag_pipeline.py` + a small CLI to ask questions
  from the terminal.

- [ ] **Phase 4 — Streamlit interface**
  Minimal UI (`app.py`): text box for the question, shows the answer and the
  retrieved source chunks/documents.

- [ ] **Phase 5 — Evaluation dataset & metrics**
  `evaluation/eval_questions.json`: ≥20 questions with the expected source
  document(s) per question. `evaluation/evaluate.py`: runs the pipeline on all
  questions, computes Precision@k, Recall@k, a faithfulness score, and response
  time; saves results to `evaluation/results.csv`.

- [ ] **Phase 6 — Visualizations**
  `visualizations/plot_results.py`: reads `evaluation/results.csv` and produces
  PNG charts (precision/recall, response time distribution, etc.) saved to
  `visualizations/`.

- [ ] **Phase 7 — Report**
  Write the PDF report (≥10 pages: intro, architecture, data, implementation,
  evaluation + graphs, discussion, conclusion, references) using the real
  results from Phases 5-6.

- [ ] **Phase 8 — Final review & demo prep**
  Update `README.md` with setup/run instructions, clean up any leftover files,
  final commit, and prepare a short demo script for the oral defense.
