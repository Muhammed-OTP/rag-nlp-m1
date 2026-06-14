# RAG System over NLP/ML Concepts

A Retrieval-Augmented Generation (RAG) pipeline built for the M1 NLP module
project. The system answers questions about core NLP/ML concepts
(Transformers, embeddings, retrieval, evaluation metrics, etc.) by retrieving
relevant passages from a corpus of Wikipedia articles and feeding them to an
LLM to generate grounded answers.

See [`Sujet_Projet_RAG.pdf`](../Sujet_Projet_RAG.pdf) for the assignment
specification and grading rubric, and [`PHASES.md`](PHASES.md) for the
project roadmap and progress.

## Architecture

```
Wikipedia articles --> clean & chunk --> embed (MiniLM) --> ChromaDB index
                                                                  |
                                                                  v
                User question --> embed --> retrieve top-k chunks --> Groq LLM --> answer
```

## Tech stack

- **Language**: Python 3.13
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector store**: ChromaDB (persistent local store, `chroma_db/`)
- **LLM**: Groq API
- **UI**: Streamlit
- **Evaluation / plots**: pandas, matplotlib, seaborn

Shared parameters (chunk size, overlap, embedding model, `top_k`) are
centralized in [`config.py`](config.py).

## Repository structure

```
rag-nlp-m1/
├── collect_corpus.py      # downloads the Wikipedia corpus into data/raw/
├── config.py               # shared parameters (chunk size, overlap, model, top_k, Groq model)
├── src/
│   ├── prepare_data.py     # cleans & chunks the corpus -> data/processed/chunks.jsonl
│   ├── build_index.py      # embeds chunks and stores them in ChromaDB (chroma_db/)
│   └── rag_pipeline.py     # retriever + Groq LLM pipeline, interactive CLI
├── data/
│   ├── raw/                 # raw Wikipedia articles (gitignored)
│   └── processed/
│       └── chunks.jsonl     # cleaned, chunked corpus (1443 chunks)
├── chroma_db/               # persistent ChromaDB vector index (gitignored, rebuilt by build_index.py)
├── evaluation/              # evaluation dataset & metrics scripts (Phase 5)
├── visualizations/          # result charts (Phase 6)
├── notebooks/               # exploratory notebooks
├── requirements.txt
└── PHASES.md                # project roadmap & progress log
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Groq API key:

```
GROQ_API_KEY=your_key_here
```

## Pipeline so far

1. **Corpus collection** (`collect_corpus.py`): downloads 36 Wikipedia
   articles covering NLP/ML concepts into `data/raw/`.
2. **Cleaning & chunking** (`src/prepare_data.py`): drops disambiguation
   pages, strips Wikipedia boilerplate (See also/References/Notes/External
   links), and splits the remaining 34 documents into overlapping chunks
   (`CHUNK_SIZE=512`, `CHUNK_OVERLAP=50`), producing 1443 chunks in
   `data/processed/chunks.jsonl`.
3. **Vector index** (`src/build_index.py`): embeds every chunk with
   `all-MiniLM-L6-v2` and stores the vectors, text, and source article in a
   persistent ChromaDB collection (`nlp_corpus`) under `chroma_db/`.
4. **RAG pipeline** (`src/rag_pipeline.py`): embeds the user's question,
   retrieves the top `TOP_K` chunks from `chroma_db/`, fills a prompt
   template with that context, and calls the Groq LLM (`GROQ_MODEL`) to
   produce a grounded answer along with its source chunks.

## Asking a question

```bash
python -m src.rag_pipeline
```

This starts an interactive prompt: type a question, get an answer plus the
source chunks it was grounded in, and type `exit` to quit. Requires
`chroma_db/` to exist (run `python -m src.build_index` once to build it) and
a valid `GROQ_API_KEY` in `.env`.

Upcoming steps (Streamlit UI, evaluation, visualizations, report) are tracked
in [`PHASES.md`](PHASES.md).

## Project status

Phases 0-3 are complete. See [`PHASES.md`](PHASES.md) for the full roadmap
and per-phase results.
