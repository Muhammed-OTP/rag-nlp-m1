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
├── config.py               # shared parameters (chunk size, overlap, model, top_k)
├── src/
│   └── prepare_data.py     # cleans & chunks the corpus -> data/processed/chunks.jsonl
├── data/
│   ├── raw/                 # raw Wikipedia articles (gitignored)
│   └── processed/
│       └── chunks.jsonl     # cleaned, chunked corpus (1443 chunks)
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

Upcoming steps (vector index, retrieval + LLM pipeline, Streamlit UI,
evaluation, visualizations, report) are tracked in
[`PHASES.md`](PHASES.md).

## Project status

Phases 0 and 1 are complete. See [`PHASES.md`](PHASES.md) for the full
roadmap and per-phase results.
