# RAG System over NLP/ML Concepts

A Retrieval-Augmented Generation (RAG) pipeline built for the M1 NLP module
project. The system answers questions about core NLP/ML concepts
(Transformers, embeddings, retrieval, evaluation metrics, etc.) by retrieving
relevant passages from a corpus of Wikipedia articles and feeding them to an
LLM to generate grounded answers.

**Author:** Mohamed Salem Ebnou Echvagha Oubeid — C34613  
**Repository:** https://github.com/Muhammed-OTP/rag-nlp-m1  
**Report:** LaTeX sources in [`report/overleaf/`](report/overleaf/) (compile on Overleaf)  
**Oral demo script:** [`DEMO.md`](DEMO.md)

See [`Sujet_Projet_RAG.pdf`](../Sujet_Projet_RAG.pdf) for the assignment
specification and [`PHASES.md`](PHASES.md) for the project roadmap.

## Architecture

```
Wikipedia articles --> clean & chunk --> embed (MiniLM) --> ChromaDB index
                                                                  |
                                                                  v
                User question --> embed --> retrieve top-k chunks --> Groq LLM --> answer
```

## Tech stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.13 |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (persistent, `chroma_db/`) |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| UI | Streamlit + CLI |
| Evaluation / plots | pandas, matplotlib, seaborn, plotly |

Shared parameters live in [`config.py`](config.py) (`CHUNK_SIZE`, `CHUNK_OVERLAP`,
`EMBEDDING_MODEL`, `TOP_K`, `GROQ_MODEL`).

## Repository structure

```
rag-nlp-m1/
├── app.py                    # Streamlit UI (Chat / Evaluation / Corpus)
├── collect_corpus.py         # download Wikipedia articles -> data/raw/
├── config.py                 # shared parameters
├── DEMO.md                   # oral defense demo script
├── src/
│   ├── prepare_data.py       # clean & chunk -> data/processed/chunks.jsonl
│   ├── build_index.py        # embed chunks -> chroma_db/
│   └── rag_pipeline.py       # retriever + Groq LLM + CLI
├── evaluation/
│   ├── eval_questions.json   # 34 test questions
│   ├── evaluate.py           # batch evaluation -> results.csv
│   └── results.csv           # precision@k, recall@k, faithfulness, time
├── visualizations/
│   └── plot_results.py       # PNG charts from results.csv
├── report/
│   ├── overleaf/             # LaTeX report (French, for Overleaf)
│   └── architecture_diagram.py
├── data/
│   ├── raw/                  # raw Wikipedia articles (gitignored)
│   └── processed/chunks.jsonl
├── chroma_db/                # vector index (gitignored, rebuilt locally)
├── requirements.txt
└── PHASES.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Get a free key at https://console.groq.com/

## First-time pipeline (from scratch)

Run once after cloning (requires network for corpus download on step 1):

```bash
python collect_corpus.py              # 36 Wikipedia articles -> data/raw/
python -m src.prepare_data            # clean + chunk -> chunks.jsonl (1443 chunks)
python -m src.build_index             # embed + store in chroma_db/
```

Steps 2–3 can be skipped if `data/processed/chunks.jsonl` is already in the repo;
you still need step 3 locally because `chroma_db/` is gitignored.

## Usage

### Streamlit (recommended for demo)

```bash
streamlit run app.py
```

Three tabs:

- **Chat** — ask questions, see answer + source chunks + faithfulness score
- **Evaluation** — load a test set, run metrics, export CSV/PDF
- **Corpus Management** — corpus stats, upload documents, rebuild index

Sidebar: pick Groq model, adjust `top-k` and similarity threshold.

### CLI

```bash
python -m src.rag_pipeline
```

Interactive prompt; type `exit` to quit.

## Evaluation

34-question test set covering all corpus documents:

```bash
python -m evaluation.evaluate
python -m visualizations.plot_results
```

**Mean results** (`evaluation/results.csv`):

| Metric | Value |
|--------|-------|
| Precision@3 | 0.922 |
| Recall@3 | 0.985 |
| Faithfulness | 0.887 |
| Response time | 0.811 s |

Charts are saved to `visualizations/` and copied into `report/overleaf/figures/`.

## Report

The PDF report is written in French using LaTeX:

1. Upload the folder `report/overleaf/` to [Overleaf](https://www.overleaf.com)
2. Replace `figures/university_logo.png` with your university logo
3. Recompile → download `main.pdf`

See [`report/overleaf/README.md`](report/overleaf/README.md) for details.

## Project status

All phases (0–8) are complete. See [`PHASES.md`](PHASES.md) for per-phase notes.

## Oral defense

See [`DEMO.md`](DEMO.md) for a step-by-step demo script (15 June 2026).
