# Project: RAG system over NLP/ML concepts corpus

## Context
University project (NLP module). Deadline: code + report due 14/06/2026 evening,
oral defense + live demo 15/06/2026 11:00. See `Sujet_Projet_RAG.pdf` (parent folder)
for the full grading rubric.

Domain chosen: **NLP/ML concepts** (Transformers, embeddings, retrieval, evaluation
metrics, etc.) sourced from Wikipedia articles.

## Tech stack (already decided, don't change without discussion)
- Python 3.13
- Embeddings: `sentence-transformers` with `all-MiniLM-L6-v2`
- Vector DB: ChromaDB (persistent, local folder `chroma_db/`, gitignored)
- LLM: Groq API (key in `.env` as `GROQ_API_KEY`)
- UI: Streamlit
- Eval/plots: pandas, matplotlib/seaborn
- All shared parameters (chunk size, overlap, embedding model, top_k) live in
  `config.py` — import from there, don't hardcode duplicates.

## How work is organized
See `PHASES.md` for the ordered list of phases. Each phase = one focused session:
1. Read `PHASES.md`, find the first phase not marked done.
2. Implement only that phase's scope. Don't jump ahead.
3. Run the code to prove it works (print sample output / row counts / etc).
4. Update `PHASES.md`: mark the phase done with a one-line note of what was produced.
5. Git: stage the new/changed files and commit with a `feat:`/`fix:` message
   describing that phase only. One commit per phase (small follow-up fixes get
   their own `fix:` commit).
6. Explain what you did in plain language, as if to someone who has never built
   a RAG pipeline — what the step is for, what file does it, what the output looks
   like.

## Code style rules
- No dead code, no unused imports, no speculative abstractions/classes "for later".
- Plain functions + scripts are fine. Don't build a framework.
- No try/except around things that can't realistically fail; only handle real
  edge cases (e.g. empty retrieval results, missing API key).
- Minimal comments — only for non-obvious "why" (e.g. a Wikipedia quirk we work
  around).
- Reuse `config.py` constants everywhere instead of redefining magic numbers.

## Known data quirks (handle in Phase 1)
- `data/raw/BM25.txt`, `Tokenization.txt`, `Named-entity_recognition.txt` are
  Wikipedia disambiguation pages, not real content — drop them during cleaning
  (we still have 33 usable docs, above the 30 minimum; `Tokenization_(lexical_analysis).txt`
  already covers tokenization).
