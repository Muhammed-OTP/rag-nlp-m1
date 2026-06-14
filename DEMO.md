# Oral Defense Demo Script

**Project:** RAG over NLP/ML concepts  
**Author:** Mohamed Salem Ebnou Echvagha Oubeid — C34613  
**Duration:** ~10–12 minutes (+ Q&A)  
**Date:** 15 June 2026, 11:00

---

## Before the session (5 min checklist)

- [ ] `.env` contains a valid `GROQ_API_KEY`
- [ ] `chroma_db/` exists (`python -m src.build_index` if needed)
- [ ] Terminal ready in project root, venv activated
- [ ] Browser tab open on `http://localhost:8501` (after `streamlit run app.py`)
- [ ] GitHub repo open: https://github.com/Muhammed-OTP/rag-nlp-m1
- [ ] PDF report ready (compiled from `report/overleaf/`)

---

## 1. Introduction (~1 min)

> *French talking point:*
>
> « Bonjour. Mon projet consiste à concevoir un système RAG — Retrieval-Augmented Generation —
> appliqué aux concepts fondamentaux du NLP et du machine learning. Le système répond à des
> questions en s'appuyant sur un corpus de 34 articles Wikipedia, une base vectorielle ChromaDB,
> et le LLM Llama 3.3 via l'API Groq. Le code est sur GitHub et le rapport détaille
> l'architecture, l'évaluation et les résultats. »

**Show:** title slide or report cover + GitHub repo page.

---

## 2. Architecture (~2 min)

> *Explain the two-phase pipeline:*

1. **Offline indexing:** Wikipedia → clean/chunk → embed (`all-MiniLM-L6-v2`) → ChromaDB
2. **Online query:** question → embed → retrieve top-3 chunks → prompt → Groq LLM → answer

**Show:** figure from report (`report/overleaf/figures/architecture.png`) or README diagram.

**Mention key files:**
- `collect_corpus.py`, `src/prepare_data.py`, `src/build_index.py`, `src/rag_pipeline.py`
- Parameters centralized in `config.py`

---

## 3. Live demo — Streamlit Chat (~3 min)

**Run (if not already running):**
```bash
streamlit run app.py
```

### Demo question 1 (core concept)

Type in Chat:
```
What is the transformer architecture and why was it introduced?
```

**Point out while it answers:**
- Answer is grounded in retrieved context (not pure LLM memory)
- Source chunks shown below (from *Transformer (machine learning)* article)
- Response time badge (~1 s)
- Faithfulness score (lexical overlap with context)

### Demo question 2 (retrieval quality)

```
What is retrieval-augmented generation and what problem does it address?
```

**Say:**
> « Here the retriever finds chunks from the RAG article itself — this is exactly what
> the evaluation measures with Precision@3 and Recall@3. »

### Optional — show sidebar

- Change `top-k` from 3 to 5 → more context, slightly slower
- Show API key status and index stats (1443 chunks, 34 documents)

---

## 4. Code walkthrough (~2 min)

Open in IDE (pick 2–3, don't read everything):

### `src/rag_pipeline.py` — prompt template (anti-hallucination)

```python
PROMPT_TEMPLATE = """Answer the question using only the context below. \
If the context does not contain the answer, say you don't know.
...
```

### `src/rag_pipeline.py` — retrieve()

```python
def retrieve(question, model, collection, top_k=TOP_K):
    query_embedding = model.encode([question])
    results = collection.query(query_embeddings=..., n_results=top_k)
```

**Say:** same embedding model for indexing and querying → cosine similarity in ChromaDB.

---

## 5. Evaluation results (~2 min)

**Show:** charts in `visualizations/` or Evaluation tab in Streamlit.

| Metric | Mean | Meaning |
|--------|------|---------|
| Precision@3 | 0.922 | 92% of retrieved chunks from expected document |
| Recall@3 | 0.985 | Expected document found in top-3 almost always |
| Faithfulness | 0.887 | Answer words overlap with retrieved context |
| Response time | 0.81 s | End-to-end latency |

**Mention one harder case (honesty):**
> « For closely related topics like FastText vs Word2vec, precision drops to 0.33 because
> semantically similar chunks from neighbor articles are retrieved — still useful content,
> but penalized by strict document-level metrics. »

**Reproduce eval (optional, if time):**
```bash
python -m evaluation.evaluate
```

---

## 6. Conclusion (~1 min)

> *French closing:*
>
> « En résumé : corpus de 34 documents Wikipedia, pipeline RAG complet, interface Streamlit
> fonctionnelle, évaluation sur 34 questions avec métriques demandées par le sujet, et rapport
> PDF. Perspectives : re-ranking cross-encoder, LLM-as-judge pour la fidélité, corpus
> multilingue. Merci — je suis prêt pour vos questions. »

---

## Likely Q&A — short answers

| Question | Answer |
|----------|--------|
| Why ChromaDB? | Simple, persistent, no external server — good for a local academic project |
| Why MiniLM? | Fast on CPU, ~80 MB, strong sentence-level retrieval |
| Why Groq? | Free tier, very low latency for demo |
| Chunk size 512? | Balance between specificity and enough context per chunk |
| How do you limit hallucinations? | Prompt restricts answers to context; faithfulness metric tracks overlap |
| Why English corpus? | Wikipedia NLP articles are rich in English; answers match corpus language |
| What if index missing? | Run `python -m src.build_index`; Corpus Management tab can rebuild too |

---

## Fallback if API fails during demo

1. Show pre-generated `evaluation/results.csv` and charts in `visualizations/`
2. Walk through CLI with a question already tested
3. Show Evaluation tab results / exported PDF from a previous run

---

## Sample questions bank (for practice)

```
What does BERT stand for and what is it used for?
What is attention used for in machine learning?
What is Word2vec and how does it learn word representations?
What is fine-tuning in deep learning?
What does TF-IDF stand for and how is it computed?
What is named-entity recognition?
```
