"""Phase 5: run the RAG pipeline on the evaluation question set and compute metrics.

Usage: python -m evaluation.evaluate
Writes evaluation/results.csv with one row per question: precision@k, recall@k,
faithfulness (lexical overlap between answer and retrieved context), and response time.
"""

import json
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from config import GROQ_MODEL, TOP_K
from src.rag_pipeline import build_prompt, load_collection, retrieve

QUESTIONS_FILE = os.path.join(os.path.dirname(__file__), "eval_questions.json")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.csv")


def lexical_overlap(answer: str, context: str) -> float:
    """Fraction of the answer's words that also appear in the retrieved context."""
    answer_tokens = set(re.findall(r"\w+", answer.lower()))
    context_tokens = set(re.findall(r"\w+", context.lower()))
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def main():
    load_dotenv()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    collection = load_collection()
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    rows = []
    for i, item in enumerate(questions, 1):
        question = item["question"]
        expected_sources = item["expected_sources"]

        start = time.perf_counter()
        chunks = retrieve(question, model, collection, top_k=TOP_K)
        prompt = build_prompt(question, chunks)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content or ""
        elapsed = time.perf_counter() - start

        retrieved_sources = [c["source"] for c in chunks]
        relevant = [s for s in retrieved_sources if s in expected_sources]
        precision = len(relevant) / len(retrieved_sources)
        recall = len(set(relevant)) / len(set(expected_sources))
        context = " ".join(c["text"] for c in chunks)

        rows.append(
            {
                "question": question,
                "expected_sources": "; ".join(expected_sources),
                "retrieved_sources": "; ".join(retrieved_sources),
                f"precision@{TOP_K}": round(precision, 3),
                f"recall@{TOP_K}": round(recall, 3),
                "faithfulness": round(lexical_overlap(answer, context), 3),
                "response_time_s": round(elapsed, 3),
            }
        )
        print(f"[{i}/{len(questions)}] {question} -> "
              f"precision={rows[-1][f'precision@{TOP_K}']} recall={rows[-1][f'recall@{TOP_K}']} "
              f"faithfulness={rows[-1]['faithfulness']} time={rows[-1]['response_time_s']}s")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_FILE, index=False)

    print(f"\nSaved {len(df)} rows to {RESULTS_FILE}")
    print(df[[f"precision@{TOP_K}", f"recall@{TOP_K}", "faithfulness", "response_time_s"]].mean().round(3))


if __name__ == "__main__":
    main()
