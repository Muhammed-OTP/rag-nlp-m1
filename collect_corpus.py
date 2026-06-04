import os
import time
import requests

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ARTICLES = [
    "Transformer (machine learning)",
    "BERT (language model)",
    "GPT (language model)",
    "Word2vec",
    "GloVe (machine learning)",
    "FastText",
    "Attention (machine learning)",
    "Named-entity recognition",
    "Text classification",
    "Sentiment analysis",
    "Question answering (computing)",
    "Information retrieval",
    "TF-IDF",
    "BM25",
    "Cosine similarity",
    "Recurrent neural network",
    "Long short-term memory",
    "Seq2seq",
    "Transfer learning",
    "Fine-tuning (deep learning)",
    "Tokenization",
    "Stemming",
    "Lemmatization",
    "Stop word",
    "N-gram",
    "Bag-of-words model",
    "Perplexity (machine learning)",
    "BLEU",
    "ROUGE (metric)",
    "Hallucination (artificial intelligence)",
    "Retrieval-augmented generation",
    "LangChain",
    "Vector database",
    "Sentence embedding",
    "Text segmentation",
]

API_URL = "https://en.wikipedia.org/w/api.php"
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "NLP-Corpus-Collector/1.0 (educational project; contact: student)"
})

DELAY = 2.0
MAX_RETRIES = 5


def fetch_article(title):
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "redirects": True,
        "titles": title,
        "format": "json",
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = SESSION.get(API_URL, params=params, timeout=30)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 10 * (attempt + 1)))
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            data = resp.json()
            pages = data["query"]["pages"]
            page = next(iter(pages.values()))
            if "missing" in page:
                raise ValueError(f"Page not found: {title}")
            text = page.get("extract", "")
            if not text:
                raise ValueError("Empty extract returned")
            return text
        except (requests.RequestException, ValueError, KeyError) as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt * 2
                time.sleep(wait)
            else:
                raise


succeeded = 0
failed = 0

for i, title in enumerate(ARTICLES):
    filename = title.replace(" ", "_") + ".txt"
    filepath = os.path.join(OUTPUT_DIR, filename)
    print(f"[{i+1}/{len(ARTICLES)}] {title} ...", end=" ", flush=True)
    try:
        text = fetch_article(title)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"OK ({len(text):,} chars)")
        succeeded += 1
    except Exception as e:
        print(f"FAILED ({e})")
        failed += 1
    time.sleep(DELAY)

print(f"\nDone. {succeeded} downloaded, {failed} failed.")
print(f"Files saved to: {OUTPUT_DIR}")
