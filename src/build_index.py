import json
import os

import chromadb
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CHUNKS_FILE = os.path.join(PROCESSED_DIR, "chunks.jsonl")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "nlp_corpus"


def load_chunks():
    chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{"source": chunk["source"]} for chunk in chunks],
    )

    print(f"Stored {collection.count()} embeddings in '{COLLECTION_NAME}' at {CHROMA_DIR}")

    # Sanity check: retrieve nearest chunks for a sample query
    query = "What is attention in machine learning?"
    query_embedding = model.encode([query])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=3)
    print(f"\nSample query: {query!r}")
    metadatas = results["metadatas"] or [[]]
    distances = results["distances"] or [[]]
    for chunk_id, meta, distance in zip(results["ids"][0], metadatas[0], distances[0]):
        print(f"  {chunk_id} (source: {meta['source']}, distance: {distance:.4f})")


if __name__ == "__main__":
    main()
