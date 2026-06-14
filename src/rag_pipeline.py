import os

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, GROQ_MODEL, TOP_K

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "nlp_corpus"

PROMPT_TEMPLATE = """Answer the question using only the context below. \
If the context does not contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


def retrieve(question, model, collection, top_k=TOP_K):
    query_embedding = model.encode([question])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=top_k)
    chunks = []
    for chunk_id, text, meta in zip(results["ids"][0], results["documents"][0], results["metadatas"][0]):
        chunks.append({"id": chunk_id, "text": text, "source": meta["source"]})
    return chunks


def build_prompt(question, chunks):
    context = "\n\n".join(chunk["text"] for chunk in chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(question, model, collection, groq_client):
    chunks = retrieve(question, model, collection)
    prompt = build_prompt(question, chunks)
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content
    return answer, chunks


def main():
    load_dotenv()
    model = SentenceTransformer(EMBEDDING_MODEL)
    collection = load_collection()
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    print("RAG over NLP/ML concepts. Type a question (or 'exit' to quit).")
    while True:
        question = input("\n> ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            break

        answer, chunks = generate_answer(question, model, collection, groq_client)
        print(f"\n{answer}")
        print("\nSources:")
        for chunk in chunks:
            print(f"  - {chunk['source']} ({chunk['id']})")


if __name__ == "__main__":
    main()
